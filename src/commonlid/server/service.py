"""Framework-agnostic core of the scoring server.

:class:`ScoringService` holds the gold labels, hands out label-free inputs,
scores submitted predictions with the same metric code as ``commonlid run``
and enforces the submission budget. The HTTP layer in
:mod:`commonlid.server.app` is a thin wrapper around it.

Nothing returned by this class ever contains gold labels: inputs carry only
``(id, text)``, and scores are restricted to the metrics named in the config.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import secrets
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from commonlid import __version__
from commonlid.core.lid_model import LIDModel
from commonlid.evaluation.results import Result, text_hash, write_predictions, write_summary
from commonlid.metrics.core import UND_TOKEN, compute_per_language_metrics
from commonlid.server.config import PER_LANGUAGE_PREFIX, DatasetConfig, ServerConfig

if TYPE_CHECKING:
    from commonlid.core.lid_dataset import LIDDataset

logger = logging.getLogger(__name__)

STATE_FILENAME = "state.json"
SUBMISSIONS_SUBDIR = "submissions"
_STATE_VERSION = 2
#: Per-language fields that give away exact gold counts in one submission.
_COUNT_FIELDS = ("gt_count", "correct", "predictions")
#: How many offending ids an error message lists before truncating.
_MAX_IDS_IN_ERROR = 10


class ScoringError(Exception):
    """Base class for errors reported back to the client."""

    status_code = 400


class UnknownDatasetError(ScoringError):
    status_code = 404


class InvalidSubmissionError(ScoringError):
    status_code = 422


class SubmissionLimitError(ScoringError):
    status_code = 429


class StateMismatchError(RuntimeError):
    """The served items no longer match the ids handed out before a restart."""


@dataclass(slots=True)
class _ServedDataset:
    """A dataset in served order: position ``i`` is the item with id ``i``.

    ``order[i]`` is that item's position in the dataset (the ``idx`` of
    ``commonlid run`` predictions).
    """

    config: DatasetConfig
    revision: str | None
    texts: list[str]
    golds: list[str | None]
    order: list[int]

    @property
    def fingerprint(self) -> str:
        """Hash of the served texts in served order.

        It changes whenever an item id would point to a different text: a
        new shuffle seed, another ``limit`` or changed dataset contents.
        """
        digest = hashlib.sha256()
        for text in self.texts:
            digest.update(text_hash(text).encode("ascii"))
        return digest.hexdigest()[:16]


def _round(value: Any, digits: int | None) -> Any:
    if digits is None:
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


def select_metrics(
    summary: Mapping[str, Any], metrics: Iterable[str], round_digits: int | None
) -> dict[str, Any]:
    """Project a ``summary.json`` dict onto the configured metric names.

    The result keeps the nesting of ``summary.json``: ``macro.f1_gold_only``
    becomes ``{"macro": {"f1_gold_only": ...}}`` and ``per_language.f1``
    becomes ``{"per_language": {"eng": {"f1": ...}, ...}}``.
    """
    out: dict[str, Any] = {}
    per_language_fields: list[str] = []
    for name in metrics:
        head, _, tail = name.partition(".")
        if not tail:
            out[head] = summary[head]
        elif head == "per_language":
            per_language_fields.append(tail)
        else:
            out.setdefault(head, {})[tail] = summary[head][tail]
    if per_language_fields:
        out["per_language"] = {
            lang: {f: row[f] for f in per_language_fields}
            for lang, row in summary["per_language"].items()
        }
    rounded: dict[str, Any] = _round(out, round_digits)
    return rounded


class ScoringService:
    """Serve inputs, score submissions and count them.

    ``datasets`` maps dataset ids to already-constructed :class:`LIDDataset`
    instances; when omitted they are looked up in the registry. Every dataset
    is loaded eagerly so a misconfigured server fails at start-up rather than
    on the agent's first request.
    """

    def __init__(
        self,
        config: ServerConfig,
        datasets: Mapping[str, LIDDataset] | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._submissions: list[dict[str, Any]] = []
        state = self._load_state()
        self._datasets: dict[str, _ServedDataset] = {}
        for ds_config in config.datasets:
            dataset = datasets[ds_config.id] if datasets is not None else None
            self._datasets[ds_config.id] = self._load_dataset(ds_config, dataset, self._seed)
        self._check_fingerprints(state.get("datasets", {}))
        self._warn_per_language()
        self._save_state()

    # ------------------------------------------------------------------ state

    @property
    def _state_path(self) -> Path | None:
        if self.config.state_dir is None:
            return None
        return self.config.state_dir / STATE_FILENAME

    def _load_state(self) -> dict[str, Any]:
        """Restore past submissions, pick the shuffle seed, return the raw state."""
        path = self._state_path
        state: dict[str, Any] = {}
        if path is not None and path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            self._submissions = list(state.get("submissions", []))
            logger.info("Restored %d previous submission(s) from %s", len(self._submissions), path)
        elif path is None:
            logger.warning(
                "No state_dir configured: the submission counter lives in memory and "
                "resets when the server restarts."
            )

        if self.config.shuffle_seed is not None:
            seed = self.config.shuffle_seed
        elif "shuffle_seed" in state:
            seed = int(state["shuffle_seed"])
        else:
            seed = secrets.randbits(63)
        self._seed = seed
        return state

    def _save_state(self) -> None:
        path = self._state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": _STATE_VERSION,
            "shuffle_seed": self._seed,
            "datasets": {
                ds_id: {
                    "n_items": len(served.texts),
                    "revision": served.revision,
                    "fingerprint": served.fingerprint,
                }
                for ds_id, served in self._datasets.items()
            },
            "submissions": self._submissions,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _check_fingerprints(self, previous: Mapping[str, Any]) -> None:
        """Refuse to start if item ids handed out before now point elsewhere.

        Only matters once a dataset has accepted submissions: an agent may
        still hold the old inputs, and scoring them against the new order
        would burn its budget on meaningless scores.
        """
        for ds_id, served in self._datasets.items():
            before = previous.get(ds_id, {}).get("fingerprint")
            if before is None or before == served.fingerprint:
                continue
            if self._used(ds_id) == 0:
                logger.warning(
                    "Items of %s changed since the last start (shuffle seed, limit or "
                    "dataset revision); no submissions yet, so the new order is used.",
                    ds_id,
                )
                continue
            path = self._state_path
            msg = (
                f"the items served for {ds_id!r} changed since the last start (shuffle "
                f"seed, limit or dataset revision), so ids from earlier inputs would "
                f"point to other texts. {self._used(ds_id)} submission(s) were already "
                f"accepted. Restore the previous settings, or delete {path} to start over "
                "(this resets the submission counter)."
            )
            raise StateMismatchError(msg)

    def _warn_per_language(self) -> None:
        for served in self._datasets.values():
            fields = [
                m.removeprefix(PER_LANGUAGE_PREFIX)
                for m in served.config.metrics
                if m.startswith(PER_LANGUAGE_PREFIX)
            ]
            if not fields:
                continue
            counts = [f for f in fields if f in _COUNT_FIELDS]
            logger.warning(
                "%s returns per-language metrics %s: they reveal which languages the gold "
                "set contains%s.",
                served.config.id,
                fields,
                f", and {counts} reveal how many samples each has" if counts else "",
            )

    # --------------------------------------------------------------- datasets

    def _load_dataset(
        self, ds_config: DatasetConfig, dataset: LIDDataset | None, seed: int
    ) -> _ServedDataset:
        if dataset is None:
            from commonlid.core.registry import get_dataset

            dataset = get_dataset(ds_config.id)
        logger.info("Loading dataset %s (limit=%s) ...", ds_config.id, ds_config.limit)
        hf = dataset.load(limit=ds_config.limit)
        texts = list(hf[dataset.text_column])
        golds = list(hf[dataset.target_column])
        order = list(range(len(texts)))
        if self.config.shuffle:
            random.Random(f"{seed}:{ds_config.id}").shuffle(order)
        logger.info("Serving %s: %d items", ds_config.id, len(texts))
        return _ServedDataset(
            config=ds_config,
            revision=dataset.hf_revision,
            texts=[texts[i] for i in order],
            golds=[golds[i] for i in order],
            order=order,
        )

    def _get(self, dataset_id: str) -> _ServedDataset:
        try:
            return self._datasets[dataset_id]
        except KeyError:
            msg = f"unknown dataset {dataset_id!r}; available: {sorted(self._datasets)}"
            raise UnknownDatasetError(msg) from None

    # ----------------------------------------------------------------- budget

    def _used(self, dataset_id: str | None = None) -> int:
        if dataset_id is None:
            return len(self._submissions)
        return sum(1 for s in self._submissions if s["dataset_id"] == dataset_id)

    @staticmethod
    def _remaining(limit: int | None, used: int) -> int | None:
        return None if limit is None else max(limit - used, 0)

    def _budget(self, dataset_id: str) -> dict[str, int | None]:
        ds_limit = self._datasets[dataset_id].config.max_submissions
        return {
            "submissions_used": self._used(),
            "submissions_remaining": self._remaining(self.config.max_submissions, self._used()),
            "dataset_submissions_used": self._used(dataset_id),
            "dataset_submissions_remaining": self._remaining(ds_limit, self._used(dataset_id)),
        }

    def _check_budget(self, served: _ServedDataset) -> None:
        if self._remaining(self.config.max_submissions, self._used()) == 0:
            msg = f"submission limit reached ({self.config.max_submissions} in total)"
            raise SubmissionLimitError(msg)
        ds_limit = served.config.max_submissions
        if self._remaining(ds_limit, self._used(served.config.id)) == 0:
            msg = f"submission limit for dataset {served.config.id!r} reached ({ds_limit})"
            raise SubmissionLimitError(msg)

    # ------------------------------------------------------------- public API

    def status(self) -> dict[str, Any]:
        """Describe the datasets, the returned metrics and the remaining budget."""
        with self._lock:
            return {
                "max_submissions": self.config.max_submissions,
                "submissions_used": self._used(),
                "submissions_remaining": self._remaining(self.config.max_submissions, self._used()),
                "datasets": [self._describe(ds) for ds in self._datasets.values()],
            }

    def _describe(self, served: _ServedDataset) -> dict[str, Any]:
        used = self._used(served.config.id)
        return {
            "id": served.config.id,
            "n_items": len(served.texts),
            "metrics": list(served.config.metrics),
            "max_submissions": served.config.max_submissions,
            "submissions_used": used,
            "submissions_remaining": self._remaining(served.config.max_submissions, used),
        }

    def inputs(
        self, dataset_id: str, *, offset: int = 0, limit: int | None = None
    ) -> dict[str, Any]:
        """Return the label-free items ``[{id, text}]`` of a dataset (paginated)."""
        served = self._get(dataset_id)
        if offset < 0 or (limit is not None and limit < 0):
            msg = "offset and limit must be non-negative"
            raise ScoringError(msg)
        end = len(served.texts) if limit is None else min(offset + limit, len(served.texts))
        return {
            "dataset_id": dataset_id,
            "total": len(served.texts),
            "offset": offset,
            "items": [{"id": i, "text": served.texts[i]} for i in range(offset, end)],
        }

    def submit(
        self, dataset_id: str, predictions: Iterable[tuple[int, str | None]]
    ) -> dict[str, Any]:
        """Score one complete set of predictions and count it against the budget.

        ``predictions`` must hold every item id of the dataset exactly once.
        Rejected submissions (over budget or malformed) are not counted.
        Predicted codes go through the same ISO 639-3 conformation as model
        outputs, so ``"en"`` scores as ``"eng"``; ``None`` scores as ``und``.
        """
        served = self._get(dataset_id)
        with self._lock:
            self._check_budget(served)
            by_id = self._validate(served, predictions)
            raw_preds = [by_id[i] for i in range(len(served.texts))]
            preds = [LIDModel._conform(p) for p in raw_preds]

            per_language = compute_per_language_metrics(
                served.golds,
                preds,
                sample_count_threshold=served.config.sample_count_threshold,
            )
            submission_id = self._used() + 1
            timestamp = datetime.now(timezone.utc).isoformat()
            result = Result(
                model_id=f"submission-{submission_id:04d}",
                dataset_id=dataset_id,
                dataset_revision=served.revision,
                per_language=per_language,
                n_samples=len(preds),
                n_samples_with_gold=sum(1 for g in served.golds if g is not None),
                limit=served.config.limit,
                timestamp=timestamp,
                commonlid_version=__version__,
            )
            summary = result.summary()
            scores = select_metrics(summary, served.config.metrics, self.config.round_digits)
            self._archive(served, raw_preds, preds, result)
            self._submissions.append({
                "submission_id": submission_id,
                "dataset_id": dataset_id,
                "timestamp": timestamp,
                "scores": scores,
                # Operator-side only, never returned to the client.
                "macro_f1_gold_only": summary["macro"]["f1_gold_only"],
            })
            self._save_state()
            logger.info(
                "Submission %d on %s: macro_f1_gold_only=%.4f",
                submission_id,
                dataset_id,
                summary["macro"]["f1_gold_only"],
            )
            return {
                "submission_id": submission_id,
                "dataset_id": dataset_id,
                "timestamp": timestamp,
                "scores": scores,
                **self._budget(dataset_id),
            }

    def history(self) -> list[dict[str, Any]]:
        """Every accepted submission with the scores that were returned for it."""
        with self._lock:
            return [
                {k: s[k] for k in ("submission_id", "dataset_id", "timestamp", "scores")}
                for s in self._submissions
            ]

    # -------------------------------------------------------------- internals

    @staticmethod
    def _validate(
        served: _ServedDataset, predictions: Iterable[tuple[int, str | None]]
    ) -> dict[int, str | None]:
        n = len(served.texts)
        by_id: dict[int, str | None] = {}
        duplicates: list[int] = []
        unknown: list[int] = []
        for item_id, pred in predictions:
            if not 0 <= item_id < n:
                unknown.append(item_id)
            elif item_id in by_id:
                duplicates.append(item_id)
            else:
                by_id[item_id] = pred
        missing = [i for i in range(n) if i not in by_id] if len(by_id) < n else []

        problems = []
        for label, ids in (("unknown", unknown), ("duplicate", duplicates), ("missing", missing)):
            if ids:
                shown = ", ".join(str(i) for i in ids[:_MAX_IDS_IN_ERROR])
                more = f", ... ({len(ids)} total)" if len(ids) > _MAX_IDS_IN_ERROR else ""
                problems.append(f"{label} ids: {shown}{more}")
        if problems:
            msg = (
                f"submission for {served.config.id!r} must contain each id in 0..{n - 1} "
                f"exactly once; " + "; ".join(problems)
            )
            raise InvalidSubmissionError(msg)
        return by_id

    def _archive(
        self,
        served: _ServedDataset,
        raw_preds: list[str | None],
        preds: list[str | None],
        result: Result,
    ) -> None:
        """Keep the full summary and per-item predictions for the operator.

        The layout and rows match ``commonlid run --output-dir``
        (``<dataset>/<model>/predictions.jsonl``, rows in dataset order keyed
        by ``idx``), so ``commonlid breakdown --local-dir <state_dir>/submissions``
        reads them. ``id`` is the item id the agent saw, ``pred_raw`` the code
        it submitted.
        """
        if self.config.state_dir is None:
            return
        run_dir = self.config.state_dir / SUBMISSIONS_SUBDIR / served.config.id / result.model_id
        served_ids = sorted(range(len(served.order)), key=served.order.__getitem__)
        rows = (
            {
                "idx": served.order[i],
                "id": i,
                "text_hash": text_hash(served.texts[i]),
                "gold": served.golds[i],
                "pred": preds[i],
                "pred_raw": raw_preds[i],
                "score": None,
                "correct": bool(
                    served.golds[i] is not None and served.golds[i] == (preds[i] or UND_TOKEN)
                ),
            }
            for i in served_ids
        )
        write_predictions(rows, run_dir / "predictions.jsonl")
        write_summary(result, run_dir / "summary.json")
