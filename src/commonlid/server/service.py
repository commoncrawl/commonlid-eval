"""Framework-agnostic core of the scoring server.

:class:`ScoringService` holds the gold labels, hands out inputs, scores
submitted predictions with the same metric code as ``commonlid run`` and
enforces the submission budget. The HTTP layer in :mod:`commonlid.server.app`
is a thin wrapper around it.

Gold labels only leave this class for datasets configured with
``expose_labels = true`` (development sets). For every other dataset, inputs
carry only ``(id, text)`` and scores are restricted to the configured metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import secrets
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from commonlid import __version__
from commonlid.analysis.breakdown import labels_match
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
#: How many distinct invalid / converted codes a response lists.
_MAX_CODES_IN_RESPONSE = 20
#: Example ids listed per invalid code.
_MAX_EXAMPLE_IDS = 5
#: A prediction must look like a bare language code before it is looked up,
#: so names ("English") and tags ("en-US", "eng_Latn") are rejected.
_CODE_PATTERN = re.compile(r"[a-z]{2,3}")


class ScoringError(Exception):
    """Base class for errors reported back to the client.

    ``extra`` holds structured details merged into the JSON error body.
    """

    status_code = 400

    def __init__(self, message: str, extra: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.extra = dict(extra or {})


class UnknownDatasetError(ScoringError):
    status_code = 404


class InvalidSubmissionError(ScoringError):
    status_code = 422


class SubmissionLimitError(ScoringError):
    status_code = 429


class SubmissionClosedError(ScoringError):
    status_code = 409


class StartupError(RuntimeError):
    """The server must not start with this configuration and state."""


class StateMismatchError(StartupError):
    """The served items no longer match the ids handed out before a restart."""


class LabelOverlapError(StartupError):
    """A dataset with exposed labels contains texts of an evaluation dataset."""


def check_prediction(raw: str | None) -> tuple[str | None, bool]:
    """Validate one submitted prediction.

    Returns ``(code, converted)``: the ISO 639-3 code to score and whether it
    differs from what was submitted (an ISO 639-1 or 639-2/B code, upper
    case, surrounding spaces or a deprecated code). ``None`` stays ``None``
    and scores as ``und``.

    Raises :class:`ValueError` for anything that is not a language code:
    names, locale or script tags, unknown or retired codes.
    """
    if raw is None:
        return None, False
    cleaned = raw.strip().lower()
    if not _CODE_PATTERN.fullmatch(cleaned):
        msg = f"not a language code: {raw!r}"
        raise ValueError(msg)
    code = LIDModel._conform(cleaned)
    if code is None:
        msg = f"unknown or retired language code: {raw!r}"
        raise ValueError(msg)
    return code, code != raw


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


_MATCHING_TEXT = {
    "exact": (
        "Codes must match the gold label exactly. A macrolanguage code (for example "
        '"zho" or "ara") does not match an individual-language label (for example "cmn" '
        'or "arb"), and the other way round.'
    ),
    "macro": (
        "An individual language and its macrolanguage count as the same (for example "
        '"cmn" and "zho", or "arb" and "ara"). Otherwise codes must match exactly.'
    ),
    "sibling": (
        "An individual language and its macrolanguage count as the same (for example "
        '"cmn" and "zho"), and so do two individual languages of the same macrolanguage '
        '(for example "arz" and "arb"). Otherwise codes must match exactly.'
    ),
}


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
        self._check_label_overlap()
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

    def _check_label_overlap(self) -> None:
        """Refuse to start if a dev dataset would reveal labels of an evaluation dataset.

        ``commonlid_nano`` is a subset of ``commonlid``, for example: exposing
        its labels would hand out part of the test set.
        """
        exposed = {
            ds_id: set(served.texts)
            for ds_id, served in self._datasets.items()
            if served.config.expose_labels
        }
        if not exposed:
            return
        problems = []
        for ds_id, served in self._datasets.items():
            if served.config.expose_labels:
                continue
            texts = set(served.texts)
            for dev_id, dev_texts in exposed.items():
                shared = len(texts & dev_texts)
                if shared:
                    problems.append(
                        f"{shared} of the {len(texts)} texts of {ds_id!r} are in {dev_id!r}"
                    )
        if problems:
            msg = (
                "datasets with expose_labels = true would reveal gold labels of evaluation "
                "datasets: " + "; ".join(problems) + ". Use dev datasets that share no texts "
                "with the evaluation datasets."
            )
            raise LabelOverlapError(msg)

    def _warn_per_language(self) -> None:
        for served in self._datasets.values():
            if served.config.expose_labels:
                continue
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
        logger.info(
            "Serving %s: %d items%s",
            ds_config.id,
            len(texts),
            " (labels exposed)" if ds_config.expose_labels else "",
        )
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

    def _counts_globally(self, dataset_id: str) -> bool:
        """Whether submissions on a dataset use the global budget.

        Scores on a dataset whose labels are exposed reveal nothing new, so
        they only count against that dataset's own cap. Submissions on a
        dataset no longer configured keep counting.
        """
        served = self._datasets.get(dataset_id)
        return served is None or not served.config.expose_labels

    def _used(self, dataset_id: str | None = None) -> int:
        if dataset_id is None:
            return sum(1 for s in self._submissions if self._counts_globally(s["dataset_id"]))
        return sum(1 for s in self._submissions if s["dataset_id"] == dataset_id)

    def _final(self, dataset_id: str) -> int | None:
        """Id of the dataset's submission marked final, if any."""
        for s in self._submissions:
            if s["dataset_id"] == dataset_id and s.get("final"):
                return int(s["submission_id"])
        return None

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

    def _check_open(self, served: _ServedDataset) -> None:
        ds_id = served.config.id
        final = self._final(ds_id)
        if final is not None:
            msg = f"dataset {ds_id!r} is closed: submission {final} was marked final"
            raise SubmissionClosedError(msg)
        if self._counts_globally(ds_id) and (
            self._remaining(self.config.max_submissions, self._used()) == 0
        ):
            msg = f"submission limit reached ({self.config.max_submissions} in total)"
            raise SubmissionLimitError(msg)
        ds_limit = served.config.max_submissions
        if self._remaining(ds_limit, self._used(ds_id)) == 0:
            msg = f"submission limit for dataset {ds_id!r} reached ({ds_limit})"
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
        cfg = served.config
        used = self._used(cfg.id)
        return {
            "id": cfg.id,
            "description": cfg.description,
            "n_items": len(served.texts),
            "labels_exposed": cfg.expose_labels,
            "macro_match": cfg.macro_match,
            "metrics": list(cfg.metrics),
            "max_submissions": cfg.max_submissions,
            "submissions_used": used,
            "submissions_remaining": self._remaining(cfg.max_submissions, used),
            "uses_global_budget": self._counts_globally(cfg.id),
            "final_submission_id": self._final(cfg.id),
        }

    def instructions(self, dataset_id: str) -> str:
        """Plain-text task description for an agent: what to predict and how to submit."""
        served = self._get(dataset_id)
        with self._lock:
            return self._instructions(served)

    def _instructions(self, served: _ServedDataset) -> str:
        cfg = served.config
        n = len(served.texts)
        ds_left = self._remaining(cfg.max_submissions, self._used(cfg.id))
        global_left = self._remaining(self.config.max_submissions, self._used())
        if self._counts_globally(cfg.id):
            budget = (
                f"{'unlimited' if ds_left is None else ds_left} submission(s) left for this "
                f"dataset, {'unlimited' if global_left is None else global_left} left overall "
                "(the overall budget is shared by all evaluation datasets)."
            )
        else:
            budget = (
                f"{'unlimited' if ds_left is None else ds_left} submission(s) left for this "
                "dataset. They do not use the overall budget."
            )
        final = self._final(cfg.id)
        item_keys = '"id", "text", "label"' if cfg.expose_labels else '"id", "text"'
        lines = [
            f"Task: identify the language of each of the {n} texts in dataset {cfg.id!r}.",
        ]
        if cfg.description:
            lines.append(cfg.description)
        if cfg.expose_labels:
            lines.append(
                "This is a development dataset: every item includes its gold label as "
                '"label". Use it to check your approach before submitting to the evaluation '
                "datasets."
            )
        else:
            lines.append("Gold labels are not available for this dataset.")
        lines += [
            "",
            f"Inputs: GET /datasets/{cfg.id}/inputs returns items with {item_keys}. Page "
            "through them with ?offset=&limit=. Ids run from 0 to "
            f"{n - 1} in a shuffled order and carry no information.",
            "",
            f"Submit: POST /datasets/{cfg.id}/submissions with a JSON body like",
            '  {"predictions": [{"id": 0, "pred": "eng"}, {"id": 1, "pred": "cmn"}], '
            '"final": false}',
            f"- Include every id from 0 to {n - 1} exactly once.",
            '- "pred" is an ISO 639-3 code: three lowercase letters, for example "eng", '
            '"deu", "cmn" or "swh". Use null if you cannot tell; it counts as wrong.',
            '- ISO 639-1 codes ("en"), ISO 639-2/B codes ("ger") and upper-case codes are '
            'converted to ISO 639-3 and reported under "warnings". Anything else, such as '
            'language names ("English"), locale or script tags ("en-US", "eng_Latn") or '
            "unknown codes, is rejected with status 422: the response lists the invalid "
            "values and the submission is not counted.",
            f"- {_MATCHING_TEXT[cfg.macro_match]}",
            f"- Returned scores: {', '.join(cfg.metrics)}.",
            f"- Budget: {budget} Rejected submissions (malformed, invalid codes, over "
            "budget) are not counted.",
            '- Set "final": true on the submission you want evaluated. The dataset accepts '
            "no further submissions after it. Without a final submission, your last "
            "accepted one counts.",
        ]
        if final is not None:
            lines.append(f"- Closed: submission {final} was marked final.")
        return "\n".join(lines)

    def inputs(
        self, dataset_id: str, *, offset: int = 0, limit: int | None = None
    ) -> dict[str, Any]:
        """Return the items ``[{id, text}]`` of a dataset (paginated) and the task.

        Items carry ``label`` too when the dataset exposes its labels.
        """
        served = self._get(dataset_id)
        if offset < 0 or (limit is not None and limit < 0):
            msg = "offset and limit must be non-negative"
            raise ScoringError(msg)
        end = len(served.texts) if limit is None else min(offset + limit, len(served.texts))
        if served.config.expose_labels:
            items = [
                {"id": i, "text": served.texts[i], "label": served.golds[i]}
                for i in range(offset, end)
            ]
        else:
            items = [{"id": i, "text": served.texts[i]} for i in range(offset, end)]
        with self._lock:
            instructions = self._instructions(served)
        return {
            "dataset_id": dataset_id,
            "total": len(served.texts),
            "offset": offset,
            "labels_exposed": served.config.expose_labels,
            "instructions": instructions,
            "items": items,
        }

    def submit(
        self,
        dataset_id: str,
        predictions: Iterable[tuple[int, str | None]],
        *,
        final: bool = False,
    ) -> dict[str, Any]:
        """Score one complete set of predictions and count it against the budget.

        ``predictions`` must hold every item id of the dataset exactly once,
        each with an ISO 639-3 code or ``None`` (scored as ``und``); see
        :func:`check_prediction`. Rejected submissions (closed, over budget,
        malformed) are not counted. ``final`` marks the submission as the one
        to evaluate and closes the dataset.
        """
        served = self._get(dataset_id)
        with self._lock:
            self._check_open(served)
            raw_preds, preds, warnings = self._validate(served, predictions)
            mode = served.config.macro_match
            # A match under the mode counts as the gold label, as in
            # `commonlid breakdown --level model`.
            scored = [
                gold if gold is not None and labels_match(gold, pred or UND_TOKEN, mode) else pred
                for gold, pred in zip(served.golds, preds, strict=True)
            ]
            per_language = compute_per_language_metrics(
                served.golds,
                scored,
                sample_count_threshold=served.config.sample_count_threshold,
            )
            submission_id = len(self._submissions) + 1
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
                "final": final,
                "scores": scores,
                # Operator-side only, never returned to the client.
                "macro_f1_gold_only": summary["macro"]["f1_gold_only"],
                "macro_match": mode,
            })
            self._save_state()
            logger.info(
                "Submission %d on %s%s: macro_f1_gold_only=%.4f",
                submission_id,
                dataset_id,
                " (final)" if final else "",
                summary["macro"]["f1_gold_only"],
            )
            return {
                "submission_id": submission_id,
                "dataset_id": dataset_id,
                "timestamp": timestamp,
                "final": final,
                "scores": scores,
                "warnings": warnings,
                **self._budget(dataset_id),
            }

    def history(self) -> list[dict[str, Any]]:
        """Every accepted submission with the scores that were returned for it."""
        with self._lock:
            return [
                {
                    "submission_id": s["submission_id"],
                    "dataset_id": s["dataset_id"],
                    "timestamp": s["timestamp"],
                    "final": bool(s.get("final", False)),
                    "scores": s["scores"],
                }
                for s in self._submissions
            ]

    # -------------------------------------------------------------- internals

    @staticmethod
    def _validate(
        served: _ServedDataset, predictions: Iterable[tuple[int, str | None]]
    ) -> tuple[list[str | None], list[str | None], list[str]]:
        """Check ids and codes; return submitted codes, scored codes and warnings.

        Both lists are in served order. Every problem is reported at once, so
        a rejected submission can be fixed in one go.
        """
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

        conformed: dict[int, str | None] = {}
        invalid: dict[str, list[int]] = defaultdict(list)
        reasons: dict[str, str] = {}
        converted: Counter[tuple[str, str]] = Counter()
        for item_id, pred in by_id.items():
            try:
                code, was_converted = check_prediction(pred)
            except ValueError as exc:
                invalid[str(pred)].append(item_id)
                reasons[str(pred)] = str(exc).split(":", 1)[0]
                continue
            conformed[item_id] = code
            if was_converted and pred is not None and code is not None:
                converted[pred, code] += 1

        problems = []
        for label, ids in (("unknown", unknown), ("duplicate", duplicates), ("missing", missing)):
            if ids:
                shown = ", ".join(str(i) for i in ids[:_MAX_IDS_IN_ERROR])
                more = f", ... ({len(ids)} total)" if len(ids) > _MAX_IDS_IN_ERROR else ""
                problems.append(f"{label} ids: {shown}{more}")
        if invalid:
            n_invalid = sum(len(ids) for ids in invalid.values())
            ranked = sorted(invalid.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            shown = ", ".join(
                f"{value!r} ({len(ids)})" for value, ids in ranked[:_MAX_IDS_IN_ERROR]
            )
            more = f", ... ({len(invalid)} distinct)" if len(invalid) > _MAX_IDS_IN_ERROR else ""
            problems.append(
                f"{n_invalid} prediction(s) are not ISO 639-3 codes: {shown}{more}. Use "
                'three-letter ISO 639-3 codes such as "eng", or null'
            )
        if problems:
            msg = f"submission for {served.config.id!r} rejected and not counted: " + "; ".join(
                problems
            )
            extra: dict[str, Any] = {}
            if invalid:
                extra["invalid_predictions"] = [
                    {
                        "pred": value,
                        "reason": reasons[value],
                        "count": len(ids),
                        "example_ids": sorted(ids)[:_MAX_EXAMPLE_IDS],
                    }
                    for value, ids in ranked[:_MAX_CODES_IN_RESPONSE]
                ]
            if unknown or duplicates or missing:
                extra["id_errors"] = {
                    "unknown": unknown[:_MAX_IDS_IN_ERROR],
                    "duplicate": duplicates[:_MAX_IDS_IN_ERROR],
                    "missing": missing[:_MAX_IDS_IN_ERROR],
                }
            raise InvalidSubmissionError(msg, extra)

        warnings = []
        if converted:
            ranked_conv = converted.most_common(_MAX_CODES_IN_RESPONSE)
            shown = ", ".join(f"{src!r} -> {dst!r} ({k})" for (src, dst), k in ranked_conv)
            more = f", ... ({len(converted)} distinct)" if len(converted) > len(ranked_conv) else ""
            warnings.append(
                f"{sum(converted.values())} prediction(s) were converted to ISO 639-3: "
                f"{shown}{more}"
            )
        raw = [by_id[i] for i in range(n)]
        return raw, [conformed[i] for i in range(n)], warnings

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
        it submitted and ``correct`` uses the dataset's ``macro_match``.
        """
        if self.config.state_dir is None:
            return
        mode = served.config.macro_match
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
                    served.golds[i] is not None
                    and labels_match(served.golds[i], preds[i] or UND_TOKEN, mode)
                ),
            }
            for i in served_ids
        )
        write_predictions(rows, run_dir / "predictions.jsonl")
        write_summary(result, run_dir / "summary.json")
