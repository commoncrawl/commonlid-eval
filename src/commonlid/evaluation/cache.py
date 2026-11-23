"""Per-(model, dataset) prediction cache.

The cache is a JSONL file under ``{cache_dir}/{dataset_id}/{model_id}.jsonl``.
Each line is ``{"text_hash": "...", "pred": "..." | null, "score": 0.9 | null,
"pred_raw": "..." | null}``. The hash key is
``sha256(model_id || dataset_revision || text)[:16]`` so the cache invalidates
automatically when a dataset revision is bumped or the model changes identity.

``score`` and ``pred_raw`` were added later. Entries written before them are
read back with a ``None`` score and an unknown raw output
(:data:`~commonlid.core.lid_model.RAW_UNSET`) rather than being discarded.
``pred_raw`` is only written when the raw output is known, so a missing key
means "unknown" while ``null`` means "the model produced no output". When a
text appears on several lines, the last one wins.

Until the language-code fix, every Danish code (``dan``) conformed to ``None``,
so caches written before it hold ``"pred": null`` for Danish outputs. Such an
entry is read back as ``dan`` when its raw output is a Danish label
(``__label__dan_Latn``, ``dan``, ``da``). Older entries without a raw output
cannot be repaired; delete the cache of a local model to predict them again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from commonlid.core.lid_model import RAW_UNSET, LIDPrediction

_DANISH_RAW = frozenset({"da", "dan"})


def _digest(model_id: str, dataset_revision: str | None, text: str) -> str:
    key = f"{model_id}|{dataset_revision or ''}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _is_danish_raw(raw: object) -> bool:
    """Whether a raw model label is Danish: ``__label__dan_Latn``, ``dan``, ``da-DK``."""
    if not isinstance(raw, str):
        return False
    code = raw.strip().removeprefix("__label__").replace("-", "_").split("_", 1)[0]
    return code.lower() in _DANISH_RAW


def _repair_pred(pred: str | None, raw: object) -> str | None:
    """Undo the old ``dan`` -> ``None`` conformation (see the module docstring)."""
    if pred is None and _is_danish_raw(raw):
        return "dan"
    return pred


class PredictionCache:
    """File-backed cache of per-text predictions, scoped to (model, dataset)."""

    def __init__(
        self,
        cache_dir: str | Path,
        model_id: str,
        dataset_id: str,
        dataset_revision: str | None,
    ) -> None:
        self._path = Path(cache_dir) / dataset_id / f"{model_id}.jsonl"
        self._model_id = model_id
        self._dataset_revision = dataset_revision
        self._store: dict[str, LIDPrediction] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                raw = entry.get("pred_raw", RAW_UNSET)
                self._store[entry["text_hash"]] = LIDPrediction(
                    _repair_pred(entry["pred"], raw), entry.get("score"), raw
                )

    def get(self, text: str) -> tuple[bool, LIDPrediction]:
        """Return ``(hit, prediction)``; the prediction is empty when ``hit`` is ``False``."""
        key = _digest(self._model_id, self._dataset_revision, text)
        if key in self._store:
            return True, self._store[key]
        return False, LIDPrediction(None, None)

    def put(self, text: str, pred: LIDPrediction) -> None:
        """Record a prediction for ``text`` and append to the JSONL."""
        self.put_many([(text, pred)])

    def put_many(self, pairs: Iterable[tuple[str, LIDPrediction]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            for text, pred in pairs:
                key = _digest(self._model_id, self._dataset_revision, text)
                self._store[key] = pred
                row: dict[str, object] = {
                    "text_hash": key,
                    "pred": pred.iso639_3,
                    "score": pred.score,
                }
                if pred.has_raw:
                    row["pred_raw"] = pred.raw
                f.write(json.dumps(row))
                f.write("\n")

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._store)
