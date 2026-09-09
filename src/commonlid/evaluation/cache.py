"""Per-(model, dataset) prediction cache.

The cache is a JSONL file under ``{cache_dir}/{dataset_id}/{model_id}.jsonl``.
Each line is ``{"text_hash": "...", "pred": "..." | null, "score": 0.9 | null}``.
The hash key is ``sha256(model_id || dataset_revision || text)[:16]`` so the
cache invalidates automatically when a dataset revision is bumped or the model
changes identity. ``score`` was added later, so entries written before it are
read back with a ``None`` score rather than being discarded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from commonlid.core.lid_model import LIDPrediction


def _digest(model_id: str, dataset_revision: str | None, text: str) -> str:
    key = f"{model_id}|{dataset_revision or ''}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


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
                self._store[entry["text_hash"]] = LIDPrediction(entry["pred"], entry.get("score"))

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
                f.write(
                    json.dumps({
                        "text_hash": key,
                        "pred": pred.iso639_3,
                        "score": pred.score,
                    })
                )
                f.write("\n")

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._store)
