"""Dataclass and JSON/JSONL IO for a single (model, dataset) evaluation run."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from commonlid.metrics.aggregate import macro_average, micro_average
from commonlid.metrics.core import LanguageMetrics

SCHEMA_VERSION = 3


@dataclass(slots=True)
class Result:
    """Aggregate outcome of one model evaluated on one dataset.

    ``supported_languages`` follows a tri-state convention shared with
    :meth:`LIDModel.discover_supported_languages`: ``None`` means the
    model's support set is undefined (e.g. LLM-based models that can be
    prompted for any language), a list of ISO 639-3 codes is the closed
    set the model declares, and an empty list is the degenerate "supports
    zero languages" case. The leaderboard's ``(cov.)`` view consumes this.
    """

    model_id: str
    dataset_id: str
    dataset_revision: str | None
    per_language: dict[str, LanguageMetrics] = field(default_factory=dict)
    samples_per_second: float = 0.0
    n_samples: int = 0
    n_samples_with_gold: int = 0
    limit: int | None = None
    timestamp: str = ""
    commonlid_version: str = ""
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    supported_languages: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return the serialisable JSON summary for this run."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "commonlid_version": self.commonlid_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "timestamp": self.timestamp,
            "limit": self.limit,
            "n_samples": self.n_samples,
            "n_samples_with_gold": self.n_samples_with_gold,
            "samples_per_second": self.samples_per_second,
            "macro": macro_average(self.per_language),
            "micro": micro_average(self.per_language),
            "per_language": {lang: asdict(m) for lang, m in sorted(self.per_language.items())},
            "supported_languages": self.supported_languages,
            "extra": self.extra,
        }


def write_summary(result: Result, path: str | Path) -> Path:
    """Write ``result.summary()`` to ``path`` as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.summary(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_summary(path: str | Path) -> dict[str, Any]:
    """Read a summary JSON file (returned as a plain dict)."""
    with Path(path).open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def write_predictions(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write per-sample prediction rows to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    return path
