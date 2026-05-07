"""Pull leaderboard rows from the published results HF dataset.

Layout expected on the HF repo (created via ``huggingface-cli upload`` from
``data/results/``)::

    <dataset_id>/<model_id>/summary.json
    <dataset_id>/<model_id>/predictions.jsonl

This module reads only ``summary.json`` (predictions stay in the dataset for
audit / drilldown but the leaderboard table is summary-only).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "commoncrawl/commonlid-results"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    """One row in the leaderboard — derived from a single ``summary.json``.

    The metric fields (``macro_f1`` / ``macro_precision`` / ``macro_recall`` /
    ``micro_f1``) use the **paper / gold-only** definition, averaged over
    languages with ``gt_count > 0`` and matching the published CommonLID
    numbers.

    The ``n_languages`` field uses the **observed** count (``set(gold) |
    set(pred)``) -- i.e. how many distinct languages the model is willing
    to emit on this dataset, including any spurious labels outside the
    gold set. That's a model-property number, not a paper headline, and
    it stays consistent across rows: every model is reported on the same
    "what languages did you actually output here" basis.
    """

    dataset_id: str
    model_id: str
    macro_f1: float
    macro_precision: float
    macro_recall: float
    micro_f1: float
    mean_fpr: float
    n_languages: int
    n_samples: int
    n_samples_with_gold: int
    samples_per_second: float
    dataset_revision: str | None
    commonlid_version: str
    timestamp: str
    is_imported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_mean_fpr(per_language: dict[str, dict[str, Any]]) -> float:
    """Mean of paper-style ``fpr`` per language, ignoring ``None`` entries."""
    vals = [m["fpr"] for m in per_language.values() if m.get("fpr") is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _row_from_summary(summary: dict[str, Any], dataset_id: str, model_id: str) -> LeaderboardRow:
    macro = summary.get("macro", {})
    micro = summary.get("micro", {})
    extra = summary.get("extra", {}) or {}
    return LeaderboardRow(
        dataset_id=dataset_id,
        model_id=model_id,
        macro_f1=float(macro.get("f1_gold_only", 0.0)),
        macro_precision=float(macro.get("precision_gold_only", 0.0)),
        macro_recall=float(macro.get("recall_gold_only", 0.0)),
        micro_f1=float(micro.get("f1_gold_only", 0.0)),
        mean_fpr=_safe_mean_fpr(summary.get("per_language", {})),
        n_languages=int(macro.get("n_languages_observed", 0)),
        n_samples=int(summary.get("n_samples", 0)),
        n_samples_with_gold=int(summary.get("n_samples_with_gold", 0)),
        samples_per_second=float(summary.get("samples_per_second", 0.0)),
        dataset_revision=summary.get("dataset_revision"),
        commonlid_version=str(summary.get("commonlid_version", "")),
        timestamp=str(summary.get("timestamp", "")),
        is_imported=("imported_from" in extra),
    )


def _walk_local(root: Path, allowed_datasets: set[str] | None) -> Iterable[LeaderboardRow]:
    """Yield rows for every ``<dataset>/<model>/summary.json`` under ``root``."""
    for summary_path in sorted(root.glob(f"*/*/{SUMMARY_FILENAME}")):
        dataset_id = summary_path.parent.parent.name
        model_id = summary_path.parent.name
        if allowed_datasets is not None and dataset_id not in allowed_datasets:
            continue
        try:
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("skipping %s: %s", summary_path, exc)
            continue
        yield _row_from_summary(summary, dataset_id, model_id)


def load_results(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    allowed_datasets: Iterable[str] | None = None,
    local_dir: str | Path | None = None,
) -> Any:
    """Snapshot ``repo_id`` from the Hub, then walk it into a tidy DataFrame.

    Parameters
    ----------
    repo_id:
        HF dataset repo id holding ``<dataset>/<model>/summary.json`` files.
    revision:
        Optional commit SHA / branch / tag to pin.
    cache_dir:
        Where to cache the snapshot; defaults to the standard HF cache.
    allowed_datasets:
        If set, restrict the result to these dataset_ids (one per row).
    local_dir:
        Skip the network entirely and read summaries from this directory
        instead. Useful for offline development and tests.
    """
    import pandas as pd

    if local_dir is not None:
        root = Path(local_dir)
    else:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=[f"*/*/{SUMMARY_FILENAME}"],
        )
        root = Path(path)

    allowed = set(allowed_datasets) if allowed_datasets is not None else None
    rows = [r.to_dict() for r in _walk_local(root, allowed)]
    if not rows:
        return pd.DataFrame(columns=[f.name for f in LeaderboardRow.__dataclass_fields__.values()])
    return pd.DataFrame.from_records(rows)
