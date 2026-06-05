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
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from commonlid.metrics.core import LanguageMetrics
from commonlid.metrics.fpr import mean_false_positive_rate, mean_stats_with_coverage

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

    The ``*_cov`` mirror fields are the same metrics restricted to gold
    samples whose language is in the model's declared support set
    (``supported_languages``). They are ``None`` when no support set is
    available — either the field is missing from ``summary.json`` (legacy
    file), the field is JSON ``null`` (LLM-style models whose support set
    is undefined), or the field is an empty list (degenerate "supports
    zero languages"). All three render as em-dashes in the cov view.
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
    supported_languages: list[str] | None
    macro_f1_cov: float | None
    macro_precision_cov: float | None
    macro_recall_cov: float | None
    micro_f1_cov: float | None
    mean_fpr_cov: float | None
    n_languages_cov: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_mean_fpr(per_language: dict[str, dict[str, Any]]) -> float:
    """Mean of paper-style ``fpr`` per language, ignoring ``None`` entries."""
    vals = [m["fpr"] for m in per_language.values() if m.get("fpr") is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _hydrate_per_language(
    per_language: Mapping[str, Mapping[str, Any]],
) -> dict[str, LanguageMetrics]:
    """Reconstruct :class:`LanguageMetrics` objects from the serialised dict form."""
    out: dict[str, LanguageMetrics] = {}
    for lang, m in per_language.items():
        out[lang] = LanguageMetrics(
            gt_count=int(m.get("gt_count", 0)),
            predictions=int(m.get("predictions", 0)),
            correct=int(m.get("correct", 0)),
            precision=float(m.get("precision", 0.0) or 0.0),
            recall=float(m.get("recall", 0.0) or 0.0),
            f1=float(m.get("f1", 0.0) or 0.0),
            fpr=None if m.get("fpr") is None else float(m["fpr"]),
        )
    return out


def _micro_average_over(rows: Mapping[str, LanguageMetrics]) -> tuple[float, float, float]:
    """Pooled precision/recall/F1 over a (filtered) per-language slice.

    Mirrors :func:`commonlid.metrics.aggregate.micro_average`'s
    ``*_gold_only`` math but accepts a pre-filtered subset, which the
    public helper does not.
    """
    total_correct = sum(m.correct for m in rows.values())
    total_predictions = sum(m.predictions for m in rows.values())
    total_gold = sum(m.gt_count for m in rows.values())
    precision = total_correct / total_predictions if total_predictions > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 and not math.isclose(precision + recall, 0.0)
        else 0.0
    )
    return precision, recall, f1


def _compute_cov_fields(
    per_language_raw: Mapping[str, Mapping[str, Any]],
    supported: list[str] | None,
) -> dict[str, float | int | None]:
    """Return the six cov-variant fields, or all ``None`` when no cov data.

    ``supported`` semantics:

    - ``None`` — model's support set is undefined (e.g. LLM); no cov data.
    - ``[]`` — model declared zero supported languages; every cov metric
      would divide by zero, so render as no-data.
    - non-empty list — compute the cov metrics.
    """
    none_result: dict[str, float | int | None] = {
        "macro_f1_cov": None,
        "macro_precision_cov": None,
        "macro_recall_cov": None,
        "micro_f1_cov": None,
        "mean_fpr_cov": None,
        "n_languages_cov": None,
    }
    if not supported:
        return none_result

    supported_set = set(supported)
    per_language = _hydrate_per_language(per_language_raw)
    stats = mean_stats_with_coverage(per_language, model_supported_languages=supported_set)
    cov = stats["cov"]
    n_languages_cov = int(cov.get("cov_count", 0))
    if n_languages_cov == 0:
        # Supported set has no overlap with the dataset's gold; nothing
        # meaningful to report.
        return none_result

    cov_rows = {
        lang: m for lang, m in per_language.items() if m.gt_count > 0 and lang in supported_set
    }
    _micro_precision, _micro_recall, micro_f1 = _micro_average_over(cov_rows)
    mean_fpr_cov = mean_false_positive_rate(per_language, language_whitelist=supported_set)

    return {
        "macro_f1_cov": float(cov["f1"]),
        "macro_precision_cov": float(cov["precision"]),
        "macro_recall_cov": float(cov["recall"]),
        "micro_f1_cov": float(micro_f1),
        "mean_fpr_cov": float(mean_fpr_cov),
        "n_languages_cov": n_languages_cov,
    }


def _row_from_summary(summary: dict[str, Any], dataset_id: str, model_id: str) -> LeaderboardRow:
    macro = summary.get("macro", {})
    micro = summary.get("micro", {})
    extra = summary.get("extra", {}) or {}
    per_language = summary.get("per_language", {}) or {}

    # Tri-state: missing key, JSON null, or list. Anything else (e.g. an
    # accidentally-serialised set) collapses to "unknown".
    raw_supported = summary.get("supported_languages")
    supported: list[str] | None = list(raw_supported) if isinstance(raw_supported, list) else None
    cov = _compute_cov_fields(per_language, supported)

    n_languages_cov = cov["n_languages_cov"]
    return LeaderboardRow(
        dataset_id=dataset_id,
        model_id=model_id,
        macro_f1=float(macro.get("f1_gold_only", 0.0)),
        macro_precision=float(macro.get("precision_gold_only", 0.0)),
        macro_recall=float(macro.get("recall_gold_only", 0.0)),
        micro_f1=float(micro.get("f1_gold_only", 0.0)),
        mean_fpr=_safe_mean_fpr(per_language),
        n_languages=int(macro.get("n_languages_observed", 0)),
        n_samples=int(summary.get("n_samples", 0)),
        n_samples_with_gold=int(summary.get("n_samples_with_gold", 0)),
        samples_per_second=float(summary.get("samples_per_second", 0.0)),
        dataset_revision=summary.get("dataset_revision"),
        commonlid_version=str(summary.get("commonlid_version", "")),
        timestamp=str(summary.get("timestamp", "")),
        is_imported=("imported_from" in extra),
        supported_languages=supported,
        macro_f1_cov=cov["macro_f1_cov"],
        macro_precision_cov=cov["macro_precision_cov"],
        macro_recall_cov=cov["macro_recall_cov"],
        micro_f1_cov=cov["micro_f1_cov"],
        mean_fpr_cov=cov["mean_fpr_cov"],
        n_languages_cov=int(n_languages_cov) if n_languages_cov is not None else None,
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
