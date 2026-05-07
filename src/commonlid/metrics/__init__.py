"""Evaluation metrics.

Public API:
- :func:`compute_per_language_metrics` — precision/recall/F1/FPR per language.
- :func:`macro_average`, :func:`micro_average` — aggregate across languages.
- :func:`false_positive_rate` — classical per-language FPR on raw ytrue/ypred.
- :func:`mean_false_positive_rate` — paper-style mean FPR over a language subset.
- :func:`mean_stats_with_coverage` — notebook-style ``all`` / ``cov.`` slice.
- :func:`load_support_matrix` / :func:`save_support_matrix` — CSV round-trip.
"""

from commonlid.metrics.aggregate import macro_average, mean_stats, micro_average
from commonlid.metrics.core import (
    LanguageMetrics,
    classification_report,
    compute_per_language_metrics,
)
from commonlid.metrics.fpr import (
    false_positive_rate,
    mean_false_positive_rate,
    mean_stats_with_coverage,
    stats_per_model_supported,
)
from commonlid.metrics.support_matrix import load_support_matrix, save_support_matrix

__all__ = [
    "LanguageMetrics",
    "classification_report",
    "compute_per_language_metrics",
    "false_positive_rate",
    "load_support_matrix",
    "macro_average",
    "mean_false_positive_rate",
    "mean_stats",
    "mean_stats_with_coverage",
    "micro_average",
    "save_support_matrix",
    "stats_per_model_supported",
]
