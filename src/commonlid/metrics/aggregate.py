"""Aggregate a per-language metric dict into overall scores.

Two views are computed for both macro and micro averages:

* ``*_gold_only`` — averaged over the ``N_gold`` languages with ``gt_count > 0``.
  This is the paper / notebook ``"all"``-column definition. It is the
  load-bearing headline number used by the leaderboard.
* ``*_observed`` — averaged over the languages in ``set(gold) | set(pred)``,
  i.e. every language that either appears in the gold set or the model emits.
  Penalises spurious / hallucinated predictions and is what we use to compare
  models on their willingness to predict outside the test set.

The two views collapse to the same numbers when no predicted-only languages
exist (e.g. legacy-imported summaries that only carry the gold subset).
"""

from __future__ import annotations

from collections.abc import Mapping

from commonlid.metrics.core import UND_TOKEN, LanguageMetrics


def _filter(
    metrics: Mapping[str, LanguageMetrics], include_und: bool, und_token: str
) -> list[tuple[str, LanguageMetrics]]:
    if include_und:
        return list(metrics.items())
    return [(lang, m) for lang, m in metrics.items() if lang != und_token]


def _macro_zeros() -> dict[str, float]:
    return {
        "f1_gold_only": 0.0,
        "precision_gold_only": 0.0,
        "recall_gold_only": 0.0,
        "n_languages_gold": 0,
        "f1_observed": 0.0,
        "precision_observed": 0.0,
        "recall_observed": 0.0,
        "n_languages_observed": 0,
    }


def _micro_zeros() -> dict[str, float]:
    return {
        "f1_gold_only": 0.0,
        "precision_gold_only": 0.0,
        "recall_gold_only": 0.0,
        "n_correct_gold": 0,
        "n_predictions_gold": 0,
        "n_gold_samples": 0,
        "f1_observed": 0.0,
        "precision_observed": 0.0,
        "recall_observed": 0.0,
        "n_correct_observed": 0,
        "n_predictions_observed": 0,
    }


def macro_average(
    metrics: Mapping[str, LanguageMetrics],
    *,
    include_und: bool = False,
    und_token: str = UND_TOKEN,
) -> dict[str, float]:
    """Return macro (unweighted) precision/recall/F1 in two views.

    See module docstring for the difference between ``*_gold_only`` (paper
    headline) and ``*_observed`` (penalises spurious predictions).
    """
    pairs = _filter(metrics, include_und, und_token)
    if not pairs:
        return _macro_zeros()

    observed = [m for _, m in pairs]
    gold_only = [m for _, m in pairs if m.gt_count > 0]
    n_obs = len(observed)
    n_gold = len(gold_only)

    out: dict[str, float] = _macro_zeros()
    if n_gold > 0:
        out.update({
            "f1_gold_only": sum(m.f1 for m in gold_only) / n_gold,
            "precision_gold_only": sum(m.precision for m in gold_only) / n_gold,
            "recall_gold_only": sum(m.recall for m in gold_only) / n_gold,
            "n_languages_gold": n_gold,
        })
    if n_obs > 0:
        out.update({
            "f1_observed": sum(m.f1 for m in observed) / n_obs,
            "precision_observed": sum(m.precision for m in observed) / n_obs,
            "recall_observed": sum(m.recall for m in observed) / n_obs,
            "n_languages_observed": n_obs,
        })
    return out


def micro_average(
    metrics: Mapping[str, LanguageMetrics],
    *,
    include_und: bool = False,
    und_token: str = UND_TOKEN,
) -> dict[str, float]:
    """Return micro (sample-weighted) precision/recall/F1 in two views."""
    pairs = _filter(metrics, include_und, und_token)
    if not pairs:
        return _micro_zeros()

    observed = [m for _, m in pairs]
    gold_only = [m for _, m in pairs if m.gt_count > 0]

    def _block(rows: list[LanguageMetrics]) -> tuple[float, float, float, int, int, int]:
        total_gt = sum(m.gt_count for m in rows)
        total_pred = sum(m.predictions for m in rows)
        total_correct = sum(m.correct for m in rows)
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1, precision, recall, total_correct, total_pred, total_gt

    g_f1, g_p, g_r, g_correct, g_pred, g_gt = _block(gold_only)
    o_f1, o_p, o_r, o_correct, o_pred, _ = _block(observed)

    return {
        "f1_gold_only": g_f1,
        "precision_gold_only": g_p,
        "recall_gold_only": g_r,
        "n_correct_gold": g_correct,
        "n_predictions_gold": g_pred,
        "n_gold_samples": g_gt,
        "f1_observed": o_f1,
        "precision_observed": o_p,
        "recall_observed": o_r,
        "n_correct_observed": o_correct,
        "n_predictions_observed": o_pred,
    }


def mean_stats(
    metrics: Mapping[str, LanguageMetrics],
    *,
    include_und: bool = False,
    und_token: str = UND_TOKEN,
) -> dict[str, float]:
    """Alias for :func:`macro_average` preserved for notebook compatibility."""
    return macro_average(metrics, include_und=include_und, und_token=und_token)
