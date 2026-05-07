"""False-positive-rate helpers.

Two flavors of FPR are exposed:

1. :func:`false_positive_rate` — *classical* FPR ``FP / (FP + TN)`` where
   ``TN`` is every non-target sample that was *not* labelled as the target.
   Operates on the raw ``(ytrue, ypred)`` lists.

2. :func:`mean_false_positive_rate` — *paper-style* mean FPR over a set of
   languages, mirroring the notebook formula in
   ``wmdqs/eval/notebooks/eval_langid_metrics_for_llms.ipynb``: per-language
   ``TN`` is the sum of *correct predictions across other languages in the
   subset*. Used to reproduce Table 3 of the CommonLID paper.

Per-language paper-style FPR is also stored on each
:class:`commonlid.metrics.core.LanguageMetrics` (computed across the whole
evaluation slice). For a subset of languages, recompute via
:func:`mean_false_positive_rate` so ``TN`` is restricted to the subset.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from commonlid.metrics.core import UND_TOKEN, LanguageMetrics


def false_positive_rate(
    ytrue: Sequence[str | None],
    ypred: Sequence[str | None],
    *,
    language: str,
    model_supported_languages: set[str] | None = None,
    und_token: str = UND_TOKEN,
) -> float:
    """Fraction of non-``language`` samples that the model labelled as ``language``.

    If ``model_supported_languages`` is provided, samples whose gold code is
    outside that set are excluded — so the rate reflects only confusions
    among languages the model claims to support.
    """
    if len(ytrue) != len(ypred):
        msg = f"ytrue/ypred length mismatch: {len(ytrue)} vs {len(ypred)}"
        raise ValueError(msg)

    relevant = 0
    false_positives = 0
    for gold, pred in zip(ytrue, ypred, strict=True):
        if gold is None:
            continue
        if model_supported_languages is not None and gold not in model_supported_languages:
            continue
        if gold == language:
            continue
        relevant += 1
        pred_code = und_token if pred is None else pred
        if pred_code == language:
            false_positives += 1

    return false_positives / relevant if relevant > 0 else 0.0


def stats_per_model_supported(
    results: Iterable[Any],
    support_matrix: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """Filter per-language metrics to languages the model supports and compute averages.

    ``results`` should be an iterable of objects exposing ``model_id``,
    ``dataset_id``, and ``per_language`` (a ``dict[str, LanguageMetrics]``).
    """
    out: list[dict[str, Any]] = []
    for result in results:
        supported = support_matrix.get(result.model_id, set())
        selected = {lang: lm for lang, lm in result.per_language.items() if lang in supported}
        n = len(selected)
        if n == 0:
            continue
        out.append({
            "model_id": result.model_id,
            "dataset_id": result.dataset_id,
            "n_supported_languages_evaluated": n,
            "mean_precision": sum(m.precision for m in selected.values()) / n,
            "mean_recall": sum(m.recall for m in selected.values()) / n,
            "mean_f1": sum(m.f1 for m in selected.values()) / n,
        })
    return out


def mean_stats_with_coverage(
    per_language: Mapping[str, LanguageMetrics],
    *,
    model_supported_languages: Iterable[str] | None = None,
    language_whitelist: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Notebook-style mean precision/recall/F1 with the dual ``all`` / ``cov.`` slice.

    Mirrors ``stats_per_model_supported`` from the eval-langid-metrics
    notebook: returns a dict with two entries.

    * ``"all"`` — mean over languages that have ``gt_count > 0`` (and are in
      ``language_whitelist`` when provided). This matches the notebook's
      ``language iso639-3 == 'all'`` row.
    * ``"cov"`` — same, additionally restricted to languages the model
      supports (``model_supported_languages``). Matches the
      ``'cov.'`` row, including the ``cov_count`` of distinct supported
      languages evaluated.

    Both slices use a simple unweighted mean over the per-language metrics
    (matching the notebook's ``DataFrame.mean()`` semantics on filtered
    rows).
    """
    whitelist = set(language_whitelist) if language_whitelist is not None else None
    supported = set(model_supported_languages) if model_supported_languages is not None else None

    eligible = {
        lang: m
        for lang, m in per_language.items()
        if m.gt_count > 0 and (whitelist is None or lang in whitelist)
    }

    def _mean(rows: Mapping[str, LanguageMetrics]) -> dict[str, float]:
        n = len(rows)
        if n == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_languages": 0}
        return {
            "precision": sum(m.precision for m in rows.values()) / n,
            "recall": sum(m.recall for m in rows.values()) / n,
            "f1": sum(m.f1 for m in rows.values()) / n,
            "n_languages": n,
        }

    cov_rows = (
        eligible
        if supported is None
        else {lang: m for lang, m in eligible.items() if lang in supported}
    )
    cov = _mean(cov_rows)
    cov["cov_count"] = len(cov_rows)
    return {"all": _mean(eligible), "cov": cov}


def mean_false_positive_rate(
    per_language: Mapping[str, LanguageMetrics],
    *,
    language_whitelist: Iterable[str] | None = None,
) -> float:
    """Mean per-language paper-style FPR over a (possibly restricted) language set.

    When ``language_whitelist`` is ``None``, averages the per-language
    ``LanguageMetrics.fpr`` already stored on each entry (computed across
    the whole evaluation slice; ``None`` entries are skipped).

    When ``language_whitelist`` is provided, **recomputes** every per-language
    FPR within the subset: ``TN`` only counts correct predictions for *other
    languages in the whitelist*. This matches the notebook's behaviour where
    ``language_whitelist`` filters the rows before counting and is the right
    thing for paper Table 3 reproduction.

    Languages where both ``predictions == 0`` and the in-scope ``TN == 0``
    (no signal at all) are dropped, mirroring the notebook's ``return None``
    branch.
    """
    if language_whitelist is None:
        valid = [m.fpr for m in per_language.values() if m.fpr is not None]
        return sum(valid) / len(valid) if valid else 0.0

    whitelist = set(language_whitelist)
    # Languages absent from ``per_language`` still count toward the average:
    # they had no predictions and no gold rows in the eval, so FP = 0 and
    # FPR = 0 for them — matching the legacy CSV which pre-allocates a row
    # for every (model, lang, dataset) regardless of gt_count.
    zero_metric = LanguageMetrics(
        gt_count=0, predictions=0, correct=0, precision=0.0, recall=0.0, f1=0.0
    )
    in_scope = {lang: per_language.get(lang, zero_metric) for lang in whitelist}
    total_correct_in_scope = sum(m.correct for m in in_scope.values())

    fprs: list[float] = []
    for m in in_scope.values():
        fp = m.predictions - m.correct
        tn = total_correct_in_scope - m.correct
        if m.predictions == 0 and tn == 0:
            continue
        denom = fp + tn
        fprs.append((fp / denom) if denom > 0 else 0.0)

    return sum(fprs) / len(fprs) if fprs else 0.0
