"""Per-language precision / recall / F1.

Thin wrapper around :func:`sklearn.metrics.precision_recall_fscore_support`
with the CommonLID counting conventions layered on top:

- Samples where ``gold`` is ``None`` are skipped entirely.
- Predictions of ``None`` are mapped to the ``und_token`` (default ``"und"``)
  and counted as a regular language bucket (downstream callers may filter
  it from aggregate averages).

The hand-written formulas that used to live here are gone; they matched
sklearn exactly and the smoke-parity tests still assert 1e-6 agreement
with the legacy research code under ``tests/legacy/``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sklearn.metrics import (
    classification_report as _sk_classification_report,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
)

UND_TOKEN = "und"


@dataclass(frozen=True, slots=True)
class LanguageMetrics:
    """Per-language counters and derived metrics.

    ``fpr`` is the *paper-style* per-language false positive rate over the
    whole evaluation slice this metric was computed for: ``FP / (FP + TN)``
    where ``TN`` is the sum of *correct predictions across other languages*
    (matches ``wmdqs/eval/notebooks/eval_langid_metrics_for_llms.ipynb``).
    ``None`` when the language has no signal at all (no predictions and no
    correct other-language predictions). For subset analyses, recompute via
    :func:`commonlid.metrics.fpr.mean_false_positive_rate`.
    """

    gt_count: int
    predictions: int
    correct: int
    precision: float
    recall: float
    f1: float
    fpr: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _prepare(
    ytrue: Sequence[str | None],
    ypred: Sequence[str | None],
    und_token: str,
) -> tuple[list[str], list[str]]:
    """Drop gold-None samples and map pred-None to ``und_token``."""
    if len(ytrue) != len(ypred):
        msg = f"ytrue/ypred length mismatch: {len(ytrue)} vs {len(ypred)}"
        raise ValueError(msg)
    gold: list[str] = []
    pred: list[str] = []
    for g, p in zip(ytrue, ypred, strict=True):
        if g is None:
            continue
        gold.append(g)
        pred.append(und_token if p is None else p)
    return gold, pred


def compute_per_language_metrics(
    ytrue: Sequence[str | None],
    ypred: Sequence[str | None],
    *,
    und_token: str = UND_TOKEN,
    sample_count_threshold: int = 0,
) -> dict[str, LanguageMetrics]:
    """Compute per-language precision / recall / F1 via sklearn.

    ``ytrue`` and ``ypred`` must be the same length. Gold ``None`` samples
    are skipped (no counters updated). Predicted ``None`` is mapped to
    ``und_token`` (default ``"und"``) and counted as a regular language.

    Languages whose gold sample count is below ``sample_count_threshold`` are
    excluded from the returned dict.
    """
    gold, pred = _prepare(ytrue, ypred, und_token)
    if not gold:
        return {}

    labels = sorted(set(gold) | set(pred))
    precisions, recalls, f1s, support = precision_recall_fscore_support(
        gold, pred, labels=labels, average=None, zero_division=0
    )
    # sklearn gives per-class P/R/F1 + support (gt_count) but not the
    # prediction / correct counts we surface; compute them here.
    pred_counts: Counter[str] = Counter(pred)
    correct_counts: Counter[str] = Counter(g for g, p in zip(gold, pred, strict=True) if g == p)
    total_correct_all = sum(correct_counts.values())

    out: dict[str, LanguageMetrics] = {}
    for i, label in enumerate(labels):
        gt_count = int(support[i])
        if gt_count < sample_count_threshold:
            continue
        preds = int(pred_counts[label])
        correct = int(correct_counts[label])
        # Paper-style FPR (matches generate_nano_datasets / Table 3 notebook):
        # TN = sum of correct predictions across OTHER languages, not the classical
        # "all non-target predictions of non-target".
        fp = preds - correct
        tn = total_correct_all - correct
        if preds == 0 and tn == 0:
            fpr: float | None = None
        else:
            denom = fp + tn
            fpr = (fp / denom) if denom > 0 else 0.0
        out[label] = LanguageMetrics(
            gt_count=gt_count,
            predictions=preds,
            correct=correct,
            precision=float(precisions[i]),
            recall=float(recalls[i]),
            f1=float(f1s[i]),
            fpr=fpr,
        )
    return out


def classification_report(
    ytrue: Sequence[str | None],
    ypred: Sequence[str | None],
    *,
    und_token: str = UND_TOKEN,
    include_und: bool = False,
    digits: int = 4,
) -> dict[str, Any]:
    """Return :func:`sklearn.metrics.classification_report` as a dict.

    Applies the CommonLID preprocessing (drop gold-None, map pred-None to
    ``und_token``). When ``include_und`` is false the ``und`` bucket is
    hidden from the report, matching the default behaviour of
    :func:`macro_average` / :func:`micro_average`.
    """
    gold, pred = _prepare(ytrue, ypred, und_token)
    if not gold:
        return {}

    labels = sorted(set(gold) | set(pred))
    if not include_und and und_token in labels:
        labels.remove(und_token)

    report = _sk_classification_report(
        gold,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
        digits=digits,
    )
    return dict(report)
