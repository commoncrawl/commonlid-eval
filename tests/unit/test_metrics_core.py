from __future__ import annotations

import math

import pytest

from commonlid.metrics.core import LanguageMetrics, compute_per_language_metrics


def test_perfect_predictions() -> None:
    ytrue = ["eng", "eng", "deu", "deu"]
    ypred = ["eng", "eng", "deu", "deu"]
    metrics = compute_per_language_metrics(ytrue, ypred)
    assert set(metrics) == {"eng", "deu"}
    for m in metrics.values():
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0


def test_all_wrong_predictions() -> None:
    ytrue = ["eng", "deu"]
    ypred = ["fra", "spa"]
    metrics = compute_per_language_metrics(ytrue, ypred)
    # eng/deu have gt=1, preds=0 (not predicted), so precision=0 recall=0.
    # fra/spa have gt=0, preds=1, so precision=0 recall=0 (0/0).
    for lang in ("eng", "deu", "fra", "spa"):
        assert metrics[lang].f1 == 0.0


def test_mixed_precision_recall() -> None:
    ytrue = ["eng", "eng", "eng", "deu"]
    ypred = ["eng", "eng", "deu", "deu"]
    metrics = compute_per_language_metrics(ytrue, ypred)
    # eng: gt=3, preds=2, correct=2 -> p=1.0, r=2/3, f1=2*1*2/3/(1+2/3)=0.8
    eng = metrics["eng"]
    assert eng.gt_count == 3
    assert eng.predictions == 2
    assert eng.correct == 2
    assert math.isclose(eng.precision, 1.0)
    assert math.isclose(eng.recall, 2 / 3)
    assert math.isclose(eng.f1, 0.8)

    # deu: gt=1, preds=2, correct=1 -> p=0.5, r=1, f1=2/3
    deu = metrics["deu"]
    assert math.isclose(deu.precision, 0.5)
    assert math.isclose(deu.recall, 1.0)
    assert math.isclose(deu.f1, 2 / 3)


def test_none_pred_maps_to_und() -> None:
    ytrue = ["eng", "eng"]
    ypred = ["eng", None]
    metrics = compute_per_language_metrics(ytrue, ypred)
    assert "und" in metrics
    und = metrics["und"]
    assert und.gt_count == 0
    assert und.predictions == 1
    assert und.correct == 0


def test_none_gold_is_skipped() -> None:
    ytrue = ["eng", None, "eng"]
    ypred = ["eng", "eng", "eng"]
    metrics = compute_per_language_metrics(ytrue, ypred)
    # The None-gold sample should not increment anything; eng gt=2, preds=2, correct=2.
    eng = metrics["eng"]
    assert eng.gt_count == 2
    assert eng.predictions == 2
    assert eng.correct == 2


def test_sample_count_threshold_filters_rare_languages() -> None:
    ytrue = ["eng", "eng", "eng", "fra"]
    ypred = ["eng", "eng", "eng", "fra"]
    metrics = compute_per_language_metrics(ytrue, ypred, sample_count_threshold=2)
    assert "eng" in metrics
    assert "fra" not in metrics


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_per_language_metrics(["eng"], ["eng", "deu"])


def test_language_metrics_to_dict_is_json_friendly() -> None:
    m = LanguageMetrics(gt_count=3, predictions=2, correct=2, precision=1.0, recall=2 / 3, f1=0.8)
    d = m.to_dict()
    assert d["gt_count"] == 3
    assert d["correct"] == 2


def test_classification_report_returns_sklearn_dict() -> None:
    from commonlid.metrics.core import classification_report

    ytrue = ["eng", "eng", "deu", "fra", "fra"]
    ypred = ["eng", "eng", "deu", "fra", "spa"]
    report = classification_report(ytrue, ypred)
    # sklearn's output_dict shape: one entry per label + accuracy + macro + weighted averages.
    assert "eng" in report
    assert "deu" in report
    assert "fra" in report
    assert "macro avg" in report
    assert "weighted avg" in report
    assert math.isclose(report["fra"]["precision"], 1.0)
    assert math.isclose(report["fra"]["recall"], 0.5)
    assert math.isclose(report["fra"]["f1-score"], 2 / 3)


def test_classification_report_excludes_und_by_default() -> None:
    from commonlid.metrics.core import classification_report

    # "fra" gold + pred=None -> the pred becomes "und". By default we hide it.
    ytrue = ["eng", "eng", "fra"]
    ypred = ["eng", "eng", None]
    report = classification_report(ytrue, ypred)
    assert "und" not in report
    report_with_und = classification_report(ytrue, ypred, include_und=True)
    assert "und" in report_with_und


def test_classification_report_empty_after_filtering() -> None:
    from commonlid.metrics.core import classification_report

    assert classification_report([None, None], [None, None]) == {}


def test_core_agrees_with_sklearn_precision_recall_fscore_support() -> None:
    """``compute_per_language_metrics`` is sklearn + our counts."""
    from sklearn.metrics import precision_recall_fscore_support

    ytrue = ["eng", "eng", "deu", "fra", "fra", "fra"]
    ypred = ["eng", "deu", "deu", "fra", "eng", "spa"]
    metrics = compute_per_language_metrics(ytrue, ypred)
    labels = sorted({*ytrue, *ypred})
    p, r, f, _ = precision_recall_fscore_support(
        ytrue, ypred, labels=labels, average=None, zero_division=0
    )
    for i, lang in enumerate(labels):
        assert math.isclose(metrics[lang].precision, float(p[i]))
        assert math.isclose(metrics[lang].recall, float(r[i]))
        assert math.isclose(metrics[lang].f1, float(f[i]))
