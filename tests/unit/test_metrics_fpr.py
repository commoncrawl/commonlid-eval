from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from commonlid.metrics.core import LanguageMetrics
from commonlid.metrics.fpr import (
    false_positive_rate,
    mean_false_positive_rate,
    mean_stats_with_coverage,
    stats_per_model_supported,
)


def test_false_positive_rate_basic() -> None:
    # 3 non-eng samples, 1 falsely predicted as eng
    ytrue = ["eng", "deu", "deu", "fra"]
    ypred = ["eng", "eng", "deu", "fra"]
    fpr = false_positive_rate(ytrue, ypred, language="eng")
    assert math.isclose(fpr, 1 / 3)


def test_false_positive_rate_no_relevant() -> None:
    ytrue = ["eng", "eng"]
    ypred = ["eng", "eng"]
    assert false_positive_rate(ytrue, ypred, language="eng") == 0.0


def test_false_positive_rate_skips_out_of_support() -> None:
    ytrue = ["deu", "spa", "zho"]
    ypred = ["eng", "eng", "eng"]
    # Only eng/deu in supported set -> relevant non-eng samples = 1 (deu), which was predicted as eng.
    fpr = false_positive_rate(
        ytrue, ypred, language="eng", model_supported_languages={"eng", "deu"}
    )
    assert fpr == 1.0


def test_false_positive_rate_skips_none_gold() -> None:
    ytrue = ["eng", None, "deu"]
    ypred = ["eng", "eng", "eng"]
    fpr = false_positive_rate(ytrue, ypred, language="eng")
    assert fpr == 1.0  # one non-eng sample (deu), falsely predicted as eng


def test_false_positive_rate_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        false_positive_rate(["eng"], ["eng", "deu"], language="eng")


@dataclass
class _FakeResult:
    model_id: str
    dataset_id: str
    per_language: dict[str, LanguageMetrics]


def test_stats_per_model_supported_computes_means() -> None:
    per_lang = {
        "eng": LanguageMetrics(10, 9, 8, 0.9, 0.8, 0.85),
        "deu": LanguageMetrics(10, 10, 5, 0.5, 0.5, 0.5),
        "xyz": LanguageMetrics(10, 10, 10, 1.0, 1.0, 1.0),
    }
    result = _FakeResult("mymodel", "mydataset", per_lang)
    support = {"mymodel": {"eng", "deu"}}
    rows = stats_per_model_supported([result], support)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_supported_languages_evaluated"] == 2
    # Only eng and deu counted; xyz ignored.
    assert math.isclose(row["mean_precision"], (0.9 + 0.5) / 2)


def test_stats_per_model_supported_skips_empty_intersection() -> None:
    result = _FakeResult("unknown", "ds", {"eng": LanguageMetrics(1, 1, 1, 1.0, 1.0, 1.0)})
    rows = stats_per_model_supported([result], {"other": {"eng"}})
    assert rows == []


# --- Notebook-style FPR (paper Table 3) ----------------------------------


def test_compute_per_language_metrics_includes_paper_fpr() -> None:
    """``LanguageMetrics.fpr`` follows the notebook formula FP / (FP + TN_correct_other)."""
    from commonlid.metrics.core import compute_per_language_metrics

    # Setup: 5 eng samples (4 predicted eng correctly, 1 predicted deu),
    #        5 deu samples (5 predicted deu correctly).
    # There's also 1 fra sample predicted as eng → an English false-positive.
    ytrue = ["eng"] * 5 + ["deu"] * 5 + ["fra"]
    ypred = ["eng"] * 4 + ["deu"] + ["deu"] * 5 + ["eng"]
    pl = compute_per_language_metrics(ytrue, ypred)
    # eng: predictions=5 (4 correct), so FP=1; TN = correct(deu) + correct(fra) = 5 + 0 = 5
    #      FPR = 1 / (1 + 5) = 1/6
    assert math.isclose(pl["eng"].fpr, 1 / 6)
    # deu: predictions=6 (5 correct), so FP=1; TN = correct(eng) + correct(fra) = 4 + 0 = 4
    #      FPR = 1 / (1 + 4) = 1/5
    assert math.isclose(pl["deu"].fpr, 1 / 5)
    # fra: predictions=0; TN = correct(eng) + correct(deu) = 4 + 5 = 9
    #      FPR = 0 / (0 + 9) = 0.0
    assert pl["fra"].fpr == 0.0


def test_compute_per_language_metrics_fpr_is_none_when_no_signal() -> None:
    """A language with no predictions and no correct other-lang predictions has FPR=None."""
    from commonlid.metrics.core import compute_per_language_metrics

    # All eng samples wrong (predicted xx). No correct predictions anywhere.
    ytrue = ["eng", "eng"]
    ypred = ["xx", "xx"]
    pl = compute_per_language_metrics(ytrue, ypred)
    # eng: predictions=0 (xx is the only predicted lang), TN=correct(xx)=0 → fpr=None
    assert pl["eng"].fpr is None


def test_mean_false_positive_rate_uses_stored_per_lang_when_no_whitelist() -> None:
    """Without a whitelist, the mean is the simple average of stored fpr fields."""
    per_lang = {
        "eng": LanguageMetrics(10, 10, 5, 0.5, 0.5, 0.5, fpr=0.10),
        "deu": LanguageMetrics(10, 10, 5, 0.5, 0.5, 0.5, fpr=0.20),
        "fra": LanguageMetrics(10, 10, 5, 0.5, 0.5, 0.5, fpr=None),
    }
    assert math.isclose(mean_false_positive_rate(per_lang), (0.10 + 0.20) / 2)


def test_mean_false_positive_rate_recomputes_within_whitelist() -> None:
    """Whitelist restricts both target rows and the TN pool."""
    per_lang = {
        "eng": LanguageMetrics(10, 11, 9, 0.0, 0.0, 0.0),  # FP=2
        "deu": LanguageMetrics(10, 9, 7, 0.0, 0.0, 0.0),  # FP=2, correct=7
        "fra": LanguageMetrics(10, 8, 6, 0.0, 0.0, 0.0),  # correct=6
        # 'xyz' is outside the whitelist and must not contribute.
        "xyz": LanguageMetrics(10, 10, 10, 0.0, 0.0, 0.0),
    }
    # Whitelist = eng, deu, fra:
    #   total_correct_in_scope = 9 + 7 + 6 = 22
    #   eng: FP=2  TN = 22 - 9 = 13   -> FPR = 2/15
    #   deu: FP=2  TN = 22 - 7 = 15   -> FPR = 2/17
    #   fra: FP=2  TN = 22 - 6 = 16   -> FPR = 2/18
    expected = (2 / 15 + 2 / 17 + 2 / 18) / 3
    actual = mean_false_positive_rate(per_lang, language_whitelist={"eng", "deu", "fra"})
    assert math.isclose(actual, expected)


def test_mean_stats_with_coverage_yields_all_and_cov_slices() -> None:
    per_lang = {
        "eng": LanguageMetrics(10, 9, 8, 0.9, 0.8, 0.85),
        "deu": LanguageMetrics(10, 10, 5, 0.5, 0.5, 0.5),
        "ita": LanguageMetrics(10, 7, 6, 0.6, 0.6, 0.6),
        # gt_count=0 → ignored entirely (not in 'all' or 'cov').
        "und": LanguageMetrics(0, 5, 0, 0.0, 0.0, 0.0),
    }
    out = mean_stats_with_coverage(
        per_lang,
        model_supported_languages={"eng", "deu"},  # ita unsupported
        language_whitelist={"eng", "deu", "ita"},  # und excluded by whitelist anyway
    )
    # 'all' = eng, deu, ita → mean f1 = (0.85 + 0.5 + 0.6) / 3
    assert math.isclose(out["all"]["f1"], (0.85 + 0.5 + 0.6) / 3)
    assert out["all"]["n_languages"] == 3
    # 'cov' = eng, deu → mean f1 = (0.85 + 0.5) / 2
    assert math.isclose(out["cov"]["f1"], (0.85 + 0.5) / 2)
    assert out["cov"]["n_languages"] == 2
    assert out["cov"]["cov_count"] == 2


def test_mean_stats_with_coverage_no_filters() -> None:
    """Without any filters 'all' and 'cov' both equal the full per-lang mean."""
    per_lang = {
        "eng": LanguageMetrics(10, 9, 8, 0.9, 0.8, 0.85),
        "deu": LanguageMetrics(10, 10, 5, 0.5, 0.5, 0.5),
    }
    out = mean_stats_with_coverage(per_lang)
    assert math.isclose(out["all"]["f1"], (0.85 + 0.5) / 2)
    assert math.isclose(out["cov"]["f1"], (0.85 + 0.5) / 2)
    assert out["cov"]["cov_count"] == 2
