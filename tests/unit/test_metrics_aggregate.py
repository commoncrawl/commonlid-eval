from __future__ import annotations

import math

from commonlid.metrics.aggregate import macro_average, mean_stats, micro_average
from commonlid.metrics.core import LanguageMetrics


def _fixture() -> dict[str, LanguageMetrics]:
    return {
        "eng": LanguageMetrics(
            gt_count=10,
            predictions=9,
            correct=8,
            precision=8 / 9,
            recall=0.8,
            f1=2 * (8 / 9) * 0.8 / ((8 / 9) + 0.8),
        ),
        "deu": LanguageMetrics(
            gt_count=5,
            predictions=6,
            correct=4,
            precision=4 / 6,
            recall=0.8,
            f1=2 * (4 / 6) * 0.8 / ((4 / 6) + 0.8),
        ),
        "und": LanguageMetrics(
            gt_count=0, predictions=3, correct=0, precision=0.0, recall=0.0, f1=0.0
        ),
    }


def _fixture_with_predicted_only() -> dict[str, LanguageMetrics]:
    """Same as ``_fixture`` plus a non-und language with predictions but no gold."""
    base = _fixture()
    # ``fra`` is a "spurious" prediction: model emits it, no gold samples for it.
    base["fra"] = LanguageMetrics(
        gt_count=0, predictions=4, correct=0, precision=0.0, recall=0.0, f1=0.0
    )
    return base


def test_macro_average_excludes_und_by_default() -> None:
    avg = macro_average(_fixture())
    # No predicted-only non-und langs in the basic fixture, so both views agree.
    assert avg["n_languages_gold"] == 2
    assert avg["n_languages_observed"] == 2
    expected_f1 = (_fixture()["eng"].f1 + _fixture()["deu"].f1) / 2
    assert math.isclose(avg["f1_gold_only"], expected_f1)
    assert math.isclose(avg["f1_observed"], expected_f1)


def test_macro_average_includes_und_when_requested() -> None:
    avg = macro_average(_fixture(), include_und=True)
    # und has gt_count=0, so it counts under "observed" but not "gold_only".
    assert avg["n_languages_gold"] == 2
    assert avg["n_languages_observed"] == 3


def test_macro_average_views_diverge_with_predicted_only_languages() -> None:
    """``fra`` is predicted but has no gold -> gold-only excludes it, observed includes."""
    avg = macro_average(_fixture_with_predicted_only())
    assert avg["n_languages_gold"] == 2  # eng, deu
    assert avg["n_languages_observed"] == 3  # eng, deu, fra (und still excluded by default)
    # The gold-only view is unchanged from the base fixture.
    base_avg = macro_average(_fixture())
    assert math.isclose(avg["f1_gold_only"], base_avg["f1_gold_only"])
    # Observed F1 is dragged down by the spurious ``fra`` (f1=0).
    expected_observed_f1 = (_fixture()["eng"].f1 + _fixture()["deu"].f1 + 0.0) / 3
    assert math.isclose(avg["f1_observed"], expected_observed_f1)
    assert avg["f1_observed"] < avg["f1_gold_only"]


def test_macro_empty_returns_zeros() -> None:
    avg = macro_average({})
    assert avg["f1_gold_only"] == 0.0
    assert avg["f1_observed"] == 0.0
    assert avg["n_languages_gold"] == 0
    assert avg["n_languages_observed"] == 0


def test_micro_average() -> None:
    avg = micro_average(_fixture())
    # In the basic fixture, gold-only and observed differ only by und exclusion
    # (which is the default), so eng+deu carries both views.
    assert avg["n_predictions_gold"] == 15
    assert avg["n_correct_gold"] == 12
    assert avg["n_gold_samples"] == 15
    assert math.isclose(avg["precision_gold_only"], 12 / 15)
    assert math.isclose(avg["recall_gold_only"], 12 / 15)
    # Observed and gold-only collapse here (no non-und predicted-only langs).
    assert math.isclose(avg["precision_observed"], avg["precision_gold_only"])


def test_micro_average_views_diverge_on_spurious_predictions() -> None:
    avg = micro_average(_fixture_with_predicted_only())
    # Adding ``fra`` (4 spurious preds, 0 correct) inflates total_pred for the
    # observed view; gold-only view is unchanged.
    assert avg["n_predictions_gold"] == 15
    assert avg["n_predictions_observed"] == 19
    assert avg["n_correct_observed"] == avg["n_correct_gold"]  # spurious adds no correct
    assert avg["precision_observed"] < avg["precision_gold_only"]
    # Recall denominator is total_gt; spurious lang has gt=0 so unchanged.
    assert math.isclose(avg["recall_observed"], avg["recall_gold_only"])


def test_micro_average_empty() -> None:
    avg = micro_average({})
    assert avg["precision_gold_only"] == 0.0
    assert avg["precision_observed"] == 0.0
    assert avg["n_gold_samples"] == 0


def test_mean_stats_is_macro_alias() -> None:
    assert mean_stats(_fixture()) == macro_average(_fixture())
