from __future__ import annotations

from collections import Counter

from commonlid.datasets_tools.stratified_sample import (
    stratified_sample_with_minimum_per_class,
)


def test_minimum_quota_and_dropped_classes() -> None:
    # Languages: eng x 40, deu x 20, rare x 2
    labels = ["eng"] * 40 + ["deu"] * 20 + ["rare"] * 2
    result = stratified_sample_with_minimum_per_class(labels, max_size=30, min_size=5, seed=42)
    assert "rare" in result.dropped_classes
    counts = Counter(labels[i] for i in result.selected)
    assert counts["rare"] == 0
    # eng/deu each receive at least min_size samples.
    assert counts["eng"] >= 5
    assert counts["deu"] >= 5
    # Total is approximately max_size + min_size * n_kept_classes.
    assert 30 <= len(result.selected) <= 30 + 5 * 2


def test_none_labels_are_treated_as_a_class() -> None:
    """``None`` is a valid class key (mirrors the original generate_small_version).

    Callers that don't want None-labeled rows should pre-filter the labels.
    """
    labels = ["eng"] * 10 + ["deu"] * 10 + [None, None]
    # min_size=2 keeps the two None-labeled rows alongside eng/deu.
    result = stratified_sample_with_minimum_per_class(labels, max_size=10, min_size=2, seed=1)
    selected_labels = [labels[i] for i in result.selected]
    assert None in selected_labels


def test_none_labels_dropped_when_below_min_size() -> None:
    labels = ["eng"] * 10 + ["deu"] * 10 + [None]  # only 1 None < min_size=2
    result = stratified_sample_with_minimum_per_class(labels, max_size=10, min_size=2, seed=1)
    assert None in result.dropped_classes
    selected_labels = [labels[i] for i in result.selected]
    assert None not in selected_labels


def test_deterministic_given_seed() -> None:
    labels = ["eng"] * 30 + ["deu"] * 30
    a = stratified_sample_with_minimum_per_class(labels, max_size=10, min_size=2, seed=42)
    b = stratified_sample_with_minimum_per_class(labels, max_size=10, min_size=2, seed=42)
    assert a.selected == b.selected
