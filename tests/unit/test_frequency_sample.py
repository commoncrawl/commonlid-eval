"""Behaviour of ``filter_by_language_frequency_and_sample``."""

from __future__ import annotations

import pandas as pd

from commonlid.datasets_tools import filter_by_language_frequency_and_sample


def _toy_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend([{"lang": "eng", "text": f"e{i}"} for i in range(50)])
    rows.extend([{"lang": "deu", "text": f"d{i}"} for i in range(20)])
    rows.extend([{"lang": "fra", "text": f"f{i}"} for i in range(5)])
    return pd.DataFrame(rows)


def test_drops_classes_below_min_count() -> None:
    out = filter_by_language_frequency_and_sample(
        _toy_df(), language_column="lang", min_count=10, n_samples_per_language=5
    )
    assert set(out["lang"].unique()) == {"eng", "deu"}
    assert (out["lang"] == "eng").sum() == 5
    assert (out["lang"] == "deu").sum() == 5


def test_seeded_run_is_deterministic() -> None:
    a = filter_by_language_frequency_and_sample(
        _toy_df(), language_column="lang", min_count=10, n_samples_per_language=5, seed=42
    )
    b = filter_by_language_frequency_and_sample(
        _toy_df(), language_column="lang", min_count=10, n_samples_per_language=5, seed=42
    )
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_samples() -> None:
    a = filter_by_language_frequency_and_sample(
        _toy_df(), language_column="lang", min_count=10, n_samples_per_language=10, seed=1
    )
    b = filter_by_language_frequency_and_sample(
        _toy_df(), language_column="lang", min_count=10, n_samples_per_language=10, seed=2
    )
    assert not a.equals(b)


def test_total_size_equals_n_per_language_times_kept_classes() -> None:
    out = filter_by_language_frequency_and_sample(
        _toy_df(), language_column="lang", min_count=10, n_samples_per_language=4
    )
    assert len(out) == 4 * 2  # eng + deu kept, 4 samples each
