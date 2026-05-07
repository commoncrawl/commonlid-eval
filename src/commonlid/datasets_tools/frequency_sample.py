"""Frequency-filter + per-class sampler used to build the *_300 nano benchmarks.

Ported from the research notebooks' ``filter_by_language_frequency_and_sample``
(see ``wmdqs/eval/langid_datasets.py``). Two behaviours preserved verbatim:

1. Drop every class whose row count is below ``min_count``.
2. From each surviving class, sample exactly ``n_samples_per_language`` rows.

The original notebook calls ``pd.DataFrame.sample(n=...)`` without a seed, so
repeated runs produced different row content. This port adds a ``seed``
keyword (default 42) so reproductions are deterministic.
"""

from __future__ import annotations

import pandas as pd


def filter_by_language_frequency_and_sample(
    df: pd.DataFrame,
    *,
    language_column: str,
    min_count: int,
    n_samples_per_language: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Drop low-frequency classes, then sample exactly N rows per surviving class."""
    counts = df[language_column].value_counts()
    keep = sorted(counts[counts >= min_count].index)
    pieces: list[pd.DataFrame] = []
    for lang in keep:
        group = df[df[language_column] == lang]
        pieces.append(group.sample(n=n_samples_per_language, random_state=seed))
    if not pieces:
        return df.iloc[0:0].reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)
