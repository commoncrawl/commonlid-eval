"""SmolSent build-from-source vs private cache parity check.

Skipped by default (``slow`` + ``network``). Run manually:

    uv run pytest tests/integration/test_smolsent_build_vs_cache.py \\
        -m "slow and network" -v -s

Requires Huggingface auth for ``commoncrawl/commonlid-cache_smolsent_300_iso639_3``
(``huggingface-cli login`` or ``HF_TOKEN``). The build path itself only needs
public access to ``google/smol``.

The original notebook's sampler was unseeded, so the cached row *content*
can differ from a fresh build — but schema, language set, and per-language
counts should match.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

pytest.importorskip("datasets")


@pytest.fixture(scope="module")
def built() -> Any:
    from commonlid.datasets.smolsent import SmolSentDataset

    return SmolSentDataset().build_from_source()


@pytest.fixture(scope="module")
def cached() -> Any:
    from datasets import load_dataset

    from commonlid.datasets.smolsent import SmolSentDataset

    cls = SmolSentDataset
    return load_dataset(
        cls.cache_hf_repo,
        cls.cache_hf_config,
        split=cls.cache_hf_split,
        revision=cls.cache_hf_revision,
    )


@pytest.mark.slow
@pytest.mark.network
def test_columns_match(built: Any, cached: Any) -> None:
    assert sorted(built.column_names) == sorted(cached.column_names), (
        f"column names differ: built={built.column_names} cached={cached.column_names}"
    )


@pytest.mark.slow
@pytest.mark.network
def test_total_row_count_matches(built: Any, cached: Any) -> None:
    assert len(built) == len(cached), f"total rows differ: built={len(built)} cached={len(cached)}"


@pytest.mark.slow
@pytest.mark.network
def test_language_sets_match(built: Any, cached: Any) -> None:
    built_langs = set(built["tl_iso693_3"])
    cached_langs = set(cached["tl_iso693_3"])
    only_built = sorted(built_langs - cached_langs)
    only_cached = sorted(cached_langs - built_langs)
    assert built_langs == cached_langs, (
        f"language sets differ.\n  only in build:  {only_built}\n  only in cache: {only_cached}"
    )


@pytest.mark.slow
@pytest.mark.network
def test_per_language_counts_match(built: Any, cached: Any) -> None:
    built_counts = Counter(built["tl_iso693_3"])
    cached_counts = Counter(cached["tl_iso693_3"])
    diffs = {
        lang: (built_counts.get(lang, 0), cached_counts.get(lang, 0))
        for lang in set(built_counts) | set(cached_counts)
        if built_counts.get(lang, 0) != cached_counts.get(lang, 0)
    }
    assert not diffs, f"per-language counts differ: {diffs}"


@pytest.mark.slow
@pytest.mark.network
def test_each_language_has_exactly_300_samples(built: Any) -> None:
    counts = Counter(built["tl_iso693_3"])
    off = {lang: n for lang, n in counts.items() if n != 300}
    assert not off, f"languages without exactly 300 samples: {off}"


@pytest.mark.slow
@pytest.mark.network
def test_spot_check_rows_come_from_upstream(built: Any) -> None:
    """Every (tl, id, trg) triple in the build should exist in raw ``google/smol``.

    Validates the build pipeline only ever passes raw samples through —
    nothing fabricated, no cross-row leakage.
    """
    from datasets import load_dataset

    from commonlid.datasets.smolsent import SmolSentDataset

    cls = SmolSentDataset
    raw = load_dataset(
        cls.source_hf_repo,
        data_files=cls._UPSTREAM_HF_PATTERN,
        split=cls.source_hf_split,
        revision=cls.source_hf_revision,
    )
    raw_keys = {(row["tl"], row["id"], row["trg"]) for row in raw}

    sample_idxs = [0, len(built) // 4, len(built) // 2, 3 * len(built) // 4, len(built) - 1]
    missing: list[tuple[str, int, str]] = []
    for i in sample_idxs:
        row = built[i]
        key = (row["tl"], row["id"], row["trg"])
        if key not in raw_keys:
            missing.append(key)
    assert not missing, f"spot-checked rows not found upstream: {missing}"
