"""Nano build-from-source vs private cache parity check.

Skipped by default (``slow`` + ``network``). Run manually:

    uv run pytest tests/integration/test_nano_build_vs_cache.py \\
        -m "slow and network" -v -s

Requires Huggingface auth for ``commoncrawl/commonlid-cache_nano`` plus
whatever each parent dataset's ``build_from_source()`` needs (e.g.
``COMMONLID_BIBLES_RAW_PATH`` for bibles).

The original ``generate_nano_datasets.generate_small_version`` was unseeded,
so cached row *content* may differ from a fresh build — but schema, language
set, and per-language counts should match.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

pytest.importorskip("datasets")


_NANO_CLASSES = pytest.mark.parametrize(
    "nano_cls_path",
    [
        "commonlid.datasets.nano:CommonLIDDatasetNano",
        "commonlid.datasets.nano:FloresDevDatasetNano",
        "commonlid.datasets.nano:UDHRDatasetNano",
        "commonlid.datasets.nano:SmolSentDatasetNano",
        "commonlid.datasets.nano:BiblesDatasetNano",
        "commonlid.datasets.nano:SocialMediaDatasetNano",
    ],
)


def _resolve(nano_cls_path: str) -> type:
    module_name, _, attr = nano_cls_path.partition(":")
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def _load_cache(cls: type) -> Any:
    from datasets import load_dataset

    return load_dataset(
        cls.cache_hf_repo,
        split=cls.cache_hf_split,
        revision=cls.cache_hf_revision,
    )


@pytest.mark.slow
@pytest.mark.network
@_NANO_CLASSES
def test_columns_match(nano_cls_path: str) -> None:
    cls = _resolve(nano_cls_path)
    built = cls().build_from_source()
    cached = _load_cache(cls)
    assert sorted(built.column_names) == sorted(cached.column_names), (
        f"{cls.__name__}: column names differ.\n"
        f"  built:  {built.column_names}\n  cached: {cached.column_names}"
    )


@pytest.mark.slow
@pytest.mark.network
@_NANO_CLASSES
def test_total_row_count_matches(nano_cls_path: str) -> None:
    cls = _resolve(nano_cls_path)
    built = cls().build_from_source()
    cached = _load_cache(cls)
    assert len(built) == len(cached), (
        f"{cls.__name__}: total rows differ — built={len(built)} cached={len(cached)}"
    )


@pytest.mark.slow
@pytest.mark.network
@_NANO_CLASSES
def test_language_sets_match(nano_cls_path: str) -> None:
    cls = _resolve(nano_cls_path)
    built = cls().build_from_source()
    cached = _load_cache(cls)
    built_langs = set(built["language_iso639_3"])
    cached_langs = set(cached["language_iso639_3"])
    only_built = sorted(built_langs - cached_langs)
    only_cached = sorted(cached_langs - built_langs)
    assert built_langs == cached_langs, (
        f"{cls.__name__}: language sets differ.\n"
        f"  only in build:  {only_built}\n  only in cache: {only_cached}"
    )


@pytest.mark.slow
@pytest.mark.network
@_NANO_CLASSES
def test_per_language_counts_match(nano_cls_path: str) -> None:
    cls = _resolve(nano_cls_path)
    built = cls().build_from_source()
    cached = _load_cache(cls)
    built_counts = Counter(built["language_iso639_3"])
    cached_counts = Counter(cached["language_iso639_3"])
    diffs = {
        lang: (built_counts.get(lang, 0), cached_counts.get(lang, 0))
        for lang in set(built_counts) | set(cached_counts)
        if built_counts.get(lang, 0) != cached_counts.get(lang, 0)
    }
    assert not diffs, f"{cls.__name__}: per-language counts differ: {diffs}"
