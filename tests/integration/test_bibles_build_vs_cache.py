"""Bibles-300 build-from-source vs private cache parity check.

Skipped by default (``slow`` + ``network``). Run manually:

    uv run pytest tests/integration/test_bibles_build_vs_cache.py \\
        -m "slow and network" -v -s

Requires:
    * Huggingface auth for ``commoncrawl/commonlid-cache_bibles_300_iso639_3``
      (``huggingface-cli login`` or ``HF_TOKEN``).
    * ``COMMONLID_BIBLES_RAW_PATH`` set to a TSV — read from process env first,
      then ``.env`` at the repo root. Two formats are auto-detected:

      - **Raw** (2 columns, ``language`` + ``text``, the upstream
        ``bibles_with_lang_labels.tsv`` shared on request): the build
        pipeline runs end-to-end and is compared to the cache.
      - **Preprocessed** (3 columns, ``language`` + ``text`` +
        ``lang_iso639_3`` — the *output* of the original notebook saved as
        ``bibles_with_lang_labels__300_iso639_3.tsv``): used as a local
        snapshot of what the cache should contain. The build itself is
        skipped because the input is already filtered/sampled.

The notebook's sampler was unseeded, so cached row *content* differs from a
fresh build — but schema, language set, and per-language counts should match.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("datasets")


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_VAR = "COMMONLID_BIBLES_RAW_PATH"
# Where to try resolving relative paths.
_PATH_ROOTS: tuple[Path, ...] = (
    Path.cwd(),
    _REPO_ROOT,
    _REPO_ROOT.parent / "wmdqs",  # sister repo where the data files live
)


def _read_env_var_from_dotenv() -> str | None:
    env_file = _REPO_ROOT / ".env"
    if not env_file.is_file():
        return None
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key.strip() == _ENV_VAR:
            return value.strip().strip("'\"")
    return None


def _resolve(path_str: str) -> Path | None:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p if p.is_file() else None
    for root in _PATH_ROOTS:
        candidate = (root / p).resolve()
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="module")
def raw_path() -> Path:
    path_str = os.environ.get(_ENV_VAR) or _read_env_var_from_dotenv()
    if not path_str:
        pytest.skip(f"{_ENV_VAR} is not set in the environment or .env")
    resolved = _resolve(path_str)
    if resolved is None:
        pytest.skip(
            f"{_ENV_VAR}={path_str!r} could not be resolved against any of: "
            f"{[str(r) for r in _PATH_ROOTS]}"
        )
    return resolved


@pytest.fixture(scope="module")
def built(raw_path: Path) -> Any:
    """Either run the build pipeline (raw input) or load the snapshot (preprocessed).

    A preprocessed snapshot is the original notebook's *output* TSV stored
    locally; treating it as the build result lets us still verify the HF
    cache is faithful to what was preprocessed. If the file is the raw
    upstream TSV (2 columns), the actual ``BiblesDataset.build_from_source``
    pipeline runs.
    """
    import pandas as pd
    from datasets import Dataset

    from commonlid.datasets.bibles import BiblesDataset

    head = pd.read_csv(raw_path, sep="\t", nrows=0)
    columns = set(head.columns)
    if columns >= {"language", "text", "lang_iso639_3"}:
        df = pd.read_csv(raw_path, sep="\t")
        return Dataset.from_pandas(df[["language", "text", "lang_iso639_3"]], preserve_index=False)
    # Otherwise assume raw 2-column TSV without a header.
    return BiblesDataset().build_from_source(source_path=raw_path)


@pytest.fixture(scope="module")
def cached() -> Any:
    from datasets import load_dataset

    from commonlid.datasets.bibles import BiblesDataset

    cls = BiblesDataset
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
    built_langs = set(built["lang_iso639_3"])
    cached_langs = set(cached["lang_iso639_3"])
    only_built = sorted(built_langs - cached_langs)
    only_cached = sorted(cached_langs - built_langs)
    assert built_langs == cached_langs, (
        f"language sets differ.\n  only in build:  {only_built}\n  only in cache: {only_cached}"
    )


@pytest.mark.slow
@pytest.mark.network
def test_per_language_counts_match(built: Any, cached: Any) -> None:
    built_counts = Counter(built["lang_iso639_3"])
    cached_counts = Counter(cached["lang_iso639_3"])
    diffs = {
        lang: (built_counts.get(lang, 0), cached_counts.get(lang, 0))
        for lang in set(built_counts) | set(cached_counts)
        if built_counts.get(lang, 0) != cached_counts.get(lang, 0)
    }
    assert not diffs, f"per-language counts differ: {diffs}"


@pytest.mark.slow
@pytest.mark.network
def test_each_language_has_exactly_300_samples(built: Any) -> None:
    counts = Counter(built["lang_iso639_3"])
    off = {lang: n for lang, n in counts.items() if n != 300}
    assert not off, f"languages without exactly 300 samples: {off}"
