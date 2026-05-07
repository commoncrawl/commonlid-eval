"""Bibles-300 LID benchmark.

The cached HF repo (``commoncrawl/commonlid-cache_bibles_300_iso639_3``) is
private. The dataset itself is built from the ``bibles_with_lang_labels.tsv``
file that can be shared upon request.;
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset

_RAW_PATH_ENV = "COMMONLID_BIBLES_RAW_PATH"
_BUILD_HINT = (
    "Request `bibles_with_lang_labels.tsv` from the authors via email "
    f"and set ${_RAW_PATH_ENV} to its local path "
    "(or call BiblesDataset().build_from_source(source_path=...))"
)


@register_dataset
class BiblesDataset(LIDDataset):
    dataset_id = "bibles_300"
    title = "Bibles-300"
    description = (
        "Bible translations sampled to 300 verses per language across 1000+ language "
        "varieties. Built from a curated TSV the authors share on request; "
        "preprocessed cache shipped privately because the source corpus has mixed "
        "redistribution licenses."
    )
    reference_url = "https://arxiv.org/abs/2601.18026"
    main_score = "macro_f1"
    license_name = "not specified"

    cache_hf_repo = "commoncrawl/commonlid-cache_bibles_300_iso639_3"
    cache_hf_split = "train"
    cache_hf_revision = "dc87d660ebf9f5d61616d3c4beda6e8365c8bae5"
    text_column = "text"
    target_column = "lang_iso639_3"
    is_cache_private = True
    build_source_hint = _BUILD_HINT

    _MIN_COUNT = 1000
    _N_SAMPLES_PER_LANGUAGE = 300
    _BUILD_SEED = 42

    def build_from_source(self, *, source_path: str | os.PathLike[str] | None = None) -> Any:
        """Rebuild the bibles_300 nano benchmark from the raw TSV.

        ``source_path`` defaults to the ``COMMONLID_BIBLES_RAW_PATH`` env var;
        the file must be a TSV with two columns (``language``, ``text``) — the
        format published in the upstream Google Drive folder.
        """
        import pandas as pd
        from datasets import Dataset

        from commonlid.datasets_tools import filter_by_language_frequency_and_sample
        from commonlid.preprocess import convert_and_conform_language

        path = Path(source_path) if source_path is not None else _resolve_raw_path()
        df = pd.read_csv(path, sep="\t", names=["language", "text"])
        df["lang_iso639_3"] = convert_and_conform_language(df["language"], script_separators=("_",))
        df = df.dropna(subset=["lang_iso639_3"])
        df = filter_by_language_frequency_and_sample(
            df,
            language_column="lang_iso639_3",
            min_count=self._MIN_COUNT,
            n_samples_per_language=self._N_SAMPLES_PER_LANGUAGE,
            seed=self._BUILD_SEED,
        )
        return Dataset.from_pandas(df[["language", "text", "lang_iso639_3"]], preserve_index=False)


def _resolve_raw_path() -> Path:
    raw = os.environ.get(_RAW_PATH_ENV)
    if not raw:
        msg = f"BiblesDataset.build_from_source() needs the raw TSV — {_BUILD_HINT}."
        raise FileNotFoundError(msg)
    path = Path(raw)
    if not path.is_file():
        msg = f"${_RAW_PATH_ENV}={raw!r} does not exist or is not a file."
        raise FileNotFoundError(msg)
    return path
