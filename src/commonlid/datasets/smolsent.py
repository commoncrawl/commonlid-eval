"""SmolSent-300 LID benchmark.

The cached HF repo (``commoncrawl/commonlid-cache_smolsent_300_iso639_3``) is
private; the underlying data is the public ``google/smol`` dataset (the
``smolsent`` config). :meth:`SmolSentDataset.build_from_source` reproduces
the preprocessing performed in ``dataset_preprocess_smolsent.ipynb``.
"""

from __future__ import annotations

from typing import Any

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset

_BUILD_HINT = (
    "rebuild from public `google/smol` (smolsent config) via SmolSentDataset().build_from_source()"
)


@register_dataset
class SmolSentDataset(LIDDataset):
    dataset_id = "smolsent_300"
    title = "SmolSent-300"
    description = (
        "Stratified 300-per-language sample of Google's SMOL sentence-level "
        "translation dataset. The cache mirrors the artifact shipped with the "
        "CommonLID paper; rebuildable from the public ``google/smol`` source."
    )
    reference_url = "https://huggingface.co/datasets/google/smol"
    main_score = "macro_f1"
    license_name = "cc-by-4.0"
    license_url = "https://creativecommons.org/licenses/by/4.0/"

    source_hf_repo = "google/smol"
    source_hf_split = "train"
    source_hf_revision = "43939e46fd7a708726df6ed9a23a980ab9806545"
    cache_hf_repo = "commoncrawl/commonlid-cache_smolsent_300_iso639_3"
    cache_hf_split = "train"
    cache_hf_revision = "60b4e8fd20a9f6d37cdf3de40053ce2f39f947f9"
    text_column = "trg"
    target_column = "tl_iso693_3"
    is_cache_private = True
    build_source_hint = _BUILD_HINT

    _UPSTREAM_HF_PATTERN = "smolsent/*.jsonl"
    _MIN_COUNT = 300
    _N_SAMPLES_PER_LANGUAGE = 300
    _BUILD_SEED = 42

    def build_from_source(self) -> Any:
        """Rebuild smolsent_300 from the public ``google/smol`` HF dataset."""
        from datasets import Dataset, load_dataset

        from commonlid.datasets_tools import filter_by_language_frequency_and_sample
        from commonlid.preprocess import convert_and_conform_language

        smolsent = load_dataset(
            self.source_hf_repo,
            data_files=self._UPSTREAM_HF_PATTERN,
            split=self.source_hf_split,
            revision=self.source_hf_revision,
        )
        df = smolsent.to_pandas()
        df["tl_iso693_3"] = convert_and_conform_language(df["tl"], script_separators=("_",))
        df = filter_by_language_frequency_and_sample(
            df,
            language_column="tl_iso693_3",
            min_count=self._MIN_COUNT,
            n_samples_per_language=self._N_SAMPLES_PER_LANGUAGE,
            seed=self._BUILD_SEED,
        )
        return Dataset.from_pandas(df, preserve_index=False)
