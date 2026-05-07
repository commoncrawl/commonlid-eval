"""Nano variants of the shipped LID benchmarks.

A "nano" subset of dataset *X* is a stratified sample of *X* — each surviving
language contributes a guaranteed minimum (``_MIN_SIZE=5``) plus a
proportional share of ``_MAX_SIZE=1000`` distributed across the remaining
samples. This matches the algorithm in
``wmdqs/eval/generate_nano_datasets.py:generate_small_version`` byte-for-byte
(via the port in ``commonlid.datasets_tools.stratified_sample``).

Each nano lives in its own HF cache repo
(``commoncrawl/commonlid-cache_<base_id>_nano``) so visibility (public vs
private) can track the parent dataset independently. All caches use the
schema ``(index, text, language_iso639_3)`` and ``split="test"``. The
``index`` column records each row's position in the parent so the slice can
be replayed against the full benchmark when needed.

If the cache is unreachable, each nano falls back to ``build_from_source()``,
which loads the parent dataset (via the parent class's own ``load()`` chain)
and re-runs the sampler. The build is deterministic via ``_BUILD_SEED=42``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset
from commonlid.datasets.bibles import BiblesDataset
from commonlid.datasets.commonlid import CommonLIDDataset
from commonlid.datasets.flores_dev import FloresDevDataset
from commonlid.datasets.smolsent import SmolSentDataset
from commonlid.datasets.social_media import SocialMediaDataset
from commonlid.datasets.udhr import UDHRDataset

_BUILD_HINT = (
    "rebuild from the parent dataset's source via "
    "<NanoClass>().build_from_source() (which itself calls the parent's load() chain)"
)


class _NanoLIDDataset(LIDDataset):
    """Shared scaffolding for every nano subset.

    Subclasses set ``parent_dataset_cls`` and ``cache_hf_repo`` (one HF repo
    per nano so visibility can track the parent dataset). The sampler config
    (``_MAX_SIZE``, ``_MIN_SIZE``, ``_BUILD_SEED``) mirrors the original
    notebook and is overridable per subclass if needed.
    """

    parent_dataset_cls: ClassVar[type[LIDDataset]]

    cache_hf_split = "test"
    is_cache_private = True
    build_source_hint = _BUILD_HINT

    text_column = "text"
    target_column = "language_iso639_3"

    _MAX_SIZE: ClassVar[int] = 1000
    _MIN_SIZE: ClassVar[int] = 5
    _BUILD_SEED: ClassVar[int] = 42

    def build_from_source(self) -> Any:
        """Re-derive the nano subset from the parent dataset's full version."""
        from datasets import Dataset

        from commonlid.datasets_tools import stratified_sample_with_minimum_per_class

        parent = self.parent_dataset_cls()
        parent_ds = parent.load()

        labels = list(parent_ds[parent.target_column])
        sampled = stratified_sample_with_minimum_per_class(
            labels,
            max_size=self._MAX_SIZE,
            min_size=self._MIN_SIZE,
            seed=self._BUILD_SEED,
        )
        rows = parent_ds.select(sampled.selected)
        return Dataset.from_dict({
            "index": sampled.selected,
            "text": rows[parent.text_column],
            "language_iso639_3": rows[parent.target_column],
        })


_NANO_DESC_SUFFIX = (
    " Nano slice — stratified sample (max 1000 + min 5 per language) of the "
    "parent benchmark, with the schema normalised to (index, text, "
    "language_iso639_3)."
)


@register_dataset
class BiblesDatasetNano(_NanoLIDDataset):
    dataset_id = "bibles_300_nano"
    parent_dataset_cls = BiblesDataset
    title = "Bibles-300 (nano)"
    description = (BiblesDataset.description or "") + _NANO_DESC_SUFFIX
    reference_url = BiblesDataset.reference_url
    license_name = BiblesDataset.license_name
    license_url = BiblesDataset.license_url

    cache_hf_repo = "commoncrawl/commonlid-cache_bibles_300_nano"
    cache_hf_revision = "28eadc7351dcee7b92fc7c8b68d6097363aebd1f"


@register_dataset
class CommonLIDDatasetNano(_NanoLIDDataset):
    dataset_id = "commonlid_nano"
    parent_dataset_cls = CommonLIDDataset
    title = "CommonLID (nano)"
    description = (CommonLIDDataset.description or "") + _NANO_DESC_SUFFIX
    reference_url = CommonLIDDataset.reference_url
    license_name = CommonLIDDataset.license_name
    license_url = CommonLIDDataset.license_url

    cache_hf_repo = "commoncrawl/commonlid-cache_commonlid_nano"
    cache_hf_revision = "6dd634be70beaad9623d02cc3e4e4faa008bef9d"

    is_cache_private = False


@register_dataset
class FloresDevDatasetNano(_NanoLIDDataset):
    dataset_id = "flores_dev_nano"
    parent_dataset_cls = FloresDevDataset
    title = "FLORES+ (nano)"
    description = (FloresDevDataset.description or "") + _NANO_DESC_SUFFIX
    reference_url = FloresDevDataset.reference_url
    license_name = FloresDevDataset.license_name
    license_url = FloresDevDataset.license_url

    cache_hf_repo = "commoncrawl/commonlid-cache_flores_dev_nano"
    cache_hf_revision = "cccd03009f539f3dbfbc0c558587b894d9895a1f"


@register_dataset
class SmolSentDatasetNano(_NanoLIDDataset):
    dataset_id = "smolsent_300_nano"
    parent_dataset_cls = SmolSentDataset
    title = "SmolSent-300 (nano)"
    description = (SmolSentDataset.description or "") + _NANO_DESC_SUFFIX
    reference_url = SmolSentDataset.reference_url
    license_name = SmolSentDataset.license_name
    license_url = SmolSentDataset.license_url

    cache_hf_repo = "commoncrawl/commonlid-cache_smolsent_300_nano"
    cache_hf_revision = "8ef2a59ba070fa512e13f70e627062bfbcf5a92f"


@register_dataset
class UDHRDatasetNano(_NanoLIDDataset):
    dataset_id = "udhr_nano"
    parent_dataset_cls = UDHRDataset
    title = "UDHR-LID (nano)"
    description = (UDHRDataset.description or "") + _NANO_DESC_SUFFIX
    reference_url = UDHRDataset.reference_url
    license_name = UDHRDataset.license_name
    license_url = UDHRDataset.license_url

    cache_hf_repo = "commoncrawl/commonlid-cache_udhr_nano"
    cache_hf_revision = "78235af5d72df01e1d1f83617de9424d56595cb5"


@register_dataset
class SocialMediaDatasetNano(_NanoLIDDataset):
    dataset_id = "social_media_300_nano"
    parent_dataset_cls = SocialMediaDataset
    title = "Social Media-300 (nano)"
    description = (SocialMediaDataset.description or "") + _NANO_DESC_SUFFIX
    reference_url = SocialMediaDataset.reference_url
    license_name = SocialMediaDataset.license_name
    license_url = SocialMediaDataset.license_url

    cache_hf_repo = "commoncrawl/commonlid-cache_social_media_300_nano"
    cache_hf_revision = "aaddae8c1a8fbb242a1e3d6a4173bc0e08f6dff3"
