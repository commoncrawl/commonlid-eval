"""FLORES+ dev split (openlanguagedata/flores_plus)."""

from __future__ import annotations

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset


@register_dataset
class FloresDevDataset(LIDDataset):
    dataset_id = "flores_dev"
    title = "FLORES+"
    description = (
        "FLORES+ — a multilingual evaluation benchmark with parallel sentences in "
        "200+ languages. The ``dev`` split is reused for language identification by "
        "joining each sentence with its labelled language code."
    )
    reference_url = "https://huggingface.co/datasets/openlanguagedata/flores_plus"
    main_score = "macro_f1"
    license_name = "cc-by-sa-4.0"
    license_url = "https://creativecommons.org/licenses/by-sa/4.0/"

    source_hf_repo = "openlanguagedata/flores_plus"
    source_hf_split = "dev"
    source_hf_revision = "6a4c3d50537cc73d24777cde59fa6040234c1906"
    text_column = "text"
    target_column = "iso_639_3"
