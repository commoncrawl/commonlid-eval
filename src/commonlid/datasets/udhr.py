"""UDHR-LID test split (cis-lmu/udhr-lid)."""

from __future__ import annotations

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset


@register_dataset
class UDHRDataset(LIDDataset):
    dataset_id = "udhr"
    title = "UDHR-LID"
    description = (
        "Universal Declaration of Human Rights LID benchmark — sentence-level "
        "translations of the UDHR across 400+ language varieties, packaged for "
        "language identification by CIS-LMU."
    )
    reference_url = "https://huggingface.co/datasets/cis-lmu/udhr-lid"
    main_score = "macro_f1"
    license_name = "apache-2.0"
    license_url = "https://www.apache.org/licenses/LICENSE-2.0"

    source_hf_repo = "cis-lmu/udhr-lid"
    source_hf_split = "test"
    source_hf_revision = "6908db2a27c296158da7e69782d15df911652184"
    text_column = "sentence"
    target_column = "iso639-3"
