"""The CommonLID benchmark (commoncrawl/CommonLID)."""

from __future__ import annotations

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset


@register_dataset
class CommonLIDDataset(LIDDataset):
    dataset_id = "commonlid"
    title = "CommonLID"
    description = (
        "Common Crawl's language identification benchmark, sampled from real-world "
        "web text and human-validated across hundreds of language varieties."
    )
    reference_url = "https://huggingface.co/datasets/commoncrawl/CommonLID"
    main_score = "macro_f1"
    license_name = "common-crawl-tou"
    license_url = "https://commoncrawl.org/terms-of-use"

    source_hf_repo = "commoncrawl/CommonLID"
    source_hf_split = "test"
    source_hf_revision = "1ab8feb9fa051f7ad60e80f8592fac0d973ead9b"
    text_column = "text"
    target_column = "tag"
