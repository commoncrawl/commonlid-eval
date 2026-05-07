"""Social media LID benchmark (commoncrawl/commonlid-cache_social-media_300_iso639_3)."""

from __future__ import annotations

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset


@register_dataset
class SocialMediaDataset(LIDDataset):
    dataset_id = "social_media_300"
    title = "Social Media-300"
    description = (
        "Social-media posts sampled to 300 per language across ~60 "
        "language varieties. The text is shipped privately because the original "
        "platform terms-of-service do not permit public redistribution."
    )
    reference_url = "https://arxiv.org/abs/2601.18026"
    main_score = "macro_f1"
    license_name = "not specified"

    cache_hf_repo = "commoncrawl/commonlid-cache_social-media_300_iso639_3"
    cache_hf_split = "train"
    cache_hf_revision = "ea94443fc737b37c762d9d654a596072f43f20aa"
    text_column = "text"
    target_column = "lang_iso639_3"
    is_cache_private = True
