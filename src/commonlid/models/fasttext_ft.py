"""Facebook fasttext-language-identification wrapper."""

from __future__ import annotations

from commonlid.core.registry import register_model
from commonlid.models._fasttext_base import FastTextHubModel


@register_model
class FasttextLIDModel(FastTextHubModel):
    model_id = "fasttext"
    hf_repo_id = "facebook/fasttext-language-identification"
