"""OpenLID-v2 model wrapper (laurievb/OpenLID-v2)."""

from __future__ import annotations

from commonlid.core.registry import register_model
from commonlid.models._fasttext_base import FastTextHubModel


@register_model
class OpenLIDv2Model(FastTextHubModel):
    model_id = "OpenLID-v2"
    hf_repo_id = "laurievb/OpenLID-v2"
