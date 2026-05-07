"""GlotLID model wrapper (cis-lmu/glotlid)."""

from __future__ import annotations

from commonlid.core.registry import register_model
from commonlid.models._fasttext_base import FastTextHubModel


@register_model
class GlotLIDModel(FastTextHubModel):
    model_id = "GlotLID"
    hf_repo_id = "cis-lmu/glotlid"
