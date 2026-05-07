"""AfroLID model wrapper (UBC-NLP/afrolid_1.5).

Requires the ``commonlid[afrolid]`` extra (transformers + torch). Device
selection mirrors the original research code: MPS if available, else CUDA,
else CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model


@register_model
class AfroLIDModel(LIDModel):
    model_id = "AfroLID"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            msg = (
                "AfroLID requires torch and transformers. "
                "Install with: pip install 'commonlid[afrolid]'"
            )
            raise ImportError(msg) from exc

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._pipeline = pipeline("text-classification", model="UBC-NLP/afrolid_1.5", device=device)
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        assert self._pipeline is not None  # load() has run
        results = self._pipeline(list(texts), truncation=True, max_length=512)
        out: list[str | None] = []
        for entry in results:
            label = entry["label"]
            out.append(None if label == "nan_lang" else label)
        return out

    def discover_supported_languages(self) -> frozenset[str]:
        """Read labels from the text-classification pipeline's model config."""
        if self._pipeline is None:
            self.load()
        assert self._pipeline is not None
        id2label = self._pipeline.model.config.id2label
        codes: set[str] = set()
        for label in id2label.values():
            if label == "nan_lang":
                continue
            conformed = self._conform(label)
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
