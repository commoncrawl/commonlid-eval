"""CommonLingua: PleIAs' byte-level LID model (PleIAs/CommonLingua).

Requires the ``commonlid[commonlingua]`` extra (torch). The checkpoint
embeds its own ``lang2idx`` map, so no separate metadata file is fetched.
Device selection mirrors AfroLID: MPS > CUDA > CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model

if TYPE_CHECKING:
    import torch


@register_model
class CommonLinguaModel(LIDModel):
    model_id = "commonlingua"
    # Byte-level model: casing carries strong language signal, so we feed
    # raw UTF-8 and skip the OpenLID normer (which lowercases everything).
    requires_preprocessing: ClassVar[bool] = False

    _REPO_ID: ClassVar[str] = "PleIAs/CommonLingua"
    _CHECKPOINT_FILENAME: ClassVar[str] = "model.pt"
    _INTERNAL_BATCH: ClassVar[int] = 256

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._idx2lang: dict[int, str] | None = None
        self._max_len: int | None = None
        self._device: str | None = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
        except ImportError as exc:
            msg = "CommonLingua requires torch. Install with: pip install 'commonlid[commonlingua]'"
            raise ImportError(msg) from exc

        from huggingface_hub import hf_hub_download

        from commonlid.vendor.commonlingua.model import CONFIGS, ByteHybrid

        ckpt_path = hf_hub_download(repo_id=self._REPO_ID, filename=self._CHECKPOINT_FILENAME)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = ByteHybrid(  # type: ignore[no-untyped-call]
            num_classes=ckpt["num_classes"],
            max_len=ckpt["max_len"],
            **CONFIGS[ckpt["config"]],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)

        self._model = model
        self._idx2lang = {v: k for k, v in ckpt["lang2idx"].items()}
        self._max_len = int(ckpt["max_len"])
        self._device = device
        super().load()

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        import numpy as np
        import torch

        assert self._max_len is not None
        out = np.full((len(texts), self._max_len), 256, dtype=np.int64)
        for i, t in enumerate(texts):
            raw = t.encode("utf-8", errors="replace")[: self._max_len]
            if raw:
                out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        return torch.from_numpy(out)

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        import torch

        if not self._loaded:
            self.load()
        assert self._idx2lang is not None
        assert self._device is not None

        results: list[str | None] = []
        for start in range(0, len(texts), self._INTERNAL_BATCH):
            chunk = list(texts[start : start + self._INTERNAL_BATCH])
            batch = self._encode(chunk).to(self._device)
            with torch.no_grad():
                logits = self._model(batch)
                pred_idx = logits.argmax(dim=-1).cpu().tolist()
            results.extend(self._idx2lang[int(i)] for i in pred_idx)
        return results

    def discover_supported_languages(self) -> frozenset[str]:
        """Return every ISO 639-3 code in the model's ``lang2idx`` map."""
        if not self._loaded:
            self.load()
        assert self._idx2lang is not None
        codes: set[str] = set()
        for code in self._idx2lang.values():
            conformed = self._conform(code)
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
