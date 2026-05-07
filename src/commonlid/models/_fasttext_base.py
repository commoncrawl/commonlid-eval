"""Shared plumbing for fasttext-based LID models (GlotLID, OpenLID-v2, fasttext-ft)."""

from __future__ import annotations

from collections.abc import Sequence

import fasttext
from huggingface_hub import hf_hub_download

from commonlid.core.lid_model import LIDModel


class FastTextHubModel(LIDModel):
    """Base class for HF-hosted fasttext LID models.

    Subclasses set :attr:`hf_repo_id` and :attr:`hf_filename` (default
    ``"model.bin"``). The loaded fasttext model is stashed on the instance.
    """

    hf_repo_id: str
    hf_filename: str = "model.bin"

    def __init__(self) -> None:
        super().__init__()
        self._ft: fasttext.FastText._FastText | None = None

    def load(self) -> None:
        if self._loaded:
            return
        path = hf_hub_download(repo_id=self.hf_repo_id, filename=self.hf_filename)
        self._ft = fasttext.load_model(path)
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        assert self._ft is not None  # load() has run
        labels = self._predict_labels(list(texts))
        out: list[str | None] = []
        for results in labels:
            label = results[0]
            if not label.startswith("__label__"):
                msg = f"Unexpected fasttext label format from {self.model_id}: {label!r}"
                raise ValueError(msg)
            code = label.split("__")[2].split("_")[0]
            out.append(code)
        return out

    def _predict_labels(self, texts: list[str]) -> list[list[str]]:
        """Return ``[[label, ...], ...]`` from either ``fasttext-wheel`` or ``fasttext-predict``.

        ``fasttext-wheel`` returns ``(labels, probs)`` from ``predict``;
        ``fasttext-predict`` returns a plain ``labels`` list. The stock
        ``fasttext.FastText._FastText.predict`` wrapper in the latter also
        happens to be broken on Python 3.13 (it tries to unpack a 1-tuple
        as 2 values), so we sidestep it and call ``multilinePredict``
        directly when available.
        """
        assert self._ft is not None
        mp = getattr(self._ft.f, "multilinePredict", None)
        if mp is not None:
            prepared = [t if t.endswith("\n") else t + "\n" for t in texts]
            result = mp(prepared, 1, 0.0, "strict")
            # fasttext-wheel: ([[labels], ...], [[probs], ...])  — a 2-tuple of lists.
            # fasttext-predict: [[labels], ...]                   — a bare list of label lists.
            if isinstance(result, tuple) and len(result) == 2:
                return list(result[0])
            return list(result)

        # Fallback: call the wrapper and unpack whatever shape it returns.
        predicted = self._ft.predict(list(texts))
        if isinstance(predicted, tuple) and len(predicted) == 2:
            return list(predicted[0])
        return list(predicted)

    def discover_supported_languages(self) -> frozenset[str]:
        """Enumerate every ``__label__{code}`` exposed by the loaded fasttext model."""
        if self._ft is None:
            self.load()
        assert self._ft is not None
        codes: set[str] = set()
        for label in self._ft.get_labels():
            if not label.startswith("__label__"):
                continue
            raw = label.split("__")[2].split("_")[0]
            conformed = self._conform(raw)
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
