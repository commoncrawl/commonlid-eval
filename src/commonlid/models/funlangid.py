"""Wrapper around the vendored FunLangID classifier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from commonlid.core.lid_model import LIDModel, LIDPrediction
from commonlid.core.registry import register_model


@register_model
class FunLangIDModel(LIDModel):
    model_id = "funlangid"

    def __init__(self) -> None:
        super().__init__()
        self._classifier: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        from commonlid.vendor.fun_langid import FunLangID

        self._classifier = FunLangID()
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
        assert self._classifier is not None  # load() has run
        out: list[LIDPrediction] = []
        for text in texts:
            # `predict` is `predict_top` with the score kept: the vendored
            # classifier scores each candidate as the fraction of the text's
            # 4-grams that its lexicon covers, so scores do not sum to 1.
            ranked = self._classifier.predict(text, 1)
            if not ranked or not ranked[0][1]:
                out.append(LIDPrediction(None, None))
                continue
            raw, score = ranked[0]
            # Output is BCP-47 like 'lang-script'.
            code = raw.split("-")[0]
            out.append(LIDPrediction(None if code == "und" else code, float(score)))
        return out

    def discover_supported_languages(self) -> frozenset[str]:
        """Enumerate every language that appears in :attr:`FunLangID.lex_dict`."""
        if self._classifier is None:
            self.load()
        assert self._classifier is not None
        raw_langs = {lang for langs in self._classifier.lex_dict.values() for lang in langs}
        codes: set[str] = set()
        for raw in raw_langs:
            conformed = self._conform(raw.split("-")[0])
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
