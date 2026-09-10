"""py3langid wrapper: does its own normalization; zxx and sub-threshold predictions abstain."""

from __future__ import annotations

from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel, LIDPrediction
from commonlid.core.registry import register_model

# codes the generic 639-1 -> 639-3 conformation would not reach
_OUT_MAP = {"ar": "arb", "sw": "swh", "az": "azj", "or": "ory", "bcl": "bik"}
_MIN_CONFIDENCE = 0.25  # best micro+macro F1 on commonlid


@register_model
class Py3LangIDModel(LIDModel):
    model_id = "py3langid"
    requires_preprocessing = False

    def load(self) -> None:
        from py3langid.langid import MODEL_FILE, LanguageIdentifier

        self._identifier = LanguageIdentifier.from_model_file(
            MODEL_FILE, norm_probs=True, min_confidence=_MIN_CONFIDENCE
        )
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
        out: list[LIDPrediction] = []
        for text in texts:
            code, score = self._identifier.classify(text)
            label = None if code in ("zxx", "und") else _OUT_MAP.get(code, code)
            out.append(LIDPrediction(label, float(score)))
        return out

    def discover_supported_languages(self) -> frozenset[str]:
        self.load()
        codes: set[str] = set()
        for code in self._identifier.labels:
            if code == "zxx":
                continue
            conformed = self._conform(_OUT_MAP.get(code, code))
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
