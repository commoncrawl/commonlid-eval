"""pyfranc (franc) model wrapper."""

from __future__ import annotations

from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model


@register_model
class PyfrancModel(LIDModel):
    model_id = "pyfranc"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        from pyfranc import franc

        out: list[str | None] = []
        for text in texts:
            detected = franc.lang_detect(text)
            # `lang_detect` returns a list of (code, score); take the top one.
            out.append(detected[0][0])
        return out

    def discover_supported_languages(self) -> frozenset[str]:
        """Return every language code in ``franc.data`` (keyed by script)."""
        from pyfranc import franc

        codes: set[str] = set()
        for script_langs in franc.data.values():
            for code in script_langs:
                conformed = self._conform(code)
                if conformed is not None:
                    codes.add(conformed)
        return frozenset(codes)
