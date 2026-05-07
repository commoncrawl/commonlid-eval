"""pycld2 wrapper.

CLD2's output format is a ``(isReliable, bytes_found, details)`` tuple; the
first ``details`` entry is ``(name, code, percent, score)``. We only take the
two-letter code, strip any script suffix, and map the sentinel "unknown"
codes (``un``, ``xx``, ``zzp``) to ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model

_UNKNOWN_CODES = frozenset({"un", "xx", "zzp"})


def _detect_one(text: str) -> str | None:
    import pycld2 as cld2

    _, _, details = cld2.detect(text, isPlainText=True)
    raw_code: str = details[0][1].split("-")[0]
    if raw_code in _UNKNOWN_CODES:
        return None
    return raw_code


@register_model
class CLD2Model(LIDModel):
    model_id = "cld2"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return [_detect_one(t) for t in texts]

    def discover_supported_languages(self) -> frozenset[str]:
        """Return every ISO 639-3 code enumerable from :data:`pycld2.LANGUAGES`."""
        import pycld2 as cld2

        codes: set[str] = set()
        for _name, code in cld2.LANGUAGES:
            if code in _UNKNOWN_CODES:
                continue
            short = code.split("-")[0]
            conformed = self._conform(short)
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
