"""CLD3 model wrapper.

Wraps Google's CLD3 neural language identifier via the
`cld3-py <https://pypi.org/project/cld3-py/>`_ PyPI package, which
exposes the original C++ inference code as Python bindings under the
import name ``gcld3`` (drop-in replacement for the abandoned
``gcld3`` package). Pulled in via the optional ``commonlid[cld3]``
extra; if the bindings are missing, :meth:`load` raises a helpful
``ImportError`` instead of silently skipping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model

# Per the upstream cld3-py README, CLD3 emits one of these BCP-47 codes
# (bare language or ``language-Script``). Enumerated statically so
# ``discover_supported_languages()`` does not need a loaded detector.
_CLD3_BCP47_CODES: tuple[str, ...] = (
    "af",
    "am",
    "ar",
    "bg",
    "bg-Latn",
    "bn",
    "bs",
    "ca",
    "ceb",
    "co",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "el-Latn",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "haw",
    "hi",
    "hi-Latn",
    "hmn",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "iw",
    "ja",
    "ja-Latn",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "ru-Latn",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zh-Latn",
    "zu",
)


@register_model
class CLD3Model(LIDModel):
    model_id = "cld3"

    # Mirror the legacy `tests/legacy/langid_models.py` configuration so the
    # parity smoke tests are bit-exact: changing `max_num_bytes` changes
    # which prefix CLD3 actually scores.
    _MIN_NUM_BYTES: ClassVar[int] = 0
    _MAX_NUM_BYTES: ClassVar[int] = 4096

    def __init__(self) -> None:
        super().__init__()
        self._detector: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import gcld3
        except ImportError as exc:
            msg = (
                "The 'cld3' Python bindings are not installed. Install the "
                "'commonlid[cld3]' extra to enable this model."
            )
            raise ImportError(msg) from exc
        self._detector = gcld3.NNetLanguageIdentifier(
            min_num_bytes=self._MIN_NUM_BYTES,
            max_num_bytes=self._MAX_NUM_BYTES,
        )
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        out: list[str | None] = []
        for text in texts:
            result = self._detector.FindLanguage(text=text)
            code = result.language.split("-")[0]
            out.append(None if code == "und" else code)
        return out

    def discover_supported_languages(self) -> frozenset[str]:
        """Return every ISO 639-3 code derivable from CLD3's BCP-47 outputs."""
        codes: set[str] = set()
        for raw in _CLD3_BCP47_CODES:
            short = raw.split("-")[0]
            conformed = self._conform(short)
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
