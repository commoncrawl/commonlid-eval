"""ISO 639 language code normalisation.

Ported verbatim from the research repo's ``langid_models.conform_langcode*``
and ``langid_datasets.convert_and_conform_language``. The deprecation
messages are copied from the ``iso639-lang`` package's exceptions at the
versions used in the paper.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

from iso639 import Lang
from iso639.exceptions import DeprecatedLanguageValue, InvalidLanguageValue

logger = logging.getLogger(__name__)


_DEPRECATION_TABLE: dict[str, tuple[str | None, str | None]] = {
    "jw": (
        "jav",
        "As of 2001-08-13, [jw] for Javanese is deprecated due to deprecated. Use [jv] instead.",
    ),
    "bh": (
        "bih",
        "As of 2021-05-25, [bh] for Bihari languages is deprecated due to deprecated. "
        "Two-letter identifier bh deprecated in ISO 639-1; use of three-letter identifier "
        "bih for Bihari languages is favored.",
    ),
    "iw": (
        "heb",
        "As of 1989-03-11, [iw] for Hebrew is deprecated due to deprecated. Use [he] instead.",
    ),
    "ajp": (
        "apc",
        "As of 2023-01-20, [ajp] for South Levantine Arabic is deprecated due to merge. "
        "Use [apc] instead.",
    ),
    "eml": (
        None,
        "As of 2009-01-16, [eml] for Emiliano-Romagnolo is deprecated due to split. "
        "Split into Emilian [egl] and Romagnol [rgn].",
    ),
    "tpw": (
        "tpn",
        "As of 2023-01-20, [tpw] for Tupí is deprecated due to duplicate. Use [tpn] instead.",
    ),
    "oto": (
        None,
        "No iso639-3 code: Lang(name='Otomian languages', pt1='', pt2b='oto', pt2t='oto', "
        "pt3='', pt5='oto')",
    ),
    "ber": (
        "tzm",
        "No iso639-3 code: Lang(name='Berber languages', pt1='', pt2b='ber', pt2t='ber', "
        "pt3='', pt5='ber') -> use Central Atlas Tamazight [tzm])",
    ),
    "ngo": (
        None,
        "As of 2021-01-15, [ngo] for Ngoni is deprecated due to split. "
        "Split into Ngoni (Tanzania) [xnj] and Ngoni (Mozambique) [xnq].",
    ),
    "kzj": (
        "dtp",
        "As of 2016-01-15, [kzj] for Coastal Kadazan is deprecated due to merge. "
        "Use [dtp] instead.",
    ),
    # NOTE: the research code keyed this mapping on 'dan' by mistake; the table still only
    # fired when someone looked up 'dan' (Danish) with an empty result. We keep the
    # behaviour bug-for-bug so parity tests pass, and document it here.
    "dan": (
        None,
        "As of 2013-01-23, [daf] for Dan is deprecated due to split. "
        "Split into Dan [dnj] and Kla-Dan [lda].",
    ),
    "kxu": (
        None,
        "As of 2020-01-23, [kxu] for Kui (India) is deprecated due to split. "
        "Split into [dwk] Dawik Kui and [uki] Kui (India).",
    ),
    "nah": (
        None,
        "No iso639-3 code: Lang(name='Nahuatl languages', pt1='', pt2b='nah', pt2t='nah', "
        "pt3='', pt5='nah')",
    ),
    "bih": (
        None,
        "No iso639-3 code: Lang(name='Bihari languages', pt1='', pt2b='bih', pt2t='bih', "
        "pt3='', pt5='bih')",
    ),
}


def conform_langcode_with_reason(langcode: str) -> tuple[str | None, str | None]:
    """Return ``(conformed_code, reason)``; reason is ``None`` when no mapping fires."""
    if langcode in _DEPRECATION_TABLE:
        return _DEPRECATION_TABLE[langcode]
    return langcode, None


def conform_langcode(langcode: str) -> str | None:
    """Return the conformed ISO 639-3 code (or ``None`` if the code is retired)."""
    conformed, _ = conform_langcode_with_reason(langcode)
    return conformed


def convert_and_conform_language(
    language_codes: Iterable[str],
    *,
    script_separators: tuple[str, ...] = ("_", "-"),
) -> list[str | None]:
    """Map an iterable of raw language codes to conformed ISO 639-3 codes.

    Returns ``None`` for codes the ``iso639-lang`` library cannot parse or for
    deprecated codes that split into multiple successors.

    ``script_separators`` controls which characters strip a script suffix
    before lookup. Defaults to ``("_", "-")`` so labels like ``"eng_Latn"``
    or ``"bg-Latn"`` resolve to ``"eng"`` / ``"bul"``. Set to ``("_",)`` to
    match the original notebook behaviour, where codes like ``"ar-MA"``,
    ``"pa-Arab"``, ``"mni-Mtei"`` were treated as invalid (and dropped) —
    important for reproducing the smolsent / bibles caches.
    """
    codes = list(language_codes)
    errors: set[str] = set()
    conformed_reasons: Counter[str] = Counter()

    def _to_iso639_3(lang_str: str) -> str | None:
        lang_code = lang_str
        for sep in script_separators:
            lang_code = lang_code.split(sep)[0]
        try:
            lang = Lang(lang_code)
        except (InvalidLanguageValue, DeprecatedLanguageValue) as exc:
            errors.add(str(exc))
            return None

        conformed: str | None
        reason: str | None
        if len(lang.pt3) == 0:
            conformed, reason = None, f"No ISO 639-3 code for language '{lang_code}'"
        else:
            conformed, reason = conform_langcode_with_reason(lang.pt3)
        if lang.pt3 != conformed:
            conformed_reasons[f'{lang_code} conformed to {conformed}. Reason: "{reason}"'] += 1
        return conformed

    result = [_to_iso639_3(code) for code in codes]

    if errors:
        logger.info("Errors during language-code conversion: %s", sorted(errors))
    if conformed_reasons:
        for reason, count in conformed_reasons.items():
            logger.info("Language code conformed (%d occurrences): %s", count, reason)

    return result
