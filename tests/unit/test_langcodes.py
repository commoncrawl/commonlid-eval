from __future__ import annotations

import pytest

from commonlid.preprocess.langcodes import (
    _DEPRECATION_TABLE,
    conform_langcode,
    conform_langcode_with_reason,
    convert_and_conform_language,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eng", ("eng", None)),
        ("jpn", ("jpn", None)),
        ("fra", ("fra", None)),
        ("jw", ("jav", _DEPRECATION_TABLE["jw"][1])),
        ("iw", ("heb", _DEPRECATION_TABLE["iw"][1])),
        ("ajp", ("apc", _DEPRECATION_TABLE["ajp"][1])),
        ("eml", (None, _DEPRECATION_TABLE["eml"][1])),
        ("ber", ("tzm", _DEPRECATION_TABLE["ber"][1])),
        ("kzj", ("dtp", _DEPRECATION_TABLE["kzj"][1])),
        ("tpw", ("tpn", _DEPRECATION_TABLE["tpw"][1])),
    ],
)
def test_conform_langcode_with_reason(raw: str, expected: tuple[str | None, str | None]) -> None:
    assert conform_langcode_with_reason(raw) == expected


def test_conform_langcode_returns_none_for_split() -> None:
    assert conform_langcode("eml") is None
    assert conform_langcode("nah") is None


def test_conform_langcode_passes_through_unknown() -> None:
    # Unknown codes (not in the deprecation table) are returned unchanged.
    assert conform_langcode("xyz") == "xyz"


def test_convert_and_conform_language_basic() -> None:
    codes = ["en", "de", "fr"]
    result = convert_and_conform_language(codes)
    assert result == ["eng", "deu", "fra"]


def test_convert_and_conform_language_deprecated_iso1_returns_none() -> None:
    # `iso639-lang` raises DeprecatedLanguageValue for 'jw'; the helper catches
    # the exception and yields None (the 'jw'->'jav' conform table is only hit
    # when a *model* emits 'jw' directly).
    assert convert_and_conform_language(["jw"]) == [None]


def test_convert_and_conform_language_strips_script_suffix() -> None:
    assert convert_and_conform_language(["en-US", "zh_Hant"]) == ["eng", "zho"]


def test_convert_and_conform_language_invalid_code_returns_none() -> None:
    result = convert_and_conform_language(["xxxxxx"])
    assert result == [None]
