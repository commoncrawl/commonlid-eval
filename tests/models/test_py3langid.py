from __future__ import annotations

import pytest

pytest.importorskip("py3langid")

from commonlid.models.py3langid import _MIN_CONFIDENCE, _OUT_MAP, Py3LangIDModel


def test_returns_conformed_codes() -> None:
    m = Py3LangIDModel()
    preds = m.predict([
        "The quick brown fox jumps over the lazy dog.",
        "Le renard brun saute par-dessus le chien paresseux.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
    ])
    assert preds == ["eng", "fra", "deu"]


def test_predict_scored_returns_the_posterior() -> None:
    m = Py3LangIDModel()
    scored = m.predict_scored(["The quick brown fox jumps over the lazy dog.", "123 456"])
    assert scored[0].iso639_3 == "eng"
    assert scored[0].score is not None
    assert _MIN_CONFIDENCE < scored[0].score <= 1.0
    assert scored[1].iso639_3 is None
    assert scored[1].score is not None
    assert 0.0 <= scored[1].score <= 1.0


def test_abstains_on_empty_and_gibberish() -> None:
    m = Py3LangIDModel()
    assert m.predict(["", "   ", "123 456", "!!!"]) == [None, None, None, None]


def test_out_map_overrides_default_conformation() -> None:
    m = Py3LangIDModel()
    assert _OUT_MAP["ar"] == "arb"
    assert m.predict(["هذا نص قصير مكتوب باللغة العربية الفصحى الحديثة."]) == ["arb"]
    bikol = "An mga tawo dapat magkaminootan asin magtinabangan sa lambang aldaw."
    assert m.predict([bikol]) == ["bik"]


def test_discover_supported_languages() -> None:
    m = Py3LangIDModel()
    langs = m.discover_supported_languages()
    assert {"eng", "fra", "deu", "arz", "vec"} <= langs
    assert "zxx" not in langs
    assert "bik" in langs
    assert not {"bcl", "ar", "und"} & langs


def test_min_confidence_is_in_effect() -> None:
    m = Py3LangIDModel()
    m.load()
    assert m._identifier.min_confidence == _MIN_CONFIDENCE
