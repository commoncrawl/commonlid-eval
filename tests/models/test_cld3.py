"""Unit tests for the CLD3 model wrapper."""

from __future__ import annotations

import sys

import pytest

from commonlid.models.cld3 import CLD3Model

pytest.importorskip("gcld3")


def test_load_initialises_detector() -> None:
    model = CLD3Model()
    assert model._detector is None
    model.load()
    assert model._detector is not None
    assert model._loaded is True


def test_load_is_idempotent() -> None:
    model = CLD3Model()
    model.load()
    detector = model._detector
    model.load()
    assert model._detector is detector


def test_predict_returns_iso639_3() -> None:
    model = CLD3Model()
    preds = model.predict([
        "This is an English sentence written in clear English prose.",
        "Bonjour le monde, ceci est un texte écrit en français.",
        "Это предложение написано на русском языке для теста.",
        "你好世界 这是一段用中文书写的测试文本",
    ])
    assert preds == ["eng", "fra", "rus", "zho"]


def test_predict_maps_und_to_none() -> None:
    """A raw ``und`` from CLD3 must surface as None, not an iso639 conform error.

    The pybind ``NNetLanguageIdentifier`` does not allow attribute monkeypatching,
    so we swap the whole detector out for a fake — exercises the
    ``code == "und"`` branch in ``_predict_batch`` end-to-end.
    """

    class _FakeResult:
        def __init__(self, language: str) -> None:
            self.language = language
            self.is_reliable = True
            self.probability = 1.0
            self.proportion = 1.0

    class _FakeDetector:
        def FindLanguage(self, text: str) -> _FakeResult:  # noqa: N802 — mirrors gcld3 API
            return _FakeResult("und")

    model = CLD3Model()
    model.load()
    model._detector = _FakeDetector()
    assert model.predict(["whatever"]) == [None]


def test_discover_supported_languages_includes_majors() -> None:
    supported = CLD3Model().discover_supported_languages()
    assert isinstance(supported, frozenset)
    assert len(supported) > 90
    assert all(len(code) == 3 for code in supported)
    assert {"eng", "deu", "fra", "spa", "zho", "jpn", "rus"} <= supported


def test_load_raises_helpful_error_without_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gcld3", None)
    with pytest.raises(ImportError, match=r"commonlid\[cld3\]"):
        CLD3Model().load()
