"""Unit tests for the cld2 wrapper (uses the real pycld2 library)."""

from __future__ import annotations

import pytest

pytest.importorskip("pycld2")

from commonlid.models.cld2 import CLD2Model


def test_detects_english() -> None:
    m = CLD2Model()
    preds = m.predict(["The quick brown fox jumps over the lazy dog."])
    assert preds == ["eng"]


def test_detects_german() -> None:
    m = CLD2Model()
    preds = m.predict(["Der schnelle braune Fuchs springt über den faulen Hund"])
    assert preds == ["deu"]


def test_returns_none_for_gibberish() -> None:
    m = CLD2Model()
    preds = m.predict(["xxxxxxxxxxxxxxxx"])
    # xxxxx should produce one of the sentinel codes -> None.
    assert preds == [None]
