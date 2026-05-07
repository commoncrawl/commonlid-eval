"""Unit tests for the shared fasttext base class using a stub ft model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from commonlid.models._fasttext_base import FastTextHubModel


class _StubCppBinding:
    """Matches the shape of ``fasttext.FastText._FastText.f`` (the C++ object)."""

    def __init__(self, labels: Sequence[Sequence[str]], *, return_tuple: bool) -> None:
        self._labels = [list(group) for group in labels]
        self._return_tuple = return_tuple

    def multilinePredict(self, texts: list[str], _k: int, _th: float, _onerr: str):  # noqa: N802
        if self._return_tuple:
            # fasttext-wheel shape: ([[labels], ...], [[probs], ...])
            return self._labels, [[1.0] * len(group) for group in self._labels]
        # fasttext-predict shape: [[labels], ...]
        return self._labels


class _StubFT:
    """Stubs ``_FastText``; exposes the attributes the base class reads."""

    def __init__(self, labels: Sequence[Sequence[str]], *, return_tuple: bool = True) -> None:
        self.f = _StubCppBinding(labels, return_tuple=return_tuple)
        self._labels = [list(group) for group in labels]

    def predict(self, texts: list[str]) -> tuple[list[list[str]], list[list[float]]]:
        # Used only when ``self.f.multilinePredict`` is absent; exercised
        # explicitly in ``test_predict_fallback_when_no_multiline``.
        return self._labels, [[1.0] * len(group) for group in self._labels]


class _StubModel(FastTextHubModel):
    model_id = "_stub"
    hf_repo_id = "stub/stub"


def test_parses_label_and_conforms_tuple_shape() -> None:
    """fasttext-wheel returns (labels, probs) as a tuple."""
    m = _StubModel()
    m._ft = _StubFT([["__label__jw_Latn"], ["__label__eng_Latn"]], return_tuple=True)
    m._loaded = True
    assert m.predict(["Hello", "world"]) == ["jav", "eng"]


def test_parses_label_and_conforms_list_shape() -> None:
    """fasttext-predict returns just the labels list."""
    m = _StubModel()
    m._ft = _StubFT([["__label__jw_Latn"], ["__label__eng_Latn"]], return_tuple=False)
    m._loaded = True
    assert m.predict(["Hello", "world"]) == ["jav", "eng"]


def test_predict_fallback_when_no_multiline() -> None:
    """If the C++ binding lacks ``multilinePredict`` we fall back to ``.predict``."""
    m = _StubModel()
    ft = _StubFT([["__label__eng_Latn"]])
    ft.f = object()  # no multilinePredict attr
    m._ft = ft
    m._loaded = True
    assert m.predict(["hi"]) == ["eng"]


def test_unexpected_label_format_raises() -> None:
    m = _StubModel()
    m._ft = _StubFT([["no-prefix-here"]])
    m._loaded = True
    with pytest.raises(ValueError, match="Unexpected fasttext label format"):
        m.predict(["hi"])


def test_load_downloads_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {"download": 0, "load": 0}

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls["download"] += 1
        assert repo_id == "stub/stub"
        assert filename == "model.bin"
        return "/tmp/fake.bin"

    def fake_load(path: str) -> _StubFT:
        calls["load"] += 1
        assert path == "/tmp/fake.bin"
        return _StubFT([["__label__eng_Latn"]])

    monkeypatch.setattr("commonlid.models._fasttext_base.hf_hub_download", fake_download)
    monkeypatch.setattr("commonlid.models._fasttext_base.fasttext.load_model", fake_load)

    m = _StubModel()
    m.load()
    m.load()  # idempotent
    assert calls["download"] == 1
    assert calls["load"] == 1
    assert m.predict(["hello"]) == ["eng"]
