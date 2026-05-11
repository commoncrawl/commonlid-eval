"""Unit tests for CommonLinguaModel.

The real model needs the ``[commonlingua]`` extra (torch), which is not in
``dev``. We mock the heavy bits and exercise the wrapper logic.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import pytest

from commonlid.models import commonlingua as commonlingua_mod
from commonlid.models.commonlingua import CommonLinguaModel


def test_load_raises_helpful_error_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"commonlid\[commonlingua\]"):
        CommonLinguaModel().load()


def test_predict_returns_codes_from_idx2lang(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTensor:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def argmax(self, dim: int = -1) -> _FakeTensor:
            return self

        def cpu(self) -> _FakeTensor:
            return self

        def tolist(self) -> list[int]:
            return self._values

        def to(self, _device: str) -> _FakeTensor:
            return self

    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, batch: Any) -> _FakeTensor:
            self.calls += 1
            # Indices 0, 1, 2 -> eng, fra, deu via fake idx2lang.
            return _FakeTensor([0, 1, 2])

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: Any) -> None:
            return None

    fake_torch = type(sys)("torch")
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    fake_torch.from_numpy = lambda arr: _FakeTensor(list(arr.flatten()))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_load(self: CommonLinguaModel) -> None:
        self._model = _FakeModel()
        self._idx2lang = {0: "eng", 1: "fra", 2: "deu"}
        self._max_len = 512
        self._device = "cpu"
        self._loaded = True

    monkeypatch.setattr(CommonLinguaModel, "load", fake_load)
    preds = CommonLinguaModel().predict(["Hello", "Bonjour", "Hallo"])
    assert preds == ["eng", "fra", "deu"]


def test_discover_supported_languages_conforms_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Marker:
        idx2lang: ClassVar[dict[int, str]] = {0: "eng", 1: "jw", 2: "xxxxx"}

    def fake_load(self: CommonLinguaModel) -> None:
        self._idx2lang = dict(_Marker.idx2lang)
        self._loaded = True

    monkeypatch.setattr(CommonLinguaModel, "load", fake_load)
    langs = CommonLinguaModel().discover_supported_languages()
    assert "eng" in langs
    assert "jav" in langs  # jw -> jav via _conform
    assert "xxxxx" not in langs


def test_model_registered() -> None:
    from commonlid.core.registry import get_model

    model = get_model("commonlingua")
    assert isinstance(model, CommonLinguaModel)
    # Keep the module reference alive for coverage/mypy.
    assert commonlingua_mod.CommonLinguaModel is CommonLinguaModel
