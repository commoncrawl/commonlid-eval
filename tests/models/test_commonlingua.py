"""Unit tests for CommonLinguaModel.

The real model needs the ``[commonlingua]`` extra (torch), which is not in
``dev``. We mock the heavy bits and exercise the wrapper logic.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import pytest

from commonlid.core.lid_model import LIDPrediction
from commonlid.models import commonlingua as commonlingua_mod
from commonlid.models.commonlingua import CommonLinguaModel


def test_load_raises_helpful_error_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"commonlid\[commonlingua\]"):
        CommonLinguaModel().load()


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a `torch` stub covering only what ``_predict_batch`` touches."""

    class _FakeTensor:
        def __init__(self, values: list[Any]) -> None:
            self._values = values

        def cpu(self) -> _FakeTensor:
            return self

        def tolist(self) -> list[Any]:
            return self._values

        def to(self, _device: str) -> _FakeTensor:
            return self

        def max(self, dim: int = -1) -> tuple[_FakeTensor, _FakeTensor]:
            """Mimic ``Tensor.max(dim)`` -> ``(values, indices)``."""
            values = [max(row) for row in self._values]
            indices = [row.index(max(row)) for row in self._values]
            return _FakeTensor(values), _FakeTensor(indices)

    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, batch: Any) -> _FakeTensor:
            self.calls += 1
            # Argmax at 0, 1, 2 -> eng, fra, deu via the fake idx2lang.
            return _FakeTensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

    class _NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_a: Any) -> None:
            return None

    fake_torch = type(sys)("torch")
    fake_torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    fake_torch.from_numpy = lambda arr: _FakeTensor(list(arr.flatten()))  # type: ignore[attr-defined]
    # The stub model already emits normalised rows, so softmax is the identity.
    fake_torch.softmax = lambda tensor, _dim=-1: tensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_load(self: CommonLinguaModel) -> None:
        self._model = _FakeModel()
        self._idx2lang = {0: "eng", 1: "fra", 2: "deu"}
        self._max_len = 512
        self._device = "cpu"
        self._loaded = True

    monkeypatch.setattr(CommonLinguaModel, "load", fake_load)


def test_predict_returns_codes_from_idx2lang(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    preds = CommonLinguaModel().predict(["Hello", "Bonjour", "Hallo"])
    assert preds == ["eng", "fra", "deu"]


def test_predict_scored_returns_the_softmax_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)
    scored = CommonLinguaModel().predict_scored(["Hello", "Bonjour", "Hallo"])
    assert scored == [
        LIDPrediction("eng", 0.7),
        LIDPrediction("fra", 0.8),
        LIDPrediction("deu", 0.6),
    ]


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
