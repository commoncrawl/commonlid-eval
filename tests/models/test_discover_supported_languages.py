"""Tests for ``discover_supported_languages()`` across shipped models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

pytest.importorskip("pycld2")
pytest.importorskip("pyfranc")


def test_cld2_enumerates_known_languages() -> None:
    from commonlid.models.cld2 import CLD2Model

    langs = CLD2Model().discover_supported_languages()
    assert "eng" in langs
    assert "deu" in langs
    assert "fra" in langs
    # cld2 exposes ~180 distinct ISO 639-3 codes; a few dozen is the floor.
    assert len(langs) >= 50


def test_pyfranc_enumerates_known_languages() -> None:
    from commonlid.models.pyfranc import PyfrancModel

    langs = PyfrancModel().discover_supported_languages()
    assert "eng" in langs
    assert "spa" in langs
    assert len(langs) >= 50


def test_fasttext_base_parses_labels() -> None:
    from commonlid.models._fasttext_base import FastTextHubModel

    class _StubFT:
        def predict(self, texts: list[str]) -> tuple[list[list[str]], list[list[float]]]:
            return [["__label__eng_Latn"] for _ in texts], [[1.0] for _ in texts]

        def get_labels(self) -> list[str]:
            return [
                "__label__eng_Latn",
                "__label__jw_Latn",  # conforms to 'jav'
                "__label__xyz_Unknown",  # invalid ISO 639-3, dropped
                "not-a-label",
            ]

    class _Model(FastTextHubModel):
        model_id = "_fasttext_stub"
        hf_repo_id = "stub/stub"

    m = _Model()
    m._ft = _StubFT()  # type: ignore[assignment]
    m._loaded = True
    langs = m.discover_supported_languages()
    assert "eng" in langs
    assert "jav" in langs  # 'jw' was conformed to 'jav'
    assert "xyz" not in langs


def test_base_default_returns_supported_languages_attr() -> None:
    from commonlid.core.lid_model import LIDModel

    class _Bounded(LIDModel):
        model_id = "_bounded"
        supported_languages = frozenset({"eng", "deu"})

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["eng"] * len(texts)

    assert _Bounded().discover_supported_languages() == frozenset({"eng", "deu"})


def test_base_default_returns_none_when_unset() -> None:
    from commonlid.core.lid_model import LIDModel

    class _Unknown(LIDModel):
        model_id = "_unknown_supp"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return [None] * len(texts)

    assert _Unknown().discover_supported_languages() is None


def test_funlangid_enumerates_languages() -> None:
    """FunLangID enumerates via its lex_dict; smoke-check it includes common codes."""
    pytest.importorskip("commonlid.vendor.fun_langid")
    from commonlid.models.funlangid import FunLangIDModel

    langs = FunLangIDModel().discover_supported_languages()
    assert "eng" in langs
    # FunLangID's ~150 languages should survive ISO 639-3 conformance.
    assert len(langs) >= 100


def test_cld3_discover_returns_frozenset() -> None:
    """CLD3 enumerates a static BCP-47 list; no detector load needed."""
    from commonlid.models.cld3 import CLD3Model

    langs = CLD3Model().discover_supported_languages()
    assert isinstance(langs, frozenset)
    assert "eng" in langs
    assert "zho" in langs
    # The upstream README lists 107 BCP-47 codes; after stripping script
    # suffixes and conforming to ISO 639-3 we expect ~100 distinct languages.
    assert len(langs) >= 95


def test_afrolid_discover_reads_id2label(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import ClassVar

    from commonlid.models import afrolid as afrolid_mod
    from commonlid.models.afrolid import AfroLIDModel

    class _FakeConfig:
        id2label: ClassVar[dict[int, str]] = {
            0: "eng",
            1: "nan_lang",
            2: "jw",  # -> jav after conform
            3: "xxxxx",
        }

    class _FakeModel:
        config = _FakeConfig()

    class _FakePipeline:
        model = _FakeModel()

        def __call__(self, *_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
            return []

    def fake_load(self: AfroLIDModel) -> None:
        self._pipeline = _FakePipeline()
        self._loaded = True

    monkeypatch.setattr(AfroLIDModel, "load", fake_load)
    langs = AfroLIDModel().discover_supported_languages()
    assert "eng" in langs
    assert "jav" in langs
    assert "xxxxx" not in langs

    # Reach the module to keep the import alive for mypy/coverage.
    _ = afrolid_mod.AfroLIDModel
