"""Every shipped model module registers exactly one model_id."""

from __future__ import annotations

import importlib

import pytest

from commonlid.core.registry import list_models


@pytest.fixture(autouse=True)
def _import_models() -> None:
    importlib.import_module("commonlid.models")


EXPECTED_MODEL_IDS = {
    "AfroLID",
    "GlotLID",
    "OpenLID-v2",
    "cld2",
    "cld3",
    "fasttext",
    "funlangid",
    "pyfranc",
}


def test_expected_models_registered() -> None:
    assert set(list_models()) == EXPECTED_MODEL_IDS


def test_dspy_llm_is_not_autoregistered() -> None:
    # DSPyLLMModel is importable but not in the registry.
    from commonlid.models.dspy_llm import DSPyLLMModel  # noqa: F401

    assert "dspy_llm" not in list_models()
