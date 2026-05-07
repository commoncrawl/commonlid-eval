"""Integration tests for the ``commonlid generate-support-matrix`` subcommand."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from commonlid.cli import app
from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model
from commonlid.metrics.support_matrix import load_support_matrix

runner = CliRunner()


class _KnownModel(LIDModel):
    model_id = "_known_supp"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return ["eng"] * len(texts)

    def discover_supported_languages(self) -> frozenset[str]:
        return frozenset({"eng", "deu", "fra"})


class _UnknownModel(LIDModel):
    model_id = "_unknown_supp_cli"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return [None] * len(texts)

    def discover_supported_languages(self) -> frozenset[str] | None:
        return None


class _BrokenModel(LIDModel):
    model_id = "_broken_supp"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return [None] * len(texts)

    def discover_supported_languages(self) -> frozenset[str]:
        msg = "boom"
        raise RuntimeError(msg)


@pytest.fixture(autouse=True)
def _register(fresh_registry: None) -> None:
    register_model(_KnownModel)
    register_model(_UnknownModel)
    register_model(_BrokenModel)


def test_generate_support_matrix_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "support.csv"
    result = runner.invoke(
        app,
        [
            "generate-support-matrix",
            "--out",
            str(out),
            "--model",
            "_known_supp",
        ],
    )
    assert result.exit_code == 0, result.stdout
    matrix = load_support_matrix(out)
    assert matrix == {"_known_supp": {"eng", "deu", "fra"}}


def test_generate_support_matrix_skips_unenumerable_and_broken(tmp_path: Path) -> None:
    out = tmp_path / "support.csv"
    result = runner.invoke(
        app,
        [
            "generate-support-matrix",
            "--out",
            str(out),
            "--model",
            "_known_supp",
            "--model",
            "_unknown_supp_cli",
            "--model",
            "_broken_supp",
        ],
    )
    assert result.exit_code == 0, result.stdout
    matrix = load_support_matrix(out)
    assert set(matrix) == {"_known_supp"}


def test_generate_support_matrix_errors_when_no_models_produce(tmp_path: Path) -> None:
    out = tmp_path / "support.csv"
    result = runner.invoke(
        app,
        [
            "generate-support-matrix",
            "--out",
            str(out),
            "--model",
            "_unknown_supp_cli",
            "--model",
            "_broken_supp",
        ],
    )
    assert result.exit_code != 0
    assert not out.exists()
