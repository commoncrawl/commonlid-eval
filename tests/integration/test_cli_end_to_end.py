"""End-to-end CLI tests using `typer.testing.CliRunner`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from commonlid.cli import app
from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_dataset, register_model

runner = CliRunner()


class _TinyDataset(LIDDataset):
    dataset_id = "_tiny_cli"
    source_hf_repo = "fake/fake"
    source_hf_revision = "rev-cli"
    text_column = "text"
    target_column = "iso639_3"

    SAMPLES: ClassVar[list[dict[str, str]]] = [
        {"text": "hello", "iso639_3": "eng"},
        {"text": "hallo", "iso639_3": "deu"},
    ]

    def load(self, *, limit: int | None = None):  # type: ignore[override]
        from datasets import Dataset

        ds = Dataset.from_list(self.SAMPLES)
        if limit is not None and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        self._dataset = ds
        return ds


class _TinyModel(LIDModel):
    model_id = "_tiny_cli_model"
    requires_preprocessing = False

    _ANSWERS: ClassVar[dict[str, str]] = {"hello": "eng", "hallo": "deu"}

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return [self._ANSWERS.get(t) for t in texts]


@pytest.fixture(autouse=True)
def _register_fixtures(fresh_registry: None) -> None:
    register_model(_TinyModel)
    register_dataset(_TinyDataset)


def test_run_writes_summary_and_predictions(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "_tiny_cli_model",
            "--dataset",
            "_tiny_cli",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary_path = tmp_path / "_tiny_cli" / "_tiny_cli_model" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["macro"]["f1_gold_only"] == pytest.approx(1.0)


def test_export_csv(tmp_path: Path) -> None:
    # First run eval to generate summary.json files.
    runner.invoke(
        app,
        [
            "run",
            "--model",
            "_tiny_cli_model",
            "--dataset",
            "_tiny_cli",
            "--output-dir",
            str(tmp_path),
        ],
    )

    out_csv = tmp_path / "flat.csv"
    result = runner.invoke(
        app,
        [
            "export-csv",
            "--results-dir",
            str(tmp_path),
            "--out",
            str(out_csv),
        ],
    )
    assert result.exit_code == 0, result.stdout
    lines = out_csv.read_text().strip().splitlines()
    assert lines[0].startswith("dataset_id,model_id,language")
    # Two rows (eng + deu).
    assert len(lines) == 3


def test_export_csv_empty_dir_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "export-csv",
            "--results-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code != 0
    # The CSV file must not have been created on failure.
    assert not (tmp_path / "out.csv").exists()


def test_predict_with_text(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["predict", "--model", "_tiny_cli_model", "--text", "hello"],
    )
    assert result.exit_code == 0
    line = result.stdout.strip()
    parsed = json.loads(line)
    assert parsed["pred"] == "eng"


def test_predict_with_text_file(tmp_path: Path) -> None:
    f = tmp_path / "inputs.txt"
    f.write_text("hello\nhallo\n")
    result = runner.invoke(
        app,
        ["predict", "--model", "_tiny_cli_model", "--text-file", str(f)],
    )
    assert result.exit_code == 0
    preds = [json.loads(line)["pred"] for line in result.stdout.strip().splitlines()]
    assert preds == ["eng", "deu"]


def test_run_dspy_spec_requires_api_base(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "dspy:azure/gpt-4o-mini",
            "--dataset",
            "_tiny_cli",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0


def test_run_dspy_spec_requires_model_name(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "dspy:",
            "--dataset",
            "_tiny_cli",
            "--api-base",
            "https://example",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0


def test_run_dspy_spec_builds_dspy_llm_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--model dspy:NAME spawns a DSPyLLMModel and runs it through the evaluator."""
    from commonlid.models import dspy_llm as dspy_llm_mod

    built: dict[str, object] = {}

    class _FakeDSPy(LIDModel):
        model_id = "dspy_azure_gpt-4o-mini"
        requires_preprocessing = False

        def __init__(self, *, llm_model_name: str, **kwargs: object) -> None:
            super().__init__()
            built["llm_model_name"] = llm_model_name
            built["kwargs"] = kwargs

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["eng"] * len(texts)

    monkeypatch.setattr(dspy_llm_mod, "DSPyLLMModel", _FakeDSPy)

    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "dspy:azure/gpt-4o-mini",
            "--dataset",
            "_tiny_cli",
            "--api-base",
            "https://endpoint",
            "--azure-ad-token",
            "--temperature",
            "0.7",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert built["llm_model_name"] == "azure/gpt-4o-mini"
    kwargs = built["kwargs"]
    assert kwargs["api_base"] == "https://endpoint"  # type: ignore[index]
    assert kwargs["azure_ad_token"] is True  # type: ignore[index]
    assert kwargs["temperature"] == 0.7  # type: ignore[index]

    # The evaluator wrote output under its (LLM-instance) model_id.
    summary_path = tmp_path / "_tiny_cli" / "dspy_azure_gpt-4o-mini" / "summary.json"
    assert summary_path.exists()


def test_run_llm_command_is_gone(tmp_path: Path) -> None:
    """The old `commonlid run-llm` subcommand no longer exists."""
    result = runner.invoke(app, ["run-llm", "--help"])
    assert result.exit_code != 0
