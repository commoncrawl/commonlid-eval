from __future__ import annotations

import json

from typer.testing import CliRunner

from commonlid import __version__
from commonlid.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_list_models_json_contains_known_models() -> None:
    result = runner.invoke(app, ["list-models", "--json"])
    assert result.exit_code == 0
    ids = json.loads(result.stdout.strip())
    assert "GlotLID" in ids
    assert "cld2" in ids
    assert "dspy_llm" not in ids


def test_list_datasets_includes_commonlid() -> None:
    result = runner.invoke(app, ["list-datasets"])
    assert result.exit_code == 0
    assert "commonlid" in result.stdout
    assert "udhr" in result.stdout


def test_predict_requires_input() -> None:
    result = runner.invoke(app, ["predict", "--model", "noop"])
    assert result.exit_code == 2


def test_run_unknown_model(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "does-not-exist",
            "--dataset",
            "also-not-exist",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
