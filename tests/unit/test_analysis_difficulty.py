"""Tests for ``commonlid.analysis`` (top-k model difficulty analysis, offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from commonlid.analysis import (
    attach_texts,
    language_difficulty,
    load_predictions,
    sample_difficulty,
    select_top_models,
)
from commonlid.analysis.difficulty import to_macrolanguage
from commonlid.cli import app
from commonlid.evaluation.evaluator import _text_hash

pd = pytest.importorskip("pandas")

runner = CliRunner()

DATASET = "toy"
TEXTS = ["hello world", "hallo welt", "bonjour", "labas", "???"]
GOLDS = ["eng", "deu", "fra", "lvs", "eng"]
# Per model predictions for TEXTS. Sample 1 looks mislabelled (everybody says
# nld), sample 2 is simply hard (models disagree), sample 3 is a
# macrolanguage mismatch (lvs gold, lav predicted).
PREDS: dict[str, list[str | None]] = {
    "A": ["eng", "nld", "ita", "lav", None],
    "B": ["eng", "nld", "spa", "lav", "eng"],
    "C": ["eng", "nld", "fra", "lav", None],
}
MACRO_F1 = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.1}


def _write_run(root: Path, model_id: str, preds: list[str | None], macro_f1: float) -> None:
    out = root / DATASET / model_id
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "idx": i,
            "text_hash": _text_hash(text),
            "gold": gold,
            "pred": pred,
            "correct": gold == pred,
        }
        for i, (text, gold, pred) in enumerate(zip(TEXTS, GOLDS, preds, strict=True))
    ]
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "model_id": model_id,
        "dataset_id": DATASET,
        "macro": {"f1_gold_only": macro_f1},
        "micro": {},
        "per_language": {},
    }
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    for model_id, preds in PREDS.items():
        _write_run(tmp_path, model_id, preds, MACRO_F1[model_id])
    # D has a summary but no predictions file.
    _write_run(tmp_path, "D", PREDS["A"], MACRO_F1["D"])
    (tmp_path / DATASET / "D" / "predictions.jsonl").unlink()
    return tmp_path


def _results_frame() -> Any:
    return pd.DataFrame({
        "dataset_id": [DATASET, DATASET, DATASET, "other", DATASET],
        "model_id": ["A", "B", "C", "Z", "L"],
        "macro_f1": [0.7, 0.9, 0.8, 1.0, 0.95],
        "macro_f1_cov": [0.7, 0.9, 0.8, 1.0, None],
    })


def test_select_top_models_ranks_and_filters() -> None:
    df = _results_frame()
    assert select_top_models(df, DATASET, 2) == ["L", "B"]
    assert select_top_models(df, DATASET, 2, exclude=["L"]) == ["B", "C"]
    # Rows without a value for the metric are skipped.
    assert select_top_models(df, DATASET, 10, metric="macro_f1_cov") == ["B", "C", "A"]


def test_select_top_models_validates_args() -> None:
    df = _results_frame()
    with pytest.raises(ValueError, match="k must be"):
        select_top_models(df, DATASET, 0)
    with pytest.raises(ValueError, match="unknown metric"):
        select_top_models(df, DATASET, 1, metric="nope")


def test_load_predictions_local(results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A", "B", "D"], local_dir=results_dir)
    assert list(preds.columns) == ["model_id", "idx", "text_hash", "gold", "pred"]
    assert list(dict.fromkeys(preds["model_id"])) == ["A", "B"]
    assert len(preds) == 2 * len(TEXTS)
    # JSON null stays None (not NaN) so it maps to und downstream.
    assert preds.loc[(preds["model_id"] == "A") & (preds["idx"] == 4), "pred"].item() is None


def test_load_predictions_empty(tmp_path: Path) -> None:
    preds = load_predictions(DATASET, ["A"], local_dir=tmp_path)
    assert preds.empty
    assert sample_difficulty(preds).empty
    assert language_difficulty(preds).empty


def test_load_predictions_hub(monkeypatch: pytest.MonkeyPatch, results_dir: Path) -> None:
    calls: dict[str, Any] = {}

    def fake_snapshot(repo_id: str, **kwargs: Any) -> str:
        calls["repo_id"] = repo_id
        calls.update(kwargs)
        return str(results_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    preds = load_predictions(DATASET, ["A"], repo_id="org/results", revision="main")
    assert len(preds) == len(TEXTS)
    assert calls["repo_id"] == "org/results"
    assert calls["revision"] == "main"
    assert calls["allow_patterns"] == [f"{DATASET}/A/predictions.jsonl"]


def test_sample_difficulty(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    s = sample_difficulty(preds).set_index("idx")

    assert list(s.columns[-3:]) == ["pred:A", "pred:B", "pred:C"]
    assert s.loc[0, "n_correct"] == 3
    assert s.loc[0, "error_rate"] == 0.0
    assert s.loc[0, "top_wrong_pred"] is None
    assert not s.loc[0, "label_suspect"]

    # Unanimous alternative label -> annotation-error candidate.
    assert s.loc[1, "error_rate"] == 1.0
    assert s.loc[1, "consensus_pred"] == "nld"
    assert s.loc[1, "consensus_share"] == 1.0
    assert s.loc[1, "label_suspect"]

    # Disagreement -> hard but not suspect; ties broken alphabetically.
    assert s.loc[2, "n_wrong"] == 2
    assert s.loc[2, "top_wrong_pred"] == "ita"
    assert not s.loc[2, "label_suspect"]

    # None counts as und and und consensus is never suspect.
    assert s.loc[4, "consensus_pred"] == "und"
    assert s.loc[4, "n_wrong"] == 2
    assert not s.loc[4, "label_suspect"]
    assert pd.isna(s.loc[4, "pred:A"])

    # Sorted hardest first.
    ordered = sample_difficulty(preds)
    assert list(ordered["idx"][:2]) == [1, 3]
    assert ordered["error_rate"].is_monotonic_decreasing


def test_sample_difficulty_agreement_threshold(results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A", "B"], local_dir=results_dir)
    lax = sample_difficulty(preds, suspect_min_agreement=0.5).set_index("idx")
    # A=ita, B=spa: each 50% of the models agree on a wrong label.
    assert lax.loc[2, "label_suspect"]
    strict = sample_difficulty(preds, suspect_min_agreement=1.0).set_index("idx")
    assert not strict.loc[2, "label_suspect"]
    assert strict.loc[1, "label_suspect"]


def test_single_model_is_never_suspect(results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A"], local_dir=results_dir)
    assert not sample_difficulty(preds)["label_suspect"].any()


def test_collapse_macrolanguages(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    strict = sample_difficulty(preds).set_index("idx")
    assert strict.loc[3, "error_rate"] == 1.0
    assert strict.loc[3, "label_suspect"]

    collapsed = sample_difficulty(preds, collapse_macrolanguages=True).set_index("idx")
    assert collapsed.loc[3, "error_rate"] == 0.0
    assert not collapsed.loc[3, "label_suspect"]
    # The raw gold label is kept in the output.
    assert collapsed.loc[3, "gold"] == "lvs"


def test_to_macrolanguage() -> None:
    assert to_macrolanguage("lvs") == "lav"
    assert to_macrolanguage("eng") == "eng"
    assert to_macrolanguage("und") == "und"
    assert to_macrolanguage("not-a-code") == "not-a-code"


def test_language_difficulty(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    lang = language_difficulty(preds).set_index("language")

    assert list(lang.columns[-3:]) == ["recall:A", "recall:B", "recall:C"]
    assert lang.loc["eng", "n_samples"] == 2
    # eng: sample 0 all right, sample 4 two of three wrong.
    assert lang.loc["eng", "error_rate"] == pytest.approx((0 + 2 / 3) / 2)
    assert lang.loc["eng", "all_correct_share"] == 0.5
    assert lang.loc["eng", "all_wrong_share"] == 0.0
    assert lang.loc["eng", "top_confusion"] == "und"
    assert lang.loc["eng", "recall:B"] == 1.0
    assert lang.loc["eng", "recall:A"] == 0.5

    assert lang.loc["deu", "n_label_suspect"] == 1
    assert lang.loc["deu", "top_confusion"] == "nld"
    assert lang.loc["deu", "top_confusion_share"] == 1.0

    assert lang.loc["fra", "top_confusion_share"] == 0.5

    ordered = language_difficulty(preds)
    assert ordered["error_rate"].is_monotonic_decreasing
    assert set(language_difficulty(preds, min_samples=2)["language"]) == {"eng"}

    collapsed = language_difficulty(preds, collapse_macrolanguages=True).set_index("language")
    assert collapsed.loc["lvs", "error_rate"] == 0.0
    assert collapsed.loc["lvs", "top_confusion"] is None


def test_language_difficulty_reuses_samples(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    samples = sample_difficulty(preds)
    pd.testing.assert_frame_equal(
        language_difficulty(preds, samples=samples), language_difficulty(preds)
    )


def test_gold_mismatch_warns(results_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    preds.loc[(preds["model_id"] == "B") & (preds["idx"] == 0), "gold"] = "sco"
    sample_difficulty(preds)
    assert "different gold labels" in caplog.text


class _FakeDataset:
    dataset_id = DATASET
    text_column = "text"

    def __init__(self, texts: list[str]) -> None:
        from datasets import Dataset

        self._ds = Dataset.from_dict({"text": texts})

    def load(self) -> Any:
        return self._ds


def test_attach_texts(results_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    samples = sample_difficulty(preds)
    with_text = attach_texts(samples, _FakeDataset(TEXTS))  # type: ignore[arg-type]
    for idx, text in zip(with_text["idx"], with_text["text"], strict=True):
        assert text == TEXTS[idx]

    # A dataset whose text drifted (and is shorter) leaves text empty + warns.
    drifted = [*TEXTS[:2], "changed"]
    partial = attach_texts(samples, _FakeDataset(drifted)).set_index("idx")  # type: ignore[arg-type]
    assert partial.loc[0, "text"] == TEXTS[0]
    assert pd.isna(partial.loc[2, "text"])
    assert pd.isna(partial.loc[4, "text"])
    assert "could not be matched" in caplog.text

    empty = attach_texts(samples.iloc[0:0], _FakeDataset(TEXTS))  # type: ignore[arg-type]
    assert "text" in empty.columns


def test_attach_texts_by_dataset_id(monkeypatch: pytest.MonkeyPatch, results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A"], local_dir=results_dir)
    monkeypatch.setattr(
        "commonlid.core.registry.get_dataset", lambda _dataset_id: _FakeDataset(TEXTS)
    )
    out = attach_texts(sample_difficulty(preds), DATASET)
    assert out["text"].notna().all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(results_dir: Path, *args: str) -> Any:
    return runner.invoke(
        app, ["difficulty", "--dataset", DATASET, "--local-dir", str(results_dir), *args]
    )


def test_cli_language_level_top_k(results_dir: Path) -> None:
    result = _cli(results_dir, "--top-k", "2")
    assert result.exit_code == 0, result.output
    assert "Models (2): A, B" in result.output
    assert "recall:A" in result.output
    assert "recall:C" not in result.output


def test_cli_top_k_skips_models_without_predictions(results_dir: Path) -> None:
    result = _cli(results_dir, "--top-k", "10")
    assert result.exit_code == 0, result.output
    assert "Models (3): A, B, C" in result.output


def test_cli_sample_level_suspect_out(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out" / "suspect.csv"
    result = _cli(
        results_dir, "--level", "sample", "--only-suspect", "--out", str(out), "--show", "0"
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert sorted(df["idx"]) == [1, 3]
    assert df["label_suspect"].all()

    collapsed = _cli(
        results_dir,
        "--level",
        "sample",
        "--only-suspect",
        "--collapse-macrolanguages",
        "--out",
        str(out),
    )
    assert collapsed.exit_code == 0, collapsed.output
    assert list(pd.read_csv(out)["idx"]) == [1]


@pytest.mark.parametrize("suffix", [".jsonl", ".parquet"])
def test_cli_out_formats(results_dir: Path, tmp_path: Path, suffix: str) -> None:
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    out = tmp_path / f"langs{suffix}"
    result = _cli(results_dir, "--model", "A", "--model", "C", "--out", str(out))
    assert result.exit_code == 0, result.output
    df = pd.read_json(out, lines=True) if suffix == ".jsonl" else pd.read_parquet(out)
    assert {"recall:A", "recall:C"} <= set(df.columns)


def test_cli_min_error_rate_and_min_samples(results_dir: Path) -> None:
    result = _cli(results_dir, "--min-error-rate", "0.5", "--min-samples", "1")
    assert result.exit_code == 0, result.output
    assert "3 language row(s)" in result.output


def test_cli_with_text(monkeypatch: pytest.MonkeyPatch, results_dir: Path) -> None:
    monkeypatch.setattr(
        "commonlid.core.registry.get_dataset", lambda _dataset_id: _FakeDataset(TEXTS)
    )
    result = _cli(results_dir, "--level", "sample", "--with-text")
    assert result.exit_code == 0, result.output
    assert "hallo welt" in result.output


def test_cli_with_text_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, results_dir: Path
) -> None:
    def boom(_dataset_id: str) -> Any:
        raise RuntimeError("gated")

    monkeypatch.setattr("commonlid.core.registry.get_dataset", boom)
    result = _cli(results_dir, "--level", "sample", "--with-text")
    assert result.exit_code == 0, result.output
    assert "Could not attach texts" in result.output


def test_cli_errors(results_dir: Path) -> None:
    bad_metric = _cli(results_dir, "--metric", "nope")
    assert bad_metric.exit_code == 2
    assert "unknown metric" in bad_metric.output

    no_models = runner.invoke(
        app, ["difficulty", "--dataset", "missing", "--local-dir", str(results_dir)]
    )
    assert no_models.exit_code == 1
    assert "No models" in no_models.output

    no_preds = _cli(results_dir, "--model", "D")
    assert no_preds.exit_code == 1
    assert "No predictions" in no_preds.output


def test_preview_text() -> None:
    from commonlid.cli import _preview_text

    assert _preview_text(None) == ""
    assert _preview_text("a\nb") == "a b"
    assert _preview_text("x" * 100, width=10) == "x" * 9 + "…"
