"""Tests for ``commonlid.analysis`` (top-k model breakdown, offline)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from commonlid.analysis import (
    attach_texts,
    labels_match,
    language_breakdown,
    load_predictions,
    model_breakdown,
    sample_breakdown,
    select_top_models,
    to_macrolanguage,
)
from commonlid.cli import app
from commonlid.evaluation.results import text_hash
from commonlid.metrics.aggregate import macro_average
from commonlid.metrics.core import compute_per_language_metrics

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
# Only A recorded raw outputs.
RAW: dict[str, list[str | None]] = {"A": ["en", "nl", "it", "lv", None]}


def _write_run(root: Path, model_id: str, preds: list[str | None], macro_f1: float) -> None:
    out = root / DATASET / model_id
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "idx": i,
            "text_hash": text_hash(text),
            "gold": gold,
            "pred": pred,
            "score": 0.5,
            "correct": gold == pred,
        }
        for i, (text, gold, pred) in enumerate(zip(TEXTS, GOLDS, preds, strict=True))
    ]
    for row, raw in zip(rows, RAW.get(model_id, []), strict=False):
        row["pred_raw"] = raw
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
    assert list(preds.columns) == [
        "model_id",
        "idx",
        "text_hash",
        "gold",
        "pred",
        "pred_raw",
        "score",
    ]
    assert list(dict.fromkeys(preds["model_id"])) == ["A", "B"]
    assert len(preds) == 2 * len(TEXTS)
    # JSON null stays None (not NaN) so it maps to und downstream.
    assert preds.loc[(preds["model_id"] == "A") & (preds["idx"] == 4), "pred"].item() is None
    # pred_raw is kept where recorded and None where the file has no such key.
    assert preds.loc[(preds["model_id"] == "A") & (preds["idx"] == 1), "pred_raw"].item() == "nl"
    assert preds.loc[preds["model_id"] == "B", "pred_raw"].isna().all()


def test_load_predictions_empty(tmp_path: Path) -> None:
    preds = load_predictions(DATASET, ["A"], local_dir=tmp_path)
    assert preds.empty
    assert sample_breakdown(preds).empty
    assert language_breakdown(preds).empty


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


def test_sample_breakdown(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    s = sample_breakdown(preds).set_index("idx")

    assert list(s.columns[-3:]) == ["pred:A", "pred:B", "pred:C"]
    assert not s["gold_conflict"].any()
    assert s.loc[0, "n_correct"] == 3
    assert s.loc[0, "error_rate"] == 0.0
    assert s.loc[0, "top_wrong_pred"] is None
    assert not s.loc[0, "label_suspect"]

    # Unanimous alternative label -> annotation-error candidate.
    assert s.loc[1, "error_rate"] == 1.0
    assert s.loc[1, "consensus_pred"] == "nld"
    assert s.loc[1, "agreement"] == 1.0
    assert s.loc[1, "label_suspect"]

    # Disagreement -> hard but not suspect; ties broken alphabetically.
    assert s.loc[2, "n_wrong"] == 2
    assert s.loc[2, "top_wrong_pred"] == "ita"
    assert not s.loc[2, "label_suspect"]

    # None counts as und and und consensus is never suspect.
    assert s.loc[4, "consensus_pred"] == "und"
    assert s.loc[4, "n_wrong"] == 2
    assert not s.loc[4, "label_suspect"]
    assert s.loc[4, "pred:A"] is None

    # Sorted by error_rate, highest first.
    ordered = sample_breakdown(preds)
    assert list(ordered["idx"][:2]) == [1, 3]
    assert ordered["error_rate"].is_monotonic_decreasing


def test_sample_breakdown_agreement_threshold(results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A", "B"], local_dir=results_dir)
    lax = sample_breakdown(preds, suspect_min_agreement=0.5).set_index("idx")
    # A=ita, B=spa: each 50% of the models agree on a wrong label.
    assert lax.loc[2, "label_suspect"]
    strict = sample_breakdown(preds, suspect_min_agreement=1.0).set_index("idx")
    assert not strict.loc[2, "label_suspect"]
    assert strict.loc[1, "label_suspect"]


def test_single_model_is_never_suspect(results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A"], local_dir=results_dir)
    assert not sample_breakdown(preds)["label_suspect"].any()


def test_macro_match(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    strict = sample_breakdown(preds).set_index("idx")
    assert strict.loc[3, "error_rate"] == 1.0
    assert strict.loc[3, "label_suspect"]

    lenient = sample_breakdown(preds, macro_match="macro").set_index("idx")
    assert lenient.loc[3, "error_rate"] == 0.0
    assert not lenient.loc[3, "label_suspect"]
    # Correct predictions count as the gold label; the published ones are kept.
    assert lenient.loc[3, "consensus_pred"] == "lvs"
    assert lenient.loc[3, "pred:A"] == "lav"
    assert lenient.loc[3, "gold"] == "lvs"


@pytest.mark.parametrize(
    ("gold", "pred", "exact", "macro", "sibling"),
    [
        ("eng", "eng", True, True, True),
        ("lvs", "lav", False, True, True),
        ("zho", "cmn", False, True, True),
        # Two individual languages of the same macrolanguage.
        ("arz", "arb", False, False, True),
        ("cmn", "yue", False, False, True),
        ("ind", "zsm", False, False, True),
        ("eng", "deu", False, False, False),
        ("und", "und", True, True, True),
        ("eng", "und", False, False, False),
    ],
)
def test_labels_match(gold: str, pred: str, exact: bool, macro: bool, sibling: bool) -> None:
    assert labels_match(gold, pred, "exact") is exact
    assert labels_match(gold, pred, "macro") is macro
    assert labels_match(gold, pred, "sibling") is sibling
    assert labels_match(pred, gold, "macro") is macro


def test_macro_match_validates_mode(results_dir: Path) -> None:
    preds = load_predictions(DATASET, ["A"], local_dir=results_dir)
    with pytest.raises(ValueError, match="macro_match must be one of"):
        sample_breakdown(preds, macro_match="loose")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="macro_match must be one of"):
        labels_match("eng", "deu", "loose")  # type: ignore[arg-type]


def _duplicate_text(preds: Any, src: int, dst: int, gold: str) -> Any:
    """Make sample ``dst`` the same text as ``src`` with gold label ``gold``."""
    preds = preds.copy()
    src_hash = preds.loc[preds["idx"] == src, "text_hash"].iloc[0]
    preds.loc[preds["idx"] == dst, "text_hash"] = src_hash
    preds.loc[preds["idx"] == dst, "gold"] = gold
    return preds


def test_gold_conflict(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    # Sample 4 becomes a copy of sample 0 ("hello world") labelled sco.
    s = sample_breakdown(_duplicate_text(preds, 0, 4, "sco")).set_index("idx")
    assert s.loc[0, "gold_conflict"]
    assert s.loc[0, "other_golds"] == "sco"
    assert s.loc[4, "gold_conflict"]
    assert s.loc[4, "other_golds"] == "eng"
    assert not s.loc[1, "gold_conflict"]
    assert s.loc[1, "other_golds"] is None

    lang = language_breakdown(_duplicate_text(preds, 0, 4, "sco")).set_index("language")
    assert lang.loc["eng", "n_gold_conflict"] == 1
    assert lang.loc["sco", "n_gold_conflict"] == 1

    # The same label twice is a plain duplicate, not a conflict.
    same = sample_breakdown(_duplicate_text(preds, 0, 4, "eng"))
    assert not same["gold_conflict"].any()

    # lvs and its macrolanguage lav only conflict under exact matching.
    macro_dup = _duplicate_text(preds, 3, 4, "lav")
    assert sample_breakdown(macro_dup)["gold_conflict"].sum() == 2
    assert not sample_breakdown(macro_dup, macro_match="macro")["gold_conflict"].any()


def test_misaligned_predictions_raise(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    preds.loc[(preds["model_id"] == "B") & (preds["idx"] == 2), "text_hash"] = "0" * 16
    with pytest.raises(ValueError, match="do not line up"):
        sample_breakdown(preds)


def test_include_raw(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    assert "raw:A" not in sample_breakdown(preds).columns
    s = sample_breakdown(preds, include_raw=True).set_index("idx")
    assert list(s.columns[-3:]) == ["raw:A", "raw:B", "raw:C"]
    assert s.loc[1, "raw:A"] == "nl"
    assert s.loc[1, "raw:B"] is None


def test_partial_models_are_logged(results_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    preds = preds[~((preds["model_id"] == "C") & (preds["idx"] == 0))]
    with caplog.at_level(logging.INFO, logger="commonlid.analysis.breakdown"):
        s = sample_breakdown(preds).set_index("idx")
    assert s.loc[0, "n_models"] == 2
    assert "1 of 5 sample(s) lack a prediction" in caplog.text


def test_model_breakdown(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    exact = model_breakdown(preds).set_index("model_id")
    # Exact matching reproduces the official metric code on the raw predictions.
    a = preds[preds["model_id"] == "A"]
    expected = macro_average(compute_per_language_metrics(list(a["gold"]), list(a["pred"])))
    assert exact.loc["A", "macro_f1"] == pytest.approx(expected["f1_gold_only"])
    assert (exact["macro_f1"] == exact["macro_f1_exact"]).all()
    assert (exact["macro_f1_gain"] == 0).all()
    assert (exact["rank"] == exact["rank_exact"]).all()
    assert exact.loc["A", "n_samples"] == len(TEXTS)

    lenient = model_breakdown(preds, macro_match="macro").set_index("model_id")
    # Every model predicted lav for the lvs sample, which now counts.
    assert (lenient["macro_f1_gain"] > 0).all()
    pd.testing.assert_series_equal(lenient["macro_f1_exact"], exact["macro_f1_exact"])
    assert lenient.loc["C", "rank"] == 1
    assert model_breakdown(preds, macro_match="macro")["macro_f1"].is_monotonic_decreasing
    assert model_breakdown(preds.iloc[0:0]).empty


def test_to_macrolanguage() -> None:
    assert to_macrolanguage("lvs") == "lav"
    assert to_macrolanguage("eng") == "eng"
    assert to_macrolanguage("und") == "und"
    assert to_macrolanguage("not-a-code") == "not-a-code"


def test_language_breakdown(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    lang = language_breakdown(preds).set_index("language")

    assert list(lang.columns[-3:]) == ["recall:A", "recall:B", "recall:C"]
    assert lang.loc["eng", "n_samples"] == 2
    # eng: sample 0 all right, sample 4 two of three wrong.
    assert lang.loc["eng", "error_rate"] == pytest.approx((0 + 2 / 3) / 2)
    assert lang.loc["eng", "all_correct_share"] == 0.5
    assert lang.loc["eng", "all_wrong_share"] == 0.0
    assert lang.loc["eng", "top_confusion"] == "und"
    assert lang.loc["eng", "recall:B"] == 1.0
    assert lang.loc["eng", "recall:A"] == 0.5
    # eng: sample 0 unanimous, sample 4 two of three agree on und.
    assert lang.loc["eng", "agreement"] == pytest.approx((1 + 2 / 3) / 2)
    assert lang.loc["fra", "agreement"] == pytest.approx(1 / 3)

    assert lang.loc["deu", "n_label_suspect"] == 1
    assert lang.loc["deu", "top_confusion"] == "nld"
    assert lang.loc["deu", "top_confusion_share"] == 1.0

    assert lang.loc["fra", "top_confusion_share"] == 0.5

    ordered = language_breakdown(preds)
    assert ordered["error_rate"].is_monotonic_decreasing
    assert set(language_breakdown(preds, min_samples=2)["language"]) == {"eng"}

    lenient = language_breakdown(preds, macro_match="macro").set_index("language")
    assert lenient.loc["lvs", "error_rate"] == 0.0
    assert lenient.loc["lvs", "top_confusion"] is None


def test_language_breakdown_reuses_samples(results_dir: Path) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    samples = sample_breakdown(preds)
    pd.testing.assert_frame_equal(
        language_breakdown(preds, samples=samples), language_breakdown(preds)
    )


def test_gold_mismatch_warns(results_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    preds = load_predictions(DATASET, list(PREDS), local_dir=results_dir)
    preds.loc[(preds["model_id"] == "B") & (preds["idx"] == 0), "gold"] = "sco"
    sample_breakdown(preds)
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
    samples = sample_breakdown(preds)
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
    out = attach_texts(sample_breakdown(preds), DATASET)
    assert out["text"].notna().all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(results_dir: Path, *args: str) -> Any:
    return runner.invoke(
        app, ["breakdown", "--dataset", DATASET, "--local-dir", str(results_dir), *args]
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

    lenient = _cli(
        results_dir,
        "--level",
        "sample",
        "--only-suspect",
        "--macro-match",
        "macro",
        "--out",
        str(out),
    )
    assert lenient.exit_code == 0, lenient.output
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


def test_cli_sort_and_max_error_rate(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "sorted.csv"
    result = _cli(results_dir, "--order", "asc", "--out", str(out))
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert df["error_rate"].is_monotonic_increasing
    assert df["language"].iloc[0] == "eng"

    by_agreement = _cli(results_dir, "--sort-by", "agreement", "--order", "asc", "--out", str(out))
    assert by_agreement.exit_code == 0, by_agreement.output
    df = pd.read_csv(out)
    assert df["agreement"].is_monotonic_increasing
    assert df["language"].iloc[0] == "fra"

    bad = _cli(results_dir, "--sort-by", "nope")
    assert bad.exit_code == 2
    assert "unknown --sort-by" in bad.output

    capped = _cli(results_dir, "--level", "sample", "--max-error-rate", "0", "--out", str(out))
    assert capped.exit_code == 0, capped.output
    assert list(pd.read_csv(out)["idx"]) == [0]


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
        app, ["breakdown", "--dataset", "missing", "--local-dir", str(results_dir)]
    )
    assert no_models.exit_code == 1
    assert "No models" in no_models.output

    no_preds = _cli(results_dir, "--model", "D")
    assert no_preds.exit_code == 1
    assert "No predictions" in no_preds.output


def test_cli_model_level(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "models.csv"
    result = _cli(results_dir, "--level", "model", "--macro-match", "macro", "--out", str(out))
    assert result.exit_code == 0, result.output
    assert "3 model row(s)" in result.output
    df = pd.read_csv(out)
    assert df["macro_f1"].is_monotonic_decreasing
    assert {"rank", "rank_exact", "macro_f1_exact", "macro_f1_gain"} <= set(df.columns)


def test_cli_only_gold_conflict_and_raw(results_dir: Path, tmp_path: Path) -> None:
    # Make sample 4 ("???") the same text as sample 0 in every model's file.
    for model_id in PREDS:
        path = results_dir / DATASET / model_id / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[4]["text_hash"] = rows[0]["text_hash"]
        rows[4]["gold"] = "sco"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "conflicts.csv"
    result = _cli(
        results_dir, "--level", "sample", "--only-gold-conflict", "--with-raw", "--out", str(out)
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert sorted(df["idx"]) == [0, 4]
    assert "raw:A" in df.columns


def test_cli_misaligned_predictions(results_dir: Path) -> None:
    path = results_dir / DATASET / "B" / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[2]["text_hash"] = "0" * 16
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = _cli(results_dir)
    assert result.exit_code == 1
    assert "do not line up" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--only-suspect"], "--only-suspect only applies to --level sample"),
        (["--with-text"], "--with-text only applies to --level sample"),
        (["--level", "model", "--with-raw"], "--with-raw only applies to --level sample"),
        (["--level", "sample", "--min-samples", "2"], "--min-samples only applies"),
        (["--level", "model", "--max-error-rate", "0.5"], "--max-error-rate only applies"),
    ],
)
def test_cli_rejects_options_for_other_levels(
    results_dir: Path, args: list[str], message: str
) -> None:
    result = _cli(results_dir, *args)
    assert result.exit_code == 2
    assert message in " ".join(re.sub(r"\x1b\[[0-9;]*m|[│╭╮╰╯─]", " ", result.output).split())


def test_preview_text() -> None:
    from commonlid.cli import _preview_text

    assert _preview_text(None) == ""
    assert _preview_text("a\nb") == "a b"
    assert _preview_text("x" * 100, width=10) == "x" * 9 + "…"
