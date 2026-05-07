"""Tests for ``commonlid.leaderboard.data.load_results`` (offline path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def _write_summary(
    root: Path,
    dataset_id: str,
    model_id: str,
    *,
    macro_f1: float = 0.5,
    fpr_per_lang: dict[str, float | None] | None = None,
) -> None:
    out = root / dataset_id / model_id
    out.mkdir(parents=True, exist_ok=True)
    fpr_per_lang = fpr_per_lang or {"eng": 0.01, "deu": 0.02}
    summary = {
        "schema_version": 2,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_revision": "abc123",
        "commonlid_version": "test-0.1",
        "timestamp": "2026-05-05T00:00:00",
        "n_samples": 100,
        "n_samples_with_gold": 100,
        "samples_per_second": 1234.5,
        "macro": {
            "precision_gold_only": 0.6,
            "recall_gold_only": 0.55,
            "f1_gold_only": macro_f1,
            "n_languages_gold": len(fpr_per_lang),
            "precision_observed": 0.6,
            "recall_observed": 0.55,
            "f1_observed": macro_f1,
            "n_languages_observed": len(fpr_per_lang),
        },
        "micro": {
            "precision_gold_only": 0.7,
            "recall_gold_only": 0.65,
            "f1_gold_only": 0.67,
            "n_correct_gold": 65,
            "n_gold_samples": 100,
            "n_predictions_gold": 95,
            "precision_observed": 0.7,
            "recall_observed": 0.65,
            "f1_observed": 0.67,
            "n_correct_observed": 65,
            "n_predictions_observed": 95,
        },
        "per_language": {
            lang: {
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "fpr": fpr,
                "gt_count": 10,
                "predictions": 10,
                "correct": 5,
            }
            for lang, fpr in fpr_per_lang.items()
        },
        "extra": {},
    }
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_load_results_walks_local_dir(tmp_path: Path) -> None:
    from commonlid.leaderboard.data import load_results

    _write_summary(tmp_path, "commonlid", "GlotLID", macro_f1=0.65)
    _write_summary(tmp_path, "commonlid_nano", "GlotLID", macro_f1=0.50)
    _write_summary(tmp_path, "flores_dev", "GlotLID", macro_f1=0.95)

    df = load_results(local_dir=tmp_path)
    assert sorted(df["dataset_id"].unique()) == ["commonlid", "commonlid_nano", "flores_dev"]
    assert df["model_id"].nunique() == 1


def test_load_results_filters_visible_datasets(tmp_path: Path) -> None:
    from commonlid.leaderboard.data import load_results

    _write_summary(tmp_path, "commonlid", "GlotLID")
    _write_summary(tmp_path, "flores_dev", "GlotLID")
    df = load_results(local_dir=tmp_path, allowed_datasets=["commonlid"])
    assert list(df["dataset_id"].unique()) == ["commonlid"]


def test_mean_fpr_skips_none_entries(tmp_path: Path) -> None:
    from commonlid.leaderboard.data import load_results

    _write_summary(
        tmp_path,
        "commonlid",
        "GlotLID",
        fpr_per_lang={"eng": 0.10, "deu": 0.20, "fra": None},
    )
    df = load_results(local_dir=tmp_path)
    row = df.iloc[0]
    # Mean of [0.10, 0.20] = 0.15; None skipped.
    assert row["mean_fpr"] == pytest.approx(0.15)


def test_imported_flag_set_when_extra_has_imported_from(tmp_path: Path) -> None:
    from commonlid.leaderboard.data import load_results

    out = tmp_path / "commonlid" / "dspy_azure_GPT-4o"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(
        json.dumps({
            "macro": {
                "f1_gold_only": 0.62,
                "precision_gold_only": 0.0,
                "recall_gold_only": 0.0,
                "n_languages_gold": 1,
                "f1_observed": 0.62,
                "precision_observed": 0.0,
                "recall_observed": 0.0,
                "n_languages_observed": 1,
            },
            "micro": {
                "f1_gold_only": 0.0,
                "precision_gold_only": 0.0,
                "recall_gold_only": 0.0,
                "n_correct_gold": 0,
                "n_predictions_gold": 0,
                "n_gold_samples": 1,
                "f1_observed": 0.0,
                "precision_observed": 0.0,
                "recall_observed": 0.0,
                "n_correct_observed": 0,
                "n_predictions_observed": 0,
            },
            "per_language": {
                "eng": {
                    "f1": 0,
                    "precision": 0,
                    "recall": 0,
                    "fpr": 0,
                    "gt_count": 1,
                    "predictions": 1,
                    "correct": 0,
                }
            },
            "extra": {"imported_from": "/legacy/file.json"},
            "model_id": "dspy_azure_GPT-4o",
            "dataset_id": "commonlid",
            "dataset_revision": "abc",
            "commonlid_version": "legacy-import",
            "timestamp": "",
            "n_samples": 1,
            "n_samples_with_gold": 1,
            "samples_per_second": 0.0,
            "schema_version": 2,
        }),
        encoding="utf-8",
    )
    df = load_results(local_dir=tmp_path)
    assert bool(df.iloc[0]["is_imported"]) is True


def test_load_results_returns_empty_frame_when_no_summaries(tmp_path: Path) -> None:
    from commonlid.leaderboard.data import load_results

    df = load_results(local_dir=tmp_path)
    assert df.empty
    assert "macro_f1" in df.columns


def test_header_links_to_blog_and_paper(tmp_path: Path) -> None:
    """The Blocks header surfaces blog + paper links via module-level URL constants."""
    pytest.importorskip("gradio")
    from commonlid.leaderboard import app as app_module

    _write_summary(tmp_path, "commonlid", "GlotLID", macro_f1=0.65)
    demo = app_module.build_app(repo_id="some-org/some-results", local_dir=tmp_path)
    markdown_values = [
        b.value for b in demo.blocks.values() if isinstance(getattr(b, "value", None), str)
    ]
    text = "\n".join(markdown_values)
    assert app_module.BLOG_URL in text
    assert app_module.PAPER_URL in text
    # Footer links to the HF dataset page (lives in its own Markdown block).
    assert any(
        "huggingface.co/datasets/some-org/some-results" in v and "Source:" in v
        for v in markdown_values
    )
    # The constants point at the real docs.
    assert "commoncrawl.org/blog/commonlid" in app_module.BLOG_URL
    assert "arxiv.org/abs/2601.18026" in app_module.PAPER_URL


def test_dataset_metadata_markdown_uses_registry() -> None:
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _dataset_metadata_markdown, _tab_label

    md = _dataset_metadata_markdown("commonlid")
    assert "CommonLID" in md
    # License is rendered as ``License: `<name>`` and linked to license_url.
    assert "License: [`common-crawl-tou`](https://commoncrawl.org/terms-of-use)" in md
    assert "Reference" in md
    assert "Main score: `macro_f1`" in md
    assert _tab_label("commonlid") == "CommonLID"


def test_dataset_metadata_markdown_unlinked_license() -> None:
    """A dataset without ``license_url`` shows just the name without a link."""
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _dataset_metadata_markdown

    md = _dataset_metadata_markdown("bibles_300")
    assert "License: `not specified`" in md
    assert "License: [" not in md  # no link form for an unlinked license


def test_dataset_metadata_markdown_unknown_dataset_falls_back_gracefully() -> None:
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _dataset_metadata_markdown, _tab_label

    md = _dataset_metadata_markdown("does_not_exist_xyz")
    assert "does_not_exist_xyz" in md
    assert "unavailable" in md.lower()
    assert _tab_label("does_not_exist_xyz") == "does_not_exist_xyz"


def test_format_table_sorts_by_macro_f1_desc(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _format_table
    from commonlid.leaderboard.data import load_results

    _write_summary(tmp_path, "commonlid", "AAA", macro_f1=0.30)
    _write_summary(tmp_path, "commonlid", "BBB", macro_f1=0.90)
    _write_summary(tmp_path, "commonlid", "CCC", macro_f1=0.60)
    df = load_results(local_dir=tmp_path)
    table = _format_table(df)
    assert list(table["Model"]) == ["BBB", "CCC", "AAA"]
    # Numeric values are rendered as fixed-decimal strings for consistent
    # alignment (``0.0`` vs ``0`` etc.). Sort order still reflects raw floats.
    assert list(table["Macro F1"]) == ["90.0", "60.0", "30.0"]
    # Sample / version columns were dropped from the headline table.
    assert "Samples" not in table.columns
    assert "Version" not in table.columns


def test_format_table_rounds_to_one_decimal_except_fpr(tmp_path: Path) -> None:
    """Macro/Micro/Samples-per-second round to 1 decimal, FPR rounds to 2."""
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _format_table
    from commonlid.leaderboard.data import load_results

    _write_summary(
        tmp_path,
        "commonlid",
        "M",
        macro_f1=0.123456,
        fpr_per_lang={"eng": 0.123456, "deu": 0.123456},
    )
    df = load_results(local_dir=tmp_path)
    table = _format_table(df)
    row = table.iloc[0]
    # Macro/Micro: 1 decimal, expressed as percentages, formatted as strings.
    assert row["Macro F1"] == "12.3"
    # Mean FPR: 2 decimals.
    assert row["Mean FPR (%)"] == "12.35"


def test_styled_value_right_aligns_non_first_columns() -> None:
    """``_styled_value`` emits per-cell CSS that right-aligns numeric columns."""
    pytest.importorskip("gradio")
    import pandas as pd

    from commonlid.leaderboard.app import _styled_value

    df = pd.DataFrame(
        [
            ["GlotLID", "90.0", "0.05"],
            ["cld2", "45.0", "0.10"],
        ],
        columns=["Model", "Macro F1", "Mean FPR (%)"],
    )

    out = _styled_value(df)
    assert out["headers"] == ["Model", "Macro F1", "Mean FPR (%)"]
    assert out["data"][0] == ["GlotLID", "90.0", "0.05"]
    styling = out["metadata"]["styling"]
    assert len(styling) == 2  # one row per data row
    # First column (Model) is left-aligned (empty string); others are right.
    assert styling[0] == ["", "text-align: right", "text-align: right"]
    assert styling[1] == ["", "text-align: right", "text-align: right"]


def test_format_table_pads_zero_values_with_decimals(tmp_path: Path) -> None:
    """A 0.0 macro F1 must render as ``0.0`` (not ``0``); 0 FPR as ``0.00``."""
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _format_table
    from commonlid.leaderboard.data import load_results

    _write_summary(
        tmp_path,
        "commonlid",
        "Z",
        macro_f1=0.0,
        fpr_per_lang={"eng": 0.0, "deu": 0.0},
    )
    df = load_results(local_dir=tmp_path)
    row = _format_table(df).iloc[0]
    assert row["Macro F1"] == "0.0"
    assert row["Mean FPR (%)"] == "0.00"
