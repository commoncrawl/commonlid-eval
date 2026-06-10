"""Tests for ``commonlid.leaderboard.data.load_results`` (offline path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")


_UNSET = object()


def _write_summary(
    root: Path,
    dataset_id: str,
    model_id: str,
    *,
    macro_f1: float = 0.5,
    fpr_per_lang: dict[str, float | None] | None = None,
    per_language: dict[str, dict[str, Any]] | None = None,
    supported_languages: object = _UNSET,
) -> None:
    out = root / dataset_id / model_id
    out.mkdir(parents=True, exist_ok=True)
    fpr_per_lang = fpr_per_lang or {"eng": 0.01, "deu": 0.02}
    summary: dict[str, Any] = {
        "schema_version": 3,
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
        "per_language": per_language
        or {
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
    if supported_languages is not _UNSET:
        summary["supported_languages"] = supported_languages
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


# ----- (cov.) variant ---------------------------------------------------------


def _per_language_block(
    spec: dict[str, tuple[int, int, int, float]],
) -> dict[str, dict[str, Any]]:
    """Build a per_language dict from a compact (gt, pred, correct, fpr) spec."""
    return {
        lang: {
            "gt_count": gt,
            "predictions": pred,
            "correct": correct,
            "precision": (correct / pred) if pred else 0.0,
            "recall": (correct / gt) if gt else 0.0,
            "f1": (
                2 * (correct / pred) * (correct / gt) / ((correct / pred) + (correct / gt))
                if pred and gt and correct
                else 0.0
            ),
            "fpr": fpr,
        }
        for lang, (gt, pred, correct, fpr) in spec.items()
    }


def test_row_cov_fields_filter_to_supported_languages(tmp_path: Path) -> None:
    """With supported={eng,fra}, cov metrics exclude deu (an unsupported gold lang)."""
    from commonlid.leaderboard.data import load_results

    per_lang = _per_language_block({
        "eng": (10, 10, 10, 0.01),  # perfect
        "fra": (10, 10, 5, 0.02),  # P=0.5 R=0.5 F1=0.5
        "deu": (10, 10, 0, 0.03),  # zero (excluded from cov)
    })
    _write_summary(
        tmp_path,
        "commonlid",
        "M",
        per_language=per_lang,
        supported_languages=["eng", "fra"],
    )
    df = load_results(local_dir=tmp_path)
    row = df.iloc[0]
    # Macro F1 cov = mean(F1 over {eng, fra}) = (1.0 + 0.5) / 2 = 0.75; "all" includes deu and drops.
    assert row["macro_f1_cov"] == pytest.approx(0.75)
    assert row["macro_f1_cov"] > row["macro_f1"]
    # Micro F1 cov pools eng+fra: P = 15/20 = 0.75, R = 15/20 = 0.75, F1 = 0.75.
    assert row["micro_f1_cov"] == pytest.approx(0.75)
    # n_languages_cov is the supported-and-have-gold intersection size.
    assert row["n_languages_cov"] == 2
    # supported_languages round-trips as a sorted list.
    assert row["supported_languages"] == ["eng", "fra"]


@pytest.mark.parametrize(
    ("kind", "supported"),
    [
        ("missing", _UNSET),  # legacy file, no key
        ("null", None),  # LLM row, undefined support set
        ("empty_list", []),  # degenerate "supports zero languages"
    ],
)
def test_row_cov_fields_none_when_no_support_data(
    tmp_path: Path, kind: str, supported: object
) -> None:
    from commonlid.leaderboard.data import load_results

    _write_summary(
        tmp_path,
        "commonlid",
        f"M_{kind}",
        supported_languages=supported,
    )
    df = load_results(local_dir=tmp_path)
    row = df.iloc[0]
    assert row["macro_f1_cov"] is None
    assert row["macro_precision_cov"] is None
    assert row["macro_recall_cov"] is None
    assert row["micro_f1_cov"] is None
    assert row["mean_fpr_cov"] is None
    assert row["n_languages_cov"] is None
    # supported_languages is preserved as-is in the row (None for both
    # missing and null) so the UI can choose to show a tooltip later.
    expected = None if supported is _UNSET or supported is None else supported
    assert row["supported_languages"] == expected


def test_row_cov_fields_none_when_supported_set_misses_all_gold(tmp_path: Path) -> None:
    """If the model's support set has no overlap with the dataset's gold, cov collapses to None."""
    from commonlid.leaderboard.data import load_results

    per_lang = _per_language_block({"eng": (10, 10, 8, 0.01)})
    _write_summary(
        tmp_path,
        "commonlid",
        "M",
        per_language=per_lang,
        supported_languages=["xyz", "qqq"],  # no overlap
    )
    df = load_results(local_dir=tmp_path)
    row = df.iloc[0]
    assert row["macro_f1_cov"] is None
    assert row["n_languages_cov"] is None


def test_format_table_cov_scope_renders_em_dashes(tmp_path: Path) -> None:
    """Rows without ``supported_languages`` show em-dashes and sort to the bottom."""
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import _format_table
    from commonlid.leaderboard.data import load_results

    per_lang = _per_language_block({
        "eng": (10, 10, 10, 0.01),
        "fra": (10, 10, 5, 0.02),
    })
    # Model with a declared support set -> real cov numbers.
    _write_summary(
        tmp_path,
        "commonlid",
        "WITH_SUPPORT",
        per_language=per_lang,
        supported_languages=["eng", "fra"],
    )
    # LLM-style row -> JSON null -> em-dashes in cov view.
    _write_summary(
        tmp_path,
        "commonlid",
        "NO_SUPPORT",
        per_language=per_lang,
        supported_languages=None,
    )
    df = load_results(local_dir=tmp_path)
    cov_table = _format_table(df, scope="cov")
    # Real-cov row should sort above the em-dash row.
    assert list(cov_table["Model"]) == ["WITH_SUPPORT", "NO_SUPPORT"]
    assert cov_table.iloc[0]["Macro F1"] == "75.0"
    assert cov_table.iloc[1]["Macro F1"] == "—"
    assert cov_table.iloc[1]["Languages"] == "—"
    # Samples/s is unaffected by the toggle (it's a model property).
    assert cov_table.iloc[0]["Samples/s"] == "1234.5"
    assert cov_table.iloc[1]["Samples/s"] == "1234.5"


def test_row_select_handler_loads_drilldown_from_row_value(tmp_path: Path) -> None:
    """Clicking a row resolves model_id via ``evt.row_value[0]`` and renders the drilldown.

    Regression: an earlier version pulled the model_id from the Dataframe
    input, but Gradio 6 preprocesses Dataframe inputs into ``pandas.DataFrame``
    objects rather than the ``{"data", "headers"}`` dict the app feeds in,
    so the handler silently returned the "click a row" placeholder.
    """
    pytest.importorskip("gradio")
    from types import SimpleNamespace

    from commonlid.leaderboard.app import _make_select_handler

    per_lang = _per_language_block({"eng": (10, 10, 8, 0.01), "fra": (10, 10, 4, 0.02)})
    _write_summary(tmp_path, "commonlid", "GlotLID", per_language=per_lang)
    handler = _make_select_handler("commonlid", tmp_path)

    evt = SimpleNamespace(
        index=(0, 0),
        value="GlotLID",
        row_value=["GlotLID", "80.0", "60.0", "0.10", "2", "1234.5"],
    )
    label, payload = handler(evt)
    assert "GlotLID" in label
    assert "commonlid" in label
    assert payload is not None
    headers = payload["headers"]
    assert headers[:2] == ["Language", "F1"]
    languages = [row[0] for row in payload["data"]]
    assert set(languages) == {"eng", "fra"}


def test_row_select_handler_returns_placeholder_when_index_missing(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from types import SimpleNamespace

    from commonlid.leaderboard.app import _make_select_handler

    handler = _make_select_handler("commonlid", tmp_path)
    evt = SimpleNamespace(index=None, value=None, row_value=None)
    label, payload = handler(evt)
    assert "Click a row" in label
    assert payload is None


def test_scope_radio_change_swaps_table_and_legend(tmp_path: Path) -> None:
    """The scope-change handler returns a fresh styled table + legend Markdown."""
    pytest.importorskip("gradio")
    from commonlid.leaderboard.app import (
        _HEADLINE_COLUMN_HELP_COV,
        _columns_help_markdown,
        _make_scope_handler,
    )
    from commonlid.leaderboard.data import load_results

    per_lang = _per_language_block({"eng": (10, 10, 10, 0.01), "fra": (10, 10, 5, 0.02)})
    _write_summary(
        tmp_path,
        "commonlid",
        "M",
        per_language=per_lang,
        supported_languages=["eng", "fra"],
    )
    df = load_results(local_dir=tmp_path)
    handler = _make_scope_handler(df)
    table_payload, legend = handler("cov")
    assert legend == _columns_help_markdown(_HEADLINE_COLUMN_HELP_COV)
    assert table_payload["headers"][1] == "Macro F1"
    # Row data is the cov computation, not the "all" one.
    assert table_payload["data"][0][1] == "75.0"
