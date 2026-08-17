"""Gradio leaderboard app for ``commonlid`` + ``commonlid_nano`` results.

The app loads results from a published HF dataset (default
``commoncrawl/commonlid-results``) at launch time, builds a tidy
``pandas.DataFrame``, and renders one tab per benchmark with a sortable
summary table and a per-row drilldown showing per-language F1 and FPR.

Use :func:`build_app` from any host (HF Space, local script, ``commonlid
leaderboard`` CLI). ``gradio`` and ``pandas`` come from the ``[leaderboard]``
optional extra; importing this module without them raises ``ImportError``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

# Module-level so ``typing.get_type_hints`` (used by Gradio's event
# introspection) can resolve ``gr.SelectData`` annotations on closures —
# importing inside ``_make_select_handler`` puts ``gr`` only in that
# function's locals, not the module globals where get_type_hints looks.
import gradio as gr

from commonlid.leaderboard.data import (
    DEFAULT_REPO_ID,
    SUMMARY_FILENAME,
    load_results,
)

logger = logging.getLogger(__name__)

#: Datasets the leaderboard surfaces — others (full FLORES+, smolsent, etc.)
#: are kept in the results dataset for archival but hidden from the UI.
VISIBLE_DATASETS: tuple[str, ...] = ("commonlid", "commonlid_nano")

#: Background reading shown in the header.
BLOG_URL = (
    "https://commoncrawl.org/blog/"
    "commonlid-re-evaluating-state-of-the-art-language-identification-performance-on-web-data"
)
PAPER_URL = "https://arxiv.org/abs/2601.18026"

WEBSITE_URL = "https://commonlid.org/"

NEW_MODEL_URL = (
    "https://github.com/commoncrawl/commonlid-eval/blob/main/docs/contributing/adding_a_model.md"
)

Scope = Literal["all", "cov"]

#: Radio choices shown above each dataset's results table.
SCOPE_CHOICES: list[tuple[str, Scope]] = [
    ("Scores are calculated over the whole dataset.", "all"),
    (
        "Scores are calculated on the subset of language varieties covered by the model. (cov.)",
        "cov",
    ),
]

#: Sentinel string used when a row has no cov data (rendered as em-dash).
_NA_DISPLAY = "—"

#: Display columns in the headline table (in order). Macro F1 is the headline metric.
_HEADLINE_COLUMNS: list[tuple[str, str]] = [
    ("model_id", "Model"),
    ("macro_f1", "Macro F1"),
    ("micro_f1", "Micro F1"),
    ("mean_fpr", "Mean FPR (%)"),
    ("n_languages", "Languages"),
    ("samples_per_second", "Samples/s"),
]

#: Same columns, projected from the ``*_cov`` source fields. Display
#: labels stay identical so the table layout doesn't shift when the
#: scope radio is toggled.
_HEADLINE_COLUMNS_COV: list[tuple[str, str]] = [
    ("model_id", "Model"),
    ("macro_f1_cov", "Macro F1"),
    ("micro_f1_cov", "Micro F1"),
    ("mean_fpr_cov", "Mean FPR (%)"),
    ("n_languages_cov", "Languages"),
    ("samples_per_second", "Samples/s"),
]

#: Right-aligned numeric columns get the ``number`` Gradio datatype which
#: pushes values to the right edge of the cell.
_GradioDtype = Literal["str", "number", "bool", "date", "markdown", "html"]
_HEADLINE_DATATYPES: list[_GradioDtype] = [
    "str",  # Model
    "number",  # Macro F1
    "number",  # Micro F1
    "number",  # Mean FPR
    "number",  # Languages
    "number",  # Samples/s
]
#: Drilldown table: Language is text; everything else is numeric (right-aligned).
_DRILLDOWN_DATATYPES: list[_GradioDtype] = [
    "str",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
]


#: Per-column human descriptions for the headline (model summary) table.
_HEADLINE_COLUMN_HELP: list[tuple[str, str]] = [
    ("Model", "Identifier of the language identification model."),
    (
        "Macro F1",
        "Unweighted mean of per-language F1 (x100), averaged over languages with "
        "at least one gold sample in this dataset (paper / gold-only definition). "
        "**Higher is better.** This is the headline ranking column.",
    ),
    (
        "Micro F1",
        "Sample-weighted F1 (x100): pooled correct / pooled predictions across "
        "all gold samples. Less affected by rare languages than macro F1. "
        "**Higher is better.**",
    ),
    (
        "Mean FPR (%)",
        "Mean per-language false-positive rate (paper-style): how often the model "
        "labels a non-target sentence as the target language. **Lower is better.**",
    ),
    (
        "Languages",
        "Number of distinct languages the model emitted on this dataset "
        "(`set(gold) | set(pred)`). Reflects the model's output vocabulary on "
        "this test, not the gold language count.",
    ),
    (
        "Samples/s",
        "Throughput during evaluation (samples processed per second). Hardware-"
        "dependent; useful for relative comparison only.",
    ),
]


#: Per-column human descriptions for the drilldown (per-language) table.
_DRILLDOWN_COLUMN_HELP: list[tuple[str, str]] = [
    ("Language", "ISO 639-3 code of the gold and/or predicted language."),
    ("F1", "Per-language F1 score (x100). Harmonic mean of precision and recall."),
    (
        "Precision",
        "Per-language precision (x100) = correct / predictions for this language. "
        "How often the model is right when it predicts this language.",
    ),
    (
        "Recall",
        "Per-language recall (x100) = correct / gold-count for this language. "
        "How much of this language's gold set the model recovers.",
    ),
    (
        "FPR (%)",
        "Paper-style false-positive rate: FP / (FP + TN_correct_other). Counts "
        "how often samples in *other* languages are misclassified as this one.",
    ),
    ("GT", "Gold-truth sample count for this language."),
    ("Predictions", "Number of times the model predicted this language."),
    ("Correct", "Predictions that match the gold label."),
]


#: Per-column human descriptions for the **(cov.)** view — same metrics,
#: but restricted to the model's declared support set.
_HEADLINE_COLUMN_HELP_COV: list[tuple[str, str]] = [
    ("Model", "Identifier of the language identification model."),
    (
        "Macro F1",
        "Unweighted mean of per-language F1 (x100) **restricted to languages the "
        "model declares it supports** (paper `(cov.)` definition). Languages outside "
        "the model's support set are excluded from the average — a model that covers "
        "a small but accurate subset of the benchmark is no longer penalised for the "
        "long tail of languages it never claimed to handle. **Higher is better.** "
        f"Models without a declared support set show `{_NA_DISPLAY}`.",
    ),
    (
        "Micro F1",
        "Sample-weighted F1 (x100) pooled over the **model-supported subset** of "
        "gold samples only. **Higher is better.** "
        f"`{_NA_DISPLAY}` when no support set is declared.",
    ),
    (
        "Mean FPR (%)",
        "Mean per-language false-positive rate computed only on samples whose gold "
        "language is in the model's support set; TN counts confusion across other "
        "supported languages, not the long tail. **Lower is better.** "
        f"`{_NA_DISPLAY}` when no support set is declared.",
    ),
    (
        "Languages",
        "Number of model-supported languages that have at least one gold sample in "
        "this dataset (`|supported ∩ gold|`). This is the size of the slice every "
        "other `(cov.)` metric is averaged over.",
    ),
    (
        "Samples/s",
        "Throughput during evaluation (samples processed per second). Unaffected by "
        "the scope toggle — it is a model-property, not a metric.",
    ),
]


def _columns_help_markdown(items: list[tuple[str, str]]) -> str:
    """Render a (column, description) list as a Markdown bullet block."""
    return "\n".join(f"- **{label}** — {desc}" for label, desc in items)


#: Per-column decimal precision for display rendering. Columns absent
#: from this map (Model / Language) are rendered by ``str(value)``.
#: Load-bearing: the underlying cell value stays numeric so TanStack's
#: Dataframe sort picks its numeric compare path; the strings in
#: ``metadata.display_value`` only affect what the user sees.
_DISPLAY_DECIMALS: dict[str, int] = {
    "Macro F1": 1,
    "Micro F1": 1,
    "Mean FPR (%)": 2,
    "Languages": 0,
    "Samples/s": 1,
    "F1": 1,
    "Precision": 1,
    "Recall": 1,
    "FPR (%)": 2,
    "GT": 0,
    "Predictions": 0,
    "Correct": 0,
}


def _styled_value(
    table: Any,
    right_align_after_col: int = 0,
    *,
    display: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Wrap a DataFrame in the ``{"data", "headers", "metadata"}`` structure
    Gradio expects, keeping raw numeric ``data`` and shipping formatted
    strings via ``metadata.display_value``.

    Every cell in column ``> right_align_after_col`` gets ``text-align:
    right``; the first column (Model / Language) is left-aligned by default.

    Passing ``display`` (a list-of-lists of formatted display strings shaped
    like ``data``) is the mechanism that keeps the visible cell text as
    fixed-decimal strings while the underlying cell values stay numeric so
    TanStack Table's sort picks its numeric path. When ``display`` is
    ``None`` the display defaults to ``str(value)`` per cell (used only by
    the empty-drilldown seed).

    Documented at
    https://www.gradio.app/guides/styling-the-gradio-dataframe — and Gradio
    6's Dataframe passes the metadata dict through verbatim to
    ``DataframeData`` (see ``gradio/components/dataframe.py`` postprocess).
    """
    headers = list(table.columns)
    # ``.values.tolist()`` can leak pandas NA sentinels from nullable dtypes
    # (Float64/Int64) that don't serialise via json.dumps and confuse the
    # frontend's numeric comparators. Normalise every null-ish scalar to
    # Python ``None`` before it leaves this boundary.
    raw = table.values.tolist()
    data = [[None if _is_nullish(v) else v for v in row] for row in raw]
    styling = [
        ["text-align: right" if col > right_align_after_col else "" for col in range(len(headers))]
        for _ in range(len(data))
    ]
    if display is None:
        display = [[_NA_DISPLAY if v is None else str(v) for v in row] for row in data]
    return {
        "data": data,
        "headers": headers,
        "metadata": {"styling": styling, "display_value": display},
    }


def _is_nullish(value: Any) -> bool:
    """True for ``None``, ``float('nan')``, and pandas ``NA`` (from nullable dtypes)."""
    import pandas as pd

    if value is None:
        return True
    try:
        # ``pd.isna`` handles NaN, pd.NA, NaT — but raises on non-scalars.
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, decimals: int, *, scale: float = 1.0) -> str:
    """Format a numeric value with ``decimals`` precision, em-dash for null-ish values."""
    if _is_nullish(value):
        return _NA_DISPLAY
    return f"{float(value) * scale:.{decimals}f}"


def _display_matrix(table: Any) -> list[list[str]]:
    """Build the ``metadata.display_value`` matrix for a formatted table.

    Each column is looked up in :data:`_DISPLAY_DECIMALS` for its decimal
    precision; unknown columns (Model / Language) pass through as
    ``str(value)``. ``None`` / ``NaN`` renders as an em-dash regardless of
    column, so cov-view "no data" cells look right without a second
    lookup path.
    """
    headers = list(table.columns)
    decimals_by_col: list[int | None] = [_DISPLAY_DECIMALS.get(h) for h in headers]
    rows: list[list[str]] = []
    for row in table.values.tolist():
        rendered: list[str] = []
        for value, decimals in zip(row, decimals_by_col, strict=True):
            if decimals is None:
                rendered.append(_NA_DISPLAY if value is None else str(value))
            else:
                rendered.append(_fmt(value, decimals))
        rows.append(rendered)
    return rows


def _format_table(df: Any, scope: Scope = "all") -> Any:
    """Project a results DataFrame down to the headline columns for one tab.

    Numeric columns stay **numeric** (``float`` / ``int`` / ``None``) so
    the Dataframe frontend's click-sort compares magnitudes rather than
    lexicographic char codes. The visible cell strings (fixed-decimal
    formatting) are produced later by :func:`_display_matrix` and shipped
    through ``metadata.display_value``.

    Macro / Micro / Mean-FPR raw fields live on a 0-1 scale; they are
    scaled to percent here so the numeric cell values match the rendered
    strings (``79.3`` on screen means ``79.3`` under the hood).

    In ``scope="cov"``, rows without ``supported_languages`` data have
    ``None`` in every cov metric — ``sort_values(..., na_position="last")``
    sinks them to the bottom of the default sort.
    """
    import pandas as pd

    columns = _HEADLINE_COLUMNS_COV if scope == "cov" else _HEADLINE_COLUMNS
    display_labels = [label for _, label in columns]
    if df.empty:
        return pd.DataFrame(columns=display_labels)

    out = df.copy()
    sort_key = "macro_f1_cov" if scope == "cov" else "macro_f1"
    out = out.sort_values(sort_key, ascending=False, kind="stable", na_position="last")
    out = out.reset_index(drop=True)

    macro_key = "macro_f1_cov" if scope == "cov" else "macro_f1"
    micro_key = "micro_f1_cov" if scope == "cov" else "micro_f1"
    fpr_key = "mean_fpr_cov" if scope == "cov" else "mean_fpr"

    # Scale 0-1 metrics to percent so the numeric value seen by the sort
    # matches the string the user reads. ``* 100`` on a NaN stays NaN, so
    # missing cov cells remain None/NaN for the na_position sink to catch.
    out[macro_key] = out[macro_key].astype("Float64") * 100
    out[micro_key] = out[micro_key].astype("Float64") * 100
    out[fpr_key] = out[fpr_key].astype("Float64") * 100

    out = out[[k for k, _ in columns]]
    out.columns = display_labels
    return out


_DRILLDOWN_HEADERS: list[str] = [
    "Language",
    "F1",
    "Precision",
    "Recall",
    "FPR (%)",
    "GT",
    "Predictions",
    "Correct",
]


def _per_language_drilldown(snapshot_root: Path, dataset_id: str, model_id: str) -> Any:
    """Return a numeric per-language metrics DataFrame for a (model, dataset), sorted by F1 desc.

    Same principle as :func:`_format_table`: numeric columns stay numeric
    so the Dataframe's click-sort compares magnitudes. The display strings
    (1-decimal F1/Precision/Recall, 2-decimal FPR (%)) are produced via
    :func:`_display_matrix` and shipped through ``metadata.display_value``.
    """
    import pandas as pd

    summary_path = snapshot_root / dataset_id / model_id / SUMMARY_FILENAME
    if not summary_path.is_file():
        return pd.DataFrame(columns=_DRILLDOWN_HEADERS)
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    for lang, m in summary.get("per_language", {}).items():
        rows.append({
            "Language": lang,
            "F1": (m.get("f1") or 0.0) * 100,
            "Precision": (m.get("precision") or 0.0) * 100,
            "Recall": (m.get("recall") or 0.0) * 100,
            "FPR (%)": (m.get("fpr") or 0.0) * 100,
            "GT": int(m.get("gt_count", 0)),
            "Predictions": int(m.get("predictions", 0)),
            "Correct": int(m.get("correct", 0)),
        })
    if not rows:
        return pd.DataFrame(columns=_DRILLDOWN_HEADERS)
    df = pd.DataFrame.from_records(rows)
    df = df.sort_values("F1", ascending=False, kind="stable").reset_index(drop=True)
    return df


def _snapshot_root(
    repo_id: str,
    revision: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
) -> Path:
    """Resolve where summary.json files live on disk (Hub snapshot or local)."""
    if local_dir is not None:
        return Path(local_dir)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=[f"*/*/{SUMMARY_FILENAME}"],
        )
    )


def _tab_label(dataset_id: str) -> str:
    """Human-friendly tab title; falls back to the raw dataset_id."""
    try:
        from commonlid.core.registry import get_dataset

        cls = type(get_dataset(dataset_id))
    except KeyError:
        return dataset_id
    return cls.title or dataset_id


def _dataset_metadata_markdown(dataset_id: str) -> str:
    """Render the dataset card (title, description, reference, license) as Markdown.

    Pulls metadata directly off the registered ``LIDDataset`` subclass.
    Falls back to a minimal placeholder when the dataset_id isn't registered
    in the running interpreter (e.g. an uploaded summary refers to a
    dataset_id from a newer version of the package).
    """
    try:
        from commonlid.core.registry import get_dataset

        cls = type(get_dataset(dataset_id))
    except KeyError:
        return f"### {dataset_id}\n\n_Dataset metadata unavailable in this build._"

    title = cls.title or dataset_id
    parts: list[str] = [f"### {title}"]
    if cls.description:
        parts.append(cls.description)
    meta_bits: list[str] = []
    if cls.reference_url:
        meta_bits.append(f"[Reference]({cls.reference_url})")
    license_bit = _format_license(cls.license_name, cls.license_url)
    if license_bit:
        meta_bits.append(license_bit)
    meta_bits.append(f"Main score: `{cls.main_score}`")
    parts.append("  •  ".join(meta_bits))
    return "\n\n".join(parts)


def _format_license(license_name: str, license_url: str | None) -> str:
    """Render the license as ``License: <name>`` and link to ``license_url`` when set."""
    if not license_name:
        return ""
    if license_url:
        return f"License: [`{license_name}`]({license_url})"
    return f"License: `{license_name}`"


def _make_select_handler(
    dataset_id: str,
    snapshot_root: Path,
) -> Any:
    """Build the row-select callback as a closure over the captured state.

    Uses ``gr.SelectData.row_value`` (Gradio's per-click payload that
    contains the clicked row as a 1-D list) so the drilldown picks up the
    *current* table ordering — switching the scope radio and then clicking
    a row resolves to the row at its post-toggle position. Passing the
    Dataframe component as an event input would not work: Gradio 6
    preprocesses Dataframe inputs into ``pandas.DataFrame`` objects, not
    the ``{"data", "headers"}`` dict we feed in via ``_styled_value``.

    Gradio inspects ``__defaults__`` when registering events, and comparing a
    DataFrame default against a type annotation hits an unimplemented arrow
    dtype path. A closure keeps the state out of the function signature.
    """

    def _on_select(evt: gr.SelectData) -> tuple[str, Any]:
        if evt.index is None or not evt.row_value:
            return ("_Click a row to load per-language metrics._", None)
        try:
            model_id = evt.row_value[0]
        except (IndexError, TypeError):
            return ("_Could not resolve clicked row._", None)
        per_lang = _per_language_drilldown(snapshot_root, dataset_id, model_id)
        return (
            f"### Per-language detail — `{model_id}` on `{dataset_id}`",
            _styled_value(per_lang, display=_display_matrix(per_lang)),
        )

    return _on_select


def _make_scope_handler(sub_df: Any) -> Any:
    """Build the scope-radio change callback: swap the table data + legend in lockstep."""

    def _on_change(scope: Scope) -> tuple[Any, str]:
        help_items = _HEADLINE_COLUMN_HELP_COV if scope == "cov" else _HEADLINE_COLUMN_HELP
        table = _format_table(sub_df, scope=scope)
        return (
            _styled_value(table, display=_display_matrix(table)),
            _columns_help_markdown(help_items),
        )

    return _on_change


def build_app(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Any:
    """Build (but do not launch) the Gradio Blocks app.

    The data is loaded eagerly at build time — restart the Space to pick up
    new uploads.
    """
    snapshot_root = _snapshot_root(repo_id, revision, cache_dir, local_dir)
    df = load_results(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=snapshot_root,
        allowed_datasets=VISIBLE_DATASETS,
    )

    revision_label = revision[:12] if revision else "HEAD"
    header = (
        f"# CommonLID Leaderboard\n"
        f"Results for the **CommonLID** and **CommonLID-nano** benchmarks. "
        f"Headline metric: **macro F1**. Models are ranked by macro F1 "
        f"within each tab; click a row to see per-language metrics.\n"
        f"\n"
        f"🌐 [Website]({WEBSITE_URL})  •  📝 [Blog post]({BLOG_URL})  •  📄 [Paper]({PAPER_URL})  •  🆕 [Add a model]({NEW_MODEL_URL})"
    )
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    if revision:
        repo_url += f"/tree/{revision}"
    footer = f"_Source: [`{repo_id}`]({repo_url}) @ `{revision_label}`._"

    with gr.Blocks(title="CommonLID Leaderboard") as demo:
        gr.Markdown(header)
        with gr.Tabs():
            for dataset_id in VISIBLE_DATASETS:
                tab_label = _tab_label(dataset_id)
                with gr.Tab(label=tab_label):
                    gr.Markdown(_dataset_metadata_markdown(dataset_id))
                    sub = df[df["dataset_id"] == dataset_id]
                    table = _format_table(sub, scope="all")
                    if table.empty:
                        gr.Markdown(
                            f"_No results for `{dataset_id}` in `{repo_id}` yet."
                            " Upload `summary.json` files via "
                            f"`huggingface-cli upload {repo_id} <local-dir> "
                            "--repo-type=dataset` and restart this Space._"
                        )
                        continue

                    scope_radio = gr.Radio(
                        choices=SCOPE_CHOICES,
                        value="all",
                        label="Scoring scope",
                        interactive=True,
                    )
                    leaderboard = gr.Dataframe(
                        value=_styled_value(table, display=_display_matrix(table)),
                        datatype=_HEADLINE_DATATYPES,
                        interactive=False,
                        wrap=True,
                        label=f"{dataset_id} — sorted by Macro F1",
                    )
                    with gr.Accordion("What do these columns mean?", open=False):
                        legend = gr.Markdown(_columns_help_markdown(_HEADLINE_COLUMN_HELP))
                    drilldown_label = gr.Markdown("_Click a row to load per-language metrics._")
                    # Seed the drilldown grid with an empty DataFrame so the Component
                    # has stable column headers before the first row click.
                    drilldown_seed = _per_language_drilldown(snapshot_root, "", "")
                    drilldown = gr.Dataframe(
                        value=_styled_value(
                            drilldown_seed, display=_display_matrix(drilldown_seed)
                        ),
                        datatype=_DRILLDOWN_DATATYPES,
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Accordion("What do these per-language columns mean?", open=False):
                        gr.Markdown(_columns_help_markdown(_DRILLDOWN_COLUMN_HELP))

                    scope_radio.change(
                        _make_scope_handler(sub),
                        inputs=[scope_radio],
                        outputs=[leaderboard, legend],
                    )
                    leaderboard.select(
                        _make_select_handler(dataset_id, snapshot_root),
                        outputs=[drilldown_label, drilldown],
                    )
        gr.Markdown(footer)

    return demo


__all__ = ["VISIBLE_DATASETS", "build_app"]
