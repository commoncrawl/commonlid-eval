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

#: Display columns in the headline table (in order). Macro F1 is the headline metric.
_HEADLINE_COLUMNS: list[tuple[str, str]] = [
    ("model_id", "Model"),
    ("macro_f1", "Macro F1"),
    ("micro_f1", "Micro F1"),
    ("mean_fpr", "Mean FPR (%)"),
    ("n_languages", "Languages"),
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


def _columns_help_markdown(items: list[tuple[str, str]]) -> str:
    """Render a (column, description) list as a Markdown bullet block."""
    return "\n".join(f"- **{label}** — {desc}" for label, desc in items)


def _styled_value(table: Any, right_align_after_col: int = 0) -> dict[str, Any]:
    """Wrap a formatted DataFrame in the ``{"data", "headers", "metadata"}``
    structure Gradio expects when you want per-cell ``<td>`` CSS.

    Every cell in column ``> right_align_after_col`` gets ``text-align:
    right``; the first column (Model / Language) is left-aligned by
    default. Documented at
    https://www.gradio.app/guides/styling-the-gradio-dataframe.
    """
    headers = list(table.columns)
    data = table.values.tolist()
    styling = [
        ["text-align: right" if col > right_align_after_col else "" for col in range(len(headers))]
        for _ in range(len(data))
    ]
    return {"data": data, "headers": headers, "metadata": {"styling": styling}}


def _format_table(df: Any) -> Any:
    """Project + format a results DataFrame for one Gradio tab.

    Numeric columns are converted to **fixed-decimal strings** (e.g. ``0.00``
    not ``0``) so the rendered cells line up vertically; sort ordering is
    preserved by sorting on the raw ``macro_f1`` *before* formatting.

    - Macro F1 / Micro F1 / Samples/s use **1 decimal**.
    - Mean FPR (%) uses **2 decimals**.
    """
    import pandas as pd

    if df.empty:
        return pd.DataFrame(columns=[label for _, label in _HEADLINE_COLUMNS])
    out = df.copy()
    # Sort on the raw float so the resulting order is correct; format only
    # afterwards (string sort would order "10" before "9").
    out = out.sort_values("macro_f1", ascending=False, kind="stable").reset_index(drop=True)
    out["macro_f1"] = (out["macro_f1"] * 100).map(lambda x: f"{x:.1f}")
    out["micro_f1"] = (out["micro_f1"] * 100).map(lambda x: f"{x:.1f}")
    out["mean_fpr"] = (out["mean_fpr"] * 100).map(lambda x: f"{x:.2f}")
    out["samples_per_second"] = out["samples_per_second"].map(lambda x: f"{x:.1f}")
    out = out[[k for k, _ in _HEADLINE_COLUMNS]]
    out.columns = [label for _, label in _HEADLINE_COLUMNS]
    return out


def _per_language_drilldown(snapshot_root: Path, dataset_id: str, model_id: str) -> Any:
    """Return a sorted (asc by F1) per-language metrics DataFrame for a (model, dataset)."""
    import pandas as pd

    summary_path = snapshot_root / dataset_id / model_id / SUMMARY_FILENAME
    if not summary_path.is_file():
        return pd.DataFrame(
            columns=[
                "Language",
                "F1",
                "Precision",
                "Recall",
                "FPR (%)",
                "GT",
                "Predictions",
                "Correct",
            ]
        )
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    # Keep raw floats around for sorting; format to fixed-decimal strings
    # afterwards so the rendered cells line up vertically.
    rows = []
    for lang, m in summary.get("per_language", {}).items():
        rows.append({
            "Language": lang,
            "F1": (m.get("f1") or 0.0) * 100,
            "Precision": (m.get("precision") or 0.0) * 100,
            "Recall": (m.get("recall") or 0.0) * 100,
            "FPR (%)": (m.get("fpr") or 0.0) * 100,
            "GT": m.get("gt_count", 0),
            "Predictions": m.get("predictions", 0),
            "Correct": m.get("correct", 0),
        })
    if not rows:
        return pd.DataFrame(
            columns=[
                "Language",
                "F1",
                "Precision",
                "Recall",
                "FPR (%)",
                "GT",
                "Predictions",
                "Correct",
            ]
        )
    df = pd.DataFrame.from_records(rows)
    df = df.sort_values("F1", ascending=False, kind="stable").reset_index(drop=True)
    df["F1"] = df["F1"].map(lambda x: f"{x:.1f}")
    df["Precision"] = df["Precision"].map(lambda x: f"{x:.1f}")
    df["Recall"] = df["Recall"].map(lambda x: f"{x:.1f}")
    df["FPR (%)"] = df["FPR (%)"].map(lambda x: f"{x:.2f}")
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
    table: Any,
    snapshot_root: Path,
) -> Any:
    """Build the row-select callback as a closure over the captured state.

    Gradio inspects ``__defaults__`` when registering events, and comparing a
    DataFrame default against a type annotation hits an unimplemented arrow
    dtype path. A closure keeps the state out of the function signature.
    """

    def _on_select(evt: gr.SelectData) -> tuple[str, Any]:
        if evt.index is None:
            return ("_Click a row to load per-language metrics._", None)
        row_idx = evt.index[0] if isinstance(evt.index, list | tuple) else evt.index
        try:
            model_id = table.iloc[row_idx]["Model"]
        except (IndexError, KeyError):
            return ("_Could not resolve clicked row._", None)
        per_lang = _per_language_drilldown(snapshot_root, dataset_id, model_id)
        return (
            f"### Per-language detail — `{model_id}` on `{dataset_id}`",
            _styled_value(per_lang),
        )

    return _on_select


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
        f"📝 [Blog post]({BLOG_URL})  •  📄 [Paper]({PAPER_URL})"
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
                    table = _format_table(sub)
                    if table.empty:
                        gr.Markdown(
                            f"_No results for `{dataset_id}` in `{repo_id}` yet."
                            " Upload `summary.json` files via "
                            f"`huggingface-cli upload {repo_id} <local-dir> "
                            "--repo-type=dataset` and restart this Space._"
                        )
                        continue

                    leaderboard = gr.Dataframe(
                        value=_styled_value(table),
                        datatype=_HEADLINE_DATATYPES,
                        interactive=False,
                        wrap=True,
                        label=f"{dataset_id} — sorted by Macro F1",
                    )
                    with gr.Accordion("What do these columns mean?", open=False):
                        gr.Markdown(_columns_help_markdown(_HEADLINE_COLUMN_HELP))
                    drilldown_label = gr.Markdown("_Click a row to load per-language metrics._")
                    # Seed the drilldown grid with an empty DataFrame so the Component
                    # has stable column headers before the first row click.
                    drilldown = gr.Dataframe(
                        value=_styled_value(_per_language_drilldown(snapshot_root, "", "")),
                        datatype=_DRILLDOWN_DATATYPES,
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Accordion("What do these per-language columns mean?", open=False):
                        gr.Markdown(_columns_help_markdown(_DRILLDOWN_COLUMN_HELP))

                    leaderboard.select(
                        _make_select_handler(dataset_id, table, snapshot_root),
                        outputs=[drilldown_label, drilldown],
                    )
        gr.Markdown(footer)

    return demo


__all__ = ["VISIBLE_DATASETS", "build_app"]
