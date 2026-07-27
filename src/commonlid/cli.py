"""Typer command-line interface.

This is a thin facade over :mod:`commonlid.evaluation.evaluator` and the
registry. All heavy lifting happens in the package.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import typer

from commonlid import __version__
from commonlid.core.registry import (
    get_dataset,
    get_model,
    list_datasets,
    list_models,
)
from commonlid.evaluation.evaluator import Evaluator
from commonlid.evaluation.results import load_summary
from commonlid.logging import setup_logging
from commonlid.metrics.support_matrix import save_support_matrix

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="commonlid",
    help="Evaluate language identification models on CommonLID and other benchmarks.",
    no_args_is_help=True,
    add_completion=False,
)


def _ensure_registry_loaded() -> None:
    """Import the subpackages whose submodule imports populate the registries."""
    import commonlid.datasets  # noqa: F401
    import commonlid.models  # noqa: F401


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    setup_logging(verbose=verbose)
    _ensure_registry_loaded()


@app.command("version")
def version_cmd() -> None:
    """Print the installed commonlid version."""
    typer.echo(__version__)


@app.command("list-models")
def list_models_cmd(
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """List all registered LID models."""
    ids = list_models()
    if as_json:
        typer.echo(json.dumps(ids))
    else:
        for model_id in ids:
            typer.echo(model_id)


@app.command("list-datasets")
def list_datasets_cmd(
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """List all registered LID evaluation datasets."""
    ids = list_datasets()
    if as_json:
        typer.echo(json.dumps(ids))
    else:
        for dataset_id in ids:
            typer.echo(dataset_id)


DSPY_SPEC_PREFIX = "dspy:"


@app.command()
def run(
    model: Annotated[
        list[str],
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model id (repeat to add more). Use 'dspy:<llm-model-name>' "
                "(e.g. 'dspy:azure/gpt-4o-mini') to evaluate an LLM via DSPy."
            ),
        ),
    ],
    dataset: Annotated[
        list[str], typer.Option("--dataset", "-d", help="Dataset id (repeat to add more).")
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to write results into."),
    ] = Path("./results"),
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
    limit: Annotated[int, typer.Option("--limit", help="Cap samples (0 = no limit).")] = 0,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable prediction cache.")] = False,
    sample_count_threshold: Annotated[
        int,
        typer.Option("--sample-threshold", help="Skip langs with fewer gold samples than this."),
    ] = 0,
    # --- DSPy LLM flags (only used when one of --model is 'dspy:...') ---
    api_base: Annotated[
        str | None,
        typer.Option("--api-base", help="LLM provider base URL (required by dspy: models)."),
    ] = None,
    api_version: Annotated[str | None, typer.Option("--api-version")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
    azure_ad_token: Annotated[
        bool, typer.Option("--azure-ad-token", help="Use Azure DefaultAzureCredential.")
    ] = False,
    temperature: Annotated[float | None, typer.Option("--temperature")] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    max_completion_tokens: Annotated[int | None, typer.Option("--max-completion-tokens")] = None,
    llm_n_threads: Annotated[
        int,
        typer.Option("--llm-n-threads", help="Threads in the DSPy evaluator for LLM models."),
    ] = 1,
) -> None:
    """Run the evaluator over the requested (model, dataset) pairs.

    Mixes registry-backed classical LID models with DSPy-backed LLMs:
    pass ``--model GlotLID`` for a registered model or
    ``--model dspy:azure/gpt-4o-mini`` to spin up a DSPy LLM on the fly.
    """
    llm_kwargs = {
        "api_base": api_base,
        "api_version": api_version,
        "api_key": api_key,
        "azure_ad_token": azure_ad_token,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "batch_size": batch_size,
        "n_threads": llm_n_threads,
        "cache_dir": output_dir / ".dspy_cache",
    }
    models = [_resolve_model_spec(spec, llm_kwargs) for spec in model]
    datasets = [get_dataset(d) for d in dataset]
    Evaluator(
        models=models,
        datasets=datasets,
        output_dir=output_dir,
        batch_size=batch_size,
        use_cache=not no_cache,
        limit=limit if limit > 0 else None,
        sample_count_threshold=sample_count_threshold,
    ).run()


def _resolve_model_spec(spec: str, llm_kwargs: dict[str, Any]) -> Any:
    """Resolve a CLI --model spec to a loaded :class:`LIDModel` instance."""
    if spec.startswith(DSPY_SPEC_PREFIX):
        llm_model_name = spec.removeprefix(DSPY_SPEC_PREFIX)
        if not llm_model_name:
            msg = "'dspy:' model spec requires a model name, e.g. 'dspy:azure/gpt-4o-mini'"
            raise typer.BadParameter(msg)
        if not llm_kwargs.get("api_base"):
            msg = "DSPy LLM models require --api-base (e.g. your Azure endpoint URL)"
            raise typer.BadParameter(msg)
        from commonlid.models.dspy_llm import DSPyLLMModel

        return DSPyLLMModel(llm_model_name=llm_model_name, **llm_kwargs)
    return get_model(spec)


@app.command("estimate-tokens")
def estimate_tokens_cmd(
    model: Annotated[
        list[str],
        typer.Option(
            "--model",
            "-m",
            help="LLM model name or 'dspy:<name>' spec (e.g. 'dspy:openai/gpt-5'). Repeat to add more.",
        ),
    ],
    dataset: Annotated[
        list[str], typer.Option("--dataset", "-d", help="Dataset id (repeat to add more).")
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Measure only the first N samples (0 = full dataset). Projects to the full set when N < size.",
        ),
    ] = 0,
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", help="Override the DSPy system instruction used for estimation."),
    ] = None,
    output_tokens: Annotated[
        int, typer.Option("--output-tokens", help="Assumed completion tokens per sample.")
    ] = 10,
    reasoning_tokens: Annotated[
        int,
        typer.Option(
            "--reasoning-tokens",
            help="Assumed reasoning tokens per sample (reasoning models); billed as completion tokens.",
        ),
    ] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON instead of text.")] = False,
) -> None:
    """Estimate token usage and USD cost of an LLM run without making API calls.

    Reconstructs the exact DSPy prompt per dataset sample and counts tokens with
    the model's tokenizer (tiktoken for OpenAI/Azure, HF AutoTokenizer for HF
    models); pricing comes from LiteLLM. Accounts for the system prompt and an
    optional per-sample reasoning-token assumption.
    """
    from commonlid.evaluation import token_cost

    instr = instruction if instruction is not None else token_cost.DEFAULT_INSTRUCTION
    estimates: list[token_cost.TokenCostEstimate] = []
    for dataset_id in dataset:
        ds = get_dataset(dataset_id)
        for model_name in model:
            estimates.append(
                token_cost.estimate(
                    model_name,
                    ds,
                    instruction=instr,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    limit=limit,
                )
            )

    if as_json:
        typer.echo(json.dumps([e.to_dict() for e in estimates], indent=2))
        return
    for est in estimates:
        _echo_estimate(est)


def _fmt_usd(value: float | None) -> str:
    return f"${value:,.4f}" if value is not None else "n/a (model not in LiteLLM price map)"


def _echo_estimate(est: Any) -> None:
    typer.echo(f"\n{est.model}  on  {est.dataset_id}")
    typer.echo(f"  samples measured:        {est.n_samples:,} of {est.n_total:,}")
    typer.echo(f"  per-request overhead:    {est.per_request_overhead_tokens:,} tokens (system prompt + scaffolding)")
    typer.echo(
        f"  input tokens:            {est.total_input_tokens:,} total "
        f"({est.mean_input_tokens:,.1f} mean/sample)"
    )
    typer.echo(
        f"  output tokens:           {est.total_output_tokens:,} total "
        f"({est.output_tokens_per_sample} completion + {est.reasoning_tokens_per_sample} reasoning per sample)"
    )
    typer.echo(f"  input cost:              {_fmt_usd(est.input_cost_usd)}")
    typer.echo(f"  output cost:             {_fmt_usd(est.output_cost_usd)}")
    typer.echo(f"  total cost:              {_fmt_usd(est.total_cost_usd)}")
    if est.extrapolated:
        typer.echo(
            f"  → projected full run:    {est.full_total_input_tokens:,} input + "
            f"{est.full_total_output_tokens:,} output tokens, {_fmt_usd(est.full_total_cost_usd)} "
            f"(extrapolated from {est.n_samples:,} samples)"
        )
    if est.reasoning_tokens_per_sample == 0:
        typer.echo("  note: reasoning tokens assumed 0; set --reasoning-tokens for reasoning models (gpt-5/o-series).")
    if est.tokenizer_note:
        typer.echo(f"  note: {est.tokenizer_note}")


@app.command()
def predict(
    model: Annotated[str, typer.Option("--model", "-m", help="Model id.")],
    text: Annotated[str | None, typer.Option("--text", help="A single input string.")] = None,
    text_file: Annotated[
        Path | None,
        typer.Option("--text-file", help="Newline-delimited text file ('-' for stdin)."),
    ] = None,
) -> None:
    """Run a model against ad-hoc text, writing JSONL predictions to stdout."""
    if text is None and text_file is None:
        typer.echo("Either --text or --text-file must be provided.", err=True)
        raise typer.Exit(code=2)

    texts: list[str] = []
    if text is not None:
        texts.append(text)
    if text_file is not None:
        texts.extend(_read_lines(text_file))

    lid_model = get_model(model)
    preds = lid_model.predict(texts)
    for t, p in zip(texts, preds, strict=True):
        typer.echo(json.dumps({"text": t, "pred": p, "model": model}))


@app.command("generate-support-matrix")
def generate_support_matrix(
    out: Annotated[Path, typer.Option("--out", help="Destination CSV path.")],
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Restrict to these model ids (repeatable). Default: every registered model.",
        ),
    ] = None,
) -> None:
    """Build the language x model support CSV by asking each model to enumerate its languages.

    Models that cannot enumerate a concrete language list (e.g. LLMs, or cld3
    when its optional bindings are not installed) are reported as skipped and
    absent from the CSV. Heavy models (AfroLID, fasttext) will download
    weights on first run.
    """
    ids = models if models else list_models()
    matrix: dict[str, set[str]] = {}
    skipped: list[tuple[str, str]] = []
    for model_id in ids:
        try:
            model = get_model(model_id)
            supported = model.discover_supported_languages()
        except Exception as exc:
            skipped.append((model_id, f"{type(exc).__name__}: {exc}"))
            continue
        if supported is None:
            skipped.append((model_id, "discover_supported_languages() returned None"))
            continue
        matrix[model_id] = set(supported)

    if not matrix:
        typer.echo("No models produced a language list.", err=True)
        raise typer.Exit(code=1)

    save_support_matrix(matrix, out)
    typer.echo(f"Wrote support matrix for {len(matrix)} model(s) to {out}")
    for model_id, reason in skipped:
        typer.echo(f"  skipped {model_id}: {reason}", err=True)


leaderboard_app = typer.Typer(
    name="leaderboard",
    help="Serve the CommonLID leaderboard or push results to the HF dataset.",
    no_args_is_help=True,
)
app.add_typer(leaderboard_app, name="leaderboard")


@leaderboard_app.command("serve")
def leaderboard_serve(
    repo_id: Annotated[
        str,
        typer.Option(
            "--repo-id",
            help="HF dataset repo id holding <dataset>/<model>/summary.json files.",
        ),
    ] = "commoncrawl/commonlid-results",
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Optional commit SHA / branch / tag to pin."),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Override the HF snapshot cache directory."),
    ] = None,
    local_dir: Annotated[
        Path | None,
        typer.Option(
            "--local-dir",
            help="Skip the network and read summaries from this local directory instead.",
        ),
    ] = None,
    server_name: Annotated[
        str, typer.Option("--server-name", help="Bind address (set 0.0.0.0 for HF Spaces).")
    ] = "127.0.0.1",
    server_port: Annotated[int, typer.Option("--port", help="Port to bind.")] = 7860,
    share: Annotated[
        bool, typer.Option("--share/--no-share", help="Expose a public gradio.live URL.")
    ] = False,
) -> None:
    """Launch the CommonLID leaderboard Gradio app (requires the [leaderboard] extra)."""
    try:
        from commonlid.leaderboard.app import build_app
    except ImportError as exc:
        typer.echo(
            "leaderboard requires `commonlid[leaderboard]` (gradio + pandas). "
            f"Install it via `uv sync --extra leaderboard`. (underlying error: {exc})",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    demo = build_app(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )
    demo.launch(server_name=server_name, server_port=server_port, share=share)


@leaderboard_app.command("upload")
def leaderboard_upload(
    repo_id: Annotated[
        str,
        typer.Option(
            "--repo-id",
            help="HF dataset repo id (e.g. commoncrawl/commonlid-results).",
        ),
    ],
    local_dir: Annotated[
        Path,
        typer.Option(
            "--local-dir",
            help="Local directory to upload (e.g. ./data/results).",
        ),
    ],
    commit_message: Annotated[
        str | None,
        typer.Option("--commit-message", help="PR title; defaults to a refresh message."),
    ] = None,
    commit_description: Annotated[
        str | None,
        typer.Option("--commit-description", help="Optional PR body."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            help="Extra glob to skip (repeatable). `.cache/**` is always excluded.",
        ),
    ] = None,
    skip_predictions: Annotated[
        bool,
        typer.Option(
            "--skip-predictions",
            help="Skip per-row predictions.jsonl files (keeps repo small; the leaderboard only reads summary.json).",
        ),
    ] = False,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Base branch for the PR (defaults to the repo's default)."),
    ] = None,
) -> None:
    """Push a results folder to the HF dataset as a Pull Request.

    Folder layout must be ``<dataset_id>/<model_id>/summary.json`` so the
    leaderboard can read it. The upload is opened as a PR (never pushed
    directly to the default branch) so the dataset owner can review before
    merging.
    """
    if not local_dir.is_dir():
        typer.echo(f"--local-dir {local_dir} not found or not a directory.", err=True)
        raise typer.Exit(code=2)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - hub is a base dep
        typer.echo(f"huggingface-hub is required: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    ignore_patterns = [".cache/**", *(exclude or [])]
    if skip_predictions:
        ignore_patterns.append("**/predictions.jsonl")

    msg = commit_message or f"Refresh results from {local_dir.name}"
    logger.info("Uploading %s -> %s (PR; ignore=%s)", local_dir, repo_id, ignore_patterns)
    commit_info = HfApi().upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        ignore_patterns=ignore_patterns,
        commit_message=msg,
        commit_description=commit_description,
        create_pr=True,
    )
    pr_url = getattr(commit_info, "pr_url", None) or str(commit_info)
    typer.echo(f"Opened PR: {pr_url}")


@app.command("export-csv")
def export_csv(
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory containing summary.json files.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Destination CSV file.")],
) -> None:
    """Walk a results directory and flatten every per-language summary into one CSV."""
    rows = list(_iter_summary_rows(results_dir))
    if not rows:
        typer.echo(f"No summary.json files found under {results_dir}", err=True)
        raise typer.Exit(code=1)

    fieldnames = [
        "dataset_id",
        "model_id",
        "language",
        "gt_count",
        "predictions",
        "correct",
        "precision",
        "recall",
        "f1",
        "samples_per_second",
        "timestamp",
        "commonlid_version",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    typer.echo(f"Wrote {len(rows)} rows to {out}")


def _read_lines(path: Path) -> list[str]:
    if str(path) == "-":
        return [line.rstrip("\n") for line in sys.stdin if line.strip()]
    return [
        line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _iter_summary_rows(results_dir: Path) -> Iterator[dict[str, Any]]:
    for summary_path in sorted(results_dir.rglob("summary.json")):
        summary = load_summary(summary_path)
        for language, m in summary.get("per_language", {}).items():
            yield {
                "dataset_id": summary["dataset_id"],
                "model_id": summary["model_id"],
                "language": language,
                "gt_count": m["gt_count"],
                "predictions": m["predictions"],
                "correct": m["correct"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "samples_per_second": summary.get("samples_per_second"),
                "timestamp": summary.get("timestamp"),
                "commonlid_version": summary.get("commonlid_version"),
            }


if __name__ == "__main__":  # pragma: no cover
    app()
