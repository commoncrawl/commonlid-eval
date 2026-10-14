"""Backfill ``pred_raw`` into existing ``predictions.jsonl`` files, and check reruns.

Two subcommands:

``llm``
    Adds ``pred_raw`` to the legacy GPT results (``<results>/<dataset>/GPT-*``)
    from the original wmdqs DSPy caches (``cache_azure_<model>_*.jsonl``),
    which hold each prompt text and the field exactly as the LLM wrote it.
    Nothing is called: the LLMs ran on Azure at temperature 0.7, so the
    cache is the only faithful source.

    Those caches are keyed by the text the LLM saw, i.e. after the OpenLID
    normer, so each row's ``text_hash`` is resolved to its dataset text and
    normalised before the lookup. A raw value is written only when it
    conforms back to the row's stored ``pred`` (an unmappable answer was
    stored as ``"und"``, which is how the metrics count ``None`` too).
    Rows without exactly one such value keep no ``pred_raw`` key, which
    means "unknown". No other field is changed.

``check``
    Compares a rerun against a copy of the results taken before it, and
    fails if any ``pred``, ``score``, ``correct``, ``gold`` or summary metric
    moved. Use it after ``commonlid run --refresh-raw`` on the local models.

Examples::

    cp -R data/results /tmp/results-before
    commonlid run -m GlotLID -d udhr_nano -o data/results --refresh-raw
    python scripts/backfill_pred_raw.py check --before /tmp/results-before --after data/results

    python scripts/backfill_pred_raw.py llm --results-dir data/results \\
        --cache-dir ../wmdqs/eval/data/backup_data/cache --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("backfill_pred_raw")

# Leaderboard model folder -> model name in the wmdqs cache file names.
LLM_CACHE_MODELS: dict[str, str] = {
    "GPT-4o": "azure_GPT-4o",
    "GPT-4o-mini": "azure_GPT-4O-MINI-CONTENTFILTER-DISABLED",
    "GPT-5": "azure_gpt-5",
    "GPT-5-mini": "azure_gpt-5-mini",
}

# Row fields a rerun must reproduce exactly.
_CHECKED_ROW_FIELDS = ("idx", "text_hash", "gold", "pred", "score", "correct")
_CHECKED_SUMMARY_FIELDS = ("n_samples", "n_samples_with_gold", "per_language", "macro", "micro")

_UND = "und"


def load_llm_cache(cache_dir: Path, cache_model: str) -> dict[str, Counter[str | None]]:
    """Map each normalised prompt text to the raw answers cached for it."""
    raws: dict[str, Counter[str | None]] = defaultdict(Counter)
    for path in sorted(cache_dir.glob(f"cache_{cache_model}_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                raw = entry.get("language_iso639_3")
                raws[entry["text"]][raw if isinstance(raw, str) else None] += 1
    return dict(raws)


def _as_pred(raw: str | None) -> str:
    """What the metrics see for a raw LLM answer: conformed code, or ``"und"``."""
    from commonlid.core.lid_model import LIDModel
    from commonlid.models.dspy_llm import DSPyLLMModel

    return LIDModel._conform(DSPyLLMModel._coerce(raw)) or _UND


def pick_raw(candidates: Mapping[str | None, int] | None, pred: str | None) -> tuple[str, Any]:
    """Return ``(status, raw)`` for one row; ``raw`` is only meaningful when status is ``ok``.

    ``candidates`` holds every raw answer cached for the row's text (several
    when the text was run more than once). Only answers that conform to the
    stored ``pred`` count, so a cache from a different run can't leak in.
    """
    if not candidates:
        return "missing", None
    target = pred or _UND
    matching = {raw for raw in candidates if _as_pred(raw) == target}
    if len(matching) == 1:
        return "ok", next(iter(matching))
    return ("ambiguous" if matching else "mismatch"), None


def backfill_rows(
    rows: list[dict[str, Any]],
    texts_by_hash: Mapping[str, str],
    cache: Mapping[str, Mapping[str | None, int]],
) -> Counter[str]:
    """Add ``pred_raw`` in place where it can be recovered; return per-status counts."""
    from commonlid.preprocess import openlid_normer_clean_line

    stats: Counter[str] = Counter()
    for row in rows:
        if "pred_raw" in row:
            stats["already_set"] += 1
            continue
        text = texts_by_hash.get(row["text_hash"])
        if text is None:
            stats["no_text"] += 1
            continue
        status, raw = pick_raw(cache.get(openlid_normer_clean_line(text)), row["pred"])
        stats[status] += 1
        if status == "ok":
            row["pred_raw"] = raw
    return stats


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _texts_by_hash(dataset_id: str) -> dict[str, str]:
    import commonlid.datasets  # noqa: F401  (registers the datasets)
    from commonlid.core.registry import get_dataset
    from commonlid.evaluation.evaluator import _text_hash

    dataset = get_dataset(dataset_id)
    dataset.load()
    return {
        _text_hash(text): text
        for texts, _ in dataset.iter_batches(batch_size=10_000)
        for text in texts
    }


def cmd_llm(args: argparse.Namespace) -> int:
    from commonlid.evaluation.results import write_predictions

    results_dir: Path = args.results_dir
    models = args.model or list(LLM_CACHE_MODELS)
    caches = {m: load_llm_cache(args.cache_dir, LLM_CACHE_MODELS[m]) for m in models}
    for model, cache in caches.items():
        if not cache:
            logger.error("No cache files for %s under %s", model, args.cache_dir)
            return 1

    datasets = args.dataset or sorted(
        p.name for p in results_dir.iterdir() if any((p / m).is_dir() for m in models)
    )
    totals: Counter[str] = Counter()
    for dataset_id in datasets:
        targets = [
            m for m in models if (results_dir / dataset_id / m / "predictions.jsonl").exists()
        ]
        if not targets:
            continue
        texts_by_hash = _texts_by_hash(dataset_id)
        for model in targets:
            path = results_dir / dataset_id / model / "predictions.jsonl"
            rows = _read_rows(path)
            before = [{k: v for k, v in row.items() if k != "pred_raw"} for row in rows]
            stats = backfill_rows(rows, texts_by_hash, caches[model])
            after = [{k: v for k, v in row.items() if k != "pred_raw"} for row in rows]
            assert before == after, "backfill must only add pred_raw"
            totals.update(stats)
            logger.info(
                "%s/%s: %d rows %s", dataset_id, model, len(rows), dict(sorted(stats.items()))
            )
            if stats["ok"] and not args.dry_run:
                write_predictions(rows, path)
    logger.info("total: %s%s", dict(sorted(totals.items())), " (dry run)" if args.dry_run else "")
    return 0


def compare_rows(before: Iterable[dict[str, Any]], after: Iterable[dict[str, Any]]) -> list[str]:
    """Describe every row whose checked fields differ, matched by position."""
    before, after = list(before), list(after)
    problems = []
    if len(before) != len(after):
        problems.append(f"row count {len(before)} -> {len(after)}")
    for old, new in zip(before, after, strict=False):
        for field in _CHECKED_ROW_FIELDS:
            if old.get(field) != new.get(field):
                problems.append(
                    f"idx {old.get('idx')}: {field} {old.get(field)!r} -> {new.get(field)!r}"
                )
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    n_files = n_bad = n_raw = n_rows = 0
    for old_path in sorted(args.before.glob("*/*/predictions.jsonl")):
        rel = old_path.relative_to(args.before)
        new_path = args.after / rel
        if not new_path.exists():
            continue
        n_files += 1
        new_rows = _read_rows(new_path)
        problems = compare_rows(_read_rows(old_path), new_rows)
        n_rows += len(new_rows)
        n_raw += sum("pred_raw" in row for row in new_rows)

        old_summary = json.loads((old_path.parent / "summary.json").read_text(encoding="utf-8"))
        new_summary = json.loads((new_path.parent / "summary.json").read_text(encoding="utf-8"))
        problems += [
            f"summary {field} changed"
            for field in _CHECKED_SUMMARY_FIELDS
            if old_summary.get(field) != new_summary.get(field)
        ]
        if problems:
            n_bad += 1
            logger.error("%s: %d difference(s)", rel.parent, len(problems))
            for problem in problems[:10]:
                logger.error("  %s", problem)
    logger.info(
        "compared %d file(s), %d with differences; %d of %d rows have pred_raw",
        n_files,
        n_bad,
        n_raw,
        n_rows,
    )
    return 1 if n_bad or not n_files else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    llm = sub.add_parser("llm", help="Backfill the legacy GPT results from the wmdqs DSPy caches.")
    llm.add_argument("--results-dir", type=Path, default=Path("data/results"))
    llm.add_argument("--cache-dir", type=Path, required=True)
    llm.add_argument("--dataset", action="append", help="Dataset id (repeatable; default: all).")
    llm.add_argument("--model", action="append", choices=list(LLM_CACHE_MODELS))
    llm.add_argument("--dry-run", action="store_true", help="Report coverage without writing.")
    llm.set_defaults(func=cmd_llm)

    check = sub.add_parser("check", help="Fail if a rerun changed any prediction or metric.")
    check.add_argument("--before", type=Path, required=True)
    check.add_argument("--after", type=Path, required=True)
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
