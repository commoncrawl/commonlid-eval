"""Backfill the ``supported_languages`` field in existing ``summary.json`` files.

Walks ``<source_dir>/<dataset>/<model>/summary.json``, groups paths by
``model_id``, loads each model once via the registry, calls
``discover_supported_languages()``, and writes the result (sorted ISO 639-3
list, or JSON ``null`` for models whose support set is undefined) back into
every matching summary.

Run after ``pip install commonlid[<extras>]`` for whichever model classes
you need to enumerate. Models whose extras are absent are reported as
skipped and their summary files are left untouched (rather than clobbered
with ``null``), so a partial-env run doesn't lose previously-correct entries.

Skips files that already have a ``supported_languages`` key — even when its
value is JSON ``null`` (that's a real answer for LLM-style models, not a
placeholder). Pass ``--overwrite`` to refresh anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("backfill_supported_languages")


def _discover_one(model_id: str) -> list[str] | None:
    """Instantiate ``model_id`` from the registry and call ``discover_supported_languages()``.

    Returns the sorted list, or ``None`` if the model declines to enumerate
    (the canonical "support set undefined" sentinel — must round-trip as
    JSON ``null``). Raises on import or load errors so callers can decide
    whether to skip the file.
    """
    from commonlid.core.registry import get_model

    model = get_model(model_id)
    supported = model.discover_supported_languages()
    if supported is None:
        return None
    return sorted(supported)


def _update_summary(
    path: Path,
    value: list[str] | None,
    *,
    overwrite: bool,
    dry_run: bool,
) -> str:
    """Return one of ``written|skipped-has-key|skipped-dry-run``."""
    with path.open(encoding="utf-8") as f:
        summary = json.load(f)
    if "supported_languages" in summary and not overwrite:
        return "skipped-has-key"
    summary["supported_languages"] = value
    if dry_run:
        return "skipped-dry-run"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return "written"


def _collect_summaries(
    source_dir: Path,
    only_models: list[str] | None,
) -> dict[str, list[Path]]:
    """Group ``<dataset>/<model>/summary.json`` paths under ``source_dir`` by model_id."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source_dir.glob("*/*/summary.json")):
        model_id = path.parent.name
        if only_models and model_id not in only_models:
            continue
        grouped[model_id].append(path)
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/results"),
        help="Root directory holding <dataset>/<model>/summary.json files.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        dest="models",
        help="Restrict to this model id (repeatable). Default: every model_id found.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refresh files even if they already have a supported_languages key.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover support sets but do not modify any files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    source_dir: Path = args.source_dir
    if not source_dir.is_dir():
        logger.error("source dir %s not found", source_dir)
        return 2

    grouped = _collect_summaries(source_dir, args.models)
    if not grouped:
        logger.error(
            "no summary.json files matched under %s (filters: --model=%s)",
            source_dir,
            args.models,
        )
        return 1

    # Import the models package so every model_id registers itself with the
    # registry before we look any of them up.
    import commonlid.models  # noqa: F401

    counts: dict[str, int] = defaultdict(int)
    skipped_models: list[tuple[str, str]] = []
    for model_id, paths in grouped.items():
        try:
            supported = _discover_one(model_id)
            reason = "undefined support set" if supported is None else None
        except KeyError:
            # Unknown model_id: usually legacy imports of LLM runs (e.g. GPT-4o).
            # There is no class we can ask, so persist the explicit "undefined"
            # sentinel rather than skipping; the data layer treats it the same
            # as a registered model whose discover() returned None.
            supported = None
            reason = "unknown model_id (legacy LLM import?)"
        except Exception as exc:
            # Load / import failures: skip rather than clobber. The user can
            # rerun from an env that has the missing extras.
            logger.warning(
                "skipping %s (%d file(s)) -- %s: %s",
                model_id,
                len(paths),
                type(exc).__name__,
                exc,
            )
            skipped_models.append((model_id, f"{type(exc).__name__}: {exc}"))
            counts["skipped-model-error"] += len(paths)
            continue

        if supported is None:
            logger.info(
                "%s: %s -- writing JSON null to %d file(s)",
                model_id,
                reason,
                len(paths),
            )
        else:
            logger.info(
                "%s: %d languages -- writing to %d file(s)",
                model_id,
                len(supported),
                len(paths),
            )

        for path in paths:
            try:
                outcome = _update_summary(
                    path, supported, overwrite=args.overwrite, dry_run=args.dry_run
                )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to update %s: %s", path, exc)
                counts["error"] += 1
                continue
            counts[outcome] += 1
            logger.debug("%s %s", outcome, path)

    summary: dict[str, Any] = {
        "models_processed": len(grouped) - len(skipped_models),
        "models_skipped": len(skipped_models),
        **counts,
    }
    logger.info("done: %s", json.dumps(summary))
    for model_id, reason in skipped_models:
        logger.info("  skipped model %s: %s", model_id, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
