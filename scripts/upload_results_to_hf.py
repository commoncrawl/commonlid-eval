"""Upload ``data/results/*`` to the published results HF dataset.

Thin wrapper around ``huggingface_hub.HfApi.upload_folder`` so the same
``<dataset>/<model>/{summary.json, predictions.jsonl}`` layout the
leaderboard expects gets published in one shot. Equivalent to::

    huggingface-cli upload commoncrawl/commonlid-results \\
        ./data/results --repo-type=dataset

The script just adds a couple of conveniences: dry-run, deletion of stale
files server-side, and a confirmation prompt before pushing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="commoncrawl/commonlid-results",
        help="HF dataset repo id to upload into.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/results"),
        help="Local directory whose contents become the repo root.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update CommonLID evaluation results",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=["*/*/summary.json", "*/*/predictions.jsonl"],
        help="Glob patterns to upload (relative to --source-dir).",
    )
    parser.add_argument(
        "--delete-patterns",
        nargs="*",
        default=None,
        help=(
            "Glob patterns to delete server-side before the upload. Use "
            "'*' to fully replace the repo contents — destructive."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it doesn't exist (default: public).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"source dir {args.source_dir} not found", file=sys.stderr)
        return 2

    summary_count = sum(1 for _ in args.source_dir.glob("*/*/summary.json"))
    pred_count = sum(1 for _ in args.source_dir.glob("*/*/predictions.jsonl"))
    print(
        f"about to push {summary_count} summary.json + {pred_count} predictions.jsonl "
        f"from {args.source_dir} -> {args.repo_id} (private={args.private})",
        flush=True,
    )
    if args.delete_patterns:
        print(f"  delete patterns: {args.delete_patterns}", flush=True)
    if args.dry_run:
        print("dry-run: nothing pushed.", flush=True)
        return 0
    if not args.yes:
        ans = input("continue? [y/N] ").strip().lower()
        if ans not in {"y", "yes"}:
            print("aborted.", flush=True)
            return 1

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=str(args.source_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        allow_patterns=args.allow_patterns,
        delete_patterns=args.delete_patterns,
    )
    print(f"\npushed: {commit_info.commit_url}", flush=True)
    print(f"oid:    {commit_info.oid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
