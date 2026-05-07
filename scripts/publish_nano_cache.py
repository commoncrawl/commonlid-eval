"""Build every nano subset from its parent and push each to its own private HF repo.

One repo per nano (visibility can track the parent dataset). Always uploads
as ``private=True`` — flip to public on the HF UI afterwards if needed.

Prerequisite: HF auth with write access to the ``commoncrawl`` org plus read
access to each parent's private cache (or its build-from-source prerequisites).

Usage::

    uv run --extra dev python scripts/publish_nano_cache.py
    uv run --extra dev python scripts/publish_nano_cache.py --dry-run
    uv run --extra dev python scripts/publish_nano_cache.py --only smolsent_300_nano

After the run finishes, the script prints the HEAD SHA of each per-nano repo
plus a copy-pastable Python snippet you can paste into ``src/commonlid/datasets/nano.py``
to set ``cache_hf_revision`` on each subclass.
"""

from __future__ import annotations

import argparse
import sys
import time


def _nano_classes() -> list[type]:
    from commonlid.datasets.nano import (
        BiblesDatasetNano,
        CommonLIDDatasetNano,
        FloresDevDatasetNano,
        SmolSentDatasetNano,
        SocialMediaDatasetNano,
        UDHRDatasetNano,
    )

    return [
        CommonLIDDatasetNano,
        FloresDevDatasetNano,
        UDHRDatasetNano,
        SmolSentDatasetNano,
        BiblesDatasetNano,
        SocialMediaDatasetNano,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every nano but skip the actual push_to_hub step.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of dataset_ids to publish (defaults to all six).",
    )
    args = parser.parse_args()

    classes = _nano_classes()
    if args.only:
        wanted = set(args.only)
        classes = [cls for cls in classes if cls.dataset_id in wanted]
        missing = wanted - {cls.dataset_id for cls in classes}
        if missing:
            print(f"unknown dataset_ids: {sorted(missing)}", file=sys.stderr)
            return 2

    pinned: dict[str, str] = {}

    for cls in classes:
        repo = cls.cache_hf_repo
        print(f"--- {cls.dataset_id}  (repo={repo!r}) ---", flush=True)
        t0 = time.perf_counter()
        ds = cls().build_from_source()
        elapsed = time.perf_counter() - t0
        print(
            f"    built {len(ds)} rows x {len(ds.column_names)} columns in {elapsed:.1f}s",
            flush=True,
        )
        if args.dry_run:
            print("    (dry-run) skipping push_to_hub", flush=True)
            continue
        ds.push_to_hub(
            repo_id=repo,
            split="test",
            private=True,
        )
        print(f"    pushed to {repo} (private)", flush=True)

    if args.dry_run:
        print("\ndry-run complete; nothing pushed", flush=True)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    print()
    for cls in classes:
        repo = cls.cache_hf_repo
        commits = api.list_repo_commits(repo, repo_type="dataset")
        head = commits[0].commit_id
        pinned[cls.__name__] = head
        print(f"  {cls.__name__:25s}  {repo}  HEAD={head}", flush=True)

    print(
        "\n# Paste the following revisions into the matching subclasses in "
        "src/commonlid/datasets/nano.py:",
        flush=True,
    )
    for name, sha in pinned.items():
        print(f'#   {name}.cache_hf_revision = "{sha}"', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
