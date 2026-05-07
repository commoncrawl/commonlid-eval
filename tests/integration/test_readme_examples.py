"""Execute every tagged Python code block in README.md.

Keeps the README executable — whenever an example drifts away from the
package API, CI fails. Each ``python`` code block preceded by an HTML
marker like ``<!-- readme-test: fast; id=my-example -->`` becomes a
parametrised test case. Modes:

- ``fast``  — runs in the default ``pytest`` invocation.
- ``slow``  — marked ``slow`` + ``network``; only runs under
  ``pytest -m slow``.
- ``skip``  — parsed but not exercised (e.g. requires real Azure creds).

Executed blocks get a fresh module namespace and a per-test working
directory; registry state is restored afterwards so test-local model /
dataset registrations don't bleed into sibling tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

README_PATH = Path(__file__).parents[2] / "README.md"

_BLOCK_RE = re.compile(
    r"<!--\s*readme-test:\s*"
    r"(?P<mode>fast|slow|skip)"
    r"(?:\s*;\s*id=(?P<id>[\w-]+))?"
    r"[^>]*-->\s*\n```python\n(?P<body>.*?)\n```",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ReadmeBlock:
    mode: str
    identifier: str
    source: str


def _extract() -> list[ReadmeBlock]:
    text = README_PATH.read_text(encoding="utf-8")
    blocks: list[ReadmeBlock] = []
    for idx, match in enumerate(_BLOCK_RE.finditer(text)):
        ident = match.group("id") or f"block-{idx}"
        blocks.append(
            ReadmeBlock(mode=match.group("mode"), identifier=ident, source=match.group("body"))
        )
    return blocks


_ALL_BLOCKS = _extract()
_FAST = [b for b in _ALL_BLOCKS if b.mode == "fast"]
_SLOW = [b for b in _ALL_BLOCKS if b.mode == "slow"]


def _run(block: ReadmeBlock) -> None:
    namespace: dict[str, Any] = {"__name__": "__readme__"}
    compiled = compile(block.source, f"README.md::{block.identifier}", "exec")
    exec(compiled, namespace)


@pytest.fixture
def _populate_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Seed a minimal `./results` directory for `load_results`-style examples."""
    import json

    run_dir = tmp_path / "results" / "udhr" / "GlotLID"
    run_dir.mkdir(parents=True)

    summary = {
        "schema_version": 2,
        "model_id": "GlotLID",
        "dataset_id": "udhr",
        "dataset_revision": "deadbeef",
        "commonlid_version": "0.1.0",
        "python_version": "3.13.0",
        "platform": "test",
        "timestamp": "2026-04-20T00:00:00+00:00",
        "limit": None,
        "n_samples": 3,
        "n_samples_with_gold": 3,
        "samples_per_second": 100.0,
        "macro": {
            "precision_gold_only": 1.0,
            "recall_gold_only": 1.0,
            "f1_gold_only": 1.0,
            "n_languages_gold": 1,
            "precision_observed": 1.0,
            "recall_observed": 1.0,
            "f1_observed": 1.0,
            "n_languages_observed": 1,
        },
        "micro": {
            "precision_gold_only": 1.0,
            "recall_gold_only": 1.0,
            "f1_gold_only": 1.0,
            "n_correct_gold": 3,
            "n_predictions_gold": 3,
            "n_gold_samples": 3,
            "precision_observed": 1.0,
            "recall_observed": 1.0,
            "f1_observed": 1.0,
            "n_correct_observed": 3,
            "n_predictions_observed": 3,
        },
        "per_language": {
            "eng": {
                "gt_count": 3,
                "predictions": 3,
                "correct": 3,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        },
        "extra": {},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    rows = [
        {
            "idx": i,
            "text_hash": f"hash{i}",
            "gold": "eng",
            "pred": "eng",
            "correct": True,
        }
        for i in range(3)
    ]
    (run_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_registry(fresh_registry: None) -> None:
    """Every block gets a pristine copy of the registry snapshot."""


def test_readme_has_at_least_one_runnable_block() -> None:
    assert _ALL_BLOCKS, "README.md contains no tagged python blocks"
    assert _FAST, "README.md must have at least one fast-tested block"


@pytest.mark.parametrize("block", _FAST, ids=[b.identifier for b in _FAST])
def test_readme_fast_block(
    block: ReadmeBlock,
    _populate_results: Path,  # noqa: PT019
) -> None:
    _run(block)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("block", _SLOW, ids=[b.identifier for b in _SLOW])
def test_readme_slow_block(
    block: ReadmeBlock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _run(block)
