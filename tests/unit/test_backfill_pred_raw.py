"""Tests for ``scripts/backfill_pred_raw.py`` (loaded by path: scripts aren't a package)."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_pred_raw.py"


@pytest.fixture(scope="module")
def backfill() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backfill_pred_raw", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_llm_cache_collects_raw_answers_per_text(backfill: ModuleType, tmp_path: Path) -> None:
    rows = [
        {"text": "hallo welt", "language_iso639_3": "deu"},
        {"text": "hallo welt", "language_iso639_3": " deu"},
        # A failed call stores no answer, sometimes with an `error` column.
        {"text": "bonjour", "language_iso639_3": None, "error": "timeout"},
    ]
    path = tmp_path / "cache_azure_gpt-5_0123456789abcache_0-of-1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "cache_azure_other_0123456789abcache_0-of-1.jsonl").write_text(
        json.dumps({"text": "ignored", "language_iso639_3": "eng"}) + "\n", encoding="utf-8"
    )

    cache = backfill.load_llm_cache(tmp_path, "azure_gpt-5")
    assert cache == {
        "hallo welt": Counter({"deu": 1, " deu": 1}),
        "bonjour": Counter({None: 1}),
    }


@pytest.mark.parametrize(
    ("candidates", "pred", "expected"),
    [
        ({"en": 1}, "eng", ("ok", "en")),
        # An unmappable answer was stored as "und", as was a failed call.
        ({"hindi": 1}, "und", ("ok", "hindi")),
        ({None: 1}, "und", ("ok", None)),
        ({"nah": 1}, None, ("ok", "nah")),
        # A cache from another run can't leak in: only conforming answers count.
        ({"eng": 1, "deu": 1}, "deu", ("ok", "deu")),
        ({"fra": 1}, "deu", ("mismatch", None)),
        ({"en": 1, "eng": 1}, "eng", ("ambiguous", None)),
        (None, "eng", ("missing", None)),
    ],
)
def test_pick_raw(
    backfill: ModuleType,
    candidates: dict[str | None, int] | None,
    pred: str | None,
    expected: tuple[str, str | None],
) -> None:
    assert backfill.pick_raw(candidates, pred) == expected


def test_backfill_rows_only_adds_pred_raw(backfill: ModuleType) -> None:
    rows = [
        {"idx": 0, "text_hash": "h0", "gold": "eng", "pred": "eng", "correct": True},
        {"idx": 1, "text_hash": "h1", "gold": "deu", "pred": "fra", "correct": False},
        {"idx": 2, "text_hash": "h2", "gold": "ita", "pred": "ita", "correct": True},
        {"idx": 3, "text_hash": "h3", "gold": "eng", "pred": "eng", "pred_raw": "x"},
    ]
    # The cache is keyed by the normalised text the LLM saw.
    texts = {"h0": "Hello, World!", "h1": "Hallo Welt", "h3": "Hi"}
    cache = {"hello world": {"en": 1}, "hallo welt": {"deu": 1}}

    stats = backfill.backfill_rows(rows, texts, cache)

    assert stats == Counter(ok=1, mismatch=1, no_text=1, already_set=1)
    assert rows[0]["pred_raw"] == "en"
    assert "pred_raw" not in rows[1]
    assert "pred_raw" not in rows[2]
    assert rows[3]["pred_raw"] == "x"


def _write_run(root: Path, rows: list[dict[str, object]], macro: float) -> None:
    run = root / "ds" / "model"
    run.mkdir(parents=True)
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (run / "summary.json").write_text(json.dumps({"macro": {"f1": macro}}), encoding="utf-8")


def test_check_passes_when_only_pred_raw_was_added(backfill: ModuleType, tmp_path: Path) -> None:
    row = {"idx": 0, "text_hash": "h", "gold": "eng", "pred": "eng", "score": 0.5, "correct": True}
    _write_run(tmp_path / "before", [row], macro=1.0)
    _write_run(tmp_path / "after", [{**row, "pred_raw": "en"}], macro=1.0)
    argv = ["check", "--before", str(tmp_path / "before"), "--after", str(tmp_path / "after")]
    assert backfill.main(argv) == 0


def test_check_fails_when_a_prediction_moved(backfill: ModuleType, tmp_path: Path) -> None:
    row = {"idx": 0, "text_hash": "h", "gold": "eng", "pred": "eng", "score": 0.5, "correct": True}
    _write_run(tmp_path / "before", [row], macro=1.0)
    _write_run(tmp_path / "after", [{**row, "score": 0.6}], macro=1.0)
    argv = ["check", "--before", str(tmp_path / "before"), "--after", str(tmp_path / "after")]
    assert backfill.main(argv) == 1


def test_check_fails_when_a_metric_moved(backfill: ModuleType, tmp_path: Path) -> None:
    row = {"idx": 0, "text_hash": "h", "gold": "eng", "pred": "eng", "score": 0.5, "correct": True}
    _write_run(tmp_path / "before", [row], macro=1.0)
    _write_run(tmp_path / "after", [row], macro=0.9)
    argv = ["check", "--before", str(tmp_path / "before"), "--after", str(tmp_path / "after")]
    assert backfill.main(argv) == 1
