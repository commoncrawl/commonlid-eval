from __future__ import annotations

import json

from commonlid.evaluation.results import (
    SCHEMA_VERSION,
    Result,
    load_summary,
    write_predictions,
    write_summary,
)
from commonlid.metrics.core import LanguageMetrics


def _result() -> Result:
    return Result(
        model_id="m",
        dataset_id="d",
        dataset_revision="abc123",
        per_language={
            "eng": LanguageMetrics(
                gt_count=2, predictions=2, correct=2, precision=1.0, recall=1.0, f1=1.0
            ),
        },
        samples_per_second=123.4,
        n_samples=2,
        n_samples_with_gold=2,
        commonlid_version="0.1.0",
        timestamp="2026-04-20T10:00:00+00:00",
    )


def test_summary_has_schema_version() -> None:
    assert _result().summary()["schema_version"] == SCHEMA_VERSION


def test_summary_includes_per_language_and_macros() -> None:
    summary = _result().summary()
    assert summary["per_language"]["eng"]["f1"] == 1.0
    # Schema v2: paper-style "gold-only" view is the headline metric.
    assert summary["macro"]["f1_gold_only"] == 1.0
    assert summary["macro"]["f1_observed"] == 1.0  # collapse since no spurious preds
    assert summary["micro"]["n_correct_gold"] == 2


def test_write_and_load_summary(tmp_path) -> None:
    path = tmp_path / "sum.json"
    write_summary(_result(), path)
    data = load_summary(path)
    assert data["model_id"] == "m"
    assert data["dataset_id"] == "d"
    assert data["dataset_revision"] == "abc123"


def test_write_predictions_roundtrip(tmp_path) -> None:
    rows = [
        {"idx": 0, "gold": "eng", "pred": "eng", "correct": True},
        {"idx": 1, "gold": "deu", "pred": None, "correct": False},
    ]
    path = tmp_path / "preds.jsonl"
    write_predictions(rows, path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == rows


def test_summary_supported_languages_roundtrips_list() -> None:
    """A declared support set ships through ``summary()`` as the sorted list."""
    r = _result()
    r.supported_languages = ["eng", "fra"]
    assert _result().summary()["supported_languages"] is None  # default still None
    assert r.summary()["supported_languages"] == ["eng", "fra"]


def test_summary_supported_languages_roundtrips_none(tmp_path) -> None:
    """``None`` means *undefined* (e.g. LLM rows) and must serialize as JSON ``null``.

    Round-trip via write_summary -> load_summary guards against an accidental
    coercion to ``[]``, which has a distinct semantic meaning (a model that
    declared zero supported languages).
    """
    path = tmp_path / "sum.json"
    write_summary(_result(), path)
    raw = path.read_text(encoding="utf-8")
    # Verify JSON literal -- ``[]`` would be a wrong but type-compatible answer.
    assert '"supported_languages": null' in raw
    assert load_summary(path)["supported_languages"] is None


def test_summary_supported_languages_preserves_empty_list() -> None:
    """``[]`` is degenerate but real -- distinct from ``None``."""
    r = _result()
    r.supported_languages = []
    assert r.summary()["supported_languages"] == []
