from __future__ import annotations

import json

import pytest

from commonlid.preprocess import openlid_normer_clean_line
from tests.conftest import FIXTURES_DIR


def _golden_pairs() -> list[tuple[str, str]]:
    path = FIXTURES_DIR / "preprocess_golden.jsonl"
    pairs: list[tuple[str, str]] = []
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            pairs.append((entry["raw"], entry["clean"]))
    return pairs


@pytest.mark.parametrize(("raw", "clean"), _golden_pairs())
def test_openlid_normer_matches_golden(raw: str, clean: str) -> None:
    assert openlid_normer_clean_line(raw) == clean


def test_openlid_normer_empty_string() -> None:
    assert openlid_normer_clean_line("") == ""


def test_openlid_normer_whitespace_only() -> None:
    assert openlid_normer_clean_line("   \n\t  ") == ""
