from __future__ import annotations

import pytest

from commonlid.metrics.support_matrix import (
    LANGUAGE_COLUMN,
    load_support_matrix,
    save_support_matrix,
)
from tests.conftest import FIXTURES_DIR


def test_load_fixture() -> None:
    matrix = load_support_matrix(FIXTURES_DIR / "tiny_support_matrix.csv")
    assert set(matrix) == {"model_a", "model_b", "model_c"}
    assert matrix["model_a"] == {"eng", "deu", "fra", "spa", "rus"}
    assert matrix["model_c"] == {"eng", "fra", "zho", "jpn", "rus"}


def test_save_and_reload_round_trip(tmp_path) -> None:
    matrix = {
        "m1": {"eng", "deu"},
        "m2": {"eng", "fra", "zho"},
    }
    path = tmp_path / "matrix.csv"
    save_support_matrix(matrix, path)
    reloaded = load_support_matrix(path)
    assert reloaded == matrix


def test_load_missing_language_column(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("lang,m1\neng,1\n")
    with pytest.raises(ValueError, match=LANGUAGE_COLUMN):
        load_support_matrix(path)
