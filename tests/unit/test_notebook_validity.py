"""Lightweight sanity checks for the analysis notebook.

We only parse the JSON structure and check that each code cell's source can
be byte-compiled; execution is left to the user (requires a running kernel).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).parents[2] / "notebooks" / "paper_tables.ipynb"


def test_notebook_is_valid_json() -> None:
    data = json.loads(NOTEBOOK_PATH.read_text())
    assert data["nbformat"] == 4
    assert isinstance(data["cells"], list)
    assert len(data["cells"]) > 0


@pytest.mark.parametrize("cell_idx", range(len(json.loads(NOTEBOOK_PATH.read_text())["cells"])))
def test_code_cells_are_syntactically_valid(cell_idx: int) -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    cell = nb["cells"][cell_idx]
    if cell["cell_type"] != "code":
        return
    source = "".join(cell["source"])
    ast.parse(source)
