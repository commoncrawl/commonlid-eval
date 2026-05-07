"""Load and save the language/model support matrix as a CSV."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

LANGUAGE_COLUMN = "language iso639-3"


def load_support_matrix(path: str | Path) -> dict[str, set[str]]:
    """Read a support-matrix CSV into ``{model_id: {iso639_3, ...}}``.

    The CSV must have a ``language iso639-3`` column and one column per
    model id with values ``0``/``1``.
    """
    path = Path(path)
    result: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or LANGUAGE_COLUMN not in reader.fieldnames:
            msg = f"CSV {path} is missing required column '{LANGUAGE_COLUMN}'"
            raise ValueError(msg)
        model_ids = [c for c in reader.fieldnames if c != LANGUAGE_COLUMN]
        for model_id in model_ids:
            result[model_id] = set()
        for row in reader:
            language = row[LANGUAGE_COLUMN]
            for model_id in model_ids:
                if row[model_id].strip() == "1":
                    result[model_id].add(language)
    return result


def save_support_matrix(
    matrix: Mapping[str, set[str]],
    path: str | Path,
) -> None:
    """Write ``{model_id: {iso639_3, ...}}`` out as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_languages = sorted({lang for langs in matrix.values() for lang in langs})
    model_ids = sorted(matrix)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([LANGUAGE_COLUMN, *model_ids])
        for language in all_languages:
            row = [language] + [
                "1" if language in matrix[model_id] else "0" for model_id in model_ids
            ]
            writer.writerow(row)
