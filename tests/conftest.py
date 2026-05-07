"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import commonlid.datasets  # noqa: F401

# Importing these at module load populates the model/dataset registries up-front
# so the `fresh_registry` fixture snapshots the full baseline.
import commonlid.models  # noqa: F401
from commonlid.core import registry as _registry

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "results"
    out.mkdir()
    return out


@pytest.fixture
def fresh_registry() -> Iterator[None]:
    """Snapshot + restore the global model/dataset registries for a test."""
    model_snapshot = dict(_registry._MODEL_REGISTRY)
    dataset_snapshot = dict(_registry._DATASET_REGISTRY)
    try:
        yield
    finally:
        _registry._MODEL_REGISTRY.clear()
        _registry._MODEL_REGISTRY.update(model_snapshot)
        _registry._DATASET_REGISTRY.clear()
        _registry._DATASET_REGISTRY.update(dataset_snapshot)
