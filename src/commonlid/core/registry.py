"""Decorator-based registry for models and datasets.

Subclasses of :class:`LIDModel` and :class:`LIDDataset` register themselves
by decoration. The :mod:`commonlid.models` and :mod:`commonlid.datasets`
packages import every submodule in their ``__init__`` so side-effects fire
as soon as either package is imported.
"""

from __future__ import annotations

from typing import TypeVar

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.lid_model import LIDModel

_MODEL_REGISTRY: dict[str, type[LIDModel]] = {}
_DATASET_REGISTRY: dict[str, type[LIDDataset]] = {}

_ModelT = TypeVar("_ModelT", bound=LIDModel)
_DatasetT = TypeVar("_DatasetT", bound=LIDDataset)


def register_model(cls: type[_ModelT]) -> type[_ModelT]:
    """Class decorator that registers a :class:`LIDModel` subclass."""
    model_id = getattr(cls, "model_id", None)
    if not model_id:
        msg = f"{cls.__name__} must set a non-empty class attribute `model_id`"
        raise ValueError(msg)
    if model_id in _MODEL_REGISTRY and _MODEL_REGISTRY[model_id] is not cls:
        msg = f"Duplicate model_id registered: {model_id!r}"
        raise ValueError(msg)
    _MODEL_REGISTRY[model_id] = cls
    return cls


def register_dataset(cls: type[_DatasetT]) -> type[_DatasetT]:
    """Class decorator that registers a :class:`LIDDataset` subclass."""
    dataset_id = getattr(cls, "dataset_id", None)
    if not dataset_id:
        msg = f"{cls.__name__} must set a non-empty class attribute `dataset_id`"
        raise ValueError(msg)
    if dataset_id in _DATASET_REGISTRY and _DATASET_REGISTRY[dataset_id] is not cls:
        msg = f"Duplicate dataset_id registered: {dataset_id!r}"
        raise ValueError(msg)
    _DATASET_REGISTRY[dataset_id] = cls
    return cls


def get_model(model_id: str) -> LIDModel:
    """Instantiate a registered model by id. Raises :class:`KeyError` if unknown."""
    if model_id not in _MODEL_REGISTRY:
        msg = f"Unknown model_id: {model_id!r}. Known ids: {sorted(_MODEL_REGISTRY)}"
        raise KeyError(msg)
    return _MODEL_REGISTRY[model_id]()


def get_dataset(dataset_id: str) -> LIDDataset:
    """Instantiate a registered dataset by id. Raises :class:`KeyError` if unknown."""
    if dataset_id not in _DATASET_REGISTRY:
        msg = f"Unknown dataset_id: {dataset_id!r}. Known ids: {sorted(_DATASET_REGISTRY)}"
        raise KeyError(msg)
    return _DATASET_REGISTRY[dataset_id]()


def list_models() -> list[str]:
    """Return registered model ids, sorted alphabetically."""
    return sorted(_MODEL_REGISTRY)


def list_datasets() -> list[str]:
    """Return registered dataset ids, sorted alphabetically."""
    return sorted(_DATASET_REGISTRY)


def _clear_registry_for_tests() -> None:
    """Reset both registries. Test fixtures may use this; not public API."""
    _MODEL_REGISTRY.clear()
    _DATASET_REGISTRY.clear()
