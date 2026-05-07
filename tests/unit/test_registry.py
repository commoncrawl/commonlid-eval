from __future__ import annotations

from collections.abc import Sequence

import pytest

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import (
    get_dataset,
    get_model,
    list_datasets,
    list_models,
    register_dataset,
    register_model,
)


class _DummyModel(LIDModel):
    model_id = ""

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return ["eng"] * len(texts)


class _DummyDataset(LIDDataset):
    dataset_id = ""
    source_hf_repo = "nonexistent/none"
    text_column = "text"
    target_column = "iso639_3"


def test_register_and_get_model(fresh_registry) -> None:
    @register_model
    class FooModel(_DummyModel):
        model_id = "foo"

    assert "foo" in list_models()
    inst = get_model("foo")
    assert isinstance(inst, FooModel)
    assert inst.predict(["hi"]) == ["eng"]


def test_register_requires_model_id(fresh_registry) -> None:
    class UnnamedModel(_DummyModel):
        pass

    with pytest.raises(ValueError, match="model_id"):
        register_model(UnnamedModel)


def test_register_rejects_duplicates(fresh_registry) -> None:
    @register_model
    class A(_DummyModel):
        model_id = "same"

    class B(_DummyModel):
        model_id = "same"

    with pytest.raises(ValueError, match="Duplicate model_id"):
        register_model(B)


def test_register_dataset(fresh_registry) -> None:
    @register_dataset
    class FooDS(_DummyDataset):
        dataset_id = "foo_ds"

    assert "foo_ds" in list_datasets()
    assert isinstance(get_dataset("foo_ds"), FooDS)


def test_register_dataset_requires_dataset_id(fresh_registry) -> None:
    class _Unnamed(_DummyDataset):
        pass

    with pytest.raises(ValueError, match="dataset_id"):
        register_dataset(_Unnamed)


def test_register_dataset_rejects_duplicates(fresh_registry) -> None:
    @register_dataset
    class X(_DummyDataset):
        dataset_id = "dup"

    class Y(_DummyDataset):
        dataset_id = "dup"

    with pytest.raises(ValueError, match="Duplicate dataset_id"):
        register_dataset(Y)


def test_get_model_unknown(fresh_registry) -> None:
    with pytest.raises(KeyError):
        get_model("never-registered")


def test_get_dataset_unknown(fresh_registry) -> None:
    with pytest.raises(KeyError):
        get_dataset("never-registered")


def test_register_same_class_twice_is_ok(fresh_registry) -> None:
    @register_model
    class Z(_DummyModel):
        model_id = "z"

    register_model(Z)  # re-registering the same class must not raise
    assert "z" in list_models()
