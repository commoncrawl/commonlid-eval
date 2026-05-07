"""CommonLID — language identification model/benchmark evaluation."""

# Import the submodule packages so every shipped model/dataset registers itself
# on bare ``import commonlid``. These imports are side-effect-only (the heavy
# dependencies — fasttext weights, transformers, dspy — load lazily inside each
# model's ``load()``), so this stays cheap.
from commonlid import datasets as _tasks  # noqa: F401
from commonlid import models as _models  # noqa: F401
from commonlid._version import __version__
from commonlid.core.lid_dataset import LIDDataset, PrivateDatasetAccessError
from commonlid.core.lid_model import LIDModel, LIDPrediction
from commonlid.core.registry import (
    get_dataset,
    get_model,
    list_datasets,
    list_models,
    register_dataset,
    register_model,
)
from commonlid.evaluation.evaluator import Evaluator
from commonlid.evaluation.results import Result

__all__ = [
    "Evaluator",
    "LIDDataset",
    "LIDModel",
    "LIDPrediction",
    "PrivateDatasetAccessError",
    "Result",
    "__version__",
    "get_dataset",
    "get_model",
    "list_datasets",
    "list_models",
    "register_dataset",
    "register_model",
]
