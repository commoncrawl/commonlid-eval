"""Parity smoke test: new Evaluator vs legacy research code.

Runs GlotLID, cld2 and cld3 on a small slice of UDHR-LID and asserts that
the per-language F1 computed by the new pipeline matches the legacy
pipeline within ``1e-6``. Marked ``slow`` because it downloads the GlotLID
weights and the UDHR dataset from HuggingFace.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

pytest.importorskip("datasets")
pytest.importorskip("gcld3")

SAMPLE_LIMIT = 200

# The legacy harness registers CLD3 under the historic ``gcld3`` name; the
# new package exposes it as ``cld3``. Normalise legacy keys so per-language
# F1 dicts can be compared directly.
_LEGACY_TO_NEW_MODEL_ID = {"gcld3": "cld3"}


@pytest.mark.slow
@pytest.mark.network
def test_glotlid_cld2_cld3_match_legacy() -> None:
    from datasets import load_dataset

    ds = load_dataset(
        "cis-lmu/udhr-lid",
        split="test",
        revision="6908db2a27c296158da7e69782d15df911652184",
    ).select(range(SAMPLE_LIMIT))

    legacy_f1 = _run_legacy(ds)
    new_f1 = _run_new(ds)

    # Same language set.
    assert set(legacy_f1) == set(new_f1), (
        f"Language sets differ. Legacy only: {set(legacy_f1) - set(new_f1)}, "
        f"new only: {set(new_f1) - set(legacy_f1)}"
    )
    for model_lang, f1_legacy in legacy_f1.items():
        f1_new = new_f1[model_lang]
        assert math.isclose(f1_new, f1_legacy, abs_tol=1e-6), (
            f"F1 mismatch for {model_lang}: legacy={f1_legacy}, new={f1_new}"
        )


def _run_legacy(dataset: Any) -> dict[tuple[str, str], float]:
    from tests.legacy.langid_datasets import EvalLangIDDatasets
    from tests.legacy.langid_models import EvalLangIDModels

    models = EvalLangIDModels(models_to_register=["GlotLID", "cld2", "gcld3"])
    datasets = EvalLangIDDatasets()
    datasets.register_dataset(
        label="udhr",
        dataset=dataset,
        text_column="sentence",
        target_iso639_3_column="iso639-3",
    )
    records, _ = datasets.eval_all(models, batch_size=64)
    out: dict[tuple[str, str], float] = {}
    for r in records:
        legacy_model = r["model"]
        model = _LEGACY_TO_NEW_MODEL_ID.get(legacy_model, legacy_model)
        out[model, r["language iso639-3"]] = r["f1"]
    return out


def _run_new(dataset: Any) -> dict[tuple[str, str], float]:
    from collections.abc import Iterator

    from commonlid.core.lid_dataset import LIDDataset
    from commonlid.core.registry import get_model
    from commonlid.metrics.core import compute_per_language_metrics

    class _UDHRSlice(LIDDataset):
        dataset_id = "_udhr_slice"
        source_hf_repo = "cis-lmu/udhr-lid"
        source_hf_revision = "6908db2a27c296158da7e69782d15df911652184"
        text_column = "sentence"
        target_column = "iso639-3"

        def load(self, *, limit: int | None = None):  # type: ignore[override]
            self._dataset = dataset
            return dataset

        def iter_batches(  # type: ignore[override]
            self, batch_size: int = 64, *, limit: int | None = None
        ) -> Iterator[tuple[list[str], list[str | None]]]:
            for batch in dataset.iter(batch_size=batch_size):
                yield batch[self.text_column], list(batch[self.target_column])

    lid_dataset = _UDHRSlice()
    out: dict[tuple[str, str], float] = {}
    for model_id in ("GlotLID", "cld2", "cld3"):
        model = get_model(model_id)
        ytrue: list[str | None] = []
        ypred: list[str | None] = []
        for texts, golds in lid_dataset.iter_batches(batch_size=64):
            preds = model.predict(texts)
            ytrue.extend(golds)
            ypred.extend(preds)
        metrics = compute_per_language_metrics(ytrue, ypred)
        for lang, m in metrics.items():
            out[model_id, lang] = m.f1
    return out
