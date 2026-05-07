"""Parity smoke test on CommonLID (200 samples, GlotLID + cld2 + cld3).

Same shape as ``test_smoke_parity.py`` but points at the public
``commoncrawl/CommonLID`` dataset. Marked slow + network.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

pytest.importorskip("datasets")
pytest.importorskip("gcld3")

SAMPLE_LIMIT = 200

# Legacy registers CLD3 as ``gcld3``; the new package exposes it as ``cld3``.
_LEGACY_TO_NEW_MODEL_ID = {"gcld3": "cld3"}


@pytest.mark.slow
@pytest.mark.network
def test_glotlid_cld2_cld3_match_legacy_on_commonlid() -> None:
    from datasets import load_dataset

    ds = load_dataset("commoncrawl/CommonLID", split="test").select(range(SAMPLE_LIMIT))

    legacy_f1 = _run_legacy(ds)
    new_f1 = _run_new(ds)

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
        label="commonlid",
        dataset=dataset,
        text_column="text",
        target_iso639_3_column="tag",
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

    class _CommonLIDSlice(LIDDataset):
        dataset_id = "_commonlid_slice"
        source_hf_repo = "commoncrawl/CommonLID"
        text_column = "text"
        target_column = "tag"

        def load(self, *, limit: int | None = None):  # type: ignore[override]
            self._dataset = dataset
            return dataset

        def iter_batches(  # type: ignore[override]
            self, batch_size: int = 64, *, limit: int | None = None
        ) -> Iterator[tuple[list[str], list[str | None]]]:
            for batch in dataset.iter(batch_size=batch_size):
                yield batch[self.text_column], list(batch[self.target_column])

    lid_dataset = _CommonLIDSlice()
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
