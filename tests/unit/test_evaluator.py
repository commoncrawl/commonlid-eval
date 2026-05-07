from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.lid_model import LIDModel
from commonlid.evaluation.evaluator import Evaluator


class _FakeDataset(LIDDataset):
    dataset_id = "_fake"
    source_hf_repo = "fake/fake"
    source_hf_revision = "rev-1"
    text_column = "text"
    target_column = "iso639_3"

    SAMPLES: ClassVar[list[dict[str, str | None]]] = [
        {"text": "hello world", "iso639_3": "eng"},
        {"text": "hallo welt", "iso639_3": "deu"},
        {"text": "bonjour", "iso639_3": "fra"},
        {"text": "ciao", "iso639_3": "ita"},
    ]

    def load(self, *, limit: int | None = None):  # type: ignore[override]
        from datasets import Dataset

        ds = Dataset.from_list(self.SAMPLES)
        if limit is not None and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        self._dataset = ds
        return ds


class _AlwaysEng(LIDModel):
    model_id = "_always_eng"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return ["eng"] * len(texts)


class _PerfectOracle(LIDModel):
    """Looks up the known fixture answer by text."""

    model_id = "_oracle"
    requires_preprocessing = False

    _ANSWERS: ClassVar[dict[str, str]] = {
        s["text"]: s["iso639_3"] for s in _FakeDataset.SAMPLES if s["iso639_3"] is not None
    }

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return [self._ANSWERS.get(t) for t in texts]


def test_end_to_end_writes_expected_files(tmp_output_dir: Path) -> None:
    ev = Evaluator(
        models=[_AlwaysEng(), _PerfectOracle()],
        datasets=[_FakeDataset()],
        output_dir=tmp_output_dir,
        batch_size=2,
    )
    results = ev.run()
    assert len(results) == 2

    eng_dir = tmp_output_dir / "_fake" / "_always_eng"
    assert (eng_dir / "predictions.jsonl").exists()
    summary = json.loads((eng_dir / "summary.json").read_text())
    assert summary["model_id"] == "_always_eng"
    assert summary["n_samples"] == 4
    # _always_eng gets eng right 1/4 times; recall(eng)=1, precision(eng)=1/4
    per_lang = summary["per_language"]
    assert per_lang["eng"]["precision"] == pytest.approx(0.25)
    assert per_lang["eng"]["recall"] == pytest.approx(1.0)

    oracle_sum = json.loads((tmp_output_dir / "_fake" / "_oracle" / "summary.json").read_text())
    assert oracle_sum["macro"]["f1_gold_only"] == pytest.approx(1.0)


def test_cache_reuses_predictions_second_run(tmp_output_dir: Path) -> None:
    model = _PerfectOracle()
    ev = Evaluator(
        models=[model], datasets=[_FakeDataset()], output_dir=tmp_output_dir, batch_size=2
    )
    ev.run()

    # Break the model's predictions; the cache should serve the original.
    class _Broken(_PerfectOracle):
        model_id = "_oracle"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["xxx"] * len(texts)

    ev2 = Evaluator(
        models=[_Broken()], datasets=[_FakeDataset()], output_dir=tmp_output_dir, batch_size=2
    )
    ev2.run()
    summary = json.loads((tmp_output_dir / "_fake" / "_oracle" / "summary.json").read_text())
    # If the cache served correctly, we still get perfect predictions.
    assert summary["macro"]["f1_gold_only"] == pytest.approx(1.0)


def test_no_cache_runs_fresh_predictions(tmp_output_dir: Path) -> None:
    ev = Evaluator(
        models=[_AlwaysEng()],
        datasets=[_FakeDataset()],
        output_dir=tmp_output_dir,
        batch_size=4,
        use_cache=False,
    )
    ev.run()
    cache_dir = tmp_output_dir / ".cache"
    assert not cache_dir.exists()


def test_limit_truncates_dataset(tmp_output_dir: Path) -> None:
    ev = Evaluator(
        models=[_AlwaysEng()],
        datasets=[_FakeDataset()],
        output_dir=tmp_output_dir,
        batch_size=4,
        limit=2,
    )
    ev.run()
    lines = (
        (tmp_output_dir / "_fake" / "_always_eng" / "predictions.jsonl").read_text().splitlines()
    )
    assert len(lines) == 2
