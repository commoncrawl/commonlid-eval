"""The evaluation driver: orchestrates models x datasets -> JSONL + summary JSON."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from commonlid.evaluation.cache import PredictionCache
from commonlid.evaluation.results import Result, write_predictions, write_summary
from commonlid.metrics.core import UND_TOKEN, compute_per_language_metrics

if TYPE_CHECKING:
    from commonlid.core.lid_dataset import LIDDataset
    from commonlid.core.lid_model import LIDModel

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_SUBDIR = ".cache"


@dataclass(slots=True)
class EvaluatorConfig:
    output_dir: Path
    batch_size: int = 64
    use_cache: bool = True
    limit: int | None = None
    sample_count_threshold: int = 0


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _macro_for_log(per_language: Mapping[str, Any]) -> dict[str, Any]:
    """Cheap macro F1 (gold-only / paper view) for the per-pair completion log line."""
    from commonlid.metrics.aggregate import macro_average

    return macro_average(per_language)


class Evaluator:
    """Run every (model, dataset) pair and persist predictions + summaries."""

    def __init__(
        self,
        models: list[LIDModel],
        datasets: list[LIDDataset],
        output_dir: str | Path,
        *,
        batch_size: int = 64,
        use_cache: bool = True,
        limit: int | None = None,
        sample_count_threshold: int = 0,
    ) -> None:
        self.models = models
        self.datasets = datasets
        self.config = EvaluatorConfig(
            output_dir=Path(output_dir),
            batch_size=batch_size,
            use_cache=use_cache,
            limit=limit,
            sample_count_threshold=sample_count_threshold,
        )

    def run(self) -> list[Result]:
        """Evaluate every (model, dataset) pair. Returns the results in iteration order."""
        pairs: list[tuple[LIDModel, LIDDataset]] = [
            (model, dataset) for dataset in self.datasets for model in self.models
        ]
        n_pairs = len(pairs)
        logger.info(
            "Starting evaluation: %d model(s) x %d dataset(s) = %d pair(s); output_dir=%s",
            len(self.models),
            len(self.datasets),
            n_pairs,
            self.config.output_dir,
        )
        start_total = time.perf_counter()
        results: list[Result] = []
        for pair_index, (model, dataset) in enumerate(pairs, start=1):
            try:
                result = self._run_one(model, dataset, pair_index=pair_index, n_pairs=n_pairs)
            except KeyboardInterrupt:
                logger.warning(
                    "Interrupted during [%d/%d] %s on %s; partial results kept on disk.",
                    pair_index,
                    n_pairs,
                    model.model_id,
                    dataset.dataset_id,
                )
                raise
            results.append(result)
        elapsed_total = time.perf_counter() - start_total
        logger.info(
            "Finished %d pair(s) in %.1fs; output_dir=%s",
            n_pairs,
            elapsed_total,
            self.config.output_dir,
        )
        return results

    def _run_one(
        self,
        model: LIDModel,
        dataset: LIDDataset,
        *,
        pair_index: int = 1,
        n_pairs: int = 1,
    ) -> Result:
        from commonlid import __version__

        prefix = f"[{pair_index}/{n_pairs}]"
        logger.info("%s Evaluating %s on %s", prefix, model.model_id, dataset.dataset_id)
        logger.info("%s   loading model %s ...", prefix, model.model_id)
        model.load()
        logger.info("%s   loading dataset %s ...", prefix, dataset.dataset_id)
        dataset.load(limit=self.config.limit)

        cache = self._make_cache(model, dataset)
        ytrue: list[str | None] = []
        ypred: list[str | None] = []
        prediction_rows: list[dict[str, object]] = []

        total = len(dataset) if self.config.limit is None else min(len(dataset), self.config.limit)
        logger.info("%s   running predictions on %d samples ...", prefix, total)
        start = time.perf_counter()
        with tqdm(total=total, desc=f"{prefix} {model.model_id}/{dataset.dataset_id}") as pbar:
            idx = 0
            for texts, golds in dataset.iter_batches(
                batch_size=self.config.batch_size,
                limit=self.config.limit,
            ):
                preds = self._predict_with_cache(model, cache, texts)
                for text, gold, pred in zip(texts, golds, preds, strict=True):
                    pred_for_metrics = UND_TOKEN if pred is None else pred
                    prediction_rows.append({
                        "idx": idx,
                        "text_hash": _text_hash(text),
                        "gold": gold,
                        "pred": pred,
                        "correct": bool(gold is not None and gold == pred_for_metrics),
                    })
                    ytrue.append(gold)
                    ypred.append(pred)
                    idx += 1
                pbar.update(len(texts))
        elapsed = time.perf_counter() - start

        per_language = compute_per_language_metrics(
            ytrue, ypred, sample_count_threshold=self.config.sample_count_threshold
        )
        n_with_gold = sum(1 for g in ytrue if g is not None)
        samples_per_second = (len(ytrue) / elapsed) if elapsed > 0 else 0.0
        result = Result(
            model_id=model.model_id,
            dataset_id=dataset.dataset_id,
            dataset_revision=dataset.hf_revision,
            per_language=per_language,
            samples_per_second=samples_per_second,
            n_samples=len(ytrue),
            n_samples_with_gold=n_with_gold,
            limit=self.config.limit,
            timestamp=datetime.now(timezone.utc).isoformat(),
            commonlid_version=__version__,
        )

        run_dir = self.config.output_dir / dataset.dataset_id / model.model_id
        write_predictions(prediction_rows, run_dir / "predictions.jsonl")
        write_summary(result, run_dir / "summary.json")
        # Compact one-line summary so multi-pair runs stay greppable. We log
        # the paper-style "gold-only" view because that's what the leaderboard
        # surfaces and what reviewers compare against published numbers.
        macro = _macro_for_log(per_language)
        logger.info(
            "%s   done in %.1fs (%.0f samples/s) -- "
            "macro_f1=%.4f, languages=%d, gold=%d -- wrote %s",
            prefix,
            elapsed,
            samples_per_second,
            macro["f1_gold_only"],
            macro["n_languages_gold"],
            n_with_gold,
            run_dir,
        )
        return result

    def _make_cache(self, model: LIDModel, dataset: LIDDataset) -> PredictionCache | None:
        if not self.config.use_cache:
            return None
        return PredictionCache(
            cache_dir=self.config.output_dir / _DEFAULT_CACHE_SUBDIR,
            model_id=model.model_id,
            dataset_id=dataset.dataset_id,
            dataset_revision=dataset.hf_revision,
        )

    @staticmethod
    def _predict_with_cache(
        model: LIDModel,
        cache: PredictionCache | None,
        texts: list[str],
    ) -> list[str | None]:
        if cache is None:
            return model.predict(texts)

        preds: list[str | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, text in enumerate(texts):
            hit, pred = cache.get(text)
            if hit:
                preds[i] = pred
            else:
                missing_idx.append(i)
                missing_texts.append(text)

        if missing_texts:
            fresh = model.predict(missing_texts)
            pairs: list[tuple[str, str | None]] = []
            for offset, pred in enumerate(fresh):
                i = missing_idx[offset]
                preds[i] = pred
                pairs.append((missing_texts[offset], pred))
            cache.put_many(pairs)
        return preds
