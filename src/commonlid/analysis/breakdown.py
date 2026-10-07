"""Sample- and language-level breakdown across the top-k models of a dataset.

The published results dataset stores one ``predictions.jsonl`` per
``(dataset, model)`` pair (``idx``, ``text_hash``, ``gold``, ``pred``, ...).
This module lines those files up by ``idx`` for a handful of models and
reports, per sample and per gold language, how often the models get it right
or wrong, what they predict instead and how much they agree.

Useful readings of the output:

- **Hard samples / languages** -- high ``error_rate``: most strong models miss
  them. ``top_confusion`` / ``top_wrong_pred`` names the usual culprit.
- **Easy samples / languages** -- ``error_rate`` near 0: every model gets
  them right (``all_correct_share`` at language level).
- **Annotation-error candidates** -- ``label_suspect``: the models not only
  miss the gold label, they *agree* on a different (non-``und``) one. With
  several strong, independently trained models that is often a sign the gold
  label itself is wrong. It is a triage signal, not a verdict.

Everything operates on plain DataFrames so the functions can later back a
leaderboard drilldown without touching the loading code.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from commonlid.metrics.core import UND_TOKEN

if TYPE_CHECKING:
    from commonlid.core.lid_dataset import LIDDataset

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.jsonl"

#: Default share of models that must agree on the same non-gold label for a
#: sample to be flagged as a possible annotation error.
DEFAULT_SUSPECT_MIN_AGREEMENT = 0.8

_PRED_PREFIX = "pred:"
_RECALL_PREFIX = "recall:"


def select_top_models(
    results: Any,
    dataset_id: str,
    k: int,
    *,
    metric: str = "macro_f1",
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Return the ``k`` best model ids on ``dataset_id`` ranked by ``metric``.

    ``results`` is the DataFrame returned by
    :func:`commonlid.leaderboard.load_results`. Models without a value for
    ``metric`` (e.g. a ``*_cov`` metric for an LLM) are skipped. Ties are
    broken by model id so the selection is deterministic.
    """
    if k < 1:
        msg = f"k must be >= 1, got {k}"
        raise ValueError(msg)
    if metric not in results.columns:
        msg = f"unknown metric {metric!r}; available: {sorted(results.columns)}"
        raise ValueError(msg)
    rows = results[results["dataset_id"] == dataset_id]
    excluded = set(exclude or ())
    rows = rows[~rows["model_id"].isin(excluded) & rows[metric].notna()]
    rows = rows.sort_values([metric, "model_id"], ascending=[False, True], kind="stable")
    return [str(m) for m in rows["model_id"].head(k)]


def _snapshot_predictions(
    repo_id: str,
    dataset_id: str,
    model_ids: Sequence[str],
    revision: str | None,
    cache_dir: str | Path | None,
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            allow_patterns=[f"{dataset_id}/{m}/{PREDICTIONS_FILENAME}" for m in model_ids],
        )
    )


def load_predictions(
    dataset_id: str,
    model_ids: Sequence[str],
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Any:
    """Load per-sample predictions of ``model_ids`` on ``dataset_id``.

    Returns a long DataFrame with columns ``model_id``, ``idx``,
    ``text_hash``, ``gold`` and ``pred`` (``None`` predictions are kept as
    ``None``). Only the requested ``predictions.jsonl`` files are downloaded.
    Models without a predictions file are logged and skipped.

    ``local_dir`` bypasses the Hub and reads ``<local_dir>/<dataset>/<model>/``
    directly (same layout as ``commonlid run --output-dir``).
    """
    import pandas as pd

    from commonlid.leaderboard.data import DEFAULT_REPO_ID

    if local_dir is not None:
        root = Path(local_dir)
    else:
        root = _snapshot_predictions(
            repo_id or DEFAULT_REPO_ID, dataset_id, model_ids, revision, cache_dir
        )

    columns = ["model_id", "idx", "text_hash", "gold", "pred"]
    frames = []
    for model_id in model_ids:
        path = root / dataset_id / model_id / PREDICTIONS_FILENAME
        if not path.is_file():
            logger.warning(
                "no %s for %s on %s; skipping", PREDICTIONS_FILENAME, model_id, dataset_id
            )
            continue
        # dtype=False keeps language codes as plain strings (no numeric/date coercion).
        df = pd.read_json(path, lines=True, dtype=False)
        for col in ("text_hash", "gold", "pred"):
            if col not in df.columns:
                df[col] = None
        df["model_id"] = model_id
        frames.append(df[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    out["idx"] = out["idx"].astype(int)
    # Missing values come back as NaN from JSON; normalise to None.
    return out.astype(object).where(out.notna(), None).astype({"idx": int})


def _model_order(preds: Any) -> list[str]:
    return list(dict.fromkeys(preds["model_id"]))


@cache
def to_macrolanguage(code: str) -> str:
    """Map an individual ISO 639-3 code to its macrolanguage (``lvs`` -> ``lav``).

    Codes that are not part of a macrolanguage (or unknown to ``iso639``)
    are returned unchanged.
    """
    from iso639 import Lang

    try:
        macro = Lang(code).macro()
    except Exception:
        return code
    return str(macro.pt3) if macro is not None and macro.pt3 else code


def _prepare(preds: Any, *, collapse_macrolanguages: bool = False) -> Any:
    """Drop gold-less rows and add ``gold_cmp`` / ``pred_norm`` / ``correct`` columns.

    ``gold_cmp`` and ``pred_norm`` are the labels actually compared: ``None``
    predictions become ``und`` and, with ``collapse_macrolanguages``, both
    sides are mapped to their macrolanguage.
    """
    df = preds[preds["gold"].notna()].copy()
    df["pred_norm"] = df["pred"].where(df["pred"].notna(), UND_TOKEN)
    df["gold_cmp"] = df["gold"]
    if collapse_macrolanguages:
        df["pred_norm"] = df["pred_norm"].map(to_macrolanguage)
        df["gold_cmp"] = df["gold_cmp"].map(to_macrolanguage)
    df["correct"] = df["pred_norm"] == df["gold_cmp"]
    return df


def _top_label(df: Any, key: str, label: str, count_name: str) -> Any:
    """Most frequent ``label`` per ``key`` (ties broken alphabetically)."""
    counts = df.groupby([key, label], sort=False).size().reset_index(name=count_name)
    counts = counts.sort_values(
        [key, count_name, label], ascending=[True, False, True], kind="stable"
    )
    return counts.drop_duplicates(key).set_index(key)


def sample_breakdown(
    preds: Any,
    *,
    suspect_min_agreement: float = DEFAULT_SUSPECT_MIN_AGREEMENT,
    collapse_macrolanguages: bool = False,
) -> Any:
    """Per-sample breakdown across the models present in ``preds``.

    ``preds`` is the long frame from :func:`load_predictions`. Samples without
    a gold label are dropped. ``None`` predictions count as ``und`` (the same
    convention as the evaluator), so abstaining is an error. With
    ``collapse_macrolanguages`` an individual language and its macrolanguage
    (``lvs`` / ``lav``) count as a match, and ``consensus_pred`` /
    ``top_wrong_pred`` are reported at the macrolanguage level.

    Returned columns (sorted by ``error_rate`` descending, then ``agreement``
    descending):

    - ``idx``, ``text_hash``, ``gold``
    - ``n_models`` -- models with a prediction for this sample
    - ``n_correct`` / ``n_wrong`` / ``error_rate`` (= ``n_wrong / n_models``)
    - ``consensus_pred`` / ``agreement`` -- the most common prediction and the
      share of models that made it (1.0 = all models predict the same label,
      whether right or wrong)
    - ``top_wrong_pred`` / ``top_wrong_count`` -- most common wrong prediction
    - ``label_suspect`` -- at least two models, and at least
      ``suspect_min_agreement`` of them agree on the same label that is
      neither the gold nor ``und``
    - ``pred:<model_id>`` -- each model's raw prediction
    """
    import pandas as pd

    df = _prepare(preds, collapse_macrolanguages=collapse_macrolanguages)
    models = _model_order(df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "idx",
                "text_hash",
                "gold",
                "n_models",
                "n_correct",
                "n_wrong",
                "error_rate",
                "consensus_pred",
                "agreement",
                "top_wrong_pred",
                "top_wrong_count",
                "label_suspect",
                *(f"{_PRED_PREFIX}{m}" for m in models),
            ]
        )

    grouped = df.groupby("idx", sort=True)
    out = grouped.agg(
        text_hash=("text_hash", "first"),
        gold=("gold", "first"),
        gold_cmp=("gold_cmp", "first"),
        n_models=("model_id", "size"),
        n_correct=("correct", "sum"),
    )
    n_gold_labels = grouped["gold"].nunique()
    if (n_gold_labels > 1).any():
        logger.warning(
            "%d sample(s) have different gold labels across models; using the first",
            int((n_gold_labels > 1).sum()),
        )
    out["n_correct"] = out["n_correct"].astype(int)
    out["n_wrong"] = out["n_models"] - out["n_correct"]
    out["error_rate"] = out["n_wrong"] / out["n_models"]

    consensus = _top_label(df, "idx", "pred_norm", "consensus_count")
    out["consensus_pred"] = consensus["pred_norm"]
    out["agreement"] = consensus["consensus_count"] / out["n_models"]

    wrong = _top_label(df[~df["correct"]], "idx", "pred_norm", "top_wrong_count")
    out["top_wrong_pred"] = wrong["pred_norm"].reindex(out.index)
    out["top_wrong_pred"] = out["top_wrong_pred"].astype(object)
    out["top_wrong_pred"] = out["top_wrong_pred"].where(out["top_wrong_pred"].notna(), None)
    out["top_wrong_count"] = wrong["top_wrong_count"].reindex(out.index).fillna(0).astype(int)

    out["label_suspect"] = (
        (out["n_models"] >= 2)
        & (out["consensus_pred"] != out["gold_cmp"])
        & (out["consensus_pred"] != UND_TOKEN)
        & (out["agreement"] >= suspect_min_agreement)
    )

    wide = df.pivot(index="idx", columns="model_id", values="pred")
    for m in models:
        out[f"{_PRED_PREFIX}{m}"] = wide[m].reindex(out.index)

    out = out.drop(columns="gold_cmp").reset_index()
    out = out.sort_values(
        ["error_rate", "agreement", "idx"], ascending=[False, False, True], kind="stable"
    )
    return out.reset_index(drop=True)


def language_breakdown(
    preds: Any,
    *,
    samples: Any | None = None,
    suspect_min_agreement: float = DEFAULT_SUSPECT_MIN_AGREEMENT,
    collapse_macrolanguages: bool = False,
    min_samples: int = 1,
) -> Any:
    """Per-gold-language breakdown across the models present in ``preds``.

    Returned columns (sorted by ``error_rate`` descending):

    - ``language`` -- gold ISO 639-3 code
    - ``n_samples`` -- samples with this gold label
    - ``error_rate`` -- mean per-sample error rate (= 1 - mean model recall)
    - ``agreement`` -- mean per-sample ``agreement`` (how often the models
      predict the same label as each other)
    - ``all_wrong_share`` / ``all_correct_share`` -- share of samples every
      model got wrong / right
    - ``n_label_suspect`` -- samples flagged by :func:`sample_breakdown`
    - ``top_confusion`` / ``top_confusion_share`` -- most common wrong
      prediction over all (model, sample) pairs and its share of the errors
    - ``recall:<model_id>`` -- each model's recall on this language

    ``samples`` may pass a precomputed :func:`sample_breakdown` frame to avoid
    recomputing it (it must use the same ``collapse_macrolanguages``).
    Languages with fewer than ``min_samples`` samples are dropped.
    """
    import pandas as pd

    df = _prepare(preds, collapse_macrolanguages=collapse_macrolanguages)
    models = _model_order(df)
    if samples is None:
        samples = sample_breakdown(
            preds,
            suspect_min_agreement=suspect_min_agreement,
            collapse_macrolanguages=collapse_macrolanguages,
        )
    columns = [
        "language",
        "n_samples",
        "error_rate",
        "agreement",
        "all_wrong_share",
        "all_correct_share",
        "n_label_suspect",
        "top_confusion",
        "top_confusion_share",
        *(f"{_RECALL_PREFIX}{m}" for m in models),
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    s = samples.assign(
        all_wrong=samples["n_correct"] == 0,
        all_correct=samples["n_wrong"] == 0,
    )
    out = s.groupby("gold", sort=True).agg(
        n_samples=("idx", "size"),
        error_rate=("error_rate", "mean"),
        agreement=("agreement", "mean"),
        all_wrong_share=("all_wrong", "mean"),
        all_correct_share=("all_correct", "mean"),
        n_label_suspect=("label_suspect", "sum"),
    )
    out["n_label_suspect"] = out["n_label_suspect"].astype(int)

    errors = df[~df["correct"]]
    confusion = _top_label(errors, "gold", "pred_norm", "confusion_count")
    n_errors = errors.groupby("gold").size()
    out["top_confusion"] = confusion["pred_norm"].reindex(out.index).astype(object)
    out["top_confusion"] = out["top_confusion"].where(out["top_confusion"].notna(), None)
    out["top_confusion_share"] = (
        (confusion["confusion_count"] / n_errors).reindex(out.index).fillna(0.0)
    )

    recall = df.groupby(["gold", "model_id"])["correct"].mean().unstack("model_id")
    for m in models:
        out[f"{_RECALL_PREFIX}{m}"] = recall[m].reindex(out.index)

    out = out[out["n_samples"] >= min_samples]
    out = out.rename_axis("language").reset_index()
    out = out.sort_values(
        ["error_rate", "n_samples", "language"], ascending=[False, False, True], kind="stable"
    )
    return out[columns].reset_index(drop=True)


def attach_texts(samples: Any, dataset: LIDDataset | str) -> Any:
    """Return ``samples`` with a ``text`` column joined in from ``dataset``.

    Predictions only carry a 16-char text hash, so the text is looked up by
    ``idx`` in the registered dataset and cross-checked against ``text_hash``.
    Rows whose hash does not match (e.g. the dataset revision moved) get
    ``text=None`` and a warning is logged. ``dataset`` may be a registered
    dataset id or an :class:`LIDDataset` instance.
    """
    from commonlid.evaluation.evaluator import _text_hash

    if isinstance(dataset, str):
        import commonlid.datasets  # noqa: F401  # populate the registry
        from commonlid.core.registry import get_dataset

        dataset = get_dataset(dataset)

    out = samples.copy()
    if out.empty:
        out["text"] = []
        return out
    ds = dataset.load()
    n_rows = len(ds)
    indices = [int(i) for i in out["idx"]]
    in_range = [i for i in indices if 0 <= i < n_rows]
    texts_by_idx = dict(zip(in_range, ds.select(in_range)[dataset.text_column], strict=True))

    texts: list[str | None] = []
    mismatched = 0
    for idx, expected in zip(indices, out["text_hash"], strict=True):
        text = texts_by_idx.get(idx)
        if text is None or (expected is not None and _text_hash(text) != expected):
            mismatched += 1
            texts.append(None)
        else:
            texts.append(text)
    if mismatched:
        logger.warning(
            "%d of %d sample(s) could not be matched to %s by idx + text_hash; text left empty",
            mismatched,
            len(indices),
            dataset.dataset_id,
        )
    out["text"] = texts
    return out
