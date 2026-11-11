"""Sample-, language- and model-level breakdown across several models of a dataset.

The published results dataset stores one ``predictions.jsonl`` per
``(dataset, model)`` pair (``idx``, ``text_hash``, ``gold``, ``pred``, ...).
This module lines those files up by ``idx`` for a handful of models and
reports, per sample and per gold language, how often the models get it right
or wrong, what they predict instead and how much they agree. It can also
rescore the models with a more lenient macrolanguage matching.

Useful readings of the output:

- **Hard samples / languages** -- high ``error_rate``: most strong models miss
  them. ``top_confusion`` / ``top_wrong_pred`` names the usual culprit.
- **Easy samples / languages** -- ``error_rate`` near 0: every model gets
  them right (``all_correct_share`` at language level).
- **Annotation-error candidates** -- ``label_suspect``: the models not only
  miss the gold label, they *agree* on a different (non-``und``) one. With
  several strong, independently trained models that is often a sign the gold
  label itself is wrong. It is a triage signal, not a verdict.
- **Conflicting labels** -- ``gold_conflict``: the same text appears more
  than once in the dataset with labels that do not match each other. At
  least one of those rows is wrong for every model.

Everything operates on plain DataFrames so the functions can later back a
leaderboard drilldown without touching the loading code.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from commonlid.metrics.core import UND_TOKEN

if TYPE_CHECKING:
    from commonlid.core.lid_dataset import LIDDataset

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.jsonl"

#: Default share of models that must agree on the same non-gold label for a
#: sample to be flagged as a possible annotation error.
DEFAULT_SUSPECT_MIN_AGREEMENT = 0.8

#: How a predicted label is compared with the gold label:
#:
#: - ``exact``: the codes must be equal (what the official metrics do).
#: - ``macro``: an individual language also matches its own macrolanguage
#:   (``lvs`` / ``lav``, ``cmn`` / ``zho``), in either direction.
#: - ``sibling``: additionally, two individual languages of the same
#:   macrolanguage match each other (``arz`` / ``arb``, ``cmn`` / ``yue``,
#:   ``ind`` / ``zsm``). These are usually real confusions, so this is the
#:   most lenient view.
MacroMatch = Literal["exact", "macro", "sibling"]
MACRO_MATCH_MODES: tuple[str, ...] = get_args(MacroMatch)

_PRED_PREFIX = "pred:"
_RAW_PREFIX = "raw:"
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


_COLUMNS = ["model_id", "idx", "text_hash", "gold", "pred", "pred_raw", "score"]


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
    ``text_hash``, ``gold``, ``pred``, ``pred_raw`` and ``score``. Missing
    values are ``None``: a ``None`` prediction means the model gave no
    supported label, and ``pred_raw`` is ``None`` both when the model gave no
    output and when the raw output was not recorded (older results). Only
    the requested ``predictions.jsonl`` files are downloaded. Models without
    a predictions file are logged and skipped.

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
        for col in _COLUMNS[2:]:
            if col not in df.columns:
                df[col] = None
        df["model_id"] = model_id
        frames.append(df[_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
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


def _check_macro_match(macro_match: str) -> None:
    if macro_match not in MACRO_MATCH_MODES:
        msg = f"macro_match must be one of {MACRO_MATCH_MODES}, got {macro_match!r}"
        raise ValueError(msg)


@cache
def labels_match(gold: str, pred: str, macro_match: MacroMatch = "exact") -> bool:
    """Return whether ``pred`` counts as a correct prediction of ``gold``.

    See :data:`MacroMatch` for the modes. The relation is symmetric. ``und``
    only ever matches ``und``.
    """
    _check_macro_match(macro_match)
    if gold == pred:
        return True
    if macro_match == "exact" or UND_TOKEN in (gold, pred):
        return False
    gold_macro, pred_macro = to_macrolanguage(gold), to_macrolanguage(pred)
    if macro_match == "macro":
        return gold_macro == pred or pred_macro == gold
    return gold_macro == pred_macro


def _prepare(preds: Any, *, macro_match: MacroMatch = "exact") -> Any:
    """Drop gold-less rows and add ``pred_norm`` / ``correct`` columns.

    ``pred_norm`` is the label used for counting: ``None`` predictions become
    ``und``, and a prediction that matches the gold label under
    ``macro_match`` is replaced by the gold label, so that all correct
    predictions of a sample count as the same label. Wrong predictions keep
    their own code.
    """
    _check_macro_match(macro_match)
    df = preds[preds["gold"].notna()].copy()
    df["pred_norm"] = df["pred"].where(df["pred"].notna(), UND_TOKEN)
    df["correct"] = [
        labels_match(g, p, macro_match) for g, p in zip(df["gold"], df["pred_norm"], strict=True)
    ]
    df["pred_norm"] = df["pred_norm"].where(~df["correct"], df["gold"])
    return df


def _top_label(df: Any, key: str, label: str, count_name: str) -> Any:
    """Most frequent ``label`` per ``key`` (ties broken alphabetically)."""
    counts = df.groupby([key, label], sort=False).size().reset_index(name=count_name)
    counts = counts.sort_values(
        [key, count_name, label], ascending=[True, False, True], kind="stable"
    )
    return counts.drop_duplicates(key).set_index(key)


def _check_alignment(df: Any) -> None:
    """Raise if two models have a different text at the same ``idx``.

    Samples are lined up by position, which is only valid if every model was
    run on the same dataset revision.
    """
    hashes = df[df["text_hash"].notna()].groupby("idx")["text_hash"].nunique()
    bad = hashes[hashes > 1]
    if not bad.empty:
        examples = ", ".join(str(i) for i in bad.index[:5])
        msg = (
            f"{len(bad)} sample(s) have a different text_hash across models "
            f"(e.g. idx {examples}); the predictions do not line up. Were the "
            "models run on different revisions of the dataset?"
        )
        raise ValueError(msg)


def _gold_conflicts(samples: Any, macro_match: MacroMatch) -> tuple[Any, Any]:
    """Per sample: whether the same text has a non-matching gold label elsewhere.

    Returns ``(gold_conflict, other_golds)`` indexed like ``samples``.
    ``other_golds`` lists the conflicting labels, comma separated.
    """
    import pandas as pd

    conflict = pd.Series(False, index=samples.index)
    others = pd.Series(None, index=samples.index, dtype=object)
    with_hash = samples[samples["text_hash"].notna()]
    golds_per_text = with_hash.groupby("text_hash")["gold"].agg(lambda s: sorted(set(s)))
    golds_per_text = golds_per_text[golds_per_text.map(len) > 1]
    if golds_per_text.empty:
        return conflict, others
    dup = with_hash[with_hash["text_hash"].isin(golds_per_text.index)]
    for idx, text_hash, gold in zip(dup.index, dup["text_hash"], dup["gold"], strict=True):
        bad = [g for g in golds_per_text[text_hash] if not labels_match(gold, g, macro_match)]
        if bad:
            conflict[idx] = True
            others[idx] = ",".join(bad)
    return conflict, others


def sample_breakdown(
    preds: Any,
    *,
    suspect_min_agreement: float = DEFAULT_SUSPECT_MIN_AGREEMENT,
    macro_match: MacroMatch = "exact",
    include_raw: bool = False,
) -> Any:
    """Per-sample breakdown across the models present in ``preds``.

    ``preds`` is the long frame from :func:`load_predictions`. Samples without
    a gold label are dropped. ``None`` predictions count as ``und`` (the same
    convention as the evaluator), so abstaining is an error. ``macro_match``
    sets how predictions are compared with the gold label (see
    :data:`MacroMatch`).

    Raises :class:`ValueError` if the models disagree on the text at some
    ``idx`` (see :func:`_check_alignment`).

    Returned columns (sorted by ``error_rate`` descending, then ``agreement``
    descending):

    - ``idx``, ``text_hash``, ``gold``
    - ``n_models`` -- models with a prediction for this sample
    - ``n_correct`` / ``n_wrong`` / ``error_rate`` (= ``n_wrong / n_models``)
    - ``consensus_pred`` / ``agreement`` -- the most common prediction and the
      share of models that made it (1.0 = all models predict the same label,
      whether right or wrong). Correct predictions count as the gold label.
    - ``top_wrong_pred`` / ``top_wrong_count`` -- most common wrong prediction
    - ``label_suspect`` -- at least two models, and at least
      ``suspect_min_agreement`` of them, agree on the same wrong label that
      is not ``und``
    - ``gold_conflict`` / ``other_golds`` -- the same text (by
      ``text_hash``) appears elsewhere with a gold label that does not match
      this one, and those labels
    - ``pred:<model_id>`` -- each model's prediction as published
    - ``raw:<model_id>`` -- each model's raw output, with ``include_raw``
    """
    import pandas as pd

    df = _prepare(preds, macro_match=macro_match)
    models = _model_order(df)
    columns = [
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
        "gold_conflict",
        "other_golds",
        *(f"{_PRED_PREFIX}{m}" for m in models),
        *((f"{_RAW_PREFIX}{m}" for m in models) if include_raw else ()),
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    _check_alignment(df)

    grouped = df.groupby("idx", sort=True)
    out = grouped.agg(
        text_hash=("text_hash", "first"),
        gold=("gold", "first"),
        n_models=("model_id", "size"),
        n_correct=("correct", "sum"),
    )
    n_gold_labels = grouped["gold"].nunique()
    if (n_gold_labels > 1).any():
        logger.warning(
            "%d sample(s) have different gold labels across models; using the first",
            int((n_gold_labels > 1).sum()),
        )
    n_partial = int((out["n_models"] < len(models)).sum())
    if n_partial:
        logger.info(
            "%d of %d sample(s) lack a prediction from some of the %d models",
            n_partial,
            len(out),
            len(models),
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

    # Correct predictions were normalised to the gold label, so a consensus
    # different from the gold label is a wrong one.
    out["label_suspect"] = (
        (out["n_models"] >= 2)
        & (out["consensus_pred"] != out["gold"])
        & (out["consensus_pred"] != UND_TOKEN)
        & (out["agreement"] >= suspect_min_agreement)
    )
    out["gold_conflict"], out["other_golds"] = _gold_conflicts(out, macro_match)

    wide = df.pivot(index="idx", columns="model_id", values="pred")
    for m in models:
        out[f"{_PRED_PREFIX}{m}"] = wide[m].reindex(out.index)
    if include_raw:
        wide_raw = df.pivot(index="idx", columns="model_id", values="pred_raw")
        for m in models:
            out[f"{_RAW_PREFIX}{m}"] = wide_raw[m].reindex(out.index)

    # Missing labels are None (as in load_predictions), not NaN.
    label_cols = ["other_golds", *(c for c in columns if c.startswith((_PRED_PREFIX, _RAW_PREFIX)))]
    for col in label_cols:
        out[col] = out[col].astype(object).where(out[col].notna(), None)

    out = out.reset_index()
    out = out.sort_values(
        ["error_rate", "agreement", "idx"], ascending=[False, False, True], kind="stable"
    )
    return out[columns].reset_index(drop=True)


def language_breakdown(
    preds: Any,
    *,
    samples: Any | None = None,
    suspect_min_agreement: float = DEFAULT_SUSPECT_MIN_AGREEMENT,
    macro_match: MacroMatch = "exact",
    min_samples: int = 1,
) -> Any:
    """Per-gold-language breakdown across the models present in ``preds``.

    Returned columns (sorted by ``error_rate`` descending):

    - ``language`` -- gold ISO 639-3 code
    - ``n_samples`` -- samples with this gold label
    - ``error_rate`` -- mean per-sample error rate. When every model has a
      prediction for every sample this is ``1 - mean(model recall)``.
    - ``agreement`` -- mean per-sample ``agreement`` (how often the models
      predict the same label as each other)
    - ``all_wrong_share`` / ``all_correct_share`` -- share of samples every
      model got wrong / right
    - ``n_label_suspect`` / ``n_gold_conflict`` -- samples flagged by
      :func:`sample_breakdown`
    - ``top_confusion`` / ``top_confusion_share`` -- most common wrong
      prediction over all (model, sample) pairs and its share of the errors
    - ``recall:<model_id>`` -- each model's recall on this language

    ``samples`` may pass a precomputed :func:`sample_breakdown` frame to avoid
    recomputing it (it must use the same ``macro_match``).
    Languages with fewer than ``min_samples`` samples are dropped.
    """
    import pandas as pd

    df = _prepare(preds, macro_match=macro_match)
    models = _model_order(df)
    columns = [
        "language",
        "n_samples",
        "error_rate",
        "agreement",
        "all_wrong_share",
        "all_correct_share",
        "n_label_suspect",
        "n_gold_conflict",
        "top_confusion",
        "top_confusion_share",
        *(f"{_RECALL_PREFIX}{m}" for m in models),
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    if samples is None:
        samples = sample_breakdown(
            preds, suspect_min_agreement=suspect_min_agreement, macro_match=macro_match
        )

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
        n_gold_conflict=("gold_conflict", "sum"),
    )
    for col in ("n_label_suspect", "n_gold_conflict"):
        out[col] = out[col].astype(int)

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


def model_breakdown(preds: Any, *, macro_match: MacroMatch = "exact") -> Any:
    """Per-model macro scores, recomputed from ``preds`` under ``macro_match``.

    A prediction that matches the gold label under ``macro_match`` is scored
    as the gold label; everything else is scored as predicted, with the
    official metric code (gold-only macro average over the gold languages).
    With ``macro_match="exact"`` this reproduces the leaderboard numbers.

    Returned columns (sorted by ``macro_f1`` descending):

    - ``model_id``, ``n_samples`` (samples with a gold label)
    - ``macro_f1`` / ``macro_precision`` / ``macro_recall`` and ``rank``
    - ``macro_f1_exact`` / ``rank_exact`` and ``macro_f1_gain``
      (``macro_f1 - macro_f1_exact``), so the effect of the matching mode
      on each model and on the ranking is visible in one table

    Ranks are among the models in ``preds`` only (1 = best; ties share the
    best rank).
    """
    import pandas as pd

    from commonlid.metrics.aggregate import macro_average
    from commonlid.metrics.core import compute_per_language_metrics

    columns = [
        "model_id",
        "n_samples",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "rank",
        "macro_f1_exact",
        "rank_exact",
        "macro_f1_gain",
    ]
    df = _prepare(preds, macro_match=macro_match)
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for model_id, group in df.groupby("model_id", sort=False):
        ytrue = list(group["gold"])
        lenient = macro_average(compute_per_language_metrics(ytrue, list(group["pred_norm"])))
        exact = macro_average(compute_per_language_metrics(ytrue, list(group["pred"])))
        rows.append({
            "model_id": model_id,
            "n_samples": len(group),
            "macro_f1": lenient["f1_gold_only"],
            "macro_precision": lenient["precision_gold_only"],
            "macro_recall": lenient["recall_gold_only"],
            "macro_f1_exact": exact["f1_gold_only"],
        })
    out = pd.DataFrame(rows)
    out["rank"] = out["macro_f1"].rank(ascending=False, method="min").astype(int)
    out["rank_exact"] = out["macro_f1_exact"].rank(ascending=False, method="min").astype(int)
    out["macro_f1_gain"] = out["macro_f1"] - out["macro_f1_exact"]
    out = out.sort_values(["macro_f1", "model_id"], ascending=[False, True], kind="stable")
    return out[columns].reset_index(drop=True)


def attach_texts(samples: Any, dataset: LIDDataset | str) -> Any:
    """Return ``samples`` with a ``text`` column joined in from ``dataset``.

    Predictions only carry a 16-char text hash, so the text is looked up by
    ``idx`` in the registered dataset and cross-checked against ``text_hash``.
    Rows whose hash does not match (e.g. the dataset revision moved) get
    ``text=None`` and a warning is logged. ``dataset`` may be a registered
    dataset id or an :class:`LIDDataset` instance.
    """
    from commonlid.evaluation.results import text_hash

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
        if text is None or (expected is not None and text_hash(text) != expected):
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
