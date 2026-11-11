"""Cross-model analysis of published predictions.

Public API (all pure pandas, so the leaderboard can reuse them later):

- :func:`select_top_models` — rank the models on one dataset by a
  leaderboard metric and keep the best ``k``.
- :func:`load_predictions` — pull ``<dataset>/<model>/predictions.jsonl``
  from the results HF dataset into one long DataFrame.
- :func:`sample_breakdown` — one row per sample: how many of the selected
  models got it wrong, what they predicted instead, whether the models
  agree on a different label (``label_suspect``) and whether the same text
  has conflicting gold labels (``gold_conflict``).
- :func:`language_breakdown` — the same, aggregated per gold language.
- :func:`model_breakdown` — macro scores per model, recomputed under a
  macrolanguage matching mode, next to the exact-match scores.
- :func:`labels_match` / :data:`MacroMatch` — the matching modes.
- :func:`attach_texts` — join the sample text back in from the registered
  dataset (predictions only store a text hash).

See ``docs/breakdown.md`` for a user guide.
"""

from __future__ import annotations

from commonlid.analysis.breakdown import (
    MACRO_MATCH_MODES,
    PREDICTIONS_FILENAME,
    MacroMatch,
    attach_texts,
    labels_match,
    language_breakdown,
    load_predictions,
    model_breakdown,
    sample_breakdown,
    select_top_models,
    to_macrolanguage,
)

__all__ = [
    "MACRO_MATCH_MODES",
    "PREDICTIONS_FILENAME",
    "MacroMatch",
    "attach_texts",
    "labels_match",
    "language_breakdown",
    "load_predictions",
    "model_breakdown",
    "sample_breakdown",
    "select_top_models",
    "to_macrolanguage",
]
