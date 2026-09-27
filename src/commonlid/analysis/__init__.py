"""Cross-model analysis of published predictions.

Public API (all pure pandas, so the leaderboard can reuse them later):

- :func:`select_top_models` — rank the models on one dataset by a
  leaderboard metric and keep the best ``k``.
- :func:`load_predictions` — pull ``<dataset>/<model>/predictions.jsonl``
  from the results HF dataset into one long DataFrame.
- :func:`sample_difficulty` — one row per sample: how many of the selected
  models got it wrong, what they predicted instead, and whether the models
  agree on a different label (a candidate annotation error).
- :func:`language_difficulty` — the same, aggregated per gold language.
- :func:`attach_texts` — join the sample text back in from the registered
  dataset (predictions only store a text hash).
"""

from __future__ import annotations

from commonlid.analysis.difficulty import (
    PREDICTIONS_FILENAME,
    attach_texts,
    language_difficulty,
    load_predictions,
    sample_difficulty,
    select_top_models,
)

__all__ = [
    "PREDICTIONS_FILENAME",
    "attach_texts",
    "language_difficulty",
    "load_predictions",
    "sample_difficulty",
    "select_top_models",
]
