"""Leaderboard scaffolding: load published results, build a Gradio app.

Public API:
- :func:`load_results` — pull every ``<dataset>/<model>/summary.json`` from the
  published HF dataset and return a single tidy ``pandas.DataFrame``.
- :func:`build_app` — build the Gradio Blocks UI; not imported here so callers
  without the optional ``[leaderboard]`` extra installed still get a clean
  ``ImportError`` only when they try to launch the app.
"""

from __future__ import annotations

from commonlid.leaderboard.data import (
    DEFAULT_REPO_ID,
    SUMMARY_FILENAME,
    LeaderboardRow,
    load_results,
)

__all__ = [
    "DEFAULT_REPO_ID",
    "SUMMARY_FILENAME",
    "LeaderboardRow",
    "load_results",
]
