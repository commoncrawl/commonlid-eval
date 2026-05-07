"""HF Space entrypoint for the CommonLID Leaderboard.

This file is intentionally tiny: all rendering logic lives inside the
``commonlid.leaderboard.app`` module so the Space stays in lockstep with
whatever's pinned in ``requirements.txt``. The Space environment sets
``GRADIO_SERVER_NAME=0.0.0.0`` automatically for public hosting.
"""

from __future__ import annotations

import os

from commonlid.leaderboard.app import build_app

REPO_ID = os.environ.get("COMMONLID_RESULTS_REPO", "commoncrawl/commonlid-results")
REVISION = os.environ.get("COMMONLID_RESULTS_REVISION") or None

demo = build_app(repo_id=REPO_ID, revision=REVISION)

if __name__ == "__main__":
    demo.launch()
