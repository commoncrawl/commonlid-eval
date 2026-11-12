"""Scoring server: score an agent's predictions without exposing the labels.

The server runs in an environment that holds the gold labels; the agent only
sees the label-free inputs and the scores the config allows. The HTTP app
needs the ``[server]`` extra (FastAPI + uvicorn); :mod:`.config` and
:mod:`.service` import without it.
"""

from __future__ import annotations

from commonlid.server.config import (
    ConfigError,
    DatasetConfig,
    ServerConfig,
    available_metrics,
    load_config,
    parse_config,
)
from commonlid.server.service import ScoringService

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "ScoringService",
    "ServerConfig",
    "available_metrics",
    "load_config",
    "parse_config",
]
