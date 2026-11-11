"""Configuration for the scoring server.

The server is configured from a TOML file. Everything that decides what the
evaluated agent can learn lives here: which datasets it may score against,
which metrics come back, and how many submissions it gets.

Example::

    max_submissions = 10          # across all datasets; omit or 0 for unlimited
    state_dir = "./eval-server-state"
    round_digits = 4
    per_language_metrics = true   # needed for the per_language.* metric below

    [[datasets]]
    id = "commonlid"
    metrics = ["macro.f1_gold_only", "micro.f1_gold_only"]
    max_submissions = 5           # optional per-dataset cap

    [[datasets]]
    id = "flores_dev"
    limit = 1000
    metrics = ["macro.f1_gold_only", "per_language.f1"]
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from commonlid.metrics.aggregate import macro_average, micro_average
from commonlid.metrics.core import LanguageMetrics

#: Scalar summary fields that may be exposed besides the macro/micro blocks.
_TOP_LEVEL_METRICS = frozenset({"n_samples", "n_samples_with_gold"})

DEFAULT_METRICS: tuple[str, ...] = ("macro.f1_gold_only",)

#: Prefix of the per-language metrics. They reveal which languages the gold
#: set contains (and, for the count fields, how many samples each has), so a
#: dataset may only list them when ``per_language_metrics`` is enabled.
PER_LANGUAGE_PREFIX = "per_language."

#: Environment variables that override the config file. Env wins so a
#: container can keep secrets and switches out of the mounted config.
ENV_API_TOKEN = "COMMONLID_SERVER_TOKEN"
ENV_PER_LANGUAGE_METRICS = "COMMONLID_SERVER_PER_LANGUAGE_METRICS"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def available_metrics() -> list[str]:
    """Every metric name a dataset's ``metrics`` list may contain.

    Names are dotted paths into ``summary.json``: ``macro.<key>``,
    ``micro.<key>``, ``per_language.<field>`` and a couple of top-level counts.
    ``per_language.*`` names also need ``per_language_metrics`` enabled.
    """
    macro = [f"macro.{k}" for k in macro_average({})]
    micro = [f"micro.{k}" for k in micro_average({})]
    per_lang = [f"per_language.{f.name}" for f in fields(LanguageMetrics)]
    return sorted([*_TOP_LEVEL_METRICS, *macro, *micro, *per_lang])


class ConfigError(ValueError):
    """Raised for an invalid server configuration."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """One dataset exposed by the server."""

    id: str
    metrics: tuple[str, ...] = DEFAULT_METRICS
    #: Cap on samples served and scored (``None`` = the whole dataset).
    limit: int | None = None
    #: Per-dataset submission cap, on top of the global one (``None`` = no extra cap).
    max_submissions: int | None = None
    #: Drop languages with fewer gold samples than this from the metrics.
    sample_count_threshold: int = 0


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Top-level scoring-server configuration."""

    datasets: tuple[DatasetConfig, ...]
    #: Total submissions allowed across all datasets (``None`` = unlimited).
    max_submissions: int | None = 10
    #: Where the submission counter, the shuffle seed and an archive of every
    #: submission are kept. ``None`` keeps state in memory only, so a restart
    #: resets the counter.
    state_dir: Path | None = None
    #: Decimal places of returned scores (``None`` = full precision).
    round_digits: int | None = 4
    #: Serve inputs in a seeded random order with opaque ids, so dataset
    #: order (often grouped by language) does not leak labels.
    shuffle: bool = True
    #: Fixed shuffle seed. ``None`` generates one on first start and keeps it
    #: in ``state_dir`` so ids stay stable across restarts.
    shuffle_seed: int | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    #: Allow ``per_language.*`` metrics in the datasets' ``metrics`` lists.
    per_language_metrics: bool = False
    #: Bearer token every request except ``/health`` must send
    #: (``Authorization: Bearer <token>``). ``None`` disables authentication.
    api_token: str | None = None

    def dataset(self, dataset_id: str) -> DatasetConfig | None:
        for ds in self.datasets:
            if ds.id == dataset_id:
                return ds
        return None


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{name} must be a non-negative integer, got {value!r}"
        raise ConfigError(msg)
    # 0 means "no cap", the same convention as the CLI's --limit.
    return value or None


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{name} must be true or false, got {value!r}"
        raise ConfigError(msg)
    return value


def _env_bool(environ: Mapping[str, str], name: str) -> bool | None:
    value = environ.get(name)
    if value is None:
        return None
    normalised = value.strip().lower()
    if normalised in _TRUE:
        return True
    if normalised in _FALSE:
        return False
    msg = f"environment variable {name} must be true or false, got {value!r}"
    raise ConfigError(msg)


def _parse_dataset(raw: Any, index: int, *, per_language_metrics: bool) -> DatasetConfig:
    if not isinstance(raw, dict):
        msg = f"datasets[{index}] must be a table"
        raise ConfigError(msg)
    known = {f.name for f in fields(DatasetConfig)}
    unknown = set(raw) - known
    if unknown:
        msg = f"datasets[{index}] has unknown keys: {sorted(unknown)}"
        raise ConfigError(msg)
    if "id" not in raw or not isinstance(raw["id"], str):
        msg = f"datasets[{index}] needs a string `id`"
        raise ConfigError(msg)

    metrics = raw.get("metrics", list(DEFAULT_METRICS))
    if not isinstance(metrics, list) or not metrics or not all(isinstance(m, str) for m in metrics):
        msg = f"datasets[{index}].metrics must be a non-empty list of strings"
        raise ConfigError(msg)
    allowed = set(available_metrics())
    bad = [m for m in metrics if m not in allowed]
    if bad:
        msg = (
            f"datasets[{index}] ({raw['id']}) has unknown metrics {bad}; "
            f"choose from: {', '.join(sorted(allowed))}"
        )
        raise ConfigError(msg)
    per_language = [m for m in metrics if m.startswith(PER_LANGUAGE_PREFIX)]
    if per_language and not per_language_metrics:
        msg = (
            f"datasets[{index}] ({raw['id']}) lists per-language metrics {per_language}, "
            "which reveal the languages of the gold set. Enable them with "
            f"`per_language_metrics = true` or {ENV_PER_LANGUAGE_METRICS}=1."
        )
        raise ConfigError(msg)

    threshold = raw.get("sample_count_threshold", 0)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
        msg = f"datasets[{index}].sample_count_threshold must be a non-negative integer"
        raise ConfigError(msg)

    return DatasetConfig(
        id=raw["id"],
        metrics=tuple(dict.fromkeys(metrics)),
        limit=_optional_positive_int(raw.get("limit"), f"datasets[{index}].limit"),
        max_submissions=_optional_positive_int(
            raw.get("max_submissions"), f"datasets[{index}].max_submissions"
        ),
        sample_count_threshold=threshold,
    )


def parse_config(
    raw: dict[str, Any],
    *,
    base_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ServerConfig:
    """Validate a decoded config mapping and build a :class:`ServerConfig`.

    Relative ``state_dir`` paths resolve against ``base_dir`` (the config
    file's directory) when given. ``environ`` (``os.environ`` by default)
    supplies the ``COMMONLID_SERVER_*`` overrides.
    """
    environ = os.environ if environ is None else environ
    known = {f.name for f in fields(ServerConfig)}
    unknown = set(raw) - known
    if unknown:
        msg = f"unknown top-level config keys: {sorted(unknown)}"
        raise ConfigError(msg)

    per_language_metrics = _bool(raw.get("per_language_metrics", False), "per_language_metrics")
    env_per_language = _env_bool(environ, ENV_PER_LANGUAGE_METRICS)
    if env_per_language is not None:
        per_language_metrics = env_per_language

    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        msg = "config needs at least one [[datasets]] entry"
        raise ConfigError(msg)
    datasets = tuple(
        _parse_dataset(d, i, per_language_metrics=per_language_metrics)
        for i, d in enumerate(raw_datasets)
    )
    ids = [d.id for d in datasets]
    if len(set(ids)) != len(ids):
        msg = f"duplicate dataset ids in config: {ids}"
        raise ConfigError(msg)

    state_dir: Path | None = None
    if raw.get("state_dir") is not None:
        state_dir = Path(raw["state_dir"]).expanduser()
        if base_dir is not None and not state_dir.is_absolute():
            state_dir = base_dir / state_dir

    round_digits = raw.get("round_digits", 4)
    if round_digits is not None and (
        not isinstance(round_digits, int) or isinstance(round_digits, bool) or round_digits < 0
    ):
        msg = "round_digits must be a non-negative integer"
        raise ConfigError(msg)

    seed = raw.get("shuffle_seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        msg = "shuffle_seed must be an integer"
        raise ConfigError(msg)

    host = raw.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        msg = f"host must be a non-empty string, got {host!r}"
        raise ConfigError(msg)
    port = raw.get("port", 8000)
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        msg = f"port must be an integer in 0..65535, got {port!r}"
        raise ConfigError(msg)

    api_token = environ.get(ENV_API_TOKEN) or raw.get("api_token")
    if api_token is not None and (not isinstance(api_token, str) or not api_token.strip()):
        msg = "api_token must be a non-empty string"
        raise ConfigError(msg)

    return ServerConfig(
        datasets=datasets,
        max_submissions=_optional_positive_int(raw.get("max_submissions", 10), "max_submissions"),
        state_dir=state_dir,
        round_digits=round_digits,
        shuffle=_bool(raw.get("shuffle", True), "shuffle"),
        shuffle_seed=seed,
        host=host,
        port=port,
        per_language_metrics=per_language_metrics,
        api_token=api_token,
    )


def load_config(path: str | Path) -> ServerConfig:
    """Read and validate a TOML config file."""
    if sys.version_info >= (3, 11):  # pragma: no cover - interpreter dependent
        import tomllib
    else:  # pragma: no cover - interpreter dependent
        import tomli as tomllib

    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return parse_config(raw, base_dir=path.parent)
