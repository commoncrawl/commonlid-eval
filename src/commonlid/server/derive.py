"""Derive a served dataset from a registered one (``source``, ``exclude``, ...).

The steps run in a fixed order: exclude, length filter, stratified sample,
limit. Each takes and returns positions in the source dataset, so the result
is a list of source positions; it depends only on the source texts, the
excluded texts and the config (the sampler is seeded).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

from commonlid.server.config import DatasetConfig


class DerivationError(ValueError):
    """Raised when a derived dataset cannot be built from its source."""


def normalize_text(text: str) -> str:
    """Key for comparing texts: NFC, whitespace runs collapsed, ends stripped."""
    return " ".join(unicodedata.normalize("NFC", text).split())


@dataclass(slots=True)
class Derivation:
    """Source positions of a derived dataset and how each step changed the count."""

    positions: list[int]
    steps: list[tuple[str, int]] = field(default_factory=list)

    def describe(self) -> str:
        """``"373230 -> exclude: 371708 -> min_chars=20: 351310 -> ..."``."""
        return " -> ".join(f"{name}: {n}" if name else str(n) for name, n in self.steps)


def derive(
    config: DatasetConfig,
    texts: Sequence[str],
    labels: Sequence[str | None],
    excluded: Callable[[], Collection[str]] | Collection[str] = (),
) -> Derivation:
    """Select the source positions ``config`` keeps.

    ``excluded`` holds the normalised texts to drop (see :func:`normalize_text`),
    or returns them; it is only called when ``config.exclude`` is set.
    """
    positions = list(range(len(texts)))
    result = Derivation(positions, [("", len(positions))])

    if config.exclude:
        keys = excluded() if callable(excluded) else excluded
        positions = [i for i in positions if normalize_text(texts[i]) not in keys]
        result.steps.append((f"exclude {','.join(config.exclude)}", len(positions)))

    if config.min_chars is not None or config.max_chars is not None:
        low = config.min_chars or 0
        high = config.max_chars
        lengths = {i: len(normalize_text(texts[i])) for i in positions}
        positions = [
            i for i in positions if lengths[i] >= low and (high is None or lengths[i] <= high)
        ]
        bounds = [
            f"{key}={value}"
            for key, value in (("min_chars", config.min_chars), ("max_chars", config.max_chars))
            if value is not None
        ]
        result.steps.append((" ".join(bounds), len(positions)))

    if config.sample is not None:
        positions = _stratified(config, positions, labels)
        s = config.sample
        result.steps.append((
            f"sample max_size={s.max_size} min_per_language={s.min_per_language} seed={s.seed}",
            len(positions),
        ))

    if config.limit is not None and config.limit < len(positions):
        positions = positions[: config.limit]
        result.steps.append((f"limit={config.limit}", len(positions)))

    result.positions = positions
    return result


def _stratified(
    config: DatasetConfig, positions: list[int], labels: Sequence[str | None]
) -> list[int]:
    from collections import Counter

    from commonlid.datasets_tools import stratified_sample_with_minimum_per_class

    sample = config.sample
    assert sample is not None
    pool_labels = [labels[i] for i in positions]
    counts = Counter(pool_labels)
    beyond_minimum = sum(
        n - sample.min_per_language for n in counts.values() if n >= sample.min_per_language
    )
    if sample.max_size > beyond_minimum:
        msg = (
            f"{config.id}: sample.max_size = {sample.max_size}, but only {beyond_minimum} of "
            f"the {len(positions)} remaining texts are left once every language has its "
            f"{sample.min_per_language}. Lower max_size, or drop `sample` to serve them all."
        )
        raise DerivationError(msg)
    picked = stratified_sample_with_minimum_per_class(
        pool_labels,
        max_size=sample.max_size,
        min_size=sample.min_per_language,
        seed=sample.seed,
    )
    return [positions[j] for j in picked.selected]
