"""Usage quantities and low/expected/high ranges for cost estimation.

Usage is what a provider meters (tokens, characters, instance hours), kept
apart from what it charges per unit (:mod:`commonlid.cost.rate_cards`). This
module has no heavy dependencies so :mod:`commonlid.core.lid_model` can import
it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Meter name -> quantity, e.g. ``{"input_tokens": 298.0, "characters": 205.0}``.
Usage = dict[str, float]

# Canonical meter names. Models and rate cards may use others; these are the
# ones the built-in models and cards agree on.
INPUT_TOKENS = "input_tokens"
OUTPUT_TOKENS = "output_tokens"
REASONING_TOKENS = "reasoning_tokens"
CHARACTERS = "characters"
INSTANCE_HOURS = "instance_hours"

# z-score for the 95% confidence interval used on calibrated means.
_Z_95 = 1.96


@dataclass(frozen=True, slots=True)
class Range:
    """A quantity known only within bounds: ``low <= expected <= high``."""

    low: float
    expected: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.expected <= self.high:
            msg = (
                f"Range needs low <= expected <= high, got {self.low}, {self.expected}, {self.high}"
            )
            raise ValueError(msg)

    @classmethod
    def exact(cls, value: float) -> Range:
        return cls(value, value, value)

    @classmethod
    def parse(cls, spec: str) -> Range:
        """Parse ``"E"`` (exact) or ``"LOW:EXPECTED:HIGH"``."""
        parts = spec.split(":")
        try:
            values = [float(p) for p in parts]
        except ValueError as exc:
            msg = f"expected a number or LOW:EXPECTED:HIGH, got {spec!r}"
            raise ValueError(msg) from exc
        if len(values) == 1:
            return cls.exact(values[0])
        if len(values) == 3:
            return cls(*values)
        msg = f"expected a number or LOW:EXPECTED:HIGH, got {spec!r}"
        raise ValueError(msg)

    @classmethod
    def from_observations(cls, values: Iterable[float]) -> Range:
        """Mean of per-sample observations with a 95% confidence interval.

        The interval covers the *mean* (what a full-dataset total scales
        from), not individual samples, and is clipped at zero since usage is
        never negative.
        """
        data = list(values)
        if not data:
            msg = "need at least one observation"
            raise ValueError(msg)
        mean = math.fsum(data) / len(data)
        if len(data) < 2:
            return cls.exact(mean)
        variance = math.fsum((v - mean) ** 2 for v in data) / (len(data) - 1)
        half_width = _Z_95 * math.sqrt(variance / len(data))
        return cls(max(0.0, mean - half_width), mean, mean + half_width)

    @property
    def is_exact(self) -> bool:
        return self.low == self.high

    def scale(self, factor: float) -> Range:
        """Multiply by a non-negative factor."""
        return Range(self.low * factor, self.expected * factor, self.high * factor)

    def reciprocal(self) -> Range:
        """``1 / x``; the bounds swap, so a high throughput yields a low duration."""
        if self.low <= 0:
            msg = f"cannot take the reciprocal of a range reaching {self.low}"
            raise ValueError(msg)
        return Range(1 / self.high, 1 / self.expected, 1 / self.low)

    def __add__(self, other: Range) -> Range:
        return Range(self.low + other.low, self.expected + other.expected, self.high + other.high)


def add_usage(total: Usage, part: Mapping[str, float]) -> None:
    """Accumulate ``part`` into ``total`` in place."""
    for meter, quantity in part.items():
        total[meter] = total.get(meter, 0.0) + quantity
