"""Budget a (model, dataset) run before starting it.

The estimate combines three kinds of per-meter usage:

* **counted**: exact, from :meth:`LIDModel.estimate_usage` over every sample
  (characters sent, prompt tokens). Makes no API calls.
* **assumed**: per-sample ranges for what cannot be counted offline
  (generated and reasoning tokens, hardware throughput), from the model's
  defaults or the caller.
* **calibrated**: assumptions replaced by a real, paid run on a random sample,
  as a mean with a 95% confidence interval.

Usage is then priced with a :class:`RateCard`: the model's own (API pricing)
or a hardware card (instance hours, for self-hosted models).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from commonlid.cost.usage import INSTANCE_HOURS, Range, Usage, add_usage

if TYPE_CHECKING:
    from commonlid.core.lid_dataset import LIDDataset
    from commonlid.core.lid_model import LIDModel
    from commonlid.cost.rate_cards import RateCard

logger = logging.getLogger(__name__)

UsageSource = Literal["counted", "assumed", "calibrated"]

_SECONDS_PER_HOUR = 3600


@dataclass(frozen=True, slots=True)
class MeterEstimate:
    """Usage and cost of one meter over the whole dataset."""

    meter: str
    source: UsageSource
    per_sample: Range
    total: Range
    rate: float | None
    """USD per unit, or ``None`` when the rate card does not price this meter."""

    @property
    def cost(self) -> Range | None:
        return None if self.rate is None else self.total.scale(self.rate)


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The budget for running one model over one dataset."""

    model_id: str
    dataset_id: str
    n_samples: int
    rate_card: RateCard | None
    meters: list[MeterEstimate]
    notes: list[str] = field(default_factory=list)

    @property
    def total_cost(self) -> Range | None:
        """Summed cost, or ``None`` when nothing could be priced."""
        costs = [m.cost for m in self.meters if m.cost is not None]
        if not costs:
            return None
        total = costs[0]
        for cost in costs[1:]:
            total += cost
        return total

    def to_dict(self) -> dict[str, Any]:
        total = self.total_cost
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "n_samples": self.n_samples,
            "rate_card": None if self.rate_card is None else _card_dict(self.rate_card),
            "meters": [
                {
                    "meter": m.meter,
                    "source": m.source,
                    "per_sample": asdict(m.per_sample),
                    "total": asdict(m.total),
                    "usd_per_unit": m.rate,
                    "cost_usd": None if m.cost is None else asdict(m.cost),
                }
                for m in self.meters
            ],
            "total_cost_usd": None if total is None else asdict(total),
            "notes": self.notes,
        }


def _card_dict(card: RateCard) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "rates": dict(card.rates),
        "as_of": card.as_of,
        "source": card.source,
        "notes": list(card.notes),
    }


def estimate_cost(
    model: LIDModel,
    dataset: LIDDataset,
    *,
    assumptions: Mapping[str, Range] | None = None,
    rate_overrides: Mapping[str, float] | None = None,
    hardware: RateCard | None = None,
    throughput: Range | None = None,
    calibrate: int = 0,
    seed: int = 0,
    batch_size: int = 64,
) -> CostEstimate:
    """Estimate what running ``model`` over all of ``dataset`` would cost.

    Parameters
    ----------
    assumptions:
        Per-sample ranges overriding or adding to
        :meth:`LIDModel.usage_assumptions`.
    rate_overrides:
        USD per unit, replacing or adding to the rate card's rates.
    hardware:
        Price compute time on this card instead of the model's API pricing.
        Needs ``throughput`` or ``calibrate``.
    throughput:
        Samples per second on ``hardware``.
    calibrate:
        Predict this many random samples for real (this costs money on
        metered models) and replace assumptions with what was measured.
    """
    notes: list[str] = []
    n_samples = len(dataset)
    calibration_texts = _sample_texts(dataset, calibrate, seed) if calibrate > 0 else []

    if hardware is not None:
        per_sample = _compute_hours_per_sample(
            model, throughput, calibration_texts, batch_size, notes
        )
        usage: list[tuple[str, UsageSource, Range]] = [
            (INSTANCE_HOURS, "calibrated" if calibration_texts else "assumed", per_sample)
        ]
        card: RateCard | None = hardware
    else:
        usage = _api_usage(
            model, dataset, n_samples, assumptions or {}, calibration_texts, batch_size, notes
        )
        card = model.rate_card()
        if not usage:
            notes.append(
                f"{model.model_id} is not billed per call. To price compute time, pass "
                "--hardware or --hourly-rate, with --throughput or --calibrate."
            )

    if card is not None:
        card = card.with_rates(rate_overrides or {})
    elif rate_overrides:
        from commonlid.cost.rate_cards import RateCard

        card = RateCard(card_id="custom", rates=dict(rate_overrides), notes=("user-supplied",))
    elif usage:
        notes.append(f"No price known for {model.model_id}; pass --rate METER=USD_PER_UNIT.")

    meters = []
    for meter, source, per_sample_range in usage:
        rate = None if card is None else card.rates.get(meter)
        if card is not None and rate is None:
            notes.append(f"Rate card {card.card_id} does not price {meter!r}; left out of total.")
        meters.append(
            MeterEstimate(
                meter=meter,
                source=source,
                per_sample=per_sample_range,
                total=per_sample_range.scale(n_samples),
                rate=rate,
            )
        )
    return CostEstimate(
        model_id=model.model_id,
        dataset_id=dataset.dataset_id,
        n_samples=n_samples,
        rate_card=card,
        meters=meters,
        notes=notes,
    )


def _api_usage(
    model: LIDModel,
    dataset: LIDDataset,
    n_samples: int,
    overrides: Mapping[str, Range],
    calibration_texts: list[str],
    batch_size: int,
    notes: list[str],
) -> list[tuple[str, UsageSource, Range]]:
    """Per-sample usage of each meter for a model billed per call."""
    counted: Usage = {}
    for texts, _golds in dataset.iter_batches(batch_size=batch_size):
        batch_usage = model.estimate_usage(texts)
        if batch_usage is None:
            break
        add_usage(counted, batch_usage)

    usage: list[tuple[str, UsageSource, Range]] = [
        (meter, "counted", Range.exact(total / n_samples if n_samples else 0.0))
        for meter, total in counted.items()
    ]
    assumed: dict[str, tuple[UsageSource, Range]] = {
        meter: ("assumed", r) for meter, r in {**model.usage_assumptions(), **overrides}.items()
    }

    if calibration_texts:
        observations = model.measure_usage(calibration_texts)
        if observations:
            for meter in assumed:
                values = [obs.get(meter, 0.0) for obs in observations]
                assumed[meter] = ("calibrated", Range.from_observations(values))
            notes.append(
                f"Calibrated on {len(calibration_texts)} random samples "
                f"({len(observations)} requests)."
            )
        else:
            notes.append(f"{model.model_id} reports no usage; calibration changed nothing.")

    for meter, (source, per_sample) in assumed.items():
        if meter in counted:
            # Counting is exact, so it beats an assumption for the same meter.
            continue
        usage.append((meter, source, per_sample))
    return usage


def _compute_hours_per_sample(
    model: LIDModel,
    throughput: Range | None,
    calibration_texts: list[str],
    batch_size: int,
    notes: list[str],
) -> Range:
    """Instance hours per sample, from a throughput assumption or a timed run."""
    if calibration_texts:
        seconds = _time_per_sample(model, calibration_texts, batch_size)
        notes.append(
            f"Throughput measured on {len(calibration_texts)} random samples on this "
            "machine; it only transfers to --hardware if that is what this ran on."
        )
        return seconds.scale(1 / _SECONDS_PER_HOUR)
    if throughput is None:
        msg = "pricing hardware needs --throughput (samples/second) or --calibrate N"
        raise ValueError(msg)
    return throughput.reciprocal().scale(1 / _SECONDS_PER_HOUR)


def _time_per_sample(model: LIDModel, texts: list[str], batch_size: int) -> Range:
    """Seconds per sample, timing each batch as one observation."""
    model.load()  # loading weights is a one-off, not per-sample cost
    # The first batch pays for lazy initialisation (JIT, caches, CUDA
    # kernels) and runs several times slower, so it is run once untimed.
    model.predict_scored(texts[:batch_size])
    observations = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        began = time.perf_counter()
        model.predict_scored(batch)
        observations.append((time.perf_counter() - began) / len(batch))
    return Range.from_observations(observations)


def _sample_texts(dataset: LIDDataset, k: int, seed: int) -> list[str]:
    """A seeded uniform sample of ``k`` texts (reservoir sampling in one pass)."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for texts, _golds in dataset.iter_batches(batch_size=1024):
        for text in texts:
            if seen < k:
                reservoir.append(text)
            else:
                slot = rng.randint(0, seen)
                if slot < k:
                    reservoir[slot] = text
            seen += 1
    return reservoir


def format_estimate(estimate: CostEstimate) -> str:
    """Render an estimate as a plain-text table."""
    lines = [
        f"{estimate.model_id} on {estimate.dataset_id} ({estimate.n_samples:,} samples)",
    ]
    card = estimate.rate_card
    if card is not None:
        provenance = ", ".join(p for p in (card.as_of and f"as of {card.as_of}", card.source) if p)
        lines.append(f"Rate card: {card.card_id}" + (f" ({provenance})" if provenance else ""))

    if estimate.meters:
        header = ("meter", "source", "per sample", "total", "rate", "cost (expected, low-high)")
        rows = [
            (
                m.meter,
                m.source,
                _fmt_range(m.per_sample, _fmt_quantity),
                _fmt_range(m.total, _fmt_quantity),
                "n/a" if m.rate is None else _fmt_rate(m.rate),
                "n/a" if m.cost is None else _fmt_range(m.cost, _fmt_usd),
            )
            for m in estimate.meters
        ]
        widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]
        lines.append("")
        for row in (header, *rows):
            cells = (cell.ljust(w) for cell, w in zip(row, widths, strict=True))
            lines.append("  ".join(cells).rstrip())

    total = estimate.total_cost
    lines.append("")
    lines.append("Total: " + ("n/a" if total is None else _fmt_range(total, _fmt_usd)))
    lines.extend(f"Note: {note}" for note in estimate.notes)
    return "\n".join(lines)


def _fmt_range(r: Range, fmt: Any) -> str:
    if r.is_exact:
        return str(fmt(r.expected))
    return f"{fmt(r.expected)} ({fmt(r.low)}-{fmt(r.high)})"


def _fmt_quantity(value: float) -> str:
    if value == 0 or abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _fmt_rate(usd_per_unit: float) -> str:
    # Token and character prices read naturally per million units.
    if 0 < usd_per_unit < 0.01:
        return f"${usd_per_unit * 1_000_000:.4g}/M"
    return f"${usd_per_unit:.4g}"


def _fmt_usd(value: float) -> str:
    if 0 < value < 0.01:
        return f"${value:.2g}"
    return f"${value:,.2f}"
