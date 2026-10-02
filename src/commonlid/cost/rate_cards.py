"""Rate cards: what a provider charges per unit of each usage meter.

Prices drift, so every card carries the date it was checked and the page it
came from. LLM token prices are read from LiteLLM's model map (:func:`litellm_rate_card`), which is
maintained upstream. A paid API's own price list lives on its model class as
:attr:`LIDModel.rate_card`; this module holds what is shared: the hardware a
self-hosted model can run on.

Rates are linear USD per unit. Free tiers, volume tiers and committed-use
discounts are deliberately not modelled: the estimate is an upper-end budget
at list price.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from commonlid.cost.usage import (
    INPUT_TOKENS,
    INSTANCE_HOURS,
    OUTPUT_TOKENS,
    REASONING_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RateCard:
    """USD per unit for each meter a provider bills."""

    rates: Mapping[str, float]
    as_of: str | None = None
    source: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def with_rates(self, overrides: Mapping[str, float]) -> RateCard:
        """Return a copy with ``overrides`` replacing or adding meter rates."""
        if not overrides:
            return self
        return replace(
            self,
            rates={**self.rates, **overrides},
            notes=(*self.notes, f"rates overridden: {', '.join(sorted(overrides))}"),
        )


_HF_ENDPOINTS_SOURCE = "https://huggingface.co/docs/inference-endpoints/pricing"
_AWS_EC2_SOURCE = "https://aws.amazon.com/ec2/pricing/on-demand/"


def _hourly(usd_per_hour: float, *, source: str, note: str) -> RateCard:
    return RateCard(
        rates={INSTANCE_HOURS: usd_per_hour},
        as_of="2026-10-02",
        source=source,
        notes=(note,),
    )


# Hardware a self-hosted model can run on, billed per instance hour whether
# busy or idle. EC2 prices are on-demand in us-east-1.
HARDWARE_CARDS: dict[str, RateCard] = {
    "aws:g5.xlarge": _hourly(1.006, source=_AWS_EC2_SOURCE, note="1x NVIDIA A10G, us-east-1"),
    "aws:g6.xlarge": _hourly(0.8048, source=_AWS_EC2_SOURCE, note="1x NVIDIA L4, us-east-1"),
    "hf:intel-spr-x1": _hourly(0.033, source=_HF_ENDPOINTS_SOURCE, note="1 vCPU, AWS"),
    "hf:nvidia-t4-x1": _hourly(0.50, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA T4, AWS"),
    "hf:nvidia-l4-x1": _hourly(0.80, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA L4, AWS"),
    "hf:nvidia-a10g-x1": _hourly(1.00, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA A10G, AWS"),
    "hf:nvidia-a100-x1": _hourly(2.50, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA A100, AWS"),
}


def get_hardware_card(name: str) -> RateCard:
    """Look up built-in hardware by name, e.g. ``"aws:g5.xlarge"``."""
    try:
        return HARDWARE_CARDS[name]
    except KeyError:
        known = ", ".join(sorted(HARDWARE_CARDS))
        msg = f"unknown hardware {name!r}; known hardware: {known}"
        raise KeyError(msg) from None


def hourly_rate_card(usd_per_hour: float) -> RateCard:
    """An ad-hoc hardware card for hardware not in :data:`HARDWARE_CARDS`."""
    return RateCard(
        rates={INSTANCE_HOURS: usd_per_hour},
        notes=("user-supplied hourly rate",),
    )


def litellm_rate_card(model_name: str) -> RateCard | None:
    """Build a token rate card from LiteLLM's model map, or ``None`` if it has no price.

    Reasoning tokens are billed as output tokens unless LiteLLM lists a
    separate reasoning rate.
    """
    try:
        import litellm
    except ImportError:
        logger.debug("litellm is not installed; no token prices for %s", model_name)
        return None

    try:
        info = litellm.get_model_info(model_name)
    except Exception as exc:  # LiteLLM raises assorted errors for unknown models
        logger.debug("get_model_info failed for %s (%s): %s", model_name, type(exc).__name__, exc)
        return None

    input_rate = info.get("input_cost_per_token")
    output_rate = info.get("output_cost_per_token")
    # An absent or zero price means "unknown", not "free".
    if not input_rate and not output_rate:
        return None
    output_rate = float(output_rate or 0.0)
    reasoning_rate = info.get("output_cost_per_reasoning_token")
    return RateCard(
        rates={
            INPUT_TOKENS: float(input_rate or 0.0),
            OUTPUT_TOKENS: output_rate,
            REASONING_TOKENS: float(reasoning_rate) if reasoning_rate else output_rate,
        },
        source=f"LiteLLM model map (litellm {_litellm_version()})",
    )


def _litellm_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("litellm")
    except PackageNotFoundError:  # pragma: no cover - imported above
        return "unknown"
