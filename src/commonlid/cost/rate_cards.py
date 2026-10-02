"""Rate cards: what a provider charges per unit of each usage meter.

Prices drift, so every card carries the date it was checked and the page it
came from. LLM token prices are not listed here; they are read from LiteLLM's
model map (:func:`litellm_rate_card`), which is maintained upstream.

Rates are linear USD per unit. Free tiers, volume tiers and committed-use
discounts are deliberately not modelled: the estimate is an upper-end budget
at list price.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from commonlid.cost.usage import (
    CHARACTERS,
    INPUT_TOKENS,
    INSTANCE_HOURS,
    OUTPUT_TOKENS,
    REASONING_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RateCard:
    """USD per unit for each meter a provider bills."""

    card_id: str
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


_GOOGLE_TRANSLATE_SOURCE = "https://cloud.google.com/translate/pricing"
_HF_ENDPOINTS_SOURCE = "https://huggingface.co/docs/inference-endpoints/pricing"
_AWS_EC2_SOURCE = "https://aws.amazon.com/ec2/pricing/on-demand/"


def _hourly(card_id: str, usd_per_hour: float, *, source: str, note: str) -> RateCard:
    return RateCard(
        card_id=card_id,
        rates={INSTANCE_HOURS: usd_per_hour},
        as_of="2026-10-02",
        source=source,
        notes=(note,),
    )


# Language detection costs the same on both editions, and v3's one request per
# text is not billed separately: only characters are.
_API_CARDS: dict[str, RateCard] = {
    "google-translate-v2": RateCard(
        card_id="google-translate-v2",
        rates={CHARACTERS: 20.0 / 1_000_000},
        as_of="2026-10-02",
        source=_GOOGLE_TRANSLATE_SOURCE,
        notes=("Basic edition language detection",),
    ),
    "google-translate-v3": RateCard(
        card_id="google-translate-v3",
        rates={CHARACTERS: 20.0 / 1_000_000},
        as_of="2026-10-02",
        source=_GOOGLE_TRANSLATE_SOURCE,
        notes=("Advanced edition language detection",),
    ),
}

# Hardware a self-hosted model can run on, billed per instance hour whether
# busy or idle. EC2 prices are on-demand in us-east-1.
HARDWARE_CARDS: dict[str, RateCard] = {
    card.card_id: card
    for card in (
        _hourly("aws:g5.xlarge", 1.006, source=_AWS_EC2_SOURCE, note="1x NVIDIA A10G, us-east-1"),
        _hourly("aws:g6.xlarge", 0.8048, source=_AWS_EC2_SOURCE, note="1x NVIDIA L4, us-east-1"),
        _hourly("hf:intel-spr-x1", 0.033, source=_HF_ENDPOINTS_SOURCE, note="1 vCPU, AWS"),
        _hourly("hf:nvidia-t4-x1", 0.50, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA T4, AWS"),
        _hourly("hf:nvidia-l4-x1", 0.80, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA L4, AWS"),
        _hourly("hf:nvidia-a10g-x1", 1.00, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA A10G, AWS"),
        _hourly("hf:nvidia-a100-x1", 2.50, source=_HF_ENDPOINTS_SOURCE, note="1x NVIDIA A100, AWS"),
    )
}

RATE_CARDS: dict[str, RateCard] = {**_API_CARDS, **HARDWARE_CARDS}


def get_rate_card(card_id: str) -> RateCard:
    """Look up a built-in rate card by id."""
    try:
        return RATE_CARDS[card_id]
    except KeyError:
        known = ", ".join(sorted(RATE_CARDS))
        msg = f"unknown rate card {card_id!r}; known cards: {known}"
        raise KeyError(msg) from None


def hourly_rate_card(usd_per_hour: float) -> RateCard:
    """An ad-hoc hardware card for hardware not in :data:`HARDWARE_CARDS`."""
    return RateCard(
        card_id=f"custom:{usd_per_hour:g}/h",
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
        card_id=f"litellm:{model_name}",
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
