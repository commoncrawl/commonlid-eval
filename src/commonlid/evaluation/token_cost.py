"""Offline token & cost estimation for LLM evaluation runs.

Reconstructs the exact DSPy request the evaluator would send for each dataset
sample (system instruction + per-sample user message) and counts tokens with
LiteLLM's ``token_counter`` -- which uses ``tiktoken`` for OpenAI/Azure models
and an HF ``AutoTokenizer`` for Hugging Face models, i.e. "the corresponding
tokenizer" for whatever model is passed. Pricing comes from LiteLLM's model
map (``cost_per_token``). No API calls are made.

The heavy dependencies (``dspy``, ``litellm``) are imported lazily so the core
package stays importable without the ``llm`` extra.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from commonlid.models.dspy_llm import DEFAULT_INSTRUCTION, _build_signature

if TYPE_CHECKING:
    from commonlid.core.lid_dataset import LIDDataset

__all__ = ["DEFAULT_INSTRUCTION", "TokenCostEstimate", "estimate"]

logger = logging.getLogger(__name__)

DSPY_SPEC_PREFIX = "dspy:"

# Fallback when no tokenizer can be resolved for the model: rough chars/token.
_HEURISTIC_CHARS_PER_TOKEN = 4


@dataclass
class TokenCostEstimate:
    """Per-``(model, dataset)`` token and cost estimate.

    ``*_cost_usd`` fields are ``None`` when LiteLLM has no price for the model.
    When ``extrapolated`` is ``True`` the ``full_*`` fields carry a projection
    of the measured subset onto the full dataset (``n_total`` samples).
    """

    model: str
    dataset_id: str
    n_samples: int
    n_total: int
    per_request_overhead_tokens: int
    total_input_tokens: int
    mean_input_tokens: float
    output_tokens_per_sample: int
    reasoning_tokens_per_sample: int
    total_output_tokens: int
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None
    tokenizer_note: str | None
    price_available: bool
    extrapolated: bool
    full_total_input_tokens: int | None
    full_total_output_tokens: int | None
    full_total_cost_usd: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_spec_prefix(model_name: str) -> str:
    """Accept the same ``dspy:<name>`` spec the ``run`` command uses."""
    return model_name.removeprefix(DSPY_SPEC_PREFIX) if model_name.startswith(DSPY_SPEC_PREFIX) else model_name


def _render_messages(instruction: str, text: str) -> list[dict[str, Any]]:
    """Render the exact chat messages DSPy would send for one sample."""
    import dspy

    signature = _build_signature(instruction)
    messages = dspy.ChatAdapter().format(signature=signature, demos=[], inputs={"text": text})
    return cast("list[dict[str, Any]]", messages)


def _count_tokens(model: str, *, text: str | None = None, messages: list[dict[str, Any]] | None = None) -> int | None:
    """Count tokens via LiteLLM. Returns ``None`` if no tokenizer resolves."""
    import litellm

    try:
        return int(litellm.token_counter(model=model, text=text, messages=messages))
    except Exception as exc:  # LiteLLM raises assorted errors on unknown tokenizers
        logger.debug("token_counter failed for %s (%s): %s", model, type(exc).__name__, exc)
        return None


def _heuristic_tokens(text: str) -> int:
    return max(1, len(text) // _HEURISTIC_CHARS_PER_TOKEN) if text else 0


def estimate(
    model_name: str,
    dataset: LIDDataset,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    output_tokens: int = 10,
    reasoning_tokens: int = 0,
    limit: int = 0,
) -> TokenCostEstimate:
    """Estimate token usage and USD cost of running ``model_name`` over ``dataset``.

    ``output_tokens`` and ``reasoning_tokens`` are per-sample assumptions;
    reasoning tokens are billed as completion tokens and folded into the output
    total. ``limit`` (>0) measures only the first N samples and, when N is
    smaller than the full dataset, also projects the mean onto all samples.
    """
    model = _strip_spec_prefix(model_name)
    tokenizer_note: str | None = None

    overhead = _count_tokens(model, messages=_render_messages(instruction, ""))
    if overhead is None:
        tokenizer_note = (
            f"No tokenizer resolved for {model!r}; counts are approximate "
            f"(~{_HEURISTIC_CHARS_PER_TOKEN} chars/token heuristic). "
            "Install `transformers` for accurate Hugging Face tokenization."
        )
        overhead = len(_render_messages(instruction, "")[0]["content"]) // _HEURISTIC_CHARS_PER_TOKEN

    total_input_tokens = 0
    n_samples = 0
    for texts, _golds in dataset.iter_batches(limit=limit if limit > 0 else None):
        for text in texts:
            text_tokens = _count_tokens(model, text=text)
            if text_tokens is None:
                text_tokens = _heuristic_tokens(text)
            total_input_tokens += overhead + text_tokens
            n_samples += 1

    n_total = len(dataset)
    mean_input_tokens = total_input_tokens / n_samples if n_samples else 0.0
    total_output_tokens = n_samples * (output_tokens + reasoning_tokens)

    input_cost, output_cost, price_available = _price(
        model, prompt_tokens=total_input_tokens, completion_tokens=total_output_tokens
    )
    total_cost = input_cost + output_cost if price_available else None

    extrapolated = 0 < n_samples < n_total
    full_input = full_output = None
    full_cost: float | None = None
    if extrapolated:
        full_input = round(mean_input_tokens * n_total)
        full_output = n_total * (output_tokens + reasoning_tokens)
        f_in, f_out, f_ok = _price(model, prompt_tokens=full_input, completion_tokens=full_output)
        full_cost = f_in + f_out if f_ok else None

    return TokenCostEstimate(
        model=model,
        dataset_id=dataset.dataset_id,
        n_samples=n_samples,
        n_total=n_total,
        per_request_overhead_tokens=overhead,
        total_input_tokens=total_input_tokens,
        mean_input_tokens=mean_input_tokens,
        output_tokens_per_sample=output_tokens,
        reasoning_tokens_per_sample=reasoning_tokens,
        total_output_tokens=total_output_tokens,
        input_cost_usd=input_cost if price_available else None,
        output_cost_usd=output_cost if price_available else None,
        total_cost_usd=total_cost,
        tokenizer_note=tokenizer_note,
        price_available=price_available,
        extrapolated=extrapolated,
        full_total_input_tokens=full_input,
        full_total_output_tokens=full_output,
        full_total_cost_usd=full_cost,
    )


def _price(model: str, *, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, bool]:
    """Return ``(input_cost, output_cost, price_available)`` for the token counts."""
    import litellm

    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
    except Exception as exc:  # LiteLLM raises when the model isn't in its price map
        logger.debug("cost_per_token failed for %s (%s): %s", model, type(exc).__name__, exc)
        return 0.0, 0.0, False
    input_cost, output_cost = float(input_cost), float(output_cost)
    # LiteLLM returns (0.0, 0.0) for models absent from its price map rather than
    # raising; treat an all-zero cost on non-zero tokens as "no price available".
    has_tokens = prompt_tokens > 0 or completion_tokens > 0
    if has_tokens and input_cost <= 0.0 and output_cost <= 0.0:
        return 0.0, 0.0, False
    return input_cost, output_cost, True
