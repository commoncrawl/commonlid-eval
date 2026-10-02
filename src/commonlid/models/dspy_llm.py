"""DSPy-based LLM wrapper for language identification.

Encapsulates the DSPy signature, optional threaded batched prediction, and
Azure AD-token authentication into a :class:`LIDModel` subclass. The heavy
dependencies (``dspy``, ``azure-identity``) are loaded lazily so the core
package stays importable on a bare install.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from commonlid.core.lid_model import LIDModel
from commonlid.cost.rate_cards import litellm_rate_card
from commonlid.cost.usage import (
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    REASONING_TOKENS,
    Range,
    Usage,
)

# NOTE: :class:`DSPyLLMModel` is NOT auto-registered with ``@register_model``
# because it requires per-instance configuration (API endpoint, model name,
# auth). The CLI builds one on the fly when ``--model dspy:<llm-model-name>``
# is passed to ``commonlid run``; Python API users instantiate it directly.

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "You are a language expert. Identify the language of the input text in ISO 639-3."
)

# Fallback when no tokenizer resolves for the model: rough chars per token.
_HEURISTIC_CHARS_PER_TOKEN = 4

# Per-sample assumptions for what the model generates. The ChatAdapter answer,
# the output field marker plus the code plus the completion marker, is ~16
# tokens; the high end allows for chatter around it. Reasoning budgets are a
# guess for a one-word answer and dominate the estimate for reasoning models,
# which is what `--calibrate` is for.
_OUTPUT_TOKENS_ASSUMPTION = Range(12, 16, 32)
_REASONING_TOKENS_ASSUMPTION = Range(0, 256, 1024)


def _build_signature(instruction: str) -> Any:
    import dspy

    class LangIDSignature(dspy.Signature):  # type: ignore[misc]
        text = dspy.InputField(desc="Input text")
        language_iso639_3 = dspy.OutputField(
            desc="Language of input text as ISO 639-3 (three-letter code)"
        )

    LangIDSignature.__doc__ = instruction
    return LangIDSignature


class DSPyLangIDModule:
    """Lightweight replacement for ``llm_eval.dspy_langid_module.DSPyLangIDModule``."""

    def __init__(self, instruction: str = DEFAULT_INSTRUCTION) -> None:
        import dspy

        self.signature = _build_signature(instruction)
        self.predictor = dspy.Predict(self.signature)

    def __call__(self, text: str) -> Any:
        return self.predictor(text=text)


class DSPyLLMModel(LIDModel):
    """Evaluate an LLM (via DSPy) as a LID model.

    Parameters
    ----------
    llm_model_name:
        DSPy model id, e.g. ``"azure/gpt-4o-mini"``.
    api_base:
        Base URL of the LLM provider (e.g. the Azure endpoint).
    api_version:
        Optional API version string (Azure).
    api_key:
        Optional API key (when not using AAD bearer tokens).
    azure_ad_token:
        If ``True``, use ``DefaultAzureCredential`` to obtain a bearer token.
    temperature:
        Sampling temperature passed to ``dspy.LM``.
    max_tokens:
        Max generated tokens.
    cache_dir:
        Directory for the per-batch prediction cache; ``None`` disables caching.
    batch_size:
        Size of DSPy evaluation batches.
    n_threads:
        Number of threads in the DSPy evaluator.
    """

    model_id: str = "dspy_llm"
    requires_preprocessing = False

    def __init__(
        self,
        *,
        llm_model_name: str,
        api_base: str,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 100,
        n_threads: int = 1,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        super().__init__()
        self.llm_model_name = llm_model_name
        self.api_base = api_base
        self.api_version = api_version
        self.api_key = api_key
        self.azure_ad_token = azure_ad_token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.instruction = instruction
        self._module: DSPyLangIDModule | None = None
        self._lm: Any = None
        self._overhead_tokens: int | None = None
        # Priced by model name, so the card is per instance.
        self.rate_card = litellm_rate_card(llm_model_name)
        # Customise the registered id when a model name is supplied so multiple
        # instantiations of this class end up under unique cache folders.
        self.model_id = f"dspy_{llm_model_name.replace('/', '_')}"

    def load(self) -> None:
        if self._loaded:
            return

        import dspy

        self._lm = self._build_lm()
        dspy.configure(lm=self._lm, cache=False)
        self._module = DSPyLangIDModule(instruction=self.instruction)
        super().load()

    def _build_lm(self) -> Any:
        import dspy

        kwargs: dict[str, Any] = {
            "model": self.llm_model_name,
            "api_base": self.api_base,
            "cache": False,
        }
        if self.api_version is not None:
            kwargs["api_version"] = self.api_version
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.azure_ad_token:
            kwargs["azure_ad_token_provider"] = _azure_token_provider()
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        return dspy.LM(**kwargs)

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        import dspy

        assert self._module is not None  # load() has run
        examples = [dspy.Example(text=t).with_inputs("text") for t in texts]
        df = _batched_predict(
            examples=examples,
            module=self._module,
            batch_size=self.batch_size,
            n_threads=self.n_threads,
            cache_path=self._cache_path(texts),
        )
        return [self._coerce(code) for code in df["language_iso639_3"].tolist()]

    def _estimate_usage(self, texts: Sequence[str]) -> Usage:
        """Count prompt tokens with the model's own tokenizer, via LiteLLM.

        Each request is the system instruction plus DSPy's chat scaffolding
        (measured once on an empty input) plus the sample text.
        """
        if self._overhead_tokens is None:
            empty = self._render_messages("")
            counted = _count_tokens(self.llm_model_name, messages=empty)
            if counted is None:
                logger.warning(
                    "No tokenizer resolved for %r; counting ~%d chars/token instead. "
                    "Install `transformers` for Hugging Face models.",
                    self.llm_model_name,
                    _HEURISTIC_CHARS_PER_TOKEN,
                )
                counted = _heuristic_tokens("".join(str(m["content"]) for m in empty))
            self._overhead_tokens = counted
        total = 0
        for text in texts:
            counted = _count_tokens(self.llm_model_name, text=text)
            total += self._overhead_tokens + (
                counted if counted is not None else _heuristic_tokens(text)
            )
        return {INPUT_TOKENS: float(total)}

    def _render_messages(self, text: str) -> list[dict[str, Any]]:
        """The chat messages DSPy sends for one sample."""
        import dspy

        signature = _build_signature(self.instruction)
        messages = dspy.ChatAdapter().format(signature=signature, demos=[], inputs={"text": text})
        return cast("list[dict[str, Any]]", messages)

    def usage_assumptions(self) -> dict[str, Range]:
        assumptions = {OUTPUT_TOKENS: _OUTPUT_TOKENS_ASSUMPTION}
        if _supports_reasoning(self.llm_model_name):
            assumptions[REASONING_TOKENS] = _REASONING_TOKENS_ASSUMPTION
        return assumptions

    def measure_usage(self, texts: Sequence[str]) -> list[Usage]:
        """Predict ``texts`` live and read each request's usage from the LM history."""
        if not self._loaded:
            self.load()
        # A batch cache hit would skip the API and leave no history to read.
        cache_dir, self.cache_dir = self.cache_dir, None
        self._lm.history.clear()
        try:
            self.predict_scored(texts)
        finally:
            self.cache_dir = cache_dir
        # DSPy caps the history at `max_history_size` entries, so a very
        # large calibration keeps only the latest; the mean is unaffected.
        return [_usage_from_response(entry.get("usage") or {}) for entry in self._lm.history]

    @staticmethod
    def _coerce(code: str | None) -> str | None:
        if code is None:
            return None
        code = code.strip()
        if not code:
            return None
        return code

    def _cache_path(self, texts: Sequence[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            json.dumps(list(texts), sort_keys=False, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.model_id}_{digest}"


def _count_tokens(
    model: str, *, text: str | None = None, messages: list[dict[str, Any]] | None = None
) -> int | None:
    """Count tokens via LiteLLM; ``None`` if no tokenizer resolves for ``model``."""
    import litellm

    try:
        return int(litellm.token_counter(model=model, text=text, messages=messages))
    except Exception as exc:  # LiteLLM raises assorted errors on unknown tokenizers
        logger.debug("token_counter failed for %s (%s): %s", model, type(exc).__name__, exc)
        return None


def _heuristic_tokens(text: str) -> int:
    return max(1, len(text) // _HEURISTIC_CHARS_PER_TOKEN) if text else 0


def _supports_reasoning(model: str) -> bool:
    import litellm

    try:
        return bool(litellm.supports_reasoning(model=model))
    except Exception:  # unknown models raise instead of returning False
        return False


def _usage_from_response(usage: dict[str, Any]) -> Usage:
    """Split a LiteLLM usage block into billable meters.

    OpenAI-style ``completion_tokens`` already include reasoning tokens, so
    they are subtracted out to avoid billing them twice.
    """
    details = usage.get("completion_tokens_details")
    reasoning = (
        details.get("reasoning_tokens")
        if isinstance(details, dict)
        else getattr(details, "reasoning_tokens", None)
    ) or 0
    completion = usage.get("completion_tokens") or 0
    return {
        INPUT_TOKENS: float(usage.get("prompt_tokens") or 0),
        OUTPUT_TOKENS: float(completion - reasoning),
        REASONING_TOKENS: float(reasoning),
    }


def _azure_token_provider() -> Any:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )


def _batched_predict(
    *,
    examples: list[Any],
    module: DSPyLangIDModule,
    batch_size: int,
    n_threads: int,
    cache_path: Path | None,
) -> Any:
    """Run DSPy's evaluator in batches, appending cached batches to a JSONL file."""
    import pandas as pd

    dfs: list[Any] = []
    for batch_idx in range(0, len(examples), batch_size):
        batch = examples[batch_idx : batch_idx + batch_size]
        batch_cache = (
            None
            if cache_path is None
            else cache_path.with_name(f"{cache_path.name}_batch_{batch_idx // batch_size}.jsonl")
        )
        df = _predict_batch_with_cache(
            examples=batch, module=module, n_threads=n_threads, cache_path=batch_cache
        )
        dfs.append(df)
    return pd.concat(dfs) if len(dfs) > 1 else dfs[0]


def _predict_batch_with_cache(
    *,
    examples: list[Any],
    module: DSPyLangIDModule,
    n_threads: int,
    cache_path: Path | None,
) -> Any:
    import pandas as pd

    if cache_path is not None and cache_path.exists():
        logger.info("Loading DSPy predictions from cache: %s", cache_path)
        return pd.read_json(cache_path, lines=True)

    import dspy

    evaluator = dspy.Evaluate(
        devset=examples,
        metric=lambda _example, _pred, _trace=None: True,
        num_threads=n_threads,
        display_progress=True,
        provide_traceback=True,
    )
    eval_out = evaluator(program=module.predictor)
    df = pd.DataFrame([
        {**example.toDict(), **prediction} for example, prediction, _ in eval_out.results
    ])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(cache_path, orient="records", lines=True)
        logger.info("Saved DSPy predictions to cache: %s", cache_path)
    return df
