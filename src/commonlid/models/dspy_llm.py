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
from typing import Any

from commonlid.core.lid_model import LIDModel

# NOTE: :class:`DSPyLLMModel` is NOT auto-registered with ``@register_model``
# because it requires per-instance configuration (API endpoint, model name,
# auth). The CLI builds one on the fly when ``--model dspy:<llm-model-name>``
# is passed to ``commonlid run``; Python API users instantiate it directly.

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "You are a language expert. Identify the language of the input text in ISO 639-3."
)


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
        # Customise the registered id when a model name is supplied so multiple
        # instantiations of this class end up under unique cache folders.
        self.model_id = f"dspy_{llm_model_name.replace('/', '_')}"

    def load(self) -> None:
        if self._loaded:
            return

        import dspy

        dspy.configure(lm=self._build_lm(), cache=False)
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
