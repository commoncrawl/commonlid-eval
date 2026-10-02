"""DSPyLLMModel tests — heavily mocked so no real LLM is called."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest


def test_model_id_uses_llm_model_name() -> None:
    from commonlid.models.dspy_llm import DSPyLLMModel

    m = DSPyLLMModel(llm_model_name="azure/gpt-4o", api_base="https://example")
    assert m.model_id == "dspy_azure_gpt-4o"


def test_cache_path_shape(tmp_path: Path) -> None:
    from commonlid.models.dspy_llm import DSPyLLMModel

    m = DSPyLLMModel(llm_model_name="azure/gpt", api_base="https://example", cache_dir=tmp_path)
    path = m._cache_path(["hello", "world"])
    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith(m.model_id)


def test_predict_batch_goes_through_batched_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from commonlid.models.dspy_llm import DSPyLLMModel

    captured: dict[str, Any] = {}

    def fake_batched_predict(
        *,
        examples: list[Any],
        module: Any,
        batch_size: int,
        n_threads: int,
        cache_path: Path | None,
    ) -> pd.DataFrame:
        captured["n_examples"] = len(examples)
        captured["batch_size"] = batch_size
        captured["n_threads"] = n_threads
        return pd.DataFrame({"language_iso639_3": ["eng", "deu"]})

    def fake_load_self(self: DSPyLLMModel) -> None:
        self._module = object()  # only tested via reference
        self._loaded = True

    monkeypatch.setattr("commonlid.models.dspy_llm._batched_predict", fake_batched_predict)
    monkeypatch.setattr(DSPyLLMModel, "load", fake_load_self)

    model = DSPyLLMModel(
        llm_model_name="azure/gpt", api_base="https://example", batch_size=50, n_threads=2
    )
    preds = model.predict(["hi", "tag"])
    assert preds == ["eng", "deu"]
    assert captured["batch_size"] == 50
    assert captured["n_threads"] == 2


def test_coerce_strips_and_nones_empty() -> None:
    from commonlid.models.dspy_llm import DSPyLLMModel

    assert DSPyLLMModel._coerce("  eng  ") == "eng"
    assert DSPyLLMModel._coerce("") is None
    assert DSPyLLMModel._coerce(None) is None


def test_requires_preprocessing_is_false() -> None:
    from commonlid.models.dspy_llm import DSPyLLMModel

    m = DSPyLLMModel(llm_model_name="azure/gpt", api_base="https://example")
    assert m.requires_preprocessing is False


def test_build_signature_has_instruction_docstring() -> None:
    from commonlid.models.dspy_llm import _build_signature

    sig = _build_signature("pick the language")
    assert sig.__doc__ == "pick the language"
    # dspy.Signature classes expose the declared fields on their namespace.
    assert "text" in sig.model_fields
    assert "language_iso639_3" in sig.model_fields


def test_dspy_langid_module_instantiation() -> None:
    from commonlid.models.dspy_llm import DSPyLangIDModule

    module = DSPyLangIDModule(instruction="any instruction")
    assert module.predictor is not None


def test_load_configures_dspy_lm_and_builds_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import dspy

    from commonlid.models import dspy_llm as dspy_llm_mod
    from commonlid.models.dspy_llm import DSPyLLMModel

    captured: dict[str, Any] = {}

    class _FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            captured["lm_kwargs"] = kwargs

    def fake_configure(*, lm: Any, cache: bool) -> None:
        captured["configured_lm"] = lm
        captured["configured_cache"] = cache

    class _FakeModule:
        def __init__(self, instruction: str) -> None:
            captured["module_instruction"] = instruction

    monkeypatch.setattr(dspy_llm_mod, "DSPyLangIDModule", _FakeModule)
    monkeypatch.setattr(dspy, "LM", _FakeLM)
    monkeypatch.setattr(dspy, "configure", fake_configure)

    model = DSPyLLMModel(
        llm_model_name="azure/gpt",
        api_base="https://example",
        api_version="v1",
        api_key="key",
        temperature=0.7,
        max_tokens=128,
        max_completion_tokens=64,
        instruction="identify it",
    )
    model.load()

    assert model._loaded is True
    assert captured["lm_kwargs"]["model"] == "azure/gpt"
    assert captured["lm_kwargs"]["api_version"] == "v1"
    assert captured["lm_kwargs"]["api_key"] == "key"
    assert captured["lm_kwargs"]["temperature"] == 0.7
    assert captured["lm_kwargs"]["max_tokens"] == 128
    assert captured["lm_kwargs"]["max_completion_tokens"] == 64
    assert captured["configured_cache"] is False
    assert captured["module_instruction"] == "identify it"

    # A second load() call is a no-op.
    captured.clear()
    model.load()
    assert "lm_kwargs" not in captured


def test_load_uses_azure_token_provider_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    import dspy

    from commonlid.models import dspy_llm as dspy_llm_mod
    from commonlid.models.dspy_llm import DSPyLLMModel

    captured: dict[str, Any] = {}

    class _FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            captured["lm_kwargs"] = kwargs

    monkeypatch.setattr(dspy, "LM", _FakeLM)
    monkeypatch.setattr(dspy, "configure", lambda **_kw: None)
    monkeypatch.setattr(dspy_llm_mod, "DSPyLangIDModule", lambda **_kw: object())
    monkeypatch.setattr(dspy_llm_mod, "_azure_token_provider", lambda: "TOKEN_PROVIDER")

    model = DSPyLLMModel(
        llm_model_name="azure/gpt", api_base="https://example", azure_ad_token=True
    )
    model.load()
    assert captured["lm_kwargs"]["azure_ad_token_provider"] == "TOKEN_PROVIDER"


def test_batched_predict_concatenates_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from commonlid.models import dspy_llm as dspy_llm_mod

    def fake_predict_batch_with_cache(
        *,
        examples: list[Any],
        module: Any,
        n_threads: int,
        cache_path: Path | None,
    ) -> pd.DataFrame:
        return pd.DataFrame({"language_iso639_3": [f"tag-{len(examples)}"]})

    monkeypatch.setattr(dspy_llm_mod, "_predict_batch_with_cache", fake_predict_batch_with_cache)

    result = dspy_llm_mod._batched_predict(
        examples=[object(), object(), object(), object(), object()],
        module=object(),
        batch_size=2,
        n_threads=1,
        cache_path=None,
    )
    # 5 examples / batch_size 2 → batches of [2, 2, 1]
    assert list(result["language_iso639_3"]) == ["tag-2", "tag-2", "tag-1"]


def test_batched_predict_single_batch_returns_first_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from commonlid.models import dspy_llm as dspy_llm_mod

    df = pd.DataFrame({"language_iso639_3": ["eng", "deu"]})
    monkeypatch.setattr(
        dspy_llm_mod,
        "_predict_batch_with_cache",
        lambda **_kw: df,
    )
    out = dspy_llm_mod._batched_predict(
        examples=[object(), object()],
        module=object(),
        batch_size=10,
        n_threads=1,
        cache_path=None,
    )
    # Single batch path must return the df unchanged (no extra concat copy).
    assert out is df


def test_predict_batch_with_cache_reads_from_disk(tmp_path: Path) -> None:
    import pandas as pd

    from commonlid.models.dspy_llm import _predict_batch_with_cache

    cache_path = tmp_path / "cache.jsonl"
    pd.DataFrame({"language_iso639_3": ["eng", "deu"]}).to_json(
        cache_path, orient="records", lines=True
    )

    df = _predict_batch_with_cache(
        examples=[object(), object()], module=object(), n_threads=1, cache_path=cache_path
    )
    assert list(df["language_iso639_3"]) == ["eng", "deu"]


def test_predict_batch_with_cache_writes_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import dspy

    from commonlid.models.dspy_llm import _predict_batch_with_cache

    class _FakeExample:
        def toDict(self) -> dict[str, Any]:  # noqa: N802 - dspy API name
            return {"text": "hello"}

    class _FakeResult:
        results: ClassVar[list[tuple[Any, dict[str, str], float]]] = [
            (_FakeExample(), {"language_iso639_3": "eng"}, 1.0)
        ]

    class _FakeEvaluator:
        def __init__(self, **_kw: Any) -> None:
            pass

        def __call__(self, *, program: Any) -> _FakeResult:
            return _FakeResult()

    class _FakeModule:
        predictor = "predictor"

    monkeypatch.setattr(dspy, "Evaluate", _FakeEvaluator)

    cache_path = tmp_path / "cache.jsonl"
    df = _predict_batch_with_cache(
        examples=[_FakeExample()],
        module=_FakeModule(),  # type: ignore[arg-type]
        n_threads=1,
        cache_path=cache_path,
    )
    assert cache_path.exists()
    assert list(df["language_iso639_3"]) == ["eng"]


def test_predict_batch_with_cache_without_cache_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dspy

    from commonlid.models.dspy_llm import _predict_batch_with_cache

    class _FakeExample:
        def toDict(self) -> dict[str, Any]:  # noqa: N802
            return {"text": "world"}

    class _FakeResult:
        results: ClassVar[list[tuple[Any, dict[str, str], float]]] = [
            (_FakeExample(), {"language_iso639_3": "deu"}, 1.0)
        ]

    class _FakeEvaluator:
        def __init__(self, **_kw: Any) -> None:
            pass

        def __call__(self, *, program: Any) -> _FakeResult:
            return _FakeResult()

    class _FakeModule:
        predictor = "predictor"

    monkeypatch.setattr(dspy, "Evaluate", _FakeEvaluator)
    df = _predict_batch_with_cache(
        examples=[_FakeExample()],
        module=_FakeModule(),  # type: ignore[arg-type]
        n_threads=1,
        cache_path=None,
    )
    assert list(df["language_iso639_3"]) == ["deu"]


def test_azure_token_provider_uses_default_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import azure.identity

    from commonlid.models.dspy_llm import _azure_token_provider

    captured: dict[str, Any] = {}

    class _FakeCredential:
        def __init__(self) -> None:
            captured["credential"] = self

    def fake_get_bearer_token_provider(credential: Any, scope: str) -> str:
        captured["scope"] = scope
        captured["given_credential"] = credential
        return "TOKEN_PROVIDER"

    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", _FakeCredential)
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider", fake_get_bearer_token_provider)
    provider = _azure_token_provider()
    assert provider == "TOKEN_PROVIDER"
    assert captured["scope"].endswith(".default")


def _model(name: str = "openai/gpt-x") -> Any:
    from commonlid.models.dspy_llm import DSPyLLMModel

    return DSPyLLMModel(llm_model_name=name, api_base="https://example")


def test_estimate_usage_counts_overhead_plus_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    def fake_counter(model: str, text: Any = None, messages: Any = None) -> int:
        return 100 if messages is not None else len(text)

    monkeypatch.setattr(litellm, "token_counter", fake_counter)
    model = _model()
    assert model.estimate_usage(["abc", "hello"]) == {"input_tokens": 208.0}
    assert model._overhead_tokens == 100


def test_estimate_usage_falls_back_to_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    def boom(**_kw: Any) -> int:
        raise ValueError("no tokenizer")

    monkeypatch.setattr(litellm, "token_counter", boom)
    model = _model()
    usage = model.estimate_usage(["a" * 40])
    assert model._overhead_tokens > 0
    assert usage == {"input_tokens": float(model._overhead_tokens + 10)}


def test_usage_assumptions_add_reasoning_for_reasoning_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setattr(litellm, "supports_reasoning", lambda model: model == "openai/r")
    assert set(_model("openai/r").usage_assumptions()) == {"output_tokens", "reasoning_tokens"}
    assert set(_model("openai/plain").usage_assumptions()) == {"output_tokens"}

    def boom(model: str) -> bool:
        raise ValueError("unknown model")

    monkeypatch.setattr(litellm, "supports_reasoning", boom)
    assert set(_model().usage_assumptions()) == {"output_tokens"}


def test_rate_card_comes_from_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    from commonlid.models import dspy_llm as dspy_llm_mod

    monkeypatch.setattr(dspy_llm_mod, "litellm_rate_card", lambda name: f"card:{name}")
    assert _model("azure/m").rate_card() == "card:azure/m"


def test_usage_from_response_splits_out_reasoning() -> None:
    from types import SimpleNamespace

    from commonlid.models.dspy_llm import _usage_from_response

    usage = {
        "prompt_tokens": 300,
        "completion_tokens": 120,
        "completion_tokens_details": {"reasoning_tokens": 100},
    }
    expected = {"input_tokens": 300.0, "output_tokens": 20.0, "reasoning_tokens": 100.0}
    assert _usage_from_response(usage) == expected
    usage["completion_tokens_details"] = SimpleNamespace(reasoning_tokens=100)
    assert _usage_from_response(usage) == expected
    assert _usage_from_response({}) == {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "reasoning_tokens": 0.0,
    }


def test_measure_usage_reads_lm_history_and_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from commonlid.models.dspy_llm import DSPyLLMModel

    model = DSPyLLMModel(llm_model_name="openai/x", api_base="https://e", cache_dir=tmp_path)
    model._lm = type("LM", (), {"history": [{"usage": {"prompt_tokens": 1}}]})()
    model._loaded = True
    seen_cache: list[Any] = []

    def fake_predict(texts: list[str]) -> list[str]:
        seen_cache.append(model.cache_dir)
        model._lm.history.extend(
            {"usage": {"prompt_tokens": 10, "completion_tokens": 3}} for _ in texts
        )
        return ["eng"] * len(texts)

    monkeypatch.setattr(model, "_predict_batch", fake_predict)
    usage = model.measure_usage(["a", "b"])

    # Stale history is dropped, the cache is off during the call and restored after.
    assert usage == [{"input_tokens": 10.0, "output_tokens": 3.0, "reasoning_tokens": 0.0}] * 2
    assert seen_cache == [None]
    assert model.cache_dir == tmp_path
