"""Unit tests for offline token & cost estimation (``commonlid.evaluation.token_cost``)."""

from __future__ import annotations

import litellm
import pytest

from commonlid.evaluation import token_cost
from commonlid.models.dspy_llm import DEFAULT_INSTRUCTION

_OVERHEAD = 100  # fake per-request overhead returned for message-based counts
_IN_RATE = 1e-6  # fake USD per input token
_OUT_RATE = 2e-6  # fake USD per output token


class StubDataset:
    """Minimal LIDDataset-shaped stub: no HF download, deterministic texts."""

    dataset_id = "stub"
    text_column = "text"

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def iter_batches(self, batch_size: int = 64, *, limit: int | None = None):
        texts = self._texts if not limit else self._texts[:limit]
        yield list(texts), [None] * len(texts)

    def __len__(self) -> int:
        return len(self._texts)


def _fake_token_counter(model: str = "", text=None, messages=None, **_kw) -> int:
    if messages is not None:
        return _OVERHEAD
    return len(text or "")


def _fake_cost_per_token(
    model: str = "", prompt_tokens: int = 0, completion_tokens: int = 0, **_kw
):
    return prompt_tokens * _IN_RATE, completion_tokens * _OUT_RATE


@pytest.fixture
def priced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "token_counter", _fake_token_counter)
    monkeypatch.setattr(litellm, "cost_per_token", _fake_cost_per_token)


def test_input_output_token_arithmetic(priced: None) -> None:
    texts = ["abc", "hello world"]  # lengths 3, 11
    est = token_cost.estimate(
        "openai/gpt-5", StubDataset(texts), output_tokens=5, reasoning_tokens=0
    )

    assert est.n_samples == 2
    assert est.per_request_overhead_tokens == _OVERHEAD
    # each input = overhead + len(text)
    assert est.total_input_tokens == (_OVERHEAD + 3) + (_OVERHEAD + 11)
    assert est.total_output_tokens == 2 * 5
    assert est.mean_input_tokens == est.total_input_tokens / 2
    assert est.to_dict()["total_input_tokens"] == est.total_input_tokens


def test_reasoning_tokens_roll_into_output(priced: None) -> None:
    texts = ["a", "b", "c"]
    est = token_cost.estimate(
        "openai/gpt-5", StubDataset(texts), output_tokens=10, reasoning_tokens=300
    )

    assert est.total_output_tokens == 3 * (10 + 300)
    assert est.output_cost_usd == pytest.approx(est.total_output_tokens * _OUT_RATE)
    assert est.total_cost_usd == pytest.approx(
        est.total_input_tokens * _IN_RATE + est.total_output_tokens * _OUT_RATE
    )


def test_dspy_prefix_is_stripped(priced: None) -> None:
    est = token_cost.estimate("dspy:openai/gpt-5", StubDataset(["x"]))
    assert est.model == "openai/gpt-5"


def test_unknown_model_reports_tokens_without_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "token_counter", _fake_token_counter)

    def _raise(**_kw):
        raise ValueError("model not in price map")

    monkeypatch.setattr(litellm, "cost_per_token", _raise)

    est = token_cost.estimate("some/unknown-model", StubDataset(["abc"]))
    assert est.price_available is False
    assert est.input_cost_usd is None
    assert est.output_cost_usd is None
    assert est.total_cost_usd is None
    assert est.total_input_tokens == _OVERHEAD + 3  # tokens still populated


def test_zero_price_treated_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # LiteLLM returns (0, 0) for models absent from its price map instead of raising.
    monkeypatch.setattr(litellm, "token_counter", _fake_token_counter)
    monkeypatch.setattr(litellm, "cost_per_token", lambda **_kw: (0.0, 0.0))

    est = token_cost.estimate("huggingface/org/model", StubDataset(["abc"]))
    assert est.price_available is False
    assert est.total_cost_usd is None
    assert est.total_input_tokens == _OVERHEAD + 3


def test_missing_tokenizer_falls_back_with_note(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_tc(**_kw):
        raise KeyError("no tokenizer")

    monkeypatch.setattr(litellm, "token_counter", _raise_tc)
    monkeypatch.setattr(litellm, "cost_per_token", _fake_cost_per_token)

    est = token_cost.estimate("huggingface/org/model", StubDataset(["abcdefgh"]))
    assert est.tokenizer_note is not None
    assert "transformers" in est.tokenizer_note
    # heuristic still produces a positive count -- never crashes
    assert est.total_input_tokens > 0


def test_limit_triggers_full_dataset_extrapolation(priced: None) -> None:
    texts = [f"sample-{i}" for i in range(5)]  # each len 8
    est = token_cost.estimate(
        "openai/gpt-5", StubDataset(texts), output_tokens=4, reasoning_tokens=0, limit=2
    )

    assert est.n_samples == 2
    assert est.n_total == 5
    assert est.extrapolated is True
    assert est.full_total_input_tokens == round(est.mean_input_tokens * 5)
    assert est.full_total_output_tokens == 5 * 4
    assert est.full_total_cost_usd == pytest.approx(
        est.full_total_input_tokens * _IN_RATE + est.full_total_output_tokens * _OUT_RATE
    )


def test_no_extrapolation_when_limit_covers_dataset(priced: None) -> None:
    est = token_cost.estimate("openai/gpt-5", StubDataset(["a", "b"]), limit=5)
    assert est.extrapolated is False
    assert est.full_total_input_tokens is None


def test_render_messages_reproduces_dspy_prompt() -> None:
    messages = token_cost._render_messages(DEFAULT_INSTRUCTION, "Bonjour le monde")
    assert messages[0]["role"] == "system"
    assert "ISO 639-3" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Bonjour le monde" in messages[1]["content"]
