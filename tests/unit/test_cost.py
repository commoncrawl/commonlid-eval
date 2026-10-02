"""Unit tests for cost estimation (``commonlid.cost``)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import litellm
import pytest
from typer.testing import CliRunner

from commonlid.cli import app
from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_dataset, register_model
from commonlid.cost import (
    Range,
    RateCard,
    estimate_cost,
    format_estimate,
    get_hardware_card,
    hourly_rate_card,
    litellm_rate_card,
)
from commonlid.cost.estimate import _sample_texts
from commonlid.cost.usage import add_usage

runner = CliRunner()

TEXTS = ["abc", "hello world", "", "xyz!"]


class StubDataset:
    """LIDDataset-shaped stub: no HF download."""

    dataset_id = "stub"

    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = TEXTS if texts is None else texts

    def iter_batches(self, batch_size: int = 64, *, limit: int | None = None):
        for start in range(0, len(self._texts), batch_size):
            batch = self._texts[start : start + batch_size]
            yield list(batch), [None] * len(batch)

    def __len__(self) -> int:
        return len(self._texts)


class MeteredModel(LIDModel):
    """Billed per character, with an assumed meter it can also measure."""

    model_id = "metered"
    requires_preprocessing = False

    def __init__(self, measured: list[dict[str, float]] | None = None) -> None:
        super().__init__()
        self.measured = measured
        self.measured_texts: list[str] = []

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return ["eng"] * len(texts)

    def _estimate_usage(self, texts: Sequence[str]) -> dict[str, float]:
        return {"characters": float(sum(len(t) for t in texts))}

    def usage_assumptions(self) -> dict[str, Range]:
        return {"output_tokens": Range(1, 2, 4)}

    def rate_card(self) -> RateCard:
        return RateCard("stub-card", {"characters": 0.5, "output_tokens": 1.0})

    def measure_usage(self, texts: Sequence[str]) -> list[dict[str, float]] | None:
        self.measured_texts = list(texts)
        return self.measured


class LocalModel(LIDModel):
    model_id = "local"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return ["eng"] * len(texts)


# --- Range -------------------------------------------------------------------


def test_range_parse_exact_and_triple() -> None:
    assert Range.parse("5") == Range(5, 5, 5)
    assert Range.parse("0:200:1000") == Range(0, 200, 1000)


@pytest.mark.parametrize("spec", ["", "1:2", "a", "3:2:1"])
def test_range_parse_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        Range.parse(spec)


def test_range_arithmetic() -> None:
    r = Range(1, 2, 4)
    assert r.scale(10) == Range(10, 20, 40)
    assert r + Range.exact(1) == Range(2, 3, 5)
    assert r.reciprocal() == Range(0.25, 0.5, 1.0)
    assert not r.is_exact
    assert Range.exact(3).is_exact


def test_range_reciprocal_rejects_zero() -> None:
    with pytest.raises(ValueError):
        Range(0, 1, 2).reciprocal()


def test_range_from_observations_is_mean_with_ci() -> None:
    r = Range.from_observations([1.0, 2.0, 3.0])
    assert r.expected == pytest.approx(2.0)
    # Half-width is 1.96 standard errors; the sample stdev here is 1.
    assert r.high - r.expected == pytest.approx(1.96 / 3**0.5)
    assert Range.from_observations([7.0]) == Range.exact(7.0)
    # Clipped at zero: usage is never negative.
    assert Range.from_observations([0.0, 0.0, 10.0]).low == 0.0
    with pytest.raises(ValueError):
        Range.from_observations([])


def test_add_usage_accumulates() -> None:
    total = {"a": 1.0}
    add_usage(total, {"a": 2.0, "b": 3.0})
    assert total == {"a": 3.0, "b": 3.0}


# --- rate cards ----------------------------------------------------------------


def test_hardware_cards() -> None:
    assert get_hardware_card("aws:g5.xlarge").rates["instance_hours"] == pytest.approx(1.006)
    with pytest.raises(KeyError, match="known hardware"):
        get_hardware_card("nope")


def test_rate_card_defaults_to_class_pricing() -> None:
    assert LocalModel().rate_card() is None

    class Priced(LocalModel):
        pricing = RateCard("priced", {"characters": 1.0})

    assert Priced().rate_card() is Priced.pricing


def test_hourly_rate_card() -> None:
    card = hourly_rate_card(0.25)
    assert card.rates == {"instance_hours": 0.25}


def test_with_rates_overrides_and_notes() -> None:
    card = RateCard("c", {"a": 1.0})
    assert card.with_rates({}) is card
    updated = card.with_rates({"a": 2.0, "b": 3.0})
    assert updated.rates == {"a": 2.0, "b": 3.0}
    assert "overridden" in updated.notes[-1]


def test_litellm_rate_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda _model: {"input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6},
    )
    card = litellm_rate_card("openai/x")
    assert card is not None
    # Reasoning is billed as output when LiteLLM lists no separate rate.
    assert card.rates == {"input_tokens": 1e-6, "output_tokens": 4e-6, "reasoning_tokens": 4e-6}


def test_litellm_rate_card_unknown_or_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str) -> Any:
        raise ValueError("unknown")

    monkeypatch.setattr(litellm, "get_model_info", boom)
    assert litellm_rate_card("x") is None
    monkeypatch.setattr(litellm, "get_model_info", lambda _model: {"input_cost_per_token": 0})
    assert litellm_rate_card("x") is None


# --- estimate_cost -------------------------------------------------------------


def test_counted_and_assumed_meters() -> None:
    est = estimate_cost(MeteredModel(), StubDataset())
    meters = {m.meter: m for m in est.meters}

    chars = meters["characters"]
    assert chars.source == "counted"
    assert chars.total == Range.exact(18)  # 3 + 11 + 0 + 4
    assert chars.cost == Range.exact(9.0)

    out = meters["output_tokens"]
    assert out.source == "assumed"
    assert out.total == Range(4, 8, 16)
    assert est.total_cost == Range(13, 17, 25)


def test_user_assumptions_override_defaults() -> None:
    est = estimate_cost(
        MeteredModel(), StubDataset(), assumptions={"output_tokens": Range.exact(10)}
    )
    out = next(m for m in est.meters if m.meter == "output_tokens")
    assert out.total == Range.exact(40)


def test_counted_meter_beats_assumption() -> None:
    est = estimate_cost(MeteredModel(), StubDataset(), assumptions={"characters": Range.exact(99)})
    chars = [m for m in est.meters if m.meter == "characters"]
    assert len(chars) == 1
    assert chars[0].source == "counted"


def test_rate_overrides_and_unpriced_meter() -> None:
    est = estimate_cost(
        MeteredModel(),
        StubDataset(),
        assumptions={"widgets": Range.exact(1)},
        rate_overrides={"characters": 1.0},
    )
    meters = {m.meter: m for m in est.meters}
    assert meters["characters"].cost == Range.exact(18.0)
    assert meters["widgets"].cost is None
    assert any("does not price 'widgets'" in n for n in est.notes)


def test_calibration_replaces_assumptions() -> None:
    model = MeteredModel(measured=[{"output_tokens": 3.0}, {"output_tokens": 5.0}])
    est = estimate_cost(model, StubDataset(), calibrate=2, seed=1)
    out = next(m for m in est.meters if m.meter == "output_tokens")
    assert out.source == "calibrated"
    assert out.per_sample.expected == pytest.approx(4.0)
    assert len(model.measured_texts) == 2
    assert any("Calibrated on 2" in n for n in est.notes)


def test_calibration_without_reported_usage_keeps_assumptions() -> None:
    est = estimate_cost(MeteredModel(measured=None), StubDataset(), calibrate=2)
    out = next(m for m in est.meters if m.meter == "output_tokens")
    assert out.source == "assumed"
    assert any("reports no usage" in n for n in est.notes)


def test_unmetered_model_without_hardware() -> None:
    est = estimate_cost(LocalModel(), StubDataset())
    assert est.meters == []
    assert est.total_cost is None
    assert any("not billed per call" in n for n in est.notes)


def test_unpriced_model_gets_a_note() -> None:
    class Unpriced(MeteredModel):
        def rate_card(self) -> None:  # type: ignore[override]
            return None

    est = estimate_cost(Unpriced(), StubDataset())
    assert est.total_cost is None
    assert any("No price known" in n for n in est.notes)

    est = estimate_cost(Unpriced(), StubDataset(), rate_overrides={"characters": 1.0})
    assert est.rate_card is not None
    assert est.rate_card.card_id == "custom"
    assert est.total_cost == Range.exact(18.0)


def test_hardware_with_throughput() -> None:
    card = hourly_rate_card(3600.0)  # $1 per second, for round numbers
    est = estimate_cost(LocalModel(), StubDataset(), hardware=card, throughput=Range(1, 2, 4))
    (meter,) = est.meters
    assert meter.meter == "instance_hours"
    assert meter.source == "assumed"
    # 4 samples at 4 / 2 / 1 samples per second -> 1 / 2 / 4 seconds.
    total = est.total_cost
    assert total is not None
    assert (total.low, total.expected, total.high) == pytest.approx((1.0, 2.0, 4.0))


def test_hardware_needs_throughput_or_calibration() -> None:
    with pytest.raises(ValueError, match="--throughput"):
        estimate_cost(LocalModel(), StubDataset(), hardware=hourly_rate_card(1.0))


def test_hardware_calibration_times_batches() -> None:
    est = estimate_cost(
        LocalModel(),
        StubDataset(["a"] * 10),
        hardware=hourly_rate_card(1.0),
        calibrate=6,
        batch_size=2,
    )
    (meter,) = est.meters
    assert meter.source == "calibrated"
    assert meter.per_sample.expected > 0
    assert any("this machine" in n for n in est.notes)


def test_sample_texts_is_seeded_and_bounded() -> None:
    ds = StubDataset([str(i) for i in range(100)])
    first = _sample_texts(ds, 10, seed=3)
    assert len(first) == 10
    assert first == _sample_texts(ds, 10, seed=3)
    assert first != _sample_texts(ds, 10, seed=4)
    assert len(_sample_texts(ds, 1000, seed=0)) == 100


def test_format_and_to_dict() -> None:
    est = estimate_cost(MeteredModel(), StubDataset())
    text = format_estimate(est)
    assert "metered on stub (4 samples)" in text
    assert "Rate card: stub-card" in text
    assert "Total: $17.00 ($13.00-$25.00)" in text

    data = est.to_dict()
    assert data["total_cost_usd"] == {"low": 13.0, "expected": 17.0, "high": 25.0}
    assert data["meters"][0]["source"] == "counted"
    json.dumps(data)  # serialisable


def test_format_small_rates_and_costs() -> None:
    card = RateCard("per-million", {"characters": 20e-6})
    model = MeteredModel()
    model.rate_card = lambda: card  # type: ignore[method-assign]
    text = format_estimate(estimate_cost(model, StubDataset()))
    assert "$20/M" in text
    assert "$0.00036" in text


# --- CLI -----------------------------------------------------------------------


@pytest.fixture
def stub_registry(fresh_registry: None) -> None:
    register_model(MeteredModel)
    register_model(LocalModel)
    register_dataset(StubDataset)


def test_cli_estimate_cost_text_and_json(stub_registry: None) -> None:
    result = runner.invoke(
        app, ["estimate-cost", "-m", "metered", "-d", "stub", "--assume", "output_tokens=1:2:3"]
    )
    assert result.exit_code == 0, result.output
    assert "Total: $17.00 ($13.00-$21.00)" in result.stdout

    result = runner.invoke(app, ["estimate-cost", "-m", "metered", "-d", "stub", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["model_id"] == "metered"


def test_cli_estimate_cost_hardware(stub_registry: None) -> None:
    result = runner.invoke(
        app,
        [
            "estimate-cost",
            *("-m", "local", "-d", "stub"),
            *("--hardware", "aws:g5.xlarge", "--throughput", "1:2:4"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "instance_hours" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--assume", "no-equals"],
        ["--assume", "x=1:2"],
        ["--rate", "x=abc"],
        ["--hardware", "nope"],
        ["--hardware", "aws:g5.xlarge", "--hourly-rate", "1"],
        ["--hourly-rate", "1"],  # no throughput
    ],
)
def test_cli_estimate_cost_rejects_bad_options(stub_registry: None, args: list[str]) -> None:
    result = runner.invoke(app, ["estimate-cost", "-m", "local", "-d", "stub", *args])
    assert result.exit_code == 2


def test_cli_dspy_spec_needs_api_base_only_to_calibrate(
    stub_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from commonlid.models import dspy_llm

    monkeypatch.setattr(dspy_llm.DSPyLLMModel, "_estimate_usage", lambda _self, _texts: {})
    monkeypatch.setattr(dspy_llm.DSPyLLMModel, "usage_assumptions", lambda _self: {})
    monkeypatch.setattr(dspy_llm.DSPyLLMModel, "rate_card", lambda _self: None)
    result = runner.invoke(app, ["estimate-cost", "-m", "dspy:openai/x", "-d", "stub"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app, ["estimate-cost", "-m", "dspy:openai/x", "-d", "stub", "--calibrate", "2"]
    )
    assert result.exit_code == 2


def test_cli_list_rate_cards() -> None:
    result = runner.invoke(app, ["list-rate-cards"])
    assert result.exit_code == 0
    assert "google-translate-v2" in result.stdout
    assert "aws:g5.xlarge" in result.stdout

    result = runner.invoke(app, ["list-rate-cards", "--json"])
    assert "hf:nvidia-a10g-x1" in json.loads(result.stdout)
