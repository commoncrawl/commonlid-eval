"""Scoring server: config parsing, the scoring service and the HTTP layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from commonlid.cli import app as cli_app
from commonlid.core.lid_dataset import LIDDataset
from commonlid.server.app import create_app
from commonlid.server.config import (
    ENV_API_TOKEN,
    ENV_PER_LANGUAGE_METRICS,
    ConfigError,
    ServerConfig,
    available_metrics,
    load_config,
    parse_config,
)
from commonlid.server.service import (
    InvalidSubmissionError,
    ScoringService,
    StateMismatchError,
    SubmissionLimitError,
    UnknownDatasetError,
    select_metrics,
)


class _FakeDataset(LIDDataset):
    dataset_id = "_fake"
    source_hf_repo = "fake/fake"
    source_hf_revision = "rev-1"
    text_column = "text"
    target_column = "iso639_3"

    SAMPLES: ClassVar[list[dict[str, str | None]]] = [
        {"text": "hello world", "iso639_3": "eng"},
        {"text": "good morning", "iso639_3": "eng"},
        {"text": "hallo welt", "iso639_3": "deu"},
        {"text": "bonjour", "iso639_3": "fra"},
        {"text": "ciao", "iso639_3": "ita"},
        {"text": "???", "iso639_3": None},
    ]

    def load(self, *, limit: int | None = None):  # type: ignore[override]
        from datasets import Dataset

        ds = Dataset.from_list(self.SAMPLES)
        if limit is not None and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        self._dataset = ds
        return ds


class _OtherDataset(_FakeDataset):
    dataset_id = "_other"


GOLD_BY_TEXT = {s["text"]: s["iso639_3"] for s in _FakeDataset.SAMPLES}


@pytest.fixture(autouse=True)
def _no_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's COMMONLID_SERVER_* variables out of the tests."""
    monkeypatch.delenv(ENV_API_TOKEN, raising=False)
    monkeypatch.delenv(ENV_PER_LANGUAGE_METRICS, raising=False)


def _config(**overrides: Any) -> ServerConfig:
    raw: dict[str, Any] = {
        "datasets": [
            {"id": "_fake", "metrics": ["macro.f1_gold_only", "micro.f1_gold_only"]},
            {"id": "_other", "metrics": ["per_language.f1"], "max_submissions": 1},
        ],
        "max_submissions": 3,
        "per_language_metrics": True,
    }
    raw.update(overrides)
    return parse_config(raw)


def _service(config: ServerConfig | None = None) -> ScoringService:
    return ScoringService(
        config or _config(), datasets={"_fake": _FakeDataset(), "_other": _OtherDataset()}
    )


def _perfect(service: ScoringService, dataset_id: str = "_fake") -> list[tuple[int, str | None]]:
    items = service.inputs(dataset_id)["items"]
    return [(it["id"], GOLD_BY_TEXT[it["text"]]) for it in items]


# ---------------------------------------------------------------------- config


def test_available_metrics_cover_summary_blocks() -> None:
    names = available_metrics()
    assert "macro.f1_gold_only" in names
    assert "micro.precision_observed" in names
    assert "per_language.f1" in names
    assert "n_samples" in names


def test_parse_config_defaults() -> None:
    cfg = parse_config({"datasets": [{"id": "commonlid"}]})
    assert cfg.max_submissions == 10
    assert cfg.datasets[0].metrics == ("macro.f1_gold_only",)
    assert cfg.datasets[0].limit is None
    assert cfg.state_dir is None
    assert cfg.shuffle is True
    assert cfg.per_language_metrics is False
    assert cfg.api_token is None
    assert cfg.dataset("commonlid") is cfg.datasets[0]
    assert cfg.dataset("nope") is None


def test_parse_config_zero_means_unlimited() -> None:
    cfg = parse_config({"max_submissions": 0, "datasets": [{"id": "x", "limit": 0}]})
    assert cfg.max_submissions is None
    assert cfg.datasets[0].limit is None


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({}, "at least one"),
        ({"datasets": []}, "at least one"),
        ({"datasets": [{"id": "x"}], "bogus": 1}, "unknown top-level"),
        ({"datasets": ["x"]}, "must be a table"),
        ({"datasets": [{"metrics": ["macro.f1_gold_only"]}]}, "string `id`"),
        ({"datasets": [{"id": "x", "colour": "red"}]}, "unknown keys"),
        ({"datasets": [{"id": "x", "metrics": []}]}, "non-empty list"),
        ({"datasets": [{"id": "x", "metrics": ["macro.nope"]}]}, "unknown metrics"),
        ({"datasets": [{"id": "x"}, {"id": "x"}]}, "duplicate"),
        ({"datasets": [{"id": "x", "limit": -1}]}, "non-negative"),
        ({"datasets": [{"id": "x"}], "max_submissions": "5"}, "non-negative"),
        ({"datasets": [{"id": "x", "sample_count_threshold": -1}]}, "sample_count_threshold"),
        ({"datasets": [{"id": "x"}], "round_digits": -1}, "round_digits"),
        ({"datasets": [{"id": "x"}], "shuffle_seed": "abc"}, "shuffle_seed"),
        ({"datasets": [{"id": "x"}], "shuffle": "false"}, "shuffle must be true or false"),
        ({"datasets": [{"id": "x"}], "per_language_metrics": "yes"}, "per_language_metrics"),
        ({"datasets": [{"id": "x"}], "port": "8000"}, "port must be an integer"),
        ({"datasets": [{"id": "x"}], "port": 70000}, "port must be an integer"),
        ({"datasets": [{"id": "x"}], "host": 1}, "host must be"),
        ({"datasets": [{"id": "x"}], "api_token": ""}, "api_token"),
        ({"datasets": [{"id": "x", "metrics": ["per_language.f1"]}]}, "per_language_metrics"),
    ],
)
def test_parse_config_rejects_invalid(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_config(raw)


def test_per_language_metrics_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {"datasets": [{"id": "x", "metrics": ["per_language.f1"]}]}
    on = parse_config({**raw, "per_language_metrics": True}, environ={})
    assert on.per_language_metrics is True
    assert parse_config(raw, environ={ENV_PER_LANGUAGE_METRICS: "1"}).per_language_metrics
    # The environment wins over the file, in both directions.
    with pytest.raises(ConfigError, match="per-language"):
        parse_config(
            {**raw, "per_language_metrics": True}, environ={ENV_PER_LANGUAGE_METRICS: "false"}
        )
    with pytest.raises(ConfigError, match=ENV_PER_LANGUAGE_METRICS):
        parse_config(raw, environ={ENV_PER_LANGUAGE_METRICS: "maybe"})
    # environ defaults to os.environ.
    monkeypatch.setenv(ENV_PER_LANGUAGE_METRICS, "true")
    assert parse_config(raw).per_language_metrics is True


def test_api_token_from_config_or_env() -> None:
    raw: dict[str, Any] = {"datasets": [{"id": "x"}], "api_token": "from-file"}
    assert parse_config(raw, environ={}).api_token == "from-file"
    assert parse_config(raw, environ={ENV_API_TOKEN: "from-env"}).api_token == "from-env"


def test_load_config_resolves_state_dir_relative_to_file(tmp_path: Path) -> None:
    path = tmp_path / "server.toml"
    path.write_text(
        'state_dir = "state"\nmax_submissions = 2\nport = 9000\nper_language_metrics = true\n\n'
        '[[datasets]]\nid = "udhr"\nmetrics = ["macro.f1_gold_only", "per_language.f1"]\n'
    )
    cfg = load_config(path)
    assert cfg.state_dir == tmp_path / "state"
    assert cfg.max_submissions == 2
    assert cfg.port == 9000
    assert cfg.datasets[0].metrics == ("macro.f1_gold_only", "per_language.f1")


def test_example_config_is_valid() -> None:
    example = Path(__file__).parents[2] / "docs" / "eval_server.example.toml"
    cfg = load_config(example)
    assert cfg.datasets


# --------------------------------------------------------------------- service


def test_inputs_never_contain_labels() -> None:
    service = _service()
    page = service.inputs("_fake")
    assert page["total"] == len(_FakeDataset.SAMPLES)
    assert [it["id"] for it in page["items"]] == list(range(page["total"]))
    for item in page["items"]:
        assert set(item) == {"id", "text"}
    assert sorted(it["text"] for it in page["items"]) == sorted(GOLD_BY_TEXT)


def test_inputs_pagination() -> None:
    service = _service()
    everything = service.inputs("_fake")["items"]
    page = service.inputs("_fake", offset=2, limit=3)
    assert page["offset"] == 2
    assert page["items"] == everything[2:5]
    assert service.inputs("_fake", offset=5, limit=10)["items"] == everything[5:]


def test_inputs_rejects_negative_paging() -> None:
    with pytest.raises(Exception, match="non-negative"):
        _service().inputs("_fake", offset=-1)


def test_shuffle_is_seeded_and_can_be_disabled() -> None:
    a = _service(_config(shuffle_seed=1)).inputs("_fake")["items"]
    b = _service(_config(shuffle_seed=1)).inputs("_fake")["items"]
    assert a == b
    ordered = _service(_config(shuffle=False)).inputs("_fake")["items"]
    assert [it["text"] for it in ordered] == [s["text"] for s in _FakeDataset.SAMPLES]


def test_unknown_dataset() -> None:
    service = _service()
    with pytest.raises(UnknownDatasetError):
        service.inputs("nope")
    with pytest.raises(UnknownDatasetError):
        service.submit("nope", [])


def test_perfect_submission_scores_one() -> None:
    service = _service()
    out = service.submit("_fake", _perfect(service))
    assert out["submission_id"] == 1
    assert out["scores"] == {"macro": {"f1_gold_only": 1.0}, "micro": {"f1_gold_only": 1.0}}
    assert out["submissions_used"] == 1
    assert out["submissions_remaining"] == 2
    assert out["dataset_submissions_used"] == 1
    assert out["dataset_submissions_remaining"] is None


def test_codes_are_conformed_and_none_is_und() -> None:
    service = _service()
    preds = [(i, {"eng": "en", "deu": "de"}.get(g or "", g)) for i, g in _perfect(service)]
    assert service.submit("_fake", preds)["scores"]["macro"]["f1_gold_only"] == 1.0

    all_none = [(i, None) for i, _ in _perfect(service)]
    assert service.submit("_fake", all_none)["scores"]["macro"]["f1_gold_only"] == 0.0


def test_per_language_metric_selection() -> None:
    service = _service()
    preds = [(i, "eng") for i, _ in _perfect(service, "_other")]
    scores = service.submit("_other", preds)["scores"]
    assert set(scores) == {"per_language"}
    assert scores["per_language"]["eng"] == {"f1": pytest.approx(4 / 7, abs=1e-4)}
    assert scores["per_language"]["deu"] == {"f1": 0.0}


def test_select_metrics_rounding_and_top_level() -> None:
    summary = {"n_samples": 3, "macro": {"f1_gold_only": 0.123456}}
    assert select_metrics(summary, ["n_samples", "macro.f1_gold_only"], 2) == {
        "n_samples": 3,
        "macro": {"f1_gold_only": 0.12},
    }
    assert select_metrics(summary, ["macro.f1_gold_only"], None) == {
        "macro": {"f1_gold_only": 0.123456}
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p[:-1], "missing ids"),
        (lambda p: [*p, p[0]], "duplicate ids"),
        (lambda p: [*p, (99, "eng")], "unknown ids"),
        (lambda p: [*p[:-1], (-1, "eng")], "unknown ids: -1"),
    ],
)
def test_invalid_submissions_are_rejected_and_not_counted(mutate: Any, match: str) -> None:
    service = _service()
    with pytest.raises(InvalidSubmissionError, match=match):
        service.submit("_fake", mutate(_perfect(service)))
    assert service.status()["submissions_used"] == 0


def test_error_lists_are_truncated() -> None:
    service = _service()
    with pytest.raises(InvalidSubmissionError, match=r"\(20 total\)"):
        service.submit("_fake", [*_perfect(service), *((100 + i, "eng") for i in range(20))])


def test_global_submission_limit() -> None:
    service = _service()
    for _ in range(3):
        service.submit("_fake", _perfect(service))
    with pytest.raises(SubmissionLimitError, match="3 in total"):
        service.submit("_fake", _perfect(service))
    status = service.status()
    assert status["submissions_used"] == 3
    assert status["submissions_remaining"] == 0


def test_per_dataset_submission_limit() -> None:
    service = _service()
    service.submit("_other", _perfect(service, "_other"))
    with pytest.raises(SubmissionLimitError, match="'_other'"):
        service.submit("_other", _perfect(service, "_other"))
    # The global budget still allows other datasets.
    service.submit("_fake", _perfect(service))
    datasets = {d["id"]: d for d in service.status()["datasets"]}
    assert datasets["_other"]["submissions_remaining"] == 0
    assert datasets["_fake"]["submissions_used"] == 1
    assert datasets["_fake"]["n_items"] == len(_FakeDataset.SAMPLES)
    assert datasets["_fake"]["metrics"] == ["macro.f1_gold_only", "micro.f1_gold_only"]


def test_unlimited_submissions() -> None:
    service = _service(_config(max_submissions=0))
    for _ in range(5):
        out = service.submit("_fake", _perfect(service))
    assert out["submissions_remaining"] is None


def test_history_has_no_operator_fields() -> None:
    service = _service()
    service.submit("_fake", _perfect(service))
    (entry,) = service.history()
    assert set(entry) == {"submission_id", "dataset_id", "timestamp", "scores"}


def test_dataset_limit_caps_served_items() -> None:
    cfg = parse_config({"datasets": [{"id": "_fake", "limit": 2}], "shuffle": False})
    service = ScoringService(cfg, datasets={"_fake": _FakeDataset()})
    assert service.inputs("_fake")["total"] == 2


def test_datasets_resolve_from_registry(fresh_registry: None) -> None:
    from commonlid.core.registry import register_dataset

    register_dataset(_FakeDataset)
    service = ScoringService(parse_config({"datasets": [{"id": "_fake"}]}))
    assert service.inputs("_fake")["total"] == len(_FakeDataset.SAMPLES)


def test_state_persists_counter_seed_and_archive(tmp_path: Path) -> None:
    cfg = _config(state_dir=str(tmp_path / "state"))
    first = _service(cfg)
    order = first.inputs("_fake")["items"]
    first.submit("_fake", _perfect(first))

    state = json.loads((tmp_path / "state" / "state.json").read_text())
    assert len(state["submissions"]) == 1
    run_dir = tmp_path / "state" / "submissions" / "_fake" / "submission-0001"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["macro"]["f1_gold_only"] == 1.0
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == len(_FakeDataset.SAMPLES)
    assert all(r["correct"] for r in rows if r["gold"] is not None)
    # Rows are in dataset order and keyed by the dataset idx, as for `commonlid run`.
    assert [r["idx"] for r in rows] == list(range(len(_FakeDataset.SAMPLES)))
    assert [r["gold"] for r in rows] == [s["iso639_3"] for s in _FakeDataset.SAMPLES]
    served_text = {it["id"]: it["text"] for it in order}
    assert all(GOLD_BY_TEXT[served_text[r["id"]]] == r["gold"] for r in rows)

    # A restart keeps the counter and the item ids.
    second = _service(cfg)
    assert second.inputs("_fake")["items"] == order
    assert second.status()["submissions_used"] == 1
    assert second.submit("_fake", _perfect(second))["submission_id"] == 2


def test_archive_reads_with_breakdown(tmp_path: Path) -> None:
    from commonlid.analysis import load_predictions, model_breakdown

    service = _service(_config(state_dir=str(tmp_path)))
    service.submit("_fake", _perfect(service))
    preds = load_predictions("_fake", ["submission-0001"], local_dir=tmp_path / "submissions")
    assert len(preds) == len(_FakeDataset.SAMPLES)
    assert model_breakdown(preds)["macro_f1"].tolist() == [1.0]


def test_changed_items_after_submissions_refuse_to_start(tmp_path: Path) -> None:
    state_dir = str(tmp_path / "state")
    first = _service(_config(state_dir=state_dir, shuffle_seed=1))
    first.submit("_fake", _perfect(first))
    # `_other` has no submissions yet, so only `_fake` blocks a new order.
    with pytest.raises(StateMismatchError, match="'_fake'"):
        _service(_config(state_dir=state_dir, shuffle_seed=2))


def test_changed_items_without_submissions_are_accepted(tmp_path: Path) -> None:
    state_dir = str(tmp_path / "state")
    _service(_config(state_dir=state_dir, shuffle_seed=1))
    second = _service(_config(state_dir=state_dir, shuffle_seed=2))
    reloaded = _service(_config(state_dir=state_dir, shuffle_seed=2))
    assert reloaded.inputs("_fake") == second.inputs("_fake")


def test_per_language_metrics_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _config(
        datasets=[{"id": "_fake", "metrics": ["per_language.f1", "per_language.gt_count"]}]
    )
    with caplog.at_level("WARNING", logger="commonlid.server.service"):
        ScoringService(cfg, datasets={"_fake": _FakeDataset()})
    assert "reveal which languages" in caplog.text
    assert "['gt_count'] reveal how many samples" in caplog.text


# ------------------------------------------------------------------------ HTTP


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_service()))


def test_http_flow(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    status = client.get("/status").json()
    assert status["max_submissions"] == 3
    assert [d["id"] for d in status["datasets"]] == ["_fake", "_other"]

    items = client.get("/datasets/_fake/inputs").json()["items"]
    preds = [{"id": it["id"], "pred": GOLD_BY_TEXT[it["text"]]} for it in items]
    resp = client.post("/datasets/_fake/submissions", json={"predictions": preds})
    assert resp.status_code == 200
    assert resp.json()["scores"]["macro"]["f1_gold_only"] == 1.0
    assert len(client.get("/submissions").json()) == 1

    page = client.get("/datasets/_fake/inputs", params={"offset": 1, "limit": 2}).json()
    assert page["items"] == items[1:3]


def test_http_errors(client: TestClient) -> None:
    assert client.get("/datasets/nope/inputs").status_code == 404
    bad = client.post("/datasets/_fake/submissions", json={"predictions": [{"id": 0, "pred": "x"}]})
    assert bad.status_code == 422
    assert "missing ids" in bad.json()["detail"]
    schema = client.post("/datasets/_fake/submissions", json={"preds": []})
    assert schema.status_code == 422

    items = client.get("/datasets/_other/inputs").json()["items"]
    body = {"predictions": [{"id": it["id"], "pred": None} for it in items]}
    assert client.post("/datasets/_other/submissions", json=body).status_code == 200
    limited = client.post("/datasets/_other/submissions", json=body)
    assert limited.status_code == 429
    assert "limit" in limited.json()["detail"]


def test_http_bearer_token() -> None:
    client = TestClient(create_app(_service(_config(api_token="s3cret"))))
    assert client.get("/health").status_code == 200
    for path in ("/status", "/submissions", "/datasets/_fake/inputs"):
        resp = client.get(path)
        assert resp.status_code == 401, path
        assert resp.headers["www-authenticate"] == "Bearer"
    assert client.post("/datasets/_fake/submissions", json={"predictions": []}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.get("/status", headers=wrong).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert client.get("/status", headers=ok).status_code == 200


# ------------------------------------------------------------------------- CLI


def test_cli_list_server_metrics() -> None:
    result = CliRunner().invoke(cli_app, ["list-server-metrics"])
    assert result.exit_code == 0
    assert "macro.f1_gold_only" in result.output.splitlines()


def test_cli_serve_rejects_invalid_config(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('max_submissions = 1\n[[datasets]]\nid = "x"\nmetrics = ["nope"]\n')
    result = CliRunner().invoke(cli_app, ["serve", "--config", str(path)])
    assert result.exit_code == 2
    assert "unknown metrics" in result.output


def test_cli_serve_starts_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_registry: None
) -> None:
    from commonlid.core.registry import register_dataset

    register_dataset(_FakeDataset)
    path = tmp_path / "server.toml"
    path.write_text('port = 8123\n[[datasets]]\nid = "_fake"\n')
    calls: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = CliRunner().invoke(cli_app, ["serve", "--config", str(path), "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 8123
    assert TestClient(calls["app"]).get("/health").status_code == 200
