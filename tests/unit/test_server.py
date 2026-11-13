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
from commonlid.server.derive import normalize_text
from commonlid.server.service import (
    DatasetBuildError,
    InvalidSubmissionError,
    LabelOverlapError,
    ScoringService,
    StateMismatchError,
    SubmissionClosedError,
    SubmissionLimitError,
    UnknownDatasetError,
    check_prediction,
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


class _DevDataset(_FakeDataset):
    """Labelled development data that shares no texts with the others."""

    dataset_id = "_dev"
    SAMPLES: ClassVar[list[dict[str, str | None]]] = [
        {"text": "thank you", "iso639_3": "eng"},
        {"text": "danke", "iso639_3": "deu"},
        {"text": "merci", "iso639_3": "fra"},
    ]


class _MacroDataset(_FakeDataset):
    """Individual-language gold labels whose macrolanguage codes models often predict."""

    dataset_id = "_macro"
    SAMPLES: ClassVar[list[dict[str, str | None]]] = [
        {"text": "labdien", "iso639_3": "lvs"},
        {"text": "ni hao", "iso639_3": "cmn"},
        {"text": "hello there", "iso639_3": "eng"},
    ]


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
        ({"datasets": [{"id": "x", "macro_match": "loose"}]}, "macro_match must be one of"),
        ({"datasets": [{"id": "x", "expose_labels": "yes"}]}, "expose_labels"),
        ({"datasets": [{"id": "x", "description": " "}]}, "description"),
        ({"datasets": [{"id": "x", "source": ""}]}, "source must be"),
        ({"datasets": [{"id": "x", "exclude": [1]}]}, "exclude must be a list"),
        ({"datasets": [{"id": "x", "exclude": ["x"]}]}, "cannot exclude itself"),
        ({"datasets": [{"id": "x", "min_chars": -1}]}, "min_chars must be a non-negative"),
        ({"datasets": [{"id": "x", "min_chars": 9, "max_chars": 5}]}, "not be larger"),
        ({"datasets": [{"id": "x", "sample": 100}]}, "sample must be a table"),
        ({"datasets": [{"id": "x", "sample": {"max_size": 10}}]}, "needs seed"),
        ({"datasets": [{"id": "x", "sample": {"seed": 1}}]}, "needs max_size"),
        ({"datasets": [{"id": "x", "sample": {"max_size": 0, "seed": 1}}]}, "positive"),
        ({"datasets": [{"id": "x", "sample": {"max_size": 5, "seed": 1, "n": 2}}]}, "unknown"),
        (
            {"datasets": [{"id": "a", "exclude": ["b"]}, {"id": "b", "exclude": ["a"]}]},
            "cycle: a -> b -> a",
        ),
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


def test_parse_derived_dataset() -> None:
    cfg = parse_config({
        "datasets": [
            {"id": "train", "source": "pool", "exclude": ["test", "dev"]},
            {
                "id": "dev",
                "source": "pool",
                "exclude": "test",
                "min_chars": 20,
                "sample": {"max_size": 100, "seed": 7},
            },
            {"id": "test"},
        ]
    })
    train, dev, test = cfg.datasets
    assert (train.source_id, dev.source_id, test.source_id) == ("pool", "pool", "test")
    assert dev.exclude == ("test",)
    assert dev.sample is not None
    assert (dev.sample.max_size, dev.sample.seed, dev.sample.min_per_language) == (100, 7, 5)
    assert test.sample is None
    assert test.exclude == ()
    # Each dataset is built after the configured datasets it excludes.
    assert [d.id for d in cfg.build_order()] == ["test", "dev", "train"]


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
    assert set(entry) == {"submission_id", "dataset_id", "timestamp", "final", "scores"}
    assert entry["final"] is False


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


# ------------------------------------------------------------ code validation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, (None, False)),
        ("eng", ("eng", False)),
        ("und", ("und", False)),
        ("zho", ("zho", False)),
        ("en", ("eng", True)),
        ("ger", ("deu", True)),
        ("ENG", ("eng", True)),
        (" eng ", ("eng", True)),
        ("iw", ("heb", True)),
    ],
)
def test_check_prediction_accepts_codes(raw: str | None, expected: tuple[str, bool]) -> None:
    assert check_prediction(raw) == expected


@pytest.mark.parametrize(
    "raw", ["English", "english", "en-US", "eng_Latn", "zh-Hans", "xyz", "mo", "", "e"]
)
def test_check_prediction_rejects_non_codes(raw: str) -> None:
    with pytest.raises(ValueError, match="language code"):
        check_prediction(raw)


def test_invalid_codes_are_rejected_listed_and_not_counted() -> None:
    service = _service()
    preds = _perfect(service)
    preds = [(i, "English" if g == "eng" else g) for i, g in preds]
    preds[-1] = (preds[-1][0], "en-US")
    with pytest.raises(InvalidSubmissionError) as excinfo:
        service.submit("_fake", preds)
    err = excinfo.value
    assert "not ISO 639-3 codes" in str(err)
    listed = {row["pred"]: row for row in err.extra["invalid_predictions"]}
    assert listed["English"]["count"] >= 1
    assert listed["English"]["reason"] == "not a language code"
    assert listed["en-US"]["example_ids"] == [preds[-1][0]]
    assert "id_errors" not in err.extra
    assert service.status()["submissions_used"] == 0


def test_invalid_codes_and_id_errors_are_reported_together() -> None:
    service = _service()
    preds = [(i, "xyz") for i, _ in _perfect(service)][1:]
    with pytest.raises(InvalidSubmissionError) as excinfo:
        service.submit("_fake", preds)
    assert "missing ids" in str(excinfo.value)
    assert excinfo.value.extra["invalid_predictions"][0]["reason"] == (
        "unknown or retired language code"
    )
    assert len(excinfo.value.extra["id_errors"]["missing"]) == 1


def test_converted_codes_are_scored_with_a_warning() -> None:
    service = _service()
    preds = [(i, {"eng": "en", "deu": "GER"}.get(g or "", g)) for i, g in _perfect(service)]
    out = service.submit("_fake", preds)
    assert out["scores"]["macro"]["f1_gold_only"] == 1.0
    (warning,) = out["warnings"]
    assert "3 prediction(s) were converted" in warning
    assert "'en' -> 'eng' (2)" in warning
    assert "'GER' -> 'deu' (1)" in warning
    assert service.submit("_fake", _perfect(service))["warnings"] == []


# ------------------------------------------------------------------- final


def test_final_submission_closes_the_dataset(tmp_path: Path) -> None:
    cfg = _config(state_dir=str(tmp_path))
    service = _service(cfg)
    service.submit("_fake", [(i, "eng") for i, _ in _perfect(service)])
    out = service.submit("_fake", _perfect(service), final=True)
    assert out["final"] is True
    with pytest.raises(SubmissionClosedError, match="submission 2 was marked final"):
        service.submit("_fake", _perfect(service))
    # Other datasets stay open, and a rejected submission is not counted.
    service.submit("_other", _perfect(service, "_other"))
    assert service.status()["submissions_used"] == 3
    datasets = {d["id"]: d for d in service.status()["datasets"]}
    assert datasets["_fake"]["final_submission_id"] == 2
    assert datasets["_other"]["final_submission_id"] is None
    assert [e["final"] for e in service.history()] == [False, True, False]
    assert "Closed: submission 2 was marked final" in service.instructions("_fake")

    restarted = _service(cfg)
    with pytest.raises(SubmissionClosedError):
        restarted.submit("_fake", _perfect(restarted))


# ------------------------------------------------------------- instructions


def test_inputs_carry_instructions() -> None:
    service = _service()
    page = service.inputs("_fake", offset=0, limit=2)
    assert page["labels_exposed"] is False
    text = page["instructions"]
    n = len(_FakeDataset.SAMPLES)
    assert f"each of the {n} texts in dataset '_fake'" in text
    assert "POST /datasets/_fake/submissions" in text
    assert f"every id from 0 to {n - 1} exactly once" in text
    assert "ISO 639-3 code" in text
    assert '"final": true' in text
    assert "Codes must match the gold label exactly" in text
    assert "Returned scores: macro.f1_gold_only, micro.f1_gold_only." in text
    assert "3 submission(s) left overall" not in text  # no per-dataset cap here
    assert "unlimited submission(s) left for this dataset, 3 left overall" in text
    assert "Gold labels are not available" in text

    service.submit("_fake", _perfect(service))
    assert "2 left overall" in service.inputs("_fake")["instructions"]


def test_instructions_show_description_and_matching_mode() -> None:
    cfg = _config(
        datasets=[
            {"id": "_macro", "macro_match": "macro", "description": "Web text from 2024."},
        ]
    )
    service = ScoringService(cfg, datasets={"_macro": _MacroDataset()})
    text = service.instructions("_macro")
    assert "Web text from 2024." in text
    assert "An individual language and its macrolanguage count as the same" in text


# ------------------------------------------------------------ macro matching


@pytest.mark.parametrize(("mode", "expected"), [("exact", 1 / 3), ("macro", 1.0)])
def test_macro_match_scoring(tmp_path: Path, mode: str, expected: float) -> None:
    cfg = _config(
        datasets=[{"id": "_macro", "macro_match": mode}], state_dir=str(tmp_path), shuffle=False
    )
    service = ScoringService(cfg, datasets={"_macro": _MacroDataset()})
    out = service.submit("_macro", [(0, "lav"), (1, "zho"), (2, "eng")])
    assert out["scores"]["macro"]["f1_gold_only"] == pytest.approx(expected, abs=1e-4)
    rows = tmp_path / "submissions" / "_macro" / "submission-0001" / "predictions.jsonl"
    correct = [json.loads(line)["correct"] for line in rows.read_text().splitlines()]
    assert correct == ([False, False, True] if mode == "exact" else [True, True, True])


# -------------------------------------------------------------- dev datasets


def _dev_service(**overrides: Any) -> ScoringService:
    cfg = _config(
        datasets=[
            {"id": "_fake"},
            {"id": "_dev", "expose_labels": True, "metrics": ["per_language.f1"]},
        ],
        max_submissions=1,
        per_language_metrics=False,
        **overrides,
    )
    return ScoringService(cfg, datasets={"_fake": _FakeDataset(), "_dev": _DevDataset()})


def test_dev_dataset_serves_labels() -> None:
    service = _dev_service()
    page = service.inputs("_dev")
    assert page["labels_exposed"] is True
    gold = {s["text"]: s["iso639_3"] for s in _DevDataset.SAMPLES}
    assert all(it["label"] == gold[it["text"]] for it in page["items"])
    assert "development dataset" in page["instructions"]
    assert "They do not use the overall budget" in page["instructions"]
    # Evaluation datasets stay label-free.
    assert all(set(it) == {"id", "text"} for it in service.inputs("_fake")["items"])


def test_dev_dataset_skips_global_budget_and_allows_per_language() -> None:
    service = _dev_service()
    dev_preds = [(it["id"], it["label"]) for it in service.inputs("_dev")["items"]]
    for _ in range(3):
        out = service.submit("_dev", dev_preds)
    assert out["scores"]["per_language"]["eng"] == {"f1": 1.0}
    assert out["submissions_remaining"] == 1
    service.submit("_fake", _perfect(service))
    with pytest.raises(SubmissionLimitError, match="in total"):
        service.submit("_fake", _perfect(service))
    datasets = {d["id"]: d for d in service.status()["datasets"]}
    assert datasets["_dev"]["uses_global_budget"] is False
    assert datasets["_dev"]["submissions_used"] == 3
    # Submission ids stay unique across datasets.
    assert [e["submission_id"] for e in service.history()] == [1, 2, 3, 4]


def test_dev_dataset_sharing_texts_with_an_evaluation_dataset_is_refused() -> None:
    cfg = _config(datasets=[{"id": "_fake"}, {"id": "_other", "expose_labels": True}])
    with pytest.raises(LabelOverlapError, match="6 of the 6 texts of '_fake' are in '_other'"):
        ScoringService(cfg, datasets={"_fake": _FakeDataset(), "_other": _OtherDataset()})


# ------------------------------------------------------------ derived datasets


class _PoolDataset(_FakeDataset):
    """A larger labelled pool: 20 texts each in three languages, two short texts
    and whitespace variants of two `_fake` texts."""

    dataset_id = "_pool"
    SAMPLES: ClassVar[list[dict[str, str | None]]] = [
        *(
            {"text": f"{lang} sentence number {i}", "iso639_3": lang}
            for lang in ("eng", "deu", "fra")
            for i in range(20)
        ),
        {"text": "ok", "iso639_3": "eng"},
        {"text": "ja", "iso639_3": "deu"},
        {"text": " hello   world ", "iso639_3": "eng"},
        {"text": "hallo\nwelt", "iso639_3": "deu"},
    ]


def _derived_service(datasets: list[dict[str, Any]], **overrides: Any) -> ScoringService:
    cfg = parse_config({"datasets": datasets, **overrides})
    return ScoringService(cfg, datasets={"_fake": _FakeDataset(), "_pool": _PoolDataset()})


def _texts(service: ScoringService, dataset_id: str) -> set[str]:
    return {it["text"] for it in service.inputs(dataset_id)["items"]}


def test_derived_dataset_excludes_texts_after_normalisation() -> None:
    service = _derived_service([
        {"id": "_fake"},
        {"id": "dev", "source": "_pool", "exclude": ["_fake"], "expose_labels": True},
    ])
    dev = _texts(service, "dev")
    assert len(dev) == len(_PoolDataset.SAMPLES) - 2
    assert not {normalize_text(t) for t in dev} & set(GOLD_BY_TEXT)
    # Labels come from the source dataset.
    gold = {s["text"]: s["iso639_3"] for s in _PoolDataset.SAMPLES}
    assert all(it["label"] == gold[it["text"]] for it in service.inputs("dev")["items"])


def test_excluding_a_registered_dataset_removes_all_of_it() -> None:
    # `_fake` is served with limit = 1, but `exclude` drops every text of it.
    service = _derived_service([
        {"id": "_fake", "limit": 1},
        {"id": "dev", "source": "_pool", "exclude": ["_fake"], "expose_labels": True},
    ])
    assert service.inputs("dev")["total"] == len(_PoolDataset.SAMPLES) - 2


def test_overlap_check_compares_normalised_texts() -> None:
    with pytest.raises(LabelOverlapError, match="2 of the 6 texts of '_fake' are in 'dev'"):
        _derived_service([{"id": "_fake"}, {"id": "dev", "source": "_pool", "expose_labels": True}])


def test_length_filter() -> None:
    service = _derived_service([
        {"id": "long", "source": "_pool", "min_chars": 3},
        {"id": "short", "source": "_pool", "max_chars": 11},
    ])
    assert not {"ok", "ja"} & _texts(service, "long")
    # Whitespace runs count as one character: " hello   world " has 11.
    assert _texts(service, "short") == {"ok", "ja", " hello   world ", "hallo\nwelt"}


def test_sample_is_stratified_and_reproducible() -> None:
    dev = {"id": "dev", "source": "_pool", "min_chars": 3, "sample": {"max_size": 15, "seed": 1}}
    first = _derived_service([dev], shuffle_seed=1)
    again = _derived_service([dev], shuffle_seed=2)
    other_seed = _derived_service([{**dev, "sample": {"max_size": 15, "seed": 2}}])

    texts = _texts(first, "dev")
    # 5 per language plus 15 spread proportionally over the two remaining
    # deu/eng/fra pools (eng and deu also keep one whitespace variant each).
    assert len(texts) == 30
    langs = [t.split()[0] for t in texts if "sentence" in t]
    assert all(langs.count(lang) >= 5 for lang in ("eng", "deu", "fra"))
    # Same config: same items, whatever the shuffle seed; another seed: other items.
    assert _texts(again, "dev") == texts
    assert _texts(other_seed, "dev") != texts
    assert first._datasets["dev"].content_hash == again._datasets["dev"].content_hash
    assert first.inputs("dev")["items"] != again.inputs("dev")["items"]


def test_disjoint_splits_by_excluding_configured_datasets() -> None:
    service = _derived_service([
        {
            "id": "train",
            "source": "_pool",
            "exclude": ["_fake", "dev"],
            "expose_labels": True,
        },
        {
            "id": "dev",
            "source": "_pool",
            "exclude": ["_fake"],
            "sample": {"max_size": 6, "min_per_language": 2, "seed": 3},
            "expose_labels": True,
        },
        {"id": "_fake"},
    ])
    train, dev = _texts(service, "train"), _texts(service, "dev")
    assert not train & dev
    assert len(train | dev) == len(_PoolDataset.SAMPLES) - 2
    # Status lists the datasets in config order.
    assert [d["id"] for d in service.status()["datasets"]] == ["train", "dev", "_fake"]


def test_derived_archive_is_keyed_by_source_position(tmp_path: Path) -> None:
    service = _derived_service(
        [{"id": "long", "source": "_pool", "min_chars": 15, "limit": 5}],
        state_dir=str(tmp_path),
    )
    gold = {s["text"]: s["iso639_3"] for s in _PoolDataset.SAMPLES}
    items = service.inputs("long")["items"]
    service.submit("long", [(it["id"], gold[it["text"]]) for it in items])
    rows = tmp_path / "submissions" / "long" / "submission-0001" / "predictions.jsonl"
    idx = [json.loads(line)["idx"] for line in rows.read_text().splitlines()]
    assert idx == [0, 1, 2, 3, 4]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["datasets"]["long"]["content_hash"] == service._datasets["long"].content_hash


@pytest.mark.parametrize(
    ("dataset", "match"),
    [
        ({"id": "x", "source": "_nope"}, "unknown source '_nope'"),
        ({"id": "_nope"}, "unknown dataset '_nope'"),
        ({"id": "x", "source": "_pool", "exclude": ["_nope"]}, "excludes '_nope'"),
        ({"id": "x", "source": "_pool", "min_chars": 1000}, "is empty after"),
        (
            {"id": "x", "source": "_pool", "sample": {"max_size": 60, "seed": 1}},
            "only 49 of the 64 remaining texts",
        ),
    ],
)
def test_derived_dataset_errors(dataset: dict[str, Any], match: str) -> None:
    with pytest.raises(DatasetBuildError, match=match):
        _derived_service([dataset])


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
    assert bad.json()["invalid_predictions"][0]["pred"] == "x"
    schema = client.post("/datasets/_fake/submissions", json={"preds": []})
    assert schema.status_code == 422

    items = client.get("/datasets/_other/inputs").json()["items"]
    body = {"predictions": [{"id": it["id"], "pred": None} for it in items]}
    assert client.post("/datasets/_other/submissions", json=body).status_code == 200
    limited = client.post("/datasets/_other/submissions", json=body)
    assert limited.status_code == 429
    assert "limit" in limited.json()["detail"]


def test_http_final_submission(client: TestClient) -> None:
    items = client.get("/datasets/_fake/inputs").json()
    assert "ISO 639-3" in items["instructions"]
    body = {
        "predictions": [
            {"id": it["id"], "pred": GOLD_BY_TEXT[it["text"]]} for it in items["items"]
        ],
        "final": True,
    }
    assert client.post("/datasets/_fake/submissions", json=body).json()["final"] is True
    closed = client.post("/datasets/_fake/submissions", json=body)
    assert closed.status_code == 409
    assert "marked final" in closed.json()["detail"]


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


def test_cli_serve_refuses_label_overlap(tmp_path: Path, fresh_registry: None) -> None:
    from commonlid.core.registry import register_dataset

    register_dataset(_FakeDataset)
    register_dataset(_OtherDataset)
    path = tmp_path / "server.toml"
    path.write_text(
        '[[datasets]]\nid = "_fake"\n\n[[datasets]]\nid = "_other"\nexpose_labels = true\n'
    )
    result = CliRunner().invoke(cli_app, ["serve", "--config", str(path)])
    assert result.exit_code == 2
    assert "would reveal gold labels" in result.output


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
