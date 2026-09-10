"""Unit tests for the Cloud Translation Advanced (v3) model wrapper.

Never touches the network: every test either swaps ``_client`` for a fake or
monkeypatches the client library out of ``sys.modules``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest

from commonlid.core.lid_model import LIDPrediction
from commonlid.models.google_translate_v3 import (
    FALLBACK_PROJECT_ID_ENV,
    PROJECT_ID_ENV,
    GoogleTranslateV3Model,
)


@dataclass
class _FakeLanguage:
    language_code: str
    confidence: float | None


@dataclass
class _FakeResponse:
    languages: list[_FakeLanguage]


class _FakeClient:
    """Stands in for ``google.cloud.translate.TranslationServiceClient``."""

    def __init__(self, responses: list[Any] | None = None, languages: Any = None) -> None:
        self.calls: list[str] = []
        self._responses = responses
        self._languages = languages if languages is not None else []

    def detect_language(self, *, parent: str, content: str, mime_type: str) -> Any:
        self.calls.append(content)
        if self._responses is not None:
            return self._responses.pop(0)
        return _FakeResponse([_FakeLanguage("en", 1.0)])

    def get_supported_languages(self, *, parent: str) -> Any:
        return _FakeResponse(self._languages)


def _loaded_model(client: _FakeClient, **kwargs: Any) -> GoogleTranslateV3Model:
    """A model wired to a fake client, skipping the real ``load()``."""
    model = GoogleTranslateV3Model(**kwargs)
    model._client = client
    model._parent = "projects/p/locations/global"
    model._loaded = True
    return model


def _patch_auth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adc_project: str | None,
    quota_project: str | None,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class _Creds:
        quota_project_id = quota_project

    creds = _Creds()
    monkeypatch.setattr("google.auth.default", lambda: (creds, adc_project))

    def fake_client(*, credentials: Any) -> _FakeClient:
        seen["credentials"] = credentials
        return _FakeClient()

    monkeypatch.setattr("google.cloud.translate.TranslationServiceClient", fake_client)
    monkeypatch.delenv(PROJECT_ID_ENV, raising=False)
    monkeypatch.delenv(FALLBACK_PROJECT_ID_ENV, raising=False)
    return seen


def test_load_raises_helpful_error_without_client_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "google.cloud", None)
    with pytest.raises(ImportError, match=r"commonlid\[google-translate\]"):
        GoogleTranslateV3Model().load()


def test_load_raises_when_no_project_can_be_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auth(monkeypatch, adc_project=None, quota_project=None)
    with pytest.raises(RuntimeError, match=PROJECT_ID_ENV):
        GoogleTranslateV3Model().load()


@pytest.mark.parametrize(
    ("explicit", "env", "fallback_env", "adc", "quota", "expected"),
    [
        ("arg", "env", "fallback", "adc", "quota", "arg"),
        (None, "env", "fallback", "adc", "quota", "env"),
        (None, None, "fallback", "adc", "quota", "fallback"),
        (None, None, None, "adc", "quota", "adc"),
        # User ADC reports no project of its own, only a quota project.
        (None, None, None, None, "quota", "quota"),
    ],
)
def test_project_id_resolution_order(
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    env: str | None,
    fallback_env: str | None,
    adc: str | None,
    quota: str | None,
    expected: str,
) -> None:
    _patch_auth(monkeypatch, adc_project=adc, quota_project=quota)
    if env:
        monkeypatch.setenv(PROJECT_ID_ENV, env)
    if fallback_env:
        monkeypatch.setenv(FALLBACK_PROJECT_ID_ENV, fallback_env)

    model = GoogleTranslateV3Model(project_id=explicit)
    model.load()
    assert model._parent == f"projects/{expected}/locations/global"


def test_load_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_auth(monkeypatch, adc_project="p", quota_project=None)
    model = GoogleTranslateV3Model()
    model.load()
    client = model._client
    model.load()
    assert model._client is client


def test_predict_conforms_and_keeps_confidence() -> None:
    client = _FakeClient(
        responses=[
            _FakeResponse([_FakeLanguage("en", 0.98)]),
            # v3 emits BCP-47; the region suffix is dropped before conformance.
            _FakeResponse([_FakeLanguage("zh-CN", 1.0)]),
            # `tl` is what v3 returns for Tagalog, where v2 returns `fil`.
            _FakeResponse([_FakeLanguage("tl", 0.9)]),
            _FakeResponse([]),
        ]
    )
    model = _loaded_model(client, max_workers=1)

    assert model.predict_scored(["english", "chinese", "tagalog", "nothing"]) == [
        LIDPrediction("eng", 0.98),
        LIDPrediction("zho", 1.0),
        LIDPrediction("tgl", 0.9),
        LIDPrediction(None, None),
    ]


def test_confidence_survives_an_unmappable_code() -> None:
    client = _FakeClient(responses=[_FakeResponse([_FakeLanguage("zzzz", 0.4)])])
    model = _loaded_model(client, max_workers=1)
    assert model.predict_scored(["gibberish"]) == [LIDPrediction(None, 0.4)]


def test_blank_text_never_reaches_the_api() -> None:
    client = _FakeClient(responses=[_FakeResponse([_FakeLanguage("en", 1.0)])])
    model = _loaded_model(client, max_workers=1)

    # "123" survives .strip() but the OpenLID normer strips digits to nothing.
    assert model.predict(["   ", "123", "english text"]) == [None, None, "eng"]
    assert client.calls == ["english text"]


def test_all_blank_batch_makes_no_call() -> None:
    client = _FakeClient()
    model = _loaded_model(client)
    assert model.predict(["", "   "]) == [None, None]
    assert client.calls == []


def test_concurrent_calls_keep_input_order() -> None:
    class _EchoClient(_FakeClient):
        def detect_language(self, *, parent: str, content: str, mime_type: str) -> Any:
            self.calls.append(content)
            # Echo the code encoded in the text so a shuffled completion order
            # would show up as a shuffled result.
            return _FakeResponse([_FakeLanguage(content.split()[-1], 1.0)])

    model = _loaded_model(_EchoClient(), max_workers=4)
    texts = ["text en", "text de", "text fr", "text ru", "text es", "text it"]
    assert model.predict(texts) == ["eng", "deu", "fra", "rus", "spa", "ita"]


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core import exceptions as gexc

    monkeypatch.setattr(GoogleTranslateV3Model, "_BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[int] = []

    class _FlakyClient(_FakeClient):
        def detect_language(self, *, parent: str, content: str, mime_type: str) -> Any:
            attempts.append(1)
            if len(attempts) < 3:
                raise gexc.ResourceExhausted("slow down")
            return _FakeResponse([_FakeLanguage("en", 1.0)])

    model = _loaded_model(_FlakyClient(), max_workers=1)
    assert model.predict(["english text"]) == ["eng"]
    assert len(attempts) == 3


def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core import exceptions as gexc

    monkeypatch.setattr(GoogleTranslateV3Model, "_BACKOFF_BASE_SECONDS", 0.0)

    class _DeadClient(_FakeClient):
        def detect_language(self, *, parent: str, content: str, mime_type: str) -> Any:
            raise gexc.ServiceUnavailable("down")

    # A dead endpoint must surface, not be scored as an abstention.
    with pytest.raises(gexc.ServiceUnavailable):
        _loaded_model(_DeadClient(), max_workers=1).predict(["english text"])


def test_non_retryable_error_propagates() -> None:
    from google.api_core import exceptions as gexc

    class _ForbiddenClient(_FakeClient):
        def detect_language(self, *, parent: str, content: str, mime_type: str) -> Any:
            raise gexc.PermissionDenied("no access")

    with pytest.raises(gexc.PermissionDenied):
        _loaded_model(_ForbiddenClient(), max_workers=1).predict(["english text"])


def test_oversized_text_is_clipped_to_the_api_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GoogleTranslateV3Model, "_MAX_CHARS_PER_TEXT", 10)
    client = _FakeClient(responses=[_FakeResponse([_FakeLanguage("en", 1.0)])])
    model = _loaded_model(client, max_workers=1)
    model.predict(["a" * 500])
    assert client.calls == ["a" * 10]


def test_discover_supported_languages() -> None:
    client = _FakeClient(
        languages=[
            _FakeLanguage("en", None),
            _FakeLanguage("zh-CN", None),
            _FakeLanguage("zh-TW", None),
            _FakeLanguage("iw", None),
            _FakeLanguage("zzz", None),
        ]
    )
    supported = _loaded_model(client).discover_supported_languages()
    assert isinstance(supported, frozenset)
    # zh-CN and zh-TW collapse to one code; "zzz" drops out.
    assert supported == frozenset({"eng", "zho", "heb"})
