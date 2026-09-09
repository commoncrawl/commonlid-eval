"""Unit tests for the Google Cloud Translation model wrapper.

Never touches the network: every test either swaps ``_client`` for a fake or
monkeypatches the client library out of ``sys.modules``.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from commonlid.core.lid_model import LIDPrediction
from commonlid.models.google_translate import API_KEY_ENV, GoogleTranslateModel


class _FakeClient:
    """Stands in for ``google.cloud.translate_v2.Client``."""

    def __init__(self, responses: list[Any] | None = None, languages: Any = None) -> None:
        self.calls: list[list[str]] = []
        self._responses = responses
        self._languages = languages if languages is not None else []

    def detect_language(self, values: list[str]) -> Any:
        self.calls.append(list(values))
        if self._responses is not None:
            return self._responses.pop(0)
        return [{"language": "en", "confidence": 1} for _ in values]

    def get_languages(self) -> Any:
        return self._languages


def _loaded_model(client: _FakeClient, **kwargs: Any) -> GoogleTranslateModel:
    """A model wired to a fake client, skipping the real ``load()``."""
    model = GoogleTranslateModel(**kwargs)
    model._client = client
    model._loaded = True
    return model


def test_load_raises_helpful_error_without_client_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    monkeypatch.setitem(sys.modules, "google.cloud", None)
    with pytest.raises(ImportError, match=r"commonlid\[google-translate\]"):
        GoogleTranslateModel().load()


def test_load_raises_when_api_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        GoogleTranslateModel().load()


def test_load_passes_api_key_credentials_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_credentials(token: str) -> str:
        seen["token"] = token
        return f"creds:{token}"

    def fake_client(credentials: str) -> _FakeClient:
        seen["credentials"] = credentials
        return _FakeClient()

    monkeypatch.setattr("google.auth.api_key.Credentials", fake_credentials)
    monkeypatch.setattr("google.cloud.translate_v2.Client", fake_client)

    model = GoogleTranslateModel(api_key="explicit-key")
    model.load()
    client = model._client
    model.load()

    assert seen == {"token": "explicit-key", "credentials": "creds:explicit-key"}
    assert model._client is client


def test_predict_conforms_bcp47_and_legacy_codes() -> None:
    client = _FakeClient(
        responses=[
            [
                {"language": "en"},
                {"language": "zh-CN"},
                {"language": "iw"},
                {"language": "und"},
                {"confidence": 1},
            ]
        ]
    )
    model = _loaded_model(client)
    texts = ["english text", "chinese text", "hebrew text", "gibberish", "unknown"]

    assert model.predict(texts) == ["eng", "zho", "heb", None, None]
    assert len(client.calls) == 1


def test_confidence_is_preserved() -> None:
    client = _FakeClient(
        responses=[
            [
                {"language": "en", "confidence": 0.98},
                {"language": "zh-CN", "confidence": 1},
                {"language": "de"},
            ]
        ]
    )
    model = _loaded_model(client)

    assert model.predict_scored(["english", "chinese", "german"]) == [
        LIDPrediction("eng", 0.98),
        # An integer confidence is normalised to float.
        LIDPrediction("zho", 1.0),
        # `confidence` is documented as optional; its absence is not an error.
        LIDPrediction("deu", None),
    ]


def test_confidence_survives_an_unmappable_code() -> None:
    client = _FakeClient(responses=[[{"language": "und", "confidence": 0.4}]])
    model = _loaded_model(client)

    assert model.predict_scored(["gibberish"]) == [LIDPrediction(None, 0.4)]


def test_blank_text_never_reaches_the_api() -> None:
    client = _FakeClient(responses=[[{"language": "en"}]])
    model = _loaded_model(client)

    # "123" survives .strip() but the OpenLID normer strips digits to nothing.
    assert model.predict(["   ", "123", "english text"]) == [None, None, "eng"]
    assert client.calls == [["english text"]]


def test_all_blank_batch_makes_no_call() -> None:
    client = _FakeClient()
    model = _loaded_model(client)

    assert model.predict(["", "   "]) == [None, None]
    assert client.calls == []


def test_single_element_dict_response_is_coerced_to_a_list() -> None:
    client = _FakeClient(responses=[{"language": "fr"}])
    model = _loaded_model(client)

    assert model.predict(["texte francais"]) == ["fra"]


def test_batch_is_chunked_and_order_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GoogleTranslateModel, "_MAX_SEGMENTS_PER_REQUEST", 2)
    codes = ["en", "de", "fr", "ru", "es"]
    client = _FakeClient(
        responses=[
            [{"language": "en"}, {"language": "de"}],
            [{"language": "fr"}, {"language": "ru"}],
            [{"language": "es"}],
        ]
    )
    model = _loaded_model(client, max_workers=1)

    assert model.predict([f"text {c}" for c in codes]) == ["eng", "deu", "fra", "rus", "spa"]
    assert [len(call) for call in client.calls] == [2, 2, 1]


def test_chunking_respects_the_character_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GoogleTranslateModel, "_MAX_CHARS_PER_REQUEST", 20)
    client = _FakeClient(responses=[[{"language": "en"}], [{"language": "de"}]])
    model = _loaded_model(client, max_workers=1)

    assert model.predict(["a" * 15, "b" * 15]) == ["eng", "deu"]
    assert [len(call) for call in client.calls] == [1, 1]


def test_oversized_text_is_clipped_to_the_api_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GoogleTranslateModel, "_MAX_CHARS_PER_TEXT", 10)
    client = _FakeClient(responses=[[{"language": "en"}]])
    model = _loaded_model(client, max_workers=1)

    model.predict(["a" * 500])
    assert client.calls == [["a" * 10]]


def test_concurrent_chunks_keep_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GoogleTranslateModel, "_MAX_SEGMENTS_PER_REQUEST", 1)

    class _OutOfOrderClient(_FakeClient):
        def detect_language(self, values: list[str]) -> Any:
            self.calls.append(list(values))
            # Echo the language encoded in the text so a shuffled completion
            # order would show up as a shuffled result.
            return [{"language": v.split()[-1]} for v in values]

    client = _OutOfOrderClient()
    model = _loaded_model(client, max_workers=4)
    texts = ["text en", "text de", "text fr", "text ru", "text es", "text it"]

    assert model.predict(texts) == ["eng", "deu", "fra", "rus", "spa", "ita"]


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core import exceptions as gexc

    monkeypatch.setattr(GoogleTranslateModel, "_BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[int] = []

    class _FlakyClient(_FakeClient):
        def detect_language(self, values: list[str]) -> Any:
            attempts.append(1)
            if len(attempts) < 3:
                raise gexc.TooManyRequests("slow down")
            return [{"language": "en"} for _ in values]

    model = _loaded_model(_FlakyClient())

    assert model.predict(["english text"]) == ["eng"]
    assert len(attempts) == 3


def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core import exceptions as gexc

    monkeypatch.setattr(GoogleTranslateModel, "_BACKOFF_BASE_SECONDS", 0.0)

    class _DeadClient(_FakeClient):
        def detect_language(self, values: list[str]) -> Any:
            raise gexc.ServiceUnavailable("down")

    model = _loaded_model(_DeadClient())

    # A dead endpoint must surface, not be scored as an abstention.
    with pytest.raises(gexc.ServiceUnavailable):
        model.predict(["english text"])


def test_non_retryable_error_propagates() -> None:
    from google.api_core import exceptions as gexc

    class _ForbiddenClient(_FakeClient):
        def detect_language(self, values: list[str]) -> Any:
            raise gexc.Forbidden("bad key")

    with pytest.raises(gexc.Forbidden):
        _loaded_model(_ForbiddenClient()).predict(["english text"])


def test_discover_supported_languages() -> None:
    client = _FakeClient(
        languages=[
            {"language": "en"},
            {"language": "zh-CN"},
            {"language": "zh-TW"},
            {"language": "iw"},
            {"language": ""},
            {"language": "zzz"},
        ]
    )
    model = _loaded_model(client)
    supported = model.discover_supported_languages()

    assert isinstance(supported, frozenset)
    # zh-CN and zh-TW collapse to one code; "" and "zzz" drop out.
    assert supported == frozenset({"eng", "zho", "heb"})
