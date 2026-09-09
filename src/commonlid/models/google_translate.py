"""Google Cloud Translation language-detection wrapper.

Wraps the Cloud Translation `detect
<https://docs.cloud.google.com/translate/docs/reference/rest/v2/detect>`_
method via the ``google-cloud-translate`` client library, pulled in by the
optional ``commonlid[google-translate]`` extra.

**Why v2 and not v3.** Cloud Translation v3 ``detectLanguage`` rejects API-key
authentication outright (``401 API keys are not supported by this API``) on
both the gRPC and REST transports; it wants OAuth2 / Application Default
Credentials plus a project id in the ``parent`` path. The v2 surface accepts an
API key, needs no project id, and — unlike v3, whose ``content`` field is a
single string — detects a whole list of texts in one request, which cuts a
373k-sample benchmark from 373k HTTP calls to a few thousand.

The API key is read from the process environment, which the caller is expected
to have set:

* ``GOOGLE_TRANSLATE_API_KEY`` — an API key with the Cloud Translation API
  enabled on its project.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from commonlid.core.lid_model import LIDModel, LIDPrediction
from commonlid.core.registry import register_model

logger = logging.getLogger(__name__)

API_KEY_ENV = "GOOGLE_TRANSLATE_API_KEY"

_MISSING_DEPS_MSG = (
    "The Google Cloud Translation client is not installed. Install the "
    "'commonlid[google-translate]' extra to enable this model."
)
_MISSING_KEY_MSG = f"GoogleTranslate needs {API_KEY_ENV} to be set in the environment."


@register_model
class GoogleTranslateModel(LIDModel):
    """Google Cloud Translation ``detect`` as a LID model."""

    model_id = "GoogleTranslate"

    # A v2 request is capped at 100K bytes of payload. Chunk well under that,
    # and cap the segment count too — the API rejects very long `q` lists.
    _MAX_SEGMENTS_PER_REQUEST: ClassVar[int] = 100
    _MAX_CHARS_PER_REQUEST: ClassVar[int] = 20_000
    # Hard API ceiling, not a cost-saving truncation: a single text longer than
    # this cannot be sent at all. Nothing in the shipped benchmarks comes close.
    _MAX_CHARS_PER_TEXT: ClassVar[int] = 20_000
    # Chunks of one batch go out concurrently. The v2 quota is 300k requests
    # per minute, so this is bounded by politeness, not by the quota.
    _MAX_WORKERS: ClassVar[int] = 8
    _MAX_ATTEMPTS: ClassVar[int] = 5
    _BACKOFF_BASE_SECONDS: ClassVar[float] = 1.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._max_workers = max_workers if max_workers is not None else self._MAX_WORKERS
        self._client: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import google.auth.api_key
            from google.cloud import translate_v2
        except ImportError as exc:
            raise ImportError(_MISSING_DEPS_MSG) from exc

        api_key = self._api_key or os.environ.get(API_KEY_ENV)
        if not api_key:
            raise RuntimeError(_MISSING_KEY_MSG)

        # `translate_v2.Client` ignores `client_options={"api_key": ...}` and
        # falls through to `google.auth.default()`, so hand it API-key
        # credentials directly — those attach the `x-goog-api-key` header.
        credentials = google.auth.api_key.Credentials(api_key)  # type: ignore[no-untyped-call]
        self._client = translate_v2.Client(credentials=credentials)
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
        out: list[LIDPrediction] = [LIDPrediction(None, None)] * len(texts)
        # Blank input never reaches the API: it would be rejected, and a failed
        # request still counts against the per-character bill.
        billable = [(i, t.strip()) for i, t in enumerate(texts) if t.strip()]
        if not billable:
            return out

        chunks = list(self._chunk(billable))
        workers = max(1, min(self._max_workers, len(chunks)))
        if workers == 1:
            results = [self._detect_chunk(chunk) for chunk in chunks]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._detect_chunk, chunks))

        for chunk, codes in zip(chunks, results, strict=True):
            for (index, _), code in zip(chunk, codes, strict=True):
                out[index] = code
        return out

    @classmethod
    def _chunk(cls, billable: Sequence[tuple[int, str]]) -> Iterator[list[tuple[int, str]]]:
        """Split into request-sized groups, honouring the segment and size caps."""
        current: list[tuple[int, str]] = []
        chars = 0
        for index, text in billable:
            clipped = text[: cls._MAX_CHARS_PER_TEXT]
            if current and (
                len(current) >= cls._MAX_SEGMENTS_PER_REQUEST
                or chars + len(clipped) > cls._MAX_CHARS_PER_REQUEST
            ):
                yield current
                current, chars = [], 0
            current.append((index, clipped))
            chars += len(clipped)
        if current:
            yield current

    def _detect_chunk(self, chunk: Sequence[tuple[int, str]]) -> list[LIDPrediction]:
        detections = self._call_with_retry([text for _, text in chunk])
        out: list[LIDPrediction] = []
        for detection in detections:
            if not isinstance(detection, dict):
                out.append(LIDPrediction(None, None))
                continue
            code = detection.get("language")
            # `confidence` is documented as optional and is absent on some
            # responses, so a missing score is not an error.
            score = detection.get("confidence")
            score = float(score) if score is not None else None
            if not code or code == "und":
                out.append(LIDPrediction(None, score))
                continue
            # v2 emits BCP-47, so `zh-CN` / `zh-TW` collapse to `zh` before the
            # ISO 639-3 upgrade in `LIDModel._conform`, as in `cld3.py`.
            out.append(LIDPrediction(code.split("-")[0], score))
        return out

    def _call_with_retry(self, values: list[str]) -> list[Any]:
        from google.api_core import exceptions as gexc

        retryable = (
            gexc.TooManyRequests,
            gexc.ServiceUnavailable,
            gexc.InternalServerError,
            gexc.GatewayTimeout,
        )
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                result = self._client.detect_language(values)
            except retryable as exc:
                if attempt == self._MAX_ATTEMPTS:
                    raise
                delay = self._BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    "%s: %s on attempt %d/%d, retrying in %.1fs",
                    self.model_id,
                    type(exc).__name__,
                    attempt,
                    self._MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
            else:
                # A one-element list still comes back as a bare dict.
                return result if isinstance(result, list) else [result]
        raise AssertionError("unreachable")  # pragma: no cover

    def discover_supported_languages(self) -> frozenset[str]:
        """Ask the API which languages it supports, as ISO 639-3."""
        if not self._loaded:
            self.load()
        codes: set[str] = set()
        for entry in self._client.get_languages():
            raw = entry.get("language")
            if not raw:
                continue
            conformed = self._conform(raw.split("-")[0])
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
