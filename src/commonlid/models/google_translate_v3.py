"""Google Cloud Translation **Advanced (v3)** language-detection wrapper.

A sibling of :mod:`commonlid.models.google_translate_v2`, which wraps the Basic
(v2) ``detect`` method. Both ship so the two editions can be compared; neither
replaces the other.

The editions differ in three ways that matter here, all per the official
reference:

* **Auth.** Basic "accept[s] API keys for authentication as well as service
  accounts"; Advanced "requires service account authentication that's
  integrated with IAM roles" and "does not support API keys". So this model
  uses Application Default Credentials and needs a project id, while the v2
  wrapper needs only ``GOOGLE_TRANSLATE_API_KEY``.
* **Batching.** v2's ``q`` is repeatable, so one request detects a whole list.
  v3's ``content`` is a single string, so a batch here is a thread pool over
  one request per text.
* **Confidence.** v2 documents both ``isReliable`` and ``confidence`` as
  deprecated ("We recommend not basing any decisions or thresholds on the
  isReliable or confidence values"). v3 carries no such warning.

Google does not document whether the two editions share a detection model.
v3's optional ``model`` field currently accepts only the default language
detection model, which hints that they do, but that is inference, not a
documented guarantee.

Credentials, all from the process environment:

* Application Default Credentials, via ``gcloud auth application-default
  login`` or ``GOOGLE_APPLICATION_CREDENTIALS`` pointing at a service account
  key.
* A project id, resolved from ``GOOGLE_TRANSLATE_PROJECT_ID``, then
  ``GOOGLE_CLOUD_PROJECT``, then whatever ADC reports (its own project for a
  service account, else its quota project).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

from commonlid.core.lid_model import LIDModel, LIDPrediction
from commonlid.core.registry import register_model

logger = logging.getLogger(__name__)

PROJECT_ID_ENV = "GOOGLE_TRANSLATE_PROJECT_ID"
FALLBACK_PROJECT_ID_ENV = "GOOGLE_CLOUD_PROJECT"

_MISSING_DEPS_MSG = (
    "The Google Cloud Translation client is not installed. Install the "
    "'commonlid[google-translate]' extra to enable this model."
)
_MISSING_PROJECT_MSG = (
    f"GoogleTranslate-v3 could not resolve a project id. Set {PROJECT_ID_ENV} "
    f"or {FALLBACK_PROJECT_ID_ENV}, or configure Application Default "
    "Credentials that carry one."
)


@register_model
class GoogleTranslateV3Model(LIDModel):
    """Cloud Translation Advanced (v3) ``detectLanguage`` as a LID model."""

    model_id = "GoogleTranslate-v3"

    _LOCATION: ClassVar[str] = "global"
    # v3 takes one text per call, so a batch fans out. The project-wide quota
    # is 6000 v3 requests per minute; 16 in flight stays well under it.
    _MAX_WORKERS: ClassVar[int] = 16
    _MAX_ATTEMPTS: ClassVar[int] = 5
    _BACKOFF_BASE_SECONDS: ClassVar[float] = 1.0
    # Hard API ceiling: v3 rejects content longer than 30k code points.
    _MAX_CHARS_PER_TEXT: ClassVar[int] = 30_000

    def __init__(
        self,
        *,
        project_id: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        super().__init__()
        self._project_id = project_id
        self._max_workers = max_workers if max_workers is not None else self._MAX_WORKERS
        self._client: Any = None
        self._parent: str | None = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import google.auth
            from google.cloud import translate
        except ImportError as exc:
            raise ImportError(_MISSING_DEPS_MSG) from exc

        credentials, adc_project = google.auth.default()
        project_id = (
            self._project_id
            or os.environ.get(PROJECT_ID_ENV)
            or os.environ.get(FALLBACK_PROJECT_ID_ENV)
            or adc_project
            # User ADC reports no project of its own, only a quota project.
            or getattr(credentials, "quota_project_id", None)
        )
        if not project_id:
            raise RuntimeError(_MISSING_PROJECT_MSG)

        self._client = translate.TranslationServiceClient(credentials=credentials)
        self._parent = f"projects/{project_id}/locations/{self._LOCATION}"
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
        if not texts:
            return []
        workers = max(1, min(self._max_workers, len(texts)))
        if workers == 1:
            return [self._detect_one(t) for t in texts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._detect_one, texts))

    def _detect_one(self, text: str) -> LIDPrediction:
        # Blank input never reaches the API: v3 rejects it, and a failed
        # request still counts against the per-character bill.
        content = text.strip()
        if not content:
            return LIDPrediction(None, None)
        response = self._call_with_retry(content[: self._MAX_CHARS_PER_TEXT])
        languages = getattr(response, "languages", None) or []
        if not languages:
            return LIDPrediction(None, None)
        top = languages[0]
        score = float(top.confidence) if top.confidence is not None else None
        code = top.language_code
        if not code:
            return LIDPrediction(None, score)
        # v3 emits BCP-47, so `zh-CN` / `zh-TW` collapse to `zh` before the
        # ISO 639-3 upgrade in `LIDModel._conform`, as in `cld3.py`.
        return LIDPrediction(code.split("-")[0], score)

    def _call_with_retry(self, content: str) -> Any:
        from google.api_core import exceptions as gexc

        retryable = (
            gexc.ResourceExhausted,
            gexc.ServiceUnavailable,
            gexc.InternalServerError,
            gexc.DeadlineExceeded,
        )
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return self._client.detect_language(
                    parent=self._parent, content=content, mime_type="text/plain"
                )
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
        raise AssertionError("unreachable")  # pragma: no cover

    def discover_supported_languages(self) -> frozenset[str]:
        """Ask the API which languages it supports, as ISO 639-3."""
        if not self._loaded:
            self.load()
        response = self._client.get_supported_languages(parent=self._parent)
        codes: set[str] = set()
        for language in response.languages:
            conformed = self._conform(language.language_code.split("-")[0])
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
