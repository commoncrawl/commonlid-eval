"""Abstract base class for language-identification evaluation datasets."""

from __future__ import annotations

import logging
from abc import ABC
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from commonlid.preprocess import conform_langcode_with_reason

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LIDSample:
    """One evaluation sample."""

    text: str
    iso639_3: str | None


class PrivateDatasetAccessError(RuntimeError):
    """Raised when a dataset's private cache is inaccessible.

    Indicates ``cache_hf_repo`` is gated or 404s for the caller — typically
    because the user lacks access or has not authenticated. The message
    points them at the access-request URL and ``huggingface-cli login``,
    plus a ``build_from_source()`` fallback when the subclass exposes one.
    """

    def __init__(
        self,
        *,
        dataset_id: str,
        cache_hf_repo: str,
        build_hint: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.cache_hf_repo = cache_hf_repo
        msg = (
            f"Dataset '{dataset_id}' cache (cache_hf_repo={cache_hf_repo!r}) is private "
            f"and could not be loaded. Request access at "
            f"https://huggingface.co/datasets/{cache_hf_repo} and authenticate via "
            f"`huggingface-cli login` (or set the HF_TOKEN env var). If you already "
            f"have access, double-check that your token is valid."
        )
        if build_hint:
            msg += f"\n\nAlternatively, rebuild from the public source: {build_hint}"
        super().__init__(msg)


def _is_hf_access_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like an HF auth/access/missing-repo failure."""
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return False
    if isinstance(exc, GatedRepoError | RepositoryNotFoundError):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403, 404}:
            return True
    return False


class LIDDataset(ABC):
    """A registered evaluation dataset.

    Two HF-repo attributes disambiguate the public source from a private cache:

    * ``source_hf_repo`` — the canonical *public* HF dataset. For benchmarks
      that ship in usable form (FLORES+, UDHR-LID, CommonLID), :meth:`load`
      reads from this directly. For benchmarks that need preprocessing
      (smolsent), :meth:`build_from_source` fetches the raw data here.
    * ``cache_hf_repo`` — an optional pre-built (often private) HF artifact
      of the preprocessed/sampled subset. When set, :meth:`load` tries this
      first, falling back to ``build_from_source()`` if the cache is gated.

    At least one of the two must be set. ``is_cache_private=True`` enables
    the private-cache → build-from-source fallback.
    """

    dataset_id: ClassVar[str]

    # Human-readable metadata (used by leaderboards / docs / dataset cards).
    title: ClassVar[str | None] = None
    description: ClassVar[str | None] = None
    reference_url: ClassVar[str | None] = None
    main_score: ClassVar[str] = "macro_f1"
    #: SPDX-style identifier in lowercase (``cc-by-nc-4.0`` etc.). For custom
    #: licenses use a slug like ``common-crawl-tou`` and pair it with
    #: ``license_url``. ``"not specified"`` when the license is unknown.
    license_name: ClassVar[str] = "not specified"
    #: Optional URL to the license text. Required when ``license_name`` is a
    #: custom slug; omit (``None``) for well-known SPDX licenses.
    license_url: ClassVar[str | None] = None

    # Public source.
    source_hf_repo: ClassVar[str | None] = None
    source_hf_split: ClassVar[str] = "test"
    source_hf_revision: ClassVar[str | None] = None
    source_hf_config: ClassVar[str | None] = None

    # Optional pre-built (often private) cache.
    cache_hf_repo: ClassVar[str | None] = None
    cache_hf_split: ClassVar[str] = "train"
    cache_hf_revision: ClassVar[str | None] = None
    cache_hf_config: ClassVar[str | None] = None

    text_column: ClassVar[str]
    target_column: ClassVar[str]
    is_cache_private: ClassVar[bool] = False
    build_source_hint: ClassVar[str | None] = None

    def __init__(self) -> None:
        self._dataset: Any = None

    def load(self, *, limit: int | None = None) -> Any:
        """Load the underlying HF dataset. Subsequent calls return the cached handle.

        Resolution order:
            1. If ``cache_hf_repo`` is set, try loading it. On private-access
               failure with ``is_cache_private=True``, fall to step 2.
            2. Try :meth:`build_from_source`. If that yields a Dataset, use it.
            3. If neither cache nor build worked, raise
               :class:`PrivateDatasetAccessError`.
            4. If no cache is configured, load from ``source_hf_repo`` directly.
        """
        if self._dataset is None:
            self._dataset = self._fetch()
            self._check_gold_conformity()
        if limit is not None and limit > 0:
            return self._dataset.select(range(min(limit, len(self._dataset))))
        return self._dataset

    def _fetch(self) -> Any:
        from datasets import load_dataset

        if self.cache_hf_repo is not None:
            try:
                return load_dataset(
                    self.cache_hf_repo,
                    self.cache_hf_config,
                    split=self.cache_hf_split,
                    revision=self.cache_hf_revision,
                )
            except Exception as exc:
                if self.is_cache_private and _is_hf_access_error(exc):
                    return self._fallback_to_build(exc)
                raise

        if self.source_hf_repo is not None:
            return load_dataset(
                self.source_hf_repo,
                self.source_hf_config,
                split=self.source_hf_split,
                revision=self.source_hf_revision,
            )

        msg = (
            f"Dataset {self.dataset_id!r} has neither `cache_hf_repo` nor "
            f"`source_hf_repo` set; nothing to load."
        )
        raise RuntimeError(msg)

    @property
    def hf_revision(self) -> str | None:
        """Revision pin for whichever HF repo ``load()`` would canonically use.

        Cache-pinned when a cache is configured (the cache is what `load()`
        tries first), otherwise source-pinned. Used by the evaluator's
        results metadata + prediction-cache key.
        """
        if self.cache_hf_repo is not None:
            return self.cache_hf_revision
        return self.source_hf_revision

    def build_from_source(self) -> Any:
        """Rebuild the dataset from its public upstream source.

        Subclasses override this to fetch the raw data and re-run the
        preprocessing pipeline. Returns a ``datasets.Dataset`` matching the
        cached schema (``text_column`` and ``target_column`` populated).
        """
        msg = (
            f"Dataset {self.dataset_id!r} does not implement build_from_source(); "
            f"only the cached HF artifact at {self.cache_hf_repo!r} is available."
        )
        raise NotImplementedError(msg)

    def _fallback_to_build(self, cache_error: BaseException) -> Any:
        """Try ``build_from_source()``; on failure, raise the wrapped access error."""
        assert self.cache_hf_repo is not None
        try:
            return self.build_from_source()
        except NotImplementedError:
            raise PrivateDatasetAccessError(
                dataset_id=self.dataset_id,
                cache_hf_repo=self.cache_hf_repo,
            ) from cache_error
        except Exception as build_exc:
            raise PrivateDatasetAccessError(
                dataset_id=self.dataset_id,
                cache_hf_repo=self.cache_hf_repo,
                build_hint=(
                    f"{self.build_source_hint or 'see build_from_source()'} "
                    f"(build attempt failed: {build_exc})"
                ),
            ) from cache_error

    def iter_batches(
        self, batch_size: int = 64, *, limit: int | None = None
    ) -> Iterator[tuple[list[str], list[str | None]]]:
        """Yield ``(texts, gold_iso639_3)`` in batches of ``batch_size``."""
        ds = self.load(limit=limit)
        for batch in ds.iter(batch_size=batch_size):
            yield batch[self.text_column], list(batch[self.target_column])

    def __len__(self) -> int:
        if self._dataset is None:
            self.load()
        return len(self._dataset)

    def _check_gold_conformity(self) -> None:
        counter: Counter[str] = Counter()
        for code in self._dataset[self.target_column]:
            if code is None:
                continue
            conformed, reason = conform_langcode_with_reason(code)
            if conformed != code:
                counter[
                    f"dataset has '{code}' -> conformed to {conformed}. Reason: \"{reason}\""
                ] += 1
        if counter:
            logger.info(
                "Dataset %s has non-conforming ISO 639-3 codes in column '%s':",
                self.dataset_id,
                self.target_column,
            )
            for message, count in counter.items():
                logger.info("  %d occurrences: %s", count, message)
