"""Abstract base class for language-identification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from iso639 import Lang
from iso639.exceptions import DeprecatedLanguageValue, InvalidLanguageValue

from commonlid.preprocess import conform_langcode, openlid_normer_clean_line


@dataclass(frozen=True, slots=True)
class LIDPrediction:
    """A single prediction: ISO 639-3 code (or ``None`` for "und") and optional score."""

    iso639_3: str | None
    score: float | None = None


class LIDModel(ABC):
    """Base class for all LID models.

    Subclasses declare ``model_id`` and implement :meth:`_predict_batch`.

    The public :meth:`predict` applies the OpenLID-v2 normer to each input
    (unless ``requires_preprocessing`` is ``False``) before delegating.
    Predicted codes are passed through :func:`conform_langcode` so every
    model emits canonical ISO 639-3 codes.
    """

    # `model_id` is a class attribute set by subclasses; a small number of
    # models (e.g. :class:`~commonlid.models.dspy_llm.DSPyLLMModel`) override
    # it per-instance, so it is not marked ``ClassVar``.
    model_id: str
    supported_languages: ClassVar[frozenset[str] | None] = None
    requires_preprocessing: ClassVar[bool] = True
    default_batch_size: ClassVar[int] = 64

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        """Lazy weight/binary load hook. Safe to call multiple times."""
        self._loaded = True

    @abstractmethod
    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        """Core prediction hook; must return an ISO 639-3 code or ``None`` per input."""

    def predict(self, texts: Sequence[str]) -> list[str | None]:
        """Return an ISO 639-3 code (or ``None`` for undetermined) per input text."""
        if not self._loaded:
            self.load()
        if self.requires_preprocessing:
            prepared = [openlid_normer_clean_line(t) for t in texts]
        else:
            prepared = list(texts)
        raw = self._predict_batch(prepared)
        return [self._conform(code) for code in raw]

    def supports(self, iso639_3: str) -> bool:
        """Whether the model declares support for the given ISO 639-3 language."""
        if self.supported_languages is None:
            return True
        return iso639_3 in self.supported_languages

    def discover_supported_languages(self) -> frozenset[str] | None:
        """Return the set of ISO 639-3 codes the model claims to support.

        Default implementation returns :attr:`supported_languages` (which may
        be ``None`` meaning "unknown"). Subclasses with enumerable label sets
        (fasttext, cld2, pyfranc, funlangid, AfroLID) override this to walk
        the model's own metadata.

        Callers that want a CSV matrix via the ``commonlid
        generate-support-matrix`` CLI must call :meth:`load` first; this
        method is free to assume the backing model has been loaded.
        """
        return self.supported_languages

    @staticmethod
    def _conform(code: str | None) -> str | None:
        """Conform a raw model output to canonical ISO 639-3.

        Matches the two-stage conformation from the legacy pipeline:
        first the hand-written deprecation table
        (:func:`~commonlid.preprocess.conform_langcode`), then
        ``iso639.Lang(...).pt3`` to upgrade ISO 639-1/2 codes to ISO 639-3.
        Deprecated or unknown codes map to ``None``.
        """
        if code is None:
            return None
        conformed = conform_langcode(code)
        if conformed is None:
            return None
        try:
            lang = Lang(conformed)
        except (InvalidLanguageValue, DeprecatedLanguageValue):
            return None
        return lang.pt3 or None
