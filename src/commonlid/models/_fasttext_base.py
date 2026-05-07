"""Shared plumbing for fasttext-based LID models (GlotLID, OpenLID-v2, fasttext-ft)."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download

from commonlid.core.lid_model import LIDModel

_FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314


def _read_labels_from_bin(path: str | Path) -> list[str]:
    """Parse the labels block out of a fasttext ``model.bin`` file.

    The runtime dep ``fasttext-predict`` is inference-only and exposes only
    ``predict`` / ``multilinePredict`` — no ``get_labels()``. Read the file
    directly so :meth:`FastTextHubModel.discover_supported_languages` works
    under either binding.

    File layout (little-endian; mirrors FastText's ``saveModel`` +
    ``Dictionary::save`` in the upstream C++ source):
      - 8 bytes:  magic ``int32`` + version ``int32``
      - 56 bytes: 12 ``int32`` + 1 ``double`` (Args block)
      - 12 bytes: ``size_``, ``nwords_``, ``nlabels_`` (3 ``int32``)
      - 16 bytes: ``ntokens_``, ``pruneidx_size_`` (2 ``int64``)
      - then ``size_`` entries: NUL-terminated UTF-8 word, ``int64`` count,
        ``int8`` entry type (``0`` = word, ``1`` = label).
    """
    with Path(path).open("rb") as f:
        magic, _version = struct.unpack("<ii", f.read(8))
        if magic != _FASTTEXT_FILEFORMAT_MAGIC_INT32:
            msg = f"Not a fasttext model.bin (magic={magic:#x}): {path}"
            raise ValueError(msg)
        f.read(56)  # Args: 12 int32 + 1 double
        size_, _nwords, nlabels_ = struct.unpack("<3i", f.read(12))
        f.read(16)  # ntokens_ + pruneidx_size_
        labels: list[str] = []
        for _ in range(size_):
            word_bytes = bytearray()
            while True:
                b = f.read(1)
                if not b or b == b"\0":
                    break
                word_bytes.extend(b)
            f.read(8)  # count int64
            entry_type = f.read(1)
            if entry_type == b"\x01":
                labels.append(word_bytes.decode("utf-8"))
                if len(labels) == nlabels_:
                    break
    return labels


class FastTextHubModel(LIDModel):
    """Base class for HF-hosted fasttext LID models.

    Subclasses set :attr:`hf_repo_id` and :attr:`hf_filename` (default
    ``"model.bin"``). The loaded fasttext model is stashed on the instance.
    """

    hf_repo_id: str
    hf_filename: str = "model.bin"

    def __init__(self) -> None:
        super().__init__()
        self._ft: fasttext.FastText._FastText | None = None
        self._model_path: str | None = None

    def load(self) -> None:
        if self._loaded:
            return
        path = hf_hub_download(repo_id=self.hf_repo_id, filename=self.hf_filename)
        self._model_path = path
        self._ft = fasttext.load_model(path)
        super().load()

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        assert self._ft is not None  # load() has run
        labels = self._predict_labels(list(texts))
        out: list[str | None] = []
        for results in labels:
            label = results[0]
            if not label.startswith("__label__"):
                msg = f"Unexpected fasttext label format from {self.model_id}: {label!r}"
                raise ValueError(msg)
            code = label.split("__")[2].split("_")[0]
            out.append(code)
        return out

    def _predict_labels(self, texts: list[str]) -> list[list[str]]:
        """Return ``[[label, ...], ...]`` from either ``fasttext-wheel`` or ``fasttext-predict``.

        ``fasttext-wheel`` returns ``(labels, probs)`` from ``predict``;
        ``fasttext-predict`` returns a plain ``labels`` list. The stock
        ``fasttext.FastText._FastText.predict`` wrapper in the latter also
        happens to be broken on Python 3.13 (it tries to unpack a 1-tuple
        as 2 values), so we sidestep it and call ``multilinePredict``
        directly when available.
        """
        assert self._ft is not None
        mp = getattr(self._ft.f, "multilinePredict", None)
        if mp is not None:
            prepared = [t if t.endswith("\n") else t + "\n" for t in texts]
            result = mp(prepared, 1, 0.0, "strict")
            # fasttext-wheel: ([[labels], ...], [[probs], ...])  — a 2-tuple of lists.
            # fasttext-predict: [[labels], ...]                   — a bare list of label lists.
            if isinstance(result, tuple) and len(result) == 2:
                return list(result[0])
            return list(result)

        # Fallback: call the wrapper and unpack whatever shape it returns.
        predicted = self._ft.predict(list(texts))
        if isinstance(predicted, tuple) and len(predicted) == 2:
            return list(predicted[0])
        return list(predicted)

    def discover_supported_languages(self) -> frozenset[str]:
        """Enumerate every ``__label__{code}`` exposed by the loaded fasttext model.

        ``fasttext-wheel`` exposes ``get_labels()`` on the loaded model;
        ``fasttext-predict`` (the lighter inference-only fork we depend on at
        runtime) does not. Fall back to parsing the dictionary block out of
        ``model.bin`` directly when ``get_labels`` is missing.
        """
        if self._ft is None:
            self.load()
        assert self._ft is not None
        get_labels = getattr(self._ft, "get_labels", None)
        if get_labels is not None:
            raw_labels = list(get_labels())
        else:
            assert self._model_path is not None  # set by load()
            raw_labels = _read_labels_from_bin(self._model_path)
        codes: set[str] = set()
        for label in raw_labels:
            if not label.startswith("__label__"):
                continue
            raw = label.split("__")[2].split("_")[0]
            conformed = self._conform(raw)
            if conformed is not None:
                codes.add(conformed)
        return frozenset(codes)
