"""CHALIS — Challenging Language Identification Samples (Tichý & Libovický, 2026).

Upstream ships the whole benchmark as one JSON file
(``dataset.json`` in ``michal-tichy/CHALIS``) whose top-level keys are the
sections; every item has ``text``, ``label`` (a list of ``<iso639_3>_<Script>``
codes) and, for the generated sections, ``original`` (the text before the
modification). Each single-label section is registered as its own dataset,
``chalis_<section>``, so it is scored on its own as in the paper:

* ``chalis_single``: sentences valid in only one language of a cousin pair
  (Czech/Slovak, Spanish/Catalan, Portuguese/Galician, Danish/Norwegian),
  built from words the two languages share; plus Dutch.
* transliteration: ``chalis_cyrillic_to_english`` (Bulgarian, Macedonian,
  Russian, Serbian in Latin script, English-style),
  ``chalis_cyrillic_to_czech`` (Russian in Czech orthography),
  ``chalis_arabic`` (Kazakh in Arabic script);
* orthography noise: ``chalis_diacritics`` (Czech, Slovak, Vietnamese without
  diacritics), ``chalis_homoglyphs`` (Czech, Slovak, English with Latin letters
  swapped for look-alike Cyrillic/Greek ones), ``chalis_antspeak``
  (``s p a c e d`` English), ``chalis_leetspeak`` (``l33t`` English).

The gold label is the language of the source text, whatever the script it is
now written in (Arabic-script Kazakh is ``kaz``). Two sections are not
registered, because the framework scores exactly one gold label per text:
``joint`` (sentences valid in both languages of a pair, two labels each; the
paper reports how predictions split between the two) and ``non-language``
(e-mail addresses, ISBNs, ...; no label). Items without a label are dropped
from the registered sections: in the pinned revision that is the 1,500
Ukrainian sentences of ``cyrilic_to_english``, which the paper does not score
either.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset

_REPO = "michal-tichy/CHALIS"
_REVISION = "e4fa744e9c799bb2120c8b907e4a6ff3f70b92ce"
_FILENAME = "dataset.json"


@lru_cache(maxsize=2)
def _download_sections(repo: str, revision: str) -> dict[str, list[dict[str, Any]]]:
    """Download and parse ``dataset.json`` once per process (all sections share it)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, _FILENAME, repo_type="dataset", revision=revision)
    with Path(path).open(encoding="utf-8") as f:
        sections: dict[str, list[dict[str, Any]]] = json.load(f)
    return sections


def iso639_3_from_label(label: str) -> str:
    """``"ces_Latn"`` -> ``"ces"``: CHALIS labels are ISO 639-3 plus a script."""
    return label.split("_", 1)[0]


class _ChalisDataset(LIDDataset):
    """One section of ``dataset.json`` as a dataset of ``(text, language_iso639_3)``.

    The source is a single JSON file with heterogeneous sections, which
    ``datasets.load_dataset`` cannot read as one table, so :meth:`load` goes
    through :meth:`build_from_source` instead of ``load_dataset``.
    """

    #: Top-level key in ``dataset.json``.
    section: ClassVar[str]

    reference_url = "https://arxiv.org/abs/2606.06088"
    main_score = "macro_f1"
    # The HF card says CC BY-NC 4.0 (the paper says both CC BY-NC and CC BY).
    license_name = "cc-by-nc-4.0"
    license_url = "https://creativecommons.org/licenses/by-nc/4.0/"

    source_hf_repo = _REPO
    source_hf_revision = _REVISION
    source_hf_split = "test"
    text_column = "text"
    target_column = "language_iso639_3"

    def _fetch(self) -> Any:
        return self.build_from_source()

    def build_from_source(self) -> Any:
        """Build the section from the pinned ``dataset.json``."""
        from datasets import Dataset

        assert self.source_hf_repo is not None
        assert self.source_hf_revision is not None
        items = _download_sections(self.source_hf_repo, self.source_hf_revision)[self.section]
        return Dataset.from_list(items_to_rows(items))


def items_to_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for :class:`datasets.Dataset`; items without exactly one label are dropped.

    ``original`` is ``None`` for sections that were collected, not generated.
    """
    rows = []
    for index, item in enumerate(items):
        labels = item["label"]
        if len(labels) != 1:
            continue
        rows.append({
            "index": index,
            "text": item["text"],
            "language_iso639_3": iso639_3_from_label(labels[0]),
            "label": labels[0],
            "original": item.get("original"),
        })
    return rows


_PAPER_SUFFIX = (
    " Part of CHALIS (Challenging Language Identification Samples), a challenge set "
    "for cousin languages and non-standard orthography."
)


@register_dataset
class ChalisSingleDataset(_ChalisDataset):
    dataset_id = "chalis_single"
    section = "single"
    title = "CHALIS: single"
    description = (
        "Sentences valid in only one language of a cousin pair (Czech/Slovak, "
        "Spanish/Catalan, Portuguese/Galician, Danish/Norwegian) but made of words "
        "both languages share, validated by native speakers; plus Dutch." + _PAPER_SUFFIX
    )


@register_dataset
class ChalisCyrillicToEnglishDataset(_ChalisDataset):
    dataset_id = "chalis_cyrillic_to_english"
    section = "cyrilic_to_english"
    title = "CHALIS: Cyrillic to Latin (English)"
    description = (
        "Bulgarian, Macedonian, Russian and Serbian Wikipedia sentences transliterated "
        "into Latin script with English-style conventions (CyrTranslit); the label is "
        "the source language." + _PAPER_SUFFIX
    )


@register_dataset
class ChalisCyrillicToCzechDataset(_ChalisDataset):
    dataset_id = "chalis_cyrillic_to_czech"
    section = "cyrilic_to_czech"
    title = "CHALIS: Cyrillic to Latin (Czech)"
    description = (
        "Russian Wikipedia sentences transliterated into Latin script with Czech "
        "orthography; the label is Russian." + _PAPER_SUFFIX
    )


@register_dataset
class ChalisArabicDataset(_ChalisDataset):
    dataset_id = "chalis_arabic"
    section = "arabic"
    title = "CHALIS: Kazakh in Arabic script"
    description = (
        "Kazakh sentences transliterated from Cyrillic into Arabic script; the label "
        "is Kazakh." + _PAPER_SUFFIX
    )


@register_dataset
class ChalisDiacriticsDataset(_ChalisDataset):
    dataset_id = "chalis_diacritics"
    section = "diacritics"
    title = "CHALIS: diacritics removed"
    description = (
        "Czech, Slovak and Vietnamese sentences with every diacritic removed." + _PAPER_SUFFIX
    )


@register_dataset
class ChalisHomoglyphsDataset(_ChalisDataset):
    dataset_id = "chalis_homoglyphs"
    section = "homoglyphs"
    title = "CHALIS: homoglyphs"
    description = (
        "Czech, Slovak and English sentences in which random Latin letters are replaced "
        "by look-alike Cyrillic or Greek letters." + _PAPER_SUFFIX
    )


@register_dataset
class ChalisAntspeakDataset(_ChalisDataset):
    dataset_id = "chalis_antspeak"
    section = "antspeak"
    title = "CHALIS: antspeak"
    description = "English sentences with a space between every letter." + _PAPER_SUFFIX


@register_dataset
class ChalisLeetspeakDataset(_ChalisDataset):
    dataset_id = "chalis_leetspeak"
    section = "leet"
    title = "CHALIS: leetspeak"
    description = (
        "English sentences with letters replaced by look-alike digits and symbols "
        "(pyleetspeak)." + _PAPER_SUFFIX
    )
