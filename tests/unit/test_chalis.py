"""CHALIS: sections of one JSON file as single-label datasets (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from commonlid.core.registry import get_dataset, list_datasets
from commonlid.datasets import chalis

_SECTIONS: dict[str, list[dict[str, Any]]] = {
    "single": [
        {"text": "Ale nie je to tak.", "label": ["slk_Latn"]},
        {"text": "Het is een erg vreemd gevoel.", "label": ["nld_Latn"]},
    ],
    "joint": [{"text": "Je to obrovský problém.", "label": ["ces_Latn", "slk_Latn"]}],
    "non-language": [{"text": "kratochvil.cz", "label": []}],
    "cyrilic_to_english": [
        {"text": "Lyubimov", "label": ["rus_Cyrl"], "original": "Любимов"},
        {"text": "p'jesy", "label": [], "original": "п'єси"},
        {"text": "Makedonija", "label": ["mkd_Cyrl"], "original": "Македонија"},
    ],
    "arabic": [{"text": "بەرىلگەن", "label": ["kaz_Cyrl"], "original": "берілген"}],
}


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def _fake(repo: str, revision: str) -> dict[str, list[dict[str, Any]]]:
        calls.append((repo, revision))
        return _SECTIONS

    monkeypatch.setattr(chalis, "_download_sections", _fake)
    return calls


def test_registered_sections() -> None:
    assert [d for d in list_datasets() if d.startswith("chalis_")] == [
        "chalis_antspeak",
        "chalis_arabic",
        "chalis_cyrillic_to_czech",
        "chalis_cyrillic_to_english",
        "chalis_diacritics",
        "chalis_homoglyphs",
        "chalis_leetspeak",
        "chalis_single",
    ]


@pytest.mark.parametrize(
    ("label", "code"),
    [("ces_Latn", "ces"), ("kaz_Cyrl", "kaz"), ("nob_Latn", "nob"), ("eng", "eng")],
)
def test_iso639_3_from_label(label: str, code: str) -> None:
    assert chalis.iso639_3_from_label(label) == code


def test_items_to_rows_keeps_single_label_items_only() -> None:
    rows = chalis.items_to_rows(
        _SECTIONS["cyrilic_to_english"] + _SECTIONS["joint"] + _SECTIONS["non-language"]
    )
    assert rows == [
        {
            "index": 0,
            "text": "Lyubimov",
            "language_iso639_3": "rus",
            "label": "rus_Cyrl",
            "original": "Любимов",
        },
        {
            "index": 2,
            "text": "Makedonija",
            "language_iso639_3": "mkd",
            "label": "mkd_Cyrl",
            "original": "Македонија",
        },
    ]


def test_load_builds_the_section_from_the_pinned_file(fake_download: list[tuple[str, str]]) -> None:
    ds = get_dataset("chalis_cyrillic_to_english")
    loaded = ds.load()
    assert fake_download == [("michal-tichy/CHALIS", chalis._REVISION)]
    assert loaded["text"] == ["Lyubimov", "Makedonija"]
    assert loaded["language_iso639_3"] == ["rus", "mkd"]
    assert ds.hf_revision == chalis._REVISION
    assert list(ds.iter_batches(batch_size=1)) == [
        (["Lyubimov"], ["rus"]),
        (["Makedonija"], ["mkd"]),
    ]


def test_collected_sections_have_no_original(fake_download: list[tuple[str, str]]) -> None:
    loaded = get_dataset("chalis_single").load()
    assert loaded["language_iso639_3"] == ["slk", "nld"]
    assert loaded["original"] == [None, None]


def test_gold_is_the_source_language_not_the_script(fake_download: list[tuple[str, str]]) -> None:
    assert get_dataset("chalis_arabic").load()["language_iso639_3"] == ["kaz"]


def test_every_section_key_is_an_upstream_key() -> None:
    from commonlid.core.registry import _DATASET_REGISTRY

    upstream = {
        "single",
        "joint",
        "non-language",
        "cyrilic_to_czech",
        "cyrilic_to_english",
        "arabic",
        "diacritics",
        "homoglyphs",
        "antspeak",
        "leet",
    }
    sections = {
        cls.section for cls in _DATASET_REGISTRY.values() if issubclass(cls, chalis._ChalisDataset)
    }
    assert sections <= upstream
    assert {"joint", "non-language"}.isdisjoint(sections)
