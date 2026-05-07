"""Unit coverage for the nano LIDDataset variants."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
import pytest


def test_each_nano_class_points_at_its_parent() -> None:
    from commonlid.datasets.bibles import BiblesDataset
    from commonlid.datasets.commonlid import CommonLIDDataset
    from commonlid.datasets.flores_dev import FloresDevDataset
    from commonlid.datasets.nano import (
        BiblesDatasetNano,
        CommonLIDDatasetNano,
        FloresDevDatasetNano,
        SmolSentDatasetNano,
        SocialMediaDatasetNano,
        UDHRDatasetNano,
    )
    from commonlid.datasets.smolsent import SmolSentDataset
    from commonlid.datasets.social_media import SocialMediaDataset
    from commonlid.datasets.udhr import UDHRDataset

    assert BiblesDatasetNano.parent_dataset_cls is BiblesDataset
    assert CommonLIDDatasetNano.parent_dataset_cls is CommonLIDDataset
    assert FloresDevDatasetNano.parent_dataset_cls is FloresDevDataset
    assert SmolSentDatasetNano.parent_dataset_cls is SmolSentDataset
    assert UDHRDatasetNano.parent_dataset_cls is UDHRDataset
    assert SocialMediaDatasetNano.parent_dataset_cls is SocialMediaDataset


def test_nano_classes_share_normalised_schema() -> None:
    from commonlid.datasets.nano import (
        BiblesDatasetNano,
        CommonLIDDatasetNano,
        FloresDevDatasetNano,
        SmolSentDatasetNano,
        SocialMediaDatasetNano,
        UDHRDatasetNano,
    )

    for cls in (
        BiblesDatasetNano,
        CommonLIDDatasetNano,
        FloresDevDatasetNano,
        SmolSentDatasetNano,
        SocialMediaDatasetNano,
        UDHRDatasetNano,
    ):
        assert cls.cache_hf_split == "test"
        assert cls.text_column == "text"
        assert cls.target_column == "language_iso639_3"
    # Visibility tracks the parent dataset; private by default, but the
    # commonlid nano is published publicly because the parent is public.
    assert CommonLIDDatasetNano.is_cache_private is False
    for cls in (
        BiblesDatasetNano,
        FloresDevDatasetNano,
        SmolSentDatasetNano,
        SocialMediaDatasetNano,
        UDHRDatasetNano,
    ):
        assert cls.is_cache_private is True


def test_each_nano_has_its_own_per_dataset_cache_repo() -> None:
    from commonlid.datasets.nano import (
        BiblesDatasetNano,
        CommonLIDDatasetNano,
        FloresDevDatasetNano,
        SmolSentDatasetNano,
        SocialMediaDatasetNano,
        UDHRDatasetNano,
    )

    expected = {
        BiblesDatasetNano: "commoncrawl/commonlid-cache_bibles_300_nano",
        CommonLIDDatasetNano: "commoncrawl/commonlid-cache_commonlid_nano",
        FloresDevDatasetNano: "commoncrawl/commonlid-cache_flores_dev_nano",
        SmolSentDatasetNano: "commoncrawl/commonlid-cache_smolsent_300_nano",
        UDHRDatasetNano: "commoncrawl/commonlid-cache_udhr_nano",
        SocialMediaDatasetNano: "commoncrawl/commonlid-cache_social_media_300_nano",
    }
    for cls, repo in expected.items():
        assert cls.cache_hf_repo == repo
    # All six repos must be distinct.
    assert len({cls.cache_hf_repo for cls in expected}) == len(expected)


def test_build_from_source_normalises_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The nano build outputs (index, text, language_iso639_3) regardless of parent schema."""
    from datasets import Dataset

    from commonlid.datasets.nano import SmolSentDatasetNano
    from commonlid.datasets.smolsent import SmolSentDataset

    # Need each surviving class > target_size = round(MAX_SIZE * len/total).
    # With 3 equal classes of 2000, target is ~333 per class, < 1995 surviving.
    rows: list[dict[str, str]] = []
    for lang in ("eng", "deu", "fra"):
        rows += [{"trg": f"{lang}-{i}", "tl_iso693_3": lang} for i in range(2000)]
    fake = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

    def fake_load(self: SmolSentDataset, *, limit: int | None = None) -> Any:
        return fake

    monkeypatch.setattr(SmolSentDataset, "load", fake_load)
    out = SmolSentDatasetNano().build_from_source()
    assert out.column_names == ["index", "text", "language_iso639_3"]
    assert set(out["language_iso639_3"]) <= {"eng", "deu", "fra"}


def test_build_from_source_drops_classes_below_min_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Languages with fewer than ``_MIN_SIZE`` rows are dropped entirely."""
    from datasets import Dataset

    from commonlid.datasets.nano import SmolSentDatasetNano
    from commonlid.datasets.smolsent import SmolSentDataset

    rows = [{"trg": f"e{i}", "tl_iso693_3": "eng"} for i in range(2000)]
    rows += [{"trg": "d-only", "tl_iso693_3": "deu"} for _ in range(2)]  # 2 < min_size=5
    fake = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
    monkeypatch.setattr(SmolSentDataset, "load", lambda *_a, **_kw: fake)

    out = SmolSentDatasetNano().build_from_source()
    assert "deu" not in set(out["language_iso639_3"])


def test_build_is_deterministic_given_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from datasets import Dataset

    from commonlid.datasets.nano import SmolSentDatasetNano
    from commonlid.datasets.smolsent import SmolSentDataset

    rows = [{"trg": f"e{i}", "tl_iso693_3": "eng"} for i in range(2000)]
    rows += [{"trg": f"d{i}", "tl_iso693_3": "deu"} for i in range(2000)]
    fake = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
    monkeypatch.setattr(SmolSentDataset, "load", lambda *_a, **_kw: fake)

    a = SmolSentDatasetNano().build_from_source()
    b = SmolSentDatasetNano().build_from_source()
    assert list(a["index"]) == list(b["index"])
    assert list(a["text"]) == list(b["text"])


def test_build_total_size_respects_max_plus_min_quotas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Total sample count = ~MAX_SIZE + MIN_SIZE * surviving_classes."""
    from datasets import Dataset

    from commonlid.datasets.nano import SmolSentDatasetNano
    from commonlid.datasets.smolsent import SmolSentDataset

    rows: list[dict[str, str]] = []
    for lang in ("eng", "deu", "fra", "ita", "spa"):
        rows += [{"trg": f"{lang}-{i}", "tl_iso693_3": lang} for i in range(400)]
    fake = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)
    monkeypatch.setattr(SmolSentDataset, "load", lambda *_a, **_kw: fake)

    out = SmolSentDatasetNano().build_from_source()
    counts = Counter(out["language_iso639_3"])
    assert set(counts) == {"eng", "deu", "fra", "ita", "spa"}
    assert all(n >= SmolSentDatasetNano._MIN_SIZE for n in counts.values())
    # Stratified sample of 1000 over 5 equally-sized classes: each gets 200,
    # plus a guaranteed min_size=5 → ~205 each, total ~1025.
    assert abs(len(out) - (SmolSentDatasetNano._MAX_SIZE + SmolSentDatasetNano._MIN_SIZE * 5)) <= 5
