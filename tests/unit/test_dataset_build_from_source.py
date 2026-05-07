"""Build-from-source fallback for private-cache datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from commonlid.core.lid_dataset import LIDDataset, PrivateDatasetAccessError


def _hf_access_error() -> Exception:
    """Build a GatedRepoError instance, bypassing the HF __init__ shape requirements."""
    from huggingface_hub.errors import GatedRepoError

    exc = GatedRepoError.__new__(GatedRepoError)
    Exception.__init__(exc, "gated")

    class _Resp:
        status_code = 403

    exc.response = _Resp()  # type: ignore[attr-defined]
    return exc


def _patch_load_dataset_to_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets as datasets_mod

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise _hf_access_error()

    monkeypatch.setattr(datasets_mod, "load_dataset", _raise)


def test_load_falls_back_to_build_when_cache_private(monkeypatch: pytest.MonkeyPatch) -> None:
    from datasets import Dataset

    class _BuildableDS(LIDDataset):
        dataset_id = "_buildable_test"
        cache_hf_repo = "fake/private"
        cache_hf_split = "train"
        text_column = "text"
        target_column = "lang"
        is_cache_private = True

        def build_from_source(self) -> Any:
            return Dataset.from_pandas(
                pd.DataFrame({"text": ["hello"], "lang": ["eng"]}),
                preserve_index=False,
            )

    _patch_load_dataset_to_fail(monkeypatch)
    out = _BuildableDS().load()
    assert list(out["text"]) == ["hello"]
    assert list(out["lang"]) == ["eng"]


def test_build_failure_raises_private_dataset_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenBuildDS(LIDDataset):
        dataset_id = "_broken_build_test"
        cache_hf_repo = "fake/private"
        cache_hf_split = "train"
        text_column = "text"
        target_column = "lang"
        is_cache_private = True
        build_source_hint = "download foo.tsv"

        def build_from_source(self) -> Any:
            raise FileNotFoundError("source not configured")

    _patch_load_dataset_to_fail(monkeypatch)
    with pytest.raises(PrivateDatasetAccessError) as excinfo:
        _BrokenBuildDS().load()
    assert "download foo.tsv" in str(excinfo.value)
    assert "build attempt failed" in str(excinfo.value)


def test_no_build_method_raises_private_dataset_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoBuildDS(LIDDataset):
        dataset_id = "_no_build_test"
        cache_hf_repo = "fake/private"
        cache_hf_split = "train"
        text_column = "text"
        target_column = "lang"
        is_cache_private = True

    _patch_load_dataset_to_fail(monkeypatch)
    with pytest.raises(PrivateDatasetAccessError):
        _NoBuildDS().load()


def test_smolsent_build_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """SmolSent.build_from_source orchestrates conform + filter + sample correctly."""
    from datasets import Dataset

    from commonlid.datasets.smolsent import SmolSentDataset

    rows: list[dict[str, object]] = []
    for i in range(400):
        rows.append({
            "sl": "en",
            "tl": "eng",
            "id": i,
            "src": "x",
            "trg": f"e{i}",
            "is_src_orig": True,
        })
    for i in range(400):
        rows.append({
            "sl": "en",
            "tl": "fr",
            "id": i,
            "src": "x",
            "trg": f"f{i}",
            "is_src_orig": True,
        })
    for i in range(50):
        rows.append({
            "sl": "en",
            "tl": "ber-Latn",
            "id": i,
            "src": "x",
            "trg": f"b{i}",
            "is_src_orig": True,
        })
    raw = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

    def _fake_load(*_args: Any, **_kwargs: Any) -> Dataset:
        return raw

    import datasets as datasets_mod

    monkeypatch.setattr(datasets_mod, "load_dataset", _fake_load)
    out = SmolSentDataset().build_from_source()
    counts = pd.Series(out["tl_iso693_3"]).value_counts()
    assert set(counts.index) == {"eng", "fra"}
    assert counts["eng"] == 300
    assert counts["fra"] == 300


def test_bibles_build_pipeline(tmp_path: Path) -> None:
    """BiblesDataset.build_from_source reads a TSV, conforms codes, samples per class."""
    from commonlid.datasets.bibles import BiblesDataset

    rows: list[tuple[str, str]] = []
    for i in range(1500):
        rows.append(("eng_Latn", f"e{i}"))
    for i in range(1200):
        rows.append(("deu_Latn", f"d{i}"))
    for i in range(500):
        rows.append(("fra_Latn", f"f{i}"))  # below min_count=1000

    tsv = tmp_path / "bibles.tsv"
    tsv.write_text("\n".join(f"{lang}\t{text}" for lang, text in rows))

    out = BiblesDataset().build_from_source(source_path=tsv)
    counts = pd.Series(out["lang_iso639_3"]).value_counts()
    assert set(counts.index) == {"eng", "deu"}
    assert counts["eng"] == 300
    assert counts["deu"] == 300


def test_bibles_build_without_path_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from commonlid.datasets.bibles import BiblesDataset

    monkeypatch.delenv("COMMONLID_BIBLES_RAW_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="bibles_with_lang_labels"):
        BiblesDataset().build_from_source()


def test_bibles_build_with_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from commonlid.datasets.bibles import BiblesDataset

    rows = [("eng_Latn", f"e{i}") for i in range(1100)] + [
        ("deu_Latn", f"d{i}") for i in range(1100)
    ]
    tsv = tmp_path / "bibles.tsv"
    tsv.write_text("\n".join(f"{lang}\t{text}" for lang, text in rows))
    monkeypatch.setenv("COMMONLID_BIBLES_RAW_PATH", str(tsv))

    out = BiblesDataset().build_from_source()
    assert len(out) == 600  # 300 per surviving class * 2 classes


def test_bibles_now_uses_commonlid_cache_repo() -> None:
    from commonlid.datasets.bibles import BiblesDataset

    assert BiblesDataset.cache_hf_repo == "commoncrawl/commonlid-cache_bibles_300_iso639_3"
    assert BiblesDataset.source_hf_repo is None
