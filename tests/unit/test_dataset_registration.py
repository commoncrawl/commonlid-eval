"""Every shipped dataset module registers the expected dataset id."""

from __future__ import annotations

import importlib

import pytest

from commonlid.core.registry import list_datasets


@pytest.fixture(autouse=True)
def _import_datasets() -> None:
    importlib.import_module("commonlid.datasets")


EXPECTED_DATASET_IDS = {
    "bibles_300",
    "bibles_300_nano",
    "commonlid",
    "commonlid_nano",
    "flores_dev",
    "flores_dev_nano",
    "smolsent_300",
    "smolsent_300_nano",
    "social_media_300",
    "social_media_300_nano",
    "udhr",
    "udhr_nano",
}


def test_expected_datasets_registered() -> None:
    assert set(list_datasets()) == EXPECTED_DATASET_IDS


def test_commonlid_metadata() -> None:
    from commonlid.datasets.commonlid import CommonLIDDataset

    assert CommonLIDDataset.source_hf_repo == "commoncrawl/CommonLID"
    assert CommonLIDDataset.cache_hf_repo is None
    assert CommonLIDDataset.text_column == "text"
    assert CommonLIDDataset.target_column == "tag"


def test_flores_has_pinned_revision() -> None:
    from commonlid.datasets.flores_dev import FloresDevDataset

    assert FloresDevDataset.source_hf_revision is not None
    assert len(FloresDevDataset.source_hf_revision) == 40  # full git SHA


def test_every_registered_dataset_has_metadata_set() -> None:
    """Every shipped class declares title/description/reference_url + a license."""
    from commonlid.core.lid_dataset import LIDDataset
    from commonlid.core.registry import _DATASET_REGISTRY

    for dataset_id, cls in _DATASET_REGISTRY.items():
        assert issubclass(cls, LIDDataset)
        assert cls.title, f"{dataset_id} missing title"
        assert cls.description, f"{dataset_id} missing description"
        assert cls.reference_url, f"{dataset_id} missing reference_url"
        assert cls.license_name, f"{dataset_id} missing license_name"
        assert cls.license_name == cls.license_name.lower(), (
            f"{dataset_id} license_name must be lowercase, got {cls.license_name!r}"
        )
        # license_url is optional but, when set, must look like an http(s) URL.
        if cls.license_url is not None:
            assert cls.license_url.startswith(("http://", "https://")), (
                f"{dataset_id} license_url should be an http(s) URL, got {cls.license_url!r}"
            )
        assert cls.main_score == "macro_f1", (
            f"{dataset_id} main_score should be macro_f1 (got {cls.main_score!r})"
        )


def test_nano_inherits_metadata_from_parent() -> None:
    """Nano variants inherit license + reference_url from their parent class."""
    from commonlid.datasets.bibles import BiblesDataset
    from commonlid.datasets.commonlid import CommonLIDDataset
    from commonlid.datasets.nano import BiblesDatasetNano, CommonLIDDatasetNano

    assert CommonLIDDatasetNano.license_name == CommonLIDDataset.license_name
    assert CommonLIDDatasetNano.license_url == CommonLIDDataset.license_url
    assert CommonLIDDatasetNano.reference_url == CommonLIDDataset.reference_url
    assert BiblesDatasetNano.license_name == BiblesDataset.license_name
    assert "(nano)" in CommonLIDDatasetNano.title
    assert "Nano slice" in CommonLIDDatasetNano.description
