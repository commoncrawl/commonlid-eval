"""Private-cache datasets surface a friendly error on HF auth/404 failures."""

from __future__ import annotations

from typing import Any

import pytest

from commonlid.core.lid_dataset import LIDDataset, PrivateDatasetAccessError


class _PrivateDS(LIDDataset):
    dataset_id = "_private_test"
    cache_hf_repo = "fake-org/fake-private"
    cache_hf_split = "train"
    text_column = "text"
    target_column = "lang"
    is_cache_private = True


class _PublicCacheDS(LIDDataset):
    dataset_id = "_public_cache_test"
    cache_hf_repo = "fake-org/fake-public"
    cache_hf_split = "train"
    text_column = "text"
    target_column = "lang"


def _patch_load_dataset(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    import datasets as datasets_mod

    monkeypatch.setattr(datasets_mod, "load_dataset", _raise)


def _make_hf_error(cls: type, status_code: int) -> Exception:
    """Build an ``HfHubHTTPError`` (or subclass) without going through __init__.

    HfHubHTTPError's constructor pulls a lot of attributes off ``response``
    (headers, url, request, ...). We just need an instance of the right class
    with a ``response.status_code`` attribute, so we bypass __init__ entirely.
    """
    from types import SimpleNamespace

    exc = cls.__new__(cls)
    Exception.__init__(exc, f"http {status_code}")
    exc.response = SimpleNamespace(status_code=status_code)  # type: ignore[attr-defined]
    return exc


def test_private_dataset_translates_gated_repo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import GatedRepoError

    _patch_load_dataset(monkeypatch, _make_hf_error(GatedRepoError, 403))
    with pytest.raises(PrivateDatasetAccessError) as excinfo:
        _PrivateDS().load()
    assert "fake-org/fake-private" in str(excinfo.value)
    assert "huggingface-cli login" in str(excinfo.value)
    assert excinfo.value.dataset_id == "_private_test"
    assert excinfo.value.cache_hf_repo == "fake-org/fake-private"


def test_private_dataset_translates_repository_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    _patch_load_dataset(monkeypatch, _make_hf_error(RepositoryNotFoundError, 404))
    with pytest.raises(PrivateDatasetAccessError):
        _PrivateDS().load()


def test_private_dataset_translates_401(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import HfHubHTTPError

    _patch_load_dataset(monkeypatch, _make_hf_error(HfHubHTTPError, 401))
    with pytest.raises(PrivateDatasetAccessError):
        _PrivateDS().load()


def test_private_dataset_passes_through_unrelated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load_dataset(monkeypatch, ValueError("totally unrelated"))
    with pytest.raises(ValueError, match="totally unrelated"):
        _PrivateDS().load()


def test_public_cache_dataset_does_not_translate(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub.errors import GatedRepoError

    _patch_load_dataset(monkeypatch, _make_hf_error(GatedRepoError, 403))
    with pytest.raises(GatedRepoError):
        _PublicCacheDS().load()


def test_shipped_private_datasets_have_flag_set() -> None:
    from commonlid.datasets.bibles import BiblesDataset
    from commonlid.datasets.smolsent import SmolSentDataset
    from commonlid.datasets.social_media import SocialMediaDataset

    assert BiblesDataset.is_cache_private is True
    assert SmolSentDataset.is_cache_private is True
    assert SocialMediaDataset.is_cache_private is True


def test_public_datasets_default_to_public_cache() -> None:
    from commonlid.datasets.commonlid import CommonLIDDataset
    from commonlid.datasets.flores_dev import FloresDevDataset
    from commonlid.datasets.udhr import UDHRDataset

    assert CommonLIDDataset.is_cache_private is False
    assert FloresDevDataset.is_cache_private is False
    assert UDHRDataset.is_cache_private is False
