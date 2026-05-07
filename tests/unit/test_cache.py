from __future__ import annotations

from commonlid.evaluation.cache import PredictionCache


def test_put_and_get(tmp_path) -> None:
    cache = PredictionCache(cache_dir=tmp_path, model_id="m", dataset_id="d", dataset_revision="r1")
    hit, _ = cache.get("hello")
    assert hit is False
    cache.put("hello", "eng")
    hit, pred = cache.get("hello")
    assert hit is True
    assert pred == "eng"


def test_persistence_across_instances(tmp_path) -> None:
    c1 = PredictionCache(tmp_path, "m", "d", "r")
    c1.put_many([("t1", "eng"), ("t2", None)])
    c2 = PredictionCache(tmp_path, "m", "d", "r")
    assert c2.get("t1") == (True, "eng")
    assert c2.get("t2") == (True, None)
    assert len(c2) == 2


def test_revision_scopes_the_cache(tmp_path) -> None:
    c1 = PredictionCache(tmp_path, "m", "d", "revA")
    c1.put("hello", "eng")
    c2 = PredictionCache(tmp_path, "m", "d", "revB")
    hit, _ = c2.get("hello")
    assert hit is False


def test_model_id_scopes_the_cache(tmp_path) -> None:
    c1 = PredictionCache(tmp_path, "m1", "d", "r")
    c1.put("hello", "eng")
    c2 = PredictionCache(tmp_path, "m1", "d", "r")
    assert c2.get("hello") == (True, "eng")
    c3 = PredictionCache(tmp_path, "m2", "d", "r")
    # Because the key hashes model_id, same text lives under a different key.
    hit, _ = c3.get("hello")
    assert hit is False


def test_path_property(tmp_path) -> None:
    cache = PredictionCache(tmp_path, "m", "d", "r")
    assert cache.path == tmp_path / "d" / "m.jsonl"
