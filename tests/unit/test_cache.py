from __future__ import annotations

import json

from commonlid.core.lid_model import LIDPrediction
from commonlid.evaluation.cache import PredictionCache


def test_put_and_get(tmp_path) -> None:
    cache = PredictionCache(cache_dir=tmp_path, model_id="m", dataset_id="d", dataset_revision="r1")
    hit, _ = cache.get("hello")
    assert hit is False
    cache.put("hello", LIDPrediction("eng", 0.9))
    hit, pred = cache.get("hello")
    assert hit is True
    assert pred == LIDPrediction("eng", 0.9)


def test_persistence_across_instances(tmp_path) -> None:
    c1 = PredictionCache(tmp_path, "m", "d", "r")
    c1.put_many([("t1", LIDPrediction("eng", 0.5)), ("t2", LIDPrediction(None, None))])
    c2 = PredictionCache(tmp_path, "m", "d", "r")
    assert c2.get("t1") == (True, LIDPrediction("eng", 0.5))
    assert c2.get("t2") == (True, LIDPrediction(None, None))
    assert len(c2) == 2


def test_entries_written_before_scores_load_with_a_none_score(tmp_path) -> None:
    """Caches written by an older commonlid have no ``score`` key."""
    path = tmp_path / "d" / "m.jsonl"
    path.parent.mkdir(parents=True)
    legacy = PredictionCache(tmp_path, "m", "d", "r")
    key = next(iter(legacy._store), None)
    assert key is None  # nothing there yet; build the legacy line by hand
    legacy.put("t1", LIDPrediction("eng", 0.5))
    (key,) = legacy._store
    path.write_text(json.dumps({"text_hash": key, "pred": "eng"}) + "\n", encoding="utf-8")

    reloaded = PredictionCache(tmp_path, "m", "d", "r")
    assert reloaded.get("t1") == (True, LIDPrediction("eng", None))


def test_revision_scopes_the_cache(tmp_path) -> None:
    c1 = PredictionCache(tmp_path, "m", "d", "revA")
    c1.put("hello", LIDPrediction("eng", None))
    c2 = PredictionCache(tmp_path, "m", "d", "revB")
    hit, _ = c2.get("hello")
    assert hit is False


def test_model_id_scopes_the_cache(tmp_path) -> None:
    c1 = PredictionCache(tmp_path, "m1", "d", "r")
    c1.put("hello", LIDPrediction("eng", None))
    c2 = PredictionCache(tmp_path, "m1", "d", "r")
    assert c2.get("hello") == (True, LIDPrediction("eng", None))
    c3 = PredictionCache(tmp_path, "m2", "d", "r")
    # Because the key hashes model_id, same text lives under a different key.
    hit, _ = c3.get("hello")
    assert hit is False


def test_path_property(tmp_path) -> None:
    cache = PredictionCache(tmp_path, "m", "d", "r")
    assert cache.path == tmp_path / "d" / "m.jsonl"
