from __future__ import annotations

from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel, LIDPrediction


class _Echo(LIDModel):
    """Tiny model that echoes the first word of each normed text as a fake code."""

    model_id = "_echo"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return [(t.split()[0] if t else None) for t in texts]


def test_predict_applies_preprocessing_by_default() -> None:
    # After the normer: "Hola mundo!" -> "hola mundo"; first token "hola" is NOT
    # an ISO 639 code so _conform maps it to None.
    assert _Echo().predict(["Hola mundo!"]) == [None]


def test_predict_can_skip_preprocessing() -> None:
    class Raw(LIDModel):
        model_id = "_raw"
        requires_preprocessing = False

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["en"] * len(texts)

    # "en" is a valid ISO 639-1 code -> conformed to ISO 639-3 "eng".
    assert Raw().predict(["anything"]) == ["eng"]


def test_predict_conforms_langcode() -> None:
    class JwModel(LIDModel):
        model_id = "_jw"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["jw"] * len(texts)

    # 'jw' -> 'jav' via the deprecation table, then Lang('jav').pt3 == 'jav'.
    assert JwModel().predict(["irrelevant"]) == ["jav"]


def test_predict_maps_iso639_1_to_pt3() -> None:
    class EnModel(LIDModel):
        model_id = "_en"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["en"] * len(texts)

    assert EnModel().predict(["anything"]) == ["eng"]


def test_predict_drops_unknown_codes() -> None:
    class BadModel(LIDModel):
        model_id = "_bad"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["xxxx"] * len(texts)

    assert BadModel().predict(["anything"]) == [None]


def test_predict_handles_none_output() -> None:
    class UndModel(LIDModel):
        model_id = "_und"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return [None] * len(texts)

    assert UndModel().predict(["foo"]) == [None]


def test_supports_without_declared_list_is_true() -> None:
    assert _Echo().supports("eng") is True


def test_supports_with_declared_list() -> None:
    class Limited(_Echo):
        model_id = "_lim"
        supported_languages = frozenset({"eng", "deu"})

    lim = Limited()
    assert lim.supports("eng") is True
    assert lim.supports("fra") is False


def test_lid_prediction_is_frozen() -> None:
    p = LIDPrediction(iso639_3="eng", score=0.9)
    assert p.iso639_3 == "eng"
    assert p.score == 0.9


def test_load_is_idempotent() -> None:
    model = _Echo()
    model.load()
    model.load()
    model.predict(["hi"])  # should not re-trigger load failure
