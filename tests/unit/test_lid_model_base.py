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


def test_predict_scored_defaults_to_a_none_score() -> None:
    """A model returning bare codes reports no confidence."""

    class Bare(LIDModel):
        model_id = "_bare"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
            return ["en"] * len(texts)

    assert Bare().predict_scored(["anything"]) == [LIDPrediction("eng", None)]


def test_predict_scored_keeps_a_reported_confidence() -> None:
    class Scored(LIDModel):
        model_id = "_scored"

        def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
            return [LIDPrediction("jw", 0.75)] * len(texts)

    # The raw code is still conformed; only the score passes through untouched.
    assert Scored().predict_scored(["anything"]) == [LIDPrediction("jav", 0.75)]


def test_predict_scored_keeps_the_score_when_the_code_does_not_conform() -> None:
    """A confident prediction of an unmappable code keeps its confidence."""

    class Unmappable(LIDModel):
        model_id = "_unmappable"

        def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
            return [LIDPrediction("xxxx", 0.99)] * len(texts)

    assert Unmappable().predict_scored(["anything"]) == [LIDPrediction(None, 0.99)]


def test_predict_drops_the_score() -> None:
    """`predict` stays a list of bare codes."""

    class Scored(LIDModel):
        model_id = "_scored2"

        def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
            return [LIDPrediction("en", 0.5)] * len(texts)

    assert Scored().predict(["anything"]) == ["eng"]


def test_predict_scored_accepts_a_mixed_batch() -> None:
    """Backends may return bare codes and scored predictions side by side."""

    class Mixed(LIDModel):
        model_id = "_mixed"

        def _predict_batch(self, texts: Sequence[str]) -> list[str | LIDPrediction | None]:
            return ["en", LIDPrediction("de", 0.5), None]

    assert Mixed().predict_scored(["a", "b", "c"]) == [
        LIDPrediction("eng", None),
        LIDPrediction("deu", 0.5),
        LIDPrediction(None, None),
    ]


def test_lid_prediction_is_frozen() -> None:
    p = LIDPrediction(iso639_3="eng", score=0.9)
    assert p.iso639_3 == "eng"
    assert p.score == 0.9


def test_load_is_idempotent() -> None:
    model = _Echo()
    model.load()
    model.load()
    model.predict(["hi"])  # should not re-trigger load failure
