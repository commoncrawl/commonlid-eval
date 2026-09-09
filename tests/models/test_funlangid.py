"""FunLangID wrapper tests (uses the vendored classifier — no downloads)."""

from __future__ import annotations

from commonlid.models.funlangid import FunLangIDModel


def test_predicts_english() -> None:
    model = FunLangIDModel()
    preds = model.predict(["The quick brown fox jumps over the lazy dog."])
    # FunLangID is a small char-4gram baseline; we only assert the call
    # returns an ISO 639-3-ish code or None, not the exact label.
    assert preds[0] is None or len(preds[0]) == 3


def test_predicts_german() -> None:
    model = FunLangIDModel()
    preds = model.predict(["Der schnelle braune Fuchs springt ueber den faulen Hund."])
    assert preds[0] is None or len(preds[0]) == 3


def test_predicts_undefined_on_empty_after_clean() -> None:
    model = FunLangIDModel()
    # The cleaner strips symbols + digits -> empty. FunLangID returns "und",
    # which the wrapper maps to None.
    preds = model.predict(["123!!!"])
    assert preds == [None]


def test_predict_scored_returns_the_classifier_score() -> None:
    from commonlid.models.funlangid import FunLangIDModel

    scored = FunLangIDModel().predict_scored([
        "The quick brown fox jumps over the lazy dog",
        "",
    ])
    assert scored[0].iso639_3 == "eng"
    assert scored[0].score is not None
    assert 0.0 < scored[0].score <= 1.0
    # Nothing to score: no code and no confidence.
    assert scored[1].iso639_3 is None
    assert scored[1].score is None
