from __future__ import annotations

import pytest

pytest.importorskip("pyfranc")

from commonlid.models.pyfranc import PyfrancModel


def test_returns_top_code() -> None:
    m = PyfrancModel()
    # pyfranc needs longer text to be confident.
    preds = m.predict([
        "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow."
    ])
    # PyfrancModel is expected to emit an ISO 639-3-ish code (e.g. 'eng').
    assert preds[0] is None or len(preds[0]) in (2, 3)
