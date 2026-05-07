"""Smoke-test the deployed HF Space entrypoint (`hf-space/app.py`).

The existing leaderboard tests exercise ``build_app`` directly inside the
``commonlid`` package. They don't catch:

- Regressions in the Space's own glue (``hf-space/app.py``).
- Gradio's *internal* import chain breaking — e.g. the ``HfFolder``
  ``ImportError`` from gradio 4.x against ``huggingface-hub>=1.0`` that
  took the production Space down. ``pytest.importorskip("gradio")``
  silently skips on any ``ImportError``, so an internally-broken gradio
  would have produced a green CI run.

This test loads ``hf-space/app.py`` exactly as the Space runtime does —
as a top-level module, not via the package — with the data layer patched
to a local directory so we never hit the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import pytest


def _gradio_installed() -> bool:
    """True when gradio is installed; orthogonal to whether it imports cleanly.

    Using package metadata (not ``import gradio``) lets the test *fail* on
    a broken gradio install rather than skip silently.
    """
    try:
        _pkg_version("gradio")
    except PackageNotFoundError:
        return False
    return True


SPACE_APP_PATH = Path(__file__).resolve().parents[2] / "hf-space" / "app.py"


def _write_minimal_summary(root: Path, dataset_id: str, model_id: str) -> None:
    out = root / dataset_id / model_id
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 2,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "dataset_revision": "test",
        "commonlid_version": "test-0.0",
        "timestamp": "2026-05-05T00:00:00",
        "n_samples": 1,
        "n_samples_with_gold": 1,
        "samples_per_second": 1.0,
        "macro": {
            "precision_gold_only": 1.0,
            "recall_gold_only": 1.0,
            "f1_gold_only": 1.0,
            "n_languages_gold": 1,
            "precision_observed": 1.0,
            "recall_observed": 1.0,
            "f1_observed": 1.0,
            "n_languages_observed": 1,
        },
        "micro": {
            "precision_gold_only": 1.0,
            "recall_gold_only": 1.0,
            "f1_gold_only": 1.0,
            "n_correct_gold": 1,
            "n_gold_samples": 1,
            "n_predictions_gold": 1,
            "precision_observed": 1.0,
            "recall_observed": 1.0,
            "f1_observed": 1.0,
            "n_correct_observed": 1,
            "n_predictions_observed": 1,
        },
        "per_language": {
            "eng": {
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "fpr": 0.0,
                "gt_count": 1,
                "predictions": 1,
                "correct": 1,
            }
        },
        "extra": {},
    }
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _load_space_module(name: str) -> Any:
    """Import ``hf-space/app.py`` as a top-level module, mirroring the Space.

    The Space runtime executes ``app.py`` directly from the repo root, not
    as part of any package, so we use ``spec_from_file_location`` to match
    that behaviour exactly.
    """
    spec = importlib.util.spec_from_file_location(name, SPACE_APP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover -- defensive
        msg = f"could not load module spec for {SPACE_APP_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.mark.skipif(not _gradio_installed(), reason="gradio extra not installed")
def test_space_app_py_boots_and_returns_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading hf-space/app.py must produce a working ``demo`` Blocks instance.

    Regression guard for:
      - Gradio's transitive import chain (e.g. the HfFolder/huggingface-hub
        1.0 break that hit production).
      - Drift between the Space's tiny glue file and the package's
        ``build_app`` signature.
    """
    import gradio as gr

    from commonlid.leaderboard import app as app_module

    # Populate a local results tree so build_app finds at least one summary
    # and doesn't go down the empty-frame branch.
    _write_minimal_summary(tmp_path, "commonlid", "GlotLID")

    # Redirect the network-bound build_app to read from tmp_path instead of
    # snapshot_download'ing the live HF dataset. This is the only place the
    # Space's app.py touches the network.
    real_build_app = app_module.build_app

    def _build_app_local(**kwargs: Any) -> Any:
        kwargs.setdefault("local_dir", tmp_path)
        return real_build_app(**kwargs)

    monkeypatch.setattr(app_module, "build_app", _build_app_local)

    module = _load_space_module("commonlid_space_app_under_test")
    try:
        assert hasattr(module, "demo"), "Space app.py must define a top-level `demo`"
        assert isinstance(module.demo, gr.Blocks), (
            f"`demo` must be a gradio.Blocks instance, got {type(module.demo).__name__}"
        )
        # The HfFolder ImportError surfaces inside gradio.oauth at gradio
        # *import* time, well before our Blocks-construction code runs;
        # touching `gr.Blocks` above is what makes this test sensitive to it.
    finally:
        sys.modules.pop("commonlid_space_app_under_test", None)


@pytest.mark.skipif(not _gradio_installed(), reason="gradio extra not installed")
def test_gradio_imports_cleanly() -> None:
    """``import gradio`` must succeed when the package is installed.

    Direct guard for the HfFolder-style failure mode: ``pytest.importorskip``
    swallows ``ImportError`` and would mark the test skipped, hiding a
    broken-gradio production environment behind a green CI badge. This test
    only skips when gradio is genuinely *missing* (``_gradio_installed`` is
    False); a present-but-broken install raises and fails the suite.
    """
    import gradio as gr

    assert hasattr(gr, "Blocks")
    assert hasattr(gr, "Tabs")
    assert hasattr(gr, "Dataframe")
