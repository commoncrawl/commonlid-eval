# Contributing

Thanks for considering a contribution to `commonlid`. This document
covers the local dev workflow, the quality gates CI enforces, how to
add a new README-level example, and how releases are cut.

Before changing user-visible behaviour, skim `docs/architecture.md` for
the package layout.

## Dev environment

The project supports Python 3.10, 3.11, 3.12, and 3.13 (CI runs the
test suite on every interpreter). It is managed with
[uv](https://docs.astral.sh/uv/), and the common workflows are wrapped
in a `Makefile` so local dev and CI run identical commands.

```bash
git clone https://github.com/commoncrawl/commonlid-eval.git
cd commonlid-eval
make install        # uv sync --extra dev (ruff + mypy + pytest + DSPy + Azure + cld3-py + gradio)
```

The `dev` extra pulls in `dspy` and `azure-identity` so the DSPy test
paths execute against mocked transports rather than skipping. It also
pulls in `cld3-py` and `gradio` so every shipped wrapper can be
exercised by the test suite.

For notebook work add the `notebooks` extra:

```bash
make install-notebooks
make notebooks      # uv run jupyter lab notebooks/paper_tables.ipynb
```

The AfroLID wrapper needs its own heavy extra
(`make install-afrolid`) because `torch` + `transformers` are several
hundred MB. The Gradio leaderboard preview lives behind
`make install-leaderboard` (`leaderboard` extra).

To pin a specific interpreter when syncing (e.g. to reproduce the CI
floor), pass `PYTHON=3.x`:

```bash
make install PYTHON=3.10
```

`make help` lists every available target.

## Quality gates

Local commands must be green before opening a PR. CI
(`.github/workflows/ci.yml`) calls the exact same `make` targets.

```bash
make lint           # uv run ruff check src tests
make format-check   # uv run ruff format --check src tests   (use `make format` to fix)
make typecheck      # uv run mypy src/commonlid              (strict)
make test           # uv run pytest                          (fast suite + coverage; ~2s)
```

`make check` chains the four together if you want a single command
before pushing.

For the full parity smoke test (downloads GlotLID weights + a slice of
UDHR / CommonLID):

```bash
make test-slow      # uv run pytest -m slow                  (~5 min)
```

Coverage threshold is **90 %** (`pytest --cov-fail-under=90` runs by
default). Current coverage sits around **96 %**. If a change drops
coverage below 90 %, either add tests or — for thin wrappers over
optional extras — update the `omit` list in `pyproject.toml` with a
short rationale comment.

## Adding a README example

The README is executable: every Python code block tagged with an HTML
`readme-test` marker is run as a pytest case
(`tests/integration/test_readme_examples.py`). Markers:

- `<!-- readme-test: fast; id=my-example -->` — runs in the default
  CI test matrix; examples should be self-contained and **not** need
  network or downloaded weights. Prefer `cld2` / `pyfranc` over
  fasttext models.
- `<!-- readme-test: slow; id=... -->` — only runs under `pytest -m
  slow`; fine for examples that load GlotLID weights or hit HF Hub.
- `<!-- readme-test: skip; id=... -->` — parsed so the sentinel test
  sees the block but never executed. Use for things that require
  real credentials (e.g. the DSPy LLM Azure example).

Examples that load previous results get a `./results/` directory
pre-seeded by the `_populate_results` fixture in the test module.

Prefer examples with `assert`s over print-style blocks — silent drift
in behaviour is the main failure mode `readme-test: fast` is designed
to catch.

## Adding a new model

Create `src/commonlid/models/my_model.py`:

```python
from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import register_model


@register_model
class MyModel(LIDModel):
    model_id = "my_model"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        return ["eng"] * len(texts)  # one ISO 639-3 code (or None) per input
```

Then wire the side-effect import into `src/commonlid/models/__init__.py`:

```python
from commonlid.models import my_model as _my_model  # noqa: F401
```

Add a unit test under `tests/models/`. If the model has a discoverable
language list, override `discover_supported_languages()` so the
`commonlid generate-support-matrix` CLI can include it (see
`_fasttext_base.py` or `cld2.py` for examples).

## Adding a new dataset

Create `src/commonlid/datasets/my_task.py`. At least one of
`source_hf_repo` (the canonical public dataset) or `cache_hf_repo`
(an optional pre-built/sampled artifact) must be set; for
reproducibility, always pin a full git SHA on whichever revision
field corresponds to the repo `load()` will actually read:

```python
from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import register_dataset


@register_dataset
class MyTask(LIDDataset):
    dataset_id = "my_task"
    source_hf_repo = "me/my-lid-dataset"
    source_hf_revision = "abcdef1234567890abcdef1234567890abcdef12"  # full SHA
    source_hf_split = "test"
    text_column = "text"
    target_column = "iso639_3"
```

If the dataset ships only as a private/preprocessed cache, set
`cache_hf_repo` / `cache_hf_revision` instead (and `is_cache_private = True`
when access is gated). See `src/commonlid/datasets/bibles.py` for that
pattern, including `build_from_source()` for the public-source fallback.

Import it from `src/commonlid/datasets/__init__.py` so the decorator fires.

If the gold labels aren't already ISO 639-3, run a preprocessing step
before publishing the dataset — the evaluator validates (but does not
rewrite) gold codes via `LIDDataset._check_gold_conformity()`.

## PR workflow

- Branch from `main`.
- Keep commits focused; prefer a conventional-commits-style prefix
  (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`)
  so release notes auto-generate cleanly.
- Open a PR against `main`; the `lint-type-test` workflow must pass.
- Numeric-parity-affecting changes should re-run
  `pytest -m slow` locally and include the result in the PR
  description (the smoke parity tests assert 1e-6 F1 agreement with
  the frozen legacy pipeline).

## Releasing

Releases are cut by a manually-triggered GitHub Actions workflow at
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). It
takes two inputs:

- `version_bump` (required) — `patch` / `minor` / `major`. Ignored on
  dry runs.
- `dry_run` (checkbox, default off) — see below.

### Full release

Leave `dry_run` unchecked and pick a bump level, either in the Actions
tab or from the CLI:

```bash
gh workflow run publish.yml -f version_bump=patch
```

The workflow then:

1. Bumps the version in `pyproject.toml` + `uv.lock` via `uv version
   --bump <level>`, commits the bump to the current branch, and
   pushes a matching `vX.Y.Z` tag.
2. Builds the sdist + wheel with `uv build`.
3. Publishes to **TestPyPI** under the `testpypi` GitHub Environment.
4. Publishes to **PyPI** under the `pypi` GitHub Environment, gated
   on TestPyPI succeeding.
5. Creates a GitHub Release on the tag with the built artefacts
   attached and auto-generated release notes.

### Test deployment (dry run)

Tick the `dry_run` checkbox (or pass `-f dry_run=true`) to rehearse
the publish pipeline against TestPyPI without touching `main`,
production PyPI, or GitHub Releases:

```bash
gh workflow run publish.yml -f version_bump=patch -f dry_run=true
```

In this mode the workflow:

- Reads the current version from `pyproject.toml` (e.g. `0.2.0`),
  appends `.dev${GITHUB_RUN_NUMBER}` to produce a unique PEP 440
  pre-release (e.g. `0.2.0.dev42`), and writes it back to
  `pyproject.toml` inside the runner only.
- **Does not** commit, tag, or push anything — the main branch stays
  untouched.
- Builds sdist + wheel and uploads only to **TestPyPI** (with
  `skip-existing: true`, so a rerun is idempotent).
- Skips the `publish-pypi` and `github-release` jobs entirely.

Each rehearsal gets a fresh dev suffix from the GitHub Actions run
number, so you can iterate freely without hitting TestPyPI's
"can't re-upload the same version" rule. `version_bump` is still
required by the workflow form but has no effect in dry-run mode.

Install a dry-run artefact to smoke-test it before cutting the real
release:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            commonlid==0.2.0.dev42
```

### One-time prerequisites

- Configure two *pending* trusted publishers on the PyPI side — one
  on [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
  and one on [pypi.org](https://pypi.org/manage/account/publishing/) —
  both pointing at `commoncrawl/commonlid-eval` with workflow name
  `publish.yml` and environment names `testpypi` and `pypi`
  respectively.
- Create the `testpypi` and `pypi` environments under the repo's
  *Settings → Environments* (add required reviewers for `pypi` if you
  want a human gate before production publishes).

No PyPI API tokens are stored in the repo — the workflow uses OIDC
via PyPI Trusted Publishing for authentication.

### Verifying a release

After the workflow completes:

```bash
pip install commonlid==X.Y.Z
commonlid version             # should print X.Y.Z
commonlid list-models         # sanity-check registrations
```

If something went wrong after the TestPyPI publish but before PyPI,
yank the TestPyPI release and re-trigger the workflow with a fresh
patch bump.
