# Convenience wrappers around the `uv` workflow used by CI and CONTRIBUTING.md.
# Run `make help` to list every target.

# Optional Python pin. Leave unset locally to use uv's default; CI sets this
# per matrix row (e.g. `make install PYTHON=3.10`) so the dependency sync
# resolves against the right interpreter.
PYTHON ?=
PYTHON_FLAG := $(if $(PYTHON),--python $(PYTHON),)

SRC_DIRS := src tests
PACKAGE := src/commonlid

.DEFAULT_GOAL := help

.PHONY: help venv \
        install install-all install-afrolid install-notebooks install-leaderboard \
        lint format format-check typecheck \
        test test-slow test-all check \
        build clean \
        notebooks leaderboard leaderboard-upload

help:
	@echo "Targets:"
	@echo "  venv                  Create a uv-managed virtualenv (.venv)"
	@echo "  install               Sync runtime + dev extras (lint/type/test)"
	@echo "  install-afrolid       install + the heavy [afrolid] extra (torch + transformers)"
	@echo "  install-notebooks     install + the [notebooks] extra (jupyterlab + matplotlib)"
	@echo "  install-leaderboard   install + the [leaderboard] extra (gradio)"
	@echo "  install-all           install + every optional extra"
	@echo ""
	@echo "  lint                  ruff check"
	@echo "  format                ruff format (rewrite in place)"
	@echo "  format-check          ruff format --check (CI variant)"
	@echo "  typecheck             mypy strict on the package"
	@echo "  test                  pytest (fast suite + coverage; ~2s)"
	@echo "  test-slow             pytest -m slow (downloads weights, ~5 min)"
	@echo "  test-all              pytest including slow + network markers"
	@echo "  check                 lint + format-check + typecheck + test"
	@echo ""
	@echo "  build                 uv build (sdist + wheel)"
	@echo "  notebooks             jupyter lab notebooks/paper_tables.ipynb"
	@echo "  leaderboard           Serve the Gradio leaderboard from ./data/results"
	@echo "  leaderboard-upload    Open a PR on commoncrawl/commonlid-results from ./data/results"
	@echo "  clean                 Remove build artefacts and tool caches"
	@echo ""
	@echo "Override the Python version with PYTHON=3.x (e.g. make install PYTHON=3.10)."

venv:
	uv venv $(PYTHON_FLAG)

install:
	uv sync --extra dev $(PYTHON_FLAG)

install-afrolid:
	uv sync --extra dev --extra afrolid $(PYTHON_FLAG)

install-notebooks:
	uv sync --extra dev --extra notebooks $(PYTHON_FLAG)

install-leaderboard:
	uv sync --extra dev --extra leaderboard $(PYTHON_FLAG)

install-all:
	uv sync --extra dev --extra afrolid --extra notebooks --extra leaderboard $(PYTHON_FLAG)

lint:
	uv run ruff check $(SRC_DIRS)

format:
	uv run ruff format $(SRC_DIRS)

format-check:
	uv run ruff format --check $(SRC_DIRS)

typecheck:
	uv run mypy $(PACKAGE)

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

test-all:
	uv run pytest -m "slow or network or not slow"

check: lint format-check typecheck test

build:
	uv build

notebooks: install-notebooks
	uv run jupyter lab notebooks/paper_tables.ipynb

# Leaderboard helpers. Override LEADERBOARD_DIR / LEADERBOARD_REPO at the
# command line if you want to point at a different results tree or HF repo.
LEADERBOARD_DIR ?= ./data/results
LEADERBOARD_REPO ?= commoncrawl/commonlid-results

leaderboard: install-leaderboard
	uv run commonlid leaderboard serve --local-dir $(LEADERBOARD_DIR)

leaderboard-upload: install-leaderboard
	uv run commonlid leaderboard upload \
	  --repo-id $(LEADERBOARD_REPO) \
	  --local-dir $(LEADERBOARD_DIR)

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
