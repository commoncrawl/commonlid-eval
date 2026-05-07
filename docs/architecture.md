# commonlid — Architecture

This is a short orientation doc. For the API surface and how-to, see the
top-level `README.md`.

## Packages

```
src/commonlid/
├── core/               # Abstract base classes + decorator registry
│   ├── lid_model.py        # LIDModel ABC, LIDPrediction, _conform pipeline
│   ├── lid_dataset.py      # LIDDataset ABC, pinned HF revision, iter_batches
│   └── registry.py         # @register_model / @register_dataset
├── models/                 # One submodule per LID model; auto-registered
│   ├── _fasttext_base.py   # Shared HF-fasttext plumbing (GlotLID/OpenLID-v2/fasttext)
│   ├── cld2.py, cld3.py, glotlid.py, openlidv2.py, fasttext_ft.py,
│   │ pyfranc.py, afrolid.py, funlangid.py
│   └── dspy_llm.py         # DSPyLLMModel — NOT auto-registered (per-instance config)
├── datasets/                  # One submodule per evaluation dataset; auto-registered
│   ├── commonlid.py        # commoncrawl/CommonLID
│   ├── flores_dev.py, udhr.py, bibles.py, smolsent.py, social_media.py
│   └── nano.py             # Stratified-sample variants of every full dataset
├── evaluation/
│   ├── evaluator.py        # cartesian product (models x datasets) → files
│   ├── results.py          # Result dataclass, summary.json, predictions.jsonl
│   └── cache.py            # Per-(model, dataset) text-hash-keyed cache
├── metrics/
│   ├── core.py             # compute_per_language_metrics, LanguageMetrics
│   ├── aggregate.py        # macro_average / micro_average (gold-only + observed views)
│   ├── fpr.py              # false_positive_rate, stats_per_model_supported
│   └── support_matrix.py   # CSV load/save of language x model support
├── leaderboard/            # Optional gradio app (extra `commonlid[leaderboard]`)
│   ├── app.py              # build_app() — tabs per dataset + drilldowns
│   └── data.py             # LeaderboardRow, load_results() (HF dataset / local dir)
├── preprocess/
│   ├── openlid_normer.py   # OpenLID-v2 language-agnostic text cleaner
│   └── langcodes.py        # ISO 639 deprecation table + Lang() upgrade pipeline
├── datasets_tools/
│   ├── stratified_sample.py # Stratified sampler for nano-subset construction
│   └── frequency_sample.py  # Frequency-weighted sampler (smolsent / social_media prep)
├── vendor/
│   └── fun_langid.py       # Vendored simple char-4gram LID (1.7k lines of tables)
├── cli.py                  # Typer app: version, list-models, list-datasets, run,
│                           # predict, generate-support-matrix, export-csv,
│                           # leaderboard {serve,upload}
├── logging.py
└── py.typed
```

## Data flow (happy path)

```
CLI `commonlid run --model GlotLID --dataset udhr`
   │
   ├─► `commonlid.cli._ensure_registry_loaded` imports `models/` and `datasets/`
   │   → every submodule's `@register_model` / `@register_dataset` fires.
   │
   ├─► Instantiate model + dataset via registry.get_*.
   │
   ├─► `Evaluator.run()`:
   │     for each (model, dataset):
   │         model.load()                       # lazy weight download
   │         dataset.load(limit=limit)             # pinned HF revision
   │         cache = PredictionCache(...)       # per-(model, dataset)
   │         for batch in dataset.iter_batches():
   │             preds = _predict_with_cache(model, cache, texts)
   │                │
   │                ├─► cache.get(text)         # served if hit
   │                └─► model.predict(texts)    # miss: run + cache.put_many
   │         metrics = compute_per_language_metrics(ytrue, ypred)
   │         write_predictions(...)             # predictions.jsonl
   │         write_summary(Result(...))          # summary.json
   │
   └─► Each run writes exactly two files per (model, dataset) pair.
```

## `LIDModel.predict` contract

Every shipped model flows through:

1. **Input:** a sequence of raw strings.
2. **(optional) preprocess:** `openlid_normer_clean_line` unless
   `requires_preprocessing = False`. (LLMs skip it.)
3. **`_predict_batch`** (abstract): subclass returns one raw code (or
   `None`) per input.
4. **`_conform`:**
   - Map via the hand-written deprecation table
     (`preprocess.langcodes.conform_langcode`): e.g. `jw` → `jav`, `iw` →
     `heb`, `eml` → `None`.
   - Pipe through `iso639.Lang(...).pt3` to upgrade ISO 639-1/2 codes
     (e.g. `en` → `eng`, `de` → `deu`) and to drop codes without an
     ISO 639-3 successor.
   - Deprecated or unknown codes collapse to `None` — downstream the
     evaluator treats `None` as a prediction of the `und` bucket.

This two-stage conformation exactly mirrors the legacy pipeline (confirmed
by `tests/integration/test_smoke_parity.py` — GlotLID + cld2 on 200 UDHR
samples match the old code within 1e-6 on every per-language F1).

## Registry invariants

- Model ids and dataset ids are unique.
- Registering the same class twice is a no-op; registering a **different**
  class with an already-used id raises.
- Tests snapshot and restore the registry via the `fresh_registry`
  fixture so test-defined classes don't leak into other tests.

## Result schema

`summary.json` carries a top-level `"schema_version": 2`. Any
non-backwards-compatible change bumps the version so notebooks/readers
can branch on it. Fields are emitted by `Result.summary()`
(`src/commonlid/evaluation/results.py`).

Schema v2 splits the `macro` / `micro` blocks into two parallel views:

- `*_gold_only` — averaged over languages with `gt_count > 0`
  (the paper definition).
- `*_observed` — averaged over `set(gold) | set(pred)`, so spurious
  predictions for languages with no gold drag the score down.

Both views ship together (`f1_gold_only` + `f1_observed`,
`precision_gold_only` + `precision_observed`, `n_languages_gold` +
`n_languages_observed`, …).

## Test layout

- `tests/unit/` — pure unit tests against fixtures in `tests/fixtures/`.
- `tests/models/` — model-specific tests (mocked hf_hub + fasttext).
- `tests/integration/` — CLI end-to-end, evaluator end-to-end,
  `test_smoke_parity.py` (marked `slow` + `network`).
- `tests/legacy/` — frozen copies of the legacy `langid_models.py`
  and `langid_datasets.py` used only by the parity smoke test.

The default `pytest` run deselects `slow` and `network` markers.
