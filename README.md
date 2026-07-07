# CommonLID

Evaluate language-identification (LID) models on the CommonLID benchmark
and five public complementary datasets.

- 📄 **Paper:** [*CommonLID: A benchmark for web-scale language identification*](https://arxiv.org/abs/2601.18026)
- 📦 **Dataset:** [`commoncrawl/CommonLID`](https://huggingface.co/datasets/commoncrawl/CommonLID) on Hugging Face
- 🧪 **What this repo gives you:** a Python package (`commonlid`) + CLI
  (`commonlid`) with first-class abstractions for models and datasets, an
  evaluator that writes `predictions.jsonl` + `summary.json` to disk, and
  a set of pre-registered classical LID models + LLM support via DSPy.

## Minimal example

<!-- readme-test: slow; id=minimal-example -->
```python
from commonlid import Evaluator, get_model, get_dataset

Evaluator(
    models=[get_model("GlotLID")],
    datasets=[get_dataset("udhr")],
    output_dir="./results",
).run()
```

Or from the shell:

```bash
commonlid run --model GlotLID --dataset udhr --output-dir ./results
```

Both produce `./results/udhr/GlotLID/summary.json` and `predictions.jsonl`.

## Installation

From PyPI:

```bash
pip install commonlid                      # core deps + classical LID models
pip install "commonlid[llm]"               # + DSPy-based LLM evaluation
pip install "commonlid[afrolid]"           # + torch/transformers for AfroLID
pip install "commonlid[commonlingua]"      # + torch for the CommonLingua byte-level model
pip install "commonlid[notebooks]"         # + jupyterlab + matplotlib for paper_tables.ipynb
pip install "commonlid[all]"               # everything runtime-facing
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add commonlid                           # runtime only
uv add "commonlid[all]"                    # all runtime extras
```

For local development (tests, linter, full type-checking dev extra):

```bash
git clone https://github.com/commoncrawl/commonlid-eval.git
cd commonlid-eval
make install                               # uv sync --extra dev
make check                                 # ruff + mypy + pytest (matches CI)
source .venv/bin/activate
```

The `Makefile` wraps every common workflow (`make help` lists them) and
is what CI runs, so local and CI builds stay in lock-step. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full dev workflow.

> **Supported Python versions:** 3.10, 3.11, 3.12, 3.13 (CI tests every
> interpreter on each PR).

## CLI

List every registered model and dataset:

```bash
commonlid list-models
commonlid list-datasets
```

Evaluate multiple models across multiple datasets in one go:

```bash
commonlid run \
  --model GlotLID --model OpenLID-v2 --model cld2 \
  --dataset commonlid --dataset flores_dev --dataset udhr \
  --output-dir ./results \
  --limit 500                  # cap samples per dataset (0 = all)
```

Results land at `./results/{dataset_id}/{model_id}/predictions.jsonl` and
`summary.json`. Re-running is fast: a per-(model, dataset) cache keyed on
text hash + dataset revision serves cached predictions.

Ad-hoc prediction on a single string — no dataset, no result files:

```bash
commonlid predict --model GlotLID --text "Le chat dort sur le canapé."
# {"text": "Le chat dort sur le canapé.", "pred": "fra", "model": "GlotLID"}
```

Pipe a file through a model (use `-` for stdin):

```bash
commonlid predict --model cld2 --text-file my_sentences.txt
```

Flatten every summary in a results directory into one CSV:

```bash
commonlid export-csv --results-dir ./results --out ./results.csv
```

Build the per-model ISO 639-3 support matrix (one row per language, one
column per model, `1`/`0`):

```bash
commonlid generate-support-matrix --out ./results/support_matrix.csv
```

Models that cannot enumerate a concrete language list (e.g. LLMs) are
skipped and reported on stderr.

## LLM evaluation (via DSPy)

LLMs are just another model id. Prefix the DSPy model name with `dspy:`
and mix it freely with classical models in the same `run`:

```bash
commonlid run \
  --model GlotLID --model cld2 \
  --model dspy:azure/gpt-4o-mini \
  --api-base https://your-azure-endpoint.openai.azure.com/ \
  --api-version 2024-12-01-preview \
  --azure-ad-token \
  --temperature 0.7 \
  --dataset commonlid --dataset udhr \
  --output-dir ./results
```

The `--api-base`, `--api-version`, `--api-key`, `--azure-ad-token`,
`--temperature`, `--max-tokens`, `--max-completion-tokens`, and
`--llm-n-threads` flags are only consumed by `dspy:` models and ignored
when every model is a classical one.

## Python API

The `commonlid` import auto-registers every shipped model and dataset, so
`get_model` / `get_dataset` work immediately.

### Evaluate multiple models on multiple datasets

<!-- readme-test: slow; id=multi-eval -->
```python
from commonlid import Evaluator, get_model, get_dataset

results = Evaluator(
    models=[get_model("GlotLID"), get_model("cld2"), get_model("pyfranc")],
    datasets=[get_dataset("udhr"), get_dataset("commonlid")],
    output_dir="./results",
    batch_size=64,
    limit=500,          # optional cap per dataset
    use_cache=True,     # text-hash-keyed per-(model, dataset) cache
).run()

for r in results:
    s = r.summary()
    print(
        f"{r.model_id:12} on {r.dataset_id:10}  "
        f"macro F1={s['macro']['f1_gold_only']:.3f}  "
        f"samples/sec={s['samples_per_second']:.0f}"
    )
```

### Ad-hoc prediction (no dataset, no files)

<!-- readme-test: fast; id=ad-hoc-predict -->
```python
from commonlid import get_model

model = get_model("cld2")
preds = model.predict([
    "The quick brown fox jumps over the lazy dog",
    "Der schnelle braune Fuchs springt über den faulen Hund",
    "素早い茶色の狐が怠け者の犬を飛び越える",
])
assert preds == ["eng", "deu", "jpn"]
```

### List models / datasets

<!-- readme-test: fast; id=list -->
```python
from commonlid import list_models, list_datasets

assert list_models() == [
    "AfroLID", "GlotLID", "OpenLID-v2", "cld2", "cld3",
    "commonlingua", "fasttext", "funlangid", "pyfranc",
]
assert list_datasets() == [
    "bibles_300", "bibles_300_nano",
    "commonlid", "commonlid_nano",
    "flores_dev", "flores_dev_nano",
    "smolsent_300", "smolsent_300_nano",
    "social_media_300", "social_media_300_nano",
    "udhr", "udhr_nano",
]
```

### Compute metrics without running the evaluator

Useful when you already have `(ytrue, ypred)` from a custom pipeline:

<!-- readme-test: fast; id=metrics -->
```python
import math

from commonlid.metrics import (
    compute_per_language_metrics,
    macro_average,
    false_positive_rate,
)

ytrue = ["eng", "eng", "deu", "fra", "fra"]
ypred = ["eng", "eng", "deu", "fra", "spa"]

per_lang = compute_per_language_metrics(ytrue, ypred)
assert per_lang["fra"].precision == 1.0
assert per_lang["fra"].recall == 0.5
assert math.isclose(per_lang["fra"].f1, 2 / 3)

macro = macro_average(per_lang)
# Both views are returned. "gold-only" averages over languages with
# gt_count > 0 (the paper definition); "observed" averages over every
# language seen in either gold or predictions. Here ``spa`` is a
# spurious prediction (no gold) so the views diverge.
assert macro["n_languages_gold"] == 3      # eng, deu, fra
assert macro["n_languages_observed"] == 4  # + spa
assert macro["precision_gold_only"] == 1.0  # all gold langs have perfect precision
assert macro["precision_observed"] == 0.75  # spa drags the mean down

# 1 of 5 non-Spanish samples was mislabelled as Spanish
assert false_positive_rate(ytrue, ypred, language="spa") == 0.2
```

### Evaluate an LLM (DSPy) as a LID model

<!-- readme-test: skip; id=dspy-llm (requires a real Azure endpoint) -->
```python
from commonlid import Evaluator, get_dataset
from commonlid.models.dspy_llm import DSPyLLMModel

model = DSPyLLMModel(
    llm_model_name="azure/gpt-4o-mini",
    api_base="https://your-endpoint.openai.azure.com/",
    api_version="2024-12-01-preview",
    azure_ad_token=True,     # uses DefaultAzureCredential
    temperature=0.7,
    batch_size=100,
    n_threads=4,
    cache_dir="./results/.dspy_cache",
)

Evaluator(
    models=[model],
    datasets=[get_dataset("commonlid")],
    output_dir="./results",
).run()
```

### Load a previous run's results

<!-- readme-test: fast; id=load-results (fixture populates ./results) -->
```python
import json
from pathlib import Path

from commonlid.evaluation.results import load_summary

results_dir = Path("./results")

for summary_path in sorted(results_dir.rglob("summary.json")):
    s = load_summary(summary_path)
    print(s["model_id"], s["dataset_id"], s["macro"]["f1_gold_only"])

# Stream every per-sample prediction for one run:
preds_path = next(results_dir.rglob("predictions.jsonl"))
for line in preds_path.read_text().splitlines():
    row = json.loads(line)
    assert "gold" in row and "pred" in row and "correct" in row
```

## Registered models

| `model_id` | Upstream | Notes |
|---|---|---|
| `cld2` | [pycld2](https://pypi.org/project/pycld2/) | Pure-Python C++ binding, CPU-only |
| `cld3` | [cld3-py](https://pypi.org/project/cld3-py/) | Google CLD3 C++ via modernised Python bindings. Optional extra `commonlid[cld3]` |
| `GlotLID` | [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) | 2100+ languages, fasttext |
| `OpenLID-v2` | [laurievb/OpenLID-v2](https://huggingface.co/laurievb/OpenLID-v2) | fasttext |
| `fasttext` | [facebook/fasttext-language-identification](https://huggingface.co/facebook/fasttext-language-identification) | fasttext |
| `pyfranc` | [pyfranc](https://pypi.org/project/pyfranc/) | Pure Python |
| `AfroLID` | [UBC-NLP/afrolid_1.5](https://huggingface.co/UBC-NLP/afrolid_1.5) | Requires `[afrolid]` extra |
| `commonlingua` | [PleIAs/CommonLingua](https://huggingface.co/PleIAs/CommonLingua) | 2.35M-param byte-level model, 334 languages; requires `[commonlingua]` extra |
| `funlangid` | Vendored in `src/commonlid/vendor/fun_langid.py` | Simple char-4gram baseline |

LLM models are instantiated dynamically (`DSPyLLMModel`) and not
auto-registered — they need per-instance configuration (endpoint + key).

## Registered datasets

Each registered dataset declares two HF-repo attributes:

* `source_hf_repo` — the canonical *public* HF dataset.
* `cache_hf_repo` — an optional pre-built (often private) HF artifact of a
  preprocessed/sampled subset. When set, `load()` tries this first.

| `dataset_id` | `source_hf_repo` (public) |
|---|---|
| `commonlid` | `commoncrawl/CommonLID` |
| `flores_dev` | `openlanguagedata/flores_plus` |
| `udhr` | `cis-lmu/udhr-lid` |
| `bibles_300` | — |
| `smolsent_300` | `google/smol` |
| `social_media_300` | — |
| `bibles_300_nano` | — |
| `commonlid_nano` | — |
| `flores_dev_nano` | — |
| `smolsent_300_nano` | — |
| `udhr_nano` | — |
| `social_media_300_nano` | — |

Cache repos, splits, and pinned revisions live on each `LIDDataset` subclass
in `src/commonlid/datasets/`.

The six `*_nano` variants are stratified samples (~`max_size=1000` + `min_size=5`
per language) of their parent benchmarks. Each lives in its own HF repo
(`commoncrawl/commonlid-cache_<base>_nano`) so visibility (public / private)
can track the parent dataset. All caches share the schema
`(index, text, language_iso639_3)`. Their `build_from_source()` recursively
calls the parent's `load()` then applies `stratified_sample_with_minimum_per_class()`
from `src/commonlid/datasets_tools/stratified_sample.py` (a byte-equivalent
port of the original `generate_small_version`).

Datasets with `is_cache_private=True` (`bibles_300`, `smolsent_300`,
`social_media_300`) require an access grant + authenticated client
(`huggingface-cli login` or the `HF_TOKEN` env var). The underlying source is
public; the cache is private because we cannot redistribute the
preprocessed/sampled artifact. If the cache is unreachable, `LIDDataset.load()`
falls back to `build_from_source()` when the subclass implements it
(currently `bibles_300` and `smolsent_300`):

| `dataset_id` | Public source for `build_from_source()` |
|---|---|
| `bibles_300` | `bibles_with_lang_labels.tsv` — request the raw file from the maintainers, then point `COMMONLID_BIBLES_RAW_PATH` at its location |
| `smolsent_300` | [`google/smol`](https://huggingface.co/datasets/google/smol) (smolsent config), fetched automatically |

When neither path resolves, `commonlid.PrivateDatasetAccessError` is raised
with both the access-request URL and the build-from-source instructions.

## Language code normalisation

The shipped models emit raw language codes in several different formats.
`LIDModel.predict()` funnels every raw code through a two-stage
normalisation pipeline so downstream metrics always see canonical ISO
639-3 codes and a single `None` sentinel for "undetermined":

1. **Model-specific sentinel handling** — each wrapper maps the
   library's "no prediction" token to `None` *before* the raw code
   leaves `_predict_batch`:
   - `cld2` → `un`, `xx`, `zzp` (`src/commonlid/models/cld2.py`)
   - `cld3` / `funlangid` → `und` (`src/commonlid/models/cld3.py`,
     `funlangid.py`)
   - `AfroLID` → `nan_lang` (`src/commonlid/models/afrolid.py`)
   - fasttext-based models (`GlotLID`, `OpenLID-v2`, `fasttext`) parse
     `__label__{code}_{script}` down to just `{code}`
     (`src/commonlid/models/_fasttext_base.py`)
2. **`LIDModel._conform()`** (`src/commonlid/core/lid_model.py:86`)
   runs on every non-`None` raw code and performs two more steps:
   1. **Deprecation-table rewrite.**
      `preprocess.langcodes.conform_langcode(...)` rewrites deprecated
      codes from the hand-written table at the top of
      `src/commonlid/preprocess/langcodes.py`. Codes whose language
      split into multiple successors resolve to `None`.
   2. **ISO 639-3 upgrade.** The surviving code is passed to
      [`iso639-lang`](https://pypi.org/project/iso639-lang/) via
      `iso639.Lang(...).pt3`. This accepts any ISO 639-1/2/3/5 code
      and emits the canonical ISO 639-3 form
      (`en`→`eng`, `de`→`deu`, `zh`→`zho`, ...). Codes `iso639-lang`
      can't parse become `None`.

For reference, the full deprecation table baked into `conform_langcode`
(each `Reason` is the text that `iso639-lang` raises for that code):

| Input | Output | Reason |
|-------|--------|--------|
| `jw`  | `jav`  | As of 2001-08-13, [jw] for Javanese is deprecated due to deprecated. Use [jv] instead. |
| `bh`  | `bih`  | As of 2021-05-25, [bh] for Bihari languages is deprecated due to deprecated. Two-letter identifier bh deprecated in ISO 639-1; use of three-letter identifier bih for Bihari languages is favored. |
| `iw`  | `heb`  | As of 1989-03-11, [iw] for Hebrew is deprecated due to deprecated. Use [he] instead. |
| `ajp` | `apc`  | As of 2023-01-20, [ajp] for South Levantine Arabic is deprecated due to merge. Use [apc] instead. |
| `eml` | `None` | As of 2009-01-16, [eml] for Emiliano-Romagnolo is deprecated due to split. Split into Emilian [egl] and Romagnol [rgn]. |
| `tpw` | `tpn`  | As of 2023-01-20, [tpw] for Tupí is deprecated due to duplicate. Use [tpn] instead. |
| `oto` | `None` | No iso639-3 code: `Lang(name='Otomian languages', pt1='', pt2b='oto', pt2t='oto', pt3='', pt5='oto')`. |
| `ber` | `tzm`  | No iso639-3 code: `Lang(name='Berber languages', pt1='', pt2b='ber', pt2t='ber', pt3='', pt5='ber')` → use Central Atlas Tamazight [tzm]. |
| `ngo` | `None` | As of 2021-01-15, [ngo] for Ngoni is deprecated due to split. Split into Ngoni (Tanzania) [xnj] and Ngoni (Mozambique) [xnq]. |
| `kzj` | `dtp`  | As of 2016-01-15, [kzj] for Coastal Kadazan is deprecated due to merge. Use [dtp] instead. |
| `dan` | `None` | As of 2013-01-23, [daf] for Dan is deprecated due to split. Split into Dan [dnj] and Kla-Dan [lda]. *(Keyed on `dan` bug-for-bug from the legacy pipeline; see the code comment in `langcodes.py`.)* |
| `kxu` | `None` | As of 2020-01-23, [kxu] for Kui (India) is deprecated due to split. Split into [dwk] Dawik Kui and [uki] Kui (India). |
| `nah` | `None` | No iso639-3 code: `Lang(name='Nahuatl languages', pt1='', pt2b='nah', pt2t='nah', pt3='', pt5='nah')`. |
| `bih` | `None` | No iso639-3 code: `Lang(name='Bihari languages', pt1='', pt2b='bih', pt2t='bih', pt3='', pt5='bih')`. |

A helper
(`preprocess.langcodes.convert_and_conform_language`) adds a fourth
normalisation step for *raw* codes coming from external data — it
trims any ISO-639 tag at the first `-` or `_` before running the
pipeline above, so `en-US` → `eng`, `zh_Hant` → `zho`. The model
wrappers do the `-` split themselves (`cld3`, `funlangid`) or rely on
the fasttext label format, so this helper is only used when you load
a dataset whose gold labels arrive in BCP-47 / locale form.

Dataset gold labels are checked (but not rewritten) when the dataset
loads: `LIDDataset._check_gold_conformity()`
(`src/commonlid/core/lid_dataset.py:73`) iterates every target column
value through `conform_langcode_with_reason` and logs a warning when
codes would change. This keeps the ground truth in the HF dataset as-is
while surfacing drift.

In the metrics layer, any `None` prediction is bucketed as `"und"`
(`src/commonlid/metrics/core.py:_prepare`) so per-language P/R/F1 can
still report an abstention rate; `macro_average` / `micro_average`
exclude the `und` bucket by default (toggle with `include_und=True`).

## Adding a new model

A guide for adding a new model can be found [here](docs/contributing/adding_a_model.md).

<!-- readme-test: fast; id=add-model (registers into an isolated registry) -->
```python
# src/commonlid/models/my_model.py
from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import get_model, register_model


@register_model
class MyModel(LIDModel):
    model_id = "my_model"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        # Return one ISO 639-3 code (or None for undetermined) per input.
        # `texts` arrives post-OpenLID-normer cleaning by default;
        # set `requires_preprocessing = False` to receive raw text.
        return ["eng"] * len(texts)


assert get_model("my_model").predict(["hi"]) == ["eng"]
```

Then import it from `src/commonlid/models/__init__.py` so the decorator
fires on `import commonlid`:

```python
from commonlid.models import my_model as _my_model  # noqa: F401
```

Add a test under `tests/models/`.

## Adding a new dataset

<!-- readme-test: fast; id=add-dataset -->
```python
# src/commonlid/datasets/my_task.py
from commonlid.core.lid_dataset import LIDDataset
from commonlid.core.registry import get_dataset, register_dataset


@register_dataset
class MyTask(LIDDataset):
    dataset_id = "my_task"
    source_hf_repo = "me/my-lid-dataset"
    source_hf_revision = "abcdef1234567890..."   # pin a full git SHA
    source_hf_split = "test"
    text_column = "text"
    target_column = "iso639_3"


assert get_dataset("my_task").dataset_id == "my_task"
```

Import from `src/commonlid/datasets/__init__.py`:

```python
from commonlid.datasets import my_task as _my_task  # noqa: F401
```

## Result format

Each `(model, dataset)` run produces two files.

### `summary.json`

```json
{
  "schema_version": 1,
  "model_id": "GlotLID",
  "dataset_id": "udhr",
  "dataset_revision": "6908db2a27c296158da7e69782d15df911652184",
  "commonlid_version": "0.1.0",
  "python_version": "3.13.12",
  "platform": "macOS-15.2-arm64-arm-64bit",
  "timestamp": "2026-04-20T10:00:00+00:00",
  "limit": null,
  "n_samples": 2800,
  "n_samples_with_gold": 2800,
  "samples_per_second": 1842.3,
  "macro": {
    "f1_gold_only": 0.905, "precision_gold_only": 0.91, "recall_gold_only": 0.90,
    "n_languages_gold": 197,
    "f1_observed": 0.85, "precision_observed": 0.85, "recall_observed": 0.86,
    "n_languages_observed": 213
  },
  "micro": {
    "f1_gold_only": 0.88, "precision_gold_only": 0.88, "recall_gold_only": 0.88,
    "n_correct_gold": 2464, "n_predictions_gold": 2800, "n_gold_samples": 2800,
    "f1_observed": 0.86, "precision_observed": 0.84, "recall_observed": 0.88,
    "n_correct_observed": 2464, "n_predictions_observed": 2920
  },
  "per_language": {
    "eng": {
      "gt_count": 14, "predictions": 14, "correct": 14,
      "precision": 1.0, "recall": 1.0, "f1": 1.0
    }
  },
  "extra": {}
}
```

### `predictions.jsonl`

One line per sample:

```json
{"idx": 0, "text_hash": "abcd1234efgh5678", "gold": "eng", "pred": "eng", "correct": true}
```

## Analysis

Once you have a results directory, the reference notebook regenerates the
paper-style tables and plots:

```bash
make notebooks      # installs the [notebooks] extra and launches jupyter lab
```

See `notebooks/README.md` for what the notebook produces and how to point
it at your own results directory.

## Hugging Face Space

The leaderboard runs as a public Gradio Space at
[huggingface.co/spaces/commoncrawl/commonlid](https://huggingface.co/spaces/commoncrawl/commonlid).
It reads results from the
[`commoncrawl/commonlid-results`](https://huggingface.co/datasets/commoncrawl/commonlid-results)
dataset (one `summary.json` per `<dataset_id>/<model_id>`) and renders one
tab per benchmark.

### Local preview

```bash
make leaderboard                                # serve from ./data/results
# or against the live results dataset:
uv run commonlid leaderboard serve
```

`make leaderboard` installs the `[leaderboard]` extra on first run and
forwards the local results tree (`LEADERBOARD_DIR`, default
`./data/results`) to `commonlid leaderboard serve`.

### Refresh the results data (PR-based)

```bash
hf auth login                                   # token with write access to the results dataset
make leaderboard-upload                         # opens a Pull Request from ./data/results
# Override the target with: make leaderboard-upload LEADERBOARD_REPO=other/repo LEADERBOARD_DIR=./elsewhere
# Optional: pass --skip-predictions via `uv run commonlid leaderboard upload ...` directly.
```

The CLI always opens a Pull Request rather than pushing to the default
branch, so the dataset owner reviews before merging.

### Deploy / update the Space (CLI)

The Space is just a git repo on the Hub holding `app.py`, `README.md`
(with a Gradio front-matter), and `requirements.txt`. Three files live
under `hf-space/` in this repo and map 1:1 to the Space root.

```bash
# One-time: create the Space (skip if already created via the web UI)
hf repo create --type space --space-sdk gradio commoncrawl/commonlid

# Push (or update) the entrypoint files
hf upload --repo-type=space commoncrawl/commonlid ./hf-space .
```

`hf upload` does an incremental upload and triggers a rebuild on the
Space. Optional environment variables on the Space:

- `COMMONLID_RESULTS_REPO` — override the dataset repo id.
- `COMMONLID_RESULTS_REVISION` — pin a specific results commit so the
  Space doesn't drift while you iterate on the dataset.

## Contributing

Dev environment setup, quality gates, adding models/datasets, adding
tested README examples, and the manually-triggered release workflow
are all documented in [CONTRIBUTING.md](CONTRIBUTING.md). See also
`docs/architecture.md` for the package layout.

## Citing

If you use this package or the CommonLID benchmark, please cite the paper
([arXiv:2601.18026](https://arxiv.org/abs/2601.18026)):

```bibtex
@misc{ortizsuarez2026commonlid,
  title = {CommonLID: Re-evaluating State-of-the-Art Language Identification Performance on Web Data},
  author = {Ortiz Suarez, Pedro and Burchell, Laurie and Arnett, Catherine and Mosquera-G{\'o}mez, Rafael and Hincapie-Monsalve, Sara and Vaughan, Thom and Stewart, Damian and Ostendorff, Malte and Abdulmumin, Idris and Marivate, Vukosi and Muhammad, Shamsuddeen Hassan and Tonja, Atnafu Lambebo and Al-Khalifa, Hend and Ghezaiel Hammouda, Nadia and Otiende, Verrah and Wong, Tack Hwa and Saydaliev, Jakhongir and Nobakhtian, Melika and Habibi, Muhammad Ravi Shulthan and Kranti, Chalamalasetti and Muchemi, Carol and Nguyen, Khang and Adam, Faisal Muhammad and Salim, Luis Frentzen and Alqifari, Reem and Amol, Cynthia and Imperial, Joseph Marvin and Kesen, Ilker and Mustafid, Ahmad and Stepachev, Pavel and Choshen, Leshem and Anugraha, David and Nayel, Hamada and Yimam, Seid Muhie and Putra, Vallerie Alexandra and Nguyen, My Chiffon and Wasi, Azmine Toushik and Vadithya, Gouthami and van der Goot, Rob and ar C'horr, Lanwenn and Dua, Karan and Yates, Andrew and Bangera, Mithil and Bangera, Yeshil and Patel, Hitesh Laxmichand and Okabe, Shu and Ilasariya, Fenal Ashokbhai and Gaynullin, Dmitry and Winata, Genta Indra and Li, Yiyuan and Mart{\'\i}nez, Juan Pablo and Agarwal, Amit and Hanif, Ikhlasul Akmal and Abu Ahmad, Raia and Adenuga, Esther and Tjiaranata, Filbert Aurelian and Buaphet, Weerayut and Anugraha, Michael and Vajjala, Sowmya and Rice, Benjamin and Amirudin, Azril Hafizi and Alabi, Jesujoba O. and Panda, Srikant and Toughrai, Yassine and Kyomuhendo, Bruhan and Ruffinelli, Daniel and A, Akshata and Goul{\~a}o, Manuel and Zhou, Ej and Franco Ramirez, Ingrid Gabriela and Aggazzotti, Cristina and Dobler, Konstantin and Kevin, Jun and Pag{\`e}s, Quentin and Andrews, Nicholas and Ibrahim, Nuhu and Ruckdeschel, Mattes and Keleg, Amr and Zhang, Mike and Muziri, Casper and Samuel, Saron and Takeshita, Sotaro and Kerdthaisong, Kun and Foppiano, Luca and Dent, Rasul and Green, Tommaso and Wali, Ahmad Mustapha and Makaaka, Kamohelo and Feliren, Vicky and Idris, Inshirah and Celikkanat, Hande and Abubakar, Abdulhamid and Maillard, Jean and Sagot, Beno{\^i}t and Cl{\'e}rice, Thibault and Murray, Kenton and Luger, Sarah},
  year = {2026},
  eprint = {2601.18026},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2601.18026},
  url = {https://arxiv.org/abs/2601.18026},
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
