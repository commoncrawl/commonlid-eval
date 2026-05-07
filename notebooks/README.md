# notebooks/

Notebooks are kept only for **final analysis and plotting**. All core
evaluation logic (metrics, preprocessing, dataset loading, support matrix)
lives in the `commonlid` package.

## Current state

| Notebook | Purpose |
|---|---|
| `paper_tables.ipynb` | Loads `summary.json` + `predictions.jsonl` from a results directory and regenerates the core analysis tables: macro/micro F1 per (model, dataset), per-language F1 pivot, supported-language means (using the `support_matrix.csv` from `commonlid generate-support-matrix`), and a false-positive-rate drill-down. |
| `creating_commonlid.ipynb` | Walks through the computational analysis + manual inspection used to turn the raw annotated dump into the CommonLID evaluation dataset described in the paper (initial filtering, English-contamination checks, conflict resolution, final assembly). |

## How to reproduce the paper tables

1. Generate evaluation results via the CLI:
   ```bash
   commonlid run \
     --model GlotLID --model OpenLID-v2 --model cld2 --model fasttext \
     --model pyfranc --model funlangid \
     --task commonlid --task flores_dev --task udhr \
     --task bibles_300 --task smolsent_300 --task social_media_300 \
     --output-dir ./results
   ```
2. (Optional) generate the support matrix used to restrict averages to
   each model's declared languages:
   ```bash
   commonlid generate-support-matrix \
     --out ./results/support_matrix.csv \
     -m GlotLID -m OpenLID-v2 -m cld2 -m fasttext -m pyfranc -m funlangid
   ```
3. Launch Jupyter (requires the `notebooks` extra):
   ```bash
   uv sync --extra notebooks
   uv run jupyter lab
   ```
4. Open `notebooks/paper_tables.ipynb`, adjust `RESULTS_DIR` at the top of
   the setup cell if needed, and run.

## Also see

- `docs/architecture.md` — package layout and data-flow overview.
