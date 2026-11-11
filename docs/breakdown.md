# Cross-model breakdown (`commonlid breakdown`)

The leaderboard gives one number per model and dataset. `commonlid breakdown`
looks underneath it. It lines up the per-sample predictions of several models
on one dataset and answers questions such as:

- Which languages, and which samples, do even the best models get wrong? What
  do they predict instead?
- Where do the models agree with each other, and where do they disagree?
- Which gold labels look wrong, because the models agree on a different
  label, or because the same text appears in the dataset with different labels?
- How much do the scores and the ranking change if a prediction of `lav`
  for a gold `lvs` (Latvian vs Standard Latvian) counts as correct?

It works on the `predictions.jsonl` files published in the results dataset
([`commoncrawl/commonlid-results`](https://huggingface.co/datasets/commoncrawl/commonlid-results))
or on a local results directory written by `commonlid run`. It does not run
any model.

## Quick start

```bash
# Languages ranked by how often the 5 best models on CommonLID miss them
commonlid breakdown --dataset commonlid --top-k 5

# Languages where the models disagree most with each other
commonlid breakdown --dataset commonlid --top-k 5 --sort-by agreement --order asc

# Candidate annotation errors, with the text, as a CSV for review
commonlid breakdown --dataset commonlid --top-k 5 --level sample \
  --only-suspect --macro-match macro --with-text --out suspects.csv

# Texts that occur more than once with conflicting gold labels
commonlid breakdown --dataset commonlid --level sample --only-gold-conflict \
  --with-text --out conflicts.csv

# All models rescored with macrolanguage matching, next to the official scores
commonlid breakdown --dataset commonlid_nano --level model --top-k 100 --macro-match macro
```

The command prints the selected models and a preview of the table (`--show N`
rows, 20 by default) to the terminal. `--out` writes the full table as `.csv`,
`.jsonl` or `.parquet`.

## How it works

1. **Pick the models.** By default the `--top-k` best models on the dataset by
   a leaderboard column (`--metric`, `macro_f1` by default). `--exclude-model X`
   leaves a model out of that ranking. `--model X --model Y` picks models
   explicitly instead.
2. **Load their predictions.** Only the needed `predictions.jsonl` files are
   downloaded (`--repo-id`, `--revision`, `--cache-dir`), or they are read from
   `--local-dir` (the layout of `commonlid run --output-dir`). Models without a
   predictions file are skipped with a warning.
3. **Line up the samples by `idx`**, the sample's position in the dataset.
   This only works if every model ran on the same dataset revision. Each row
   stores a hash of its text, and the command stops with an error if two models
   have a different text at the same `idx`. A model may lack rows (for example
   a run with `--limit`): statistics for a sample use the models that have a
   prediction for it (`n_models`).
4. **Compare each prediction with the gold label** under the chosen
   `--macro-match` mode (below). Samples without a gold label are left out. A
   missing prediction (`null`, the model gave no supported label) counts as
   `und` and is wrong, as in the official metrics.

## Macrolanguage matching

Many languages in the benchmarks belong to an ISO 639-3 *macrolanguage*.
Models and annotators do not always use the same level: one says `lav`
(Latvian), the other `lvs` (Standard Latvian). The official metrics match codes
exactly, so these count as errors and often dominate the error lists.
`--macro-match` sets how a prediction is compared with the gold label:

| mode | counts as correct | examples that match |
|---|---|---|
| `exact` (default) | the same code; what the leaderboard does | `eng`/`eng` |
| `macro` | also an individual language and its own macrolanguage, in either direction | `lvs`/`lav`, `cmn`/`zho`, `swh`/`swa`, `arz`/`ara` |
| `sibling` | also two individual languages of the same macrolanguage | `arz`/`arb`, `cmn`/`yue`, `ind`/`zsm`, `ltg`/`lvs` |

Use `macro` to remove the code-convention mismatches without hiding real
mistakes. `sibling` is much more lenient: Egyptian vs Standard Arabic,
Mandarin vs Cantonese and Indonesian vs Malay are distinct languages that LID
is expected to tell apart, so treat it as an upper bound.

The mapping comes from the [`iso639-lang`](https://pypi.org/project/iso639-lang/)
package. `fil`/`tgl`, `hbo`/`heb` and `ell`/`grc` are separate languages there,
not macrolanguage members, so no mode matches them.

When a prediction counts as correct under the mode, it is counted as the gold
label (for `consensus_pred`, `agreement` and the confusion columns). Wrong
predictions keep their own code. The `pred:<model>` columns always show the
predictions as published.

## Levels and columns

### `--level language` (default)

One row per gold language, sorted by `error_rate`, highest first.

| column | meaning |
|---|---|
| `language`, `n_samples` | gold label and its number of samples |
| `error_rate` | mean share of the models that are wrong on a sample of this language. When every model has every sample this is `1 - mean recall` |
| `agreement` | mean share of the models that make the most common prediction on a sample, right or wrong. Low values mean the models disagree with each other |
| `all_wrong_share` / `all_correct_share` | share of the samples that every model gets wrong / right |
| `n_label_suspect`, `n_gold_conflict` | number of samples with these flags (see below) |
| `top_confusion` / `top_confusion_share` | most common wrong prediction, and its share of all wrong predictions for this language |
| `recall:<model>` | each model's recall on this language |

`--min-samples N` drops languages with fewer than `N` samples.

### `--level sample`

One row per sample, sorted by `error_rate`, then `agreement`.

| column | meaning |
|---|---|
| `idx`, `text_hash`, `gold` | sample position, text hash and gold label |
| `n_models`, `n_correct`, `n_wrong`, `error_rate` | how many models have a prediction, are right, are wrong, and `n_wrong / n_models` |
| `consensus_pred`, `agreement` | most common prediction and the share of models that made it |
| `top_wrong_pred`, `top_wrong_count` | most common wrong prediction and how many models made it |
| `label_suspect` | candidate annotation error (below) |
| `gold_conflict`, `other_golds` | the same text has a non-matching gold label elsewhere, and those labels |
| `pred:<model>` | each model's prediction |
| `raw:<model>` | each model's raw output, with `--with-raw` |
| `text` | the sample text, with `--with-text` |

Sample-level options:

- `--only-suspect` and `--only-gold-conflict` keep only flagged samples.
- `--with-text` joins the text back in. The predictions only store a hash, so
  this downloads the dataset and needs access to it, which some datasets
  restrict. Each text is checked against the stored hash. If the dataset has
  changed since the run, the text is left empty and a warning is printed.
- `--with-raw` adds what each model actually returned before it was mapped to
  an ISO 639-3 code (for example `hindi` from an LLM, where `pred` is empty).
  It is empty for results recorded before raw outputs were kept.

### `--level model`

One row per selected model, rescored from the predictions under
`--macro-match`. With `exact` the numbers are the leaderboard's. Use a large
`--top-k` (or several `--model`) to rescore every model.

| column | meaning |
|---|---|
| `model_id`, `n_samples` | model and its samples with a gold label |
| `macro_f1`, `macro_precision`, `macro_recall`, `rank` | macro scores under the mode (gold languages only, as on the leaderboard) and the rank among the selected models |
| `macro_f1_exact`, `rank_exact` | the same with exact matching |
| `macro_f1_gain` | `macro_f1 - macro_f1_exact` |

All gold languages are kept as separate classes. A correct match under the mode
is scored as the gold label, so it is neither a miss for the gold language nor
a false positive for the predicted one.

## Flags for annotation review

**`label_suspect`**: at least two models, and at least `--suspect-min-agreement`
of them (0.8 by default), agree on the same wrong label, and that label is not
`und`. When several strong models independently agree on another label, the
gold label is often wrong. It is a triage signal, not a verdict:

- Models trained on similar data make similar mistakes. Pick models of
  different kinds (n-gram, neural, LLM) with `--model` or `--exclude-model`.
  Otherwise one family of models can make up most of the agreement.
- Close varieties, such as vocalised Biblical Hebrew (`hbo`) labelled
  correctly but predicted as `heb`, show up as suspects even when the label is
  right.
- Use `--macro-match macro`, or the list is dominated by `lvs`/`lav`-style
  code mismatches.

**`gold_conflict`**: the same text (by hash) occurs more than once in the
dataset with gold labels that do not match each other under `--macro-match`.
Every model is wrong on at least one of those rows, whatever it predicts. A
plain duplicate with the same label is not a conflict, and neither is
`ara`/`arb` under `macro`.

Group the exported rows by `(gold, consensus_pred)` or `(gold, other_golds)` to
find systematic labelling issues instead of one-offs.

## Filtering and sorting

- `--sort-by COLUMN` sorts by any output column (default `error_rate`, or
  `macro_f1` at model level). `--order desc|asc` sets the direction (default
  `desc`, highest first). Ties keep the default order.
- `--min-error-rate` / `--max-error-rate` keep rows within an error-rate range
  (sample and language level). `--max-error-rate 0` lists what every model gets
  right.

An option that does not apply to the chosen `--level` is an error rather than
being ignored.

## Python API

The same functions are available from `commonlid.analysis`. They take and
return pandas DataFrames, so they can be used in notebooks or other tools:

```python
from commonlid.analysis import (
    language_breakdown, load_predictions, model_breakdown, sample_breakdown,
    select_top_models,
)
from commonlid.leaderboard import load_results

results = load_results(allowed_datasets=["commonlid"])
models = select_top_models(results, "commonlid", k=5)
preds = load_predictions("commonlid", models)  # long frame: one row per (model, sample)

samples = sample_breakdown(preds, macro_match="macro")
languages = language_breakdown(preds, samples=samples, macro_match="macro")
scores = model_breakdown(preds, macro_match="macro")
```

`labels_match(gold, pred, mode)` and `to_macrolanguage(code)` expose the
matching rules, and `attach_texts(samples, "commonlid")` joins in the texts.

## Caveats

- Small datasets give noisy numbers. The `*_nano` datasets have about 1,500
  samples and as few as 5 per language, so differences of a few thousandths of
  macro F1 between models are not meaningful.
- The top-k models differ between datasets. For example, the LLMs only have
  results on the `*_nano` datasets.
- `--level model` ranks only the selected models. The model selection itself
  always uses the published (exact-match) leaderboard column.
