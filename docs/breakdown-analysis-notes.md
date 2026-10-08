# Top-k model breakdown: findings and next steps

Working notes for the `commonlid breakdown` analysis tooling
(branch `claude/topk-model-eval-sampling-6pb5j6`). The goal is to see which
samples and languages the best models get wrong (or right), and to use
cross-model agreement to surface candidate annotation errors.

## What exists

- `src/commonlid/analysis/breakdown.py` (exported from `commonlid.analysis`):
  - `select_top_models(results_df, dataset_id, k, metric="macro_f1")`
    ranks models with the leaderboard's `load_results()` frame.
  - `load_predictions(dataset_id, model_ids, ...)` downloads only the needed
    `<dataset>/<model>/predictions.jsonl` files from
    `commoncrawl/commonlid-results` (or reads `--local-dir`).
  - `sample_breakdown(preds)` gives one row per sample: `n_correct`, `n_wrong`,
    `error_rate`, `consensus_pred`, `agreement`, `top_wrong_pred`,
    `label_suspect` and `pred:<model>`.
  - `language_breakdown(preds)` gives one row per gold language:
    `error_rate`, `agreement`, `all_wrong_share`, `all_correct_share`,
    `n_label_suspect`, `top_confusion(_share)` and `recall:<model>`.
  - `attach_texts(samples, dataset)` joins the text by `idx` and verifies it
    against `text_hash`.
  - `collapse_macrolanguages=True` counts `lvs`/`lav`, `cmn`/`zho`,
    `zsm`/`msa` etc. as a match (uses `iso639-lang`).
- CLI: `commonlid breakdown --dataset D [--level language|sample] [-k 5]
  [--metric macro_f1] [--model X ...] [--exclude-model X]
  [--sort-by error_rate|agreement|<col>] [--order desc|asc]
  [--min-error-rate/--max-error-rate] [--only-suspect]
  [--suspect-min-agreement 0.8] [--min-samples N]
  [--collapse-macrolanguages] [--with-text] [--out f.csv|.jsonl|.parquet]`.
- Tests: `tests/unit/test_analysis_breakdown.py`. README section
  "Per-sample and per-language breakdown across the top models".

Definitions:

- `error_rate`: share of the selected models that are wrong on a sample.
  A `None` prediction counts as `und` and therefore as wrong, as in the
  evaluator.
- `agreement`: share of models that make the most common prediction,
  whether it is right or wrong.
- `label_suspect`: at least 2 models, and at least `--suspect-min-agreement`
  of them, agree on the same label that is neither the gold label nor `und`.

## Key findings so far

### Data and format observations

- Predictions only store `text_hash`, not text. Join on `idx`, because
  `text_hash` is not unique (duplicate texts exist).
- Not every model has a row for every sample: `commonlid_nano/GPT-4o-mini`
  has 1,503 of 1,507 rows. Per-sample statistics use the models that are
  present (`n_models`).
- The GPT models only have results on the `*_nano` datasets, so the top-k
  differs by dataset:
  - `commonlid`: py3langid, GlotLID, commonlingua, cld2, fasttext
  - `commonlid_nano`: GPT-5, py3langid, GlotLID, GPT-5-mini, GPT-4o
- Without `--collapse-macrolanguages`, macrolanguage code mismatches
  (`lvs`→`lav`, `cmn`→`zho`, `zsm`→`msa`, `arz`/`ars`/`apd`→`ara`,
  `gaz`→`orm`, `bik`↔`bcl`) dominate both the error and the suspect lists.
  The official metrics use exact matching, so these also lower the
  leaderboard scores.
- `commoncrawl/commonlid-cache_commonlid_nano` is now public, so
  `--with-text` works on `commonlid_nano`. It always worked on `commonlid`.

### `commonlid` (373,230 samples, top 5, macrolanguages collapsed)

- 1,588 samples are flagged `label_suspect`. Spot checks include clear
  label errors:
  - idx 1616: French "Littérature en langue créole en Martinique" labelled
    `acf`
  - idx 3111: Spanish "viernes, 3 de julio de 2015" labelled `arg`
  - idx 1983/2355: Spanish legal text labelled `arb`
  - idx 733: Italian-ish text labelled `vec`, where all 5 models say `ita`
- Highest-error languages:

  | language | n | error_rate | all models wrong | suspect | top confusion |
  |---|---|---|---|---|---|
  | lin | 55 | 1.00 | 100% | 18 | yor (64%) |
  | acf | 603 | 0.89 | 46% | 14 | hat (37%) |
  | fil | 58 | 0.82 | 10% | 45 | tgl (86%) |
  | nyn | 5 | 0.84 | 20% | 2 | lug |

  - `lin`: every model misses every sample. Check whether the labels are
    wrong or whether none of the top 5 supports Lingala. GlotLID should.
  - `fil` vs `tgl`: a labelling convention question more than a model error.

### `commonlid_nano` (1,507 samples, top 5 incl. GPT models, collapsed)

- 24 `label_suspect` samples, all with text. They fall into three groups:
  - **Clear label errors:**
    - Spanish labelled `ara` (idx 268)
    - English labelled `jpn` (541) and `nso` (1187)
    - French labelled `swh` (1406), `gcf` (1413) and `aeb` (96)
    - Russian Wikipedia signature labelled `tat` (920)
  - **Variant pairs where the models may be wrong rather than the label:**
    - `hbo`→`heb` (24, 723, 146, 929): vocalised biblical or rabbinic Hebrew
    - `ell`→`grc` (1015, 1450): polytonic Greek
    - `fil`→`tgl` (452, 978, 1074)
  - **Plausible model confusions:** `vec`→`lij`/`ita`, `uzb`→`uig`,
    `crh`→`tat`, `aze`→`tur`, `oci`→`por`, `jav`→`ind`.
- Lowest-agreement languages, where the models disagree with each other:
  `lin` (0.28), `gcr`, `acf`, `nyn`, `ext`.
- Many high-resource languages (`tel`, `rus`, `por`, `mal`, `ben`, ...) have
  `error_rate` 0 and `all_correct_share` 1.0.

### Effect of macrolanguage handling on the `commonlid_nano` ranking

Macro F1 (gold-only) recomputed from the predictions. The exact-match column
reproduces the published leaderboard. Two variants:

- *Collapsed*: gold and prediction both mapped to their macrolanguage; gold
  classes merge from 101 to 83.
- *Lenient*: the 101 gold languages are kept, and a prediction is correct if it
  is in the same macrolanguage as the gold label. Siblings count too
  (`arz` gold, `arb` predicted).

| model | exact | collapsed | lenient | rank exact → collapsed → lenient |
|---|---|---|---|---|
| GPT-5 | 0.725 | 0.867 | 0.884 | 1 → 1 → 2 |
| py3langid | 0.724 | 0.857 | 0.876 | 2 → 3 → 3 |
| GlotLID | 0.702 | 0.866 | 0.885 | 3 → 2 → 1 |
| GPT-5-mini | 0.665 | 0.833 | 0.848 | 4 → 4 → 4 |
| GPT-4o | 0.659 | 0.821 | 0.841 | 5 → 5 → 5 |

- The top 5 stay the same; only the top three reorder. GlotLID gains the
  most, going from 0.023 behind GPT-5 to 0.002 ahead under lenient matching.
- All models gain about 0.14–0.18 macro F1, which is much more than the gaps
  between them.
- Elsewhere there are only adjacent swaps: commonlingua and
  GoogleTranslate-v3 (6/7, lenient), cld2 and fasttext (10/11).
- The top-three gaps of 0.001–0.002 are likely within noise: 1,507 samples,
  as few as 5 per language.
- Method: per model, `compute_per_language_metrics` + `macro_average` on
  `load_predictions(...)`, with `to_macrolanguage` applied to gold and/or
  pred. Not yet part of the CLI (see next steps).

## Reproduce

```bash
make install   # needs uv >= 0.9: older uv cannot parse `exclude-newer = "7 days"`
               # and silently re-resolves uv.lock. Use `uv run --frozen ...`.
commonlid breakdown --dataset commonlid -k 5 --collapse-macrolanguages
commonlid breakdown --dataset commonlid_nano -k 5 --level sample \
  --only-suspect --collapse-macrolanguages --with-text --out nano_suspects.csv
commonlid breakdown --dataset commonlid_nano -k 5 --sort-by agreement --order asc
```

## Next steps

1. **Review the suspects.** Export the full `commonlid` suspect list with text
   (`--level sample --only-suspect --collapse-macrolanguages --with-text`)
   and triage it by hand or with an LLM judge. Group by `(gold,
   consensus_pred)` to find systematic labelling issues rather than
   one-offs. Feed confirmed errors back into the CommonLID annotations.
2. **Investigate `lin` and `acf`.** Check model support for these languages
   (`commonlid generate-support-matrix`) against the labels, and use
   `--model` to compare models that do support them.
3. **Decide on macrolanguages.** Decide whether the official metrics should
   also count macrolanguage matches, or report them as a separate view. The
   breakdown shows how much they affect the numbers. On `commonlid_nano` the
   top 5 are stable but the top three reorder (see above). Repeat the ranking
   comparison on full `commonlid`, add bootstrap confidence intervals, and
   consider making it a reusable function or leaderboard toggle.
4. **Get a more independent ensemble.** Rerun with
   `--exclude-model`/`--model` to mix architectures (n-gram, neural, LLM)
   so the agreement signal is less correlated. Tune
   `--suspect-min-agreement` (0.8 is a first guess).
5. **Other datasets.** Run on `flores_dev`, `udhr`, `smolsent_300`,
   `bibles_300` and their nano variants.
6. **Leaderboard.** Reuse `language_breakdown` / `sample_breakdown` in a
   leaderboard drill-down tab: per-language error and agreement, plus a
   suspect-sample browser. These are pure pandas functions with Hub loading
   kept separate.
7. **Small polish.**
   - Left-align the `text` column in the stdout preview.
   - The existing `LIDDataset` private-cache fallback does not recognise
     `datasets.DatasetNotFoundError` as an access error (`_is_hf_access_error`
     in `core/lid_dataset.py`). That should be fixed separately.
8. **Housekeeping.** The commits are on the fork `malteos/commonlid-eval-2`.
   Pushing to `commoncrawl/commonlid-eval` failed (no access from that
   session). Open the PR from the fork or push the branch upstream when
   ready.
