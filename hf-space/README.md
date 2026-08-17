---
title: CommonLID Leaderboard
emoji: 🌍
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.14.0
python_version: 3.12
app_file: app.py
pinned: false
license: apache-2.0
---

# CommonLID Leaderboard

Live results table for the [CommonLID](https://huggingface.co/datasets/commoncrawl/CommonLID)
benchmark and its `commonlid_nano` slice. Models are ranked by **macro
F1**; click any row for per-language detail.

## How the data flows

1. Each run writes `<dataset>/<model>/{summary.json, predictions.jsonl}` to
   `data/results/` via `commonlid run …`.
2. The contents of `data/results/` are uploaded to
   [`commoncrawl/commonlid-results`](https://huggingface.co/datasets/commoncrawl/commonlid-results)
   with `huggingface-cli upload`.
3. This Space loads the dataset on boot and renders one tab per benchmark.

## Local preview

```bash
uv sync --extra leaderboard
uv run commonlid leaderboard serve --local-dir ./data/results
```

## Pinning a specific results revision

Set `COMMONLID_RESULTS_REVISION` (or `--revision` on the CLI) to lock the
Space to a specific commit on the results dataset.
