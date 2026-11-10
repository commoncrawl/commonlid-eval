# Scoring server

`commonlid serve` runs a small HTTP server that scores an agent's language
predictions **without giving the agent the gold labels**. Run the server in
an environment that holds the labels. Let the agent reach it only over HTTP.

```
Agent environment (limited network)         Eval environment (labels, HF access)
  GET  /datasets/{id}/inputs        ───►    [{id, text}], no labels
  POST /datasets/{id}/submissions   ───►    configured metrics only
```

Each server serves one agent, so there is no authentication. Bind it to an
address only that agent can reach.

## Install and run

```bash
pip install "commonlid[server]"     # fastapi + uvicorn
commonlid list-server-metrics       # metric names a config may expose
commonlid serve --config docs/eval_server.example.toml [--host 0.0.0.0] [--port 8000]
```

The server loads every configured dataset at start-up (from Hugging Face,
using the pinned revisions), so the eval environment needs HF access. The
agent environment does not.

## Configuration

See [`eval_server.example.toml`](eval_server.example.toml).

| Key | Default | Meaning |
|---|---|---|
| `max_submissions` | `10` | Accepted submissions across all datasets. `0` = unlimited. |
| `state_dir` | none | Keeps the submission counter, the shuffle seed and an archive of every submission. Without it the counter resets on restart. |
| `round_digits` | `4` | Decimal places of returned scores. |
| `shuffle` | `true` | Serve items in a seeded random order under opaque ids `0..n-1`. |
| `shuffle_seed` | generated | Fixed seed; otherwise one is generated and stored in `state_dir`. |
| `host` / `port` | `127.0.0.1` / `8000` | Bind address; `--host` / `--port` override. |
| `[[datasets]] id` | required | A registered dataset id (`commonlid list-datasets`). |
| `[[datasets]] metrics` | `["macro.f1_gold_only"]` | Metrics returned for this dataset. |
| `[[datasets]] limit` | all | Serve and score only the first N samples. |
| `[[datasets]] max_submissions` | none | Extra per-dataset cap on top of the global one. |
| `[[datasets]] sample_count_threshold` | `0` | Drop languages with fewer gold samples from the metrics. |

Metric names are dotted paths into `summary.json`: `macro.<key>`,
`micro.<key>`, `per_language.<field>` (for example `per_language.f1`),
`n_samples` and `n_samples_with_gold`. Scores come back with the same
nesting, e.g. `{"macro": {"f1_gold_only": 0.8123}}`. Metric code is shared
with `commonlid run`, so a submission scores exactly like the same
predictions from a registered model.

## API

Interactive docs are served at `/docs`.

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/status` | Datasets, item counts, returned metrics, used/remaining submissions. |
| `GET` | `/datasets/{id}/inputs?offset=&limit=` | Label-free items `{"total", "offset", "items": [{"id", "text"}]}`. |
| `POST` | `/datasets/{id}/submissions` | Body `{"predictions": [{"id": 0, "pred": "eng"}, ...]}`. |
| `GET` | `/submissions` | Accepted submissions and the scores that were returned. |

A submission must contain every item id exactly once. `pred` is a language
code; ISO 639-1/2 codes are upgraded to ISO 639-3 as for models, and `null`
counts as `und`. Responses:

- `200`: `{"submission_id", "scores", "submissions_used", "submissions_remaining", "dataset_submissions_used", "dataset_submissions_remaining"}`.
- `422`: malformed submission (missing, duplicate or unknown ids). **Not counted.**
- `429`: submission budget used up. **Not counted.**
- `404`: unknown dataset.

Minimal agent-side client:

```python
import requests

base = "http://eval-server:8000"
items = requests.get(f"{base}/datasets/commonlid/inputs").json()["items"]
preds = [{"id": it["id"], "pred": my_lid(it["text"])} for it in items]
print(requests.post(f"{base}/datasets/commonlid/submissions", json={"predictions": preds}).json())
```

## Operator notes and leakage

- **Lock down the agent's network.** The test sets are public on Hugging Face
  (`commoncrawl/CommonLID`, the nano caches), and
  `commoncrawl/commonlid-results` publishes per-text gold labels keyed by text
  hash. Block `huggingface.co`, `*.hf.co`, the GitHub repo and web search from
  the agent environment, or allowlist only what the agent needs.
- **Every score is information.** With unlimited submissions and fine-grained
  scores, an agent can recover labels by changing single predictions and
  watching the score. Keep `max_submissions` low, round scores, and prefer
  aggregate metrics. `per_language.*` metrics also reveal which languages the
  gold set contains. For iteration, give the agent a dev dataset with a
  generous budget and keep the test dataset at one or a few submissions.
- **Malformed submissions are free** because validation only checks ids,
  which carries no label information. Predicted codes are not checked
  against the gold label set for the same reason.
- **`state_dir` contains gold labels.** `submissions/NNNN_<dataset>/` holds
  the full `summary.json` and a `predictions.jsonl` with `gold`, the
  submitted and conformed `pred`, and `correct` per item. Keep it out of the
  agent's reach. Delete `state.json` to reset the counter; a new shuffle seed
  is then generated unless `shuffle_seed` is set.
