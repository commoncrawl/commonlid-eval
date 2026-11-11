# Scoring server

`commonlid serve` runs a small HTTP server that scores an agent's language
predictions **without giving the agent the gold labels**. Run the server in
an environment that holds the labels. Let the agent reach it only over HTTP.

```
Agent environment (limited network)         Eval environment (labels, HF access)
  GET  /datasets/{id}/inputs        ───►    [{id, text}], no labels
  POST /datasets/{id}/submissions   ───►    configured metrics only
```

Each server serves one agent. Bind it to an address only that agent can
reach, and set an API token when it listens on anything but localhost (in a
container it always does).

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
| `api_token` | none | Bearer token required on every endpoint except `/health`. `COMMONLID_SERVER_TOKEN` overrides it. |
| `per_language_metrics` | `false` | Allow `per_language.*` metrics. `COMMONLID_SERVER_PER_LANGUAGE_METRICS=1` (or `0`) overrides it. |
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

Per-language metrics are off by default. Every `per_language.*` metric
returns one entry per language of the gold set (and per predicted language),
so a single submission tells the agent which languages the test set
contains. `per_language.gt_count`, `correct` and `predictions` also give
away how many samples each language has. A config that lists them is
rejected unless `per_language_metrics = true` or
`COMMONLID_SERVER_PER_LANGUAGE_METRICS=1`, and the server logs a warning at
start-up when they are on.

Environment variables override the file, so a container can keep the token
out of the mounted config:

| Variable | Overrides |
|---|---|
| `COMMONLID_SERVER_TOKEN` | `api_token` |
| `COMMONLID_SERVER_PER_LANGUAGE_METRICS` | `per_language_metrics` (`1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`) |

## API

Interactive docs are served at `/docs`. With an `api_token`, every endpoint
except `/health` needs `Authorization: Bearer <token>` and answers `401`
without it.

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
- `401`: missing or wrong bearer token (when `api_token` is set).

Minimal agent-side client:

```python
import requests

base = "http://eval-server:8000"
headers = {"Authorization": "Bearer <token>"}  # if the server sets api_token
items = requests.get(f"{base}/datasets/commonlid/inputs", headers=headers).json()["items"]
preds = [{"id": it["id"], "pred": my_lid(it["text"])} for it in items]
resp = requests.post(
    f"{base}/datasets/commonlid/submissions", json={"predictions": preds}, headers=headers
)
print(resp.json())
```

## Running in Docker

The repository's `Dockerfile` builds an image with the `[server]` extra. It
runs `commonlid serve --config /config/server.toml --host 0.0.0.0 --port 8000`
as a non-root user (uid 1000).

```bash
docker build -t commonlid-server .
docker run -d --name commonlid-server -p 8000:8000 \
  -e COMMONLID_SERVER_TOKEN="$(openssl rand -hex 32)" \
  -e HF_TOKEN \
  -v "$PWD/server.toml:/config/server.toml:ro" \
  -v commonlid-state:/state \
  -v commonlid-hf:/hf-cache \
  commonlid-server
```

- `/config/server.toml`: the config. The image ships
  [`docker/server.toml`](../docker/server.toml) (CommonLID, macro F1,
  10 submissions); mount your own over it. Use `state_dir = "/state"`: a
  relative path would resolve inside the read-only `/config`. `host` and
  `port` in the file are ignored, because the command sets them.
- `/state`: the counter, the shuffle seed and the archive with gold labels.
  Use a volume so they survive the container; a bind-mounted host directory
  must be writable by uid 1000. Never mount it into the agent's container.
- `/hf-cache` (`HF_HOME`): the downloaded datasets, so a restart does not
  download them again.
- `HF_TOKEN`: only needed for datasets whose cache is private (the Bibles,
  SmolSent and social-media sets). It stays in the server's container.
- The container is healthy once `/health` answers. Datasets load first, so
  the health check allows ten minutes to start.

Put the agent on a network that can reach the server but not the internet
(for example a Docker network created with `--internal`, joined by both
containers, with the server also attached to a network that can reach
Hugging Face). `docker/smoke_test.sh <image>` starts an image on a small
public dataset and runs one scored submission; CI runs it on every PR.

## Restarts and the submission archive

With `state_dir`, a restart keeps the counter, the shuffle seed and
therefore the item ids. `state.json` also records a fingerprint of the served
items per dataset. If the items change after a dataset has accepted
submissions (another `shuffle_seed`, another `limit`, or a new dataset
revision), the ids an agent already holds would point to other texts, so the
server refuses to start. Restore the previous settings, or delete
`state.json` to start over, which resets the counter. Before the first
submission on a dataset, a change only logs a warning.

Every accepted submission is archived as
`state_dir/submissions/<dataset>/submission-NNNN/` with `summary.json` and
`predictions.jsonl`, in the layout of `commonlid run --output-dir`. The rows
are in dataset order, keyed by the dataset `idx` like model predictions, plus
the agent-facing `id` and the submitted code as `pred_raw`. So the
breakdown tool reads them directly:

```bash
commonlid breakdown --local-dir eval-server-state/submissions \
  --dataset commonlid --model submission-0001 --model submission-0002 --level model
```

To compare a submission with published models, copy the models'
`<dataset>/<model>/predictions.jsonl` from the results dataset into the same
directory and add them with `--model`.

## Operator notes and leakage

- **Lock down the agent's network.** The test sets are public on Hugging Face
  (`commoncrawl/CommonLID`, the nano caches), and
  `commoncrawl/commonlid-results` publishes per-text gold labels keyed by text
  hash. Block `huggingface.co`, `*.hf.co`, the GitHub repo and web search from
  the agent environment, or allowlist only what the agent needs.
- **Every score is information.** With unlimited submissions and fine-grained
  scores, an agent can recover labels by changing single predictions and
  watching the score. Keep `max_submissions` low, round scores, and prefer
  aggregate metrics. Leave `per_language_metrics` off unless the agent may
  know the language set (see above). For iteration, give the agent a dev dataset with a
  generous budget and keep the test dataset at one or a few submissions.
- **Malformed submissions are free** because validation only checks ids,
  which carries no label information. Predicted codes are not checked
  against the gold label set for the same reason.
- **`state_dir` contains gold labels.** `submissions/<dataset>/submission-NNNN/`
  holds the full `summary.json` and a `predictions.jsonl` with `gold`, the
  submitted (`pred_raw`) and conformed (`pred`) code, and `correct` per item.
  Keep it out of the agent's reach. Delete `state.json` to reset the counter;
  a new shuffle seed is then generated unless `shuffle_seed` is set.
