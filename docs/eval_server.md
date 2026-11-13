# Scoring server

`commonlid serve` runs a small HTTP server that scores an agent's language
predictions **without giving the agent the gold labels**. Run the server in
an environment that holds the labels. Let the agent reach it only over HTTP.

```
Agent environment (limited network)         Eval environment (labels, HF access)
  GET  /datasets/{id}/inputs        ───►    [{id, text}] + task instructions
  POST /datasets/{id}/submissions   ───►    configured metrics only
```

Each server serves one agent. Bind it to an address only that agent can
reach, and set an API token when it listens on anything but localhost (in a
container it always does). To evaluate several agents, run one server (or
container) with its own `state_dir` per agent.

## Install and run

```bash
pip install "commonlid[server]"     # fastapi + uvicorn
commonlid list-server-metrics       # metric names a config may expose
commonlid serve --config docs/eval_server.example.toml [--host 0.0.0.0] [--port 8000]
```

The server loads every configured dataset at start-up (from Hugging Face,
using the pinned revisions), so the eval environment needs HF access. The
agent environment does not. For a container, see
[Running in Docker](#running-in-docker).

## Configuration

See [`eval_server.example.toml`](eval_server.example.toml).

| Key | Default | Meaning |
|---|---|---|
| `max_submissions` | `10` | Accepted submissions across all evaluation datasets. `0` = unlimited. |
| `state_dir` | none | Keeps the submission counter, the shuffle seed and an archive of every submission. Without it the counter resets on restart. |
| `round_digits` | `4` | Decimal places of returned scores. |
| `shuffle` | `true` | Serve items in a seeded random order under opaque ids `0..n-1`. |
| `shuffle_seed` | generated | Fixed seed; otherwise one is generated and stored in `state_dir`. |
| `host` / `port` | `127.0.0.1` / `8000` | Bind address; `--host` / `--port` override. |
| `api_token` | none | Bearer token required on every endpoint except `/health`. `COMMONLID_SERVER_TOKEN` overrides it. |
| `per_language_metrics` | `false` | Allow `per_language.*` metrics on evaluation datasets. `COMMONLID_SERVER_PER_LANGUAGE_METRICS=1` (or `0`) overrides it. |
| `[[datasets]] id` | required | A registered dataset id (`commonlid list-datasets`), or any name for a dataset with `source`. |
| `[[datasets]] source` | `id` | Registered dataset to derive this one from ([below](#derived-datasets)). |
| `[[datasets]] exclude` | none | Datasets whose texts are removed. |
| `[[datasets]] min_chars` / `max_chars` | none | Drop shorter / longer texts. |
| `[[datasets]] sample` | none | Stratified sample: `{ max_size = N, seed = S, min_per_language = 5 }`. |
| `[[datasets]] metrics` | `["macro.f1_gold_only"]` | Metrics returned for this dataset. |
| `[[datasets]] macro_match` | `"exact"` | How predictions are matched with gold labels: `exact`, `macro` or `sibling` ([below](#matching)). |
| `[[datasets]] expose_labels` | `false` | Development dataset: serve the gold labels with the inputs ([below](#evaluation-and-development-datasets)). |
| `[[datasets]] description` | none | Free text added to the dataset's instructions for the agent. |
| `[[datasets]] limit` | all | Serve and score only the first N samples (after the steps above). |
| `[[datasets]] max_submissions` | none | Cap for this dataset, on top of the global one. |
| `[[datasets]] sample_count_threshold` | `0` | Drop languages with fewer gold samples from the metrics. |

### Metrics

Metric names are dotted paths into `summary.json`: `macro.<key>`,
`micro.<key>`, `per_language.<field>` (for example `per_language.f1`),
`n_samples` and `n_samples_with_gold`. Scores come back with the same
nesting, e.g. `{"macro": {"f1_gold_only": 0.8123}}`. Metric code is shared
with `commonlid run`, so with `macro_match = "exact"` a submission scores
exactly like the same predictions from a registered model.

Per-language metrics are off by default. Every `per_language.*` metric
returns one entry per language of the gold set (and per predicted language),
so a single submission tells the agent which languages the test set
contains. `per_language.gt_count`, `correct` and `predictions` also give
away how many samples each language has. A config that lists them for an
evaluation dataset is rejected unless `per_language_metrics = true` or
`COMMONLID_SERVER_PER_LANGUAGE_METRICS=1`, and the server logs a warning at
start-up when they are on. Development datasets may always use them.

### Environment variables

Environment variables override the file, so a container can keep the token
out of the mounted config:

| Variable | Overrides |
|---|---|
| `COMMONLID_SERVER_TOKEN` | `api_token` |
| `COMMONLID_SERVER_PER_LANGUAGE_METRICS` | `per_language_metrics` (`1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`) |

## Evaluation and development datasets

A dataset is an **evaluation dataset** by default: the agent gets only the
texts, and every accepted submission uses the global `max_submissions`
budget and the dataset's own cap.

With `expose_labels = true`, it is a **development dataset**: each item also
carries its gold `label`, so the agent can check its approach before it
spends the evaluation budget. Its scores reveal nothing the agent does not
already have, so:

- its submissions only count against its own `max_submissions`, not the
  global budget;
- it may return `per_language.*` metrics without `per_language_metrics`.

The server refuses to start if a development dataset shares any text with an
evaluation dataset (compared after whitespace and Unicode normalisation).
For example, `commonlid_nano` is a subset of `commonlid`, so exposing the
labels of `commonlid` would hand out the test set. Use a dataset from
another source, such as `flores_dev`, or derive one from the same source
without the evaluation texts, as below.

## Derived datasets

A `[[datasets]]` entry with `source` serves a dataset built from a registered
one, so dev and train sets need no dataset class. The steps run in this
order, each on what the previous one left:

1. `exclude`: drop every text that occurs in the listed datasets. A
   registered dataset is excluded in full, whatever `limit` or filters it is
   served with in this config; a dataset defined only in this config, as it
   is served. Texts are compared after whitespace and Unicode normalisation,
   so duplicates in the source (CommonLID has about 1,800) are caught as well.
2. `min_chars` / `max_chars`: drop texts outside the length range, counted
   after collapsing whitespace runs and stripping the ends.
3. `sample`: the stratified sampler of the nano datasets. Languages with
   fewer than `min_per_language` texts (default 5) are dropped, every other
   language keeps `min_per_language` texts, and `max_size` more are spread
   over the languages in proportion to what each has left: about
   `max_size + min_per_language × languages` texts in total. `seed` is
   required.
4. `limit`: keep the first N.

```toml
[[datasets]]              # hidden evaluation set
id = "commonlid_nano"

[[datasets]]              # labelled dev set: no nano texts, nothing under 20 characters
id = "commonlid_dev"
source = "commonlid"
exclude = ["commonlid_nano"]
min_chars = 20
sample = { max_size = 1000, seed = 1 }
expose_labels = true

[[datasets]]              # more labelled data, disjoint from both
id = "commonlid_train"
source = "commonlid"
exclude = ["commonlid_nano", "commonlid_dev"]
min_chars = 20
sample = { max_size = 5000, seed = 2 }
expose_labels = true
```

Without `sample`, `commonlid_train` would serve everything left (about
350,000 texts). Datasets are built after the ones they exclude, and a cycle
is a config error.

The result depends only on the source revision (pinned in its dataset
class), the excluded datasets and these settings, so the same config yields
the same items on every machine; the shuffle seed changes only their order
and ids. At start-up the server logs each step's count and a content hash
per dataset (also kept in `state.json`); equal hashes mean equal items:

```
Derived commonlid_dev from commonlid: 373230 -> exclude commonlid_nano: 371708 -> min_chars=20: 351353 -> sample max_size=1000 min_per_language=5 seed=1: 1479
Serving commonlid_dev: 1479 items, 96 languages, content hash 3373a2d87d89ab15 (labels exposed)
```

The filters apply to evaluation datasets too (`id = "commonlid_nano"` with
`min_chars = 20` serves nano without its short texts). In the submission
archive, `idx` is the position in the source dataset.

## Matching

`macro_match` sets when a prediction counts as correct, as in
[`commonlid breakdown`](breakdown.md#macrolanguage-matching):

| mode | counts as correct |
|---|---|
| `exact` (default) | the same code, as on the leaderboard |
| `macro` | also an individual language and its own macrolanguage (`lvs`/`lav`, `cmn`/`zho`) |
| `sibling` | also two individual languages of the same macrolanguage (`arz`/`arb`, `cmn`/`yue`) |

An agent cannot see which convention the gold labels follow, so `macro`
avoids penalising it for `zho` where the gold label says `cmn`. Keep `exact`
when the scores must be comparable with the leaderboard. `sibling` is lenient
and best treated as an upper bound. A match under the mode is scored as the
gold label, for the metrics and for `correct` in the archive.

## API

Interactive docs are served at `/docs`. With an `api_token`, every endpoint
except `/health` needs `Authorization: Bearer <token>` and answers `401`
without it.

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/status` | Datasets (item count, description, metrics, matching mode, labels exposed, used/remaining submissions, final submission) and the global budget. |
| `GET` | `/datasets/{id}/inputs?offset=&limit=` | `{"dataset_id", "total", "offset", "labels_exposed", "instructions", "items": [{"id", "text"}]}`. Items also have `"label"` on development datasets. |
| `POST` | `/datasets/{id}/submissions` | Body `{"predictions": [{"id": 0, "pred": "eng"}, ...], "final": false}`. |
| `GET` | `/submissions` | Accepted submissions, whether each was final, and the scores that were returned. |

### Instructions for the agent

The inputs endpoint returns `instructions`, a plain-text task description
generated for the dataset. It covers the task, the request format, the code
format, the matching mode, the returned metrics, the remaining budget and the
`final` flag. An operator can hand it to the agent as is, or point the agent
at the endpoint. For an evaluation dataset with 3 submissions left (an
excerpt; the real text has the actual item count):

```text
Task: identify the language of each of the 6 texts in dataset 'commonlid'.
Gold labels are not available for this dataset.

Inputs: GET /datasets/commonlid/inputs returns items with "id", "text". Page through them with ?offset=&limit=. Ids run from 0 to 5 in a shuffled order and carry no information.

Submit: POST /datasets/commonlid/submissions with a JSON body like
  {"predictions": [{"id": 0, "pred": "eng"}, {"id": 1, "pred": "cmn"}], "final": false}
- Include every id from 0 to 5 exactly once.
- "pred" is an ISO 639-3 code: three lowercase letters, for example "eng", "deu", "cmn" or "swh". Use null if you cannot tell; it counts as wrong.
- ISO 639-1 codes ("en"), ISO 639-2/B codes ("ger") and upper-case codes are converted to ISO 639-3 and reported under "warnings". Anything else, such as language names ("English"), locale or script tags ("en-US", "eng_Latn") or unknown codes, is rejected with status 422: the response lists the invalid values and the submission is not counted.
- Codes must match the gold label exactly. A macrolanguage code (for example "zho" or "ara") does not match an individual-language label (for example "cmn" or "arb"), and the other way round.
- Returned scores: macro.f1_gold_only.
- Budget: 3 submission(s) left for this dataset, 10 left overall (the overall budget is shared by all evaluation datasets). Rejected submissions (malformed, invalid codes, over budget) are not counted.
- Set "final": true on the submission you want evaluated. The dataset accepts no further submissions after it. Without a final submission, your last accepted one counts.
```

### Submissions

A submission must contain every item id exactly once. Each `pred` is
checked:

| `pred` | result |
|---|---|
| an ISO 639-3 code (`"eng"`, `"cmn"`, `"und"`) | scored as is |
| `null` | scored as `und` (wrong) |
| an ISO 639-1 (`"en"`) or ISO 639-2/B (`"ger"`) code, upper case (`"ENG"`), surrounding spaces, or a deprecated code with one successor (`"iw"`) | converted to ISO 639-3, scored, and reported in `warnings` |
| anything else: names (`"English"`), locale or script tags (`"en-US"`, `"eng_Latn"`), unknown or retired codes (`"xyz"`, `"mo"`) | the whole submission is rejected with `422` and not counted |

Responses:

- `200`: `{"submission_id", "dataset_id", "timestamp", "final", "scores", "warnings", "submissions_used", "submissions_remaining", "dataset_submissions_used", "dataset_submissions_remaining"}`.
  `warnings` lists converted codes, e.g. `"3 prediction(s) were converted to ISO 639-3: 'en' -> 'eng' (2), 'GER' -> 'deu' (1)"`.
- `422`: malformed submission, **not counted**. `detail` describes every
  problem at once. `invalid_predictions` lists invalid values as
  `{"pred", "reason", "count", "example_ids"}`, and `id_errors` the unknown,
  duplicate and missing ids.
- `409`: the dataset is closed because a submission was marked final. **Not counted.**
- `429`: submission budget used up. **Not counted.**
- `404`: unknown dataset.
- `401`: missing or wrong bearer token (when `api_token` is set).

### Final submissions

`"final": true` marks the submission to evaluate. It is scored and counted
like any other, and afterwards the dataset answers `409` to further
submissions. `/status` shows it as `final_submission_id`, `/submissions` as
`"final": true`, and `state.json` keeps it across restarts. When an agent
never marks a submission final, the convention is that its last accepted
submission counts.

### Minimal agent-side client

```python
import requests

base = "http://eval-server:8000"
headers = {"Authorization": "Bearer <token>"}  # if the server sets api_token
page = requests.get(f"{base}/datasets/commonlid/inputs", headers=headers).json()
print(page["instructions"])
preds = [{"id": it["id"], "pred": my_lid(it["text"])} for it in page["items"]]
resp = requests.post(
    f"{base}/datasets/commonlid/submissions",
    json={"predictions": preds, "final": True},
    headers=headers,
)
print(resp.status_code, resp.json())
```

## Running in Docker

The repository's `Dockerfile` builds an image with the `[server]` extra. It
runs `commonlid serve --config /config/server.toml --host 0.0.0.0 --port 8000`
as a non-root user (uid 1000). The
[Docker image workflow](../.github/workflows/docker.yml) pushes it to
`ghcr.io/commoncrawl/commonlid-server`: `:main` and `:sha-<commit>` for every
push to `main`, and `:<version>` and `:latest` for every release.

```bash
docker pull ghcr.io/commoncrawl/commonlid-server:latest   # or: docker build -t commonlid-server .
docker run -d --name commonlid-server -p 8000:8000 \
  -e COMMONLID_SERVER_TOKEN="$(openssl rand -hex 32)" \
  -e HF_TOKEN \
  -v "$PWD/server.toml:/config/server.toml:ro" \
  -v commonlid-state:/state \
  -v commonlid-hf:/hf-cache \
  ghcr.io/commoncrawl/commonlid-server:latest
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
- `HF_TOKEN`: needed for gated or private datasets. `commonlid` itself is
  gated (request access on the
  [dataset page](https://huggingface.co/datasets/commoncrawl/CommonLID)), and
  the Bibles, SmolSent and social-media caches are private. The token stays
  in the server's container.
- The container is healthy once `/health` answers. Datasets load first, so
  the health check allows ten minutes to start.

Put the agent on a network that can reach the server but not the internet
(for example a Docker network created with `--internal`, joined by both
containers, with the server also attached to a network that can reach
Hugging Face).

`docker/smoke_test.sh <image>` starts an image on a small public dataset. It
checks the token, the instructions, the rejection of an invalid code, a final
submission and the `409` after it. CI runs it on every PR, and the image
workflow runs it before every push.

## Restarts and the submission archive

With `state_dir`, a restart keeps the counter, the shuffle seed, final
submissions and therefore the item ids. `state.json` also records a
fingerprint of the served items per dataset. If the items change after a
dataset has accepted submissions (another `shuffle_seed`, another `limit`, or
a new dataset revision), the ids an agent already holds would point to other
texts, so the server refuses to start. Restore the previous settings, or
delete `state.json` to start over, which resets the counter. Before the first
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
  know the language set (see above). For iteration, give the agent a
  development dataset and keep each evaluation dataset at one or a few
  submissions.
- **Rejected submissions are free.** Validation only looks at the
  submission itself (ids and the format of the codes), which carries no
  label information. Predicted codes are not checked against the gold label
  set for the same reason.
- **`state_dir` contains gold labels.** `submissions/<dataset>/submission-NNNN/`
  holds the full `summary.json` and a `predictions.jsonl` with `gold`, the
  submitted (`pred_raw`) and conformed (`pred`) code, and `correct` per item.
  Keep it out of the agent's reach. Delete `state.json` to reset the counter;
  a new shuffle seed is then generated unless `shuffle_seed` is set.
