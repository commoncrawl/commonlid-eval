#!/usr/bin/env bash
# Smoke-test the scoring-server image: start it on a small public dataset,
# check auth, fetch inputs and score one submission.
#
#   docker build -t commonlid-server . && docker/smoke_test.sh commonlid-server
set -euo pipefail

image="${1:-commonlid-server}"
port="${PORT:-18000}"
name="commonlid-smoke-$$"
token="smoke-token"
workdir="$(mktemp -d)"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true; rm -rf "$workdir"' EXIT

cat > "$workdir/server.toml" <<'TOML'
max_submissions = 1
state_dir = "/state"
[[datasets]]
id = "udhr_nano"
limit = 20
TOML
chmod a+r "$workdir/server.toml"

docker run -d --name "$name" -p "127.0.0.1:$port:8000" \
  -e COMMONLID_SERVER_TOKEN="$token" \
  -v "$workdir/server.toml:/config/server.toml:ro" \
  "$image" >/dev/null

base="http://127.0.0.1:$port"
for _ in $(seq 1 90); do
  curl -sf "$base/health" >/dev/null && break
  if [ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]; then
    docker logs "$name"; echo "container exited" >&2; exit 1
  fi
  sleep 2
done
curl -sf "$base/health" >/dev/null || { docker logs "$name"; echo "no /health" >&2; exit 1; }

code="$(curl -s -o /dev/null -w '%{http_code}' "$base/status")"
[ "$code" = "401" ] || { echo "expected 401 without token, got $code" >&2; exit 1; }

auth=(-H "Authorization: Bearer $token")
curl -sf "${auth[@]}" "$base/datasets/udhr_nano/inputs" > "$workdir/inputs.json"
python3 - "$workdir/inputs.json" "$workdir/submission.json" <<'PY'
import json, sys
items = json.load(open(sys.argv[1]))["items"]
assert items and all(set(it) == {"id", "text"} for it in items), items[:1]
json.dump({"predictions": [{"id": it["id"], "pred": "eng"} for it in items]}, open(sys.argv[2], "w"))
PY
curl -sf "${auth[@]}" -H 'Content-Type: application/json' \
  -d @"$workdir/submission.json" "$base/datasets/udhr_nano/submissions" > "$workdir/result.json"
python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); assert r["submissions_remaining"] == 0, r; print("score:", r["scores"])' "$workdir/result.json"

docker exec "$name" test -f /state/submissions/udhr_nano/submission-0001/predictions.jsonl
echo "smoke test passed"
