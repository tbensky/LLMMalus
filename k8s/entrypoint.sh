#!/bin/sh
# What the lab container runs: wait for the Ollama sidecar, run the batch,
# then tell the sidecar to stop so the Job can complete.
#
# Settings come from the Job's environment (see k8s/job-gpu.yaml):
#   MALUS_MODEL      Ollama model tag, e.g. qwen3:8b
#   MALUS_RUNS       independent runs, one after another
#   MALUS_NUM_CTX    context window handed to Ollama per request
#   MALUS_TIMEOUT    seconds allowed per model call
#   MALUS_ARGS       anything else for run_lab.py, e.g. "--keep-last 10"
set -u

OLLAMA_URL="${MALUS_OLLAMA_URL:-http://localhost:11434}"
DONE_FILE="${MALUS_DONE_FILE:-/signal/lab.done}"
OUT_DIR="${MALUS_OUT_DIR:-/workspace/Runs}"

# Always release the sidecar, whatever happens below; otherwise the Ollama
# container runs forever and the Job never completes.
trap 'touch "$DONE_FILE" 2>/dev/null || true' EXIT

echo "malus: model=${MALUS_MODEL} runs=${MALUS_RUNS} num_ctx=${MALUS_NUM_CTX}" \
     "timeout=${MALUS_TIMEOUT} args=${MALUS_ARGS:-} version=${MALUS_VERSION:-unknown}"

# Ollama starts alongside this container; give it time to listen.
python3 - "$OLLAMA_URL" "${MALUS_OLLAMA_WAIT:-300}" <<'PY' || exit 1
import sys, time, urllib.request
url, wait = sys.argv[1].rstrip("/") + "/api/tags", float(sys.argv[2])
deadline = time.monotonic() + wait
while True:
    try:
        with urllib.request.urlopen(url, timeout=5):
            print("malus: Ollama is up")
            break
    except Exception as exc:
        if time.monotonic() > deadline:
            sys.exit(f"malus: Ollama did not answer at {url} within {wait:.0f} s ({exc})")
        time.sleep(2)
PY

# Which node and pod this batch ran on, next to its runs. The GPU is a
# variable too: look the node up with
#   kubectl get node <node> -L nvidia.com/gpu.product
model_dir="$(python3 -c 'import sys; from malus.lab import safe_model_name; print(safe_model_name(sys.argv[1]))' "$MALUS_MODEL")"
mkdir -p "$OUT_DIR/$model_dir"
printf '%s pod=%s node=%s version=%s runs=%s num_ctx=%s timeout=%s args=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${POD_NAME:-?}" "${NODE_NAME:-?}" \
  "${MALUS_VERSION:-unknown}" "$MALUS_RUNS" "$MALUS_NUM_CTX" "$MALUS_TIMEOUT" \
  "${MALUS_ARGS:-}" >> "$OUT_DIR/$model_dir/jobs.log"

# MALUS_ARGS is split into words on purpose.
# shellcheck disable=SC2086
python3 -u run_lab.py --provider ollama --llm-url "$OLLAMA_URL/api/chat" \
  --model "$MALUS_MODEL" --runs "$MALUS_RUNS" --num-ctx "$MALUS_NUM_CTX" \
  --timeout "$MALUS_TIMEOUT" --out-dir "$OUT_DIR" ${MALUS_ARGS:-}
status=$?
echo "malus: run_lab.py exited with status $status"
exit "$status"
