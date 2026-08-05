#!/usr/bin/env bash
# Run VitaBench against the EnvFactory model (agent) with dmxapi DeepSeek-V3.2 as the
# user simulator AND evaluator (EnvFactory paper setup). Runs in the `factory` env.
#
# Prereq: serve the model (hermes tool parser):
#   bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
#
# Usage:
#   NUM_TASKS=1 bash scripts/eval/vitabench/run_vita.sh delivery     # smoke
#   bash scripts/eval/vitabench/run_vita.sh cross_domain            # full (paper: VitaBench avg)
#
# VitaBench POSTs directly to each model's base_url, so base_url must be the FULL
# chat-completions URL. Per-model base_url overrides the yaml default (deep-merge).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/../common/envfactory_config.py"
VITA_DIR="/data1/lvnuoyan/dataset/agent/vitabench"
EF_ROOT="/home/lvnuoyan/EnvFactory"

DOMAIN="${1:-delivery}"
NUM_TASKS="${NUM_TASKS:-1}"
LANGUAGE="${LANGUAGE:-english}"
AGENT_ENDPOINT="${AGENT_ENDPOINT:-http://localhost:8100/v1}"
SERVED_NAME="${SERVED_NAME:-envfactory-eval}"
RUN_DIR="${RUN_DIR:-/data1/lvnuoyan/eval_runs/vitabench}"
AGENT_TEMP="$(python "$CONFIG" --print temperature)"
SAVE_TO="${SAVE_TO:-vita_${DOMAIN}_$(date +%s)}"
DMX_KEY="$(grep -E '^EMBEDDING_API_KEY=' "$EF_ROOT/.env" | cut -d= -f2-)"
# Embedding calls (tau2 banking retriever / vita evaluator use text-embedding-3-large)
# have no explicit api_base -> route them to dmxapi (which serves that model).
export OPENAI_API_KEY="$DMX_KEY"
export OPENAI_BASE_URL="https://www.dmxapi.cn/v1"

mkdir -p "$RUN_DIR"
MODELS_YAML="$RUN_DIR/models.yaml"

# pre-flight: agent endpoint reachable
if ! curl -sf "$AGENT_ENDPOINT/models" | grep -q "$SERVED_NAME"; then
    echo "[ERROR] agent endpoint $AGENT_ENDPOINT not serving \"$SERVED_NAME\". Start serve_model.sh first." >&2
    exit 1
fi

# Generate models.yaml (agent -> vLLM, user+evaluator -> dmxapi DeepSeek-V3.2).
cat > "$MODELS_YAML" << YAML
default:
  temperature: 0.0
  max_input_tokens: 32768
  headers:
    Content-Type: "application/json"

models:
  - name: ${SERVED_NAME}
    base_url: ${AGENT_ENDPOINT}/chat/completions
    temperature: ${AGENT_TEMP}
    max_tokens: 8192
    max_input_tokens: 32768
    headers:
      Authorization: "Bearer EMPTY"
      Content-Type: "application/json"
  - name: DeepSeek-V3.2
    base_url: https://www.dmxapi.cn/v1/chat/completions
    temperature: 0.0
    max_tokens: 8192
    max_input_tokens: 32768
    headers:
      Authorization: "Bearer ${DMX_KEY}"
      Content-Type: "application/json"
YAML

# models.yaml must carry the real Authorization header for the live HTTP call (vita's
# yaml loader has no env-var interpolation), but it's only needed for the duration of
# this run -- restrict its permissions and redact it below once the run is done.
chmod 600 "$MODELS_YAML"
export VITA_MODEL_CONFIG_PATH="$MODELS_YAML"
echo "[run_vita] domain=$DOMAIN num_tasks=$NUM_TASKS lang=$LANGUAGE agent=$SERVED_NAME (temp=$AGENT_TEMP) user/eval=DeepSeek-V3.2@dmxapi"

if [ "$NUM_TASKS" = "all" ]; then NT_ARGS=(); else NT_ARGS=(--num-tasks "$NUM_TASKS"); fi
cd "$VITA_DIR"
python -m vita.cli run \
    --domain "$DOMAIN" \
    --agent-llm "$SERVED_NAME" \
    --user-llm DeepSeek-V3.2 \
    --evaluator-llm DeepSeek-V3.2 \
    --enable-think \
    --save-to "$SAVE_TO" \
    --num-trials 1 \
    "${NT_ARGS[@]}" \
    --max-concurrency 1 \
    --language "$LANGUAGE"

# vita embeds the full model config (including the Authorization header) verbatim
# into the saved simulation JSON. Redact the dmxapi key from every artifact this run
# touched so it never sits on disk in plaintext once the run is done.
python3 - "$DMX_KEY" "$MODELS_YAML" "$VITA_DIR/data/simulations/$SAVE_TO" << 'PYEOF'
import sys
key = sys.argv[1]
for path in sys.argv[2:]:
    try:
        with open(path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        continue
    if key and key in text:
        with open(path, "w") as f:
            f.write(text.replace(key, "REDACTED"))
PYEOF

echo "[run_vita] done. Simulations under: $VITA_DIR/data/simulations/"
