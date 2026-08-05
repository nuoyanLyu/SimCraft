#!/usr/bin/env bash
# Run tau2-bench against the EnvFactory model (agent) with a dmxapi DeepSeek-V3.2
# user simulator, per the EnvFactory paper setup. Runs in the `factory` conda env.
#
# Prereq: serve the model first (hermes tool parser), e.g.
#   bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
#
# Usage:
#   bash scripts/eval/tau2/run_tau2.sh [DOMAIN]           # DOMAIN: mock|airline|retail|telecom|banking_knowledge
#   NUM_TASKS=1 bash scripts/eval/tau2/run_tau2.sh mock    # tiny smoke
#
# Env: NUM_TASKS, AGENT_ENDPOINT, SERVED_NAME, RUN_DIR, USER_TEMP
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/../common/envfactory_config.py"
TAU2_DIR="${TAU2_DIR:-/root/autodl-tmp/dataset/agent/tau2-bench}"
EF_ROOT="${EF_ROOT:-/root/SimCraft}"
BENCH_PY="${BENCH_PY:-python}"

DOMAIN="${1:-mock}"
NUM_TASKS="${NUM_TASKS:-1}"
AGENT_ENDPOINT="${AGENT_ENDPOINT:-http://localhost:8001/v1}"
SERVED_NAME="${SERVED_NAME:-Qwen3-8B}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/eval_runs/tau2}"
# Agent = model under test (paper temp 0.7); User simulator = dmxapi DeepSeek-V3.2.
AGENT_TEMP="$("$BENCH_PY" "$CONFIG" --print temperature)"
USER_TEMP="${USER_TEMP:-0.0}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://www.dmxapi.cn/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V3.2}"
DMX_KEY_VAR="${DMX_KEY_VAR:-EMBEDDING_API_KEY}"
DMX_KEY="$(grep -E "^${DMX_KEY_VAR}=" "$EF_ROOT/.env" | cut -d= -f2-)"
if [ -z "$DMX_KEY" ]; then echo "[ERROR] $DMX_KEY_VAR not found in $EF_ROOT/.env" >&2; exit 1; fi
# Embedding calls (tau2 banking retriever / vita evaluator use text-embedding-3-large)
# have no explicit api_base -> route them to dmxapi (which serves that model).
export OPENAI_API_KEY="$DMX_KEY"
export OPENAI_BASE_URL="https://www.dmxapi.cn/v1"

mkdir -p "$RUN_DIR"
SAVE_TO="$RUN_DIR/${DOMAIN}_sim.json"

# pre-flight: agent endpoint reachable
if ! curl -sf "$AGENT_ENDPOINT/models" | grep -q "$SERVED_NAME"; then
    echo "[ERROR] agent endpoint $AGENT_ENDPOINT not serving \"$SERVED_NAME\". Start serve_model.sh first." >&2
    exit 1
fi

echo "[run_tau2] domain=$DOMAIN num_tasks=$NUM_TASKS agent=openai/$SERVED_NAME@$AGENT_ENDPOINT (temp=$AGENT_TEMP)"
echo "[run_tau2] user-sim=openai/$JUDGE_MODEL@dmxapi (temp=$USER_TEMP)  save_to=$SAVE_TO"

if [ "$NUM_TASKS" = "all" ]; then NT_ARGS=(); else NT_ARGS=(--num-tasks "$NUM_TASKS"); fi
cd "$TAU2_DIR"
# Note: user-llm-args intentionally omits api_key. tau2 saves llm_args verbatim into
# the result JSON, so passing the raw dmxapi key here would leak it into every
# simulation file on disk. litellm falls back to the OPENAI_API_KEY env var
# (exported above) for the "openai/" provider even with a custom api_base.
"$BENCH_PY" -m tau2.cli run \
    --domain "$DOMAIN" \
    --agent-llm "openai/$SERVED_NAME" \
    --agent-llm-args "{\"temperature\": $AGENT_TEMP, \"api_base\": \"$AGENT_ENDPOINT\", \"api_key\": \"EMPTY\"}" \
    --user-llm "openai/$JUDGE_MODEL" \
    --user-llm-args "{\"temperature\": $USER_TEMP, \"api_base\": \"$JUDGE_BASE_URL\"}" \
    "${NT_ARGS[@]}" \
    --num-trials 1 \
    --max-concurrency 1 \
    --save-to "$SAVE_TO"

echo "[run_tau2] done. Simulation saved to: $SAVE_TO"
