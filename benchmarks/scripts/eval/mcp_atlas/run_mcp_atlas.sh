#!/usr/bin/env bash
# Smoke-run MCP-Atlas against the EnvFactory model.
#
# Assumes two long-running services are already up (see mcp_atlas/README.md):
#   1) vLLM endpoint serving the model WITH hermes tool parser (serve_model.sh)
#   2) MCP-Atlas docker sandbox (:1984) + TS harness (:3001)
#
# This script: prepare tasks -> run_eval (N tasks) -> score with dmxapi judge.
#
# Usage:  bash scripts/eval/mcp_atlas/run_mcp_atlas.sh [NUM_TASKS]
#         EF_PROMPT=1 bash scripts/eval/mcp_atlas/run_mcp_atlas.sh 1   # EnvFactory system prompt
# Env:    MODEL (bare served name), RUN_DIR, HARNESS_URL, ONLY_DEFAULT (1/0), TEMPERATURE
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/../common/envfactory_config.py"
ATLAS_DIR="/data1/lvnuoyan/dataset/agent/mcp-atlas"

NUM_TASKS="${1:-2}"
MODEL="${MODEL:-envfactory-eval}"
RUN_DIR="${RUN_DIR:-/data1/lvnuoyan/eval_runs/mcp_atlas/smoke}"
HARNESS_URL="${HARNESS_URL:-http://localhost:3001}"
ONLY_DEFAULT="${ONLY_DEFAULT:-1}"
# Hyperparameter consistency with the paper (thinking model -> 0.7).
TEMPERATURE="${TEMPERATURE:-$(python "$CONFIG" --print temperature)}"

mkdir -p "$RUN_DIR"
TASKS_CSV="$RUN_DIR/tasks.csv"
OUT_CSV="$RUN_DIR/outputs.csv"

# 0) pre-flight: services reachable
curl -sf http://localhost:1984/enabled-servers >/dev/null || {
    echo "[ERROR] sandbox not up on :1984 (run 'make run-docker' in $ATLAS_DIR)"; exit 1; }

# 1) build the local tasks CSV (subset)
PREP_ARGS=(--out "$TASKS_CSV" --num "$NUM_TASKS")
if [ "${EF_SUBSET:-0}" = "1" ]; then
    # EnvFactory paper Appendix F subset: 291 tasks, 30 servers. USER_EXCLUDE_SERVERS
    # (default lara-translate,google-maps: wallet-gated, no key) are dropped too.
    USER_EXCLUDE_SERVERS="${USER_EXCLUDE_SERVERS-lara-translate,google-maps}"
    PREP_ARGS+=(--envfactory-subset)
    [ -n "$USER_EXCLUDE_SERVERS" ] && PREP_ARGS+=(--exclude-servers "$USER_EXCLUDE_SERVERS")
    echo "[run_mcp_atlas] EnvFactory paper subset ON (291 tasks / 30 servers). Start the sandbox with MCP_SERVERS_MODE=envfactory bash setup_env.sh so ENABLED_SERVERS matches."
elif [ "$ONLY_DEFAULT" = "1" ]; then
    PREP_ARGS+=(--only-default-servers)
fi
python "$HERE/prepare_tasks.py" "${PREP_ARGS[@]}"

# 2) run the agent loop (through the TS harness + docker sandbox)
#    - temperature forwarded verbatim into the completion request (paper-consistent)
#    - EF_PROMPT=1 prepends EnvFactory's training system prompt to every task
RUN_ARGS=(--model "$MODEL" --input "$TASKS_CSV" --output "$OUT_CSV" --num-tasks "$NUM_TASKS"
          --concurrency 2 --extra-llm-params "{\"temperature\": $TEMPERATURE}")
if [ "${EF_PROMPT:-1}" = "1" ]; then
    RUN_ARGS+=(--system-prompt "$(python "$CONFIG" --print system_prompt)")
    echo "[run_mcp_atlas] EnvFactory system prompt: ON"
else
    echo "[run_mcp_atlas] EnvFactory system prompt: OFF (stock harness)"
fi
echo "[run_mcp_atlas] model=$MODEL temp=$TEMPERATURE num_tasks=$NUM_TASKS"
cd "$ATLAS_DIR"
HARNESS_URL="$HARNESS_URL" python run_eval.py "${RUN_ARGS[@]}"

# 3) score with the dmxapi DeepSeek-V3.2 judge (EVAL_LLM_* from mcp-atlas/.env)
set -a; . "$ATLAS_DIR/.env"; set +a
python services/scoring/score_claims.py \
    --groundtruth-file "$TASKS_CSV" \
    --model-file "$OUT_CSV" \
    --model-name "$MODEL" \
    --output-dir "$RUN_DIR/score"

echo "[run_mcp_atlas] done. Outputs + scores under: $RUN_DIR"
