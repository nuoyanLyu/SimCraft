#!/usr/bin/env bash
# Run BFCL against a pre-served EnvFactory model (OpenAI-compatible vLLM endpoint).
# Runs directly in the `factory` conda env.
#
# Prereq: serve the target checkpoint first (also factory env):
#   bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
#   (evaluate the released model by serving IT under the same name envfactory-eval)
#
# Usage:
#   bash scripts/eval/bfcl/run_bfcl.sh [TEST_CATEGORY]
#   SMOKE_N=3 bash scripts/eval/bfcl/run_bfcl.sh simple_python        # tiny smoke
#   EF_PROMPT=1 bash scripts/eval/bfcl/run_bfcl.sh multi_turn          # inject EnvFactory system prompt
#
# Config knobs (all optional):
#   EF_PROMPT=1            inject EnvFactory's training system prompt (default 0 = stock BFCL)
#   EF_SYSTEM_PROMPT_MODE  prepend (default) | replace
#   TEMPERATURE            default from envfactory_config (0.7, thinking models)
#   MODEL_DIR SERVED_NAME ENDPOINT THREADS BFCL_RUN_ROOT SMOKE_N
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/../common/envfactory_config.py"
# The harnesses live in their own venv (see benchmarks/env.autodl.sh); put it
# first on PATH so the `bfcl` console script resolves there too.
BENCH_PY="${BENCH_PY:-python}"
[ "$BENCH_PY" != "python" ] && export PATH="$(dirname "$BENCH_PY"):$PATH"
# The playbook renderer imports qwen_agentworld, which lives in the simcraft
# venv, not the benchmark one -- so it needs its own interpreter.
SIMCRAFT_PY="${SIMCRAFT_PY:-/root/autodl-tmp/envs/simcraft/bin/python}"
REPO_ROOT="${REPO_ROOT:-$(cd "$HERE/../../../.." && pwd)}"

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3-8B}"
ENDPOINT="${ENDPOINT:-http://localhost:8001/v1}"
SERVED_NAME="${SERVED_NAME:-Qwen3-8B}"
REGISTRY_KEY="${REGISTRY_KEY:-qwen3-8b-agentworld}"
CATEGORY="${1:-simple_python}"
THREADS="${THREADS:-8}"
# Hyperparameters: default temperature from the canonical config (paper 0.7 for thinking).
TEMPERATURE="${TEMPERATURE:-$("$BENCH_PY" "$CONFIG" --print temperature)}"
BFCL_RUN_ROOT="${BFCL_RUN_ROOT:-/root/autodl-tmp/eval_runs/bfcl}"

mkdir -p "$BFCL_RUN_ROOT"
export BFCL_PROJECT_ROOT="$BFCL_RUN_ROOT"
export REMOTE_OPENAI_BASE_URL="$ENDPOINT"
export REMOTE_OPENAI_API_KEY="EMPTY"
export REMOTE_OPENAI_TOKENIZER_PATH="$MODEL_DIR"

# System-prompt handling (configurable tool/prompt interface).
if [ "${EF_PROMPT:-1}" = "1" ]; then
    export EF_SYSTEM_PROMPT="$("$BENCH_PY" "$CONFIG" --print system_prompt)"
    export EF_SYSTEM_PROMPT_MODE="${EF_SYSTEM_PROMPT_MODE:-prepend}"
    echo "[run_bfcl] EnvFactory system prompt: ON (mode=$EF_SYSTEM_PROMPT_MODE)"
else
    export EF_SYSTEM_PROMPT=""
    echo "[run_bfcl] EnvFactory system prompt: OFF (stock BFCL)"
fi

# Playbook arm. Appended to (never replacing) the base prompt above, so the two
# A/B arms are byte-identical apart from the playbook text itself. Rendered by
# the same function the training loop uses -- see benchmarks/playbook_prompt.py.
if [ -n "${PLAYBOOK:-}" ]; then
    PLAYBOOK_TEXT="$("$SIMCRAFT_PY" "$REPO_ROOT/benchmarks/playbook_prompt.py" --iteration "$PLAYBOOK" --describe)"
    if [ -z "$PLAYBOOK_TEXT" ]; then
        echo "[ERROR] PLAYBOOK=$PLAYBOOK rendered an empty playbook -- that is the" >&2
        echo "        baseline arm, so running it here would silently duplicate it." >&2
        exit 1
    fi
    export EF_SYSTEM_PROMPT="${EF_SYSTEM_PROMPT}

${PLAYBOOK_TEXT}"
    echo "[run_bfcl] playbook: ON ($PLAYBOOK, ${#PLAYBOOK_TEXT} chars)"
else
    echo "[run_bfcl] playbook: OFF (baseline arm)"
fi

echo "[run_bfcl] category=$CATEGORY endpoint=$ENDPOINT served=$SERVED_NAME temp=$TEMPERATURE smoke=${SMOKE_N:-off}"

# 1) sanity: endpoint reachable and serving the expected model name
if ! curl -sf "$ENDPOINT/models" | grep -q "$SERVED_NAME"; then
    echo "[ERROR] endpoint $ENDPOINT not serving \"$SERVED_NAME\". Start serve_model.sh first." >&2
    exit 1
fi

# 2) register EnvFactory model into BFCL (skip when SKIP_REGISTER=1 for parallel runs)
if [ "${SKIP_REGISTER:-0}" != "1" ]; then "$BENCH_PY" "$HERE/register_models.py"; fi

# 3) generate
GEN_ARGS=(--model "$REGISTRY_KEY" --test-category "$CATEGORY" --skip-server-setup --num-threads "$THREADS" --temperature "$TEMPERATURE")
if [ -n "${SMOKE_N:-}" ]; then
    "$BENCH_PY" - "$CATEGORY" "$SMOKE_N" "$BFCL_RUN_ROOT/test_case_ids_to_generate.json" << "PY"
import json, sys
from bfcl_eval.utils import load_dataset_entry
cat, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ids = [e["id"] for e in load_dataset_entry(cat)][:n]
json.dump({cat: ids}, open(out, "w"))
print("[run_bfcl] smoke ids:", ids)
PY
    GEN_ARGS+=(--run-ids)
fi
bfcl generate "${GEN_ARGS[@]}"

# 4) score
EVAL_ARGS=(--model "$REGISTRY_KEY" --test-category "$CATEGORY")
if [ -n "${SMOKE_N:-}" ]; then EVAL_ARGS+=(--partial-eval); fi
bfcl evaluate "${EVAL_ARGS[@]}"

echo "[run_bfcl] done. Scores under: $BFCL_RUN_ROOT/score"
