#!/usr/bin/env bash
# Run the full no-key benchmark suite for ONE served model, sequentially.
# BFCL first (fast, self-contained), then tau2, then VitaBench. Continues on failure.
#
# Usage: run_model_evals.sh <STEP> <PORT>
#   STEP e.g. step20 ; PORT e.g. 8101 (a vLLM endpoint already serving envfactory-eval)
set -uo pipefail
EF="/home/lvnuoyan/EnvFactory"
MODEL_ROOT="/data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf"
SWEEP="/data1/lvnuoyan/eval_runs/sweep"

STEP="${1:?need STEP}"; PORT="${2:?need PORT}"
MODEL_DIR="${3:-$MODEL_ROOT/$STEP}"
ENDPOINT="http://localhost:$PORT/v1"
OUT="$SWEEP/$STEP"
mkdir -p "$OUT"

source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
log(){ echo "[$(date +%H:%M:%S)] [$STEP] $*"; }

log "=== START (endpoint $ENDPOINT) ==="

# ---- BFCL (priority: fast, no external services) ----
for cat in single_turn multi_turn; do
    log "BFCL $cat"
    BFCL_RUN_ROOT="$OUT/bfcl" MODEL_DIR="$MODEL_DIR" ENDPOINT="$ENDPOINT" \
        EF_PROMPT=1 SKIP_REGISTER=1 THREADS=8 \
        bash "$EF/scripts/eval/bfcl/run_bfcl.sh" "$cat" \
        && log "BFCL $cat OK" || log "BFCL $cat FAILED"
done

# ---- tau2-Bench (dmxapi user-sim + judge; per-run isolated output) ----
for d in mock airline retail telecom banking_knowledge; do
    log "tau2 $d"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/tau2" \
        bash "$EF/scripts/eval/tau2/run_tau2.sh" "$d" \
        && log "tau2 $d OK" || log "tau2 $d FAILED"
done

# ---- VitaBench (dmxapi user + evaluator; unique save-to per model+domain) ----
for d in delivery instore ota cross_domain; do
    log "vita $d"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/vita" \
        SAVE_TO="sweep_${STEP}_${d}" \
        bash "$EF/scripts/eval/vitabench/run_vita.sh" "$d" \
        && log "vita $d OK" || log "vita $d FAILED"
done

log "=== ALL DONE for $STEP ==="
