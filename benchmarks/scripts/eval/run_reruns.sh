#!/usr/bin/env bash
# Rerun the benchmarks that failed in the sweep (dmxapi quota + vita tool_calls bug),
# now that balance is restored and format_messages is patched.
#   tau2 : retail, telecom, banking_knowledge  (mock/airline already OK from the sweep)
#   vita : delivery, instore, ota, cross_domain (all)
#
# Usage: run_reruns.sh <STEP> <PORT> [MODEL_DIR]
set -uo pipefail
EF="/home/lvnuoyan/EnvFactory"
MROOT="/data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf"
SWEEP="/data1/lvnuoyan/eval_runs/sweep"
VSIMS="/data1/lvnuoyan/dataset/agent/vitabench/data/simulations"

STEP="${1:?need STEP}"; PORT="${2:?need PORT}"
MODEL_DIR="${3:-$MROOT/$STEP}"
ENDPOINT="http://localhost:$PORT/v1"
OUT="$SWEEP/$STEP"
mkdir -p "$OUT"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
log(){ echo "[$(date +%H:%M:%S)] [$STEP] $*"; }

log "=== RERUN START (endpoint $ENDPOINT) ==="

# tau2 (fresh: overwrite the prior None/failed domains)
for d in retail telecom banking_knowledge; do
    rm -rf "$OUT/tau2/${d}_sim.json"
    log "tau2 $d"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/tau2" \
        bash "$EF/scripts/eval/tau2/run_tau2.sh" "$d" \
        && log "tau2 $d OK" || log "tau2 $d FAILED"
done

# vita (clear the prior empty/failed sim dirs first, then run all domains)
for d in delivery instore ota cross_domain; do
    rm -rf "$VSIMS/sweep_${STEP}_${d}"
    log "vita $d"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/vita" \
        SAVE_TO="sweep_${STEP}_${d}" \
        bash "$EF/scripts/eval/vitabench/run_vita.sh" "$d" \
        && log "vita $d OK" || log "vita $d FAILED"
done

log "=== RERUN DONE for $STEP ==="
