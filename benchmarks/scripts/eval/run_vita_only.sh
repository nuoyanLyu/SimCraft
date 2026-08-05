#!/usr/bin/env bash
# Run all VitaBench domains for one served model (dmxapi DeepSeek-V3.2 user+evaluator).
# Usage: run_vita_only.sh <STEP> <PORT> [MODEL_DIR]
set -uo pipefail
EF="/home/lvnuoyan/EnvFactory"
SWEEP="/data1/lvnuoyan/eval_runs/sweep"
VSIMS="/data1/lvnuoyan/dataset/agent/vitabench/data/simulations"
STEP="${1:?need STEP}"; PORT="${2:?need PORT}"
ENDPOINT="http://localhost:$PORT/v1"
OUT="$SWEEP/$STEP"; mkdir -p "$OUT"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
log(){ echo "[$(date +%H:%M:%S)] [$STEP] $*"; }
log "=== VITA START ($ENDPOINT) ==="
for d in delivery instore ota cross_domain; do
    rm -rf "$VSIMS/sweep_${STEP}_${d}"
    log "vita $d"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/vita" SAVE_TO="sweep_${STEP}_${d}" \
        bash "$EF/scripts/eval/vitabench/run_vita.sh" "$d" \
        && log "vita $d OK" || log "vita $d FAILED"
done
log "=== VITA DONE for $STEP ==="
