#!/usr/bin/env bash
# Correct cross-scenario VitaBench (main metric): --domain "delivery,instore,ota" -> 100 cross tasks.
set -uo pipefail
EF=/home/lvnuoyan/EnvFactory; SWEEP=/data1/lvnuoyan/eval_runs/sweep
VSIMS=/data1/lvnuoyan/dataset/agent/vitabench/data/simulations
STEP="${1:?}"; PORT="${2:?}"; ENDPOINT="http://localhost:$PORT/v1"; OUT="$SWEEP/$STEP"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
log(){ echo "[$(date +%H:%M:%S)] [$STEP] $*"; }
rm -rf "$VSIMS/sweep_${STEP}_cross"
log "=== VITA CROSS START (--domain delivery,instore,ota n=100) ==="
NUM_TASKS=100 AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/vita" SAVE_TO="sweep_${STEP}_cross" \
  bash "$EF/scripts/eval/vitabench/run_vita.sh" "delivery,instore,ota" \
  && log "vita cross OK" || log "vita cross FAILED"
log "=== VITA CROSS DONE for $STEP ==="
