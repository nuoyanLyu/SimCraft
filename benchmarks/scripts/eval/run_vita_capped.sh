#!/usr/bin/env bash
# Capped VitaBench: cross_domain full (100, the headline metric),
# single-scenario domains (delivery/instore/ota) capped to 30 each.
set -uo pipefail
EF=/home/lvnuoyan/EnvFactory
SWEEP=/data1/lvnuoyan/eval_runs/sweep
VSIMS=/data1/lvnuoyan/dataset/agent/vitabench/data/simulations
STEP="${1:?}"; PORT="${2:?}"
ENDPOINT="http://localhost:$PORT/v1"
OUT="$SWEEP/$STEP"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
log(){ echo "[$(date +%H:%M:%S)] [$STEP] $*"; }
log "=== VITA(capped) START ==="
for d in delivery:30 instore:30 ota:30 cross_domain:100; do
  IFS=: read dom n <<< "$d"
  rm -rf "$VSIMS/sweep_${STEP}_${dom}"
  log "vita $dom (n=$n)"
  NUM_TASKS="$n" AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/vita" SAVE_TO="sweep_${STEP}_${dom}" \
    bash "$EF/scripts/eval/vitabench/run_vita.sh" "$dom" \
    && log "vita $dom OK" || log "vita $dom FAILED"
done
log "=== VITA DONE for $STEP ==="
