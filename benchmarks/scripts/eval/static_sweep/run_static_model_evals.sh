#!/usr/bin/env bash
# Per-model BFCL + tau2 for the Static sweep (NO vita, NO mock/banking, NO atlas).
# Atlas is handled separately (run_atlas_all.sh) because it needs a shared
# sandbox/harness stack and must run sequentially.
#
# Usage: run_static_model_evals.sh <STEP> <PORT>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on

STEP="${1:?need STEP}"; PORT="${2:?need PORT}"
MODEL_DIR="$MROOT/$STEP"
ENDPOINT="http://localhost:$PORT/v1"
OUT="$SWEEP/$STEP"
mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] [$STEP] $*"; }

log "=== START BFCL+tau2 (endpoint $ENDPOINT) ==="

# ---- BFCL (single_turn + multi_turn) ----
for cat in single_turn multi_turn; do
    log "BFCL $cat"
    BFCL_RUN_ROOT="$OUT/bfcl" MODEL_DIR="$MODEL_DIR" ENDPOINT="$ENDPOINT" \
        EF_PROMPT=1 SKIP_REGISTER=1 THREADS=8 \
        bash "$EF/scripts/eval/bfcl/run_bfcl.sh" "$cat" \
        && log "BFCL $cat OK" || log "BFCL $cat FAILED"
done

# ---- tau2 (airline / retail / telecom only) ----
for d in airline retail telecom; do
    log "tau2 $d"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$OUT/tau2" \
        bash "$EF/scripts/eval/tau2/run_tau2.sh" "$d" \
        && log "tau2 $d OK" || log "tau2 $d FAILED"
done

log "=== BFCL+tau2 DONE for $STEP ==="
