#!/usr/bin/env bash
# Post-training orchestration for the Static-repair 4B checkpoint sweep.
# Steps: convert missing ckpts -> serve 6 models on 3 cards -> BFCL+tau2 (parallel)
#        -> MCP-Atlas (sequential) -> collect table -> teardown -> reserve 2 cards.
# Designed to run unattended in tmux `static_orchestrate`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on
mkdir -p "$LOGDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch] $*" | tee -a "$LOGDIR/orchestrate.log"; }

log "================ STATIC SWEEP ORCHESTRATION START ================"
log "ckpt_root=$CKPT_ROOT"

# ---- 1) convert FSDP -> hf for the late checkpoints (skip if present) ----
for S in "${CONVERT_STEPS[@]}"; do
    if [ -d "$MROOT/step$S" ] && [ -n "$(ls -A "$MROOT/step$S" 2>/dev/null)" ]; then
        log "convert: step$S already present -> skip"; continue
    fi
    if [ ! -d "$CKPT_ROOT/global_step_$S/actor" ]; then
        log "convert: global_step_$S/actor MISSING -> skip (checkpoint not saved?)"; continue
    fi
    log "convert: merging global_step_$S -> hf/step$S"
    CUDA_VISIBLE_DEVICES="" bash "$EF/scripts/convert_ckpt.sh" "$S" "$CKPT_ROOT" "$MROOT/step$S" \
        >"$LOGDIR/convert_step$S.log" 2>&1 && log "convert step$S OK" || log "convert step$S FAILED (see log)"
done

# ---- 2) serve all present checkpoints (sequential start, wait ready) -----
wait_ready(){ local port="$1"; for i in $(seq 1 90); do
    curl -sf "http://localhost:$port/v1/models" 2>/dev/null | grep -q envfactory-eval && return 0; sleep 10; done; return 1; }

SERVED=()
for entry in "${MAP[@]}"; do
    IFS=: read -r step gpu port util <<<"$entry"
    [ -d "$MROOT/$step" ] || { log "serve: $MROOT/$step missing -> skip $step"; continue; }
    if curl -sf "http://localhost:$port/v1/models" 2>/dev/null | grep -q envfactory-eval; then
        log "serve: $step already up on :$port"; SERVED+=("$step"); continue
    fi
    tmux kill-session -t "srv_static_$step" 2>/dev/null || true
    tmux new-session -d -s "srv_static_$step" \
        "source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; \
         bash $EF/scripts/eval/common/serve_model.sh $MROOT/$step $gpu $port envfactory-eval $util 32768 2>&1 | tee $LOGDIR/srv_$step.log"
    log "serve: $step on gpu$gpu :$port util$util ; waiting..."
    if wait_ready "$port"; then log "serve: $step READY"; SERVED+=("$step"); else log "serve: $step NOT READY (see $LOGDIR/srv_$step.log)"; fi
done
log "served: ${SERVED[*]:-none}"

# ---- 3) register BFCL model once ----------------------------------------
python "$EF/scripts/eval/bfcl/register_models.py" >"$LOGDIR/bfcl_register.log" 2>&1 || log "bfcl register warn"

# ---- 4) BFCL + tau2 per model, in parallel ------------------------------
for step in "${SERVED[@]}"; do
    port="$(port_of "$step")"
    tmux kill-session -t "eval_static_$step" 2>/dev/null || true
    tmux new-session -d -s "eval_static_$step" \
        "bash $HERE/run_static_model_evals.sh $step $port 2>&1 | tee $LOGDIR/eval_$step.log"
    log "launched BFCL+tau2 eval_static_$step -> $LOGDIR/eval_$step.log"
done

log "waiting for all BFCL+tau2 to finish..."
while true; do
    done=1
    for step in "${SERVED[@]}"; do
        grep -qa "BFCL+tau2 DONE for $step" "$LOGDIR/eval_$step.log" 2>/dev/null || done=0
    done
    [ "$done" = 1 ] && break
    sleep 120
done
log "BFCL+tau2 ALL DONE"
python "$HERE/collect_static.py" > "$SWEEP/RESULTS_static.md" 2>&1 || true
log "interim table -> $SWEEP/RESULTS_static.md"

# ---- 5) MCP-Atlas, sequential through shared sandbox+harness ------------
log "starting MCP-Atlas (sequential)..."
bash "$HERE/run_atlas_all.sh" 2>&1 | tee -a "$LOGDIR/atlas_all.log"
log "MCP-Atlas done"

# ---- 6) final table ------------------------------------------------------
python "$HERE/collect_static.py" > "$SWEEP/RESULTS_static.md" 2>&1 || true
log "FINAL table -> $SWEEP/RESULTS_static.md"
cat "$SWEEP/RESULTS_static.md" | tee -a "$LOGDIR/orchestrate.log"

# ---- 7) teardown eval serves, then reserve 2 cards ----------------------
log "tearing down eval serves + atlas services"
for entry in "${MAP[@]}"; do IFS=: read -r step gpu port util <<<"$entry"; tmux kill-session -t "srv_static_$step" 2>/dev/null || true; done
tmux kill-session -t atlas_harness 2>/dev/null || true
docker rm -f static_atlas_sandbox >/dev/null 2>&1 || true
sleep 15  # let GPU memory free

log "reserving GPUs $RESERVE_GPUS with $RESERVE_STEP (TP=2) on :$RESERVE_PORT"
tmux kill-session -t reserve_serve 2>/dev/null || true
tmux new-session -d -s reserve_serve \
    "source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; \
     VLLM_TP=2 bash $EF/scripts/eval/common/serve_model.sh $MROOT/$RESERVE_STEP $RESERVE_GPUS $RESERVE_PORT $RESERVE_NAME 0.85 32768 2>&1 | tee $LOGDIR/reserve_serve.log"
log "reserve serve launched in tmux 'reserve_serve' (model=$RESERVE_NAME port=$RESERVE_PORT gpus=$RESERVE_GPUS)"
log "================ ORCHESTRATION COMPLETE ================"
