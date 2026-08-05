#!/usr/bin/env bash
# MCP-Atlas for all Static-sweep checkpoints, SEQUENTIALLY through one shared
# sandbox (docker :1984, envfactory 28-server mode) and one harness (:3001) that
# is re-pointed at each model's vLLM port between runs. Scores each with the
# dmxapi DeepSeek-V3.2 claim judge.
#
# Prereq: the 6 vLLM serves (one per checkpoint) are already up (orchestrate.sh).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on
mkdir -p "$LOGDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [atlas] $*"; }

# --- 1) sandbox up (envfactory server set) --------------------------------
setup_first_port="$(port_of ${STEPS[0]})"
LLM_BASE_URL="http://localhost:${setup_first_port}" MCP_SERVERS_MODE=envfactory \
    bash "$EF/scripts/eval/mcp_atlas/setup_env.sh" >/dev/null 2>&1 || true

if curl -sf "http://localhost:${ATLAS_SANDBOX_PORT}/enabled-servers" >/dev/null 2>&1; then
    log "sandbox already UP on :${ATLAS_SANDBOX_PORT}"
else
    log "starting sandbox (docker, envfactory mode)"
    docker rm -f static_atlas_sandbox >/dev/null 2>&1 || true
    ( cd "$ATLAS_DIR" && docker run -d --rm --name static_atlas_sandbox \
        -p ${ATLAS_SANDBOX_PORT}:1984 --env-file .env agent-environment:latest ) \
        >"$LOGDIR/atlas_sandbox.log" 2>&1
    for i in $(seq 1 60); do
        curl -sf "http://localhost:${ATLAS_SANDBOX_PORT}/enabled-servers" >/dev/null 2>&1 && break
        sleep 5
    done
    curl -sf "http://localhost:${ATLAS_SANDBOX_PORT}/enabled-servers" >/dev/null 2>&1 \
        && log "sandbox UP" || { log "ERROR sandbox failed to start; see $LOGDIR/atlas_sandbox.log"; }
fi
log "enabled servers: $(curl -sf http://localhost:${ATLAS_SANDBOX_PORT}/enabled-servers 2>/dev/null | tr -d '\n' | cut -c1-300)"

start_harness(){ # port -> point harness at this model, restart it, wait ready
    local port="$1"
    LLM_BASE_URL="http://localhost:${port}" MCP_SERVERS_MODE=envfactory \
        bash "$EF/scripts/eval/mcp_atlas/setup_env.sh" >/dev/null 2>&1 || true
    tmux kill-session -t atlas_harness 2>/dev/null || true
    sleep 2
    tmux new-session -d -s atlas_harness \
        "source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; cd $ATLAS_DIR && make run-harness 2>&1 | tee $LOGDIR/atlas_harness.log"
    for i in $(seq 1 60); do
        # harness is ready once the port answers with ANY http status (even 404)
        code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${ATLAS_HARNESS_PORT}/" 2>/dev/null || echo 000)"
        [ "$code" != "000" ] && return 0
        sleep 5
    done
    return 1
}

# --- 2) per-checkpoint: run 291 tasks + score -----------------------------
for STEP in "${STEPS[@]}"; do
    PORT="$(port_of "$STEP")"
    OUT="$SWEEP/$STEP/mcp_atlas"; mkdir -p "$OUT"
    if [ -f "$OUT/score/coverage_stats_${STEP}_all.json" ]; then
        log "$STEP already scored -> skip"; continue
    fi
    log "$STEP : point harness at model port $PORT"
    if ! start_harness "$PORT"; then
        log "ERROR harness not ready for $STEP -> skip"; continue
    fi
    log "$STEP : run_eval 291 tasks (concurrency $ATLAS_CONCURRENCY)"
    ( cd "$ATLAS_DIR" && HARNESS_URL="http://localhost:${ATLAS_HARNESS_PORT}" \
        python run_eval.py --model envfactory-eval --input "$ATLAS_TASKS" \
        --output "$OUT/outputs.csv" --concurrency "$ATLAS_CONCURRENCY" --timeout 1800 ) \
        >"$LOGDIR/atlas_run_${STEP}.log" 2>&1 && log "$STEP run_eval OK" || log "$STEP run_eval FAILED (see log)"

    log "$STEP : scoring with dmxapi judge"
    ( cd "$ATLAS_DIR"; set -a; . "$ATLAS_DIR/.env"; set +a
      python services/scoring/score_claims.py \
        --groundtruth-file "$ATLAS_TASKS" --model-file "$OUT/outputs.csv" \
        --model-name "$STEP" --output-dir "$OUT/score" ) \
        >"$LOGDIR/atlas_score_${STEP}.log" 2>&1 && log "$STEP score OK" || log "$STEP score FAILED (see log)"
done

tmux kill-session -t atlas_harness 2>/dev/null || true
log "ALL ATLAS DONE"
