#!/usr/bin/env bash
# Waits for GPU index 1 to free up (currently running the user's own test:
# 3x VLLM::EngineCore processes, ~77.7GB/81.9GB, 100% util as of the check
# that preceded this script), then serves Qwen3-14B on it and runs the live
# smoke test, then tears the server back down. Meant to run inside tmux so a
# dropped SSH connection doesn't kill it.
set -uo pipefail

REPO=~/Qwen-AgentWorld
LOG_DIR="$REPO/scripts/gpu1_run_logs"
mkdir -p "$LOG_DIR"
MARKER="$LOG_DIR/DONE"
rm -f "$MARKER"

POLL_INTERVAL=180          # seconds between nvidia-smi checks
FREE_THRESHOLD_MIB=5000    # below this on GPU1 counts as "free"
PORT=8501
MODEL_NAME=Qwen3-14B
MODEL_PATH=/data1/model/Qwen3-14B
VLLM_LOG="$LOG_DIR/vllm_serve.log"
SMOKE_LOG="$LOG_DIR/smoke_test.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/monitor.log"; }

log "monitor started, watching GPU index 1 (threshold ${FREE_THRESHOLD_MIB} MiB, poll every ${POLL_INTERVAL}s)"

while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    log "GPU1 memory.used=${USED} MiB"
    if [ "$USED" -lt "$FREE_THRESHOLD_MIB" ]; then
        log "GPU1 is free, proceeding"
        break
    fi
    sleep "$POLL_INTERVAL"
done

log "launching vllm serve for $MODEL_NAME on GPU1, port $PORT"
cd "$REPO"
CUDA_VISIBLE_DEVICES=1 nohup ~/anaconda3/envs/factory/bin/vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --disable-log-requests \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
log "vllm serve pid=$VLLM_PID, waiting for health check on port $PORT"

READY=0
for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
        READY=1
        log "vllm server is healthy after ${i} checks"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vllm serve process died before becoming healthy, see $VLLM_LOG"
        break
    fi
    sleep 15
done

if [ "$READY" -ne 1 ]; then
    log "vllm server never became healthy, aborting smoke test, killing pid=$VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null
    echo "FAILED: vllm server did not become healthy" > "$MARKER"
    exit 1
fi

log "running smoke test"
export AGENT_URL="http://localhost:${PORT}/v1"
export AGENT_API_KEY="EMPTY"
~/anaconda3/envs/simcraft/bin/python scripts/live_smoke_test.py \
    --iterations 2 \
    --tasks-per-iteration 2 \
    --agent-model "$MODEL_NAME" \
    --output-dir "$LOG_DIR/smoke_test_results" \
    > "$SMOKE_LOG" 2>&1
SMOKE_STATUS=$?
log "smoke test exited with status $SMOKE_STATUS"

log "tearing down vllm serve pid=$VLLM_PID"
kill "$VLLM_PID" 2>/dev/null
sleep 5
kill -9 "$VLLM_PID" 2>/dev/null

if [ "$SMOKE_STATUS" -eq 0 ]; then
    echo "OK: smoke test completed, results in $LOG_DIR/smoke_test_results" > "$MARKER"
else
    echo "FAILED: smoke test exited with status $SMOKE_STATUS, see $SMOKE_LOG" > "$MARKER"
fi
log "done, marker written to $MARKER"
