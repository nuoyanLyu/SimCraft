#!/bin/bash
# Bring up both vLLM servers co-hosted on the single RTX PRO 6000, using the
# bf16 settings measured on 2026-07-27 (see serve_simulator.sh header). fp8
# would give more headroom but changes simulator fidelity, and the simulator is
# the environment under study -- not a knob to vary mid-experiment.
set -uo pipefail
cd /root/SimCraft || exit 1
LOG_DIR=/root/autodl-tmp/serve_logs
mkdir -p "$LOG_DIR"

SIMULATOR_GPU_UTIL=0.72 SIMULATOR_MAX_SEQS=64 \
  nohup bash scripts/serve_simulator.sh > "$LOG_DIR/simulator.log" 2>&1 &
echo "simulator pid=$!"

# Agent waits for the simulator to finish claiming its share; starting both at
# once makes the memory profiler race and one of them OOMs.
until curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models | grep -q 200; do
  sleep 20
  echo "[$(date +%H:%M:%S)] waiting for simulator..."
done
echo "simulator READY"

AGENT_GPU_UTIL=0.20 AGENT_MAX_LEN=8192 \
  nohup bash scripts/serve_agent.sh > "$LOG_DIR/agent.log" 2>&1 &
echo "agent pid=$!"
until curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/v1/models | grep -q 200; do
  sleep 20
  echo "[$(date +%H:%M:%S)] waiting for agent..."
done
echo "BOTH SERVERS READY"
nvidia-smi --query-gpu=memory.used --format=csv
