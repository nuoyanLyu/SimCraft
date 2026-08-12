#!/bin/bash
# Serve the Agent (Qwen3-8B).
#
# bf16 weights are ~15.3GiB. Solo, util 0.25 gives a comfortable KV cache. When
# co-hosting with the bf16 Simulator, AGENT_GPU_UTIL=0.20 AGENT_MAX_LEN=8192 is
# the measured ceiling -- 0.25 OOMs during CUDA graph capture with the Simulator
# resident. See the sizing note in serve_simulator.sh.
#
# Usage: bash scripts/serve_agent.sh [extra vllm args...]
set -euo pipefail
ENV_ROOT=/root/autodl-tmp/envs/simcraft
source "$ENV_ROOT/bin/activate"

MODEL=${AGENT_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-8B}
PORT=${AGENT_PORT:-8001}

# Same flashinfer/CUDA-13 requirement as the Simulator; harmless for this dense
# model but keeps both servers on one toolchain.
export CUDA_HOME="${CUDA_HOME:-$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13}"

exec vllm serve "$MODEL" \
    --served-model-name Qwen3-8B \
    --port "$PORT" \
    --max-model-len "${AGENT_MAX_LEN:-32768}" \
    --gpu-memory-utilization "${AGENT_GPU_UTIL:-0.25}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    "$@"
