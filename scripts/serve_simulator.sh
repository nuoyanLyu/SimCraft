#!/bin/bash
# Serve the Simulator (Qwen-AgentWorld-35B-A3B).
#
# Sizing for the single RTX PRO 6000 (96GB -> 94.97GiB usable, Blackwell/sm_120).
# Both configs below were measured on 2026-07-27, not estimated.
#
#   bf16 solo      : (defaults)                          88.9GiB, KV 663k tokens
#   bf16 co-hosted : SIMULATOR_GPU_UTIL=0.72             69.5GiB, KV  76k tokens
#                    SIMULATOR_MAX_SEQS=64
#                    + AGENT_GPU_UTIL=0.20 AGENT_MAX_LEN=8192
#                                                        93.0GiB of 95.6 total
#
# Co-hosting bf16 does fit, with ~2.6GiB to spare -- but the Agent OOMs during
# CUDA graph capture at AGENT_GPU_UTIL=0.25, so 0.20 is the ceiling, and both
# KV caches end up small (9.3x / 2.3x concurrency at 8192 tokens). For real
# headroom use SIMULATOR_QUANT=fp8 (native on Blackwell, ~35GiB) instead.
#
# Note that SIMULATOR_MAX_SEQS must come down with the util: the Mamba cache
# blocks scale with whatever memory is left after the weights, and vLLM refuses
# to capture CUDA graphs if max_num_seqs exceeds them (809 blocks at util 0.90,
# only 93 at 0.72).
#
# Usage: bash scripts/serve_simulator.sh [extra vllm args...]
set -euo pipefail
ENV_ROOT=/root/autodl-tmp/envs/simcraft
source "$ENV_ROOT/bin/activate"

MODEL=${SIMULATOR_MODEL_PATH:-/root/autodl-tmp/models/Qwen-AgentWorld-35B-A3B}
PORT=${SIMULATOR_PORT:-8000}
QUANT=${SIMULATOR_QUANT:-}

# The MoE path JIT-compiles a cutlass sm_120 kernel through flashinfer, which
# needs CUDA >= 12.9. The system toolkit is 12.4, so point flashinfer at the
# CUDA 13 toolkit that ships inside the venv -- otherwise the engine dies with
# "No supported CUDA architectures found for major versions [12]".
export CUDA_HOME="${CUDA_HOME:-$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13}"

ARGS=(
    --served-model-name Qwen-AgentWorld-35B-A3B
    --port "$PORT"
    --max-model-len "${SIMULATOR_MAX_LEN:-8192}"
    # Hybrid model: every decode sequence also needs a Mamba cache block, and at
    # util 0.90 only 809 of those fit. vLLM's default max_num_seqs of 1024
    # exceeds that and aborts CUDA graph capture. 512 matches vLLM's largest
    # cudagraph capture size and is far more concurrency than we ever use.
    --max-num-seqs "${SIMULATOR_MAX_SEQS:-512}"
    --gpu-memory-utilization "${SIMULATOR_GPU_UTIL:-0.90}"
    --reasoning-parser qwen3
    --language-model-only   # checkpoint is language-only despite the VL arch
    --trust-remote-code
)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")

exec vllm serve "$MODEL" "${ARGS[@]}" "$@"
