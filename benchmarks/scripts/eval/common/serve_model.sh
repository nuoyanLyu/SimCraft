#!/usr/bin/env bash
# Serve a HuggingFace model via vLLM as an OpenAI-compatible endpoint.
# The benchmark harnesses call this endpoint through the OpenAI interface.
#
# Usage:
#   serve_model.sh MODEL_PATH [GPU] [PORT] [SERVED_NAME] [MEM_UTIL] [MAX_LEN]
#
# Defaults are tuned to be gentle on a shared GPU (low memory utilization).
# Run this inside the `factory` conda env (which has vllm==0.11).
set -euo pipefail

MODEL_PATH="${1:?need MODEL_PATH}"
GPU="${2:-0}"
PORT="${3:-8100}"
SERVED_NAME="${4:-envfactory-eval}"
MEM_UTIL="${5:-0.35}"
MAX_LEN="${6:-32768}"
TP="${VLLM_TP:-1}"   # paper uses tensor-parallel 2; default 1 to stay on one GPU

echo "[serve] model=$MODEL_PATH gpu=$GPU port=$PORT served_name=$SERVED_NAME mem_util=$MEM_UTIL max_len=$MAX_LEN tp=$TP"

exec env CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len "$MAX_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-log-requests \
    ${EXTRA_VLLM_ARGS:-}
