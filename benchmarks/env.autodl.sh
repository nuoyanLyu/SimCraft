#!/usr/bin/env bash
# Machine-local settings for the benchmark harness ported from 79.
#
# Source this before any scripts/eval/**/run_*.sh:
#   source benchmarks/env.autodl.sh
#
# Everything here is an override of a default that was baked for the 79 box
# (/data1 paths, the `factory` conda env, an EnvFactory checkpoint on :8100).
# The harness scripts themselves are left as close to the originals as possible
# so they stay diffable against 79.

# --- where the benchmark repos landed ---
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-/root/autodl-tmp/dataset/agent}"
export TAU2_DIR="${TAU2_DIR:-$BENCH_DATA_ROOT/tau2-bench}"
export BFCL_RUN_ROOT="${BFCL_RUN_ROOT:-/root/autodl-tmp/eval_runs/bfcl}"
export RUN_DIR="${RUN_DIR:-/root/autodl-tmp/eval_runs/tau2}"

# --- model under test -------------------------------------------------------
# On 79 this was an EnvFactory checkpoint served on :8100 as `envfactory-eval`.
# Here it is the Agent that scripts/serve_agent.sh is already serving, so the
# benchmarks hit the exact model the playbook was evolved against.
export MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3-8B}"
export ENDPOINT="${ENDPOINT:-http://localhost:8001/v1}"
export AGENT_ENDPOINT="${AGENT_ENDPOINT:-$ENDPOINT}"
export SERVED_NAME="${SERVED_NAME:-Qwen3-8B}"
export REGISTRY_KEY="${REGISTRY_KEY:-qwen3-8b-agentworld}"

# --- external judge / user simulator ---------------------------------------
# 79 read the dmxapi key from EnvFactory/.env under the name EMBEDDING_API_KEY;
# this repo's .env calls it DMX_API_KEY and also carries DMX_URL.
export EF_ROOT="${EF_ROOT:-/root/SimCraft}"
export DMX_KEY_VAR="${DMX_KEY_VAR:-DMX_API_KEY}"

# --- python -----------------------------------------------------------------
# A SEPARATE venv from simcraft, on purpose: BFCL hard-pins numpy 1.26.4 and
# friends, and simcraft is the env currently running both vLLM servers. The
# harnesses only speak HTTP to the endpoint, so they need no torch/vllm at all
# and nothing is lost by isolating them.
export BENCH_PY="${BENCH_PY:-/root/autodl-tmp/envs/bench/bin/python}"

mkdir -p "$BFCL_RUN_ROOT" "$RUN_DIR"
echo "[env.autodl] model=$SERVED_NAME@$ENDPOINT data=$BENCH_DATA_ROOT py=$BENCH_PY"
