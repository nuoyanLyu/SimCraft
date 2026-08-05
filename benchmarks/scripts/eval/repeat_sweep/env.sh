#!/usr/bin/env bash
# Shared config for the fair-comparison REPEAT sweep.
#
# 5 models x 3 seeds, each run through BFCL (single+multi) + tau2 (airline/
# retail/telecom) + MCP-Atlas (Pass/Cov). base runs solo on GPU3 at 0.85 util;
# after base finishes we co-locate TWO workers on the same GPU3 (0.45 util each,
# distinct ports/sandboxes) so two models run in parallel to double throughput.
# Every model/seed uses the SAME eval config (EnvFactory training system prompt,
# temperature 0.7) so the base-vs-RL-vs-Static comparison is apples-to-apples.
# The only thing that varies across the 3 repeats is the vLLM sampling --seed.
#
# Sourced by every script here. Edit values here only.
EF="/home/lvnuoyan/EnvFactory"
SWEEP="/data1/lvnuoyan/eval_runs/repeat_sweep"
ATLAS_DIR="/data1/lvnuoyan/dataset/agent/mcp-atlas"
ATLAS_TASKS="/data1/lvnuoyan/eval_runs/mcp_atlas_tasks.csv"
LOGDIR="$SWEEP/_logs"
QUEUE="$SWEEP/_queue"

SEEDS=(0 1 2)

# Model name -> checkpoint path. Names are used as output-dir / atlas-model names.
declare -A MODELS=(
  [base]="/data1/model/Qwen3-4B"
  [rl-step100]="/data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf/step100"
  [rl-step110]="/data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf/step110"
  [static-step100]="/data1/lvnuoyan/llm_model/factory/EnvFactory-Static-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260714-1800/hf/step100"
  [static-step110]="/data1/lvnuoyan/llm_model/factory/EnvFactory-Static-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260714-1800/hf/step110"
)
# Order pairs the two step100s and the two step110s adjacently so that, with two
# parallel workers, rl-step100 runs alongside static-step100 (and the two 110s
# together) — the fair-comparison pairs finish at roughly the same time.
MODEL_ORDER=(base rl-step100 static-step100 rl-step110 static-step110)

# Per-card (worker) resources. Index 0 = card A, 1 = card B.
# Single-card run: BOTH workers live on GPU3 (GPU5 was lost to another user),
# each vLLM capped at 0.45 util so two 4B models co-exist on one 80GB card.
CARD_GPU=(3 3)
CARD_VLLM_PORT=(8301 8302)
CARD_SANDBOX_PORT=(1984 1985)
CARD_HARNESS_PORT=(3001 3002)
CARD_SANDBOX_NAME=(rsweep_atlas_a rsweep_atlas_b)

VLLM_UTIL=0.45
MAX_LEN=32768
ATLAS_CONCURRENCY=4

# Final GPU reservation after everything finishes (hold both cards).
HOLD_A_MODEL=static-step110
HOLD_B_MODEL=rl-step110
HOLD_A_PORT=8401
HOLD_B_PORT=8402
HOLD_UTIL=0.90

conda_on(){ source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; }

path_of(){ echo "${MODELS[$1]}"; }
