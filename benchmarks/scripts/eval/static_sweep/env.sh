#!/usr/bin/env bash
# Shared configuration for the Static-model checkpoint sweep (post-training).
# Sourced by every script in this directory. Edit values here only.
EF="/home/lvnuoyan/EnvFactory"
CKPT_ROOT="/data1/lvnuoyan/llm_model/factory/EnvFactory-Static-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260714-1800"
MROOT="$CKPT_ROOT/hf"
SWEEP="/data1/lvnuoyan/eval_runs/static_sweep"
ATLAS_DIR="/data1/lvnuoyan/dataset/agent/mcp-atlas"
ATLAS_TASKS="/data1/lvnuoyan/eval_runs/mcp_atlas_tasks.csv"
LOGDIR="$SWEEP/_logs"

# Checkpoints to evaluate, and the FSDP global_step dirs that must be converted
# to hf first (step20/40/60 are already converted; 80/100/110 are produced/late).
STEPS=("step20" "step40" "step60" "step80" "step100" "step110")
CONVERT_STEPS=("80" "100" "110")

# step:gpu:port:util  -- 6 models on 3 cards (GPU 3,4,5), 2 per card, gentle util.
MAP=(
  "step20:3:8121:0.45"  "step40:3:8122:0.45"
  "step60:4:8123:0.45"  "step80:4:8124:0.45"
  "step100:5:8125:0.45" "step110:5:8126:0.45"
)

# MCP-Atlas single shared stack (sandbox stays up; harness restarted per model).
ATLAS_HARNESS_PORT=3001
ATLAS_SANDBOX_PORT=1984
ATLAS_CONCURRENCY=4

# Final GPU reservation after all evals: serve the newest static ckpt on 2 cards.
RESERVE_STEP="step110"
RESERVE_GPUS="5,6"
RESERVE_PORT=8200
RESERVE_NAME="static-step110"

port_of(){ local s="$1"; for e in "${MAP[@]}"; do IFS=: read -r st g p u <<<"$e"; [ "$st" = "$s" ] && { echo "$p"; return; }; done; }

conda_on(){ source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; }
