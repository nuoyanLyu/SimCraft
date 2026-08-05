#!/usr/bin/env bash
# Parallel sweep: serve all 6 checkpoints and run the full no-key benchmark suite
# (BFCL + tau2 + VitaBench) on each, in tmux daemons that survive disconnects.
#
# GPU layout: 6 models on 3 cards (2 per card). vLLM gpu_memory_utilization is a
# fraction of TOTAL device memory and counts the co-located model against the budget,
# so the 2nd model on each card gets a higher util. Serves start SEQUENTIALLY (wait
# for each to be ready) to avoid startup memory races.
#   step20 ->gpu1:8101 util0.45   step40 ->gpu1:8102 util0.90
#   step60 ->gpu4:8103 util0.45   step80 ->gpu4:8104 util0.90
#   step100->gpu6:8105 util0.45   step110->gpu6:8106 util0.90
#
# Usage: bash scripts/eval/run_sweep.sh
set -uo pipefail
EF="/home/lvnuoyan/EnvFactory"
MROOT="/data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory

# step:gpu:port:util  (first-on-card 0.45, second-on-card 0.90)
MAP=(
  "step20:1:8101:0.45" "step40:1:8102:0.45"
  "step60:4:8103:0.45" "step80:4:8104:0.45"
  "step100:6:8105:0.45" "step110:6:8106:0.45"
)

echo "[sweep] registering EnvFactory model in BFCL once (avoids concurrent rewrites)"
python "$EF/scripts/eval/bfcl/register_models.py"

wait_ready(){ # port
    local port="$1"
    for i in $(seq 1 60); do
        curl -sf "http://localhost:$port/v1/models" | grep -q envfactory-eval && return 0
        sleep 10
    done
    return 1
}

# ---- Phase 1: serve all 6, SEQUENTIALLY (wait each ready before next) ----
for entry in "${MAP[@]}"; do
    IFS=: read -r step gpu port util <<< "$entry"
    tmux kill-session -t "srv_$step" 2>/dev/null || true
    tmux new-session -d -s "srv_$step" \
        "source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; \
         bash $EF/scripts/eval/common/serve_model.sh $MROOT/$step $gpu $port envfactory-eval $util 32768 2>&1 | tee /tmp/srv_$step.log"
    echo "[sweep] serving $step on gpu$gpu port$port util$util ; waiting for ready..."
    if wait_ready "$port"; then echo "[sweep] $step READY on :$port"; else echo "[sweep] ERROR: $step not ready, see /tmp/srv_$step.log"; fi
done

# ---- Phase 2: launch per-model eval sequences (tmux eval_<step>) ----
for entry in "${MAP[@]}"; do
    IFS=: read -r step gpu port util <<< "$entry"
    curl -sf "http://localhost:$port/v1/models" | grep -q envfactory-eval || { echo "[sweep] skip eval_$step (endpoint down)"; continue; }
    tmux kill-session -t "eval_$step" 2>/dev/null || true
    tmux new-session -d -s "eval_$step" \
        "bash $EF/scripts/eval/run_model_evals.sh $step $port 2>&1 | tee /tmp/eval_$step.log"
    echo "[sweep] launched eval_$step -> /tmp/eval_$step.log"
done

echo "[sweep] all launched. Monitor: tmux ls | grep -E 'srv_|eval_'"
echo "[sweep] results under /data1/lvnuoyan/eval_runs/sweep/<step>/{bfcl,tau2,vita}"
