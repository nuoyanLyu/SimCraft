#!/usr/bin/env bash
# Launch the full fair-comparison repeat sweep: register BFCL once, prepare the
# shared MCP-Atlas .env (envfactory 28-server set + dmxapi judge), then start the
# two card-workers in their own tmux sessions. When both finish, print the
# mean+/-std table and hold both GPUs with a vLLM serve.
#
# Usage:
#   bash launch.sh            # start workers (idempotent; skips done model/seeds)
#   bash launch.sh collect    # just (re)print the results table
#   bash launch.sh hold       # just (re)start the final GPU-holding serves
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on
mkdir -p "$LOGDIR" "$QUEUE"

do_collect(){ python "$HERE/collect.py" | tee "$SWEEP/RESULTS_repeat.md"; echo "[launch] table -> $SWEEP/RESULTS_repeat.md"; }

do_hold(){
  # Single-card run: GPU5 was lost to another user mid-run, so only GPU3 is ours.
  # Hold ONLY GPU3 — serving on GPU5 would OOM against the other job.
  local pa="$(path_of "$HOLD_A_MODEL")"
  tmux kill-session -t rsweep_hold_a 2>/dev/null || true
  tmux new-session -d -s rsweep_hold_a \
    "source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; \
     bash $EF/scripts/eval/common/serve_model.sh $pa ${CARD_GPU[0]} $HOLD_A_PORT $HOLD_A_MODEL $HOLD_UTIL $MAX_LEN"
  echo "[launch] holding GPU${CARD_GPU[0]} ($HOLD_A_MODEL:$HOLD_A_PORT)"
}

case "${1:-run}" in
  collect) do_collect; exit 0;;
  hold)    do_hold; exit 0;;
esac

# --- one-time setup -------------------------------------------------------
echo "[launch] register BFCL EnvFactory model (once)"
python "$EF/scripts/eval/bfcl/register_models.py"

echo "[launch] write shared MCP-Atlas .env (envfactory servers + dmxapi judge)"
MCP_SERVERS_MODE=envfactory LLM_BASE_URL="http://localhost:${CARD_VLLM_PORT[0]}" \
  bash "$EF/scripts/eval/mcp_atlas/setup_env.sh"

# --- start the two card-workers ------------------------------------------
for c in 0 1; do
  tmux kill-session -t "rsweep_worker_$c" 2>/dev/null || true
  tmux new-session -d -s "rsweep_worker_$c" \
    "bash $HERE/worker.sh $c; echo WORKER_${c}_EXIT=\$? >> $LOGDIR/worker_card${c}.log"
  echo "[launch] started worker $c (tmux rsweep_worker_$c) on GPU${CARD_GPU[$c]}"
done

# auto collect + hold GPUs when both workers finish
tmux kill-session -t rsweep_watch 2>/dev/null || true
tmux new-session -d -s rsweep_watch "bash $HERE/watch.sh"
echo "[launch] started watcher (tmux rsweep_watch) -> auto collect + hold on completion"

echo "[launch] workers running. Monitor with: tmux attach -t rsweep_worker_0"
echo "[launch] progress: bash $HERE/launch.sh collect"
