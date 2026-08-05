#!/usr/bin/env bash
# Wait until both card-workers have finished, then print the mean+/-std table
# and hold both GPUs with a vLLM serve. Started automatically by launch.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [watch] $*" | tee -a "$LOGDIR/watch.log"; }

log "watching for both workers to finish ..."
while true; do
  running=0
  for c in 0 1; do
    tmux has-session -t "rsweep_worker_$c" 2>/dev/null && running=1
  done
  [ "$running" = 0 ] && break
  sleep 120
done

# guard: only proceed if every model is actually done
alldone=1
for m in "${MODEL_ORDER[@]}"; do [ -f "$QUEUE/${m}.done" ] || alldone=0; done
if [ "$alldone" != 1 ]; then
  log "WARNING workers gone but not all models done — NOT holding GPUs. Inspect $LOGDIR."
  python "$HERE/collect.py" | tee "$SWEEP/RESULTS_repeat.md" || true
  exit 1
fi

log "all models done -> collect + hold GPUs"
python "$HERE/collect.py" | tee "$SWEEP/RESULTS_repeat.md"
bash "$HERE/launch.sh" hold
log "DONE. Results: $SWEEP/RESULTS_repeat.md ; GPUs held (tmux rsweep_hold_a/b)."
