#!/usr/bin/env bash
# One-shot transition: keep base running SOLO on GPU3 at 0.85 util, and the moment
# base's 3 seeds are all done, tear down the single 0.85 worker and relaunch TWO
# workers co-located on GPU3 (0.45 util each, distinct ports/sandboxes) so the
# remaining models run two-at-a-time. Ordering in env.sh pairs rl-step100 with
# static-step100 (and the two 110s) so each fair-comparison pair runs in parallel.
#
# Launched once in tmux (rsweep_upgrade); it polls, does the swap once, then exits.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [upgrade] $*" | tee -a "$LOGDIR/upgrade.log"; }

log "waiting for base to finish all 3 seeds (queue/base.done) before going parallel ..."
while [ ! -f "$QUEUE/base.done" ]; do sleep 60; done
log "base DONE — switching GPU3 to two parallel 0.45-util workers"

# 1) stop the current single worker + watcher so they don't fight the swap.
for s in rsweep_watch rsweep_worker_0 rsweep_worker_1; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

# 2) kill any vLLM still holding GPU3 on our two ports; wait for memory to free
#    so the two fresh 0.45 servers (~37 GiB each) have room to start.
pkill -9 -f "vllm serve .* --port ${CARD_VLLM_PORT[0]} " 2>/dev/null || true
pkill -9 -f "vllm serve .* --port ${CARD_VLLM_PORT[1]} " 2>/dev/null || true
for i in $(seq 1 60); do
  used="$(nvidia-smi -i "${CARD_GPU[0]}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  log "GPU${CARD_GPU[0]} used=${used}MiB (waiting for < 15000 before relaunch)"
  [ -n "$used" ] && [ "$used" -lt 15000 ] 2>/dev/null && break
  sleep 10
done

# 3) clear orphaned per-model queue locks. base is gated by base.done (won't be
#    re-run), but any model the killed worker had claimed mid-flight left a stale
#    *.lock that would block the fresh workers from claiming it. Seed-level .done
#    markers still make the reruns idempotent (only the interrupted seed reruns).
rm -rf "$QUEUE"/*.lock 2>/dev/null || true
log "cleared stale queue locks; remaining .done: $(find "$SWEEP" -name .done | wc -l) seed(s)"

# 4) relaunch: launch.sh kills+starts both workers (now GPU3/GPU3 @0.45 from the
#    edited env.sh) and the watcher. Done models are skipped via their .done files.
log "relaunching two parallel workers via launch.sh"
bash "$HERE/launch.sh"
log "DONE — two workers now co-located on GPU${CARD_GPU[0]} (rl/static step100 in parallel, then the 110s)."
