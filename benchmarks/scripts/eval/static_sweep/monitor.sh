#!/usr/bin/env bash
# Watch the running Static training; when it finishes (global_step_110 saved AND
# the trainer process has exited), launch the full eval orchestration in tmux.
# While waiting, opportunistically pre-convert step80/step100 to hf so we still
# have them even if verl were to prune older global_step_* dirs.
# Runs unattended in tmux `static_monitor`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on
mkdir -p "$LOGDIR"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [monitor] $*" | tee -a "$LOGDIR/monitor.log"; }

log "monitor start. Waiting for global_step_110 + trainer exit."
log "ckpt_root=$CKPT_ROOT"

pre_convert(){ # convert a global_step to hf if not already present (CPU-only)
    local S="$1"
    [ -d "$CKPT_ROOT/global_step_$S/actor" ] || return 0
    { [ -d "$MROOT/step$S" ] && [ -n "$(ls -A "$MROOT/step$S" 2>/dev/null)" ]; } && return 0
    log "pre-convert global_step_$S -> hf/step$S (CPU)"
    CUDA_VISIBLE_DEVICES="" bash "$EF/scripts/convert_ckpt.sh" "$S" "$CKPT_ROOT" "$MROOT/step$S" \
        >"$LOGDIR/convert_step$S.log" 2>&1 && log "pre-convert step$S OK" || log "pre-convert step$S FAILED"
}

while true; do
    trainer_up=0; pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1 && trainer_up=1
    have110=0; [ -d "$CKPT_ROOT/global_step_110/actor" ] && have110=1
    latest="$(cat "$CKPT_ROOT/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')"

    # opportunistic pre-conversion of intermediate ckpts while we wait
    [ -d "$CKPT_ROOT/global_step_80/actor" ]  && pre_convert 80
    [ -d "$CKPT_ROOT/global_step_100/actor" ] && pre_convert 100

    if [ "$have110" = 1 ] && [ "$trainer_up" = 0 ]; then
        log "DETECTED training finished (latest=$latest, global_step_110 present, trainer gone)."
        sleep 60  # settle: ensure final checkpoint fully flushed
        pre_convert 110
        break
    fi
    log "waiting... latest_ckpt=$latest trainer_up=$trainer_up have110=$have110"
    sleep 300
done

log "launching orchestration in tmux 'static_orchestrate'"
tmux kill-session -t static_orchestrate 2>/dev/null || true
tmux new-session -d -s static_orchestrate "bash $HERE/orchestrate.sh"
log "orchestration launched. Follow: tail -f $LOGDIR/orchestrate.log"
log "monitor exiting."
