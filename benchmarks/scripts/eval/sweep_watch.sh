#!/usr/bin/env bash
# Watch the sweep; snapshot the comparison table when BFCL finishes on all models,
# and again when the full suite (BFCL+tau2+vita) finishes. Robust to disconnects.
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
EF="/home/lvnuoyan/EnvFactory"
STEPS="base step20 step40 step60 step80 step100 step110"
bfcl_reported=0

while true; do
    # BFCL complete for a model = its eval log has progressed past BFCL (tau2 started) or ALL DONE
    bfcl_done=1
    for s in $STEPS; do
        grep -qaE "tau2 mock|ALL DONE for $s" /tmp/eval_$s.log 2>/dev/null || bfcl_done=0
    done
    if [ "$bfcl_done" = 1 ] && [ "$bfcl_reported" = 0 ]; then
        python "$EF/scripts/eval/collect_sweep.py" > /tmp/sweep_bfcl_summary.txt 2>&1
        echo "[$(date '+%m-%d %H:%M')] BFCL_ALL_DONE" >> /tmp/sweep_watch.log
        bfcl_reported=1
    fi

    all_done=1
    for s in $STEPS; do
        grep -qa "ALL DONE for $s" /tmp/eval_$s.log 2>/dev/null || all_done=0
    done
    if [ "$all_done" = 1 ]; then
        python "$EF/scripts/eval/collect_sweep.py" > /tmp/sweep_full_summary.txt 2>&1
        echo "[$(date '+%m-%d %H:%M')] ALL_DONE" >> /tmp/sweep_watch.log
        break
    fi
    sleep 120
done
