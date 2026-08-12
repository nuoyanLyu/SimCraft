#!/bin/bash
# The playbook study, end to end, on the fixed pipeline (2026-07-29).
#
# Chained rather than launched in parallel: every GPU stage needs the whole
# card, and overlapping them only makes both slower and the timings
# unattributable. The audit stage talks to the teacher relay instead, so it
# could overlap in principle -- it is kept in line for reproducibility.
#
#   1. bank fill (train 24 / val 12 / eval 48, gc=3)
#   2. screen every split for baseline pass rate  [GPU]
#   3. audit task quality, then write the verdicts onto the bank  [teacher]
#   4. compose the frozen eval set (band + hard + a ceiling sample)
#   5. evolve a playbook off the train split, held-out val rollback armed  [GPU]
#   6. A/B the checkpoints on the frozen eval set  [GPU]
#
# Stage 2 addresses the 2026-07-28 null result, where 11 of 12 tasks were
# pinned. But stage 3 is what makes stage 2 usable: screening alone cannot
# distinguish "the agent cannot do this yet" from "nobody could do this", since
# both read as 0.0 -- and on this bank 0.0 is the biggest group after the
# ceiling. Discarding it wholesale (as a 0.2-0.8 band does) threw away 15 of 47
# eval tasks, 11 of which the audit then found perfectly well-posed.
#
# Resume at a later stage with:  STAGE=5 bash scripts/run_playbook_study.sh
set -uo pipefail
cd /root/SimCraft || exit 1
source /root/autodl-tmp/envs/simcraft/bin/activate
export PYTHONPATH=/root/SimCraft

LOGS=/root/autodl-tmp/serve_logs
BIGRUN=smoke_test_results/bigrun_0729
ABRUN=abtest/run_0729
WORKERS=6
STAGE=${STAGE:-1}

stage() { [ "$STAGE" -le "$1" ]; }

if stage 1; then
  echo "##### [$(date +%H:%M)] stage 1: waiting for the bank fill"
  while ! grep -q "BANK FILL DONE" "$LOGS/fill_bank.log" 2>/dev/null; do sleep 60; done
  python -c "
from qwen_agentworld.teacher.task_bank import TaskBank
import json; print('bank:', json.dumps(TaskBank().stats()))"
fi

if stage 2; then
  for split in eval train val; do
    echo "##### [$(date +%H:%M)] stage 2: screening the $split split"
    python scripts/screen_task_difficulty.py --split "$split" --graph-complexity 3 \
      --reps 5 --workers "$WORKERS" --out-dir "abtest/screening_0729_$split" 2>&1 \
      | grep -v "HTTP Request"
  done
fi

if stage 3; then
  echo "##### [$(date +%H:%M)] stage 3: auditing task quality"
  for f in abtest/zeros_*.json abtest/nonzero_*.json; do
    out="abtest/quality/$(basename "${f%.json}").json"
    [ -f "$out" ] && continue
    python scripts/audit_task_quality.py --tasks "$f" --out "$out"
  done
  python scripts/backfill_audit_verdicts.py abtest/quality/zeros_*.json abtest/quality/nonzero_*.json
fi

if stage 4; then
  echo "##### [$(date +%H:%M)] stage 4: composing the eval set"
  python scripts/select_eval_set.py --out "$ABRUN/eval_tasks.json" \
    --rationale "$ABRUN/eval_set_rationale.json"
  python -c "
from qwen_agentworld.teacher.task_bank import TaskBank
b = TaskBank()
for split in ('train', 'val', 'eval'):
    usable = b.draw('mcp_notes', 3, 999, split=split, band=(0.0, 0.8),
                    require_screened=True, drop_audit_failed=True)
    print(f'{split}: {len(usable)} usable (screened, <=0.8, audit-clean)')"
fi

if stage 5; then
  echo "##### [$(date +%H:%M)] stage 5: evolve run"
  python scripts/live_smoke_real_sim.py \
    --iterations 4 --tasks-per-iteration 4 --graph-complexity 3 \
    --bank-train --screened-train --validation-tasks 8 --validation-reps 2 --screened-val \
    --output-dir "$BIGRUN" 2>&1 | grep -v "HTTP Request"
fi

if stage 6; then
  echo "##### [$(date +%H:%M)] stage 6: A/B on the frozen eval set"
  python scripts/ab_test.py --reps 5 --graph-complexity 3 \
    --reuse-tasks "$ABRUN/eval_tasks.json" --workers "$WORKERS" \
    --bigrun-dir "$BIGRUN" --mid-iteration iteration_2.json --final-iteration iteration_4.json \
    --out-dir "$ABRUN" 2>&1 | grep -v "HTTP Request"
fi

echo "##### [$(date +%H:%M)] STUDY DONE"
