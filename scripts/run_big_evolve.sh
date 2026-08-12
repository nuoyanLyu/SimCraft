#!/usr/bin/env bash
# Large-scale evolve run: does the playbook mechanism actually evolve?
#
# Two things have to happen together for the answer to be yes:
#   * the curriculum raises difficulty as the agent improves, and
#   * the agent keeps up -- pass rate does not collapse at the higher rung.
#
# The 2026-07-29 short run could not answer this. It edited the playbook once,
# in iteration 1, and then had nothing to learn from: the curriculum moved to
# gc=4 after iteration 2 and the bank had no gc=4 bucket, so every later batch
# was freshly generated, unscreened, and easy (pass 1.0, 0.875). Live-generated
# tasks are the wrong training signal -- they are easy by default, and a batch
# the agent passes carries no gradient.
#
# So this fills a bank deep enough to feed 20 iterations at both rungs, screens
# it so only tasks inside the discriminative band are served, and only then
# evolves. Note the ceiling: max_graph_complexity is min(4, len(tools)) and the
# notes family has 5 non-destructive tools, so gc=4 is the top rung available.
# The difficulty axis is 3 rungs wide by construction; that limit is part of
# what this run is measuring, not something it can escape.
set -uo pipefail
cd /root/SimCraft || exit 1
source /root/autodl-tmp/envs/simcraft/bin/activate
export PYTHONPATH=/root/SimCraft

OUT=smoke_test_results/bigevolve_0729
STAGE=${STAGE:-1}
stage() { [ "$STAGE" -le "$1" ]; }

if stage 1; then
  echo "##### [$(date +%H:%M)] stage 1: deepening the train bank"
  # val/eval stay as they are: the held-out set must not change mid-study.
  python scripts/fill_task_bank.py --graph-complexity 3 --train 96 --val 0 --eval 0 \
    --workers 8 2>&1 | grep -vE "HTTP Request|Retrying"
  python scripts/fill_task_bank.py --graph-complexity 4 --train 72 --val 0 --eval 0 \
    --workers 8 2>&1 | grep -vE "HTTP Request|Retrying"
  python -c "
from qwen_agentworld.teacher.task_bank import TaskBank
import json; print('bank:', json.dumps(TaskBank().stats(), indent=1))"
fi

if stage 2; then
  for gc in 3 4; do
    echo "##### [$(date +%H:%M)] stage 2: screening train at gc=$gc"
    python scripts/screen_task_difficulty.py --split train --graph-complexity "$gc" \
      --reps 3 --workers 6 --out-dir "abtest/screen_big_gc$gc" 2>&1 | grep -v "HTTP Request"
  done
  python -c "
from qwen_agentworld.teacher.task_bank import TaskBank
b = TaskBank()
for gc in (3, 4):
    n = len(b.draw('mcp_notes', gc, 999, split='train', band=(0.0, 0.8), require_screened=True))
    print(f'gc{gc}: {n} train tasks in band')"
fi

if stage 3; then
  echo "##### [$(date +%H:%M)] stage 3: 20-iteration evolve"
  # Starting at gc=3 so validation can draw from the audited gc3 val pool; the
  # curriculum is then free to move up to 4 (or back down to 2) on its own.
  python scripts/live_smoke_real_sim.py \
    --iterations 20 --tasks-per-iteration 6 --graph-complexity 3 \
    --bank-train --screened-train --validation-tasks 8 --validation-reps 2 --screened-val \
    --output-dir "$OUT" 2>&1 | grep -v "HTTP Request"
fi

echo "##### [$(date +%H:%M)] BIG EVOLVE DONE"
