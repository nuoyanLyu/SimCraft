#!/bin/bash
# Final verification at the hardest setting, with every checker fix in place:
# broadened step-wise trigger + no character-count assertions + null-predicate
# retry + canonical-value allowlist. gc=4 is where the baseline was worst
# (3/8 clean, 0/8 step-wise) and where 2 of 8 tasks were dropped in v2.
cd /root/SimCraft || exit 1
source /root/autodl-tmp/envs/simcraft/bin/activate
export PYTHONPATH=/root/SimCraft

while ! grep -q "ALL DONE v3" abtest/quality/audit_v3.log 2>/dev/null; do sleep 60; done
echo "##################### v3 finished, starting v4 (gc=4, all fixes)"
python scripts/audit_task_quality.py --generate 8 --graph-complexity 4 \
  --out "abtest/quality/audit_gc4_v4.json"
echo "##################### ALL DONE v4"
