#!/bin/bash
# Re-audit gc=3,4 after broadening the step-wise-predicate trigger in
# checker_synth. Baseline (narrow trigger): gc3 = 6/8 TOO_WEAK with 1/8
# step-wise predicates, gc4 = 4/8 TOO_WEAK with 0/8. Teacher-only, no GPU.
cd /root/SimCraft || exit 1
source /root/autodl-tmp/envs/simcraft/bin/activate
export PYTHONPATH=/root/SimCraft
mkdir -p abtest/quality
for gc in 3 4; do
  echo "##################### graph_complexity=$gc (new prompt)"
  python scripts/audit_task_quality.py --generate 8 --graph-complexity "$gc" \
    --out "abtest/quality/audit_gc${gc}_v2.json"
done
echo "##################### ALL DONE v2"
