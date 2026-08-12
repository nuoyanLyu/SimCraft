#!/bin/bash
# Audit generated-task quality across graph complexities. Teacher-only, no GPU.
cd /root/SimCraft || exit 1
source /root/autodl-tmp/envs/simcraft/bin/activate
export PYTHONPATH=/root/SimCraft
mkdir -p abtest/quality
for gc in 2 3 4; do
  echo "##################### graph_complexity=$gc"
  python scripts/audit_task_quality.py --generate 8 --graph-complexity "$gc" \
    --out "abtest/quality/audit_gc${gc}_v2.json"
done
echo "##################### ALL DONE"
