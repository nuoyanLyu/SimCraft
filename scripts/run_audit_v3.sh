#!/bin/bash
# Chained after the v2 audit so the two never hit the teacher API concurrently.
# v3 = v2's broadened step-wise trigger PLUS the ban on character-length
# assertions, which produced both UNPASSABLE tasks measured so far.
cd /root/SimCraft || exit 1
source /root/autodl-tmp/envs/simcraft/bin/activate
export PYTHONPATH=/root/SimCraft

while ! grep -q "ALL DONE v2" abtest/quality/audit_v2.log 2>/dev/null; do sleep 60; done
echo "##################### v2 finished, starting v3 (gc=3, no char-count assertions)"
python scripts/audit_task_quality.py --generate 8 --graph-complexity 3 \
  --out "abtest/quality/audit_gc3_v3.json"
echo "##################### ALL DONE v3"
