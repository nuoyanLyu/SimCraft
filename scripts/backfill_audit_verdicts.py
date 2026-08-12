"""Copy `audit_task_quality` verdicts from its report files onto the bank entries.

Run after any audit. Idempotent, and later reports win over earlier ones so a
re-audit of the same task corrects the record.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/SimCraft")

from qwen_agentworld.teacher.task_bank import TaskBank

ap = argparse.ArgumentParser()
ap.add_argument("reports", nargs="+", help="abtest/quality/*.json audit reports")
args = ap.parse_args()

bank = TaskBank()
applied = missing = 0
tally = {"clean": 0, "unpassable": 0, "too_weak": 0}
for report in args.reports:
    rows = json.loads(Path(report).read_text()).get("rows", [])
    for r in rows:
        unpassable, too_weak = bool(r.get("unpassable")), bool(r.get("too_weak"))
        if bank.set_audit_verdict(r["task_id"], unpassable=unpassable, too_weak=too_weak):
            applied += 1
            tally["unpassable" if unpassable else "too_weak" if too_weak else "clean"] += 1
        else:
            missing += 1
    print(f"{report}: {len(rows)} rows")

print(f"applied {applied} verdicts ({tally}), {missing} task_ids not in bank")
