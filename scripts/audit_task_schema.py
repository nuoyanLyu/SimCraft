"""Audit a task bank against its families' declared canonical-state schemas.

Two questions, one per half of a task:

  * does `initial_state` conform to the schema (`state_schema.validate_state`)?
  * does the checker only read keys the schema declares
    (`checker_synth.audit_schema_conformance`)?

Both are now enforced at generation time, so a bank generated after
2026-08-05 should report zero. The value of running it anyway is on the older
banks -- it measures what the pre-schema pipeline produced -- and as the
acceptance test for a regenerated one.

This replaces `audit_checker_schema.py`, which had to approximate the second
question because no schema existed: it compared each checker against the
*union* of fields observed anywhere in the bank, so every count it reported
was a lower bound (a checker reading `tags` looked fine even on a task whose
own state had no `tags`). With a declaration to compare against, the counts
are exact.

`--mark-unusable` closes the loop on the checker half. A predicate that reads
an undeclared key can never pass: the key is not in the schema, so neither
`conform_state` nor `complete_fields` will ever produce it, and the predicate
raises KeyError on every rollout forever. That is exactly what
`TaskBank.set_audit_verdict(unpassable=True)` means, and marking it lets the
existing `draw(drop_audit_failed=True)` keep those tasks out of an experiment
instead of letting them contribute guaranteed failures to the pass rate. The
mark is applied only where the predicate is *observed* to raise against a
conformant state, so a guarded read (`n.get('id')`) is left alone.

    python scripts/audit_task_schema.py                          # whole bank
    python scripts/audit_task_schema.py --split train --json report.json
    python scripts/audit_task_schema.py --bank smoke_test_results/newbank
    python scripts/audit_task_schema.py --mark-unusable          # dry run
    python scripts/audit_task_schema.py --mark-unusable --apply  # writes verdicts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agentworld.simulator_gym.env import ACTION_LOG_KEY
from qwen_agentworld.teacher.checker_synth import audit_schema_conformance
from qwen_agentworld.tools.state_schema import get_schema
from scripts.live_smoke_test import NOTES_TOOLS

DEFAULT_BANK = Path(__file__).resolve().parents[1] / "task_bank"

# The toy family's tools live in a script, not the catalog.
FAMILY_TOOLS = {"mcp_notes": NOTES_TOOLS}


def _tools_for(family: str):
    if family in FAMILY_TOOLS:
        return FAMILY_TOOLS[family]
    from qwen_agentworld.tools.families import ALL_FAMILIES

    return ALL_FAMILIES.get(family, [])


def load_bank(bank: Path, split: str | None) -> list[dict]:
    records = []
    for path in sorted(bank.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        task = data.get("task", data)
        if not isinstance(task, dict) or "checker" not in task:
            continue
        meta = data.get("meta") or {}
        if split and meta.get("split") != split:
            continue
        records.append({"path": path, "task": task, "meta": meta})
    return records


def audit_task(task: dict) -> dict:
    schema = get_schema(task.get("tool_family"))
    if schema is None:
        return {"skipped": "no declared schema for this family"}

    tools = _tools_for(task["tool_family"])
    state_problems = schema.validate_state(
        task.get("initial_state") or {}, ignore_keys=frozenset({ACTION_LOG_KEY})
    )

    checker = task.get("checker") or {}
    checker_problems = audit_schema_conformance(
        checker.get("executable_predicate") or "True", schema, tools
    )
    step_wise = checker.get("step_wise_predicate")
    if step_wise:
        checker_problems += audit_schema_conformance(step_wise, schema, tools)

    return {"state_problems": state_problems, "checker_problems": checker_problems}


def predicate_raises_on_conformant_state(task: dict) -> str | None:
    """Return the exception name if this task's checker blows up against a
    fully-populated, schema-conformant state, else None.

    Auditing the AST says the predicate *reads* an undeclared key; running it
    says the read is unguarded and therefore fatal. Only the second justifies
    condemning the task, so this is what `--mark-unusable` acts on.

    The probe is `schema.as_example()` — one object per declared list, every
    declared field present — not the task's own initial state. A predicate that
    subscripts `n['id']` inside a comprehension over an *empty* list never runs
    the subscript, so the task's own state can (and on this bank does) let a
    fatal read pass unnoticed. The example state is the best case the pipeline
    can ever hand the predicate: if it raises there, it raises everywhere.
    """
    from qwen_agentworld.core.schemas import CheckerSpec
    from qwen_agentworld.judge.paired_audit import judge_checker_with_reason

    schema = get_schema(task.get("tool_family"))
    if schema is None:
        return None
    state = schema.as_example()
    checker = CheckerSpec.model_validate(task["checker"])
    _, reason = judge_checker_with_reason(checker, state, states=[state], task_id=task["task_id"])
    return reason.split(":", 1)[1] if reason.startswith("checker_raised:") else None


def mark_unusable(bank: Path, report: list[dict], apply: bool) -> None:
    from qwen_agentworld.teacher.task_bank import TaskBank

    condemned = []
    for item in report:
        if not item["checker_problems"]:
            continue
        exc = predicate_raises_on_conformant_state(item["task"])
        if exc:
            condemned.append((item["task_id"], exc))

    print(f"\ncheckers that provably raise on a conformant state: {len(condemned)}")
    for task_id, exc in condemned:
        print(f"    {task_id}  {exc}")
    if not apply:
        print("  dry run — no verdicts written. Add --apply to mark them unpassable.")
        return

    task_bank = TaskBank(bank)
    written = sum(
        1 for task_id, _ in condemned
        if task_bank.set_audit_verdict(task_id, unpassable=True, too_weak=False)
    )
    print(f"  marked {written}/{len(condemned)} unpassable; draw(drop_audit_failed=True) now skips them")


def _kind(problem: str) -> str:
    """Group problems by kind, dropping the task-specific index and list."""
    return problem.split(" (")[0].split(";")[0].split(":")[0][:90]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--split", help="train | val | eval (default: all)")
    ap.add_argument("--json", help="write the per-task report here")
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--mark-unusable", action="store_true",
                    help="flag checkers that provably raise as unpassable in the bank")
    ap.add_argument("--apply", action="store_true", help="with --mark-unusable, actually write")
    args = ap.parse_args()

    records = load_bank(Path(args.bank), args.split)
    if not records:
        print(f"no tasks found under {args.bank}")
        return

    report, skipped = [], 0
    for record in records:
        result = audit_task(record["task"])
        if "skipped" in result:
            skipped += 1
            continue
        if result["state_problems"] or result["checker_problems"]:
            report.append(
                {
                    "task_id": record["task"]["task_id"],
                    "split": record["meta"].get("split"),
                    "path": str(record["path"]),
                    "task": record["task"],  # stripped before the JSON report is written
                    **result,
                }
            )

    n = len(records) - skipped
    bad_state = [r for r in report if r["state_problems"]]
    bad_checker = [r for r in report if r["checker_problems"]]

    print(
        f"audited {n} tasks against their declared schema"
        + (f" ({skipped} skipped: no schema declared)" if skipped else "")
    )
    print(f"  initial_state does not conform : {len(bad_state)}/{n} ({len(bad_state)/n:.1%})")
    print(f"  checker reads undeclared keys  : {len(bad_checker)}/{n} ({len(bad_checker)/n:.1%})")
    print(f"  clean tasks                    : {n - len(report)}/{n} ({(n - len(report))/n:.1%})")

    for label, key in (("initial_state", "state_problems"), ("checker", "checker_problems")):
        reasons = Counter(_kind(p) for r in report for p in r[key])
        if reasons:
            print(f"\n  most common {label} problems:")
            for reason, count in reasons.most_common(6):
                print(f"    {count:5d}  {reason}")

    print()
    for item in report[: args.show]:
        print(f"- {item['task_id']} [{item['split']}]")
        for problem in (item["state_problems"] + item["checker_problems"])[:3]:
            print(f"    {problem}")

    print(f"\nby split: {dict(Counter(item['split'] for item in report))}")

    if args.mark_unusable:
        mark_unusable(Path(args.bank), report, args.apply)

    if args.json:
        Path(args.json).write_text(
            json.dumps([{k: v for k, v in r.items() if k != "task"} for r in report], indent=2)
        )
        print(f"full report -> {args.json}")


if __name__ == "__main__":
    main()
