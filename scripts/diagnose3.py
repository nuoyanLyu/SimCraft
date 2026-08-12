"""Clause-level anatomy of the checkers, offline.

A predicate like

    len(state['notes']) == 3
    and any(n['title']=='Project Plan' and 'important' in n['tags'] ...)
    and any(n['title']=='Meeting Notes' and n['content']=='Discussed budget...' and n['tags']==[] ...)

mixes two different demands. The second conjunct asks whether the agent did the
job. The first and third ask whether it left everything else byte-identical —
they are already true before the agent starts, and can only ever be lost. Call
those *preservation* clauses and the rest *achievement* clauses.

The distinction decides how to read a 0.00 task. If it fails an achievement
clause, the agent could not do the work: real difficulty, keep it. If it fails
only preservation clauses, the agent did the work and was scored down for a
side effect the prompt never forbade: a checker artifact, not difficulty.

Split each top-level conjunct, evaluate it against the initial state to label
it, then evaluate it against each rollout's final state to see which kind of
clause actually broke.
"""

from __future__ import annotations

import ast
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/SimCraft")
import logging
logging.disable(logging.WARNING)

from qwen_agentworld.teacher.safe_predicate import evaluate_predicate
from qwen_agentworld.teacher.task_bank import TaskBank

bank = TaskBank("task_bank")
tasks = {t.task_id: t for _, _, t in bank._iter_bucket("mcp_notes", 3)}
diag = {r["task_id"]: r for r in
        json.loads(Path("abtest/screening_0807/quality_diagnosis.json").read_text())["rows"]}

rollouts = collections.defaultdict(list)
for line in Path("abtest/screening_0807/screening.jsonl").read_text().splitlines():
    if line.strip():
        rec = json.loads(line)
        rollouts[rec["task_id"]].append(rec)


def conjuncts(pred: str) -> list[str]:
    """Top-level `and` operands, as source text."""
    try:
        tree = ast.parse(pred.strip(), mode="eval").body
    except SyntaxError:
        return []
    out: list[str] = []

    def walk(node):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for v in node.values:
                walk(v)
        else:
            out.append(ast.unparse(node))

    walk(tree)
    return out


def truth(clause: str, state: dict):
    try:
        return bool(evaluate_predicate(clause, state))
    except Exception:  # noqa: BLE001 - an unevaluable clause is its own category
        return None


summary = collections.defaultdict(lambda: {"pres": 0, "ach": 0, "bad": 0})
broke = collections.defaultdict(lambda: collections.Counter())
per_task = []

for tid, r in diag.items():
    if r["rate"] is None or tid not in tasks:
        continue
    group = "ceiling" if r["rate"] == 1.0 else "floor" if r["rate"] == 0.0 else "band"
    task = tasks[tid]
    cs = conjuncts(task.checker.executable_predicate)
    if not cs:
        continue

    kinds = []
    for c in cs:
        t0 = truth(c, task.initial_state)
        kinds.append("bad" if t0 is None else "pres" if t0 else "ach")
    for k in kinds:
        summary[group][k] += 1

    # Which kind of clause actually broke, on the rollouts that failed?
    failed_pres = failed_ach = failed_bad = 0
    n_failed_rollouts = 0
    for rec in rollouts.get(tid, []):
        fs = rec.get("final_state")
        if fs is None:
            continue
        vals = [truth(c, fs) for c in cs]
        if all(v is True for v in vals):
            continue
        n_failed_rollouts += 1
        for c, k, v in zip(cs, kinds, vals):
            if v is not True:
                if v is None:
                    failed_bad += 1
                elif k == "pres":
                    failed_pres += 1
                else:
                    failed_ach += 1
    broke[group]["pres"] += failed_pres
    broke[group]["ach"] += failed_ach
    broke[group]["bad"] += failed_bad

    if group == "floor" and n_failed_rollouts:
        per_task.append({
            "task_id": tid,
            "n_clauses": len(cs),
            "n_pres": kinds.count("pres"),
            "n_ach": kinds.count("ach"),
            "failed_pres": failed_pres,
            "failed_ach": failed_ach,
            "failed_bad": failed_bad,
            "only_preservation_broke": failed_ach == 0 and failed_bad == 0 and failed_pres > 0,
        })

print("=" * 74)
print("CLAUSE COMPOSITION  (pres = already true before the agent acts; ach = the actual job)")
print(f"{'group':9} {'clauses':>8} {'preservation':>13} {'achievement':>12} {'unevaluable':>12}")
for g in ("ceiling", "band", "floor"):
    s = summary[g]
    tot = s["pres"] + s["ach"] + s["bad"]
    if not tot:
        continue
    print(f"{g:9} {tot:>8} {s['pres']:>8} ({100*s['pres']/tot:>3.0f}%) "
          f"{s['ach']:>7} ({100*s['ach']/tot:>3.0f}%) {s['bad']:>7} ({100*s['bad']/tot:>3.0f}%)")

print()
print("=" * 74)
print("WHICH KIND OF CLAUSE BROKE, on failing rollouts")
print(f"{'group':9} {'preservation':>13} {'achievement':>12} {'unevaluable':>12}")
for g in ("ceiling", "band", "floor"):
    b = broke[g]
    tot = sum(b.values())
    if not tot:
        continue
    print(f"{g:9} {b['pres']:>8} ({100*b['pres']/tot:>3.0f}%) "
          f"{b['ach']:>7} ({100*b['ach']/tot:>3.0f}%) {b['bad']:>7} ({100*b['bad']/tot:>3.0f}%)")

print()
print("=" * 74)
only_pres = [p for p in per_task if p["only_preservation_broke"]]
print(f"FLOOR tasks whose EVERY failure was a preservation clause: {len(only_pres)} of {len(per_task)}")
print("  (the agent did the job and was scored down for touching something else)")
for p in only_pres:
    print(f"    {p['task_id']}  clauses={p['n_clauses']} (pres={p['n_pres']} ach={p['n_ach']})")

print()
print("FLOOR task detail:")
print(f"  {'task_id':22} {'clauses':>7} {'pres':>5} {'ach':>4} | {'broke_pres':>10} {'broke_ach':>9} {'broke_bad':>9}")
for p in sorted(per_task, key=lambda x: -x["failed_pres"]):
    print(f"  {p['task_id']:22} {p['n_clauses']:>7} {p['n_pres']:>5} {p['n_ach']:>4} | "
          f"{p['failed_pres']:>10} {p['failed_ach']:>9} {p['failed_bad']:>9}")

Path("abtest/screening_0807/clause_anatomy.json").write_text(
    json.dumps({"per_task_floor": per_task,
                "composition": {k: dict(v) for k, v in summary.items()},
                "broke": {k: dict(v) for k, v in broke.items()}}, ensure_ascii=False, indent=2))
print("\nwrote abtest/screening_0807/clause_anatomy.json")
