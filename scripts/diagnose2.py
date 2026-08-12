"""Follow-ups to diagnose_task_quality.py, all offline.

1. Does re-judging reproduce the rate screening recorded? A mismatch would mean
   the number the eval set is selected on is not the number the A/B measures.
2. Ceiling tasks are not vacuous, so why does the agent pass 80% of them? Test
   the remaining checker-weakness reading: how much does a checker actually
   assert? A predicate with one clause is far easier to satisfy by accident
   than one with eight.
3. Is graph complexity a difficulty dial? The bank has screened gc=3 and gc=4
   buckets, so this is a direct comparison rather than an argument.
"""

from __future__ import annotations

import ast
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/SimCraft")
import logging
logging.disable(logging.WARNING)

from qwen_agentworld.judge.paired_audit import judge_checker_with_reason
from qwen_agentworld.teacher.task_bank import TaskBank

bank = TaskBank("task_bank")
diag = {r["task_id"]: r for r in json.loads(Path("abtest/screening_0807/quality_diagnosis.json").read_text())["rows"]}
stored = {r["task_id"]: r["baseline_pass_rate"]
          for r in json.loads(Path("abtest/screening_0807/screening.json").read_text())["rows"]}

# ---------------------------------------------------------------- 1. agreement
print("=" * 70)
print("1. RE-JUDGE vs STORED RATE")
mismatch = [(t, stored[t], diag[t]["rate"]) for t in diag if t in stored and stored[t] != diag[t]["rate"]]
print(f"   {len(diag)} tasks compared, {len(mismatch)} disagree")
for t, s, d in mismatch[:10]:
    print(f"     {t}  stored={s}  rejudged={d}  reasons={diag[t]['reasons']}")

# ---------------------------------------------------------------- 2. how much does a checker assert?
def n_clauses(pred: str) -> int:
    """Top-level conjuncts in the predicate: how many things must be true."""
    try:
        tree = ast.parse(pred.strip(), mode="eval").body
    except SyntaxError:
        return -1
    def walk(node):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return sum(walk(v) for v in node.values)
        return 1
    return walk(tree)


tasks = {t.task_id: t for _, _, t in bank._iter_bucket("mcp_notes", 3)}
buckets = collections.defaultdict(list)
for tid, r in diag.items():
    if r["rate"] is None or tid not in tasks:
        continue
    group = "ceiling" if r["rate"] == 1.0 else "floor" if r["rate"] == 0.0 else "band"
    pred = tasks[tid].checker.executable_predicate
    buckets[group].append((n_clauses(pred), len(pred), r["rate"]))

print()
print("=" * 70)
print("2. CHECKER STRICTNESS BY OUTCOME GROUP")
print(f"   {'group':8} {'n':>4} {'clauses (median)':>18} {'pred chars (median)':>21}")
for g in ("ceiling", "band", "floor"):
    b = buckets[g]
    if not b:
        continue
    print(f"   {g:8} {len(b):>4} {statistics.median(c for c, _, _ in b):>18.1f} "
          f"{statistics.median(l for _, l, _ in b):>21.0f}")

allpairs = [(c, r) for g in buckets for c, _, r in buckets[g] if c > 0]
n = len(allpairs)
mx = sum(p[0] for p in allpairs) / n
my = sum(p[1] for p in allpairs) / n
num = sum((x - mx) * (y - my) for x, y in allpairs)
dx = sum((x - mx) ** 2 for x, _ in allpairs) ** 0.5
dy = sum((y - my) ** 2 for _, y in allpairs) ** 0.5
print(f"   corr(n_clauses, pass_rate) = {num/(dx*dy):+.3f}   n={n}")
by_clause = collections.defaultdict(list)
for c, r in allpairs:
    by_clause[min(c, 6)].append(r)
print(f"   {'clauses':>8} {'n':>4} {'mean rate':>10}")
for c in sorted(by_clause):
    v = by_clause[c]
    print(f"   {c:>8} {len(v):>4} {sum(v)/len(v):>10.2f}")

# ---------------------------------------------------------------- 3. gc dial
print()
print("=" * 70)
print("3. GRAPH COMPLEXITY AS A DIFFICULTY DIAL (bank-wide, screened only)")
for gc in (3, 4):
    rates = [m.get("baseline_pass_rate") for _, m, _ in bank._iter_bucket("mcp_notes", gc)
             if m.get("baseline_pass_rate") is not None]
    if not rates:
        print(f"   gc={gc}: nothing screened")
        continue
    pinned_hi = sum(1 for r in rates if r == 1.0)
    pinned_lo = sum(1 for r in rates if r == 0.0)
    band_n = len(rates) - pinned_hi - pinned_lo
    print(f"   gc={gc}: n={len(rates):>3}  mean={sum(rates)/len(rates):.3f}  "
          f"at 1.00={pinned_hi:>3} ({100*pinned_hi/len(rates):.0f}%)  "
          f"at 0.00={pinned_lo:>3} ({100*pinned_lo/len(rates):.0f}%)  "
          f"in band={band_n:>2} ({100*band_n/len(rates):.0f}%)")

# ---------------------------------------------------------------- 4. samples
print()
print("=" * 70)
print("4. SAMPLE PREDICATES")
for g in ("ceiling", "band"):
    items = [(tid, r) for tid, r in diag.items()
             if tid in tasks and r["rate"] is not None
             and ((g == "ceiling" and r["rate"] == 1.0) or (g == "band" and 0 < r["rate"] < 1))]
    items.sort(key=lambda x: x[0])
    print(f"\n--- {g} ---")
    for tid, r in items[:3]:
        t = tasks[tid]
        print(f"\n  [{tid}] rate={r['rate']}  clauses={n_clauses(t.checker.executable_predicate)}")
        print(f"  prompt: {t.natural_language_prompt[:220]}")
        print(f"  pred  : {t.checker.executable_predicate[:400]}")

# ---------------------------------------------------------------- 5. broken checkers bank-wide
print()
print("=" * 70)
print("5. HOW MANY BANKED CHECKERS ARE BROKEN AT ALL? (evaluated vs their own initial state)")
broken = collections.Counter()
examples = {}
for gc in (3, 4):
    for _, m, t in bank._iter_bucket("mcp_notes", gc):
        ok, why = judge_checker_with_reason(t.checker, t.initial_state,
                                            states=[t.initial_state], task_id=t.task_id)
        if why.startswith("checker_raised") or why == "checker_unsafe":
            broken[why] += 1
            examples.setdefault(why, (t.task_id, t.checker.executable_predicate[:200]))
        elif ok:
            broken["TRUE_at_initial_state(vacuous)"] += 1
            examples.setdefault("TRUE_at_initial_state(vacuous)", (t.task_id, t.checker.executable_predicate[:200]))
        else:
            broken["ok"] += 1
total = sum(broken.values())
for k, v in broken.most_common():
    print(f"   {k:38s} {v:>4}  ({100*v/total:.1f}%)")
    if k != "ok" and k in examples:
        print(f"       e.g. {examples[k][0]}: {examples[k][1][:160]}")
