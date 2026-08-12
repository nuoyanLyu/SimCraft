"""Why is the screened difficulty distribution bimodal? Offline, no GPU.

The 2026-08-07 screening of 152 freshly generated gc=3 eval tasks came out
121 at 1.00, 24 at 0.00, 7 in between. Two competing readings:

  ceiling  the task is genuinely easy  OR  the checker was already satisfied
           before the agent did anything (a vacuous task).
  floor    the agent genuinely cannot do it  OR  the checker is broken /
           unsatisfiable and every rollout is scored a failure it did not earn.

Both alternatives are decidable from artifacts already on disk: the screening
jsonl carries the initial state, the final state and the full trajectory of
every rollout, and the bank carries the checker. So re-run the judge against
states we already have, and read off which of the four `judge_checker_with_reason`
verdicts each rollout actually produced.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/SimCraft")

from qwen_agentworld.core.schemas import Task
from qwen_agentworld.judge.paired_audit import judge_checker_with_reason
from qwen_agentworld.teacher.task_bank import TaskBank

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default="abtest/screening_0807/screening.jsonl")
ap.add_argument("--out", default="abtest/screening_0807/quality_diagnosis.json")
a = ap.parse_args()

# ---------------------------------------------------------------- load
bank = TaskBank("task_bank")
tasks: dict[str, Task] = {}
meta_of: dict[str, dict] = {}
for _, meta, task in bank._iter_bucket("mcp_notes", 3):
    tasks[task.task_id] = task
    meta_of[task.task_id] = meta

rollouts: dict[str, list[dict]] = collections.defaultdict(list)
for line in Path(a.jsonl).read_text().splitlines():
    if line.strip():
        rec = json.loads(line)
        rollouts[rec["task_id"]].append(rec)

print(f"{len(rollouts)} tasks, {sum(len(v) for v in rollouts.values())} rollouts\n")


def states_of(rec: dict, task: Task) -> list[dict]:
    """initial + one canonical state per executed step, as the judge sees it."""
    steps = ((rec.get("trajectory") or {}).get("steps")) or []
    return [task.initial_state] + [(s.get("simulator_raw_output") or {}).get("next_state", {}) for s in steps]


def n_steps(rec: dict) -> int:
    return len(((rec.get("trajectory") or {}).get("steps")) or [])


rows = []
for task_id, recs in rollouts.items():
    task = tasks.get(task_id)
    if task is None:
        continue
    checker = task.checker

    # 1. VACUITY: is the checker already true of the state the agent starts in?
    #    A task that passes before any tool call measures nothing.
    vacuous, vacuous_reason = judge_checker_with_reason(
        checker, task.initial_state, states=[task.initial_state], task_id=task_id)

    # 2. Re-judge every rollout, keeping the reason rather than the bit.
    reasons = collections.Counter()
    step_counts = []
    passed_step_counts = []
    for rec in recs:
        fs = rec.get("final_state")
        if fs is None:
            reasons["rollout_errored"] += 1
            continue
        ok, why = judge_checker_with_reason(checker, fs, states=states_of(rec, task), task_id=task_id)
        reasons[why] += 1
        step_counts.append(n_steps(rec))
        if ok:
            passed_step_counts.append(n_steps(rec))

    n = sum(reasons.values())
    n_pass = reasons["pass"]
    rate = n_pass / n if n else None

    # 3. For a task that never passes: was the predicate ever true at ANY state
    #    the agent visited? True-then-false means the end-state predicate is
    #    scoring a reversal (or a step-wise checker that should have been set).
    ever_true = False
    if rate == 0.0:
        for rec in recs:
            for st in states_of(rec, task):
                try:
                    ok, _ = judge_checker_with_reason(checker, st, states=[st], task_id=task_id)
                except Exception:  # noqa: BLE001 - a broken predicate is the finding, not a crash
                    ok = False
                if ok:
                    ever_true = True
                    break
            if ever_true:
                break

    rows.append({
        "task_id": task_id,
        "rate": rate,
        "vacuous": bool(vacuous),
        "vacuous_reason": vacuous_reason,
        "reasons": dict(reasons),
        "ever_true_mid_trajectory": ever_true,
        "mean_steps": sum(step_counts) / len(step_counts) if step_counts else None,
        "mean_steps_when_passed": (sum(passed_step_counts) / len(passed_step_counts)
                                   if passed_step_counts else None),
        "n_graph_nodes": len(task.task_graph.nodes) if getattr(task, "task_graph", None) else None,
        "predicate_len": len(checker.executable_predicate),
        "step_wise": bool(checker.step_wise_diagnostics),
        "initial_state_size": len(json.dumps(task.initial_state)),
        "audit": meta_of.get(task_id, {}).get("audit"),
    })

# ---------------------------------------------------------------- report
ceiling = [r for r in rows if r["rate"] == 1.0]
floor = [r for r in rows if r["rate"] == 0.0]
band = [r for r in rows if r["rate"] not in (None, 0.0, 1.0)]

print("=" * 68)
print(f"CEILING (rate==1.00): {len(ceiling)} tasks")
vac = [r for r in ceiling if r["vacuous"]]
print(f"  checker already TRUE in the initial state : {len(vac)}  <-- vacuous, measures nothing")
zero_step = [r for r in ceiling if (r["mean_steps_when_passed"] or 99) <= 1]
print(f"  passed with <=1 tool call on average      : {len(zero_step)}")
print(f"  mean tool calls when passing              : "
      f"{sum(r['mean_steps_when_passed'] or 0 for r in ceiling)/max(len(ceiling),1):.2f}")

print()
print(f"FLOOR (rate==0.00): {len(floor)} tasks")
agg = collections.Counter()
for r in floor:
    for k, v in r["reasons"].items():
        agg[k] += v
print(f"  rollout-level verdicts: {dict(agg)}")
harness = sum(v for k, v in agg.items() if k.startswith("checker_raised") or k == "checker_unsafe")
total = sum(agg.values())
print(f"  scored 0 by a BROKEN checker (raised/unsafe): {harness}/{total} rollouts"
      f"  = {100*harness/max(total,1):.1f}%   <-- misjudged, not hard")
print(f"  predicate true somewhere mid-trajectory     : "
      f"{sum(1 for r in floor if r['ever_true_mid_trajectory'])} tasks  <-- did the work then lost it")
print(f"  vacuous but still 0.00 (contradiction)      : {sum(1 for r in floor if r['vacuous'])}")
print(f"  mean tool calls attempted                   : "
      f"{sum(r['mean_steps'] or 0 for r in floor)/max(len(floor),1):.2f}")

print()
print(f"BAND (0<rate<1): {len(band)} tasks")
for r in sorted(band, key=lambda x: x["rate"]):
    print(f"  {r['task_id']} rate={r['rate']:.2f} steps={r['mean_steps']:.1f} "
          f"nodes={r['n_graph_nodes']} pred_len={r['predicate_len']}")

# ---------------------------------------------------------------- features
print()
print("=" * 68)
print("STATIC FEATURES vs measured rate (does any of them predict difficulty?)")
measured = [r for r in rows if r["rate"] is not None]


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


for feat in ("n_graph_nodes", "predicate_len", "initial_state_size", "mean_steps"):
    pairs = [(r[feat], r["rate"]) for r in measured if r[feat] is not None]
    if len(pairs) < 3:
        continue
    r_ = corr([p[0] for p in pairs], [p[1] for p in pairs])
    if r_ is None:  # a feature with no variance predicts nothing, by definition
        print(f"  {feat:20s} constant across all {len(pairs)} tasks — no signal available")
        continue
    lo = [p[1] for p in pairs if p[0] <= sorted(p2[0] for p2 in pairs)[len(pairs) // 3]]
    hi = [p[1] for p in pairs if p[0] >= sorted(p2[0] for p2 in pairs)[2 * len(pairs) // 3]]
    print(f"  {feat:20s} r={r_:+.3f}  n={len(pairs)}   "
          f"bottom-third rate={sum(lo)/max(len(lo),1):.2f}  top-third rate={sum(hi)/max(len(hi),1):.2f}")

Path(a.out).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))
print(f"\nwrote {a.out}")
