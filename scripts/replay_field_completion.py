"""Offline replay: how many recorded judgments change once the simulator's
dropped fields are completed against the family's declared state schema?

The agent's actions are fixed — they are read straight out of the recorded
trajectories — so this is a counterfactual on *judging only*: same rollouts,
same checkers, the states rebuilt with `complete_fields` applied at each step.
Anything that flips here was decided by a missing key rather than by the
agent's behaviour.

GPU-free: it reads `smoke_test_results/<run>/iteration_*.json` and calls
nothing but the predicate evaluator.

    python scripts/replay_field_completion.py smoke_test_results/bigevolve_0729
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agentworld.core.schemas import CheckerSpec, ToolCall
from qwen_agentworld.judge.paired_audit import _PREDICATE_RUNTIME_ERRORS, judge_checker
from qwen_agentworld.simulator_gym.env import complete_fields, record_action
from qwen_agentworld.tools.state_schema import get_schema
from qwen_agentworld.teacher.safe_predicate import evaluate_predicate, evaluate_step_wise_predicate


def _raises(checker: CheckerSpec, final_state: dict, states: list[dict]) -> str | None:
    """The exception type a predicate raises against these states, if any."""
    try:
        if checker.step_wise_diagnostics and checker.step_wise_predicate:
            evaluate_step_wise_predicate(checker.step_wise_predicate, states)
        else:
            evaluate_predicate(checker.executable_predicate, final_state)
    except _PREDICATE_RUNTIME_ERRORS as exc:
        return type(exc).__name__
    except Exception as exc:  # UnsafePredicateError and friends
        return type(exc).__name__
    return None


def replay(record: dict) -> dict:
    task = record["task"]
    checker = CheckerSpec.model_validate(task["checker"])
    # The schema is what makes completion possible on a collection that starts
    # empty, which is exactly the case sibling inference cannot reach.
    schema = get_schema(task.get("tool_family"))

    state = dict(task["initial_state"])
    old_states = [dict(task["initial_state"])]
    new_states = [dict(task["initial_state"])]

    for step in record["trajectory"]["steps"]:
        raw = step.get("simulator_raw_output") or {}
        recorded_next = raw.get("next_state")
        if not isinstance(recorded_next, dict):
            continue
        old_states.append(recorded_next)
        call = ToolCall(**step["tool_call"])
        state = record_action(state, complete_fields(state, recorded_next, schema), call)
        new_states.append(state)

    old_final = record["final_state"]
    new_final = state

    return {
        "task_id": task["task_id"],
        "recorded_passed": record["checker_passed"],
        "old_passed": judge_checker(checker, old_final, states=old_states, task_id=task["task_id"]),
        "new_passed": judge_checker(checker, new_final, states=new_states, task_id=task["task_id"]),
        "old_error": _raises(checker, old_final, old_states),
        "new_error": _raises(checker, new_final, new_states),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. smoke_test_results/bigevolve_0729")
    args = ap.parse_args()

    logging.disable(logging.WARNING)  # the per-task warnings are the thing being counted

    rows = []
    for path in sorted(Path(args.run_dir).glob("iteration_*.json")):
        data = json.loads(path.read_text())
        for record in data.get("tasks", []):
            if "trajectory" not in record:
                continue
            row = replay(record)
            row["iteration"] = data["iteration"]
            rows.append(row)

    if not rows:
        print("no task records with trajectories found")
        return

    n = len(rows)
    old_err = [r for r in rows if r["old_error"]]
    new_err = [r for r in rows if r["new_error"]]
    flips_to_pass = [r for r in rows if r["new_passed"] and not r["old_passed"]]
    flips_to_fail = [r for r in rows if r["old_passed"] and not r["new_passed"]]

    print(f"judgments replayed: {n}")
    print(f"  predicate errored before completion: {len(old_err)} ({len(old_err)/n:.1%})")
    print(f"  predicate errored after  completion: {len(new_err)} ({len(new_err)/n:.1%})")
    print(f"  pass rate before: {sum(r['old_passed'] for r in rows)/n:.3f}")
    print(f"  pass rate after : {sum(r['new_passed'] for r in rows)/n:.3f}")
    print(f"  fail -> pass: {len(flips_to_pass)}")
    print(f"  pass -> fail: {len(flips_to_fail)}")

    by_type: dict[str, int] = {}
    for r in old_err:
        by_type[r["old_error"]] = by_type.get(r["old_error"], 0) + 1
    print(f"  error types before: {by_type}")
    if new_err:
        print(f"  still erroring after: {sorted({r['task_id'] for r in new_err})[:10]}")

    out = Path(args.run_dir) / "field_completion_replay.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"per-judgment rows -> {out}")


if __name__ == "__main__":
    main()
