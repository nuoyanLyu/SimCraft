"""Audit whether generated tasks are actually scorable: is the checker's
predicate consistent with the instruction the agent is given?

Motivated by the 2026-07-28 null A/B. Checkers were synthesised from the tool
graph alone, so the teacher invented concrete values the instruction never
mentioned ("tag with 'dinner'" scored against `tags == ['recipe']`). Four of
twelve eval tasks were unpassable by construction, and they were exactly the
four that scored 0.000 in every arm. Several of the rest were the opposite
failure -- a predicate so weak that ignoring the instruction still passed.

Two failure modes, both fatal to a pass-rate measurement:

  UNPASSABLE  the predicate cannot be satisfied by following the instruction,
              so the task is a permanent 0 no matter how good the agent is;
  TOO_WEAK    the predicate is satisfied by states that ignore most of the
              instruction, so the task is a near-permanent 1.

Either one pins a task at floor or ceiling, where it carries no information
about the playbook.

Usage:
    # calibrate the auditor against the known-bad set
    python scripts/audit_task_quality.py --tasks abtest/ds4_0728/eval_tasks.json
    # audit a freshly generated batch
    python scripts/audit_task_quality.py --generate 12 --graph-complexity 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_agentworld.core.schemas import Task
from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, TaskBank
from qwen_agentworld.teacher.task_generator import generate_task

import live_smoke_real_sim as smoke

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("audit_task_quality")

_AUDIT_SYSTEM = (
    "You audit whether an automatically generated tool-use task can be scored fairly. "
    "You are given the instruction shown to the agent, the starting state, and the Python "
    "boolean predicate used to decide whether the agent succeeded. The predicate is "
    "evaluated against the final state as a dict named `state`. "
    "A task may ALSO carry a step-wise predicate, evaluated against the whole sequence of "
    "states as a list named `states` (states[0] is the start, states[-1] the end). Both "
    "predicates must hold for the task to pass, so judge them together: a step the end "
    "state cannot show (something created and then deleted again) counts as verified if "
    "the step-wise predicate checks it. Only call the pair TOO_WEAK if NEITHER checks it. "
    "Decide two things, independently and strictly.\n"
    "1. UNPASSABLE: is there NO final state reachable by correctly following the instruction "
    "that satisfies the predicate? This is the case when the predicate demands a concrete "
    "value the instruction never asks for (a different tag, a different title, an edit to an "
    "unrelated entity), or when it contradicts what the instruction asks for. Entities the "
    "instruction does not mention should be required to keep their starting values -- that is "
    "correct, not a demand.\n"
    "2. TOO_WEAK: would a final state that ignores most of the instruction still satisfy the "
    "predicate? For example the instruction asks for specific content and the predicate only "
    "counts how many items exist.\n"
    "Reply with a single JSON object and nothing else: "
    '{"unpassable": <bool>, "unpassable_reason": "<one sentence or empty>", '
    '"too_weak": <bool>, "too_weak_reason": "<one sentence or empty>"}'
)


def audit_one(judge, task: Task) -> dict:
    step_wise = task.checker.step_wise_predicate
    user = (
        f"Instruction shown to the agent:\n{task.natural_language_prompt}\n\n"
        f"Starting state:\n{json.dumps(task.initial_state, indent=2, ensure_ascii=False)}\n\n"
        f"End-state predicate (over `state`):\n{task.checker.executable_predicate}\n\n"
        + (f"Step-wise predicate (over `states`):\n{step_wise}\n\n"
           if step_wise else "Step-wise predicate: none — the end-state predicate is the whole checker.\n\n")
        + "Produce the JSON object described in the system prompt."
    )
    for attempt in range(1, 4):
        result = judge.chat(
            messages=[{"role": "system", "content": _AUDIT_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=600,
        )
        try:
            payload = extract_json_object(result.content or "")
            return {
                "unpassable": bool(payload["unpassable"]),
                "unpassable_reason": payload.get("unpassable_reason", ""),
                "too_weak": bool(payload["too_weak"]),
                "too_weak_reason": payload.get("too_weak_reason", ""),
            }
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("audit parse failed (attempt %d): %s", attempt, exc)
    return {"unpassable": None, "unpassable_reason": "audit failed", "too_weak": None, "too_weak_reason": ""}


def load_tasks(path: str) -> list[Task]:
    return [Task.model_validate(t) for t in json.loads(Path(path).read_text())]


def generate_tasks(teacher, n: int, gc: int, bank: TaskBank) -> list[Task]:
    tasks = []
    for i in range(n):
        try:
            t = generate_task(teacher, smoke.NOTES_TOOLS, smoke.TOOL_FAMILY, min_nodes=gc, max_nodes=gc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("generation %d/%d failed: %s", i + 1, n, exc)
            continue
        bank.save(t, split=SPLIT_EVAL, origin="audit_task_quality",
                  teacher_model=getattr(teacher, "model", ""), gc=gc)
        tasks.append(t)
        logger.info("generated %d/%d", i + 1, n)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="audit an existing eval_tasks.json")
    ap.add_argument("--generate", type=int, default=0, help="generate this many fresh tasks first")
    ap.add_argument("--from-bank", type=int, default=0,
                    help="audit this many already-banked tasks (no generation cost)")
    ap.add_argument("--graph-complexity", type=int, default=3)
    ap.add_argument("--bank-dir", default="task_bank")
    ap.add_argument("--split", default=SPLIT_EVAL,
                    help="which --from-bank split to audit (default: eval)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    teacher = TeacherClient()
    logger.info("teacher model: %s", teacher.model)

    if a.generate:
        tasks = generate_tasks(teacher, a.generate, a.graph_complexity, TaskBank(a.bank_dir))
    elif a.from_bank:
        tasks = TaskBank(a.bank_dir).draw(
            smoke.TOOL_FAMILY, a.graph_complexity, a.from_bank, split=a.split)
        logger.info("drew %d banked tasks", len(tasks))
    elif a.tasks:
        tasks = load_tasks(a.tasks)
    else:
        ap.error("pass --tasks, --from-bank or --generate")

    rows = []
    for i, t in enumerate(tasks, 1):
        verdict = audit_one(teacher, t)
        rows.append({"task_id": t.task_id,
                     "natural_language_prompt": t.natural_language_prompt,
                     "executable_predicate": t.checker.executable_predicate,
                     "step_wise_predicate": t.checker.step_wise_predicate,
                     **verdict})
        flags = [k.upper() for k in ("unpassable", "too_weak") if verdict[k]]
        logger.info("%d/%d %s %s", i, len(tasks), t.task_id, ",".join(flags) or "ok")

    n = len(rows)
    bad = sum(1 for r in rows if r["unpassable"])
    weak = sum(1 for r in rows if r["too_weak"])
    clean = sum(1 for r in rows if not r["unpassable"] and not r["too_weak"])
    print("\n============ TASK QUALITY AUDIT ============")
    print(f"tasks audited : {n}")
    print(f"UNPASSABLE    : {bad}/{n}  (permanent 0 — checker contradicts the instruction)")
    print(f"TOO_WEAK      : {weak}/{n}  (near-permanent 1 — checker barely tests the instruction)")
    print(f"clean         : {clean}/{n}")
    print("============================================\n")
    for r in rows:
        if r["unpassable"] or r["too_weak"]:
            print(f"--- {r['task_id']}")
            print(f"    NL   : {r['natural_language_prompt'][:160]}")
            print(f"    PRED : {r['executable_predicate'][:200]}")
            if r["unpassable"]:
                print(f"    UNPASSABLE: {r['unpassable_reason']}")
            if r["too_weak"]:
                print(f"    SW   : {str(r.get('step_wise_predicate'))[:200]}")
                print(f"    TOO_WEAK  : {r['too_weak_reason']}")

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"n": n, "unpassable": bad, "too_weak": weak, "clean": clean, "rows": rows},
            ensure_ascii=False, indent=2))
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
