"""Does the LLM judge agree with the checker, and where it disagrees, who is right?

GPU-free. The 2026-08-07 screening already wrote every rollout it made to
`abtest/screening_0807/screening.jsonl` -- instruction, initial state, full
trajectory, final state. That is exactly what `judge_with_llm` needs, so the
judge can be evaluated on real rollouts without re-running a single one.

This is the measurement that has to happen before `--judge-mode llm` is
trusted with an A/B. Three things it answers:

  1. Does the judge parse reliably against the live relay? A judge that fails
     to return usable JSON 10% of the time is a broken instrument no matter how
     good its verdicts are.
  2. How often does it agree with the predicate? Blanket disagreement means it
     is measuring something else, not fixing anything.
  3. On the 8 tasks the clause anatomy identified as *misjudged* -- 3 with
     syntactically broken checkers, 5 that failed only preservation clauses --
     does it flip them to pass? Those are the cases the judge exists for, and
     they are the only ones where the checker is known to be wrong, so they are
     the only place agreement is the wrong thing to want.

Usage:
    python scripts/audit_llm_judge.py --per-group 20 --workers 12
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_agentworld.core.schemas import Task, Trajectory
from qwen_agentworld.judge.llm_judge import DEFAULT_JUDGE_THRESHOLD, judge_with_llm
from qwen_agentworld.judge.paired_audit import judge_checker_with_reason
from qwen_agentworld.judge.verdict import states_for
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.teacher.task_bank import TaskBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qwen_agentworld.judge.llm_judge").setLevel(logging.ERROR)
logger = logging.getLogger("audit_llm_judge")

# From scripts/diagnose3.py: floor tasks the clause anatomy showed were scored
# down by the checker rather than by the agent. Listed rather than recomputed so
# this script stays readable; re-derive with diagnose3.py if the bank changes.
BROKEN_CHECKER = {"task_64b4bcd258fa", "task_b0eb10eb9111", "task_c054f2b0d771"}
PRESERVATION_ONLY = {"task_1135f23d629e", "task_20ad5ba3a8dc", "task_92eef12855c9",
                     "task_95f7112d536e", "task_adfefb11df60"}
KNOWN_MISJUDGED = BROKEN_CHECKER | PRESERVATION_ONLY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="abtest/screening_0807/screening.jsonl")
    ap.add_argument("--bank-dir", default="task_bank")
    ap.add_argument("--tool-family", default="mcp_notes")
    ap.add_argument("--graph-complexity", type=int, default=3)
    ap.add_argument("--per-group", type=int, default=20,
                    help="ceiling tasks to sample; floor and band are always taken whole "
                         "(they are small, and they are where the disagreement lives)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_JUDGE_THRESHOLD)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="abtest/screening_0807/llm_judge_audit.json")
    a = ap.parse_args()

    bank = TaskBank(a.bank_dir)
    tasks = {t.task_id: t for _, _, t in bank._iter_bucket(a.tool_family, a.graph_complexity)}

    rollouts: dict[str, list[dict]] = collections.defaultdict(list)
    for line in Path(a.jsonl).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("final_state") is not None and rec["task_id"] in tasks:
                rollouts[rec["task_id"]].append(rec)

    # Group by the rate the checker measured, recomputed here rather than read
    # off the bank so the grouping and the comparison use one judge.
    rate: dict[str, float] = {}
    for tid, recs in rollouts.items():
        passes = sum(
            judge_checker_with_reason(tasks[tid].checker, r["final_state"],
                                      states=_states(tasks[tid], r), task_id=tid)[0]
            for r in recs
        )
        rate[tid] = passes / len(recs)

    floor = [t for t, r in rate.items() if r == 0.0]
    band = [t for t, r in rate.items() if 0.0 < r < 1.0]
    ceiling = [t for t, r in rate.items() if r == 1.0]
    random.Random(a.seed).shuffle(ceiling)
    selected = floor + band + ceiling[: a.per_group]
    jobs = [(tid, r) for tid in selected for r in rollouts[tid]]
    logger.info("judging %d rollouts over %d tasks (floor=%d band=%d ceiling=%d of %d)",
                len(jobs), len(selected), len(floor), len(band), min(a.per_group, len(ceiling)),
                len(ceiling))

    judge = TeacherClient()
    lock = threading.Lock()
    done = [0]

    def one(job):
        tid, rec = job
        task = tasks[tid]
        traj = Trajectory.model_validate(rec["trajectory"]) if rec.get("trajectory") else None
        checker_passed, checker_reason = judge_checker_with_reason(
            task.checker, rec["final_state"], states=_states(task, rec), task_id=tid)
        v = judge_with_llm(judge, task, rec["final_state"], trajectory=traj, threshold=a.threshold)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                logger.info("%d/%d", done[0], len(jobs))
        return {"task_id": tid, "rep": rec.get("rep"), "checker_passed": checker_passed,
                "checker_reason": checker_reason, "llm_score": v.score, "llm_passed": v.passed,
                "llm_reason": v.reason, "llm_error": v.error}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(one, jobs))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"threshold": a.threshold, "rows": rows},
                                      ensure_ascii=False, indent=2))
    report(rows, rate, a.threshold)
    logger.info("wrote %s", a.out)


def _states(task: Task, rec: dict) -> list[dict]:
    traj = Trajectory.model_validate(rec["trajectory"]) if rec.get("trajectory") else None
    return states_for(task, traj)


def report(rows: list[dict], rate: dict[str, float], threshold: float) -> None:
    def group_of(tid):
        r = rate[tid]
        return "ceiling" if r == 1.0 else "floor" if r == 0.0 else "band"

    print("\n" + "=" * 74)
    errs = [r for r in rows if r["llm_error"]]
    print(f"1. RELIABILITY: {len(errs)}/{len(rows)} judge calls returned no usable score "
          f"({100 * len(errs) / max(len(rows), 1):.1f}%)")
    for e in errs[:3]:
        print(f"     {e['task_id']} rep={e['rep']}: {e['llm_error']}")

    ok = [r for r in rows if not r["llm_error"]]
    print()
    print("=" * 74)
    print("2. AGREEMENT WITH THE CHECKER, per rollout")
    print(f"   {'group':9} {'n':>5} {'agree':>7} {'checker=T llm=F':>17} {'checker=F llm=T':>17}")
    for g in ("ceiling", "band", "floor"):
        rs = [r for r in ok if group_of(r["task_id"]) == g]
        if not rs:
            continue
        agree = sum(r["checker_passed"] == r["llm_passed"] for r in rs)
        tf = sum(r["checker_passed"] and not r["llm_passed"] for r in rs)
        ft = sum((not r["checker_passed"]) and r["llm_passed"] for r in rs)
        print(f"   {g:9} {len(rs):>5} {100 * agree / len(rs):>6.0f}% {tf:>17} {ft:>17}")

    print()
    print("=" * 74)
    print("3. THE 8 TASKS THE CLAUSE ANATOMY CALLED MISJUDGED")
    print("   (checker says 0.00 on all of them; a judge that is working rescues them)")
    print(f"   {'task_id':22} {'why':16} {'llm mean':>9} {'llm rate':>9}")
    rescued = 0
    for tid in sorted(KNOWN_MISJUDGED):
        rs = [r for r in ok if r["task_id"] == tid]
        if not rs:
            continue
        mean = sum(r["llm_score"] for r in rs) / len(rs)
        prate = sum(r["llm_passed"] for r in rs) / len(rs)
        rescued += prate > 0
        why = "broken_checker" if tid in BROKEN_CHECKER else "preservation"
        print(f"   {tid:22} {why:16} {mean:>9.2f} {prate:>9.2f}")
    print(f"   -> {rescued}/{len(KNOWN_MISJUDGED)} now pass at least one rollout")

    print()
    print("=" * 74)
    print("4. WHAT THE PASS-RATE DISTRIBUTION LOOKS LIKE UNDER EACH JUDGE")
    print("   (the point of the exercise: a distribution with tasks between the pins)")
    for name, key in (("checker", "checker_passed"), ("llm", "llm_passed")):
        by_task = collections.defaultdict(list)
        for r in ok:
            by_task[r["task_id"]].append(r[key])
        rates = [sum(v) / len(v) for v in by_task.values()]
        pinned_hi = sum(1 for r in rates if r == 1.0)
        pinned_lo = sum(1 for r in rates if r == 0.0)
        inband = len(rates) - pinned_hi - pinned_lo
        print(f"   {name:8} n={len(rates):>3}  mean={sum(rates) / len(rates):.3f}  "
              f"at 1.00={pinned_hi:>3}  at 0.00={pinned_lo:>3}  "
              f"strictly between={inband:>3} ({100 * inband / len(rates):.0f}%)")
    print("   (this sample over-weights floor and band by construction, so read the")
    print("    band share as a comparison between judges, not as a bank-wide yield)")

    print()
    print("=" * 74)
    print("5. SAMPLE DISAGREEMENTS (checker=False, llm=True) — read these by hand")
    flips = [r for r in ok if not r["checker_passed"] and r["llm_passed"]]
    for r in flips[:6]:
        print(f"\n   {r['task_id']} rep={r['rep']} score={r['llm_score']:.2f} "
              f"checker_reason={r['checker_reason']}")
        print(f"     judge: {r['llm_reason'][:300]}")


if __name__ == "__main__":
    main()
