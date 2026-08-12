"""Re-score the already-recorded screening rollouts with the LLM judge, GPU-free.

Switching `--judge-mode` to `llm` invalidates every stored `baseline_pass_rate`
in the bank: those rates were measured by the checker, and the eval set is drawn
band-only *on those rates*. Re-running 608 rollouts to fix that would cost an
hour of GPU for no new information -- the 2026-08-07 screening already saved
every rollout's instruction, initial state, trajectory and final state.

So re-judge the saved rollouts instead. Same agent, same playbook fingerprint,
same rollouts; only the judge changes, which is exactly the intended change.

The checker-measured rates this overwrites stay recoverable in
`abtest/screening_0807/screening.json`.

Usage:
    python scripts/rejudge_bank.py --workers 16
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_agentworld.core.schemas import Trajectory
from qwen_agentworld.judge.llm_judge import DEFAULT_JUDGE_THRESHOLD, judge_with_llm
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.core.schemas import Playbook
from qwen_agentworld.playbook_store.store import fingerprint
from qwen_agentworld.teacher.task_bank import TaskBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qwen_agentworld.judge.llm_judge").setLevel(logging.ERROR)
logger = logging.getLogger("rejudge_bank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="abtest/screening_0807/screening.jsonl")
    ap.add_argument("--bank-dir", default="task_bank")
    ap.add_argument("--tool-family", default="mcp_notes")
    ap.add_argument("--graph-complexity", type=int, default=3)
    ap.add_argument("--agent-model", default="Qwen3-8B")
    ap.add_argument("--threshold", type=float, default=DEFAULT_JUDGE_THRESHOLD)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--band-low", type=float, default=0.2)
    ap.add_argument("--band-high", type=float, default=0.6)
    ap.add_argument("--out", default="abtest/screening_0807/rejudged.json")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not touch the bank")
    a = ap.parse_args()

    bank = TaskBank(a.bank_dir)
    tasks = {t.task_id: t for _, _, t in bank._iter_bucket(a.tool_family, a.graph_complexity)}

    rollouts: dict[str, list[dict]] = collections.defaultdict(list)
    for line in Path(a.jsonl).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("final_state") is not None and rec["task_id"] in tasks:
                rollouts[rec["task_id"]].append(rec)

    jobs = [(tid, r) for tid, recs in rollouts.items() for r in recs]
    logger.info("re-judging %d saved rollouts over %d tasks at threshold %.2f",
                len(jobs), len(rollouts), a.threshold)

    judge = TeacherClient()
    lock = threading.Lock()
    done = [0]

    def one(job):
        tid, rec = job
        traj = Trajectory.model_validate(rec["trajectory"]) if rec.get("trajectory") else None
        v = judge_with_llm(judge, tasks[tid], rec["final_state"], trajectory=traj,
                           threshold=a.threshold)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                logger.info("%d/%d", done[0], len(jobs))
        return tid, v

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one, jobs))

    scored: dict[str, list] = collections.defaultdict(list)
    errors = 0
    for tid, v in results:
        if v.error:
            errors += 1
            continue
        scored[tid].append(v)

    # A task whose every rollout failed to judge has no rate; leaving the stale
    # checker rate in place would be worse than leaving it unscreened, because
    # `draw` cannot tell the two apart.
    rates = {tid: sum(v.passed for v in vs) / len(vs) for tid, vs in scored.items() if vs}
    # Must be byte-identical to what screen_task_difficulty.py computes for the
    # baseline arm, or `draw(screened_by=...)` treats every rate as someone
    # else's measurement and the eval set comes back empty.
    screened_by = fingerprint(Playbook(version=1), a.agent_model)
    logger.info("judge errors: %d/%d rollouts; %d tasks have a new rate; screened_by=%s",
                errors, len(jobs), len(rates), screened_by)

    if not a.dry_run:
        written = sum(bank.set_baseline_pass_rate(tid, r, screened_by=screened_by)
                      for tid, r in rates.items())
        logger.info("wrote %d/%d rates back into the bank", written, len(rates))

    band = [t for t, r in rates.items() if a.band_low <= r <= a.band_high]
    strictly = [t for t, r in rates.items() if 0.0 < r < 1.0]
    print(f"\n  tasks re-judged      : {len(rates)}")
    print(f"  pinned at 1.00       : {sum(1 for r in rates.values() if r == 1.0)}")
    print(f"  pinned at 0.00       : {sum(1 for r in rates.values() if r == 0.0)}")
    print(f"  strictly between     : {len(strictly)}")
    print(f"  in band [{a.band_low}, {a.band_high}] : {len(band)}   <-- what the eval set draws from")
    print(f"  mean rate            : {sum(rates.values()) / max(len(rates), 1):.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"threshold": a.threshold, "screened_by": screened_by, "judge_errors": errors,
         "rates": rates}, ensure_ascii=False, indent=2))
    logger.info("wrote %s", a.out)


if __name__ == "__main__":
    main()

