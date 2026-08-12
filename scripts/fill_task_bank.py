"""Fill the task bank with three disjoint pools, under the fixed generator.

Everything already banked was generated before the 2026-07-29 checker and
generator fixes (broadened step-wise trigger, no character-count assertions,
the `_action_log` witness for read-only steps, no destructive tasks) and is all
tagged `eval`. Mixing those into this study would mean the arms differ in task
provenance as well as in playbook, so this generates a clean set instead:

    train -- what the evolve run learns from
    val   -- what each playbook edit is accepted or rolled back against
    eval  -- never seen until the A/B

Generation is the expensive step (two teacher calls per task, ~2 min), so it
runs concurrently and appends to the bank as tasks land: a crash loses only the
in-flight tasks, not the batch.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.teacher.task_bank import TaskBank
from qwen_agentworld.teacher.task_generator import generate_task
from qwen_agentworld.tools.graph_map import graph_complexity_for

import live_smoke_real_sim as smoke

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fill_bank")


def make_one(gc: int, split: str, bank: TaskBank, run_tag: str):
    # One client per task: TeacherClient is not documented as thread-safe, and a
    # shared HTTP session is not worth debugging inside a 2-hour batch.
    teacher = TeacherClient()
    task = generate_task(teacher, smoke.NOTES_TOOLS, smoke.TOOL_FAMILY, min_nodes=gc, max_nodes=gc)
    bank.save(task, split=split, origin=f"fill_task_bank:{run_tag}", gc=gc)
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-model", default="Qwen3-8B",
                    help="whose difficulty this batch is being generated for; decides the "
                         "graph size via tools/tool_graph_map.json")
    ap.add_argument("--graph-complexity", type=int, default=None,
                    help="override the map for a one-off; the map is where a measured number goes")
    ap.add_argument("--train", type=int, default=24)
    ap.add_argument("--val", type=int, default=12)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bank-dir", default="task_bank")
    a = ap.parse_args()

    if a.graph_complexity is None:
        a.graph_complexity = graph_complexity_for(a.agent_model, smoke.TOOL_FAMILY)
        logger.info("graph_complexity=%d resolved from tool_graph_map for agent %s",
                    a.graph_complexity, a.agent_model)

    bank = TaskBank(a.bank_dir)
    run_tag = time.strftime("%m%d_%H%M")
    wanted = [("train", a.train), ("val", a.val), ("eval", a.eval)]
    jobs = [split for split, n in wanted for _ in range(n)]
    logger.info("generating %d tasks at gc=%d: %s", len(jobs), a.graph_complexity,
                {s: n for s, n in wanted})

    done = {"train": 0, "val": 0, "eval": 0}
    failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(make_one, a.graph_complexity, s, bank, run_tag): s for s in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            split = futures[fut]
            try:
                fut.result()
                done[split] += 1
            except Exception as exc:  # noqa: BLE001 - one bad task must not end the batch
                failed += 1
                logger.warning("generation failed (%s): %s: %s", split, type(exc).__name__, exc)
            if i % 5 == 0 or i == len(jobs):
                logger.info("progress %d/%d  banked=%s failed=%d", i, len(jobs), done, failed)

    logger.info("BANK FILL DONE banked=%s failed=%d", done, failed)
    logger.info("bank stats: %s", bank.stats())


if __name__ == "__main__":
    main()
