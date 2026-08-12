"""Measure each banked task's baseline pass rate, so the A/B can be run on
tasks that are actually able to show an effect.

The 2026-07-28 A/B was a null result partly because 11 of its 12 tasks were
pinned: 7 passed in every rollout of every arm and 4 failed in every rollout of
every arm. The two ends are not symmetric, though, and calling them both
"unmeasurable" (as this file did) was a mistake worth naming. A task at 1.0 has
genuinely no room to improve. A task at 0.0 has the *most* room -- as long as it
is passable in principle. Screening cannot tell a hard task from an impossible
one, because both look like 0.0; `audit_task_quality.py` can, and it is
GPU-free. So the workflow is: screen, then audit the zeros, then treat the
passable zeros as prime A/B material rather than discarding them. Screening moves that filter
before the experiment: run the *baseline* agent (empty playbook v1) R times per
task, record the pass rate on the bank entry, and let `ab_test --screened-only`
draw from the discriminative band.

Note the asymmetry with the quality audit (`audit_task_quality.py`): the audit
asks the teacher whether a task is *scorable in principle* and needs no GPU;
this asks the agent how hard it is *in fact* and needs the served models up.
Run the audit first — screening an unpassable task just burns rollouts to
rediscover that it is a permanent 0.

Usage:
    python scripts/screen_task_difficulty.py --reps 5 --graph-complexity 3
    python scripts/screen_task_difficulty.py --tasks abtest/run/eval_tasks.json --reps 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_agentworld.capability_probe.prober import DEFAULT_DIFFICULTY_BAND
from qwen_agentworld.core.schemas import Playbook, Task
from qwen_agentworld.judge.llm_judge import DEFAULT_JUDGE_THRESHOLD
from qwen_agentworld.judge.verdict import JUDGE_MODES, MODE_CHECKER, JudgeConfig, judge_rollout
from qwen_agentworld.llm_clients.agent_qwen3 import AgentClient
from qwen_agentworld.llm_clients.simulator_qwen_aw import SimulatorClient
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.playbook_store.store import fingerprint
from qwen_agentworld.simulator_gym.env import rollout
from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, SPLIT_TRAIN, SPLIT_VAL, TaskBank
from qwen_agentworld.tools.graph_map import graph_complexity_for

import live_smoke_real_sim as smoke

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("screen_difficulty")

NOTES_TOOLS = smoke.NOTES_TOOLS
TOOL_FAMILY = smoke.TOOL_FAMILY


def screen_one(agent, simulator, task: Task, reps: int, traj_f, lock=None,
               playbook: Playbook | None = None, judge: JudgeConfig | None = None) -> float:
    """Pass rate over `reps` rollouts, against an empty playbook by default.

    `playbook` screens against an evolved agent instead. That is the same
    measurement, not a different one: difficulty is defined relative to whoever
    is being asked, so re-running this with the current playbook between rounds
    is how the band stays honest as the agent improves.
    """
    baseline = playbook or Playbook(version=1)
    passes = 0
    counted = 0
    for rep in range(reps):
        traj = final_state = None
        error = None
        judged = None
        try:
            traj, final_state = rollout(agent, simulator, task, NOTES_TOOLS, playbook=baseline)
            judged = judge_rollout(task, final_state, traj, judge)
            verdict = judged.passed
        except Exception as exc:  # noqa: BLE001 — one bad rollout must not sink the sweep
            logger.warning("rollout failed (task %s rep %d): %s", task.task_id, rep, exc)
            verdict = None
            error = f"{type(exc).__name__}: {exc}"
        if verdict is not None:
            counted += 1
            passes += int(verdict)
        if traj_f is not None:
            line = json.dumps({
                "task_id": task.task_id, "rep": rep, "verdict": verdict, "error": error,
                **(judged.record() if judged is not None else {}),
                "natural_language_prompt": task.natural_language_prompt,
                "initial_state": task.initial_state, "final_state": final_state,
                "executable_predicate": task.checker.executable_predicate,
                "trajectory": traj.model_dump(mode="json") if traj is not None else None,
            }, ensure_ascii=False) + "\n"
            if lock is not None:
                with lock:
                    traj_f.write(line)
                    traj_f.flush()
            else:
                traj_f.write(line)
                traj_f.flush()
    # A task whose every rollout errored has no measured rate. Returning 0.0
    # would mislabel it "hard"; leave it unscreened so a band draw skips it.
    return passes / counted if counted else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="screen this eval_tasks.json instead of the bank")
    ap.add_argument("--bank-dir", default="task_bank")
    ap.add_argument("--split", default=SPLIT_EVAL, choices=[SPLIT_TRAIN, SPLIT_VAL, SPLIT_EVAL])
    ap.add_argument("--graph-complexity", type=int, default=None,
                    help="default: whatever tools/tool_graph_map.json records for --agent-model")
    ap.add_argument("--max-tasks", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--agent-model", default="Qwen3-8B")  # matches --served-model-name in serve_agent.sh
    # One source of truth for the curriculum band: prober.DEFAULT_DIFFICULTY_BAND.
    # This script used to hardcode (0.2, 0.8) while the orchestrator ran on
    # (0.2, 0.6), so "in band" here and "in band" during training named
    # different task sets.
    ap.add_argument("--band-low", type=float, default=DEFAULT_DIFFICULTY_BAND[0])
    ap.add_argument("--band-high", type=float, default=DEFAULT_DIFFICULTY_BAND[1])
    ap.add_argument("--workers", type=int, default=1,
                    help="rollouts in flight at once; independent, so this is pure wall-clock")
    ap.add_argument("--rescreen", action="store_true", help="also re-measure tasks already screened")
    ap.add_argument("--playbook", default=None,
                    help="screen against this playbook JSON instead of an empty one. Rates are "
                         "stored under its fingerprint, so an evolved agent's measurements never "
                         "overwrite the baseline's — they sit alongside as a separate reading.")
    ap.add_argument("--out-dir", default="abtest/screening")
    ap.add_argument("--judge-mode", default=MODE_CHECKER, choices=list(JUDGE_MODES),
                    help="checker=executable predicate (default); llm=LLM score, thresholded; "
                         "both=compute both, checker still decides, LLM score recorded alongside")
    ap.add_argument("--judge-threshold", type=float, default=DEFAULT_JUDGE_THRESHOLD)
    a = ap.parse_args()

    # Resolved once and logged: a screening run whose node count is implicit is
    # not reproducible, and the rate it writes back is only meaningful for the
    # bucket it was measured in.
    if a.graph_complexity is None:
        a.graph_complexity = graph_complexity_for(a.agent_model, TOOL_FAMILY)
        logger.info("graph_complexity=%d resolved from tool_graph_map for agent %s",
                    a.graph_complexity, a.agent_model)

    judge = JudgeConfig(mode=a.judge_mode, threshold=a.judge_threshold,
                        client=TeacherClient() if a.judge_mode != MODE_CHECKER else None)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = TaskBank(a.bank_dir)

    playbook = Playbook.model_validate(json.loads(Path(a.playbook).read_text())) if a.playbook else Playbook(version=1)
    screened_by = fingerprint(playbook, a.agent_model)

    if a.tasks:
        tasks = [Task.model_validate(t) for t in json.loads(Path(a.tasks).read_text())]
    else:
        tasks = bank.draw(TOOL_FAMILY, a.graph_complexity, a.max_tasks, split=a.split)
        if not a.rescreen:
            # A band of the full range with require_screened is exactly "every
            # task that already carries a measurement"; skip those. Scoped to
            # this agent, so a task screened against a previous playbook counts
            # as unscreened and gets re-measured — that re-measurement is the
            # mechanism, not an inefficiency to optimise away.
            screened = {t.task_id for t in bank.draw(
                TOOL_FAMILY, a.graph_complexity, a.max_tasks, split=a.split,
                band=(0.0, 1.0), require_screened=True, screened_by=screened_by)}
            tasks = [t for t in tasks if t.task_id not in screened]
    logger.info("screening %d tasks x %d reps (agent=%s playbook=%s judge=%s)",
                len(tasks), a.reps, a.agent_model, screened_by, a.judge_mode)
    if not tasks:
        return

    agent = AgentClient(model=a.agent_model)
    simulator = SimulatorClient()

    rows = []
    write_lock = threading.Lock()
    traj_path = out_dir / "screening.jsonl"
    with traj_path.open("a", encoding="utf-8") as traj_f:
        # `traj_f` is written from several threads; screen_one takes the lock
        # around each line so records never interleave mid-JSON.
        def screen_task(t):
            return t, screen_one(agent, simulator, t, a.reps, traj_f, lock=write_lock,
                                 playbook=playbook, judge=judge)

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            done = pool.map(screen_task, tasks)
            for i, (t, rate) in enumerate(done, 1):
                if rate is not None:
                    # Persist per task, not at the end: screening is the expensive
                    # part and a crash at task 30 should not discard tasks 1..29.
                    bank.set_baseline_pass_rate(t.task_id, rate, screened_by=screened_by)
                in_band = rate is not None and a.band_low <= rate <= a.band_high
                rows.append({"task_id": t.task_id, "baseline_pass_rate": rate, "in_band": in_band,
                             "natural_language_prompt": t.natural_language_prompt})
                logger.info("%d/%d %s rate=%s %s", i, len(tasks), t.task_id,
                            "n/a" if rate is None else f"{rate:.2f}", "IN-BAND" if in_band else "")
                (out_dir / "screening.json").write_text(json.dumps(
                    {"updated_at": time.time(), "reps": a.reps,
                     "band": [a.band_low, a.band_high], "rows": rows},
                    ensure_ascii=False, indent=2))

    measured = [r for r in rows if r["baseline_pass_rate"] is not None]
    in_band = [r for r in measured if r["in_band"]]
    floor = [r for r in measured if r["baseline_pass_rate"] == 0.0]
    ceil = [r for r in measured if r["baseline_pass_rate"] == 1.0]
    print("\n============ DIFFICULTY SCREENING ============")
    print(f"screened      : {len(measured)}/{len(rows)} tasks x {a.reps} reps")
    print(f"in band       : {len(in_band)}  ({a.band_low}-{a.band_high}, usable for the A/B)")
    print(f"pinned at 0.0 : {len(floor)}  (baseline never passes — full headroom IF passable; "
          f"audit these with audit_task_quality.py before discarding)")
    print(f"pinned at 1.0 : {len(ceil)}  (baseline always passes — at ceiling)")
    print("==============================================\n")
    logger.info("wrote %s and %s", out_dir / "screening.json", traj_path)


if __name__ == "__main__":
    main()
