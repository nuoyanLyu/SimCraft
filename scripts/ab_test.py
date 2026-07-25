"""Fixed-difficulty A/B evaluation of playbook capability gain.

Holds task difficulty and the task set FIXED; the only variable is which
playbook checkpoint is injected into the agent's system prompt. This isolates
the causal effect of the self-evolved playbook on agent capability, which the
escalating-curriculum evolve loop (live_smoke_real_sim.py) cannot show because
it moves difficulty at the same time.

Design
------
1. Freeze a held-out eval set of N tasks at a single graph_complexity (none of
   these tasks were seen during the evolve run that produced the playbook).
2. For each playbook checkpoint (v1 empty baseline -> ... -> final evolved) run
   every task R independent times (R = replications, for a stability estimate)
   and judge with the step-wise-aware checker.
3. Report pass_rate per checkpoint, per-replication (each replication is a full
   independent pass over the eval set), and the paired per-task win/loss of the
   final checkpoint vs the empty baseline.

Usage:
    PYTHONPATH=. python ab_test.py --n-tasks 16 --reps 3 --graph-complexity 3 \
        --out-dir abtest/run1 [--reuse-tasks abtest/run1/eval_tasks.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from qwen_agentworld.core.schemas import Playbook, Task
from qwen_agentworld.judge.paired_audit import judge_checker
from qwen_agentworld.llm_clients.agent_qwen import AgentClient
from qwen_agentworld.llm_clients.simulator_qwen_aw import SimulatorClient
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.simulator_gym.env import rollout
from qwen_agentworld.teacher.task_generator import generate_task

import live_smoke_real_sim as smoke  # reuse the exact NOTES_TOOLS / TOOL_FAMILY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ab_test")

NOTES_TOOLS = smoke.NOTES_TOOLS
TOOL_FAMILY = smoke.TOOL_FAMILY


def build_eval_tasks(teacher, n_tasks, gc, out_path):
    tasks = []
    for i in range(n_tasks):
        for attempt in range(1, 4):
            try:
                t = generate_task(teacher, NOTES_TOOLS, TOOL_FAMILY, min_nodes=gc, max_nodes=gc)
                tasks.append(t)
                logger.info("generated eval task %d/%d (%d steps) sw=%s",
                            i + 1, n_tasks, len(t.task_graph.nodes),
                            t.checker.step_wise_diagnostics)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("gen task %d attempt %d failed: %s", i + 1, attempt, exc)
        else:
            logger.warning("skipping eval task %d after 3 failures", i + 1)
    out_path.write_text(json.dumps([t.model_dump(mode="json") for t in tasks], ensure_ascii=False, indent=2))
    logger.info("froze %d eval tasks -> %s", len(tasks), out_path)
    return tasks


def load_eval_tasks(path):
    raw = json.loads(Path(path).read_text())
    tasks = [Task.model_validate(t) for t in raw]
    logger.info("loaded %d frozen eval tasks from %s", len(tasks), path)
    return tasks


def load_checkpoints(bigrun_dir, spec):
    """spec: list of (label, iteration_file_or_None). None => empty v1 baseline."""
    ckpts = []
    for label, itfile in spec:
        if itfile is None:
            ckpts.append((label, Playbook(version=1)))
        else:
            d = json.loads((Path(bigrun_dir) / itfile).read_text())
            pb = Playbook.model_validate(d["playbook_after"])
            ckpts.append((label, pb))
    for label, pb in ckpts:
        logger.info("checkpoint %s: playbook v%s, %d modules", label, pb.version, len(pb.modules))
    return ckpts


def judge_with_states(task, trajectory, final_state):
    states = [task.initial_state] + [
        (step.simulator_raw_output or {}).get("next_state", {}) for step in trajectory.steps
    ]
    return judge_checker(task.checker, final_state, states=states)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = TeacherClient()
    simulator = SimulatorClient()
    agent = AgentClient(model=args.agent_model)

    tasks_path = out_dir / "eval_tasks.json"
    if args.reuse_tasks:
        tasks = load_eval_tasks(args.reuse_tasks)
    elif tasks_path.exists():
        tasks = load_eval_tasks(tasks_path)
    else:
        tasks = build_eval_tasks(teacher, args.n_tasks, args.graph_complexity, tasks_path)

    ckpt_spec = [
        ("v1_empty", None),
        ("v3_mid", "iteration_2.json"),
        ("v6_final", "iteration_4.json"),
    ]
    checkpoints = load_checkpoints(args.bigrun_dir, ckpt_spec)

    results = {
        "started_at": time.time(),
        "config": {"n_tasks": len(tasks), "reps": args.reps, "graph_complexity": args.graph_complexity,
                   "agent_model": args.agent_model, "checkpoints": [c[0] for c in checkpoints]},
        "checkpoints": {},
    }

    for label, pb in checkpoints:
        logger.info("===== checkpoint %s (v%s, %d modules) =====", label, pb.version, len(pb.modules))
        # per_task_pass[task_id] = list of bool over reps
        per_task = {t.task_id: [] for t in tasks}
        rep_pass_rates = []
        for rep in range(args.reps):
            passes = 0
            counted = 0
            for t in tasks:
                try:
                    traj, final_state = rollout(agent, simulator, t, NOTES_TOOLS, playbook=pb)
                    verdict = judge_with_states(t, traj, final_state)
                except Exception as exc:  # noqa: BLE001 — isolate one rollout failure
                    logger.warning("rollout failed (ckpt %s rep %d task %s): %s", label, rep, t.task_id, exc)
                    verdict = None
                per_task[t.task_id].append(verdict)
                if verdict is not None:
                    counted += 1
                    passes += 1 if verdict else 0
            rate = passes / counted if counted else None
            rep_pass_rates.append(rate)
            logger.info("ckpt %s rep %d/%d: pass_rate=%.3f (%d/%d)",
                        label, rep + 1, args.reps, rate if rate is not None else -1, passes, counted)
        # aggregate
        all_verdicts = [v for vs in per_task.values() for v in vs if v is not None]
        overall = sum(1 for v in all_verdicts if v) / len(all_verdicts) if all_verdicts else None
        results["checkpoints"][label] = {
            "playbook_version": pb.version,
            "n_modules": len(pb.modules),
            "rep_pass_rates": rep_pass_rates,
            "overall_pass_rate": overall,
            "per_task": per_task,
        }
        (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        logger.info("ckpt %s overall pass_rate=%.3f", label, overall if overall is not None else -1)

    results["finished_at"] = time.time()

    # paired gain: final vs baseline, per task (mean over reps)
    base = results["checkpoints"]["v1_empty"]["per_task"]
    fin = results["checkpoints"]["v6_final"]["per_task"]
    def mean(vs):
        vs = [v for v in vs if v is not None]
        return sum(1 for v in vs if v) / len(vs) if vs else None
    paired = []
    wins = losses = ties = 0
    for tid in base:
        b, f = mean(base[tid]), mean(fin.get(tid, []))
        if b is None or f is None:
            continue
        paired.append({"task_id": tid, "base": b, "final": f, "delta": f - b})
        if f > b: wins += 1
        elif f < b: losses += 1
        else: ties += 1
    results["paired_final_vs_baseline"] = {"wins": wins, "losses": losses, "ties": ties, "per_task": paired}
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # summary print
    print("\n================ A/B SUMMARY ================")
    for label, _ in checkpoints:
        c = results["checkpoints"][label]
        print(f"{label:10s} v{c['playbook_version']} mods={c['n_modules']} "
              f"overall={c['overall_pass_rate']:.3f} reps={[round(r,3) if r is not None else None for r in c['rep_pass_rates']]}")
    gv = results["checkpoints"]["v6_final"]["overall_pass_rate"]
    bv = results["checkpoints"]["v1_empty"]["overall_pass_rate"]
    print(f"\nGAIN v6_final - v1_empty = {gv - bv:+.3f}")
    print(f"Paired per-task (mean over reps): wins={wins} losses={losses} ties={ties}")
    print("=============================================\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=16)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--graph-complexity", type=int, default=3)
    p.add_argument("--agent-model", default="Qwen3-8B")
    p.add_argument("--bigrun-dir", default="smoke_test_results/bigrun_0722_1537")
    p.add_argument("--reuse-tasks", default=None)
    p.add_argument("--out-dir", default="abtest/run1")
    run(p.parse_args())
