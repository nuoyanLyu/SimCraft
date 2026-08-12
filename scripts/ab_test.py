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
4. Write every rollout to trajectories/<checkpoint>.jsonl — steps, final state,
   verdict, and the instruction and predicate it was judged against. A null
   result is only useful if you can tell whether the agent failed or the
   checker was wrong, and that needs the trajectory.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from qwen_agentworld.core.schemas import Task
from qwen_agentworld.judge.llm_judge import DEFAULT_JUDGE_THRESHOLD
from qwen_agentworld.judge.verdict import JUDGE_MODES, MODE_CHECKER, JudgeConfig, judge_rollout
from qwen_agentworld.llm_clients.agent_qwen3 import AgentClient
from qwen_agentworld.llm_clients.simulator_qwen_aw import SimulatorClient
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.simulator_gym.env import rollout
from qwen_agentworld.tools.graph_map import graph_complexity_for
from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, TaskBank
from qwen_agentworld.teacher.task_generator import generate_task
from qwen_agentworld.playbook_store.store import fingerprint
from qwen_agentworld.study.checkpoints import empty_playbook, load_playbook

import live_smoke_real_sim as smoke  # reuse the exact NOTES_TOOLS / TOOL_FAMILY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ab_test")

NOTES_TOOLS = smoke.NOTES_TOOLS
TOOL_FAMILY = smoke.TOOL_FAMILY


def build_eval_tasks(teacher, n_tasks, gc, out_path, bank=None, band=None):
    """Assemble an eval set of `n_tasks`, reusing banked tasks before paying
    the teacher for new ones.

    `band` restricts the draw to tasks whose screened baseline pass rate falls
    inside it. A task at 1.0 carries no signal about the playbook: it is already
    passing and cannot pass harder. A task at 0.0 is the opposite — it has the
    whole range available — *provided* it is passable at all, which screening
    cannot tell but `audit_task_quality.py` can. So the useful band is
    (0.0, 0.8) over audit-clean tasks, not (0.2, 0.8) over everything; the
    2026-07-28 run failed because 7 of its 12 tasks were at ceiling, and
    discarding the floor as well would have left almost nothing to measure.

    For a hand-composed set (band tasks plus a ceiling sample kept as a harm
    detector) build it with `select_eval_set.py` and pass `--reuse-tasks`.
    """
    bank = bank if bank is not None else TaskBank()
    tasks = list(bank.draw(TOOL_FAMILY, gc, n_tasks, split=SPLIT_EVAL,
                           band=band, require_screened=band is not None,
                           drop_audit_failed=True))
    if tasks:
        logger.info("reused %d/%d eval tasks from the bank (band=%s)", len(tasks), n_tasks, band)

    teacher_model = getattr(teacher, "model", "")
    for i in range(len(tasks), n_tasks):
        for attempt in range(1, 4):
            try:
                t = generate_task(teacher, NOTES_TOOLS, TOOL_FAMILY, min_nodes=gc, max_nodes=gc)
                # Bank it before anything else can fail: the teacher calls are
                # already spent either way.
                bank.save(t, split=SPLIT_EVAL, origin="ab_test.build_eval_tasks",
                          teacher_model=teacher_model, gc=gc)
                tasks.append(t)
                logger.info("generated eval task %d/%d (%d steps) sw=%s",
                            i + 1, n_tasks, len(t.task_graph.nodes),
                            t.checker.step_wise_diagnostics)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("gen task %d attempt %d failed: %s", i + 1, attempt, exc)
        else:
            logger.warning("skipping eval task %d after 3 failures", i + 1)
        # Rewrite after each task so a crash leaves a usable partial set.
        out_path.write_text(
            json.dumps([t.model_dump(mode="json") for t in tasks], ensure_ascii=False, indent=2)
        )
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
            ckpts.append((label, empty_playbook()))
        else:
            # Refuses a pre-entry-redesign checkpoint loudly rather than letting
            # pydantic drop `modules` and hand back an empty playbook — see
            # study/checkpoints.py.
            ckpts.append((label, load_playbook(Path(bigrun_dir) / itfile)))
    # Two checkpoints with identical entry text are the same experiment; running
    # both spends GPU on a duplicate and invites reading the gap between them as
    # a trend. Compare content rather than version: a rolled-back edit bumps the
    # version without changing what the agent is told.
    deduped: list = []
    for label, pb in ckpts:
        fp = fingerprint(pb)
        for i, (kept_label, kept_pb, kept_fp) in enumerate(deduped):
            if kept_fp == fp:
                merged = f"{kept_label}+{label}"
                logger.info("checkpoint %s is identical to %s — collapsing into %s",
                            label, kept_label, merged)
                deduped[i] = (merged, kept_pb, kept_fp)
                break
        else:
            deduped.append((label, pb, fp))

    ckpts = [(label, pb) for label, pb, _ in deduped]
    for label, pb in ckpts:
        logger.info("checkpoint %s: playbook v%s, %d entries", label, pb.version, len(pb.entries))
    return ckpts


MAX_ROLLOUT_STEPS = 10  # keep in sync with simulator_gym.env.rollout's default


def classify_rollout_exception(exc):
    """Bucket an exception escaping `rollout` into a failure reason.

    `simulate_next_state` ends in `extract_json_object(...)["next_state"]`, so
    a simulator that replied with prose, truncated JSON, or a well-formed
    object missing the expected key surfaces here as ValueError/JSONDecodeError
    /KeyError. Those are the simulator's failures, not the agent's, and lumping
    them in with agent failures inflates the apparent task difficulty.
    """
    if isinstance(exc, KeyError) and "next_state" in str(exc):
        return "sim_missing_next_state"
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "sim_invalid_json"
    return f"rollout_error:{type(exc).__name__}"


def classify_failure(verdict, reason, trajectory, error):
    """Single failure label per rollout, most-proximate cause first.

    Without this every non-pass in a null result looks identical, and the
    triage question — is the playbook not helping, or is the harness eating
    the signal? — is unanswerable from the results file alone.
    """
    if error is not None:
        return error  # already a classify_rollout_exception label
    if verdict:
        return "pass"
    # Harness defects outrank agent behaviour: a raised predicate would have
    # scored not-passed no matter what the agent did, so attributing it to the
    # agent (out of steps, no tool call) would be wrong.
    if reason and reason != "checker_false":
        return reason
    n_steps = len(trajectory.steps) if trajectory is not None else 0
    if n_steps == 0:
        return "agent_no_tool_call"
    if n_steps >= MAX_ROLLOUT_STEPS:
        return "agent_max_steps"
    return reason or "other"


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = TeacherClient()
    simulator = SimulatorClient()
    agent = AgentClient(model=args.agent_model)
    # The judge is the same object for every arm, on purpose: an A/B whose arms
    # were scored differently measures the judge, not the playbook.
    judge_config = JudgeConfig(
        mode=args.judge_mode, threshold=args.judge_threshold,
        client=teacher if args.judge_mode != MODE_CHECKER else None)
    logger.info("judging with mode=%s threshold=%.2f", args.judge_mode, args.judge_threshold)

    if args.graph_complexity is None:
        args.graph_complexity = graph_complexity_for(args.agent_model, TOOL_FAMILY)
        logger.info("graph_complexity=%d resolved from tool_graph_map for agent %s",
                    args.graph_complexity, args.agent_model)

    tasks_path = out_dir / "eval_tasks.json"
    if args.reuse_tasks:
        tasks = load_eval_tasks(args.reuse_tasks)
        # Copy the reused set in, so the run directory alone is enough to
        # re-read what was actually evaluated.
        if Path(args.reuse_tasks).resolve() != tasks_path.resolve():
            tasks_path.write_text(
                json.dumps([t.model_dump(mode="json") for t in tasks], ensure_ascii=False, indent=2)
            )
    elif tasks_path.exists():
        tasks = load_eval_tasks(tasks_path)
    else:
        band = (args.band_low, args.band_high) if args.screened_only else None
        tasks = build_eval_tasks(teacher, args.n_tasks, args.graph_complexity, tasks_path,
                                 bank=TaskBank(args.bank_dir), band=band)

    # Labels are positional, not version-numbered: the version each checkpoint
    # actually carries is read off the playbook and reported separately.
    ckpt_spec = [
        ("baseline", None),
        ("mid", args.mid_iteration),
        ("final", args.final_iteration),
    ]
    checkpoints = load_checkpoints(args.bigrun_dir, ckpt_spec)

    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    results = {
        "started_at": time.time(),
        "config": {"n_tasks": len(tasks), "reps": args.reps, "graph_complexity": args.graph_complexity,
                   "agent_model": args.agent_model, "checkpoints": [c[0] for c in checkpoints]},
        "checkpoints": {},
    }

    for label, pb in checkpoints:
        logger.info("===== checkpoint %s (v%s, %d entries) =====", label, pb.version, len(pb.entries))
        # per_task_pass[task_id] = list of bool over reps
        per_task = {t.task_id: [] for t in tasks}
        rep_pass_rates = []
        traj_path = traj_dir / f"{label}.jsonl"
        with traj_path.open("w", encoding="utf-8") as traj_f:
            write_lock = threading.Lock()

            def one_rollout(job, label=label, pb=pb, traj_f=traj_f, lock=write_lock):
                rep, t = job
                traj = final_state = None
                error = error_reason = None
                reason = None
                judged = None
                try:
                    traj, final_state = rollout(agent, simulator, t, NOTES_TOOLS, playbook=pb)
                    judged = judge_rollout(t, final_state, traj, judge_config)
                    verdict, reason = judged.passed, judged.reason
                except Exception as exc:  # noqa: BLE001 — isolate one rollout failure
                    logger.warning("rollout failed (ckpt %s rep %d task %s): %s", label, rep, t.task_id, exc)
                    verdict = None
                    error = f"{type(exc).__name__}: {exc}"
                    error_reason = classify_rollout_exception(exc)
                failure_reason = classify_failure(verdict, reason, traj, error_reason)
                # One line per rollout. The instruction and predicate are
                # duplicated in here on purpose: attributing a verdict means
                # reading the three of them side by side, and a run should
                # not need the task file joined back in to do that.
                line = json.dumps({
                    "checkpoint": label,
                    "playbook_version": pb.version,
                    "rep": rep,
                    "task_id": t.task_id,
                    "verdict": verdict,
                    **(judged.record() if judged is not None else {}),
                    "failure_reason": failure_reason,
                    "error": error,
                    "n_steps": len(traj.steps) if traj is not None else 0,
                    "natural_language_prompt": t.natural_language_prompt,
                    "initial_state": t.initial_state,
                    "final_state": final_state,
                    "executable_predicate": t.checker.executable_predicate,
                    "step_wise_predicate": t.checker.step_wise_predicate,
                    "trajectory": traj.model_dump(mode="json") if traj is not None else None,
                }, ensure_ascii=False) + "\n"
                with lock:
                    traj_f.write(line)
                    traj_f.flush()
                return rep, t.task_id, verdict, failure_reason

            # Every (rep, task) pair is an independent rollout, so the whole
            # checkpoint runs in one pool rather than rep by rep. Per-rep pass
            # rates are recovered afterwards from the verdicts.
            jobs = [(rep, t) for rep in range(args.reps) for t in tasks]
            by_rep = {rep: [] for rep in range(args.reps)}
            reason_counts: dict[str, int] = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for rep, task_id, verdict, failure_reason in pool.map(one_rollout, jobs):
                    per_task[task_id].append(verdict)
                    by_rep[rep].append(verdict)
                    reason_counts[failure_reason] = reason_counts.get(failure_reason, 0) + 1
            for rep in range(args.reps):
                vs = [v for v in by_rep[rep] if v is not None]
                rate = sum(1 for v in vs if v) / len(vs) if vs else None
                rep_pass_rates.append(rate)
                logger.info("ckpt %s rep %d/%d: pass_rate=%.3f (%d/%d)",
                            label, rep + 1, args.reps, rate if rate is not None else -1,
                            sum(1 for v in vs if v), len(vs))
        logger.info("ckpt %s trajectories -> %s", label, traj_path)
        # aggregate
        all_verdicts = [v for vs in per_task.values() for v in vs if v is not None]
        overall = sum(1 for v in all_verdicts if v) / len(all_verdicts) if all_verdicts else None
        results["checkpoints"][label] = {
            "playbook_version": pb.version,
            "n_entries": len(pb.entries),
            "tags": pb.tags(),
            "rep_pass_rates": rep_pass_rates,
            "overall_pass_rate": overall,
            "failure_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
            "per_task": per_task,
        }
        logger.info("ckpt %s failure reasons: %s", label, results["checkpoints"][label]["failure_reasons"])
        (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        logger.info("ckpt %s overall pass_rate=%.3f", label, overall if overall is not None else -1)

    results["finished_at"] = time.time()

    # paired gain: final vs baseline, per task (mean over reps)
    base = results["checkpoints"]["baseline"]["per_task"]
    fin = results["checkpoints"]["final"]["per_task"]
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
        print(f"{label:10s} v{c['playbook_version']} entries={c['n_entries']} "
              f"overall={c['overall_pass_rate']:.3f} reps={[round(r,3) if r is not None else None for r in c['rep_pass_rates']]}")
        total = sum(c["failure_reasons"].values()) or 1
        breakdown = "  ".join(f"{k}={v} ({v/total:.0%})" for k, v in c["failure_reasons"].items())
        print(f"{'':10s} {breakdown}")
    gv = results["checkpoints"]["final"]["overall_pass_rate"]
    bv = results["checkpoints"]["baseline"]["overall_pass_rate"]
    print(f"\nGAIN final - baseline = {gv - bv:+.3f}")
    print(f"Paired per-task (mean over reps): wins={wins} losses={losses} ties={ties}")
    print("=============================================\n")


def build_parser() -> argparse.ArgumentParser:
    """See live_smoke_real_sim.build_parser: the study driver calls this stage
    with an argv list, so the flags stay in exactly one place."""
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=40)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--graph-complexity", type=int, default=None,
                   help="default: whatever tools/tool_graph_map.json records for --agent-model")
    p.add_argument("--agent-model", default="Qwen3-8B")
    p.add_argument("--bigrun-dir", default="smoke_test_results/bigrun_0722_1537")
    p.add_argument("--mid-iteration", default="iteration_2.json")
    p.add_argument("--final-iteration", default="iteration_4.json")
    p.add_argument("--reuse-tasks", default=None)
    p.add_argument("--out-dir", default="abtest/run1")
    p.add_argument("--workers", type=int, default=1,
                   help="rollouts in flight at once; independent, so this is pure wall-clock")
    p.add_argument("--bank-dir", default="task_bank")
    p.add_argument("--screened-only", action="store_true",
                   help="draw only tasks whose screened baseline pass rate is in the band")
    # Deliberately NOT prober.DEFAULT_DIFFICULTY_BAND: that band is the training
    # curriculum (serve the agent what it can just barely do), this one is an
    # eval draw filter (keep everything with headroom left). 0.0 is inside it on
    # purpose — paired with drop_audit_failed it means "hard but well-posed",
    # which is where the headroom is. Same word, two different jobs.
    p.add_argument("--judge-mode", default=MODE_CHECKER, choices=list(JUDGE_MODES),
                   help="checker=executable predicate (default); llm=LLM score, thresholded; "
                        "both=compute both, checker still decides, LLM score recorded alongside")
    p.add_argument("--judge-threshold", type=float, default=DEFAULT_JUDGE_THRESHOLD)
    p.add_argument("--band-low", type=float, default=0.0)
    p.add_argument("--band-high", type=float, default=0.8)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
