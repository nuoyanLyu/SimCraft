"""Live end-to-end smoke test of the closed loop, against real backends:
Teacher = Claude Sonnet 5 (AUTODL relay), Simulator = Claude Sonnet 5
(temporary stand-in, same relay), Agent = Qwen3-14B served locally via vLLM
(AGENT_URL env var must already point at it).

Not a permanent part of the framework — a toy "notes" tool family exists
only here, since no real tool domain has been committed to yet. Purpose is
twofold: (1) verify the wired-up pipeline actually runs against real
backends end-to-end, not just mocks, and (2) persist every intermediate
artifact (task, trajectory, evidence, diagnosis, playbook deltas) so a human
can later judge Teacher/Simulator/Agent quality from the raw transcripts
instead of a pass/fail number.

Usage (run from the qwen_agentworld repo root, with the target vLLM server
already reachable at $AGENT_URL):
    AGENT_BASE_URL=http://localhost:8001/v1 AGENT_MODEL=Qwen3-8B \\
        ~/anaconda3/envs/simcraft/bin/python scripts/live_smoke_test.py \\
        --iterations 2 --tasks-per-iteration 2 --output-dir smoke_test_results/run1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agentworld.capability_probe.prober import RollingPassRateTracker
from qwen_agentworld.core.schemas import ToolFunctionSpec, ToolSpec
from qwen_agentworld.evidence_gate.gate import EvidenceGate
from qwen_agentworld.judge.paired_audit import judge_checker
from qwen_agentworld.llm_clients.agent_qwen3 import AgentClient
from qwen_agentworld.llm_clients.simulator_temp_claude import TemporarySimulatorClient
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.optimizer import build_optimizer
from qwen_agentworld.orchestrator import loop as loop_module
from qwen_agentworld.playbook_store.leak_audit import forbidden_terms_from_tools
from qwen_agentworld.orchestrator.loop import stop_criterion
from qwen_agentworld.orchestrator.validation import validate_and_maybe_rollback
from qwen_agentworld.playbook_store.store import PlaybookStore
from qwen_agentworld.teacher.task_bank import TaskBank
from qwen_agentworld.core.schemas import Playbook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("live_smoke_test")

TOOL_FAMILY = "mcp_notes"

NOTES_TOOLS = [
    ToolSpec(
        function=ToolFunctionSpec(
            name="write_note",
            description="Create a new note with a title and text content.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                "required": ["title", "content"],
            },
        ),
        family=TOOL_FAMILY,
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="update_note",
            description="Replace the content of an existing note, identified by title.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                "required": ["title", "content"],
            },
        ),
        family=TOOL_FAMILY,
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="delete_note",
            description="Delete a note by title.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        ),
        family=TOOL_FAMILY,
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="search_notes",
            description="Return titles of notes whose title or content contains the query string.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        family=TOOL_FAMILY,
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="tag_note",
            description="Attach a tag string to an existing note, identified by title.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "tag": {"type": "string"}},
                "required": ["title", "tag"],
            },
        ),
        family=TOOL_FAMILY,
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="list_notes",
            description="List the titles of all existing notes.",
            parameters={"type": "object", "properties": {}},
        ),
        family=TOOL_FAMILY,
    ),
]


def _dump(model) -> dict:
    return model.model_dump(mode="json")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = TeacherClient()
    simulator = TemporarySimulatorClient()
    agent = AgentClient(model=args.agent_model)
    # Logged after construction: every role's model is env-configurable
    # (--agent-model defaults to None so $AGENT_MODEL wins), so only the
    # constructed client knows what actually resolved.
    logger.info(
        "building clients: teacher=%s, simulator=%s(temp), agent=%s",
        teacher.model,
        simulator.model,
        agent.model,
    )

    # Arm the leak audit: a module that names a tool it was evolved on is
    # memorisation, not a transferable meta-skill, and must not be stored.
    playbook_store = PlaybookStore(forbidden_terms=forbidden_terms_from_tools(NOTES_TOOLS))
    playbook_store.seed(Playbook(version=1))

    # Held-out tasks for validation. Drawn from the bank's eval split so they are
    # never the tasks the playbook was evolved on: scoring a playbook on the
    # rollouts that produced it measures memorisation, not utility.
    heldout_tasks = (
        TaskBank(args.bank_dir).draw(
            TOOL_FAMILY, args.graph_complexity, args.validation_tasks, split="val"
        )
        if args.validation_tasks > 0
        else []
    )
    if args.validation_tasks > 0 and not heldout_tasks:
        logger.warning(
            "no eval-split tasks in the bank for %s/gc%d — running without validation, "
            "so no mutation can be rejected",
            TOOL_FAMILY,
            args.graph_complexity,
        )
    tracker = RollingPassRateTracker()
    gate = EvidenceGate()
    optimizer = build_optimizer(teacher)  # U1 default: TextGrad

    config = loop_module.OrchestratorConfig(
        tool_family=TOOL_FAMILY,
        tasks_per_iteration=args.tasks_per_iteration,
        graph_complexity=args.graph_complexity,
    )

    run_summary = {
        "started_at": time.time(),
        "teacher_model": teacher.model,
        "simulator_model": simulator.model,
        "agent_model": agent.model,
        "iterations_requested": args.iterations,
        "iterations_completed": 0,
        "iteration_errors": [],
    }

    for i in range(args.iterations):
        logger.info("=== iteration %d/%d ===", i + 1, args.iterations)
        captured_trajectories: dict[str, tuple] = {}
        captured_diagnoses: list[tuple[str, dict]] = []

        orig_rollout = loop_module.rollout
        orig_diagnose = loop_module.diagnose

        def _wrapped_rollout(agent_, simulator_, task, tools, playbook=None, max_steps=10, _orig=orig_rollout):
            trajectory, final_state = _orig(agent_, simulator_, task, tools, playbook=playbook, max_steps=max_steps)
            captured_trajectories[task.task_id] = (trajectory, final_state)
            return trajectory, final_state

        # Signature must track `loop._process_task`'s call, which passes the
        # current playbook so the teacher can credit/retire existing entries.
        # A stale signature here does not fail loudly: the TypeError is caught
        # by run_iteration's per-task guard, every task is silently skipped,
        # and the run finishes reporting zero playbook edits.
        def _wrapped_diagnose(teacher_, trajectory, checker_passed, playbook=None, _orig=orig_diagnose, **kwargs):
            diagnosis = _orig(teacher_, trajectory, checker_passed, playbook=playbook, **kwargs)
            captured_diagnoses.append((trajectory.task_id, diagnosis))
            return diagnosis

        loop_module.rollout = _wrapped_rollout
        loop_module.diagnose = _wrapped_diagnose

        playbook_version_before = playbook_store.current.version
        try:
            tasks = loop_module.run_iteration(
                teacher=teacher,
                agent=agent,
                simulator=simulator,
                tools=NOTES_TOOLS,
                playbook_store=playbook_store,
                optimizer=optimizer,
                gate=gate,
                tracker=tracker,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 — smoke test must not die on one bad iteration
            logger.exception("iteration %d failed", i + 1)
            run_summary["iteration_errors"].append(
                {"iteration": i + 1, "error": str(exc), "traceback": traceback.format_exc()}
            )
            loop_module.rollout = orig_rollout
            loop_module.diagnose = orig_diagnose
            continue
        finally:
            loop_module.rollout = orig_rollout
            loop_module.diagnose = orig_diagnose

        diagnosis_by_task = dict(captured_diagnoses)
        task_records = []
        for task in tasks:
            entry = captured_trajectories.get(task.task_id)
            trajectory, final_state = entry if entry else (None, None)
            # Rebuild the same ordered state list `loop._process_task` used, or a
            # step-wise checker silently falls back to the end-state predicate
            # and every reversible task gets persisted as a failure.
            states = (
                [task.initial_state]
                + [(s.simulator_raw_output or {}).get("next_state", {}) for s in trajectory.steps]
                if trajectory is not None
                else None
            )
            checker_passed = (
                judge_checker(task.checker, final_state, states=states, task_id=task.task_id)
                if final_state is not None
                else None
            )
            diagnosis = diagnosis_by_task.get(task.task_id)
            task_records.append(
                {
                    "task": _dump(task),
                    "trajectory": _dump(trajectory) if trajectory is not None else None,
                    "final_state": final_state,
                    "checker_passed": checker_passed,
                    "diagnosis": _dump(diagnosis) if diagnosis is not None else None,
                }
            )
            logger.info(
                "task %s: %d steps, checker_passed=%s",
                task.task_id,
                len(trajectory.steps) if trajectory else -1,
                checker_passed,
            )

        iteration_record = {
            "iteration": i + 1,
            "config": {
                "graph_complexity": config.graph_complexity,
                "tasks_per_iteration": config.tasks_per_iteration,
            },
            "tasks": task_records,
            "tracker_results": [
                {
                    "tool_family": r.tool_family,
                    "graph_complexity": r.graph_complexity,
                    "n_samples": r.n_samples,
                    "pass_rate": r.pass_rate,
                    "in_band": r.in_band,
                }
                for r in tracker.all_results()
            ],
            "playbook_version_before": playbook_version_before,
            "playbook_version_after": playbook_store.current.version,
            "playbook_after": _dump(playbook_store.current),
        }
        out_path = output_dir / f"iteration_{i + 1}.json"
        out_path.write_text(json.dumps(iteration_record, indent=2, ensure_ascii=False))
        logger.info("wrote %s", out_path)
        run_summary["iterations_completed"] += 1

        # U7: score the (possibly just-mutated) playbook on held-out tasks and
        # revert if it regressed. Without this the loop accepts every proposal
        # unconditionally — a harmful edit is indistinguishable from a good one.
        if heldout_tasks:
            result, rolled_back = validate_and_maybe_rollback(
                playbook_store,
                agent,
                simulator,
                NOTES_TOOLS,
                heldout_tasks,
                reps=args.validation_reps,
                tolerance=args.validation_tolerance,
            )
            logger.info(
                "validation: utility=%.3f (%d/%d, %d errored) rolled_back=%s",
                result.utility,
                result.n_passed,
                result.n_rollouts,
                result.n_errored,
                rolled_back,
            )
            iteration_record["validation"] = {
                "utility": result.utility,
                "n_passed": result.n_passed,
                "n_rollouts": result.n_rollouts,
                "n_errored": result.n_errored,
                "rolled_back": rolled_back,
            }
            out_path.write_text(json.dumps(iteration_record, indent=2, ensure_ascii=False))

            if stop_criterion(playbook_store.history):
                logger.info("stop criterion fired: validation utility stopped improving")
                run_summary["stopped_early"] = True
                break

    run_summary["finished_at"] = time.time()
    run_summary["playbook_history_versions"] = [p.version for p in playbook_store.history]
    (output_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False))
    logger.info("smoke test done: %s", run_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--tasks-per-iteration", type=int, default=2)
    parser.add_argument("--graph-complexity", type=int, default=2)
    # None -> AgentClient falls back to $AGENT_MODEL, then to Qwen3-8B.
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--output-dir", default="smoke_test_results/run1")
    parser.add_argument(
        "--validation-tasks",
        type=int,
        default=0,
        help="held-out eval-split tasks to score the playbook on each iteration (0 = off)",
    )
    parser.add_argument("--validation-reps", type=int, default=1)
    parser.add_argument(
        "--validation-tolerance",
        type=float,
        default=0.1,
        help="regression absorbed before rolling back; 0 reverts on sampling noise alone",
    )
    parser.add_argument("--bank-dir", default="task_bank")
    run(parser.parse_args())
