"""The closed loop (data/code-architecture-plan.md §4): probe pass rate ->
generate a task batch at the resulting difficulty -> roll out in the
simulator -> evidence-gate every step -> checker-judge the outcome ->
diagnose accepted trajectories -> optimizer.propose/select -> playbook_store
update, repeated until `stop_criterion` fires.

Every external capability (Teacher/Agent/Simulator LLMClient, the optimizer
engine, the tools list) is passed in rather than constructed here, so U1
(optimizer engine) and U2 (agent backend) are config choices made by the
caller, never a reason to touch this file.

Deliberately NOT embedded here: `judge.paired_audit`'s with/without-playbook
comparison. It requires running each task twice (double simulator+agent
cost), so it belongs in a periodic/selective audit pass over promoted
playbooks, not the per-task hot loop — see judge/paired_audit.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qwen_agentworld.capability_probe.prober import RollingPassRateTracker, suggest_difficulty_adjustment
from qwen_agentworld.core.schemas import Playbook, Task, ToolSpec, Trajectory
from qwen_agentworld.evidence_gate.adjudication import adjudicate
from qwen_agentworld.evidence_gate.counterfactual_replay import build_counterfactual_probe
from qwen_agentworld.evidence_gate.gate import EvidenceGate
from qwen_agentworld.judge.paired_audit import judge_checker
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.playbook_store.store import PlaybookStore
from qwen_agentworld.simulator_gym.env import rollout, simulate_next_state
from qwen_agentworld.teacher.reflection import diagnose
from qwen_agentworld.teacher.task_generator import generate_task

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    tool_family: str
    tasks_per_iteration: int = 4
    n_agreement_samples: int = 3
    graph_complexity: int = 2
    min_graph_complexity: int = 2
    max_graph_complexity: int = 4
    task_generation_attempts: int = 2


def _adjust_difficulty(tracker: RollingPassRateTracker, config: OrchestratorConfig) -> None:
    """Curriculum step: nudge `config.graph_complexity` toward the tracker's
    target pass-rate band using last iteration's recorded outcomes for the
    bucket at the *current* complexity, before this iteration's tasks are
    generated at that (possibly adjusted) complexity.
    """
    result = tracker.result(config.tool_family, config.graph_complexity)
    delta = suggest_difficulty_adjustment(result, band=tracker.band)
    config.graph_complexity = min(
        config.max_graph_complexity, max(config.min_graph_complexity, config.graph_complexity + delta)
    )


def score_trajectory(
    gate: EvidenceGate,
    simulator: LLMClient,
    trajectory: Trajectory,
    n_agreement_samples: int = 3,
    adjudicator: LLMClient | None = None,
) -> None:
    """Mutates `trajectory.steps[*].evidence` / `.accepted` in place.

    `adjudicator` is optional and, when given, should be a *separate*
    LLMClient instance from whichever teacher generated this task/checker
    (evidence_gate/adjudication.py) — passing the same instance that set the
    question would let it grade its own homework. Adjudication only fires
    for steps the confidence-band leaves ambiguous (`gate.needs_adjudication`);
    omitting `adjudicator` just leaves those steps at their blended score.
    """
    for step in trajectory.steps:
        raw = step.simulator_raw_output or {}
        prior_state, next_state = raw.get("prior_state", {}), raw.get("next_state", {})
        agreement_samples = [next_state] + [
            simulate_next_state(simulator, prior_state, step.tool_call)
            for _ in range(max(0, n_agreement_samples - 1))
        ]

        counterfactual_output, invariant_fields = None, None
        probe = build_counterfactual_probe(prior_state, next_state)
        if probe is not None:
            perturbed_prior_state, invariant_fields = probe
            counterfactual_output = simulate_next_state(simulator, perturbed_prior_state, step.tool_call)

        evidence = gate.score(
            candidate_output=next_state,
            response_schema=None,
            agreement_samples=agreement_samples,
            counterfactual_output=counterfactual_output,
            invariant_fields=invariant_fields,
        )

        if adjudicator is not None and gate.needs_adjudication(evidence):
            verdict = adjudicate(adjudicator, prior_state, step.tool_call.tool_name, step.tool_call.arguments, next_state)
            evidence = gate.score(
                candidate_output=next_state,
                response_schema=None,
                agreement_samples=agreement_samples,
                counterfactual_output=counterfactual_output,
                invariant_fields=invariant_fields,
                adjudicated=verdict,
            )

        step.evidence = evidence
        step.accepted = gate.accept(evidence)


def trajectory_evidence_accepted(trajectory: Trajectory) -> bool:
    """Whole-trajectory evidence gate: every step's simulated transition has
    to clear the gate, since a diagnosis/optimizer mutation downstream can't
    tell which single untrusted step it should discount.
    """
    return bool(trajectory.steps) and all(step.accepted for step in trajectory.steps)


def stop_criterion(playbook_history: list[Playbook], window: int = 3) -> bool:
    """U7: stop once `validation_utility` hasn't improved for `window`
    consecutive evaluated playbooks. Playbooks without a recorded
    validation_utility (never separately evaluated) don't count as evidence
    either way and are skipped rather than treated as regressions.
    """
    evaluated = [p.validation_utility for p in playbook_history if p.validation_utility is not None]
    if len(evaluated) < window + 1:
        return False
    recent = evaluated[-(window + 1) :]
    best_before_window = max(recent[:-window])
    return all(v <= best_before_window for v in recent[-window:])


def _generate_tasks_resiliently(
    teacher: LLMClient,
    tools: list[ToolSpec],
    config: OrchestratorConfig,
    effective_complexity: int,
) -> list[Task]:
    """Generate up to `config.tasks_per_iteration` tasks, retrying each a
    bounded number of times and skipping any that keep failing rather than
    aborting the iteration.

    Task generation makes three teacher calls (NL/state instantiation +
    checker synthesis), each already guarded against empty relay replies.
    But a sustained burst of empty content (observed live against the AUTODL
    relay) can still exhaust those inner retries and raise — which, left
    unguarded, killed the entire iteration and discarded every task already
    generated. Isolating failures per task lets the iteration proceed on
    whatever tasks did generate.
    """
    tasks: list[Task] = []
    for _ in range(config.tasks_per_iteration):
        for attempt in range(1, max(1, config.task_generation_attempts) + 1):
            try:
                tasks.append(
                    generate_task(
                        teacher,
                        tools,
                        config.tool_family,
                        min_nodes=effective_complexity,
                        max_nodes=effective_complexity,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001 — isolate one task's generation failure
                logger.warning(
                    "task generation attempt %d/%d failed: %s", attempt, config.task_generation_attempts, exc
                )
        else:
            logger.warning(
                "skipping a task after %d failed generation attempts", config.task_generation_attempts
            )
    return tasks


def _process_task(
    teacher: LLMClient,
    agent: LLMClient,
    simulator: LLMClient,
    tools: list[ToolSpec],
    playbook_store: PlaybookStore,
    optimizer: PlaybookOptimizer,
    gate: EvidenceGate,
    tracker: RollingPassRateTracker,
    config: OrchestratorConfig,
    task: Task,
    adjudicator: LLMClient | None = None,
) -> None:
    """Run one task through rollout -> evidence gate -> checker -> diagnosis
    -> optimizer, mutating `playbook_store`/`tracker` in place. Isolated into
    its own function so `run_iteration` can catch a single task's failure and
    keep processing the rest of the batch (Q8).
    """
    trajectory, final_state = rollout(agent, simulator, task, tools, playbook=playbook_store.current)
    score_trajectory(gate, simulator, trajectory, config.n_agreement_samples, adjudicator=adjudicator)

    # Ordered canonical states (initial + post-step) let a step-wise checker
    # verify reversible tasks whose final state equals the initial one;
    # end-state checkers ignore the extra argument.
    states = [task.initial_state] + [
        (step.simulator_raw_output or {}).get("next_state", {}) for step in trajectory.steps
    ]
    checker_passed = judge_checker(task.checker, final_state, states=states)
    tracker.record(task.tool_family, task.difficulty_meta.graph_complexity, checker_passed)

    if not trajectory_evidence_accepted(trajectory):
        return  # simulator evidence too weak to trust this trajectory for learning

    diagnosis = diagnose(teacher, trajectory, checker_passed)
    candidates = optimizer.propose(playbook_store.current, diagnosis)
    if candidates:
        playbook_store.update(optimizer.select(candidates))


def run_iteration(
    teacher: LLMClient,
    agent: LLMClient,
    simulator: LLMClient,
    tools: list[ToolSpec],
    playbook_store: PlaybookStore,
    optimizer: PlaybookOptimizer,
    gate: EvidenceGate,
    tracker: RollingPassRateTracker,
    config: OrchestratorConfig,
    adjudicator: LLMClient | None = None,
) -> list[Task]:
    """Runs one batch of tasks through the full closed loop; mutates
    `playbook_store` and `tracker` in place. `playbook_store` must already be
    seeded (`playbook_store.seed(...)`) before the first call. Returns the
    tasks generated this iteration, mainly for test/inspection purposes.
    """
    _adjust_difficulty(tracker, config)

    # `sample_task_graph` silently clamps to `len(tools)` if asked for more
    # nodes than the pool has — request no more than that up front so the
    # task's *actual* difficulty_meta.graph_complexity always lands in the
    # bucket the curriculum step just aimed for, instead of drifting into a
    # smaller bucket the tracker never adjusted toward.
    effective_complexity = min(config.graph_complexity, len(tools))

    tasks = _generate_tasks_resiliently(teacher, tools, config, effective_complexity)

    for task in tasks:
        try:
            _process_task(
                teacher,
                agent,
                simulator,
                tools,
                playbook_store,
                optimizer,
                gate,
                tracker,
                config,
                task,
                adjudicator,
            )
        except Exception as exc:  # noqa: BLE001 — one task's failure must not sink the whole batch
            logger.warning("skipping task %s after processing error: %s", task.task_id, exc)

    return tasks
