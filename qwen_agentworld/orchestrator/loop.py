"""The closed loop (data/code-architecture-plan.md §4): probe pass rate ->
generate a task batch at the resulting difficulty -> roll out in the
simulator -> (optionally evidence-gate every step) -> checker-judge the
outcome -> diagnose the trajectory -> optimizer.propose/select ->
playbook_store update, repeated until `stop_criterion` fires.

The evidence gate is off by default since 2026-08-05; see
`OrchestratorConfig.use_evidence_gate` for the measurement behind that.

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

from collections.abc import Callable

import logging
from dataclasses import dataclass, field, replace

from qwen_agentworld.capability_probe.prober import DEFAULT_DIFFICULTY_BAND, RollingPassRateTracker
from qwen_agentworld.core.schemas import Playbook, Task, ToolSpec, Trajectory
from qwen_agentworld.evidence_gate.adjudication import adjudicate
from qwen_agentworld.evidence_gate.counterfactual_replay import build_counterfactual_probe
from qwen_agentworld.evidence_gate.gate import EvidenceGate
from qwen_agentworld.judge.verdict import JudgeConfig, judge_rollout
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.playbook_store.store import PlaybookStore
from qwen_agentworld.simulator_gym.env import rollout, simulate_next_state
from qwen_agentworld.tools.graph_map import graph_complexity_for
from qwen_agentworld.tools.state_schema import StateSchema, get_schema
from qwen_agentworld.teacher.reflection import diagnose
from qwen_agentworld.teacher.task_generator import generate_task

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    tool_family: str
    tasks_per_iteration: int = 4
    n_agreement_samples: int = 3
    # How many tools a generated task's graph strings together. Still not a
    # per-task difficulty dial — see the note in prober.py — but it does move
    # the *distribution*, and where it should start is a property of the agent
    # being trained, so `None` means "look it up for `agent_model` in
    # tools/tool_graph_map.json". An explicit int overrides the map, for a
    # deliberate one-off; it is not the way to record a new model's number.
    graph_complexity: int | None = None
    agent_model: str | None = None
    # The curriculum's only knob. Difficulty is not a property of a task; it is
    # the relationship between a task and the agent reading it, so it is
    # defined as the measured pass rate and selected for with a fixed
    # acceptance band. Tasks the agent already passes carry no gradient, and
    # tasks it never passes may not be learnable at all; what is left in the
    # band is what it can just barely do. The band does not move, and it does
    # not have to: as the agent improves, tasks drift up out of the band and
    # stop being served, so the surviving pool is by definition harder than the
    # one before it. Difficulty rises without anyone defining what "harder"
    # means. The cost of this definition, and its only cost, is that the rate
    # has to be re-measured when the agent changes — see
    # `TaskBank.set_baseline_pass_rate(screened_by=...)`.
    difficulty_band: tuple[float, float] = DEFAULT_DIFFICULTY_BAND
    task_generation_attempts: int = 2
    # Off by default (downgraded 2026-08-05). The gate was built to discard
    # trajectories whose simulated transitions can't be trusted, but as wired
    # it is structurally always-pass: the self-consistency leg re-samples the
    # *same* simulator, so agreement sits at ~1.0 and confidence with it, and
    # no recorded run has had a step rejected. It therefore filtered nothing
    # while costing (n_agreement_samples - 1) + 1 extra simulator calls per
    # step — roughly 3x the rollout's simulator budget — and made every run
    # slower without changing a single learning decision.
    #
    # Turning it on is still meaningful, but only once the agreement leg draws
    # from a *heterogeneous* source (different temperature/prompt/model);
    # until then leave it off and treat the gate as an appendix component.
    use_evidence_gate: bool = False
    # How a rollout is scored. Defaults to the executable predicate, i.e. the
    # behaviour every run before 2026-08-07 had. See judge/verdict.py.
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def resolved_graph_complexity(self) -> int:
        if self.graph_complexity is not None:
            return self.graph_complexity
        return graph_complexity_for(self.agent_model, self.tool_family)


def score_trajectory(
    gate: EvidenceGate,
    simulator: LLMClient,
    trajectory: Trajectory,
    n_agreement_samples: int = 3,
    adjudicator: LLMClient | None = None,
    schema: StateSchema | None = None,
) -> None:
    """Mutates `trajectory.steps[*].evidence` / `.accepted` in place.

    `schema` must be the same one `rollout` used, or the agreement leg
    compares unlike states: the recorded `next_state` went through
    `complete_fields`, so re-simulating without the schema would score the
    simulator down for a field completion added on its behalf.

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
            simulate_next_state(simulator, prior_state, step.tool_call, schema)
            for _ in range(max(0, n_agreement_samples - 1))
        ]

        counterfactual_output, invariant_fields = None, None
        probe = build_counterfactual_probe(prior_state, next_state)
        if probe is not None:
            perturbed_prior_state, invariant_fields = probe
            counterfactual_output = simulate_next_state(
                simulator, perturbed_prior_state, step.tool_call, schema
            )

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
    if config.use_evidence_gate:
        score_trajectory(
            gate,
            simulator,
            trajectory,
            config.n_agreement_samples,
            adjudicator=adjudicator,
            schema=get_schema(task.tool_family),
        )

    verdict = judge_rollout(task, final_state, trajectory, config.judge)
    checker_passed = verdict.passed
    tracker.record(task.tool_family, task.difficulty_meta.graph_complexity, checker_passed)

    if config.use_evidence_gate and not trajectory_evidence_accepted(trajectory):
        return  # simulator evidence too weak to trust this trajectory for learning

    # The playbook goes into the diagnosis so the teacher can say *which*
    # entries the agent followed, helpfully or not. Without it the teacher can
    # only invent new lessons, never credit or retire existing ones, and the
    # playbook accumulates guidance nothing can ever falsify.
    before = playbook_store.current
    diagnosis = diagnose(teacher, trajectory, checker_passed, playbook=before)
    candidates = optimizer.propose(before, diagnosis)
    if candidates:
        selected = optimizer.select(candidates)
        playbook_store.update(selected)
        logger.info(
            "playbook v%s -> v%s: %d -> %d entries, tags=%s",
            before.version,
            selected.version,
            len(before.entries),
            len(selected.entries),
            selected.tags(),
        )


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
    task_source: Callable[[int, int, tuple[float, float]], list[Task]] | None = None,
) -> list[Task]:
    """Runs one batch of tasks through the full closed loop; mutates
    `playbook_store` and `tracker` in place. `playbook_store` must already be
    seeded (`playbook_store.seed(...)`) before the first call. Returns the
    tasks used this iteration, mainly for test/inspection purposes.

    `task_source(graph_complexity, n, difficulty_band)` supplies the batch
    instead of generating it, e.g. from the task bank's train split. The band
    is passed rather than read from a closure because it is the curriculum:
    selecting on measured pass rate is the whole of the difficulty control, so
    a source that ignores it is not serving a curriculum at all. Returning
    fewer than `n` tasks is allowed and the shortfall is generated as usual, so
    a bank that runs dry mid-run degrades instead of stalling.
    """
    # `sample_task_graph` silently clamps to `len(tools)` if asked for more
    # nodes than the pool has — request no more than that up front so the
    # task's *actual* difficulty_meta.graph_complexity matches the bucket it
    # gets banked under, instead of drifting into a smaller one.
    effective_complexity = min(config.resolved_graph_complexity(), len(tools))

    tasks: list[Task] = []
    if task_source is not None:
        tasks = list(task_source(effective_complexity, config.tasks_per_iteration, config.difficulty_band))
        logger.info("task_source supplied %d/%d tasks", len(tasks), config.tasks_per_iteration)
    if len(tasks) < config.tasks_per_iteration:
        shortfall = config.tasks_per_iteration - len(tasks)
        batch_config = replace(config, tasks_per_iteration=shortfall)
        tasks += _generate_tasks_resiliently(teacher, tools, batch_config, effective_complexity)

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
