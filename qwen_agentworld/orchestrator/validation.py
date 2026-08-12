"""Held-out validation of a playbook, and the U7 rollback it feeds.

`Playbook.validation_utility` has existed since Stage 6 and drives two
mechanisms — `loop.stop_criterion` and `PlaybookStore.rollback_to_best_validation`
— but nothing in production ever wrote it. `stop_criterion` therefore could not
return True, `best_validation_playbook()` raised `RuntimeError` unconditionally,
and rollback was unreachable. Combined with `loop._process_task` accepting every
proposed mutation via `playbook_store.update(optimizer.select(candidates))`, the
loop had no mechanism at all for rejecting or undoing a harmful edit: a mutation
that lowered the agent's pass rate was indistinguishable from one that raised it.

Validation lives here rather than inside `run_iteration` for two reasons. It has
to run on *held-out* tasks — scoring the playbook on the same rollouts that
produced it measures memorisation, not utility — and it costs a full rollout per
task, so the caller decides the cadence rather than paying it every iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qwen_agentworld.core.schemas import Playbook, Task, ToolSpec
from qwen_agentworld.judge.verdict import JudgeConfig, judge_rollout
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.playbook_store.store import PlaybookStore
from qwen_agentworld.simulator_gym.env import rollout

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    utility: float
    n_passed: int
    n_rollouts: int
    n_errored: int


def _task_passes(
    agent: LLMClient,
    simulator: LLMClient,
    tools: list[ToolSpec],
    playbook: Playbook,
    task: Task,
    judge: JudgeConfig | None = None,
) -> bool:
    trajectory, final_state = rollout(agent, simulator, task, tools, playbook=playbook)
    # Must be the same judge the loop scores with, or accept/rollback is decided
    # on a different measurement from the one it is meant to protect.
    return judge_rollout(task, final_state, trajectory, judge).passed


def evaluate_playbook(
    agent: LLMClient,
    simulator: LLMClient,
    tools: list[ToolSpec],
    playbook: Playbook,
    tasks: list[Task],
    *,
    reps: int = 1,
    judge: JudgeConfig | None = None,
) -> ValidationResult:
    """Pass rate of `playbook` over held-out `tasks`, `reps` rollouts each.

    Deliberately does not run the evidence gate, the diagnosis, or the
    optimizer: this is a measurement, and anything that could write back to the
    playbook would make the measurement circular.

    Rollouts that raise are counted as failures rather than skipped. A playbook
    whose guidance makes the agent emit malformed tool calls would otherwise
    look *better* the more often it broke, since only its successes would reach
    the denominator.
    """
    if not tasks:
        raise ValueError("held-out task set is empty; validation utility would be undefined")

    passed = errored = total = 0
    for task in tasks:
        for _ in range(max(1, reps)):
            total += 1
            try:
                passed += int(_task_passes(agent, simulator, tools, playbook, task, judge))
            except Exception as exc:  # noqa: BLE001 — a broken rollout is a failed rollout
                errored += 1
                logger.warning("validation rollout failed on task %s: %s", task.task_id, exc)

    return ValidationResult(
        utility=passed / total, n_passed=passed, n_rollouts=total, n_errored=errored
    )


def validate_and_maybe_rollback(
    store: PlaybookStore,
    agent: LLMClient,
    simulator: LLMClient,
    tools: list[ToolSpec],
    tasks: list[Task],
    *,
    reps: int = 1,
    tolerance: float = 0.0,
    judge: JudgeConfig | None = None,
) -> tuple[ValidationResult, bool]:
    """Score the store's current playbook on held-out tasks, record the result,
    and roll back if it came out worse than the best version seen so far.

    `tolerance` is the regression the caller is willing to absorb before
    reverting. It should not be zero in practice: the utility is a pass rate over
    a finite task set, so consecutive evaluations of an *identical* playbook
    differ by sampling noise alone, and a zero tolerance would revert on that
    noise and freeze the loop at whichever version got lucky first.

    Returns the result and whether a rollback happened.
    """
    result = evaluate_playbook(agent, simulator, tools, playbook=store.current, tasks=tasks, reps=reps, judge=judge)
    store.record_validation(result.utility)

    best = store.best_validation_playbook()
    if best.playbook_id == store.current.playbook_id:
        return result, False  # the current playbook *is* the best; nothing to revert to

    if result.utility + tolerance >= best.validation_utility:
        return result, False

    logger.info(
        "rolling back: current playbook scored %.3f against best %.3f (tolerance %.3f)",
        result.utility,
        best.validation_utility,
        tolerance,
    )
    store.rollback_to_best_validation()
    return result, True
