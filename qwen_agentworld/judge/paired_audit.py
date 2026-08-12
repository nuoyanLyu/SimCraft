"""Checker scoring + paired ("with candidate playbook" vs "without") audit.

U5-f's reward-hacking concern: a playbook mutation could look like it improves
pass rate while actually exploiting a checker weakness (or the simulator's
hallucination tendencies) rather than teaching genuinely better tool use. The
mitigation this module implements is a paired trajectory comparison — run the
*same* task with and without the candidate playbook change, on the *same*
simulator/agent — so a pass-rate delta is attributable to the playbook change
specifically, not to task-to-task variance or a checker quirk that both runs
would trip identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qwen_agentworld.core.schemas import CheckerSpec
from qwen_agentworld.teacher.safe_predicate import UnsafePredicateError, evaluate_predicate, evaluate_step_wise_predicate

logger = logging.getLogger(__name__)

# Runtime errors a *validated* predicate can still raise against a real state:
# a key the simulator never wrote, a type mismatch, an empty sequence. These are
# checker-synthesis defects, so they score as "not passed" like any other
# malformed checker -- but they are logged, because a checker that always errors
# would otherwise look indistinguishable from a task the agent always fails.
_PREDICATE_RUNTIME_ERRORS = (KeyError, IndexError, TypeError, ValueError, AttributeError, ZeroDivisionError, NameError)


def judge_checker_with_reason(
    checker: CheckerSpec, final_state: dict, states: list[dict] | None = None, task_id: str = "?"
) -> tuple[bool, str]:
    """`judge_checker`, but also reports *why* the verdict came out that way.

    A False from `judge_checker` collapses three different events into one
    bit: the agent genuinely failed the task, the checker was rejected by AST
    validation, or the checker blew up against the state it was handed. The
    first is a result; the other two are defects in our own pipeline, and a
    run that cannot separate them cannot tell a null effect from a broken
    harness. The 2026-07-29 evolve run scored 9.3% of its judgments off a
    raised predicate while reading as ordinary failures.

    Reason is one of: `pass`, `checker_false`, `checker_unsafe`,
    `checker_raised:<ExcType>`.
    """
    try:
        if checker.step_wise_diagnostics and checker.step_wise_predicate and states is not None:
            passed = evaluate_step_wise_predicate(checker.step_wise_predicate, states)
        else:
            passed = evaluate_predicate(checker.executable_predicate, final_state)
        return passed, ("pass" if passed else "checker_false")
    except UnsafePredicateError:
        return False, "checker_unsafe"
    except _PREDICATE_RUNTIME_ERRORS as exc:
        logger.warning(
            "task %s: checker predicate raised %s at evaluation time, scoring as not-passed: %s",
            task_id,
            type(exc).__name__,
            exc,
        )
        return False, f"checker_raised:{type(exc).__name__}"


def judge_checker(
    checker: CheckerSpec, final_state: dict, states: list[dict] | None = None, task_id: str = "?"
) -> bool:
    """Evaluate a checker against the final canonical state. Returns False
    (not raise) on a malformed predicate — whether it fails AST validation or
    blows up against the actual state — since a checker synthesis bug
    should read as "task not passed," not crash the audit pipeline — the
    audit trail (`CheckerAuditError` in checker_synth) is where malformed
    predicates are supposed to get caught before this point.

    For a step-wise checker (a reversible task whose final observable state
    can equal the initial state), pass `states` — the ordered list of
    canonical states (initial + one per executed step) — and the step-wise
    predicate is used instead of the end-state one. Without `states` it
    falls back to the end-state predicate (e.g. the paired audit, which
    only tracks final states and tolerates a checker quirk both runs trip).

    `task_id` only ever reaches the warning log, but without it a run's
    warnings cannot be attributed: "50 predicates raised KeyError" could be
    one broken checker hit 50 times or 50 separate tasks, and those two
    readings call for opposite fixes. Use `judge_checker_with_reason` when the
    caller needs to record which of those two readings applies.
    """
    return judge_checker_with_reason(checker, final_state, states, task_id)[0]


@dataclass
class PairedAuditResult:
    task_id: str
    with_playbook_passed: bool
    without_playbook_passed: bool

    @property
    def genuinely_improved(self) -> bool:
        """The playbook change is credited only if it turns a failure into a
        pass on the *same* task — a pass on both runs isn't attributable to
        the change, and could be the trivial "checker was already satisfied
        by the initial state" case worth flagging separately.
        """
        return self.with_playbook_passed and not self.without_playbook_passed

    @property
    def regressed(self) -> bool:
        return self.without_playbook_passed and not self.with_playbook_passed


def paired_audit(
    task_id: str,
    checker: CheckerSpec,
    final_state_with_playbook: dict,
    final_state_without_playbook: dict,
) -> PairedAuditResult:
    return PairedAuditResult(
        task_id=task_id,
        with_playbook_passed=judge_checker(checker, final_state_with_playbook, task_id=task_id),
        without_playbook_passed=judge_checker(checker, final_state_without_playbook, task_id=task_id),
    )
