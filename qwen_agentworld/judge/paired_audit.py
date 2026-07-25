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

from dataclasses import dataclass

from qwen_agentworld.core.schemas import CheckerSpec
from qwen_agentworld.teacher.safe_predicate import UnsafePredicateError, evaluate_predicate, evaluate_step_wise_predicate


def judge_checker(
    checker: CheckerSpec, final_state: dict, states: list[dict] | None = None
) -> bool:
    """Evaluate a checker against the final canonical state. Returns False
    (not raise) on a malformed predicate, since a checker synthesis bug
    should read as "task not passed," not crash the audit pipeline — the
    audit trail (`CheckerAuditError` in checker_synth) is where malformed
    predicates are supposed to get caught before this point.

    For a step-wise checker (a reversible task whose final observable state
    can equal the initial state), pass `states` — the ordered list of
    canonical states (initial + one per executed step) — and the step-wise
    predicate is used instead of the end-state one. Without `states` it
    falls back to the end-state predicate (e.g. the paired audit, which
    only tracks final states and tolerates a checker quirk both runs trip).
    """
    try:
        if checker.step_wise_diagnostics and checker.step_wise_predicate and states is not None:
            return evaluate_step_wise_predicate(checker.step_wise_predicate, states)
        return evaluate_predicate(checker.executable_predicate, final_state)
    except UnsafePredicateError:
        return False


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
        with_playbook_passed=judge_checker(checker, final_state_with_playbook),
        without_playbook_passed=judge_checker(checker, final_state_without_playbook),
    )
