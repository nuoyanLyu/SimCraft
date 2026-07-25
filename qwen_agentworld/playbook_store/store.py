"""Versioned playbook storage + Pareto frontier + U7 rollback.

The orchestrator loop (data/code-architecture-plan.md §4) only ever talks to
this store, never to raw PlaybookModule objects floating around elsewhere —
that's what makes "roll back to best validation version instead of endlessly
appending memory" (U7 stop criterion) an enforceable operation rather than a
convention someone has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qwen_agentworld.core.schemas import Playbook
from qwen_agentworld.playbook_store.leak_audit import audit_leakage


class LeakageDetectedError(ValueError):
    def __init__(self, violations: dict[str, list[str]]) -> None:
        self.violations = violations
        super().__init__(f"refusing to write playbook: leaked forbidden terms {violations}")


def dominates(a: Playbook, b: Playbook) -> bool:
    """Standard Pareto dominance over (task_coverage, audit_acceptance, compactness),
    aggregated across a playbook's modules as the mean per axis. Higher is
    better on every axis.
    """
    a_scores = _mean_scores(a)
    b_scores = _mean_scores(b)
    at_least_as_good = all(a_val >= b_val for a_val, b_val in zip(a_scores, b_scores))
    strictly_better = any(a_val > b_val for a_val, b_val in zip(a_scores, b_scores))
    return at_least_as_good and strictly_better


def _mean_scores(playbook: Playbook) -> tuple[float, float, float]:
    modules = list(playbook.modules.values())
    if not modules:
        return (0.0, 0.0, 0.0)
    n = len(modules)
    return (
        sum(m.pareto_scores.task_coverage for m in modules) / n,
        sum(m.pareto_scores.audit_acceptance for m in modules) / n,
        sum(m.pareto_scores.compactness for m in modules) / n,
    )


@dataclass
class PlaybookStore:
    forbidden_terms: set[str] = field(default_factory=set)
    _history: list[Playbook] = field(default_factory=list)
    _frontier: list[Playbook] = field(default_factory=list)

    @property
    def current(self) -> Playbook:
        if not self._history:
            raise RuntimeError("playbook store is empty; call seed() first")
        return self._history[-1]

    @property
    def history(self) -> list[Playbook]:
        return list(self._history)

    def seed(self, playbook: Playbook) -> None:
        self._write(playbook)

    def update(self, playbook: Playbook) -> None:
        self._write(playbook)
        self._update_frontier(playbook)

    def _write(self, playbook: Playbook) -> None:
        violations = audit_leakage(playbook, self.forbidden_terms)
        if violations:
            raise LeakageDetectedError(violations)
        self._history.append(playbook)

    def _update_frontier(self, candidate: Playbook) -> None:
        if any(dominates(existing, candidate) for existing in self._frontier):
            return  # candidate is dominated by something already on the frontier
        self._frontier = [existing for existing in self._frontier if not dominates(candidate, existing)]
        self._frontier.append(candidate)

    @property
    def frontier(self) -> list[Playbook]:
        return list(self._frontier)

    def best_validation_playbook(self) -> Playbook:
        """U7 rollback target: highest validation_utility ever recorded.
        Playbooks with validation_utility=None (never evaluated) are ignored.
        """
        evaluated = [p for p in self._history if p.validation_utility is not None]
        if not evaluated:
            raise RuntimeError("no playbook in history has a validation_utility set yet")
        return max(evaluated, key=lambda p: p.validation_utility)

    def rollback_to_best_validation(self) -> Playbook:
        best = self.best_validation_playbook()
        self._history.append(best)  # rollback is itself a recorded event, not a history rewrite
        return best
