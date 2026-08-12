"""Versioned playbook storage + Pareto frontier + U7 rollback.

The orchestrator loop (data/code-architecture-plan.md §4) only ever talks to
this store, never to raw PlaybookModule objects floating around elsewhere —
that's what makes "roll back to best validation version instead of endlessly
appending memory" (U7 stop criterion) an enforceable operation rather than a
convention someone has to remember.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from qwen_agentworld.core.schemas import Playbook
from qwen_agentworld.playbook_store.leak_audit import audit_leakage


class LeakageDetectedError(ValueError):
    def __init__(self, violations: dict[str, list[str]]) -> None:
        self.violations = violations
        super().__init__(f"refusing to write playbook: leaked forbidden terms {violations}")


def fingerprint(playbook: Playbook, agent_model: str = "") -> str:
    """Stable identity of "the agent a measurement was taken against".

    Anything cached per-agent — a screened pass rate above all — has to say
    which agent, or it silently becomes a claim about a model that no longer
    exists. Version numbers will not do the job: a rolled-back edit bumps the
    version without changing a word, and two runs that converge on the same
    text should be allowed to share measurements. So this hashes the entry
    text itself, plus the model serving it.

    Entries are sorted by text rather than taken in list order: two playbooks
    holding the same set of rules are the same artifact as far as the agent
    reading them is concerned, and keying on insertion history would deny two
    such runs a shared measurement.
    """
    body = "\n".join(sorted(f"{entry.tag}:{entry.content}" for entry in playbook.entries))
    return hashlib.sha1(f"{agent_model}\n{body}".encode()).hexdigest()[:12]


def dominates(a: Playbook, b: Playbook) -> bool:
    """Standard Pareto dominance over (task_coverage, audit_acceptance,
    compactness). Higher is better on every axis.
    """
    a_scores = _scores(a)
    b_scores = _scores(b)
    at_least_as_good = all(a_val >= b_val for a_val, b_val in zip(a_scores, b_scores))
    strictly_better = any(a_val > b_val for a_val, b_val in zip(a_scores, b_scores))
    return at_least_as_good and strictly_better


def _scores(playbook: Playbook) -> tuple[float, float, float]:
    s = playbook.pareto_scores
    return (s.task_coverage, s.audit_acceptance, s.compactness)


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

    def record_validation(self, utility: float) -> Playbook:
        """Attach a held-out validation utility to the current playbook.

        This replaces the last history entry rather than appending a new one:
        the utility is an *annotation* on a playbook that already exists, not a
        new version of it, and appending would let one playbook be counted
        repeatedly by `stop_criterion`'s improvement window.
        """
        updated = self.current.model_copy(update={"validation_utility": utility})
        self._history[-1] = updated
        return updated

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
