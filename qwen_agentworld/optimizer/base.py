"""PlaybookOptimizer abstract interface.

U1 (which engine) is settled on TextGrad as the default (see
`optimizer/__init__.py::build_optimizer`), with GEPA available as an
alternative — every engine implements this same interface so
`orchestrator/loop.py` never depends on which one is selected; swapping is a
one-line config change, not a refactor.

Hard rule from the architecture plan (data/code-architecture-plan.md §3,
"两条硬规则"): optimizer/* only reads evidence_gate/judge output — never raw
simulator output. This is enforced by the type signature itself: `propose()`
takes a `Diagnosis` (teacher/reflection.py's output), which carries
step-level verdicts/feedback/suggested_category but never the raw
`simulator_raw_output` that produced them — an engine physically cannot reach
into the simulator's raw predictions through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookEntry


def _pareto(playbook: Playbook) -> tuple[float, float, float]:
    s = playbook.pareto_scores
    return (s.task_coverage, s.audit_acceptance, s.compactness)


def _entry_rank(entry: PlaybookEntry) -> tuple[int, int]:
    """Order for keeping one of two versions of the same lesson: earned credit
    first, then how recently it was revised.
    """
    return (entry.stats.utility, entry.version)


class PlaybookOptimizer(ABC):
    @abstractmethod
    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        """Produce one or more candidate edited playbooks in response to a
        single diagnosis. Each candidate should differ from `current` by a
        small batch of `optimizer.ops` operations, not by a rewrite: credit
        assignment is the whole point of taking a `Diagnosis` instead of a
        scalar reward, and it is only usable if the edit it justifies is
        localised to the entries it implicates.
        """

    def select(self, candidates: list[Playbook]) -> Playbook:
        """Default selection: prefer a non-dominated (Pareto-frontier)
        candidate, tie-broken by summed pareto score. Engines may override
        this, but most shouldn't need to — Pareto selection is a property of
        the scores, not of how the candidates were generated.
        """
        if not candidates:
            raise ValueError("cannot select from an empty candidate list")
        non_dominated = [
            c for c in candidates if not any(_dominates(other, c) for other in candidates if other is not c)
        ]
        pool = non_dominated or candidates
        return max(pool, key=lambda p: sum(_pareto(p)))

    def merge(self, a: Playbook, b: Playbook) -> Playbook:
        """Union the two entry lists, keeping one version per entry lineage.

        Union rather than pick-a-winner: two playbooks that evolved from the
        same seed have mostly learned *different* things, so choosing between
        them wholesale discards half the run. Entries collide only when they
        share an `entry_id` — the same lesson, revised on both sides — and
        those are resolved by earned credit.
        """
        by_id: dict[str, PlaybookEntry] = {e.entry_id: e for e in a.entries}
        for entry in b.entries:
            incumbent = by_id.get(entry.entry_id)
            if incumbent is None or _entry_rank(entry) > _entry_rank(incumbent):
                by_id[entry.entry_id] = entry
        ordered = [by_id[e.entry_id] for e in a.entries]
        ordered += [by_id[e.entry_id] for e in b.entries if e.entry_id not in {x.entry_id for x in a.entries}]
        return Playbook(version=max(a.version, b.version) + 1, entries=ordered)


def _dominates(a: Playbook, b: Playbook) -> bool:
    a_scores, b_scores = _pareto(a), _pareto(b)
    at_least_as_good = all(x >= y for x, y in zip(a_scores, b_scores))
    strictly_better = any(x > y for x, y in zip(a_scores, b_scores))
    return at_least_as_good and strictly_better
