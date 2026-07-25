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

from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookModule


def _mean_pareto(playbook: Playbook) -> tuple[float, float, float]:
    modules = list(playbook.modules.values())
    if not modules:
        return (0.0, 0.0, 0.0)
    n = len(modules)
    return (
        sum(m.pareto_scores.task_coverage for m in modules) / n,
        sum(m.pareto_scores.audit_acceptance for m in modules) / n,
        sum(m.pareto_scores.compactness for m in modules) / n,
    )


def _module_score(module: PlaybookModule) -> float:
    s = module.pareto_scores
    return s.task_coverage + s.audit_acceptance + s.compactness


class PlaybookOptimizer(ABC):
    @abstractmethod
    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        """Produce one or more candidate mutated playbooks in response to a
        single diagnosis. A diagnosis about `schema_grounding` should mutate
        that category's module, not `error_recovery` — credit assignment is
        the whole point of taking a `Diagnosis` instead of a scalar reward.
        """

    def select(self, candidates: list[Playbook]) -> Playbook:
        """Default selection: prefer a non-dominated (Pareto-frontier)
        candidate, tie-broken by summed mean pareto score. Engines may
        override this, but most shouldn't need to — Pareto selection is a
        property of the scores, not of how the candidates were generated.
        """
        if not candidates:
            raise ValueError("cannot select from an empty candidate list")
        non_dominated = [
            c for c in candidates if not any(_dominates(other, c) for other in candidates if other is not c)
        ]
        pool = non_dominated or candidates
        return max(pool, key=lambda p: sum(_mean_pareto(p)))

    def merge(self, a: Playbook, b: Playbook) -> Playbook:
        """Combine two playbooks module-by-module, keeping whichever module
        has the higher summed pareto score in each category.
        """
        modules = dict(a.modules)
        for category, b_module in b.modules.items():
            a_module = modules.get(category)
            if a_module is None or _module_score(b_module) > _module_score(a_module):
                modules[category] = b_module
        return Playbook(modules=modules)


def _dominates(a: Playbook, b: Playbook) -> bool:
    a_scores, b_scores = _mean_pareto(a), _mean_pareto(b)
    at_least_as_good = all(x >= y for x, y in zip(a_scores, b_scores))
    strictly_better = any(x > y for x, y in zip(a_scores, b_scores))
    return at_least_as_good and strictly_better
