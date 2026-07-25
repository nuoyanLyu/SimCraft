import pytest

from qwen_agentworld.core.schemas import Diagnosis, ParetoScores, Playbook, PlaybookCategory, PlaybookModule
from qwen_agentworld.optimizer.base import PlaybookOptimizer


class _StubOptimizer(PlaybookOptimizer):
    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        return []


def module(category, task_coverage=0.0, audit_acceptance=0.0, compactness=0.0) -> PlaybookModule:
    return PlaybookModule(
        category=category,
        content="x",
        pareto_scores=ParetoScores(
            task_coverage=task_coverage, audit_acceptance=audit_acceptance, compactness=compactness
        ),
    )


def test_select_raises_on_empty_candidates():
    optimizer = _StubOptimizer()
    with pytest.raises(ValueError):
        optimizer.select([])


def test_select_prefers_non_dominated_candidate():
    optimizer = _StubOptimizer()
    weak = Playbook(modules={PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING, 0.1, 0.1, 0.1)})
    strong = Playbook(modules={PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING, 0.9, 0.9, 0.9)})
    assert optimizer.select([weak, strong]) is strong


def test_select_tie_breaks_by_summed_mean_pareto_among_non_dominated():
    optimizer = _StubOptimizer()
    # neither dominates the other (each better on a different axis) -> tie-break by sum
    a = Playbook(modules={PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING, 0.9, 0.1, 0.1)})
    b = Playbook(modules={PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING, 0.1, 0.9, 0.9)})
    assert optimizer.select([a, b]) is b


def test_merge_keeps_higher_scoring_module_per_category():
    optimizer = _StubOptimizer()
    a = Playbook(
        modules={
            PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING, 0.9, 0.9, 0.9),
            PlaybookCategory.ERROR_RECOVERY: module(PlaybookCategory.ERROR_RECOVERY, 0.1, 0.1, 0.1),
        }
    )
    b = Playbook(
        modules={
            PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING, 0.1, 0.1, 0.1),
            PlaybookCategory.ERROR_RECOVERY: module(PlaybookCategory.ERROR_RECOVERY, 0.9, 0.9, 0.9),
        }
    )
    merged = optimizer.merge(a, b)
    assert merged.modules[PlaybookCategory.SCHEMA_GROUNDING] is a.modules[PlaybookCategory.SCHEMA_GROUNDING]
    assert merged.modules[PlaybookCategory.ERROR_RECOVERY] is b.modules[PlaybookCategory.ERROR_RECOVERY]


def test_merge_adds_categories_only_present_in_b():
    optimizer = _StubOptimizer()
    a = Playbook(modules={PlaybookCategory.SCHEMA_GROUNDING: module(PlaybookCategory.SCHEMA_GROUNDING)})
    b = Playbook(modules={PlaybookCategory.ERROR_RECOVERY: module(PlaybookCategory.ERROR_RECOVERY)})
    merged = optimizer.merge(a, b)
    assert set(merged.modules) == {PlaybookCategory.SCHEMA_GROUNDING, PlaybookCategory.ERROR_RECOVERY}
