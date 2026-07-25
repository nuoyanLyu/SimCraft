import pytest

from qwen_agentworld.core.schemas import ParetoScores, Playbook, PlaybookCategory, PlaybookModule
from qwen_agentworld.playbook_store.store import LeakageDetectedError, PlaybookStore, dominates


def module(coverage, audit, compactness, content="generic advice"):
    return PlaybookModule(
        category=PlaybookCategory.SCHEMA_GROUNDING,
        content=content,
        pareto_scores=ParetoScores(task_coverage=coverage, audit_acceptance=audit, compactness=compactness),
    )


def playbook(coverage, audit, compactness, validation_utility=None):
    return Playbook(
        modules={PlaybookCategory.SCHEMA_GROUNDING: module(coverage, audit, compactness)},
        validation_utility=validation_utility,
    )


def test_seed_and_current():
    store = PlaybookStore()
    pb = playbook(0.5, 0.5, 0.5)
    store.seed(pb)
    assert store.current.playbook_id == pb.playbook_id


def test_current_raises_when_empty():
    store = PlaybookStore()
    with pytest.raises(RuntimeError):
        _ = store.current


def test_update_refuses_leaked_content():
    store = PlaybookStore(forbidden_terms={"search_docs"})
    leaked = playbook(0.5, 0.5, 0.5)
    leaked.modules[PlaybookCategory.SCHEMA_GROUNDING].content = "Retry search_docs with broader terms."
    with pytest.raises(LeakageDetectedError):
        store.update(leaked)


def test_dominance_strictly_better_on_all_axes():
    better = playbook(0.9, 0.9, 0.9)
    worse = playbook(0.5, 0.5, 0.5)
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_dominance_mixed_axes_is_incomparable():
    a = playbook(0.9, 0.1, 0.5)
    b = playbook(0.1, 0.9, 0.5)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_frontier_keeps_only_non_dominated_candidates():
    store = PlaybookStore()
    store.seed(playbook(0.5, 0.5, 0.5))
    store.update(playbook(0.9, 0.9, 0.9))  # dominates the seed
    assert len(store.frontier) == 1
    assert store.frontier[0].modules[PlaybookCategory.SCHEMA_GROUNDING].pareto_scores.task_coverage == 0.9

    store.update(playbook(0.1, 0.95, 0.5))  # higher on audit_acceptance, lower elsewhere -> incomparable
    assert len(store.frontier) == 2


def test_rollback_to_best_validation_ignores_unevaluated_playbooks():
    store = PlaybookStore()
    store.seed(playbook(0.5, 0.5, 0.5, validation_utility=0.3))
    store.update(playbook(0.6, 0.6, 0.6, validation_utility=None))  # never evaluated -> shouldn't win
    store.update(playbook(0.4, 0.4, 0.4, validation_utility=0.8))  # lower pareto scores but best validated

    rolled_back = store.rollback_to_best_validation()
    assert rolled_back.validation_utility == 0.8
    assert store.current is rolled_back  # rollback is recorded as the new current


def test_best_validation_raises_when_nothing_evaluated():
    store = PlaybookStore()
    store.seed(playbook(0.5, 0.5, 0.5))
    with pytest.raises(RuntimeError):
        store.best_validation_playbook()
