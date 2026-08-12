import pytest

from qwen_agentworld.core.schemas import (
    Diagnosis,
    EntryStats,
    ParetoScores,
    Playbook,
    PlaybookEntry,
)
from qwen_agentworld.optimizer.base import PlaybookOptimizer


class _StubOptimizer(PlaybookOptimizer):
    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        return []


def entry(entry_id="e1", content="x", tag="t", helpful=0, harmful=0, version=1) -> PlaybookEntry:
    return PlaybookEntry(
        entry_id=entry_id,
        tag=tag,
        content=content,
        version=version,
        stats=EntryStats(helpful=helpful, harmful=harmful),
    )


def playbook(task_coverage=0.0, audit_acceptance=0.0, compactness=0.0, entries=None) -> Playbook:
    return Playbook(
        entries=entries if entries is not None else [entry()],
        pareto_scores=ParetoScores(
            task_coverage=task_coverage, audit_acceptance=audit_acceptance, compactness=compactness
        ),
    )


def test_select_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        _StubOptimizer().select([])


def test_select_prefers_non_dominated_candidate():
    weak, strong = playbook(0.1, 0.1, 0.1), playbook(0.9, 0.9, 0.9)
    assert _StubOptimizer().select([weak, strong]) is strong


def test_select_tie_breaks_by_summed_pareto_among_non_dominated():
    # neither dominates the other (each better on a different axis) -> tie-break by sum
    a, b = playbook(0.9, 0.1, 0.1), playbook(0.1, 0.9, 0.9)
    assert _StubOptimizer().select([a, b]) is b


def test_pareto_scores_live_on_the_playbook_so_dominance_can_actually_fire():
    """The regression that flattened the frontier: the axes used to sit on each
    module, but a rollout measures the whole playbook, so every module carried
    the identical number and `dominates` could never be true."""
    winner = _StubOptimizer().select([playbook(0.2, 0.2, 0.2), playbook(0.3, 0.3, 0.3)])
    assert winner.pareto_scores.task_coverage == 0.3


def test_merge_unions_entries_rather_than_choosing_a_winner():
    """Two playbooks evolved from the same seed have mostly learned *different*
    lessons; picking one wholesale would discard half the run."""
    a = playbook(entries=[entry("a1", "learned A")])
    b = playbook(entries=[entry("b1", "learned B")])
    merged = _StubOptimizer().merge(a, b)
    assert [e.content for e in merged.entries] == ["learned A", "learned B"]


def test_merge_resolves_a_shared_lineage_by_earned_credit():
    a = playbook(entries=[entry("shared", "vague version", helpful=0)])
    b = playbook(entries=[entry("shared", "sharpened version", helpful=5)])
    merged = _StubOptimizer().merge(a, b)
    assert len(merged.entries) == 1
    assert merged.entries[0].content == "sharpened version"


def test_merge_keeps_the_incumbent_when_the_challenger_has_not_earned_more():
    a = playbook(entries=[entry("shared", "incumbent", helpful=3)])
    b = playbook(entries=[entry("shared", "challenger", helpful=1)])
    assert _StubOptimizer().merge(a, b).entries[0].content == "incumbent"
