import pytest

from qwen_agentworld.core.schemas import ParetoScores, Playbook, PlaybookEntry
from qwen_agentworld.optimizer.scoring import (
    DEFAULT_ENTRY_WORD_BUDGET,
    compactness_score,
    score_playbook,
)


def pb(*contents, **scores) -> Playbook:
    return Playbook(
        entries=[PlaybookEntry(tag="t", content=c) for c in contents],
        pareto_scores=ParetoScores(**scores),
    )


def test_empty_playbook_is_maximally_compact():
    assert compactness_score(pb()) == 1.0


def test_compactness_falls_to_zero_at_the_budget_and_stays_there():
    assert compactness_score(pb(" ".join(["word"] * 100)), entry_word_budget=100) == 0.0
    # Well past the budget must not go negative, or a bloated entry would look
    # *worse than impossible* and distort the Pareto comparison.
    assert compactness_score(pb(" ".join(["word"] * 1000)), entry_word_budget=100) == 0.0


def test_compactness_is_monotonically_decreasing_in_entry_length():
    short = compactness_score(pb("a b c"))
    long = compactness_score(pb(" ".join(["word"] * 200)))
    assert 1.0 > short > long >= 0.0


def test_compactness_does_not_punish_learning_more_things():
    """The axis exists to keep an individual rule tight, not to cap how much the
    playbook has learned. Penalising entry *count* would make a run that learned
    twenty lessons score worse than one that learned two — exactly backwards."""
    two = compactness_score(pb("short rule one", "short rule two"))
    twenty = compactness_score(pb(*["short rule one"] * 20))
    assert two == twenty


def test_a_non_positive_budget_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        compactness_score(pb("anything"), entry_word_budget=0)


def test_score_playbook_recomputes_compactness_from_current_entries():
    stale = pb(" ".join(["word"] * 38), compactness=1.0)
    assert score_playbook(stale, entry_word_budget=40).pareto_scores.compactness == pytest.approx(0.05)


def test_score_playbook_keeps_observed_axes_when_no_measurement_is_supplied():
    """A caller updating only the length must not silently zero out coverage and
    acceptance, which are measured elsewhere and far more expensive to obtain."""
    scored = score_playbook(pb("short", task_coverage=0.7, audit_acceptance=0.9))
    assert scored.pareto_scores.task_coverage == 0.7
    assert scored.pareto_scores.audit_acceptance == 0.9


def test_score_playbook_overwrites_observed_axes_when_measurements_are_supplied():
    scored = score_playbook(pb("short", task_coverage=0.7), task_coverage=0.2, audit_acceptance=0.3)
    assert scored.pareto_scores.task_coverage == 0.2
    assert scored.pareto_scores.audit_acceptance == 0.3


def test_score_playbook_does_not_mutate_its_argument():
    original = pb(" ".join(["word"] * 500), compactness=1.0)
    score_playbook(original)
    assert original.pareto_scores.compactness == 1.0


def test_an_essay_sized_entry_scores_zero_compactness():
    """Regression guard for the growth this module was written to catch: the
    evolved incremental_execution module reached 746 words with nothing pushing
    back. An entry is meant to be one rule, not a document."""
    assert compactness_score(pb(" ".join(["word"] * 746)), DEFAULT_ENTRY_WORD_BUDGET) == 0.0
