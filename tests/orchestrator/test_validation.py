"""Tests for held-out validation and U7 rollback — the mechanism that lets the
loop reject a harmful mutation instead of accepting every one unconditionally."""

from unittest.mock import MagicMock, patch

import pytest

from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    Playbook,
    PlaybookEntry,
    Task,
    TaskGraph,
    TaskGraphNode,
    Trajectory,
)
from qwen_agentworld.judge.verdict import Verdict
from qwen_agentworld.orchestrator import validation as V
from qwen_agentworld.playbook_store.store import PlaybookStore


def make_task(task_id: str = "task_1") -> Task:
    return Task(
        task_id=task_id,
        tool_family="notes",
        task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="write_note")]),
        natural_language_prompt="do the thing",
        initial_state={},
        checker=CheckerSpec(executable_predicate="True"),
        difficulty_meta=DifficultyMeta(graph_complexity=1),
    )


def playbook(content: str = "guidance") -> Playbook:
    return Playbook(
        entries=[
            PlaybookEntry(
                tag="schema-grounding", content=content
            )
        ]
    )


def stub_rollouts(outcomes):
    """Patch the rollout+checker pair so each call yields the next outcome."""
    results = iter(outcomes)

    def fake_rollout(*args, **kwargs):
        outcome = next(results)
        if isinstance(outcome, Exception):
            raise outcome
        return Trajectory(task_id="t", playbook_version="pb_1", steps=[]), {"passed": outcome}

    return (
        patch.object(V, "rollout", side_effect=fake_rollout),
        patch.object(V, "judge_rollout",
                     side_effect=lambda task, final_state, trajectory=None, config=None:
                         Verdict(passed=final_state["passed"], reason="", source="checker")),
    )


def evaluate(outcomes, tasks, **kwargs):
    roll, judge = stub_rollouts(outcomes)
    with roll, judge:
        return V.evaluate_playbook(MagicMock(), MagicMock(), [], playbook(), tasks, **kwargs)


# ------------------------------------------------------------- evaluation --


def test_utility_is_the_pass_rate_over_held_out_tasks():
    result = evaluate([True, False, True, True], [make_task(f"t{i}") for i in range(4)])
    assert result.utility == 0.75
    assert (result.n_passed, result.n_rollouts) == (3, 4)


def test_reps_multiply_the_rollouts_per_task():
    result = evaluate([True, False, True], [make_task()], reps=3)
    assert result.n_rollouts == 3
    assert result.utility == pytest.approx(2 / 3)


def test_a_crashed_rollout_counts_as_a_failure_not_as_a_skip():
    """Otherwise a playbook that makes the agent emit malformed calls would
    score higher the more often it broke, since only successes would reach the
    denominator."""
    result = evaluate([True, RuntimeError("agent emitted garbage")], [make_task("a"), make_task("b")])
    assert result.utility == 0.5
    assert (result.n_errored, result.n_rollouts) == (1, 2)


def test_an_empty_held_out_set_is_rejected_rather_than_scored():
    with pytest.raises(ValueError):
        evaluate([], [])


# --------------------------------------------------------------- rollback --


def seeded_store(*playbooks_with_utilities):
    store = PlaybookStore()
    for content, utility in playbooks_with_utilities:
        store.seed(playbook(content))
        if utility is not None:
            store.record_validation(utility)
    return store


def run_rollback(store, outcomes, **kwargs):
    roll, judge = stub_rollouts(outcomes)
    with roll, judge:
        return V.validate_and_maybe_rollback(
            store, MagicMock(), MagicMock(), [], [make_task("h1"), make_task("h2")], **kwargs
        )


def test_record_validation_annotates_rather_than_appending_a_version():
    """Appending would let one playbook be counted repeatedly by
    stop_criterion's improvement window."""
    store = PlaybookStore()
    store.seed(playbook())
    store.record_validation(0.4)
    assert len(store.history) == 1
    assert store.current.validation_utility == 0.4


def test_a_regression_rolls_back_to_the_best_earlier_playbook():
    store = seeded_store(("good", 0.8), ("bad", None))
    result, rolled_back = run_rollback(store, [False, False])

    assert result.utility == 0.0
    assert rolled_back
    assert store.current.entries[0].content == "good"


def test_an_improvement_is_kept():
    store = seeded_store(("good", 0.5), ("better", None))
    result, rolled_back = run_rollback(store, [True, True])

    assert result.utility == 1.0
    assert not rolled_back
    assert store.current.entries[0].content == "better"


def test_a_regression_within_tolerance_is_absorbed():
    """The utility is a pass rate over a finite set, so an identical playbook
    scores differently run to run; a zero tolerance would revert on noise."""
    store = seeded_store(("good", 1.0), ("equally_good", None))
    _, rolled_back = run_rollback(store, [True, False], tolerance=0.6)
    assert not rolled_back


def test_the_first_ever_evaluation_has_nothing_to_roll_back_to():
    store = PlaybookStore()
    store.seed(playbook("first"))
    result, rolled_back = run_rollback(store, [False, False])

    assert result.utility == 0.0
    assert not rolled_back
    assert store.current.validation_utility == 0.0


def test_recorded_utilities_make_the_stop_criterion_reachable():
    """The end-to-end point of this module: with utilities actually written,
    U7 can fire, which it never could when nothing set the field."""
    from qwen_agentworld.orchestrator.loop import stop_criterion

    store = seeded_store(("a", 0.8), ("b", 0.7), ("c", 0.6), ("d", 0.5))
    assert stop_criterion(store.history)
