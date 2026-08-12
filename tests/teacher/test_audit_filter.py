"""`drop_audit_failed` must remove known-bad tasks without requiring an audit.

Screening asks whether the agent passes; auditing asks whether the task is worth
passing. Both ends break the A/B, in opposite directions: an unpassable task
drags every arm down equally, and a too-weak checker scores for reasons that
have nothing to do with the arm.
"""

import pytest

from qwen_agentworld.core.schemas import CheckerSpec, DifficultyMeta, Task, TaskGraph, TaskGraphNode
from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, TaskBank


def make_task(prompt):
    return Task(
        tool_family="mcp_notes",
        task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="write_note")]),
        natural_language_prompt=prompt,
        initial_state={"notes": []},
        checker=CheckerSpec(executable_predicate="len(state['notes']) == 1"),
        difficulty_meta=DifficultyMeta(graph_complexity=1),
    )


@pytest.fixture
def bank(tmp_path):
    b = TaskBank(tmp_path)
    ids = {}
    for label, verdict in {
        "clean": (False, False),
        "unpassable": (True, False),
        "weak": (False, True),
        "unaudited": None,
    }.items():
        t = make_task(label)
        b.save(t, split=SPLIT_EVAL, gc=1)
        ids[label] = t.task_id
        if verdict is not None:
            b.set_audit_verdict(t.task_id, unpassable=verdict[0], too_weak=verdict[1])
    b.ids = ids
    return b


def _drawn(bank, **kw):
    return {t.task_id for t in bank.draw("mcp_notes", 1, 10, split=SPLIT_EVAL, **kw)}


def test_off_by_default_keeps_everything(bank):
    assert len(_drawn(bank)) == 4


def test_drops_both_failure_modes(bank):
    drawn = _drawn(bank, drop_audit_failed=True)
    assert bank.ids["unpassable"] not in drawn
    assert bank.ids["weak"] not in drawn


def test_keeps_unaudited_tasks(bank):
    # The flag means "drop what is known bad", not "require an audit" -- so it
    # can never empty a bank that simply has not been audited yet.
    assert _drawn(bank, drop_audit_failed=True) == {bank.ids["clean"], bank.ids["unaudited"]}


def test_verdict_survives_a_later_screening_write(bank):
    bank.set_baseline_pass_rate(bank.ids["unpassable"], 0.0)
    assert bank.ids["unpassable"] not in _drawn(bank, drop_audit_failed=True)


def test_unknown_task_id_reports_miss(bank):
    assert bank.set_audit_verdict("task_nope", unpassable=True, too_weak=False) is False
