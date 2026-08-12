from unittest.mock import MagicMock

from qwen_agentworld.core.schemas import TaskGraph, TaskGraphNode, ToolFunctionSpec, ToolSpec
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.teacher.checker_synth import CheckerAuditError, audit_no_nl_leak, synthesize_checker

import json
import pytest


def make_tool(name: str) -> ToolSpec:
    return ToolSpec(
        function=ToolFunctionSpec(name=name, description=f"do {name}", parameters={"type": "object", "properties": {}}),
        family="mcp_A",
    )


def make_graph() -> TaskGraph:
    return TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="write_note")])


def test_audit_accepts_clean_predicate():
    assert audit_no_nl_leak("state['status'] == 'completed'") == []


def test_audit_flags_wordy_string_literal_as_nl_leak():
    violations = audit_no_nl_leak("state['note'] == 'the correct final answer the user was looking for'")
    assert violations


def test_audit_flags_unsafe_expression():
    violations = audit_no_nl_leak("__import__('os').system('rm -rf /')")
    assert violations


def test_synthesize_checker_parses_valid_reply():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content=json.dumps(
            {
                "executable_predicate": "state.get('notes') and len(state['notes']) > 0",
                "step_wise_diagnostics": True,
                "step_wise_predicate": "any(len(s.get('notes', [])) > 0 for s in states)",
            }
        )
    )
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"notes": []}, nl_prompt="Write a note titled Draft.")
    assert checker.executable_predicate == "state.get('notes') and len(state['notes']) > 0"
    assert checker.step_wise_diagnostics is True
    assert checker.step_wise_predicate == "any(len(s.get('notes', [])) > 0 for s in states)"


def test_synthesize_checker_rejects_nl_leak_from_teacher():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content=json.dumps(
            {
                "executable_predicate": "state['answer'] == 'the note should mention the docs were found and saved'",
                "step_wise_diagnostics": False,
            }
        )
    )
    with pytest.raises(CheckerAuditError):
        synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"notes": []}, nl_prompt="Write a note titled Draft.")
    # exhausted every retry against a teacher that never fixes the leak
    assert teacher.chat.call_count == 4


def test_synthesize_checker_retries_after_audit_rejection_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"executable_predicate": "isinstance(state['x'], int)", "step_wise_diagnostics": False})),
        ChatResult(content=json.dumps({"executable_predicate": "state['x'] > 0", "step_wise_diagnostics": False})),
    ]
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"x": 0}, nl_prompt="Write a note titled Draft.")
    assert checker.executable_predicate == "state['x'] > 0"
    assert teacher.chat.call_count == 2
    # the retry prompt included the audit's own violation message as feedback
    retry_message = teacher.chat.call_args.kwargs["messages"][-1]
    assert retry_message["role"] == "user"
    assert "isinstance" in retry_message["content"]


def test_synthesize_checker_retries_after_empty_content_then_succeeds(monkeypatch):
    monkeypatch.setattr("qwen_agentworld.teacher.checker_synth.time.sleep", lambda *a: None)
    # Regression: a real Claude teacher returned empty content for checker
    # synthesis at graph_complexity=3, and the loop crashed because only audit
    # violations were retried, not parse failures. Now an empty reply is fed
    # back and retried within the same bounded loop.
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=""),
        ChatResult(content=json.dumps({"executable_predicate": "state['x'] > 0", "step_wise_diagnostics": False})),
    ]
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"x": 0}, nl_prompt="Write a note titled Draft.")
    assert checker.executable_predicate == "state['x'] > 0"
    assert teacher.chat.call_count == 2
    retry_message = teacher.chat.call_args.kwargs["messages"][-1]
    assert retry_message["role"] == "user"
    assert "empty or not a valid JSON" in retry_message["content"]


def test_synthesize_checker_retries_after_malformed_json_then_succeeds(monkeypatch):
    monkeypatch.setattr("qwen_agentworld.teacher.checker_synth.time.sleep", lambda *a: None)
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content="here you go: not json at all"),
        ChatResult(content=json.dumps({"executable_predicate": "len(state['notes']) == 0", "step_wise_diagnostics": False})),
    ]
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"notes": [{"title": "Draft"}]}, nl_prompt="Write a note titled Draft.")
    assert checker.executable_predicate == "len(state['notes']) == 0"
    assert teacher.chat.call_count == 2


def test_synthesize_checker_raises_after_exhausting_on_persistent_empty_content(monkeypatch):
    monkeypatch.setattr("qwen_agentworld.teacher.checker_synth.time.sleep", lambda *a: None)
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content="")
    with pytest.raises(CheckerAuditError):
        synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"x": 1}, nl_prompt="Write a note titled Draft.")
    assert teacher.chat.call_count == 4


def test_synthesize_checker_end_state_only_leaves_step_wise_predicate_none():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content=json.dumps(
            {"executable_predicate": "state.get('status') == 'done'", "step_wise_diagnostics": False}
        )
    )
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"status": "new"}, nl_prompt="Write a note titled Draft.")
    assert checker.step_wise_diagnostics is False
    assert checker.step_wise_predicate is None


def test_synthesize_checker_retries_when_step_wise_predicate_missing_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(
            content=json.dumps(
                {"executable_predicate": "len(state.get('notes', [])) == 0", "step_wise_diagnostics": True}
            )
        ),
        ChatResult(
            content=json.dumps(
                {
                    "executable_predicate": "len(state.get('notes', [])) == 0",
                    "step_wise_diagnostics": True,
                    "step_wise_predicate": "any(len(s.get('notes', [])) > 0 for s in states) and len(states[-1].get('notes', [])) == 0",
                }
            )
        ),
    ]
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"notes": []}, nl_prompt="Write a note titled Draft.")
    assert checker.step_wise_predicate is not None
    assert teacher.chat.call_count == 2


def test_synthesize_checker_rejects_tautology_predicate():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content=json.dumps(
            {"executable_predicate": "state.get('x') == 'a' or state.get('x') != 'a'", "step_wise_diagnostics": False}
        )
    )
    with pytest.raises(CheckerAuditError):
        synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"x": None}, nl_prompt="Write a note titled Draft.")


def test_audit_flags_constant_predicate():
    assert audit_no_nl_leak("1 == 1")


def test_audit_flags_tautology():
    assert audit_no_nl_leak("state['x'] == 'a' or state['x'] != 'a'")


def test_synthesize_checker_rejects_predicate_already_true_of_initial_state():
    """A checker that already holds before the agent acts scores the task
    passed for doing nothing, so it measures nothing. The teacher gets the
    violation as feedback and its corrected predicate is accepted.
    """
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"executable_predicate": "len(state['notes']) == 1", "step_wise_diagnostics": False})),
        ChatResult(content=json.dumps({"executable_predicate": "len(state['notes']) == 2", "step_wise_diagnostics": False})),
    ]
    checker = synthesize_checker(
        teacher, make_graph(), [make_tool("write_note")], {"notes": [{"title": "Draft"}]},
        nl_prompt="Add a second note titled Recap.",
    )
    assert checker.executable_predicate == "len(state['notes']) == 2"
    retry_message = teacher.chat.call_args.kwargs["messages"][-1]
    assert "already true of the initial state" in retry_message["content"]


def test_checker_prompt_carries_the_task_instruction():
    """Regression for the 2026-07-28 null A/B: the teacher wrote checkers from
    the tool graph alone and invented values the instruction never asked for,
    making 4 of 12 eval tasks unpassable. The instruction must reach the prompt.
    """
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"executable_predicate": "len(state['notes']) == 2", "step_wise_diagnostics": False})),
    ]
    synthesize_checker(
        teacher, make_graph(), [make_tool("write_note")], {"notes": [{"title": "Draft"}]},
        nl_prompt="Tag the note titled 'Chicken Soup' with 'dinner'.",
    )
    user_msg = teacher.chat.call_args.kwargs["messages"][1]["content"]
    assert "Tag the note titled 'Chicken Soup' with 'dinner'." in user_msg
