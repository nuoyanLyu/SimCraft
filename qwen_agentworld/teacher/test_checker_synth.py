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
            {"executable_predicate": "state.get('notes') and len(state['notes']) > 0", "step_wise_diagnostics": True}
        )
    )
    checker = synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"notes": []})
    assert checker.executable_predicate == "state.get('notes') and len(state['notes']) > 0"
    assert checker.step_wise_diagnostics is True


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
        synthesize_checker(teacher, make_graph(), [make_tool("write_note")], {"notes": []})
