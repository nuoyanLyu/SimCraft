"""The schema reaching the three places that used to invent it separately:
the teacher writing `initial_state`, the teacher writing the checker, and the
simulator predicting the next state.
"""

import json

import pytest

from qwen_agentworld.core.schemas import TaskGraph, TaskGraphNode, ToolCall, ToolFunctionSpec, ToolSpec
from qwen_agentworld.simulator_gym.env import complete_fields, simulate_next_state
from qwen_agentworld.teacher.checker_synth import audit_schema_conformance, synthesize_checker
from qwen_agentworld.teacher.task_generator import SchemaViolation, instantiate_nl_and_state
from qwen_agentworld.tools.state_schema import MCP_NOTES_SCHEMA

NOTES_TOOLS = [
    ToolSpec(
        function=ToolFunctionSpec(
            name="write_note",
            description="Create a note.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                "required": ["title", "content"],
            },
        ),
        family="mcp_notes",
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="tag_note",
            description="Tag a note.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "tag": {"type": "string"}},
                "required": ["title", "tag"],
            },
        ),
        family="mcp_notes",
    ),
]

GRAPH = TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="write_note", depends_on=[])])


class _ScriptedTeacher:
    """Replies with each queued string in turn, recording what it was sent."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts = []

    def chat(self, messages, max_tokens=None, tools=None):
        self.prompts.append(messages[-1]["content"])
        self.system = messages[0]["content"]

        class _R:
            content = self._replies.pop(0)
            tool_calls = None

        return _R()


# --- checker audit ------------------------------------------------------- #


def test_audit_rejects_the_invented_id_field():
    """`note['id']` — the single most common defect in the bank (5 of 7)."""
    violations = audit_schema_conformance(
        "any(n['id'] == 'n1' for n in state['notes'])", MCP_NOTES_SCHEMA, NOTES_TOOLS
    )
    assert any("'id'" in v for v in violations)


def test_audit_rejects_a_misspelled_tool_parameter():
    """`arguments['note_title']` where the tool declares `title`."""
    violations = audit_schema_conformance(
        "any(c['arguments']['note_title'] == 'A' for c in state['_action_log'])",
        MCP_NOTES_SCHEMA,
        NOTES_TOOLS,
    )
    assert any("note_title" in v for v in violations)


def test_audit_rejects_a_nonexistent_tool():
    violations = audit_schema_conformance(
        "any(c['tool'] == 'archive_note' for c in state['_action_log'])",
        MCP_NOTES_SCHEMA,
        NOTES_TOOLS,
    )
    assert any("archive_note" in v for v in violations)


def test_audit_accepts_a_conforming_predicate():
    predicate = (
        "any(n['title'] == 'A' and 'urgent' in n['tags'] for n in state['notes']) "
        "and any(c['tool'] == 'tag_note' and c['arguments']['tag'] == 'urgent' "
        "for c in state['_action_log'])"
    )
    assert audit_schema_conformance(predicate, MCP_NOTES_SCHEMA, NOTES_TOOLS) == []


def test_audit_reads_dict_get_the_same_as_a_subscript():
    violations = audit_schema_conformance(
        "any(n.get('id') for n in state['notes'])", MCP_NOTES_SCHEMA, NOTES_TOOLS
    )
    assert any("'id'" in v for v in violations)


def test_synthesize_checker_retries_a_schema_violating_predicate():
    bad = json.dumps(
        {"executable_predicate": "any(n['id'] == 'n1' for n in state['notes'])",
         "step_wise_diagnostics": False}
    )
    good = json.dumps(
        {"executable_predicate": "any(n['title'] == 'B' for n in state['notes'])",
         "step_wise_diagnostics": False}
    )
    teacher = _ScriptedTeacher([bad, good])
    checker = synthesize_checker(
        teacher,
        GRAPH,
        NOTES_TOOLS,
        {"notes": [{"title": "A", "content": "a", "tags": []}]},
        nl_prompt="Create a note titled B.",
        schema=MCP_NOTES_SCHEMA,
    )
    assert checker.executable_predicate == "any(n['title'] == 'B' for n in state['notes'])"
    assert "'id'" in teacher.prompts[-1], "the violation must be fed back as feedback"


def test_checker_prompt_carries_the_schema():
    good = json.dumps(
        {"executable_predicate": "any(n['title'] == 'B' for n in state['notes'])",
         "step_wise_diagnostics": False}
    )
    teacher = _ScriptedTeacher([good])
    synthesize_checker(
        teacher, GRAPH, NOTES_TOOLS, {"notes": []},
        nl_prompt="Create a note titled B.", schema=MCP_NOTES_SCHEMA,
    )
    assert "Canonical state schema" in teacher.prompts[0]


# --- task generation ----------------------------------------------------- #


def test_instantiation_retries_a_nonconforming_initial_state():
    bad = json.dumps(
        {"natural_language_prompt": "Tag note A as urgent.",
         "initial_state": {"notes": [{"title": "A", "content": "a"}]}}  # no tags
    )
    good = json.dumps(
        {"natural_language_prompt": "Tag note A as urgent.",
         "initial_state": {"notes": [{"title": "A", "content": "a", "tags": []}]}}
    )
    teacher = _ScriptedTeacher([bad, good])
    _, state = instantiate_nl_and_state(teacher, GRAPH, NOTES_TOOLS, schema=MCP_NOTES_SCHEMA)
    assert state["notes"][0]["tags"] == []
    assert "tags" in teacher.prompts[-1]


def test_instantiation_raises_when_the_state_never_conforms():
    """Dropping the task beats banking one whose checker can never pass."""
    bad = json.dumps(
        {"natural_language_prompt": "x", "initial_state": {"notes": [{"title": "A"}]}}
    )
    teacher = _ScriptedTeacher([bad] * 3)
    with pytest.raises(SchemaViolation):
        instantiate_nl_and_state(teacher, GRAPH, NOTES_TOOLS, schema=MCP_NOTES_SCHEMA)


def test_instantiation_without_a_schema_is_unchanged():
    reply = json.dumps(
        {"natural_language_prompt": "x", "initial_state": {"anything": 1}}
    )
    teacher = _ScriptedTeacher([reply])
    _, state = instantiate_nl_and_state(teacher, GRAPH, NOTES_TOOLS)
    assert state == {"anything": 1}


# --- simulator ----------------------------------------------------------- #


class _StubSimulator:
    def __init__(self, content):
        self._content = content
        self.system = None

    def chat(self, messages, max_tokens=None, tools=None):
        self.system = messages[0]["content"]

        class _R:
            content = self._content
            tool_calls = None

        _R.content = self._content
        return _R()


def test_schema_completes_a_field_no_sibling_could_supply():
    """The 2026-07-29 failure that field completion alone could not repair:
    the collection started empty, so the one note created is also the only
    one, and nothing in the data says `tags` should exist."""
    prior = {"notes": []}
    predicted = {"notes": [{"title": "A", "content": "a"}]}

    assert "tags" not in complete_fields(prior, predicted)["notes"][0]
    assert complete_fields(prior, predicted, MCP_NOTES_SCHEMA)["notes"][0]["tags"] == []


def test_schema_does_not_overwrite_a_populated_field():
    prior = {"notes": [{"title": "A", "content": "a", "tags": ["x"]}]}
    predicted = {"notes": [{"title": "A", "content": "a", "tags": ["x", "y"]}]}
    completed = complete_fields(prior, predicted, MCP_NOTES_SCHEMA)
    assert completed["notes"][0]["tags"] == ["x", "y"]


def test_schema_restores_a_collection_absent_from_both_states():
    assert complete_fields({}, {}, MCP_NOTES_SCHEMA) == {"notes": []}


def test_simulator_prompt_carries_the_schema():
    simulator = _StubSimulator('{"next_state": {"notes": []}}')
    simulate_next_state(
        simulator, {"notes": []}, ToolCall(tool_name="write_note", arguments={}), MCP_NOTES_SCHEMA
    )
    assert "Canonical state schema" in simulator.system
    assert "tags" in simulator.system


def test_simulate_next_state_without_a_schema_is_unchanged():
    simulator = _StubSimulator('{"next_state": {"notes": [{"title": "A"}]}}')
    result = simulate_next_state(
        simulator, {"notes": []}, ToolCall(tool_name="write_note", arguments={})
    )
    assert result == {"notes": [{"title": "A"}]}
