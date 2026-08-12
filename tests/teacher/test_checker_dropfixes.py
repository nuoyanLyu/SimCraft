"""Tests for the two bugs that silently discarded generated tasks.

Both were measured on the gc=4 re-audit of 2026-07-29, where 2 of 8 tasks never
came out of `synthesize_checker`.
"""

import json
from unittest.mock import MagicMock

from qwen_agentworld.core.schemas import TaskGraph, TaskGraphNode, ToolFunctionSpec, ToolSpec
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.teacher.checker_synth import (
    audit_no_nl_leak,
    collect_canonical_strings,
    synthesize_checker,
)

LONG_LITERAL = "Brunch with Sam at 11am, then a walk in the park."


def tools():
    return [ToolSpec(function=ToolFunctionSpec(name="write_note", description="d"), family="notes")]


def graph():
    return TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="write_note")])


def synth(teacher, initial_state=None, nl_prompt="do the thing"):
    return synthesize_checker(
        teacher,
        graph(),
        tools(),
        initial_state if initial_state is not None else {"notes": []},
        nl_prompt=nl_prompt,
    )


# ------------------------------------------------ null executable_predicate --


def test_a_null_predicate_is_retried_rather_than_crashing_the_task():
    """`payload["executable_predicate"]` raises KeyError only when the key is
    absent; an explicit null used to reach compile() and the TypeError escaped
    the retry loop, discarding the whole task."""
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"executable_predicate": None})),
        ChatResult(content=json.dumps({"executable_predicate": "len(state['notes']) == 1"})),
    ]
    checker = synth(teacher)
    assert checker.executable_predicate == "len(state['notes']) == 1"
    assert teacher.chat.call_count == 2


def test_an_empty_string_predicate_is_also_retried():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"executable_predicate": "   "})),
        ChatResult(content=json.dumps({"executable_predicate": "len(state['notes']) == 1"})),
    ]
    assert synth(teacher).executable_predicate == "len(state['notes']) == 1"


# ------------------------------------------------------- canonical literals --


def test_a_long_literal_the_teacher_invented_is_still_a_leak():
    predicate = f"state['notes'][0]['content'] == {LONG_LITERAL!r}"
    assert audit_no_nl_leak(predicate, canonical_strings=set())


def test_a_long_literal_copied_from_the_initial_state_is_allowed():
    """A checker is supposed to compare against exact canonical text; the
    alternative the teacher reaches for otherwise — asserting a character count —
    is what made tasks unpassable."""
    predicate = f"state['notes'][0]['content'] == {LONG_LITERAL!r}"
    assert audit_no_nl_leak(predicate, canonical_strings={LONG_LITERAL}) == []


def test_the_allowlist_matches_verbatim_only():
    predicate = f"state['notes'][0]['content'] == {LONG_LITERAL!r}"
    assert audit_no_nl_leak(predicate, canonical_strings={LONG_LITERAL + " Extra."})


def test_canonical_strings_are_collected_from_nested_state_and_prompt():
    state = {"notes": [{"title": "Weekend", "content": LONG_LITERAL, "tags": ["fun"]}]}
    found = collect_canonical_strings(state, "please update the Weekend note")
    assert LONG_LITERAL in found
    assert "please update the Weekend note" in found
    assert "fun" in found
    assert "notes" in found  # keys too: a checker may compare against a field name


def test_synthesis_accepts_a_predicate_quoting_the_initial_state_verbatim():
    teacher = MagicMock()
    predicate = f"state['notes'][0]['content'] == {LONG_LITERAL!r} and len(state['notes']) == 2"
    teacher.chat.return_value = ChatResult(content=json.dumps({"executable_predicate": predicate}))

    checker = synth(teacher, initial_state={"notes": [{"content": LONG_LITERAL}]})
    assert checker.executable_predicate == predicate
    assert teacher.chat.call_count == 1  # accepted first try, not fought through retries
