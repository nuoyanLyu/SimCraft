"""Tests for the two wiring fixes: the engines' entry length budget, and
deriving the leak audit's term set from the tools list."""

import json
from unittest.mock import MagicMock

from qwen_agentworld.core.schemas import (
    Diagnosis,
    Playbook,
    PlaybookEntry,
    StepDiagnosis,
    ToolFunctionSpec,
    ToolSpec,
)
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.optimizer.gepa_engine import GEPAEngine, build_entry_length_rule
from qwen_agentworld.optimizer.textgrad_engine import TextGradEngine
from qwen_agentworld.playbook_store.leak_audit import forbidden_terms_from_tools


def diagnosis() -> Diagnosis:
    return Diagnosis(
        task_id="t",
        overall_verdict="failure",
        summary="",
        step_diagnoses=[
            StepDiagnosis(
                step_id="s1", verdict="erroneous", feedback="missing arg", suggested_tag="schema-grounding"
            )
        ],
    )


def playbook_with(content: str) -> Playbook:
    return Playbook(entries=[PlaybookEntry(tag="schema-grounding", content=content)])


def ops_reply(content: str) -> ChatResult:
    return ChatResult(content=json.dumps({"ops": [{"op": "add", "tag": "x", "content": content}]}))


def tool(name: str) -> ToolSpec:
    return ToolSpec(function=ToolFunctionSpec(name=name, description="d"), family="notes")


# ------------------------------------------------------------- length budget --


def test_length_rule_states_the_number_the_model_must_hit():
    assert "25" in build_entry_length_rule(25)


def test_the_budget_bounds_one_entry_not_the_whole_playbook():
    """The old budget was per-module with a fixed module count, so it was really
    a cap on total learned content and it forced deletion once reached. Here it
    only keeps a single entry from turning into an essay."""
    rule = build_entry_length_rule(40).lower()
    assert "each entry" in rule


def test_gepa_tells_the_model_the_budget():
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply("short")
    GEPAEngine(teacher, entry_word_budget=25).propose(playbook_with("old"), diagnosis())

    system = teacher.chat.call_args.kwargs["messages"][0]
    assert system["role"] == "system"
    assert "25" in system["content"]


def test_textgrad_tells_the_model_the_budget_on_the_edit_call():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"critique": "the playbook never mentions the required id field"})),
        ops_reply("short"),
    ]
    TextGradEngine(teacher, entry_word_budget=25).propose(playbook_with("old"), diagnosis())

    edit_system = teacher.chat.call_args_list[1].kwargs["messages"][0]
    assert "25" in edit_system["content"]


def test_gepa_scores_compactness_of_the_playbook_it_just_wrote():
    """The regression that disabled Pareto selection entirely: the candidate used
    to inherit the parent's scores, which were the (0,0,0) default."""
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply("two words")
    candidates = GEPAEngine(teacher, entry_word_budget=100).propose(Playbook(), diagnosis())

    assert candidates[0].pareto_scores.compactness == 0.98


def test_gepa_scores_an_over_budget_entry_at_zero():
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply(" ".join(["word"] * 500))
    candidates = GEPAEngine(teacher, entry_word_budget=100).propose(Playbook(), diagnosis())

    assert candidates[0].pareto_scores.compactness == 0.0


def test_textgrad_scores_compactness_of_the_playbook_it_just_wrote():
    teacher = MagicMock()
    teacher.chat.side_effect = [ChatResult(content=json.dumps({"critique": "too vague"})), ops_reply("two words")]
    candidates = TextGradEngine(teacher, entry_word_budget=100).propose(Playbook(), diagnosis())

    assert candidates[0].pareto_scores.compactness == 0.98


# --------------------------------------------------------- forbidden term set --


def test_terms_are_derived_from_the_tools_the_playbook_was_evolved_on():
    terms = forbidden_terms_from_tools([tool("write_note"), tool("delete_note")])
    assert terms == {"write_note", "delete_note"}


def test_very_short_tool_names_are_dropped():
    """A 2-3 character name matches inside ordinary English ('ls' in 'else'),
    which would fail every playbook regardless of what it says."""
    assert forbidden_terms_from_tools([tool("ls"), tool("cat"), tool("grep_files")]) == {"grep_files"}


def test_an_empty_tools_list_yields_an_empty_set():
    assert forbidden_terms_from_tools([]) == set()
