import json
from unittest.mock import MagicMock

import pytest

from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookEntry, StepDiagnosis
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.optimizer.textgrad_engine import TextGradEngine


def diagnosis_tagged(tag="schema-grounding") -> Diagnosis:
    return Diagnosis(
        task_id="task_1",
        overall_verdict="failure",
        summary="missed a required field",
        step_diagnoses=[
            StepDiagnosis(step_id="s1", verdict="erroneous", feedback="missing arg", suggested_tag=tag)
        ],
    )


def critique(text="the playbook never mentions required fields") -> ChatResult:
    return ChatResult(content=json.dumps({"critique": text}))


def ops_reply(*ops) -> ChatResult:
    return ChatResult(content=json.dumps({"ops": list(ops)}))


def test_propose_critiques_then_edits_in_two_separate_calls():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        critique(),
        ops_reply({"op": "add", "tag": "schema-grounding", "content": "validate required fields first"}),
    ]
    after = TextGradEngine(teacher).propose(Playbook(version=2), diagnosis_tagged())[0]

    assert teacher.chat.call_count == 2
    assert after.version == 3
    assert after.entries[0].content == "validate required fields first"


def test_the_critique_is_passed_verbatim_into_the_edit_call():
    """The edit is meant to be grounded in an explicit, inspectable critique —
    that separation is the whole reason this engine costs two calls."""
    teacher = MagicMock()
    teacher.chat.side_effect = [critique("entry e1 is too vague to prevent this"), ops_reply()]
    current = Playbook(entries=[PlaybookEntry(entry_id="e1", tag="t", content="be careful")])
    TextGradEngine(teacher).propose(current, diagnosis_tagged())

    edit_user = teacher.chat.call_args_list[1].kwargs["messages"][1]["content"]
    assert "entry e1 is too vague to prevent this" in edit_user


def test_propose_leaves_prior_entries_intact():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        critique(),
        ops_reply({"op": "add", "tag": "x", "content": "a distinct new lesson about ordering"}),
    ]
    keep = PlaybookEntry(entry_id="keep", tag="t", content="an earlier lesson worth keeping")
    after = TextGradEngine(teacher).propose(Playbook(entries=[keep]), diagnosis_tagged())[0]
    assert after.by_id("keep").content == keep.content


def test_propose_skips_both_calls_on_a_clean_success():
    teacher = MagicMock()
    diagnosis = Diagnosis(task_id="t", overall_verdict="success", summary="")
    assert TextGradEngine(teacher).propose(Playbook(), diagnosis) == []
    teacher.chat.assert_not_called()


def test_propose_raises_after_exhausting_retries_on_a_malformed_critique():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content='{"critique": "unterminated')
    with pytest.raises(ValueError):
        TextGradEngine(teacher).propose(Playbook(), diagnosis_tagged())
    assert teacher.chat.call_count == 3
