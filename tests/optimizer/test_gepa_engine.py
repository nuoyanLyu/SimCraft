import json
from unittest.mock import MagicMock

import pytest

from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookEntry, StepDiagnosis
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.optimizer.gepa_engine import GEPAEngine, observed_tags, parse_ops


def diagnosis_tagged(*tags, verdict="failure") -> Diagnosis:
    return Diagnosis(
        task_id="task_1",
        overall_verdict=verdict,
        summary="missed a required field",
        step_diagnoses=[
            StepDiagnosis(step_id=f"s{i}", verdict="erroneous", feedback="missing arg", suggested_tag=tag)
            for i, tag in enumerate(tags)
        ],
    )


def ops_reply(*ops) -> ChatResult:
    return ChatResult(content=json.dumps({"ops": list(ops)}))


def test_observed_tags_keeps_every_distinct_lesson_not_just_the_majority():
    """`most_implicated_category` reduced a whole trajectory to one bucket, so a
    trajectory that failed for two reasons could only ever teach one."""
    diagnosis = diagnosis_tagged("error-recovery", "error-recovery", "schema-grounding")
    assert observed_tags(diagnosis) == ["error-recovery", "schema-grounding"]


def test_observed_tags_is_empty_when_nothing_was_labelled():
    assert observed_tags(Diagnosis(task_id="t", overall_verdict="success", summary="")) == []


def test_propose_skips_the_teacher_call_on_a_clean_success():
    teacher = MagicMock()
    diagnosis = Diagnosis(task_id="t", overall_verdict="success", summary="")
    assert GEPAEngine(teacher).propose(Playbook(), diagnosis) == []
    teacher.chat.assert_not_called()


def test_propose_adds_an_entry_without_disturbing_the_existing_ones():
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply(
        {"op": "add", "tag": "schema-grounding", "content": "always validate required fields first"}
    )
    keep = PlaybookEntry(entry_id="keep", tag="error-recovery", content="retry once after a transient error")
    current = Playbook(version=3, entries=[keep])

    candidates = GEPAEngine(teacher).propose(current, diagnosis_tagged("schema-grounding"))

    assert len(candidates) == 1
    after = candidates[0]
    assert after.version == 4
    assert after.by_id("keep").content == keep.content
    assert "always validate required fields first" in [e.content for e in after.entries]


def test_propose_can_update_an_entry_the_teacher_named():
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply(
        {"op": "update", "entry_ids": ["e1"], "content": "confirm every required field, including optional-looking ones"}
    )
    current = Playbook(entries=[PlaybookEntry(entry_id="e1", tag="schema-grounding", content="check fields")])

    after = GEPAEngine(teacher).propose(current, diagnosis_tagged("schema-grounding"))[0]
    assert after.by_id("e1").version == 2
    assert len(after.entries) == 1


def test_propose_records_credit_even_when_the_teacher_proposes_no_edit():
    """"The agent followed this entry and still failed" is information about the
    entry; an iteration that produces no ops must not throw it away."""
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply()
    current = Playbook(entries=[PlaybookEntry(entry_id="e1", tag="t", content="some guidance")])
    diagnosis = diagnosis_tagged("schema-grounding")
    diagnosis.harmful_entry_ids = ["e1"]

    candidates = GEPAEngine(teacher).propose(current, diagnosis)
    assert candidates[0].by_id("e1").stats.harmful == 1


def test_propose_returns_nothing_when_there_is_neither_an_edit_nor_credit():
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply()
    assert GEPAEngine(teacher).propose(Playbook(), diagnosis_tagged("schema-grounding")) == []


def test_propose_retries_on_a_truncated_reply_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content='{"ops": [{"op": "add", "content": "unterminated'),
        ops_reply({"op": "add", "tag": "x", "content": "always validate required fields first"}),
    ]
    candidates = GEPAEngine(teacher).propose(Playbook(), diagnosis_tagged("schema-grounding"))
    assert teacher.chat.call_count == 2
    assert candidates[0].entries[0].content == "always validate required fields first"


def test_propose_raises_after_exhausting_retries_on_a_malformed_reply():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content='{"ops": [{"op": "add", "content": "unterminated')
    with pytest.raises(ValueError):
        GEPAEngine(teacher).propose(Playbook(), diagnosis_tagged("schema-grounding"))
    assert teacher.chat.call_count == 3


def test_parse_ops_drops_one_malformed_op_and_keeps_the_rest():
    """A batch is a set of independent edits, not a transaction."""
    ops = parse_ops(json.dumps({"ops": [{"op": "nonsense"}, {"op": "add", "content": "good one"}]}))
    assert len(ops) == 1 and ops[0].content == "good one"


def test_parse_ops_rejects_a_reply_whose_ops_field_is_not_a_list():
    with pytest.raises(TypeError):
        parse_ops(json.dumps({"ops": "add something"}))


def test_the_teacher_is_shown_the_current_entries_with_their_ids():
    """It cannot update, merge or credit an entry it cannot name."""
    teacher = MagicMock()
    teacher.chat.return_value = ops_reply()
    current = Playbook(entries=[PlaybookEntry(entry_id="e1", tag="t", content="existing guidance")])
    GEPAEngine(teacher).propose(current, diagnosis_tagged("schema-grounding"))

    user = teacher.chat.call_args.kwargs["messages"][1]["content"]
    assert "e1" in user and "existing guidance" in user
