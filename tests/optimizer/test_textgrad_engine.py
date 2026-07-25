import json
from unittest.mock import MagicMock

import pytest

from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookCategory, PlaybookModule, StepDiagnosis
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.optimizer.textgrad_engine import TextGradEngine


def diagnosis_implicating(category: PlaybookCategory) -> Diagnosis:
    return Diagnosis(
        task_id="task_1",
        overall_verdict="failure",
        summary="missed a required field",
        step_diagnoses=[
            StepDiagnosis(step_id="s1", verdict="erroneous", feedback="missing arg", suggested_category=category)
        ],
    )


def test_propose_returns_empty_list_when_nothing_implicated():
    teacher = MagicMock()
    engine = TextGradEngine(teacher)
    diagnosis = Diagnosis(task_id="t", overall_verdict="success", summary="", step_diagnoses=[])
    assert engine.propose(Playbook(), diagnosis) == []
    teacher.chat.assert_not_called()


def test_propose_makes_two_separate_calls_critique_then_edit():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"critique": "doesn't check required fields before calling"})),
        ChatResult(content=json.dumps({"content": "check required fields are present before calling"})),
    ]
    engine = TextGradEngine(teacher)

    current = Playbook(
        version=1,
        modules={
            PlaybookCategory.SCHEMA_GROUNDING: PlaybookModule(category=PlaybookCategory.SCHEMA_GROUNDING, content="old")
        },
    )
    candidates = engine.propose(current, diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))

    assert teacher.chat.call_count == 2
    assert len(candidates) == 1
    new_module = candidates[0].modules[PlaybookCategory.SCHEMA_GROUNDING]
    assert new_module.content == "check required fields are present before calling"
    assert candidates[0].version == 2


def test_propose_retries_a_truncated_critique_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content='{"critique": "unterminated'),
        ChatResult(content=json.dumps({"critique": "doesn't check required fields before calling"})),
        ChatResult(content=json.dumps({"content": "check required fields are present before calling"})),
    ]
    engine = TextGradEngine(teacher)

    candidates = engine.propose(Playbook(), diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))
    assert teacher.chat.call_count == 3
    new_module = candidates[0].modules[PlaybookCategory.SCHEMA_GROUNDING]
    assert new_module.content == "check required fields are present before calling"


def test_propose_retries_a_truncated_edit_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=json.dumps({"critique": "doesn't check required fields before calling"})),
        ChatResult(content='{"content": "unterminated'),
        ChatResult(content=json.dumps({"content": "check required fields are present before calling"})),
    ]
    engine = TextGradEngine(teacher)

    candidates = engine.propose(Playbook(), diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))
    assert teacher.chat.call_count == 3
    new_module = candidates[0].modules[PlaybookCategory.SCHEMA_GROUNDING]
    assert new_module.content == "check required fields are present before calling"


def test_propose_raises_after_exhausting_retries_on_malformed_critique():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content='{"critique": "unterminated')
    engine = TextGradEngine(teacher)

    with pytest.raises(ValueError):
        engine.propose(Playbook(), diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))
    assert teacher.chat.call_count == 3
