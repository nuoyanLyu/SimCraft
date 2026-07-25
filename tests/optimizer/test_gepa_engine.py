import json
from unittest.mock import MagicMock

import pytest

from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookCategory, PlaybookModule, StepDiagnosis
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.optimizer.gepa_engine import GEPAEngine, most_implicated_category


def diagnosis_implicating(category: PlaybookCategory) -> Diagnosis:
    return Diagnosis(
        task_id="task_1",
        overall_verdict="failure",
        summary="missed a required field",
        step_diagnoses=[
            StepDiagnosis(step_id="s1", verdict="erroneous", feedback="missing arg", suggested_category=category)
        ],
    )


def test_most_implicated_category_picks_majority_vote():
    diagnosis = Diagnosis(
        task_id="t",
        overall_verdict="failure",
        summary="",
        step_diagnoses=[
            StepDiagnosis(step_id="s1", verdict="erroneous", feedback="", suggested_category=PlaybookCategory.ERROR_RECOVERY),
            StepDiagnosis(step_id="s2", verdict="erroneous", feedback="", suggested_category=PlaybookCategory.ERROR_RECOVERY),
            StepDiagnosis(step_id="s3", verdict="erroneous", feedback="", suggested_category=PlaybookCategory.SCHEMA_GROUNDING),
        ],
    )
    assert most_implicated_category(diagnosis) == PlaybookCategory.ERROR_RECOVERY


def test_most_implicated_category_none_when_no_suggestions():
    diagnosis = Diagnosis(task_id="t", overall_verdict="success", summary="", step_diagnoses=[])
    assert most_implicated_category(diagnosis) is None


def test_propose_returns_empty_list_when_nothing_implicated():
    teacher = MagicMock()
    engine = GEPAEngine(teacher)
    diagnosis = Diagnosis(task_id="t", overall_verdict="success", summary="", step_diagnoses=[])
    assert engine.propose(Playbook(), diagnosis) == []
    teacher.chat.assert_not_called()


def test_propose_mutates_the_implicated_module_and_bumps_version():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content=json.dumps({"content": "always validate required fields first"}))
    engine = GEPAEngine(teacher)

    current = Playbook(
        version=3,
        modules={
            PlaybookCategory.SCHEMA_GROUNDING: PlaybookModule(
                category=PlaybookCategory.SCHEMA_GROUNDING, content="old guidance", version=2
            )
        },
    )
    candidates = engine.propose(current, diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))

    assert len(candidates) == 1
    new_playbook = candidates[0]
    assert new_playbook.version == 4
    new_module = new_playbook.modules[PlaybookCategory.SCHEMA_GROUNDING]
    assert new_module.content == "always validate required fields first"
    assert new_module.version == 3
    assert current.modules[PlaybookCategory.SCHEMA_GROUNDING].module_id in new_module.provenance


def test_propose_retries_on_truncated_content_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content='{"content": "unterminated'),
        ChatResult(content=json.dumps({"content": "always validate required fields first"})),
    ]
    engine = GEPAEngine(teacher)

    candidates = engine.propose(Playbook(), diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))
    assert teacher.chat.call_count == 2
    new_module = candidates[0].modules[PlaybookCategory.SCHEMA_GROUNDING]
    assert new_module.content == "always validate required fields first"


def test_propose_raises_after_exhausting_retries_on_malformed_content():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content='{"content": "unterminated')
    engine = GEPAEngine(teacher)

    with pytest.raises(ValueError):
        engine.propose(Playbook(), diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))
    assert teacher.chat.call_count == 3


def test_propose_leaves_other_categories_untouched():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content=json.dumps({"content": "new content"}))
    engine = GEPAEngine(teacher)

    untouched = PlaybookModule(category=PlaybookCategory.ERROR_RECOVERY, content="keep me")
    current = Playbook(
        modules={
            PlaybookCategory.ERROR_RECOVERY: untouched,
            PlaybookCategory.SCHEMA_GROUNDING: PlaybookModule(category=PlaybookCategory.SCHEMA_GROUNDING, content="old"),
        }
    )
    candidates = engine.propose(current, diagnosis_implicating(PlaybookCategory.SCHEMA_GROUNDING))
    assert candidates[0].modules[PlaybookCategory.ERROR_RECOVERY] is untouched
