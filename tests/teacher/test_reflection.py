import json
from unittest.mock import MagicMock

import pytest

from qwen_agentworld.core.schemas import PlaybookCategory, Step, ToolCall, Trajectory
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.teacher.reflection import diagnose


def make_trajectory() -> Trajectory:
    traj = Trajectory(task_id="task_x", playbook_version="v1")
    traj.steps.append(Step(tool_call=ToolCall(tool_name="search_docs", arguments={"query": "foo"}), accepted=True))
    traj.steps.append(Step(tool_call=ToolCall(tool_name="write_note", arguments={}), accepted=False))
    return traj


def test_diagnose_parses_reply_into_diagnosis():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content=json.dumps(
            {
                "overall_verdict": "partial",
                "summary": "second step used a malformed argument",
                "steps": [
                    {
                        "step_id": "s1",
                        "verdict": "correct",
                        "feedback": "found the right doc",
                        "suggested_category": None,
                    },
                    {
                        "step_id": "s2",
                        "verdict": "erroneous",
                        "feedback": "missing required argument",
                        "suggested_category": "schema_grounding",
                    },
                ],
            }
        )
    )
    trajectory = make_trajectory()
    diagnosis = diagnose(teacher, trajectory, checker_passed=False)

    assert diagnosis.task_id == "task_x"
    assert diagnosis.overall_verdict == "partial"
    assert len(diagnosis.step_diagnoses) == 2
    assert diagnosis.step_diagnoses[1].suggested_category == PlaybookCategory.SCHEMA_GROUNDING
    assert diagnosis.step_diagnoses[0].suggested_category is None


def test_diagnose_retries_on_truncated_content_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content='{"overall_verdict": "partial'),
        ChatResult(content=json.dumps({"overall_verdict": "partial", "summary": "s", "steps": []})),
    ]
    trajectory = make_trajectory()
    diagnosis = diagnose(teacher, trajectory, checker_passed=False)

    assert diagnosis.overall_verdict == "partial"
    assert teacher.chat.call_count == 2


def test_diagnose_raises_after_exhausting_retries_on_malformed_content():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content='{"overall_verdict": "partial')
    trajectory = make_trajectory()

    with pytest.raises(ValueError):
        diagnose(teacher, trajectory, checker_passed=False, max_attempts=3)
    assert teacher.chat.call_count == 3
