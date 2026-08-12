import json

import pytest

from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    Step,
    Task,
    TaskGraph,
    TaskGraphNode,
    ToolCall,
    Trajectory,
)
from qwen_agentworld.judge.llm_judge import build_judge_prompt, judge_with_llm
from qwen_agentworld.judge.verdict import (
    MODE_BOTH,
    MODE_CHECKER,
    MODE_LLM,
    JudgeConfig,
    judge_rollout,
    states_for,
)
from qwen_agentworld.llm_clients.base import ChatResult


class FakeJudge:
    """Replies from a scripted list, recording what it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        return ChatResult(content=self.replies.pop(0) if self.replies else "")


def make_task(predicate="state['status'] == 'done'", prompt="Mark the report as done."):
    return Task(
        tool_family="mcp_notes",
        task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="update_note")]),
        natural_language_prompt=prompt,
        initial_state={"status": "pending"},
        checker=CheckerSpec(executable_predicate=predicate),
        difficulty_meta=DifficultyMeta(graph_complexity=1),
    )


def make_trajectory(task, next_states):
    return Trajectory(
        task_id=task.task_id,
        playbook_version="1",
        steps=[
            Step(
                tool_call=ToolCall(tool_name="update_note", arguments={"i": i}),
                simulator_raw_output={"next_state": ns},
            )
            for i, ns in enumerate(next_states)
        ],
    )


# --------------------------------------------------------------- llm_judge


def test_score_and_threshold():
    judge = FakeJudge([json.dumps({"score": 0.75, "reason": "two of three parts done"})])
    v = judge_with_llm(judge, make_task(), {"status": "done"}, threshold=0.5)
    assert v.score == 0.75
    assert v.passed is True
    assert v.error is None

    judge = FakeJudge([json.dumps({"score": 0.4, "reason": "no"})])
    assert judge_with_llm(judge, make_task(), {}, threshold=0.5).passed is False
    # The threshold is ours to move; the same score flips with a lower bar.
    judge = FakeJudge([json.dumps({"score": 0.4, "reason": "no"})])
    assert judge_with_llm(judge, make_task(), {}, threshold=0.3).passed is True


def test_retries_an_unparseable_reply_then_succeeds():
    judge = FakeJudge(["I think it went well!", json.dumps({"score": 1.0, "reason": "ok"})])
    v = judge_with_llm(judge, make_task(), {"status": "done"})
    assert v.score == 1.0
    assert len(judge.calls) == 2


def test_persistently_unparseable_reply_records_an_error_instead_of_raising():
    """Our own failure to produce a verdict must not read as a low pass rate --
    the same conflation `judge_checker_with_reason` exists to prevent."""
    judge = FakeJudge(["nope", "still nope", "nope again"])
    v = judge_with_llm(judge, make_task(), {"status": "done"})
    assert v.passed is False
    assert v.error.startswith("judge_unparseable:")  # JSONDecodeError, a ValueError subclass


def test_out_of_range_score_is_rejected():
    judge = FakeJudge([json.dumps({"score": 7, "reason": "x"}), json.dumps({"score": 1.0, "reason": "y"})])
    assert judge_with_llm(judge, make_task(), {}).score == 1.0


def test_prompt_carries_instruction_states_and_tool_calls():
    task = make_task(prompt="Tag the meeting note as urgent.")
    traj = make_trajectory(task, [{"status": "done"}])
    text = build_judge_prompt(task, {"status": "done"}, traj)
    assert "Tag the meeting note as urgent." in text
    assert "pending" in text  # before
    assert "update_note" in text  # what the agent did


def test_prompt_says_so_when_there_were_no_tool_calls():
    text = build_judge_prompt(make_task(), {"status": "pending"}, None)
    assert "no tool calls" in text


# ----------------------------------------------------------------- verdict


def test_states_for_is_initial_plus_one_per_step():
    task = make_task()
    traj = make_trajectory(task, [{"status": "a"}, {"status": "b"}])
    assert states_for(task, traj) == [{"status": "pending"}, {"status": "a"}, {"status": "b"}]
    assert states_for(task, None) == [{"status": "pending"}]


def test_checker_mode_is_unchanged_behaviour_and_never_calls_an_llm():
    v = judge_rollout(make_task(), {"status": "done"}, None, JudgeConfig(mode=MODE_CHECKER))
    assert v.passed is True
    assert v.source == MODE_CHECKER
    assert v.llm_score is None


def test_llm_mode_decides_the_verdict():
    judge = FakeJudge([json.dumps({"score": 0.9, "reason": "done"})])
    # The predicate would say False; the LLM says the job was done anyway.
    v = judge_rollout(
        make_task(predicate="state['status'] == 'DONE'"),
        {"status": "done"},
        None,
        JudgeConfig(mode=MODE_LLM, client=judge),
    )
    assert v.passed is True
    assert v.source == MODE_LLM
    assert v.llm_score == 0.9


def test_both_mode_keeps_the_checker_as_the_verdict_and_records_the_llm():
    """`both` exists to measure agreement. If the LLM could override here, the
    comparison it is for would be unreadable afterwards."""
    judge = FakeJudge([json.dumps({"score": 1.0, "reason": "looks done to me"})])
    v = judge_rollout(
        make_task(predicate="state['status'] == 'DONE'"),
        {"status": "done"},
        None,
        JudgeConfig(mode=MODE_BOTH, client=judge),
    )
    assert v.passed is False
    assert v.source == MODE_CHECKER
    assert v.checker_passed is False
    assert v.llm_score == 1.0


def test_default_config_is_checker_only():
    assert JudgeConfig().mode == MODE_CHECKER
    assert JudgeConfig().uses_llm is False


def test_llm_modes_refuse_to_be_built_without_a_client():
    with pytest.raises(ValueError):
        JudgeConfig(mode=MODE_LLM)
    with pytest.raises(ValueError):
        JudgeConfig(mode=MODE_BOTH)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        JudgeConfig(mode="vibes")


def test_record_is_flat_and_carries_both_readings():
    judge = FakeJudge([json.dumps({"score": 0.6, "reason": "partial"})])
    rec = judge_rollout(make_task(), {"status": "done"}, None,
                        JudgeConfig(mode=MODE_BOTH, client=judge)).record()
    assert rec["verdict"] is True
    assert rec["verdict_source"] == MODE_CHECKER
    assert rec["llm_score"] == 0.6
    json.dumps(rec)  # must survive the per-rollout JSONL writers
