"""The action log: the only witness a read-only step leaves behind.

Motivated by the 2026-07-29 task audit, where every surviving TOO_WEAK verdict
was a step the state could not observe ("the predicates do not require the agent
to actually perform the search step").
"""

import json
from unittest.mock import MagicMock

from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    Task,
    TaskGraph,
    TaskGraphNode,
    ToolCall,
    ToolFunctionSpec,
    ToolSpec,
)
from qwen_agentworld.llm_clients.base import ChatResult, ToolCallResult
from qwen_agentworld.simulator_gym.env import (
    ACTION_LOG_KEY,
    _domain_state,
    record_action,
    rollout,
    simulate_next_state,
)


def call(name="web_search", **arguments):
    return ToolCall(tool_name=name, arguments=arguments)


# ------------------------------------------------------------ record_action --


def test_a_read_only_call_leaves_a_trace_even_when_the_state_is_unchanged():
    state = {"notebook": []}
    after = record_action(state, {"notebook": []}, call(query="tide tables"))
    assert after[ACTION_LOG_KEY] == [{"tool": "web_search", "arguments": {"query": "tide tables"}}]


def test_the_log_accumulates_in_call_order():
    s0 = {"notebook": []}
    s1 = record_action(s0, {"notebook": []}, call(query="a"))
    s2 = record_action(s1, {"notebook": ["n"]}, call(name="save_note", headline="h"))
    assert [e["tool"] for e in s2[ACTION_LOG_KEY]] == ["web_search", "save_note"]
    assert s2["notebook"] == ["n"]


def test_the_log_is_ours_not_the_simulators():
    """A simulator that invents its own `_action_log` must not be able to
    fabricate steps that were never taken, or the one ground-truth field in the
    state becomes as unreliable as the rest of the prediction."""
    hallucinated = {"notebook": [], ACTION_LOG_KEY: [{"tool": "save_note", "arguments": {}}]}
    after = record_action({"notebook": []}, hallucinated, call(query="a"))
    assert after[ACTION_LOG_KEY] == [{"tool": "web_search", "arguments": {"query": "a"}}]


def test_a_malformed_simulator_reply_still_degrades_rather_than_raising():
    assert record_action({"a": 1}, ["not", "a", "dict"], call()) == ["not", "a", "dict"]


def test_the_simulator_is_not_shown_the_log():
    assert _domain_state({"notebook": [], ACTION_LOG_KEY: [{"tool": "x"}]}) == {"notebook": []}


def test_simulate_next_state_strips_the_log_from_its_prompt():
    simulator = MagicMock()
    simulator.chat.return_value = ChatResult(content=json.dumps({"next_state": {"notebook": []}}))
    simulate_next_state(simulator, {"notebook": [], ACTION_LOG_KEY: [{"tool": "web_search"}]}, call())
    prompt = simulator.chat.call_args.kwargs["messages"][1]["content"]
    assert ACTION_LOG_KEY not in prompt


# -------------------------------------------------------------- in rollout --


def tools():
    return [ToolSpec(function=ToolFunctionSpec(name="web_search", description="d"), family="web_research")]


def task():
    return Task(
        tool_family="web_research",
        task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="web_search")]),
        natural_language_prompt="look up the tide tables",
        initial_state={"notebook": [], ACTION_LOG_KEY: []},
        checker=CheckerSpec(executable_predicate="len(state['notebook']) == 0"),
        difficulty_meta=DifficultyMeta(graph_complexity=1),
    )


def test_a_rollout_of_a_pure_search_produces_a_checkable_state():
    agent = MagicMock()
    agent.chat.side_effect = [
        ChatResult(
            content=None,
            tool_calls=[ToolCallResult(id="c1", name="web_search", arguments='{"query": "tide tables"}')],
        ),
        ChatResult(content="done"),
    ]
    simulator = MagicMock()
    simulator.chat.return_value = ChatResult(content=json.dumps({"next_state": {"notebook": []}}))

    trajectory, final_state = rollout(agent, simulator, task(), tools())

    # The domain state is untouched -- which is exactly why the old end-state
    # predicate could not tell this run from one that skipped the search.
    assert final_state["notebook"] == []
    assert final_state[ACTION_LOG_KEY] == [
        {"tool": "web_search", "arguments": {"query": "tide tables"}}
    ]
    predicate = "any(c['tool'] == 'web_search' for c in state['_action_log'])"
    from qwen_agentworld.teacher.safe_predicate import evaluate_predicate

    assert evaluate_predicate(predicate, final_state)
    assert not evaluate_predicate(predicate, task().initial_state)
    assert trajectory.steps[0].simulator_raw_output["next_state"][ACTION_LOG_KEY]
