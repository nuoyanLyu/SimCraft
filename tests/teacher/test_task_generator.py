import random
from unittest.mock import MagicMock

import pytest

from qwen_agentworld.core.schemas import ToolFunctionSpec, ToolSpec
from qwen_agentworld.llm_clients.base import ChatResult
from qwen_agentworld.teacher.task_generator import (
    _INSTANTIATION_SYSTEM_PROMPT,
    instantiate_nl_and_state,
    sample_task_graph,
)


def make_tool(name: str) -> ToolSpec:
    return ToolSpec(
        function=ToolFunctionSpec(name=name, description=f"do {name}", parameters={"type": "object", "properties": {}}),
        family="mcp_A",
    )


def test_sample_task_graph_is_a_linear_chain_over_distinct_tools():
    tools = [make_tool("search_docs"), make_tool("write_note"), make_tool("read_note")]
    graph = sample_task_graph(tools, min_nodes=3, max_nodes=3, rng=random.Random(0))
    assert len(graph.nodes) == 3
    assert graph.nodes[0].depends_on == []
    assert graph.nodes[1].depends_on == [graph.nodes[0].node_id]
    assert graph.nodes[2].depends_on == [graph.nodes[1].node_id]
    assert len({n.tool_name for n in graph.nodes}) == 3


def test_sample_task_graph_rejects_empty_tools():
    with pytest.raises(ValueError):
        sample_task_graph([], rng=random.Random(0))


def test_sample_task_graph_caps_node_count_to_available_tools():
    tools = [make_tool("only_tool")]
    graph = sample_task_graph(tools, min_nodes=1, max_nodes=5, rng=random.Random(0))
    assert len(graph.nodes) == 1


def test_instantiate_nl_and_state_parses_json_reply():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content='{"natural_language_prompt": "Find the doc and save a note.", "initial_state": {"notes": []}}'
    )
    tools = [make_tool("search_docs")]
    graph = sample_task_graph(tools, min_nodes=1, max_nodes=1, rng=random.Random(0))

    prompt, state = instantiate_nl_and_state(teacher, graph, tools)
    assert prompt == "Find the doc and save a note."
    assert state == {"notes": []}


def test_instantiate_nl_and_state_tolerates_code_fence():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(
        content='```json\n{"natural_language_prompt": "p", "initial_state": {}}\n```'
    )
    tools = [make_tool("search_docs")]
    graph = sample_task_graph(tools, min_nodes=1, max_nodes=1, rng=random.Random(0))

    prompt, state = instantiate_nl_and_state(teacher, graph, tools)
    assert prompt == "p"
    assert state == {}


def test_instantiate_nl_and_state_retries_on_empty_content_then_succeeds():
    teacher = MagicMock()
    teacher.chat.side_effect = [
        ChatResult(content=""),
        ChatResult(content='{"natural_language_prompt": "p", "initial_state": {}}'),
    ]
    tools = [make_tool("search_docs")]
    graph = sample_task_graph(tools, min_nodes=1, max_nodes=1, rng=random.Random(0))

    prompt, state = instantiate_nl_and_state(teacher, graph, tools)
    assert prompt == "p"
    assert state == {}
    assert teacher.chat.call_count == 2


def test_instantiate_nl_and_state_raises_after_exhausting_retries():
    teacher = MagicMock()
    teacher.chat.return_value = ChatResult(content="")
    tools = [make_tool("search_docs")]
    graph = sample_task_graph(tools, min_nodes=1, max_nodes=1, rng=random.Random(0))

    with pytest.raises(ValueError):
        instantiate_nl_and_state(teacher, graph, tools, max_attempts=3)
    assert teacher.chat.call_count == 3


def test_instantiation_system_prompt_grounds_tasks_in_simulator_domains():
    for domain in ["MCP", "Search", "SWE", "Terminal", "Android", "Web", "OS"]:
        assert domain in _INSTANTIATION_SYSTEM_PROMPT
    assert "goal-directed" in _INSTANTIATION_SYSTEM_PROMPT
