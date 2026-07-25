import json
from unittest.mock import MagicMock

from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    Playbook,
    PlaybookCategory,
    PlaybookModule,
    Task,
    TaskGraph,
    TaskGraphNode,
    ToolFunctionSpec,
    ToolSpec,
)
from qwen_agentworld.llm_clients.base import ChatResult, ToolCallResult
from qwen_agentworld.simulator_gym.env import rollout, simulate_next_state


def tool(name, family="mcp_A"):
    return ToolSpec(
        function=ToolFunctionSpec(name=name, description=f"{name} tool", parameters={"type": "object"}),
        family=family,
    )


def make_task(graph_complexity=1) -> Task:
    return Task(
        tool_family="mcp_A",
        task_graph=TaskGraph(nodes=[TaskGraphNode(node_id="n1", tool_name="search_docs")]),
        natural_language_prompt="find X",
        initial_state={"docs": []},
        checker=CheckerSpec(executable_predicate="True"),
        difficulty_meta=DifficultyMeta(graph_complexity=graph_complexity),
    )


def agent_call(tool_name="search_docs", arguments='{"query": "X"}', call_id="call_1"):
    return ChatResult(content=None, tool_calls=[ToolCallResult(id=call_id, name=tool_name, arguments=arguments)])


def stop_call():
    return ChatResult(content="done", tool_calls=[])


def simulator_reply(next_state: dict) -> ChatResult:
    return ChatResult(content=json.dumps({"next_state": next_state}))


def test_simulate_next_state_extracts_next_state_from_json_reply():
    simulator = MagicMock()
    simulator.chat.return_value = simulator_reply({"docs": ["X"]})
    result = simulate_next_state(simulator, {"docs": []}, tool_call=MagicMock(tool_name="search_docs", arguments={}))
    assert result == {"docs": ["X"]}


def test_rollout_records_one_step_with_prior_and_next_state():
    agent = MagicMock()
    agent.chat.side_effect = [agent_call(), stop_call()]
    simulator = MagicMock()
    simulator.chat.return_value = simulator_reply({"docs": ["X"]})

    task = make_task()
    trajectory, final_state = rollout(agent, simulator, task, tools=[tool("search_docs")])

    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.tool_call.tool_name == "search_docs"
    assert step.tool_call.arguments == {"query": "X"}
    assert step.simulator_raw_output["prior_state"] == {"docs": []}
    assert step.simulator_raw_output["next_state"] == {"docs": ["X"]}
    assert step.evidence is None  # scoring is evidence_gate's job, not rollout's
    assert final_state == {"docs": ["X"]}
    assert trajectory.task_id == task.task_id
    assert trajectory.playbook_version == "none"


def test_rollout_stops_at_max_steps_when_agent_never_yields():
    agent = MagicMock()
    agent.chat.return_value = agent_call()  # always wants to call a tool
    simulator = MagicMock()
    simulator.chat.return_value = simulator_reply({"docs": ["X"]})

    trajectory, _ = rollout(agent, simulator, make_task(), tools=[tool("search_docs")], max_steps=3)

    assert len(trajectory.steps) == 3


def test_rollout_injects_playbook_content_into_agent_system_prompt():
    agent = MagicMock()
    agent.chat.side_effect = [stop_call()]
    simulator = MagicMock()

    playbook = Playbook(
        version=5,
        modules={
            PlaybookCategory.PRECONDITION_CHECK: PlaybookModule(
                category=PlaybookCategory.PRECONDITION_CHECK, content="always confirm the query is non-empty"
            )
        },
    )

    trajectory, _ = rollout(agent, simulator, make_task(), tools=[tool("search_docs")], playbook=playbook)

    system_message = agent.chat.call_args_list[0].kwargs["messages"][0]
    assert system_message["role"] == "system"
    assert "always confirm the query is non-empty" in system_message["content"]
    assert trajectory.playbook_version == "5"
