"""Real smoke test against the AUTODL Claude relay, exercising the full
task_generator -> checker_synth -> reflection chain on the MCP tool family.
Excluded by default (see conftest.py); run explicitly with `pytest -m live`.
"""

import random

import pytest

from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    Step,
    Task,
    ToolCall,
    ToolFunctionSpec,
    ToolSpec,
    Trajectory,
)
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.teacher.checker_synth import synthesize_checker
from qwen_agentworld.teacher.reflection import diagnose
from qwen_agentworld.teacher.task_generator import instantiate_nl_and_state, sample_task_graph

_MCP_TOOLS = [
    ToolSpec(
        function=ToolFunctionSpec(
            name="search_docs",
            description="Search internal documentation by query string.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        family="mcp_A",
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="write_note",
            description="Save a note with a title and body to the workspace.",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                "required": ["title", "body"],
            },
        ),
        family="mcp_A",
    ),
]


@pytest.mark.live
def test_full_teacher_pipeline_on_mcp_domain():
    teacher = TeacherClient(max_retries=2)

    graph = sample_task_graph(_MCP_TOOLS, min_nodes=2, max_nodes=2, rng=random.Random(0))
    nl_prompt, initial_state = instantiate_nl_and_state(teacher, graph, _MCP_TOOLS)
    assert isinstance(nl_prompt, str) and nl_prompt
    assert isinstance(initial_state, dict)

    checker = synthesize_checker(teacher, graph, _MCP_TOOLS, initial_state, nl_prompt=nl_prompt)
    assert isinstance(checker, CheckerSpec)
    assert checker.executable_predicate

    task = Task(
        tool_family="mcp_A",
        task_graph=graph,
        natural_language_prompt=nl_prompt,
        initial_state=initial_state,
        checker=checker,
        difficulty_meta=DifficultyMeta(graph_complexity=len(graph.nodes)),
    )

    trajectory = Trajectory(task_id=task.task_id, playbook_version="v1")
    trajectory.steps.append(
        Step(tool_call=ToolCall(tool_name="search_docs", arguments={"query": "example"}), accepted=True)
    )
    trajectory.steps.append(
        Step(tool_call=ToolCall(tool_name="write_note", arguments={}), accepted=False)
    )

    diagnosis = diagnose(teacher, trajectory, checker_passed=False)
    assert diagnosis.task_id == task.task_id
    assert diagnosis.overall_verdict in {"success", "partial", "failure"}
    assert len(diagnosis.step_diagnoses) == len(trajectory.steps)
