import pytest
from pydantic import ValidationError

from qwen_agentworld.core.schemas import (
    CheckerSpec,
    DifficultyMeta,
    EvidenceScore,
    Playbook,
    PlaybookEntry,
    Step,
    Task,
    TaskGraph,
    TaskGraphNode,
    ToolCall,
    ToolFunctionSpec,
    ToolSpec,
    Trajectory,
)


def make_tool_spec(name="search_docs", family="mcp_A") -> ToolSpec:
    return ToolSpec(
        function=ToolFunctionSpec(
            name=name,
            description="Search internal docs by query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        family=family,
    )


def test_tool_spec_to_wire_strips_family():
    spec = make_tool_spec()
    wire = spec.to_wire()
    assert "family" not in wire
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "search_docs"
    assert set(wire.keys()) == {"type", "function"}


def test_tool_spec_name_property():
    spec = make_tool_spec(name="run_command")
    assert spec.name == "run_command"


def make_task(tool_family="mcp_A") -> Task:
    graph = TaskGraph(
        nodes=[
            TaskGraphNode(node_id="n1", tool_name="search_docs"),
            TaskGraphNode(node_id="n2", tool_name="write_note", depends_on=["n1"]),
        ]
    )
    checker = CheckerSpec(executable_predicate="state.notes | length > 0")
    return Task(
        tool_family=tool_family,
        task_graph=graph,
        natural_language_prompt="Find the doc and save a note about it.",
        initial_state={"notes": []},
        checker=checker,
        difficulty_meta=DifficultyMeta(graph_complexity=2),
    )


def test_task_graph_rejects_empty_nodes():
    with pytest.raises(ValidationError):
        TaskGraph(nodes=[])


def test_task_round_trips_through_json():
    task = make_task()
    restored = Task.model_validate_json(task.model_dump_json())
    assert restored.task_id == task.task_id
    assert restored.task_graph.nodes[1].depends_on == ["n1"]
    assert restored.difficulty_meta.target_pass_rate_band == (0.2, 0.6)


def test_evidence_score_confidence_bounds():
    with pytest.raises(ValidationError):
        EvidenceScore(schema_valid=True, agreement_score=0.5, counterfactual_pass=True, confidence=1.5)


def test_trajectory_accumulates_steps_with_evidence():
    traj = Trajectory(task_id="task_x", playbook_version="v1")
    evidence = EvidenceScore(schema_valid=True, agreement_score=0.8, counterfactual_pass=True, confidence=0.75)
    step = Step(tool_call=ToolCall(tool_name="search_docs", arguments={"query": "foo"}), evidence=evidence, accepted=True)
    traj.steps.append(step)
    assert len(traj.steps) == 1
    assert traj.steps[0].evidence.confidence == 0.75


def test_playbook_entry_defaults():
    entry = PlaybookEntry(tag="postcondition-verification", content="Always re-check state after a write.")
    assert entry.version == 1
    assert entry.stats.helpful == entry.stats.harmful == 0
    assert entry.provenance == []


def test_tags_are_normalized_so_one_lesson_is_one_group():
    """The tag vocabulary is invented per call, so the same idea arrives spelled
    three ways; unnormalized they would render as three separate groups."""
    variants = ["Schema Grounding", "schema_grounding", "  SCHEMA-grounding "]
    assert {PlaybookEntry(tag=v, content="x").tag for v in variants} == {"schema-grounding"}


def test_an_unusable_tag_falls_back_rather_than_raising():
    assert PlaybookEntry(tag="!!!", content="x").tag == "general"


def test_playbook_groups_entries_by_tag_in_first_appearance_order():
    pb = Playbook(entries=[
        PlaybookEntry(tag="b", content="one"),
        PlaybookEntry(tag="a", content="two"),
        PlaybookEntry(tag="b", content="three"),
    ])
    assert pb.tags() == ["b", "a"]
    assert [e.content for e in pb.entries_by_tag("b")] == ["one", "three"]


def test_playbook_has_no_fixed_set_of_tags():
    """The point of the redesign: a lesson that fits no predefined category is
    still representable, because there are no predefined categories."""
    pb = Playbook(entries=[PlaybookEntry(tag="reuse-ids-returned-by-the-tool", content="x")])
    assert pb.tags() == ["reuse-ids-returned-by-the-tool"]
