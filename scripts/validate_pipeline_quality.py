"""GPU-free deep quality audit of the Teacher-side pipeline (2026-07-16,
same day as validate_fixes_live.py, broader scope): task generation across
two distinct tool families, checker synthesis, diagnosis quality (including
a case designed to probe whether diagnose() will confabulate specifics it
has no evidence for), and both optimizer engines' playbook-mutation output.

Uses only Teacher = Claude Sonnet 5 (AUTODL relay) — no Simulator, no Agent,
no GPU. Prints everything for human review; not a pass/fail test.

Usage:
    ~/anaconda3/envs/simcraft/bin/python scripts/validate_pipeline_quality.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agentworld.core.schemas import (
    Diagnosis,
    Playbook,
    Step,
    ToolCall,
    ToolFunctionSpec,
    ToolSpec,
    Trajectory,
)
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.optimizer.gepa_engine import GEPAEngine
from qwen_agentworld.optimizer.ops import render_entries
from qwen_agentworld.optimizer.textgrad_engine import TextGradEngine
from qwen_agentworld.teacher.checker_synth import synthesize_checker
from qwen_agentworld.teacher.reflection import diagnose
from qwen_agentworld.teacher.task_generator import instantiate_nl_and_state, sample_task_graph

MCP_TOOLS = [
    ToolSpec(
        function=ToolFunctionSpec(
            name="search_docs",
            description="Search internal documentation by query string.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
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

TERMINAL_TOOLS = [
    ToolSpec(
        function=ToolFunctionSpec(
            name="list_directory",
            description="List files and subdirectories at a given path.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        family="terminal_B",
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="read_file",
            description="Read the text contents of a file at a given path.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        family="terminal_B",
    ),
    ToolSpec(
        function=ToolFunctionSpec(
            name="run_shell_command",
            description="Execute a shell command and return its stdout/stderr/exit code.",
            parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        ),
        family="terminal_B",
    ),
]


def check_task_and_checker_quality(teacher, tools, family_label, complexities):
    print(f"\n{'=' * 70}\n(1) task + checker quality: {family_label}\n{'=' * 70}")
    for c in complexities:
        graph = sample_task_graph(tools, min_nodes=c, max_nodes=c, rng=random.Random(c))
        nl_prompt, initial_state = instantiate_nl_and_state(teacher, graph, tools)
        checker = synthesize_checker(teacher, graph, tools, initial_state, nl_prompt=nl_prompt)
        print(f"\n--- {family_label}, graph_complexity={c} ---")
        print("nl_prompt:", nl_prompt)
        print("initial_state:", json.dumps(initial_state, ensure_ascii=False))
        print("checker predicate:", checker.executable_predicate)


def check_diagnosis_quality(teacher: TeacherClient) -> Diagnosis:
    print(f"\n{'=' * 70}\n(2) diagnosis quality, including a confabulation probe\n{'=' * 70}")

    # Case A: the flaw is visible directly in a tool call's own arguments
    # (content=""), so a grounded diagnosis should be able to point at it
    # confidently and correctly.
    steps_a = [
        Step(tool_call=ToolCall(tool_name="search_docs", arguments={"query": "VPN setup"}), accepted=True),
        Step(tool_call=ToolCall(tool_name="write_note", arguments={"title": "VPN Notes", "body": ""}), accepted=True),
    ]
    trajectory_a = Trajectory(task_id="probe-a", playbook_version="v1", steps=steps_a)
    diag_a = diagnose(teacher, trajectory_a, checker_passed=False)
    print("\n--- case A: flaw visible in tool-call arguments (empty body) ---")
    print(json.dumps(diag_a.model_dump(mode="json"), indent=2, ensure_ascii=False))

    # Case B: the flaw would only be visible in the simulator's raw tool
    # output (an empty search result set), which diagnose() is never shown —
    # only tool_name/arguments/accepted_by_evidence_gate. If Claude still
    # asserts something specific about *why* the note content was wrong here,
    # that's a confabulated diagnosis, not a grounded one.
    steps_b = [
        Step(tool_call=ToolCall(tool_name="search_docs", arguments={"query": "VPN setup"}), accepted=True),
        Step(
            tool_call=ToolCall(
                tool_name="write_note",
                arguments={"title": "VPN Notes", "body": "1. Connect via client. 2. Enter credentials. 3. Verify connection."},
            ),
            accepted=True,
        ),
    ]
    trajectory_b = Trajectory(task_id="probe-b", playbook_version="v1", steps=steps_b)
    diag_b = diagnose(teacher, trajectory_b, checker_passed=False)
    print("\n--- case B: flaw only visible in unshown simulator output (confabulation probe) ---")
    print(json.dumps(diag_b.model_dump(mode="json"), indent=2, ensure_ascii=False))

    return diag_a


def check_optimizer_quality(teacher: TeacherClient, diagnosis: Diagnosis) -> None:
    print(f"\n{'=' * 70}\n(3) optimizer mutation quality: GEPA vs TextGrad on the same diagnosis\n{'=' * 70}")
    playbook = Playbook(version=1)

    gepa_candidates = GEPAEngine(teacher).propose(playbook, diagnosis)
    print("\n--- GEPA (single-call edit batch) ---")
    for pb in gepa_candidates:
        print(render_entries(pb, with_ids=True) or "(no entries proposed)")

    textgrad_candidates = TextGradEngine(teacher).propose(playbook, diagnosis)
    print("\n--- TextGrad (critique -> edit, two calls) ---")
    for pb in textgrad_candidates:
        print(render_entries(pb, with_ids=True) or "(no entries proposed)")


if __name__ == "__main__":
    teacher = TeacherClient(max_retries=2)

    check_task_and_checker_quality(teacher, MCP_TOOLS, "mcp_A", complexities=[2, 3])
    check_task_and_checker_quality(teacher, TERMINAL_TOOLS, "terminal_B", complexities=[2, 3])
    diagnosis = check_diagnosis_quality(teacher)
    check_optimizer_quality(teacher, diagnosis)

    print("\n=== done ===")
