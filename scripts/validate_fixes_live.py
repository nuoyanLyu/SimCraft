"""GPU-free live validation of three fixes (2026-07-16):
1) instantiate_nl_and_state now retries on empty/malformed content instead
   of crashing on the first bad reply.
2) _INSTANTIATION_SYSTEM_PROMPT now grounds task generation in the
   simulator's 7 trained domains (MCP/Search/SWE/Terminal/Android/Web/OS)
   and asks for genuinely agentic, goal-directed tasks.
3) score_trajectory now actually engages the counterfactual-replay leg of
   the evidence gate (previously always trivially passing).

Uses only Teacher=Claude and Simulator=Claude (both via the AUTODL relay) —
no GPU/vLLM required, so this runs even while GPU1 is fully occupied.

Usage:
    ~/anaconda3/envs/simcraft/bin/python scripts/validate_fixes_live.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_agentworld.core.schemas import ToolCall, ToolFunctionSpec, ToolSpec
from qwen_agentworld.evidence_gate.counterfactual_replay import build_counterfactual_probe
from qwen_agentworld.evidence_gate.gate import EvidenceGate
from qwen_agentworld.llm_clients.simulator_temp_claude import TemporarySimulatorClient
from qwen_agentworld.llm_clients.teacher_claude import TeacherClient
from qwen_agentworld.simulator_gym.env import simulate_next_state
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


def check_1_and_2_task_generation(teacher: TeacherClient, n: int) -> None:
    print(f"\n=== (1)+(2) instantiate_nl_and_state, {n} runs, domain-grounded prompt ===")
    for i in range(n):
        graph = sample_task_graph(MCP_TOOLS, min_nodes=2, max_nodes=2, rng=random.Random(i))
        nl_prompt, initial_state = instantiate_nl_and_state(teacher, graph, MCP_TOOLS)
        print(f"\n--- run {i + 1} ---")
        print("nl_prompt:", nl_prompt)
        print("initial_state:", json.dumps(initial_state, ensure_ascii=False))


def check_3_counterfactual_leg(simulator: TemporarySimulatorClient) -> None:
    print("\n=== (3) counterfactual-replay leg, real Simulator ===")
    prior_state = {"notes": [{"title": "shopping", "content": "milk, eggs"}], "cwd": "/home/user"}
    tool_call = ToolCall(tool_name="write_note", arguments={"title": "todo", "content": "call dentist"})

    next_state = simulate_next_state(simulator, prior_state, tool_call)
    print("prior_state:", json.dumps(prior_state, ensure_ascii=False))
    print("next_state:", json.dumps(next_state, ensure_ascii=False))

    probe = build_counterfactual_probe(prior_state, next_state)
    if probe is None:
        print("build_counterfactual_probe returned None (no safe untouched key found) -- unexpected for this fixture")
        return
    perturbed_prior_state, invariant_fields = probe
    print("invariant_fields:", invariant_fields)
    print("perturbed_prior_state:", json.dumps(perturbed_prior_state, ensure_ascii=False))

    counterfactual_output = simulate_next_state(simulator, perturbed_prior_state, tool_call)
    print("counterfactual_output:", json.dumps(counterfactual_output, ensure_ascii=False))

    gate = EvidenceGate()
    evidence = gate.score(
        candidate_output=next_state,
        response_schema=None,
        agreement_samples=[next_state],
        counterfactual_output=counterfactual_output,
        invariant_fields=invariant_fields,
    )
    print("counterfactual_pass:", evidence.counterfactual_pass, "| confidence:", evidence.confidence)


if __name__ == "__main__":
    teacher = TeacherClient(max_retries=2)
    simulator = TemporarySimulatorClient()

    check_1_and_2_task_generation(teacher, n=3)
    check_3_counterfactual_leg(simulator)
    print("\n=== done ===")
