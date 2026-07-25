"""Agent rollout against the simulated environment (D2: the simulator is the
"practice gym" the agent tries and fails in — its raw output is never trusted
on its own; `evidence_gate` scores every step before anything downstream
treats it as real).

Two separate LLMClient roles drive one rollout: `agent` picks tool calls
given the task + the current playbook injected as guidance; `simulator`
predicts the next canonical_state given the prior state and the tool call
the agent made. Neither is assumed live here — both are passed in, so tests
and Stage-4-blocked callers can inject mocks.
"""

from __future__ import annotations

import json

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Playbook, Step, Task, ToolCall, ToolSpec, Trajectory
from qwen_agentworld.llm_clients.base import LLMClient

_SIMULATOR_SYSTEM_PROMPT = (
    "You simulate the effect of one tool call on an environment's canonical state. "
    "Given the current state and a tool call (name + arguments), predict the resulting state. "
    'Reply with a single JSON object: {"next_state": {...}}. Do not explain your reasoning.'
)


def _build_playbook_context(playbook: Playbook | None) -> str:
    if playbook is None or not playbook.modules:
        return ""
    sections = [f"[{module.category.value}]\n{module.content}" for module in playbook.modules.values()]
    return "Guidance from prior experience:\n" + "\n\n".join(sections)


def _build_agent_system_prompt(task: Task, playbook: Playbook | None) -> str:
    parts = [
        "You are completing a tool-use task. Call tools step by step; when the task is fully "
        "done, stop calling tools and reply with a short confirmation instead.",
    ]
    context = _build_playbook_context(playbook)
    if context:
        parts.append(context)
    return "\n\n".join(parts)


def _parse_tool_arguments(raw: str | None) -> dict:
    """Best-effort parse of a tool call's JSON arguments.

    A model occasionally emits malformed JSON (a truncated string, a trailing
    token, a missing delimiter). One bad tool call should degrade to empty
    arguments — recorded, scoreable, still wrong if the task needed them — not
    crash the entire rollout and discard every other step.
    """
    if not raw:
        return {}
    try:
        parsed = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def simulate_next_state(simulator: LLMClient, state: dict, tool_call: ToolCall) -> dict:
    prompt = (
        f"Current state:\n{json.dumps(state, indent=2)}\n\n"
        f"Tool call: {tool_call.tool_name}({json.dumps(tool_call.arguments)})\n\n"
        "Produce the JSON object described in the system prompt."
    )
    result = simulator.chat(
        messages=[
            {"role": "system", "content": _SIMULATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
    )
    payload = extract_json_object(result.content or "")
    return payload["next_state"]


def rollout(
    agent: LLMClient,
    simulator: LLMClient,
    task: Task,
    tools: list[ToolSpec],
    playbook: Playbook | None = None,
    max_steps: int = 10,
) -> tuple[Trajectory, dict]:
    """Runs the agent against the simulator for up to `max_steps` tool calls.

    `tools` are the ToolSpec objects for `task.tool_family` (callers look
    these up via `tools/registry.ToolRegistry`; this module doesn't own tool
    lookup, only the rollout loop).

    Returns the raw `Trajectory` (steps have `evidence=None, accepted=False`
    — scoring is `evidence_gate`'s job, not this module's) and the final
    canonical state reached, for the checker to evaluate.
    """
    tools_wire = [t.to_wire() for t in tools]
    messages: list[dict] = [
        {"role": "system", "content": _build_agent_system_prompt(task, playbook)},
        {"role": "user", "content": task.natural_language_prompt},
    ]
    state = dict(task.initial_state)
    trajectory = Trajectory(task_id=task.task_id, playbook_version=str(playbook.version) if playbook else "none")

    for step_idx in range(max_steps):
        result = agent.chat(messages=messages, tools=tools_wire or None)
        if not result.tool_calls:
            break

        # Every tool call must carry a stable, non-null id. vLLM's OpenAI
        # server matches each `tool` response back to its `assistant` tool_call
        # by id; some tool-call parsers emit a null id, which then makes the
        # *next* agent turn raise KeyError('id') when the server reloads the
        # conversation. Synthesize one when missing, and reuse the exact same
        # id for the paired tool response so the link stays well-formed.
        call_ids = [tc.id or f"call_{step_idx}_{j}" for j, tc in enumerate(result.tool_calls)]

        messages.append(
            {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for cid, tc in zip(call_ids, result.tool_calls)
                ],
            }
        )
        for cid, tc in zip(call_ids, result.tool_calls):
            arguments = _parse_tool_arguments(tc.arguments)
            tool_call = ToolCall(tool_name=tc.name, arguments=arguments)
            prior_state = state
            next_state = simulate_next_state(simulator, prior_state, tool_call)
            trajectory.steps.append(
                Step(
                    tool_call=tool_call,
                    simulator_raw_output={"prior_state": prior_state, "next_state": next_state},
                )
            )
            state = next_state
            messages.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(next_state)})

    return trajectory, state
