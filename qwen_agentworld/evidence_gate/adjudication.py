"""Layer 4 (triggered, not default): when schema/agreement/counterfactual
signals land in the ambiguous middle band, ask an independent Teacher call to
adjudicate whether the simulator's output is plausible.

This is deliberately *not* the same call that generated the task/checker for
this trajectory — see the "Claude both sets the question and grades it"
review objection addressed in data/simulator_playbook_research_plan.md.
Callers should pass a fresh TeacherClient instance for this purpose.
"""

from __future__ import annotations

import json

from qwen_agentworld.llm_clients.base import ChatResult, LLMClient

_ADJUDICATION_SYSTEM_PROMPT = (
    "You are adjudicating whether a simulated tool-execution result is plausible, "
    "given the tool call that produced it. You did not write the task or the checker "
    "for this trajectory. Judge only physical/logical plausibility and internal "
    "consistency with the given prior state — not whether it helps any particular task. "
    "Reply with exactly one line: ACCEPT or REJECT, optionally followed by a short reason."
)


def build_adjudication_prompt(prior_state: dict, tool_name: str, arguments: dict, candidate_output: dict) -> str:
    return (
        f"Prior state:\n{json.dumps(prior_state, sort_keys=True)}\n\n"
        f"Tool call: {tool_name}({json.dumps(arguments, sort_keys=True)})\n\n"
        f"Candidate simulated output:\n{json.dumps(candidate_output, sort_keys=True)}"
    )


def adjudicate(
    teacher: LLMClient,
    prior_state: dict,
    tool_name: str,
    arguments: dict,
    candidate_output: dict,
) -> bool:
    result: ChatResult = teacher.chat(
        messages=[
            {"role": "system", "content": _ADJUDICATION_SYSTEM_PROMPT},
            {"role": "user", "content": build_adjudication_prompt(prior_state, tool_name, arguments, candidate_output)},
        ],
        max_tokens=50,
    )
    content = (result.content or "").strip().upper()
    return content.startswith("ACCEPT")
