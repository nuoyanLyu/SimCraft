"""Step-wise diagnosis of a completed trajectory against its checker outcome.

This feeds the optimizer's playbook edits (optimizer/*): not just pass/fail,
but *which step* went wrong, what the reusable lesson is, and — crucially —
which existing playbook entries were followed, helpfully or not.

The category vocabulary used to be closed: the prompt listed five names and
told the teacher to pick the one "most responsible". That made the taxonomy
an input to diagnosis rather than an output of it. Anything that did not fit
one of the five was filed under the nearest neighbour, and no run could ever
surface a kind of mistake the designers had not thought of first. Now the
teacher names the failure mode in its own words (`suggested_tag`), and the
tag set of a run is whatever the run produced.
"""

from __future__ import annotations

import json

from qwen_agentworld.core.schemas import Diagnosis, Playbook, StepDiagnosis, Trajectory
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.optimizer.ops import render_entries

_REFLECTION_SYSTEM_PROMPT = (
    "You are diagnosing a completed tool-use trajectory. You are given the sequence of tool "
    "calls actually made, whether an automated checker judged the final state as passing, and "
    "the guidance entries the agent had in context. "
    "For each step, decide if it was correct, suboptimal (recoverable but not ideal), or "
    "erroneous, and give a one-sentence reason. "
    "If a step was not fully correct, also give `suggested_tag`: a short kebab-case label you "
    "choose yourself for the *kind* of mistake it is (for example 'unverified-precondition' or "
    "'stale-identifier-reuse'). There is no fixed list — reuse a tag already present in the "
    "guidance when it genuinely fits, and invent a new one when it does not. Describe the "
    "mistake, not the tool involved. "
    "Then, for the guidance entries you were shown, list the ids the agent visibly followed to "
    "its benefit (`helpful_entry_ids`) and the ids it followed into a mistake "
    "(`harmful_entry_ids`). Leave both empty if neither applies. "
    'Reply with a single JSON object: {"overall_verdict": "success|partial|failure", '
    '"summary": "...", "steps": [{"step_id": "...", "verdict": "correct|suboptimal|erroneous", '
    '"feedback": "...", "suggested_tag": "<short-label>"|null}, ...], '
    '"helpful_entry_ids": [...], "harmful_entry_ids": [...]}.'
)


_MAX_DIAGNOSIS_ATTEMPTS = 3


def _build_reflection_prompt(trajectory: Trajectory, checker_passed: bool, playbook: Playbook | None) -> str:
    steps_summary = [
        {
            "step_id": s.step_id,
            "tool_name": s.tool_call.tool_name,
            "arguments": s.tool_call.arguments,
            "accepted_by_evidence_gate": s.accepted,
        }
        for s in trajectory.steps
    ]
    parts = [f"checker_passed: {checker_passed}", f"steps:\n{json.dumps(steps_summary, indent=2)}"]
    # The entry ids have to be visible here or `helpful_entry_ids` cannot be
    # produced at all — credit assignment needs the teacher to be able to name
    # what it is crediting.
    guidance = render_entries(playbook, with_ids=True) if playbook is not None else ""
    parts.append(
        f"guidance the agent had in context:\n{guidance}"
        if guidance
        else "guidance the agent had in context: (none — the playbook is empty)"
    )
    return "\n\n".join(parts)


def _parse_diagnosis(trajectory: Trajectory, content: str) -> Diagnosis:
    payload = extract_json_object(content)
    step_diagnoses = [
        StepDiagnosis(
            step_id=s["step_id"],
            verdict=s["verdict"],
            feedback=s["feedback"],
            suggested_tag=s.get("suggested_tag") or None,
        )
        for s in payload.get("steps", [])
    ]
    return Diagnosis(
        task_id=trajectory.task_id,
        overall_verdict=payload["overall_verdict"],
        step_diagnoses=step_diagnoses,
        summary=payload.get("summary", ""),
        helpful_entry_ids=list(payload.get("helpful_entry_ids") or []),
        harmful_entry_ids=list(payload.get("harmful_entry_ids") or []),
    )


def diagnose(
    teacher: LLMClient,
    trajectory: Trajectory,
    checker_passed: bool,
    playbook: Playbook | None = None,
    max_attempts: int = _MAX_DIAGNOSIS_ATTEMPTS,
) -> Diagnosis:
    messages = [
        {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_reflection_prompt(trajectory, checker_passed, playbook)},
    ]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        result = teacher.chat(messages=messages, max_tokens=1000)
        try:
            return _parse_diagnosis(trajectory, result.content or "")
        except (ValueError, KeyError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": result.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "That reply was empty or not valid JSON. Reply again with the JSON "
                    "object described in the system prompt.",
                }
            )
    raise last_error
