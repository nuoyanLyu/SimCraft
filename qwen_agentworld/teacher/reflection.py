"""Step-wise diagnosis of a completed trajectory against its checker outcome.

This feeds the optimizer's playbook mutation (see optimizer/*, not yet
built): not just pass/fail, but *which step* went wrong and *which playbook
category* (schema grounding vs precondition check vs ...) would plausibly
have prevented it, so GEPA/TextGrad-style engines get a specific target to
rewrite instead of a single scalar reward.
"""

from __future__ import annotations

import json

from qwen_agentworld.core.schemas import Diagnosis, PlaybookCategory, StepDiagnosis, Trajectory
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.core.json_utils import extract_json_object

_REFLECTION_SYSTEM_PROMPT = (
    "You are diagnosing a completed tool-use trajectory. You are given the sequence of tool "
    "calls actually made and whether an automated checker judged the final state as passing. "
    "For each step, decide if it was correct, suboptimal (recoverable but not ideal), or "
    "erroneous, give a one-sentence reason, and if it was not fully correct, name the single "
    "playbook category most responsible for the mistake: schema_grounding, precondition_check, "
    "incremental_execution, error_recovery, or postcondition_verification. "
    'Reply with a single JSON object: {"overall_verdict": "success|partial|failure", '
    '"summary": "...", "steps": [{"step_id": "...", "verdict": "correct|suboptimal|erroneous", '
    '"feedback": "...", "suggested_category": "<category>"|null}, ...]}.'
)


_MAX_DIAGNOSIS_ATTEMPTS = 3


def _build_reflection_prompt(trajectory: Trajectory, checker_passed: bool) -> str:
    steps_summary = [
        {
            "step_id": s.step_id,
            "tool_name": s.tool_call.tool_name,
            "arguments": s.tool_call.arguments,
            "accepted_by_evidence_gate": s.accepted,
        }
        for s in trajectory.steps
    ]
    return f"checker_passed: {checker_passed}\n\nsteps:\n{json.dumps(steps_summary, indent=2)}"


def _parse_diagnosis(trajectory: Trajectory, content: str) -> Diagnosis:
    payload = extract_json_object(content)
    step_diagnoses = [
        StepDiagnosis(
            step_id=s["step_id"],
            verdict=s["verdict"],
            feedback=s["feedback"],
            suggested_category=PlaybookCategory(s["suggested_category"]) if s.get("suggested_category") else None,
        )
        for s in payload.get("steps", [])
    ]
    return Diagnosis(
        task_id=trajectory.task_id,
        overall_verdict=payload["overall_verdict"],
        step_diagnoses=step_diagnoses,
        summary=payload.get("summary", ""),
    )


def diagnose(
    teacher: LLMClient,
    trajectory: Trajectory,
    checker_passed: bool,
    max_attempts: int = _MAX_DIAGNOSIS_ATTEMPTS,
) -> Diagnosis:
    messages = [
        {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_reflection_prompt(trajectory, checker_passed)},
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
