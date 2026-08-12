"""Score a rollout with an LLM instead of an executable predicate.

The synthesized checker is a Python expression the teacher writes blind, and
the 2026-08-07 diagnosis measured what that costs. Of 24 tasks the agent never
passed, 8 were not hard — they were misjudged:

  * 3 had checkers that were syntactically wrong (`all(a, b, c)` instead of
    `all([a, b, c])`), so every evaluation raised TypeError and scored
    not-passed;
  * 5 satisfied every clause describing the actual job and failed only on
    clauses asserting that untouched parts of the state were byte-identical —
    "preservation" clauses the task text never asked for.

Across failing rollouts, 34% of broken clauses in the floor group were
preservation clauses and 10% were unevaluable, i.e. 44% of floor failures were
not about capability at all. Debugging synthesized predicates one at a time
does not converge; the generator writes new ones faster than they get fixed.

So this module offers the other option: hand the instruction, the starting
state, the tool calls and the resulting state to an LLM and ask for a score in
[0, 1]. It fixes both failure modes structurally rather than case by case — a
judge cannot have a syntax error, and it is told explicitly to ignore
incidental changes nobody asked for.

What it gives up is worth naming, because it is the reason this is a *setting*
and not a replacement. The predicate is deterministic, free, and cannot be
talked out of its verdict; the judge is none of those. It costs one teacher
call per rollout (screening 152 tasks x 4 reps = 608 calls), it will disagree
with itself across runs, and it is the same model family that wrote the task,
so it can forgive its own ambiguous phrasing. Run `mode="both"` first and read
the agreement rate before trusting it alone — see `judge/verdict.py`.

The judge is asked for a continuous score rather than a verdict because the
threshold is then ours to set and to move: a 0/1 answer bakes in a cutoff we
cannot inspect, and partial credit is exactly the signal a pass rate pinned at
0.00 or 1.00 was missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Task, Trajectory
from qwen_agentworld.llm_clients.base import LLMClient

logger = logging.getLogger(__name__)

# Measured, not guessed. Replaying 204 recorded rollouts through this judge
# (scripts/audit_llm_judge.py, 2026-08-07) produced a coarse score distribution
# clustered on 0.0 / 0.5 / 1.0, so the choice is really "does a half-done task
# count". At 0.5 it does, and the sampled pass rate goes 0.451 -> 0.941 with no
# task left at 0.00 -- which erases the very spread the difficulty band needs.
# 0.7 keeps all 8 known-misjudged tasks rescued, keeps 17 of 20 ceiling tasks,
# and projects the bank's band yield at 18.5% against the checker's 4.6%.
DEFAULT_JUDGE_THRESHOLD = 0.7
_MAX_JUDGE_ATTEMPTS = 3
# Enough state to judge, small enough not to blow the context on a long bank
# entry. A truncated state is flagged in the prompt rather than silently cut,
# so the judge can say it could not tell instead of guessing.
_MAX_STATE_CHARS = 6000


@dataclass
class LLMVerdict:
    score: float
    passed: bool
    reason: str
    raw: str | None = None
    error: str | None = None


_JUDGE_SYSTEM_PROMPT = (
    "You grade whether an AI agent completed a user's request in a simulated tool environment.\n"
    "\n"
    "You are given: the user's instruction, the environment state before the agent acted, the "
    "tool calls the agent made, and the environment state afterwards. Decide how much of the "
    "instruction is satisfied by the final state.\n"
    "\n"
    "Grading rules:\n"
    "1. Grade the final state against the instruction, and nothing else. The tool calls are "
    "evidence of how the state got there; a correct final state reached by an odd route still "
    "scores full marks.\n"
    "2. Judge only what the instruction actually asked for. If the agent also changed, added or "
    "reformatted something the instruction never mentioned, ignore it — that is not an error "
    "unless the instruction explicitly required it to stay unchanged.\n"
    "3. Do not require exact wording. If the instruction did not pin down a title, phrasing, "
    "ordering or format, any reasonable choice is correct.\n"
    "4. Give partial credit. If the instruction has several parts, score roughly the fraction "
    "that is satisfied.\n"
    "5. If the agent made no relevant change, score 0.0.\n"
    "\n"
    "Score anchors: 1.0 = every part of the request is satisfied. 0.5 = about half. 0.0 = none.\n"
    "\n"
    'Reply with a single JSON object and nothing else: {"score": <number between 0 and 1>, '
    '"reason": "<one or two sentences citing the specific state fields you checked>"}'
)


def _clip(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(text) <= _MAX_STATE_CHARS:
        return text
    return text[:_MAX_STATE_CHARS] + f"\n... [truncated, {len(text)} chars total]"


def _tool_calls(trajectory: Trajectory | None) -> list[dict]:
    if trajectory is None:
        return []
    return [
        {"step": i, "tool_name": s.tool_call.tool_name, "arguments": s.tool_call.arguments}
        for i, s in enumerate(trajectory.steps, 1)
    ]


def build_judge_prompt(task: Task, final_state: dict, trajectory: Trajectory | None = None) -> str:
    calls = _tool_calls(trajectory)
    calls_block = _clip(calls) if calls else "(the agent made no tool calls)"
    return (
        f"USER INSTRUCTION:\n{task.natural_language_prompt}\n\n"
        f"ENVIRONMENT STATE BEFORE:\n{_clip(task.initial_state)}\n\n"
        f"TOOL CALLS THE AGENT MADE:\n{calls_block}\n\n"
        f"ENVIRONMENT STATE AFTER:\n{_clip(final_state)}\n\n"
        "Grade it. Reply with the JSON object described above."
    )


def _parse(content: str) -> tuple[float, str]:
    payload = extract_json_object(content)
    score = float(payload["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score {score} outside [0, 1]")
    return score, str(payload.get("reason", ""))


def judge_with_llm(
    judge: LLMClient,
    task: Task,
    final_state: dict,
    trajectory: Trajectory | None = None,
    threshold: float = DEFAULT_JUDGE_THRESHOLD,
    max_attempts: int = _MAX_JUDGE_ATTEMPTS,
) -> LLMVerdict:
    """Ask `judge` for a 0-1 score on this rollout.

    A judge call that never returns usable JSON yields `score=0.0` with `error`
    set, matching how a raised predicate is handled: our own failure to produce
    a verdict is never allowed to crash a sweep. It is recorded rather than
    swallowed, because a run where 30% of judgments failed to parse is a broken
    run, not a run with a low pass rate — the same conflation
    `judge_checker_with_reason` exists to prevent on the predicate side.
    """
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(task, final_state, trajectory)},
    ]
    last_error: Exception | None = None
    last_content: str | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        result = judge.chat(messages=messages, max_tokens=600)
        last_content = result.content or ""
        try:
            score, reason = _parse(last_content)
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "task %s: judge reply unparseable on attempt %d/%d: %s",
                task.task_id, attempt, max_attempts, exc,
            )
            messages.append({"role": "assistant", "content": last_content})
            messages.append({
                "role": "user",
                "content": 'That was not a usable reply. Answer with exactly {"score": <0-1>, '
                           '"reason": "..."} and nothing else.',
            })
            continue
        return LLMVerdict(score=score, passed=score >= threshold, reason=reason, raw=last_content)

    return LLMVerdict(
        score=0.0,
        passed=False,
        reason="",
        raw=last_content,
        error=f"judge_unparseable:{type(last_error).__name__}",
    )
