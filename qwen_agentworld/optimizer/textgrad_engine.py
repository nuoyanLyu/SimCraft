"""TextGrad-style engine: a two-step "textual gradient" — first critique the
current playbook against the failure feedback (the "gradient"), then apply
that critique as edits (the "gradient step") — kept as a separate LLM call
from the critique so the edit is grounded in an explicit, inspectable
critique rather than asking the model to critique-and-rewrite in one shot.

Strong baseline per design-decisions.md (U1); same `PlaybookOptimizer`
interface as `gepa_engine.GEPAEngine` so swapping between them is a one-line
config change in the orchestrator. Both engines now emit the same incremental
ops (optimizer/ops.py) — what distinguishes them is how the edit is derived,
which is the thing U1 is meant to compare, not the shape of the artifact.
"""

from __future__ import annotations

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Diagnosis, Playbook
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.optimizer.gepa_engine import (
    _EDIT_SYSTEM_PROMPT,
    build_edit_prompt,
    build_entry_length_rule,
    observed_tags,
    parse_ops,
)
from qwen_agentworld.optimizer.ops import PlaybookOp, apply_credit, apply_ops, render_entries
from qwen_agentworld.optimizer.scoring import DEFAULT_ENTRY_WORD_BUDGET, score_playbook

_CRITIQUE_SYSTEM_PROMPT = (
    "You critique a domain-agnostic tool-use playbook — a list of short, self-contained rules — "
    "against concrete failure feedback. Identify precisely what guidance is missing, what "
    "existing entry was too vague to prevent the mistake, and what entries duplicate or "
    "contradict each other. This critique is the only signal an editor will use to revise the "
    "playbook, so be specific and actionable, and refer to entries by their id. "
    'Reply with a single JSON object: {"critique": "<your critique>"}.'
)

_MAX_STEP_ATTEMPTS = 3


def _chat_for_field(teacher: LLMClient, messages: list[dict], max_tokens: int, field: str) -> str:
    """Retries on an empty or malformed (e.g. truncated mid-string) reply,
    same failure mode `task_generator.instantiate_nl_and_state` and
    `gepa_engine._chat_for_ops` already guard against.
    """
    last_error: Exception | None = None
    for _ in range(_MAX_STEP_ATTEMPTS):
        result = teacher.chat(messages=messages, max_tokens=max_tokens)
        try:
            return extract_json_object(result.content or "")[field]
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


def _chat_for_ops(teacher: LLMClient, messages: list[dict], max_tokens: int) -> list[PlaybookOp]:
    last_error: Exception | None = None
    for _ in range(_MAX_STEP_ATTEMPTS):
        result = teacher.chat(messages=messages, max_tokens=max_tokens)
        try:
            return parse_ops(result.content or "")
        except (ValueError, KeyError, TypeError) as exc:
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


class TextGradEngine(PlaybookOptimizer):
    def __init__(self, teacher: LLMClient, entry_word_budget: int = DEFAULT_ENTRY_WORD_BUDGET) -> None:
        self._teacher = teacher
        self._entry_word_budget = entry_word_budget

    def _critique(self, current: Playbook, diagnosis: Diagnosis) -> str:
        return _chat_for_field(
            self._teacher,
            messages=[
                {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": build_edit_prompt(current, diagnosis)},
            ],
            max_tokens=900,
            field="critique",
        )

    def _apply_critique(self, current: Playbook, critique: str) -> list[PlaybookOp]:
        entries = render_entries(current, with_ids=True, with_stats=True) or "(empty — no entries yet)"
        return _chat_for_ops(
            self._teacher,
            messages=[
                {
                    "role": "system",
                    "content": _EDIT_SYSTEM_PROMPT + build_entry_length_rule(self._entry_word_budget),
                },
                {
                    "role": "user",
                    "content": f"Current playbook entries:\n{entries}\n\nCritique to apply:\n{critique}",
                },
            ],
            max_tokens=1500,
        )

    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        credited = apply_credit(current, diagnosis)
        if diagnosis.overall_verdict == "success" and not observed_tags(diagnosis):
            return [credited] if credited != current else []

        critique = self._critique(credited, diagnosis)
        ops = self._apply_critique(credited, critique)

        candidate, report = apply_ops(credited, ops)
        if report.n_changes == 0:
            return [credited] if credited != current else []
        return [score_playbook(candidate, entry_word_budget=self._entry_word_budget)]
