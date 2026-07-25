"""GEPA-style engine: reflective mutation of a single playbook module,
targeted at the category a diagnosis implicates most.

"GEPA" here means the core idea this repo currently leans toward for U1 —
use natural-language reflection on failures to rewrite a text artifact (the
playbook module content) — not the full published population-based Pareto
search; that variant is future work once U1 is settled, not a blocker to
having one working, testable engine now.
"""

from __future__ import annotations

from collections import Counter

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookCategory, PlaybookModule
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer

_MUTATION_SYSTEM_PROMPT = (
    "You improve a reusable, domain-agnostic tool-use playbook module based on concrete failure "
    "feedback. The module must stay general: never mention specific tool names, endpoints, or "
    "task answers — only general strategy (e.g. 'always confirm required parameters are present "
    "before calling a tool', not 'call search_docs with a query parameter'). "
    'Reply with a single JSON object: {"content": "<the rewritten module text>"}.'
)

_MAX_MUTATION_ATTEMPTS = 3


def _chat_for_content(teacher: LLMClient, messages: list[dict], max_tokens: int) -> str:
    """Retries on an empty or malformed (e.g. truncated mid-string) reply,
    same failure mode `task_generator.instantiate_nl_and_state` and
    `checker_synth.synthesize_checker` already guard against.
    """
    last_error: Exception | None = None
    for _ in range(_MAX_MUTATION_ATTEMPTS):
        result = teacher.chat(messages=messages, max_tokens=max_tokens)
        try:
            return extract_json_object(result.content or "")["content"]
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


def most_implicated_category(diagnosis: Diagnosis) -> PlaybookCategory | None:
    counts = Counter(sd.suggested_category for sd in diagnosis.step_diagnoses if sd.suggested_category is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def build_mutation_prompt(module: PlaybookModule, diagnosis: Diagnosis) -> str:
    feedback = "\n".join(
        f"- step {sd.step_id} ({sd.verdict}): {sd.feedback}"
        for sd in diagnosis.step_diagnoses
        if sd.suggested_category == module.category
    )
    return (
        f"Current module ({module.category.value}):\n{module.content or '(empty — no guidance yet)'}\n\n"
        f"Failure feedback implicating this category:\n{feedback}\n\n"
        f"Overall trajectory verdict: {diagnosis.overall_verdict}. Summary: {diagnosis.summary}\n\n"
        "Rewrite the module content to address this feedback."
    )


class GEPAEngine(PlaybookOptimizer):
    def __init__(self, teacher: LLMClient) -> None:
        self._teacher = teacher

    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        category = most_implicated_category(diagnosis)
        if category is None:
            return []  # nothing actionable in this diagnosis

        existing = current.modules.get(category, PlaybookModule(category=category, content=""))
        content = _chat_for_content(
            self._teacher,
            messages=[
                {"role": "system", "content": _MUTATION_SYSTEM_PROMPT},
                {"role": "user", "content": build_mutation_prompt(existing, diagnosis)},
            ],
            max_tokens=1200,
        )
        new_module = PlaybookModule(
            category=category,
            content=content,
            version=existing.version + 1,
            provenance=[*existing.provenance, existing.module_id],
            pareto_scores=existing.pareto_scores,
        )
        candidate_modules = dict(current.modules)
        candidate_modules[category] = new_module
        return [Playbook(version=current.version + 1, modules=candidate_modules)]
