"""TextGrad-style engine: a two-step "textual gradient" — first critique the
current module against the failure feedback (the "gradient"), then apply
that critique as an edit (the "gradient step") — kept as a separate LLM call
from the critique so the edit is grounded in an explicit, inspectable
critique rather than asking the model to critique-and-rewrite in one shot.

Strong baseline per design-decisions.md (U1); same `PlaybookOptimizer`
interface as `gepa_engine.GEPAEngine` so swapping between them is a one-line
config change in the orchestrator.
"""

from __future__ import annotations

from qwen_agentworld.core.json_utils import extract_json_object
from qwen_agentworld.core.schemas import Diagnosis, Playbook, PlaybookModule
from qwen_agentworld.llm_clients.base import LLMClient
from qwen_agentworld.optimizer.base import PlaybookOptimizer
from qwen_agentworld.optimizer.gepa_engine import build_mutation_prompt, most_implicated_category

_CRITIQUE_SYSTEM_PROMPT = (
    "You critique a domain-agnostic tool-use playbook module against concrete failure feedback. "
    "Identify precisely what guidance is missing or wrong — this critique is the only signal an "
    "editor will use to revise the module, so be specific and actionable. "
    'Reply with a single JSON object: {"critique": "<your critique>"}.'
)

_EDIT_SYSTEM_PROMPT = (
    "You revise a domain-agnostic tool-use playbook module by applying a critique. Keep the "
    "module general: never mention specific tool names, endpoints, or task answers. "
    'Reply with a single JSON object: {"content": "<the revised module text>"}.'
)

_MAX_STEP_ATTEMPTS = 3


def _chat_for_field(teacher: LLMClient, messages: list[dict], max_tokens: int, field: str) -> str:
    """Retries on an empty or malformed (e.g. truncated mid-string) reply,
    same failure mode `task_generator.instantiate_nl_and_state` and
    `gepa_engine._chat_for_content` already guard against.
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


class TextGradEngine(PlaybookOptimizer):
    def __init__(self, teacher: LLMClient) -> None:
        self._teacher = teacher

    def _critique(self, module: PlaybookModule, diagnosis: Diagnosis) -> str:
        return _chat_for_field(
            self._teacher,
            messages=[
                {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": build_mutation_prompt(module, diagnosis)},
            ],
            max_tokens=900,
            field="critique",
        )

    def _apply_critique(self, module: PlaybookModule, critique: str) -> str:
        return _chat_for_field(
            self._teacher,
            messages=[
                {"role": "system", "content": _EDIT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current module content:\n{module.content or '(empty — no guidance yet)'}\n\n"
                        f"Critique to apply:\n{critique}"
                    ),
                },
            ],
            max_tokens=1300,
            field="content",
        )

    def propose(self, current: Playbook, diagnosis: Diagnosis) -> list[Playbook]:
        category = most_implicated_category(diagnosis)
        if category is None:
            return []

        existing = current.modules.get(category, PlaybookModule(category=category, content=""))
        critique = self._critique(existing, diagnosis)
        new_content = self._apply_critique(existing, critique)

        new_module = PlaybookModule(
            category=category,
            content=new_content,
            version=existing.version + 1,
            provenance=[*existing.provenance, existing.module_id],
            pareto_scores=existing.pareto_scores,
        )
        candidate_modules = dict(current.modules)
        candidate_modules[category] = new_module
        return [Playbook(version=current.version + 1, modules=candidate_modules)]
