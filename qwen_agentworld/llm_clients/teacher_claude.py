"""Teacher role client: reached via an OpenAI-compatible relay (default: the
AUTODL relay). The *role* (Teacher) is what business code depends on, not the
provider — so which model/endpoint answers is a config choice (env vars), not
a code change in teacher/task_generator.py etc.

Configuration (all via .env / environment, see qwen_agentworld/config.py):
  TEACHER_MODEL            model id on the relay (default "claude-sonnet-5";
                           e.g. "deepseek-v4-pro")
  TEACHER_BASE_URL         relay base url (default: AUTODL_URL)
  TEACHER_API_KEY          relay api key  (default: AUTODL_API_KEY)
  TEACHER_MIN_OUTPUT_TOKENS  floor applied to every call's max_tokens (default
                           8000)

Two provider quirks are handled here so every task-generation call site stays
model-agnostic:

1. System role is folded into the first user turn. The AUTODL relay silently
   ignores/overrides the `system` role (verified live: it substitutes its own
   Claude-Code-style system prompt). Instructions in the `user` role are
   always honored, so folding is a safe superset that works whether or not a
   given model/relay respects the system role.

2. A max_tokens floor. Reasoning models (e.g. deepseek-v4-pro) spend the token
   budget on a hidden reasoning trace *before* emitting the answer, and on the
   AUTODL relay that trace counts against max_tokens. Measured: deepseek-v4-pro
   reasons ~600-900 tokens, so a call site asking for max_tokens=500 (the old
   checker_synth budget) gets finish_reason="length" with EMPTY content — the
   exact empty-reply failure the retry loops choke on (every retry re-truncates
   at 500). Non-reasoning models (Claude) stop at their natural `stop`, so a
   high cap is harmless for them. "How many tokens this model needs to think"
   is a property of the backend, not of business logic, so the floor lives here
   rather than being threaded through every call site.
"""

from __future__ import annotations

from typing import Any

from qwen_agentworld.config import get_env
from qwen_agentworld.llm_clients.base import ChatResult, LLMClient

DEFAULT_TEACHER_MODEL = get_env("TEACHER_MODEL", "claude-sonnet-5")
DEFAULT_TEACHER_MIN_OUTPUT_TOKENS = int(get_env("TEACHER_MIN_OUTPUT_TOKENS", "8000"))


def _fold_system_into_first_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold any `system` content into the first user turn (see module docstring,
    quirk 1)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not system_parts:
        return rest

    system_text = "\n\n".join(system_parts)
    for i, m in enumerate(rest):
        if m["role"] == "user":
            rest[i] = {**m, "content": f"{system_text}\n\n{m['content']}"}
            return rest
    return [{"role": "user", "content": system_text}] + rest


class TeacherClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        min_output_tokens: int = DEFAULT_TEACHER_MIN_OUTPUT_TOKENS,
        **kwargs,
    ) -> None:
        self.min_output_tokens = min_output_tokens
        super().__init__(
            base_url=get_env("TEACHER_BASE_URL") or get_env("AUTODL_URL", required=True),
            api_key=get_env("TEACHER_API_KEY") or get_env("AUTODL_API_KEY", required=True),
            model=model or DEFAULT_TEACHER_MODEL,
            **kwargs,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Floor the answer budget so a reasoning model's hidden trace can't eat
        # the whole allowance and leave content empty (see module docstring,
        # quirk 2). A caller asking for more than the floor is left untouched.
        requested = kwargs.get("max_tokens")
        kwargs["max_tokens"] = max(requested or 0, self.min_output_tokens)
        return super().chat(_fold_system_into_first_user(messages), tools=tools, **kwargs)
