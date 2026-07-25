"""Agent role client. Backend is intentionally undecided (see U2 in
data/design-decisions.md and the 2026-07-15 note in
data/code-architecture-plan.md: Qwen3-8B local serving vs. a hosted API such
as Gemini). Both are OpenAI-compatible from this client's point of view, so
no code here changes when that decision lands — only `base_url`/`model`.
"""

from __future__ import annotations

from qwen_agentworld.config import get_env
from qwen_agentworld.llm_clients.base import LLMClient

DEFAULT_AGENT_MODEL = "Qwen3-8B"
DEFAULT_AGENT_BASE_URL = "http://localhost:8001/v1"


class AgentClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url or get_env("AGENT_BASE_URL", DEFAULT_AGENT_BASE_URL),
            api_key=get_env("AGENT_API_KEY", "EMPTY"),
            model=model or get_env("AGENT_MODEL", DEFAULT_AGENT_MODEL),
            **kwargs,
        )
