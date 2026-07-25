"""Agent backend: Qwen3-14B served locally via vLLM's OpenAI-compatible API.

U2 is settled: no bespoke `chat()`-override subclass is needed for
correctness. Unlike the AUTODL relay (`TeacherClient`), a vanilla vLLM
server has no known system-role quirk — it's a standard OpenAI-compatible
server, so the base `LLMClient.chat()` already does the right thing. This
class exists only for the same reason `TeacherClient` does: to centralize
env-var wiring (`AGENT_URL`/`AGENT_API_KEY`/`AGENT_MODEL`) so callers don't
hardcode a vLLM base_url, not to work around a backend bug.

Two things are assumed but NOT yet live-verified — the server's GPUs are
currently saturated by an active GRPO training run, so no vLLM instance for
this model has been started yet:
  1. vLLM honors the `system` role normally (no folding needed, unlike
     TeacherClient).
  2. Tool-calling actually populates `choice.tool_calls` — for Qwen3 this
     requires the server to be launched with
     `--enable-auto-tool-choice --tool-call-parser hermes` (or an
     equivalent Qwen-compatible parser); without those flags, tool calls
     silently come back empty instead of populating `tool_calls`.
Re-verify both the first time a live vLLM instance for this model exists.
"""

from __future__ import annotations

from qwen_agentworld.config import get_env
from qwen_agentworld.llm_clients.base import LLMClient

DEFAULT_AGENT_MODEL = "Qwen3-14B"


class AgentClient(LLMClient):
    def __init__(self, model: str = DEFAULT_AGENT_MODEL, **kwargs) -> None:
        super().__init__(
            base_url=get_env("AGENT_URL", required=True),
            api_key=get_env("AGENT_API_KEY", default="EMPTY"),
            model=model,
            **kwargs,
        )
