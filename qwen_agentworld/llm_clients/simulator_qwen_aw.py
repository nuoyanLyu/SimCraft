"""Simulator role client: Qwen-AgentWorld-35B-A3B, served locally as an
OpenAI-compatible endpoint (e.g. vLLM `--served-model-name`).

STATUS (2026-07-22): weights are served via vLLM 0.25.1 (simcraft env) on a
single A800, launched with `--language-model-only --reasoning-parser qwen3`
and `VLLM_USE_FLASHINFER_SAMPLER=0` (the flashinfer JIT sampler fails to
compile in this pip-mixed CUDA env). Point `SIMULATOR_BASE_URL` at it.

Thinking is force-disabled for the Simulator role: Qwen-AgentWorld is a
"thinking" model that, left unchecked, spends the whole token budget on a
visible reasoning trace and leaves `content` empty, which breaks the
`extract_json_object(result.content)` in `simulator_gym.simulate_next_state`.
For a pure next-state predictor the reasoning trace is wasted latency, so we
inject `chat_template_kwargs={"enable_thinking": False}` on every call, which
makes the model emit the `{"next_state": {...}}` JSON directly in `content`.
"""

from __future__ import annotations

from typing import Any

from qwen_agentworld.config import get_env
from qwen_agentworld.llm_clients.base import ChatResult, LLMClient

DEFAULT_SIMULATOR_MODEL = "Qwen-AgentWorld-35B-A3B"
DEFAULT_SIMULATOR_BASE_URL = "http://localhost:8000/v1"


class SimulatorClient(LLMClient):
    def __init__(
        self,
        model: str = DEFAULT_SIMULATOR_MODEL,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_url=base_url or get_env("SIMULATOR_BASE_URL", DEFAULT_SIMULATOR_BASE_URL),
            api_key=get_env("SIMULATOR_API_KEY", "EMPTY"),  # local vLLM servers accept any non-empty key
            model=model,
            **kwargs,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Force thinking off (see module docstring). Merge, do not clobber, any
        # extra_body / chat_template_kwargs a caller may already have set.
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        return super().chat(messages, tools=tools, extra_body=extra_body, **kwargs)
