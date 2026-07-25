"""Shared LLMClient base: every role (Teacher / Simulator / Agent) talks to
its backend through this one interface (see code-architecture-plan.md §1).

All three current backends are OpenAI-compatible HTTP endpoints (confirmed for
the Teacher relay; Simulator/Agent are planned to be vLLM/SGLang OpenAI-compat
servers), so this wraps the `openai` client rather than inventing a bespoke
transport. If a future backend isn't OpenAI-compatible, subclass and override
`chat()` — callers only depend on this base class's interface, not on the
`openai` SDK.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError)


@dataclass
class ToolCallResult:
    id: str
    name: str
    arguments: str  # raw JSON string, as returned by the backend


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    raw: Any = None  # escape hatch for debugging; do not rely on this in business logic


class LLMClient:
    """Thin retrying wrapper. Subclasses fix role-specific base_url/api_key/model."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        # `client` injection exists purely for tests: it lets us swap in a
        # fake transport without touching the network, while production
        # callers just get a real OpenAI client wired to base_url/api_key.
        self._client = client or OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    **kwargs,
                )
                choice = response.choices[0].message
                tool_calls = [
                    ToolCallResult(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                    for tc in (choice.tool_calls or [])
                ]
                return ChatResult(content=choice.content, tool_calls=tool_calls, raw=response)
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning("transient error on attempt %d/%d: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
            except APIStatusError:
                raise
        assert last_exc is not None
        raise last_exc
