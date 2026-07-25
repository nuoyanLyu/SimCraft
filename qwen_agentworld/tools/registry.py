"""ToolSpec registry with family tagging.

Stage 0 building block (see data/code-architecture-plan.md §3, Stage 0). Pure
in-memory registry — no I/O, no external services, so it needs no mocking to
test.
"""

from __future__ import annotations

from qwen_agentworld.core.schemas import ToolSpec


class DuplicateToolError(ValueError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolSpec]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def by_family(self, family: str) -> list[ToolSpec]:
        return [t for t in self._tools.values() if t.family == family]

    def families(self) -> set[str]:
        return {t.family for t in self._tools.values()}

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def to_wire(self, family: str | None = None) -> list[dict]:
        """Qwen3-compatible ``tools=[...]`` payload, optionally scoped to one family."""
        tools = self.by_family(family) if family is not None else self.all()
        return [t.to_wire() for t in tools]
