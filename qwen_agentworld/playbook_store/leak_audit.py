"""Hard rule #1 (code-architecture-plan.md §3): a playbook module must never
contain a specific tool name, endpoint, or test answer. That would let
"meta-skill" content silently become "tool-specific memorization", defeating
the entire unseen-tool-transfer claim (D6/D7).

This is a necessary-but-not-sufficient check: it catches known forbidden
terms (tool names from the registry, benchmark answer strings, etc.) but
can't prove a module is *semantically* domain-agnostic. Treat a clean audit
as "no known leak", not "definitely safe".
"""

from __future__ import annotations

from qwen_agentworld.core.schemas import Playbook, ToolSpec


def audit_leakage(playbook: Playbook, forbidden_terms: set[str]) -> dict[str, list[str]]:
    """Returns {entry_id: [violating terms]} for every entry that contains a
    forbidden term as a case-insensitive substring. Empty dict = clean.

    The entry's `tag` is audited alongside its content: tags are free-form and
    written by the teacher, so a tag is just as capable of naming a tool as a
    rule is, and leaving it unaudited would open a leak channel the content
    check never looks at.
    """
    violations: dict[str, list[str]] = {}
    for entry in playbook.entries:
        haystack = f"{entry.tag} {entry.content}".lower()
        hits = [term for term in forbidden_terms if term and term.lower() in haystack]
        if hits:
            violations[entry.entry_id] = hits
    return violations


def forbidden_terms_from_tools(tools: list[ToolSpec]) -> set[str]:
    """The tool names a playbook trained on `tools` must not name.

    Callers were expected to assemble this by hand and every one of them passed
    nothing, leaving the audit disabled. Deriving it from the tools list the
    caller already holds removes the opportunity to forget.

    Very short names are dropped: a two-character tool name would match inside
    ordinary English words and fail every playbook regardless of content.
    """
    return {
        tool.function.name
        for tool in tools
        if tool.function.name and len(tool.function.name) > 3
    }
