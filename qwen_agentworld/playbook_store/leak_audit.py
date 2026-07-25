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

from qwen_agentworld.core.schemas import Playbook


def audit_leakage(playbook: Playbook, forbidden_terms: set[str]) -> dict[str, list[str]]:
    """Returns {module_id: [violating terms]} for every module that contains
    a forbidden term as a case-insensitive substring. Empty dict = clean.
    """
    violations: dict[str, list[str]] = {}
    for module in playbook.modules.values():
        content_lower = module.content.lower()
        hits = [term for term in forbidden_terms if term and term.lower() in content_lower]
        if hits:
            violations[module.module_id] = hits
    return violations
