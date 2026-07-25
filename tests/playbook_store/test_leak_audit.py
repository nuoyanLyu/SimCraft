from qwen_agentworld.core.schemas import ParetoScores, Playbook, PlaybookCategory, PlaybookModule
from qwen_agentworld.playbook_store.leak_audit import audit_leakage


def make_playbook(content: str) -> Playbook:
    module = PlaybookModule(category=PlaybookCategory.ERROR_RECOVERY, content=content)
    return Playbook(modules={PlaybookCategory.ERROR_RECOVERY: module})


def test_clean_content_has_no_violations():
    pb = make_playbook("Always re-check postconditions after a write operation.")
    assert audit_leakage(pb, forbidden_terms={"search_docs", "api.internal.example.com"}) == {}


def test_leaked_tool_name_is_detected():
    pb = make_playbook("If unsure, call search_docs again with a broader query.")
    violations = audit_leakage(pb, forbidden_terms={"search_docs"})
    assert len(violations) == 1
    assert "search_docs" in next(iter(violations.values()))


def test_case_insensitive_match():
    pb = make_playbook("Retry via SEARCH_DOCS if the first call times out.")
    violations = audit_leakage(pb, forbidden_terms={"search_docs"})
    assert violations
