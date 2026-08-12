from qwen_agentworld.core.schemas import Playbook, PlaybookEntry
from qwen_agentworld.playbook_store.leak_audit import audit_leakage


def make_playbook(content: str, tag: str = "error-recovery") -> Playbook:
    return Playbook(entries=[PlaybookEntry(tag=tag, content=content)])


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


def test_a_tool_name_hidden_in_a_tag_is_caught_too():
    """Tags are free text written by the teacher, so they can name a tool just
    as easily as the rule body can. Auditing only content would leave the tag
    as an unwatched leak channel."""
    pb = make_playbook("Retry once before giving up.", tag="search-docs-retry")
    assert audit_leakage(pb, forbidden_terms={"search_docs", "search-docs"})
