import pytest

from qwen_agentworld.core.schemas import ToolFunctionSpec, ToolSpec
from qwen_agentworld.tools.family_split import assert_family_isolation, audit_family_overlap
from qwen_agentworld.tools.registry import DuplicateToolError, ToolRegistry


def tool(name, params=None, family="mcp_A"):
    return ToolSpec(
        function=ToolFunctionSpec(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": params or {}},
        ),
        family=family,
    )


def test_registry_rejects_duplicates():
    reg = ToolRegistry()
    reg.register(tool("search_docs"))
    with pytest.raises(DuplicateToolError):
        reg.register(tool("search_docs"))


def test_registry_scopes_by_family_and_strips_internal_field():
    reg = ToolRegistry()
    reg.register(tool("search_docs", family="mcp_A"))
    reg.register(tool("run_command", family="terminal_A"))
    wire = reg.to_wire(family="mcp_A")
    assert len(wire) == 1
    assert "family" not in wire[0]
    assert reg.families() == {"mcp_A", "terminal_A"}


def test_overlap_audit_detects_name_collision():
    train = [tool("search_docs", family="mcp_A")]
    eval_ = [tool("search_docs", family="mcp_B")]
    report = audit_family_overlap(train, eval_, "mcp_A", "mcp_B")
    assert not report.is_clean
    assert "search_docs" in report.name_overlap


def test_overlap_audit_detects_schema_key_collision_even_with_different_names():
    train = [tool("search_docs", params={"query": {"type": "string"}}, family="mcp_A")]
    eval_ = [tool("lookup_docs", params={"query": {"type": "string"}}, family="mcp_B")]
    report = audit_family_overlap(train, eval_, "mcp_A", "mcp_B")
    assert not report.is_clean
    assert "query" in report.schema_key_overlap
    assert not report.name_overlap


def test_assert_family_isolation_passes_for_disjoint_families():
    train = [tool("search_docs", params={"query": {"type": "string"}}, family="mcp_A")]
    eval_ = [tool("read_file", params={"path": {"type": "string"}}, family="terminal_B")]
    report = assert_family_isolation(train, eval_, "mcp_A", "terminal_B")
    assert report.is_clean


def test_assert_family_isolation_raises_on_violation():
    train = [tool("search_docs", family="mcp_A")]
    eval_ = [tool("search_docs", family="mcp_B")]
    with pytest.raises(ValueError, match="tool family isolation violated"):
        assert_family_isolation(train, eval_, "mcp_A", "mcp_B")
