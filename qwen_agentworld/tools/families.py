"""Real multi-domain tool families (C1 cross-domain transfer substrate).

Until now the pipeline only ever exercised a throwaway toy "notes" family
(defined ad-hoc inside scripts/). This module promotes tool families to a
first-class, reusable catalog spanning four *text-tool* agentic domains that
Qwen-AgentWorld covers and whose next-state prediction is reliable enough to
train on (MCP-style API orchestration, Terminal/filesystem, Web research,
and SWE/code-repo). GUI domains (Android/Web-DOM/OS) are intentionally left
out for now (see data/paper-story1.md: fidelity/curriculum tradeoffs).

Design invariant that makes both C1 (cross-domain transfer) and D7
(unseen-tool held-out) work at once: the four domains use **mutually disjoint
parameter vocabularies** (no shared tool names, no shared parameter keys), so
`tools/family_split.assert_family_isolation` is clean for *any* pair of
domains chosen as train vs. eval. `initial_state` is not defined here — it is
synthesized per task by `teacher/task_generator.instantiate_nl_and_state`
from the tool graph, against the family's declared schema in
`tools/state_schema.py` — that module owns the canonical-state shape and is
the single description shown to the teacher, shown to the simulator, and used
to audit both. (It replaced a prose `FAMILY_STATE_HINTS` dict here that no
model was ever shown; see its docstring for what that cost.)
"""

from __future__ import annotations

from qwen_agentworld.core.schemas import ToolFunctionSpec, ToolSpec
from qwen_agentworld.tools.registry import ToolRegistry


def _t(name: str, description: str, properties: dict, required: list[str], family: str) -> ToolSpec:
    return ToolSpec(
        function=ToolFunctionSpec(
            name=name,
            description=description,
            parameters={"type": "object", "properties": properties, "required": required},
        ),
        family=family,
    )


# --------------------------------------------------------------------------- #
# Domain 1 - MCP-style record/API orchestration (keys: collection/filter_expr/fields/record_id)
# --------------------------------------------------------------------------- #
_MCP = "mcp_api"
MCP_API_TOOLS: list[ToolSpec] = [
    _t("list_collections", "List the names of all record collections available in the workspace.",
       {}, [], _MCP),
    _t("query_records", "Return records in a collection matching a filter expression; returns [] if none match.",
       {"collection": {"type": "string"}, "filter_expr": {"type": "string"}}, ["collection", "filter_expr"], _MCP),
    _t("create_record", "Create a new record in a collection from a fields object; returns the new record id.",
       {"collection": {"type": "string"}, "fields": {"type": "object"}}, ["collection", "fields"], _MCP),
    _t("update_record", "Overwrite the given fields of an existing record identified by record_id.",
       {"record_id": {"type": "string"}, "fields": {"type": "object"}}, ["record_id", "fields"], _MCP),
    _t("delete_record", "Permanently delete the record identified by record_id.",
       {"record_id": {"type": "string"}}, ["record_id"], _MCP),
]

# --------------------------------------------------------------------------- #
# Domain 2 - Terminal / filesystem (keys: directory/path/text/source/target)
# --------------------------------------------------------------------------- #
_TERM = "terminal_ops"
TERMINAL_OPS_TOOLS: list[ToolSpec] = [
    _t("list_directory", "List the entries (files and subdirectories) directly under a directory.",
       {"directory": {"type": "string"}}, ["directory"], _TERM),
    _t("read_file", "Return the full text contents of the file at path; errors if it does not exist.",
       {"path": {"type": "string"}}, ["path"], _TERM),
    _t("write_file", "Create or overwrite the file at path with the given text.",
       {"path": {"type": "string"}, "text": {"type": "string"}}, ["path", "text"], _TERM),
    _t("move_path", "Move (rename) a file or directory from source to target.",
       {"source": {"type": "string"}, "target": {"type": "string"}}, ["source", "target"], _TERM),
    _t("remove_path", "Permanently remove the file or directory at path.",
       {"path": {"type": "string"}}, ["path"], _TERM),
    _t("make_directory", "Create a new directory (including missing parents).",
       {"directory": {"type": "string"}}, ["directory"], _TERM),
]

# --------------------------------------------------------------------------- #
# Domain 3 - Web research (keys: query/rank/headline/body)
# --------------------------------------------------------------------------- #
_SEARCH = "web_research"
WEB_RESEARCH_TOOLS: list[ToolSpec] = [
    _t("web_search", "Run a web search for query; returns a ranked list of result stubs (rank, title, url).",
       {"query": {"type": "string"}}, ["query"], _SEARCH),
    _t("open_result", "Open the search result at the given 1-based rank and return its page metadata.",
       {"rank": {"type": "integer"}}, ["rank"], _SEARCH),
    _t("extract_snippet", "Extract the main readable text snippet from the result at the given rank.",
       {"rank": {"type": "integer"}}, ["rank"], _SEARCH),
    _t("save_note", "Persist a research note with a headline and body into the research notebook.",
       {"headline": {"type": "string"}, "body": {"type": "string"}}, ["headline", "body"], _SEARCH),
    _t("list_saved_notes", "List the headlines of all notes currently saved in the research notebook.",
       {}, [], _SEARCH),
]

# --------------------------------------------------------------------------- #
# Domain 4 - SWE / code repo (keys: module/diff/suite/branch)
# --------------------------------------------------------------------------- #
_SWE = "code_repo"
CODE_REPO_TOOLS: list[ToolSpec] = [
    _t("show_file", "Return the current source of a module (file) in the repository.",
       {"module": {"type": "string"}}, ["module"], _SWE),
    _t("apply_patch", "Apply a unified-diff patch to a module; fails if the diff does not apply cleanly.",
       {"module": {"type": "string"}, "diff": {"type": "string"}}, ["module", "diff"], _SWE),
    _t("run_test_suite", "Run a named test suite and return pass/fail counts and failing test names.",
       {"suite": {"type": "string"}}, ["suite"], _SWE),
    _t("vcs_status", "Return the version-control status: modified, staged, and untracked modules.",
       {}, [], _SWE),
    _t("open_branch", "Create and switch to a new version-control branch.",
       {"branch": {"type": "string"}}, ["branch"], _SWE),
]


ALL_FAMILIES: dict[str, list[ToolSpec]] = {
    _MCP: MCP_API_TOOLS,
    _TERM: TERMINAL_OPS_TOOLS,
    _SEARCH: WEB_RESEARCH_TOOLS,
    _SWE: CODE_REPO_TOOLS,
}

# Convenience defaults aligned with data/paper-story1.md + evaluation-benchmarks.md.
DEFAULT_TRAINING_FAMILIES: tuple[str, ...] = (_MCP, _TERM, _SWE)
DEFAULT_HELDOUT_FAMILY: str = _SEARCH


# Tools that irreversibly destroy pre-existing data. They stay in every agent's
# tool list -- refusing to reach for them unprompted is a real capability -- but
# tasks are never *built* around them: a task bank full of "delete X then create
# Y" both rewards destructive behaviour and puts the generated playbook at war
# with itself (the evolved precondition_check module learned to refuse deletions
# outright, which then depressed the pass rate on exactly those tasks).
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset(
    {"delete_record", "remove_path", "delete_note"}  # delete_note: the mcp_notes toy family in scripts/
)


def non_destructive(tools: list[ToolSpec]) -> list[ToolSpec]:
    """`tools` minus anything in `DESTRUCTIVE_TOOLS`."""
    return [t for t in tools if t.name not in DESTRUCTIVE_TOOLS]


def family_names() -> list[str]:
    """All registered family names, in catalog order."""
    return list(ALL_FAMILIES.keys())


def get_family(name: str) -> list[ToolSpec]:
    """The ToolSpec list for one family; raises KeyError if unknown."""
    return ALL_FAMILIES[name]


def build_registry(families: list[str] | None = None) -> ToolRegistry:
    """Build a ToolRegistry populated with the given families (default: all).

    Tool names are unique across the whole catalog, so registering multiple
    families never collides.
    """
    registry = ToolRegistry()
    for fam in families if families is not None else family_names():
        registry.register_many(ALL_FAMILIES[fam])
    return registry


def training_split(heldout_family: str = DEFAULT_HELDOUT_FAMILY) -> tuple[list[ToolSpec], list[ToolSpec]]:
    """Return (train_tools, heldout_tools) for an unseen-domain split.

    `train_tools` = every family except `heldout_family`, flattened;
    `heldout_tools` = the held-out family. Because domains have disjoint
    parameter vocabularies, `family_split.assert_family_isolation` is clean.
    """
    if heldout_family not in ALL_FAMILIES:
        raise KeyError(f"unknown family: {heldout_family!r}; known: {family_names()}")
    train: list[ToolSpec] = []
    for fam, tools in ALL_FAMILIES.items():
        if fam != heldout_family:
            train.extend(tools)
    return train, ALL_FAMILIES[heldout_family]
