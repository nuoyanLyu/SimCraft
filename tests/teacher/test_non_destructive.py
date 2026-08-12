"""Tasks are never built around destroying data the starting state contains.

Rationale in `families.DESTRUCTIVE_TOOLS`: the evolved playbook's
`precondition_check` module learned to refuse deletions, which is sensible
agent behaviour and was in direct conflict with a task bank whose tasks
demanded them.
"""

import random

from qwen_agentworld.teacher.task_generator import sample_task_graph
from qwen_agentworld.tools.families import (
    DESTRUCTIVE_TOOLS,
    MCP_API_TOOLS,
    TERMINAL_OPS_TOOLS,
    non_destructive,
)


def test_destructive_tools_never_appear_in_a_sampled_graph():
    pool = MCP_API_TOOLS + TERMINAL_OPS_TOOLS
    for seed in range(50):
        graph = sample_task_graph(pool, min_nodes=2, max_nodes=4, rng=random.Random(seed))
        assert not (DESTRUCTIVE_TOOLS & {n.tool_name for n in graph.nodes})


def test_they_remain_in_the_family_so_the_agent_still_has_them():
    """Declining an unrequested deletion is a capability worth keeping
    measurable; only task *generation* excludes them."""
    assert "delete_record" in {t.name for t in MCP_API_TOOLS}
    assert "remove_path" in {t.name for t in TERMINAL_OPS_TOOLS}


def test_non_destructive_filters_exactly_the_named_tools():
    kept = {t.name for t in non_destructive(MCP_API_TOOLS)}
    assert kept == {t.name for t in MCP_API_TOOLS} - {"delete_record"}


def test_a_graph_can_still_reach_the_requested_size_after_filtering():
    graph = sample_task_graph(MCP_API_TOOLS, min_nodes=4, max_nodes=4, rng=random.Random(0))
    assert len(graph.nodes) == 4  # 5 MCP tools, 1 destructive -> exactly 4 left
