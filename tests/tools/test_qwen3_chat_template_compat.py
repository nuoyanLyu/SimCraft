"""Verify ToolSpec.to_wire() is actually consumable by the real Qwen3-family
chat template, not just structurally plausible.

This renders against the chat_template.jinja shipped with
Qwen-AgentWorld-35B-A3B, which only needs the small template file, not the
weights. The checkout location is machine-specific, so resolve it from
$SIMULATOR_MODEL_PATH -- the same variable scripts/serve_simulator.sh reads --
falling back to that script's default. Skips gracefully if the file isn't
present, e.g. on a machine without the model checkout.
"""

from __future__ import annotations

import os

import jinja2
import pytest

from qwen_agentworld.core.schemas import ToolFunctionSpec, ToolSpec
from qwen_agentworld.tools.registry import ToolRegistry

MODEL_DIR = os.environ.get(
    "SIMULATOR_MODEL_PATH", "/root/autodl-tmp/models/Qwen-AgentWorld-35B-A3B"
)
CHAT_TEMPLATE_PATH = os.path.join(MODEL_DIR, "chat_template.jinja")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CHAT_TEMPLATE_PATH),
    reason="simulator chat_template.jinja not present on this machine",
)


def _render(messages, tools):
    with open(CHAT_TEMPLATE_PATH) as f:
        template_src = f.read()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_src)
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        add_vision_id=False,
    )


def test_tool_spec_wire_format_renders_in_real_qwen_template():
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            function=ToolFunctionSpec(
                name="search_docs",
                description="Search internal docs by query.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            family="mcp_A",
        )
    )
    tools_wire = reg.to_wire(family="mcp_A")
    rendered = _render(
        messages=[{"role": "user", "content": "Find docs about retries."}],
        tools=tools_wire,
    )
    assert "search_docs" in rendered
    assert "<tools>" in rendered and "</tools>" in rendered
    assert "<tool_call>" in rendered
    # the internal-only field must never leak into the rendered prompt
    assert "family" not in rendered
