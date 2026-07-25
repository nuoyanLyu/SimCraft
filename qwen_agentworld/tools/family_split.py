"""Train/eval tool-family isolation and overlap auditing (D7 / Gate 0).

The design decision this enforces: at training time the agent must never see
tools from the family used for the unseen-tool held-out evaluation. This
module makes that a checkable invariant instead of a documentation promise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qwen_agentworld.core.schemas import ToolSpec


def _schema_fingerprint(tool: ToolSpec) -> frozenset[str]:
    """Structural fingerprint of a tool's parameter schema, for overlap detection.

    Not just the tool name: two differently-named tools that expose an
    identical parameter surface would still leak information about the
    held-out family, so we fingerprint on the sorted parameter keys too.
    """
    props = tool.function.parameters.get("properties", {})
    return frozenset({tool.name, *sorted(props.keys())})


@dataclass
class OverlapReport:
    train_family: str
    eval_family: str
    name_overlap: set[str] = field(default_factory=set)
    schema_key_overlap: set[str] = field(default_factory=set)

    @property
    def is_clean(self) -> bool:
        return not self.name_overlap and not self.schema_key_overlap


def audit_family_overlap(
    train_tools: list[ToolSpec],
    eval_tools: list[ToolSpec],
    train_family: str,
    eval_family: str,
) -> OverlapReport:
    """Gate 0 kill-switch check: train family A vs. eval family B must share
    zero tool names and zero parameter-schema keys.
    """
    train_names = {t.name for t in train_tools}
    eval_names = {t.name for t in eval_tools}

    train_keys: set[str] = set()
    for t in train_tools:
        train_keys |= set(t.function.parameters.get("properties", {}).keys())
    eval_keys: set[str] = set()
    for t in eval_tools:
        eval_keys |= set(t.function.parameters.get("properties", {}).keys())

    return OverlapReport(
        train_family=train_family,
        eval_family=eval_family,
        name_overlap=train_names & eval_names,
        schema_key_overlap=train_keys & eval_keys,
    )


def assert_family_isolation(
    train_tools: list[ToolSpec],
    eval_tools: list[ToolSpec],
    train_family: str,
    eval_family: str,
) -> OverlapReport:
    """Raise if the isolation invariant is violated; otherwise return the report.

    Intended to run as a CI/pre-flight check before any orchestrator run
    (Stage 0 pass criterion in data/code-architecture-plan.md).
    """
    report = audit_family_overlap(train_tools, eval_tools, train_family, eval_family)
    if not report.is_clean:
        raise ValueError(
            f"tool family isolation violated between '{train_family}' and '{eval_family}': "
            f"name_overlap={report.name_overlap}, schema_key_overlap={report.schema_key_overlap}"
        )
    return report
