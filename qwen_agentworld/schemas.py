"""Core data contracts shared across all modules.

See data/code-architecture-plan.md §2 for the design rationale. This module is
the interface boundary between every other package in this framework — changes
here ripple everywhere, so keep it minimal and stable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# 2.1 ToolSpec (Qwen3 tool-calling format)
# --------------------------------------------------------------------------- #


class ToolFunctionSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolSpec(BaseModel):
    """A single tool definition, wire-compatible with Qwen3's chat template.

    ``family`` is a framework-internal field only — it must never be included
    when serializing for ``apply_chat_template(tools=...)`` or an
    OpenAI-compatible ``tools=`` payload. Use ``to_wire()`` for that.
    """

    type: Literal["function"] = "function"
    function: ToolFunctionSpec
    family: str = Field(description="Tool family label, e.g. 'mcp_A' / 'terminal_B'. Internal only.")

    def to_wire(self) -> dict[str, Any]:
        """Serialize without the internal ``family`` field."""
        return {"type": self.type, "function": self.function.model_dump()}

    @property
    def name(self) -> str:
        return self.function.name


# --------------------------------------------------------------------------- #
# 2.2 / 2.3 Task and CheckerSpec
# --------------------------------------------------------------------------- #


class TaskGraphNode(BaseModel):
    node_id: str
    tool_name: str
    depends_on: list[str] = Field(default_factory=list)
    is_failure_recovery: bool = False


class TaskGraph(BaseModel):
    """Sampled *before* natural-language instantiation (research plan: task-graph-first)."""

    nodes: list[TaskGraphNode]
    branch_points: list[str] = Field(default_factory=list, description="node_ids with >1 successor")

    @field_validator("nodes")
    @classmethod
    def _non_empty(cls, v: list[TaskGraphNode]) -> list[TaskGraphNode]:
        if not v:
            raise ValueError("task graph must contain at least one node")
        return v


class DifficultyMeta(BaseModel):
    target_pass_rate_band: tuple[float, float] = (0.2, 0.6)
    graph_complexity: int = Field(ge=1, description="e.g. node count, used for curriculum control")


class CheckerSpec(BaseModel):
    """Post-condition checker. MUST only reference canonical_state fields.

    ``executable_predicate`` is not natural language — it is a structured
    assertion (e.g. a small DSL / jsonpath-based expression) evaluated against
    canonical_state. Enforcing "no NL reference answer" is a Stage 3 audit
    responsibility (see teacher/checker_synth.py), not something this schema
    can fully guarantee on its own.
    """

    checker_id: str = Field(default_factory=lambda: _new_id("chk"))
    executable_predicate: str
    step_wise_diagnostics: bool = False
    forbidden_patterns: list[str] = Field(default_factory=list)


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: _new_id("task"))
    tool_family: str
    task_graph: TaskGraph
    natural_language_prompt: str
    initial_state: dict[str, Any]
    checker: CheckerSpec
    difficulty_meta: DifficultyMeta


# --------------------------------------------------------------------------- #
# 2.5 EvidenceScore
# --------------------------------------------------------------------------- #


class EvidenceScore(BaseModel):
    schema_valid: bool
    agreement_score: float = Field(ge=0.0, le=1.0)
    counterfactual_pass: bool
    adjudicated: bool | None = None
    confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# 2.4 Trajectory / Step
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class Step(BaseModel):
    step_id: str = Field(default_factory=lambda: _new_id("step"))
    tool_call: ToolCall
    simulator_raw_output: Any = None
    evidence: EvidenceScore | None = None
    accepted: bool = False


class Trajectory(BaseModel):
    task_id: str
    steps: list[Step] = Field(default_factory=list)
    playbook_version: str
    outcome_judged_by_checker: bool | None = None


# --------------------------------------------------------------------------- #
# 2.6 PlaybookModule
# --------------------------------------------------------------------------- #


class PlaybookCategory(str, Enum):
    SCHEMA_GROUNDING = "schema_grounding"
    PRECONDITION_CHECK = "precondition_check"
    INCREMENTAL_EXECUTION = "incremental_execution"
    ERROR_RECOVERY = "error_recovery"
    POSTCONDITION_VERIFICATION = "postcondition_verification"


# --------------------------------------------------------------------------- #
# Diagnosis (teacher/reflection.py output; consumed by optimizer.propose())
# --------------------------------------------------------------------------- #


class StepDiagnosis(BaseModel):
    step_id: str
    verdict: Literal["correct", "suboptimal", "erroneous"]
    feedback: str
    suggested_category: PlaybookCategory | None = None


class Diagnosis(BaseModel):
    task_id: str
    overall_verdict: Literal["success", "partial", "failure"]
    step_diagnoses: list[StepDiagnosis] = Field(default_factory=list)
    summary: str


class ParetoScores(BaseModel):
    task_coverage: float = 0.0
    audit_acceptance: float = 0.0
    compactness: float = 0.0


class PlaybookModule(BaseModel):
    module_id: str = Field(default_factory=lambda: _new_id("mod"))
    category: PlaybookCategory
    content: str
    version: int = 1
    provenance: list[str] = Field(default_factory=list, description="mutation history, e.g. parent module_ids")
    pareto_scores: ParetoScores = Field(default_factory=ParetoScores)


class Playbook(BaseModel):
    """A full playbook snapshot: one module per domain-agnostic category.

    This is what `playbook_store` versions and what `optimizer.propose()`
    mutates — never a single module in isolation, since Pareto selection
    (task_coverage / audit_acceptance / compactness) is evaluated at the
    whole-playbook level.
    """

    playbook_id: str = Field(default_factory=lambda: _new_id("pb"))
    version: int = 1
    modules: dict[PlaybookCategory, PlaybookModule] = Field(default_factory=dict)
    validation_utility: float | None = Field(
        default=None, description="held-out simulated utility, set after eval; drives U7 rollback"
    )
