"""Core data contracts shared across all modules.

See data/code-architecture-plan.md §2 for the design rationale. This module is
the interface boundary between every other package in this framework — changes
here ripple everywhere, so keep it minimal and stable.
"""

from __future__ import annotations

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
    step_wise_predicate: str | None = Field(
        default=None,
        description=(
            "Predicate over a variable `states` (ordered list: initial state plus the "
            "canonical state after every executed step). Set only when "
            "step_wise_diagnostics is True, for tasks whose final observable state can "
            "equal the initial state (e.g. create-then-delete): an end-state predicate "
            "cannot tell did the work then reverted it from did nothing, so scoring "
            "inspects intermediate states instead."
        ),
    )
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
# 2.6 Playbook: an open, incrementally collected set of entries
# --------------------------------------------------------------------------- #
#
# This used to be a closed five-value `PlaybookCategory` enum with exactly one
# module per category. Two things were wrong with that, and they compounded.
#
# The enum decided in advance what an agent is allowed to have learned. A
# lesson that did not fit one of the five buckets ("re-read the listing the
# tool returned instead of reconstructing the id yourself") had nowhere to be
# written down, and the reflection prompt enumerated the five names, so the
# teacher was forced to file every observation under the nearest one — the
# taxonomy shaped the diagnosis rather than the diagnosis shaping the
# taxonomy. GEPA and comparable work do not predefine the buckets; the
# artifact accumulates the units the run actually produces.
#
# Worse, one-module-per-category made every update a *whole-module rewrite*
# under a word budget, i.e. lossy: folding a new lesson in meant deleting an
# old one, so the playbook could not accumulate at all. Entries fix that by
# making the update unit an atomic bullet: a new lesson is an ADD, and prior
# lessons survive untouched unless something explicitly supersedes them.


def normalize_tag(raw: str) -> str:
    """Fold a free-form tag onto a stable key.

    Tags are invented by the teacher at diagnosis time, not chosen from a
    list, so the same idea arrives as "Schema Grounding", "schema-grounding"
    and "schema_grounding" across calls. Without normalization those are three
    groups in the rendered playbook and three separate histories.
    """
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in raw.strip().lower())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or "general"


# --------------------------------------------------------------------------- #
# Diagnosis (teacher/reflection.py output; consumed by optimizer.propose())
# --------------------------------------------------------------------------- #


class StepDiagnosis(BaseModel):
    step_id: str
    verdict: Literal["correct", "suboptimal", "erroneous"]
    feedback: str
    suggested_tag: str | None = Field(
        default=None,
        description=(
            "Free-form short label the teacher writes for the *kind* of mistake, e.g. "
            "'stale-identifier-reuse'. Not drawn from a fixed vocabulary: the set of tags "
            "in a run is an output of the run, not an input to it."
        ),
    )

    @field_validator("suggested_tag")
    @classmethod
    def _normalize(cls, v: str | None) -> str | None:
        return normalize_tag(v) if v else None


class Diagnosis(BaseModel):
    task_id: str
    overall_verdict: Literal["success", "partial", "failure"]
    step_diagnoses: list[StepDiagnosis] = Field(default_factory=list)
    summary: str
    helpful_entry_ids: list[str] = Field(
        default_factory=list,
        description="Existing playbook entries the trajectory visibly followed to its benefit.",
    )
    harmful_entry_ids: list[str] = Field(
        default_factory=list,
        description="Existing playbook entries the trajectory followed into a mistake.",
    )


class ParetoScores(BaseModel):
    """Whole-playbook Pareto axes.

    These sat on `PlaybookModule` before, but every writer set them from a
    rollout — and a rollout exercises the whole playbook, so all modules
    always carried identical values, which made `dominates()` unable to ever
    be true and quietly flattened the frontier. They are playbook-level
    measurements; store them where they are measured. Per-entry credit lives
    on `EntryStats` instead, where it can actually differ between entries.
    """

    task_coverage: float = 0.0
    audit_acceptance: float = 0.0
    compactness: float = 0.0


class EntryStats(BaseModel):
    """Per-entry credit assignment, accumulated across rollouts."""

    helpful: int = 0
    harmful: int = 0

    @property
    def utility(self) -> int:
        return self.helpful - self.harmful


class PlaybookEntry(BaseModel):
    """One atomic, self-contained piece of guidance.

    An entry should be a single actionable rule, not a section: it is the unit
    that gets added, superseded, merged and credited, so anything that bundles
    several lessons together can only be updated by rewriting all of them.
    """

    entry_id: str = Field(default_factory=lambda: _new_id("e"))
    tag: str = Field(description="Free-form grouping label; see normalize_tag.")
    content: str
    version: int = 1
    provenance: list[str] = Field(
        default_factory=list, description="entry_ids this entry was derived from (update/merge parents)"
    )
    stats: EntryStats = Field(default_factory=EntryStats)
    created_at_playbook_version: int = 1
    updated_at_playbook_version: int = 1

    @field_validator("tag")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_tag(v)


class Playbook(BaseModel):
    """A full playbook snapshot: an ordered, open-ended list of entries.

    This is what `playbook_store` versions and what `optimizer.propose()`
    mutates — never a single entry in isolation, since Pareto selection
    (task_coverage / audit_acceptance / compactness) is evaluated at the
    whole-playbook level.

    There is no cap on `entries`: growth is bounded by deduplication and
    merging (`optimizer/ops.py`), not by a size limit, so nothing is ever
    dropped merely for arriving late.
    """

    playbook_id: str = Field(default_factory=lambda: _new_id("pb"))
    version: int = 1
    entries: list[PlaybookEntry] = Field(default_factory=list)
    pareto_scores: ParetoScores = Field(default_factory=ParetoScores)
    validation_utility: float | None = Field(
        default=None, description="held-out simulated utility, set after eval; drives U7 rollback"
    )

    def by_id(self, entry_id: str) -> PlaybookEntry | None:
        return next((e for e in self.entries if e.entry_id == entry_id), None)

    def tags(self) -> list[str]:
        """Tags in first-appearance order — the grouping used when rendering."""
        seen: list[str] = []
        for entry in self.entries:
            if entry.tag not in seen:
                seen.append(entry.tag)
        return seen

    def entries_by_tag(self, tag: str) -> list[PlaybookEntry]:
        return [e for e in self.entries if e.tag == normalize_tag(tag)]

    @property
    def word_count(self) -> int:
        return sum(len(e.content.split()) for e in self.entries)
