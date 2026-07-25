"""Fuses schema/agreement/counterfactual[/adjudication] signals into a single
EvidenceScore and accept/reject decision.

`optimizer/*` is only ever allowed to read this module's output, never raw
simulator output directly (hard rule #2 in code-architecture-plan.md §3) —
that boundary is enforced by module structure, not by this file, but keep it
in mind if you're tempted to pass raw output further downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from qwen_agentworld.core.schemas import EvidenceScore
from qwen_agentworld.evidence_gate.counterfactual_replay import counterfactual_pass
from qwen_agentworld.evidence_gate.schema_check import validate_schema
from qwen_agentworld.evidence_gate.self_consistency import compute_agreement

# Weights for the non-hard-gated signals. schema_valid is a hard gate (fail
# closed), not part of this weighted blend.
_AGREEMENT_WEIGHT = 0.6
_COUNTERFACTUAL_WEIGHT = 0.4


@dataclass
class GateThresholds:
    accept_confidence: float = 0.6
    adjudication_band: tuple[float, float] = (0.4, 0.6)


class EvidenceGate:
    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds()

    def score(
        self,
        candidate_output: str | dict,
        response_schema: dict | None,
        agreement_samples: list,
        counterfactual_output: dict | None = None,
        invariant_fields: list[str] | None = None,
        adjudicated: bool | None = None,
    ) -> EvidenceScore:
        schema_valid, _errors = validate_schema(candidate_output, response_schema)
        agreement_score = compute_agreement(agreement_samples)

        if counterfactual_output is not None and invariant_fields:
            parsed = candidate_output if isinstance(candidate_output, dict) else {}
            cf_pass = counterfactual_pass(parsed, counterfactual_output, invariant_fields)
        else:
            # nothing to check against -> inconclusive, treated as non-blocking
            cf_pass = True

        confidence = (
            _AGREEMENT_WEIGHT * agreement_score + _COUNTERFACTUAL_WEIGHT * (1.0 if cf_pass else 0.0)
            if schema_valid
            else 0.0
        )

        if adjudicated is not None:
            confidence = 1.0 if adjudicated else 0.0

        return EvidenceScore(
            schema_valid=schema_valid,
            agreement_score=agreement_score,
            counterfactual_pass=cf_pass,
            adjudicated=adjudicated,
            confidence=confidence,
        )

    def needs_adjudication(self, evidence: EvidenceScore) -> bool:
        if evidence.adjudicated is not None:
            return False  # already adjudicated
        if not evidence.schema_valid:
            return False  # hard-rejected already, adjudication wouldn't help
        lo, hi = self.thresholds.adjudication_band
        return lo <= evidence.confidence <= hi

    def accept(self, evidence: EvidenceScore) -> bool:
        return (
            evidence.schema_valid
            and evidence.counterfactual_pass
            and evidence.confidence >= self.thresholds.accept_confidence
        )
