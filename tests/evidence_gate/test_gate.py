from qwen_agentworld.evidence_gate.gate import EvidenceGate, GateThresholds

SCHEMA = {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}


def test_high_agreement_and_valid_schema_is_accepted():
    gate = EvidenceGate()
    samples = [{"status": "ok"}] * 3
    evidence = gate.score(candidate_output={"status": "ok"}, response_schema=SCHEMA, agreement_samples=samples)
    assert evidence.schema_valid
    assert gate.accept(evidence)


def test_schema_invalid_is_hard_rejected_regardless_of_agreement():
    gate = EvidenceGate()
    samples = [{"status": "ok"}] * 3  # high agreement, but...
    evidence = gate.score(candidate_output={"wrong_field": "ok"}, response_schema=SCHEMA, agreement_samples=samples)
    assert not evidence.schema_valid
    assert evidence.confidence == 0.0
    assert not gate.accept(evidence)


def test_low_agreement_lands_in_adjudication_band():
    gate = EvidenceGate(thresholds=GateThresholds(accept_confidence=0.7, adjudication_band=(0.3, 0.65)))
    samples = [{"status": "ok"}, {"error": "not found", "code": 404}]
    evidence = gate.score(candidate_output={"status": "ok"}, response_schema=SCHEMA, agreement_samples=samples)
    assert gate.needs_adjudication(evidence)
    assert not gate.accept(evidence)


def test_adjudication_overrides_confidence():
    gate = EvidenceGate()
    samples = [{"status": "ok"}, {"status": "error"}]
    evidence = gate.score(
        candidate_output={"status": "ok"}, response_schema=SCHEMA, agreement_samples=samples, adjudicated=True
    )
    assert evidence.confidence == 1.0
    assert not gate.needs_adjudication(evidence)  # already adjudicated


def test_counterfactual_drift_blocks_acceptance_even_with_high_agreement():
    gate = EvidenceGate()
    samples = [{"status": "ok", "resource": {"id": "r1"}}] * 3
    evidence = gate.score(
        candidate_output={"status": "ok", "resource": {"id": "r1"}},
        response_schema=SCHEMA,
        agreement_samples=samples,
        counterfactual_output={"status": "ok", "resource": {"id": "DIFFERENT"}},
        invariant_fields=["resource.id"],
    )
    assert not evidence.counterfactual_pass
    assert not gate.accept(evidence)
