from qwen_agentworld.evidence_gate.self_consistency import compute_agreement, pairwise_similarity


def test_identical_samples_have_perfect_agreement():
    samples = [{"a": 1, "b": "x"}] * 3
    assert compute_agreement(samples) == 1.0


def test_wildly_different_samples_have_low_agreement():
    samples = [{"status": "ok"}, {"error": "not found", "code": 404}]
    assert compute_agreement(samples) < 0.5


def test_empty_and_single_sample_edge_cases():
    assert compute_agreement([]) == 0.0
    assert compute_agreement([{"a": 1}]) == 1.0


def test_pairwise_similarity_is_symmetric():
    a, b = {"x": 1}, {"x": 2}
    assert pairwise_similarity(a, b) == pairwise_similarity(b, a)
