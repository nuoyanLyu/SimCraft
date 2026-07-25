from qwen_agentworld.evidence_gate.counterfactual_replay import (
    build_counterfactual_probe,
    counterfactual_pass,
    get_path,
)


def test_get_path_nested():
    obj = {"resource": {"id": "r1", "meta": {"count": 3}}}
    assert get_path(obj, "resource.id") == "r1"
    assert get_path(obj, "resource.meta.count") == 3
    assert get_path(obj, "resource.missing") is None


def test_counterfactual_pass_when_invariants_hold():
    original = {"resource": {"id": "r1"}, "unrelated_field": "before"}
    perturbed = {"resource": {"id": "r1"}, "unrelated_field": "after"}
    assert counterfactual_pass(original, perturbed, invariant_fields=["resource.id"])


def test_counterfactual_fails_when_invariant_drifts():
    original = {"resource": {"id": "r1"}}
    perturbed = {"resource": {"id": "r2"}}
    assert not counterfactual_pass(original, perturbed, invariant_fields=["resource.id"])


def test_empty_invariants_trivially_pass():
    assert counterfactual_pass({"a": 1}, {"a": 2}, invariant_fields=[])


def test_build_counterfactual_probe_perturbs_an_untouched_key():
    prior_state = {"a": 0, "b": "same"}
    next_state = {"a": 1, "b": "same"}

    result = build_counterfactual_probe(prior_state, next_state)

    assert result is not None
    perturbed_state, invariant_fields = result
    assert invariant_fields == ["a"]
    assert perturbed_state["a"] == 0  # the touched/invariant key is left alone
    assert perturbed_state["b"] != "same"  # the untouched key is what got perturbed
    assert prior_state == {"a": 0, "b": "same"}  # original left untouched (deep-copied)


def test_build_counterfactual_probe_returns_none_when_prior_state_has_no_untouched_key():
    assert build_counterfactual_probe({}, {"a": 1}) is None
    assert build_counterfactual_probe({"a": 0}, {"a": 1}) is None


def test_build_counterfactual_probe_treats_added_and_removed_keys_as_touched():
    prior_state = {"a": 0, "b": "keep"}
    next_state = {"a": 0, "c": "new"}  # b removed, c added, a untouched

    result = build_counterfactual_probe(prior_state, next_state)

    assert result is not None
    perturbed_state, invariant_fields = result
    assert invariant_fields == ["b", "c"]
    assert perturbed_state["a"] != 0
