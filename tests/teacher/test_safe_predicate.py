import pytest

from qwen_agentworld.teacher.safe_predicate import evaluate_step_wise_predicate, is_trivial_tautology, UnsafePredicateError, evaluate_predicate, validate_predicate_ast


def test_accepts_simple_comparison():
    validate_predicate_ast("state['status'] == 'completed'")


def test_accepts_get_and_len_and_boolop():
    validate_predicate_ast("len(state.get('notes', [])) > 0 and state['done'] is True")


def test_accepts_any_over_comprehension_with_bound_variable():
    validate_predicate_ast("any(tag == 'security' for tag in state['doc']['tags'])")


def test_accepts_slice_indexing():
    validate_predicate_ast("state['items'][:3] == ['a', 'b', 'c']")


def test_evaluate_predicate_with_slice():
    assert evaluate_predicate("state['items'][-1] == 'done'", {"items": ["a", "done"]}) is True


def test_evaluate_predicate_with_any_comprehension():
    assert evaluate_predicate("any(t == 'security' for t in state['tags'])", {"tags": ["hr", "security"]}) is True


def test_rejects_unbound_comprehension_variable_used_outside_its_loop():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("tag == 'x'")


def test_rejects_import():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("__import__('os').system('echo hi')")


def test_rejects_dunder_attribute_sandbox_escape():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("().__class__.__bases__[0]")


def test_rejects_disallowed_call():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("eval('1')")


def test_rejects_unknown_identifier():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("other_var == 1")


def test_rejects_syntax_error():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("state[")


def test_evaluate_predicate_true():
    assert evaluate_predicate("state['status'] == 'completed'", {"status": "completed"}) is True


def test_evaluate_predicate_false():
    assert evaluate_predicate("len(state.get('items', [])) > 3", {"items": [1, 2]}) is False


def test_evaluate_step_wise_predicate_true_when_intermediate_state_satisfies():
    states = [{"notes": []}, {"notes": [{"id": "x"}]}, {"notes": []}]
    pred = "any(any(n['id'] == 'x' for n in s['notes']) for s in states) and len(states[-1]['notes']) == 0"
    assert evaluate_step_wise_predicate(pred, states) is True


def test_evaluate_step_wise_predicate_false_when_never_created():
    states = [{"notes": []}, {"notes": []}]
    pred = "any(any(n['id'] == 'x' for n in s['notes']) for s in states) and len(states[-1]['notes']) == 0"
    assert evaluate_step_wise_predicate(pred, states) is False


def test_is_trivial_tautology_detects_eq_or_neq():
    assert is_trivial_tautology("state['x'] == 'a' or state['x'] != 'a'") is True
    assert is_trivial_tautology("state['x'] != 'a' or state['x'] == 'a'") is True


def test_is_trivial_tautology_false_for_real_predicate():
    assert is_trivial_tautology("state['x'] == 'a' or state['y'] == 'b'") is False


def test_validate_rejects_constant_predicate():
    with pytest.raises(UnsafePredicateError):
        validate_predicate_ast("1 == 1")
