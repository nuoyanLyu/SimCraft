import pytest

from qwen_agentworld.teacher.safe_predicate import UnsafePredicateError, evaluate_predicate, validate_predicate_ast


def test_accepts_simple_comparison():
    validate_predicate_ast("state['status'] == 'completed'")


def test_accepts_get_and_len_and_boolop():
    validate_predicate_ast("len(state.get('notes', [])) > 0 and state['done'] is True")


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
