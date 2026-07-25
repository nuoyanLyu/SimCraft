from qwen_agentworld.core.schemas import CheckerSpec
from qwen_agentworld.judge.paired_audit import judge_checker, paired_audit


def checker(predicate: str) -> CheckerSpec:
    return CheckerSpec(executable_predicate=predicate)


def test_judge_checker_true_and_false():
    assert judge_checker(checker("state['status'] == 'done'"), {"status": "done"}) is True
    assert judge_checker(checker("state['status'] == 'done'"), {"status": "pending"}) is False


def test_judge_checker_returns_false_on_unsafe_predicate_instead_of_raising():
    assert judge_checker(checker("__import__('os').system('echo hi')"), {}) is False


def test_paired_audit_genuinely_improved_only_on_fail_to_pass_flip():
    checker_spec = checker("state['status'] == 'done'")
    result = paired_audit(
        "task_1", checker_spec, final_state_with_playbook={"status": "done"}, final_state_without_playbook={"status": "pending"}
    )
    assert result.genuinely_improved is True
    assert result.regressed is False


def test_paired_audit_not_credited_when_both_pass():
    checker_spec = checker("state['status'] == 'done'")
    result = paired_audit(
        "task_1", checker_spec, final_state_with_playbook={"status": "done"}, final_state_without_playbook={"status": "done"}
    )
    assert result.genuinely_improved is False
    assert result.regressed is False


def test_paired_audit_flags_regression():
    checker_spec = checker("state['status'] == 'done'")
    result = paired_audit(
        "task_1", checker_spec, final_state_with_playbook={"status": "pending"}, final_state_without_playbook={"status": "done"}
    )
    assert result.regressed is True
    assert result.genuinely_improved is False
