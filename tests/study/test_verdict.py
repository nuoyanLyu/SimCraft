"""Tests for the decision rule.

Two things are being protected here. First, that a verdict follows the
confidence interval and never the point estimate — with two dozen tasks the
point estimate moves further than the effect being looked for. Second, that a
run which could not have shown an effect is reported as `void` rather than as
"no effect": those are the same numbers and opposite conclusions, and only one
of them is about the method.
"""

import pytest

from qwen_agentworld.study.stats import Interval
from qwen_agentworld.study.verdict import (
    AxisResult,
    Precondition,
    bfcl_axis,
    build_report,
    check_preconditions,
    classify,
    format_report,
    in_domain_axis,
)


def axis(name, delta_point, low, high, verdict):
    return AxisResult(
        name=name, baseline=0.3, evolved=0.3 + delta_point,
        delta=Interval(delta_point, low, high), n_units=20, p_value=0.01, verdict=verdict,
    )


def ok_preconditions():
    return check_preconditions(
        baseline_fingerprint="aaa",
        evolved_fingerprint="bbb",
        n_entries=7,
        n_playbook_edits=4,
        train_task_ids={"t1"},
        eval_task_ids={"t2"},
        baseline_pass_rate=0.3,
    )


# ------------------------------------------------------------- classify --


def test_a_positive_point_estimate_alone_is_not_a_gain():
    """The single most tempting error available in this study."""
    assert classify(Interval(0.12, -0.05, 0.29)) == "no_effect"


def test_an_interval_clear_of_zero_is_a_gain():
    assert classify(Interval(0.12, 0.03, 0.21)) == "gain"


def test_an_interval_entirely_below_zero_is_a_regression():
    assert classify(Interval(-0.12, -0.21, -0.03)) == "regression"


# ------------------------------------------------------------ in domain --


def test_in_domain_pairs_by_task_and_averages_over_reps():
    base = {"t1": [False, False], "t2": [True, False]}
    evolved = {"t1": [True, True], "t2": [True, True]}
    result = in_domain_axis(base, evolved)
    assert result.baseline == pytest.approx(0.25)
    assert result.evolved == pytest.approx(1.0)
    assert result.n_units == 2


def test_the_unit_is_the_task_not_the_rollout():
    """Counting each rep as its own paired observation would inflate n by the
    rep count while the real sample is the number of tasks."""
    base = {"t1": [False] * 10, "t2": [False] * 10}
    assert in_domain_axis(base, {"t1": [True] * 10, "t2": [True] * 10}).n_units == 2


def test_a_task_scored_in_only_one_arm_is_dropped():
    result = in_domain_axis({"t1": [True], "t2": [False]}, {"t1": [True]})
    assert result.n_units == 1


def test_a_task_whose_every_rollout_errored_is_dropped_not_counted_as_failed():
    result = in_domain_axis({"t1": [None, None], "t2": [False]}, {"t1": [True], "t2": [True]})
    assert result.n_units == 1


def test_no_shared_task_yields_an_empty_axis_rather_than_a_crash():
    result = in_domain_axis({"t1": [True]}, {"t9": [True]})
    assert result.n_units == 0
    assert "no task was scored in both arms" in result.notes


def test_an_eval_set_where_nothing_moved_says_so_in_the_notes():
    """The signature of a ceiling/floor eval set, which reads identically to
    an inert playbook unless the report names the possibility."""
    identical = {"t1": [True], "t2": [False]}
    result = in_domain_axis(identical, dict(identical))
    assert any("no task changed verdict" in n for n in result.notes)


def test_a_consistent_per_task_improvement_is_called_a_gain():
    base = {f"t{i}": [False] for i in range(16)}
    evolved = {f"t{i}": [True] for i in range(16)}
    assert in_domain_axis(base, evolved).verdict == "gain"


# ----------------------------------------------------------------- bfcl --


def test_bfcl_axis_reports_which_entries_the_playbook_fixed_and_broke():
    paired = [("a", False, True), ("b", False, True), ("c", True, False), ("d", True, True)]
    result = bfcl_axis(paired)
    assert any("2 fixed by the playbook, 1 broken by it" in n for n in result.notes)
    assert result.n_units == 4


def test_bfcl_axis_flags_a_playbook_that_changed_nothing_at_all():
    """Distinguishes 'the guidance did not help' from 'the guidance never
    reached the model' — the second is a plumbing bug, not a result."""
    paired = [("a", True, True), ("b", False, False)]
    assert any("not one entry changed verdict" in n for n in bfcl_axis(paired).notes)


def test_bfcl_axis_on_no_shared_entries_is_empty_not_neutral():
    assert bfcl_axis([]).n_units == 0


def test_the_bfcl_verdict_never_outruns_the_exact_test():
    """On 0/1 pairs the percentile bootstrap runs slightly anti-conservative:
    9 fixed vs 2 broken clears zero by CI but sits at McNemar p=0.065. The
    exact test is the authority when one exists."""
    paired = (
        [(f"f{i}", False, True) for i in range(9)]
        + [(f"b{i}", True, False) for i in range(2)]
        + [(f"p{i}", True, True) for i in range(68)]
        + [(f"n{i}", False, False) for i in range(21)]
    )
    result = bfcl_axis(paired)
    assert result.delta.excludes_zero  # the CI alone would have said "gain"
    assert result.p_value > 0.05
    assert result.verdict == "no_effect"
    assert any("McNemar" in n for n in result.notes)


def test_a_bfcl_gain_that_both_tests_agree_on_is_a_gain():
    paired = (
        [(f"f{i}", False, True) for i in range(20)]
        + [(f"b{i}", True, False) for i in range(3)]
        + [(f"p{i}", True, True) for i in range(77)]
    )
    result = bfcl_axis(paired)
    assert result.p_value < 0.05 and result.verdict == "gain"


def test_a_bfcl_regression_is_also_held_to_the_exact_test():
    """Symmetric: a harm claim on too few discordant entries is as unfounded
    as a gain claim on too few."""
    paired = (
        [(f"b{i}", True, False) for i in range(9)]
        + [(f"f{i}", False, True) for i in range(2)]
        + [(f"p{i}", True, True) for i in range(89)]
    )
    assert bfcl_axis(paired).verdict == "no_effect"


# -------------------------------------------------------- preconditions --


def test_an_empty_evolved_playbook_fails_its_precondition():
    checks = {c.name: c for c in check_preconditions(
        baseline_fingerprint="a", evolved_fingerprint="b", n_entries=0,
        n_playbook_edits=3, train_task_ids=set(), eval_task_ids=set(), baseline_pass_rate=0.2,
    )}
    assert not checks["evolved_playbook_non_empty"].ok


def test_identical_fingerprints_fail_because_the_variable_never_varied():
    checks = {c.name: c for c in check_preconditions(
        baseline_fingerprint="same", evolved_fingerprint="same", n_entries=3,
        n_playbook_edits=3, train_task_ids=set(), eval_task_ids=set(), baseline_pass_rate=0.2,
    )}
    assert not checks["arms_differ"].ok


def test_a_run_that_made_no_playbook_edits_fails():
    """The 2026-07-29 failure mode: four iterations, one edit, and a null A/B
    read as 'the playbook does not help'."""
    checks = {c.name: c for c in check_preconditions(
        baseline_fingerprint="a", evolved_fingerprint="b", n_entries=1,
        n_playbook_edits=0, train_task_ids=set(), eval_task_ids=set(), baseline_pass_rate=0.2,
    )}
    assert not checks["loop_made_edits"].ok


def test_training_on_an_eval_task_fails_the_held_out_check():
    checks = {c.name: c for c in check_preconditions(
        baseline_fingerprint="a", evolved_fingerprint="b", n_entries=3, n_playbook_edits=3,
        train_task_ids={"t1", "t2"}, eval_task_ids={"t2"}, baseline_pass_rate=0.2,
    )}
    assert not checks["eval_held_out"].ok
    assert "memorisation" in checks["eval_held_out"].detail


def test_an_eval_set_at_ceiling_fails_the_headroom_check():
    """A task the baseline already passes cannot pass harder; a set of them
    can only ever produce a null."""
    checks = {c.name: c for c in check_preconditions(
        baseline_fingerprint="a", evolved_fingerprint="b", n_entries=3, n_playbook_edits=3,
        train_task_ids=set(), eval_task_ids=set(), baseline_pass_rate=0.99,
    )}
    assert not checks["eval_has_headroom"].ok


def test_all_preconditions_pass_on_a_well_formed_run():
    assert all(c.ok for c in ok_preconditions())


# --------------------------------------------------------------- report --


def test_a_failed_precondition_voids_the_study_even_with_a_significant_gain():
    """The whole point: an infrastructure failure must not be publishable as a
    scientific result in either direction."""
    broken = check_preconditions(
        baseline_fingerprint="a", evolved_fingerprint="a", n_entries=3, n_playbook_edits=3,
        train_task_ids=set(), eval_task_ids=set(), baseline_pass_rate=0.2,
    )
    report = build_report(broken, axis("in_domain", 0.2, 0.1, 0.3, "gain"), None)
    assert report.verdict == "void"
    assert "never varied" in report.headline


def test_in_domain_gain_with_flat_bfcl_is_supported():
    report = build_report(
        ok_preconditions(),
        axis("in_domain", 0.15, 0.05, 0.25, "gain"),
        axis("bfcl", 0.01, -0.02, 0.04, "no_effect"),
    )
    assert report.verdict == "supported"


def test_in_domain_gain_paid_for_with_a_bfcl_regression_is_not_supported():
    """A playbook that fits the task generator and costs general tool use has
    not produced a transferable skill, whatever the in-domain number says."""
    report = build_report(
        ok_preconditions(),
        axis("in_domain", 0.15, 0.05, 0.25, "gain"),
        axis("bfcl", -0.04, -0.07, -0.01, "regression"),
    )
    assert report.verdict == "in_domain_only"
    assert "fitting the task generator" in report.headline


def test_an_unmeasured_bfcl_is_not_treated_as_a_pass():
    """`--skip-bfcl` must not let the transfer half of the claim succeed by
    default."""
    report = build_report(ok_preconditions(), axis("in_domain", 0.15, 0.05, 0.25, "gain"), None)
    assert report.verdict == "in_domain_only"
    assert "unmeasured" in report.headline


def test_no_in_domain_effect_is_not_supported_regardless_of_bfcl():
    report = build_report(
        ok_preconditions(),
        axis("in_domain", 0.04, -0.06, 0.14, "no_effect"),
        axis("bfcl", 0.05, 0.01, 0.09, "gain"),
    )
    assert report.verdict == "not_supported"


def test_an_in_domain_regression_is_reported_as_one():
    report = build_report(
        ok_preconditions(), axis("in_domain", -0.12, -0.2, -0.04, "regression"), None
    )
    assert report.verdict == "regression"


def test_a_missing_in_domain_arm_voids_rather_than_reporting_bfcl_alone():
    report = build_report(ok_preconditions(), None, axis("bfcl", 0.05, 0.01, 0.09, "gain"))
    assert report.verdict == "void"


def test_the_report_serialises_and_formats_without_a_bfcl_arm():
    report = build_report(ok_preconditions(), axis("in_domain", 0.15, 0.05, 0.25, "gain"), None)
    payload = report.as_dict()
    assert payload["verdict"] == "in_domain_only"
    assert payload["bfcl"] is None
    assert payload["in_domain"]["ci95"] == [0.05, 0.25]
    text = format_report(report)
    assert "in_domain" in text and "preconditions" in text


def test_the_formatted_report_marks_which_precondition_failed():
    failing = [Precondition("arms_differ", False, "both arms fingerprint abc")]
    text = format_report(build_report(failing, None, None))
    assert "[FAIL] arms_differ" in text
