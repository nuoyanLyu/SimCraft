"""Tests for the significance machinery.

These exist because every one of these functions can return a plausible-looking
number while being wrong, and the number goes straight into a claim about
whether the method works.
"""

import pytest

from qwen_agentworld.study.stats import (
    binomial_two_sided_p,
    discordant_counts,
    mcnemar_exact_p,
    paired_bootstrap_ci,
    pass_rate,
    wilson_interval,
)


# ------------------------------------------------------------- pass_rate --


def test_pass_rate_ignores_unjudged_rollouts():
    assert pass_rate([True, False, None, True]) == pytest.approx(2 / 3)


def test_pass_rate_of_nothing_judged_is_none():
    """Not 0.0. An arm where every rollout crashed has no measured pass rate,
    and returning zero would report the crash as perfect failure."""
    assert pass_rate([None, None]) is None
    assert pass_rate([]) is None


# -------------------------------------------------------------- wilson --


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """The reason this is not a Wald interval: at 0/20 the normal
    approximation puts the lower bound below zero."""
    lo = wilson_interval(0, 20)
    hi = wilson_interval(20, 20)
    assert lo.low == 0.0 and 0.0 < lo.high < 0.3
    assert hi.high == 1.0 and 0.7 < hi.low < 1.0


def test_wilson_narrows_as_n_grows():
    small = wilson_interval(5, 10)
    large = wilson_interval(50, 100)
    assert (large.high - large.low) < (small.high - small.low)


def test_wilson_of_an_empty_sample_is_degenerate_not_an_error():
    assert wilson_interval(0, 0) == wilson_interval(0, 0)
    assert wilson_interval(0, 0).high == 0.0


# ----------------------------------------------------------- bootstrap --


def test_bootstrap_ci_brackets_a_real_effect_and_excludes_zero():
    pairs = [(0.0, 1.0)] * 12  # every task fixed by the playbook
    ci = paired_bootstrap_ci(pairs)
    assert ci.point == pytest.approx(1.0)
    assert ci.low > 0.0 and ci.excludes_zero


def test_bootstrap_ci_includes_zero_when_wins_and_losses_cancel():
    pairs = [(0.0, 1.0)] * 6 + [(1.0, 0.0)] * 6
    ci = paired_bootstrap_ci(pairs)
    assert ci.point == pytest.approx(0.0)
    assert ci.low < 0.0 < ci.high
    assert not ci.excludes_zero


def test_bootstrap_is_reproducible_from_the_seed_alone():
    """Every CI in a report has to be recomputable from the results file."""
    pairs = [(0.2, 0.5), (0.0, 0.4), (0.6, 0.6), (0.4, 0.2)]
    assert paired_bootstrap_ci(pairs, seed=7) == paired_bootstrap_ci(pairs, seed=7)


def test_the_seed_does_not_move_the_bounds_enough_to_change_a_verdict():
    """Reproducibility is worth nothing if the answer depends on which seed was
    picked. At the default resample count the Monte-Carlo error has to be far
    smaller than the effects being classified."""
    pairs = [(0.2, 0.5), (0.0, 0.4), (0.6, 0.6), (0.4, 0.2), (0.1, 0.3), (0.5, 0.9)]
    a, b = paired_bootstrap_ci(pairs, seed=1), paired_bootstrap_ci(pairs, seed=99)
    assert abs(a.low - b.low) < 0.05 and abs(a.high - b.high) < 0.05


def test_a_single_task_gets_an_interval_that_admits_it_knows_nothing():
    """One unit carries no between-unit spread. Returning a zero-width
    interval would let n=1 declare significance."""
    ci = paired_bootstrap_ci([(0.0, 1.0)])
    assert ci.point == 1.0
    assert not ci.excludes_zero


def test_bootstrap_of_no_pairs_is_degenerate_not_an_error():
    assert paired_bootstrap_ci([]).point == 0.0


# ------------------------------------------------------------ mcnemar --


def test_mcnemar_is_significant_when_the_playbook_only_ever_fixes_things():
    assert mcnemar_exact_p(only_base_passed=0, only_final_passed=10) < 0.01


def test_mcnemar_is_not_significant_on_a_balanced_swap():
    assert mcnemar_exact_p(only_base_passed=5, only_final_passed=5) == pytest.approx(1.0)


def test_mcnemar_ignores_the_concordant_pairs():
    """With model, task and temperature fixed, an entry both arms pass is one
    the playbook did not touch — it must not dilute the p-value."""
    assert mcnemar_exact_p(1, 8) == mcnemar_exact_p(1, 8)  # concordant count never enters


def test_mcnemar_with_no_discordant_pairs_is_p_one():
    """Nothing changed, so nothing is evidence of change."""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_binomial_two_sided_is_symmetric():
    assert binomial_two_sided_p(2, 10) == pytest.approx(binomial_two_sided_p(8, 10))


def test_binomial_of_a_fair_split_is_one():
    assert binomial_two_sided_p(5, 10) == pytest.approx(1.0)


# --------------------------------------------------------- discordant --


def test_discordant_counts_partitions_every_pair():
    pairs = [(True, True), (True, False), (False, True), (False, False), (False, True)]
    both, only_base, only_final, neither = discordant_counts(pairs)
    assert (both, only_base, only_final, neither) == (1, 1, 2, 1)
    assert both + only_base + only_final + neither == len(pairs)
