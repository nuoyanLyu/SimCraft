import pytest

from qwen_agentworld.capability_probe.prober import RollingPassRateTracker, suggest_difficulty_adjustment


def test_pass_rate_none_when_unrecorded():
    tracker = RollingPassRateTracker()
    assert tracker.pass_rate("mcp_A", 2) is None
    assert tracker.is_in_band("mcp_A", 2) is None


def test_pass_rate_and_band_membership():
    tracker = RollingPassRateTracker(window=10, band=(0.2, 0.6))
    for passed in [True, True, True, False, False]:  # 3/5 = 0.6
        tracker.record("mcp_A", 2, passed)
    assert tracker.pass_rate("mcp_A", 2) == 0.6
    assert tracker.is_in_band("mcp_A", 2) is True


def test_window_evicts_oldest_samples():
    tracker = RollingPassRateTracker(window=3)
    for passed in [True, True, True]:
        tracker.record("mcp_A", 2, passed)
    assert tracker.pass_rate("mcp_A", 2) == 1.0
    tracker.record("mcp_A", 2, False)  # evicts the first True
    assert tracker.pass_rate("mcp_A", 2) == pytest.approx(2 / 3)


def test_buckets_are_independent_per_tool_family_and_complexity():
    tracker = RollingPassRateTracker()
    tracker.record("mcp_A", 2, True)
    tracker.record("mcp_A", 3, False)
    tracker.record("mcp_B", 2, False)
    assert tracker.pass_rate("mcp_A", 2) == 1.0
    assert tracker.pass_rate("mcp_A", 3) == 0.0
    assert tracker.pass_rate("mcp_B", 2) == 0.0


def test_all_results_covers_every_bucket():
    tracker = RollingPassRateTracker()
    tracker.record("mcp_A", 2, True)
    tracker.record("mcp_B", 3, False)
    results = tracker.all_results()
    keys = {(r.tool_family, r.graph_complexity) for r in results}
    assert keys == {("mcp_A", 2), ("mcp_B", 3)}


def test_suggest_difficulty_adjustment_needs_minimum_samples():
    tracker = RollingPassRateTracker()
    tracker.record("mcp_A", 2, True)  # only 1 sample
    result = tracker.result("mcp_A", 2)
    assert suggest_difficulty_adjustment(result) == 0


def test_suggest_difficulty_adjustment_raises_when_too_easy():
    tracker = RollingPassRateTracker()
    for _ in range(6):
        tracker.record("mcp_A", 2, True)  # pass_rate 1.0, above band hi=0.6
    result = tracker.result("mcp_A", 2)
    assert suggest_difficulty_adjustment(result) == 1


def test_suggest_difficulty_adjustment_lowers_when_too_hard():
    tracker = RollingPassRateTracker()
    for _ in range(6):
        tracker.record("mcp_A", 2, False)  # pass_rate 0.0, below band lo=0.2
    result = tracker.result("mcp_A", 2)
    assert suggest_difficulty_adjustment(result) == -1


def test_suggest_difficulty_adjustment_holds_when_in_band():
    tracker = RollingPassRateTracker()
    for passed in [True, True, True, False, False, False]:  # 0.5, in [0.2, 0.6]
        tracker.record("mcp_A", 2, passed)
    result = tracker.result("mcp_A", 2)
    assert suggest_difficulty_adjustment(result) == 0
