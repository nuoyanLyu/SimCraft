"""Rolling pass-rate tracking, bucketed by (tool_family, graph_complexity).

The research plan targets a pass-rate band (default 20-60%) so tasks are
neither trivial nor impossible for the current agent+playbook. A single
point-estimate pass rate is noisy at small sample counts, so this tracks a
rolling window per bucket rather than an all-time average — old evidence from a
since-mutated playbook shouldn't keep dragging on the current estimate.

This used to also hand the orchestrator a +/-1 nudge to `graph_complexity`,
on the assumption that a longer tool chain is a harder task. Measurement on
2026-07-29 killed that assumption: 83 screened gc=3 tasks spanned pass rates
0.0 to 1.0, and no static property of a task predicted where in that range it
landed (chain length, number of conditions in the reward function, predicate
length, initial-state size — best correlation r=-0.21, p=0.053, best of six).
A gc=4 sample came out *easier* than gc=3. So the band no longer moves a
difficulty dial; it selects tasks by their measured rate, and the rate is the
difficulty. See `OrchestratorConfig.difficulty_band`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

DEFAULT_DIFFICULTY_BAND = (0.2, 0.6)


def _bucket_key(tool_family: str, graph_complexity: int) -> tuple[str, int]:
    return (tool_family, graph_complexity)


@dataclass
class ProbeResult:
    tool_family: str
    graph_complexity: int
    n_samples: int
    pass_rate: float | None
    in_band: bool | None


class RollingPassRateTracker:
    """Tracks pass/fail outcomes per (tool_family, graph_complexity) bucket
    in a fixed-size sliding window, so the estimate reflects recent behavior
    under the current playbook rather than the whole run's history.
    """

    def __init__(self, window: int = 20, band: tuple[float, float] = DEFAULT_DIFFICULTY_BAND) -> None:
        self.window = window
        self.band = band
        self._buckets: dict[tuple[str, int], deque[bool]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, tool_family: str, graph_complexity: int, passed: bool) -> None:
        self._buckets[_bucket_key(tool_family, graph_complexity)].append(passed)

    def pass_rate(self, tool_family: str, graph_complexity: int) -> float | None:
        bucket = self._buckets.get(_bucket_key(tool_family, graph_complexity))
        if not bucket:
            return None
        return sum(bucket) / len(bucket)

    def is_in_band(self, tool_family: str, graph_complexity: int) -> bool | None:
        rate = self.pass_rate(tool_family, graph_complexity)
        if rate is None:
            return None
        lo, hi = self.band
        return lo <= rate <= hi

    def result(self, tool_family: str, graph_complexity: int) -> ProbeResult:
        bucket = self._buckets.get(_bucket_key(tool_family, graph_complexity), ())
        rate = self.pass_rate(tool_family, graph_complexity)
        return ProbeResult(
            tool_family=tool_family,
            graph_complexity=graph_complexity,
            n_samples=len(bucket),
            pass_rate=rate,
            in_band=self.is_in_band(tool_family, graph_complexity),
        )

    def all_results(self) -> list[ProbeResult]:
        return [self.result(tf, gc) for (tf, gc) in self._buckets]
