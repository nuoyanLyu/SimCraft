"""Rolling pass-rate tracking, bucketed by (tool_family, graph_complexity).

This feeds `teacher/task_generator.py`'s difficulty control: the research
plan targets a pass-rate band (default 20-60%) so tasks are neither trivial
nor impossible for the current agent+playbook. A single point-estimate pass
rate is noisy at small sample counts, so this tracks a rolling window per
bucket rather than an all-time average — old evidence from a since-mutated
playbook shouldn't keep dragging on the current estimate.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

_DEFAULT_BAND = (0.2, 0.6)


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

    def __init__(self, window: int = 20, band: tuple[float, float] = _DEFAULT_BAND) -> None:
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


def suggest_difficulty_adjustment(result: ProbeResult, band: tuple[float, float] = _DEFAULT_BAND) -> int:
    """Returns a signed adjustment to `graph_complexity` for the next batch
    of sampled tasks: +1 if the agent is passing too easily (raise
    difficulty), -1 if it's failing too much (lower it), 0 if in-band or
    under-sampled (not enough evidence to move yet, needs >= 5 samples).
    """
    if result.n_samples < 5 or result.pass_rate is None or result.in_band:
        return 0
    _, hi = band
    return 1 if result.pass_rate > hi else -1
