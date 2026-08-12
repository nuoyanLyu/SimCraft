"""Significance machinery for the playbook A/B.

Both arms of every comparison here run the *same* tasks (or the same benchmark
entries) against the *same* agent, differing only in the injected playbook
text. That pairing is the whole reason these numbers are worth anything: the
between-task variance in a tool-use eval dwarfs any plausible playbook effect,
so an unpaired comparison of two pass rates would need far more tasks than the
GPU budget allows to see a real effect at all.

Stdlib only, deliberately. A seeded `random.Random` makes every reported
interval reproducible from the results file alone, and the sample sizes here
(tens of tasks, hundreds of benchmark entries) are nowhere near needing numpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# 10k resamples puts the Monte-Carlo error on a 95% bound at well under a
# percentage point, which is far finer than the sampling noise of the pass rate
# it is bounding.
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def pass_rate(verdicts) -> float | None:
    """Pass rate over verdicts, ignoring `None` (a rollout that never produced
    a judgement at all). Returns None when nothing is left to average."""
    scored = [v for v in verdicts if v is not None]
    if not scored:
        return None
    return sum(1 for v in scored if v) / len(scored)


def wilson_interval(k: int, n: int, alpha: float = DEFAULT_ALPHA) -> Interval:
    """Wilson score interval for a single arm's pass rate.

    Not Wald: at the rates these evals live at (often below 0.2, sometimes 0)
    the normal approximation puts the lower bound below zero and the interval
    stops meaning anything. Wilson stays inside [0, 1] and behaves at the ends.
    """
    if n <= 0:
        return Interval(0.0, 0.0, 0.0)
    z = _z_for(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_ci(
    pairs: list[tuple[float, float]],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Interval:
    """Percentile CI for mean(final - base), resampling *units*, not rollouts.

    The unit is the task (or benchmark entry). Resampling individual rollouts
    instead would treat the 5 reps of one task as 5 independent observations
    and shrink the interval by roughly sqrt(reps) — the tasks are the thing
    being generalised over, and there are only ever a few dozen of them.
    """
    if not pairs:
        return Interval(0.0, 0.0, 0.0)
    deltas = [final - base for base, final in pairs]
    point = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        # One unit carries no information about between-unit spread; report the
        # point estimate with an interval that admits as much.
        return Interval(point, -1.0, 1.0)

    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return Interval(point, _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5.

    Symmetric under p=0.5, so doubling the smaller tail is exact rather than
    an approximation of the "sum of outcomes at most as likely" definition.
    """
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def mcnemar_exact_p(only_base_passed: int, only_final_passed: int) -> float:
    """Exact McNemar over the discordant pairs.

    Pairs where both arms agree carry no information about which arm is
    better — with the model, the task and the temperature held fixed, they are
    the cases the playbook did not touch. Under the null "the playbook changes
    nothing", each of the remaining pairs is an independent coin flip.
    """
    discordant = only_base_passed + only_final_passed
    return binomial_two_sided_p(only_final_passed, discordant)


def discordant_counts(pairs: list[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    """(both_passed, only_base, only_final, both_failed) over boolean pairs."""
    both = only_base = only_final = neither = 0
    for base, final in pairs:
        if base and final:
            both += 1
        elif base and not final:
            only_base += 1
        elif final and not base:
            only_final += 1
        else:
            neither += 1
    return both, only_base, only_final, neither


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _z_for(alpha: float) -> float:
    """Normal quantile at 1 - alpha/2, for the handful of alphas anyone uses.

    Falls back to an inverse-CDF approximation rather than raising: an unusual
    alpha should widen or narrow the interval, not abort the run that spent
    hours producing the numbers.
    """
    table = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
    if alpha in table:
        return table[alpha]
    return _norm_ppf(1 - alpha / 2)


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the normal inverse CDF (|err| < 1e-9
    over the range that matters here)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
