"""The decision rule: from two arms' raw verdicts to one claim about the playbook.

The claim under test has two halves, and they are not interchangeable:

  (1) in-domain — on held-out tasks from the same generator the playbook was
      evolved against, does the agent pass more of them?
  (2) transfer  — on a real external benchmark (BFCL) the playbook never saw,
      is general tool-use at least not *worse*?

(1) is the effect being claimed. (2) is a harm check, not a second claim: a
playbook that lifts in-domain pass rate by memorising the task generator's
quirks and costs three points of BFCL has not produced a transferable skill,
and the report should say so in those words rather than leading with the
in-domain number.

The other job of this module is refusing to answer. Most ways this study can
fail produce a *null* result that is indistinguishable, in the numbers alone,
from an honest "the playbook does not help": an evolve run that made zero
edits, a stale checkpoint that loaded empty, an eval set the agent already
passes at ceiling. Those are checked first and reported as `void`, because
publishing "no gain" off a run that never varied its independent variable is
the single most expensive mistake available here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qwen_agentworld.study.stats import (
    Interval,
    binomial_two_sided_p,
    discordant_counts,
    mcnemar_exact_p,
    paired_bootstrap_ci,
    wilson_interval,
)

AxisVerdict = Literal["gain", "no_effect", "regression"]
OverallVerdict = Literal[
    "void",
    "supported",
    "in_domain_only",
    "not_supported",
    "regression",
]


@dataclass
class Precondition:
    name: str
    ok: bool
    detail: str


@dataclass
class AxisResult:
    name: str
    baseline: float
    evolved: float
    delta: Interval
    n_units: int
    p_value: float | None = None
    verdict: AxisVerdict = "no_effect"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "evolved": self.evolved,
            "delta": self.delta.point,
            "ci95": [self.delta.low, self.delta.high],
            "n_units": self.n_units,
            "p_value": self.p_value,
            "verdict": self.verdict,
            "notes": self.notes,
        }


def classify(delta: Interval) -> AxisVerdict:
    """A verdict from the interval, never from the point estimate.

    `delta.point > 0` is not evidence of anything: with 16 tasks and 5 reps the
    point estimate moves by more than the effect size being looked for. Only a
    CI that clears zero counts.
    """
    if delta.low > 0.0:
        return "gain"
    if delta.high < 0.0:
        return "regression"
    return "no_effect"


# --------------------------------------------------------------------------- #
# In-domain axis
# --------------------------------------------------------------------------- #


def in_domain_axis(
    per_task_baseline: dict[str, list[bool | None]],
    per_task_evolved: dict[str, list[bool | None]],
    *,
    seed: int = 0,
) -> AxisResult:
    """Paired over tasks, from `ab_test.py`'s `results.json` per-task verdicts.

    The unit is the task and the value is that task's mean over reps. Treating
    each rep as its own paired observation would be wrong twice over: the reps
    are not matched to each other in any way (rep 3 of the baseline arm has no
    special relationship to rep 3 of the evolved arm), and it would inflate n
    by the rep count while the real sample is the number of tasks.
    """
    pairs: list[tuple[float, float]] = []
    task_ids: list[str] = []
    for task_id, base_verdicts in per_task_baseline.items():
        evolved_verdicts = per_task_evolved.get(task_id)
        if evolved_verdicts is None:
            continue
        base = _mean(base_verdicts)
        evolved = _mean(evolved_verdicts)
        if base is None or evolved is None:
            continue  # every rollout errored in one arm; no verdict to pair
        pairs.append((base, evolved))
        task_ids.append(task_id)

    if not pairs:
        return AxisResult(
            name="in_domain",
            baseline=0.0,
            evolved=0.0,
            delta=Interval(0.0, 0.0, 0.0),
            n_units=0,
            notes=["no task was scored in both arms"],
        )

    baseline = sum(b for b, _ in pairs) / len(pairs)
    evolved = sum(f for _, f in pairs) / len(pairs)
    delta = paired_bootstrap_ci(pairs, seed=seed)

    wins = sum(1 for b, f in pairs if f > b)
    losses = sum(1 for b, f in pairs if f < b)
    ties = len(pairs) - wins - losses
    # Sign test over the tasks that moved at all. Reported alongside the
    # bootstrap rather than instead of it: it ignores effect size entirely, so
    # it can call a run significant that moved 9 tasks by 0.2 each, and can
    # miss one that moved 3 tasks from 0 to 1.
    p_value = binomial_two_sided_p(wins, wins + losses)

    notes = [f"per-task wins={wins} losses={losses} ties={ties}"]
    if ties == len(pairs):
        notes.append(
            "no task changed verdict in either direction — either the playbook is inert "
            "or the eval set has no headroom; check the ceiling precondition"
        )

    return AxisResult(
        name="in_domain",
        baseline=baseline,
        evolved=evolved,
        delta=delta,
        n_units=len(pairs),
        p_value=p_value,
        verdict=classify(delta),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Transfer axis (BFCL)
# --------------------------------------------------------------------------- #


def bfcl_axis(
    paired: list[tuple[str, bool, bool]],
    *,
    seed: int = 0,
    notes=None,
    alpha: float = 0.05,
) -> AxisResult:
    """Paired over benchmark entries; one rollout each, so McNemar applies exactly.

    Unlike the in-domain axis, the verdict here needs the bootstrap CI *and*
    McNemar to agree. They are measuring the same paired binary quantity, and
    on 0/1 pairs the percentile bootstrap runs slightly anti-conservative — it
    will clear zero at a handful of discordant entries where the exact test
    puts p near 0.07. When an exact test is available it is the authority, and
    a transfer claim built on eleven entries that disagreed is not one worth
    printing.
    """
    notes = list(notes or [])
    if not paired:
        return AxisResult(
            name="bfcl",
            baseline=0.0,
            evolved=0.0,
            delta=Interval(0.0, 0.0, 0.0),
            n_units=0,
            notes=notes + ["no entry ran in both arms"],
        )

    bools = [(base, evolved) for _, base, evolved in paired]
    both, only_base, only_evolved, neither = discordant_counts(bools)
    baseline = sum(1 for b, _ in bools if b) / len(bools)
    evolved = sum(1 for _, f in bools if f) / len(bools)
    delta = paired_bootstrap_ci([(float(b), float(f)) for b, f in bools], seed=seed)
    p_value = mcnemar_exact_p(only_base, only_evolved)

    notes.append(
        f"discordant: {only_evolved} fixed by the playbook, {only_base} broken by it "
        f"({both} passed in both, {neither} failed in both)"
    )
    if only_base + only_evolved == 0:
        notes.append(
            "not one entry changed verdict — the playbook text reached the model but "
            "changed nothing it did; check the prompt actually carried through"
        )

    verdict = classify(delta)
    if verdict != "no_effect" and p_value > alpha:
        notes.append(
            f"CI clears zero but McNemar p={p_value:.3f} does not; reported as no_effect "
            f"({only_base + only_evolved} discordant entries is too few to call)"
        )
        verdict = "no_effect"

    return AxisResult(
        name="bfcl",
        baseline=baseline,
        evolved=evolved,
        delta=delta,
        n_units=len(bools),
        p_value=p_value,
        verdict=verdict,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #


def check_preconditions(
    *,
    baseline_fingerprint: str,
    evolved_fingerprint: str,
    n_entries: int,
    n_playbook_edits: int,
    train_task_ids: set[str],
    eval_task_ids: set[str],
    baseline_pass_rate: float | None,
) -> list[Precondition]:
    """Everything that makes a null result uninterpretable, checked up front.

    Each of these has actually happened in a recorded run, and each produced a
    number that looked exactly like "the playbook does not help".
    """
    checks = [
        Precondition(
            "evolved_playbook_non_empty",
            n_entries > 0,
            f"the evolved arm carries {n_entries} entries"
            + ("" if n_entries else " — it is the baseline arm under another name"),
        ),
        Precondition(
            "arms_differ",
            baseline_fingerprint != evolved_fingerprint,
            "baseline and evolved playbooks fingerprint differently"
            if baseline_fingerprint != evolved_fingerprint
            else f"both arms fingerprint {evolved_fingerprint}: the independent variable never varied",
        ),
        Precondition(
            "loop_made_edits",
            n_playbook_edits > 0,
            f"the evolve run applied {n_playbook_edits} playbook edit(s)"
            + ("" if n_playbook_edits else " — nothing was learned, so nothing can be measured"),
        ),
        Precondition(
            "eval_held_out",
            not (train_task_ids & eval_task_ids),
            "no eval task was seen during evolution"
            if not (train_task_ids & eval_task_ids)
            else f"{len(train_task_ids & eval_task_ids)} eval task(s) were trained on; "
            "the in-domain number measures memorisation, not capability",
        ),
        Precondition(
            "eval_has_headroom",
            baseline_pass_rate is None or baseline_pass_rate < 0.95,
            f"baseline passes {baseline_pass_rate:.2f} of the eval set"
            if baseline_pass_rate is not None
            else "baseline pass rate unknown",
        ),
    ]
    return checks


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class StudyReport:
    verdict: OverallVerdict
    headline: str
    preconditions: list[Precondition]
    in_domain: AxisResult | None
    bfcl: AxisResult | None
    arm_intervals: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "preconditions": [
                {"name": p.name, "ok": p.ok, "detail": p.detail} for p in self.preconditions
            ],
            "in_domain": self.in_domain.as_dict() if self.in_domain else None,
            "bfcl": self.bfcl.as_dict() if self.bfcl else None,
            "arm_intervals": self.arm_intervals,
        }


def build_report(
    preconditions: list[Precondition],
    in_domain: AxisResult | None,
    bfcl: AxisResult | None,
) -> StudyReport:
    failed = [p for p in preconditions if not p.ok]
    if failed:
        return StudyReport(
            verdict="void",
            headline="VOID — " + "; ".join(p.detail for p in failed),
            preconditions=preconditions,
            in_domain=in_domain,
            bfcl=bfcl,
        )

    if in_domain is None:
        return StudyReport(
            verdict="void",
            headline="VOID — the in-domain arm produced no result; there is nothing to claim",
            preconditions=preconditions,
            in_domain=None,
            bfcl=bfcl,
        )

    if in_domain.verdict == "regression":
        verdict: OverallVerdict = "regression"
        headline = (
            f"REGRESSION — the evolved playbook lowered in-domain pass rate by "
            f"{-in_domain.delta.point:.3f} (95% CI "
            f"[{in_domain.delta.low:.3f}, {in_domain.delta.high:.3f}])"
        )
    elif in_domain.verdict == "no_effect":
        verdict = "not_supported"
        headline = (
            f"NOT SUPPORTED — in-domain delta {in_domain.delta.point:+.3f}, 95% CI "
            f"[{in_domain.delta.low:.3f}, {in_domain.delta.high:.3f}] includes zero "
            f"over {in_domain.n_units} tasks"
        )
    elif bfcl is None:
        verdict = "in_domain_only"
        headline = (
            f"IN-DOMAIN ONLY — pass rate up {in_domain.delta.point:+.3f} (95% CI "
            f"[{in_domain.delta.low:.3f}, {in_domain.delta.high:.3f}]), transfer unmeasured"
        )
    elif bfcl.verdict == "regression":
        verdict = "in_domain_only"
        headline = (
            f"IN-DOMAIN ONLY — pass rate up {in_domain.delta.point:+.3f} but BFCL down "
            f"{-bfcl.delta.point:.3f} (95% CI [{bfcl.delta.low:.3f}, {bfcl.delta.high:.3f}]); "
            f"the playbook is fitting the task generator, not teaching tool use"
        )
    else:
        verdict = "supported"
        headline = (
            f"SUPPORTED — in-domain {in_domain.delta.point:+.3f} (95% CI "
            f"[{in_domain.delta.low:.3f}, {in_domain.delta.high:.3f}]), BFCL "
            f"{bfcl.delta.point:+.3f} (95% CI [{bfcl.delta.low:.3f}, {bfcl.delta.high:.3f}], "
            f"{bfcl.verdict})"
        )

    return StudyReport(
        verdict=verdict,
        headline=headline,
        preconditions=preconditions,
        in_domain=in_domain,
        bfcl=bfcl,
    )


def format_report(report: StudyReport) -> str:
    """Plain-text summary for the terminal and the run log."""
    lines = ["", "=" * 68, report.headline, "=" * 68, "", "preconditions:"]
    for p in report.preconditions:
        lines.append(f"  [{'ok' if p.ok else 'FAIL'}] {p.name}: {p.detail}")
    for axis in (report.in_domain, report.bfcl):
        if axis is None:
            continue
        p_text = f", p={axis.p_value:.4f}" if axis.p_value is not None else ""
        lines += [
            "",
            f"{axis.name}: {axis.baseline:.3f} -> {axis.evolved:.3f} "
            f"({axis.delta.point:+.3f}, 95% CI [{axis.delta.low:.3f}, {axis.delta.high:.3f}], "
            f"n={axis.n_units}{p_text}) [{axis.verdict}]",
        ]
        for note in axis.notes:
            lines.append(f"    - {note}")
    lines.append("=" * 68)
    return "\n".join(lines)


def arm_interval(verdicts: list[bool | None]) -> Interval:
    """Wilson interval over one arm's raw rollouts, for context in the report."""
    scored = [v for v in verdicts if v is not None]
    return wilson_interval(sum(1 for v in scored if v), len(scored))


def _mean(verdicts: list[bool | None]) -> float | None:
    scored = [v for v in verdicts if v is not None]
    if not scored:
        return None
    return sum(1 for v in scored if v) / len(scored)
