"""One scoring entry point, so "how a rollout is judged" is a setting.

Five places score a rollout — `orchestrator/loop.py`, `orchestrator/validation.py`,
`scripts/ab_test.py`, `scripts/screen_task_difficulty.py`, the smoke scripts —
and each had its own copy of "build the states list, call judge_checker". That
was fine while there was one judge. With two, it would mean five places to keep
in agreement, and an A/B whose arms were screened under different judges is not
an A/B at all: the eval set's `baseline_pass_rate` and the run's pass rate have
to be the same measurement or the difficulty band means nothing.

Three modes:

  checker  the executable predicate alone. Unchanged behaviour, still the
           default, and still the only free and deterministic option.
  llm      the LLM judge's score, thresholded. Use when the predicate is the
           thing being worked around (see judge/llm_judge.py for the
           measurement that motivated it).
  both     compute both; the *checker* remains the verdict and the LLM score
           rides along in the record. This is the mode to run first — it is
           what turns "do I trust the judge" from an opinion into an agreement
           rate, at the cost of one teacher call per rollout and no change to
           any number the pipeline already produces.

`both` deliberately does not let the LLM override the checker. A mode that
quietly changed the verdict would make the comparison it exists to enable
impossible to read afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qwen_agentworld.config import get_env
from qwen_agentworld.core.schemas import Task, Trajectory
from qwen_agentworld.judge.llm_judge import DEFAULT_JUDGE_THRESHOLD, judge_with_llm
from qwen_agentworld.judge.paired_audit import judge_checker_with_reason
from qwen_agentworld.llm_clients.base import LLMClient

logger = logging.getLogger(__name__)

MODE_CHECKER = "checker"
MODE_LLM = "llm"
MODE_BOTH = "both"
JUDGE_MODES = (MODE_CHECKER, MODE_LLM, MODE_BOTH)


@dataclass
class JudgeConfig:
    mode: str = MODE_CHECKER
    threshold: float = DEFAULT_JUDGE_THRESHOLD
    client: LLMClient | None = None

    def __post_init__(self) -> None:
        if self.mode not in JUDGE_MODES:
            raise ValueError(f"unknown judge mode {self.mode!r}; expected one of {JUDGE_MODES}")
        if self.mode in (MODE_LLM, MODE_BOTH) and self.client is None:
            raise ValueError(f"judge mode {self.mode!r} needs a `client`; none was supplied")

    @property
    def uses_llm(self) -> bool:
        return self.mode in (MODE_LLM, MODE_BOTH)


@dataclass
class Verdict:
    """The bit the pipeline runs on, plus everything needed to re-litigate it."""

    passed: bool
    reason: str
    source: str  # which judge decided `passed`
    checker_passed: bool | None = None
    checker_reason: str | None = None
    llm_score: float | None = None
    llm_reason: str | None = None
    llm_error: str | None = None

    def record(self) -> dict:
        """Flat dict for the per-rollout JSONL lines."""
        return {
            "verdict": self.passed,
            "verdict_source": self.source,
            "verdict_reason": self.reason,
            "checker_passed": self.checker_passed,
            "checker_reason": self.checker_reason,
            "llm_score": self.llm_score,
            "llm_reason": self.llm_reason,
            "llm_error": self.llm_error,
        }


def states_for(task: Task, trajectory: Trajectory | None) -> list[dict]:
    """Initial state plus one canonical state per executed step.

    A step-wise checker needs these to verify a reversible task whose final
    state equals its initial one; an end-state checker ignores them. Four call
    sites had this three-line list inlined, and one of them drifting is a
    silent change of verdict, not a crash.
    """
    steps = trajectory.steps if trajectory is not None else []
    return [task.initial_state] + [
        (step.simulator_raw_output or {}).get("next_state", {}) for step in steps
    ]


def judge_rollout(
    task: Task,
    final_state: dict,
    trajectory: Trajectory | None = None,
    config: JudgeConfig | None = None,
) -> Verdict:
    config = config or JudgeConfig()

    checker_passed = checker_reason = None
    if config.mode in (MODE_CHECKER, MODE_BOTH):
        checker_passed, checker_reason = judge_checker_with_reason(
            task.checker, final_state, states=states_for(task, trajectory), task_id=task.task_id
        )

    llm = None
    if config.uses_llm:
        llm = judge_with_llm(
            config.client, task, final_state, trajectory=trajectory, threshold=config.threshold
        )

    if config.mode == MODE_LLM:
        return Verdict(
            passed=llm.passed,
            reason=llm.error or ("llm_pass" if llm.passed else "llm_fail"),
            source=MODE_LLM,
            llm_score=llm.score,
            llm_reason=llm.reason,
            llm_error=llm.error,
        )

    return Verdict(
        passed=bool(checker_passed),
        reason=checker_reason or "",
        source=MODE_CHECKER,
        checker_passed=checker_passed,
        checker_reason=checker_reason,
        llm_score=llm.score if llm else None,
        llm_reason=llm.reason if llm else None,
        llm_error=llm.error if llm else None,
    )


def judge_config_from_env(client_factory=None) -> JudgeConfig:
    """Read `JUDGE_MODE` / `JUDGE_THRESHOLD` from the environment.

    `client_factory` is called only when the mode actually needs a judge, so
    importing this in a checker-only run never constructs a teacher client.
    """
    mode = (get_env("JUDGE_MODE", MODE_CHECKER) or MODE_CHECKER).strip().lower()
    threshold = float(get_env("JUDGE_THRESHOLD", str(DEFAULT_JUDGE_THRESHOLD)))
    client = None
    if mode in (MODE_LLM, MODE_BOTH):
        if client_factory is None:
            from qwen_agentworld.llm_clients.teacher_claude import TeacherClient

            client_factory = TeacherClient
        client = client_factory()
    return JudgeConfig(mode=mode, threshold=threshold, client=client)
