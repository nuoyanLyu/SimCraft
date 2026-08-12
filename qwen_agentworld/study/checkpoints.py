"""Loading playbook checkpoints out of an evolve run's artifacts.

One loader, used by every consumer (`scripts/ab_test.py`, the BFCL prompt
renderer, the study driver). The alternative — each consumer calling
`Playbook.model_validate` on whatever JSON it found — is how a stale checkpoint
becomes a silent empty playbook: pydantic drops unknown fields, so a run
recorded before the entry redesign validates cleanly into a playbook with zero
entries, and every arm downstream then measures the baseline against itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from qwen_agentworld.core.schemas import Playbook
from qwen_agentworld.playbook_store.store import fingerprint


class StalePlaybookFormatError(ValueError):
    """A checkpoint written before the entry redesign (top-level `modules`)."""


def playbook_from_dict(raw: dict, *, source: str = "<dict>") -> Playbook:
    """Validate a playbook dict, refusing the pre-entry format loudly.

    An iteration record is accepted directly (its `playbook_after` is used), so
    callers can hand over whichever of the two shapes they happen to hold.
    """
    body = raw.get("playbook_after", raw)
    if "modules" in body and not body.get("entries"):
        raise StalePlaybookFormatError(
            f"{source} is in the pre-entry playbook format (`modules`). Re-run the "
            "evolution loop to produce checkpoints in the current format; loading it "
            "as-is would silently yield an empty playbook and every arm would then be "
            "measuring the baseline against itself."
        )
    return Playbook.model_validate(body)


def load_playbook(path: str | Path) -> Playbook:
    p = Path(path)
    return playbook_from_dict(json.loads(p.read_text()), source=str(p))


def empty_playbook() -> Playbook:
    """The baseline arm. Not "a playbook we chose not to evolve" — the A1 arm
    starts from an empty table by design, so every entry an arm carries was
    produced by execution."""
    return Playbook(version=1)


_ITERATION_RE = re.compile(r"iteration_(\d+)\.json$")


def iteration_files(run_dir: str | Path) -> list[Path]:
    """Iteration records in numeric order.

    Sorted by the parsed integer, not lexically: `iteration_10.json` sorts
    before `iteration_2.json` as a string, which would silently label the wrong
    checkpoint "final".
    """
    files = []
    for p in Path(run_dir).glob("iteration_*.json"):
        m = _ITERATION_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return [p for _, p in sorted(files)]


@dataclass(frozen=True)
class Checkpoint:
    label: str
    playbook: Playbook
    source: str

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.playbook)


def collect_checkpoints(run_dir: str | Path, *, mid: bool = True) -> list[Checkpoint]:
    """The arms of the study: empty baseline, optional midpoint, final.

    The midpoint is not a third condition to compare against; it is there to
    show whether the gain accumulated over iterations or arrived all at once,
    which is the difference between "the loop learns" and "one lucky edit".
    """
    files = iteration_files(run_dir)
    if not files:
        raise FileNotFoundError(f"no iteration_*.json under {run_dir}; the evolve stage produced nothing")

    checkpoints = [Checkpoint("baseline", empty_playbook(), "<empty>")]
    if mid and len(files) >= 3:
        mid_file = files[len(files) // 2 - 1]
        checkpoints.append(Checkpoint("mid", load_playbook(mid_file), str(mid_file)))
    checkpoints.append(Checkpoint("final", load_playbook(files[-1]), str(files[-1])))
    return checkpoints


def dedupe_checkpoints(checkpoints: list[Checkpoint]) -> list[Checkpoint]:
    """Collapse arms whose playbook text is identical.

    Compares content rather than version: a rolled-back edit bumps the version
    without changing a word the agent reads. Running both spends GPU on a
    duplicate and invites reading the gap between them as a trend.
    """
    kept: list[Checkpoint] = []
    for ckpt in checkpoints:
        for i, existing in enumerate(kept):
            if existing.fingerprint == ckpt.fingerprint:
                kept[i] = Checkpoint(f"{existing.label}+{ckpt.label}", existing.playbook, existing.source)
                break
        else:
            kept.append(ckpt)
    return kept
