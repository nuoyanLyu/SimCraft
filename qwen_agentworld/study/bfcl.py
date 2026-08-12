"""Read BFCL run artifacts and pair the two arms entry by entry.

BFCL writes, per model registry key and per category:

    <run_root>/result/<key>/BFCL_v4_<category>_result.json   one JSON object per
        generated entry, each carrying an `id`. This is the *universe* of
        entries the arm actually ran.
    <run_root>/score/<key>/BFCL_v4_<category>_score.json     first line is a
        summary object (`accuracy`, `correct_count`, `total_count`); every
        subsequent line is one *failed* entry, carrying its `id` and an error.

Both are JSON Lines despite the `.json` extension. The score file lists only
failures, so per-entry correctness is recovered as `ran - failed` rather than
read off directly — which is why the result file is required and not optional.

Why pair at all: the two arms are the same model at the same temperature on the
same entries, differing only in the appended playbook text. An unpaired
comparison of two accuracy numbers throws that away and needs a far larger
category before a real effect clears the noise. Paired, only the entries that
*changed* verdict carry signal, and there are usually few enough of them to
read by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class BfclArtifactError(RuntimeError):
    """A BFCL run directory that cannot be read as a study arm."""


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BfclArtifactError(f"{path}:{i} is not valid JSON: {exc}") from exc
    return records


def _locate(run_root: Path, kind: str, registry_key: str, category: str) -> Path:
    """Find the result/score file for one category.

    Globbed rather than formatted: the filename carries the BFCL dataset
    version (`BFCL_v3_` / `BFCL_v4_`), and hardcoding it would break silently
    on a harness upgrade — as a *missing file*, which the caller would then be
    tempted to treat as "no failures".
    """
    directory = run_root / kind / registry_key
    if not directory.is_dir():
        raise BfclArtifactError(
            f"{directory} does not exist. The arm did not run, or BFCL_RUN_ROOT / "
            f"REGISTRY_KEY differ from what the harness was given."
        )
    matches = sorted(p for p in directory.glob(f"*{category}*{kind}.json"))
    if not matches:
        raise BfclArtifactError(f"no {kind} file for category '{category}' under {directory}")
    if len(matches) > 1:
        # `simple` also matches `simple_python`, `live_simple`, ... — an
        # ambiguous category silently averaging several is worse than stopping.
        raise BfclArtifactError(
            f"category '{category}' matched several {kind} files under {directory}: "
            f"{[p.name for p in matches]}. Name the category exactly."
        )
    return matches[0]


@dataclass
class BfclArm:
    label: str
    category: str
    ran_ids: list[str]
    failed_ids: set[str] = field(default_factory=set)
    reported_accuracy: float | None = None
    reported_total: int | None = None

    @property
    def passed_ids(self) -> set[str]:
        return set(self.ran_ids) - self.failed_ids

    @property
    def accuracy(self) -> float:
        return len(self.passed_ids) / len(self.ran_ids) if self.ran_ids else 0.0

    def passed(self, entry_id: str) -> bool:
        return entry_id not in self.failed_ids


def load_arm(run_root: str | Path, *, label: str, registry_key: str, category: str) -> BfclArm:
    root = Path(run_root)
    result_path = _locate(root, "result", registry_key, category)
    score_path = _locate(root, "score", registry_key, category)

    ran_ids = [r["id"] for r in _read_jsonl(result_path) if "id" in r]
    if not ran_ids:
        raise BfclArtifactError(f"{result_path} lists no entry ids; the generate step produced nothing")

    score_records = _read_jsonl(score_path)
    if not score_records:
        raise BfclArtifactError(f"{score_path} is empty; the evaluate step did not run")
    summary = score_records[0]
    failed_ids = {r["id"] for r in score_records[1:] if "id" in r}

    arm = BfclArm(
        label=label,
        category=category,
        ran_ids=ran_ids,
        failed_ids=failed_ids,
        reported_accuracy=summary.get("accuracy"),
        reported_total=summary.get("total_count"),
    )

    # The recomputed accuracy must agree with the one BFCL printed. If it does
    # not, the file layout assumed above is wrong for this harness version and
    # every per-entry verdict derived from it is fiction — better to stop than
    # to report a confident gain built on a misread file.
    if arm.reported_accuracy is not None and abs(arm.accuracy - arm.reported_accuracy) > 1e-6:
        raise BfclArtifactError(
            f"{score_path}: recomputed accuracy {arm.accuracy:.4f} disagrees with the "
            f"reported {arm.reported_accuracy:.4f}. The score-file layout assumed by "
            f"this reader (line 0 = summary, rest = failures) does not hold here."
        )
    return arm


def paired_entries(base: BfclArm, evolved: BfclArm) -> list[tuple[str, bool, bool]]:
    """(entry_id, base_passed, evolved_passed) over entries both arms ran.

    Restricted to the intersection: a generate step that dropped an entry in
    one arm only (a timeout, a truncated run) would otherwise contribute a
    phantom win or loss to whichever arm still has it.
    """
    shared = [eid for eid in base.ran_ids if eid in set(evolved.ran_ids)]
    return [(eid, base.passed(eid), evolved.passed(eid)) for eid in shared]


def coverage_warning(base: BfclArm, evolved: BfclArm) -> str | None:
    """Describe an arm-size mismatch, or None when the arms line up.

    Not an exception: a handful of dropped entries is a caveat on the number,
    not grounds for discarding hours of benchmark run time. It belongs in the
    report where a reader will see it.
    """
    shared = len(paired_entries(base, evolved))
    if shared == len(base.ran_ids) == len(evolved.ran_ids):
        return None
    return (
        f"arms cover different entry sets: baseline ran {len(base.ran_ids)}, "
        f"evolved ran {len(evolved.ran_ids)}, {shared} in common — the comparison "
        f"uses the {shared} shared entries only"
    )
