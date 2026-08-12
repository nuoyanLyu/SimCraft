"""On-disk bank of generated tasks, so teacher calls are spent once.

Generating one task costs two teacher calls (NL + initial state, then checker
synthesis, the latter with up to 4 audit-retry rounds). At deepseek-v4-pro
latencies that is ~2 minutes and real money per task, and a 40-task eval set is
over an hour. Tasks are immutable once generated, so regenerating an equivalent
set for every experiment is pure waste.

Two things this fixes beyond caching:

* **Incremental save.** `ab_test.build_eval_tasks` used to write its output file
  only after the whole batch finished, so a crash at task 39 of 40 threw away
  every call already paid for. Tasks land on disk the moment they exist.
* **Train/eval provenance.** A task carries the split it was drawn for. The A/B
  is only meaningful if the eval set was never seen by the evolve run that
  produced the playbook, and once tasks are pooled and reused that separation
  stops being automatic — `draw` refuses to hand a train-split task to eval and
  vice versa.

Layout (bank root is gitignored):

    task_bank/<tool_family>/gc<graph_complexity>/<task_id>.json

Each file is `{"task": <Task>, "meta": {...}}`. There is no separate index: the
directory tree *is* the index, which means a partially-written bank is still a
valid bank and two processes appending concurrently cannot corrupt each other.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from qwen_agentworld.core.schemas import Task

logger = logging.getLogger(__name__)

DEFAULT_BANK_DIR = os.environ.get("TASK_BANK_DIR", "task_bank")

# Splits are enforced, not advisory: reusing a training task at eval time would
# silently turn the A/B into a memorisation measurement.
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_EVAL = "eval"
# Three pools, not two: `val` is what the orchestrator's rollback checks each
# playbook edit against. Drawing that from `eval` would mean the A/B reports a
# gain measured on the very tasks the edits were selected on.
_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_EVAL)


class TaskBank:
    def __init__(self, bank_dir: str | Path = DEFAULT_BANK_DIR) -> None:
        self.root = Path(bank_dir)

    # ---------------------------------------------------------------- paths --
    def _bucket(self, tool_family: str, gc: int) -> Path:
        return self.root / tool_family / f"gc{gc}"

    # ----------------------------------------------------------------- write --
    def save(
        self,
        task: Task,
        *,
        split: str,
        origin: str = "",
        teacher_model: str = "",
        gc: int | None = None,
    ) -> Path:
        """Persist one task immediately. Returns the file written."""
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        gc = gc if gc is not None else len(task.task_graph.nodes)
        bucket = self._bucket(task.tool_family, gc)
        bucket.mkdir(parents=True, exist_ok=True)
        path = bucket / f"{task.task_id}.json"
        payload = {
            "task": task.model_dump(mode="json"),
            "meta": {
                "split": split,
                "origin": origin,
                "teacher_model": teacher_model,
                "graph_complexity": gc,
                "created_at": time.time(),
                # Filled in by a screening pass (scripts that measure baseline
                # pass rate); None means "difficulty unknown".
                "baseline_pass_rate": None,
                # Which agent that rate was measured against
                # (`playbook_store.fingerprint`). A rate without one is from
                # before this field existed; `draw` treats it as matching
                # nothing in particular rather than guessing.
                "screened_by": None,
            },
        }
        # Write-then-rename: a killed process leaves either the old file or the
        # new one, never a half-written JSON that later reads as corrupt.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(path)
        return path

    def set_baseline_pass_rate(self, task_id: str, rate: float, *, screened_by: str = "") -> bool:
        """Record a screening result so difficulty-band selection can reuse it.

        `screened_by` names the agent the rate was measured against. It matters
        because the rate *is* the difficulty definition: an agent that has since
        improved makes every stored rate an understatement, and a band drawn on
        stale rates serves tasks the agent has already outgrown. Passing it
        lets `draw(screened_by=...)` tell a current measurement from an expired
        one instead of trusting all of them equally.
        """
        for path in self.root.rglob(f"{task_id}.json"):
            payload = json.loads(path.read_text())
            payload["meta"]["baseline_pass_rate"] = rate
            payload["meta"]["screened_by"] = screened_by or None
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(path)
            return True
        return False

    def set_audit_verdict(self, task_id: str, *, unpassable: bool, too_weak: bool) -> bool:
        """Record an `audit_task_quality` verdict so `draw` can act on it."""
        for path in self.root.rglob(f"{task_id}.json"):
            payload = json.loads(path.read_text())
            payload["meta"]["audit"] = {"unpassable": unpassable, "too_weak": too_weak}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(path)
            return True
        return False

    # ------------------------------------------------------------------ read --
    def _iter_bucket(self, tool_family: str, gc: int):
        bucket = self._bucket(tool_family, gc)
        if not bucket.is_dir():
            return
        for path in sorted(bucket.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
                yield path, payload["meta"], Task.model_validate(payload["task"])
            except Exception as exc:  # noqa: BLE001 — a bad file must not sink the batch
                logger.warning("skipping unreadable bank entry %s: %s", path, exc)

    def draw(
        self,
        tool_family: str,
        gc: int,
        n: int,
        *,
        split: str,
        band: tuple[float, float] | None = None,
        require_screened: bool = False,
        drop_audit_failed: bool = False,
        screened_by: str | None = None,
    ) -> list[Task]:
        """Up to `n` tasks from the bank matching the split (and difficulty band).

        `band` filters on the screened baseline pass rate. Tasks screened
        outside the band are excluded; unscreened tasks are excluded only when
        `require_screened` is set, so a band filter degrades to "everything
        known to be in band, plus anything not yet measured" by default.

        `drop_audit_failed` excludes tasks a quality audit flagged as either
        unpassable or too weak. Unaudited tasks are kept: the flag means "drop
        what is known bad", not "require an audit", so turning it on can never
        empty a bank that has not been audited yet.

        `screened_by` demands the rate have been measured against that specific
        agent; a rate from any other one is treated as no rate at all, which is
        what it is. Left as None, any measurement counts — right for a run
        pinned to the baseline agent, wrong for one whose agent keeps changing.
        """
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        out: list[Task] = []
        for _, meta, task in self._iter_bucket(tool_family, gc):
            if meta.get("split") != split:
                continue
            audit = meta.get("audit") or {}
            if drop_audit_failed and (audit.get("unpassable") or audit.get("too_weak")):
                continue
            rate = meta.get("baseline_pass_rate")
            if screened_by is not None and meta.get("screened_by") != screened_by:
                rate = None  # measured against a different agent; not evidence about this one
            if rate is None:
                if require_screened:
                    continue
            elif band is not None and not (band[0] <= rate <= band[1]):
                continue
            out.append(task)
            if len(out) >= n:
                break
        return out

    def stats(self) -> dict:
        counts: dict[str, dict] = {}
        for path in self.root.rglob("*.json"):
            try:
                meta = json.loads(path.read_text())["meta"]
            except Exception:  # noqa: BLE001
                continue
            key = f"{path.parent.parent.name}/{path.parent.name}"
            b = counts.setdefault(key, {"train": 0, "eval": 0, "screened": 0})
            b[meta.get("split", "train")] = b.get(meta.get("split", "train"), 0) + 1
            if meta.get("baseline_pass_rate") is not None:
                b["screened"] += 1
        return counts

    # ----------------------------------------------------------------- prune --
    def prune(
        self,
        *,
        max_age_days: float | None = None,
        max_per_bucket: int | None = None,
        drop_unscreened: bool = False,
        dry_run: bool = False,
    ) -> list[Path]:
        """Delete bank entries, newest-first retention within each bucket.

        Left unbounded a bank grows without limit and slows every `draw` while
        holding tasks generated by a teacher or prompt version no longer in use.
        """
        removed: list[Path] = []
        now = time.time()
        buckets: dict[Path, list[tuple[float, Path, dict]]] = {}
        for path in self.root.rglob("*.json"):
            try:
                meta = json.loads(path.read_text())["meta"]
            except Exception:  # noqa: BLE001 — unreadable entries are exactly what pruning is for
                removed.append(path)
                if not dry_run:
                    path.unlink(missing_ok=True)
                continue
            buckets.setdefault(path.parent, []).append((meta.get("created_at", 0.0), path, meta))

        for _, entries in buckets.items():
            entries.sort(key=lambda e: e[0], reverse=True)  # newest first
            kept = 0
            for created_at, path, meta in entries:
                doomed = False
                if max_age_days is not None and (now - created_at) > max_age_days * 86400:
                    doomed = True
                if drop_unscreened and meta.get("baseline_pass_rate") is None:
                    doomed = True
                if not doomed and max_per_bucket is not None and kept >= max_per_bucket:
                    doomed = True
                if doomed:
                    removed.append(path)
                    if not dry_run:
                        path.unlink(missing_ok=True)
                else:
                    kept += 1
        return removed


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Inspect or prune the task bank.")
    p.add_argument("command", choices=["stats", "prune"])
    p.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    p.add_argument("--max-age-days", type=float, default=None)
    p.add_argument("--max-per-bucket", type=int, default=None)
    p.add_argument("--drop-unscreened", action="store_true")
    p.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    a = p.parse_args()

    bank = TaskBank(a.bank_dir)
    if a.command == "stats":
        for bucket, counts in sorted(bank.stats().items()):
            print(f"{bucket:24} {counts}")
        return

    if a.max_age_days is None and a.max_per_bucket is None and not a.drop_unscreened:
        p.error("prune needs at least one of --max-age-days / --max-per-bucket / --drop-unscreened")
    removed = bank.prune(
        max_age_days=a.max_age_days,
        max_per_bucket=a.max_per_bucket,
        drop_unscreened=a.drop_unscreened,
        dry_run=not a.apply,
    )
    verb = "removed" if a.apply else "would remove"
    print(f"{verb} {len(removed)} entries")
    for path in removed[:20]:
        print("  ", path)
    if len(removed) > 20:
        print(f"   ... and {len(removed) - 20} more")


if __name__ == "__main__":
    _main()
