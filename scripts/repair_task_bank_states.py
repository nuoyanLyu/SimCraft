"""Bring banked tasks' `initial_state` back into conformance with the declared
state schema, deterministically and in place.

Why this exists
---------------
`audit_task_schema.py` found 55/240 banked tasks (22.9%) whose `initial_state`
does not conform to `tools/state_schema.py` — overwhelmingly a missing list
field (`tags`) that the state was simply never given. Generation-time
enforcement exists now, so a freshly generated bank is clean; the tasks banked
before it do not repair themselves.

A missing field is not cosmetic. The step-wise checker is evaluated against the
ordered state list whose *first* element is `task.initial_state` verbatim — it
never passes through `complete_fields` — so a predicate that reads the absent
key raises KeyError at states[0] and the task scores not-passed no matter what
the agent did.

What it does and does not touch
-------------------------------
Repairs are exactly `schema.conform_state`: add declared fields at their
neutral value, never remove, never overwrite an existing value. Anything
`conform_state` cannot fix on its own — notably the 7 tasks whose *checker*
reads a key the schema never declares — is reported and left alone, because
inventing a field to satisfy a wrong predicate would hide a bad checker rather
than fix it.

Repair invalidates screening: `baseline_pass_rate` was measured against the old
state, so it is cleared along with `screened_by` and the repaired task drops
out of any band draw until re-screened. That is the cost of running this, and
it is the reason for --apply rather than doing it silently.

Usage:
    python scripts/repair_task_bank_states.py                 # dry run, reports only
    python scripts/repair_task_bank_states.py --apply         # writes, after backing up
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_agentworld.tools.state_schema import get_schema


def iter_bank(bank_dir: Path):
    for path in sorted(bank_dir.rglob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable, skipped: {path}")
            continue
        if "task" in payload:
            yield path, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default="task_bank")
    ap.add_argument("--apply", action="store_true", help="write the repairs (default: dry run)")
    a = ap.parse_args()

    bank_dir = Path(a.bank_dir)
    if not bank_dir.exists():
        print(f"no such bank dir: {bank_dir}")
        return 1

    to_repair: list[tuple[Path, dict, dict, list[str]]] = []
    total = no_schema = 0

    for path, payload in iter_bank(bank_dir):
        total += 1
        task = payload["task"]
        schema = get_schema(task.get("tool_family", ""))
        if schema is None:
            no_schema += 1
            continue
        before = task.get("initial_state") or {}
        after = schema.conform_state(before)
        if after == before:
            continue
        added = sorted(set(_flat_keys(after)) - set(_flat_keys(before)))
        to_repair.append((path, payload, after, added))

    print(f"scanned {total} banked tasks in {bank_dir} ({no_schema} with no declared schema)")
    print(f"  initial_state needs conforming : {len(to_repair)}")

    field_counts: dict[str, int] = {}
    for _, _, _, added in to_repair:
        for key in added:
            field_counts[key] = field_counts.get(key, 0) + 1
    for key, n in sorted(field_counts.items(), key=lambda kv: -kv[1]):
        print(f"    +{key}: {n} tasks")

    screened = sum(
        1 for _, p, _, _ in to_repair if (p.get("meta") or {}).get("baseline_pass_rate") is not None
    )
    print(f"  of those, already screened     : {screened}  (screening will be cleared)")

    if not a.apply:
        print("\ndry run — nothing written. Re-run with --apply to repair.")
        return 0

    backup = bank_dir.parent / f"{bank_dir.name}_pre_state_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(bank_dir, backup)
    print(f"\nbacked up {bank_dir} -> {backup}")

    for path, payload, after, _ in to_repair:
        payload["task"]["initial_state"] = after
        meta = payload.setdefault("meta", {})
        meta["baseline_pass_rate"] = None
        meta["screened_by"] = None
        meta["state_repaired_at"] = time.time()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(path)

    print(f"repaired {len(to_repair)} tasks; {screened} lost their screening and need re-screening")
    return 0


def _flat_keys(state: dict, prefix: str = "") -> list[str]:
    """Dotted key paths one level into lists of objects, so a repair that adds
    a field to every note reads as `notes[].tags` rather than just `notes`.
    """
    keys: list[str] = []
    for key, value in state.items():
        path = f"{prefix}{key}"
        keys.append(path)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            for item in value:
                keys.extend(_flat_keys(item, f"{path}[]."))
        elif isinstance(value, dict):
            keys.extend(_flat_keys(value, f"{path}."))
    return keys


if __name__ == "__main__":
    raise SystemExit(main())
