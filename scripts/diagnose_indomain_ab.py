"""Triage a null in-domain A/B: where did the measurement power go?

`select_eval_set.py` composes the eval set from three groups that answer
different questions -- band (sensitive), hard (headroom), ceiling (harm
detector) -- and its own docstring warns that a single mean over a set this
heterogeneous hides whichever effect is smaller. The pooled number in
`report.json` is the verdict; this script is what turns a `not_supported` into
a next step.

The number it exists to print is the count of tasks that sat at 0.00 or 1.00 in
*both* arms. Those tasks cost the same GPU time as any other and contribute
nothing but a tie to the paired statistic, so the effective sample size is the
remainder -- which can be a small fraction of what the report's `n` claims.

Usage:
    python scripts/diagnose_indomain_ab.py studies/run_0806
"""

import argparse
import json
from pathlib import Path


def rate(verdicts) -> float | None:
    judged = [v for v in verdicts if v is not None]
    return sum(bool(v) for v in judged) / len(judged) if judged else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a study directory written by verify_playbook_effect.py")
    args = ap.parse_args()

    run = Path(args.run_dir)
    results = json.loads((run / "indomain" / "results.json").read_text())
    group_of = json.loads((run / "eval_set_rationale.json").read_text())["group_of"]
    arms = results["checkpoints"]

    final_label = next((l for l in arms if "final" in l), None)
    if final_label is None:
        raise SystemExit(f"no final arm in {sorted(arms)}")

    per_task: dict[str, dict[str, float | None]] = {}
    for label, payload in arms.items():
        for tid, verdicts in payload.get("per_task", {}).items():
            per_task.setdefault(tid, {})[label] = rate(verdicts)

    print(f"arms: {list(arms)}   comparing baseline -> {final_label}\n")
    print(f"{'group':8} {'n':>3} {'baseline':>9} {'final':>9} {'delta':>8}  wins/losses/ties")

    buckets: dict[str, list[tuple[float, float]]] = {}
    for tid, per_arm in per_task.items():
        b, f = per_arm.get("baseline"), per_arm.get(final_label)
        if b is not None and f is not None:
            buckets.setdefault(group_of.get(tid, "?"), []).append((b, f))

    for group in ("band", "hard", "ceiling", "?"):
        pairs = buckets.get(group)
        if not pairs:
            continue
        base = sum(p[0] for p in pairs) / len(pairs)
        final = sum(p[1] for p in pairs) / len(pairs)
        wins = sum(1 for x, y in pairs if y > x)
        losses = sum(1 for x, y in pairs if y < x)
        print(f"{group:8} {len(pairs):3} {base:9.3f} {final:9.3f} {final - base:+8.3f}  "
              f"{wins}/{losses}/{len(pairs) - wins - losses}")

    print("\ntasks that moved:")
    moved = 0
    for tid, per_arm in sorted(per_task.items()):
        b, f = per_arm.get("baseline"), per_arm.get(final_label)
        if b is None or f is None or b == f:
            continue
        moved += 1
        print(f"  {tid}  {group_of.get(tid, '?'):8} {b:.2f} -> {f:.2f}  ({f - b:+.2f})")

    floor = [t for t, p in per_task.items() if p.get("baseline") == 0.0 and p.get(final_label) == 0.0]
    ceil = [t for t, p in per_task.items() if p.get("baseline") == 1.0 and p.get(final_label) == 1.0]
    n = len(per_task)
    print(f"\ndead weight: {len(floor)} tasks at 0.00 in both arms, {len(ceil)} at 1.00 in both")
    print(f"effective sample: {n - len(floor) - len(ceil)} of {n} tasks could have shown anything; "
          f"{moved} actually moved")


if __name__ == "__main__":
    main()
