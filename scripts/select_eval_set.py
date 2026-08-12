"""Compose a frozen A/B eval set from tasks that can actually show an effect.

The 2026-07-29 version of this file kept three groups -- band, hard (rate 0.0,
audit-clean) and a ceiling sample -- on the argument that the screened pool was
too bimodal for a pure band draw to yield anything. The 2026-08-06 A/B is the
measurement that settles it. Of its 23 tasks, 11 scored 1.00 in both arms and 7
scored 0.00 in both; five tasks could move and four did. Pooling 23 tasks with
an effective sample of five produced +0.052 with a 95% CI of [-0.026, 0.139] --
a null that says nothing about the playbook, only about the instrument.

So the band is now the whole set. A task is admitted iff its *measured* pass
rate is strictly between 0 and 1: difficulty is defined relative to this agent,
and only a task the agent sometimes passes has room to move in either
direction. That last clause is why dropping the ceiling group costs nothing --
harm shows up as a band task falling, which this set measures directly, whereas
a task pinned at 1.00 contributes a guaranteed zero to the mean and nothing to
the variance budget.

The pool is small, so two knobs matter:

* `--splits` may name more than the eval split. A train-split task the evolve
  run never drew is held out in fact, which is what held-out means; pass
  `--exclude-seen-in <evolve_dir>` and every task id that run touched is
  removed. Selection is recorded per task in the rationale, so the claim is
  auditable rather than asserted.
* `--min-tasks` refuses to write an eval set too small to answer anything. The
  2026-08-06 run would have failed this check instead of costing two hours of
  GPU to discover its own n.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/root/SimCraft")

from qwen_agentworld.teacher.task_bank import SPLIT_EVAL, TaskBank

ap = argparse.ArgumentParser()
ap.add_argument("--tool-family", default="mcp_notes")
ap.add_argument("--graph-complexity", type=int, default=3)
ap.add_argument("--splits", default=SPLIT_EVAL,
                help="comma separated; a non-eval split is only honest with --exclude-seen-in")
ap.add_argument("--exclude-seen-in", default=None,
                help="evolve dir; every task id in its iteration_*.json is dropped")
ap.add_argument("--band-low", type=float, default=0.0)
ap.add_argument("--band-high", type=float, default=1.0)
ap.add_argument("--min-tasks", type=int, default=20,
                help="refuse to write an eval set smaller than this")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="abtest/run/eval_tasks.json")
ap.add_argument("--rationale", default="abtest/run/eval_set_rationale.json")
args = ap.parse_args()

splits = [s.strip() for s in args.splits.split(",") if s.strip()]

seen: set[str] = set()
if args.exclude_seen_in:
    for path in sorted(Path(args.exclude_seen_in).glob("iteration_*.json")):
        record = json.loads(path.read_text())
        for entry in record.get("tasks", []):
            task_id = (entry.get("task") or {}).get("task_id")
            if task_id:
                seen.add(task_id)

bank = TaskBank()
selected: list[tuple] = []
rejected: list[dict] = []

for _, meta, task in bank._iter_bucket(args.tool_family, args.graph_complexity):
    split = meta.get("split")
    if split not in splits:
        continue
    if task.task_id in seen:
        rejected.append({"task_id": task.task_id, "why": "seen_during_evolve"})
        continue
    rate = meta.get("baseline_pass_rate")
    if rate is None:
        rejected.append({"task_id": task.task_id, "why": "unscreened"})
        continue
    audit = meta.get("audit") or {}
    if audit.get("unpassable") or audit.get("too_weak"):
        why = "unpassable" if audit.get("unpassable") else "too_weak"
        rejected.append({"task_id": task.task_id, "why": f"audit:{why}", "rate": rate})
        continue
    # Strictly between the pins, then inside the requested band. The first test
    # is the one that matters and is not expressible as a band: with 5 reps a
    # band of [0.0, 1.0] still has to exclude the 0.0 and 1.0 endpoints.
    if not 0.0 < rate < 1.0:
        rejected.append({"task_id": task.task_id,
                         "why": "pinned_floor" if rate == 0.0 else "pinned_ceiling",
                         "rate": rate})
        continue
    if not args.band_low <= rate <= args.band_high:
        rejected.append({"task_id": task.task_id, "why": "out_of_band", "rate": rate})
        continue
    selected.append((task, rate, split))

rng = random.Random(args.seed)
rng.shuffle(selected)

why_counts = {w: sum(1 for r in rejected if r["why"] == w) for w in {r["why"] for r in rejected}}
print(f"selected {len(selected)} band tasks from splits={splits}")
print(f"rejected {len(rejected)}: " + json.dumps(why_counts))

if len(selected) < args.min_tasks:
    raise SystemExit(
        f"refusing to write: {len(selected)} band tasks < --min-tasks {args.min_tasks}. "
        f"Screen more candidates (screen_task_difficulty.py) or widen --splits; "
        f"an A/B this small cannot separate a real effect from noise."
    )

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([t.model_dump(mode="json") for t, _, _ in selected],
                          ensure_ascii=False, indent=2))

rationale = {
    "mode": "band_only",
    "splits": splits,
    "band": [args.band_low, args.band_high],
    "excluded_seen_in": args.exclude_seen_in,
    "n_seen_during_evolve": len(seen),
    "n_selected": len(selected),
    "n_rejected": len(rejected),
    "group_of": {t.task_id: "band" for t, _, _ in selected},
    "split_of": {t.task_id: s for t, _, s in selected},
    "baseline_rate_of": {t.task_id: r for t, r, _ in selected},
    "rejected_counts": why_counts,
    "rejected": rejected,
    "seed": args.seed,
}
Path(args.rationale).parent.mkdir(parents=True, exist_ok=True)
Path(args.rationale).write_text(json.dumps(rationale, ensure_ascii=False, indent=2))
print(f"wrote {out} and {args.rationale}")
