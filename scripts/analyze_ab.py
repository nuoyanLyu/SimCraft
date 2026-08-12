"""Analyse the A/B by the groups the eval set was built from.

A single pooled mean over band + hard + ceiling hides the thing the ceiling
group was put there to catch. The three groups answer different questions:
gains should appear in band and hard, and the ceiling group is the harm check --
if it drops, the playbook is buying its gains by changing behaviour that was
already correct.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

RUN = Path("abtest/run_0729")
res = json.loads((RUN / "results.json").read_text())
rat = json.loads((RUN / "eval_set_rationale.json").read_text())
group_of = rat["group_of"]
base_rate = rat["baseline_rate_of"]

arms = list(res["checkpoints"])
per_task = {a: res["checkpoints"][a]["per_task"] for a in arms}


def rate(results):
    """Pass rate over the rollouts that actually produced a verdict.

    A None is a rollout that errored out, not a failure: counting it as one
    would charge an arm for infrastructure noise. They are reported separately.
    """
    scored = [r for r in results if r is not None]
    return (sum(scored) / len(scored)) if scored else None


print("errored rollouts:",
      {a: sum(r is None for v in per_task[a].values() for r in v) for a in arms})
A, B = arms[0], arms[1]

print(f"arms: {A} (v{res['checkpoints'][A]['playbook_version']}) vs "
      f"{B} (v{res['checkpoints'][B]['playbook_version']})\n")

# ---- per-group means --------------------------------------------------------
by_group = defaultdict(lambda: {A: [], B: []})
for tid, g in group_of.items():
    if tid not in per_task[A]:
        continue
    if rate(per_task[A][tid]) is None or rate(per_task[B][tid]) is None:
        continue  # nothing to pair against
    for arm in (A, B):
        by_group[g][arm].append(rate(per_task[arm][tid]))

print(f"{'group':9} {'n':>3}  {A:>18}  {B:>18}   delta")
for g in ("band", "hard", "ceiling"):
    a = by_group[g][A]
    b = by_group[g][B]
    if not a:
        continue
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    print(f"{g:9} {len(a):>3}  {ma:>18.3f}  {mb:>18.3f}   {mb - ma:+.3f}")
n_pooled = sum(len(v[A]) for v in by_group.values())
ma = sum(sum(v[A]) for v in by_group.values()) / n_pooled
mb = sum(sum(v[B]) for v in by_group.values()) / n_pooled
print(f"{'POOLED':9} {n_pooled:>3}  {ma:>18.3f}  {mb:>18.3f}   {mb - ma:+.3f}\n")

# ---- task-level movement ----------------------------------------------------
up = down = same = 0
movers = []
for tid in per_task[A]:
    ra, rb = rate(per_task[A][tid]), rate(per_task[B][tid])
    if ra is None or rb is None:
        continue
    if rb > ra:
        up += 1
    elif rb < ra:
        down += 1
    else:
        same += 1
    if rb != ra:
        movers.append((rb - ra, tid, group_of.get(tid), base_rate.get(tid), ra, rb))
print(f"tasks improved: {up}   worsened: {down}   unchanged: {same}")

# Sign test on the tasks that moved: under the null that the playbook does
# nothing, each mover is an independent coin flip.
n = up + down
if n:
    p = sum(math.comb(n, k) for k in range(up, n + 1)) / 2 ** n
    print(f"sign test on the {n} movers: one-sided p = {p:.4f}")

print("\nlargest movements:")
for d, tid, g, screened, ra, rb in sorted(movers, reverse=True):
    print(f"  {d:+.2f}  {tid}  {g:<8} screened={screened}  {ra:.2f} -> {rb:.2f}")

# ---- rep-level separation ---------------------------------------------------
ra_reps = res["checkpoints"][A]["rep_pass_rates"]
rb_reps = res["checkpoints"][B]["rep_pass_rates"]
print(f"\nrep ranges: {A} [{min(ra_reps):.3f}, {max(ra_reps):.3f}]  "
      f"{B} [{min(rb_reps):.3f}, {max(rb_reps):.3f}]")
print("disjoint" if max(ra_reps) < min(rb_reps) else "overlapping")
