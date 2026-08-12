#!/usr/bin/env python3
"""Collect the repeat sweep into the EnvFactory main-table columns, reporting
mean +/- sample std across the 3 seeds for every metric:

  BFCL Single | BFCL Multi | MCP-Atlas Pass | MCP-Atlas Cov |
  tau2-airline | tau2-retail | tau2-telecom | tau2-avg

Per (model, seed) it reads:
  BFCL : repeat_sweep/<model>/seed<k>/bfcl/score/data_overall.csv
  tau2 : repeat_sweep/<model>/seed<k>/tau2/<domain>_sim.json[/results.json]
  atlas: repeat_sweep/<model>/seed<k>/mcp_atlas/score/scored_<model>_seed<k>.csv
"""
import glob
import json
import os
import statistics as st

SWEEP = "/data1/lvnuoyan/eval_runs/repeat_sweep"
MODELS = ["base", "rl-step100", "rl-step110", "static-step100", "static-step110"]
SEEDS = [0, 1, 2]
TAU2 = ["airline", "retail", "telecom"]
METRICS = ["BFCL-Single", "BFCL-Multi", "Atlas-Pass", "Atlas-Cov",
           "tau2-air", "tau2-retail", "tau2-tele", "tau2-avg"]


def _num(x):
    try:
        return float(str(x).replace("%", ""))
    except Exception:
        return None


def bfcl(d):
    f = f"{d}/bfcl/score/data_overall.csv"
    if not os.path.exists(f):
        return None, None
    try:
        import pandas as pd
        r = pd.read_csv(f).iloc[0]
        nl, lv, mt = _num(r.get("Non-Live AST Acc")), _num(r.get("Live Acc")), _num(r.get("Multi Turn Acc"))
        single = round((nl + lv) / 2, 2) if nl is not None and lv is not None else None
        return single, mt
    except Exception:
        return None, None


def _avg_reward(sim_file):
    if not sim_file or not os.path.exists(sim_file):
        return None
    try:
        d = json.load(open(sim_file))
        sims = d.get("simulations", d if isinstance(d, list) else [])
        rs = []
        for s in sims:
            ri = s.get("reward_info") or {}
            r = ri.get("reward", s.get("reward"))
            if r is not None:
                rs.append(float(r))
        return round(sum(rs) / len(rs) * 100, 2) if rs else None
    except Exception:
        return None


def tau2(d):
    out = {}
    for dom in TAU2:
        base = f"{d}/tau2/{dom}_sim.json"
        f = base + "/results.json" if os.path.isdir(base) else base
        out[dom] = _avg_reward(f)
    return out


def atlas(d, model, seed):
    scored = f"{d}/mcp_atlas/score/scored_{model}_seed{seed}.csv"
    cands = [scored] + glob.glob(f"{d}/mcp_atlas/score/scored_*.csv")
    for f in cands:
        if not os.path.exists(f):
            continue
        try:
            import pandas as pd
            s = pd.read_csv(f)["coverage_score"].dropna().to_numpy()
            if len(s):
                cov = round(float(s.mean()) * 100, 2)
                pas = round(float((s >= 0.75).sum() / len(s) * 100), 2)
                return pas, cov
        except Exception:
            continue
    return None, None


def seed_metrics(model, seed):
    d = f"{SWEEP}/{model}/seed{seed}"
    single, multi = bfcl(d)
    pas, cov = atlas(d, model, seed)
    t = tau2(d)
    tv = [t[x] for x in TAU2]
    # Only report the average once ALL three domains are in. A partial average
    # (e.g. air+retail while telecom is still running) is not comparable against
    # a complete one, since telecom scores far lower and drags the mean down.
    tavg = round(sum(tv) / len(tv), 2) if all(isinstance(v, (int, float)) for v in tv) else None
    return {"BFCL-Single": single, "BFCL-Multi": multi, "Atlas-Pass": pas, "Atlas-Cov": cov,
            "tau2-air": t["airline"], "tau2-retail": t["retail"], "tau2-tele": t["telecom"], "tau2-avg": tavg}


def fmt(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return "    -   "
    m = sum(v) / len(v)
    sd = st.stdev(v) if len(v) >= 2 else 0.0
    return f"{m:.2f}±{sd:.2f}" + ("" if len(v) == len(SEEDS) else f"(n={len(v)})")


def main():
    print("# EnvFactory repeat sweep — mean ± sample-std over 3 seeds "
          "(temp 0.7, EF system prompt, identical eval config for all models)\n")
    hdr = ["model"] + METRICS
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    rows = {}
    for m in MODELS:
        per = {k: [] for k in METRICS}
        for s in SEEDS:
            sm = seed_metrics(m, s)
            for k in METRICS:
                per[k].append(sm[k])
        rows[m] = per
        print("| " + m + " | " + " | ".join(fmt(per[k]) for k in METRICS) + " |")
    print()

    # Also dump the raw per-seed numbers for transparency / debugging.
    print("## per-seed raw\n")
    print("| model | seed | " + " | ".join(METRICS) + " |")
    print("|" + "|".join(["---"] * (len(METRICS) + 2)) + "|")
    for m in MODELS:
        for i, s in enumerate(SEEDS):
            cells = []
            for k in METRICS:
                v = rows[m][k][i]
                cells.append(f"{v:.2f}" if isinstance(v, (int, float)) else "-")
            print(f"| {m} | {s} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    main()
