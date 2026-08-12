#!/usr/bin/env python3
"""Collect the Static-model sweep into the EnvFactory main-table columns:

  BFCL Single | BFCL Multi | MCP-Atlas Pass | MCP-Atlas Cov |
  tau2-airline | tau2-retail | tau2-telecom | tau2-avg

Robust to in-progress runs (missing pieces show as '-'). Reads per step from
  BFCL : static_sweep/<step>/bfcl/score/data_overall.csv
  tau2 : static_sweep/<step>/tau2/<domain>_sim.json/results.json
  atlas: static_sweep/<step>/mcp_atlas/score/scored_<step>.csv  (coverage_score col)
"""
import glob, json, os

SWEEP = "/data1/lvnuoyan/eval_runs/static_sweep"
STEPS = ["step20", "step40", "step60", "step80", "step100", "step110"]
TAU2 = ["airline", "retail", "telecom"]


def _num(x):
    try:
        return float(str(x).replace("%", ""))
    except Exception:
        return None


def bfcl(step):
    f = f"{SWEEP}/{step}/bfcl/score/data_overall.csv"
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


def tau2(step):
    out = {}
    for d in TAU2:
        base = f"{SWEEP}/{step}/tau2/{d}_sim.json"
        f = base + "/results.json" if os.path.isdir(base) else base
        out[d] = _avg_reward(f)
    return out


def atlas(step):
    """Return (pass_rate_0.75, mean_coverage*100). Prefer the scored CSV."""
    scored = f"{SWEEP}/{step}/mcp_atlas/score/scored_{step}.csv"
    if os.path.exists(scored):
        try:
            import pandas as pd
            s = pd.read_csv(scored)["coverage_score"].dropna().to_numpy()
            if len(s):
                cov = round(float(s.mean()) * 100, 2)
                pas = round(float((s >= 0.75).sum() / len(s) * 100), 2)
                return pas, cov
        except Exception:
            pass
    j = f"{SWEEP}/{step}/mcp_atlas/score/coverage_stats_{step}_all.json"
    if os.path.exists(j):
        try:
            d = json.load(open(j))
            mc, pr = d.get("mean_coverage"), d.get("pass_rate_0.75")
            return pr, (round(mc * 100, 2) if mc is not None else None)
        except Exception:
            pass
    return None, None


def f(v):
    return f"{v:6.2f}" if isinstance(v, (int, float)) else "   -  "


def main():
    print("# EnvFactory Static-repair sweep — main table")
    print(f"model dir: {SWEEP.replace('eval_runs/static_sweep','llm_model/.../Static-...-20260714-1800')}\n")
    hdr = ["step", "BFCL-Single", "BFCL-Multi", "Atlas-Pass", "Atlas-Cov",
           "tau2-air", "tau2-retail", "tau2-tele", "tau2-avg"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for st in STEPS:
        single, multi = bfcl(st)
        pas, cov = atlas(st)
        t = tau2(st)
        vals = [t[d] for d in TAU2]
        present = [v for v in vals if isinstance(v, (int, float))]
        avg = round(sum(present) / len(present), 2) if present else None
        row = [st, single, multi, pas, cov, t["airline"], t["retail"], t["telecom"], avg]
        print("| " + " | ".join(x if isinstance(x, str) else f(x).strip() for x in row) + " |")
    print()


if __name__ == "__main__":
    main()
