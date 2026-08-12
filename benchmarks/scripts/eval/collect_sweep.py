#!/usr/bin/env python3
"""Collect the parallel sweep results into a comparison table across checkpoints.

Reads, per step, whatever has completed so far (robust to in-progress runs):
  BFCL  : sweep/<step>/bfcl/score/data_overall.csv  (Non-Live AST, Live, Multi Turn)
  tau2  : sweep/<step>/tau2/<domain>_sim.json        (avg reward per domain)
  vita  : vitabench/data/simulations/sweep_<step>_<domain>.json (avg reward per domain)

Usage (factory env):  python collect_sweep.py
"""
import glob
import json
import os

SWEEP = "/data1/lvnuoyan/eval_runs/sweep"
VITA_SIMS = "/data1/lvnuoyan/dataset/agent/vitabench/data/simulations"
STEPS = ["base", "step20", "step40", "step60", "step80", "step100", "step110"]
TAU2_DOMAINS = ["mock", "airline", "retail", "telecom", "banking_knowledge"]
VITA_DOMAINS = ["delivery", "instore", "ota", "cross_domain"]


def _num(x):
    try:
        return float(str(x).replace("%", ""))
    except Exception:
        return None


def bfcl_scores(step):
    f = f"{SWEEP}/{step}/bfcl/score/data_overall.csv"
    if not os.path.exists(f):
        return {}
    try:
        import pandas as pd
        r = pd.read_csv(f).iloc[0]
        nl, lv, mt = _num(r.get("Non-Live AST Acc")), _num(r.get("Live Acc")), _num(r.get("Multi Turn Acc"))
        st = round((nl + lv) / 2, 2) if nl is not None and lv is not None else None
        return {"single_turn": st, "non_live": nl, "live": lv, "multi_turn": mt}
    except Exception as e:
        return {"error": str(e)[:40]}


def _avg_reward(sim_file):
    """Average reward across simulations in a tau2/vita sim json."""
    if not os.path.exists(sim_file):
        return None
    try:
        d = json.load(open(sim_file))
        sims = d.get("simulations", d if isinstance(d, list) else [])
        rewards = []
        for s in sims:
            ri = s.get("reward_info") or {}
            r = ri.get("reward", s.get("reward"))
            if r is not None:
                rewards.append(float(r))
        return round(sum(rewards) / len(rewards) * 100, 2) if rewards else None
    except Exception:
        return None


def tau2_scores(step):
    out = {}
    for d in TAU2_DOMAINS:
        out[d] = _avg_reward(f"{SWEEP}/{step}/tau2/{d}_sim.json/results.json")
    return out


def vita_scores(step):
    out = {}
    for d in VITA_DOMAINS:
        matches = glob.glob(f"{VITA_SIMS}/sweep_{step}_{d}*.json")
        f = matches[0] if matches else None
        if f and os.path.isdir(f):
            f = os.path.join(f, "results.json")
        out[d] = _avg_reward(f) if f else None
    return out


def fmt(v):
    return f"{v:6.2f}" if isinstance(v, (int, float)) else "   -  "


def main():
    print("\n=== BFCL (paper Base 85.15/33.50; EnvFactory-4B 85.46/48.50) ===")
    print(f"{'step':8s} {'SingleTurn':>11s} {'NonLive':>8s} {'Live':>7s} {'MultiTurn':>10s}")
    for s in STEPS:
        b = bfcl_scores(s)
        print(f"{s:8s} {fmt(b.get('single_turn')):>11s} {fmt(b.get('non_live')):>8s} "
              f"{fmt(b.get('live')):>7s} {fmt(b.get('multi_turn')):>10s}")

    print("\n=== tau2-Bench avg reward % (per domain) ===")
    print(f"{'step':8s} " + " ".join(f"{d[:6]:>7s}" for d in TAU2_DOMAINS))
    for s in STEPS:
        t = tau2_scores(s)
        print(f"{s:8s} " + " ".join(fmt(t.get(d)) + " " for d in TAU2_DOMAINS))

    print("\n=== VitaBench avg reward % (per domain) ===")
    print(f"{'step':8s} " + " ".join(f"{d[:8]:>9s}" for d in VITA_DOMAINS))
    for s in STEPS:
        v = vita_scores(s)
        print(f"{s:8s} " + " ".join(f"{fmt(v.get(d)):>9s}" for d in VITA_DOMAINS))
    print()


if __name__ == "__main__":
    main()
