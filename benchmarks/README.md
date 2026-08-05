# Agentic benchmark harness (ported from 79)

Source: `79:~/EnvFactory/scripts/eval` + `79:/data1/lvnuoyan/dataset/agent`.
Copied 2026-07-28. The harness scripts are kept as close to the 79 originals as
possible so they stay diffable; every change is listed under "Deltas" below.

**The scripts here are tracked; every `*.env` under this directory is not.**
`benchmarks/**/*.env` is gitignored wholesale, so the files that hold live
third-party API keys — `scripts/eval/mcp_atlas/keys.env` above all — can never
reach a commit. Their `*.env.template` siblings are tracked instead: copy a
template, fill it in locally, and it stays local. Add new secrets only to a
`*.env`, never to a script or a README. If a key does leak into a commit,
rotate it — rewriting history is not enough once it has been pushed.

The benchmark repos and every run output live outside the tree
(`/root/autodl-tmp/dataset/agent`, `/root/autodl-tmp/eval_runs`), so tracking
this directory costs ~270KB of shell and Python, no data.

## What works here

| Benchmark | Status | Smoke result (Qwen3-8B) |
|-----------|--------|--------------------------|
| BFCL v4 (`simple_python`) | ✅ verified | 2/2 = 100% |
| tau2-bench (`mock`) | ✅ verified | reward 1.000, DB match 1/1 |
| MCP-Atlas / VitaBench / MCP-Universe | ⛔ not ported | data not copied; need API keys |

## Layout

```
benchmarks/
  env.autodl.sh        # machine-local overrides -- source this first
  playbook_prompt.py   # renders a playbook into the injected system prompt
  scripts/eval/        # verbatim-ish copy of 79's harness
```
Benchmark repos live outside the repo tree at `/root/autodl-tmp/dataset/agent/`
(BFCL 181M, tau2 778M); results at `/root/autodl-tmp/eval_runs/`.

## Quick start

Both vLLM servers must already be up (`scripts/serve_simulator.sh` :8000,
`scripts/serve_agent.sh` :8001). The benchmarks talk to the **Agent** (:8001).

```bash
cd /root/SimCraft
source benchmarks/env.autodl.sh

# BFCL: baseline arm, then the evolved-playbook arm
SMOKE_N=2 bash benchmarks/scripts/eval/bfcl/run_bfcl.sh simple_python
PLAYBOOK=smoke_test_results/<run>/iteration_4.json \
    BFCL_RUN_ROOT=/root/autodl-tmp/eval_runs/bfcl_playbook \
    bash benchmarks/scripts/eval/bfcl/run_bfcl.sh simple_python

# tau2
NUM_TASKS=1 bash benchmarks/scripts/eval/tau2/run_tau2.sh mock
```

Give each arm its own `BFCL_RUN_ROOT` / `RUN_DIR` — the harnesses key results by
model name, so two arms sharing a root overwrite each other.

## The A/B knob

`PLAYBOOK=<iteration_N.json>` appends the evolved playbook to the base system
prompt; unset is the baseline arm. It is *appended*, never substituted, so the
two arms are byte-identical apart from the playbook text.

`playbook_prompt.py` deliberately calls `simulator_gym.env._build_playbook_context`
— the same renderer the training loop uses. If eval formatted the playbook even
slightly differently, any measured gain would be confounded by the prompt
difference rather than attributable to the playbook.

## Deltas from the 79 setup

1. **Separate venv** (`/root/autodl-tmp/envs/bench`) instead of 79's single
   `factory` conda env. BFCL hard-pins `numpy==1.26.4` and friends, and here the
   equivalent env (`simcraft`) is the one actively running both vLLM servers —
   a bad resolve would take the servers down mid-experiment. The harnesses only
   speak HTTP to the endpoint, so they need no torch/vllm and lose nothing by
   being isolated. Verified after install: simcraft still on numpy 2.3.5 /
   torch 2.11.0+cu130 / vllm 0.25.1.
2. **Paths** `/data1/lvnuoyan/*` → `/root/autodl-tmp/*`, `/home/lvnuoyan/EnvFactory`
   → `/root/SimCraft`. `TAU2_DIR` and `EF_ROOT` were hardcoded on 79; they are
   `${VAR:-default}` now.
3. **Model under test** is the already-served `Qwen3-8B` on :8001, not an
   EnvFactory checkpoint on :8100 — so the benchmarks hit the exact model the
   playbook was evolved against.
4. **`register_models.py` honors `REGISTRY_KEY`/`SERVED_NAME`.** On 79 both were
   hardcoded to the 4B checkpoint, so `run_bfcl.sh`'s knobs were silently ignored
   and `bfcl generate` died with `Unknown model_name`.
5. **dmxapi key lookup** is `${DMX_KEY_VAR:-EMBEDDING_API_KEY}`; this repo's
   `.env` names it `DMX_API_KEY`. The script errors out loudly if it resolves
   empty (79 would have proceeded with an empty key).
6. **Extra deps** BFCL needed here: `soundfile` (+ `libsndfile1`), `tree_sitter`,
   `tree_sitter_java`, `tree_sitter_javascript`.

## Not verified

- BFCL categories beyond `simple_python`; `multi_turn` especially (the paper's
  headline) has not been run.
- tau2 domains beyond `mock` — `airline`/`retail` are the ones that matter and
  each costs real dmxapi tokens for the user simulator.
- Whether the playbook arm actually *helps*. The mechanism is proven; the
  measurement needs a finished bigrun playbook.
