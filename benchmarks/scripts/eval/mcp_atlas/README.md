# MCP-Atlas evaluation

Evaluate the EnvFactory model on MCP-Atlas. Unlike BFCL, MCP-Atlas is a multi-service
stack that exercises the model through the **real MCP `list_tools`/`call_tool` interface**
against production MCP servers running in a Docker sandbox.

Everything runs in the `factory` conda env. Harness source: `/data1/lvnuoyan/dataset/agent/mcp-atlas`.

## Architecture
```
run_eval.py  --HTTP-->  TS harness (:3001)  --HTTP-->  Docker sandbox (:1984, 20-36 MCP servers)
                              |
                              +--chat/completions (tools=)--> vLLM (:8100, hermes tool parser)
score_claims.py  --claim judge-->  dmxapi DeepSeek-V3.2
```
- Model under test: vLLM OpenAI endpoint, served **with `--enable-auto-tool-choice
  --tool-call-parser hermes`** so native `tools=` function-calling works (serve_model.sh
  already sets this). Model name passed to the harness is the bare served name
  `envfactory-eval` (the harness does NOT strip a LiteLLM `openai/` prefix).
- Judge: dmxapi DeepSeek-V3.2 (EnvFactory paper setting), configured via `EVAL_LLM_*`.

## One-time services (start once, in separate tmux)
```bash
conda activate factory

# 0) config: writes mcp-atlas/.env (model endpoint + dmxapi judge, key from EnvFactory/.env)
bash scripts/eval/mcp_atlas/setup_env.sh

# 1) serve the model (GPU 0, gentle mem util) -- hermes tool parser enabled
bash scripts/eval/common/serve_model.sh \
    /data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf/step20 \
    0 8100 envfactory-eval 0.35

# 2) MCP sandbox (docker, port 1984) -- image already pulled + tagged agent-environment:latest
cd /data1/lvnuoyan/dataset/agent/mcp-atlas && make run-docker   # wait for "Uvicorn running on :1984"

# 3) TS agent harness (port 3001)
cd /data1/lvnuoyan/dataset/agent/mcp-atlas && make run-harness
```
Verify: `curl -s localhost:1984/enabled-servers | jq` (20 default no-key servers OK).

## Smoke run (proves correctness, then stop)
```bash
# 1 task, restricted to the 20 no-key servers, with judge scoring
SMOKE_N ignored; use NUM_TASKS arg:
bash scripts/eval/mcp_atlas/run_mcp_atlas.sh 1
```
`run_mcp_atlas.sh N` = prepare_tasks (parquet->CSV, only-default-servers) -> run_eval N tasks
-> score_claims with dmxapi judge. Outputs under `/data1/lvnuoyan/eval_runs/mcp_atlas/`.

Full run later: use `ONLY_DEFAULT=0` and a larger N (needs API keys in .env for key-gated
servers; ~18% of tasks run with the 20 default servers only).

## Fixes applied (dmxapi / local-endpoint compatibility)
1. **`LLM_BASE_URL` must NOT include `/v1`** — the TS harness builds
   `${LLM_BASE_URL}/v1/chat/completions` itself (litellm-strategy.ts). setup_env.sh sets
   `http://localhost:8100`.
2. **Model name is passed verbatim** to vLLM — use `envfactory-eval`, not `openai/...`.
3. **Judge JSON parse made lenient** — dmxapi DeepSeek-V3.2 ignores
   `response_format=json_schema`, so `score_claims.py` now extracts JSON from fenced /
   prose-wrapped judge replies (`_loads_lenient`, patched in the harness). `EVAL_LLM_BASE_URL`
   is `https://www.dmxapi.cn` (no `/v1`; scorer appends `/v1/chat/completions`),
   `EVAL_LLM_MODEL=DeepSeek-V3.2` (no prefix).

## Smoke result (step20)
Full chain verified: the model drives real MCP tools through the sandbox (e.g. reads
`/data/Barber Shop.csv` via `filesystem_read_text_file` -> answers "Customer"), run_eval
writes trajectories, and the dmxapi judge scores claim coverage end-to-end.

## EnvFactory paper subset (Appendix F) — reproducing their MCP-Atlas number

The paper evaluates a subset: **30 of 36 servers, 291 of 500 tasks**, excluding
`mongodb, oxylabs, brave-search, wikipedia, slack, google-workspace`
("due to network connectivity constraints"). This is now a toggle:

- **Servers (30)**: start the sandbox with the paper's server set —
  ```bash
  MCP_SERVERS_MODE=envfactory bash scripts/eval/mcp_atlas/setup_env.sh
  cd /data1/lvnuoyan/dataset/agent/mcp-atlas && make run-docker   # sandbox now enables the 30
  ```
  (Key-gated servers among the 30 — github, notion, google-maps, airtable, alchemy,
  exa, twelvedata, lara-translate, e2b — still need their keys in mcp-atlas/.env to start.)

- **Tasks (291)**: `EF_SUBSET=1` selects the 291 tasks whose *gold trajectory* does not
  use any excluded server (matches the paper exactly; verified = 291). Note the filter is
  on the gold TRAJECTORY, not ENABLED_TOOLS — the latter contains distractor tools.
  ```bash
  EF_SUBSET=1 bash scripts/eval/mcp_atlas/run_mcp_atlas.sh 291
  ```

`prepare_tasks.py` also exposes the general form:
`--envfactory-subset` (the 6 above) or `--exclude-servers a,b,c` (arbitrary set).
Compare the resulting pass-rate / mean-coverage to the release target (9.97 / 21.89).
