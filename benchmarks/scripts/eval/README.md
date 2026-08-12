# Benchmark Evaluation Pipeline

Evaluate EnvFactory-trained checkpoints on tool-use benchmarks. The model is served
once via `vllm serve` (OpenAI-compatible endpoint); each benchmark harness calls that
endpoint over the OpenAI interface. Benchmarks needing an external user-simulator /
LLM-as-judge use DeepSeek-V3.2 via dmxapi (see `common/judge.env`), per the EnvFactory paper.

Everything runs in the single **`factory`** conda env -- no extra env needed.

## Layout
```
scripts/eval/
  common/
    serve_model.sh   # vllm serve wrapper (GPU-gentle defaults)
    judge.env        # external judge = dmxapi DeepSeek-V3.2 (key from EnvFactory/.env)
                     #   gitignored; the tracked copy is judge.env.template
  bfcl/              # Berkeley Function Calling Leaderboard  [DONE]
    register_models.py   # inject EnvFactory model into BFCL (QwenFCHandler, idempotent)
    run_bfcl.sh          # register -> generate -> evaluate  (SMOKE_N=N for a tiny run)
    tests/test_bfcl_pipeline.py
  mcp_atlas/         # MCP-Atlas             [DONE]
  tau2/              # tau2-Bench            [DONE]
  vitabench/         # VitaBench             [DONE]
  mcp_universe/      # MCP-Universe          [DONE]
```
Benchmark harness sources live in `/data1/lvnuoyan/dataset/agent/` (see its README).

## Env note (why factory works, no conflict)
BFCL hard-pins `numpy==1.26.4` / `networkx==3.3` / `filelock==3.20.0`, which would
downgrade factorys numpy 2.2.6 / torch 2.8 / vllm 0.11. We install BFCL with
`--no-deps` + a constraints file (`/tmp/bfcl_constraints.txt`) that locks those
versions, so only additive deps (tree_sitter, qwen-agent, faiss-cpu, sentence-transformers,
soundfile, ...) get added. Verified: numpy/torch/vllm unchanged after install.

## Quick start (BFCL)
```bash
conda activate factory

# 1) serve the checkpoint, GPU 0, low mem util
bash scripts/eval/common/serve_model.sh \
    /data1/lvnuoyan/llm_model/factory/EnvFactory-RL-Qwen3-4B-no_kl-grpo-1e-6-0.7-20260703-1745/hf/step20 \
    0 8100 envfactory-eval 0.35

# 2) tiny smoke (prove correctness), then real categories when ready
SMOKE_N=3 bash scripts/eval/bfcl/run_bfcl.sh simple_python
bash scripts/eval/bfcl/run_bfcl.sh single_turn     # full (later)
bash scripts/eval/bfcl/run_bfcl.sh multi_turn       # full (later)
```
Results/scores: `/data1/lvnuoyan/eval_runs/bfcl/{result,score}`.

## How the model plugs in
`register_models.py` adds a `MODEL_CONFIG_MAPPING` entry using BFCLs `QwenFCHandler`,
whose hard-coded chat template matches EnvFactorys training format exactly
(tools in `<tools>`, output `<tool_call>{"name","arguments"}</tool_call>`, `<tool_response>`, `<think>`).
`model_name` must equal vLLM `--served-model-name` (`envfactory-eval`); the tokenizer is
loaded locally via `REMOTE_OPENAI_TOKENIZER_PATH`. Temperature defaults to 0.7 (thinking).

Note: BFCL on `main` ships v4 data; EnvFactory reports v3. Categories used by the paper
(single_turn / multi_turn) exist; for strict v3 parity checkout an older gorilla tag.
