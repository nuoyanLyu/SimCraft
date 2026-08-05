# tau2-Bench evaluation

Agent = EnvFactory model (vLLM, hermes tool parser). User simulator + LLM judge =
dmxapi DeepSeek-V3.2 (EnvFactory paper setup). Runs in the `factory` conda env.
tau2-eval was installed with `pip install -e . --no-deps` + text-only deps under a
constraints file (voice deps elevenlabs/livekit/pyaudio skipped; numpy/torch/vllm protected).

## Run
```bash
conda activate factory
bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
NUM_TASKS=1 bash scripts/eval/tau2/run_tau2.sh mock      # smoke (mock domain, 1 task)
bash scripts/eval/tau2/run_tau2.sh airline              # full domain (paper: tau2 avg)
```
Domains: mock | airline | retail | telecom | banking_knowledge.
Output: `/data1/lvnuoyan/eval_runs/tau2/<domain>_sim.json`.

## Wiring
- `--agent-llm openai/envfactory-eval --agent-llm-args {temperature:0.7, api_base:<vllm>, api_key:EMPTY}`
- `--user-llm openai/DeepSeek-V3.2 --user-llm-args {temperature:0.0, api_base:https://www.dmxapi.cn/v1, api_key:<dmx from .env>}`
- Agent temperature 0.7 from common/envfactory_config.py (paper, thinking models).

## Smoke result (EnvFactory-4B, mock, 1 task)
Average Reward 1.0, Write 1/1, DB Match OK, judge 0 errors -> agent+user-sim+judge chain works.

## Note
tau2's agent system prompt is the per-domain policy; EnvFactory's generic training
system prompt (EF_PROMPT) is not injected here (would require modifying the agent). The
domain policy is the intended tau2 agent prompt.
