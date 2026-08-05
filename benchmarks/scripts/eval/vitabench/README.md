# VitaBench evaluation

Agent = EnvFactory model (vLLM, hermes tool parser). User simulator + evaluator =
dmxapi DeepSeek-V3.2 (EnvFactory paper setup). Runs in the `factory` conda env.
Installed with `pip install -e . --no-deps` + text deps under a constraints file
(numpy/torch/vllm protected). VitaBench reuses much of tau2-bench's codebase.

## Run
```bash
conda activate factory
bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
NUM_TASKS=1 bash scripts/eval/vitabench/run_vita.sh delivery     # smoke
bash scripts/eval/vitabench/run_vita.sh cross_domain            # full (paper: VitaBench avg)
```
Domains: delivery | instore | ota | cross_domain. Language: english (default here) | chinese.
Simulations saved under `vitabench/data/simulations/`.

## Wiring
run_vita.sh generates a `models.yaml` at runtime (VITA_MODEL_CONFIG_PATH) with per-model
`base_url` (VitaBench POSTs directly to base_url, so it must be the FULL
`.../v1/chat/completions` URL):
- agent `envfactory-eval` -> vLLM endpoint, temperature 0.7 (from envfactory_config.py)
- user + evaluator `DeepSeek-V3.2` -> https://www.dmxapi.cn/v1/chat/completions (dmx key from .env)
- `--enable-think` (Qwen3 thinking model).

## Smoke result (EnvFactory-4B, delivery, 1 task)
1/1 simulation completed; RewardType.NL_ASSERTION 0.80 (final reward 0 on this task) ->
agent + user-sim + evaluator chain works, model I/O reasonable.
