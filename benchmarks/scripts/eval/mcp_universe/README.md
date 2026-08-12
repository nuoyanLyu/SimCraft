# MCP-Universe evaluation

Agent = EnvFactory model (vLLM, hermes tool parser), driven via MCP-Universe's
`function_call` agent over real MCP servers. Runs in the `factory` conda env.
The model is wired by pointing MCP-Universe's `openai` LLM type at our endpoint:
`OPENAI_BASE_URL` -> vLLM, `OPENAI_API_KEY=EMPTY`, `model_name: envfactory-eval`.

## Run
```bash
conda activate factory
bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
bash scripts/eval/mcp_universe/run_mcp_universe.sh              # default: financial_smoke.yaml (1 yfinance task)
bash scripts/eval/mcp_universe/run_mcp_universe.sh mcpuniverse/financial_analysis.yaml   # full domain
```
Config files: `mcpuniverse/benchmark/configs/mcpuniverse/*.yaml`. The smoke config
`financial_smoke.yaml` (1 task, yfinance + calculator servers) needs NO external API keys.

## Domains & keys
Most domains need external services in `MCP-Universe/.env`:
web_search (SerpAPI), location_navigation (Google Maps), repository_management (GitHub),
multi_server (Notion, ...). AWM excludes 3d_design (Blender) and repository_management.
Only `financial_analysis` (yfinance + calculator) runs key-free.

## Install notes (factory env)
mcpuniverse pkg present; the runner import chain pulls provider SDKs -- installed
`xai_sdk claude_code_sdk playwright yfinance mcp_server_calculator` under the numpy/torch/vllm
constraints file. Did NOT `pip install -r requirements.txt` (it hard-pins pydantic==2.11.7,
which would downgrade factory's 2.13.4 and risk vllm).

## Smoke result (EnvFactory-4B, financial_smoke, 1 task)
Model I/O is correct and reasonable: framework loaded yfinance+calculator MCP servers,
listed tools, and the model selected `get_historical_stock_prices` with correct arguments
(ticker MSFT, right dates, interval 1d) and reasoned in <think>, handling tool errors
gracefully. The task did not pass eval only because the yfinance server itself failed to
fetch data: TLS error (curl_cffi `OPENSSL_internal:invalid library`) + Yahoo rate-limiting
-- an external data-source issue, independent of the model / framework / wiring.
Model-side I/O integration is confirmed. (yfinance TLS/rate-limit is a follow-up if full
financial_analysis numbers are needed.)
