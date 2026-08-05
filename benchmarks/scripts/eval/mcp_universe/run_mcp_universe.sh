#!/usr/bin/env bash
# Run an MCP-Universe benchmark against the EnvFactory model. Runs in `factory` env.
# The model under test is wired via OPENAI_BASE_URL -> local vLLM (hermes tool parser),
# so the function_call agent's OpenAI tools= calls hit our endpoint.
#
# Prereq: serve the model:
#   bash scripts/eval/common/serve_model.sh <MODEL_DIR> 0 8100 envfactory-eval 0.35
#
# Usage:
#   bash scripts/eval/mcp_universe/run_mcp_universe.sh [CONFIG]
#   default CONFIG = mcpuniverse/financial_smoke.yaml (1 yfinance task, no external keys)
#
# Domains needing external API keys (SerpAPI/Maps/GitHub/Notion) require those in
# MCP-Universe/.env. financial_analysis (yfinance+calculator) needs none.
set -euo pipefail
MU_DIR="/data1/lvnuoyan/dataset/agent/MCP-Universe"
AGENT_ENDPOINT="${AGENT_ENDPOINT:-http://localhost:8100/v1}"
SERVED_NAME="${SERVED_NAME:-envfactory-eval}"
CONFIG="${1:-mcpuniverse/financial_smoke.yaml}"

export OPENAI_BASE_URL="$AGENT_ENDPOINT"
export OPENAI_API_KEY="EMPTY"
export PYTHONPATH="$MU_DIR"
export MCPUniverse_DIR="$MU_DIR"

# pre-flight: agent endpoint reachable
if ! curl -sf "$AGENT_ENDPOINT/models" | grep -q "$SERVED_NAME"; then
    echo "[ERROR] agent endpoint $AGENT_ENDPOINT not serving \"$SERVED_NAME\". Start serve_model.sh first." >&2
    exit 1
fi

echo "[run_mcp_universe] config=$CONFIG agent=$SERVED_NAME@$AGENT_ENDPOINT"
cd "$MU_DIR"
mkdir -p log/mcpuniverse
python - "$CONFIG" << 'PY'
import asyncio, sys, traceback
from mcpuniverse.tracer.collectors import FileCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks

async def main():
    tc = FileCollector(log_file="log/mcpuniverse/smoke.log")
    bench = BenchmarkRunner(sys.argv[1])
    results = await bench.run(trace_collector=tc, callbacks=get_vprint_callbacks())
    print("=" * 60)
    for r in results:
        for task_name, tr in r.task_results.items():
            evals = tr.get("evaluation_results", [])
            passed = sum(1 for e in evals if getattr(e, "passed", False))
            print(f"TASK {task_name}: {passed}/{len(evals)} checks passed")
            for e in evals:
                print("   check passed=%s func=%s" % (getattr(e, "passed", "?"),
                                                      getattr(getattr(e, "config", None), "func", "?")))

asyncio.run(main())
PY
echo "[run_mcp_universe] done."
