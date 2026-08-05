#!/usr/bin/env bash
# Run MCP-Atlas EnvFactory 291-task subset for one model via its harness.
# Agent loop only (no dmxapi); scoring is a separate later step.
# Usage: run_atlas_model.sh <STEP> <HARNESS_PORT>
set -uo pipefail
STEP="${1:?}"; HPORT="${2:?}"
TASKS=/data1/lvnuoyan/eval_runs/mcp_atlas_tasks.csv
OUT=/data1/lvnuoyan/eval_runs/sweep/$STEP/mcp_atlas
mkdir -p "$OUT"
source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory
cd /data1/lvnuoyan/dataset/agent/mcp-atlas
echo "[$(date +%H:%M)] [$STEP] MCP-Atlas run_eval start (harness :$HPORT, 291 tasks)"
HARNESS_URL="http://localhost:$HPORT" python run_eval.py \
    --model envfactory-eval \
    --input "$TASKS" \
    --output "$OUT/outputs.csv" \
    --concurrency 2 \
    --timeout 1800
echo "[$(date +%H:%M)] [$STEP] MCP-Atlas ATLAS_DONE"
