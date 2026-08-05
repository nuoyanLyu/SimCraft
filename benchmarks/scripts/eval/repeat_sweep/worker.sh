#!/usr/bin/env bash
# One card-worker. Brings up its own vLLM (per model/seed), its own MCP-Atlas
# sandbox+harness stack (fixed ports for this card), then pulls models from the
# shared queue and runs the full benchmark suite for each of the 3 seeds.
#
# Usage: worker.sh <CARD_IDX>     (0 = card A / GPU3, 1 = card B / GPU5)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
conda_on

CARD="${1:?need CARD idx 0|1}"
GPU="${CARD_GPU[$CARD]}"
PORT="${CARD_VLLM_PORT[$CARD]}"
SBX="${CARD_SANDBOX_PORT[$CARD]}"
HRN="${CARD_HARNESS_PORT[$CARD]}"
SBXNAME="${CARD_SANDBOX_NAME[$CARD]}"
ENDPOINT="http://localhost:$PORT/v1"
TAG="card$CARD/gpu$GPU"

mkdir -p "$LOGDIR" "$QUEUE"
WLOG="$LOGDIR/worker_card${CARD}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$TAG] $*" | tee -a "$WLOG"; }

TEMPERATURE="$(python "$EF/scripts/eval/common/envfactory_config.py" --print temperature)"
SYS_PROMPT="$(python "$EF/scripts/eval/common/envfactory_config.py" --print system_prompt)"

# ---------------------------------------------------------------- atlas stack
start_sandbox(){
  if curl -sf "http://localhost:$SBX/enabled-servers" >/dev/null 2>&1; then
    log "sandbox already up on :$SBX"; return 0; fi
  docker rm -f "$SBXNAME" >/dev/null 2>&1 || true
  ( cd "$ATLAS_DIR" && docker run -d --rm --name "$SBXNAME" \
      -p "$SBX:1984" --env-file .env agent-environment:latest ) \
      >"$LOGDIR/atlas_sandbox_card${CARD}.log" 2>&1
  # first-time warmup connects all 28 MCP servers (git clones/uv installs) before
  # uvicorn binds the port; this can take ~8-10 min. Wait generously.
  for i in $(seq 1 120); do
    curl -sf "http://localhost:$SBX/enabled-servers" >/dev/null 2>&1 && { log "sandbox UP :$SBX (after $((i*10))s)"; return 0; }
    sleep 10
  done
  log "ERROR sandbox failed to start on :$SBX"; return 1
}

start_harness(){
  tmux kill-session -t "rsweep_harness_$CARD" 2>/dev/null || true
  sleep 1
  tmux new-session -d -s "rsweep_harness_$CARD" \
    "source ~/anaconda3/etc/profile.d/conda.sh; conda activate factory; cd $ATLAS_DIR/services/agent-harness; \
     PORT=$HRN LLM_BASE_URL=http://localhost:$PORT LLM_API_KEY=EMPTY MCP_SANDBOX_URL=http://localhost:$SBX \
     LLM_TIMEOUT_MS=600000 LIST_TOOLS_TIMEOUT_MS=180000 TOOL_CALL_TIMEOUT_MS=120000 \
     npm run dev 2>&1 | tee $LOGDIR/atlas_harness_card${CARD}.log"
  for i in $(seq 1 60); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$HRN/" 2>/dev/null || echo 000)"
    [ "$code" != "000" ] && { log "harness UP :$HRN"; return 0; }
    sleep 5
  done
  log "ERROR harness not ready on :$HRN"; return 1
}

# ---------------------------------------------------------------- vLLM serve
SERVE_PID=""
serve_model(){ # <model_path> <seed>
  local mpath="$1" seed="$2"
  kill_serve
  log "serve $(basename "$mpath") seed=$seed on GPU$GPU:$PORT"
  EXTRA_VLLM_ARGS="--seed $seed" \
    bash "$EF/scripts/eval/common/serve_model.sh" "$mpath" "$GPU" "$PORT" \
      envfactory-eval "$VLLM_UTIL" "$MAX_LEN" \
      >"$LOGDIR/vllm_card${CARD}.log" 2>&1 &
  SERVE_PID=$!
  for i in $(seq 1 120); do
    if curl -sf "$ENDPOINT/models" 2>/dev/null | grep -q envfactory-eval; then
      log "vLLM ready (pid $SERVE_PID)"; return 0; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      log "ERROR vLLM died during startup (see vllm_card${CARD}.log)"; return 1; fi
    sleep 5
  done
  log "ERROR vLLM not ready after 10min"; return 1
}
kill_serve(){
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
  pkill -f "vllm serve .* --port $PORT " 2>/dev/null || true
  # wait for graceful shutdown; escalate to -9 if the engine lingers
  for i in $(seq 1 12); do
    curl -sf "$ENDPOINT/models" >/dev/null 2>&1 || break
    sleep 5
  done
  [ -n "$SERVE_PID" ] && kill -9 "$SERVE_PID" 2>/dev/null || true
  pkill -9 -f "vllm serve .* --port $PORT " 2>/dev/null || true
  SERVE_PID=""
  # Wait until GPU3 has enough free room for one 0.45-util (~37 GiB) reservation
  # before serving again. NOTE: two workers co-locate on GPU3, so we canNOT wait
  # for used<5GiB (the sibling worker legitimately holds ~37GiB). Instead require
  # used<=42GiB, i.e. our own previous serve has released and at most one sibling
  # remains. Solo runs (base) still pass instantly once memory drops to ~0.
  for i in $(seq 1 36); do
    used="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    [ -n "$used" ] && [ "$used" -le 42000 ] 2>/dev/null && break
    sleep 5
  done
  sleep 3
}

# ---------------------------------------------------------------- benchmarks
run_bfcl(){ # <out_dir>
  local out="$1"; mkdir -p "$out"
  for cat in single_turn multi_turn; do
    log "BFCL $cat -> $out"
    BFCL_RUN_ROOT="$out" MODEL_DIR="$MODEL_PATH" ENDPOINT="$ENDPOINT" \
      EF_PROMPT=1 SKIP_REGISTER=1 THREADS=8 TEMPERATURE="$TEMPERATURE" \
      bash "$EF/scripts/eval/bfcl/run_bfcl.sh" "$cat" \
      >>"$out/bfcl_${cat}.log" 2>&1 && log "BFCL $cat OK" || log "BFCL $cat FAILED"
  done
}

run_tau2(){ # <out_dir>
  local out="$1"; mkdir -p "$out"
  for d in airline retail telecom; do
    log "tau2 $d -> $out"
    NUM_TASKS=all AGENT_ENDPOINT="$ENDPOINT" RUN_DIR="$out" \
      bash "$EF/scripts/eval/tau2/run_tau2.sh" "$d" \
      >>"$out/tau2_${d}.log" 2>&1 && log "tau2 $d OK" || log "tau2 $d FAILED"
  done
}

run_atlas(){ # <out_dir> <score_name>
  local out="$1" name="$2"; mkdir -p "$out"
  log "atlas run_eval ($ATLAS_TASKS) -> $out"
  ( cd "$ATLAS_DIR" && HARNESS_URL="http://localhost:$HRN" MCP_SANDBOX_URL="http://localhost:$SBX" \
      python run_eval.py --model envfactory-eval --input "$ATLAS_TASKS" \
      --output "$out/outputs.csv" --concurrency "$ATLAS_CONCURRENCY" --timeout 1800 \
      --system-prompt "$SYS_PROMPT" --extra-llm-params "{\"temperature\": $TEMPERATURE}" ) \
      >"$out/atlas_run.log" 2>&1 && log "atlas run OK" || log "atlas run FAILED"
  log "atlas score (dmxapi judge) -> $out/score"
  ( cd "$ATLAS_DIR"; set -a; . "$ATLAS_DIR/.env"; set +a
    python services/scoring/score_claims.py \
      --groundtruth-file "$ATLAS_TASKS" --model-file "$out/outputs.csv" \
      --model-name "$name" --output-dir "$out/score" ) \
      >"$out/atlas_score.log" 2>&1 && log "atlas score OK" || log "atlas score FAILED"
}

# ---------------------------------------------------------------- queue claim
claim(){ # <model> -> 0 if this worker won it
  mkdir "$QUEUE/${1}.lock" 2>/dev/null
}

process_model(){ # <model>
  local m="$1"; MODEL_PATH="$(path_of "$m")"
  log "=== CLAIMED $m ($MODEL_PATH) ==="
  for seed in "${SEEDS[@]}"; do
    local base="$SWEEP/$m/seed$seed"
    if [ -f "$base/.done" ]; then log "$m seed$seed already done -> skip"; continue; fi
    if ! serve_model "$MODEL_PATH" "$seed"; then log "$m seed$seed serve failed -> skip seed"; continue; fi
    run_bfcl  "$base/bfcl"
    run_tau2  "$base/tau2"
    run_atlas "$base/mcp_atlas" "${m}_seed${seed}"
    kill_serve
    touch "$base/.done"
    log "=== DONE $m seed$seed ==="
  done
  touch "$QUEUE/${m}.done"
  log "=== FINISHED $m (all seeds) ==="
}

# ---------------------------------------------------------------- main
log "worker start (gpu=$GPU vllm=$PORT sandbox=$SBX harness=$HRN)"
start_sandbox || exit 1
start_harness || exit 1

while true; do
  claimed_any=0
  for m in "${MODEL_ORDER[@]}"; do
    [ -f "$QUEUE/${m}.done" ] && continue
    if claim "$m"; then claimed_any=1; process_model "$m"; fi
  done
  # exit when every model is done
  alldone=1
  for m in "${MODEL_ORDER[@]}"; do [ -f "$QUEUE/${m}.done" ] || alldone=0; done
  [ "$alldone" = 1 ] && break
  [ "$claimed_any" = 0 ] && sleep 60   # nothing free to claim; other worker busy
done

kill_serve
tmux kill-session -t "rsweep_harness_$CARD" 2>/dev/null || true
docker rm -f "$SBXNAME" >/dev/null 2>&1 || true
log "worker DONE — all models finished"
