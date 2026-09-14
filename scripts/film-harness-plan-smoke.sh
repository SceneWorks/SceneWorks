#!/usr/bin/env bash
# Real-LLM smoke for the local PLANNER (epic 22708, sc-22713).
#
# Starts a SceneWorks API and the native worker from THIS checkout, waits for a worker advertising
# `prompt_refine`, and drives `film-harness plan` end to end: the brief plus the approved reference
# pack in, `plan.json` + `compiled.json` out, then `film-harness validate` over what was written.
# Everything it touches lives under $SCENEWORKS_SMOKE_DIR (default
# ./film-harness-runs/plan-smoke-<utc>) except the prompt-refine weights, which come from the
# ambient HF cache.
#
#   scripts/film-harness-plan-smoke.sh                    # plan the six-beat courier brief
#   BRIEF=... REFERENCES=... scripts/film-harness-plan-smoke.sh
#   MAX_REPAIR_ROUNDS=0 scripts/film-harness-plan-smoke.sh  # one planning call, no repair
#
# NO VIDEO WEIGHTS ARE LOADED and nothing is rendered: the only model that loads is the
# prompt-refine checkpoint (~16 GB, `TheDrummer/Anubis-Mini-8B-v1`), once, reused across calls.
# Expect one decode for the plan (plus one per repair round) and one per shot for the prompt
# refinement — on the dev Mac, roughly 10-25 minutes for the six-beat brief, and bounded above by
# --llm-timeout-seconds per call. `--skip-install-check` is deliberate: planning validates against
# MiniMax-H3's DECLARED menus, which needs the catalog entry, not the weights.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRIEF="${BRIEF:-$ROOT/config/film-harness/courier-workshop/brief.jsonc}"
REFERENCES="${REFERENCES:-$ROOT/config/film-harness/courier-workshop/references.jsonc}"
MAX_REPAIR_ROUNDS="${MAX_REPAIR_ROUNDS:-2}"
LLM_TIMEOUT="${LLM_TIMEOUT_SECONDS:-1200}"
PORT="${SCENEWORKS_API_PORT:-8766}"
API_URL="http://127.0.0.1:$PORT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_DIR="${SCENEWORKS_SMOKE_DIR:-$ROOT/film-harness-runs/plan-smoke-$STAMP}"
PROFILE="${CARGO_PROFILE:-release}"
# The lane the PLANNING worker claims on. `Settings::from_env` defaults `SCENEWORKS_GPU_ID` to
# "cpu" (crates/sceneworks-worker/src/settings.rs), and a cpu worker spawns the utility pool, which
# advertises no `prompt_refine` — so this script died at its own 180 s registration wait with a
# worker that had started perfectly well (sc-22713 smoke run 1). Same block as
# scripts/film-harness-smoke.sh; planning loads no video weights on either lane, only the
# ~16 GB prompt-refine checkpoint.
if [ -z "${SCENEWORKS_GPU_ID:-}" ]; then
  case "$(uname -s)" in
    Darwin) GPU_ID="mlx" ;;
    *) GPU_ID="0" ;;
  esac
else
  GPU_ID="$SCENEWORKS_GPU_ID"
fi

mkdir -p "$SMOKE_DIR/data" "$SMOKE_DIR/config"

echo "film-harness-plan-smoke: building sceneworks-rust-api + film-harness ($PROFILE)"
if [ "$PROFILE" = "release" ]; then
  cargo build -p sceneworks-rust-api --release --bin sceneworks-rust-api --bin film-harness
  BIN_DIR="$ROOT/target/release"
else
  cargo build -p sceneworks-rust-api --bin sceneworks-rust-api --bin film-harness
  BIN_DIR="$ROOT/target/debug"
fi

API_PID=""
WORKER_PID=""
cleanup() {
  status=$?
  if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
    # SIGTERM only: the worker finishes what it is decoding and exits.
    kill -TERM "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
    kill -TERM "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  echo "film-harness-plan-smoke: logs in $SMOKE_DIR (api.log, worker.log); documents under $SMOKE_DIR/planned"
  exit "$status"
}
trap cleanup EXIT INT TERM

export SCENEWORKS_DATA_DIR="$SMOKE_DIR/data"
export SCENEWORKS_CONFIG_DIR="$SMOKE_DIR/config"
export SCENEWORKS_API_HOST=127.0.0.1
export SCENEWORKS_API_PORT="$PORT"
export SCENEWORKS_TRUST_LOOPBACK=1
unset SCENEWORKS_ACCESS_TOKEN

echo "film-harness-plan-smoke: starting API on $API_URL"
"$BIN_DIR/sceneworks-rust-api" >"$SMOKE_DIR/api.log" 2>&1 &
API_PID=$!

i=0
until curl -fsS "$API_URL/api/v1/health" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 120 ]; then
    echo "film-harness-plan-smoke: API did not become healthy; see $SMOKE_DIR/api.log" >&2
    exit 1
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "film-harness-plan-smoke: API exited early; see $SMOKE_DIR/api.log" >&2
    exit 1
  fi
  sleep 1
done

echo "film-harness-plan-smoke: starting the worker (prompt_refine, SCENEWORKS_GPU_ID=$GPU_ID)"
SCENEWORKS_WORKER_ONLY=1 SCENEWORKS_API_URL="$API_URL" SCENEWORKS_WORKER_ID="film-harness-plan-smoke" \
  SCENEWORKS_GPU_ID="$GPU_ID" \
  "$BIN_DIR/sceneworks-rust-api" >"$SMOKE_DIR/worker.log" 2>&1 &
WORKER_PID=$!

i=0
until curl -fsS "$API_URL/api/v1/workers" 2>/dev/null | grep -q '"prompt_refine"'; do
  i=$((i + 1))
  if [ "$i" -gt 180 ]; then
    echo "film-harness-plan-smoke: no worker advertising prompt_refine registered; see $SMOKE_DIR/worker.log" >&2
    exit 1
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "film-harness-plan-smoke: worker exited early; see $SMOKE_DIR/worker.log" >&2
    exit 1
  fi
  sleep 1
done

echo "film-harness-plan-smoke: planning $BRIEF (max $MAX_REPAIR_ROUNDS repair round(s))"
"$BIN_DIR/film-harness" plan \
  --brief "$BRIEF" \
  --references "$REFERENCES" \
  --api "$API_URL" \
  --out "$SMOKE_DIR/planned" \
  --max-repair-rounds "$MAX_REPAIR_ROUNDS" \
  --llm-timeout-seconds "$LLM_TIMEOUT" \
  --poll-seconds 5 \
  --skip-install-check

echo "film-harness-plan-smoke: validating the generated plan against the live catalog"
# `--no-export` and `--skip-install-check` for the same reason the plan step skips the install
# check: this smoke PLANS and renders nothing, so it starts a prompt_refine worker and no utility
# worker. Without --no-export, `validate` reports the missing `timeline_export` worker and the
# smoke fails on a machine where the planner did its job perfectly (sc-22713 smoke run 4); without
# --skip-install-check it would demand the video weights that planning deliberately does not need.
# Everything else `validate` checks — the documents, the catalog entry, the declared menus, the
# compiled requests against the plan they claim — still runs.
"$BIN_DIR/film-harness" validate \
  --plan "$SMOKE_DIR/planned/plan.json" \
  --references "$REFERENCES" \
  --compiled "$SMOKE_DIR/planned/compiled.json" \
  --api "$API_URL" \
  --no-export \
  --skip-install-check

echo "film-harness-plan-smoke: OK — $SMOKE_DIR/planned/plan.json and compiled.json validate"
