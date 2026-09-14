#!/usr/bin/env bash
# Real-render smoke for the local filmmaking harness (epic 22708, sc-22710).
#
# Starts a SceneWorks API (with the in-process utility worker for the ffmpeg export) and the
# native GPU worker from THIS checkout, waits for both to register, renders the selected shots of
# the courier/workshop fixture through `film-harness run`, prints the run record path, and tears
# both processes down. Everything the run touches lives under $SCENEWORKS_SMOKE_DIR (default
# ./film-harness-runs/smoke-<utc>) except the model weights, which come from the ambient HF cache.
#
#   scripts/film-harness-smoke.sh                       # SH010 + SH020 on MiniMax-H3 q4 (MLX)
#   SHOTS=SH010 scripts/film-harness-smoke.sh           # one shot
#   PLAN=... REFERENCES=... scripts/film-harness-smoke.sh
#
# Runs one GPU render at a time; budget the wall clock from the plan's `limits` (7200 s for the
# fixture). Requires `ffmpeg` on PATH (or SCENEWORKS_FFMPEG) for the export.
#
# On macOS this builds the RELEASE profile, so export the RELEASE prebuilt libmlx first or
# pmetal-mlx-sys runs its ~6 minute cmake build of MLX inside this script:
#
#   eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)"
#   export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
#
# (the fetch script defaults to Debug, which is what `cargo test` consumes — a Debug directory is
# the wrong key for this build and fails it rather than falling back).
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAN="${PLAN:-$ROOT/config/film-harness/courier-workshop/plan.jsonc}"
REFERENCES="${REFERENCES:-$ROOT/config/film-harness/courier-workshop/references.jsonc}"
SHOTS="${SHOTS:-SH010,SH020}"
PORT="${SCENEWORKS_API_PORT:-8765}"
API_URL="http://127.0.0.1:$PORT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_DIR="${SCENEWORKS_SMOKE_DIR:-$ROOT/film-harness-runs/smoke-$STAMP}"
PROFILE="${CARGO_PROFILE:-release}"
# The GPU lane the RENDER worker claims on. `Settings::from_env` defaults `SCENEWORKS_GPU_ID` to
# "cpu" (crates/sceneworks-worker/src/settings.rs), and a cpu worker spawns the utility pool and
# advertises no `video_generate` at all — the smoke then always died at its own registration wait
# with a worker that had started perfectly well. The API's in-process utility worker reads
# SCENEWORKS_RUST_WORKER_GPU_ID instead, so it stays on cpu and still serves `timeline_export`.
if [ -z "${SCENEWORKS_GPU_ID:-}" ]; then
  case "$(uname -s)" in
    Darwin) GPU_ID="mlx" ;;
    *) GPU_ID="0" ;;
  esac
else
  GPU_ID="$SCENEWORKS_GPU_ID"
fi

mkdir -p "$SMOKE_DIR/data" "$SMOKE_DIR/config"

# The pack directory is WRITTEN TO (sc-23404): a run speaks every `dialogue` entry carrying `text`
# and leaves the WAV in `sound/` beside the beds. Pointing --references at the checked-in pack would
# therefore write `sound/<role>.tts-<sha>.wav` into the source tree, and two smokes running at once
# would race on that one filename. So copy the pack — document, `references/`, `sound/` — into
# $SMOKE_DIR and run from there, mirroring what `Harness::fixture_pack` does for the tests. The
# document is byte-for-byte the shipped one, so the run record's `referencePack.sha256` is
# unchanged; only its directory moves. An explicit REFERENCES= is taken as given and not copied —
# a caller pointing at their own pack has already chosen where it lives.
if [ "$REFERENCES" = "$ROOT/config/film-harness/courier-workshop/references.jsonc" ]; then
  PACK_SRC="$(dirname "$REFERENCES")"
  PACK_DIR="$SMOKE_DIR/references"
  mkdir -p "$PACK_DIR"
  cp "$REFERENCES" "$PACK_DIR/"
  for sub in references sound; do
    if [ -d "$PACK_SRC/$sub" ]; then
      mkdir -p "$PACK_DIR/$sub"
      cp "$PACK_SRC/$sub"/* "$PACK_DIR/$sub/"
    fi
  done
  REFERENCES="$PACK_DIR/$(basename "$REFERENCES")"
  echo "film-harness-smoke: pack copied to $PACK_DIR (the run writes its spoken clips there)"
fi

if [ -z "${SCENEWORKS_FFMPEG:-}" ] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "film-harness-smoke: ffmpeg is not on PATH and SCENEWORKS_FFMPEG is unset; the export needs it" >&2
  exit 1
fi

echo "film-harness-smoke: building sceneworks-rust-api + film-harness ($PROFILE)"
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
    # SIGTERM only: the worker finishes its current command buffer and exits; never SIGKILL an
    # MLX render mid-flight.
    kill -TERM "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
    kill -TERM "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  echo "film-harness-smoke: logs in $SMOKE_DIR (api.log, worker.log); run record under $SMOKE_DIR/run"
  exit "$status"
}
trap cleanup EXIT INT TERM

export SCENEWORKS_DATA_DIR="$SMOKE_DIR/data"
export SCENEWORKS_CONFIG_DIR="$SMOKE_DIR/config"
export SCENEWORKS_API_HOST=127.0.0.1
export SCENEWORKS_API_PORT="$PORT"
export SCENEWORKS_RUN_UTILITY_INPROCESS=1
export SCENEWORKS_TRUST_LOOPBACK=1
unset SCENEWORKS_ACCESS_TOKEN

echo "film-harness-smoke: starting API on $API_URL"
"$BIN_DIR/sceneworks-rust-api" >"$SMOKE_DIR/api.log" 2>&1 &
API_PID=$!

i=0
until curl -fsS "$API_URL/api/v1/health" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 120 ]; then
    echo "film-harness-smoke: API did not become healthy; see $SMOKE_DIR/api.log" >&2
    exit 1
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "film-harness-smoke: API exited early; see $SMOKE_DIR/api.log" >&2
    exit 1
  fi
  sleep 1
done

echo "film-harness-smoke: starting the GPU worker (SCENEWORKS_GPU_ID=$GPU_ID)"
SCENEWORKS_WORKER_ONLY=1 SCENEWORKS_API_URL="$API_URL" SCENEWORKS_WORKER_ID="film-harness-smoke-gpu" \
  SCENEWORKS_GPU_ID="$GPU_ID" \
  "$BIN_DIR/sceneworks-rust-api" >"$SMOKE_DIR/worker.log" 2>&1 &
WORKER_PID=$!

i=0
until curl -fsS "$API_URL/api/v1/workers" 2>/dev/null | grep -q '"video_generate"'; do
  i=$((i + 1))
  if [ "$i" -gt 180 ]; then
    echo "film-harness-smoke: no worker advertising video_generate registered; see $SMOKE_DIR/worker.log" >&2
    exit 1
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "film-harness-smoke: worker exited early; see $SMOKE_DIR/worker.log" >&2
    exit 1
  fi
  sleep 1
done

echo "film-harness-smoke: validating the plan against the live catalog"
"$BIN_DIR/film-harness" validate --plan "$PLAN" --references "$REFERENCES" --api "$API_URL" --shots "$SHOTS"

echo "film-harness-smoke: rendering $SHOTS"
"$BIN_DIR/film-harness" run \
  --plan "$PLAN" \
  --references "$REFERENCES" \
  --api "$API_URL" \
  --shots "$SHOTS" \
  --out "$SMOKE_DIR/run" \
  --poll-seconds 10

# Ctrl-C during the render above, or `film-harness cancel --out "$SMOKE_DIR/run"` from another
# shell, stops dispatch and cancels the live job; `film-harness resume --out "$SMOKE_DIR/run"`
# picks it back up without re-rendering anything that finished, and `film-harness status --out
# "$SMOKE_DIR/run"` prints where it got to.
