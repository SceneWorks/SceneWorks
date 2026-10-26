#!/usr/bin/env bash
# sc-24216 (epic 24212): the Supra2-IMG SceneWorks worker real-weight tests on the SECOND Mac
# (nax-macos-2) — the install-time conversion, one default render through the production MLX arm,
# and the installed-model job lifecycle (offline: per-step progress, cancel mid-denoise, a recovery
# job, generator eviction and a reload). Dispatch-only, gates nothing (FEATURE_DEVELOPMENT.md
# "Gate teardown"). Modelled on scripts/ci/iris-mlx-real-weights/run.sh.
#
# Safety, before any GPU work:
#   * the runner must be nax-macos-2 — the dev Mac (nax-macos) is never used, whatever label routed
#     the job here;
#   * `memory_pressure` free GiB >= the test's cap + 10 GiB headroom right before EACH test;
#   * every test binary runs under the Iris lane's external phys_footprint guard
#     (../iris-mlx-real-weights/tree_guard.py) that SIGTERMs the tree at a stated cap and at a
#     wall-clock limit — the only kill path; MLX is never SIGKILLed mid-buffer from here.
#
# Supra2 is small (104M-parameter DiT + Flan-T5-Base encoder + SD VAE, ~1.6 GiB of f32 weights at
# 256x256), so every cap here is 16 GiB: far above the model, far below the box.
set -euo pipefail

out="${SUPRA2_EVIDENCE_DIR:?}"
mkdir -p "$out"
sha="$(git rev-parse HEAD)"
target="$HOME/sceneworks-supra2-campaign-target/$sha"
lane="$(cd "$(dirname "$0")" && pwd)"
guard_py="$lane/../iris-mlx-real-weights/tree_guard.py"

# ---- 1. the box -----------------------------------------------------------------------------------
mem_bytes="$(sysctl -n hw.memsize)"
mem_gib=$((mem_bytes / 1073741824))
{
  echo "runner=${RUNNER_NAME:-unknown} source_sha=$sha"
  sysctl -n hw.model machdep.cpu.brand_string
  echo "unified memory: $mem_gib GiB ($mem_bytes bytes)"
  /usr/bin/memory_pressure -Q || true
  df -h "$HOME" | tail -1
} | tee "$out/host.txt"
if [[ "${RUNNER_NAME:-}" != "nax-macos-2" ]]; then
  echo "::error::refusing: this lane runs only on nax-macos-2 (landed on ${RUNNER_NAME:-unknown}); no GPU work was started"
  exit 1
fi

# ---- 2. the weights (fetched here, offline at test time) ------------------------------------------
# The box's task-owned real-weight hub. Exactly the pinned files the engine reads
# (gen_core::supra2::{SUPRA2_SOURCE, FLAN_T5_BASE, SD_VAE}); the tests re-verify sizes + SHA-256.
hub="$HOME/sceneworks-rw-weights/hub"
mkdir -p "$hub"
python3.12 --version | grep -E '^Python 3\.12\.'
python3.12 -m venv "$RUNNER_TEMP/supra2-hf-venv"
"$RUNNER_TEMP/supra2-hf-venv/bin/python" -m pip install --disable-pip-version-check --no-input "huggingface_hub==0.36.0" >/dev/null
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
SUPRA2_HUB="$hub" "$RUNNER_TEMP/supra2-hf-venv/bin/python" - <<'PY'
import os
from huggingface_hub import hf_hub_download
hub = os.environ["SUPRA2_HUB"]
for repo, rev, files in [
    ("SupraLabs/Supra2-IMG", "10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b", ["model_final_ema.pt"]),
    ("google/flan-t5-base", "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
     ["config.json", "model.safetensors", "tokenizer.json"]),
    ("stabilityai/sd-vae-ft-mse", "31f26fdeee1355a5c34592e401dd41e45d25a493",
     ["config.json", "diffusion_pytorch_model.safetensors"]),
]:
    snap = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", rev)
    for name in files:
        if os.path.isfile(os.path.join(snap, name)):
            print(f"resident: {repo}@{rev}/{name}")
            continue
        print(f"fetching: {repo}@{rev}/{name}", flush=True)
        hf_hub_download(repo_id=repo, filename=name, revision=rev, cache_dir=hub, token=False)
PY

# ---- 3. build (run-scoped per source SHA; unstripped, rust-lang/rust#157750) ---------------------
# Other source revisions may still be queued or running; retain their target trees.
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
build_start=$(date +%s)
cargo test --locked --release -p sceneworks-worker --lib --no-run
bin="$(cargo test --locked --release -p sceneworks-worker --lib --no-run --message-format=json \
  | tee "$out/native-build.jsonl" | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "sceneworks_worker":
        print(m["executable"])' | tail -1)"
test -x "$bin"
native_binding_script="$(cd "$(dirname "$0")/.." && pwd)/memory-catalog/bind-metallib.mjs"
PMETAL_METALLIB_PATH="$(WORK_DIR_NATIVE="$out" node "$native_binding_script" "$out/native-build.jsonl" --supra2-test --print-path)"
export PMETAL_METALLIB_PATH
bin="$(dirname "$PMETAL_METALLIB_PATH")/sceneworks_worker"
test -x "$bin"
echo "build $(( $(date +%s) - build_start ))s; binary $bin" | tee "$out/build.txt"

# ---- 4. guarded runs: each invocation must run EXACTLY one passing test --------------------------
failed=0
SCENEWORKS_METAL_ENTRY_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/metal_execution_entry.sh"
export SCENEWORKS_METAL_ENTRY_SCRIPT out mem_gib bin guard_py hub bundle
# shellcheck source=scripts/ci/metal_execution_entry.sh
source "$SCENEWORKS_METAL_ENTRY_SCRIPT"
run_guarded() { # $1 test path, $2 cap GiB, $3 wall s
  local test=$1 cap_gib=$2 limit_s=$3 name="${1##*::}"
  if [[ -z "${SCENEWORKS_METAL_EXECUTION_LOCK_FD+x}" ]]; then
    export -f run_guarded
    if ! python3.12 -I "$(dirname "$SCENEWORKS_METAL_ENTRY_SCRIPT")/metal_execution_lock.py" \
      --wait-seconds "${SCENEWORKS_METAL_EXECUTION_WAIT_SECONDS:-$limit_s}" -- \
      bash -c 'source "$SCENEWORKS_METAL_ENTRY_SCRIPT"; failed=0; run_guarded "$@"; exit "$failed"' _ "$@"; then
      failed=1
    fi
    return
  fi
  metal_execution_entry "$limit_s" true
  local log="$out/$name.log" free_pct free_gib
  free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
  free_gib=$(( mem_gib * ${free_pct:-0} / 100 ))
  if [[ -z "$free_pct" || "$free_gib" -lt $((cap_gib + 10)) ]]; then
    echo "::error::$name not started: ${free_gib} GiB free (memory_pressure ${free_pct:-?}% of ${mem_gib} GiB) < cap ${cap_gib} + 10 GiB" | tee "$log"
    failed=1
    return
  fi
  local guard rc
  set +e
  guard="$(SUPRA2_HF_HUB_CACHE="$hub" HF_HUB_CACHE="$hub" HF_HUB_OFFLINE=1 SUPRA2_RENDER_OUT="$out" \
    SUPRA2_BUNDLE_OUT="$bundle" SUPRA2_BUNDLE_DIR="$bundle" SUPRA2_LIFECYCLE_LOG="$out/supra2_real_weight_offline_lifecycle_cancel_recover_unload.log" \
    TMPDIR="$RUNNER_TEMP" python3.12 "$guard_py" "$cap_gib" "$limit_s" "$log" -- \
    "$bin" "$test" --ignored --exact --nocapture --test-threads=1)"
  rc=$?
  set -e
  cat "$log"
  grep '^GUARD:' <<<"$guard" | tee -a "$out/guard.txt" || true
  echo "[[SUPRA2_MLX]] $name $(grep '^exit=' <<<"$guard" | tail -1)" | tee -a "$out/summary.txt"
  if [[ "$rc" != 0 ]] || ! grep -q "test result: ok. 1 passed" "$log"; then
    echo "::error::$test did not run exactly one passing test"
    failed=1
  fi
}

# The convert test writes the bundle here (SUPRA2_BUNDLE_OUT); the acceptance matrix, a separate
# process, loads it (SUPRA2_BUNDLE_DIR) and writes its PNGs + supra2_matrix_mlx.json into $out.
# The installed layout (<data dir>/models/mlx/supra2_img): the worker admits a modelPath only
# inside its own data dir, which the matrix takes from this path.
rm -rf "$RUNNER_TEMP/supra2-data"
bundle="$RUNNER_TEMP/supra2-data/models/mlx/supra2_img"
run_guarded "model_jobs::supra2_convert_tests::supra2_real_weight_convert_writes_a_loadable_bundle" 16 600
run_guarded "image_jobs::supra2::tests::mlx_real_weights::supra2_real_weight_render_through_the_worker_arm" 16 900
run_guarded "image_jobs::supra2::tests::mlx_real_weights::supra2_real_weight_offline_lifecycle_cancel_recover_unload" 16 1200
# The matrix folds the lifecycle test's real result (its log above, SUPRA2_LIFECYCLE_LOG) into
# cell 7 of supra2_matrix_mlx.json, so the summary file is the whole truthful verdict.
run_guarded "image_jobs::supra2::tests::mlx_real_weights::supra2_real_weight_acceptance_matrix" 16 1200
exit "$failed"
