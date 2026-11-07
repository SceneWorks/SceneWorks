#!/usr/bin/env bash
# sc-25690 (epic 25678 terminal evidence): the Iris-3B SceneWorks worker real-weight smokes on the
# SECOND Mac (nax-macos-2) through the production job path — generation, depth and restoration
# jobs, and the generation (LoRA / LoKr cancel→resume / two-rank DDP / FULL fine-tune
# cancel→resume), depth and restoration (LoRA and the paper's FULL fine-tune) training jobs with
# their trained artifacts used by a follow-up job. Dispatch-only, gates nothing
# (FEATURE_DEVELOPMENT.md "Gate teardown").
#
# Safety, in order, before any GPU work:
#   * the runner must be nax-macos-2 — the dev Mac (nax-macos) is never used, whatever label routed
#     the job here;
#   * >= 128 GiB unified memory, and `memory_pressure` free GiB >= the test's cap + 10 GiB headroom
#     right before EACH test;
#   * every test binary runs under an external phys_footprint guard summed over its whole process
#     tree (tree_guard.py; the two-rank DDP smoke spawns rank processes) that SIGTERMs the tree at a
#     stated cap and at a wall-clock limit — the only kill path; MLX is never SIGKILLed mid-buffer
#     from here.
#
# ONE unit: every memory figure here is GiB (2^30 bytes) — host memory, free memory, the caps and
# the peaks — and the guard reads BYTES from the kernel ledger (`proc_pid_rusage`
# `ri_phys_footprint` / `ri_lifetime_max_phys_footprint`), never `footprint`'s rounded "N GB" text.
#
# The depth and restoration jobs (inference AND training) run against an HF hub that holds ONLY
# the speridlabs/iris-3b snapshot (+ restoration's three training resources): the Qwen3-VL text
# encoder is provably absent from what they can resolve.
#
# Inputs (env): IRIS_SMOKE_FILTER = all | inference | training | full_ft | <one test name> | cleanup.
# `all` / `training` do NOT include the two FULL fine-tune smokes (each is a ~90 GiB run of its
# own); `full_ft` selects both, and each is selectable by name — one per dispatch fits the job's
# 60-minute ceiling.
set -euo pipefail

out="${IRIS_EVIDENCE_DIR:?}"
mkdir -p "$out"
filter="${IRIS_SMOKE_FILTER:-all}"
sha="$(git rev-parse HEAD)"
target="$HOME/sceneworks-iris-campaign-target/$sha"

# ---- 1. the box -----------------------------------------------------------------------------------
mem_bytes="$(sysctl -n hw.memsize)"
mem_gib=$((mem_bytes / 1073741824))
{
  echo "runner=${RUNNER_NAME:-unknown} source_sha=$sha filter=$filter"
  sysctl -n hw.model machdep.cpu.brand_string
  echo "unified memory: $mem_gib GiB ($mem_bytes bytes)"
  /usr/bin/memory_pressure -Q || true
  df -h "$HOME" | tail -1
} | tee "$out/host.txt"
if [[ "${RUNNER_NAME:-}" != "nax-macos-2" ]]; then
  echo "::error::refusing: this lane runs only on nax-macos-2 (landed on ${RUNNER_NAME:-unknown}); no GPU work was started"
  exit 1
fi
if (( mem_gib < 128 )); then
  echo "::error::refusing: $mem_gib GiB unified memory (< 128 GiB)"
  exit 1
fi

if [[ "$filter" == "cleanup" ]]; then
  # Only this lane's own build trees.
  rm -rf "$HOME/sceneworks-iris-campaign-target"
  echo "removed $HOME/sceneworks-iris-campaign-target" | tee -a "$out/host.txt"
  exit 0
fi

# ---- 2. the weights (offline at test time) --------------------------------------------------------
# The box's task-owned real-weight hub (inference's Iris lanes already materialized the generation
# backbone, upscaler/, Qwen3-VL and the restoration training resources there). Anything pinned
# below that is not yet resident (depth/, the REPA teacher the stage-1 full fine-tune aligns to) is
# fetched here through the HF cache layout.
hub="$HOME/sceneworks-rw-weights/hub"
iris_rev=7445443349bc9abe3c96f01ff793e2098ca012b3
iris_repo_dir="$hub/models--speridlabs--iris-3b"
qwen_dir="$hub/models--Qwen--Qwen3-VL-4B-Instruct"
critic_dir="$hub/models--laion--CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
vgg_dir="$hub/models--timm--vgg16.tv_in1k"
lpips_dir="$hub/models--vivym--lpips"
python3.12 --version | grep -E '^Python 3\.12\.'
python3.12 -m venv "$RUNNER_TEMP/iris-hf-venv"
"$RUNNER_TEMP/iris-hf-venv/bin/python" -m pip install --disable-pip-version-check --no-input "huggingface_hub==0.36.0" >/dev/null
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
IRIS_HUB="$hub" IRIS_REV="$iris_rev" "$RUNNER_TEMP/iris-hf-venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, rev, patterns in [
    ("speridlabs/iris-3b", os.environ["IRIS_REV"], ["config.yaml", "model.safetensors", "depth/*", "upscaler/*"]),
    ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17",
     ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
      "model.safetensors.index.json", "model-*.safetensors"]),
    ("laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup", "9f3e8ee3f383c672388d9178afe70af9e63ac9df", None),
    ("timm/vgg16.tv_in1k", "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f", None),
    ("vivym/lpips", "270571f1fb2a2c4c5f920cec96cd87838c345982", None),
    # The REPA teacher of the official stage1_256 curriculum (`IRIS_REPA_TEACHER_*`).
    ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415", ["model.safetensors"]),
]:
    snap = os.path.join(os.environ["IRIS_HUB"], "models--" + repo.replace("/", "--"), "snapshots", rev)
    # Resident = every literal pattern present and every directory pattern non-empty; a glob-only
    # repo (no patterns) = a non-empty snapshot. A partial snapshot is completed by the fetch.
    def present(p):
        path = os.path.join(snap, p.split("*")[0].rstrip("/"))
        return os.path.isdir(path) and bool(os.listdir(path)) if p.endswith("/*") else (
            os.path.exists(path) if "*" not in p else os.path.isdir(snap))
    have = all(present(p) for p in patterns) if patterns else (
        os.path.isdir(snap) and bool(os.listdir(snap)))
    if have:
        print(f"resident: {repo}@{rev}")
        continue
    print(f"fetching: {repo}@{rev} {patterns or 'whole repo'}", flush=True)
    snapshot_download(repo_id=repo, revision=rev, cache_dir=os.environ["IRIS_HUB"],
                      allow_patterns=patterns, token=False, max_workers=4)
PY
snap="$iris_repo_dir/snapshots/$iris_rev"
for f in config.yaml model.safetensors depth/config.yaml depth/model.safetensors depth/empty_prompt.safetensors \
         upscaler/config.yaml upscaler/model.safetensors upscaler/empty_prompt.safetensors; do
  test -s "$snap/$f" || { echo "::error::missing $snap/$f"; exit 1; }
done
test -d "$qwen_dir/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
test -s "$hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415/model.safetensors"

# The encoder-free hub for the depth / restoration jobs: the Iris repo (+ restoration's training
# resources), and nothing else. A real directory tree whose files are APFS CLONES of the resolved
# hub files (clone_hub.py: `cp -c`, no bytes copied): the worker canonicalizes a training base
# path and requires it inside the app-managed hub, which a symlink into the shared hub would leave
# — and a hard link would share the blob's inode, so a write through this hub could mutate the
# shared one. A clone cannot.
no_encoder="$RUNNER_TEMP/iris-hub-no-encoder"
rm -rf "$no_encoder"
mkdir -p "$no_encoder"
python3.12 "$(dirname "$0")/clone_hub.py" "$no_encoder" "$iris_repo_dir" "$critic_dir" "$vgg_dir" "$lpips_dir"
if find -L "$no_encoder" -maxdepth 1 -iname '*qwen*' | grep -q .; then
  echo "::error::the encoder-free hub resolves a Qwen repository"; exit 1
fi
find "$no_encoder" -maxdepth 1 -exec ls -ld {} + | tee "$out/no-encoder-hub.txt"

# ---- 3. build (run-scoped per source SHA; unstripped, rust-lang/rust#157750) ---------------------
# Keep the existing source-SHA cache layout. Never sweep other SHA trees: another owned
# invocation may still be compiling or reading one. Only the explicit cleanup input is cleanup.
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
build_start=$(date +%s)
cargo test --locked --release -p sceneworks-worker --lib --no-run
bin="$(cargo test --locked --release -p sceneworks-worker --lib --no-run --message-format=json \
  | tee "$out/native-build.jsonl" \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "sceneworks_worker":
        print(m["executable"])' | tail -1)"
test -x "$bin"
# The actual selected no-run artifact is copied before any model starts. Both this executable
# and its selected metallib live outside the uploaded evidence directory; only hashes/receipt
# are retained there. No new Cargo build/target, directory guess, or shared-cache fallback.
native_binding_script="$(cd "$(dirname "$0")/.." && pwd)/memory-catalog/bind-metallib.mjs"
PMETAL_METALLIB_PATH="$(WORK_DIR_NATIVE="$out" node "$native_binding_script" "$out/native-build.jsonl" --iris-test --print-path)"
export PMETAL_METALLIB_PATH
bin="$(dirname "$PMETAL_METALLIB_PATH")/sceneworks_worker"
test -x "$bin"
echo "build ${SECONDS}s total, $(( $(date +%s) - build_start ))s cargo; binary $bin" | tee "$out/build.txt"

# ---- 4. guarded runs ------------------------------------------------------------------------------
failed=0
SCENEWORKS_METAL_ENTRY_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/metal_execution_entry.sh"
export SCENEWORKS_METAL_ENTRY_SCRIPT out mem_gib bin
# shellcheck source=scripts/ci/metal_execution_entry.sh
source "$SCENEWORKS_METAL_ENTRY_SCRIPT"
run_guarded() { # $1 test path, $2 cap GiB, $3 wall s, $4 hub
  local test=$1 cap_gib=$2 limit_s=$3 hub_dir=$4 name="${1##*::}"
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
  guard="$(HF_HUB_CACHE="$hub_dir" HF_HUB_OFFLINE=1 TMPDIR="$RUNNER_TEMP" \
    python3.12 "$(dirname "$0")/tree_guard.py" "$cap_gib" "$limit_s" "$log" -- \
    "$bin" "$test" --ignored --exact --nocapture --test-threads=1)"
  rc=$?
  set -e
  cat "$log"
  grep '^GUARD:' <<<"$guard" | tee -a "$out/guard.txt" || true
  echo "[[IRIS_MLX_SMOKE]] $name $(grep '^exit=' <<<"$guard" | tail -1) hub=$hub_dir" | tee -a "$out/summary.txt"
  if [[ "$rc" != 0 ]] || ! grep -q "test result: ok. 1 passed" "$log"; then
    echo "::error::$test did not run exactly one passing test"
    failed=1
  fi
}

inference_set=(
  "iris_gpu_smoke::iris_worker_gpu_smoke_native_prompt_batch 45 900 $hub"
  "depth_jobs::gpu_smoke::iris_worker_gpu_smoke_depth 30 600 $no_encoder"
  "iris_gpu_smoke::iris_worker_gpu_smoke_restore 40 900 $no_encoder"
)
training_set=(
  "iris_training_gpu_smoke::iris_training_gpu_smoke 45 1200 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_lokr_cancel_resume 45 1200 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_two_rank_ddp 80 1500 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_depth 90 1200 $no_encoder"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_depth_checkpointed_cancel_resume 90 1500 $no_encoder"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_restore 50 1500 $no_encoder"
)
# sc-25690: the FULL fine-tunes through the job path — the admission gate reads the full rows, a
# real job runs under them. Generation: the official stage1_256 (REPA, Muon) canceled after its
# first durable checkpoint, resumed, exported, and the registered checkpoint rendered by a
# generation job. Restoration: the paper's full recipe at a 256² crop, exported and selected by id
# in a restore job. Each run holds the full-model state (f32 masters + grads + optimizer + EMA).
#
# Caps: the guard reads the process FOOTPRINT, which includes MLX's buffer cache. The measured need
# (active peak, cache off) is ~61 GiB for a generation full fine-tune at 512² and ~67 GiB for
# restoration at a 512² crop, and less at these 256² shapes — but a production run keeps the cache,
# so its footprint may climb to the allocator's GC ceiling (0.95 x the recommended working set:
# 109.68 GB = 102.2 GiB on this Mac) without needing it. 104 GiB sits just above that ceiling, so
# the guard only fires on a run that escapes MLX's own limit.
full_ft_set=(
  "iris_training_gpu_smoke::iris_training_gpu_smoke_full_ft_generation_cancel_resume 104 3000 $hub"
  "iris_training_gpu_smoke::iris_training_gpu_smoke_full_ft_restore 104 2400 $no_encoder"
)
case "$filter" in
  all) selected=("${inference_set[@]}" "${training_set[@]}") ;;
  inference) selected=("${inference_set[@]}") ;;
  training) selected=("${training_set[@]}") ;;
  full_ft) selected=("${full_ft_set[@]}") ;;
  *)
    selected=()
    for row in "${inference_set[@]}" "${training_set[@]}" "${full_ft_set[@]}"; do
      [[ "${row%% *}" == *"::$filter" ]] && selected+=("$row")
    done
    ;;
esac
if (( ${#selected[@]} == 0 )); then
  echo "::error::filter '$filter' selects no Iris smoke"
  exit 1
fi
for row in "${selected[@]}"; do
  # shellcheck disable=SC2086
  run_guarded $row
done
exit "$failed"
