#!/usr/bin/env bash
# SW-7202: the Qwen3.8 27B describer's MLX validation on nax-macos-2 — the MLX twin of
# qwen38-describer-validation.yml (CUDA). Dispatch-only (qwen38-describer-mlx-validation.yml), never
# ordinary CI. Describes one synthetic scene with the production describer on one pinned MLX tier
# and records the process's exact peak footprint, the anchor the MLX `minMemoryGbByTier` floor is
# derived from (floor = ceil(peak GiB / 0.9), the catalog's MLX fit rule).
#
# Safety, in order, before any GPU work:
#   * the runner must be the one QWEN38_MAC_RUNNER selects: nax-macos-2, or nax-macos (Michael's dev
#     Mac) only as the loan he approved on 2026-11-18 while nax-macos-2's model disk is full; on
#     nax-macos the hub and the build target live on /Volumes/Models, never the internal disk;
#   * >= 128 GiB unified memory, and `memory_pressure` free GiB >= the cap + 10 GiB right before the run;
#   * the test binary runs under the external phys_footprint guard (iris-mlx-real-weights/tree_guard.py)
#     that SIGTERMs it at the cap or the wall limit (MLX is never SIGKILLed mid-buffer from here), and
#     under the shared Metal execution lease.
#
# Inputs (env): QWEN38_TIER = q4 | q8 | bf16, QWEN38_IMAGE_SIDE = default | 1536, QWEN38_EVIDENCE_DIR,
# QWEN38_MAC_RUNNER = nax-macos | nax-macos-2.
set -euo pipefail

out="${QWEN38_EVIDENCE_DIR:?}"
tier="${QWEN38_TIER:?}"
side="${QWEN38_IMAGE_SIDE:-1536}"
mkdir -p "$out"
here="$(cd "$(dirname "$0")" && pwd)"
guard_dir="$(cd "$here/../iris-mlx-real-weights" && pwd)"

mem_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
{
  echo "runner=${RUNNER_NAME:-unknown} source_sha=$(git rev-parse HEAD) tier=$tier image_side=$side"
  echo "unified memory: $mem_gib GiB"
  /usr/bin/memory_pressure -Q || true
  df -h "$HOME" | tail -1
} | tee "$out/host.txt"
selected="${QWEN38_MAC_RUNNER:?}"
case "$selected" in
  nax-macos-2) weights_root="$HOME/sceneworks-rw-weights" ;;
  nax-macos)
    [[ -d /Volumes/Models && -w /Volumes/Models ]] || { echo "::error::/Volumes/Models is not writable"; exit 1; }
    weights_root="/Volumes/Models/sceneworks-rw-weights" ;;
  *) echo "::error::QWEN38_MAC_RUNNER must be nax-macos or nax-macos-2, got $selected"; exit 1 ;;
esac
if [[ "${RUNNER_NAME:-}" != "$selected" ]]; then
  echo "::error::refusing: selected $selected, landed on ${RUNNER_NAME:-unknown}; no GPU work was started"
  exit 1
fi
if (( mem_gib < 128 )); then
  echo "::error::refusing: $mem_gib GiB unified memory (< 128 GiB)"
  exit 1
fi
case "$tier" in
  q4) cap_gib=40 ;;
  q8) cap_gib=56 ;;
  bf16) cap_gib=90 ;;
  *) echo "::error::QWEN38_TIER must be q4, q8 or bf16, got $tier"; exit 1 ;;
esac

# ---- the pinned tier, from the catalog row (registry specs.json), into the task-owned hub ----------
hub="$weights_root/hub"
mkdir -p "$hub"
python3.12 -m venv "$RUNNER_TEMP/qwen38-hf-venv"
"$RUNNER_TEMP/qwen38-hf-venv/bin/python" -m pip install --disable-pip-version-check --no-input "huggingface_hub==0.36.0" >/dev/null
free_gb=$(( $(df -k "$hub" | awk 'NR==2 {print $4}') / 1048576 ))
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
QWEN38_HUB="$hub" QWEN38_TIER="$tier" QWEN38_FREE_GB="$free_gb" \
  "$RUNNER_TEMP/qwen38-hf-venv/bin/python" - <<'PY' | tee "$out/provision.txt"
import json, os, sys
from huggingface_hub import snapshot_download
specs = json.load(open("config/manifests/registry/specs.json"))
tier = os.environ["QWEN38_TIER"]
rows = [r for r in specs["vision_describe_qwen3_8_27b"]["product"]["downloads"]
        if r.get("variant") == tier and not r.get("coRequisite")
        and (tier == "bf16" or "macos" in r.get("platforms", []))]
if len(rows) != 1:
    sys.exit(f"expected one macOS {tier} row, found {len(rows)}")
row = rows[0]
snap = os.path.join(os.environ["QWEN38_HUB"], "models--" + row["repo"].replace("/", "--"), "snapshots", row["revision"])
present = all(os.path.isfile(os.path.join(snap, f)) for f in row["files"] if "*" not in f)
if not present:
    need = row["estimatedSizeBytes"] / 1e9
    free = int(os.environ["QWEN38_FREE_GB"])
    if free - need < 25:
        sys.exit(f"::error::{free} GB free; the {need:.1f} GB tier download would leave less than 25 GB")
    snapshot_download(repo_id=row["repo"], revision=row["revision"], cache_dir=os.environ["QWEN38_HUB"],
                      allow_patterns=row["files"], token=False, max_workers=4)
print(f"tier {tier}: {row['repo']}@{row['revision']} files={row['files']} resident={present}")
PY

# ---- build (run-scoped per source SHA; unstripped, rust-lang/rust#157750) -------------------------
target="$weights_root/sceneworks-qwen38-mlx-target/$(git rev-parse HEAD)"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target" CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
cargo test --locked --release -p sceneworks-worker --lib --no-run --message-format=json > "$out/native-build.jsonl"
# The worker's release test executable and its own mlx-sys metallib (the binding the Iris lane uses for
# the same `sceneworks_worker` test artifact).
PMETAL_METALLIB_PATH="$(WORK_DIR_NATIVE="$out" node "$here/../memory-catalog/bind-metallib.mjs" "$out/native-build.jsonl" --iris-test --print-path)"
export PMETAL_METALLIB_PATH
bin="$(dirname "$PMETAL_METALLIB_PATH")/sceneworks_worker"
test -x "$bin"

# ---- the guarded run ------------------------------------------------------------------------------
free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
free_gib=$(( mem_gib * ${free_pct:-0} / 100 ))
if [[ -z "$free_pct" || "$free_gib" -lt $((cap_gib + 10)) ]]; then
  echo "::error::not started: ${free_gib} GiB free < cap ${cap_gib} + 10 GiB"
  exit 1
fi
test_name=qwen38_describer_gpu_smoke::qwen38_describer_describes_a_synthetic_scene_on_mlx
log="$out/describer.log"
mkdir -p "$out/result"
set +e
guard="$(python3.12 -I "$here/../metal_execution_lock.py" --wait-seconds 3600 -- \
  env HF_HUB_CACHE="$hub" HF_HUB_OFFLINE=1 TMPDIR="$RUNNER_TEMP" MODEL_PROOF_OUTPUT_DIR="$out/result" \
  SCENEWORKS_QWEN38_DESCRIBER_TIER="$tier" SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE="$side" \
  python3.12 "$guard_dir/tree_guard.py" "$cap_gib" 2400 "$log" -- \
  "$bin" "$test_name" --ignored --exact --nocapture --test-threads=1)"
rc=$?
set -e
cat "$log"
grep '^GUARD:' <<<"$guard" | tee "$out/guard.txt" || true
if [[ "$rc" != 0 ]] || ! grep -q "test result: ok. 1 passed" "$log"; then
  echo "::error::$test_name did not run exactly one passing test"
  exit 1
fi
{
  echo "## Qwen3.8 27B describer, MLX $tier at image side $side (nax-macos-2)"
  echo
  grep '^GUARD:' <<<"$guard" | sed 's/^/    /'
  echo
  echo "The MLX floor is ceil(root_lifetime_max GiB / 0.9)."
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
