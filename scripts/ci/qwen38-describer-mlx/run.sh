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
#     under the shared Metal execution lease; guarded_run.sh records the guard's summary line.
#
# QWEN38_JOB = expander (default describer) runs the MiniMax-H3-Image prompt expander smoke instead:
# one production prompt_refine expand job on the selected MLX tier (q4 or q8 only), with only that
# tier's folder and the upstream prompt/ files fetched, under the same guards and caps (the same
# 27B model at the same bit widths). QWEN38_FRESH_HUB=true (expander only) fetches them into a new
# hub scoped to this run (<weights root>/fresh-hubs/<run id>-<attempt>/hub), removed on exit, so the
# smoke records tier_isolated=true: nothing but the selected tier was there to load.
# QWEN38_EXPANDER_MODEL = abliterated (expander only, SW-7208; default minimax) runs the expansion with
# the abliterated describer's own MLX tier folder (q4-full/ or q8-full/ of
# SceneWorks/qwen3.8-27b-abliterated-mlx) and the same upstream prompt files instead.
#
# Both jobs also fetch the two Qwen3.8-27B block drafts (SW-7204: the DFlash2 and DSpine soft
# co-requisites of the entry that runs), each at its pinned revision, and QWEN38_BLOCK_DRAFT = record
# (default) | on | no_fit picks what the smoke checks about the drafter (see the workflow).
#
# Inputs (env): QWEN38_JOB = describer | expander, QWEN38_FRESH_HUB = true | false (default false),
# QWEN38_EXPANDER_MODEL = minimax | abliterated (default minimax),
# QWEN38_TIER = q4 | q8 | bf16,
# QWEN38_IMAGE_SIDE = default | 1536, QWEN38_EVIDENCE_DIR, QWEN38_MAC_RUNNER = nax-macos | nax-macos-2.
set -euo pipefail

out="${QWEN38_EVIDENCE_DIR:?}"
job="${QWEN38_JOB:-describer}"
tier="${QWEN38_TIER:?}"
side="${QWEN38_IMAGE_SIDE:-1536}"
case "$job" in
  describer) ;;
  expander)
    [[ "$tier" == q4 || "$tier" == q8 ]] || { echo "::error::the prompt expander has no published $tier tier; use q4 or q8"; exit 1; } ;;
  *) echo "::error::QWEN38_JOB must be describer or expander, got $job"; exit 1 ;;
esac
expander_model="${QWEN38_EXPANDER_MODEL:-minimax}"
case "$expander_model" in
  minimax) ;;
  abliterated)
    [[ "$job" == expander ]] || { echo "::error::QWEN38_EXPANDER_MODEL applies only to the expander job"; exit 1; } ;;
  *) echo "::error::QWEN38_EXPANDER_MODEL must be minimax or abliterated, got $expander_model"; exit 1 ;;
esac
block_draft="${QWEN38_BLOCK_DRAFT:-record}"
case "$block_draft" in
  record|on|no_fit) ;;
  *) echo "::error::QWEN38_BLOCK_DRAFT must be record, on or no_fit, got $block_draft"; exit 1 ;;
esac
fresh_hub="${QWEN38_FRESH_HUB:-false}"
case "$fresh_hub" in
  false) ;;
  true)
    [[ "$job" == expander ]] || { echo "::error::QWEN38_FRESH_HUB applies only to the expander job"; exit 1; }
    [[ "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ && "${GITHUB_RUN_ATTEMPT:-}" =~ ^[0-9]+$ ]] \
      || { echo "::error::a fresh hub needs the workflow run id and attempt"; exit 1; } ;;
  *) echo "::error::QWEN38_FRESH_HUB must be true or false, got $fresh_hub"; exit 1 ;;
esac
mkdir -p "$out"
here="$(cd "$(dirname "$0")" && pwd)"
guard_dir="$(cd "$here/../iris-mlx-real-weights" && pwd)"

mem_gib=$(( $(sysctl -n hw.memsize) / 1073741824 ))
{
  if [[ "$job" == expander ]]; then
    echo "runner=${RUNNER_NAME:-unknown} source_sha=$(git rev-parse HEAD) job=expander expander_model=$expander_model tier=$tier block_draft=$block_draft"
  else
    echo "runner=${RUNNER_NAME:-unknown} source_sha=$(git rev-parse HEAD) tier=$tier image_side=$side block_draft=$block_draft"
  fi
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
if [[ "$fresh_hub" == true ]]; then
  fresh_root="$weights_root/fresh-hubs/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
  [[ ! -e "$fresh_root" ]] || { echo "::error::the run-scoped hub $fresh_root already exists"; exit 1; }
  # Only this run's own hub, by its exact run-scoped path.
  trap 'rm -rf "$fresh_root"' EXIT
  hub="$fresh_root/hub"
  echo "fresh run-scoped hub: $hub" | tee -a "$out/host.txt"
else
  hub="$weights_root/hub"
fi
mkdir -p "$hub"
python3.12 -m venv "$RUNNER_TEMP/qwen38-hf-venv"
"$RUNNER_TEMP/qwen38-hf-venv/bin/python" -m pip install --disable-pip-version-check --no-input "huggingface_hub==0.36.0" >/dev/null
free_gb=$(( $(df -k "$hub" | awk 'NR==2 {print $4}') / 1048576 ))
if [[ "$job" == expander ]]; then
# The expander: the catalog's macOS row for the tier (its whole `q4-full/` or `q8-full/` folder; for
# the abliterated model the describer's row and the files it names) and the `prompt` co-requisite
# (five upstream files), each at its pinned revision. Each file is checked at its published size,
# so a partial folder is completed rather than taken as resident.
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
QWEN38_HUB="$hub" QWEN38_TIER="$tier" QWEN38_FREE_GB="$free_gb" QWEN38_EXPANDER_MODEL="$expander_model" \
  "$RUNNER_TEMP/qwen38-hf-venv/bin/python" - <<'PY' | tee "$out/provision.txt"
import json, os, sys
from fnmatch import fnmatchcase
from huggingface_hub import HfApi, snapshot_download
specs = json.load(open("config/manifests/registry/specs.json"))
tier = os.environ["QWEN38_TIER"]
downloads = specs["minimax_h3_image_pe"]["product"]["downloads"]
weights = (specs["vision_describe_qwen3_8_27b"]["product"]["downloads"]
           if os.environ["QWEN38_EXPANDER_MODEL"] == "abliterated" else downloads)
tier_rows = [r for r in weights if r.get("variant") == tier and not r.get("coRequisite")
             and not r.get("pendingArtifact") and "macos" in r.get("platforms", [])]
prompt_rows = [r for r in downloads if r.get("coRequisite") and r.get("componentId") == "prompt"]
if len(tier_rows) != 1 or len(prompt_rows) != 1:
    sys.exit(f"expected one macOS {tier} row and one prompt row, found {len(tier_rows)} and {len(prompt_rows)}")
free = int(os.environ["QWEN38_FREE_GB"])
for label, row in (("tier " + tier, tier_rows[0]), ("prompt", prompt_rows[0])):
    info = HfApi().model_info(row["repo"], revision=row["revision"], files_metadata=True, token=False)
    if info.sha != row["revision"]:
        sys.exit(f"::error::{row['repo']} resolved {row['revision']} to {info.sha}")
    expected = {s.rfilename: s.size for s in info.siblings
                if any(fnmatchcase(s.rfilename, pattern) for pattern in row["files"])}
    total = sum(expected.values())
    if not expected or total != row["estimatedSizeBytes"]:
        sys.exit(f"::error::{row['repo']}@{row['revision']} {row['files']} total {total} bytes, the catalog says {row['estimatedSizeBytes']}")
    snap = os.path.join(os.environ["QWEN38_HUB"], "models--" + row["repo"].replace("/", "--"), "snapshots", row["revision"])
    def missing():
        return [n for n, size in expected.items()
                if not os.path.isfile(os.path.join(snap, n)) or os.path.getsize(os.path.join(snap, n)) != size]
    absent = missing()
    if absent:
        need = sum(expected[n] for n in absent) / 1e9
        if free - need < 25:
            sys.exit(f"::error::{free} GB free; the {need:.1f} GB {label} download would leave less than 25 GB")
        snapshot_download(repo_id=row["repo"], revision=row["revision"], cache_dir=os.environ["QWEN38_HUB"],
                          allow_patterns=row["files"], token=False, max_workers=4)
        if missing():
            sys.exit(f"::error::{label}: files have the wrong size after download: {missing()}")
    print(f"{label}: {row['repo']}@{row['revision']} files={row['files']} ({len(expected)} files, {total} bytes) resident={not absent}")
PY
else
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
fi

# ---- the block drafts (SW-7204): the running entry's DFlash2 and DSpine co-requisite rows ---------
if [[ "$job" == expander && "$expander_model" == minimax ]]; then draft_entry=minimax_h3_image_pe
else draft_entry=vision_describe_qwen3_8_27b; fi
free_gb=$(( $(df -k "$hub" | awk 'NR==2 {print $4}') / 1048576 ))
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
QWEN38_HUB="$hub" QWEN38_FREE_GB="$free_gb" QWEN38_DRAFT_ENTRY="$draft_entry" \
  "$RUNNER_TEMP/qwen38-hf-venv/bin/python" - <<'PY' | tee -a "$out/provision.txt"
import json, os, sys
from huggingface_hub import HfApi, snapshot_download
specs = json.load(open("config/manifests/registry/specs.json"))
drafts = ("SceneWorks/qwen3.8-27b-dflash2", "SceneWorks/qwen3.8-27b-dspine-b16")
rows = [r for r in specs[os.environ["QWEN38_DRAFT_ENTRY"]]["product"]["downloads"]
        if r.get("coRequisite") and r.get("repo") in drafts]
if len(rows) != len(drafts):
    sys.exit(f"::error::expected {len(drafts)} block-draft rows, found {len(rows)}")
free = int(os.environ["QWEN38_FREE_GB"])
for row in rows:
    info = HfApi().model_info(row["repo"], revision=row["revision"], files_metadata=True, token=False)
    if info.sha != row["revision"]:
        sys.exit(f"::error::{row['repo']} resolved {row['revision']} to {info.sha}")
    expected = {s.rfilename: s.size for s in info.siblings if s.rfilename in row["files"]}
    total = sum(expected.values())
    if set(expected) != set(row["files"]) or total != row["footprint"]["diskSizeBytes"]:
        sys.exit(f"::error::{row['repo']}@{row['revision']} {row['files']} total {total} bytes, the catalog says {row['footprint']['diskSizeBytes']}")
    snap = os.path.join(os.environ["QWEN38_HUB"], "models--" + row["repo"].replace("/", "--"), "snapshots", row["revision"])
    def missing():
        return [n for n, size in expected.items()
                if not os.path.isfile(os.path.join(snap, n)) or os.path.getsize(os.path.join(snap, n)) != size]
    absent = missing()
    if absent:
        need = sum(expected[n] for n in absent) / 1e9
        if free - need < 25:
            sys.exit(f"::error::{free} GB free; the {need:.1f} GB block-draft download would leave less than 25 GB")
        snapshot_download(repo_id=row["repo"], revision=row["revision"], cache_dir=os.environ["QWEN38_HUB"],
                          allow_patterns=row["files"], token=False, max_workers=4)
        if missing():
            sys.exit(f"::error::{row['repo']}: files have the wrong size after download: {missing()}")
        free -= int(need)
    print(f"block draft: {row['repo']}@{row['revision']} ({len(expected)} files, {total} bytes) resident={not absent}")
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
if [[ "$job" == expander ]]; then
  test_name=h3_prompt_expander_gpu_smoke::h3_prompt_expander_expands_a_prompt_on_mlx
  log="$out/expander.log"
  summary="## MiniMax-H3-Image prompt expander ($expander_model), MLX $tier (${RUNNER_NAME:-unknown})"
else
  test_name=qwen38_describer_gpu_smoke::qwen38_describer_describes_a_synthetic_scene_on_mlx
  log="$out/describer.log"
  summary="## Qwen3.8 27B describer, MLX $tier at image side $side (${RUNNER_NAME:-unknown})"
fi
mkdir -p "$out/result"
echo "$summary" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
bash "$here/guarded_run.sh" "$out" "$log" "$test_name" -- \
  python3.12 -I "$here/../metal_execution_lock.py" --wait-seconds 3600 -- \
  env HF_HUB_CACHE="$hub" HF_HUB_OFFLINE=1 TMPDIR="$RUNNER_TEMP" MODEL_PROOF_OUTPUT_DIR="$out/result" \
  SCENEWORKS_QWEN38_DESCRIBER_TIER="$tier" SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE="$side" \
  SCENEWORKS_H3_EXPANDER_TIER="$tier" SCENEWORKS_H3_EXPANDER_MODEL="$expander_model" \
  SCENEWORKS_QWEN38_BLOCK_DRAFT="$block_draft" \
  python3.12 "$guard_dir/tree_guard.py" "$cap_gib" 2400 "$log" -- \
  "$bin" "$test_name" --ignored --exact --nocapture --test-threads=1
