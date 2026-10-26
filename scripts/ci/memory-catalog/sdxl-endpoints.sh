#!/usr/bin/env bash
# Explicit manual profile: no catalog walk or publication. Preserve the workflow's existing
# CUDA concurrency lease. The adapter rechecks physical GPU1 UUID, activity and headroom after
# compilation and immediately before each load; an unused context is part of the stable baseline.
source "$(dirname "$0")/common.sh"
require_campaign
if [[ ! "$BASE_REF" =~ ^[0-9a-f]{40}$ ]] || [[ "$(git rev-parse HEAD)" != "$BASE_REF" ]]; then
  echo '::error::SDXL endpoint ref must be the exact checked-out 40-hex source SHA'
  exit 1
fi
WORK_DIR="$(work_dir_unix)"
WORK_DIR_NATIVE="$(native_path "$WORK_DIR")"
mkdir -p "$WORK_DIR"
args=(--backend candle --adapter "$ADAPTER" --inference-repo "$INFERENCE_REPO"
  --work-dir "$WORK_DIR_NATIVE" --campaign "$CAMPAIGN")
while IFS= read -r root; do
  [[ -n "$root" ]] && args+=(--hf-cache "$root")
done < <(hf_cache_roots)
if [[ "${MEMORY_CAMPAIGN_PROFILE:-}" == "sdxl-cuda-admitted-replay" ]]; then
  args+=(--admitted-replay)
else
  [[ "${DOWNLOAD_MISSING:-false}" == true ]] && args+=(--download-missing)
fi
node scripts/measure-sdxl-cuda-endpoints.mjs "${args[@]}"
