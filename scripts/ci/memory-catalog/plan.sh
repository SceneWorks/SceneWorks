#!/usr/bin/env bash
# Print what the walk WOULD do, before it touches a GPU (sc-22738).
#
# Two cheap passes, both of which fail the job before any render if the dispatch is wrong:
#   * `--list`   -- the per-anchor status/reason table, published as the job summary. This is where
#                   an operator reads "unavailable" (weights are not on this box) or "current"
#                   (already anchored at the pin) without waiting for the walk.
#   * `--dry-run` -- the same planning path the real walk takes, including the preflight that
#                   compares the inference checkout against the compiled INFERENCE_PIN, checks the
#                   adapter binary exists, and (candle) refuses an unset CUDA_VISIBLE_DEVICES.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

build_catalog_args

LIST_OUT="${WORK_DIR}/plan-list.txt"
node scripts/measure-memory-catalog.mjs "${CATALOG_ARGS[@]}" --list | tee "$LIST_OUT"

{
  echo "#### Planned anchors"
  echo
  echo '```'
  cat "$LIST_OUT"
  echo '```'
  echo
} >> "$GITHUB_STEP_SUMMARY"

echo "--- dry run ---"
node scripts/measure-memory-catalog.mjs "${CATALOG_ARGS[@]}" --adapter "$ADAPTER" --dry-run \
  | tee "${WORK_DIR}/plan-dry-run.txt"
