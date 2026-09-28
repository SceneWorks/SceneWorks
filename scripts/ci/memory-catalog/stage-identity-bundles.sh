#!/usr/bin/env bash
# Bind the operator-staged identity stacks for this campaign run (sc-22738).
#
# WHY THIS EXISTS. `instantid_realvisxl` and `pulid_flux` load an identity stack that is NOT a
# manifest download on either lane: the worker fetches it on first use, so `--download-missing` has
# no repository to pass the hub CLI and the walk plans those four cells `weights_missing` naming the
# variable. Run 34356681566 did exactly that on the CUDA box, which exports none of the three.
#
# WHY IT IS COPY-IF-SET RATHER THAN A JOB-LEVEL `env:`. A job-level `env:` with an empty
# `${{ inputs.x }}` would EXPORT AN EMPTY STRING and thereby blank a runner that already exports
# the real path in its service environment — turning "the operator left the input alone" into
# "the operator unstaged the bundle". Writing to $GITHUB_ENV only for a non-empty input leaves the
# runner's own value untouched, so either way of staging works and the input simply wins.
#
# NOT VALIDATED HERE ON PURPOSE. Whether a named directory exists and holds the right files is
# `--list`'s question, and it already answers it by name (`stagedEnv` / `bundle` in
# `scripts/measure-memory-catalog.mjs`). Re-checking it here would only produce a second, worse
# error message for the same condition.
set -euo pipefail

staged=()
for pair in \
  "SCENEWORKS_INSTANTID_WEIGHTS:${INPUT_INSTANTID_WEIGHTS:-}" \
  "SCENEWORKS_INSTANTID_CONTROLNET:${INPUT_INSTANTID_CONTROLNET:-}" \
  "SCENEWORKS_PULID_WEIGHTS:${INPUT_PULID_WEIGHTS:-}"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [ -n "$value" ]; then
    printf '%s=%s\n' "$name" "$value" >> "$GITHUB_ENV"
    staged+=("$name")
  fi
done

if [ "${#staged[@]}" -eq 0 ]; then
  echo "identity bundles: no dispatch input set; using whatever this runner already exports"
else
  echo "identity bundles: bound from the dispatch inputs -> ${staged[*]}"
fi
