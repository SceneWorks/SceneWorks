#!/usr/bin/env bash
# Shared cooperative execution ownership, never memory admission or a universal scheduler.
# Source once at a Mac GPU launch boundary. Existing argv/options remain unchanged.
metal_execution_entry() {
  local budget="$1"
  shift
  [[ "$(uname -s)" == "Darwin" ]] || return 0
  local helper
  helper="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/metal_execution_lock.py"
  budget="${SCENEWORKS_METAL_EXECUTION_WAIT_SECONDS:-$budget}"
  if [[ -n "${SCENEWORKS_METAL_EXECUTION_LOCK_FD+x}" ]]; then
    # Verify the actual inherited open-file description/kernel flock, never an env-only claim.
    python3.12 -I "$helper" --wait-seconds "$budget" -- python3.12 -I -c 'pass'
    return $?
  fi
  # The helper owns the command tree and defers signals until it drains; no GPU kill here.
  exec python3.12 -I "$helper" --wait-seconds "$budget" -- "$@"
}
