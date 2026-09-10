#!/usr/bin/env bash
# Walk every planned anchor for this backend, pushing the branch as measurements land (sc-22738).
#
# WHY A BACKGROUND PUSHER. The walk is a single long-lived node process that commits one anchor at
# a time and can run for many hours, and it has no push of its own. If the only push happened after
# the process returned, a cancel, a job timeout or a wedged render would throw away every anchor
# already measured -- hours of GPU time that cannot be re-derived cheaply. So the walk runs in the
# background and a poller pushes the branch every time
# $PUSH_EVERY new commits have appeared. `push.sh` then runs under `if: always()` and pushes the
# remainder on success, failure, timeout and cancellation alike.
#
# WHY THERE IS NO PER-ANCHOR TIMEOUT HERE. There is one, and it belongs to the harness rather than
# to this script (sc-22738, run 34356681566): `measure-memory-catalog.mjs` bounds each capture's
# process tree by that anchor's own `PROBE_BUDGET_MINUTES` and reports the cell
# `runtime_budget_exceeded` before moving to the NEXT anchor. A timeout here could only kill the
# whole walk, which is precisely the outcome -- 47 unreached cells -- the bound exists to prevent.
#
# The poller only ever runs `git push` -- it never commits, never checks out and never touches the
# worktree -- so it cannot make the tree dirty underneath the harness's stability assertion.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

build_catalog_args
CATALOG_ARGS+=(--adapter "$ADAPTER")
while IFS= read -r flag; do
  [[ -z "$flag" ]] && continue
  CATALOG_ARGS+=("$flag")
done < <(download_missing_args)

LOG="${WORK_DIR}/campaign.log"
: > "$LOG"

echo "walking: node scripts/measure-memory-catalog.mjs ${CATALOG_ARGS[*]}"

node scripts/measure-memory-catalog.mjs "${CATALOG_ARGS[@]}" >>"$LOG" 2>&1 &
WALK_PID=$!

# Stream the walk into the job log so an operator can watch progress live.
tail -f "$LOG" &
TAIL_PID=$!

if [[ ! "${PUSH_EVERY:-1}" =~ ^[1-9][0-9]*$ ]]; then
  echo "::warning title=Bad push_every::'${PUSH_EVERY:-}' is not a positive integer; pushing after every anchor"
  PUSH_EVERY=1
fi

pushed=0
base_count="$(git rev-list --count HEAD)"

cleanup() {
  kill "$TAIL_PID" 2>/dev/null || true
}
trap cleanup EXIT

while kill -0 "$WALK_PID" 2>/dev/null; do
  sleep 60
  count="$(git rev-list --count HEAD)"
  landed=$(( count - base_count ))
  if (( landed - pushed >= PUSH_EVERY )); then
    if git push --set-upstream origin "HEAD:${CAMPAIGN_BRANCH}"; then
      pushed=$landed
      echo "pushed ${pushed} landed anchor commit(s) to ${CAMPAIGN_BRANCH}"
    else
      # A transient push failure must not kill a multi-hour walk; the always() push retries.
      echo "::warning title=Campaign push failed::could not push after ${landed} commits; will retry"
    fi
  fi
done

set +e
wait "$WALK_PID"
WALK_STATUS=$?
set -e

cleanup
# Give tail a moment to flush the tail of the log before the step ends.
sleep 2

echo "walk exited with status ${WALK_STATUS}"
{
  echo "#### Walk result"
  echo
  echo "Exit status \`${WALK_STATUS}\` (2 means at least one anchor did not reach \`committed\`/\`captured\`)."
  echo
  echo '```'
  tail -n 80 "$LOG"
  echo '```'
  echo
} >> "$GITHUB_STEP_SUMMARY"

exit "$WALK_STATUS"
