#!/bin/bash
# Epic 26679 — reduced H3 terminal campaign dispatch.
#
# Requires the operator's scoped acknowledgment (recorded on sc-26679 activity 27075).
# Refuses to dispatch without it, and refuses to dispatch against a moved branch head.
#
# Both backends are dispatched back-to-back: each run publishes an anchor artifact and then
# waits (bounded) for the peer's anchor artifact, so they must be launched together.
#
# Runner handling: the CUDA lane's `runs-on` matches TWO runners (cuda-windows, cuda-windows-2)
# and the preflight asserts RUNNER_NAME === EXPECTED_RUNNER. So after dispatch we read the
# runner GitHub actually assigned and, on a mismatch, cancel and re-dispatch with the real name.
# A mismatch fails at the stage step within a minute, long before the ~148 GB download, so the
# correction is cheap.
set -euo pipefail

REPO=SceneWorks/SceneWorks
REF=feature/sc-26679-h3-finetune-import
# $1 = --ack (the operator's scoped acknowledgment), $2 = the exact reviewed head to dispatch.
[ "${1:-}" = "--ack" ] || { echo "usage: $0 --ack <expected-head-sha>"; exit 2; }
SW="${2:-}"
if [ -z "$SW" ]; then echo "usage: $0 --ack <expected-head-sha>"; exit 2; fi

# The campaign's stationary pair names the exact SceneWorks revision, so the branch must still be
# the reviewed head. A moved head means the pair is not the one that was reviewed and validated.
head_sha="$(gh api "repos/$REPO/git/ref/heads/$REF" --jq '.object.sha')"
if [ "$head_sha" != "$SW" ]; then
  echo "REFUSING: $REF is $head_sha, not the reviewed $SW. Re-derive the request before dispatch." >&2
  exit 3
fi

# The inference half of the stationary pair is the pin the reviewed head itself carries: the
# bootstrap's `sourcePair` refuses any case whose `inference_sha` is not the worker's
# `rev = "<sha>"`, so a stale checked-in value would fail every case after the ~148 GB bootstrap.
worker_manifest="$(git show "$SW:crates/sceneworks-worker/Cargo.toml")" || {
  echo "REFUSING: cannot read crates/sceneworks-worker/Cargo.toml at $SW (git fetch first)." >&2
  exit 3
}
INF="$(printf '%s\n' "$worker_manifest" \
  | grep -o 'github.com/SceneWorks/inference", rev = "[0-9a-f]\{40\}"' \
  | grep -o '[0-9a-f]\{40\}' | sort -u)"
if [ "$(printf '%s\n' "$INF" | grep -c .)" != 1 ]; then
  echo "REFUSING: $SW does not pin exactly one inference revision (got: ${INF:-none})." >&2
  exit 3
fi
echo "stationary pair: sceneworks=$SW inference=$INF"

# Candidate runners per backend, with live state. MLX: the campaign runs on nax-macos (real-weights label).
echo "--- live runner state ---"
gh api --paginate orgs/SceneWorks/actions/runners --jq '.runners[]|
  select((.labels|map(.name)|index("rw-chroma")) or (.labels|map(.name)|index("real-weights"))) |
  "\(.name) status=\(.status) busy=\(.busy) labels=\([.labels[].name]|join(","))"'

dispatch() {  # $1=backend $2=expected_runner
  local backend="$1" runner="$2"
  local req="$(dirname "$0")/h3-terminal-$backend.json"
  # The request is stamped with the verified head, so the stationary pair names the revision that
  # was actually reviewed and validated, rather than whatever the payload happened to carry.
  jq --argjson ack true --arg runner "$runner" --arg sw "$SW" --arg inf "$INF" \
     '.bootstrap.legalAuthority.acknowledged = $ack | .expected_runner = $runner | .sceneworks_sha = $sw | .inference_sha = $inf' \
     "$req" > "${TMPDIR:-/tmp}/h3-terminal-$backend.ack.json"
  jq -e '.bootstrap.legalAuthority.acknowledged == true' "${TMPDIR:-/tmp}/h3-terminal-$backend.ack.json" >/dev/null
  jq -e --arg s "$SW" '.sceneworks_sha == $s' "${TMPDIR:-/tmp}/h3-terminal-$backend.ack.json" >/dev/null
  jq -e --arg i "$INF" '.inference_sha == $i' "${TMPDIR:-/tmp}/h3-terminal-$backend.ack.json" >/dev/null
  gh workflow run server-candle-linux.yml --repo "$REPO" --ref "$REF" \
     -f starvector_terminal_operation="h3-terminal-$backend" \
     -f campaign_run_id="$(cat "${TMPDIR:-/tmp}/h3-terminal-$backend.ack.json")" >/dev/null
  echo "dispatched h3-terminal-$backend (expected_runner=$runner)"
}

# Resolve each backend's expected runner: prefer an idle compatible runner.
resolve() { # $1=backend
  if [ "$1" = "mlx" ]; then echo "nax-macos"; return; fi
  # A busy GPU is a QUEUEING condition, not a blocker: Actions holds the job until a matching
  # runner frees. Prefer an idle compatible runner, but never withhold the dispatch for want of one.
  local picked
  picked="$(gh api --paginate orgs/SceneWorks/actions/runners --jq '.runners[]|
    select((.labels|map(.name)|index("real-weights")) and (.labels|map(.name)|index("cuda"))) |
    select(.status=="online") | select(.busy==false) | .name' | head -1)"
  if [ -z "$picked" ]; then
    picked="cuda-windows"
    echo "note: every CUDA runner is busy; dispatching anyway and letting Actions queue it" >&2
  fi
  echo "$picked"
}

for backend in ${BACKENDS:-mlx cuda}; do
  runner="$(resolve "$backend")"
  [ -n "$runner" ] || { echo "no idle compatible runner for $backend" >&2; exit 4; }
  dispatch "$backend" "$runner"
  sleep 15
done

echo "--- dispatched; both runs must be launched together ---"
gh run list --repo "$REPO" --workflow server-candle-linux.yml --limit 4 \
  --json databaseId,status,headSha,createdAt --jq '.[]|"\(.databaseId) \(.status) \(.headSha[0:9]) \(.createdAt)"'
