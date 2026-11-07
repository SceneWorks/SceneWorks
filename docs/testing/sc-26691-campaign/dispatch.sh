#!/bin/bash
# Epic 26679 — reduced H3 terminal campaign dispatch.
#
# Requires the operator's scoped acknowledgment (recorded on sc-26679 activity 27075).
# Refuses to dispatch without it, and refuses to dispatch against a moved branch head.
#
# Two dispatch shapes:
#   * Paired adaptive matrix (no PRIOR_MATRIX_RUN): both backends are dispatched back-to-back; each
#     run publishes an anchor artifact and then waits (bounded) for the peer's, so they must be
#     launched together.
#   * Declared prior decision (PRIOR_MATRIX_RUN=<run>[/<attempt>]): each lane adopts the matrix
#     decision that run already MEASURED (re-validated from its retained artifacts), exchanges no
#     anchors and needs no peer. CASE_IDS narrows a lane to an explicit subset.
#
# Environment knobs (all optional):
#   BACKENDS          "mlx cuda" (default) or one of them.
#   MLX_RUNNER        nax-macos-2 (default) | nax-macos. Either Mac carries real-weights; the job is
#                     pinned to the named one by its own label (rw-mage / rw-starvector).
#   MLX_HF_HOME       macOS HF home the weights are READ from (stamped as mlx_hf_home; `~/` allowed).
#                     Unset: the workflow default /Volumes/Models/huggingface. Missing pinned files
#                     are downloaded into the run's private staging under the work root.
#   WORK_ROOT         where the campaign WRITES (stamped as work_root on every dispatched lane).
#   CUDA_WORK_ROOT / MLX_WORK_ROOT   per-backend overrides of WORK_ROOT. Unset: the workflow
#                     defaults D:\sc26691-campaign (Windows) and ~/sc26691-campaign (macOS main drive).
#                     A work root on E: or under the macOS HF home is refused by the lane.
#   CASE_IDS          comma-separated case ids; each id must start with "<backend>." and is stamped
#                     only into that backend's request (a lane with no listed id gets no case_ids).
#                     Requires PRIOR_MATRIX_RUN: an adaptive (paired) lane runs its measured allocation
#                     and would ignore an explicit subset, so the kit refuses rather than drop it.
#   PRIOR_MATRIX_RUN  <run_id>[/<attempt>] (attempt defaults to 1): stamped as prior_matrix_decision.
#   PRIOR_LANE_RUN    <run_id>[/<attempt>]: the dispatched lane's OWN earlier run on the same declared
#                     decision (stamped as prior_lane_run), when the decision run is the other backend's
#                     (e.g. MLX run 41331644188 on cuda run 41213935677's decision). Its passed cases are
#                     excluded from, and refused in, the subset. Requires PRIOR_MATRIX_RUN and exactly one
#                     backend in BACKENDS.
#   DRY_RUN           non-empty: stamp and print the requests, dispatch nothing.
#
# Runner handling: the CUDA lane's `runs-on` names the requested runner as a label, and the
# preflight asserts RUNNER_NAME === EXPECTED_RUNNER on both backends.
set -euo pipefail

REPO=SceneWorks/SceneWorks
REF=feature/sc-26679-h3-finetune-import
# $1 = --ack (the operator's scoped acknowledgment), $2 = the exact reviewed head to dispatch.
[ "${1:-}" = "--ack" ] || { echo "usage: $0 --ack <expected-head-sha>"; exit 2; }
SW="${2:-}"
if [ -z "$SW" ]; then echo "usage: $0 --ack <expected-head-sha>"; exit 2; fi

MLX_RUNNER="${MLX_RUNNER:-nax-macos-2}"
case "$MLX_RUNNER" in nax-macos|nax-macos-2) ;; *) echo "REFUSING: MLX_RUNNER must be nax-macos or nax-macos-2, not $MLX_RUNNER" >&2; exit 2;; esac
PRIOR_RUN_ID="" PRIOR_ATTEMPT=""
if [ -n "${PRIOR_MATRIX_RUN:-}" ]; then
  if [[ "$PRIOR_MATRIX_RUN" =~ ^([0-9]+)(/([0-9]+))?$ ]]; then
    PRIOR_RUN_ID="${BASH_REMATCH[1]}"; PRIOR_ATTEMPT="${BASH_REMATCH[3]:-1}"
  else
    echo "REFUSING: PRIOR_MATRIX_RUN must be <run_id>[/<attempt>], not $PRIOR_MATRIX_RUN" >&2; exit 2
  fi
fi
LANE_RUN_ID="" LANE_ATTEMPT=""
if [ -n "${PRIOR_LANE_RUN:-}" ]; then
  [ -n "$PRIOR_RUN_ID" ] || { echo "REFUSING: PRIOR_LANE_RUN continues a lane on a declared prior decision; it needs PRIOR_MATRIX_RUN" >&2; exit 2; }
  case "${BACKENDS:-mlx cuda}" in mlx|cuda) ;; *) echo "REFUSING: PRIOR_LANE_RUN names one lane's own run; set BACKENDS to that one backend, not '${BACKENDS:-mlx cuda}'" >&2; exit 2;; esac
  if [[ "$PRIOR_LANE_RUN" =~ ^([0-9]+)(/([0-9]+))?$ ]]; then
    LANE_RUN_ID="${BASH_REMATCH[1]}"; LANE_ATTEMPT="${BASH_REMATCH[3]:-1}"
  else
    echo "REFUSING: PRIOR_LANE_RUN must be <run_id>[/<attempt>], not $PRIOR_LANE_RUN" >&2; exit 2
  fi
  [ "$LANE_RUN_ID" != "$PRIOR_RUN_ID" ] || { echo "REFUSING: PRIOR_LANE_RUN must differ from PRIOR_MATRIX_RUN" >&2; exit 2; }
fi
# Every CASE_IDS entry must belong to a dispatched backend, so no id is silently dropped.
if [ -n "${CASE_IDS:-}" ] && [ -z "$PRIOR_RUN_ID" ]; then
  echo "REFUSING: CASE_IDS needs PRIOR_MATRIX_RUN; the paired adaptive matrix runs its measured allocation and would ignore an explicit subset" >&2; exit 2
fi
if [ -n "${CASE_IDS:-}" ]; then
  IFS=',' read -r -a _ids <<<"$CASE_IDS"
  for id in "${_ids[@]}"; do
    owner="${id%%.*}"
    case " ${BACKENDS:-mlx cuda} " in *" $owner "*) ;; *) echo "REFUSING: CASE_IDS entry $id belongs to no dispatched backend (${BACKENDS:-mlx cuda})" >&2; exit 2;; esac
  done
fi

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
if [ -n "$PRIOR_RUN_ID" ]; then
  # The prior decision's anchors ran at the prior run's SHA; record the difference plainly here too
  # (the lane records `git diff --name-only <prior>..<this>` in its evidence).
  prior_sha="$(gh api "repos/$REPO/actions/runs/$PRIOR_RUN_ID" --jq '.head_sha')"
  echo "prior matrix decision: run $PRIOR_RUN_ID attempt $PRIOR_ATTEMPT at sceneworks=$prior_sha"
  [ -z "$LANE_RUN_ID" ] || echo "prior lane run (its passed cases are not re-run): run $LANE_RUN_ID attempt $LANE_ATTEMPT at sceneworks=$(gh api "repos/$REPO/actions/runs/$LANE_RUN_ID" --jq '.head_sha')"
  if [ "$prior_sha" != "$SW" ]; then
    echo "--- files changed $prior_sha..$SW ---"
    git diff --name-only "$prior_sha..$SW" || echo "(git fetch first to list the difference locally)"
  fi
fi

# Candidate runners per backend, with live state.
echo "--- live runner state ---"
gh api --paginate orgs/SceneWorks/actions/runners --jq '.runners[]|
  select((.labels|map(.name)|index("rw-chroma")) or (.labels|map(.name)|index("real-weights"))) |
  "\(.name) status=\(.status) busy=\(.busy) labels=\([.labels[].name]|join(","))"'

dispatch() {  # $1=backend $2=expected_runner
  local backend="$1" runner="$2"
  local req="$(dirname "$0")/h3-terminal-$backend.json"
  local work_root
  if [ "$backend" = "mlx" ]; then work_root="${MLX_WORK_ROOT:-${WORK_ROOT:-}}"; else work_root="${CUDA_WORK_ROOT:-${WORK_ROOT:-}}"; fi
  # The request is stamped with the verified head, so the stationary pair names the revision that
  # was actually reviewed and validated, rather than whatever the payload happened to carry.
  local out="${TMPDIR:-/tmp}/h3-terminal-$backend.ack.json"
  # node, not jq: this box (and the Windows runners) have no standalone jq. Stamp, then re-read the
  # written file and assert every stamped field, so a write that lost one never dispatches.
  node -e '
    const fs = require("fs");
    const [req, out, backend, runner, sw, inf, workRoot, hfHome, caseIds, priorRun, priorAttempt, laneRun, laneAttempt] = process.argv.slice(1);
    const p = JSON.parse(fs.readFileSync(req, "utf8"));
    p.bootstrap.legalAuthority.acknowledged = true;
    p.expected_runner = runner;
    p.sceneworks_sha = sw;
    p.inference_sha = inf;
    if (workRoot) p.work_root = workRoot;
    if (backend === "mlx" && hfHome) p.mlx_hf_home = hfHome;
    const ids = caseIds ? caseIds.split(",").map(v => v.trim()).filter(id => id.startsWith(backend + ".")) : [];
    if (ids.length) p.case_ids = ids;
    if (priorRun) p.prior_matrix_decision = { run_id: Number(priorRun), attempt: Number(priorAttempt) };
    if (laneRun) p.prior_lane_run = { run_id: Number(laneRun), attempt: Number(laneAttempt) };
    fs.writeFileSync(out, JSON.stringify(p, null, 2) + "\n");
    const back = JSON.parse(fs.readFileSync(out, "utf8"));
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    if (back.bootstrap.legalAuthority.acknowledged !== true || back.expected_runner !== runner ||
        back.sceneworks_sha !== sw || back.inference_sha !== inf ||
        (workRoot && back.work_root !== workRoot) || (backend === "mlx" && hfHome && back.mlx_hf_home !== hfHome) ||
        (ids.length && !same(back.case_ids, ids)) ||
        (priorRun && !same(back.prior_matrix_decision, { run_id: Number(priorRun), attempt: Number(priorAttempt) })) ||
        (laneRun && !same(back.prior_lane_run, { run_id: Number(laneRun), attempt: Number(laneAttempt) }))) {
      console.error("REFUSING: stamped request did not round-trip"); process.exit(1);
    }' "$req" "$out" "$backend" "$runner" "$SW" "$INF" "$work_root" "${MLX_HF_HOME:-}" "${CASE_IDS:-}" "$PRIOR_RUN_ID" "$PRIOR_ATTEMPT" "$LANE_RUN_ID" "$LANE_ATTEMPT"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: would dispatch h3-terminal-$backend (expected_runner=$runner) with $out:"
    cat "$out"
    return
  fi
  gh workflow run server-candle-linux.yml --repo "$REPO" --ref "$REF" \
     -f starvector_terminal_operation="h3-terminal-$backend" \
     -f campaign_run_id="$(cat "$out")" >/dev/null
  echo "dispatched h3-terminal-$backend (expected_runner=$runner)"
}

# Resolve each backend's expected runner: prefer an idle compatible runner.
resolve() { # $1=backend
  if [ "$1" = "mlx" ]; then echo "$MLX_RUNNER"; return; fi
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
  [ -n "${DRY_RUN:-}" ] || sleep 15
done

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN: nothing dispatched"; exit 0; fi
if [ -n "$PRIOR_RUN_ID" ]; then
  echo "--- dispatched on the declared prior decision of run $PRIOR_RUN_ID; no anchor pairing ---"
else
  echo "--- dispatched; both runs must be launched together ---"
fi
gh run list --repo "$REPO" --workflow server-candle-linux.yml --limit 4 \
  --json databaseId,status,headSha,createdAt --jq '.[]|"\(.databaseId) \(.status) \(.headSha[0:9]) \(.createdAt)"'
