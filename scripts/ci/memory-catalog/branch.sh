#!/usr/bin/env bash
# Cut -- or RESUME -- the campaign branch the walk will commit onto (sc-22738).
#
# `measure-memory-catalog.mjs` commits on the CURRENT branch and refuses a detached HEAD, so the
# branch has to exist before the first anchor. It is never the dispatched ref itself, so a campaign
# can never write onto `feature/*` or `main` directly.
#
# WHY A RE-RUN RESUMES THE BRANCH INSTEAD OF CUTTING A SECOND ONE (sc-22738, run 34490358777).
# ------------------------------------------------------------------------------------------------
# The branch name is keyed on $GITHUB_RUN_ID, and a re-run REUSES the run id -- `github.run_attempt`
# is the counter that moves. `git switch -c` met that head-on twelve seconds into
# `gh run rerun --failed`:
#
#   fatal: a branch named 'story/sc-22738-candle-campaign-34490358777' already exists
#
# The branch survives in the persistent self-hosted workspace because actions/checkout's `clean`
# removes FILES, not refs. So the documented recovery for a transient infra fault was itself a trap:
# the operator re-ran, waited, and got a second failure that looked like the first.
#
# Three ways to make a re-run work, and only one of them is right:
#
#   * A PER-ATTEMPT BRANCH (`...-<run_id>-<attempt>`) never collides, but it SPLITS the campaign.
#     Attempt 2 would be cut from `ref` WITHOUT attempt 1's anchors, so `--skip-current` would
#     re-measure every one of them -- hours of GPU each -- and the two branches would then collide
#     on the same matrix and anchor-store files at merge time. Rejected.
#   * DELETING the existing branch discards committed measurements. Rejected outright: the
#     commit-and-push-per-anchor design exists precisely so an interrupted walk keeps what it
#     measured.
#   * RESUMING it is what "re-run the failed job" ought to mean. The walk continues on top of the
#     anchors the earlier attempt landed, so the history stays linear at one commit per anchor with
#     nothing duplicated, the anchors already measured plan as `current` and are skipped, and the
#     pull request that is already open simply picks up the new commits.
#
# NOTHING AN EARLIER ATTEMPT MEASURED IS EVER DISCARDED, so there are three arms below rather than
# two. The remote copy wins when it exists, because a push is the only durable record and a re-run
# may be served by any of the box's runner listeners; a local branch with commits that never left
# the box is still measurements, so it is kept and this attempt's push carries them up.
#
# The arms are chosen from what EXISTS rather than from $GITHUB_RUN_ATTEMPT: the run id is in the
# branch name, so a branch under that name -- remote or local -- can only be an earlier attempt of
# THIS run, and that is a fact about the refs rather than about the counter.
#
# THE RESUME FETCH IS `--depth 1` ON PURPOSE. Only the branch TIP is needed: a shallow commit still
# carries the COMPLETE tree, which is what the plan reads to classify the earlier attempt's anchors
# as already captured and what the harness commits onto. Deepening far enough to reach the base
# commit would pull this repo's 2+ GiB history over the box's slow link -- the cost the campaign
# checkout is shallow to avoid in the first place.
#
# THE THREE SHAs THIS EXPORTS, and why they are three:
#   REF_SHA      the dispatched ref's tip in this checkout. Names the base in the PR body.
#   RESUMED_FROM the tip an earlier attempt left behind, or empty on a fresh cut. Lets the PR body
#                say which commits predate this attempt.
#   BASE_SHA     the newest commit of this branch the REMOTE already has. push.sh counts
#                `BASE_SHA..HEAD`, so that is exactly "what this branch has that the remote does
#                not" -- it neither re-counts a previous attempt's pushed anchors nor strands the
#                commits a previous attempt failed to push.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

require_campaign

BRANCH="$(campaign_branch)"
REF_SHA="$(git rev-parse HEAD)"
WORK_DIR="$(work_dir_unix)"
WORK_DIR_NATIVE="$(native_path "$WORK_DIR")"

# The identity every per-anchor commit is authored under. Local to this checkout.
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

RESUMED_FROM=""
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  # An earlier attempt PUSHED anchors, so the remote holds the authoritative copy of them.
  echo "${BRANCH} is on the remote: an earlier attempt of run ${GITHUB_RUN_ID} pushed to it"
  git fetch --no-tags --depth 1 origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
  # --force-create, because a local branch of the same name may also have survived in this
  # workspace and could be behind what was pushed.
  git switch --force-create "$BRANCH" "refs/remotes/origin/${BRANCH}"
  RESUMED_FROM="$(git rev-parse HEAD)"
  BASE_SHA="$RESUMED_FROM"
  echo "resumed ${BRANCH} at ${RESUMED_FROM}; this attempt commits on top of it"
elif git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  # An earlier attempt on THIS listener left the branch behind without pushing it -- which is the
  # failure the fatal above actually came from, since that attempt died at the build step with
  # nothing measured. Keep whatever is on it: BASE_SHA stays the ref tip, so anything it does carry
  # is counted as unpushed and goes up with this attempt's first push.
  git switch "$BRANCH"
  BASE_SHA="$REF_SHA"
  LOCAL_TIP="$(git rev-parse HEAD)"
  if [[ "$LOCAL_TIP" != "$REF_SHA" ]]; then
    RESUMED_FROM="$LOCAL_TIP"
    echo "reusing the local ${BRANCH} at ${LOCAL_TIP}; it carries commits no attempt managed to push"
  else
    echo "reusing the local ${BRANCH}, which an earlier attempt left empty at the ref tip"
  fi
else
  git switch --create "$BRANCH" "$REF_SHA"
  BASE_SHA="$REF_SHA"
fi

mkdir -p "$WORK_DIR"

{
  echo "CAMPAIGN_BRANCH=$BRANCH"
  echo "BASE_SHA=$BASE_SHA"
  echo "REF_SHA=$REF_SHA"
  echo "RESUMED_FROM=$RESUMED_FROM"
  echo "WORK_DIR=$WORK_DIR"
  echo "WORK_DIR_NATIVE=$WORK_DIR_NATIVE"
  echo "PUSHED_COUNT=0"
} >> "$GITHUB_ENV"

{
  echo "### ${CAMPAIGN} ${BACKEND} catalog campaign"
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| branch | \`$BRANCH\` |"
  echo "| cut from | \`$BASE_REF\` at \`$REF_SHA\` |"
  if [[ -n "$RESUMED_FROM" ]]; then
    echo "| resumed | run attempt \`${GITHUB_RUN_ATTEMPT:-?}\` continues the existing branch at \`$RESUMED_FROM\` |"
  fi
  echo "| runner | \`$RUNNER_NAME\` |"
  echo "| work dir | \`$WORK_DIR\` |"
  echo
} >> "$GITHUB_STEP_SUMMARY"
