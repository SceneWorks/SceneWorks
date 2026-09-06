#!/usr/bin/env bash
# Cut the campaign branch the walk will commit onto (sc-22738).
#
# `measure-memory-catalog.mjs` commits on the CURRENT branch and refuses a detached HEAD, so the
# branch has to exist before the first anchor. It is a NEW branch per run -- never the dispatched
# ref itself -- so a campaign can never write onto `feature/*` or `main` directly, and two runs of
# the same campaign cannot collide.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

require_campaign

BRANCH="$(campaign_branch)"
BASE_SHA="$(git rev-parse HEAD)"
WORK_DIR="$(work_dir_unix)"
WORK_DIR_NATIVE="$(native_path "$WORK_DIR")"

# The identity every per-anchor commit is authored under. Local to this checkout.
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git switch -c "$BRANCH"

mkdir -p "$WORK_DIR"

{
  echo "CAMPAIGN_BRANCH=$BRANCH"
  echo "BASE_SHA=$BASE_SHA"
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
  echo "| cut from | \`$BASE_REF\` at \`$BASE_SHA\` |"
  echo "| runner | \`$RUNNER_NAME\` |"
  echo "| work dir | \`$WORK_DIR\` |"
  echo
} >> "$GITHUB_STEP_SUMMARY"
