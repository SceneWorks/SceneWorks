#!/usr/bin/env bash
# Push whatever the walk committed, on EVERY outcome (sc-22738).
#
# This step is wired with `if: always()`, so it runs after a success, a failure, a job timeout and a
# cancellation. Anchors are hours of GPU time each and the harness commits them one at a time
# precisely so a crash loses at most the anchor in flight -- that guarantee is only real if the
# commits leave the box.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

if [[ -z "${CAMPAIGN_BRANCH:-}" ]]; then
  echo "no campaign branch was cut; nothing to push"
  exit 0
fi

# The walk can be interrupted mid-anchor and leave the tree dirty. Never commit that residue: the
# harness rolls a failed post-step back itself, and anything still uncommitted here is by definition
# not a completed measurement.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "::warning title=Uncommitted residue::the checkout is dirty after the walk; it is NOT being committed"
  git status --porcelain
fi

# $BASE_SHA is the newest commit of this branch the REMOTE already has (branch.sh), so this counts
# what is NOT up there yet. On a resumed re-run that excludes the anchors an earlier attempt pushed
# and includes any it committed but failed to push.
COUNT="$(git rev-list --count "${BASE_SHA}..HEAD")"
echo "${COUNT} anchor commit(s) not yet on the remote ${CAMPAIGN_BRANCH}"
if [[ "$COUNT" == "0" ]]; then
  echo "nothing landed; not pushing an empty branch"
  exit 0
fi

git push --set-upstream --force-with-lease origin "HEAD:${CAMPAIGN_BRANCH}"

{
  echo "#### Pushed"
  echo
  echo "\`${COUNT}\` new anchor commit(s) pushed to \`${CAMPAIGN_BRANCH}\`."
  echo
} >> "$GITHUB_STEP_SUMMARY"
