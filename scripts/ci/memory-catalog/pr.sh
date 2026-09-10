#!/usr/bin/env bash
# Open the results PR into the dispatched ref, if anything landed (sc-22738).
#
# Never merges, never enables auto-merge. `feature/*` has no required checks and is merge-commit
# only; whether these measurements are accepted is a review decision, not a CI one.
#
# THIS STEP MUST NOT RED A SALVAGED JOB (sc-22738, run 34356681566). It runs under `if: always()`
# precisely so a cancelled or timed-out walk's evidence is never stranded, and the campaign branch
# with its 20 anchor commits had already been pushed when this step turned the whole job red with
#
#   pull request create failed: GraphQL: GitHub Actions is not permitted to create or approve
#   pull requests (createPullRequest)
#
# That is the repository/organization "Allow GitHub Actions to create and approve pull requests"
# setting. No `permissions:` block can override it -- `pull-requests: write` is already granted --
# and the workflow token cannot ask for it. The branch is pushed either way, so the only thing
# missing is a click. `PR_CREATION_FORBIDDEN` below is the one failure that degrades to a warning
# plus a ready-to-click compare URL; every OTHER failure still fails the step, because a `gh` that
# cannot authenticate, a base ref that does not exist or a rejected body is a real defect and
# swallowing it would hide the loss of the PR entirely.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

PR_CREATION_FORBIDDEN="GitHub Actions is not permitted to create or approve pull requests"

if [[ -z "${CAMPAIGN_BRANCH:-}" ]]; then
  echo "no campaign branch was cut; no PR to open"
  exit 0
fi

if ! git ls-remote --exit-code --heads origin "$CAMPAIGN_BRANCH" >/dev/null 2>&1; then
  echo "${CAMPAIGN_BRANCH} was never pushed (no anchor landed); no PR to open"
  exit 0
fi

EXISTING="$(gh pr list --head "$CAMPAIGN_BRANCH" --base "$BASE_REF" --state open --json url --jq '.[0].url' || true)"
if [[ -n "$EXISTING" ]]; then
  echo "PR already open: $EXISTING"
  echo "[$CAMPAIGN_BRANCH]($EXISTING)" >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi

BODY_FILE="${WORK_DIR}/pr-body.md"
{
  echo "Measurements from the \`${BACKEND}\` memory-catalog campaign \`${CAMPAIGN}\`, one commit per anchor."
  echo
  echo "- workflow run: ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
  echo "- runner: \`${RUNNER_NAME}\`"
  echo "- pinned inference revision: \`${INFERENCE_PIN:-unknown}\`"
  echo "- cut from \`${BASE_REF}\` at \`${BASE_SHA}\`"
  echo
  echo "Each commit is one \`capture -> check -> ingest -> extract -> stamp -> matrix\` pass from"
  echo "\`scripts/measure-memory-catalog.mjs\`. The run's summary JSON and per-anchor logs are attached"
  echo "to the workflow run as an artifact."
  echo
  echo "Planned anchors:"
  echo
  echo '```'
  cat "${WORK_DIR}/plan-list.txt" 2>/dev/null || echo "(the plan table was not captured)"
  echo '```'
} > "$BODY_FILE"

# `gh` is a NATIVE Windows binary on the candle lane, so it is handed the native spelling of the
# body file rather than the `/d/a/_temp/...` form bash writes it under. Git Bash's argv mangling
# usually converts an absolute POSIX path on the way to a non-MSYS program, but that is a heuristic
# on the argument's shape -- not a guarantee -- and `native_path` is a no-op everywhere else.
#
# stderr is MERGED into stdout and captured rather than left to stream: the policy refusal above is
# only recognisable by its message, and `set -e` would otherwise end the step before it is read.
set +e
OUTPUT="$(gh pr create \
  --base "$BASE_REF" \
  --head "$CAMPAIGN_BRANCH" \
  --title "chore(${CAMPAIGN}): ${BACKEND} catalog campaign results" \
  --body-file "$(native_path "$BODY_FILE")" 2>&1)"
STATUS=$?
set -e

if [[ "$STATUS" -eq 0 ]]; then
  echo "opened $OUTPUT"
  {
    echo "#### Pull request"
    echo
    echo "$OUTPUT"
    echo
  } >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi

echo "$OUTPUT" >&2

if [[ "$OUTPUT" != *"$PR_CREATION_FORBIDDEN"* ]]; then
  echo "::error title=Results PR was not opened::gh pr create failed for ${CAMPAIGN_BRANCH} -> ${BASE_REF}; see the step log"
  exit "$STATUS"
fi

# GitHub renders `compare/<base>...<head>` with a "Create pull request" button already filled in,
# and slashes in either branch name need no escaping there.
COMPARE_URL="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/compare/${BASE_REF}...${CAMPAIGN_BRANCH}?expand=1"
echo "::warning title=Results PR must be opened by hand::this repository does not let GitHub Actions create pull requests, so the ${CAMPAIGN_BRANCH} branch was pushed but no PR was opened. Open it here: ${COMPARE_URL}"
{
  echo "#### Pull request"
  echo
  echo "The measurements are pushed to \`${CAMPAIGN_BRANCH}\`, but this repository does not permit"
  echo "GitHub Actions to create pull requests, so **the PR must be opened by hand**:"
  echo
  echo "[Open the PR: \`${BASE_REF}\` <- \`${CAMPAIGN_BRANCH}\`](${COMPARE_URL})"
  echo
  echo "The body this step would have used is attached to the run as \`pr-body.md\` in the campaign"
  echo "work-dir artifact."
  echo
} >> "$GITHUB_STEP_SUMMARY"
# EXIT 0 ON PURPOSE: the branch is safe, the evidence is uploaded, and only a click is missing.
exit 0
