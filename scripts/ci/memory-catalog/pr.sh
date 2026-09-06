#!/usr/bin/env bash
# Open the results PR into the dispatched ref, if anything landed (sc-22738).
#
# Never merges, never enables auto-merge. `feature/*` has no required checks and is merge-commit
# only; whether these measurements are accepted is a review decision, not a CI one.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

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
URL="$(gh pr create \
  --base "$BASE_REF" \
  --head "$CAMPAIGN_BRANCH" \
  --title "chore(${CAMPAIGN}): ${BACKEND} catalog campaign results" \
  --body-file "$(native_path "$BODY_FILE")")"

echo "opened $URL"
{
  echo "#### Pull request"
  echo
  echo "$URL"
  echo
} >> "$GITHUB_STEP_SUMMARY"
