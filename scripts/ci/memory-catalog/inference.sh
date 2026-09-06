#!/usr/bin/env bash
# Put a CLEAN inference checkout at the pinned revision beside (never inside) the SceneWorks tree.
#
# PIN RESOLUTION IS DERIVED, NEVER TYPED. `measure-memory-catalog.mjs` reads the pin out of
# `crates/sceneworks-memory-adapter/src/lib.rs` (`compiledInferencePin()` -> `pub const
# INFERENCE_PIN`) and refuses the walk when the inference checkout's HEAD does not equal it. That
# constant is the one this script reads too, with the identical regex, so the two can never
# disagree: bumping the pin with `npm run bump:inference` moves both at once and this workflow needs
# no edit. Nothing here writes to any pin site.
#
# The clone lives OUTSIDE the checkout (a sibling of $GITHUB_WORKSPACE), because the harness asserts
# the SceneWorks tree stays clean for the whole walk and counts untracked paths.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

PIN="$(sed -n 's/^pub const INFERENCE_PIN: &str = "\([0-9a-f]\{40\}\)";.*/\1/p' \
  crates/sceneworks-memory-adapter/src/lib.rs | head -n 1)"
if [[ ! "$PIN" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error title=No INFERENCE_PIN::crates/sceneworks-memory-adapter/src/lib.rs declares no 40-hex INFERENCE_PIN"
  exit 1
fi
echo "pinned inference revision: $PIN"

WORKSPACE_U="$(unix_path "$GITHUB_WORKSPACE")"
REPO_U="${WORKSPACE_U}/../memory-catalog-inference"

if [[ ! -d "$REPO_U/.git" ]]; then
  rm -rf "$REPO_U"
  git clone --no-checkout https://github.com/SceneWorks/inference "$REPO_U"
fi

# Fetch every ref, not just the default branch: `anchor-loader-closure.mjs --stamp-anchors` digests
# each packaged anchor at ITS OWN record's inference revision, which is generally older than the
# pin and need not sit on any branch tip.
git -C "$REPO_U" fetch --all --tags --prune --force
if ! git -C "$REPO_U" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  git -C "$REPO_U" fetch origin "$PIN"
fi

# A reused clone can carry leftovers from an interrupted earlier run. Reset it hard: the harness
# refuses a dirty inference checkout, and a "dirty" verdict here would look like a code problem.
git -C "$REPO_U" checkout --force "$PIN"
git -C "$REPO_U" reset --hard "$PIN"
git -C "$REPO_U" clean -ffdx

HEAD_SHA="$(git -C "$REPO_U" rev-parse HEAD)"
if [[ "$HEAD_SHA" != "$PIN" ]]; then
  echo "::error title=Inference checkout is not at the pin::HEAD is $HEAD_SHA, expected $PIN"
  exit 1
fi
if [[ -n "$(git -C "$REPO_U" status --porcelain)" ]]; then
  echo "::error title=Inference checkout is dirty::the harness refuses a dirty inference tree"
  exit 1
fi

{
  echo "INFERENCE_PIN=$PIN"
  echo "INFERENCE_REPO=$(native_path "$REPO_U")"
} >> "$GITHUB_ENV"

{
  echo "Pinned inference revision \`$PIN\`, derived from \`crates/sceneworks-memory-adapter/src/lib.rs\`."
  echo
} >> "$GITHUB_STEP_SUMMARY"
