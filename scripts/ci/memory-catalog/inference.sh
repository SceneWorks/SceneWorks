#!/usr/bin/env bash
# Put a CLEAN inference checkout at the pinned revision in a PERSISTENT per-runner clone (sc-22738).
#
# PIN RESOLUTION IS DERIVED, NEVER TYPED. `measure-memory-catalog.mjs` reads the pin out of
# `crates/sceneworks-memory-adapter/src/lib.rs` (`compiledInferencePin()` -> `pub const
# INFERENCE_PIN`) and refuses the walk when the inference checkout's HEAD does not equal it. That
# constant is the one this script reads too, with the identical regex, so the two can never
# disagree: bumping the pin with `npm run bump:inference` moves both at once and this workflow needs
# no edit. Nothing here writes to any pin site.
#
# WHY THIS RUNS BEFORE `cargo fetch`, AND WHY THE CLONE OUTLIVES THE JOB.
# ---------------------------------------------------------------------------------------------
# The inference repo is ~650 MB packed and the campaign used to pull it TWICE per dispatch: once by
# `cargo fetch --locked` (the workspace pins candle-kernels at a `SceneWorks/inference` rev, and
# `.cargo/config.toml` sets `git-fetch-with-cli = true`) and once here. On the Windows CUDA box that
# is a link, not a repo, problem: run 34044786385 spent 70 minutes at ~150 KB/s inside cargo's git
# fetch before dying with `fetch-pack: unexpected disconnect / early EOF`, then started a second
# 70-minute attempt. Nothing was reused, because the workspace and `$RUNNER_TEMP` are both wiped
# between jobs.
#
# So this step now runs FIRST and does the ONE network fetch the job needs, and the workflow points
# cargo's git transport at the result with a `url.<file://clone>.insteadOf` rewrite (see the
# GIT_CONFIG_* exports at the bottom). Two things make that cheap:
#
#   * PERSISTENT. The clone lives at $INFERENCE_CLONE_DIR, a fixed per-runner path OUTSIDE the
#     workspace, so the next dispatch fetches only what it is missing -- normally nothing at all.
#   * SHALLOW. `--depth 1` for each wanted commit, never a full history clone. The 2+ GiB of history
#     buys this job nothing: every consumer reads TREES at named revisions, never a log, a
#     merge-base or a describe.
#
# WHICH REVISIONS. Not just the pin. `anchor-loader-closure.mjs --stamp-anchors`, which the harness
# runs per anchor, digests each packaged anchor's loader closure AT ITS OWN record's inference
# revision (or an attested one), which is generally older than the pin and need not sit on any
# branch tip. `--anchor-revisions` prints exactly that set, derived from the same store the stamping
# walks, and needs no clone -- it IS the fetch list for a shallow clone, and deriving it from the
# store is what stops the two drifting apart. A revision missing here surfaces as a stamping failure
# hours into the walk, so the fetch of the list is a hard failure, not a warning.
#
# The clone must sit OUTSIDE the checkout: the harness asserts the SceneWorks tree stays clean for
# the whole walk and counts untracked paths.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

INFERENCE_URL="https://github.com/SceneWorks/inference"

PIN="$(sed -n 's/^pub const INFERENCE_PIN: &str = "\([0-9a-f]\{40\}\)";.*/\1/p' \
  crates/sceneworks-memory-adapter/src/lib.rs | head -n 1)"
if [[ ! "$PIN" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error title=No INFERENCE_PIN::crates/sceneworks-memory-adapter/src/lib.rs declares no 40-hex INFERENCE_PIN"
  exit 1
fi
echo "pinned inference revision: $PIN"

# WHERE THE PERSISTENT CLONE LIVES. $INFERENCE_CLONE_DIR wins (the workflow sets it per lane); the
# fallbacks are the same paths, chosen so a runner that never sets it still gets a persistent dir
# rather than a per-job one. On the Windows box that is D:, where the runner already keeps its other
# per-runner caches and where there is room for a 500 MB clone; if D: is somehow absent, $HOME is a
# correct if less conventional home for it.
if [[ -z "${INFERENCE_CLONE_DIR:-}" ]]; then
  if command -v cygpath >/dev/null 2>&1 && [[ -d /d ]]; then
    INFERENCE_CLONE_DIR="/d/memory-catalog-inference"
  else
    INFERENCE_CLONE_DIR="${HOME}/memory-catalog-inference"
  fi
fi
REPO_U="$(unix_path "$INFERENCE_CLONE_DIR")"
echo "persistent inference clone: $REPO_U"

# REUSE ONLY A CLONE THAT IS ACTUALLY THIS REPO. Anything else -- a half-written directory from a
# killed job, a clone of some other remote -- is discarded rather than fetched into, which would
# otherwise fail much later with a confusing "revision not found".
if [[ -d "$REPO_U/.git" ]] \
  && [[ "$(git -C "$REPO_U" remote get-url origin 2>/dev/null || true)" == "$INFERENCE_URL" ]]; then
  echo "reusing the existing clone"
else
  echo "no usable clone at $REPO_U; initialising one"
  rm -rf "$REPO_U"
  mkdir -p "$REPO_U"
  git init --quiet "$REPO_U"
  git -C "$REPO_U" remote add origin "$INFERENCE_URL"
fi

# The pin first, then every revision the packaged anchors cite. `--anchor-revisions` reads only
# checked-in JSON, so it runs against THIS checkout and needs no clone of its own.
WANTED=("$PIN")
while IFS= read -r rev; do
  [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || continue
  [[ "$rev" == "$PIN" ]] && continue
  WANTED+=("$rev")
done < <(node scripts/anchor-loader-closure.mjs --anchor-revisions)
echo "revisions this walk needs: ${#WANTED[@]} (pin + $(( ${#WANTED[@]} - 1 )) anchor revisions)"

# A space-separated string, not an array: the macOS runners' `bash` is 3.2, where `${#arr[@]}` on an
# EMPTY array is an unbound-variable error under `set -u` -- and "nothing is missing" is the case
# this step exists to make normal. A 40-hex SHA needs no quoting, so word splitting is exact here.
MISSING=""
MISSING_COUNT=0
for rev in "${WANTED[@]}"; do
  if ! git -C "$REPO_U" cat-file -e "${rev}^{commit}" 2>/dev/null; then
    MISSING="${MISSING}${rev} "
    MISSING_COUNT=$(( MISSING_COUNT + 1 ))
  fi
done

if (( MISSING_COUNT == 0 )); then
  echo "every needed revision is already in the persistent clone; no fetch"
else
  # ONE fetch for all of them, not one per revision: a single pack lets the server delta the ~500 MB
  # tree of each revision against its siblings, where N separate fetches would send N full trees.
  echo "fetching ${MISSING_COUNT} missing revision(s) shallowly"
  fetched=0
  for attempt in 1 2 3 4 5; do
    # shellcheck disable=SC2086  # deliberate word splitting of the SHA list
    if git -C "$REPO_U" fetch --depth 1 --no-tags origin $MISSING; then
      fetched=1
      break
    fi
    # The observed failure mode is a slow transfer the server drops mid-pack ("early EOF"), which
    # retries fine; a wrong revision would fail identically fast five times and then say so.
    echo "::warning title=inference fetch failed::attempt ${attempt}/5 did not complete; retrying"
    sleep "$(( attempt * 15 ))"
  done
  if (( fetched == 0 )); then
    echo "::error title=Could not fetch the inference revisions::5 shallow fetch attempts of ${MISSING}from $INFERENCE_URL all failed"
    exit 1
  fi
fi

# A reused clone can carry leftovers from an interrupted earlier run. Reset it hard: the harness
# refuses a dirty inference checkout, and a "dirty" verdict here would look like a code problem.
git -C "$REPO_U" checkout --force --detach "$PIN"
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

CLONE_URL="$(file_url "$REPO_U")"

{
  echo "INFERENCE_PIN=$PIN"
  echo "INFERENCE_REPO=$(native_path "$REPO_U")"
  echo "INFERENCE_CLONE_DIR=$REPO_U"
  echo "INFERENCE_CLONE_URL=$CLONE_URL"
  # POINT CARGO'S GIT TRANSPORT AT THE CLONE WE JUST FETCHED, for every later step in the job.
  #
  # These are git's own env-config channel (git >= 2.31): GIT_CONFIG_COUNT=N makes git read N
  # KEY/VALUE pairs out of the environment as if they were config. It has to be the environment
  # rather than `git config --global`, because `.cargo/config.toml`'s `git-fetch-with-cli = true`
  # path spawns git with an EMPTY HOME and GIT_CONFIG_NOSYSTEM=1 for this public repo (sc-17879) --
  # so neither the global nor the system config file is read at all, while GIT_CONFIG_COUNT still
  # is. It is exported job-wide rather than per-step so it applies identically to the bash steps and
  # to the candle lane's `shell: cmd` build.
  #
  # The rewrite is git-level, so Cargo.lock and the cargo db name still key on the github.com URL:
  # nothing about the resolved dependency changes, only where the bytes come from. Cargo warns
  # "rejected refs/commit/<pin> because shallow roots are not allowed to be updated" when it copies
  # a ref out of a shallow source; the commit itself still lands, which is why that is a warning
  # here and not a failure.
  echo "GIT_CONFIG_COUNT=1"
  echo "GIT_CONFIG_KEY_0=url.${CLONE_URL}.insteadOf"
  echo "GIT_CONFIG_VALUE_0=${INFERENCE_URL}"
} >> "$GITHUB_ENV"

{
  echo "Pinned inference revision \`$PIN\`, derived from \`crates/sceneworks-memory-adapter/src/lib.rs\`."
  echo
  echo "Persistent shallow clone \`$REPO_U\`; ${MISSING_COUNT} of ${#WANTED[@]} needed revision(s) fetched this run."
  echo
} >> "$GITHUB_STEP_SUMMARY"
