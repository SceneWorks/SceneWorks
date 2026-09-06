# shellcheck shell=bash
# Shared helpers for the sc-22738 memory-catalog campaign workflow
# (.github/workflows/memory-catalog-campaign.yml). Sourced, never executed.
#
# WHY A NATIVE/UNIX PATH SPLIT. The candle lane runs these steps under Git Bash on Windows, where
# `$RUNNER_TEMP` arrives as `D:\a\_temp`. Bash builtins want a `/d/a/_temp` form; node (and the
# harness's `assertOutsideRepo`, which realpaths what it is given) wants the native form. Every
# path this campaign hands to node therefore has a `_N` (native) spelling alongside the `_U` (unix)
# spelling bash uses.

set -euo pipefail

# Native spelling of a path, for anything handed to node.
native_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

# Unix spelling of a path, for bash itself.
unix_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$1"
  else
    printf '%s' "$1"
  fi
}

# One path segment, exactly the shape `measure-memory-catalog.mjs --campaign` accepts. Validated
# here as well because it is interpolated into a branch name and a directory name.
require_campaign() {
  if [[ ! "${CAMPAIGN:-}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "::error title=Bad campaign id::campaign must be one path segment matching [A-Za-z0-9._-]+, got '${CAMPAIGN:-}'"
    exit 1
  fi
}

campaign_branch() {
  printf 'story/%s-%s-campaign-%s' "$CAMPAIGN" "$BACKEND" "$GITHUB_RUN_ID"
}

work_dir_unix() {
  printf '%s/memory-catalog-%s-%s' "$(unix_path "$RUNNER_TEMP")" "$CAMPAIGN" "$GITHUB_RUN_ID"
}

# The Hugging Face hub roots to hand the walk, one per line.
#
# Precedence: the dispatch input, then a runner-level $SCENEWORKS_MEMORY_CAMPAIGN_HF_CACHE, then
# this lane's documented default ($DEFAULT_HF_CACHE, set per job). Roots that do not exist are NOT
# an error -- `hubRoots()` already falls back to the HF env convention and the app cache, and an
# anchor whose snapshot is under none of them is reported `unavailable` by the plan rather than
# failing the run -- so a missing root only earns a warning.
hf_cache_roots() {
  local raw=""
  if [[ -n "${HF_CACHE_INPUT:-}" ]]; then
    raw="$HF_CACHE_INPUT"
  elif [[ -n "${SCENEWORKS_MEMORY_CAMPAIGN_HF_CACHE:-}" ]]; then
    raw="$SCENEWORKS_MEMORY_CAMPAIGN_HF_CACHE"
  else
    raw="${DEFAULT_HF_CACHE:-}"
  fi
  # `;` and newlines both separate; empty entries are dropped.
  printf '%s' "$raw" | tr ';' '\n' | while IFS= read -r root; do
    root="$(echo "$root" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$root" ]] && continue
    printf '%s\n' "$root"
  done
}

# Every flag the walk takes except --dry-run / --list, appended to the global array CATALOG_ARGS.
build_catalog_args() {
  CATALOG_ARGS=(
    --backend "$BACKEND"
    --campaign "$CAMPAIGN"
    --inference-repo "$INFERENCE_REPO"
    --work-dir "$WORK_DIR_NATIVE"
  )
  if [[ "${SKIP_CURRENT:-false}" == "true" ]]; then
    CATALOG_ARGS+=(--skip-current)
  fi
  if [[ -n "${ANCHORS_INPUT:-}" ]]; then
    CATALOG_ARGS+=(--anchors "$ANCHORS_INPUT")
  fi
  # `--model` is repeatable; the input is a space-separated list.
  if [[ -n "${MODELS_INPUT:-}" ]]; then
    local model
    for model in $MODELS_INPUT; do
      CATALOG_ARGS+=(--model "$model")
    done
  fi
  local root
  while IFS= read -r root; do
    [[ -z "$root" ]] && continue
    if [[ ! -d "$(unix_path "$root")" ]]; then
      echo "::warning title=Hugging Face hub root missing::'$root' is not a directory on this runner; anchors whose snapshots live there will plan as unavailable"
    fi
    CATALOG_ARGS+=(--hf-cache "$root")
  done < <(hf_cache_roots)
}
