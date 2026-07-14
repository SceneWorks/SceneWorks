#!/usr/bin/env bash
# sc-4482 (epic 3720): version-skew guard for the backend-neutral gen-core contract.
#
# THE TRAP: if a platform runtime resolves `sceneworks-gen-core` at revision A while the worker's
# direct contract dependency resolves revision B, Cargo silently builds both. Provider factories
# then speak revision A's traits while worker requests use revision B's nominally-similar types.
#
# This gate fails the build if more than one distinct `sceneworks-gen-core` resolution exists in
# the root package's dependency graph. It is reusable: pass the root package as $1 (default
# `sceneworks-worker`). `--target all` is REQUIRED so the macOS and CUDA runtime bundles are
# resolved even when this runs on a Linux CI lane (otherwise the skew is invisible off-macOS).
#
# Self-test: `check-gen-core-skew.sh --self-test` exercises the verdict logic on canned input
# (one-resolution => pass, two-resolutions => fail) so CI proves the gate actually fires without
# needing a deliberately-broken pin checked in.
set -euo pipefail

CRATE="sceneworks-gen-core"

# Decide from a newline-delimited list of distinct resolution lines on stdin.
# Exit 0 iff exactly one resolution; otherwise print the skew explanation and exit 1.
evaluate() {
  local pkg="$1"
  local lines=()
  local line
  while IFS= read -r line; do
    [ -n "$line" ] && lines+=("$line")
  done
  local count=${#lines[@]}

  if [ "$count" -eq 1 ]; then
    echo "OK: exactly one ${CRATE} in ${pkg}'s build graph: ${lines[0]}"
    return 0
  fi

  if [ "$count" -eq 0 ]; then
    echo "ERROR (sc-4482): ${CRATE} was not found in ${pkg}'s build graph at all." >&2
    echo "Expected the worker to depend on ${CRATE} (the backend-neutral gen-core contract)." >&2
    return 1
  fi

  {
    echo "ERROR (sc-4482 version skew): found ${count} distinct ${CRATE} resolutions in ${pkg}'s build graph:"
    printf '  %s\n' "${lines[@]}"
    cat <<'MSG'

Two gen-core revisions split the supposedly-neutral contract identity across the product and its
runtime bundle. Align `sceneworks-gen-core`, `runtime-macos`, and `runtime-cuda` to the same immutable
SceneWorks/inference release in crates/sceneworks-worker/Cargo.toml.
MSG
  } >&2
  return 1
}

self_test() {
  local rc=0
  echo "self-test: single resolution should PASS"
  if printf '%s\n' "sceneworks-gen-core v0.1.0 (git+https://example/repo?rev=AAA#AAA)" \
      | evaluate "self-test" >/dev/null; then
    echo "  ok"
  else
    echo "  FAIL: single resolution was rejected"; rc=1
  fi

  echo "self-test: two distinct resolutions should FAIL"
  if printf '%s\n%s\n' \
      "sceneworks-gen-core v0.1.0 (git+https://example/repo?rev=AAA#AAA)" \
      "sceneworks-gen-core v0.1.0 (git+https://example/repo?rev=BBB#BBB)" \
      | evaluate "self-test" >/dev/null 2>&1; then
    echo "  FAIL: skew was NOT detected"; rc=1
  else
    echo "  ok"
  fi

  echo "self-test: zero resolutions should FAIL"
  if printf '' | evaluate "self-test" >/dev/null 2>&1; then
    echo "  FAIL: missing dependency was NOT detected"; rc=1
  else
    echo "  ok"
  fi

  if [ "$rc" -eq 0 ]; then echo "self-test: PASS"; else echo "self-test: FAIL"; fi
  return "$rc"
}

# Args: [PKG] [--features <list>]. PKG defaults to sceneworks-worker. `--features` is
# passed through to `cargo tree` so the Windows candle lane (epic 5558, sc-5562) can
# resolve the optional `runtime-cuda` bundle (`--features backend-candle`) that the default check
# cannot see.
PKG=""
FEATURES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test)
      self_test
      exit $?
      ;;
    --features)
      FEATURES="${2:-}"
      shift 2
      ;;
    --features=*)
      FEATURES="${1#--features=}"
      shift
      ;;
    *)
      PKG="$1"
      shift
      ;;
  esac
done
PKG="${PKG:-sceneworks-worker}"

# Run `cargo tree`, optionally with the requested features. A comma-separated feature
# list is a single shell token, so the branch keeps quoting simple under `set -u`.
run_tree() {
  if [ -n "$FEATURES" ]; then
    cargo tree -p "$PKG" --features "$FEATURES" --target all --color never --prefix none 2>/dev/null
  else
    cargo tree -p "$PKG" --target all --color never --prefix none 2>/dev/null
  fi
}

# Flatten the tree (`--prefix none`), strip the ` (*)` dedupe marker cargo appends to repeated
# nodes, keep only the contract crate, and unique-sort. Each unique (version + source) line is one
# distinct resolution; two revs differ in the `#<rev>` source fragment.
#
# `--color never` overrides CI's `CARGO_TERM_COLOR=always` (which would otherwise wrap the ` (*)`
# marker in ANSI codes so the `(*)$` strip misses it and a deduped node looks "distinct" — a false
# skew). The ESC-strip sed is a portable backstop (works on BSD + GNU sed).
esc=$(printf '\033')
run_tree \
  | sed "s/${esc}\\[[0-9;]*m//g" \
  | sed 's/ (\*)$//' \
  | grep -E "^${CRATE} v" \
  | sort -u \
  | evaluate "$PKG"
