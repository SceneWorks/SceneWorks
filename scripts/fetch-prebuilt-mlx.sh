#!/usr/bin/env bash
# Fetch the prebuilt MLX archives for the mlx-rs revision pinned in Cargo.lock (sc-21382).
#
# pmetal-mlx-sys's build.rs runs a ~6 minute cmake build of libmlx (plus a network fetch of MLX)
# on every cold build. The SceneWorks/mlx-rs fork publishes that build's `build/lib` per commit
# as a GitHub Release `prebuilt-<sha12>` (see its .github/workflows/prebuilt.yml); build.rs links
# it when PMETAL_MLX_PREBUILT_DIR points at the extracted directory, after checking the
# directory's manifest against its own key. A wrong or stale directory FAILS the build, it does
# not fall back -- so this script is allowed to be dumb: pick the asset by key, verify its
# sha256, extract, print the directory.
#
# Usage:
#   scripts/fetch-prebuilt-mlx.sh [--deployment-target 26.2] [--build-type Debug|Release]
#                                 [--target aarch64-apple-darwin] [--dest DIR] [--github-env]
#
#   deployment target defaults to $MACOSX_DEPLOYMENT_TARGET, else the value in .cargo/config.toml
#   build type defaults to Debug (what `cargo build`/`cargo test` consume; --release is Release)
#   target defaults to the host triple
#   dest defaults to ${PMETAL_MLX_PREBUILT_CACHE:-$HOME/.cache/pmetal/prebuilt}/<sha12>/<cell>
#   --github-env appends both variables below to $GITHUB_ENV for later CI steps
#
# Prints `PMETAL_MLX_PREBUILT_DIR=<dir>` and `PMETAL_METALLIB_PATH=<dir>/mlx.metallib` on success
# (sc-20799: the tarball ships the Metal kernel library next to the archives, and the fork's
# runtime resolver order is $PMETAL_METALLIB_PATH -> the build's compiled-in METAL_PATH ->
# ~/.cache/pmetal/lib/mlx.metallib. `cargo test`/`cargo run` binaries have NO compiled-in path,
# and the user cache is refreshed only when pmetal-mlx-sys's build.rs re-runs -- which is exactly
# what a prebuilt, or a warm cargo cache, skips. Without the first entry those binaries die at
# "Failed to load the default metallib"). Exit codes: 1 = no release/asset for this
# rev+cell (e.g. the fork rev was just bumped and prebuilt-mlx has not published yet -- the one
# case where building from source is the right answer); 2 = usage/environment; 3 = the asset
# exists but is corrupt or is not the cell it claims to be (never tolerate that). Local use:
#
#   eval "$(scripts/fetch-prebuilt-mlx.sh)" \
#     && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH && cargo build ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_REPO="${PMETAL_MLX_PREBUILT_REPO:-SceneWorks/mlx-rs}"
FEATURES="accelerate-metal" # the crate defaults; any other feature set builds from source

deployment_target="${MACOSX_DEPLOYMENT_TARGET:-}"
build_type="Debug"
target=""
dest=""
github_env=0
while [ $# -gt 0 ]; do
  case "$1" in
    --deployment-target) deployment_target="$2"; shift 2 ;;
    --build-type) build_type="$2"; shift 2 ;;
    --target) target="$2"; shift 2 ;;
    --dest) dest="$2"; shift 2 ;;
    --github-env) github_env=1; shift ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "fetch-prebuilt-mlx: unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -z "$deployment_target" ]; then
  deployment_target="$(sed -n 's/^MACOSX_DEPLOYMENT_TARGET *= *"\([0-9.]*\)".*/\1/p' "$ROOT/.cargo/config.toml" | head -1)"
  [ -n "$deployment_target" ] || { echo "fetch-prebuilt-mlx: cannot determine the deployment target (set MACOSX_DEPLOYMENT_TARGET or pass --deployment-target)" >&2; exit 2; }
fi
case "$build_type" in Debug|Release) ;; *) echo "fetch-prebuilt-mlx: --build-type must be Debug or Release" >&2; exit 2 ;; esac
[ -n "$target" ] || target="$(rustc -vV | sed -n 's/^host: //p')"

rev="$(sed -n 's|^source = "git+https://github.com/[^/]*/mlx-rs?rev=\([0-9a-f]\{40\}\)#.*|\1|p' "$ROOT/Cargo.lock" | head -1)"
[ -n "$rev" ] || { echo "fetch-prebuilt-mlx: no mlx-rs git revision in $ROOT/Cargo.lock" >&2; exit 2; }
sha12="${rev:0:12}"

cell="${target}-dt${deployment_target}-${build_type}-${FEATURES}"
asset="pmetal-mlx-${sha12}-${cell}.tar.zst"
[ -n "$dest" ] || dest="${PMETAL_MLX_PREBUILT_CACHE:-$HOME/.cache/pmetal/prebuilt}/${sha12}/${cell}"
manifest="$dest/pmetal-mlx-prebuilt.txt"
metallib="$dest/mlx.metallib"

# Deliberately unguarded on the metallib's existence: both call sites below have already proved
# the file is there (the cache branch requires it, the extract branch verifies it and deletes the
# directory otherwise). A "skip the variable if the file is missing" guard would turn a corrupt
# prebuilt back into the silent "default metallib not found" runtime failure this emit prevents.
emit() {
  echo "PMETAL_MLX_PREBUILT_DIR=$dest"
  echo "PMETAL_METALLIB_PATH=$metallib"
  if [ "$github_env" = 1 ]; then
    [ -n "${GITHUB_ENV:-}" ] || { echo "fetch-prebuilt-mlx: --github-env but GITHUB_ENV is unset" >&2; exit 2; }
    {
      echo "PMETAL_MLX_PREBUILT_DIR=$dest"
      echo "PMETAL_METALLIB_PATH=$metallib"
    } >> "$GITHUB_ENV"
  fi
}

if [ -f "$manifest" ] && [ -f "$dest/libmlx.a" ] && [ -f "$dest/libmlxc.a" ] && [ -f "$dest/mlx.metallib" ]; then
  echo "fetch-prebuilt-mlx: using cached $dest" >&2
  emit
  exit 0
fi

command -v zstd >/dev/null || { echo "fetch-prebuilt-mlx: zstd is required to extract $asset (brew install zstd)" >&2; exit 2; }
base="https://github.com/${RELEASE_REPO}/releases/download/prebuilt-${sha12}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/pmetal-mlx-prebuilt.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
echo "fetch-prebuilt-mlx: $base/$asset" >&2
if ! curl -fsSL --retry 3 -o "$tmp/$asset" "$base/$asset" || ! curl -fsSL --retry 3 -o "$tmp/$asset.sha256" "$base/$asset.sha256"; then
  echo "fetch-prebuilt-mlx: no prebuilt for mlx-rs $sha12 cell $cell at $base (run the fork's prebuilt-mlx workflow for that commit, or build from source)" >&2
  exit 1
fi
(cd "$tmp" && shasum -a 256 -c "$asset.sha256" >&2) || { echo "fetch-prebuilt-mlx: sha256 mismatch for $asset" >&2; exit 3; }
mkdir -p "$dest"
zstd -dc "$tmp/$asset" | tar -x -C "$dest"
for f in libmlx.a libmlxc.a mlx.metallib pmetal-mlx-prebuilt.txt; do
  [ -f "$dest/$f" ] || { echo "fetch-prebuilt-mlx: $asset did not contain $f" >&2; rm -rf "$dest"; exit 3; }
done
grep -qx "deployment_target=${deployment_target}" "$manifest" || { echo "fetch-prebuilt-mlx: $asset manifest does not say deployment_target=${deployment_target}:" >&2; cat "$manifest" >&2; rm -rf "$dest"; exit 3; }
grep -qx "build_type=${build_type}" "$manifest" || { echo "fetch-prebuilt-mlx: $asset manifest does not say build_type=${build_type}" >&2; rm -rf "$dest"; exit 3; }
echo "fetch-prebuilt-mlx: extracted to $dest" >&2
emit
