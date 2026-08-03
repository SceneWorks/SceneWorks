#!/usr/bin/env bash
# Provision a self-hosted macOS "nax" Actions runner for .github/workflows/macos-mlx.yml.
#
# WHY CAPACITY IS MANUAL HERE: the MLX lane is the only place in CI that compiles MLX
# from source at the workspace 26.2 deployment target and actually RUNS `nax_guard`, the
# guard for the NAX 16-bit kernels (sc-2772 / sc-3019). GitHub's hosted macOS images top
# out below macOS 26.2, so they can only build at a lowered target with the NAX kernels
# compiled out — which defeats the point. The lane's capacity is therefore exactly the
# set of 26.2+ Macs somebody registered by hand, and one box means every PR and every
# main commit queues behind it. This script adds another box to that pool.
#
# DELIBERATELY NOT `--ephemeral`. An ephemeral runner discards its working directory
# after every job, and this lane's whole cost model depends on the opposite: the runner
# reuses its `target/` so MLX-from-source compiles once instead of every run (a cold
# build is the difference between the healthy ~6-14 min and most of the 45 min timeout).
#
# THE LAUNCHD PATH TRAP, which is what separates "registered" from "actually green":
# `svc.sh` installs a LaunchAgent whose plist injects no PATH, and `bin/runsvc.sh`
# recovers PATH solely by sourcing a `.path` file that `config.sh` writes from whatever
# PATH the configuring shell happened to have. Configure from a shell without
# ~/.cargo/bin and every job dies at `cargo build` with "command not found" — while the
# runner still shows healthy/idle in the Actions UI. This script writes `.path`
# explicitly and verifies cargo/node resolve through it.
#
# A LaunchAgent (user session) is also correct rather than a LaunchDaemon: Metal work
# needs a real logged-in GUI session, so the box must be logged in and not asleep.
#
# Usage:
#   scripts/setup-nax-runner.sh --check                 # preflight only, mutates nothing
#   scripts/setup-nax-runner.sh                         # preflight, then install+register
#   scripts/setup-nax-runner.sh --name nax-studio-m3 --group nax-macos
#
# Options:
#   --check            Run preflight and exit. Safe on an already-provisioned box.
#   --name NAME        Runner name. Default: nax-<local hostname>. Must be unique in scope.
#   --org ORG          Organization to register with. Default: SceneWorks.
#   --repo OWNER/REPO  Register to a single repo instead of the org (not the default).
#   --group GROUP      Org runner group. Default: self-hosted-gpu (scoped visibility).
#   --labels a,b       Extra labels appended to `nax`. self-hosted/macOS/ARM64 are automatic.
#   --runner-dir DIR   Install root. Default: ~/actions-runner-nax.
#   --replace          Replace an existing registration with the same name.
#   --no-service       Configure but do not install/start the launchd service.
#   --skip-metal       Downgrade the full-Xcode/Metal check to a warning. See the note below.
#
# `--skip-metal` exists only so a box can be staged before Xcode finishes installing. A
# runner registered without the Metal toolchain WILL take jobs and WILL fail them, so it
# should not be left started in that state.
set -euo pipefail

# Pinned runner release. Verified by downloading the asset and hashing it, not by
# trusting the release notes alone; `verify_tarball` re-checks on every install. The
# runner self-updates after registration, so this pin only governs bootstrap — bump it
# when the published SHA below is refreshed from
# https://github.com/actions/runner/releases.
RUNNER_VERSION="2.336.0"
RUNNER_SHA256="8e8839c49b7060b6b2154f4931f815df330c27f167d53ef2239ee3dfce28b079"

# The workspace runtime floor for the NAX fast path (docs/rust-mlx-build.md). Below this
# the 16-bit kernels miscompile rather than merely compiling out, so this is a hard gate.
MIN_MACOS="26.2"
MIN_NODE_MAJOR="20"

# The OTHER half of what `nax` means, and the one a version check cannot see. MLX's
# `is_nax_available()` requires macOS >= 26.2 AND a GPU architecture generation >= 17,
# which is Apple M5 or newer. (MLX asks for 18 rather than 17 on one architecture class;
# `is_nax_available()` in the vendored MLX is the authority — the gate below is a
# deliberately conservative proxy for it, since this script cannot run MLX to ask.)
#
# Why this is a HARD gate rather than a warning: `nax_guard` has no hardware gate of its
# own (crates/sceneworks-worker/tests/nax_guard.rs — it is a pure numeric SDPA comparison
# asserting worst_16bit < 0.05). On a pre-M5 box NAX never dispatches, the fallback kernel
# is numerically fine, and the assertion goes GREEN while testing nothing about NAX. The
# single tripwire this entire lane exists for would be dead, and CI could not tell us.
MIN_CHIP_GEN="5"

# A clean MLX-from-source build plus a warm target dir. Hard floor; the lane cannot
# complete a cold build under it.
MIN_DISK_GB="40"
# The `workflow_dispatch` calibration paths provision an ~57 GiB public snapshot on top
# of that. Advisory only — the PR/main path does not need it.
CALIBRATION_DISK_GB="150"

ORG="SceneWorks"
TARGET_REPO=""
# NOT "Default". The runner group is what bounds which repositories may schedule onto
# this box, and the org's `Default` group has visibility=all — every repo in the org,
# which for a PUBLIC repo with self-hosted runners is the widest possible setting and
# undercuts the containment the job-level `if:` is also there to provide. The org's
# existing self-hosted boxes (nax-macos, cuda-windows, cuda-windows-2) all live in
# `self-hosted-gpu`, whose visibility is `selected` and scoped to SceneWorks/SceneWorks
# and SceneWorks/inference. Match that; pass --group explicitly to override.
RUNNER_GROUP="self-hosted-gpu"
EXTRA_LABELS=""
RUNNER_DIR="${HOME}/actions-runner-nax"
RUNNER_NAME=""
CHECK_ONLY=0
DO_REPLACE=0
INSTALL_SERVICE=1
SKIP_METAL=0

PREFLIGHT_FAILURES=0
PREFLIGHT_WARNINGS=0

note() { printf '  %s\n' "$*"; }
ok() { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() {
  printf '  \033[33mwarn\033[0m  %s\n' "$*" >&2
  PREFLIGHT_WARNINGS=$((PREFLIGHT_WARNINGS + 1))
}
bad() {
  printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2
  PREFLIGHT_FAILURES=$((PREFLIGHT_FAILURES + 1))
}
die() {
  printf 'setup-nax-runner: %s\n' "$*" >&2
  exit 1
}
heading() { printf '\n%s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --name) RUNNER_NAME="${2:-}"; [[ -n "$RUNNER_NAME" ]] || die "--name needs a value"; shift 2 ;;
    --org) ORG="${2:-}"; [[ -n "$ORG" ]] || die "--org needs a value"; shift 2 ;;
    --repo) TARGET_REPO="${2:-}"; [[ -n "$TARGET_REPO" ]] || die "--repo needs a value"; shift 2 ;;
    --group) RUNNER_GROUP="${2:-}"; [[ -n "$RUNNER_GROUP" ]] || die "--group needs a value"; shift 2 ;;
    --labels) EXTRA_LABELS="${2:-}"; [[ -n "$EXTRA_LABELS" ]] || die "--labels needs a value"; shift 2 ;;
    --runner-dir) RUNNER_DIR="${2:-}"; [[ -n "$RUNNER_DIR" ]] || die "--runner-dir needs a value"; shift 2 ;;
    --replace) DO_REPLACE=1; shift ;;
    --no-service) INSTALL_SERVICE=0; shift ;;
    --skip-metal) SKIP_METAL=1; shift ;;
    -h|--help) sed -n '/^# Usage:/,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -n "$TARGET_REPO" && "$TARGET_REPO" != */* ]]; then
  die "--repo must be OWNER/REPO, got '$TARGET_REPO'"
fi

if [[ -z "$RUNNER_NAME" ]]; then
  host="$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
  # Runner names allow no spaces; lowercase for consistency with the existing `nax-macos`.
  RUNNER_NAME="nax-$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9-' '-' | sed 's/-*$//')"
fi

LABELS="nax"
[[ -n "$EXTRA_LABELS" ]] && LABELS="nax,${EXTRA_LABELS}"

# Exit 0 iff $1 >= $2 as dotted versions.
version_at_least() {
  local have="$1" want="$2"
  [[ "$(printf '%s\n%s\n' "$want" "$have" | sort -V | head -n1)" == "$want" ]]
}

# ---------------------------------------------------------------------------
# Preflight. Every check the macos-mlx.yml lane assumes but never verifies, so a
# misprovisioned box fails here instead of by taking a job and reporting a red build
# that looks like a code regression.
# ---------------------------------------------------------------------------
preflight() {
  heading "Host"
  if [[ "$(uname -s)" != "Darwin" ]]; then
    bad "not macOS (uname -s = $(uname -s)); the nax lane is macOS-only"
    return
  fi
  local arch macos
  arch="$(uname -m)"
  if [[ "$arch" != "arm64" ]]; then
    bad "architecture is $arch, not arm64 — runs-on requires the ARM64 label and MLX needs Apple silicon"
  else
    ok "Apple silicon (arm64)"
  fi

  macos="$(sw_vers -productVersion)"
  if version_at_least "$macos" "$MIN_MACOS"; then
    ok "macOS ${macos} (>= ${MIN_MACOS})"
  else
    bad "macOS ${macos} is below ${MIN_MACOS}. Below the floor the NAX 16-bit kernels miscompile rather than compiling out, so this box must not carry the nax label."
  fi

  # See MIN_CHIP_GEN above: the OS floor alone does not make a box NAX-capable, and a
  # box that carries `nax` without the hardware turns nax_guard into a vacuous pass.
  local brand chip_gen
  brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  chip_gen="$(printf '%s' "$brand" | sed -nE 's/^Apple M([0-9]+).*/\1/p')"
  if [[ -z "$chip_gen" ]]; then
    bad "could not read an Apple M-series generation from '${brand:-unknown}'. NAX requires Apple M${MIN_CHIP_GEN} or newer; refusing to guess, because guessing wrong yields a green nax_guard that tests nothing."
  elif (( chip_gen < MIN_CHIP_GEN )); then
    bad "${brand} is below the Apple M${MIN_CHIP_GEN} floor required by MLX's is_nax_available() (GPU architecture generation >= 17). This box must NOT carry the nax label: it would build and test happily while nax_guard passed vacuously, leaving the NAX kernels unguarded."
  else
    ok "${brand} (>= M${MIN_CHIP_GEN}, NAX-capable)"
  fi

  heading "Toolchain"
  local dev_dir
  dev_dir="$(xcode-select -p 2>/dev/null || true)"
  if [[ -z "$dev_dir" ]]; then
    bad "xcode-select -p resolved nothing; install full Xcode and select it"
  elif [[ "$dev_dir" == */CommandLineTools ]]; then
    if (( SKIP_METAL )); then
      warn "xcode-select points at the Command Line Tools ($dev_dir). MLX cannot compile its Metal kernels — this box will fail every job until full Xcode is selected."
    else
      bad "xcode-select points at the Command Line Tools ($dev_dir), not Xcode. Install full Xcode, then: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    fi
  else
    ok "Xcode developer dir: ${dev_dir}"
  fi

  # Since Xcode 26 the Metal toolchain is an OPTIONAL ~700 MB component, not part of the
  # Xcode install. Its absence does not stop `metal` from resolving on PATH — the tool
  # stub is there and only fails when invoked ("cannot execute tool 'metal' due to
  # missing Metal Toolchain"). So `xcrun --find metal` is not a sufficient probe: it
  # passes on a box that cannot compile a single MLX kernel. Actually run the compiler.
  local metal_report
  if ! metal_report="$(xcrun metal --version 2>&1)"; then
    local metal_hint="Install it: xcodebuild -downloadComponent MetalToolchain (or Xcode > Settings > Components > Metal Toolchain). Install it as the same user the runner service runs as."
    if (( SKIP_METAL )); then
      warn "the Metal toolchain cannot compile: ${metal_report%%$'\n'*}"
    else
      bad "the Metal toolchain is not usable — MLX cannot build its kernels. ${metal_report%%$'\n'*} ${metal_hint}"
    fi
  else
    ok "Metal toolchain usable ($(printf '%s' "${metal_report%%$'\n'*}"))"
  fi

  # A non-interactive runner cannot answer a license prompt: until the agreement is
  # accepted, every xcodebuild/clang invocation fails and every job goes red.
  if [[ -n "$dev_dir" && "$dev_dir" != */CommandLineTools ]]; then
    local license_report
    if license_report="$(xcodebuild -version 2>&1)"; then
      ok "Xcode license accepted ($(printf '%s' "${license_report%%$'\n'*}"))"
    else
      bad "xcodebuild refuses to run, likely an unaccepted license: ${license_report%%$'\n'*}. Fix (needs your password): sudo xcodebuild -license accept"
    fi
  fi

  # Accept a rustup install that is not on the CURRENT shell's PATH. rustup only edits
  # interactive shell profiles, so a perfectly good toolchain is invisible to a
  # non-login shell — and to launchd. That is not a provisioning blocker, because
  # repair_runner_path puts ~/.cargo/bin into the runner's `.path` regardless; treating
  # it as one would send you chasing a toolchain that is already installed.
  local cargo_bin=""
  if command -v cargo >/dev/null 2>&1; then
    cargo_bin="$(command -v cargo)"
  elif [[ -x "${HOME}/.cargo/bin/cargo" ]]; then
    cargo_bin="${HOME}/.cargo/bin/cargo"
  fi
  if [[ -n "$cargo_bin" ]]; then
    ok "Rust $("${cargo_bin%/cargo}/rustc" --version 2>/dev/null | awk '{print $2}') at ${cargo_bin}"
    if ! command -v cargo >/dev/null 2>&1; then
      note "(not on this shell's PATH; it is written into the runner's .path so the service still finds it)"
    fi
  else
    bad "no cargo/rustc found on PATH or in ~/.cargo/bin. Install rustup (https://rustup.rs); rust-toolchain.toml pins stable."
  fi

  if command -v node >/dev/null 2>&1; then
    local node_major
    node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
    if (( node_major >= MIN_NODE_MAJOR )); then
      ok "node $(node -v) at $(command -v node)"
    else
      bad "node $(node -v) is below v${MIN_NODE_MAJOR}, which package.json requires for the calibration harness"
    fi
  else
    bad "no node on PATH; the workflow_dispatch calibration steps invoke scripts/memory-calibration-harness.mjs"
  fi

  # Only the workflow_dispatch provisioning steps use it, and they assert the exact
  # minor themselves ("python3.12 --version | grep -E '^Python 3\.12\.'"), so a missing
  # interpreter is not a PR-path blocker.
  if command -v python3.12 >/dev/null 2>&1; then
    ok "$(python3.12 --version 2>&1) at $(command -v python3.12)"
  else
    warn "no python3.12; the workflow_dispatch Qwen/Z-Image provisioning steps will fail on this box (the PR and main paths are unaffected)"
  fi

  heading "Capacity"
  local avail_gb
  avail_gb="$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -z "$avail_gb" ]]; then
    warn "could not determine free space on $HOME"
  elif (( avail_gb < MIN_DISK_GB )); then
    bad "${avail_gb} GiB free on $HOME; a cold MLX-from-source build plus a warm target dir needs at least ${MIN_DISK_GB} GiB"
  elif (( avail_gb < CALIBRATION_DISK_GB )); then
    ok "${avail_gb} GiB free on $HOME"
    note "(under ${CALIBRATION_DISK_GB} GiB: fine for PR/main runs, but the calibration dispatch pulls an ~57 GiB snapshot)"
  else
    ok "${avail_gb} GiB free on $HOME"
  fi

  # A sleeping Mac silently stops accepting jobs: the runner shows offline and work
  # queues onto the rest of the pool, which looks exactly like the overload a second box
  # was added to relieve. Read the setting explicitly rather than letting a missing
  # `sleep` line fall through to a pass.
  local sleep_setting
  sleep_setting="$(pmset -g 2>/dev/null | awk '$1=="sleep" {print $2; exit}')"
  if [[ -z "$sleep_setting" ]]; then
    warn "could not read the pmset sleep setting; confirm this Mac never sleeps"
  elif [[ "$sleep_setting" == "0" ]]; then
    ok "system sleep disabled"
  else
    warn "system sleep is set to ${sleep_setting} minutes; the runner goes offline when the Mac sleeps. Fix (needs your password): sudo pmset -a sleep 0 disablesleep 1"
  fi

  heading "GitHub access"
  if ! command -v gh >/dev/null 2>&1; then
    bad "no gh CLI. Install it (brew install gh) and authenticate: gh auth login"
    return
  fi
  ok "gh $(gh --version 2>/dev/null | head -1 | awk '{print $3}')"
  if gh auth status >/dev/null 2>&1; then
    ok "gh is authenticated"
  else
    bad "gh is not authenticated; run: gh auth login"
    return
  fi

  # Minting a registration token requires admin on the scope. Probe read access first so
  # the failure is attributable instead of surfacing as an opaque 403 mid-install.
  if [[ -n "$TARGET_REPO" ]]; then
    if gh api "repos/${TARGET_REPO}/actions/runners" >/dev/null 2>&1; then
      ok "can read runners for repo ${TARGET_REPO}"
    else
      bad "cannot read runners for ${TARGET_REPO}; repo admin is required to register a runner"
    fi
  else
    if gh api "orgs/${ORG}/actions/runners" >/dev/null 2>&1; then
      ok "can read runners for org ${ORG}"
    else
      # Note the scope split: `manage_runners:org` mints registration tokens but cannot
      # list runners, and listing is what powers the name-collision guard that protects
      # the boxes already in the pool. So this is a blocker even though registration
      # itself might well succeed.
      bad "cannot list runners for org ${ORG}, so the name-collision guard cannot run. Needs the 'admin:org' scope ('manage_runners:org' alone can register but not list): gh auth refresh -h github.com -s admin:org"
    fi
  fi
}

# Deliberately does NOT swallow API failure. A forbidden list endpoint used to fall
# through to an empty list here, which made the name-collision guard below silently
# vacuous — it reported "no collision" when it had in fact checked nothing. That is the
# one failure this script must not have: reusing an existing runner's name detaches the
# box that currently holds it.
existing_runner_names() {
  if [[ -n "$TARGET_REPO" ]]; then
    gh api "repos/${TARGET_REPO}/actions/runners" --paginate --jq '.runners[].name' 2>/dev/null
  else
    gh api "orgs/${ORG}/actions/runners" --paginate --jq '.runners[].name' 2>/dev/null
  fi
}

report_pool() {
  heading "Current nax pool"
  local any=0 line
  # `runs-on: [self-hosted, macOS, ARM64, nax]` matches on labels, so the pool for this
  # lane is every registered runner carrying `nax` — regardless of scope or name.
  while IFS=$'\t' read -r name status labels; do
    [[ -z "$name" ]] && continue
    # Count only rows that survive the `nax` filter, so a scope holding runners but no
    # nax box reports "(none visible)" rather than an empty section that reads as though
    # the pool were listed successfully.
    if [[ ",${labels}," == *",nax,"* ]]; then
      any=1
      note "$(printf '%-24s %-8s %s' "$name" "$status" "$labels")"
    fi
  done < <(
    if [[ -n "$TARGET_REPO" ]]; then
      gh api "repos/${TARGET_REPO}/actions/runners" --paginate \
        --jq '.runners[] | [.name, .status, ([.labels[].name] | join(","))] | @tsv' 2>/dev/null || true
    else
      gh api "orgs/${ORG}/actions/runners" --paginate \
        --jq '.runners[] | [.name, .status, ([.labels[].name] | join(","))] | @tsv' 2>/dev/null || true
    fi
  )
  (( any )) || note "(none visible, or insufficient permission to list)"
}

verify_tarball() {
  local tarball="$1" actual
  actual="$(shasum -a 256 "$tarball" | awk '{print $1}')"
  if [[ "$actual" != "$RUNNER_SHA256" ]]; then
    die "runner tarball checksum mismatch
  expected ${RUNNER_SHA256}
  actual   ${actual}
Refusing to unpack. If you intentionally bumped RUNNER_VERSION, update RUNNER_SHA256 from the release notes."
  fi
  ok "tarball checksum matches the pin"
}

# `config.sh` snapshots the configuring shell's PATH into `.path`, and bin/runsvc.sh
# makes that file the ONLY PATH the launchd service ever sees. Ensure the directories the
# lane actually needs are in it, and prove cargo/node resolve through it.
repair_runner_path() {
  local path_file="${RUNNER_DIR}/.path"
  # Derive the directories from where the tools ACTUALLY are rather than assuming
  # Homebrew. A hardcoded list quietly excludes real installs: node under nvm lives at
  # ~/.nvm/versions/node/<version>/bin, which is both absent from any static list and
  # version-pinned, so it would break on the next node upgrade even if hardcoded once.
  local wanted="${HOME}/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  local tool tool_path
  for tool in cargo node npm rustc; do
    tool_path="$(command -v "$tool" 2>/dev/null || true)"
    [[ -n "$tool_path" ]] && wanted="$(dirname "$tool_path"):${wanted}"
  done

  local current=""
  [[ -f "$path_file" ]] && current="$(cat "$path_file")"

  local merged="$current"
  local dir
  local saved_ifs="$IFS"
  IFS=':'
  for dir in $wanted; do
    [[ -d "$dir" ]] || continue
    if [[ ":${merged}:" != *":${dir}:"* ]]; then
      merged="${merged:+${merged}:}${dir}"
    fi
  done
  IFS="$saved_ifs"
  printf '%s' "$merged" > "$path_file"
  ok "wrote ${path_file}"

  # FAIL CLOSED. This file's header calls the launchd PATH trap "what separates
  # 'registered' from 'actually green'", so warning here and then starting the service
  # anyway would produce precisely the outcome it warns about: a runner that reports
  # healthy and idle in the Actions UI while failing every job at `cargo build`. A box
  # that cannot build is worse than no box, because its failures look like code
  # regressions and it consumes the scheduling slot either way.
  local missing=""
  for tool in cargo node; do
    if ! PATH="$merged" command -v "$tool" >/dev/null 2>&1; then
      missing="${missing:+${missing}, }${tool}"
    fi
  done
  if [[ -n "$missing" ]]; then
    die "'${missing}' does not resolve through ${path_file}.
bin/runsvc.sh gives the launchd service no PATH except this file, so the runner would come
up healthy and fail every job at the build step. Not starting the service. Install the
missing tool, or pass its directory on PATH when re-running this script."
  fi
  ok "cargo and node both resolve through .path"
}

install_runner() {
  heading "Runner package"
  local tarball="${RUNNER_DIR}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"

  # Verify whenever the archive is present, BEFORE the already-unpacked early return
  # below. Checking only on the first install made the header's "re-checks on every
  # install" a stated security property the code did not actually hold.
  if [[ -f "$tarball" ]]; then
    verify_tarball "$tarball"
  fi

  if [[ -x "${RUNNER_DIR}/config.sh" ]]; then
    ok "runner package already unpacked at ${RUNNER_DIR}"
    return
  fi

  mkdir -p "$RUNNER_DIR"
  # The runner dir holds .credentials (the runner's private key), so keep it owner-only.
  chmod 700 "$RUNNER_DIR"

  if [[ -f "$tarball" ]]; then
    note "reusing already-downloaded ${tarball##*/}"
  else
    note "downloading actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
    curl -fsSL --retry 3 --retry-delay 2 \
      -o "$tarball" \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz" \
      || die "failed to download the runner package"
  fi
  verify_tarball "$tarball"
  tar xzf "$tarball" -C "$RUNNER_DIR" || die "failed to unpack the runner package"
  ok "unpacked into ${RUNNER_DIR}"
}

configure_runner() {
  heading "Registration"

  if [[ -f "${RUNNER_DIR}/.runner" ]]; then
    local configured
    configured="$(/usr/bin/plutil -extract agentName raw -o - "${RUNNER_DIR}/.runner" 2>/dev/null \
      || node -e 'try{console.log(require(process.argv[1]).agentName||"")}catch(e){console.log("")}' "${RUNNER_DIR}/.runner" 2>/dev/null \
      || true)"
    ok "already registered as '${configured:-unknown}' — leaving the existing registration alone"
    note "to re-register: (cd ${RUNNER_DIR} && ./svc.sh stop && ./svc.sh uninstall && ./config.sh remove) then re-run this script"
    return
  fi

  local names
  if ! names="$(existing_runner_names)"; then
    die "cannot list the runners already registered in this scope, so the name-collision
guard below cannot run. Registering blind risks taking over the existing nax box's
registration and knocking it offline, which is exactly the outage this pool exists to
avoid. Grant the listing scope and re-run:

  gh auth refresh -h github.com -s admin:org

('manage_runners:org' is enough to mint a registration token but NOT to list runners, so
a token with only that scope would register successfully while leaving this guard blind.)"
  fi

  if printf '%s\n' "$names" | grep -Fxq "$RUNNER_NAME" && (( DO_REPLACE == 0 )); then
    die "a runner named '${RUNNER_NAME}' is already registered in this scope.
Runner names must be unique, and reusing one detaches the other box. Pass --name with a
distinct value (this is a pool: the existing box keeps its own name), or --replace if you
genuinely mean to take over that registration."
  fi

  # Short-lived (about one hour), single-use registration token, minted per run so no
  # long-lived credential is ever written to disk or passed around.
  local token
  if [[ -n "$TARGET_REPO" ]]; then
    token="$(gh api -X POST "repos/${TARGET_REPO}/actions/runners/registration-token" --jq .token)" \
      || die "could not mint a registration token for ${TARGET_REPO} (repo admin required)"
  else
    token="$(gh api -X POST "orgs/${ORG}/actions/runners/registration-token" --jq .token)" \
      || die "could not mint a registration token for org ${ORG} (org admin required)"
  fi
  [[ -n "$token" ]] || die "registration token came back empty"

  local url
  local -a args=()
  if [[ -n "$TARGET_REPO" ]]; then
    url="https://github.com/${TARGET_REPO}"
  else
    # A runner group only exists at org scope, and it is what bounds which repositories
    # may schedule onto this box — the public-repo blast radius the workflow's job-level
    # `if:` also guards. Repo-scope registration has no group.
    url="https://github.com/${ORG}"
    args+=(--runnergroup "$RUNNER_GROUP")
  fi
  if (( DO_REPLACE )); then
    args+=(--replace)
  fi

  note "registering '${RUNNER_NAME}' with labels ${LABELS} at ${url}"
  # `${args[@]+"${args[@]}"}` rather than a plain `"${args[@]}"`: macOS ships bash 3.2,
  # where expanding an empty array under `set -u` is an "unbound variable" error — which
  # would break the --repo path (its args array is empty) and nothing else.
  (
    cd "$RUNNER_DIR"
    # Run from a shell whose PATH already includes cargo, because config.sh snapshots
    # PATH into .path here. repair_runner_path fixes it up afterwards regardless.
    PATH="${HOME}/.cargo/bin:${PATH}" ./config.sh \
      --unattended \
      --url "$url" \
      --token "$token" \
      --name "$RUNNER_NAME" \
      --labels "$LABELS" \
      --work _work \
      ${args[@]+"${args[@]}"}
  ) || die "config.sh failed; nothing was registered"
  ok "registered as '${RUNNER_NAME}'"
}

install_service() {
  heading "launchd service"
  if [[ -f "${RUNNER_DIR}/.service" ]]; then
    ok "service already installed ($(cat "${RUNNER_DIR}/.service"))"
    # RESTART, not start. bin/runsvc.sh reads `.path` once at process start, so on an
    # already-provisioned box — the one that most needs the repair — rewriting `.path`
    # silently no-ops until the service is bounced. `stop` is tolerant: it is expected to
    # fail when the service is installed but not currently running, and that is not an
    # error worth aborting a re-run over.
    ( cd "$RUNNER_DIR" && ./svc.sh stop ) || note "(service was not running)"
  else
    ( cd "$RUNNER_DIR" && ./svc.sh install ) || die "svc.sh install failed"
    ok "service installed"
  fi
  ( cd "$RUNNER_DIR" && ./svc.sh start ) || die "svc.sh start failed"
  ok "service started"
  note "status: (cd ${RUNNER_DIR} && ./svc.sh status)"
  note "logs:   ~/Library/Logs/$(basename "$(cat "${RUNNER_DIR}/.service" 2>/dev/null || echo actions.runner)" .plist)/"
}

main() {
  printf 'setup-nax-runner: %s\n' "$(
    if [[ -n "$TARGET_REPO" ]]; then printf 'repo %s' "$TARGET_REPO"; else printf 'org %s (group: %s)' "$ORG" "$RUNNER_GROUP"; fi
  )"
  printf '  runner name: %s\n  labels:      %s (+ automatic self-hosted, macOS, ARM64)\n  install dir: %s\n' \
    "$RUNNER_NAME" "$LABELS" "$RUNNER_DIR"

  preflight

  if (( PREFLIGHT_FAILURES > 0 )); then
    heading "Preflight: ${PREFLIGHT_FAILURES} blocking problem(s), ${PREFLIGHT_WARNINGS} warning(s)"
    note "Nothing was installed or registered. Resolve the FAIL lines above and re-run."
    exit 1
  fi
  heading "Preflight: passed with ${PREFLIGHT_WARNINGS} warning(s)"

  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    report_pool
  fi

  if (( CHECK_ONLY )); then
    heading "--check requested; stopping before any changes"
    exit 0
  fi

  install_runner
  configure_runner
  repair_runner_path
  if (( INSTALL_SERVICE )); then
    install_service
  else
    heading "launchd service"
    note "--no-service: skipped. Foreground run: (cd ${RUNNER_DIR} && ./run.sh)"
  fi

  report_pool
  heading "Done"
  note "The box must stay logged in and awake — Metal needs a GUI session, and a sleeping runner just pushes work back onto the rest of the pool."
  note "Optional shared-cache speedup for cold builds: see the sccache section of docs/rust-mlx-build.md"
}

main "$@"
