# Building the native MLX Rust GPU worker (macOS)

The Rust GPU worker (`crates/sceneworks-worker`) links the [`mlx-gen`](https://github.com/michaeltrefry/mlx-gen)
engine (epic 2337) as a `cfg(target_os = "macos")` Cargo dependency and runs image
(and, later, video) generation **in-process** — no Python adapter, no sidecar venv,
no subprocess. This is the consumer side of epic 3018.

`mlx-gen` and its `mlx-rs` fork are **public**, git-pinned by SHA, so Linux and
Windows builds *resolve* them but never *compile* them (the target gate excludes
them). Only macOS compiles MLX from source.

## Requirements (macOS only)

- **macOS 26.2 or newer** at runtime for the NAX fast path (Apple matrix-unit
  kernels). The app's runtime floor is pinned at cutover (sc-3032).
- **Full Xcode + the Metal Toolchain** (`xcode-select -p` must point at Xcode, not
  the Command Line Tools; `xcrun --find metal` must resolve).
- A recent stable Rust toolchain (`rust-toolchain.toml` pins `stable`).

## The deployment-target build seam

`/.cargo/config.toml` pins `MACOSX_DEPLOYMENT_TARGET = "26.2"`. This **must** live in
the SceneWorks workspace: Cargo does not read a dependency's `.cargo/config.toml`, so
mlx-gen's own 26.2 pin does not travel to this consumer. If the pin is missing, MLX's
`mlx-sys` build.rs floors the target at macOS 14, the NAX kernels compile out
(`-DMLX_METAL_NO_NAX`), and the Mac path regresses ~2.5×. Worse, at 26.0 the 16-bit
kernels miscompile to garbage. The `nax_guard` integration test is the loud tripwire
for a slip:

```sh
cargo test -p sceneworks-worker --test nax_guard -- --nocapture   # needs macOS >= 26.2
```

The pin is **not forced**, so you (or a hosted-runner lane that cannot provide the
26.2 SDK) can override it via the environment. This only buys a correctness build —
the NAX fast path is not present:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build -p sceneworks-worker   # correctness-only, no NAX
```

> Note: `mlx-sys`'s build.rs has no `rerun-if-env-changed`, so changing the
> deployment target needs a clean rebuild of `pmetal-mlx-sys` to take effect.

## CI: the self-hosted NAX runner pool

The macOS lane (`.github/workflows/macos-mlx.yml`) runs on **self-hosted macOS
26.2+ runners**, not GitHub-hosted ones. GitHub's hosted macOS images top out well
below macOS 26.2, so they can only build at a lowered deployment target and **cannot
exercise NAX at all** — which defeats the purpose. A self-hosted 26.2+ runner builds
at the workspace 26.2 pin and actually *runs* `nax_guard`, so CI guards the NAX fast
path. It also keeps the heavy MLX build warm across runs.

The consequence is that **this lane's capacity is exactly the set of 26.2+ Macs
somebody registered by hand.** There is no elastic pool to fall back on, so a single
box makes every PR and every main commit queue behind it. Add Macs to spread the load.

### Label topology (read before adding a box)

`nax` is not exclusive to this repo's MLX lane. As of the 2026-07-25 org promotion the
self-hosted macOS runners are **org-level**, and four lanes across two repositories
schedule onto them by label:

| Label | Consumed by | Requires on the host |
| --- | --- | --- |
| `nax` | `macos-mlx.yml` (this repo), `SceneWorks/inference` `ci.yml` | macOS >= 26.2, full Xcode + Metal, Rust |
| `signing` | `release.yml` (this repo) | Developer ID cert in the **login keychain** + notary `.p8` on disk |
| `real-weights` | `SceneWorks/inference` `real-weights.yml` (22 jobs, weekly) | the full real-weight snapshot set + runner-local HF auth |

The two non-`nax` labels gate on **host state that does not travel with the label** —
a signing cert and a weights cache. A box that carries the label without the state takes
the job and fails it. So a new box gets `nax` **only**, unless you have actually
provisioned the rest.

This is why `release.yml` pins on `signing` rather than plain `nax`: it is tag-triggered
with `cancel-in-progress: false`, so scheduling it onto an unsigned box means a broken
release that needs manual recovery, after paying a full cold MLX build first.

> **Sequencing:** the `signing` pin must be merged to `main` **before** a second `nax`
> box is registered. A release uses the workflow file at the tagged commit, so until the
> pin has landed the job still matches bare `nax` and can schedule anywhere in the pool.

Adding a plain-`nax` box is nonetheless the high-value move: the weekly `real-weights`
marathon runs 22 jobs strictly one at a time and has head-of-line-blocked the `nax` PR
lane for six hours at a stretch. A second `nax` box decouples PR/main MLX work from that
marathon without needing weights parity.

### Bootstrapping Xcode on a fresh box

Do this before anything else; `setup-nax-runner.sh --check` gates on all of it. The
ordering matters — every command after the first needs Xcode installed and selected.

1. **Install Xcode** (2.4 GB download, [App Store id497799835](https://apps.apple.com/us/app/xcode/id497799835),
   or `mas install 497799835`). The Command Line Tools are **not** enough: they ship no
   Metal compiler, so MLX cannot build a single kernel against them.
2. Select it, accept the license, and run the first-launch provisioning that Xcode
   otherwise only does interactively — a launchd runner cannot answer those prompts, and
   until they are done every `xcodebuild`/`clang` call fails and every job goes red:

   ```sh
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   sudo xcodebuild -license accept
   sudo xcodebuild -runFirstLaunch
   ```

3. **Install the Metal Toolchain.** Since Xcode 26 this is an optional ~700 MB component
   that is *not* bundled with Xcode:

   ```sh
   xcodebuild -downloadComponent MetalToolchain     # or Xcode > Settings > Components
   ```

   Install it as the **same user the runner service runs as** — a toolchain installed by
   one account is not reliably visible to `xcrun` for another.

> Watch for the trap here: without the component, `metal` still *resolves* on PATH and
> only fails when invoked ("cannot execute tool 'metal' due to missing Metal Toolchain").
> `xcrun --find metal` therefore passes on a box that cannot compile anything, which is
> why the preflight runs `xcrun metal --version` instead.

### Provisioning a runner

`scripts/setup-nax-runner.sh` does the whole thing, and is safe to re-run:

```sh
scripts/setup-nax-runner.sh --check
```

That preflights every prerequisite the lane assumes but never verifies (the 26.2 floor,
Apple silicon, full Xcode + Metal, a Rust toolchain, node >= 20, disk headroom, sleep
settings, `gh` auth and admin) and changes nothing. Drop `--check` to install:

```sh
scripts/setup-nax-runner.sh
```

It downloads the pinned runner release **and verifies its SHA256** before unpacking,
registers to the `SceneWorks` org with the `nax` label under a name derived from the
machine's hostname, repairs the launchd PATH (see below), and installs and starts the
launchd service. Useful flags:

| Flag | Why |
| --- | --- |
| `--name NAME` | Runner names must be unique in scope. The default is `nax-<hostname>`; reusing an existing name **detaches that other box**. |
| `--repo OWNER/REPO` | Register at repo scope instead of the org (needs repo admin, not org admin). |
| `--group GROUP` | Org runner group. This is what bounds which repositories may schedule onto the box. |
| `--no-service` | Configure without launchd; run in the foreground with `./run.sh`. |
| `--skip-metal` | Stage a box while Xcode is still installing. It will take jobs and fail them, so do not leave it started. |

The workflow's `runs-on: [self-hosted, macOS, ARM64, nax]` matches on **labels**, so
every registered runner carrying `nax` joins the pool regardless of whether it was
registered at org or repo scope — the two can be mixed. An older self-hosted Mac
without the label never picks up the NAX-requiring job.

Runners are deliberately **not** `--ephemeral`. An ephemeral runner discards its working
directory after every job, and this lane's cost model depends on the opposite: reusing
`target/` so MLX-from-source compiles once rather than every run. That is the difference
between the healthy ~6–14 min and most of the 45-minute timeout.

### The launchd PATH trap

`svc.sh` installs a **LaunchAgent** whose plist injects no `PATH`, and `bin/runsvc.sh`
recovers `PATH` solely by sourcing a `.path` file that `config.sh` wrote from whatever
`PATH` the configuring shell happened to have. Configure from a shell without
`~/.cargo/bin` and every job dies at `cargo build` with "command not found" — while the
runner still reports healthy and idle in the Actions UI. `setup-nax-runner.sh` writes
`.path` explicitly and asserts that `cargo` and `node` resolve through it; if you
register by hand, check `cat ~/actions-runner-nax/.path`.

A LaunchAgent (user session) rather than a LaunchDaemon is correct here: Metal needs a
real logged-in GUI session. So the box must **stay logged in and never sleep**
(`sudo pmset -a sleep 0 disablesleep 1`). A sleeping runner goes offline silently and
just pushes its share of the work back onto the rest of the pool.

> **Public-repo security:** this repo is public, and GitHub warns against self-hosted
> runners on public repos because a fork PR can run arbitrary code on the runner. The
> workflow's job-level `if:` restricts execution to same-repo branches (the epic's
> story branches), and "Require approval for all external contributors' workflows"
> should stay on in the repo Actions settings. Every added Mac widens this blast
> radius, so keep org runners in a group whose repository access is scoped
> deliberately rather than left open to all repositories.

## Heavy-recompile mitigation (compile MLX once)

Building MLX from source is the slow part of a clean build, and a fresh git worktree
gets its own `target/`, recompiling MLX from scratch. To share compiled artifacts
across worktrees/clean builds, opt in via the environment (not pinned in the committed
config, so machines/CI lanes without the tool are unaffected):

```sh
brew install sccache
export RUSTC_WRAPPER=sccache
export CARGO_TARGET_DIR=~/.cache/sceneworks-target
```

## Local mlx-gen co-development

The dependency is git-pinned by SHA. To iterate against a local checkout without the
push-and-bump cycle, add a workspace-root `[patch]` (do **not** commit it — a path
that is absent on CI breaks resolution):

```toml
# Cargo.toml (workspace root), local only
[patch."https://github.com/michaeltrefry/mlx-gen"]
mlx-gen = { path = "../mlx-gen" }
mlx-gen-z-image = { path = "../mlx-gen/mlx-gen-z-image" }
```

When a co-dev change lands in mlx-gen, bump the `rev` in
`crates/sceneworks-worker/Cargo.toml` (and the matching `mlx-rs` rev in
`[dev-dependencies]`) to the new SHA and drop the patch.
