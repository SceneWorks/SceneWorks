# Contributing to SceneWorks

First off — thank you. SceneWorks is free, open-source software
([`AGPL-3.0-or-later`](LICENSE)), and it gets better because people like you
take the time to file a bug, improve the docs, or send a patch.

This guide covers how to propose changes. It's written to be friendly to
first-time contributors, so if anything here is unclear, that's a bug in this
document — please open an issue.

## Table of contents

- [Code of Conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Before you start on something big](#before-you-start-on-something-big)
- [Development setup](#development-setup)
- [Running the checks locally](#running-the-checks-locally)
- [Opening a pull request](#opening-a-pull-request)
- [Commit messages](#commit-messages)
- [Sign your work (DCO)](#sign-your-work-dco)
- [How your contribution is licensed](#how-your-contribution-is-licensed)
- [Reporting security issues](#reporting-security-issues)

## Code of Conduct

This project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you're expected to uphold it. Please report unacceptable behavior
to **michael@trefry.net**.

## Ways to contribute

You don't have to write code to help:

- **Report a bug** — open a [bug report](https://github.com/SceneWorks/SceneWorks/issues/new?template=bug_report.yml).
- **Request a feature** — open a [feature request](https://github.com/SceneWorks/SceneWorks/issues/new?template=feature_request.yml).
- **Improve the docs** — READMEs, `docs/`, and the per-crate `ARCHITECTURE.md`
  files are all fair game.
- **Send a fix or feature** — see the flow below.

> **Note on planning:** the maintainers track internal work in a private tracker
> using `sc-####` identifiers you'll see in commit messages. You don't need
> access to that — **GitHub Issues and Pull Requests are the front door for all
> outside contributions.** Reference GitHub issue numbers (e.g. `#123`), not
> `sc-####`, in your PRs.

## Before you start on something big

- **Small, obvious fixes** (typos, a clear bug, a docs correction) — just send a
  PR.
- **Anything larger** (a new feature, a refactor, a new model, a behavior
  change) — **please open an issue first** so we can agree on the approach
  before you invest time. SceneWorks has platform-specific engines (MLX on
  macOS, candle/CUDA on Windows) and a lot of moving parts; a quick conversation
  up front saves everyone rework.

## Development setup

SceneWorks is a Rust workspace (backend, workers, native inference) plus a
React + Vite web app, packaged as a Tauri desktop app and an optional Docker
server. Full instructions live in the
[README](README.md#development) and, for the desktop app,
[`apps/desktop/README.md`](apps/desktop/README.md).

**You'll need:**

- A **Rust** toolchain with `rustfmt` and `clippy` (pinned in
  [`rust-toolchain.toml`](rust-toolchain.toml)).
- **Node.js ≥ 20** (see [`package.json`](package.json) `engines`).
- Platform-specific, depending on what you're touching:
  - **macOS (Apple Silicon)** for the MLX GPU worker,
  - **Windows (NVIDIA)** for the candle/CUDA worker,
  - **Docker** for the headless server path.

You can contribute to a lot of the codebase (web UI, Rust API, docs, shared
crates) without a GPU. Changes to the inference engines are best validated on
the matching platform — tell us in the PR what you were able to test.

## Running the checks locally

CI runs the same commands, so running them before you push is the fastest way to
a green PR. From the repo root:

**Rust** (formatting, lints, tests, build — this is the full gate CI runs):

```bash
npm run rust:check
```

That expands to `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D
warnings`, `cargo test`, and `cargo build`. Clippy warnings **fail** the build,
so keep it clean.

**The "neither" build** — CI's `parity` lane runs the Rust clippy on **Linux with
default features (no `backend-candle`)**. The worker's generation harness
(`crates/sceneworks-worker/src/image_jobs/base.rs`) is compiled only on macOS **or** the
`backend-candle` lane, so anything used *only* by that gated code is dead on this "neither
backend" build and fails `-D warnings`. Reproduce it locally with:

```bash
npm run rust:check:neither
```

On Linux/Windows that's a native clippy — the host already *is* the neither build. On
macOS it runs the same clippy inside a Linux Docker container, because a native macOS
clippy always compiles `base.rs` and can never see the trap. It's also wired into the
optional pre-push hook (installed by `npm run hooks:install`; skip a run with
`SKIP_NEITHER_CHECK=1` or `git push --no-verify`).

> **The base.rs / candle cfg rule.** Any `sceneworks-worker` item — a `use`, struct, or
> helper fn — used **only** by `base.rs` or other candle-only code MUST carry the same cfg
> as `base.rs`, or it's dead code on the neither build:
>
> ```rust
> #[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]
> ```
>
> `npm run rust:check` on **macOS cannot catch a violation** (there `target_os` is always
> `macos`, so `base.rs` always compiles) — use `npm run rust:check:neither`, the pre-push
> hook, or the CI parity lane. This trap has bitten sc-10404 (`PhaseTimer`) and sc-8390
> (`run_blocking_with_heartbeat`).

**The "candle" build** — CI's `windows-candle` lane compiles the worker with
`--features backend-candle` on Windows, i.e. the
`all(not(target_os = "macos"), feature = "backend-candle")` configuration. That's where
`vram_gate`, `krea_control_fit`, the candle-control image modules and `generate_candle_video`
live — and **neither** `npm run rust:check` (macOS pins `target_os`, so the cfg is false
whatever features you pass) **nor** `rust:check:neither` (candle off) compiles a line of it.
It carries the most cfg-gated code in the repo. Reproduce it locally with:

```bash
npm run rust:check:candle
```

macOS hosts cross-compile to `x86_64-unknown-linux-gnu` (`rustup target add
x86_64-unknown-linux-gnu` once); Linux/Windows hosts build for the host. **No CUDA toolkit is
required**: `cudarc` and `candle-kernels` only need `nvcc`/`nvidia-smi` in their *build
scripts*, and `cargo check`/`clippy` never link or run a kernel, so the script generates a stub
`nvcc` and sets `CUDA_COMPUTE_CAP`. A real toolkit, if present, is used instead.

So the three Rust configurations each have a local command:

| command | configuration | CI lane |
| --- | --- | --- |
| `npm run rust:check` | macOS / mlx | `nax-worker` |
| `npm run rust:check:neither` | Linux, candle **off** | `parity` |
| `npm run rust:check:candle` | not-macOS, candle **on** | `windows-candle` |

> **`rust:check:candle` TYPECHECKS — it does not run the candle tests.** `cargo test` links,
> which needs a real libcuda, so candle-gated `#[test]`s still execute for the first time on
> CI. Verify their fixtures and thresholds by other means (arithmetic, or by exercising the
> shared non-gated helper on a Mac): sc-12306 shipped a candle test whose fixture asserted the
> exact opposite of its intent, and it compiled perfectly. Green here means "it compiles under
> the cfg CI compiles it under" — which is precisely the class (E0425 from a cfg-gated symbol,
> dead code under `-D warnings`) that used to cost a full round-trip on a self-hosted runner.

**Web** (from the repo root):

```bash
npm --prefix apps/web run lint
npm --prefix apps/web run test
npm --prefix apps/web run build
```

**Scaffold / compose sanity checks:**

```bash
npm run check
npm run check:compose
```

**Optional but recommended** — install the local Git hook that auto-formats Rust
before commits:

```bash
npm run hooks:install
```

If your change alters an API response shape, CI's contract/parity snapshot tests
(run via `pytest -m parity`) may need regenerating — CI will tell you, and a
maintainer can help if you're unsure.

## Opening a pull request

1. **Fork** the repo and create a branch off `main` with a descriptive name
   (e.g. `fix/video-export-crash`).
2. Make your change. Keep the PR **focused** — one logical change per PR is much
   easier to review than a grab-bag.
3. Run the [checks above](#running-the-checks-locally).
4. Push and open a PR. Fill out the PR template — it takes a minute and helps
   reviewers a lot.
5. Make sure **CI is green** and address review feedback. Maintainers may ask
   for changes; that's a normal part of the process, not a rejection.

## Commit messages

SceneWorks uses [Conventional Commits](https://www.conventionalcommits.org/).
The type and an optional scope prefix the summary:

```
feat(web): add dark-mode toggle to settings
fix(worker): guard against zero-length video export
docs: clarify LoRA training prerequisites
chore(deps): bump vite to 6.4.2
```

Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`.
Reference the GitHub issue you're closing in the PR description (e.g.
`Closes #123`).

## Sign your work (DCO)

SceneWorks uses the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) instead of a CLA. It's a lightweight, one-line certification that you wrote
the patch (or otherwise have the right to submit it under the project's license).
There's no form to sign — you just add a `Signed-off-by` line to each commit:

```bash
git commit -s -m "fix(worker): guard against zero-length video export"
```

The `-s` flag appends a line using your configured Git name and email:

```
Signed-off-by: Jane Doe <jane@example.com>
```

By signing off, you certify the four points of the
[DCO 1.1](https://developercertificate.org/). Every commit in a PR needs a
sign-off. Forgot one? `git commit --amend -s` (single commit) or an interactive
rebase / `git rebase --signoff main` (multiple) will fix it up.

## How your contribution is licensed

SceneWorks is licensed under **AGPL-3.0-or-later**, and contributions follow the
same terms (inbound = outbound). **By submitting a contribution, you agree that
it is licensed under `AGPL-3.0-or-later`**, the same license as the project, and
your DCO sign-off certifies you have the right to do so. There is **no separate
CLA** and no copyright assignment — you keep the copyright to your work.

Note that **model weights are not covered** by this license — SceneWorks
downloads third-party weights at runtime, and each keeps its own license (see
[Licensing](README.md#licensing)).

## Reporting security issues

**Please do not report security vulnerabilities through public GitHub issues.**
See [SECURITY.md](SECURITY.md) for how to report them privately.
