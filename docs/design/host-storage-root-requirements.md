# Shared dev host root on external storage — requirements (DRAFT r5)

Status: draft for Michael's review. No CodeGraph items until approved.
- r2 (2026-11-13) folded in review round 1 (12 findings).
- r3 folded in round 2 (14 findings) and Michael's answers.
- r4 folds in round 3 (11 findings), main ones checked against `origin/main`.

## Problem

On 2026-11-13 the dev Mac's 4 TB system drive held 3.5 TB of data, after all
models had supposedly moved to the external drive `/Volumes/Models`. Most of it
was disposable runner, agent and campaign state, kept forever and in many
places:

| Where | Size (11-13) | Owner |
|---|---|---|
| Rust `target/` in agent worktrees (`~/Repos/*/.claude/worktrees/*`, `~/Repos/SceneWorks-wt/*`, `~/Repos/inference-worktrees/*`, `~/Repos/inference-wt`) | ~760 GB | Claude/Codex agents |
| Agent temp folders (`/private/tmp/claude-501/*/scratchpad`, e.g. one session's 4 parallel build lanes) | ~260 GB, growing | Claude agents |
| Agent output inside and beside `~/SceneWorks/data` (see E10) | ~200 GB | Agents and campaigns |
| `/Users/Shared/SceneWorks` (StarVector weights 83 GB, pins, source closure, Metal GPU lock) | 126 GB | StarVector workflows, Metal lock |
| Docker Desktop disk image (`Docker.raw`) | ~108 GB allocated | Docker-based CI and smokes |
| `~/actions-runner-sceneworks/_work` (checkouts, `../cargo-target`, `RUNNER_TEMP`) | 57 GB | GitHub Actions runner |
| `~/Library/Developer/Xcode/DerivedData` (68 entries), `~/.cargo/registry`, `~/.npm`, `~/Library/Caches/ms-playwright`, `~/.codex/targets`, `~/.cache/*`, `~/kv-poc`, `~/models`, `~/minimax-h3-*.safetensors` | ~20 GB measured, DerivedData not measured | Builds and workflows |

Build state also already sits loose on the external drive
(`/Volumes/Models/sw-7119`, 116 GB; `real-weights.yml` writes to
`/Volumes/Models/Codex-builds/...`).

The workflows, scripts and tools contain about 430 machine-path matches
across about 134 files in SceneWorks and inference: 191 matches in 67
SceneWorks files and 239 in 67 inference files. DevWorks has 21 more,
LensWorks 3 and ChatWorks none. Also:
- About 87 SceneWorks and 16 inference sites assume build output in
  `<repo>/target`.
- About 165 sites set `CARGO_TARGET_DIR` themselves.
- Several scripts call cargo by absolute path or through `rustup run`.

Nothing gives these a shared root or a variable, and nothing cleans them up.

`/Users/Shared/SceneWorks` came from the StarVector campaign (sc-22261,
2026-08-29/30) without a recorded reason. The Metal GPU lock was later put
there too. On Windows the CUDA locks live in
`%PUBLIC%\SceneWorks\gpu-locks` and `service-locks`.

## Scope

**Repos:** SceneWorks, inference, DevWorks, ChatWorks, LensWorks. No
LensWorks runner exists any more. A stale, unloaded
`actions.runner.michaeltrefry-LensWorks.lensworks-macos.plist` gets removed.

**Hosts:**

| Host | Root (`DEV_HOST_ROOT`) | Budgets (E9) |
|---|---|---|
| This Mac (runner `nax-macos`; labels incl. `rw-starvector`, `nax`, `signing`) and agent sessions on it | `/Volumes/Work/dev-host`, an SMB share on the Tower NAS, shared with `nax-macos-2` | full, for this host's own folders |
| `nax-macos-2` (labels incl. `rw-mage`, `rw-llm`) | the same `/Volumes/Work/dev-host` share | full, for this host's own folders |
| Windows CUDA runners (`[self-hosted, Windows, X64, cuda]`) | `D:\dev-host` | full, unless `D:` is smaller |
| GitHub-hosted runners (`ubuntu-latest`, `macos-*`, `windows-*`) | none: disposable, out of scope | — |

**Amendment r5 (2026-11-22).** Michael moved the Macs from external USB
drives to the Tower NAS: both Macs mount the same SMB share at
`/Volumes/Work` and use one root, `/Volumes/Work/dev-host`. `/Volumes/Models` will not come back as an
external drive; it is now the NAS's `Models` share, mounted on both Macs,
and holds the HF hub cache. Michael measured no build-speed loss against the NAS. One
root shared by two machines changes E2, E3, E4, E5, E8, E9 and E13 below:
anything one host writes and decides about alone (locks, build slots,
scratch, campaign work, Docker, the runner folder) moves under a per-host
folder, `hosts/<host>/`, and each host's cleanup only ever deletes inside its
own folder. The Windows box keeps its local `D:\dev-host` with the same
layout, so all hosts run the same code.

## Goal

On every in-scope host, all runner, agent and campaign state lives under one
root on dedicated storage. That covers builds, caches, test weights, scratch,
temp files, campaign state and GPU locks. Every tool finds it the same way.
The system drive holds only source code, apps, toolchains, personal files
(including the user's own app data) and the `dev-host` tool itself.

## Requirements

- **E1 One root, one variable, set by the host.**
  - Each in-scope host sets `DEV_HOST_ROOT` in its runner `.env` and in the
    agent harness environment (E4).
  - Every workflow step and script that runs on a self-hosted host reads
    machine-local state locations from `DEV_HOST_ROOT`. It never hardcodes
    `/Users/…`, `$HOME/…`, `~/…`, `/tmp/…`, `/Volumes/…`, `%PUBLIC%`,
    `%USERPROFILE%`, `C:\…` or `E:\…` for that state.
  - A self-hosted job fails when `DEV_HOST_ROOT` is unset; it never falls
    back to a default. On hosted runners, steps that would need it don't
    run.
  - **Exceptions:** GPU and service lock paths (E3), and short-lived Unix
    socket rendezvous folders under `/tmp` (E13).
- **E2 Fixed layout under the root.**
  - **Shared by every host using the root:** `weights/`, `cache/` and
    `campaigns/<project>/<id>/evidence/`. Writers to these never rewrite a
    file in place: they write a temporary name and rename it, so two hosts
    can't corrupt one file. `cache/` therefore holds only caches whose
    entries are written whole and then published by rename (downloaded
    artifacts, pmetal, torch hub files). A cache that a tool keeps updating
    in place, such as Xcode DerivedData, the cargo registry and git
    checkouts, Playwright browsers or the docker-smoke state, is per host,
    under `hosts/<host>/cache/<project>/`, because two Macs building at
    once would modify the same build database (r5).
  - **Per host:** everything else lives under `hosts/<host>/`, where
    `<host>` is the machine's local host name as the OS reports it
    (`scutil --get LocalHostName` on macOS, `COMPUTERNAME` read from the OS,
    not the environment, on Windows). `dev-host` never takes it from an
    overridable variable. The folders listed below as `locks/`, `build/`,
    `t/`, `campaigns/<project>/<id>/work/`, `runner/` and `docker/` are all
    under `hosts/<host>/` (r5).
  - `locks/` holds GPU, service and other cross-process locks. It is
    per machine, not per project.
  - `cache/<project>/` (shared) holds non-model caches written whole and
    published by rename: pmetal, torch. `hosts/<host>/cache/<project>/`
    holds the ones a tool updates in place: docker-smoke, cargo registry/git
    for CI, Playwright browsers, Xcode DerivedData (r5).
  - `weights/<project>/` holds model weights that tests and campaigns load
    and that aren't in the HF hub. It includes a manifest naming each set,
    its source and the workflows that use it. HF weights stay in the HF
    hub cache at `/Volumes/Models/huggingface/hub`, now on the NAS's
    `Models` share and shared by both Macs, and are referenced, not copied
    (r5).
  - `build/<project>/<build-key>/slot-<n>/` holds Rust build output (E4).
  - `t/<short-id>/` holds per-run and per-session scratch and `TMPDIR`
    (E13). The path is kept short on purpose.
  - `campaigns/<project>/<id>/` holds `work/`, which is disposable, and
    `evidence/`, which is retained.
  - `runner/<runner-name>/` holds the GitHub Actions runner's work folder.
  - `docker/` holds Docker Desktop's disk image (macOS only).
- **E3 GPU and service locks: fixed path, safe transition.**
  - **Fixed path.** Each lock path is a fixed constant per host class,
    never derived from `DEV_HOST_ROOT`, `%PUBLIC%` or any other overridable
    variable:
    - Macs: `/Volumes/Work/dev-host/hosts/<host>/locks/apple-metal.lock`
      (r5). `<host>` comes from the OS as in E2, so each Mac locks only its
      own GPU and the two never block each other.
    - Windows: `D:\dev-host\locks\gpu\<UUID>.lock` and
      `D:\dev-host\locks\service\…`, moving under
      `D:\dev-host\hosts\<host>\` with the r5 layout.
  - **No mkdir on a missing volume.** Before opening a lock, the helper
    runs the E5 check. It never creates any folder above `locks/`. Today's
    `mkdir(parents=True)` calls in `metal_execution_lock.py` and
    `windows_cuda_execution.py` (GPU at :206 and service at :348) go away.
  - **Transition order.** Updated holders take the old lock first, then the
    new one, always in that order. Old-only holders are still excluded,
    and fixed ordering can't deadlock.
  - **Mac lease handoff.** A child process borrows its parent's lock through
    an inherited descriptor.
    - `SCENEWORKS_METAL_EXECUTION_LOCK_FD` keeps carrying the old lock;
      `SCENEWORKS_METAL_EXECUTION_LOCK_FD_V2` carries the new one.
    - A borrower whose V2 descriptor is missing or fails validation ignores
      it. That covers an old helper earlier in the chain dropping or
      un-mapping it. The borrower then takes the new lock itself, which is
      safe because it already holds the old lock via V1.
    - An old-only parent with a new child: the child validates V1, then
      takes the new lock itself.
    - The lease record (`METAL_EXECUTION_LEASE`) gets a v2 protocol that
      carries both paths. The v1 record stays, unchanged, for old borrowers.
    - The ownership marker in `apple-metal-lease.mjs` (shared file offset)
      is defined for each lock separately.
  - **Windows.** There is no descriptor handoff. Holders take the old then
    the new `msvcrt` byte-0 lock in fixed order. Service locks follow the
    same rule.
  - **Every lock user is updated together.** The plan's inventory step
    re-runs the search so none are missed. Known users:
    - **Mac helpers:** `metal_execution_lock.py` and
      `apple-metal-execution-lock.mjs` (both repos where present, including
      the SHA-256 pin in `apple-metal-execution-lock.test.mjs`),
      `metal_execution_fds.py`, `metal_execution_node_bridge.py`,
      `metal_execution_entry.sh`, `apple-metal-lease.mjs`.
    - **Mac borrowers:** the iris and supra2 `run.sh`; inference
      `model_proof_resources.py:504` and `model_inference_proofs.py:1273`.
    - **Tests that pin the path:** `test_metal_execution_lock.py` (both
      repos).
    - **Windows, SceneWorks:** `windows_cuda_execution.py:206,348`,
      `vendor/veda_cuda_guard.py:129`,
      `moss-training-acceptance/bounded_product_claim.py:80`,
      `imported-nvfp4-cuda-guard.py:115` (evidence string).
    - **Windows, inference:** `model_proof_resources.py:532`,
      `model_proof_cuda_primitives.py:72`, `model_proof_multi_cuda.py:120`,
      `scripts/ci/vendor/veda_cuda_guard.py:150`,
      `release/veda_cuda_guard.py`, `release/veda_s3_cuda.py:135`.
  - **Retiring the old locks** (amended twice on 2026-11-13). They retire
    once every one of these refs contains the dual-lock helper, so no
    old-only holder remains. The check covers both repos together, because
    the Mac lock is machine-wide:
    - `main` and `release/next`
    - every `feature/*` branch with a commit in the last 14 days
    - the inference revision pinned by each of those

    `release/next` gets the dual lock by hotfix. Published
    `release/X.Y.Z` branches are immutable, so they keep the old path; a
    hotfix cut from one applies the dual-lock change first.
  - The existing symlink and hard-link refusals stay.
- **E4 Build output: keyed, sticky per worktree, set by environment.**
  - **The `dev-host` tool** decides where every Rust build goes. For a
    worktree (or CI checkout) it assigns:
    - **a build key:** project, Rust toolchain, target triple, and a hash
      of the build-relevant variables cargo doesn't track (at least
      `MACOSX_DEPLOYMENT_TARGET`, the Xcode strip variables and the MLX
      build flags). Builds with different keys never share a folder.
      `mlx-sys`'s `build.rs` has no `rerun-if-env-changed`, so sharing
      would silently drop the NAX Metal kernels.
    - **a sticky slot:** one slot per worktree path and key, recorded under
      `hosts/<host>/build/` and kept until E8 reclaims it. Two hosts never
      share a slot or a slot registry, even for the same worktree path (r5). The worktree always builds
      incrementally in the same slot, and no two worktrees ever share a
      slot, so one can't overwrite another's binaries. Slots are capped per
      key. When none is free, the least recently used idle slot is
      reassigned.
  - **Set by environment, not by PATH order.** The tool's job is to export
    `CARGO_TARGET_DIR` (and `TMPDIR`, E13). It does this at:
    - **CI:** a setup step at job start, run after any toolchain action, so
      the result lands in `GITHUB_ENV`.
    - **Claude Code:** a SessionStart hook in each repo's
      `.claude/settings.json`.
    - **Codex:** its session config.
    - **Michael's shells:** a `cd` hook. Optional, never required.

    Because it's an environment variable, every cargo launch inherits it:
    absolute-path cargo (`h3_admission_mutation_proof.py:217`,
    `ios_prequant.py:378`, `ios_pricing_compile_only.py:93,255`),
    `rustup run` (`h3-finetune-terminal.yml:722-723`), Tauri, nested
    build-script cargo, and Windows `prepare-rust-runner`.
  - **A `cargo` shim is a backstop only.** It sets the same variable when
    the environment didn't.
  - **Caller-set `CARGO_TARGET_DIR`.** A value the caller already set wins
    only if it lies under `DEV_HOST_ROOT`; otherwise the tool fails. The
    about 165 existing sites (e.g. `real-weights.yml:392`,
    `moss_edit_metal.py:76`) are repointed in the migration.
  - **Consumers find the folder, never assume it.** Every script, workflow
    and tool that reads build output gets the folder from
    `cargo metadata --format-version 1` `.target_directory`, which follows
    `CARGO_TARGET_DIR`. None assumes `<repo>/target`. This covers the Tauri
    sidecar build (`apps/desktop/scripts/build-sidecar.mjs:63,89,276,380`),
    the signing job in `release.yml:166,193–203`, `check.yml:232-233`, the
    film-harness and epic-2123 scripts, and `check-scaffold.mjs:96-146`.
    The plan starts with a full inventory.
  - **CI's warm build.** `macos-mlx.yml`'s `../cargo-target` is kept warm on
    purpose; a cold PR-gate build costs about 27 min. It becomes that
    checkout's sticky slot, marked `warm`. Least-recently-used reassignment
    and eviction (E9) take warm slots last; the 7-day idle rule (E8) still
    applies.
- **E5 Fail clearly when the storage is missing.**
  - **Where the tool lives.** The `dev-host` executable is installed on the
    system drive (`~/.local/bin` on macOS,
    `%LOCALAPPDATA%\dev-host\bin` on Windows), not under the root. If the
    root is missing, it is still there to refuse loudly, instead of a
    missing tool silently letting cargo fall back to `<repo>/target`.
  - **`dev-host check`** verifies three things:
    - `DEV_HOST_ROOT` is set.
    - The root sits on the intended volume. On macOS its device ID differs
      from `/`'s; on Windows `D:` is a mounted fixed volume.
    - The `.dev-host-root` marker, written once at setup, is present.
    - **Every other share the host needs is mounted (r5).** The root's
      `dev-host.conf` lists them, one `required_mount = <path>` line each.
      On the Macs that is `/Volumes/Models`, which holds the HF hub cache.
      For each one, the check verifies that the path is a mount point
      whose device differs from `/`'s. Otherwise a dropped `Models` share
      would pass the check while `Work` stays up, and jobs would fail
      later, at model loading, or write into an empty local folder. The
      paths live in the config, not in `dev-host`'s source (E14).

    If any check fails, it exits non-zero within 10 s, naming the missing
    mount. Nothing creates the root or any folder above a verified root.
  - **Who runs it:**
    - The runner pre-job hook (`ACTIONS_RUNNER_HOOK_JOB_STARTED`, set in the
      runner `.env`). It runs before checkout, and a non-zero exit fails
      the job. The runner may already have downloaded actions into
      `_work`, which is on the root by then (E6).
    - The runner launcher. On this Mac that's
      `~/Library/LaunchAgents/com.sceneworks.actions-runner.plist`
      (`KeepAlive`, runs `run.sh`); it becomes `dev-host check && exec
      run.sh`. Runner self-update runs inside `run-helper.sh`, so it isn't
      affected. While the volume is missing, launchd retries every 10 s,
      which is accepted.
    - Not the Windows runner services. `D:` on the CUDA machine is an
      internal NVMe drive that is never missing (Michael, 2026-11-15), so
      the pre-job hook alone covers Windows: it still catches an unset
      `DEV_HOST_ROOT` or a missing marker.
    - The SessionStart hook for agents, the `cargo` shim, and the lock
      helpers (E3).
  - Docker Desktop on macOS starts only after the check passes. It's
    launched by hand today, so this is a documented rule rather than code.
- **E6 Runner, Docker and Xcode state on the root.**
  - Each in-scope runner's work folder moves to
    `runner/<runner-name>/_work`.
  - Docker Desktop's disk image location moves to `docker/`.
  - Xcode DerivedData for CI and agent builds goes to
    `hosts/<host>/cache/<project>/DerivedData` (per host, r5), via
    `-derivedDataPath` or the Xcode setting.
- **E7 Guard against new hardcoded paths.**
  - A CI check in each in-scope repo looks only at lines **added** in the
    PR diff of workflows, scripts and tools.
  - It fails on a new path from E1's list outside `DEV_HOST_ROOT`.
  - It fails on a new reference to `~/SceneWorks` or
    `~/Library/Application Support/SceneWorks` (E10).
  - It fails on a new `<repo>/target` assumption, a new absolute-path cargo
    launch, or a new `CARGO_TARGET_DIR` outside the root (E4).
  - **Exceptions:** a line carrying an explicit `dev-host: allow-path
    <reason>` comment, comments and docs, the E1 exceptions, and a short
    allowlist of system tools (e.g. `/opt/homebrew/bin`).
  - It runs on every PR.
- **E8 Automatic cleanup with liveness checks.**
  - **Own folder only (r5).** A host's cleanup deletes only inside
    `hosts/<host>/`. It can't see another host's processes, so it never
    deletes anything another host might be using. Shared folders
    (`weights/`, `cache/`, campaign evidence) are only measured and reported.
  - A scheduled job on each in-scope host:
    - deletes `build/` slots whose stamp file (touched by `dev-host` at
      every assignment) is older than 7 days
    - deletes `t/` entries and `campaigns/*/*/work/` whose newest file is
      older than 7 days, using a deep scan rather than directory mtime
  - It keeps `campaigns/*/*/evidence/`, `weights/` and `locks/`.
  - It never deletes a folder whose cargo lock is held, or that a live
    process has open or as its working directory.
  - It reports what it removed, and the root's size per top-level folder.
- **E9 Size budget.**
  - Budgets for this Mac:
    - `build/` ≤ 300 GB
    - `t/` ≤ 100 GB
    - `cache/` ≤ 50 GB
    - `campaigns/*/work` ≤ 100 GB
    - `docker/` ≤ 120 GB
  - These budgets are per host: each Mac gets them for its own
    `hosts/<host>/` folder, so the two Macs together may use twice the
    `build/`, `t/`, campaign-work and `docker/` figures on the NAS (r5).
    `cache/` (50 GB) is shared and counted once. Windows uses the Mac
    figures without `docker/`, scaled down if `D:` is smaller.
  - `weights/` is sized by its manifest.
  - The cleanup job reports any folder over budget. In `build/` and `t/` it
    deletes the least recently used idle entries until under budget, using
    the E8 liveness rules, and warm CI slots go last. It never fails or
    interrupts a running build.
- **E10 User app data is off limits.**
  - `~/SceneWorks/data` is Michael's own SceneWorks application data (his
    `SCENEWORKS_DATA_DIR`). The app's macOS default is
    `~/Library/Application Support/SceneWorks/data`.
  - Runners, agents, tests, smokes and campaigns on in-scope hosts never
    read or write either folder, or anything else under `~/SceneWorks`.
  - Anything that needs a SceneWorks data, config or cache folder gets one
    under `t/` by setting `SCENEWORKS_DATA_DIR`, `SCENEWORKS_CONFIG_DIR`
    and the cache override explicitly.
  - The GitHub-hosted `h3-admission-proof` job in `server-candle-linux.yml`
    masks `$HOME/SceneWorks/data` with a tmpfs. That mask protects the
    folder, so it stays.
- **E11 Migration with no lost state.**
  - **Order, per host, keeping CI green throughout:**
    1. Set up the root and the marker.
    2. Install `dev-host` (E14).
    3. Add the dual locks (E3).
    4. Repoint workflows, scripts and consumers (E1, E4).
    5. Move the runners.
    6. Move data.
    7. Delete the old copies.

    `nax-macos-2` does steps 1–2 on the NAS share before any workflow
    cut-over on main (step 4), so its jobs never meet an unset
    `DEV_HOST_ROOT` (r5: no new drive is needed).
  - **Data that moves:**
    - test and campaign weights, including StarVector's
    - evidence
    - `~/kv-poc`, `~/models` and the reference files `~/minimax-h3-*.safetensors`
    - `~/.cache/sceneworks-*`
    - the agent output in and beside `~/SceneWorks/data`:
      - inside `data/`: `sc-19120-scratch`
      - beside `data/`:
        - `qwen-image-2-1-*tiers*` and `diagnostics`
        - `calib-sc-24114*`, `cuda-evidence-*`, `render-validation*` and
          `render-verify-*`
        - `film-harness-evidence`, `gpu-guards`, `qwen21-edit-refs` and
          `storage-audits`
    - the non-build contents of `/Volumes/Models/sw-7119` and
      `/Volumes/Models/Codex-builds`

    Each is verified by content hash or file count before the source is
    deleted. Evidence goes to `campaigns/<project>/<id>/evidence`.
  - **Rebuilt, not moved:** build folders, runner `_work`, Python
    virtualenvs and anything else with absolute paths baked in. The old
    copies are deleted after the first successful rebuild.
  - The old lock files stay until E3 retires them.
  - References to folders that no longer exist are removed or repointed.
- **E12 Documented.** `AGENTS.md` and `FEATURE_DEVELOPMENT.md`, or the
  equivalent, in each in-scope repo state the following, so new agents use
  the root from the start:
  - the root, the layout and the variable
  - the `dev-host` tool
  - the allow-path comment
  - the rule never to search from `/` or `$HOME`, which triggers iCloud
    downloads on macOS
- **E13 Temp files under the root, with short paths.**
  - `TMPDIR` (macOS) and `TEMP`/`TMP` (Windows) point at
    `$DEV_HOST_ROOT/hosts/<host>/t/<short-id>/`. The short id is 8
    characters, so the macOS path stays well under the socket limit.
  - **SMB caveat (r5).** On the NAS, scratch files sit on SMB. dev-host's
    own suite failed 4 of 75 tests with `TMPDIR` on the share (same-process
    lock visibility, one cleanup test, two install-staging tests) and passed
    all 75 with local scratch (2026-11-20). Whether SceneWorks and inference
    tests pass with `TMPDIR` on SMB is an open question; see below.
  - `dev-host` sets them at job and session start (E4); a per-run id
    can't live in the runner `.env`.
  - **Socket exception.** Tools that need a Unix socket path under macOS's
    104-byte limit may still use a short-lived, mode-0700 folder under
    `/tmp`, as `memory-calibration-watchdog.py:917-923` already does. Such
    folders hold only sockets and tiny rendezvous files, and are removed
    when the process exits.
- **E14 `dev-host` is built, versioned and installed like a product.**
  - **Source.** It lives in DevWorks (Michael's decision), as a separate
    crate with no host names, paths or runner names in the source. Each
    host's values come from its environment and a local config file under
    the root.
  - **Releases.** DevWorks gains a tag-triggered release that publishes
    binaries for macOS arm64 and Windows x64.
  - **Install.** Each in-scope repo pins a `dev-host` version. Its CI setup
    step installs that version, or checks that it is already installed,
    and fails on a mismatch.
  - **Bootstrap.** `dev-host` can be built directly with the real cargo
    (`cargo build -p dev-host` with the wrapper bypassed), so it never
    depends on itself to build.

- **E15 Only trusted hosts can write the shared storage (r5).**
  - A network share that holds anything a job runs, builds or loads
    (checkouts, build output, caches, weights, scratch) must only be
    writable by authenticated accounts of the in-scope hosts. Guest or
    anonymous write access is not allowed. Each Mac mounts `/Volumes/Work`
    and `/Volumes/Models` with its own authenticated account.
  - It must hold before MERGE, the first point where workflows on main
    write build or cache state under the root: HOST-1 and HOST-3 verify the
    authenticated mounts, and they block MERGE. Until then, nothing moves
    the signing runner's (`nax-macos`, label `signing`) work folder, build
    slots or caches onto the share, and the integration branches don't run
    `release.yml`. `release.yml` runs repository
    npm code and builds while the Apple and Tauri signing secrets are set,
    so a writable-by-anyone checkout or build folder would let any machine
    on the network get its own code signed and notarized.
  - **Signing never uses state on the share.** Authenticated accounts stop
    strangers, but `nax-macos-2`'s account can still write the whole
    share, including `hosts/nax-macos/` and the shared caches, and the
    NAS may not support per-folder permissions. So `release.yml`'s signing
    and notarizing jobs build in a fresh folder on the runner's local disk,
    with no root build slot, no shared or per-host cache from the share and
    no `TMPDIR` on it. `dev-host env` gets an option that leaves these
    jobs' build and temp paths local. Weights and HF models aren't loaded
    by signing jobs.
  - Setting up the accounts and share permissions on Tower is Michael's
    step (credentials). The check is a test: a guest mount of each share
    can't create a file.

## Feature acceptance tests

1. **Nothing lands on the system drive.** On this Mac, run one full CI pass
   of SceneWorks and inference plus one agent worktree build per repo.
   Record files written outside the root with `fs_usage`.
   - Allowed: toolchain installs, logs, the source checkout, Claude Code's
     own small temp files, the installed `dev-host` binary, and E13
     socket folders under `/tmp`.
   - Not allowed: build output, weights, caches or scratch.
2. **Missing storage fails before anything is written.** Run with the
   NAS share unmounted, and again with a plain folder created at
   `/Volumes/Work` (that needs admin rights; Michael runs this part):
   - the runner launcher refuses to start;
   - a queued job fails in the pre-job hook within 10 s, naming the mount;
   - `dev-host check`, the `cargo` shim and the lock helper all fail the
     same way;
   - nothing is created under the plain folder or in `<repo>/target`.

   Repeat on a Mac with only `/Volumes/Models` unmounted and `/Volumes/Work`
   still up: the same checks fail, naming `/Volumes/Models` (r5).

   On Windows, where `D:` is internal and never missing, rename the
   marker instead: the pre-job hook and the lock helper fail naming the
   marker.
3. **Locks stay exclusive across the transition.** Using the CPU-only test
   seams of each implementation:
   - on Mac and Windows, an old-only holder and a dual holder never hold
     the lock at the same time;
   - on Mac, each of these chains succeeds without deadlock:
     - new parent → old-only child
     - new parent → new child
     - new parent → old helper → new grandchild
     - old-only parent → new child
4. **Build output lands where it should.**
   - Two agent worktrees with the same key get two different sticky slots
     under `build/<project>/<key>/`, and each reuses its own slot on the
     next build.
   - Changing `MACOSX_DEPLOYMENT_TARGET` gives a new key.
   - No worktree `target/` and nothing in `/private/tmp` gets build output
     when cargo is launched by `PATH`, by absolute path, through
     `rustup run`, or by the Tauri CLI.
   - The Tauri sidecar build and the `release.yml` signing job still find
     their outputs.
5. **The guard catches new paths only.**
   - It fails a PR that adds `$HOME/foo`, `E:\foo`, `~/SceneWorks/x`, a
     `./target/release` assumption, or an absolute-path cargo launch.
   - It passes a PR whose test line asserts an existing path with an
     allow-path comment.
   - The guard's pattern tests live once, in `dev-host` (amended
     2026-11-13). Each repo shows its CI runs the guard with one fixture
     PR that adds a forbidden path.
6. **Migration is complete.**
   - A one-time full-tree scan (the guard's matcher over each in-scope
     repo) finds no unannotated hardcoded state paths.
   - These are gone: `/Users/Shared/SceneWorks` (except the old lock until
     it is retired), `~/kv-poc`, `~/models`, `~/minimax-h3-*.safetensors`,
     the agent output in and beside `~/SceneWorks/data`, the Docker disk
     image and DerivedData on the system drive, and the old runner
     `_work`.
   - Each runner's `.runner` `workFolder` points under
     `runner/<runner-name>/`.
   - The StarVector workflows and every lock user in both repos pass on the
     new paths.
7. **Cleanup respects liveness and budgets.** The cleanup job:
   - removes an idle `t/` folder whose newest file is 8 days old, and a
     build slot with an 8-day-old stamp;
   - skips a same-age folder held open by a running process, and a build
     slot whose cargo lock is held;
   - when `build/` or `t/` is over budget, deletes the least recently used
     idle entries, with warm CI slots last, and reports the per-folder
     sizes.
8. **The old locks retire cleanly.** Once the E3 `git grep` condition holds,
   removing the old paths from the helpers leaves CI green, and test 3's
   lock checks still pass with only the new locks.
9. **The docs are in place.** Each in-scope repo's agent docs contain the E12
   items, checked by a one-line doc test or a review checklist.
10. **The unset variable fails and app data is untouched.**
    - A self-hosted job step with `DEV_HOST_ROOT` unset fails immediately.
    - A SceneWorks smoke run on this Mac resolves its data, config and
      cache folders under `t/`, and leaves `~/SceneWorks` and
      `~/Library/Application Support/SceneWorks` unchanged: same file
      count, no file newer than the run's start.
11. **`dev-host` installs and bootstraps.**
    - A tagged DevWorks release produces both binaries.
    - A repo's CI setup step installs the pinned version and fails on a
      mismatch.
    - A clean machine builds `dev-host` with plain cargo.

## Accepted properties

- Builds on the NAS showed no speed loss (Michael, 2026-11-22). On the
  share, rustc's incremental compilation can't create its session lock
  (macOS error 45), so every build on the Macs runs with
  `CARGO_INCREMENTAL=0`, and no script may set it back to `1` for a build
  on the share (`remote-macos-verification.mjs` does today; SW-4b fixes
  it). Edit-compile loops in agent worktrees therefore
  rebuild whole crates instead of only what changed.
- The NAS is a single point of failure for both Macs' CI and agent builds at
  once, and so is the network path to it. E5 makes the failure loud and
  early. Its SMB mount must come back by itself after a reboot or a drop.
- The Mac volume (now the SMB share) has ownership disabled, so file permissions don't separate
  runner state from agent state.
- Spotlight indexing is read-only on the Mac volume, and Time Machine
  already excludes it.
- Claude Code chooses its own temp folder (`/private/tmp/claude-501`).
  Builds and `TMPDIR` are redirected (E4, E13). Claude Code's own small
  files stay on the system drive.
- While a Mac's volume is missing, its runner launcher retries every 10 s.

## Decided (2026-11-13)

- One shared root for all projects: `dev-host` on each host's external or
  `D:` storage. Both Macs mount at `/Volumes/Models`. (Superseded by r5: the
  Macs share one root on the NAS.)
- Builds go on the external drive (now the NAS).
- Cleanup age: 7 days.
- `nax-macos-2` gets a new 8 TB drive matching this Mac's (ordered; about
  one week out). Its current drive has no room for a temporary root
  (2026-11-15), so it gets its root on the new drive, and the cut-over on
  main waits for it.
- Docker's disk image moves. Docker Desktop isn't in regular use, so the
  restart costs nothing.
- The Windows CUDA runners are in scope, and consolidate on `D:` (changed from `E:` on 2026-11-15).
- The `dev-host` tool lives in DevWorks.
- `~/SceneWorks` is user app data:
  - `data/models/imports` (166 GB) holds Michael's own kreamania fine-tune
    imports.
  - `data/models/resolved` (163 GB) is the app's rebuildable resolved-model
    cache. Some entries may come from agent runs (11-03 04:42–04:46,
    low-to-medium confidence). Whether to clear it is Michael's call,
    outside this feature.

- (Superseded by r5: both Macs now use the NAS share at `/Volumes/Work`.)

## Decided (2026-11-22, r5)

- Both Macs use one root on the Tower NAS, `/Volumes/Work/dev-host`.
  `/Volumes/Models` will not come back as an external drive; the name is
  now the NAS's `Models` share, which holds the HF hub cache.
- `nax-macos-2` needs no new drive; it mounts the same share.

## Open questions

1. **Scratch on SMB.** Keep `TMPDIR` on the share, or put `hosts/<host>/t/`
   on each Mac's local disk? Recommendation: keep it on the share, and run
   the SceneWorks and inference non-inference test suites once with
   `TMPDIR` there (part of SW-1/INF-1). Move it local only if they fail the
   way dev-host's 4 tests did.
2. **Automatic remount.** How does each Mac mount `/Volumes/Work` and
   `/Volumes/Models` at boot, and do they come back after a network drop?
   With E15 the mounts also need each Mac's authenticated account.
   On 2026-11-22 both shares had dropped on this Mac while its runner was
   listening (its pre-job hook would have failed every job), and Michael
   remounted them by hand.
   Both shares are mounted as `GUEST` today, so any machine on the network
   that can reach Tower can also write to them.
