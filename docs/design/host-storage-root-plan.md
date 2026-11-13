# Shared dev host root — implementation plan (DRAFT p3.1)

Status: draft for Michael's review. No CodeGraph items until approved.
Requirements: [host-storage-root-requirements.md](host-storage-root-requirements.md)
(r4, amended; E1–E14, acceptance tests 1–11). Items inherit those
requirements and don't restate them.

- p2 folded in plan review round 1 (13 findings).
- p3 folds in plan review round 2 (13 findings, main ones checked against
  the repos and the runner). Main changes:
  - Host wiring no longer changes how existing jobs and agents build until
    main is ready.
  - Data is copied before the items that need it, and deleted last.
  - Cut-over is split into a code merge and an ops switch.
  - The old-lock retirement condition is restated so it can actually hold
    (Amendment 3).

## Shape and branches

- **Walking skeleton:** HOST-1 creates roots and copies data, without
  changing how anything builds. DW-1 (the `dev-host` tool) runs alongside.
- **Then, in parallel:** the lock transition, tool releases, and per-repo
  adoption.
- **Cut-over:**
  - MERGE puts SceneWorks and inference on main with the new pin.
  - SWITCH turns on the build redirection for everyone, re-registers the
    runners, and deletes old copies once active branches have caught up.
- **After that:** `nax-macos-2`'s drive swap, inference's old-lock
  retirement, and one terminal item.

**Branches** follow `FEATURE_DEVELOPMENT.md`:
- SceneWorks and inference items target mirrored integration branches
  named `feature/<feature-key>-dev-host-root`, created once the CodeGraph
  feature exists. MERGE merges them to main.
- INF-3 and FINAL come after MERGE and target `main` directly. FINAL's pin
  bump is part of its own lock-retirement PR.
- DevWorks, ChatWorks and LensWorks items target their default branches.

**20 items** (p3.1 adds RELNEXT). That's above the usual 5–12 for three reasons:
- **Five repos:** DevWorks requires one item per PR, and every repo needs
  its own PR.
- **Ops work:** HOST-1, SWITCH and HOST-3 are ops, not code.
- **Size:** SceneWorks and inference adoption would exceed about 2000
  changed lines each as single items.

No two items share both a mechanism and a repo.

## Items

### HOST-1 — Roots, runner wiring and seeded data on every host (task, ops)
- **Goal:** Gets every host ready without changing how any existing job or
  agent builds. Roots, markers, the runner check and copied data are all in
  place. No `cargo` shim is on `PATH` yet, and agent harnesses aren't
  redirected yet.
- **Acceptance:**
  - Each host has a verified root (marker present, device differs from the
    system volume) with `locks/`:
    - this Mac: `/Volumes/Models/dev-host`
    - `nax-macos-2`: a temporary root on its current `/Volumes/Models`
    - Windows CUDA: `E:\dev-host`
  - Runner wiring is in place:
    - the pinned `dev-host` binary is installed;
    - each runner `.env` sets `DEV_HOST_ROOT` and
      `ACTIONS_RUNNER_HOOK_JOB_STARTED`, and the hook only runs
      `dev-host check`;
    - the launchd plist runs `dev-host check && exec run.sh`, and the
      Windows service is wrapped where it allows;
    - the per-host budget config and the cleanup schedule are installed;
    - the stale LensWorks plist is removed.

    A queued job fails before checkout when the marker is renamed, and
    main's existing jobs pass unchanged when it's restored.
  - Weights, evidence and caches are **copied** (not moved) into the layout
    on each host and verified by hash or file count. Sources:
    - StarVector, `~/kv-poc`, `~/models`, `~/minimax-h3-*`
    - `~/.cache/sceneworks-*`
    - the agent output in and beside `~/SceneWorks/data`
    - the non-build parts of `/Volumes/Models/sw-7119` and `Codex-builds`

    `weights/<project>/manifest.json` lists each set. The originals stay in
    place.
- **Inherits:** E1, E2, E5, E8, E9, E11, E14
- **Touches:** host filesystems, `~/actions-runner-*/.env`, the launchd
  plists, the Windows runner services
- **Blocked by:** DW-2, DW-3 (for the binary and the schedule). Root
  creation and the data copy can start before them.

### DW-1 — `dev-host` tool: check, env, sticky slots (story, DevWorks)
- **Goal:** One place that verifies the root and decides where builds and
  temp files go.
- **Acceptance:**
  - `dev-host check` passes on a real root. It fails within 10 s, naming
    the mount, when `DEV_HOST_ROOT` is unset, points at a folder on the
    system volume, or has no marker.
  - `dev-host env` prints:
    - a sticky `CARGO_TARGET_DIR` slot under `build/<project>/<key>/`,
      reused on the next call, different for a second worktree with the
      same key, and under a new key when `MACOSX_DEPLOYMENT_TARGET`
      changes;
    - `TMPDIR`/`TEMP`/`TMP` under `t/<8-char id>`.
  - The `cargo` shim refuses a caller-set `CARGO_TARGET_DIR` outside the
    root, and refuses to run when `check` fails.
- **Inherits:** E1, E2, E4, E5, E13
- **Touches:** new `crates/dev-host`, workspace `Cargo.toml`
- **Blocked by:** none
- **Validation:** unit tests with temp-dir fakes for the device and marker;
  one manual run on this Mac.

### DW-3 — `dev-host cleanup` and `dev-host guard` (story, DevWorks)
- **Goal:** Keeps the root bounded, and gives every repo one shared checker
  for new hardcoded paths and for the E12 doc items.
- **Acceptance:**
  - `cleanup` applies the E8 and E9 rules: it reads per-host budgets from
    the root's config, skips held cargo locks and live processes, and
    reports per-folder sizes. `cleanup --migrate <path>` deletes an old
    copy under the same liveness checks.
  - `guard --diff <base>` fails on each E7 pattern in added lines and
    accepts `dev-host: allow-path`. `--all` scans a whole tree.
  - `guard --docs` fails when a repo's agent docs lack an E12 item.
- **Inherits:** E7, E8, E9, E12
- **Touches:** `crates/dev-host`
- **Blocked by:** DW-1
- **Validation:** unit tests with a temp root (stale and fresh stamps, a
  held flock, a live working directory, over-budget ordering). Guard tests
  for every E7 pattern; these are acceptance test 5's cases.

### DW-2 — `dev-host` releases and DevWorks adoption (task, DevWorks)
- **Goal:** Makes `dev-host` installable at a pinned version on macOS and
  Windows, and moves DevWorks itself onto the root.
- **Acceptance:**
  - Pushing a `dev-host-v*` tag publishes macOS arm64 and Windows x64
    binaries. `dev-host install --version X` installs them, and fails on a
    checksum or version mismatch. `cargo build -p dev-host` succeeds with
    no `dev-host` installed.
  - DevWorks PRs run `dev-host guard --diff`. `guard --all` and `--docs`
    pass on DevWorks.
  - DevWorks's SessionStart hook runs `dev-host env`.
- **Inherits:** E1, E7, E12, E14
- **Touches:** a new release workflow, `.github/workflows/ci.yml`,
  `.claude/settings.json`, `AGENTS.md`
- **Blocked by:** DW-3
- **Validation:** one tagged pre-release, installed on this Mac. Hosted CI
  builds the Windows binary.

### SW-1 — SceneWorks locks: dual-lock transition (task, SceneWorks)
- **Goal:** Moves SceneWorks's Metal and Windows CUDA/service locks to the
  fixed root paths, with no window where two jobs share a GPU.
- **Acceptance:**
  - `git grep` for the old lock paths and lock helpers lists only migrated
    call sites or annotated tests. That covers all files that mention the
    lock today (about 27–29), including memory-calibration-harness,
    yue2-*, starvector-terminal-*, load-progress-acceptance and
    cv-provider-validation.
  - The updated helpers take the old lock, then the new one; run the mount
    and marker check natively; and create nothing above `locks/`.
    Acceptance test 3's four Mac chains and the Windows exclusion case pass
    on the CPU-only seam.
  - One real Metal smoke job on this Mac passes holding both locks.
- **Inherits:** E3, E5
- **Touches:** `scripts/ci/metal_execution_lock.py`,
  `scripts/lib/apple-metal-execution-lock.mjs`, `apple-metal-lease.mjs`,
  `metal_execution_*`, `windows_cuda_execution.py`, the lock users found by
  the grep, and their tests
- **Blocked by:** HOST-1 (root creation only)

### INF-1 — inference locks: dual-lock transition (task, inference)
- **Goal:** The same transition for inference's lock copy, its launcher,
  and its Windows CUDA sites.
- **Acceptance:**
  - `git grep` for the old lock paths lists only migrated or annotated
    sites. That includes `inference_metal_launcher.py` (60 workflow
    references), `real-weights.yml`, `test_starvector_terminal_policy.py`,
    and the evidence string at `model_proof_resources.py:584`.
  - The SW-1 rules hold in inference. Test 3's chains pass on the CPU-only
    seam.
  - `scripts/ci/lock_chain_cross_repo.sh`, run once on this Mac with both
    repos checked out, passes SceneWorks parent → inference child and the
    reverse.
- **Inherits:** E3, E5
- **Touches:** `scripts/ci/metal_execution_lock.py`,
  `inference_metal_launcher.py`, `model_proof_resources.py`,
  `model_inference_proofs.py`, the E3 Windows sites, and tests
- **Blocked by:** HOST-1 (root creation only)

### SW-2 — SceneWorks setup step, agent hook, guard, docs (task, SceneWorks)
- **Goal:** Every SceneWorks self-hosted job and agent session starts from
  `dev-host env` on the integration branch, and new hardcoded paths are
  blocked.
- **Acceptance:**
  - A composite setup action runs in every job whose `runs-on` resolves to
    a self-hosted runner, after its toolchain steps. It installs the pinned
    `dev-host` and exports its environment. Hosted jobs skip it, and a
    self-hosted job with `DEV_HOST_ROOT` unset fails at that step.
  - The `.claude/settings.json` SessionStart hook runs `dev-host env`.
    `AGENTS.md` and `FEATURE_DEVELOPMENT.md` pass `dev-host guard --docs`
    and state E5's rule that Docker Desktop starts only after the check
    passes.
  - The `dev-host guard --diff` PR check fails a fixture PR that adds
    `$HOME/foo`.
- **Inherits:** E1, E4, E5, E7, E12, E13, E14
- **Touches:** `.github/actions/dev-host-setup/`,
  `.github/workflows/*.yml`, `.claude/settings.json`, `AGENTS.md`,
  `FEATURE_DEVELOPMENT.md`
- **Blocked by:** HOST-1

### SW-3 — SceneWorks state paths, app data, DerivedData (task, SceneWorks)
- **Goal:** No SceneWorks workflow or script hardcodes machine state, and
  nothing touches `~/SceneWorks`.
- **Acceptance:**
  - `dev-host guard --all` finds no unannotated state paths in `.github`,
    `scripts`, `tools` or `apps/desktop/scripts`. References to folders
    that no longer exist are removed.
  - A SceneWorks smoke on this Mac resolves its data, config and cache
    folders under `t/`, and leaves `~/SceneWorks` and
    `~/Library/Application Support/SceneWorks` unchanged (acceptance test
    10).
  - The StarVector readiness workflow passes reading the HOST-1 copy at
    `weights/sceneworks/starvector`. xcodebuild steps write DerivedData
    under `cache/sceneworks/`.
- **Inherits:** E1, E2, E6, E10
- **Touches:** `.github/workflows/*`, `scripts/**`, `apps/desktop/scripts/**`
- **Blocked by:** SW-2

### SW-4a — SceneWorks build-output consumers (task, SceneWorks)
- **Goal:** Nothing in SceneWorks assumes `<repo>/target`.
- **Acceptance:**
  - About 87 consumers resolve the build folder through a shared
    `cargo metadata` helper (one each for `.mjs`, `.py` and `.sh`). That
    covers the Tauri sidecar, `release.yml`, `check.yml`, the film-harness
    and epic-2123 scripts, and `check-scaffold.mjs`.
  - A desktop bundle build, and a `release.yml` dry run on the signing
    runner, find their outputs with `CARGO_TARGET_DIR` under the root.
- **Inherits:** E4
- **Touches:** `apps/desktop/scripts/build-sidecar.mjs`, `release.yml`,
  `check.yml`, `scripts/**`, the new helpers
- **Blocked by:** SW-2

### SW-4b — SceneWorks cargo launches and `CARGO_TARGET_DIR` sites (task, SceneWorks)
- **Goal:** Every SceneWorks cargo launch builds into a root slot.
- **Acceptance:**
  - The 81 `CARGO_TARGET_DIR` lines in 41 files, the absolute-path cargo
    launches and the `rustup run` sites are removed, or defer to
    `dev-host env`. That includes `macos-mlx.yml:249/410/940`.
    `dev-host guard --all` finds no target-path patterns.
  - `macos-mlx.yml`'s warm build runs in a `warm` sticky slot, and a
    second run of the job starts warm.
- **Inherits:** E4, E7
- **Touches:** `.github/workflows/*`, `scripts/ci/**`
- **Blocked by:** SW-2

### INF-2a — inference setup step, agent hook, guard, docs (task, inference)
- **Goal:** Every inference self-hosted job and agent session starts from
  `dev-host env` on the integration branch, and new hardcoded paths are
  blocked.
- **Acceptance:**
  - The composite setup action runs in every job whose `runs-on` resolves
    to self-hosted. That includes the 17 dynamic ones
    (`needs.config.outputs.runs_on`, `matrix.runner`). Hosted jobs skip
    it, and an unset `DEV_HOST_ROOT` fails the job.
  - A new `.claude/settings.json` SessionStart hook runs `dev-host env`.
    `AGENTS.md` passes `dev-host guard --docs`.
  - The `dev-host guard --diff` PR check fails a fixture PR that adds
    `$HOME/foo`.
- **Inherits:** E1, E4, E7, E12, E13, E14
- **Touches:** `.github/actions/dev-host-setup/`,
  `.github/workflows/*.yml`, `.claude/settings.json`, `AGENTS.md`
- **Blocked by:** HOST-1

### INF-2b — inference state paths, build consumers, DerivedData (task, inference)
- **Goal:** No inference workflow or script hardcodes machine state or
  assumes `<repo>/target`.
- **Acceptance:**
  - `dev-host guard --all` over inference finds no unannotated state or
    target-path patterns. That covers about 287 path matches, 119
    `CARGO_TARGET_DIR` lines, 16 consumers, and the absolute-path cargo
    launches.
  - The 25 xcodebuild sites write DerivedData under `cache/inference/`.
  - One real-weights job and one kv-poc job pass, reading HOST-1's copies.
- **Inherits:** E1, E2, E4, E6, E10
- **Touches:** `.github/workflows/*`, `scripts/**`
- **Blocked by:** INF-2a
- **Size note:** most changes are 1–2-line repoints. If the diff passes
  about 2000 lines, it splits at paths versus build output.

### CW-1 — ChatWorks adoption (task, ChatWorks)
- **Goal:** ChatWorks's agent builds, and its one self-hosted Windows CUDA
  job, use the root.
- **Acceptance:**
  - The `[self-hosted, windows, cuda]` job (`package-validation.yml:152`)
    runs the setup step. Its `CARGO_TARGET_DIR` (:176, currently
    `RUNNER_TEMP`) comes from `dev-host env`, and packaging still finds
    its outputs.
  - A new `.claude/settings.json` hook and a new `AGENTS.md` pass
    `dev-host guard --docs`. The guard runs on PRs.
- **Inherits:** E1, E4, E7, E12
- **Touches:** `.github/workflows/package-validation.yml`,
  `.github/workflows/*`, `.claude/settings.json`, `AGENTS.md`
- **Blocked by:** HOST-1

### LW-1 — LensWorks adoption (task, LensWorks)
- **Goal:** LensWorks's 8 self-hosted `runs-on` lines in 5 workflows and its
  3 path matches use the root. No LensWorks runner is registered today.
- **Acceptance:**
  - Each self-hosted job either runs the setup step on a live runner label,
    or is removed or moved to hosted runners.
  - `dev-host guard --all` passes. A new `.claude/settings.json` hook and
    a new `AGENTS.md` pass `guard --docs`. The guard runs on PRs.
- **Inherits:** E1, E7, E12
- **Touches:** `.github/workflows/*`, `.claude/settings.json`, `AGENTS.md`
- **Blocked by:** HOST-1

### MERGE — SceneWorks and inference to main with the new pin (task, SceneWorks)
- **Goal:** Puts the root-aware code on main in both repos, in
  `FEATURE_DEVELOPMENT.md` §5 order, while the old data locations still
  exist.
- **Acceptance:**
  - inference's integration branch is merged to inference main.
    SceneWorks's pin is set with
    `scripts/bump-inference.mjs --sha <inference-main-sha>`, and
    `git grep` finds no copy of the old revision. SceneWorks's integration
    branch is then merged to main.
  - CI is green on both mains, with self-hosted jobs reading HOST-1's
    copies through `DEV_HOST_ROOT`.
- **Inherits:** E1, E11
- **Touches:** both repos' main branches; SceneWorks pin sites (root and
  crate `Cargo.toml`s, `Cargo.lock`, `config/*closures*/third-party-source.json`,
  catalog-baseline fixtures)
- **Blocked by:** SW-1, SW-3, SW-4a, SW-4b, INF-1, INF-2b

### SWITCH — Turn on redirection, move runners, delete old copies (task, ops)
- **Goal:** Once main is root-aware, makes everyone build on the root, and
  then frees the system drives.
- **Acceptance:**
  - On every host:
    - the `cargo` shim is on `PATH` and the host-level Codex config runs
      `dev-host env`;
    - runners are re-registered with `workFolder` under
      `runner/<name>/_work`;
    - Docker's disk image is in `docker/`.

    Main CI stays green afterwards.
  - Once every `feature/*` branch with a commit in the last 14 days, in
    both repos, contains MERGE, a final delta copy runs. Then the old
    copies are deleted through `dev-host cleanup --migrate`. Afterwards,
    `/Users/Shared/SceneWorks` holds only the old lock file, and the
    acceptance test 6 "gone" list holds.
- **Inherits:** E4, E6, E11
- **Touches:** host filesystems, runner registrations, Codex config on each
  host, `PATH` on each host
- **Blocked by:** MERGE, CW-1, LW-1

### RELNEXT — Dual lock onto `release/next` by hotfix (task, SceneWorks)
- **Goal:** Lets the old locks retire without waiting for a release.
  SceneWorks `release/next` and its pinned inference revision (`2b09938e`)
  carry old-only lock code today. A hotfix job and a main job could share
  the GPU once the old lock is removed.
- **Acceptance:**
  - Through the sceneworks-hotfix process, inference gets a hotfix revision
    based on `2b09938e` that carries INF-1's dual-lock change. SceneWorks
    `release/next` carries SW-1's dual-lock change and pins that inference
    revision.
  - `release/next` CI is green, and a Metal job on it takes both locks.
- **Inherits:** E3
- **Touches:** SceneWorks `release/next` lock helpers and pin sites; an
  inference hotfix branch from `2b09938e`
- **Blocked by:** SW-1, INF-1

### HOST-3 — `nax-macos-2` moves to its new 8 TB drive (task, ops)
- **Goal:** Replaces `nax-macos-2`'s temporary root with the new drive, at
  full budgets.
- **Acceptance:**
  - The new drive is mounted as `/Volumes/Models` with the marker. The
    temporary root's contents are copied and verified, and the budget
    config is set to full.
  - A real-weights job passes on `nax-macos-2` after the swap.
- **Inherits:** E5, E9, E11
- **Touches:** `nax-macos-2` filesystems and runner config
- **Blocked by:** SWITCH, and the drive's arrival (about 2026-11-20; a
  hardware dependency)

### INF-3 — inference retires the old locks (task, inference, targets main)
- **Goal:** inference stops taking the old lock paths once no old-only
  holder remains in either repo.
- **Acceptance:**
  - The amended E3 condition (Amendment 3) holds for both repos. The old
    paths are gone from inference's helpers, and test 3's checks pass with
    only the new locks.
- **Inherits:** E3
- **Touches:** inference lock helpers and tests
- **Blocked by:** SWITCH, RELNEXT

### FINAL — SceneWorks retires the old locks and accepts the feature (task, SceneWorks, targets main)
- **Goal:** Integration proof, run once.
- **Acceptance:**
  - SceneWorks's old lock paths are removed in a PR that also bumps the pin
    to include INF-3. `/Users/Shared/SceneWorks` is deleted.
  - Acceptance tests 1, 2, 4, 7 and 8 pass once on this Mac, and test 2's
    Windows part passes once on a CUDA runner. The plain-folder case of
    test 2 needs admin rights; Michael runs it.
  - `dev-host guard --all` and `--docs` pass in all five repos.
- **Inherits:** all
- **Touches:** SceneWorks lock helpers and pin; `/Users/Shared/SceneWorks`
- **Blocked by:** INF-3, HOST-3

## Dependencies

```
DW-1 ─ DW-3 ─ DW-2 ─┐
                    HOST-1 ─┬─ SW-1 ────────────────────┐
  (roots + copy may │       ├─ INF-1 ───────────────────┤
   start earlier)   │       ├─ SW-2 ─┬─ SW-3 ───────────┤
                    │       │        ├─ SW-4a ──────────┤
                    │       │        └─ SW-4b ──────────┤
                    │       ├─ INF-2a ─ INF-2b ─────────┴─ MERGE ─┐
                    │       ├─ CW-1 ──────────────────────────────┤
                    │       └─ LW-1 ──────────────────────────────┴─ SWITCH ─┬─ HOST-3 ─┐
                    │                                                        └─ INF-3 ──┴─ FINAL
SW-1 + INF-1 ─ RELNEXT ─ INF-3
```

SW-1 and INF-1 need only HOST-1's root creation, which can happen on day
one. The other edges into HOST-1 are for the binary and the schedule.

## Requirement coverage

| Req | Items |
|---|---|
| E1 | DW-1, DW-2, HOST-1, SW-2, SW-3, INF-2a, INF-2b, CW-1, LW-1, MERGE |
| E2 | DW-1, HOST-1 (layout + weights manifest), SW-3, INF-2b |
| E3 | SW-1, INF-1, RELNEXT, INF-3, FINAL |
| E4 | DW-1, SW-2, SW-4a, SW-4b, INF-2a, INF-2b, CW-1, SWITCH |
| E5 | DW-1, HOST-1, SW-1, INF-1, SW-2 (Docker rule), HOST-3 |
| E6 | SW-3, INF-2b, SWITCH |
| E7 | DW-2, DW-3, SW-2, SW-4b, INF-2a, CW-1, LW-1 |
| E8 | DW-3, HOST-1 |
| E9 | DW-3, HOST-1, HOST-3 |
| E10 | SW-3, INF-2b |
| E11 | HOST-1, MERGE, SWITCH, HOST-3 |
| E12 | DW-2, DW-3, SW-2, INF-2a, CW-1, LW-1 |
| E13 | DW-1, SW-2, INF-2a |
| E14 | DW-2, HOST-1, SW-2, INF-2a |

Acceptance tests:
- 3 (SW-1, INF-1)
- 5 (DW-3, plus each repo's fixture PR)
- 6 (SWITCH, FINAL)
- 9 (DW-3 `--docs`, run by each repo item)
- 10 (SW-3)
- 11 (DW-2)
- 1, 2, 4, 7 and 8 (FINAL)

## Requirement amendments

Amendments 1 and 2 were approved by Michael on 2026-11-13 and applied to r4.

1. The old-lock retirement scope (superseded in wording by Amendment 3).
2. The guard's pattern tests live once, in `dev-host`.
3. **Approved by Michael 2026-11-13, with the hotfix route.** The retirement condition as worded was
   circular. "`git grep` finds no old path on `main`" can't hold while
   `main` carries the dual lock, which names the old path. Proposed
   wording: the old locks retire once every one of these refs **contains
   the dual-lock helper**, so no old-only holder remains:
   - `main` and `release/next`
   - every `feature/*` branch with a commit in the last 14 days
   - the inference revision each of those pins

   The condition is checked for both repos together, because the Mac lock is
   machine-wide. `release/next` gets the dual lock by hotfix (RELNEXT), not
   by waiting for a promotion.

## Open questions

1. ~~`release/next` timing~~ Decided 2026-11-13: hotfix (RELNEXT).
2. **Project split (checked).** SW owns SceneWorks, inference and
   ChatWorks; DW owns DevWorks; LW owns LensWorks.
   - The main feature goes in SW. DW gets a linked feature for DW-1–3, and
     LW one for LW-1.
   - If CodeGraph refuses `blocked_by` across projects, those edges go into
     the item descriptions.

## Decided

- `dev-host` lives at `crates/dev-host` in the DevWorks workspace
  (2026-11-13).
