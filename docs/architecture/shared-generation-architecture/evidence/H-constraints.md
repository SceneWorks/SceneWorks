# H — Constraints, process rules, prior design attempts and in-flight work

Research topic H for the SceneWorks + inference re-architecture design. Read-only survey, 2026-11-03.

**Baselines read.** SceneWorks worktree at `origin/main` `a1d95e2c4` (commit date 2026-11-03).
Inference read via `git show origin/main:` at `a43724b82` (merge of PR #1405). SceneWorks pins
exactly that inference commit (`crates/sceneworks-memory-adapter/src/lib.rs:13`,
`crates/sceneworks-worker/src/catalog_semantic_jobs.rs:51`, 48 `rev = "a43724b8…"` manifest
occurrences), so the two `main`s are in lockstep today.

**Citation conventions.** `SW:path:line` = SceneWorks file at a1d95e2c4. `INF:path:line` = inference
file at `origin/main`. `mem:<name>` = memory note
`/Users/michael/.claude/projects/-Users-michael-Repos-SceneWorks/memory/<name>.md`. `skill:` =
`~/.claude/skills/sceneworks-feature/…`. `CG:` = CodeGraph object, read-only. Memory notes are dated
observations. Where a note conflicts with current source, this document says so.

---

## 1. Process constraints

### 1.1 The process contract and who owns it

- **Authority order.** The skills are the executable procedure. FEATURE_DEVELOPMENT.md is the
  contract, and the skills win when the two disagree (`SW:FEATURE_DEVELOPMENT.md:7-15`,
  `SW:AGENTS.md:36-37`). The named skills are `shortcut-plan`, `sceneworks-epic` and
  `shortcut-story`. Since the tracker moved to CodeGraph, the live equivalents are `codegraph-plan`,
  `sceneworks-feature` and `codegraph-item` (global CLAUDE.md; `skill:SKILL.md` frontmatter). The
  repository docs still say "Shortcut epic" and `sc-` naming in places.
- **Autonomy.** An instruction to run an epic authorizes end-to-end delivery, including tracker
  updates, CI repairs, review fixes, PRs, merges and closeout. Only a "system wall" justifies
  escalation: access, credentials, external resources or an enforced permission
  (`SW:AGENTS.md:15-21`, `SW:FEATURE_DEVELOPMENT.md:169-184`). "Repair demonstrably wrong patterns
  and affected contracts" is explicitly in scope (`SW:AGENTS.md:18-19`).
- **Only three legitimate blockers:** hardware you don't have, a credential only Michael can set,
  or a product decision only Michael can make (`mem:a_followup_story_is_a_deferment_do_it_now`,
  `mem:gates_are_dismantled_an_unwired_script_is_never_a_blocker` §4).

### 1.2 Branch model and two-repo lockstep

- **One integration branch per epic:** `feature/sc-<epic>-<slug>` from `origin/main`. If inference
  is in scope, inference gets an **identically named** branch (`SW:FEATURE_DEVELOPMENT.md:45-48`).
- **Publication branches** are `story/sc-<story>-epic-<epic>-<slug>`, and the slug must equal the
  feature slug. Inference enforces this in CI via `scripts/ci/feature_epic_policy.py`
  (`SW:FEATURE_DEVELOPMENT.md:63-80`; `INF:scripts/ci/feature_epic_policy.py:29-41`, regexes
  `FEATURE_RE`/`STORY_RE`).
- **Hazard: CodeGraph keys and inference policy.** Inference `main`'s policy accepts only `sc-NNNN`
  names. The CodeGraph topology (`feature/sw-fNNN-…`, `story/sw-NNNN-feature-sw-fNNN-…`) is added
  by SW-F327's inference commits `f4c5c7f64`/`e39ebb5cb` (PR #1406). Those commits exist only on
  `feature/sw-f327-code-review-remediation`, not on inference `main` (§4.2). A new re-architecture
  feature with a `sw-f` key will red inference's "select affected lanes" job until SW-F327's
  inference half lands on main. A `claude/...` head reds the same job
  (`mem:epic_16774_ios_runtime_lane`).
- **Invariants** (`SW:FEATURE_DEVELOPMENT.md:39-61`):
  - No incomplete feature merges to `main`.
  - Feature branches never merge into `release/next` or into another feature.
  - SceneWorks `main` pins an inference commit reachable from inference `main`.
  - Never force-push shared feature branches.
  - Never commit on `main`.
- **Cross-repo ordering.** Publish and merge the inference batch first, then the consumer batch,
  using the same story-branch name in both repos. Coalesce the interim feature pin into the
  consumer batch via `node scripts/bump-inference.mjs --sha …`. Never open a pin-only PR
  (`SW:FEATURE_DEVELOPMENT.md:265-277`).
- **Terminal story order:**
  1. Freeze batches.
  2. One combined review per repo.
  3. Inference feature→main PR, then record the landed merge commit.
  4. SceneWorks final pin to that inference **main** SHA.
  5. One measurement campaign.
  6. SceneWorks feature→main PR.
  7. Verify and close.

  (`SW:FEATURE_DEVELOPMENT.md:290-332`.) The 2026-10-16 addition requires a whole-branch
  adversarial review before every feature→main merge, capped at 2 reviews and 2 fix passes per repo
  (`mem:process_is_one_review_one_fix_strict_at_epic_small_at_story` §5).
- **Main syncs.** Refresh only for a relevant change, a conflict or an enforced strict policy, never
  for base movement alone (`SW:FEATURE_DEVELOPMENT.md:279-288`). During a running campaign, defer
  main conflict resolution to one pass at merge time (`mem:defer_main_conflicts_until_campaigns_finish`).
- **Interim feature pins.** The default is to coalesce them with consumer batches. Epic 26584
  received an explicit, epic-scoped "approve interim feature pins" (`mem:epic_26584_interim_feature_pins`).
  The skill now allows exact reviewed inference-feature pins (`skill:references/sceneworks-hazards.md`
  "Engine changes are a two-repository lockstep"). That approval does not extend to interim
  campaigns.

### 1.3 Ruleset / merge state (verified 2026-10-24 in the doc; re-query before relying on it)

| | SceneWorks | inference |
|---|---|---|
| main required checks | `web`, `parity`, `candle`, `build-windows`, `check-linux`, `check-macos`, `macOS build, lint and workspace tests (hosted)` | `CI gate` |
| strict (up-to-date) | no | no — re-enabled twice by drift, reset 2026-08-23; never re-enable (`mem:never_reenable_strict_status_checks_on_inference_main`) |
| merge methods on main | merge/squash/rebase | merge commit only |
| feature/* rulesets | none (all removed, incl. deletion guards) | none |
| merge queue | none (removed 2026-08-11) | none |

Sources: `SW:FEATURE_DEVELOPMENT.md:82-107` and `SW:AGENTS.md:31-35`. Merge queues were removed
after 103 groups caught zero integration failures at a median cost of +21 min per merge.
`merge_group:` triggers are retained so re-adding a queue is a ruleset-only change
(`SW:FEATURE_DEVELOPMENT.md:378-384`; `SW:.github/workflows/check.yml:4-11`;
`INF:.github/workflows/ci.yml:4-22`).

Implication: with `strict: false`, there is no forced re-run of open PRs after each merge. That
makes many parallel refactor PRs viable. The same setting means a merged tree that no PR ever
tested is possible, which is why inference keeps a post-merge `push: main` run
(`INF:.github/workflows/ci.yml:4-15`, sc-18825).

### 1.4 Gate teardown (binding)

- On 2026-08-15/16 the pin-keyed verification gates were deliberately dismantled
  (`SW:FEATURE_DEVELOPMENT.md:135-165`, `SW:AGENTS.md:48-54`):
  - `68670a3ee` (sc-19758) set four `check.yml` steps to `if: false`. They are still visible at
    `SW:.github/workflows/check.yml:196,275,287,368`.
  - sc-19751 made license coverage report-only.
  - `e14171984` stopped pin bumps from invalidating capability dumps.
- A disabled or unwired script is the designed state. Never re-enable one, wire one up or file a
  story about one. Measurement work (capability dumps, matrix regeneration, calibration, VRAM and
  canary runs) runs **once at epic end** after the final pin, or on explicit request
  (`SW:FEATURE_DEVELOPMENT.md:149-159`).
- **Runtime never demotes a stale measurement.** Currency (`loaderClosureDigest`, `is_current`,
  `StaleBundle`) is a probe-tooling concept only. Any PR that makes admission read currency is
  "wrong on arrival", and a pin bump has zero runtime cost (`mem:runtime_never_demotes_a_stale_measurement`;
  `SW:FEATURE_DEVELOPMENT.md:137-142` sc-23692, sc-22738).
- **Tests over measured corpora assert shape, not pinned counts.** A re-capture that trips a count
  means the test is wrong (`SW:FEATURE_DEVELOPMENT.md:163-165`, `mem:pinned_population_assertions_are_frozen_corpus_gates`).
- **"No new gates"** covers memory and measurement gates only: calibration, matrix, anchors,
  capability currency, anti-drift guards, workspace invariants and lint sweeps
  (`mem:no_new_gates_runtime_catching_is_the_chosen_tradeoff`). Gates that prevent CI failures are
  wanted. Runtime refusals that *are* the fix are favoured. Mutation testing is not a gate.
- **Tension with SW-F327.** SW-F327's E3 ("make documented architecture, layering, dependency,
  dispatch/capability and cfg invariants executable, including guards against new helper copies")
  is a new *structural* guard (`CG:SW-F327` requirements). Michael evidently approved it for
  SW-F327. The rule text, however, lists "a workspace invariant · an anti-drift guard" as banned.
  The re-architecture plan should ask Michael explicitly whether structural layering guards are
  allowed, rather than assume either way (see §7, Uncertain).

### 1.5 Review and story-sizing rules ("1 review + 1 fix")

From `mem:process_is_one_review_one_fix_strict_at_epic_small_at_story`, global CLAUDE.md
"Implementation Plans", and `SW:FEATURE_DEVELOPMENT.md:186-203`:

- **Strict at the epic, small at the story.** The epic carries numbered E1…En requirements and its
  acceptance tests, written once. Each story carries 1–3 local, observable bullets plus
  `Inherits: E…`. Stories contain no process bullets, no restated epic invariants and no terminal
  gates (pin, campaign, real-weight run, feature→main).
- **Slice by mechanism.** N cells differing only by a parameter make one story with a table.
  Target 5–12 stories, and treat more than 20 as a signal to re-slice. Build a walking skeleton
  first. One terminal integration story owns the pin, campaign, feature-end review and →main.
- **Per-story cost model:**
  1. One implementation.
  2. One adversarial review against local AC (Opus).
  3. One self-verifying fix pass (mutation per assertion, `MUTATIONS_RUN`).
  4. Merge on green.

  Re-review happens only on a `partial` blocker. Strict review happens once on the integration
  branch, plus the pre-merge whole-branch review (≤2 rounds).
- **Measured failures behind the rule:**
  - Epic 20762: 37 stories, 20 of them the same change parameterised by (engine, mode).
  - Epic 17137: 95 stories for one model port, at about 2.5 reviews per story.
  - A reviewer handed epic-level AC flags every story as incomplete.
- **Batch publication.** Track stories separately, but batch small related stories into one
  canonical branch, one reviewed PR and one CI cycle. Parallel agents implement locally. One
  coordinator owns all pushes, PRs and CI (`SW:AGENTS.md:38-47`, `SW:FEATURE_DEVELOPMENT.md:21-37,232-263`).
- **Do not anchor on the per-PR shape of SW-F327.** Its open PRs are 2k–12k LOC and 140–290 files
  each (§4.1). That pattern comes from batching 274 cards and is not a template.

### 1.6 Deferment rules

- **A follow-up story is a deferment** (`mem:a_followup_story_is_a_deferment_do_it_now`):
  - Filing a story does not launder the deferment.
  - Before every report, count stories closed vs filed.
  - "Another epic owns that contract" and "size" are invalid excuses.
  - Discovered gaps get fixed in place, in the same PR or a second PR in the same session.
- **"If I can articulate the deeper fix, that is the assignment"**
  (`mem:feedback_recognized_deeper_fix_means_do_it_now`).
- **Exception/waiver registers are laundered deferments.** `config/backend-capabilities/exceptions.json`
  carried 47 cells of unbuilt Candle work on a paraphrased one-word approval. Before treating any
  waiver artifact as required, prove the gate fails without it
  (`mem:owner_signature_registers_are_laundered_deferments`).
- **Global CLAUDE.md "No Deferral" rule.** Implement the full capability surface. Never write off
  scope as a "follow-on" in a code comment or story. Surface anything that genuinely cannot be done
  and file a tracked item for it.
- **Never write contract-only scaffolding.** Build the walking skeleton first. The definition of
  done is a real artifact, not "it compiles" (`mem:feedback_no_contract_only_walking_skeleton_first`).

### 1.7 No real model execution in ordinary CI

- Michael, 2026-10-23: *"Tests that are performing real inference should NOT be in CI."* This
  includes tiny or synthetic weights. Model-level forward, generation, encode/decode and training go
  to a separately dispatched workflow with recorded evidence. Source may merge, but the policy must
  structurally block execution in PR/main CI (`mem:real_model_execution_is_not_normal_ci`).
- **Inference enforcement** (`INF:AGENTS.md:60-104`):
  - Moved model tests carry `#[ignore = "model inference; explicit model-proof workflow"]`.
  - Model tests are registered in `scripts/ci/model-inference-proofs.json`.
  - `scripts/check_model_inference_policy.py` enforces bidirectional coverage.
  - `scripts/ci/test_suites.py` + `test_suites.json` partition suites into `audited-safe`, `routine`
    and `exhaustive` (`INF:scripts/ci/test_suites.py:1-10`, `INF:.github/workflows/ci.yml:514,607,1569-1586`).
  - Proofs run through `model-inference-proofs.yml` / `model-test-suites.yml`, which are
    dispatch-only (`INF:.github/workflows/model-test-suites.yml:1-16`).
- **What ordinary CI may still run:** raw matmul/conv/SDPA/norm ops, schema/math/tokenizer/header
  checks, mocks, and constructor/load-only checks. A learned module's `.forward` is not exempt
  (`INF:AGENTS.md:71-78`).
- **Consequence for the refactor.** The historical mlx-gen "parity over assertion" discipline (tiny
  Python-dumped fixtures at 1e-2 tolerance, `INF:crates/media/mlx-gen/ARCHITECTURE.md:85-97`) now
  runs only in dispatched proof workflows. **A refactor of numeric leaves therefore gets no
  automatic numeric regression signal from PR CI.** Equivalence evidence must come from explicit
  `model-inference-proofs.yml` dispatches with recorded evidence.

### 1.8 Releasing and the inference pin

- SceneWorks pins inference by exact 40-char SHA. A fix lands in inference first, then a SceneWorks
  pin PR. `scripts/bump-inference.mjs` is the authority. Never hand-edit license, provenance,
  capability or calibration digests (`SW:RELEASING.md:94-151`).
- **The pin lives in more places than documented.** The memory note counts five sites
  (`mem:inference_pin_has_five_sites_not_three`), and the skill hazard file still says "three
  manifests". At a1d95e2c4 the actual set is:
  - root `Cargo.toml` (3 occurrences)
  - `crates/sceneworks-worker/Cargo.toml` (18)
  - `crates/sceneworks-memory-adapter/Cargo.toml` (23)
  - `crates/sceneworks-core/Cargo.toml` (4)
  - `vendor/zip9-mlx-gen/Cargo.toml` (4)
  - `vendor/zip9-candle-audio-kokoro/Cargo.toml` (4)
  - two Rust consts: `INFERENCE_PIN` and `INFERENCE_RUNTIME_REVISION`
  - `Cargo.lock`

  That is **at least 8 sites plus Cargo.lock**. The verification recipe still holds: grep for the
  old rev and expect zero hits.
- The `mlx-rs` fork rev rides with the pin. A new fork rev needs the `prebuilt-<sha12>` release or
  the macOS lanes build libmlx from source (`SW:RELEASING.md:134-141`).
- The fork commits no `Cargo.lock`, so its CI resolves dependencies fresh each run and can red with
  a zero diff (`mem:mlx_rs_fork_resolves_deps_fresh_every_ci_run`).
- **The pin must be on inference main.** SceneWorks `main` once pinned an inference `release/next`
  rev that was not on inference `main`, and feature syncs could not take either side's pin
  (`mem:sceneworks_main_can_pin_an_inference_release_rev_not_on_main`).
- **Every release merges back.** Each release, including a hotfix, ends with a `release/next`→`main`
  merge-commit PR (`SW:AGENTS.md:7-11`). A multi-month refactor will see several such merges land
  on `main` underneath it.

### 1.9 CI workflows: what runs per PR and what it costs

**SceneWorks** has 32 workflow files and 10,808 lines of YAML (`SW:.github/workflows/`).

Lanes that run automatically on PRs and pushes:

| workflow | jobs | runner | when |
|---|---|---|---|
| `check.yml` (512 l) | `changes`, `web`, `supply-chain`, `parity`, `parity-rust`, `parity-scaffold`, `parity-history`, `parity-digests`, `parity-docker`, `candle` | ubuntu-latest | every PR plus push to main; a `changes` gate job instead of `paths:` (`SW:check.yml:3-20,476-480`) |
| `macos-mlx.yml` (1480 l) | `macos-checks` (hosted macos-26), `browser-fixture-artifact`, `moss-native-api-mlx`, **`nax-worker` (self-hosted `nax`)**, `nax-kernel-validation`, `manual-verification` | hosted macos-26, plus self-hosted nax | PRs to `main` and `feature/*`. **`nax-worker` runs only when base==main** (`SW:macos-mlx.yml:907`) |
| `windows-candle.yml` (2218 l) | `candle-worker` (self-hosted `cuda`), plus 6 dispatch-only proof jobs | self-hosted Windows cuda | **PRs to main only**, path-filtered (`SW:windows-candle.yml:137-139`). Not required. A feature PR never fires it; use `workflow_dispatch --ref` (`mem:rust_check_candle_closes_the_macos_candle_blind_spot`) |
| `desktop-windows.yml` | `changes`, `build-windows` (windows-2022), `package-windows` (self-hosted cuda, main/dispatch only `SW:desktop-windows.yml:196-202`) | | PR + main |
| `desktop-macos-check.yml` / `desktop-linux-check.yml` | `check-macos` (macos-15) / `check-linux` (ubuntu-22.04) | hosted | PR + main |
| `starvector-metrics-windows.yml`, `changed-paths.yml` | small | hosted | PR |

About 20 further workflows are dispatch-only real-weight, campaign or terminal lanes:

- `memory-catalog-campaign`, `h3-*`, `flux3-*`, `qwen-image-2-1-app-lifecycle`, `load-progress-acceptance`
- `moss-training-acceptance`, `starvector-terminal*`, `iris-mlx-real-weights`, `supra2-mlx-real-weights`
- `imported-nvfp4-validation`, `server-candle-linux`, `nax-kernel-validation`, `flux3-quant-matrix`

Each targets labels such as `real-weights`, `rw-starvector`, `rw-mage`, `rw-krea` and `rw-llm`.

**Inference** has 16 workflows. `ci.yml` (1863 l) is the only automatic one.

- Its jobs are:
  - `changes`, `resolve`, `format`, `workspace`, `contracts`, `contracts-windows`
  - `candle-cpu-lint`, `candle-cpu-test`, `candle-cpu-conformance`, `candle-cpu-bundles`
  - `macos-metal` (hosted macos-15), `ios-build` (hosted macos-15)
  - `macos-nax` (self-hosted, **manual only**)
  - `windows-cuda-check` (self-hosted `[windows, cuda]`; same-repo PRs)
  - `windows-cuda` (manual)
  - `supply-chain`, `docs`, `release-metadata`, `CI gate`

  (`INF:.github/workflows/ci.yml:71-1863`.)
- `windows-cuda-check` caps `CARGO_BUILD_JOBS` at 12 because two listeners share a 48-thread host
  (`INF:ci.yml:1318-1335`).
- Concurrency is keyed per SHA on main so merges cannot evict each other's verdicts
  (`INF:ci.yml:38-62`).
- `real-weights.yml` (7020 l), `real-weights-yue.yml`, `model-inference-proofs.yml`,
  `model-test-suites.yml`, `ltx25-*`, `yue2-*`, `veda-ab-cuda`, `kv-poc-campaign` and the moss lanes
  are all dispatch or workflow_call.

**Cost per push:**

- The skill baseline (2026-08-15) is about **5 SceneWorks workflows per push, with the macOS lane
  around 37 min**. An inference push takes about 20 min, bounded by `macos-metal`
  (`skill:references/sceneworks-hazards.md` "Every push is expensive"; `mem:sceneworks_epic_skill`).
  These are cold-cache figures and must be re-verified.
- Hosted macOS is an **org-wide cap of 5 concurrent jobs shared across both repos**. A queued macOS
  lane is that cap, not the dev Mac (`mem:subagent_briefs_need_a_standing_verification_contract`,
  CI-attribution clause).
- A push can silently fire zero runs because of a dropped webhook. Diagnose via the check-suites API
  (`mem:github_can_drop_a_push_event_zero_check_runs`).
- **Platform coverage gaps:**
  - `parity` (Linux, no candle) does not compile `image_jobs/base.rs`.
  - `macos-mlx` is the sole coverage for the macOS+default config (`mem:worker_base_rs_compiles_only_on_macos_or_candle`).
  - Candle-only worker tests execute **only** on the feature→main PR's `candle-worker` lane. On two
    epics (19703, 22723) that lane was the first place real defects surfaced
    (`mem:candle_only_worker_tests_run_nowhere_on_the_mac`).
  - Local pre-push requires three configs: `npm run rust:check`, `rust:check:candle` (nvcc stub,
    typecheck only) and `rust:check:neither` (`mem:rust_check_candle_closes_the_macos_candle_blind_spot`).

### 1.10 What the process rules imply for a multi-month refactor

1. **Structure.** The refactor is one CodeGraph feature (or a short chain of features), each with
   mirrored `feature/sw-fNNN-<slug>` branches in both repos. It carries 5–12 mechanism-sliced
   items, not one item per family per backend. "Migrate family X to the shared seam" is a parameter
   cell, so it becomes one item with a family table.
2. **Size the feature for wall clock.** Main moves fast: SW-F327 alone merges about 60 commits a
   day into its feature branch. A refactor feature that runs for months will either drift thousands
   of commits behind (19048 is now 4,011 behind and dead) or pay a sync tax on every relevant main
   change. The plan needs several **sequential features that each reach main**, not one long-lived
   branch.
3. **Equivalence evidence.** It must come from weights-free contract and conformance tests that run
   in ordinary CI, plus explicitly dispatched model-proof workflows with recorded evidence. Never
   from PR-CI model execution.
4. **Defer measurement.** Do not re-run campaigns per slice. One terminal campaign per feature, and
   only where a measurement contract actually changed. Stale records are expected mid-refactor and
   must not change runtime behaviour.
5. **Engine and consumer move together.** Every engine-contract change needs the inference batch
   merged first and a coalesced pin in the consumer batch. Contract-shape changes in `gen-core`
   therefore have a two-repo cost per batch.
6. **Pre-push checks per batch.** Run all three local Rust configs, the inference
   `check-workspace.py`, the clock-assertion ratchet and `cargo metadata --locked`. Before any
   feature→main touching candle-gated worker code, dispatch `windows-candle.yml` against the
   feature branch.

---

## 2. Prior design attempts relevant to sharing

### 2.1 Memory-strategy contract and pipeline target design (SceneWorks docs)

- **`SW:docs/memory-strategy-contract.md` (570 l).** SC-15449 defines one worker-owned selection
  path for every image provider: `sceneworks_worker::memory_strategy::select_strategy`. Providers
  may reject a strategy but "must not contain a second least-cost selector" (`:1-8`).
  - The ownership table assigns:
    - **Provider:** capabilities, lifecycle, estimate, realization, fingerprint.
    - **Manifest:** coefficients, envelopes, provenance, coverage.
    - **Worker:** budget, selection, precision invariance, fallback, telemetry (`:24-30`).
  - The five ordered strategies are Resident, StagedResidency, BoundedDecode, BoundedAttention and
    BoundedTransformerResidency.
  - Backend budget semantics are explicitly not interchangeable (`:31-37`).
- **`SW:docs/memory-strategy-pipeline-target-design.md` (994 l).** Status is "proposal, not
  ratified". The reference implementation is MLX `z_image_turbo`, and scope is the 53 image catalog
  entries (`:3-8`).
  - **One-sentence goal:** "One pipeline decides, once, per request … every fact that decision
    rests on is declared in exactly one place" (`:12-19`).
  - **Candle is "an adopter, not a constraint"** (`:21-27`).
  - **Single-encoding audit (§9, `:687-702`).** It counts the encodings per fact, for example
    rung-4 prerequisites at 4 encodings and the escalation order at 2. The rule: ">1 is a place two
    implementations can disagree; 0 is a rule nothing enforces." This is directly reusable as the
    re-architecture's success metric.
  - **Test discipline (`:704-716`):** mutation-check every shared-contract test (SC-15793's
    near-vacuous kernel test), and "a test asserting a default value is a false green".
  - **Migration order (§11, `:762-780`):**
    1. Mechanical rename.
    2. Explicit prerequisites.
    3. Rung-1 reshape.
    4. Feasibility/refusal.
    5. Tier integrity before calibration.
    6. Delete `mempolicy`.
    7. Reference certification.
    8. Adopters.

    Rationale: "nothing is measured twice". This is the sequencing argument a refactor must
    re-apply: **ABI- or fingerprint-changing renames go first, while evidence is cheapest to
    invalidate.**
  - **§14 Portability (`:941-994`).** Rung 2 (`gen_core::tiling`) is the only planner fully shared
    across image and video. Three things blocked video and audio:
    1. Naming. Since fixed by SC-15804.
    2. `MemoryStrategy::ALL: [Self; 5]`, which makes a sixth rung (KV-cache bounding, expert
       residency) a breaking change.
    3. `MemoryGeometry` carries no frames or duration axis.

    Audio's streaming output conflicts with rung 1's release point.
- **What landed:**
  - The lane-neutral rename (SC-15804).
  - Tier integrity (SC-15799).
  - The rung contract in `gen-core`.
  - The z-image reference.
  - Anchor collapse (sc-22505/22513/22514, most recent commits on both docs).

  What stalled: the adopter rollout. CodeGraph still has 13 per-family stories "Implement the
  shared image memory ladder for X — MLX/Metal" in `in_progress` (SW-4802, 4805–4812, 4816, 4819,
  4821), a per-family horizontal rollout that never closed.

### 2.2 Epics 18093 / 18803 / 19048 (memory apparatus)

- **18093 "right-size calibration, estimates admit full ladder"**, filed 2026-08-07 after Michael
  said the apparatus "stands in the way of progress" (`mem:epic_18093_right_size_calibration_estimates_admit_full_ladder`):
  - **Diagnosis:** 11 days after v0.8.1 there were 421 commits at a **2.6:1 fix-to-feat ratio**,
    44 new `.mjs` files, and a 22 MB / 9,140-cell matrix over 65 evidence records. In the note's
    words, "the apparatus was armor against agent failure modes, not a domain requirement."
  - **What survived:** "The runtime core (fit gates, phase curves, evidence records, closure
    digest) is the good 20% and stays."
  - **Decisions:** estimates admit the full ladder on both backends, measurement becomes a runbook
    rather than per-model stories, and margins derive from repeat-capture variance.
  - **Outcome:** all 20 original stories were done in 2 sessions (11 PRs). The sole planned
    measurement (sc-18104, flux2_dev) was never executable because `mlx:flux2_dev` had no adapter
    arm.
  - **Process lesson:** "Every adversarial review this session found something real, and the
    implementer never found it first."
  - **Phase-0 lesson:** land, fix or park in-flight PRs *before* the refactor. The refactor
    "rewrites every surface those PRs regenerate", and a 22 MB regenerated matrix cannot be
    rebased, only regenerated.
- **18803 "video memory ladder"** (`mem:epic_18803_memory_ladder_is_image_only_by_construction`).
  The apparatus was image-only by construction, with four structural seams:
  1. Matrix filter `type === "image"`.
  2. A two-coefficient `fixedGb`/`perMpxGb` phase curve. `vram_gate.rs` is candle-only, and MLX
     never reads phase curves (`mem:phase_vram_curves_are_read_only_by_the_candle_gate`).
  3. Capture arms refuse `frames != 1`.
  4. `video_request.rs` bypasses strategy selection entirely.

  **Lesson:** "add video rows" cannot work when the abstraction lacks the axis. Status at 2026-08-12
  was 11/29 done. The SceneWorks feature branch is now merged (0 ahead).
- **19048 "converge candle and MLX admission"** (`mem:epic_19048_converge_candle_and_mlx_memory_admission`):
  - **Michael's directive:** "I want them both converged." He rejected the framing that the candle
    image lane is "a working path worth not disturbing", because it is "a neglected patchwork". Only
    7 models were hand-wired into `candle_memory_strategy.rs`.
  - **Target:** one prediction mechanism with two bases (evidence extrapolation + declared floor).
    Axes (area, frames, batch, reference_count) are defined once. Backends supply only the budget
    probe, the margin and the failure posture. Coefficients stay per lane forever.
  - **Migration discipline:** a decision-DIFF baseline harness (sc-19049), not byte-identity.
  - **Cross-epic collision set:** `vram_gate.rs`, `video_jobs/candle.rs`,
    `model-manifest.schema.json` and `memory-matrix.json` (fingerprinted on 8 sources; the
    second-merging branch must regenerate).
  - **What happened:** the SceneWorks branch accumulated **137 commits** (last on 2026-08-22, e.g.
    "promote SC-19057 Candle video curve") and **never merged**. It is now 4,011 commits behind main.
    CodeGraph still shows SW-5586 "Epic 19048 acceptance: one mechanism, two lanes, divergence
    inventory empty" as `in_progress`.
  - **Lesson:** a convergence epic that cannot reach main inside a few weeks dies under main's
    churn. Its mechanism may be partly superseded by later teardown epics (22505 anchor teardown,
    22657 contract memory derivation, 22723 measurability), but this was not verified.
- **Current size of the convergence targets:**

  | file | lines |
  |---|---|
  | `SW:crates/sceneworks-worker/src/mlx_fit_gate.rs` | 23,311 |
  | `SW:crates/sceneworks-worker/src/candle_memory_strategy.rs` | 10,132 |
  | `SW:crates/sceneworks-worker/src/vram_gate.rs` | 8,217 |
  | `SW:crates/sceneworks-core/src/video_request.rs` | 7,430 |

  These are the surfaces a shared admission layer would replace.

### 2.3 Inference contract work (gen-core, consolidation, rearchitecture)

- **`INF:docs/architecture/inference-rearchitecture.md`** (accepted and implemented 2026-07-13).
  Five repos became one workspace with one lockfile. Registration is explicit (no `inventory`, no
  force-link). Named runtime bundles are `runtime-macos`, `runtime-cuda` and `runtime-cpu`.
  - The migration was staged so each step was observable before removal (`:293-311`):
    1. Primitives.
    2. Asymmetric providers.
    3. Catalogs with exact-surface tests.
    4. Consumer cutover.
    5. Remove the legacy path.
    6. Cut the consumers.
  - **Invariants for future changes (`:313-362`):**
    - Contracts are tensor-neutral.
    - Family or feature host code lives in sibling `gen-core-*` crates, not `gen-core`: 13 exist
      today, including `gen-core-train`, `-tokenizer`, `-image`, `-h3`, `-ltx` and `-memory-wan`.
    - Duplicate provider IDs fail.
    - MLX/Candle catalog differences are allowed but pinned in tests.
    - **Inference never fetches weights or derives a cache location.** This is enforced by
      `scripts/check-workspace.py` and `deny.toml`.

  Any new shared hook layer must respect these.
- **`INF:crates/media/mlx-gen/ARCHITECTURE.md`, "disciplined hybrid" (`:11-25`):**
  - Numeric leaves are a faithful mirror of the reference, so parity localizes to one op.
  - Weight keys are a compatibility contract.
  - Orchestration (loaders, dispatch, pipelines, adapters, quant) is clean Rust.
  - "Dispatch & pipelines (deferred, non-speculative)": introduce a `Pipeline` trait when the
    second family lands, not before (`:79-83`). The trait was never introduced; there are now 43
    `mlx-gen-*` dirs.
- **`INF:crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md` (399 l):**
  - One `Generator` for all prompt→media and a separate `Transform`.
  - `Conditioning` is a typed enum; requests are `Default`-able structs.
  - "Config unifies within a family, not across" (§3.2).
  - The core stays fixed as models are added (`:24-36,374-399`).
  - The §7 strain list covers upscalers registered as `Generator`, conditioning variety,
    guidance-distilled vs CFG, video modes, and audio as a modality.
- **CodeGraph SW-F89 "unified gen-core contract (backend-neutral mlx-gen + candle-gen)" is fully
  complete:**
  - Neutral contract crate (SW-1144).
  - Sampler policy split (SW-1145).
  - Registry-derived capability advertisement (SW-1146).
  - "Worker: single backend-neutral paths (generate/train/caption/transform) + named carve-outs"
    (SW-1147).
  - Conformance testkit (SW-1518).
  - One gen-core per graph (SW-1519).

  Even so, the 2026-10-27 review finds MLX↔Candle twin drift in the worker. **The neutral contract
  reached the call boundary, but the worker's job adapters and memory-adapter binaries stayed
  twinned.**
- **CodeGraph SW-F134 "Provider crate boilerplate reduction" is partial:**
  - Done: macros (SW-2474/2475/2479/2546), SDXL-family trainer hoist (SW-2476), candle shared
    flow-matching trainer (SW-2480).
  - **Not done:** shared `components()` lazy-cache helper + seed loop (SW-2481, ready), hoisting
    per-model scalar schedule math into gen-core (SW-2482, ready), and parametric `flux1_mmdit`
    block shared by flux + chroma (SW-2477, backlog).

  These are exactly the cross-family hook points a re-architecture would own.

### 2.4 Candle consolidations — what landed, what re-grew

From `INF:crates/media/candle-gen/CODE_REVIEW_2026-07-01.md` and `…07-12.md`:

**2026-07-01 findings** (78 findings):

| finding | duplication |
|---|---|
| F-019 | Loader `component_vb` (sorted safetensors → unsafe mmap → VarBuilder) duplicated **~34×**, including 5 copies in qwen-image alone, with error-string and behavioural drift (flux2 single-file check unreachable) |
| F-025 | **4 `QLinear` seams** with load-bearing drift: flux2 uses `QMatMul::forward` int8 fast_mmq, which lens documents as corrupting under outliers (sc-7702) |
| F-018 | Adapter-merge skeleton copied across **8 crates (~7,300 LOC)** |
| F-020–F-024, F-026 | Wan ×4 UMT5 encode; Kolors ×3, Z-Image ×3, FLUX.1 ×3 and FLUX.2 ×3 pipeline scaffolding; budgeted VAE tiling 1:1 between wan and ltx, with "de-dupe is a tracked follow-up" written without a story id |

**2026-07-12 re-review.** 71/78 were fully fixed. "The consolidation the last review asked for
genuinely landed: one audited loader, one quant seam, one adapter-merge skeleton, shared
seeds/tiling/testkit". **But re-accretion followed:**

- **F-018 partial.** The per-crate `adapters.rs` count grew **8→10** (~8,400 LOC), PEFT prefixes are
  still copied, and 8 crates keep their own `merge_adapters` shell.
- **F-106.** A new wave of per-crate `QLinear{Dense,Packed}` (×5) and `QEmbedding` (×3) wrappers
  accreted *over* the shared seam, about 2,900 LOC across 12 `quant.rs` files: "six types named
  `QLinear` again coexist".
- **F-059 partial.** Cosmetic duplication is growing: `to_image` went 4→12 copies and `repeat_kv`
  12→16 files.
- **F-081/F-102/F-103.** The i32 attention-overflow guard is missing at 5 sites, and the
  steps==0/floor sweep missed 5 lanes. The poisoned-mutex class re-appeared at about 12 sites from
  a later RoPE-cache wave. **New crates re-grow previously fixed bug classes.**

### 2.5 Inference whole-repo review 2026-08-01 (the memory-ladder adoption layer)

- "**The contract is ready; the adoption template is not.**" Per-adopter ladder plumbing runs
  200–1000 lines, of which about 250 per story is copy-paste gate/translation/scope machinery.
  Divergence ran at about 3 semantic defects per 7 adopters, none detectable by conformance
  (`INF:CODE_REVIEW_2026-08-01.md:3-12`).
- Recommended **before** rollout:
  - F-182/F-183/F-187: hoist the safety gate and selection→`GenerationMemory` translation into
    gen-core.
  - F-184: behavioral testkit probes.
  - F-185: add the load-shape axis to the evidence key while evidence is cheap.
- Related findings: F-194 ("MemoryAssetFacts semantics have forked four ways"), F-197 (request-scope
  lifecycle "sextuplicated with drifted guarantees") and F-198 ("flux2's ported legacy gate is a
  template-poisoning counter-example").

### 2.6 Lessons extracted for this plan

1. **Hoist the adoption template, not just the contract.** SW-F89 neutralized the call boundary;
   the 08-01 and 10-27 reviews show divergence lives in per-adopter glue. The design must make the
   shared path the *only* path. Every family should supply data and narrow hooks, and the
   gate/translation/scope/lifecycle code should exist once.
2. **Re-accretion is the default outcome** (F-106, F-018 partial, F-059 growth). A consolidation
   without a structural reason to use the shared seam re-grows within 11 days. Options:
   - make the shared type the only exported one (seal per-crate variants);
   - a review checklist item;
   - an `architecture_tests.rs`-style guard, which is in tension with "no new gates" (§1.4).
3. **Sequence ABI/evidence-key changes first** (pipeline doc §11; F-185). Every measured record
   keyed to an old ABI must be invalidated anyway, and invalidation is cheapest before the next
   campaign.
4. **Add axes, not rows** (18803). The shared request, geometry and admission model must carry
   frames, duration, batch and reference count from day one, or video/audio/edit lanes will bypass
   it again.
5. **Convergence epics must reach main in weeks** (19048 dead at 137 commits).
6. **Use decision-diff migration harnesses**, not byte-identity (19048 sc-19049). Byte-identity is
   reserved for numeric paths.
7. **Per-family rollouts stall.** 13 "Implement the shared image memory ladder for X" items are
   still `in_progress`. Migration should be mechanical and batched by mechanism, or done by the
   same PR that creates the seam (CodeGraph rule: slice by mechanism, never by cell).
8. **Characterize before consolidating:** add tests that pin current behaviour of each copy first
   (`mem:consolidation_deferred_for_test_coverage`, PR #176/#177). This is also SW-F327 E6:
   "structural splits … move existing tests first".
9. **Keep the disciplined hybrid.** Numeric leaves stay faithful mirrors and orchestration is
   shared, so a refactor must not "clean up" the parity-pinned numerics. The sanctioned exceptions
   (seedvr2/lens/minimax-h3/ltx Linear stacks, sc-27321) stay exceptions unless goldens are
   re-recorded via a dispatched proof (`INF:crates/media/mlx-gen/ARCHITECTURE.md:56-66`).
10. **Apparatus as armor.** The 2.6:1 fix:feat lesson argues that the design should not grow
    tooling (generated matrices, multi-file evidence chains, hash-pinned producers) as a way to
    police agents. Prefer simpler runtime code with runtime refusals.

---

## 3. Code-review themes and recommended fixes

### 3.1 SceneWorks CODE_REVIEW_2026-10-27.md (274 findings; 98 redundant, 55 architecture-drift)

Part 1 Rust themes (`SW:CODE_REVIEW_2026-10-27.md:2840-2853`):

1. **Hardened-then-re-copied helpers.**
   - Atomic temp+rename ×9 with differing durability, streaming hash ×13, hex ×10,
     `safe_download_dir` ×3, 8+ safe-relative-path predicates, 4 HF repo-id validators, safetensors
     header reading ×6, the `MemoryEvidence` literal ×10.
   - Root cause: core utilities are `pub(crate)`.
   - Fix: a public core `hashing`/`env`/`fs` module.
2. **MLX ↔ Candle twin drift**, which the review calls highest leverage.
   - The same tier clamp, quant parse, adapter loop and model-dir resolver exist once per backend
     with "mirrors X" comments. Floor semantics diverged (W2-001: `any` vs `all`).
   - The memory-adapter binaries have 15 byte-identical functions (X1-006..008).
   - "Video is consistently one hardening generation behind image" (W1).
   - Verdict: "**A backend-neutral shared layer is the highest-leverage structural change.**"
3. **Shared worker layers only partially adopted.** `analysis_jobs_common`, `progress.rs`,
   `smoke_support.rs`, `payload::quant_int` and `ort_cuda` each have long tails of verbatim clones.
   Cancel semantics drift in hand-rolled loops, making cancellation the main recurring runtime risk.
4. **Rules enforced by comments.** `architecture_tests.rs` does not enforce layering or cfg rules,
   and `use super::*` appears in 42 worker modules and 44 of 61 rust-api modules.
5. **Trust-boundary validation applied at entry points only**, not to values read back or to sibling
   code.
6. **Loopback trust model** (A2-001 + X1-001). This is the High security item: no Origin/Host check
   on `/api/*`, and the `/mcp` defence is off in LAN mode.
7. **SQLite stores lack a common open/configure/migrate/begin-write entry point** (`project.db` is
   missing `BEGIN IMMEDIATE`).
8. **God files and `include!`:**

   | file | size |
   |---|---|
   | `image_jobs/base.rs` | 19.6k lines in the review, **20,329** at a1d95e2c4 |
   | `run_image_generate_job_unexplained` | 1,292 lines |
   | `rust-api/src/lib.rs` | 6.3k |
   | `models.rs` | 15.8k |
   | `memory-adapter/src/bin/mlx.rs` | 33.8k (**34,023** now) |
   | `bin/candle.rs` | 13,140 |

   Inline tests are 30–70% of many files.
9. **The cfg predicate is repeated about 1,900 times across about 100 files.** I count 1,959
   occurrences of `all(not(target_os = "macos"), feature = "backend-candle")` in
   `crates/sceneworks-worker/src` alone. A `build.rs` cfg alias is "the biggest single readability
   win".
10. **Drift guards are strong where they exist** (job types, family allow-lists, geometry, version
    alignment) **and absent one layer down**: web `fallbackModels` vs `builtin.models.jsonc`,
    env-flag truthiness, crate dependency direction. CONTRIBUTING and per-crate READMEs lag the gate
    teardown.
11. Positive notes.

Part 2 (web/tooling) themes (`:2855-2864`):

- Extract-once-never-migrate: the `api/` layer is bypassed at 236 call sites.
- God components: `App.jsx` 4.3k, `ImageStudio` 4.9k, `ImageEditor` 4.5k.
- Polling/lifecycle gaps.
- Server-owned rules re-implemented client-side.
- Parallel Simple-UI stacks.
- Dead CSS.
- `scripts/` is 139k LOC of one-epic harnesses re-implementing sha256 and canonical JSON with
  different key ordering.
- 192 MB of raw frames tracked in git.

**Remediation order recommended by the review** (`:2866-2873`):

1. Security middleware.
2. Correctness (BEGIN IMMEDIATE, floor `any`/`all`, cancel loops).
3. A public core utility module, then delete copies and extend `architecture_tests.rs`.
4. Structural work: cfg alias, split `image_jobs`/`rust-api/lib.rs`/`models.rs`, extract inline
   tests.
5. Web.

**This review is already being executed** as CodeGraph feature **SW-F327** (§4.1). The
re-architecture must not re-plan those 274 cards. It should build on what SW-F327 lands and avoid
the files it is mid-way through.

### 3.2 Inference mlx-gen reviews

- **2026-07-01** (150 findings, ~233k LOC, 31 crates).
  - Headline: new crates and seams "each re-introduce one instance of a bug class the older crates
    already fixed".
  - Duplication findings:
    - F-056: six copy-pasted registry load functions.
    - F-067: triplicated LyCORIS resolve-and-install skeleton.
    - F-068: copy-pasted cancel/eval/progress preamble in three curated-sampler drivers.
    - F-072: WanVace validators.
    - F-078: qwen/krea TE leaves.
    - F-082: SDXL pipeline triplication.
    - F-086: boogu duplicating flux's VAE key remap.
    - F-090: four z-image + two sana load bodies.
  - Contract findings:
    - F-007: steps≥1 floor claimed but absent.
    - F-008: `Error::Unsupported` documented as load-bearing but never constructed.
    - F-009: "Generator conformance testkit is opt-in and covers 2 of ~35 registered ids".
- **2026-07-11** (181 findings, ~265k LOC, 33 packages).
  - All 28 tracked 07-01 items were fixed.
  - The dominant systemic theme remains "**fixes don't travel**": "roughly a third of all findings
    are a guard/fix that exists in one crate … missing from a sibling".
  - The sequential-residency rollout (epic 10834) shipped as an **8-way copy-pasted scaffold** with
    cancel and `clear_cache` gaps:
    - F-173: CancelFlag at stage boundaries (8 sites).
    - F-174: cleanup on error/cancel exits (8 sites).
    - F-175: "Hoist the eight-way copy-pasted residency scaffold into one shared seam".
    - F-172: wire Sequential across z-image.
  - Other recurring classes:
    - Advertised-but-inert axes: F-115 (anima scheduler), F-109 (flux1 control ignores `use_pid`),
      F-159 (scail2 advertises MultiReference and silently drops it).
    - Hand-rolled validate skipping the shared floor: F-018 (SD3), F-158 (scail2 generate skips
      self-validate), F-106 (PuLID).
    - Step-invariant hoists missing: F-151, F-153.
- **Recommended fix pattern across both reviews:** move the guard, validation floor, cancellation
  checkpoints, progress accounting, cleanup-on-exit and step-invariant caches into shared drivers
  that providers cannot bypass. Make the conformance testkit mandatory for every registered id with
  behavioural probes.

### 3.3 Inference candle-gen reviews (07-01: 78 findings, 124k LOC; 07-12: 85 findings, 182k LOC)

- 07-01 headline: "hard-won fixes not propagating across sibling crates" and "copy-paste pipeline
  scaffolding … has already produced at least one behavioral drift bug".
- 07-12 headline:
  - Two blocking defects in the never-reviewed full-Bernini pipeline.
  - Five unguarded attention sites (i32 overflow).
  - Wan MoE trainer locking each expert to half the dataset.
  - Systemic: "new crates re-grow previously-fixed bug classes (steps==0 floors, per-forward RoPE
    rebuilds, silent config swallows)".
- **Recommended fixes:**
  - One shared `component_vb`.
  - One `QLinear` with an explicit `MatmulStrategy { Int8Fast, DequantDense }`.
  - A generic `candle_gen::quant::detect::{QLinear,QEmbedding}`.
  - Lift the adapter-merge skeleton into `candle_gen::train::lora` with a per-family
    `resolve_targets` closure.
  - A generic `decode_tiled(decode_fn, plan, …)` + budget resolver.
  - F-105: `[workspace.dependencies]` instead of per-crate pins.

### 3.4 Inference CODE_REVIEW_2026-08-01.md

See §2.5. Hoist gate/translation/scope into gen-core (F-182/183/187). Add behavioural conformance
probes (F-184) and the load-shape evidence axis (F-185). F-200 notes the audio lane sits entirely
outside the ladder while the gen-core doc implies coverage.

### 3.5 Cross-cutting themes (the union)

| # | theme | evidence | recommended direction |
|---|---|---|---|
| T1 | Fixes don't travel; siblings re-grow fixed bug classes | mlx 07-11 headline; candle 07-12 F-081/102/103; SW 10-27 themes 1,3,5 | One shared driver per stage; providers implement narrow hooks; guards live in the driver |
| T2 | MLX↔Candle twins at the worker/adapter layer | SW 10-27 theme 2 (W1, W2-001, X1-006..008); 19048 | Backend-neutral worker job layer; backends supply budget probe, margin, failure posture, realization |
| T3 | Contract ready, adoption template not | inf 08-01 F-182..187, F-194, F-197, F-198 | Hoist the adoption template into gen-core; make the declarative surface the only one |
| T4 | Consolidations re-accrete | candle 07-12 F-018 partial, F-106, F-059 | Seal shared types; delete per-crate variants; one name per concept |
| T5 | Advertised-but-inert capability axes | mlx F-115, F-109, F-159; candle F-005, F-072, F-088 | Capabilities derived from what the shared driver actually consumes; reachability tests |
| T6 | Cancellation, progress and cleanup drift in hand-rolled loops | SW W4-003; mlx F-173/174, F-006/014/018/019 | Shared denoise/decode loop with eval-point cancel and RAII cleanup |
| T7 | Image-only abstractions leave video/audio a generation behind | SW W1; 18803 seams; F-200 | Axes (frames/duration/refs/batch) in the shared geometry from the start |
| T8 | God files, `include!` and cfg repetition | SW themes 8, 9 | cfg alias; module splits as pure moves with tests moved first |
| T9 | Rules as comments; drift guards absent one layer down | SW themes 4, 10 | Michael must decide whether structural guards count as "new gates" (§1.4) |
| T10 | Step-invariant work rebuilt in hot loops | candle F-012/F-089/F-107/F-108; mlx F-029/F-151/F-153 | Shared per-request "prepare" stage with cached invariants |

---

## 4. In-flight work that would collide

### 4.1 SW-F327 "SceneWorks Code Review 2026-10-27": active, large, overlapping

- **CodeGraph state** (`CG:SW-F327` v8):
  - 275 items: 71 complete, 4 closed, 62 in progress, 102 in review, 36 ready.
  - Requirements E1–E8. E2, E3 and E5 are not yet covered by any complete item.
  - Its E5 is "MLX/Candle consolidation preserves every supported route and intentional backend
    difference" — the same subject as this re-architecture's core.
  - The execution ledger (2026-11-03) shows a coordinator (`codex`) running multiple warm worktrees.
    It has "admitted" frontier batches for "full MLX/Candle adapter owner splits + neutral
    helpers/typed accessors/provider constants" (SW-6720–23, 6807–08) and video/audio studio splits
    (SW-6881/6882).
- **Branches:**
  - SceneWorks `feature/sw-f327-code-review-remediation` is 59 commits ahead of main and 0 behind.
  - Inference `feature/sw-f327-code-review-remediation` is 8 commits ahead (PR #1406 CodeGraph branch
    policy; PR #1407 "own native CV engines and shared ORT setup").
  - Inference touches `crates/media/gen-cv/` (new), `crates/bundles/runtime-{cuda,macos}`,
    `scripts/ci/feature_epic_policy.py` and the workflows.
- **Open SceneWorks PRs, all targeting the SW-F327 feature, all updated 2026-11-03.** Each touches
  140–290 files except where noted:

| PR | title | +/− | dominant dirs |
|---|---|---|---|
| #3271 | share artifact, display, environment and UTC helper contracts | +12092/−11963, 278 f | `sceneworks-core/src` (56), `rust-api/src` (17), memory-adapter, mcp, workflows |
| #3270 | enforce shared native architecture and dependency contracts (9 cards) | +8623/−9671, 242 f | core (41), **worker (24)**, `worker/build.rs`, `worker/ARCHITECTURE.md`, rust-api |
| #3269 | share deterministic test settings and smoke support (7 cards) | +10139/−11701, 291 f | core (43), worker (21), rust-api |
| #3268 | derive UI model facts from live authoritative sources | +4071/−2885, 173 f | `apps/web/src` (78), rust-api |
| #3267 | centralize confinement and streaming hashes (16 cards) | +5359/−4066, 189 f | core (41), **worker (37)**, `core/build.rs`, `config/mlx-workspace-anchors.json` |
| #3266 | extract Model Manager component owners | +4628/−2441, 144 f | web |
| #3265 | extract App action owners and shell concerns | +5778/−3519, 148 f | web |
| #3264 | **unify memory contracts and bounded device helpers (18 cards)** | +3970/−3082, 133 f | **worker (82)**, core (12), `config/mlx-workspace-anchors.json` |
| #3262 | bound HTTP access and preserve web request ownership | +3791/−1566, 139 f | web, rust-api |
| #3260 | move CV execution into pinned inference engines (SW-6775, SW-6680) | +2519/−2442, 36 f | worker (10), cv scripts, workflows (pins inference SW-F327 head) |
| #3259 | consolidate tooling and preserve archived evidence | +2721/−79440, 272 f | docs/calibration, workflows, Dockerfiles |
| #3258 | join producers and validate shared payloads | +1983/−2010, 80 f | **worker (72)** |
| #3250 | share durable atomic file publication | +1263/−415, 22 f | core, worker, rust-api, desktop |

Other open PRs: #3244 and #3243 (dependabot actions and cargo bumps to main). Inference has **zero
open PRs**.

**SW-F327 items that occupy refactor-core files** (in-progress, many claimed by `codex`):

| item | finding | subject |
|---|---|---|
| SW-6651 | F-014 | MLX/Candle twin helpers in Qwen/Z-Image control |
| SW-6652 | F-015 | image vs video LoRA resolution |
| SW-6653 | F-016 | 1,292-line `run_image_generate_job_unexplained` |
| SW-6654 / SW-6730 | F-017 / F-093 | `image_jobs/base.rs` `include!` ladder split |
| SW-6661 | F-024 | provider ids hard-coded in generic gates |
| SW-6696 | F-059 | safetensors header ×6 |
| SW-6701 / 6702 | — | family vocabularies |
| SW-6720–6723 | F-083..086 | memory-adapter MLX/Candle twins and god files |
| SW-6743 | F-106 | WeightsSource helpers |
| SW-6751 | F-114 | stub-render scaffolding image vs video |
| SW-6786 | F-149 | image/video request-validation asymmetry |

**Collision assessment.** SW-F327 is mid-flight on nearly every SceneWorks file a backend-neutral
worker layer would rewrite:

- `crates/sceneworks-worker/src/image_jobs*` and `video_jobs*`
- the memory-adapter `bin/{mlx,candle}.rs`
- `sceneworks-core` utilities and `build.rs`, where SW-F327 adds the cfg alias (F-040/F-042)
- `architecture_tests.rs`

**The re-architecture's SceneWorks half cannot start on those files until SW-F327's structural
splits (E6) and twin consolidation (E5) land on main**, or it must be planned explicitly as a
continuation of them. Inference is comparatively clear: only `gen-cv`, bundles and CI policy are in
flight.

### 4.2 Feature branches (both repos), ahead/behind main (GitHub compare API, 2026-11-03)

- SceneWorks has 37 `feature/*` refs. **Only 3 carry unmerged commits:**
  - `feature/sw-f327-code-review-remediation`: 59 ahead / 0 behind (active).
  - `feature/sc-19048-memory-admission-convergence`: 137 ahead / 4,011 behind, last commit
    2026-08-22. Stalled (§2.2); it touches `vram_gate.rs`, `video_jobs/candle.rs`,
    `candle_memory_strategy.rs`, the manifest schema and the matrix (93 files).
  - `feature/sc-25899-qwen-image-2-1-turbo`: 33 ahead / 1,245 behind, last 2026-10-16. The epic is
    done per `mem:epic_25899_qwen_image_2_1_turbo`, so this is residue.
  - The other 34 are 0 ahead (merged, retained inert).
- Inference has 35 `feature/*` refs. **Unmerged:**
  - `feature/sw-f327-code-review-remediation`: 8 ahead (active).
  - `feature/sc-20414-advanced-multi-pass-sampling`: 47 ahead / 4,251 behind, last 2026-08-21.
    Stalled; CodeGraph SW-5758 "versioned denoise-pass contract" and SW-5765 "chained denoise passes
    across families" are still `in_progress`. **This is directly a shared-pipeline-hook design**
    (denoise passes as a cross-family contract).
  - `feature/sc-25327-h3-video-upscale`: 4 ahead / 1,377 behind, last 2026-10-11.
- **Epics named in memory as active are now merged to main:**

  | epic | SceneWorks merge | inference merge |
  |---|---|---|
  | 16774 iOS | #3223 | #1368/#1375 |
  | 26584 MOSS training | #3203 | #1361–#1365 |
  | 26679 H3 fine-tune import | #3196 | — |
  | 27980 generic load progress | #3239 | #1397–#1404 |
  | 22988 YuE2 | merged (0 ahead) | — |

  Memory notes `epic_16774_ios_runtime_lane`, `epic_26679_h3_finetune_import_handoff`,
  `epic_22988_yue2` and `epic_19703_resolved_model_hot_cache` are therefore historical. Their
  constraints persist in code:
  - iOS 8 GB device lane, `runtime-ios`.
  - Generic load-progress at the shared IO floor (`mem:load_progress_must_be_generic_and_truthful`).
  - Hot-cache resolver: one resolver, loaders receive resolved paths only, "any `model.id == "..."`
    branch in availability logic is a defect" (`mem:epic_19703_resolved_model_hot_cache`).
- **Still-open CodeGraph items** (in_progress, not on a live branch; stale imports to be
  re-validated, not collisions):
  - 13 per-family memory-ladder adoption stories (SW-4802..4821).
  - SW-5586 (19048 acceptance).
  - SW-6008 / SW-6125 / SW-6332 terminal items.
  - SW-5331 ladder peak re-check.
  - SW-4158/4171 candle Wan GGUF levers.

### 4.3 Files a refactor would touch that in-flight work also touches

| file / area | in-flight owner | nature |
|---|---|---|
| `SW:crates/sceneworks-worker/src/image_jobs.rs` + `image_jobs/base.rs` (20k lines, `include!`) | SW-F327 SW-6653/6654/6730 (split), #3258/#3264/#3270 | pure-move splits in progress |
| `SW:crates/sceneworks-worker/src/video_jobs*` | SW-F327 SW-6652, SW-6751, SW-6786 | image/video helper unification |
| `SW:crates/sceneworks-memory-adapter/src/bin/{mlx,candle}.rs` | SW-F327 SW-6720–6723, SW-6808 (claimed by codex) | twin consolidation + god-file split |
| `SW:crates/sceneworks-worker/src/{mlx_fit_gate,candle_memory_strategy,vram_gate}.rs` | #3264 "unify memory contracts" (18 cards); stale 19048 branch | admission convergence |
| `SW:crates/sceneworks-core/src/*` utilities, `core/build.rs`, `worker/build.rs` (cfg alias) | #3250, #3267, #3269, #3270, #3271 | public core utility owners; cfg alias F-040/042 |
| `SW:crates/sceneworks-worker/ARCHITECTURE.md`, `architecture_tests.rs` | #3270 | executable architecture invariants (E3) |
| `SW:config/mlx-workspace-anchors.json` | #3264, #3267 | memory anchors |
| `SW:.github/workflows/*` | #3259, #3260, #3267, #3269–3271 | CI consolidation |
| `INF:crates/media/gen-cv/`, `crates/bundles/runtime-{macos,cuda}` | inference SW-F327 (#1407) | CV engines moved into inference |
| `INF:scripts/ci/feature_epic_policy.py` | inference SW-F327 (#1406) | CodeGraph branch topology (prerequisite for any `sw-f` feature) |
| gen-core denoise/sampler contract | stale `feature/sc-20414` (47 commits); SW-5758/5765 | cross-family denoise pass contract |
| `Cargo.lock`, the pin sites | every consumer batch + dependabot #3243 | lock merge hazard (§5) |

---

## 5. Hazards a refactor must not regress

Each entry gives the rule, its source, and its specific relevance to a sharing refactor.

### 5.1 Runtime and numerics (MLX / Metal)

1. **Lazy weight loads trip the GPU watchdog, and MLX hid the error**
   (`mem:lazy_weight_loads_trip_the_gpu_watchdog_and_mlx_hid_the_error`).
   - An MLX `Load` runs on the CPU stream. Unread `Weights` make the first GPU forward `encodeWait`
     on disk inside a command buffer, which fails with Timeout/`SubmissionsIgnored` or a silently
     wrong image.
   - The fix pattern is `materialize_accessed()` at the loader before quantize, and
     `eval_pending_loads` before derived arrays. It was applied across ~28 crates in inference#1091.
     **"A NEW loader that skips this re-opens the bug."**
   - The contrary rule: "Never materialize eagerly inside a block stream." Qwen went from 2 to
     22 GiB. Lazy per-block reads must stay (#1099).
   - **Relevance:** a shared loader must encode *both* rules and expose stream-vs-resident loading
     explicitly.
2. **MLX thread-state bugs are invisible to in-crate tests**
   (`mem:mlx_thread_state_bugs_are_invisible_to_in_crate_tests`).
   - `apply_gpu_memory_limit` is a no-op under `cfg(test)`, so the test process has different MLX
     runtime state than a real worker. A mutation check stayed green.
   - **Relevance:** worker-layer refactors touching MLX call order, threads or `spawn_blocking`
     need process-level (API→queue→worker) validation, not unit tests.
3. **Never signal-kill an MLX render mid-command-buffer**
   (`mem:never_signal_kill_an_mlx_render_mid_command_buffer`,
   `mem:mlx_submissions_ignored_poisons_the_process_not_always_the_host`).
   - A kill wedges the GPU client class for the host. First discriminate process scope from host
     scope; a reboot is Michael's call.
   - **Relevance:** shared cancellation must use eval-point cancel, never signals. Test harnesses
     must not `pkill`.
4. **MLX Metal is not thread-safe under the parallel test harness.** mlx-gen sets
   `RUST_TEST_THREADS=1` and uses process-unique temp fixtures (`INF:crates/media/mlx-gen/ARCHITECTURE.md:85-97`).
   SceneWorks worker suites *do* run in parallel; `.cargo/config.toml` does not set it
   (`mem:epic_19703_resolved_model_hot_cache`). Inference mlx crates have one generated
   `integration` test binary per crate, and a new `tests/*.rs` needs
   `consolidate_integration_tests.py regen` or it silently never compiles
   (`mem:inference_mlx_crates_have_one_integration_test_binary_per_crate`).
5. **Every GPU/LLM run needs an external RSS guard.** A planner smoke grew the worker to 90 GB and
   panicked the 128 GB Mac (`mem:planner_smoke_worker_grew_to_90gb_and_panicked_the_mac`).
6. **Admission semantics are deliberate.**
   - Runtime never demotes stale measurements (§1.4).
   - The tier is a whole-pipeline contract: never run a component above the user's tier
     (`mem:tier_is_a_whole_pipeline_contract`). Every model has all three tiers
     (`mem:feedback_all_models_all_three_tiers`).
   - Memory-optimization coverage always expands (`mem:memory_optimization_coverage_always_expands`).
   - MLX OOM is process death, so margins are tight; candle OOM is recoverable, so margins are
     looser (`mem:epic_18093_…`).
   - Backend budget semantics differ by design (`SW:docs/memory-strategy-contract.md:31-37`).
   - **A shared admission layer must keep the backend-specific posture as a hook, not average it
     away.**
7. **Frozen producer scripts are hash-pinned.** Editing one reds 4 tooling tests and demands a
   real-weight re-record (`mem:frozen_producer_scripts_are_hash_pinned_and_cannot_be_edited`). A
   refactor of evidence or parity tooling paths must leave producers byte-identical.
8. **Worker smokes can replicate an engine loader instead of calling it.** SenseNova smokes were
   green while the real registry load was broken
   (`mem:worker_smokes_can_replicate_an_engine_loader_instead_of_calling_it`). Shared seams must be
   *called* by smokes, never re-implemented.

### 5.2 Correctness of claims and review

9. **Capability declaration ≠ enforcement ≠ reachability.** `impl_generator!`'s `generate` does not
   call `validate` (`mem:capability_declaration_is_not_reachability`). Engine geometry gates can be
   no-ops because `video_request.rs` normalization pre-clamps
   (`mem:sceneworks_normalization_makes_engine_geometry_gates_no_ops`). Behaviour claims about a
   refactored seam need a constructed request run through the real entry point; the weights-free
   `unloaded()` + `validate` path makes this cheap.
10. **Story boundaries leave integration gaps nobody owns.** In 19703, mechanism and activation were
    split across stories and no production caller existed; both stories passed review
    (`mem:story_boundaries_leave_integration_gaps_nobody_owns`). **For a refactor, each item that
    introduces a seam must also migrate at least one production caller and delete the replaced path
    in the same item.** Grep for callers of new entry points.
11. **Delegated mutation stubs can ship.** An `if true { return None; }` was left in `mlx_fit_gate.rs`
    (`mem:a_delegates_mutation_stub_can_ship_in_source`). The integrator greps for `if true`,
    `if false`, unconditional early returns and `&& false`. `git add -A` after a formatter, or
    mid-mutation, commits the mutation (`mem:git_add_all_after_a_formatter_is_the_hazard`).
12. **False-green classes** (`skill:references/sceneworks-hazards.md` "Test-integrity traps";
    `mem:test_asserting_a_default_value_is_a_false_green`;
    `mem:cargo_feature_unification_makes_dash_p_the_false_green`):
    - ignored real-weight tests "passing";
    - `cargo test -p` hiding feature unification;
    - assertions on default values;
    - test args placed before `--`;
    - cosine-only checks.
13. **A changing set of failing tests is a leaked lock, not flakiness**
    (`mem:an_unstable_failing_set_is_a_leaked_lock_not_flakiness`).
14. **Every implementation brief needs the verification contract** (verify premises, place guards on
    the faulting value, run one mutation per assertion, no swallowed defaults, no machine-dependent
    goldens, doc attachment, don't assert away the class) and the CI-attribution clause
    (`mem:subagent_briefs_need_a_standing_verification_contract`). Subagents can compact away a
    mid-task directive and reverse it (`mem:a_subagent_can_compact_away_a_mid_task_directive_and_reverse_it`).
    Put directives in the brief, not mid-task messages.

### 5.3 Build and platform configuration

15. **cfg-gated code compiles on only one platform.**
    - `image_jobs/base.rs` is `include!`d only on macOS or non-macOS+candle; Linux-no-candle never
      compiles it. Symbols whose callers are all candle-gated become dead code on macOS
      (`mem:worker_base_rs_compiles_only_on_macos_or_candle`).
    - A macOS-gated helper called from an all-platform path breaks the Linux parity lane
      (`mem:targeted_checks_arent_ci_and_macos_gated_helper_breaks_linux_parity`).
    - macOS clippy compiles no CUDA-gated code (`mem:macos_clippy_cannot_see_cuda_gated_code`).
    - Run `rust:check`, `rust:check:candle` and `rust:check:neither` before every push touching
      cfg-gated worker code.
16. **Candle-only tests run nowhere on the Mac.** `rust:check:candle` only type-checks. The first
    execution is the feature→main `candle-worker` lane, unless you run
    `cargo test -p sceneworks-worker --lib` under the candle cfg in the Docker candle image, or
    dispatch `windows-candle.yml --ref feature/...`
    (`mem:candle_only_worker_tests_run_nowhere_on_the_mac`,
    `mem:rust_check_candle_closes_the_macos_candle_blind_spot`). **A backend-neutral refactor moves
    a lot of logic across this boundary.** Plan an explicit candle test dispatch per batch that
    touches candle-gated code.
17. **Cargo.lock merge hazard.** Taking one side's lock drops the other side's dependency edges.
    Local clippy silently rewrites the lock, CI then dies `--locked`, and 11 checks went red on
    #3196 (`mem:merged_cargo_lock_must_pass_locked_metadata`). After any lock resolution, run
    `cargo metadata --locked` and `git status`. Pin moves rewrite about 91 lock entries.
18. **Build-script output dirs.** In inference, `cargo check`/`build`/`test` got different
    `pmetal-mlx-sys` OUT_DIRs, each taking about 6 min for the libmlx cmake build, unless the
    build-override debug settings match
    (`mem:cargo_check_build_test_get_separate_build_script_outputs_unless_build_override_matches`).
19. **Toolchain traps in agent shells:**
    - Homebrew rustc 1.99 shadows rustup 1.97.1 (`mem:homebrew_rustc_shadows_rustup_in_agent_shells`).
      Note that SW `check.yml` pins toolchain "1.98.0", so versions have moved; verify.
    - Xcode 27 strip flags (`mem:local_rust_builds_need_strip_false_on_xcode27`).
    - CLT SDK newer than Xcode ld (`MEMORY.md` index, `clt_sdk_ahead_of_xcode_breaks_rust_link`).

### 5.4 Resources and fleet

20. **Agent worktree `target/` dirs are 30–42 GB each** (inference worktrees 10–14 GB). Eight agents
    filled the disk to ENOSPC and locked out every tool. Cap at **3–4 concurrently building
    SceneWorks agents**, keep at least 150 GB container free, and `rm` finished `target/` dirs
    (`mem:agent_worktree_target_dirs_fill_the_internal_disk`). Agent Docker builds took Michael's
    containers down (`mem:agent_docker_builds_can_take_down_michaels_containers`).
21. **Cap subagents at 8–10 across all roles** (token budget, plus CPU contention producing false
    test failures at load 33–57). Use an explicit wind-down protocol on limits
    (`mem:subagent_concurrency_cap_is_eight_to_ten`). The current host guidance allows up to 10
    excluding the coordinator (`SW:docs/host-and-runner-guidance.md` §5).
22. **The dev Mac GPU is Michael's.** Campaigns are a loan: state the duration, get a go, never
    chain passes (`mem:the_dev_mac_gpu_is_michaels_not_the_campaigns`). Use nax-macos-2 via
    workflows for MLX real weights (`mem:use_nax_macos_2_via_workflows_not_the_dev_mac`). Pre-push
    Rust checks stay local, immediate and `nice -n 10`. Never load the host by driving Michael's
    apps (`mem:never_load_the_host_by_driving_michaels_own_applications`).
23. **Ending a turn "waiting on CI" without a background waiter** stalls for hours
    (`mem:ending_a_turn_on_ci_needs_a_background_waiter`).
24. **Never halt for a licensing finding.** Record it and keep building
    (`mem:never_halt_work_for_a_license_finding`).

### 5.5 Product-surface rules a refactor must preserve

25. Preserve every control and permutation, including advanced and conditional ones
    (`mem:feedback_design_refactor_preserve_all_controls`). Preserve public crate names, provider
    IDs, serialized fields and weight keys unless making an explicit compatibility change
    (`INF:AGENTS.md:43-50`).
26. **Load progress must be generic and truthful,** instrumented at the shared IO floor (vendored
    candle-core safetensors, the mlx-rs fork pending-load path, the gen-core seal), never per
    provider (`mem:load_progress_must_be_generic_and_truthful`). This is a precedent for the "hooks
    at every stage" goal: epic 27980 already built one cross-cutting hook (`crates/contracts/load-progress`
    in inference).
27. **Resolver discipline** (epic 19703): manifests are declarative, there is one resolver, loaders
    get resolved paths, and there are no model-id branches. Inference never fetches weights
    (`INF:docs/architecture/inference-rearchitecture.md` invariants).
28. **No `inventory`, global loaders or force-link.** Every family is named in its platform catalog,
    and adding a model updates the stable-surface id lists (`INF:AGENTS.md:47-50`,
    `mem:inference_monorepo_consolidation_runtime_cutover`).

---

## 6. Hardware, runners and validation implications

### 6.1 Inventory (dated; `SW:docs/host-and-runner-guidance.md` §1–4, verified 2026-10-24; AGENTS.md says verify live)

**Six self-hosted runners**, all org-scoped in the `self-hosted-gpu` group and accessible to
SceneWorks, inference and ChatWorks. The 2026-10-24 move made `-3`/`-4` org-scoped, so
`mem:inference_cannot_reach_two_of_the_four_cuda_runners` is **superseded**.

- **`cuda-windows`, `-2`, `-3`, `-4`:** four listeners on **one** physical host, `MICHAEL-TRX50`.
  - 2× RTX PRO 6000 Blackwell Max-Q at about 95.6 GiB each.
  - Only the first two carry `real-weights`, since they share one disk with the weights.
  - Automatic Candle PR CI selects physical GPU1.
  - "Four listeners … do not provide four GPUs."
- **`nax-macos`:** primary Mac, MacBook Pro M5 Max, 128 GiB unified memory, which is also
  Michael's dev Mac. Labels: `nax`, `signing`, `real-weights`, `weights`, `rw-starvector`.
- **`nax-macos-2`:** secondary Mac. Labels: `nax`, `weights`, `rw-mage`, `rw-sa3`, `rw-krea`,
  `rw-audio`, `rw-llm`, `rw-chroma`. Hardware is not documented in the guidance; collect it on the
  host.
- **Label caveats.** Inference `real-weights.yml` still routes `rw-mage` to "nax-macos" per the
  older `mem:real_weights_macos_lane_splits_by_rw_labels`, but the current doc lists `rw-mage` on
  nax-macos-2. Treat the doc as current, and re-verify before dispatch.

**Hosted runners:** ubuntu-latest/22.04/24.04, windows-2022/latest, macos-15, macos-26. Hosted
macOS has an **org cap of 5 concurrent jobs** across both repos.

**Devices for iOS:** one iPhone 16 Pro Max (8 GB) (`mem:epic_16774_ios_runtime_lane`). It is not a
CI runner.

### 6.2 Implications for validating a refactor across ~40 families × 2 backends

- **Scale.**
  - Inference has 43 `crates/media/mlx-gen/mlx-gen-*` dirs and 40 `candle-gen-*` dirs, plus 16
    `crates/audio/candle-audio-*` crates and LLM engines, across 138 `Cargo.toml` files.
  - The SceneWorks catalog has about 53 image entries (pipeline doc) plus about 10 video models in
    6 families (18803 note), with three tiers each.
  - The real-weight matrix is roughly (40 families × modes × 3 tiers × 2 backends). That is far
    beyond what is affordable per slice.
- **What ordinary CI can and cannot prove.** Weights-free paths:
  - descriptor conformance (`gen-core-testkit`, opt-in coverage noted as a gap in mlx F-009);
  - `unloaded()` + `validate`;
  - exact catalog surfaces;
  - decision-diff harnesses for admission and routing;
  - raw-kernel unit tests.

  **No learned forward runs in PR/main CI** (§1.7). The automatic signal for a numeric-leaf change
  is compile-only.
- **Numeric equivalence must come from dispatched proofs.**
  - Inference: `model-inference-proofs.yml` (registered cases, recorded evidence),
    `model-test-suites.yml` (`exhaustive`/`baseline`), `real-weights.yml` (per-label jobs).
  - SceneWorks: the per-family dispatch lanes listed in §1.9.
  - **Physical capacity for those proofs:** one Windows box with 2 GPUs (only `cuda-windows`/`-2`
    have weights), plus two Macs, one of which is Michael's working machine.
- **Design consequences:**
  1. **Prefer refactors that are provably numerics-neutral by construction**: pure moves, shared
     orchestration around unchanged numeric leaves. That matches the "disciplined hybrid". Leave
     the leaves alone, or change them only with tiny parity fixtures run via dispatch.
  2. **Use representative-family proof sets, not the full grid.** The AGENTS.md matrix rule
     (measure the lowest and highest memory fits and estimate between them, `SW:AGENTS.md:22-27`)
     is the closest precedent: pick one representative family per mechanism (e.g. z-image for
     image, LTX or Wan for video, one audio model) plus the families whose code paths are bespoke.
  3. **Batch all proof dispatches into one unfiltered pass per feature terminal**, scheduled as an
     explicit GPU loan for the dev Mac and preferably routed to nax-macos-2 and the CUDA box
     (`mem:campaign_coverage_comes_from_the_runner_listing_not_a_hand_list`: coverage comes from
     the runner's unfiltered `--list`, never a hand list).
  4. **CUDA capacity is one host with four listeners.** Dispatch to compatible labels and let
     Actions queue; never cancel running inference to admit another (`INF:AGENTS.md:93-96`,
     `mem:cuda_lane_cap_must_count_open_prs_not_agents`, where the old caps are obsolete).
  5. **Worker-level refactors need process-level validation for MLX thread-state classes** (§5.1
     item 2). They also need candle-cfg test execution via Docker or a `windows-candle.yml`
     dispatch, because candle tests do not run on PRs to `feature/*`.
  6. **No campaign solely for refactoring.** SW-F327 E7 states this outright. Measurement
     re-capture is warranted only where the refactor changes a measured contract (evidence key,
     ABI, loader closure semantics). Even then, stale records keep working at runtime.

---

## 7. Design implications (rules the plan must follow)

1. **Tracker and branch topology.**
   - One CodeGraph feature per shippable phase, each with strict E1…En and acceptance tests on the
     feature, 5–12 mechanism-sliced items with 1–3 local AC, a walking skeleton first and one
     terminal item.
   - Mirrored `feature/sw-fNNN-<slug>` branches in both repos.
   - **Prerequisite:** inference's `feature_epic_policy.py` must accept CodeGraph keys (today only
     on SW-F327's inference branch), or the feature must use an `sc-` style name.
2. **Phase the program into features that each reach main within weeks.** Do not build one
   multi-month integration branch; 19048 and 20414 died that way at 4,000+ commits behind.
3. **Sequence after, or explicitly fold in, SW-F327.**
   - Its 13 open PRs and about 160 in-progress/in-review cards occupy `image_jobs*`, `video_jobs*`,
     the memory-adapter binaries, core utilities, the cfg alias and `architecture_tests.rs`.
   - Either start the SceneWorks half after SW-F327's E5/E6 land, or re-home specific SW-F327 cards
     (with Michael's agreement).
   - Never fork parallel edits to the same split. Inference-side engine work can start sooner.
4. **Land, fix or park in-flight work first** (18093 Phase 0). Get explicit dispositions for:
   - stale `feature/sc-19048` (137 commits);
   - `feature/sc-20414` (47 commits; denoise-pass contract);
   - `feature/sc-25327`;
   - the 13 per-family ladder items SW-4802..4821;
   - SW-5586, SW-5758 and SW-5765.

   Some are direct prior art for the shared hooks.
5. **Measure success by the single-encoding audit** (pipeline doc §9). Every fact is declared once
   and derived everywhere, with counts recorded before and after. Pair it with "model count changes
   manifest data and fixtures only, never the volume of control code" (19703 bar).
6. **Hoist the adoption template, not just the contract.** Shared drivers own:
   - validation floor
   - cancellation checkpoints
   - progress
   - cleanup-on-exit
   - step-invariant prepare/cache
   - gate→`GenerationMemory` translation
   - request-scope lifecycle
   - load materialization policy

   Providers supply narrow hooks and data. Capabilities must derive from what the driver consumes
   (no advertised-but-inert axes).
7. **Design the shared request, geometry and admission types with all axes from day one:** area,
   frames, duration, batch, reference count, and stream-vs-buffered output. Allow an open rung set,
   not `[Self; 5]`. Image-only abstractions have been rebuilt for video twice.
8. **Keep backend differences as explicit hooks:**
   - budget probe
   - margin
   - failure posture (MLX OOM = process death)
   - stream/materialize policy
   - matmul strategy (Int8Fast vs DequantDense)

   Do not average them away. Keep catalog differences pinned in tests.
9. **Every seam-introducing item migrates at least one real production caller and deletes the path
   it replaces, in the same item.** No contract-only scaffolding, and no "mechanism now, activation
   later" splits across items. Verify callers by grep at item boundaries.
10. **Characterize, then consolidate.** Pin each copy's current behaviour with weights-free tests,
    then merge copies. Use structural splits as pure-move PRs with tests moved first (SW-F327 E6),
    and decision-diff harnesses for admission and routing.
11. **Order ABI, fingerprint and evidence-key changes first** in each feature, before any measured
    work accrues against the old key (pipeline §11; F-185).
12. **No model execution in PR/main CI.** Numeric equivalence goes through dispatched proof
    workflows with recorded evidence (`model-inference-proofs.yml`, `real-weights.yml`,
    `windows-candle.yml` dispatch) on a representative-family set. Batch them into one terminal
    pass per feature. No campaign solely for refactoring.
13. **Measurement and gates:**
    - No new memory or measurement gates. Never re-enable `if: false` steps.
    - Runtime never reads currency.
    - Tests assert shape, not populations.
    - Get an explicit ruling from Michael on structural/architecture guards (SW-F327 E3 vs the "no
      new gates" text) before planning any.
14. **Two-repo cost discipline.**
    - Inference batch first, then the consumer batch with a coalesced pin.
    - Never a pin-only PR. Run `bump-inference.mjs` and verify zero old-rev hits across all pin
      sites (at least 8 plus the lock).
    - Run `cargo metadata --locked` after any lock merge.
    - The final pin must be on inference `main`.
15. **Pre-push validation per batch:**
    - `rust:check`, `rust:check:candle`, `rust:check:neither`;
    - inference `check-workspace.py` and `check_clock_assertions.py`;
    - `cargo fmt`/clippy `-D warnings`;
    - a candle-cfg test execution (Docker or `windows-candle.yml --ref`) whenever candle-gated
      worker code moves;
    - process-level worker validation for MLX threading/runtime-state changes.
16. **Preserve the numerics and the invariants:**
    - numeric leaves stay faithful mirrors; sanctioned Linear exceptions stay unless goldens are
      re-recorded via dispatch;
    - weight keys, provider IDs and serialized fields are stable;
    - no `inventory`/global loaders; inference never fetches weights;
    - lazy-load materialize rules (materialize at the loader, never eagerly inside block streams);
    - tier-as-whole-pipeline;
    - all three tiers per model;
    - every UI control preserved.
17. **Review budget:** one adversarial review against local AC plus one self-verifying fix pass per
    item, a whole-branch review of at most 2 rounds per repo before feature→main, and the
    verification contract pasted into every brief.
18. **Fleet discipline:**
    - at most 8–10 subagents, and at most 3–4 concurrently building SceneWorks agents;
    - remove finished `target/` dirs;
    - one coordinator owns publication;
    - batch CI repairs;
    - always leave a background CI waiter;
    - GPU work via nax-macos-2 and CUDA workflows; the dev Mac GPU only on an explicit loan;
    - never `pkill` MLX.
19. **No deferment.**
    - Discovered gaps are fixed in place.
    - Count items closed vs filed.
    - No waiver registers.
    - Anything genuinely blocked (hardware, credential, product decision) is surfaced in the reply
      and tracked.
20. **Prefer simplicity over apparatus.** No new generated matrices, multi-file evidence chains or
    hash-pinned producers to police agents. Runtime refusal is the preferred error mechanism.

---

## 8. Uncertain / not verified

- **CI timing figures** (5 SceneWorks workflows per push, ~37 min macOS; inference ~20 min) are the
  2026-08-15 cold-cache baseline from the skill. I did not measure current run durations or
  fan-out.
- **Whether SW `candle-worker` on PRs to main executes any learned computation**, which would
  conflict with the no-model-execution rule, was not checked. Inference's policy is explicit;
  SceneWorks' equivalent enforcement was not located.
- **Exact pin-site count.** I counted manifest occurrences by grep plus two consts. I did not run
  `bump-inference.mjs` to confirm its rewrite set, and did not check whether `vendor/zip9-*`
  manifests are rewritten by it.
- **Whether `feature/sc-19048`'s mechanism is superseded** by later epics (22505/22657/22723 or
  SW-F327 #3264 "unify memory contracts") was not analysed. I only verified that it is stale and
  unmerged.
- **What `feature/sc-20414`'s 47 commits contain** (denoise-pass contract) was not read. It is a
  likely prior-art source for cross-family hooks and should be reviewed by the topic owner for
  sampling and pipelines.
- **The SW-F327 plan for SW-6720–6723 and #3264.** I read titles, file-dir counts and the feature
  ledger, not diffs. How far SW-F327 will go toward a backend-neutral worker layer, and whether its
  E5 already subsumes part of this re-architecture, needs a direct read of those PRs and their card
  AC.
- **Hardware specs of `nax-macos-2`** are not in the guidance doc; not collected (read-only).
- **Runner inventory** is from the 2026-10-24 doc, not live (`gh api` runner queries were out of
  scope by instruction). The `rw-mage` host assignment conflicts between an older memory note
  (nax-macos) and the current doc (nax-macos-2).
- **Whether Michael considers structural/architecture guards** (SW-F327 E3, `architecture_tests.rs`
  extension) acceptable under "no new gates" is inferred, not confirmed. SW-F327's requirement
  suggests yes for that feature.
- **CodeGraph `in_progress` items** listed in §4.2 (e.g. SW-4802..4821, SW-5586) may be stale
  Shortcut imports rather than live work. I did not check their claims, attempts or handoffs.
- **The per-crate counts** (43 mlx-gen-* dirs, 40 candle-gen-* dirs) include core/catalog/testkit
  dirs matching the prefix glob, so actual provider-family counts are somewhat lower. Not refined.
- **Toolchain version.** SW `check.yml` pins Rust "1.98.0" while a memory note cites rustup 1.97.1
  vs Homebrew 1.99. The current local toolchain alignment was not verified.
