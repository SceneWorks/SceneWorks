# SceneWorks Multi-Repository Rearchitecture Plan

> **Status:** Inference Phases 0–5 are implemented, released, and locally validated at
> `runtime-2026.07.0`. Hosted SceneWorks validation uses its configured scoped private-repository
> read secret. Product-repository consolidation and legacy-repository archival remain separate
> decisions.
>
> **Date:** 2026-07-12
>
> **Scope:** The SceneWorks organization repositories checked out together in the
> parent workspace: `SceneWorks`, `ChatWorks`, `SoundWorks`, `ui`, `core-gen`,
> `core-llm`, `llm-engine-core`, `mlx-gen`, `candle-gen`, `mlx-llm`, `candle-llm`,
> and `mlx-rs`.
>
> **Planning convention:** All effort estimates are active Codex/agent execution
> time, not human engineering time. CI queueing, GPU-runner availability,
> signing/notarization, downloads, and long-running real-weight validation can add
> wall-clock latency without adding active execution time.

## 1. Executive Decision

Reorganize the current repositories into two primary monorepos plus one deliberate
upstream-derived fork:

1. **`sceneworks-inference`** — Apache-licensed inference contracts, MLX and
   Candle engines, media providers, fixtures, conformance suites, model metadata,
   and platform runtime bundles.
2. **`sceneworks-studio`** — SceneWorks, ChatWorks, SoundWorks, shared UI,
   product protocols, shared application foundations, and product build tooling.
3. **`mlx-rs`** — retained separately while it remains an upstream-derived fork
   with an independent upstream synchronization and native-build lifecycle.

The inference consolidation is the first and non-optional part of this plan. The
product consolidation follows only after a licensing/access-control checkpoint.
If a mixed-license product repository is undesirable, SceneWorks, ChatWorks, and
SoundWorks may remain separate repositories while consuming the same consolidated
inference runtime and a properly released UI package.

This plan intentionally does **not** create a single organization-wide mega-repo.
A repository is treated as a transaction, release, and governance boundary; crate
and package boundaries continue to provide code modularity within each repository.

## 2. Problem Statement

The current repository boundaries do not match the code's change boundaries.
Contracts, two tensor backends, dozens of provider crates, product integration,
and shared UI frequently move together but are committed and validated separately.
The resulting coordination mechanisms are Git SHA ledgers, source-identity rules,
manual skew checks, repeated provider dependency declarations, and cross-repository
PR sequences.

The principal problems are:

- The backend-neutral generative-media contract is implemented inside `mlx-gen`,
  while `candle-gen` and SceneWorks fetch it from that backend repository by SHA.
- `core-gen` and `llm-engine-core` are placeholder repositories while the working
  contracts live elsewhere.
- Provider discovery uses link-time `inventory` registries. Cargo resolving two
  source identities for the same contract can split registration and lookup into
  different registries, causing runtime provider-resolution failures.
- SceneWorks directly declares approximately 31 MLX provider dependencies and 28
  Candle provider dependencies in a roughly 1,900-line worker manifest.
- MLX and Candle have 28 corresponding media-provider families, but changes and
  tests cannot land atomically.
- Shared UI extraction is incomplete: consumers and the UI source repository have
  diverged versions, SceneWorks still owns duplicate shell/theme styles, and
  SoundWorks retains overlapping tokens.
- CI ownership is incomplete. Some contract, backend, product, and UI repositories
  rely on consumer integration to discover breakage.
- Large product modules and worker modules concentrate unrelated responsibilities;
  repository consolidation alone will not make them maintainable.
- Generated architecture documentation can become materially stale and currently
  contradicts the native Rust worker architecture.

## 3. Goals

### 3.1 Primary goals

- Make one coherent inference change land as one atomic commit and one PR.
- Give backend-neutral contracts an actual neutral home.
- Eliminate internal cross-repository Git dependency skew inside the inference
  stack.
- Make provider inclusion explicit, validated, and inspectable.
- Reduce SceneWorks product integration to a small number of runtime bundle
  dependencies.
- Run contract and conformance tests at the component boundary before consumer
  repositories update.
- Co-locate MLX/Candle provider twins, shared fixtures, and cross-backend tests
  without forcing inappropriate tensor abstractions.
- Preserve independent product artifacts and versions.
- Preserve usable Git history and maintain an auditable old-SHA-to-new-SHA map.
- Establish rollback points at every migration boundary.

### 3.2 Secondary goals

- Establish one source of truth for the studio design tokens and shared primitives.
- Standardize build tooling and generated Rust-to-TypeScript protocol contracts.
- Split worker hosting, platform inference, and utility execution into explicit
  components.
- Make model capability metadata authoritative for runtime discovery and product
  policy.
- Reduce large integration modules along capability boundaries.

## 4. Non-Goals

The first migration must not also become a rewrite. Specifically:

- Do not redesign `gen-core` or `core-llm` while importing their histories.
- Do not merge image/video generation and LLM serving into one universal contract.
- Do not force MLX and Candle tensor implementations through generic tensor traits.
- Do not rename every crate during the repository import.
- Do not convert all frontend JavaScript to TypeScript as part of product import.
- Do not change model outputs, default samplers, quantization formats, weight keys,
  runtime capability policy, or product-visible engine identifiers.
- Do not replace all current product persistence in the repository migration.
- Do not require every platform backend to compile in one Cargo invocation.
- Do not rewrite Git history in the existing repositories.
- Do not delete or make existing repositories read-only until their consumers have
  cut over and rollback tags have been verified.

## 5. Architectural Principles

### 5.1 Transaction boundaries follow change coupling

Code that must be revised, validated, and released together belongs in the same
repository. Internal package boundaries remain narrow even when repository
boundaries become broader.

### 5.2 Mechanical migration precedes semantic refactoring

Each source tree is first imported with behavior and package names preserved.
Path-dependency conversion and CI stabilization happen before registries, bundles,
provider layout, or APIs are changed.

### 5.3 Model-first organization, backend-specific implementation

Corresponding MLX and Candle providers live beside one another with shared fixtures
and behavioral expectations. Their numerical source remains separate unless a
specific piece of code is proven to be tensor-neutral.

### 5.4 Explicit composition over linker side effects

The application should be able to point to the source that lists every shipped
provider. Provider factories are assembled by platform bundle crates; registration
must not depend on a linker retaining otherwise-unused crates.

### 5.5 Capability and policy are different layers

Inference descriptors report what an implementation can technically do. Product
policy determines whether a capability is installed, validated, licensed, or
advertised. The product policy layer must not redefine backend capabilities from
scratch.

### 5.6 Platform matrices are named configurations

`runtime-macos`, `runtime-cuda`, and `runtime-cpu` are supported build products.
`--workspace --all-features` is not a supported configuration because MLX, CUDA,
and native packaging features are not meaningfully additive.

### 5.7 Every phase is reversible

A phase must have a recorded input state, explicit validation, a cutover point, and
a rollback mechanism. The old topology remains available until the replacement has
passed the same production gates.

## 6. Current Repository Disposition

| Current repository | Planned disposition | Notes |
|---|---|---|
| `core-gen` | Archive after inference cutover | Placeholder only; the working implementation is imported from `mlx-gen/gen-core` first. |
| `llm-engine-core` | Archive after inference cutover | Placeholder only; `core-llm` is the working contract. |
| `core-llm` | Import into `sceneworks-inference` | Preserve package name and history initially. |
| `mlx-llm` | Import into `sceneworks-inference` | MLX platform engine. |
| `candle-llm` | Import into `sceneworks-inference` | Candle CPU/CUDA/Metal engine. |
| `mlx-gen` | Import into `sceneworks-inference` | Split contract, backend foundations, and model providers after mechanical import. |
| `candle-gen` | Import into `sceneworks-inference` | Convert its gen-core Git dependency to a workspace path. |
| `mlx-rs` | Keep separate initially | True upstream-derived fork; establish tags/releases instead of floating or personal-fork URLs. |
| `SceneWorks` | Import into `sceneworks-studio` after checkpoint | Remains the initial integration owner during inference migration. |
| `ChatWorks` | Import into `sceneworks-studio` after checkpoint | Independent product release; consumes LLM runtime bundle. |
| `SoundWorks` | Import into `sceneworks-studio` after checkpoint | Preserve PolyForm license boundaries and audio-specific domain. |
| `ui` | Import into `sceneworks-studio` | Internal consumers use workspace dependency; external publication remains possible. |

## 7. Target Inference Repository

The exact names are provisional, but the ownership structure is authoritative:

```text
sceneworks-inference/
  Cargo.toml
  Cargo.lock
  rust-toolchain.toml
  deny.toml
  README.md
  CONTRIBUTING.md
  docs/
    architecture/
    compatibility/
    releases/

  crates/
    contracts/
      gen-core/
      gen-core-testkit/
      llm-core/
      llm-core-testkit/

    backends/
      mlx-media/
      candle-media/
      mlx-llm/
      candle-llm/

    providers/
      z-image/
        mlx/
        candle/
        fixtures/
      flux/
        mlx/
        candle/
        fixtures/
      qwen-image/
        mlx/
        candle/
        fixtures/
      wan/
        mlx/
        candle/
        fixtures/
      ...

    catalog/
      model-catalog/
      capability-validation/

    bundles/
      runtime-macos/
      runtime-cuda/
      runtime-cpu/

  xtask/
  scripts/
```

### 7.1 Contract rules

- Contract crates have no tensor dependencies.
- Contract crates compile and test on Linux, Windows, and macOS.
- Testkits remain separate crates so conformance-only dependencies do not enter
  runtime graphs.
- Contract changes require both relevant backends to compile and pass synthetic
  conformance before merge.
- Published identifiers and serialized request fields remain backward-compatible
  during the initial migration.

### 7.2 Provider rules

- Each provider exposes one or more ordinary factories and descriptors.
- A provider crate does not mutate a global registry as a side effect of linking.
- Each provider family owns its shared fixtures and backend parity expectations.
- Shared code may be extracted only when it is demonstrably tensor-neutral or a
  stable model-domain concept.
- Weight-name compatibility and existing parity tolerances are preserved.
- Backend-specific deviations are documented beside the family, not in product
  pin comments.

### 7.3 Explicit registry design

The target registry is an ordinary value assembled by a bundle:

```rust
pub struct ProviderFactory {
    pub descriptor: fn() -> ProviderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
}

pub fn registry() -> Result<Registry> {
    Registry::builder()
        .add(z_image_mlx::factory())
        .add(flux_mlx::schnell_factory())
        .add(flux_mlx::dev_factory())
        .build()
}
```

The builder validates, in all profiles:

- Duplicate provider IDs.
- Invalid identifier formats.
- Duplicate aliases.
- Descriptor/backend mismatches.
- Impossible capability combinations.
- Required loading/probing hooks.
- Product-consumed descriptor schema compatibility.

Compatibility functions such as `gen_core::load` and `core_llm::load_for_model`
may remain temporarily, backed by a supplied or initialized explicit registry.
Global implicit registration is removed only after all consumers use bundles.

### 7.4 Runtime bundles

Bundle crates are the supported composition boundary:

- `runtime-macos`: MLX media providers, MLX LLM provider, and macOS utilities.
- `runtime-cuda`: Candle CUDA media providers and Candle CUDA LLM provider.
- `runtime-cpu`: tensor-free or CPU utility capabilities and supported CPU LLM
  paths where appropriate.

Each bundle:

- Owns the explicit provider list.
- Exposes a validated registry and machine-readable capability catalog.
- Has a minimal smoke binary/test.
- Defines supported target triples and external native prerequisites.
- Produces an SBOM/dependency inventory in release CI.
- Is versioned as part of one compatible inference release train.

### 7.5 `mlx-rs` boundary

`mlx-rs` remains separate during this plan because it contains upstream-derived
history, an `mlx-c` submodule, native build logic, and patches that may be rebased or
upstreamed independently.

The improved consumption contract is:

- Canonical organization-owned URL, not a personal-fork URL.
- Immutable release tags or published internal crate versions.
- One declared version in the inference workspace.
- A documented mapping from the tag to MLX/MLX-C upstream versions and local
  patches.
- A compatibility smoke in `sceneworks-inference` before accepting a new tag.

## 8. Target Product Repository

Subject to the licensing/access checkpoint:

```text
sceneworks-studio/
  Cargo.toml
  Cargo.lock
  pnpm-workspace.yaml
  pnpm-lock.yaml
  rust-toolchain.toml

  apps/
    sceneworks/
      web/
      desktop/
      api/
      workers/
    chatworks/
      web/
      desktop/
      server/
    soundworks/
      web/
      desktop/

  crates/
    studio-protocol/
    app-foundation/
    sceneworks-domain/
    chatworks-domain/
    soundworks-domain/

  packages/
    ui/
    eslint-config/
    test-utils/

  schemas/
  scripts/
  docs/
```

### 8.1 Licensing rule

The repository must use per-package licensing metadata and a root licensing policy:

- SceneWorks: AGPL-3.0-or-later.
- SoundWorks: PolyForm Noncommercial 1.0.0.
- ChatWorks and shared UI: Apache-2.0 unless deliberately changed.
- Every distributable crate/package declares its own license.
- SPDX headers or REUSE-compatible metadata identify ambiguous files.
- Shared code accepts only a license compatible with all intended consumers.

If repository visibility, contributor policy, or mixed-license presentation makes
this unacceptable, retain separate product repos. This changes Phase 7 only; it
does not block the inference plan.

### 8.2 Shared product foundations

Candidates for sharing, after comparing real implementations:

- Job IDs, states, progress, cancellation, and terminal-error envelopes.
- Artifact/file references and portable relative-path rules.
- Model installation state and capability summaries.
- Provenance/source-reference envelopes.
- Theme and preference handling.
- Tauri shell startup, logging, and updater helpers.
- Generated API/protocol types.
- Test fixtures and fake runtime boundaries.

Explicitly product-specific:

- Scene/image/video generation recipes and project semantics.
- Chat messages, conversations, tools, and OpenAI server behavior.
- Audio assets, voices, compositions, stems, loops, and audio rights policy.
- Product workflows and screen state.

### 8.3 UI consolidation

- Import `ui` as `packages/ui` without redesigning it.
- Resolve the current source/package version discrepancy before further release.
- Internal applications use `workspace:*` rather than published versions.
- Keep publication support for consumers outside the monorepo.
- Migrate SceneWorks tokens and shell styles first.
- Migrate SoundWorks duplicated tokens next.
- Move primitives only after consumers have tests or stories for their behavior.
- Add visual or screenshot regression coverage for tokens, theme variants, and
  core primitives.
- JavaScript and TypeScript coexist; type declarations are generated or maintained
  at the package boundary until gradual conversion is justified.

## 9. Target Worker Architecture

Replace one feature-heavy worker role with shared hosting plus thin binaries:

```text
studio-protocol
       |
  worker-host
   /    |    \
utility mlx   cuda
worker  worker worker
```

### 9.1 `worker-host`

Owns:

- Job claim and lease protocol.
- Heartbeats.
- Cancellation and shutdown.
- Progress/event reporting.
- Structured logging and correlation IDs.
- Terminal-state guarantees.
- Worker capability advertisement envelope.
- Common retry/error classification.

### 9.2 Thin worker binaries

- `utility-worker`: downloads, imports, conversion orchestration, FFmpeg, file and
  cache operations, and other non-GPU work.
- `mlx-worker`: SceneWorks job adapters plus `runtime-macos`.
- `cuda-worker`: SceneWorks job adapters plus `runtime-cuda`.

The installer may still ship the appropriate workers as one product. The split is
about compilation, process isolation, failure containment, and clear capability
ownership, not imposing new user-visible deployment complexity.

### 9.3 Job-family modules

The SceneWorks adapter layer is decomposed into:

```text
jobs/
  image/
  video/
  training/
  captioning/
  analysis/
  tracking/
  preprocessing/
  utility/
```

Each job family owns request translation, eligibility checks specific to the
product, progress translation, and output persistence coordination. Model loading,
tensor execution, and provider capability declarations remain in the inference
repository.

## 10. Migration Safety Rules

These rules apply throughout execution:

1. Do not combine source import and behavioral refactoring in one commit.
2. Tag or record every source repository SHA before importing it.
3. Preserve original author/date metadata through filtered-history imports.
4. Produce an old-to-new commit mapping file for each imported repository.
5. Keep existing repositories writable until cutover validation passes.
6. Do not force-push existing repository history.
7. Do not enable MLX and CUDA features in the same catch-all CI command.
8. Do not publish from both old and new locations after cutover.
9. Keep provider IDs, weight keys, serialized payloads, and default behavior stable
   until compatibility tests prove otherwise.
10. Any migration failure must be reproducible from the recorded release set.
11. Existing user worktrees and unrelated changes are preserved during all moves.
12. Large generated or binary fixtures require an explicit storage decision before
   history import finalization.

## 11. Phased Execution Plan

### Phase 0 — Baseline and governance

**Agent time:** 2–4 hours.

**Objective:** Establish a reproducible, reviewable starting point.

Tasks:

- Create `release-set.toml` or equivalent containing repository URL, branch, full
  SHA, dirty-state assertion, toolchain, and relevant published package versions.
- Capture current Cargo package IDs for SceneWorks MLX and Candle graphs.
- Capture current provider descriptor/catalog snapshots for each supported bundle.
- Capture current UI package/source versions and consumer lockfile resolutions.
- Record current CI lanes and required self-hosted runner labels.
- Define canonical future repository URLs and ownership.
- Add or enable build/test CI for components currently validated only by consumers.
- Write ADRs for repository topology, explicit registry, and product licensing.
- Identify large fixture objects and choose Git LFS, artifact-store, or retained Git
  storage per fixture category.

Validation:

- Every repository in scope has a clean-state or documented-dirty-state record.
- All SHAs resolve from their canonical remotes.
- Existing release/build commands can be traced to the recorded SHAs.
- Contract and synthetic tests pass at the baseline.
- Supported MLX and Candle worker builds pass, subject to runner availability.

Rollback:

- No cutover occurs. Delete only newly created planning/bootstrap branches if the
  phase is abandoned.

Exit gate:

- Baseline release set is committed and independently reproducible.

### Phase 1 — Mechanical inference history import

**Agent time:** 3–6 hours.

**Objective:** Create the new repository with history preserved and source trees
unchanged in behavior.

Tasks:

- Create the empty `sceneworks-inference` repository and bootstrap branch.
- Import filtered histories for `core-llm`, `mlx-llm`, `candle-llm`, `mlx-gen`, and
  `candle-gen` into temporary non-final directories.
- Import relevant placeholder histories for archival traceability if desired.
- Store filter-repo commit maps under `docs/migration/commit-maps/`.
- Add a provisional root README, license policy, CODEOWNERS, and contributor guide.
- Establish a root Cargo workspace without changing package source content.
- Preserve existing crate/package names.

Validation:

- Representative old commits map to expected new commits.
- `git log --follow` remains useful for imported files.
- The imported trees match their source SHAs byte-for-byte, excluding deliberate
  path and repository-metadata adjustments.
- No source repository has been mutated.

Rollback:

- Delete the new bootstrap branch/repository; existing repositories remain the
  source of truth.

Exit gate:

- History and source equivalence review passes.

### Phase 2 — Workspace dependency normalization

**Agent time:** 4–8 hours.

**Objective:** Make the imported inference graph self-contained.

Tasks:

- Move the working `gen-core` and its testkit to the neutral contract directory.
- Move/import `core-llm` and its testkit to the neutral contract directory.
- Convert all intra-inference Git dependencies to workspace path dependencies.
- Centralize shared dependency versions under `[workspace.dependencies]` where it
  reduces actual skew risk.
- Establish one committed Cargo lockfile and one Rust toolchain policy.
- Retain target-specific and backend feature behavior.
- Add a gate rejecting internal SceneWorks/michaeltrefry Git URLs from inference
  manifests.
- Keep `mlx-rs` as the only expected SceneWorks-controlled external Git or tagged
  dependency until it has a published release.
- Normalize repository metadata without renaming public packages.

Validation:

- `cargo metadata` resolves one instance of each contract package.
- Contract crates compile without tensor backends.
- Candle CPU tests pass.
- MLX tests pass on the supported macOS runner.
- CUDA build/test passes on supported runners.
- No internal path depends back on an old inference repository.
- Existing provider IDs and descriptor snapshots are unchanged.

Rollback:

- Revert the normalization commits in the new repository; old repositories remain
  authoritative.

Exit gate:

- The new inference workspace is behaviorally equivalent and internally uses path
  dependencies.

### Phase 3 — Unified inference CI and release train

**Agent time:** 3–6 hours.

**Objective:** Make the new repository independently trustworthy before consumers
cut over.

Tasks:

- Create contract, Candle CPU, MLX, CUDA, formatting, lint, supply-chain, and docs
  workflows.
- Define package/path filters that expand through dependency relationships.
- Separate fast synthetic checks from real-weight and long-running GPU suites.
- Add nightly/manual real-weight matrices with explicit fixture/model revisions.
- Create a runtime release manifest listing all included crate versions and the
  `mlx-rs` tag/SHA.
- Define tag format, initially `runtime-YYYY.MM.patch` or semver equivalent.
- Produce source/archive artifacts and SBOMs without publishing crates yet.

Validation:

- Contract change exercises both relevant backend compile/conformance lanes.
- Provider-only change exercises its backend and platform bundle.
- CI never relies on `--all-features` as a universal matrix.
- A dry-run runtime tag can be consumed by a small external smoke project.

Rollback:

- The repository remains non-authoritative; no product dependency has changed.

Exit gate:

- A tagged inference release candidate passes all current production-equivalent
  gates.

### Phase 4 — Explicit registries and platform bundles

**Agent time:** 12–24 hours.

**Objective:** Remove source-identity-sensitive provider discovery and product-side
provider assembly.

Tasks:

- Implement ordinary registry builders for generative-media and LLM providers.
- Define factory/descriptor APIs and release-profile validation.
- Create compatibility adapters for existing `load` and `load_for_model` APIs.
- Add `runtime-macos`, `runtime-cuda`, and `runtime-cpu` bundle crates.
- Migrate provider registrations incrementally from `inventory` to explicit lists.
- Add bundle catalog snapshots and duplicate/capability tests.
- Ensure providers that are intentionally bespoke utilities have explicit bundle
  ownership even if they do not implement the general `Generator` trait.
- Separate technical capability descriptors from product validation/advertising
  policy.
- Remove linker-retention `use crate as _` patterns after their provider migrates.
- Remove `inventory` only after no supported consumer depends on it.

Validation:

- Every provider visible in old registry snapshots exists in the expected bundle.
- No unexpected provider is advertised.
- Duplicate IDs fail deterministically in release tests.
- Model-first LLM resolution maintains existing architecture/capability behavior.
- MLX/Candle conformance and real-weight smoke coverage remain green.
- Bundle smoke binaries enumerate and probe the expected provider set.

Rollback:

- Compatibility adapters allow a commit-level rollback to `inventory` while the
  repository and path-dependency consolidation remain intact.

Exit gate:

- All supported provider discovery flows use explicit validated registries.

### Phase 5 — SceneWorks and ChatWorks consumer cutover

**Agent time:** 5–10 hours.

**Objective:** Make products consume the consolidated runtime without changing
product behavior.

Tasks:

- Replace SceneWorks' individual MLX provider dependencies with `runtime-macos`.
- Replace SceneWorks' individual Candle provider dependencies with `runtime-cuda`.
- Depend on neutral contract packages from the same inference tag/source.
- Replace direct force-linking with bundle registry initialization.
- Replace ChatWorks' separate core/MLX/Candle Git dependencies with the relevant
  runtime bundles and contract from one inference tag.
- Preserve compile-time backend selection and current packaging behavior.
- Reduce or delete obsolete pin-skew scripts and manifest comment ledgers.
- Add product-to-runtime compatibility smoke tests.
- Record the exact old release set and new runtime tag in cutover documentation.

Validation:

- SceneWorks default Linux/CPU checks pass.
- SceneWorks macOS MLX worker tests pass.
- SceneWorks Windows/Linux CUDA worker tests pass.
- Desktop and server package smokes pass.
- ChatWorks macOS and Candle builds/tests pass.
- Provider capability snapshots match pre-cutover baselines.
- No product manifest lists individual media provider Git dependencies.

Rollback:

- Revert product dependency commits to the recorded release-set SHAs.
- Keep the inference release tag immutable even if cutover is reverted.

Exit gate:

- Products ship or produce release-equivalent artifacts against the new runtime.

### Phase 6 — Model-first provider relocation and fixture cleanup

**Agent time:** 8–16 hours.

**Objective:** Make cross-backend model maintenance local and visible.

Tasks:

- Relocate MLX/Candle provider pairs under family directories in small batches.
- Move shared fixtures and family documentation beside each pair.
- Deduplicate identical tokenizer/config/fixture assets.
- Move large fixtures to the storage mechanism selected in Phase 0.
- Establish per-family cross-backend descriptor and request-validation tests.
- Extract proven tensor-neutral helpers without altering numerical leaves.
- Add ownership metadata per family/backend.

Validation:

- Relocation commits are path-only before any extraction commits.
- Parity and conformance results match before/after each family batch.
- CI path filtering still selects all affected packages.
- Runtime bundles enumerate identical provider sets after relocation.

Rollback:

- Revert individual family batches independently.

Exit gate:

- All common provider families have one family-level maintenance home.

### Phase 7 — Product repository checkpoint and optional consolidation

**Agent time:** 8–14 hours for import/tooling; 12–30 additional hours for actual UI
and shared-foundation adoption.

**Objective:** Consolidate products only if governance permits and concrete sharing
benefits justify it.

Checkpoint questions:

- Can AGPL, PolyForm Noncommercial, and Apache code coexist under clear per-package
  licensing and contributor policy?
- Do all products have the same repository visibility and contributor access?
- Is one issue/PR namespace acceptable?
- Can independent product releases remain operationally clear?

If **yes**:

- Import SceneWorks, ChatWorks, SoundWorks, and UI histories into temporary paths.
- Create one Cargo workspace and one pnpm workspace.
- Preserve per-product application versions and release workflows.
- Add per-package licenses and root licensing guidance.
- Convert internal UI consumption to `workspace:*`.
- Standardize Node/toolchain/lint/test configuration incrementally.
- Generate shared protocol types from authoritative Rust/schema definitions.
- Migrate UI tokens, shell primitives, and common app foundations in separate PRs.

If **no**:

- Keep the three product repositories.
- Repair and independently release `@sceneworks/ui` with full CI.
- Remove the stated requirement that consumers move in lockstep; use semver and
  compatibility tests instead.
- Create a small shared protocol crate/package only where multiple products have a
  real runtime consumer.

Validation:

- Each product can build/test/package independently.
- A change limited to one product does not require unrelated GPU or packaging jobs.
- License scanning attributes every distributable file correctly.
- Internal UI source and consumer versions cannot diverge.

Rollback:

- Existing product repositories remain authoritative until the product cutover.
- If governance fails, use the separate-product fallback without reverting the
  inference migration.

Exit gate:

- Product topology is explicitly accepted and all product release lanes pass.

### Phase 8 — Worker and product module decomposition

**Agent time:** 30–70 hours, intentionally performed after topology stabilization.

**Objective:** Reduce internal maintenance hotspots exposed by the current reviews.

Tasks:

- Extract `worker-host` and thin utility/MLX/CUDA binaries.
- Split image, video, training, media, tracking, and preprocessing job adapters.
- Separate SceneWorks project storage, job storage, asset storage, and model catalog
  services behind explicit interfaces.
- Remove duplicated backend routing lists in favor of bundle/catalog queries.
- Split large React screens into route/screen components and focused state hooks.
- Make SoundWorks' provider catalog drive actual execution rather than parallel
  decorative and hard-coded catalogs.
- Split SoundWorks runtime hosting, queueing, provider adapters, persistence, and
  overview/view-model construction.
- Add dependency-boundary tests and deny forbidden imports.

Validation:

- No product-visible behavior changes without explicit tests and release notes.
- Job lifecycle, cancellation, heartbeat, and persistence contract tests pass.
- Platform packaging retains the same user-visible installation shape.
- Module splits reduce file/responsibility size rather than merely moving code into
  includes.

Rollback:

- Each subsystem split is an independent change after the repository cutovers.

Exit gate:

- High-risk god modules no longer own unrelated routing, execution, and persistence
  responsibilities.

### Phase 9 — Old repository archival

**Agent time:** 2–4 hours.

**Objective:** End dual ownership without losing traceability.

Tasks:

- Create final immutable tags on every old repository.
- Merge README migration notices with new paths and commit-map links.
- Disable old publish/release workflows.
- Mark repositories read-only/archive them after a defined observation period.
- Update organization links, issue templates, docs, package metadata, and security
  reporting locations.
- Verify no active manifest or lockfile consumes an archived repository.

Validation:

- Organization-wide search finds no unsupported old Git dependency URLs.
- Published packages point to the new source repository.
- Old commit URLs remain available for audit.
- A clean checkout using only new authoritative repositories can build supported
  products.

Rollback:

- Re-enable an old repository only for an explicitly documented emergency patch;
  forward-port the patch immediately to the new source of truth.

Exit gate:

- There is one authoritative home for every active package.

## 12. Effort Summary

| Milestone | Active agent time | Cumulative range |
|---|---:|---:|
| Baseline and mechanical history import | 5–10 hours | 5–10 hours |
| Self-contained inference workspace and CI | 7–14 hours | 12–24 hours |
| Explicit registries and platform bundles | 12–24 hours | 24–48 hours |
| Product consumer cutover | 5–10 hours | 29–58 hours |
| Model-first provider layout | 8–16 hours | 37–74 hours |
| Product import/tooling, if approved | 8–14 hours | 45–88 hours |
| UI/shared-foundation adoption | 12–30 hours | 57–118 hours |
| Major internal decomposition | 30–70 hours | 87–188 hours |
| Archival and closeout | 2–4 hours | 89–192 hours |

Recommended initial commitment: complete through **Phase 5**, approximately
**29–58 active agent hours**. This delivers the primary correctness and coordination
benefits without coupling success to the optional product monorepo or broad internal
module refactoring.

## 13. CI Design

### 13.1 Required inference lanes

| Lane | Runner | Trigger |
|---|---|---|
| Formatting and manifest policy | Linux | Every PR |
| Contract build/clippy/test | Linux, Windows, macOS | Contract or shared-policy changes |
| Candle CPU | Linux and Windows | Candle/core/provider changes |
| MLX synthetic/parity | macOS ARM | MLX/core/provider changes |
| CUDA compile/synthetic | Windows/Linux CUDA | CUDA/core/provider changes |
| Bundle catalog smoke | Per supported platform | Bundle/provider/catalog changes |
| Real-weight provider smoke | Self-hosted/manual/nightly | Provider/runtime release candidates |
| Supply-chain/SBOM/license | Linux | Every release candidate |

### 13.2 Required product lanes

- Web lint/test/build per affected app.
- Rust domain/API/worker tests per dependency graph.
- Tauri compile tests per target.
- Platform package smoke for release candidates.
- Inference tag compatibility smoke.
- UI unit and visual/theme regression tests.
- Schema generation drift check.
- License-policy validation.

### 13.3 CI selection rules

- Path filtering alone is insufficient; changes to a dependency trigger downstream
  packages.
- Contract changes always trigger both supported backend families.
- Bundle changes always trigger catalog and consumer smoke tests.
- Documentation-only changes skip native GPU builds unless they modify generated
  catalogs or release manifests.
- Real-weight tests are revision-pinned and explicitly report skipped prerequisites.
- Required PR checks use synthetic fixtures; large/slow hardware validation gates
  release candidates rather than every documentation or isolated UI change.

## 14. Release and Versioning Model

### 14.1 Inference release train

Initially use one immutable inference tag containing a compatible set of packages:

```text
runtime-2026.07.0
runtime-2026.07.1
```

The release manifest records:

- Contract package versions.
- Backend package versions.
- Provider package commits/versions.
- Runtime bundle versions.
- `mlx-rs`, MLX, MLX-C, Candle, CUDA, and toolchain versions.
- Supported target triples.
- Catalog/schema version.
- Required migrations or compatibility notes.

Products depend on a few facade/bundle packages from this one tag. Publishing every
provider crate is deferred until there is a real external-consumer need.

### 14.2 Product releases

- SceneWorks, ChatWorks, and SoundWorks retain independent application versions.
- A studio monorepo does not imply one synchronized application version.
- Path-filtered workflows build only affected products plus downstream shared-code
  consumers.
- UI may publish semver releases for external consumers, while internal consumers
  use workspace source.

## 15. History Migration Procedure

The precise command script is produced in Phase 1 and reviewed before execution.
The intended method is:

1. Create fresh mirror clones of each source repository.
2. Run `git filter-repo --to-subdirectory-filter <temporary-path>` in each mirror.
3. Save each generated commit map.
4. Add each filtered repository as a temporary remote in the destination.
5. Merge the imported history with explicit import commits.
6. Verify tree equality against the recorded source SHA.
7. Only then move paths mechanically within the destination repository.

Existing repositories are never filter-rewritten or force-pushed. The new repository
necessarily has different commit IDs; commit maps and final source tags preserve
traceability.

## 16. Binary Fixture Strategy

The inference repository will otherwise inherit substantial and growing binary
history, including large safetensors goldens and duplicated tokenizer/fixture assets.

Classify fixtures as:

- **Tiny synthetic fixtures:** keep directly in Git.
- **Moderate stable fixtures:** keep in Git or Git LFS based on total clone impact.
- **Large real-model goldens:** store in a checksummed artifact bucket/release asset
  and fetch only in the relevant test lane.
- **Derived fixtures:** generate deterministically in CI when generation is cheaper
  than long-term storage.

Every external fixture has:

- Immutable content checksum.
- Generator/source revision.
- License and redistribution status.
- Owning provider/test.
- Cache key and offline failure message.

No fixture migration may silently weaken required PR coverage.

## 17. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cargo feature unification enables invalid backend combinations | Medium | High | Named bundle builds; never use a universal all-features gate. |
| History import loses traceability | Low | High | Mirror clones, filter-repo maps, tree equality checks, immutable old tags. |
| Registry migration drops a provider | Medium | High | Baseline descriptor snapshots and bundle catalog equality tests. |
| Provider outputs change during move | Low | High | Path-only commits, parity fixtures, no refactor during import. |
| Mixed product licenses create ambiguity | Medium | High | Explicit checkpoint, per-package licenses, REUSE policy, separate-product fallback. |
| CI becomes too slow or expensive | Medium | Medium | Dependency-aware lanes, synthetic PR tests, real-weight release/nightly tests. |
| Large fixtures make new clones unwieldy | High | Medium | Deduplicate and classify fixtures before final history layout. |
| Old and new repositories both publish | Medium | High | Cutover checklist, immutable tags, disable old workflows before archival. |
| Personal/canonical Git URLs still split sources | Medium | High | Manifest lint and canonical organization-owned dependency URLs. |
| Product cutover couples to unfinished refactors | Medium | High | Compatibility facades; postpone decomposition until after cutover. |
| Shared abstractions erase backend-specific correctness | Medium | High | Share only proven tensor-neutral code; preserve separate numeric leaves. |
| Generated docs become stale again | High | Medium | Generate topology/catalog docs in CI and label hand-authored vs generated authority. |

## 18. Rollback Strategy

### Before product cutover

The existing repositories remain authoritative. Failure requires no consumer change;
the new repository can be corrected or discarded.

### During product cutover

Each product dependency update is an isolated commit. Revert it to the recorded
release-set SHAs. Do not delete the new inference tag; immutable failed cutover tags
remain auditable.

### After product cutover but before archival

Revert products to the last known old release set if a severe runtime regression is
found. Fix forward in the new repository, issue a new inference tag, and repeat the
cutover. Avoid patching both homes.

### After archival

Unarchive only for emergency investigation or an unavoidable patch to a still-shipped
old release. The new repository remains the source of truth and receives the forward
port immediately.

## 19. Completion Metrics

### Inference topology

- Zero internal Git dependencies among inference packages.
- One resolved instance of each contract crate per supported product graph.
- SceneWorks has no individual `mlx-gen-*` or `candle-gen-*` Git declarations.
- ChatWorks does not independently assemble contract and backend Git sources.
- Provider bundles validate duplicate IDs and descriptor consistency in release mode.
- One inference story can modify contract, MLX, Candle, fixtures, and bundle in one
  PR.

### CI and releases

- Every active package has an owning CI lane.
- Every release records the exact runtime, tensor library, and toolchain set.
- No product consumes a mutable branch dependency for a released build.
- Old repositories cannot publish after archival.

### Product architecture

- One authoritative theme/token implementation.
- Product domains remain independent and do not depend on tensor backends.
- Shared protocol types are generated or tested for drift.
- Worker host behavior is tested independently from MLX/CUDA execution.
- Large modules are decomposed by responsibility with no behavior loss.

### Documentation

- Generated repository/package graphs are current and CI-verified.
- Historical architecture documents are clearly marked historical.
- Each active subsystem has one named authoritative architecture document.

## 20. Work Item Breakdown

These identifiers are local planning IDs until corresponding tracker stories exist.

| ID | Work item | Depends on | Agent time |
|---|---|---|---:|
| MRP-001 | Record baseline release set and clean-state inventory | — | 1–2 h |
| MRP-002 | Capture provider/package/catalog compatibility snapshots | MRP-001 | 1–2 h |
| MRP-003 | Decide canonical repository names, licenses, and ownership | MRP-001 | 0.5–1 h |
| MRP-004 | Classify large binary fixtures | MRP-001 | 1–2 h |
| MRP-010 | Bootstrap inference repository | MRP-003 | 0.5–1 h |
| MRP-011 | Filter/import contract and LLM histories | MRP-010 | 1–2 h |
| MRP-012 | Filter/import MLX/Candle media histories | MRP-010 | 2–3 h |
| MRP-013 | Verify commit maps and tree equivalence | MRP-011, MRP-012 | 1–2 h |
| MRP-020 | Establish root workspace/toolchain/lockfile | MRP-013 | 1–2 h |
| MRP-021 | Move neutral contracts and convert path dependencies | MRP-020 | 2–4 h |
| MRP-022 | Add internal-Git dependency policy gate | MRP-021 | 0.5–1 h |
| MRP-023 | Stabilize contract/Candle CPU lanes | MRP-021 | 1–2 h |
| MRP-024 | Stabilize MLX and CUDA lanes | MRP-021 | 3–6 h |
| MRP-030 | Implement explicit generative-media registry | MRP-024 | 3–6 h |
| MRP-031 | Implement explicit LLM registry | MRP-024 | 2–4 h |
| MRP-032 | Create macOS/CPU/CUDA bundles | MRP-030, MRP-031 | 3–5 h |
| MRP-033 | Migrate providers and remove linker force-links | MRP-032 | 4–9 h |
| MRP-034 | Add bundle catalog validation/snapshots | MRP-032 | 1–2 h |
| MRP-040 | Cut SceneWorks to inference bundles | MRP-033, MRP-034 | 3–6 h |
| MRP-041 | Cut ChatWorks to inference bundles | MRP-033, MRP-034 | 1–3 h |
| MRP-042 | Run production-equivalent product/package gates | MRP-040, MRP-041 | 2–4 h active |
| MRP-050 | Relocate provider families model-first | MRP-042 | 5–10 h |
| MRP-051 | Deduplicate/migrate shared fixtures | MRP-004, MRP-050 | 3–6 h |
| MRP-060 | Decide product monorepo at license/access checkpoint | MRP-042 | 0.5–1 h |
| MRP-061 | Import product/UI histories if approved | MRP-060 | 4–7 h |
| MRP-062 | Establish studio Cargo/pnpm workspaces | MRP-061 | 2–4 h |
| MRP-063 | Migrate UI source and internal consumers | MRP-062 | 8–16 h |
| MRP-064 | Extract proven shared product foundations | MRP-062 | 8–20 h |
| MRP-070 | Split worker host and platform binaries | MRP-042 | 8–16 h |
| MRP-071 | Decompose SceneWorks job-family modules | MRP-070 | 12–28 h |
| MRP-072 | Decompose product storage and frontend hotspots | MRP-062 or MRP-042 | 10–26 h |
| MRP-080 | Tag, redirect, and archive old repositories | Cutovers complete | 2–4 h |

## 21. First Execution Tranche

When execution is authorized, begin only with the following bounded tranche:

1. **MRP-001:** create the machine-readable baseline release set.
2. **MRP-002:** capture contract/provider/catalog snapshots and current dependency
   identities.
3. **MRP-003:** confirm future repository names and ownership.
4. **MRP-004:** produce the fixture storage inventory and recommendation.

This tranche makes no repository-history changes, creates no external repository,
and changes no product dependency. Its deliverables are reviewable inputs to the
first irreversible-looking operation: the filtered-history import into a new
repository.

The explicit approval checkpoint after this tranche is:

> Approve creating `sceneworks-inference` and importing filtered histories using
> the recorded baseline and commit-map procedure.

## 22. Open Decisions

These must be resolved at the indicated phase, not guessed during implementation:

1. Final repository names and canonical organization URLs — before Phase 1.
2. Whether to preserve `sceneworks-gen-core` as the package name or introduce
   `core-gen` with a compatibility shim — after Phase 5, not during import.
3. Whether `core-llm` keeps its current package name — default yes.
4. Whether large fixtures use Git LFS or a checksummed artifact store — Phase 0.
5. Initial inference tag convention — before Phase 3 release candidate.
6. Whether product repository mixed licensing/access is acceptable — Phase 7
   checkpoint.
7. Whether runtime bundles are consumed from Git tags or an internal/public crate
   registry — Git tags first unless external consumers require publication.
8. Observation period before archiving old repositories — before Phase 9.

## 23. Approval Record

Execution should record approvals here or in linked ADRs/PRs:

- [ ] Target topology approved.
- [ ] Canonical repository names approved.
- [ ] History migration procedure approved.
- [ ] Fixture storage policy approved.
- [ ] Explicit registry direction approved.
- [ ] Inference release/tag model approved.
- [ ] Product licensing/access checkpoint resolved.
- [ ] Old repository archival approved after cutover.
