# Phase 0 Migration Checkpoint

> **Status:** Complete; the Phase 1 history import was subsequently authorized and
> completed.
>
> **Completed:** 2026-07-12

## Completed work

- **MRP-001:** Recorded repository HEADs, tree IDs, tracking refs/divergence,
  remotes, versions, toolchains, lock/catalog hashes, and initial worktree state in
  [`baseline/release-set.toml`](baseline/release-set.toml).
- **MRP-002:** Captured the default and Candle-enabled SceneWorks Cargo identities,
  lockfile source forms, skew-gate results, UI source/release mismatch, model
  signatures, and worker engine map in
  [`baseline/dependency-identities.md`](baseline/dependency-identities.md) and
  [`baseline/catalog-baseline.json`](baseline/catalog-baseline.json).
- **MRP-003:** Recorded the proposed canonical repository names, organizational
  ownership, placeholder disposition, package-name policy, and `candle-gen`
  behind-tracking treatment in
  [`REPOSITORY_DECISIONS.md`](REPOSITORY_DECISIONS.md).
- **MRP-004:** Inventoried large inference fixtures, exact duplicate groups, and the
  recommended Git/artifact policy in
  [`BINARY_FIXTURE_INVENTORY.md`](BINARY_FIXTURE_INVENTORY.md).
- Added a reproducible catalog capture/drift checker at
  `scripts/rearchitecture/capture-catalog-baseline.mjs`.

## Material baseline findings

1. `candle-gen` checkout HEAD is four commits behind its tracking `origin/main`.
   The proposed import includes complete history but uses the checked-out SHA as
   the compatibility baseline.
2. The Candle-enabled SceneWorks graph resolves one `gen-core`, but two different
   `candle-llm` revisions.
3. Inference and product manifests contain 77 internal Git dependency declarations
   across seven manifests.
4. The current product catalog contains 64 model entries, including 49 with an MLX
   route block and 18 with a Candle route block; the worker has 45 explicit
   SceneWorks-to-engine rows.
5. Files at least 500 KB total 444,604,022 current-tree bytes across the five
   inference repos. Eleven exact duplicate groups account for 33,481,987
   reclaimable bytes after model-first relocation.
6. UI source and Git tag report 0.1.0 while ChatWorks and SoundWorks resolve the
   published 0.2.0 package.
7. `core-llm`, `candle-llm`, SoundWorks, and UI have no repository-local CI;
   ChatWorks has only a dependency supply-chain gate.

## Validation completed

- Catalog snapshot regeneration matches byte-for-byte.
- Catalog JSON parses.
- Release-set TOML parses.
- Node syntax check passes for the capture tool.
- Markdown/source whitespace and conflict-marker checks pass.
- `git diff --check` passes.
- Default SceneWorks gen-core skew gate passes.
- Candle-enabled SceneWorks gen-core skew gate passes.
- Skew-gate self-test passes.

No product source, runtime source, dependency declaration, repository history, or
remote repository has been changed in Phase 0.

## Phase 1 authorization record

Authorize the following as one bounded next tranche:

1. Create repository `SceneWorks/inference` with local checkout `inference/`.
2. Import full filtered histories for `core-llm`, `mlx-llm`, `candle-llm`,
   `mlx-gen`, and `candle-gen` without changing source behavior.
3. Preserve and commit filter-repo old-to-new commit maps.
4. Use the recorded checked-out SHAs as compatibility baselines, while retaining
   later reachable history such as the four newer `candle-gen` commits.
5. Preserve binary fixtures byte-for-byte during the first import.
6. Preserve current crate/package names through the first consumer cutover.
7. Retain `mlx-rs` as a separate repository.

This authorization does not yet permit product dependency cutover, provider
registry refactoring, old-repository archival, or product monorepo creation.

The tranche was approved and executed. The repository was created private rather
than adopting the original public-visibility recommendation; no later migration
instruction changes that visibility.
