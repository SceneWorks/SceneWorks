# Phase 5 Inference Consumer Cutover

> **Status:** Immutable inference release published; both product cutovers pass local validation.
> Hosted SceneWorks validation awaits the scoped private-repository read secret described below.
> **Cutover release:** `SceneWorks/inference` tag `runtime-2026.07.0`
> **Release commit:** `48cc2d87e14de0189ac4f7763fddc0a8581c2e68`

## Decision

SceneWorks and ChatWorks consume one canonical inference release instead of assembling contracts,
MLX engines, Candle engines, and media-provider repositories independently. The release tag is the
only product-level inference version. Platform and product differences are expressed as named Cargo
packages and features from that release, not as different repository revisions.

SceneWorks selects:

- `runtime-macos` with its complete media catalog on Apple platforms;
- `runtime-cuda` with its complete media catalog when `backend-candle` is enabled;
- `sceneworks-gen-core` from the same release for product-facing contract types.

ChatWorks selects exactly one LLM-only profile:

- `runtime-macos` with default features disabled on macOS;
- `runtime-cpu` with default features disabled by default on Windows/Linux;
- `runtime-cuda` with default features disabled for CUDA builds.

Each product owns one immutable runtime catalog value. Provider discovery and loading are routed
through that value; dependency presence and linker retention do not change the shipped catalog.

## Why this refactor

This was a release-boundary correction, not a repository-count cleanup. Before the migration, one
inference change routinely crossed backend-neutral contracts, MLX and Candle engines, provider
families, and two product repositories. Those pieces could not be reviewed, locked, tested, or
released atomically even though they behaved as one runtime.

The old shape imposed concrete failure modes and recurring coordination cost:

- Cargo treats the same package name from different Git source identities as different packages.
  A product could therefore compile one contract identity while a provider registered against
  another, producing a valid build with an empty or incomplete runtime registry.
- Link-time `inventory` registration made provider availability depend on which transitive crates
  happened to survive linking. Product manifests doubled as an implicit runtime catalog, so a
  dependency edit could silently change behavior.
- SceneWorks carried roughly 59 direct MLX/Candle provider declarations in an approximately
  1,900-line worker manifest. The product had to understand backend internals merely to assemble a
  supported runtime.
- MLX and Candle implementations of the same model families lived in separate histories and release
  trains. Cross-backend contract changes required synchronized commits and SHA ledger updates rather
  than one tested change.
- CI ownership followed repository location instead of change impact, so consumer integration was
  often the first place contract or provider skew became visible.

The canonical inference repository makes that coupled system one transaction: neutral contracts,
backend implementations, provider families, conformance fixtures, explicit catalogs, and platform
bundles share one lockfile and one immutable runtime release. Products select a supported bundle and
retain product-specific orchestration; they no longer assemble inference from provider internals.

The accepted price is a larger inference workspace, a wider affected-lane CI matrix, synchronized
runtime releases, and scoped credentials when public products consume the private canonical source.
That cost is explicit and testable. It replaces open-ended cross-repository skew, hidden linker
behavior, and repeated product-side assembly.

This decision deliberately stops short of an organization-wide mega-repository. SceneWorks,
ChatWorks, and SoundWorks have distinct licensing, access, and release concerns, so product
consolidation remains a separate checkpoint. Inference is unified because it is already one change
and release boundary; products remain separate because they are not yet proven to be one.

## Pre-cutover release set

The complete machine-readable Phase 0 checkpoint remains
[`baseline/release-set.toml`](baseline/release-set.toml), with product graph details in
[`baseline/dependency-identities.md`](baseline/dependency-identities.md). The exact inference
identities replaced by this cutover were:

| Product graph | Logical component | Previous source identity |
|---|---|---|
| SceneWorks | `core-llm` | `SceneWorks/core-llm`, `branch=main#54cbac806304e823470ce3ded08f78589acdbb62` |
| SceneWorks | `sceneworks-gen-core` + MLX media | `michaeltrefry/mlx-gen`, `rev=b8c415261a9fc6a2409a8ffc989881f0e6a3c99b` |
| SceneWorks | `mlx-llm` | `SceneWorks/mlx-llm`, `rev=7041411f395e43c542770d1cfb3c3945c8c9a875` |
| SceneWorks | Candle media | `michaeltrefry/candle-gen`, `rev=0bb56647c60f192d2b59a12e0ffc2acdfbfa0f3b` |
| SceneWorks | Candle LLM | `SceneWorks/candle-llm`, `rev=3d9fdf04047bf3b1fbf323ab56c919f3a03f0794` and `rev=d0ba3e66b4d53420bb0b0745a185b975822089be` |
| ChatWorks | `core-llm` | `SceneWorks/core-llm`, `branch=main#54cbac806304e823470ce3ded08f78589acdbb62` |
| ChatWorks | `mlx-llm` | `SceneWorks/mlx-llm`, `branch=main#4b1f090e6524bbf743d780afc73679fff83ed28e` |
| ChatWorks | `candle-llm` | `SceneWorks/candle-llm`, `branch=main#8673651a3b78684a6c5cb59971f9797d5b756721` |

## New release set

Every migrated dependency resolves from:

```text
git = https://github.com/SceneWorks/inference
tag = runtime-2026.07.0
commit = 48cc2d87e14de0189ac4f7763fddc0a8581c2e68
```

The inference release contains one workspace lockfile, 74 path-owned packages, explicit media/LLM
catalogs, deterministic source/SBOM metadata, and no `inventory` dependency. The external products
retain their own lockfiles, which record the release tag and resolved commit for every selected
inference package.

## Private repository access

SceneWorks and ChatWorks remain public while the canonical inference repository remains private.
That accepted visibility boundary has two operational consequences:

- local product builds require a GitHub identity with read access to `SceneWorks/inference`; and
- product workflows that invoke Cargo require a least-privilege
  `SCENEWORKS_INFERENCE_READ_TOKEN` repository secret.

The workflows pass the credential to Git only through per-job environment configuration; it is not
committed to a manifest, lockfile, or Git credential store. Secrets are unavailable to untrusted
fork pull requests, so those pull requests cannot execute the Rust inference build unless a
maintainer runs the commit in a trusted branch. Publishing inference, distributing its packages, or
changing either repository's visibility remains a separate decision.

## Administrative release configuration

The private inference repository has access to the organization runner group `self-hosted-gpu`.
GitHub's assigned-job records identify both `nax-macos` and `cuda-windows` in that group, and the
release candidate executed successfully on both machines. The `nax`, `cuda`, and `real-weights`
labels are preserved. Six repository variables select revision-addressed persistent snapshot paths;
the real-weight workflow materializes a missing immutable snapshot on demand and reuses it on later
runs.

Public product workflows still require a least-privilege `SCENEWORKS_INFERENCE_READ_TOKEN` secret
before a hosted Cargo job can fetch the private release. Local validation uses the authenticated
system Git client and does not weaken that boundary.

Release evidence for exact commit `48cc2d87e14de0189ac4f7763fddc0a8581c2e68` is:

- manual CI run [`29284987010`](https://github.com/SceneWorks/inference/actions/runs/29284987010):
  hosted Linux/macOS, self-hosted NAX, and self-hosted CUDA suites passed;
- real-weight run [`29285222380`](https://github.com/SceneWorks/inference/actions/runs/29285222380):
  MLX and Candle LLM/media jobs passed against pinned SmolLM2, Qwen3, and Z-Image snapshots;
- tag CI run [`29293208430`](https://github.com/SceneWorks/inference/actions/runs/29293208430):
  the hosted matrix, tagged archive, checksums, SPDX SBOM, and external consumer passed; and
- GitHub Release
  [`runtime-2026.07.0`](https://github.com/SceneWorks/inference/releases/tag/runtime-2026.07.0):
  the four exact CI-produced release inputs are attached to the immutable tag.

## Compatibility evidence

- The 64-model product catalog and 45 worker engine mappings remain pinned by
  [`baseline/catalog-baseline.json`](baseline/catalog-baseline.json).
- Runtime bundle tests pin complete ordered MLX and Candle provider surfaces without loading model
  weights.
- SceneWorks passes scaffold/compose/NC-weight/skew guards, strict Clippy, the complete Rust suite,
  the Rust build, 1,658 web tests, 12 non-e2e Python contract tests, 4 CI-equivalent end-to-end tests
  with the documented FFmpeg case skipped, and 12 parity snapshots. The web lint and production
  build also pass; its existing warning-only lint baseline remains unchanged.
- ChatWorks passes strict Clippy, 79 Rust unit tests, the Rust build, the canonical-source pin gate,
  migration-touched Rust formatting, npm lint, and the production web build. Full-workspace
  `cargo fmt --check` still reports unrelated pre-existing formatting drift and was not mass-applied
  as part of this cutover.
- The final clean `runtime-2026.07.0` release build produced a 364,600,050-byte source archive, a
  455-package SPDX document, verified checksums, and a passing external consumer build against the
  extracted neutral contracts. Its manifest records 74 workspace packages, `dirty: false`, and the
  exact release commit above.
- CUDA, NAX, and all four real-weight executions passed as platform-owned release evidence; no
  queued or unexecuted platform gate is counted as a pass.

## Rollback

Rollback is a product change, not a mutation of the inference release:

1. Revert the SceneWorks and/or ChatWorks cutover commit.
2. Restore that product's exact pre-cutover dependency identities above and regenerate its lockfile.
3. Run the baseline catalog/skew checks before publishing the rollback build.
4. Leave `runtime-2026.07.0` and all migration checkpoint tags immutable for auditability.
5. Fix forward in `SceneWorks/inference` and issue a new runtime tag before attempting another
   product cutover.

The former repositories remain unchanged during this phase. Archival, workflow disabling,
visibility changes, and deletion require a separate explicit approval after the new release is
proven in products.
