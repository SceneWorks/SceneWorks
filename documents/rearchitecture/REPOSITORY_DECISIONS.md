# Repository Naming and Ownership Decisions

> **Status:** Accepted for the inference migration. `SceneWorks/inference` is the
> canonical inference source; product consolidation and old-repository archival
> remain separate, unapproved decisions.

## Accepted canonical topology

| Purpose | GitHub repository | Local checkout | Visibility | Organizational owner |
|---|---|---|---|---|
| Inference source of truth | `SceneWorks/inference` | `inference/` | Private; retain actual visibility unless separately approved | SceneWorks organization |
| Product sources of truth | Existing SceneWorks, ChatWorks, and SoundWorks repositories | Existing checkouts | Retain existing visibility pending mixed-license/access review | Existing repository owners |
| MLX Rust fork | `SceneWorks/mlx-rs` | `mlx-rs/` | Retain current visibility | SceneWorks organization |

The initial recommendation was public visibility because the imported inference
repositories were public. The repository was actually created private. The
migration does not broaden its own authority by changing that setting, and the
consumer cutover records the private cross-repository authentication consequence
explicitly. Package/crate names remain independent of repository names.

## Inference repository ownership

- Default branch: `main`.
- License: Apache-2.0 at the repository root, subject to per-import verification.
- Security reporting: inherit the SceneWorks organization policy and add a local
  `SECURITY.md` before consumer cutover.
- Code ownership categories:
  - Contracts and registry.
  - MLX backend/runtime.
  - Candle backend/runtime.
  - Provider families.
  - Platform CI/release.
- Merge policy: required contract/affected-backend checks; no universal
  `--all-features` requirement.
- Release authority: immutable `runtime-*` tags created only after bundle matrices
  pass.

## Import source decision

The import must use the SHAs recorded in
[`baseline/release-set.toml`](baseline/release-set.toml), not an implicit checkout
of the latest remote branch.

One exception requires an explicit decision before filtered-history import:
`candle-gen` was four commits behind its tracking branch at capture time. The
recommended import procedure is:

1. Import the complete reachable repository history, including current remote
   `main`, so no history is omitted.
2. Create the compatibility baseline branch/tag from the recorded checked-out SHA
   `102436012b24072c9cfc58b1b88e1c95655872e5`.
3. Treat later commits through tracking SHA
   `b0fc60a3c39fa7c15a563e26453a2bde123c3413` as post-baseline changes to replay or
   merge only after the byte-equivalence checkpoint.

This preserves all history without silently changing the release-set input.

## Placeholder disposition

- Importing `core-gen` and `llm-engine-core` source is not required for runtime
  functionality.
- Preserve their history/README context in migration documentation or lightweight
  archival import branches.
- The working `mlx-gen/gen-core` tree becomes the initial neutral generative
  contract source.
- Keep the package name `sceneworks-gen-core` through consumer cutover.
- Reconsider a package rename to `core-gen` only after Phase 5, with a compatibility
  shim if external consumers exist.
- Keep `core-llm` as the working LLM contract package name.

## Product repository decision

Creation of `SceneWorks/studio` remains conditional. Before it is authorized,
confirm:

- Mixed AGPL/PolyForm/Apache licensing is acceptable in one repository.
- All products have compatible contributor access and visibility.
- Independent release workflows remain legible.
- Shared UI and protocol benefits justify combined issue/PR scope.

Failure of this checkpoint leaves the three product repositories separate and does
not alter the inference migration.

## Phase 0 exit approval record

- [x] `SceneWorks/inference` is the canonical inference repository URL.
- [x] `inference/` is the local checkout directory.
- [x] Full history was imported while exact recorded SHAs remained compatibility baselines.
- [x] Current package names remain through the first consumer cutover.
- [x] `mlx-rs` remains a separate repository.

Execution was approved in stages on 2026-07-12 and 2026-07-13. Those approvals did
not authorize product monorepo creation, old-repository archival, or any
repository visibility change.
