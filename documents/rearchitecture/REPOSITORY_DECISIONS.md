# Repository Naming and Ownership Decisions

> **Status:** Proposed for the Phase 0 exit review. No repository has been created.

## Recommended canonical topology

| Purpose | GitHub repository | Local checkout | Visibility | Organizational owner |
|---|---|---|---|---|
| Inference source of truth | `SceneWorks/inference` | `inference/` | Public, matching the current Apache inference repositories | SceneWorks organization |
| Product source of truth, if approved at the license checkpoint | `SceneWorks/studio` | `studio/` | To be decided with mixed-license/access review | SceneWorks organization |
| MLX Rust fork | `SceneWorks/mlx-rs` | `mlx-rs/` | Retain current visibility | SceneWorks organization |

The repository display names may be `sceneworks-inference` and
`sceneworks-studio`; the recommended URLs stay concise. Package/crate names remain
independent of repository names.

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

## Phase 0 exit approval requested

- [ ] Approve `SceneWorks/inference` as the canonical inference repository URL.
- [ ] Approve `inference/` as the local checkout directory.
- [ ] Approve importing full history while baselining exact recorded SHAs.
- [ ] Approve keeping current package names through the first consumer cutover.
- [ ] Approve retaining `mlx-rs` as a separate repository.

