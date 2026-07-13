# Pre-Migration Dependency Identities

Captured for MRP-002 on 2026-07-12 from the release set in
[`release-set.toml`](release-set.toml). These are compatibility facts, not the
target dependency design.

## Summary

- Internal SceneWorks/michaeltrefry Git dependency declarations: **77**.
- Manifests containing those declarations: **7**.
- SceneWorks direct MLX-generation declarations: **31**.
- SceneWorks direct Candle-generation declarations: **28**.
- SceneWorks worker manifest size: **1,922 lines**.
- Current default worker graph resolves exactly one `sceneworks-gen-core`.
- Current `backend-candle` worker graph also resolves exactly one
  `sceneworks-gen-core`.
- The Candle-enabled graph resolves **two different `candle-llm` revisions**:
  `3d9fdf0…` through `candle-gen-joycaption` and `d0ba3e6…` through the worker's
  direct dependency. Both resolve the same `core-llm` source in the SceneWorks
  lockfile, so this is not a current gen-core skew failure, but the duplicate engine
  package is part of the baseline the consolidated graph must intentionally remove
  or preserve during comparison.

## SceneWorks default worker graph

Generated with:

```sh
cargo tree -p sceneworks-worker --target all --color never --prefix none
```

Relevant unique package identities:

```text
core-llm v0.0.0 (https://github.com/SceneWorks/core-llm?branch=main#54cbac80)
mlx-gen v0.0.0 (https://github.com/michaeltrefry/mlx-gen?rev=b8c415261a9fc6a2409a8ffc989881f0e6a3c99b#b8c41526)
mlx-llm v0.0.0 (https://github.com/SceneWorks/mlx-llm?rev=7041411f395e43c542770d1cfb3c3945c8c9a875#7041411f)
pmetal-mlx-rs v0.25.8 (https://github.com/michaeltrefry/mlx-rs?rev=38e1cc1730a11b1e40c2c8ecda01606763a12d51#38e1cc17)
pmetal-mlx-sys v0.2.4 (https://github.com/michaeltrefry/mlx-rs?rev=38e1cc1730a11b1e40c2c8ecda01606763a12d51#38e1cc17)
sceneworks-gen-core v0.1.0 (https://github.com/michaeltrefry/mlx-gen?rev=b8c415261a9fc6a2409a8ffc989881f0e6a3c99b#b8c41526)
```

## SceneWorks Candle-enabled worker graph

Generated with:

```sh
cargo tree -p sceneworks-worker --features backend-candle --target all --color never --prefix none
```

Relevant unique package identities:

```text
candle-gen v0.0.0 (https://github.com/michaeltrefry/candle-gen?rev=0bb56647c60f192d2b59a12e0ffc2acdfbfa0f3b#0bb56647)
candle-llm v0.0.0 (https://github.com/SceneWorks/candle-llm?rev=3d9fdf04047bf3b1fbf323ab56c919f3a03f0794#3d9fdf04)
candle-llm v0.0.0 (https://github.com/SceneWorks/candle-llm?rev=d0ba3e66b4d53420bb0b0745a185b975822089be#d0ba3e66)
core-llm v0.0.0 (https://github.com/SceneWorks/core-llm?branch=main#54cbac80)
mlx-gen v0.0.0 (https://github.com/michaeltrefry/mlx-gen?rev=b8c415261a9fc6a2409a8ffc989881f0e6a3c99b#b8c41526)
mlx-llm v0.0.0 (https://github.com/SceneWorks/mlx-llm?rev=7041411f395e43c542770d1cfb3c3945c8c9a875#7041411f)
pmetal-mlx-rs v0.25.8 (https://github.com/michaeltrefry/mlx-rs?rev=38e1cc1730a11b1e40c2c8ecda01606763a12d51#38e1cc17)
pmetal-mlx-sys v0.2.4 (https://github.com/michaeltrefry/mlx-rs?rev=38e1cc1730a11b1e40c2c8ecda01606763a12d51#38e1cc17)
sceneworks-gen-core v0.1.0 (https://github.com/michaeltrefry/mlx-gen?rev=b8c415261a9fc6a2409a8ffc989881f0e6a3c99b#b8c41526)
```

## Skew-gate result

The following checks passed at capture time:

```text
scripts/check-gen-core-skew.sh
scripts/check-gen-core-skew.sh --features backend-candle
scripts/check-gen-core-skew.sh --self-test
```

Both real graphs contained the one expected gen-core source:

```text
sceneworks-gen-core v0.1.0
git+https://github.com/michaeltrefry/mlx-gen
rev b8c415261a9fc6a2409a8ffc989881f0e6a3c99b
```

## Lockfile source identities

The active lockfiles contain several source forms and resolved revisions for the
same logical components:

| Lockfile | Logical dependency | Resolved source |
|---|---|---|
| `mlx-llm/Cargo.lock` | `core-llm` | `branch=main#15d9ff9…` |
| `candle-llm/Cargo.lock` | `core-llm` | `branch=main#15d9ff9…` |
| `mlx-gen/Cargo.lock` | `core-llm` | `branch=main#3870ed1…` |
| `candle-gen/Cargo.lock` | `core-llm` | `branch=main#54cbac8…` |
| `SceneWorks/Cargo.lock` | `core-llm` | `branch=main#54cbac8…` |
| `ChatWorks/Cargo.lock` | `core-llm` | `branch=main#54cbac8…` |
| `mlx-gen/Cargo.lock` | `mlx-llm` | `rev=7041411…` |
| `SceneWorks/Cargo.lock` | `mlx-llm` | `rev=7041411…` |
| `ChatWorks/Cargo.lock` | `mlx-llm` | `branch=main#4b1f090…` |
| `candle-gen/Cargo.lock` | `candle-llm` | `rev=3d9fdf0…` |
| `SceneWorks/Cargo.lock` | `candle-llm` | `rev=3d9fdf0…` and `rev=d0ba3e6…` |
| `ChatWorks/Cargo.lock` | `candle-llm` | `branch=main#8673651…` |

The inference monorepo must replace these internal Git source identities with path
dependencies before any registry refactor. A single new lockfile is expected to
change package IDs; compatibility is judged using contract/catalog snapshots and
tests, not by preserving the old Git-form package IDs.

## UI identity discrepancy

At capture time:

- `ui/package.json`: `0.1.0`.
- `ui/src/index.js`: `0.1.0`.
- UI Git tag/describe: `v0.1.0`.
- ChatWorks declares and resolves `@sceneworks/ui` `0.2.0`.
- SoundWorks declares and resolves `@sceneworks/ui` `0.2.0`.
- Both consumer lockfiles resolve the public npm tarball
  `@sceneworks/ui/-/ui-0.2.0.tgz`.

This is recorded as a pre-existing release-source discrepancy. It is not corrected
in Phase 0.

