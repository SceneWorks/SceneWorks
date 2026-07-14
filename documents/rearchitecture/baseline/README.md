# Phase 0 Migration Baseline

This directory is the reproducible pre-migration checkpoint for MRP-001 and
MRP-002. It describes the checked-out source trees and the product dependency
graphs; it is not a claim that every checkout was the latest remote commit.

## Artifacts

- [`release-set.toml`](release-set.toml) — repository HEADs, tree IDs, tracking
  state, toolchains, versions, and authoritative file hashes.
- [`dependency-identities.md`](dependency-identities.md) — product Cargo source
  identities, skew-gate results, and UI release/source discrepancy.
- [`catalog-baseline.json`](catalog-baseline.json) — stable model signatures and
  SceneWorks-to-engine mappings.
- [`../REPOSITORY_DECISIONS.md`](../REPOSITORY_DECISIONS.md) — proposed canonical
  names, ownership, and the `candle-gen` behind-tracking handling decision.
- [`../BINARY_FIXTURE_INVENTORY.md`](../BINARY_FIXTURE_INVENTORY.md) — binary
  footprint, duplicates, and storage recommendation.

Regenerate or verify the catalog snapshot from the SceneWorks root:

```sh
node scripts/rearchitecture/capture-catalog-baseline.mjs
node scripts/rearchitecture/capture-catalog-baseline.mjs \
  --check documents/rearchitecture/baseline/catalog-baseline.json
```

## Imported inference scale

The first inference import is expected to contain **69 Rust packages** before any
new bundle/catalog packages are added:

| Source | Packages |
|---|---:|
| `core-llm` | 2 |
| `mlx-llm` | 2 |
| `candle-llm` | 1 |
| `mlx-gen` | 33 |
| `candle-gen` | 31 |

The source Git packs total roughly 205 MiB, dominated by `mlx-gen` fixtures.

## Current CI inventory

| Repository | Current workflow coverage |
|---|---|
| SceneWorks | Linux web/parity, Windows desktop, macOS MLX, Windows CUDA, manual Linux Candle server, desktop release |
| `mlx-gen` | Linux contract and macOS MLX |
| `candle-gen` | CPU matrix and self-hosted Windows CUDA |
| `mlx-llm` | macOS MLX |
| `mlx-rs` | macOS validation and docs |
| ChatWorks | Git dependency supply-chain pin gate only |
| `core-llm` | No workflow |
| `candle-llm` | No workflow |
| SoundWorks | No workflow |
| `ui` | No workflow |

Self-hosted/special runner labels currently referenced:

- `[self-hosted, macOS, ARM64, nax]`
- `[self-hosted, Windows, X64, cuda]`
- `[self-hosted, windows, cuda]`
- `blaze/macos-14`
- `blaze/macos-15`
- `blaze/<matrix runner>`

Hosted runners currently referenced include `ubuntu-latest`, `ubuntu-22.04`,
`windows-2022`, and `macos-15`.

## Baseline caveats

- SceneWorks was clean at its recorded application/runtime HEAD; the README and
  rearchitecture plan were uncommitted planning changes when Phase 0 began.
- `candle-gen` was four commits behind its known `origin/main` tracking ref.
- Cargo needed to fetch the exact locked git revisions before `cargo tree` could
  reproduce the product graph locally; those revisions are now present in the local
  Cargo cache.
- The SceneWorks Candle-enabled graph contains two `candle-llm` revisions. This is
  recorded rather than corrected in the baseline.
- The UI source repo reports 0.1.0 while ChatWorks and SoundWorks resolve npm 0.2.0.

