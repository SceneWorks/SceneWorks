# SCAIL-2 coverage across the H3 duration pin

The application moves from `8754126942f51043f5d5b803cf69e8e32bff857f` to
`154983e5bc0b6eff2cb82f3fbc3f6bba4eb2f217`, delivered by inference PR #1512.
The packaged physical captures keep their original revisions, receipts, bounds and hardware axes.
The two exact-pair entries in `config/scail2-memory-pin-coverage.json` preserve their use for
this target on MLX and CUDA. They do not qualify any later pin.

The complete source-tree delta has SHA-256
`d6903458a0f89bd35f9d28afdec52d051a6fa508aaad4a916f765fb3b9b691ee` (the raw Git tree diff format used by
`scripts/scail2-fit-pin-coverage.mjs`). It contains only these files:

- `crates/contracts/gen-core-h3/src/h3_memory.rs`
- `crates/contracts/gen-core-h3/src/veda/workspace.rs`
- `crates/contracts/gen-core/src/h3_memory.rs`
- `crates/media/candle-gen/candle-gen-minimax-h3/src/denoise/geometry.rs`
- `crates/media/candle-gen/candle-gen-minimax-h3/src/pipeline.rs`
- `crates/media/mlx-gen/mlx-gen-minimax-h3/src/denoise/geometry.rs`
- `crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy/routes.rs`
- `crates/media/mlx-gen/mlx-gen-minimax-h3/src/pipeline.rs`
- `crates/media/mlx-gen/mlx-gen-minimax-h3/tests/duration_sweep_real.rs`
- `scripts/check-workspace.py`
- `scripts/tests/test_check_workspace.py`

Review of the entire delta confirms:

- Executable changes are confined to H3 temporal geometry, duration validation and H3 memory
  admission. `gen-core/src/h3_memory.rs` is an H3-only module; its exports and frame limits are
  not SCAIL-2 allocation inputs. The H3 Veda workspace change is test-only.
- Neither SCAIL-2 provider, its fit contract, allocation policy, kernel, weight loader, decoder,
  build input, toolchain nor dependency lock changes. No shared request structure changes.
- The remaining changes are H3 tests, comments, and repository geometry/tooling guards.

The broad crate-level provider and shared-contract hashes do change because they include H3
source files. This is a source-reviewed compatibility qualification, not a claim that those
broad hashes are equal. The generic certificate generator conservatively refuses such changes;
its comparison-only mode supplied both source and target hashes for this review.
The existing runtime exact-pair and approved-index checks remain unchanged.

Reproduce the comparison without running model inference:

```powershell
node scripts/scail2-fit-pin-coverage.mjs INFERENCE_REPO 8754126942f51043f5d5b803cf69e8e32bff857f 154983e5bc0b6eff2cb82f3fbc3f6bba4eb2f217 mlx
node scripts/scail2-fit-pin-coverage.mjs INFERENCE_REPO 8754126942f51043f5d5b803cf69e8e32bff857f 154983e5bc0b6eff2cb82f3fbc3f6bba4eb2f217 cuda
```

The certificates record the target hashes returned by those commands. No physical measurement
or historical calibration record was rewritten, and no new measurement is claimed.
