# SCAIL-2 coverage across the combined inference pin

The application pins `6f9c24047a0ebbb473dd83a4aa6247d7fe5e36a8`, the inference merge
containing both the shared runtime-cache fix and H3 duration support from inference PR #1512.
The packaged physical captures retain their original
`8754126942f51043f5d5b803cf69e8e32bff857f` revision, receipts, bounds and hardware axes.

The canonical records in `config/scail2-memory-source-reviews.json` review the complete
25-file capture-to-target delta for MLX and CUDA, including both old and new provider and
shared-contract hashes. The runtime certificates in `config/scail2-memory-pin-coverage.json`
cover only that immutable pair and backend; later pins are not implicitly qualified.

The cache delta to `2ea42202b2658476441e0da7447d16ada69d43f0` was already reviewed on main
in PR #3493. It preserves native formats, shapes, scheduling, residency and allocation policy.
The additional delta from that revision to the combined target contains only:

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

Executable changes in this additional delta are confined to H3 temporal geometry, duration
validation and H3 memory admission. The H3 Veda workspace change is test-only. No SCAIL-2
provider, fit contract, kernel, decoder, allocation strategy or execution policy changes.

Broad crate-level hashes include H3 source files and therefore change. The source reviews
record the actual target hashes and full Git tree-diff digest; they do not assert equal hashes
or new physical measurements. The existing exact-pair and approved-index checks remain active.

Reproduce and authenticate each canonical source review using the existing tool:

```powershell
node scripts/scail2-fit-pin-coverage.mjs INFERENCE_REPO 8754126942f51043f5d5b803cf69e8e32bff857f 6f9c24047a0ebbb473dd83a4aa6247d7fe5e36a8 mlx b1240ff79fb6e856eceed280d16ec9f23ff657fda1a7f2a2a4d352c3329bdcc5
node scripts/scail2-fit-pin-coverage.mjs INFERENCE_REPO 8754126942f51043f5d5b803cf69e8e32bff857f 6f9c24047a0ebbb473dd83a4aa6247d7fe5e36a8 cuda b1240ff79fb6e856eceed280d16ec9f23ff657fda1a7f2a2a4d352c3329bdcc5
```

No physical measurement or historical calibration record was rewritten.
