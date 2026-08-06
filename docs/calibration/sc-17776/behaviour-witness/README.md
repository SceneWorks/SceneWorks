# Behaviour witness (sc-17776 candidate 5)

What `flux2_dev`'s memory ladder **declares**, rendered and compared across revisions — as opposed to
what the compiler emitted for it, which is what every other unit in this study hashes.

`sc17776-witness.rs` is applied to a scratch inference worktree as
`crates/media/candle-gen/candle-gen-flux2/examples/sc17776-witness.rs` by `git apply` and is **never
committed to inference**, the same status as the mutation patches in `../mutations.md`. It needs no
weights, no GPU work and no device allocation.

```bash
cp sc17776-witness.rs <worktree>/crates/media/candle-gen/candle-gen-flux2/examples/
cargo run -q --profile bench --locked -p candle-gen-flux2 --features cuda --example sc17776-witness
```

## Why the rendering is comparable across the window

`MemoryProviderContract` derives `Debug`, and `crates/contracts/gen-core/src/memory_strategy.rs` is
**byte-identical** between `5ffd7612` and `fbb00d6b` — so the type being rendered cannot have changed
shape between the two runs. Verified with `git diff --quiet 5ffd7612 fbb00d6b -- <that file>`.

`rendering-fbb00d6b.txt` is the reference output (18,764 bytes, six spec rows).

## Results

All runs on the RTX PRO 6000 box, 2026-08-05. Mutations are applied on top of `fbb00d6b`.

| case | sha256 of the rendering | verdict |
| --- | --- | --- |
| `fbb00d6b` (baseline) | `5b3dd71f…979129` | — |
| `5ffd7612` — the epic's window | `5b3dd71f…979129` | **identical** |
| `M-klein` | `5b3dd71f…979129` | identical |
| `M-rms` | `5b3dd71f…979129` | identical — **false green** |
| `M-safety` | `5b3dd71f…979129` | identical — **false green in this probe**, see below |
| `M-devfp` | `3aa8c73e…79438b` | DIFFER |

The `M-devfp` delta is exactly the rendered
`MemoryCalibrationIdentity { fingerprint: "…blocks-v2" → "…blocks-v3" }`, in all six spec rows.

## What this probe does not cover

It renders `provider_contract` and `resolved_numeric_tier` and stops there, so it is blind to the
**admission** path — `M-safety` widens `validate_context`'s batch guard and the rendering does not
move. That is a gap in the probe rather than in the idea, and it is the design requirement for any
production version: also drive the admission probes that
`crates/contracts/gen-core-testkit/src/memory_strategy.rs::check_memory_strategy_registry` already
performs, weights-free, for every registered memory strategy in a catalog.

`M-rms` is the boundary that no witness of this shape can cross: the contract table is the *declared*
ladder, the calibration measures *realised* VRAM, and a numerics change moves the second without
moving the first. A behaviour witness composes with a code digest; it does not replace one.
