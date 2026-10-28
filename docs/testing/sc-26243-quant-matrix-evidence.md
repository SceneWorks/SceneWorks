# sc-26243 FLUX 3 quant/quality/memory evidence

Generated 2026-10-27. Source of truth: `docs/testing/sc-26243-quant-matrix-evidence.json`.

- Feature integration head: `0058de93cdbb3e7a1fafd26b1d4c284bd5ac8de9`
- Inference pin: `6f9bc0f23fe74a223e843a5c76208e8541c2a2ff` (on inference `main`), moved from tested pin `da7534d5ea663ee42234422cf717bf5f62f922c1` during the final main integration. The FLUX3 runtime closure is unchanged between them (only `UPSTREAM.md` and `pup_parity.rs` in the FLUX3 crate), so the measurements remain valid at the final pin.
- Harness source: MLX `819a80a1f0922956ec9188eafbc4025389dd9431`; Candle `0e5f40732490f3eae9b301b47dcf92608f3fa16f`
- Request: 1024x1024, count 1, seed 26243, fixed prompt; distilled 16 steps, base 50 steps / CFG 3.5
- Candle campaign: workflow run [40553173175](https://github.com/SceneWorks/SceneWorks/actions/runs/40553173175) on `cuda-windows-2` (physical GPU1)
- MLX campaign: `nax-macos` (Apple M5 Max, 128 GiB unified), one ignored footprint cell per process

| model | tier | backend | resident GiB | peak GiB | lowest fit GiB | PNG (sha256 prefix) | visual |
|---|---|---|---|---|---|---|---|
| flux3_klein | bf16 | candle | 24.73 | 26.77 | 32.00 | `dfc771438a7f` | PASS |
| flux3_klein | bf16 | mlx | 18.14 | 35.48 | 35.48 | `61d3fabdb56c` | PASS |
| flux3_klein | q4 | candle | 18.63 | 20.68 | 27.00 | `661880745ab6` | PASS |
| flux3_klein | q4 | mlx | 11.67 | 29.01 | 29.01 | `707116989cec` | PASS |
| flux3_klein | q8 | candle | 20.61 | 22.65 | 29.00 | `854348380ec7` | PASS |
| flux3_klein | q8 | mlx | 13.92 | 31.26 | 31.26 | `ff2cd4829555` | PASS |
| flux3_klein | fp8r | candle | 20.19 | 22.30 | 28.00 | `a07364ca9183` | PASS |
| flux3_klein | nvfp4 | candle | 19.54 | 21.55 | 27.00 | `7de3ecba2534` | PASS |
| flux3_klein_base | bf16 | candle | 25.04 | 27.06 | 32.00 | `650656e9f013` | PASS |
| flux3_klein_base | bf16 | mlx | 18.12 | 35.46 | 35.46 | `4f64c17f0164` | PASS |
| flux3_klein_base | q4 | candle | 18.96 | 20.95 | 27.00 | `5ca293b682dd` | PASS |
| flux3_klein_base | q4 | mlx | 11.65 | 28.99 | 28.99 | `147d5c064ea2` | PASS |
| flux3_klein_base | q8 | candle | 20.92 | 22.93 | 29.00 | `26952ab051eb` | PASS |
| flux3_klein_base | q8 | mlx | 13.90 | 31.24 | 31.24 | `07bab3c68f65` | PASS |
| flux3_klein_base | fp8r | candle | 19.86 | 21.87 | 28.00 | `75cf78707425` | PASS |
| flux3_klein_base | nvfp4 | candle | 19.16 | 21.62 | 27.00 | `b392c54ac4eb` | PASS |

Candle resident/peak are whole-device deltas above a quiet desktop baseline (2.0-2.2 GiB). MLX
resident/peak are allocator process deltas. `lowestFitBytes = max(measured peak, declared
manifest floor)`; the declared floor is the product's supported admission minimum.

All 16 retained PNGs were visually inspected and accepted. Q4/Q8 packages were converted offline
on their own host (CPU only) and verified against the catalog SHA/size pins.
