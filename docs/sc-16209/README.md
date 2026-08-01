# sc-16209 — Provider-owned MLX activation anchors

This is the measurement and decision record for replacing the generic 14 GiB
1024² activation charge with provider-owned anchors. The raw canonical rows are
in [`resolution-sweep.json`](./resolution-sweep.json); the ignored real-weight
harness is `crates/sceneworks-worker/src/resolution_sweep.rs`.

## Outcome

The estimator now has three deliberately separate quantities:

```text
load-time gate:     component weights + the existing family load allowance
request fixed term: 2 GiB remaining OS/app reserve
request area term:  provider activation anchor × max(width×height / 1024², 1)
```

The load-time gate is unchanged. A provider declares only the bare activation
anchor for an exact generator route. If that route has no measurement, the
worker retains the conservative 14 GiB area term. This removes the earlier
unit mismatch where generic `HEADROOM_GB` was a transient plus OS reserve while
`LENS_DENSE_HEADROOM_GB` was a measured transient alone.

The motivating Krea 2 Turbo bf16 request at 1152×2048 now models:

```text
33.22 GiB assets + 2 GiB fixed + 7.67 GiB × 2.25 = 52.4775 GiB
```

That admits under the 62 GiB request budget on a 64 GiB Mac. The old generic
anchor modeled 66.72 GiB and rejected it. The sc-16195 real peak for the same
cell was 49.27 GiB, so the corrected estimate retains about 3.21 GiB of margin.

## Method

Measurements ran on an Apple M5 Max with 128 GB unified memory, macOS 26.5.2
(25F84), through the production MLX/Metal generator seam. Each canonical row is
a completed warm 1024×1024 render with real weights:

```text
clear_cache(); reset_peak_memory();
generate(1024, 1024);
peak = get_peak_memory();
clear_cache(); resident = get_active_memory();
transient = peak - resident;
```

The allocator cache drained to zero after every recorded request. Every render
was nonblank and none tripped the harness's low-contrast flag. One denoise step
is sufficient to reach the step/decoder high-water. SenseNova and Mage were
also run at two steps as a control; SenseNova was byte-identical and Mage moved
by only 2,097,152 bytes, so the larger Mage row is the published ceiling.

Anchors are never rounded down: each published value is the maximum observed
warm transient rounded upward to the next 0.01 GiB. These are local Metal
measurements only; they make no CUDA or Windows runtime claim.

## New real-weight rows

| provider route | tier | warm transient | published anchor | status |
|---|---:|---:|---:|---|
| `sana_sprint_1600m` | q8 | 13.0317 GiB | 13.04 GiB | registered |
| `anima_base` | q8 | 7.6633 GiB | 7.67 GiB | registered |
| `sensenova_u1_8b` | q8 | 1.3325 GiB | 1.34 GiB | registered |
| `flux2_klein_9b` | bf16 | 14.0644 GiB | 14.07 GiB | registered |
| `flux1_dev` | bf16 | 14.0520 GiB | 14.06 GiB | registered |
| `mage_flow` | bf16 | 1.5674 GiB | 1.57 GiB | registered |
| `bernini` | bf16 | 29.6911 GiB full staged peak | — | excluded; generic fallback |

Bernini stages and releases components between phases, so its post-request
resident count is zero. Subtracting that zero from the peak leaves a full staged
phase peak that still includes weights, not an activation-only transient.
Registering 29.70 GiB would therefore make SceneWorks add those weights twice.
The run is retained as an excluded observation, and `bernini` keeps the generic
fallback until the harness can isolate a phase's resident weights and activation
high-water under the same staged model.

Five expanded harness turnkeys could not produce a credible local row:
`ideogram_4`, `kolors`, `chroma1_hd`, `boogu_image_turbo`, and
`sd3_5_large_turbo`. They remain on the generic 14 GiB fallback. Missing local
weights and incomplete turnkeys are evidence gaps, not zero-memory results.

## Published route table

The registry combines six new activation-only rows with sc-16195 anchors:

| exact provider route | anchor | evidence |
|---|---:|---|
| `krea_2_turbo` | 7.67 GiB | sc-16195 q8 + bf16 |
| `qwen_image` | 7.67 GiB | sc-16195 q8 |
| `sdxl` | 14.05 GiB | sc-16195 dense + packed SDXL rows |
| `z_image_turbo` | 14.05 GiB | sc-16195 q4 |
| the six registered routes in the table above | per row | this campaign |

An anchor is route-wide across weight tiers only when that provider's
measurements establish tier-independence. Lens is the known counterexample:
its packed source uses the 14 GiB fallback while dense/MXFP4 uses the existing
format-aware 29.88 GiB worker allowance, so neither Lens route publishes the
new route-only carrier. An anchor is also not provider-package wide. Distinct
graphs remain unmeasured, including Krea Raw, Qwen Edit/Control,
Sana Base, Anima Turbo/Aesthetic, SenseNova Fast, FLUX control/edit/KV/Schnell,
the other Mage variants, Bernini, and Bernini Renderer. Registry tests pin both the
positive table and these important negative boundaries.

## Conservativeness and regression proof

Every published anchor is above its activation-only row. The smallest new-row margin is
about 5.6 MiB (FLUX.2 Klein); the largest rounding margin is under 10 MiB. At
larger resolutions the existing sc-16195 area curve remains unchanged:
`anchor × megapixels`, with the 2 GiB reserve fixed. Routes without evidence do
not inherit a neighbor's smaller anchor.

Focused tests exercise the actual production lookup and formula. The Krea
regression fails if its plan is mutated back to the generic `2 + 14×MP` model,
and a separate Krea Raw regression proves the generic fallback remains intact.
The expanded harness exposes 19 provider-family tests so future installed
turnkeys can add measurements without another harness rewrite.
