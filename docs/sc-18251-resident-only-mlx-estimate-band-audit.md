# SC-18251 resident-only MLX estimate-band audit

## Result

At the shipped manifest sizes, pinned inference revision, and exact pinned control-branch sizes, the
1.10x MLX estimate margin changes admission at the requested 48/64/96/128 GiB host sizes only for
these resident-only loaded-provider configurations:

| Provider | Tier | Host | Shipped assets | Incremental budget after assets + 2 GiB legacy reserve | Raw incremental estimate | 1.10x estimate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `chroma1_base` + FLUX.1 control branch | bf16 | 48 GiB | 29.593 GiB | 16.407 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |
| `chroma1_flash` + FLUX.1 control branch | bf16 | 48 GiB | 29.593 GiB | 16.407 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |
| `chroma1_hd` + FLUX.1 control branch | bf16 | 48 GiB | 29.593 GiB | 16.407 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |
| `sd3_5_large` (when its exact calibrated artifact/optimized legs are unavailable) | q8 | 48 GiB | 29.102 GiB | 16.898 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |
| `sd3_5_large_turbo` (resident-only at this pin) | q8 | 48 GiB | 29.094 GiB | 16.906 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |

There are no flips at 64, 96, or 128 GiB among the audited configurations. In particular,
`flux2_dev_edit` q4 on a 48 GiB host is **not** a flip: the earlier audit compared its full estimate
to all 48 GiB and omitted the legacy reserve and post-load accounting. It is also served by the
provider-owned safety path in `image_jobs/flux2.rs`, not by the generic MLX request selector audited
here.

## Production accounting

The executable audit is
`mlx_fit_gate::tests::shipped_resident_only_mlx_estimate_band_audit_uses_the_production_budget_path`.
It uses the production seams rather than treating physical host capacity as the selector budget:

1. `live_request_budget` records the physical host total and the 2 GiB legacy foreign-resident
   reserve.
2. After load, provider assets remain in `committed_bytes`.
3. The request gate credits the same loaded provider assets out of the modeled full peak.
4. `CandidateBasis::EstimateFloor` widens the remaining incremental estimate by
   `MLX_ESTIMATE_MARGIN` (0.10).

For the ordinary unanchored 1024x1024 routes, the incremental raw estimate is 16 GiB: the remaining
2 GiB fixed OS/app allowance plus the generic 14 GiB activation transient. Thus a 48 GiB SD3 q8 load
has roughly 16.9 GiB left after its assets and the legacy reserve: 16 GiB fit before the epic, while
17.6 GiB does not. The Chroma bf16 rows have roughly 16.4 GiB left after adding the exact 3.988 GiB
FLUX.1 control branch to the shipped base assets, so they cross the same band. The audit creates a
sparse file at the exact checkpoint length and lets `MlxRequestPlan::for_spec_and_manifest`'s shared
production core discover it; the control is not a zero-byte marker or hand-added arithmetic.

Lens uses a different production rule. The clean unmeasured Base q8/bf16 and Turbo q4/q8/bf16 cells
are included. Packed q4/q8 remain disk-derived with the generic activation fallback. Bf16 retains
the provider's 30.07 GiB materialized text-encoder footprint (17.241 GiB above its three MXFP4 source
shards) and the architecture-specific 29.88 GiB activation transient. None of those five cells lands
inside the 10% band at the audited host sizes.

## Completeness

The audit reads the shipped `estimatedSizeBytes`/`diskSizeBytes` values directly from
`config/manifests/builtin.models.jsonc`, injects the exact provider-owned activation and Lens
footprint facts exported by the pinned inference sources, and evaluates all 39 resident-only tier
routes at 48/64/96/128 GiB. The injected seam is the platform-neutral core of the production request
planner: macOS obtains the same facts from its registry, while the default Linux workspace (which
intentionally has an empty media registry) runs the complete audit deterministically.

| Family / routes | Shipped asset range | Resident-only condition | Band result |
| --- | ---: | --- | --- |
| Chroma: `chroma1_base`, `chroma1_flash`, `chroma1_hd` | 18.07-29.59 GiB including the 3.988 GiB control branch | a control overlay disables conditional optimized legs | all three bf16 variants flip at 48 GiB |
| SD3: `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium` | 22.55-36.13 GiB | Turbo/Medium lack exact calibration; Large also fails closed when its exact artifact or clean/streamable conditions are absent | q8 Large and Large-Turbo flip at 48 GiB only |
| FLUX.1: `flux1_schnell`, `flux1_dev` | 12.94-35.42 GiB including the 3.988 GiB control branch | a control overlay disables conditional optimized legs | no flips |
| FLUX.2 Klein: base/edit/KV provider routes | 24.67-39.89 GiB including the 7.667 GiB control branch | a control overlay disables conditional optimized legs; a non-adopting registered route uses the worker's Resident-only compatibility contract | no flips |
| Lens: clean Base q8/bf16 and Turbo q4/q8/bf16 | 20.33-45.79 GiB after bf16 materialization | entry/tier cross-products are unmeasured; Turbo bf16's optimized contract additionally requires Sequential | no flips |

`shipped_resident_only_audit_inventory_matches_the_manifest` pins the route/tier inventory to the
shipped manifest without consulting a platform-only registry. Video and utility providers without
an image `MemoryProviderContract`, and the custom FLUX.2-dev edit safety path, are outside the generic
selector seam rather than silently counted as passing rows.

## Source values

- Host reserve: `fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB`.
- Generic activation/fixed decomposition: `mlx_fit_gate::{HEADROOM_GB, OS_APP_RESERVE_GB}`.
- Estimate widening: `ladder_margin_policy::MLX_ESTIMATE_MARGIN`.
- Shipped assets: `config/manifests/builtin.models.jsonc`.
- FLUX.1 control: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` at the pinned
  `5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5` revision, exact checkpoint size 4,281,779,224 bytes.
- FLUX.2 control: `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` at the pinned
  `b3dcd7836a0e926248dac3ccba8fc0853495764b` revision, exact checkpoint size 8,232,506,680 bytes.
- Provider activation anchors and Lens load-exact footprint: pinned `mlx-gen-catalog` and
  `mlx-gen-lens` sources at inference revision `40fa7583a01974617e2a7275052d6d446688c956`.
