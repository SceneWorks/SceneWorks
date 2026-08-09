# SC-18251 resident-only MLX estimate-band audit

## Result

At the shipped manifest sizes and the pinned inference revision, the 1.10x MLX estimate margin
changes admission at the requested 48/64/96/128 GiB host sizes only for these resident-only loaded
provider configurations:

| Provider | Tier | Host | Shipped assets | Incremental budget after assets + 2 GiB legacy reserve | Raw incremental estimate | 1.10x estimate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
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

For the unanchored 1024x1024 routes, the incremental raw estimate is 16 GiB: the remaining 2 GiB
fixed OS/app allowance plus the generic 14 GiB activation transient. Thus a 48 GiB SD3 q8 load has
roughly 16.9 GiB left after its assets and the legacy reserve: 16 GiB fit before the epic, while
17.6 GiB does not.

## Completeness

The audit reads the shipped `estimatedSizeBytes`/`diskSizeBytes` values directly from
`config/manifests/builtin.models.jsonc`, resolves provider-owned activation anchors through the
pinned runtime catalog, and evaluates every tier at 48/64/96/128 GiB for the shipped image families
whose loaded contracts can be resident-only:

| Family / routes | Shipped asset range | Resident-only condition | Band result |
| --- | ---: | --- | --- |
| Chroma: `chroma1_base`, `chroma1_flash`, `chroma1_hd` | 14.08-25.61 GiB | eager/overlaid/non-Sequential loads disable conditional optimized legs | no flips |
| SD3: `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium` | 22.55-36.13 GiB | Turbo/Medium lack exact calibration; Large also fails closed when its exact artifact or clean/streamable conditions are absent | q8 Large and Large-Turbo flip at 48 GiB only |
| FLUX.1: `flux1_schnell`, `flux1_dev` | 8.95-31.43 GiB | overlaid/non-Sequential loads disable conditional optimized legs | no flips |
| FLUX.2 Klein: base/edit/KV provider routes | 17.00-32.22 GiB | overlaid/non-Sequential loads disable conditional optimized legs; a non-adopting registered route uses the worker's Resident-only compatibility contract | no flips |

`shipped_conditional_mlx_contracts_really_have_a_resident_only_load` pins that source-side
classification against the actual provider registry at the pinned inference revision. Video and
utility providers without an image `MemoryProviderContract`, and the custom FLUX.2-dev edit safety
path, are outside the generic selector seam rather than silently counted as passing rows.

## Source values

- Host reserve: `fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB`.
- Generic activation/fixed decomposition: `mlx_fit_gate::{HEADROOM_GB, OS_APP_RESERVE_GB}`.
- Estimate widening: `ladder_margin_policy::MLX_ESTIMATE_MARGIN`.
- Shipped assets: `config/manifests/builtin.models.jsonc`.
- Provider anchors and loaded contracts: pinned `runtime-macos`/`mlx-gen` registry.

