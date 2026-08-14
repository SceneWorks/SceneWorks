# SC-18251 resident-only MLX estimate-band audit

## Result

At the shipped manifest sizes and pinned inference revision, the 1.10x MLX estimate margin changes
admission at the requested 48/64/96/128 GiB host sizes only for these two source-derived
Resident-only configurations:

| Provider | Tier | Host | Shipped assets | Incremental budget after assets + 2 GiB legacy reserve | Raw incremental estimate | 1.10x estimate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sd3_5_large` | q8 | 48 GiB | 29.102 GiB | 16.898 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |
| `sd3_5_large_turbo` | q8 | 48 GiB | 29.094 GiB | 16.906 GiB | 16.000 GiB (fits) | 17.600 GiB (refuses) |

There are no flips at 64, 96, or 128 GiB. The earlier Chroma flips were impossible configurations:
Chroma has no strict-control lane, and its base providers cannot be combined with the FLUX.1 Dev
control checkpoint. FLUX.1 Schnell and every FLUX.2 Klein base/edit variant likewise have no strict
control checkpoint.

Strict FLUX control exists only for `flux_dev` and `flux2_dev`. Production routes them to the
dedicated `flux1_dev_control` and `flux2_dev_control` providers, respectively. Those real control
routes remain in the audit and do not flip at the tested host sizes.

`flux2_dev_edit` is also not part of this generic estimate-band result. Production sends that route
through the provider-owned calibrated multi-reference safety path in `image_jobs/flux2.rs`.

## Production accounting

The executable audit is
`mlx_fit_gate::tests::shipped_resident_only_mlx_estimate_band_audit_uses_the_production_budget_path`.
It uses the same accounting seams as the production selector:

1. `live_request_budget` records physical host capacity and the 2 GiB legacy foreign-resident
   reserve.
2. Loaded provider assets remain in `committed_bytes`.
3. The request gate credits those same loaded assets out of the modeled full peak.
4. `CandidateBasis::EstimateFloor` widens the remaining incremental estimate by
   `MLX_ESTIMATE_MARGIN` (0.10).

For the ordinary unanchored 1024x1024 routes, the incremental raw estimate is 16 GiB: the remaining
2 GiB fixed OS/app allowance plus the generic 14 GiB activation transient. A 48 GiB SD3 q8 load has
about 16.9 GiB left after assets and the legacy reserve. The former 16 GiB estimate fit; the current
17.6 GiB estimate does not.

The audit builds sparse artifacts at each exact shipped logical size, queries provider footprint and
1024x1024 activation facts from the pinned runtime registry, constructs the production
`MlxRequestPlan`, and calls the production request selector. Control bytes are discovered only after
the production router has resolved a dedicated strict-control provider; they are never injected into
a base-provider contract.

## Source-bound inventory

The eight-family story scope contains 65 source-selected route/tier cells. The audit exhaustively
classifies them as 62 generic-selector candidates plus the three q4/q8/bf16 PuLID-FLUX cells.
Provider/manifest contracts classify 35 of the 62 generic cells as Resident-only, and the budget
walk evaluates every one of those 35. Exact set equality binds both the generic inventory and the
explicit-exclusion inventory, so a route that produces no Resident-only result cannot silently
disappear:

- every shipped MLX image entry in the story-scoped Boogu, Chroma, FLUX.1, FLUX.2 Dev, FLUX.2
  Klein, Ideogram, Lens, and SD3 families, discovered from
  `config/manifests/builtin.models.jsonc` rather than repeated as a model-id list;
- every shipped macOS q4/q8/bf16 tier and its complete download-byte total, including multi-download
  component layouts;
- base providers from production `MODEL_TABLE` resolution;
- edit providers from the production FLUX.2 edit router;
- strict-control providers from the production FLUX control router used by both availability arms;
- provider existence, footprint, activation facts, and strategy contracts from the pinned runtime
  registry;
- manifest strategy implementations for tier bindings whose deliberately sparse fixture cannot
  reproduce a full turnkey component tree; and
- the production measured-load-shape seam, which keeps base Lens q4 on its exact measured contract
  instead of misclassifying it as Resident-only.

`pulid_flux_dev` is the one explicit exclusion inside the eight-family scope. It is a shipped
`family=flux`, `type=image`, MLX entry with three tiers, but deliberately has no `MODEL_TABLE` row:
production recognizes the identity-conditioned request directly and loads the inventory-registered
`pulid_flux` provider with `LoadSpec::identity`, adding the adapter, EVA face encoder, and face stack
outside the generic base-image plan. Treating its backbone as a plain FLUX route would undercount
that composition, so all three cells are source-accounted and explicitly excluded from the generic
fixture instead.

The classifier also pins Mage even though `family=mage-flow` is outside the story's eight-family
result scope. Its six production routes contribute 18 q4/q8/bf16 cells. Every Mage tier is a split
load whose model-specific DiT is combined with shared, tier-specific text-encoder and VAE
co-requisites and component precision floors. The generic single-root sparse fixture cannot model
that accounting truthfully, so those 18 cells are separately source-accounted as an explicit
split-component exclusion. This prevents a future family/scope edit from making Mage vanish by
accident while keeping it out of the 65-cell story result.

The resulting Resident-only subset is:

| Family / routes | Resident-only cells | Why they remain in the estimate audit | Band result |
| --- | ---: | --- | --- |
| SD3: `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium` base | 9 | the loaded cells expose no implemented optimized strategy at this pin | Large q8 and Large-Turbo q8 flip at 48 GiB only |
| Boogu: base, turbo, edit | 3 | registered providers use the compatibility Resident contract | no flips |
| Ideogram: base and turbo | 6 | registered providers use the compatibility Resident contract | no flips |
| FLUX.1: `flux_dev` strict control | 3 | only the dedicated control provider is Resident-only; Schnell has no control route | no flips |
| FLUX.2 Dev: base and strict control | 6 | base and the dedicated control provider are Resident-only; Dev edit uses provider-owned safety | no flips |
| FLUX.2 Klein: KV edit | 3 | this routed edit provider uses the compatibility Resident contract; no Klein route has a control checkpoint | no flips |
| Lens: base q8/bf16 and Turbo q4/q8/bf16 | 5 | these entry/tier cells lack an applicable implemented optimized contract; base Lens q4 is measured and excluded | no flips |

Chroma, clean FLUX.1 base, and the remaining FLUX.2 Klein base/edit cells resolve to implemented
optimized contracts and are filtered out before the Resident-only walk. The three Boogu downloads
have no `variant`; their manifest `mlx.quantize: 8` supplies the shipped q8 tier.

## Mutation coverage

`resident_only_audit_rejects_impossible_control_routes_and_keeps_real_ones` injects the prior bad
strict-control assumptions for all three Chroma entries, FLUX.1 Schnell, and the FLUX.2 Klein
base/KV/True-V2 entries. Every mutation must fail closed at the production control router. The same
test proves that:

- `flux_dev` resolves to `flux1_dev_control` with the exact FLUX.1 checkpoint size;
- `flux2_dev` resolves to `flux2_dev_control` with the exact FLUX.2 checkpoint size;
- no other provider receives nonzero control bytes;
- base Lens q4 remains excluded by its measured contract; and
- `flux2_dev_edit` remains excluded from the generic selector audit.

`resident_only_audit_inventory_rejects_duplicate_and_zero_cell_drop_mutations` restores the
independent completeness proof. Replacing `flux_schnell` with a second `chroma1_base` declaration
must fail before any set conversion can hide the duplicate. Dropping `flux_schnell` entirely is the
more subtle mutation: Schnell contributes zero Resident-only cells, so the old 35-cell subset and
two-flip summary remain unchanged, but exact equality against the 62-cell manifest/router source
inventory fails and names the missing route.

`resident_only_audit_exclusions_reject_pulid_and_mage_scope_mutations` pins the complementary
exclusion inventory. Removing PuLID, changing its disposition, or changing its `flux` family must
fail; removing the six-model Mage scope or changing a Mage route's `mage-flow` family must likewise
fail. The test also binds the expected three PuLID tiers and 18 Mage split-component tiers to current
production provider resolution and manifest byte accounting.

## Source values

- Host reserve: `fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB`.
- Generic activation/fixed decomposition: `mlx_fit_gate::{HEADROOM_GB, OS_APP_RESERVE_GB}`.
- Estimate widening: `ladder_margin_policy::MLX_ESTIMATE_MARGIN`.
- Shipped assets: `config/manifests/builtin.models.jsonc`.
- FLUX.1 control: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` at revision
  `5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5`, exact checkpoint size 4,281,779,224 bytes.
- FLUX.2 control: `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` at revision
  `b3dcd7836a0e926248dac3ccba8fc0853495764b`, exact checkpoint size 8,232,506,680 bytes.
- Provider facts and contracts: the pinned SceneWorks inference revision.
