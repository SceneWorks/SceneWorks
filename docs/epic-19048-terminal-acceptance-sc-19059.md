# Epic 19048 terminal acceptance record (SC-19059)

This is the source-level and decision-diff record for the integrated feature head. It is deliberately
not a claim that SC-19059 is complete. The CPU-only acceptance pass can close generator provenance,
exercise the promotion plumbing, and enumerate what the repository proves. Real Candle capture,
CUDA-linked capability resolution, privileged workflows, and the unresolved owner decisions below
remain terminal gates.

## Frozen integration point

- SceneWorks feature head: `5a6b46762a238df26b97e27027306f5249f33f4a`.
- Reviewed main-to-feature sync head: `ed60f55e0483bf44d7bd1b797bcd31cde4f08b07`.
- Synced SceneWorks main: `a74ba399923aeb52c5fff5ad5d1887ca9ea32813`.
- Inference pin: `4013049764172ee7dc707101c7da8c83c1483f2d`.
- SC-19049 Rust-produced decision baseline: `760e69b6b375cee417df7bfcc0f63f69b93e623a`.

The final acceptance must repeat this reconciliation at the eventual feature head. These identifiers
are the starting point for the terminal campaign, not permission to accept a stale head.

## Reopened SC-19055 compatibility migration

The non-owner-choice portion of SC-19055 was resumed from feature head
`a9d0be3080e5112654c18e0cb87ce245653375d8`. It reduces shared-selector-unreached from 38 routes to
15 without assigning policy to an evidence-free route:

- 14 image routes with an existing scalar admission ceiling now submit that already-normalized
  ceiling through resident-only `MemoryProviderContract::compatibility_default` as
  `LegacyScalar`: `boogu_image`, `boogu_image_turbo`, `ideogram_4`, `ideogram_4_turbo`, `kolors`,
  `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium`, and all six `sensenova_u1_8b*` routes.
- `instantid_realvisxl` and `bernini_image` submit their existing on-disk structural lower bounds
  through the same resident-only contract as `StructuralFloor`. Neither is labelled `Measured` or
  `EstimateFloor` and neither receives an invented estimate margin.
- The seven existing flat-video paths submit their already-normalized ceilings as typed Candle-lane
  `LegacyVideo` candidates: `ltx_2_3`, `mochi_1`, `scail2_14b`, `svd`, `wan_2_2`,
  `wan_2_2_i2v_14b`, and `wan_2_2_t2v_14b`. The selector adds neither another 4% grade nor another
  reserve.

The inventory parses these exact source-owned 14 + 2 + 7 sets and validates the production calls,
typed Candle lane, and `reserved_headroom_gb: 0.0` normalization. The regenerated inventory reports
20 named, 4 declaration-catch-all, 1 bespoke, 14 scalar-compatibility, 2 structural-floor, 7
legacy-video, and 15 unreached routes. The remaining set is exactly 14 evidence-free image routes
plus Bernini video. Their fallback-versus-disable policy remains an owner decision.

The non-circular Rust decision producer remains byte-identical at digest
`sha256:2aecbc27a99e467d78488df09441452434d72ddcf0cb432613fb611581125bc9`.
Comparing all 2,540 generated Candle coordinates across prediction, fit decision, and load-plan
fields reports **zero decision changes**. Inventory mechanism/provenance fields change as expected;
user-visible availability does not.

## Mechanical inventory and provenance

`scripts/generate-candle-admission-inventory.mjs` derives the route universe from production routing
and reads the decision surface written by
`crates/sceneworks-worker/src/candle_admission_decisions.rs`. The JavaScript generator does not write
the decisions file. A non-circular refresh therefore requires this producer first:

```sh
SCENEWORKS_REGENERATE_CANDLE_ADMISSION=1 \
  cargo test -p sceneworks-worker candle_admission_decisions
npm run generate:candle-admission
```

At the frozen head the derived route inventory contains 63 Candle routes: 55 image and 8 video.
The generated artifact is the exhaustive route-level record; its important unresolved classes are:

- The shared selector is unreached for **38** routes: **30 image + 8 video**. This count is distinct
  from `summary.byMechanism.unreached = 16`; the latter means no admission mechanism at all, while a
  legacy scalar, conditioning, or flat-video gate may still leave the shared selector unreached.
- The 30 image routes split exactly into four disjoint classes:
  - 15 have no admission mechanism: `anima_aesthetic`, `anima_base`, `anima_turbo`,
    `bernini_image`, `boogu_image_edit`, `chroma1_base`, `chroma1_flash`, `chroma1_hd`,
    `illustrious_xl_v1`, `illustrious_xl_v2`, `realvisxl`, `realvisxl_lightning`, `sana_1600m`,
    `sana_sprint_1600m`, and `sdxl`;
  - 13 reach only the legacy scalar gate: `boogu_image`, `boogu_image_turbo`, `ideogram_4`,
    `ideogram_4_turbo`, `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium`, and the six
    `sensenova_u1_8b*` manifest routes;
  - `kolors` reaches conditioning plus the legacy scalar gate; and
  - `instantid_realvisxl` reaches conditioning only.
- The 8 video routes are the seven route-local flat-fit users (`ltx_2_3`, `mochi_1`, `scail2_14b`,
  `svd`, `wan_2_2`, `wan_2_2_i2v_14b`, and `wan_2_2_t2v_14b`) plus served Bernini, which has no
  Candle manifest block, fit symbol, or pre-load admission gate.
- All 38 records declare no Candle provider contract, but contract absence is not a reason to bypass
  the selector. Pinned inference already supplies `MemoryProviderContract::compatibility_default`:
  an honest resident-only contract with every optimized rung `Missing`, no calibration identity,
  and no fabricated evidence. A route can therefore enter the selector with that compatibility
  view and its existing sourced scalar/floor as the base estimate. The owner decision is narrower:
  when selection returns `Unverified`, preserve the current legacy fallback or disable/refuse the
  route. SC-19059 must not silently choose between those product behaviors.

The memory matrix now fingerprints every named production input SC-19059 inherited:

1. `crates/sceneworks-worker/src/candle_scalar_gate.rs`
2. `crates/sceneworks-worker/src/candle_memory_strategy.rs`
3. `crates/sceneworks-worker/src/video_admission.rs`
4. `crates/sceneworks-worker/src/conditioning_fit.rs`
5. `crates/sceneworks-worker/src/krea_control_fit.rs`
6. `crates/sceneworks-core/src/video_memory_curves.rs`
7. `crates/sceneworks-worker/src/payload.rs`

The generator test pins the exact sorted set, and the terminal regeneration rotates the matrix
provenance. This closes the former `memory-matrix-source-paths-under-cover-candle-admission` gap.

## Evidence that is structural rather than circular

- Geometry is carried by `estimate_synthesis::MemoryGeometry`; curve synthesis prices both pixels
  and frames. The video curve reader uses the explicit `VideoCurveGeometry` boundary recorded as a
  residual below.
- Measurement provenance uses the shared `MeasurementLane` enum. Generic fitted-basis synthesis
  refuses a foreign container lane, `video_memory_curves` refuses a foreign curve lane, and
  `vram_gate` has a Candle-reader/MLX-curve negative test.
- `candle_admission_decisions.rs` is the independent producer for image decision rows. Comparing two
  copies regenerated only by the JavaScript reader is not evidence.
- The Krea grid enumerations are corroborating evidence only. The shipped Krea image lane fixes
  `frames = 1` and has no `perMpxFrameGb`, so those grids cannot witness the temporal term, the
  frame-zero guard, or the voxel-hull branches. Focused unit tests are the load-bearing evidence for
  those symbols.
- A synthetic Candle curve with a deliberately non-live closure can prove the producer, packaging,
  Docker copy, lane coexistence, and generated-artifact gates. It cannot prove runtime binding or a
  decision movement. Only a real curve with the live closure may move the route from floor to fitted
  curve. The sanitized command transcript, red-gate/clear ledger, transient hashes, and byte-identical
  cleanup proof are retained in
  `docs/calibration/sc-19059/synthetic-candle-promotion-rehearsal.md` and its adjacent checksum
  manifest; neither file contains synthetic measurements.

## Source-level divergence inventory

The following search surfaces were reviewed: prediction/evaluation arithmetic in
`mlx_fit_gate.rs`, `vram_gate.rs`, `candle_memory_strategy.rs`, `video_admission.rs`,
`conditioning_fit.rs`, `krea_control_fit.rs`, `estimate_synthesis.rs`, and
`video_memory_curves.rs`; manifest-to-number parsing in `payload.rs`; provider contracts; and every
generated known-gap entry. The inventory is not empty.

### Prediction-law residuals requiring an owner decision

| ID | Residual | Effect | Minimum honest disposition |
| --- | --- | --- | --- |
| D1 | `video_memory_curves::VideoPhaseCurve::evaluate` is a second affine evaluator outside `estimate_synthesis`. | It associates `pixels / 1e6` before multiplication, adds a phase residual, rejects zero pixels/frames, and ceils GiB to bytes. The shared evaluator uses a deliberately different floating-point association and has different boundary semantics. Mechanical unification would change decisions. | Choose and diff one canonical contract, or explicitly approve this cross-crate typed residual. |
| D2 | `KreaTurboPhasePeaks::peak_gb` and `estimate_synthesis::binding_phase` disagree on NaN. | `f64::max` discards a NaN operand while the argmax can keep a first-position NaN as the binding label. Numeric strings reach this path through `payload::json_f64` without a finite guard, so peak and label can disagree. | Prefer a finite guard when constructing the triple, or specify one fail-closed NaN rule and record its decision diff. |
| D3 | The Krea Turbo entry point checks the area hull but not the voxel hull. | It is inert for today's image-only `frames = 1` manifest, but a future temporal or tighter-voxel declaration could be admitted under a contract the generic mechanism would refuse. | Specify whether the entry point is permanently image-only or adopt the full generic hull conjunct with a decision ledger. |
| D4 | `mlx_fit_gate::spec_headroom_bytes` supplies an MLX-calibrated allowance to `video_admission::floor_phase_peaks` on both lanes. | On Candle the imported allowance is floor-additive and therefore conservative: it can over-refuse but cannot create an over-admit. It is still an untagged cross-lane numeric input. | Measure/declare a Candle allowance, parameterize the lane, or owner-approve the conservative residual. |
| D5 | The ultimate Candle image fallback still consumes manifest `vramGbByTier` as a legacy scalar without a runtime lane tag. | This is the designed fail-closed destination when a fitted basis is missing, malformed, stale, or foreign. It preserves the pre-epic floor but is not evidence-classed. | Owner-approve the legacy destination explicitly, or replace it only with a sourced lane-tagged floor. |

For D1, the capability choices are concrete and incompatible:

1. Move the runtime curve contract into a dependency-neutral shared core while preserving its current
   residual, zero-input, rounding, and floating-point association semantics. The generic caller must
   adapt to that contract or retain a typed adapter.
2. Adopt the generic Krea association and boundaries for video curves. This changes at least one
   lane's byte output and requires a complete decision-diff review.
3. Retain both evaluators and approve D1 as an intentional cross-crate residual, with tests pinning
   the semantic differences so neither is later described as pure duplication.

There is no repository source of truth choosing among these. SC-19059 must not silently make this
product decision under the label of refactoring.

### Capability, evidence, and coverage residuals

| ID | Residual | Current effect / terminal requirement |
| --- | --- | --- |
| C1 | Selector reach and compatibility contracts | The 23 source-backed routes described above now reach the shared selector through resident-only no-fabrication compatibility contracts. Fifteen remain unreached: 14 evidence-free image routes plus Bernini video. Their fallback-versus-disable policy is still unresolved and was not chosen here. |
| C2 | Bernini Candle video | A real served route has no Candle block, fit symbol, or pre-load gate. It remains entirely ungated. |
| C3 | Bare `measured` boolean | The manifest cannot distinguish a fitted curve, a single measured point, and a declared floor. No evidence-class producer exists to adopt. |
| C4 | No packaged Candle video curve before capture | The bundle contains one MLX curve and zero Candle curves. The real SC-19057 campaign must add, validate, and bind the first Candle curve; synthetic evidence must not be retained. |
| C5 | Linked Candle sequential capability | The CPU lane cannot resolve the actual `engine_supports_sequential` answers because the Candle provider bundle needs CUDA to link. The baseline therefore records both capability inputs. Resolve in the privileged Candle lane. |
| C6 | `bounded_by` on Candle | The composition law and tests exist, but no current Candle provider declares a non-`None` `bounded_by`; this is unexercised future-contract behavior, not production Candle coverage. |
| C7 | Decision artifact blind spots | Video and Krea-runtime rows are `not_evaluated`. Their admission evidence lives in focused tests and reviewed ledgers until the generator has a non-fabricated evaluable context. Do not infer stability from byte-identical JSON. |
| C8 | SC-20573 bespoke-route dispatch | The current exclusion is inaccurate. `instantid_realvisxl` has no inference memory-calibration contract or envelope, so it is not calibration-scoped and must leave the scoped census rather than be described as an unswept gate-bearing model. `pulid_flux_dev` is calibration-scoped and CPU-probeable at pin despite having no `MODEL_TABLE` row. On MLX, `runtime_macos::providers::pulid` exposes provider id `pulid_flux`; `mlx-gen-pulid` registers both `MemoryRegistration` and `MemoryBehaviorRegistration`, with public `weights_free_contract` and `registered_fixture`. On Candle, `runtime_cuda::providers::pulid::memory_strategy` exposes the same provider id and public `provider_contract`/`safety_check`; its crate test fixture constructs minimal safetensors for the FLUX base, PuLID, EVA, and three face networks, so the contract probe needs no CUDA device or real weights. SC-20573 should dispatch `pulid_flux_dev` through these backend-specific provider fixtures before the `MODEL_TABLE` lookup, remove InstantID from the calibration census, and then tighten exact-set equality. That dispatch repair is not implemented by SC-19059. |

## Decision-diff ledger from SC-19049 to the frozen head

The baseline and current Rust-produced files each contain 2,540 rows. Fifty-two routes are evaluated
and 11 are recorded as not evaluated. The end-to-end JSON comparison reports 2,512 rows with metadata
movement, mostly `geometrySensitive`, `scalarClass`, and prediction fields introduced by the new
mechanism. Exactly 31 rows change an admission or load-plan verdict. Every change is conservative;
there are no new admits.

For the first nine groups below, capable changes are `fits -> offload` and `resident -> sequential`;
incapable changes are `fits -> too_big` and `resident -> reject`.

| Model / engine | Tier | Budget (GiB) | Geometries | Rows | Justification |
| --- | --- | ---: | --- | ---: | --- |
| `flux2_dev` / `flux2_dev` | bf16 | 48 | 1024, 1536, 2048 square | 3 | The 48 GiB declared estimate is conservatively graded to 49.92 GiB. |
| `flux_dev` / `flux1_dev` | bf16 | 24 | 1024, 1536, 2048 square | 3 | The 24 GiB declared estimate is graded to 24.96 GiB. |
| `flux_dev` / `flux1_dev` | q4 | 24 | 1536, 2048 square | 2 | Geometry-sensitive synthesis reaches the same graded 24.96 GiB boundary. |
| `flux_schnell` / `flux1_schnell` | bf16 | 24 | 1024, 1536, 2048 square | 3 | Declared-scalar grading removes an unproven boundary admission. |
| `flux_schnell` / `flux1_schnell` | q4 | 24 | 1024, 1536, 2048 square | 3 | Same conservative declared-scalar grade. |
| `sd3_5_large` | q8 | 48 | 1024, 1536, 2048 square | 3 | The 48 GiB floor is graded to 49.92 GiB. |
| `sd3_5_large` | nvfp4 | 48 | 1024, 1536, 2048 square | 3 | Same conservative floor grade. |
| `sd3_5_large_turbo` | q8 | 48 | 1024, 1536, 2048 square | 3 | Same conservative floor grade. |
| `sd3_5_large_turbo` | nvfp4 | 48 | 1024, 1536, 2048 square | 3 | Same conservative floor grade. |
| `z_image_turbo` | q8 | 12 | 1536, 2048 square | 2 | Sequential plan `11.8 -> 12.272` GiB; capable load plan changes `sequential -> reject`. |
| `z_image_turbo` | nvfp4 | 12 | 1024, 1536, 2048 square | 3 | Same sequential grade; `sequential -> reject`. |

The committed image artifact cannot witness the video changes. The reviewed video ledger adds:

- Wan A14B q8 T2V: `29.950 -> 31.148` GiB. A 30.7 GiB budget changes admit to refuse; a full
  32.0 GiB budget remains admitted.
- Wan A14B q8 I2V: `30.020 -> 31.221` GiB, with the same 30.7/refuse and 32.0/admit boundary.
- SCAIL-2 14B: advertised minimum `105 -> 109` GiB, aligned with graded demand 108.28 GiB
  (ceil 109). A 105 GiB host now refuses and 109 GiB admits.
- Wan 2.2 5B and Mochi enter the declared-floor mechanism, but the reviewed ledger records no
  boundary verdict flip in the tested shipped budgets. LTX and SVD deliberately retain their
  pre-existing ungraded postures for documented, route-specific reasons.

After the real Candle curve is promoted, this ledger is necessarily provisional. A correctly bound
curve is expected to move the captured route from `EstimateFloor` to `EstimateFittedCurve`. Zero
movement is suspicious and requires checking the inference closure handshake. The Rust producer and
the full ledger must be rerun at the final head.

## Cross-epic merge and regeneration order

1. Epic 18803 established the two-axis video ladder and the existing MLX curve.
2. Epic 18472 added Candle capability routes and provider contracts while this feature branch was
   open. MiniMax/main changes were merged and reviewed into the frozen head above.
3. SC-19057 runs once at the frozen final paired SceneWorks/inference head. The capture artifacts
   must carry the real Candle lane and live inference closure; synthetic rehearsals are discarded.
4. Promote the real curve through the documented source-catalog entry and both Docker copies, then
   run both lane-specific curve checks. Regenerate the Rust decision surface, Candle inventory, and
   memory matrix in that order.
5. Run the privileged `windows-candle` and `macos-mlx` workflows at the exact final feature head and
   retain their URLs and terminal conclusions.
6. Merge inference first if its final head moved, update the exact SceneWorks pin, then merge the
   reviewed SceneWorks feature PR. Re-run post-merge generated-artifact checks on `main` before any
   Shortcut closeout.

## Remaining terminal gates

- Owner disposition for D1-D5, the 14 evidence-free image routes' C1
  `Unverified` fallback-versus-disable choice, and Bernini non-T2V/C2-C3; the inventory is currently
  non-empty.
- Real SC-19057 Candle capture and promotion, with no synthetic curve or fabricated closure retained.
- CUDA-linked sequential-capability resolution and Candle acceptance.
- Final-head macOS MLX and Windows Candle workflow URLs with terminal required CI.
- Non-circular decision regeneration and a final post-capture decision-diff review.
- Reconciliation and regeneration at the eventual merge head, followed by post-merge `main`
  verification.
