# SC-16170 — Z-Image Candle/CUDA memory-ladder certification

This certification covers the hosted `SceneWorks/z-image-mlx` q4, q8, and bf16 tiers at revision
`c74f74c2ad193294fc9ff3f8a5be71daa00d22ab`, the base Fun-Controlnet Union 2.1 checkpoint pinned at
`755999a934909bd5832e20718bb7c639d2a63eb9`, the rank-1 LoRA fixture with SHA-256
`cd248717d74f77dce1964680ce7764c45c3648571d09d11ff550b59862bc7072`, and inference revision
`65833f24409bce33c418499c2029bee76d842740`. The runtime source baseline is SceneWorks revision
`e87abd092f2e83452c0e527d69f85fc4d6465242`.

The committed byte inventories bind those identities to the inputs used by the probes. Their canonical
tree/file SHA-256 values are q4 `a05de90591f2255bfeac50f5dd04a2862ae6c19f377c680adb1383fd3852ec34`,
q8 `27d1a1d479cd8fd8afb9995a84de0bf615cc08d32b827149b599ecd7b978e530`, bf16
`4f53a37548a038d63503fa71ff620fa18195873fe4c2759a4c8beee11e575940`, ControlNet
`2393b0c58c52a12134f6ffd96ff9b6ea3c80bb233665fb2c3b9aebcee71ae3e4`, and the LoRA fixture hash
shown above. During deterministic reconstruction, the certification script joins each physical or
comparison transcript to the separately captured inventory for its declared tier and overlay. The
resulting typed source session carries those exact inputs, byte counts, repositories, revisions,
variants, the transcript's committed path, and its exact-byte hash; validation rejects missing or
mismatched tier/overlay inputs.

Measurements used GPU 0 of an NVIDIA RTX PRO 6000 Blackwell Max-Q workstation (compute capability
12.0, 96 GB), driver 596.36 and CUDA 12.9. Every physical memory trial ran in a fresh process at
1024×1024, batch 1, one frame, with an idle baseline below 1 GB. The committed transcripts under
`docs/calibration/sc-16170/` are treated as binary because the evidence bundle hashes their exact bytes.

## Direct strict-control memory envelope

The strict-control route is the family high-water and was measured directly for all 15 tier/rung
combinations. Values are decimal GB. The committed prediction rounds above the largest observed
phase; records for plain and LoRA routes identify these control trials as conservative upper-bound
derivations rather than pretending they were separate physical measurements.

| Tier | Rung | Pre-decode | Decode | Certified overall |
|---|---|---:|---:|---:|
| q4 | resident | 27.117 | 27.117 | 27.2 |
| q4 | staged | 16.816 | 16.044 | 16.9 |
| q4 | bounded decode | 16.816 | 11.212 | 16.9 |
| q4 | bounded attention | 13.297 | 11.210 | 13.4 |
| q4 | streamed transformer | 13.297 | 2.016 | 13.4 |
| q8 | resident | 32.050 | 32.050 | 32.1 |
| q8 | staged | 19.802 | 15.876 | 19.9 |
| q8 | bounded decode | 19.802 | 14.232 | 19.9 |
| q8 | bounded attention | 16.042 | 14.264 | 16.1 |
| q8 | streamed transformer | 13.297 | 2.989 | 13.4 |
| bf16 | resident | 40.573 | 40.573 | 40.7 |
| bf16 | staged | 25.272 | 19.702 | 25.4 |
| bf16 | bounded decode | 25.272 | 19.702 | 25.4 |
| bf16 | bounded attention | 21.444 | 19.700 | 21.5 |
| bf16 | streamed transformer | 13.096 | 2.956 | 13.2 |

Bounded decode now transfers latent spatial rows in the advertised 512/128 tuple and reassembles the
exact full latent before whole-frame CPU VAE decode. This makes the tuple operational while preserving
the VAE's global normalization. The transfer has cancellation checks at row boundaries and before the
CPU VAE call; certification claims mid-denoise/boundary cancellation, not interruption inside the CPU
VAE kernel.

## Quality gates

Each tier compares fixed-seed resident output with every ladder rung. Staged output is byte-identical to
resident. The host-decode rungs share an identical output within each tier:

| Tier | Maximum RGB8 delta | Mean RGB8 delta | Normalized maximum / mean |
|---|---:|---:|---:|
| q4 | 7 | 0.280577342 | 0.027451 / 0.001100 |
| q8 | 24 | 0.297257741 | 0.094118 / 0.001166 |
| bf16 | 37 | 0.436488469 | 0.145098 / 0.001712 |

The calibrated contract is maximum normalized RGB8 error ≤ 0.15 and mean error ≤ 0.002. The maximum
threshold is intentionally tier-aware evidence, not the earlier q4-only 0.04 assumption. A retained
independently normalized tiled-decode mutation measures 69/255 maximum and 3.278849/255 mean, failing
both final thresholds and proving the gate detects the known bad decoder.

## Overlay and mode evidence

- LoRA: every tier runs one fixed-seed resident and one streamed-transformer render. Each run reloads
  an unadapted generator and requires adapted output to differ; resident-versus-streamed comparison
  then enforces the final-output quality contract. Optimized admission adds the
  actual request's adapter-resident bytes before fit selection at q4, q8, and dense bf16, so arbitrary
  adapter sizes cannot borrow the tiny fixture's memory envelope.
- Style/reference: every tier runs a reference-conditioned generation and a fixed-seed plain
  comparison. Reference-on versus reference-off differs substantially in all tiers, proving the field is
  consumed.
- Control: every tier/rung physically loads the pinned base Fun-Controlnet checkpoint. Environment
  overrides remain available for resident execution but are explicitly ineligible for optimized evidence
  because they did not pass the pinned revision/LFS path.
- Mode aliases and lighter-overlay memory reuse are recorded as `identical_component`,
  `shared_implementation`, or `conservative_upper_bound` derivations. They are never labeled direct.

## Provenance and fail-closed behavior

The committed historical capture set describes 90 logical cases across 49 source sessions: five exact
artifact inventories, 15 physical control-memory trials, 15 resident-versus-rung image comparisons,
three style trials, six LoRA trials, three LoRA comparisons, one lifecycle test transcript, and one
negative-mutation comparison. The reconstruction uses the 48 `6583-*` sessions plus the retained
`7410-negative-mutation.log`; the other `7410-*` transcripts are historical and are not attached to any
case.

These captures predate the schema-v4 `loadShape` axis. Because eager versus deferred materialization was
not recorded during collection, the 90 cases are validated as historical evidence but are not promoted
into the current authoritative bundle and do not create manifest calibration bindings. The current
bundle therefore retains only the 24 independently typed Qwen records. Re-certifying Z-Image requires
new measurements that record the declared `deferred_materialization` load shape from the plan.

The historical reconstructor rejects missing source sessions, duplicate claims, cross-tier memory/quality
reuse, cross-rung memory reuse, and direct overlay claims from a different overlay. Schema and runtime
validation reject incomplete typed evidence, stale inference pins, calibration-fingerprint mismatches,
unknown budgets, wrong geometry, and uncertified ControlNet overrides. The production base route resolves the immutable
`SceneWorks/z-image-mlx` snapshot revision rather than mutable `refs/main`; explicit `modelPath` and any
resolved base outside that pinned tier remain resident-only.

## Reproduction and verification

The inference repository provides `control_vram_probe`, `sequential_vram_probe`, and the unit/integration
suite. SceneWorks validates the historical sources and writes the fail-closed current bundle with:

```powershell
node scripts/sc-16170-certification.mjs --write
node scripts/sc-16170-certification.mjs
node scripts/memory-calibration-harness.mjs check --input docs/generated/memory-calibration-evidence.json
node scripts/generate-memory-matrix.mjs
node scripts/calibration-cost-model.mjs
```

The check command reconstructs all 90 historical cases from every committed transcript and inventory
log, recomputes each transcript's exact-byte hash and every inventory digest, and verifies that the
checked-in evidence and manifests contain no untyped SC-16170 promotion. Consequently the current matrix
does not report these Z-Image/Candle cells as Verified. Their raw measurements remain available for audit,
while runtime admission fails closed until schema-v4 measurements replace them.

The base and LoRA cells bind the packed-sidecar contract
`z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2`; strict-control cells bind
`z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2`. Keeping these identities
separate prevents control evidence from admitting a base request (or vice versa) after either execution
structure changes.
