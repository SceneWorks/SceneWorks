# Epic 18093 end-to-end validation record (sc-18101)

On-device validation of the whole right-sized calibration apparatus, run after every epic 18093
story had merged to `main`. This file is the durable record; the harness that produced it is
`crates/sceneworks-worker/src/ladder_e2e_sc18101.rs` (four `#[ignore]`d tests, one per criterion).

**Verdict: all five success criteria pass. No margin breach.**

## Environment

| | |
|---|---|
| Host | Apple M5 Max, `Mac17,6`, 128 GiB unified memory, macOS 26.5.2 (the `nax-macos` self-hosted runner) |
| SceneWorks revision under test | `8842036d` (`main` after the sc-18100 merge — the last epic 18093 story) |
| Pre-epic baseline revision | `de756026` (`main` immediately before the sc-18094 merge `cc799d1a`) |
| Inference pin | `40fa7583a01974617e2a7275052d6d446688c956` |
| Evidence corpus | `docs/generated/memory-calibration-evidence.json`, 65 records (50 MLX / 15 candle) |
| Lane currency | 8 stale, 0 current, 1 unmeasured (`npm run report:stale-lanes`) |
| Profile | `dev` (`cargo test -p sceneworks-worker --lib`) |

Artifacts (renders, per-criterion selection logs, machine-readable rows) are written outside the
repo to `~/SceneWorks/render-validation-sc18101/`.

## What the emulated cap does and does not prove

`SCENEWORKS_MLX_MEMORY_CAP_GB` replaces the machine's real unified-memory total in
`mlx_fit_gate::resolve_budget`. It drives admission arithmetic exactly as a smaller Mac would, and
it is read on **every** request and on **every** cold load — so the whole production chain
(load-time residency decision → provider contract → request-scoped ladder) sees it.

It does not shrink the Metal heap. The render that follows still runs against the real 128 GiB
pool, so a peak sampled under a cap is what that geometry and rung genuinely cost. On a machine
that really was that small, page-cache eviction would add re-materialization traffic on top; the
numbers below are therefore a **lower bound** on the small-machine cost. That is the safe direction
for validating a margin — an emulated run cannot flatter the prediction.

## Criterion 1 — an unmeasured cell engages a deep rung and completes a real render

**Subject:** `qwen_image` q8, text-to-image, no overlay, **768×768**.
Artifact `SceneWorks/qwen-image-mlx@8080a4171f1c8b7fca6c30491eafbe6ffab754bf:q8` — the exact
snapshot the shipped opt-in names, present in this machine's HF cache.

The MLX corpus holds `qwen_image` q8 only at 1024². The 768² request cell has **no calibration
record and no manifest binding**, so `evidence_admission_route` finds no geometry match, routes to
`AdmissionPath::Legacy` with `fallback_reason = OutOfEnvelope`, and the request is served entirely
from synthesized estimates.

| | |
|---|---|
| Emulated cap | **32 GB** |
| Load-time residency decision | `OffloadPolicy::Sequential` (chosen by `apply_residency_policy` under the same cap) |
| Rung engaged | **`BoundedTransformerResidency`** (rung 4) |
| Parameters | `decode_tile_edge=256`, `decode_overlap=64`, `attention_chunk_size=67108864`, `transformer_window_size=1` |
| Basis | `floor` (weights + headroom; no measured basis exists for this cell) |
| Raw floor estimate | 26 967 416 751 B = **25.12 GiB** |
| Estimate margin applied | **0.10** (`MLX_ESTIMATE_MARGIN`) |
| Admitted ceiling (predicted-with-margin) | 29 664 158 427 B = **27.63 GiB** |
| Observed peak (`mlx_rs::memory::get_peak_memory`, active) | 15 975 492 864 B = **14.88 GiB** |
| Headroom against the admitted ceiling | **13 688 665 563 B = 12.75 GiB (46.1 % under)** |
| Render | 768×768, 8 steps, seed 18101, 92.6 s, pixel std 46.25, non-degenerate |
| PNG | `~/SceneWorks/render-validation-sc18101/c1-qwen_image-q8-768x768-main.png` |

Log evidence (`c1-selection-main.log`), estimate-scoped selection and the applied margin:

```
INFO  synthesized weights+headroom floor estimate candidate route="qwen_image" backend="mlx"
      strategy=BoundedTransformerResidency raw_peak_bytes=26967416751
INFO  estimate-backed memory-strategy candidate admitted with widened margin route="qwen_image"
      backend="mlx" strategy=BoundedTransformerResidency basis="floor"
      raw_peak_bytes=26967416751 widened_peak_bytes=29664158427 estimate_margin=0.1
WARN  memory-strategy selection uses an estimate-backed candidate at the widened peak
      route="qwen_image" backend="mlx" strategy=BoundedTransformerResidency basis="floor"
      raw_peak_bytes=26967416751 widened_peak_bytes=29664158427 estimate_margin=0.1
```

Independent corroboration of the observed number: the corpus's **measured** rung-4 cell for the
same provider/tier at the *larger* 1024² geometry (`imc-4426a6e84c4d39d9bff3`) recorded
15 977 625 784 B observed. The unmeasured 768² run came in at 15 975 492 864 B — within 0.014 % of
it. Rung 4 windows the transformer, so the request peak is carried by the area-flat conditioning
phase, which is exactly why the two geometries agree; the floor estimate does not model that and
over-predicts by 46 %, which is the conservative direction.

### The cap ladder

Recorded from the harness's sweep mode (same subject, same request):

| cap (GB) | load policy | rung selected | raw estimate |
|---:|---|---|---:|
| 128 → 56 | `Resident` | `Resident` | 45.57 GiB |
| 48 → 36 | `Sequential` | `StagedResidency` | 30.13 GiB |
| **32, 30** | `Sequential` | **`BoundedTransformerResidency`** | **25.12 GiB** |
| 28 → 24 | `Sequential` | refused — `needs 27.63 GiB but only 26.00 GiB is safely available` |  |
| ≤ 22 | — | refused at the **load-time** gate before the ladder is reached |  |

## Criterion 2 — a measured-current cell selects identically to pre-epic main

**Commits compared:** `8842036d` (main, all of epic 18093) vs **`de756026`** (main immediately
before sc-18094 merged).

**Subject:** `qwen_image` q8 1024² — a cell with a shipped binding and a verified record — at a
96 GB emulated cap.

### Which reading of "measured-current" was used, and why

The shipped corpus has **zero** closure-current lanes, so the phrase has two possible readings:

1. *measured and admitted* — take a cell as it ships. **Rejected.** Every shipped MLX cell is
   closure-STALE, and pre-epic main excluded stale candidates outright. Comparing that against main
   would compare the epic's intended behaviour change with itself and prove nothing.
2. *measured and closure-current* — the calibrated happy path the epic promises not to disturb.
   **This is the reading used.** Since no shipped lane is current, the cell was made current with a
   **scratch closure table**: `config/inference-provider-closures.json`'s
   `providers["mlx:qwen_image"].digest` was temporarily set to
   `54a7b45b03eb8301e6e85fd1d67558d007ebe5031eb491fe25e95b8f081f4374` — the digest the q8 ladder
   was captured under — in *both* checkouts, i.e. the world in which nobody has touched the
   provider since the measurement. The edit was reverted after the runs; neither checkout carries
   it now.

Reading 2 is the only one that can distinguish "the epic left the calibrated path alone" from "the
epic changed everything", which is what the criterion exists to check.

### Result: byte-for-byte identical

Both the selection fingerprint and the **entire** captured tracing output are identical
(`diff` clean) between the two commits:

```
SELECTED strategy=BoundedAttention
  parameters=MemoryStrategyParameters { decode_tile_edge: Some(512), decode_overlap: Some(64),
             attention_chunk_size: Some(67108864), transformer_window_size: None,
             transformer_window_component: None }
  tier=MemoryNumericTier { precision: Bf16, quant: Some(Q8), component_precision_floors: [] }
  predicted_peak_bytes=46305116160 process_limit_bytes=Some(52684932164)
  stage_residency=false tile_vae_decode=true chunk_attention=true stream_transformer_blocks=false
```

Both commits emit `path=Evidence fallback_reason=None`, select record
`imc-37f40254d20bc43fa925`, and emit **no** widening line — a current candidate is graded at its
raw peak, unwidened, on both sides.

The baseline checkout carried exactly one local, uncommitted change beyond the scratch closure
table: the harness module itself, plus its one-line registration in `lib.rs` (and, in that
checkout, `margin_literals_match_the_policy` elided, since `ladder_margin_policy` does not exist
before sc-18094). No production source differs between the two runs.

### Contrast: the same cell as it actually ships (stale)

With the shipped (stale) closure table, the two commits diverge exactly as the epic intends:

| | pre-epic `de756026` | main `8842036d` |
|---|---|---|
| admission path | `Legacy`, `fallback_reason=StaleIdentity` | `Evidence`, `fallback_reason=None` |
| measured candidate | none — excluded before the selector | admitted, carrying its captured digest |
| rung selected | `Resident` (frozen to the baseline) | `BoundedAttention` |
| needed | 45.57 GiB (resident estimate) | 45.28 GiB (measured peak, widened 5 %) |

## Criterion 3 — a stale lane still admits, at the widened peak, and logs it

**Lane:** `mlx:qwen_image` — stale as shipped (live closure `9930aa538259…`, captured
`54a7b45b03eb…`). No test seam was needed; this is the corpus's actual state.

| | |
|---|---|
| Cell | `qwen_image` q8 1024², rung `BoundedAttention`, record `imc-37f40254d20bc43fa925` |
| Raw measured peak | 46 305 116 160 B = **43.125 GiB** |
| Stale margin applied | **0.05** (`MLX_STALE_MEASURED_MARGIN`) |
| Widened admitted peak | 48 620 371 968 B = **45.281 GiB** |

`widened == ceil(raw × 1.05)` is asserted, not eyeballed. Log evidence:

```
INFO  stale-closure memory-strategy candidate admitted with widened margin route="qwen_image"
      backend="mlx" strategy=BoundedAttention raw_peak_bytes=46305116160
      widened_peak_bytes=48620371968 stale_margin=0.05
      candidate_closure_digest="54a7b45b03eb8301e6e85fd1d67558d007ebe5031eb491fe25e95b8f081f4374"
      expected_closure_digest="9930aa538259f7c576c13e3241e872a6486b049fe62b423e5dca3b3fe56f7bae"
WARN  memory-strategy selection uses stale-closure evidence at the widened peak route="qwen_image"
      backend="mlx" strategy=BoundedAttention raw_peak_bytes=46305116160
      widened_peak_bytes=48620371968 stale_margin=0.05
```

The pre-epic contrast above is the proof that the widening is doing work rather than decorating a
decision that would have happened anyway: on `de756026` this same cell produced no measured
candidate at all.

## Criterion 4 — an oversized request is refused, not OOM-killed

**Request:** `qwen_image` q8 **2048×2048** at a 24 GB emulated cap, loaded `Sequential` so the
provider offers its deepest possible ladder.

The full five-rung estimate ladder is synthesized and every rung is graded — so the refusal is
"nothing fits with margins", not "a rung was missing":

| rung | raw floor | widened (×1.10) |
|---|---:|---:|
| `Resident` | 73 641 068 818 | 81 005 175 700 |
| `StagedResidency` | 57 056 202 530 | 62 761 822 784 |
| `BoundedDecode` | 73 640 535 842 | 81 004 589 427 |
| `BoundedAttention` | 73 640 535 842 | 81 004 589 427 |
| `BoundedTransformerResidency` | 51 674 216 124 | **56 841 637 737 = 52.94 GiB** |

Refusal, quoting the deepest widened requirement:

```
qwen_image request 2048x2048 count 1 needs 52.94 GiB but only 22.00 GiB is safely available
```

**Proof it refused rather than averted an OOM** is structural, not statistical:

* the refusal is produced by `mlx_fit_gate::evaluate_request`, which runs **before**
  `generator.generate`, so no allocation for the request is ever attempted;
* MLX's default error handler calls `exit(-1)` on an allocator overshoot, so an OOM would end the
  process. The test continues in the same process and re-submits the identical request at a 512 GB
  cap, which is admitted. Both facts are asserted.

## Criterion 5 — suites

See the PR for CI. Locally: `npm run check`, `npm run rust:check`, the parity pytest gate
(`python -m pytest -q tests/ -m "not e2e and not parity" --strict-markers`), and `cargo fmt --all`.

## Findings worth carrying forward

1. **No MLX provider+tier is wholly unmeasured in the corpus.** All three MLX lanes
   (`qwen_image`, `z_image_turbo`, `krea_2_turbo_control`) have records; the single unmeasured lane
   is `candle:z_image`, which has no adapter arm. "An unmeasured cell" on the MLX side therefore
   means an uncalibrated *coordinate* — a geometry, tier, or rung with no record — which is what
   criterion 1 exercises. Worth stating plainly so a future reader does not go looking for an
   unmeasured MLX lane that does not exist.

2. **On a floor-only ladder, rung 4 is the only deep rung that can ever be engaged.** By deliberate
   design (`mlx_fit_gate.rs:1485-1488`), bounded decode and bounded attention bound *transients*,
   not weights, and take no floor reduction on an unmeasured cell — so their floors equal
   `Resident`'s and no budget can admit them while excluding `Resident`. A provider that does not
   implement `BoundedTransformerResidency` therefore has a **flat** estimate ladder: under a cap it
   can reach `StagedResidency` at best and then refuses. `krea_2_turbo` bf16 is such a provider
   (its rung 4 needs `streamable_transformer`, which the bf16 route does not offer), measured here
   at a 44 GB cap selecting `StagedResidency` at a 34.62 GiB floor. This is correct behaviour, not
   a defect — but it does mean the epic's "estimates admit the full ladder" delivers a *two*-rung
   improvement, not a five-rung one, on providers without a windowed transformer.

3. **The memory-matrix source fingerprint makes `base.rs` expensive to touch, even by one keyword.**
   `image_jobs/base.rs` is one of the sixteen `SOURCE_PATHS` the matrix hashes, so widening
   `apply_measured_mlx_load_shape` from `fn` to `pub(crate) fn` — purely so a test could call it —
   rotated `generatedFrom.sceneWorksRevision` and reddened `npm run check:memory-matrix`. The
   harness therefore mirrors the route list instead, with a text-coupling test
   (`load_shape_mirror_matches_the_documented_routes`) that reds if production's list changes. Any
   future change to a fingerprinted source should expect to regenerate the 860 KB artifact and
   every binding's `matrixSourceRevision`.

4. **Which rungs a provider declares is a function of the `LoadSpec`, not of the request.**
   `BoundedTransformerResidency` requires `LoadShape::DeferredMaterialization` (z-image, lens) or
   `OffloadPolicy::Sequential` (krea). Both are reached only through the LOAD-time gate, so a
   request-scoped test that loads `Resident` sees a flat ladder for reasons unrelated to epic
   18093. The harness runs the cap through `apply_residency_policy` first for exactly this reason.

5. **Moving only the manifest binding's `inferenceClosureDigest` does not make a lane current.**
   `EvidenceBundle::evidence_for` compares the binding's digest against the record's own stamp too,
   so a one-sided edit makes the record unfindable and the cell routes to `StaleIdentity` with no
   measured candidate at all — observed directly during this validation. A real re-measurement
   moves both halves together, which is why `docs/calibration-runbook.md` §7d insists on both. A
   scratch closure table is the correct single-edit lever for a test.

6. **The q8 `bounded_attention` binding is not reachable through the production load path.**
   `image_jobs::apply_measured_mlx_load_shape` forces `DeferredMaterialization` on every
   `qwen_image` directory load, while that binding was captured `eager_materialization`; under the
   production spec the candidate is excluded `FingerprintMismatch` and the cell falls to the
   estimate ladder. Criteria 2 and 3 therefore run with `SC18101_MEASURED_EAGER=1`, which loads the
   eager spec the binding was captured under. Worth a look on its own terms — see the follow-up
   note on the story.

## Re-running

```text
OUT=~/SceneWorks/render-validation-sc18101

# criterion 1 — cap sweep first (unset SC18101_C1_CAP_GB prints the ladder and fails with it)
SC18101_TAG=main SC18101_C1_REPO=models--SceneWorks--qwen-image-mlx SC18101_C1_ENGINE=qwen_image \
SC18101_C1_TIER=q8 SC18101_C1_W=768 SC18101_C1_H=768 SC18101_C1_CAP_GB=32 SC18101_C1_STEPS=8 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c1_unmeasured_cell_engages_a_deep_rung_and_renders

# criterion 3 (stale, as shipped) and criterion 4 (refusal)
SC18101_TAG=main SC18101_MEASURED_EAGER=1 SC18101_C3_ADMIT_CAP_GB=96 \
SC18101_C1_REPO=models--SceneWorks--qwen-image-mlx SC18101_C1_ENGINE=qwen_image \
SC18101_C1_TIER=q8 SC18101_C4_CAP_GB=24 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c3_stale_lane_admits_at_the_widened_peak c4_oversized

# criterion 2 — needs the scratch closure table in BOTH checkouts first:
#   config/inference-provider-closures.json providers["mlx:qwen_image"].digest = 54a7b45b03eb…
# then run the same command in each and diff $OUT/c2-fingerprint-{main,baseline}.log
SC18101_TAG=main SC18101_MEASURED_EAGER=1 SC18101_C2_CAP_GB=96 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c2_measured_current_cell_selection
```
