# Epic 18093 end-to-end validation record (sc-18101)

On-device validation of the whole right-sized calibration apparatus, run after every epic 18093
story had merged to `main`. This file is the durable record; the harness that produced it is
`crates/sceneworks-worker/src/ladder_e2e_sc18101.rs` (five `#[ignore]`d scenarios plus a portable
regression probe).

**Verdict: all five success criteria pass, no margin breach — and the validation found and fixed a
REGRESSION this epic shipped.** That regression is the most important thing in this document, so it
comes first.

## Environment

| | |
|---|---|
| Host | Apple M5 Max, `Mac17,6`, 128 GiB unified memory, macOS 26.5.2 (the `nax-macos` self-hosted runner) |
| SceneWorks revision under test | `main` after the sc-18100 and sc-18104 merges, plus this branch |
| Pre-epic baseline revision | `de756026` (`main` immediately before the sc-18094 merge `cc799d1a`) |
| Inference pin | `40fa7583a01974617e2a7275052d6d446688c956` |
| Evidence corpus | `docs/generated/memory-calibration-evidence.json`, 65 records (50 MLX / 15 candle) |
| Lane currency | 8 stale, 0 current, 1 unmeasured (`npm run report:stale-lanes`) |
| Profile | `dev` (`cargo test -p sceneworks-worker --lib`) |

Artifacts (renders, per-scenario selection logs, machine-readable rows) are written outside the repo
to `~/SceneWorks/render-validation-sc18101/`.

## What the emulated cap does and does not prove

`SCENEWORKS_MLX_MEMORY_CAP_GB` replaces the machine's real unified-memory total in
`mlx_fit_gate::resolve_budget`. It drives admission arithmetic exactly as a smaller Mac would, and it
is read on **every** request and on **every** cold load — so the whole production chain (load-time
residency decision → provider contract → request-scoped ladder) sees it.

It does not shrink the Metal heap. The render that follows still runs against the real 128 GiB pool,
so a peak sampled under a cap is what that geometry and rung genuinely cost. On a machine that really
was that small, page-cache eviction would add re-materialization traffic on top; the numbers below
are therefore a **lower bound** on the small-machine cost. That is the safe direction for validating
a margin — an emulated run cannot flatter the prediction.

---

# 0. The regression this epic shipped, and its fix

**The flagship `qwen_image` q8 route hard-refused its most common geometry, at a 96 GB cap on a
128 GiB machine, under the production `LoadSpec`.**

```
InvalidPayload("qwen_image request 1024x1024 count 1 has no structurally admissible MLX
                memory strategy (FingerprintMismatch); refusing to enter MLX's
                process-terminating allocation path")
```

`image_jobs/base.rs` propagates that with `?` from its per-generation `evaluate_request` call, so it
is a user-facing job failure, not a degraded prediction.

### Why it happened

`MemoryCalibrationIdentity` has three fields — `abi`, `fingerprint`, and `load_shape`. gen-core's
`optimized_eligibility` rejects a candidate whose `key.load_shape` disagrees with either
`contract.load_shape` or `identity.load_shape`, returning `FingerprintMismatch`. The identity
demotion in `evaluate_request_with_budget_using_bundle` compared only the first two fields, so a
shape mismatch sailed past it, reached `AdmissionPath::Evidence`, lost every candidate inside
`select_strategy` — and refused, because estimate synthesis runs **only** on the Legacy route.

The mismatch itself is real and pre-existing: the shipped q8 `bounded_attention` binding was captured
`eager_materialization`, while `image_jobs::apply_measured_mlx_load_shape` forces
`DeferredMaterialization` on every `qwen_image` directory load. What this epic changed is what
happens next. Before sc-18096, `evidence_admission_route`'s identity filter also required
`binding.query.inference_closure_digest == expected_closure_digest`; every shipped MLX lane is
closure-stale, so that conjunct pre-demoted the cell to `AdmissionPath::Legacy` long before
eligibility ran, and it was served. sc-18096 retired the conjunct — correctly, that was its whole
point — and in doing so made a previously unreachable refusal reachable on a shipping route.

### The evidence, from both commits

Identical source (`c0_production_loadspec_probe`, which records the outcome and asserts nothing, so
it can run on both sides), same weights, same cap:

| commit | admission route | outcome |
|---|---|---|
| `de756026` (pre-epic) | `Legacy`, `fallback_reason=StaleIdentity` | **ADMITTED** `Resident`, `predicted_peak_bytes=48934269317` |
| epic `main` | `Evidence`, `fallback_reason=None` | **REFUSED** `FingerprintMismatch` |
| this branch (fixed) | `Legacy`, `fallback_reason=StaleIdentity` | **ADMITTED** `Resident`, `predicted_peak_bytes=48934269317` |

`diff c0-outcome-main.log c0-outcome-baseline.log` is clean after the fix. That equality is of the
OUTCOME — same route, same rung, same `predicted_peak_bytes` — and it is worth being precise about
what it does *not* cover, because the earlier draft of this file overclaimed it as "byte-identical".

**The admitted requirement moved: `needed_gb` 45.57 → 50.13 GiB.** Same peak, but the legacy
resident floor is now graded behind the 10 % estimate margin, which pre-epic did not exist. So there
is a band — `[45.57, 50.13)` GiB of effective budget — where pre-epic admitted and this branch
refuses.

That band is real, and reachable through the probe's path: `c0_production_loadspec_probe` passes
`OffloadPolicy::Resident` directly, and at a 48 GB cap (46.00 GiB effective) the pre-epic commit
ADMITS at 45.57 while this branch refuses with `needs 50.13 GiB but only 46.00 GiB is safely
available`. Measured, not argued.

Whether the *production* chain reaches it is a separate question, and the answer is more interesting
than "no" — see the next section.

### A second behaviour change at the same cap, which is NOT a regression

Running the full production chain (cap → `apply_residency_policy` → contract → ladder) at
`qwen_image` q8 **1024²** on both commits:

| cap (GB) | `de756026` | this branch |
|---:|---|---|
| 56 and up | `Resident`, 45.57 GiB | `Resident`, 45.57 GiB |
| **48** | **ADMITS** `Resident` (load policy `Sequential`), 45.57 GiB | **REFUSES** — `needs at least 63.34 GiB at its smallest verified MLX host boundary` |
| 44 and below | refuses (`no safely verified MLX memory strategy (Missing)`) | refuses |

So at a 48 GB cap the production chain does diverge. The cause is not the estimate margin: under
`Sequential` the provider declares rung 4 `Implemented`, so the deferred rung-4 record now passes
both legs of the filter, reaches the Evidence budget pre-check, and is refused because its **captured
host boundary is 63.34 GiB** against a 46 GiB budget.

**This one is left alone deliberately.** It is the epic working as designed, not the §0 hole:

* §0 refused on `FingerprintMismatch` — a STRUCTURAL verdict carrying no information about whether
  the request fits. Refusing on it is the gate failing to do its job, and there was no fallback.
* This refuses because a measurement says the request needs 63.34 GiB and the machine has 46. That
  is the gate doing exactly its job, with better information than the 45.57 GiB estimate the
  pre-epic commit admitted on. On a lane where allocator overshoot terminates the process, the
  pre-epic admit was the unsafe answer.

An earlier attempt at this validation did try to degrade this case to the estimate ladder too. It
was reverted: it broke
`a_moved_provider_closure_admits_the_stale_ladder_behind_the_widened_margin`, whose whole point is
that the widened stale peak GATES — and it would have made `MLX_STALE_MEASURED_MARGIN`
non-binding on the MLX Evidence path, destroying the very property criterion 3 brackets. A stale
peak's *widening* is a signal; a stale peak's *refusal* is still a gate, and sc-18095/18096 chose
that deliberately.

It remains a user-visible narrowing — a 48 GB Mac that could render this cell before now cannot —
so it is called out here rather than buried, and it is worth a product decision on whether a
measured "does not fit" should be able to veto a cheaper unmeasured rung. Tracked with sc-18237.

### The fix

`crates/sceneworks-worker/src/mlx_fit_gate.rs`: on the Evidence route, filter out candidates the
loaded provider **cannot serve**, and degrade to the Legacy estimate ladder only if that leaves
nothing. A candidate is unusable when either

* it was measured under a materialization shape this load does not use, or
* it sits on a rung the loaded contract does not declare `Implemented` — which `select_strategy`
  skips *without even recording an exclusion*, so the request died with a bare `Missing`.

Two design points are load-bearing:

1. **It is a per-candidate filter, not a whole-route demotion.** A route legitimately carries
   bindings captured under different shapes — `qwen_image` q8 ships `bounded_attention` measured
   eager and `bounded_transformer_residency` measured deferred. Demoting the whole route when any
   sibling mismatches would discard the candidate that *does* match and silently downgrade a
   calibrated request to an estimate. An earlier iteration of this fix did exactly that, and
   criterion 2 caught it by flipping from `BoundedAttention` to `Resident`; the regression test's
   positive arm now pins it.
2. **Degrading, not refusing, is the epic's own thesis.** A calibration identity that does not match
   the loaded provider is a reason to stop *claiming* the measurement, not a reason to deny service.
   The estimate ladder the cell falls to is strictly more capable than the pre-epic resident-only
   freeze it replaces.

Guarded by `mlx_fit_gate::tests::a_load_shape_mismatch_degrades_to_estimates_instead_of_refusing`
(not `#[ignore]`d), with four arms — degrade, spare-the-sibling, the Resident exemption, and the
rung-support leg — plus
`mlx_fit_gate::tests::gen_core_accepts_a_resident_cell_whose_load_shape_disagrees`, which pins the
exemption's premise against the pinned gen-core rather than restating it.

**Every leg is mutation-checked**, and this mattered: the guard's first version could not tell the
per-candidate filter from a whole-route demotion, because the fixture bundle held a single record
and there was never a sibling to spare. Replacing `retain` with `clear` — the exact bug this fix
had already hit once — left it green. The fixture now carries a second record on
`bounded_transformer_residency` captured `deferred_materialization`, and the arms assert on the
surviving SET. Current matrix:

| mutation | result |
|---|---|
| `evidence.retain(usable)` → `evidence.clear()` (whole-route demotion) | **red** |
| shape test disabled (`shape_agrees = true`) | **red** |
| Resident exemption dropped (over-strict) | **red** |
| rung-support conjunct dropped | **red** |

The Resident exemption is a **mirror, not a tightening** — `optimized_eligibility` short-circuits
`Ok(())` for a non-optimized selection before it compares load shapes, so a resident cell measured
under the other shape is one the downstream gate ACCEPTS. Filtering it here would be stricter than
the gate this filter exists to anticipate, and would silently discard a usable measurement —
the same failure mode as the whole-route demotion. Reachable in the shipped corpus: `qwen_image`
bf16 carries a resident binding captured eager against a production deferred load.

### What is still open

The measured cell is still *unused* under the production shape — it degrades to an estimate rather
than serving its 43.125 GiB measurement. That half remains **sc-18237**, rescoped: the hard refusal
is fixed here, the wasted measurement is not.

---

# 1. An unmeasured cell engages a deep rung and completes a real render

**Subject:** `qwen_image` q8, text-to-image, no overlay, **768×768**, artifact
`SceneWorks/qwen-image-mlx@8080a4171f1c8b7fca6c30491eafbe6ffab754bf:q8` — the exact snapshot the
shipped opt-in names, present in this machine's HF cache, with **real resolved provenance** supplied
from the shipped binding.

Provenance decides *which* legacy arm the request takes, and the arm is the thing under test. With
`resolved_artifact: None` the plan is `MlxCalibrationConfig::Unproven` and `evidence_admission_route`
short-circuits to `NoProvenance` — an arm that returns `estimate_bases: Vec::new()` and
`lower_alternative: None`, so a fitted-curve estimate can never be synthesized and no refusal
alternative can ever be named. The route a real install actually takes is the geometry miss, and that
arm calls both `collect_estimate_bases` and `verified_lower_alternative`. The recorded run takes it:

```
path=Legacy fallback_reason=Some(OutOfEnvelope) width=768 height=768 count=1
```

The MLX corpus holds `qwen_image` q8 only at 1024², so the 768² request cell has no calibration
record and no binding, and is served entirely from synthesized estimates.

| | |
|---|---|
| Emulated cap | **32 GB** |
| Load-time residency decision | `OffloadPolicy::Sequential` (from `apply_residency_policy`, same cap) |
| Rung engaged | **`BoundedTransformerResidency`** (rung 4) |
| Parameters | `decode_tile_edge=256`, `decode_overlap=64`, `attention_chunk_size=67108864`, `transformer_window_size=1` |
| Basis | `floor` (weights + headroom) |
| Raw floor estimate | 26 967 416 751 B = **25.12 GiB** |
| Estimate margin applied | **0.10** (`MLX_ESTIMATE_MARGIN`) |
| Admitted ceiling (predicted-with-margin) | 29 664 158 427 B = **27.63 GiB** |
| Observed peak (`mlx_rs::memory::get_peak_memory`, active) | 15 975 492 864 B = **14.88 GiB** |
| Render | 768×768, 8 steps, seed 18101, 91.7 s, pixel std 46.25, non-degenerate |
| PNG | `~/SceneWorks/render-validation-sc18101/c1-qwen_image-q8-768x768-main.png` |

Two different over-prediction figures, both true and easy to conflate:

* the **admitted ceiling** is 27.63 GiB and the render used 14.88 GiB, so **46.1 % of the ceiling
  went unused** — that is the headroom the margin check had;
* the **raw floor estimate** before any margin is 25.12 GiB, which **over-predicts the observed peak
  by 68.8 %** — that is how conservative the weights+headroom floor is on this cell.

Corroboration: the corpus's **measured** rung-4 cell for the same provider/tier at the *larger* 1024²
geometry (`imc-4426a6e84c4d39d9bff3`) recorded 15 977 625 784 B observed. The unmeasured 768² run came
in at 15 975 492 864 B — within 0.014 %. Rung 4 windows the transformer, so the request peak is
carried by the area-flat conditioning phase; the floor estimate does not model that, which is exactly
why it over-predicts, and in the conservative direction.

### The cap ladder

| cap (GB) | load policy | rung selected | raw estimate |
|---:|---|---|---:|
| 128 → 56 | `Resident` | `Resident` | 45.57 GiB |
| 48 → 36 | `Sequential` | `StagedResidency` | 30.13 GiB |
| **32, 30** | `Sequential` | **`BoundedTransformerResidency`** | **25.12 GiB** |
| 28 | `Sequential` | refused — `needs 27.63 GiB but only 26.00 GiB is safely available` | |
| 26 | `Sequential` | refused — `… but only 24.00 GiB is safely available` | |
| 24 | `Sequential` | refused — `… but only 22.00 GiB is safely available` | |
| ≤ 22 | — | refused at the **load-time** gate, before the ladder is reached | |

### Scope gap: no *wholly* unmeasured MLX model reaches a deep rung

Criterion 1 is satisfied at the granularity of an uncalibrated **coordinate** — a geometry with no
record — not at the granularity of a wholly unmeasured **route**. The distinction is not pedantic,
and the consequence is a real limit on what this validation demonstrates:

* No MLX provider+tier is wholly unmeasured in the corpus (finding 1 below), so an "unmeasured MLX
  model" in the strict sense does not exist to test.
* The nearest thing that does exist — `krea_2_turbo` plain text-to-image, which has zero records and
  zero bindings at any tier — reaches only **`StagedResidency`** (rung 1) under a cap, then refuses:

  | cap (GB) | rung | raw estimate |
  |---:|---|---:|
  | 128 → 56 | `Resident` | 42.89 GiB |
  | 48, 44 | `StagedResidency` | 34.62 GiB |
  | 40 → 28 | refused — at cap 40, `needs 38.08 GiB but only 38.00 GiB is safely available` | |
  | ≤ 26 | refused at the load-time gate | |

  It cannot go deeper for a structural reason unrelated to this epic (finding 2): its rung 4 needs a
  streamable transformer, which the bf16 route does not offer.

So the honest statement is: **an uncalibrated coordinate of a partially measured route engages rung 4
and renders; a wholly unmeasured route engages rung 1.** The epic's machinery is validated; its reach
on a route with no windowed transformer is two rungs, not five.

---

# 2. A measured-current cell selects identically to pre-epic main

**Commits compared:** this branch vs **`de756026`** (main immediately before the sc-18094 merge
`cc799d1a`). Cell: `qwen_image` q8 1024² at a 96 GB cap.

### Which reading of "measured-current" was used, and why

The shipped corpus has **zero** closure-current lanes, so the phrase has two possible readings:

1. *measured and admitted* — take a cell as it ships. **Rejected.** Every shipped MLX cell is
   closure-STALE, and pre-epic main excluded stale candidates outright. Comparing that against main
   would compare the epic's intended behaviour change with itself and prove nothing.
2. *measured and closure-current* — the calibrated happy path the epic promises not to disturb.
   **This is the reading used.** Since no shipped lane is current, the cell was made current with a
   **scratch closure table**: `config/inference-provider-closures.json`'s
   `providers["mlx:qwen_image"].digest` was temporarily set to
   `54a7b45b03eb8301e6e85fd1d67558d007ebe5031eb491fe25e95b8f081f4374` — the digest the q8 ladder was
   captured under — in *both* checkouts, i.e. the world in which nobody has touched the provider
   since the measurement. Reverted after the runs; neither checkout carries it now.

**The consequence, plainly: `CandidateCurrency::Current` is a state no shipped lane occupies, so this
criterion has zero coverage of today's product.** It proves the calibrated path is undisturbed in the
world the corpus is *supposed* to be in, and nothing about the world it is actually in. The world it
is actually in is covered by criterion 3 (stale) and by §0 (mismatched).

### Result: byte-for-byte identical

Both the selection fingerprint and the **entire** captured tracing output are identical (`diff`
clean) between the two commits:

```
SELECTED strategy=BoundedAttention
  parameters=MemoryStrategyParameters { decode_tile_edge: Some(512), decode_overlap: Some(64),
             attention_chunk_size: Some(67108864), transformer_window_size: None,
             transformer_window_component: None }
  tier=MemoryNumericTier { precision: Bf16, quant: Some(Q8), component_precision_floors: [] }
  predicted_peak_bytes=46305116160 process_limit_bytes=Some(52684932164)
  stage_residency=false tile_vae_decode=true chunk_attention=true stream_transformer_blocks=false
```

Both emit `path=Evidence fallback_reason=None`, select record `imc-37f40254d20bc43fa925`, and emit
**no** widening line — a current candidate is graded at its raw peak, unwidened, on both sides. This
also pins the §0 fix: an over-broad version of it changed this selection to `Resident`, and was
caught here.

The baseline checkout carried only the harness module, its one-line `lib.rs` registration, and the
two-line margin-constant substitution the harness documents. No production source differed.

### Contrast: the same cell as it actually ships (stale)

| | `de756026` | this branch |
|---|---|---|
| admission path | `Legacy`, `fallback_reason=StaleIdentity` | `Evidence`, `fallback_reason=None` |
| measured candidate | none — excluded before the selector | admitted, carrying its captured digest |
| rung selected | `Resident` (frozen to the baseline) | `BoundedAttention` |
| needed | 45.57 GiB (resident estimate) | 45.28 GiB (measured peak, widened 5 %) |

### The fitted-curve arm

`CandidateBasis::EstimateFittedCurve` is the other half of sc-18096's estimate machinery — per-phase
extrapolation from a measured cell rather than the weights+headroom floor — and nothing in the first
round of this validation reached it. Reaching it needs three things at once, none of which the floor
path needs: proven provenance (so the request takes the `OutOfEnvelope` arm, the only caller of
`collect_estimate_bases`), a **closure-current** nearby-geometry binding (that collector deliberately
refuses stale seeds, so the 0.05 drift allowance can never be stacked under an extrapolation), and a
load shape matching the basis record's.

All three hold for `qwen_image` q8 at 768² under the production deferred spec with the scratch table
in place. Result (`c5_fitted_curve_estimate_is_synthesized_and_admitted`):

| | |
|---|---|
| Admission route | `Legacy`, `fallback_reason=OutOfEnvelope` |
| Basis record | the deferred `bounded_transformer_residency` cell at 1024² |
| Rung admitted | `BoundedTransformerResidency`, `basis="fitted_curve"` |
| Fitted raw peak | 16 777 216 000 B = **15.625 GiB** |

The request is *smaller* than the basis, so the area scale clamps to 1.0, the extrapolated binding
phase equals the measured one, and `ESTIMATE_ADMISSION_REQUIRES_MEASURED_BINDING_PHASE` is satisfied.
Worth setting against criterion 1's floor: the fitted curve predicts **15.63 GiB** where the floor
predicted **25.12 GiB** for the same cell, against an observed 14.88 GiB. When a measured basis
exists the estimate is dramatically tighter — which is also why the corpus's currency matters more
than the margin does.

---

# 3. A stale lane still admits, at the widened peak, and logs it

**Lane:** `mlx:qwen_image` — stale as shipped (live closure `9930aa538259…`, captured
`54a7b45b03eb…`). No test seam needed; this is the corpus's actual state.

| | |
|---|---|
| Cell | `qwen_image` q8 1024², rung `BoundedAttention`, record `imc-37f40254d20bc43fa925` |
| Raw measured peak | 46 305 116 160 B = **43.125 GiB** |
| Stale margin applied | **0.05** (`MLX_STALE_MEASURED_MARGIN`) |
| Widened admitted peak | 48 620 371 968 B = **45.281 GiB** |

`widened == ceil(raw × (1 + STALE_MARGIN))` is asserted, not eyeballed.

### The bracket: the widened number actually gates

Asserting the multiplication off one log line proves the gate can multiply; it does not show the
widened peak gated anything. So the budget is bisected down to the admit/refuse boundary and the
boundary itself is checked:

| | |
|---|---|
| Admit/refuse boundary | **92.2146 GB** (admits above, refuses below; converged to < 1e-3 GB) |
| Requirement the refusal quotes | **92.21 GiB** — `needs at least 92.21 GiB at its smallest verified MLX host boundary` |
| What a zero-margin gate would have required | **90.05 GiB** |
| Window a zeroed margin would wrongly admit | **[90.05, 92.21) GiB** |

The gating quantity on this path is not the peak alone: the Evidence route charges each candidate's
captured foreign reserve on top, and both the selection event's `needed_gb` and the refusal quote
that sum. The margin is visible in it — the enforced requirement exceeds the unwidened one by exactly
`widened − raw` = 2.16 GiB, three orders of magnitude above the bisection's resolution.

**Production-conditions status:** this bracket is measured on the eager `LoadSpec` the binding was
captured under (`SC18101_MEASURED_EAGER=1`), not on the deferred spec production uses for
`qwen_image`. Under the production spec this cell degrades to the estimate ladder (§0), so the
stale-measured path is exercised on a real shipped binding and real weights, but not on a
configuration `qwen_image` reaches in production today. Closing that is sc-18237.

The pre-epic contrast is the proof the widening is doing work rather than decorating a decision that
would have happened anyway: on `de756026` this same cell produces no measured candidate at all.

---

# 4. An oversized request is refused, not OOM-killed

**Request:** `qwen_image` q8 **2048×2048** at a 24 GB cap, loaded `Sequential` so the provider offers
its deepest possible ladder. The full five-rung estimate ladder is synthesized and every rung graded,
so the refusal is "nothing fits with margins", not "a rung was missing":

| rung | raw floor | widened (×1.10) |
|---|---:|---:|
| `Resident` | 73 641 068 818 | 81 005 175 700 |
| `StagedResidency` | 57 056 202 530 | 62 761 822 784 |
| `BoundedDecode` | 73 640 535 842 | 81 004 589 427 |
| `BoundedAttention` | 73 640 535 842 | 81 004 589 427 |
| `BoundedTransformerResidency` | 51 674 216 124 | **56 841 637 737 = 52.94 GiB** |

```
qwen_image request 2048x2048 count 1 needs 52.94 GiB but only 22.00 GiB is safely available
```

The assertion requires *that* message specifically and explicitly rejects the structural refusal —
accepting `no structurally admissible` as well would have accepted the §0 regression as proof that
budget refusal works.

**Proof it refused rather than averted an OOM** is structural, not statistical: the refusal is
produced by `mlx_fit_gate::evaluate_request`, which runs **before** `generator.generate`, so no
allocation for the request is ever attempted; and MLX's default error handler calls `exit(-1)`, so an
OOM would end the process — the test continues in the same process and re-submits the identical
request at a 512 GB cap, which is admitted. Both are asserted.

---

# 5. Suites

See the PR for CI. Locally: `npm run check`, `npm run rust:check`, the parity pytest gate
(`python -m pytest -q tests/ -m "not e2e and not parity" --strict-markers`), and `cargo fmt --all`.

`docs/generated/memory-matrix.{json,md}` are regenerated in this branch: `mlx_fit_gate.rs` is one of
the sixteen `SOURCE_PATHS` the matrix fingerprints, so the §0 fix rotates
`generatedFrom.sceneWorksRevision`.

---

# Findings worth carrying forward

1. **No MLX provider+tier is wholly unmeasured in the corpus.** All three MLX lanes (`qwen_image`,
   `z_image_turbo`, `krea_2_turbo_control`) have records; the single unmeasured lane is
   `candle:z_image`, which has no adapter arm. On MLX, "an unmeasured cell" necessarily means an
   uncalibrated coordinate. See the criterion-1 scope gap for what that costs the validation.

2. **On a floor-only ladder, rung 4 is the only deep rung that can ever be engaged.** By deliberate
   design (`mlx_fit_gate.rs`, `estimate_floor_weights_bytes`) bounded decode and bounded attention
   bound *transients*, not weights, and take no floor reduction on an unmeasured cell — so their
   floors equal `Resident`'s and no budget can admit them while excluding `Resident`. A provider that
   does not implement `BoundedTransformerResidency` therefore has a **flat** ladder: under a cap it
   reaches `StagedResidency` at best, then refuses (`krea_2_turbo` bf16, measured above: rung 1 at
   caps 48 and 44, refusing from cap 40 down). Correct behaviour, not a defect — but "estimates admit
   the full ladder" delivers a two-rung improvement, not a five-rung one, on providers without a
   windowed transformer.

3. **Which rungs a provider declares is a function of the `LoadSpec`, not of the request.** Rung 4
   needs `LoadShape::DeferredMaterialization` (z-image, lens) or `OffloadPolicy::Sequential` (krea,
   qwen), both reached only through the LOAD-time gate. A request-scoped test that loads `Resident`
   sees a flat ladder for reasons unrelated to this epic; every scenario here runs the cap through
   `apply_residency_policy` first.

4. **A structural exclusion inside `select_strategy` has no fallback behind it.** Estimate synthesis
   runs only on the Legacy route, so anything that empties the Evidence route's candidate list after
   admission has already chosen it becomes a refusal. §0 is one instance (load shape); the
   rung-support case is a second, and it is *silent* — `select_strategy` skips an unimplemented rung
   without recording an exclusion, so the request dies with a bare `Missing` and no log line naming a
   cause. Both are now filtered before the selector, but the shape of the hazard remains: any future
   structural predicate added downstream of admission needs a fallback, not just a verdict.

5. **`image_jobs/base.rs` and `mlx_fit_gate.rs` are both fingerprinted matrix sources.** Even a
   visibility keyword in the former rotates `generatedFrom.sceneWorksRevision` and reds
   `npm run check:memory-matrix`; the harness therefore mirrors that file's route list rather than
   calling into it, with a text-coupling test. Budget for a matrix regeneration whenever a
   fingerprinted source changes.

6. **Moving only the manifest binding's `inferenceClosureDigest` does NOT make a lane current.**
   `EvidenceBundle::evidence_for` compares the binding's digest against the record's own stamp too,
   so a one-sided edit makes the record unfindable and the cell routes to `StaleIdentity` with no
   measured candidate at all — observed directly here. A real re-measurement moves both halves
   together, which is why `docs/calibration-runbook.md` §7d insists on both. A scratch closure table
   is the correct single-edit lever for a test.

---

# Re-running

```text
OUT=~/SceneWorks/render-validation-sc18101
QWEN="SC18101_C1_REPO=models--SceneWorks--qwen-image-mlx SC18101_C1_ENGINE=qwen_image SC18101_C1_TIER=q8"

# §0 regression probe — run in BOTH checkouts and diff $OUT/c0-outcome-{main,baseline}.log
SC18101_TAG=main SC18101_C0_CAP_GB=96 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture c0_production_loadspec_probe

# criterion 1 — omit SC18101_C1_CAP_GB to print the cap sweep and fail with it
SC18101_TAG=main $QWEN SC18101_C1_W=768 SC18101_C1_H=768 SC18101_C1_CAP_GB=32 SC18101_C1_STEPS=8 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c1_unmeasured_cell_engages_a_deep_rung_and_renders

# criteria 3 (stale, as shipped) and 4 (refusal)
SC18101_TAG=main SC18101_MEASURED_EAGER=1 SC18101_C3_ADMIT_CAP_GB=96 $QWEN SC18101_C4_CAP_GB=24 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c3_stale_lane_admits_at_the_widened_peak c4_oversized

# criterion 2 and the fitted-curve arm — put the scratch closure table in BOTH checkouts first:
#   config/inference-provider-closures.json providers["mlx:qwen_image"].digest = 54a7b45b03eb…
# then run c2 in each and diff $OUT/c2-fingerprint-{main,baseline}.log
SC18101_TAG=main SC18101_MEASURED_EAGER=1 SC18101_C2_CAP_GB=96 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c2_measured_current_cell_selection
SC18101_TAG=main SC18101_C5_CAP_GB=32 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 c5_fitted_curve
# …then revert the scratch table.
```
