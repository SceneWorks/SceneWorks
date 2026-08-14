# Epic 18093 end-to-end validation record (sc-18101)

On-device validation of the whole right-sized calibration apparatus, run after every epic 18093
story had merged to `main`. This file is the durable record; the harness that produced it is
`crates/sceneworks-worker/src/ladder_e2e_sc18101.rs` (five `#[ignore]`d scenarios plus a portable
regression probe).

**Verdict: all five success criteria pass, no margin breach — and the validation found and fixed a
REGRESSION this epic shipped.** That regression is the most important thing in this document, so it
comes first.

> **Current-source note (SC-18237, 2026-08-09):** this document preserves the historical SC-18101
> capture below, but its old Qwen closure-stale state is no longer the shipped state. Qwen q8 now has
> two production-deferred records whose binding and record stamps match the live `9930aa538259…`
> provider closure. Criterion 3 now exercises that current lane's exact static boundary; stale-margin
> behavior is covered separately by explicit moved-digest tests. See
> `docs/epic-18093-sc-18237-qwen-production-evidence.md` for the current evidence.

## Historical SC-18101 environment

| | |
|---|---|
| Host | Apple M5 Max, `Mac17,6`, 128 GiB unified memory, macOS 26.5.2 (the `nax-macos` self-hosted runner) |
| SceneWorks revision under test | `main` after the sc-18100 and sc-18104 merges, plus this branch |
| Pre-epic baseline revision | `de756026` (`main` immediately before the sc-18094 merge `cc799d1a`) |
| Inference pin | `40fa7583a01974617e2a7275052d6d446688c956` |
| Evidence corpus | `docs/generated/memory-calibration-evidence.json`, 65 records (50 MLX / 15 candle) |
| Lane currency at capture time | 8 stale, 0 current, 1 unmeasured (`npm run report:stale-lanes`) |
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
`binding.query.inference_closure_digest == expected_closure_digest`; every MLX lane shipped at the
time of SC-18101 was closure-stale, so that conjunct pre-demoted the cell to
`AdmissionPath::Legacy` long before eligibility ran, and it was served. sc-18096 retired the
conjunct — correctly, that was its whole point — and in doing so made a previously unreachable
refusal reachable on a shipping route.

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

So at a 48 GB cap the production chain does diverge. The cause is not the estimate margin.

### What the 63.34 GiB actually is — and it is not "this work needs 63 GiB"

`MlxAdmissionEnvelope::required_host_bytes` is `peak_bytes + foreign_reserve_bytes`
(`crates/sceneworks-core/src/memory_calibration.rs:908-910`), and `foreign_reserve_bytes` is derived
entirely from the **capture host's** counters (`:920-923`, computed at `:939`):
`memoryBytes − min(mlxMemoryLimitBytes, wiredLimitBytes)`. For this corpus that is

```
137,438,953,472 − 87,044,670,532 = 50,394,282,940 B = 46.93 GiB
```

and it is **identical on all 50 MLX records** — the corpus has exactly one distinct hardware tuple,
because everything was captured on this same 128 GiB M5 Max. It is never rescaled to the live host.

So the refusal decomposes as:

| term | value | what it is |
|---|---:|---|
| widened request peak (rung 4, ×1.05 stale) | **16.41 GiB** | what this render actually costs |
| capture-host foreign reserve | **46.93 GiB** | what the 128 GiB *capture machine* had outside the MLX process |
| `required_host_bytes` | **63.34 GiB** | the sum the gate compares against the live host |

The rule being enforced is therefore: **a measurement is only usable on a host at least as large as
the one it was captured on.** The refusal means "this measurement was taken on a bigger machine",
NOT "your machine is too small for this work" — the work itself is 16.41 GiB. Getting this wrong
would send the follow-up product call down entirely the wrong path, so it is stated here explicitly.

The same decomposition explains criterion 3's bisected boundary exactly: 45.28125 (widened
`BoundedAttention` peak) + 46.933 = **92.2146**, which is the boundary to four decimal places.

### Why this is still left alone

**This is not the §0 hole, and the distinction is not the size of the number:**

* §0 refused on `FingerprintMismatch` — a STRUCTURAL verdict carrying no information about the
  request at all, with no fallback behind it. That is the gate failing to do its job.
* This refuses under a deliberate conservative rule: *do not extrapolate a measurement downward
  across host sizes*. That rule is defensible on its own terms even though its input is a capture
  artefact rather than a property of the request.

**The gate PREDATES this epic.** `required_host_bytes`, `fits_host_bytes` and the "smallest verified
MLX host boundary" refusal are all present at `de756026`, with their own tests
(`de756026:crates/sceneworks-core/src/memory_calibration.rs:2313-2316`). What sc-18096 changed is
that retiring the closure conjunct made the gate **reachable on stale lanes**, which is where every
shipped MLX lane sat during the SC-18101 capture. That is the honest statement of what this epic did:
it did not introduce the rule, it exposed it. SC-18237 subsequently made Qwen q8 closure-current.

**The safety argument rests on the actual margin, not on 63.34.** Pre-epic admitted this request at a
45.57 GiB estimate against **46.00 GiB** available — 0.43 GiB of headroom, about **1 %** — on a path
where an allocator overshoot calls `exit(-1)` and takes the worker with it. Shipping a refusal in
place of a 1 %-margin admit is defensible; shipping it because "the measurement says 63 GiB" would
not have been, because the measurement says no such thing.

### Verified blast radius: one cap step

From the committed sweeps on both commits, not extrapolated:

| cap (GB) | `de756026` | this branch | same? |
|---:|---|---|---|
| 128 → 56 | `Resident`, 45.57 GiB | `Resident`, 45.57 GiB | identical |
| **48** | **ADMITS** `Resident`, 45.57 GiB | **REFUSES** (host boundary) | **diverges** |
| 44 and below | refuses | refuses | both refuse |

It is **one cap step**, not "everything under 64 GiB". The reason ≥ 56 GB is untouched is structural,
not luck: the load-time gate keeps `OffloadPolicy::Resident` there, so rung 4 is not `Implemented`,
so the rung-support leg of the §0 filter drops the only surviving measured candidate and the route
degrades to the estimate ladder — which is exactly the path the pre-epic commit took as well.

### The obvious fix is provably wrong

An earlier attempt at this validation degraded "measured candidates survive but none fit" to the
estimate ladder. It was written and **reverted**, and the arithmetic shows why it could not be right:
the Legacy floor it would fall to is 50.13 GiB with **no foreign-reserve charge at all**, so it
admits above 50.13 — and the 92.21 GiB boundary criterion 3 brackets would then never refuse
anything. `MLX_STALE_MEASURED_MARGIN` would gate nothing on the Evidence path, destroying the very
property criterion 3 exists to prove. It also broke
`a_moved_provider_closure_admits_the_stale_ladder_behind_the_widened_margin`, correctly.

SC-18237 subsequently made that design change in the pre-existing gate: it rescales the captured
foreign reserve to the live host while preserving the measured peak, stale margin, and absolute MLX
process ceiling. The user-visible 48 GB narrowing described by this historical SC-18101 run is no
longer the production behavior.

### Product decision: keep proportional foreign-reserve scaling (SC-18352)

On 2026-08-09 Michael explicitly chose to keep SC-18237's proportional scaling. The captured
foreign reserve is the capture host size minus the minimum of its host size, MLX memory limit, and
wired limit. It is a host-policy fraction, not a fixed cost of the request; the wired limit happens
to be the limiting value in the current corpus. Carrying a 128 GiB capture host's absolute 46.93 GiB
reserve onto every smaller Mac therefore applies the capture host's policy unchanged. Scaling the
same fraction to the live host matches the chosen interpretation of that reserve while retaining
the measured request peak, stale-evidence margin, and absolute MLX process ceiling.

This is an admission-policy decision, not new measurement evidence. The regenerated matrix changed
18 of 20 published `requiredHostBytes` values, with reductions ranging from roughly 9 to 85 percent
depending on the cell, without re-running those captures. The more permissive small-host boundary
therefore remains OOM-sensitive: MLX allocation failure terminates the worker instead of returning
a recoverable error. UTM runs across several configured memory sizes will be used to compare
observed admission and OOM boundaries with
`required_host_bytes_for`. A UTM result may update the evidence corpus only if the guest exposes
authentic memory counters and runs the exact production MLX/Metal path; otherwise it is supporting
emulation evidence rather than a physical calibration record.

**Release-note callout:** MLX minimum-memory requirements derived from captured evidence now scale
the capture host's foreign-reserve fraction to the live Mac instead of adding the capture host's
fixed reserve. This removes refusals caused solely by carrying that fixed capture-host reserve onto
smaller Macs; it does not weaken the measured peak, stale-evidence margin, or absolute MLX process
ceiling.

### The fix

`crates/sceneworks-worker/src/mlx_fit_gate.rs`: on the Evidence route, filter out candidates the
loaded provider **cannot serve**, and degrade to the Legacy estimate ladder only if that leaves
nothing. A candidate is unusable when either

* it was measured under a materialization shape this load does not use, or
* it sits on a rung the loaded contract does not declare `Implemented` — which `select_strategy`
  skips *without even recording an exclusion*, so the request died with a bare `Missing`.

Two design points are load-bearing:

1. **It is a per-candidate filter, not a whole-route demotion.** At the time of this validation the
   q8 route mixed an eager `bounded_attention` record with a deferred
   `bounded_transformer_residency` record. Demoting the whole route when any sibling mismatches
   would discard the candidate that *does* match and silently downgrade a calibrated request to an
   estimate. An earlier iteration of this fix did exactly that, and criterion 2 caught it by
   flipping from `BoundedAttention` to `Resident`; the regression test's positive arm now pins it.
   SC-18237 subsequently replaced that mixed historical pair with two production-deferred records.
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

### SC-18237 resolution

SC-18237 closed the remaining gap on 2026-08-09. The q8 bounded-attention case was recaptured through
the production-deferred adapter and promoted alongside a fresh deferred bounded-transformer-
residency record. The shipped q8 pair now has one executable load shape, and real-weight probes prove
Resident reaches bounded attention at 96 GiB while Sequential reaches bounded transformer residency
at 48 GiB. See `docs/epic-18093-sc-18237-qwen-production-evidence.md` for exact records, revisions,
memory accounting, and the audited shipped-binding matrix.

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

### Historical method and current status

At the time of the original SC-18101 comparison the corpus had **zero** closure-current lanes. The
test therefore made the q8 cell current in both checkouts with a **scratch closure table**:
`config/inference-provider-closures.json`'s `providers["mlx:qwen_image"].digest` was temporarily set to
`54a7b45b03eb8301e6e85fd1d67558d007ebe5031eb491fe25e95b8f081f4374`, the digest the old q8 ladder was
captured under. That historical edit was reverted after the runs.

That limitation no longer applies. SC-18237 promoted two production-deferred Qwen q8 records stamped
with the live `9930aa538259…` closure and updated the shipped binding to the same digest. The current
exact-head test requires binding, record, and live closure to agree without a seam. Thus the original
comparison below remains a historical pre-epic parity record, while the rerunnable current lane now
has direct product coverage.

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

In the historical comparison both emit `path=Evidence fallback_reason=None`, select record
`imc-37f40254d20bc43fa925`, and emit **no** widening line — a current candidate is graded at its raw
peak, unwidened, on both sides. This also pinned the §0 fix: an over-broad version of it changed this
selection to `Resident`, and was caught here.

The baseline checkout carried only the harness module, its one-line `lib.rs` registration, and the
two-line margin-constant substitution the harness documents. No production source differed.

### Historical contrast: the same cell as it shipped during SC-18101 (stale)

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

All three held for `qwen_image` q8 at 768² under the production deferred spec with the historical
scratch table in place. They now hold directly against the shipped SC-18237 current binding. Result
(`c5_fitted_curve_estimate_is_synthesized_and_admitted`):

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

# 3. Historical stale bracket and current-lane exact boundary

### Historical SC-18101 stale bracket

During SC-18101, `mlx:qwen_image` was stale as shipped (live closure `9930aa538259…`, captured
`54a7b45b03eb…`). The following table is the preserved historical capture, not current product state.

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

In this historical SC-18101 capture, the Evidence route charged each candidate's captured absolute
foreign reserve on top of the peak, and both the selection event's `needed_gb` and the refusal quoted
that sum. The margin was visible in it: the enforced requirement exceeded the unwidened one by
exactly `widened − raw` = 2.16 GiB, three orders of magnitude above the bisection's resolution.
SC-18237 later replaced that absolute smaller-host charge with proportional normalization and a
static solved boundary; the stale peak remains widened, but the boundary delta is no longer simply
the peak delta.

**Historical production-conditions status:** this SC-18101 bracket was measured on the eager
`LoadSpec` the old binding was captured under (`SC18101_MEASURED_EAGER=1`), not on the deferred spec
production uses for `qwen_image`. SC-18237 subsequently replaced that unreachable binding with the
production-deferred q8 records and verified the live 96 GiB Resident / 48 GiB Sequential routes.

Note also that the boundary this brackets — 92.2146 GB — is itself `45.28125 + 46.933`, i.e. the
widened peak plus the CAPTURE host's foreign reserve, not a property of the live machine. The
bracket still proves the margin gates (the 2.16 GiB it moves the boundary by is the margin, and only
the margin), but the absolute number is a capture artefact. See §0's decomposition.

The pre-epic contrast is the proof the widening is doing work rather than decorating a decision that
would have happened anyway: on `de756026` this same cell produces no measured candidate at all.

### Current SC-18237 lane and stale-margin coverage

The shipped Qwen q8 binding is now closure-current: its two production-deferred records and the live
provider closure all carry `9930aa538259…`. The real-weight Criterion 3 test is therefore
`c3_current_lane_enforces_exact_static_boundary`; it refuses to call the lane stale, asserts there is
no stale-widening event, selects current record `imc-56c1f11bd03822d9c241`, and bisects the real gate
against that record's **73,113,341,306 B (68.0921 GiB)** exact static host boundary.

Stale-margin coverage remains explicit and mutation-sensitive without falsifying product currency:

* `a_moved_provider_closure_admits_the_stale_ladder_behind_the_widened_margin` supplies a deliberately
  moved live digest and brackets the admit/refuse window introduced by the 5 % stale margin.
* `capture_host_reserve_scales_to_48_gib_without_erasing_the_stale_margin` independently proves host
  normalization preserves that stale-margin delta.

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

5. **The original SC-18101 absolute-reserve boundary was a capture-host artefact.** At that time
   `required_host_bytes` charged the 128 GiB capture host's full `foreign_reserve_bytes` without
   rescaling it to the live host, producing the historical 63.34 and 92.21 GiB boundaries above.
   SC-18237 replaced that behavior with proportional host normalization while preserving the
   measured peak, stale margin when applicable, and absolute MLX process ceiling. Current Qwen q8
   boundaries are therefore derived from the live-host scale; the Resident record's exact boundary
   is 68.0921 GiB, and the Sequential record is admitted at 48 GiB.

6. **`image_jobs/base.rs` and `mlx_fit_gate.rs` are both fingerprinted matrix sources.** Even a
   visibility keyword in the former rotates `generatedFrom.sceneWorksRevision` and reds
   `npm run check:memory-matrix`; the harness therefore mirrors that file's route list rather than
   calling into it, with a text-coupling test. Budget for a matrix regeneration whenever a
   fingerprinted source changes.

7. **Moving only the manifest binding's `inferenceClosureDigest` does NOT make a lane current.**
   `EvidenceBundle::evidence_for` compares the binding's digest against the record's own stamp too,
   so a one-sided edit makes the record unfindable and the cell routes to `StaleIdentity` with no
   measured candidate at all — observed directly here. A real re-measurement moves both halves
   together, which is why `docs/calibration-runbook.md` §7d insists on both. A scratch closure table
   was the correct historical single-edit lever; current Qwen coverage uses the shipped matching
   binding and records directly.

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

# criteria 3 (current exact boundary) and 4 (refusal)
SC18101_TAG=main SC18101_C3_ADMIT_CAP_GB=96 $QWEN SC18101_C4_CAP_GB=24 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c3_current_lane_enforces_exact_static_boundary c4_oversized

# criterion 2 and the fitted-curve arm now use the shipped closure-current Qwen records directly
SC18101_TAG=main SC18101_C2_CAP_GB=96 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 \
    c2_measured_current_cell_selection
SC18101_TAG=main SC18101_C5_CAP_GB=32 \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture --test-threads=1 c5_fitted_curve
```
