# Round 2: architecture review of Shared Generation Architecture draft 0.2

Date: 2026-11-03. Reviewer: Codex. Scope: README, REQUIREMENTS and IMPLEMENTATION_PLAN, the round-0 dispositions, and targeted supporting evidence and source checks. Reviewed design: SceneWorks `3bc17e3a0587b0c2aff069f6bab384eca9eef047`. Source checks below use the design's stated baselines: inference `a43724b82fbae35c9227cef4bcdabba6dd1da12c` and SceneWorks `a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f`.

CodeGraph discovery was used first. Its inference search index reported `35159482…`, and its service summary was incomplete; those results were used for navigation, not as evidence of baseline behavior. Exact behavior was checked with `git show` at the revisions above. This is a design review, not an implementation or a complete re-audit of the eight research reports. No builds, model execution, GPU use, runner dispatches or tracker-item creation were necessary.

## Assessment

**The architecture is heading in the right direction. Revise the contracts below before approving implementation. Confidence: high.**

The strongest choices are shared ownership of orchestration, tensor-free planning policy, backend-specific execution hooks, explicit numerical variants, request-scoped mutable state, structural applicability, and adoption through real routes. These directly address the multiplication of cross-cutting work across families and backends. Keep those choices.

The distinction between a shared driver and an essential `FamilyDriver` is also appropriate. A layer-level backend seam with pilots is a reasonable investigation; confidence in a successful backend-generic rollout is **medium**, rather than an established architectural result. Pipeline sharing need not wait for that result.

The main unresolved risk is that the plan names extension points without completely defining the state and invariants crossing them. Multi-resolution denoising is a useful first consumer precisely because it exposes those gaps. The corrections below preserve the requested capability surface; they do not propose dropping modalities, modes or later shared components.

Findings are **Open**. Proposed fixes and closure checks are review recommendations, not approved requirement changes. BLOCKER means a core contract needs a design decision before implementation approval; MAJOR means a material correctness or sequencing gap must be resolved before its owning feature starts.

## R2-01 — BLOCKER: Specify the noisy-state versus clean-estimate transition

**Confidence: high.** References: REQUIREMENTS PIPE-4, PIPE-6, FEAT-1, DEC-10; PLAN §3.1 `Transition`, §3.3 stage execution, §6.1, especially lines 182–185 and 585.

DEC-10 recommends re-noising **from x0**, but §6.1 resamples the current latent and then applies `Renoise(σ_K)`. At an interior boundary, the current latent is generally still noisy. The shared driver and `FamilyDriver::run_slice` return a latent; the contract does not say that a clean estimate is retained, which evaluation produced it, or whether another forward is needed. The last estimate from Heun or DPM++ SDE may belong to an intermediate evaluation rather than the boundary state. Resampling a noisy latent and applying a clean-sample re-noise law can apply noise twice.

The noise law also differs by sampling convention. The existing `ModelSampling::noise_scaling_coeffs` explicitly distinguishes `x0 + σ·noise` from flow's `(1−σ)·x0 + σ·noise`, and documents a prior out-of-distribution flow bug from using the former. See [model_sampling.rs:59–68](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/model_sampling.rs#L59) and [flow override:227–231](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/model_sampling.rs#L227). The existing [sampler returns only the integrated latent](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/unified.rs#L29).

**Proposed resolution:** define a typed boundary state distinguishing `x_sigma`, clean estimates and their evaluation provenance. Specify how the chosen estimate is obtained, the destination sampling convention, interpolation/normalization, seed stream, casts and any additional forward. Define a separate noisy-to-noisy transition if that behavior is wanted. Reuse the existing noise-scaling contract rather than implementing a second family-specific rule.

**Closure:** weights-free equation tests for flow and VE/EDM, plus boundary traces for Euler, a multi-evaluation solver and a bespoke driver. Tests must detect re-noising an already noisy input as if it were clean. Price and count any extra evaluation.

## R2-02 — BLOCKER: Define where phase boundaries live before composing Features

**Confidence: high.** References: REQUIREMENTS PIPE-3/4/5, HOOK-3; PLAN §3.1 lines 173–176, §3.4 lines 306–308, §6.1.

“Merge boundaries” is defined for one global schedule, but the architecture also supports fresh passes, explicit tables and independent stream schedules. It provides no common coordinate or pass identity. For example, with a 20-step first pass and a 10-step Hires.fix pass, does low-res `K=5` apply once, within each pass, or only to one selected pass? Where does a guidance interval attach? A Wan expert threshold is a per-evaluation sigma rule, while H3 audio and video have different sigma values at the same outer step. Sorting integer boundaries does not resolve these distinctions.

Axis ownership also leaves out schedule, transition and history contributions. Two Features can both install a transition without claiming the same listed axis. Conversely, two geometry operations may be compatible in different passes but appear to conflict if their domain is unspecified. Declared hook order makes execution repeatable; it does not define the intended trajectory.

**Proposed resolution:** give each boundary and override an explicit trajectory/pass domain and coordinate, with a rule for mapping joint-stream outer steps to their schedules. Specify ownership and composition of schedules, transitions and history, including native route stages. Define unsupported combinations as typed conflicts. Validate the **whole rewritten plan** for geometry, schedules, applicability, ordering cycles and transition/history compatibility; §3.1 currently validates route structure before rewrites and only history afterward.

**Closure:** worked plan snapshots for low-res + Hires.fix, low-res + Krea phases, coincident low-res/Wan expert boundaries, and an AV route. Include a three-Feature dependency cycle and parameter-dependent invalid geometry, beyond the proposed pairwise synthetic-route check.

## R2-03 — MAJOR: Carry execution state and RNG position across slices explicitly

**Confidence: high.** References: REQUIREMENTS PIPE-3/6/7/11/14; PLAN §3.3 lines 247–249, 271 and 281. This is a remaining gap after round-0 ENG-2, not a request to undo its `Carry`/`Reset` correction.

The revised plan says when history may carry, but does not define the lifetime of the state that carries it. Today's `Sampler::sample` owns history as local variables. Calling it on successive slices loses that state. Wan native intentionally passes **one mutable scheduler** through both expert ranges ([moe.rs:33–46](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/mlx-gen-wan/src/pipeline/moe.rs#L33)). DPM++ 2M also needs the previous timestep interval, not just the last x0 ([solvers.rs:219](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L219), [264](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L264)).

Stochastic continuity is a separate concern. DPM++ SDE draws noise with `(seed, 2*i)` and `(seed, 2*i+1)` ([solvers.rs:380](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L380), [403](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L403)). Restarting a slice at local index zero repeats draws even if the seed scheme stays unchanged. Resetting solver history must not accidentally restart the RNG stream. Also, the stated `Carry` validation permits a same-geometry `Continue` while changing `SolverId`; arbitrary solver histories are not interchangeable.

**Proposed resolution:** describe a request-owned execution state shared across slices: solver state, previous intervals, absolute draw/evaluation indices, and any bespoke driver state. Keep history reset, RNG continuation and driver-cache retention separate. Carry history only across compatible solvers and sampling conventions, or an explicitly supported conversion.

**Closure:** an inert split reproduces an unsplit trace and draw sequence; geometry changes reset history while preserving the defined stochastic stream; illegal solver/history combinations refuse. Include native Wan and a stochastic multi-evaluation solver.

## R2-04 — MAJOR: Prepared conditioning must depend on adapter scales and branches

**Confidence: high.** References: REQUIREMENTS PIPE-5 acceptance, PIPE-14; PLAN §3.3 lines 239–245 and 268, §3.8.

PIPE-5 keys prepared state by `(geometry, model slot, conditioning)`. The preparation method receives no `ForwardArgs`, even though the forward does. Krea already proves that equal geometry, base slot and encoded prompt do **not** imply equal prepared conditioning: an adapter may target the text-fusion aggregator run during `prepare`. Its multiphase implementation deliberately builds preparation from each phase's adapted clone ([pipeline.rs:1112–1148](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/mlx-gen-krea/src/pipeline.rs#L1112)). Replacing those clones with runtime scales without changing preparation semantics reuses stale fused contexts.

Branch structure is another dependency: Wan's cross-KV preparation builds batch 1 or 2 depending on whether unconditional context is active ([denoise.rs:33–49](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/mlx-gen-wan/src/pipeline/denoise.rs#L33)). A guidance change can therefore change the prepared batch structure even when the encoded contexts already exist.

**Proposed resolution:** declare preparation dependencies and include the relevant adapter stack/scales, branch configuration, effective model identity and conditioning identity in cache keys. Pass request-scoped preparation arguments through the same immutable-module contract. Keep geometry-only preparations reusable where their dependencies allow it.

**Closure:** base-only → adapter → base-only Krea phases rebuild or reuse the correct preparation and reproduce legacy traces. Add CFG-off → CFG-on and two interleaved requests with different preparation scales.

## R2-05 — BLOCKER: Price concurrent live allocations, not the largest stream

**Confidence: high.** References: REQUIREMENTS MEM-1/3, ADM-3; PLAN §3.9 line 416.

The stated formula is “maximum over phases × streams, plus cross-phase residency, plus Feature memory effects.” Streams are not generally sequential alternatives. H3 runs one joint forward and keeps video and audio latents alive together; both velocity results and shared packed-model workspaces can coexist. Its actual loop explicitly evaluates `[video, audio]` together ([denoise/mod.rs:1–13](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/mlx-gen-minimax-h3/src/denoise/mod.rs#L1)). A maximum of stream costs is smaller than their simultaneous live total.

Taking a maximum across sequential phases is appropriate only after constructing each phase's concurrent live set. The peak may also occur inside a transition while old and new latents, an upsampler or codec workspace, and retained contexts coexist. Joint/packed representations must not then be counted again as independent physical buffers.

**Proposed resolution:** define peak bytes as the maximum over execution points of the total distinct live allocations, with shared weights/buffers deduplicated and joint workspaces explicit. If “streams” in §3.9 was intended to mean an already aggregated joint estimate, say so and specify that aggregation.

**Closure:** pure pricing fixtures where two individually fitting streams jointly exceed the budget, an upsample transition exceeds either steady phase, and aliased/packed buffers are counted once. No model execution is needed for these checks.

## R2-06 — MAJOR: Integrate geometry-changing plans with admission when they first ship

**Confidence: high.** References: REQUIREMENTS PIPE-2, MEM-5, ADM-3; PLAN §6.1 line 588, ENG-1/2 and APP-4.

ENG-1 says admission is unchanged because final geometry is largest and transition allocations are negligible beside weights. That is an unbounded assumption: the permitted geometry, count, tier and memory-rung combinations can make transition buffers material near the admission limit. A successful Tier-1 peak on a selected configuration does not establish a conservative bound over those configurations. Hires.fix in ENG-2 adds a pixel round trip and codec residency, while all-route plan-aware admission is scheduled only after ENG-6.

There is also an identity issue independent of peak size. Today's scope validates the geometry/mode it was admitted for. SceneWorks already has a special first-pass context adjustment because handing the smaller Hires pass the final-pass context caused every scope-adopting provider to refuse ([base.rs:8515–8542](https://github.com/SceneWorks/SceneWorks/blob/a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f/crates/sceneworks-worker/src/image_jobs/base.rs#L8515)). Merely calling `enter_phase` does not specify how the new runner supplies the corresponding phase identities.

**Proposed resolution:** the first geometry-changing feature must deliver its conservative transition pricing and phase-scope identities for its adopted routes, through a shared interface usable by existing admission consumers. Full admission unification may remain later. An unchanged estimator is acceptable only with an explicit bound already covering every added allocation and supported configuration.

**Closure:** header/plan-only decision tests at budget boundaries, scope-identity traces for both geometries, and dispatched peak checks in the existing validation tiers. Include count > 1 and a memory-rung case; do not run a Cartesian measurement grid.

## R2-07 — MAJOR: Hires.fix needs pass-local stages, not only a pixel transition

**Confidence: high.** References: REQUIREMENTS PIPE-1/4, HOOK-2; PLAN §3.3 lines 261–274, ENG-2. This extends the round-0 ENG-14 resolution.

The shared runner performs prompt preparation, encoding and references once before the phase loop. Adding `PixelRoundTrip` supplies the intermediate image but does not reproduce everything that happens between today's Hires passes. The first pass deliberately disables prompt enhancement; the final pass applies it, potentially changing the prompt and therefore encoded conditioning. The final pass also receives the resized first output as its single reference and clears the original multi-references and mask. See [base.rs:8699–8743](https://github.com/SceneWorks/SceneWorks/blob/a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f/crates/sceneworks-worker/src/image_jobs/base.rs#L8699), [8752–8762](https://github.com/SceneWorks/SceneWorks/blob/a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f/crates/sceneworks-worker/src/image_jobs/base.rs#L8752), and [8787–8809](https://github.com/SceneWorks/SceneWorks/blob/a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f/crates/sceneworks-worker/src/image_jobs/base.rs#L8787).

Moving enhancement to S1 changes the first pass. Reusing its conditioning for the second changes the refinement. Hiding all of this in a family transition recreates bespoke orchestration outside the shared stage owner and hides residency from the plan.

**Proposed resolution:** describe pass-local stage invocations and dependencies, including prompt preparation, re-encoding and conditioning replacement, with the shared runner still owning execution, events and cleanup. Characterize every currently supported Hires route before deleting the worker path. This is about preserving existing behavior, not adding a general-purpose workflow language.

**Closure:** a fake enhancer and fake generator record the same two prompts, conditioning inputs, seeds, progress and enhancement report ordering before and after. Include the codec/encoder/denoiser residency trace.

Related factual correction: PIPE-16 uses chunked Krea-realtime output as an existing migration case, but the cited baseline describes whole-clip generation followed by one decode and leaves per-chunk decode to a streaming epic ([t2v.rs:12–16](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs#L12)). Keep streaming support in the architectural contract, but distinguish new adoption work from preservation of the current batch route and reconcile its owning feature.

## R2-08 — MAJOR: Extract backend primitives below both media and LLM runtimes

**Confidence: high.** References: REQUIREMENTS BACK-1, COMP-1, XL-2; PLAN §1.1/1.2, §3.7, ENG-4/8/11/14.

The target map puts the shared Linear spine in `mlx-gen`/`candle-gen` and says the LLM crates consume it. But `mlx-gen` already depends on `mlx-llm` for shared text sampling ([Cargo.toml:159–161](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/Cargo.toml#L159)); [text_sample.rs:1–14](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/media/mlx-gen/src/text_sample.rs#L1) explains why. Adding `mlx-llm → mlx-gen::layers` creates a Cargo cycle. The flat layer diagram does not resolve that concrete crate edge, and ENG-4's scope does not sequence its removal.

**Proposed resolution:** identify a backend primitive owner below both runtimes, such as one backend-layer crate per backend, and draw the concrete dependency edges. Keep shared cancellation/load contracts below that edge as appropriate. Move consumers with the extraction so this does not leave cross-feature bridges. Preserve iOS partitioning and feature flags.

**Closure:** an explicit acyclic target crate graph and `cargo metadata --locked`/workspace dependency checks in the owning implementation feature. Specify when LLM projection adoption happens; BACK-1's acceptance must not claim it before that adoption.

## R2-09 — MAJOR: Complete the low-res recipe and its user-visible acceptance

**Confidence: high that the contract is incomplete; medium on achievable quality/speed until measured.** References: REQUIREMENTS FEAT-1, PERF-3, DEC-10; PLAN §3.1 compilation and §6.1.

FEAT-1 acceptance checks dimensions, refusal and speed. An image of the right size that loses the prompt or is badly blurred could satisfy those checks. As a recipe Feature it need not match the full-resolution image bit-for-bit, but it still needs an explicit usability and quality assessment.

Several parameters remain undefined despite appearing in the design:

- Is K counted in requested steps, actual executed img2img steps after strength truncation, or a particular pass? What happens at K=0, K=N, or when only one step executes?
- Which axes does `scale` affect? “final × scale” must distinguish spatial dimensions from video frames, audio duration/tokens and count. Minimum grids and aspect-ratio rounding need deterministic rules.
- Does low-res use the final geometry's schedule shift/mu, as evidence B §4 recommends, or a per-phase shift? PIPE-5 derives mu per phase, §6.1 exposes `mu_policy`, and DEC-10 does not settle that policy. Changing mu while slicing a global schedule needs an explicit continuity rule.
- §6.1 exposes `{renoise, mu_policy}` in addition to K and scale, while FEAT-1's UI lists only K and scale. Decide which are fixed/versioned recipe semantics and which are user parameters. Persist the resolved semantics for replay.
- DEC-10 initially refuses edit/reference/control routes, but §6.1's applicability predicate checks only geometry change and loop support. Carry the mode/conditioning restriction into the actual structural predicate and its export.

**Proposed resolution:** finish the recipe specification and add acceptance for deterministic replay, supported parameter boundaries, mode-specific refusal and prompt/content quality. Keep claims about speed conditional on the actual recipe and any boundary evaluation from R2-01.

**Closure:** weights-free plan cases for the listed boundaries and geometries, plus a bounded dispatched real-weight comparison covering representative prompts/seeds and relevant modes on each backend. Record quality observations or an agreed quality criterion alongside speed and peak memory. Do not turn this into a full grid or require active-recipe bit identity.

## Decision guidance and next revision

The recommendations for shared host policy, the three schedule kinds, explicit numerical classes, recipe/equivalent/diagnostic distinctions, stable seeds and outer-step progress are well motivated. Retain the pilot gate on backend-generic models. DEC-10 needs the additional recipe decisions in R2-01/09; they should not be left to individual family migrations.

The remaining business/process choices in §10 are still Michael's decisions. This review does not approve structural CI guards, change proof cadence, retire artifacts, or claim live runner capacity. No current capacity assumption is necessary for this document review.

Before implementation approval, a revised design should show four complete, weights-free worked traces:

1. FLUX.1 low-res start: request → validated plan → boundary estimate/noise law → phase scopes → output and replay.
2. Low-res + Hires.fix: explicit pass coordinates, intermediate decode/encode, prompt preparation and event semantics.
3. Krea adapter phases: effective preparation/cache keys and request-owned execution state.
4. Wan/H3: history and RNG continuity, per-evaluation model selection and concurrent AV memory pricing.

These traces are closure evidence for the design contracts. Native implementation, numerical fidelity and performance remain subject to the planned validation tiers. The full roadmap can remain; each prerequisite must land with the first feature that consumes it.
