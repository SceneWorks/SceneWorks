# Shared Generation Architecture: Requirements

| | |
|---|---|
| **Status** | **Draft 0.5, decision review in progress.** All four round-5 findings are addressed. Michael approved DEC-1…3 on 2026-11-03; DEC-4 reuse strategy is approved; its chained-pass product scope and DEC-5…19 remain open. This is not overall design approval or implementation evidence. Design-only: no program tracker items, pilots or implementation start from an individual decision approval. |
| **Date** | 2026-11-03 |
| **Repositories** | SceneWorks (`main` @ `a1d95e2c4`) and inference (`origin/main` @ `a43724b82`). All citations refer to those revisions. |
| **Companion** | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): target design, migration strategy and phased roadmap. |
| **Evidence** | [evidence/](evidence/): eight read-only research reports, cited as `[A §n]` … `[H §n]`. The internal reviews are summarized in [REVIEW_LOG.md](REVIEW_LOG.md). |
| **Origin** | We asked whether a "low-res first steps" acceleration could be added for every model. It could not be added once, because the engine has no shared pipeline. An audit then showed that the same structural cause is behind the month-plus memory-ladder effort. |

---

## How to review this document

1. **Every requirement has an ID** (`PIPE-3`, `ADM-2`, …). Comment by ID.
2. **Requirements state *what* must be true.** Design belongs in the implementation plan. Flag any requirement that reads like a design choice.
3. **§10 lists the decisions only Michael can make.** Each has a recommendation. A requirement that depends on an open decision is marked `⟂DEC-n`.
4. **The program acceptance tests (§7) define done for the whole program.** Each future CodeGraph feature takes the relevant subset as its own feature-level acceptance tests.
5. **Confidence labels.** `[H]`, `[M]` and `[L]` (high, medium, low) are taken from the evidence. §8 targets are goals, not gates.

---

## 1. Problem statement

### 1.1 The symptom

A cross-cutting capability costs roughly **families × backends × routes** of hand wiring:

- A low-res-first-steps option or a new memory rung must be built about 25–80 times.
- The memory ladder took over a month. Its gen-core design records "20 family stories". It left about 125k lines of per-engine `memory_strategy*` code (tests included), next to about 14k shared lines [E §1; prior audit].

### 1.2 The structural causes

**Engine (inference):**

| # | Cause | Evidence |
|---|---|---|
| C1 | **No pipeline or stage contract.** `Generator` exposes one opaque `generate`. The `Pipeline` trait deferred until "the second family lands" was never built. Each family hand-writes its own orchestration. | `mlx-gen/ARCHITECTURE.md:79-81`; `gen-core/src/generator/traits.rs:8-285` [A §2, P7] |
| C2 | **Design rule:** "the core stays fixed as models are added; each family crate owns its blocks". This pushes every cross-cutting feature into every provider. | `mlx-gen/docs/MODEL_ARCHITECTURE.md:24-36,134-139` [A §14] |
| C3 | **Every model is ported twice.** flux, qwen-image and wan share 0–5% of their code between MLX and candle. The newer "host math once" crates share 25–40%. Twins are hand-mirrored. | [D §2-3] |
| C4 | **Components are copied.** About 15 VAE ports per backend. Qwen-family encoder blocks appear 13× (candle) and 6× (MLX). Gemma is ported twice per backend. There are 194 Attention structs, 304 RoPE functions, about 45 timestep-sinusoid copies, and 68 `*Linear` types. | [C §1-6; E §2] |
| C5 | **The request is a 61-field union, about half of it family-specific.** Several fields are silently ignored rather than refused. | `gen-core/src/generator/request.rs:9-381` [A §5, P2, P12] |
| C6 | **No shared per-step or per-phase seam.** The step loop lives inside `Sampler::sample`. "Change something mid-trajectory" is implemented six different ways: Krea phases, LTX pass scales, Wan expert swap, Bernini omega rescale, flux `timestep_to_start_cfg`, and SenseNova `cfg_interval`. | [B §3.4, §3.7; A P9] |
| C7 | **Capabilities are hand-declared**, as 160–177 `Capabilities` literals. They are policed by a 4,516-line Rust source parser. | `candle-gen-catalog/src/preview_advertising.rs` [E §4] |
| C8 | **Weights, quantization and adapters are per family.** There are 74 loader files (about 45k lines) and 101–137 key-remap functions. The declarative `LogicalWeightPlan` has 5 mappings. MLX applies adapters at runtime; candle uses four different merge strategies. Candle keeps vendored model copies for two reasons: there is no hookable `Linear` (IP and packed variants), and its fused kernels have no backward pass (training copies). | [E §1-3, §5; C §6] |
| C9 | **iOS has its own engine-side admission and pricing system.** It spans `ios_admission.rs` (about 185 KB) and per-family `ios_pricing/` modules. It is separate from both the gen-core memory contract and SceneWorks admission. | `mlx-gen-ios-catalog/src/ios_admission.rs`; review [ENG-12] |

**Application (SceneWorks):**

| # | Cause | Evidence |
|---|---|---|
| C10 | **Model facts are restated in many hand tables.** There are more than 20 named tables and about 230 `const` slice tables across core, worker and API `[M]`. Drift is policed by tripwires: a 255k-line memory matrix built by regex-parsing 19 Rust files, and a backend-capability matrix that byte-digests 61 source files. | [G §1-2, §7]; `routing/matrix.rs:3882-3912`; review [APP-15, APP-26] |
| C11 | **Admission is fragmented across at least 25 paths.** The 18 inventoried paths [F §4] plus MOSS, SenseNova, InstantID, Iris training/restore/depth, and API-side training refusals. Each backend has its own budget structs and probes, and floor semantics differ. Some routes skipped admission entirely. | [F §4]; review [APP-7]; `conditioning_gate.rs:17` |
| C12 | **There is no lane harness.** The image dispatch function is about 1,305 lines, backed by large families of per-lane `*_available`, `raw_settings`, `ensure_*` and `generate_*stream` functions. Video already has a near-harness (`generate_video_using`). | [F §1, §3] |
| C13 | **Validation is duplicated** across web, API, core and worker. Hires.fix rules appear in about 9 places. One advanced option touches about 9 files. | [G §3-4] |
| C14 | **Models travel as untyped JSON.** The typed `ModelManifestEntry` is used only by one fixture test. 58% of the manifest's bytes are generated memory data. | [G §1] |

### 1.3 Prior attempts and why they stalled

[H §2]

- **The gen-core memory contract landed.** The candle consolidations (loaders 34→1, `QLinear` 4→1, merge skeletons 8→1) grew back within 11 days. The copy-paste lived in each adopter's *wiring*.
- **Long convergence branches died 4,000+ commits behind `main`.** These are `feature/sc-19048` (137 commits) and `feature/sc-20414` (47 commits).
  - sc-20414 contains a chained denoise-pass executor (`gen-core/src/sampling/pass_executor.rs`), a backend-neutral conformance testkit, and ComfyUI pass-schedule math.
  - It is **fresh-schedule-per-pass only**. It always re-noises between passes, has no geometry change, and leaves guidance combine to the host. It explicitly refuses to coexist with Krea `phases`.
  - It was adopted only for the new `advanced.denoisePasses` request, on curated lanes of Krea, SD3 and SDXL.
  - Its *invariants and testkit* are reusable prior art. Its executor is not a phase-runner base [review ENG-1].
- **Two designs were proposed and never ratified:** the SceneWorks `docs/memory-strategy-pipeline-target-design.md`, and the engine `Pipeline` trait.

**Lessons carried in:**
- Hoist the *adoption template* (loop ownership, hooks, cancellation, progress, cleanup), not just the contract.
- Ship features that reach `main` within weeks.
- The feature that replaces a source migrates every consumer of it (AP9).

---

## 2. Goals

| ID | Goal |
|---|---|
| **G1** | **One change per cross-cutting feature.** A generation feature (low-res first steps, a guidance trick, per-phase adapters, a preview mode, a memory rung) is implemented once in shared code. Every route whose structure supports it gets it with **no family-file edits**. Routes that cannot support it refuse with a typed reason. For plan-rewrite and latent-op Features, "once" means one module. For model-internal hooks, it means one module per backend until the pilot-gated generic rollout under approved DEC-1. |
| **G2** | **Hooks and entrypoints at every stage** of the stack: request and options, plan, weight lifecycle, pipeline stages, phases, steps, blocks, layers, decode, output, admission, lane execution and events. |
| **G3** | **Every fact has exactly one primary declaration; everything else is derived.** This covers model facts, capabilities, limits, defaults, routes and memory facts. No hand table restates another source. No generator reverse-engineers facts from source text. |
| **G4** | **Slim the codebase** by deleting duplicate implementations, not by moving them. Targets are in §8. |
| **G5** | **Share model code across backends** as far as the evidence supports. Approved DEC-1 establishes shared host math and identical layer APIs as the baseline, with generic model graphs conditional on both BACK-5 pilots passing. If either fails, the baseline remains a valid outcome. |
| **G6** | **Adding a model is data and graph only.** A new model in an existing class needs config, graph, weight plan, route record, product-manifest entry and its declared measured anchors (PAT-3). It needs no SceneWorks code tables, admission wiring, lane wiring or per-family memory code. |
| **G7** | **No regression** in output, speed, peak memory, features, UI controls or platform coverage (macOS, iOS, Windows/Linux CUDA, CPU), except changes explicitly approved under AP13 class (c). |
| **G8** | **Safe, incremental delivery.** Each feature lands on `main` within weeks, protected by characterization tests, and deletes what it replaces. |

## 3. Non-goals

- **Changing user-visible behaviour, models or defaults as a side effect.** New features such as low-res start are explicit and opt-in. Unavoidable output changes need per-route approval (AP13 (c)).
- **Changing checkpoints, provider IDs, weight keys or persisted artifacts.** Exceptions are named in a requirement and upgraded under OPT-6.
- **Cross-backend output identity.** MLX and candle draw different noise and use different kernels, so they are not, and will not become, bit-identical to each other. Fidelity is always judged per backend.
- **A dynamic plugin ABI** or third-party runtime code loading. This preserves the 2026-07-13 inference decision.
- **Replacing the build or CI system, or re-enabling any dismantled measurement gate.** See FEATURE_DEVELOPMENT.md *Gate teardown*.
- **Forcing essential bespoke loops into the shared driver.** Essential variance stays family-owned behind a typed escape hatch (PIPE-7).
- **A lowest-common-denominator engine.** Under approved DEC-1, any pilot-gated generic model rollout keeps the seam at the *layer* level and retains backend kernels and fusions [D §10.1].

## 4. Scope

| Area | In scope | Notes |
|---|---|---|
| Image generation, all families, both backends | Yes | |
| Video and audio-video (Wan, LTX, MiniMax-H3, Mochi, SVD, SCAIL-2, Bernini, Krea-realtime, SeedVR2) | Yes | All axes are designed from day one. Image-only abstractions have been rebuilt for video twice [H §7.7]. |
| Audio generation (`candle-audio-*`) | Yes: weights, Linear, adapters, runtime contracts, residency vocabulary, and the runner where a flow/diffusion loop exists | DEC-2 approved; autoregressive routes retain their token driver |
| LLM crates | Yes, as the shared decoder-block and encoder library, and for unified runtime contracts | Chat-product features are out of scope |
| Training (LoRA trainers, Iris curricula, control training) | Yes, via a differentiable mode of shared modules | DEC-2 approved; trainers retain optimizer, loss and data orchestration |
| iOS (`mlx-gen-ios-catalog`, `runtime-ios`, engine-side iOS admission) | Yes | MEM-6, XL-3 |
| SceneWorks web, API, core, worker, memory-adapter, scripts, config, MCP | Yes | |
| Film harness, training plans, desktop seeding | Where they carry options or model facts (OPT-6, APPREG-5) | |
| External consumers: ChatWorks (pins inference runtime tags), SceneWorks-mobile (future `runtime-ios` consumer), SceneWorks-Art (consumes share envelopes and recipes) | Their contracts must keep working | XL-4 |

**Approved DEC-2 scope and sequencing:** audio, LLM-based encoders and training are all program deliverables. Design their shared interfaces from the start, with adoption staged around the image/video core. Early primitive extraction includes LLM projections and training-compatible variants; broader adoption follows the named roadmap features. Sharing preserves the distinct denoising, token-generation and training drivers and the existing platform partitions.

---

## 5. Architectural principles

Every requirement inherits these. A design that violates one needs an explicit decision.

| ID | Principle |
|---|---|
| **AP1** | **Policy is tensor-free; kernels belong to the backend.** Plans, schedules, phase math, hook ordering, admission math and capability derivation live in tensor-free contract crates. Backends implement tensor operations only. This is proven by tiling, attention budget, block window and residency [A §8]. |
| **AP2** | **Shared code owns the loop; families supply narrow hooks and data.** Families never re-implement cancellation, progress, preview, cleanup, phase iteration or step-invariant caching. |
| **AP3** | **Off means absent, and absent means byte-identical.** A disabled feature or hook produces numerics byte-identical to today's per backend, with peak memory within PERF-2. |
| **AP4** | **Refuse, never silently ignore.** A request field or feature that a route cannot honour is refused at plan time with a typed reason. |
| **AP5** | **One primary declaration per fact; everything else derived.** Structural facts that only a family knows (prediction type, time convention, legal grids, `DriverSupport`, geometry-change support) are declared once, by the family. Every *behavioural* declaration that could disagree with the code has a Tier 0 conformance check (MIG-3). |
| **AP6** | **Structural facts, not identities.** Shared code and feature predicates key on declared structure, never on model or family id strings. |
| **AP7** | **Essential variance is typed; accidental variance is deleted.** The classification follows the variance catalog [B §3]. |
| **AP8** | **Backend differences are explicit hooks, not averages.** Examples: budget probe, margins, failure posture, materialize/eval, matmul strategy, layout [H §7.8; D §10]. |
| **AP9** | **No cross-feature bridges; the producer migrates its consumers.** A new seam migrates real production callers and deletes the path it replaces in the same change. A feature that replaces a source (an export, a contract, a request shape) migrates *every* consumer of that source in the same feature. A consumer feature starts only after its producer is complete. There is no contract-only scaffolding, no "mechanism now, activation later", and no interim adapter that outlives its feature [H §7.9; review APP-3]. |
| **AP10** | **Characterize, then consolidate.** Pin each copy's behaviour before merging copies [H §7.10]. |
| **AP11** | **Simplicity over apparatus.** No new generated matrices, multi-file evidence chains or hash-pinned producers. Runtime refusal is the preferred error mechanism [H §7.20]. |
| **AP12** | **Stable external contracts.** Provider IDs, weight keys, persisted payloads, recipes, share envelopes, the persisted per-image seed relation and stamp keys stay readable. Request shapes change only through versioned upgrade (OPT-6). |
| **AP13** | **Numerics classes.** Every migration that touches computation declares one class per route: **(a) bit-exact** (the default); **(b) component-tolerance**, a bounded divergence measured at a component boundary (encoder output, one denoiser forward, one decode) on shape-faithful weights; **(c) accepted re-baseline**, a visible output change that needs per-route sign-off from Michael. On chaotic samplers a one-ULP difference produces a visibly different image (`candle-gen-sdxl/src/adapters.rs:6-11`), so such changes are class (c), not (b). ⟂DEC-7 |

---

## 6. Requirements

Format: **ID — statement.** *Rationale and evidence.* **Accept:** observable acceptance.

### 6.1 Engine pipeline (PIPE)

**PIPE-1 — Every generation route executes through one shared pipeline runner with a fixed stage vocabulary.** Plan compiles the complete run once; each pass then invokes the stages it needs in this order, with explicit dependencies on earlier pass outputs:
- Plan
- Prompt-prep
- Encode
- Reference
- Schedule
- Init
- per phase: Enter, Denoise, Exit
- Post-latent
- Decode
- Output

A **pass** has its own resolved schedules, prompt/conditioning bindings, init and decode policy. A **phase** is a slice within that pass. Stage results may be reused only when their declared dependencies match. Intermediate decode/re-encode and later prompt preparation remain runner-owned stage invocations, not hidden work in a transition or driver. The plan lists their residency and event visibility. This is a finite generation-stage vocabulary, not a general workflow language.

Encode includes declared dependencies among encoder components, such as Qwen edit's vision features feeding its text encoder. Reference denotes subsequent latent/aligned conditioning; it must not force vision inputs needed by Encode to be produced afterward. Those component dependencies and lifetimes are explicit in the same plan.

*C1; the stage model is derived from 14 family walks [B §2].* **Accept:** for every row of the route inventory (PIPE-18), a weights-free trace test shows the runner invoking the stages in dependency order. The Hires.fix trace preserves first-pass unenhanced prompts, final-pass enhancement/encoding, reference replacement and event ordering. No route has a hand-written `generate` that bypasses the runner.

**PIPE-2 — The request is compiled into a typed `RenderPlan` before any weights move.** The plan holds every decision a Feature may rewrite: passes and their stage dependencies; per-phase, per-stream geometry; schedules and coupling; guidance; conditioning requirements; transitions and boundary estimates; decode; memory inputs; seeds; and output. After all rewrites, the complete plan is validated for applicability, legal grids and step domains, ordering cycles, ownership, sampling conventions, driver support, stage dependencies and history compatibility. A changed admission selection revalidates and re-prices the resolved plan before execution. **Accept:** for every route, a plan can be built, inspected and serialized without loading weights. Invalid combinations refuse before loading. Admission consumes that exact resolved plan (ADM-3).

**PIPE-3 — Phases are first-class.** A phase is a span of the trajectory with optional overrides: per-stream geometry, model slot, adapter scales, guidance plan, solver and **solver-history policy** (`Carry` or `Reset`).
- **Model-slot selection** may be a σ-keyed rule evaluated per model evaluation. Wan curated MoE selects the expert per evaluation by σ, including inside Heun and DPM++-SDE sub-evaluations (`mlx-gen-wan/src/pipeline/moe.rs:139-145,187-194`).
- **Solver history is never carried across a geometry change or a re-noise.** It may be carried across a continue-with-same-geometry boundary. Wan native MoE carries UniPC history across the expert switch (`moe.rs:33-40`); Krea resets it per slice.
- **Execution state is request-owned and outlives phase calls.** Carry requires the same solver implementation/variant, sampling convention and compatible schedule continuation; any explicitly supported conversion must declare and test its state mapping. Previous timestep intervals, driver caches and absolute stochastic draw coordinates have explicit lifetimes. Resetting solver history does not implicitly reset RNG position or an autoregressive cache. A pass boundary uses a declared legacy or new-stream RNG policy, never accidental local-index restart.
- **Guidance and Feature state have their own lifetime rules.** A boundary-estimate evaluation reads a snapshot and cannot commit momentum, feature-cache history, counters or causal state. Geometry/branch/model changes invalidate shape/configuration-dependent state according to declared dependencies; unchanged native trajectories retain their existing update cadence. This is distinct from solver-history Reset and RNG continuation.
- **The six existing mid-trajectory mechanisms (C6) become phase data** and their family-local implementations are deleted, each class (a) unless listed otherwise.

**Accept:** a Tier 0 runner trace equals the legacy `denoise_moe` / `denoise_moe_curated` traces over a mocked backend. An inert split preserves the unsplit solver and stochastic draw trace; incompatible Carry refuses; geometry/re-noise resets history independently of RNG position. A boundary estimate leaves APG/Feature state unchanged, and geometry changes cannot reuse old-shape momentum/cache tensors. All six mechanisms are re-expressed and their local implementations deleted.

**PIPE-4 — Schedules and transitions are explicit and typed.**
- **Per-pass schedule source kinds** (DEC-3 approved); all resolve to one immutable schedule representation, sliced by pass-local executed outer-step ranges:
  - **global-sliced**: one schedule, contiguous slices (Krea, low-res start);
  - **fresh-per-pass**: ComfyUI `steps/denoise` semantics, from sc-20414;
  - **explicit table**: a baked per-phase σ table (LTX stage 1 runs to σ=0, then a baked 3-step stage-2 table, `mlx-gen-ltx/src/pipeline.rs:51-56,591`).
- **Source resolution does not select transition or history policy.** Each pass owns its schedule and step numbering; a phase split continues within that domain. Re-noise, clean-estimate conversion and solver-state compatibility remain explicit under PIPE-3/4.
- **Transitions:**
  - continue;
  - re-noise to a σ with a named seed stream;
  - latent resample (nearest, bilinear, or a learned upsampler resolved as a route *component* with a declared normalization);
  - **pixel round-trip** (decode, pixel resize, re-encode; required by Hires.fix, `worker/src/image_jobs/base.rs:8548-8615`);
  - compositions of the above;
  - a **route-supplied transition** where a family's existing transition must stay bit-exact. LTX's dtype-preserving re-noise with pre-drawn stage noise is one example (`pipeline.rs:157-162`).
- **Stream targeting:** every transition names the streams it applies to (LTX upsamples video but only re-noises audio, `pipeline.rs:1579-1588`).
- **Boundary state is typed:** noisy `x_sigma`, a clean estimate with its evaluation provenance, or a route-specific state with a declared conversion. Re-noise-from-clean never accepts a noisy latent. The destination sampling convention supplies the noise scaling; flow and VE/EDM laws are distinct. The plan declares how any required estimate is obtained, including a read-only endpoint evaluation, its slot/adapter/guidance configuration, dtype, memory and evaluation cost. A cached intermediate solver estimate is not silently treated as an endpoint estimate.

**Accept:** a weights-free conformance suite covers every schedule source and transition kind, stream targeting, flow versus VE/EDM noise laws, estimate provenance, deterministic seeds and progress continuity. A transition cannot double-noise by interpreting `x_sigma` as x0. Numerical and native-transition variants remain explicit under AP13.

**PIPE-5 — Latents are stream bundles, and geometry-dependent state is derived per phase.**
- **Streams:** a latent is a bundle keyed by stream (video, audio, joint rows). Each stream has its own `LatentSpace`, its own geometry (grid, frames, audio tokens or duration), its own schedule and its own σ.
  - H3 steps video and audio on two schedules (`mlx-gen-minimax-h3/src/denoise/mod.rs:1-13`).
  - LTX uses per-token time; the time input may be a per-token builder.
- **Derived per phase and per stream:** RoPE and position ids, packed layouts, masks, cross-KV, AdaLN tables, grid-aligned conditioning and preview projection. Schedule `mu`/shift is resolved per pass from its declared geometry policy, not recomputed implicitly when a phase changes geometry.
- **Preparation dependencies are declared.** They include effective model/weight identity, conditioning identity, relevant adapter stack/scales and branch structure where those affect the result. Preparation receives request-scoped arguments, as forward does. Geometry-only caches can retain narrower keys. A prepared cache is request-scoped unless an immutable cross-request cache includes all dependencies.
- **The step-context cache** evicts on phase exit unless a later phase reuses the complete dependency key; retained allocations remain priced.

**Accept:** a geometry-changing phase on any supporting route produces the requested final size. The trace shows correct cache reuse and invalidation for Krea base → adapter → base preparation, CFG-off → CFG-on, and interleaved requests; no stale fused conditioning is reused. Keys are built once per complete dependency set and evicted per the rule.

**PIPE-6 — One sampling driver, identical in shape on both backends.**
- **The driver owns:** solver lookup, per-evaluation cancellation, progress, preview counting, latent observation and span iteration.
- **`StepInfo` carries:** pass, phase, step, outer step, evaluation index and per-stream σ.
- **Shared tensor-free code replaces** the duplicated pieces: schedule resolution, img2img start step, `mu` and shift, and the ≥6 candle `init_time_step` copies [A §10, P8-P11].
- **User-visible progress stays in outer steps.** Per-evaluation accounting stays internal (⟂DEC-19).

**Accept:** both backends' drivers have the same signature and hook struct. The duplicated functions exist once, in gen-core.

**PIPE-7 — Essential bespoke loops run under the runner through a typed `FamilyDriver` escape hatch.**
- **Loops that use it:** FLUX3 flow-map, Sana Sprint SCM, Boogu Turbo DMD, MOSS-Image DMD, MiniMax-H3, LTX native, SenseNova, Mage, the Bernini MAR planner, Krea-realtime (autoregressive self-forcing, KV across blocks), Iris FlowDPM-Solver++ (with iOS quanta), the SDXL native ancestral and Wan native UniPC parity loops, the SeedVR2 loop, and the PiD decoder sampler (hosted inside the Decode stage).
- **Drivers are generic over a forward seam** (the precedent is H3's `JointVelocity`, `denoise/mod.rs:25-29`). This lets a Tier 0 conformance suite verify every capability the driver declares in `DriverSupport`.
- **Each driver honours `SliceInput`:** arbitrary entry σ, geometry, `Continue`/`Renoise`/`Resample`, and the history policy. Any refusal of these must be approved under ⟂DEC-13.

**Accept:** every listed loop runs under the runner. Its `DriverSupport` claims pass conformance. Refusals are on the approved list.

**PIPE-8 — Guidance is one library of named, bit-exact operators.**
- **Operators:** plain CFG, true-CFG, norm/std rescale, CFG++, APG, CFG-Zero* and N-branch, each in every **variant** in use today (operand order, combine dtype, sign convention). Boogu combines in model dtype, then casts and negates (`mlx-gen-boogu/src/pipeline.rs:563-576`). Candle SDXL computes `uncond + (cond−uncond)·s` (`denoise.rs:349`).
- **Each route declares its combine variant and its branch row order.** Unifying them is a separate, optional AP13 (c) decision.
- **The family's predictor receives the set of branches** and decides how to batch them. Batched CFG stays batched (`candle-gen-sdxl/src/denoise.rs:300-349`).
- **A branch may carry a model perturbation**, such as LTX STG skip-layers or isolated modality, which block hooks honour.

**Accept:** no family file contains a guidance combine. Every migrated route's goldens are unchanged (class (a)).

**PIPE-9 — Schedules are data.** Every native σ builder, shift function and baked table is registered as a named schedule in gen-core. Families select a schedule and its parameters. *[C §4; A P10.]* **Accept:** no per-family σ or shift function exists outside gen-core.

**PIPE-10 — Numerical faithfulness follows AP13.**
- **The default is class (a), bit-exact per backend** against the pre-migration route for the same seed and settings.
- **The implementation plan lists, for every feature, which routes are expected in class (b) or (c), and why.**
- **Class (c) changes ship only with Michael's per-route sign-off**, recorded in the PR.

**Accept:** a Tier 1 result per migrated route (MIG-3); class (b) and (c) entries carry recorded evidence; class (c) entries carry sign-off.

**PIPE-11 — Seeds follow a route-declared `SeedScheme`.**
- **Today's derivations become named legacy schemes, preserved bit-exact.** These include `base.wrapping_add(i)` per image, LTX `seed ^ 0x5DEE_CE66`, candle `+1..+3`, and parity RNGs.
- **New noise streams use one splitmix derivation with per-stream salts.** Feature re-noise and recipe-introduced draws get named domains; adding an inert phase boundary does not create a new noise stream or restart existing draws. The precedent is sc-20414's `DENOISE_PASS_RENOISE_SALT`.
- **The image-index → seed relation is exported per route.** It is the relation SceneWorks persists for regenerate and replay (`worker/src/image_jobs/stream.rs:441`).

**Accept:** replay is bit-identical; the persisted per-image seeds are unchanged; no legacy scheme changes without a class (c) decision.

**PIPE-12 — Outputs and reports are typed.**
- **Outputs:** `Images`, `ImagesRgba`, `Video{frames, fps, audio}`, `Audio`, interleaved text and images, and streamed chunks (PIPE-16).
- **Execution reports** come back from the run as a typed report, not from `&self` getters [A P1, P13].

**Accept:** `Generator` has no family-named methods.

**PIPE-13 — Count and batch semantics are explicit.**
- **The plan states, per route:** whether count = N runs as N sequential trajectories or one batched latent; whether multiple samples are packed into one sequence (Mage); per-image seed derivation (PIPE-11); and whether previews, cancellation and outputs are per image or per batch.

**Accept:** the plan snapshot shows these per route; Tier 0 tests cover count > 1 on each kind.

**PIPE-14 — Concurrency is safe by construction.**
- **Shared resident modules are immutable during a run.** Today's invariant "keep the shared resident DiT unmutated" (`gen-core-family/src/krea_multiphase.rs:36-45`) holds. Per-request state is passed through forward arguments or the step context, never stored on shared modules. That state includes adapter scales, step context and guidance state.

**Accept:** a Tier 0 test runs two interleaved requests with different phase scales on one provider; each yields its single-request trace.

**PIPE-15 — The engine has a typed error taxonomy and failure posture.**
- **Errors are classed** as request-local, provider-poisoned or process-poisoned.
- **GPU cleanup hooks run only for request-local errors.** A Metal error that poisons the process skips GPU work, consistent with the standing notes on `SubmissionsIgnored` and the GPU watchdog. Host-side accounting/claim release is unconditional and never retries poisoned GPU operations or masks the original error.
- **Cancellation is typed** at every evaluation and phase boundary, and during loads and model-slot swaps (WGT-5).

**Accept:** a Tier 0 fault-injection suite covers each class at each stage.

**PIPE-16 — The runner supports chunked and incremental output.** Chunks are runner-owned repeated stage invocations delivered through the event sink. Buffering is bounded, cancellation is supported, and causal codec state and residency are declared.
- **Today's state:** Krea-realtime generates a whole clip and then decodes it once (`mlx-gen-krea-realtime/src/t2v.rs:12-16`).
- **Production use:** the existing batch path uses the same output contract as one terminal chunk. Its public output, events and persistence stay unchanged; internal chunk metadata does not create a new streaming product surface. Repeated delivery uses that same primitive, rather than a second unused hook.
- **Ownership:** decoded streaming product delivery belongs to **SW-F148 (`sc-8432`)**, including its feasibility spike and permitted NO-GO outcome. SGA owns the shared contract and batch adoption. A NO-GO leaves the contract's batch production user intact. Stream-only hooks are implemented with their first production consumer under HOOK-1, not as dormant SGA scaffolding.

**Accept:**
- The production batch path goes through the shared contract and yields the same final output/event sequence as before.
- A mocked multi-chunk producer proves ordering, backpressure, cancellation and bounded retention in Tier 0. This is contract evidence, not proof of decoded streaming delivery.
- Where SW-F148 has delivered a streaming path, it adopts that same contract with production evidence in its owning feature; otherwise SGA makes no streaming-delivery completion claim.

**PIPE-17 — Telemetry flows through `gen-core-telemetry`** at stage, phase and lifecycle hooks. Existing telemetry fields stay stable (OPT-8). **Accept:** existing telemetry consumers see unchanged field names.

**PIPE-18 — A route inventory is generated from the catalogs.** It has one row per (family, route, backend), with the planned loop kind, the owning feature and the expected AP13 class. It is a typed dump (AP11), not a hand list. **Accept:** every program feature's exit criterion is stated against inventory rows; the program ends with every row migrated or refused by an approved decision.

### 6.2 Hooks and features (HOOK)

**HOOK-1 — The stack exposes a documented hook map.** Each hook point declares its stage, its timing (before, replace, after or observe), its inputs and outputs, its ordering and whether it is result-changing. The minimum set is in the implementation plan §5. **The hook enum in code is the source of truth.** Any listing is a typed dump, not a checked-in matrix (AP11). **Accept:** every hook point has at least one production user, or it is removed.

**HOOK-2 — A cross-cutting capability is a Feature.** A Feature has a tensor-free, object-safe **`FeatureSpec`**:
- id and typed parameter schema;
- class: recipe, equivalent or diagnostic;
- applicability predicate over route structure;
- relations: conflicts, requirements, ordering;
- plan rewrite;
- memory effect, including transition residency;

and optional per-backend **`FeatureHooks`**, for model-internal hooks such as caches or attention perturbation.

**Accept:** these are implemented as Features with no family-file edits: low-res start (FEAT-1), per-phase adapters and guidance, and Hires.fix (via pass-local stage bindings and the pixel round-trip transition, with all intermediate residency priced).

**HOOK-3 — Features compose deterministically.**
- **Pass producers establish stable schedule domains before targeted phase contributions.** Boundaries carry a pass identity and executed outer-step coordinate; stream-coupling rules supply the corresponding per-stream sigma. Ownership includes schedule policy, transition composition and history as well as geometry/model/adapters/guidance/solver. Within a pass, each override axis has one owner per stream/span; competing boundary operations need an explicit typed composition or refuse. Native route defaults expose only their declared override surfaces.
- **Ordering at a shared hook point is declared**, never left to registration order.

**Accept:** a conformance test covers pairwise composition of all registered Features on a synthetic route, plus whole-active-set cycle/ownership validation. Worked plans cover low-res + Hires, Krea phases and a coincident Wan expert/geometry boundary. Every resulting pass is revalidated under PIPE-2; a registration order change cannot change the resolved plan.

**HOOK-4 — Applicability is computed and exported.** For each route and backend, the engine computes which Features apply, why the others do not, and the parameter limits. That export is the UI-availability source for SceneWorks (REG-3, OPT-2). The worker's linked engine is authoritative at run time (REG-3). **Accept:** removing a Feature's structural precondition in a test flips availability in the export and the UI with no table edits.

**HOOK-5 — Hooks cost nothing when unused.** A route with no active Feature runs exactly today's computation (AP3), with no measurable per-step overhead (PERF-1).

### 6.3 Shared components (COMP)

**COMP-1 — Text and condition encoders are a shared library built on the llm crates.**
- **Existing assets come first.** llm `CausalLm` already covers Llama, Mistral, Qwen2/3, Gemma-2, GLM-4 and Gemma 4. It already exposes a text-encoder tap API with positions and padding masks (`mlx-llm/src/models/llama.rs:1081-1135`; candle twin `candle-llm/src/models/llama.rs:1289-1391`). flux3, iris, lens and sensenova already depend on it.
- **`TapPolicy` extends that API** with early exit, layer subsets and combine modes.
- **Named blockers:**
  - causal-only masks (`llama.rs:416`);
  - every hidden state is returned (a memory cost for 12B encoders);
  - quant codec mismatch: candle-llm uses Q4_0/Q8_0/Q4_K, media uses Q4_1 and the MLX affine tiers;
  - missing architectures, including the Gemma 3 and Qwen2.5-VL vision towers.
- **Encoder swaps change conditioning bits**, so each migration declares class (b) or (c).
- **Towers** (T5/UMT5, CLIP-L/G/OpenCLIP, SigLIP, ViT) get one implementation per backend each.

**Accept:** no family crate defines a text-encoder transformer block.

**COMP-2 — Latent codecs are a parametric library behind one `LatentCodec` contract.** The contract covers encode, decode, latent-space descriptor, tiling and decoder override.
- **AutoencoderKL-2D** covers SDXL, FLUX.1, SD3, Z-Image, Chroma, FLUX.2 and FLUX.3.
- **Wan causal-3D** covers Wan 2.1/2.2, Qwen-Image and Qwen-Image 2.1.
- **A causal-3D kit** serves LTX and H3.
- **Singletons stay on the kit:** DC-AE, the Mage NeRF decoder and the audio codecs.
- **Each codec declares its tiling law** (Wan vae16/vae22, LTX causal temporal overlap, DC-AE).

**Accept:** each VAE architecture is implemented once per backend (or once overall under DEC-1 (c)), with tiling inside the codec.

**COMP-3 — Transformer stacks run through a shared `BlockStack` driver with per-block hooks:** `pre_block`, `extra_kv`, `post_block`, stream/evict and branch perturbation. Families declare their block layout. The driver owns iteration, R4 streaming, ControlNet/VACE residuals, IP/PuLID extra-KV, preview taps and checkpointing. **Accept:** block streaming, control and IP injection exist once per backend; no family `block_stream.rs` remains.

**COMP-4 — A sub-block kit replaces per-family copies:**
- norms;
- adaLN;
- MLPs;
- QK-norm;
- RoPE (1D/2D/3D axial, NTK/YaRN);
- the attention prologue (MLX `qkv.rs`, ported to candle);
- timestep embedding;
- patchify;
- position-id builders.

Kit replacements that change operation order declare class (b) or (c). **Accept:** families compose kit parts; family-local definitions remain only for documented essential variants.

**COMP-5 — Conditioning modules are shared and declared per route:** ControlNet, VACE, IP-Adapter, PuLID, InstantID and reference-token concatenation. Grid-aligned conditioning is typed separately from geometry-free conditioning. **Accept:** each conditioning kind is implemented once per backend.

**COMP-6 — Host-side pre- and post-processing exists once:** resize, normalize, Qwen-VL smart-resize, image and alpha conversion, frame packing and audio fit.

### 6.4 Backend layer (BACK)

**BACK-1 — There is one `Linear` spine per backend.**
- **Bases:** dense, MLX affine-packed tiers, GGML, GGUF k-quants, NVFP4, FP8/INT8 row, W4A8, ConvRot and existing LLM-specific projection variants (including Prism). Extraction preserves their supported operations and formats; no tier or variant is dropped to simplify the spine.
- **Plus:** dtype policy, residual adapters, a **request-scoped** scale context passed through the forward (PIPE-14), and a trainable mode.
- **The llm crates use the same spine.**
- **The backend primitive owner sits below both media and LLM runtimes.** A shared primitive must not create a media↔LLM dependency cycle. Its extraction migrates its consumers in the same feature, preserves platform feature partitions and does not introduce cross-backend dependencies.

**Accept:** one Linear spine per backend, including LLM projections, and an acyclic locked Cargo graph. ENG-4a owns extraction and media/LLM projection adoption; program-wide acceptance follows audio adoption in ENG-14. Later LLM adoption covers other runtime contracts. The vendored candle copies created for IP and packed variants are deleted in ENG-4b: `ip_dit` and `packed_dit`, plus `packed_te` (the named files total about 11.9k lines; [C] estimates about 20k including others `[M]`).

**BACK-2 — There is one attention operation per backend.** Every block routes through it. It carries masks, budgeted and bounded execution, sparse modes, branch perturbations and the i32-overflow guard. **Accept:** no raw SDPA call remains outside it.

**BACK-3 — Execution policy is a backend hook set:** materialize/eval, fusion and compile regions, memory scope, cancel points, stream scope and load activity. The MLX watchdog rule is enforced inside the loader and decode stage. That rule is: materialize pending loads before any GPU-derived view, and never eager-read inside a block stream. **Accept:** no family code calls eval or materialize directly.

**BACK-4 — Both backends expose an identical layer-kit API**: names, shapes, broadcasting and dtype promotion. Layout is explicit at conv boundaries and converted at load. **Accept:** for migrated families, the two backends' graph files differ only in imports and backend types.

**BACK-5 — Backend-generic model code (DEC-1 approved, rollout conditional).**
- Model graphs are written once, generic over a layer-level backend trait.
- Two pilots gate any rollout: (A) a small family; (B) FLUX.3 end to end, including LoRA training.
- Both must preserve the declared capability surface and per-backend numerical fidelity, with speed/peak memory satisfying PERF-1/2 and iOS binary size satisfying PERF-2. Existing pilot acceptance retains bit-exact output against each backend's current twin, measured A/B on the same runner.
- Rollout proceeds only if both pass. Otherwise retain option (b), the shared-host-math/identical-layer-API baseline, and record the failing seam; do not force all models through an abstraction that failed a pilot. Shared pipeline work and low-res start do not wait for this outcome.
- This is an approved design direction, not authorization to run pilots or start implementation during decision review.

### 6.5 Weights, quantization, adapters, training (WGT)

**WGT-1 — Every load goes through one declarative `WeightPlan`.**
- **The plan declares:** components, source dialects, key rules, layout rules, dtype policy, quant predicate, block layout and expectations. It compiles header-only and fails closed.
- **SceneWorks MLX affine tiers, GGML and GGUF are registered codecs.**
- **Each component declares its quantization tier**, for example a q4 DiT with a bf16 text encoder.

**Accept:** no family loader or remap function remains; plan receipts are snapshot-tested in Tier 0.

**WGT-2 — The weight lifecycle has staged hooks:** stage → pin/seal → plan → decode → layout → quantize → residency → bind → adapters → receipt. **Accept:** load progress is generic and truthful for every model (standing rule `load_progress_must_be_generic_and_truthful`).

**WGT-3 — Adapters are one layer.**
- **Host routing is generated** from canonical module names, and alias tables are data.
- **The install policy is declared per route.** Today's policy stays the default: fold-at-load on candle dense bases exists for parity, and switching it is class (c).
- **A route switches to residual only when an active Feature needs runtime scales** (for example per-phase adapters). That switch is result-changing and declared in the plan.
- **Non-commutative adapters (DoRA/BoRA) use ordered fold.**

**Accept:** per-phase adapter Features work on every route that is residual-capable or can switch to residual; merge-only routes refuse structurally.

**WGT-4 — Training reuses inference modules in differentiable mode (DEC-2 approved).** The backend switches from fused to composable kernels. Trainers get adapter injection, frozen-stage caching, x0 decode and checkpoint boundaries from the shared layers; optimizer, loss and data orchestration remain trainer-owned. **Accept:** no trainer-specific model-graph copy remains (`dit_train`, `train_dit` and kin).

**WGT-5 — Loads and model-slot swaps are cancellable** at component and batch boundaries, with cleanup. For example, a Wan expert swap reloads 14B parameters mid-run. **Accept:** a Tier 0 cancellation test at each lifecycle stage passes.

### 6.6 Memory (MEM)

**MEM-1 — Each memory rung is implemented once, in the shared layer that owns its resource:**
- R1 → runner stage driver;
- R2 → latent codecs;
- R3 → attention op;
- R4 → `BlockStack`.

Families supply **declarations only**: a component lifetime graph across stages, the codec tiling law, the block layout and quantum labels. They never supply rung logic. **Accept:** per-family `memory_strategy*` code is reduced to those declarations; a new rung needs one shared layer plus its Feature declaration.

**MEM-2 — Memory contracts are computed from the composed route** and its declarations. **Accept:** the `register_memory_contract_fixture` calls (139 on `a43724b82`) and the hand `Implemented`/`Missing` matches are deleted.

**MEM-3 — Memory geometry carries every axis from day one:**
- area;
- latent frames;
- audio tokens or duration;
- batch;
- reference count **and reference pixel geometry** (vision-language encoders price vision tokens);
- per-phase, per-stream geometry;
- streamed or buffered output.

The rung set is open.

**Peak memory counts what is live together:**
- **Concurrent streams and joint workspaces are summed, never maxed.** Example: H3 video plus audio plus their joint workspace.
- **The request bound covers both neighboring phases and each transition operation,** including the workspace of the component/evaluation actually executing there.
- **Aliases and shared weights are counted once.**
- **Budget domains are kept apart:** host, device, and unified memory (counted once).

Pricing uses per-component weights, existing aggregate phase/operation peak evidence (two anchors plus derivation) and shape-computed additional buffers. Transition operations include their executing component/evaluation workspace and the components actually resident at that operation. Evidence states whether weights, ordinary latents, streams and joint workspace are already included; adding them again is invalid. A transition is not bounded by weights and latent shapes alone. This remains component/phase accounting, not a per-allocation model or a new measurement gate.

**Accept:**
- Low-res start, multi-phase, video and audio-video plans are priced by one mechanism.
- Pure fixtures cover two streams that each fit alone but exceed the budget together, a transition peak above either steady phase, and alias deduplication.
- A learned-transition workspace is priced above latent-only deltas; a joint-total evidence value is not added to its already included stream totals. Ambiguous evidence follows the existing route's unpriced-evidence posture and is reported as such, rather than silently reinterpreted.

**MEM-4 — Equivalent-class Features stay within a declared, recorded quality bound.**
- **Equivalent-class Features are output-preserving within a declared and recorded quality bound.** Tiled decode changes pixels (`gen-core/src/memory_strategy/decode_quality.rs`), so "output-preserving" cannot mean byte-identical here.
- **Admission never selects a recipe-class Feature and never changes the quantization tier**, because tier is a creative choice.

**Accept:** admission decisions contain only equivalent-class selections.

**MEM-5 — The runner drives today's `MemoryRequestScope` lifecycle** (47 implementations; `gen-core/src/memory_strategy/run.rs:191-217`) for every migrated route **until that route's rung is relocated**. Each phase supplies its actual geometry, conditioning/mode and selected rungs under the admitted whole-plan reservation; a final-phase identity is never reused for a different phase. The first geometry-changing feature delivers that integration and conservative pricing for its adopted routes. **Accept:** scope-identity traces pass at each boundary, admitted rungs keep applying after each migration, and the Tier 1 peak-memory check shows no regression. Later admission unification is not a prerequisite for correcting these first-consumer inputs.

**MEM-6 — The engine-side iOS admission and pricing system is relocated onto the shared memory geometry and plan.** That system is `ios_admission.rs`, per-family `ios_pricing/` and the solver quanta checks. Per-quantum OS-headroom checks stay as hooks. **Accept:** no per-family `ios_pricing` module remains; iOS admits from plan bytes.

### 6.7 Engine registry and capabilities (REG)

**REG-1 — A route is a typed composition record.** It holds:
- text encoders, conditioning modules, upsamplers and other components;
- the denoiser binding (`FamilyModel` or `FamilyDriver`);
- the latent codec;
- the adapter surface and install policy;
- the residency lifetime graph;
- the weight plan;
- product limits;
- seed scheme, count and batch semantics;
- **model defaults** (native steps, guidance, schedule, shift, strength).

One call registers the generator, memory contract, fixtures and behaviour. **Accept:** providers register only through route records.

**REG-2 — Capabilities are computed from route records and their primary declarations (AP5).** **Accept:** the hand `Capabilities` literals, the 4,516-line preview scanner and the additivity scanner are deleted; every behavioural declaration has a Tier 0 conformance check.

**REG-3 — There is one versioned capability export per backend.** It is a **hardware-free build product**: a pure function of the linked registry, produced in hosted CI for both backends' feature sets, with no GPU and no off-Mac step.
- **It feeds SceneWorks' UI availability and registry.**
- **The worker's linked engine is authoritative at run time.** On disagreement, the worker refuses with a typed reason.
- **Each pin bump regenerates the export from the pinned engine** and checks it in. A Tier 0 check confirms the checked-in export equals the export generated from the linked registry.

If any part cannot be produced without hardware, the requirement and the critical path must say so. **Accept:** the export is produced by hosted CI; no hand table restates its fields.

**REG-4 — The request contract is a typed core plus typed extension blocks.** Feature parameters live in a feature map. Selectors are enums at the boundary. Id-string gates move into route declarations. **Accept:** the core request has no family-specific top-level field; no shared code compares model ids.

**REG-5 — SceneWorks links the engine only through bundles**, not about 17 individual family crates. **Accept:** SceneWorks' Cargo graph depends on runtime bundles and contract crates only.

**REG-6 — Every engine path is a registered route**, including today's bespoke worker-side providers (for example Kolors IP, which imports `runtime_cuda::providers::*` directly at `kolors_ipadapter.rs:311`). **Accept:** no SceneWorks file imports `runtime_*::providers` directly.

### 6.8 SceneWorks model registry (APPREG)

**APPREG-1 — Each model has one typed `ModelSpec`, built in two layers.**
- **The static layer** is built at build time from the capability export, the slim product manifest and the evidence store.
- **The runtime layer** holds imported models, ComfyUI external bases from disk scans, Iris trained-checkpoint overlays, user overrides and hides, OS, install, decoder-availability and credential projections, and converted-model paths.
- **One merge rule** combines the layers. The plan states where each runtime source lives (API or worker).

**Accept:** web, API, core and worker consume the same projection. Acceptance fixtures cover a builtin model, an imported model, a ComfyUI external base and an Iris trained checkpoint.

**APPREG-2 — No hand table restates registry facts.** An **enumerated, checked-in inventory** lists every table that restates model facts, by path and line. Each entry is classified *derive*, *keep as product data* or *delete*. The inventory is built from [G §1.4, §2.1] and review [APP-15], with about 230 const tables `[M]`. **Accept:** every *derive* or *delete* entry is resolved; the acceptance criterion is that inventory, not an open-ended grep.

**APPREG-3 — Imported and user models use the same registry path as builtins.**

**APPREG-4 — Memory data moves out of the hand-authored manifest** into the evidence store. The evidence store **keeps the existing identity keys verbatim**: no re-keying, and no recapture campaign. **Accept:** the manifest projection script is deleted; the manifest holds product data only; existing evidence resolves unchanged.

**APPREG-5 — Manifest schema v2 has an upgrader for user `*.jsonc` files.** Desktop seeding ships the evidence store. Every consumer of manifest memory keys migrates in the same feature: scripts, Rust readers and the memory-adapter (review [APP-16] lists them). **Accept:** old user manifests load; no consumer reads removed keys.

**APPREG-6 — The job carries a resolved, typed `ModelSpec` snapshot.** This covers non-builtin models and multi-worker deployments such as RunPod. A worker never re-resolves a different spec. Legacy job rows that carry `modelManifestEntry` are upgraded by the upgrader. **Accept:** imported, external and trained models run on a remote worker; retry of an old job row works.

### 6.9 Options, validation and UI (OPT)

**OPT-1 — One option registry (`OptionSpec`) covers every user-settable generation option.** It is seeded from the **union** of:
- `ADVANCED_KEY_RULES` (76);
- `TECHNIQUE_KEYS` (45);
- every production `advanced.get("…")` literal (for example `enhanceTemperature`, `enhanceMaxTokens`, `useUncensoredEnhancer`, `referenceStrength`, `referenceImageShortEdge`, `timestepShift`);
- film-plan `model.advanced` keys.

Each spec carries:
- id and aliases;
- wire location;
- type and shape;
- default;
- range bound to declared limits;
- applicability predicate;
- **declared side effects and exclusions** (for example, enabling Hires.fix or upscale clears PiD, `ImageStudio.jsx:809-815`);
- **intent class:** WhatToMake, MachineBudget, or **Context** (timeline, editor and provenance keys);
- UI metadata;
- engine binding.

**Accept:** no `advanced.*` key read anywhere lacks an `OptionSpec`.

**OPT-2 — Availability is derived from `ModelSpec` and the Feature export.** It never comes from id literals or per-backend gating modules. **Accept:** `macGating.js`, `candleGating.js` and the web id-literal checks are deleted.

**OPT-3 — One validator runs identically at every boundary:** API create, retry and duplicate, MCP tools, and worker preflight. The worker keeps only checks that need local state. **Accept:** each rule exists once; the duplicated constants appear once.

**OPT-4 — The web renders the shared validator's results and implements no rule logic of its own.** ⟂DEC-9 chooses the mechanism. Cold-start and offline behaviour is defined. **Accept:** no rule logic in web modules; validation latency on the RunPod proxy is within the UX bound set by DEC-9.

**OPT-5 — The advanced-option UI is generated from `OptionSpec`s.**
- **Custom renderers are enumerated** for every structured shape: phase editor, poses, layout editors, structured prompt, H3 Veda settings, MOSS options and reports, prompt batch, sample index, number pair, the YuE2 lab, the Ideogram structured caption and the Character builder.
- **Scope covers every studio**, including the Simple studios.
- **Every existing control and behaviour is preserved:** defaults, clamps, persistence keys, emit-when-default rules, tooltips and side effects.
- **Accessibility is at least today's level:** labels, keyboard access and titles.
- **Disclosures default to collapsed.**

**Accept:** PAT-8.

**OPT-6 — Persisted and external request shapes are versioned and upgraded.** The shapes covered are:
- job payloads;
- recipes;
- studio settings;
- share envelopes;
- film plans;
- training plans;
- the public HTTP DTOs and MCP request shapes;
- SceneWorks-Art's consumption of envelopes and recipes.

Rules:
- **A core upgrader maps legacy shapes.**
- **Forward compatibility:** writers emit v1-compatible shares whenever the content is expressible in v1.
- **Unknown higher versions are refused with a typed reason**; rollback behaviour is defined.
- **The upgrade corpus is built from historical builder outputs in git history**, never from user data.

**Accept:** the corpus upgrades losslessly; an old desktop build reads every new share that is expressible in v1.

**OPT-7 — The request pipeline is typed end to end.** The worker receives resolved, typed, defaulted options. **Accept:** zero `advanced.get` reads outside the upgrader, film compile included.

**OPT-8 — Refusals and telemetry are stable product surfaces.**
- **Typed refusals are rendered** in the web UI and in MCP responses. Raw engine strings no longer reach the queue UI.
- **Telemetry field names stay stable.**
- **The stamp key `advanced.memoryStrategy` stays a replay contract** (`regenerateImage.js:33`).

**Accept:** the refusal-render tests pass; existing telemetry consumers are unchanged.

### 6.10 Admission (ADM)

**ADM-1 — One admission pipeline serves every job kind and backend:** image, video, audio, training, upscale and restore, depth, VQA and interleave, and API-side training refusals. **Accept:** no budget probe and no admission refusal exists outside the pipeline and its `BackendBudget` implementations. This is checked by grep over probe and refusal call patterns (`nvidia_vram_budget_gb*`, `total_unified_memory_gb`, `host_available_bytes`, `mlx_tier_budget`, `live_request_budget`, …).

**ADM-2 — Backend specifics sit behind `BackendBudget`:** probe, reclaimable bytes, margins and evict-to-reclaim, with one byte-denominated budget vocabulary. **Failure posture is a property of (backend × evidence basis), declared per route or derived.** It is not a per-backend constant: some MLX routes admit when unpriced today (`instantid.rs:285-288`; SenseNova). **Accept:** only `BackendBudget` implementations contain backend-conditional admission code.

**ADM-3 — Admission consumes the `RenderPlan` and the route's memory contract**, including concurrent live-set pricing (MEM-3) and Feature memory effects. It runs in two phases: Plan (before load) and Run (in-cache, warm/cold-aware). Every feature that first adds geometry, stages or allocations to a route updates its existing admission consumers in that feature; APP-4 later unifies those consumers. An existing estimator can remain unchanged only if a documented bound covers every additional live allocation over the supported settings. The admitted resolved plan is the plan executed; Run rechecks local budget/cache facts and any equivalent-feature reselection revalidates/re-prices the plan. **Accept:** budget-boundary decision tests and phase-scope traces pass when each adopted route first ships; low-res start, multi-phase, video and audio-video plans ultimately use one admission pipeline.

**ADM-4 — Refusals are typed:** binding phase, bytes needed and available, and suggestions. **Accept:** no admission refusal is built as a bare string.

**ADM-5 — One stamp function writes memory decisions, checkpoint facts and receipts.** The `advanced.memoryStrategy` key name stays stable (OPT-8). **Accept:** grep finds a single writer.

**ADM-6 — Every route is admitted.** **Accept:** a test enumerates the route inventory and asserts that every route passes through admission.

**ADM-7 — No previously admitted cell is newly refused.** A cell is a (route, tier, geometry, budget) combination. Exceptions are listed and approved. Every behaviour change (floor-versus-ladder order, probe source, Plan-phase admission of video) carries its measured consequence: refusal delta and refusal latency. **Accept:** the decision-diff over the legacy paths shows zero unapproved new refusals.

**ADM-8 — Probe failure is not a refusal.** An unavailable budget is a typed `BudgetUnavailable`, handled by the route's posture. The cost of probe context creation is stated, for example the cudarc driver context against today's NVML-only preflight.

### 6.11 Worker lane harness (LANE)

**LANE-1 — Every `JobType` is classified.** A checked-in table maps each of the about 33 job types to one of: generation lane, `LaneHooks` lane, or utility (no lane). Each entry names its admission owner and its cache owner. Image, video, audio, VQA and interleave, vector, upscale, restore, extend, bridge and person-replace generation run through **one generation lane**: plan → admit → load → generate → consume → write → receipt. **Accept:** every `JobType` is in the table; the image dispatch function and the per-lane function families (grep-defined patterns, not frozen counts) are deleted.

**LANE-2 — Routing is data**, derived from the registry. **Accept:** adding a route in an existing class requires no worker code.

**LANE-3 — Every engine is cached, residency-managed and preview-wired by construction** (REG-6). **Accept:** the manual pool bookkeeping (`note_loaded_peak`, `gate_with_evict_reclaim`) and the preview source-scan test are deleted.

**LANE-4 — One event consumer serves image, video and audio**, parameterized by output sink and watchdog policy. Coordinate with SW-F327's shared stream supervisor (#3258).

**LANE-5 — The backend split is one module boundary.** Coordinate with SW-F327's cfg alias (#3270).

**LANE-6 — The API→worker dependency reaches a defined end state.** Validation and admission math live in core. The API links no lane code. The in-process worker mode for desktop is preserved. Coordinate with SW-F327 SW-6824. **Accept:** the API's Cargo graph does not include worker lane modules.

### 6.12 Tooling and generated artifacts (TOOL)

**TOOL-1 — Every derived artifact comes from a typed dump.** Nothing regex-parses source or byte-digests source. This specifically covers:
- the memory matrix generator (⟂DEC-17);
- the backend-capability matrix: `routing/matrix.rs`, which digests 61 sources, plus `matrix.json` and `exceptions.json` (⟂DEC-16).

**Accept:** each artifact is deleted or regenerated from typed dumps per its decision.

**TOOL-2 — The memory-adapter drives routes through the shared runner and registry**, with no per-family arms. That covers `bin/mlx.rs`, `bin/candle.rs` and the auxiliary bins (`mlx_ltx25`, `flux3_*`, `*_wan_scail2`, `*_iris`, `*_supra2`). Workflow consumers (`memory-catalog-campaign.yml`) are migrated. Proof of the rewrite follows ⟂DEC-18. **Accept:** no per-family arm remains.

**TOOL-3 — Hand mirrors are deleted:** the web `modelCatalog.js` seed (being deleted by SW-F327 #3268), `tierSuggestion.js`, and duplicated vocabularies.

**TOOL-4 — No new memory or measurement gates** (FEATURE_DEVELOPMENT.md *Gate teardown*). Structural checks follow ⟂DEC-6. If they are rejected, each affected acceptance criterion is checked once by a grep recorded in the feature's terminal item.

### 6.13 Cross-lane and external adoption (XL)

**XL-1 — Audio adopts** the weight plan, the Linear spine, adapters (ordered fold for DoRA and BoRA), runtime contracts and the residency vocabulary (DEC-2 approved). It adopts the runner wherever a flow or diffusion loop exists; autoregressive audio retains its token-generation driver.

**XL-2 — LLM runtime contracts unify with gen-core:** cancel, progress, load activity, weight source and pins.

**XL-3 — iOS stays feature-partitioned:** no CUDA codecs, no training, no video in the iOS graph. An iOS binary-size and resident-memory baseline is captured before the first engine feature. **Accept:** within PERF-2.

**XL-4 — External consumers keep working.**
- **ChatWorks** pins inference runtime tags (for example `runtime-2026.10.0`); each feature that changes the candle or LLM public surface states its tag compatibility.
- **SceneWorks-Art** relies on envelope and recipe compatibility (OPT-6).
- **SceneWorks-mobile** will consume `runtime-ios`; its surface is kept stable.

**Accept:** each affected feature records consumer compatibility.

### 6.14 Migration and quality (MIG)

**MIG-1 — Every program feature reaches `main` within the window set by ⟂DEC-12.** There is no multi-month integration branch.

**MIG-2 — No cross-feature dual paths (AP9).** **Accept:** at each feature's end, grep finds no legacy path for the migrated routes, and no interim adapter outlives its feature.

**MIG-3 — Validation runs in three tiers.**

**Tier 0: ordinary PR CI.** It never runs a model forward, encode, decode or training step, even with tiny or synthetic weights (standing rule `real_model_execution_is_not_normal_ci`). It covers:
- plan snapshots;
- weight-plan and bind receipts (header-only);
- runner traces over a mocked backend;
- `FamilyDriver` and Feature conformance over mocked forwards;
- primitive op checks;
- admission and routing decision-diffs;
- receipt shape checks;
- recorded engine-request equivalence.

Pure admission and routing logic lives cfg-free in `sceneworks-core`, so it runs on Linux CI.

**Tier 1: the dispatched equivalence workflow, with recorded evidence.**
- The base commit and the head commit are built and run **on the same runner, in the same job**.
- They run on **shape-faithful random weights**: real dimensions, truncated depth. Tiny weights do not exercise quantized kernel paths (`candle-llm/src/primitives/quant.rs:8`).
- The comparison is in-job. No goldens are committed (standing rule `never_commit_a_machine_dependent_value_in_a_golden`).
- MLX runs on `nax-macos-2` through an `rw-*` label (the workflow needs that route added); candle runs on a CUDA runner labelled `real-weights`.

**Tier 2: real-weight proof on the proof set.** Run at feature end. A/B per ⟂DEC-15.

**Accept:** every consolidation PR shows Tier 0 green before and after. Tier 1 runs at the cadence set by ⟂DEC-14. Tier 2 follows ⟂DEC-15.

**MIG-4 — The proof set covers each latent layout, loop kind, guidance kind, stream kind and modality.** It is listed in the implementation plan §8. **Accept:** recorded evidence for every family whose code changed.

**MIG-5 — External contracts stay stable (AP12, XL-4).** **Accept:** the compatibility corpus test passes.

**MIG-6 — No deferment.** Gaps found during a feature are fixed in that feature, or surfaced for a decision and tracked [H §7.19].

**MIG-7 — Live work is coordinated.** Each feature lists its overlap with in-flight work: SW-F327 cards and PRs, and new model epics. Each overlap is marked *delivered elsewhere*, *absorbed with sign-off* or *conflicting, so sequenced*. Pins are choreographed: whichever inference feature lands first, the other merges inference `main` before its final pin, then runs `cargo metadata --locked`. **Accept:** the overlap ledger exists and is current at feature planning.

### 6.15 Performance (PERF)

**PERF-1 — No speed regression.** Seconds per step and end-to-end time for every proof-set route stay within ±2% per backend, or within measured run-to-run noise if that is larger. This is measured **A/B on the same runner in the same window**: pre-feature `main` against the feature head. It is never measured against a months-old baseline.

**PERF-2 — No memory or size regression.** Peak resident memory and iOS binary size stay within +2% of the A/B baseline for the proof set.

**PERF-3 — Features deliver their stated benefit.** For low-res start, the speedup at K = N/2 and scale 0.5 is measured on the proof set. The expected speedup is about 1.5–1.7× `[L]`.

### 6.16 First consumer: low-res start (FEAT)

**FEAT-1 — "Low-res start" is an opt-in recipe-class Feature.** The mechanics are recipe v1, in IMPLEMENTATION_PLAN §6.1 and §3.1 (⟂DEC-10).

- **Behaviour:**
  - It runs the first K executed steps of the first denoising pass at reduced spatial resolution.
  - It then converts a clean estimate to full resolution and re-noises it to that pass's boundary σ.
  - Composed with Hires.fix, it applies once, to the base pass only.
- **Parameters:**
  - K must satisfy `1 ≤ K < M`, where M is the number of steps the pass actually executes.
  - Scale reduces spatial dimensions only. Frames, audio and count are unchanged.
  - Invalid values refuse; they are never clamped.
  - With the Feature absent, the output is byte-identical to today.
- **Availability:**
  - It is offered on every route whose targeted pass supports the required resize, estimator and transition.
  - Every other route refuses it with a typed reason.
  - Initially, edit, reference and control inputs on the targeted pass refuse, until COMP-5 rebuilds aligned conditioning. Ordinary img2img stays supported.
  - Unavoidable exceptions are listed under DEC-13.
  - The expected-route list comes from the route inventory, so availability cannot pass vacuously.
- **UI:** a number field for K; a linked slider showing K/M; a scale control; the resolved dimensions shown.
- **Replay:** the resolved recipe is persisted and versioned. Replay with the same seed is deterministic per backend, and unknown recipe versions refuse.
- **Estimator choice:** endpoint evaluation remains the default. The optional previous-estimate candidate completes all K low-res steps with their normal solver evaluations, state updates and events, then uses a specifically identified earlier clean estimate. It does not skip a solver integration. Solver support and estimator identity/provenance are explicit and persisted; a quality comparison cannot silently change replay or admission's recipe selection.

**Accept:**
- Correct plans and refusals at the parameter boundaries, under img2img truncation, on legal-grid limits, with spatial-only scaling on AV routes, and under composition.
- Correct output size on every expected route.
- Deterministic replay.
- Default and candidate traces preserve the same K completed low-res steps, solver sub-evaluations, noise-draw coordinates and outer progress. Unsupported estimator/solver combinations refuse explicitly without removing the default estimator's supported routes.
- A bounded prompt/seed comparison against full resolution, per backend, checking prompt and content retention, structure, blur, seams, non-finite values and img2img retention. An unusable result blocks that route until it is repaired, or decided under DEC-13. There is no bit-identity requirement and no Cartesian campaign.
- Speedup measured per PERF-3, including estimator and transition costs. A benefit is claimed only where measurement and quality support it.

---

## 7. Program acceptance tests (PAT)

| ID | Test |
|---|---|
| **PAT-1** | **One-change feature.** After APP-2, a **new** recipe Feature is added, for example a new guidance operator (implementation plan §6.4). Its diff touches no family crate and no SceneWorks lane, route, admission or web-gating file. That diff is the measurement. (Low-res start ships earlier, under FEAT-1, and is not the measure.) |
| **PAT-2** | **One-change rung.** A new equivalent-class memory measure is added by changing one shared layer plus its Feature declaration. Admission prices and selects it on every route that uses that layer. No family or SceneWorks edit is needed. |
| **PAT-3** | **New model is data plus graph.** A new model in an existing class is added with only these inputs: config, graph, weight plan, route record, product-manifest entry, its measured anchors as data (two anchors per tier, per standing rule `ios_evidence_is_two_anchors_plus_derivation_not_a_grid`), and fixtures. No other code edit is needed. It appears in the UI with correct options. Any other input found (LoRA signatures, converter entries, prompt guides) must be derived, or listed here by decision. |
| **PAT-4** | **Single encoding.** The fact audit (implementation plan §10.2) shows each listed fact written as a literal in exactly one place, except the enumerated allowed exceptions (product menus, upgrader alias tables). |
| **PAT-5** | **No model ids in shared conditionals.** No builtin model-id literal appears in a conditional in production code outside the manifest, route records, tests and an enumerated exemption list. |
| **PAT-6** | **Fidelity.** Every proof-set route meets its declared AP13 class, per backend. |
| **PAT-7** | **Performance.** PERF-1 and PERF-2 hold, measured A/B. |
| **PAT-8** | **Controls preserved.** The control-inventory tool (implementation plan §10.3) shows zero removed controls and zero changed per-control behaviour fields, per (model, backend, mode, studio), between before and after. |
| **PAT-9** | **Admission coverage.** ADM-1's grep check and ADM-6's enumeration both pass. |
| **PAT-10** | **Census recorded.** The LOC census (§8) is recorded at every feature end. It is reported, never gating. |

---

## 8. Success metrics and targets

These are measured at baseline and again at every feature end. They are **goals, not gates** (AP11). Targets were re-derived after review [APP-37].

| Metric | Baseline | Target |
|---|---|---|
| Files touched by a new cross-cutting generation feature (PAT-1) | about 50–100+ estimated `[L]` | ≤ about 10; zero family files |
| Admission code paths | ≥25 `[M]` | 1 pipeline |
| `*Linear` types (engine) | 68 `[M]` | 2 (1 if DEC-1 (c)) |
| VAE implementations (engine, both backends) | about 30 `[M]` | about 8–10 |
| Text-encoder block implementations (media) | about 19 Qwen-family, Gemma ×4, T5/CLIP/UMT5 copies `[M]` | 1 per architecture per backend, in llm |
| Per-engine `memory_strategy*` lines (tests included) | about 125k `[M]` | under 20k |
| Inventoried hand tables that restate model facts (APPREG-2) | about 20 named + about 230 const tables `[M]` | 0 of the *derive*/*delete* class |
| Model-id literals in shared conditionals | about 2,000 lines `[M]` | about 0 outside exemptions |
| Engine media + audio + llm source lines | 1.43M src / 2.03M total `[M]` | −20% to −35% with DEC-1 (b); −35% to −50% with (c) `[L]` |
| SceneWorks worker + core + API + memory-adapter + scripts lines | about 950k including tests `[M]` | **−15% to −25%** `[L]` (bottom-up: routing −9–14k, `models.rs` −6–7.5k, drift tooling −10–15k, memory-adapter about −47k, worker lanes and admission −60–100k; characterization tests add some back) |

---

## 9. Constraints (process and environment)

[H §1, §7; review APP]

1. **Tracker:**
   - Strict requirements (E1…En) and acceptance tests at the feature level.
   - 1–3 local acceptance criteria per item.
   - Walking skeleton first, a terminal item last.
   - One review and one fix pass per item.
   - Whole-branch review of at most two rounds before feature → `main`.
2. **Branches:**
   - Mirrored `feature/*` branches in both repos.
   - Merge-commit only, `strict: false`, no merge queue.
   - The inference branch policy must accept CodeGraph keys before the first feature (⟂DEC-11).
3. **Pin:**
   - At least 8 sites plus `Cargo.lock`. Constraint 3 in [H] supersedes the "five sites" figure in [G §7.2].
   - Coalesce to one pin bump per feature. Never a pin-only PR.
   - The final pin is on inference `main`.
   - Run `cargo metadata --locked` after every lock merge.
4. **No model execution in PR or `main` CI** (MIG-3 Tier 0).
5. **Hardware:** discover the live runner pool at each feature's planning. Query both registration scopes and the runner groups. Record registered, label-compatible and idle counts separately. A snapshot from 2026-11-03:
   - All runners are in org group `self-hosted-gpu`.
   - There are 4 `cuda-windows*` listeners, likely one host; only 2 carry `real-weights`.
   - `nax-macos` is the dev Mac (Michael's GPU, used only on an explicit loan). `nax-macos-2` takes macOS real-weight runs.
   - Never `pkill` MLX mid-buffer.
6. **Agents:**
   - 8–10 subagents at most.
   - 3–4 SceneWorks agents building at once.
   - Clean finished `target/` directories.
   - One coordinator owns publication.
   - Docker only with an explicit free-space check.
7. **Pre-push checks:**
   - `rust:check`, `rust:check:candle`, `rust:check:neither`;
   - inference `check-workspace.py`;
   - fmt and clippy with `-D warnings`.
   - **Candle tests:** feature CI only typechecks candle (`check.yml:470-473`), and `windows-candle.yml` runs only on PRs to `main`. So one `windows-candle.yml` dispatch per SceneWorks batch that moves candle-gated code is budgeted.
8. **SW-F327 coordination** (⟂DEC-5; implementation plan §7.5 overlap ledger).
9. **In-flight stale branches need dispositions** before any overlapping work starts: `sc-19048`, `sc-20414`, `sc-25327`, and SW-4802…4821 [H §4].

---

## 10. Decisions and approval status

**DEC-1…3 approved by Michael on 2026-11-03. DEC-4 reuse strategy is approved; its chained-pass product scope and DEC-5…19 remain open.** Approvals here select the design direction; they do not approve the whole design or authorize implementation during this design-only discussion.

| ID | Decision | Options | Recommendation / decision (confidence) | Affects |
|---|---|---|---|---|
| **DEC-1 — Approved** | **Backend strategy.** The conditional generic-model path reverses the former non-goal only within the pilot gates (`inference-rearchitecture.md:81`). | (a) status quo; (b) shared host math + identical layer-kit API; (c) generic model graphs over a layer-level trait; (d) Burn; (f) Candle on Mac | **Michael approved (b) as the baseline, with (c) conditional on both pilots passing (2026-11-03).** Pilot A: a small representative model. Pilot B: FLUX.3 end to end, including LoRA training. Preserve capabilities and per-backend fidelity, speed, memory and binary-size bounds. If either fails, stay at (b). Backend kernels/execution policy remain specialized; shared pipeline and low-res work are not blocked by generic rollout. No framework/backend replacement selected. Design-only, not permission to execute pilots. (High confidence in the approved choice/baseline; medium in unproven generic feasibility.) | BACK-1…5, G5, §8 |
| **DEC-2 — Approved** | **Lane scope.** | Include audio, LLM-as-encoder and training immediately; staged adoption; exclude | **Michael approved all three as program deliverables (2026-11-03): design shared interfaces from the start and stage adoption around the image/video core.** Share weights, layers, adapters, encoders and lifecycle contracts. Diffusion/flow audio uses the runner; autoregressive audio retains its token driver. Training shares differentiable model modules and retains optimizer/loss/data orchestration. No new chat-product scope. Early primitive consumers and broader adoption remain named roadmap obligations. Design-only. (Medium-high.) | XL-1/2, WGT-4, COMP-1; ENG-4a/8/11/14 |
| **DEC-3 — Approved** | **Schedule source kinds.** | Global-sliced; fresh-per-pass; explicit table; all three | **Michael approved all three (2026-11-03), resolved into one immutable schedule representation before execution.** Each pass owns its schedule and step numbering; phases select executed outer-step ranges within it. LTX needs explicit tables; Krea and low-res need slicing; chained recipes need fresh passes. Transitions, re-noise and solver-history behavior remain independently explicit. Design-only. (High.) | PIPE-1/3/4, HOOK-3 |
| **DEC-4 — Reuse approved; product scope open** | **sc-20414 prior art** (4,212 commits behind at the review baseline). | Port the executor; selectively reuse parts; abandon | **Michael approved selective reuse (2026-11-03): scheduling helpers, conformance tests and execution safeguards; build the runner around the approved pass/phase architecture.** Check extracted pieces against current contracts, including error cleanup and progress policy. The old request schema informs behavior/compatibility analysis; the shared option system owns any resulting schema. **Still open:** include user-configurable chained denoise passes as a fresh-per-pass Feature after APP-2 (recommend yes). Review `sc-19048` and `sc-25327` for salvage before overlapping work. Design-only. (High in reuse strategy; medium in unchanged code reuse.) | PIPE-3/4; chained-pass scope pending |
| **DEC-5** | **SW-F327 coordination.** | Program waits for SW-F327; engine-only work now with SceneWorks halves gated; full parallel | **Inference-only (`[INF]`) items start now, behind unchanged worker-facing APIs. Every SceneWorks (`[SW]`) half waits for the SW-F327 PRs on the same files to merge.** Those PRs today are #3258, #3260, #3264, #3268 and #3270. Cards already in review land first; SGA re-baselines after them. No absorbing in-review cards. (High.) | MIG-7, all `[SW]` items |
| **DEC-6** | **Structural guards:** static checks such as no raw SDPA outside the attention op, no per-family σ functions, no guidance combine in family files, no model-id literals in shared code. | Allow, inside the existing `architecture_tests.rs` (no new CI lane); forbid | **Allow.** On 2026-08-15 you banned "a static-analysis check · a lint rule · … an anti-drift guard". On 2026-10-11 you scoped that ban to memory and measurement gates, and said gates that prevent CI failures are wanted. You approved AST architecture checks for SW-F327 (E3, #3270 SW-6686). On 2026-10-23 you asked for a structural CI policy check. Without guards, consolidations regrew within 11 days. **If you forbid them,** each affected criterion is checked once by a grep recorded in the feature's terminal item. (Medium; this one is genuinely yours.) | PIPE-8/9, BACK-2, APPREG-2, PAT-5, TOOL-4 |
| **DEC-7** | **Numerics classes (AP13).** | Bit-exact only; (a)/(b)/(c) classes | **(a)/(b)/(c).** Class (c) needs your sign-off per route. Each feature lists its expected (b) and (c) entries at planning. (High.) | PIPE-10, COMP-1/4, WGT-3 |
| **DEC-8** | **Feature classes.** | Reuse the "approximation" regime; add a recipe class | **Recipe / equivalent (within a declared quality bound) / diagnostic.** Admission selects only equivalent-class Features. (High.) | HOOK-2, MEM-4 |
| **DEC-9** | **Web validation mechanism.** | Server endpoint per change; core validator compiled to WebAssembly and run in the browser; hybrid (availability cached per (model, backend, mode) plus server round-trip for cross-field rules) | **WebAssembly-compiled core validator.** One Rust implementation, instant, works offline and over the RunPod proxy. Cost: wasm build and bundle size. **Fallback:** hybrid. (Medium.) | OPT-4 |
| **DEC-10** | **Low-res start recipe v1** (mechanics: PLAN §6.1, §3.1). | Coverage and modes; step domain; scale and grid policy; estimator; schedule policy; quality acceptance | **Coverage:** every structurally supporting route; noise-start plus img2img. Edit, reference and control refuse on the targeted pass until COMP-5. **Step domain:** the first denoising pass, once; `1 ≤ K < M` executed steps; default `max(1, floor(M/2))`. **Scale:** 0.25–0.9, default 0.5, floor-to-quantum, refuse when invalid. **Schedule:** full-pass schedule and `mu` kept. **Estimator:** `EndpointEvaluation` default; `PreviousEvaluation` is a candidate that completes K steps normally and retains the last step's named guided primary x0. It skips no integration/sub-evaluation; it adds no endpoint forward. Require solver support and bounded quality/time/memory comparison before recipe finalization, with no percentage saving assumed. **Fixed:** estimator identity and semantics are persisted/versioned; no runtime/admission estimator switch; only K/scale user controls. **Quality:** FEAT-1's bounded checks and replay. Proposed, not approved by this revision. (Medium on quality/benefit; high on explicit semantics.) | FEAT-1, PIPE-2/4/5, ADM-3 |
| **DEC-11** | **Tracker naming in inference CI.** | Land the CodeGraph-key policy fix first; use `sc-` names | **Policy fix first, as a prerequisite PR.** (High.) | MIG-1 |
| **DEC-12** | **Branch lifetime window.** | ≤4, ≤6 or ≤8 weeks | **≤6 weeks, with a re-plan if it is exceeded.** (Medium.) | MIG-1 |
| **DEC-13** | **Routes that cannot change geometry or adapters per phase** (Mage watermark, merge-only adapters, fixed-grid planners), plus any driver refusal under PIPE-7. | Refuse; change product behaviour | **Refuse with a typed reason.** Each refusal is listed for your approval when its feature is planned. (High.) | HOOK-4, FEAT-1, PIPE-7 |
| **DEC-14** | **Tier 1 cadence.** This is an explicit exception to FEATURE_DEVELOPMENT.md's "no per-story validation matrix". | Feature end only; per migration batch; per PR | **Per migration batch, on `nax-macos-2` and a CUDA `real-weights` runner, about 1–3 runner-hours per batch.** Refactors touching numerics on about 40 families × 2 backends cannot wait for the feature end. (Medium-high.) | MIG-3 |
| **DEC-15** | **Tier 2 for refactor-only features.** SW-F327 E7 says "no campaign solely for refactoring". | None; A/B proof subset per feature | **A/B on the proof-set routes the feature touched, same runner, same window.** Not a campaign. (Medium.) | MIG-3, PERF-1/2 |
| **DEC-16** | **Backend-capability matrix** (`routing/matrix.rs` digesting 61 sources, `matrix.json`, `exceptions.json`). | Delete; regenerate from `ModelSpec` without source digests | **Delete at the start of APP-1.** `ModelSpec` plus the export make it redundant, and it would fire on almost every APP PR. (Medium-high.) | TOOL-1 |
| **DEC-17** | **Memory matrix generator** (255k lines, regex over 19 Rust files). | Delete; regenerate from typed dumps | **Delete.** The evidence store and admission replace its purpose (AP11). (Medium.) | TOOL-1 |
| **DEC-18** | **Proof for the memory-adapter rewrite** (real model execution). | One dispatched old-vs-new run on `nax-macos-2` and CUDA, duration stated; no proof | **One dispatched equivalence run, duration stated up front.** (Medium.) | TOOL-2 |
| **DEC-19** | **Progress unit.** | Outer steps (today); model evaluations (sc-20414) | **Outer steps, user-visible. Evaluations stay internal.** (High.) | PIPE-6 |

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Route** | One executable generation path: model, mode and conditioning shape on one backend, identified by engine id plus route key. |
| **Route inventory** | The generated table of every (family, route, backend), with loop kind, owning feature and AP13 class (PIPE-18). |
| **Family** | A model architecture lineage. After the program, it owns config, graph and primary structural declarations only. |
| **RenderPlan** | The typed, weights-free compilation of a request. |
| **Pass / schedule domain** | One ordered denoising invocation with stable identity, resolved per-stream schedules and stage bindings; later passes may consume earlier outputs. |
| **Stream** | One latent of a bundle (video, audio, joint rows), with its own space, geometry, schedule and σ. |
| **Phase / span / transition** | A pass-local executed outer-step range with overrides; its half-open coordinate range; and the typed boundary-state operation that enters it. Global, fresh and explicit describe the enclosing pass's schedule source. |
| **BoundaryState / execution state** | A noisy latent, provenanced clean estimate or declared route state; and the separately request-owned solver history, RNG coordinates, driver caches and preparation state that survive slice calls. |
| **FeatureSpec / FeatureHooks** | A Feature's tensor-free declaration, and its optional per-backend model-internal hooks. |
| **Recipe / equivalent / diagnostic** | Feature classes: result-changing and user-chosen; within a declared quality bound and selectable by admission; observation only. |
| **FamilyDriver** | The typed escape hatch for essential bespoke loops, verified by conformance tests over a forward seam. |
| **AP13 class (a)/(b)/(c)** | Bit-exact; component-tolerance; accepted re-baseline with sign-off. |
| **Tier 0/1/2** | PR CI with no model execution; dispatched same-runner base-versus-head equivalence on shape-faithful weights; real-weight proof set. |
| **ModelSpec (static and runtime layers)** | SceneWorks' typed model record. The static layer is built at build time from the export, manifest and evidence store. The runtime layer adds imports, scans, overlays, user data and projections. |
| **OptionSpec** | SceneWorks' single declaration of a user-settable option. |
| **`[INF]` / `[SW]`** | The inference half and the SceneWorks half of a feature item. |
