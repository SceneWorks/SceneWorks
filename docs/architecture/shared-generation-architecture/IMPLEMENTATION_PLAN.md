# Shared Generation Architecture: Implementation Plan

| | |
|---|---|
| **Status** | **Draft 0.7, pre-planning revision.** DEC-1…19 retain their November 3 approval. This roadmap incorporates the five blockers and twelve major findings in round 7. The [closure review](evidence/reviews/round8-codex.md) separates design verification from future product acceptance. |
| **Date** | 2026-11-07 |
| **Implements** | [REQUIREMENTS.md](REQUIREMENTS.md). Requirement IDs and decisions (`DEC-n`) refer to that document. |
| **Evidence** | [A–H research](evidence/) uses SceneWorks `a1d95e2c4` and inference `a43724b82`. [Round 7](evidence/reviews/round7-codex.md) and the revised walkthroughs use SceneWorks `e1ffa54df6` and its inference pin `2b09938e33`. |
| **Code sketches** | Rust and JSON blocks are **sketches**. Names and shapes are illustrative; they are not final APIs. |

---

## Contents

1. [Target architecture](#1-target-architecture)
2. [Reuse inventory](#2-reuse-inventory)
3. [Engine design](#3-engine-design)
4. [SceneWorks design](#4-sceneworks-design)
5. [The hook map](#5-the-hook-map)
6. [Worked examples (files touched)](#6-worked-examples-files-touched)
7. [Migration strategy and coordination](#7-migration-strategy-and-coordination)
8. [Validation and proof plan](#8-validation-and-proof-plan)
9. [Roadmap](#9-roadmap)
10. [Metrics, baselines and tools](#10-metrics-baselines-and-tools)
11. [Risks](#11-risks)
12. [Reviewer checklist](#12-reviewer-checklist)

---

## 1. Target architecture

**Approved DEC-2:** audio, LLM-based encoders and training are all in scope. Shared interfaces accommodate them from the start; adoption is staged around the image/video core. ENG-4a includes media/LLM projections and preserves trainable variants, ENG-8 delivers differentiable primitives and removes training graph copies, ENG-11 consolidates encoders, and ENG-14 completes audio/LLM runtime adoption. Diffusion/flow audio uses the denoising runner; autoregressive audio keeps its token driver. Training shares model modules while retaining optimizer/loss/data orchestration. Existing platform partitions remain. This is design sequencing, not authorization to implement.

### 1.1 Layers and dependency direction

```text
INFERENCE REPO
┌────────────────────────────────────────────────────────────────────────────────────┐
│ L0  CONTRACTS (tensor-free; zero tensor deps — unchanged rule)                     │
│     gen-core:  Request core + typed extension blocks · RenderPlan (streams, phases, │
│                passes, typed boundary states, seeds, count/batch) · FeatureSpec ·  │
│                RouteStructure · runner policy (generic over backend op traits) ·   │
│                sampling (solvers, named schedules, guidance CombineVariants) ·     │
│                memory math (rungs, selection, per-phase/per-stream geometry) ·     │
│                WeightPlan + codec registry · RouteRecord + capability computation │
│                · runtime sinks + error taxonomy (shared with core-llm)            │
│     gen-core-<family>: host math only (configs, ids, packers, family schedules)   │
├────────────────────────────────────────────────────────────────────────────────────┤
│ L1  BACKEND PRIMITIVES (mlx-layers · candle-layers; proposed names)                │
│     Layer kit: Linear spine · AttentionOp · Norm · Conv(layout) · Rope · Embed      │
│     ExecPolicy (materialize · fuse/compile · memory scope · cancel · stream)        │
│     LatentOps + LatentGeometryOps (bundle-aware resample, pack/unpack)              │
│     WeightPlan executor + codecs · adapter runtime · residency rt · tiling rt      │
│     BlockStack driver · FeatureHooks<B> contracts                                 │
├────────────────────────────────────────────────────────────────────────────────────┤
│ L2  SHARED COMPONENTS + MEDIA/LLM RUNTIMES (spec in L0; impl per backend)           │
│     encoders (llm CausalLm + tap API extension; towers) · latent codecs · learned  │
│     upsamplers · conditioning modules · mlx-gen/candle-gen orchestration bindings  │
├────────────────────────────────────────────────────────────────────────────────────┤
│ L3  FAMILIES: config + graph + FamilyModel (or FamilyDriver) + primary structural   │
│               declarations + RouteRecord(s) + WeightPlan data                      │
├────────────────────────────────────────────────────────────────────────────────────┤
│ L4  CATALOGS & BUNDLES: register RouteRecords + Features · capability export        │
│     (hardware-free build product, REG-3) · route inventory dump (PIPE-18)          │
└────────────────────────────────────────────────────────────────────────────────────┘
                                   │ bundles only (REG-5)
SCENEWORKS                         ▼
  core   : registry::ModelSpec (static layer ⊕ runtime layer) · OptionSpec + validate()
           (pure rules also compiled to wasm for web, DEC-9) · request upgrader · admission core
           (cfg-free, pure) · routing table
  worker : generation lane + LaneHooks · BackendBudget impls · one event consumer ·
           backend module (mlx | cuda | none)
  api    : runtime-layer sources (imports, scans, overlays) · serves ModelSpec ·
           validate() on every path · links no lane code (LANE-6)
  web    : generated option panels + enumerated custom renderers · renders Issues/refusals
  tooling: typed dumps only; memory-adapter = one generic driver
```

**Dependency rules.**
- **Kept:** contracts stay tensor-free; no cross-backend edges; nothing depends upward; the graph is acyclic (`scripts/check-workspace.py`).
- **Added**, protected by narrow structural checks under approved DEC-6 as each consolidation lands:
  - Family crates may depend on L0, L1 and L2. They define no layer-kit types, guidance combines, σ builders or encoder blocks.
  - Shared SceneWorks code has no model-id literals in conditionals.
  - SceneWorks links bundles and contracts only.
  - No SceneWorks file imports `runtime_*::providers`.

**Approved DEC-6 enforcement:** prefer Rust visibility/types and Cargo dependency boundaries. Add focused syntax checks only where those leave gaps, extending the existing architecture test suite and CI lanes. Scope each check to its agreed ownership rule, with documented essential backend/native exceptions and actionable diagnostics. Prove a known violation is rejected and legitimate variants pass. No model execution, new CI lane, source-hash snapshot, evidence-currency assertion or formatting-sensitive check is introduced; memory/measurement gates remain prohibited. These checks protect source structure, while numerical/runtime evidence proves behavior. No implementation is authorized during design review.

### 1.2 Current → target crate map (engine)

| Today | Target |
|---|---|
| `gen-core` (contracts, plus some family data [A P16]) | `gen-core`, contracts only. Family data moves to `gen-core-<family>` or is computed from codec and route declarations. |
| `gen-core-family`, `gen-core-{flux3,iris,ltx,h3}`, `gen-core-memory-wan` | `gen-core-<family>` holds host math. `gen-core-memory-wan` folds into computed contracts (MEM-2). |
| `mlx-gen/src`, `candle-gen/candle-gen/src` | Extract L1 `mlx-layers`/`candle-layers` below media and LLM consumers: `layers/`, `exec/`, `latent/`, weight execution, adapters and residency primitives. Media orchestration bindings remain in `mlx-gen`/`candle-gen` and consume L1. `BlockStack`/tiling primitives join L1 in their owning features; component implementations sit above L1. |
| 39 family twins | Same crates, holding config, graph, `FamilyModel`/`FamilyDriver`, declarations and `RouteRecord`. Under DEC-1 (c), each collapses to one `gen-<f>` crate. |
| `mlx-llm`, `candle-llm` | Owners of decoder-LM blocks and the encoder tap API. They use the shared Linear spine. |
| `mlx-gen-ios-catalog` (incl. `ios_admission.rs`, per-family `ios_pricing/`) | iOS admission computed from plan bytes and shared memory geometry (MEM-6). Quantum checks remain as hooks. |
| `*-catalog`, `bundles/runtime-*` | Register RouteRecords and Features and emit descriptor-only product schemas. The coverage check reconciles this surface with independently callable product/driver rows under PIPE-18 (§6.11). |
| `candle-audio-*` | Adopt the weights, Linear and adapter layers and the runtime contracts. Use the runner where a flow loop exists (XL-1). |

**Concrete crate direction (R2-08).** `mlx-layers → {mlx-rs, tensor-free contracts}`; `mlx-llm → mlx-layers`; `mlx-gen → {mlx-layers, mlx-llm, contracts}`; media family/component crates consume those lower layers. Candle mirrors this direction with its own backend only. There is no `*-layers → *-gen` or `*-layers → *-llm` edge. Media's existing text-sampling dependency on `mlx-llm` remains legal. Error and runtime-sink types needed below both owners live in the existing tensor-free contract layer, not a media runtime. L1 has no catalog, model or training-application dependency. Platform-specific codecs, kernels and training primitives remain feature-partitioned so `runtime-ios` does not acquire CUDA/training/video. ENG-4a extracts the Linear/adapter primitive owner and migrates both media and LLM projections; later features extend that same owner. Validate `cargo metadata --locked` and `check-workspace.py` when implemented. Names here are illustrative; the dependency direction is the contract.

---

## 2. Reuse inventory

The design grows existing seams rather than inventing new ones.

| Existing asset | Where | Becomes |
|---|---|---|
| **sc-20414 executor invariants and testkit** | `feature/sc-20414`: `gen-core/src/sampling/pass_executor.rs` (844 lines), `gen-core-testkit/src/denoise_passes.rs` | **Selective reuse approved under DEC-4; extraction remains unimplemented.** Reuse candidates: monotone progress, cancellation at every evaluation/boundary, cleanup guarantees adapted to §3.3's error policy, domain-separated re-noise seeds, backend-neutral conformance tests, and ComfyUI pass-schedule math (`denoise_pass_schedule_steps`, `terminal_pass_segment`). Check each against current contracts. The executor is fresh-per-pass only, always re-noises (`:274`), has no geometry, leaves combine to the host (`:36,143`), and refuses Krea phases (`denoise_passes.rs:433-435`), so it is not the runner base. The old wire schema informs behavior/compatibility analysis; the shared option system owns any resulting schema. [ENG-1] |
| Krea multiphase host math | `gen-core-family/src/krea_multiphase.rs` | Generic global-sliced phase math in gen-core |
| H3 `JointVelocity` loop seam | `mlx-gen-minimax-h3/src/denoise/mod.rs:25-29` | The pattern for `FamilyDriver`s that are generic over a forward seam (PIPE-7 conformance) |
| candle `AvLatents` (two-stream ops) | `candle-gen/src/sampler.rs:374,420-468` | The precedent for `LatentBundle` |
| `Residency::run_staged_*` | `gen-core/src/residency.rs:623` | R1 inside the runner stage driver |
| `MemoryRequestScope` (47 implementations) | `gen-core/src/memory_strategy/run.rs:191-217` | Driven by the runner until each rung is relocated (MEM-5) |
| `run_windowed`, `BlockWindowBackend` | `gen-core/src/block_window.rs:152,194` | R4 path of `BlockStack` |
| Tiling planner and backend tiling; decode-quality records | `gen-core/src/tiling.rs`, `mlx-gen/src/vae_tiling.rs`, `gen-core/src/memory_strategy/decode_quality.rs` | R2 inside `LatentCodec`, with quality bounds (MEM-4) |
| `attention_budget`, `sdpa_budgeted_*` | gen-core and backend `attention.rs` | `AttentionOp` |
| `LatentSpace` | `gen-core/src/latent.rs:144` | Per-stream geometry/layout key |
| `LogicalKeyMapping` → `LogicalWeightPlan`, `CheckpointCodecRegistry` | `gen-core/src/checkpoint_codec.rs` | `WeightPlan` |
| MLX `AdaptableLinear`, `AdaptableHost`, adapter loaders | `mlx-gen/src/adapters.rs:604,1307`, `adapters/loader/` | Template for the Linear spine and the adapter runtime on both backends |
| MLX `qkv.rs` | `mlx-gen/src/qkv.rs:706,1099` | Attention prologue in the kit, ported to candle |
| **llm `CausalLm` + text-encoder tap API** | `mlx-llm/src/models/llama.rs:3-8,1081-1135`; `candle-llm/src/models/llama.rs:1289-1391`; already used by flux3, iris, lens, sensenova | Encoder library. `TapPolicy` extends the existing tap API [ENG-20]. |
| `text_decoder::Qwen3TextDecoderLayer` | `mlx-gen/src/text_decoder.rs:197` | Moves into the llm crates (COMP-1) |
| `LatentDecoder` seam | `mlx-gen/src/decoder.rs:29`, `candle-gen/src/decoder.rs:33` | `LatentCodec` decoder override |
| `gen-core/src/guidance.rs` | Used by Lens, MLX SDXL, MLX and candle Bernini, and `candle-gen/src/sampler.rs` | Guidance library with bit-exact `CombineVariant`s |
| `approximation` / `execution_domains` | `gen-core/src/approximation.rs` | Feature classes (DEC-8) |
| SceneWorks `run_utility_job_inner` | `worker/src/lib.rs:2484` | Outer job harness, kept |
| SceneWorks `generate_video_using` | `worker/src/video_jobs/wan.rs:2806` | Template for the generation lane, with typed hooks replacing the flat field bag |
| SceneWorks `CandleStrictControl` | `worker/src/image_jobs/candle_strict_control.rs:45` | Template for `LaneHooks` |
| SceneWorks `memory_strategy::select_strategy` | worker | Selection step inside the admission pipeline |
| SW-F327 deliverables on main | #3258 stream supervisor, #3264 memory-contract/probe unification, #3270 cfg alias and architecture tests | Reuse the shipped contracts (§7.5). |
| SceneWorks `ADVANCED_KEY_RULES`, `TECHNIQUE_KEYS` | `core/src/workflow_share.rs:1411-1833`; training | Part of the `OptionSpec` seed (OPT-1) |
| Engine capability dumps and their producer `dump-engine-capabilities` | `config/engine-capabilities/*.json`; `memory-catalog-campaign.yml:911` | Replaced by the hardware-free capability export (REG-3). Every consumer migrates in the same feature (AP9). |

---

## 3. Engine design

### 3.1 Request, plan, streams (PIPE-2…5, PIPE-11, PIPE-13, REG-4)

```rust
// gen-core — SKETCH
pub struct GenerationRequest {
    pub core: CoreRequest,              // prompts, negative, mode, OutputSpec (per-stream final geometry),
                                        // count, seed, steps, guidance, sampler: SolverId,
                                        // scheduler: ScheduleId, strength, conditioning
    pub features: FeatureParams,        // FeatureId → typed params
    pub family: Option<FamilyExtension>,// per-family typed blocks (moss, veda, h3, ltx, …)
    pub memory: Option<GenerationMemory>,// admission only (MEM-4)
}

pub struct RenderPlan {
    pub route: RouteKey,
    pub streams: StreamSet,                     // e.g. {video} | {video, audio} | {joint_rows} | {image}
    pub output: OutputSpec,
    pub count: CountPlan,                       // Sequential{n} | Batched{n} | Packed{samples} (PIPE-13)
    pub seeds: SeedPlan,                        // route SeedScheme (legacy, bit-exact) + new-stream salts
    pub program: StageProgram,                  // fixed passes or bounded native region over typed stage templates
    pub features: Vec<AppliedFeature>,
    pub memory: PricingPlan,                    // per-component weights, per-phase peaks, transition deltas (§3.9)
}

// StageProgram::Conditional binds route-owned guards to finite runner-owned invocations (§6.10).
pub struct PassPlan {
    pub id: PassId,                             // stable route/recipe role, not registration order
    pub stages: PassStageBindings,              // prompt/encode/reference/init/decode + explicit reuse
    pub schedules: PerStream<SchedulePlan>,     // resolved once; phase geometry cannot silently re-shift it
    pub clock: StreamCoupling,                  // outer-step → per-stream schedule node/evaluation rules
    pub phases: Vec<PhasePlan>,                 // a validated partition of this pass's executed outer steps
    pub input: PassInput,                       // initial request or typed output of an earlier pass
}

pub struct PhasePlan {
    pub span: StepRange,                        // [from, to) executed outer steps in the enclosing PassId
    pub geometry: PerStream<Geometry>,
    pub model: ModelSlotRule,                   // Fixed(slot) | SigmaKeyed{boundary, hi, lo}  (Wan curated)
    pub adapters: AdapterScales,                // passed per request through ForwardArgs (PIPE-14)
    pub guidance: GuidancePlan,                 // CombineVariant + branches (+ perturbations)
    pub solver: SolverId,
    pub history: SolverHistory,                 // Carry | Reset; validated against execution-state compatibility
    pub transition_in: Transition,
}

pub enum SchedulePlan {                         // all three resolve to immutable per-pass node sequences
    Global { schedule: ScheduleRef },            // native/global schedule, sliced by phases (Krea, low-res)
    Fresh { steps: u32, denoise: f32, schedule: ScheduleId }, // independent pass; ComfyUI semantics
    Explicit { table: ScheduleTableId },         // LTX native pass tables
}

pub enum Transition {                           // each is typed by accepted boundary-state kind
    Start,
    Continue,
    FromClean { streams: StreamSet, estimate: EstimateSource, resize: Option<ResampleKind>,
                to: PerStream<SigmaRef>, noise: NoiseScalingRef, seed: SeedStream },
    PixelRoundTrip { stages: PassStageRefs },     // intermediate Decode → pixel resize → Reference/Encode
    Route(TransitionId),                         // family-supplied, bit-exact (LTX); declares state law
}
pub enum ResampleKind { Nearest, Bilinear, Learned(ComponentId) }    // learned = route component + normalization
pub enum BoundaryState<L> {
    Noisy { x: L, sigma: PerStream<f32> },
    CleanEstimate { x0: L, provenance: EstimateProvenance },
    Route { state: L, kind: BoundaryKind },
}
```

**Approved DEC-3:** support global-sliced, fresh-per-pass and explicit-table sources. Resolve each into the same immutable per-pass schedule representation before execution; source-specific logic stays in plan compilation. Each pass owns its schedule and step numbering, and phases select executed outer-step ranges within it. Source choice does not implicitly select a transition, re-noise law or solver-history policy.

**Coordinates and composition (R2-02).**

*Coordinates*
- A pass owns one schedule domain.
- Phase spans are half-open ranges of **executed** outer steps within that pass. Both endpoint nodes are available to the phase.
- Img2img truncation is resolved before K and other span parameters are applied.
- Global, fresh and explicit describe how a pass's schedule is built. They are not three ways of indexing a phase.
- A stream-coupling declaration maps each outer step and evaluation to its per-stream σ. For example, joint H3 uses the same outer-step ordinal for audio and video, with different σ values. No ordering between those σ values is implied.
- Stream schedules of different lengths need an explicit route coupling rule. Without one, compilation refuses.

*Passes*
- Pass-producing Features run before the span contributions that target their passes. This order comes from declared `requires` and ordering edges.
- Stable pass roles identify the base, refinement and native stages. A reference to a missing or ambiguous role refuses.
- By default, low-res start targets the first denoising pass, once.
- A generic per-phase option must name its pass, or state an explicit all-pass rule.
- Wan expert thresholds stay as model-slot functions evaluated at every actual σ, including intermediate and endpoint evaluations. They are never rounded to an outer-step boundary.

*Ownership*
- Within a pass, boundaries merge by executed step ordinal.
- Each override is owned by one `(PassId, stream set, axis, span)`. The axes are geometry, model rule, adapters, guidance, solver and schedule policy.
- Native route defaults can be overridden only through declared surfaces.
- Each boundary has exactly one typed transition composition and one resolved history policy. Competing transition owners refuse, unless a registered composition defines their operation order and their input and output states.
- For low-res start, the composition order is: estimate → unpack and resize the clean sample → destination noise scaling → repack → history reset.
- Hires.fix owns a separate pass boundary, so it never conflicts with an interior low-res boundary.
- No tie is broken by registration order, and nothing is silently overwritten.

**Plan compilation.** Compilation is weights-free and deterministic. It runs in this order:
1. Resolve route defaults, request versions and recipe versions; validate raw parameter types/ranges.
2. Build native passes, their schedule sources and stream coupling, including strength truncation and stage dependencies.
3. Validate Feature relations as a graph (including multi-Feature cycles). Apply pass producers and then targeted span contributions in a unique declared order. Resolve ownership, boundary transition compositions and final phase partitions.
4. Validate the **entire rewritten plan** against `RouteStructure`: every geometry/grid, mode/conditioning restriction, schedule node, transition-state law, estimator capability, slot, solver, history rule and component dependency. Applicability is re-evaluated with the composed plan, not just the original request.
5. Resolve conditioning and preparation dependencies per pass/phase, including negative branches, scales and stage-result reuse; construct their immutable dependency keys.
6. Derive `PricingPlan` from existing aggregate component/phase/operation evidence and additional coexisting transition buffers (§3.9), plus all boundary-evaluation work. Compile the exact resolved plan used by admission and execution. Admission-selected equivalent Features use the same validation and pricing path; each finite candidate is considered once in deterministic order, and no unvalidated rewrite occurs after selection. On a fresh generation, Run admission can change a selection only by repeating that path before execution. Exact replay locks the recorded output-affecting selection and refuses if unavailable; recipe Features, estimator choice and tiers stay fixed.

**Boundary conversion (R2-01).**

*Getting the clean estimate.* `EstimateSource` is declared in the plan and recorded with provenance: (pass, stream, σ, input state, model/adapter/branch configuration, dtype).

| Source | What it does | Status |
|---|---|---|
| `EndpointEvaluation` **(default)** | One additional **read-only** model evaluation on the low-res `x_σ` after step K, at the actual endpoint σ[K]. It uses the departing phase's guidance and adapter configuration, and that phase's slot rule evaluated at σ[K]. The runner applies the route's `ModelSampling::denoised_coeffs`, or a conformance-tested driver estimator. | Default |
| `PreviousEvaluation` | Complete **all K** low-res outer steps normally. Retain the guided primary x0 from the last step's primary evaluation at σ[K−1], with its actual input/configuration provenance. At the boundary, resample/re-noise that earlier clean estimate to σ[K]. No endpoint forward is added, and no integration or sub-evaluation is skipped. | Candidate under DEC-10; fixed semantics below, then bounded quality/time comparison. `EndpointEvaluation` remains default. |

**Previous-estimate protocol (R5-02).** This is an explicitly different estimator, not a claim that the earlier x0 equals the endpoint estimate.

- Capture `GuidedPrimaryX0` from outer step `K−1` after the normal prediction/guidance conversion. Keep its immutable value without substituting the last callback's output. Heun's corrector and DPM++ SDE's midpoint are not this primary estimate.
- Finish that entire step: all sub-evaluations, stochastic draws, solver updates, post-step hooks, materialization, cancellation and the normal completed-step progress event occur. State at the boundary is exactly the ordinary post-K-step state; there is no partially committed step or rollback.
- Then use the captured x0 for the declared clean-sample transition at destination σ[K]. Preserve its original σ[K−1], input-state, slot, adapter/branch and dtype provenance; do not relabel it as an endpoint estimate. At a coincident expert boundary its source slot may differ from the slot chosen at σ[K]. The next trajectory evaluation uses the destination phase's normal slot rule.
- The transition resets incompatible solver/guidance/Feature state and preserves absolute RNG coordinates as for the default recipe. Capture itself neither adds a guidance update nor a noise draw. Retaining x0 through the corrector/midpoint can increase peak memory; price that retained buffer where it coexists with the operation evidence (§3.9).
- A solver/driver declares whether it exposes this precise guided primary estimate without another model forward and with normal step completion. Shared Euler, Heun and DPM++ SDE paths can use the same capture seam; each needs its trace. Other native drivers must prove the same contract or refuse this **candidate**, leaving the default estimator's coverage intact. No new solver or early-exit variant is introduced.
- Resolve estimator choice at recipe finalization under approved DEC-10, persist it and its semantic version/provenance with the resolved recipe, and reject unsupported or unknown identities. It is not a runtime quality switch or an admission-selected approximation. The approved v1 UI controls remain K and scale.
- There is no fixed percentage saving. Compare full elapsed time, quality and peak memory using the existing bounded proof set. The removed endpoint work's cost depends on scale, M/K, solver and encode/decode overhead; retention can offset some of the benefit.

**Weights-free protocol examples (not runtime evidence):** for K=2, Euler completes steps 0 and 1 and retains the primary x0 from step 1. Heun additionally completes each required corrector, but still retains step 1's **primary** estimate, not its corrector. DPM++ SDE completes step 1's midpoint and its draw addresses 2 and 3 (and earlier step 0 draws 0 and 1); the next step uses addresses 4 and 5 if that step draws. Both estimators have two completed low-res progress ticks. Only the default then performs the read-only endpoint evaluation; the candidate transitions from its captured earlier x0. Tests assert the actual conditional draw sequence, state commits and hooks rather than forcing draws on a terminal node where the solver omits them.

- **No silent reuse.** An earlier primary or midpoint x0 is never used just because it is the most recent estimate. A source is valid only when declared, and a route reuses an estimate only if it certifies the declared state and configuration.
- **The endpoint evaluation is read-only.** It cannot commit autoregressive KV or advance stochastic draws. Drivers without a read-only compatible estimator refuse structurally.
- **It is accounted internally.** It counts toward internal evaluations, cancellation and pricing. It adds no outer progress step and no duplicate preview.

*Resample and re-noise.*
- **Resample clean, normalized codec latents** in their physical spatial axes, after unpacking. Never interpolate a packed token dimension, an audio axis or residual noise. Never apply a second codec mean/scale transform.
- **`low-res-start/1` arithmetic:**
  - bilinear half-pixel sampling (`align_corners=false`, border clamp);
  - interpolation and noise arithmetic in f32;
  - conversion to the destination route's declared solver-latent dtype and layout afterwards.

  Backend kernels implement this same rule. Legacy and native route transitions keep their existing casts and noise draws.
- **Noise:** independent target-grid noise from the Feature's domain-separated seed. For destination σ = s, `ModelSampling::noise_scaling_coeffs(s)` gives `(a_noise, a_clean)`, and `x_dst = a_clean·resize(x0) + a_noise·ε`:
  - flow: `(1−s)·resize(x0) + s·ε`;
  - VE/EDM: `resize(x0) + s·ε`.
- **Noisy state is never treated as clean.** A noisy-to-noisy transform needs its own declared law. `FromClean` cannot accept a `Noisy` state.
- **Other streams are untouched** unless the transition names them.
- **History.** Any geometry change or re-noise resets the joint solver history, unless the route explicitly partitions independent solver states.
- **Evidence:** inference `gen-core/src/sampling/model_sampling.rs:59-68,223-231`; round-2 R2-01.

### 3.2 Route structure and primary declarations (AP5, HOOK-4)

```rust
pub struct RouteStructure {
    // primary declarations (family-owned; each behavioural one has a Tier 0 conformance check)
    pub streams: Vec<StreamDecl>,               // LatentSpace, geometry rules (multiples, legal grids), time convention
    pub prediction: PredictionType,
    pub geometry_change: GeometryChange,        // supported axes/grids/modes; conformance-checked
    pub loop_kind: LoopKind,                    // SharedDriver | FamilyDriver(DriverSupport)  — DriverSupport conformance-checked
    pub guidance: GuidanceSurface,              // CombineVariant(s), row order, distilled
    pub seed_scheme: SeedScheme,                // legacy schemes preserved bit-exact (PIPE-11)
    pub count: CountSemantics,
    pub residency: ComponentLifetimes,          // which component is live in which stage/phase (MEM-1)
    pub preparation: PreparationDependencies,   // model/scales/branches/conditioning/geometry dependencies
    pub boundary: BoundarySupport,             // accepted states, read-only estimate + noise law
    // computed from the RouteRecord
    pub conditioning: Vec<ConditioningKind>,    // each GridAligned | GeometryFree
    pub adapters: AdapterSurface,               // install policy per route; residual-capable?
    pub decode: DecodeSurface,                  // codec, tiling law, overrides (PiD)
    pub schedules: Vec<ScheduleId>, pub solvers: Vec<SolverId>, pub models: Vec<ModelSlot>,
}
```

### 3.3 Pipeline runner (PIPE-1, 6, 7, 14, 15)

The runner is policy in gen-core, generic over backend operation traits. The precedents are `Sampler<L: LatentOps>`, `GuidanceOps`, sc-20414's `execute_denoise_plan<L>`, and candle `AvLatents` [ENG "right"].

**Dispatch shape [ENG-22]:**
- Families are reached through an object-safe `dyn` interface. The per-step dynamic call is negligible next to a model forward.
- Step contexts and driver state may be type-erased (owner-local `Box<dyn Any>`), with request-owned lifetimes; immutable preparations are built once per complete dependency key.
- Tensor operations inside the backend stay statically dispatched.
- Registration factories, job messages and immutable tensor-free specs can cross threads. Create, run and destroy each loaded model/codec, hooks and scratch on its executor. External cancellation sets a flag. The owner checks it before work and performs safe cleanup. Poisoned native state is retired by its owner/supervisor without another GPU operation. Routes default to serial execution. A stronger policy requires evidence for the device, RNG and every loaded component.
- This avoids instantiating the runner about 40 times per binary, which would cost iOS size and compile time.

```rust
// gen-core — SKETCH
pub trait RunnerBackend {                       // implemented once per backend (L1)
    type Latent: Clone;                         // a LatentBundle of per-stream tensors
    fn ops(&self) -> &dyn BundleOps<Self::Latent>;      // LatentOps + geometry ops, per stream
    fn exec(&self) -> &dyn ExecPolicy;
}

pub trait FamilyModel<B: RunnerBackend> {     // object-safe
    fn structure(&self) -> &RouteStructure;
    fn prepare_step_context(&self, key: &StepCtxKey, cond: &CondBundle<B>, refs: &AlignedRefs<B>,
                            args: &PrepareArgs)
        -> Result<StepCtx>;                                  // erased; cached by key; evicted on phase exit
    /// Evaluate the requested branches; the family decides batching (PIPE-8).
    fn predict(&self, ctx: &StepCtx, x: &B::Latent, t: &TimeInput, branches: &BranchSet,
               args: &ForwardArgs, info: &StepInfo) -> Result<BranchOutputs<B>>;
}
pub struct ForwardArgs<'a> { pub adapter_scales: &'a AdapterScales, pub perturbation: Option<&'a Perturbation> }
pub struct PrepareArgs<'a> { pub adapter_scales: &'a AdapterScales, pub branches: &'a BranchSet,
                            pub effective_model: ModelIdentity }

pub trait FamilyDriver<B: RunnerBackend> {    // PIPE-7
    fn support(&self) -> &DriverSupport;                    // conformance-checked
    fn run_slice(&self, input: SliceInput<'_, B>, state: &mut DriverRunState<B>,
                 hooks: &mut StepHooks<'_, B>) -> Result<BoundaryState<B::Latent>>;
}

pub enum EvaluationKind { Trajectory, BoundaryEstimate }
pub struct StepInfo { pub pass: PassId, pub phase: u16, pub step: u32, pub outer_step: u32, pub eval_in_step: u32,
                      pub kind: EvaluationKind,
                      pub sigma: PerStream<f32>, pub next_sigma: PerStream<f32> }
pub enum TimeInput { Sigma(PerStream<f32>), FlowMap{ t: f32, t_ref: f32 },
                     PerToken(TokenTimeBuilder), RowTimesteps(RowTimes), Angle(f32) }
```

**Request execution state (R2-03).**
- **`RunState` is request-owned.** It holds solver and driver state, previous nodes and intervals, stochastic draw coordinates, request caches and any autoregressive causal state.
- **No fresh history per slice.** The shared driver takes `&mut SolverRunState` and the native driver takes `&mut DriverRunState`.
- **Absolute schedule positions.** A phase addresses its range in the **whole pass schedule**, not as a freshly zero-indexed solver call.
- **`Carry` requires** the same solver variant and numerical convention, a contiguous schedule and compatible geometry. A solver change resets, unless an explicit, tested conversion exists.
- **Native routes keep their behaviour.** Wan continues one scheduler across the expert crossing; Krea keeps its declared Reset.

*RNG and caches*
- **RNG coordinates and solver history are separate fields.** Reset clears numerical history and previous intervals. It does not touch RNG coordinates or unrelated KV state.
- **Inert splits keep absolute draw indices.** A draw addressed by `i`, `2*i` or `2*i+1` keeps its absolute pass index.
- **New boundary noise is salted** by `(recipe version, PassId, stream, sample identity, transition role)` and never advances the solver's draw stream.
- **Pass restarts** follow the route's legacy policy or a named new recipe stream.
- **Driver-cache retention and reset rules** are explicit, and are checked on geometry change.
- **Inert boundaries can be elided** at execution, so they never add a materialization, terminal σ or cast.
- **Evidence:** inference `sampling/solvers.rs:219,264,380,403`; Wan `pipeline/moe.rs:33-46`.

**Guidance and Feature state (round-3 correction).**
- **`RunState` also owns** guidance momentum and mutable Feature state.
- **`EvaluationKind::BoundaryEstimate` propagates** through prediction, combine and hooks. It evaluates the departing configuration against a snapshot and discards every update to momentum, history and cache counters. It only emits internal evaluation and telemetry accounting.
- **It cannot** mutate live APG state, advance outer progress, commit KV or consume solver draws. Immutable preparation caches can be reused through their normal dependency keys. Snapshot and scratch allocations are priced.
- **Features that cannot be evaluated read-only refuse.** Composed-plan validation refuses the combination instead of silently bypassing it.
- **State descriptors declare** their dependencies and their invalidation rules for geometry, branch-layout, model and adapter changes.
- **Recipe v1:** a geometry change resets incompatible, shape-dependent momentum and residual caches before the next trajectory evaluation. Geometry-independent state keeps its declared cadence. Native paths with no Feature keep their existing update behaviour.
- **Why:** APG's `MomentumBuffer::update` stores a running tensor that guidance updates on every call, independently of solver history (`gen-core/src/guidance.rs:168-190,201-208`).
- **Tier 0 check:** a fake stateful guidance operator verifies snapshot non-mutation and invalidation before any new-shape arithmetic, with no learned forward.

**Preparation and cache dependencies (R2-04).**
- **Order per evaluation:** resolve the model slot for the **actual evaluation** → ensure weights and residency → derive aligned inputs → prepare context with `PrepareArgs`.
- **Keys contain the declared subset of:** request, pass and sample identity; effective weights and tier; geometry; encoded conditioning; adapter identity and scales; branch layout; perturbation.
- **Examples:** an adapter-dependent aggregator never shares a key with base-only preparation (Krea, `mlx-gen-krea/src/pipeline.rs:1127-1148`). A CFG batch-1 cache never stands in for batch 2 (Wan).
- **Caches with different dependency subsets are split.** A geometry-only RoPE cache survives scale or branch-count changes.
- **Reuse is not free.** Retaining a cache for future use is an explicit, priced live allocation.
- **Nothing per-request is installed** on shared resident modules.

**Stage execution:**

```text
S0  plan = compile(request, route); select/admit exact validated plan before load
for trajectory in plan.count:                                 [sequential/batched/packed, legacy semantics]
  run = request_owned_state(trajectory, plan.seeds)
  for pass in admitted_invocations(plan.program, run):         [fixed order, or guarded templates bounded by S0]
    check invocation and retained state against admitted limits  [conditional calls and realized path, §6.10]
    S1 prompt = resolve_or_run(pass.prompt_prep)                [enhancement only where declared]
    S2 cond   = resolve_or_encode(pass.conditioning)             [reuse only with equal dependencies]
    S3 refs   = resolve_references(pass, earlier_outputs)        [geometry-free + aligned builders]
    S4 schedule = pass.resolved_schedules                       [no implicit phase-local mu change]
    S5 state = init_or_enter_pass(pass.input, refs, run)         [typed output; source decode if required]
    for phase in pass.phases:
      S6 scope.enter_transition(actual_boundary_identity)      [includes source/destination work and estimator]
         apply declared transition to state                   [estimate/resize/noise law; price each operation and coexisting buffers]
         scope.enter_phase(actual_phase_identity, selected_rungs)
      S7 drive whole-schedule range with run.solver/run.driver + hooks
         per evaluation: cancel → slot_for(actual_sigma) → ensure weights → align/prepare → predict/combine
      S8 exit: retain/evict declared state and run safe end hooks
    S9 post_latent; S10 decode_if_requested(pass)                [intermediate or final, own visibility]
    S11 emit_declared_output(pass)                              [final artifacts or bounded chunks]
finish request accounting and host resource release on every exit; GPU cleanup only where safe
```

**Stage notes.**
- **`PixelRoundTrip` points at runner-owned stages:** intermediate decode, pixel resize, and next-pass reference and encode. It never hides them inside a `FamilyDriver`, and never decodes the same intermediate twice. Hires.fix refinement can re-run S1 and S2 after that decode.
- **Encode has an internal dependency graph.** A vision encoder can run before the text encoder that consumes its features (Qwen edit). S3 refers to latent and aligned reference preparation, not to every image input.
- **Chunked output.** A streaming route repeats chunk stage invocations under the same owners, with explicit causal state and bounded sinks. Today's Krea-realtime batch path stays a single terminal decode (PIPE-16).
- **Scope.** Execution is limited to these typed generation stages and their earlier-pass, bounded conditional and chunk dependencies.

**Runner invariants.** These are salvaged from sc-20414 and corrected [ENG-2, ENG-19]:
- **Progress is monotone in outer steps.** Evaluation counts stay internal (DEC-19).
- **Cancellation is checked** at every evaluation, every phase boundary and every model-slot swap or load.
- **GPU end hooks run on request-local errors only.** Provider- or process-poisoned errors skip GPU work; host accounting and claim release still run without masking the original error.
- **Solver history never crosses an incompatible boundary.** Geometry change/re-noise forces Reset; `Carry` on `Continue` additionally requires compatible solver/convention and schedule continuation. RNG and driver-cache policies are independent.
- **Re-noise seeds are domain-separated** for new streams; legacy schemes stay bit-exact.
- **Shared resident modules are never mutated.** Per-request state flows through `ForwardArgs` and `StepCtx`.
- **An orchestration-only class (a) migration must equal its matched legacy route bit-exactly.** Inert splits preserve numerical and stochastic traces.

**Approved DEC-4:** Michael selected both selective prior-art reuse and user-configurable chained denoise passes on 2026-11-03. ENG-16 delivers FEAT-2 as a shared recipe Feature after APP-2/3 and ENG-7. It produces typed passes before phase contributions, uses fresh-per-pass schedules and declares transition, history and seed policies. Preserve low-res start's first-pass-only targeting and validate native-pass composition through stable pass roles; ambiguous ownership refuses. Derive the versioned advanced controls from the shared option schema, inventory/disposition the old controls and persist the resolved recipe. Admission prices the complete chain through the existing `PricingPlan` interface; it does not wait for APP-4. No separate family executor or hand-written wire contract is introduced. No implementation or tracker creation is authorized during this design discussion.

### 3.4 Features (HOOK-1…5) [ENG-11]

```rust
// gen-core (tensor-free, dyn) — SKETCH
pub trait FeatureSpec: Send + Sync {
    fn id(&self) -> FeatureId;
    fn class(&self) -> FeatureClass;                 // Recipe | Equivalent{quality_bound} | Diagnostic
    fn params(&self) -> &ParamSchema;
    fn applicability(&self, s: &RouteStructure, p: Option<&FeatureParams>, plan: Option<&RenderPlan>) -> Applicability;
    fn relations(&self) -> &FeatureRelations;        // conflicts, requires, ordering keys, owned override axes
    fn rewrite_plan(&self, p: &FeatureParams, plan: &mut RenderPlan) -> Result<()> { Ok(()) }
    fn memory_effect(&self, p: &FeatureParams, plan: &RenderPlan) -> MemoryEffect { MemoryEffect::None }
}
// L1 (per backend) — only for model-internal hooks (caches, attention perturbation, token pruning)
pub trait FeatureHooks<B: RunnerBackend> {
    fn id(&self) -> FeatureId;
    fn install(&self, p: &FeatureParams, hooks: &mut HookSet<B>);
}
```

- **Approved DEC-8:** recipe Features are activated by explicit user choice or the selected recipe; equivalent Features preserve requested results within declared, evidenced quality bounds and are the only class admission may select; diagnostic Features are explicitly configured observation. Admission never adds a recipe, changes quantization tier or enables diagnostics to fit memory. If permitted equivalent candidates cannot fit, it returns a typed refusal. These runtime classes are separate from AP13 migration-fidelity classes; quality evidence uses DEC-7's bounded effort. No implementation is authorized during design review.
- **Composition.** Pass producers and targeted phase contributions follow §3.1's typed coordinates, transition ownership and complete-plan validation. Applicability exports separate structural availability from parameter/combination restrictions; a broad static “available” flag never overrides runtime composed-plan validation. Every active Feature's requirements, conflicts and ordering apply to the full active set, not only pairwise checks.
- **Hook ordering** uses declared keys; equal keys are a registration error.
- **Memory effects include transition residency:** both latents during a resample, the learned upsampler, the VAE during a pixel round trip, and step contexts kept for later phases [ENG-23].

### 3.5 Guidance, schedules, seeds (PIPE-8, 9, 11) [ENG-3, ENG-4]

- **Guidance.**
  - `gen-core::guidance` has named operators. Each operator has `CombineVariant`s that reproduce today's formulas bit-exactly: operand order, combine dtype, sign, and casts. Boogu's model-dtype combine with cast and negate is one such variant.
  - A route declares its variant and its branch row order.
  - `BranchSet` can include perturbation branches (STG skip-layers, isolated modality) that `BlockStack` honours.
  - Unifying variants or row orders is optional and needs a class (c) decision.
- **Schedules.**
  - Every native σ builder, shift and baked table is a named `ScheduleId`.
  - `mu` and shift, `img2img_start` and the `/16` hard-code are computed from `LatentSpace`.
  - LTX stage tables are `ScheduleTableId`s used by `SchedulePlan::Explicit` in separate native passes; phase ranges slice those resolved tables.
- **Seeds.**
  - `SeedScheme` is a route declaration. Legacy schemes include `WrappingAddPerImage`, `LtxStageXor`, `CandlePlusN` and parity RNGs (MLX `random::key`, `seeded_noise_nchw`, bf16-rounded, Gaussian-Shading, the InstantID global RNG).
  - The splitmix-plus-salt derivation applies only to new streams.
  - The image-index → seed relation is part of the capability export, and SceneWorks persists it unchanged.

### 3.6 Shared components (COMP-1…6)

**Encoders.** `TextEncoder<B>` wraps llm `CausalLm` and the existing tap API (`hidden_states_with_mask_at`). `TapPolicy` adds the following.

| Addition | Detail |
|---|---|
| Early exit | Stop after the deepest requested layer, so a 12B encoder does not materialize all hidden states |
| Layer subset | Tap only the requested layers |
| Combine | `Concat`, `Stack` or `Interleave` |
| Drop prefix | Drop leading prefix tokens |
| Non-causal mask | Where an encoder needs one |

- **Blockers to resolve** [ENG-20]: masks are causal-only (`llama.rs:416`); the quant codec differs (llm Q4_0/Q8_0/Q4_K vs media Q4_1/affine), so it needs a codec choice or class (b) evidence; and architectures are missing (Gemma 3, the Qwen2.5-VL vision tower).
- **gpt-oss is not required.** No media consumer was found.
- **Towers:** T5/UMT5, CLIP-L/G/OpenCLIP, SigLIP and ViT each get one implementation per backend.
- **Numerics:** each encoder migration is declared class (b) or (c).

**Latent codecs.**

```rust
pub trait LatentCodec<B> {
    fn space(&self) -> &LatentSpace;
    fn tiling_law(&self) -> &TilingLaw;         // declared per codec (Wan vae16/22, LTX causal overlap, DC-AE)
    fn encode(&self, px: &B::Pixels, plan: &EncodePlan) -> Result<B::Latent>;
    fn decode(&self, z: &B::Latent, plan: &DecodePlan, hooks: &mut DecodeHooks<B>) -> Result<B::Pixels>;
}
```

| Codec | Covers |
|---|---|
| `AutoencoderKl2d` | SDXL, FLUX.1, SD3, Z-Image, Chroma, FLUX.2, FLUX.3 |
| `WanCausal3d` | Wan 2.1/2.2, Qwen-Image, Qwen-Image 2.1 |
| Causal-3D kit | LTX and H3 |
| Singletons on the kit | DC-AE, Mage NeRF, audio VAE/vocoders |

PiD stays a decoder override, and its sampler is hosted in S10.

**BlockStack.** `BlockStack<B>` provides `run_block(i, io, ctx, args)` and a block layout derived from `WeightPlan.blocks`.
- `BlockHooks` provide `pre_block`, `extra_kv`, `post_block` and `perturb(i, branch)`.
- The driver owns R4 streaming and eviction, ControlNet/VACE residuals, IP/PuLID extra-KV, preview taps and checkpointing.

**Conditioning.** Conditioning modules are shared. `RefBundle` separates geometry-free outputs from grid-aligned ones.

### 3.7 Backend layer kit (BACK-1…5)

| Kit part | Contract | Replaces |
|---|---|---|
| `Linear` | **Bases:** Dense, AffinePacked, Ggml, GgufK, Nvfp4, Fp8Row, Int8Row, W4A8, ConvRot and existing LLM projection variants including Prism. **Plus:** dtype policy, residual adapters reading request-scoped preparation/forward scales (PIPE-14), and a trainable mode. Preserve all supported formats and operations. Install policy (fold or residual) stays per route (WGT-3). | Media and LLM projection copies; the LTX-local Linear; candle `QLinear`+`AdaptLinear`+`LoraLinear` |
| `AttentionOp` | `attend(q, k, v, mask, AttnPolicy{budget, bounded, sparse, i32_guard}, perturbation)` | Raw SDPA in 38 `mlx-gen*` directories; the overflow-guard sites |
| Norm, Mlp, Modulation, QkNorm, Rope, Embed, TimestepEmbed, Patchify | Parameterized kit types. Replacements that change operation order are declared class (b) or (c). | About 135 MLPs, 304 RoPE functions, about 45 sinusoid copies |
| `Conv{1,2,3}d` | Explicit `Layout`, converted at load | Ad hoc NHWC/NCHW handling |
| `ExecPolicy` | `materialize`, `fuse`, `memory_scope`, `cancel_point`, `stream_scope`, `load_activity` | Direct eval/compile calls in families |

**Trainable mode** switches to composable operations instead of fused CustomOps on candle (WGT-4).

**MLX watchdog ordering** lives in the loader and `ExecPolicy::materialize`.

**Approved DEC-1:** option (b), shared host math and identical layer APIs, is the baseline. Option (c) follows [D §9] only through pilot A (a small representative family) and pilot B (FLUX.3 end to end including LoRA training). Both must pass BACK-5's fidelity/capability, speed, memory and binary-size criteria before generic rollout; if either fails, remain at (b). Keep backend-specific kernels, fusion/materialization and autodiff policy. The shared runner and low-res start proceed independently of generic rollout. This design approval does not start either pilot during the design-only discussion.

### 3.8 Weights and adapters (WGT-1…5)

```rust
pub struct ComponentPlan {
    pub id: ComponentId, pub tier: QuantTier,     // per-component tier (q4 DiT + bf16 TE)
    pub sources: Vec<SourceDialect>,              // SplitTier{affine-q4-g64-v1…} | SingleFile | Gguf | Diffusers
    pub keys: KeyRules, pub layout: LayoutRules, pub dtype: DtypePolicy, pub quant: QuantPredicate,
    pub blocks: Option<BlockLayout>, pub expect: Expectations,
}
```

- **Lifecycle:** stage → pin/seal → plan → decode → layout → quantize → residency → bind → adapters → receipt. Each stage is a hook point and a cancellation point (WGT-5).
- **Codecs:** the MLX affine triple, GGML Q4_1/Q8_0 and GGUF k-quants are registered. Candle's MLX-tier → GGML conversion stays a cached codec.
- **Adapters:**
  - Routing is generated from `KeyRules`; aliases are data.
  - **The install policy is per route, and today's policy is the default.** Candle dense bases keep fold-at-load for parity.
  - A route switches to residual only when a recipe Feature requests runtime scales. That switch is declared in the plan (WGT-3) [ENG-9].
  - DoRA/BoRA use ordered fold.

### 3.9 Memory relocation (MEM-1…6)

| Rung | New home | Family supplies (declarations only) |
|---|---|---|
| R1 staged residency | Runner stage driver: a lifetime graph over stages and phases, including model-slot swaps (Wan experts), mid-pipeline components (LTX upsampler, H3 keyframe VAE encode), LLM pre-stages (FLUX3 PUP, LTX enhancer) and denoiser-owned prefill (SenseNova) | `ComponentLifetimes` |
| R2 bounded/tiled decode | `LatentCodec`, within declared quality bounds (`decode_quality.rs`) | Codec tiling law (one per codec, not per family) |
| R3 bounded attention | `AttentionOp` | Nothing |
| R4 bounded transformer residency | `BlockStack` + `WeightPlan.blocks` | Block layout (data) |
| iOS quanta | Hooks at plan and step quanta (`STEP_QUANTUM` and `PLAN_QUANTUM` checks, as in the Iris solver) | Quantum labels |

- **Until relocation, the runner drives today's `MemoryRequestScope` lifecycle** for each migrated route (MEM-5): `enter_phase`, `configure_decode`, `materialize_transformer_window` and `finish`.
- **Contracts are computed from the route record plus these declarations (MEM-2).**
- **Pricing (MEM-3, R2-05, C4-02): sum what is live together, at component and phase granularity, on existing evidence.**
  - **Building blocks:** per-component weight bytes from the weight plan; existing aggregate peak evidence with declared coverage (two anchors plus derivation where applicable); and additional buffers computed from shapes/dtypes. Use the same mechanism for endpoint estimation, learned upsampling, codec encode/decode and ordinary phase execution. This adds no per-model allocation graph, re-keying or recapture campaign.
  - **Evidence basis:** each reused value declares its existing coverage at component/phase granularity: `TotalPeak` (already includes named resident components and ordinary latent/stream/workspace roles) or `ActivationPeak` (excludes weight residency and states which ordinary buffers are included). A joint AV total includes both streams and joint workspace; separate stream estimates may be summed only when their coverage is disjoint, with shared weights/workspace counted once. Do not add a total and its included parts.
  - **Normalize before composing:** for `TotalPeak`, keep the total and add only required resident components or coexisting buffers absent from its coverage. For `ActivationPeak`, add the required resident weights once, then only missing buffers. Do not subtract an assumed component cost from a measured total or treat an unclassified total as activation-only. Existing identity keys are preserved; coverage is a declaration about the existing evidence, not a new measured grid. If its basis cannot be established from the source, keep the existing unpriced-evidence posture (ADM-2/7), mark it unpriced and expose the reason; never manufacture a smaller priced value or silently change refusal policy. For a newly added recipe, an unbounded increment is a typed recipe refusal (§6.12), independent of legacy-job posture.
  - **Operation peak:** per budget domain, `normalized operation evidence + extra resident component bytes + extra coexisting buffers`, where both extras exclude everything already covered. The required resident set comes from that operation's component dependencies/lifetimes; it need not equal the departing phase's weights. Include retained contexts, previous-estimate capture/snapshot state, source/destination latents, fresh noise and casts where they actually coexist.
  - **Transition peak:** the maximum of its operation peaks (endpoint forward, clean-latent resize, learned upsampler, intermediate codec work, re-noise and preparation as applicable). The operation evidence includes its activation/workspace peak. Do not model learned upsampler/codec workspace from output latent sizes alone or omit it because the weights are resident. A sequential operation can release its scratch before the next; only retained buffers cross that boundary.
  - **Request peak:** the maximum over all steady phase/stage peaks and transition-operation peaks. Neighboring phase peaks remain candidates; they are not blindly summed. Feature deltas are applied at the operation where they coexist, not added to an unrelated global maximum.
  - **Deduplication:** aliases (a latent reused in place, packed views, shared weights) are counted once, through declared alias rules. Per-allocation identity is used **only** where it matters, mainly latent aliasing at transitions. It is not a program-wide modelling obligation.
  - **Budget domains:** unified memory is charged once. Separate host and CUDA-device budgets are priced separately.
  - **Feature memory effects** are expressed as component and transition deltas, never as an unexplained additive correction.
  - **Closure fixtures (pure, no model execution):**
    - two streams that each fit alone but not together;
    - a transition peak above both steady phases;
    - alias deduplication;
    - a learned transition with workspace larger than its shape-only delta;
    - a joint-total evidence value that must not count streams/weights twice;
    - equivalent declared `TotalPeak` and `ActivationPeak` inputs producing the same normalized result, plus an unknown basis retaining the existing unpriced posture.
  - **What this deliberately avoids:** a per-allocation live graph covering every family's activations. That would re-create the dismantled calibration apparatus.
- **First-consumer admission (R2-06).**
  - **ENG-1b** delivers the pricing interface (phase peaks plus transition deltas) and the actual phase and scope identities for every adopted FLUX.1 route.
  - **Existing SceneWorks admission paths consume the validated plan** or its derived priced inputs. They never guess from the final width and height.
  - **`MemoryRequestScope` receives,** for each phase, its geometry, mode and conditioning, selected rungs and transition identity, under the whole-plan budget reservation.
  - **ENG-2** extends the same integration to Hires.fix's intermediate codec and encoder stages, and to its own adopted routes.
  - **These are production interfaces that APP-4 keeps**, not temporary parallel implementations. APP-4 later centralizes the callers and budget hooks. ENG-1b and ENG-2 do not depend on it.
  - **Baseline:** #3264 reached main with SW-F327. Recheck the actual consumers at ENG-1b against §7.5.
- **iOS admission** (`ios_admission.rs`, per-family `ios_pricing/`) is replaced by plan-byte pricing plus quantum hooks (MEM-6).

**Aggregate accounting walkthroughs (R5-01; illustrative units, not measurements).**

- A learned transition operation has an existing total peak of 180, covering its required resident weights=100, ordinary buffers=20 and workspace=60. Additional coexisting retained/source/noise buffers absent from that evidence total 96. Its priced operation peak is **276**, not weights+delta=196. If neighboring phase peaks are 160 and 200, a budget of 250 must refuse this priced candidate. Equivalent activation-only evidence=80 plus the same weights=100 and extras=96 gives the same 276.
- Joint AV total evidence=190 already covers weights=100, video=40, audio=30 and joint workspace=20. Its base peak remains **190**, not 280 from adding the already included non-weight parts again. A new disjoint 40-unit transition buffer raises that operation to 230 only while it coexists. A view alias adds zero.
- If a legacy record's inclusion basis is unknown, the calculation is **unpriced with a reason**, handled by the route's existing posture. These examples justify neither remeasurement nor stricter refusal policy by themselves.

### 3.10 Route records and the capability export (REG-1…6)

```rust
pub struct RouteRecord {
    pub key: RouteKey,                      // stable provider IDs (AP12)
    pub modes: Vec<Mode>,
    pub components: Components,             // encoders, codec, conditioning modules, upsamplers, enhancers
    pub model: ModelBinding,                // FamilyModel | FamilyDriver constructor
    pub structure: StructureDecls,          // primary declarations (AP5)
    pub weights: WeightPlan,
    pub defaults: RouteDefaults,            // native steps, guidance, schedule, shift, strength
    pub limits: ProductLimits,
}
```

- **The capability export** (`sga.capabilities/2`) is a pure function of the linked registry. Hosted descriptor-only builds cover the shipping feature sets without GPU access or a dedicated off-Mac runner (REG-3). [APP-14]
- **Verification item.** ENG-7 builds descriptor-only graphs on hosted CI with the declared macOS/MLX, iOS subset, and Windows/Linux Candle feature sets represented. Compare exported IDs/modes to the corresponding shipping registries, including bespoke/audio boundaries. A pin-only change with equal semantics produces identical canonical bytes. If toolkit/device linkage is indispensable, report the missing partition and revise REG-3 before changing the proof lane.
- **Authority.** The export feeds UI availability. The worker's linked engine is authoritative at runtime and refuses with a typed reason when the two disagree [APP-13].
- **Fields:** modes, defaults, limits, solvers, schedules, guidance surface, conditioning (aligned flag), adapters (formats, install policy, per-phase capability), quant tiers per component, preview, decoders, seed scheme and image-index → seed relation, count semantics, Feature applicability and limits, and memory rung support.

---

## 4. SceneWorks design

### 4.1 Model registry (APPREG-1…6) [APP-11, APP-12, APP-16, APP-17]

```text
STATIC LAYER (build time; checked-in snapshot; usable by an API without engines)
  capability export (REG-3) ⊕ slim product manifest (identity, downloads, licensing, copy, menus)
  ⊕ evidence store (memory facts moved out of the manifest; identity keys preserved verbatim)
RUNTIME LAYER (API-owned unless noted)
  imports (user.models.jsonc, manifest schema v2 + upgrader) · ComfyUI external-base scans ·
  Iris trained-checkpoint overlays · user hides/overrides · OS/install/decoder/credential projections ·
  worker-local artifact binding (shipping jobs receive portable identities plus authorized intent)
ONE MERGE RULE  → ModelSpec (typed)
```

- **Job snapshot and local binding.** Persist logical provider/variant, immutable repository/revision or artifact digest, adapter/component identities and permitted request/product choices. Derived client availability is advisory. The API applies its canonical repository/component restrictions before fetch or import. User aliases and hides affect presentation and selection only. The worker resolves the same identity under its own root, checks source confinement, verifies bytes and holds its local lease for the attempt. Credentials, consent evidence, absolute locators, machine paths and live install observations stay with their authority. Only permitted logical IDs, public refusal codes and execution facts leave it. Upgrade a legacy carrier through the same validation. Test root A → B with identical identity, missing bytes, alias substitution and a protected inherited component whose overlay clears a flag.
- **Desktop seeding** ships the evidence store.
- **Manifest memory-key consumers** migrate in the same feature (APPREG-5 list).
- **Hand-table cutover:** for each inventoried table, generate it from `ModelSpec`, assert equality in a test, switch consumers, then delete the table and the test.

### 4.2 Options, validation, UI (OPT-1…8) [APP-18…22, APP-41, APP-42]

```rust
pub struct OptionSpec {
    pub id: OptionId, pub aliases: &'static [&'static str], pub location: WireLocation,
    pub value: ValueShape, pub default: DefaultRule, pub range: RangeRule,
    pub applies_when: Predicate, pub effects: &'static [OptionEffect],   // e.g. enabling hiresFix clears usePid
    pub intent: IntentClass,                                            // WhatToMake | MachineBudget | Context
    pub ui: UiMeta, pub binds: EngineBinding,
}
```

- **Seed:** the union of `ADVANCED_KEY_RULES`, `TECHNIQUE_KEYS`, production `advanced.get` literals and film-plan keys.
- **Custom renderers**, enumerated: phase editor, poses, layout editors, StructuredPrompt, H3VedaSettings, the MOSS family, PromptBatch, SampleIndex, NumberPair, the YuE2 lab, the Ideogram caption and the Character builder.
- **Studios:** Image, Video, Audio, Training, the Simple studios and Character.
- **Approved DEC-9 validation:** isolate the pure Rust validator from native runtime dependencies and compile it to WebAssembly for local feedback, using the same rules as API/MCP/worker validation. The server always revalidates submissions; live installed-model/device/memory checks stay server/worker-owned. Browser results use their metadata snapshot. APP-2 verifies a shared native/wasm corpus, compatible schema/validator versions, startup/bundle cost, offline behavior and measured interaction latency on desktop/RunPod. Set concrete UX budgets at its planning. If portability or bundle cost is disproportionate, use cached availability plus server cross-field validation, retaining one rule owner; no JavaScript rule copy.
- **Versioning (OPT-6):** the upgrader handles payloads, recipes, studio settings, share envelopes, film plans, training plans, and the HTTP and MCP DTOs.
  - Keep the existing v1 writer for expressible content. An active low-res or chained recipe requires outer `schemaVersion: 2`; revised readers support v1/v2 and validate the nested recipe identity. Exercise the real v1 reader.
  - A newer version is refused with a typed reason.
  - The corpus comes from historical builder outputs.
- **Refusals** render in the web and MCP. Telemetry field names and `advanced.memoryStrategy` stay stable (OPT-8).
- **Exact replay** binds logical source/components and adapters, backend and applicable numerical runtime identity, seed/draw scheme, geometry, recipe version, schedule, estimator, solver and every quality-affecting selected policy (including tiling/rung). If replay cannot use that identity or budget, return `ReplayUnavailable`. A separate user action may start a fresh generation with a disclosed change. Portable import revalidates compatibility, availability and redactions. Existing regeneration continues to discard machine-specific stamps and choose its existing seed policy.

### 4.3 Admission pipeline (ADM-1…8) [APP-7…10]

This is the [F §4.6] contract, with the `RenderPlan` as input and a per-route failure posture.

```rust
pub struct AdmissionRequest<'a> { pub route: RouteKey<'a>, pub phase: AdmissionPhase, pub plan: &'a RenderPlan,
    pub tiers: TierCandidates<'a>, pub load: &'a LoadSpec, pub contract: &'a MemoryContract,
    pub evidence: EvidenceSet<'a>, pub posture: FailurePosture /* per (backend × evidence basis), per route */,
    pub cache: CacheFacts }
pub trait BackendBudget { fn probe(&self) -> Result<MemoryBudget, BudgetUnavailable>; fn reclaimable(&self) -> u64;
    fn margins(&self) -> &MarginPolicy; fn evict_to_reclaim(&self) -> Option<&dyn Fn() -> Result<bool>>; }
```

- **Scope (ADM-1).** There are at least 25 paths: A1–A18 [F §4.1] plus:
  - MOSS (`moss.rs`)
  - SenseNova (`sensenova_jobs.rs`)
  - InstantID (`instantid.rs`)
  - Iris training (`iris_training.rs`)
  - Iris restore (`iris_restore.rs:502`)
  - depth (`depth_jobs.rs:617`)
  - API-side training refusals

  APP-4, item 0, re-inventories by grep over probe and refusal patterns.
- **Behaviour changes are measured and approved (ADM-7).**
  - Floor-versus-ladder ordering stays as it is today per route unless a change is approved. For example, Krea control is floor-first, and making it ladder-first would make cheap refusals pay the full ladder cost.
  - Video moves to Plan-and-Run admission. Its refusal delta is measured, and any newly refused cell needs approval.
  - Probe consolidation runs as a decision-diff.
- **Probe failure is not a refusal (ADM-8).** A `cudarc` context-creation failure is `BudgetUnavailable`, handled by the route's posture.
- **Pure core.** The admission core is cfg-free in `sceneworks-core`, so decision-diffs run on Linux CI [APP-31].

### 4.4 Generation lane and job types (LANE-1…6) [APP-23…25]

- **APP-5, item 0, delivers the `JobType` table.** All job types (38 at the November 7 baseline) are classified as generation lane, `LaneHooks`, or utility, each with its admission owner and cache owner. The generation lane covers:
  - image, video, audio
  - VQA/interleave (SenseNova)
  - vector (StarVector)
  - image and video upscale (SeedVR2)
  - Iris restore
  - extend/bridge
  - person-replace
  - image detail

  `LaneHooks` covers training and captioning. Prompt-refine keeps its own evict-before-load cache. Depth and segment get their own classification in the table.
- **The generation lane** (inside the existing `run_utility_job_inner`):
  1. upgrade
  2. validate (local checks)
  3. `ModelSpec` snapshot
  4. routing table
  5. `engine.compile` (`RenderPlan`)
  6. admit (Plan)
  7. cache acquire
  8. admit (Run)
  9. run with one event consumer (builds on SW-F327 #3258)
  10. stage output files under the claim attempt and verify their identities
  11. commit ordered outputs and terminal receipt for that attempt, then emit completion
- **The backend module** builds on SW-F327 #3270's cfg alias.
- **Publication.** The current claim/attempt is the write authority. Staged media and document references stay private until complete. Finalize under the store's claim check, using an idempotent attempt/finalization key, a single authoritative artifact list and terminal receipt. Use the current jobs-store terminal transition and its durable pending-side-effects handoff. A lost acknowledgement returns the committed result on redelivery. Final asset indexing/visibility recovers idempotently from that manifest, including a crash between terminal acceptance and the catalog write. Use the existing [terminal-state and recovery owner](https://github.com/SceneWorks/SceneWorks/blob/e1ffa54df69a6d37b9efaa61f8a6dbce9d279ff8/apps/rust-api/src/jobs.rs#L2100). The receipt and ordered IDs cannot change on retry. A superseded attempt cannot commit. Cancel or failure before commit leaves no published partial result and cleans only owned staging. Preserve any product-defined partial result through an explicit typed outcome. Fake-store faults after first write, after last write, immediately before commit and after commit/before notification must preserve referential integrity, one terminal outcome and post-commit event order.
- **LANE-6 composition.** Target a worker-free API library and standalone server feature/profile. The desktop composition root remains the shipped `sceneworks-api` sidecar built with explicit worker support and the existing platform feature set. It supports normal API/in-process mode, `SCENEWORKS_WORKER_ONLY`, GPU preflight, Iris rank re-entry and desktop supervision. Assert Cargo edges on the API-only target/profile and package/relaunch the desktop binary separately. APP-6 owns the boundary and both build profiles.

### 4.5 Tooling (TOOL-1…4) [APP-26…28]

- **APP-1, item 0, handles DEC-16:** delete `routing/matrix.rs`, `config/backend-capabilities/matrix.json` and `exceptions.json`. Without that, every APP PR would trip the 61-source digest.
- **DEC-17:** delete the memory-matrix generator, its tests (`generate-memory-matrix.test.mjs`, `matrix-removed-sources.test.mjs`) and `docs/generated/memory-matrix.json`.
- DEC-16/17 are approved deletions, not completed work. Their owning features migrate consumers and retain required measured evidence/provenance before removing mirrors. No source-digest gate or replacement matrix is introduced.
- **The memory-adapter becomes one generic driver.** It covers `bin/mlx.rs`, `bin/candle.rs` and the auxiliary bins, and its workflow consumers in `memory-catalog-campaign.yml` are migrated. DEC-18 governs the single dispatched equivalence run.

---

## 5. The hook map

The hook enum in code is the source of truth (HOOK-1). The table below is the designed set.

| Layer | Hook point | Timing | Owner | Result-changing | Typical users |
|---|---|---|---|---|---|
| App options | `upgrade` · `validate` · `bind` · `effects` | replace/after | core | no | legacy shapes; all options; side effects |
| App admission | `candidates` · `price` · `select` · `refuse` · `stamp` | stages | core+worker | no | rungs, tiers (tier never changed, MEM-4) |
| App lane | `plan` · `admit` · `load` · `run` · `write` · `receipt` | stages | worker | — | non-generation jobs |
| Plan (S0) | `rewrite_plan` · `validate_plan` | replace/after | FeatureSpec/route | yes | low-res start, phases, Hires.fix, Wan MoE boundaries, LTX stages |
| Weights | 10 lifecycle points | before/after | lifecycle | no | progress, pins, iOS quanta, tier integrity, cancel |
| S1 prompt | `enhance` | replace | component | yes | LTX enhancer, FLUX3 PUP, SenseNova think |
| S2 encode | `materialize` | after | ExecPolicy | no | R1 |
| S3 refs | `grid_aligned(geometry)` | replace | conditioning module | — | control, init, masks, I2V `y`, keyframes |
| Pass boundary | `stage_bindings` · `prepare_context` · `estimate_endpoint` | declared stages/after | runner + components | yes | Hires final-pass enhancement, adapter-dependent preparation, typed low-res estimate |
| S5 init | `noise_source` · `init_blend` | replace/after | route/Feature | yes | parity RNGs, watermark; img2img, inpaint |
| S6 phase | `transition` | replace | **runner, Feature or route** | yes | resample, renoise, pixel round trip, LTX bit-exact |
| S6 phase | `enter` (slot, adapter scales, guidance) | after | runner | yes | MoE, per-phase LoRA |
| S7 step | `pre_eval` · `post_step` | before/after | Feature/route (FeatureHooks) | yes | caches; I2V pin, inpaint blend, H3 row freeze |
| S7 step | `combine` | replace | guidance library | yes | `CombineVariant`s |
| S7 step | `observe` | observe | sinks | no | preview, progress, PiD capture |
| S8 phase | `exit` | after | runner | — | history policy, context eviction, revert |
| Block | `pre_block` · `extra_kv` · `post_block` · `perturb` | around | conditioning/FeatureHooks | yes | ControlNet/VACE, IP/PuLID, STG, preview taps |
| Block | `stream`/`evict` | around | residency | no | R4 |
| Layer | `Linear` residuals with `ForwardArgs.adapter_scales` | inside | adapter runtime | yes | LoRA family, per-phase scales |
| Layer | `AttentionOp.policy` | inside | memory/Feature | within bound (R3) / yes (sparse) | R3, sparse attention |
| S10 decode | `tile_plan` · `decoder_override` · `progress` | replace/observe | codec | within bound / yes (PiD) | R2, PiD |
| S11 output | `post_process` · `emit` (terminal or repeated chunks) | after | Feature/route | yes | alpha, trim, inline upscale, audio fit; whole-clip Krea uses one terminal chunk; SW-F148 owns repeated decoded streaming |

---

## 6. Worked examples (files touched)

### 6.1 Low-res start: FEAT-1 (ships in ENG-1b, extended as routes migrate)

| Where | Change |
|---|---|
| Engine `[INF]` | `gen-core/src/features/low_res_start.rs`. **Params:** `{version: 1, k, scale}`. **Resolved constants:** first denoising PassId only; fixed final-pass-geometry schedule/mu; bilinear clean-latent resample; destination noise law; Reset. **Default estimator:** read-only endpoint x0; the completed-K previous-estimate candidate is separately declared in §3.1/DEC-10. No exposed `renoise` or `mu_policy` knob in v1. **Applicability:** supported spatial axes/grids/modes, endpoint estimator, transition, stream coupling and composed-plan checks (§3.1, FEAT-1). Edit/reference/control initially refuse until ENG-12 rebuild support; ordinary img2img init remains supported. **Plan rewrite:** split that pass's M executed steps at K, use reduced spatial dimensions for `[0,K)` and full pass dimensions for `[K,M)`. Other axes and later passes are unchanged. All transition allocations are priced. |
| Backend `[INF]` | None after ENG-1a, which builds `BundleOps::resample` and pack/unpack once. |
| Families | None after adoption. `prepare_step_context` uses its complete declared dependency key (PIPE-5), and model/guidance/Feature hooks honor boundary-estimate nonmutation and state invalidation (§3.3). Refusals come from composed route support. |
| SceneWorks `[SW]` | The `lowResStart` option entry, common parameter validation and web controls, all gated by the export. **Persistence:** register the versioned `lowResStart` block and emit outer share v2 when active. Extend the share reader/writer in the first feature introducing unrepresentable content. ENG-1b owns this for low-res, whichever app feature lands first. Exercise the actual old reader. Ordinary expressible shares remain v1. APP-2's upgrader later adopts this block as one of its shapes (C4-05). **Admission:** ENG-1b updates admission on adopted routes to the shared plan and pricing interface and to per-phase scope identities (§3.9). It ships with ENG-1b against the delivered #3264 baseline. |
| **Not the PAT-1 measure** | ENG-1b legitimately touches SceneWorks option, validation and web files. PAT-1 is measured on a later Feature (§6.4). |

**Version-1 recipe choices (approved DEC-10, R2-09).**

- M is the selected pass's executed step count after strength truncation. Valid K is `1 ≤ K < M`; default `max(1, floor(M/2))` for M ≥ 2; slider fraction K/M. K=0/M or M<2 refuses. Changing steps/strength to invalidate K produces an issue, not a silent clamp. Feature absent is the no-op.
- Scale is finite in [0.25,0.9], default 0.5. Spatial dimensions use `floor(full_dim * scale / legal_quantum) * legal_quantum`, then minimum, strict reduction and joint-grid checks. Invalid dimensions refuse with legal limits. Frames, audio duration/tokens and count remain unchanged. Native fixed tables permit only eligible interior boundaries.
- `EndpointEvaluation` remains the default. The precisely defined `PreviousEvaluation` protocol in §3.1 is an optional candidate for the declared supporting solvers/drivers, evaluated using the same bounded proof set. It retains the guided primary estimate after normal completion of all K steps. It is not an early-exit or terminal-step replacement.
- Persist resolved recipe/estimator identity and semantic version, target pass, dimensions, K, unchanged schedule/shift policy, transition arithmetic and seed derivation. Runtime provenance also distinguishes source outer-step/evaluation kind, sigma and configuration; do not serialize tensor data. Use the v2 outer envelope for active recipes; unknown identities refuse. Do not change estimator defaults for a persisted recipe or silently fall back to a different estimator on replay.
- The active recipe is intentionally result-changing. Disabled paths retain class (a). Estimate retention and any capture conversion cost are included in time/memory comparison; no fixed saving is claimed.

**Feature acceptance.**

- Tier 0 covers FEAT-1 boundary, composition, replay and refusal cases with mocks, including completed-K capture traces for Euler, Heun and DPM++ SDE, plus unsupported driver refusal.
- Tier 1 prices and checks the default endpoint forward, transition operation workspace and candidate estimate retention. Trace assertions identify which evaluations/draws/state commits occur; they do not infer equivalence from the final output size.
- Tier 2 reuses the feature's bounded proof set for prompt/content retention, structure, blur/seams, finite outputs and img2img retention, alongside full elapsed time and peak memory. Compare the candidate only after its solver contract is satisfied. A quality failure is repaired before that route advertises the recipe, or surfaced under DEC-13. No Cartesian campaign or fixed speedup promise.
- Execution reports count the default's extra read-only evaluation; the candidate records capture of its named primary estimate without an extra forward. Both report M completed outer steps. Persisted estimator identity and semantics cannot change on replay.

### 6.2 New equivalent-class memory measure (PAT-2)

An `AttentionOp` policy variant (per backend, or once under DEC-1 (c)), plus a `FeatureSpec { class: Equivalent { quality_bound } }` with its `memory_effect`. No family or SceneWorks edits.

### 6.3 New model in an existing class (PAT-3)

| Where | Files |
|---|---|
| inference | `gen-core-<f>` (host math if needed); `<backend>-gen-<f>` (graph from kit parts, `FamilyModel`, primary declarations, `RouteRecord` with component ids, `WeightPlan`, defaults and limits); catalog registration; Tier 0 conformance; Tier 1 vs the reference |
| SceneWorks | Product-manifest entry; evidence store: two measured anchors per tier (data, produced by one dispatched run) |
| Must be derived (or listed by decision) | LoRA family signatures (`lora_family.rs` `SIGNATURES`), converter registry entries, prompt-guide entries, sampler labels [APP-38] |

### 6.4 New per-step guidance operator (the PAT-1 measure)

- **Engine:** a `CombineVariant`/operator in `gen-core::guidance`, plus a `FeatureSpec { class: Recipe }` that sets it in `GuidancePlan`. It applies where the guidance surface includes CFG.
- **SceneWorks:** one `OptionSpec`; the UI is generated.
- **The diff is the PAT-1 measurement.**

### 6.5 Design trace: FLUX.1 low-res start (R2-01/06/09)

This and §6.6–6.8 are **weights-free design walkthroughs**, not executable tests, measured memory or runtime evidence. Implementation features turn them into the Tier 0 checks and dispatched proofs already required by MIG-3.

Take a t2i pass `base` with M=8 executed outer steps, K=4, full 1024×1024 and scale 0.5 on a route whose declared legal grid accepts 512×512. The schedule is resolved once for 1024×1024. `[0,4)` executes at 512×512 and returns noisy x at sigma[4]. At the boundary:

1. Enter the priced transition scope; check cancellation. Resolve the departing configuration's slot at sigma[4], ensure it and prepare its dependency-keyed context. Perform one read-only endpoint forward and convert its output to x0 using the declared sampling convention.
2. Unpack and bilinearly resize **x0** to the 1024-grid latent dimensions. Draw feature-domain target-grid noise. For FLUX flow, use `(1−sigma[4])*x0_resized + sigma[4]*noise`; never feed the noisy endpoint to this clean-sample formula.
3. Repack/cast to the declared solver latent representation, release transition-only buffers, Reset numerical history and incompatible geometry-dependent guidance/Feature state, retain the defined absolute stochastic coordinates, enter the full-size phase scope and rebuild the geometry-dependent context. The boundary estimate has not committed a guidance momentum/cache update.
4. Execute `[4,8)`, decode once and persist final output plus resolved recipe. Progress is eight outer steps; the report includes the boundary evaluation. Price all transition buffers even if the steady full-size phase appears larger in one measurement.

With the Feature absent, there is one full-size phase and no boundary estimate, feature noise draw or transition allocation. Img2img resolves strength first: if only three steps execute, K=4 refuses before load. Count > 1 follows the route's legacy sequential/batched relation; every sample's new transition stream includes that sample identity.

### 6.6 Design trace: low-res plus Hires.fix (R2-02/07)

Hires produces two stable passes before low-res contributes its span: `base` (M=8 at the caller's base size) and `refine` (M=4 at the existing Hires target size and img2img strength semantics). Low-res K=4 targets **base only**, giving base phases `[0,4)` and `[4,8)`. Refinement remains its own four-step pass; it does not receive another low-res start.

The runner invokes base prompt prep with enhancement disabled, encodes the unenhanced prompt, binds the original base conditioning, initializes and denoises. It then performs the existing RGB decode and pixel-fit operation with unchanged numerical variants. The intermediate artifact is not persisted and its decode notification is hidden as today. The pass boundary records that pixel output as the refinement pass's single reference, removes original multi-references/mask and preserves the existing PiD/output policy. The refinement pass applies the existing enhancer once, publishes its one report, re-encodes any changed prompt, encodes the new reference and runs its independent schedule. Encoder/codec/denoiser release/reload order comes from the declared live graph. Identical stage dependencies may reuse results; changed prompts or references may not.

User progress runs monotonically through the legacy combined outer-step total (8+4 in this example), with no extra tick for the low-res endpoint evaluation. Seed derivations, event visibility, original controls and final output channels match the legacy two-pass contract. A fake enhancer/generator can verify exact prompt/conditioning/event order. A conflict at an interior base boundary is checked within `base`; Hires's separate pass boundary does not overwrite it.

### 6.7 Design trace: Krea preparation and request state (R2-03/04)

In one pass, consider base-only → adapter A → base-only phases at the same geometry, with the middle phase enabling CFG. Encode negative context once if any phase needs it. Preparation keys distinguish effective base versus adapter A's aggregator scales and branch layout. The first base cache cannot be used for the adapted CFG phase. A final base phase can reuse the first base preparation only if all its dependencies match and that allocation was explicitly retained/priced; otherwise it rebuilds. Adapter scales are passed to both preparation and forward, never written into the shared resident model. A second request with adapter B has an independent execution/cache namespace.

Each Krea slice retains its native declared solver Reset, with its legacy RNG-coordinate policy recorded rather than inferred from local loop indices. Separately, an inert split on a route that declares Carry keeps one solver state, the prior interval and the absolute draw sequence; its trace equals the unsplit pass. A changed solver with unsupported Carry refuses. This separates fidelity of existing Krea phase semantics from continuity of the generic runner.

### 6.8 Design trace: Wan/H3 evaluation state and concurrent memory (R2-02/03/05)

For native Wan MoE, expert ranges share one mutable scheduler; the sigma-keyed model rule selects the correct expert **per evaluation** on curated solvers. At a low-res boundary coincident with an expert threshold, evaluate the endpoint rule rather than assuming the outgoing expert, prepare that slot's context under departing guidance/scales, perform the typed geometry transition, then Reset history. At an expert-only boundary, preserve the native history policy and all absolute stochastic draw indices. The staged loader is cancellable and its outgoing/incoming lifetimes are priced.

For H3, one outer step maps to `(sigma_video[i], sigma_audio[i])`, invokes one joint model forward and advances both streams under the route's coupling. A spatial low-res contribution changes video spatial axes only; it neither scales audio tokens/duration nor compares the two sigma numbers to order boundaries. If the route cannot supply the required read-only coupled endpoint estimate, the active recipe refuses under the declared support/DEC-13 process; the native route still executes. Any extra audio result from a joint endpoint forward is transient workspace, not a second audio transition.

Illustrative allocation arithmetic, **not a device estimate**: weights=100 units, live video=40, live audio=30 and joint workspace=20 means a simultaneous peak of 190, not `max(160,150)=160`. A budget of 180 therefore refuses this candidate although either isolated stream fits. A 40-unit transition buffer raises the live peak to 230 only where its lifetime overlaps; an alias of the existing video buffer adds zero. Component/operation pricing also includes output buffering, retained caches and temporary casts. This fixture checks aggregation/deduplication; actual bytes and policy margins still come from the route contract, evidence and backend budget hooks.

### 6.9 Production chunk adoption boundary (R5-03)

- **SGA owner, ENG-6:** route each existing Krea whole-clip result through the shared `emit` contract as one terminal chunk (`sequence=0`, `final=true`). The existing collector/event adapter unwraps it into the same public `GenerationOutput`, final events and persisted artifact. Internal chunk metadata is not a new wire format or a claim of interactive streaming. Transfer/reference the existing payload without a second full-clip copy. This permanent external-boundary mapping replaces the old batch finalization path in the same feature; it is not a parallel legacy path or temporary cross-feature bridge. Batch memory/residency and count behavior remain the baseline.
- **One implementation:** terminal and repeated chunks use one ordered output/sink primitive with bounded retention and cancellation/backpressure. The batch route is its production user; a mocked multi-chunk producer tests repetition and faults without claiming production multi-chunk delivery. No separate dormant `stream_chunk` hook, streaming codec implementation or unused registry entry is installed solely for the mock. This satisfies HOOK-1/AP9 without an exception.
- **Streaming owner, SW-F148 (`sc-8432`):** retains its spike, interactive session/transport and decoded streaming/re-anchor product scope. If its streaming path exists before ENG-6, ENG-6 migrates that real consumer in the same feature. If SW-F148 delivers later, its implementing feature adopts the existing shared contract and adds its stream-only production hooks with their first consumers. MIG-7 records which order applies; SGA does not duplicate the product work.
- **NO-GO:** SW-F148 may decide NO-GO. ENG-6 can still finish with unchanged batch output through the production shared primitive and mock multi-chunk conformance. The architecture supports repeated output, but no native decoded-streaming completion is claimed. No orphan hook survives solely because an external epic was planned.
- **Closure:** fake-output contract traces compare legacy versus wrapped batch payload/event ordering; multi-chunk traces cover ordering, bounded queues, backpressure and cancel; the owning feature validates any actual decoded streaming path. These are planned implementation checks, not runs performed by this document revision.


---

### 6.10 Conditional stages and owning-thread execution

At ENG-1a define a bounded native region with a finite set of stage templates and typed guard results. SenseNova's text decoder chooses “stop” or “image”. The runner owns repeated text rollout, image denoise, generated-image feedback into the conditioning caches and ordered output. The compiler fixes geometry, remaining-token budget, maximum images and denoise steps, and reserves for the largest admitted image invocation, cumulative KV/cache and retained output at those limits. It records possible paths before load and only realized stages in the receipt. A driver cannot add a template, increase those limits or consume another pass's budget. ENG-6 binds the production SenseNova driver to this same contract. Within a conditional region, a Feature targeting the first denoising pass binds to the first realized image invocation only. Zero images consume no image Feature work. Require finite priced templates and an unambiguous role before enabling the combination. Charge worst-case retained KV and output through the final possible feedback.

Example with maximum three images and four steps per image:

- End token before any image produces text, zero image invocations and no image progress bar.
- Two image-start tokens execute text → image 0 → feedback → text → image 1 → feedback → text → end. Retain causal state, charge both retained outputs and count eight actual outer steps. The existing public bar spans 12 and emits its terminal fill to 12 for an early finish. Record the fill separately from completed steps. At the image/token limit, finish without another image.
- Cancellation during second-image denoise or feedback stops future invocations, releases attempt-local state on its owner, and follows the partial-publication rule in §4.4. The next request cannot see the canceled caches.

**Executor trace.** Registration and cancellation messages can cross threads. Load, prepare, seed/draw, dispatch, materialize and destroy run on the selected device-owning executor. For a provider using global RNG, submitting seed 22 while seed 11 is running queues the entire second trajectory. Expected order is seed 11, all its draws/materialization and finish, then seed 22 and its draws. Each result equals its isolated trace. An explicit per-request RNG provider can opt in to safe overlap only after the same two-trace comparison checks adapters, state, kernel ownership and teardown. A poison event retires the executor under the existing supervisor without new GPU calls. ENG-1a owns the mock and the existing worker-entry validation when this boundary changes.

### 6.11 Route population and first-wave owners

Build the old side from the shipped job dispatcher/API modes, native bundle/registry surfaces and actual bespoke worker selectors, independent of the proposed export. Join on platform, backend, stable provider, public operation/mode and conditioning meaning. Separate native/imported routes and output variants whenever implementations differ. A descriptor's generic “reference” bit alone cannot distinguish an init image from an IP adapter. Include direct engine/external bundle consumers and supported CPU variants even when no SceneWorks job exposes them. For a product row, follow the entry point into the selected implementation. For an engine-only row, retain its actual external entry point. Record unsupported cells as negative witnesses. A present registry ID is insufficient evidence of execution or device fit.

At the November 7 baseline the 38 `JobType` variants all have an explicit dispatch arm. The packaged media descriptors enumerate 75 MLX and 63 Candle IDs. There are 21 Candle audio IDs, four imported family categories in the existing app matrix, and a separate iOS subset of seven generators plus its transform surface. The sets overlap; summing them would double-count routes. Expand each applicable cell using its provider-specific operations/conditioning, keep the origin and assign the owners below. Consume the existing matrix as a *baseline input* before DEC-16 removes it. The reconciliation is a test over typed sources and public entry points; there is no new checked-in capability matrix.

Two checks establish why the join matters. On both backends StarVector is registered in the runtime snapshot's text-LLM surface rather than the media generator list. In the Candle runtime snapshot four Qwen edit aliases point to `qwen_image_edit`, absent from the media-generator list. Its [worker selector](https://github.com/SceneWorks/SceneWorks/blob/e1ffa54df69a6d37b9efaa61f8a6dbce9d279ff8/crates/sceneworks-core/src/jobs_store/routing/candle.rs#L1565) preserves the actually supported conditions. ENG-2 owns Qwen edit behavior and ENG-7 owns registration cleanup. Neither a missing media ID nor a mapped alias alone decides public eligibility.

| Public or directly callable cohort | Conditional behavior and product output | Engine migration / app acceptance |
|---|---|---|
| FLUX.1 dev/schnell T2I on linked desktop MLX/Candle | RGB image. Base T2I starts from noise. Plain img2img has no base witness on these providers. | ENG-1a, then applicable low-res in ENG-1b. |
| FLUX.1 XLabs IP, dev strict ControlNet and PuLID | Resolve each actual MLX or Candle selector independently. Identity/structural conditioning stays distinct from an init image. Preserve composed-mode refusals. | ENG-1a migrates each shipping route, including bespoke dispatch parity. ENG-7 registers any residual bespoke entrypoint. |
| FLUX.2 MLX dev plain img2img (native public generator); SD3.5 Medium MLX (SceneWorks public job) | Source-image VAE init and strength blend, RGB image. Other edits and Candle Klein's reserved reference modes have their own keys. | ENG-2, trace §6.13. |
| Other catalog image and video generation | Public image/edit and video modes resolve via runtime maps, including count, alpha, conditioning and AV outputs. Match each provider/mode to the named shared-driver or native cohort in §9.2. Imported FLUX.2, Krea, Mage and SDXL use their actual selectors. Native and curated SDXL, Kolors and Wan are split by loop kind. | ENG-2/3 shared drivers, ENG-4b Krea, ENG-5/6 native/specialized, APP-5 image and APP-6 video. |
| SenseNova VQA and interleave | Preserve text-only responses and documents referencing the actual ordered image/text segments. Admit conditional feedback (§6.10). | ENG-6 driver, APP-6 ordered document publication. |
| StarVector 1B/8B on MLX and Candle | Native image-to-SVG through the token driver, existing finish reason, sanitized canonical SVG, derived preview and terminal metadata. Other vector modes keep their own eligibility/refusal. | ENG-7 owns registered output contract and native binding; APP-5 owns public vector lane and artifacts. |
| Audio registry and audio jobs | Typed audio outputs. Flow/diffusion providers use the runner. Autoregressive token providers keep token generation and shared lifecycle. | ENG-14, then APP-6. |
| iOS and external bundles | iOS image registry, including Sana/Sprint, Anima Turbo, SDXL, Iris and Z-Image Turbo/control; separately registered transforms, including Iris restore. Record device admission independently. ChatWorks, Art and mobile get compatibility checks at the feature changing their contract (XL-4). | Respective graph cohort, ENG-7 registration, ENG-13 iOS, consuming feature for external compatibility. |
| Bespoke dispatch and non-denoising jobs | Expand native image detail/restore/upscale, extend/bridge/person-replace and other product entrypoints from dispatch. Record existing artifact types, cache/admission owners and typed refusals. Classify training, downloads/imports and utility jobs by their actual lifecycle (LANE-1); a utility classification never implies a denoising migration. | ENG-7 owns residual bespoke registration and contract gaps; APP-4 admission, APP-5 image/vector, APP-6 remaining lanes; ENG-8 and subsequent owning features qualify each trainer before graph removal. |

**Cutover test.** Every legacy supported row must resolve to its owner, candidate binding, output contract and AP13 evidence, or an approved per-route exception. An unknown owner fails inventory publication. Deliberately remove StarVector's binding while keeping its old dispatch witness: reconciliation must name the missing provider/mode. A new catalog row likewise requires a consumer or legitimate engine-only disposition. ENG-6 closes only its assigned cohort. Later owners close theirs independently.

### 6.12 A byte bound for the first FLUX.1 recipe

For SC-15823, Candle `flux1_dev`, q4, 1024×1024 T2I, batch 1, 8 steps, seed 42, Resident, the recorded live-allocation high-water is **17,651,074,970 B**. The artifact is `SceneWorks/flux1-dev-mlx@323fd12d79f78ad444e882e8d8e871914584f2b9`, inventory `38892dfdd0177068e4996834a3ef6666309db5a0034f2548bd13f814e742b341`. It covers the complete ordinary render. Its repeated “conditioning/denoise/decode” values denote that same total. Source, captured at inference `5f973a73bf`: [SC-15823 evidence](https://github.com/SceneWorks/SceneWorks/blob/e1ffa54df69a6d37b9efaa61f8a6dbce9d279ff8/docs/flux1-candle-memory-ladder-sc-15823.md#L33).

Take K=4, scale 0.5. The 16-channel unpacked f32 latent at 512² is 262,144 B, and at 1024² it is 1,048,576 B. For the resize/re-noise suboperation, a bounded implementation may hold at most two reduced buffers (source and unpacked clean) and four full buffers (resized clean, independent noise, blend result and destination pack/cast). Counting each as f32 and all six simultaneously bounds **those buffers** at **4,718,592 B**. Alias elimination can only reduce that subtotal. This layout is an implementation condition: extra backend scratch, RNG staging/host transfers, retained snapshot or rollout state and an endpoint forward each require their own covered bound and budget domain. The existing total already includes ordinary weights and buffers; do not add them again or subtract fictitious phase measurements.

For a stage whose matching ordinary live set is proven bounded by B, the above layout needs at most B + 4,718,592 B **only if** its other increments are covered or zero. With the recorded value as a numerical fixture B, the cutoff is 17,655,793,562 B of effective device budget after reserve/margin; equality admits that fixture and 17,655,793,561 B refuses it. In the real record the endpoint snapshot/workspace, small-grid operation and changed live set are **unclassified**. The measured high-water alone does not prove B for the new stage. Until a source-derived upper bound or the planned bounded component proof closes every operation, the actual new cell returns a typed unpriced-recipe refusal even above that illustrative cutoff. The original T2I posture remains intact. ENG-1b may claim availability only on cells with a reconciled bound and quality proof; lack of any usable first cohort is an explicit FEAT-1 blocker, never a successful rollout.

### 6.13 Plain img2img initialization at reduced spatial resolution

For MLX FLUX.2 dev, take a 1200×700 source, final pass 1024×768, scale 0.5, 10 native steps and strength 0.4. FLUX.2 computes `start_step = floor(10 × 0.4) = 4`, leaving M=6 executed steps on the full-geometry schedule. With K=3, phase one uses 512×384, phase two 1024×768. Bind the single source as ordinary init; an XLabs identity reference or strict control would follow its own applicability rule.

Apply this route's native Lanczos pixel fit to **512×384** and normalize input to [-1,1]. Run its mean VAE encode, transpose, even crop/grid match, patchify, codec normalization and pack at that grid. The initial latent is 64×48, then the packed token grid is 32×24 (width × height). Blend it with the route's per-sample seeded noise using the full-pass sigma at node 4. Execute nodes 4→7 at the reduced size, perform the declared clean-sample transition at node 7, then execute nodes 7→10 at full size. Initial encode, retained clean init, noise and first forward have their own admission scopes. Price the source pixel buffer as well as the fitted input and actual codec workspace. Persist init semantics in recipe v1 and prove through this route's ordinary entrypoint; lower-strength and legal-grid refusal cases use the same order. [Init implementation](https://github.com/SceneWorks/inference/blob/2b09938e33370915d4f853389961709b8bbf8bf9/crates/media/mlx-gen/mlx-gen-flux2/src/model.rs#L885)

The native FLUX.2 proof enters through the exported dev generator, as in its [img2img example](https://github.com/SceneWorks/inference/blob/2b09938e33370915d4f853389961709b8bbf8bf9/crates/media/mlx-gen/mlx-gen-flux2/examples/flux2_dev_img2img.rs). The public SceneWorks proof uses SD3.5 Medium MLX through the normal T2I job with a reference asset and strength. The [worker](https://github.com/SceneWorks/SceneWorks/blob/e1ffa54df69a6d37b9efaa61f8a6dbce9d279ff8/crates/sceneworks-worker/src/image_jobs/conditioning.rs#L1721) and [provider](https://github.com/SceneWorks/inference/blob/2b09938e33370915d4f853389961709b8bbf8bf9/crates/media/mlx-gen/mlx-gen-sd3/src/model.rs#L390) both implement an init-image path. Use SD3's own sigma, fitting and normalization rules there. A direct FLUX.2 witness makes no claim about which UI mode reaches it.

### 6.14 Hires.fix last-use owners

| Existing production consumer under `sceneworks-worker/src/image_jobs/` | Cutover and deletion owner |
|---|---|
| `mlx.rs` and `candle.rs`, general base dispatch and surface-aware rendering | Each engine cohort switches its own supported rows in ENG-2…6. ENG-7 owns the final audit and removal of the legacy generic call sites. |
| `krea_imported.rs`, admission inputs, imported execution and specialized two-pass path | ENG-4b migrates imported Krea using its public request. Preserve phase/pose conflicts and enhancement behavior. |
| `sdxl_imported.rs`, imported admission and two-pass execution | ENG-5 migrates imported native SDXL and preserves its conditioning/refinement refusals. |
| `memory.rs`, refinement reference-count helper | Keep pricing live for unmigrated rows. ENG-7 deletes its last legacy dependency after cohort pricing is active. |
| `render.rs`, resolver, reference-count and both Hires generation helpers | ENG-7 deletes the shared implementation after every production caller above has moved. Update the structural/behavior tests in the same change. |

Compare old/new imported Krea and SDXL at their respective exits: admitted dimensions/conditioning count, unenhanced base prompt, final-only enhanced prompt, fitted single refinement reference, seed/step events, stored final output and absence of a public intermediate. At ENG-7, search for the legacy type, resolver, reference-count and render helpers, compile both backend call paths and confirm zero surviving production calls. Any newly discovered live caller joins its own cutover and the final deletion remains with its last actual consumer.

### 6.15 Whole-job work limits

After composing native and user passes, calculate with checked integer arithmetic the sum of executed outer steps and maximum solver evaluations per pass, including correctors, read-only endpoints, conditional region maxima and sample/count semantics. Budget total output bytes across retained artifacts and separately price the maximum simultaneous working set. Cap expansion **before** allocating a vector from untrusted input. Counts, sums and products that overflow or exceed the common bound fail pure validation at browser, HTTP, MCP and worker preflight. Job limits are versioned product values chosen against supported legacy maxima at ENG-16 planning. A pre-existing native request remains covered by its existing limits.

For a fixture cap of 36 evaluations, three sequential samples × two passes × (two steps × three evaluations) uses exactly 36. A single extra job-wide evaluation would yield 37 and refuse. A per-sample endpoint adds three and refuses at 39. Test retained output exactly at/one byte above its cap and an overflowing count product with identical issue codes at every boundary. These are fixture limits, independent of the final product values.

## 7. Migration strategy and coordination

### 7.1 Rules

1. **Route-at-a-time strangler.** During a feature, the runner hosts migrated routes while legacy routes keep their `generate`. Cut over an explicitly named provider/mode/backend cohort. Delete the old path for that cohort in the same feature. Native/curated, imported and bespoke rows can have different owners. A shared helper keeps serving only unmigrated rows until the named last consumer deletes it. No row may retain two production paths at feature exit.
2. **The producer migrates its consumers (AP9).** A feature that replaces a source migrates every consumer in the same feature, in both repos. Example: ENG-7 replaces the capability dumps, so ENG-7 migrates SceneWorks' dump consumers, including APP-1's static layer if APP-1 has already landed. That feature's `[SW]` items are then gated by DEC-5.
3. **Characterize before consolidating** (§7.2).
4. **List behaviour changes; never let them slip in.** Each change is a named PR entry with its AP13 class and evidence. Class (c) needs sign-off.
5. **Order within a feature:** ABI, fingerprint and evidence-key changes first; then the walking skeleton; then mechanism slices; then the terminal item (pin, Tier 1/2, whole-branch review, merge).
6. **File ownership.** Each feature's plan includes a file-ownership table. Features that overlap in time own disjoint files.
7. **New-model rule.** Once ENG-1a lands, new families target the runner. Once the kit, weights and components features land, new families use them.

### 7.2 Characterization tiers (MIG-3)

| Check | Tier | Pins |
|---|---|---|
| Plan snapshot | 0 | `RenderPlan` per route over a request corpus |
| Runner trace over a mocked backend | 0 | Pass/stage/phase/hook sequence; Hires prompt/conditioning/event equality; Wan MoE trace equality with legacy; §6.5–6.8 |
| `FamilyDriver` and Feature conformance over a mocked forward | 0 | Every `DriverSupport` and geometry-change claim [ENG-16, ENG-21] |
| Executor and RNG isolation | 0 | Owner-thread lifecycle, queued global-seed providers and opt-in safe interleaving. Compare two isolated request traces (§6.10). |
| Boundary-state and split conformance | 0 | Flow/VE noise law and estimate provenance; inert split versus unsplit history/draw coordinates; incompatible Carry refusal; read-only guidance/Feature state and geometry invalidation |
| Context dependency conformance | 0 | Krea base/adapter/base, CFG-off/on and request isolation; retained allocation pricing |
| Fault injection | 0 | Error classes × stages (PIPE-15) |
| Weight-plan receipt | 0 | Header-only compile and bind receipts |
| Engine-request equivalence | 0 | SceneWorks-built requests recorded by a fake generator |
| Admission and routing decision-diffs | 0 | Per (route, tier, geometry, budget); cfg-free core runs on Linux CI |
| Component/operation pricing | 0 | Concurrent/joint evidence without double-counting, learned-transition workspace, total versus activation-only basis, aliases, count > 1, selected rungs and phase/transition scopes at budget boundaries |
| Receipt/`raw_settings` shape | 0 | Shape assertions |
| Primitive op checks | 0 | Kit ops on synthetic tensors |
| Base-versus-head equivalence | 1 | Same runner, same job, shape-faithful random weights (real dims, truncated depth), in-job comparison, no committed goldens |
| Real-weight proof | 2 | Proof-set A/B for fidelity, s/step and peak memory (DEC-15) |

### 7.3 AP13 expected classes per feature (initial; refined at planning)

**Approved DEC-7 policy (2026-11-03):** prefer bit-exact results without an open-ended GPU repair loop for small differences. Before numerical validation, declare the component metric, bound and output checks, the finite proof set and investigation budget. Reuse applicable evidence and the feature's already planned Tier 1/2 runs; this adds no campaign or dispatch cadence. “Less than 1%” is not a universal tensor/image tolerance.

- Diagnose a mismatch with source/operation-order, dtype, seed and state inspection first. A valid comparison inside the declared bound with acceptable output evidence supports class (b), without more GPU runs solely to chase zero error.
- Repeat learned checks only for a concrete unresolved correctness/measurement question or a numerical change affecting that evidence. Unchanged evidence is reused; no post-hoc widening of a limit to pass a result.
- At the planned investigation limit, report the measured difference, confidence and accept/fix/retain-existing-path choice. Material output/replay/capability changes remain explicit; class (c) still needs route approval. Pilot exceptions to BACK-5 remain explicit rather than silently becoming tolerance-based.
- Component-specific limits and the run budget are specified at feature planning; policy approval grants no blanket route re-baseline or pilot-gate waiver. Learned execution remains separately dispatched under DEC-14/15, never ordinary CI. This decision authorizes no execution during design review.

| Feature | Expected class | Notes |
|---|---|---|
| ENG-1a, ENG-2, ENG-3, ENG-5, ENG-6 (pipeline adoption) | (a) | Loops, combines, seeds and history are preserved bit-exactly by design |
| ENG-4a (primitive extraction, media/LLM projections) | (a) target, per projection variant | Preserve existing codec, operation order and fused/trainable paths; an unavoidable change is listed and handled under AP13 |
| ENG-4b (ScaleContext, per-phase adapters) | (a) for fold-kept routes; Krea candle per-phase `[M]` (b)/(c) | Krea candle re-adapts in place today; preparation must receive phase scales |
| ENG-7 (route records, export) | (a) | No numerics |
| ENG-8 (kit II) | (a) for `AttentionOp` with the same kernel; (b)/(c) for norm, RoPE and sinusoid swaps, listed per family | |
| ENG-9 (weights) | (a) | Bit-identical loads |
| ENG-10 (codecs) | (a) target; (b) where kernels merge | |
| ENG-11 (encoders) | (b)/(c) likely | Quant codec and mask differences |
| ENG-12 (BlockStack) | (a)/(b) | |
| ENG-13 (memory contracts, iOS) | (a) | Pricing changes go through ADM-7 decision-diffs |
| ENG-14 (audio/LLM) | per route | |

### 7.4 Pins and branches

- **Mirrored `feature/*` branches.** One coalesced pin bump per feature, inside its consuming SceneWorks change.
- **Concurrent inference features:** the first to land merges into inference `main`; the other merges inference `main` before its final pin. Then run `cargo metadata --locked`. The coordinator of the second-landing feature resolves the conflict [APP-6].
- **Pin sites:** run `bump-inference.mjs` and verify that no site anywhere (8+ plus the lock) still holds the old revision.

### 7.5 Delivered dependencies and current overlap

As of November 7, SceneWorks main `e1ffa54df6` includes SW-F327 merge `40bac3f2d7`, the #3258 stream supervisor and #3264 admission work. #3268 and #3270's delivered changes are on main. Preserve their owners and behavior when extending the shared lane, admission and structural tests. Inference `2b09938e33` already accepts CodeGraph feature/story branch names in `scripts/ci/feature_epic_policy.py`. Reuse this policy at ENG-1a.

| Surface | Rebaseline before the first touching feature |
|---|---|
| Hires.fix and Krea | Use the actual consumers and per-route deletion sequence in §6.14. |
| Product registry and controls | Include SW-F330 appearance/share behavior and expanded sampling from PRs #3335/#3334 in the before-side fixtures (§10.3). |
| H3 memory | SW-F329 shipped the reference and policy pair in PR #3331. §8.1 specifies impact and evidence reuse. |
| Krea streaming | SW-F148 retains product streaming, including a permitted NO-GO. ENG-6 adopts its delivered consumer if present, and the whole-clip terminal path either way (§6.9). |

DEC-5 still applies to new overlap. Verify current main ancestry before an overlapping change, preserve worker-facing pin compatibility, and record who owns any new route. The old November 3 wait on SW-F327 is satisfied.

---

## 8. Validation and proof plan

### 8.1 Proof set (both backends where shipped)

| Route | Why |
|---|---|
| FLUX.1 dev/schnell base T2I, plus separately inventoried ControlNet, XLabs IP and PuLID | Preserve packed tokens, guidance, conditioning and existing refusals per backend. Plain img2img is outside the base cells. |
| FLUX.2 MLX dev native plain img2img, plus SceneWorks SD3.5 Medium MLX reference-guided T2I | Use the native real strength-dependent init encoder and include non-square source fitting (§6.13). |
| SDXL native + curated + inpaint, plus InstantID | UNet, ε, ancestral parity loop, inpaint blend, global RNG |
| Qwen-Image t2i + edit | Reference tokens, norm-rescaled CFG, candle merge-at-load |
| Z-Image Turbo + Base | Few-step, C,T,H,W image latent |
| Krea Raw multi-phase | Phases, per-phase adapters/guidance |
| Wan 2.2 A14B T2V (native and curated) + I2V | Video, MoE with σ-keyed slot and history carry, I2V `y` |
| LTX-2.5 distilled + dev | Two streams, explicit-table span, route transition, learned upsampler, STG branches |
| MiniMax-H3 T2VA and the shipped SW-F329 long-clip CUDA case | Test two coupled schedules. Evaluate changes against the matched sparse/window and dense-fallback envelopes below. |
| FLUX.3 Klein distilled + base | Flow-map driver; Qwen3-VL encoder on llm |
| Boogu Turbo; MOSS-Image | DMD drivers, model-dtype combine variant |
| Krea-realtime | Autoregressive streaming driver |
| Iris | iOS family, FlowDPM driver, quanta |
| One audio route (Stable Audio 3 or YuE2), from ENG-14 | Audio adoption |
| Mage-Flow, when touched | Watermark. Weights exist only on `nax-macos-2` and the SMB share |

**H3 operating envelope.** ENG-6 (driver), ENG-8 (attention/kernel), ENG-12 (block windows) and ENG-13 (admission) check the affected portion of the accepted q4, 1344×768, 345-frame, 24 fps, Turbo eight-step, Veda-enabled request with generated stereo audio. Retain the physical GPU identity, requested adapters, schedule and precision. The small arm uses a 21 GiB private pool and 3 GiB external reserve. The 95 GiB Resident control uses a 92 GiB pool and 3 GiB reserve with the same explicit 32 MiB cuBLAS workspace policy as the sparse 24 GiB arm. Keep the original request's Veda setting and effective sparse/dense path distinct. Assert actual text-encoder **and** DiT transfers and sparse calls on the windowed arm. Check dense fallback against its own cap. Compare sparse outputs only to equivalent sparse controls. Reuse [the matched control](https://github.com/SceneWorks/SceneWorks/actions/runs/42335898368) and the [terminal contract](https://github.com/SceneWorks/SceneWorks/blob/e1ffa54df69a6d37b9efaa61f8a6dbce9d279ff8/docs/testing/sw-f329-cuda-memory-terminal.md). A changed numerical or lifetime assumption selects the smallest affected witness under DEC-7.

**Training.** Before ENG-8 removes a graph, freeze trainer variant, seeds, batch, target names, optimizer state and precision. Reuse the existing FLUX.3 MLX/Candle gradient/update and checkpoint oracles and each affected trainer's equivalent fixture. Compare trainable/frozen sets, nonzero expected gradients or matched deltas, unchanged frozen parameters, and next update after save/resume versus uninterrupted. Capture forward parity, update parity and resume separately in the existing dispatched tier. The optional ENG-15 pilots have a different purpose.

### 8.2 Where runs happen

- **Approved DEC-14/15/18 scope:** Tier 1 is per migration batch; Tier 2 is the affected proof subset at feature end; APP-7 includes one bounded old-vs-new memory-adapter proof on each designated backend, combined with applicable feature-end evidence. Declare finite proof/investigation budgets and duration before dispatch, reuse unchanged evidence and apply DEC-7's rerun criteria. The earlier 1–3 runner-hour estimate per Tier 1 batch is low-confidence, not a measured runtime or permission to spend it.
- **Before actual dispatch:** discover both organization and repository runner scopes, group/repository access, labels and workflow restrictions; acquire the required shared-resource holds. Dated host names below describe routing intent, not current capacity. No dispatch, GPU loan or model execution occurs during this design-only approval.
- **Tier 0:** ordinary PR CI.
- **Tier 1 and Tier 2:** dispatched, with evidence recorded.
  - **Workflow change:** add an MLX route to `model-inference-proofs.yml` targeting `nax-macos-2` via an `rw-*` label. Today it targets hosted `macos-15`, and its `nax` label also matches the dev Mac [APP-30].
  - **Candle:** a CUDA runner labelled `real-weights`.
- **The dev Mac GPU** is used only on an explicit loan, with the duration stated.
- **Every MLX run** has an external RSS guard and a stated memory cap.
- **iOS.** ENG-1a records same-target/profile/architecture binary baselines for media-enabled and LLM-only builds. ENG-13 replays cold, resident and falling-headroom admission on the existing fixture families. Changes to footprint, incremental headroom or quantum behavior require applicable physical iPhone/iPad evidence for that OS/device/profile, including a post-admission headroom drop. Reuse a matched receipt or take one bounded affected-device witness. Simulator and desktop results have their own scopes.
  For a fixture with 600 MiB demand, footprint 200 MiB at admission and 400 MiB at recheck, only 200 MiB of growth is credited. Remaining demand is 400 MiB and projected footprint is 800 MiB. With an 800 MiB cap, 400 MiB available fits; 399 MiB refuses. Cold with no growth requires the full 600 MiB. Credit a known reusable model only through the route's existing reuse evidence. An unrelated resident object cannot lower the cold demand. Exercise both release/media and release/LLM-only builds against their own pre-feature artifacts.
- **Candle-gated SceneWorks tests:** one `windows-candle.yml` dispatch per batch (constraint 7).

---

## 9. Roadmap

### 9.1 Conventions

- **Item tags:** `[INF]` identifies inference work; independent items can proceed once implementation is authorized. `[SW]` overlap is checked against approved DEC-5's main-delivery gate.
- **Feature size:** each feature is ≤6 weeks (approved DEC-12). Split larger estimates at complete consumer boundaries during planning; if the branch window is exceeded, re-plan while retaining full scope and terminal acceptance.
- **Estimates:** agent-time at **low confidence**, re-estimated with `codegraph-plan`.
- **Slices are sketches.** Each feature becomes 5–12 mechanism slices with 1–3 local acceptance criteria.
- **Every feature's exit** is stated against route-inventory rows (PIPE-18).

### 9.2 Engine features

| Feature | Scope (slices sketched) | Depends on | Est. |
|---|---|---|---|
| **ENG-1a: Runner core + FLUX.1 migration** (walking skeleton; no user-visible change) | **0** `[INF]` prerequisites: verify the delivered CodeGraph-key branch policy or land its missing prerequisite PR (DEC-11); the independent route reconciliation (§6.11); source-identified baselines (LOC census, fact audit, iOS slices); the Tier 1 workflow route on `nax-macos-2`; the prior-art salvage review. **1** `[INF]` the pass/phase `RenderPlan` with stream bundles; typed boundary states; request-owned solver, RNG and guidance state (`RunState`); dependency-keyed preparation; whole-plan validation, executor/RNG ownership and a bounded native-region contract exercised by the SenseNova design fixture (§6.10); `BundleOps`; the runner driving `MemoryRequestScope` with actual phase identities; the error and count contracts. **2** `[INF]` **every shipped FLUX.1 provider/mode on its actual backend** (base T2I and existing ControlNet, XLabs IP and PuLID routes; no fabricated base img2img) moves onto the runner, class (a), and its legacy orchestration is deleted. **3** terminal. **Complete consumer boundary:** FLUX.1 runs entirely on the runner and the old path is gone (AP9). | — | 5–6 wk `[L]` |
| **ENG-1b: Low-res start (first Feature)** | **1** `[INF]` `FeatureSpec` registry; parameter and composition applicability; the `features` field on the existing dump format; typed refusal, which fixes the silent `phases` ignore. **2** `[INF]` low-res recipe v1: `EstimateSource` (endpoint default; candidate completes K steps and retains named primary x0), the typed noise law, aggregate operation evidence plus transition buffers, scope identities; bounded quality/time/memory and replay acceptance (§6.1, §6.5). **3** `[SW]` existing admission consumers move to the plan and pricing inputs; the `lowResStart` option, validation, versioned recipe block, outer share v2 old-reader refusal, UI and exact-replay identity; prove the scoped byte bound or refuse the new recipe (§6.12). **4** terminal. | ENG-1a; DEC-3 approved; DEC-10/13 approved; specific route refusals reviewed at planning; DEC-5 baseline refreshed (§7.5) | 4–6 wk `[L]` |
| **ENG-2: Shared-driver image wave** | flux2, chroma, lens, anima, sd3, sana base, boogu base, qwen-image (t2i, edit, control), qwen-2.1, z-image (base, turbo, control), kolors curated, sdxl curated. Schedules as data, `img2img_start`, `mu`/shift; `RefBundle` grid-aligned typing. **Hires.fix becomes an engine Feature** (pass-local prompt/encode/reference bindings plus `PixelRoundTrip`, §6.6); preserve final-pass-only enhancement, legacy pixel-fit, seeds, events and output policies. `[SW]` migrate its existing admission consumers to `PricingPlan` component/phase/operation inputs and actual pass scopes, cut over only the ENG-2 cohort. Preserve the legacy helper for its remaining callers as listed in §6.14. Prove ordinary img2img on MLX FLUX.2 (§6.13). | ENG-1b | 5–6 wk (split at planning if needed) |
| **ENG-3: Shared-driver video wave** | Wan curated (σ-keyed MoE slot, history `Carry`), Mochi, SVD, SCAIL-2 | ENG-1b | 4–6 wk |
| **ENG-4a: Backend primitives + media/LLM Linear spine** | Extract `mlx-layers`/`candle-layers` below media and LLM (§1.2); move primitive contracts without a media↔LLM edge. Migrate media Linears (including LTX-local) and LLM projections with existing dense/quantized/Prism/fused/trainable variants preserved. Resolve all consumers of the moved source in this feature; locked Cargo graph and iOS partitions remain valid. | ENG-2, ENG-3 | 5–6 wk `[L]`; re-estimate against the full projection inventory; split at a complete consumer boundary if larger |
| **ENG-4b: Request-scoped scales + per-phase adapters** | Preparation **and** forward receive phase scales/branches; dependency-keyed caches (§6.7); keep per-route fold defaults. Cut over imported Krea Hires.fix and its memory inputs (§6.14). Move Krea multiphase to runner phases; LTX pass scales and Wan filtering use the same scale seam. Deliver per-phase adapter Feature, delete candle `ip_dit`, `packed_dit`, `packed_te`. `[SW]` Krea uses existing generic generation entrypoints after #3264; no dependency on APP-5’s future lane harness. | ENG-4a | 5–6 wk `[L]` |
| **ENG-5: FamilyDrivers I** | SDXL native (with inpaint, IP, control), including imported SDXL Hires.fix and its admission path, + InstantID; Kolors native; Wan native UniPC (`Carry`); FLUX.3 flow-map; Boogu Turbo DMD; Sana Sprint SCM; MOSS-Image DMD; Iris FlowDPM (quanta hooks); Ideogram; Supra2 | ENG-4b | 5–6 wk |
| **ENG-6: FamilyDrivers II (multi-stream, autoregressive)** | LTX (explicit tables, route transition, learned upsampler component, AV streams, per-token time); MiniMax-H3 (two schedules); Mage; SenseNova (conditional text/image region, feedback and typed interleaved output, §6.10); Bernini; Krea-realtime (existing whole-clip output adopts the shared `emit` primitive as one terminal chunk; mock traces prove multi-chunk behavior; any already delivered streaming path also migrates. Later streaming delivery/NO-GO stays with SW-F148, §6.9 and MIG-7); SeedVR2; PiD sampler hosted in S10. **Exit:** all image/video denoising cohorts assigned through ENG-6 have adopted the runner. ENG-7 owns remaining bespoke registration and last Hires.fix use, ENG-14 owns audio adoption, and app cohorts retain their named obligations. Verify H3's affected long-clip behavior (§8.1); the shared chunk contract has a production batch user and conformance evidence, with no orphan stream-only hook; SW-F148 remains the streaming product owner. | ENG-5 | 6 wk (likely 6a/6b) |
| **ENG-7: Route records + capability export + registration** | `RouteRecord`, defaults and limits on records; capabilities computed from records; hardware-free export (item 0: hosted candle build check); `Generator` cleanup and typed reports; typed request blocks; REG-6 remaining bespoke entrypoints registered, including StarVector's native token driver with a typed sanitized SVG, preview and terminal metadata. Resolve all outstanding generic Hires.fix callers and delete the shared helper (§6.14). `[SW]` migrate every dump consumer, SceneWorks linking via bundles only, no direct `runtime_*::providers` imports. | ENG-6 | 5–6 wk |
| **ENG-8: Layer kit II** | `AttentionOp` (all raw SDPA), RoPE/qkv/norm/MLP/timestep kit, `ExecPolicy`, trainable op-set switch; delete each training graph copy only with its backward/update and resume witness (WGT-4, §8.1); apply H3 impact checks | ENG-6 | 5–6 wk |
| **ENG-9: Weights** | Codec registry, `WeightPlan` lifecycle, loaders migrated by dialect, per-component tiers, load cancellation, generic load progress, generated adapter routing | ENG-7 | 6 wk (likely 9a/9b) |
| **ENG-10: Latent codecs + R2** | KL-2D, WanCausal3D, causal-3D kit; tiling laws; quality bounds | ENG-8 | 5–6 wk |
| **ENG-11: Encoders on llm + towers** | `TapPolicy` on the existing tap API; codec, mask and architecture blockers; towers | ENG-8, ENG-9 | 5–6 wk |
| **ENG-12: BlockStack + conditioning + R3/R4** | `BlockStack`, block hooks, conditioning modules; R3 in `AttentionOp`, R4 in `BlockStack`; verify affected H3 window/fallback contracts (§8.1) | ENG-8, ENG-9 | 5–6 wk |
| **ENG-13: Computed memory contracts + R1 + iOS** | MEM-2 (delete the fixtures); R1 in the lifetime graph; retire `MemoryRequestScope`; MEM-6 iOS incremental admission and quantum cases (§8.2); H3 policy/host/device preservation (§8.1). `[SW]` migrate the admission contract input if APP-4 has landed. | ENG-10, ENG-11, ENG-12 | 5–6 wk |
| **ENG-14: Audio + LLM adoption** | XL-1/2. Flow/diffusion audio completes runner adoption; autoregressive audio retains its token driver. Close the audio rows against both the audio registry and job dispatch, preserving typed outputs. | ENG-9, ENG-13 | 5–6 wk |
| **ENG-15: DEC-1 (c) pilots** | Approved design direction: pilot A then pilot B, with no rollout until both satisfy BACK-5. Failure leaves baseline (b) in place. Successful rollout waves remain separately planned; no pilot execution is authorized by decision review. | ENG-8…12; DEC-1 approved | 4–6 wk pilots; conditional rollout 3–6 months `[L]` |
| **ENG-16: Configurable chained denoise passes** `[INF]+[SW]` | FEAT-2: inventory/disposition sc-20414 pass controls; shared recipe schema and checked whole-job limits (§6.15); fresh schedule resolution, explicit transitions/history/seeds; whole-chain pricing and typed refusals across the supported route inventory. Persist/version the recipe; expose advanced controls through APP-3's common renderer. Prove replay, cancellation and low-res/native-pass composition; bounded separately dispatched output/quality/time/memory evidence on both shipped backends. At ENG-16 exit qualify all then-shipping migrated rows. When ENG-14 lands later, it owns chain coverage on each newly adopted audio row; if ENG-14 lands first, ENG-16 owns those existing rows. Program FEAT-2 acceptance includes both sets. No family chain loop or parallel schema. | ENG-7, APP-2, APP-3; DEC-4 approved; `[SW]` DEC-5 gate | 4–6 wk `[L]`; re-estimate after schema/route inventory; split at a complete consumer boundary if larger |

### 9.3 SceneWorks features

| Feature | Scope | Depends on | Est. |
|---|---|---|---|
| **APP-1: Model registry** | **0** pin the old-product UI population and capture before-side behavior before any registry/gating edit (§10.3); migrate live consumers and delete the capability digest matrix under DEC-16. Inventory hand tables (APPREG-2). Static and runtime layers; one merge rule; job carries portable logical identity, with server-owned policy and worker-local rebinding (§4.1); manifest v2 with upgrader; evidence store (keys verbatim); seeding; consumer migration. The static layer ingests **today's** dumps, and ENG-7 later migrates that ingest (producer migrates consumers). | DEC-5 verified (§7.5) | 5–6 wk |
| **APP-2: Options core** | **0** extend the pre-change control inventory with old-tree rendering and public-path fixtures (§10.3). `OptionSpec` union seed; `validate()` at every boundary; portable-validator wasm build and native/browser corpus, startup/bundle/latency budgets, hybrid fallback (approved DEC-9); upgrader and versioning for all shapes; `IntentClass::Context`; option effects; typed refusal rendering; telemetry and stamp stability | APP-1 | 5–6 wk |
| **APP-3: Generated panels** | Image, Video, Audio, Training, Simple and Character studios; enumerated custom renderers; accessibility floor; delete `macGating.js`/`candleGating.js`/id gating. Zero diff on the control inventory. | APP-2 | 6 wk (split by studio if needed) |
| **APP-4: Admission pipeline** | **0** re-inventory by grep. Unify the `PricingPlan` component/phase/operation inputs already delivered for migrated routes in ENG-1a…6; `BackendBudget`, per-route posture, decision-diff, ≥25 paths, typed refusal, one stamp, ADM-7 approvals. Keep legacy memory-contract data until ENG-13 migrates that input. APP-4 centralizes consumers; it does not first activate safe pricing for new Features. | ENG-6 (image/video cohort); APP-1. Integrate already-shipping bespoke/audio job admission on its existing declared inputs; ENG-7/14 migrate their inputs with the producer | 5–6 wk |
| **APP-5: Generation lane I** | **0** `JobType` table. Image and vector generation lanes, one event consumer (on #3258), typed vector output, attempt-owned publication and fault/redelivery tests (§4.4), routing table | ENG-7 (REG-6), APP-1, APP-4 | 5–6 wk |
| **APP-6: Generation lane II** | Video, audio and utility lanes including multi-artifact publication; backend module (on #3270); API-only and shipped desktop build/launch acceptance (§4.4) | APP-5; audio lane after ENG-14 | 5–6 wk |
| **APP-7: Tooling** | Generic memory-adapter with one bounded paired proof per designated backend (approved DEC-18), typed dumps, consumer migration and evidence preservation before deleting the memory matrix (DEC-17) and mirrors | ENG-9, APP-1 | 4–5 wk |

### 9.4 Sequencing and delivery bounds

```text
ENGINE  ENG-1a → ENG-1b → [ENG-2 + ENG-3] → ENG-4a → ENG-4b → ENG-5 → ENG-6
        ENG-6 → ENG-7 → ENG-9; ENG-6 → ENG-8
        ENG-8 → ENG-10; [ENG-8 + ENG-9] → [ENG-11 + ENG-12]
        [ENG-10 + ENG-11 + ENG-12] → ENG-13 → ENG-14
        [ENG-8…ENG-12] + DEC-1 → ENG-15 pilots
APP     (DEC-5 cleared) APP-1 → APP-2 → APP-3
                         └──────────(after ENG-6)→ APP-4 → (after ENG-7) APP-5 → APP-6
                         └──────────(after ENG-9)→ APP-7
        ENG-14 → APP-6 (audio adoption)
RECIPE  [ENG-7 + APP-2 + APP-3] → ENG-16
```

- **Delivery estimates `[L]`, restored (C4-06).** These are my estimates, at low confidence, re-estimated when each feature is planned.
  - **Per feature:** 4–6 weeks each, as shown in the tables. Any feature planned above 6 weeks is split at a complete consumer boundary (DEC-12).
  - **Engine track:** ENG-1a…14 contains **16 features** including the 1a/1b and 4a/4b splits; ENG-16 adds one cross-repository recipe feature. There are **17 non-pilot engine features and 25 roadmap nodes** including ENG-15 pilots and APP-1…7. With the parallelism the graph allows (ENG-2∥3, ENG-8∥9, ENG-10∥11∥12 on disjoint files), the aggregate remains **about 14–18 months**, assuming ENG-16 can run alongside later component waves after ENG-7 and APP-3. Serializing it adds its 4–6 weeks; this is a staffing assumption, not a guaranteed date.
  - **SceneWorks track:** APP-1…3 can run against the refreshed baseline, in roughly months 2–8. APP-4…7 follow their engine producers and finish around months 14–18.
  - **DEC-1 (c):** backend-generic rollout adds 3–6 months if both approved pilots pass.
  - **Why longer than draft 0.2's 13–16 months:** the R2 prerequisites (pass model, state contracts, first-consumer admission, the primitives extraction) now land inside early features.
  - **Main uncertainty:** the size of the inventories found at each planning — projection inventory, admission consumers, route-inventory exceptions.
- **Value milestones.**

| Milestone | Prerequisite | What lands |
|---|---|---|
| M1 | ENG-1b complete, including its `[SW]` admission and persistence inputs, against the delivered #3264 baseline | Low-res start on FLUX.1 (about 2.5–3.5 months `[L]`) |
| M2 | ENG-2/3 complete | Low-res start on supported shared-driver image/video routes |
| M3 | ENG-4b mechanism, extended as remaining routes migrate | Per-phase LoRA on every structurally supported migrated route |
| M4 | ENG-6 and APP-1/2 complete | Assigned image/video cohorts on the runner; static registry and shared option validation. Bespoke, audio and remaining app-lane exits follow their owners. |
| M5 | APP-4/5/6 complete, with their engine dependencies | One admission pipeline; one generation lane |
| M6 | program end | Shared components; computed memory contracts; slimming targets |

---

## 10. Metrics, baselines and tools

### 10.1 LOC census

A checked-in script counts `git ls-files` lines per area in both repos, separating src from tests where possible. It runs at baseline and at each feature end. Results are reported, never gating (PAT-10).

### 10.2 Fact audit (PAT-4)

**Definition.** An *encoding* is a source location where a fact's value or id-set is written as a literal. Consumers that read it and derived copies do not count. **Allowed exceptions:** product menus, upgrader alias tables, test fixtures. The fact list is fixed now; each fact gets a grep recipe at baseline.

1. default steps
2. default guidance
3. supported samplers
4. supported schedulers
5. supported modes
6. img2img support
7. inpaint support
8. control modes
9. reference/multi-reference support
10. pose support
11. LoRA/LoKr compatibility
12. quantization tiers per component
13. max steps
14. width/height limits and multiples
15. frame and fps limits
16. duration limits
17. Hires.fix eligibility
18. prompt-enhancement eligibility and limits
19. negative-prompt support
20. guidance kinds
21. preview support
22. decoder options (PiD)
23. memory rung support
24. engine id ↔ model id mapping
25. trainer support and techniques
26. per-image seed relation

### 10.3 Control-inventory tool (PAT-8)

- **What it is:** an old/new comparison using vitest and React Testing Library on jsdom. The first feature touching a surface owns a pre-change capture or renders that surface from its pinned old tree in the comparison. APP-2 generalizes the harness.
- **What it renders:** every studio, for the union of the original product's public model/mode/backend surfaces and the candidate `ModelSpec` fixtures. A removed fixture remains present on the old side.
- **What it records per control:** role, label, option id, default, min/max/step, availability and disabled reason, persistence key, emitted payload when default and when changed, tooltip text, and declared side effects.
- **When it runs:** before the first changing feature, including APP-1 and ENG-1b if either touches the surface first. Record the full old SHA and app/engine pairing. Exercise menu navigation and cardinality, authored and effective prompts in actual batch submissions, re-imports, provider-specific sampling defaults and invalid overrides. Zero unapproved removals or changed behavior. A dropped candidate model must still fail against the old population.
- **Model execution:** none.

### 10.4 Performance and size

- **PERF-1/2.** Predeclare the fixed paired sample count/budget, warm-up, warm/cold cache mode, equivalent host/device, allocator/workspace, precision, effective adapters, schedule, route, seed and resolution. For timing, use the median paired head/base ratio for seconds per outer step and end-to-end. Use a one-sided ceiling of 1.02 or a larger allowance frozen from comparable baseline variability before evaluating the head. Apply the predeclared uncertainty interval method to the paired ratio: upper bound at or below the ceiling passes, lower bound above fails, overlap or mismatched conditions is inconclusive. Gains are allowed. An inconclusive result at the finite budget returns for a decision. Peak memory uses matched per-domain high-water and the +2% ceiling; size uses identical binary slice/profile against its own media-on/off baseline.
- **iOS size:** binary size is baselined in ENG-1a, item 0.

---

## 11. Risks

| # | Risk | L/I | Mitigation |
|---|---|---|---|
| R1 | Program fatigue and branch death across a long program; calendar needs re-estimation | High/High | ≤6-week features to `main`; value milestones; no integration branch; re-plan after every three features |
| R2 | Numeric drift | Med/High | AP13 classes, bit-exact by design for pipeline adoption (variants, seed schemes, history policy); Tier 1 per batch; class (c) only with sign-off |
| R3 | Performance loss (dynamic dispatch, batching, lost MLX fusion) | Med/High | Families batch branches; dyn only at family granularity; `ExecPolicy::fuse`; PERF-1 A/B |
| R4 | Collisions with in-flight work | High/Med | DEC-5 gating; overlap ledger refreshed at each planning; file-ownership tables; new-model rule |
| R5 | Agent failure modes (regrowth, stubs, scope narrowing, boundary gaps) | High/Med | Approved DEC-6 structural checks; verification contract; stub grep at item end; terminal integration sweep |
| R6 | iOS size and memory | Med/Med | Baseline in ENG-1a; feature-partitioned crates; MEM-6 |
| R7 | Training regressions | Med/Med | The graph-removal feature owns backward/update and resume evidence for each affected trainer (§8.1). |
| R8 | Hardware contention | High/Med | Live pool discovery per planning; Tier 1 off the dev Mac; batching |
| R9 | Feature-specific exception and acceptance latency | Med/Med | All §10 policies are approved; batch concrete route exceptions with evidence at planning and keep independent work moving |
| R10 | Hidden coupling (API embeds worker; 17 family crates linked) | Med/Med | REG-5 and REG-6 in ENG-7; LANE-6 in APP-6 |
| R11 | MLX runtime and thread-state bugs invisible to in-crate tests | Med/High | Process-level worker validation whenever runner, `ExecPolicy` or residency changes |
| R12 | Disk exhaustion from agent `target/` directories | High/Med | Clean finished `target/`; cap building agents |
| R13 | Descriptor export needs a native dependency | Med/Med | ENG-7 verifies the hardware-free target feature graphs. If that fails, resolve the descriptor/runtime partition or explicitly revise REG-3 before changing the critical path. |
| R14 | `FamilyDriver` claims diverge from behaviour | Med/High | Drivers generic over a forward seam; conformance suite for slicing, read-only estimates and state continuity (PIPE-7) |
| R15 | A new Feature uses unpriced transitions or stale preparation | Med/High | ENG-1b first-consumer pricing/scope identities; complete dependency keys; §6.5–6.8 design traces and scoped implementation proofs |

---

## 12. Reviewer checklist

1. **Stage model:** can any inventory row fail to be expressed by S0–S11 + `FamilyDriver` + stream bundles + the three span kinds + the transitions? Check against [B §3] and [ENG-6/7/15].
2. **Hook coverage:** is any imaginable cross-cutting feature not expressible through §5 without a family edit, other than model-internal hooks, which are per backend until DEC-1 (c)?
3. **Composition:** do the axis-ownership rules give a deterministic, refusable answer for every pair of today's phase-like mechanisms?
4. **Bit-exactness:** is every pipeline-adoption migration genuinely class (a) under the variant, seed-scheme and history-policy design? Where might it still not be?
5. **Backend seam:** is the layer-level seam enough for MLX fusion and candle kernels? Does DEC-1 follow from [D]?
6. **Primary declarations:** is any fact declared twice, or any behavioural declaration unchecked?
7. **SceneWorks single source:** after APP-1…7, is any model fact restated in code, or any rule implemented twice?
8. **AP9:** does any feature leave a bridge or dual path for a later feature? Is any feature longer than 6 weeks unsplit?
9. **Constraints:** do the CI, hardware, pin and tracker assumptions match REQUIREMENTS §9 and the live state?
10. **Decisions:** does each DEC recommendation follow from its cited evidence? Are any decisions missing?
