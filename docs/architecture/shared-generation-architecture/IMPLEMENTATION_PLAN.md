# Shared Generation Architecture: Implementation Plan

| | |
|---|---|
| **Status** | **Draft 0.5, decision review in progress.** Addresses all four round-5 findings. DEC-1…4 were approved by Michael on 2026-11-03 (including DEC-4's chained-pass product scope); DEC-5…19 remain open. Dispositions and self-review are in [REVIEW_LOG.md](REVIEW_LOG.md). Design-only: individual decisions do not authorize implementation, tracker items or pilot/model execution. |
| **Date** | 2026-11-03 |
| **Implements** | [REQUIREMENTS.md](REQUIREMENTS.md). Requirement IDs and decisions (`DEC-n`) refer to that document. |
| **Evidence** | [evidence/](evidence/) `[A]`…`[H]`; review findings `[ENG-n]` (engine) and `[APP-n]` (app/process) in [REVIEW_LOG.md](REVIEW_LOG.md). Inference `origin/main` `a43724b82`; SceneWorks `main` `a1d95e2c4`. |
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
           (also compiled to wasm for web, DEC-9) · request upgrader · admission core
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
- **Added**, enforced by structural tests if DEC-6 approves, otherwise one recorded grep per terminal item:
  - Family crates may depend on L0, L1 and L2. They define no layer-kit types, guidance combines, σ builders or encoder blocks.
  - Shared SceneWorks code has no model-id literals in conditionals.
  - SceneWorks links bundles and contracts only.
  - No SceneWorks file imports `runtime_*::providers`.

### 1.2 Current → target crate map (engine)

| Today | Target |
|---|---|
| `gen-core` (contracts, plus some family data [A P16]) | `gen-core`, contracts only. Family data moves to `gen-core-<family>` or is computed from codec and route declarations. |
| `gen-core-family`, `gen-core-{flux3,iris,ltx,h3}`, `gen-core-memory-wan` | `gen-core-<family>` holds host math. `gen-core-memory-wan` folds into computed contracts (MEM-2). |
| `mlx-gen/src`, `candle-gen/candle-gen/src` | Extract L1 `mlx-layers`/`candle-layers` below media and LLM consumers: `layers/`, `exec/`, `latent/`, weight execution, adapters and residency primitives. Media orchestration bindings remain in `mlx-gen`/`candle-gen` and consume L1. `BlockStack`/tiling primitives join L1 in their owning features; component implementations sit above L1. |
| 39 family twins | Same crates, holding config, graph, `FamilyModel`/`FamilyDriver`, declarations and `RouteRecord`. Under DEC-1 (c), each collapses to one `gen-<f>` crate. |
| `mlx-llm`, `candle-llm` | Owners of decoder-LM blocks and the encoder tap API. They use the shared Linear spine. |
| `mlx-gen-ios-catalog` (incl. `ios_admission.rs`, per-family `ios_pricing/`) | iOS admission computed from plan bytes and shared memory geometry (MEM-6). Quantum checks remain as hooks. |
| `*-catalog`, `bundles/runtime-*` | Register RouteRecords and Features. Emit the capability export and the route inventory as hardware-free build products. |
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
| SW-F327 deliverables (in review) | #3258 stream supervisor, #3264 memory-contract/probe unification, #3270 cfg alias + AST architecture checks | Built upon, not redone. See the §7.5 overlap ledger. |
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
    pub passes: Vec<PassPlan>,                  // ordered, stable PassIds; ≥1 for denoising routes
    pub features: Vec<AppliedFeature>,
    pub memory: PricingPlan,                    // per-component weights, per-phase peaks, transition deltas (§3.9)
}

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
6. Derive `PricingPlan` from existing aggregate component/phase/operation evidence and additional coexisting transition buffers (§3.9), plus all boundary-evaluation work. Compile the exact resolved plan used by admission and execution. Admission-selected equivalent Features use the same validation and pricing path; each finite candidate is considered once in deterministic order, and no unvalidated rewrite occurs after selection. Run admission can change a selection only by repeating that path before execution; recipe Features, estimator choice and tiers stay fixed.

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
- Resolve estimator choice at recipe finalization under DEC-10, persist it and its semantic version/provenance with the resolved recipe, and reject unsupported or unknown identities. It is not a runtime quality switch or an admission-selected approximation. The only proposed UI controls remain K and scale.
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
- Step contexts and driver state may be type-erased (`Box<dyn Any + Send>`), with request-owned lifetimes; immutable preparations are built once per complete dependency key.
- Tensor operations inside the backend stay statically dispatched.
- This avoids instantiating the runner about 40 times per binary, which would cost iOS size and compile time.

```rust
// gen-core — SKETCH
pub trait RunnerBackend {                       // implemented once per backend (L1)
    type Latent: Clone;                         // a LatentBundle of per-stream tensors
    fn ops(&self) -> &dyn BundleOps<Self::Latent>;      // LatentOps + geometry ops, per stream
    fn exec(&self) -> &dyn ExecPolicy;
}

pub trait FamilyModel<B: RunnerBackend>: Send + Sync {     // object-safe
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

pub trait FamilyDriver<B: RunnerBackend>: Send + Sync {    // PIPE-7
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
  for pass in plan.passes:                                    [stable ids, explicit output dependencies]
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
- **Scope.** Execution is limited to these typed generation stages and their earlier-pass and chunk dependencies.

**Runner invariants.** These are salvaged from sc-20414 and corrected [ENG-2, ENG-19]:
- **Progress is monotone in outer steps.** Evaluation counts stay internal (DEC-19).
- **Cancellation is checked** at every evaluation, every phase boundary and every model-slot swap or load.
- **GPU end hooks run on request-local errors only.** Provider- or process-poisoned errors skip GPU work; host accounting and claim release still run without masking the original error.
- **Solver history never crosses an incompatible boundary.** Geometry change/re-noise forces Reset; `Carry` on `Continue` additionally requires compatible solver/convention and schedule continuation. RNG and driver-cache policies are independent.
- **Re-noise seeds are domain-separated** for new streams; legacy schemes stay bit-exact.
- **Shared resident modules are never mutated.** Per-request state flows through `ForwardArgs` and `StepCtx`.
- **A single-pass, single-phase plan equals the legacy route bit-exactly.** Inert splits preserve numerical and stochastic traces.

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
pub trait FeatureHooks<B: RunnerBackend>: Send + Sync {
    fn id(&self) -> FeatureId;
    fn install(&self, p: &FeatureParams, hooks: &mut HookSet<B>);
}
```

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
pub trait LatentCodec<B>: Send + Sync {
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
  - **Building blocks:** per-component weight bytes from the weight plan; existing aggregate phase/operation peak evidence (two anchors plus derivation); and additional buffers computed from shapes/dtypes. Use the same mechanism for endpoint estimation, learned upsampling, codec encode/decode and ordinary phase execution. This adds no per-model allocation graph, re-keying or recapture campaign.
  - **Evidence basis:** each reused value declares its existing coverage at component/phase granularity: `TotalPeak` (already includes named resident components and ordinary latent/stream/workspace roles) or `ActivationPeak` (excludes weight residency and states which ordinary buffers are included). A joint AV total includes both streams and joint workspace; separate stream estimates may be summed only when their coverage is disjoint, with shared weights/workspace counted once. Do not add a total and its included parts.
  - **Normalize before composing:** for `TotalPeak`, keep the total and add only required resident components or coexisting buffers absent from its coverage. For `ActivationPeak`, add the required resident weights once, then only missing buffers. Do not subtract an assumed component cost from a measured total or treat an unclassified total as activation-only. Existing identity keys are preserved; coverage is a declaration about the existing evidence, not a new measured grid. If its basis cannot be established from the source, keep the existing unpriced-evidence posture (ADM-2/7), mark it unpriced and expose the reason; never manufacture a smaller priced value or silently change refusal policy.
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
  - **Gate:** ENG-1b's admission edits touch files in SW-F327 #3264, so they wait for that PR to merge (DEC-5).
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

- **The capability export** (`sga.capabilities/2`) is a pure function of the linked registry. Bundles produce it in hosted CI for both backends' feature sets, with no GPU and no off-Mac step (REG-3). [APP-14]
- **Verification item.** ENG-7, item 0, confirms that the candle feature set builds on hosted Linux without CUDA. If it does not, the plan is re-cut and the CUDA host goes on that feature's critical path.
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
  converted-model paths
ONE MERGE RULE  → ModelSpec (typed)
```

- **Jobs carry a resolved, typed `ModelSpec` snapshot**, not a revision reference. This works for non-builtin models and multi-worker RunPod. The upgrader converts legacy `modelManifestEntry` carriers.
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
- **Validation (DEC-9):** `core::validate` is compiled to wasm for the web. It is the same Rust code the API, MCP and worker call, so it runs instantly, offline and through the RunPod proxy. The hybrid approach is the fallback.
- **Versioning (OPT-6):** the upgrader handles payloads, recipes, studio settings, share envelopes, film plans, training plans, and the HTTP and MCP DTOs.
  - Writers emit v1-compatible shares when the share is expressible in v1.
  - A newer version is refused with a typed reason.
  - The corpus comes from historical builder outputs.
- **Refusals** render in the web and MCP. Telemetry field names and `advanced.memoryStrategy` stay stable (OPT-8).

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

- **APP-5, item 0, delivers the `JobType` table.** All ~33 job types are classified as generation lane, `LaneHooks`, or utility, each with its admission owner and cache owner. The generation lane covers:
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
  10. write
  11. stamp
- **The backend module** builds on SW-F327 #3270's cfg alias.
- **LANE-6 end state:** validation and admission math live in core; the API links no lane modules; in-process worker mode is preserved for desktop. This is coordinated with SW-6824.

### 4.5 Tooling (TOOL-1…4) [APP-26…28]

- **APP-1, item 0, handles DEC-16:** delete `routing/matrix.rs`, `config/backend-capabilities/matrix.json` and `exceptions.json`. Without that, every APP PR would trip the 61-source digest.
- **DEC-17:** delete the memory-matrix generator, its tests (`generate-memory-matrix.test.mjs`, `matrix-removed-sources.test.mjs`) and `docs/generated/memory-matrix.json`.
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
| SceneWorks `[SW]` | The `lowResStart` option entry, common parameter validation and web controls, all gated by the export. **Persistence:** a versioned `lowResStart` block (`{version, k, scale}` plus the resolved fields) travels through the existing `ADVANCED_KEY_RULES` and share path. There is no new upgrader: no legacy shape exists. Forward compatibility follows the existing share-envelope rules. APP-2's upgrader later adopts this block as one of its shapes (C4-05). **Admission:** ENG-1b updates admission on adopted routes to the shared plan and pricing interface and to per-phase scope identities (§3.9). It does not wait for APP-4, but it does wait for #3264 (DEC-5). |
| **Not the PAT-1 measure** | ENG-1b legitimately touches SceneWorks option, validation and web files. PAT-1 is measured on a later Feature (§6.4). |

**Version-1 recipe choices (proposed under DEC-10, R2-09).**

- M is the selected pass's executed step count after strength truncation. Valid K is `1 ≤ K < M`; default `max(1, floor(M/2))` for M ≥ 2; slider fraction K/M. K=0/M or M<2 refuses. Changing steps/strength to invalidate K produces an issue, not a silent clamp. Feature absent is the no-op.
- Scale is finite in [0.25,0.9], default 0.5. Spatial dimensions use `floor(full_dim * scale / legal_quantum) * legal_quantum`, then minimum, strict reduction and joint-grid checks. Invalid dimensions refuse with legal limits. Frames, audio duration/tokens and count remain unchanged. Native fixed tables permit only eligible interior boundaries.
- `EndpointEvaluation` remains the default. The precisely defined `PreviousEvaluation` protocol in §3.1 is an optional candidate for the declared supporting solvers/drivers, evaluated using the same bounded proof set. It retains the guided primary estimate after normal completion of all K steps. It is not an early-exit or terminal-step replacement.
- Persist resolved recipe/estimator identity and semantic version, target pass, dimensions, K, unchanged schedule/shift policy, transition arithmetic and seed derivation. Runtime provenance also distinguishes source outer-step/evaluation kind, sigma and configuration; do not serialize tensor data. Use the existing versioned block/share path; unknown identities refuse. Do not change estimator defaults for a persisted recipe or silently fall back to a different estimator on replay.
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

## 7. Migration strategy and coordination

### 7.1 Rules

1. **Route-at-a-time strangler.** During a feature, the runner hosts migrated routes while legacy routes keep their `generate`. A family-migration item moves **all** of that family's routes on both backends, then deletes the legacy orchestration and the per-family copies of the shared piece the feature introduces.
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
| Concurrency | 0 | Two interleaved requests on one provider (PIPE-14) |
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

### 7.5 SW-F327 overlap ledger (snapshot 2026-11-03; refreshed at each feature's planning)

**Live refresh during decision review:** #3258 and #3264 are merged into `feature/sw-f327-code-review-remediation`, with merge commits `1c3e121bc2` and `c04f551f9a`; neither is an ancestor of fetched `origin/main`. #3260, #3268 and #3270 remain open against that feature branch. The dispositions below describe ownership and intended reuse, not acceptance on main. Recheck at implementation planning. DEC-5 remains open, and this snapshot authorizes no implementation.

| SW-F327 PR | Cards | SGA requirements touched | Disposition |
|---|---|---|---|
| #3258 | SW-6642 shared stream/heartbeat/cancel supervisor | LANE-4 | **SW-F327-owned; merged to its feature branch.** LANE-4 builds on it after main delivery and re-baselining. |
| #3260 | inference pin to the SW-F327 head | Pin choreography | Sequence per §7.4 |
| #3264 | SW-6659 (`apply_vram_cap` sites), SW-6662 (YuE/YuE2 admission), SW-6753 (Mochi/SVD twins), SW-6754 (`hw.memsize` probes), SW-6640 (floor semantics); touches `krea_multiphase.rs`, `candle_memory_strategy.rs`, `generator_cache.rs`, `base_admission.rs`, `conditioning_gate.rs`, `moss.rs`, `instantid.rs` | ADM-2 (part), ENG-4b `[SW]`, APP-4 | **SW-F327-owned subset; merged to its feature branch.** Overlapping `[SW]` items wait for main delivery. APP-4 re-baselines after it. |
| #3268 | Deletes `apps/web/src/modelCatalog.js` | TOOL-3, APPREG-2 | **SW-F327-owned; PR open.** SGA does not duplicate its deletion; re-baseline after main delivery. |
| #3270 | SW-6677/6679 cfg alias, SW-6686 AST architecture checks, SW-6812/6817/6824 dependency de-dup and direction guard | LANE-5, DEC-6, LANE-6 | **SW-F327-owned; PR open.** SGA builds on main-delivered changes. Structural guards extend `architecture_tests.rs` if DEC-6 allows. |
| SW-F148 (`sc-8432`) | Krea Realtime interactive streaming, spike-gated; NO-GO permitted | PIPE-16, HOOK-1 | **Separate product owner; sequenced adoption (§6.9).** ENG-6 puts current batch output through the production terminal/repeated-chunk primitive. If a streaming path already exists, ENG-6 migrates it too; if delivered later, SW-F148 adopts that primitive with its first stream-only production hooks. NO-GO retains the batch user and removes no existing capability. No duplicate streaming delivery or orphan mock-only hook. |

**Rule (DEC-5).**
- `[INF]` items run now, behind unchanged worker-facing APIs.
- Every `[SW]` item waits until the PRs above that touch its files have merged to `main`.
- In-review cards are never absorbed.

---

## 8. Validation and proof plan

### 8.1 Proof set (both backends where shipped)

| Route | Why |
|---|---|
| FLUX.1-dev t2i + img2img (+ ControlNet on candle, IP and PuLID on MLX) | Packed tokens, `mu`, guidance-embed, conditioning, PuLID's `timestep_to_start_cfg` |
| SDXL native + curated + inpaint, plus InstantID | UNet, ε, ancestral parity loop, inpaint blend, global RNG |
| Qwen-Image t2i + edit | Reference tokens, norm-rescaled CFG, candle merge-at-load |
| Z-Image Turbo + Base | Few-step, C,T,H,W image latent |
| Krea Raw multi-phase | Phases, per-phase adapters/guidance |
| Wan 2.2 A14B T2V (native and curated) + I2V | Video, MoE with σ-keyed slot and history carry, I2V `y` |
| LTX-2.5 distilled + dev | Two streams, explicit-table span, route transition, learned upsampler, STG branches |
| MiniMax-H3 T2VA | Two schedules, joint rows, `FamilyDriver` |
| FLUX.3 Klein distilled + base | Flow-map driver; Qwen3-VL encoder on llm |
| Boogu Turbo; MOSS-Image | DMD drivers, model-dtype combine variant |
| Krea-realtime | Autoregressive streaming driver |
| Iris | iOS family, FlowDPM driver, quanta |
| One audio route (Stable Audio 3 or YuE2), from ENG-14 | Audio adoption |
| Mage-Flow, when touched | Watermark. Weights exist only on `nax-macos-2` and the SMB share |

### 8.2 Where runs happen

- **Tier 0:** ordinary PR CI.
- **Tier 1 and Tier 2:** dispatched, with evidence recorded.
  - **Workflow change:** add an MLX route to `model-inference-proofs.yml` targeting `nax-macos-2` via an `rw-*` label. Today it targets hosted `macos-15`, and its `nax` label also matches the dev Mac [APP-30].
  - **Candle:** a CUDA runner labelled `real-weights`.
- **The dev Mac GPU** is used only on an explicit loan, with the duration stated.
- **Every MLX run** has an external RSS guard and a stated memory cap.
- **Candle-gated SceneWorks tests:** one `windows-candle.yml` dispatch per batch (constraint 7).

---

## 9. Roadmap

### 9.1 Conventions

- **Item tags:** `[INF]` items run in inference and can start now. `[SW]` items are gated by DEC-5.
- **Feature size:** each feature is ≤6 weeks (DEC-12). A feature estimated above that is split at planning.
- **Estimates:** agent-time at **low confidence**, re-estimated with `codegraph-plan`.
- **Slices are sketches.** Each feature becomes 5–12 mechanism slices with 1–3 local acceptance criteria.
- **Every feature's exit** is stated against route-inventory rows (PIPE-18).

### 9.2 Engine features

| Feature | Scope (slices sketched) | Depends on | Est. |
|---|---|---|---|
| **ENG-1a: Runner core + FLUX.1 migration** (walking skeleton; no user-visible change) | **0** `[INF]` prerequisites: the DEC-11 policy PR; the route-inventory dump; baselines (LOC census, fact audit, iOS size); the Tier 1 workflow route on `nax-macos-2`; the prior-art salvage review. **1** `[INF]` the pass/phase `RenderPlan` with stream bundles; typed boundary states; request-owned solver, RNG and guidance state (`RunState`); dependency-keyed preparation; whole-plan validation; `BundleOps`; the runner driving `MemoryRequestScope` with actual phase identities; the error and count contracts. **2** `[INF]` **every FLUX.1 route on both backends** (t2i, img2img, ControlNet, IP, PuLID) moves onto the runner, class (a), and its legacy orchestration is deleted. **3** terminal. **Complete consumer boundary:** FLUX.1 runs entirely on the runner and the old path is gone (AP9). | — | 5–6 wk `[L]` |
| **ENG-1b: Low-res start (first Feature)** | **1** `[INF]` `FeatureSpec` registry; parameter and composition applicability; the `features` field on the existing dump format; typed refusal, which fixes the silent `phases` ignore. **2** `[INF]` low-res recipe v1: `EstimateSource` (endpoint default; candidate completes K steps and retains named primary x0), the typed noise law, aggregate operation evidence plus transition buffers, scope identities; bounded quality/time/memory and replay acceptance (§6.1, §6.5). **3** `[SW]` existing admission consumers move to the plan and pricing inputs; the `lowResStart` option, validation, versioned persistence block and UI (no upgrader, C4-05). **4** terminal. | ENG-1a; DEC-3 approved; DEC-10/13 approval at planning; DEC-5 (`[SW]` waits for SW-F327 #3264) | 4–6 wk `[L]` |
| **ENG-2: Shared-driver image wave** | flux2, chroma, lens, anima, sd3, sana base, boogu base, qwen-image (t2i, edit, control), qwen-2.1, z-image (base, turbo, control), kolors curated, sdxl curated. Schedules as data, `img2img_start`, `mu`/shift; `RefBundle` grid-aligned typing. **Hires.fix becomes an engine Feature** (pass-local prompt/encode/reference bindings plus `PixelRoundTrip`, §6.6); preserve final-pass-only enhancement, legacy pixel-fit, seeds, events and output policies. `[SW]` migrate its existing admission consumers to `PricingPlan` component/phase/operation inputs and actual pass scopes, then delete the worker `HiresFixPlan` path. No wait for APP-4. | ENG-1b | 5–6 wk (split at planning if needed) |
| **ENG-3: Shared-driver video wave** | Wan curated (σ-keyed MoE slot, history `Carry`), Mochi, SVD, SCAIL-2 | ENG-1b | 4–6 wk |
| **ENG-4a: Backend primitives + media/LLM Linear spine** | Extract `mlx-layers`/`candle-layers` below media and LLM (§1.2); move primitive contracts without a media↔LLM edge. Migrate media Linears (including LTX-local) and LLM projections with existing dense/quantized/Prism/fused/trainable variants preserved. Resolve all consumers of the moved source in this feature; locked Cargo graph and iOS partitions remain valid. | ENG-2, ENG-3 | 5–6 wk `[L]`; re-estimate against the full projection inventory; split at a complete consumer boundary if larger |
| **ENG-4b: Request-scoped scales + per-phase adapters** | Preparation **and** forward receive phase scales/branches; dependency-keyed caches (§6.7); keep per-route fold defaults. Move Krea multiphase to runner phases; LTX pass scales and Wan filtering use the same scale seam. Deliver per-phase adapter Feature, delete candle `ip_dit`, `packed_dit`, `packed_te`. `[SW]` Krea uses existing generic generation entrypoints after #3264; no dependency on APP-5’s future lane harness. | ENG-4a | 5–6 wk `[L]` |
| **ENG-5: FamilyDrivers I** | SDXL native (with inpaint, IP, control) + InstantID; Kolors native; Wan native UniPC (`Carry`); FLUX.3 flow-map; Boogu Turbo DMD; Sana Sprint SCM; MOSS-Image DMD; Iris FlowDPM (quanta hooks); Ideogram; Supra2 | ENG-4b | 5–6 wk |
| **ENG-6: FamilyDrivers II (multi-stream, autoregressive)** | LTX (explicit tables, route transition, learned upsampler component, AV streams, per-token time); MiniMax-H3 (two schedules); Mage; SenseNova (interleaved output); Bernini; Krea-realtime (existing whole-clip output adopts the shared `emit` primitive as one terminal chunk; mock traces prove multi-chunk behavior; any already delivered streaming path also migrates. Later streaming delivery/NO-GO stays with SW-F148, §6.9 and MIG-7); SeedVR2; PiD sampler hosted in S10. **Exit:** PIPE-1 complete for every inventory row; the shared chunk contract has a production batch user and conformance evidence, with no orphan stream-only hook; SW-F148 remains the streaming product owner. | ENG-5 | 6 wk (likely 6a/6b) |
| **ENG-7: Route records + capability export + registration** | `RouteRecord`, defaults and limits on records; capabilities computed from records; hardware-free export (item 0: hosted candle build check); `Generator` cleanup and typed reports; typed request blocks; REG-6 bespoke providers registered. `[SW]` migrate every dump consumer, SceneWorks linking via bundles only, no direct `runtime_*::providers` imports. | ENG-6 | 5–6 wk |
| **ENG-8: Layer kit II** | `AttentionOp` (all raw SDPA), RoPE/qkv/norm/MLP/timestep kit, `ExecPolicy`, trainable op-set switch; delete training model copies (WGT-4) | ENG-6 | 5–6 wk |
| **ENG-9: Weights** | Codec registry, `WeightPlan` lifecycle, loaders migrated by dialect, per-component tiers, load cancellation, generic load progress, generated adapter routing | ENG-7 | 6 wk (likely 9a/9b) |
| **ENG-10: Latent codecs + R2** | KL-2D, WanCausal3D, causal-3D kit; tiling laws; quality bounds | ENG-8 | 5–6 wk |
| **ENG-11: Encoders on llm + towers** | `TapPolicy` on the existing tap API; codec, mask and architecture blockers; towers | ENG-8, ENG-9 | 5–6 wk |
| **ENG-12: BlockStack + conditioning + R3/R4** | `BlockStack`, block hooks, conditioning modules; R3 in `AttentionOp`, R4 in `BlockStack` | ENG-8, ENG-9 | 5–6 wk |
| **ENG-13: Computed memory contracts + R1 + iOS** | MEM-2 (delete the fixtures); R1 in the lifetime graph; retire `MemoryRequestScope`; MEM-6 iOS admission. `[SW]` migrate the admission contract input if APP-4 has landed. | ENG-10, ENG-11, ENG-12 | 5–6 wk |
| **ENG-14: Audio + LLM adoption** | XL-1/2 | ENG-9, ENG-13 | 5–6 wk |
| **ENG-15: DEC-1 (c) pilots** | Approved design direction: pilot A then pilot B, with no rollout until both satisfy BACK-5. Failure leaves baseline (b) in place. Successful rollout waves remain separately planned; no pilot execution is authorized by decision review. | ENG-8…12; DEC-1 approved | 4–6 wk pilots; conditional rollout 3–6 months `[L]` |
| **ENG-16: Configurable chained denoise passes** `[INF]+[SW]` | FEAT-2: inventory/disposition sc-20414 pass controls; shared recipe schema and bounded pass-list validation; fresh schedule resolution, explicit transitions/history/seeds; whole-chain pricing and typed refusals across the supported route inventory. Persist/version the recipe; expose advanced controls through APP-3's common renderer. Prove replay, cancellation and low-res/native-pass composition; bounded separately dispatched output/quality/time/memory evidence on both shipped backends. No family chain loop or parallel schema. | ENG-7, APP-2, APP-3; DEC-4 approved; `[SW]` DEC-5 gate | 4–6 wk `[L]`; re-estimate after schema/route inventory; split at a complete consumer boundary if larger |

### 9.3 SceneWorks features (all `[SW]`; start after the DEC-5 gate)

| Feature | Scope | Depends on | Est. |
|---|---|---|---|
| **APP-1: Model registry** | **0** delete the backend-capability matrix (DEC-16); hand-table inventory (APPREG-2). Static and runtime layers; one merge rule; job carries the `ModelSpec` snapshot; manifest v2 with upgrader; evidence store (keys verbatim); seeding; consumer migration. The static layer ingests **today's** dumps, and ENG-7 later migrates that ingest (producer migrates consumers). | DEC-5 gate | 5–6 wk |
| **APP-2: Options core** | **0** control-inventory tool and "before" capture (§10.3). `OptionSpec` union seed; `validate()` at every boundary; wasm build (DEC-9); upgrader and versioning for all shapes; `IntentClass::Context`; option effects; typed refusal rendering; telemetry and stamp stability | APP-1 | 5–6 wk |
| **APP-3: Generated panels** | Image, Video, Audio, Training, Simple and Character studios; enumerated custom renderers; accessibility floor; delete `macGating.js`/`candleGating.js`/id gating. Zero diff on the control inventory. | APP-2 | 6 wk (split by studio if needed) |
| **APP-4: Admission pipeline** | **0** re-inventory by grep. Unify the `PricingPlan` component/phase/operation inputs already delivered for migrated routes in ENG-1a…6; `BackendBudget`, per-route posture, decision-diff, ≥25 paths, typed refusal, one stamp, ADM-7 approvals. Keep legacy memory-contract data until ENG-13 migrates that input. APP-4 centralizes consumers; it does not first activate safe pricing for new Features. | ENG-6 (`RenderPlan` for all routes); APP-1; #3264 merged | 5–6 wk |
| **APP-5: Generation lane I** | **0** `JobType` table. Image generation lane, one event consumer (on #3258), routing table | ENG-7 (REG-6), APP-1, APP-4 | 5–6 wk |
| **APP-6: Generation lane II** | Video, audio and utility lanes; backend module (on #3270); LANE-6 API end state | APP-5; audio lane after ENG-14 | 5–6 wk |
| **APP-7: Tooling** | Generic memory-adapter (DEC-18 run), typed dumps, delete the memory matrix (DEC-17) and mirrors | ENG-9, APP-1 | 4–5 wk |

### 9.4 Sequencing and delivery bounds

```text
ENGINE  ENG-1a → ENG-1b → [ENG-2 + ENG-3] → ENG-4a → ENG-4b → ENG-5 → ENG-6
        ENG-6 → ENG-7 → ENG-9; ENG-6 → ENG-8
        ENG-8 → ENG-10; [ENG-8 + ENG-9] → [ENG-11 + ENG-12]
        [ENG-10 + ENG-11 + ENG-12] → ENG-13 → ENG-14
        [ENG-8…ENG-12] + DEC-1 → ENG-15 pilots
APP     (DEC-5 gate) APP-1 → APP-2 → APP-3
                         └──────────(after ENG-6)→ APP-4 → (after ENG-7) APP-5 → APP-6
                         └──────────(after ENG-9)→ APP-7
RECIPE  [ENG-7 + APP-2 + APP-3] → ENG-16
```

- **Delivery estimates `[L]`, restored (C4-06).** These are my estimates, at low confidence, re-estimated when each feature is planned.
  - **Per feature:** 4–6 weeks each, as shown in the tables. Any feature planned above 6 weeks is split at a complete consumer boundary (DEC-12).
  - **Engine track:** ENG-1a…14 contains **16 features** including the 1a/1b and 4a/4b splits; ENG-16 adds one cross-repository recipe feature. There are **17 non-pilot engine features and 25 roadmap nodes** including ENG-15 pilots and APP-1…7. With the parallelism the graph allows (ENG-2∥3, ENG-8∥9, ENG-10∥11∥12 on disjoint files), the aggregate remains **about 14–18 months**, assuming ENG-16 can run alongside later component waves after ENG-7 and APP-3. Serializing it adds its 4–6 weeks; this is a staffing assumption, not a guaranteed date.
  - **SceneWorks track:** APP-1…3 can run from the DEC-5 gate onward, in roughly months 2–8. APP-4…7 follow their engine producers and finish around months 14–18.
  - **DEC-1 (c):** backend-generic rollout adds 3–6 months if both approved pilots pass.
  - **Why longer than draft 0.2's 13–16 months:** the R2 prerequisites (pass model, state contracts, first-consumer admission, the primitives extraction) now land inside early features.
  - **Main uncertainty:** the size of the inventories found at each planning — projection inventory, admission consumers, route-inventory exceptions.
- **Value milestones.**

| Milestone | Prerequisite | What lands |
|---|---|---|
| M1 | ENG-1b complete, including its `[SW]` admission and persistence inputs, which wait for SW-F327 #3264 | Low-res start on FLUX.1 (about 2.5–3.5 months `[L]`) |
| M2 | ENG-2/3 complete | Low-res start on supported shared-driver image/video routes |
| M3 | ENG-4b mechanism, extended as remaining routes migrate | Per-phase LoRA on every structurally supported migrated route |
| M4 | ENG-6 and APP-1/2 complete | Every route on the runner; one registry and options pipeline |
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

- **What it is:** a Tier 0 test harness built in APP-2, item 0, using vitest and React Testing Library on jsdom.
- **What it renders:** every studio, for every (model, backend, mode) in a `ModelSpec` fixture set.
- **What it records per control:** role, label, option id, default, min/max/step, availability and disabled reason, persistence key, emitted payload when default and when changed, tooltip text, and declared side effects.
- **When it runs:** the "before" capture runs before APP-3 changes any UI. The diff must show zero removals and zero changed behaviour fields, unless approved.
- **Model execution:** none.

### 10.4 Performance and size

- **PERF-1 and PERF-2:** A/B on the same runner and in the same window (pre-feature `main` against the feature head), on proof-set routes the feature touched.
- **iOS size:** binary size is baselined in ENG-1a, item 0.

---

## 11. Risks

| # | Risk | L/I | Mitigation |
|---|---|---|---|
| R1 | Program fatigue and branch death across a long program; calendar needs re-estimation | High/High | ≤6-week features to `main`; value milestones; no integration branch; re-plan after every three features |
| R2 | Numeric drift | Med/High | AP13 classes, bit-exact by design for pipeline adoption (variants, seed schemes, history policy); Tier 1 per batch; class (c) only with sign-off |
| R3 | Performance loss (dynamic dispatch, batching, lost MLX fusion) | Med/High | Families batch branches; dyn only at family granularity; `ExecPolicy::fuse`; PERF-1 A/B |
| R4 | Collisions with in-flight work | High/Med | DEC-5 gating; overlap ledger refreshed at each planning; file-ownership tables; new-model rule |
| R5 | Agent failure modes (regrowth, stubs, scope narrowing, boundary gaps) | High/Med | DEC-6 guards or terminal greps; verification contract; stub grep at item end; terminal integration sweep |
| R6 | iOS size and memory | Med/Med | Baseline in ENG-1a; feature-partitioned crates; MEM-6 |
| R7 | Training regressions | Med/Med | Trainers in Tier 1 from ENG-8 |
| R8 | Hardware contention | High/Med | Live pool discovery per planning; Tier 1 off the dev Mac; batching |
| R9 | Decision latency (DEC-1…19) | Med/Med | Each feature depends on a named decision subset |
| R10 | Hidden coupling (API embeds worker; 17 family crates linked) | Med/Med | REG-5 and REG-6 in ENG-7; LANE-6 in APP-6 |
| R11 | MLX runtime and thread-state bugs invisible to in-crate tests | Med/High | Process-level worker validation whenever runner, `ExecPolicy` or residency changes |
| R12 | Disk exhaustion from agent `target/` directories | High/Med | Clean finished `target/`; cap building agents |
| R13 | Capability export cannot be produced hardware-free for candle | Med/Med | ENG-7 item 0 checks this first; otherwise the CUDA host goes on the critical path and the plan is re-cut |
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
