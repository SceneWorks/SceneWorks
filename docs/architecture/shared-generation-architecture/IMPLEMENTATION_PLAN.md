# Shared Generation Architecture: Implementation Plan

| | |
|---|---|
| **Status** | **Draft 0.2.** It incorporates two internal adversarial reviews (see [REVIEW_LOG.md](REVIEW_LOG.md)). It is ready for review round 1 with Michael. Nothing is approved. No tracker items will be created until it is. |
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

### 1.1 Layers and dependency direction

```text
INFERENCE REPO
┌────────────────────────────────────────────────────────────────────────────────────┐
│ L0  CONTRACTS (tensor-free; zero tensor deps — unchanged rule)                     │
│     gen-core:  Request core + typed extension blocks · RenderPlan (streams, phases, │
│                spans, transitions, seeds, count/batch) · FeatureSpec registry ·     │
│                RouteStructure · runner policy (generic over backend op traits) ·   │
│                sampling (solvers, named schedules, guidance CombineVariants) ·     │
│                memory math (rungs, selection, per-phase/per-stream geometry) ·     │
│                WeightPlan + codec registry · RouteRecord + capability computation │
│                · runtime sinks + error taxonomy (shared with core-llm)            │
│     gen-core-<family>: host math only (configs, ids, packers, family schedules)   │
├────────────────────────────────────────────────────────────────────────────────────┤
│ L1  BACKEND RUNTIME (mlx-gen core · candle-gen core) — identical API shape         │
│     Layer kit: Linear spine · AttentionOp · Norm · Conv(layout) · Rope · Embed      │
│     ExecPolicy (materialize · fuse/compile · memory scope · cancel · stream)        │
│     LatentOps + LatentGeometryOps (bundle-aware resample, pack/unpack)              │
│     WeightPlan executor + codecs · adapter runtime · residency rt · tiling rt      │
│     BlockStack driver · FeatureHooks<B> registrations                              │
├────────────────────────────────────────────────────────────────────────────────────┤
│ L2  SHARED COMPONENTS (spec in L0; impl per backend — or generic if DEC-1 (c))     │
│     encoders (llm CausalLm + tap API extension; towers) · latent codecs · learned  │
│     upsamplers · conditioning modules · sub-block kit                              │
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
| `mlx-gen/src`, `candle-gen/candle-gen/src` | L1 backend runtime with identical module layout: `layers/`, `exec/`, `latent/`, `weights/`, `adapters/`, `residency/`, `tiling/`, `blocks/`, `runner/`. |
| 39 family twins | Same crates, holding config, graph, `FamilyModel`/`FamilyDriver`, declarations and `RouteRecord`. Under DEC-1 (c), each collapses to one `gen-<f>` crate. |
| `mlx-llm`, `candle-llm` | Owners of decoder-LM blocks and the encoder tap API. They use the shared Linear spine. |
| `mlx-gen-ios-catalog` (incl. `ios_admission.rs`, per-family `ios_pricing/`) | iOS admission computed from plan bytes and shared memory geometry (MEM-6). Quantum checks remain as hooks. |
| `*-catalog`, `bundles/runtime-*` | Register RouteRecords and Features. Emit the capability export and the route inventory as hardware-free build products. |
| `candle-audio-*` | Adopt the weights, Linear and adapter layers and the runtime contracts. Use the runner where a flow loop exists (XL-1). |

---

## 2. Reuse inventory

The design grows existing seams rather than inventing new ones.

| Existing asset | Where | Becomes |
|---|---|---|
| **sc-20414 executor invariants and testkit** | `feature/sc-20414`: `gen-core/src/sampling/pass_executor.rs` (844 lines), `gen-core-testkit/src/denoise_passes.rs` | Salvaged per DEC-4: monotone progress, cancel at every evaluation and boundary, `end_pass` on every exit path, the domain-separated re-noise salt, the CPU-ops conformance suite, and the ComfyUI pass-schedule math (`denoise_pass_schedule_steps`, `terminal_pass_segment`). **Not reused:** the executor itself. It is fresh-per-pass only, always re-noises (`:274`), has no geometry, leaves the combine to the host (`:36,143`), and refuses Krea phases (`denoise_passes.rs:433-435`). The `denoise_passes.rs` wire contract is also not reused. [ENG-1] |
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
    pub phases: Vec<PhasePlan>,                 // ≥1
    pub conditioning: ConditioningPlan,         // streams to encode, needs_negative, grid-aligned refs
    pub decode: DecodePlan,
    pub features: Vec<AppliedFeature>,
    pub memory: MemoryPlanInputs,               // per-phase × per-stream geometry, refs (count + pixels)
}

pub struct PhasePlan {
    pub span: PhaseSpan,                        // per stream (H3 has two schedules)
    pub geometry: PerStream<Geometry>,
    pub model: ModelSlotRule,                   // Fixed(slot) | SigmaKeyed{boundary, hi, lo}  (Wan curated)
    pub adapters: AdapterScales,                // passed per request through ForwardArgs (PIPE-14)
    pub guidance: GuidancePlan,                 // CombineVariant + branches (+ perturbations)
    pub solver: SolverId,
    pub history: SolverHistory,                 // Carry | Reset  — Carry forbidden across geometry change/renoise
    pub transition_in: Transition,
}

pub enum PhaseSpan {
    GlobalSlice { schedule: ScheduleRef, from: usize, to: usize },   // Krea, low-res start
    FreshPass   { steps: u32, denoise: f32, schedule: ScheduleId },  // ComfyUI chained KSampler
    Explicit    { table: ScheduleTableId },                          // LTX stage-2 baked table
}

pub enum Transition {
    Start,
    Continue,
    Renoise   { streams: StreamSet, to: SigmaRef, seed: SeedStream },
    Resample  { streams: StreamSet, kind: ResampleKind, then: Option<Box<Transition>> },
    PixelRoundTrip { resize: PixelResize, reencode: CodecRef },      // Hires.fix
    Route(TransitionId),                                             // family-supplied, bit-exact (LTX)
}
pub enum ResampleKind { Nearest, Bilinear, Learned(ComponentId) }    // learned = route component + normalization
```

**Plan compilation.** Compilation is weights-free and deterministic. It runs in this order:
1. Resolve route defaults.
2. Validate against `RouteStructure`; refuse anything it cannot honour (AP4).
3. Apply each active Feature's `rewrite_plan` in declared order (HOOK-3), merging phase boundaries.
4. Validate history policy against transitions.
5. Compute conditioning needs (any phase with CFG ⇒ encode the negative).
6. Compute `MemoryPlanInputs`.

### 3.2 Route structure and primary declarations (AP5, HOOK-4)

```rust
pub struct RouteStructure {
    // primary declarations (family-owned; each behavioural one has a Tier 0 conformance check)
    pub streams: Vec<StreamDecl>,               // LatentSpace, geometry rules (multiples, legal grids), time convention
    pub prediction: PredictionType,
    pub geometry_change: GeometryChange,        // checked by conformance over a mocked forward
    pub loop_kind: LoopKind,                    // SharedDriver | FamilyDriver(DriverSupport)  — DriverSupport conformance-checked
    pub guidance: GuidanceSurface,              // CombineVariant(s), row order, distilled
    pub seed_scheme: SeedScheme,                // legacy schemes preserved bit-exact (PIPE-11)
    pub count: CountSemantics,
    pub residency: ComponentLifetimes,          // which component is live in which stage/phase (MEM-1)
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
- Step contexts are type-erased (`Box<dyn Any + Send>`) and built once per key.
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
    fn prepare_step_context(&self, key: &StepCtxKey, cond: &CondBundle<B>, refs: &AlignedRefs<B>)
        -> Result<StepCtx>;                                  // erased; cached by key; evicted on phase exit
    /// Evaluate the requested branches; the family decides batching (PIPE-8).
    fn predict(&self, ctx: &StepCtx, x: &B::Latent, t: &TimeInput, branches: &BranchSet,
               args: &ForwardArgs, info: &StepInfo) -> Result<BranchOutputs<B>>;
}
pub struct ForwardArgs<'a> { pub adapter_scales: &'a AdapterScales, pub perturbation: Option<&'a Perturbation> }

pub trait FamilyDriver<B: RunnerBackend>: Send + Sync {    // PIPE-7
    fn support(&self) -> &DriverSupport;                    // conformance-checked
    fn run_slice(&self, input: SliceInput<'_, B>, hooks: &mut StepHooks<'_, B>) -> Result<B::Latent>;
}

pub struct StepInfo { pub phase: u16, pub step: u32, pub outer_step: u32, pub eval_in_step: u8,
                      pub sigma: PerStream<f32>, pub next_sigma: PerStream<f32> }
pub enum TimeInput { Sigma(PerStream<f32>), FlowMap{ t: f32, t_ref: f32 },
                     PerToken(TokenTimeBuilder), RowTimesteps(RowTimes), Angle(f32) }
```

**Stage execution:**

```text
S0  plan      = compile(request, route)                        [Plan hooks; refusals typed]
S1  prompt'   = prompt_prep(plan)                              [enhancer components]
S2  cond      = encoders.encode(plan.conditioning)             [lifetime graph: Text]
S3  refs      = references(plan)  → {geometry_free, grid_aligned: Fn(PerStream<Geometry>)}
S5  latent    = init(phase0.geometry, seeds, refs)             [route NoiseSource; img2img/inpaint blend]
for p in phases:
  S6  latent  = transition(p.transition_in, latent)            [runner | Feature | route-supplied]
      ctx     = cache.get_or_build(key(p), …); aligned = refs.grid_aligned(p.geometry) if changed
      models.ensure(p.model)                                    [lifetime graph; cancellable swap — WGT-5]
      scope.enter_phase(p)                                      [MEM-5: drive MemoryRequestScope]
  S7  latent  = SharedDriver(p.solver, p.history).run(span(p), predict ⊕ guidance(p.guidance), hooks)
                | family_driver.run_slice(…)
  S8  exit(p): evict ctx keys not reused; end hooks (request-local errors only — PIPE-15)
S9  post_latent; S10 codec.decode (R2; PiD sampler hosted here); S11 finalize; scope.finish()
```

**Runner invariants.** These are salvaged from sc-20414 and corrected [ENG-2, ENG-19]:
- **Progress is monotone in outer steps.** Evaluation counts stay internal (DEC-19).
- **Cancellation is checked** at every evaluation, every phase boundary and every model-slot swap or load.
- **End hooks run on request-local errors only.** Provider- or process-poisoned errors skip GPU work.
- **Solver history never crosses a geometry change or a re-noise.** `Carry` is allowed only on `Continue` with an unchanged geometry.
- **Re-noise seeds are domain-separated** for new streams; legacy schemes stay bit-exact.
- **Shared resident modules are never mutated.** Per-request state flows through `ForwardArgs` and `StepCtx`.
- **A single-phase plan equals the legacy route bit-exactly.**

### 3.4 Features (HOOK-1…5) [ENG-11]

```rust
// gen-core (tensor-free, dyn) — SKETCH
pub trait FeatureSpec: Send + Sync {
    fn id(&self) -> FeatureId;
    fn class(&self) -> FeatureClass;                 // Recipe | Equivalent{quality_bound} | Diagnostic
    fn params(&self) -> &ParamSchema;
    fn applicability(&self, s: &RouteStructure, p: Option<&FeatureParams>) -> Applicability;
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

- **Composition.** Phase boundaries from all active Features merge into one list. Each override axis is owned by at most one Feature per span. The axes are geometry, model slot, adapter scales, guidance and solver. A second claimant on an owned axis is a typed conflict.
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
  - LTX stage tables are `ScheduleTableId`s used by `PhaseSpan::Explicit`.
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
| `Linear` | **Bases:** Dense, AffinePacked, Ggml, GgufK, Nvfp4, Fp8Row, Int8Row, W4A8, ConvRot. **Plus:** dtype policy, residual adapters reading **`ForwardArgs.adapter_scales`** (request-scoped, PIPE-14) and a trainable mode. The install policy (fold or residual) is per route (WGT-3). | 68 `*Linear` types; the LTX-local Linear; candle `QLinear`+`AdaptLinear`+`LoraLinear` |
| `AttentionOp` | `attend(q, k, v, mask, AttnPolicy{budget, bounded, sparse, i32_guard}, perturbation)` | Raw SDPA in 38 `mlx-gen*` directories; the overflow-guard sites |
| Norm, Mlp, Modulation, QkNorm, Rope, Embed, TimestepEmbed, Patchify | Parameterized kit types. Replacements that change operation order are declared class (b) or (c). | About 135 MLPs, 304 RoPE functions, about 45 sinusoid copies |
| `Conv{1,2,3}d` | Explicit `Layout`, converted at load | Ad hoc NHWC/NCHW handling |
| `ExecPolicy` | `materialize`, `fuse`, `memory_scope`, `cancel_point`, `stream_scope`, `load_activity` | Direct eval/compile calls in families |

**Trainable mode** switches to composable operations instead of fused CustomOps on candle (WGT-4).

**MLX watchdog ordering** lives in the loader and `ExecPolicy::materialize`.

**The DEC-1 (c) path** follows [D §9]: pilot A (a small family), then pilot B (FLUX.3 including LoRA training), with A/B acceptance on the same runner. Until then, BACK-4's identical API makes twins mechanically mirrorable.

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
- **Pricing (MEM-3)** takes the maximum over phases × streams, plus cross-phase residency, plus Feature memory effects, plus reference pixel geometry for vision-language encoders.
- **iOS admission** (`ios_admission.rs`, per-family `ios_pricing/`) is replaced by plan-byte pricing plus quantum hooks (MEM-6).

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
| S11 output | `post_process` · `stream_chunk` | after | Feature/route | yes | alpha, trim, inline upscale, audio fit, Krea-realtime chunks |

---

## 6. Worked examples (files touched)

### 6.1 Low-res start: FEAT-1 (ships in ENG-1, extended as routes migrate)

| Where | Change |
|---|---|
| Engine `[INF]` | `gen-core/src/features/low_res_start.rs`. **Params:** `{k, scale, renoise, mu_policy}`. **Applicability:** `geometry_change == Supported && loop_kind honours Resample+Renoise`. **Plan rewrite:** split at K, set `geometry = final × scale` rounded to the stream geometry rules, `transition = Resample(Bilinear) → Renoise(σ_K)`, `history = Reset`. **Memory effect:** transition residency, meaning both latents briefly. Tier 0 plan snapshots, composition tests and conformance. |
| Backend `[INF]` | None after ENG-1, which builds `BundleOps::resample` and pack/unpack once. |
| Families | None. Each route's `prepare_step_context` is keyed on geometry (PIPE-5). Refusals come from the route structure. |
| SceneWorks `[SW]` | The `lowResStart` option entry, core validation, and a web control (number field, linked slider, scale) gated by the export's Feature applicability. **Admission is unchanged.** The maximum over phases is the final-geometry phase, which today's pricing already covers. The transition transient (two latents) is negligible next to the weights. This is an explicit claim to verify in the ENG-1 Tier 1 peak-memory check. |
| **Not the PAT-1 measure** | ENG-1 legitimately touches SceneWorks option, validation and web files. PAT-1 is measured on a later Feature (§6.4). |

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

---

## 7. Migration strategy and coordination

### 7.1 Rules

1. **Route-at-a-time strangler.** During a feature, the runner hosts migrated routes while legacy routes keep their `generate`. A family-migration item moves **all** of that family's routes on both backends, then deletes the legacy orchestration and the per-family copies of the shared piece the feature introduces.
2. **The producer migrates its consumers (AP9).** A feature that replaces a source migrates every consumer in the same feature, in both repos. Example: ENG-7 replaces the capability dumps, so ENG-7 migrates SceneWorks' dump consumers, including APP-1's static layer if APP-1 has already landed. That feature's `[SW]` items are then gated by DEC-5.
3. **Characterize before consolidating** (§7.2).
4. **List behaviour changes; never let them slip in.** Each change is a named PR entry with its AP13 class and evidence. Class (c) needs sign-off.
5. **Order within a feature:** ABI, fingerprint and evidence-key changes first; then the walking skeleton; then mechanism slices; then the terminal item (pin, Tier 1/2, whole-branch review, merge).
6. **File ownership.** Each feature's plan includes a file-ownership table. Features that overlap in time own disjoint files.
7. **New-model rule.** Once ENG-1 lands, new families target the runner. Once the kit, weights and components features land, new families use them.

### 7.2 Characterization tiers (MIG-3)

| Check | Tier | Pins |
|---|---|---|
| Plan snapshot | 0 | `RenderPlan` per route over a request corpus |
| Runner trace over a mocked backend | 0 | Stage/phase/hook sequence; Wan MoE trace equality with legacy [ENG-2] |
| `FamilyDriver` and Feature conformance over a mocked forward | 0 | Every `DriverSupport` and geometry-change claim [ENG-16, ENG-21] |
| Concurrency | 0 | Two interleaved requests on one provider (PIPE-14) |
| Fault injection | 0 | Error classes × stages (PIPE-15) |
| Weight-plan receipt | 0 | Header-only compile and bind receipts |
| Engine-request equivalence | 0 | SceneWorks-built requests recorded by a fake generator |
| Admission and routing decision-diffs | 0 | Per (route, tier, geometry, budget); cfg-free core runs on Linux CI |
| Receipt/`raw_settings` shape | 0 | Shape assertions |
| Primitive op checks | 0 | Kit ops on synthetic tensors |
| Base-versus-head equivalence | 1 | Same runner, same job, shape-faithful random weights (real dims, truncated depth), in-job comparison, no committed goldens |
| Real-weight proof | 2 | Proof-set A/B for fidelity, s/step and peak memory (DEC-15) |

### 7.3 AP13 expected classes per feature (initial; refined at planning)

| Feature | Expected class | Notes |
|---|---|---|
| ENG-1…3, ENG-5, ENG-6 (pipeline adoption) | (a) | Loops, combines, seeds and history are preserved bit-exactly by design |
| ENG-4 (Linear spine, ScaleContext) | (a) for fold-kept routes; Krea candle per-phase `[M]` (b)/(c) | Krea candle re-adapts in place today |
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

| SW-F327 PR | Cards | SGA requirements touched | Disposition |
|---|---|---|---|
| #3258 | SW-6642 shared stream/heartbeat/cancel supervisor | LANE-4 | **Delivered by SW-F327.** LANE-4 builds on it. APP-5 sequenced after its merge. |
| #3260 | inference pin to the SW-F327 head | Pin choreography | Sequence per §7.4 |
| #3264 | SW-6659 (`apply_vram_cap` sites), SW-6662 (YuE/YuE2 admission), SW-6753 (Mochi/SVD twins), SW-6754 (`hw.memsize` probes), SW-6640 (floor semantics); touches `krea_multiphase.rs`, `candle_memory_strategy.rs`, `generator_cache.rs`, `base_admission.rs`, `conditioning_gate.rs`, `moss.rs`, `instantid.rs` | ADM-2 (part), ENG-4 `[SW]`, APP-4 | **Partly delivered by SW-F327.** All `[SW]` items touching these files are gated on its merge. APP-4 re-baselines after it. |
| #3268 | Deletes `apps/web/src/modelCatalog.js` | TOOL-3, APPREG-2 | **Delivered by SW-F327.** Removed from SGA scope. |
| #3270 | SW-6677/6679 cfg alias, SW-6686 AST architecture checks, SW-6812/6817/6824 dependency de-dup and direction guard | LANE-5, DEC-6, LANE-6 | **Delivered by SW-F327.** SGA builds on it. Structural guards extend `architecture_tests.rs` if DEC-6 allows. |

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
| **ENG-1: Runner skeleton + FLUX.1 + low-res start** | **0** `[INF]` DEC-11 policy PR; route-inventory dump; baselines (LOC census, fact audit, iOS size); Tier 1 workflow route for `nax-macos-2`; salvage review of sc-20414, sc-19048 and sc-25327. **1** `[INF]` `RenderPlan` (stream-bundle types from day one), runner, unified driver signature, `BundleOps` (pack/unpack/resample), runner driving `MemoryRequestScope`, error taxonomy, count semantics. **2** `[INF]` **all FLUX.1 routes on both backends** (t2i, img2img, ControlNet, IP, PuLID) with their `CombineVariant`s and `SeedScheme`s, class (a). **3** `[INF]` `FeatureSpec` registry, applicability, typed refusal (fixes the silent `phases` ignore); `features` field added to the **existing** capability-dump format. **4** `[INF]` `low_res_start`. **5** `[SW]` option, validation, web control. **6** terminal. | — | 4–6 wk |
| **ENG-2: Shared-driver image wave** | flux2, chroma, lens, anima, sd3, sana base, boogu base, qwen-image (t2i, edit, control), qwen-2.1, z-image (base, turbo, control), kolors curated, sdxl curated. Schedules as data, `img2img_start`, `mu`/shift; `RefBundle` grid-aligned typing. **Hires.fix becomes an engine Feature** (`PixelRoundTrip` transition, VAE residency priced). `[SW]` delete the worker `HiresFixPlan` path (`image_jobs/base.rs:8390-8615`). | ENG-1 | 5–6 wk (split at planning if needed) |
| **ENG-3: Shared-driver video wave** | Wan curated (σ-keyed MoE slot, history `Carry`), Mochi, SVD, SCAIL-2 | ENG-1 | 4–6 wk |
| **ENG-4: Linear spine + request-scoped scales + per-phase adapters** | Candle Linear spine (port of the `AdaptableLinear` contract); MLX unification (LTX-local Linear folded in); `ForwardArgs.adapter_scales`; per-route install policy (fold kept); Krea multi-phase onto runner phases on both backends; LTX pass scales and Wan MoE adapter filtering via scales; per-phase adapter Feature; delete candle `ip_dit`, `packed_dit` and `packed_te`. `[SW]` the Krea multiphase lane collapses into the generic lane (gated on #3264). | ENG-2, ENG-3 | 5–6 wk |
| **ENG-5: FamilyDrivers I** | SDXL native (with inpaint, IP, control) + InstantID; Kolors native; Wan native UniPC (`Carry`); FLUX.3 flow-map; Boogu Turbo DMD; Sana Sprint SCM; MOSS-Image DMD; Iris FlowDPM (quanta hooks); Ideogram; Supra2 | ENG-4 | 5–6 wk |
| **ENG-6: FamilyDrivers II (multi-stream, autoregressive)** | LTX (explicit tables, route transition, learned upsampler component, AV streams, per-token time); MiniMax-H3 (two schedules); Mage; SenseNova (interleaved output); Bernini; Krea-realtime (streaming); SeedVR2; PiD sampler hosted in S10. **Exit:** PIPE-1 complete for every inventory row. | ENG-5 | 6 wk (likely 6a/6b) |
| **ENG-7: Route records + capability export + registration** | `RouteRecord`, defaults and limits on records; capabilities computed from records; hardware-free export (item 0: hosted candle build check); `Generator` cleanup and typed reports; typed request blocks; REG-6 bespoke providers registered. `[SW]` migrate every dump consumer, SceneWorks linking via bundles only, no direct `runtime_*::providers` imports. | ENG-6 | 5–6 wk |
| **ENG-8: Layer kit II** | `AttentionOp` (all raw SDPA), RoPE/qkv/norm/MLP/timestep kit, `ExecPolicy`, trainable op-set switch; delete training model copies (WGT-4) | ENG-6 | 5–6 wk |
| **ENG-9: Weights** | Codec registry, `WeightPlan` lifecycle, loaders migrated by dialect, per-component tiers, load cancellation, generic load progress, generated adapter routing | ENG-7 | 6 wk (likely 9a/9b) |
| **ENG-10: Latent codecs + R2** | KL-2D, WanCausal3D, causal-3D kit; tiling laws; quality bounds | ENG-8 | 5–6 wk |
| **ENG-11: Encoders on llm + towers** | `TapPolicy` on the existing tap API; codec, mask and architecture blockers; towers | ENG-8, ENG-9 | 5–6 wk |
| **ENG-12: BlockStack + conditioning + R3/R4** | `BlockStack`, block hooks, conditioning modules; R3 in `AttentionOp`, R4 in `BlockStack` | ENG-8, ENG-9 | 5–6 wk |
| **ENG-13: Computed memory contracts + R1 + iOS** | MEM-2 (delete the fixtures); R1 in the lifetime graph; retire `MemoryRequestScope`; MEM-6 iOS admission. `[SW]` migrate the admission contract input if APP-4 has landed. | ENG-10, 11, 12 | 5–6 wk |
| **ENG-14: Audio + LLM adoption** | XL-1/2 | ENG-9, ENG-13 | 5–6 wk |
| **ENG-15: DEC-1 (c) pilots** | Pilot A, pilot B, go/no-go; rollout waves as separate features | ENG-8…12; DEC-1 | 4–6 wk pilots; rollout 3–6 months `[L]` |

### 9.3 SceneWorks features (all `[SW]`; start after the DEC-5 gate)

| Feature | Scope | Depends on | Est. |
|---|---|---|---|
| **APP-1: Model registry** | **0** delete the backend-capability matrix (DEC-16); hand-table inventory (APPREG-2). Static and runtime layers; one merge rule; job carries the `ModelSpec` snapshot; manifest v2 with upgrader; evidence store (keys verbatim); seeding; consumer migration. The static layer ingests **today's** dumps, and ENG-7 later migrates that ingest (producer migrates consumers). | DEC-5 gate | 5–6 wk |
| **APP-2: Options core** | **0** control-inventory tool and "before" capture (§10.3). `OptionSpec` union seed; `validate()` at every boundary; wasm build (DEC-9); upgrader and versioning for all shapes; `IntentClass::Context`; option effects; typed refusal rendering; telemetry and stamp stability | APP-1 | 5–6 wk |
| **APP-3: Generated panels** | Image, Video, Audio, Training, Simple and Character studios; enumerated custom renderers; accessibility floor; delete `macGating.js`/`candleGating.js`/id gating. Zero diff on the control inventory. | APP-2 | 6 wk (split by studio if needed) |
| **APP-4: Admission pipeline** | **0** re-inventory by grep. Types, `BackendBudget`, per-route posture, decision-diff, ≥25 paths, typed refusal, one stamp, ADM-7 approvals. Consumes **today's** contracts; ENG-13 migrates that input later. | ENG-6 (`RenderPlan` for all routes); APP-1; #3264 merged | 5–6 wk |
| **APP-5: Generation lane I** | **0** `JobType` table. Image generation lane, one event consumer (on #3258), routing table | ENG-7 (REG-6), APP-1, APP-4 | 5–6 wk |
| **APP-6: Generation lane II** | Video, audio and utility lanes; backend module (on #3270); LANE-6 API end state | APP-5; audio lane after ENG-14 | 5–6 wk |
| **APP-7: Tooling** | Generic memory-adapter (DEC-18 run), typed dumps, delete the memory matrix (DEC-17) and mirrors | ENG-9, APP-1 | 4–5 wk |

### 9.4 Sequencing and calendar

```text
ENGINE  ENG-1 → ENG-2 ┐
              → ENG-3 ┴→ ENG-4 → ENG-5 → ENG-6 → ENG-7 → ENG-9 ─┐
                                          └→ ENG-8 ──────────────┼→ ENG-10 ┐
                                                                 ├→ ENG-11 ┼→ ENG-13 → ENG-14 → (ENG-15)
                                                                 └→ ENG-12 ┘
APP     (DEC-5 gate) APP-1 → APP-2 → APP-3
                         └──────────(after ENG-6)→ APP-4 → (after ENG-7) APP-5 → APP-6
                         └──────────(after ENG-9)→ APP-7
```

- **Calendar `[L]`.**
  - Engine ENG-1…14: about 14–16 features. With the two-way parallelism the dependency graph allows (ENG-2∥3, ENG-8∥9, ENG-10∥11∥12 on disjoint files), that is **about 13–16 months**.
  - SceneWorks APP-1…3 can run in months 2–7. APP-4…7 follow their engine producers, finishing around months 12–16.
  - DEC-1 (c) rollout adds 3–6 months if approved.
  - This is longer than draft 0.1. The reviews showed that 0.1's sequencing violated its own rules [ENG-5, APP-4].
- **Value milestones.**

| Milestone | Approximate time | What lands |
|---|---|---|
| M1 | ~1–1.5 months | Low-res start on FLUX.1 |
| M2 | ~3–4 months | Low-res start on most shared-driver image and video routes |
| M3 | ~5–6 months | Per-phase LoRA everywhere it is supported |
| M4 | ~7–9 months | Every route on the runner; one registry and options pipeline |
| M5 | ~10–12 months | One admission pipeline; one generation lane |
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
- **iOS size:** binary size is baselined in ENG-1, item 0.

---

## 11. Risks

| # | Risk | L/I | Mitigation |
|---|---|---|---|
| R1 | Program fatigue and branch death, now with a 13–16-month horizon | High/High | ≤6-week features to `main`; value milestones; no integration branch; re-plan after every three features |
| R2 | Numeric drift | Med/High | AP13 classes, bit-exact by design for pipeline adoption (variants, seed schemes, history policy); Tier 1 per batch; class (c) only with sign-off |
| R3 | Performance loss (dynamic dispatch, batching, lost MLX fusion) | Med/High | Families batch branches; dyn only at family granularity; `ExecPolicy::fuse`; PERF-1 A/B |
| R4 | Collisions with in-flight work | High/Med | DEC-5 gating; overlap ledger refreshed at each planning; file-ownership tables; new-model rule |
| R5 | Agent failure modes (regrowth, stubs, scope narrowing, boundary gaps) | High/Med | DEC-6 guards or terminal greps; verification contract; stub grep at item end; terminal integration sweep |
| R6 | iOS size and memory | Med/Med | Baseline in ENG-1; feature-partitioned crates; MEM-6 |
| R7 | Training regressions | Med/Med | Trainers in Tier 1 from ENG-8 |
| R8 | Hardware contention | High/Med | Live pool discovery per planning; Tier 1 off the dev Mac; batching |
| R9 | Decision latency (DEC-1…19) | Med/Med | Each feature depends on a named decision subset |
| R10 | Hidden coupling (API embeds worker; 17 family crates linked) | Med/Med | REG-5 and REG-6 in ENG-7; LANE-6 in APP-6 |
| R11 | MLX runtime and thread-state bugs invisible to in-crate tests | Med/High | Process-level worker validation whenever runner, `ExecPolicy` or residency changes |
| R12 | Disk exhaustion from agent `target/` directories | High/Med | Clean finished `target/`; cap building agents |
| R13 | Capability export cannot be produced hardware-free for candle | Med/Med | ENG-7 item 0 checks this first; otherwise the CUDA host goes on the critical path and the plan is re-cut |
| R14 | `FamilyDriver` claims diverge from behaviour | Med/High | Drivers generic over a forward seam; conformance suite (PIPE-7) |

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
