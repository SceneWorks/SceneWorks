# A — The gen-core contract layer as it exists (inference origin/main a43724b82)

Research topic A for the SceneWorks + inference re-architecture. Read-only survey of the
backend-neutral contract surface that the ~83 engine crates implement and that SceneWorks consumes.

**Citation conventions.** Every path is on inference `origin/main` (a43724b82) unless prefixed
`SW:` (SceneWorks worktree `claude/multi-resolution-inference-848ec4`, pinned to the same inference
rev). Abbreviations:

- `gc/` = `crates/contracts/gen-core/src/`
- `mlx/` = `crates/media/mlx-gen/src/` (the MLX engine core crate)
- `cdl/` = `crates/media/candle-gen/candle-gen/src/` (the Candle engine core crate)
- `fam/` = `crates/contracts/gen-core-family/src/`

`file:N` is a line number; `file:N-M` a range. Line numbers were read with `git show origin/main:<path> | nl`.

---

## 0. Executive summary (design-relevant facts)

1. **One object-safe trait, `Generator`, is the entire engine contract** (`gc/generator/traits.rs:8-285`):
   2 required methods (`validate`, `generate`) + `descriptor`, plus **17 default-implemented methods**, of
   which **6 are family-named** (MOSS ×4, Veda ×2). There is *no* pipeline/stage trait: each engine's
   `generate` is a black box that owns encode → denoise loop → decode. The deferred `Pipeline` trait
   promised in `crates/media/mlx-gen/ARCHITECTURE.md:79-81` ("introduced when the second family lands")
   was never introduced; ~43 MLX + ~40 Candle family crates exist.
2. **`GenerationRequest` is a 61-field `Default`-able union struct** (`gc/generator/request.rs:9-381`).
   Extension is by adding `Option<_>` fields (no extras map, no typed extension slots). Roughly half
   the fields are family-specific (FLUX 3, MOSS, Krea, Wan-VACE, SVD, SeedVR2, LTX DFR, H3/Veda, PuLID,
   PiD, Iris, Qwen 2.1). The shared validation floor contains hard-coded **model-id string checks**
   (`gc/generator/capabilities.rs:657-698`).
3. **Factory = explicit registry of fn pointers**: `ModelRegistration { descriptor: fn()->ModelDescriptor,
   load: fn(&LoadSpec)->Result<Box<dyn Generator>>, footprint }` (`gc/registry/registration.rs:97-107`),
   resolved by id string (`gc/registry/providers.rs:41-54`). Backend is chosen **at compile time** by
   which runtime bundle is linked (`SW: crates/sceneworks-worker/src/inference_runtime.rs:17-20`); tier
   is `LoadSpec.quantize` (`gc/runtime/load_spec.rs:54`). There is no (id, backend, tier) → engine
   lookup; there is (id) → `fn(&LoadSpec)`.
4. **Existing cross-cutting hook mechanisms already in the contract** (the seeds of a hook layer):
   - `MemoryRequestScope` lifecycle trait with `configure_request / enter_phase / leave_phase /
     configure_decode / configure_attention / materialize_transformer_window / finish`
     (`gc/memory_strategy/run.rs:191-217`).
   - `Residency<Text, Heavy, R>` phase-order driver with `prepare → encode → materialize_encoded →
     denoise → materialize_mid → decode` closures (`gc/residency.rs:623-640`) — the closest thing to a
     shared pipeline skeleton, adopted by ~29 crates.
   - `block_window::run_windowed` + `BlockWindowBackend` (`gc/block_window.rs:152-230`).
   - The unified sampler: `Sampler<L: LatentOps>::sample(ops, ms, denoise, x, sigmas, seed)`
     (`gc/sampling/unified.rs:29-50`) — loop is *inside* the solver; the only per-step hooks are the
     `denoise` closure and `LatentOps::observe_solver`.
   - Approximation surface (`gc/approximation.rs`) — the precedent for **result-changing** levers
     (feature cache, token pruning), gated per-capability and currently refused unless characterized.
5. **Sampling orchestration is per backend and per engine.** `run_curated_sampler`/`run_flow_sampler`
   exist in both backends with **different signatures** (MLX: `SamplerRun` struct + latent hook,
   `mlx/sampler.rs:422-772`; Candle: 9 positional args + `PreviewHook`, `cdl/sampler.rs:188-275`).
   `resolve_schedule`/`resolve_flow_schedule` are **tensor-free yet byte-duplicated** in both backends
   (`mlx/sampler.rs:904-934`, `cdl/sampler.rs:282-322`). img2img `init_time_step` is shared on MLX
   (`mlx/img2img.rs:50-59`) but re-implemented in ≥6 Candle families. `mu` is computed by 4+ different
   family functions.
6. **Multi-phase denoise (Krea) is the existing "split one trajectory" primitive**: one global sigma
   schedule sliced into contiguous phases (`fam/krea_multiphase.rs:1-45,86,190`), driven by calling
   `run_flow_sampler` once per slice from the running latent. It lives in a *family* crate, and the
   `phases` request field has **no capability flag** — unsupported models silently ignore it
   (`gc/generator/request.rs:316-330`).

---

## 1. Crate map — `crates/contracts/*` and `crates/bundles/*`

LOC = total lines of `.rs` files (src + tests + examples) on origin/main.

| Crate (package) | .rs files | LOC | What it holds |
|---|---:|---:|---|
| `gen-core` (`sceneworks-gen-core`, lib `gen_core`) | 143 | 87,362 (68,951 non-test) | Contract traits, request/output/descriptor, registry, memory-strategy contract, residency/tiling/attention/block-window planners, sampling framework, checkpoint codec + weightsmeta, licenses, encoder contract, family request-surface halves (MOSS, FLUX3, Veda, H3) |
| `gen-core-family` | 16 | 5,826 | Small single-family host policy: Krea multiphase + schedule, Ideogram scheduler, Lens resolution, Qwen-Image 2.1 config/reference/scheduler, SD3 encoder artifacts/request, SDXL LDM, SeedVR2 video, InstantID kps, face align, FX program (`crates/contracts/gen-core-family/src/lib.rs` header) |
| `gen-core-checkpoint` | 5 | 3,191 | torch `.pth` zip+pickle reader, Supra2 converter (`.../gen-core-checkpoint/src/lib.rs` header) |
| `gen-core-flux3` | 41 | 18,550 | FLUX 3 Klein: frozen upstream identity, package layout, conversion, PUP prompt upsampler, references/layouts, phase resources, training |
| `gen-core-iris` | 20 | 22,709 | Iris-3B task identity, config, FlowDPM step plan, restoration/depth, training incl. distributed contract |
| `gen-core-ltx` | 6 | 6,042 | LTX split-checkpoint resolver, Gemma assets, DFR keyframes, duration head |
| `gen-core-h3` | 14 | 11,292 | MiniMax-H3 memory planners, Veda predictor/top-k/workspace, reference, audio config, keyframes, schedule, chunking, pruned adaLN |
| `gen-core-memory-wan` | 10 | 5,418 | Wan2.2 I2V sealed physical/request identity, inventory, receipts, routes, selection |
| `gen-core-moss-image` | 9 | 4,015 | MOSS schedule/DMD/guidance math, package validation, edit sizing, training admission/recovery |
| `gen-core-image` | 4 | 1,509 | OpenEXR I/O + HDR conversions |
| `gen-core-train` | 10 | 6,261 | Trainer helpers: request validation, aspect buckets, subject masks, aux schedules, resume, body/face loss geometry |
| `gen-core-tokenizer` | 1 | 436 | HF `tokenizers` wrapper + typed chat template |
| `gen-core-dist` | 2 | 2,134 | Iris TCP collective transport |
| `gen-core-telemetry` | 1 | 218 | **H3-only** opt-in per-step RSS/wall-time `Collector` (`.../gen-core-telemetry/src/lib.rs:1-102`) |
| `gen-core-testkit` | 31 | 15,404 | Conformance suites: `check_validate_honesty`, `check_progress`, `check_cancellation`, `check_precancellation`, `check_seed_determinism`, `check_cfg_off_render`, `check_component_load_gate`, `conformance` (`.../gen-core-testkit/src/generator_checks.rs:175-926`), plus audio/trainer/captioner/memory-strategy/registry checks |
| `load-progress` | 2 | 1,089 | Request-scoped I/O progress (`LoadActivity`, `LoadActivityContext`, `LoadActivityObserver`) re-exported by `gc/load_activity.rs:2` |
| `core-llm` (+ `core-llm-testkit`) | 44 | 29,515 | Tensor-free LLM contract (`TextLlm`), registry, templates, speculative, KV compression; re-exported as `gen_core::core_llm` (`gc/lib.rs:293-304`) |
| `bundles/runtime-catalog` | 1 | 1,830 | `RuntimeCatalog` validation + `RuntimeCatalogSnapshot` JSON |
| `bundles/runtime-macos` | 6 | 10,915 | MLX media + MLX LLM + Candle audio + perf bench |
| `bundles/runtime-cuda` | 1 | 552 | Candle CUDA media + LLM + audio |
| `bundles/runtime-cpu` | 1 | 628 | Candle CPU |
| `bundles/runtime-ios` | 1 | 1,820 | MLX iOS subset + per-family admitted facades |

gen-core module-group sizes (src incl. in-module tests): `memory_strategy` 9,029; `generator` 8,038;
`registry` 7,710; `license` 6,461; `encoder_contract` 6,203; `runtime` 5,731; `sampling` 4,953;
`checkpoint_codec` 4,774. The crate claims "zero tensor dependencies" (`gc/lib.rs:1-14`); numeric
types restricted to `f32/f64/Vec<f32>/Vec<i32>/&[u8]`.

gen-core still carries family-named modules despite sc-27316's split: `flux3`, `h3_adapter_memory`,
`h3_conditioning_memory`, `h3_memory`, `h3_reference`, `moss_image`, `moss_image_edit`,
`moss_image_assets`, `veda` (`gc/lib.rs:57-77,329`), and named checkpoint-adapter constants
`FLUX2_/KREA_2_/MAGE_FLOW_/QWEN_IMAGE_/SDXL_/WAN_/Z_IMAGE_CHECKPOINT_ADAPTER` (`gc/lib.rs:265-267`), plus
per-family `LatentSpace` constants (`gc/latent.rs:343-543`) and `VaeTiling::{LTX,WAN,WAN22,QWEN_IMAGE,
QWEN_IMAGE_2_1}` (`gc/tiling.rs:93-154`). The stated rule (`docs/architecture/inference-rearchitecture.md:323-330`)
is that gen-core keeps "only what the contract names — a request field's type, a `Generator` method's
argument" — i.e. family types stay in gen-core *because* the contract names them.

---

## 2. Provider traits

### 2.1 `Generator` (`gc/generator/traits.rs:8-285`)

`pub trait Generator` — object-safe, **not** `Send`/`Sync`-bounded, synchronous ("the worker runs each
job on its own thread", `traits.rs:5-7`).

| # | Method | Line | Default | Generic? |
|---|---|---:|---|---|
| 1 | `descriptor(&self) -> &ModelDescriptor` | 10 | required | generic |
| 2 | `generate_moss_image_edit(&self, &GenerationRequest, MossImageEditInputs, &mut dyn FnMut(Progress)) -> Result<MossImageAdvancedOutput>` | 13-22 | `Err(Unsupported)` | **MOSS-only** |
| 3 | `moss_image_edit_request_peak(&self, &GenerationRequest, MossImageEditInputs) -> Result<u64>` | 24-32 | `Err(Unsupported)` | **MOSS-only** |
| 4 | `moss_image_edit_execution_report(&self) -> Option<MossImageEditExecutionReport>` | 34-38 | `None` | **MOSS-only** |
| 5 | `veda_execution_report(&self) -> Option<VedaExecutionReport>` | 41-43 | `None` | **H3/Veda-only** |
| 6 | `moss_image_execution_report(&self) -> Option<MossImageExecutionReport>` | 49-51 | `None` | **MOSS-only** |
| 7 | `moss_image_owned_weight_bytes(&self) -> Option<u64>` | 57-59 | `None` | **MOSS-only** |
| 8 | `adapter_apply_reports(&self) -> Vec<AdapterApplyReport>` | 67-69 | empty | generic |
| 9 | `checkpoint_weight_facts(&self) -> Option<CheckpointWeightFacts>` | 90-92 | `None` | generic (opt-in) |
| 10 | `memory_strategy_contract(&self) -> Option<&MemoryProviderContract>` | 96-98 | `None` | generic (opt-in) |
| 11 | `structural_resident_evidence(&self) -> Option<&MemoryStructuralResidentEvidence>` | 102-104 | `None` | generic |
| 12 | `predicted_memory_peak_from_base(&self, u64) -> MemoryPeakBreakdown` | 109-117 | via contract | generic |
| 13 | `veda_workspace_bound(&self, veda::workspace::Request) -> Result<Option<WorkspaceBound>>` | 121-126 | `Ok(None)` | **H3/Veda-only** |
| 14 | `memory_strategy_safety_check(&self, &MemoryRunContext) -> MemorySafetyDecision` | 131-144 | default safety check / Resident-only | generic |
| 15 | `begin_memory_strategy_request(&self, &MemoryRunContext) -> Result<Option<Box<dyn MemoryRequestScope + '_>>>` | 148-163 | `Ok(None)` (errors if contract advertises Implemented optimized rung) | generic |
| 16 | `validate(&self, &GenerationRequest) -> Result<()>` | 167 | required | generic |
| 17 | `generate(&self, &GenerationRequest, &mut dyn FnMut(Progress)) -> Result<GenerationOutput>` | 170-174 | required | generic |
| 18 | `generate_streaming(&self, req, on_chunk: &mut dyn FnMut(AudioChunk), on_progress) -> Result<GenerationOutput>` | 209-225 | calls `generate`, emits one chunk | audio |
| 19 | `generate_with_report(&self, req, on_progress) -> Result<GenerationReport>` | 241-248 | wraps `generate` | generic (YuE2 origin) |
| 20 | `open_conversation(&self, &GenerationRequest) -> Result<Box<dyn ConversationSession + '_>>` | 275-284 | `Err(Unsupported)` | audio |

Observations:
- **Every extension has been made by adding a default method** (documented rationale at
  `traits.rs:187-193` for streaming: `Progress` is `Copy + Eq` and exhaustively matched, so new payloads
  go into new methods instead). This is additive but grows a single god-trait.
- Post-generate **receipts are side-channel getters on `&self`** (`*_execution_report`,
  `adapter_apply_reports`) that require the caller to serialize generation and receipt retrieval
  (`traits.rs:45-48`). There is no per-request result object other than `GenerationReport`.
- `requires_memory_request_scope` (`traits.rs:287-295`) makes the default `begin_memory_strategy_request`
  fail if a provider advertises an implemented optimized rung but did not override it.

`ConversationSession` (`traits.rs:317-345`): `step(&mut self, &ConversationTurn, on_chunk, on_progress)
-> Result<AudioTrack>`, `finish(&mut self) -> Result<()>`. Law: session ≡ batch `ConversationHistory`
render (`traits.rs:304-311`).

### 2.2 Other provider / helper traits in gen-core

`git grep '^pub trait' gc/` yields 26 traits:

| Trait | Location | Role |
|---|---|---|
| `Generator`, `ConversationSession` | `generator/traits.rs:8,317` | media generation |
| `Transform` | `transform.rs:14` | non-prompt image→image (SeedVR2 design target; SeedVR2 still registers as Generator, `mlx-gen/docs/MODEL_ARCHITECTURE.md:149-154`) |
| `Trainer` | `train/request.rs:242` | LoRA/LoKr training |
| `Captioner`, `Transcriber` | `caption.rs:14`, `transcribe.rs:18` | image/audio→text |
| `ImageEmbedder`, `TextEmbedder`, `VoiceEmbedder`, `AudioEmbedder`, `FaceEmbedder` | `image_embed.rs:25`, `text_embed.rs:14`, `voice_embed.rs:31`, `audio_embed.rs:44`, `face.rs:40` | embeddings |
| `AudioTransform` | `audio_transform.rs:49` | audio→audio |
| `ControlBranch: Generator` | `control.rs:77` | shared ControlNet/Fun-Union boilerplate as provided methods (id, accepted kinds, message overrides, `resolve_control`) |
| `MossImageAdvancedGenerator` | `moss_image.rs:1118-1125` | **family trait in the contract crate** (`generate_moss_advanced`) |
| `MemoryRequestScope` | `memory_strategy/run.rs:191` | per-request memory lifecycle hooks (§7.5) |
| `ResidencyRuntime`, `StagedHeavy` | `residency.rs:18,45` | backend hooks for the residency driver (§8.1) |
| `BlockWindowBackend` | `block_window.rs:152` | backend hooks for rung 4 (§8.4) |
| `SamplerPolicy` (legacy), `Sampler<L>`, `CfgPpSampler<L>`, `LatentOps`, `ModelSampling`, `GuidanceOps: LatentOps` | `sampling.rs:100`, `sampling/unified.rs:29`, `sampling/cfgpp.rs:41`, `sampling/latent_ops.rs:26`, `sampling/model_sampling.rs:41`, `guidance.rs:88` | sampling framework (§10) |
| `CodecResidencyPolicy`, `LogicalKeyMapping` | `checkpoint_codec/spec.rs:110`, `checkpoint_codec/transform.rs:11` | declarative weight plan (§12) |

### 2.3 Macros that implement/registers traits

- `register_generators!` (`gc/macros.rs:5-25`): defines a `const ModelRegistration` from a descriptor
  fn path and a load fn path, optional `; footprint = path`. Parallel macros for trainer, captioner,
  transcriber, image/text/voice/audio embedder, audio transform (`gc/macros.rs:27-125`). No `Transform`
  macro.
- `impl_generator!` (`gc/macros.rs:127-156`): implements only `descriptor/validate/generate` by
  delegating to `self.descriptor` and a named inherent method.
- `delegate_memory_request_scope!` (`gc/memory_strategy/run.rs:147-180`): forwards the scope hooks to a
  field (backends wrap `MlxRequestScopeCore`/`CandleRequestScopeCore`).

---

## 3. Registry, factory, catalog — how an engine is chosen and built

### 3.1 Registration records (`gc/registry/registration.rs`)

| Record | Line | Shape |
|---|---:|---|
| `ModelRegistration` | 97-107 | `descriptor: fn() -> ModelDescriptor`, `load: fn(&LoadSpec) -> Result<Box<dyn Generator>>`, `footprint: Option<fn(&LoadSpec) -> Result<PerComponentBytes>>` |
| `PerComponentBytes` | 35-92 | `text_encoder`, `dit`, `vae` u64 + `from_spec_subdirs` / `from_root_subdirs` |
| `MemoryRegistration` | 333-344 | `provider_id`, `contract: fn(&LoadSpec)->Result<MemoryProviderContract>`, `safety_check: fn(&LoadSpec,&MemoryProviderContract,&MemoryRunContext)->MemorySafetyDecision` |
| `MemoryContractFixtureRegistration` | 352-363 | provider_id, contract fn, `surface_specs: fn()->Vec<MemoryContractSurfaceSpec>` |
| `MemoryContractSurfaceResolverRegistration` | 373-377 | provider_id, `contract: fn(&MemoryContractSurfaceSpec)->Result<...>` |
| `ResidentOnlyMemoryContractRegistration` | 384-389 | resident-only contract + surface specs |
| `MemoryBehaviorRegistration` | 477-486 | `valid_fixtures: fn(&LoadSpec,&Contract,MemoryStrategy)->Result<Vec<MemoryBehaviorFixture>>`, `begin_request: MemoryBehaviorBeginRequest` (468-472) |
| `MemoryBehaviorFixture` | 395-466 | `context: MemoryRunContext`, `request: GenerationRequest`, `load_spec: Option<LoadSpec>`; `new()` synthesizes a request from geometry |
| `ActivationMemoryRegistration` | 493-497 | provider_id + `ActivationMemoryAnchor { bytes_1024 }` |
| `ReferenceWorkspaceRegistration` | 503-508 | `fn(&LoadSpec,&MemoryReferenceResourceFacts)->Result<u64>` |
| `EncoderContractRouteRegistration` | 319-325 | `route_id`, `provider_id` |
| `CheckpointAdapterRegistration` + 10 sub-records | 114-290 | imported-checkpoint dialects/signatures/topology/base-compatibility/canonical mappings/config recovery/backend bindings/operations/capabilities |
| `ImportedModelRegistration` | 292-310 | family, source, operation, provider_id, required_components, inherit_adapters |
| `Transform/AudioTransform/Trainer/Captioner/Transcriber/ImageEmbedder/TextEmbedder/VoiceEmbedder/AudioEmbedderRegistration` | 511-581 | uniform `{ descriptor: fn()->D, load: fn(&LoadSpec)->Result<Box<dyn T>> }` |

### 3.2 `ProviderRegistryBuilder` → `ProviderRegistry`

- Builder holds 22 `Vec`s, one per registration kind + `rejected_quants: Vec<(Quant,&'static str)>`
  (`gc/registry/builder.rs:9-32`). `register_*` methods are macro-generated pushes (`builder.rs:34-41`);
  `register_checkpoint_codec` (491), `register_composed_memory_strategy` (543), `reject_quant(quant,
  reason)` (593), `build()` (599) validates duplicate ids, adapter catalogs, memory-registration ↔
  generator pairing (see comment at `crates/media/candle-gen/candle-gen-catalog/src/lib.rs:116-118`).
- `ProviderRegistry` (`gc/registry/providers.rs:6-30`) is immutable, boxed slices.
- **Load path** (macro `explicit_registry_kind!`, `providers.rs:32-56`):
  ```rust
  pub fn load(&self, id: &str, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
      let registration = self.generators().find(|r| (r.descriptor)().id == id)...;   // linear scan, calls descriptor() per entry
      self.ensure_quant_supported(id, spec, "generator")?;
      let _activity = spec.load_activity_context().install();
      spec.read_prepared_files_unchanged(|| (registration.load)(spec))
  }
  ```
  Same macro yields `load_transform`, `load_audio_transform`, `load_trainer`, `load_captioner`,
  `load_transcriber`, `load_image_embedder`, `load_text_embedder`, `load_voice_embedder`,
  `load_audio_embedder` (`providers.rs:616-695`).
- `ensure_quant_supported` (`providers.rs:570-614`) is the single load gate. It contains a **family
  leak**: it refuses non-default `spec.yue2_ar_mode` / `yue2_compute_policy` for any id other than
  `"yue2"` (576-587), then checks `rejected_quants`, then special-cases `Quant::Fp8Rowwise` against the
  descriptor's `supported_quants` (601-610).
- Other lookups: `footprint(id, spec)` (702-716), `memory_strategy_contract(id, spec)` (722),
  `memory_contract_surfaces()` (445), `provider_encoder_contract(id)` (292), `imported_model_descriptor`
  (341), `activation_memory_bytes_1024` (368), `reference_workspace_bytes` (387),
  `descriptor_conformance_errors()` (756), `checkpoint_adapter_catalog_conformance_errors(other)` (78).

### 3.3 Composition chain (family → catalog → bundle → product)

1. **Family crate** exposes `pub fn register_providers(ProviderRegistryBuilder) -> ProviderRegistryBuilder`
   chaining `register_generator(REGISTRATION)`, encoder routes, checkpoint adapters, and (cfg-gated)
   memory behaviour/surface registrations — e.g. Candle Z-Image
   `crates/media/candle-gen/candle-gen-z-image/src/lib.rs:115-145`.
2. **Platform catalog**: `candle-gen-catalog::register_providers` registers the engine codec table once,
   then calls 31 family builders in fixed order (`crates/media/candle-gen/candle-gen-catalog/src/lib.rs:90-135`);
   `provider_registry()` (138-140); `memory_contract_surface_registry()` (145-170) adds contract-only
   surfaces on non-CUDA builds; `vae_tiling(provider_id)` chains family lookups (176-180).
   `BESPOKE_UTILITY_CRATES` (60) and `BESPOKE_MEMORY_ROUTE_WAIVERS` (79-87, PuLID) list things that do
   *not* go through the registry. MLX twin: `crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs` (not read
   line-by-line; same pattern per `docs/architecture/inference-rearchitecture.md:173-185`).
3. **Runtime bundle** (`crates/bundles/runtime-{macos,cuda,cpu,ios}`): constants `PLATFORM`, `BACKEND`,
   `AUDIO_BACKEND`, `SUPPORTED_TARGET_TRIPLES`, `NATIVE_PREREQUISITES` (cuda `lib.rs:56-69`; macOS
   `lib.rs:120-134` — macOS audio backend is **candle**), `catalog() -> RuntimeCatalog` (cuda 133+,
   macOS 192), `memory_contract_surface_registry()`, plus bundle-level helpers that are really
   family/VAE policy (`conservative_video_decode_memory_profile`, `budgeted_video_decode_memory_profile`,
   `resolved_video_memory_numeric_tier`, macOS `lib.rs:33-113`). Re-exports the whole provider set as
   `providers::*` (cuda `lib.rs:39-41`).
4. **`RuntimeCatalog`** (`crates/bundles/runtime-catalog/src/lib.rs:81-219`): `try_new(platform,
   backend, media, text, preparers)` validates that every descriptor/preparer belongs to the declared
   backend; accessors `media()`, `text()`, `preparers()`, `audio()`, `snapshot()`;
   `RuntimeCatalogSnapshot::to_json` (843) is the weights-free capability dump SceneWorks checks in.
5. **iOS** bypasses generic `load(id)` for media: per-family admitted facades `load_sana`, `load_sdxl`,
   `load_anima_turbo`, `load_z_image_turbo` (and `_with`, `_with_observer` variants)
   (`crates/bundles/runtime-ios/src/lib.rs:402-483`), behind a generic `load_admitted` guard (header
   lines 18-25).

### 3.4 How a caller picks and builds an engine (model id + backend + tier)

- **Backend**: compile-time. `SW: crates/sceneworks-worker/src/inference_runtime.rs:17-20`
  (`#[cfg(target_os="macos")] use runtime_macos as platform_runtime;` else `feature="backend-candle"` →
  `runtime_cuda`). One process = one backend. `ModelDescriptor.backend: &'static str` (`"mlx"|"candle"`,
  `gc/generator/descriptor.rs:31-34`) is informational.
- **Model id**: SceneWorks keeps its own `MODEL_TABLE: &[ModelRow]` mapping `sceneworks_id → engine_id`
  plus `default_repo`, `default_steps`, `default_guidance`, `adapter_label`
  (`SW: crates/sceneworks-worker/src/engines.rs:22-37`), then joins with the descriptor by scanning the
  registry (`engines.rs:990-1000`, `mlx_model`). **Defaults (steps/guidance) are not on the descriptor**
  (`git grep default_steps gc/generator` → none); they live in SceneWorks.
- **Tier**: `LoadSpec.quantize: Option<Quant>` + `precision: Precision` (`gc/runtime/load_spec.rs:54-55`);
  `Quant::{Q4,Q8,Nvfp4,Fp8Rowwise}` (`gc/runtime/quant.rs:14-32`). Descriptor advertises
  `supported_quants` and `component_precision_floors` (`gc/generator/capabilities.rs:375-379`); the
  bundle can `reject_quant` platform-wide (`gc/registry/builder.rs:593`).
- **Build**: `inference_runtime::load(id, spec) = media().load(id, spec)` (`SW: inference_runtime.rs:426-428`).
- SceneWorks also depends **directly** on many family crates (e.g. `mlx-gen-krea`, `mlx-gen-ltx`,
  `candle-gen-ltx`, `mlx-gen-moss-image`, `candle-gen-scail2` — `SW: crates/sceneworks-memory-adapter/Cargo.toml:24-56`,
  `SW: crates/sceneworks-worker/Cargo.toml`), contrary to the invariant "every supported product
  consumes one named runtime bundle rather than assembling backend crates"
  (`docs/architecture/inference-rearchitecture.md:337-338`).

---

## 4. `ModelDescriptor` and `Capabilities`

### 4.1 `ModelDescriptor` (`gc/generator/descriptor.rs:20-100`)

| Field | Line | Notes |
|---|---:|---|
| `encoder_contract: Option<EncoderContract>` | 24 | substitutable text-encoder contract |
| `denoiser_output_latent_space: Option<&'static LatentSpace>` | 28 | 71 `Some` vs 8 `None` sites in `crates/media` |
| `id: &'static str` | 29 | registry key |
| `family: &'static str` | 30 | stringly |
| `backend: &'static str` | 34 | `"mlx"|"candle"`, stringly |
| `modality: Modality` | 35 | `Image|Video|Both|Audio` (`descriptor.rs:6-16`) |
| `capabilities: Capabilities` | 36 | |
| `required_components: &'static [&'static str]` | 52 | ids the consumer must stage into `LoadSpec.components` |
| `control_kinds: Option<AcceptedControlKinds>` | 80 | |

`compatible_decoder_options()` (`descriptor.rs:86-99`) filters the global `DECODER_OPTIONS` table by
backend/provider id/latent-space compatibility. Supporting enums: `SizeFloor` (`108-164`: `RangeChecked`,
`RangeCheckedOnGrid{multiple}`, `ResolvedDownstream`, `ResolvedDownstreamExplicitGrid{multiple}`),
`PrecisionFloorComponent`/`ComponentPrecisionFloor` (172-224), `ActivationMemoryAnchor` (236-239),
`StagedResidencyAvailability` (243-255), `StepSupport::{Unconstrained, Exact(Vec<u32>), Range{min,max}}`
(282-344).

### 4.2 `Capabilities` (`gc/generator/capabilities.rs:49-450`, 54 pub fields)

`#[derive(Default)]` = "supports nothing"; every construction must end in `..Default::default()`, enforced
by the `capabilities_are_constructed_additively` integration test that parses every `.rs` file
(`capabilities.rs:5-47`; `crates/contracts/gen-core/tests/capabilities_are_constructed_additively.rs`).
Measured cost of adding a field: 1 file / 1 line (vs 70 files pre-conversion) (`capabilities.rs:29-44`).

Field groups (line numbers):
- Prompt/guidance: `supports_negative_prompt` 50, `supports_guidance` 51, `supports_true_cfg` 52,
  `supported_guidance_methods: Vec<&'static str>` 61, `supports_cfg_interval` 176.
- Conditioning/adapters: `conditioning: Vec<ConditioningKind>` 53, `supports_lora` 54, `supports_lokr` 55.
- Sampling menu: `samplers: Vec<&'static str>` 56, `schedulers: Vec<&'static str>` 57,
  `supported_steps: StepSupport` 106, `requires_sigma_shift` 382.
- Geometry/count: `min_size` 62, `max_size` 63, `max_count` 64, `size_floor` 129.
- Platform: `mac_only` 107, `supported_quants` 375, `component_precision_floors` 379, `supports_kv_cache` 381.
- Audio (≈15 fields): `audio_sample_rates` 137 … `supports_song_cover` 278.
- Output/UI: `supports_preview` 162, `supports_hdr` 172, `supports_alpha_output` 207,
  `supports_streaming` 226, `supports_prompt_enhancement` 215, `max_prompt_tokens` 191.
- Batch/caption: `supports_prompt_batch` 179, `supports_caption_overflow_policy` 182.
- LTX-2.5 axes: `supports_auto_duration` 308, `supports_generated_keyframes` 317,
  `max_temporal_upsample_rounds` 328, `supports_diffusion_decoder` 343.
- Conversation: `supports_conversation_history` 355, `supports_conversation_session` 367.
- Memory: `request_memory_strategies: &'static [MemoryStrategy]` 290 (audio lane),
  `supports_sequential_offload` 396, `unconditionally_engages_staged_residency` 404,
  `execution: ExecutionSurface` 423, `approximation: ApproximationSurface` 444.

**Missing**: no `supports_phases`/multi-phase flag, no default steps/guidance/shift, no
resolution-schedule or step-hook capability. Defaults live either in engine code or in SceneWorks'
`MODEL_TABLE`.

### 4.3 The shared validation floor

`Capabilities::validate_request(id, req)` (`capabilities.rs:561-601`) → `validate_request_inner`
(651+); variants `validate_request_skip_size` (613) and `validate_request_audio` (626);
`approximation_plan(id, req)` (640-646) is "exactly one place in the workspace that turns a request
into a plan". Free helpers: `reject_unsupported_adapters` (459-472), `max_enhance_tokens_for_model`
(489-494 — **id-string match on `"flux3_klein" | "flux3_klein_base"`**).

Hard-coded family/id checks inside the *shared* floor (`capabilities.rs:657-698`):
- `flux3_layout`/`flux3_layout_bindings`/`flux3_reference_output` allowed only for
  `flux3::DISTILLED_MODEL_ID | BASE_MODEL_ID` (657-676);
- `moss_image` requires `MossImageVariant::from_provider_id(id)` (688-694);
- `veda.enabled` requires `id == "minimax_h3"` (695-699).
Generic checks then follow: HDR (705), cfg_interval (713-724), prompt_batch (725-747), caption_overflow
(748), RGBA (758), `req.memory` calibration-fault gating (765-768), `self.execution.validate(id,
req.memory)` (788), `steps == Some(0)` refusal (811), finiteness floor via
`GenerationRequest::first_nonfinite_float` whose **exhaustive destructure** forces every new field to be
classified (`gc/generator/request.rs:540-600`). `phases` is **not** checked by the floor.

---

## 5. Request, params, conditioning, load spec

### 5.1 `GenerationRequest` (`gc/generator/request.rs:9-381`, 61 pub fields, `Clone + Debug`, `Default` at 383-452)

"The request union (lifted from the SceneWorks worker's `ImageRequest`/`VideoRequest`). Most fields
are optional; a model reads what it supports and `validate()` rejects the rest. A single
`Default`-able struct (no builder)" (`request.rs:5-7`).

| Group | Field (line) | Type | Generic vs family |
|---|---|---|---|
| Core | `prompt` 11 | `String` | generic |
| | `moss_image` 13 | `Option<MossImageOptions>` | **MOSS** |
| | `negative_prompt` 14, `width` 15, `height` 16, `count` 18 | | generic |
| Sampling | `seed` 21, `steps` 22, `guidance` 23, `true_cfg` 24 | `Option` | generic |
| | `timestep_to_start_cfg` 30 | `Option<u32>` | "Today only PuLID-FLUX honors it" (25-29) |
| | `sampler` 31, `scheduler` 32 | `Option<String>` | generic, **stringly** |
| | `scheduler_shift` 33 | `Option<f32>` | generic |
| | `guidance_method` 42 | `Option<String>` | generic, stringly (`cfg|cfg_rescale|apg|cfg_pp`) |
| | `guidance_eta` 46, `guidance_momentum` 49, `guidance_norm_threshold` 52 | `Option<f32>` | APG-only params at top level |
| | `cfg_interval` 59 | `Option<(f32,f32)>` | generic (gated) |
| | `prompt_batch` 67, `caption_overflow` 71 | | Iris origin (gated) |
| Conditioning | `conditioning` 74 | `Vec<Conditioning>` | generic |
| | `flux3_reference_output` 81, `flux3_layout` 85, `flux3_layout_bindings` 88 | | **FLUX 3** |
| | `strength` 90 | `Option<f32>` | generic img2img |
| | `control_scale` 95 | | **Wan-VACE** |
| | `text_style_gain` 104 | | **Krea/Qwen** |
| | `image_guidance` 112 | | **FLUX.2 edit** (+ `FLUX2_IMG_GUIDANCE` env override) |
| Video | `frames` 115, `fps` 116, `duration` 117, `video_mode` 118 (`Option<String>`) | | generic video |
| | `veda` 120, `h3_terminal_telemetry` 123 | | **H3** |
| | `trim_first_frames` 129 | | **Wan** |
| | `num_generated_keyframes` 148, `temporal_upsample_rounds` 195, `auto_duration` 204 | | **LTX-2.5** |
| | `reference_image_short_edge` 169, `reference_image_resize_mode` 184 | | **H3** |
| | `motion_bucket_id` 210, `noise_aug_strength` 214, `decode_chunk_size` 218, `conditioning_fps` 224 | | **SVD** |
| | `softness` 232 | | **SeedVR2** |
| Prompt enhancement | `enhance_prompt` 241, `use_uncensored_enhancer` 245, `enhance_max_tokens` 248, `enhance_temperature` 251, `prompt_enhancement: PromptEnhancementSink` 254 | | LTX/FLUX.2/FLUX 3 |
| Decoder | `use_pid` 264, `pid_capture_sigma` 278 | | **PiD** latent spaces |
| Memory | `memory` 288 | `Option<GenerationMemory>` | generic ("quality-preserving") |
| Approximation | `approximation` 307 | `Option<ApproximationRequest>` | generic ("result-changing") |
| Audio | `audio` 314 | `Option<AudioParams>` | audio sub-block |
| Multi-phase | `phases` 330 | `Option<Vec<GenerationPhase>>` | Krea (doc says "Krea MLX today"; Candle Krea also consumes it) |
| Control | `cancel: CancelFlag` 333, `preview: PreviewSink` 348 | | generic |
| HDR | `hdr` 362 | `Option<HdrRequest>` | LTX-2.5 (gated) |
| Alpha | `output_channels` 380 | `OutputChannels` | Qwen 2.1 (gated) |

Helpers on the request (`request.rs:454-1160`): `image_reference_count`, `memory_reference_count`,
`first_nonfinite_float` (exhaustive), `ensure_finite_floats`, borrowed views `keyframes()`,
`video_clips()`, `control_clip()`, **`scail2_animation_conditioning()`** (1036, family-named),
`audio_edit()`, `audio_edit_regions()`; free fns `default_seed` (1163),
`effective_reference_image_short_edge` / `validate_reference_image_short_edge` (1213-1227, H3).

**Extension mechanism**: add a typed `Option<_>` (or `bool`) field + `Default` entry + classify it in
`first_nonfinite_float` + (if gated) a `Capabilities` bool + a floor check. Sub-blocks are used for
audio (`AudioParams`), memory (`GenerationMemory`), approximation, MOSS, HDR, Veda. There is **no
untyped extras map** and **no per-family typed extension slot** (e.g. `HashMap<TypeId, Box<dyn Any>>`).

### 5.2 Parameter groups (`gc/generator/params.rs`)

- **`GenerationMemory`** (`params.rs:10-117`, `Copy + Default`): rung booleans `stage_residency` 17,
  `tile_vae_decode` 19, `chunk_attention` 22, `stream_transformer_blocks` 26; rung params
  `decode_tile_edge` 42, `decode_overlap` 45, `attention_chunk_size` 48, `transformer_window_size` 51,
  `transformer_window_component` 62; execution domains `graph_eval_cadence` 86, `ffn_chunk` 90,
  `cfg_batching` 95, `prefix_kv_cache` 103; hidden calibration fault fields 110-116.
- **`AudioParams`** (138-202): voice, language, target_duration, sample_rate, bpm, musical_key, lyrics,
  script (`Vec<SpeechSegment>`), segments, max_new_tokens_per_segment, repetition_penalty,
  reference_region, output_limiter, `song: Option<SongParams>`, `artifacts`.
- **`SongParams`** (207-239), `SongCover`/`SongCoverMode`/`SongCoverVoice` (244-275) — YuE2-specific.
- `SpeechSegment` (281-296), `ConversationRole`/`ConversationTurn` (302-350).
- **`GenerationPhase { steps: u32, guidance: Option<f32>, adapters: Vec<PhaseAdapter> }`** (361-380);
  **`PhaseAdapter { adapter: usize, weight: Option<f32> }`** (386-397) — `adapter` indexes into
  `LoadSpec.adapters` (cross-reference between load-time and request-time objects). No per-phase
  scheduler/resolution/sampler (`request.rs:328-329`: "Per-phase *scheduler* selection is a deliberate
  follow-on").

### 5.3 `Conditioning` (`gc/generator/conditioning.rs:18-299`)

17 variants: `Reference{image,strength}` 21, `ReferenceRgba` 42, `ReferenceAudio` 51,
`ReferenceVideo{frames,fps,audio}` 107, `AudioEdit{audio,mode,region,strength}` 135,
`AudioEditRegions` 183, `VoiceEmbedding` 198, `MultiReference{images}` 203, `ReduxRefs{refs}` 205,
`Control{image,kind,scale}` 211, `Depth` 217, `Mask` 219, `Keyframe{image,frame_idx,strength}` 225,
`VideoClip{frames,frame_idx,strength}` 235, `ControlClip{frames,mask,masking_strength,start_frame,mode}`
245, `VideoSync{frames}` 280, `ConversationHistory{turns}` 298. Mirror `ConditioningKind` (349-394);
`ControlKind::{Pose,Canny,Depth,Other(String)}` (340-346); `ReplacementMode` (328-337). Borrowed views
in `gc/generator/views.rs` (`KeyframeRef`, `VideoClipRef`, `AudioEditRef`, `ControlClipRef`,
`Scail2AnimationConditioningRef` 104).

### 5.4 `LoadSpec` (`gc/runtime/load_spec.rs:9-172`, 24 fields)

| Field (line) | Type | Generic? |
|---|---|---|
| `weights` 10 | `WeightsSource::{Dir(PathBuf), File(PathBuf)}` (`gc/runtime/source.rs:45-50`; no hub-fetch by design 37-43) | generic |
| `resolved_route` 15 | `Option<String>` | generic (catalog route served by shared provider) |
| `decode_geometry_policies` 20, `decode_geometry_policy_authoritative` 24, `decode_quality_runtime_identity` 27 | memory-quality evidence | generic |
| `prepared_file_pins` 42, `prepared_receipt_paths` 47, `prepared_encoder_receipt` 51, `source_inventory` 53 | `pub(super)` identity/sealing | generic |
| `quantize` 54, `precision` 55 | `Option<Quant>`, `Precision::{Bf16,Fp32}` | generic |
| `yue2_ar_mode` 57, `yue2_compute_policy` 59 | | **YuE2-only** |
| `control` 65, `extra_controls` 72, `ip_adapter` 78 | typed overlays | ControlNet / MultiControlNet (SDXL) / IP-Adapter |
| `adapters` 83 | `Vec<AdapterSpec>` | generic (load-time LoRA/LoKr; changing set ⇒ reload, 79-83) |
| `pid` 92 | `Option<PidWeights{checkpoint,gemma}>` | PiD |
| `identity` 100 | `Option<IdentityWeights{encoder,eva,face_dir}>` | PuLID/InstantID |
| `text_encoder` 107 | `Option<WeightsSource>` | external TE (LTX) |
| `offload_policy` 114 | `OffloadPolicy::{Resident,Sequential}` | generic, advisory |
| `load_shape` 118, `load_shape_declaration_result` 124 | `LoadShape::{Eager,Deferred}Materialization` | generic |
| `components` 171 | `BTreeMap<String, WeightsSource>` | **the generic extension map** (ids documented as a table in the doc comment 146-157) |

So LoadSpec already has the "typed slots + generic keyed map" hybrid (`components` chosen deliberately
over typed slots, 138-142) that GenerationRequest lacks. Builders `with_quant`, `with_yue2_*`,
`with_text_encoder`, `with_offload_policy`, `with_load_shape`, `with_control`, `with_component`,
`with_extra_control`, `with_ip_adapter`, `with_adapters`, `with_pid` (`load_spec.rs:1216-1354`), plus a
large file-pinning/sealing API (420-1175).

`AdapterSpec { path, scale, kind: AdapterKind::{Lora,Lokr}, pass_scales: Option<Vec<f32>>, moe_expert:
Option<MoeExpert::{High,Low}> }` (`gc/runtime/adapters.rs:7-88`); `AdapterApplyReport` (34-45).
String component-id constants: `BASE_SNAPSHOT_COMPONENT`, `VAE_COMPONENT`,
`LTX_SPATIAL_UPSCALER_COMPONENT`, `COMFYUI_TEXT_ENCODER_COMPONENT`, `COMFYUI_VAE_COMPONENT`,
`KREA_CONVROT_DIT_COMPONENT` (`gc/runtime/source.rs:12-35`).

---

## 6. Output, events, sinks, telemetry

### 6.1 Outputs (`gc/generator/output.rs`)

- `GenerationOutput::{Images(Vec<Image>), ImagesRgba(Vec<RgbaImage>), Video{frames,fps,audio},
  Audio(AudioTrack)}` (9-34).
- `GenerationReport { output: Option<GenerationOutput>, artifacts: Option<ArtifactRecord>, warnings:
  Vec<GenerationWarning{code,message}>, prefix_kv_cache: Option<PrefixKvCacheReport> }` (128-156).
- `HdrRequest { color_space, sink: HdrFrameSink }` (101-119); `OutputChannels::{Rgb,Rgba}` (84-93);
  `CaptionOverflowPolicy` (39-68); song/audio artifact types (179-249).

### 6.2 Progress & callbacks (`gc/runtime/sinks.rs`)

- `Progress` (`sinks.rs:316-355`, `Copy + Eq`): `PromptUpsamplingComplete`, `PromptEncodingComplete`,
  `ConditioningComplete`, `Step{current,total}`, `Decoding`, `Loading(LoadPhase)`,
  `Upsampling{tokens,max_tokens}`, `PostProcessing`. `LoadPhase::{EncodingPrompt,TextEncoder,Renderer}`
  (360-369). Helper `with_prompt_encoding` (373-381). No "phase N of M", no resolution/stage events.
- `CancelFlag(Arc<AtomicBool>)` (8-50) — a request field; engines poll it.
- `PreviewSink(Option<Arc<dyn Fn(PreviewFrame)+Send+Sync>>)` (94-115), `PreviewFrame{current,total,image}`
  (72-79). Request field, not a Progress variant (`request.rs:342-347`).
- `HdrFrameSink` (165-186), `HdrOutputFrame` (141-150).
- `PromptEnhancementSink` (290-306), `PromptEnhancementReport`/`Outcome` (203-285).
- Streaming audio: `on_chunk: &mut dyn FnMut(AudioChunk)` via `generate_streaming`.

Backends each implement a preview helper: Candle `PreviewHook`/`PreviewCounter` (`cdl/preview.rs:88,226`)
passed as `Option<&PreviewHook>` into the sampler driver; MLX `PreviewCounter` (`mlx/preview.rs:19`) and
an `on_latent` closure on `*_with_latent_hook` drivers.

### 6.3 Load I/O progress (`crates/contracts/load-progress/src/lib.rs`)

`LoadActivityPhase::{Checking,Verifying,Loading}` (11-15); `LoadActivity{phase, component, file,
bytes_read, total_bytes, component_bytes_read, component_total_bytes, files_checked, total_files}`
(18-30); `LoadActivityContext` (75+: `capture`, `install`, `with_components`, `register_files`,
`report`, `file`, `file_with_bytes`); `LoadActivityObserver::new(Fn(Option<LoadActivity>))` (272-289).
Captured contexts travel with lazy reads, never process-global (header 1-2). Registry load installs it
(`gc/registry/providers.rs:52`).

### 6.4 Telemetry

`gen-core-telemetry` is **not** general telemetry: an H3-only `Collector` sampling process RSS on a
40 ms thread and recording per-step JSON (`crates/contracts/gen-core-telemetry/src/lib.rs:1-120`),
enabled by the request flag `h3_terminal_telemetry` (`request.rs:123`). No generic per-step timing/
memory event stream exists in the contract.

### 6.5 Errors (`gc/error.rs:10-52`)

`Error::{Backend(Box<dyn Error>), MissingTensor(String), Io, Unsupported(String), Canceled,
GeometryRefused{reason, requested_width, requested_height, alternative}, Msg(String)}`. `Canceled` and
`Unsupported` are contract-load-bearing (header 1-5).

---

## 7. Memory-strategy contract

### 7.1 Ladder and selection vocabulary

- `MemoryStrategy::{Resident=0, StagedResidency=1, BoundedDecode=2, BoundedAttention=3,
  BoundedTransformerResidency=4}` (`gc/memory_strategy/strategy.rs:6-14`); `ALL` (17-23);
  `is_optimized` (26); `requires()` (66-74 — rung 4 requires `LoadShape::DeferredMaterialization`,
  157-160); `engages(rung)` (116-127): StagedResidency only by explicit selection; 2-4 cumulative.
- Cost order is ordinal and backend-neutral; selector takes **first rung whose measured peak fits**
  (`gc/memory_strategy.rs:36-61`). Caller owns live budget and least-cost selection (`memory_strategy.rs:1-11`).
- `MemoryStrategySupport::{Implemented, StructurallyNotApplicable{reason}, Missing}` (163-171);
  `MemoryStrategyCapability{strategy, support, parameters: MemoryParameterRanges}` (174-181);
  `MemoryParameterRanges{decode_tile_edges, decode_overlaps, decode_geometry_policies,
  attention_chunk_sizes, transformer_window_sizes, transformer_window_components}` (230-243);
  `MemoryStrategyParameters` (268-287, duplicates `GenerationMemory`'s rung params with `Option<bool>`
  `stage_residency`); `TransformerComponent::{Dit,TextEncoder,Both}` (203-227);
  `MemoryPhase::{Conditioning,Denoise,Decode}` (290-295); `MemoryLifecycleCapabilities{phases,
  synchronized_phase_release, decode_tiling, attention_chunking, transformer_window_materialization}`
  (298-306).
- Selection types (`gc/memory_strategy/selection.rs`): `MemoryNumericTier{precision, quant,
  component_precision_floors}` 7-12; `MemorySelection{strategy, parameters, tier}` 17-20;
  `MemoryGeometry{width,height,batch,frames,reference_count}` 25-32; `MemoryBudget{total_bytes,
  committed_bytes, reclaimable_bytes, reserved_headroom_bytes}` 69-142 (`fits`, `required_total_bytes`);
  `MemoryCacheState::{Cold,Warm}` 147; `MemoryOptimizationAuthority::{Resident,Calibrated,Estimated}`
  159-167; `MemoryMode::{TextToImage,ImageToImage,Edit,Other(String)}` 171-188 (stringly escape hatch);
  `MemoryRejection` 194-199.

### 7.2 `MemoryProviderContract` (`gc/memory_strategy/contract.rs:71-108`)

Fields: `provider_id: String`, `backend: MemoryBackendRealization`, `strategies:
Vec<MemoryStrategyCapability>`, `decode_geometry_policy_authoritative`, `pid_decode_routes`,
`load_shape`, `additional_prerequisites`, `default_engagement_exclusions`, `resident_request_memory`,
`lifecycle`, `formula: MemoryFormulaKind`, `calibration: Option<MemoryCalibrationIdentity>`,
`asset_facts`, `architecture_facts`, `phase_facts: Option<MemoryPhaseFacts>`, `runtime:
MemoryRuntimeSemantics{cache, warm_run, cancellation, error}` (29-35).
Methods: `compatibility_default` (114), `predicted_peak_from_base` (155), `decompose_predicted_peak`
(167), resident-byte accessors (198-245), `capability` (255), decode-geometry policy adoption/lookup
(267-360), `requires`/`engages`/`engages_selection`/`engaged_composition*` (444-528),
`generation_memory(selection)` (539 — converts selection → `GenerationMemory`),
`representative_selection` (580), `selection_space` (642), `window_materialization` (751),
`conformance_errors` (757), `validate_selection` (1079).

`MemoryPhaseFacts{architecture: Option<ImagePipelineArchitecture>, staged_weights, transformer_stream,
decoder_workspace, native_quant (H3)}` (`gc/memory_phases.rs:86-95`); `ImagePipelineArchitecture::
{SdxlUnetWithDualClip, ZImageDit, SanaLinearDit{cfg}}` (74-83) — a closed per-family enum in the contract.

### 7.3 Run context

`MemoryRunContext` (`gc/memory_strategy/run.rs:18-50`): `selection`, `optimization_authority`,
`calibration_abi`, `calibration_fingerprint`, `load_shape`, `mode`, `has_reference`, `use_pid`,
**`has_phases`**, `geometry`, `reference_resources`, `overlay: Option<String>`, `budget`,
`predicted_peak_bytes`, `cache_state`, `evidence_revision`. `standard_memory_behavior_context` (74-115)
builds a weights-free context (fixed 1024², 8 GiB budget).

### 7.4 Safety

`MemorySafetyDecision::{Accept, Reject{reason}}` (`gc/memory_strategy/safety.rs:352-375`);
`standard_memory_strategy_safety_check` (10), `default_memory_strategy_safety_check` (114),
`default_registered_memory_strategy_safety_check` (125), `validate_bounded_decode_params` (380),
`admit_memory_safety` (409). Provider safety can reject but **cannot substitute** a strategy or tier
(`memory_strategy.rs:8-11`).

### 7.5 `MemoryRequestScope` — the existing per-request hook trait (`gc/memory_strategy/run.rs:191-217`)

```rust
pub trait MemoryRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()>;
    fn configure_admitted_request(&mut self, _request: &GenerationRequest) -> Result<()> { Err(Unsupported) }
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()>;
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()>;
    fn configure_decode(&mut self, tile_edge: u32, overlap: u32, geometry: MemoryGeometry) -> Result<()>;
    fn configure_attention(&mut self, chunk_size: u32) -> Result<()>;
    fn materialize_transformer_window(&mut self, first_block: u32, block_count: u32) -> Result<()>;
    fn finish(&mut self, outcome: MemoryRunOutcome) -> Result<()>;
}
```
Opened by `Generator::begin_memory_strategy_request` after the safety check. Backend cores implement it:
`MlxRequestScopeCore` (`mlx/request_scope.rs:263-534`, 1,343 lines; config `MlxRequestScopeConfig` 58)
and `CandleRequestScopeCore` (`cdl/request_scope.rs:73-260`, 475 lines). The hooks are **memory-specific
callbacks invoked by the engine** — the engine still owns the loop and decides when to call them.
`configure_request(&mut GenerationRequest)` lets the scope rewrite the request (e.g. inject
`GenerationMemory`).

### 7.6 `ProviderMemoryProfile` — table-driven provider memory (`gc/memory_strategy/profile.rs`)

Introduced to replace hand-copied skeletons (header 1-34). Hook fn-types:
`MemoryProfileGate = fn(&Contract, &MemoryRunContext, MemoryNumericTier) -> Result<()>` (46-47),
`MemoryProfileTierResolver = fn(&LoadSpec, &Contract) -> Result<MemoryNumericTier>` (55-56),
`MemoryProfileScope = fn(&'static str, &LoadSpec, &Contract, &MemoryRunContext, MemoryNumericTier)
-> Result<Option<Box<dyn MemoryRequestScope>>>` (62-68). `MemoryProfileGates{before, within, after}`
(112-150); `ProviderMemoryProfile{tier, gates, rejection, fixtures, fingerprint, architecture}`
(407-428, `STANDARD` const). Standard skeleton functions: `standard_safety_check` (467),
`standard_validate_context` (489), `standard_registered_safety_check` (503), `standard_begin_request`
(519-530), `standard_registered_valid_fixture` (535), `standard_architecture_facts` (596). Referenced in
49 media/audio crates. `check-workspace.py` ratchets providers still on bespoke skeletons
(profile.rs:30-32). **This is the precedent for "data + hook fn-pointers + one shared skeleton"** that a
generic pipeline layer could follow.

---

## 8. Residency, tiling, attention budget, block window (planner vs kernel split)

The rung modules explicitly adopt a **planner (gen-core, pure) / kernel (per backend)** split
(`gc/attention_budget.rs:3-15`, `gc/block_window.rs:11-18`).

### 8.1 `Residency` (`gc/residency.rs`, 1,846 lines)

"Backend-neutral component residency and phase-order driver … Backend crates provide only
`ResidencyRuntime` hooks and their tensor phase bodies" (1-7).
- `ResidencyRuntime { type Error: From<Error>; fn after_component_drop(); }` (18-22).
- `StagedHeavy { type Light; type DecodeView; fn shed_dit(self)->Light; fn decode_view; fn light_view }` (45-55).
- `Residency<Text, Heavy, R>` constructors: `resident`, `sequential`, `request_scoped`,
  `request_scoped_with_resident(_cancelable)`, `from_policy`, `request_scoped_from_policy`,
  `from_policy_with_resident` (69-262); `with_staged_advisory`, `evict_warm` (264-286).
- Run drivers: `run(encode, materialize_before_text_drop, render)` (398-420), `run_request_scoped`
  (421), `run_request_scoped_with_prepare(prepare, encode, materialize, render)` (450-460),
  `run_exclusive_staged` (528), `run_staged(encode, materialize_encoded, denoise, materialize_mid,
  decode)` (545-560), `run_staged_request_scoped` (575), `run_staged_request_scoped_with_prepare`
  (623-640), free fn `run_two_phase` (754).
- Adopted by ~29 crates (mlx: anima, boogu, chroma, flux, flux2, flux3, ideogram, iris, kolors, krea,
  lens, mage, minimax-h3, qwen-image, qwen-image-2-1, sana, sd3, sdxl, z-image; candle: anima, flux,
  flux2, flux3, iris, krea, qwen-image, qwen-image-2-1, z-image; plus both cores).
- **This is the de facto 3-stage pipeline (prepare/encode → denoise → decode)**, but the stage bodies
  are opaque closures and the `render`/`denoise` closure contains the whole sampler loop.

### 8.2 Tiling (`gc/tiling.rs`, 1,713 lines)

Pure geometry: `VaeTiling` presets (74-183), `TilingConfig` presets/auto (269-395),
`TilePlan` (584), `trapezoidal_mask` (438), temporal overlap policy (719-789), `budgeted_plan` (894),
`VideoDecodeMemoryProfile` (198-250). Backends own the blend loop (`mlx/vae_tiling.rs`,
`cdl/vae_tiling.rs`, `cdl/bounded_decode.rs`).

### 8.3 Attention budget (`gc/attention_budget.rs`, 609 lines)

`AttentionBudget` (80) with `UNBOUNDED`/`CONSTRAINED` (120-127), `query_block_rows` (166), `head_chunks`
(201), `query_block` (247); `AttentionPlan<'a>` (284-317, carries cancel); cross-backend chunk boundary
cases (336-360). Kernels: `mlx::attention::sdpa_budgeted_bhsd`, candle `sdpa_budgeted_*` (header table 5-9).

### 8.4 Block window (`gc/block_window.rs`, 579 lines)

`BlockPlan::{new, resident, windows(), window_count}` (91-144); `BlockWindowBackend { type View; fn
open_view(&mut self) -> Result<View>; fn release(&self); }` (152-176); `run_windowed(backend, plan,
cancel, init, apply, materialize) -> Result<S>` (194-230+) — owns loop order, cancellation and release
on every exit path. Backend impls: `mlx/block_residency.rs:63,109`, `cdl/block_window.rs:126,161`.
Used by ~28 crates.

---

## 9. Execution domains and approximation — existing precedents for cross-cutting levers

### 9.1 Execution domains (`gc/execution_domains.rs`, 1,082 lines)

Equivalence-preserving per-request schedule knobs, declared on `Capabilities::execution:
ExecutionSurface`, selected in `GenerationMemory`, refused at the floor if undeclared
(header 1-40): `GraphEvalCadence` (111-148), `FfnChunk` (164-190), `CfgBatching::{Batched,Sequential}`
(223-249), `PrefixKvCacheMode::{Cached,Prefill}` (306-347, with `select(budget, …)`),
`PrefixKvCacheReport` (364), `ExecutionValueDomain` (446-462). Deliberately *not* ladder rungs.

### 9.2 Approximation (`gc/approximation.rs`, 1,529 lines)

**Result-changing** levers (header 1-60): `FeatureCachePolicy` (192, `recomputes_step(index)`,
`is_inert_over(steps)`), `CacheReuseInterval` (118, rejects 0 and 1), `CacheWarmupSteps` (169),
`TokenPruningPolicy` (402, `prunes_step`), `TokenDropStride` (333). Declared by
`ApproximationSurface{denoise_feature_cache, token_pruning, characterization}` (617-624); requested by
`ApproximationRequest{denoise_feature_cache, token_pruning, characterization: Option<CharacterizationRef>}`
(828-839); resolved to `ApproximationPlan::{Exact, Approximate{...}}` (874-886) by
`Capabilities::approximation_plan`. `CharacterizationFamily` is an **uninhabited enum** (504), so every
approximate selection is structurally refused until a measurement campaign defines one (504-516).
Rules: "off is absence" (22-47), provider branches on `is_exact()` exactly once at the top of denoise
(49-55). **A low-resolution-first-K-steps feature is result-changing and would fall under this regime**
as currently designed.

---

## 10. Sampling framework

### 10.1 Two layers in gen-core (`gc/sampling.rs` + `gc/sampling/*`)

- **Legacy policy layer**: `StepCoeffs{timestep, c_in, a_x, a_out, a_noise}` (61-72), `StepDtype` (89-95),
  `trait SamplerPolicy { num_steps; init_noise_scale; step_dtype; coeffs(i); sigma(i) }` (100-113),
  `AlphaSchedule` (123), `FlowMatchPolicy` (425), `LightningPolicy` (488), `LcmPolicy` (567),
  `TcdPolicy` (670). Header (25-31): engines "stay on `SamplerPolicy` until P3/P4 migrate them per-engine".
- **Unified framework** (epic 7114):
  - `trait LatentOps { type Latent: Clone; observe_solver(&self, SolverObservation) {}; scale; add; sub;
    axpy; randn_like(x, seed, step) }` (`sampling/latent_ops.rs:26-58`), `CpuLatentOps` (68).
  - `trait ModelSampling { prediction; sigma_min; sigma_max; input_scale(σ); denoised_coeffs(σ);
    noise_scaling_coeffs(σ); timestep(σ); sigma(t); num_timesteps; sigma_table }`
    (`sampling/model_sampling.rs:41-95`); impls `FlowModelSampling::{new(conv), with_shift(conv, mu)}`
    (176-241), `DiscreteModelSampling` (264-333, `sdxl()`), `EdmModelSampling` (346-389, `svd()`);
    free fns `denoise(ops, ms, x, σ, predict)` (113), `cfgpp_denoise` (142).
  - `type DenoiseFn<'a,L> = dyn FnMut(&L::Latent, f32) -> Result<L::Latent>` (`sampling/unified.rs:19-20`);
    `trait Sampler<L> { fn sample(&self, ops, ms, denoise, x, sigmas: &[f32], seed) -> Result<Latent> }`
    (29-50); `Euler` (77-130); `apply_coeffs` (142-156).
  - Solvers: `EulerAncestral` 53, `Heun` 126, `Dpmpp2m` 207, `DpmppSde` 305, `UniPc` 429, `Lcm` 517,
    `Ddim` 571, `ErSde` 644, `Dpmpp2mSde` 814 (`sampling/solvers.rs`); `enum Solver` (945, `from_name`,
    `is_stochastic`, `ALL` 10 entries, `boxed`); `SolverObservation*` (1050-1194);
    `sampler_by_name(name)` (1279).
  - Schedulers: `enum Scheduler` (`sampling/schedulers.rs:22`, 8 entries), `schedule_sigmas(sched, ms,
    steps)` (79), `karras/exponential/normal/simple/ddim_uniform/beta_sigmas` (128-246).
  - CFG++: `CfgPpDenoiseFn`, `trait CfgPpSampler`, `EulerCfgPp`, `Dpmpp2mCfgPp`, `cfgpp_sampler_for`
    (`sampling/cfgpp.rs:31-234`).
- **The loop lives inside `Sampler::sample`.** It iterates the full `sigmas` slice; there is no
  per-step hook beyond the `denoise` closure (called per model evaluation, possibly twice per step for
  Heun/DPM++SDE) and `observe_solver`. A between-step latent transform (e.g. upsample) cannot be
  injected without either slicing `sigmas` and calling `sample` repeatedly (what Krea multiphase does)
  or abusing the denoise closure.

### 10.2 Shift / mu

- gen-core: `image_seq_len(w,h) = (h/16)·(w/16)` (`sampling.rs:224-226`, FLUX 2×2-packing assumption
  baked in), `compute_mu(seq_len, steps)` piecewise fit (233-245), `time_shift_exponential` (252),
  `build_flow_sigmas(steps, mu)` (259-275), `flow_capture_plan` (310, PiD early stop),
  `vp_capture_plan` (383).
- Family-specific mu/schedule functions elsewhere: Krea `mu_for_seq_len` + constants
  (`fam/krea_schedule.rs`), Qwen-Image 2.1 `image_tokens`/`mu_for_tokens`/`sigmas`/`sigmas_for_image`/
  `resample_saved_sigmas` (`fam/qwen_image_2_1/scheduler.rs`), Ideogram `LogitNormalSchedule`/
  `preset_mu_std` (`fam/ideogram_scheduler.rs`), H3 schedule (`gen-core-h3/src/h3_schedule.rs`), MOSS
  schedule (`gen-core-moss-image`), Iris FlowDPM step plan (`gen-core-iris`). No shared
  "schedule-for-(model, resolution, steps)" contract; mu is resolution-dependent, which matters for any
  feature that changes resolution mid-trajectory.

### 10.3 Backend drivers (signatures)

MLX (`mlx/sampler.rs`):
```rust
pub struct SamplerRun<'a> { sampler_name: Option<&'a str>, sigmas: &'a [f32], seed: u64,
                            cancel: &'a CancelFlag, on_progress: &'a mut dyn FnMut(Progress) }   // 422-431
pub fn run_curated_sampler(run: SamplerRun, ms: &dyn ModelSampling, latents: Array,
                           predict: impl FnMut(&Array, f32) -> Result<Array>) -> Result<Array>   // 457-464
pub fn run_curated_sampler_with_latent_hook(run, ms, latents, on_latent: impl FnMut(&Array, f32), predict) // 518-524
pub fn run_cfgpp_sampler(run, ms, latents, predict_pair: impl FnMut(&Array,f32)->Result<(Array,Array)>) // 626
pub fn run_cfgpp_sampler_with_latent_hook(...)                                                    // 638
pub fn run_flow_sampler(run, conv: TimestepConvention, latents, predict)                          // 750-757
pub fn run_flow_sampler_with_latent_hook(run, conv, latents, on_latent, predict)                  // 763-772
pub fn run_av_curated_sampler(..., predict: impl FnMut(&AvLatents, f32) -> Result<AvLatents>)     // 842
pub fn resolve_flow_schedule(scheduler_name, mu, steps, native: &[f32]) -> Vec<f32>              // 904-912
pub fn resolve_schedule(scheduler_name, ms, steps, native) -> Vec<f32>                            // 923-934
pub fn curated_sampler_names() / curated_scheduler_names()                                       // 941, 952
```
Plus legacy `trait DiffusionSampler` (45), `StepRng` (77), `MlxLatentOps` (166), wrapper samplers
`LcmSampler`/`LightningSampler`/`TcdSampler`/`FlowMatchSampler` (965-1209), and `CuratedStepObserver`
which hard-codes which solvers evaluate twice (472-512). `mlx/flow_sampling.rs` holds a host-only
`FlowSamplingDefaults`/`FlowSamplingRecipe` replay (9-60) that is tensor-free but lives in the MLX crate.

Candle (`cdl/sampler.rs`):
```rust
pub fn run_curated_sampler(sampler_name: Option<&str>, ms: &dyn ModelSampling, sigmas: &[f32],
    latents: Tensor, seed: u64, cancel: &CancelFlag, on_progress: &mut dyn FnMut(Progress),
    preview: Option<&PreviewHook<'_>>, predict: impl FnMut(&Tensor, f32) -> Result<Tensor>) -> Result<Tensor> // 188-232
pub fn run_flow_sampler(sampler_name, conv, sigmas, latents, seed, cancel, on_progress, preview, predict)  // 247-275
pub fn run_av_curated_sampler(...)        // 434
pub fn run_scm_sampler / run_scm_sampler_from(...)   // 624, 651 (Sana sCM)
pub fn resolve_schedule / resolve_flow_schedule      // 282-293, 314-322 — byte-identical to MLX
```
Candle's driver emits `Progress::Step` from inside the denoise closure by counting schedule nodes
above σ (`cdl/sampler.rs:211-218`), checks cancel per eval (212-214), and the driver owns the
`PreviewCounter` (209). The two backends express the same concerns (cancel, progress, preview/latent
observation, solver lookup, N3 fallback to Euler for unknown names) with **different call shapes**.
48 media crates call these drivers.

### 10.4 img2img start step

- MLX shared: `DEFAULT_IMG2IMG_STRENGTH = 0.5`, `resolve_strength`, `resolve_reference`,
  `init_time_step(num_steps, strength) = max(1, floor(n·s))` or 0, `preprocess_init_image`
  (`mlx/img2img.rs:13-87`), plus a noise-interpolation blend.
- Candle: re-implemented per family — `candle-gen-boogu/src/pipeline.rs:109,1176`,
  `candle-gen-ideogram/src/pipeline.rs:1154` (default 0.6 in `config.rs:50`),
  `candle-gen-krea/src/pipeline.rs:2833`, `candle-gen-sana/src/pipeline.rs:73-82`,
  `candle-gen-sd3/src/pipeline.rs:112,166`, `candle-gen-z-image/src/common.rs:85`,
  `candle-gen-kolors/src/pipeline.rs:974` (default 0.3). Not in gen-core at all; the `SamplerPolicy::
  sigma(i)` doc mentions "flow-match img2img start-step blending" (`sampling.rs:110-112`).

### 10.5 CFG variants

- Scalar CFG / true-CFG: request `guidance`, `true_cfg`, `timestep_to_start_cfg`, `cfg_interval`,
  `image_guidance`; capability flags §4.2. Combination done inside each engine's `predict` closure
  ("RAW (already CFG-combined) output", `mlx/sampler.rs:449-450`).
- Guidance axis (`gc/guidance.rs`): `GuidanceMethod::{Cfg,CfgRescale,Apg,CfgPp}` (40-76),
  `trait GuidanceOps: LatentOps` (88, axis-parameterized norms/dots), `cfg` (128), `cfg_rescale` (145),
  `MomentumBuffer` (170), `normalized_guidance(_chain)` (231-251), `apg_delta` (299). CFG++ realized at the
  sampler layer (`sampling/cfgpp.rs`).
- CFG batching is an execution domain (`CfgBatching`, §9.1).

### 10.6 Phases (multi-phase denoise)

- Contract: `GenerationRequest.phases: Option<Vec<GenerationPhase>>` (`request.rs:316-330`),
  `GenerationPhase{steps, guidance, adapters: Vec<PhaseAdapter>}` (`params.rs:361-397`); memory context
  carries `has_phases` (`run.rs:33`).
- Host primitive: `fam/krea_multiphase.rs` — `PhaseSlice` (55-71), `resolve_phase_slices(phase_steps,
  model)` (86), `total_phase_steps` (110), `phase_uses_cfg` (117), `resolve_phase_adapters` (142),
  `ResolvedPhase` (169), `resolve_phases(phases, default_guidance, loaded_adapter_count, model)`
  (190-215), `phase_spec_subset` (218). Doc (1-45): ONE global schedule for the total step budget,
  contiguous inclusive slices sharing boundary sigmas; drivers "drive `run_flow_sampler` over each
  slice from the running latent"; MLX clones the DiT per phase, Candle re-adapts one job-local DiT.
- Consumers: `candle-gen-krea/src/lib.rs:475,886,943,1334-1385` and `mlx-gen-krea/src/multiphase.rs`;
  several other families only *refuse/route* on `request.phases.is_some()` in their memory strategy
  (anima, bernini, boogu). No capability flag; non-Krea engines ignore the field.

---

## 11. Other contract crates (detail beyond the table in §1)

- **`gen-core-family`** is the grab-bag that holds generic-looking primitives under family names: the
  only multi-phase slicer (`krea_multiphase`), resolution resolvers (`lens_resolution::resolve_resolution
  (base, aspect)`), and three bespoke flow schedulers (Krea, Qwen 2.1, Ideogram).
- **`gen-core-h3`, `gen-core-moss-image`, `gen-core-flux3`, `gen-core-iris`, `gen-core-ltx`,
  `gen-core-memory-wan`**: per-family "host halves" so MLX and Candle twins share one definition
  (e.g. gen-core-flux3 header: "the MLX provider, a Candle twin and the SceneWorks worker read one
  definition"). These are the pattern "tensor-free family policy shared across backends", but each is
  bespoke and the interface between it and the backend driver is ad hoc.
- **`gen-core-testkit`** enforces behavioural laws per `Generator` (cancellation typed, progress
  monotone, seed determinism, capability honesty, cfg-off render, component load gate,
  `generate ≡ generate_streaming`, session ≡ batch).
- **`core-llm`** is an independent tensor-free LLM contract re-exported through gen-core (inverted dep).

---

## 12. Weights / checkpoint layer

### 12.1 Declarative logical weight plan (`gc/checkpoint_codec*`)

Two halves (`gc/checkpoint_codec.rs:1-25`):
1. **Codec registry** — `CheckpointCodecRegistration` (one per stored tensor format; registered once at
   engine/catalog level, e.g. `cdl/logical_weights.rs` via `register_checkpoint_codecs`, and the GGUF
   container row in the catalog `candle-gen-catalog/src/lib.rs:94-99`). Exported codec constants:
   `DENSE_BF16/F16/F32_CODEC`, `FP8_E4M3/E5M2_SCALAR_CODEC`, `INT8_PER_ROW_CODEC`, `MXFP8_CODEC`,
   `NVFP4_CODEC`, `GGUF_CONTAINER_CODEC`, `COMFY_QUANT_CODECS` (`gc/lib.rs:119-129`).
2. **`LogicalWeightPlan`** compiled from the safetensors header + `.comfy_quant` descriptor bytes + the
   family's `LogicalKeyMapping` + codec registry + backend `CodecResidencyPolicy`
   (`compile_logical_weight_plan[_with_metadata]`, `checkpoint_codec/compile.rs:25,66`). Fails closed on
   unmapped/duplicate keys, unknown format, malformed descriptor, missing scale companion, bad MXFP8
   geometry.

What a plan can express:
- `LogicalTensorPlan{logical_key, physical_key, encoding, shape, source_bytes, codec_id,
  resident_encoding, codec: TensorCodecSpec, residency: PlannedResidency, transform:
  Option<LogicalTensorTransform>}` (`checkpoint_codec/plan.rs:8-33`).
- `TensorCodecSpec::{Dense, ScalarFp8{scale,input_scale,fpmm}, Mxfp8{scale,stored_shape,...},
  Int8PerRow{scale,...}, Nvfp4{block_scale,global_scale,input_scale,stored_shape,logical_shape,...}}`
  (`checkpoint_codec/spec.rs:18-64`) — per-tensor (mixed checkpoints dispatch per layer).
- `PlannedResidency{mode: ResidencyMode::{Dense,Packed}, resident_bytes}` (spec.rs:92-104), chosen by
  `trait CodecResidencyPolicy::residency(codec, spec, stored_shape)` (110-117), default
  `DenseResidencyPolicy` (121-133).
- `CompanionTensorPlan{physical_key, role, owner_physical_key, source_bytes, resident_bytes}` (plan.rs:184-192).
- `LogicalWeightPlan{mapping_id, tensors, companions, source_bytes}` (197-203) + `resident_bytes()` (253),
  `resident_tensor_headers` (288).
- `trait LogicalKeyMapping { mapping_id; logical_key(physical) -> Option<String>; logical_shape(logical)
  -> Option<Vec<usize>>; logical_transform(physical) -> Option<LogicalTransformDeclaration> }`
  (`checkpoint_codec/transform.rs:11-43`). Transforms are pure row relabels: `rename`, `row_slice`
  (fused QKV split), `half_swap` (AdaLN shift/scale order) (80-160). `IdentityKeyMapping` (584-597).
- Receipt: `LogicalWeightReceipt` (resident bytes measured from decoded arrays, never copied from header).

What it does **not** express: module structure, block topology, per-block grouping for windowed
residency, quantization *targets* (affine Q4/Q8 at load), adapter injection points.

**Adoption is narrow**: plan/mapping consumers are found in `candle-gen` core, `candle-gen-flux2`,
`candle-gen-krea`, `candle-gen-ltx`, `candle-gen-wan`, `mlx-gen` core, `mlx-gen-krea` only
(`git grep compile_logical_weight_plan|LogicalKeyMapping`). Most families load via backend
`Weights` helpers (`mlx/weights.rs`, `cdl/weights.rs`, `cdl/loader.rs`) with hand-written key prefixes
(`ARCHITECTURE.md:34-39`: `from_weights(w, prefix)`).

### 12.2 `weightsmeta` (`gc/weightsmeta.rs`, 2,794 lines)

Tensor-free safetensors header/byte-view reader + adapter string/metadata parsing (header 1-17):
`Dtype` (43-123), `read_safetensors_header(_len)` (228-265), `safetensors_file_metadata` (285),
`CheckpointMeta::{from_file, from_bytes, from_dir, keys, contains, metadata, tensor}` (397-534),
`SafetensorsTensorHeader` (536-590), `safetensors_path_tensor_headers` (597),
`safetensors_file_tensor_locations(_observed)` (695-725), payload readers (923-960), byte accounting
(`safetensors_dir_bytes` 1004, `safetensors_path_bytes` 1051, `materialized_*_bytes` 1110-1119), LoKr/LoHa/
kohya suffix tables (1145-1176) and **H3-trainer-specific constants** (1181-1188).

### 12.3 Shards and inventory

- `gc/safetensors_shards.rs` (516 lines): `snapshot_shard_roots(dir)` (14) authorizes an HF snapshot's
  `blobs/`; `resolve_safetensors_shards` (54), `ResolvedShardPath` (74),
  `resolve_indexed_safetensors_shards` (83), `resolve_directory_safetensors_shards` (175).
- `gc/snapshot_inventory.rs` (479 lines): one walker policy replacing seven drifted Candle copies
  (header 1-25): `InventoryPolicy::{STRUCTURAL, REJECT_MARKERS}` (46-57), `collect_regular_files` (73),
  `sha256_file(_counted)` (187-193), `update_framed` (211), `hex_lower` (217).
- `gc/source_inventory.rs` (`SourceInventory`, bound on `LoadSpec` via `bind_source_inventory`,
  `load_spec.rs:822-837`).

---

## 13. Bundles (how engines are assembled and linked)

- Bundles are **composition + platform metadata**, not engines: runtime-cuda is 552 lines
  (`crates/bundles/runtime-cuda/src/lib.rs`), re-exporting `candle_gen_catalog::{media, vae_tiling,
  providers::*, nvfp4_quant_tiers, BESPOKE_MEMORY_ROUTE_WAIVERS}`, `candle_llm as llm`,
  `runtime_catalog::{core_llm, gen_core, memory_strategy, RuntimeCatalog, ...}` (3-53), and building
  `catalog()` from the media registry, `candle_llm::cuda_text_registry()`, preparers and the audio lane
  (96-140).
- runtime-macos (10,915 lines) additionally carries a perf benchmark binary and video VAE memory
  profile helpers (`lib.rs:29-113`).
- runtime-ios (1,820 lines) carries the admission guard and per-family facades (§3.3).
- **SceneWorks linking**: git deps pinned to rev `a43724b82` for `runtime-macos`, `runtime-cuda`,
  `sceneworks-gen-core*`, `sceneworks-gen-core-{checkpoint,iris,flux3,memory-wan,moss-image,h3,
  telemetry,train,testkit}`, and individual engine crates `mlx-gen`, `mlx-gen-{flux2,krea,
  krea-realtime,ltx,mage,minimax-h3,instantid,scail2,sdxl,wan,moss-image}`, `candle-gen`,
  `candle-gen-{scail2,wan,sdxl,ltx,moss-image}` (`SW: crates/sceneworks-memory-adapter/Cargo.toml:24-56`,
  `SW: crates/sceneworks-worker/Cargo.toml`, `SW: crates/sceneworks-core/Cargo.toml`), plus `[patch]`
  of `candle-core`/`candle-kernels`/`candle-metal-kernels` at the same rev (`SW: Cargo.toml:160-162`).
  Process-wide `OnceLock<RuntimeCatalog>` (`SW: inference_runtime.rs:27-34`).

---

## 14. Design docs on origin/main — stated principles and promises

### 14.1 `docs/architecture/inference-rearchitecture.md` (393 lines; "Accepted and implemented", 2026-07-13)

- Decision: one repo/workspace/lockfile; explicit composition (no `inventory`, no force-link, no
  global loaders) (8-22).
- Layering: contracts → MLX/Candle engines + shared runtime primitives → provider-family crates →
  platform catalog → application composition root (96-110).
- Non-goals explicitly include "merge MLX and Candle implementations behind a lowest-common-denominator
  engine" and "redesign model implementations … as a side effect" (78-88).
- Lateral provider deps allowed: same backend only, never upward, acyclic; extracting `*-common`
  crates "deliberately not done" (112-132).
- Invariants (315-362): contracts tensor-free; host transports and family code in sibling crates;
  gen-core keeps only what the contract names; every product consumes one named runtime bundle;
  duplicate ids fail; weights-free conformance; no self-fetching weights (`WeightsSource` has no hub
  variant, enforced by `scripts/check-workspace.py`).
- Revisit criteria: third-party providers without recompiling, independently versioned catalogs,
  CI scale (383-393).

### 14.2 `crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md` (399 lines; "Status: adopted")

- One `Generator` for all prompt→media, modality as property + output variant (§3.1, 79-126); separate
  `Transform` for non-prompt image→image (§3.3, 141-187; SeedVR2 still a Generator, 149-154).
- "Config unifies *within* a family, not across" — families don't share a forward (§3.2, 134-139).
- Goal 3: "The core (`mlx-gen`) stays fixed as models are added … `mlx-gen` stops getting bigger" (30-32).
- Adapters are core; LoRA+LoKr multiples/mixed; applied at load (§3.4, 189-207).
- Locked decisions (§9, 374-395): `GenerationRequest`/`TransformRequest` are single `Default`-able structs
  (no builders); `Conditioning` is a typed enum; registry via explicit builders.
- Known additive extensions (396-398): `Conditioning::VideoClip` (done) and "a typed video/audio guider
  block" (§8.2 362-366: LTX per-modality guiders; Wan A14B `guidance_scale_2`/`boundary_ratio`/dual-LoRA)
  — **not done**; video guidance is still scalar + per-family fields.
- Scope note: torch/diffusers models were out of scope; superseded (since ported) (340-349, 387-391).

### 14.3 `crates/media/mlx-gen/ARCHITECTURE.md` (107 lines)

- "Disciplined hybrid": numeric leaves are a faithful mirror of the Python reference; tensor keys are
  a compatibility contract; **orchestration (loaders, dispatch, pipelines, config) is "Clean Rust"**
  because Python-isms "would be copied across 5+ families + video" (11-25).
- Module conventions: plain structs, `forward(&self)`, `from_weights(w, prefix)`, dimension-parametric
  (29-39). `AdaptableLinear` everywhere with named exceptions (41-67).
- **"Dispatch & pipelines (deferred, non-speculative): Model-family dispatch (an enum) and a `Pipeline`
  trait are introduced when the second family lands, not before"** (79-81). Never done.

### 14.4 Others

- `crates/contracts/gen-core/README.md` (92 lines): restates ownership — provider contracts, request/
  output, load types, explicit registry, pure host policy, "request-surface halves of family contracts";
  sibling-crate table.
- `crates/media/candle-gen/README.md`: candle-gen "implement[s] the same backend-neutral gen_core
  contract"; core crate supplies device/dtype seam, weight loaders, seeded noise, unified sampler, LoRA
  harness, quant seam; 20+ families.
- Root `README.md` (68 lines): migration status; layout; points at getting-started, model-catalog
  reference, rearchitecture rationale.
- `docs/migration/SC_15449_SHARED_MEMORY_STRATEGY_CONTRACT.md`, `SC_15804_MEMORY_STRATEGY_RENAME.md`,
  `2026-07-26-generation-memory.md` etc. document the memory contract evolution (not read in detail).

---

## 15. Pain points visible in the contract layer itself

| # | Pain point | Evidence |
|---|---|---|
| P1 | **Family methods on the universal trait.** 6 of 20 `Generator` methods are MOSS/Veda-named; a family trait (`MossImageAdvancedGenerator`) and `ControlBranch` live in gen-core. | `gc/generator/traits.rs:13-59,121-126`; `gc/moss_image.rs:1118`; `gc/control.rs:77` |
| P2 | **Request union bloat.** 61 fields, ~half family-specific, extension only by new top-level `Option` fields; APG params at top level; SVD/SeedVR2/LTX/H3/FLUX3 fields. | `gc/generator/request.rs:9-381` |
| P3 | **Id-string checks in the shared floor and load gate.** flux3/moss/veda id gates; `max_enhance_tokens_for_model` id match; `ensure_quant_supported` YuE2 id gate. | `capabilities.rs:489-494,657-699`; `providers.rs:576-587` |
| P4 | **Family-specific LoadSpec fields** (`yue2_ar_mode`, `yue2_compute_policy`) despite a generic `components` map existing. | `load_spec.rs:56-59,171` |
| P5 | **Stringly-typed selectors**: `sampler`, `scheduler`, `guidance_method`, `video_mode` are `Option<String>`; capability menus are `Vec<&'static str>`; `descriptor.family/backend` strings; `MemoryMode::Other(String)`, `ControlKind::Other(String)`; overlay `Option<String>`. Typed enums exist (`Solver`, `Scheduler`, `GuidanceMethod`) but the boundary is strings with N3 fallback-to-Euler in drivers. | `request.rs:31-42,118`; `capabilities.rs:56-61`; `mlx/sampler.rs:444-447`; `cdl/sampler.rs:203-206` |
| P6 | **Duplicated vocabularies**: `GenerationMemory` vs `MemoryStrategyParameters` (same rung params, different optionality, converted by `generation_memory`); three phase enums (`Progress::Loading(LoadPhase)`, `MemoryPhase`, `LoadActivityPhase`) plus `GenerationPhase` meaning something else; `Conditioning` vs `ConditioningKind` mirror. | `params.rs:10-117`; `strategy.rs:268-295`; `sinks.rs:361`; `load-progress/src/lib.rs:11` |
| P7 | **No pipeline/stage contract.** The deferred `Pipeline` trait never landed; `generate` is opaque; the shared skeletons (`Residency::run_staged_*`, `run_windowed`, sampler drivers) are invoked from inside each engine's hand-written `generate`. | `ARCHITECTURE.md:79-81`; `residency.rs:623-640` |
| P8 | **Sampler driver drift between backends**: same responsibilities, different signatures (struct vs positional; `on_latent` closure vs `PreviewHook`); tensor-free schedule resolution duplicated byte-for-byte. | `mlx/sampler.rs:422-464,750-772,904-934`; `cdl/sampler.rs:188-322` |
| P9 | **Loop inside the solver** — no between-step hook; multi-segment trajectories require slicing sigmas and re-calling `sample`, losing multistep solver history (DPM++2M/UniPC) across the boundary. | `gc/sampling/unified.rs:29-50`; `fam/krea_multiphase.rs:31-33` |
| P10 | **Shift/mu and seq-len are per family**, gen-core's `image_seq_len` hard-codes /16; four bespoke mu/schedule functions in `gen-core-family`. | `sampling.rs:224-245`; `fam/krea_schedule.rs`, `fam/qwen_image_2_1/scheduler.rs`, `fam/ideogram_scheduler.rs` |
| P11 | **img2img start-step not in gen-core**; ≥6 Candle re-implementations with differing defaults (0.3/0.5/0.6). | §10.4 |
| P12 | **Silent-ignore exceptions to "refuse, don't ignore"**: `phases` has no capability gate; `timestep_to_start_cfg`, `control_scale`, `text_style_gain`, `trim_first_frames`, SVD fields are documented "other models ignore it". | `request.rs:25-30,91-104,124-129,206-232,325-330` |
| P13 | **Receipts via `&self` getters after generate**, requiring callers to serialize calls; no per-request report object for most facts. | `traits.rs:33-69` |
| P14 | **Model defaults outside the descriptor**: SceneWorks `MODEL_TABLE` owns default steps/guidance/repo; engine internals own native schedules. | `SW: engines.rs:22-37` |
| P15 | **Product bypasses bundles**: SceneWorks depends on ~17 individual engine crates; iOS uses per-family facades. | §3.4, §13; `runtime-ios/src/lib.rs:402-483` |
| P16 | **Contract crate still hosts family data**: family latent spaces, VAE tiling presets, checkpoint-adapter constants, `ImagePipelineArchitecture` closed enum, H3 trainer constants in weightsmeta, H3-only telemetry crate. | `gc/latent.rs:343-543`; `gc/tiling.rs:93-154`; `gc/lib.rs:265-267`; `gc/memory_phases.rs:74-83`; `gc/weightsmeta.rs:1181-1188` |
| P17 | **Declarative weight plan adopted by ~5 families**; most use hand-coded `from_weights(prefix)`. | §12.1 |
| P18 | **Registry lookup is a linear scan calling `descriptor()` (which allocates `Vec`s in `Capabilities`) per entry** on every load/footprint. | `providers.rs:42-44,703-705` |

---

## 16. Design implications (what a shared pipeline/hook layer must preserve, extend or replace)

### Preserve

1. **Tensor-free contract crate + backend kernels** split, and the planner/kernel decomposition already
   proven by tiling (§8.2), attention budget (§8.3), block window (§8.4) and residency (§8.1). A hook
   layer's *policy* (which steps, which resolution, which rung) belongs in gen-core; only tensor ops
   (resample latent, VAE encode/decode, forward) belong in backends.
2. **Explicit composition** (registration values, catalogs, bundles, duplicate-id rejection,
   weights-free conformance) — `docs/architecture/inference-rearchitecture.md:313-362`.
3. **Byte-identity when a feature is off.** Every existing lever is "off is absence"
   (`approximation.rs:22-47`; `request.rs:325-327`; `GenerationMemory` `None` = historical path).
   Hooks must default to no-ops that do not perturb numerics (cf. `run_*_with_latent_hook` legacy
   wrappers, `mlx/sampler.rs:514-517`).
4. **Refuse-don't-ignore gating** via `Capabilities` + the shared floor + testkit honesty checks.
   A cross-cutting feature needs a declared per-provider surface and floor validation (and should fix
   P12 for `phases`).
5. **The memory contract's separation of concerns**: caller owns budget and least-cost selection;
   provider can reject but not substitute (`memory_strategy.rs:1-11`). A multi-resolution or
   multi-phase feature changes peak by phase, so admission must price the max over phases (`MemoryGeometry`
   is single-resolution today, `selection.rs:25-32`; `MemoryRunContext.has_phases` is only a bool).
6. **Result-changing vs equivalence-preserving classification** (`execution_domains` vs
   `approximation`). Low-res-first-K is result-changing; today `CharacterizationFamily` is uninhabited,
   so as designed it would be structurally refused — the re-architecture must decide whether to add a
   characterization family or a separate "creative/recipe" class (it is arguably a sampling recipe like
   `phases`, not an approximation).
7. **Typed cancellation, monotone progress, seed determinism, preview dedup by schedule position**
   (testkit §11; `cdl/sampler.rs:175-183`). Progress `Step{current,total}` must stay monotone across
   phase/resolution segments; `Progress` is `Copy+Eq` and exhaustively matched, so new events need care
   (`traits.rs:187-193`).
8. **LoadSpec's "typed slots + generic `components` map"** pattern and the no-self-fetch rule.

### Extend

9. **Promote the Krea multiphase primitive to a generic trajectory-segmentation contract** in gen-core:
   ONE global schedule, contiguous slices with shared boundary sigmas (`fam/krea_multiphase.rs:10-20`),
   per-segment {guidance, adapters, **resolution**, sampler?}. Add a `supports_phases`-style capability.
   Resolution-changing segments additionally need (a) a latent resample op on `LatentOps` (or a new
   backend op trait) informed by `LatentSpace{spatial_compression, patch_layout}` (`gc/latent.rs:144-149`),
   (b) per-segment seq-len/mu (P10) — decide whether mu is fixed by the *final* resolution or recomputed,
   (c) a rule for multistep solver history across boundaries (P9).
10. **A shared sampling driver in gen-core, generic over `LatentOps`**, owning: solver lookup, cancel
    check per eval, progress counting, preview counter, latent observation, and segment iteration —
    leaving backends only `LatentOps` impls and the `predict` closure. Both backend drivers already
    have this shape (`cdl/sampler.rs:199-231`); unify signatures (P8) and move `resolve_schedule`/
    `init_time_step`/img2img strength resolution into gen-core (P11).
11. **Generalize `MemoryRequestScope` into a request-scoped hook/observer set** (phase enter/leave,
    per-step, per-segment, pre/post decode) or add a sibling trait; it is already the per-request
    lifecycle object the engine calls into (`run.rs:191-217`) and backends already have shared cores.
    Keep memory hooks distinct from creative hooks to preserve the "provider cannot substitute" rule.
12. **Lift `Residency::run_staged_*` into an explicit stage pipeline** (prepare → encode → [segments of
    denoise] → decode) where the denoise stage is the shared driver of item 10 rather than an opaque
    closure. Item P7's deferred `Pipeline` trait is the documented placeholder for exactly this.
13. **Move model defaults onto the descriptor** (native steps/guidance/schedule/shift/strength default)
    so SceneWorks' `MODEL_TABLE` and per-family constants stop being second sources of truth (P14).
14. **Typed extension slots on the request** (per-family typed sub-blocks, like `audio`/`moss_image`/
    `veda`) instead of top-level fields, and move id-string gates into per-provider declarations
    (P2-P4). `first_nonfinite_float`'s exhaustive destructure must be preserved or replaced by an
    equivalent per-block float sweep.

### Replace / retire

15. Family-named `Generator` methods → per-family extension traits discovered via a typed
    downcast/query on the generator, or a generic `report()` object returned by generate (P1, P13).
16. Stringly selectors → typed enums at the contract boundary (keep string names only in
    serialization) (P5).
17. Duplicated vocabularies (P6): one memory-parameter struct; one phase vocabulary with distinct names
    for "lifecycle stage" vs "trajectory segment".
18. Legacy `SamplerPolicy` layer and MLX `DiffusionSampler`/wrapper samplers once engines migrate
    (`sampling.rs:25-31`).

---

## 17. Uncertain / not verified

- **MLX catalog composition** (`mlx-gen-catalog/src/lib.rs`) was not read line-by-line; assumed to mirror
  the Candle catalog per the rationale doc. MLX family count (43 dirs) includes non-registry utilities
  (`_vendor`, `docs`, `scripts`, `tests`, `tools`, `src` are not families; actual `mlx-gen-*` count is
  43 including catalog crates and utilities like `mlx-gen-clip`, `-face`, `-depth`, `-sam2/3`, `-pid`,
  `-perceptual`, `-body`).
- Counts like "48 crates call the sampler drivers", "~29 Residency users", "49 ProviderMemoryProfile
  users", "102 files with Generator impls" are `git grep -l` file/crate counts, not call-site audits;
  some hits may be tests or re-exports.
- I did not read `mlx-gen-krea/src/multiphase.rs` / `candle-gen-krea` driver bodies to confirm *how*
  solver state is handled across phase boundaries (whether each slice re-instantiates the solver,
  dropping DPM++/UniPC history). The doc says "drives `run_flow_sampler` over each slice", which implies
  re-instantiation; unverified.
- `GenerationRequest` field classification "generic vs family" is my judgement from doc comments; some
  "family" fields (e.g. `use_pid`, `enhance_prompt`) are honored by several families.
- `Capabilities` field count (54) and request field count (61) are regex counts of `pub name:` lines.
- `RuntimeCatalog::try_new` validation details and `RuntimeCatalogSnapshot` schema were not read.
- `docs/migration/*` memory-contract history docs and `docs/guide/getting-started.md` were not read.
- Whether `scheduler_shift` is validated/consumed uniformly was not checked.
- The SceneWorks side (how the worker maps job payloads into `GenerationRequest`, MemoryRunContext
  construction, admission) is only sampled here (`inference_runtime.rs`, `engines.rs`); other
  researchers presumably cover it.
- I did not verify `load-progress/src/reader.rs` or `gen-core-testkit` memory-strategy checks.
