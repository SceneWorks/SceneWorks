# Adversarial review: engine side of Shared Generation Architecture (Draft 0.1)

Reviewer scope: REQUIREMENTS PIPE, HOOK, COMP, BACK, WGT, MEM, REG, XL, FEAT, PERF; IMPLEMENTATION_PLAN §1–3, §5, §6, §7, §9 ENG-1..ENG-8.
Code read: inference `origin/main` = `a43724b82`; prior art `origin/feature/sc-20414-advanced-multi-pass-sampling` = `34222b1de` (merge-base `0843ff0923`, 47 commits ahead, 4,212 behind); SceneWorks `a1d95e2c4`. Every `file:line` below was checked by me. `B=` means the sc-20414 branch; unprefixed inference paths are `origin/main`.

## Summary verdict

**The direction is right, but the engine design is not implementable as written.** Five defects block it:

1. **The base is mis-described.** The prior-art executor the design builds on is not what the plan says it is. It runs fresh schedules only and always re-noises between passes. It cannot change geometry. Its host owns the guidance combine. It explicitly refuses Krea phases. Porting it is close to a rewrite.
2. **The runner rules contradict the bit-identity requirement for Wan.** The invariants in §3.3 (one solver per phase, step-granular phases) contradict PIPE-10 for Wan MoE, which carries solver history across the expert switch and picks the expert per evaluation.
3. **The seed rule breaks identity and stored seeds.** PIPE-11's unified seed derivation contradicts PIPE-10 bit-identity. It also breaks a SceneWorks contract that is persisted today.
4. **The family contract changes speed and numbers.** `FamilyModel::forward` runs one branch at a time and the runner owns the combine. That cannot reproduce today's batched-CFG forwards or the family-specific combine formulas without a speed loss and changed bits.
5. **The sequencing is inverted.** ScaleContext and the Linear spine land in ENG-4/5. ENG-1 (Krea) and ENG-2b (LTX pass scales, Wan MoE adapters) depend on them.

Beyond these, the design makes three wrong assumptions:

- **A single latent stream.** The stage model assumes one latent, one scalar σ and one schedule. LTX AV and H3 break all three.
- **Two schedule modes.** It assumes two schedule modes cover every case; LTX's baked stage-2 table is neither.
- **A tolerance class covers numeric changes.** It assumes a "tolerance class" is enough for unavoidable numeric changes. On chaotic samplers they are visible changes, not tolerances.

The memory relocation (§3.9) understates what families must still supply. It omits the engine-side iOS admission and pricing system entirely.

The family lists in ENG-1/2 miss several generation families: MOSS-Image, Supra2, Iris, SeedVR2, PiD, InstantID and PuLID. Krea-realtime is an autoregressive self-forcing loop but is listed as a shared-driver family. So "no route bypasses the runner" cannot be reached as planned.

None of this argues against the program. It argues for a revised §3.3 contract, a revised DEC-3/DEC-7/PIPE-11, and a re-sequenced ENG-1/2/4/5 before any tracker items are written.

---

## Findings

### 1. BLOCKER: sc-20414 is not "the core of the phase runner"; ENG-1 item 1 is infeasible as written

**Where:** REQUIREMENTS §1.3, DEC-4; PLAN §2 row 1, §3.3, §7.5, ENG-1 item 1.

**Problem.** The plan describes the sc-20414 executor as already implementing the runner invariants and as "adopted by Krea, SD3 and SDXL on both backends". It then has ENG-1 item 1 "port … and re-express Krea multi-phase through it". The code says otherwise:

- **Fresh schedule per pass only.** It never slices a global schedule. Every pass after pass 0 is unconditionally re-noised: `let renoised = pass.index > 0;` (B:`crates/contracts/gen-core/src/sampling/pass_executor.rs:274`). There is no `Continue` transition.
- **Krea phases are explicitly a different mechanism and mutually exclusive.** The adapter lists have *opposite* meanings: for a phase, empty means bare base; for a pass, empty means leave the load-time stack. See B:`crates/contracts/gen-core/src/denoise_passes.rs:32-35, 50-54` and the `MutuallyExclusiveWithPhases` refusal at `:433-435`. The executor cannot host Krea without new semantics.
- **No geometry, resample or step-context concept.** The latent shape is invariant; it only does `randn_like` + `axpy` (B:`pass_executor.rs:282`).
- **The host owns the CFG combine.** "returning the RAW (already CFG-combined) output" (B:`pass_executor.rs:36, 143`). This is the opposite of PIPE-8 and §3.3 "runner owns CFG combine".
- **"Adoption" was for the new `denoisePasses` request only.** SDXL's multi-pass chain "rides the CURATED lane … never the bespoke ancestral default" (B:`crates/media/mlx-gen/mlx-gen-sdxl/src/model.rs:239-246`). Single-pass and native loops were left on legacy paths.
- **Port distance.** On `main` since the merge-base:
  - `gen-core/src/generator.rs` lost 5,126 lines and `registry.rs` lost 4,605 (both split into modules);
  - `mlx-gen-sdxl/src/model.rs` +1,835/−695;
  - candle `krea/src/lib.rs` +2,143;
  - `sampling/solvers.rs` +573.

  The executor also depends on `Solver::model_evals` and the `Rk67s` solver, which exist only on the branch: main has 0 `fn model_evals` and 0 `Rk67s` hits in `gen-core/src/sampling`. The branch also changed ~60 family files to add `supports_denoise_passes`. The 3,388-line `denoise_passes.rs` is a hand-written JSON wire contract (`advanced.denoisePasses`). It competes with the typed `RenderPlan`/`OptionSpec` path in REG-4/OPT-6.

**Proposed fix.**
- **Restate DEC-4:** "salvage the executor's invariants, conformance testkit and ComfyUI pass-schedule math (`denoise_pass_schedule_steps`, `terminal_pass_segment`); do not port `denoise_passes.rs` as a wire contract."
- **Decide separately** whether `denoisePasses` becomes a `FreshPerPass` Feature, with its `ParamSchema` derived from that contract.
- **Rewrite §2 row 1** to list what the executor lacks: GlobalSliced/Continue, geometry, step context, runner-owned combine and phase adapter semantics.
- **Change the walking skeleton** (ENG-1 item 1) to FLUX.1 single-phase on the runner plus low-res start. Move the Krea re-expression after ScaleContext exists (see finding 5).

### 2. BLOCKER: The runner's "one solver instance per phase" and step-granular phases contradict bit-identity for Wan MoE and Wan curated MoE

**Where:** PLAN §3.3 invariants ("one solver instance per phase"); PIPE-3 accept ("MoE … re-expressed as phase lists … goldens unchanged"); §3.1 `PhasePlan.steps: StepSpan`, `PhasePlan.model: ModelSlot`.

**Problem.**
- **Wan native MoE carries multistep solver history across the expert switch.** "One continuous `sched` advances across the split, so a multistep solver's history is unbroken (bit-exact …)" (`crates/media/mlx-gen/mlx-gen-wan/src/pipeline/moe.rs:33-40, 99-102`). A fresh solver per phase changes UniPC/DPM++ output.
- **Krea is the opposite case.** It runs `run_flow_sampler` per slice, so history resets. The runner needs both behaviours.
- **The curated MoE path selects the expert per evaluation, keyed on σ.** It does this inside predict, so a Heun/DPM++-SDE sub-evaluation at an intermediate σ can use the other expert: `let t = (sigma * nt).trunc(); if t >= boundary …` (`moe.rs:139-145, 187-194, 237`). A phase boundary at step granularity cannot express this bit-identically.
- **§3.3 contradicts itself.** S8 mentions a "solver-history policy", but the invariant list states one solver per phase.

**Proposed fix.**
- **Separate segment boundaries from solver lifetime.** Add `PhasePlan.history: SolverHistory::{Carry, Reset}`. The rule:
  - `Continue` with the same geometry may carry history;
  - any geometry change or `Renoise` forces `Reset` (a typed validation in Plan).
- **Make the model slot σ-keyed.** `ModelSlot` selection becomes a σ-keyed rule, `slot_for(σ)`, evaluated per model evaluation. Step-span phases are kept for guidance and adapters.
- **Rewrite the invariant** to: "solver history never crosses a geometry change or re-noise".
- **Add a Tier 0 test:** runner MoE equals the legacy `denoise_moe` and `denoise_moe_curated` traces over `CpuLatentOps`.

### 3. BLOCKER: PIPE-11 seed unification contradicts PIPE-10 and breaks a persisted SceneWorks contract

**Where:** PIPE-11 ("One splitmix-style derivation replaces today's `wrapping_add(i)`, LTX constants and candle `+1..+3` variants"); PLAN §3.5 Seeds; DEC-7.

**Problem.**
- **Every count>1 output changes.** Replacing per-image `base_seed.wrapping_add(index)` changes every image after the first. Examples: `candle-gen-chroma/src/pipeline.rs:442`, `candle-gen-ideogram/src/pipeline.rs:667`, `candle-gen-kolors/src/pipeline.rs:613`; 158 `wrapping_add` seed sites across media.
- **This is not a tolerance.** It is a different random draw, so DEC-7's tolerance classes cannot cover it.
- **The worker relies on it.** The SceneWorks worker labels image *i* of an engine batch with `seed.wrapping_add(index)` (`crates/sceneworks-worker/src/image_jobs/stream.rs:441`). That per-image seed is persisted metadata used for regenerate and replay. A unified splitmix silently breaks "regenerate image #3", against AP12/MIG-5.
- **LTX would change too.** LTX's `seed ^ 0x5DEE_CE66` (MLX) and `seed+1..+3` (candle) are parity-pinned noises. Unifying them changes LTX outputs.

**Proposed fix.**
- **Make seed derivation a route-declared `SeedScheme`.** Today's derivations become named legacy schemes that are bit-exact and recorded in receipts.
- **Use the splitmix/salt derivation only for new streams,** such as feature re-noise and new phases.
- **Add to PIPE-11:** "existing per-image and per-stage derivations are preserved; the image-index → seed relation exported in the capability record is the one SceneWorks persists."
- **Delete** "replaces today's `wrapping_add(i)`".

### 4. BLOCKER: `FamilyModel::forward(branch)` + runner-owned combine cannot reproduce batched CFG or family combine formulas

**Where:** PLAN §3.3 `FamilyModel::forward` ("One model evaluation for one guidance branch. The runner owns CFG combine"); PIPE-8 ("standardized row order"; "no family file contains a guidance combine"; "goldens unchanged").

**Problem.**
- **Many families batch cond and uncond into one B=2 forward.** For example candle SDXL: "the cond + uncond rows in one batched forward" / `Tensor::cat(&[&latents,&latents],0)`, then `chunk(2,0)` and combine (`crates/media/candle-gen/candle-gen-sdxl/src/denoise.rs:300-349`). Wan does the same (B=2 `[cond, uncond]`, per [B §1.5]). Per-branch forwards either double the launches and halve the effective batch (a PERF-1 regression) or change bits: batch-1 and batch-2 kernels differ in tiling.
- **The combine formulas differ and are not interchangeable bitwise:**
  - Boogu computes `cond + (s−1)(cond−uncond)` in model dtype, then casts to f32 and negates (`crates/media/mlx-gen/mlx-gen-boogu/src/pipeline.rs:563-576`);
  - candle SDXL computes `uncond + (cond−uncond)·cfg` (`denoise.rs:349`);
  - FLUX3 base rounds dtype inline [B §1.8];
  - LTX dev has four branches, including an STG-perturbed and an isolated-modality *model* variant [B §1.6].

  STG and isolated modality are model-internal perturbations, not conditioning rows.
- **PIPE-8's accept conflicts with PIPE-10.** "Standardized row order" plus "goldens unchanged" cannot both hold except by luck.

**Proposed fix.**
- **Change the contract** to `predict(ctx, x, t, branches: &BranchSet) -> BranchOutputs`. The family decides how to batch.
- **Make `Branch` able to carry a model perturbation** (skip-layer, modality isolation) that `BlockStack` hooks honour.
- **Key the guidance library by `CombineVariant`.** Each today-formula is kept bit-exact: operand order, dtype of combine, sign convention.
- **Treat canonical unification as a separate recipe or re-baseline decision,** not part of the migration.
- **Restate PIPE-8:** "every combine is a *named* operator in gen-core; row order is a declared property of the route", rather than one standardized order.

### 5. BLOCKER: Dependency inversion in §9: per-phase adapters are needed in ENG-1/ENG-2 but built in ENG-4/ENG-5

**Where:** PLAN §9 ENG-1 item 1 (Krea re-expression), ENG-2b ("LTX … pass scales via `ScaleContext`", "Wan … MoE as phases"); ENG-4 ("`ScaleContext`"); ENG-5 ("per-phase adapters on every residual-capable route"); PIPE-3 accept (delete the family-local mechanisms).

**Problem.** Krea per-phase adapters work differently on each backend:
- MLX clones the DiT per phase;
- candle re-adapts one job-local, non-`Clone` transformer in place (`crates/contracts/gen-core-family/src/krea_multiphase.rs:36-45`).

Re-expressing Krea, LTX pass scales and Wan MoE adapter filtering through the runner, *and* deleting the family mechanisms (PIPE-3 accept), requires the shared adapter scale context first. The roadmap puts that 2–4 features later. The alternative is a temporary family hook that survives across features, which is forbidden by MIG-2/AP9.

**Proposed fix.** Pull the Linear-spine subset needed for scales forward into a feature that precedes ENG-2b: `ScaleContext`, the residual install policy and the candle `AdaptLinear`→spine unification. Alternatively, move PIPE-3's adapter axis (Krea, LTX pass scales, Wan adapter filtering) out of ENG-1/2 into the feature that introduces ScaleContext. Remove Krea from ENG-1.

### 6. MAJOR: The stage model is single-stream; AV families need latent bundles, per-stream schedules and stream-targeted transitions

**Where:** PLAN §3.1 `Geometry`, `Transition`; §3.3 `RunnerBackend::Latent`, `StepInfo { sigma: f32, next_sigma: f32 }`; PIPE-4, PIPE-5, PIPE-6.

**Problem.**
- **H3 runs one forward that yields `(v_video, v_audio)`,** stepped by **two schedules** with shifts 12.0 and 3.0 (`crates/media/mlx-gen/mlx-gen-minimax-h3/src/denoise/mod.rs:1-13`).
- **LTX upsamples video but only re-noises audio.** "re-noise audio (never upsampled)" (`crates/media/mlx-gen/mlx-gen-ltx/src/pipeline.rs:1579-1588`).
- **The plan's types cannot express either.** `Transition::Resample` has no stream target. `StepInfo` carries one σ.
- **A precedent exists:** candle already has a two-stream `AvLatents` (`crates/media/candle-gen/candle-gen/src/sampler.rs:374, 420-468`).
- **LTX also uses per-token timesteps** (σ·mask) [B §1.6], so `TimeInput` cannot be a scalar.

**Proposed fix.**
- **Make `B::Latent` a `LatentBundle`** keyed by stream, each stream with its own `LatentSpace`.
- **Give schedules per-stream σ.** `ResolvedSchedule` becomes per stream, and `StepInfo` carries a per-stream σ or a σ-map.
- **Give `Transition` stream targets,** for example `Resample { streams: StreamSet, … }`.
- **Let `TimeInput` carry an optional per-token time tensor builder.**
- **Specify `Geometry` per stream:** audio tokens/duration versus the video grid. Today it holds only w/h/frames/batch/refs.

### 7. MAJOR: DEC-3's two schedule modes don't cover LTX; shared transitions are not bit-exact with LTX's re-noise

**Where:** DEC-3; PIPE-4; PLAN §3.3 "Schedule modes".

**Problem.**
- **LTX is neither slice nor fresh schedule.** Its stage 1 runs a *full* 8-step trajectory to σ=0 (`STAGE1_SIGMAS` ends at `0.0`, `pipeline.rs:51-53`). It then re-noises to `STAGE2_SIGMAS[0]=0.909375` and runs a **baked** 3-step table (`:56, :591`). That is not a slice of a global schedule, and it is not a ComfyUI `steps/denoise` fresh schedule.
- **LTX's re-noise is dtype-preserving,** with `1 − scale` computed in latent dtype (`pipeline.rs:157-162`). The executor re-noises with f32 scalars, `ops.axpy(k_x0, …)` (B:`pass_executor.rs:282`), and draws `randn_like(…, denoise_pass_renoise_seed(..))`, not LTX's pre-drawn stage noises. The bits differ.

**Proposed fix.**
- **Add a third `StepSpan` kind,** `Explicit(ScheduleId/table)`.
- **Allow a route to supply a bit-exact transition implementation.** In the hook map, give S6 `transition` an owner of "runner | Feature | **route**". The default shared transition is used only for new recipes.
- **Expand DEC-3's options** to "global-sliced, fresh-per-pass, explicit-table".

### 8. MAJOR: DEC-7 / PIPE-10 tolerance policy is under-specified and partly unworkable

**Where:** PIPE-10, DEC-7, DEC-14, MIG-3 Tier 1, AP3.

**Problem 1: changed numerics are not confined to three cases.** DEC-7 names only fold→residual, CFG row order and merged VAE kernels. The migrations that will change numerics also include:
- encoder consolidation onto llm;
- VAE parametric merge;
- the sub-block kit (RMSNorm via fused vs manual, RoPE variants, ~45 sinusoid copies with different f32/f64 order);
- native dtype-preserving loops (LTX "legacy dtype-preserving Euler") moved onto f32 gen-core solvers;
- per-branch vs batched forwards (finding 4);
- seed unification (finding 3);
- MLX compile-glue regions moving.

**Problem 2: on chaotic samplers a ULP is not a tolerance.** candle SDXL documents that `(W+δ)x` vs `Wx+δx` "differs … by ~1 ULP, which cascades to a visibly different image" (`crates/media/candle-gen/candle-gen-sdxl/src/adapters.rs:6-11`). An image-level tolerance class therefore cannot certify these changes. They are re-baselines, so a product decision.

**Problem 3: Tier 1 synthetic or tiny weights do not exercise the real kernel paths.** Quantized block formats need real dims: "real model dims satisfy this, but tiny synthetic weights may not" (`crates/llm/candle-llm/src/primitives/quant.rs:8`). MLX also selects kernels by shape.

**Problem 4: Tier 1 bit-identity has no specified reference.** Checked-in goldens are machine-dependent (standing rule: no machine goldens).

**Problem 5: AP3 "allocation order" is untestable and unachievable** once orchestration moves.

**Proposed fix.**
- **Define three classes:**
  - (a) *bit-exact*;
  - (b) *component-tolerance*: a bounded divergence measured at a component boundary (TE output, single DiT forward, VAE decode) on shape-faithful weights;
  - (c) *accepted re-baseline*: a visible change that needs per-route sign-off from Michael, listed in the PR per PLAN §7.3.
- **Enumerate the expected class per ENG feature now.**
- **Tier 1 runs:** "build base and head on the same runner, run shape-faithful random weights (real dims, truncated depth), compare".
- **Replace "allocation order"** with "peak memory within PERF-2".

### 9. MAJOR: "Candle defaults to residual" (PLAN §3.8) is a self-inflicted fidelity and performance regression

**Where:** PLAN §3.8 Adapters; WGT-3; PIPE-10 example; PERF-1.

**Problem.** Fold-at-load on dense bases exists *for* parity: SDXL, Z-Image, Wan, SD3, Krea single-phase, SCAIL2 and Lens (`candle-gen-sdxl/src/adapters.rs:6-11`; [E §3.4 table, line 432]).
- **Fidelity.** Switching to residual by default is a visible output change on every LoRA render of those families. It is not a necessary operation-order change, and it violates the non-goal "re-tuning numerics" and AP3.
- **Performance.** Residual costs 2 extra GEMM launches per adapter per Linear per forward on CUDA. [E §3.6] gives ~1–4% compute per adapter, plus launch overhead, unmeasured. Multi-LoRA stacks will likely breach PERF-1's ±2%.
- **Memory is the one gain.** Residual saves the transient host dense copy, but fold already works on dense bases.

**Proposed fix.**
- **Keep today's per-route install policy as the default.** That satisfies AP3: off means byte-identical.
- **Switch a route to residual only when a Feature needs runtime scales,** such as per-phase adapters. This is result-changing anyway and declared in the plan.
- **Optionally use a stack-fused residual** (concatenated ranks) to bound launches.
- **Delete** "Candle defaults to residual" and the matching PIPE-10 example.

### 10. MAJOR: `ScaleContext` must be request-scoped, not state on a shared resident Linear

**Where:** BACK-1 ("residual adapters … with a request-time scale context"); PLAN §3.8 ("`ScaleContext` is set by the runner per phase").

**Problem.** Today's invariant is "keep the **shared resident DiT unmutated** — the concurrency-safety invariant". Per-phase adapter work happens on job-local copies (`gen-core-family/src/krea_multiphase.rs:36-45`). A runner that *sets* a scale on a shared `Linear` races concurrent generates on one cached provider.

**Proposed fix.** Specify that the scale context is passed through the forward as part of `StepCtx`/`ForwardArgs`, or thread-scoped per request, never stored on shared modules. Add a Tier 0 concurrency test: two interleaved requests with different phase scales.

### 11. MAJOR: The `Feature` trait sketch is not object-safe; "one module per Feature" holds only for plan-rewrite Features

**Where:** PLAN §3.4 (`trait Feature … fn hooks<B: RunnerBackend>(&self, …)`; "The bundle composes the Feature list"); G1, PAT-1.

**Problem.**
- **The registry needs `dyn Feature`.** A generic method makes `Feature` non-dyn-compatible, yet a heterogeneous Feature list requires it.
- **Tensor hooks can't live in gen-core.** Hooks that touch tensors beyond `LatentOps`/`GuidanceOps` cannot be written in gen-core, which has zero tensor deps (`crates/contracts/gen-core/Cargo.toml`, "depends on NOTHING tensor-related"). Examples are feature caches, STG/PAG-style attention perturbation and token pruning. They need per-backend code until DEC-1 (c).

**Proposed fix.**
- **Split the trait** into `FeatureSpec` (tensor-free and dyn: params, class, applicability, relations, `rewrite_plan`, `memory_effect`) and per-backend `FeatureHooks<B>`, registered in L1.
- **State in G1/PAT-1** that "one module" applies to plan-rewrite and LatentOps-expressible Features. Model-internal hooks are one module *per backend* until DEC-1 (c).

### 12. MAJOR: The memory relocation (§3.9, MEM-1) overstates "families supply nothing" and omits iOS admission and pricing

**Where:** MEM-1, MEM-2, MEM-3, XL-3; PLAN §3.9 table.

**Problem.**
- **R1 (staged residency) needs a per-route component lifetime graph, not nothing.** Examples:
  - H3 encodes keyframes with the VAE *before* the DiT is mapped and decodes at the end [B §1.7];
  - LTX builds the transformer after text is freed, and the upsampler lives between stages [B §1.6];
  - Wan's staged expert swap loads, drops and evicts mid-denoise (`moe.rs` `denoise_moe_curated_swapped`);
  - FLUX3 PUP and the LTX enhancer are LLM pre-stages;
  - SenseNova's "encoder" is the denoiser's own KV prefill.
- **R2 tiling laws are codec-specific code.** Examples are Wan vae16/vae22, LTX causal temporal overlap and DC-AE. Tiled decode is *not* output-preserving: gen-core keeps tiled-decode quality records (`crates/contracts/gen-core/src/memory_strategy/decode_quality.rs:1-12`). See finding 13.
- **Vision-language (VL) staged text encoders are priced by reference pixel geometry.** Qwen-Image edit and FLUX3 references produce vision tokens. MEM-3 lists only "reference count".
- **iOS is not covered at all.** The engine has its own admission and pricing system:
  - `mlx-gen-ios-catalog/src/ios_admission.rs` (~185 KB), `evidence.rs`, `pricing.rs`;
  - per-family `ios_pricing/` modules (sana 15 files, z-image 9, sdxl 3, iris 3);
  - per-quantum OS-headroom checks before each graph construction, including inside the Iris solver (`crates/media/mlx-gen/mlx-gen-iris/src/solver.rs:12-15`, `STEP_QUANTUM`/`PLAN_QUANTUM`).

  ADM-1's "18 paths" are SceneWorks-only. XL-3 says only "iOS quantum pricing is fed from plan bytes".
- **The runner must honour 47 `MemoryRequestScope` implementations until ENG-6.** They exist today (`impl … MemoryRequestScope for` = 47; trait at `crates/contracts/gen-core/src/memory_strategy/run.rs:191-217`), with 70 `memory_strategy*` files. Strategies are relocated only in ENG-6. ENG-1..5 must call `enter_phase`/`configure_decode`/`materialize_transformer_window`/`finish` from the runner. Otherwise admitted rungs silently stop applying, and MLX OOM kills the process.

**Proposed fix.**
- **Replace the "nothing" column** with the actual declarations: component lifetimes per stage, codec tiling law, block layout and quantum labels.
- **Add `MEM-5`:** "the runner drives the existing `MemoryRequestScope` lifecycle for every migrated route until its rung is relocated".
- **Add an iOS admission requirement** under ADM or XL-3, with its own relocation item.
- **Add reference pixel geometry** to MEM-3.

### 13. MAJOR: MEM-4's acceptance ("admission never changes the output") is false for today's rungs

**Where:** MEM-4, HOOK-2 class "equivalent (output-preserving, for example memory rungs)", PAT-2.

**Problem.** Tiled decode changes pixels: there is a dedicated quality-receipt machinery for it (`decode_quality.rs`). Bounded or chunked attention changes the reduction order. ADM-4 "tiers tried" implies tier selection by admission, and tier is a creative choice (standing rule).

**Proposed fix.**
- **Redefine `Equivalent`** as "within a declared, recorded quality bound".
- **Make the accept clause:** "admission never selects a recipe-class Feature and never changes quant tier".
- **Remove "tiers tried"** unless tier demotion is explicitly in scope.

### 14. MAJOR: Hires.fix cannot be an engine phase recipe with the proposed `Transition` set

**Where:** HOOK-2 accept (Hires.fix as a Feature); PLAN §5 ("Hires.fix (as a phase recipe)").

**Problem.** Today the Hires.fix base pass *decodes to RGB*, and the result becomes the img2img reference of the refinement pass. It also has its own first-pass memory selection (`crates/sceneworks-worker/src/image_jobs/base.rs:8548-8575, 8611-8615`). Its parameters are `upscale_by` and denoise strength (`sceneworks-core/src/contracts.rs:889-893`). `Transition` only has latent `Nearest | Bilinear | Learned`. A latent re-expression is a user-visible behaviour change, against the non-goal.

**Proposed fix.** Either add `Transition::PixelRoundTrip { resize: PixelResize, codec_reencode }` and price VAE residency across phases, or drop Hires.fix from HOOK-2's acceptance and pick a different existing behaviour.

### 15. MAJOR: The escape-hatch list and ENG-1/2 family lists are incomplete, so "no route bypasses the runner" cannot be reached

**Where:** PIPE-1 accept, PIPE-7 list, ENG-2 lists, ENG-2 exit.

**Problem.** All of these are on main and not addressed:
- **Krea-realtime** is "autoregressive, self-forcing, CFG-off" with a KV cache across blocks (`crates/media/mlx-gen/mlx-gen-krea-realtime/src/pipeline.rs:1-12`). It is listed as a shared-driver family in ENG-2b, but it needs a `FamilyDriver` and a chunked/streaming output model.
- **Missing from all lists:**
  - **MOSS-Image** (own FP32 host noise + DMD sampling; `mlx-gen-moss-image/src/pipeline.rs:1`);
  - **Supra2**;
  - **Iris**, the iOS family, with its own FlowDPM-Solver++ and iOS quanta (`mlx-gen-iris/src/solver.rs:1-15`);
  - **SeedVR2**;
  - **PiD**, a decoder with its own `sampler.rs`, which the runner must host inside S10 with cancel and progress;
  - **InstantID** (seeds the MLX global RNG, `mlx-gen-instantid/src/model.rs:999`) and **PuLID**.

**Proposed fix.**
- **Add a route inventory to the plan:** generated from the catalogs, one row per (family, route, backend), with the planned loop kind and feature.
- **Make ENG-2's exit** "every row migrated or refused by an approved decision".
- **Add** Krea-realtime, MOSS-Image, Iris and PiD-as-decoder-sampler to PIPE-7 or justify their absence.

### 16. MAJOR: Low-res start coverage on the default SDXL and Wan routes depends on FamilyDriver support the plan never requires

**Where:** PAT-1, FEAT-1, PLAN §6.1 ("Families: none"), §3.2 `DriverSupport`.

**Problem.** SDXL native ancestral (the default) and Wan native UniPC (the default) are `FamilyDriver`s (PIPE-7). Whether low-res start works there depends entirely on each driver's `DriverSupport`, which is hand-declared. If an ENG-2 migration implements `run_slice` only at a fixed geometry and full σ range, low-res start "refuses with a typed reason" on the most-used routes. PAT-1 then passes vacuously.

`DriverSupport` can also lie: it is unverifiable without a mocked forward. H3 shows the pattern that makes this testable: its loop is written against a `JointVelocity` trait (`denoise/mod.rs:25-29`).

**Proposed fix.**
- **ENG-2 acceptance for each FamilyDriver:** honour `SliceInput` with arbitrary entry σ, geometry, `Continue`/`Renoise`/`Resample` and the history policy. Each refusal must be approved by Michael under DEC-13.
- **Require drivers to be generic over a forward seam,** so a Tier 0 conformance suite verifies every declared `DriverSupport` bit.
- **Have PAT-1 list the routes expected to support FEAT-1.**

### 17. MAJOR: ENG-1 is too large for a walking skeleton and contradicts §7.1 and MIG-2

**Where:** PLAN ENG-1 items 0–8; §7.1; §7.4.2; MIG-2; AP9.

**Problem.**
- **Too much for one feature.** ENG-1 bundles:
  - the sc-20414 port (finding 1) and Krea on both backends (finding 5);
  - `RenderPlan` and the unified driver on both backends, plus FLUX.1;
  - `LatentGeometryOps` and seeds;
  - the Feature trait and RouteStructure derivation;
  - capability export v2, typed refusal and an admission adapter;
  - SceneWorks UI;
  - three more family migrations;
  - a Tier 2 proof.

  That is in 3–5 weeks, alongside DEC-11 and two salvage reviews.
- **It contradicts §7.1.** Item 2 migrates "FLUX.1 t2i", but §7.1 says a family migration moves *all* of that family's routes on both backends: FLUX ControlNet (candle), IP (MLX) and img2img.
- **Item 5 is cross-feature scaffolding.** It is "a thin adapter, replaced in APP-3", which MIG-2 ("dual paths … never across features") forbids.

**Proposed fix.**
- **Skeleton:** `RenderPlan` + runner + FLUX.1 (all FLUX.1 routes, both backends) + `LatentGeometryOps` + `low_res_start` + per-phase geometry fed into today's FLUX admission inside the same feature.
- **Move out:** Krea, SDXL/Qwen/Z-Image and export v2 move to ENG-2a.
- **Or state the exception explicitly:** if the admission adapter must outlive ENG-1, record it as a DEC exception.

### 18. MAJOR: ENG-2 and ENG-4 overlap and run past the 6-week window, with a high conflict hazard

**Where:** PLAN §9 ENG-2 (8–12 weeks), ENG-2b scope, ENG-4 ("may overlap ENG-2"), ENG-6 (8–12 weeks), DEC-12, MIG-1.

**Problem.**
- **ENG-2b alone is 13 families on both backends.** LTX, H3, FLUX3, Mage, SenseNova and Bernini are among the hardest code in the repo. It cannot fit in ≤6 weeks.
- **ENG-4 touches all 38 MLX crates that call raw SDPA.** Raw SDPA appears in 38 `mlx-gen*` directories (`git grep -l scaled_dot_product_attention`). ENG-4 also rewrites candle FLUX/Z-Image `packed_dit`/`ip_dit` (`candle-gen-flux/src/packed_dit.rs` 2,513 lines, `ip_dit.rs` 1,759). These are the forwards that ENG-1/2 `FamilyModel` impls wrap.
- **Conflicts surface late.** With merge-commit, `strict:false` mirrored branches, semantic conflicts appear on `main`.

**Proposed fix.**
- **Split ENG-2 by mechanism** into ≤6-week features:
  - 2a flow-DiT shared driver;
  - 2b video shared driver (Wan/Mochi/SVD/SCAIL);
  - 2c FamilyDrivers I (SDXL/Kolors/Wan-native/FLUX3/Boogu/Sana);
  - 2d FamilyDrivers II (LTX, H3, Mage, SenseNova, Bernini, Krea-realtime).
- **Restrict ENG-4** to families whose ENG-2 migration has merged, or run the kit *before* ENG-2c/d.
- **Add a file-ownership table** per feature.

### 19. MAJOR: Missing engine requirements

**Where:** §6 (PIPE/BACK/WGT/XL).

**Problem.** The design never specifies the following, and each is a real decision the runner must make:

- **Count and batching.**
  - Is a multi-image request N sequential runs (Krea reuses per-phase clones "across the seed loop") or one batched latent?
  - Mage packs several samples of different grids into one sequence [B §1.9].
  - The answer sets per-image seeds, previews, cancellation granularity and incremental output.
- **Concurrency model.** Are multiple generates on one cached provider allowed? This determines findings 10 and 22.
- **Cancellation during loads and staged swaps.** For example, a Wan expert swap reloads 14B mid-run. WGT-2 has no cancel contract.
- **Error taxonomy and failure posture in the engine.**
  - "`end` hooks run on error" is unsafe after an MLX Metal error that poisons the process (standing notes on `SubmissionsIgnored`, the GPU watchdog).
  - `FailurePosture` exists only in admission (ADM-2).
  - A typed `RunError` class is needed: request-local, provider-poisoned or process-poisoned.
- **Progress units.** The executor counts *model evaluations*, so Heun shows 2× steps (B:`pass_executor.rs:22-26`). That is a UI behaviour change, so state it.
- **Streaming outputs.** Krea-realtime streams; video may be written incrementally.
- **Interleaved text+image outputs.** SenseNova produces them (PIPE-12 lists them, but the stage model has no text stream).
- **Per-component quant tiers.** For example a q4 DiT with a bf16 TE. Should `RouteRecord` declare it?
- **Cross-backend determinism is a non-goal.** MLX and candle use different noise sources, so say so explicitly.
- **Telemetry.** `gen-core-telemetry` exists and is not mentioned.

**Proposed fix.** Add PIPE-13 (count and batch semantics), PIPE-14 (concurrency), PIPE-15 (engine error taxonomy and posture), WGT-5 (load cancellation) and an explicit cross-backend determinism non-goal.

### 20. MAJOR: COMP-1 (encoders on llm) has unlisted blockers and ignores existing assets

**Where:** COMP-1; PLAN §3.6 "Work needed in the llm crates"; §2 reuse inventory.

**Problem.**
- **Existing assets are not credited.**
  - The llm `CausalLm` already covers Llama, Mistral, Qwen2, Qwen3, Gemma-2, GLM-4 and Gemma 4 (`crates/llm/mlx-llm/src/models/llama.rs:3-8`).
  - It already has a text-encoder tap API, `hidden_states_with_mask_at` with explicit positions and padding masks (`mlx-llm/src/models/llama.rs:1081-1135`; candle twin at `candle-llm/src/models/llama.rs:1289-1391`).
  - flux3, iris, lens and sensenova already depend on the llm crates.

  So the reuse inventory misses the main seam.
- **Blockers that are not listed:**
  - masks are causal-only ("this decoder builds causal masks only", `llama.rs:416`);
  - the API returns all `num_layers+1` hidden states, with no early exit (a memory concern for 12B encoders);
  - the quant codec mismatch: candle-llm stores Q4_0/Q8_0/Q4_K (`candle-llm/src/primitives/quant.rs:54-56`) while the media tiers differ;
  - every encoder swap changes conditioning bits, so it needs DEC-7 class (b) or (c), not "operation order must change".
- **"gpt-oss" is listed as needed,** but no media family that needs it is cited. I could not verify one.

**Proposed fix.** Add the existing tap API to §2 and scope `TapPolicy` as an extension of it (early exit, layer subset, combine). List the codec, mask and numerics blockers explicitly. Drop or justify gpt-oss.

### 21. MAJOR: "Derived, not hand-declared" capabilities overclaim; name the primary declarations and test the ones that can lie

**Where:** AP5, REG-2, HOOK-4, PLAN §3.2.

**Problem.** Many `RouteStructure` facts are, and must remain, declarations, made once:
- `geometry_change` ("from FamilyModel impl");
- `prediction`, `time`, the guidance surface (distilled, no CFG; H3 pins this with a test, `denoise/mod.rs:31-40`);
- legal grids;
- `DriverSupport`;
- preview projection;
- product limits.

That is fine under G3 ("declared once"). AP5's wording, "Hand-declared facts that restate structure are not allowed", plus "derived" in REG-2 will be read as forbidding them. The risky ones are those that can disagree with behaviour (`DriverSupport`, `geometry_change`).

**Proposed fix.**
- **Reword AP5/REG-2** to "every fact has exactly one primary declaration; everything else is derived".
- **Add an acceptance** that each primary behavioural declaration has a Tier 0 conformance check (see finding 16).

### 22. MINOR: Monomorphization and dispatch shape of the runner are unspecified

**Where:** PLAN §3.3 (`FamilyModel<B>` with associated `StepCtx`), R3, XL-3/PERF-2.

**Problem.** `FamilyModel` has an associated type, so `dyn FamilyModel<B>` is impossible without erasure. A generic `run<F: FamilyModel<B>>` instantiates the runner once per family per backend: about 40 per desktop binary. That costs binary size and compile time; generic code is codegen'd in downstream catalog crates, and agent `target/` dirs are already 30–42 GB. Separately, there is no iOS binary-size baseline ([E §7.6] "no measured numbers found"), yet PERF-2 gates on +2%.

**Proposed fix.**
- **Erase `StepCtx`** behind `Box<dyn Any>`, built once per geometry, and dispatch `forward` dynamically. The per-step dyn call is negligible next to a model forward.
- **Keep static dispatch** only for the tensor ops inside the backend.
- **Add an iOS size-baseline capture** to ENG-1 item 0.

### 23. MINOR: Step-context cache and transition memory not specified

**Where:** PLAN §3.3 (`cache.get_or_build(p.geometry, p.model, cond)`), FEAT-1 (`memory_effect: None`).

**Problem.** Today Wan builds the per-expert cross-KV/RoPE cache once per range and drops it afterwards (`moe.rs:56`). A cache keyed across phases holds both experts' or both geometries' contexts simultaneously. A learned resample needs upsampler residency. Low-res start's `memory_effect: None` ignores both.

**Proposed fix.** Add an eviction rule (evict on phase exit unless a later phase reuses the key), and make `memory_effect` include transition residency.

### 24. MINOR: `ResampleKind::Learned(UpsamplerId)` hides a component and a normalization contract

**Where:** PLAN §3.1.

**Problem.** The LTX upsample is `upsample_latents(v, upsampler, mean, std)`, which works in normalized latent space [B §1.6]. It is a route component with weights and residency, not a `LatentGeometryOps` op.

**Proposed fix.** Make it `Learned(ComponentId)`, resolved through `RouteRecord.components`, with a declared normalization.

### 25. MINOR: Factual errors in the docs and evidence

- **PLAN §2:** "`MemoryRequestScope` | `gen-core/src/run.rs:191-217`". There is no `gen-core/src/run.rs`. The trait is at `crates/contracts/gen-core/src/memory_strategy/run.rs:191`.
- **PLAN §2:** "`gen-core/src/guidance.rs` (used by Lens and MLX SDXL only)". It is also used by mlx-gen-bernini, candle-gen-bernini (4 files) and `candle-gen/src/sampler.rs`.
- **REQUIREMENTS §1.3 and PLAN §2:** the sc-20414 executor "adopted by Krea, SD3 and SDXL on both backends". It was adopted only for the new `denoisePasses` request, on curated lanes (finding 1).
- **[E §3.6]:** "re-baseline goldens to residual … also gives cross-backend parity for free". This is false: MLX and candle draw different noise, so outputs differ across backends regardless.
- **MEM-2:** "the 129 `register_memory_contract_fixture` calls". I count 139 call lines on `a43724b82`, excluding the definition. The order of magnitude is right; re-count in §10 baselines.
- **BACK-1:** "about 20k lines of vendored copies". The five named files total 11,856 lines (packed_dit flux 2,513 + z-image 1,986; ip_dit 1,759; dit_train ltx 1,575 + lens 809 + wan 659; train_dit krea 1,406; packed_te flux 775 + z-image 374). Either list the rest or restate. Also, the *training* copies exist because fused kernels lack backward (C8/WGT-4), not because Linear is unhookable. C8 conflates the two.
- **Confirmed accurate:** 4,212/47 commits, the executor's 844 lines and `denoise_passes.rs` 3,388, the 4,516-line `preview_advertising.rs`, 39 twins, the `Residency` :623, `block_window` :152/:194 and `LatentSpace` :144 citations, the `mlx-gen/ARCHITECTURE.md:79-81` deferral, and `one_pass_plan_matches_a_direct_solver_run_bitwise` (B:`pass_executor.rs:520`).

### 26. MINOR: HOOK-1's "hook map generated from code" needs to be framed against AP11

**Problem.** "Generated" plus "each hook has a production user or is deleted" is a new generated artifact and an enforced check. AP11 and TOOL-4 are wary of exactly that.

**Proposed fix.** Specify a typed dump, with no regex and no checked-in matrix, plus a DEC-6 structural test. Alternatively drop the generation and keep the hook enum as the source.

---

## Things that are right and should not change

- **Policy in gen-core, generic over a backend op trait.** This is compatible with the zero-tensor-dep rule. The precedent is real: `Sampler<L: LatentOps>`, `GuidanceOps`, the sc-20414 `execute_denoise_plan<L: LatentOps>`, and candle's `AvLatents` ops. On MLX it works provided `ExecPolicy::materialize` is called at each step boundary: H3 shows the per-step `eval` is what makes cancellation real (`denoise/mod.rs:42-55`).
- **Typed escape hatch (PIPE-7) instead of forcing bespoke loops into the shared driver;** and refuse-never-ignore (AP4).
- **Salvaging sc-20414's invariants:** monotone eval-based progress, cancellation at every eval and boundary, `end_pass` on every exit path, domain-separated re-noise salt, and the CPU-ops conformance testkit.
- **Grid-aligned vs geometry-free conditioning split (COMP-5/`RefBundle`);** `LatentSpace` as the geometry key.
- **Recipe vs equivalent vs diagnostic Feature classes (DEC-8),** subject to the redefinition in finding 13.
- **Layer-level backend seam with DEC-1 (b) first and (c) gated by pilots;** keeping `Fold` available for non-commutative adapters.
- **Characterize-then-consolidate, and no model execution in PR CI (MIG-3 tiers).**
