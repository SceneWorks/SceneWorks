# B — Family pipeline anatomy, unified stage model, and variance catalog

Researcher B of 8. Scope: end-to-end anatomy of representative family pipelines on both backends,
derived stage model + hook points, the variance catalog (essential vs accidental), and how two
cross-cutting features ("first K steps at low resolution", "per-phase LoRA/guidance") would map.

**Source of truth:** inference `origin/main` = `a43724b82`, read only via `git show origin/main:<path>`.
Nothing was built or run. Line numbers are from that revision.

**Path abbreviations used throughout**

| Abbrev | Path |
|---|---|
| `M/` | `crates/media/mlx-gen/` (e.g. `M/src/sampler.rs` = shared MLX core, `M/mlx-gen-flux/src/model.rs` = family) |
| `C/` | `crates/media/candle-gen/` (`C/candle-gen/src/…` = shared candle core) |
| `GC/` | `crates/contracts/gen-core/src/` |
| `GCF/` | `crates/contracts/gen-core-family/src/` |
| `GC3/`, `GCH3/`, `GCL/` | `crates/contracts/gen-core-flux3/src/`, `gen-core-h3/src/`, `gen-core-ltx/src/` |

Confidence tags: **[H]** read directly at the cited line; **[M]** inferred from adjacent code or doc comments;
**[L]** plausible, not verified (collected again in the final section).

---

## 0. What already exists that a shared pipeline can build on

Before the per-family walk, here is what is already shared. The design should extend these, not replace them.

| Seam | Where | What it covers | Adoption |
|---|---|---|---|
| Curated solver driver (any prediction type) | `M/src/sampler.rs:457` `run_curated_sampler`, `:518` `_with_latent_hook`; `C/candle-gen/src/sampler.rs:188` | Solver loop, per-eval cancel + progress, MLX per-step `eval` (`M/src/sampler.rs:237`,`:264`), observer of actual outer steps (`CuratedStepObserver`, `M/src/sampler.rs:478-511`) | Via `run_flow_sampler` wrappers (`M/src/sampler.rs:750`,`:763`; `C/…/sampler.rs:247`). About 15 families **[H]** for flux/qwen/z-image/boogu/sana/sdxl-curated |
| CFG++ driver | `M/src/sampler.rs:626`,`:638` | Solver that renoises from the uncond x0 | **MLX SDXL only** (`M/mlx-gen-sdxl/src/pipeline.rs:1231`). No candle twin **[H]** |
| Joint AV curated driver | `M/src/sampler.rs:783` `AvLatents`, `:842` `run_av_curated_sampler`; `C/…/sampler.rs:434` | Two-stream latents under one solver | LTX curated only **[H]** |
| SCM (TrigFlow) driver | `C/candle-gen/src/sampler.rs:624` `run_scm_sampler`, `:651` | Angle-space consistency loop | **Candle core only.** MLX keeps it family-local (`M/mlx-gen-sana/src/pipeline.rs` `denoise_sprint*`, ~`:631`) **[H]** |
| Schedule resolution | `M/src/sampler.rs:904`,`:923`; `C/…/sampler.rs:282`,`:314` | native σ verbatim, or a curated scheduler over `mu` | Nearly every flow family. Third variant: `M/src/sampler/flow_sampling.rs` `ResolvedFlowSampling` (z-image MLX) **[H]** |
| Guidance library | `GC/guidance.rs:40` `GuidanceMethod`, `:128` `cfg`, `:145` `cfg_rescale`, `:170` `MomentumBuffer`, `:231` `normalized_guidance`, `:299` `apg_delta` | Combine operators over `GuidanceOps` | `cfg` used only by MLX SDXL (`M/mlx-gen-sdxl/src/pipeline.rs:947`,`:1133`,`:1263`). APG used by Bernini (`M/mlx-gen-bernini/src/guidance.rs:1-30` delegates). `cfg_rescale` used by Lens. Everyone else inlines **[H]** |
| Two-phase residency (Text → Heavy) | `GC/residency.rs:421` `run_request_scoped`, `:450` `_with_prepare` | Load text encoder → encode → materialize → drop → load heavy → render | flux, qwen, chroma, flux2, iris, krea, sdxl, qwen-2.1 and others (grep list) **[H]** |
| Three-phase residency (Text → Heavy → shed DiT → Decode view) | `run_staged_request_scoped` (gen-core `StagedHeavy`: `M/mlx-gen-z-image/src/model.rs:148-170`, `M/mlx-gen-flux3/src/model.rs:140-151`) | Adds a DiT drop before decode | z-image, sdxl, flux3, mage, anima, lens (MLX) and candle-flux3 **[H]**. Candle z-image instead uses its own `run_three_stage` (`C/candle-gen-z-image/src/pipeline.rs:406`) |
| SDXL-family stage skeleton | `M/mlx-gen-sdxl/src/stages.rs:28` `FamilyStages` {`build_init`, `denoise`}, `:50` `run_output`, `:82` `decode_output` | prepare_conditioning → build_init → denoise → decode | sdxl (`M/mlx-gen-sdxl/src/model.rs:1849`) and kolors (`M/mlx-gen-kolors/src/registry.rs:1049`) only. **This is the only existing typed stage model** **[H]** |
| img2img helpers | `M/src/img2img.rs:50` `init_time_step`, `:93` `add_noise_by_interpolation` | strength → start index; flow blend | ~10 MLX families. Candle **duplicates** `init_time_step` per family (`C/candle-gen-z-image/src/common.rs:85`, "mirrored from mlx-gen") **[H]** |
| Decode seam | `M/src/decoder.rs:29` `LatentDecoder` {`input_latent_space`, `decode`, `decode_tiled` `:41`}; `GC/latent.rs:144` `LatentSpace` | Native VAE / PiD overlay / tiling | Most image families **[H]** |
| PiD early-stop capture | `flow_capture_for_request` + `resolve_pid_decoder_at_sigma` (`M/mlx-gen-flux/src/model.rs:832-835`) | Truncates the schedule (`keep`) and decodes x_k with a student decoder | MLX: boogu, chroma, flux, flux2, krea, lens, qwen, z-image. `resolve_pid_decoder` in about 13 families per backend **[H]**. **This is the clearest existing example of a cross-cutting post-denoise feature, and it was implemented family by family** |
| Previews | `M/src/preview.rs` `PreviewCounter`; candle driver-owned `PreviewHook` (`C/…/sampler.rs:188-245` doc) | Per-step latent→RGB projection | About 20 families, wired in four different ways (see §4) **[H]** |
| Multi-phase request vocabulary | `GC/generator/request.rs:330` `phases: Option<Vec<GenerationPhase>>`; `GC/generator/params.rs:362` `GenerationPhase{steps, guidance, adapters}`; host math in `GCF/krea_multiphase.rs:1-40`, `:86` `resolve_phase_slices` | One global σ schedule, contiguous slices, per-phase CFG + adapter subset | **Krea only** (`M/mlx-gen-krea/src/pipeline.rs:1143` `for phase in phases`). The doc at `request.rs:327` says "Only the Krea MLX family reads it today; other models ignore it" **[H]** |
| Per-pass adapter strength | `GC/runtime/adapters.rs:18` `AdapterSpec::pass_scales` ("LTX-2.3 only"), `:25` `moe_expert` (Wan only) | Adapter strength per denoise pass / per MoE expert | LTX (`M/mlx-gen-ltx/src/transformer/linear.rs:308` `set_lora_pass`), Wan **[H]** |
| Family-neutral host crates | `GC3/` (flux3), `GCH3/` (h3), `GCL/` (ltx), `crates/contracts/gen-core-moss-image`, `GCF/` | Schedules, layouts, ids, policy shared by both backends of one family | Pattern from sc-27316 **[H]** |

**Takeaway [H]:** the solver/schedule axis is largely shared. Guidance is shared in name but mostly inlined. Residency
has two and a half shared variants. **Nothing shared owns the stage sequence, the latent geometry, or phase
boundaries.** Only SDXL/Kolors (`FamilyStages`) and Krea (`krea_multiphase`) have typed stage and phase
models, and each is local to its own family.

---

## 1. Per-family stage tables

Each table lists stages in order with `file:line`. **R** = resolution-dependent, **S** = per-step,
**P** = per-phase (changes at a phase/stage boundary). The "MLX vs candle" notes list divergences.

### 1.1 FLUX.1 (dev / schnell), with IP-Adapter (MLX) and ControlNet (candle) as the conditioning routes

**MLX** (`M/mlx-gen-flux/src/model.rs`, `pipeline.rs`, `transformer.rs`)

| Stage | Location | Notes |
|---|---|---|
| Validate + early cancel | `model.rs:487-500` | validate before encoding the reference image |
| Reference / IP encode | `model.rs:503-512` (`encoder.encode(image)`, `FluxIpInjector::new`); black-image negative embeds `:523-524` | Resolution-independent (CLIP 224²). Done **before** residency, outside the Text/Heavy phases |
| Residency phase A (T5 + CLIP) | `model.rs:615` `run_request_scoped`; encode `FluxTextOwned::encode` `:133`; materialize `:628-633` | Sequential stage residency drops the encoders before the DiT loads |
| Rungs (window / attention / decode tiling) | `model.rs:637-651`, `:839` `memory_strategy::decode_tiling(req)` | Per-request memory rungs |
| Schedule | `model.rs:815` `build_sigmas_with` → `pipeline.rs:35`; native `pipeline.rs:113` `build_linear_sigmas` (MLX `linspace`); `mu = linear_sigma_mu(w,h)` `pipeline.rs:18` | **R**: dev `mu` is linear in `w·h`. schnell is unshifted |
| PiD capture truncation | `model.rs:832` `flow_capture_for_request` → `keep` | Truncates the schedule |
| Noise / packing | `model.rs:856` `create_noise(seed,w,h)` → `pipeline.rs:53` draws **already packed** `[1,(h/16)(w/16),64]` with `random::key(seed)` | **R**. Packed token layout |
| Compile glue | `model.rs:851` `CompileGlueGuard::enable()` | MLX-only perf toggle |
| Denoise | `model.rs:860` `run_flow_sampler_with_latent_hook`, `TimestepConvention::Sigma`; preview in the hook `:866-877` | Shared driver |
| Predict | closure `:878-887` → `transformer.forward_injected_memory_windowed` (`transformer.rs:354`). **RoPE is rebuilt inside every forward** (`transformer.rs:448` `self.pos_embed.forward(...)`) | **R**, **S** (recomputed each step) |
| Guidance | Guidance-embed (distilled scalar `guidance` passed to the DiT). Optional true-CFG on a dev reference: `model.rs:748-768`, `neg + true_cfg·(pos − neg)`, gated on `t >= timestep_to_start_cfg` | **P**-like CFG start-step gate. `step_idx` is reconstructed from σ (`model.rs:885`, "the curated samplers expose no step index") |
| Unpack + decode | `model.rs:890` `unpack_latents` (`pipeline.rs:89`); `:891` `decode_latents_via_seam` (`:921`) with PiD/native and tiled/untiled | **R** |
| Output | `GenerationOutput::Images` | |

**Candle** (`C/candle-gen-flux/src/pipeline.rs`, `control_provider.rs`)

| Stage | Location | Notes |
|---|---|---|
| Text phase | `pipeline.rs:1171` `encode_residency`, `:1146` load | Same Text/Heavy split through `C/candle-gen/src/residency.rs` |
| Heavy load | `pipeline.rs:1342` `load_heavy_seq`, `:1373` `_with_memory` | Stock (BFL) and Packed (diffusers q4/q8) DiT variants as an enum |
| Per-image loop | `pipeline.rs:1430` `candle_gen::for_each_image_seed` | Shared candle helper (`C/candle-gen/src/seed.rs:101`). MLX has no equivalent |
| Noise | `pipeline.rs:1431` `flux1_load::seeded_noise` produces **NCHW**, then `State::new` (`:1439`) packs it (candle-transformers) | Different RNG and layout path from MLX |
| Schedule | `pipeline.rs:1441-1443` candle-transformers `get_schedule(steps, Some((seq_len, 0.5, 1.15)))`; curated `mu = flow_mu(variant, seq_len)` `:133` | **R**. MLX keys `mu` on `w·h/256`, candle on `seq_len` (the same number) |
| Conditioning prep | `pipeline.rs:744` `dit.prepare_conditioning(img, img_ids, txt, txt_ids)` **once per image** | Candle caches RoPE/ids. MLX recomputes per step (accidental divergence) |
| Denoise | `pipeline.rs:745` `candle_gen::run_flow_sampler(..., Some(preview), ...)` | Driver-owned preview |
| Decode | `pipeline.rs:1461` `bounded_host_decode` route (CPU host decode tile transfer `:1535`), else `decode`/`decode_packed` | Candle-only "bounded host" decode rung |
| Control route | `control_provider.rs:367` `encode_control_latent`: VAE-encode the hint and pack it with `State::new` (`:385`), **constant across steps**; ControlNet residual injection (`residual_interval` `:239`); denoise `:396` → `run_flow_sampler` `:424` | **R**: the control latent is aligned to the target grid |

**MLX vs candle divergences [H]:** RoPE caching (per-step vs once); noise layout and RNG; MLX has
the IP-Adapter and PuLID injector in-crate while candle splits them into `ip_provider.rs`, `ref_backbone.rs` and `candle-gen-pulid`;
candle has the bounded host decode; previews are wired differently (hook vs driver).

### 1.2 Qwen-Image (T2I + Edit)

**MLX T2I** (`M/mlx-gen-qwen-image/src/model.rs`, `pipeline.rs`)

| Stage | Location | Notes |
|---|---|---|
| Rungs | `model.rs:363-367` `resolve_request_rungs` (`pipeline.rs:79`) | |
| Run params | `model.rs:371` `resolve_run_params` (`pipeline.rs:125`): Lightning flag, steps, guidance (`true_cfg ∨ guidance`), seed, σ | Shared across T2I, Edit and Control ("F-117") |
| img2img start | `model.rs:374-377` `resolve_reference` + `init_time_step` | Lightning refuses img2img `:382-386` |
| Phase A encode | `model.rs:393` `residency.run`; `encode_prompt` (`pipeline.rs:174`, bf16); negative only when not Lightning | |
| Init latents | `model.rs:434` `encode_init_latents` (hoisted, seed-independent); per seed `create_noise` (`pipeline.rs:319`), blend `add_noise_by_interpolation` `model.rs:486` | **R** |
| Schedule | `pipeline.rs:427` `qwen_sigmas` with `mu = qwen_mu(w,h)` `:415` (**R**) and terminal rescale; Lightning `M/mlx-gen-qwen-image/src/sampler.rs:52` (static `ln 3`) | |
| PiD capture | `model.rs:443-454` | |
| Per-image loop | `model.rs:466` `decode_and_collect` (`pipeline.rs:200`) | Family-local loop helper |
| Denoise | `model.rs:490` → `pipeline.rs:531` `denoise_with_progress_windowed` → `run_flow_sampler` `:601`. Preview is emitted **inside predict** `:568` | Shared driver |
| Predict + guidance | pos/neg forwards `transformer.forward_windowed(latents, embeds, …, lh, lw, …)`; combine `compute_guided_noise` `pipeline.rs:462`: `neg + g(pos−neg)` **rescaled to the per-token L2 norm of pos** | This is `cfg_rescale` (`GC/guidance.rs:145`), re-implemented. RoPE is built in the forward (`transformer/transformer.rs:349`) **[M on caching]** |
| Decode | inside `decode_and_collect`: unpack, decode / tiled / PiD / alternate decoder | **R** |

**MLX Edit** (`M/mlx-gen-qwen-image/src/model_edit.rs`)

| Stage | Location | Notes |
|---|---|---|
| Phase A = VL encode | `model_edit.rs:330` `encode_phase_a`: preprocess the **first** reference, vision tower once, LM over pos/neg reusing the vision features (`:312` `encode_edit`, f16) | Text conditioning depends on the reference image. Vision runs at **reference** resolution |
| Reference latents | `model_edit.rs:469` `condition_resize_dims(last ref)`; `:481` `encode_reference_latents` per ref; concatenated on the sequence axis `:493` | Sized by the reference aspect, **not** by the output geometry. Static across steps |
| Denoise | `model_edit.rs:529` → `pipeline.rs:790` `denoise_edit_with_progress_windowed`: per step `concat([noise, refs], seq)` `:824`, forward with `cond_grids` (multi-grid RoPE), `slice_seq` back to the noise prefix (`:879`), norm-rescaled CFG | **S**: concat + slice every step |

**Candle** (`C/candle-gen-qwen-image/src/lib.rs`, `edit.rs`, `pipeline.rs`)

| Stage | Location | Notes |
|---|---|---|
| T2I | `lib.rs:509` `encode_phase`, `:534` `render_phase`, `:568` `denoise_and_decode`; schedule `:608` `resolve_flow_schedule(req.scheduler, mu, …)`; `:616` `for_each_image_seed`; `:627` `run_flow_sampler`; combine `:659` `pipeline::compute_guided_noise` (`pipeline.rs:191`, a candle copy of the same rescale) | **No img2img and no Lightning for T2I on candle** (`lib.rs:1076`: "img2img/Edit deferred") |
| Edit | `edit.rs:697` `denoise_and_decode`; **`resolve_flow_schedule(None, …)` at `:722` and `run_flow_sampler(None, …)` at `:749`** ignore the request scheduler and sampler; single seed `req.seed` (`:723`); joint concat `:760`, `narrow` back | Accidental divergence: MLX Edit honors `req.sampler`/`req.scheduler` through `resolve_run_params`. Candle Edit does not |
| Control | `pipeline.rs:86` `pack_fun_control_context` (control latent + mask + inpaint concatenated to 33 channels) | **R** |

### 1.3 Z-Image (Turbo + Base + Control)

**MLX Turbo** (`M/mlx-gen-z-image/src/model.rs`, `pipeline.rs`, `sampling.rs`)

| Stage | Location | Notes |
|---|---|---|
| img2img start | `model.rs:630` `init_time_step` | |
| Rungs | `model.rs:639-646` `resolve_request_rungs` (a different struct from Qwen's: adds `stage_residency`, `streamable`, `encoder_window`) | |
| Three-phase residency | `model.rs:647` `run_staged_request_scoped`: encode, then denoise (`:671-757`), then materialize latents, then **shed DiT**, then decode (`:758-765`) | |
| Text encode | `pipeline.rs:708` `encode_prompt` (windowed encoder `EncoderWindow` `:1030`) | |
| Schedule | `model.rs:695` `sampling::resolve_full` → `sampling.rs:18`: native `FlowMatchEuler::for_static_shift(steps, 3.0)` + `ResolvedFlowSampling::resolve(… mu=ln 3, OneMinusSigma)` | **Not R** (static shift). Third schedule-resolution API (`ResolvedFlowSampling`) |
| Init | `pipeline.rs:1105` `denoise_batch`: per seed `create_noise(..).as_dtype(Bfloat16)` `:1146`; img2img blend `:1155` | bf16 noise (flux/qwen use f32) |
| Conditioning prep | `pipeline.rs:138` `transformer.prepare(shape, cap_feats)`: patchify metadata + RoPE **once per loop** ("F-042") | **R** cached |
| Denoise | `pipeline.rs:175` `run_flow_sampler_with_latent_hook`, `OneMinusSigma` (`:183`); preview in the hook | Latent is unbatched **C,T,H,W** (comment at `pipeline.rs:154-156`) |
| Decode | `model.rs:760` `decode_batch` (`pipeline.rs:1191`): unpack, then `decode` / `decode_tiled` | |

**MLX Base:** the same transformer. `model_base.rs:1-23` lists the deltas: static shift 6.0, 50 steps, real CFG. CFG loop
`pipeline.rs:270` `denoise_cfg_with_progress_and_preview` prepares cond and uncond separately (`:289`,`:292`) and combines inline
`v_uncond + g(v_cond − v_uncond)` (`:309`). That is plain CFG inlined rather than `gen_core::guidance::cfg`. Uses `run_staged_request_scoped` (`model_base.rs:232`).

**Control:** `pipeline.rs:338-590` four more copies of the loop (`denoise_control_*`, `denoise_control_cfg_*`), each with its own
`run_flow_sampler_with_latent_hook` call (`:436`,`:575`). There are **six near-identical loop bodies** in one file. The constant control context comes from
`pipeline.rs:638` `encode_control_context` (**R**).

**Candle** (`C/candle-gen-z-image/src/pipeline.rs`, `base.rs`, `common.rs`)

| Stage | Location | Notes |
|---|---|---|
| Residency | `pipeline.rs:406` **local** `run_three_stage` (text, renderer, decoder with `device.synchronize()` before each drop) | Duplicates the gen-core staged residency |
| Schedule (Turbo) | `pipeline.rs:1334-1346`: candle-transformers `FlowMatchEulerDiscreteScheduler(z_image_turbo())` with `calculate_shift(image_seq_len, …)` mu; curated `resolve_flow_schedule(req.scheduler, **0.0**, …)` | MLX Turbo uses a static shift of 3.0 with `mu = ln 3`. **Possible numeric divergence** (see Uncertain) |
| Init | `pipeline.rs:1328` `common::seed_noise`; inline img2img blend `clean·(1−σ) + noise·σ` `:1350-1355` | Re-implements `add_noise_by_interpolation` |
| Conditioning prep | `prepare_inputs` (candle-transformers, pads to `SEQ_MULTI_OF` with a mask), then `transformer.prepare_conditioning(…)` once | |
| Denoise | `pipeline.rs:1367` `run_flow_sampler(..., OneMinusSigma, ...)`; the predict closure **negates** the output (`.neg()`) | MLX negates inside the model. Same math, different sign placement |
| Base | `pipeline.rs:1395` `denoise_base_sequential`, `:1424` static `base_scheduler_config()` (shift 6.0, `:327`) | |

### 1.4 SDXL (native ancestral default + curated + img2img / inpaint / IP / control)

**MLX** (`M/mlx-gen-sdxl/src/model.rs`, `pipeline.rs`, `stages.rs`, `sampler.rs`, `sampling.rs`)

| Stage | Location | Notes |
|---|---|---|
| Sampling resolution | `sampling.rs:164` `ResolvedSampling::resolve`: route ∈ {native ancestral, acceleration (LCM/Lightning/Hyper/TCD), curated} (`:25` `SamplingRoute`) | Family-local sampling recipe type |
| Residency | `model.rs:1307` `run_staged_request_scoped` | |
| Text | `pipeline.rs:53` `encode_conditioning` (dual CLIP L+G, pooled, long-prompt windows `:83`); `text_time_ids` `:38` | `time_ids` is a **hardcoded** `[512,512,0,0,512,512]` (`pipeline.rs:38-45`; candle `denoise.rs:48-58` documents it as a vendored quirk), so it is **not R** **[H]** |
| Stage 1 `prepare_conditioning` | `model.rs:1716`: ControlNet cond embed precomputed (`cn.embed_cond` `:1736`), CFG-batched control image, IP tokens (zeros uncond row), `SdxlMode::resolve` (Txt2Img / Img2Img / Inpaint / Ip / Control) | Control image is **R** |
| Init latents | `model.rs:1771` `build_init_latents`: tiled VAE encode of the init image, inpaint mask preprocess | **R** |
| Stage 2 `build_init` | `model.rs:1849` `FamilyStages` impl → `build_curated_init` `:1887` (k-diffusion σ-space, `noise·σ_max` or `x0 + ε·σ_start`) / `build_ancestral_init` `:1924` (global-RNG prior `sample_prior` `:1979`, img2img `add_noise`, inpaint fixed noise + `InpaintBlend::new` `:1954`) | Noise comes from the **global seeded MLX RNG stream** (`stages.rs:35-36` "caller has already seeded") |
| Stage 3 denoise (native) | `pipeline.rs:832` `denoise_core`: own loop `for i in 0..steps` `:881`; `scale_model_input` `:889`; preview with **synthetic σ = step index** (`:900-925`, `emit_nhwc_preview`); CFG via `guided_eps_pair` `:765` (batched rows `[text, neg]` or sequential halves with per-half `eval`) and then `gen_core::guidance::cfg` `:947`; `sampler.step` `:970`; inpaint blend `:978` | Bespoke loop. ε-prediction. Ancestral noise per step |
| Stage 3 denoise (curated) | `pipeline.rs:1012` `denoise_curated` → `run_curated_sampler_with_latent_hook` `:1097` over `DiscreteModelSampling`; CFG++ `:1231` | |
| Stage 4 decode | `stages.rs:82` `decode_output` → `pipeline.rs:1508` `decode_image_tiled` (VAE or PiD) | VP rescale for PiD capture `stages.rs:66` |
| Output loop | `model.rs:1494` `stages::run_output` per seed | |

**Candle** (`C/candle-gen-sdxl/src/denoise.rs`, `pipeline.rs`, `sampler.rs`)

| Stage | Location | Notes |
|---|---|---|
| Noise | `denoise.rs:64` `draw_noise(&mut StdRng…)`; `:73` `seeded_prior` | `rand::StdRng`, not the MLX RNG |
| Native loop | `denoise.rs:257` `denoise_ip_multi_control`: own loop over `(t, t_prev)` pairs; MultiControlNet residual sum; **CFG rows `[uncond, cond]`** (`:345-350`, inline `uncond + (cond−uncond)·cfg`); ancestral noise drawn every step (`:356`); `PreviewCounter::with_steps` `:289` | MLX row order is `[text, neg]` (`pipeline.rs:791`,`:826`). **Accidental inconsistency** |
| Curated | `denoise.rs:430` `denoise_curated` → `run_curated_sampler` `:449` | No CFG++ on candle |
| Edit / inpaint | `lib.rs:165-171` `SdxlEdit` (separate provider: `edit_provider.rs`) | MLX folds inpaint into the main `Sdxl` modes |

### 1.5 Wan 2.x (T2V / I2V / TI2V; A14B MoE)

**MLX** (`M/mlx-gen-wan/src/pipeline/*.rs`, `model/*.rs`, `scheduler.rs`)

| Stage | Location | Notes |
|---|---|---|
| Geometry | `pipeline/geometry.rs:7` `align_dim`, `:110` `latent_shape`, `:130` `seq_len` | **R** + frames |
| Memory preflight | `model/a14b_generate.rs:204-240` `preflight_denoise_memory_guard` | Fail fast before heavy load |
| Stage 1 text | `a14b_generate.rs:246` `encode_text_staged_for_tier_with_progress` (UMT5, pos + neg; encoder dropped, then `clear_cache` `:262`) | |
| Noise | `a14b_generate.rs:269` `random::normal(lat=[z,t,h,w], key)` (no batch axis) | **R** |
| Stage 1b I2V | `a14b_generate.rs:287` `build_i2v_y` → `pipeline/conditioning.rs:89`: VAE-encode the first frame + mask into `y [20,F,H,W]`, **channel-concatenated to every forward** (`pipeline/denoise.rs:106`) | **R**, constant across steps and experts |
| TI2V | `model/ti2v_generate.rs:211-241` encode the keyframe into `z_img`, then mask; `pipeline/conditioning.rs:211` `ti2v_blend_init`; per-step re-blend in `denoise_ti2v` (`pipeline/denoise.rs:381`) | **S** mask pin |
| Step cache | `pipeline/denoise.rs:16` `StepCache` {cross-KV per block, RoPE cos/sin, batch}; `:37` `build_cache` (`prepare_cross_kv` `:48`, `prepare_rope(grid)` `:49`), built **once per expert range** | RoPE is **R**. Cross-KV is per expert (**P**) |
| Schedule / solver (native) | `scheduler.rs:45` `WanScheduler` trait; `:73` `make_scheduler`; family-local `FlowMatchEuler` `:121`, `FlowDpmpp2m` `:180`, `FlowUniPC` `:280`; integer timesteps `σ·num_train` `:108` | **A family-local solver library parallel to gen-core's solvers** |
| MoE phases | `pipeline/moe.rs:26` `crossing_index`; `:45` `denoise_range` (one expert over a step range, one continuous scheduler); `:86` `denoise_moe` (high `0..k`, then low `k..`); `:289` `staged_expert_swap` (load, use, drop, evict); `:8` `Expert{transformer, ctx_cond, ctx_uncond, guidance}` | **P**: model, contexts and guidance swap together at the boundary |
| Curated | `pipeline/denoise.rs:304` `denoise_curated`; `moe.rs:146` `denoise_moe_curated` (expert chosen **per eval** inside predict), `:204` `_swapped` | |
| CFG | batched B=2 (`[cond, uncond]`), `cfg_combine` `pipeline/denoise.rs:6` inline | |
| Approximation (feature cache / token pruning) | `pipeline/denoise.rs:95` `predict_approx(trunk, prune)`; `:192` `refuse_unwired_approximation` refuses on TI2V, curated and MoE routes | Existing cross-cutting feature that **must be refused per route**. A design precedent |
| Decode | `a14b_generate.rs:451-464` `Progress::Decoding`, `decode_to_frames(vae, latents, tiling)` (`pipeline/decode.rs:9`; vae16 / vae22 tiling planners) | **R** + temporal |

**Candle** (`C/candle-gen-wan/src/wan14b.rs`, `scheduler.rs`, `pipeline.rs`, `vace.rs`)

- Separate solver copy: `scheduler.rs:98` `FlowScheduler::new`, `:138` `dpmpp_2m_step`, `:169` `unipc_step` **[H]**.
- MoE copies: `wan14b.rs:937` `crossing_index` (scheduler-based), `:967` `staged_expert_swap`, `:540` `denoise_range`; and **another** `crossing_index` at `vace.rs:644` **[H]**.
- CFG `pipeline.rs:85` `cfg` (yet another inline copy).

**Divergences:** the same structure, duplicated three ways (MLX, candle wan14b, candle vace) plus Bernini (§1.12).

### 1.6 LTX-2.x (2-stage audio+video; 2.5 dev/distilled; IC-LoRA; DFR)

**MLX** (`M/mlx-gen-ltx/src/pipeline.rs`, `model/*.rs`, `dev_sampler.rs`, `conditioning.rs`, `upsampler.rs`)

| Stage | Location | Notes |
|---|---|---|
| Prompt enhancement | `model/generate.rs:456-478`: Gemma TE plus an optional enhancer LLM rewrites the prompt; fallback reason surfaced | **Pre-encode LLM stage** |
| Text / connector | Gemma TE, then `connector.rs` into the video and audio contexts (and negatives for dev) | |
| Geometry | `model/staging.rs:240` `latent_dims` → `(frames, h/2/32, w/2/32, h/32, w/32)`: **the stage-1 grid is half resolution** | **R**, **P** |
| Positions | `model/staging.rs:280` `position_grids` (pos1 at half-res, pos2 at full-res, audio) | **R**, **P** |
| Noise | caller pre-draws `video_s1`, `video_s2`, `audio_s1`, `audio_s2` (`model/generate.rs:178-190` signature) | One noise per stage per modality |
| Heavy build | `model/generate.rs:295` `build_transformer(req)` **after** the text phase is freed; dropped `:394` before decode | Hand-rolled residency (no gen-core `Residency`) |
| Stage-1 conditioning | `pipeline.rs:994` `stage_keyframe_state` (replace-latent I2V/keyframes, `I2vConditioning`), `conditioning.rs:283` `token_timesteps` (**per-token σ·mask**), `:862` `apply_denoise_mask` | **S**: per-token timesteps + post-step pin |
| Stage-1 denoise | `pipeline.rs:1504` `dit.set_lora_pass(0)`; branch on variant: dev → `denoise_av_dev` (`:793`, four branches per step via `dev_sampler.rs:178` `ExecutionPlan::execute`: Uncond / Cond / STG-perturbed / IsolatedModality, combined by `dev_sampler.rs:94` `combine_guidance` = CFG + STG + modality + std-rescale); distilled → `denoise_av` (`:722`, legacy Euler `:80`) or `denoise_av_curated` (`:914`) | **P**: dev guidance applies only in stage 1 |
| RoPE | `pipeline.rs:120`,`:745`,… `dit.next_rope_epoch()` (one memo epoch per stage) | **R** cache keyed per stage |
| Stage transition | `pipeline.rs:1578` `upsample_latents(v, upsampler, mean, std)` (**learned 2× latent upsampler**, `upsampler.rs:450`); `stage_keyframe_state(v, keyframes, false)` re-pins keyframes at the stage-2 resolution; `renoise(v, s2_noise, STAGE2_SIGMAS[0])` (`:157`); audio renoised but **never upsampled** (`:1588`); `set_lora_pass(1)` `:1589` | **The native "low-res first, then refine" implementation** |
| Stage-2 denoise | `STAGE2_SIGMAS` (`:56`) = the tail of the stage-1 table; curated stage-2 seed `seed.wrapping_add(0x5DEE_CE66)` `:1603` | |
| Progress | `model/generate.rs:281` `StageProgressFold` folds the stages into one monotone bar | |
| Decode | `model/generate.rs:397-406` video VAE decode (tiling planner `pipeline.rs:302-404`) + `decode_audio_track` (audio VAE + vocoder) | |

**Duplication inside MLX LTX [H]:** the stage-boundary choreography (`set_lora_pass(0)`, then denoise, then upsample, then
renoise, then `set_lora_pass(1)`, then denoise) appears three times: `generate_t2v_latents` (`:579`,`:592`),
`generate_av_latents_for_variant` (`:1504`–`:1589`) and `generate_av_latents_iclora` (`:1707`–`:1759`).

**Candle** (`C/candle-gen-ltx/src/pipeline.rs`)

- `TwoStageGeometry` (`:31`), `two_stage_geometry` `:195` with `h1 = h2/2` (latent halving) **[H]**.
- `TwoStageOrchestration` typestate (`:59-190`): phases `Start → Stage1 → Upsampled → Stage2`, `stage1_setup` (adapter pass 0), `learned_upsample`, `stage2_renoise` (adapter pass 1), `stage2_keyframes`, `stage2_forward`. It refuses out-of-order calls **[H]**. **This is the closest thing in the repo to a typed phase-transition contract.**
- Seeds: `video_stage2 = seed+1`, audio `+2/+3` (`:80-85`). MLX instead pre-draws the noises and uses `seed ^ 0x5DEE_CE66` for the curated stage 2. The derivations differ (accidental, but may be parity-pinned).
- `StageProgressFold::new(offset, stage_steps, total)` `:271` has a different signature from the MLX one (`new(total)`).

### 1.7 MiniMax-H3 (joint T2VA / FL2VA / Ref2VA)

**MLX** (`M/mlx-gen-minimax-h3/src/denoise/*.rs`, `model/generate.rs`, `pipeline.rs`)

| Stage | Location | Notes |
|---|---|---|
| Task resolve | `model/generate.rs:58` `MiniMaxH3Task::resolve(keyframes, references)` | |
| Keyframe fit | `model/generate.rs:76-85`: anchors, then images fitted once (shared by the vision tower and the VAE encode) | **R** |
| Text | `model/generate.rs:96-105` `Loading(TextEncoder)`, `encode_prompt` or `encode_prompt_grounded(refs)` | Grounded (reference-image) prompt encode |
| Keyframe VAE encode | `model/generate.rs:112-140` (before the DiT is mapped) | **R** |
| Layout | `:155` `t2va_layout` / `:157` `fl2va_layout` (`pipeline.rs:695`,`:729`) → `denoise/packing.rs:137` `PackedLayout::build`: **one packed sequence of text, video and audio rows**, with conditioning rows prepended, row classes, tags and position ids | **R**: rows per frame and seq_len depend on the latent grid |
| Noise | `:159` `initial_latents(geometry, patch, seed)` (`pipeline.rs:559`) | |
| Schedule | `denoise/schedule.rs`: **two σ schedules** (video shift 12.0, audio shift 3.0) in `JointSchedule`; `adaln_schedule` `denoise/mod.rs:136` precomputes the AdaLN modulation table for **all** steps × row classes | AdaLN table: **S** precomputed |
| Denoise | `denoise/mod.rs:198` `denoise_av` / `:216` `_with_observer`: one forward per step (`JointVelocity` trait `:121`); `step_generated_tail` `:154` steps only the generated rows (conditioning rows survive); per-step `eval` | Bespoke loop. **No CFG**, pinned by test (`mod.rs:29-41`) |
| Decode | `model/generate.rs:257-262` `Decoding`, then `decode_video` (`:294`, `spatial_tiling`) and `decode_audio` (`:312`, separate audio VAE) | |
| Sparse attention | Veda (`GCH3/veda.rs`) | Attention-plan variance |

**Candle:** mirrors the structure. `C/candle-gen-minimax-h3/src/denoise/mod.rs:109` duplicates the `JointVelocity` trait, `:188` `denoise_av` **[H]**.
Packing duplicated in `C/…/denoise/packing.rs`. The tensor-free half is shared via `GCH3/`.

### 1.8 FLUX 3 Image Klein (flow-map distilled + base CFG)

**MLX** (`M/mlx-gen-flux3/src/pipeline.rs`, `model.rs`; host math in `GC3/lib.rs`)

| Stage | Location | Notes |
|---|---|---|
| Prompt upsampler (PUP) | `M/mlx-gen-flux3/src/pup.rs:37` `Flux3PromptUpsampler`; `:156` `effective_prompt` | LLM pre-encode stage (same idea as the LTX enhancer, different implementation) |
| Residency | `model.rs:140` `StagedHeavy` impl, so `run_staged_request_scoped` | |
| Text | `model.rs:657` `te.encode(prompt)`, negative when guidance is set (base) | Qwen3-VL, 8 stacked hidden states |
| References | `model.rs:675` `references::encode(ae, …)` (`references.rs:13`) as image-conditioning tokens + ids | Token concatenation (`SampleTokens.image_cond`) |
| Schedule | `GC3/lib.rs:1073` `flow_map_schedule(steps, alpha)` = `α·t/(1+(α−1)t)` over `linspace(1,0)` | Static, **not R** |
| Noise | `pipeline.rs:18` `noise()` packed `[h·w, C]` rounded through bf16 | |
| Denoise (distilled) | `pipeline.rs:49` `denoise_flow_map_with_references`: `x += dt·dit(x, t, target_t)` (**flow-map: the model is conditioned on the target time** `target_timestep: Some(schedule[step+1])`) | Bespoke loop. Not the curated driver (a flow map is not a velocity field at a single t) |
| Denoise (base) | `pipeline.rs:123` `denoise_cfg_with_references`: batched `[neg, text]` rows, inline CFG with dtype rounding | |
| Ids / RoPE | `pipeline.rs:179` `image_ids(h, w, image_rope_scale)` | **R** (scaled ids) |
| Decode | `model.rs:713` `decode_tokens` (`pipeline.rs:283`) | |

**Candle:** `C/candle-gen-flux3/src/pipeline.rs` mirrors this, sharing `GC3/` (not walked line by line **[M]**).

### 1.9 Mage-Flow (bespoke)

**MLX** (`M/mlx-gen-mage/src/pipeline.rs`, `latent.rs`, `vae/*`)

| Stage | Location | Notes |
|---|---|---|
| Encode phase | `pipeline.rs:148` `encode_generation_phase` | Qwen3-VL final hidden state |
| Schedule | `pipeline.rs:1334` `mage_flow_sigmas(steps)`: ATen-exact linspace plus static shift | Not R |
| Noise | `pipeline.rs:1429` `initial_tokens(h,w,seed,gs_key,…)` → `latent.rs:725` `encode_noise_host`: **Gaussian-Shading watermark noise** (the message is embedded in the initial noise) | Noise is not i.i.d. N(0,1). **R** (bit positions keyed to the token count) |
| Batch packing | `pipeline.rs:208` `denoise_generation_phase`: several samples (different grids) **concatenated into one sequence** (`concatenate_axis(refs,1)`), `generation_layout` `:1701` (segment-aware) | Multi-sample sequence packing |
| Denoise | `pipeline.rs:1556` `denoise_capture`: own loop; **CFG is fused on the sequence axis** (`[img; img]` with a fused layout, then split), optional `renormalize`; `flow_euler_step` in f32 | Bespoke. No curated solvers |
| Decode | `pipeline.rs:308` `decode_generation_phase`: the Mage-VAE decoder is **itself a pixel denoiser** (NeRF head; `vae/denoiser.rs`); tiling reads `GenerationMemory` fields **inline** (`decode_with_memory`, ~`:1665`) | |

### 1.10 SenseNova-U1 (unified AR + flow, pixel space)

**MLX** (`M/mlx-gen-sensenova/src/t2i.rs`, `fm.rs`, `runtime.rs`)

| Stage | Location | Notes |
|---|---|---|
| Optional think rollout | `runtime.rs:305` `generate_think` (text AR with a KV cache) | AR text before the image |
| Prefix prefill | `t2i.rs:739` `prefill_ids` → `KvCache` for cond / uncond / img-cond | **KV cache of the prompt prefix reused every step** |
| Schedule | `fm.rs:125` `apply_time_schedule(t, shift)`; `T2iOptions.timestep_shift` (`t2i.rs:274`) | |
| Noise | SplitMix64 Box–Muller (`t2i.rs:41-43`) | Own RNG |
| Denoise | own loop. Gen-path forward over the image patches attending to the cached prefix; **x-prediction** `x_pred`, then `fm.rs:140` `velocity(x_pred, z, t, t_eps)`, then `fm.rs:135` `euler_step` | Pixel-space patches (`fm.rs:147` `patchify`) |
| Guidance | `t2i.rs:253` `CfgNorm` {None, Global, Channel (= cfg_rescale), CfgZeroStar}; `T2iOptions.cfg_interval` `:272` (**guidance only inside a t-interval**); dual text + image CFG (`img_cfg_scale`, three branches; `t2i.rs:1827-1951`) | **P**-like guidance interval |
| Decode | `fm_head` + `unpatchify` (`fm.rs:182`). **No VAE** | |
| Output | images, or interleaved text + image (`t2i.rs:329` `InterleaveOutput`) | |

Candle: `C/candle-gen-sensenova/` mirrors it (it has `preview.rs`, which MLX lacks) **[H for file presence]**.

### 1.11 Boogu (Base true-CFG + Turbo DMD)

**MLX** (`M/mlx-gen-boogu/src/pipeline.rs`)

| Stage | Location | Notes |
|---|---|---|
| Encode | `pipeline.rs:348` `encode_base_preflight` / `:366` `encode_turbo_preflight` / `:381` `encode_edit_preflight` (Qwen3-VL, optional vision tower) | |
| Base schedule | `pipeline.rs:1249` `base_shift_mu()` = `lin_mu(4096)` = 1.15 constant (saturated) | Not R |
| Base denoise | `pipeline.rs:630` `render_base_t2i`: `init_noise` `:1146`, `run_flow_sampler(OneMinusSigma)`, `base_velocity` `:554`: CFG written as `cond + (s−1)(cond−uncond)`, then **negated** (`× −1`) | Sign flip done in the family |
| Turbo (DMD) native | `pipeline.rs:698` `render_turbo_t2i`; `:851` `denoise_turbo_native`: **predict x0 (`x += (1−σ)v`), then renoise with fresh noise** `init_noise(h,w,seed,step+1)` toward the next σ, where "σ" is an **ascending clean fraction** (`dmd_sigmas` `:1198`) | Bespoke loop. **The fresh noise is drawn per step at the current geometry** |
| Turbo curated | `pipeline.rs:705-725`: `run_flow_sampler` over `turbo_native_sigmas`, sampler defaults to `lcm` | |
| Edit | `pipeline.rs:581` `edit_velocity` → `forward_edit_multi(x, ref_latents, …)` (refs prepended; reference geometry is its own) | |
| Decode | `pipeline.rs:902` `decode_latents` (no tiling path) | |

Candle: `C/candle-gen-boogu/src/pipeline.rs` (it has `preview.rs`; MLX does not).

### 1.12 SANA / SANA-Sprint (SCM)

**MLX** (`M/mlx-gen-sana/src/pipeline.rs`, `scm.rs`)

- Base: `pipeline.rs:173` `denoise_cfg_with_memory` → `run_flow_sampler_with_latent_hook(Sigma)` (`:256`), but the trunk embeds **σ·1000** (timestep scaling inside the predict closure, ~`:214`). CFG is inline (`:219-245`) **[H]**.
- Sprint: `scm.rs:1-30` explains why SCM is not the unified sampler. It works in angle space `t∈[0,π/2]`, `scm_timestep = sin/(cos+sin)` (`scm.rs` `scm_timestep`), uses input scaling, `x0 = cos(s)x − sin(s)·out`, renoise `cos(t')x0 + sin(t')·noise·σ_data`, and **guidance is an embedded scalar** (no CFG pair). The loop `denoise_sprint*` is family-local (~`:631`) **[H]**. Candle promoted the same loop to `C/candle-gen/src/sampler.rs:624` `run_scm_sampler` **[H]**.
- DC-AE 32× decoder with its own decode / encode tile plans (`pipeline.rs:388-520`).

### 1.13 Bernini (MAR planner, then Wan-A14B renderer)

**MLX** (`M/mlx-gen-bernini/src/*.rs`)

| Stage | Location | Notes |
|---|---|---|
| Planner (MAR) | `mar.rs` header: `post_process_input_embeds` (all gen slots masked), `mar_schedule` (MaskGIT cosine reveal, seeded order), `sample_vit_embed` (Qwen2.5-VL backbone over three streams → `clip_diff.sample` per revealed token with **triple CFG** → scatter) | **Masked-autoregressive loop of per-token diffusion** (`clip_diff.rs:42` own `FlowMatchScheduler`, `:280` `DiffLossFm`) |
| Handoff | `mar.rs` `feat_to_renderer`, `four_streams` (`wtxt_wvit`, `wtxt_wovit`, `wotxt_wvit`, `wotxt_wovit`) | Four conditioning streams |
| Renderer | `pipeline.rs:212` `denoise_bernini`: reuses `mlx_gen_wan` `make_scheduler(UniPC)` (`:230`), transformer and VAE, but **re-implements the MoE switch inline** (`t >= boundary`) and rescales all omegas once on the first low-noise step (`:256-258`) | **P** guidance rescale at the expert switch |
| Packed conditioning | `forward.rs:66` `PackedForward`: source media patch-embedded and concatenated on the token axis with a source-id rotary (`rope.rs`) | Token concatenation |
| Guidance | `forward.rs:333` `guided_velocity`: 7 modes with chain / APG over momentum buffers (`MomentumBuffer` per term, persisting across steps); x-space APG via `GC/guidance.rs` (`guidance.rs:1-30`) | **Stateful guidance** |
| Progress | `pipeline.rs` ~`:250` `on_step(i+1)` fires **before** the step's compute (Wan's `denoise_range` fires after: `moe.rs:62-75`) | Accidental |

### 1.14 Krea (reference point for multi-phase; not in the requested list)

`M/mlx-gen-krea/src/pipeline.rs:1143` `for phase in phases`, one `run_flow_sampler` per slice over the shared schedule. The DiT is cloned per
phase with the phase's adapter subset (`GCF/krea_multiphase.rs:37-40`). Candle's `Krea2Transformer` is not `Clone`
(a Mutex-backed RoPE cache), so candle uses a different adapter-switch mechanism (same doc) **[H]**.

---

## 2. Unified STAGE MODEL proposal

### 2.1 Design principles derived from the walk

1. **The request is compiled to a Plan before any weights move.** Every family already does an ad-hoc version: `resolve_run_params` (qwen), `ResolvedSampling` (sdxl), `ResolvedFlowSampling` (z-image), `ExecutionPlan` (ltx dev), `T2iOptions` (sensenova), `TurboOptions` (boogu), `TwoStageGeometry` (candle ltx), `JointGeometry` (h3). A shared `RenderPlan` should hold *every* decision a cross-cutting feature might rewrite, as data.
2. **Phases are first-class and continuous.** Wan MoE, LTX 2-stage, Krea phases, Bernini omega rescale, PuLID `timestep_to_start_cfg`, and SenseNova `cfg_interval` are all "a contiguous slice of one trajectory with its own {geometry, model, adapters, guidance, conditioning}". Krea's invariant ("ONE global schedule, contiguous slices", `GCF/krea_multiphase.rs:10-20`) is right **except** for LTX, which deliberately renoises to `STAGE2_SIGMAS[0]` (a reset to a σ that already occurred in stage 1). So the phase transition must allow "continue at σ_k" **and** "renoise to σ_j".
3. **Geometry-dependent state is derived, keyed and invalidated at phase boundaries.** RoPE/ids (flux, qwen, z-image, wan, ltx), packed layouts (h3, mage), aligned conditioning latents (control, init, I2V `y`, TI2V mask, inpaint mask, keyframes), `mu` (flux-dev, qwen) and preview geometry (SDXL `time_ids` would belong here, but it is hardcoded today). Today these are built once before the loop under the implicit assumption that geometry is loop-constant (z-image `transformer.prepare` `pipeline.rs:138`, wan `build_cache` `denoise.rs:37`, candle flux `prepare_conditioning` `pipeline.rs:744`).
4. **The solver driver stays the inner loop. The phase runner is the outer loop.** The curated driver already expresses one phase (a σ slice from a latent). Multi-phase = repeated driver calls with a transition in between, exactly as Krea does.

### 2.2 Stages and the data passed between them

```
S0 Plan      Request ─► RenderPlan { geometry_by_phase, phases[], schedule_recipe, guidance_plan,
                                     conditioning_requirements, decode_plan, memory_rungs, output_spec }
S1 PromptPrep  RenderPlan ─► EffectivePrompt          (LTX enhancer, FLUX3 PUP, SenseNova think)
S2 Encode    [Residency: Text]  EffectivePrompt (+ref images for VL encoders) ─► CondBundle
             CondBundle = { streams: Map<StreamId, {embeds, mask, pooled?}>, kv_prefix?: KvCache,
                            vision_feats? }                                   (geometry-independent*)
S3 Reference [Residency: Heavy or Encoder]  images/video/audio refs ─► RefBundle
             RefBundle = { geometry_free: IP embeds, VL refs, edit refs at own size,
                           grid_aligned: Fn(Geometry) -> {control ctx, init latent, mask, i2v y,
                                                          keyframes, fl2va rows} }
S4 Schedule  RenderPlan ─► Schedule { points[], convention, mu, slices: PhaseSlice[],
                                      truncation: {start (strength), keep (PiD capture)} }
S5 Init      Geometry(phase0), Schedule, RefBundle, seed ─► LatentState
             LatentState = { tensors: Vec<Latent> (video/audio streams), space: LatentSpace+Geometry,
                             cond_rows/mask/per-token-σ state, rng_cursor }
─── per phase p ─────────────────────────────────────────────────────────────────────────────
S6 PhaseEnter  (LatentState, Phase p) ─► (LatentState', StepContext_p)
             model/expert residency swap, adapter selection, guidance params,
             geometry change: resample latent, renoise, rebuild grid-aligned refs,
             StepContext_p = { rope/ids, cross-KV, adaln table, packed layout, attn masks,
                               control residual ctx, cfg batching }   (keyed on geometry+model+cond)
S7 Denoise   Driver(solver) × Predictor(model fwd ⊕ guidance combine) over slice p
             per-step hooks: pre_eval(x,σ), post_combine(v), post_step(x,σ) [pin/blend/renoise],
             observe(x,σ) [preview/progress]
S8 PhaseExit   LatentState ─► LatentState   (solver-history policy, state for next phase)
──────────────────────────────────────────────────────────────────────────────────────────────
S9 PostLatent  LatentState ─► DecodeInput   (unpack, PiD capture rescale, keyframe slot extraction,
                                             drop conditioning rows, DiT shed)
S10 Decode   [Residency: Decode view]  DecodeInput ─► Pixels/Waveform  (VAE/PiD/DC-AE/pixel head/
                                             NeRF decoder/audio VAE+vocoder; tiling)
S11 Output   Pixels ─► GenerationOutput { Images | ImagesRgba | Video{frames,fps,audio} | Audio }
```

\*S2 is geometry-independent for every walked family except SenseNova (the prefix KV is fine, but think-mode output can change the request),
Qwen-Edit (vision tokens at reference resolution, which is still independent of the *output* geometry), and H3 grounded refs.

### 2.3 Hook points per stage

| Stage | before | replace | after | observe | Who needs it |
|---|---|---|---|---|---|
| S0 Plan | — | `rewrite_plan(&mut RenderPlan)` | `validate_plan` (a capability check per family per feature, like Wan's `refuse_unwired_approximation` `denoise.rs:192`) | — | low-res-first (insert a phase with a smaller geometry), per-phase LoRA (phases from request), memory rungs |
| S1 PromptPrep | gate | provide an enhancer | — | log effective prompt | LTX, FLUX3, SenseNova |
| S2 Encode | — | — | `materialize` (exists: `run_request_scoped` 3rd closure) | conditioning memory phase | all |
| S3 Reference | — | per-kind encoder | — | — | Must expose **grid-aligned refs as a function of geometry** for low-res-first |
| S4 Schedule | — | `native(steps, geometry)`, curated scheduler | `slice(phases)`, `truncate(start, keep)` | — | PiD capture (exists), img2img strength, phases |
| S5 Init | — | noise source (iid / watermark / bf16-rounded / global RNG) | img2img blend, inpaint fixed noise | initial latent | Mage watermark, SDXL RNG |
| S6 PhaseEnter | — | **`transition(latent, from_geom, to_geom)`**: nearest / bilinear latent resample, learned upsampler (LTX), decode-encode | renoise policy, `rebuild_step_context`, `select_adapters`, `select_guidance`, `swap_model` | — | low-res-first, LTX stage 2, Wan MoE, Krea, Bernini omega rescale |
| S7 Denoise (per eval) | `pre_eval(x_in, σ)` | `predict` (family) | `combine(cond, uncond, …)` (guidance library), `post_step(x, σ)` (TI2V/I2V pin, inpaint blend, H3 cond-row freeze, DMD renoise) | `observe(x, σ, step)` for preview, progress, PiD capture | previews, PiD, approximations (feature cache), per-step guidance schedule |
| S8 PhaseExit | — | — | solver-history reset or carry | — | multistep solvers across geometry change |
| S9 PostLatent | — | — | unpack, VP rescale (`M/mlx-gen-sdxl/src/stages.rs:66`) | — | PiD |
| S10 Decode | DiT shed (exists: `StagedHeavy::shed_dit`) | decoder override (`LatentDecoder`, exists) | — | decode progress | PiD, tiling rungs |
| S11 Output | — | — | alpha plane, trim, audio fit (`M/mlx-gen-minimax-h3/src/pipeline.rs:475`) | — | — |

### 2.4 Family-provided "Predictor" contract (what stays per family)

From the walk, the irreducible per-family piece is:

```text
trait FamilyModel {
  type Latent;                         // Array | AvLatents | (rows + layout)
  fn latent_space(&self, g: &Geometry) -> LatentSpace;   // packed / NCHW / NCTHW / rows
  fn prepare_step_context(&self, g: &Geometry, cond: &CondBundle, refs: &AlignedRefs,
                          phase: &PhaseModelState) -> StepContext;
  fn forward(&self, ctx: &StepContext, x: &Latent, t: TimeInput, branch: Branch) -> Output;
     // TimeInput = {sigma} | {sigma, target_sigma} (flow-map) | {per_token: Array} (LTX)
     //            | {row_timesteps} (H3) | {angle} (SCM)
     // Branch = Cond | Uncond(stream) | Perturbed(STG blocks) | IsolatedModality | ...
  fn prediction(&self) -> ModelSampling; // flow v (Sigma / OneMinusSigma, sign) | eps | v | x0 | flow-map
}
```

Everything else (loop, CFG combine, schedule, phases, previews, PiD, residency) moves into shared code. Families with
bespoke loops keep a `FamilyDriver` escape hatch (§5.3).

---

## 3. VARIANCE CATALOG

"E" = essential (the domain requires it). "A" = accidental (the same concept, implemented differently for no domain reason). "E/A" = essential
math with an accidental implementation split.

### 3.1 Latent layout and space

| Axis | Values | Families (evidence) | E/A |
|---|---|---|---|
| Layout fed to the denoiser | **Packed tokens** `[1, N, C·p²]` | flux (`M/mlx-gen-flux/src/pipeline.rs:53`,`:71`), qwen (`M/mlx-gen-qwen-image/src/pipeline.rs:319`,`:340`), flux3 (`pipeline.rs:18`), mage (`initial_tokens`), boogu, sensenova (pixel patches) | E |
| | **NCHW** | sdxl (`M/mlx-gen-sdxl/src/model.rs` `latent_shape [i32;4]`), sana | E |
| | **C,T,H,W unbatched** | z-image MLX (`pipeline.rs:154-156`), wan (`a14b_generate.rs:269`) | E (wan); A (z-image image model uses T=1 to share a video-shaped VAE) |
| | **NCFHW (B,C,F,H,W) + packed tokens in the forward** | ltx (`pipeline.rs:103-140`) | E |
| | **Joint packed rows (text + video + audio)** | h3 (`denoise/packing.rs:137`) | E |
| | **Multi-sample packed sequence** | mage (`pipeline.rs:208-240`) | E/A (a perf choice that bakes the batch into the layout) |
| Who packs | in noise creation (MLX flux) vs after NCHW noise (candle flux `State::new`) | `M/mlx-gen-flux/src/pipeline.rs:53` vs `C/candle-gen-flux/src/pipeline.rs:1431-1439` | A |
| Latent space descriptor | `LatentSpace{channels, compression, patch_layout, temporal_law, normalization}` (`GC/latent.rs:144`) exists. Families pass `denoiser_output_latent_space` to the decode seam (`M/mlx-gen-flux/src/model.rs:893`) | about 13 image families | Shared vocabulary is **already there**. Use it as the geometry key |
| Pixel space (no VAE) | SenseNova (`fm.rs:147` patchify RGB) | E |
| Decoder is itself generative | Mage NeRF pixel denoiser (`vae/denoiser.rs`), PiD student (overlay) | E |

### 3.2 Prediction type and time input

| Axis | Values | Families | E/A |
|---|---|---|---|
| Prediction | flow velocity | flux, qwen, z-image, boogu, wan, ltx (velocity → `to_denoised`), h3, sana base, mage | E |
| | ε | sdxl (`DiscreteModelSampling`), kolors | E |
| | v (EDM) | svd | E |
| | **x0 / x-pred** | sensenova (`fm.rs:140` `velocity(x_pred,…)`), DMD student (boogu turbo, used as x0: `pipeline.rs:870-877`) | E |
| | **flow map Φ(x, t → t_ref)** | flux3 distilled (`pipeline.rs:83` `target_timestep: Some(target)`) | E |
| | **consistency (TrigFlow)** | sana sprint (`scm.rs:1-30`) | E |
| Time convention | σ (raw) | flux, qwen, sana (σ·1000 in closure) | E/A (the scale belongs in the convention, not the closure) |
| | 1−σ | z-image, boogu | E |
| | integer `trunc(σ·1000)` | wan (`scheduler.rs:108`) | E |
| | per-token σ·mask | ltx I2V/keyframes (`conditioning.rs:283`) | E |
| | per-row-class timesteps (two schedules) | h3 (`packing.rs:696` `row_timesteps`) | E |
| | angle → `scm_t` | sana sprint | E |
| | ascending clean fraction | boogu DMD (`dmd_sigmas` `:1198`) | A (the same σ grid re-expressed) |
| Output sign | negated in family closure | z-image candle `.neg()` (`C/candle-gen-z-image/src/pipeline.rs:~1390`), boogu `× −1` (`M/mlx-gen-boogu/src/pipeline.rs:574-577`), h3 "reversed velocity sign" (`denoise/mod.rs:20` table) | E/A (the sign is essential; where it lives is not) |

### 3.3 Schedule

| Axis | Values | Families | E/A |
|---|---|---|---|
| Shift | resolution-dependent `mu(w·h)` | flux-dev (`pipeline.rs:18`), qwen (`pipeline.rs:415`), candle z-image turbo (`calculate_shift`, **[L]**) | E |
| | static | z-image (3.0 / 6.0), wan (`shift` arg), boogu (saturated 1.15), flux3 (α), mage, h3 (12 / 3) | E |
| | baked tables | ltx `STAGE1/2_SIGMAS` (`pipeline.rs:51-56`), qwen Lightning (`sampler.rs:52`), DMD grid | E |
| Terminal rescale | qwen (`pipeline.rs:450-456`) | E |
| Native σ computed on the device vs on the host | flux MLX uses MLX `linspace` for parity (`pipeline.rs:124-127`); qwen on the host; mage ATen-exact host | E/A (parity-pinned) |
| Schedule-resolution API | `resolve_flow_schedule` (most), `ResolvedFlowSampling` (z-image), `ResolvedSampling` (sdxl), `WanScheduler::set_timesteps` (wan), `ExecutionPlan` (ltx dev), `JointSchedule` (h3) | A (6 shapes for "native + curated + truncation") |
| Multi-schedule | two modality schedules in one step (h3), two stage schedules (ltx), per-phase slices (krea) | E |

### 3.4 Solver / loop

| Axis | Values | Families | E/A |
|---|---|---|---|
| Shared curated driver | flux, qwen, z-image, boogu base, sana base, sdxl-curated, wan-curated, ltx-curated | — |
| Family native loop | sdxl ancestral (`pipeline.rs:832`), wan UniPC/DPM (`scheduler.rs`), ltx legacy Euler (`pipeline.rs:103`), h3 (`denoise/mod.rs:216`), flux3 (`pipeline.rs:49`), mage (`:1556`), sensenova, boogu DMD (`:851`), sana SCM, bernini (`pipeline.rs:212`) | E for flux3 / SCM / DMD / h3 / MAR; **A for wan UniPC/DPM++** (gen-core has `UniPc`, `Dpmpp2m`: `M/src/sampler.rs:494-505`; Wan keeps its own for N1 byte parity), A/E for sdxl ancestral (parity default) |
| Per-step stochasticity | ancestral noise (sdxl, `denoise.rs:356`), DMD fresh noise (boogu `:883`), SCM renoise, curated SDE | E |
| RNG | MLX `random::key(seed)` (flux, qwen, wan), MLX global-seeded stream (sdxl `stages.rs:35`), `StdRng` (candle sdxl), candle `seeded_noise_nchw` (`C/candle-gen/src/seed.rs:72`), SplitMix64 Box–Muller (sensenova), Gaussian-Shading (mage), bf16-rounded (flux3, z-image) | E/A (parity-locked to references, but the *seed derivation per image / stage* is accidental: `wrapping_add(i)` everywhere, LTX `0x5DEE_CE66` vs candle `+1..+3`) |
| Multi-eval solvers and step index | flux reconstructs `step_idx` from σ (`model.rs:885`); Wan refuses approximations on curated because "the step index never reaches the predict closure" (`denoise.rs:180-185`) | A: the driver should pass `StepInfo{index, outer_step, eval_in_step}` |

### 3.5 Guidance

| Kind | Families | Implementation | E/A |
|---|---|---|---|
| None (distilled) | flux schnell, z-image turbo, qwen lightning, boogu turbo, h3 (pinned by test), ltx distilled, flux3 distilled, sana sprint | — | E |
| Guidance-embed scalar | flux dev (`guidance` into the DiT), sana sprint (embedded), flux3? (not checked) | model input | E |
| Plain CFG | z-image base (`pipeline.rs:309` inline), wan (`denoise.rs:6`), sdxl (`gen_core::guidance::cfg`, MLX) / inline (candle `denoise.rs:350`), flux true-CFG (`model.rs:761`), flux3 base, mage, sana base, boogu base (`cond + (s−1)(cond−uncond)` form) | **6+ inline copies** | E (math) / A (copies) |
| Norm-rescaled CFG (cfg_rescale) | qwen (`compute_guided_noise` MLX `pipeline.rs:462`, candle `pipeline.rs:191`), sensenova `CfgNorm::Channel`/`Global`, mage `renormalize`, ltx dev std-rescale (`dev_sampler.rs:131-144`) | 4 different implementations; gen-core `cfg_rescale` (`GC/guidance.rs:145`) is used by Lens only | E/A |
| CFG++ | sdxl MLX only (`M/src/sampler.rs:626`) | sampler-level | E |
| APG (momentum, stateful) | bernini (`forward.rs:333`, momentum buffers persist across steps) | gen-core delegated | E |
| CFG-Zero* | sensenova (`t2i.rs:262`, step-0 zeroing) | family | E |
| Multi-branch | ltx dev: uncond + cond + STG-perturbed (skip blocks) + isolated modality (`dev_sampler.rs:84-130`); sensenova text + image CFG; bernini 4 streams / 7 modes | family | E |
| CFG batching | batch rows (sdxl, wan, flux3 `[neg, text]`), **sequence-fused** (mage), sequential halves with per-half eval (sdxl `CfgBatching`, `plan.rs:102`) | | E/A (memory rung). Row order varies: MLX sdxl `[text, neg]` vs candle sdxl `[uncond, cond]` vs wan `[cond, uncond]` vs flux3 `[neg, text]` = **A** |
| Guidance schedule over time | flux PuLID `timestep_to_start_cfg` (`model.rs:748`), sensenova `cfg_interval` (`t2i.rs:272`), bernini omega rescale at the expert switch (`pipeline.rs:256`), wan per-expert guidance (`moe.rs:13-14`), krea per-phase CFG on/off, ltx dev guidance only in stage 1 | 6 different mechanisms | E (need) / A (mechanisms) |

### 3.6 Conditioning injection

| Kind | Families | Resolution-aligned? |
|---|---|---|
| Cross-attention text (+ cached cross-KV) | wan (`StepCache.cross_kv`), sdxl, sana, ltx | no |
| Joint-attention text tokens (MMDiT) | flux, qwen, z-image, flux3, mage, boogu, h3 | no (but text length enters the packed layout: h3, mage) |
| Prefix KV cache (LLM backbone) | sensenova (`prefill_ids` `t2i.rs:739`), bernini planner | no |
| Pooled / micro-conditioning | sdxl `time_ids` (hardcoded `[512,512,0,0,512,512]`, `M/mlx-gen-sdxl/src/pipeline.rs:38-45`), flux CLIP pooled | no (sdxl does not pass real sizes) |
| Reference tokens concatenated on the sequence axis | qwen edit (`pipeline.rs:824`), flux3 refs, boogu edit (`forward_edit_multi`), bernini packed sources, h3 fl2va/ref2va conditioning rows, ltx IC-LoRA appended clips (`conditioning.rs:462`) | refs sized by **their own** geometry (qwen `condition_resize_dims`), except h3 / ltx anchors which are at the target grid |
| Channel concat | wan I2V `y` (`denoise.rs:106`), qwen Fun-control 33-ch context (candle `pipeline.rs:86`), z-image control context | **yes** |
| Replace-latent + per-token σ mask | ltx I2V / keyframes (`conditioning.rs:160-283`), wan TI2V mask blend (`conditioning.rs:211`) | **yes** |
| Residual control branch (ControlNet / VACE) | flux candle (`control_provider.rs`), sdxl (`model.rs:1736`), z-image control, qwen Fun-control, wan VACE | **yes** (the control latent and hint are on the target grid) |
| Adapter injectors | flux IP / PuLID (`DitImageInjector`), sdxl IP tokens, InstantID | no (image embeds) |
| Inpaint mask blend per step | sdxl (`pipeline.rs:978`), candle sdxl edit | **yes** |

### 3.7 Multi-stream / multi-stage / multi-model

| Axis | Families |
|---|---|
| Joint video + audio latents | ltx (`AvLatents` `M/src/sampler.rs:783`; separate audio noise per stage), h3 (rows in one sequence, two schedules) |
| Multi-stage with geometry change | ltx (half-res stage 1 → learned 2× upsample → stage 2), ltx DFR temporal rounds (`TEMPORAL_SIGMAS` `pipeline.rs:58`) |
| Mid-trajectory model swap | wan A14B high/low (`moe.rs:86`), bernini renderer (inline) |
| Two-model pipeline (planner → renderer) | bernini (MAR planner → Wan renderer) |
| AR stage before diffusion | sensenova think, ltx enhancer, flux3 PUP, bernini MAR |
| Phase-scoped adapters | ltx `pass_scales` (`GC/runtime/adapters.rs:18`), wan `moe_expert` (`:25`), krea `GenerationPhase.adapters` (`GC/generator/params.rs:362-380`) |

### 3.8 Decode and output

| Axis | Values | Families |
|---|---|---|
| Decoder | VAE KL (flux / qwen / z-image / sdxl / boogu), Wan VAE z16 / z48 (vae22), LTX video VAE + audio VAE + vocoder, H3 video VAE + audio VAE, DC-AE (sana), Flux3 AE (BatchNorm-stat latents), Mage NeRF decoder, fm_head (sensenova), PiD overlay (4× SR) |
| Tiling | spatial (most), spatio-temporal (wan, ltx planners), bounded host decode (candle flux `pipeline.rs:1535`), none (boogu `decode_latents`) |
| Output | `Images`, `ImagesRgba` (qwen-image-2.1 both backends, candle lens), `Video{frames,fps,audio}` (wan, ltx, h3), interleaved text + image (sensenova) | `GC/generator/output.rs:10` |

---

## 4. Accidental differences (same concept, different implementation, no domain reason)

All **[H]** unless marked.

1. **Plain CFG combine is re-implemented inline in at least 8 places** despite `GC/guidance.rs:128` `cfg`: z-image base `M/mlx-gen-z-image/src/pipeline.rs:309` (and `:567`), wan `M/mlx-gen-wan/src/pipeline/denoise.rs:6`, candle wan `C/candle-gen-wan/src/pipeline.rs:85`, flux true-CFG `M/mlx-gen-flux/src/model.rs:761-765`, candle sdxl `C/candle-gen-sdxl/src/denoise.rs:347-350` and `:538`, sana `M/mlx-gen-sana/src/pipeline.rs:219-245`, boogu `M/mlx-gen-boogu/src/pipeline.rs:566-570` (in the `(s−1)` form), flux3 `M/mlx-gen-flux3/src/pipeline.rs:160-166`. Only MLX SDXL calls the shared one.
2. **Norm-rescaled CFG exists four times:** qwen MLX `pipeline.rs:462`, qwen candle `pipeline.rs:191`, sensenova `CfgNorm::Channel`, ltx dev std-rescale. The shared `cfg_rescale` (`GC/guidance.rs:145`) is the same math as the qwen one. Note: the LTX one is a *global sample-std* rescale (`dev_sampler.rs:131-144`) while the qwen one is per-token L2. The "reduction axes" parameter of `GuidanceOps` (`GC/guidance.rs:88`) is designed for exactly this difference.
3. **CFG batch row order is inconsistent:** MLX sdxl `[text, neg]` (`pipeline.rs:791`,`:826`), candle sdxl `[uncond, cond]` (`denoise.rs:23`,`:345`), wan `[cond, uncond]` (`denoise.rs:120-122`), flux3 `[neg, text]` (`pipeline.rs:146`).
4. **Wan solver library duplicated twice** (MLX `M/mlx-gen-wan/src/scheduler.rs:45-519`, candle `C/candle-gen-wan/src/scheduler.rs:98-270`) alongside gen-core `Solver::{UniPc, Dpmpp2m}` (`M/src/sampler.rs:494-505`). The native path stays for N1 byte parity, which is a legitimate constraint, but it is still a third and fourth copy of UniPC/DPM++.
5. **MoE crossing logic exists four times:** `M/mlx-gen-wan/src/pipeline/moe.rs:26`, `C/candle-gen-wan/src/wan14b.rs:937`, `C/candle-gen-wan/src/vace.rs:644`, and inline in `M/mlx-gen-bernini/src/pipeline.rs:~250` (`if t >= boundary`). `staged_expert_swap` is duplicated MLX/candle (`moe.rs:289` / `wan14b.rs:967`).
6. **The LTX two-stage boundary is duplicated three times in MLX** (`pipeline.rs:579/592`, `:1504-1589`, `:1707-1759`) and formalized differently in candle (`TwoStageOrchestration`, `C/candle-gen-ltx/src/pipeline.rs:59-190`). `StageProgressFold` has a different constructor per backend (`new(total)` vs `new(offset, stage_steps, total)`).
7. **Staged residency duplicated:** gen-core `run_staged_request_scoped` vs candle z-image's local `run_three_stage` (`C/candle-gen-z-image/src/pipeline.rs:406`) vs LTX and Wan hand-rolled `drop` + `clear_cache` sequences (`M/mlx-gen-ltx/src/model/generate.rs:394-395`; `a14b_generate.rs:262`).
8. **Per-image seed loop helpers:** candle `for_each_image_seed` (`C/candle-gen/src/seed.rs:101`); MLX has none, so each family writes its own: flux manual loop (`model.rs:854`), qwen `decode_and_collect` (`pipeline.rs:200`), z-image `denoise_batch` (`pipeline.rs:1105`), sdxl `run_output` (`stages.rs:50`), flux3 `decode_batch` (`model.rs:481`), mage packs.
9. **Preview wiring takes four shapes:** (a) the MLX on-latent hook (flux `model.rs:866-877`, z-image `pipeline.rs:185`); (b) inside the predict closure (qwen `pipeline.rs:568`,`:822`, which relies on counter dedup for multi-eval solvers); (c) bespoke loops with a synthetic σ (sdxl MLX `pipeline.rs:900-925`) or `with_steps` (candle sdxl `denoise.rs:289`); (d) the candle driver-owned `PreviewHook` (`C/candle-gen/src/sampler.rs:188-245`). Coverage also differs by backend: candle boogu, sensenova and kolors have previews, MLX boogu and sensenova do not; no wan, ltx, h3, flux3, mage or bernini previews on either backend.
10. **Shared driver API shape differs per backend:** MLX `SamplerRun{…}` struct + `_with_latent_hook` variants (`M/src/sampler.rs:422`), candle positional args + `Option<&PreviewHook>` (`C/…/sampler.rs:188-198`). CFG++ is MLX-only; SCM is a candle core driver (`:624`) while MLX keeps it family-local.
11. **Schedule resolution has six shapes** (§3.3). Candle qwen-edit **drops** `req.scheduler` and `req.sampler` (`C/candle-gen-qwen-image/src/edit.rs:722`,`:749`) while MLX edit honors them via `resolve_run_params` (`M/…/pipeline.rs:125`).
12. **img2img blend re-implemented:** shared `add_noise_by_interpolation` (`M/src/img2img.rs:93`) vs inline in candle z-image (`pipeline.rs:1350-1355`); `init_time_step` duplicated in candle (`common.rs:85`, "mirrored from mlx-gen").
13. **RoPE/ids caching:** cached once (candle flux `pipeline.rs:744`, z-image `pipeline.rs:138`, wan `denoise.rs:49`, ltx `rope_epoch` `pipeline.rs:120`, mage `pack_context` `:1584` area) vs recomputed every step (MLX flux `transformer.rs:448`; qwen in forward `transformer/transformer.rs:349` **[M]**, internal memo not checked).
14. **Memory-rung resolution:** flux `memory_strategy::decode_tiling(req)` + `attention_plan(req)` (`model.rs:637-651`,`:839`), qwen `resolve_request_rungs` → `RequestRungs{block_window, attention_budget, decode_tiling}` (`model.rs:363`), z-image `resolve_request_rungs` with 6 fields (`model.rs:639-646`), mage reads `GenerationMemory` fields inline (`pipeline.rs:~1665-1700`), candle flux reads `req.memory.transformer_window_size` inline (`pipeline.rs:751-756`).
15. **Velocity sign placement:** inside the model (MLX z-image), in the closure (candle z-image `.neg()`), or in a family velocity fn (boogu `× −1`).
16. **Timestep scaling placement:** sana multiplies σ·1000 in the closure (`pipeline.rs:~214`), wan truncates σ·num_train in its scheduler (`scheduler.rs:108`), and z-image applies `t_scale` inside the model.
17. **Progress callback timing:** after the step (wan `moe.rs:72-74`, h3 `mod.rs:10-13`, flux3 `pipeline.rs:96`) vs before the compute (bernini `pipeline.rs:~250`).
18. **Step index for CFG scheduling** is reconstructed from σ in flux (`model.rs:885`) because the driver does not pass it.
19. **Per-phase mechanisms (6 of them)** for one concept, "a slice of the trajectory with different {model, adapters, guidance}": krea phases, ltx `set_lora_pass`, wan `Expert` + `moe_expert` adapters, bernini omega rescale, flux `timestep_to_start_cfg`, sensenova `cfg_interval`.
20. **Z-Image turbo schedule:** MLX static shift 3.0 (`M/mlx-gen-z-image/src/sampling.rs:20`) vs candle `calculate_shift(image_seq_len)` (`C/…/pipeline.rs:1334-1343`) with a curated mu of `0.0` (`:1346`). Either a real numeric divergence or a config that makes `mu` inert. **[L] verify** (candle-transformers is not vendored locally).

---

## 5. Mapping the two cross-cutting features

### 5.1 Feature A: "run the first K denoise steps at low resolution, then continue at full resolution"

**Shared mechanics** (all in S0 / S4 / S6):

1. **S0 rewrite_plan:** split the phases into `[P_low: steps 0..K, geometry g_low = g/s] + [P_high: K..N, geometry g]`. Validate `g_low` against the family size multiple (flux and qwen 16, ltx 64 for two-stage, h3 legal frame counts `GCH3/` and `denoise/geometry.rs:133`).
2. **S4 schedule:** **one global schedule computed at the target geometry** (Krea's continuity invariant). For resolution-dependent `mu` (flux-dev, qwen), decide whether phase 1 uses the target `mu` (continuous trajectory, recommended) or `mu(g_low)` (a discontinuous σ at the boundary). That is a recipe knob, not a family decision.
3. **S6 transition:** `latent' = resample(latent, g_low → g)` in **unpacked** latent space (requires the `LatentSpace` pack/unpack, `GC/latent.rs:144`), then optionally `renoise(latent', fresh_noise(g), σ_K)`, as LTX does (`pipeline.rs:157`, `:1586-1588`). The renoise noise needs a dedicated seed stream (LTX shows the need: `TwoStageSeeds`).
4. **S6 rebuild:** invalidate and rebuild the StepContext (RoPE/ids/packed layout/attention masks) and the **grid-aligned refs** (control latent, init latent, inpaint mask, I2V `y`, TI2V mask, keyframes).
5. **S8:** reset multistep solver history (DPM++/UniPC carry `x0` history whose shape changes). Per-step RNG of stochastic solvers is fine if noise is drawn `randn_like(x)` (gen-core `LatentOps::randn_like`, `M/src/sampler.rs:363`).
6. **S7 observe:** previews must re-key their projector geometry (the candle `preview::hook(&req.preview, w, h)` is built per image with fixed w, h: `C/candle-gen-flux/src/pipeline.rs:1449`).

**Per family**

| Family | Fits shared mechanism? | What changes at the boundary | Escape hatch needed? |
|---|---|---|---|
| FLUX.1 (both) | Yes | repack tokens; RoPE (MLX already per-step, candle `prepare_conditioning` must re-run); `mu` choice; candle ControlNet control latent re-encode at `g`; IP embeds unchanged; PiD capture `keep` index must stay in the global schedule | No |
| Qwen T2I | Yes | `lh, lw` arguments to the forward; `mu`; img2img clean latent at both geometries | No |
| Qwen Edit | Yes | the noise grid changes; **reference tokens stay at their own size** (`condition_resize_dims`), so at low-res the ref:noise token ratio inflates (a quality risk, not a mechanism problem) | No |
| Z-Image (turbo / base / control) | Yes | `transformer.prepare(shape, cap)` re-run (`pipeline.rs:138`); control context re-encode (`pipeline.rs:638`); turbo is only 4 steps, so K ∈ {1, 2} | No |
| SDXL native ancestral | Mostly | UNet is conv (resolution-flexible); `time_ids` is a constant (`pipeline.rs:38-45`), so nothing to update; ancestral noise drawn per step at the current shape; inpaint `InpaintBlend` holds a fixed noise + x0 + mask at g (must be rebuilt for `g_low`); ControlNet `cond_embed` at both geometries; the MLX global RNG stream advances differently (parity break, acceptable). | Native ancestral `DiffusionSampler` (`stages.rs` `FamilyStages::denoise`) must expose a slice API. **Yes, partial**: the `AncestralEuler` built from `(steps, start_time)` (`model.rs:1924-1990`) needs slicing |
| SDXL curated | Yes | as flux | No |
| Wan T2V/I2V | Yes in principle | `StepCache` RoPE rebuild (cross-KV unchanged); I2V `y` re-encode at `g_low`; TI2V mask/keyframe at `g_low`; the native UniPC scheduler is stateful (`WanScheduler::step` `&mut self`) so it needs a history reset or a resample; the VAE temporal geometry is unchanged | **Partial**: the native `WanScheduler` loop is not the shared driver; low-res-first must either route through curated or add a `reset_history` to `WanScheduler` |
| Wan A14B MoE | Yes, composes | low-res phase vs expert boundary: these are **two orthogonal phase axes** (geometry and model). The phase list must be the product of the boundaries (`crossing_index` and K). Staged swap ordering: the expert swap and the geometry change may coincide | No if the phases are a merged list |
| LTX | **Already native** (stage 1 at half res, learned upsampler, renoise) | Express the existing stage boundary as the shared S6 transition with `transition = LearnedUpsampler`; audio is not upsampled (a per-stream transition policy) | Its distilled σ tables are fixed, so K is not free. The feature = "native 2-stage". Keep the family transition as a plug-in |
| MiniMax-H3 | Hard | `PackedLayout` (text + video + audio rows, conditioning rows) and the precomputed AdaLN table must be rebuilt; video rows change, audio rows do not; fl2va anchors must be re-encoded at `g_low`; legality is geometry-restricted | **Yes** (FamilyDriver with a phase-aware layout rebuild). Feasible but not "free" |
| FLUX3 distilled (flow-map) | Partial | the flow map jumps `t → t_ref`; a boundary at schedule index K is fine; `image_ids(h, w, image_rope_scale)` rebuild; reference tokens unchanged | Bespoke loop (`pipeline.rs:49`) must take a slice + start latent: **small escape hatch** |
| Mage | **No (by default)** | Gaussian-Shading watermark noise is bit-keyed to the token count (`latent.rs:725`), so starting at `g_low` breaks watermark decodability; multi-sample packed sequences need per-sample geometry | **Yes**: refuse, or watermark only the final-res renoise (a product decision) |
| SenseNova | Partial / hard | pixel-space patches; the prefix KV is fine; `rope_theta_hw` 2D RoPE over image tokens must change; the interleave/think flows fix the resolution via `interleave_resolution_for` (`t2i.rs:351`) | **Yes** (family loop). Low priority |
| Boogu turbo (DMD) | **Natural fit** | the DMD loop already renoises with **fresh noise per step** (`pipeline.rs:883`), so the boundary is "x0 at g_low → upsample x0 → renoise at g with fresh noise", no trajectory discontinuity concern | Native loop needs a geometry-per-step hook. Small |
| Boogu base | Yes | as flux (shared driver) | No |
| Sana base | Yes | DC-AE 32× latent (small grids, so `g_low` may hit the minimum size quickly) | No |
| Sana Sprint (SCM) | Low value | 1–4 steps. The SCM renoise step (`cos(t')x0 + sin(t')·noise·σ_data`) is a natural transition point (like DMD) | Small escape hatch (SCM loop is family-local in MLX, shared in candle) |
| Bernini | **No** (planner) / Partial (renderer) | the MAR planner works on a fixed ViT token grid; the renderer is Wan (as Wan, with packed source tokens re-patched at `g_low`) | **Yes** for the planner; renderer feasible |

### 5.2 Feature B: per-phase LoRA and guidance

**Shared mechanics:**

- **Vocabulary exists:** `GenerationRequest.phases` (`GC/generator/request.rs:330`) and `GenerationPhase{steps, guidance, adapters: Vec<PhaseAdapter>}` (`GC/generator/params.rs:362-380`), with host math in `GCF/krea_multiphase.rs` (`resolve_phase_slices` `:86`, `resolve_phase_adapters` `:142`). Promote it from Krea-only to the shared S0 / S6. Extend `GenerationPhase` with an optional `geometry` (Feature A rides the same list) and optional `guidance_method` / params.
- **Adapter switch mechanism (S6 `select_adapters`):** three implementations exist. (a) Clone the DiT per phase with a different `AdaptableLinear` stack (MLX krea; `AdaptableLinear` is `Clone` and residual-form, `M/src/adapters.rs:604`); (b) a pass index read by each linear (LTX `LoraStack.pass: Cell<usize>`, `M/mlx-gen-ltx/src/transformer/linear.rs:299-310`); (c) per-expert stacks (wan `moe_expert`). **Recommendation:** generalize (b) into the core `AdaptableLinear`: per-adapter `scale_by_phase: Vec<f32>` plus a thread-local or context-carried active phase. That avoids clones and works when the DiT is not `Clone` (candle krea).
- **Blocker:** families that **merge adapters into the dense weights at load** cannot switch per phase without unmerging. The grep for merge/fuse terms hits many crates (candle: anima, bernini, ideogram, instantid, iris, krea, lens, qwen-image, qwen-2.1, scail2, sd3, sdxl, sensenova, wan, z-image; MLX: flux2, instantid, iris, kolors, krea, ltx, mage, qwen-2.1, sam2, sana, scail2, sdxl, sensenova, wan) **[M: a grep hit is not proof of a merge path in the production route]**. For those, per-phase adapters need either residual mode (a quantized-safe residual exists in MLX core) or a refusal at S0.
- **Guidance switch (S6 `select_guidance`):** the guidance plan per phase = {method, scale(s), branches}. CFG on/off changes the number of forwards (Krea `phase_uses_cfg` `GCF/krea_multiphase.rs:117`) **and** whether negative conditioning must be encoded in S2. The S0 plan must therefore compute `any_phase_uses_cfg` (`:124`) before encoding (Qwen's Lightning path shows the cost of not encoding neg: `model.rs:405-417`).
- **Stateful guidance (APG momentum):** buffers must live in the phase runner and be reset or carried by policy (Bernini carries them across the expert switch, `pipeline.rs:240-243`).

**Per family**

| Family | Per-phase guidance | Per-phase LoRA | Escape hatch? |
|---|---|---|---|
| FLUX.1 | dev: guidance-embed scalar per phase is trivial; true-CFG per phase is already approximated by `timestep_to_start_cfg` (`model.rs:748`), which becomes a phase | MLX `AdaptableLinear` residual (Hyper-FLUX LoRA at 0.125 is the canonical "turbo phase" case) | No |
| Qwen | Lightning (CFG-off) vs true-CFG per phase = exactly the Krea workflow; Lightning schedules differ (`sampler.rs:52`) → only one global schedule allowed | candle qwen fuses LoRA into dense weights at load (`C/candle-gen-qwen-image/src/lib.rs:23`, "into the dense MMDiT weights at load") → needs residual mode | **Candle: yes** (merge-at-load) |
| Z-Image | turbo vs base share the transformer (`model_base.rs:6-9`): a "base-CFG phase then turbo phase" is a weight-identity question (separate checkpoints), not a LoRA; CFG per phase fine | candle merge-at-load **[M]** | Turbo↔base swap = a **model-swap phase** (like Wan MoE) |
| SDXL | CFG scale per phase fine (bespoke loop gets a per-step scale); CFG++ is a sampler-level switch, so it must be phase-constant or the driver must split | MLX `AdaptableLinear`; LCM/Lightning acceleration is a *sampler route*, not a phase (`sampling.rs:25`) | Native ancestral loop needs a per-step guidance callback |
| Wan A14B | **already per-phase** (per-expert guidance `Expert.guidance`, `moe.rs:13`) | `moe_expert` adapters exist; per-phase within an expert needs (b). Wan merges adapters (`M/mlx-gen-wan/src/adapters/merge.rs`) on some routes | Partial |
| LTX | **already per-pass** (`set_lora_pass`, `pass_scales`); dev guidance (4-branch) is stage-1 only | native | Map `pass_scales[i]` onto phases; LTX's 2 passes become 2 phases |
| H3 | guidance-distilled (no CFG, pinned by test, `denoise/mod.rs:29-41`), so per-phase guidance is N/A | LoRA via `pass_scales`-like support exists in candle/MLX H3 (grep hit) **[M]** | Guidance: refuse |
| FLUX3 | distilled flow-map has no CFG; base has CFG; switching distilled ↔ base mid-trajectory is a model swap with **different time inputs** (`target_timestep` vs none) | n/a | **Yes** |
| Mage | CFG per phase OK (sequence-fused CFG layout changes with on/off: `cond_layout.fused_cfg`) | Turbo / Base / RL are separate checkpoints | Partial |
| SenseNova | `cfg_interval` is already a 2-boundary phase schedule; CfgNorm per phase fine; **distill LoRA is merged** (`fm.rs:59` `merge_distill_lora`, `distill.rs:97`) | merged → refuse | **Yes** (adapters) |
| Boogu | base true-CFG ↔ turbo DMD **are different samplers over different weights** (`lib.rs` doc: "Turbo is just a different sampler over the (separately-distilled) DiT weights") → a model-swap phase plus a solver switch | — | **Yes** |
| Sana Sprint | embedded guidance scalar per phase trivial | — | No |
| Bernini | omega rescale at the switch is already a 2-phase guidance schedule | — | Express as phases |

### 5.3 Families that need a `FamilyDriver` escape hatch

Shared phase runner (S6/S8 + shared driver S7) covers: flux, qwen, z-image, boogu-base, sana-base, sdxl-curated, wan-curated, krea,
chroma, lens, flux2 (by adoption list).

**FamilyDriver required** (the phase runner calls `driver.run_slice(state, slice, step_ctx, hooks)` and the family owns the inner loop):

| Family | Reason (essential) |
|---|---|
| FLUX3 distilled | flow-map `(t, t_ref)` integrator (`pipeline.rs:49-100`) |
| Sana Sprint / SCM | angle-space consistency renoise (`scm.rs:1-30`) |
| Boogu Turbo native | DMD predict + fresh-noise renoise (`pipeline.rs:851`) |
| MiniMax-H3 | joint rows, two schedules, cond-row freeze, AdaLN table (`denoise/mod.rs:198-300`) |
| LTX native | per-token σ, denoise-mask pin, 4-branch dev guidance, legacy Euler parity (`pipeline.rs:103`, `dev_sampler.rs:178`) |
| SenseNova | x-prediction + prefix KV + CfgZeroStar + dual CFG |
| Mage | sequence-fused CFG over packed multi-sample sequences + watermark noise |
| SDXL native ancestral, Wan native UniPC/DPM | **N1 byte-parity defaults** (accidental in kind, but contractual). Hooks must be offered through the driver trait |
| Bernini planner | MAR masked AR loop (`mar.rs`) |

The FamilyDriver contract must still accept: the slice (σ indices), StepContext rebuilt by the shared S6, the shared `observe` / `post_step` hooks,
and the per-phase adapter/guidance selection. That is how cross-cutting features still "work for all models", with an explicit
per-family `supports(feature)` refusal like Wan's `refuse_unwired_approximation` (`denoise.rs:192-215`) where they cannot.

---

## 6. Design implications

1. **Make the phase list the backbone of the shared pipeline.** Generalize `GenerationPhase` (`GC/generator/params.rs:362`) to `{steps, geometry?, model_variant?, adapters, guidance_plan, transition_in}` and have *every* family's S0 produce a phase list (single-phase by default). Wan MoE, LTX two-stage, Krea, Bernini and flux `timestep_to_start_cfg` then become data, and both features in §5 are just phase-list rewrites.
2. **Keep one global schedule; make the transition explicit.** Krea's contiguity invariant plus LTX's renoise-to-σ_j are both supported by `transition_in = Continue | Renoise{sigma_index, seed_stream} | Upsample{kind: Nearest|Bilinear|Learned(id)} + Renoise`. Seed streams are named per phase/stream (adopt candle LTX's `TwoStageSeeds` idea; `C/candle-gen-ltx/src/pipeline.rs:40-46`).
3. **Promote the existing typestate ideas.** Candle LTX's `TwoStageOrchestration` (phase order enforced, adapter pass set at the right boundary) and SDXL's `FamilyStages` (init/denoise split with retained-state lifetime) are the two shapes to generalize into the S5–S8 runner.
4. **Separate grid-aligned conditioning from geometry-free conditioning at the type level.** `RefBundle::grid_aligned: Fn(Geometry) -> AlignedRefs` (control latents, init latents, masks, I2V `y`, keyframes, fl2va rows). Without this, low-res-first silently uses full-res control maps or crashes on shape. Today all of these are "hoisted once" (qwen `model.rs:434`, sdxl `model.rs:1771`, wan `a14b_generate.rs:287`).
5. **StepContext is a cache keyed on (model, geometry, conditioning).** Unify RoPE/ids/cross-KV/AdaLN/packed-layout preparation (wan `StepCache`, z-image `prepare`, candle flux `prepare_conditioning`, ltx `rope_epoch`, h3 `PackedLayout`, mage `pack_context`) behind `FamilyModel::prepare_step_context`. That also fixes the per-step RoPE recompute in MLX flux.
6. **Pass a `StepInfo{phase, step, outer_step, eval_in_step, sigma, next_sigma}` to predict and hooks.** That removes the σ-count reconstruction in flux (`model.rs:885`) and lets Wan's feature cache and per-step guidance schedules work under multi-eval solvers.
7. **Finish the guidance-library adoption.** Route every combine through `GC/guidance.rs` with an explicit reduction geometry (`GuidanceOps` axes), add a `std_rescale` (LTX) and `CfgZeroStar` (SenseNova) operator, and standardize the CFG branch ordering (§4.3). Multi-branch guidance (LTX STG/modality, SenseNova dual, Bernini 4-stream) needs `Branch` enumeration in the predictor plus a combine operator over N branches.
8. **One driver API per backend, same shape.** Align MLX `SamplerRun` + latent hook and candle positional + `PreviewHook` into one hooks struct. Move CFG++ to candle and SCM into MLX core so feature availability stops depending on backend.
9. **Single residency engine with N phases.** The `Text → Heavy → (shed) → Decode` lifecycle plus expert swaps (`staged_expert_swap`) plus LTX's manual drops should all be one residency runner whose phases align with S2 / S6 / S10, so a per-phase model swap (Wan experts, turbo↔base) reuses the same evict-then-load path.
10. **Feature capability negotiation at S0.** Each family declares `supports(Feature::LowResFirst{transition}) / supports(Feature::PhaseAdapters{residual|merged})`. Merge-at-load families and fixed-geometry families (Mage watermark, Bernini planner, H3 legal grids) refuse with a reason. That matches the existing refusal precedent (Wan approximations, LTX curated-on-I2V `model/generate.rs:247-259`).
11. **Do not try to absorb the essential bespoke loops.** FLUX3 flow-map, SCM, DMD, H3 joint rows, LTX per-token σ, SenseNova x-pred + KV, Mage fused CFG and the Bernini MAR planner are essential variance. Give them the `FamilyDriver::run_slice` contract (§5.3) so phases, previews and adapters still work around them.
12. **Use `LatentSpace` as the geometry/layout key.** `GC/latent.rs:144` already describes channels, compression, packing and the temporal law. Pack/unpack/resample for transitions and previews can be implemented once against it instead of per family (`unpack_latents` exists in flux, qwen, z-image, mage and h3 separately).

---

## 7. Uncertain / not verified

- **[L] Z-Image candle vs MLX Turbo schedule:** whether `SchedulerConfig::z_image_turbo()` in candle-transformers (rev `1e6aa85e`, not vendored or available locally) has `use_dynamic_shifting = true` (a real resolution-dependent divergence from MLX's static 3.0) or makes the `calculate_shift` mu inert. The curated path passes `mu = 0.0` (`C/candle-gen-z-image/src/pipeline.rs:1346`) while MLX passes `ln 3`. This could be a real divergence on the curated-scheduler path either way.
- (Resolved during writing) SDXL `time_ids` are hardcoded `[512,512,0,0,512,512]` on both backends (`M/mlx-gen-sdxl/src/pipeline.rs:38-45`, `C/candle-gen-sdxl/src/denoise.rs:48-58`), a vendored quirk, so they are not resolution-dependent today.
- **[M] Qwen MLX RoPE per step:** `QwenRope3d::forward_multi` is called inside `forward_windowed` (`transformer/transformer.rs:349`). I did not check for an internal memo.
- **[M] Merge-at-load list:** derived from a keyword grep (`merge_lora|merge_adapters|fuse.*lora|into the dense`). Some hits may be training-only or alternate routes. Verify per family before declaring refusal.
- **Not walked line by line:** candle twins of flux3, mage, sensenova, boogu, bernini, sana-sprint, h3 beyond confirming file and function presence and the duplicated traits. Divergences there are not catalogued.
- **Not walked:** MLX Qwen ControlNet route (`denoise_control_with_progress`, `pipeline.rs:624-790`), z-image control loops beyond counting them, Wan VACE, LTX DFR internals (`dfr.rs`), H3 ref2va layout details, flux2, chroma, krea internals beyond multiphase, kolors, iris, lens, sd3, svd.
- **Guidance-embed for FLUX3 base:** not checked whether the base DiT also takes a guidance-embed scalar alongside CFG.
- **Bernini on_step-before-compute** (§4.17): read from the loop body at `pipeline.rs:~245-250`. The exact line is approximate.
- **Line numbers marked `~`** were derived from an offset within a viewed range rather than a direct grep hit. They are accurate to within a few lines.
- I did not check whether `GenerationRequest.phases` is *refused* (vs silently ignored) by non-Krea families. The doc comment at `GC/generator/request.rs:327` says "other models ignore it", which suggests silent ignore. That would be a correctness gap if phases become a shared feature.
