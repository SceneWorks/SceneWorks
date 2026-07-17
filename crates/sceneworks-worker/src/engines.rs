//! Backend-neutral engine dispatch table + registry-derived capability advertisement
//! (sc-3723, epic 3720 Phase 0).
//!
//! [`MODEL_TABLE`] is the SceneWorks-id → mlx-gen-registry-id map plus the per-variant
//! defaults the worker needs that are NOT on the engine descriptor (HF repo, step/guidance
//! defaults, the asset `adapter` label). It is **all-targets** (no `#[cfg(target_os = …)]`):
//! the table is pure data, and keeping it neutral lets the registry-derived advertisement run
//! off-macOS (where the provider crates aren't linked, so the registry is empty and the derived
//! capability set is correctly empty — "absence, not runtime failure").
//!
//! The two descriptor-duplicating flags that used to live on each row
//! (`supports_guidance` / `supports_negative_prompt`) are gone: they are now read from the
//! linked gen_core descriptor through [`ResolvedModel`], so a row can never drift from the
//! engine's own advertised surface. A future candle backend lights up with **zero** worker
//! changes — it registers its descriptors into the same `inventory` registry and
//! [`registry_capabilities`] picks them up.

/// One engine-backed image family: how a SceneWorks model id maps onto the linked
/// mlx-gen registry, and the per-variant defaults (all chosen for parity with the
/// Python `MODEL_TARGETS` + the per-family MLX adapter). Adding a family = one row
/// here + its provider crate dep + a `use mlx_gen_<x> as _;` in `image_jobs.rs`.
pub(crate) struct ModelRow {
    /// SceneWorks model id (the job payload `model`).
    pub sceneworks_id: &'static str,
    /// registry id passed to `crate::inference_runtime::load`.
    pub engine_id: &'static str,
    /// Default HuggingFace repo when the manifest entry omits `repo`.
    pub default_repo: &'static str,
    /// Default denoise steps (Python `MODEL_TARGETS[...]["steps"]`).
    pub default_steps: u32,
    /// Default guidance when supported and the request omits it.
    pub default_guidance: f32,
    /// The `adapter` id recorded on generated assets (the Python MLX adapter id).
    pub adapter_label: &'static str,
}

pub(crate) const MODEL_TABLE: &[ModelRow] = &[
    ModelRow {
        sceneworks_id: "z_image_turbo",
        engine_id: "z_image_turbo",
        // SceneWorks pre-built quant-matrix turnkey (sc-8670, epic 8506): q4/ (default) + q8/ + bf16/
        // packed subdirs, resolved by `standard_tier_subdir`. Re-host of Tongyi-MAI/Z-Image-Turbo.
        default_repo: "SceneWorks/z-image-turbo-mlx",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_z_image",
    },
    // Base (non-distilled) Z-Image (epic 8236, sc-8320). The undistilled foundation model from the
    // `Tongyi-MAI/Z-Image` diffusers snapshot — the same `ZImageTransformer` as Turbo, but the
    // `z_image` engine descriptor uses a shift=6.0 schedule, ~50 default steps, and REAL CFG
    // (`supports_guidance` + negative prompt; the card recommends guidance 3.0–5.0, default 4.0) vs
    // Turbo's 4-step guidance-distilled CFG-free path. Ships its own fast `tokenizer/tokenizer.json`,
    // so it needs NO derived-tokenizer overlay. Routes to the base t2i path (not Turbo); strict-pose /
    // canny / depth control routes to the `z_image_control` engine variant (sc-8251).
    ModelRow {
        sceneworks_id: "z_image",
        engine_id: "z_image",
        // SceneWorks pre-built quant-matrix turnkey (sc-8670): q4/ (default) + q8/ + bf16/ subdirs.
        // Re-host of the undistilled Tongyi-MAI/Z-Image base (bf16-native source).
        default_repo: "SceneWorks/z-image-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_z_image",
    },
    // Ideogram 4 (epic 4725) — native MLX, gated. Structured JSON-caption text-to-image; the
    // turnkey ships packed q4/ (default) + q8/ subdirs (resolve_ideogram_model_dir picks one).
    // V4_QUALITY_48 preset default (48 steps); asymmetric-CFG guidance 7.0.
    ModelRow {
        sceneworks_id: "ideogram_4",
        engine_id: "ideogram_4",
        default_repo: "SceneWorks/ideogram-4-mlx",
        default_steps: 48,
        default_guidance: 7.0,
        adapter_label: "mlx_ideogram",
    },
    // Ideogram 4 Turbo (mlx-gen #488) — the CFG-free, single-DiT few-step variant: the same
    // turnkey base (q4/q8 subdirs) plus the bundled ostris TurboTime LoRA the engine installs at
    // load. 8 steps; guidance is INERT (the `ideogram_4_turbo` descriptor advertises
    // supports_guidance=false, so `resolve_guidance` returns None and never forwards a value).
    ModelRow {
        sceneworks_id: "ideogram_4_turbo",
        engine_id: "ideogram_4_turbo",
        default_repo: "SceneWorks/ideogram-4-mlx",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_ideogram",
    },
    // Z-Image-Edit (epic 3529) — img2img/edit. No dedicated Edit checkpoint exists yet, so
    // (like the Python `MODEL_TARGETS` row) it runs the **Turbo weights** through the engine's
    // img2img path (`Conditioning::Reference` — VAE-encode the source + denoise from
    // `init_time_step(steps, strength)`), so it shares the `z_image_turbo` engine model. The
    // `z_image_turbo` `edit_image` mode resolves to the same img2img call (`resolve_zimage_edit_init`).
    ModelRow {
        sceneworks_id: "z_image_edit",
        engine_id: "z_image_turbo",
        // Shares the Turbo turnkey (sc-8670) — img2img runs the same Turbo weights/tier subdirs.
        default_repo: "SceneWorks/z-image-turbo-mlx",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_z_image",
    },
    ModelRow {
        sceneworks_id: "flux_schnell",
        engine_id: "flux1_schnell",
        // sc-8669: SceneWorks pre-quantized q4/q8/bf16 turnkey re-host (Apache-2.0).
        default_repo: "SceneWorks/flux1-schnell-mlx",
        default_steps: 4,
        default_guidance: 0.0,
        adapter_label: "mlx_flux",
    },
    ModelRow {
        sceneworks_id: "flux_dev",
        engine_id: "flux1_dev",
        // sc-8669: SceneWorks pre-quantized q4/q8/bf16 turnkey re-host (FLUX.1 [dev] Non-Commercial).
        default_repo: "SceneWorks/flux1-dev-mlx",
        default_steps: 28,
        default_guidance: 3.5,
        adapter_label: "mlx_flux",
    },
    ModelRow {
        // Non-distilled true-CFG base: 20 steps + guidance 4.0 + negative prompt
        // (Python MODEL_TARGETS / MlxQwenAdapter). mlx-gen's own default is 4 steps,
        // so steps are passed explicitly. Edit moves to MLX (sc-3397, the `qwen_image_edit`
        // engine model below); base-Qwen strict-pose ControlNet routes to the
        // `qwen_image_control` engine variant when `advanced.poses` is present
        // (epic 3401 / sc-3575).
        sceneworks_id: "qwen_image",
        engine_id: "qwen_image",
        // sc-8669: SceneWorks pre-quantized q4/q8/bf16 turnkey re-host (Apache-2.0); replaces the
        // gated-free but dense Qwen/Qwen-Image-2512 source + install-time quantize.
        default_repo: "SceneWorks/qwen-image-mlx",
        default_steps: 20,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    // Qwen-Image-Edit (sc-3397) — the three base edit ids all resolve to the engine's
    // single `qwen_image_edit` model (Reference/MultiReference, true CFG, LoRA/LoKr, Q4/Q8);
    // `qwen_image_edit`/`_2509` alias to the 2511 weights (Python MODEL_TARGETS, sc-2160).
    // 40 steps (engine's own default is 4 — passed explicitly, like the txt2img row). The
    // edit path resolves guidance from `trueCfgScale` (4.0), NOT `guidanceScale`; see
    // `resolve_qwen_edit_guidance`. The `_2511_lightning` distill (4-step, CFG-off) shares
    // these weights but adds the `lightning` sampler + the lightx2v distill LoRA — see the
    // row below and [`qwen_edit_lightning`] (sc-3398).
    ModelRow {
        sceneworks_id: "qwen_image_edit",
        engine_id: "qwen_image_edit",
        // sc-8669: SceneWorks pre-quantized q4/q8/bf16 Edit-2511 turnkey re-host (shared by all
        // edit ids + the lightning distill, same checkpoint).
        default_repo: "SceneWorks/qwen-image-edit-2511-mlx",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    ModelRow {
        sceneworks_id: "qwen_image_edit_2509",
        engine_id: "qwen_image_edit",
        default_repo: "SceneWorks/qwen-image-edit-2511-mlx",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    ModelRow {
        sceneworks_id: "qwen_image_edit_2511",
        engine_id: "qwen_image_edit",
        default_repo: "SceneWorks/qwen-image-edit-2511-mlx",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_qwen",
    },
    // Lightning 4-step distill (sc-3398): same `qwen_image_edit` engine model + base
    // Qwen-Image-Edit-2511 weights as the rows above, but the generate path passes the
    // `lightning` sampler (static-shift schedule + CFG-off single forward) and stacks the
    // lightx2v distill LoRA ahead of any user LoRAs (see [`qwen_edit_lightning`] +
    // [`generate_qwen_edit_stream`]). Python parity (MODEL_TARGETS): 4 steps, guidance 1.0,
    // CFG off — so no negative prompt. The distill LoRA is a CFG-distilled adapter, so the
    // engine runs a single forward/step regardless of `default_guidance`.
    ModelRow {
        sceneworks_id: "qwen_image_edit_2511_lightning",
        engine_id: "qwen_image_edit",
        default_repo: "SceneWorks/qwen-image-edit-2511-mlx",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_qwen",
    },
    // FLUX.2-klein (sc-3025) — MLX-only family (no torch fallback). All three SceneWorks
    // variants share the engine's single txt2img model `flux2_klein_9b` (edit + KV-cache
    // are the separate `*_edit`/`*_kv_edit` engine models, story sc-3029); the variants
    // differ only in their weights. Distilled klein runs guidance 1.0 (CFG-free) with no
    // negative prompt; the engine accepts guidance but rejects a negative prompt.
    // `default_repo` is the SceneWorks pre-quantized q4/q8/bf16 turnkey re-host (sc-8711,
    // epic 8506) — the model is registered in `STANDARD_TIER_MODELS`, so both the MLX and the
    // candle txt2img lanes resolve the packed subdir through the shared `resolve_weights_dir`
    // → `standard_tier_subdir` (base.rs). It must match the manifest `downloads[].repo`; the
    // stale gated `black-forest-labs/FLUX.2-klein-9B` default made `model_repo` probe the wrong
    // HF-cache dir and fail with "MLX weights not found or incomplete".
    ModelRow {
        sceneworks_id: "flux2_klein_9b",
        engine_id: "flux2_klein_9b",
        default_repo: "SceneWorks/flux2-klein-9b-mlx",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_flux2",
    },
    ModelRow {
        // Separately-distilled checkpoint, same architecture — txt2img loads through the base
        // `flux2_klein_9b` loader. `default_repo` is the SceneWorks q4/q8/bf16 turnkey re-host
        // (sc-8711); like the base 9B it is a `STANDARD_TIER_MODELS` member resolved via
        // `standard_tier_subdir`, so it must match the manifest `downloads[].repo` rather than
        // the pre-rehost gated `black-forest-labs/FLUX.2-klein-9b-kv`.
        sceneworks_id: "flux2_klein_9b_kv",
        engine_id: "flux2_klein_9b",
        default_repo: "SceneWorks/flux2-klein-9b-kv-mlx",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_flux2",
    },
    ModelRow {
        // wikeeyang community fine-tune (sc-2220/2235): UNDISTILLED, so 24 steps. Its raw
        // repo is single-file (GGUF/safetensors) with no diffusers tree, so it loads from a
        // locally-assembled converted dir via the `modelPath` seam (manifest `modelPath`),
        // NOT the source repo below. The convert step is now native Rust/MLX
        // (runtime_macos::providers::flux2::convert_and_assemble, sc-3136; run by the model_convert job).
        sceneworks_id: "flux2_klein_9b_true_v2",
        engine_id: "flux2_klein_9b",
        default_repo: "wikeeyang/Flux2-Klein-9B-True-V2",
        default_steps: 24,
        default_guidance: 1.0,
        adapter_label: "mlx_flux2",
    },
    // FLUX.2-dev (epic 5914) — the guidance-distilled 32B flagship. A SEPARATE engine
    // model `flux2_dev` (Mistral3 TE + 48/48/15360 DiT), NOT a klein weight variant, so it
    // maps to its own engine id. Embedded distilled guidance (FLUX.1-dev pattern, NOT
    // true-CFG): the descriptor advertises `supports_guidance` but not negative prompt, so
    // the engine takes the guidance scalar (default 4.0) over ~28 steps. `default_repo` is the
    // SceneWorks pre-quantized q4/q8/bf16 turnkey re-host (sc-8513, epic 8506) — a
    // `STANDARD_TIER_MODELS` member, so both the MLX and the candle txt2img lanes packed-load
    // the chosen tier's subdir via the shared `resolve_weights_dir` → `standard_tier_subdir`
    // (sc-9092 retired the old candle dense-BFL load for the generic lane). It must match the
    // manifest `downloads[].repo`; the pre-rehost gated `black-forest-labs/FLUX.2-dev` default
    // made `model_repo` probe the wrong HF-cache dir. NOTE: the bespoke off-Mac candle *edit*
    // lane in `flux2_edit_candle.rs` still keys its own dense-BFL default for both flux2 klein
    // and dev, which no longer matches the re-hosted packed turnkey the catalog downloads — a
    // separate off-Mac gap that needs candle-hardware verification, tracked as sc-10222 (epic
    // 9083 gap #3), not touched here.
    ModelRow {
        sceneworks_id: "flux2_dev",
        engine_id: "flux2_dev",
        default_repo: "SceneWorks/flux2-dev-mlx",
        default_steps: 28,
        default_guidance: 4.0,
        adapter_label: "mlx_flux2",
    },
    // SDXL (sc-3026) — U-Net, real CFG (negative prompt + guidance 7.0), 30 steps.
    // `sdxl` and the `realvisxl` finetune share the engine's single `sdxl` model
    // (identical arch), differing only in weights. Replaces the in-process
    // _vendor/mlx_sd path. The engine supports Q4/Q8 (the Python vendored path had
    // none); Q8 is the default here (engine-validated; saves ~half the U-Net memory).
    ModelRow {
        sceneworks_id: "sdxl",
        engine_id: "sdxl",
        // SceneWorks pre-built quant-matrix turnkey (sc-8746, epic 8506, Group-B): standard
        // q4/q8/bf16 subdirs (standard_tier_subdir) — UNet + both CLIP text encoders packed,
        // VAE dense. Public/ungated re-host of stabilityai/stable-diffusion-xl-base-1.0.
        default_repo: "SceneWorks/sdxl-base-mlx",
        default_steps: 30,
        default_guidance: 7.0,
        adapter_label: "mlx_sdxl",
    },
    ModelRow {
        sceneworks_id: "realvisxl",
        engine_id: "sdxl",
        // SceneWorks pre-built quant-matrix turnkey (sc-8746, epic 8506, Group-B): standard
        // q4/q8/bf16 subdirs (standard_tier_subdir). Re-host of SG161222/RealVisXL_V5.0.
        default_repo: "SceneWorks/realvisxl-mlx",
        default_steps: 30,
        default_guidance: 7.0,
        adapter_label: "mlx_sdxl",
    },
    // Illustrious-XL (epic 10609) — Danbooru-tag anime SDXL finetunes from OnomaAI. Architecturally
    // vanilla SDXL (identical UNet shapes, dual CLIP-L + OpenCLIP-bigG, eps-pred, VAE
    // scaling_factor 0.13025), so both share the `sdxl` engine via a weights swap. Upstream ships a
    // SINGLE-FILE LDM checkpoint, which no gen crate can read — the turnkeys are built offline by
    // `scripts/build_sdxl_turnkey.py` (sc-10610), not converted at install.
    //
    // v1.0 and v2.0 are SEPARATE ids, not a version toggle: v2.0 is the `v2.0-STABLE` snapshot of a
    // cosine-annealing run, behaviourally distinct, and it duplicates the subject in wide frames
    // where v1.0 does not (sc-10620 — hence their different `limits.resolutions`).
    ModelRow {
        sceneworks_id: "illustrious_xl_v1",
        engine_id: "sdxl",
        default_repo: "SceneWorks/illustrious-xl-v1-mlx",
        default_steps: 30,
        default_guidance: 7.0,
        adapter_label: "mlx_sdxl",
    },
    ModelRow {
        sceneworks_id: "illustrious_xl_v2",
        engine_id: "sdxl",
        default_repo: "SceneWorks/illustrious-xl-v2-mlx",
        default_steps: 30,
        default_guidance: 7.0,
        adapter_label: "mlx_sdxl",
    },
    // RealVisXL Lightning (sc-6075) — standalone few-step *distilled* sibling of RealVisXL_V5.0.
    // Same SDXL arch, so it shares the `sdxl` engine via a weights swap; differs only in the
    // distilled checkpoint + the few-step recipe: ~5 steps at guidance 1.0 (CFG off). The
    // distillation is baked into the checkpoint (no acceleration LoRA), and the worker pins the
    // engine's `lightning` Euler-trailing sampler for this id (see `generate_stream`). txt2img only
    // (the accel sampler is engine-incompatible with reference/img2img conditioning).
    ModelRow {
        sceneworks_id: "realvisxl_lightning",
        engine_id: "sdxl",
        // SceneWorks pre-built quant-matrix turnkey (sc-8746, epic 8506, Group-B): standard
        // q4/q8/bf16 subdirs (standard_tier_subdir). Re-host of SG161222/RealVisXL_V5.0_Lightning.
        default_repo: "SceneWorks/realvisxl-lightning-mlx",
        default_steps: 5,
        default_guidance: 1.0,
        adapter_label: "mlx_sdxl",
    },
    // Kolors (epic 3090, sc-3875) — Kwai-Kolors SDXL-architecture U-Net + ChatGLM3-6B text
    // encoder + SDXL VAE, EulerDiscrete sampler. Real CFG (negative prompt + guidance 5.0).
    // Python `MODEL_TARGETS` / `KolorsDiffusersAdapter` parity: 25 steps, guidance 5.0. The engine
    // `kolors` model (sc-3874) supports the full surface — img2img / ControlNet-pose /
    // IP-Adapter-Plus / Q8/Q4 / LoRA/LoKr — but this base row drives plain T2I (+ quant + LoRA)
    // through `generate_stream`; the advanced conditioning modes are gated to torch by
    // `kolors_mlx_eligible` until their dedicated streams land (subsequent epic-3090 slices).
    ModelRow {
        sceneworks_id: "kolors",
        // sc-9946 (epic 8506): flipped to the SceneWorks re-host `SceneWorks/kolors-mlx`, which ships
        // the pre-quantized q4/q8/bf16 turnkey tiers (mlx-gen #659). Was upstream
        // `Kwai-Kolors/Kolors-diffusers` (dense + install-time quant). The derived fast tokenizer is
        // baked into every tier, so the install-time tokenizer overlay is no longer needed here.
        engine_id: "kolors",
        default_repo: "SceneWorks/kolors-mlx",
        default_steps: 25,
        default_guidance: 5.0,
        adapter_label: "mlx_kolors",
    },
    // Chroma (epic 3531, sc-3843) — FLUX.1-schnell-derived DiT, T5-only conditioning. The engine
    // is a TRUE-CFG family: its descriptor advertises `supports_guidance=false` +
    // `supports_negative_prompt=true`, so the CFG scale is forwarded as `true_cfg` (NOT the
    // distilled `guidance` scalar, which the engine rejects) — see [`uses_true_cfg`] /
    // [`resolve_true_cfg`]. HD/Base are full true-CFG (the manifest pre-fills 40 steps + guidance
    // 3.0; the engine's own defaults are 28 steps + 4.0 — the request carries the manifest values).
    // Each SceneWorks id maps 1:1 to the engine registry id of the same name.
    ModelRow {
        sceneworks_id: "chroma1_hd",
        engine_id: "chroma1_hd",
        // SceneWorks pre-built quant-matrix turnkey (sc-8777, epic 8506, Group-B): standard
        // q4/q8/bf16 subdirs (standard_tier_subdir) — transformer packed, T5-XXL + VAE dense.
        // Public/ungated re-host of lodestones/Chroma1-HD.
        default_repo: "SceneWorks/chroma1-hd-mlx",
        default_steps: 40,
        default_guidance: 3.0,
        adapter_label: "mlx_chroma",
    },
    ModelRow {
        sceneworks_id: "chroma1_base",
        engine_id: "chroma1_base",
        // SceneWorks pre-built quant-matrix turnkey (sc-8777, epic 8506, Group-B): standard
        // q4/q8/bf16 subdirs (standard_tier_subdir) — transformer packed, T5-XXL + VAE dense.
        // Public/ungated re-host of lodestones/Chroma1-Base.
        default_repo: "SceneWorks/chroma1-base-mlx",
        default_steps: 40,
        default_guidance: 3.0,
        adapter_label: "mlx_chroma",
    },
    // Flash is the few-step distilled checkpoint: ~12 Heun steps, CFG baked toward 1.0 (single forward —
    // the negative prompt is effectively inert at true_cfg≈1). It shares the true-CFG descriptor,
    // so `true_cfg` still carries the scale (default 1.0).
    ModelRow {
        sceneworks_id: "chroma1_flash",
        engine_id: "chroma1_flash",
        // SceneWorks pre-built quant-matrix turnkey (sc-8777, epic 8506, Group-B): standard
        // q4/q8/bf16 subdirs (standard_tier_subdir) — transformer packed, T5-XXL + VAE dense.
        // Public/ungated re-host of lodestones/Chroma1-Flash.
        default_repo: "SceneWorks/chroma1-flash-mlx",
        default_steps: 12,
        default_guidance: 1.0,
        adapter_label: "mlx_chroma",
    },
    // SenseNova-U1 (epic 3180, sc-3900) — NEO-Unify: a dense dual-path Qwen3-MoT AR LLM + a
    // flow-matching image generator (no separate VAE / text encoder). Unlike every other family
    // here it uses BOTH CFG knobs: the descriptor's `supports_guidance=true` carries the text CFG
    // via `guidance` (defaults 4.0 base / 1.0 fast), and `supports_true_cfg` carries the it2i
    // image-guidance via `true_cfg` (edit ≈ 1.0 / character ≈ 1.5) — so it is NOT a
    // [`uses_true_cfg`] family (which is for engines that read the *single* CFG knob from
    // `true_cfg`). The descriptor advertises no negative prompt. Plain T2I rides
    // [`generate_stream`]; edit (`Reference`) + Character Studio (`MultiReference`) divert to
    // [`generate_sensenova_edit_stream`] where the dual CFG + reference conditioning are built.
    // `_fast` is the same base weights with the 8-step distill LoRA merged internally at load
    // (`load_fast`); the worker only selects the engine id, the engine resolves + merges the
    // curated distill LoRA itself (no user LoRA slot — `supports_lora=false`). Both ids map 1:1 to
    // the engine registry id of the same name.
    ModelRow {
        sceneworks_id: "sensenova_u1_8b",
        engine_id: "sensenova_u1_8b",
        // sc-8771: SceneWorks MLX quant-matrix re-host (q4/q8/bf16 packed tiers, mlx-gen #623).
        default_repo: "SceneWorks/sensenova-u1-8b-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_sensenova",
    },
    ModelRow {
        // Infographic-V2 (epic 9959): coexisting checkpoint refresh of the base NEO-unify model.
        // Its config + tensor layout are byte-identical to the base, so it rides the SAME engine
        // (`engine_id: "sensenova_u1_8b"`) with no engine change — only a distinct SceneWorks
        // quant-matrix re-host (q4/q8/bf16 packed tiers, epic 9959 S1). Same defaults as the base.
        sceneworks_id: "sensenova_u1_8b_infographic_v2",
        engine_id: "sensenova_u1_8b",
        default_repo: "SceneWorks/sensenova-u1-8b-infographic-v2-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_sensenova",
    },
    ModelRow {
        sceneworks_id: "sensenova_u1_8b_fast",
        engine_id: "sensenova_u1_8b_fast",
        // sc-8775: SceneWorks MLX quant-matrix re-host of the *distilled* variant — q4/q8/bf16 packed
        // tiers with the 8-step distill LoRA PRE-MERGED into the generation path at convert time (a
        // distinct checkpoint from the base re-host). Each tier carries a `distill_merged.json` marker
        // so the engine's `load_fast` loads it directly without re-merging (a packed base can't
        // re-merge). Replaces the old dense-base + distill-LoRA-at-load path.
        default_repo: "SceneWorks/sensenova-u1-8b-fast-mlx",
        default_steps: 8,
        default_guidance: 1.0,
        adapter_label: "mlx_sensenova",
    },
    ModelRow {
        // Infographic-V2 8-step distilled variant (epic 9959, sc-9963): the V1 distill LoRA merges
        // cleanly onto V2 (296/296 gen-path targets) and renders coherent 8-step infographics. Same
        // pre-merged + packed layout as the base fast (distill_merged.json marker → load_fast skip),
        // so it rides the SAME `sensenova_u1_8b_fast` engine id — only the re-host repo differs.
        sceneworks_id: "sensenova_u1_8b_infographic_v2_fast",
        engine_id: "sensenova_u1_8b_fast",
        default_repo: "SceneWorks/sensenova-u1-8b-infographic-v2-fast-mlx",
        default_steps: 8,
        default_guidance: 1.0,
        adapter_label: "mlx_sensenova",
    },
    // Microsoft Lens / Lens-Turbo (epic 3164 engine / sc-5105 cutover) — gpt-oss-20b MoE text
    // encoder + 48-layer dual-stream MMDiT + the Flux.2 VAE. Pure **T2I** (the descriptor advertises
    // no conditioning — no img2img / ControlNet / IP), so both ids ride the base [`generate_stream`]
    // path with quant (Q8 default) + LoRA/LoKr. Standard guidance family: `supports_guidance=true` +
    // `supports_negative_prompt=true` (NOT [`uses_true_cfg`]), so the CFG scale flows through
    // `guidance` and the negative prompt is forwarded. `mac_only` — there is no torch fallback on the
    // macOS path (the Python `/opt/lens-venv` sidecar is retired on Mac; Win/Linux/Docker keep it).
    // The two SceneWorks ids map 1:1 to the engine registry ids of the same name and differ only in
    // their step/guidance defaults: base `lens` is 20-step / CFG 5.0, distilled `lens_turbo` is
    // 4-step / guidance 1.0 (≈ no CFG) — Python `MODEL_TARGETS` parity. Each variant resolves its own
    // SceneWorks re-host: base `lens` → `SceneWorks/lens-mlx` (recovered/rebuilt from Comfy-Org/Lens,
    // sc-8767; original microsoft/Lens source is dead), distilled `lens_turbo` → `SceneWorks/lens-turbo-mlx`.
    // Both are pre-quantized per-tier turnkey snapshots (q4/q8/bf16, standardTierLayout).
    ModelRow {
        sceneworks_id: "lens",
        engine_id: "lens",
        default_repo: "SceneWorks/lens-mlx",
        default_steps: 20,
        default_guidance: 5.0,
        adapter_label: "mlx_lens",
    },
    ModelRow {
        sceneworks_id: "lens_turbo",
        engine_id: "lens_turbo",
        default_repo: "SceneWorks/lens-turbo-mlx",
        default_steps: 4,
        default_guidance: 1.0,
        adapter_label: "mlx_lens",
    },
    // Bernini still-image companion (epic 4699 / sc-5424) — the image-typed catalog id maps to the
    // SAME engine registry id (`bernini`) the video `bernini` id uses (`Modality::Both`), mirroring
    // the `z_image_edit → z_image_turbo` two-id/one-engine row above. The dedicated
    // `generate_bernini_image_stream` path (image_jobs/bernini.rs) builds the engine request itself
    // — forcing `frames:1` + `video_mode:"t2i"|"i2i"` so the engine returns a single still — so it
    // does NOT ride the generic `generate_stream`; this row supplies the `mlx_model` join the worker
    // uses for `adapter_id` / `mlx_weights_gap` / the descriptor-capability lookup. Engine defaults:
    // 40 steps, guidance (omega_txt) 4.0 (mlx-gen-bernini `FullDefaults`). No LoRA (descriptor
    // `supports_lora: false`). `default_repo` is the turnkey snapshot, but the dedicated path
    // resolves the dir via `resolve_bernini_model_dir` (env / app-managed / download), not this repo.
    ModelRow {
        sceneworks_id: "bernini_image",
        engine_id: "bernini",
        default_repo: "SceneWorks/bernini-mlx",
        default_steps: 40,
        default_guidance: 4.0,
        adapter_label: "mlx_bernini",
    },
    // Boogu-Image-0.1 (epic 6387) — native MLX, ungated (Apache-2.0). ~10.3B Lumina-Image-2.0 /
    // OmniGen2-lineage flow-matching DiT + Qwen3-VL-8B condition encoder + FLUX.1 VAE. Three variants,
    // one engine crate (`mlx-gen-boogu`); each id maps 1:1 to its gen_core descriptor id. The turnkey
    // `SceneWorks/boogu-image-mlx` ships pre-packed Q8 `base/ turbo/ edit/` subfolders (default) +
    // `*-bf16/`; `resolve_boogu_model_dir` (image_jobs/base.rs) points the engine at the variant
    // subfolder. The packed weights auto-detect their quant on load, so the worker's Q8 quant spec is
    // a no-op there. Base = true-CFG T2I (50 steps / guidance 4.0).
    ModelRow {
        sceneworks_id: "boogu_image",
        engine_id: "boogu_image",
        default_repo: "SceneWorks/boogu-image-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_boogu",
    },
    // Boogu Turbo — the DMD few-step, CFG-free distilled variant (`turbo/` checkpoint). 4 steps;
    // guidance is INERT (the `boogu_image_turbo` descriptor advertises supports_guidance=false, so
    // `resolve_guidance` returns None and never forwards a value).
    ModelRow {
        sceneworks_id: "boogu_image_turbo",
        engine_id: "boogu_image_turbo",
        default_repo: "SceneWorks/boogu-image-mlx",
        default_steps: 4,
        default_guidance: 0.0,
        adapter_label: "mlx_boogu",
    },
    // Boogu Edit — instruction image-edit (`edit/` checkpoint). The source image is read by the
    // Qwen3-VL vision tower + VAE-encoded into the DiT's spatial reference latent; the prompt is the
    // edit instruction. true-CFG (50 steps / guidance 4.0). The worker's `resolve_boogu_edit` builds
    // the source `Conditioning::Reference` (no mask path).
    ModelRow {
        sceneworks_id: "boogu_image_edit",
        engine_id: "boogu_image_edit",
        default_repo: "SceneWorks/boogu-image-mlx",
        default_steps: 50,
        default_guidance: 4.0,
        adapter_label: "mlx_boogu",
    },
    // Krea 2 Turbo (epic 7565) — native MLX, CFG-free rectified-flow few-step T2I (Qwen3-VL-4B TE +
    // 28-block single-stream DiT + Qwen-Image VAE). 8 steps; guidance is INERT (the `krea_2_turbo`
    // descriptor advertises supports_guidance=false, so `resolve_guidance` returns None and never
    // forwards a value). Loads the packed Q8 (default) / Q4 turnkey subdir (`krea_model_subdir`).
    ModelRow {
        sceneworks_id: "krea_2_turbo",
        engine_id: "krea_2_turbo",
        default_repo: "SceneWorks/krea-2-turbo-mlx",
        default_steps: 8,
        default_guidance: 0.0,
        adapter_label: "mlx_krea",
    },
    // Krea 2 Raw (epic 9992) — native MLX, the undistilled 12B DiT run with TRUE classifier-free guidance
    // (the `krea_2_raw` descriptor advertises supports_guidance + supports_negative_prompt, unlike the
    // CFG-free Turbo). 52 steps / guidance 3.5. Loads the packed bf16 / Q8 (default) / Q4 turnkey subdir
    // (`SceneWorks/krea-2-raw-mlx`, the same `krea_model_subdir` resolver as Turbo). Shares the Krea
    // pipeline with Turbo (arch-identical); the `krea_2_raw` id is also the LoRA-training base (Path 1).
    ModelRow {
        sceneworks_id: "krea_2_raw",
        engine_id: "krea_2_raw",
        default_repo: "SceneWorks/krea-2-raw-mlx",
        default_steps: 52,
        default_guidance: 3.5,
        adapter_label: "mlx_krea",
    },
    // Stable Diffusion 3.5 Large (epic 7841 / sc-7871) — native MLX, gated. 8B MMDiT + triple text
    // encoder (CLIP-L + CLIP-G + T5-XXL) + 16-ch VAE. True-CFG flagship: 28 steps / guidance 3.5 +
    // negative prompt (the `sd3_5_large` descriptor advertises supports_guidance + supports_negative
    // + supports_true_cfg). Installs a packed Q8 dir (`sd3_5_large_quant` converter, model_jobs.rs).
    ModelRow {
        sceneworks_id: "sd3_5_large",
        engine_id: "sd3_5_large",
        // SceneWorks pre-built quant-matrix turnkey (sc-8513): standard_tier_subdir points the engine
        // at the chosen tier subdir (q4 default / q8 / bf16). Re-host of the gated Stability source.
        default_repo: "SceneWorks/sd3.5-large-mlx",
        default_steps: 28,
        default_guidance: 3.5,
        adapter_label: "mlx_sd3",
    },
    // SD3.5 Large Turbo (epic 7841 / sc-7871) — the ADD-distilled few-step, CFG-free sibling: same 8B
    // MMDiT + triple TE + 16-ch VAE backbone + snapshot layout, distilled checkpoint. 4 steps; guidance
    // is INERT (the `sd3_5_large_turbo` descriptor advertises supports_guidance=false, so
    // `resolve_guidance` returns None and never forwards a value — the pipeline's `denoise_cfg` skips the
    // uncond forward at guidance 1.0). No negative prompt. The z_image_turbo / boogu / krea turbo pattern.
    ModelRow {
        sceneworks_id: "sd3_5_large_turbo",
        engine_id: "sd3_5_large_turbo",
        // SceneWorks pre-built quant-matrix turnkey (sc-8513): q4 default / q8 / bf16 via
        // standard_tier_subdir. Re-host of the gated Stability source.
        default_repo: "SceneWorks/sd3.5-large-turbo-mlx",
        default_steps: 4,
        default_guidance: 0.0,
        adapter_label: "mlx_sd3",
    },
    // SD3.5 Medium (epic 7841 / sc-7869 M3, wired in sc-7871) — the MMDiT-X variant: 2.5B, 24 joint
    // blocks (first 13 dual-attention), hidden 1536, `pos_embed_max_size` 384. True-CFG like Large but a
    // distinct (smaller) transformer + its own recipe — 40 steps / guidance 5.0 (Stability's card notes
    // Medium is more guidance-sensitive than Large). The `sd3_5_medium` descriptor advertises
    // supports_guidance + supports_negative + supports_true_cfg. Installs a packed dir via the
    // `sd3_5_medium_quant` converter (model_jobs.rs). `runtime-macos` explicitly includes the M3
    // registration in its platform catalog.
    ModelRow {
        sceneworks_id: "sd3_5_medium",
        engine_id: "sd3_5_medium",
        // SceneWorks pre-built quant-matrix turnkey (sc-8513): q4 default / q8 / bf16 via
        // standard_tier_subdir. Re-host of the gated Stability source.
        default_repo: "SceneWorks/sd3.5-medium-mlx",
        default_steps: 40,
        default_guidance: 5.0,
        adapter_label: "mlx_sd3",
    },
    // SANA 1600M 1024px (epic 8485 / sc-8489) — native MLX, NVIDIA non-commercial. NVIDIA's efficient
    // Linear-DiT (ReLU linear-attn + Mix-FFN + NoPE) 1.6B trunk + a gemma-2-2b-it CHI caption encoder +
    // the 32× DC-AE (f32) decoder. True-CFG text-to-image: 20 steps / guidance 4.5 + negative prompt
    // (the `sana_1600m` descriptor advertises supports_guidance + supports_negative + supports_true_cfg).
    // Loads the un-gated `SceneWorks/Sana_1600M_1024px_mlx` MLX snapshot (transformer/ vae/ text_encoder/,
    // the latter bundling the SceneWorks/gemma-2-2b-it TE so the load path resolves one snapshot dir —
    // SanaTextEncoder::from_snapshot reads `<dir>/text_encoder/gemma-2-2b-it.safetensors` + tokenizer.json).
    // The runtime catalog exposes this through the generic MODEL_TABLE / `generate_stream` path.
    // Quant matrix (sc-8489/sc-8513): ships
    // pre-packed q4/q8/bf16 tiers (transformer + Gemma-2 TE packed, DC-AE VAE dense), packed-detected
    // on load — NOT the (unported) 2-bit SANA quant. 32× DC-AE divisor → W/H must be multiples of 32.
    ModelRow {
        sceneworks_id: "sana_1600m",
        engine_id: "sana_1600m",
        default_repo: "SceneWorks/Sana_1600M_1024px_mlx",
        default_steps: 20,
        default_guidance: 4.5,
        adapter_label: "mlx_sana",
    },
    // SANA-Sprint 1.6B 1024px (epic 8485 / sc-8490) — the few-step distillation of SANA over the SAME
    // Linear-DiT trunk + gemma-2-2b-it CHI encoder + 32× DC-AE decoder, ported natively to mlx-gen (engine
    // id `sana_sprint_1600m`, explicitly included by `runtime-macos`). Sprint
    // is CFG-FREE: a guidance scalar is folded into the trunk via a guidance-embedding (no negative-prompt
    // second pass) and sampled by the SCM (continuous-time consistency / trigflow) sampler — so it runs in
    // ~2 steps (the `sana_sprint_1600m` descriptor advertises NO supports_true_cfg / supports_negative).
    // Loads the un-gated `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` MLX snapshot (same transformer/ vae/
    // text_encoder/ layout as base SANA; the text_encoder/ bundles the SceneWorks/gemma-2-2b-it TE so it is
    // NOT duplicated). Quant matrix (sc-8490/sc-8513): pre-packed q4/q8/bf16 tiers, packed-detected on
    // load. 32× DC-AE divisor → width/height multiples of 32. NVIDIA non-commercial (NSCLv1) — the
    // re-host carries the upstream LICENSE + NOTICE.
    ModelRow {
        sceneworks_id: "sana_sprint_1600m",
        engine_id: "sana_sprint_1600m",
        default_repo: "SceneWorks/Sana_Sprint_1.6B_1024px_mlx",
        default_steps: 2,
        default_guidance: 4.5,
        adapter_label: "mlx_sana",
    },
    // Anima 2B anime t2i (epic 10512, sc-10523) — native MLX, CircleStone Labs Non-Commercial License
    // v1.2. Cosmos-Predict2 `CosmosTransformer3DModel` DiT (28 layers, 17-ch patch-embed, 3-axis NTK
    // RoPE) + the bundled `AnimaTextConditioner` (T5 query tokens → cross-attn into Qwen3-0.6B states)
    // + the Qwen-Image VAE. Three variants share ONE architecture, differing only in the DiT weights
    // file + defaults. **Convert-at-install** (NC — SceneWorks never redistributes converted weights):
    // the worker packs the Cosmos DiT on-device to q4/q8/bf16 from the ungated `circlestone-labs/Anima`
    // `split_files/` source (the conditioner + Qwen3 TE + Qwen-Image VAE stay dense bf16), so
    // `default_repo` is that source repo, not a SceneWorks re-host. The engine descriptor advertises the
    // full curated sampler/scheduler menu (er_sde default, sc-10519); the manifest `limits` menu is a
    // subset (the drift guard). All three reach the generic `generate_stream` path via
    // `runtime-macos` catalog. supports_lora/lokr = true; quant + LoRA together is unsupported (sc-10578).
    ModelRow {
        sceneworks_id: "anima_base",
        engine_id: "anima_base",
        default_repo: "circlestone-labs/Anima",
        default_steps: 30,
        default_guidance: 4.5,
        adapter_label: "mlx_anima",
    },
    ModelRow {
        sceneworks_id: "anima_aesthetic",
        engine_id: "anima_aesthetic",
        default_repo: "circlestone-labs/Anima",
        default_steps: 30,
        default_guidance: 4.5,
        adapter_label: "mlx_anima",
    },
    // Turbo — the merged CFG-free few-step student: 10 steps, guidance INERT (the descriptor advertises
    // supports_guidance=false, so `resolve_guidance` returns None; the 1.0 default is a nominal no-op).
    ModelRow {
        sceneworks_id: "anima_turbo",
        engine_id: "anima_turbo",
        default_repo: "circlestone-labs/Anima",
        default_steps: 10,
        default_guidance: 1.0,
        adapter_label: "mlx_anima",
    },
];

/// The mlx-gen registry ids of the video generators this worker serves (the engine ids
/// `wan_engine_id` / `ltx_engine_id` / `svd_engine_id` map TO). All-targets so the
/// registry-derived advertisement ([`registry_capabilities`]) can classify a `Video`
/// descriptor without the macOS-only video dispatch in scope.
pub(crate) const VIDEO_ENGINE_IDS: &[&str] = &[
    "wan2_2_ti2v_5b",
    "wan2_2_t2v_14b",
    "wan2_2_i2v_14b",
    "ltx_2_3",
    // The candle LTX provider registers a distinct engine id (`ltx_2_3_distilled`, not the MLX
    // `ltx_2_3`); listed here so the registry-derived `video_generate` advertisement
    // ([`registry_capabilities`]) picks up the candle LTX descriptor too (sc-5097).
    "ltx_2_3_distilled",
    "svd_xt",
    // Mochi 1 (epic 1788 / sc-11991). ONE id covers BOTH backends — unlike LTX above, `mlx-gen-mochi`
    // and `candle-gen-mochi` both register `MODEL_ID = "mochi_1"`, so a single row is correct and a
    // `_distilled`-style sibling would resolve to nothing.
    "mochi_1",
];

/// The trainer registry ids this worker serves (the ids `engine_trainer_id` maps TO). Used by
/// [`registry_capabilities`] as the "is this a trainer the worker actually serves" filter: the
/// training capabilities light up when an enabled backend has a registered trainer whose id is one
/// of these. Trainer descriptors DO carry `backend` (sc-4906), so the derivation gates per-backend
/// — a candle trainer lights training up only under `backend_candle_enabled`, an mlx one only under
/// `backend_mlx_enabled` (see the gate in `registry_capabilities`). `lens` is the mlx (sc-5148) +
/// candle (sc-7817) Lens trainer. The mlx backend registers all of these; the candle backend only
/// the subset {`sdxl`, `z_image_turbo`, `lens`, `wan2_2_t2v_14b`} (the Wan 5B / I2V A14B + Kolors /
/// LTX have no candle trainer — `jobs_store::training_job_is_candle_eligible` keeps them off candle).
pub(crate) const TRAINER_IDS: &[&str] = &[
    "z_image_turbo",
    "sdxl",
    "kolors",
    "lens",
    // SD3.5 LoRA-training bases (epic 7841 T3 sc-7884): the engine registers the LoRA/LoKr trainer
    // under the same id as the inference generator of the training base — Large (sc-7883) and the
    // MMDiT-X Medium (sc-7885). mlx-only (no candle SD3 trainer; epic 7982).
    "sd3_5_large",
    "sd3_5_medium",
    "ltx_2_3",
    "wan2_2_ti2v_5b",
    "wan2_2_t2v_14b",
    "wan2_2_i2v_14b",
    // Anima (Cosmos-Predict2 DiT + AnimaTextConditioner; epic 10512, sc-10522): the `mlx-gen-anima`
    // trainer registers LoRA/LoKr under the same ids as the inference generators of the three variants
    // (base/aesthetic/turbo). mlx-only (no candle/torch Anima trainer). The trained adapter targets the
    // DiT AND the bundled `llm_adapter` conditioner (508 targets), applying back via `apply_anima_adapters`.
    "anima_base",
    "anima_aesthetic",
    "anima_turbo",
];

/// A [`ModelRow`] paired with the linked gen_core descriptor for its engine id — the merged
/// view the image path reads. The row supplies the worker-side defaults; the descriptor
/// supplies the capability surface (`supports_guidance` / `supports_negative_prompt` /
/// `backend`) so a row can never drift from the engine's own advertisement (sc-3723).
///
/// Compiled on the macOS MLX path AND the Windows candle lane (sc-5096): the join is purely
/// backend-neutral (`MODEL_TABLE` row + whichever provider crate registered the engine id), so the
/// candle `generate_candle_stream` reuses it exactly like the MLX `generate_stream` — `cfg(target_os)`
/// only decides which provider crate registered the descriptor, not how it is resolved.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) struct ResolvedModel {
    pub row: &'static ModelRow,
    pub descriptor: gen_core::ModelDescriptor,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
impl ResolvedModel {
    pub fn engine_id(&self) -> &'static str {
        self.row.engine_id
    }
    pub fn default_repo(&self) -> &'static str {
        self.row.default_repo
    }
    pub fn default_steps(&self) -> u32 {
        self.row.default_steps
    }
    pub fn default_guidance(&self) -> f32 {
        self.row.default_guidance
    }
    // The MLX adapter label (`mlx_<family>`). The candle lane reports `candle_<family>` via the free
    // `image_jobs::candle_adapter_label` instead, so this accessor is MLX-path-only — silence the
    // dead-code lint on the candle-only build where the macOS dispatch is cfg'd out.
    #[cfg_attr(
        all(not(target_os = "macos"), feature = "backend-candle"),
        allow(dead_code)
    )]
    pub fn adapter_label(&self) -> &'static str {
        self.row.adapter_label
    }
    /// Whether the engine accepts a guidance scale (descriptor-derived; distilled variants
    /// — z-image-turbo, flux schnell — are `false`).
    pub fn supports_guidance(&self) -> bool {
        self.descriptor.capabilities.supports_guidance
    }
    /// Whether the engine accepts a negative prompt / true CFG (descriptor-derived).
    pub fn supports_negative_prompt(&self) -> bool {
        self.descriptor.capabilities.supports_negative_prompt
    }
    /// Whether the engine advertises any Q4/Q8 quantization (descriptor-derived). The candle SDXL
    /// family advertises Q4/Q8 as of sc-10767 (it packed-detects the pre-quantized MLX tier from disk);
    /// the remaining sc-5096 candle families advertise none (dense only). Lens advertises Q4/Q8 (sc-5126).
    /// Used on BOTH lanes: the candle lane has always gated quant on this; the MLX lane gates on it
    /// too as of sc-8489 so SANA (the lone generic-MLX family with `supported_quants: &[]`, whose
    /// `load` rejects any quant) loads dense, while every pre-existing family (all Q4/Q8) is
    /// unaffected.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    pub fn supports_quant(&self) -> bool {
        !self.descriptor.capabilities.supported_quants.is_empty()
    }
    /// Whether the engine accepts LoRA/LoKr adapters (descriptor-derived). Lens is the first candle
    /// family to advertise either (sc-5126); the others advertise neither. Candle-lane-only for the
    /// same reason as [`Self::supports_quant`].
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    pub fn supports_adapters(&self) -> bool {
        self.descriptor.capabilities.supports_lora || self.descriptor.capabilities.supports_lokr
    }
    /// The tensor backend that registered this engine (`"mlx"` | `"candle"`).
    pub fn backend(&self) -> &'static str {
        self.descriptor.backend
    }
}

/// The engine-backed family for a SceneWorks model id, if any — the row joined with its
/// linked gen_core descriptor. `None` when the id is not in [`MODEL_TABLE`] or no provider
/// crate registered its engine id (keeps the existing fail-loud-when-not-MLX behavior).
///
/// Backend-neutral despite the `mlx_` name (sc-5096): on the Windows candle lane the registry holds
/// the candle descriptors, so this resolves the candle engine for `request.model` the same way it
/// resolves the MLX engine on macOS — the candle `generate_candle_stream` calls it directly.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn mlx_model(sceneworks_id: &str) -> Option<ResolvedModel> {
    let row = MODEL_TABLE
        .iter()
        .find(|r| r.sceneworks_id == sceneworks_id)?;
    let descriptor = crate::inference_runtime::generators()
        .map(|reg| (reg.descriptor)())
        .find(|d| d.id == row.engine_id)?;
    Some(ResolvedModel { row, descriptor })
}

/// The registry-DERIVED subset of the MLX worker's capabilities (sc-3723): exactly the
/// capabilities backed by a linked generator/trainer/captioner descriptor whose backend is
/// enabled in `settings`. Off-macOS the provider crates aren't linked, so the registry is
/// empty and this returns an empty vec — the capability is *absent*, not a runtime failure.
/// A future candle backend lights up here with zero worker changes: it registers descriptors
/// with `backend = "candle"` and (with `backend_candle_enabled`) they are picked up.
///
/// The carve-outs the worker advertises that are NOT expressible as a single registered
/// generator descriptor (ImageEdit/ImageDetail/Vqa/Interleave, the advanced video modes,
/// pose/kps/upscale/person detect+track) stay hardcoded at the [`crate::gpu::mlx_gpu`] call
/// site; this function returns only the descriptor-derived core.
pub(crate) fn registry_capabilities(
    settings: &crate::Settings,
) -> Vec<sceneworks_core::contracts::WorkerCapability> {
    registry_capabilities_from(
        settings,
        crate::inference_runtime::media(),
        crate::inference_runtime::text(),
    )
}

fn registry_capabilities_from(
    settings: &crate::Settings,
    media: &gen_core::ProviderRegistry,
    text: &gen_core::core_llm::TextLlmRegistry,
) -> Vec<sceneworks_core::contracts::WorkerCapability> {
    use sceneworks_core::contracts::WorkerCapability as Cap;

    let mut backends: Vec<&'static str> = Vec::new();
    if settings.backend_mlx_enabled {
        backends.push("mlx");
    }
    if settings.backend_candle_enabled {
        backends.push("candle");
    }

    let mut caps: Vec<Cap> = Vec::new();
    let push = |c: Cap, caps: &mut Vec<Cap>| {
        if !caps.contains(&c) {
            caps.push(c);
        }
    };

    for reg in media.generators() {
        let d = (reg.descriptor)();
        if !backends.contains(&d.backend) {
            continue;
        }
        let in_image = MODEL_TABLE.iter().any(|r| r.engine_id == d.id);
        let in_video = VIDEO_ENGINE_IDS.contains(&d.id);
        match d.modality {
            gen_core::Modality::Image if in_image => push(Cap::ImageGenerate, &mut caps),
            gen_core::Modality::Video if in_video => push(Cap::VideoGenerate, &mut caps),
            gen_core::Modality::Both => {
                if in_image {
                    push(Cap::ImageGenerate, &mut caps);
                }
                if in_video {
                    push(Cap::VideoGenerate, &mut caps);
                }
            }
            _ => {}
        }
    }

    // Trainers/captioners now carry `backend` (sc-4906), so gate them per-backend exactly like the
    // generators above — a candle-only trainer no longer lights up under `backend_mlx_enabled`
    // alone, and vice versa. `lora_train` (dry-run plan validation) and `lora_train_execute` (real
    // run) are both served in-process by the same trainer registry, so they light up together.
    if media.trainers().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && TRAINER_IDS.contains(&d.id)
    }) {
        push(Cap::LoraTrain, &mut caps);
        push(Cap::LoraTrainExecute, &mut caps);
    }
    // The JoyCaption captioner registers under the HF repo id (mlx-gen `JOY_CAPTION_MODEL_ID`),
    // not a short name.
    if media.captioners().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && d.id == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    }) {
        push(Cap::TrainingCaption, &mut caps);
    }
    // Dataset Doctor CLIP embedders (sc-6535/sc-6537): mlx-gen-clip registers paired image/text
    // embedders under `clip_vit_l14` + `clip_vit_l14_text`. Advertise `dataset_analysis` only when
    // both are registered on an enabled backend, so the worker cannot claim a caption-alignment job
    // with only half the CLIP pair linked.
    let has_clip_image = media.image_embedders().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && d.id == "clip_vit_l14"
    });
    let has_clip_text = media.text_embedders().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend) && d.id == "clip_vit_l14_text"
    });
    if has_clip_image && has_clip_text {
        push(Cap::DatasetAnalysis, &mut caps);
    }
    // Prompt-refinement (epic 7153). Both native lanes now run through the unified LLM engine — a
    // generic `core_llm::TextLlm` registered in core-llm's registry (NOT gen_core's), resolved
    // model-first: mlx-llm's `mlx-llama` on macOS (sc-7158), candle-llm's `candle-llama` on the
    // Windows/CUDA candle build (sc-7404). Light up `prompt_refine` when an enabled backend has a
    // core-llm text (non-vision) provider linked — the vision providers (mlx-joycaption / candle-llava)
    // set `supports_vision` and are excluded. The Python torch `PromptRefiner` stays the fallback on
    // platforms with neither.
    //
    // sc-8105: the `image_caption` task (reference image → Ideogram JSON caption) rides on this SAME
    // `PromptRefine` capability + job — it is a payload `task` discriminator, not a separate cap, so it
    // needs no capability-gate change. The gate below correctly keys on the WEIGHTLESS non-vision
    // descriptor because `image_caption` is served by the SAME text+Json provider as plain refinement:
    // `mlx-llama` statically advertises `supports_vision: false` + `[Constraint::Json]` and `can_load`s a
    // Qwen-VL (`qwen3_5`) snapshot, flipping `supports_vision` on only at LOAD time (mlx-llm
    // provider.rs:267). Its loaded `vision` tower then reads the `Content::Image` at GENERATE time.
    // Resolution itself must NOT demand vision (core-llm `select`/`meets` filters on the STATIC
    // descriptor, which has no vision+Json provider for a Qwen-VL snapshot); the worker resolves the
    // image_caption job on the JSON constraint alone (see `prompt_refine_jobs.rs`). Do NOT broaden this
    // gate to admit vision-only providers (e.g. `mlx-joycaption`) — they carry no constraints and would
    // over-advertise `prompt_refine` on a vision-only worker without serving the Json caption path.
    let native_prompt_refine = text.registrations().any(|r| {
        let d = (r.descriptor)();
        backends.contains(&d.backend.as_str()) && !d.capabilities.supports_vision
    });
    if native_prompt_refine {
        push(Cap::PromptRefine, &mut caps);
    }
    caps
}

#[cfg(test)]
mod tests {
    use super::*;
    use sceneworks_core::contracts::WorkerCapability as Cap;

    // A test Settings with the two backend toggles set; everything else is from_env defaults.
    // (Tests set no backend env vars, so from_env() yields mlx=on / candle=off by default; the
    // helper overrides both explicitly so each case is self-contained regardless of env.)
    fn settings_with_backends(mlx: bool, candle: bool) -> crate::Settings {
        let mut s = crate::Settings::from_env();
        s.backend_mlx_enabled = mlx;
        s.backend_candle_enabled = candle;
        s
    }

    // ── epic 7114 P5 / sc-7126 (+ sc-7432 bespoke coverage): manifest ⊆ engine drift guard ─────────
    // The builtin manifest's advertised sampler/scheduler menu for a model MUST be a subset of what
    // the linked engine actually honors on the ACTIVE backend, or the worker N3-falls the name back to
    // the default (sc-7127) — i.e. the UI offers a knob the engine silently ignores. This test parses
    // the embedded manifest and, for every image model (via `mlx_model` / MODEL_TABLE), every video
    // model (via `video_descriptor`, sc-7296), AND every bespoke out-of-MODEL_TABLE image model
    // (InstantID / PuLID via `bespoke_advertised`, sc-7432) with a source on the active backend, asserts
    // the per-backend-effective menu (base `limits` overridden by `<backend>.limits`) is honored. It
    // checks `mlx` on macOS (where the MLX provider crates are linked) and `candle` on the
    // `backend-candle` build — whichever registry is active — so each backend's truthfulness is enforced
    // on its own lane. `"default"` is the engine-default sentinel, always allowed.
    // All-targets (no cfg gate): the embedded manifest + jsonc strip live in `sceneworks_core`, which
    // compiles on every platform, so this parses on the plain Linux/Windows check lane too. The
    // sampler/scheduler drift guards below still gate themselves on macos/candle (they need a linked
    // provider registry), but the character_image engine-wiring guards (sc-9513) read only the manifest
    // + the declarative wiring table, so they run everywhere `cargo test` does — including CI's ubuntu lane.
    fn parse_builtin_models() -> serde_json::Value {
        let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1;
        let stripped = sceneworks_core::jsonc::strip_jsonc_comments(text);
        serde_json::from_str(&stripped).expect("parse builtin.models.jsonc")
    }

    // ── sc-9513 (F-059 follow-up of sc-8861): character_image ⇄ engine-wiring honesty guards ─────────
    //
    // These three guards were extracted to `tests/test_builtin_manifest_audit.py` by sc-8861 but still
    // cross-referenced the (now-deleted) Python worker's `MODEL_TARGETS` engine table via a lazy
    // `importorskip` — so once epic-8283 deleted `apps/worker` they would have degraded to a clean
    // SKIP and lost their coverage. They are reimplemented here against the Rust worker's own engine
    // wiring, reading the SAME embedded `config/manifests/builtin.models.jsonc` the Python audit parsed,
    // so the character_image ⇄ engine-declaration invariants keep running Python-free (and on CI's ubuntu
    // lane, since they need neither a linked provider registry nor a macos/candle build).
    //
    // Source of truth for the engine facts: the worker's character-image identity/pose engine wiring —
    // the bespoke identity/pose providers `resolve_image_route` / `resolve_candle_image_route`
    // (`image_jobs/base.rs`) dispatch to. That dispatch ladder is `#[cfg]`-gated to the macOS (MLX) and
    // `backend-candle` builds, so it is not even compiled on the plain check lane; this small declarative
    // table restates its identity/pose facts as all-targets data (the Rust analog of the Python
    // `ipAdapter`/`instantId`/`pulidFlux`/`controlNetPose` blocks). Keep it in sync when a model gains or
    // loses an identity/pose backbone:
    //   * `flux_dev`            → XLabs FLUX IP-Adapter          (image_jobs/flux_ipadapter.rs)
    //   * `sdxl` / `realvisxl`  → h94 IP-Adapter-plus-face        (image_jobs/sdxl_ipadapter.rs)
    //   * `kolors`              → Kolors IP-Adapter-Plus + pose CN (image_jobs/kolors_ipadapter.rs + kolors_control.rs)
    //   * `instantid_realvisxl` → InstantID IdentityNet           (image_jobs/instantid.rs)
    //   * `pulid_flux_dev`      → PuLID-FLUX face identity          (image_jobs/pulid.rs + pulid_candle.rs)
    // (Base `qwen_image` also carries a strict-pose ControlNet, but it is NOT a reference-identity engine
    // and its pose picker is gated by manifest `ui.poseLibrary` alone — not `character_image` — so it is
    // outside these identity-honesty guards and intentionally not a row here.)
    struct CharacterEngineWiring {
        sceneworks_id: &'static str,
        /// A dedicated reference-identity backbone (IP-Adapter / InstantID / PuLID-FLUX) — the Rust
        /// equivalent of Python `bool(MODEL_TARGETS[id].ipAdapter or .instantId or .pulidFlux)`.
        identity_engine: bool,
        /// The strict-pose ControlNet repo the model carries, if any (Python `controlNetPose.repo`).
        control_net_pose_repo: Option<&'static str>,
    }

    const CHARACTER_IMAGE_ENGINE_WIRING: &[CharacterEngineWiring] = &[
        CharacterEngineWiring {
            sceneworks_id: "flux_dev",
            identity_engine: true,
            control_net_pose_repo: None,
        },
        CharacterEngineWiring {
            sceneworks_id: "sdxl",
            identity_engine: true,
            control_net_pose_repo: None,
        },
        CharacterEngineWiring {
            sceneworks_id: "realvisxl",
            identity_engine: true,
            control_net_pose_repo: None,
        },
        // Illustrious-XL (epic 10609): character_image via the shared SDXL IP-Adapter lane
        // (is_sdxl_ipadapter_model), so identity_engine is true, like the rest of the plain SDXL
        // family. No strict-pose ControlNet.
        CharacterEngineWiring {
            sceneworks_id: "illustrious_xl_v1",
            identity_engine: true,
            control_net_pose_repo: None,
        },
        CharacterEngineWiring {
            sceneworks_id: "illustrious_xl_v2",
            identity_engine: true,
            control_net_pose_repo: None,
        },
        CharacterEngineWiring {
            sceneworks_id: "kolors",
            identity_engine: true,
            control_net_pose_repo: Some("Kwai-Kolors/Kolors-ControlNet-Pose"),
        },
        CharacterEngineWiring {
            sceneworks_id: "instantid_realvisxl",
            identity_engine: true,
            control_net_pose_repo: None,
        },
        CharacterEngineWiring {
            sceneworks_id: "pulid_flux_dev",
            identity_engine: true,
            control_net_pose_repo: None,
        },
    ];

    fn character_engine_wiring(id: &str) -> Option<&'static CharacterEngineWiring> {
        CHARACTER_IMAGE_ENGINE_WIRING
            .iter()
            .find(|row| row.sceneworks_id == id)
    }

    /// True iff the manifest `model` lists `capability` in its `capabilities` array.
    fn advertises_capability(model: &serde_json::Value, capability: &str) -> bool {
        model["capabilities"]
            .as_array()
            .is_some_and(|caps| caps.iter().any(|c| c.as_str() == Some(capability)))
    }

    // Guard 1 (was `test_character_image_capability_implies_engine_or_tuning_declaration`, sc-2018): every
    // builtin that advertises `character_image` must have EITHER a worker identity engine (IP-Adapter /
    // InstantID / PuLID-FLUX) OR a `ui.variationStrength` declaration. Otherwise the capability flag is
    // dishonest — the picker shows the model in "With character" mode but the worker silently ignores the
    // reference (the shape of z_image_turbo's pre-sc-2005 bug). The cross-backbone guard: a future
    // character_image backbone added without engine wiring fails here before it ever reaches a user.
    #[test]
    fn character_image_capability_implies_engine_or_tuning_declaration() {
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        let mut misleading: Vec<String> = Vec::new();
        for model in models {
            let Some(id) = model["id"].as_str() else {
                continue;
            };
            if !advertises_capability(model, "character_image") {
                continue;
            }
            let has_engine = character_engine_wiring(id).is_some_and(|w| w.identity_engine);
            let has_variation_ui = model
                .get("ui")
                .and_then(|ui| ui.get("variationStrength"))
                .is_some_and(|v| !v.is_null());
            if !(has_engine || has_variation_ui) {
                misleading.push(id.to_owned());
            }
        }
        assert!(
            misleading.is_empty(),
            "Models advertise `character_image` without an identity engine (IP-Adapter / InstantID / \
             PuLID-FLUX in CHARACTER_IMAGE_ENGINE_WIRING) or a `ui.variationStrength` declaration: {misleading:?}. \
             Wire an identity engine for a reference/face-ID backbone, or declare `ui.variationStrength` \
             for an edit-style backbone (sc-2017), or drop the capability flag (the z_image_turbo bug, sc-2005)."
        );
    }

    // Guard 2 (was `test_kolors_declares_strict_pose_controlnet`, sc-2264): Kolors is the strict pose
    // tier — the manifest must advertise `ui.poseLibrary` AND the worker wiring must carry the
    // Kolors-ControlNet-Pose repo so the pose picker offers it and the adapter can load the pose
    // ControlNet. Identity still rides the IP-Adapter; the pose path composes both.
    #[test]
    fn kolors_declares_strict_pose_controlnet() {
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        let kolors = models
            .iter()
            .find(|m| m["id"].as_str() == Some("kolors"))
            .expect("kolors manifest entry");
        assert_eq!(
            kolors
                .get("ui")
                .and_then(|ui| ui.get("poseLibrary"))
                .and_then(serde_json::Value::as_bool),
            Some(true),
            "kolors must declare ui.poseLibrary so the pose picker offers the strict tier (sc-2264)."
        );
        let wiring = character_engine_wiring("kolors").expect("kolors character-engine wiring");
        assert_eq!(
            wiring.control_net_pose_repo,
            Some("Kwai-Kolors/Kolors-ControlNet-Pose"),
            "kolors wiring must carry the Kolors-ControlNet-Pose repo for the strict pose path."
        );
        assert!(
            wiring.identity_engine,
            "kolors pose path needs the IP-Adapter for identity."
        );
    }

    // Guard 3 (was `test_models_with_engine_block_advertise_character_image`): the reverse-drift guard.
    // Any model that ships an identity engine exists to serve Character Studio's reference flow — the
    // manifest MUST advertise `character_image` so the picker surfaces it. Catches the case where someone
    // wires the worker engine but forgets to flip the manifest flag, leaving the engine unreachable.
    #[test]
    fn models_with_engine_block_advertise_character_image() {
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        let mut unreachable_ids: Vec<String> = Vec::new();
        for wiring in CHARACTER_IMAGE_ENGINE_WIRING {
            if !wiring.identity_engine {
                continue;
            }
            let Some(model) = models
                .iter()
                .find(|m| m["id"].as_str() == Some(wiring.sceneworks_id))
            else {
                // Identity engine wired but not exposed as a built-in (unwired path) — mirrors the
                // Python guard's `if builtin is None: continue`.
                continue;
            };
            if !advertises_capability(model, "character_image") {
                unreachable_ids.push(wiring.sceneworks_id.to_owned());
            }
        }
        assert!(
            unreachable_ids.is_empty(),
            "Models have an identity engine in CHARACTER_IMAGE_ENGINE_WIRING but the builtin manifest \
             does not advertise `character_image`: {unreachable_ids:?}. Add the capability to \
             `capabilities` and `ui.recommendedFor` so the Image Studio \"With character\" picker surfaces \
             the model."
        );
    }

    // The effective `limits[key]` list for `backend`: the per-backend `<backend>.limits[key]` override
    // if present, else the base `limits[key]`. `None` => the model advertises no list for that axis.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn effective_list(model: &serde_json::Value, backend: &str, key: &str) -> Option<Vec<String>> {
        let pick = |scope: &serde_json::Value| {
            scope
                .get("limits")
                .and_then(|l| l.get(key))
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_owned))
                        .collect::<Vec<_>>()
                })
        };
        model.get(backend).and_then(pick).or_else(|| pick(model))
    }

    /// The engine registry id(s) a video manifest model resolves to, across backends — the union of
    /// the mlx maps (`wan_engine_id` / `ltx_engine_id` / `svd_engine_id` + the native VACE-Fun
    /// dispatch) and the candle map (`candle_video_engine_id`) in `video_jobs`. LTX is backend-split
    /// (`ltx_2_3` on mlx, `ltx_2_3_distilled` on candle) and `ltx_2_3_eros` shares the base engine id
    /// per backend; the resolver lists both and picks whichever the active registry actually holds.
    /// `wan_2_2_vace_fun_14b` is mlx-only (candle has no VACE engine), so it resolves to `None` on the
    /// candle lane and is skipped there.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn video_engine_ids(sceneworks_id: &str) -> &'static [&'static str] {
        match sceneworks_id {
            "wan_2_2" => &["wan2_2_ti2v_5b"],
            "wan_2_2_t2v_14b" => &["wan2_2_t2v_14b"],
            "wan_2_2_i2v_14b" => &["wan2_2_i2v_14b"],
            "wan_2_2_vace_fun_14b" => &["wan2_2_vace_fun_14b"],
            "svd" => &["svd_xt"],
            "ltx_2_3" | "ltx_2_3_eros" => &["ltx_2_3", "ltx_2_3_distilled"],
            // Mochi 1 (sc-11991): the sceneworks id IS the engine id, and BOTH backends register it,
            // so one entry resolves the descriptor on either lane.
            "mochi_1" => &["mochi_1"],
            _ => &[],
        }
    }

    /// The linked gen-core descriptor for a video manifest model on the ACTIVE backend, or `None`
    /// when no provider crate registered its engine id here. Mirrors [`mlx_model`]'s registry join
    /// for the video ids that live outside [`MODEL_TABLE`] (the image path).
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn video_descriptor(sceneworks_id: &str) -> Option<gen_core::ModelDescriptor> {
        let ids = video_engine_ids(sceneworks_id);
        if ids.is_empty() {
            return None;
        }
        crate::inference_runtime::generators()
            .map(|reg| (reg.descriptor)())
            .find(|d| ids.contains(&d.id))
    }

    /// The `(samplers, schedulers)` menu the engine actually honors for a **bespoke** image model that
    /// lives OUTSIDE [`MODEL_TABLE`] (no `mlx_model` row, no video engine id) — sc-7432. These build
    /// CUSTOM request structs the worker N3-normalizes against [`crate::image_jobs::curated_image_menu`]
    /// (`instantid.rs` / `kolors_*` / `pulid*`), so the guard checks the manifest against the SAME source
    /// of truth and the two never disagree:
    ///   • `instantid_realvisxl` is a bespoke provider (`InstantId::load`) with NO `ModelDescriptor`;
    ///     both engines (mlx #538 / candle #130) honor the curated solver vocab via `Solver::from_name` /
    ///     the additive `denoise_curated` path, so the honored menu IS the curated vocab.
    ///   • `pulid_flux_dev` is the inventory-registered `pulid_flux` Generator on mlx (a real descriptor
    ///     advertising curated + flow_match/linear); on the candle lane it is the bespoke `PulidFlux`
    ///     provider (no descriptor), so fall back to the curated vocab + FLUX's native flow names. Either
    ///     way a superset of the manifest's `default`+curated menu.
    /// Kolors-conditioned is NOT here: `kolors` IS a `MODEL_TABLE` row, so the loop already resolves it
    /// via `mlx_model` and the existing descriptor check covers its (shared) menu.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn bespoke_advertised(sceneworks_id: &str) -> Option<(Vec<String>, Vec<String>)> {
        let curated = || {
            let (samplers, schedulers) = crate::image_jobs::curated_image_menu();
            (
                samplers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                schedulers.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            )
        };
        match sceneworks_id {
            "instantid_realvisxl" => Some(curated()),
            "pulid_flux_dev" => crate::inference_runtime::generators()
                .map(|reg| (reg.descriptor)())
                .find(|d| d.id == "pulid_flux")
                .map(|d| {
                    (
                        d.capabilities
                            .samplers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                        d.capabilities
                            .schedulers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    )
                })
                .or_else(|| {
                    let (mut samplers, mut schedulers) = curated();
                    samplers.push("flow_match".to_string());
                    schedulers.push("linear".to_string());
                    Some((samplers, schedulers))
                }),
            _ => None,
        }
    }

    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn manifest_menu_is_subset_of_descriptor() {
        // The active backend's registry: MLX on macOS, candle on the `backend-candle` build.
        let backend = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        };
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        let mut violations: Vec<String> = Vec::new();
        for model in models {
            let Some(id) = model["id"].as_str() else {
                continue;
            };
            // The advertised sampler/scheduler menu the engine honors, from whichever source applies:
            // image models via MODEL_TABLE (`mlx_model`); video models via their engine-id map
            // (`video_descriptor`); the bespoke out-of-MODEL_TABLE image models (InstantID / PuLID,
            // sc-7432) via `bespoke_advertised`. A model with no source on the active backend is skipped
            // (e.g. the mlx-only `wan_2_2_vace_fun_14b` on the candle lane).
            let Some((adv_samplers, adv_schedulers, adv_guidance)) = mlx_model(id)
                .map(|resolved| resolved.descriptor)
                .or_else(|| video_descriptor(id))
                .map(|descriptor| {
                    (
                        descriptor
                            .capabilities
                            .samplers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                        descriptor
                            .capabilities
                            .schedulers
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                        // sc-7447: the guidance axis (epic 7434) — the manifest's per-backend
                        // `limits.guidanceMethods` MUST be a subset of what the engine descriptor
                        // honors, exactly like samplers/schedulers.
                        descriptor
                            .capabilities
                            .supported_guidance_methods
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>(),
                    )
                })
                // Bespoke out-of-MODEL_TABLE models advertise no descriptor guidance vocab; they
                // also advertise no `limits.guidanceMethods` in the manifest, so the guidance axis is
                // simply absent for them (empty advertised set + `effective_list` => None).
                .or_else(|| bespoke_advertised(id).map(|(s, sc)| (s, sc, Vec::new())))
            else {
                continue;
            };
            for (axis, advertised) in [
                ("samplers", &adv_samplers),
                ("schedulers", &adv_schedulers),
                ("guidanceMethods", &adv_guidance),
            ] {
                if let Some(list) = effective_list(model, backend, axis) {
                    for name in list {
                        if name == "default" {
                            continue;
                        }
                        if !advertised.iter().any(|advertised| advertised == &name) {
                            violations.push(format!(
                                "{id}: {backend} {axis} {name:?} not honored by the engine (advertised: {advertised:?})"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            violations.is_empty(),
            "manifest advertises {} sampler/scheduler/guidance name(s) the {backend} engine does not honor:\n  {}",
            violations.len(),
            violations.join("\n  ")
        );
    }

    // sc-7447: the subset guard above only EXERCISES the guidance axis if a model actually advertises a
    // `limits.guidanceMethods` list — otherwise the new axis is vacuously green (the same trap sc-7432
    // closed for the bespoke sampler menus). Pin the CFG++ surface down on the MLX lane: `sdxl` +
    // `realvisxl` (epic 7434 / sc-8256) must advertise `cfg_pp`, and the linked engine descriptor must
    // honor it. Candle has no cfg_pp dispatch yet (sc-8257), so the base `limits` carries no guidance
    // vocab and this is a macOS-only assertion — the candle lane keeps the standard CFG-only surface.
    #[cfg(target_os = "macos")]
    #[test]
    fn sdxl_family_advertises_cfgpp_on_mlx() {
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        for id in [
            "sdxl",
            "realvisxl",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
        ] {
            let model = models
                .iter()
                .find(|m| m["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("{id}: manifest entry must exist"));
            let methods = effective_list(model, "mlx", "guidanceMethods")
                .unwrap_or_else(|| panic!("{id}: must advertise mlx limits.guidanceMethods"));
            assert!(
                methods.iter().any(|m| m == "cfg_pp"),
                "{id}: mlx must advertise cfg_pp (advertised: {methods:?})"
            );
            let descriptor = mlx_model(id)
                .map(|r| r.descriptor)
                .unwrap_or_else(|| panic!("{id}: must resolve an mlx descriptor"));
            assert!(
                descriptor
                    .capabilities
                    .supported_guidance_methods
                    .contains(&"cfg_pp"),
                "{id}: engine descriptor must honor cfg_pp (advertised: {:?})",
                descriptor.capabilities.supported_guidance_methods
            );
        }
    }

    // sc-7432: the subset guard above only EXERCISES the bespoke out-of-MODEL_TABLE models if
    // `bespoke_advertised` resolves a menu for them — a `None` would silently skip them and leave the
    // guard vacuously green. Pin that down: both bespoke ids must resolve a non-empty menu that includes
    // the solvers their manifest entries advertise (euler/heun), on whichever backend is active.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn bespoke_models_resolve_a_curated_menu() {
        for id in ["instantid_realvisxl", "pulid_flux_dev"] {
            let (samplers, schedulers) = bespoke_advertised(id)
                .unwrap_or_else(|| panic!("{id}: bespoke_advertised must resolve an engine menu"));
            assert!(!samplers.is_empty(), "{id}: sampler menu must be non-empty");
            assert!(
                !schedulers.is_empty(),
                "{id}: scheduler menu must be non-empty"
            );
            // The manifest advertises euler + heun for both; the engine must honor them.
            for solver in ["euler", "heun"] {
                assert!(
                    samplers.iter().any(|advertised| advertised == solver),
                    "{id}: engine must honor {solver:?} (advertised: {samplers:?})"
                );
            }
        }
    }

    /// Every image model in the shipped `builtin.models.jsonc`. A model added, removed, or renamed
    /// without updating this list trips the count tripwire in
    /// [`shipped_image_geometry_is_within_the_pinned_engine_envelope`], so a new image model cannot
    /// silently ship with its advertised geometry unchecked against its engine (sc-12384 — the image
    /// twin of [`pinned_engine_geometry`]'s `EXPECTED_VIDEO_IDS`). Backend-independent: this is the
    /// full catalog set, and each id is size-checked on whichever backend(s) resolve its engine.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    const EXPECTED_IMAGE_IDS: &[&str] = &[
        "z_image_turbo",
        "z_image",
        "z_image_edit",
        "qwen_image",
        "qwen_image_edit_2511",
        "qwen_image_edit_2511_lightning",
        "lens",
        "lens_turbo",
        "sensenova_u1_8b",
        "sensenova_u1_8b_infographic_v2",
        "sensenova_u1_8b_fast",
        "sensenova_u1_8b_infographic_v2_fast",
        "flux_schnell",
        "flux_dev",
        "ideogram_4",
        "ideogram_4_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        "krea_2_turbo",
        "krea_2_raw",
        "flux2_klein_9b",
        "flux2_klein_9b_kv",
        "flux2_klein_9b_true_v2",
        "flux2_dev",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "kolors",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sana_1600m",
        "sana_sprint_1600m",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
        "sdxl",
        "realvisxl",
        "realvisxl_lightning",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "instantid_realvisxl",
        "pulid_flux_dev",
        "bernini_image",
    ];

    /// The backend-effective default resolution (`<backend>.defaults.resolution` overriding the base
    /// `defaults.resolution`) — the scalar companion to [`effective_list`], which handles the
    /// per-backend list axes. `None` when the model declares no default at all.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    fn effective_default_resolution(model: &serde_json::Value, backend: &str) -> Option<String> {
        let pick = |scope: &serde_json::Value| {
            scope
                .get("defaults")
                .and_then(|d| d.get("resolution"))
                .and_then(|r| r.as_str())
                .map(str::to_owned)
        };
        model.get(backend).and_then(pick).or_else(|| pick(model))
    }

    /// sc-12384 — the image-lane twin of [`shipped_manifest_matches_each_engines_real_geometry`]
    /// (sc-12294, video) and [`pinned_engine_geometry`] (sc-12409): every image model's advertised
    /// geometry — each `limits.resolutions` bucket AND the shipped `defaults.resolution` — must be
    /// LEGAL on the engine that actually ships, the one pinned by `Cargo.toml`'s `runtime-*` tag.
    ///
    /// The bug this closes: `bernini_image`'s own 1024² default was engine-rejected on candle and
    /// nothing caught it, because `limits.maxPixels` / `requiresDimensionsMultipleOf` are read only
    /// on the VIDEO request path (`video_request.rs`) — an image model's manifest buckets were an
    /// unchecked claim. This asserts them against the PINNED descriptor's `[min_size, max_size]`
    /// envelope (each provider sets those from its own `RES_MIN`/`RES_MAX` const), so a catalog
    /// bucket that overshoots an engine's envelope — OR a `runtime-*` pin bump that narrows one —
    /// is RED here instead of a silent job-time reject (candle) or wrong-aspect refit (mlx).
    ///
    /// It found two live over-advertisements when written: SANA (its per-side 1024 DC-AE envelope
    /// — the 1152/1216 buckets hard-error) and SenseNova (per-side 2048 — the 2720/2496/2368 buckets
    /// were silently squashed to a wrong aspect by `image_jobs::sensenova::sensenova_dim`). Both were
    /// trimmed catalog-side; the fix is the catalog, because each engine cap is deliberate (SANA's is
    /// test-pinned, `max_size_is_the_validated_1024_envelope`).
    ///
    /// Runs on whichever backend the current lane compiles — mlx on macOS CI, candle on the
    /// `backend-candle` Windows CI — each checking the binary its own platform ships. A model that
    /// registers on only one backend is size-checked on that backend's lane (every shipped image
    /// model registers on at least one); the count tripwire is backend-independent, so a new image
    /// model must still be assessed on both.
    ///
    /// SCOPE: the `[min, max]` size envelope — the axis `Capabilities` exposes, and the axis both
    /// live bugs sat on. The per-engine ÷8/÷16/÷32 STRIDE is checked inside each engine's `validate`
    /// and is NOT on `Capabilities`; every current image bucket is on-stride, so tying image strides
    /// to a pinned const (as sc-12409 did for the wan-14B family, and sc-12587 tracks for the
    /// remaining VIDEO strides) is a coverage extension tracked in sc-12612 — the image twin — not a
    /// live miss.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn shipped_image_geometry_is_within_the_pinned_engine_envelope() {
        let backend = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        };
        let manifest = parse_builtin_models();
        let models = manifest["models"].as_array().expect("models array");
        let image_models: Vec<&serde_json::Value> = models
            .iter()
            .filter(|m| m["type"].as_str() == Some("image"))
            .collect();

        // Count/rename tripwire (backend-independent): the shipped image set must be exactly
        // EXPECTED_IMAGE_IDS. A new or renamed image model is RED here until it is added — and thereby
        // consciously assessed against its engine's envelope — rather than shipping unguarded.
        let mut shipped_ids: Vec<&str> = image_models
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        shipped_ids.sort_unstable();
        let mut expected = EXPECTED_IMAGE_IDS.to_vec();
        expected.sort_unstable();
        assert_eq!(
            shipped_ids, expected,
            "the shipped image-model set changed — add the new/renamed id to EXPECTED_IMAGE_IDS and \
             confirm its advertised buckets + default fit its engine's [min_size, max_size] envelope \
             (sc-12384); do not let a new image model ship with unchecked geometry"
        );

        let mut checked = 0usize;
        for model in image_models {
            let id = model["id"].as_str().expect("image model has an id");
            // The PINNED engine's size envelope on THIS backend, or skip: a model that registers only
            // on the other backend (a mac-only edit lane on the candle lane) or runs through a bespoke
            // provider with no gen-core descriptor (InstantID; candle PuLID) is size-checked on the
            // lane that does resolve it — the count tripwire above still forces every id to be listed.
            let Some(resolved) = mlx_model(id) else {
                continue;
            };
            let min = resolved.descriptor.capabilities.min_size;
            let max = resolved.descriptor.capabilities.max_size;

            // The backend-effective advertised buckets, plus the backend-effective default (which the
            // picker preselects and sc-12400's `default_resolution` now hands the engine verbatim).
            let buckets = effective_list(model, backend, "resolutions").unwrap_or_default();
            let default = effective_default_resolution(model, backend);
            assert!(
                !buckets.is_empty() || default.is_some(),
                "{id}: resolves engine {:?} on {backend} but advertises no resolutions or default \
                 to check — an image model with a real engine must declare its geometry",
                resolved.engine_id()
            );
            // The default must itself be one of the advertised buckets — a default the picker can't
            // land on is a UI bug (mirrors the video guard's `default is one of its buckets` check).
            if let Some(default) = &default {
                assert!(
                    buckets.iter().any(|b| b == default),
                    "{id}: default resolution {default} is not one of its advertised buckets \
                     {buckets:?} — the picker preselects a size the user cannot re-select (sc-12384)"
                );
            }
            // Every advertised bucket AND the default is a claim the engine must honor: each must sit
            // inside the PINNED engine's [min_size, max_size] envelope, on both axes.
            for res in buckets.iter().chain(default.iter()) {
                let (w, h) = res
                    .split_once('x')
                    .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
                    .unwrap_or_else(|| panic!("{id}: malformed resolution {res:?}"));
                assert!(
                    (min..=max).contains(&w) && (min..=max).contains(&h),
                    "{id}: advertised {res} is outside its PINNED engine's size envelope \
                     [{min}, {max}] on the {backend} backend — the catalog advertises a bucket the \
                     shipped engine rejects (candle) or silently refits (mlx). Trim the catalog, or \
                     move the `runtime-*` pin and the catalog in lockstep (sc-12384)."
                );
            }
            checked += 1;
        }
        assert!(
            checked > 0,
            "no image model resolved an engine on the {backend} backend — the guard would be \
             vacuously green; a linked provider registry is required (macos/backend-candle)"
        );
    }

    /// sc-11991 (epic 1788): `mochi_1` resolves through THIS module to a real gen-core descriptor on
    /// the active backend — the story's acceptance criterion, asserted directly.
    ///
    /// [`manifest_menu_is_subset_of_descriptor`] cannot stand in for this: it `continue`s past any
    /// model whose descriptor does not resolve, so it stays green whether Mochi is wired or not.
    /// Mochi also advertises NO sampler/scheduler axis, which makes its subset check vacuous. This
    /// test fails if the `VIDEO_ENGINE_IDS` row, the `video_engine_ids` mapping, or the runtime pin
    /// regresses.
    ///
    /// It further pins the descriptor facts the manifest entry is derived from, so a runtime bump
    /// that changes them fails HERE rather than silently making the manifest lie.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn mochi_resolves_one_engine_id_to_the_gen_core_descriptor() {
        assert!(
            VIDEO_ENGINE_IDS.contains(&"mochi_1"),
            "mochi_1 must be in VIDEO_ENGINE_IDS or the registry-derived video_generate \
             advertisement drops it"
        );
        // ONE engine id covers both backends — unlike LTX's `ltx_2_3` + `ltx_2_3_distilled` split,
        // `mlx-gen-mochi` and `candle-gen-mochi` both register `MODEL_ID = "mochi_1"`.
        assert_eq!(
            video_engine_ids("mochi_1"),
            &["mochi_1"],
            "mochi maps to exactly one engine id on both backends"
        );

        let descriptor = video_descriptor("mochi_1")
            .expect("mochi_1 resolves to a gen-core descriptor on the active backend");
        assert_eq!(descriptor.id, "mochi_1");
        assert!(
            matches!(descriptor.modality, gen_core::Modality::Video),
            "mochi is a video generator"
        );

        let caps = &descriptor.capabilities;
        // The manifest advertises NO sampler/scheduler axis because the engine honors none (one
        // fixed flow-match Euler integrator). If this ever becomes non-empty, revisit `limits`.
        assert!(
            caps.samplers.is_empty() && caps.schedulers.is_empty(),
            "mochi advertises no sampler/scheduler axis"
        );
        // Tiers ship PRE-QUANTIZED as directories; `spec.quantize` only asserts the tier's baked-in
        // level. An empty `supported_quants` is what makes a tier a DIR rather than a requant toggle.
        assert!(
            caps.supported_quants.is_empty(),
            "mochi tiers are pre-quantized dirs, not an on-the-fly requant toggle"
        );
        // t2v only + no adapter path — the two facts the routing rows and `loraCompatibility` encode.
        assert!(
            caps.conditioning.is_empty(),
            "mochi is text-to-video only (no conditioning kinds)"
        );
        assert!(
            !caps.supports_lora && !caps.supports_lokr,
            "mochi has no LoRA/LoKr path on either backend"
        );
        // Everything stays resident: the basis for the manifest's mlx.minMemoryGb derivation.
        assert!(
            !caps.supports_sequential_offload,
            "mochi holds T5 + DiT + VAE resident (the minMemoryGb derivation assumes it)"
        );
        assert_eq!(caps.max_count, 1, "mochi renders one clip per run");
        // The manifest's `requiresDimensionsMultipleOf: 16` + the 848x480 bucket ride these.
        assert_eq!(caps.min_size, 16);
        assert_eq!(caps.max_size, 1280);
    }

    // Explicit test registrations exercise capability derivation without mutating a process-global
    // provider inventory. Their ids deliberately cover in-table, unknown, LLM, and trainer cases.
    fn stub_mlx_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "z_image_turbo",
            family: "test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }
    fn stub_mlx_load(_spec: &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    const STUB_MLX: gen_core::ModelRegistration = gen_core::ModelRegistration {
        descriptor: stub_mlx_descriptor,
        load: stub_mlx_load,
        footprint: None,
    };

    // A candle-backed stub whose id is also in MODEL_TABLE (`sdxl`): proves a Windows/candle
    // backend lights up `image_generate` with zero worker code changes once its backend is
    // enabled — purely by registering a descriptor with `backend = "candle"`.
    fn stub_candle_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "sdxl",
            family: "test",
            backend: "candle",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }
    fn stub_candle_load(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    const STUB_CANDLE: gen_core::ModelRegistration = gen_core::ModelRegistration {
        descriptor: stub_candle_descriptor,
        load: stub_candle_load,
        footprint: None,
    };

    // An MLX-backed stub whose id is NOT in MODEL_TABLE / VIDEO_ENGINE_IDS: proves an unknown
    // engine id contributes no capability (absence, not a runtime failure).
    fn stub_unknown_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "not_a_sceneworks_engine",
            family: "test",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: gen_core::Capabilities::default(),
        }
    }
    fn stub_unknown_load(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    const STUB_UNKNOWN: gen_core::ModelRegistration = gen_core::ModelRegistration {
        descriptor: stub_unknown_descriptor,
        load: stub_unknown_load,
        footprint: None,
    };

    // A candle-backed core-llm `TextLlm` stub (backend "candle", non-vision): proves the prompt-refine
    // derivation lights up `prompt_refine` purely from a registered `core_llm::TextLlm` descriptor on an
    // enabled backend (sc-7404), so the default (Linux) CI lane exercises it without linking a real
    // provider crate. The real lanes register mlx-llama / candle-llama into this SAME core-llm registry.
    fn stub_textllm_descriptor() -> gen_core::core_llm::TextLlmDescriptor {
        gen_core::core_llm::TextLlmDescriptor {
            id: "prompt_refine".to_string(),
            family: "llama".to_string(),
            backend: "candle".to_string(),
            capabilities: gen_core::core_llm::TextLlmCapabilities::default(),
        }
    }
    fn stub_textllm_load(
        _spec: &gen_core::core_llm::LoadSpec,
    ) -> gen_core::core_llm::Result<Box<dyn gen_core::core_llm::TextLlm>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    fn stub_textllm_can_load(_spec: &gen_core::core_llm::LoadSpec) -> bool {
        false
    }
    const STUB_TEXT_LLM: gen_core::core_llm::TextLlmRegistration =
        gen_core::core_llm::TextLlmRegistration {
            descriptor: stub_textllm_descriptor,
            load: stub_textllm_load,
            can_load: stub_textllm_can_load,
            weightless_vision: None,
        };

    // A candle-backed stub `Trainer` (backend "candle") registered under an id that IS in TRAINER_IDS
    // (`sdxl`): proves a Windows/candle backend lights up `lora_train` + `lora_train_execute` from a
    // registered `backend = "candle"` trainer descriptor alone (sc-7817), so the CI lane exercises the
    // per-backend training gate without linking a real provider crate.
    //
    // Registered in EVERY build. Under the canonical `inference` runtime `ProviderRegistryBuilder::new()`
    // is an empty explicit registry — nothing self-registers by being linked — so this test's hand-built
    // `media` never receives the real `sdxl` trainer, and the backend-candle lane must supply its own
    // stub or the training caps never light up. The stub lives ONLY in this test-local registry and never
    // reaches the global `inference_runtime` catalog the GPU smokes load from, so the old first-wins
    // `load_trainer("sdxl")` collision worry (real vs `unimplemented!()` stub) no longer applies.
    fn stub_candle_trainer_descriptor() -> gen_core::TrainerDescriptor {
        gen_core::TrainerDescriptor {
            id: "sdxl",
            family: "test",
            backend: "candle",
            modality: gen_core::Modality::Image,
            supports_lora: true,
            supports_lokr: true,
            supports_control: false,
        }
    }
    fn stub_candle_trainer_load(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Trainer>> {
        unimplemented!("registry-derivation test stub never loads")
    }
    fn registry_capabilities_with_stubs(
        settings: &crate::Settings,
    ) -> Vec<sceneworks_core::contracts::WorkerCapability> {
        let media = gen_core::ProviderRegistryBuilder::new()
            .register_generator(STUB_MLX)
            .register_generator(STUB_CANDLE)
            .register_generator(STUB_UNKNOWN)
            .register_trainer(gen_core::TrainerRegistration {
                descriptor: stub_candle_trainer_descriptor,
                load: stub_candle_trainer_load,
            });
        let media = media.build().expect("test media registry");
        let text = gen_core::core_llm::TextLlmRegistryBuilder::new()
            .register(STUB_TEXT_LLM)
            .build()
            .expect("test LLM registry");
        registry_capabilities_from(settings, &media, &text)
    }

    #[test]
    fn mlx_enabled_advertises_image_generate_from_registry() {
        let caps = registry_capabilities_with_stubs(&settings_with_backends(true, false));
        assert!(
            caps.contains(&Cap::ImageGenerate),
            "MLX stub generator (z_image_turbo) should derive image_generate"
        );
    }

    #[test]
    fn mlx_disabled_drops_mlx_derived_image_generate() {
        // With both backends off, the mlx + candle stubs are filtered out → no image_generate.
        let caps = registry_capabilities_with_stubs(&settings_with_backends(false, false));
        assert!(
            !caps.contains(&Cap::ImageGenerate),
            "no enabled backend ⇒ no derived image_generate"
        );
    }

    // sc-8320: the base (non-distilled) `z_image` t2i row resolves to its OWN engine id (not Turbo's),
    // points at its own snapshot repo, and carries the undistilled defaults (real CFG guidance 4.0,
    // ~50 steps) — proving it is selectable and routes to the base path distinct from `z_image_turbo`
    // (which stays 8-step, CFG-free). Repos are the SceneWorks pre-built quant-matrix turnkeys
    // (sc-8670): base = `SceneWorks/z-image-mlx`, turbo = `SceneWorks/z-image-turbo-mlx`.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn z_image_base_row_is_distinct_from_turbo() {
        let base = MODEL_TABLE
            .iter()
            .find(|row| row.sceneworks_id == "z_image")
            .expect("z_image base MODEL_TABLE row");
        assert_eq!(base.engine_id, "z_image");
        assert_eq!(base.default_repo, "SceneWorks/z-image-mlx");
        assert_eq!(base.default_steps, 50);
        assert!((base.default_guidance - 4.0).abs() < f32::EPSILON);
        assert_eq!(base.adapter_label, "mlx_z_image");

        let turbo = MODEL_TABLE
            .iter()
            .find(|row| row.sceneworks_id == "z_image_turbo")
            .expect("z_image_turbo MODEL_TABLE row");
        // The base must NOT collapse onto the Turbo engine id / repo / step+CFG defaults.
        assert_ne!(base.engine_id, turbo.engine_id);
        assert_ne!(base.default_repo, turbo.default_repo);
        assert_ne!(base.default_steps, turbo.default_steps);
        assert_ne!(base.default_guidance, turbo.default_guidance);
    }

    // sc-8746 (epic 8506, Group-B): the SDXL-family rows point at the SceneWorks pre-built
    // quant-matrix turnkey repos (standard q4/q8/bf16 subdirs), NOT the original dense sources.
    // Each engine still shares the `sdxl` engine id; only the default_repo swings to the turnkey.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn sdxl_family_rows_point_at_sceneworks_turnkeys() {
        for (id, repo) in [
            ("sdxl", "SceneWorks/sdxl-base-mlx"),
            ("realvisxl", "SceneWorks/realvisxl-mlx"),
            ("realvisxl_lightning", "SceneWorks/realvisxl-lightning-mlx"),
            ("illustrious_xl_v1", "SceneWorks/illustrious-xl-v1-mlx"),
            ("illustrious_xl_v2", "SceneWorks/illustrious-xl-v2-mlx"),
        ] {
            let row = MODEL_TABLE
                .iter()
                .find(|row| row.sceneworks_id == id)
                .unwrap_or_else(|| panic!("{id} MODEL_TABLE row"));
            assert_eq!(row.engine_id, "sdxl", "{id} shares the sdxl engine");
            assert_eq!(row.default_repo, repo, "{id} default_repo → turnkey");
        }
    }

    #[test]
    fn candle_backend_lights_up_with_zero_worker_changes() {
        // candle enabled (mlx off) ⇒ the candle `sdxl` stub alone derives image_generate.
        let on = registry_capabilities_with_stubs(&settings_with_backends(false, true));
        assert!(
            on.contains(&Cap::ImageGenerate),
            "an enabled candle backend should derive image_generate from its descriptor alone"
        );
        // candle disabled ⇒ that descriptor contributes nothing.
        let off = registry_capabilities_with_stubs(&settings_with_backends(false, false));
        assert!(!off.contains(&Cap::ImageGenerate));
    }

    #[test]
    fn candle_textllm_lights_up_prompt_refine() {
        // candle enabled ⇒ the candle core-llm `TextLlm` stub (non-vision) derives the PromptRefine cap.
        let on = registry_capabilities_with_stubs(&settings_with_backends(false, true));
        assert!(
            on.contains(&Cap::PromptRefine),
            "an enabled candle backend should derive prompt_refine from its core-llm TextLlm descriptor"
        );
        // both off ⇒ nothing (neither the candle stub nor — on macOS — the real mlx twin is enabled).
        let off = registry_capabilities_with_stubs(&settings_with_backends(false, false));
        assert!(!off.contains(&Cap::PromptRefine));
    }

    #[test]
    fn candle_backend_lights_up_lora_train_execute() {
        // candle enabled ⇒ the candle `sdxl` trainer stub (backend "candle", id in TRAINER_IDS)
        // derives BOTH the dry-run `lora_train` and the real-run `lora_train_execute` caps — the
        // off-Mac training cutover (sc-7817). They light up together (same in-process trainer registry).
        let on = registry_capabilities_with_stubs(&settings_with_backends(false, true));
        assert!(
            on.contains(&Cap::LoraTrain),
            "an enabled candle backend with a registered trainer should derive lora_train"
        );
        assert!(
            on.contains(&Cap::LoraTrainExecute),
            "an enabled candle backend with a registered trainer should derive lora_train_execute"
        );
        // both off ⇒ no training caps from the candle trainer (and on macOS the real mlx trainers are
        // filtered out too, since neither backend is enabled).
        let off = registry_capabilities_with_stubs(&settings_with_backends(false, false));
        assert!(!off.contains(&Cap::LoraTrain));
        assert!(!off.contains(&Cap::LoraTrainExecute));
    }

    // Off-macOS the ONLY trainer linked is the candle stub above (backend "candle"); no real mlx
    // trainer crate is linked. So an mlx-only worker must NOT advertise training off a candle trainer
    // — proving the per-backend gate (sc-4906) actually keys on `descriptor.backend`. On macOS the
    // real mlx trainers ARE linked, so an mlx-only worker legitimately advertises training and this
    // isolation no longer holds (hence the cfg gate).
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn mlx_only_does_not_advertise_training_from_a_candle_trainer() {
        let mlx_only = registry_capabilities_with_stubs(&settings_with_backends(true, false));
        assert!(
            !mlx_only.contains(&Cap::LoraTrainExecute),
            "off-macOS the only trainer is candle-backed, so an mlx-only worker must not advertise \
             lora_train_execute"
        );
    }

    // Off-macOS no real mlx prompt_refine provider is linked (only the candle stub above), so an
    // mlx-only worker must not advertise prompt_refine.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn mlx_only_does_not_advertise_prompt_refine_without_the_mlx_twin() {
        let mlx_only = registry_capabilities_with_stubs(&settings_with_backends(true, false));
        assert!(
            !mlx_only.contains(&Cap::PromptRefine),
            "off-macOS there is no mlx prompt_refine provider, so an mlx-only worker must not \
             advertise it"
        );
    }

    // On macOS the named runtime catalog explicitly includes the native MLX prompt-refine provider.
    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_only_advertises_prompt_refine_via_the_mlx_twin() {
        let mlx_only = registry_capabilities(&settings_with_backends(true, false));
        assert!(
            mlx_only.contains(&Cap::PromptRefine),
            "the sc-5552 mlx prompt_refine twin should light up prompt_refine on an mlx-only worker"
        );
    }

    // Off-macOS the registry holds ONLY these three stubs (no real provider crate is linked), so
    // we can prove the unknown-id stub contributes nothing: with mlx enabled the unknown Image
    // stub (id not in MODEL_TABLE) must not derive image_generate by itself, and since no
    // video stub exists, video_generate is absent entirely — absence, not a runtime failure. On
    // macOS the real Wan/LTX/SVD engines are linked, so video_generate legitimately exists and
    // this isolation no longer holds.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn unknown_engine_id_contributes_no_capability() {
        let caps = registry_capabilities_with_stubs(&settings_with_backends(true, false));
        // image_generate is present here (the in-table z_image_turbo mlx stub), but the unknown
        // id adds nothing — and it never introduces video_generate (no video stub registered).
        assert!(
            !caps.contains(&Cap::VideoGenerate),
            "an unknown mlx engine id must not derive video_generate"
        );
    }
}
