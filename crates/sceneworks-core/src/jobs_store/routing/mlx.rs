//! MLX (macOS in-process) routing predicates. Moved out of `jobs_store.rs` (sc-8816) with
//! no behavior change.

use serde_json::{Map, Value};

use crate::contracts::{JobSnapshot, JobType};
use crate::jobs_store::routing::catalog::{
    MLX_ONLY_TRAINING_KERNELS, MLX_ROUTED_MODELS, MLX_ROUTED_TRAINING_KERNELS,
    VIDEO_MLX_ROUTED_MODELS,
};

/// Epic 3018 routing — does this image job belong on the in-process Rust MLX
/// worker? This lifts the per-family `_should_route_*_to_mlx` decision (ported
/// from the retired Python worker) up to the API claim layer, minus the
/// worker-local gates (platform / disable
/// env / sidecar presence) — those are now expressed by whether an `mlx` worker
/// is registered and idle (see `should_defer_image_to_mlx_worker`).
///
/// Routing-layer caveat: LyCORIS detection uses only the LoRA's *recorded*
/// `networkType`. The Python predicate also sniffs the safetensors header, but
/// the API has no access to the LoRA files; the mlx worker's own adapter
/// classifier (`image_jobs::classify_adapter`, sc-3022) is the backstop for an
/// unstamped third-party LyCORIS file that slips through.
pub(crate) fn image_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    // Both `image_generate` (text-to-image / character_image / reference) and the
    // distinct `image_edit` job type (Image Studio/Editor "plain Image Edit":
    // `mode=edit_image` + `sourceAssetId`, epic 2427) route through the same
    // per-model predicates. The engine dispatches on payload model+mode, not job
    // type (`run_image_generate_job`), and the per-model arms below already gate
    // `edit_image` (qwen/flux2/sdxl edit → eligible; torch-only edit models aren't
    // in `MLX_ROUTED_MODELS` → torch). Without `image_edit` in this gate, plain
    // Image Edit fell through to torch silently with no `gpu_route_decision`
    // (sc-3513).
    if !matches!(job.job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    image_request_mlx_eligible(model, &job.payload)
}

/// Per-model image MLX-eligibility dispatch, factored out of [`image_job_is_mlx_eligible`] so the
/// UI gating oracle ([`model_mac_support`], sc-3486) can probe the same per-family predicates with
/// synthetic payloads — one dispatch table, no divergence between routing and what the UI hides.
pub(crate) fn image_request_mlx_eligible(model: &str, payload: &Map<String, Value>) -> bool {
    if !MLX_ROUTED_MODELS.contains(&model) {
        return false;
    }
    match model {
        "z_image_turbo" | "z_image_edit" => z_image_mlx_eligible(payload),
        "flux_schnell" | "flux_dev" => flux_mlx_eligible(payload),
        "qwen_image" => qwen_mlx_eligible(payload),
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => qwen_edit_mlx_eligible(payload),
        "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2" | "flux2_dev" => {
            flux2_mlx_eligible(payload)
        }
        // Illustrious-XL shares the `sdxl` engine and its full conditioning surface (epic 10609).
        "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2" => {
            sdxl_mlx_eligible(payload)
        }
        "realvisxl_lightning" => realvisxl_lightning_mlx_eligible(payload),
        "instantid_realvisxl" => instantid_mlx_eligible(payload),
        "pulid_flux_dev" => pulid_flux_mlx_eligible(payload),
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => chroma_mlx_eligible(payload),
        "sensenova_u1_8b"
        | "sensenova_u1_8b_infographic_v2"
        | "sensenova_u1_8b_fast"
        | "sensenova_u1_8b_infographic_v2_fast" => sensenova_mlx_eligible(payload),
        "kolors" => kolors_mlx_eligible(payload),
        "lens" | "lens_turbo" => lens_mlx_eligible(payload),
        "bernini_image" => bernini_image_mlx_eligible(payload),
        "ideogram_4" | "ideogram_4_turbo" => ideogram_mlx_eligible(payload),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => boogu_mlx_eligible(payload),
        "krea_2_turbo" | "krea_2_raw" => krea_mlx_eligible(payload),
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => sd3_5_mlx_eligible(payload),
        "sana_1600m" | "sana_sprint_1600m" => sana_mlx_eligible(payload),
        "anima_base" | "anima_aesthetic" | "anima_turbo" => anima_mlx_eligible(payload),
        // Every model in MLX_ROUTED_MODELS must have an arm — enforced by
        // `every_mlx_routed_model_has_a_dispatch_arm` below, not just by this comment.
        _ => false,
    }
}

/// Does this `image_detail` job belong on the in-process Rust MLX worker? sc-3060 (epic 3041)
/// ports the tile-ControlNet detail refine onto the engine. Detail is SDXL-family only
/// (`sdxl` / `realvisxl`, the detail-capable backbones; the payload defaults to `realvisxl`).
/// Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the SDXL merge path too (epic 3641,
/// sc-3671), so it no longer forces torch. On Windows/Linux no `mlx` worker exists, so detail stays
/// on the Python torch path.
pub(crate) fn image_detail_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageDetail) {
        return false;
    }
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("realvisxl");
    matches!(
        model,
        "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
    )
}

/// Whether the in-process MLX worker can serve this GPU job (image_generate or image_detail).
pub(crate) fn job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    image_job_is_mlx_eligible(job) || image_detail_mlx_eligible(job)
}

/// Epic 3180 / sc-3905 routing — does this understanding job (`image_vqa` / `image_interleave`)
/// belong on the in-process Rust MLX worker on macOS? These two modes are SenseNova-U1's
/// understanding/interleave surface, served via the concrete `T2iModel` (`vqa` / `interleave_gen`)
/// because the `Generator` contract emits Images/Video only. SenseNova-U1 is the only model with an
/// in-process understanding path, so eligibility = a SenseNova-U1 id (the worker handler validates
/// the per-mode request: VQA needs a source image + question; interleave needs a prompt). Other
/// models on these job types have no MLX path and stay on the Python torch worker.
pub(crate) fn understanding_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageVqa | JobType::ImageInterleave) {
        return false;
    }
    // The understanding job types are SenseNova-specific; a missing model defaults to the base id.
    let model = job
        .payload
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("sensenova_u1_8b");
    // All SenseNova-U1 ids (base + Infographic-V2 + distilled) serve the understanding surface via
    // the same in-process T2iModel. V2 base advertises vqa/interleave; the `_fast` ids don't (their
    // manifests omit those caps, so a VQA/interleave job is never created for them) but are listed for
    // parity with the base+fast pattern — harmless.
    matches!(
        model,
        "sensenova_u1_8b"
            | "sensenova_u1_8b_infographic_v2"
            | "sensenova_u1_8b_fast"
            | "sensenova_u1_8b_infographic_v2_fast"
    )
}

/// SDXL MLX-routing conditions. sc-3026 brought txt2img + LoRA; sc-3060 (epic 3041) adds the
/// advanced shapes the Rust `mlx-gen-sdxl` engine now handles — reference/IP-Adapter, img2img
/// `edit_image`, masked inpaint, and outpaint — so they route to the in-process MLX worker on
/// Mac instead of the Python torch `SdxlDiffusersAdapter`. The torch path stays authoritative
/// on Windows/Linux (no `mlx` worker registered → nothing defers) and as the Mac fallback.
/// Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the SDXL merge path (epic 3641,
/// sc-3671), so every SDXL shape — including a LyCORIS-tagged job — is MLX-eligible.
/// `image_detail` is a separate job type with its own routing (see `image_detail_mlx_eligible`).
pub(crate) fn sdxl_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// RealVisXL Lightning MLX-routing (sc-6075). The standalone distilled checkpoint runs through the
/// `sdxl` engine on its few-step `lightning` (Euler-trailing) sampler, which the engine restricts to
/// **txt2img** (it rejects an img2img/reference init — `mlx-gen-sdxl` "acceleration sampler is
/// txt2img-only"). So only a plain text-to-image job is MLX-eligible here; any `edit_image`, source,
/// reference, or mask conditioning falls back to the torch worker (or is hidden by the manifest's
/// txt2img-only `capabilities`). LoRAs + quant are fine on the SDXL path, so they don't gate.
pub(crate) fn realvisxl_lightning_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let has_nonempty_id = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    !(has_nonempty_id("sourceAssetId")
        || has_nonempty_id("referenceAssetId")
        || has_nonempty_id("maskAssetId"))
}

/// InstantID (`instantid_realvisxl`) MLX-routing conditions. The native `mlx-gen-instantid`
/// provider now serves the FULL surface on Mac: single-identity `character_image`, the 11-view
/// Character-Studio angle set (sc-3345), AND pose-library mode + face-restore (sc-3381, on the
/// #193 engine — `generate_pose` MultiControlNet IdentityNet+OpenPose / `restore_face`). So every
/// `character_image` job with a reference face routes to MLX; only a non-character / reference-less
/// job stays off. Mirrors the worker's `instantid_available` gate so the router and worker agree.
pub(crate) fn instantid_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("character_image") {
        return false;
    }
    payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// PuLID-FLUX (`pulid_flux_dev`) MLX-routing conditions (sc-3344). The native `mlx-gen-pulid`
/// registry generator serves the single surface PuLID-FLUX has: a `character_image` job with a
/// reference face (no plain text-to-image, no `edit_image` — the engine requires the face it
/// injects). Mirrors the worker's `pulid_flux_available` gate so the router and worker agree, and
/// mirrors `instantid_mlx_eligible` (its face-identity sibling). The "person-type vs non-face"
/// split is the upstream model-id choice — a person character selects `pulid_flux_dev`; a
/// non-person reference selects `flux_dev` + the native XLabs IP-Adapter (epic 3621) — so no
/// separate fall-through gate is needed here. PuLID has no user-LoRA path (`supports_lora=false`),
/// and the torch path ignored LoRAs too, so a LoRA never changes eligibility.
pub(crate) fn pulid_flux_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("character_image") {
        return false;
    }
    payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

/// FLUX.2 MLX-routing conditions, shared by klein and dev. FLUX.2 is an **MLX-only** family (no torch
/// backend), so everything it does runs on MLX: klein txt2img (sc-3025), edit/reference + KV-cache +
/// multi-reference (sc-3029), third-party LyCORIS via the core loader (epic 3641), and FLUX.2-dev
/// txt2img (epic 5914 — dev's manifest advertises `text_to_image` only, so its edit/character modes
/// are never offered until the Pixtral path lands in sc-5919).
pub(crate) fn flux2_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// Qwen-Image (sc-3024 / strict pose sc-3575) MLX-routing conditions: text-to-image,
/// plus the base-Qwen strict pose tier (`advanced.poses`) handled by the `qwen_image_control`
/// engine variant. A reference without poses (character/edit flow) and `edit_image` stay on
/// the Python torch path. Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the core MLX
/// loader (epic 3641, sc-3642/3643), so it no longer forces torch.
pub(crate) fn qwen_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        return true;
    }
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    if has_reference {
        return false;
    }
    true
}

/// Qwen-Image-Edit (sc-3397/sc-3398) MLX-routing conditions. The `qwen_image_edit` /
/// `_2509` / `_2511` / `_2511_lightning` ids run the engine's `qwen_image_edit` model on
/// the Rust worker (the edit sibling of `qwen_mlx_eligible`). Eligible when the job carries
/// the reference the edit model requires: `edit_image` with a `sourceAssetId` (or a
/// `referenceAssetId`), or `character_image` with a `referenceAssetId` (the subject-variation
/// / best-effort-pose / angle-set flows — all reference-conditioned). The lightning distill
/// (sc-3398) shares the same gate (its sampler + distill-LoRA are worker-local). Third-party
/// LyCORIS now applies on the core MLX loader (epic 3641), so it no longer forces torch.
pub(crate) fn qwen_edit_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let has_source = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    match payload.get("mode").and_then(Value::as_str) {
        Some("edit_image") => has_source || has_reference,
        Some("character_image") => has_reference,
        _ => false,
    }
}

/// FLUX.1 (sc-3023) MLX-routing conditions, ported from `_should_route_flux_to_mlx`:
/// text-to-image only — FLUX.1 reference/IP-Adapter and `edit_image` stay on the
/// Python torch path (`FluxDiffusersAdapter`). A third-party LyCORIS LoRA also falls
/// back to torch: the engine + the worker's `classify_adapter` apply LoRA and peft
/// LoKr natively, but not arbitrary LyCORIS (which the worker would reject).
/// FLUX.1 (`flux_schnell` / `flux_dev`) MLX-routing conditions. Text-to-image and
/// **reference-image** (the XLabs IP-Adapter, epic 3621 — `referenceAssetId`, both
/// variants: the Rust engine has no diffusers `load_ip_adapter` schnell limitation,
/// so reference is native on schnell too). `edit_image` stays off — FLUX.1 has no
/// edit path on any platform (a future Kontext epic, NOT a Python-eradication gap).
/// Third-party LyCORIS now applies on the core MLX loader (epic 3641), so only `edit_image`
/// keeps a FLUX.1 job off MLX.
pub(crate) fn flux_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Z-Image (sc-3022) MLX-routing conditions, ported from
/// `_should_route_z_image_to_mlx`: text-to-image, reference-identity img2img-init
/// (sc-3619 — `referenceAssetId` without a pose set, the plain img2img path the
/// base engine already supports), reference+pose (the Fun-ControlNet pose tier
/// lives only on MLX — sc-2257/sc-2328, so a reference+pose job must NOT divert to
/// torch, which would honour count while dropping the poses), and `edit_image`
/// img2img-edit (epic 3529 — the engine's `Conditioning::Reference` img2img path with a
/// `sourceAssetId` init, shared by `z_image_turbo` edit_image mode and the `z_image_edit`
/// model, both on Turbo weights). An `edit_image` without a source asset has nothing to
/// edit, so it stays off MLX. Third-party LyCORIS now applies on the core MLX loader
/// (epic 3641), so a LoRA never forces torch.
pub(crate) fn z_image_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty());
    }
    true
}

/// Chroma (epic 3531, sc-3843) MLX-routing conditions. Chroma is **text-to-image only**
/// (`text_to_image` + `style_variations`; no edit / reference / ControlNet — those would be
/// later engine ports), so every non-edit `image_generate` job routes to the in-process Rust
/// `mlx-gen-chroma` worker on Mac. An `edit_image` mode — which Chroma has no path for on any
/// platform — stays off MLX (defensive; the UI never offers edit for Chroma). All three variants
/// (`chroma1_hd` / `chroma1_base` / `chroma1_flash`) share this gate. Third-party LyCORIS and peft
/// LoKr apply on the core MLX loader (epic 3641 / sc-3842), so a LoRA never forces torch.
pub(crate) fn chroma_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// SenseNova-U1 (sc-3900, epic 3180) MLX-routing conditions. The unified NEO-Unify model serves
/// three image modes on the single `sensenova_u1_8b` / `sensenova_u1_8b_fast` ids: plain T2I
/// (base path), instruction edit (`edit_image` → `Conditioning::Reference`), and Character Studio
/// (`character_image` → `Conditioning::MultiReference`, incl. the angle set) — all via the Rust
/// worker. It has NO ControlNet, so the strict-pose tier (`advanced.poses`) is unsupported and
/// drops to torch on non-Mac (it has no Mac path — epic 3482). Edit/character require the
/// reference the it2i path needs; plain T2I is always eligible. User LoRAs are not supported
/// (`supports_lora=false`) and the manifest surfaces no LoRA slot, so no LoRA gate is needed.
pub(crate) fn sensenova_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses {
        // No skeleton/ControlNet conditioning — strict pose is not an MLX SenseNova path.
        return false;
    }
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let has_source = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    match payload.get("mode").and_then(Value::as_str) {
        Some("edit_image") => has_source || has_reference,
        Some("character_image") => has_reference,
        // Plain T2I (text_to_image / no mode) — eligible with or without an inert reference.
        _ => true,
    }
}

/// Kolors (epic 3090) MLX-routing conditions. The engine `kolors` model (an SDXL-family U-Net under
/// a ChatGLM3-6B encoder) now runs the **full surface** on the in-process Rust worker: plain T2I
/// (sc-3875), img2img (`edit_image` + `sourceAssetId`, sc-4765), the IP-Adapter-Plus reference
/// (`referenceAssetId`, sc-4767) — all via the base `Reference` path — and the strict-pose tier
/// (`advanced.poses` + a reference, the combined pose-ControlNet + IP-Adapter-identity + img2img pass:
/// engine sc-5012 + the worker `generate_kolors_control_stream`, sc-4766). A pose set without a
/// reference is not the pose tier (torch `_pose_entries` ignores it) and falls through to the base
/// path as plain T2I — same as torch — so every Kolors job is MLX-eligible. Third-party LyCORIS / peft
/// LoKr apply on the SDXL-family loader (epic 3641), so a LoRA never forces torch.
pub(crate) fn kolors_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// Lens / Lens-Turbo (epic 3164 / sc-5105) is a pure T2I family — the `mlx-gen-lens` descriptor
/// advertises no conditioning (no img2img / ControlNet / IP), and the base + turbo ids share the
/// architecture/weights tree, differing only in their step/guidance defaults. Every non-edit
/// `image_generate` job routes to the in-process Rust `mlx-gen-lens` worker on Mac. An `edit_image`
/// mode — which Lens has no path for on any platform (`supportsEdit=false`) — stays off MLX so it is
/// never silently run as plain T2I against a dropped source image (defensive; the UI never offers
/// edit for Lens). Mirrors [`chroma_mlx_eligible`]. (LoRA/LoKr apply at load on the DiT — sc-3174 —
/// so a LoRA never forces torch; LoRA/LoKr *training* is also native MLX now — the `lens_lora` kernel
/// routes to the `mlx-gen-lens` Rust trainer via [`MLX_ROUTED_TRAINING_KERNELS`], sc-5148/sc-5180.)
pub(crate) fn lens_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Bernini still-image companion (epic 4699 / sc-5424) MLX-routing conditions. The image-typed
/// `bernini_image` id serves two still tasks on the same `engine_id:"bernini"` planner+renderer the
/// video `bernini` id uses: plain text-to-image (t2i, the base path) and `edit_image` img2img (i2i —
/// the source image is VAE/ViT-encoded as the engine's `Conditioning::Reference`, with the worker
/// forcing `frames:1` + `video_mode:"t2i"|"i2i"` so the engine returns a single still). An
/// `edit_image` mode without a `sourceAssetId` has nothing to edit, so it stays off MLX (mirrors
/// [`z_image_mlx_eligible`]); plain t2i is always eligible. There is no reference/character/pose
/// still surface (the renderer's reference path is video-only — `reference_to_video`), and the
/// engine reports `supports_lora: false`, so no LoRA gate is needed. macOS-only (the engine is
/// `mac_only`); on Windows/Linux no `mlx` worker is registered, so nothing defers.
pub(crate) fn bernini_image_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.trim().is_empty());
    }
    true
}

/// Ideogram 4 (epic 4725) MLX-routing conditions — shared by `ideogram_4` and `ideogram_4_turbo`
/// (the same base model + the bundled TurboTime LoRA). The native `mlx-gen-ideogram` engine serves
/// **text-to-image** and, since sc-6303, **img2img / mask-inpaint edit** (`mode == "edit_image"` with
/// a `sourceAssetId` + optional `maskAssetId`, resolved by the worker's `resolve_ideogram_edit`).
/// Both route to the in-process Rust worker, so every `image_generate` job is MLX-eligible. (Ideogram
/// has no identity-reference / pose path; those modes are not offered by the UI — the catalog
/// `capabilities` drive the affordances, not this predicate — so leaving them eligible here is inert
/// and preserves the pre-edit behavior of running an unsupported reference as plain T2I rather than
/// stranding it.) macOS-only (the catalog flags `macOnly`); on Windows/Linux no `mlx` worker is
/// registered, so nothing defers.
pub(crate) fn ideogram_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// Boogu Image / Turbo / Edit (epic 6387) MLX-eligibility. Text-to-image (and any non-edit mode) is
/// always eligible. `edit_image` is the **Edit checkpoint's** capability only — Base/Turbo are
/// text-to-image (their semantic-edit path is incoherent without the Edit fine-tune, E7b-3), so an
/// edit request is eligible for `boogu_image_edit` alone. This keeps `model_mac_support`'s `features.edit`
/// false for Base/Turbo (it probes with `mode: edit_image`). macOS-only (the catalog flags `macOnly`);
/// off-Mac no `mlx` worker registers.
pub(crate) fn boogu_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let is_edit = payload.get("mode").and_then(Value::as_str) == Some("edit_image");
    if is_edit {
        return payload.get("model").and_then(Value::as_str) == Some("boogu_image_edit");
    }
    true
}

/// Krea 2 Turbo (epic 7565 / sc-7572) + Krea 2 Raw (epic 9992) MLX-eligibility. Both variants serve
/// text-to-image on the native `mlx-gen-krea` engine. Krea 2 **Raw** additionally serves the
/// Kontext-style image-edit surface (epic 10871): an `edit_image` job with a conditioning image routes
/// to the dual-conditioned edit lane (source image as in-context VAE tokens + Qwen3-VL vision-tower
/// grounding), which the community `krea2_identity_edit` LoRA needs. The conditioning image can arrive as
/// a plain `sourceAssetId`, a single `referenceAssetId`, or the two-reference scene+person set
/// (`referenceAssetIds` — scene = image 1, person = image 2, `sourceAssetId` null) — the same fields the
/// worker's `edit_reference_ids` resolves, checked here by [`edit_has_reference`] so the router and worker
/// agree. Edit is Raw-only — it denoises from pure noise under full CFG (the tier the LoRA targets and the
/// one validated on Metal, sc-10881); **Turbo** runs the SAME edit forward CFG-free on the distilled
/// few-step schedule (`krea_2_turbo_edit`, guidance=0, ~8 steps, validated sc-11640), so its
/// `features.edit` (the `model_mac_support` probe) flips true too. The worker picks the engine id by model. An
/// `edit_image` shape with no conditioning image at all is rejected (the defensive shape t2i-only engines
/// reject).
pub(crate) fn krea_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        // Both Krea image variants serve the edit surface: Raw (full-CFG `krea_2_edit`, epic 10871) and
        // Turbo (CFG-free distilled `krea_2_turbo_edit`, sc-11640 -- the fast-path). The worker's
        // `krea_edit_available` picks the engine id by model; t2i/img2img on either is unrestricted.
        let is_krea_edit = matches!(
            payload.get("model").and_then(Value::as_str),
            Some("krea_2_raw") | Some("krea_2_turbo")
        );
        // The edit needs an image to condition on, but it can arrive in any of the fields the worker's
        // `edit_reference_ids` (base.rs) accepts, in the same priority: the two-reference scene+person
        // set (`referenceAssetIds`, epic 10871 — scene = image 1, person = image 2, `sourceAssetId`
        // null), a single `referenceAssetId`, or a plain `sourceAssetId`. Checking only `sourceAssetId`
        // here stranded the two-ref form: the mlx worker refused it and, with no torch/candle Krea edit
        // lane on Mac, it sat on "Waiting for an available GPU worker" forever.
        return is_krea_edit && edit_has_reference(payload);
    }
    true
}

/// Whether an `edit_image` payload carries a conditioning image in any field the worker's
/// [`edit_reference_ids`](../../../sceneworks-worker) resolves — a non-empty `referenceAssetIds`
/// list, a `referenceAssetId`, or a `sourceAssetId`. Mirrors that worker helper so the router and
/// the worker agree on what counts as a runnable edit.
fn edit_has_reference(payload: &Map<String, Value>) -> bool {
    let has_nonempty_str = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
    };
    let has_reference_list = payload
        .get("referenceAssetIds")
        .and_then(Value::as_array)
        .is_some_and(|ids| {
            ids.iter()
                .filter_map(Value::as_str)
                .any(|id| !id.trim().is_empty())
        });
    has_reference_list || has_nonempty_str("referenceAssetId") || has_nonempty_str("sourceAssetId")
}

/// Stable Diffusion 3.5 Large / Large Turbo / Medium (epic 7841, surfaced S4 sc-7873) MLX-eligibility.
/// The native `mlx-gen-sd3` engine serves the **text-to-image** surface only (Large + Medium run true
/// CFG, Turbo is the CFG-free few-step distill); there is no source/reference/edit path, so an
/// `edit_image` request is rejected (the same defensive shape Krea / Lens reject). This keeps
/// `model_mac_support`'s `features.edit` false for all three (it probes with `mode: edit_image`).
/// macOS-only (the catalog flags `macOnly`); off-Mac no `mlx` worker registers so nothing defers.
pub(crate) fn sd3_5_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// SANA 1600M (epic 8485 / sc-8489) + SANA-Sprint (sc-8490) MLX-eligibility. The native `mlx-gen-sana`
/// engine serves the **text-to-image** surface only — base SANA (true-CFG, 20 steps / guidance 4.5) and
/// the CFG-free few-step Sprint distillation (default 2 steps) share this gate; neither checkpoint has
/// img2img/control conditioning, so an `edit_image` request is rejected (the same defensive shape
/// Krea / SD3.5 / Lens reject). This keeps `model_mac_support`'s `features.edit` false (it probes with
/// `mode: edit_image`). macOS-only (the catalog flags `macOnly`); off-Mac no `mlx` worker registers so
/// nothing defers.
pub(crate) fn sana_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Anima base / aesthetic / turbo (epic 10512 / sc-10523) MLX-eligibility. The native `mlx-gen-anima`
/// engine serves the **text-to-image** surface only — the manifest declares `capabilities:
/// ["text_to_image"]` and the Cosmos-Predict2 DiT has no source/reference/edit path — so an
/// `edit_image` request is rejected, the same defensive shape SANA / SD3.5 / Krea / Lens use. All
/// three variants share the engine and differ only in checkpoint + step/guidance defaults.
///
/// Anima is `mlx_routed` with `candle_routed = false`, so this predicate is the ONLY thing that can
/// make an Anima job claimable: the mlx worker refuses a job it is not eligible for
/// (`worker_supports_job`) and no candle/torch lane advertises the family. A missing arm here left
/// every Anima job queued on "Waiting for an available worker." forever.
pub(crate) fn anima_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Epic 3018 routing (sc-3036, the video sibling of [`image_job_is_mlx_eligible`]):
/// does this video job belong on the in-process Rust MLX worker? Encodes today's
/// Python `create_video_adapter` MLX-eligibility (video_adapters.py) at the claim
/// layer, minus the worker-local gates (MPS presence / sidecar) — those are now
/// expressed by whether an `mlx` worker is registered and idle (see
/// [`should_defer_video_to_mlx_worker`]).
///
/// MLX covers `text_to_video` + `image_to_video` on Wan/LTX, `image_to_video` on SVD
/// (`svd`→`svd_xt`, image-conditioned only — sc-3523), `first_last_frame` on the FLF-capable
/// engines (LTX + Wan TI2V-5B `wan_2_2`; sc-3520), the clip-conditioning modes `extend_clip` /
/// `video_bridge` on the LTX IC-LoRA path **and Wan TI2V-5B** (sc-3522 / sc-3357, the `VideoExtend`
/// / `VideoBridge` job types — Wan via single-frame boundary keyframe conditioning), and
/// `replace_person` → native Wan-VACE (the `PersonReplace` job type, sc-3521 — see
/// [`video_mode_is_mlx_eligible`]). Still on the Python torch path: a non-MLX model, and
/// extend/bridge on the 14B Wan MoE engines (no `Keyframe` path).
/// **Third-party LyCORIS (LoHa / non-peft LoKr) and LoKr-on-Wan now run on MLX**
/// (epic 3641, sc-3671 + sc-3644): the Wan/LTX engine paths reconstruct + merge/residual the delta —
/// the peft-LoKr-on-Wan merge has existed since sc-2393, and the old `create_video_adapter` torch
/// gate was a routing caution, never an engine limit.
pub(crate) fn video_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    // The base `video_generate` job type plus the advanced job types: the clip-conditioning
    // `video_extend` / `video_bridge` (sc-3522, LTX IC-LoRA) and `person_replace` (sc-3521 →
    // Wan-VACE). The per-model/per-mode gate below keeps each mode to its capable engines.
    if !matches!(
        job.job_type,
        JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::PersonReplace
    ) {
        return false;
    }
    let Some(model) = job.payload.get("model").and_then(Value::as_str) else {
        return false;
    };
    if !VIDEO_MLX_ROUTED_MODELS.contains(&model) {
        return false;
    }
    // The advanced job types carry their mode by construction (the API maps
    // `extend_clip`→`VideoExtend` / `video_bridge`→`VideoBridge` / `replace_person`→
    // `PersonReplace`), so derive it from the job type rather than trusting the payload
    // `mode` — a missing/stale `mode` on those types must not fall through to the
    // `image_to_video` default and route incorrectly. The base `video_generate` type reads
    // the payload `mode` (default `image_to_video`, mirroring `video_request_from_job`).
    let mode = match job.job_type {
        JobType::VideoExtend => "extend_clip",
        JobType::VideoBridge => "video_bridge",
        JobType::PersonReplace => "replace_person",
        _ => job
            .payload
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("image_to_video"),
    };
    if !video_mode_is_mlx_eligible(model, mode) {
        return false;
    }
    true
}

/// Which `video_generate` modes the in-process Rust MLX worker serves for `model`. The Wan/LTX
/// engines serve `text_to_video` + `image_to_video` (sc-3034/3035); `first_last_frame` is
/// additionally MLX on the FLF-capable engines — LTX (`ltx_2_3`/`ltx_2_3_eros`, the
/// reference-grounded `Keyframe` path, sc-3052) and Wan TI2V-5B (`wan_2_2`, the mask-blend
/// multi-keyframe path, sc-3357). The 14B Wan MoE engines have no `Keyframe` path, so FLF on
/// them stays torch. **SVD (`svd`) is image-conditioned only** — it serves `image_to_video`
/// exclusively (no text→video, sc-3523). The clip-conditioning modes `extend_clip` /
/// `video_bridge` are MLX on the **LTX** engines (`ltx_2_3`/`ltx_2_3_eros`, the IC-LoRA
/// multi-frame keyframe-append path — sc-3522, engine `build_clips` sc-3052/3053) **and Wan
/// TI2V-5B** (`wan_2_2`, single-frame boundary `Keyframe` conditioning — sc-3357: extend pins the
/// source clip's last frame, bridge pins the two boundary frames, the same mask-blend primitive as
/// Wan FLF, matching the torch Wan reference which routed these to plain i2v). The 14B Wan MoE
/// engines have no `Keyframe` path so they stay torch. `replace_person` is MLX on the
/// replace-capable models (→ native Wan-VACE, sc-3521).
pub(crate) fn video_mode_is_mlx_eligible(model: &str, mode: &str) -> bool {
    if model == "svd" {
        return mode == "image_to_video";
    }
    // Bernini's renderer is Wan2.2-T2V (text-conditioned) — it has no classic
    // still-image-to-video. Beyond `text_to_video` (sc-4707) it serves the planner's
    // editing + reference-driven video tasks (sc-4703): `video_to_video` (v2v — a
    // source-clip edit, `Conditioning::VideoClip`), `reference_to_video` (r2v —
    // subject reference images, `MultiReference`), and `reference_video_to_video`
    // (rv2v — source clip + reference images); plus the multi-source modes (sc-5425):
    // `multi_video_to_video` (mv2v — several source clips) and `ads2v` (source video +
    // reference video + reference images). The engine selects the matching guidance
    // mode from `video_mode` + the supplied conditioning.
    if model == "bernini" {
        return matches!(
            mode,
            "text_to_video"
                | "video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "multi_video_to_video"
                | "ads2v"
        );
    }
    // SCAIL-2 (epic 5439) is a Wan2.1-14B I2V character-animation engine: a reference character
    // image + a driving video → an animated clip. It serves the standalone `animate_character` mode
    // (sc-5448, the worker paints the color-coded masks from native SAM3) AND cross-identity
    // `replace_person` (sc-5452, the integrated backend behind the YOLO11 → ByteTrack → SAM3
    // person-track pipeline). Both run the same engine; `replace_person` flips the engine
    // `replace_flag`. It has no classic text/image-to-video.
    if model == "scail2_14b" {
        return matches!(mode, "animate_character" | "replace_person");
    }
    // Mochi 1 (epic 1788 / sc-11991) is TEXT-conditioned only: both descriptors declare
    // `conditioning: []`, so the engine has no image/keyframe/clip path at all — not even the classic
    // still-image-to-video the generic arm below would otherwise grant it. Anything but
    // `text_to_video` is a gap, so it needs its own arm rather than the `svd`-style inversion.
    if model == "mochi_1" {
        return mode == "text_to_video";
    }
    match mode {
        "text_to_video" | "image_to_video" => true,
        "first_last_frame" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        // extend_clip / video_bridge: LTX via the IC-LoRA multi-frame keyframe-append (sc-3522),
        // and Wan (`wan_2_2`) — the worker prefers native Wan-VACE ControlClip for genuine motion
        // continuity (sc-3812, tier C: real source frames pinned at the kept positions + a
        // generated-span mask) and falls back to the TI2V-5B single-frame boundary keyframe path
        // (sc-3357) when the VACE snapshot is unprovisioned. Both run MLX-native, so `wan_2_2` is
        // eligible regardless of which the worker picks. The 14B Wan MoE engines have neither
        // path, so extend/bridge on them stay torch.
        "extend_clip" | "video_bridge" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        // replace_person → native Wan-VACE (sc-3521): the engine `wan_vace` provider serves it
        // regardless of the user-picked replace-capable model (ltx_2_3 / ltx_2_3_eros / wan_2_2,
        // the models that advertise the capability), so admit those.
        "replace_person" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        _ => false,
    }
}

/// Epic 3039 routing — does this `lora_train` job belong on the in-process Rust MLX
/// worker (vs the Python torch worker)? The training sibling of
/// [`image_job_is_mlx_eligible`]/[`video_job_is_mlx_eligible`]: the engine has a
/// native trainer for the family. Both dry-run and real runs are eligible (the
/// dry-run validates the same resolved plan). LoKr-on-Wan stays torch — the mlx Wan
/// inference path can't load a Kronecker adapter, mirroring [`video_job_is_mlx_eligible`];
/// LoKr on Z-Image/SDXL/LTX is fine (the Rust engine applies it natively).
///
/// The resolved plan is stamped into the job payload at submit (apps/rust-api
/// training.rs) for both dry-run and real runs, so the kernel + network type are
/// readable here without touching the dataset or weights.
pub(crate) fn training_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::LoraTrain) {
        return false;
    }
    let Some(plan) = job.payload.get("plan").and_then(Value::as_object) else {
        return false;
    };
    let kernel = plan
        .get("target")
        .and_then(Value::as_object)
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !MLX_ROUTED_TRAINING_KERNELS.contains(&kernel) {
        return false;
    }
    // LoKr-on-Wan stays on the torch path (no Kronecker merge in the mlx Wan path).
    if matches!(kernel, "wan_lora" | "wan_moe_lora") && training_plan_is_lokr(plan) {
        return false;
    }
    true
}

/// sc-3556 routing: SceneWorks training caption jobs keep their public
/// `captioner=joy_caption` contract while the macOS mlx worker serves them through
/// mlx-gen's JoyCaption provider. Other/unknown captioners stay off the mlx worker.
pub(crate) fn caption_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::TrainingCaption)
        && job
            .payload
            .get("captioner")
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim() == "joy_caption")
}

/// Whether an `image_upscale` job runs on the Rust/MLX path (epic 3482, sc-3489): the
/// Real-ESRGAN (RRDBNet) engine — the default — is ported to the Rust worker, and `seedvr2`
/// (the native-MLX one-step diffusion upscaler, epic 4811 / sc-4815) runs in-process via
/// `mlx-gen-seedvr2`. `aura-sr` (a 617M-param torch-only GigaGAN) was dropped on Mac after the
/// sc-3668 port-or-drop spike, so the mlx worker refuses it (it runs on the Python worker on
/// Windows/Linux). Engine defaults to `real-esrgan` when absent (mirrors `run_image_upscale`).
/// SeedVR2 is Mac-only here (a Windows/Linux Candle backend is the separate sc-5157); the Mac UI
/// gating + `imageUpscaleSeedvr2` capability keep it off non-Mac pickers.
pub(crate) fn upscale_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::ImageUpscale) {
        return false;
    }
    let engine = job
        .payload
        .get("engine")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "real-esrgan".to_owned());
    matches!(
        engine.as_str(),
        "" | "real-esrgan" | "realesrgan" | "real_esrgan" | "seedvr2"
    )
}

/// Whether a `video_upscale` job is MLX-eligible (epic 4811 / sc-4816). The only Mac engine is the
/// native-MLX SeedVR2 upscaler (`mlx-gen-seedvr2`); there is no torch fallback (mac-only). A job with
/// any other engine is refused by the mlx worker — though no other backend advertises `video_upscale`
/// today, so an unsupported engine simply has nowhere to run (surfaced as unsupported, not silently
/// dropped). Defaults to `seedvr2` when the payload omits the engine.
pub(crate) fn video_upscale_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::VideoUpscale) {
        return false;
    }
    let engine = job
        .payload
        .get("engine")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "seedvr2".to_owned());
    matches!(engine.as_str(), "" | "seedvr2" | "seedvr2_3b")
}

/// Whether an `image_upscale` job explicitly requests the SeedVR2 engine (`engine=seedvr2`, the id the
/// web sends and the worker accepts). SeedVR2 has no torch backend — it runs on MLX (Mac) or candle
/// (Windows/Linux) — so this also drives the torch worker's refusal (the inverse of the AuraSR gate).
/// The image default engine is Real-ESRGAN, so an absent engine is NOT SeedVR2.
pub(crate) fn upscale_job_requests_seedvr2(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::ImageUpscale)
        && job
            .payload
            .get("engine")
            .and_then(Value::as_str)
            .is_some_and(|engine| engine.trim().eq_ignore_ascii_case("seedvr2"))
}

/// Whether this training job targets a kernel with no torch fallback (see
/// [`MLX_ONLY_TRAINING_KERNELS`]). Such a job can only run on a Rust worker (mlx, or candle when the
/// candle exception in [`worker_supports_job`] admits it — e.g. `krea_control`), so a torch worker
/// must refuse it. Covers both the `lora_train` and the ControlNet studio (`control_training`) jobs —
/// both stamp a resolved plan whose `krea_control` kernel is no-torch-fallback (epic 10159).
pub(crate) fn training_kernel_is_mlx_only(job: &JobSnapshot) -> bool {
    if !matches!(job.job_type, JobType::LoraTrain | JobType::ControlTraining) {
        return false;
    }
    job.payload
        .get("plan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("target"))
        .and_then(Value::as_object)
        .and_then(|target| target.get("kernel"))
        .and_then(Value::as_str)
        .is_some_and(|kernel| MLX_ONLY_TRAINING_KERNELS.contains(&kernel))
}

/// Whether a resolved training plan requests a LoKr (Kronecker) adapter. The network
/// type lives in the plan's `config.advanced.networkType` (SceneWorks training
/// contract), distinct from a generation request's per-LoRA `networkType`.
pub(crate) fn training_plan_is_lokr(plan: &Map<String, Value>) -> bool {
    plan.get("config")
        .and_then(Value::as_object)
        .and_then(|config| config.get("advanced"))
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("networkType"))
        .and_then(Value::as_str)
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("lokr"))
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map, Value};

    use super::{image_request_mlx_eligible, MLX_ROUTED_MODELS};

    /// Every id in [`MLX_ROUTED_MODELS`] must have a real arm in [`image_request_mlx_eligible`]'s
    /// dispatch. An id that falls through to the `_ => false` catch-all is never MLX-eligible for
    /// ANY payload, so the mlx worker refuses to claim it (`worker_supports_job`) — and for an
    /// `mlx_routed`-only family (no candle/torch lane) the job then sits on "Waiting for an
    /// available worker." forever. That is exactly how three Anima ids shipped in sc-10523: the
    /// caps table gained `mlx_routed = true` rows, but `image_request_mlx_eligible` gained no arm,
    /// and only a prose comment guarded the invariant.
    ///
    /// Encoded as reachability rather than by inspecting the `match`: a model HAS an arm iff some
    /// payload makes it eligible. The probes below cover every conditioning shape the arms gate on
    /// (plain t2i, edit + source, character + reference), so a model that answers `false` to all
    /// three has no arm — or has one that can never fire, which strands jobs just as badly.
    #[test]
    fn every_mlx_routed_model_has_a_dispatch_arm() {
        let probes = |model: &str| -> Vec<Map<String, Value>> {
            let shapes = [
                json!({ "model": model, "mode": "text_to_image" }),
                json!({ "model": model, "mode": "edit_image", "sourceAssetId": "asset-1" }),
                json!({ "model": model, "mode": "character_image", "referenceAssetId": "asset-1" }),
            ];
            shapes
                .into_iter()
                .map(|shape| shape.as_object().expect("probe is an object").clone())
                .collect()
        };

        let stranded: Vec<&str> = MLX_ROUTED_MODELS
            .iter()
            .copied()
            .filter(|model| {
                !probes(model)
                    .iter()
                    .any(|payload| image_request_mlx_eligible(model, payload))
            })
            .collect();

        assert!(
            stranded.is_empty(),
            "MLX_ROUTED_MODELS ids with no reachable arm in `image_request_mlx_eligible` — the mlx \
             worker can never claim these, so their jobs queue forever: {stranded:?}"
        );
    }

    /// The two-reference (scene + person) Krea 2 Raw edit (epic 10871) carries its conditioning image in
    /// `referenceAssetIds` with `sourceAssetId` absent — the shape the web sends for the "Person image"
    /// surface. The router MUST route it to the mlx worker: the worker's `edit_reference_ids` accepts it,
    /// but when the router gated on `sourceAssetId` alone it refused, and with no torch/candle Krea edit
    /// lane on Mac the job stranded on "Waiting for an available GPU worker." forever.
    #[test]
    fn krea_raw_two_reference_edit_is_mlx_eligible() {
        let two_ref = json!({
            "model": "krea_2_raw",
            "mode": "edit_image",
            "sourceAssetId": Value::Null,
            "referenceAssetIds": ["asset-scene", "asset-person"],
        });
        assert!(image_request_mlx_eligible(
            "krea_2_raw",
            two_ref.as_object().expect("probe is an object")
        ));

        // A single `referenceAssetId` (no plural list, no source) is equally a valid edit source.
        let single_ref = json!({
            "model": "krea_2_raw",
            "mode": "edit_image",
            "referenceAssetId": "asset-1",
        });
        assert!(image_request_mlx_eligible(
            "krea_2_raw",
            single_ref.as_object().expect("probe is an object")
        ));

        // An edit with NO conditioning image in any field is still rejected (defensive shape).
        let no_source = json!({ "model": "krea_2_raw", "mode": "edit_image" });
        assert!(!image_request_mlx_eligible(
            "krea_2_raw",
            no_source.as_object().expect("probe is an object")
        ));

        // Turbo now serves the SAME edit surface on the CFG-free distilled recipe (sc-11640): an
        // `edit_image` job with a source is eligible, just like Raw.
        let turbo_ref = json!({
            "model": "krea_2_turbo",
            "mode": "edit_image",
            "referenceAssetIds": ["asset-scene"],
        });
        assert!(image_request_mlx_eligible(
            "krea_2_turbo",
            turbo_ref.as_object().expect("probe is an object")
        ));

        // ...but a Turbo edit with NO conditioning image is still rejected (defensive shape).
        let turbo_no_source = json!({ "model": "krea_2_turbo", "mode": "edit_image" });
        assert!(!image_request_mlx_eligible(
            "krea_2_turbo",
            turbo_no_source.as_object().expect("probe is an object")
        ));
    }
}
