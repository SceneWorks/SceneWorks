//! MLX (macOS in-process) routing predicates. Moved out of `jobs_store.rs` (sc-8816) with
//! no behavior change.

use serde_json::{Map, Value};

use crate::contracts::{JobSnapshot, JobType};
use crate::jobs_store::routing::catalog::{
    imported_image_request_family_eligible, MLX_IMPORTED_CAPS, MLX_ONLY_TRAINING_KERNELS,
    MLX_ROUTED_FAMILIES, MLX_ROUTED_MODELS, MLX_ROUTED_TRAINING_KERNELS, VIDEO_MLX_ROUTED_MODELS,
};
use crate::jobs_store::routing::{
    conditioned_reference_count, has_malformed_optional_nested_number, has_nonempty_array,
    has_nonempty_nested_array, has_nonempty_or_malformed_array,
    has_nonempty_or_malformed_nested_array, has_nonempty_or_malformed_string, has_nonempty_string,
    has_nonnull_or_malformed_nested_carrier, krea_edit_has_unsupported_carrier,
    SENSENOVA_MODEL_IDS,
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
    // `edit_image` (qwen/flux2/sdxl edit → eligible; unsupported edit models aren't
    // in `MLX_ROUTED_MODELS` and remain queued). Without `image_edit` in this gate, plain
    // Image Edit was left unclaimed with no `gpu_route_decision`
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
        // Route-by-family fallback for a non-builtin (imported/user) image model whose novel id is in
        // no routing table (sc-14109, epic 14015). S0d (sc-14019) made only the display badge
        // (`image_model_mac_support`) family-aware, so an imported checkpoint showed as usable yet its
        // job was never claimed at THIS predicate — it stranded on "Waiting for an available GPU
        // worker." A builtin always keeps its id-keyed verdict: `!is_builtin_image_model` mirrors the
        // display badge's guard so the router and the badge agree, and it keeps a (hypothetical
        // future) candle-only builtin — a builtin id absent from `MLX_ROUTED_MODELS` — not-eligible
        // here rather than family-routed. Today every builtin is `mlx_routed`, so this branch is only
        // ever reached by imported ids; the guard is defensive parity, not live behavior.
        // `MLX_IMPORTED_CAPS`: the MLX native single-file loader takes an adapter slice
        // (inference #211) — LoRAs (sc-14111) + Kontext edit (sc-14119) — and assembles the Krea
        // pose control branch around the file-loaded DiT (strict-pose sets on the krea_2 family).
        return imported_image_request_family_eligible(
            model,
            payload,
            MLX_ROUTED_FAMILIES,
            MLX_IMPORTED_CAPS,
        );
    }
    match model {
        "z_image_turbo" | "z_image_edit" => z_image_mlx_eligible(payload),
        "z_image" => z_image_base_mlx_eligible(payload),
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
        model if SENSENOVA_MODEL_IDS.contains(&model) => sensenova_mlx_eligible(payload),
        "kolors" => kolors_mlx_eligible(payload),
        "lens" | "lens_turbo" => lens_mlx_eligible(payload),
        "bernini_image" => bernini_image_mlx_eligible(payload),
        "ideogram_4" | "ideogram_4_turbo" => ideogram_mlx_eligible(payload),
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => boogu_mlx_eligible(payload),
        "krea_2_turbo" | "krea_2_raw" => krea_mlx_eligible(payload),
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => sd3_5_mlx_eligible(payload),
        "sana_1600m" | "sana_sprint_1600m" => sana_mlx_eligible(payload),
        "anima_base" | "anima_aesthetic" | "anima_turbo" => anima_mlx_eligible(payload),
        "mage_flow_base" | "mage_flow" | "mage_flow_turbo" => mage_flow_mlx_eligible(payload),
        "mage_flow_edit_base" | "mage_flow_edit" | "mage_flow_edit_turbo" => {
            mage_flow_edit_mlx_eligible(payload)
        }
        // Every model in MLX_ROUTED_MODELS must have an arm — enforced by
        // `every_mlx_routed_model_has_a_dispatch_arm` below, not just by this comment.
        _ => false,
    }
}

/// Does this `image_detail` job belong on an in-process native Rust worker? sc-3060 (epic 3041)
/// ported the tile-ControlNet detail refine onto MLX; sc-18480 adds the same provider contract to
/// Candle. Detail is SDXL-family only (`sdxl` / `realvisxl` / Illustrious; the payload defaults to
/// `realvisxl`).
/// Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the SDXL merge path too (epic 3641,
/// sc-3671), so it no longer changes eligibility. Both native schedulers reuse this exact gate so
/// their advertised `image_detail` capability cannot claim a non-SDXL family.
pub(crate) fn image_detail_native_eligible(job: &JobSnapshot) -> bool {
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
    image_job_is_mlx_eligible(job) || image_detail_native_eligible(job)
}

/// Epic 3180 / sc-3905 routing — does this understanding job (`image_vqa` / `image_interleave`)
/// belong on the in-process Rust MLX worker on macOS? These two modes are SenseNova-U1's
/// understanding/interleave surface, served via the concrete `T2iModel` (`vqa` / `interleave_gen`)
/// because the `Generator` contract emits Images/Video only. SenseNova-U1 is the only model with an
/// in-process understanding path, so eligibility = a SenseNova-U1 id (the worker handler validates
/// the per-mode request: VQA needs a source image + question; interleave needs a prompt). Other
/// models on these job types have no MLX path and remain queued.
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
    // All SenseNova-U1 ids (base + Infographic-V2/V3 + distilled) serve the understanding surface via
    // the same in-process T2iModel. The V2/V3 bases advertise vqa/interleave; the `_fast` ids don't
    // (their manifests omit those caps, so a VQA/interleave job is never created for them) but are
    // listed for parity with the base+fast pattern — harmless.
    SENSENOVA_MODEL_IDS.contains(&model)
}

/// SDXL MLX-routing conditions. sc-3026 brought txt2img + LoRA; sc-3060 (epic 3041) adds the
/// advanced shapes the Rust `mlx-gen-sdxl` engine now handles — reference/IP-Adapter, img2img
/// `edit_image`, masked inpaint, and outpaint — so they route to the in-process MLX worker on
/// Mac instead of the historical Python `SdxlDiffusersAdapter`. On Windows/Linux, no `mlx` worker is
/// registered; only a compatible native lane may claim the job.
/// Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the SDXL merge path (epic 3641,
/// sc-3671), so every SDXL shape — including a LyCORIS-tagged job — is MLX-eligible.
/// `image_detail` is a separate job type with its own routing (see `image_detail_native_eligible`).
pub(crate) fn sdxl_mlx_eligible(_payload: &Map<String, Value>) -> bool {
    true
}

/// RealVisXL Lightning MLX-routing (sc-6075). The standalone distilled checkpoint runs through the
/// `sdxl` engine on its few-step `lightning` (Euler-trailing) sampler, which the engine restricts to
/// **txt2img** (it rejects an img2img/reference init — `mlx-gen-sdxl` "acceleration sampler is
/// txt2img-only"). So only a plain text-to-image job is MLX-eligible here; any `edit_image`, source,
/// reference, or mask conditioning is refused and remains queued (or is hidden by the manifest's
/// txt2img-only `capabilities`). LoRAs + quant are fine on the SDXL path, so they don't gate.
pub(crate) fn realvisxl_lightning_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) == Some("edit_image") {
        return false;
    }
    !(has_nonempty_string(payload, "sourceAssetId")
        || has_nonempty_string(payload, "referenceAssetId")
        || has_nonempty_string(payload, "maskAssetId"))
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

/// FLUX.2 MLX-routing conditions, shared by klein and dev. FLUX.2 runs on MLX on Mac and matching
/// Candle routes off-Mac: klein txt2img (sc-3025), edit/reference + KV-cache +
/// multi-reference (sc-3029), third-party LyCORIS via the core loader (epic 3641), and FLUX.2-dev
/// txt2img plus edit/reference/character/style workflows. Reference-bearing modes require one strict,
/// non-conflicting reference carrier so malformed requests cannot fall through to unconditioned T2I.
pub(crate) fn flux2_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let mode = payload.get("mode").and_then(Value::as_str);
    if matches!(
        mode,
        Some(
            "edit_image" | "reference" | "image_to_image" | "character_image" | "style_variations"
        )
    ) {
        return conditioned_reference_count(
            payload,
            matches!(mode, Some("edit_image" | "image_to_image")),
            4,
        )
        .is_some()
            && !mlx_conditioned_edit_has_unsupported_carrier(payload, true, false, true)
            && !conditioned_true_cfg_is_malformed(payload);
    }
    matches!(mode, None | Some("image_generation" | "text_to_image"))
        && !has_nonempty_or_malformed_array(payload, "referenceAssetIds")
        && !has_nonempty_or_malformed_string(payload, "referenceAssetId")
        && !has_nonempty_or_malformed_string(payload, "sourceAssetId")
}

/// Qwen-Image (sc-3024 / strict pose sc-3575) MLX-routing conditions: text-to-image,
/// plus the base-Qwen strict pose tier (`advanced.poses`) handled by the `qwen_image_control`
/// engine variant. A reference without poses (character/edit flow) and `edit_image` are refused and
/// remain queued. Third-party LyCORIS (LoHa / non-peft LoKr) now applies on the core MLX
/// loader (epic 3641, sc-3642/3643), so it no longer changes eligibility.
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
    if has_nonempty_or_malformed_string(payload, "referenceAssetId")
        || has_nonempty_or_malformed_string(payload, "sourceAssetId")
        || has_nonempty_or_malformed_array(payload, "referenceAssetIds")
    {
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
    let mode = payload.get("mode").and_then(Value::as_str);
    matches!(mode, Some("edit_image" | "character_image"))
        && conditioned_reference_count(payload, mode == Some("edit_image"), 5).is_some()
        && !mlx_conditioned_edit_has_unsupported_carrier(payload, true, false, true)
        && !conditioned_true_cfg_is_malformed(payload)
}

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

/// Mage-Flow Base/RL/Turbo expose the native plain text-to-image path, **with user LoRA/LoKr
/// adapters** (sc-15328). Keep this predicate deliberately narrower than the generic image request
/// shape: the provider has no edit/reference, ControlNet, or mask implementation on the generation
/// variants. Reject those requests at routing time instead of letting an MLX-only model queue for a
/// worker path that cannot serve them (or silently dropping conditioning).
///
/// **`loras` is deliberately NOT in the exclusion list.** It used to be, and that is what made a
/// trained Mage adapter unrenderable: the picker offered it (the manifest declares
/// `loraCompatibility.families = ["mage-flow"]` on all three generation variants), every Mage
/// `ModelCaps` row is `candle_lora: false`, and so NO backend claimed the job — it sat `queued`
/// forever with an idle mlx worker and no error. The engine seam is real and now wired: the worker's
/// `resolve_adapters` → `LoadSpec::with_adapters` plumbing reaches `mlx_gen_mage`'s `assemble`,
/// which installs stacked/mixed LoRA + LoKr through `apply_mage_adapters` (strict — an unmatched
/// target errors rather than being silently dropped).
pub(crate) fn mage_flow_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if !matches!(
        payload.get("mode").and_then(Value::as_str),
        None | Some("text_to_image")
    ) {
        return false;
    }

    !(has_nonempty_string(payload, "sourceAssetId")
        || has_nonempty_string(payload, "referenceAssetId")
        || has_nonempty_string(payload, "maskAssetId")
        || has_nonempty_array(payload, "referenceAssetIds")
        || has_nonempty_array(payload, "controls")
        || has_nonempty_array(payload, "controlnets")
        || has_nonempty_nested_array(payload, "advanced", "poses"))
}

/// Mage-Flow Edit requires a real primary source image. Optional plural references augment that
/// source in their submitted order; they never make a source-less request eligible.
///
/// **`loras` stays refused here, and that is a decision rather than inherited copy-paste
/// (sc-15328).** The engine could serve it — the edit variants host adapters on the same
/// `MageTransformer` and the descriptor advertises `supports_lora`/`supports_lokr` for all six — but
/// the product never offers one: the three `mage_flow_edit_*` manifest rows declare no
/// `loraCompatibility`, so the picker shows no adapter as compatible, and there is no Mage edit
/// trainer to produce one (sc-15277 withdrew `mage_flow_edit_base_lora`; a real edit trainer is
/// sc-15320). Nothing advertised, nothing claimable, no gap — which is exactly what
/// `every_lora_advertising_image_model_is_claimable_with_a_lora` checks, exception-free. Adding
/// `loraCompatibility` to an edit row without relaxing this line turns that guard red.
pub(crate) fn mage_flow_edit_mlx_eligible(payload: &Map<String, Value>) -> bool {
    if payload.get("mode").and_then(Value::as_str) != Some("edit_image") {
        return false;
    }

    has_nonempty_string(payload, "sourceAssetId")
        && !has_nonempty_string(payload, "referenceAssetId")
        && !has_nonempty_string(payload, "maskAssetId")
        && !has_nonempty_array(payload, "loras")
        && !has_nonempty_array(payload, "controls")
        && !has_nonempty_array(payload, "controlnets")
        && !has_nonempty_nested_array(payload, "advanced", "poses")
}

/// Z-Image (sc-3022) MLX-routing conditions, ported from
/// `_should_route_z_image_to_mlx`: text-to-image, reference-identity img2img-init
/// (sc-3619 — `referenceAssetId` without a pose set, the plain img2img path the
/// base engine already supports), reference+pose (the Fun-ControlNet pose tier
/// lives only on MLX — sc-2257/sc-2328, so a reference+pose job must stay on MLX rather
/// than reach a claimant that would honour count while dropping the poses), and `edit_image`
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

/// Base (non-distilled, full-CFG) Z-Image (`z_image`) MLX-routing conditions (epic 8236). Distinct from
/// [`z_image_mlx_eligible`] because the base has NO dedicated `edit_image` checkpoint — that path is
/// Turbo-weights-only (`z_image_turbo` edit mode / the `z_image_edit` model). On macOS the base engine
/// serves plain text-to-image (MODEL_TABLE `z_image`, shift-6.0 / ~50-step / real CFG), reference-guided
/// img2img-init (`referenceAssetId` + `advanced.strength`, the generic `Conditioning::Reference` path —
/// NOT `mode=edit_image`), and strict pose/canny/depth control (`ImageRoute::ZImageBaseControl`), all
/// in-process. So every non-`edit_image` job is eligible; an `edit_image` mode has no base path and stays
/// off MLX (defensive — the base UI never offers edit; base `z_image` has no `edit_image` capability).
/// Third-party LyCORIS applies on the core MLX loader (epic 3641), so a LoRA never forces torch.
pub(crate) fn z_image_base_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Chroma (epic 3531, sc-3843) MLX-routing conditions. Chroma is **text-to-image only**; it has no
/// edit / reference / ControlNet surface, so every non-edit `image_generate` job routes to the
/// in-process Rust `mlx-gen-chroma` worker on Mac. The retired `style_variations` alias may still be
/// admitted defensively for legacy API payloads, but it is no longer advertised by the catalog or
/// UI. An `edit_image` mode — which Chroma has no path for on any platform — stays off MLX
/// (defensive; the UI never offers edit for Chroma). All three variants (`chroma1_hd` /
/// `chroma1_base` / `chroma1_flash`) share this gate. Third-party LyCORIS and peft LoKr apply on the
/// core MLX loader (epic 3641 / sc-3842), so a LoRA never forces torch.
pub(crate) fn chroma_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// SenseNova-U1 (sc-3900, epic 3180) MLX-routing conditions. The unified NEO-Unify model serves
/// three image modes on the single `sensenova_u1_8b` / `sensenova_u1_8b_fast` ids: plain T2I
/// (base path), instruction edit (`edit_image` → `Conditioning::Reference`), and Character Studio
/// (`character_image` → `Conditioning::MultiReference`, incl. the angle set) — all via the Rust
/// worker. It has NO ControlNet, so the strict-pose tier (`advanced.poses`) is unsupported and
/// is refused and remains queued on non-Mac (it has no alternate native path — epic 3482).
/// Edit/character require the
/// reference the it2i path needs; plain T2I is always eligible. User LoRAs are not supported
/// (`supports_lora=false`) and the manifest surfaces no LoRA slot, so no LoRA gate is needed.
pub(crate) fn sensenova_mlx_eligible(payload: &Map<String, Value>) -> bool {
    let mode = payload.get("mode").and_then(Value::as_str);
    let has_poses = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("poses"))
        .and_then(Value::as_array)
        .is_some_and(|poses| !poses.is_empty());
    if has_poses
        || mlx_conditioned_edit_has_unsupported_carrier(payload, false, true, false)
        || conditioned_true_cfg_is_malformed(payload)
    {
        // No skeleton/ControlNet conditioning — strict pose is not an MLX SenseNova path.
        return false;
    }
    match mode {
        Some("edit_image") => conditioned_reference_count(payload, true, 5).is_some(),
        Some("character_image") => conditioned_reference_count(payload, false, 5).is_some(),
        // Plain T2I (`image_generation` / `text_to_image` / no mode) is eligible only without
        // reference carriers. A reference submitted in T2I mode is malformed, not permission to
        // drop conditioning.
        None | Some("image_generation" | "text_to_image") => {
            !has_nonempty_or_malformed_array(payload, "referenceAssetIds")
                && !has_nonempty_or_malformed_string(payload, "referenceAssetId")
                && !has_nonempty_or_malformed_string(payload, "sourceAssetId")
        }
        _ => false,
    }
}

fn conditioned_true_cfg_is_malformed(payload: &Map<String, Value>) -> bool {
    ["trueCfgScale", "imageGuidanceScale"]
        .iter()
        .any(|key| has_malformed_optional_nested_number(payload, "advanced", key))
}

fn mlx_conditioned_edit_has_unsupported_carrier(
    payload: &Map<String, Value>,
    allow_loras: bool,
    reject_strength: bool,
    allow_poses: bool,
) -> bool {
    (!allow_loras && has_nonempty_or_malformed_array(payload, "loras"))
        || ["controls", "controlnets"]
            .iter()
            .any(|key| has_nonempty_or_malformed_array(payload, key))
        || has_nonempty_or_malformed_string(payload, "maskAssetId")
        || (!allow_poses && has_nonempty_or_malformed_nested_array(payload, "advanced", "poses"))
        || has_nonempty_or_malformed_nested_array(payload, "advanced", "phases")
        || [
            "controlMode",
            "controlImage",
            "controlScale",
            "controlWeights",
            "convRot",
            "quantTier",
        ]
        .iter()
        .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key))
        || (reject_strength
            && ["strength", "ipAdapterScale", "referenceStrength"]
                .iter()
                .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key)))
        || (reject_strength
            && ["strength", "referenceStrength"]
                .iter()
                .any(|key| payload.get(*key).is_some_and(|value| !value.is_null())))
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
        // here stranded the two-ref form: the MLX worker refused it and no other Mac worker owns the
        // native Krea edit lane, so it sat on "Waiting for an available GPU worker" forever. The
        // off-Mac Candle route has the same edit surface through its own routing predicate.
        return is_krea_edit
            && conditioned_reference_count(payload, true, 2).is_some()
            && !krea_edit_has_unsupported_carrier(payload);
    }
    true
}

/// Whether an `edit_image` payload carries a conditioning image in any field the worker's
/// [`edit_reference_ids`](../../../sceneworks-worker) resolves — a non-empty `referenceAssetIds`
/// list, a `referenceAssetId`, or a `sourceAssetId`. Mirrors that worker helper so the router and
/// the worker agree on what counts as a runnable edit.
/// Stable Diffusion 3.5 Large / Large Turbo / Medium (epic 7841, surfaced S4 sc-7873) MLX-eligibility.
/// The native `mlx-gen-sd3` engine serves text-to-image plus reference-guided latent-init img2img
/// (`referenceAssetId`, epic 8588 A4 / sc-10189). It does not expose the semantic `edit_image` job
/// shape, so that distinct mode remains rejected. The family also runs through Candle/CUDA off-Mac;
/// this predicate describes only the MLX half of the shared catalog contract.
pub(crate) fn sd3_5_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// SANA 1600M (epic 8485 / sc-8489) + SANA-Sprint (sc-8490) MLX-eligibility. The native `mlx-gen-sana`
/// engine serves non-edit text-to-image plus singular-reference latent-init img2img — base SANA
/// (true-CFG, 20 steps / guidance 4.5) and the CFG-free few-step Sprint distillation (default 2 steps)
/// share this gate. Edit, control, multiple-reference, adapter, pose, phase, and malformed unsupported
/// carriers are rejected instead of being silently ignored. This keeps `model_mac_support`'s
/// `features.edit` false (it probes with `mode: edit_image`). Off-Mac no MLX worker registers.
pub(crate) fn sana_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
        && payload
            .get("referenceAssetId")
            .map(|value| value.as_str().is_some_and(|id| !id.trim().is_empty()))
            .unwrap_or(true)
        && !has_nonempty_or_malformed_string(payload, "sourceAssetId")
        && !["referenceAssetIds", "controls", "controlnets"]
            .iter()
            .any(|key| has_nonempty_or_malformed_array(payload, key))
        && !has_nonempty_or_malformed_string(payload, "maskAssetId")
        && !has_nonempty_or_malformed_nested_array(payload, "advanced", "poses")
        && !has_nonempty_or_malformed_nested_array(payload, "advanced", "phases")
        && ![
            "controlMode",
            "controlImage",
            "controlScale",
            "controlWeights",
            "convRot",
            "quantTier",
        ]
        .iter()
        .any(|key| has_nonnull_or_malformed_nested_carrier(payload, "advanced", key))
        && !sana_has_malformed_quant_carrier(payload)
        && !has_nonempty_or_malformed_array(payload, "loras")
}

/// Mirror the worker's `quant_int` request contract without treating an invalid explicit override as
/// absence. Missing/null means "use the manifest default"; an integer or trimmed integer string is a
/// valid explicit tier selection. Every other shape must fail closed instead of silently selecting Q4.
fn sana_has_malformed_quant_carrier(payload: &Map<String, Value>) -> bool {
    let Some(value) = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("mlxQuantize"))
    else {
        return false;
    };
    if value.is_null() {
        return false;
    }
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|bits| bits.trim().parse().ok()))
        .is_none()
}

/// Anima base / aesthetic / turbo (epic 10512 / sc-10523) MLX-eligibility. The native `mlx-gen-anima`
/// engine serves the **text-to-image** surface only — the manifest declares `capabilities:
/// ["text_to_image"]` and the Cosmos-Predict2 DiT has no source/reference/edit path — so an
/// `edit_image` request is rejected, the same defensive shape SANA / SD3.5 / Krea / Lens use. All
/// three variants share the engine and differ only in checkpoint + step/guidance defaults.
///
/// Anima is routed through MLX on Mac and the native Candle lane off-Mac (sc-10676). This predicate
/// owns only MLX request eligibility; Candle applies its sibling catalog/descriptor gate. A missing
/// MLX arm still leaves Mac jobs queued on "Waiting for an available worker."
pub(crate) fn anima_mlx_eligible(payload: &Map<String, Value>) -> bool {
    payload.get("mode").and_then(Value::as_str) != Some("edit_image")
}

/// Epic 3018 routing (sc-3036, the video sibling of [`image_job_is_mlx_eligible`]):
/// does this video job belong on the in-process Rust MLX worker? Encodes the retired Python
/// `create_video_adapter` MLX-eligibility (video_adapters.py) at the claim
/// layer, minus the legacy worker-local gates (synthetic MPS presence / sidecar) — those are now
/// expressed by whether an `mlx` worker is registered and idle (see
/// [`should_defer_video_to_mlx_worker`]).
///
/// MLX covers `text_to_video` + `image_to_video` on Wan/LTX, `image_to_video` on SVD
/// (`svd`→`svd_xt`, image-conditioned only — sc-3523), `first_last_frame` on the FLF-capable
/// engines (LTX + Wan TI2V-5B `wan_2_2`; sc-3520), the clip-conditioning modes `extend_clip` /
/// `video_bridge` on the LTX IC-LoRA path **and Wan TI2V-5B** (sc-3522 / sc-3357, the `VideoExtend`
/// / `VideoBridge` job types — Wan via single-frame boundary keyframe conditioning), and
/// `replace_person` → native Wan-VACE (the `PersonReplace` job type, sc-3521 — see
/// [`video_mode_is_mlx_eligible`]). A non-MLX model, or extend/bridge on the 14B Wan MoE engines
/// (no `Keyframe` path), is refused and remains queued.
/// **Third-party LyCORIS (LoHa / non-peft LoKr) and LoKr-on-Wan now run on MLX**
/// (epic 3641, sc-3671 + sc-3644): the Wan/LTX engine paths reconstruct + merge/residual the delta —
/// the peft-LoKr-on-Wan merge has existed since sc-2393, and the old `create_video_adapter` torch
/// gate was a routing caution, never an engine limit.
pub(crate) fn video_job_is_mlx_eligible(job: &JobSnapshot) -> bool {
    video_request_is_mlx_eligible(&job.job_type, &job.payload)
}

/// The `(job_type, payload)` form of [`video_job_is_mlx_eligible`] — the same predicate, reachable
/// before a [`JobSnapshot`] exists (sc-19504). `create_video_job` holds exactly this pair at
/// enqueue time and must be able to ask the REAL gate whether any lane will claim the job it is
/// about to write; a second, re-typed copy of these rules is the false-green shape that let
/// GH #2074 and sc-15328 ship. See [`super::gaps::video_request_is_claimable_by_any_lane`].
pub(crate) fn video_request_is_mlx_eligible(
    job_type: &JobType,
    payload: &Map<String, Value>,
) -> bool {
    // The base `video_generate` job type plus the advanced job types: the clip-conditioning
    // `video_extend` / `video_bridge` (sc-3522, LTX IC-LoRA) and `person_replace` (sc-3521 →
    // Wan-VACE). The per-model/per-mode gate below keeps each mode to its capable engines.
    if !matches!(
        job_type,
        JobType::VideoGenerate
            | JobType::VideoExtend
            | JobType::VideoBridge
            | JobType::PersonReplace
    ) {
        return false;
    }
    let Some(model) = payload.get("model").and_then(Value::as_str) else {
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
    let mode = match job_type {
        JobType::VideoExtend => "extend_clip",
        JobType::VideoBridge => "video_bridge",
        JobType::PersonReplace => "replace_person",
        _ => payload
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("image_to_video"),
    };
    if !video_mode_is_mlx_eligible(model, mode) {
        return false;
    }
    if model == "wan_2_2" {
        let has_source = payload
            .get("sourceAssetId")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());
        let has_last = payload
            .get("lastFrameAssetId")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());
        let shape_is_exact = match mode {
            "text_to_video" => !has_source && !has_last,
            "image_to_video" => has_source && !has_last,
            "first_last_frame" => has_source && has_last,
            _ => true,
        };
        if !shape_is_exact {
            return false;
        }
    }
    if matches!(model, "ltx_2_3" | "ltx_2_3_eros")
        && matches!(mode, "extend_clip" | "video_bridge" | "replace_person")
        && !payload
            .get("loras")
            .and_then(Value::as_array)
            .is_some_and(|loras| crate::video_request::loras_contain_ltx_ic_lora(loras))
    {
        return false;
    }
    true
}

/// Which `video_generate` modes the in-process Rust MLX worker serves for `model`. The Wan/LTX
/// engines serve `text_to_video` + `image_to_video` (sc-3034/3035); `first_last_frame` is
/// additionally MLX on the FLF-capable engines — LTX (`ltx_2_3`/`ltx_2_3_eros`, the
/// reference-grounded `Keyframe` path, sc-3052) and Wan TI2V-5B (`wan_2_2`, the mask-blend
/// multi-keyframe path, sc-3357). The 14B Wan MoE engines have no `Keyframe` path, so FLF on
/// them is refused. **SVD (`svd`) is image-conditioned only** — it serves `image_to_video`
/// exclusively (no text→video, sc-3523). The clip-conditioning modes `extend_clip` /
/// `video_bridge` are MLX on the **LTX** engines (`ltx_2_3`/`ltx_2_3_eros`, the IC-LoRA
/// multi-frame keyframe-append path — sc-3522, engine `build_clips` sc-3052/3053) **and Wan
/// TI2V-5B** (`wan_2_2`, single-frame boundary `Keyframe` conditioning — sc-3357: extend pins the
/// source clip's last frame, bridge pins the two boundary frames, the same mask-blend primitive as
/// Wan FLF, matching the torch Wan reference which routed these to plain i2v). The 14B Wan MoE
/// engines have no `Keyframe` path, so those requests are refused. `replace_person` is MLX on the
/// replace-capable models (→ native Wan-VACE, sc-3521).
pub(crate) fn video_mode_is_mlx_eligible(model: &str, mode: &str) -> bool {
    if model == "svd" {
        return mode == "image_to_video";
    }
    // The two Wan2.2 14B MoE registrations are specialized descriptors, not interchangeable aliases:
    // the T2V engine advertises no conditioning, while the I2V engine advertises Reference. Keeping
    // this split in routing prevents a source image from reaching the text-only engine (or an empty
    // request from reaching the reference-required engine).
    if model == "wan_2_2_t2v_14b" {
        return mode == "text_to_video";
    }
    if model == "wan_2_2_i2v_14b" {
        return mode == "image_to_video";
    }
    // VACE-Fun is a shipped, dedicated dual-expert replacement engine. It must not fall into the
    // generic Wan text/image generation arm: the worker dispatches only `replace_person` to
    // `wan2_2_vace_fun_14b`, and explicitly refuses the model off-Mac.
    if model == "wan_2_2_vace_fun_14b" {
        return mode == "replace_person";
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
    // Krea Realtime 14B (epic 8431 / sc-8444) needs its own arm because the generic arm below grants
    // exactly `text_to_video | image_to_video`, so its `video_to_video` would fall to `_ => false` and
    // be silently DROPPED — a capability the manifest advertises (in both `capabilities` and
    // `ui.recommendedFor`), the engine implements (`conditioning: [Reference, VideoClip]`; a
    // `VideoClip` source drives the strength-controlled AR init), and the worker already maps
    // (`krea_realtime_video_task` → `"v2v"`, via `krea_realtime_conditioning`'s clip branch). Serving
    // fewer modes than the catalog advertises is exactly the failure this arm exists to prevent.
    //
    // These three and NO more: the descriptor exposes no first/last-frame, clip-extend, bridge,
    // person-replace or character-animation surface, so those correctly stay false (the generic arm's
    // `first_last_frame` / `extend_clip` / `replace_person` lists never name it either).
    if model == "krea_realtime_14b" {
        return matches!(mode, "text_to_video" | "image_to_video" | "video_to_video");
    }
    // Wan2.2 VACE-Fun A14B (epic 3456 / sc-3458), added by sc-17159 together with its missing
    // `VIDEO_MODEL_CAPS` row. It needs its OWN arm in BOTH directions: the generic arm below would
    // grant it `text_to_video | image_to_video`, which this dual-expert CONTROL checkpoint does not
    // do (its manifest deliberately advertises `replace_person` alone —
    // `builtin_manifest_registers_the_wan_vace_fun_model` pins that), while the generic
    // `replace_person` list names only ltx/wan_2_2 and so refused the one mode it does do.
    if model == "wan_2_2_vace_fun_14b" {
        return mode == "replace_person";
    }
    // MiniMax-H3 (epic 17137 / sc-17159). TWO ids, TWO different mode sets, and each needs its own
    // arm for the opposite reason:
    //
    // * `minimax_h3` is the `transformer` checkpoint — t2va + fl2va. The generic arm below grants
    //   `text_to_video | image_to_video` but lists only LTX + Wan TI2V-5B for `first_last_frame`,
    //   so fl2va — a mode the manifest advertises in `capabilities` AND `ui.recommendedFor` —
    //   would fall to `_ => false` and surface disabled in the Video Studio on the ONLY platform
    //   this family installs on.
    // * `minimax_h3_ref` is the `transformer_ref` checkpoint and serves Ref2VA ALONE. The generic
    //   arm would have done the reverse damage: granted it `text_to_video | image_to_video`, which
    //   its checkpoint does not do, while refusing `reference_to_video`, which is the only thing it
    //   does. `image_to_video` on the reference partition is not a harmless extra — the two
    //   partitions are separate 18.78 GB DiTs and routing a t2v request at the reference one loads
    //   the wrong checkpoint.
    //
    // The mode sets are exactly each entry's declared `capabilities`, and
    // `every_declared_video_capability_is_claimable_by_some_lane` (routing/catalog.rs) reads the
    // shipped manifest bytes to hold that, so a capability added to either entry without an arm
    // here is RED at the source rather than at the next user report.
    if model == "minimax_h3" {
        return matches!(
            mode,
            "text_to_video" | "image_to_video" | "first_last_frame"
        );
    }
    if model == "minimax_h3_ref" {
        // WITHHELD, NOT AN OVERSIGHT — `reference_to_video` is the ONLY mode this partition serves,
        // and it is deliberately not declared MLX-routed yet. Unblocked by **sc-17157**, and the
        // condition is exact and checkable in one place:
        //
        //   the MLX provider must declare `ConditioningKind::MultiReference` in
        //   `mlx-gen-minimax-h3/src/model.rs`'s `conditioning` vec.
        //
        // Measured at inference `75d66db5` (the revision PR #2356 pins, which is the FIRST one that
        // registers `mlx_gen_minimax_h3` at all): that vec is
        // `[Keyframe, Reference, ReferenceVideo, ReferenceAudio]` — no `MultiReference`. SceneWorks
        // requires `multiReference` for `reference_to_video`
        // (`video_mode_conditioning_requirements`), and that is the convention rather than a quirk:
        // `bernini`, the only other engine with a `reference_to_video` mapping, declares it, as do
        // `scail2_14b` and the whole edit-model family.
        //
        // So declaring it here advertises an MLX route the pinned engine does not claim to serve.
        // `dump-engine-capabilities` refuses outright — "descriptors cannot satisfy required
        // conditioning alternatives [multiReference]" — which blocks the runtime artifact for EVERY
        // model, not just this one. Withdrawing the declaration is what makes the catalog true.
        //
        // Nothing is lost today: the family's engine is absent from the currently pinned bundle, so
        // ref2va renders nowhere, and sc-19558 deliberately gave this partition no off-Mac downloads
        // either. `video_mlx_routed` stays TRUE on the `minimax_h3_ref` row on purpose — this
        // withholds ONE mapping, not the family, so `classify_video_gap` still does not claim the
        // model has no MLX engine (sc-17159's point).
        //
        // Re-enabling is a one-line revert of this arm plus deleting the matching row from
        // `KNOWN_UNCLAIMABLE_VIDEO_CAPABILITIES`; that constant is an EXACT set, so it goes red the
        // moment the pair becomes claimable and cannot be left behind.
        return false;
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
        // path, so extend/bridge on them are refused and remain queued.
        "extend_clip" | "video_bridge" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        // replace_person → native Wan-VACE (sc-3521): the engine `wan_vace` provider serves it
        // regardless of the user-picked replace-capable model (ltx_2_3 / ltx_2_3_eros / wan_2_2,
        // the models that advertise the capability), so admit those.
        //
        // `wan_2_2_vace_fun_14b` also advertises `replace_person` and was unreachable before
        // sc-17159, but it is served by its OWN arm above rather than added here: it must NOT
        // inherit this arm's neighbours (`text_to_video | image_to_video`), which the generic arm
        // would otherwise hand it. (`scail2_14b` has its own arm for the same reason.)
        "replace_person" => matches!(model, "ltx_2_3" | "ltx_2_3_eros" | "wan_2_2"),
        _ => false,
    }
}

/// Epic 3039 routing — does this `lora_train` job belong on the in-process Rust MLX
/// worker? The training sibling of
/// [`image_job_is_mlx_eligible`]/[`video_job_is_mlx_eligible`]: the engine has a
/// native trainer for the family. Both dry-run and real runs are eligible (the
/// dry-run validates the same resolved plan). LoKr-on-Wan is refused — the mlx Wan
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
    // LoKr-on-Wan is refused (no Kronecker merge in the mlx Wan path).
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
/// `mlx-gen-seedvr2`. `aura-sr` (a 617M-param historical GigaGAN backend) was dropped on Mac after
/// the sc-3668 port-or-drop spike, so the mlx worker refuses it and it remains queued.
/// Engine defaults to `real-esrgan` when absent (mirrors `run_image_upscale`).
/// SeedVR2 runs here through MLX on Mac and through the native Candle/CUDA backend on Windows/Linux
/// (sc-5928 / sc-5160). The platform capability mirrors those two production lanes.
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
/// native-MLX SeedVR2 upscaler (`mlx-gen-seedvr2`); there is no fallback. A job with any other engine
/// is refused by the mlx worker. The off-Mac candle lane mirrors this predicate for its
/// SeedVR2 provider, so both native GPU backends enforce the same contract boundary. Defaults to
/// `seedvr2` when the payload omits the engine.
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
/// web sends and the worker accepts). SeedVR2 runs on MLX (Mac) or candle (Windows/Linux), so this
/// also drives every generic worker descriptor's refusal (the inverse of the AuraSR gate).
/// The image default engine is Real-ESRGAN, so an absent engine is NOT SeedVR2.
pub(crate) fn upscale_job_requests_seedvr2(job: &JobSnapshot) -> bool {
    matches!(job.job_type, JobType::ImageUpscale)
        && job
            .payload
            .get("engine")
            .and_then(Value::as_str)
            .is_some_and(|engine| engine.trim().eq_ignore_ascii_case("seedvr2"))
}

/// Whether this training job targets a native-only kernel (see
/// [`MLX_ONLY_TRAINING_KERNELS`]). Such a job can only run on a Rust worker (mlx, or candle when the
/// candle exception in [`worker_supports_job`] admits it — e.g. `krea_control`), so any generic
/// worker descriptor must refuse it. Covers both the `lora_train` and the ControlNet studio
/// (`control_training`) jobs; both stamp a resolved plan whose `krea_control` kernel is native-only
/// (epic 10159).
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

    /// The generation variants serve plain text-to-image **and text-to-image + user adapters**
    /// (sc-15328); every other conditioning shape fails closed.
    ///
    /// The single-adapter and stacked/mixed LoRA+LoKr shapes are asserted eligible here because that
    /// is the whole capability: `mage_flow_lora`'s `limits.networkTypes` offers `lora` and `lokr`,
    /// the picker offers the trained adapter, and `apply_mage_adapters` stacks mixed kinds. Before
    /// sc-15328 the `loras` shape sat in the `unsupported` list below and no backend claimed the
    /// job — it queued forever rather than failing.
    #[test]
    fn mage_flow_routes_plain_text_to_image_and_adapters() {
        let mage_models = ["mage_flow_base", "mage_flow", "mage_flow_turbo"];
        let eligible = [
            json!({}),
            json!({ "mode": "text_to_image", "prompt": "a lighthouse in fog" }),
            json!({ "mode": "text_to_image", "loras": [], "referenceAssetIds": [] }),
            // One trained LoRA — the sc-15328 reproduction shape.
            json!({ "mode": "text_to_image", "loras": [{ "id": "adapter-1", "weight": 0.8 }] }),
            // LoKr (sc-14056 exposes it in `limits.networkTypes`) and a stacked, mixed pair — both
            // reach the same strict seam, so routing must not discriminate between them.
            json!({ "mode": "text_to_image", "loras": [{ "id": "a", "networkType": "lokr" }] }),
            json!({ "mode": "text_to_image", "loras": [
                { "id": "a", "weight": 0.8 },
                { "id": "b", "weight": 0.4, "networkType": "lokr" }
            ] }),
            // A mode-less payload with adapters is the API default (text-to-image).
            json!({ "loras": [{ "id": "adapter-1" }] }),
        ];
        for model in mage_models {
            for payload in &eligible {
                assert!(
                    image_request_mlx_eligible(
                        model,
                        payload.as_object().expect("eligible payload is an object")
                    ),
                    "{model} must route its supported plain text-to-image shape: {payload}"
                );
            }
        }

        let unsupported = [
            json!({ "mode": "edit_image", "sourceAssetId": "asset-1" }),
            json!({ "mode": "character_image", "referenceAssetId": "asset-1" }),
            json!({ "mode": "text_to_image", "sourceAssetId": "asset-1" }),
            json!({ "mode": "text_to_image", "referenceAssetId": "asset-1" }),
            json!({ "mode": "text_to_image", "referenceAssetIds": ["asset-1"] }),
            json!({ "mode": "text_to_image", "maskAssetId": "mask-1" }),
            json!({ "mode": "text_to_image", "advanced": { "poses": [{}] } }),
            json!({ "mode": "text_to_image", "controls": [{}] }),
            json!({ "mode": "text_to_image", "controlnets": [{}] }),
        ];
        for model in mage_models {
            for payload in &unsupported {
                assert!(
                    !image_request_mlx_eligible(
                        model,
                        payload
                            .as_object()
                            .expect("unsupported payload is an object")
                    ),
                    "{model} must fail closed for its unsupported request shape: {payload}"
                );
            }
        }
    }

    #[test]
    fn mage_flow_edit_requires_source_and_preserves_optional_reference_shape() {
        let models = [
            "mage_flow_edit_base",
            "mage_flow_edit",
            "mage_flow_edit_turbo",
        ];
        for model in models {
            for eligible in [
                json!({ "mode": "edit_image", "sourceAssetId": "source" }),
                json!({
                    "mode": "edit_image",
                    "sourceAssetId": "source",
                    "referenceAssetIds": ["ref-b", "ref-a"]
                }),
            ] {
                assert!(image_request_mlx_eligible(
                    model,
                    eligible.as_object().unwrap()
                ));
            }
            for rejected in [
                json!({ "mode": "edit_image", "referenceAssetIds": ["ref"] }),
                json!({ "mode": "edit_image", "sourceAssetId": "   " }),
                json!({ "mode": "text_to_image", "sourceAssetId": "source" }),
                json!({ "mode": "edit_image", "sourceAssetId": "source", "maskAssetId": "mask" }),
                json!({ "mode": "edit_image", "sourceAssetId": "source", "loras": [{"id":"x"}] }),
            ] {
                assert!(!image_request_mlx_eligible(
                    model,
                    rejected.as_object().unwrap()
                ));
            }
        }
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

    /// sc-14109 (epic 14015): an imported/user Krea 2 single-file checkpoint carries a novel id in NO
    /// routing table, but the full catalog entry the API stamps into the job (`modelManifestEntry`)
    /// declares `family: "krea_2"` — the MLX-routed family whose builtins already run in-process. A
    /// plain t2i job MUST be claim-eligible via the route-by-family fallback, or the mlx worker never
    /// claims it (`worker_supports_job`) and, with no torch/candle Krea lane on Mac, it strands on
    /// "Waiting for an available GPU worker." forever — exactly the bug this story fixes. S0d
    /// (sc-14019) had made only the display badge family-aware, leaving this claim-time predicate
    /// blindly `false` for the same id.
    #[test]
    fn imported_krea_family_t2i_is_mlx_eligible() {
        let imported_id = "user_kreamania_variant5"; // a novel id, never a builtin
        assert!(!MLX_ROUTED_MODELS.contains(&imported_id));

        let t2i = json!({
            "model": imported_id,
            "mode": "text_to_image",
            "modelManifestEntry": {
                "id": imported_id,
                "family": "krea_2",
                "paths": { "model": "/app/models/imports/kreamania_variant4" }
            },
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            t2i.as_object().expect("probe is an object")
        ));

        // A bare request with no explicit mode is equally a t2i job, equally eligible.
        let bare = json!({
            "model": imported_id,
            "modelManifestEntry": {
                "id": imported_id,
                "family": "krea_2",
                "modelPath": "/app/models/imports/kreamania_variant4.safetensors"
            },
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            bare.as_object().expect("probe is an object")
        ));
    }

    /// The route-by-family fallback fires ONLY for an MLX-routed family with a readable manifest
    /// entry. A non-routed family (the detector's `z-image`, an unsupported import), a missing
    /// manifest entry, and a manifest entry with no `family` key all stay not-eligible — unchanged
    /// from the pre-sc-14109 blanket `false`, so nothing new becomes claimable by accident.
    #[test]
    fn imported_model_without_routed_family_is_not_mlx_eligible() {
        let imported_id = "user_zimage_import";

        // z-image is a real detector family but NOT in MLX_ROUTED_FAMILIES → no family fallback.
        let z_image = json!({
            "model": imported_id,
            "mode": "text_to_image",
            "modelManifestEntry": { "id": imported_id, "family": "z-image" },
        });
        assert!(!image_request_mlx_eligible(
            imported_id,
            z_image.as_object().expect("probe is an object")
        ));

        // No manifest entry at all — nothing to read a family from.
        let no_entry = json!({ "model": imported_id, "mode": "text_to_image" });
        assert!(!image_request_mlx_eligible(
            imported_id,
            no_entry.as_object().expect("probe is an object")
        ));

        // A manifest entry present but with no `family` key.
        let no_family = json!({
            "model": imported_id,
            "mode": "text_to_image",
            "modelManifestEntry": {
                "id": imported_id,
                "paths": { "model": "/app/models/imports/user_zimage_import" }
            },
        });
        assert!(!image_request_mlx_eligible(
            imported_id,
            no_family.as_object().expect("probe is an object")
        ));
    }

    /// On MLX the imported single-file loader takes an adapter slice (inference #211), so the
    /// imported-family lane is claim-eligible for LoRAs (sc-14111), the Kontext edit surface
    /// (sc-14119, any of the edit-reference fields), AND reference-guided img2img (sc-14071) — while
    /// still rejecting the base-tier-only shapes (pose, mask, character, multi-phase) and a bare
    /// non-edit `sourceAssetId`.
    #[test]
    fn imported_krea_family_adapters_and_edit_are_mlx_eligible() {
        let imported_id = "user_kreamania_variant5";
        let entry = json!({
            "id": imported_id,
            "family": "krea_2",
            "paths": { "model": "/app/models/imports/kreamania_variant4" }
        });

        // Edit + a source image → eligible (the required identity-edit LoRA is worker-enforced).
        let edit = json!({
            "model": imported_id,
            "mode": "edit_image",
            "sourceAssetId": "asset-1",
            "modelManifestEntry": entry.clone(),
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            edit.as_object().expect("probe is an object")
        ));

        // Edit via the two-reference scene+person set → equally eligible.
        let edit_two_ref = json!({
            "model": imported_id,
            "mode": "edit_image",
            "referenceAssetIds": ["scene", "person"],
            "modelManifestEntry": entry.clone(),
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            edit_two_ref.as_object().expect("probe is an object")
        ));

        // An edit with NO conditioning image is still rejected (defensive shape).
        let edit_no_ref = json!({
            "model": imported_id,
            "mode": "edit_image",
            "modelManifestEntry": entry.clone(),
        });
        assert!(!image_request_mlx_eligible(
            imported_id,
            edit_no_ref.as_object().expect("probe is an object")
        ));

        // A t2i job carrying a LoRA stack → eligible (adapter path).
        let t2i_lora = json!({
            "model": imported_id,
            "mode": "text_to_image",
            "loras": [{ "id": "some_adapter" }],
            "modelManifestEntry": entry.clone(),
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            t2i_lora.as_object().expect("probe is an object")
        ));

        // A non-edit img2img job (a single referenceAssetId) → eligible (sc-14071, no adapter needed).
        let img2img = json!({
            "model": imported_id,
            "referenceAssetId": "asset-1",
            "modelManifestEntry": entry.clone(),
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            img2img.as_object().expect("probe is an object")
        ));

        // A strict-pose set → eligible: the trained pose control branch folds onto the file-loaded
        // imported DiT (the MLX native control entrypoint), one pose-locked image per pose.
        let pose = json!({
            "model": imported_id,
            "advanced": { "poses": [{}] },
            "modelManifestEntry": entry.clone(),
        });
        assert!(image_request_mlx_eligible(
            imported_id,
            pose.as_object().expect("probe is an object")
        ));

        // A pose set whose conditioning the pose render loop would silently drop (the plural edit
        // reference set) stays rejected — never flattened into an unposed or unreferenced render.
        let pose_with_reference_list = json!({
            "model": imported_id,
            "referenceAssetIds": ["scene", "person"],
            "advanced": { "poses": [{}] },
            "modelManifestEntry": entry.clone(),
        });
        assert!(!image_request_mlx_eligible(
            imported_id,
            pose_with_reference_list
                .as_object()
                .expect("probe is an object")
        ));

        // A bare non-edit `sourceAssetId` is still rejected (the img2img path reads referenceAssetId).
        let bare_source = json!({
            "model": imported_id,
            "sourceAssetId": "asset-1",
            "modelManifestEntry": entry.clone(),
        });
        assert!(!image_request_mlx_eligible(
            imported_id,
            bare_source.as_object().expect("probe is an object")
        ));
    }

    /// Builtin id-keyed routing is untouched by the fallback: a routed builtin still routes on its id
    /// alone, and a manifest `family` on a builtin payload is ignored (the `MLX_ROUTED_MODELS`
    /// short-circuit runs before the fallback is ever consulted). Guards the sc-14109 constraint that
    /// builtin routing stays byte-identical (with `every_mlx_routed_model_has_a_dispatch_arm` above).
    #[test]
    fn builtin_image_routing_unaffected_by_family_fallback() {
        // A routed builtin routes on its id, with or without a manifest family present.
        let builtin_t2i = json!({ "model": "krea_2_raw", "mode": "text_to_image" });
        assert!(image_request_mlx_eligible(
            "krea_2_raw",
            builtin_t2i.as_object().expect("probe is an object")
        ));
        let builtin_with_family = json!({
            "model": "krea_2_raw",
            "mode": "text_to_image",
            "modelManifestEntry": { "id": "krea_2_raw", "family": "krea_2" },
        });
        assert!(image_request_mlx_eligible(
            "krea_2_raw",
            builtin_with_family.as_object().expect("probe is an object")
        ));

        // An unsupported builtin id-keyed verdict is unchanged: `flux_dev` edit stays off MLX even if a
        // (spurious) krea_2 manifest family rides along — the builtin arm decides, never the fallback.
        let builtin_edit = json!({
            "model": "flux_dev",
            "mode": "edit_image",
            "sourceAssetId": "asset-1",
            "modelManifestEntry": { "id": "flux_dev", "family": "krea_2" },
        });
        assert!(!image_request_mlx_eligible(
            "flux_dev",
            builtin_edit.as_object().expect("probe is an object")
        ));
    }
}
