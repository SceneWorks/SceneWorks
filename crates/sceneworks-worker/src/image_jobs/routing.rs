#[cfg(target_os = "macos")]
use super::artifact_resolution::resolve_weights_dir;
#[cfg(target_os = "macos")]
use super::bernini::bernini_image_available;
#[cfg(not(target_os = "macos"))]
use super::bernini::bernini_image_candle_available;
#[cfg(not(target_os = "macos"))]
use super::bernini::CANDLE_BERNINI_IMAGE_ADAPTER;
use super::checkpoint_plan::prepare_checkpoint_plan_sources;
use super::checkpoint_plan::PreparedCheckpointPlanSources;
#[cfg(not(target_os = "macos"))]
use super::checkpoint_plan::CHECKPOINT_PLAN_ENGINE;
#[cfg(target_os = "macos")]
use super::checkpoint_plan::CHECKPOINT_PLAN_ENGINE;
#[cfg(not(target_os = "macos"))]
use super::conditioning::candle_conditioned_angle_set;
#[cfg(target_os = "macos")]
use super::conditioning::edit_grouping;
#[cfg(target_os = "macos")]
use super::conditioning::grouped_edit_image_count;
#[cfg(not(target_os = "macos"))]
use super::conditioning::is_mage_edit_model;
#[cfg(not(target_os = "macos"))]
use super::conditioning::is_qwen_image_2_1_edit;
#[cfg(not(target_os = "macos"))]
use super::conditioning::kolors_edit_candle_available;
use super::conditioning::non_empty;
use super::conditioning::pose_entries;
#[cfg(not(target_os = "macos"))]
use super::conditioning::sensenova_edit_candle_available;
#[cfg(target_os = "macos")]
use super::conditioning::EditGrouping;
#[cfg(target_os = "macos")]
use super::flux1_control::flux1_dev_control_available;
#[cfg(target_os = "macos")]
use super::flux1_control::FLUX1_DEV_CONTROL_ENGINE_ID;
#[cfg(not(target_os = "macos"))]
use super::flux1_control_candle;
#[cfg(not(target_os = "macos"))]
use super::flux1_control_candle_available;
#[cfg(target_os = "macos")]
use super::flux2::flux2_dev_control_available;
#[cfg(target_os = "macos")]
use super::flux2::flux2_edit_available;
#[cfg(target_os = "macos")]
use super::flux2::FLUX2_DEV_CONTROL_ENGINE_ID;
#[cfg(not(target_os = "macos"))]
use super::flux2_comfyui_candle;
#[cfg(not(target_os = "macos"))]
use super::flux2_control_candle;
#[cfg(not(target_os = "macos"))]
use super::flux2_control_candle_available;
#[cfg(not(target_os = "macos"))]
use super::flux2_edit_candle;
#[cfg(not(target_os = "macos"))]
use super::flux2_edit_candle_available;
#[cfg(not(target_os = "macos"))]
use super::flux_ipadapter;
#[cfg(not(target_os = "macos"))]
use super::flux_ipadapter_available;
use super::instantid::instantid_available;
use super::instantid::instantid_image_count;
#[cfg(not(target_os = "macos"))]
use super::instantid::INSTANTID_ENGINE;
#[cfg(target_os = "macos")]
use super::instantid::INSTANTID_ENGINE;
#[cfg(target_os = "macos")]
use super::kolors::kolors_control_available;
#[cfg(not(target_os = "macos"))]
use super::kolors_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::kolors_control::kolors_control_available;
#[cfg(not(target_os = "macos"))]
use super::kolors_ipadapter;
#[cfg(not(target_os = "macos"))]
use super::kolors_ipadapter_available;
#[cfg(target_os = "macos")]
use super::krea_control::krea_control_available;
#[cfg(target_os = "macos")]
use super::krea_control::KREA_CONTROL_ENGINE_ID;
#[cfg(not(target_os = "macos"))]
use super::krea_control_candle;
#[cfg(not(target_os = "macos"))]
use super::krea_control_candle_available;
#[cfg(target_os = "macos")]
use super::krea_edit::krea_edit_available;
#[cfg(not(target_os = "macos"))]
use super::krea_edit_candle;
#[cfg(not(target_os = "macos"))]
use super::krea_edit_candle_available;
#[cfg(test)]
use super::krea_imported::krea_imported_available;
#[cfg(all(
    native_lane,
    any(
        all(target_os = "macos", test),
        all(not(target_os = "macos"), feature = "backend-candle")
    ),
    any(
        all(target_os = "macos", test),
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
use super::krea_imported::krea_imported_control_available;
#[cfg(target_os = "macos")]
use super::krea_imported::prepare_krea_imported_control_sources;
use super::krea_imported::prepare_krea_imported_sources;
#[cfg(target_os = "macos")]
use super::krea_imported::PreparedKreaImportedControlSources;
use super::krea_imported::PreparedKreaImportedSources;
#[cfg(not(target_os = "macos"))]
use super::krea_imported::KREA_IMPORTED_ENGINE;
#[cfg(target_os = "macos")]
use super::krea_imported::KREA_IMPORTED_ENGINE;
use super::krea_multiphase::krea_multiphase_available;
use super::krea_multiphase::request_has_multiphase;
use super::krea_turbo_raw::krea_turbo_on_raw_available;
#[cfg(test)]
use super::mage_finetuned::mage_finetuned_available;
use super::mage_finetuned::prepare_mage_finetuned_transformer;
use super::mage_finetuned::PreparedMageFinetunedTransformer;
#[cfg(not(target_os = "macos"))]
use super::mage_finetuned::MAGE_FINETUNED_ENGINE;
#[cfg(target_os = "macos")]
use super::mage_finetuned::MAGE_FINETUNED_ENGINE;
use super::mlx_model;
#[cfg(not(target_os = "macos"))]
use super::moss::moss_adapter_label;
#[cfg(target_os = "macos")]
use super::pulid::pulid_flux_available;
#[cfg(target_os = "macos")]
use super::pulid::PULID_ADAPTER_LABEL;
#[cfg(not(target_os = "macos"))]
use super::pulid_candle;
#[cfg(not(target_os = "macos"))]
use super::pulid_candle_available;
#[cfg(target_os = "macos")]
use super::qwen::qwen_control_available;
#[cfg(target_os = "macos")]
use super::qwen::qwen_edit_available;
#[cfg(not(target_os = "macos"))]
use super::qwen_comfyui_candle;
#[cfg(not(target_os = "macos"))]
use super::qwen_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::qwen_control::qwen_control_available;
#[cfg(not(target_os = "macos"))]
use super::qwen_edit_candle;
#[cfg(not(target_os = "macos"))]
use super::qwen_edit_candle_available;
#[cfg(target_os = "macos")]
use super::sdxl::sdxl_advanced_available;
use super::sdxl_control::sdxl_control_candidate;
use super::sdxl_control::SDXL_CONTROL_ADAPTER_LABEL;
#[cfg(not(target_os = "macos"))]
use super::sdxl_edit_candle;
#[cfg(not(target_os = "macos"))]
use super::sdxl_edit_candle_available;
use super::sdxl_imported::prepare_sdxl_imported_sources;
#[cfg(test)]
use super::sdxl_imported::sdxl_imported_available;
use super::sdxl_imported::PreparedSdxlImportedSources;
#[cfg(not(target_os = "macos"))]
use super::sdxl_imported::SDXL_IMPORTED_ENGINE;
#[cfg(target_os = "macos")]
use super::sdxl_imported::SDXL_IMPORTED_ENGINE;
#[cfg(not(target_os = "macos"))]
use super::sdxl_ipadapter;
#[cfg(not(target_os = "macos"))]
use super::sdxl_ipadapter_available;
#[cfg(target_os = "macos")]
use super::sensenova::sensenova_edit_available;
#[cfg(target_os = "macos")]
use super::zimage::zimage_base_control_available;
#[cfg(target_os = "macos")]
use super::zimage::zimage_control_available;
#[cfg(target_os = "macos")]
use super::zimage::CHARACTER_ANGLE_SET_ORDER;
#[cfg(not(target_os = "macos"))]
use super::zimage_comfyui_candle;
#[cfg(not(target_os = "macos"))]
use super::zimage_control;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use super::zimage_control::zimage_control_available;
#[cfg(not(target_os = "macos"))]
use super::zimage_edit_candle_available;
#[cfg(not(target_os = "macos"))]
use super::zimage_identity_candle_available;
use super::ImagePlan;
use super::ImageRequest;
#[cfg(not(target_os = "macos"))]
use super::CANDLE_ADAPTER;
use super::STUB_ADAPTER;
#[cfg(not(target_os = "macos"))]
use crate::advanced;
#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
use crate::Path;
#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
use crate::PathBuf;
use crate::Settings;
#[cfg(not(target_os = "macos"))]
use crate::Value;
#[cfg(not(target_os = "macos"))]
use crate::WorkerError;
use crate::WorkerResult;
/// Resolve one exact imported generate surface and reject every request axis the selected provider
/// cannot execute. Source shape is part of the registry key: a sibling route for the same family is
/// never unioned in. The ComfyUI lanes are ordinary Generate routes; only a singular Reference is
/// admitted when that exact descriptor advertises it.
pub(super) fn imported_generate_request_has_unsupported_shape(request: &ImageRequest) -> bool {
    request.mode == "edit_image"
        || !pose_entries(request).is_empty()
        || request.source_asset_id.is_some()
        || request.mask_asset_id.is_some()
        || request.character_id.is_some()
        || request.character_look_id.is_some()
        || !request.reference_asset_ids.is_empty()
        || request_has_multiphase(request)
        || sceneworks_core::jobs_store::imported_control_intent_is_material(&request.advanced)
}

pub(super) fn imported_generate_request_supported(
    request: &ImageRequest,
    family: &str,
    source: gen_core::ImportedModelSource,
) -> Option<gen_core::ModelDescriptor> {
    // Reject shape before registry/path resolution so every shared Generate consumer (Mage and the
    // three ComfyUI Candle lanes) refuses material control intent at the same admission seam.
    if imported_generate_request_has_unsupported_shape(request) {
        return None;
    }
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        family,
        source,
        gen_core::ImportedModelOperation::Generate,
    )?;
    if request.reference_asset_id.is_some()
        && !descriptor
            .capabilities
            .conditioning
            .contains(&gen_core::ConditioningKind::Reference)
    {
        return None;
    }
    if (!request.loras.is_empty())
        && !(descriptor.capabilities.supports_lora || descriptor.capabilities.supports_lokr)
    {
        return None;
    }
    Some(descriptor)
}

#[cfg(target_os = "macos")]
fn mlx_available(request: &ImageRequest, settings: &Settings) -> bool {
    mlx_model(&request.model).is_some()
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImageRoute {
    /// FLUX.1's IP-Adapter and strict-control providers are separate graphs.  A reference plus
    /// pose set must not be claimed by the control route, which would otherwise consume the pose
    /// while treating the reference only as post-generation scoring input.
    FluxIpAdapterPoseReject,
    ZImageControl,
    ZImageBaseControl,
    QwenControl,
    KolorsControl,
    KreaControl,
    Flux1DevControl,
    Flux2DevControl,
    Flux2Edit,
    QwenEdit,
    KreaEdit,
    /// A Krea 2 **Raw** t2i job carrying the accelerator (turbo) LoRA (`role: accelerator`, sc-13882)
    /// → the single-phase turbo-on-Raw lane (epic 13879 S3, sc-13883): the Raw base + LoRA additive,
    /// sampled through the distilled Turbo regime (fixed mu 1.15 / ~8 steps / CFG-off) by routing to
    /// the `krea_2_turbo` engine. Wins over the generic `Mlx` arm below — `krea_2_raw` is in
    /// MODEL_TABLE, so `mlx_available` would otherwise render it as plain Raw t2i (52-step true-CFG)
    /// and never accelerate. The t2i sibling of [`ImageRoute::KreaEdit`].
    KreaTurboOnRaw,
    /// A Krea 2 **Raw** t2i job carrying an explicit `advanced.phases` list (epic 13879 S4, sc-13884)
    /// → the multi-phase denoise driver: ONE Raw trajectory over ONE global sigma schedule, each phase
    /// a contiguous step slice with its own guidance (per-phase CFG on/off) and its own active subset of
    /// the job's load-time LoRA stack (per-phase toggling). Wins over [`ImageRoute::KreaTurboOnRaw`] and
    /// the generic `Mlx` arm — an explicit phase list is the FINER-GRAINED control and takes precedence
    /// over S3's whole-job turbo-on-Raw regime. Loads the `krea_2_raw` engine (the multi-phase driver
    /// keys on that descriptor id), unlike S3 which swaps to `krea_2_turbo`. Reference/edit/pose/PiD
    /// shapes are rejected loudly by the lane (multi-phase renders from pure noise).
    KreaMultiPhase,
    /// A user model bound to a persisted checkpoint import plan (epic 20398, sc-20634): its manifest
    /// entry carries `importPlan.checkpointId`, the plan store resolves and re-verifies the plan, and
    /// the provider is selected by family + source shape + operation through the registry. Claimed
    /// BEFORE every bespoke imported lane, but PER REQUEST, not per entry (sc-20636): this skeleton
    /// serves plain text-to-image only, so a managed entry's family lane keeps the edit / pose /
    /// reference / LoRA / Hires.fix shapes it had before the import. Exactly one lane owns each
    /// REQUEST; the entry itself may be served by two. A request no lane claims at all is a typed
    /// refusal, never the stub.
    CheckpointPlan,
    /// An imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): a non-builtin
    /// `krea_2`-family model whose `modelPath` is a single `.safetensors` DiT → the bespoke in-place
    /// assembly lane, which pairs the imported transformer with a resident `krea_2` base tier (shared
    /// Qwen3-VL TE / Qwen VAE / tokenizer) and loads through the native MLX or Candle single-file
    /// entrypoint. A
    /// builtin Krea id (`krea_2_turbo` / `krea_2_raw`, in `MODEL_TABLE`) never reaches here —
    /// `resolve_imported_krea_dit` returns `None` for it, so the snapshot-dir Krea path is untouched.
    /// The resident base tier supplies the img2img/edit conditioning and LoRA/LoKr components.
    KreaImported,
    /// An imported/user single-file Krea 2 checkpoint carrying a **strict-pose set** (a non-empty
    /// `advanced.poses` outside edit mode): the trained pose control-branch overlay rides the
    /// FILE-LOADED imported DiT via the native MLX or Candle control entrypoint, one image per pose
    /// — the imported twin of
    /// [`ImageRoute::KreaControl`]. Claimed BEFORE the plain [`ImageRoute::KreaImported`] arm so a
    /// pose set gets the per-pose count + control render instead of per-image generation.
    KreaImportedControl,
    /// A fused SDXL LDM/A1111 single-file checkpoint. The file carries the UNet, both text encoders,
    /// and VAE; tokenizer assets are borrowed from the installed SDXL base turnkey.
    SdxlImported,
    /// This app's OWN full base fine-tune of Mage-Flow Base (sc-15036, epic 14034 F6): a non-builtin
    /// `mage-flow`-family catalog entry whose `paths.model` is a complete diffusers transformer
    /// component dir → the bespoke fine-tuned lane, which stages the installed base's shared text
    /// encoder + VAE and loads through `mage::load_finetuned`. A builtin Mage id never reaches here
    /// (`resolve_mage_finetuned_transformer` returns `None` for a `MODEL_TABLE` id), so the tiered
    /// snapshot path is untouched. txt2img only — the non-edit Mage variants advertise no
    /// conditioning and the fine-tuned entrypoint refuses adapters.
    MageFinetuned,
    InstantId,
    /// Generic SDXL OpenPose ControlNet, shared by the five independently accepted model ids.
    SdxlControl,
    PulidFlux,
    SdxlAdvanced,
    SensenovaEdit,
    Bernini,
    /// A strict-pose job on a WIRED MLX pose family (one with a `…_control_available` lane, i.e. a
    /// MLX wired pose id) whose control base/overlay is NOT installed — its
    /// `…_control_available` gate failed, so it reached the fall-through. Reject loudly instead of
    /// falling through to `Mlx` (plain txt2img) and silently dropping the poses (sc-11796 generalized to
    /// every wired family, sc-11814) — the MLX twin of the candle `CandleImageRoute::PoseControlBaseMissing`.
    PoseControlBaseMissing,
    /// A strict-pose job on an MLX model with NO pose-control lane that `mlx_available` would
    /// otherwise render as plain txt2img, dropping the poses. Reject loudly (sc-5968) — the MLX twin of
    /// the candle `CandleImageRoute::PoseReject`.
    PoseReject,
    Mlx,
}

/// Distinguish a missing installed control base from a model with no pose lane. A shipped
/// text-to-image route is wired when the native strict-control table registers its provider's
/// control engine; `z_image_edit` shares the Turbo provider but is edit-only, so it is excluded.
#[cfg(native_lane)]
fn wired_pose_family(model: &str, backend: &str) -> bool {
    sceneworks_core::registry::builtin_model_spec(model).is_some_and(|spec| {
        spec.product
            .fields
            .get("capabilities")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|caps| caps.iter().any(|cap| cap == "text_to_image"))
            && spec.routes.get(backend).is_some_and(|route| {
                super::strict_control::strict_control_engine(&format!("{}_control", route.id))
                    .is_some()
            })
    })
}

#[cfg(native_lane)]
pub(super) fn wired_pose_families(backend: &str) -> Vec<String> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("shipped builtin manifest")
        .1;
    let manifest: serde_json::Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
            .expect("shipped manifest parses");
    manifest["models"]
        .as_array()
        .expect("shipped models")
        .iter()
        .filter_map(|row| row["id"].as_str())
        .filter(|id| wired_pose_family(id, backend))
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
#[test]
fn wired_pose_membership_follows_typed_routes_and_control_engines() {
    let expected: std::collections::BTreeSet<String> = [
        "z_image_turbo",
        "z_image",
        "qwen_image",
        "kolors",
        "krea_2_turbo",
        "flux_dev",
        "flux2_dev",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    for backend in ["mlx", "candle"] {
        assert_eq!(
            wired_pose_families(backend)
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
        );
        assert!(!wired_pose_family("sdxl", backend));
        assert!(!wired_pose_family("chroma1_base", backend));
    }
}

/// FLUX.1-dev has independent IP-Adapter and strict-control providers, neither of which composes
/// the other's conditioning. Keep this predicate at route selection so a combined request is
/// refused before either provider can silently discard the other input.
#[cfg(native_lane)]
fn flux1_ipadapter_pose_combination(request: &ImageRequest) -> bool {
    request.model == "flux_dev"
        && request.mode != "edit_image"
        && non_empty(&request.reference_asset_id)
        && !pose_entries(request).is_empty()
}

/// The production FLUX strict-control router, expressed as the exact base-model → dedicated
/// provider mapping used by the two `..._control_available` arms below. Keeping this pure seam next
/// to [`prepare_image_route`] lets source-bound audits ask the router whether a model really has a
/// strict-control lane instead of attaching arbitrary control bytes to a base provider.
///
/// Chroma, FLUX.1 Schnell, and FLUX.2 Klein deliberately resolve to `None`: none ships a control
/// checkpoint. Only the two Dev bases route to their dedicated registry providers.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_flux_strict_control_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "flux_dev" => Some(FLUX1_DEV_CONTROL_ENGINE_ID),
        "flux2_dev" => Some(FLUX2_DEV_CONTROL_ENGINE_ID),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn resolve_image_route_with_imported_availability(
    request: &ImageRequest,
    settings: &Settings,
    checkpoint_plan_available: bool,
    imported_control_available: bool,
    imported_available: bool,
    sdxl_imported_available: bool,
    mage_finetuned_available: bool,
) -> Option<ImageRoute> {
    if flux1_ipadapter_pose_combination(request) {
        Some(ImageRoute::FluxIpAdapterPoseReject)
    } else if zimage_control_available(request, settings) {
        Some(ImageRoute::ZImageControl)
    } else if zimage_base_control_available(request, settings) {
        // Base (non-distilled, full-CFG) Z-Image strict control (advanced.poses on `z_image`) →
        // base Fun-Controlnet-Union (`z_image_control`). The base mirror of the Turbo control arm
        // above; keyed on the base model id so the Turbo path is untouched (sc-8251).
        Some(ImageRoute::ZImageBaseControl)
    } else if qwen_control_available(request, settings) {
        Some(ImageRoute::QwenControl)
    } else if kolors_control_available(request, settings) {
        Some(ImageRoute::KolorsControl)
    } else if krea_control_available(request, settings) {
        // Krea 2 Turbo strict pose (advanced.poses on `krea_2_turbo`) → the trained control-branch
        // overlay (sc-8465, epic 8459 S5). Wins over the generic `mlx_available` arm below — `krea_2_turbo`
        // is in MODEL_TABLE, so `mlx_available` would otherwise render it as plain t2i and silently drop
        // the poses. The MLX twin of the candle `CandleImageRoute::KreaControl` resolver arm.
        Some(ImageRoute::KreaControl)
    } else if flux1_dev_control_available(request, settings) {
        // FLUX.1-dev strict control (advanced.poses on flux_dev) → Shakker Union-Pro-2.0. Wins over the
        // PuLID-FLUX / generic MLX arms below: a flux_dev pose job is the real ControlNet path (sc-8244).
        Some(ImageRoute::Flux1DevControl)
    } else if flux2_dev_control_available(request, settings) {
        // FLUX.2-dev strict pose (advanced.poses) → Fun-Controlnet-Union. Wins over the edit/
        // best-effort pose tier below (`flux2_edit_available` needs a reference; a flux2_dev pose
        // job is the real ControlNet path, with the reference an opt-in img2img-init).
        Some(ImageRoute::Flux2DevControl)
    } else if flux2_edit_available(request, settings) {
        Some(ImageRoute::Flux2Edit)
    } else if qwen_edit_available(request, settings) {
        Some(ImageRoute::QwenEdit)
    } else if krea_multiphase_available(request, settings) {
        // Krea 2 Raw t2i + an explicit `advanced.phases` list (epic 13879 S4, sc-13884) → the
        // multi-phase denoise lane. Placed FIRST among the `krea_2_raw` lanes so an explicit phase list
        // takes precedence over the S3 whole-job turbo-on-Raw regime (`krea_turbo_on_raw_available`
        // below) AND the generic Raw t2i (`mlx_available`): multi-phase is the finer-grained control.
        // Only claims `krea_2_raw` + a present `advanced.phases`, so every non-phases job is unaffected;
        // the lane itself rejects edit/pose/reference/PiD shapes loudly (multi-phase renders from noise),
        // so claiming a conflicting shape here surfaces a clear error rather than diverting it silently.
        Some(ImageRoute::KreaMultiPhase)
    } else if krea_edit_available(request, settings) {
        // Krea 2 Raw Kontext-style edit (mode edit_image + a source) → the `krea_2_edit` engine
        // (epic 10871). Wins over the generic MLX arm below — `krea_2_raw` is in MODEL_TABLE, so
        // `mlx_available` would otherwise render it as plain t2i and silently drop the source.
        Some(ImageRoute::KreaEdit)
    } else if krea_turbo_on_raw_available(request, settings) {
        // Krea 2 Raw t2i + the accelerator (turbo) LoRA (sc-13882) → the single-phase turbo-on-Raw
        // lane (epic 13879 S3, sc-13883): the Raw base + LoRA additive, sampled as Turbo (fixed mu
        // 1.15 / ~8 steps / CFG-off) by routing to the `krea_2_turbo` engine. Wins over the generic
        // `mlx_available` arm below for PLAIN t2i — a plain Raw t2i would otherwise run the 52-step
        // true-CFG regime and ignore the accelerator's intent. A `krea_2_raw` job that ALSO carries an
        // img2img reference (`ui.img2img` + `referenceAssetId`) is EXCLUDED by the gate and falls
        // through to the generic `mlx_available` img2img arm (reference honored, accelerator LoRA still
        // additive) — turbo-on-Raw img2img is out of scope for this t2i story (sc-13883). The t2i
        // sibling of the `krea_edit_available` arm above.
        Some(ImageRoute::KreaTurboOnRaw)
    } else if checkpoint_plan_available {
        // A plan-backed user checkpoint (epic 20398, sc-20634): the persisted ImportPlanV1 is the
        // route's sole source of truth. Claimed before every bespoke imported lane.
        Some(ImageRoute::CheckpointPlan)
    } else if imported_control_available {
        // An imported single-file Krea 2 checkpoint + a strict-pose set: the pose control branch
        // rides the file-loaded imported DiT (the imported twin of the `KreaControl` arm above).
        // Checked BEFORE the plain imported arm so a pose set renders one pose-locked image per
        // pose instead of falling into per-image t2i (which would silently drop the poses).
        Some(ImageRoute::KreaImportedControl)
    } else if imported_available {
        // An imported/user single-file Krea 2 checkpoint (epic 14015 S0c, sc-14018): a non-builtin
        // `krea_2`-family model whose `modelPath` is a single `.safetensors` DiT → the bespoke in-place
        // assembly lane. A builtin Krea id never claims this (`resolve_imported_krea_dit` returns `None`
        // for a `MODEL_TABLE` id), so the generic registry-backed snapshot-dir arms are unchanged for
        // builtin Krea. Imported ids are not in `MODEL_TABLE`; the bespoke MLX/Candle gates claim them
        // here for text-to-image, img2img/edit, and adapter-bearing generation.
        Some(ImageRoute::KreaImported)
    } else if sdxl_imported_available {
        Some(ImageRoute::SdxlImported)
    } else if mage_finetuned_available {
        // A fine-tuned Mage-Flow base (sc-15036). The fine-tune's id is in no `MODEL_TABLE`, so
        // `mlx_available` is `false` for it — this arm is what routes it to real MLX generation at
        // all. A builtin Mage id is claimed by the generic `mlx_available` arm below, unchanged.
        Some(ImageRoute::MageFinetuned)
    } else if instantid_available(request, settings) {
        Some(ImageRoute::InstantId)
    } else if sdxl_control_candidate(request) {
        // Keep the generic SDXL ControlNet route after InstantID and before every SDXL edit,
        // IP-Adapter, and generic route. The handler validates conflicts and malformed carriers,
        // so a pose can never be silently discarded by a sibling conditioned path.
        Some(ImageRoute::SdxlControl)
    } else if pulid_flux_available(request, settings) {
        Some(ImageRoute::PulidFlux)
    } else if sdxl_advanced_available(request, settings) {
        Some(ImageRoute::SdxlAdvanced)
    } else if sensenova_edit_available(request, settings) {
        Some(ImageRoute::SensenovaEdit)
    } else if bernini_image_available(request, settings) {
        // Bernini still-image companion (sc-5424): t2i / i2i on the `bernini_image` id. Must win
        // over the generic `mlx_available` arm below — `bernini_image` is in MODEL_TABLE (so
        // `mlx_available` would match it), but the generic `generate_stream` leaves `frames`/
        // `video_mode` unset, which the engine treats as a multi-frame video request.
        Some(ImageRoute::Bernini)
    } else if request.mode != "edit_image"
        && !pose_entries(request).is_empty()
        && (wired_pose_family(&request.model, "mlx") || mlx_available(request, settings))
    {
        // A strict-pose job that fell past every `…_control_available` lane above (and the edit / identity /
        // bernini lanes) must be REJECTED, not silently rendered as plain txt2img with the poses dropped —
        // the MLX twin of the candle fall-through reject (base.rs, sc-11171/F-008 + sc-5968). Two sub-cases,
        // distinguished by whether the family has a wired MLX pose lane at all:
        //  - WIRED MLX pose family (derived wired MLX pose family): its control lane exists but the base/overlay
        //    snapshot is absent (the lane's `…_control_available` weight-gate failed) → `PoseControlBaseMissing`.
        //    Fires regardless of whether the plain base weights resolve, because the control gate can fail while
        //    `mlx_available` succeeds — for `krea_2_turbo` the control base (`resolve_krea_control_base`) diverges
        //    from the txt2img base (the reported sc-11796 silent-drop), and for `kolors` the lane additionally
        //    needs a `referenceAssetId`. Generalizes the sc-11796 krea-only reject to every wired family (sc-11814).
        //  - A non-wired MLX pose family that `mlx_available` would render as plain txt2img →
        //    the sc-5968 no-silent-T2I `PoseReject`.
        // Checked BEFORE the generic `mlx_available` arm.
        if wired_pose_family(&request.model, "mlx") {
            Some(ImageRoute::PoseControlBaseMissing)
        } else {
            Some(ImageRoute::PoseReject)
        }
    } else if mlx_available(request, settings) {
        Some(ImageRoute::Mlx)
    } else {
        None
    }
}

/// Plan-route availability for the test-only route probes, with refusals made loud.
///
/// `prepare_checkpoint_plan_sources` has four outcomes: not plan-backed (an empty selection),
/// servable (`is_available()`), a typed refusal (`Err`), and a DECLINE that retains the refusal for
/// the router's zero-lane fall-through. Only the first is "this route is unavailable"; the other
/// three mean the route OWNS the entry. Folding the refusal into `false` made every plan refusal
/// indistinguishable from a missing route in the probes.
///
/// Gated per lane rather than a bare `#[cfg(test)]`: this helper only exists where a route probe
/// does, and `tests/test_builtin_manifest_audit.py` treats the first line-initial `#[cfg(test)]`
/// as the start of this file's test module — a bare one here would truncate the audited region to
/// the first 400 lines and silently drop base.rs's `repo` / `modelPath` lanes from the inventory.
#[cfg(all(test, native_lane))]
fn checkpoint_plan_available_or_panic(request: &ImageRequest, settings: &Settings) -> bool {
    match prepare_checkpoint_plan_sources(request, settings) {
        Ok(selection) => selection.is_available(),
        Err(error) => panic!(
            "the plan-driven route refused {:?} while probing which route claims it; probe with \
             `prepare_checkpoint_plan_sources` directly to assert on a refusal: {error}",
            request.model
        ),
    }
}

/// The companion directories a candle floor prices for an imported single-file load that pairs a
/// user checkpoint with a resident family base snapshot.
///
/// The base snapshot is a COMPANION, never the weights: the imported file REPLACES the snapshot's
/// `transformer/`, so pricing the snapshot directory recursively charges the DiT twice — the exact
/// double-claim `base_admission::imported_file_floor_excludes_the_replaced_snapshot_transformer`
/// exists to forbid. A selected encoder is already represented by the spec's prepared contract
/// receipt, so the bundled `text_encoder/` is priced only when the spec has none of its own.
///
/// Shared rather than duplicated: the legacy Krea imported lane and the plan-driven checkpoint
/// route must be admitted on an identical floor for the same file + base, and equality by
/// construction is the only version of that which cannot drift.
#[cfg(any(all(not(target_os = "macos"), feature = "backend-candle"), test))]
pub(crate) fn imported_base_snapshot_companions(
    base_dir: &Path,
    spec_has_selected_text_encoder: bool,
) -> Vec<PathBuf> {
    let mut companions = vec![base_dir.join("vae")];
    if !spec_has_selected_text_encoder {
        companions.push(base_dir.join("text_encoder"));
    }
    companions
}

/// Test-facing pure route probe. Production uses [`prepare_image_route`] so payload-selected File
/// sources stay pinned from selection through dispatch instead of being resolved a second time after
/// the async preamble.
#[cfg(all(target_os = "macos", test))]
pub(super) fn resolve_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<ImageRoute> {
    resolve_image_route_with_imported_availability(
        request,
        settings,
        // Loud, not `matches!`: a plan-backed entry that REFUSES is a typed diagnostic, and
        // collapsing it into "route unavailable" would make every refusal look like a missing
        // route in the probes (sc-20634 review). Callers that want to observe a refusal call
        // `prepare_checkpoint_plan_sources` directly.
        checkpoint_plan_available_or_panic(request, settings),
        krea_imported_control_available(request, settings),
        krea_imported_available(request, settings),
        sdxl_imported_available(request, settings),
        mage_finetuned_available(request, settings),
    )
}

#[cfg(target_os = "macos")]
pub(super) enum PreparedImageRoute {
    Plain(ImageRoute),
    CheckpointPlan(Box<PreparedCheckpointPlanSources>),
    KreaImported(Box<PreparedKreaImportedSources>),
    KreaImportedControl(Box<PreparedKreaImportedControlSources>),
    SdxlImported(Box<PreparedSdxlImportedSources>),
    MageFinetuned(Box<PreparedMageFinetunedTransformer>),
}

#[cfg(target_os = "macos")]
impl PreparedImageRoute {
    pub(super) fn kind(&self) -> ImageRoute {
        match self {
            Self::Plain(route) => *route,
            Self::CheckpointPlan(_) => ImageRoute::CheckpointPlan,
            Self::KreaImported(_) => ImageRoute::KreaImported,
            Self::KreaImportedControl(_) => ImageRoute::KreaImportedControl,
            Self::SdxlImported(_) => ImageRoute::SdxlImported,
            Self::MageFinetuned(_) => ImageRoute::MageFinetuned,
        }
    }
}

/// Resolve the production route and retain every payload-selected File token it owns. Selection is
/// synchronous and fallible; the resulting route value survives all later awaits and is consumed by
/// the matching handler.
#[cfg(target_os = "macos")]
pub(super) fn prepare_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedImageRoute>> {
    // The plan-driven route is offered the request first. It claims the shapes it serves; a shape
    // it does not serve declines to the entry's own bespoke family lane when the entry has loadable
    // bytes, and refuses outright when it does not. The decline RETAINS its refusal, re-raised
    // below if no lane claims the request at all (sc-20636).
    let checkpoint_plan = prepare_checkpoint_plan_sources(request, settings)?;
    let imported_control = prepare_krea_imported_control_sources(request, settings)?;
    let imported = if imported_control.is_none() {
        prepare_krea_imported_sources(request, settings)?
    } else {
        None
    };
    let sdxl = prepare_sdxl_imported_sources(request, settings)?;
    let mage_finetuned = prepare_mage_finetuned_transformer(request, settings)?;
    let Some(kind) = resolve_image_route_with_imported_availability(
        request,
        settings,
        checkpoint_plan.is_available(),
        imported_control.is_some(),
        imported.is_some(),
        sdxl.is_some(),
        mage_finetuned.is_some(),
    ) else {
        // Zero lanes claimed the request. For a plan-backed entry that is never "no route": the
        // plan route declined only because the entry has SceneWorks-owned bytes some bespoke lane
        // could load, and here that lane did not materialize. Falling out with `Ok(None)` lands the
        // job in `generate_stub_stream`, which COMPLETES it with procedural stub output — so
        // re-raise the retained typed refusal instead (sc-20636 review).
        checkpoint_plan.into_unclaimed_refusal(request)?;
        return Ok(None);
    };
    Ok(Some(match kind {
        ImageRoute::CheckpointPlan => PreparedImageRoute::CheckpointPlan(Box::new(
            checkpoint_plan
                .into_sources()
                .expect("prepared checkpoint plan route lost its sources"),
        )),
        ImageRoute::KreaImportedControl => PreparedImageRoute::KreaImportedControl(Box::new(
            imported_control.expect("prepared imported-control route lost its sources"),
        )),
        ImageRoute::KreaImported => PreparedImageRoute::KreaImported(Box::new(
            imported.expect("prepared imported route lost its sources"),
        )),
        ImageRoute::SdxlImported => PreparedImageRoute::SdxlImported(Box::new(
            sdxl.expect("prepared SDXL imported route lost its sources"),
        )),
        ImageRoute::MageFinetuned => PreparedImageRoute::MageFinetuned(Box::new(
            mage_finetuned.expect("prepared Mage fine-tuned route lost its transformer"),
        )),
        route => PreparedImageRoute::Plain(route),
    }))
}

#[cfg(target_os = "macos")]
impl ImageRoute {
    /// True only for routes whose actual load path applies the request's LoRA/LoKr stack.
    pub(super) fn applies_request_loras(self) -> bool {
        matches!(
            self,
            ImageRoute::ZImageControl
                | ImageRoute::ZImageBaseControl
                | ImageRoute::SdxlControl
                | ImageRoute::QwenControl
                | ImageRoute::KolorsControl
                | ImageRoute::KreaControl
                | ImageRoute::Flux1DevControl
                | ImageRoute::Flux2DevControl
                | ImageRoute::Flux2Edit
                | ImageRoute::QwenEdit
                | ImageRoute::KreaEdit
                | ImageRoute::KreaTurboOnRaw
                | ImageRoute::KreaMultiPhase
                | ImageRoute::KreaImported
                | ImageRoute::KreaImportedControl
                | ImageRoute::SdxlImported
                | ImageRoute::InstantId
                | ImageRoute::SdxlAdvanced
                | ImageRoute::Mlx
        )
    }

    pub(super) fn image_count(self, request: &ImageRequest, settings: &Settings) -> u32 {
        match self {
            ImageRoute::ZImageControl
            | ImageRoute::ZImageBaseControl
            | ImageRoute::SdxlControl
            | ImageRoute::QwenControl
            | ImageRoute::KolorsControl
            | ImageRoute::KreaControl
            | ImageRoute::Flux1DevControl
            | ImageRoute::Flux2DevControl
            // The imported strict-pose lane renders one image per pose, like every control lane.
            | ImageRoute::KreaImportedControl => pose_entries(request).len() as u32,
            ImageRoute::Flux2Edit | ImageRoute::QwenEdit => grouped_edit_image_count(request),
            ImageRoute::InstantId => instantid_image_count(request, settings),
            ImageRoute::SensenovaEdit => match edit_grouping(request) {
                EditGrouping::Angles => CHARACTER_ANGLE_SET_ORDER.len() as u32,
                // SenseNova has no strict-pose (ControlNet) path; pose jobs are excluded
                // upstream, so any residual grouping preserves the requested image count.
                EditGrouping::Poses(_) | EditGrouping::Plain => request.count,
            },
            // PuLID-FLUX is one identity image per seed (no angle/pose grouping) — like the base
            // MLX + SDXL-advanced + Bernini paths, the effective count is the requested count. Krea
            // edit (epic 10871) is likewise plain per-image: `count` edits of the one source. The
            // pose reject arms (`PoseControlBaseMissing` / `PoseReject`) error before generation, so
            // their count is inert.
            ImageRoute::PulidFlux
            | ImageRoute::SdxlAdvanced
            | ImageRoute::Bernini
            | ImageRoute::KreaEdit
            // Turbo-on-Raw is plain per-image (like Krea edit + the base MLX path): `count` renders of
            // the base prompt, each its own seed. No angle/pose grouping.
            | ImageRoute::KreaTurboOnRaw
            // Multi-phase (S4) is likewise plain per-image: `count` renders, each its own seed, driven
            // through the phase plan. No angle/pose grouping.
            | ImageRoute::KreaMultiPhase
            // Imported single-file Krea 2 renders `count` images here for text-to-image or reference/edit
            // conditioning; strict-pose requests use the separate control route above.
            | ImageRoute::KreaImported
            | ImageRoute::SdxlImported
            // Plan-driven checkpoints (sc-20634) are plain per-image txt2img: `count` renders.
            | ImageRoute::CheckpointPlan
            // A fine-tuned Mage-Flow base (sc-15036) is plain per-image txt2img too: `count`
            // renders, each its own seed. No angle/pose grouping (the lane claims no conditioning).
            | ImageRoute::MageFinetuned
            | ImageRoute::FluxIpAdapterPoseReject
            | ImageRoute::PoseControlBaseMissing
            | ImageRoute::PoseReject => request.count,
            // The generic lane also serves a native prompt batch (sc-25681): `prompts × count`.
            ImageRoute::Mlx => request.image_count(),
        }
    }

    /// The label written by this route's per-asset stream. Bespoke routes that do not map directly
    /// to the request's registry row are named explicitly; the remaining MLX routes use the same
    /// descriptor label their stream resolves.
    pub(super) fn adapter_label(self, request: &ImageRequest) -> &'static str {
        match self {
            ImageRoute::SdxlControl => SDXL_CONTROL_ADAPTER_LABEL,
            ImageRoute::KreaControl => KREA_CONTROL_ENGINE_ID,
            ImageRoute::CheckpointPlan => CHECKPOINT_PLAN_ENGINE,
            ImageRoute::KreaImported | ImageRoute::KreaImportedControl => KREA_IMPORTED_ENGINE,
            ImageRoute::SdxlImported => SDXL_IMPORTED_ENGINE,
            ImageRoute::MageFinetuned => MAGE_FINETUNED_ENGINE,
            ImageRoute::InstantId => INSTANTID_ENGINE,
            ImageRoute::PulidFlux => PULID_ADAPTER_LABEL,
            ImageRoute::PoseControlBaseMissing | ImageRoute::PoseReject => STUB_ADAPTER,
            _ => mlx_model(&request.model)
                .map(|model| model.adapter_label())
                .unwrap_or(STUB_ADAPTER),
        }
    }
}

/// The candle (Windows/CUDA/Linux) image engine a `run_image_generate_job` request routes to — the
/// candle-lane sibling of [`ImageRoute`] (sc-8828, F-026). Each variant maps 1:1 to a bespoke candle
/// stream handler with the uniform `(api, settings, job, &plan, &project_path, backend, &mut asset_writes)`
/// signature; [`CandleImageRoute::PoseReject`] is the no-silent-T2I reject arm (sc-5968). Every arm is
/// gated on `settings.backend_candle_enabled`; when off (default) the resolver returns `None` so the job
/// falls through to the stub, exactly as before.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CandleImageRoute {
    /// Typed native MOSS editing, using the shared MOSS stream rather than generic conditioning.
    MossImageEdit,
    /// InstantID identity (sc-5491) — the off-Mac sibling of `ImageRoute::InstantId`. Checked first
    /// because `instantid_realvisxl` is not an `is_candle_engine` txt2img id.
    InstantId,
    /// Generic SDXL OpenPose ControlNet, shared with the macOS route through registry id `sdxl`.
    SdxlControl,
    /// SDXL img2img / inpaint / outpaint edit (sc-5487).
    SdxlEdit,
    /// FLUX.2-klein reference / img2img edit (sc-5487).
    Flux2Edit,
    /// Qwen-Image-Edit reference / dual-latent edit (sc-5487).
    QwenEdit,
    /// SenseNova-U1 instruction/character edit through the registered Reference/MultiReference path.
    SenseNovaEdit,
    /// Z-Image img2img / edit (sc-6595).
    ZimageEdit,
    /// Mage-Flow Base/RL/Turbo instruction edit. Uses the generic registry stream, but requires
    /// source-first ordered multi-reference conditioning rather than the plain T2I request shape.
    MageEdit,
    /// Qwen-Image 2.1 reference / local editing (sc-24110). `qwen_image_2_1` is ALSO a candle
    /// txt2img id, so a conditioned request must divert here first or the generic arm would render
    /// it as plain text-to-image and silently drop every reference. Uses the same generic registry
    /// stream once the ordered conditioning list is resolved.
    QwenImage21Edit,
    /// Ordered FLUX.3 reference conditioning through the ordinary registered generator.
    Flux3Reference,
    /// Krea 2 Kontext-style dual-conditioned image-edit — `krea_2_raw` + `edit_image` + a source, with
    /// the required `krea2_identity_edit` LoRA (epic 10871).
    KreaEdit,
    /// A Krea 2 **Raw** t2i job carrying the accelerator (turbo) LoRA (`role: accelerator`, sc-13882)
    /// → the single-phase turbo-on-Raw lane (epic 13879 S3, sc-13883, brought to candle by sc-13887):
    /// the Raw base + LoRA additive, sampled through the distilled Turbo regime (fixed mu 1.15 / ~8 steps
    /// / CFG-off) by routing to the `krea_2_turbo` candle engine id (which keys that regime, inference PR
    /// #204). The candle twin of the MLX `ImageRoute::KreaTurboOnRaw`; both dispatch to the SHARED,
    /// backend-neutral `generate_krea_turbo_on_raw_stream` (`krea_turbo_raw.rs`). Wins over the generic
    /// `CandleTxt2Img` arm for PLAIN t2i — `krea_2_raw` is a candle txt2img id, so it would otherwise run
    /// the 52-step true-CFG Raw regime and never accelerate.
    KreaTurboOnRaw,
    /// A Krea 2 **Raw** t2i job carrying an explicit `advanced.phases` list (epic 13879 S4, sc-13884,
    /// brought to candle by sc-13887) → the multi-phase denoise driver: ONE Raw trajectory over ONE
    /// global sigma schedule, each phase a contiguous step slice with its own guidance (per-phase CFG
    /// on/off) and its own active subset of the job's load-time LoRA stack (per-phase toggling). The
    /// candle twin of the MLX `ImageRoute::KreaMultiPhase`; both dispatch to the SHARED, backend-neutral
    /// `generate_krea_multiphase_stream` (`krea_multiphase.rs`), which passes the backend-agnostic
    /// `GenerationRequest::phases` the candle Krea engine honors (inference PR #204). Wins over
    /// `KreaTurboOnRaw` and the generic `CandleTxt2Img` arm — an explicit phase list is the finer-grained
    /// control. Loads the `krea_2_raw` engine (the driver keys on that descriptor id). Reference / edit /
    /// pose / PiD shapes are rejected loudly by the lane (multi-phase renders from pure noise).
    KreaMultiPhase,
    /// An imported/user single-file Krea 2 transformer: pair the in-place DiT with the resident
    /// shared-component base tier and load it through Candle's native single-file entrypoint. This is
    /// the off-Mac twin of [`ImageRoute::KreaImported`], required because imported IDs are not registry
    /// engine IDs and would otherwise fall through to the procedural stub.
    KreaImported,
    /// Strict-pose control over an imported Krea DiT, one image per pose.
    KreaImportedControl,
    /// Off-Mac twin of [`ImageRoute::CheckpointPlan`] (sc-20634): a plan-backed user checkpoint
    /// resolved through the plan store and rendered by the registry-bound candle provider.
    CheckpointPlan,
    /// A generated full-fine-tune Mage transformer paired with the installed Base TE/VAE.
    MageFinetuned,
    /// Off-Mac twin of [`ImageRoute::SdxlImported`], loaded by candle from the fused checkpoint plus
    /// the three caller-staged SDXL components.
    SdxlImported,
    /// SDXL IP-Adapter-Plus reference conditioning (sc-5488).
    SdxlIpAdapter,
    /// Kolors IP-Adapter-Plus reference conditioning (sc-5488).
    KolorsIpAdapter,
    /// FLUX XLabs IP-Adapter reference conditioning (sc-5872).
    FluxIpAdapter,
    /// FLUX.1's IP-Adapter and strict-control providers cannot compose one another's input.
    FluxIpAdapterPoseReject,
    /// PuLID-FLUX face identity (sc-5492).
    Pulid,
    /// Qwen-Image strict-pose ControlNet (sc-5489).
    QwenControl,
    /// Kolors strict-pose ControlNet (sc-5489).
    KolorsControl,
    /// Kolors' bespoke IP and ControlNet providers have no shared composition.  This explicit
    /// reject prevents routing precedence from silently dropping either identity reference, pose,
    /// or an unsupported IP+PiD request.
    KolorsCompositeReject,
    /// Kolors source-image img2img/edit through the registered generator.
    KolorsEdit,
    /// Z-Image strict-pose Fun-ControlNet (sc-5489).
    ZimageControl,
    /// FLUX.2-dev strict-pose Fun-Controlnet-Union (sc-7736).
    Flux2Control,
    /// FLUX.1-dev strict-control Shakker Union-Pro-2.0 (sc-8412).
    Flux1Control,
    /// Krea 2 pose-ControlNet — a trained control-branch overlay on the frozen Turbo base (sc-8464).
    KreaControl,
    /// A strict-pose job on a candle model with NO pose lane → reject loudly, never silent T2I (sc-5968).
    PoseReject,
    /// A strict-pose job on a WIRED candle pose family (one with a `…_control_available` lane) whose
    /// control base snapshot is NOT installed — the lane's local weight-gate failed, so the job reached
    /// the fall-through arm. Reject loudly ("control base snapshot not installed") rather than silently
    /// rendering plain txt2img and dropping the poses (sc-11171, F-008). Distinct from `PoseReject`,
    /// which is a family that has no candle pose lane at all.
    PoseControlBaseMissing,
    /// An in-place ComfyUI Z-Image base model (`external_base_*`) → `generate_candle_zimage_comfyui_stream`
    /// (epic 10451 Phase 2, sc-10668). Not an `is_candle_engine` id — routed off the forwarded row.
    ZimageComfyui,
    /// An in-place ComfyUI Qwen-Image base model (`external_base_*`) → `generate_candle_qwen_comfyui_stream`
    /// (epic 10451 Phase 2b, sc-10670). Not an `is_candle_engine` id — routed off the forwarded row.
    QwenImageComfyui,
    /// An in-place ComfyUI FLUX.2-dev fp8-mixed base model (`external_base_*`) →
    /// `generate_candle_flux2_comfyui_stream` (epic 10451 Phase 2e, sc-10680). Not an `is_candle_engine`
    /// id — routed off the forwarded row.
    Flux2Comfyui,
    /// Bernini still-image companion (`bernini_image`, engine id `bernini`, `frames:1`) → the dedicated
    /// `generate_candle_bernini_image_stream` (sc-10996, epic 6562). NOT an `is_candle_engine` txt2img id
    /// (the engine is `Modality::Video`, reached with `frames:1`), so — like the MLX `ImageRoute::Bernini`
    /// arm — it is routed on the model id BEFORE the generic txt2img arm; both t2i and i2i (`edit_image`)
    /// route here.
    Bernini,
    /// A plain candle txt2img engine id → `generate_candle_stream`.
    CandleTxt2Img,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl CandleImageRoute {
    /// True only for routes whose actual load path applies the request's LoRA/LoKr stack. This is
    /// also the provenance contract: adding a route here requires the same change to thread the
    /// resolved stack into its inference constructor.
    pub(super) fn applies_request_loras(self, request: &ImageRequest) -> bool {
        match self {
            CandleImageRoute::InstantId
            | CandleImageRoute::SdxlControl
            | CandleImageRoute::SdxlEdit
            | CandleImageRoute::SdxlIpAdapter
            | CandleImageRoute::KolorsIpAdapter
            | CandleImageRoute::Pulid
            | CandleImageRoute::QwenControl
            | CandleImageRoute::KolorsControl
            | CandleImageRoute::ZimageControl
            | CandleImageRoute::Flux2Control
            | CandleImageRoute::Flux1Control
            | CandleImageRoute::Flux2Edit
            | CandleImageRoute::ZimageComfyui
            | CandleImageRoute::Flux2Comfyui
            | CandleImageRoute::Bernini
            | CandleImageRoute::QwenEdit
            | CandleImageRoute::QwenImageComfyui
            | CandleImageRoute::ZimageEdit
            | CandleImageRoute::KolorsEdit
            | CandleImageRoute::MageEdit
            | CandleImageRoute::KreaEdit
            | CandleImageRoute::KreaTurboOnRaw
            | CandleImageRoute::KreaMultiPhase
            | CandleImageRoute::KreaImported
            | CandleImageRoute::KreaImportedControl
            | CandleImageRoute::SdxlImported
            | CandleImageRoute::KreaControl => true,
            // sc-24163: the Qwen-Image 2.1 reference-edit route is the SAME generic registry stream
            // as `CandleTxt2Img` (`generate_candle_stream`, which threads `resolve_adapters` into the
            // LoadSpec whenever the model supports adapters), so it consumes the stack exactly when
            // the plain arm does. Omitting it refused every LoRA/LoKr on a 2.1 edit (the CUDA
            // real-weight evidence run caught it).
            CandleImageRoute::CandleTxt2Img
            | CandleImageRoute::QwenImage21Edit
            | CandleImageRoute::Flux3Reference => {
                mlx_model(&request.model).is_some_and(|model| model.supports_adapters())
            }
            _ => false,
        }
    }

    /// The real image total this candle route produces, baked into the plan's `expectedCount` so the
    /// streamed gallery total matches what actually lands (sc-11171, F-009 — the candle sibling of the
    /// macOS `ImageRoute::image_count`). The strict-pose control lanes each render one image per pose
    /// (`pose_entries().len()`), InstantID renders its active angle/pose collection, every other lane
    /// renders the requested `count`.
    pub(super) fn image_count(self, request: &ImageRequest, settings: &Settings) -> u32 {
        match self {
            CandleImageRoute::QwenControl
            | CandleImageRoute::SdxlControl
            | CandleImageRoute::KolorsControl
            | CandleImageRoute::ZimageControl
            | CandleImageRoute::Flux2Control
            | CandleImageRoute::Flux1Control
            | CandleImageRoute::KreaControl => pose_entries(request).len() as u32,
            CandleImageRoute::KreaImportedControl => pose_entries(request).len() as u32,
            CandleImageRoute::InstantId => instantid_image_count(request, settings),
            CandleImageRoute::QwenEdit
                if qwen_edit_candle::qwen_edit_candle_pose_count(request)
                    .is_some_and(|count| count > 0) =>
            {
                qwen_edit_candle::qwen_edit_candle_pose_count(request).unwrap_or_default() as u32
            }
            CandleImageRoute::Flux2Edit
            | CandleImageRoute::QwenEdit
            | CandleImageRoute::SenseNovaEdit
                if candle_conditioned_angle_set(request) =>
            {
                sceneworks_core::angle_kps::BUILTIN_ANGLE_SET_ORDER.len() as u32
            }
            // The generic lane also serves a native prompt batch (sc-25681): `prompts × count`.
            CandleImageRoute::CandleTxt2Img | CandleImageRoute::MossImageEdit => {
                request.image_count()
            }
            // Every other lane (plain txt2img, the edit/reference/identity/comfyui/bernini lanes, and the
            // pose-reject arms — which error before generation) produces the requested count.
            _ => request.count,
        }
    }

    /// The exact label written by this route's per-asset stream. This is intentionally route-based:
    /// imported/external/bespoke IDs are not present in the generic candle catalog.
    pub(super) fn adapter_label(self, request: &ImageRequest) -> &'static str {
        match self {
            CandleImageRoute::InstantId => INSTANTID_ENGINE,
            CandleImageRoute::SdxlControl => SDXL_CONTROL_ADAPTER_LABEL,
            CandleImageRoute::SdxlEdit => sdxl_edit_candle::SDXL_EDIT_CANDLE_ENGINE,
            CandleImageRoute::Flux2Edit => flux2_edit_candle::FLUX2_EDIT_CANDLE_ENGINE,
            CandleImageRoute::QwenEdit => qwen_edit_candle::QWEN_EDIT_CANDLE_ENGINE,
            CandleImageRoute::SenseNovaEdit | CandleImageRoute::KolorsEdit => {
                candle_adapter_label(&request.model)
            }
            CandleImageRoute::ZimageEdit => candle_adapter_label(&request.model),
            CandleImageRoute::KreaEdit => krea_edit_candle::KREA_EDIT_CANDLE_ENGINE,
            CandleImageRoute::KreaTurboOnRaw | CandleImageRoute::KreaMultiPhase => {
                mlx_model(&request.model)
                    .map(|model| model.adapter_label())
                    .unwrap_or(STUB_ADAPTER)
            }
            CandleImageRoute::KreaImported => KREA_IMPORTED_ENGINE,
            CandleImageRoute::KreaImportedControl => KREA_IMPORTED_ENGINE,
            CandleImageRoute::CheckpointPlan => CHECKPOINT_PLAN_ENGINE,
            CandleImageRoute::MageFinetuned => MAGE_FINETUNED_ENGINE,
            CandleImageRoute::SdxlImported => SDXL_IMPORTED_ENGINE,
            CandleImageRoute::SdxlIpAdapter => sdxl_ipadapter::SDXL_IPADAPTER_ENGINE,
            CandleImageRoute::KolorsIpAdapter => kolors_ipadapter::KOLORS_IPADAPTER_ENGINE,
            CandleImageRoute::FluxIpAdapter => flux_ipadapter::FLUX_IPADAPTER_ENGINE,
            CandleImageRoute::Pulid => pulid_candle::PULID_CANDLE_ENGINE,
            CandleImageRoute::QwenControl => qwen_control::QWEN_CONTROL_ENGINE,
            CandleImageRoute::KolorsControl => kolors_control::KOLORS_CONTROL_ENGINE,
            CandleImageRoute::ZimageControl => zimage_control::ZIMAGE_CTRL_ENGINE,
            CandleImageRoute::Flux2Control => flux2_control_candle::FLUX2_CONTROL_CANDLE_ENGINE,
            CandleImageRoute::Flux1Control => flux1_control_candle::FLUX1_CONTROL_CANDLE_ENGINE,
            CandleImageRoute::KreaControl => krea_control_candle::KREA_CONTROL_ENGINE,
            CandleImageRoute::PoseReject
            | CandleImageRoute::PoseControlBaseMissing
            | CandleImageRoute::FluxIpAdapterPoseReject
            | CandleImageRoute::KolorsCompositeReject => STUB_ADAPTER,
            CandleImageRoute::ZimageComfyui => zimage_comfyui_candle::ZIMAGE_COMFYUI_CANDLE_ENGINE,
            CandleImageRoute::QwenImageComfyui => qwen_comfyui_candle::QWEN_COMFYUI_CANDLE_ENGINE,
            CandleImageRoute::Flux2Comfyui => flux2_comfyui_candle::FLUX2_COMFYUI_CANDLE_ENGINE,
            CandleImageRoute::Bernini => CANDLE_BERNINI_IMAGE_ADAPTER,
            CandleImageRoute::MossImageEdit
            | CandleImageRoute::MageEdit
            | CandleImageRoute::QwenImage21Edit
            | CandleImageRoute::Flux3Reference
            | CandleImageRoute::CandleTxt2Img => candle_adapter_label(&request.model),
        }
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) enum PreparedCandleImageRoute {
    Plain(CandleImageRoute),
    CheckpointPlan(Box<PreparedCheckpointPlanSources>),
    KreaImported(Box<PreparedKreaImportedSources>),
    SdxlImported(Box<PreparedSdxlImportedSources>),
    ZimageComfyui(Box<zimage_comfyui_candle::ComfyuiZImagePaths>),
    QwenImageComfyui(Box<qwen_comfyui_candle::ComfyuiQwenPaths>),
    Flux2Comfyui(Box<flux2_comfyui_candle::ComfyuiFlux2Paths>),
    MageFinetuned(Box<PreparedMageFinetunedTransformer>),
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl PreparedCandleImageRoute {
    pub(super) fn kind(&self) -> CandleImageRoute {
        match self {
            Self::Plain(route) => *route,
            Self::CheckpointPlan(_) => CandleImageRoute::CheckpointPlan,
            Self::KreaImported(_) => CandleImageRoute::KreaImported,
            Self::SdxlImported(_) => CandleImageRoute::SdxlImported,
            Self::ZimageComfyui(_) => CandleImageRoute::ZimageComfyui,
            Self::QwenImageComfyui(_) => CandleImageRoute::QwenImageComfyui,
            Self::Flux2Comfyui(_) => CandleImageRoute::Flux2Comfyui,
            Self::MageFinetuned(_) => CandleImageRoute::MageFinetuned,
        }
    }
}

/// Run the candle image dispatch predicate ladder ONCE and return the [`CandleImageRoute`] (or `None`
/// when candle is disabled / no candle engine matches → the job stubs). Mirrors the historical inline
/// `else if settings.backend_candle_enabled && <predicate>` ladder EXACTLY — same predicate order,
/// same `backend_candle_enabled` gating, same handler per family — so routing is byte-identical
/// (sc-8828). Pure decision: no I/O, no generation.
// One `…_available` flag per prepared bundle, so the count tracks the number of prepared candle
// lanes rather than any avoidable grouping.
#[allow(clippy::too_many_arguments)]
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_candle_image_route_with_prepared_availability(
    request: &ImageRequest,
    settings: &Settings,
    checkpoint_plan_available: bool,
    imported_available: bool,
    sdxl_imported_available: bool,
    zimage_comfyui_available: bool,
    qwen_comfyui_available: bool,
    flux2_comfyui_available: bool,
    mage_finetuned_available: bool,
) -> Option<CandleImageRoute> {
    if !settings.backend_candle_enabled {
        return None;
    }
    // Order matches the historical ladder: the edit / reference / identity / control lanes are all
    // checked BEFORE the generic `is_candle_engine` txt2img arm (they share candle txt2img model ids, so
    // without diverting first they'd be silently rendered as plain txt2img, dropping the source / poses).
    if sceneworks_core::moss_image::is_moss(&request.model) && request.mode == "edit_image" {
        Some(CandleImageRoute::MossImageEdit)
    } else if flux1_ipadapter_pose_combination(request) {
        Some(CandleImageRoute::FluxIpAdapterPoseReject)
    } else if instantid_available(request, settings) {
        Some(CandleImageRoute::InstantId)
    } else if sdxl_control_candidate(request) {
        // Same precedence as the core router: InstantID stays isolated, then generic SDXL control
        // claims every material pose carrier before edit/IP/generic routes can discard it.
        Some(CandleImageRoute::SdxlControl)
    } else if sdxl_edit_candle_available(request, settings) {
        Some(CandleImageRoute::SdxlEdit)
    } else if matches!(
        request.model.as_str(),
        "flux2_dev"
            | "qwen_image_edit"
            | "qwen_image_edit_2509"
            | "qwen_image_edit_2511"
            | "qwen_image_edit_2511_lightning"
    ) && candle_conditioned_pose_requires_reject(request)
    {
        // Defense in depth for direct worker calls: the scheduler rejects malformed pose carriers
        // and valid pose sets paired with unsupported conditioned modes. The worker must reject the
        // same shapes rather than letting typed parsing erase them or falling through to the stub.
        Some(CandleImageRoute::PoseReject)
    } else if flux2_control_candle_available(request, settings) {
        // Character Studio FLUX.2 pose payloads also carry a reference image. The pose/control lane
        // must win before reference edit so the skeleton is never silently dropped.
        Some(CandleImageRoute::Flux2Control)
    } else if flux2_edit_candle_available(request, settings) {
        Some(CandleImageRoute::Flux2Edit)
    } else if qwen_edit_candle_available(request, settings) {
        Some(CandleImageRoute::QwenEdit)
    } else if sensenova_edit_candle_available(request, settings) {
        Some(CandleImageRoute::SenseNovaEdit)
    } else if zimage_edit_candle_available(request, settings) {
        Some(CandleImageRoute::ZimageEdit)
    } else if is_mage_edit_model(&request.model)
        && request.mode == "edit_image"
        && request
            .source_asset_id
            .as_deref()
            .is_some_and(|id| !id.trim().is_empty())
    {
        Some(CandleImageRoute::MageEdit)
    } else if request
        .flux3_reference_request
        .as_ref()
        .is_some_and(|authored| {
            !authored
                .as_ref()
                .is_ok_and(|references| references.asset_ids.is_empty())
        })
    {
        // Also claim malformed stored reference jobs so the single trusted resolver refuses them
        // before load, rather than a missing/invalid slot falling through to a T2I or stub path.
        Some(CandleImageRoute::Flux3Reference)
    } else if is_qwen_image_2_1_edit(request) {
        // Qwen-Image 2.1 reference / local editing (sc-24110), named by the resolver for the same
        // reason Mage Edit is: the id is a candle txt2img id, so a conditioned request that fell
        // through would be rendered as plain T2I with every reference silently dropped. The core
        // router's `CandleImageLane::QwenImage21Edit` claims exactly these shapes.
        Some(CandleImageRoute::QwenImage21Edit)
    } else if request.model == "kolors"
        && ((non_empty(&request.reference_asset_id) && !pose_entries(request).is_empty())
            || (non_empty(&request.reference_asset_id)
                && advanced::flag(&request.advanced, "usePid")))
    {
        // Kolors IP-Adapter is its own provider (not a base overlay), and ControlNet owns a
        // separate pose provider. Neither implementation composes IP+Control or IP+PiD, so this
        // must fail before either precedence arm can discard an input.
        Some(CandleImageRoute::KolorsCompositeReject)
    } else if zimage_identity_candle_available(request, settings) {
        // The registered Z-Image Turbo generator owns both Reference conditioning and adapters.
        Some(CandleImageRoute::CandleTxt2Img)
    } else if sdxl_ipadapter_available(request, settings) {
        Some(CandleImageRoute::SdxlIpAdapter)
    } else if kolors_ipadapter_available(request, settings) {
        Some(CandleImageRoute::KolorsIpAdapter)
    } else if flux_ipadapter_available(request, settings) {
        Some(CandleImageRoute::FluxIpAdapter)
    } else if pulid_candle_available(request, settings) {
        Some(CandleImageRoute::Pulid)
    } else if qwen_control_available(request, settings) {
        Some(CandleImageRoute::QwenControl)
    } else if kolors_control_available(request, settings) {
        Some(CandleImageRoute::KolorsControl)
    } else if kolors_edit_candle_available(request, settings) {
        Some(CandleImageRoute::KolorsEdit)
    } else if zimage_control_available(request, settings) {
        Some(CandleImageRoute::ZimageControl)
    } else if flux1_control_candle_available(request, settings) {
        Some(CandleImageRoute::Flux1Control)
    } else if krea_control_candle_available(request, settings) {
        // Krea 2 pose-ControlNet (sc-8464): `krea_2_turbo` + `advanced.poses` is the bespoke candle
        // `Krea2Control` lane, diverted before the registry txt2img arm (which would render it as plain
        // txt2img and drop the poses). Mirrors `jobs_store::krea_control_candle_eligible`.
        Some(CandleImageRoute::KreaControl)
    } else if krea_multiphase_available(request, settings) {
        // Krea 2 Raw t2i + an explicit `advanced.phases` list (epic 13879 S4, sc-13884; candle sc-13887)
        // → the multi-phase denoise lane (`generate_krea_multiphase_stream`, SHARED with MLX). Placed
        // FIRST among the `krea_2_raw` t2i lanes so an explicit phase list takes precedence over the S3
        // whole-job turbo-on-Raw regime (`krea_turbo_on_raw_available` below) AND the generic Raw t2i
        // (`CandleTxt2Img`): multi-phase is the finer-grained control. Only claims `krea_2_raw` + a present
        // `advanced.phases`, so every non-phases job is unaffected; the lane itself rejects
        // edit/pose/reference/PiD shapes loudly (multi-phase renders from pure noise), so claiming a
        // conflicting shape here surfaces a clear error rather than diverting it silently. The candle twin
        // of the MLX `resolve_image_route` `KreaMultiPhase` arm.
        Some(CandleImageRoute::KreaMultiPhase)
    } else if krea_edit_candle_available(request, settings) {
        // Krea 2 Kontext-style edit (epic 10871): `krea_2_raw` + `edit_image` + a source is the bespoke
        // candle `KreaEdit` lane (`generate_candle_krea_edit_stream`), NOT the generic t2i stream. Diverted
        // BEFORE the generic `is_candle_engine` t2i arm below (which `krea_2_raw` now matches, sc-9994/epic
        // 9992) so an edit job runs the dual-conditioning Kontext render instead of being flattened to plain
        // txt2img. Mirrors `jobs_store::krea_edit_candle_eligible`.
        Some(CandleImageRoute::KreaEdit)
    } else if krea_turbo_on_raw_available(request, settings) {
        // Krea 2 Raw t2i + the accelerator (turbo) LoRA (sc-13882) → the single-phase turbo-on-Raw lane
        // (epic 13879 S3, sc-13883; candle sc-13887): the Raw base + LoRA additive, sampled as Turbo
        // (fixed mu 1.15 / ~8 steps / CFG-off) by routing to the `krea_2_turbo` candle engine
        // (`generate_krea_turbo_on_raw_stream`, SHARED with MLX). Wins over the generic `CandleTxt2Img`
        // arm for PLAIN t2i — a plain Raw t2i would otherwise run the 52-step true-CFG regime and ignore
        // the accelerator's intent. A `krea_2_raw` job that ALSO carries an img2img reference is EXCLUDED
        // by the gate and falls through to the generic `CandleTxt2Img` img2img path (reference honored via
        // `render_base_img2img`, accelerator LoRA still additive) — turbo-on-Raw img2img is out of scope
        // for this t2i story (sc-13883). The candle twin of the MLX `resolve_image_route` `KreaTurboOnRaw`
        // arm; placed AFTER the edit lane, BEFORE the generic txt2img arm.
        Some(CandleImageRoute::KreaTurboOnRaw)
    } else if checkpoint_plan_available {
        // A plan-backed user checkpoint (epic 20398, sc-20634): claimed before every bespoke imported
        // lane; the plan store is the route's sole source of truth.
        Some(CandleImageRoute::CheckpointPlan)
    } else if krea_imported_control_available(request, settings) {
        // An imported single-file Krea 2 checkpoint + a strict-pose set: the pose control branch rides
        // the file-loaded imported DiT. Checked BEFORE the plain imported arm so a pose set renders one
        // pose-locked image per pose instead of falling into per-image t2i (dropping the poses).
        Some(CandleImageRoute::KreaImportedControl)
    } else if imported_available {
        // Imported/user Krea 2 single-file generation: external IDs are absent from
        // `is_candle_engine`, so this bespoke text-to-image/img2img/edit route must claim them before
        // the generic/external fall-through.
        Some(CandleImageRoute::KreaImported)
    } else if mage_finetuned_available {
        // A fine-tuned Mage-Flow base (sc-15036): its id is in no `MODEL_TABLE` / `is_candle_engine`
        // arm, so this route is what dispatches it to real candle generation at all.
        Some(CandleImageRoute::MageFinetuned)
    } else if sdxl_imported_available {
        Some(CandleImageRoute::SdxlImported)
    } else if zimage_comfyui_available {
        // In-place ComfyUI Z-Image base (sc-10668): an `external_base_*` id, so it matches no
        // `is_candle_engine` arm below — route it here off the forwarded `modelManifestEntry`.
        Some(CandleImageRoute::ZimageComfyui)
    } else if qwen_comfyui_available {
        // In-place ComfyUI Qwen-Image base (sc-10670): sibling of the Z-Image comfyui lane — an
        // `external_base_*` id routed off the forwarded row (family=="qwen-image", usable).
        Some(CandleImageRoute::QwenImageComfyui)
    } else if flux2_comfyui_available {
        // In-place ComfyUI FLUX.2-dev base (sc-10680): sibling of the Qwen-Image comfyui lane — an
        // `external_base_*` id routed off the forwarded row (family=="flux2", usable).
        Some(CandleImageRoute::Flux2Comfyui)
    } else if bernini_image_candle_available(request) {
        // Bernini still-image companion (sc-10996, epic 6562): t2i / i2i on the `bernini_image` id,
        // routed to the same `engine_id:"bernini"` planner+renderer with `frames:1`. Must win over the
        // generic `is_candle_engine` txt2img arm below — `bernini_image` is NOT an `is_candle_engine` id
        // (its engine is `Modality::Video`, reached with `frames:1`), so it would otherwise fall through
        // to `None` and stub. Routed on the model id alone (like the sdxl txt2img arm) — a missing
        // `SceneWorks/bernini` snapshot fails loud at load, never silently stubs (the MLX
        // `ImageRoute::Bernini` weight-gates instead only because it must fall through to `mlx_available`).
        Some(CandleImageRoute::Bernini)
    } else if is_candle_engine(&request.model)
        && request.mode != "edit_image"
        && !pose_entries(request).is_empty()
    {
        // A strict-pose candle job that reached here was NOT claimed by any `…_control_available` lane
        // above, so it must be REJECTED, not silently rendered as plain txt2img (poses dropped). Two
        // sub-cases, distinguished by whether the family has a wired candle pose lane at all:
        //  - WIRED pose family (derived wired Candle pose family): the lane exists but its control base
        //    snapshot is absent (the lane's local weight-gate failed) → `PoseControlBaseMissing`
        //    ("control base snapshot not installed"). Previously this family was excluded from the reject
        //    entirely and fell through to `CandleTxt2Img`, silently dropping the poses (sc-11171, F-008).
        //  - No candle pose lane → the sc-5968 no-silent-T2I `PoseReject`.
        // Checked BEFORE the txt2img arm below.
        if wired_pose_family(&request.model, "candle") {
            Some(CandleImageRoute::PoseControlBaseMissing)
        } else {
            Some(CandleImageRoute::PoseReject)
        }
    } else if is_candle_engine(&request.model) && request.mode != "edit_image" {
        Some(CandleImageRoute::CandleTxt2Img)
    } else {
        None
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn candle_conditioned_pose_requires_reject(request: &ImageRequest) -> bool {
    let pose_count = match request.advanced.get("poses") {
        None | Some(Value::Null) => return false,
        Some(Value::Array(poses))
            if poses.len() <= sceneworks_core::image_request::MAX_JOB_POSES
                && poses.iter().all(Value::is_object) =>
        {
            poses.len()
        }
        Some(_) => return true,
    };
    if pose_count == 0 {
        return false;
    }
    match request.model.as_str() {
        // FLUX.2 pose ControlNet serves every non-edit mode; instruction edit plus a pose has no
        // composition contract and must reject rather than fall through after both specialists say no.
        "flux2_dev" => request.mode == "edit_image",
        // Qwen Edit's exact pose recipe is Character Studio with one identity reference; every other
        // valid pose-bearing mode/reference shape is unsupported and must likewise reject explicitly.
        "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning" => !qwen_edit_candle::qwen_edit_candle_mode(request),
        _ => false,
    }
}

/// Test-facing pure route probe. Production uses [`prepare_candle_image_route`] so imported File
/// tokens remain owned across admission and dispatch.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle", test))]
pub(super) fn resolve_candle_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> Option<CandleImageRoute> {
    resolve_candle_image_route_with_prepared_availability(
        request,
        settings,
        // See the macOS twin: a typed refusal must surface, not read as "route unavailable".
        checkpoint_plan_available_or_panic(request, settings),
        krea_imported_available(request, settings),
        sdxl_imported_available(request, settings),
        zimage_comfyui_candle::zimage_comfyui_available(request, settings),
        qwen_comfyui_candle::qwen_comfyui_available(request, settings),
        flux2_comfyui_candle::flux2_comfyui_available(request, settings),
        mage_finetuned_available(request, settings),
    )
}

/// Resolve a production candle route while retaining each payload-selected File token through every
/// later await. Only the route that wins the existing predicate order consumes its prepared bundle.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn prepare_candle_image_route(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PreparedCandleImageRoute>> {
    if !settings.backend_candle_enabled {
        // This early-out returns BEFORE `prepare_checkpoint_plan_sources` runs, so the
        // `into_unclaimed_refusal` re-raise below never fires and a plan-backed entry falls all the
        // way to `generate_stub_stream` — the job COMPLETES with procedural output under the user's
        // imported checkpoint id (sc-20651). Refuse in the same typed shape instead. The macOS twin
        // `prepare_image_route` has no such early-out, so it needs no counterpart.
        if let Some(checkpoint_id) = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(
            &request.model_manifest_entry,
        ) {
            return Err(WorkerError::InvalidPayload(format!(
                "[checkpoint-plan:no-adapter-binding] checkpoint {checkpoint_id:?}: the candle \
                 backend is disabled on this worker, and a plan-backed entry never renders \
                 procedural stub output"
            )));
        }
        return Ok(None);
    }
    // The plan-driven route is offered the request first (sc-20634). Per-request claim, exactly as
    // in the macOS twin: it takes the shapes it serves, the family's bespoke lane keeps the rest,
    // and the decline's retained refusal is re-raised below when no lane claims it (sc-20636).
    let checkpoint_plan = prepare_checkpoint_plan_sources(request, settings)?;
    // A pose-bearing imported checkpoint is claimed by the `KreaImportedControl` arm, which resolves
    // its own sources; skip pinning a second File token for the plain imported bundle in that case
    // (mirrors the macOS `prepare_image_route` guard).
    let imported = if krea_imported_control_available(request, settings) {
        None
    } else {
        prepare_krea_imported_sources(request, settings)?
    };
    let sdxl = prepare_sdxl_imported_sources(request, settings)?;
    let zimage = zimage_comfyui_candle::prepare_zimage_comfyui_sources(request, settings)?;
    let qwen = qwen_comfyui_candle::prepare_qwen_comfyui_sources(request, settings)?;
    let flux2 = flux2_comfyui_candle::prepare_flux2_comfyui_sources(request, settings)?;
    let mage_finetuned = prepare_mage_finetuned_transformer(request, settings)?;
    let Some(kind) = resolve_candle_image_route_with_prepared_availability(
        request,
        settings,
        checkpoint_plan.is_available(),
        imported.is_some(),
        sdxl.is_some(),
        zimage.is_some(),
        qwen.is_some(),
        flux2.is_some(),
        mage_finetuned.is_some(),
    ) else {
        // See the macOS twin: zero lanes claimed a plan-backed entry means the bespoke lane the
        // plan route deferred to does not exist for this shape. Raise the retained typed refusal
        // rather than dropping through to `generate_stub_stream`'s procedural output (sc-20636).
        checkpoint_plan.into_unclaimed_refusal(request)?;
        return Ok(None);
    };
    Ok(Some(match kind {
        CandleImageRoute::CheckpointPlan => PreparedCandleImageRoute::CheckpointPlan(Box::new(
            checkpoint_plan
                .into_sources()
                .expect("prepared checkpoint plan route lost its sources"),
        )),
        CandleImageRoute::KreaImported => PreparedCandleImageRoute::KreaImported(Box::new(
            imported.expect("prepared imported route lost its sources"),
        )),
        CandleImageRoute::SdxlImported => PreparedCandleImageRoute::SdxlImported(Box::new(
            sdxl.expect("prepared SDXL imported route lost its sources"),
        )),
        CandleImageRoute::ZimageComfyui => PreparedCandleImageRoute::ZimageComfyui(Box::new(
            zimage.expect("prepared Z-Image route lost its sources"),
        )),
        CandleImageRoute::QwenImageComfyui => PreparedCandleImageRoute::QwenImageComfyui(Box::new(
            qwen.expect("prepared Qwen route lost its sources"),
        )),
        CandleImageRoute::Flux2Comfyui => PreparedCandleImageRoute::Flux2Comfyui(Box::new(
            flux2.expect("prepared FLUX.2 route lost its sources"),
        )),
        CandleImageRoute::MageFinetuned => PreparedCandleImageRoute::MageFinetuned(Box::new(
            mage_finetuned.expect("prepared Mage fine-tuned route lost its transformer"),
        )),
        route => PreparedCandleImageRoute::Plain(route),
    }))
}

/// Couples a route-owned File source bundle to the immutable generation plan without widening the
/// uniform stream-handler signatures. The bundle is moved into the handler exactly once.
#[cfg(native_lane)]
pub(super) struct PreparedFileDispatch<'a, Sources> {
    pub(super) plan: &'a ImagePlan,
    pub(super) sources: Sources,
}

/// Whether `model` is served by the Candle backend's generic built-in image lane.
///
/// The scheduler's generated catalog is the source of truth. `bernini_image` is the sole built-in
/// exception: the scheduler routes it to Candle, but [`prepare_candle_image_route`] sends it through
/// the dedicated still-image Bernini lane before reaching this gate. Dynamic `external_base_*` ids
/// are likewise claimed by manifest-driven bespoke routes and never appear in the built-in catalog.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn is_candle_engine(model: &str) -> bool {
    model != "bernini_image"
        && sceneworks_core::jobs_store::candle_routed_image_models().contains(&model)
}

/// The per-asset `adapter` id recorded for a candle image engine (`candle_<family>`), the candle
/// sibling of the `MODEL_TABLE` `mlx_<family>` labels. Used by both the generic per-asset stream and
/// [`CandleImageRoute::adapter_label`] so the sidecar and generation-set result agree.
/// (sc-5099 extends this same labeling to the video + caption engines.)
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_adapter_label(model: &str) -> &'static str {
    match model {
        // Base z_image (sc-8679) shares the candle z-image family label with Turbo.
        "z_image_turbo" | "z_image" | "z_image_edit" => "candle_z_image",
        "flux_schnell" | "flux_dev" => "candle_flux",
        // The base klein + its `_kv` / `_true_v2` weight variants (sc-7459) + dev all run candle FLUX.2.
        "flux2_klein_9b" | "flux2_klein_9b_kv" | "flux2_klein_9b_true_v2" | "flux2_dev" => {
            "candle_flux2"
        }
        "qwen_image" => "candle_qwen",
        // Qwen-Image 2.1 (sc-24109) is its OWN engine, not a newer build of the row above: a
        // different snapshot on a different latent space with a different text tower. It must
        // stamp its own adapter or a replayed asset cannot say which weights produced it.
        //
        // These labels are a SEPARATE namespace from the recipe's adapter ids: this one is the
        // per-asset `adapter` the candle stream writes (`candle_<family>`, the off-Mac sibling of
        // the MODEL_TABLE `mlx_<family>` labels), while the recipe stamp is
        // `sceneworks_core::contracts::RecipeAdapter::QwenImage21` ("qwen_image_2_1"). Both exist
        // for the same reason, and neither is derived from the other.
        "qwen_image_2_1" => "candle_qwen_2_1",
        // Turbo (sc-25907): the distilled weights stamp their own label for the same reason.
        "qwen_image_2_1_turbo" => "candle_qwen_2_1_turbo",
        "flux3_klein" | "flux3_klein_base" => "candle_flux3",
        "moss_image_base" | "moss_image_sft" => moss_adapter_label(model, "candle"),
        // Supra2-IMG (sc-24214): the off-Mac sibling of the `mlx_supra2` MODEL_TABLE label.
        "supra2_img" => "candle_supra2",
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => "candle_chroma",
        "lens" | "lens_turbo" => "candle_lens",
        "kolors" => "candle_kolors",
        "sensenova_u1_8b"
        | "sensenova_u1_8b_infographic_v2"
        | "sensenova_u1_8b_infographic_v3"
        | "sensenova_u1_8b_fast"
        | "sensenova_u1_8b_infographic_v2_fast"
        | "sensenova_u1_8b_infographic_v3_fast" => "candle_sensenova",
        "ideogram_4" | "ideogram_4_turbo" => "candle_ideogram",
        "boogu_image" | "boogu_image_turbo" | "boogu_image_edit" => "candle_boogu",
        "mage_flow_base"
        | "mage_flow"
        | "mage_flow_turbo"
        | "mage_flow_edit_base"
        | "mage_flow_edit"
        | "mage_flow_edit_turbo" => "candle_mage",
        "krea_2_turbo" | "krea_2_raw" => "candle_krea",
        // Stable Diffusion 3.5 (sc-7880): Large / Large Turbo / Medium share the candle SD3.5 engine.
        "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium" => "candle_sd3",
        // SANA 1600M + SANA-Sprint (sc-11780 / sc-11781): both share the candle SANA engine (the off-Mac
        // sibling of the `mlx_sana` label).
        "sana_1600m" | "sana_sprint_1600m" => "candle_sana",
        // Anima 2B (sc-10676): base / aesthetic / turbo share the candle Anima engine (the off-Mac
        // sibling of the `mlx_anima` label).
        "anima_base" | "anima_aesthetic" | "anima_turbo" => "candle_anima",
        // Iris-3B (sc-25680): the off-Mac sibling of the `mlx_iris` label.
        "iris_3b" => "candle_iris",
        // Bernini's production stream stamps this via its own route; the request-only fallback
        // (`adapter_id`) reaches it through here.
        "bernini_image" => CANDLE_BERNINI_IMAGE_ADAPTER,
        // sdxl / realvisxl share the candle "sdxl" engine.
        _ => CANDLE_ADAPTER,
    }
}

#[cfg(native_lane)]
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_disabled_plan_backed_refusal_tests.rs"]
mod candle_disabled_plan_backed_refusal_tests;

#[cfg(native_lane)]
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_label_tests.rs"]
mod candle_label_tests;
