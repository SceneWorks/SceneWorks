#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::ltx::resolve_clip_media_path;
#[allow(unused_imports)]
use super::prelude::*;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use super::wan::ClipFramePosition;
#[cfg(target_os = "macos")]
use super::wan::{
    generate_video, resolve_wan_model_dir, resolve_wan_quant, resolve_wan_vace_adapters,
    VideoGenInput,
};
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::image_jobs::load_reference_image;

// ---------------------------------------------------------------------------
// Real MLX Wan-VACE replace_person generation (macOS, via mlx-gen-wan, sc-3521):
// route the `replace_person` mode / `PersonReplace` job to the native `wan_vace`
// provider — the equivalent of the torch `DiffusersVideoAdapter` `WanVACEPipeline`
// path. The worker builds the masked-control inputs (source clip frames + the
// onnx-track person mask + character refs) and the engine does the
// masking/neutralization + denoise. Person detect/track/segment stays upstream.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX Wan-VACE replace_person asset.
#[cfg(target_os = "macos")]
pub(super) const WAN_VACE_ADAPTER: &str = "mlx_wan_vace";

/// Per-asset adapter label for the native dual-expert Wan2.2 VACE-Fun replace_person backend
/// (`wan2_2_vace_fun_14b`, sc-3459) — distinct from single-expert `mlx_wan_vace` so the asset
/// honestly records which VACE engine produced the replacement.
#[cfg(target_os = "macos")]
pub(super) const WAN_VACE_FUN_ADAPTER: &str = "mlx_wan_vace_fun";

/// Letterbox pad colour for extracted source-clip frames — matches the Python `fit_frame`
/// background (`0x12110f` = RGB 18,17,15) so the box masks (rasterized from the same
/// normalized boxes at W×H) stay aligned with the control frames through the engine's
/// identity-resize preprocess.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const FRAME_PAD_COLOR: &str = "0x12110f";

/// Raw-settings recorded on a real Wan-VACE asset (`advanced` knobs + the real-inference
/// markers; the engine id is `wan_vace`, not the user-picked replace-capable model).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn wan_vace_raw_settings(request: &VideoRequest, model: &str) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(model.to_owned()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "replacementMode".to_owned(),
        Value::String(request.replacement_mode.clone()),
    );
    Value::Object(raw)
}

/// SceneWorks `replacementMode` string → engine [`ReplacementMode`] (default FaceOnly).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn replacement_mode_from(value: &str) -> ReplacementMode {
    match value {
        "full_person_keep_outfit" => ReplacementMode::FullPersonKeepOutfit,
        "full_person_replace_outfit" => ReplacementMode::FullPersonReplaceOutfit,
        _ => ReplacementMode::FaceOnly,
    }
}

/// Whether `dir` is a load-ready assembled Wan-VACE snapshot — the diffusers VACE
/// `transformer/` plus the shared base-Wan UMT5/VAE/tokenizer that `crate::inference_runtime::load("wan_vace")`
/// reads (sc-3467 `assemble_wan_vace_snapshot` layout).
#[cfg(target_os = "macos")]
pub(super) fn wan_vace_dir_is_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file()
}

/// Resolve (assembling on first use) the converted Wan-VACE snapshot dir. Env override
/// (`SCENEWORKS_MLX_WAN_VACE_DIR`) → the app-managed `<data>/models/mlx/wan_vace` → assemble
/// it from the diffusers VACE transformer (HF `Wan-AI/Wan2.1-VACE-1.3B-diffusers`,
/// `transformer/`) + a converted base-Wan 14B snapshot's shared UMT5/z16-VAE/tokenizer
/// (sc-3467 `assemble_wan_vace_snapshot` — packaging, not conversion). Errors clearly when a
/// component is missing rather than degrading to the stub.
#[cfg(target_os = "macos")]
pub(super) fn resolve_wan_vace_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if wan_vace_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let out_dir = settings
        .data_dir
        .join("models")
        .join("mlx")
        .join("wan_vace");
    if wan_vace_dir_is_complete(&out_dir) {
        return Ok(out_dir);
    }
    // Assemble on first use: the VACE transformer is diffusers-layout (no conversion); the
    // shared T5/VAE/tokenizer come from a converted base-Wan 14B snapshot (z16 VAE, shared
    // with VACE since both are Wan2.1-based).
    let vace_repo = "Wan-AI/Wan2.1-VACE-1.3B-diffusers";
    let transformer_dir = huggingface_snapshot_dir(&settings.data_dir, vace_repo)
        .map(|snapshot| snapshot.join("transformer"))
        .filter(|dir| dir.join("config.json").is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "replace_person: the Wan-VACE transformer ({vace_repo}) is not downloaded — \
                 fetch it via the model manager."
            ))
        })?;
    let base_wan = ["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]
        .into_iter()
        .find_map(|model| resolve_wan_model_dir(settings, model, model).ok())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "replace_person: Wan-VACE needs a converted base-Wan 14B snapshot (its shared \
                 UMT5 text encoder + z16 VAE + tokenizer). Convert/download wan_2_2_t2v_14b or \
                 wan_2_2_i2v_14b first."
                    .to_owned(),
            )
        })?;
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    runtime_macos::providers::wan::convert::assemble_wan_vace_snapshot(
        &out_dir,
        &transformer_dir,
        &base_wan,
        true,
    )
    .map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "replace_person: failed to assemble the Wan-VACE snapshot: {error}"
        ))
    })?;
    Ok(out_dir)
}

/// Whether `dir` is a load-ready assembled Wan2.2 VACE-Fun snapshot — BOTH diffusers VACE-Fun
/// expert dirs (`transformer/` high-noise + `transformer_2/` low-noise) plus the shared base-Wan
/// UMT5/VAE/tokenizer that `crate::inference_runtime::load("wan2_2_vace_fun_14b")` reads (sc-6604
/// `assemble_wan_vace_fun_snapshot` layout).
#[cfg(target_os = "macos")]
fn wan_vace_fun_dir_is_complete(dir: &Path) -> bool {
    dir.join("transformer").join("config.json").is_file()
        && dir.join("transformer_2").join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file()
}

/// Resolve (assembling on first use) the dual-expert Wan2.2 VACE-Fun snapshot dir (sc-3459). Env
/// override (`SCENEWORKS_MLX_WAN_VACE_FUN_DIR`) → the app-managed `<data>/models/mlx/wan_2_2_vace_fun`
/// → assemble it from the diffusers VACE-Fun experts (HF `linoyts/Wan2.2-VACE-Fun-14B-diffusers`,
/// `transformer/` + `transformer_2/`) + a converted base-Wan 14B snapshot's shared UMT5/z16-VAE/
/// tokenizer (sc-6604 `assemble_wan_vace_fun_snapshot` — packaging, not conversion). Errors clearly
/// when a component is missing rather than degrading to the Wan2.1 VACE backend or the stub.
#[cfg(target_os = "macos")]
fn resolve_wan_vace_fun_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_WAN_VACE_FUN_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if wan_vace_fun_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    let out_dir = settings
        .data_dir
        .join("models")
        .join("mlx")
        .join("wan_2_2_vace_fun");
    if wan_vace_fun_dir_is_complete(&out_dir) {
        return Ok(out_dir);
    }
    // Assemble on first use: the two VACE-Fun experts are diffusers-layout (read directly, no
    // conversion); the shared T5/VAE/tokenizer come from a converted base-Wan 14B snapshot (z16 VAE,
    // shared with VACE-Fun since it is Wan2.2-A14B-based with the Wan2.1 z16 VAE).
    let vace_repo = "linoyts/Wan2.2-VACE-Fun-14B-diffusers";
    let snapshot = huggingface_snapshot_dir(&settings.data_dir, vace_repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "wan_2_2_vace_fun_14b: the VACE-Fun transformers ({vace_repo}) are not downloaded — \
             fetch the model via the model manager."
        ))
    })?;
    let high = snapshot.join("transformer");
    let low = snapshot.join("transformer_2");
    if !high.join("config.json").is_file() || !low.join("config.json").is_file() {
        return Err(WorkerError::InvalidPayload(format!(
            "wan_2_2_vace_fun_14b: the {vace_repo} download is incomplete (missing transformer/ or \
             transformer_2/) — re-fetch it via the model manager."
        )));
    }
    let base_wan = ["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]
        .into_iter()
        .find_map(|model| resolve_wan_model_dir(settings, model, model).ok())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "wan_2_2_vace_fun_14b: VACE-Fun needs a converted base-Wan 14B snapshot (its shared \
                 UMT5 text encoder + z16 VAE + tokenizer). Convert/download wan_2_2_t2v_14b or \
                 wan_2_2_i2v_14b first."
                    .to_owned(),
            )
        })?;
    // CARVE-OUT(epic 3720): backend-specific weight packager; not a registry contract.
    runtime_macos::providers::wan::convert::assemble_wan_vace_fun_snapshot(
        &out_dir, &high, &low, &base_wan, true,
    )
    .map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "wan_2_2_vace_fun_14b: failed to assemble the VACE-Fun snapshot: {error}"
        ))
    })?;
    Ok(out_dir)
}

/// Decode the source clip into exactly `count` RGB frames at `width × height` (letterboxed,
/// `FRAME_PAD_COLOR`), evenly resampled across the clip — the new shared frame-extraction
/// helper (Python `load_source_video_frames`; also the seam extend/bridge will reuse). The
/// frames are the (un-neutralized) Wan-VACE control video; the engine masks them.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) async fn load_source_video_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    count: usize,
) -> WorkerResult<Vec<Image>> {
    let asset_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a source clip (sourceClipAssetId).".to_owned(),
        )
    })?;
    let media_path =
        resolve_clip_media_path(settings, &request.project_id, asset_id, project_path)?;

    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id would
    // otherwise escape `temp_dir()`. Mirrors `sw-person-track-{safe_download_dir(job.id)}` in media_jobs.
    let work_dir =
        std::env::temp_dir().join(format!("sw-replace-frames-{}", safe_download_dir(&job.id)));
    tokio::fs::create_dir_all(&work_dir).await?;
    let pattern = work_dir.join("src_%05d.png");
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24",
        width = request.width,
        height = request.height,
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let extract = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    let frames = match extract {
        Ok(()) => select_extracted_frames(work_dir.clone(), count).await,
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    frames
}

/// Collect the extracted PNG frames in `work_dir`, resample them to `count` evenly-spaced
/// indices (Python `evenly_spaced_indices` — the same arithmetic as the mask resample), and
/// decode the selected frames to engine [`Image`]s. Blocking IO/decoding runs off the runtime.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn select_extracted_frames(work_dir: PathBuf, count: usize) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "source clip produced no decodable frames".to_owned(),
            ));
        }
        let indices = crate::person_replace::resample_indices(paths.len(), count);
        indices
            .into_iter()
            .map(|index| decode_png_image(&paths[index]))
            .collect()
    })
    .await
    .map_err(|error| task_join_error("frame decode task", error))?
}

/// One immutable character-reference receipt plus its decoded RGB identity image. Keeping the
/// asset id beside the pixels lets provider-specific paths record exactly which approved identities
/// were physically conditioned, rather than reporting only an untraceable count.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) struct CharacterReferenceReceipt {
    pub(super) asset_id: String,
    pub(super) image: Image,
}

/// The approved character reference receipts (1–4) for replacement: the selected look's
/// `approvedReferenceIds`, else the character's approved `references`. Every selected identity is
/// mandatory: dropping an unreadable item or silently truncating the ordered list changes the model
/// input while retaining a misleading successful replacement receipt.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_character_reference_receipts(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<CharacterReferenceReceipt>> {
    let character_id = request.character_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload("replace_person requires a character (characterId).".to_owned())
    })?;
    let character = CharacterStore::new(&settings.data_dir, project_path.to_path_buf())
        .get_character(&request.project_id, character_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("character {character_id}: {error}"))
        })?;
    let mut ids: Vec<String> = Vec::new();
    if let Some(look_id) = request.character_look_id.as_deref() {
        if let Some(looks) = character.get("looks").and_then(Value::as_array) {
            for look in looks {
                if look.get("id").and_then(Value::as_str) == Some(look_id) {
                    if let Some(approved) =
                        look.get("approvedReferenceIds").and_then(Value::as_array)
                    {
                        ids.extend(approved.iter().filter_map(Value::as_str).map(str::to_owned));
                    }
                }
            }
        }
    }
    if ids.is_empty() {
        if let Some(references) = character.get("references").and_then(Value::as_array) {
            for reference in references {
                if reference
                    .get("approved")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Some(asset_id) = reference.get("assetId").and_then(Value::as_str) {
                        ids.push(asset_id.to_owned());
                    }
                }
            }
        }
    }
    let attempted: Vec<String> = ids.into_iter().filter(|id| !id.is_empty()).collect();
    if !(1..=4).contains(&attempted.len()) {
        return Err(WorkerError::InvalidPayload(format!(
            "replace_person requires 1–4 approved ordered character reference images (got {})",
            attempted.len()
        )));
    }
    let mut references = Vec::with_capacity(attempted.len());
    for asset_id in attempted {
        let image = load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &asset_id,
            project_path,
        )
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "replace_person: approved character reference {asset_id} is unreadable: {error}"
            ))
        })?;
        references.push(CharacterReferenceReceipt { asset_id, image });
    }
    Ok(references)
}

/// Legacy image-only resolver for existing Wan/SCAIL paths. Keep its historical best-effort
/// semantics isolated here; LTX replacement must use [`resolve_character_reference_receipts`],
/// whose immutable carrier cannot truthfully omit an approved identity.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn resolve_character_references(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Image>> {
    let character_id = request.character_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload("replace_person requires a character (characterId).".to_owned())
    })?;
    let character = CharacterStore::new(&settings.data_dir, project_path.to_path_buf())
        .get_character(&request.project_id, character_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("character {character_id}: {error}"))
        })?;
    let mut ids: Vec<String> = Vec::new();
    if let Some(look_id) = request.character_look_id.as_deref() {
        if let Some(looks) = character.get("looks").and_then(Value::as_array) {
            for look in looks {
                if look.get("id").and_then(Value::as_str) == Some(look_id) {
                    if let Some(approved) =
                        look.get("approvedReferenceIds").and_then(Value::as_array)
                    {
                        ids.extend(approved.iter().filter_map(Value::as_str).map(str::to_owned));
                    }
                }
            }
        }
    }
    if ids.is_empty() {
        if let Some(references) = character.get("references").and_then(Value::as_array) {
            for reference in references {
                if reference
                    .get("approved")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    if let Some(asset_id) = reference.get("assetId").and_then(Value::as_str) {
                        ids.push(asset_id.to_owned());
                    }
                }
            }
        }
    }
    let attempted: Vec<String> = ids
        .into_iter()
        .filter(|id| !id.is_empty())
        .take(4)
        .collect();
    let approved_count = attempted.len();
    let mut images = Vec::new();
    for asset_id in attempted {
        match load_reference_image(
            &settings.data_dir,
            &request.project_id,
            &asset_id,
            project_path,
        ) {
            Ok(image) => images.push(image),
            Err(error) => {
                tracing::warn!(
                    event = "character_reference_load_failed",
                    characterId = %character_id,
                    assetId = %asset_id,
                    error = %error,
                    "skipping an unreadable approved character reference — identity conditioning \
                     will use fewer references than approved"
                );
            }
        }
    }
    if images.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Replace Person requires at least one approved character reference image.".to_owned(),
        ));
    }
    if images.len() < approved_count {
        tracing::warn!(
            event = "character_references_partially_loaded",
            characterId = %character_id,
            loaded = images.len(),
            approved = approved_count,
            "loaded fewer character references than were approved — {} of {} approved references \
             could not be read; identity conditioning is reduced",
            approved_count - images.len(),
            approved_count
        );
    }
    Ok(images)
}

/// Convert an `image::RgbImage` (the rasterized mask) to an engine [`Image`].
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn rgb_image_to_engine(image: image::RgbImage) -> Image {
    Image {
        width: image.width(),
        height: image.height(),
        pixels: image.into_raw(),
    }
}

/// Build the Wan-VACE conditioning: one [`Conditioning::ControlClip`] (source frames + the
/// per-frame person mask; the engine neutralizes the masked region) plus one
/// [`Conditioning::Reference`] per character reference image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn build_vace_conditioning(
    frames: Vec<Image>,
    masks: Vec<image::RgbImage>,
    references: Vec<Image>,
    masking_strength: f32,
    mode: ReplacementMode,
) -> WorkerResult<Vec<Conditioning>> {
    if frames.len() != masks.len() {
        return Err(WorkerError::InvalidPayload(format!(
            "replace_person: control frames ({}) and masks ({}) length mismatch",
            frames.len(),
            masks.len()
        )));
    }
    let mask_images: Vec<Image> = masks.into_iter().map(rgb_image_to_engine).collect();
    let mut conditioning = Vec::with_capacity(1 + references.len());
    conditioning.push(Conditioning::ControlClip {
        frames,
        mask: mask_images,
        masking_strength,
        start_frame: 0,
        mode,
    });
    for image in references {
        conditioning.push(Conditioning::Reference {
            image,
            strength: None,
        });
    }
    Ok(conditioning)
}

/// The honest `replacementStatus` recorded on the asset fact (mirrors the torch
/// `replacement_status`); the API folds it into the video sidecar's normalizedSettings.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn replacement_status_value(
    track: &Value,
    track_id: &str,
    mask_mode: &str,
    masking_strength: f32,
    reference_count: usize,
    frame_count: usize,
    adapter: &str,
) -> Value {
    let status = track.get("status").and_then(Value::as_object);
    let person_tracking_active = status
        .and_then(|s| s.get("personTrackingActive"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mask_state = status
        .and_then(|s| s.get("maskState"))
        .and_then(Value::as_str)
        .unwrap_or("missing")
        .to_owned();
    let corrections = track.get("corrections").and_then(Value::as_array);
    let correction_count = corrections.map(|list| list.len()).unwrap_or(0);
    let resolved_track_id = track.get("id").and_then(Value::as_str).unwrap_or(track_id);
    json!({
        "personDetectionActive": true,
        "personTrackingActive": person_tracking_active,
        "replacementActive": true,
        "replacementAdapter": adapter,
        "maskMode": mask_mode,
        "maskState": mask_state,
        "maskingStrength": masking_strength,
        "personTrackId": resolved_track_id,
        "characterReferenceCount": reference_count,
        "controlFrameCount": frame_count,
        "usedCorrections": correction_count > 0,
        "correctionCount": correction_count,
    })
}

/// Resolve a replace_person request into a Wan-VACE generation: assemble/resolve the snapshot,
/// extract the source-clip control frames, build the per-frame person mask from the saved
/// track (corrections applied), load the character refs, run the engine, and return the decoded
/// video plus the honest `replacementStatus`. Person detect/track/segment stays upstream.
#[cfg(target_os = "macos")]
pub(super) async fn generate_wan_vace(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_wan_vace_model_dir(settings)?;
    generate_wan_vace_engine(
        api,
        settings,
        job,
        request,
        project_path,
        backend,
        "wan_vace",
        model_dir,
        WAN_VACE_ADAPTER,
        resolve_wan_quant(request),
    )
    .await
}

/// The dual-expert Wan2.2 VACE-Fun replace_person dispatch (sc-3459) — identical conditioning to
/// single-expert [`generate_wan_vace`], but resolves the dual-expert snapshot
/// ([`resolve_wan_vace_fun_model_dir`]) + the `wan2_2_vace_fun_14b` engine. Forces **Q4** by default
/// (the validated real-weight footprint; both 14B experts at bf16 would risk OOM on a 128 GB Mac),
/// still overridable via the `mlxQuantize` advanced knob.
#[cfg(target_os = "macos")]
pub(super) async fn generate_wan_vace_fun(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
) -> WorkerResult<(DecodedVideo, Value)> {
    let model_dir = resolve_wan_vace_fun_model_dir(settings)?;
    let quant = resolve_wan_quant(request).or(Some(Quant::Q4));
    generate_wan_vace_engine(
        api,
        settings,
        job,
        request,
        project_path,
        backend,
        "wan2_2_vace_fun_14b",
        model_dir,
        WAN_VACE_FUN_ADAPTER,
        quant,
    )
    .await
}

/// Shared replace_person engine dispatch for both VACE backends (single-expert `wan_vace` +
/// dual-expert `wan2_2_vace_fun_14b`): builds the source-frame + person-mask + character-reference
/// control conditioning, runs the resolved engine, and returns the decoded video + the honest
/// `replacementStatus`. Only `engine_id` / `model_dir` / `adapter` / `quant` differ between the two.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn generate_wan_vace_engine(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    engine_id: &'static str,
    model_dir: PathBuf,
    adapter: &'static str,
    quant: Option<Quant>,
) -> WorkerResult<(DecodedVideo, Value)> {
    let track_id = request.person_track_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "replace_person requires a person track (personTrackId).".to_owned(),
        )
    })?;
    let track = ProjectStore::new(settings.data_dir.clone(), "worker")
        .get_person_track(&request.project_id, track_id)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("person track {track_id}: {error}"))
        })?;

    // Source frames + masks must match in count and be `1 + 4·k` (one z16 VAE temporal chunk),
    // which `wan_frame_count` guarantees — the engine `validate()` enforces it too.
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let frames =
        load_source_video_frames(api, settings, job, request, project_path, frame_count).await?;
    let (masks, mask_mode) = crate::person_replace::person_track_masks(
        project_path,
        &track,
        request.width,
        request.height,
        frames.len(),
    )?;
    let references = resolve_character_references(settings, request, project_path)?;
    let reference_count = references.len();
    let frame_total = frames.len();

    let masking_strength = advanced::f32(&request.advanced, "maskingStrength", 1.0);
    let conditioning = build_vace_conditioning(
        frames,
        masks,
        references,
        masking_strength,
        replacement_mode_from(&request.replacement_mode),
    )?;

    let negative_prompt = non_empty_negative_prompt(request);
    let steps = super::wan::advanced_opt_u32(request, "steps");
    let guidance = super::wan::advanced_opt_f32(request, "guidanceScale");
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir,
        quant,
        adapters: resolve_wan_vace_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    let decoded = generate_video(api, settings, job, backend, &request.advanced, input).await?;
    let status = replacement_status_value(
        &track,
        track_id,
        mask_mode,
        masking_strength,
        reference_count,
        frame_total,
        adapter,
    );
    Ok((decoded, status))
}

// ---------------------------------------------------------------------------
// Wan extend_clip / video_bridge — native Wan-VACE ControlClip (sc-3812, tier C).
//
// The TI2V-5B single-frame path (`build_wan_boundary_conditioning`, sc-3357) conditions on one
// boundary still, so it morphs *from* a frozen frame and cannot inherit the source clip's motion.
// Routing these modes to the `wan_vace` engine instead lets the model attend to *several real*
// source frames pinned at the kept positions (mask black = keep) while it generates the rest of the
// timeline freely (mask white = regenerate over a neutral-gray control video). That is the whole
// point of extend/bridge — genuine motion continuity — at the cost of the smaller VACE-1.3B base
// (vs TI2V-5B), so the single-frame path stays the baseline/fallback. No reference images: the
// content comes from the kept frames, not a character (the engine's reference path is optional).
// Raw-settings record `model = wan_vace` + `fidelityTier = vace_controlclip` so the engine
// substitution under the user's `wan_2_2` pick is an inspectable fact on the asset, not a black box.

/// Mid-gray (≈0 after the engine's `2·x/255 − 1` normalization) control frame for the
/// to-generate span: a neutral `reactive = video·mask` signal so the masked region is generated
/// freely from the kept frames + prompt, never biased toward a frozen filler image.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn neutral_control_frame(width: u32, height: u32) -> Image {
    Image {
        width,
        height,
        pixels: vec![128u8; (width as usize) * (height as usize) * 3],
    }
}

/// A solid W×H mask (`0` = keep the control frame, `255` = regenerate; the engine binarizes at
/// 0.5), matching the `image::RgbImage` form `person_track_masks` produces for replace_person.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn solid_mask(width: u32, height: u32, value: u8) -> image::RgbImage {
    image::RgbImage::from_pixel(width, height, image::Rgb([value, value, value]))
}

/// How many real source frames to pin as the motion anchor per kept boundary (sc-3812). More =
/// truer continuity but fewer freely-generated frames. Overridable via advanced `motionAnchorFrames`
/// (per side); defaults to ~⅓ of the output budget (split across the two boundaries for bridge), and
/// is clamped so at least 5 frames (one z16 chunk) are left to generate.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn extend_anchor_frames(request: &VideoRequest, frame_count: usize) -> usize {
    let per_side = if request.mode == "video_bridge" { 2 } else { 1 };
    let max_total = frame_count.saturating_sub(5).max(1);
    let max_per_side = (max_total / per_side).max(1);
    let default = (frame_count / 3 / per_side).max(1);
    let requested = request
        .advanced
        .get("motionAnchorFrames")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as usize)
        .unwrap_or(default);
    requested.clamp(1, max_per_side)
}

/// Decode the `take`-end `count` frames of a source clip (its head or tail) to letterboxed W×H
/// engine [`Image`]s, in temporal order (sc-3812). Unlike [`load_source_video_frames`] — which
/// resamples the *whole* clip evenly — this keeps *consecutive* real frames so the model sees the
/// clip's actual motion velocity at the boundary. Decodes only the kept subset (`decode_png_image`).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[allow(clippy::too_many_arguments)]
pub(super) async fn load_clip_anchor_frames(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    project_id: &str,
    project_path: &Path,
    asset_id: &str,
    width: u32,
    height: u32,
    count: usize,
    take: ClipFramePosition,
) -> WorkerResult<Vec<Image>> {
    let media_path = resolve_clip_media_path(settings, project_id, asset_id, project_path)?;
    // Sanitize the job id before it becomes a temp-dir path component (F-111): a hostile id would
    // otherwise escape `temp_dir()` even with the uuid suffix. Mirrors the person-track work dir.
    let work_dir = std::env::temp_dir().join(format!(
        "sw-anchor-frames-{}-{}",
        safe_download_dir(&job.id),
        Uuid::new_v4().simple()
    ));
    tokio::fs::create_dir_all(&work_dir).await?;
    let pattern = work_dir.join("src_%05d.png");
    let filters = format!(
        "scale={width}:{height}:force_original_aspect_ratio=decrease,\
         pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={FRAME_PAD_COLOR},format=rgb24",
    );
    let ctx = FfmpegContext::new(api, settings, &job.id, CANCEL_MESSAGE);
    let extract = run_ffmpeg(
        vec![
            "ffmpeg".to_owned(),
            "-nostdin".to_owned(),
            "-y".to_owned(),
            "-i".to_owned(),
            media_path.display().to_string(),
            "-vf".to_owned(),
            filters,
            "-start_number".to_owned(),
            "0".to_owned(),
            pattern.display().to_string(),
        ],
        Some(ctx),
    )
    .await;
    let frames = match extract {
        Ok(()) => select_anchor_frames(work_dir.clone(), count, take).await,
        Err(error) => Err(error),
    };
    let _ = tokio::fs::remove_dir_all(&work_dir).await;
    frames
}

/// Pick the head/tail `count` consecutive PNGs from `work_dir` (sorted) and decode them to engine
/// [`Image`]s, preserving temporal order. Fewer available than `count` ⇒ all of them (short clip).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
async fn select_anchor_frames(
    work_dir: PathBuf,
    count: usize,
    take: ClipFramePosition,
) -> WorkerResult<Vec<Image>> {
    tokio::task::spawn_blocking(move || -> WorkerResult<Vec<Image>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&work_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(WorkerError::InvalidPayload(
                "source clip produced no decodable frames".to_owned(),
            ));
        }
        let take_n = count.min(paths.len());
        let selected = match take {
            ClipFramePosition::Last => &paths[paths.len() - take_n..],
            ClipFramePosition::First => &paths[..take_n],
        };
        selected.iter().map(|path| decode_png_image(path)).collect()
    })
    .await
    .map_err(|error| task_join_error("frame decode task", error))?
}

/// Decode one RGB PNG into an engine [`Image`] (shared by the resample + anchor frame selectors).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn decode_png_image(path: &Path) -> WorkerResult<Image> {
    let decoded = crate::image_decode::decode_image_any(path)
        .map_err(|error| {
            WorkerError::InvalidPayload(format!("source frame {}: {error}", path.display()))
        })?
        .to_rgb8();
    Ok(Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// Build the Wan-VACE extend/bridge ControlClip (sc-3812): real source frames pinned at the kept
/// positions (mask black) and a neutral-gray generated span (mask white). For `extend_clip` the
/// left-clip tail anchors the start and the continuation is generated; for `video_bridge` both
/// clips' boundary anchors are pinned at the two ends and the gap between them is generated. The
/// control clip is `frame_count` long (`1 + 4·k`, the engine's z16-chunk constraint) with no
/// reference images. `masking_strength`/`mode` are inert in the WanVACE mask math (carried for the
/// shared [`Conditioning::ControlClip`] contract), so they take the neutral defaults.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn build_extend_bridge_vace_conditioning(
    request: &VideoRequest,
    width: u32,
    height: u32,
    frame_count: usize,
    left_anchor: Vec<Image>,
    right_anchor: Option<Vec<Image>>,
) -> WorkerResult<Vec<Conditioning>> {
    let neutral = neutral_control_frame(width, height);
    let keep_mask = solid_mask(width, height, 0);
    let gen_mask = solid_mask(width, height, 255);
    let mut frames: Vec<Image> = Vec::with_capacity(frame_count);
    let mut masks: Vec<image::RgbImage> = Vec::with_capacity(frame_count);
    let left_n = left_anchor.len();
    match request.mode.as_str() {
        "extend_clip" => {
            if left_n + 1 > frame_count {
                return Err(WorkerError::InvalidPayload(format!(
                    "extend_clip: {left_n} anchor frames leave no room to generate in a \
                     {frame_count}-frame clip — reduce motionAnchorFrames."
                )));
            }
            for frame in left_anchor {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
            for _ in left_n..frame_count {
                frames.push(neutral.clone());
                masks.push(gen_mask.clone());
            }
        }
        "video_bridge" => {
            let right = right_anchor.ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
            let right_n = right.len();
            if left_n + right_n + 1 > frame_count {
                return Err(WorkerError::InvalidPayload(format!(
                    "video_bridge: {left_n}+{right_n} anchor frames leave no gap to generate in a \
                     {frame_count}-frame clip — reduce motionAnchorFrames."
                )));
            }
            for frame in left_anchor {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
            for _ in 0..(frame_count - left_n - right_n) {
                frames.push(neutral.clone());
                masks.push(gen_mask.clone());
            }
            for frame in right {
                frames.push(frame);
                masks.push(keep_mask.clone());
            }
        }
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "build_extend_bridge_vace_conditioning: unexpected mode {other}"
            )))
        }
    }
    build_vace_conditioning(frames, masks, Vec::new(), 1.0, ReplacementMode::default())
}

/// Raw-settings for a Wan-VACE extend/bridge asset: the request `advanced` knobs + the real-inference
/// markers, recording the actual engine (`wan_vace`) and `fidelityTier` so the substitution under the
/// user's `wan_2_2` pick is an inspectable fact (sc-3812). Unlike [`wan_vace_raw_settings`] there is
/// no `replacementMode` (these modes are not person-replacement).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn wan_vace_extend_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String("wan_vace".to_owned()));
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "fidelityTier".to_owned(),
        Value::String("vace_controlclip".to_owned()),
    );
    Value::Object(raw)
}

/// Resolve an extend_clip / video_bridge request into a native Wan-VACE generation (sc-3812, tier C).
/// Loads the real source-clip anchor frames (the left clip's tail for extend; both clips' boundaries
/// for bridge), builds the source-at-kept-positions + generated-span ControlClip, and runs the
/// `wan_vace` engine. The TI2V-5B single-frame path ([`generate_wan`]) remains the baseline/fallback,
/// chosen by the dispatch seam when the VACE snapshot is unprovisioned.
#[cfg(target_os = "macos")]
pub(super) async fn generate_wan_vace_extend_bridge(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    backend: &str,
    model_dir: PathBuf,
) -> WorkerResult<DecodedVideo> {
    let frame_count = wan_frame_count(request.raw_frame_count()) as usize;
    let anchor = extend_anchor_frames(request, frame_count);
    let left_id = request.source_clip_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} requires a source clip (sourceClipAssetId).",
            request.mode.replace('_', " ")
        ))
    })?;
    let left_anchor = load_clip_anchor_frames(
        api,
        settings,
        job,
        &request.project_id,
        project_path,
        left_id,
        request.width,
        request.height,
        anchor,
        ClipFramePosition::Last,
    )
    .await?;
    let right_anchor = if request.mode == "video_bridge" {
        let right_id = request
            .bridge_right_clip_asset_id
            .as_deref()
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "video_bridge requires a right-side source clip (bridgeRightClipAssetId)."
                        .to_owned(),
                )
            })?;
        Some(
            load_clip_anchor_frames(
                api,
                settings,
                job,
                &request.project_id,
                project_path,
                right_id,
                request.width,
                request.height,
                anchor,
                ClipFramePosition::First,
            )
            .await?,
        )
    } else {
        None
    };
    let conditioning = build_extend_bridge_vace_conditioning(
        request,
        request.width,
        request.height,
        frame_count,
        left_anchor,
        right_anchor,
    )?;
    let negative_prompt = non_empty_negative_prompt(request);
    let steps = super::wan::advanced_opt_u32(request, "steps");
    let guidance = super::wan::advanced_opt_f32(request, "guidanceScale");
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id: "wan_vace",
        model_dir,
        quant: resolve_wan_quant(request),
        adapters: resolve_wan_vace_adapters(settings, request)?,
        conditioning,
        prompt: request.prompt.clone(),
        negative_prompt,
        width: request.width,
        height: request.height,
        frames: frame_count as u32,
        fps: request.fps,
        steps,
        guidance,
        seed: resolve_video_seed(request) as u64,
        control_scale: Some(advanced::f32(&request.advanced, "conditioningScale", 1.0)),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}
