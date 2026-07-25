#[allow(unused_imports)]
use super::prelude::*;
#[cfg(target_os = "macos")]
use super::wan::{generate_video, VideoGenInput};

// ---------------------------------------------------------------------------
// Real MLX Stable Video Diffusion (SVD-XT) generation (macOS, via mlx-gen-svd, sc-3523):
// image→video ONLY — animates one source image into a fixed ~25-frame burst (no text prompt,
// no audio) via the `motion_bucket_id` / `noise_aug_strength` / conditioning-fps
// micro-conditioning. One engine model `svd_xt`. Source-of-truth = the torch
// `DiffusersVideoAdapter` `svd_video` path (`StableVideoDiffusionPipeline`, video_adapters.py).
// The engine loads the stock diffusers fp16 snapshot directly (vae/ + unet/ + image_encoder/),
// so there is no conversion step (unlike Wan/LTX).
//
// fps (sc-3764): the engine decouples the two cadences — the motion micro-conditioning fps
// (`added_time_ids` = fps − 1) rides `conditioning_fps` (manifest `condFps`, default 7 — the value
// the model was trained on, so MOTION stays correct), while the output/playback fps is the user's
// `request.fps` (mirroring the torch `export_to_video(fps=request.fps)`). So the burst now plays at
// the requested cadence with correct motion — full parity with the torch `svd_video` path.
// ---------------------------------------------------------------------------

/// Adapter id recorded on a real MLX SVD asset — matches the torch `svd_video` adapter id so the
/// asset sidecar reads identically across the two backends.
#[cfg(target_os = "macos")]
pub(super) const SVD_ADAPTER: &str = "svd_video";

/// The diffusers SVD-XT repo the engine loads directly (fp16 `vae/` + `unet/` + `image_encoder/`).
/// Shared by the MLX (macOS) lane and the candle (Windows/CUDA) lane (sc-5493).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) const SVD_REPO: &str = "stabilityai/stable-video-diffusion-img2vid-xt";

/// SceneWorks model id → mlx-gen registry id for the SVD family (only `svd` → `svd_xt`), or `None`.
#[cfg(target_os = "macos")]
pub(super) fn svd_engine_id(model: &str) -> Option<&'static str> {
    (model == "svd").then_some("svd_xt")
}

/// Whether the linked SVD engine can serve this request now (image→video with resolvable weights).
/// SVD is image-conditioned only, so a request without a `sourceAssetId` can never run on it.
#[cfg(target_os = "macos")]
pub(super) fn svd_available(request: &VideoRequest, settings: &Settings) -> bool {
    svd_engine_id(&request.model).is_some()
        && request.source_asset_id.is_some()
        && resolve_svd_model_dir(settings).is_ok()
}

/// Whether `dir` is a usable SVD-XT snapshot — each component subdir carries the safetensors the
/// engine reads (preferring the on-disk `.fp16` variant, else the full-precision file).
#[cfg(any(
    target_os = "macos",
    all(test, not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn svd_dir_is_complete(dir: &Path) -> bool {
    let has = |sub: &str, stem: &str| {
        dir.join(sub)
            .join(format!("{stem}.fp16.safetensors"))
            .is_file()
            || dir.join(sub).join(format!("{stem}.safetensors")).is_file()
    };
    has("vae", "diffusion_pytorch_model")
        && has("unet", "diffusion_pytorch_model")
        && has("image_encoder", "model")
}

/// Resolve the SVD-XT snapshot dir: env override (`SCENEWORKS_MLX_SVD_DIR`) → the cached HF snapshot
/// of [`SVD_REPO`]. Only a dir carrying the three component subdirs ([`svd_dir_is_complete`]) counts.
#[cfg(target_os = "macos")]
pub(super) fn resolve_svd_model_dir(settings: &Settings) -> WorkerResult<PathBuf> {
    if let Ok(override_dir) = std::env::var("SCENEWORKS_MLX_SVD_DIR") {
        let path = PathBuf::from(override_dir.trim());
        if svd_dir_is_complete(&path) {
            return Ok(path);
        }
    }
    if let Some(dir) = huggingface_snapshot_dir(&settings.data_dir, SVD_REPO) {
        if svd_dir_is_complete(&dir) {
            return Ok(dir);
        }
    }
    Err(WorkerError::InvalidPayload(format!(
        "svd: no complete SVD-XT weights found (expected vae/ + unet/ + image_encoder/ under the \
         cached {SVD_REPO} snapshot, or set $SCENEWORKS_MLX_SVD_DIR)"
    )))
}

/// Read an SVD integer knob: `advanced[adv_key]` → `modelManifestEntry[manifest_key]` → `default`,
/// then clamp to `[min, max]`. Mirrors the torch `safe_int(advanced.get(adv_key),
/// target.get(manifest_key, default), min, max)` (advanced overrides the manifest, which overrides
/// the builtin default; the resolved value is clamped).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn svd_i32(
    request: &VideoRequest,
    adv_key: &str,
    manifest_key: &str,
    default: i32,
    min: i32,
    max: i32,
) -> i32 {
    request
        .advanced
        .get(adv_key)
        .or_else(|| request.model_manifest_entry.get(manifest_key))
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as i32)
        .unwrap_or(default)
        .clamp(min, max)
}

/// Read an SVD float knob: `advanced[adv_key]` → `modelManifestEntry[manifest_key]` → `default`
/// (no clamp). Mirrors the torch `float(advanced.get(adv_key, target.get(manifest_key, default)))`.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn svd_f32(
    request: &VideoRequest,
    adv_key: &str,
    manifest_key: &str,
    default: f32,
) -> f32 {
    request
        .advanced
        .get(adv_key)
        .or_else(|| request.model_manifest_entry.get(manifest_key))
        .and_then(|v| v.as_f64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as f32)
        .unwrap_or(default)
}

/// Inference steps for an SVD request: `advanced.steps` → `modelManifestEntry.steps[quality]` (else
/// its `balanced`) → the builtin quality ladder (fast 15 / balanced 25 / best 30), clamped 1..=80.
/// Mirrors the torch `_num_inference_steps` for the `svd_video` adapter.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn svd_steps(request: &VideoRequest) -> u32 {
    let builtin = match request.quality.as_str() {
        "fast" => 15,
        "best" => 30,
        _ => 25,
    };
    let manifest_default = request
        .model_manifest_entry
        .get("steps")
        .and_then(Value::as_object)
        .and_then(|steps| {
            steps
                .get(&request.quality)
                .or_else(|| steps.get("balanced"))
        })
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(builtin);
    request
        .advanced
        .get("steps")
        .and_then(|v| v.as_i64().or_else(|| v.as_str()?.trim().parse().ok()))
        .map(|v| v as i32)
        .unwrap_or(manifest_default)
        .clamp(1, 80) as u32
}

/// The single `Reference` conditioning image (image→video source). SVD is image-conditioned only,
/// so a missing `sourceAssetId` is a hard error (the routing gate [`svd_available`] already
/// requires it; this guards the direct-call path).
#[cfg(target_os = "macos")]
fn resolve_svd_conditioning(
    settings: &Settings,
    request: &VideoRequest,
    project_path: &Path,
) -> WorkerResult<Vec<Conditioning>> {
    let asset_id = request.source_asset_id.as_deref().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "svd image→video requires a source image (sourceAssetId).".to_owned(),
        )
    })?;
    let image = load_reference_image(
        &settings.data_dir,
        &request.project_id,
        asset_id,
        project_path,
    )?;
    Ok(vec![Conditioning::Reference {
        image,
        strength: None,
    }])
}

/// Raw-settings recorded on a real SVD asset (the resolved knobs + real-inference markers). Shared by
/// the MLX (macOS) and candle (Windows/CUDA, sc-5493) lanes.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(super) fn svd_raw_settings(request: &VideoRequest) -> Value {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("model".to_owned(), Value::String(request.model.clone()));
    raw.insert(
        "numFrames".to_owned(),
        json!(svd_i32(request, "numFrames", "numFrames", 25, 1, 25)),
    );
    raw.insert(
        "motionBucketId".to_owned(),
        json!(svd_i32(
            request,
            "motionBucketId",
            "motionBucketId",
            127,
            1,
            255
        )),
    );
    raw.insert(
        "conditioningFps".to_owned(),
        json!(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30)),
    );
    // The output/playback cadence (decoupled from conditioningFps; sc-3764).
    raw.insert("fps".to_owned(), json!(request.fps));
    raw.insert(
        "noiseAugStrength".to_owned(),
        json!(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02
        )),
    );
    raw.insert(
        "decodeChunkSize".to_owned(),
        json!(svd_i32(
            request,
            "decodeChunkSize",
            "decodeChunkSize",
            8,
            1,
            64
        )),
    );
    raw.insert("steps".to_owned(), json!(svd_steps(request)));
    Value::Object(raw)
}

/// Resolve an SVD request into a [`VideoGenInput`] and run it (sc-3523). image→video only: no
/// prompt / negative / guidance (the engine uses its frame-wise CFG ramp); `frames` is the
/// model-fixed burst length (≤25); `fps` carries the motion-conditioning cadence (see the module
/// note); `motion_bucket_id` / `noise_aug_strength` / `decode_chunk_size` drive the SVD knobs.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_svd(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    request: &VideoRequest,
    project_path: &Path,
    engine_id: &'static str,
    backend: &str,
) -> WorkerResult<DecodedVideo> {
    let input = VideoGenInput {
        sampler: None,
        scheduler: None,
        engine_id,
        model_dir: resolve_svd_model_dir(settings)?,
        quant: None,
        adapters: Vec::new(),
        conditioning: resolve_svd_conditioning(settings, request, project_path)?,
        prompt: String::new(),
        negative_prompt: None,
        width: request.width,
        height: request.height,
        frames: svd_i32(request, "numFrames", "numFrames", 25, 1, 25) as u32,
        // Output/playback cadence = the user's `fps` (mirrors the torch `export_to_video(fps=request.fps)`);
        // the motion cadence rides `conditioning_fps` below (sc-3764).
        fps: request.fps,
        steps: Some(svd_steps(request)),
        guidance: None,
        seed: resolve_video_seed(request) as u64,
        motion_bucket_id: Some(
            svd_i32(request, "motionBucketId", "motionBucketId", 127, 1, 255) as f32,
        ),
        noise_aug_strength: Some(svd_f32(
            request,
            "noiseAugStrength",
            "noiseAugStrength",
            0.02,
        )),
        decode_chunk_size: Some(
            svd_i32(request, "decodeChunkSize", "decodeChunkSize", 8, 1, 64) as u32,
        ),
        conditioning_fps: Some(svd_i32(request, "conditioningFps", "condFps", 7, 1, 30) as u32),
        ..VideoGenInput::default()
    };
    generate_video(api, settings, job, backend, &request.advanced, input).await
}
