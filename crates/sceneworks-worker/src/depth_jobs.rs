//! Iris-3B monocular depth as a reusable asset (epic 25678, sc-25682).
//!
//! An `image_depth` job estimates one source image's depth with the `iris_3b_depth` task export and
//! publishes a `type: "depth"` asset:
//!
//! * `file` — the RAW prediction, float32 `H × W` row-major little-endian (`<f4`) `.npy` at the
//!   EXIF-corrected source size ([`sceneworks_core::depth_map`]). Written once, never rewritten.
//! * `preview` — a separately stored PNG derived from it (upstream's inferno colorization).
//! * `depth` — the reload contract: value convention, layout, source geometry (incl. the EXIF
//!   orientation that was applied), preprocessing choices, and model/config identity from the
//!   engine's `DepthMetadata`.
//!
//! Resources: the task's own closure only — the installed `depth/` export (`config.yaml`,
//! `model.safetensors`, `empty_prompt.safetensors`) of `speridlabs/iris-3b` at the pinned revision.
//! No text encoder and no generation backbone are staged, nothing is downloaded at job time, and a
//! missing, torn or wrong-task install is a typed failure before any load.
//!
//! Cancel: the engine checks the flag before and after its forward and returns `Canceled`; the
//! outputs are written to `.partial` files that are only renamed into place after the blocking task
//! returns success, so a canceled or failed job leaves no asset and no stray file behind.

use std::path::{Path, PathBuf};

use gen_core::iris::depth::{
    colorize_inferno, plan_depth_input, DepthInputPlan, DepthMetadata, DepthRequest,
    DepthResolution, IrisDepthEstimator, VALUE_CONVENTION, VALUE_CONVENTION_DESCRIPTION,
};
use gen_core::iris::downstream::{TaskExport, EMPTY_PROMPT_FILE};
use gen_core::iris::{
    IrisTask, BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE, UPSTREAM_WEIGHTS_REPO,
    UPSTREAM_WEIGHTS_REVISION,
};
use gen_core::{CancelFlag, LoadSpec, WeightsSource};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader};
use sceneworks_core::contracts::{JobSnapshot, JobStatus, JsonObject, ProgressStage, WorkerStatus};
use sceneworks_core::depth_map::{
    encode_npy_f32, DepthResolutionRequest, ImageDepthJobPayload, DEPTH_ASSET_TYPE, DEPTH_MODEL_ID,
    NPY_MIME_TYPE,
};
use sceneworks_core::project_store::ProjectStore;
use serde_json::{json, Value};

use crate::{
    heartbeat, progress_payload, run_blocking_with_heartbeat, update_job, ApiClient, Settings,
    WorkerError, WorkerResult,
};

/// The export's folder inside the weights repo.
pub(crate) const DEPTH_EXPORT_SUBDIR: &str = "depth";
/// Every file of the depth task's resource closure.
pub(crate) const DEPTH_EXPORT_FILES: [&str; 3] = [
    BACKBONE_CONFIG_FILE,
    BACKBONE_WEIGHTS_FILE,
    EMPTY_PROMPT_FILE,
];

/// Typed failure codes, prefixed on the job error as `[code]` (the external-library convention).
pub(crate) const NOT_INSTALLED: &str = "iris_depth_not_installed";
pub(crate) const INCOMPLETE_INSTALL: &str = "iris_depth_incomplete_install";
pub(crate) const WRONG_TASK: &str = "iris_depth_wrong_task";
pub(crate) const INVALID_SOURCE: &str = "iris_depth_invalid_source";
pub(crate) const MEMORY_ADMISSION: &str = "iris_depth_memory_admission";
/// The estimate holds no finite value at all — there is no depth to publish (sc-25690).
pub(crate) const NO_VALID_DEPTH: &str = "iris_depth_no_valid_pixels";

fn typed(code: &str, detail: impl std::fmt::Display) -> WorkerError {
    WorkerError::InvalidPayload(format!("[{code}] {detail}"))
}

/// The production loader: the backend's Iris depth provider.
fn backend_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn IrisDepthEstimator>> {
    #[cfg(target_os = "macos")]
    {
        runtime_macos::providers::iris::depth::load(spec)
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        runtime_cuda::providers::iris::depth::load(spec)
    }
}

pub(crate) async fn run_image_depth_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    run_image_depth_job_using(api, settings, job, backend_load).await
}

/// [`run_image_depth_job`] with an injected provider loader — the seam the job/asset/cancel tests
/// drive with a fake estimator, so they exercise this exact job body.
pub(crate) async fn run_image_depth_job_using<L>(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    loader: L,
) -> WorkerResult<()>
where
    L: FnOnce(&LoadSpec) -> gen_core::Result<Box<dyn IrisDepthEstimator>> + Send + 'static,
{
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    let request = ImageDepthJobPayload::from_job_payload(&job.payload)
        .map_err(WorkerError::InvalidPayload)?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            "Checking the Iris-3B depth install.",
            None,
            None,
            None,
        ),
    )
    .await?;
    // A trained depth model (sc-25687) replaces the release export; its task is checked.
    let export = match request.trained_model() {
        Some(model) => {
            let dir = crate::iris_task_training::trained_task_export_dir(
                &settings.data_dir,
                model,
                IrisTask::Depth,
            )
            .map_err(|detail| typed(WRONG_TASK, detail))?;
            TaskExport::from_dir(&dir, IrisTask::Depth, DEPTH_MODEL_ID)
                .map_err(|error| typed(WRONG_TASK, error))?
        }
        None => resolve_depth_export(settings)?,
    };

    let store = ProjectStore::new(settings.data_dir.clone(), "worker");
    let project = store
        .get_project(&request.project_id)
        .map_err(|e| WorkerError::InvalidPayload(format!("project not found: {e}")))?;
    let project_path = PathBuf::from(&project.path);
    let source_asset = store
        .get_asset(&request.project_id, &request.source_asset_id)
        .map_err(|e| {
            typed(
                INVALID_SOURCE,
                format!("source asset {}: {e}", request.source_asset_id),
            )
        })?;
    ensure_image_source(&source_asset, &request.source_asset_id)?;
    let source_rel = source_asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .ok_or_else(|| typed(INVALID_SOURCE, "the source asset has no media file"))?;
    let source_path = crate::safe_project_path(&project_path, source_rel)?;
    let source_name = request
        .display_name
        .clone()
        .or_else(|| {
            source_asset
                .get("displayName")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "Image".to_owned());

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.15,
            "Loading source image.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let source = tokio::task::spawn_blocking(move || decode_oriented_rgb8(&source_path))
        .await
        .map_err(|error| crate::task_join_error("depth source decode", error))??;
    let resolution = engine_resolution(request.resolution);
    let plan = plan_depth_input(
        source.image.width(),
        source.image.height(),
        resolution,
        export.config.model.patch_size,
    )
    .map_err(|e| typed(INVALID_SOURCE, e))?;

    // One resident engine across the Iris tasks (sc-25684): the estimator below loads its own ~6.5 GB
    // depth backbone for this job, so a warm generator (with its Qwen3-VL text encoder) or a resident
    // restoration transform leaves first. Otherwise the two would sit side by side over the user's
    // memory cap — the unified budget below is the cap minus the host reserve and does not see
    // them — and on CUDA the free-VRAM probe would count the departing model and refuse a run that
    // fits once it is gone.
    evict_resident_engines().await?;
    #[cfg(target_os = "macos")]
    let available_gb = mlx_available_gb();
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let available_gb = candle_free_gb(settings).await;
    let declaration = DepthAdmission::from_payload(&job.payload, ADMISSION_LANE)?;
    admission_decision(&declaration, &plan, available_gb)?;

    // Reserve the output names now so a cancel/failure can clean exactly these files.
    let outputs = DepthOutputPaths::reserve(&project_path);
    if let Some(parent) = outputs.npy.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.3,
            &format!(
                "Estimating depth at {}x{} (source {}x{}).",
                plan.model_width, plan.model_height, plan.source_width, plan.source_height
            ),
            None,
            None,
            None,
        ),
    )
    .await?;

    let cancel = CancelFlag::new();
    let spec = depth_load_spec(&export.dir);
    let task_cancel = cancel.clone();
    let partial_npy = outputs.partial_npy.clone();
    let partial_preview = outputs.partial_preview.clone();
    let rgb = source.image;
    let task = tokio::task::spawn_blocking(move || -> WorkerResult<DepthMetadata> {
        let estimator = loader(&spec).map_err(|error| map_engine_error("load", error))?;
        let (width, height) = (rgb.width(), rgb.height());
        let request = DepthRequest {
            image: gen_core::Image {
                width,
                height,
                pixels: rgb.into_raw(),
            },
            resolution,
            cancel: task_cancel,
        };
        let output = estimator
            .estimate(&request, &mut |_| {})
            .map_err(|error| map_engine_error("estimate", error))?;
        if (output.map.width, output.map.height) != (width, height) {
            return Err(WorkerError::Engine(format!(
                "the depth estimator returned a {}x{} map for a {width}x{height} source",
                output.map.width, output.map.height
            )));
        }
        // A map with no finite value is not depth (a numerically diverged estimate — e.g. a
        // trained export that blew up): refuse it before anything is staged, rather than publish
        // an asset every consumer would read as an all-NaN array. The raw values of any other map
        // are published untouched.
        if !output.map.values.iter().any(|value| value.is_finite()) {
            return Err(WorkerError::Engine(format!(
                "[{NO_VALID_DEPTH}] the depth estimator produced no finite depth value for the \
                 {width}x{height} source; nothing was saved"
            )));
        }
        let npy = encode_npy_f32(output.map.width, output.map.height, &output.map.values)
            .map_err(WorkerError::Engine)?;
        std::fs::write(&partial_npy, npy)?;
        let preview = colorize_inferno(&output.map);
        image::RgbImage::from_raw(preview.width, preview.height, preview.pixels)
            .ok_or_else(|| WorkerError::Engine("depth preview buffer size mismatch".to_owned()))?
            .save_with_format(&partial_preview, image::ImageFormat::Png)
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        Ok(output.metadata)
    });
    let metadata = match run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        "Depth estimation canceled.",
        "depth estimation task",
        crate::no_cancel_ack(),
        task,
    )
    .await
    {
        Ok(metadata) => metadata,
        Err(error) => {
            outputs.discard();
            return Err(error);
        }
    };
    if let Err(error) = outputs.publish() {
        outputs.discard();
        return Err(error);
    }

    let fact = depth_asset_fact(DepthFactInputs {
        outputs: &outputs,
        request: &request,
        source_name: &source_name,
        source: &source.geometry,
        metadata: &metadata,
        trained_export_path: request
            .trained_model()
            .map(|_| data_relative_path(&settings.data_dir, &export.dir)),
    })?;
    let result = depth_job_result(&outputs, &metadata, request.trained_model(), fact);
    // The API persists the asset only when it accepts this completion; if it refuses, the
    // published files would be orphans, so remove them before reporting the failure.
    if let Err(error) = update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Depth map ready.",
            None,
            Some(result),
            None,
        ),
    )
    .await
    {
        outputs.discard();
        return Err(error);
    }
    Ok(())
}

/// Drop the resident generator (and with it any staged text encoder) and the resident restoration
/// transform before a depth estimate loads. The depth estimator is per-job (never cached), so after
/// this job nothing of it stays resident either.
async fn evict_resident_engines() -> WorkerResult<()> {
    let generator = crate::generator_cache::evict_cached_generator().await?;
    let transform = crate::transform_cache::evict_resident_transform().await?;
    if generator || transform {
        tracing::info!(
            event = "iris_depth_evicted_resident_engines",
            generator,
            transform
        );
    }
    Ok(())
}

fn map_engine_error(stage: &str, error: gen_core::Error) -> WorkerError {
    match error {
        gen_core::Error::Canceled => WorkerError::Canceled("Depth estimation canceled.".to_owned()),
        gen_core::Error::Unsupported(detail) => typed(WRONG_TASK, detail),
        other => WorkerError::Engine(format!("Iris depth {stage}: {other}")),
    }
}

fn engine_resolution(request: DepthResolutionRequest) -> DepthResolution {
    match request {
        DepthResolutionRequest::Capped { max_side } => DepthResolution::Capped(max_side),
        DepthResolutionRequest::Native => DepthResolution::Native,
    }
}

/// The load spec: the `depth/` export directory as the weights, nothing else staged (the engine
/// refuses any component or text encoder), upstream's bf16 compute by default.
pub(crate) fn depth_load_spec(export_dir: &Path) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(export_dir.to_path_buf()))
}

/// Resolve the installed `depth/` export of the pinned weights revision and check it is a complete
/// DEPTH export, before any load. Cache-only: this never downloads.
pub(crate) fn resolve_depth_export(settings: &Settings) -> WorkerResult<TaskExport> {
    let resolved: Vec<(&str, Option<PathBuf>)> = DEPTH_EXPORT_FILES
        .iter()
        .map(|file| {
            (
                *file,
                crate::downloads::resolve_hf_component_file(
                    settings,
                    UPSTREAM_WEIGHTS_REPO,
                    UPSTREAM_WEIGHTS_REVISION,
                    &format!("{DEPTH_EXPORT_SUBDIR}/{file}"),
                ),
            )
        })
        .collect();
    let missing: Vec<&str> = resolved
        .iter()
        .filter(|(_, path)| path.is_none())
        .map(|(file, _)| *file)
        .collect();
    if missing.len() == DEPTH_EXPORT_FILES.len() {
        return Err(typed(
            NOT_INSTALLED,
            format!(
                "Iris 3B Depth is not installed. Install \"Iris 3B Depth\" from the Model Manager \
                 (it downloads only the {DEPTH_EXPORT_SUBDIR}/ export of {UPSTREAM_WEIGHTS_REPO})."
            ),
        ));
    }
    if !missing.is_empty() {
        return Err(typed(
            INCOMPLETE_INSTALL,
            format!(
                "the Iris 3B Depth install is incomplete — missing {DEPTH_EXPORT_SUBDIR}/{} at \
                 revision {UPSTREAM_WEIGHTS_REVISION}. Re-install \"Iris 3B Depth\" from the Model \
                 Manager.",
                missing.join(", ")
            ),
        ));
    }
    let dir = resolved[0]
        .1
        .as_ref()
        .and_then(|config| config.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| typed(INCOMPLETE_INSTALL, "the depth export has no directory"))?;
    TaskExport::from_dir(&dir, IrisTask::Depth, DEPTH_MODEL_ID).map_err(|error| match error {
        gen_core::Error::Unsupported(detail) => typed(WRONG_TASK, detail),
        other => typed(
            INCOMPLETE_INSTALL,
            format!("the installed depth export does not load: {other}"),
        ),
    })
}

fn ensure_image_source(asset: &Value, asset_id: &str) -> WorkerResult<()> {
    let asset_type = asset.get("type").and_then(Value::as_str).unwrap_or("");
    let mime = asset
        .pointer("/file/mimeType")
        .and_then(Value::as_str)
        .unwrap_or("");
    if asset_type == DEPTH_ASSET_TYPE {
        return Err(typed(
            INVALID_SOURCE,
            format!("{asset_id} is already a depth map; depth is estimated from an image"),
        ));
    }
    if asset_type == "vector"
        || mime == "image/svg+xml"
        || !(asset_type == "image" || mime.starts_with("image/"))
    {
        return Err(typed(
            INVALID_SOURCE,
            format!("{asset_id} is not a raster image (type {asset_type:?}, mime {mime:?})"),
        ));
    }
    Ok(())
}

/// The decoded source and what was done to its geometry.
pub(crate) struct OrientedSource {
    pub image: image::RgbImage,
    pub geometry: SourceGeometry,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SourceGeometry {
    /// The EXIF orientation tag that was applied (1 = none / absent).
    pub exif_orientation: u8,
    /// The stored pixel grid, before orientation.
    pub stored_width: u32,
    pub stored_height: u32,
    /// The upright (EXIF-corrected) size — the depth map's size.
    pub width: u32,
    pub height: u32,
}

/// Decode a source image upright: upstream's `ImageOps.exif_transpose(image).convert("RGB")`. A
/// format the native decoders cannot read is transcoded first (the transcoder bakes orientation into
/// pixels and PNG carries none, so its orientation is reported as 1).
pub(crate) fn decode_oriented_rgb8(path: &Path) -> WorkerResult<OrientedSource> {
    let native = || -> image::ImageResult<(DynamicImage, Orientation)> {
        let mut decoder = ImageReader::open(path)?
            .with_guessed_format()?
            .into_decoder()?;
        let orientation = decoder.orientation()?;
        Ok((DynamicImage::from_decoder(decoder)?, orientation))
    };
    let (mut image, orientation) = match native() {
        Ok(decoded) => decoded,
        Err(_) => (
            crate::image_decode::decode_image_any(path).map_err(|error| {
                typed(
                    INVALID_SOURCE,
                    format!("the source image could not be decoded: {error}"),
                )
            })?,
            Orientation::NoTransforms,
        ),
    };
    let (stored_width, stored_height) = (image.width(), image.height());
    image.apply_orientation(orientation);
    let rgb = image.to_rgb8();
    Ok(OrientedSource {
        geometry: SourceGeometry {
            exif_orientation: orientation.to_exif(),
            stored_width,
            stored_height,
            width: rgb.width(),
            height: rgb.height(),
        },
        image: rgb,
    })
}

// ---- memory admission -------------------------------------------------------------------------

/// The manifest's DECLARED depth admission (`depth.admission`, `measured: false` until the epic-end
/// campaign): a resident floor plus the activation delta observed at a reference model area, scaled
/// with the model input's pixel count. Both engines run attention through fused/budgeted kernels, so
/// the transient grows with the token count; above the reference area the extrapolation is padded by
/// `extrapolationFactor` because nothing larger has been observed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DepthAdmission {
    pub reference_model_pixels: f64,
    pub extrapolation_factor: f64,
    pub resident_gb: f64,
    pub reference_peak_gb: f64,
}

impl DepthAdmission {
    /// Read the declaration for `lane` (`"mlx"` | `"candle"`) from the job's attached catalog entry.
    pub(crate) fn from_payload(payload: &JsonObject, lane: &str) -> WorkerResult<Self> {
        // The model the job runs: a trained depth model's own catalog entry (which carries its
        // engine's depth declaration — the API attaches it in place of the release entry, S6), else
        // the release `iris_3b_depth` entry.
        let model = payload
            .get(sceneworks_core::image_restoration::IRIS_MODEL_KEY)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .unwrap_or(DEPTH_MODEL_ID);
        let entry = payload
            .get("modelManifestEntries")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .chain(payload.get("modelManifestEntry"))
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model))
            .ok_or_else(|| {
                typed(
                    MEMORY_ADMISSION,
                    format!(
                        "the job carries no {model} catalog entry, so it has no memory \
                         declaration; refusing an unbounded run"
                    ),
                )
            })?;
        let admission = entry.pointer("/depth/admission").ok_or_else(|| {
            typed(
                MEMORY_ADMISSION,
                format!("{DEPTH_MODEL_ID} declares no depth.admission; refusing an unbounded run"),
            )
        })?;
        let number = |pointer: &str| {
            admission
                .pointer(pointer)
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| {
                    typed(
                        MEMORY_ADMISSION,
                        format!("{DEPTH_MODEL_ID} depth.admission{pointer} is missing or invalid"),
                    )
                })
        };
        Ok(Self {
            reference_model_pixels: number("/referenceModelPixels")?,
            extrapolation_factor: number("/extrapolationFactor")?,
            resident_gb: number(&format!("/{lane}/residentGb"))?,
            reference_peak_gb: number(&format!("/{lane}/referencePeakGb"))?,
        })
    }

    /// Predicted process peak (GB) for a model input of `model_pixels`.
    pub(crate) fn predicted_peak_gb(&self, model_pixels: u64) -> f64 {
        let ratio = model_pixels as f64 / self.reference_model_pixels;
        let activations = (self.reference_peak_gb - self.resident_gb).max(0.0) * ratio;
        let padded = if ratio > 1.0 {
            activations * self.extrapolation_factor
        } else {
            activations
        };
        self.resident_gb + padded
    }
}

#[cfg(target_os = "macos")]
const ADMISSION_LANE: &str = "mlx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const ADMISSION_LANE: &str = "candle";

/// The MLX budget: the (capped) unified-memory total minus the host reserve. `None` when it cannot be
/// probed — which refuses rather than admitting unbounded.
#[cfg(target_os = "macos")]
fn mlx_available_gb() -> Option<f64> {
    let total_gb = crate::mlx_fit_gate::resolve_budget(
        crate::mlx_fit_gate::probe_total_unified_memory_bytes()
            .map(|bytes| bytes as f64 / (1u64 << 30) as f64),
        crate::mlx_fit_gate::mlx_memory_cap_gb(),
    )?
    .total_gb;
    Some(total_gb - crate::fit_gate::legacy_unified_reserve(total_gb).gb)
}

/// The Candle budget: free VRAM on the selected card (capped), minus the allocator headroom.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
async fn candle_free_gb(settings: &Settings) -> Option<f64> {
    crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    )
    .map(|budget| budget.free_gb - crate::vram_gate::HEADROOM_GB)
}

/// Admit or refuse BEFORE loading. Never shrinks the requested resolution: a refusal names the
/// numbers and the explicit alternative (a smaller cap).
pub(crate) fn admission_decision(
    declaration: &DepthAdmission,
    plan: &DepthInputPlan,
    available_gb: Option<f64>,
) -> WorkerResult<()> {
    let pixels = u64::from(plan.model_width) * u64::from(plan.model_height);
    let needed_gb = declaration.predicted_peak_gb(pixels);
    let available_gb = available_gb.ok_or_else(|| {
        typed(
            MEMORY_ADMISSION,
            "the GPU memory budget is unavailable; refusing an unbounded depth run",
        )
    })?;
    if needed_gb > available_gb {
        return Err(typed(
            MEMORY_ADMISSION,
            format!(
                "depth at a {}x{} model input (source {}x{}) needs about {needed_gb:.1} GB, but \
                 {available_gb:.1} GB is available. Choose a smaller max side.",
                plan.model_width, plan.model_height, plan.source_width, plan.source_height
            ),
        ));
    }
    Ok(())
}

// ---- outputs -----------------------------------------------------------------------------------

/// Where one depth asset's two files go, and their `.partial` staging names.
pub(crate) struct DepthOutputPaths {
    pub asset_id: String,
    pub generation_set_id: String,
    pub created_at: String,
    pub npy_rel: String,
    pub preview_rel: String,
    pub npy: PathBuf,
    pub preview: PathBuf,
    pub partial_npy: PathBuf,
    pub partial_preview: PathBuf,
}

impl DepthOutputPaths {
    pub(crate) fn reserve(project_path: &Path) -> Self {
        let created_at = crate::now_rfc3339();
        let generation_set_id = format!("genset_{}", uuid::Uuid::new_v4().simple());
        let asset_id = crate::fresh_asset_id();
        let suffix: String = asset_id.chars().skip(6).take(8).collect();
        let stem = format!("{}_depth_{suffix}", &created_at[..10]);
        let npy_rel = format!("assets/depth/{generation_set_id}/{stem}.npy");
        let preview_rel = format!("assets/depth/{generation_set_id}/{stem}.preview.png");
        let npy = project_path.join(&npy_rel);
        let preview = project_path.join(&preview_rel);
        Self {
            partial_npy: npy.with_extension("npy.partial"),
            partial_preview: preview.with_extension("png.partial"),
            asset_id,
            generation_set_id,
            created_at,
            npy_rel,
            preview_rel,
            npy,
            preview,
        }
    }

    /// Move both staged files into place (preview first, so a published `.npy` always has one).
    pub(crate) fn publish(&self) -> WorkerResult<()> {
        std::fs::rename(&self.partial_preview, &self.preview)?;
        std::fs::rename(&self.partial_npy, &self.npy)?;
        Ok(())
    }

    /// Remove every file this job may have produced, and the (then empty) set directory.
    pub(crate) fn discard(&self) {
        for path in [
            &self.partial_npy,
            &self.partial_preview,
            &self.npy,
            &self.preview,
        ] {
            let _ = std::fs::remove_file(path);
        }
        if let Some(dir) = self.npy.parent() {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

struct DepthFactInputs<'a> {
    outputs: &'a DepthOutputPaths,
    request: &'a ImageDepthJobPayload,
    source_name: &'a str,
    source: &'a SourceGeometry,
    metadata: &'a DepthMetadata,
    /// The data-relative export path of a trained depth model (`None` for the release export).
    trained_export_path: Option<String>,
}

fn resolution_json(resolution: DepthResolutionRequest) -> Value {
    serde_json::to_value(resolution).unwrap_or(Value::Null)
}

/// The `assetWrites` fact the API persists as the depth asset's sidecar.
fn depth_asset_fact(inputs: DepthFactInputs<'_>) -> WorkerResult<Value> {
    let DepthFactInputs {
        outputs,
        request,
        source_name,
        source,
        metadata: m,
        trained_export_path,
    } = inputs;
    let (input_resample, output_resample) = (m.input_resample, m.output_resample);
    let resolution = resolution_json(request.resolution);
    let mut fact = depth_asset_fact_json(
        outputs,
        request,
        source_name,
        source,
        m,
        resolution,
        input_resample,
        output_resample,
    );
    // A trained depth model (sc-25687): the catalog id of the Training Studio export used, on the
    // asset (`model`), its reload settings (`irisModel`) and its model identity (`trainedModelId`).
    // Its weights are the TRAINED export, not the release snapshot: the lineage names the training
    // source and the export's data-relative path, and carries no release repo/revision (S5).
    if let Some(model) = request.trained_model() {
        fact["model"] = json!(model);
        fact["rawAdapterSettings"]["depth"]["irisModel"] = json!(model);
        let lineage = &mut fact["depth"]["model"];
        lineage["trainedModelId"] = json!(model);
        lineage["weightsSource"] = json!("training");
        lineage["weightsRepo"] = Value::Null;
        lineage["weightsRevision"] = Value::Null;
        lineage["weightsSubdir"] = Value::Null;
        lineage["exportPath"] = json!(trained_export_path);
    }
    Ok(fact)
}

/// The data-relative path of a trained depth export (`models/finetunes/<id>/<export>`), or the
/// absolute path when it is not under the data dir (never expected: the resolver confines it).
fn data_relative_path(data_dir: &Path, dir: &Path) -> String {
    let data_dir = std::fs::canonicalize(data_dir).unwrap_or_else(|_| data_dir.to_path_buf());
    dir.strip_prefix(&data_dir)
        .unwrap_or(dir)
        .to_string_lossy()
        .replace('\\', "/")
}

#[allow(clippy::too_many_arguments)]
fn depth_asset_fact_json(
    outputs: &DepthOutputPaths,
    request: &ImageDepthJobPayload,
    source_name: &str,
    source: &SourceGeometry,
    m: &DepthMetadata,
    resolution: Value,
    input_resample: &str,
    output_resample: &str,
) -> Value {
    json!({
        "assetId": outputs.asset_id,
        "type": DEPTH_ASSET_TYPE,
        "mediaPath": outputs.npy_rel,
        "mimeType": NPY_MIME_TYPE,
        "width": m.source_width,
        "height": m.source_height,
        "displayName": format!("{source_name} (depth)"),
        "createdAt": outputs.created_at,
        "mode": "image_depth",
        "model": DEPTH_MODEL_ID,
        "adapter": m.backend,
        "sourceAssetId": request.source_asset_id,
        "parents": [request.source_asset_id],
        "rawAdapterSettings": {
            "depth": {
                "resolution": resolution,
                "sourceAssetId": request.source_asset_id,
            },
        },
        "preview": {
            "path": outputs.preview_rel,
            "mimeType": "image/png",
            "width": m.source_width,
            "height": m.source_height,
            "derivation": "inferno colormap of the near-bright 2nd-98th percentile scaling (upstream colorize)",
        },
        "depth": {
            "schemaVersion": 1,
            "valueConvention": VALUE_CONVENTION,
            "valueConventionDescription": VALUE_CONVENTION_DESCRIPTION,
            "nearValue": -1.0,
            "farValue": 1.0,
            "metric": false,
            "clamped": false,
            "normalized": false,
            "dtype": "float32",
            "npyDescr": "<f4",
            "byteOrder": "little",
            "layout": "row-major",
            "shape": [m.source_height, m.source_width],
            "source": {
                "assetId": request.source_asset_id,
                "width": source.width,
                "height": source.height,
                "storedWidth": source.stored_width,
                "storedHeight": source.stored_height,
                "exifOrientation": source.exif_orientation,
                "exifOrientationApplied": source.exif_orientation != 1,
            },
            "preprocessing": {
                "resolution": resolution,
                "maxSide": m.resolution.max_side(),
                "modelWidth": m.model_width,
                "modelHeight": m.model_height,
                "inputResample": input_resample,
                "outputResample": output_resample,
            },
            "model": {
                "id": m.model_id,
                "task": m.task.name(),
                "backend": m.backend,
                "codeRevision": m.code_revision,
                "weightsRepo": m.weights_repo,
                "weightsRevision": m.weights_revision,
                "weightsSubdir": DEPTH_EXPORT_SUBDIR,
                "configSha256": m.config_sha256,
                "computeDtype": m.compute_dtype,
            },
        },
    })
}

fn depth_job_result(
    outputs: &DepthOutputPaths,
    metadata: &DepthMetadata,
    // The trained depth model the job ran (S5): the result and its generation set name IT, never
    // the release `iris_3b_depth`.
    trained_model: Option<&str>,
    fact: Value,
) -> JsonObject {
    let model = trained_model.unwrap_or(DEPTH_MODEL_ID);
    let mut result = JsonObject::new();
    result.insert(
        "generationSetId".to_owned(),
        Value::String(outputs.generation_set_id.clone()),
    );
    result.insert("expectedCount".to_owned(), json!(1));
    result.insert("adapter".to_owned(), json!(metadata.backend));
    result.insert("model".to_owned(), json!(model));
    result.insert(
        "generationSet".to_owned(),
        json!({
            "id": outputs.generation_set_id,
            "mode": "image_depth",
            "model": model,
            "prompt": "",
            "negativePrompt": "",
            "count": 1,
            "createdAt": outputs.created_at,
        }),
    );
    result.insert("assetWrites".to_owned(), Value::Array(vec![fact]));
    result
}

#[cfg(test)]
mod tests;
// Real-weight acceptance on this build's native lane (ignored; GPU + installed export required).
#[cfg(test)]
mod gpu_smoke;
