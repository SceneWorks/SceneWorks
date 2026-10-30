//! Iris-3B restoration / tiled scaling (sc-25683, epic 25678).
//!
//! The restoration task of `speridlabs/iris-3b` is the registered [`gen_core::Transform`]
//! `iris_3b_restore`: a one-step model that bicubically upsamples the (optionally budget-fitted) input
//! to `floor(input · scale)`, then restores it tile by tile (1024-px tiles, 50 % overlap) with the
//! shipped empty-prompt conditioning — no text encoder, no generation backbone (E4). SceneWorks
//! serves it as the `iris` engine of the existing image-upscale surfaces (the standalone
//! `image_upscale` job behind the Image Editor and Library actions, and the Image Studio inline
//! post-processing pass), so every existing upscaler keeps its behaviour.
//!
//! This module owns the Iris-specific pieces those surfaces share:
//!
//! * **the plan** ([`preview_restore_plan`]) — the engine's own pure planner
//!   (`gen_core_iris::restoration::plan`) run with no weights, so the UI can show the real
//!   processing and output dimensions BEFORE submission, labelled honestly when the input budget
//!   downscaled the source ([`plan_summary`]); the job records the same plan on its result;
//! * **the install** ([`resolve_restore_export_dir`]) — cache-only resolution of the task's own
//!   `upscaler/` export at the pinned revision, with typed missing / incomplete / wrong-task
//!   refusals before anything loads;
//! * **admission** ([`admission_error`]) — a declared, unmeasured (`measured: false`) per-lane floor
//!   plus the host-side output canvas on unified memory; it refuses honestly and never changes the
//!   scale or the budget mode;
//! * **the run** ([`restore_image_using`]) — the cached transform (see [`crate::transform_cache`]),
//!   per-tile progress, and cancel between tiles (no image, so no partial asset).

use std::path::{Path, PathBuf};

use gen_core::{CancelFlag, InputSizing, TargetSize, TransformRequest};
use gen_core_iris::restoration::{
    self, Dims, RestorationOptions, RestorationPlan, RestorationResources, TileGeometry,
};
use sceneworks_core::image_restoration::{IrisRestoreParams, RestoreInputSizing};
use serde::Serialize;

use crate::{Settings, WorkerError, WorkerResult};

/// The registered transform id (and catalog entry id) — one name for both, asserted in the tests.
pub const IRIS_RESTORE_MODEL_ID: &str = restoration::MODEL_ID;
/// The upstream weights repo and the exact revision the `iris_3b_restore` catalog entry installs.
pub(crate) const IRIS_RESTORE_REPO: &str = gen_core_iris::UPSTREAM_WEIGHTS_REPO;
pub(crate) const IRIS_RESTORE_REVISION: &str = gen_core_iris::UPSTREAM_WEIGHTS_REVISION;
/// The restoration task's export folder inside the repo.
pub(crate) fn restore_subdir() -> &'static str {
    gen_core_iris::IrisTask::Restoration.backbone_subdir()
}
/// The three files of the export — the task's whole resource closure.
pub(crate) const RESTORE_EXPORT_FILES: [&str; 3] = [
    gen_core_iris::BACKBONE_CONFIG_FILE,
    gen_core_iris::BACKBONE_WEIGHTS_FILE,
    restoration::EMPTY_PROMPT_FILE,
];

/// The largest output (and patch-padded processing canvas) restoration admits per side / in total —
/// the same bound every image upscaler in this worker applies before allocating.
pub const MAX_RESTORE_SIDE: u32 = 8192;
pub const MAX_RESTORE_PIXELS: u64 = MAX_RESTORE_SIDE as u64 * MAX_RESTORE_SIDE as u64;

// ---------------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------------

fn target_size(params: &IrisRestoreParams) -> TargetSize {
    match params.scale {
        // `TargetSize::Scale` is f32; the engine reads it back as the shortest decimal that
        // round-trips the f32 (`1.3f32` → 1.3), so a UI-entered decimal floors exactly as upstream.
        Some(scale) => TargetSize::Scale(scale as f32),
        None => TargetSize::ModelDefault,
    }
}

fn input_sizing(params: &IrisRestoreParams) -> InputSizing {
    match params.input_sizing {
        RestoreInputSizing::Budgeted => InputSizing::Budgeted,
        RestoreInputSizing::Original => InputSizing::Original,
    }
}

/// The engine request for one restoration. Only the three Iris controls are set: seed / strength /
/// steps are refused by the engine by name, and nothing here sets them.
pub(crate) fn transform_request(
    image: gen_core::Image,
    params: &IrisRestoreParams,
    cancel: CancelFlag,
) -> TransformRequest {
    TransformRequest {
        image,
        target: target_size(params),
        input_sizing: input_sizing(params),
        color_fix: params.color_fix,
        cancel,
        ..Default::default()
    }
}

/// Resolve the controls exactly as the engine will (`RestorationOptions::from_request`), so the
/// preview and the job agree with the run on defaults and refusals.
fn restoration_options(params: &IrisRestoreParams) -> Result<RestorationOptions, String> {
    let probe = TransformRequest {
        target: target_size(params),
        input_sizing: input_sizing(params),
        color_fix: params.color_fix,
        ..Default::default()
    };
    RestorationOptions::from_request(&probe, IRIS_RESTORE_MODEL_ID).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------------------------
// Plan + preview
// ---------------------------------------------------------------------------------------------

/// Plan a restoration of a `width`×`height` source with `geometry` (the release geometry for a
/// preview, the installed export's own for a job), then apply the worker's output bounds. A refusal
/// names the reason; it never substitutes a different scale or budget mode.
pub(crate) fn plan_restore(
    width: u32,
    height: u32,
    params: &IrisRestoreParams,
    geometry: TileGeometry,
) -> Result<RestorationPlan, String> {
    let options = restoration_options(params)?;
    let plan = restoration::plan(
        Dims::new(width, height),
        &options,
        geometry,
        IRIS_RESTORE_MODEL_ID,
    )
    .map_err(|e| e.to_string())?;
    check_plan_bounds(&plan)?;
    Ok(plan)
}

fn check_plan_bounds(plan: &RestorationPlan) -> Result<(), String> {
    for (stage, dims) in [("output", plan.output), ("processing canvas", plan.padded)] {
        let pixels = u64::from(dims.width) * u64::from(dims.height);
        if dims.width > MAX_RESTORE_SIDE
            || dims.height > MAX_RESTORE_SIDE
            || pixels > MAX_RESTORE_PIXELS
        {
            return Err(format!(
                "Iris restoration of a {}x{} image at {}× ({} input {}x{}) needs a {}x{} {stage}, \
                 over the {MAX_RESTORE_SIDE}px-per-side limit. Pick a smaller scale{}.",
                plan.source.width,
                plan.source.height,
                format_scale(plan.scale),
                sizing_label(plan),
                plan.input.width,
                plan.input.height,
                dims.width,
                dims.height,
                if plan.input_sizing == InputSizing::Original {
                    " or the budgeted input"
                } else {
                    ""
                },
            ));
        }
    }
    Ok(())
}

fn sizing_label(plan: &RestorationPlan) -> &'static str {
    if plan.budget_applied {
        "budgeted"
    } else if plan.input_sizing == InputSizing::Original {
        "original"
    } else {
        "within-budget"
    }
}

/// `4`, `1.3`, `2.25` — the scale as a person typed it (no trailing zeros).
pub(crate) fn format_scale(scale: f64) -> String {
    let text = format!("{scale:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// The one-line, honest description of a plan, for the UI's pre-submission preview and the result
/// record. A budget-downscaled input is named as such — the output is `scale×` of the BUDGETED
/// input, never "N× of the original".
pub fn plan_summary(plan: &RestorationPlan) -> String {
    let (input, output) = (plan.input, plan.output);
    let tiles = plan.forward_count();
    let tiles = if tiles == 1 {
        "1 tile".to_owned()
    } else {
        format!("{tiles} tiles")
    };
    let scale = format_scale(plan.scale);
    let processing = format!(
        "processed at {}x{} in {tiles}",
        plan.padded.width, plan.padded.height
    );
    if plan.budget_applied {
        format!(
            "budgeted: {}x{} input → {}x{} ({scale}× of the budgeted input; source {}x{}, {processing})",
            input.width,
            input.height,
            output.width,
            output.height,
            plan.source.width,
            plan.source.height,
        )
    } else if (plan.scale - 1.0).abs() < f64::EPSILON {
        format!(
            "1× restore: {}x{} → {}x{} ({processing})",
            input.width, input.height, output.width, output.height
        )
    } else {
        format!(
            "{scale}×: {}x{} → {}x{} ({processing})",
            input.width, input.height, output.width, output.height
        )
    }
}

/// Width × height as JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DimsView {
    pub width: u32,
    pub height: u32,
}

impl From<Dims> for DimsView {
    fn from(dims: Dims) -> Self {
        Self {
            width: dims.width,
            height: dims.height,
        }
    }
}

/// The plan as the API returns it for the pre-submission preview and the job records on its result.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RestorePlanView {
    pub engine: &'static str,
    pub model: &'static str,
    pub source: DimsView,
    pub input_sizing: &'static str,
    pub input: DimsView,
    pub budget_applied: bool,
    pub scale: f64,
    pub output: DimsView,
    /// The size the model runs at (the output, enlarged to one full tile when smaller).
    pub processing: DimsView,
    pub enlarged: bool,
    /// The patch-padded processing canvas the tiles cover.
    pub padded: DimsView,
    pub tile: u32,
    pub tile_rows: usize,
    pub tile_cols: usize,
    pub tile_count: usize,
    pub color_fix: bool,
    pub summary: String,
}

impl RestorePlanView {
    pub fn from_plan(plan: &RestorationPlan) -> Self {
        Self {
            engine: sceneworks_core::image_restoration::IRIS_RESTORE_ENGINE,
            model: IRIS_RESTORE_MODEL_ID,
            source: plan.source.into(),
            input_sizing: match plan.input_sizing {
                InputSizing::Budgeted => "budgeted",
                InputSizing::Original => "original",
            },
            input: plan.input.into(),
            budget_applied: plan.budget_applied,
            scale: plan.scale,
            output: plan.output.into(),
            processing: plan.processing.into(),
            enlarged: plan.enlarged,
            padded: plan.padded.into(),
            tile: plan.tile,
            tile_rows: plan.tile_rows.len().max(1),
            tile_cols: plan.tile_cols.len().max(1),
            tile_count: plan.forward_count(),
            color_fix: plan.color_fix,
            summary: plan_summary(plan),
        }
    }
}

/// The pre-submission plan preview for a `width`×`height` source: the engine's pure planner with the
/// released export's tile geometry, no weights. `Err` is the refusal text the UI shows (and the job
/// would fail with) — e.g. a scale whose output exceeds the bounds.
pub fn preview_restore_plan(
    width: u32,
    height: u32,
    params: &IrisRestoreParams,
) -> Result<RestorePlanView, String> {
    preview_restore_plan_with(width, height, params, TileGeometry::RELEASE)
}

/// An export's fixed tile geometry (re-exported for the API's plan endpoint).
pub type PlanGeometry = TileGeometry;

/// [`preview_restore_plan`] with an explicit tile geometry (the selected model's — see
/// [`restore_plan_geometry`]).
pub fn preview_restore_plan_with(
    width: u32,
    height: u32,
    params: &IrisRestoreParams,
    geometry: TileGeometry,
) -> Result<RestorePlanView, String> {
    plan_restore(width, height, params, geometry).map(|plan| RestorePlanView::from_plan(&plan))
}

/// The tile geometry a restore with `trained_model` runs at — exactly what the job plans with: the
/// release export's for `None` (its `config.yaml` is pinned), the TRAINED export's own `config.yaml`
/// for a trained / imported restoration model (whose `task.tile` need not be the release's). `Err`
/// is the refusal the job would fail with (missing model, wrong-task export).
pub fn restore_plan_geometry(
    data_dir: &Path,
    trained_model: Option<&str>,
) -> Result<TileGeometry, String> {
    let Some(model) = trained_model else {
        return Ok(TileGeometry::RELEASE);
    };
    let dir = crate::iris_task_training::trained_task_export_dir(
        data_dir,
        model,
        gen_core_iris::IrisTask::Restoration,
    )
    .map_err(|detail| format!("Iris restoration: {detail}"))?;
    let resources = RestorationResources::from_dir(&dir, IRIS_RESTORE_MODEL_ID)
        .map_err(|error| format!("Iris restoration: {error}"))?;
    Ok(resources.settings.geometry(resources.config()))
}

/// The EXIF-corrected (upright) dimensions of an image file, without decoding its pixels: the
/// header's size with a 90°/270° orientation tag applied. This is the geometry restoration plans
/// against, because the job bakes the orientation into the pixels before handing them over.
/// Falls back to a full (transcoding) decode for a format the native decoders do not read.
pub fn oriented_image_dimensions(path: &Path) -> Option<(u32, u32)> {
    use image::ImageDecoder;
    let native = || -> image::ImageResult<(u32, u32)> {
        let mut decoder = image::ImageReader::open(path)?
            .with_guessed_format()?
            .into_decoder()?;
        let (width, height) = decoder.dimensions();
        let orientation = decoder.orientation()?;
        Ok(if orientation_swaps_axes(orientation) {
            (height, width)
        } else {
            (width, height)
        })
    };
    native().ok().or_else(|| {
        crate::image_decode::decode_image_any(path)
            .ok()
            .map(|image| (image.width(), image.height()))
    })
}

fn orientation_swaps_axes(orientation: image::metadata::Orientation) -> bool {
    use image::metadata::Orientation::*;
    matches!(
        orientation,
        Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH
    )
}

/// Decode a source image with its EXIF orientation baked into the pixels (upstream restoration
/// `exif_transpose`s first). A format the native decoders cannot read goes through the transcoding
/// decode, whose transcoder already bakes the orientation.
pub(crate) fn decode_oriented(path: &Path) -> image::ImageResult<image::DynamicImage> {
    use image::ImageDecoder;
    let native = || -> image::ImageResult<image::DynamicImage> {
        let mut decoder = image::ImageReader::open(path)?
            .with_guessed_format()?
            .into_decoder()?;
        let orientation = decoder.orientation()?;
        let mut image = image::DynamicImage::from_decoder(decoder)?;
        image.apply_orientation(orientation);
        Ok(image)
    };
    native().or_else(|_| crate::image_decode::decode_image_any(path))
}

// ---------------------------------------------------------------------------------------------
// Install resolution
// ---------------------------------------------------------------------------------------------

/// The restoration task's `upscaler/` export dir from the pinned `speridlabs/iris-3b` snapshot the
/// `iris_3b_restore` Model Manager entry installs — **cache-only**: there is no download half, so a
/// restore never fetches mid-job. Typed refusals, before anything is evicted or loaded:
///
/// * nothing installed → "not installed", naming the Model Manager entry;
/// * some export files present but not all → "incomplete", naming the missing files (the generation
///   task's root files never satisfy this — the export lives in its own folder);
/// * a folder whose `config.yaml` is another task's (depth, generation) → the engine's own
///   wrong-task refusal (`RestorationResources::from_dir`), surfaced as user-facing.
///
/// Returns the export dir and its validated resources (whose tile geometry plans the job exactly as
/// the engine will).
pub(crate) fn resolve_restore_export_dir(
    settings: &Settings,
) -> WorkerResult<(PathBuf, RestorationResources)> {
    let snapshot = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        IRIS_RESTORE_REPO,
        IRIS_RESTORE_REVISION,
    );
    let dir = snapshot.map(|snapshot| snapshot.join(restore_subdir()));
    let missing: Vec<&str> = match &dir {
        Some(dir) => RESTORE_EXPORT_FILES
            .iter()
            .copied()
            .filter(|file| !dir.join(file).is_file())
            .collect(),
        None => RESTORE_EXPORT_FILES.to_vec(),
    };
    let dir = match dir {
        Some(dir) if missing.is_empty() => dir,
        Some(_) if missing.len() < RESTORE_EXPORT_FILES.len() => {
            return Err(WorkerError::InvalidPayload(format!(
                "The Iris restoration install is incomplete: {} is missing from the `{}/` export \
                 of {IRIS_RESTORE_REPO}@{IRIS_RESTORE_REVISION}. Repair \"Iris 3B Restoration\" in \
                 the Model Manager.",
                missing.join(", "),
                restore_subdir(),
            )))
        }
        _ => {
            return Err(WorkerError::InvalidPayload(format!(
                "Iris restoration is not installed. Install \"Iris 3B Restoration\" from the Model \
                 Manager (the `{}/` export of {IRIS_RESTORE_REPO}, ~12 GB; it does not need the Iris \
                 text encoder).",
                restore_subdir(),
            )))
        }
    };
    let resources = RestorationResources::from_dir(&dir, IRIS_RESTORE_MODEL_ID)
        .map_err(|error| WorkerError::InvalidPayload(format!("Iris restoration: {error}")))?;
    Ok((dir, resources))
}

// ---------------------------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------------------------

/// The lane a restoration runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RestoreLane {
    /// Apple Silicon unified memory.
    Mlx,
    /// A dedicated NVIDIA card.
    Candle,
}

impl RestoreLane {
    pub(crate) const fn this_build() -> Self {
        if cfg!(target_os = "macos") {
            Self::Mlx
        } else {
            Self::Candle
        }
    }
}

/// DECLARED conservative per-lane floor (GiB) of a restoration up to the reference geometry —
/// `measured: false`, no calibration campaign (S6 owns measured values). The existing real-weight
/// evidence for a 512×384 → 2048×1536 restore (6 tiles) peaked at 11.7 GB on MLX and 10.1 GB
/// (nvidia-smi) on CUDA; the floors round those up by ~20 %. Tiles are 1024 px, so the model working
/// set is per-tile bounded; what grows with the request is the host-side output canvas
/// ([`CANVAS_BYTES_PER_PIXEL`]).
pub(crate) const MLX_FLOOR_GB: f64 = 14.0;
pub(crate) const CANDLE_FLOOR_GB: f64 = 12.0;
/// The reserve each lane's gate keeps free on top of the floor (the legacy unified foreign-resident
/// reserve on MLX, the dedicated-VRAM allocator slack on CUDA — both 2 GiB in this worker).
pub(crate) const MLX_RESERVE_GB: f64 = 2.0;
pub(crate) const CANDLE_RESERVE_GB: f64 = 2.0;
/// Peak host bytes per pixel of the patch-padded processing canvas: the f32 bicubic reference, the
/// padded model input, the fused output and its weight plane during tiling, then the colour fix's
/// per-channel wavelet planes and the restored planes after it — 96 B/px is a conservative bound.
pub(crate) const CANVAS_BYTES_PER_PIXEL: u64 = 96;
/// The processing canvas the floors' evidence covered (2048×1536).
pub(crate) const REFERENCE_CANVAS_PIXELS: u64 = 2048 * 1536;
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Predicted peak (GiB) of `plan` on `lane`. On unified memory the output canvas beyond the
/// reference geometry is added on top of the floor (host buffers draw from the same pool as the
/// model). On a dedicated card the canvas lives in host RAM and the VRAM working set is per-tile
/// bounded, so the floor alone is the VRAM prediction.
pub(crate) fn needed_gb(lane: RestoreLane, plan: &RestorationPlan) -> f64 {
    match lane {
        RestoreLane::Mlx => {
            let canvas = u64::from(plan.padded.width) * u64::from(plan.padded.height);
            let extra = canvas.saturating_sub(REFERENCE_CANVAS_PIXELS) * CANVAS_BYTES_PER_PIXEL;
            MLX_FLOOR_GB + extra as f64 / BYTES_PER_GIB
        }
        RestoreLane::Candle => CANDLE_FLOOR_GB,
    }
}

/// The admission decision for `plan` on `lane` against the host's total memory (unified total on
/// MLX, physical card VRAM on CUDA — this gate establishes a hardware boundary, not a transient
/// working-set threshold, exactly like the SVD gate). `None` admits; no budget signal admits (the
/// worker-wide convention: no invented signal means no refusal). A refusal names what the request
/// needs and the levers that actually move it; it never changes the scale or budget mode itself.
pub(crate) fn admission_error(
    lane: RestoreLane,
    plan: &RestorationPlan,
    total_gb: Option<f64>,
) -> Option<WorkerError> {
    let total_gb = total_gb?;
    let reserve = match lane {
        RestoreLane::Mlx => MLX_RESERVE_GB,
        RestoreLane::Candle => CANDLE_RESERVE_GB,
    };
    let needed = needed_gb(lane, plan);
    if total_gb + f64::EPSILON >= needed + reserve {
        return None;
    }
    let (memory, device) = match lane {
        RestoreLane::Mlx => ("unified memory", "this Mac"),
        RestoreLane::Candle => ("VRAM", "this GPU"),
    };
    let lever = match (lane, plan.input_sizing) {
        (RestoreLane::Mlx, InputSizing::Original) => {
            " Use a smaller scale or the budgeted input to shrink the output canvas, or run on a \
             Mac with more memory."
        }
        (RestoreLane::Mlx, InputSizing::Budgeted) => {
            " Use a smaller scale to shrink the output canvas, or run on a Mac with more memory."
        }
        (RestoreLane::Candle, _) => {
            " Restoration tiles are a fixed 1024 px, so neither the scale nor the input sizing \
             lowers this; run on a card with more VRAM."
        }
    };
    Some(WorkerError::InvalidPayload(format!(
        "Iris restoration ({}) needs ~{:.0} GB of {memory} plus a {reserve:.0} GB reserve, but \
         {device} has ~{total_gb:.0} GB.{lever}",
        plan_summary(plan),
        needed.ceil(),
    )))
}

/// The total memory this worker's lane admits against: the unified total on a Mac and the physical
/// (or capped) VRAM of this worker's card on CUDA. Both honour the existing small-device emulation
/// caps (`SCENEWORKS_MLX_MEMORY_CAP_GB` / `SCENEWORKS_CUDA_VRAM_CAP_GB`).
#[cfg(target_os = "macos")]
pub(crate) async fn live_total_gb(_settings: &Settings) -> Option<f64> {
    let real = crate::mlx_fit_gate::probe_total_unified_memory_bytes()
        .map(|bytes| bytes as f64 / BYTES_PER_GIB);
    crate::mlx_fit_gate::resolve_budget(real, crate::mlx_fit_gate::mlx_memory_cap_gb())
        .map(|budget| budget.total_gb)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn live_total_gb(settings: &Settings) -> Option<f64> {
    crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    )
    .map(|budget| budget.total_gb)
}

#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
pub(crate) async fn live_total_gb(_settings: &Settings) -> Option<f64> {
    None
}

// ---------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------

/// The loader seam: production loads through the registry
/// ([`crate::inference_runtime::load_transform`]); a test injects a stub transform.
pub(crate) type TransformLoader =
    fn(&str, &gen_core::LoadSpec) -> gen_core::Result<Box<dyn gen_core::Transform>>;

/// Production loader.
pub(crate) fn registry_transform_loader(
    id: &str,
    spec: &gen_core::LoadSpec,
) -> gen_core::Result<Box<dyn gen_core::Transform>> {
    crate::inference_runtime::load_transform(id, spec)
}

/// One finished restoration: the restored pixels and the plan that produced them.
#[derive(Debug)]
pub(crate) struct RestoreOutcome {
    pub image: image::RgbImage,
    pub plan: RestorationPlan,
}

/// Restore `source` (already upright) through the cached `iris_3b_restore` transform loaded from
/// `export_dir` by `loader`. `on_tile(current, total)` receives each finished tile from the cache
/// thread (forward it to the job API from async code). A cancel between tiles is
/// [`WorkerError::Canceled`] and returns no image. The returned image is checked against the plan's
/// output geometry, so a disagreement between the preview planner and the engine fails loudly
/// instead of shipping a differently sized result.
pub(crate) async fn restore_image_using(
    loader: TransformLoader,
    export_dir: PathBuf,
    plan: RestorationPlan,
    source: image::RgbImage,
    params: IrisRestoreParams,
    cancel: CancelFlag,
    on_tile: impl FnMut(u32, u32) + Send + 'static,
) -> WorkerResult<RestoreOutcome> {
    let (width, height) = (source.width(), source.height());
    let request = transform_request(
        gen_core::Image {
            width,
            height,
            pixels: source.into_raw(),
        },
        &params,
        cancel,
    );
    let mut on_tile = on_tile;
    let image = crate::transform_cache::with_cached_transform_using(
        IRIS_RESTORE_MODEL_ID,
        gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(export_dir)),
        "Iris restoration load",
        loader,
        move |transform| {
            transform.validate(&request).map_err(|error| {
                crate::classify_engine_error("Iris restoration refused the request", error)
            })?;
            transform
                .apply(&request, &mut |progress| {
                    if let gen_core::Progress::Step { current, total } = progress {
                        on_tile(current, total);
                    }
                })
                .map_err(|error| match error {
                    gen_core::Error::Canceled => {
                        WorkerError::Canceled("Iris restoration canceled by user.".to_owned())
                    }
                    other => crate::classify_engine_error("Iris restoration failed", other),
                })
        },
    )
    .await?;
    if (image.width, image.height) != (plan.output.width, plan.output.height) {
        return Err(WorkerError::Engine(format!(
            "Iris restoration returned {}x{} but the plan promised {}x{}",
            image.width, image.height, plan.output.width, plan.output.height
        )));
    }
    let image = image::RgbImage::from_raw(image.width, image.height, image.pixels)
        .ok_or_else(|| WorkerError::Engine("Iris restoration buffer size mismatch".to_owned()))?;
    Ok(RestoreOutcome { image, plan })
}

/// The applied-settings record a restored asset carries (`rawAdapterSettings.upscale`), which the
/// Image Studio / Editor read back to reopen the exact request. Everything here is the APPLIED value
/// (the resolved scale and colour fix, the plan the engine ran).
///
/// `trained_model` is the trained restoration model the job ran (sc-25688), `None` for the release
/// export: the record names it as `model` and as `irisModel` (the control a reopen selects it by).
pub(crate) fn applied_settings_record(
    plan: &RestorationPlan,
    trained_model: Option<&str>,
) -> serde_json::Value {
    let mut record = serde_json::json!({
        "enabled": true,
        "engine": sceneworks_core::image_restoration::IRIS_RESTORE_ENGINE,
        "model": trained_model.unwrap_or(IRIS_RESTORE_MODEL_ID),
        "scale": plan.scale,
        "inputSizing": match plan.input_sizing {
            InputSizing::Budgeted => "budgeted",
            InputSizing::Original => "original",
        },
        "colorFix": plan.color_fix,
        "sourceWidth": plan.source.width,
        "sourceHeight": plan.source.height,
        "width": plan.output.width,
        "height": plan.output.height,
        "plan": RestorePlanView::from_plan(plan),
    });
    if let Some(model) = trained_model {
        record["irisModel"] = serde_json::json!(model);
    }
    record
}

/// The filename tag of a restored variant. A budget-downscaled restore is NOT `scale×` of the source,
/// so it never carries the `iris{scale}x` tag — it names the budgeted output geometry instead (E6).
pub(crate) fn file_tag(plan: &RestorationPlan) -> String {
    if plan.budget_applied {
        format!("iris_budgeted_{}x{}", plan.output.width, plan.output.height)
    } else {
        format!("iris{}x", format_scale(plan.scale))
    }
}

/// A display-name suffix that cannot overstate the result: a budget-downscaled restore names the
/// budgeted input it scaled.
pub(crate) fn display_suffix(plan: &RestorationPlan) -> String {
    if plan.budget_applied {
        format!(
            "Iris restored, budgeted {}x{} → {}x{}",
            plan.input.width, plan.input.height, plan.output.width, plan.output.height
        )
    } else {
        format!("Iris restored {}×", format_scale(plan.scale))
    }
}

#[cfg(test)]
#[path = "iris_restore/tests.rs"]
pub(crate) mod tests;
