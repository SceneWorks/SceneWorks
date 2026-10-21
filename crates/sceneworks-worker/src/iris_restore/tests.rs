//! Iris restoration (sc-25683): planner-backed preview, install resolution, admission, the shipped
//! catalog closure, and the real job code (`run_iris_restore_job_using` / `restore_with_progress`)
//! driven end to end with an injected stub transform and a mock job API.

use super::*;
use gen_core::Image;
use serde_json::{json, Value};

pub(crate) const UPSCALER_CONFIG: &str = include_str!("fixtures/upscaler_config.yaml");
const DEPTH_CONFIG: &str = include_str!("fixtures/depth_config.yaml");

fn params(
    scale: Option<f64>,
    sizing: RestoreInputSizing,
    color_fix: Option<bool>,
) -> IrisRestoreParams {
    IrisRestoreParams {
        scale,
        input_sizing: sizing,
        color_fix,
    }
}

fn preview(width: u32, height: u32, p: IrisRestoreParams) -> RestorePlanView {
    preview_restore_plan(width, height, &p).expect("plan")
}

// ---------------------------------------------------------------------------------------------
// Planner-backed preview
// ---------------------------------------------------------------------------------------------

/// The reference geometry of the real-weight evidence: 512×384 at the release default (4×, budgeted,
/// colour fix on) → 2048×1536 in 6 tiles; the input is within budget, so it is NOT labelled budgeted.
#[test]
fn default_request_is_4x_budgeted_with_color_fix() {
    let view = preview(512, 384, IrisRestoreParams::default());
    assert_eq!(view.scale, 4.0);
    assert_eq!(view.input_sizing, "budgeted");
    assert!(view.color_fix);
    assert!(!view.budget_applied);
    assert_eq!((view.input.width, view.input.height), (512, 384));
    assert_eq!((view.output.width, view.output.height), (2048, 1536));
    assert_eq!((view.padded.width, view.padded.height), (2048, 1536));
    assert_eq!((view.tile_rows, view.tile_cols, view.tile_count), (2, 3, 6));
    assert_eq!(
        view.summary,
        "4×: 512x384 → 2048x1536 (processed at 2048x1536 in 6 tiles)"
    );
    assert_eq!(view.model, "iris_3b_restore");
    assert_eq!(view.engine, "iris");
}

/// A large photo under the default budget is downscaled first; the preview names the budgeted input
/// and the output it scales to, never "4× of the original".
#[test]
fn a_budget_downscaled_source_is_labelled_honestly() {
    let view = preview(4032, 3024, IrisRestoreParams::default());
    assert!(view.budget_applied);
    assert_eq!((view.source.width, view.source.height), (4032, 3024));
    assert_eq!((view.input.width, view.input.height), (683, 512));
    assert_eq!((view.output.width, view.output.height), (2732, 2048));
    assert!(
        view.summary.starts_with(
            "budgeted: 683x512 input → 2732x2048 (4× of the budgeted input; source 4032x3024"
        ),
        "{}",
        view.summary
    );
    assert!(
        !view.summary.contains("of the original"),
        "{}",
        view.summary
    );
}

/// Original-input mode on the same photo keeps the source size — and at 4× that is over the bound, a
/// typed refusal that names the levers rather than silently falling back to the budget.
#[test]
fn original_input_never_falls_back_to_the_budget() {
    let p = params(Some(4.0), RestoreInputSizing::Original, None);
    let error = preview_restore_plan(4032, 3024, &p).unwrap_err();
    assert!(error.contains("16128x12096"), "{error}");
    assert!(error.contains("budgeted input"), "{error}");
    // Within the bound, original mode plans the source as given.
    let view = preview(
        1500,
        1000,
        params(Some(2.0), RestoreInputSizing::Original, None),
    );
    assert!(!view.budget_applied);
    assert_eq!(view.input_sizing, "original");
    assert_eq!((view.output.width, view.output.height), (3000, 2000));
}

/// 1× is restore-only: same size out, processed on a full tile (enlarged then resized back).
#[test]
fn one_x_restores_at_the_input_size() {
    let view = preview(
        512,
        384,
        params(Some(1.0), RestoreInputSizing::Budgeted, None),
    );
    assert_eq!((view.output.width, view.output.height), (512, 384));
    assert!(view.enlarged);
    assert_eq!(
        (view.processing.width, view.processing.height),
        (1365, 1024)
    );
    assert_eq!(view.tile_count, 2);
    assert!(
        view.summary.starts_with("1× restore: 512x384 → 512x384"),
        "{}",
        view.summary
    );
}

/// A fractional scale on odd sizes floors per side exactly like upstream (`floor(side · scale)` of the
/// shortest decimal the f32 round-trips), in both sizing modes.
#[test]
fn fractional_scales_on_odd_sizes_floor_per_side() {
    let original = preview(
        1001,
        777,
        params(Some(1.3), RestoreInputSizing::Original, None),
    );
    assert_eq!(
        (original.output.width, original.output.height),
        (1301, 1010)
    );
    let budgeted = preview(
        1001,
        777,
        params(Some(1.3), RestoreInputSizing::Budgeted, Some(false)),
    );
    assert!(budgeted.budget_applied);
    assert_eq!((budgeted.input.width, budgeted.input.height), (660, 512));
    assert_eq!((budgeted.output.width, budgeted.output.height), (858, 665));
    assert!(!budgeted.color_fix);
    assert!(
        budgeted.summary.contains("1.3× of the budgeted input"),
        "{}",
        budgeted.summary
    );
}

/// A scale whose output rounds to nothing is refused by the engine's planner — never replaced.
#[test]
fn a_degenerate_scale_is_refused_not_replaced() {
    let error = preview_restore_plan(
        100,
        100,
        &params(Some(0.001), RestoreInputSizing::Budgeted, None),
    )
    .unwrap_err();
    assert!(error.contains("pick a larger scale"), "{error}");
}

#[test]
fn scales_print_as_typed() {
    assert_eq!(format_scale(4.0), "4");
    assert_eq!(format_scale(1.3), "1.3");
    assert_eq!(format_scale(2.25), "2.25");
}

/// The job request carries exactly the three Iris controls: a default request leaves the scale and
/// colour fix to the engine's declared defaults, and nothing sets seed / strength / steps.
#[test]
fn the_transform_request_carries_only_the_iris_controls() {
    let image = Image {
        width: 2,
        height: 2,
        pixels: vec![0; 12],
    };
    let default = transform_request(
        image.clone(),
        &IrisRestoreParams::default(),
        CancelFlag::new(),
    );
    assert_eq!(default.target, TargetSize::ModelDefault);
    assert_eq!(default.input_sizing, InputSizing::Budgeted);
    assert_eq!(default.color_fix, None);
    assert!(default.seed.is_none() && default.strength.is_none() && default.steps.is_none());
    let explicit = transform_request(
        image,
        &params(Some(1.5), RestoreInputSizing::Original, Some(false)),
        CancelFlag::new(),
    );
    assert_eq!(explicit.target, TargetSize::Scale(1.5));
    assert_eq!(explicit.input_sizing, InputSizing::Original);
    assert_eq!(explicit.color_fix, Some(false));
}

// ---------------------------------------------------------------------------------------------
// EXIF orientation
// ---------------------------------------------------------------------------------------------

/// A JPEG carrying an EXIF `Orientation = 6` (Rotate90) APP1 segment.
fn rotated_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut plain = Vec::new();
    image::RgbImage::from_fn(width, height, |x, _| image::Rgb([(x % 255) as u8, 40, 90]))
        .write_to(
            &mut std::io::Cursor::new(&mut plain),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
    // TIFF header (big-endian) + one IFD entry: tag 0x0112 SHORT count 1 value 6.
    let mut tiff = b"MM\x00\x2a\x00\x00\x00\x08".to_vec();
    tiff.extend_from_slice(&[0x00, 0x01]);
    tiff.extend_from_slice(&[
        0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x00,
    ]);
    tiff.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    let mut app1 = b"Exif\x00\x00".to_vec();
    app1.extend_from_slice(&tiff);
    let length = (app1.len() + 2) as u16;
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&length.to_be_bytes());
    jpeg.extend_from_slice(&app1);
    jpeg.extend_from_slice(&plain[2..]);
    jpeg
}

/// The plan geometry and the pixels the job hands over are both the UPRIGHT image (upstream
/// `exif_transpose`s first): a 90°-tagged 40×20 JPEG plans and decodes as 20×40.
#[test]
fn exif_orientation_is_applied_before_planning_and_restoring() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("phone.jpg");
    std::fs::write(&path, rotated_jpeg(40, 20)).unwrap();
    assert_eq!(oriented_image_dimensions(&path), Some((20, 40)));
    let decoded = decode_oriented(&path).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (20, 40));
    // A PNG (no orientation) is unchanged.
    let png = dir.path().join("plain.png");
    image::RgbImage::new(40, 20).save(&png).unwrap();
    assert_eq!(oriented_image_dimensions(&png), Some((40, 20)));
}

// ---------------------------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------------------------

pub(crate) fn plan_for(width: u32, height: u32, p: IrisRestoreParams) -> RestorationPlan {
    plan_restore(width, height, &p, TileGeometry::RELEASE).unwrap()
}

#[test]
fn admission_admits_at_the_floor_and_refuses_below_it() {
    let reference = plan_for(512, 384, IrisRestoreParams::default());
    assert_eq!(needed_gb(RestoreLane::Mlx, &reference), MLX_FLOOR_GB);
    assert!(admission_error(RestoreLane::Mlx, &reference, Some(16.0)).is_none());
    let refusal = admission_error(RestoreLane::Mlx, &reference, Some(15.5)).expect("refused");
    let text = refusal.to_string();
    assert!(
        text.contains("unified memory") && text.contains("~14 GB"),
        "{text}"
    );
    assert!(matches!(refusal, WorkerError::InvalidPayload(_)));

    assert_eq!(needed_gb(RestoreLane::Candle, &reference), CANDLE_FLOOR_GB);
    assert!(admission_error(RestoreLane::Candle, &reference, Some(14.0)).is_none());
    let text = admission_error(RestoreLane::Candle, &reference, Some(12.0))
        .expect("refused")
        .to_string();
    assert!(text.contains("VRAM") && text.contains("1024 px"), "{text}");
    // No budget signal never refuses.
    assert!(admission_error(RestoreLane::Mlx, &reference, None).is_none());
}

/// On unified memory a large original-mode canvas raises the requirement (the host output buffers
/// draw from the same pool); on a dedicated card it does not (the VRAM working set is per tile).
#[test]
fn unified_memory_admission_accounts_for_the_output_canvas() {
    let large = plan_for(
        2000,
        1000,
        params(Some(4.0), RestoreInputSizing::Original, None),
    );
    assert_eq!((large.padded.width, large.padded.height), (8000, 4000));
    let expected = MLX_FLOOR_GB
        + ((8000u64 * 4000 - REFERENCE_CANVAS_PIXELS) * CANVAS_BYTES_PER_PIXEL) as f64
            / (1024.0 * 1024.0 * 1024.0);
    assert!((needed_gb(RestoreLane::Mlx, &large) - expected).abs() < 1e-9);
    assert!(needed_gb(RestoreLane::Mlx, &large) > MLX_FLOOR_GB + 2.0);
    assert!(admission_error(RestoreLane::Mlx, &large, Some(17.0)).is_some());
    let text = admission_error(RestoreLane::Mlx, &large, Some(17.0))
        .unwrap()
        .to_string();
    assert!(text.contains("budgeted input"), "{text}");
    assert_eq!(needed_gb(RestoreLane::Candle, &large), CANDLE_FLOOR_GB);
}

fn shipped_entry(id: &str) -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == id))
        .cloned()
        .unwrap_or_else(|| panic!("{id} is in the builtin catalog"))
}

/// The catalog's declared (unmeasured) floors are the worker's floor + reserve on each lane, so the
/// Model Manager's memory guidance and the job's admission cannot disagree.
#[test]
fn declared_floors_match_the_catalog() {
    let entry = shipped_entry(IRIS_RESTORE_MODEL_ID);
    assert_eq!(entry["candle"]["measured"], json!(false));
    assert_eq!(
        entry["mlx"]["minMemoryGb"].as_f64(),
        Some(MLX_FLOOR_GB + MLX_RESERVE_GB)
    );
    assert_eq!(
        entry["candle"]["minMemoryGb"].as_f64(),
        Some(CANDLE_FLOOR_GB + CANDLE_RESERVE_GB)
    );
}

// ---------------------------------------------------------------------------------------------
// Install closure + resolution
// ---------------------------------------------------------------------------------------------

/// The catalog entry installs EXACTLY the `upscaler/` export the resolver reads — same repo, same
/// pinned revision, the three files and nothing else (no text encoder, no generation backbone).
#[test]
fn manifest_declares_the_restore_closure_the_resolver_reads() {
    let pin = crate::manifest_pins::builtin_model_pin(IRIS_RESTORE_MODEL_ID);
    assert_eq!(pin.repo, IRIS_RESTORE_REPO);
    assert_eq!(pin.revision, IRIS_RESTORE_REVISION);
    assert_eq!(
        IRIS_RESTORE_REVISION,
        "7445443349bc9abe3c96f01ff793e2098ca012b3"
    );
    let mut declared = pin.files.clone();
    declared.sort();
    let mut read: Vec<String> = RESTORE_EXPORT_FILES
        .iter()
        .map(|file| format!("{}/{file}", restore_subdir()))
        .collect();
    read.sort();
    assert_eq!(declared, read);
    let entry = shipped_entry(IRIS_RESTORE_MODEL_ID);
    let downloads = entry["downloads"].as_array().unwrap();
    assert_eq!(
        downloads.len(),
        1,
        "no co-requisite: restoration runs without a text encoder"
    );
    assert_eq!(entry["type"], json!("utility"));
}

pub(crate) fn isolate() -> crate::test_env::EnvVars {
    crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
        ("SCENEWORKS_MLX_MEMORY_CAP_GB", "512"),
        ("SCENEWORKS_CUDA_VRAM_CAP_GB", "512"),
    ])
}

pub(crate) fn snapshot_dir(data_dir: &Path) -> PathBuf {
    sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, IRIS_RESTORE_REPO)
        .expect("repo cache path")
        .join("snapshots")
        .join(IRIS_RESTORE_REVISION)
}

/// Stage the `upscaler/` export the way the `iris_3b_restore` Model Manager install leaves it (the
/// real release `config.yaml`; header-only safetensors weights stamped for the restoration task —
/// the stub transform never reads tensors).
pub(crate) fn stage_restore_install(data_dir: &Path, config: &str, files: &[&str]) -> PathBuf {
    let dir = snapshot_dir(data_dir).join(restore_subdir());
    std::fs::create_dir_all(&dir).unwrap();
    for file in files {
        let bytes = if *file == "config.yaml" {
            config.as_bytes().to_vec()
        } else {
            crate::iris_test_weights::task_weights(gen_core::iris::IrisTask::Restoration)
        };
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    dir
}

fn settings_at(data_dir: &Path) -> Settings {
    Settings::for_test(data_dir.to_path_buf())
}

#[test]
fn resolution_refuses_a_missing_install() {
    let _env = isolate();
    let temp = tempfile::tempdir().unwrap();
    let error =
        resolve_restore_export_dir(&settings_at(temp.path())).expect_err("nothing installed");
    let text = error.to_string();
    assert!(
        text.contains("not installed") && text.contains("Iris 3B Restoration"),
        "{text}"
    );
}

/// The GENERATION task's files at the snapshot root never satisfy restoration: it is its own task.
#[test]
fn the_generation_install_does_not_satisfy_restoration() {
    let _env = isolate();
    let temp = tempfile::tempdir().unwrap();
    let snapshot = snapshot_dir(temp.path());
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("config.yaml"), "model: {}").unwrap();
    std::fs::write(snapshot.join("model.safetensors"), b"gen").unwrap();
    let text = resolve_restore_export_dir(&settings_at(temp.path()))
        .expect_err("generation files are not the restoration export")
        .to_string();
    assert!(text.contains("not installed"), "{text}");
}

#[test]
fn resolution_names_the_missing_files_of_a_torn_install() {
    let _env = isolate();
    let temp = tempfile::tempdir().unwrap();
    stage_restore_install(temp.path(), UPSCALER_CONFIG, &["config.yaml"]);
    let text = resolve_restore_export_dir(&settings_at(temp.path()))
        .expect_err("torn")
        .to_string();
    assert!(text.contains("incomplete"), "{text}");
    assert!(
        text.contains("model.safetensors") && text.contains("empty_prompt.safetensors"),
        "{text}"
    );
    assert!(!text.contains("config.yaml,"), "{text}");
}

/// A `upscaler/` folder holding another task's export (here the depth config) is the engine's typed
/// wrong-task refusal, before anything loads.
#[test]
fn resolution_refuses_a_wrong_task_export() {
    let _env = isolate();
    let temp = tempfile::tempdir().unwrap();
    stage_restore_install(temp.path(), DEPTH_CONFIG, &RESTORE_EXPORT_FILES);
    let error = resolve_restore_export_dir(&settings_at(temp.path())).expect_err("wrong task");
    assert!(matches!(error, WorkerError::InvalidPayload(_)), "{error:?}");
    let text = error.to_string();
    assert!(
        text.contains("'depth' export") && text.contains("restoration"),
        "{text}"
    );
}

#[test]
fn a_complete_install_resolves_with_its_own_tile_geometry() {
    let _env = isolate();
    let temp = tempfile::tempdir().unwrap();
    let dir = stage_restore_install(temp.path(), UPSCALER_CONFIG, &RESTORE_EXPORT_FILES);
    let (resolved, resources) = resolve_restore_export_dir(&settings_at(temp.path())).unwrap();
    assert_eq!(resolved, dir);
    assert_eq!(
        resources.settings.geometry(resources.config()),
        TileGeometry::RELEASE
    );
}

/// S9b: the pre-submission plan for a TRAINED restoration model uses that export's own tile
/// geometry (an imported export's `task.tile` need not be the release's 1024), exactly as the job
/// does — never the release geometry. A missing model is the job's refusal, not a release plan.
#[test]
fn a_trained_model_previews_with_its_own_tile_geometry() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp
        .path()
        .join("models")
        .join("finetunes")
        .join("restore-512")
        .join("restored");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.yaml"),
        UPSCALER_CONFIG.replace("  tile: 1024", "  tile: 512"),
    )
    .unwrap();
    let weights = crate::iris_test_weights::task_weights(gen_core::iris::IrisTask::Restoration);
    for file in ["model.safetensors", "empty_prompt.safetensors"] {
        std::fs::write(dir.join(file), &weights).unwrap();
    }

    let release = restore_plan_geometry(temp.path(), None).unwrap();
    assert_eq!(release, TileGeometry::RELEASE);
    let trained = restore_plan_geometry(temp.path(), Some("restore-512")).unwrap();
    assert_eq!(trained.tile, 512);

    let p = IrisRestoreParams::default();
    let release_view = preview_restore_plan_with(512, 384, &p, release).unwrap();
    let trained_view = preview_restore_plan_with(512, 384, &p, trained).unwrap();
    assert_eq!(release_view.tile, 1024);
    assert_eq!(trained_view.tile, 512);
    assert!(trained_view.tile_count > release_view.tile_count);

    let missing = restore_plan_geometry(temp.path(), Some("absent-model")).unwrap_err();
    assert!(missing.contains("not installed"), "{missing}");
}

/// The linked engine's descriptor matches what the UI and admission assume, on both native lanes
/// (MLX declares `mac_only`, Candle does not).
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
#[test]
fn the_linked_restore_transform_declares_the_planned_capabilities() {
    let descriptor = crate::inference_runtime::transform_descriptor(IRIS_RESTORE_MODEL_ID)
        .expect("iris_3b_restore is registered by the native bundle");
    let caps = &descriptor.capabilities;
    assert_eq!(caps.default_scale, 4.0);
    let budget = caps.input_budget.expect("a declared input budget");
    assert_eq!((budget.short_side, budget.long_side), (512, 1024));
    assert!(caps.supports_color_fix);
    assert_eq!(caps.mac_only, cfg!(target_os = "macos"));
    assert!(!caps.is_diffusion && !caps.supports_strength);
}

// ---------------------------------------------------------------------------------------------
// Real job code with a stub transform + mock job API
// ---------------------------------------------------------------------------------------------

// The job wiring (`upscale_jobs`) compiles only where an upscaler backend does (macOS / Candle), so
// the job-level tests and their fixtures do too.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) use job_tests::*;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod job_tests {
    use super::*;

    /// Stage a TRAINED restoration model the way a Training Studio run leaves it: one complete
    /// `upscaler/`-layout export inside `<data>/models/finetunes/<id>/`.
    pub(crate) fn stage_trained_restore(data_dir: &Path, model_id: &str) {
        let dir = data_dir
            .join("models")
            .join("finetunes")
            .join(model_id)
            .join("restored");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.yaml"), UPSCALER_CONFIG).unwrap();
        let weights = crate::iris_test_weights::task_weights(gen_core::iris::IrisTask::Restoration);
        for file in ["model.safetensors", "empty_prompt.safetensors"] {
            std::fs::write(dir.join(file), &weights).unwrap();
        }
    }
    use gen_core::{Progress, Transform, TransformCapabilities, TransformDescriptor};

    /// A plan-faithful stand-in for the engine: validates and plans with the engine's own
    /// `plan_request`, reports one `Step` per planned tile (checking cancel before each, like the
    /// engine), and returns a flat image of the planned output size.
    struct StubRestorer {
        descriptor: TransformDescriptor,
        cancel_after_first_tile: bool,
    }

    impl Transform for StubRestorer {
        fn descriptor(&self) -> &TransformDescriptor {
            &self.descriptor
        }
        fn validate(&self, req: &TransformRequest) -> gen_core::Result<()> {
            restoration::plan_request(req, TileGeometry::RELEASE, IRIS_RESTORE_MODEL_ID).map(|_| ())
        }
        fn apply(
            &self,
            req: &TransformRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<Image> {
            let plan =
                restoration::plan_request(req, TileGeometry::RELEASE, IRIS_RESTORE_MODEL_ID)?;
            let total = plan.forward_count() as u32;
            for current in 1..=total {
                if req.cancel.is_cancelled() {
                    return Err(gen_core::Error::Canceled);
                }
                on_progress(Progress::Step { current, total });
                if self.cancel_after_first_tile {
                    // A user cancel arriving while tile 1 runs.
                    req.cancel.cancel();
                }
            }
            on_progress(Progress::Decoding);
            let (w, h) = (plan.output.width, plan.output.height);
            Ok(Image {
                width: w,
                height: h,
                pixels: vec![128; (w * h * 3) as usize],
            })
        }
    }

    fn stub_descriptor() -> TransformDescriptor {
        TransformDescriptor {
            id: "iris_3b_restore",
            family: "iris",
            backend: "stub",
            capabilities: TransformCapabilities::default(),
        }
    }

    pub(crate) fn stub_loader(
        id: &str,
        spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn Transform>> {
        assert_eq!(id, IRIS_RESTORE_MODEL_ID);
        // The task resource closure: the export dir alone — no components, no text encoder.
        assert!(spec.components.is_empty() && spec.text_encoder.is_none());
        Ok(Box::new(StubRestorer {
            descriptor: stub_descriptor(),
            cancel_after_first_tile: false,
        }))
    }

    fn canceling_stub_loader(
        _id: &str,
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn Transform>> {
        Ok(Box::new(StubRestorer {
            descriptor: stub_descriptor(),
            cancel_after_first_tile: true,
        }))
    }

    pub(crate) type Posts = std::sync::Arc<std::sync::Mutex<Vec<Value>>>;

    pub(crate) fn job_json(job_id: &str, payload: Value) -> Value {
        json!({
            "id": job_id, "type": "image_upscale", "status": "running",
            "projectId": payload["projectId"], "projectName": null,
            "payload": payload, "result": {}, "requestedGpu": "auto", "assignedGpu": "mlx",
            "workerId": "worker", "progress": 0.1, "stage": "running", "message": "running",
            "error": null, "etaSeconds": null, "elapsedSeconds": null, "attempts": 1,
            "sourceJobId": null, "duplicateOfJobId": null, "cancelRequested": false,
            "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z",
            "startedAt": "2026-10-11T00:00:00Z", "completedAt": null, "canceledAt": null,
            "lastHeartbeatAt": "2026-10-11T00:00:00Z"
        })
    }

    pub(crate) async fn spawn_job_api(job: Value) -> (String, Posts) {
        spawn_job_api_with(job, false).await
    }

    /// `cancel_on_progress`: progress posts answer with `cancelRequested: true` (a user cancel landing
    /// mid-run) while the cancel-peek GET keeps answering `false`.
    pub(crate) async fn spawn_job_api_with(
        job: Value,
        cancel_on_progress: bool,
    ) -> (String, Posts) {
        spawn_job_api_inner(job, cancel_on_progress, false).await
    }

    /// A user cancel landing AFTER the last per-tile progress reply: every progress post answers
    /// `cancelRequested: false`, and the job GET answers `true` only once a post reporting the final
    /// tile (`tile N/N`) has been recorded — so only a re-check after the restore can see it.
    pub(crate) async fn spawn_job_api_cancel_after_last_tile(job: Value) -> (String, Posts) {
        spawn_job_api_inner(job, false, true).await
    }

    pub(crate) fn reports_final_tile(post: &Value) -> bool {
        let Some(message) = post["message"].as_str() else {
            return false;
        };
        let Some(rest) = message.split("tile ").nth(1) else {
            return false;
        };
        let counts: Vec<&str> = rest
            .trim_end_matches(|c: char| !c.is_ascii_digit())
            .split('/')
            .collect();
        counts.len() == 2 && counts[0] == counts[1]
    }

    async fn spawn_job_api_inner(
        job: Value,
        cancel_on_progress: bool,
        cancel_get_after_last_tile: bool,
    ) -> (String, Posts) {
        use axum::extract::{Path as AxumPath, State};
        use axum::routing::{get, post};
        use axum::{Json, Router};
        #[derive(Clone)]
        struct Stub {
            job: Value,
            posts: Posts,
            cancel_on_progress: bool,
            cancel_get_after_last_tile: bool,
        }
        let posts: Posts = Default::default();
        let stub = Stub {
            job,
            posts: posts.clone(),
            cancel_on_progress,
            cancel_get_after_last_tile,
        };
        let app = Router::new()
            .route(
                "/api/v1/jobs/:job_id",
                get(|State(stub): State<Stub>| async move {
                    let mut job = stub.job.clone();
                    if stub.cancel_get_after_last_tile
                        && stub.posts.lock().unwrap().iter().any(reports_final_tile)
                    {
                        job["cancelRequested"] = json!(true);
                    }
                    Json(job)
                }),
            )
            .route(
                "/api/v1/jobs/:job_id/progress",
                post(
                    |State(stub): State<Stub>,
                     AxumPath(_id): AxumPath<String>,
                     Json(body): Json<Value>| async move {
                        stub.posts.lock().unwrap().push(body);
                        let mut job = stub.job.clone();
                        if stub.cancel_on_progress {
                            job["cancelRequested"] = json!(true);
                        }
                        Json(job)
                    },
                ),
            )
            .route(
                "/api/v1/workers/:worker_id/heartbeat",
                post(|AxumPath(worker_id): AxumPath<String>| async move {
                    Json(json!({
                        "id": worker_id, "gpuId": "mlx", "gpuName": null, "status": "busy",
                        "currentJobId": null, "capabilities": [], "loadedModels": [],
                        "registeredAt": "2026-10-11T00:00:00Z", "lastSeenAt": "2026-10-11T00:00:00Z"
                    }))
                }),
            )
            .with_state(stub);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}"), posts)
    }

    struct JobFixture {
        _temp: tempfile::TempDir,
        settings: Settings,
        project_id: String,
        project_path: PathBuf,
        source_id: String,
        source_bytes: Vec<u8>,
    }

    fn job_fixture(source: image::RgbImage) -> JobFixture {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("data");
        stage_restore_install(&data_dir, UPSCALER_CONFIG, &RESTORE_EXPORT_FILES);
        let settings = settings_at(&data_dir);
        let store =
            sceneworks_core::project_store::ProjectStore::new(data_dir.clone(), "iris-test");
        let project = store.create_project("Iris restore").unwrap();
        let file = temp.path().join("source.png");
        source.save(&file).unwrap();
        let asset = store
            .import_asset(
                &project.id,
                sceneworks_core::project_store::UploadAsset {
                    filename: "source.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: file,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .unwrap();
        let project_path = PathBuf::from(&project.path);
        let source_rel = asset["file"]["path"].as_str().unwrap().to_owned();
        let source_bytes = std::fs::read(project_path.join(&source_rel)).unwrap();
        JobFixture {
            _temp: temp,
            settings,
            project_id: project.id,
            project_path,
            source_id: asset["id"].as_str().unwrap().to_owned(),
            source_bytes,
        }
    }

    fn source_file_bytes(fixture: &JobFixture) -> Vec<u8> {
        let store = sceneworks_core::project_store::ProjectStore::new(
            fixture.settings.data_dir.clone(),
            "iris-test",
        );
        let asset = store
            .get_asset(&fixture.project_id, &fixture.source_id)
            .unwrap();
        std::fs::read(
            fixture
                .project_path
                .join(asset["file"]["path"].as_str().unwrap()),
        )
        .unwrap()
    }

    pub(crate) fn png_files_under(dir: &Path) -> Vec<PathBuf> {
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    found.extend(png_files_under(&path));
                } else if path.extension().is_some_and(|ext| ext == "png") {
                    found.push(path);
                }
            }
        }
        found
    }

    /// The standalone `image_upscale` job with `engine: "iris"` runs the REAL job code: install
    /// resolution, planning against the installed export, per-tile progress to the job API, and a NEW
    /// lineage-linked asset at the planned size with the applied settings recorded — the source asset is
    /// untouched.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_standalone_restore_writes_a_new_lineage_linked_asset_with_tile_progress() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::from_pixel(
            512,
            384,
            image::Rgb([200, 30, 30]),
        ));
        let payload = json!({
            "projectId": fixture.project_id,
            "sourceAssetId": fixture.source_id,
            "engine": "iris",
            "scale": 4,
            "inputSizing": "budgeted",
            "colorFix": true,
            "displayName": "Portrait",
        });
        let job_value = job_json("job-iris", payload);
        let (api_url, posts) = spawn_job_api(job_value.clone()).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();

        crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
            .await
            .expect("the restore job completes");

        let posts = posts.lock().unwrap().clone();
        let tiles: Vec<&str> = posts
            .iter()
            .filter_map(|post| post["message"].as_str())
            .filter(|message| message.starts_with("Restoring tile"))
            .collect();
        assert_eq!(
            tiles,
            (1..=6)
                .map(|k| format!("Restoring tile {k}/6 with Iris."))
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            "one progress post per planned tile, in order"
        );
        let completed = posts
            .iter()
            .find(|post| post["status"] == "completed")
            .expect("a completed post");
        let asset = &completed["result"]["assetWrites"][0];
        assert_eq!(
            (asset["width"].as_u64(), asset["height"].as_u64()),
            (Some(2048), Some(1536))
        );
        assert_eq!(asset["model"], json!("iris_3b_restore"));
        assert_eq!(asset["sourceAssetId"], json!(fixture.source_id));
        assert_eq!(asset["parents"], json!([fixture.source_id]));
        assert_eq!(
            asset["extra"]["upscaledFromAssetId"],
            json!(fixture.source_id)
        );
        assert_eq!(asset["displayName"], json!("Portrait (Iris restored 4×)"));
        let record = &asset["rawAdapterSettings"]["upscale"];
        assert_eq!(record["engine"], json!("iris"));
        assert_eq!(record["scale"], json!(4.0));
        assert_eq!(record["inputSizing"], json!("budgeted"));
        assert_eq!(record["colorFix"], json!(true));
        assert_eq!(record["plan"]["tileCount"], json!(6));
        assert_eq!(record["model"], json!("iris_3b_restore"));
        assert!(
            record.get("irisModel").is_none(),
            "the release export names no trained model"
        );
        assert_ne!(asset["assetId"], json!(fixture.source_id));
        // The written file is the planned size, and the source is byte-identical.
        let written = fixture
            .project_path
            .join(asset["mediaPath"].as_str().unwrap());
        let decoded = image::open(&written).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (2048, 1536));
        assert_eq!(source_file_bytes(&fixture), fixture.source_bytes);
    }

    /// A restore with a TRAINED restoration model (sc-25688) is attributed to that model: the asset's
    /// `model`, and the applied record's `model` + `irisModel` (what a reopen selects it by).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_trained_model_restore_records_the_trained_model() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::from_pixel(64, 48, image::Rgb([9, 9, 9])));
        stage_trained_restore(&fixture.settings.data_dir, "finetune_restore1");
        let payload = json!({
            "projectId": fixture.project_id,
            "sourceAssetId": fixture.source_id,
            "engine": "iris",
            "scale": 1,
            "irisModel": "finetune_restore1",
        });
        let job_value = job_json("job-iris-trained", payload);
        let (api_url, posts) = spawn_job_api(job_value.clone()).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();
        crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
            .await
            .unwrap();
        let posts = posts.lock().unwrap().clone();
        let asset = &posts
            .iter()
            .find(|post| post["status"] == "completed")
            .unwrap()["result"]["assetWrites"][0];
        assert_eq!(asset["model"], json!("finetune_restore1"));
        let record = &asset["rawAdapterSettings"]["upscale"];
        assert_eq!(record["model"], json!("finetune_restore1"));
        assert_eq!(record["irisModel"], json!("finetune_restore1"));
    }

    /// A budget-downscaled restore is recorded and named honestly.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_budgeted_restore_is_named_by_its_budgeted_input() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::from_pixel(
            1200,
            900,
            image::Rgb([5, 90, 5]),
        ));
        let payload = json!({
            "projectId": fixture.project_id,
            "sourceAssetId": fixture.source_id,
            "engine": "iris",
            "displayName": "Field",
        });
        let job_value = job_json("job-iris-budget", payload);
        let (api_url, posts) = spawn_job_api(job_value.clone()).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();
        crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
            .await
            .unwrap();
        let posts = posts.lock().unwrap().clone();
        let asset = &posts
            .iter()
            .find(|post| post["status"] == "completed")
            .unwrap()["result"]["assetWrites"][0];
        // 1200x900 → budget (short ≤ 512) → 683x512 → 4× → 2732x2048.
        assert_eq!(
            asset["displayName"],
            json!("Field (Iris restored, budgeted 683x512 → 2732x2048)")
        );
        assert_eq!(
            asset["rawAdapterSettings"]["upscale"]["plan"]["budgetApplied"],
            json!(true)
        );
    }

    /// A cancel between tiles ends the job canceled with NO asset written and the source untouched.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_canceled_restore_publishes_no_asset() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::from_pixel(512, 384, image::Rgb([1, 2, 3])));
        let payload = json!({
            "projectId": fixture.project_id,
            "sourceAssetId": fixture.source_id,
            "engine": "iris",
        });
        let job_value = job_json("job-iris-cancel", payload);
        let (api_url, posts) = spawn_job_api(job_value.clone()).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();
        let before = png_files_under(&fixture.project_path.join("assets"));
        let error = crate::upscale_jobs::run_iris_restore_job_using(
            &api,
            &settings,
            &job,
            canceling_stub_loader,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, WorkerError::Canceled(_)), "{error:?}");
        let posts = posts.lock().unwrap().clone();
        assert!(
            posts.iter().any(|post| post["status"] == "canceled"),
            "{posts:#?}"
        );
        assert!(!posts.iter().any(|post| post["status"] == "completed"));
        assert_eq!(
            png_files_under(&fixture.project_path.join("assets")),
            before
        );
        assert_eq!(source_file_bytes(&fixture), fixture.source_bytes);
    }

    /// A user cancel recorded on the job while it runs is seen on the next per-tile progress answer and
    /// stops the restore before the next tile — not only at the heartbeat's slower cancel poll, which a
    /// warm restore can outrun — so a canceled job never publishes an asset (found by the sc-25683
    /// real-weight E2E, where a warm 6-tile 4x finished inside one heartbeat interval).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_seen_on_a_tile_progress_post_stops_the_restore() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::from_pixel(512, 384, image::Rgb([9, 9, 9])));
        let job_value = job_json(
            "job-iris-tile-cancel",
            json!({ "projectId": fixture.project_id, "sourceAssetId": fixture.source_id, "engine": "iris" }),
        );
        let (api_url, posts) = spawn_job_api_with(job_value.clone(), true).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        // Keep the heartbeat's own cancel poll out of the picture for the whole run.
        settings.heartbeat_seconds = 60;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();
        let before = png_files_under(&fixture.project_path.join("assets"));
        let error =
            crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
                .await
                .unwrap_err();
        assert!(matches!(error, WorkerError::Canceled(_)), "{error:?}");
        let posts = posts.lock().unwrap().clone();
        assert!(
            posts.iter().any(|post| post["status"] == "canceled"),
            "{posts:#?}"
        );
        assert!(!posts.iter().any(|post| post["status"] == "completed"));
        assert_eq!(
            png_files_under(&fixture.project_path.join("assets")),
            before
        );
    }

    /// A cancel requested after the last per-tile progress reply (no flag sees it, the heartbeat poll
    /// is over) is caught by the re-check before the asset write: the job ends canceled, never
    /// completed, and no new PNG lands in the project.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_after_the_last_tile_publishes_no_asset() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::from_pixel(512, 384, image::Rgb([4, 5, 6])));
        let job_value = job_json(
            "job-iris-late-cancel",
            json!({ "projectId": fixture.project_id, "sourceAssetId": fixture.source_id, "engine": "iris" }),
        );
        let (api_url, posts) = spawn_job_api_cancel_after_last_tile(job_value.clone()).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        settings.heartbeat_seconds = 60;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();
        let before = png_files_under(&fixture.project_path.join("assets"));
        let error =
            crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
                .await
                .unwrap_err();
        assert!(matches!(error, WorkerError::Canceled(_)), "{error:?}");
        let posts = posts.lock().unwrap().clone();
        assert!(posts.iter().any(reports_final_tile), "{posts:#?}");
        assert!(!posts.iter().any(|post| post["status"] == "completed"));
        assert_eq!(
            png_files_under(&fixture.project_path.join("assets")),
            before
        );
        assert_eq!(source_file_bytes(&fixture), fixture.source_bytes);
    }

    /// Controls that are not Iris's are refused by name, never ignored; a missing install is the typed
    /// install refusal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn foreign_controls_and_missing_installs_are_typed_refusals() {
        let _env = isolate();
        let fixture = job_fixture(image::RgbImage::new(64, 64));
        for (key, value) in [
            ("factor", json!(4)),
            ("softness", json!(0.5)),
            ("seed", json!(7)),
        ] {
            let mut payload = json!({
                "projectId": fixture.project_id,
                "sourceAssetId": fixture.source_id,
                "engine": "iris",
            });
            payload[key] = value;
            let job_value = job_json("job-iris-refuse", payload);
            let (api_url, _posts) = spawn_job_api(job_value.clone()).await;
            let mut settings = fixture.settings.clone();
            settings.api_url = api_url;
            let api = crate::ApiClient::new(&settings);
            let job: sceneworks_core::contracts::JobSnapshot =
                serde_json::from_value(job_value).unwrap();
            let error =
                crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
                    .await
                    .unwrap_err();
            assert!(
                matches!(&error, WorkerError::InvalidPayload(text) if text.contains(key)),
                "{key}: {error:?}"
            );
        }
        // Remove the install: the job refuses with the install error, not an engine failure.
        std::fs::remove_dir_all(snapshot_dir(&fixture.settings.data_dir)).unwrap();
        let job_value = job_json(
            "job-iris-missing",
            json!({ "projectId": fixture.project_id, "sourceAssetId": fixture.source_id, "engine": "iris" }),
        );
        let (api_url, _posts) = spawn_job_api(job_value.clone()).await;
        let mut settings = fixture.settings.clone();
        settings.api_url = api_url;
        let api = crate::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(job_value).unwrap();
        let error =
            crate::upscale_jobs::run_iris_restore_job_using(&api, &settings, &job, stub_loader)
                .await
                .unwrap_err();
        assert!(error.to_string().contains("not installed"), "{error:?}");
    }

    /// Admission refuses BEFORE loading anything (the stub loader would panic if reached) and never
    /// changes the request: the emulated small device names what the request needs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admission_refuses_before_load_on_a_small_device() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("data");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
            ("SCENEWORKS_MLX_MEMORY_CAP_GB", "8"),
            ("SCENEWORKS_CUDA_VRAM_CAP_GB", "8"),
        ]);
        stage_restore_install(&data_dir, UPSCALER_CONFIG, &RESTORE_EXPORT_FILES);
        let settings = settings_at(&data_dir);
        fn never_loads(_: &str, _: &gen_core::LoadSpec) -> gen_core::Result<Box<dyn Transform>> {
            panic!("an over-budget restore must be refused before loading");
        }
        let api = crate::ApiClient::new(&settings);
        let error = crate::upscale_jobs::restore_with_progress(
            &api,
            &settings,
            "job-small",
            never_loads,
            image::RgbImage::new(512, 384),
            IrisRestoreParams::default(),
            None,
            |_, _| unreachable!(),
        )
        .await
        .expect_err("refused");
        let text = error.to_string();
        assert!(text.contains("needs ~") && text.contains("~8 GB"), "{text}");
    }
}
