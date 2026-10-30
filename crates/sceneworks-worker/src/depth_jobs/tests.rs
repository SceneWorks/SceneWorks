//! sc-25682 — the `image_depth` job end to end with an injected depth estimator.
//!
//! `run_image_depth_job_using` is the production job body (the plain name passes the backend
//! loader): payload contract, cache-only export resolution, source decode with EXIF orientation,
//! admission, the blocking estimate under the heartbeat/cancel helper, the NPY + preview writes and
//! the asset fact all run for real. Only the estimator is a fake, so these pin what the worker does
//! with its output, a cancel and an engine failure — not a re-implementation.

use super::*;
use crate::api_client::ApiClient;
use axum::extract::{Path as AxumPath, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use gen_core::Progress;
use gen_core_iris::depth::{DepthMap, DepthOutput};
use image::GenericImageView;
use sceneworks_core::depth_map::decode_npy_f32;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The inference repo's tiny depth-export config (`mlx-gen-iris/tests/fixtures/tiny-depth`): a
/// valid, supported `task: depth` config. The weights files are staged as placeholders — the
/// estimator is a fake, and the worker never reads them.
const TINY_DEPTH_CONFIG: &str = "model:
  block: single_stream
  dual_depth: 2
  final_block_text: keep
  hidden_size: 64
  depth: 3
  num_heads: 4
  num_kv_heads: 2
  gated_attention: true
  sandwich_norm: true
  patch_size: 4
  in_channels: 3
  mlp_ratio: 4.0
  qkv_bias: false
  qk_norm: true
  norm_eps: 1.0e-06
  attn_backend: sdpa
  modulation: shared_bias
  modulation_rank: 64
  timestep_max_period: 10.0
  adaln_zero_init: true
  rope_theta: 10000.0
  rope_scale: 16.0
  rope_aspect: isotropic
  rope_frame_pairs: 0
  rope_frame_theta: 10.0
  text_rope: true
  text_rope_theta: 10000.0
  text_abs_pos_embed: true
  text_dim: 32
  text_len: 10
  text_adapter: lap_blocks2
  text_lap_num_layers: 12
  text_lap_num_heads: 4
  text_lap_mlp_ratio: 1.3
  repa_layer: 0
  pixel:
    enabled: true
    depth: 2
    hidden_size: 8
    attn_hidden_size: 32
    num_heads: 2
    mlp_ratio: 4.0
    modulation: post
    abs_pos_embed: true
text_encoder:
  name: qwen3_vl
  pretrained: tiny-snapshot/text_encoder
  dim: 32
  max_length: 10
  dtype: bfloat16
  attn_implementation: sdpa
  hidden_layers:
  - 2
  - 5
  - 8
  - 11
  - 14
  - 17
  - 20
  - 23
  - 26
  - 29
  - 32
  - 35
  compile: false
  on_caption_overflow: warn
flow:
  num_train_timesteps: 1000
  shift: 4.0
  timestep_sampler: logit_normal
  logit_mean: 0.0
  logit_std: 1.0
  prediction: v
  x_pred_sigma_min: 0.05
  shift_law: none
  shift_base_tokens: 256
task:
  name: depth
";

fn settings_at(data_dir: &Path) -> Settings {
    let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
    settings.data_dir = data_dir.to_path_buf();
    settings
}

fn export_dir(data_dir: &Path) -> PathBuf {
    sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, UPSTREAM_WEIGHTS_REPO)
        .unwrap()
        .join("snapshots")
        .join(UPSTREAM_WEIGHTS_REVISION)
        .join(DEPTH_EXPORT_SUBDIR)
}

/// Stage `files` of the depth export; `config.yaml` gets `config` (the tiny depth config by
/// default), the weights get a header-only safetensors file stamped for the depth task.
fn stage_export(data_dir: &Path, files: &[&str], config: &str) {
    let dir = export_dir(data_dir);
    std::fs::create_dir_all(&dir).unwrap();
    for file in files {
        let bytes = if *file == BACKBONE_CONFIG_FILE {
            config.as_bytes().to_vec()
        } else {
            crate::iris_test_weights::task_weights(IrisTask::Depth)
        };
        std::fs::write(dir.join(file), bytes).unwrap();
    }
}

fn isolate_hf_cache() -> crate::test_env::EnvVars {
    crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ])
}

/// The shipped `iris_3b_depth` catalog entry, so admission reads the real declaration.
fn shipped_depth_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == DEPTH_MODEL_ID)
        .cloned()
        .expect("iris_3b_depth is in the builtin catalog")
}

// ---- EXIF orientation ---------------------------------------------------------------------------

/// A minimal little-endian TIFF/EXIF block holding one Orientation (0x0112) entry.
fn exif_orientation_block(orientation: u16) -> Vec<u8> {
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II\x2a\x00");
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&1u16.to_le_bytes());
    tiff.extend_from_slice(&0x0112u16.to_le_bytes());
    tiff.extend_from_slice(&3u16.to_le_bytes());
    tiff.extend_from_slice(&1u32.to_le_bytes());
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0]);
    tiff.extend_from_slice(&0u32.to_le_bytes());
    tiff
}

/// A 3x2 image whose every pixel is distinct: pixel (x, y) = [x, y, 7].
fn tagged_source() -> image::RgbImage {
    image::RgbImage::from_fn(3, 2, |x, y| image::Rgb([x as u8 * 40, y as u8 * 90, 7]))
}

fn write_png_with_orientation(path: &Path, image: &image::RgbImage, orientation: Option<u16>) {
    use image::ImageEncoder;
    let mut bytes = Vec::new();
    let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
    if let Some(orientation) = orientation {
        encoder
            .set_exif_metadata(exif_orientation_block(orientation))
            .unwrap();
    }
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// The upright image an EXIF viewer shows for `orientation`, built independently of the `image`
/// crate's `apply_orientation`: the stored pixel that lands at upright (u, v).
fn expected_upright(stored: &image::RgbImage, orientation: u16) -> image::RgbImage {
    let (w, h) = (stored.width(), stored.height());
    let swaps = orientation >= 5;
    let (ow, oh) = if swaps { (h, w) } else { (w, h) };
    image::RgbImage::from_fn(ow, oh, |u, v| {
        let (x, y) = match orientation {
            1 => (u, v),
            2 => (w - 1 - u, v),
            3 => (w - 1 - u, h - 1 - v),
            4 => (u, h - 1 - v),
            5 => (v, u),
            6 => (v, h - 1 - u),
            7 => (w - 1 - v, h - 1 - u),
            8 => (w - 1 - v, u),
            _ => unreachable!(),
        };
        *stored.get_pixel(x, y)
    })
}

#[test]
fn every_exif_orientation_is_applied_before_estimation() {
    let dir = tempfile::tempdir().unwrap();
    let stored = tagged_source();
    for orientation in 1..=8u16 {
        let path = dir.path().join(format!("o{orientation}.png"));
        write_png_with_orientation(&path, &stored, Some(orientation));
        let decoded = decode_oriented_rgb8(&path).unwrap();
        let expected = expected_upright(&stored, orientation);
        assert_eq!(
            decoded.image.dimensions(),
            expected.dimensions(),
            "orientation {orientation} dims"
        );
        assert_eq!(
            decoded.image.as_raw(),
            expected.as_raw(),
            "orientation {orientation} pixels"
        );
        assert_eq!(decoded.geometry.exif_orientation, orientation as u8);
        assert_eq!(
            (
                decoded.geometry.stored_width,
                decoded.geometry.stored_height
            ),
            (3, 2)
        );
        assert_eq!(
            (decoded.geometry.width, decoded.geometry.height),
            expected.dimensions()
        );
    }
    // No tag at all is orientation 1.
    let path = dir.path().join("plain.png");
    write_png_with_orientation(&path, &stored, None);
    let decoded = decode_oriented_rgb8(&path).unwrap();
    assert_eq!(decoded.geometry.exif_orientation, 1);
    assert_eq!(decoded.image.as_raw(), stored.as_raw());
}

/// `expected_upright` — the reference every orientation test here compares against — is PIL's
/// `ImageOps.exif_transpose`, the orientation step upstream's `DepthPredictor.__call__` applies
/// before estimating. The table is Pillow 12.3.0's output for [`tagged_source`] saved as a PNG with
/// each EXIF Orientation value (`(orientation, width, height, RGB8 row-major)`), so the upright
/// reference is the frozen upstream behaviour rather than this file's own reading of the EXIF spec.
#[test]
fn the_upright_reference_is_pils_exif_transpose() {
    #[rustfmt::skip]
    const PIL: [(u16, u32, u32, &[u8]); 8] = [
        (1, 3, 2, &[0, 0, 7, 40, 0, 7, 80, 0, 7, 0, 90, 7, 40, 90, 7, 80, 90, 7]),
        (2, 3, 2, &[80, 0, 7, 40, 0, 7, 0, 0, 7, 80, 90, 7, 40, 90, 7, 0, 90, 7]),
        (3, 3, 2, &[80, 90, 7, 40, 90, 7, 0, 90, 7, 80, 0, 7, 40, 0, 7, 0, 0, 7]),
        (4, 3, 2, &[0, 90, 7, 40, 90, 7, 80, 90, 7, 0, 0, 7, 40, 0, 7, 80, 0, 7]),
        (5, 2, 3, &[0, 0, 7, 0, 90, 7, 40, 0, 7, 40, 90, 7, 80, 0, 7, 80, 90, 7]),
        (6, 2, 3, &[0, 90, 7, 0, 0, 7, 40, 90, 7, 40, 0, 7, 80, 90, 7, 80, 0, 7]),
        (7, 2, 3, &[80, 90, 7, 80, 0, 7, 40, 90, 7, 40, 0, 7, 0, 90, 7, 0, 0, 7]),
        (8, 2, 3, &[80, 0, 7, 80, 90, 7, 40, 0, 7, 40, 90, 7, 0, 0, 7, 0, 90, 7]),
    ];
    let stored = tagged_source();
    for (orientation, width, height, pixels) in PIL {
        let upright = expected_upright(&stored, orientation);
        assert_eq!(
            upright.dimensions(),
            (width, height),
            "orientation {orientation}"
        );
        assert_eq!(
            upright.as_raw().as_slice(),
            pixels,
            "orientation {orientation}"
        );
    }
}

/// A camera JPEG carries the tag in APP1; the same decode path reads it there.
#[test]
fn a_rotated_jpeg_is_estimated_upright() {
    use image::ImageEncoder;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("photo.jpg");
    let stored = image::RgbImage::from_pixel(8, 4, image::Rgb([120, 80, 40]));
    let mut bytes = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 95);
    encoder
        .set_exif_metadata(exif_orientation_block(6))
        .unwrap();
    encoder
        .write_image(stored.as_raw(), 8, 4, image::ExtendedColorType::Rgb8)
        .unwrap();
    std::fs::write(&path, bytes).unwrap();
    let decoded = decode_oriented_rgb8(&path).unwrap();
    assert_eq!(decoded.image.dimensions(), (4, 8), "rotated 90° upright");
    assert_eq!(decoded.geometry.exif_orientation, 6);
}

// ---- install resolution --------------------------------------------------------------------------

#[test]
fn the_export_resolves_only_from_its_own_complete_depth_closure() {
    let _env = isolate_hf_cache();
    let data = tempfile::tempdir().unwrap();
    let settings = settings_at(data.path());

    let error = resolve_depth_export(&settings).err().unwrap();
    assert!(
        error.to_string().starts_with("[iris_depth_not_installed]"),
        "{error}"
    );

    // The GENERATION backbone's root files are not a depth install.
    let root = export_dir(data.path()).parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("config.yaml"), "model: {}\n").unwrap();
    std::fs::write(root.join("model.safetensors"), b"x").unwrap();
    let error = resolve_depth_export(&settings).err().unwrap();
    assert!(
        error.to_string().starts_with("[iris_depth_not_installed]"),
        "{error}"
    );

    stage_export(
        data.path(),
        &[BACKBONE_CONFIG_FILE, BACKBONE_WEIGHTS_FILE],
        TINY_DEPTH_CONFIG,
    );
    let error = resolve_depth_export(&settings).err().unwrap();
    let text = error.to_string();
    assert!(
        text.starts_with("[iris_depth_incomplete_install]"),
        "{text}"
    );
    assert!(text.contains("empty_prompt.safetensors"), "{text}");

    stage_export(data.path(), &DEPTH_EXPORT_FILES, TINY_DEPTH_CONFIG);
    let export = resolve_depth_export(&settings).unwrap();
    assert_eq!(export.dir, export_dir(data.path()));
    assert_eq!(export.task, IrisTask::Depth);
    assert_eq!(export.config_sha256.len(), 64);
    // The load spec stages the export ALONE: no component, no text encoder.
    let spec = depth_load_spec(&export.dir);
    assert!(spec.components.is_empty() && spec.text_encoder.is_none());
    assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == &export.dir));
    assert_eq!(spec.precision, gen_core::Precision::Bf16);

    // A restoration export sitting where the depth export belongs is a wrong-task install.
    stage_export(
        data.path(),
        &DEPTH_EXPORT_FILES,
        &TINY_DEPTH_CONFIG.replace("name: depth", "name: restoration"),
    );
    let error = resolve_depth_export(&settings).err().unwrap();
    assert!(
        error.to_string().starts_with("[iris_depth_wrong_task]"),
        "{error}"
    );
}

// ---- admission -----------------------------------------------------------------------------------

#[test]
fn admission_scales_with_the_model_input_and_refuses_over_budget() {
    let entry = shipped_depth_entry();
    let payload = json!({ "modelManifestEntries": [entry] })
        .as_object()
        .cloned()
        .unwrap();
    for lane in ["mlx", "candle"] {
        let declaration = DepthAdmission::from_payload(&payload, lane).unwrap();
        // At the reference area the prediction is the recorded peak.
        let reference = declaration.reference_model_pixels as u64;
        let at_reference = declaration.predicted_peak_gb(reference);
        assert!(
            (at_reference - declaration.reference_peak_gb).abs() < 1e-9,
            "{lane}"
        );
        // It grows with the pixel count and is padded above the reference.
        let quadruple = declaration.predicted_peak_gb(reference * 4);
        let linear = declaration.resident_gb
            + (declaration.reference_peak_gb - declaration.resident_gb) * 4.0;
        assert!(quadruple > linear, "{lane}: extrapolation is padded");
    }
    let mlx = DepthAdmission::from_payload(&payload, "mlx").unwrap();
    assert_eq!(mlx.reference_peak_gb, 9.4);
    let candle = DepthAdmission::from_payload(&payload, "candle").unwrap();
    assert_eq!(candle.reference_peak_gb, 8.4);

    let plan = |w: u32, h: u32| plan_depth_input(w, h, DepthResolution::Native, 16).unwrap();
    // A 4000x3000 photo at NATIVE resolution is ~12 MP of model input: refused on a 64 GB budget,
    // naming the alternative; never downscaled.
    let error = admission_decision(&mlx, &plan(4000, 3000), Some(64.0)).unwrap_err();
    let text = error.to_string();
    assert!(text.starts_with("[iris_depth_memory_admission]"), "{text}");
    assert!(text.contains("smaller max side"), "{text}");
    // The default cap of the same photo fits.
    let capped = plan_depth_input(4000, 3000, DepthResolution::default(), 16).unwrap();
    admission_decision(&mlx, &capped, Some(64.0)).unwrap();
    // No budget is never "admit unbounded".
    assert!(admission_decision(&mlx, &capped, None).is_err());
    // No declaration is never "admit unbounded" either.
    assert!(DepthAdmission::from_payload(&JsonObject::new(), "mlx").is_err());
}

// ---- the job body --------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Behavior {
    Succeeds,
    /// Wait for the request's cancel flag, then return `Canceled` like the engine does.
    WaitsForCancel,
    /// Wait for the cancel flag, then return a finished map anyway (it had passed its last cancel
    /// checkpoint) — so the job has already written its staged files when the cancel lands.
    FinishesDespiteCancel,
    Fails,
    /// Predict [`pixel_depth`]: a depth that is a function of the pixels the job handed over, so
    /// the published map says which pixel the estimator saw at each position.
    PixelDepth,
    /// Return a map with no finite value (a numerically diverged estimate).
    NoFiniteDepth,
}

/// The fixture "reference model" of the orientation tests: one depth per pixel, a function of that
/// pixel's colour alone and injective on [`job_source`]'s pixels — so `pixel_depth(image)[i]`
/// identifies the pixel at position `i`, and a map equal to `pixel_depth(upright)` proves the
/// estimator was handed exactly the upright pixels in upright order.
fn pixel_depth(rgb: &[u8]) -> Vec<f32> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|p| p[0] as f32 / 255.0 + p[1] as f32 * 7.0 / 255.0 - 1.3)
        .collect()
}

/// The stored pixel grid of the job fixture's source asset (6x4, every pixel distinct).
fn job_source() -> image::RgbImage {
    image::RgbImage::from_fn(6, 4, |x, y| image::Rgb([x as u8 * 30, y as u8 * 50, 9]))
}

struct FakeEstimator {
    behavior: Behavior,
    seen: Arc<Mutex<Option<(u32, u32, DepthResolution)>>>,
}

/// The map the fake predicts: a deterministic ramp with values outside [-1, 1] (not clamped).
fn fake_values(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|i| (i as f32) * 0.37 - 1.6)
        .collect()
}

impl IrisDepthEstimator for FakeEstimator {
    fn backend(&self) -> &'static str {
        "fake"
    }

    fn estimate(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DepthOutput> {
        let (width, height) = (req.image.width, req.image.height);
        *self.seen.lock().unwrap() = Some((width, height, req.resolution));
        match self.behavior {
            Behavior::Fails => {
                return Err(gen_core::Error::Msg("fixture engine failure".to_owned()))
            }
            Behavior::WaitsForCancel => {
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                return Err(gen_core::Error::Canceled);
            }
            Behavior::FinishesDespiteCancel => {
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            Behavior::Succeeds | Behavior::PixelDepth | Behavior::NoFiniteDepth => {}
        }
        on_progress(Progress::Step {
            current: 1,
            total: 1,
        });
        let plan = plan_depth_input(width, height, req.resolution, 4)?;
        let values = match self.behavior {
            Behavior::PixelDepth => pixel_depth(&req.image.pixels),
            Behavior::NoFiniteDepth => (0..width * height)
                .map(|i| [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][i as usize % 3])
                .collect(),
            _ => fake_values(width, height),
        };
        Ok(DepthOutput {
            map: DepthMap {
                width,
                height,
                values,
            },
            metadata: DepthMetadata::new("fake", "c".repeat(64), "bfloat16", req.resolution, &plan),
        })
    }
}

type Posts = Arc<Mutex<Vec<Value>>>;

/// `reject_completion`: the progress route answers the `completed` post with a 500, as an API that
/// failed to persist the asset would.
async fn spawn_job_api(cancel_requested: bool, reject_completion: bool) -> (String, Posts) {
    type ApiState = (Posts, bool, bool);
    async fn job_route(
        State((_, cancel, _)): State<ApiState>,
        AxumPath(job_id): AxumPath<String>,
    ) -> Response {
        Json(crate::tests::job_snapshot_json(&job_id, cancel)).into_response()
    }
    async fn progress_route(
        State((posts, cancel, reject_completion)): State<ApiState>,
        AxumPath(job_id): AxumPath<String>,
        Json(body): Json<Value>,
    ) -> Response {
        let completing = body["status"] == "completed";
        posts.lock().unwrap().push(body);
        if reject_completion && completing {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "asset persist failed",
            )
                .into_response();
        }
        Json(crate::tests::job_snapshot_json(&job_id, cancel)).into_response()
    }
    async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
        Json(json!({
            "id": worker_id, "gpuId": "cpu", "gpuName": null, "status": "busy",
            "currentJobId": "job-depth", "capabilities": [], "loadedModels": [],
            "registeredAt": "2026-10-11T00:00:00Z", "lastSeenAt": "2026-10-11T00:00:00Z"
        }))
        .into_response()
    }
    let posts: Posts = Arc::default();
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job_route))
        .route("/api/v1/jobs/:job_id/progress", post(progress_route))
        .route(
            "/api/v1/workers/:worker_id/heartbeat",
            post(heartbeat_route),
        )
        .with_state((posts.clone(), cancel_requested, reject_completion));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), posts)
}

/// A model resident in a process-global cache when the depth job starts (sc-25684 task switching).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Resident {
    /// A warm generator in the generator cache (the Iris generator carries its text encoder there).
    Generator,
    /// A resident restoration transform in the transform cache.
    Transform,
}

struct JobRun {
    outcome: WorkerResult<()>,
    /// Whether a generator / a transform was still resident in the global caches after the job.
    resident_after: (bool, bool),
    posts: Vec<Value>,
    load_calls: u32,
    seen: Option<(u32, u32, DepthResolution)>,
    project_path: PathBuf,
    project_id: String,
    data_dir: tempfile::TempDir,
    _project_dir: tempfile::TempDir,
}

impl JobRun {
    fn completed_fact(&self) -> &Value {
        let last = self.posts.last().expect("posts");
        assert_eq!(last["status"], "completed", "{:#?}", self.posts);
        &last["result"]["assetWrites"][0]
    }

    fn depth_files(&self) -> Vec<PathBuf> {
        let mut found = Vec::new();
        let mut stack = vec![self.project_path.join("assets").join("depth")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    found.push(path);
                }
            }
        }
        found
    }
}

struct JobSetup {
    behavior: Behavior,
    cancel_requested: bool,
    reject_completion: bool,
    stage: &'static [&'static str],
    memory_cap_gb: &'static str,
    orientation: Option<u16>,
    resolution: Value,
    resident: Option<Resident>,
    /// Run a TRAINED depth model of this id (staged under `models/finetunes/<id>/depth`).
    trained: Option<&'static str>,
}

impl Default for JobSetup {
    fn default() -> Self {
        Self {
            behavior: Behavior::Succeeds,
            cancel_requested: false,
            reject_completion: false,
            stage: &DEPTH_EXPORT_FILES,
            memory_cap_gb: "64",
            orientation: Some(6),
            resolution: json!({ "mode": "capped", "maxSide": 1024 }),
            resident: None,
            trained: None,
        }
    }
}

fn run_job(setup: JobSetup) -> JobRun {
    let _env = crate::test_env::EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
        #[cfg(target_os = "macos")]
        (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, setup.memory_cap_gb),
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        (crate::vram_gate::CUDA_VRAM_CAP_ENV, setup.memory_cap_gb),
    ]);
    let data = tempfile::tempdir().unwrap();
    let project_dir = tempfile::tempdir().unwrap();
    stage_export(data.path(), setup.stage, TINY_DEPTH_CONFIG);
    if let Some(model) = setup.trained {
        let dir = data
            .path()
            .join("models")
            .join("finetunes")
            .join(model)
            .join("depth");
        std::fs::create_dir_all(&dir).unwrap();
        for file in DEPTH_EXPORT_FILES {
            let bytes = if file == BACKBONE_CONFIG_FILE {
                TINY_DEPTH_CONFIG.as_bytes().to_vec()
            } else {
                crate::iris_test_weights::task_weights(IrisTask::Depth)
            };
            std::fs::write(dir.join(file), bytes).unwrap();
        }
    }

    // A real project with a real (EXIF-rotated) source image asset.
    let store = ProjectStore::new(data.path().to_path_buf(), "test");
    let project = store.create_project("Depth job").unwrap();
    let project_path = PathBuf::from(&project.path);
    let source_rel = "assets/images/genset_src/source.png";
    std::fs::create_dir_all(project_path.join("assets/images/genset_src")).unwrap();
    let stored = job_source();
    write_png_with_orientation(&project_path.join(source_rel), &stored, setup.orientation);
    store
        .persist_generated_asset(
            &project.id,
            "job-src",
            "genset_src",
            &json!({
                "assetId": "asset_source1", "mediaPath": source_rel, "mimeType": "image/png",
                "width": 6, "height": 4, "displayName": "Harbor", "mode": "text_to_image",
                "createdAt": "2026-10-11T00:00:00Z",
                "model": "iris_3b", "adapter": "mlx_iris", "prompt": "harbor", "loras": [],
            }),
        )
        .unwrap();

    let loads = Arc::new(Mutex::new(0u32));
    let seen = Arc::new(Mutex::new(None));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let project_id = project.id.clone();
    let resident_weights = tempfile::tempdir().unwrap();
    let (outcome, posts, resident_after) = runtime.block_on(async {
        match setup.resident {
            Some(Resident::Generator) => {
                crate::generator_cache::load_stub_into_global_cache(resident_weights.path())
                    .await
                    .unwrap();
                assert!(crate::generator_cache::global_generator_resident()
                    .await
                    .is_some());
            }
            Some(Resident::Transform) => {
                crate::transform_cache::load_stub_into_global_cache(resident_weights.path())
                    .await
                    .unwrap();
                assert!(crate::transform_cache::global_transform_resident().await);
            }
            None => {}
        }
        let (base_url, posts) =
            spawn_job_api(setup.cancel_requested, setup.reject_completion).await;
        let mut settings = settings_at(data.path());
        settings.api_url = base_url;
        settings.heartbeat_seconds = 5;
        let api = ApiClient::new(&settings);
        let mut job: JobSnapshot =
            serde_json::from_value(crate::tests::job_snapshot_json("job-depth", false)).unwrap();
        job.payload = json!({
            "projectId": project_id,
            "sourceAssetId": "asset_source1",
            "resolution": setup.resolution,
            "modelManifestEntries": [shipped_depth_entry()],
        })
        .as_object()
        .cloned()
        .unwrap();
        if let Some(model) = setup.trained {
            // The API attaches the trained model's OWN catalog entry (its engine's depth
            // declaration overlaid) in place of the release entry (S6).
            let mut trained_entry = shipped_depth_entry();
            trained_entry["id"] = json!(model);
            job.payload.insert("irisModel".to_owned(), json!(model));
            job.payload
                .insert("modelManifestEntries".to_owned(), json!([trained_entry]));
        }
        let loader_loads = Arc::clone(&loads);
        let loader_seen = Arc::clone(&seen);
        let behavior = setup.behavior;
        let loader = move |spec: &LoadSpec| -> gen_core::Result<Box<dyn IrisDepthEstimator>> {
            *loader_loads.lock().unwrap() += 1;
            assert!(spec.components.is_empty() && spec.text_encoder.is_none());
            Ok(Box::new(FakeEstimator {
                behavior,
                seen: loader_seen,
            }))
        };
        let outcome = tokio::time::timeout(
            Duration::from_secs(60),
            run_image_depth_job_using(&api, &settings, &job, loader),
        )
        .await
        .expect("the depth job finished");
        let posts = posts.lock().unwrap().clone();
        let resident_after = (
            crate::generator_cache::global_generator_resident()
                .await
                .is_some(),
            crate::transform_cache::global_transform_resident().await,
        );
        (outcome, posts, resident_after)
    });
    let load_calls = *loads.lock().unwrap();
    let seen = *seen.lock().unwrap();
    JobRun {
        outcome,
        resident_after,
        posts,
        load_calls,
        seen,
        project_path,
        project_id: project.id,
        data_dir: data,
        _project_dir: project_dir,
    }
}

/// S5: a TRAINED depth model's run records the trained identity everywhere lineage is read — the
/// asset, the job result and its generation set name the trained id, and the depth model block names
/// the training export (no release repo/revision) — and it runs with NO release install staged.
#[test]
fn a_trained_depth_run_records_the_trained_identity_not_the_release_weights() {
    let run = run_job(JobSetup {
        stage: &[],
        trained: Some("my_depth_run"),
        ..JobSetup::default()
    });
    run.outcome
        .as_ref()
        .expect("the trained depth job succeeds");
    let last = run.posts.last().unwrap();
    assert_eq!(last["result"]["model"], "my_depth_run", "{last:#}");
    assert_eq!(last["result"]["generationSet"]["model"], "my_depth_run");
    let fact = run.completed_fact();
    assert_eq!(fact["model"], "my_depth_run");
    let lineage = &fact["depth"]["model"];
    assert_eq!(lineage["trainedModelId"], "my_depth_run");
    assert_eq!(lineage["weightsSource"], "training");
    assert_eq!(lineage["weightsRepo"], Value::Null, "{lineage:#}");
    assert_eq!(lineage["weightsRevision"], Value::Null);
    assert_eq!(lineage["weightsSubdir"], Value::Null);
    assert_eq!(lineage["exportPath"], "models/finetunes/my_depth_run/depth");

    // The release run keeps the release identity.
    let release = run_job(JobSetup::default());
    release
        .outcome
        .as_ref()
        .expect("the release depth job succeeds");
    let last = release.posts.last().unwrap();
    assert_eq!(last["result"]["model"], DEPTH_MODEL_ID);
    assert_eq!(last["result"]["generationSet"]["model"], DEPTH_MODEL_ID);
    let lineage = &release.completed_fact()["depth"]["model"];
    assert_eq!(lineage["weightsRepo"], UPSTREAM_WEIGHTS_REPO);
    assert!(lineage.get("weightsSource").is_none());
}

#[test]
fn a_depth_job_publishes_a_raw_npy_and_a_separate_preview_at_the_upright_size() {
    let run = run_job(JobSetup::default());
    run.outcome.as_ref().unwrap();
    assert_eq!(run.load_calls, 1);
    // The estimator saw the EXIF-corrected (orientation 6: 6x4 stored -> 4x6 upright) source and the
    // requested cap.
    assert_eq!(run.seen, Some((4, 6, DepthResolution::Capped(1024))));

    let fact = run.completed_fact();
    assert_eq!(fact["type"], "depth");
    assert_eq!(fact["mimeType"], "application/x-npy");
    assert_eq!(
        (fact["width"].as_u64(), fact["height"].as_u64()),
        (Some(4), Some(6))
    );
    assert_eq!(fact["sourceAssetId"], "asset_source1");
    assert_eq!(fact["parents"], json!(["asset_source1"]));
    assert_eq!(fact["displayName"], "Harbor (depth)");
    assert_eq!(
        fact["rawAdapterSettings"]["depth"]["resolution"],
        json!({ "mode": "capped", "maxSide": 1024 })
    );
    let depth = &fact["depth"];
    assert_eq!(depth["valueConvention"], "relative_log_depth");
    assert_eq!(depth["metric"], false);
    assert_eq!(depth["clamped"], false);
    assert_eq!(depth["npyDescr"], "<f4");
    assert_eq!(depth["shape"], json!([6, 4]));
    assert_eq!(depth["source"]["exifOrientation"], 6);
    assert_eq!(depth["source"]["exifOrientationApplied"], true);
    assert_eq!(
        (
            depth["source"]["storedWidth"].as_u64(),
            depth["source"]["storedHeight"].as_u64()
        ),
        (Some(6), Some(4))
    );
    assert_eq!(depth["preprocessing"]["maxSide"], 1024);
    assert_eq!(depth["model"]["id"], DEPTH_MODEL_ID);
    assert_eq!(depth["model"]["backend"], "fake");
    assert_eq!(depth["model"]["configSha256"], "c".repeat(64));
    assert_eq!(depth["model"]["weightsRevision"], UPSTREAM_WEIGHTS_REVISION);
    assert_eq!(depth["model"]["computeDtype"], "bfloat16");

    // The raw map is exactly the estimator's values, bit for bit, row-major <f4.
    let npy_path = run.project_path.join(fact["mediaPath"].as_str().unwrap());
    let decoded = decode_npy_f32(&std::fs::read(&npy_path).unwrap()).unwrap();
    assert_eq!((decoded.width, decoded.height), (4, 6));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&decoded.values), bits(&fake_values(4, 6)));
    // The preview is a separate PNG at the same size.
    let preview_path = run
        .project_path
        .join(fact["preview"]["path"].as_str().unwrap());
    assert_ne!(preview_path, npy_path);
    assert_eq!(image::open(&preview_path).unwrap().dimensions(), (4, 6));
    // Nothing staged is left behind.
    assert!(run
        .depth_files()
        .iter()
        .all(|path| !path.to_string_lossy().ends_with(".partial")));

    // Reload: the API persists the fact; a FRESH store (the after-restart shape) exposes lineage,
    // convention, preprocessing and identity, and the separate depth-control consumer reads it
    // without Iris and without touching the raw bytes.
    let raw = std::fs::read(&npy_path).unwrap();
    ProjectStore::new(run.data_dir.path().to_path_buf(), "api")
        .persist_generated_asset(&run.project_id, "job-depth", "genset_any", fact)
        .unwrap();
    let reloaded = ProjectStore::new(run.data_dir.path().to_path_buf(), "restarted")
        .get_asset(&run.project_id, fact["assetId"].as_str().unwrap())
        .unwrap();
    assert_eq!(reloaded["type"], "depth");
    assert_eq!(reloaded["lineage"]["sourceAssetId"], "asset_source1");
    assert_eq!(reloaded["depth"], fact["depth"]);
    assert_eq!(reloaded["preview"]["path"], fact["preview"]["path"]);
    let control = crate::depth_asset::depth_asset_control_image(
        run.data_dir.path(),
        &run.project_id,
        fact["assetId"].as_str().unwrap(),
        &run.project_path,
    )
    .unwrap()
    .expect("depth asset");
    assert_eq!((control.width, control.height), (4, 6));
    assert_eq!(std::fs::read(&npy_path).unwrap(), raw);
}

/// The stored-grid position `(x, y)` of the pixel an EXIF viewer shows at upright `(u, v)` — the
/// coordinate map of [`expected_upright`], used to carry a published (upright) map back onto the
/// source's stored pixel grid.
fn stored_position(orientation: u16, stored: (u32, u32), u: u32, v: u32) -> (u32, u32) {
    let (w, h) = stored;
    match orientation {
        1 => (u, v),
        2 => (w - 1 - u, v),
        3 => (w - 1 - u, h - 1 - v),
        4 => (u, h - 1 - v),
        5 => (v, u),
        6 => (v, h - 1 - u),
        7 => (w - 1 - v, h - 1 - u),
        8 => (w - 1 - v, u),
        _ => unreachable!(),
    }
}

/// Epic AT3 (sc-25690) — depth across EXIF orientation, through the job path. The engine's
/// estimator takes already-upright RGB8 (`gen_core_iris::depth`: an `Image` carries no EXIF) and
/// is held to upstream on those pixels by the engine's own `depth_parity`; the orientation step —
/// upstream's `ImageOps.exif_transpose` — is THIS layer's. So for every EXIF orientation (3, 6 and 8
/// among them) the published raw map must be, bit for bit (declared tolerance 0: orientation is a
/// pixel permutation), the reference estimate of the UPRIGHT pixels at the upright size; and carried
/// back onto the stored grid it must be the reference depth of each stored pixel. The reference is
/// [`pixel_depth`], which identifies the pixel at each position.
#[test]
fn an_exif_rotated_source_yields_the_reference_depth_of_its_upright_pixels() {
    let stored = job_source();
    let stored_reference = pixel_depth(stored.as_raw());
    let mut distinct: Vec<u32> = stored_reference.iter().map(|v| v.to_bits()).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        stored_reference.len(),
        "the reference identifies every source pixel"
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for orientation in 1..=8u16 {
        let run = run_job(JobSetup {
            behavior: Behavior::PixelDepth,
            orientation: Some(orientation),
            resolution: json!({ "mode": "native" }),
            ..JobSetup::default()
        });
        run.outcome
            .as_ref()
            .unwrap_or_else(|error| panic!("orientation {orientation}: {error}"));
        let upright = expected_upright(&stored, orientation);
        let (width, height) = upright.dimensions();
        assert_eq!(
            (width, height),
            if orientation >= 5 { (4, 6) } else { (6, 4) },
            "orientation {orientation}"
        );
        assert_eq!(
            run.seen,
            Some((width, height, DepthResolution::Native)),
            "orientation {orientation}: the estimator saw the upright size"
        );
        let fact = run.completed_fact();
        let depth = &fact["depth"];
        assert_eq!(depth["source"]["exifOrientation"], orientation);
        assert_eq!(depth["source"]["exifOrientationApplied"], orientation != 1);
        assert_eq!(
            (
                depth["source"]["storedWidth"].as_u64(),
                depth["source"]["storedHeight"].as_u64()
            ),
            (Some(6), Some(4)),
            "orientation {orientation}"
        );
        assert_eq!(depth["shape"], json!([height, width]));
        let map = decode_npy_f32(
            &std::fs::read(run.project_path.join(fact["mediaPath"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        assert_eq!(
            (map.width, map.height),
            (width, height),
            "orientation {orientation}: the map is at the upright size"
        );
        // The reference on the upright pixels.
        assert_eq!(
            bits(&map.values),
            bits(&pixel_depth(upright.as_raw())),
            "orientation {orientation}: the map is the reference estimate of the upright pixels"
        );
        // Re-oriented to the source's stored geometry: each stored pixel carries its own depth.
        let mut on_stored = vec![f32::NAN; stored_reference.len()];
        for v in 0..height {
            for u in 0..width {
                let (x, y) = stored_position(orientation, (6, 4), u, v);
                on_stored[(y * 6 + x) as usize] = map.values[(v * width + u) as usize];
            }
        }
        assert_eq!(
            bits(&on_stored),
            bits(&stored_reference),
            "orientation {orientation}: carried back to the stored grid, every source pixel has \
             its reference depth"
        );
    }
}

/// Epic AT6 (sc-25690) — no valid depth pixels at inference. An estimate with no finite value is
/// not depth: the job fails with its typed code, and neither the raw map, the preview, a staged
/// partial, nor an asset fact is published; the source asset's file is untouched.
#[test]
fn an_estimate_with_no_valid_depth_pixels_is_refused_and_publishes_nothing() {
    let run = run_job(JobSetup {
        behavior: Behavior::NoFiniteDepth,
        ..JobSetup::default()
    });
    let error = run.outcome.as_ref().unwrap_err().to_string();
    assert!(error.contains("[iris_depth_no_valid_pixels]"), "{error}");
    assert!(
        matches!(run.outcome, Err(WorkerError::Engine(_))),
        "{:?}",
        run.outcome
    );
    assert_eq!(run.load_calls, 1, "the estimate ran");
    assert!(
        run.posts
            .iter()
            .all(|post| post["status"] != "completed" && post["result"].is_null()),
        "no completion and no asset fact was posted: {:#?}",
        run.posts
    );
    assert!(run.depth_files().is_empty(), "{:?}", run.depth_files());
    let scratch = tempfile::tempdir().unwrap();
    let pristine = scratch.path().join("source.png");
    write_png_with_orientation(&pristine, &job_source(), Some(6));
    assert_eq!(
        std::fs::read(run.project_path.join("assets/images/genset_src/source.png")).unwrap(),
        std::fs::read(&pristine).unwrap(),
        "the source asset's file is untouched"
    );
}

#[test]
fn native_processing_is_passed_through_explicitly() {
    let run = run_job(JobSetup {
        resolution: json!({ "mode": "native" }),
        orientation: None,
        ..JobSetup::default()
    });
    run.outcome.as_ref().unwrap();
    assert_eq!(run.seen, Some((6, 4, DepthResolution::Native)));
    let depth = &run.completed_fact()["depth"];
    assert_eq!(
        depth["preprocessing"]["resolution"],
        json!({ "mode": "native" })
    );
    assert_eq!(depth["preprocessing"]["maxSide"], 0);
    assert_eq!(depth["source"]["exifOrientation"], 1);
}

#[test]
fn a_canceled_depth_job_leaves_no_asset_and_no_file() {
    let run = run_job(JobSetup {
        behavior: Behavior::WaitsForCancel,
        cancel_requested: true,
        ..JobSetup::default()
    });
    assert!(
        matches!(run.outcome, Err(WorkerError::Canceled(_))),
        "{:?}",
        run.outcome
    );
    assert!(run.posts.iter().any(|post| post["status"] == "canceled"));
    assert!(run
        .posts
        .iter()
        .all(|post| post["result"]["assetWrites"].is_null()));
    assert!(run.depth_files().is_empty(), "{:?}", run.depth_files());
}

/// The cancel lands after the estimate finished and the staged `.npy` / preview were written: the
/// job still ends canceled, and the staged files are removed rather than published or orphaned.
#[test]
fn a_cancel_after_the_files_were_staged_still_leaves_nothing() {
    let run = run_job(JobSetup {
        behavior: Behavior::FinishesDespiteCancel,
        cancel_requested: true,
        ..JobSetup::default()
    });
    assert!(
        matches!(run.outcome, Err(WorkerError::Canceled(_))),
        "{:?}",
        run.outcome
    );
    assert!(run.posts.iter().any(|post| post["status"] == "canceled"));
    assert!(run.depth_files().is_empty(), "{:?}", run.depth_files());
    assert!(
        !run.project_path
            .join("assets/depth")
            .read_dir()
            .is_ok_and(|mut d| d.next().is_some()),
        "no empty set directory is left either"
    );
}

#[test]
fn an_engine_failure_leaves_no_asset_and_no_file() {
    let run = run_job(JobSetup {
        behavior: Behavior::Fails,
        ..JobSetup::default()
    });
    let error = run.outcome.as_ref().unwrap_err().to_string();
    assert!(error.contains("fixture engine failure"), "{error}");
    assert!(run.depth_files().is_empty(), "{:?}", run.depth_files());
}

/// The files are published before the completion post that hands the API the asset fact; if the
/// API refuses that post, no asset exists, so the published `.npy` and preview are removed too.
#[test]
fn a_refused_completion_leaves_no_file() {
    let run = run_job(JobSetup {
        reject_completion: true,
        ..JobSetup::default()
    });
    assert!(run.outcome.is_err(), "{:?}", run.outcome);
    assert!(
        run.posts.iter().any(|post| post["status"] == "completed"),
        "the completion was attempted: {:#?}",
        run.posts
    );
    assert!(run.depth_files().is_empty(), "{:?}", run.depth_files());
}

#[test]
fn missing_incomplete_and_over_budget_jobs_fail_typed_before_loading() {
    let missing = run_job(JobSetup {
        stage: &[],
        ..JobSetup::default()
    });
    assert!(missing
        .outcome
        .as_ref()
        .unwrap_err()
        .to_string()
        .starts_with("[iris_depth_not_installed]"));
    assert_eq!(missing.load_calls, 0);

    let torn = run_job(JobSetup {
        stage: &[BACKBONE_CONFIG_FILE, EMPTY_PROMPT_FILE],
        ..JobSetup::default()
    });
    assert!(torn
        .outcome
        .as_ref()
        .unwrap_err()
        .to_string()
        .starts_with("[iris_depth_incomplete_install]"));
    assert_eq!(torn.load_calls, 0);

    let refused = run_job(JobSetup {
        memory_cap_gb: "4",
        ..JobSetup::default()
    });
    assert!(refused
        .outcome
        .as_ref()
        .unwrap_err()
        .to_string()
        .starts_with("[iris_depth_memory_admission]"));
    assert_eq!(refused.load_calls, 0);
    assert!(refused.depth_files().is_empty());
}

// ---- task switching (sc-25684) -------------------------------------------------------------------

/// Generation ↔ depth ↔ restoration share one GPU budget: a depth job evicts a warm generator (and
/// the text encoder staged inside it) or a resident restoration transform BEFORE it admits and loads
/// its own estimator, so two Iris task models are never resident together over the cap. The
/// estimator itself loads with no text encoder and no components (asserted by the loader in
/// `run_job`), and leaves nothing resident after the job.
///
/// *Mutation that reds this:* drop the `evict_resident_engines()` call in the job body.
#[test]
fn a_depth_job_evicts_a_resident_generator_or_transform_before_loading() {
    for resident in [Resident::Generator, Resident::Transform] {
        let run = run_job(JobSetup {
            resident: Some(resident),
            ..JobSetup::default()
        });
        run.outcome.as_ref().unwrap();
        assert_eq!(run.load_calls, 1, "{resident:?}");
        assert_eq!(
            run.resident_after,
            (false, false),
            "{resident:?} must be evicted by the depth job"
        );
    }
    // Over-budget refusals still leave typed outcomes and no files after the switch.
    let refused = run_job(JobSetup {
        resident: Some(Resident::Generator),
        memory_cap_gb: "4",
        ..JobSetup::default()
    });
    assert!(refused
        .outcome
        .as_ref()
        .unwrap_err()
        .to_string()
        .starts_with("[iris_depth_memory_admission]"));
    assert_eq!(refused.load_calls, 0);
    assert!(refused.depth_files().is_empty());
}
