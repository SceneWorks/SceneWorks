//! Real-weight acceptance for Iris-3B depth (sc-25682) on whichever native lane this build links:
//! MLX on macOS, Candle/CUDA off-Mac (the dispatch-only `iris-worker-smoke` job in
//! windows-candle.yml runs it — its `iris_worker_gpu_smoke` filter matches this test too).
//!
//! Drives ONE production `image_depth` job through `run_image_depth_job` (the backend loader, the
//! installed `depth/` export resolved cache-only from `HF_HUB_CACHE`, the shipped catalog entry's
//! admission) on a real photo stored with EXIF orientation 6, and checks: the raw `.npy` is at the
//! UPRIGHT size, is finite and not constant; the preview is a separate PNG at the same size; and the
//! near-bright adapter puts the portrait subject nearer than the backdrop. An absent snapshot is a
//! failure, never a skip; nothing is downloaded.
//!
//! ```text
//! HF_HUB_CACHE=~/.cache/huggingface/hub cargo test -p sceneworks-worker --release \
//!   iris_worker_gpu_smoke_depth -- --ignored --nocapture --test-threads=1
//! ```

use super::*;
use crate::api_client::ApiClient;
use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use sceneworks_core::depth_map::decode_npy_f32;
use std::sync::{Arc, Mutex};

type Posts = Arc<Mutex<Vec<Value>>>;

fn snapshot(job_id: &str) -> Value {
    let mut job = crate::tests::job_snapshot_json(job_id, false);
    job["type"] = json!("image_depth");
    job
}

async fn spawn_api() -> (String, Posts) {
    async fn job(AxumPath(id): AxumPath<String>) -> Json<Value> {
        Json(snapshot(&id))
    }
    async fn progress(
        State(posts): State<Posts>,
        AxumPath(id): AxumPath<String>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        posts.lock().unwrap().push(body);
        Json(snapshot(&id))
    }
    async fn heartbeat(AxumPath(worker): AxumPath<String>) -> Json<Value> {
        Json(json!({
            "id": worker, "gpuId": "0", "gpuName": null, "status": "busy",
            "currentJobId": "iris-depth-smoke", "capabilities": [], "loadedModels": [],
            "registeredAt": "2026-10-11T00:00:00Z", "lastSeenAt": "2026-10-11T00:00:00Z"
        }))
    }
    let posts: Posts = Arc::default();
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(job))
        .route("/api/v1/jobs/:job_id/progress", post(progress))
        .route("/api/v1/workers/:worker_id/heartbeat", post(heartbeat))
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), posts)
}

fn shipped_depth_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .unwrap()
        .1;
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text)).unwrap();
    manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == DEPTH_MODEL_ID)
        .cloned()
        .unwrap()
}

/// A little-endian EXIF block with one Orientation entry.
fn exif_orientation(orientation: u16) -> Vec<u8> {
    let mut tiff = b"II\x2a\x00".to_vec();
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&1u16.to_le_bytes());
    tiff.extend_from_slice(&0x0112u16.to_le_bytes());
    tiff.extend_from_slice(&3u16.to_le_bytes());
    tiff.extend_from_slice(&1u32.to_le_bytes());
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    tiff
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight depth smoke; needs the pinned speridlabs/iris-3b depth/ export under HF_HUB_CACHE and a GPU"]
async fn iris_worker_gpu_smoke_depth() {
    use image::ImageEncoder;
    let hub = PathBuf::from(std::env::var("HF_HUB_CACHE").expect("set HF_HUB_CACHE"));
    assert!(
        hub.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hub.display()
    );
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_api().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.worker_id = "iris-depth-smoke".to_owned();
    settings.heartbeat_seconds = 30;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        // The worker's gpu_id is the nvidia-smi (physical, PCI-bus-ordered) index the admission
        // budget is read from; under `CUDA_VISIBLE_DEVICES` that is the visible device's index,
        // exactly as a production per-GPU worker is launched.
        settings.gpu_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|devices| devices.split(',').next().map(|d| d.trim().to_owned()))
            .filter(|device| !device.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        settings.backend_candle_enabled = true;
    }
    resolve_depth_export(&settings).expect("the depth/ export is installed (no download here)");

    // A real portrait photo, cropped landscape (1024x768) and STORED rotated: EXIF orientation 6
    // means the upright image is 768x1024.
    let photo = image::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sceneworks-memory-adapter/fixtures/pulid-reference.png"),
    )
    .expect("fixture photo decodes")
    .to_rgb8();
    let upright = image::imageops::crop_imm(&photo, 128, 0, 768, 1024).to_image();
    // Store it rotated 90° counter-clockwise; orientation 6 tells a viewer to rotate it back.
    let stored = image::imageops::rotate270(&upright);
    assert_eq!(stored.dimensions(), (1024, 768));
    let store = ProjectStore::new(settings.data_dir.clone(), "smoke");
    let project = store.create_project("Iris depth smoke").unwrap();
    let project_path = PathBuf::from(&project.path);
    let source_rel = "assets/images/genset_src/portrait.png";
    std::fs::create_dir_all(project_path.join("assets/images/genset_src")).unwrap();
    let mut bytes = Vec::new();
    let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
    encoder.set_exif_metadata(exif_orientation(6)).unwrap();
    encoder
        .write_image(stored.as_raw(), 1024, 768, image::ExtendedColorType::Rgb8)
        .unwrap();
    std::fs::write(project_path.join(source_rel), bytes).unwrap();
    store
        .persist_generated_asset(
            &project.id,
            "job-src",
            "genset_src",
            &json!({
                "assetId": "asset_portrait", "mediaPath": source_rel, "mimeType": "image/png",
                "width": 1024, "height": 768, "displayName": "Portrait", "createdAt": "2026-10-11T00:00:00Z",
                "mode": "upload", "model": "none", "adapter": "upload", "prompt": "", "loras": [],
            }),
        )
        .unwrap();

    let mut job: JobSnapshot = serde_json::from_value(snapshot("iris-depth-smoke")).unwrap();
    job.payload = json!({
        "projectId": project.id,
        "sourceAssetId": "asset_portrait",
        "resolution": { "mode": "capped", "maxSide": 1024 },
        "modelManifestEntries": [shipped_depth_entry()],
    })
    .as_object()
    .cloned()
    .unwrap();
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    run_image_depth_job(&api, &settings, &job)
        .await
        .expect("the Iris depth job completes");
    let elapsed = started.elapsed();

    let posts = posts.lock().unwrap().clone();
    let fact = posts
        .iter()
        .rev()
        .find(|post| post["status"] == "completed")
        .map(|post| post["result"]["assetWrites"][0].clone())
        .expect("a completed result");
    let depth = &fact["depth"];
    assert_eq!(depth["source"]["exifOrientation"], 6);
    assert_eq!(
        (fact["width"].as_u64(), fact["height"].as_u64()),
        (Some(768), Some(1024))
    );
    assert_eq!(
        (
            depth["preprocessing"]["modelWidth"].as_u64(),
            depth["preprocessing"]["modelHeight"].as_u64()
        ),
        (Some(768), Some(1024))
    );

    let npy_path = project_path.join(fact["mediaPath"].as_str().unwrap());
    let map = decode_npy_f32(&std::fs::read(&npy_path).unwrap()).unwrap();
    assert_eq!((map.width, map.height), (768, 1024));
    assert!(map.values.iter().all(|v| v.is_finite()), "finite depth");
    let (min, max) = map
        .values
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(max - min > 0.1, "a non-constant depth map: [{min}, {max}]");
    let preview =
        image::open(project_path.join(fact["preview"]["path"].as_str().unwrap())).unwrap();
    assert_eq!((preview.width(), preview.height()), (768, 1024));

    // The subject (face, upright centre) is nearer than the backdrop (top corners).
    let control = gen_core_iris::depth::near_bright_unit(&gen_core_iris::depth::DepthMap {
        width: map.width,
        height: map.height,
        values: map.values.clone(),
    });
    let mean = |x0: u32, y0: u32, x1: u32, y1: u32| {
        let mut sum = 0.0f64;
        let mut n = 0.0f64;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += f64::from(control[(y * map.width + x) as usize]);
                n += 1.0;
            }
        }
        sum / n
    };
    let face = mean(300, 300, 520, 560);
    let corners = (mean(0, 0, 96, 96) + mean(672, 0, 768, 96)) / 2.0;
    assert!(
        face > corners + 0.2,
        "the subject must read nearer than the backdrop: face {face:.3} vs corners {corners:.3}"
    );
    eprintln!(
        "RESULT status=pass model={DEPTH_MODEL_ID} backend={} source=1024x768+exif6 upright=768x1024 \
         model_input={}x{} elapsed_s={:.1} range=[{min:.3},{max:.3}] face_near={face:.3} \
         corners_near={corners:.3} config_sha256={} npy={}",
        depth["model"]["backend"].as_str().unwrap_or("?"),
        depth["preprocessing"]["modelWidth"],
        depth["preprocessing"]["modelHeight"],
        elapsed.as_secs_f64(),
        depth["model"]["configSha256"].as_str().unwrap_or("?"),
        npy_path.display(),
    );
}
