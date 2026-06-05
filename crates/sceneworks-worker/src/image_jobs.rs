//! Native MLX image generation jobs — runtime pipeline (epic 3018, sc-3020).
//!
//! Parses the job into an [`ImageRequest`], generates `count` images, saves each PNG
//! into the project's `assets/images/`, and reports flat "facts" the Rust API turns
//! into indexed assets. The API's `persist_reported_assets` (apps/rust-api jobs.rs)
//! runs on EVERY progress update — idempotently building each sidecar via
//! `build_image_sidecar_parts` and indexing project.db — so emitting the accumulating
//! `assetWrites` per image is what streams results into the gallery as they land.
//!
//! sc-3020 ships the pipeline with a **procedural stub** generator (adapter
//! `procedural_preview`); sc-3022 swaps the stub for real Z-Image inference via the
//! linked mlx-gen engine. The stub keeps this whole path cross-platform-testable.

use super::*;
use sceneworks_core::image_request::ImageRequest;

// Force the Z-Image provider crate to link so its `inventory::submit!` registration
// survives linker GC (an only-declared dependency can be dropped). Each per-family
// story adds its provider + a matching `use … as _;`. See mlx-gen-z-image/tests/
// registry.rs ("the SceneWorks worker"). Real inference lands in sc-3022.
#[cfg(target_os = "macos")]
use mlx_gen_z_image as _;

/// The stub adapter id recorded on generated assets (matches the contract fixture
/// `tests/fixtures/rust_migration_contracts/sidecars/asset-image.sceneworks.json`).
const STUB_ADAPTER: &str = "procedural_preview";

/// Dispatch handler for `JobType::ImageGenerate`: generate, save, and stream image
/// assets through the Rust GPU worker.
pub(crate) async fn run_image_generate_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let request = ImageRequest::from_payload(&job.payload);
    if request.project_id.trim().is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Missing payload.projectId".to_owned(),
        ));
    }
    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    tokio::fs::create_dir_all(project_path.join("assets").join("images")).await?;

    let plan = ImagePlan::new(&request);
    let backend = backend_label(&settings.gpu_id);

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            &format!("Preparing {} image(s) ({STUB_ADAPTER}).", request.count),
            None,
            backend,
        ),
    )
    .await?;

    let mut asset_writes: Vec<Value> = Vec::with_capacity(request.count as usize);
    for index in 0..request.count as usize {
        check_cancel(api, &job.id, "Image generation canceled by user.").await?;
        let fact = render_and_save(&plan, index, &project_path)?;
        asset_writes.push(Value::Object(fact));
        let progress = 0.1 + 0.85 * ((index + 1) as f64 / request.count as f64);
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Running,
                ProgressStage::Generating,
                progress,
                &format!("Generated image {}/{}.", index + 1, request.count),
                Some(streaming_result(&plan, &asset_writes)),
                backend,
            ),
        )
        .await?;
        heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    }

    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Generated {} image(s).", request.count),
            Some(streaming_result(&plan, &asset_writes)),
            backend,
        ),
    )
    .await?;
    Ok(())
}

/// Per-job invariants shared across every image in the generation set.
struct ImagePlan {
    request: ImageRequest,
    genset_id: String,
    created_at: String,
    family: String,
    slug: String,
    generation_set: Value,
}

impl ImagePlan {
    fn new(request: &ImageRequest) -> Self {
        let genset_id = format!("genset_{}", Uuid::new_v4().simple());
        let created_at = now_rfc3339();
        let family = resolve_family(request);
        let slug = slugify(&request.prompt, "image", Some(42));
        let generation_set = json!({
            "id": genset_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": request.count,
            "createdAt": created_at,
        });
        Self {
            request: request.clone(),
            genset_id,
            created_at,
            family,
            slug,
            generation_set,
        }
    }
}

/// Render image `index`, save its PNG under `assets/images/`, and return the flat
/// fact the API turns into an indexed asset (every key here is consumed by
/// `build_image_sidecar_parts`). The stub fills a deterministic per-seed gradient.
fn render_and_save(
    plan: &ImagePlan,
    index: usize,
    project_path: &Path,
) -> WorkerResult<JsonObject> {
    let request = &plan.request;
    let seed = request.seed_for(index).unwrap_or_else(random_seed);
    let pixels = stub_rgb8(request.width, request.height, seed);
    let rgb_image = image::RgbImage::from_raw(request.width, request.height, pixels)
        .ok_or_else(|| WorkerError::InvalidPayload("image buffer size mismatch".to_owned()))?;

    let filename = format!(
        "{}_{}_{}_{:04}.png",
        &plan.created_at[..10],
        request.model,
        plan.slug,
        index + 1
    );
    let media_rel = format!("assets/images/{filename}");
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");
    rgb_image
        .save_with_format(&temp_path, image::ImageFormat::Png)
        .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
    std::fs::rename(&temp_path, &media_path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })?;

    let title: String = request.prompt.chars().take(56).collect();
    let title = title.trim();
    let display_name = format!(
        "{} #{}",
        if title.is_empty() {
            "Generated image"
        } else {
            title
        },
        index + 1
    );

    let mut raw_settings = request.advanced.clone();
    // Honest marker: this is the procedural stub, not real model inference (sc-3022
    // sets realModelInference:true + repo/steps/guidance/mlxQuantize).
    raw_settings.insert("realModelInference".to_owned(), Value::Bool(false));
    raw_settings.insert("adapter".to_owned(), Value::String(STUB_ADAPTER.to_owned()));

    let fact = json!({
        "assetId": fresh_asset_id(),
        "type": "image",
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "width": request.width,
        "height": request.height,
        "normalizedWidth": request.width,
        "normalizedHeight": request.height,
        "count": request.count,
        "family": plan.family,
        "seed": seed,
        "index": index,
        "displayName": display_name,
        "createdAt": now_rfc3339(),
        "mode": request.mode,
        "model": request.model,
        "adapter": STUB_ADAPTER,
        "prompt": request.prompt,
        "negativePrompt": request.negative_prompt,
        "loras": request.loras,
        "stylePreset": request.style_preset,
        "characterId": request.character_id,
        "characterLookId": request.character_look_id,
        "sourceAssetId": request.source_asset_id,
        "rawAdapterSettings": raw_settings,
    });
    Ok(fact.as_object().cloned().expect("json! object literal"))
}

/// The job-result shape the API streams from: `assetWrites` + the `generationSet`
/// fact drive `persist_reported_assets` (idempotent per progress update).
fn streaming_result(plan: &ImagePlan, asset_writes: &[Value]) -> JsonObject {
    json!({
        "generationSetId": plan.genset_id,
        "expectedCount": plan.request.count,
        "adapter": STUB_ADAPTER,
        "model": plan.request.model,
        "generationSet": plan.generation_set,
        "assetWrites": asset_writes,
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// The asset `family` for the recipe's normalizedSettings: the resolved model
/// manifest entry wins (the UI sends it), else the linked mlx-gen descriptor's family
/// on macOS, else empty.
fn resolve_family(request: &ImageRequest) -> String {
    if let Some(family) = request
        .model_manifest_entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return family.to_owned();
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(family) = mlx_gen::registry::generators()
            .find(|registration| (registration.descriptor)().id == request.model)
            .map(|registration| (registration.descriptor)().family)
        {
            return family.to_owned();
        }
    }
    String::new()
}

/// Progress payload with the worker's real backend label (the shared
/// `progress_payload` hardcodes `cpu`; the MLX worker reports `mlx`). Peak GPU/unified
/// memory is left unset for the procedural stub — it allocates no device memory; real
/// peak-memory sampling lands with real inference (sc-3022).
fn image_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    ProgressRequest {
        status,
        stage,
        progress: number_from_f64(progress),
        message: message.to_owned(),
        error: None,
        result,
        eta_seconds: None,
        peak_gpu_memory_pct: None,
        peak_gpu_load_pct: None,
        backend: Some(backend.to_owned()),
        extra: BTreeMap::new(),
    }
}

fn backend_label(gpu_id: &str) -> &str {
    if gpu_id.trim().is_empty() {
        "cpu"
    } else {
        gpu_id
    }
}

/// A non-deterministic seed when the request supplies none (recorded on the asset so
/// the result is reproducible). Derived from a v4 UUID to avoid a new RNG dependency.
fn random_seed() -> i64 {
    (Uuid::new_v4().as_u128() as u64 & 0x7FFF_FFFF) as i64
}

/// Deterministic placeholder pixels: a vertical gradient from a per-seed base colour
/// to white, so each seed yields a distinct, valid RGB8 image of exactly `width *
/// height * 3` bytes.
fn stub_rgb8(width: u32, height: u32, seed: i64) -> Vec<u8> {
    let seed = seed as u64;
    let base = [
        (seed & 0xFF) as u8,
        ((seed >> 8) & 0xFF) as u8,
        ((seed >> 16) & 0xFF) as u8,
    ];
    let span = height.saturating_sub(1).max(1) as f32;
    let mut buffer = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for y in 0..height {
        let t = y as f32 / span;
        let row = [lerp(base[0], t), lerp(base[1], t), lerp(base[2], t)];
        for _ in 0..width {
            buffer.extend_from_slice(&row);
        }
    }
    buffer
}

/// Linear interpolate channel value `a` toward white (255) by `t` in `[0, 1]`.
fn lerp(a: u8, t: f32) -> u8 {
    let a = a as f32;
    (a + (255.0 - a) * t).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> ImageRequest {
        ImageRequest::from_payload(&value.as_object().cloned().unwrap())
    }

    #[test]
    fn render_and_save_writes_png_and_contract_fact() {
        let dir = tempfile::tempdir().unwrap();
        let project_path = dir.path();
        std::fs::create_dir_all(project_path.join("assets").join("images")).unwrap();
        // Distinct dimensions (>= the 256 min, so they survive clamping) also catch a
        // width/height transpose in the encoder.
        let req = request(json!({
            "projectId": "p", "model": "z_image_turbo", "prompt": "Mist over hills",
            "count": 2, "width": 320, "height": 256, "seed": 101,
            "stylePreset": "cinematic", "modelManifestEntry": { "family": "z-image" }
        }));
        let plan = ImagePlan::new(&req);

        let fact = render_and_save(&plan, 0, project_path).unwrap();

        // The PNG exists at the reported relative path and decodes at the requested size.
        let media_rel = fact.get("mediaPath").and_then(Value::as_str).unwrap();
        assert!(media_rel.starts_with("assets/images/"));
        assert!(media_rel.ends_with("_0001.png"));
        let decoded = image::open(project_path.join(media_rel)).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (320, 256));

        // The fact carries every field the API's build_image_sidecar_parts consumes.
        for key in [
            "assetId",
            "mediaPath",
            "mimeType",
            "width",
            "height",
            "normalizedWidth",
            "normalizedHeight",
            "count",
            "family",
            "seed",
            "displayName",
            "createdAt",
            "mode",
            "model",
            "adapter",
            "prompt",
            "negativePrompt",
            "loras",
            "stylePreset",
            "characterId",
            "characterLookId",
            "sourceAssetId",
            "rawAdapterSettings",
        ] {
            assert!(fact.contains_key(key), "fact missing key {key}");
        }
        assert_eq!(fact["adapter"], json!("procedural_preview"));
        assert_eq!(fact["family"], json!("z-image"));
        assert_eq!(fact["seed"], json!(101));
        assert_eq!(fact["mimeType"], json!("image/png"));
        assert_eq!(fact["width"], json!(320));
        assert_eq!(fact["displayName"], json!("Mist over hills #1"));
        // Honest: the stub is not real inference.
        assert_eq!(
            fact["rawAdapterSettings"]["realModelInference"],
            json!(false)
        );
    }

    #[test]
    fn distinct_seeds_produce_distinct_pixels() {
        let a = stub_rgb8(8, 8, 1);
        let b = stub_rgb8(8, 8, 5000);
        assert_eq!(a.len(), 8 * 8 * 3);
        assert_ne!(a, b);
    }

    #[test]
    fn plan_builds_generation_set_with_unique_seeds() {
        let plan = ImagePlan::new(&request(
            json!({ "projectId": "p", "prompt": "x", "count": 3, "seed": 5 }),
        ));
        assert!(plan.genset_id.starts_with("genset_"));
        assert_eq!(plan.generation_set["count"], json!(3));
        assert_ne!(plan.request.seed_for(0), plan.request.seed_for(1));
    }

    #[test]
    fn streaming_result_carries_facts_for_api_persistence() {
        let plan = ImagePlan::new(&request(
            json!({ "projectId": "p", "prompt": "x", "count": 1 }),
        ));
        let writes = vec![json!({ "assetId": "a1" })];
        let result = streaming_result(&plan, &writes);
        assert_eq!(result["generationSetId"], json!(plan.genset_id));
        assert_eq!(result["assetWrites"].as_array().map(Vec::len), Some(1));
        assert_eq!(result["adapter"], json!("procedural_preview"));
        assert!(result.contains_key("generationSet"));
    }

    #[test]
    fn backend_label_defaults_empty_to_cpu() {
        assert_eq!(backend_label("mlx"), "mlx");
        assert_eq!(backend_label(""), "cpu");
    }

    /// The Z-Image provider linked into the worker self-registered via inventory —
    /// proof the cross-crate mlx-gen registry resolves inside our binary (sc-3019).
    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_engine_registry_links_z_image() {
        assert!(mlx_gen::registry::generators().any(|reg| (reg.descriptor)().id == "z_image_turbo"));
    }
}
