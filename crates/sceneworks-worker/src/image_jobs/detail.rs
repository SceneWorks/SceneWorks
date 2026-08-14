/// The xinsir tile ControlNet repo (parity with Python `TILE_CONTROLNET_REPO`).
const TILE_CONTROLNET_REPO: &str = "xinsir/controlnet-tile-sdxl-1.0";
const DETAIL_DEFAULT_PROMPT: &str = "ultra detailed, sharp focus, fine texture, high quality";
const DETAIL_DEFAULT_NEGATIVE: &str = "blurry, soft, lowres, smooth, plastic";

// The Mac advanced-SDXL module already provides this shared-module helper. Detail is also compiled
// for Candle, where that Mac-only include is absent, so provide the identical conversion there.
#[cfg(not(target_os = "macos"))]
fn engine_image_to_rgb(image: Image) -> WorkerResult<image::RgbImage> {
    image::RgbImage::from_raw(image.width, image.height, image.pixels)
        .ok_or_else(|| WorkerError::InvalidPayload("image buffer size mismatch".to_owned()))
}

/// The locked detail recipe (sc-2437 round-2 spike defaults), resolved from `advanced`.
#[derive(Clone)]
struct DetailParams {
    strength: f32,
    cn_scale: f32,
    steps: u32,
    guidance: f32,
    tile: u32,
    overlap: u32,
    prompt: String,
    negative: String,
    seed: i64,
}

/// Events emitted by the blocking tiled-detail pass. Tile completion is load-bearing progress;
/// previews are decorative and use a non-blocking send so a slow API consumer cannot stall denoise.
enum DetailEvent {
    Tile { done: usize, total: usize },
    Preview(gen_core::PreviewFrame),
}

fn resolve_detail_params(request: &ImageRequest) -> DetailParams {
    DetailParams {
        strength: advanced::f32_clamped(&request.advanced, "strength", 0.55, 0.2..=1.0),
        cn_scale: advanced::f32_clamped(&request.advanced, "cnScale", 0.7, 0.1..=1.5),
        steps: advanced::u32_clamped(&request.advanced, "steps", 24, 1..=60),
        guidance: advanced::f32_clamped(&request.advanced, "guidanceScale", 5.0, 1.0..=15.0),
        tile: advanced::u32_clamped(&request.advanced, "tile", 1024, 512..=1536),
        overlap: advanced::u32_clamped(&request.advanced, "overlap", 128, 0..=512),
        prompt: advanced::str(&request.advanced, "prompt", DETAIL_DEFAULT_PROMPT),
        negative: advanced::str(&request.advanced, "negativePrompt", DETAIL_DEFAULT_NEGATIVE),
        // Python defaults the detail seed to 7 when the payload omits one.
        seed: request.seed.unwrap_or(7),
    }
}

/// Round a tile dimension up to the nearest multiple of 8 and clamp to the engine's
/// `[512, 2048]` SDXL bounds, so an arbitrary-sized crop can be run through the engine.
fn engine_dim(value: u32) -> u32 {
    value.div_ceil(8).saturating_mul(8).clamp(512, 2048)
}

/// Validate and resolve the Candle-only inputs before the detail provider loads. The raw Batch
/// Detail route stamps a dense request, but the shared standard-tier resolver deliberately falls
/// back to an installed sibling when the requested tier is absent. That behavior is useful for
/// ordinary generation and unsafe here: packed SDXL detail is unsupported. Require the actual
/// resolved standard-layout directory to be `bf16`, then resolve the descriptor-owned components.
/// Flat custom SDXL directories remain valid dense inputs; they have no packed tier basename and
/// are already protected by the `quantized` check.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn resolve_candle_detail_components(
    request: &ImageRequest,
    settings: &Settings,
    weights_dir: &Path,
    quantized: bool,
) -> WorkerResult<(WeightsSource, WeightsSource, WeightsSource)> {
    if quantized {
        return Err(WorkerError::InvalidPayload(
            "SDXL detail on Candle requires a dense SDXL-family base; packed/request-quantized detail is not supported"
                .to_owned(),
        ));
    }
    if uses_standard_tier_layout(request)
        && tier_key_from_resolved_dir(weights_dir) != Some("bf16")
    {
        return Err(WorkerError::InvalidPayload(format!(
            "SDXL detail on Candle requires the installed dense bf16 model tier; the requested tier resolved to {}",
            weights_dir.display()
        )));
    }
    resolve_sdxl_components(&request.model_manifest_entry, settings)
}

/// Raised-cosine alpha ramp over the `overlap` borders so tiles blend seamlessly
/// (parity with Python `_detail_feather`). Row-major `tile_h`×`tile_w` weights.
fn detail_feather(tile_w: u32, tile_h: u32, overlap: u32) -> Vec<f32> {
    fn ramp(n: u32, overlap: u32) -> Vec<f32> {
        let mut weights = vec![1.0f32; n as usize];
        if overlap > 0 && n > overlap {
            for index in 0..overlap as usize {
                let edge = 0.5
                    - 0.5 * (std::f32::consts::PI * (index as f32 + 0.5) / overlap as f32).cos();
                weights[index] = edge;
                weights[n as usize - 1 - index] = edge;
            }
        }
        weights
    }
    let wx = ramp(tile_w, overlap);
    let wy = ramp(tile_h, overlap);
    let mut out = Vec::with_capacity((tile_w * tile_h) as usize);
    for &vy in &wy {
        for &vx in &wx {
            out.push(vy * vx);
        }
    }
    out
}

/// Build the SDXL generator spec with the tile ControlNet overlay.
#[cfg(target_os = "macos")]
fn detail_spec(
    weights_dir: PathBuf,
    control_file: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir))
        .with_control(WeightsSource::File(control_file));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    spec
}

trait DetailTileRefiner {
    #[allow(clippy::too_many_arguments)]
    fn refine_tile(
        &self,
        tile: Image,
        eng_w: u32,
        eng_h: u32,
        params: &DetailParams,
        seed: i64,
        preview: &gen_core::PreviewSink,
        cancel: &CancelFlag,
    ) -> WorkerResult<Vec<u8>>;
}

// A cached MLX generator is borrowed only for the duration of `with_cached_generator`'s callback.
// Do not inherit the trait-object default `'static` bound here: doing so makes that valid callback
// borrow appear to escape on macOS (E0521).
impl DetailTileRefiner for dyn Generator + '_ {
    fn refine_tile(
        &self,
        tile: Image,
        eng_w: u32,
        eng_h: u32,
        params: &DetailParams,
        seed: i64,
        preview: &gen_core::PreviewSink,
        cancel: &CancelFlag,
    ) -> WorkerResult<Vec<u8>> {
        detail_refine_tile_generator(self, tile, eng_w, eng_h, params, seed, preview, cancel)
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl DetailTileRefiner for runtime_cuda::providers::sdxl::SdxlDetail {
    fn refine_tile(
        &self,
        tile: Image,
        eng_w: u32,
        eng_h: u32,
        params: &DetailParams,
        seed: i64,
        preview: &gen_core::PreviewSink,
        cancel: &CancelFlag,
    ) -> WorkerResult<Vec<u8>> {
        let request = runtime_cuda::providers::sdxl::SdxlDetailRequest {
            prompt: params.prompt.clone(),
            negative: params.negative.clone(),
            width: eng_w,
            height: eng_h,
            steps: params.steps as usize,
            guidance: params.guidance,
            strength: params.strength,
            control_scale: params.cn_scale,
            seed: seed as u64,
            cancel: cancel.clone(),
            preview: preview.clone(),
        };
        self.generate(&request, &tile, &tile, &mut |_| {})
            .map(|image| image.pixels)
            .map_err(|error| WorkerError::Engine(format!("detail tile failed: {error}")))
    }
}

/// Refine one tile (already sized to engine-valid `eng_w`×`eng_h`): img2img on the tile
/// with the tile as the ControlNet image (control=same). Returns the refined RGB8 buffer.
#[allow(clippy::too_many_arguments)]
fn detail_refine_tile_generator(
    generator: &(impl Generator + ?Sized),
    tile: Image,
    eng_w: u32,
    eng_h: u32,
    params: &DetailParams,
    seed: i64,
    preview: &gen_core::PreviewSink,
    cancel: &CancelFlag,
) -> WorkerResult<Vec<u8>> {
    let mut noop = |_progress: Progress| {};
    let request = GenerationRequest {
        prompt: params.prompt.clone(),
        negative_prompt: Some(params.negative.clone()),
        width: eng_w,
        height: eng_h,
        count: 1,
        seed: Some(seed as u64),
        steps: Some(params.steps),
        guidance: Some(params.guidance),
        conditioning: vec![
            Conditioning::Reference {
                image: tile.clone(),
                strength: Some(params.strength),
            },
            Conditioning::Control {
                image: tile,
                kind: ControlKind::Other("tile".to_owned()),
                // gen-core drift (sc-9940): scale is now Option<f32>; explicit tile-CN scale → Some.
                scale: Some(params.cn_scale),
            },
        ],
        preview: preview.clone(),
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = generator
        .generate(&request, &mut noop)
        .map_err(|error| WorkerError::Engine(format!("detail tile failed: {error}")))?;
    match output {
        GenerationOutput::Images(mut images) => Ok(images
            .pop()
            .ok_or_else(|| WorkerError::Engine("detail tile produced no image".to_owned()))?
            .pixels),
        _ => Err(WorkerError::Engine(
            "detail tile returned non-image output".to_owned(),
        )),
    }
}

/// Tiled feathered detail refine (parity with Python `_refine_tiled_detail`). Returns the
/// recomposed image + the tile count. Runs on the blocking thread (the generator is `!Send`).
fn refine_tiled_detail(
    generator: &(impl DetailTileRefiner + ?Sized),
    source: &image::RgbImage,
    params: &DetailParams,
    preview: &gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_tile: &mut dyn FnMut(usize, usize),
) -> WorkerResult<(image::RgbImage, usize)> {
    use image::imageops::FilterType::Lanczos3;
    let (width, height) = (source.width(), source.height());
    let step = params.tile.saturating_sub(params.overlap).max(1);
    let xs: Vec<u32> = (0..width.saturating_sub(params.overlap).max(1))
        .step_by(step as usize)
        .collect();
    let ys: Vec<u32> = (0..height.saturating_sub(params.overlap).max(1))
        .step_by(step as usize)
        .collect();
    let total = xs.len() * ys.len();
    let mut acc = vec![0.0f32; (width * height * 3) as usize];
    let mut wsum = vec![0.0f32; (width * height) as usize];
    let mut done = 0usize;
    for &y in &ys {
        for &x in &xs {
            if cancel.is_cancelled() {
                return Err(WorkerError::Canceled(
                    "Detail enhancement canceled by user.".to_owned(),
                ));
            }
            let x0 = x.min(width.saturating_sub(params.tile));
            let y0 = y.min(height.saturating_sub(params.tile));
            let tile_w = params.tile.min(width - x0);
            let tile_h = params.tile.min(height - y0);
            let crop = image::imageops::crop_imm(source, x0, y0, tile_w, tile_h).to_image();
            // Run at an engine-valid size (mult-8, ≥512), then resize the refined tile back.
            let (eng_w, eng_h) = (engine_dim(tile_w), engine_dim(tile_h));
            let eng_crop = if (eng_w, eng_h) == (tile_w, tile_h) {
                crop
            } else {
                image::imageops::resize(&crop, eng_w, eng_h, Lanczos3)
            };
            let tile_img = Image {
                width: eng_w,
                height: eng_h,
                pixels: eng_crop.into_raw(),
            };
            let refined_px = generator.refine_tile(
                tile_img,
                eng_w,
                eng_h,
                params,
                params.seed + done as i64,
                preview,
                cancel,
            )?;
            let refined = image::RgbImage::from_raw(eng_w, eng_h, refined_px).ok_or_else(|| {
                WorkerError::InvalidPayload("detail refined tile size mismatch".to_owned())
            })?;
            let refined = if (eng_w, eng_h) == (tile_w, tile_h) {
                refined
            } else {
                image::imageops::resize(&refined, tile_w, tile_h, Lanczos3)
            };
            let feather = detail_feather(tile_w, tile_h, params.overlap);
            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    let f = feather[(ty * tile_w + tx) as usize];
                    let src = refined.get_pixel(tx, ty).0;
                    let gx = x0 + tx;
                    let gy = y0 + ty;
                    let acc_base = ((gy * width + gx) * 3) as usize;
                    acc[acc_base] += src[0] as f32 * f;
                    acc[acc_base + 1] += src[1] as f32 * f;
                    acc[acc_base + 2] += src[2] as f32 * f;
                    wsum[(gy * width + gx) as usize] += f;
                }
            }
            done += 1;
            on_tile(done, total);
        }
    }
    Ok((compose_feathered(&acc, &wsum, width, height), total))
}

/// Normalize the feather-weighted accumulator back to an RGB8 image.
///
/// Each pixel is the weighted mean `acc / wsum` of every tile that covered it. The divisor
/// MUST be the true accumulated feather weight: a pixel on the IMAGE boundary is covered by a
/// single edge tile whose raised-cosine feather ramps toward ~0 over the `overlap`-wide border
/// (there is no neighboring tile to sum the partition-of-unity back to 1). A previous
/// `.max(1.0)` guard divided those border pixels by 1.0 while `acc = src * f` (f→0), stamping a
/// dark rounded-corner vignette — most of the frame in the common single-tile case. Guard only
/// against a literal divide-by-zero; every pixel is covered by ≥1 tile because the tile origins
/// are clamped to the boundary, so `wsum` is strictly positive in practice (sc-8229).
fn compose_feathered(acc: &[f32], wsum: &[f32], width: u32, height: u32) -> image::RgbImage {
    let mut out = image::RgbImage::new(width, height);
    for gy in 0..height {
        for gx in 0..width {
            let accumulated_weight = wsum[(gy * width + gx) as usize];
            let w = if accumulated_weight > 0.0 {
                accumulated_weight
            } else {
                1.0
            };
            let base = ((gy * width + gx) * 3) as usize;
            out.put_pixel(
                gx,
                gy,
                image::Rgb([
                    (acc[base] / w).clamp(0.0, 255.0) as u8,
                    (acc[base + 1] / w).clamp(0.0, 255.0) as u8,
                    (acc[base + 2] / w).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    out
}

/// Build the detail child-asset fact (lineage to the source) + generation set, matching the
/// Python `run_image_detail` result shape so `persist_reported_assets` indexes it identically.
#[allow(clippy::too_many_arguments)]
fn detail_result(
    request: &ImageRequest,
    genset_id: &str,
    created_at: &str,
    asset_id: &str,
    media_rel: &str,
    model: &str,
    params: &DetailParams,
    tiles: usize,
    width: u32,
    height: u32,
    adapter: &str,
) -> JsonObject {
    let source_asset_id = request.source_asset_id.clone().unwrap_or_default();
    let detail_settings = json!({
        "enabled": true,
        "backbone": model,
        "controlNet": TILE_CONTROLNET_REPO,
        "strength": params.strength,
        "cnScale": params.cn_scale,
        "steps": params.steps,
        "guidanceScale": params.guidance,
        "tile": params.tile,
        "overlap": params.overlap,
        "tiles": tiles,
        "width": width,
        "height": height,
    });
    let fact = json!({
        "assetId": asset_id,
        "mediaPath": media_rel,
        "mimeType": "image/png",
        "type": "image",
        "width": width,
        "height": height,
        "normalizedWidth": width,
        "normalizedHeight": height,
        "count": 1,
        "seed": params.seed,
        "displayName": "Detail enhanced",
        "createdAt": created_at,
        "mode": "image_detail",
        "model": model,
        "adapter": adapter,
        "prompt": params.prompt,
        "negativePrompt": params.negative,
        "loras": [],
        "stylePreset": "",
        "sourceAssetId": source_asset_id,
        "rawAdapterSettings": { "detail": detail_settings, "realModelInference": true },
        "parents": [source_asset_id],
        "extra": {
            "isDetailEnhanced": true,
            "detailFromAssetId": source_asset_id,
            "backbone": model,
            "strength": params.strength,
            "cnScale": params.cn_scale,
        },
    });
    let generation_set = json!({
        "id": genset_id,
        "mode": "image_detail",
        "model": model,
        "prompt": params.prompt,
        "negativePrompt": params.negative,
        "count": 1,
        "createdAt": created_at,
    });
    json!({
        "generationSetId": genset_id,
        "expectedCount": 1,
        "adapter": adapter,
        "model": model,
        "generationSet": generation_set,
        "assetWrites": [fact],
    })
    .as_object()
    .cloned()
    .expect("json! object literal")
}

/// Acknowledge a user cancel mid-refine: trip the engine flag and show a NON-terminal
/// "Cancelling…" (indeterminate bar). The terminal Canceled is deferred to after the
/// blocking refinement actually stops so the worker row isn't freed while it's still
/// grinding the current tile (sc-5516; mirrors the image path's `begin_image_cancel`,
/// sc-5515). Best-effort update.
pub(crate) async fn begin_detail_cancel(
    api: &ApiClient,
    job_id: &str,
    cancel: &CancelFlag,
    backend: &str,
) {
    cancel.cancel();
    let _ = update_job(
        api,
        job_id,
        image_progress(
            JobStatus::Running,
            ProgressStage::Generating,
            0.0,
            "Cancelling — stopping the current tile…",
            None,
            backend,
        ),
    )
    .await;
}

/// Native MLX tile-ControlNet detail refine (`JobType::ImageDetail`) on the macOS engine.
pub(crate) async fn run_image_detail_job(
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
    let model = if request.model.trim().is_empty() {
        "realvisxl".to_owned()
    } else {
        request.model.clone()
    };
    let engine_model = mlx_model(&model)
        .filter(|entry| entry.engine_id() == "sdxl")
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("{model} does not support detail enhancement."))
        })?;
    let source_id = request
        .source_asset_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Detail-enhance jobs require a source image asset.".to_owned(),
            )
        })?
        .to_owned();

    let project =
        ProjectStore::new(settings.data_dir.clone(), "worker").get_project(&request.project_id)?;
    let project_path = PathBuf::from(project.path);
    let genset_id = format!("genset_{}", Uuid::new_v4().simple());
    tokio::fs::create_dir_all(project_path.join("assets").join("images").join(&genset_id)).await?;
    let backend = backend_label(&settings.gpu_id);

    let params = resolve_detail_params(&request);
    // Reuse the model's manifest/modelPath/cache resolution; engine_model gives the default repo.
    let weights_dir = resolve_weights_dir(&request, settings)?
        .or_else(|| huggingface_snapshot_dir(&settings.data_dir, engine_model.default_repo()));
    let weights_dir = weights_dir
        .ok_or_else(|| WorkerError::InvalidPayload("SDXL detail weights not found".to_owned()))?;
    // Resolved AFTER `weights_dir` (sc-11042): the tier dir is an input to the quant resolution, so the
    // NVFP4 tier can only be picked when it is the tier that actually resolved. Pure reorder — the
    // q4/q8/bf16 mapping reads only the request and is unaffected.
    let (quant, _) = resolve_quant(&request, Some(&weights_dir));
    let adapters = resolve_adapters(&request, settings)?;
    let control_repo = advanced::str(
        &request.advanced,
        "tileControlNetRepo",
        TILE_CONTROLNET_REPO,
    );
    let control_dir =
        huggingface_snapshot_dir(&settings.data_dir, &control_repo).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "tile ControlNet weights not found (download {control_repo})."
            ))
        })?;
    let control_file = first_safetensors_path(&control_dir).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "no .safetensors under the tile ControlNet snapshot {}",
            control_dir.display()
        ))
    })?;

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let candle_paths = {
        // Resolve the exact three descriptor-declared SDXL co-requisites from the Model Manager's
        // installed cache. Detail jobs must never create an ad-hoc cache or download model weights;
        // a packed fallback or missing component fails here, before provider load, with an
        // actionable error.
        let (tokenizer_clip_l, tokenizer_clip_bigg, vae_fp16_fix) =
            resolve_candle_detail_components(
                &request,
                settings,
                &weights_dir,
                quant.is_some(),
            )?;
        let mut admission_paths = vec![
            weights_dir.as_path(),
            control_file.as_path(),
            crate::conditioning_fit::weights_source_path(&vae_fp16_fix),
        ];
        admission_paths.extend(adapters.iter().map(|adapter| adapter.path.as_path()));
        admit_candle_base_floor(
            &model,
            "SDXL detail",
            settings,
            &admission_paths,
        )
        .await?;
        runtime_cuda::providers::sdxl::SdxlDetailPaths {
            sdxl_base: weights_dir.clone(),
            tokenizer_clip_l,
            tokenizer_clip_bigg,
            vae_fp16_fix,
            tile_controlnet: WeightsSource::File(control_file.clone()),
            adapters: adapters.clone(),
        }
    };

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.1,
            "Loading source image.",
            None,
            backend,
        ),
    )
    .await?;

    let source = engine_image_to_rgb(load_reference_image(
        &settings.data_dir,
        &request.project_id,
        &source_id,
        &project_path,
    )?)?;

    let created_at = now_rfc3339();
    let asset_id = fresh_asset_id();
    let filename = format!("{}_detail_{}.png", &created_at[..10], &asset_id[6..14]);
    let media_rel = format!("assets/images/{genset_id}/{filename}");

    let cancel = CancelFlag::new();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DetailEvent>(64);
    let blocking = {
        let params_ref = params.clone();
        let cancel = cancel.clone();
        #[cfg(target_os = "macos")]
        let blocking = {
            let spec = detail_spec(weights_dir, control_file, quant, adapters);
            tokio::spawn(async move {
            crate::generator_cache::with_cached_generator(
                "sdxl",
                spec,
                "sdxl detail load failed",
                move |generator| {
                    let preview_tx = tx.clone();
                    let preview = gen_core::PreviewSink::new(move |frame| {
                        let _ = preview_tx.try_send(DetailEvent::Preview(frame));
                    });
                    let mut on_tile = |done: usize, total: usize| {
                        // A closed channel means the consumer loop returned early (POST failure /
                        // 409); trip the engine flag so refinement bails instead of grinding
                        // unheard (sc-8804, F-003 — the swallowed-closed-channel leak).
                        if tx.blocking_send(DetailEvent::Tile { done, total }).is_err() {
                            cancel.cancel();
                        }
                    };
                    refine_tiled_detail(
                        generator,
                        &source,
                        &params_ref,
                        &preview,
                        &cancel,
                        &mut on_tile,
                    )
                },
            )
            .await
            })
        };
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let blocking = tokio::task::spawn_blocking(move || {
            let detail = runtime_cuda::providers::sdxl::SdxlDetail::load(&candle_paths)
                .map_err(|error| WorkerError::Engine(format!("SDXL detail load failed: {error}")))?;
            let preview_tx = tx.clone();
            let preview = gen_core::PreviewSink::new(move |frame| {
                let _ = preview_tx.try_send(DetailEvent::Preview(frame));
            });
            let mut on_tile = |done: usize, total: usize| {
                if tx.blocking_send(DetailEvent::Tile { done, total }).is_err() {
                    cancel.cancel();
                }
            };
            refine_tiled_detail(
                &detail,
                &source,
                &params_ref,
                &preview,
                &cancel,
                &mut on_tile,
            )
        });
        blocking
    };

    // Bind the blocking detail-refine task to its cancel flag (sc-8804, F-003): the `update_job`/
    // `heartbeat` `?` in the loop below returns early on a transient POST failure or a 409
    // (stale-sweep reclaim); on that early return this guard trips the engine `CancelFlag` and
    // aborts the still-running tiled refinement instead of leaking it alongside the next claimed
    // job. `cancel` is kept alongside (it's `Clone`) for the in-loop poller.
    let mut guard = CancelJoinGuard::new(cancel.clone(), blocking);
    let mut last_cancel_check = Instant::now();
    let mut canceled = false;
    let mut latest_preview: Option<PreviewSlot> = None;
    // Heartbeat + cancel-poll on a fixed interval, not only when the blocking thread
    // finishes a tile. The cold SDXL+tile-ControlNet load and each full multi-step tile
    // refine emit nothing, so without an interval arm the worker posts no Busy heartbeat
    // (the API's staleness sweep would falsely mark the job `interrupted` — sc-4276 /
    // sc-8390) and a user cancel would only be seen at a tile boundary. Mirrors
    // `consume_gen_events` (base.rs); the shared `CancelFlag` is also inside the engine's
    // `GenerationRequest`, so tripping it here interrupts the in-flight tile promptly
    // rather than waiting for the tile-loop boundary check.
    let mut interval = tokio::time::interval(progress_report_interval(settings));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Run the tile loop capturing its Result so any `?`-error path performs the explicit awaited
    // bounded-join teardown BEFORE returning, instead of drop-and-run (sc-8804, F-003).
    let loop_result: WorkerResult<()> = async {
        loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else {
                    break;
                };
                if canceled {
                    continue; // drain so the blocking sender never blocks; terminal posts after stop.
                }
                let (done, total) = match event {
                    DetailEvent::Tile { done, total } => (done, total),
                    DetailEvent::Preview(frame) => {
                        // Match the shared image stream's latest-wins preview contract. Encoding stays
                        // off the async runtime; the data URL rides the next tile progress POST.
                        let encoded = tokio::task::spawn_blocking(move || {
                            let data_url = encode_preview_data_url(&frame)?;
                            Some(PreviewSlot {
                                index: 0,
                                current: frame.current,
                                total: frame.total,
                                data_url,
                            })
                        })
                        .await
                        .map_err(|error| {
                            crate::task_join_error("detail preview encode task", error)
                        })?;
                        if let Some(slot) = encoded {
                            latest_preview = Some(slot);
                        }
                        continue;
                    }
                };
                // sc-9618: a process shutdown is a cancel checkpoint too — short-circuit the API poll
                // so a quit stops the tile pass at this step, matching a user cancel.
                if shutdown_requested()
                    || (last_cancel_check.elapsed() >= Duration::from_secs(2) && {
                        last_cancel_check = Instant::now();
                        cancel_requested_peek(api, &job.id).await
                    })
                {
                    begin_detail_cancel(api, &job.id, &cancel, backend).await;
                    canceled = true;
                    continue;
                }
                update_job(
                    api,
                    &job.id,
                    image_progress(
                        JobStatus::Running,
                        ProgressStage::Generating,
                        0.45 + 0.5 * (done as f64 / total.max(1) as f64),
                        &format!("Refining detail tile {done}/{total}."),
                        latest_preview.as_ref().map(|slot| {
                            json!({
                                "previewFrame": {
                                    "imageIndex": slot.index,
                                    "current": slot.current,
                                    "total": slot.total,
                                    "dataUrl": slot.data_url,
                                }
                            })
                            .as_object()
                            .cloned()
                            .expect("detail preview result is an object")
                        }),
                        backend,
                    ),
                )
                .await?;
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
            }
            _ = interval.tick() => {
                heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
                // sc-9618: honor a process shutdown on every tick (local flag read, unthrottled).
                if !canceled && (shutdown_requested()
                    || (last_cancel_check.elapsed() >= Duration::from_secs(2) && {
                        last_cancel_check = Instant::now();
                        cancel_requested_peek(api, &job.id).await
                    }))
                {
                    begin_detail_cancel(api, &job.id, &cancel, backend).await;
                    canceled = true;
                }
            }
        }
        }
        Ok(())
    }
    .await;
    if let Err(error) = loop_result {
        guard.cancel_and_join().await;
        return Err(error);
    }

    // Loop exited cleanly — reclaim the handle (disarming the drop-guard) and join the finished task.
    let join = guard
        .into_handle()
        .await
        .map_err(|error| task_join_error("detail task join", error))?;
    if canceled {
        // Refinement has actually stopped now — post the TERMINAL Canceled here. This terminal
        // write frees the worker row (`jobs_store::update_job_progress`) exactly as the worker
        // returns to its claim loop, so the next queued job waits only until the GPU is genuinely
        // free (sc-5516). The engine's own early return (`join`) is discarded as the clean cancel.
        let message = "Detail enhancement canceled by user.";
        update_job(
            api,
            &job.id,
            image_progress(
                JobStatus::Canceled,
                ProgressStage::Canceled,
                1.0,
                message,
                None,
                backend,
            ),
        )
        .await?;
        return Err(WorkerError::Canceled(message.to_owned()));
    }
    let (refined, tiles) = join?;
    let (out_w, out_h) = (refined.width(), refined.height());
    let media_path = project_path.join(&media_rel);
    let temp_path = media_path.with_extension("tmp.png");
    // This pass is its own job with its own payload, so the embedded workflow describes the REFINE
    // (its own prompt, backbone, seed, strength and cnScale) and inherits nothing from the
    // generation that produced the source image — see `detail_workflow_share` (sc-15948).
    let share = workflow_source(settings, &job.payload).and_then(|payload| {
        detail_workflow_share(
            &payload,
            &model,
            &params.prompt,
            &params.negative,
            params.seed,
            out_w,
            out_h,
        )
    });
    // Encode + atomically promote the refined PNG off the async runtime thread (sc-8909 / F-107).
    let encode_tmp = temp_path.clone();
    let encode_final = media_path.clone();
    tokio::task::spawn_blocking(move || {
        write_workflow_chunk(&refined, &encode_tmp, share.as_ref())
            .map_err(|error| WorkerError::Io(std::io::Error::other(error)))?;
        std::fs::rename(&encode_tmp, &encode_final).inspect_err(|_| {
            let _ = std::fs::remove_file(&encode_tmp);
        })?;
        Ok::<_, WorkerError>(())
    })
    .await
    .map_err(|error| crate::task_join_error("detail asset encode task", error))??;

    let result = detail_result(
        &request,
        &genset_id,
        &created_at,
        &asset_id,
        &media_rel,
        &model,
        &params,
        tiles,
        out_w,
        out_h,
        if cfg!(target_os = "macos") {
            "mlx_sdxl"
        } else {
            "candle_sdxl_detail"
        },
    );
    update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Detail enhancement complete.",
            Some(result),
            backend,
        ),
    )
    .await?;
    Ok(())
}
