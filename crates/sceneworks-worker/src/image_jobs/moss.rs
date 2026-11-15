use super::adapters::resolve_prepared_adapters;
use super::events::consume_gen_events;
use super::lane_controls;
use super::load_spec::attach_required_components;
use super::moss_input_parents;
use super::resolve_seed;
#[cfg(test)]
use super::run_image_generate_job;
#[cfg(test)]
use super::run_image_generate_job_unexplained;
use super::sampling::mlx_raw_settings;
use super::stream::preview_sink_for;
use super::stream::send_gen_progress;
use super::stream::send_generated_image;
use super::stream::start_cached_gen_stream_with_request_state_using_owned;
use super::stream::CachedGeneratorFactory;
use super::stream::GenEvent;
#[cfg(test)]
use super::streaming_result;
#[cfg(test)]
use super::write_image_asset;
use super::Arc;
use super::GenerationOutput;
use super::GenerationRequest;
use super::Generator;
use super::Image;
use super::ImagePlan;
use super::ImageRequest;
use super::LoadSpec;
use super::Progress;
use super::WeightsSource;
use super::WorkflowAssetFacts;
use crate::huggingface_pinned_snapshot_dir;
use crate::safe_join;
use crate::safe_project_path;
use crate::ApiClient;
use crate::JobSnapshot;
use crate::JsonObject;
use crate::Path;
#[cfg(test)]
use crate::PathBuf;
use crate::ProjectStore;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
use crate::{fresh_asset_id, now_rfc3339};
use serde_json::json;
// MOSS uses the generic image stream/cache transaction, with its own exact author admission.
// The same function is reached from the MLX and Candle `_using` loader-injection seams.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda::providers::moss_image::memory_strategy as moss_memory;
#[cfg(target_os = "macos")]
use runtime_macos::providers::moss_image::memory_strategy as moss_memory;

fn moss_live_budget(settings: &Settings) -> WorkerResult<gen_core::MemoryBudget> {
    #[cfg(target_os = "macos")]
    {
        let _ = settings;
        crate::mlx_fit_gate::live_request_budget("moss_image_sft")
    }
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    {
        let budget = crate::vram_gate::live_capped_budget_fresh_blocking(&settings.gpu_id)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "MOSS-Image device memory budget unavailable; refusing an unbounded load"
                        .into(),
                )
            })?;
        let bytes = |n: f64| (n * 1024.0 * 1024.0 * 1024.0).max(0.0) as u64;
        Ok(gen_core::MemoryBudget {
            total_bytes: bytes(budget.total_gb),
            committed_bytes: bytes((budget.total_gb - budget.free_gb).max(0.0)),
            reclaimable_bytes: 0,
            reserved_headroom_bytes: bytes(crate::vram_gate::HEADROOM_GB),
        })
    }
}
fn moss_admit(peak: u64, owned: u64, budget: gen_core::MemoryBudget) -> WorkerResult<()> {
    // Credit belongs only to this exact cache entry: measured load ownership or its actual
    // retained native census, bounded by current committed bytes. Source estimates earn no credit.
    let incremental = peak.saturating_sub(owned.min(budget.committed_bytes));
    if incremental > budget.effective_bytes() {
        return Err(WorkerError::InvalidPayload(format!("MOSS-Image selected precision/context/batch needs {incremental} additional bytes; {} available. Reduce dimensions, image count or context, select the smaller precision, or free device memory.", budget.effective_bytes())));
    }
    Ok(())
}
/// The actual worker factory boundary: a cold estimate receives no residency credit and must
/// pass before the provider is allowed to construct or materialize any weights.
fn moss_load_admitted<T>(
    peak: u64,
    budget: gen_core::MemoryBudget,
    load: impl FnOnce() -> gen_core::Result<T>,
) -> gen_core::Result<T> {
    moss_admit(peak, 0, budget).map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
    load()
}
fn moss_execution_asset_selected(
    mode: gen_core::moss_image::MossImageAcceleration,
    repo: &str,
    path: &str,
) -> bool {
    !(mode
        == gen_core::moss_image::MossImageAcceleration::Dmd(
            gen_core::moss_image::MossDmdPrecision::FullFp32,
        )
        && repo == gen_core::moss_image::SFT_REPO
        && path.starts_with("transformer/"))
}
fn moss_owned_weight_bytes(generator: &dyn Generator, measured_load_delta: u64) -> u64 {
    // A lazy renderer can materialize after the loader delta was measured. The current provider
    // census describes only this serialized cache entry's actually retained weights.
    measured_load_delta.max(generator.moss_image_owned_weight_bytes().unwrap_or(0))
}
/// Pins every selected MOSS weight file. `on_file(done, total)` fires before each pin so a cold
/// seal (a full SHA-256 of a multi-GB file, sc-26610) is visible as progress, not a static stage.
/// `adapter_pins` (sc-26587) chains the LoRA/LoKR adapter files the training lane resolved into the
/// same load spec, so a resumed or freshly-trained adapter is identity-checked exactly like a model
/// component instead of loading unverified.
fn moss_pin_spec_with_adapters(
    spec: &mut LoadSpec,
    entry: &JsonObject,
    settings: &Settings,
    mode: gen_core::moss_image::MossImageAcceleration,
    adapter_pins: impl IntoIterator<Item = gen_core::PinnedWeightsFile>,
    on_file: &mut dyn FnMut(usize, usize),
    check: &mut dyn FnMut() -> gen_core::Result<()>,
) -> WorkerResult<()> {
    let mut targets = Vec::new();
    for download in entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let repo = download
            .get("repo")
            .and_then(Value::as_str)
            .ok_or_else(|| WorkerError::InvalidPayload("MOSS component repo missing".into()))?;
        let revision = download
            .get("revision")
            .and_then(Value::as_str)
            .ok_or_else(|| WorkerError::InvalidPayload("MOSS component revision missing".into()))?;
        let root = huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision).ok_or_else(|| WorkerError::InvalidPayload(format!("MOSS component snapshot {repo}@{revision} missing; install or repair the selected package in Models")))?;
        for relative in download
            .get("files")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let edit_asset = repo == gen_core::moss_image_edit::EDIT_REPOSITORY
                && revision == gen_core::moss_image_edit::EDIT_REVISION
                && ["adapter_config.json", "adapter_model.safetensors"].contains(&relative);
            if !edit_asset
                && gen_core::moss_image::moss_image_asset(repo, revision, relative).is_none()
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "MOSS component file is not in the frozen asset contract: {repo}/{relative}"
                )));
            }
            if !moss_execution_asset_selected(mode, repo, relative) {
                continue;
            }
            targets.push((repo, revision, relative, safe_join(&root, relative)?));
        }
    }
    let total = targets.len();
    let mut pins = Vec::with_capacity(total);
    for (done, (repo, revision, relative, path)) in targets.into_iter().enumerate() {
        on_file(done, total);
        let pin = gen_core::PinnedWeightsFile::pin_with_check(&path, &mut *check).map_err(|error| crate::classify_engine_error(&format!("MOSS component {relative} missing/incomplete; install or repair the selected package"), error))?;
        if repo == gen_core::moss_image_edit::EDIT_REPOSITORY {
            let expected = if relative == "adapter_config.json" {
                gen_core::moss_image_edit::EDIT_CONFIG_SHA256
            } else {
                gen_core::moss_image_edit::EDIT_WEIGHTS_SHA256
            };
            if sceneworks_core::hashing::hex(pin.content_sha256()) != expected {
                return Err(WorkerError::InvalidPayload(format!(
                    "MOSS edit frozen component identity mismatch: {relative}"
                )));
            }
        } else {
            gen_core::moss_image::verify_moss_image_asset(&pin, repo, revision, relative).map_err(
                |error| crate::classify_engine_error("MOSS frozen component identity", error),
            )?;
        }
        pins.push(pin);
    }
    crate::paths::prepare_load_spec_with_file_pins(
        spec,
        pins.into_iter().chain(adapter_pins),
        "MOSS component preparation",
    )
}
#[cfg(test)]
fn moss_pin_spec(
    spec: &mut LoadSpec,
    entry: &JsonObject,
    settings: &Settings,
    mode: gen_core::moss_image::MossImageAcceleration,
    on_file: &mut dyn FnMut(usize, usize),
) -> WorkerResult<()> {
    moss_pin_spec_with_adapters(
        spec,
        entry,
        settings,
        mode,
        Vec::new(),
        on_file,
        &mut || Ok(()),
    )
}
/// Generation and the actual provider receipt are read while the serialized cache transaction
/// still owns the generator. Failures/cancellation cannot publish a prior successful receipt.
fn moss_render(
    generator: &dyn Generator,
    request: &GenerationRequest,
    adapter_identities: &[lane_controls::AdapterIdentity],
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(Vec<(i64, Image)>, JsonObject)> {
    let output = generator
        .generate(request, on_progress)
        .map_err(|error| crate::classify_engine_error("MOSS generation", error))?;
    if request.cancel.is_cancelled() {
        return Err(WorkerError::Canceled("MOSS generation canceled".into()));
    }
    let receipt = generator.moss_image_execution_report().ok_or_else(|| {
        WorkerError::Engine("MOSS provider completed without its actual execution receipt".into())
    })?;
    let GenerationOutput::Images(images) = output else {
        return Err(WorkerError::Engine(
            "MOSS provider returned non-image output".into(),
        ));
    };
    if receipt.data().samples.len() != images.len() {
        return Err(WorkerError::Engine(
            "MOSS provider receipt/output count mismatch".into(),
        ));
    }
    let images = images
        .into_iter()
        .zip(&receipt.data().samples)
        .map(|(image, sample)| {
            sample.seed.map(|seed| (seed as i64, image)).ok_or_else(|| {
                WorkerError::Engine(
                    "MOSS text request produced a caller-noise receipt without an effective seed"
                        .into(),
                )
            })
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    // The sealed MOSS receipt records the published DMD stack. Custom adapters
    // use the same provider-report/selected-identity join as the generic image lane.
    let mut facts = lane_controls::image_facts(
        &[],
        adapter_identities,
        &generator.adapter_apply_reports(),
        None,
    )
    .unwrap_or_default();
    facts.insert(
        sceneworks_core::moss_image::REPORT_KEY.into(),
        serde_json::to_value(receipt).map_err(|e| WorkerError::Engine(e.to_string()))?,
    );
    Ok((images, facts))
}
fn moss_load_reference(
    data_dir: &Path,
    project_id: &str,
    id: &str,
    project_path: &Path,
) -> WorkerResult<Image> {
    use image::ImageDecoder;
    let asset = ProjectStore::new(data_dir.to_path_buf(), "worker")
        .get_asset(project_id, id)
        .map_err(|e| WorkerError::InvalidPayload(format!("MOSS reference {id}: {e}")))?;
    let mime = asset
        .pointer("/file/mimeType")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if asset.get("type").and_then(Value::as_str) != Some("image")
        || !mime.starts_with("image/")
        || mime == "image/svg+xml"
    {
        return Err(WorkerError::InvalidPayload(format!(
            "MOSS reference {id} must be a project-owned raster image"
        )));
    }
    let rel = asset
        .pointer("/file/path")
        .and_then(Value::as_str)
        .ok_or_else(|| WorkerError::InvalidPayload(format!("MOSS reference {id} media missing")))?;
    let path = crate::safe_project_path(project_path, rel)?;
    let decoded = (|| -> image::ImageResult<image::DynamicImage> {
        let mut decoder = image::ImageReader::open(&path)?
            .with_guessed_format()?
            .into_decoder()?;
        let orientation = decoder.orientation()?;
        let mut image = image::DynamicImage::from_decoder(decoder)?;
        image.apply_orientation(orientation);
        Ok(image)
    })()
    .or_else(|_| crate::image_decode::decode_image_any(&path))
    .map_err(|e| WorkerError::InvalidPayload(format!("MOSS reference {id}: {e}")))?;
    let rgb = decoded.to_rgb8();
    Ok(Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}
fn moss_edit_cold_peak(
    variant: gen_core::moss_image::MossImageVariant,
    spec: &LoadSpec,
    request: &GenerationRequest,
    references: &[Image],
    options: sceneworks_core::moss_image::EditOptions,
    advanced: gen_core::moss_image::MossImageAdvancedInputs<'_>,
) -> WorkerResult<u64> {
    moss_memory::edit_request_peak(
        variant,
        spec,
        request,
        gen_core::moss_image_edit::MossImageEditInputs {
            references,
            reference_max_edge: options.reference_max_edge,
            use_kv_cache: options.use_kv_cache,
            advanced,
        },
    )
    .map_err(|e| crate::classify_engine_error("MOSS edit cold admission", e))
}
fn moss_edit_peak(
    generator: &dyn Generator,
    request: &GenerationRequest,
    references: &[Image],
    options: sceneworks_core::moss_image::EditOptions,
    advanced: gen_core::moss_image::MossImageAdvancedInputs<'_>,
) -> WorkerResult<u64> {
    generator
        .moss_image_edit_request_peak(
            request,
            gen_core::moss_image_edit::MossImageEditInputs {
                references,
                reference_max_edge: options.reference_max_edge,
                use_kv_cache: options.use_kv_cache,
                advanced,
            },
        )
        .map_err(|e| crate::classify_engine_error("MOSS edit admission", e))
}
fn moss_edit_render(
    generator: &dyn Generator,
    request: &GenerationRequest,
    references: &[Image],
    options: sceneworks_core::moss_image::EditOptions,
    advanced: gen_core::moss_image::MossImageAdvancedInputs<'_>,
    adapter_identities: &[lane_controls::AdapterIdentity],
    progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(gen_core::moss_image::MossImageAdvancedOutput, JsonObject)> {
    let output = generator
        .generate_moss_image_edit(
            request,
            gen_core::moss_image_edit::MossImageEditInputs {
                references,
                reference_max_edge: options.reference_max_edge,
                use_kv_cache: options.use_kv_cache,
                advanced,
            },
            progress,
        )
        .map_err(|e| crate::classify_engine_error("MOSS edit", e))?;
    if request.cancel.is_cancelled() {
        return Err(WorkerError::Canceled("MOSS edit canceled".into()));
    }
    let report = generator
        .moss_image_edit_execution_report()
        .ok_or_else(|| {
            WorkerError::Engine("MOSS edit completed without its actual execution receipt".into())
        })?;
    // The provider returns the validated native receipt. Retain the worker boundary's
    // output type/count/shape checks and carry that receipt as provenance.
    if report.base.data().output_kind != advanced.output
        || report.base.data().images_per_prompt != request.count
        || report.reference_max_edge != options.reference_max_edge
        || report.use_kv_cache != options.use_kv_cache
        || report.references.len() != references.len()
        || report
            .references
            .iter()
            .zip(references)
            .any(|(receipt, image)| receipt.original_dimensions != [image.width, image.height])
    {
        return Err(WorkerError::Engine(
            "MOSS edit receipt differs from actual output/reference provenance".into(),
        ));
    }
    let output_count = match &output {
        gen_core::moss_image::MossImageAdvancedOutput::Images(images)
            if advanced.output == gen_core::moss_image::MossImageOutputKind::Images =>
        {
            images.len()
        }
        gen_core::moss_image::MossImageAdvancedOutput::Latents { values, shape }
            if advanced.output == gen_core::moss_image::MossImageOutputKind::Latents =>
        {
            let expected = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d));
            if shape[1..]
                != [
                    128,
                    request.height as usize / 16,
                    request.width as usize / 16,
                ]
                || expected != Some(values.len())
                || !values.iter().all(|v| v.is_finite())
            {
                return Err(WorkerError::Engine(
                    "MOSS native latent shape/finite mismatch".into(),
                ));
            }
            shape[0]
        }
        _ => {
            return Err(WorkerError::Engine(
                "MOSS native output kind differs from authored output".into(),
            ))
        }
    };
    if report.base.data().latent_shape
        != [
            output_count,
            128,
            request.height as usize / 16,
            request.width as usize / 16,
        ]
        || report.base.data().samples.len() != output_count
        || report
            .base
            .data()
            .samples
            .iter()
            .any(|sample| sample.seed.is_none() != advanced.noise.is_some())
    {
        return Err(WorkerError::Engine(
            "MOSS edit receipt/output count or explicit-noise seed mismatch".into(),
        ));
    }
    let mut facts = lane_controls::image_facts(
        &[],
        adapter_identities,
        &generator.adapter_apply_reports(),
        None,
    )
    .unwrap_or_default();
    facts.insert(
        sceneworks_core::moss_image::EDIT_REPORT_KEY.into(),
        serde_json::to_value(report).map_err(|e| WorkerError::Engine(e.to_string()))?,
    );
    Ok((output, facts))
}

fn write_moss_tensor_companion(
    plan: &ImagePlan,
    project: &Path,
    index: usize,
    raw: &JsonObject,
    seed: Option<u64>,
) -> WorkerResult<Value> {
    use sha2::{Digest, Sha256};
    let mut payload = plan.workflow_source.as_deref().cloned().unwrap_or_else(|| {
        JsonObject::from_iter([
            ("model".into(), json!(plan.request.model)),
            ("mode".into(), json!("edit_image")),
            ("prompt".into(), json!(plan.request.prompt)),
            ("count".into(), json!(plan.request.count)),
            ("width".into(), json!(plan.request.width)),
            ("height".into(), json!(plan.request.height)),
        ])
    });
    payload.insert("advanced".into(), Value::Object(raw.clone()));
    payload.insert(
        "referenceAssetIds".into(),
        raw.get("referenceAssetIds")
            .cloned()
            .unwrap_or_else(|| json!([])),
    );
    let share = sceneworks_core::workflow_share::embeddable_workflow_share(
        &WorkflowAssetFacts {
            mode: "edit_image".into(),
            model: plan.request.model.clone(),
            prompt: plan.request.prompt.clone(),
            negative_prompt: plan.request.negative_prompt.clone(),
            seed: seed.unwrap_or_default() as i64,
            width: Some(plan.request.width),
            height: Some(plan.request.height),
        },
        &payload,
    )
    .ok_or_else(|| {
        WorkerError::Engine("MOSS workflow companion exceeds bounded envelope size".into())
    })?;
    let bytes = serde_json::to_vec(&share).map_err(|e| WorkerError::Engine(e.to_string()))?;
    let relative = format!(
        "assets/tensors/{}/{index:04}.sceneworks-workflow.json",
        plan.genset_id
    );
    let path = safe_project_path(project, &relative)?;
    std::fs::write(path, &bytes)?;
    Ok(
        json!({"path":relative,"mimeType":"application/json","sha256":format!("{:x}",Sha256::digest(&bytes))}),
    )
}
fn emit_moss_advanced_outputs(
    tx: &tokio::sync::mpsc::Sender<GenEvent>,
    plan: &ImagePlan,
    project: &Path,
    output: gen_core::moss_image::MossImageAdvancedOutput,
    facts: &JsonObject,
    raw: JsonObject,
    request: &GenerationRequest,
) -> WorkerResult<()> {
    let report: gen_core::moss_image_edit::MossImageEditExecutionReport =
        serde_json::from_value(facts[sceneworks_core::moss_image::EDIT_REPORT_KEY].clone())
            .map_err(|e| WorkerError::Engine(e.to_string()))?;
    let samples = &report.base.data().samples;
    let mut raw = raw;
    raw.extend(facts.clone());
    if request.cancel.is_cancelled() {
        return Err(WorkerError::Canceled("MOSS output canceled".into()));
    }
    match output {
        gen_core::moss_image::MossImageAdvancedOutput::Images(images) => {
            if images.len() != plan.image_count as usize {
                return Err(WorkerError::Engine(
                    "MOSS advanced image count differs from planned outputs".into(),
                ));
            }
            for (index, (image, sample)) in images.into_iter().zip(samples).enumerate() {
                let mut settings = raw.clone();
                settings.insert("mossImageSampleIndex".into(), json!(index));
                tx.blocking_send(GenEvent::ImageFacts {
                    index,
                    facts: settings,
                })
                .map_err(|_| WorkerError::Canceled("image output consumer closed".into()))?;
                tx.blocking_send(GenEvent::Image {
                    index,
                    seed: sample.seed.map(|seed| seed as i64),
                    width: image.width,
                    height: image.height,
                    pixels: image.pixels,
                    face_likeness: None,
                })
                .map_err(|_| WorkerError::Canceled("image output consumer closed".into()))?;
            }
        }
        gen_core::moss_image::MossImageAdvancedOutput::Latents { values, shape } => {
            if shape[0] != plan.image_count as usize {
                return Err(WorkerError::Engine(
                    "MOSS latent sample count differs from planned outputs".into(),
                ));
            }
            let stride = shape[1..].iter().product::<usize>();
            for (index, (values, sample)) in values.chunks_exact(stride).zip(samples).enumerate() {
                let bytes = sceneworks_core::tensor_artifact::encode_f32(
                    &[1, shape[1], shape[2], shape[3]],
                    values,
                )
                .map_err(WorkerError::InvalidPayload)?;
                let relative = format!("assets/tensors/{}/{index:04}.npy", plan.genset_id);
                let path = safe_project_path(project, &relative)?;
                std::fs::create_dir_all(path.parent().expect("tensor directory"))?;
                std::fs::write(&path, &bytes)?;
                let metadata = sceneworks_core::tensor_artifact::inspect(&path)
                    .map_err(WorkerError::InvalidPayload)?;
                let mut settings = raw.clone();
                let reference_ids = sceneworks_core::moss_image::edit_reference_slots(&raw)
                    .map_err(WorkerError::InvalidPayload)?;
                settings.insert("referenceAssetIds".into(), json!(reference_ids));
                settings.insert("mossImageSampleIndex".into(), json!(index));
                let companion =
                    write_moss_tensor_companion(plan, project, index, &settings, sample.seed)?;
                let fact = json!({"assetId":fresh_asset_id(),"type":"tensor","mediaPath":relative,"mimeType":sceneworks_core::tensor_artifact::MIME,
                    "tensor":metadata,"batchIndex":index,"parents":moss_input_parents(&raw)?,"displayName":format!("MOSS sampled latents {}",index+1),"createdAt":now_rfc3339(),
                    "mode":"edit_image","model":plan.request.model,"adapter":plan.adapter,"prompt":plan.request.image_prompt(index),"negativePrompt":plan.request.negative_prompt,
                    "seed":sample.seed,"count":plan.request.count,"normalizedWidth":request.width,"normalizedHeight":request.height,
                    "sourceAssetId":reference_ids.first(),
                    "rawAdapterSettings":settings,"extra":{"workflowCompanion":companion,"mossOutput":{"kind":"sampled_latents","sampleIndex":index,"nativeBatchShape":shape,"artifactSha256":metadata.sha256}}});
                tx.blocking_send(GenEvent::AssetWritten {
                    index,
                    fact: fact.as_object().expect("fact object").clone(),
                })
                .map_err(|_| WorkerError::Canceled("asset output consumer closed".into()))?;
            }
        }
    }
    if request.cancel.is_cancelled() {
        return Err(WorkerError::Canceled("MOSS output canceled".into()));
    }
    Ok(())
}
#[cfg(any(test, feature = "test-support"))]
#[cfg_attr(
    all(test, not(feature = "test-support")),
    expect(
        dead_code,
        reason = "Unit tests inspect pre-load refusals; successful fields are read only by the test-support inspector."
    )
)]
struct MossPreparedPreload {
    native: gen_core::GenerationRequest,
    variant: gen_core::moss_image::MossImageVariant,
    resolved: gen_core::moss_image::MossImageResolved,
    spec: LoadSpec,
    peak: u64,
}

// Metadata-only forecast of the production NativeMossStreamRuntime preparation
// and header demand. No generator/model tensors/device execution is created here.
#[cfg(test)]
fn prepare_moss_preload(
    settings: &Settings,
    payload: &JsonObject,
    request: &ImageRequest,
    on_file: &mut dyn FnMut(usize, usize),
) -> WorkerResult<MossPreparedPreload> {
    prepare_moss_preload_with_check(settings, payload, request, on_file, &mut || Ok(()))
}
#[cfg(any(test, feature = "test-support"))]
fn prepare_moss_preload_with_check(
    settings: &Settings,
    payload: &JsonObject,
    request: &ImageRequest,
    on_file: &mut dyn FnMut(usize, usize),
    check: &mut dyn FnMut() -> gen_core::Result<()>,
) -> WorkerResult<MossPreparedPreload> {
    check().map_err(|e| crate::classify_engine_error("MOSS preload canceled", e))?;
    use gen_core::moss_image::{MossImageResolved, MossImageVariant};
    let native =
        sceneworks_core::moss_image::native_request(payload, &request.model_manifest_entry)
            .map_err(WorkerError::InvalidPayload)?
            .ok_or_else(|| WorkerError::InvalidPayload("missing MOSS model".into()))?;
    let variant = MossImageVariant::from_provider_id(&request.model).expect("validated MOSS id");
    let resolved = MossImageResolved::resolve(variant, &native)
        .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
    let entry = sceneworks_core::moss_image::selected_components(
        &request.model_manifest_entry,
        native.moss_image.as_ref().expect("options"),
    )
    .map_err(WorkerError::InvalidPayload)?;
    let runtime = NativeMossStreamRuntime;
    let spec = runtime.prepare(
        MossPreparation {
            variant,
            mode: resolved.acceleration,
            edit: false,
            entry: &entry,
            settings,
            request,
            check: Some(&mut *check),
        },
        on_file,
    )?;
    let peak = runtime.cold_peak(MossColdRequest {
        variant,
        spec: &spec,
        native: &native,
        references: &[],
        edit_options: Default::default(),
        resolved: &resolved,
        total_count: request.image_count(),
        edit: false,
        advanced: Default::default(),
        check: Some(&mut *check),
    })?;
    Ok(MossPreparedPreload {
        native,
        variant,
        resolved,
        spec,
        peak,
    })
}

/// Read-only test-support inspection of the EXACT production-selected source
/// pins, adapter stack and native header-demand routine. Never loads a model.
#[cfg(feature = "test-support")]
pub(crate) fn moss_generation_preload_for_test(
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<Value> {
    moss_generation_preload_for_test_with_check(settings, job, &mut || Ok(()))
}
#[cfg(feature = "test-support")]
pub(crate) fn moss_generation_preload_for_test_with_check(
    settings: &Settings,
    job: &JobSnapshot,
    check: &mut dyn FnMut() -> gen_core::Result<()>,
) -> WorkerResult<Value> {
    let request = ImageRequest::from_payload(&job.payload);
    let prepared =
        prepare_moss_preload_with_check(settings, &job.payload, &request, &mut |_, _| {}, check)?;
    let provenance = if cfg!(target_os = "macos") {
        "production prepare_moss_preload -> mlx moss_memory::selected_asset_facts.base_bytes + overlay_bytes + request_workspace_bytes (exact selected request/adapter pins)"
    } else {
        "production prepare_moss_preload -> candle moss_memory::selected_preload_estimate.peak_bytes (exact selected request/adapter pins, includes owned-shard staging)"
    };
    let adapters = prepared.spec.adapters.iter().map(|adapter| {
        check().map_err(|e| crate::classify_engine_error("MOSS preload canceled", e))?;
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)
            .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
        let tensor_bytes = headers.iter().try_fold(0_u64, |sum, header| sum.checked_add(header.data_bytes)
            .ok_or_else(|| WorkerError::InvalidPayload("MOSS adapter header bytes overflow".into())))?;
        let pin=gen_core::PinnedWeightsFile::pin_with_check(&adapter.path, &mut *check).map_err(|e|crate::classify_engine_error("MOSS adapter preload canceled", e))?;
        let sha256=sceneworks_core::hashing::hex(pin.content_sha256());
        pin.ensure_unchanged().map_err(|e|WorkerError::InvalidPayload(e.to_string()))?;
        Ok(json!({"path":adapter.path,"scale":adapter.scale,"tensorBytes":tensor_bytes,"sha256":sha256}))
    }).collect::<WorkerResult<Vec<Value>>>()?;
    Ok(json!({
        "kind":"moss-production-generation-preload",
        "requiredBytes":prepared.peak,"provenance":provenance,
        "parent":prepared.variant,"selectedMode":prepared.resolved.acceleration,
        "width":prepared.native.width,"height":prepared.native.height,
        "steps":prepared.resolved.steps,"guidance":prepared.resolved.guidance,
        "adapters":adapters,"weightsLoaded":false,"modelTensorsLoaded":false,
        "deviceExecutionPerformed":false
    }))
}

pub(super) struct MossPreparation<'a> {
    variant: gen_core::moss_image::MossImageVariant,
    mode: gen_core::moss_image::MossImageAcceleration,
    edit: bool,
    entry: &'a JsonObject,
    settings: &'a Settings,
    request: &'a ImageRequest,
    check: Option<&'a mut dyn FnMut() -> gen_core::Result<()>>,
}
pub(super) struct MossColdRequest<'a> {
    variant: gen_core::moss_image::MossImageVariant,
    spec: &'a LoadSpec,
    native: &'a GenerationRequest,
    references: &'a [Image],
    edit_options: sceneworks_core::moss_image::EditOptions,
    resolved: &'a gen_core::moss_image::MossImageResolved,
    total_count: u32,
    edit: bool,
    advanced: gen_core::moss_image::MossImageAdvancedInputs<'a>,
    check: Option<&'a mut dyn FnMut() -> gen_core::Result<()>>,
}
/// Private dependency boundary: production retains exact frozen seals and native admission.
/// Injected CPU providers supply their own explicitly synthetic, pinned source contract.
pub(super) trait MossStreamRuntime: Send + Sync {
    fn prepare(
        &self,
        input: MossPreparation<'_>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> WorkerResult<LoadSpec>;
    fn cold_peak(&self, input: MossColdRequest<'_>) -> WorkerResult<u64>;
    fn budget(&self, settings: &Settings) -> WorkerResult<gen_core::MemoryBudget>;
    fn cache(&self) -> crate::generator_cache::GeneratorCacheHandle {
        crate::generator_cache::GeneratorCacheHandle::global()
    }
}
pub(super) struct NativeMossStreamRuntime;
impl MossStreamRuntime for NativeMossStreamRuntime {
    fn prepare(
        &self,
        input: MossPreparation<'_>,
        progress: &mut dyn FnMut(usize, usize),
    ) -> WorkerResult<LoadSpec> {
        let MossPreparation {
            variant,
            mode,
            edit,
            entry,
            settings,
            request,
            check,
        } = input;
        let mut noop = || Ok(());
        let check = check.unwrap_or(&mut noop);
        check().map_err(|e| crate::classify_engine_error("MOSS preload canceled", e))?;
        let root = huggingface_pinned_snapshot_dir(
            &settings.data_dir,
            variant.repository(),
            variant.revision(),
        )
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{} frozen snapshot missing; install or repair this model in Models",
                variant.provider_id()
            ))
        })?;
        let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(format!(
            "{}:{:?}:edit={edit}",
            variant.provider_id(),
            mode
        ));
        spec = attach_required_components(spec, variant.provider_id(), entry, settings)?;
        for row in entry
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(id) = row.get("componentId").and_then(Value::as_str).filter(|id| {
                id.starts_with("moss-dmd-") || *id == gen_core::moss_image_edit::EDIT_COMPONENT
            }) {
                let source = if crate::portable_aux_sources::has_bound_aux_paths(&request.advanced)
                {
                    let path = crate::portable_aux_sources::bound_aux_path(&request.advanced, id)
                        .ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "portable MOSS component {id} is not bound"
                        ))
                    })?;
                    match sceneworks_core::portable_sources::portable_aux_source_shape(
                        id,
                        &Value::Object(entry.clone()),
                    ) {
                        Some(sceneworks_core::portable_sources::PortableAuxSourceShape::File) => {
                            WeightsSource::File(path)
                        }
                        Some(
                            sceneworks_core::portable_sources::PortableAuxSourceShape::Directory,
                        ) => WeightsSource::Dir(path),
                        None => {
                            return Err(WorkerError::InvalidPayload(format!(
                                "portable MOSS component {id} has no declared source shape"
                            )))
                        }
                    }
                } else {
                    crate::model_jobs::resolve_optional_component(
                        &Value::Object(entry.clone()),
                        id,
                        settings,
                    )
                    .ok_or_else(|| WorkerError::InvalidPayload(format!(
                        "MOSS component {id} missing; install the selected optional package in Models"
                    )))?
                };
                spec = spec.with_component(id, source);
            }
        }
        let prepared_adapters = resolve_prepared_adapters(request, settings)?;
        spec = spec.with_adapters(prepared_adapters.specs);
        moss_pin_spec_with_adapters(
            &mut spec,
            entry,
            settings,
            mode,
            prepared_adapters.pins,
            progress,
            check,
        )?;
        Ok(spec)
    }
    fn cold_peak(&self, input: MossColdRequest<'_>) -> WorkerResult<u64> {
        let MossColdRequest {
            variant,
            spec,
            native,
            references,
            edit_options,
            resolved,
            total_count,
            edit,
            advanced,
            mut check,
        } = input;
        let mut checked = || {
            if native.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            if let Some(check) = check.as_mut() {
                check()?;
            }
            Ok(())
        };
        checked().map_err(|e| crate::classify_engine_error("MOSS preload canceled", e))?;
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let _ = (resolved, total_count);
        #[cfg(target_os = "macos")]
        let peak = if edit {
            moss_edit_cold_peak(variant, spec, native, references, edit_options, advanced)?
        } else {
            let facts = moss_memory::selected_asset_facts_with_check(
                variant,
                spec,
                resolved.acceleration,
                &mut checked,
            )
            .map_err(|e| crate::classify_engine_error("MOSS preload facts failed", e))?;
            let expanded = if native.prompt_batch.is_empty() {
                1
            } else {
                total_count
            };
            let branches =
                if resolved.acceleration == gen_core::moss_image::MossImageAcceleration::Off {
                    1 + u32::from(resolved.guidance != 1.0)
                        + u32::from(
                            resolved
                                .slg
                                .as_ref()
                                .is_some_and(|s| s.scale > 0.0 && !s.layers.is_empty()),
                        )
                } else {
                    1
                };
            let workspace = moss_memory::request_workspace_bytes(
                native.width,
                native.height,
                expanded as usize,
                resolved.max_sequence_length,
                branches,
            )
            .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
            facts
                .base_bytes
                .checked_add(facts.overlay_bytes)
                .and_then(|v| v.checked_add(workspace))
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("MOSS memory estimate overflow".into())
                })?
        };
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let peak = if edit {
            moss_edit_cold_peak(variant, spec, native, references, edit_options, advanced)?
        } else {
            moss_memory::selected_preload_estimate_with_check(
                variant,
                spec,
                native,
                gen_core::moss_image::MossImageAdvancedInputs::default(),
                &mut checked,
            )
            .map_err(|e| crate::classify_engine_error("MOSS preload facts failed", e))?
            .peak_bytes
        };
        checked().map_err(|e| crate::classify_engine_error("MOSS preload canceled", e))?;
        Ok(peak)
    }
    fn budget(&self, settings: &Settings) -> WorkerResult<gen_core::MemoryBudget> {
        moss_live_budget(settings)
    }
}
pub(super) fn moss_adapter_label(model: &str, backend: &str) -> &'static str {
    match (model, backend) {
        ("moss_image_base", "candle") => "candle_moss_image_base",
        ("moss_image_sft", "candle") => "candle_moss_image_sft",
        ("moss_image_base", _) => "mlx_moss_image_base",
        ("moss_image_sft", _) => "mlx_moss_image_sft",
        _ => unreachable!("validated MOSS model"),
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_moss_stream_using(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
) -> WorkerResult<()> {
    generate_moss_stream_with_runtime(
        api,
        settings,
        job,
        plan,
        project_path,
        asset_writes,
        load_generator,
        Arc::new(NativeMossStreamRuntime),
        "mlx",
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_moss_stream_with_runtime(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    asset_writes: &mut Vec<Value>,
    load_generator: impl FnOnce(&str, &LoadSpec) -> gen_core::Result<Box<dyn Generator>>
        + Send
        + 'static,
    runtime: Arc<dyn MossStreamRuntime>,
    backend: &'static str,
) -> WorkerResult<()> {
    use gen_core::moss_image::{MossImageResolved, MossImageVariant};
    let request = &plan.request;
    let mut native =
        sceneworks_core::moss_image::native_request(&job.payload, &request.model_manifest_entry)
            .map_err(WorkerError::InvalidPayload)?
            .ok_or_else(|| WorkerError::InvalidPayload("missing MOSS model".into()))?;
    let edit = sceneworks_core::moss_image::is_edit(&job.payload);
    let advanced = if edit {
        sceneworks_core::moss_image_advanced::OwnedInputs::load(
            &ProjectStore::new(settings.data_dir.clone(), "worker"),
            &job.payload,
            &request.model_manifest_entry,
        )
        .map_err(WorkerError::InvalidPayload)?
    } else {
        Default::default()
    };
    let edit_options = sceneworks_core::moss_image::edit_options(&request.advanced)
        .map_err(WorkerError::InvalidPayload)?;
    let references = if edit {
        sceneworks_core::moss_image::edit_reference_ids(&job.payload)
            .map_err(WorkerError::InvalidPayload)?
            .iter()
            .map(|id| {
                moss_load_reference(&settings.data_dir, &request.project_id, id, project_path)
            })
            .collect::<WorkerResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let variant = MossImageVariant::from_provider_id(&request.model).expect("validated MOSS id");
    let resolved = MossImageResolved::resolve(variant, &native)
        .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
    if edit {
        advanced
            .validate(&native, &resolved)
            .map_err(WorkerError::InvalidPayload)?;
    }
    let entry = sceneworks_core::moss_image::selected_components_for_purpose(
        &request.model_manifest_entry,
        sceneworks_core::moss_image::request_purpose(
            &job.payload,
            native.moss_image.as_ref().expect("options"),
        ),
    )
    .map_err(WorkerError::InvalidPayload)?;
    let prepared = runtime.prepare(
        MossPreparation {
            variant,
            mode: resolved.acceleration,
            edit,
            entry: &entry,
            settings,
            request,
            check: None,
        },
        &mut |_, _| {},
    );
    let spec = prepared?;
    let peak = runtime.cold_peak(MossColdRequest {
        variant,
        spec: &spec,
        native: &native,
        references: &references,
        edit_options,
        resolved: &resolved,
        total_count: request.image_count(),
        edit,
        advanced: advanced.inputs(),
        check: None,
    })?;
    let peak = peak
        .checked_add(advanced.host_bytes as u64)
        .ok_or_else(|| WorkerError::InvalidPayload("MOSS host tensor admission overflow".into()))?;
    let adapter_identities =
        lane_controls::adapter_identities(api, settings, job, request, &spec.adapters).await;
    let settings_owned = settings.clone();
    let cold_settings = settings_owned.clone();
    let cold_runtime = Arc::clone(&runtime);
    let cache = runtime.cache();
    let job_request = request.clone();
    let engine_id = variant.provider_id();
    let count = request.image_count() as usize;
    let mut raw = mlx_raw_settings(
        request,
        variant.repository(),
        resolved.steps,
        None,
        Some(resolved.guidance),
    );
    raw.insert("steps".to_owned(), json!(resolved.steps));
    raw.insert(
        sceneworks_core::moss_image::OPTIONS_KEY.into(),
        serde_json::to_value(native.moss_image.as_ref().expect("options"))
            .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?,
    );
    if edit {
        let carriers = json!({"referenceAssetIds":request.reference_asset_ids,"sourceAssetId":request.source_asset_id,"referenceAssetId":request.reference_asset_id});
        raw.insert(
            "referenceAssetIds".into(),
            json!(sceneworks_core::moss_image::edit_reference_slots(
                carriers.as_object().expect("reference carriers")
            )
            .map_err(WorkerError::InvalidPayload)?),
        );
        raw.insert(
            sceneworks_core::moss_image::EDIT_OPTIONS_KEY.into(),
            serde_json::to_value(edit_options)
                .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?,
        );
    }
    let output_plan = plan.clone();
    let output_path = project_path.to_owned();
    let output_raw = raw.clone();
    let adapter_count = spec.adapters.len();
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using_owned(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        "MOSS exact native load".into(),
        CachedGeneratorFactory {
            cache,
            load: move |id: &str, spec: &LoadSpec| {
                moss_load_admitted(
                    peak,
                    cold_runtime
                        .budget(&cold_settings)
                        .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?,
                    || load_generator(id, spec),
                )
            },
        },
        move |generator, _state, _loaded_policy, warm_policy, _external, owned, tx, cancel| {
            warm_policy
                .decline(crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory);
            native.cancel = cancel.clone();
            let peak = if edit {
                moss_edit_peak(
                    generator,
                    &native,
                    &references,
                    edit_options,
                    advanced.inputs(),
                )?
                .checked_add(advanced.host_bytes as u64)
                .ok_or_else(|| {
                    WorkerError::InvalidPayload("MOSS host tensor admission overflow".into())
                })?
            } else {
                peak
            };
            moss_admit(
                peak,
                moss_owned_weight_bytes(generator, owned),
                runtime.budget(&settings_owned)?,
            )?;
            if edit {
                native.seed = if advanced.noise.is_some() {
                    None
                } else {
                    Some(resolve_seed(&job_request, 0) as u64)
                };
                let mut progress = |p| send_gen_progress(&tx, 0, p);
                let (output, facts) = moss_edit_render(
                    generator,
                    &native,
                    &references,
                    edit_options,
                    advanced.inputs(),
                    &adapter_identities,
                    &mut progress,
                )?;
                emit_moss_advanced_outputs(
                    &tx,
                    &output_plan,
                    &output_path,
                    output,
                    &facts,
                    output_raw.clone(),
                    &native,
                )?;
                return Ok(());
            }
            if native.prompt_batch.is_empty() {
                native.count = 1;
                for index in 0..count {
                    if cancel.is_cancelled() {
                        return Err(WorkerError::Canceled("MOSS generation canceled".into()));
                    }
                    let seed = resolve_seed(&job_request, index);
                    native.seed = Some(seed as u64);
                    let mut progress = |p| send_gen_progress(&tx, index, p);
                    let (images, facts) =
                        moss_render(generator, &native, &adapter_identities, &mut progress)?;
                    if images.len() != 1 {
                        return Err(WorkerError::Engine(
                            "MOSS single-image output count mismatch".into(),
                        ));
                    }
                    let (actual_seed, image) = images.into_iter().next().expect("one image");
                    let _ = tx.blocking_send(GenEvent::ImageFacts { index, facts });
                    if !send_generated_image(
                        &tx,
                        index,
                        (actual_seed, image.width, image.height, image.pixels),
                    ) {
                        break;
                    }
                }
                Ok(())
            } else {
                let seed = resolve_seed(&job_request, 0);
                native.seed = Some(seed as u64);
                let mut progress = |p| send_gen_progress(&tx, 0, p);
                native.preview = preview_sink_for(&tx, 0);
                let (images, facts) =
                    moss_render(generator, &native, &adapter_identities, &mut progress)?;
                if images.len() != count {
                    return Err(WorkerError::Engine(
                        "MOSS batch output count mismatch".into(),
                    ));
                }
                for (index, (actual_seed, image)) in images.into_iter().enumerate() {
                    if cancel.is_cancelled() {
                        return Err(WorkerError::Canceled("MOSS generation canceled".into()));
                    }
                    let _ = tx.blocking_send(GenEvent::ImageFacts {
                        index,
                        facts: facts.clone(),
                    });
                    if !send_generated_image(
                        &tx,
                        index,
                        (actual_seed, image.width, image.height, image.pixels),
                    ) {
                        break;
                    }
                }
                Ok(())
            }
        },
    );
    let adapter = moss_adapter_label(&request.model, backend);
    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter,
        &raw,
        count,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}

#[cfg(test)]
#[path = "tests/moss_consumer_tests.rs"]
mod moss_consumer_tests;
