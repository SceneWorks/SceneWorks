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
        let budget = crate::vram_gate::apply_vram_cap(
            crate::gpu::nvidia_vram_budget_gb_fresh_blocking(&settings.gpu_id),
            crate::vram_gate::cuda_vram_cap_gb(),
        )
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "MOSS-Image device memory budget unavailable; refusing an unbounded load".into(),
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
            if pin
                .content_sha256()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                != expected
            {
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
    crate::moss_terminal_observation::provider_run()?;
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
    crate::moss_terminal_observation::provider_run()?;
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
    report
        .validate()
        .map_err(|e| WorkerError::Engine(e.to_string()))?;
    let variant =
        gen_core::moss_image::MossImageVariant::from_provider_id(generator.descriptor().id)
            .ok_or_else(|| WorkerError::Engine("MOSS edit generator identity is missing".into()))?;
    let expected = gen_core::moss_image::MossImageResolved::resolve(variant, request)
        .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
    let expected_count = advanced
        .validate(request, &expected)
        .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
    if report.base.data().resolved != expected
        || report.base.data().output_kind != advanced.output
        || report.base.data().latent_shape
            != [
                expected_count,
                128,
                request.height as usize / 16,
                request.width as usize / 16,
            ]
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
            "MOSS edit receipt differs from the actual authored request/reference batch".into(),
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
    if report.base.data().samples.len() != output_count
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

fn moss_input_parents(raw: &JsonObject) -> WorkerResult<Vec<String>> {
    let mut parents = sceneworks_core::moss_image::edit_reference_slots(raw)
        .map_err(WorkerError::InvalidPayload)?;
    if let Some(value) = raw.get(sceneworks_core::moss_image_advanced::KEY) {
        let options: sceneworks_core::moss_image_advanced::EditAdvanced =
            serde_json::from_value(value.clone())
                .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
        for reference in sceneworks_core::moss_image_advanced::references(&options) {
            if !parents.contains(&reference.asset_id) {
                parents.push(reference.asset_id.clone());
            }
        }
    }
    Ok(parents)
}
struct MossOutputBatch {
    facts: Vec<JsonObject>,
    paths: Vec<PathBuf>,
    committed: bool,
}
impl MossOutputBatch {
    fn new() -> Self {
        Self {
            facts: Vec::new(),
            paths: Vec::new(),
            committed: false,
        }
    }
    fn stage(&mut self, path: PathBuf, bytes: &[u8]) -> WorkerResult<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        self.paths.push(path);
        file.write_all(bytes)?;
        Ok(())
    }
    fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for MossOutputBatch {
    fn drop(&mut self) {
        if !self.committed {
            for path in &self.paths {
                if let Err(error) = std::fs::remove_file(path) {
                    if error.kind() != std::io::ErrorKind::NotFound {
                        tracing::error!(event="moss_unpublished_output_cleanup_failed",%error,path=%path.display(),"Unpublished MOSS staging could not be removed");
                    }
                }
            }
        }
    }
}
fn moss_stage_image_files(
    plan: &ImagePlan,
    temp: &Path,
    media: &Path,
) -> WorkerResult<MossOutputBatch> {
    let mut files = MossOutputBatch::new();
    if sceneworks_core::moss_image::is_moss(&plan.request.model)
        && plan.request.mode == "edit_image"
    {
        files.stage(temp.to_path_buf(), &[])?;
        files.stage(media.to_path_buf(), &[])?;
        if let Some(batch) = plan.moss_staged.lock().expect("MOSS staging lock").as_mut() {
            batch.paths.extend(files.paths.iter().cloned());
        }
    }
    Ok(files)
}

/// A lost HTTP acknowledgment never licenses deletion of an accepted durable handoff.
async fn publish_moss_batch(
    api: &ApiClient,
    job: &JobSnapshot,
    plan: &ImagePlan,
    backend: &str,
    facts: &[Value],
    mut batch: MossOutputBatch,
) -> WorkerResult<()> {
    check_cancel(
        api,
        &job.id,
        "MOSS edit canceled before output publication.",
    )
    .await?;
    let expected = facts
        .iter()
        .map(|fact| fact.get("assetId").cloned().unwrap_or(Value::Null))
        .collect::<Vec<_>>();
    let matches = |saved: &JobSnapshot| {
        saved.id == job.id
            && saved.project_id == job.project_id
            && saved.attempts == job.attempts
            && saved.payload.get("model").and_then(Value::as_str)
                == Some(plan.request.model.as_str())
            && saved.result.get("generationSetId").and_then(Value::as_str)
                == Some(plan.genset_id.as_str())
            && saved.result.get("assetIds").and_then(Value::as_array) == Some(&expected)
            && saved
                .result
                .get("assets")
                .and_then(Value::as_array)
                .is_some_and(|assets| {
                    assets.len() == facts.len()
                        && assets.iter().zip(facts).all(|(asset, fact)| {
                            asset.get("id") == fact.get("assetId")
                                && asset.pointer("/file/path") == fact.get("mediaPath")
                                && asset.get("projectId").and_then(Value::as_str)
                                    == job.project_id.as_deref()
                                && asset.get("generationSetId").and_then(Value::as_str)
                                    == Some(plan.genset_id.as_str())
                                && asset.pointer("/lineage/jobId").and_then(Value::as_str)
                                    == Some(job.id.as_str())
                        })
                })
    };
    let response = update_job(
        api,
        &job.id,
        image_progress(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("Generated {} outputs.", expected.len()),
            Some(streaming_result(plan, facts)),
            backend,
        ),
    )
    .await;
    match response {
        Ok(saved) if saved.status == JobStatus::Completed && matches(&saved) => {
            batch.commit();
            Ok(())
        }
        response => {
            let authoritative = api
                .get_json::<JobSnapshot>(&format!("/api/v1/jobs/{}", job.id))
                .await;
            match authoritative {
                Ok(saved) if saved.status == JobStatus::Completed && matches(&saved) => {
                    batch.commit();
                    Ok(())
                }
                Ok(saved)
                    if saved.id == job.id
                        && matches!(
                            saved.status,
                            JobStatus::Failed | JobStatus::Canceled | JobStatus::Interrupted
                        ) =>
                {
                    Err(WorkerError::Canceled(format!(
                        "MOSS output publication was rejected for job {}",
                        job.id
                    )))
                }
                _ => {
                    // Keep exact staged media for the existing durable pending-side-effect retry.
                    // No completed output is claimed here; the server's canonical assets decide readiness.
                    batch.commit();
                    let detail = response
                        .err()
                        .map(|error| error.to_string())
                        .unwrap_or_else(|| "canonical output acknowledgment unavailable".into());
                    Err(WorkerError::Engine(format!("MOSS output publication pending or uncertain for job {}: {detail}. Poll this job; do not resubmit generation.",job.id)))
                }
            }
        }
    }
}

fn write_moss_tensor_companion(
    plan: &ImagePlan,
    project: &Path,
    index: usize,
    raw: &JsonObject,
    seed: Option<u64>,
    staged: &mut MossOutputBatch,
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
        "assets/tensors/moss-edit/{}/{index:04}.sceneworks-workflow.json",
        plan.genset_id
    );
    let path = safe_join(project, &relative)?;
    staged.stage(path, &bytes)?;
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
    let mut staged = MossOutputBatch::new();
    if request.cancel.is_cancelled() {
        return Err(WorkerError::Canceled(
            "MOSS output publication canceled".into(),
        ));
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
                let mut fact = write_image_asset(
                    plan,
                    index,
                    sample.seed.map(|n| n as i64),
                    image.width,
                    image.height,
                    image.pixels,
                    plan.adapter,
                    settings,
                    project,
                )?;
                staged.paths.push(safe_join(
                    project,
                    fact["mediaPath"].as_str().expect("written image path"),
                )?);
                fact.insert("batchIndex".into(), json!(index));
                fact.insert("parents".into(), json!(moss_input_parents(&raw)?));
                staged.facts.push(fact);
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
                let relative =
                    format!("assets/tensors/moss-edit/{}/{index:04}.npy", plan.genset_id);
                let path = safe_join(project, &relative)?;
                std::fs::create_dir_all(path.parent().expect("tensor directory"))?;
                staged.stage(path.clone(), &bytes)?;
                let metadata = sceneworks_core::tensor_artifact::inspect(&path)
                    .map_err(WorkerError::InvalidPayload)?;
                let mut settings = raw.clone();
                let reference_ids = sceneworks_core::moss_image::edit_reference_slots(&raw)
                    .map_err(WorkerError::InvalidPayload)?;
                settings.insert("referenceAssetIds".into(), json!(reference_ids));
                settings.insert("mossImageSampleIndex".into(), json!(index));
                let companion = write_moss_tensor_companion(
                    plan,
                    project,
                    index,
                    &settings,
                    sample.seed,
                    &mut staged,
                )?;
                let fact = json!({"assetId":fresh_asset_id(),"type":"tensor","mediaPath":relative,"mimeType":sceneworks_core::tensor_artifact::MIME,
                    "tensor":metadata,"batchIndex":index,"parents":moss_input_parents(&raw)?,"displayName":format!("MOSS sampled latents {}",index+1),"createdAt":now_rfc3339(),
                    "mode":"edit_image","model":plan.request.model,"adapter":plan.adapter,"prompt":plan.request.image_prompt(index),"negativePrompt":plan.request.negative_prompt,
                    "seed":sample.seed,"count":plan.request.count,"normalizedWidth":request.width,"normalizedHeight":request.height,
                    "sourceAssetId":reference_ids.first(),
                    "rawAdapterSettings":settings,"extra":{"workflowCompanion":companion,"mossOutput":{"kind":"sampled_latents","sampleIndex":index,"nativeBatchShape":shape,"artifactSha256":metadata.sha256}}});
                staged
                    .facts
                    .push(fact.as_object().expect("fact object").clone());
            }
        }
    }
    if request.cancel.is_cancelled() {
        return Err(WorkerError::Canceled(
            "MOSS output publication canceled".into(),
        ));
    }
    tx.blocking_send(GenEvent::MossAssets(staged))
        .map_err(|_| WorkerError::Canceled("MOSS output consumer closed".into()))?;
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
        let sha256=pin.content_sha256().iter().map(|byte|format!("{byte:02x}")).collect::<String>();
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

struct MossPreparation<'a> {
    variant: gen_core::moss_image::MossImageVariant,
    mode: gen_core::moss_image::MossImageAcceleration,
    edit: bool,
    entry: &'a JsonObject,
    settings: &'a Settings,
    request: &'a ImageRequest,
    check: Option<&'a mut dyn FnMut() -> gen_core::Result<()>>,
}
struct MossColdRequest<'a> {
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
trait MossStreamRuntime: Send + Sync {
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
struct NativeMossStreamRuntime;
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
                let source = crate::model_jobs::resolve_optional_component(
                    &Value::Object(entry.clone()),
                    id,
                    settings,
                )
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                    "MOSS component {id} missing; install the selected optional package in Models"
                ))
                })?;
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
fn moss_adapter_label(model: &str, backend: &str) -> &'static str {
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
async fn generate_moss_stream_using(
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
async fn generate_moss_stream_with_runtime(
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
    // Pinning blocks this runtime thread, so a spawned task posts the per-file progress.
    let (pin_progress, mut pin_events) = tokio::sync::mpsc::unbounded_channel::<(usize, usize)>();
    let poster = {
        let api = api.clone();
        let job_id = job.id.clone();
        let backend = backend_label(&settings.gpu_id).to_owned();
        tokio::spawn(async move {
            while let Some((done, total)) = pin_events.recv().await {
                let message = format!("Verifying model files ({}/{total}).", done + 1);
                let payload = image_progress(
                    JobStatus::Preparing,
                    ProgressStage::Preparing,
                    0.05,
                    &message,
                    None,
                    &backend,
                );
                if let Err(error) = update_job(&api, &job_id, payload).await {
                    tracing::debug!(
                        event = "moss_pin_progress_post_failed",
                        jobId = %job_id,
                        error = %error,
                        "pin progress post failed; the job's next progress post reports the outcome"
                    );
                }
            }
        })
    };
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
        &mut |done, total| {
            let _ = pin_progress.send((done, total));
        },
    );
    drop(pin_progress);
    let _ = poster.await;
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
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using_owned(
        job.id.clone(),
        engine_id,
        0,
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
mod moss_consumer_tests {
    use super::*;
    use gen_core::moss_image::*;
    use std::cell::{Cell, RefCell};

    // Request metadata only: no model fixture, source payload, device, tensor or
    // native backend invocation. The tests below stop before backend header pricing.
    fn preload_payload_for_pure_test(variant: MossImageVariant, mode: &str) -> JsonObject {
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .unwrap()
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
        let entry = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == variant.provider_id())
            .unwrap();
        json!({
            "model":variant.provider_id(), "modelManifestEntry":entry,
            "prompt":"source-only preload request", "width":256, "height":256,
            "count":1, "advanced":if mode == "off" {
                json!({"steps":8,"guidanceScale":1.0,"mossImage":{"acceleration":mode}})
            } else {
                // DMD is natively no-CFG; this test targets snapshot availability.
                json!({"steps":8,"mossImage":{"acceleration":mode}})
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn checked_preload_refuses_cancellation_before_any_source_lookup() {
        let temp = tempfile::tempdir().unwrap();
        let settings = Settings::for_test(temp.path().join("data"));
        let payload = json!({"model":"not_a_moss_provider","prompt":"metadata only"})
            .as_object()
            .unwrap()
            .clone();
        let request = ImageRequest::from_payload(&payload);
        let mut checks = 0;
        let error = prepare_moss_preload_with_check(
            &settings,
            &payload,
            &request,
            &mut |_, _| panic!("source lookup reached"),
            &mut || {
                checks += 1;
                Err(gen_core::Error::Canceled)
            },
        )
        .err()
        .expect("cancellation must refuse");
        assert!(matches!(&error, WorkerError::Canceled(_)), "{error:?}");
        assert_eq!(checks, 1);
    }

    #[test]
    fn prepared_preload_refuses_non_moss_before_source_lookup() {
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let settings = Settings::for_test(temp.path().join("data"));
        let payload = json!({"model":"not_a_moss_provider","prompt":"metadata only"})
            .as_object()
            .unwrap()
            .clone();
        let request = ImageRequest::from_payload(&payload);
        let error = prepare_moss_preload(&settings, &payload, &request, &mut |_, _| {})
            .err()
            .expect("non-MOSS must fail before snapshot/backend preparation");
        assert!(
            matches!(error, WorkerError::InvalidPayload(ref message) if message == "missing MOSS model"),
            "{error}"
        );
        assert!(
            !hub.exists(),
            "request refusal must not create a model cache"
        );
    }

    #[test]
    fn prepared_preload_preserves_base_dmd_refusal_before_source_lookup() {
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let settings = Settings::for_test(temp.path().join("data"));
        for mode in ["unmerged_bf16", "full_fp32"] {
            let payload = preload_payload_for_pure_test(MossImageVariant::Base, mode);
            let request = ImageRequest::from_payload(&payload);
            let error = prepare_moss_preload(&settings, &payload, &request, &mut |_, _| {})
                .err()
                .expect("Base+DMD must fail before source/backend preparation");
            assert!(
                matches!(error, WorkerError::InvalidPayload(ref message) if message.contains("DMD requires the exact SFT parent")),
                "{mode}: {error}"
            );
            assert!(
                !error.to_string().contains("snapshot missing"),
                "parent refusal must precede cache lookup: {error}"
            );
        }
        assert!(
            !hub.exists(),
            "parent refusal must not create a model cache"
        );
    }

    #[test]
    fn prepared_preload_requires_exact_frozen_snapshot_for_each_admissible_mode() {
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let settings = Settings::for_test(temp.path().join("data"));
        for (variant, mode) in [
            (MossImageVariant::Base, "off"),
            (MossImageVariant::Sft, "off"),
            (MossImageVariant::Sft, "unmerged_bf16"),
            (MossImageVariant::Sft, "full_fp32"),
        ] {
            // An unrelated empty revision must not become a fallback. No model
            // file is created, and native/backend demand functions remain unreachable.
            let wrong = hub
                .join(format!(
                    "models--{}",
                    variant.repository().replace('/', "--")
                ))
                .join("snapshots")
                .join("0000000000000000000000000000000000000000");
            std::fs::create_dir_all(&wrong).unwrap();
            let payload = preload_payload_for_pure_test(variant, mode);
            let request = ImageRequest::from_payload(&payload);
            let error = prepare_moss_preload(&settings, &payload, &request, &mut |_, _| {})
                .err()
                .expect("absent frozen snapshot must refuse before backend demand pricing");
            let expected = format!(
                "{} frozen snapshot missing; install or repair this model in Models",
                variant.provider_id()
            );
            assert!(
                matches!(error, WorkerError::InvalidPayload(ref message) if message == &expected),
                "{variant:?}/{mode}: {error}"
            );
            assert_eq!(
                std::fs::read_dir(&wrong).unwrap().count(),
                0,
                "no model payload may be staged by a pre-load refusal"
            );
        }
    }

    #[test]
    fn full_prepared_consumer_config_reaches_native_shared_preparation_without_ordinary_dit() {
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let root = |repo: &str, revision: &str| {
            hub.join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots")
                .join(revision)
        };
        let ordinary = root(SFT_REPO, SFT_REVISION);
        let full = root(DMD_REPO, DMD_REVISION);
        std::fs::create_dir_all(ordinary.join("transformer")).unwrap();
        std::fs::create_dir_all(full.join("transformer")).unwrap();
        std::fs::write(
            ordinary.join("transformer/config.json"),
            b"unused ordinary config must not be consumed",
        )
        .unwrap();
        std::fs::write(
            full.join("transformer/config.json"),
            include_bytes!("../../../../tests/fixtures/moss-transformer-config.json"),
        )
        .unwrap();
        let mode = MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32);
        let mut spec = LoadSpec::new(WeightsSource::Dir(ordinary))
            .with_resolved_route("moss_image_sft:full_fp32")
            .with_component(DMD_TRANSFORMER_COMPONENT, WeightsSource::Dir(full.clone()))
            .with_component(DMD_DATASPACE_COMPONENT, WeightsSource::Dir(full.clone()));
        // A projection containing the real frozen selected config, deliberately missing shared
        // weights. Successful admission/execution requires real weights and remains terminal-only.
        let entry = json!({"downloads":[
            {"repo":SFT_REPO,"revision":SFT_REVISION,"files":["transformer/config.json"]},
            {"repo":DMD_REPO,"revision":DMD_REVISION,"files":["transformer/config.json"]}
        ]})
        .as_object()
        .unwrap()
        .clone();
        let mut pin_progress = Vec::new();
        moss_pin_spec(
            &mut spec,
            &entry,
            &Settings::for_test(temp.path().join("data")),
            mode,
            &mut |done, total| pin_progress.push((done, total)),
        )
        .unwrap();
        assert_eq!(spec.prepared_file_pins().len(), 1);
        // sc-26610: one progress tick per selected file, before its (possibly cold) seal.
        assert_eq!(pin_progress, vec![(0, 1)]);
        #[cfg(target_os = "macos")]
        let error = moss_memory::selected_asset_facts(MossImageVariant::Sft, &spec, mode)
            .unwrap_err()
            .to_string();
        #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
        let error = moss_memory::selected_preload_estimate(
            MossImageVariant::Sft,
            &spec,
            &GenerationRequest {
                moss_image: Some(MossImageOptions {
                    acceleration: mode,
                    ..Default::default()
                }),
                ..Default::default()
            },
            MossImageAdvancedInputs::default(),
        )
        .unwrap_err()
        .to_string();
        // Native errors display host paths, including Windows backslash separators.
        let error = error.replace('\\', "/");
        assert!(error.contains("text_encoder/config.json"), "{error}");
        assert!(!error.contains("transformer/config.json"), "{error}");
        // The macOS normal registry constructor prepares sources before device/tensor work.
        // CUDA factory execution is covered by the separately admitted native worker campaign.
        #[cfg(target_os = "macos")]
        {
            let error = crate::inference_runtime::load("moss_image_sft", &spec)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("text_encoder/config.json"), "{error}");
            assert!(!error.contains("transformer/config.json"), "{error}");
        }
    }
    #[test]
    fn moss_worker_persisted_batch_recipe_keeps_per_prompt_count_and_total_generation_metadata() {
        // The Studio regression consumes this projection of a real persisted recipe.
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/moss-batch-recipe.json"
        ))
        .unwrap();
        let prompts = fixture["rawAdapterSettings"]["promptBatch"].clone();
        let payload = json!({"model":"moss_image_sft","mode":"text_to_image","prompt":prompts[0],"prompts":prompts,"count":2,"width":16,"height":16,"modelManifestEntry":{"limits":{"minDimension":16}},"advanced":{"mossImage":{"acceleration":"off"}}});
        let request = ImageRequest::from_payload(payload.as_object().unwrap());
        let plan = ImagePlan::with_count_and_adapter(&request, 4, "mlx_moss_image_sft", None);
        assert_eq!(plan.generation_set["count"], 4);
        assert_eq!(streaming_result(&plan, &[])["expectedCount"], 4);
        let temp = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(temp.path().to_path_buf(), "test");
        let project = store.create_project("MOSS batch recipe").unwrap();
        for index in 0..4 {
            let fact = write_image_asset(
                &plan,
                index,
                173,
                16,
                16,
                vec![0; 16 * 16 * 3],
                "mlx_moss_image_sft",
                request.advanced.clone(),
                Path::new(&project.path),
            )
            .unwrap();
            let asset = store
                .persist_generated_asset(
                    &project.id,
                    "job-moss",
                    &plan.genset_id,
                    &Value::Object(fact.clone()),
                )
                .unwrap();
            let path = Path::new(&project.path)
                .join("recipes")
                .join(format!("{}.recipe.json", fact["assetId"].as_str().unwrap()));
            let recipe: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            assert_eq!(recipe, asset["recipe"]);
            for pointer in [
                "/mode",
                "/model",
                "/negativePrompt",
                "/normalizedSettings/width",
                "/normalizedSettings/height",
                "/normalizedSettings/count",
                "/rawAdapterSettings/mossImage",
                "/rawAdapterSettings/promptBatch",
            ] {
                assert_eq!(
                    recipe.pointer(pointer),
                    fixture.pointer(pointer),
                    "{pointer}: {recipe}"
                );
            }
            assert_eq!(recipe["prompt"], prompts[index / request.count as usize]);
        }
    }

    #[test]
    fn moss_edit_history_writer_persists_all_semantic_slots_and_authored_controls() {
        // This exercises egress/ProjectStore only, with CPU pixels. Native execution and the
        // four-reference REST gate are tested independently; no weights or device is loaded.
        for ids in [json!(["a"]), json!(["a", "b", "a", "c"])] {
            let options = json!({"reference_max_edge":768,"use_kv_cache":false});
            let payload = json!({"model":"moss_image_sft","mode":"edit_image","prompt":"blue cup","referenceAssetIds":ids,"sourceAssetId":"a","referenceAssetId":"a","count":1,"width":512,"height":512,"advanced":{"mossImage":{"acceleration":"off","uncond_skip_layers":[]},"mossImageEdit":options,"steps":30,"guidanceScale":4,"schedulerShift":5,"referenceAssetIds":["forged-output"]}});
            let request = ImageRequest::from_payload(payload.as_object().unwrap());
            let plan = ImagePlan::with_count_and_adapter(&request, 1, "mlx_moss_image_sft", None);
            let temp = tempfile::tempdir().unwrap();
            let store = ProjectStore::new(temp.path().to_path_buf(), "test");
            let project = store.create_project("MOSS edit history").unwrap();
            let fact = write_image_asset(
                &plan,
                0,
                17,
                512,
                512,
                vec![0; 512 * 512 * 3],
                "mlx_moss_image_sft",
                request.advanced.clone(),
                Path::new(&project.path),
            )
            .unwrap();
            let asset = store
                .persist_generated_asset(
                    &project.id,
                    "job-edit-history",
                    &plan.genset_id,
                    &Value::Object(fact),
                )
                .unwrap();
            assert_eq!(
                asset["recipe"]["rawAdapterSettings"]["referenceAssetIds"],
                ids
            );
            assert_eq!(
                asset["recipe"]["rawAdapterSettings"]["mossImageEdit"],
                options
            );
            for (key, value) in [
                ("steps", json!(30)),
                ("guidanceScale", json!(4)),
                ("schedulerShift", json!(5)),
            ] {
                assert_eq!(asset["recipe"]["rawAdapterSettings"][key], value);
            }
            assert_eq!(asset["lineage"]["sourceAssetId"], "a");
            assert_eq!(asset["recipe"]["normalizedSettings"]["width"], 512);
            assert_eq!(asset["recipe"]["normalizedSettings"]["height"], 512);
            assert_eq!(asset["recipe"]["negativePrompt"], "");
            assert_eq!(asset["recipe"]["seed"], 17);
            let path = Path::new(&project.path)
                .join("recipes")
                .join(format!("{}.recipe.json", asset["id"].as_str().unwrap()));
            let persisted: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            assert_eq!(persisted, asset["recipe"]);
        }
    }

    #[test]
    fn moss_edit_history_writer_refuses_conflicting_scalar_alias_before_writing_output() {
        let payload = json!({"model":"moss_image_sft","mode":"edit_image","prompt":"blue cup","referenceAssetIds":["a","b","a","c"],"sourceAssetId":"different-source","width":16,"height":16,"advanced":{"mossImageEdit":{"reference_max_edge":512,"use_kv_cache":true},"referenceAssetIds":["forged-output"]}});
        let request = ImageRequest::from_payload(payload.as_object().unwrap());
        let plan = ImagePlan::with_count_and_adapter(&request, 1, "mlx_moss_image_sft", None);
        let temp = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(temp.path().to_path_buf(), "test");
        let project = store.create_project("MOSS conflicting history").unwrap();
        let error = write_image_asset(
            &plan,
            0,
            17,
            16,
            16,
            vec![0; 16 * 16 * 3],
            "mlx_moss_image_sft",
            request.advanced.clone(),
            Path::new(&project.path),
        )
        .unwrap_err();
        assert!(error.to_string().contains("sourceAssetId"), "{error}");
        assert_eq!(
            std::fs::read_dir(Path::new(&project.path).join("assets/images"))
                .unwrap()
                .count(),
            0
        );
        assert_eq!(
            std::fs::read_dir(Path::new(&project.path).join("recipes"))
                .unwrap()
                .count(),
            0
        );
    }
    fn completed_receipt(req: &GenerationRequest) -> MossImageExecutionReport {
        completed_receipt_for(req, MossImageVariant::Sft)
    }
    fn completed_receipt_for(
        req: &GenerationRequest,
        variant: MossImageVariant,
    ) -> MossImageExecutionReport {
        let resolved = MossImageResolved::resolve(variant, req).unwrap();
        let hash = "a".repeat(64);
        let mode = resolved.acceleration;
        let mut paths = SNAPSHOT_FILES
            .iter()
            .filter(|p| **p != "LICENSE")
            .map(|p| (*p).to_owned())
            .collect::<Vec<_>>();
        if mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) {
            paths.retain(|p| !p.starts_with("transformer/diffusion_pytorch_model-000"));
            paths.extend((1..=4).map(|i| {
                format!("transformer/diffusion_pytorch_model-{i:05}-of-00004.safetensors")
            }));
        }
        if mode != MossImageAcceleration::Off {
            paths.push("dataspace.safetensors".into());
        }
        if mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) {
            paths.extend([
                "lora.safetensors".into(),
                "time_embedding.safetensors".into(),
            ]);
        }
        let assets = paths
            .into_iter()
            .map(|path| {
                let role = if path.starts_with("transformer/") {
                    MossImageAssetRole::Transformer
                } else if path.starts_with("text_encoder/") {
                    MossImageAssetRole::TextEncoder
                } else if path.starts_with("tokenizer/") {
                    MossImageAssetRole::Tokenizer
                } else if path.starts_with("scheduler/") {
                    MossImageAssetRole::Scheduler
                } else if path == "dataspace.safetensors" {
                    MossImageAssetRole::DmdDataspace
                } else if path == "lora.safetensors" {
                    MossImageAssetRole::DmdLora
                } else if path == "time_embedding.safetensors" {
                    MossImageAssetRole::DmdTime
                } else {
                    MossImageAssetRole::Vae
                };
                let dmd = matches!(
                    role,
                    MossImageAssetRole::DmdLora
                        | MossImageAssetRole::DmdTime
                        | MossImageAssetRole::DmdDataspace
                ) || (role == MossImageAssetRole::Transformer
                    && mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32));
                MossImageAssetReceipt {
                    role,
                    repository: if dmd {
                        DMD_REPO
                    } else if role == MossImageAssetRole::Transformer {
                        variant.repository()
                    } else {
                        SFT_REPO
                    }
                    .into(),
                    revision: if dmd {
                        DMD_REVISION
                    } else if role == MossImageAssetRole::Transformer {
                        variant.revision()
                    } else {
                        SFT_REVISION
                    }
                    .into(),
                    relative_path: path,
                    sha256: hash.clone(),
                }
            })
            .collect();
        let sequence_length = resolved.max_sequence_length.min(3);
        let needs_cfg = mode == MossImageAcceleration::Off && resolved.guidance != 1.0;
        let input = MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: hash.clone(),
            sequence_length,
            valid_tokens: sequence_length,
        };
        let sampling = if mode == MossImageAcceleration::Off {
            MossImageSamplingReceipt::Euler {
                sigmas: moss_image_schedule(
                    resolved.steps,
                    resolved.shift,
                    resolved.training_shift,
                )
                .unwrap(),
            }
        } else {
            MossImageSamplingReceipt::Dmd {
                step_codes: vec![1.0; resolved.steps as usize],
                mix_matrix_sha256: hash.clone(),
            }
        };
        let mut component_dtypes = [
            ("transformer", MossImageDtype::Bf16),
            ("text_encoder", MossImageDtype::F32),
            ("vae", MossImageDtype::F32),
            ("latents", MossImageDtype::F32),
            ("time_embedding", MossImageDtype::Bf16),
            ("shared_block_modulation", MossImageDtype::Bf16),
            ("video_head_modulation", MossImageDtype::Bf16),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect::<std::collections::BTreeMap<_, _>>();
        let mut adapter_reports = vec![];
        if mode != MossImageAcceleration::Off {
            for key in [
                "time_embedding",
                "shared_block_modulation",
                "video_head_modulation",
            ] {
                component_dtypes.insert(key.into(), MossImageDtype::F32);
            }
            if mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) {
                component_dtypes.insert("lora_residual".into(), MossImageDtype::F32);
                adapter_reports.push(gen_core::AdapterApplyReport {
                    adapter_path: "/Users/private/models/lora.safetensors".into(),
                    applied: 252,
                    skipped: vec![],
                });
            } else {
                component_dtypes.insert("transformer".into(), MossImageDtype::F32);
            }
        }
        MossImageExecutionReport::new(MossImageExecutionData {
            execution_device: MossImageExecutionDevice::Cpu,
            resolved,
            assets,
            sampling,
            samples: vec![MossImageSampleReceipt {
                seed: Some(173),
                noise_sha256: hash,
            }],
            conditioning: vec![MossImageConditioningReceipt {
                positive: input.clone(),
                negative: needs_cfg.then_some(input),
            }],
            images_per_prompt: 1,
            latent_shape: [1, 128, 1, 1],
            output_kind: MossImageOutputKind::Images,
            component_dtypes,
            adapter_reports,
        })
        .unwrap()
    }
    #[test]
    fn moss_png_recipe_roundtrip_preserves_requested_mode_and_complete_actual_receipt_without_local_paths(
    ) {
        for mode in [
            MossImageAcceleration::Off,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        ] {
            let native = GenerationRequest {
                width: 16,
                height: 16,
                moss_image: Some(MossImageOptions {
                    acceleration: mode,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let report = completed_receipt(&native);
            let options = serde_json::to_value(native.moss_image.unwrap()).unwrap();
            let payload=json!({"model":"moss_image_sft","mode":"text_to_image","prompt":"日本語 <tag> {\"items\":[]}","count":1,"width":16,"height":16,"modelManifestEntry":{"limits":{"minDimension":16}},"advanced":{"mossImage":options}}).as_object().unwrap().clone();
            let request = ImageRequest::from_payload(&payload);
            let plan = ImagePlan::with_count_and_adapter(
                &request,
                1,
                "mlx_moss_image_sft",
                Some(Arc::new(payload.clone())),
            );
            let temp = tempfile::tempdir().unwrap();
            let mut raw = request.advanced.clone();
            raw.insert(
                "mossImageExecutionReport".into(),
                serde_json::to_value(&report).unwrap(),
            );
            raw.insert("numInferenceSteps".into(), json!(1));
            let fact = write_image_asset(
                &plan,
                0,
                173,
                16,
                16,
                vec![0; 16 * 16 * 3],
                "mlx_moss_image_sft",
                raw,
                temp.path(),
            )
            .unwrap();
            let path = temp.path().join(fact["mediaPath"].as_str().unwrap());
            let shared = sceneworks_core::workflow_png::read_workflow_chunk_file(&path)
                .unwrap()
                .unwrap();
            assert_eq!(shared.advanced["mossImage"], options);
            assert_eq!(
                shared.advanced["steps"],
                json!(report.data().resolved.steps)
            );
            let wire = &shared.advanced["mossImageExecutionReport"];
            let parsed: MossImageExecutionReport = serde_json::from_value(wire.clone()).unwrap();
            assert_eq!(parsed.data().assets.len(), report.data().assets.len());
            assert_eq!(parsed.data().resolved.acceleration, mode);
            assert_eq!(parsed.data().samples[0].seed, Some(173));
            assert!(!serde_json::to_string(&shared)
                .unwrap()
                .contains("/Users/private/"));
            assert!(
                sceneworks_core::workflow_png::workflow_chunk_size(&shared).unwrap()
                    < sceneworks_core::workflow_share::WORKFLOW_SHARE_MAX_BYTES
            );
            if mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) {
                assert_eq!(
                    parsed.data().adapter_reports[0].adapter_path,
                    PathBuf::from("lora.safetensors")
                );
            }
            let mut replay = payload.clone();
            replay.insert("advanced".into(), Value::Object(shared.advanced));
            sceneworks_core::moss_image::scrub_reports(&mut replay);
            assert!(replay["advanced"].get("mossImageExecutionReport").is_none());
            assert_eq!(replay["advanced"]["mossImage"], options);
        }
    }
    type AdvancedObservation = (
        Vec<u32>,
        Vec<u8>,
        Vec<String>,
        Vec<u32>,
        MossImageOutputKind,
    );
    struct Probe {
        descriptor: gen_core::ModelDescriptor,
        report: RefCell<Option<MossImageExecutionReport>>,
        edit_report: RefCell<Option<gen_core::moss_image_edit::MossImageEditExecutionReport>>,
        edit_inputs: RefCell<Vec<(u32, u32, Vec<u8>)>>,
        expected_edit_inputs: RefCell<Vec<(u32, u32, Vec<u8>)>>,
        expected_edit_cache: Cell<bool>,
        failure: Cell<bool>,
        cancel: Cell<bool>,
        reads: Cell<usize>,
        owned_weight_bytes: Cell<u64>,
        adapter_reports: RefCell<Vec<gen_core::AdapterApplyReport>>,
        adapter_report_reads: Cell<usize>,
        advanced_inputs: RefCell<Vec<AdvancedObservation>>,
    }
    impl Probe {
        fn new() -> Self {
            Self {
                descriptor: gen_core::ModelDescriptor {
                    id: "moss_image_sft",
                    family: "moss_image",
                    backend: "test",
                    modality: gen_core::Modality::Image,
                    capabilities: Default::default(),
                    denoiser_output_latent_space: None,
                    control_kinds: None,
                    encoder_contract: None,
                    required_components: &[],
                },
                report: RefCell::new(None),
                edit_report: RefCell::new(None),
                edit_inputs: RefCell::new(Vec::new()),
                expected_edit_inputs: RefCell::new(vec![(48, 16, vec![73; 48 * 16 * 3])]),
                expected_edit_cache: Cell::new(false),
                failure: Cell::new(false),
                cancel: Cell::new(false),
                reads: Cell::new(0),
                owned_weight_bytes: Cell::new(0),
                adapter_reports: RefCell::new(vec![]),
                adapter_report_reads: Cell::new(0),
                advanced_inputs: RefCell::new(Vec::new()),
            }
        }
    }
    impl Generator for Probe {
        fn moss_image_edit_request_peak(
            &self,
            request: &GenerationRequest,
            inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
        ) -> gen_core::Result<u64> {
            inputs.validate()?;
            let native = gen_core::moss_image_edit::edit_request(request)?;
            let resolved = MossImageResolved::resolve(MossImageVariant::Sft, &native)?;
            inputs.advanced.validate(&native, &resolved)?;
            self.advanced_inputs.borrow_mut().push((
                inputs
                    .advanced
                    .positive
                    .map(|e| e.values.iter().map(|v| v.to_bits()).collect())
                    .unwrap_or_default(),
                inputs
                    .advanced
                    .positive
                    .and_then(|e| e.mask)
                    .unwrap_or_default()
                    .to_vec(),
                inputs
                    .advanced
                    .negative_prompts
                    .unwrap_or_default()
                    .to_vec(),
                inputs
                    .advanced
                    .noise
                    .unwrap_or_default()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect(),
                inputs.advanced.output,
            ));
            assert_eq!(inputs.reference_max_edge, 32);
            assert_eq!(inputs.use_kv_cache, self.expected_edit_cache.get());
            assert_eq!(
                inputs
                    .references
                    .iter()
                    .map(|image| (image.width, image.height, image.pixels.clone()))
                    .collect::<Vec<_>>(),
                *self.expected_edit_inputs.borrow(),
                "the typed peak receives every semantic RGB slot in request order"
            );
            Ok(73)
        }
        fn generate_moss_image_edit(
            &self,
            req: &GenerationRequest,
            inputs: gen_core::moss_image_edit::MossImageEditInputs<'_>,
            progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<MossImageAdvancedOutput> {
            use gen_core::moss_image_edit::*;
            *self.report.borrow_mut() = None;
            *self.edit_report.borrow_mut() = None;
            let dimensions = inputs.validate()?;
            *self.edit_inputs.borrow_mut() = inputs
                .references
                .iter()
                .map(|i| (i.width, i.height, i.pixels.clone()))
                .collect();
            progress(Progress::Step {
                current: 1,
                total: 1,
            });
            if self.failure.get() {
                return Err(gen_core::Error::Unsupported("edit fixture failure".into()));
            }
            if self.cancel.get() {
                req.cancel.cancel();
            }
            let native = edit_request(req)?;
            let mut base = completed_receipt_for(
                &native,
                MossImageVariant::from_provider_id(self.descriptor.id)
                    .expect("MOSS fixture variant"),
            );
            let advanced = inputs.advanced;
            let advanced_mode = advanced.positive.is_some()
                || advanced.negative.is_some()
                || advanced.negative_prompts.is_some()
                || advanced.noise.is_some()
                || advanced.output == MossImageOutputKind::Latents;
            if advanced_mode {
                use sha2::{Digest, Sha256};
                let mut data = base.data().clone();
                let total = advanced.validate(&native, &data.resolved)?;
                data.images_per_prompt = native.count;
                data.latent_shape[0] = total;
                data.output_kind = advanced.output;
                let batch = total / native.count as usize;
                data.conditioning = (0..batch)
                    .map(|row| {
                        let positive = advanced
                            .positive
                            .map(|e| {
                                let n = e.sequence * 4096;
                                let mut hash = Sha256::new();
                                for v in &e.values[row * n..(row + 1) * n] {
                                    hash.update(v.to_le_bytes());
                                }
                                let mask =
                                    e.mask.map(|m| &m[row * e.sequence..(row + 1) * e.sequence]);
                                if let Some(mask) = mask {
                                    hash.update(mask);
                                }
                                MossImageConditioningInputReceipt {
                                    source: MossImageConditioningSource::Embeddings,
                                    sha256: format!("{:x}", hash.finalize()),
                                    sequence_length: e.sequence as u32,
                                    valid_tokens: mask.map_or(e.sequence, |m| {
                                        m.iter().filter(|v| **v == 1).count()
                                    }) as u32,
                                }
                            })
                            .unwrap_or_else(|| data.conditioning[0].positive.clone());
                        let negative = advanced
                            .negative_prompts
                            .map(|texts| MossImageConditioningInputReceipt {
                                source: MossImageConditioningSource::Text,
                                sha256: format!("{:x}", Sha256::digest(texts[row].as_bytes())),
                                sequence_length: 1,
                                valid_tokens: 1,
                            })
                            .or_else(|| data.conditioning[0].negative.clone());
                        MossImageConditioningReceipt { positive, negative }
                    })
                    .collect();
                data.samples = (0..total)
                    .map(|index| MossImageSampleReceipt {
                        seed: if advanced.noise.is_some() {
                            None
                        } else {
                            Some(173 + index as u64)
                        },
                        noise_sha256: if let Some(noise) = advanced.noise {
                            let n = noise.len() / total;
                            let bytes = noise[index * n..(index + 1) * n]
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>();
                            format!("{:x}", Sha256::digest(bytes))
                        } else {
                            "a".repeat(64)
                        },
                    })
                    .collect();
                base = MossImageExecutionReport::new(data)?;
                self.advanced_inputs.borrow_mut().push((
                    advanced
                        .positive
                        .map(|e| e.values.iter().map(|v| v.to_bits()).collect())
                        .unwrap_or_default(),
                    advanced
                        .positive
                        .and_then(|e| e.mask)
                        .unwrap_or_default()
                        .to_vec(),
                    advanced.negative_prompts.unwrap_or_default().to_vec(),
                    advanced
                        .noise
                        .unwrap_or_default()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                    advanced.output,
                ));
            }
            let total = base.data().samples.len();
            *self.edit_report.borrow_mut() = Some(MossImageEditExecutionReport {
                base,
                adapter_path: "/private/adapter_model.safetensors".into(),
                edit_contract: EDIT_CONTRACT.into(),
                adapter_repository: EDIT_REPOSITORY.into(),
                adapter_revision: EDIT_REVISION.into(),
                adapter_sha256: EDIT_WEIGHTS_SHA256.into(),
                config_sha256: EDIT_CONFIG_SHA256.into(),
                applied_projections: 252,
                references: inputs
                    .references
                    .iter()
                    .zip(dimensions)
                    .map(|(image, dimensions)| MossImageEditReferenceReceipt {
                        original_dimensions: [image.width, image.height],
                        encoded_dimensions: dimensions,
                        pixels_sha256: {
                            use sha2::Digest;
                            format!("{:x}", sha2::Sha256::digest(&image.pixels))
                        },
                        resized_pixels_sha256: "b".repeat(64),
                        latent_sha256: "c".repeat(64),
                    })
                    .collect(),
                reference_max_edge: inputs.reference_max_edge,
                use_kv_cache: inputs.use_kv_cache,
            });
            if advanced.output == MossImageOutputKind::Latents {
                Ok(MossImageAdvancedOutput::Latents {
                    values: (0..total * 128).map(|index| index as f32 / 512.).collect(),
                    shape: [total, 128, 1, 1],
                })
            } else {
                Ok(MossImageAdvancedOutput::Images(
                    (0..total)
                        .map(|_| Image {
                            width: 16,
                            height: 16,
                            pixels: vec![0; 16 * 16 * 3],
                        })
                        .collect(),
                ))
            }
        }
        fn moss_image_edit_execution_report(
            &self,
        ) -> Option<gen_core::moss_image_edit::MossImageEditExecutionReport> {
            self.reads.set(self.reads.get() + 1);
            self.edit_report.borrow().clone()
        }
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn moss_image_owned_weight_bytes(&self) -> Option<u64> {
            Some(self.owned_weight_bytes.get())
        }
        fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
            self.adapter_report_reads
                .set(self.adapter_report_reads.get() + 1);
            self.adapter_reports.borrow().clone()
        }
        fn moss_image_execution_report(&self) -> Option<MossImageExecutionReport> {
            self.reads.set(self.reads.get() + 1);
            self.report.borrow().clone()
        }
        fn generate(
            &self,
            req: &GenerationRequest,
            progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            *self.report.borrow_mut() = None;
            progress(Progress::Step {
                current: 1,
                total: 1,
            });
            if self.failure.get() {
                return Err(gen_core::Error::Unsupported("fixture failure".into()));
            }
            if self.cancel.get() {
                req.cancel.cancel();
            }
            *self.report.borrow_mut() = Some(completed_receipt(req));
            Ok(GenerationOutput::Images(vec![Image {
                width: 16,
                height: 16,
                pixels: vec![0; 16 * 16 * 3],
            }]))
        }
    }
    #[tokio::test]
    async fn moss_publication_ack_loss_reconciles_exact_completed_batch_and_retains_pending_staging(
    ) {
        use axum::{
            extract::State, http::StatusCode, response::IntoResponse, routing::get, Json, Router,
        };
        use sceneworks_core::jobs_store::{CreateJob, JobsStore};
        use std::sync::Mutex;
        for outcome in [
            "completed",
            "pending",
            "unavailable",
            "failed",
            "wrong_generation",
            "cancel",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let store = JobsStore::new(temp.path().join("jobs.db"));
            store.initialize().unwrap();
            let mut job=store.create_job(CreateJob{job_type:JobType::ImageEdit,project_id:Some("project".into()),project_name:None,payload:json!({"model":"moss_image_sft","mode":"edit_image","prompt":"edit","referenceAssetIds":["rgb"]}).as_object().unwrap().clone(),requested_gpu:"auto".into(),source_job_id:None,duplicate_of_job_id:None,attempts:1,initial_status:None}).unwrap();
            job.cancel_requested = outcome == "cancel";
            let plan = ImagePlan::new(&ImageRequest::from_payload(&job.payload));
            let relative = format!("assets/images/moss-edit/{}/output.png", plan.genset_id);
            let path = temp.path().join("owned-output");
            let mut batch = MossOutputBatch::new();
            batch.stage(path.clone(), b"owned").unwrap();
            let facts = vec![json!({"assetId":"asset_output","mediaPath":relative})];
            batch.facts = facts
                .iter()
                .map(|fact| fact.as_object().unwrap().clone())
                .collect();
            #[derive(Clone)]
            struct Mock {
                job: Arc<Mutex<JobSnapshot>>,
                posts: Arc<Mutex<Vec<Value>>>,
                outcome: &'static str,
                generation: String,
                relative: String,
            }
            let state = Mock {
                job: Arc::new(Mutex::new(job.clone())),
                posts: Arc::new(Mutex::new(Vec::new())),
                outcome,
                generation: plan.genset_id.clone(),
                relative,
            };
            let server_state = state.clone();
            let router=Router::new().route("/api/v1/jobs/:id",get(|State(state):State<Mock>|async move {
                if state.outcome=="unavailable" && !state.posts.lock().unwrap().is_empty(){return (StatusCode::SERVICE_UNAVAILABLE,Json(json!({"error":"owned fixture unavailable"}))).into_response();}
                Json(state.job.lock().unwrap().clone()).into_response()
            })).route("/api/v1/jobs/:id/progress",axum::routing::post(|State(state):State<Mock>,Json(body):Json<Value>|async move {
                state.posts.lock().unwrap().push(body.clone());
                let mut job=state.job.lock().unwrap();
                if body["status"]=="canceled" {job.status=JobStatus::Canceled;return (StatusCode::OK,Json(job.clone())).into_response();}
                job.status=if state.outcome=="failed" {JobStatus::Failed}else{JobStatus::Completed};
                job.result=json!({"generationSetId":state.generation,"assetIds":["asset_output"],"assets":[{"id":"asset_output","projectId":"project","generationSetId":state.generation,"file":{"path":state.relative},"lineage":{"jobId":job.id}}]}).as_object().unwrap().clone();
                if state.outcome=="pending" {job.result=json!({"generationSetId":state.generation,"outputPublication":{"state":"pending"}}).as_object().unwrap().clone();}
                if state.outcome=="wrong_generation" {job.result.insert("generationSetId".into(),json!("another_generation"));}
                (StatusCode::INTERNAL_SERVER_ERROR,Json(json!({"error":"lost acknowledgment after owned server commit"}))).into_response()
            })).with_state(server_state);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
            let mut settings = Settings::for_test(temp.path().join("data"));
            settings.api_url = format!("http://{address}");
            let result = publish_moss_batch(
                &ApiClient::new(&settings),
                &job,
                &plan,
                "mlx",
                &facts,
                batch,
            )
            .await;
            if outcome == "completed" {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err(), "{outcome}");
            }
            assert_eq!(
                path.exists(),
                !matches!(outcome, "failed" | "cancel"),
                "{outcome}"
            );
            assert_eq!(
                state
                    .posts
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|body| body["status"] == "completed")
                    .count(),
                usize::from(outcome != "cancel")
            );
            task.abort();
            let _ = task.await;
        }
    }

    #[test]
    fn moss_png_staging_refuses_foreign_output_and_cleans_only_its_new_temporary_file() {
        let temp = tempfile::tempdir().unwrap();
        let request = ImageRequest::from_payload(
            json!({"model":"moss_image_sft","mode":"edit_image","prompt":"edit"})
                .as_object()
                .unwrap(),
        );
        let plan = ImagePlan::new(&request);
        let temporary = temp.path().join("owned.tmp.png");
        let media = temp.path().join("foreign.png");
        std::fs::write(&media, b"original bytes").unwrap();
        assert!(moss_stage_image_files(&plan, &temporary, &media).is_err());
        assert!(!temporary.exists());
        assert_eq!(std::fs::read(&media).unwrap(), b"original bytes");
    }

    #[test]
    fn moss_staging_failure_channel_close_and_cancel_publish_no_partial_batch() {
        use gen_core::moss_image_edit::edit_request;
        for failure in [
            "second_file",
            "companion",
            "closed_channel",
            "cancel",
            "receiver_drop",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let payload=json!({"model":"moss_image_sft","mode":"edit_image","prompt":"edit","referenceAssetIds":["rgb"],"width":16,"height":16,"count":2}).as_object().unwrap().clone();
            let request = ImageRequest::from_payload(&payload);
            let plan = ImagePlan::with_count_and_adapter(
                &request,
                2,
                "mlx_moss_image_sft",
                Some(Arc::new(payload)),
            );
            let native = edit_request(&GenerationRequest {
                prompt: "edit".into(),
                width: 16,
                height: 16,
                count: 2,
                seed: Some(173),
                ..Default::default()
            })
            .unwrap();
            let refs = vec![Image {
                width: 16,
                height: 16,
                pixels: vec![0; 768],
            }];
            let probe = Probe::new();
            let (output, facts) = moss_edit_render(
                &probe,
                &native,
                &refs,
                sceneworks_core::moss_image::EditOptions::default(),
                MossImageAdvancedInputs {
                    output: MossImageOutputKind::Latents,
                    ..Default::default()
                },
                &[],
                &mut |_| {},
            )
            .unwrap();
            let dir = temp
                .path()
                .join("assets/tensors/moss-edit")
                .join(&plan.genset_id);
            std::fs::create_dir_all(&dir).unwrap();
            let collision = match failure {
                "second_file" => Some(dir.join("0001.npy")),
                "companion" => Some(dir.join("0000.sceneworks-workflow.json")),
                _ => None,
            };
            if let Some(path) = &collision {
                std::fs::write(path, b"preexisting foreign bytes").unwrap();
            }
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            if failure == "closed_channel" {
                rx.close();
            }
            if failure == "cancel" {
                native.cancel.cancel();
            }
            let mut raw = request.advanced.clone();
            raw.insert("referenceAssetIds".into(), json!(["rgb"]));
            let result =
                emit_moss_advanced_outputs(&tx, &plan, temp.path(), output, &facts, raw, &native);
            if failure == "receiver_drop" {
                assert!(result.is_ok());
                drop(rx);
            } else {
                assert!(result.is_err(), "{failure}");
                assert!(rx.try_recv().is_err());
            }
            let files = std::fs::read_dir(&dir)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect::<Vec<_>>();
            if let Some(path) = collision {
                assert_eq!(files, vec![path.clone()]);
                assert_eq!(std::fs::read(path).unwrap(), b"preexisting foreign bytes");
            } else {
                assert!(files.is_empty(), "{failure}: {files:?}");
            }
        }
    }

    #[test]
    fn advanced_edit_b2_count2_uses_identical_admission_render_views_and_persists_ordered_latent_recipes(
    ) {
        use sceneworks_core::{
            moss_image_advanced::{self, OwnedInputs},
            tensor_artifact,
        };
        let temporary = tempfile::tempdir().unwrap();
        let store = ProjectStore::new(temporary.path().to_owned(), "test");
        let project = store.create_project("advanced outputs").unwrap();
        let import = |name: &str, bytes: Vec<u8>, mime: &str| {
            let path = temporary.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            store
                .import_asset(
                    &project.id,
                    sceneworks_core::project_store::UploadAsset {
                        filename: name.into(),
                        content_type: Some(mime.into()),
                        source_path: path,
                        source_asset_id: None,
                        provenance: None,
                    },
                )
                .unwrap()
        };
        let mut rgb = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(48, 16)
            .write_to(&mut rgb, image::ImageFormat::Png)
            .unwrap();
        let reference = import("reference.png", rgb.into_inner(), "image/png");
        let positive = import(
            "positive.npy",
            tensor_artifact::encode_f32(&[2, 2, 4096], &vec![0.125; 16384]).unwrap(),
            tensor_artifact::MIME,
        );
        let mut noise = vec![0.; 512];
        noise[0] = -0.;
        noise[1] = f32::from_bits(1);
        noise[128] = 0.25;
        let noise_asset = import(
            "noise.npy",
            tensor_artifact::encode_f32(&[4, 128, 1, 1], &noise).unwrap(),
            tensor_artifact::MIME,
        );
        let tensor_ref =
            |asset: &Value| json!({"assetId":asset["id"],"sha256":asset["tensor"]["sha256"]});
        let mut payload=json!({"projectId":project.id,"model":"moss_image_sft","mode":"edit_image","prompt":"","referenceAssetIds":[reference["id"]],"count":2,"width":16,"height":16,"seed":173,"advanced":{"mossImage":{"max_sequence_length":2,"training_shift":2.5},"mossImageEdit":{"reference_max_edge":32,"use_kv_cache":false},"mossImageAdvanced":{"positive":{"values":tensor_ref(&positive)},"noise":tensor_ref(&noise_asset),"negativePrompts":["blur","text"],"output":"latents"}}}).as_object().unwrap().clone();
        let entry = json!({"limits":{"minDimension":16}})
            .as_object()
            .unwrap()
            .clone();
        moss_image_advanced::canonicalize(&store, &mut payload, &entry).unwrap();
        let owned = OwnedInputs::load(&store, &payload, &entry).unwrap();
        let native = sceneworks_core::moss_image::native_request(&payload, &entry)
            .unwrap()
            .unwrap();
        let request = ImageRequest::from_payload(&payload);
        let plan = ImagePlan::with_count_and_adapter(
            &request,
            4,
            "mlx_moss_image_sft",
            Some(Arc::new(payload.clone())),
        );
        let probe = Probe {
            expected_edit_inputs: RefCell::new(vec![(48, 16, vec![0; 48 * 16 * 3])]),
            ..Probe::new()
        };
        let refs = vec![Image {
            width: 48,
            height: 16,
            pixels: vec![0; 48 * 16 * 3],
        }];
        let options = sceneworks_core::moss_image::EditOptions {
            reference_max_edge: 32,
            use_kv_cache: false,
        };
        assert_eq!(
            moss_edit_peak(&probe, &native, &refs, options, owned.inputs()).unwrap(),
            73
        );
        let (output, facts) = moss_edit_render(
            &probe,
            &native,
            &refs,
            options,
            owned.inputs(),
            &[],
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(probe.advanced_inputs.borrow().len(), 2);
        assert_eq!(
            probe.advanced_inputs.borrow()[0],
            probe.advanced_inputs.borrow()[1]
        );
        assert_eq!(probe.advanced_inputs.borrow()[0].3[0], (-0.0f32).to_bits());
        assert_eq!(probe.advanced_inputs.borrow()[0].3[1], 1);
        let mut raw = request.advanced.clone();
        raw.insert("referenceAssetIds".into(), json!([reference["id"]]));
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        emit_moss_advanced_outputs(
            &tx,
            &plan,
            Path::new(&project.path),
            output,
            &facts,
            raw,
            &native,
        )
        .unwrap();
        drop(tx);
        let GenEvent::MossAssets(mut batch) = rx.try_recv().unwrap() else {
            panic!("expected actual tensor artifact batch")
        };
        assert_eq!(batch.facts.len(), 4);
        for (index, fact) in batch.facts.iter().cloned().enumerate() {
            assert_eq!(fact["seed"], Value::Null);
            let asset = store
                .persist_generated_asset(
                    &project.id,
                    "advanced-job",
                    &plan.genset_id,
                    &Value::Object(fact),
                )
                .unwrap();
            assert_eq!(asset["type"], "tensor");
            assert_eq!(asset["recipe"]["seed"], Value::Null);
            assert_eq!(asset["recipe"]["normalizedSettings"]["count"], 2);
            assert_eq!(
                asset["recipe"]["rawAdapterSettings"][moss_image_advanced::KEY],
                payload["advanced"][moss_image_advanced::KEY]
            );
            assert_eq!(
                asset["recipe"]["rawAdapterSettings"]["mossImageSampleIndex"],
                index
            );
            assert_eq!(asset["tensor"]["shape"], json!([1, 128, 1, 1]));
            let path = store
                .project_file(&project.id, asset["file"]["path"].as_str().unwrap())
                .unwrap()
                .path;
            let metadata = tensor_artifact::inspect(&path).unwrap();
            let bytes = tensor_artifact::read_data(&path, &metadata).unwrap();
            assert_eq!(
                f32::from_le_bytes(bytes[..4].try_into().unwrap()),
                (index * 128) as f32 / 512.
            );
            let companion = store
                .project_file(
                    &project.id,
                    asset["extra"]["workflowCompanion"]["path"]
                        .as_str()
                        .unwrap(),
                )
                .unwrap()
                .path;
            let share = sceneworks_core::workflow_share::parse_workflow_share_json(
                &std::fs::read_to_string(companion).unwrap(),
            )
            .unwrap();
            assert_eq!(share.count, Some(2));
            assert_eq!(share.seed, None);
            assert_eq!(share.advanced["mossImageSampleIndex"], index);
            assert!(share.advanced[moss_image_advanced::KEY]["noise"]
                .get("assetId")
                .is_none());
            assert_eq!(
                moss_image_advanced::requirements(&share.advanced[moss_image_advanced::KEY]).len(),
                2
            );
        }
        batch.commit();
        assert!(rx.try_recv().is_err());
        assert!(!Path::new(&project.path)
            .join(format!("assets/images/{}", plan.genset_id))
            .exists());
        drop(store);
        let reloaded = ProjectStore::new(temporary.path().to_owned(), "restart");
        assert_eq!(
            reloaded
                .get_asset(&project.id, reference["id"].as_str().unwrap())
                .unwrap()["type"],
            "image"
        );
        assert_eq!(
            reloaded
                .get_asset(&project.id, noise_asset["id"].as_str().unwrap())
                .unwrap()["tensor"],
            noise_asset["tensor"]
        );
        let reloaded_inputs = OwnedInputs::load(&reloaded, &payload, &entry).unwrap();
        assert_eq!(
            reloaded_inputs
                .noise
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            noise
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
    #[tokio::test]
    async fn direct_moss_edit_worker_refuses_invalid_controls_before_any_api_or_provider_call() {
        let temp = tempfile::tempdir().unwrap();
        let settings = Settings::for_test(temp.path().join("data"));
        let api = ApiClient::new(&settings);
        for model in ["moss_image_base", "moss_image_sft"] {
            for patch in [
                json!({"sourceAssetId":"conflict"}),
                json!({"advanced":{"mossImage":{"acceleration":"unmerged_bf16"}}}),
                json!({"advanced":{"controlImage":"pose"}}),
                json!({"referenceAssetIds":[]}),
            ] {
                let mut payload =
                    json!({"model":model,"mode":"edit_image","referenceAssetIds":["ref"]})
                        .as_object()
                        .unwrap()
                        .clone();
                payload.extend(patch.as_object().unwrap().clone());
                let job: JobSnapshot = serde_json::from_value(json!({"id":"moss-refusal","type":"image_edit","status":"queued",
                    "payload":payload,"result":{},"requestedGpu":"auto","progress":0,"stage":"queued","message":"","attempts":1,
                    "cancelRequested":false,"createdAt":"2026-10-23T00:00:00Z","updatedAt":"2026-10-23T00:00:00Z"})).unwrap();
                assert!(matches!(
                    run_image_generate_job_unexplained(&api, &settings, &job).await,
                    Err(WorkerError::InvalidPayload(_))
                ));
            }
        }
    }

    #[test]
    fn cold_edit_uses_native_typed_validation_before_provider_loading() {
        let temp = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(temp.path().to_path_buf()));
        let mut request = gen_core::moss_image_edit::edit_request(&GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        })
        .unwrap();
        let refs = [Image {
            width: 48,
            height: 16,
            pixels: vec![73; 48 * 16 * 3],
        }];
        let options = sceneworks_core::moss_image::EditOptions {
            reference_max_edge: 32,
            use_kv_cache: false,
        };
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let missing =
                moss_edit_cold_peak(variant, &spec, &request, &refs, options, Default::default())
                    .unwrap_err();
            assert!(
                missing.to_string().to_lowercase().contains("edit"),
                "{missing}"
            );
            assert!(moss_edit_cold_peak(
                variant,
                &spec,
                &request,
                &[],
                options,
                Default::default()
            )
            .is_err());
            request.moss_image.as_mut().unwrap().acceleration =
                MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16);
            assert!(moss_edit_cold_peak(
                variant,
                &spec,
                &request,
                &refs,
                options,
                Default::default()
            )
            .is_err());
            request.moss_image.as_mut().unwrap().acceleration = MossImageAcceleration::Off;
        }
    }

    #[test]
    fn edit_admission_and_typed_render_helpers_refuse_unsupported_providers() {
        let loads = Cell::new(0);
        let budget = gen_core::MemoryBudget {
            total_bytes: 100,
            committed_bytes: 30,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 10,
        };
        let load = || {
            loads.set(loads.get() + 1);
            Ok(Probe::new())
        };
        // The injected helper peak is 73 bytes; a smaller ordinary placeholder would fit.
        assert!(moss_load_admitted(73, budget, load).is_err());
        assert_eq!(
            loads.get(),
            0,
            "refusal must precede the actual provider callback"
        );
        let generator = moss_load_admitted(60, budget, load).unwrap();
        assert_eq!(loads.get(), 1);
        let request = gen_core::moss_image_edit::edit_request(&GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        })
        .unwrap();
        let references = [Image {
            width: 48,
            height: 16,
            pixels: vec![73; 48 * 16 * 3],
        }];
        let options = sceneworks_core::moss_image::EditOptions {
            reference_max_edge: 32,
            use_kv_cache: false,
        };
        let peak = moss_edit_peak(
            &generator,
            &request,
            &references,
            options,
            Default::default(),
        )
        .unwrap();
        assert_eq!(peak, 73);
        // A linked ordinary-only provider cannot silently serve an edit via Generator::generate.
        struct OrdinaryOnly(Probe);
        impl Generator for OrdinaryOnly {
            fn descriptor(&self) -> &gen_core::ModelDescriptor {
                self.0.descriptor()
            }
            fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
                self.0.validate(request)
            }
            fn generate(
                &self,
                request: &GenerationRequest,
                progress: &mut dyn FnMut(Progress),
            ) -> gen_core::Result<GenerationOutput> {
                self.0.generate(request, progress)
            }
        }
        let wrong_provider = OrdinaryOnly(Probe::new());
        assert!(moss_edit_peak(
            &wrong_provider,
            &request,
            &references,
            options,
            Default::default()
        )
        .is_err());
        assert!(moss_edit_render(
            &wrong_provider,
            &request,
            &references,
            options,
            Default::default(),
            &[],
            &mut |_| {}
        )
        .is_err());
        assert_eq!(
            wrong_provider.0.reads.get(),
            0,
            "unsupported edit must not publish ordinary receipts"
        );
        assert!(moss_admit(peak, moss_owned_weight_bytes(&generator, 0), budget).is_err());
        generator.owned_weight_bytes.set(20);
        moss_admit(peak, moss_owned_weight_bytes(&generator, 0), budget).unwrap();
        let (images, facts) = moss_edit_render(
            &generator,
            &request,
            &references,
            options,
            Default::default(),
            &[],
            &mut |_| {},
        )
        .unwrap();
        assert!(matches!(images, MossImageAdvancedOutput::Images(ref items) if items.len() == 1));
        assert_eq!(
            facts[sceneworks_core::moss_image::EDIT_REPORT_KEY]["references"][0]
                ["original_dimensions"],
            json!([48, 16])
        );
        generator.failure.set(true);
        assert!(moss_edit_render(
            &generator,
            &request,
            &references,
            options,
            Default::default(),
            &[],
            &mut |_| {}
        )
        .is_err());
        assert!(generator.edit_report.borrow().is_none());
        generator.failure.set(false);
        generator.cancel.set(true);
        assert!(matches!(
            moss_edit_render(
                &generator,
                &request,
                &references,
                options,
                Default::default(),
                &[],
                &mut |_| {}
            ),
            Err(WorkerError::Canceled(_))
        ));
        generator.cancel.set(false);
        let request = gen_core::moss_image_edit::edit_request(&GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        })
        .unwrap();
        assert!(
            matches!(moss_edit_render(&generator, &request, &references, options, Default::default(), &[], &mut |_| {}).unwrap().0, MossImageAdvancedOutput::Images(ref items) if items.len() == 1)
        );
        assert_eq!(
            loads.get(),
            1,
            "direct helper calls reused the constructed probe"
        );
    }

    #[test]
    fn edit_calls_typed_native_method_with_uncropped_rgb_and_only_publishes_actual_success() {
        let probe = Probe::new();
        let identity = lane_controls::AdapterIdentity {
            id: Some("selected-edit-custom".into()),
            path: PathBuf::from("edit-custom.safetensors"),
            sha256: Some("cd".repeat(32)),
            scale: 0.75,
            kind: gen_core::AdapterKind::Lora,
        };
        *probe.adapter_reports.borrow_mut() = vec![gen_core::AdapterApplyReport {
            adapter_path: identity.path.clone(),
            applied: 36,
            skipped: vec![],
        }];
        let req = gen_core::moss_image_edit::edit_request(&GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        })
        .unwrap();
        let refs = vec![Image {
            width: 48,
            height: 16,
            pixels: vec![73; 48 * 16 * 3],
        }];
        let options = sceneworks_core::moss_image::EditOptions {
            reference_max_edge: 32,
            use_kv_cache: false,
        };
        assert_eq!(
            moss_edit_peak(&probe, &req, &refs, options, Default::default()).unwrap(),
            73
        );
        let mut progress = 0;
        let (images, facts) = moss_edit_render(
            &probe,
            &req,
            &refs,
            options,
            Default::default(),
            std::slice::from_ref(&identity),
            &mut |_| progress += 1,
        )
        .unwrap();
        assert!(
            matches!(images, gen_core::moss_image::MossImageAdvancedOutput::Images(ref values) if values.len()==1)
        );
        assert_eq!(
            facts[sceneworks_core::moss_image::EDIT_REPORT_KEY]["base"]["samples"][0]["seed"],
            173
        );
        assert_eq!(progress, 1);
        assert_eq!(
            facts[lane_controls::ADAPTER_APPLY_KEY][0],
            json!({
                "id": "selected-edit-custom", "path": "edit-custom.safetensors",
                "sha256": "cd".repeat(32), "scale": 0.75, "kind": "lora",
                "applied": 36, "skipped": []
            })
        );
        assert_eq!(probe.adapter_report_reads.get(), 1);
        assert_eq!(
            probe.edit_inputs.borrow()[0],
            (48, 16, vec![73; 48 * 16 * 3])
        );
        assert!(facts.get(sceneworks_core::moss_image::REPORT_KEY).is_none());
        let report = &facts[sceneworks_core::moss_image::EDIT_REPORT_KEY];
        assert_eq!(
            report["adapter_sha256"],
            gen_core::moss_image_edit::EDIT_WEIGHTS_SHA256
        );
        assert_eq!(
            report["references"][0]["encoded_dimensions"],
            json!([32, 16])
        );
        assert_eq!(report["use_kv_cache"], false);
        let shared =
            sceneworks_core::workflow_share::sanitize_moss_edit_execution_report(report).unwrap();
        assert_eq!(shared["adapter_path"], "adapter_model.safetensors");
        assert!(!shared.to_string().contains("/private/"));
        probe.failure.set(true);
        assert!(moss_edit_render(
            &probe,
            &req,
            &refs,
            options,
            Default::default(),
            std::slice::from_ref(&identity),
            &mut |_| {}
        )
        .is_err());
        assert!(probe.edit_report.borrow().is_none());
        assert_eq!(
            probe.adapter_report_reads.get(),
            1,
            "failed output cannot read prior adapter reports"
        );
        assert_eq!(probe.reads.get(), 1);
        probe.failure.set(false);
        probe.cancel.set(true);
        assert!(matches!(
            moss_edit_render(
                &probe,
                &req,
                &refs,
                options,
                Default::default(),
                std::slice::from_ref(&identity),
                &mut |_| {}
            ),
            Err(WorkerError::Canceled(_))
        ));
        assert_eq!(
            probe.reads.get(),
            1,
            "canceled output must not publish a receipt"
        );
        assert_eq!(
            probe.adapter_report_reads.get(),
            1,
            "canceled output cannot read prior adapter reports"
        );
    }
    #[test]
    fn actual_provider_receipt_and_seed_reach_facts_and_failed_attempt_never_reads_stale_report() {
        let probe = Probe::new();
        let req = GenerationRequest {
            width: 16,
            height: 16,
            seed: Some(42),
            ..Default::default()
        };
        let mut progress = 0;
        let (images, facts) = moss_render(&probe, &req, &[], &mut |_| progress += 1).unwrap();
        assert_eq!(
            images[0].0, 173,
            "provider's actual seed, not requested seed"
        );
        assert_eq!(
            facts[sceneworks_core::moss_image::REPORT_KEY]["samples"][0]["seed"],
            173
        );
        assert_eq!(progress, 1);
        probe.failure.set(true);
        assert!(moss_render(&probe, &req, &[], &mut |_| {}).is_err());
        assert_eq!(probe.reads.get(), 1);
        assert!(probe.report.borrow().is_none());
    }
    // CI_IMPACT: literal pixels, metadata and provider reports only; no model/tensor execution.
    #[test]
    fn moss_custom_application_reports_join_selected_identity_into_image_facts() {
        let probe = Probe::new();
        let custom_path = PathBuf::from("custom.safetensors");
        *probe.adapter_reports.borrow_mut() = vec![gen_core::AdapterApplyReport {
            adapter_path: custom_path.clone(),
            applied: 252,
            skipped: vec![],
        }];
        let identity = lane_controls::AdapterIdentity {
            id: Some("selected-custom".into()),
            path: custom_path,
            sha256: Some("ab".repeat(32)),
            scale: 1.,
            kind: gen_core::AdapterKind::Lokr,
        };
        let req = GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        };
        let (_, facts) =
            moss_render(&probe, &req, std::slice::from_ref(&identity), &mut |_| {}).unwrap();
        let applied = &facts[lane_controls::ADAPTER_APPLY_KEY][0];
        assert_eq!(applied["id"], "selected-custom");
        assert_eq!(applied["path"], "custom.safetensors");
        assert_eq!(applied["sha256"], "ab".repeat(32));
        assert_eq!(applied["scale"], 1.);
        assert_eq!(applied["kind"], "lokr");
        assert_eq!(applied["applied"], 252);
        assert_eq!(applied["skipped"], json!([]));
        assert!(
            facts[sceneworks_core::moss_image::REPORT_KEY]["adapter_reports"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        let wrong = lane_controls::AdapterIdentity {
            path: PathBuf::from("other.safetensors"),
            ..identity
        };
        let (_, facts) = moss_render(&probe, &req, &[wrong], &mut |_| {}).unwrap();
        assert!(facts[lane_controls::ADAPTER_APPLY_KEY][0]["applied"].is_null());
    }

    #[test]
    fn cancellation_after_output_cannot_publish_a_previous_or_current_receipt() {
        let probe = Probe::new();
        probe.cancel.set(true);
        let req = GenerationRequest {
            width: 16,
            height: 16,
            ..Default::default()
        };
        assert!(matches!(
            moss_render(&probe, &req, &[], &mut |_| {}),
            Err(WorkerError::Canceled(_))
        ));
        assert_eq!(probe.reads.get(), 0);
    }
    #[test]
    fn admission_credits_only_same_entry_measured_owned_bytes_bounded_by_committed() {
        let budget = gen_core::MemoryBudget {
            total_bytes: 100,
            committed_bytes: 60,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 10,
        };
        assert!(moss_admit(40, 0, budget).is_err());
        assert!(moss_admit(40, 10, budget).is_ok());
        assert!(moss_admit(91, u64::MAX, budget).is_err());
    }

    #[test]
    fn lazy_renderer_warm_reentry_credits_actual_owned_census_without_double_charging() {
        let generator = Probe::new();
        let cold = gen_core::MemoryBudget {
            total_bytes: 47,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 2,
        };
        assert!(moss_admit(38, moss_owned_weight_bytes(&generator, 1), cold).is_ok());
        generator.owned_weight_bytes.set(30);
        let warm = gen_core::MemoryBudget {
            committed_bytes: 30,
            ..cold
        };
        assert!(
            moss_admit(38, 1, warm).is_err(),
            "loader delta alone double-charges lazy renderer"
        );
        assert!(moss_admit(38, moss_owned_weight_bytes(&generator, 1), warm).is_ok());
        assert!(
            moss_admit(38, moss_owned_weight_bytes(&Probe::new(), 1), warm).is_err(),
            "another mode/source entry owns no current weights"
        );
        assert!(
            moss_admit(46, u64::MAX, warm).is_err(),
            "even provider-owned credit is bounded by current committed bytes"
        );
    }
    #[test]
    fn each_mode_prepares_exactly_its_frozen_execution_assets_and_selected_corruption_fails_closed()
    {
        use std::collections::BTreeSet;
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .unwrap()
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            for mode in [
                MossImageAcceleration::Off,
                MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
                MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
            ] {
                if variant == MossImageVariant::Base && mode != MossImageAcceleration::Off {
                    continue;
                }
                let entry = manifest["models"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|m| m["id"] == variant.provider_id())
                    .unwrap();
                let selected = sceneworks_core::moss_image::selected_components(
                    entry.as_object().unwrap(),
                    &MossImageOptions {
                        acceleration: mode,
                        ..Default::default()
                    },
                )
                .unwrap();
                let actual: BTreeSet<_> = selected["downloads"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| {
                        row["files"].as_array().unwrap().iter().filter_map(|p| {
                            let repo = row["repo"].as_str().unwrap();
                            let path = p.as_str().unwrap();
                            moss_execution_asset_selected(mode, repo, path)
                                .then_some((repo.to_owned(), path.to_owned()))
                        })
                    })
                    .collect();
                let expected: BTreeSet<_> = MOSS_IMAGE_ASSETS
                    .iter()
                    .filter(|a| {
                        if a.repository == BASE_REPO {
                            return variant == MossImageVariant::Base;
                        }
                        if a.repository == SFT_REPO {
                            return !a.relative_path.starts_with("transformer/")
                                || (variant == MossImageVariant::Sft
                                    && mode
                                        != MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32));
                        }
                        if a.repository == DMD_REPO {
                            return match mode {
                                MossImageAcceleration::Off => false,
                                MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => [
                                    "lora.safetensors",
                                    "time_embedding.safetensors",
                                    "dataspace.safetensors",
                                ]
                                .contains(&a.relative_path),
                                MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => {
                                    a.relative_path.starts_with("transformer/")
                                        || a.relative_path == "dataspace.safetensors"
                                }
                            };
                        }
                        false
                    })
                    .map(|a| (a.repository.to_owned(), a.relative_path.to_owned()))
                    .collect();
                assert_eq!(actual, expected, "{variant:?}/{mode:?}");
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let root = hub
            .join("models--OpenMOSS-Team--MOSS-Image-Base")
            .join("snapshots")
            .join(BASE_REVISION);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("LICENSE"), b"corrupt selected bytes").unwrap();
        let settings = Settings::for_test(temp.path().join("data"));
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        let entry =
            json!({"downloads":[{"repo":BASE_REPO,"revision":BASE_REVISION,"files":["LICENSE"]}]})
                .as_object()
                .unwrap()
                .clone();
        let error = moss_pin_spec(
            &mut spec,
            &entry,
            &settings,
            MossImageAcceleration::Off,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("MOSS frozen component identity"),
            "{error}"
        );
    }
    #[test]
    fn corrupt_edit_package_is_rejected_during_file_preparation() {
        use gen_core::moss_image_edit::*;
        let temp = tempfile::tempdir().unwrap();
        let hub = temp.path().join("hub");
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HF_HOME", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let root = hub
            .join("models--chinoll--Moss-Image-Edit")
            .join("snapshots")
            .join(EDIT_REVISION);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("adapter_config.json"), b"corrupted").unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_component(EDIT_COMPONENT, WeightsSource::Dir(root));
        let entry=json!({"downloads":[{"repo":EDIT_REPOSITORY,"revision":EDIT_REVISION,"files":["adapter_config.json"]}]}).as_object().unwrap().clone();
        let error = moss_pin_spec(
            &mut spec,
            &entry,
            &Settings::for_test(temp.path().join("data")),
            MossImageAcceleration::Off,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("MOSS edit frozen component identity mismatch"),
            "{error}"
        );
        assert!(!spec.prepared_file_pins().is_prepared());
    }
    #[test]
    fn every_mode_and_content_change_has_an_independent_cache_identity() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("selected.safetensors");
        std::fs::write(&file, b"first").unwrap();
        let make = |mode: &str| {
            let mut spec = LoadSpec::new(WeightsSource::File(file.clone()))
                .with_resolved_route(format!("moss_image_sft:{mode}"));
            spec.prepare_with_file_pins([gen_core::PinnedWeightsFile::pin(&file).unwrap()])
                .unwrap();
            spec
        };
        let off = crate::generator_cache::LoadIdentity::try_from_load_spec(
            "moss_image_sft",
            &make("off"),
        )
        .unwrap();
        let mixed = crate::generator_cache::LoadIdentity::try_from_load_spec(
            "moss_image_sft",
            &make("unmerged_bf16"),
        )
        .unwrap();
        let full = crate::generator_cache::LoadIdentity::try_from_load_spec(
            "moss_image_sft",
            &make("full_fp32"),
        )
        .unwrap();
        assert_ne!(off, mixed);
        assert_ne!(mixed, full);
        std::fs::write(&file, b"other").unwrap();
        let replaced = crate::generator_cache::LoadIdentity::try_from_load_spec(
            "moss_image_sft",
            &make("off"),
        )
        .unwrap();
        assert_ne!(off, replaced);
    }

    /// Terminal-only product acceptance. Uses the normal dispatcher and loader, never the probe.
    /// Run this exact test alone after coordinator hardware admission; all artifacts stay outside
    /// the checkout. Snapshot env vars name already provisioned canonical HF snapshots (no fetch).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires coordinator-admitted native device and complete frozen MOSS weights"]
    async fn moss_real_native_product_worker_campaign() {
        use axum::{
            extract::State,
            routing::{get, post},
            Json, Router,
        };
        use std::sync::{Arc, Mutex};
        let admitted =
            std::env::var("MOSS_NATIVE_TERMINAL_ADMITTED").expect("coordinator admission required");
        #[cfg(target_os = "macos")]
        assert_eq!(admitted, "sc-26325-mlx");
        #[cfg(not(target_os = "macos"))]
        assert_eq!(admitted, "sc-26325-physical-gpu0");
        let output = PathBuf::from(
            std::env::var("MOSS_TRANSPORT_ARTIFACT_ROOT").expect("outside-repo output required"),
        )
        .join("product-worker");
        assert!(output.is_absolute());
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap();
        assert!(
            !output.starts_with(repository),
            "artifact output must be outside checkout"
        );
        std::fs::create_dir_all(&output).unwrap();
        assert!(
            !output.join("case-summary.json").exists(),
            "use a fresh campaign directory"
        );
        let source = |name| {
            std::fs::canonicalize(std::env::var(name).expect("provisioned snapshot env required"))
                .unwrap()
        };
        let base = source("MOSS_BASE_SNAPSHOT");
        let shared = source("MOSS_SHARED_SNAPSHOT");
        let dmd = source("MOSS_DMD_SNAPSHOT");
        let hub = base
            .ancestors()
            .nth(3)
            .expect("canonical HF snapshot")
            .to_owned();
        let data = output.join("data");
        let _cache = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ]);
        for (repo, revision, expected) in [
            (BASE_REPO, BASE_REVISION, &base),
            (SFT_REPO, SFT_REVISION, &shared),
            (DMD_REPO, DMD_REVISION, &dmd),
        ] {
            assert_eq!(
                std::fs::canonicalize(
                    huggingface_pinned_snapshot_dir(&data, repo, revision)
                        .expect("normal product source discovery")
                )
                .unwrap(),
                *expected
            );
        }
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .unwrap()
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
        let project = ProjectStore::new(data.clone(), "worker")
            .create_project("MOSS native product acceptance")
            .unwrap();
        let mut settings = Settings::from_env();
        settings.data_dir = data;
        settings.config_dir = output.join("config");
        settings.backend_mlx_enabled = true;
        settings.backend_candle_enabled = true;
        settings.heartbeat_seconds = 1;
        #[cfg(not(target_os = "macos"))]
        {
            assert_eq!(std::env::var("MOSS_PHYSICAL_GPU_INDEX").unwrap(), "0");
            assert!(!std::env::var("MOSS_EXPECTED_CUDA_UUID").unwrap().is_empty());
            assert_eq!(settings.gpu_id, "0");
        }
        #[derive(Clone)]
        struct Recorder {
            job: Value,
            posts: Arc<Mutex<Vec<Value>>>,
            cancel: bool,
        }
        let mut rows = Vec::new();
        // End on ordinary SFT so cancellation is a real warm-cache re-entry, after lazy weights.
        for (case, model, mode, cancel) in [
            ("base_off", "moss_image_base", "off", false),
            (
                "sft_unmerged_bf16",
                "moss_image_sft",
                "unmerged_bf16",
                false,
            ),
            ("sft_full_fp32", "moss_image_sft", "full_fp32", false),
            ("sft_off", "moss_image_sft", "off", false),
            ("sft_off_warm_cancel", "moss_image_sft", "off", true),
        ] {
            let entry = manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == model)
                .unwrap()
                .clone();
            let payload = json!({"projectId":project.id,"model":model,"mode":"text_to_image","modelManifestEntry":entry,"prompt":"日本語 <tag> a red ceramic cup on a white table {\"items\":[]}","width":64,"height":64,"count":if cancel {2}else{1},"seed":26325,"advanced":{"steps":if cancel {1000}else{1},"mossImage":{"acceleration":mode,"max_sequence_length":64},"mossImageExecutionReport":{"forged":true}}});
            let mut job_value = crate::iris_restore::tests::job_json(case, payload);
            job_value["type"] = json!("image_generate");
            let posts = Arc::new(Mutex::new(Vec::<Value>::new()));
            let recorder = Recorder {
                job: job_value.clone(),
                posts: posts.clone(),
                cancel,
            };
            let app=Router::new().route("/api/v1/jobs/:id",get(|State(s):State<Recorder>|async move{let mut job=s.job;job["cancelRequested"]=json!(s.cancel&&s.posts.lock().unwrap().iter().any(|p|p["message"].as_str().is_some_and(|m|m.contains("step "))));Json(job)}))
                .route("/api/v1/jobs/:id/progress",post(|State(s):State<Recorder>,Json(body):Json<Value>|async move{s.posts.lock().unwrap().push(body);Json(s.job)}))
                .route("/api/v1/workers/:id/heartbeat",post(||async{Json(json!({"id":"worker","gpuId":"0","status":"busy","currentJobId":null,"capabilities":[],"loadedModels":[],"registeredAt":"2026-10-18T00:00:00Z","lastSeenAt":"2026-10-18T00:00:00Z"}))})).with_state(recorder);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            settings.api_url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let api = ApiClient::new(&settings);
            let job: JobSnapshot = serde_json::from_value(job_value).unwrap();
            let result = run_image_generate_job(&api, &settings, &job).await;
            let events = posts.lock().unwrap().clone();
            server.abort();
            std::fs::write(
                output.join(format!("{case}.progress.json")),
                serde_json::to_vec_pretty(&events).unwrap(),
            )
            .unwrap();
            assert!(
                events.iter().any(|p| p["stage"] == "generating"
                    && p["message"].as_str().is_some_and(|m| m.contains("step "))),
                "actual denoise progress missing: {case}"
            );
            if cancel {
                assert!(
                    matches!(result, Err(WorkerError::Canceled(_))),
                    "{result:?}"
                );
                let terminal = events
                    .iter()
                    .find(|p| p["status"] == "canceled")
                    .expect("terminal canceled");
                assert!(
                    terminal["result"]["assetWrites"]
                        .as_array()
                        .unwrap()
                        .is_empty(),
                    "canceled run must publish no receipt/image"
                );
                rows.push(json!({"case":case,"model":model,"acceleration":mode,"status":"canceled","warmReentry":true,"images":0,"receipts":0}));
            } else {
                result.unwrap();
                let terminal = events
                    .iter()
                    .find(|p| p["status"] == "completed")
                    .expect("terminal completed");
                let facts = terminal["result"]["assetWrites"].as_array().unwrap();
                assert_eq!(facts.len(), 1);
                let fact = &facts[0];
                let report: MossImageExecutionReport = serde_json::from_value(
                    fact["rawAdapterSettings"][sceneworks_core::moss_image::REPORT_KEY].clone(),
                )
                .unwrap();
                assert_eq!(
                    report.data().resolved.acceleration,
                    serde_json::from_value::<MossImageOptions>(json!({"acceleration":mode}))
                        .unwrap()
                        .acceleration
                );
                assert_eq!(report.data().resolved.variant.provider_id(), model);
                assert_eq!(report.data().samples.len(), 1);
                let receipt_path = format!("{case}.receipt.json");
                std::fs::write(
                    output.join(&receipt_path),
                    serde_json::to_vec_pretty(&report).unwrap(),
                )
                .unwrap();
                let png_path = format!("{case}.png");
                let source = PathBuf::from(&project.path).join(fact["mediaPath"].as_str().unwrap());
                std::fs::copy(source, output.join(&png_path)).unwrap();
                let decoded = image::open(output.join(&png_path)).unwrap();
                assert_eq!((decoded.width(), decoded.height()), (64, 64));
                rows.push(json!({"case":case,"model":model,"acceleration":mode,"status":"completed","images":1,"receipts":1,"png":png_path,"receipt":receipt_path,"adapter":fact["adapter"]}));
            }
        }
        crate::generator_cache::evict_cached_generator()
            .await
            .unwrap();
        std::fs::write(output.join("case-summary.json"),serde_json::to_vec_pretty(&json!({"protocol":"sc-26325-product-worker-v1","backend":if cfg!(target_os="macos"){"mlx"}else{"candle"},"cases":rows})).unwrap()).unwrap();
    }
    include!("moss_dispatch_tests.rs");
}
