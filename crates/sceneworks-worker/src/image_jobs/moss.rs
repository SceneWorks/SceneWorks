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
            let _identity = gen_core::moss_image::moss_image_asset(repo, revision, relative).ok_or_else(|| WorkerError::InvalidPayload(format!("MOSS component file is not in the frozen asset contract: {repo}/{relative}")))?;
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
        let pin = gen_core::PinnedWeightsFile::pin(&path).map_err(|error| crate::classify_engine_error(&format!("MOSS component {relative} missing/incomplete; install or repair the selected package"), error))?;
        gen_core::moss_image::verify_moss_image_asset(&pin, repo, revision, relative).map_err(
            |error| crate::classify_engine_error("MOSS frozen component identity", error),
        )?;
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
    moss_pin_spec_with_adapters(spec, entry, settings, mode, Vec::new(), on_file)
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
struct MossPreparedPreload {
    native: gen_core::GenerationRequest,
    variant: gen_core::moss_image::MossImageVariant,
    resolved: gen_core::moss_image::MossImageResolved,
    spec: LoadSpec,
    peak: u64,
}

// Shared source/spec preparation and the unchanged production header math.
// This synchronous boundary creates neither a generator nor model tensors nor GPU work.
// The async job caller owns progress posting; metadata-only forecasts pass a no-op callback.
fn prepare_moss_preload(
    settings: &Settings,
    payload: &JsonObject,
    request: &ImageRequest,
    on_file: &mut dyn FnMut(usize, usize),
) -> WorkerResult<MossPreparedPreload> {
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
    let root = huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        variant.repository(),
        variant.revision(),
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{} frozen snapshot missing; install or repair this model in Models",
            request.model
        ))
    })?;
    let mut spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_resolved_route(format!("{}:{:?}", request.model, resolved.acceleration));
    spec = attach_required_components(spec, variant.provider_id(), &entry, settings)?;
    for row in entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(id) = row
            .get("componentId")
            .and_then(Value::as_str)
            .filter(|id| id.starts_with("moss-dmd-"))
        {
            let source = crate::model_jobs::resolve_optional_component(
                &Value::Object(entry.clone()),
                id,
                settings,
            )
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "MOSS component {id} missing; install the selected 8-step package in Models"
                ))
            })?;
            spec = spec.with_component(id, source);
        }
    }
    let prepared_adapters = resolve_prepared_adapters(request, settings)?;
    spec = spec.with_adapters(prepared_adapters.specs);
    moss_pin_spec_with_adapters(
        &mut spec,
        &entry,
        settings,
        resolved.acceleration,
        prepared_adapters.pins,
        on_file,
    )?;
    #[cfg(target_os = "macos")]
    let peak = {
        let facts = moss_memory::selected_asset_facts(variant, &spec, resolved.acceleration)
            .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
        let expanded = if native.prompt_batch.is_empty() {
            1
        } else {
            request.image_count()
        };
        let branches = if resolved.acceleration == gen_core::moss_image::MossImageAcceleration::Off
        {
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
            .ok_or_else(|| WorkerError::InvalidPayload("MOSS memory estimate overflow".into()))?
    };
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let peak = moss_memory::selected_preload_estimate(
        variant,
        &spec,
        &native,
        gen_core::moss_image::MossImageAdvancedInputs::default(),
    )
    .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?
    .peak_bytes;
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
    let request = ImageRequest::from_payload(&job.payload);
    let prepared = prepare_moss_preload(settings, &job.payload, &request, &mut |_, _| {})?;
    let provenance = if cfg!(target_os = "macos") {
        "production prepare_moss_preload -> mlx moss_memory::selected_asset_facts.base_bytes + overlay_bytes + request_workspace_bytes (exact selected request/adapter pins)"
    } else {
        "production prepare_moss_preload -> candle moss_memory::selected_preload_estimate.peak_bytes (exact selected request/adapter pins, includes owned-shard staging)"
    };
    let adapters = prepared.spec.adapters.iter().map(|adapter| {
        let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&adapter.path)
            .map_err(|e| WorkerError::InvalidPayload(e.to_string()))?;
        let tensor_bytes = headers.iter().try_fold(0_u64, |sum, header| sum.checked_add(header.data_bytes)
            .ok_or_else(|| WorkerError::InvalidPayload("MOSS adapter header bytes overflow".into())))?;
        let pin=gen_core::PinnedWeightsFile::pin(&adapter.path).map_err(|e|WorkerError::InvalidPayload(e.to_string()))?;
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
    let request = &plan.request;
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
    let prepared = prepare_moss_preload(settings, &job.payload, request, &mut |done, total| {
        let _ = pin_progress.send((done, total));
    });
    drop(pin_progress);
    let _ = poster.await;
    let MossPreparedPreload {
        mut native,
        variant,
        resolved,
        spec,
        peak,
    } = prepared?;
    let adapter_identities =
        lane_controls::adapter_identities(api, settings, job, request, &spec.adapters).await;
    let settings_owned = settings.clone();
    let cold_settings = settings_owned.clone();
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
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state_using_owned(
        job.id.clone(),
        engine_id,
        0,
        spec,
        "MOSS exact native load".into(),
        move |id, spec| {
            moss_admit(
                peak,
                0,
                moss_live_budget(&cold_settings)
                    .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?,
            )
            .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?;
            load_generator(id, spec)
        },
        move |generator, _state, _loaded_policy, warm_policy, _external, owned, tx, cancel| {
            warm_policy
                .decline(crate::execution_planner::ServedAsIsReason::RouteHasNoRequestScopedMemory);
            native.cancel = cancel.clone();
            moss_admit(
                peak,
                moss_owned_weight_bytes(generator, owned),
                moss_live_budget(&settings_owned)?,
            )?;
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
    #[cfg(target_os = "macos")]
    let (backend, adapter) = (
        "mlx",
        if variant == gen_core::moss_image::MossImageVariant::Base {
            "mlx_moss_image_base"
        } else {
            "mlx_moss_image_sft"
        },
    );
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (backend, adapter) = ("candle", candle_adapter_label(&request.model));
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
    fn completed_receipt(req: &GenerationRequest) -> MossImageExecutionReport {
        let resolved = MossImageResolved::resolve(MossImageVariant::Sft, req).unwrap();
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
                    repository: if dmd { DMD_REPO } else { SFT_REPO }.into(),
                    revision: if dmd { DMD_REVISION } else { SFT_REVISION }.into(),
                    relative_path: path,
                    sha256: hash.clone(),
                }
            })
            .collect();
        let input = MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: hash.clone(),
            sequence_length: 3,
            valid_tokens: 3,
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
                negative: (mode == MossImageAcceleration::Off).then_some(input),
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
    struct Probe {
        descriptor: gen_core::ModelDescriptor,
        report: RefCell<Option<MossImageExecutionReport>>,
        failure: Cell<bool>,
        cancel: Cell<bool>,
        reads: Cell<usize>,
        owned_weight_bytes: Cell<u64>,
        adapter_reports: RefCell<Vec<gen_core::AdapterApplyReport>>,
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
                failure: Cell::new(false),
                cancel: Cell::new(false),
                reads: Cell::new(0),
                owned_weight_bytes: Cell::new(0),
                adapter_reports: RefCell::new(vec![]),
            }
        }
    }
    impl Generator for Probe {
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
}
