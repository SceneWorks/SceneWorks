use super::*;

pub(super) struct ModelConvertJobPayload<'a> {
    pub(super) model: &'a Value,
    pub(super) model_id: &'a str,
    pub(super) mlx: &'a JsonObject,
    pub(super) source_repo: &'a str,
    pub(super) output_dir: &'a FsPath,
    pub(super) quantize_only: bool,
    pub(super) request: &'a ModelConvertRequest,
}

pub(super) fn build_model_convert_job_payload(input: ModelConvertJobPayload<'_>) -> JsonObject {
    let mut job_payload = JsonObject::new();
    job_payload.insert(
        "modelId".to_owned(),
        Value::String(input.model_id.to_owned()),
    );
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            input
                .model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(input.model_id)
                .to_owned(),
        ),
    );
    // This boolean is catalog-derived. Repository, source revision, output variant, and fixed tier
    // remain resolver/job facts and are never accepted from the client.
    insert_memory_calibration_provenance_requirement(&mut job_payload, input.model, true);
    job_payload.insert(
        "sourceRepo".to_owned(),
        Value::String(input.source_repo.to_owned()),
    );
    job_payload.insert(
        "outputDir".to_owned(),
        Value::String(input.output_dir.display().to_string()),
    );
    job_payload.insert("dtype".to_owned(), Value::String("bfloat16".to_owned()));
    // Optional converter discriminator + inputs (sc-2235). Default (absent) is the
    // mlx-video Wan converter. A FLUX.2-klein community fine-tune declares
    // `mlx.converter` + the single-file source + the base repo whose
    // VAE/text-encoder/tokenizer are borrowed during assembly.
    if let Some(converter) = input
        .mlx
        .get("converter")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert("converter".to_owned(), Value::String(converter.to_owned()));
    }
    if let Some(source_file) = input
        .mlx
        .get("convertSourceFile")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert(
            "sourceFile".to_owned(),
            Value::String(source_file.to_owned()),
        );
    }
    if let Some(base_repo) = input
        .mlx
        .get("convertBaseRepo")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert("baseRepo".to_owned(), Value::String(base_repo.to_owned()));
    }
    // The quant-tier subdir under `convertBaseRepo` to borrow the base components from (sc-14978):
    // the FLUX.2-klein re-host keeps each tier in its own subdir, so the borrowed
    // VAE/text-encoder/tokenizer live under `<tier>/`, not the snapshot root. Absent for root-layout
    // diffusers bases.
    if let Some(base_subdir) = input
        .mlx
        .get("convertBaseSubdir")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert(
            "baseSubdir".to_owned(),
            Value::String(base_subdir.to_owned()),
        );
    }
    if input.quantize_only {
        job_payload.insert("quantizeOnly".to_owned(), Value::Bool(true));
    }
    if let Some(bits) = input.request.quantize_bits {
        job_payload.insert("quantizeBits".to_owned(), Value::from(bits));
    }
    if let Some(group_size) = input.request.quantize_group_size {
        job_payload.insert("quantizeGroupSize".to_owned(), Value::from(group_size));
    }
    job_payload
}

/// Convert a model's native checkpoint into the local MLX format (macOS/Apple
/// Silicon). Only valid for models whose manifest declares `mlx.requiresConversion`
/// (Wan TI2V-5B/I2V-A14B, LTX-2.3 eros, FLUX.2-klein); turnkey MLX models need no conversion. The
/// native source checkpoint must already be downloaded; the worker converts it in-process via the
/// linked `mlx-gen-*` converters, selected by the `mlx.converter` discriminator (sc-3240).
pub(crate) async fn create_model_convert_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelConvertRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    // Derive and confine the route-selected destination before catalog assembly performs any
    // filesystem/cache inspection. Path-shaped IDs must fail at the request boundary rather than
    // falling through to a misleading catalog 404 after that work.
    let output_dir = state
        .settings
        .data_dir
        .join("models")
        .join("mlx")
        .join(&model_id);
    sceneworks_worker::resolve_model_convert_output(
        &state.settings.data_dir,
        &output_dir.display().to_string(),
    )
    .map_err(|error| ApiError::bad_request(error.to_string()))?;

    let model = require_model_by_id(&state, &model_id).await?;
    ensure_model_not_cleanup_only(&model)?;
    let mlx = model
        .get("mlx")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("Model has no MLX variant to convert"))?;
    let requires_conversion = mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let quantize = payload.quantize_bits.is_some();
    // Two sources: models that require conversion read the native checkpoint (convertSourceRepo);
    // turnkey MLX models (a pre-converted bf16 `repo`) carried a legacy in-place quantize path. The
    // native Rust converters don't re-quantize an already-converted dir, so the worker now rejects
    // `quantize_only` with a clear message (sc-3240) — quantize during native conversion instead.
    let (source_repo, quantize_only) = if requires_conversion {
        let repo = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("MLX conversion source repo is not configured"))?;
        (repo.to_owned(), false)
    } else if quantize {
        let repo = mlx
            .get("repo")
            .and_then(Value::as_str)
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("Model has no MLX repo to quantize"))?;
        (repo.to_owned(), true)
    } else {
        return Err(ApiError::bad_request(
            "Model does not require MLX conversion",
        ));
    };
    // Pre-flight the source weights (sc-14708 follow-up). The worker resolves the same HF snapshot and
    // fails the job when the checkpoint is absent, which reads as a defect to the user: the three Anima
    // variants share `circlestone-labs/Anima`, their per-variant downloads are serialized, and the
    // sibling cards flip to *downloaded* the moment THEIR files land — so converting the variant whose
    // 4 GB DiT is still streaming produced a bare "Anima source DiT is missing." Refuse it here, while
    // the request can still carry an explanation. Only `requiresConversion` reads a native checkpoint;
    // the legacy quantize-only path has its own worker-side rejection.
    if requires_conversion {
        if let Some(download) = store_call(state.clone(), {
            let model_id = model_id.clone();
            move |store, _timeout| store.find_active_model_download_job(&model_id)
        })
        .await?
        {
            let model_name = model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id.as_str());
            return Err(ApiError::conflict(format!(
                "{model_name} is still downloading ({percent}%). Wait for the download to finish \
                 before converting it.",
                percent = (download.progress.as_f64().unwrap_or(0.0) * 100.0).round() as i64,
            )));
        }
        if let Some(source_file) = mlx
            .get("convertSourceFile")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            match sceneworks_worker::convert_source_state(
                &state.settings.data_dir,
                &source_repo,
                source_file,
            ) {
                sceneworks_worker::ConvertSourceState::Ready => {}
                sceneworks_worker::ConvertSourceState::RepoNotCached => {
                    return Err(ApiError::conflict(format!(
                        "{source_repo} is not downloaded yet. Download it before converting."
                    )));
                }
                sceneworks_worker::ConvertSourceState::FileMissing => {
                    return Err(ApiError::conflict(format!(
                        "{source_file} has not finished downloading from {source_repo}. Wait for \
                         the download to complete, then convert."
                    )));
                }
            }
        }
    }

    let job_payload = build_model_convert_job_payload(ModelConvertJobPayload {
        model: &model,
        model_id: &model_id,
        mlx,
        source_repo: &source_repo,
        output_dir: &output_dir,
        quantize_only,
        request: &payload,
    });

    let job = create_generation_job(
        state,
        JobType::ModelConvert,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}
