/// Fetch the `files` glob patterns from `repo`@`revision` into the shared Hugging Face hub cache
/// using the NATIVE downloader — real byte-level progress, the sc-11939 forward-progress stall
/// watchdog, HTTP-Range resume, and Windows blob hardlinks — and return the repo's snapshot dir.
/// This is the on-demand generation/convert-time counterpart to `run_model_download_job`'s cache
/// download; it writes NO install marker (these are cache warm-ups, not user-visible installs).
///
/// Replaces the retired Python `hf`/`huggingface-cli` path (sc-12227 / sc-12232, issue #1554): that
/// subprocess pinned job progress at 12% for the whole transfer, had no stall watchdog, and only ran
/// when `hf` was on PATH (so a host without it silently couldn't pull these tiers). The native path
/// also honors a custom `huggingface_base_url`, which the CLI could not. Returns the snapshot dir, or
/// `None` only when the repo still has no snapshot after the fetch (e.g. the filter matched zero
/// files) so callers keep their existing "resolve the tier subdir, else fall back" presence checks.
pub(crate) async fn ensure_hf_files_cached(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    revision: &str,
    files: &[String],
) -> WorkerResult<Option<PathBuf>> {
    let repo = validate_hf_repo_id(repo)?;
    validate_hf_download_inputs(repo, revision, files)?;
    let repo_dir = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    let client = streaming_download_client();
    let snapshot = HuggingFaceSnapshot::resolve(&client, settings, repo, revision, files).await?;
    // A filter that matches zero files is a no-op (same net effect as the old `hf download --include`
    // with no match): don't create an empty transfer, and leave the caller's presence check to fall
    // back to a smaller complete tier.
    if !snapshot.files.is_empty() {
        let mut progress = DownloadProgress::new(
            repo,
            0,
            snapshot.total_bytes(),
            progress_report_interval(settings),
        );
        download_snapshot_into_cache(
            &DownloadContext {
                api,
                client: &client,
                settings,
                job_id: &job.id,
                cancel_message: "Model download canceled by user.",
                fresh_download: false,
            },
            &repo_dir,
            revision,
            &snapshot,
            &mut progress,
        )
        .await?;
    }
    Ok(huggingface_snapshot_dir(&settings.data_dir, repo))
}

/// Reject repo / revision / include-pattern inputs that could escape the intended Hugging Face path
/// before they reach the native downloader's URL builder (`downloads::quote_path`): leading slashes,
/// `..` traversal, backslashes, control characters, and unsupported characters. Applied by
/// `ensure_hf_files_cached` to the on-demand tier fetches (the user-facing model/LoRA download jobs
/// build their URLs the same way).
pub(crate) fn validate_hf_download_inputs(
    repo: &str,
    revision: &str,
    files: &[String],
) -> WorkerResult<()> {
    validate_hf_repo_id(repo)?;
    validate_hf_revision(revision)?;
    for pattern in files {
        validate_hf_include_pattern(pattern)?;
    }
    Ok(())
}

fn validate_hf_repo_id(repo: &str) -> WorkerResult<&str> {
    sceneworks_core::hf_repo_id::parse(repo)
        .map_err(|error| WorkerError::InvalidPayload(error.to_owned()))
}

fn validate_hf_revision(revision: &str) -> WorkerResult<()> {
    // Shared guard for both HF CLI (`--revision` is arg-isolated) and direct HTTP
    // download (`downloads::quote_path` percent-encodes the revision in URLs).
    // Keep leading slash / traversal rejection here so both paths fail closed.
    let parts = safe_hf_path_parts(revision, "Hugging Face revision")?;
    if !revision.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '/')
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face revision `{revision}`: only letters, numbers, `.`, `_`, `-`, and `/` are allowed."
        )));
    }
    for part in parts {
        if part == "." || part == ".." {
            return Err(WorkerError::InvalidPayload(format!(
                "Invalid Hugging Face revision `{revision}`: traversal components are not allowed."
            )));
        }
    }
    Ok(())
}

fn validate_hf_include_pattern(pattern: &str) -> WorkerResult<()> {
    let _parts = safe_hf_path_parts(pattern, "Hugging Face include pattern")?;
    if pattern.starts_with('/') || pattern.contains('\\') {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face include pattern `{pattern}`: absolute paths and backslashes are not allowed."
        )));
    }
    if pattern.split('/').any(|part| part == "." || part == "..") {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face include pattern `{pattern}`: traversal components are not allowed."
        )));
    }
    if !pattern.chars().all(|character| {
        character.is_ascii_alphanumeric()
            || matches!(
                character,
                '_' | '-' | '.' | '/' | '*' | '?' | '[' | ']' | '{' | '}' | ',' | '!'
            )
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "Invalid Hugging Face include pattern `{pattern}`: unsupported characters are not allowed."
        )));
    }
    Ok(())
}

fn safe_hf_path_parts<'a>(value: &'a str, label: &str) -> WorkerResult<Vec<&'a str>> {
    if value.is_empty() {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot be empty."
        )));
    }
    if value.starts_with('-') {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot start with `-`."
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot contain control characters."
        )));
    }
    if value.starts_with('/') || value.contains('\\') {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} must be a relative Hugging Face identifier."
        )));
    }
    let parts: Vec<_> = value.split('/').collect();
    if parts.iter().any(|part| part.is_empty()) {
        return Err(WorkerError::InvalidPayload(format!(
            "{label} cannot contain empty path components."
        )));
    }
    Ok(parts)
}

/// Actionable failure message for a LoRA-import safetensors inspection error.
pub(crate) fn lora_family_detection_error(error: &SafetensorsHeaderError) -> String {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect imported LoRA file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Imported LoRA file has an invalid safetensors header.".to_owned()
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => format!(
            "Imported LoRA file is incomplete or corrupt ({actual} bytes on disk, but its header \
             declares at least {declared} bytes of tensor data) — the download was likely \
             interrupted. Re-import the complete file."
        ),
    }
}

/// Whether a safetensors inspection error means the file on disk is itself bad
/// (a valid-but-truncated or malformed file) rather than a transient read
/// failure — i.e. it should be discarded so it can't be picked up as installed.
pub(crate) fn safetensors_file_is_unusable(error: &SafetensorsHeaderError) -> bool {
    matches!(
        error,
        SafetensorsHeaderError::InvalidHeader | SafetensorsHeaderError::IncompleteData { .. }
    )
}

pub(crate) async fn run_lora_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    validate_lora_import_digest_contract(&job.payload)?;
    let repo = optional_payload_string(&job.payload, "repo")
        .map(validate_hf_repo_id)
        .transpose()?;
    let source_url = optional_payload_string(&job.payload, "sourceUrl");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "loraId")
        .or_else(|| optional_payload_string(&job.payload, "name"))
        .map(str::to_owned)
        .or_else(|| repo.map(str::to_owned))
        .or_else(|| source_url.and_then(|value| lora_source_url_file_stem(value).ok()))
        .map(|value| safe_download_dir(&value))
        .unwrap_or_else(|| {
            source_path
                .and_then(|path| {
                    Path::new(path)
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .map(safe_download_dir)
                })
                .unwrap_or_else(|| "lora".to_owned())
        });
    let target_dir = resolve_lora_import_target(
        settings,
        &job.payload,
        settings.data_dir.join("loras").join(&target_name),
    )?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Importing,
            0.1,
            "Importing LoRA.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "LoRA import canceled before transfer started.",
    )
    .await?;

    if let Some(repo) = repo {
        let files = payload_string_array(&job.payload, "files");
        let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
        // Trust-boundary validation for the LAN-exposed jobs API (epic 4484): a LoRA HF import takes a
        // caller-supplied repo/revision/files and resolves them into download URLs and on-disk paths,
        // so reject anything that could escape the intended Hugging Face path (sc-13583).
        let repo = validate_hf_repo_id(repo)?;
        validate_hf_download_inputs(repo, revision, &files)?;
        let snapshot =
            HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
        let mut progress = DownloadProgress::new(
            repo,
            0,
            snapshot.total_bytes(),
            progress_report_interval(settings),
        );
        // LoRA HF imports intentionally skip the model install marker for parity with the Python worker.
        download_snapshot(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "LoRA import canceled by user.",
                fresh_download: false,
            },
            &target_dir,
            &snapshot,
            &mut progress,
        )
        .await?;
    } else if let Some(source_path) = source_path {
        let prefer_move = payload_bool(&job.payload, "uploadedSourcePath");
        // sc-8803: the source path is client-supplied over the unauthenticated jobs
        // API; confine it (staged-upload cache for uploads, app-managed roots
        // otherwise) before the copy/move reaches the filesystem.
        let source_path = resolve_import_source_path(
            settings,
            &job.payload,
            source_path,
            "lora-uploads",
            "LoRA import sourcePath",
        )?;
        if let Some(secondary_source_path) =
            optional_payload_string(&job.payload, "secondarySourcePath")
        {
            let secondary_source_path = resolve_import_source_path(
                settings,
                &job.payload,
                secondary_source_path,
                "lora-uploads",
                "LoRA import secondarySourcePath",
            )?;
            // Paired Wan A14B MoE upload (sc-1991): write both halves into one
            // record under the high/low_noise convention so the high half resolves
            // as the primary (transformer) and the low half as the transformer_2
            // sibling, regardless of the user's original upload filenames.
            let (high_name, low_name) = wan_moe_pair_filenames(&target_name);
            import_lora_source_file_as(&source_path, &target_dir, &high_name, prefer_move).await?;
            import_lora_source_file_as(&secondary_source_path, &target_dir, &low_name, prefer_move)
                .await?;
        } else {
            import_lora_source_path(&source_path, &target_dir, prefer_move).await?;
        }
    } else if let Some(source_url) = source_url {
        download_lora_source_url(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "LoRA import canceled by user.",
                fresh_download: false,
            },
            source_url,
            &target_dir,
        )
        .await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "LoRA import failed.",
            Some("Provide repo, sourceUrl, or sourcePath for LoRA import".to_owned()),
        )
        .await;
    }

    // sc-6137: verify a caller-supplied content digest. HF repo files are already
    // per-file verified during download; this covers source-URL/uploaded imports where
    // a SHA-256 is supplied out-of-band. `verify_file_sha256` removes the file on
    // mismatch, so a corrupt artifact is never left behind.
    if let Some(expected_sha256) = optional_payload_string(&job.payload, "expectedSha256") {
        if let Some(file) = first_safetensors_path(&target_dir) {
            if let Err(error) = verify_file_sha256(
                api,
                settings,
                &job.id,
                &file,
                expected_sha256,
                "Imported LoRA",
            )
            .await
            {
                return fail_job(api, &job.id, "LoRA import failed.", Some(error.to_string()))
                    .await;
            }
        }
    }

    // Validate the exact MiniMax-H3 trainer namespace after a repo/URL transfer, at the first point
    // where those routes have a local header. Local/upload imports already run the same shared
    // validator in rust-api. A partial/lookalike export must fail before its install marker is
    // written and long before a render tries to bind `lora_unet_blocks_*` at the engine.
    if let Some(adapter_path) = first_safetensors_path(&target_dir) {
        if let Ok(header) = sceneworks_core::lora_family::read_safetensors_header(&adapter_path) {
            let h3_expected =
                optional_payload_string(&job.payload, "family").is_some_and(|family| {
                    sceneworks_core::lora_family::canonical_lora_family(family) == "minimax-h3"
                });
            if let Err(error) = validate_minimax_h3_trainer_header(&header, h3_expected) {
                return fail_job(
                    api,
                    &job.id,
                    "LoRA import failed.",
                    Some(format!(
                        "Unsupported MiniMax-H3 adapter namespace or layout: {error}"
                    )),
                )
                .await;
            }
        }
    }

    // Family plus declared metadata (sc-14057). The declared half matters most for the repo/URL
    // routes: their file did not exist when the API queued the job, so this is the only point at
    // which the adapter can describe itself.
    let (detected_family, adapter_metadata) = match inspect_adapter_in_dir(&target_dir) {
        Ok(inspected) => inspected,
        Err(error) => {
            // A truncated/corrupt file must not linger: `lora_is_installed` (API side)
            // treats any `*.safetensors` under the dir as installed, so a rejected
            // partial would still look selectable. Discard it before failing (sc-6072).
            if safetensors_file_is_unusable(&error) {
                if let Some(bad_file) = first_safetensors_path(&target_dir) {
                    let _ = tokio::fs::remove_file(&bad_file).await;
                }
            }
            let detail = lora_family_detection_error(&error);
            return fail_job(api, &job.id, "LoRA import failed.", Some(detail)).await;
        }
    };
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    if let (Some(supplied), None) = (&supplied_family, &detected_family) {
        tracing::info!(
            event = "lora_import_architecture_inconclusive",
            jobId = %job.id,
            family = %supplied,
            "LoRA import: architecture detection inconclusive; accepting supplied family"
        );
    }
    // Shared reconcile policy + canonicalization (krea2/krea-2/krea_2 → krea_2),
    // matching the API pre-flight so a manual family override agrees with detection.
    let resolved_family = match reconcile_detected_family(supplied_family, detected_family) {
        Ok(family) => family,
        Err(mismatch) => {
            return fail_job(
                api,
                &job.id,
                "LoRA import failed.",
                Some(format!(
                    "LoRA file appears to be a {} model, but family was declared as {}. Re-import with family {} or pick a different file.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await;
        }
    };
    if let Some(detail) = ltx_import_base_model_error(resolved_family.as_deref(), &job.payload) {
        return fail_job(api, &job.id, "LoRA import failed.", Some(detail)).await;
    }

    write_lora_install_marker(&target_dir, &job.payload, &job.id).await?;
    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let mut manifest_entry = manifest_entry;
        if let Some(family) = resolved_family {
            manifest_entry
                .entry("family")
                .or_insert(Value::String(family));
        }
        // …and the same for what the file declares about itself (sc-14057). The API writes these
        // through the identical `sceneworks_core` writer for a local/uploaded source; for a
        // repo/URL import there was no file to read at queue time, so this is where an adapter
        // ingested by URL stops being described more poorly than the same adapter dropped in as a
        // file. `or_insert` semantics: anything already recorded wins.
        apply_adapter_metadata_to_manifest_entry(&mut manifest_entry, &adapter_metadata);
        let manifest_path = lora_manifest_target(settings, &job.payload)?;
        upsert_manifest_entry(&manifest_path, "loras", manifest_entry).await?;
    }

    let mut result = JsonObject::new();
    result.insert(
        "repo".to_owned(),
        repo.map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "sourceUrl".to_owned(),
        source_url
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "LoRA import completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

fn ltx_import_base_model_error(
    resolved_family: Option<&str>,
    payload: &JsonObject,
) -> Option<String> {
    if resolved_family != Some("ltx-video") {
        return None;
    }
    let recorded = optional_payload_string(payload, "baseModel").or_else(|| {
        payload
            .get("manifestEntry")
            .and_then(Value::as_object)
            .and_then(|entry| entry.get("baseModel"))
            .and_then(crate::payload::trimmed_value_str)
    });
    match recorded {
        None => Some(
            "LTX LoRA imports require an exact base model. Re-import in Model Manager, choose the \
             ltx-video family, and select LTX-2.3 or LTX-2.5 before queueing."
                .to_owned(),
        ),
        Some("ltx_2_3" | "ltx_2_3_eros" | "ltx_2_5") => None,
        Some(other) => Some(format!(
            "LTX LoRA imports require baseModel ltx_2_3, ltx_2_3_eros, or ltx_2_5; got {other}."
        )),
    }
}



/// A single `expectedSha256` cannot identify both files in a paired Wan MoE upload. Reject that
/// ambiguous contract before heartbeat, target creation, or moving either staged upload. Single-file
/// imports retain their existing digest verification.
pub(crate) fn validate_lora_import_digest_contract(payload: &JsonObject) -> WorkerResult<()> {
    if optional_payload_string(payload, "secondarySourcePath").is_some()
        && optional_payload_string(payload, "expectedSha256").is_some()
    {
        return Err(WorkerError::InvalidPayload(
            "LoRA import expectedSha256 is ambiguous with secondarySourcePath; provide a paired \
             upload without a single-file digest."
                .to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn model_family_detection_error(error: SafetensorsHeaderError) -> String {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            format!("Unable to inspect imported model file: {io_error}")
        }
        SafetensorsHeaderError::InvalidHeader => {
            "Imported model file has an invalid safetensors header.".to_owned()
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => format!(
            "Imported model file is incomplete or corrupt ({actual} bytes on disk, but its header \
             declares at least {declared} bytes of tensor data) — the download was likely \
             interrupted. Re-download the complete file."
        ),
    }
}

/// Outcome of re-checking a downloaded model's architecture family against the
/// catalog-declared family (sc-1663). Kept pure (no API I/O) so the decision is
/// unit-testable; [`reconcile_downloaded_model_family`] maps it to a job failure.
#[derive(Debug)]
pub(crate) enum DownloadFamilyCheck {
    /// Detection agrees, is inconclusive, or no family was declared — proceed.
    Proceed,
    /// The catalog declared one family but the weights are confidently another.
    Mismatch(FamilyMismatch),
    /// A safetensors file was found but its header could not be read.
    DetectionFailed(SafetensorsHeaderError),
}

/// Re-detect the architecture family of the downloaded weights and reconcile it
/// against the catalog-declared `supplied` family. A missing declaration or an
/// inconclusive detector result proceeds — the curated catalog is trusted when there
/// is no confident contradicting signal — so this never blocks a legitimate
/// download; only a confident conflict is a mismatch.
pub(crate) fn check_downloaded_model_family(
    supplied: Option<String>,
    model_dir: &Path,
) -> DownloadFamilyCheck {
    let detected = match detect_model_family(model_dir) {
        Ok(detected) => detected,
        Err(error) if downloaded_model_detection_io_error_is_inconclusive(&error) => {
            return DownloadFamilyCheck::Proceed;
        }
        Err(error) => return DownloadFamilyCheck::DetectionFailed(error),
    };
    match reconcile_detected_family(supplied, detected) {
        Ok(_) => DownloadFamilyCheck::Proceed,
        Err(mismatch) => DownloadFamilyCheck::Mismatch(mismatch),
    }
}

pub(crate) fn downloaded_model_detection_io_error_is_inconclusive(
    error: &SafetensorsHeaderError,
) -> bool {
    let SafetensorsHeaderError::Io(io_error) = error else {
        return false;
    };
    // Hugging Face snapshots may contain symlinks/reparse points into `blobs`.
    // On some Windows machines, opening those links fails with ERROR_UNTRUSTED_
    // MOUNT_POINT (448). The download is already complete; this optional family
    // check should become inconclusive rather than failing the model install.
    io_error.raw_os_error() == Some(448)
        || io_error
            .to_string()
            .to_ascii_lowercase()
            .contains("untrusted mount point")
}

/// Enforce family parity with model import on a completed download: verify the
/// downloaded weights match the catalog-declared family and fail the job on a
/// confident mismatch or a clearly invalid header. Windows cache-link traversal
/// errors are treated as inconclusive because the download itself has completed.
/// Returns `Ok(true)` when the download may complete, `Ok(false)` when the job was
/// already failed and the caller should return.
pub(crate) async fn reconcile_downloaded_model_family(
    api: &ApiClient,
    job: &JobSnapshot,
    model_dir: &Path,
) -> WorkerResult<bool> {
    let supplied = optional_payload_string(&job.payload, "family").map(str::to_owned);
    match check_downloaded_model_family(supplied, model_dir) {
        DownloadFamilyCheck::Proceed => Ok(true),
        DownloadFamilyCheck::DetectionFailed(error) => {
            let detail = match error {
                SafetensorsHeaderError::Io(io_error) => {
                    format!("Unable to inspect downloaded model file: {io_error}")
                }
                SafetensorsHeaderError::InvalidHeader => {
                    "Downloaded model file has an invalid safetensors header.".to_owned()
                }
                SafetensorsHeaderError::IncompleteData { declared, actual } => format!(
                    "Downloaded model file is incomplete or corrupt ({actual} bytes on disk, but \
                     its header declares at least {declared} bytes of tensor data) — the download \
                     was likely interrupted. Re-download the complete file."
                ),
            };
            fail_job(api, &job.id, "Model download failed.", Some(detail)).await?;
            Ok(false)
        }
        DownloadFamilyCheck::Mismatch(mismatch) => {
            fail_job(
                api,
                &job.id,
                "Model download failed.",
                Some(format!(
                    "Downloaded model files appear to be {}, but the catalog declared family {}. Fix the catalog entry to family {} or correct the download source.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await?;
            Ok(false)
        }
    }
}

async fn stable_model_file_sha256(
    api: &ApiClient,
    settings: &Settings,
    job_id: &str,
    file: &Path,
) -> WorkerResult<(String, ModelFileIdentity)> {
    let before = model_file_identity(file).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Imported checkpoint identity could not be read: {}",
            file.display()
        ))
    })?;
    let hash = sha256_file(api, settings, job_id, file).await?;
    let after = model_file_identity(file).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Imported checkpoint identity could not be re-read: {}",
            file.display()
        ))
    })?;
    if before != after {
        return Err(WorkerError::InvalidPayload(
            "The imported checkpoint changed while SceneWorks was hashing it; retry the import after the file is stable."
                .to_owned(),
        ));
    }
    Ok((hash, after))
}

/// The `(importSourceShape, importQuantFormat)` pair a single checkpoint file's safetensors header
/// proves, or `None` when it proves nothing a manifest entry may claim (sc-11045, epic 11037).
///
/// The managed import branch derives the same pair inline, because it can do more: it holds staged
/// bytes it is about to install, so it REFUSES an unsupported triple and a bare `mage-flow` file
/// outright. A linked import has already compiled a verified plan over the user's own file and must
/// not retroactively refuse it, so an unrecognized header stamps nothing here rather than failing
/// the job. What must not differ is the PAIR — two copies of one rule disagreeing about what a
/// checkpoint is stored as is the sc-13542 resolver-drift class, and
/// `linked_classification_tests::the_linked_classifier_matches_the_managed_branchs_mapping` pins
/// them together.
///
/// A component that is not a whole model (a text encoder, a VAE) yields `None`: it is not
/// registerable as an imported model at all, so it has no source shape to advertise.
pub(crate) fn base_weight_manifest_classification(
    path: &Path,
) -> Option<(&'static str, &'static str)> {
    let BaseWeightDetection::Recognized(verdict) = detect_base_weight_file(path).ok()? else {
        return None;
    };
    let source = match verdict.component {
        ComponentRole::Transformer => "transformer_file",
        ComponentRole::Checkpoint => "fused_checkpoint",
        ComponentRole::TextEncoder | ComponentRole::Vae => return None,
    };
    Some((source, verdict.quant.as_str()))
}

/// The manifest entry a compiled linked checkpoint becomes (epic 20398, sc-20635).
///
/// This is the stamp that makes a linked checkpoint REACHABLE. `importPlan.checkpointId` is the
/// persisted identity — `rootId + relativePath`, never an absolute path (E6) — and it is the single
/// discriminator every consumer keys on: the scheduler's imported claim
/// (`jobs_store::checkpoint_plan_checkpoint_id`), the worker's route selection
/// (`image_jobs::checkpoint_plan`), and every bespoke imported lane's decline. Without it a
/// compiled plan is a file on disk nothing can select.
///
/// A plan-backed entry carries NO `paths.model`. The plan-driven route resolves the plan's own
/// layers through the plan store; writing an install path here would both be a lie (nothing was
/// installed) and hand the entry to a bespoke lane as well, breaking the single-claim invariant.
///
/// # The header classification rides too (sc-11045, epic 11037)
///
/// `classification` is the `(importSourceShape, importQuantFormat)` pair
/// [`base_weight_manifest_classification`] proved from the linked file's own safetensors header —
/// the SAME detector the managed branch runs, over the same bytes.
///
/// It was managed-only, and that made a LINKED copy of an NVFP4 checkpoint invisible to every E8
/// surface even though the bytes are identical: `imported_entry_source_codec` answered `None`, so
/// admission's native claim was false, the catalog emitted no `runtimeQuantTiers` (the provider
/// surface early-returns without an `importSourceShape`), the asset receipt and the
/// `generation_metrics` row carried no source codec, and — worst — the worker's dispatch guard
/// `krea_imported_native_nvfp4` was false, so `imported_model_quant` would happily hand `Quant::Q4`
/// to packed E2M1 bytes. E8 is a property of the CHECKPOINT, not of who owns the copy, so the
/// classification is stamped in both ownership modes.
///
/// `None` stamps nothing. A header that proves no base-weight verdict is not a reason to invent
/// one, and an absent key is what every consumer already handles.
pub(crate) fn linked_checkpoint_manifest_entry(
    mut entry: JsonObject,
    compiled: &sceneworks_core::checkpoint_plan_store::CompiledCheckpointV1,
    classification: Option<(&str, &str)>,
) -> JsonObject {
    entry
        .entry("family")
        .or_insert_with(|| Value::String(compiled.plan.family.clone()));
    entry.insert(
        "importPlan".to_owned(),
        json!({
            "checkpointId": compiled.checkpoint_id,
            "planId": compiled.plan.plan_id,
            "semanticDigest": compiled.record.plan.semantic_digest,
        }),
    );
    if let Some((source_shape, quant_format)) = classification {
        entry.insert(
            "importSourceShape".to_owned(),
            Value::String(source_shape.to_owned()),
        );
        entry.insert(
            "importQuantFormat".to_owned(),
            Value::String(quant_format.to_owned()),
        );
    }
    // A linked checkpoint is referenced in place; nothing was copied into an app-owned install
    // directory, so any inherited install path would name a directory that does not exist.
    if let Some(paths) = entry.get_mut("paths").and_then(Value::as_object_mut) {
        paths.remove("model");
    }
    // sc-26680: a linked MiniMax-H3 DiT directory binds exactly as a managed one does. The caller
    // has already refused a plan that is not a complete assembly, so an `Err` here leaves the entry
    // without a binding — which every consumer reads as "not an imported H3 model".
    if compiled.plan.family == MINIMAX_H3_IMPORT_FAMILY {
        if let Ok(assembly) = MiniMaxH3ImportAssembly::from_declared_plan(&compiled.plan, &entry) {
            stamp_imported_minimax_h3_entry(&mut entry, &assembly);
        }
    }
    let model_type = entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("image")
        .to_owned();
    let family = entry
        .get("family")
        .and_then(Value::as_str)
        .map(str::to_owned);
    apply_model_manifest_defaults(&mut entry, &model_type, family.as_deref());
    entry
}

fn minimax_h3_source_family_error(payload: &JsonObject, detected: Option<&str>) -> Option<String> {
    for entry in [
        Some(payload),
        payload.get("manifestEntry").and_then(Value::as_object),
    ]
    .into_iter()
    .flatten()
    {
        if let Err(reason) =
            sceneworks_core::minimax_h3_import::validate_declared_source_family(entry, detected)
        {
            return Some(format!("[checkpoint-plan:family-mismatch] {reason}. Choose a checkpoint compatible with the declared family, or correct the import declaration."));
        }
    }
    None
}

fn validate_resident_h3_interfaces(
    settings: &Settings,
    entry: &JsonObject,
    plan: &sceneworks_core::checkpoint_import::ImportPlanV1,
) -> Result<(), String> {
    let config = sceneworks_core::minimax_h3_import::bound_config(&plan.context)?
        .ok_or("missing bound transformer configuration")?;
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(&settings.data_dir),
    );
    for row in entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let (Some(repo), Some(revision)) = (
            row.get("repo").and_then(Value::as_str),
            row.get("revision").and_then(Value::as_str),
        ) else {
            continue;
        };
        let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision)) else {
            continue;
        };
        // Independently queued missing/partial components remain unready, not a false architectural
        // contradiction. Complete resident components must already agree before publication.
        if sceneworks_core::minimax_h3_import::import_component_files_complete(&snapshot, row) {
            sceneworks_core::minimax_h3_import::import_component_configuration_compatible(
                &snapshot, row, &config.0,
            )?;
        }
    }
    Ok(())
}

async fn import_plan_context(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    source: &Path,
) -> WorkerResult<std::collections::BTreeMap<String, Value>> {
    let mut context = import_plan_context_configuration(api, settings, job, source).await?;
    let Some(entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .filter(|entry| {
            sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(entry)
        })
    else {
        return Ok(context);
    };
    let entry = entry.clone();
    let data_dir = settings.data_dir.clone();
    let cancel = gen_core::CancelFlag::new();
    let task_cancel = cancel.clone();
    let span = tracing::info_span!("h3_import_component_acquisition", jobId = %job.id);
    let task = crate::load_progress::spawn_blocking(move || {
        let _span = span.enter();
        let result =
            sceneworks_core::model_artifacts::provenance::capture_h3_component_closure_with_check(
                &data_dir,
                &entry,
                &|| {
                    if task_cancel.is_cancelled() {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "H3 import component acquisition canceled",
                        ))
                    } else {
                        Ok(())
                    }
                },
            );
        if task_cancel.is_cancelled() {
            return Err(WorkerError::Canceled(
                "Model import canceled while binding components.".into(),
            ));
        }
        result.map_err(WorkerError::InvalidPayload)
    });
    let closure = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        Some(cancel),
        "Model import canceled while binding components.",
        "H3 component closure",
        None::<fn() -> std::future::Ready<WorkerResult<()>>>,
        task,
    )
    .await?;
    context
        .get_mut(sceneworks_core::model_artifacts::provenance::MODEL_DECLARATION_KEY)
        .ok_or_else(|| WorkerError::InvalidPayload("missing import declaration".into()))?
        ["componentClosure"] = closure;
    Ok(context)
}

async fn import_plan_context_configuration(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    source: &Path,
) -> WorkerResult<std::collections::BTreeMap<String, Value>> {
    let entry = job.payload.get("manifestEntry").and_then(Value::as_object);
    let mut context = entry
        .map(sceneworks_core::model_artifacts::provenance::import_declaration_context)
        .transpose()
        .map_err(WorkerError::InvalidPayload)?
        .unwrap_or_default();
    if !entry.is_some_and(sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family) {
        return Ok(context);
    }
    // A source-owned sidecar/embedded claim is authoritative, including invalid claims: inspection
    // must refuse those rather than substitute a base. Never download a fallback in that case.
    let directory = if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or(source)
    };
    let has_embedded = |path: &Path| {
        sceneworks_core::lora_family::read_safetensors_header(path)
            .ok()
            .is_some_and(|h| {
                h.get("__metadata__")
                    .and_then(|m| m.get("h3_config"))
                    .is_some()
            })
    };
    let embedded = if source.is_dir() {
        std::fs::read_dir(source)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
                    && has_embedded(&entry.path())
            })
    } else {
        has_embedded(source)
    };
    if directory.join("config.json").exists() || embedded {
        return Ok(context);
    }
    let is_h3 = |path: &Path| matches!(detect_base_weight_file(path), Ok(BaseWeightDetection::Recognized(v)) if v.family.as_deref() == Some(MINIMAX_H3_IMPORT_FAMILY));
    let detected_h3 = if source.is_dir() {
        std::fs::read_dir(source)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|e| is_h3(&e.path()))
    } else {
        is_h3(source)
    };
    if !detected_h3 {
        return Ok(context);
    } // Full inspection reports the actual family incompatibility.
    let entry = entry.expect("checked");
    sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(entry)
        .map_err(WorkerError::InvalidPayload)?;
    let rows: Vec<_> = entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| {
            row.get("componentId").and_then(Value::as_str) == Some("import_transformer_config")
        })
        .collect();
    if rows.len() != 1 {
        return Err(WorkerError::InvalidPayload(
            "Weight-only H3 requires the selected base's exact configuration download".into(),
        ));
    }
    let row = rows[0];
    let repo = row
        .get("repo")
        .and_then(Value::as_str)
        .ok_or_else(|| WorkerError::InvalidPayload("Configuration repository missing".into()))?;
    let revision = row
        .get("revision")
        .and_then(Value::as_str)
        .ok_or_else(|| WorkerError::InvalidPayload("Configuration revision missing".into()))?;
    let files = row
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| WorkerError::InvalidPayload("Configuration file missing".into()))?;
    let path = files
        .first()
        .and_then(Value::as_str)
        .filter(|p| files.len() == 1 && (*p == "config.json" || p.ends_with("/config.json")))
        .ok_or_else(|| {
            WorkerError::InvalidPayload(
                "Configuration download must contain only config.json".into(),
            )
        })?;
    let parts = sceneworks_core::checkpoint_plan_store::portable_relative_path_parts(path)
        .map_err(|e| WorkerError::InvalidPayload(e.into()))?;
    let provenance = json!({"kind":"selected-base","baseModel":entry["baseModel"],"repo":repo,"revision":revision,"path":path,"componentTier":entry.get("importComponentTier")});
    // Validate immutable provenance before issuing any request; mutable main/branch is not a pin.
    context.insert(
        "transformerConfig".into(),
        sceneworks_core::minimax_h3_import::config_binding(json!({}), provenance.clone()),
    );
    sceneworks_core::minimax_h3_import::bound_config(&context)
        .map_err(WorkerError::InvalidPayload)?;
    let mut config_file = huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)
        .map(|root| root.join(&parts));
    if !config_file.as_ref().is_some_and(|p| p.is_file()) {
        ensure_hf_files_cached(api, settings, job, repo, revision, &[path.to_owned()]).await?;
        config_file = huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)
            .map(|root| root.join(&parts));
    }
    let file = config_file.filter(|p| p.is_file()).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Selected base configuration is unavailable at the exact revision".into(),
        )
    })?;
    let config = sceneworks_core::minimax_h3_codec::read_h3_descriptor(&file)
        .map_err(|e| WorkerError::InvalidPayload(format!("Invalid selected-base config: {e}")))?;
    context.insert(
        "transformerConfig".into(),
        sceneworks_core::minimax_h3_import::config_binding(config, provenance),
    );
    Ok(context)
}

/// Compile and register an approved linked source without copying or modifying its bytes.
/// A missing source config may fetch only the selected base's pinned config into the shared cache.
async fn run_linked_checkpoint_import_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    root_id: &str,
    relative_path: &str,
) -> WorkerResult<()> {
    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Importing,
            0.1,
            "Inspecting the linked checkpoint.",
            None,
            None,
            None,
        ),
    )
    .await?;
    let store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    let previous_plan_id = store
        .record(
            &sceneworks_core::checkpoint_plan_store::linked_checkpoint_id(root_id, relative_path),
        )
        .ok()
        .map(|record| record.plan.plan_id);
    // The absolute path of the linked file, resolved through the APPROVED root — the same
    // confinement `compile_linked` applies, never a path built from the payload. It is what the
    // header classification below reads; a root that no longer resolves simply yields no
    // classification (the compile below produces the typed refusal for that case).
    let linked_file = store
        .resolve_root(root_id)
        .ok()
        .zip(
            sceneworks_core::checkpoint_plan_store::portable_relative_path_parts(relative_path)
                .ok(),
        )
        .map(|(root, parts)| root.join(parts));
    let declares_h3 = [
        Some(&job.payload),
        job.payload.get("manifestEntry").and_then(Value::as_object),
    ]
    .into_iter()
    .flatten()
    .any(sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family);
    if declares_h3 {
        // Header-only, confined discovery precedes config/component work. This query never
        // publishes or invalidates a plan; an existing linked owner's state stays untouched.
        let probe_store = store.clone();
        let probe_root = root_id.to_owned();
        let probe_relative = relative_path.to_owned();
        let task = crate::load_progress::spawn_blocking(move || {
            use sceneworks_core::checkpoint_inspector::{
                discover_checkpoint, discovered_checkpoint_family, CheckpointInspectionRequestV1,
            };
            let root = probe_store
                .resolve_root(&probe_root)
                .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
            let checkpoint_id = sceneworks_core::checkpoint_plan_store::linked_checkpoint_id(
                &probe_root,
                &probe_relative,
            );
            let request = CheckpointInspectionRequestV1::linked(
                &checkpoint_id,
                root,
                &probe_relative,
                &probe_root,
            )
            .map_err(WorkerError::InvalidPayload)?;
            let discovery = discover_checkpoint(&request);
            discovered_checkpoint_family(&request, &discovery).map_err(|diagnostics| {
                WorkerError::InvalidPayload(
                    sceneworks_core::checkpoint_plan_store::CheckpointPlanError::UnrunnableSource {
                        checkpoint_id,
                        diagnostics,
                    }
                    .to_string(),
                )
            })
        });
        let detected = heartbeat_while_blocking(
            api,
            settings,
            &job.id,
            "linked source family discovery",
            task,
        )
        .await;
        match detected {
            Ok(Some(family)) => {
                if let Some(reason) = minimax_h3_source_family_error(&job.payload, Some(&family)) {
                    return fail_job(
                        api,
                        &job.id,
                        "Linked checkpoint import failed.",
                        Some(reason),
                    )
                    .await;
                }
            }
            Ok(None) => {} // Unknown/ambiguous evidence is still the full compiler's decision.
            Err(error) => {
                return fail_job(
                    api,
                    &job.id,
                    "Linked checkpoint import failed.",
                    Some(error.to_string()),
                )
                .await
            }
        }
    }
    let context = match import_plan_context(
        api,
        settings,
        job,
        linked_file.as_deref().unwrap_or(Path::new("")),
    )
    .await
    {
        Ok(context) => context,
        Err(error @ WorkerError::Canceled(_)) => return Err(error),
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Linked checkpoint import failed.",
                Some(error.to_string()),
            )
            .await;
        }
    };
    let root_id = root_id.to_owned();
    let relative_path = relative_path.to_owned();
    // Full-content compile streams every byte through SHA-256, so it runs off the async runtime and
    // under the job's heartbeat: a multi-GB source outlasts the 90s stale-worker sweep.
    let compile = crate::load_progress::spawn_blocking(move || {
        Ok::<_, WorkerError>(store.compile_linked_with_context(&root_id, &relative_path, context))
    });
    let compiled = match heartbeat_while_blocking(
        api,
        settings,
        &job.id,
        "linked checkpoint compile",
        compile,
    )
    .await
    {
        Ok(Ok(compiled)) => compiled,
        // Every refusal is already a typed `[checkpoint-plan:<code>]` diagnostic naming what is
        // wrong with the source; surface it verbatim rather than flattening it to "import failed".
        Ok(Err(error)) => {
            return fail_job(
                api,
                &job.id,
                "Linked checkpoint import failed.",
                Some(error.to_string()),
            )
            .await;
        }
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Linked checkpoint import failed.",
                Some(format!("Checkpoint inspection task failed: {error}")),
            )
            .await;
        }
    };
    if let Some(reason) = minimax_h3_source_family_error(&job.payload, Some(&compiled.plan.family))
    {
        CheckpointPlanStore::open(&settings.data_dir)
            .invalidate(&compiled.checkpoint_id)
            .map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "{reason}; could not invalidate refused linked plan: {error}"
                ))
            })?;
        return fail_job(
            api,
            &job.id,
            "Linked checkpoint import failed.",
            Some(reason),
        )
        .await;
    }

    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        // The SAME header classification the managed branch runs, over the user's own file. Reading
        // a safetensors header is a bounded seek-and-parse, but it is still blocking I/O.
        let classification = match linked_file {
            Some(path) => crate::load_progress::spawn_blocking(move || {
                base_weight_manifest_classification(&path)
            })
            .await
            .unwrap_or(None),
            None => None,
        };
        // sc-26680: an H3 DiT that is not a complete assembly is refused rather than registered.
        if compiled.plan.family == MINIMAX_H3_IMPORT_FAMILY {
            if let Err(reason) =
                MiniMaxH3ImportAssembly::from_declared_plan(&compiled.plan, &manifest_entry)
                    .and_then(|assembly| {
                        validate_resident_h3_interfaces(settings, &manifest_entry, &compiled.plan)?;
                        Ok(assembly)
                    })
            {
                // A second catalog name may reference this same approved source. A refused new
                // declaration must not destroy the unchanged binding an existing model owns.
                if previous_plan_id.as_deref() != Some(compiled.plan.plan_id.as_str()) {
                    CheckpointPlanStore::open(&settings.data_dir)
                        .invalidate(&compiled.checkpoint_id)
                        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
                }
                return fail_job(
                    api,
                    &job.id,
                    "Linked checkpoint import failed.",
                    Some(format!(
                        "[checkpoint-plan:ambiguous-component] the linked MiniMax-H3 checkpoint \
                         is not a complete DiT assembly: {reason}"
                    )),
                )
                .await;
            }
        }
        let manifest_entry =
            linked_checkpoint_manifest_entry(manifest_entry, &compiled, classification);
        let manifest_path = model_manifest_target(settings, &job.payload)?;
        check_cancel(api, &job.id, "Linked import canceled before registration.").await?;
        upsert_manifest_entry(&manifest_path, "models", manifest_entry).await?;
    }

    let mut result = JsonObject::new();
    result.insert(
        "modelId".to_owned(),
        job.payload.get("modelId").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "checkpointId".to_owned(),
        Value::String(compiled.checkpoint_id.clone()),
    );
    result.insert(
        "importPlanId".to_owned(),
        Value::String(compiled.plan.plan_id.clone()),
    );
    result.insert(
        "family".to_owned(),
        Value::String(compiled.plan.family.clone()),
    );
    result.insert(
        "duplicateCheckpointIds".to_owned(),
        json!(compiled.duplicate_checkpoint_ids),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Linked checkpoint registered.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

/// The managed install id for a resolved model-import target directory (sc-20636).
///
/// The install id is the target directory's own name, so a managed install occupies exactly the
/// path imported models have always occupied (`<data>/models/imports/<name>`) and every existing
/// consumer of `paths.model` keeps working. The store's own [`CheckpointPlanStore::install_dir`] is
/// then required to reproduce that same path, which is what proves the id addresses the intended
/// directory rather than merely looking plausible.
fn managed_install_id_for_target(
    store: &CheckpointPlanStore,
    target_dir: &Path,
) -> WorkerResult<String> {
    let install_id = target_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "Model import target {} does not name an install directory",
                target_dir.display()
            ))
        })?
        .to_owned();
    // Normalized the same way `resolve_model_import_target` normalized `target_dir`: on macOS a
    // data dir under `/var` canonicalizes to `/private/var`, so a lexical comparison would reject
    // the very directory the store derived.
    let derived = store
        .install_dir(&install_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let derived = crate::paths::normalize_existing_or_absolute(&derived)?;
    if derived != target_dir {
        return Err(WorkerError::InvalidPayload(format!(
            "Model import target {} is not the managed install directory for {install_id:?} ({})",
            target_dir.display(),
            derived.display()
        )));
    }
    Ok(install_id)
}

/// The provenance a managed install records, taken from the `importProvenance` the API normalised
/// from either the discriminated source or the legacy flat fields.
///
/// A payload with no provenance is a job queued by an older API build; it records the ingest source
/// as `legacy-import` rather than guessing, so the field is never silently wrong.
fn managed_provenance_from_payload(payload: &JsonObject) -> WorkerResult<ManagedProvenanceV1> {
    let Some(object) = payload.get("importProvenance").and_then(Value::as_object) else {
        return Ok(ManagedProvenanceV1::of_source("legacy-import"));
    };
    let field = |key: &str| crate::payload::trimmed_string(object.get(key)).map(str::to_owned);
    let provenance = ManagedProvenanceV1 {
        source: field("source").unwrap_or_else(|| "legacy-import".to_owned()),
        reference: field("reference"),
        url: field("url"),
        version_id: field("versionId"),
        file_id: field("fileId"),
        credential_host: field("credentialHost"),
        import_job_id: None,
        import_commitment: None,
    };
    provenance
        .validate()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    Ok(provenance)
}

/// A staged artifact's path relative to the staging root, in the portable `/`-separated spelling
/// the plan contract uses.
fn managed_staged_relative_path(staging_dir: &Path, file: &Path) -> WorkerResult<String> {
    let relative = file.strip_prefix(staging_dir).map_err(|_| {
        WorkerError::InvalidPayload(format!(
            "Imported checkpoint {} is not inside the import staging directory {}",
            file.display(),
            staging_dir.display()
        ))
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => parts.push(
                part.to_str()
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "Imported checkpoint path {} is not valid UTF-8",
                            relative.display()
                        ))
                    })?
                    .to_owned(),
            ),
            _ => {
                return Err(WorkerError::InvalidPayload(format!(
                    "Imported checkpoint path {} is not a confined relative path",
                    relative.display()
                )))
            }
        }
    }
    if parts.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Imported checkpoint path is empty".to_owned(),
        ));
    }
    Ok(parts.join("/"))
}

/// Reusable claim fence: an old authenticated snapshot cannot authorize a later catalog mutation.
pub(crate) fn validate_import_publication_claim(
    worker_id: &str,
    expected: &JobSnapshot,
    current: &JobSnapshot,
) -> WorkerResult<()> {
    if current.cancel_requested {
        return Err(WorkerError::Canceled(
            "Model import canceled before publication.".into(),
        ));
    }
    if current.id != expected.id
        || current.job_type != JobType::ModelImport
        || expected.job_type != JobType::ModelImport
        || current.worker_id.as_deref() != Some(worker_id)
        || !sceneworks_core::jobs_store::ACTIVE_STATUSES.contains(&current.status.as_str())
        || current.attempts != expected.attempts
        || current.project_id != expected.project_id
        || current.source_job_id != expected.source_job_id
        || crate::imports::model_import_commitment(&current.payload)
            != crate::imports::model_import_commitment(&expected.payload)
    {
        return Err(WorkerError::InvalidPayload(
            "Import claim changed or was interrupted before publication; no selection was modified"
                .into(),
        ));
    }
    Ok(())
}

async fn publish_claimed_model_import_manifest(
    api: &ApiClient,
    job: &JobSnapshot,
    path: &Path,
    data_dir: &Path,
    entry: JsonObject,
    expected: Option<&Value>,
) -> WorkerResult<()> {
    let api = api.clone();
    let job = job.clone();
    let runtime = tokio::runtime::Handle::current();
    crate::manifest::publish_model_import_manifest(path, data_dir, entry, expected, move || {
        // Called on the publication blocking thread INSIDE the acquired manifest lock, not
        // before the wait. The async API client remains on its owning runtime; no new listener.
        runtime.block_on(async {
            let current: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{}", job.id)).await?;
            validate_import_publication_claim(&api.worker_id, &job, &current)
        })
    })
    .await
}

/// Resume only an unchanged import's authenticated retry lineage. Neither caller-supplied
/// source_job_id nor destination equality is authority to adopt another transaction's bytes.
async fn authenticate_import_recovery(
    api: &ApiClient,
    job: &JobSnapshot,
    marker: &Value,
) -> WorkerResult<bool> {
    let Some(owner) = marker.get("jobId").and_then(Value::as_str) else {
        return Ok(false);
    };
    let commitment = crate::imports::model_import_commitment(&job.payload);
    if marker.get("importCommitment").and_then(Value::as_str) != Some(commitment.as_str()) {
        return Err(WorkerError::InvalidPayload(
            "Committed import request changed; retry cannot adopt this installation".into(),
        ));
    }
    // Even the original ID must still be owned by this worker. A stale snapshot is not recovery
    // authority; read ancestry from the authenticated control plane, never a mutable payload.
    let mut current: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{}", job.id)).await?;
    if current.id != job.id
        || current.worker_id.as_deref() != Some(api.worker_id.as_str())
        || matches!(
            current.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Canceled | JobStatus::Interrupted
        )
    {
        return Ok(false);
    }
    let mut seen = std::collections::HashSet::new();
    for _ in 0..64 {
        if !seen.insert(current.id.clone())
            || current.job_type != JobType::ModelImport
            || current.project_id != job.project_id
            || crate::imports::model_import_commitment(&current.payload) != commitment
        {
            return Ok(false);
        }
        if current.id == owner {
            return Ok(true);
        }
        let Some(parent) = current.source_job_id.as_deref() else {
            return Ok(false);
        };
        // IDs are server-generated, but do not interpolate an arbitrary persisted path.
        if !sceneworks_core::ids::is_safe_id(parent) {
            return Ok(false);
        }
        let previous: JobSnapshot = api.get_json(&format!("/api/v1/jobs/{parent}")).await?;
        if previous.attempts.checked_add(1) != Some(current.attempts) {
            return Ok(false);
        }
        current = previous;
    }
    Ok(false)
}

/// Resume the commit→manifest publication boundary without overwriting unrelated installations.
async fn recover_committed_h3_import(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    target: &Path,
) -> WorkerResult<bool> {
    let marker_path = target.join(".sceneworks-download-complete.json");
    let marker = match tokio::fs::read(&marker_path).await {
        Ok(bytes) => serde_json::from_slice::<Value>(&bytes).ok(),
        Err(_) => None,
    };
    let Some(marker) = marker else {
        return Ok(false);
    };
    if !authenticate_import_recovery(api, job, &marker).await? {
        return Ok(false);
    }
    if marker.get("modelId") != job.payload.get("modelId") {
        return Err(WorkerError::InvalidPayload(
            "Recovered import marker names a different model".into(),
        ));
    }
    let store = CheckpointPlanStore::open(&settings.data_dir);
    let install_id = managed_install_id_for_target(&store, target)?;
    let checkpoint_id = sceneworks_core::checkpoint_plan_store::managed_checkpoint_id(&install_id);
    let resolved = store
        .resolve(&checkpoint_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let owner = marker.get("jobId").and_then(Value::as_str);
    let commitment = marker.get("importCommitment").and_then(Value::as_str);
    if resolved.plan.layers.iter().any(|layer| !matches!(&layer.source,
        sceneworks_core::checkpoint_import::SourceLocatorV1::Managed { install_id: owner_install, provenance, .. }
        if owner_install == &install_id && provenance.import_job_id.as_deref() == owner
            && provenance.import_commitment.as_deref() == commitment)) {
        return Err(WorkerError::InvalidPayload("Committed import marker does not match immutable installation ownership; explicitly reimport".into()));
    }
    if resolved.plan.family != MINIMAX_H3_IMPORT_FAMILY {
        return Ok(false);
    }
    if let Some(reason) = minimax_h3_source_family_error(&job.payload, Some(&resolved.plan.family))
    {
        return Err(WorkerError::InvalidPayload(reason));
    }
    let mut entry = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Recovered H3 import has no manifestEntry".into())
        })?;
    let assembly = MiniMaxH3ImportAssembly::from_declared_plan(&resolved.plan, &entry)
        .and_then(|assembly| {
            validate_resident_h3_interfaces(settings, &entry, &resolved.plan)?;
            Ok(assembly)
        })
        .map_err(WorkerError::InvalidPayload)?;
    let primary =
        sceneworks_core::base_weights::minimax_h3_import_primary(target).ok_or_else(|| {
            WorkerError::InvalidPayload("Recovered H3 import has no primary weights".into())
        })?;
    let (digest, _) = stable_model_file_sha256(api, settings, &job.id, &primary).await?;
    if optional_payload_string(&job.payload, "expectedSha256")
        .and_then(normalize_sha256)
        .is_some_and(|expected| expected != digest)
    {
        return Err(WorkerError::InvalidPayload(
            "Recovered H3 import differs from expectedSha256".into(),
        ));
    }
    let semantic = resolved
        .plan
        .semantic_digest()
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    let duplicates = store
        .duplicates_of_plan(&resolved.plan, &checkpoint_id)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    entry.insert("importPlan".into(), json!({"checkpointId":checkpoint_id,"planId":resolved.plan.plan_id,"semanticDigest":semantic}));
    let classification = base_weight_manifest_classification(&primary);
    if let Some((_, quant)) = classification {
        entry.insert("importQuantFormat".into(), json!(quant));
    }
    stamp_imported_minimax_h3_entry(&mut entry, &assembly);
    entry
        .entry("paths")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            WorkerError::InvalidPayload("Recovered import paths must be an object".into())
        })?
        .insert("model".into(), json!(target.display().to_string()));
    apply_model_manifest_defaults(&mut entry, "video", Some(MINIMAX_H3_IMPORT_FAMILY));
    check_cancel(
        api,
        &job.id,
        "Model import canceled before recovered publication.",
    )
    .await?;
    let manifest = model_manifest_target(settings, &job.payload)?;
    sceneworks_core::model_artifacts::provenance::validate_entry_declaration(
        &resolved.plan,
        &entry,
    )
    .map_err(WorkerError::InvalidPayload)?;
    publish_claimed_model_import_manifest(
        api,
        job,
        &manifest,
        &settings.data_dir,
        entry,
        job.payload.get("repairExpectedBinding"),
    )
    .await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Model import recovered.",
            None,
            Some(
                json!({"modelId":job.payload.get("modelId"),"path":target.display().to_string(),
            "family":MINIMAX_H3_IMPORT_FAMILY,"checkpointId":checkpoint_id,
            "importPlanId":resolved.plan.plan_id,"duplicateCheckpointIds":duplicates,
            "completedAt":now_rfc3339()})
                .as_object()
                .expect("result")
                .clone(),
            ),
            None,
        ),
    )
    .await?;
    let _ = cleanup_uploaded_import_source(settings, &job.payload).await;
    Ok(true)
}

pub(crate) async fn run_model_import_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    // Linked (in-place) ownership is its own transaction: no transfer, no install directory, no
    // `paths.model` (sc-20635). Dispatched before anything below so the download-oriented path
    // cannot reach a plan-backed entry at all.
    if let Some(root_id) = optional_payload_string(&job.payload, "linkedRootId") {
        let Some(relative_path) = optional_payload_string(&job.payload, "linkedRelativePath")
        else {
            return fail_job(
                api,
                &job.id,
                "Linked checkpoint import failed.",
                Some(
                    "A linked checkpoint import requires both linkedRootId and \
                     linkedRelativePath."
                        .to_owned(),
                ),
            )
            .await;
        };
        let (root_id, relative_path) = (root_id.to_owned(), relative_path.to_owned());
        return run_linked_checkpoint_import_job(api, settings, job, &root_id, &relative_path)
            .await;
    }
    let repo = optional_payload_string(&job.payload, "repo")
        .map(validate_hf_repo_id)
        .transpose()?;
    let source_url = optional_payload_string(&job.payload, "sourceUrl");
    let source_path = optional_payload_string(&job.payload, "sourcePath");
    let target_name = optional_payload_string(&job.payload, "modelId")
        .map(safe_download_dir)
        .unwrap_or_else(|| "model".to_owned());
    let target_dir = resolve_model_import_target(
        settings,
        &job.payload,
        settings
            .data_dir
            .join("models")
            .join("imports")
            .join(&target_name),
    )?;

    // sc-20636 (epic 20398): every managed source — HF repo, source URL, Civitai, upload, local
    // copy — transfers into a STAGING directory and reaches `target_dir` only through one atomic
    // rename in `ManagedIngest::finalize`, after its bytes have been hashed and fully validated. A
    // cancel, a crash, a full disk, or a hash mismatch therefore all end with no directory at
    // `target_dir` at all, so no partially-transferred model is ever addressable, selectable, or
    // written to the manifest. The transfer code below is unchanged and simply points at staging.
    let plan_store = CheckpointPlanStore::open(&settings.data_dir);
    let install_id = managed_install_id_for_target(&plan_store, &target_dir)?;
    if target_dir.exists() && recover_committed_h3_import(api, settings, job, &target_dir).await? {
        return Ok(());
    }
    let mut provenance = managed_provenance_from_payload(&job.payload)?;
    provenance.import_job_id = Some(job.id.clone());
    provenance.import_commitment = Some(crate::imports::model_import_commitment(&job.payload));
    let ingest = match ManagedIngest::begin(&plan_store, &install_id, provenance) {
        Ok(ingest) => ingest,
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(error.to_string()),
            )
            .await
        }
    };
    let staging_dir = ingest.staging_dir().to_path_buf();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Importing,
            0.1,
            "Importing model.",
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Model import canceled before transfer started.",
    )
    .await?;

    if let Some(repo) = repo {
        let files = payload_string_array(&job.payload, "files");
        let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
        // Trust-boundary validation for the LAN-exposed jobs API (epic 4484): a model HF import takes a
        // caller-supplied repo/revision/files and resolves them into download URLs and on-disk paths,
        // so reject anything that could escape the intended Hugging Face path (sc-13583).
        let repo = validate_hf_repo_id(repo)?;
        validate_hf_download_inputs(repo, revision, &files)?;
        let snapshot =
            HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
        let mut progress = DownloadProgress::new(
            repo,
            0,
            snapshot.total_bytes(),
            progress_report_interval(settings),
        );
        download_snapshot(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "Model import canceled by user.",
                fresh_download: false,
            },
            &staging_dir,
            &snapshot,
            &mut progress,
        )
        .await?;
    } else if let Some(source_path) = source_path {
        // sc-8803: confine the client-supplied model import source the same way as
        // LoRA imports (staged model-upload cache for uploads, app-managed roots
        // otherwise) before the copy/move reaches the filesystem.
        let source_path = resolve_import_source_path(
            settings,
            &job.payload,
            source_path,
            "model-uploads",
            "Model import sourcePath",
        )?;
        // The staging copy is checkpoint-sized and reports no progress of its own.
        with_job_heartbeat(api, settings, &job.id, async {
            if payload_bool(&job.payload, "uploadedSourcePath") && source_path.is_file() {
                // Keep the upload as the retry/restart input until publication succeeds. A hard
                // link avoids a second checkpoint-sized copy on the normal same-volume path.
                let name = source_path.file_name().ok_or_else(|| {
                    WorkerError::InvalidPayload("Uploaded model has no filename".into())
                })?;
                if tokio::fs::hard_link(&source_path, staging_dir.join(name))
                    .await
                    .is_err()
                {
                    copy_lora_source(&source_path, &staging_dir).await?;
                }
            } else {
                copy_lora_source(&source_path, &staging_dir).await?;
            }
            Ok(())
        })
        .await?;
    } else if let Some(source_url) = source_url {
        download_model_source_url(
            &DownloadContext {
                api,
                client: http_client,
                settings,
                job_id: &job.id,
                cancel_message: "Model import canceled by user.",
                fresh_download: false,
            },
            source_url,
            &staging_dir,
        )
        .await?;
    } else {
        return fail_job(
            api,
            &job.id,
            "Model import failed.",
            Some("Provide repo, sourceUrl, or sourcePath for model import".to_owned()),
        )
        .await;
    }

    let imported_model_file = imported_model_primary_weight_file(&staging_dir)
        .or_else(|| sceneworks_core::base_weights::minimax_h3_import_primary(&staging_dir));
    let mut model_file_sha256 = None;
    let mut verified_model_file_identity = None;

    // sc-6137: verify a caller-supplied content digest for source-URL/uploaded model
    // imports (HF repo files are per-file verified during download). On mismatch the
    // file is removed and the job fails with an actionable message.
    if let Some(expected_sha256) =
        optional_payload_string(&job.payload, "expectedSha256").and_then(normalize_sha256)
    {
        if let Some(file) = imported_model_file.as_ref() {
            match stable_model_file_sha256(api, settings, &job.id, file).await {
                Ok((hash, identity)) if hash == expected_sha256 => {
                    model_file_sha256 = Some(hash);
                    verified_model_file_identity = Some(identity);
                }
                Ok((hash, _)) => {
                    let _ = tokio::fs::remove_file(file).await;
                    return fail_job(
                        api,
                        &job.id,
                        "Model import failed.",
                        Some(format!(
                            "Imported model failed its integrity check (sha256 {hash}, but the source declares {expected_sha256}); the download was corrupted. Re-download the file."
                        )),
                    )
                    .await;
                }
                Err(error) => {
                    return fail_job(
                        api,
                        &job.id,
                        "Model import failed.",
                        Some(error.to_string()),
                    )
                    .await;
                }
            }
        }
    }

    // sc-14019 (epic 14015): compatibility gate behind the now-enabled model-import kill-switch.
    // Classify the downloaded base weights with the base-weight detector and refuse the import —
    // with the detector's typed reason — unless the (family, component, quant) triple has a real
    // loader today (`import_detection_supported`). Runs over the bytes on disk, so it covers HF-repo,
    // source-URL, and uploaded imports uniformly (the API's synchronous `import_source_supported`
    // only sees on-disk uploads at queue time). NEVER a silent fallback. `target_dir` is
    // app-managed (`resolve_model_import_target`) — this gate is purely additive to confinement.
    let (base_weight_family, import_source_shape, import_quant_format) = match imported_model_file
        .as_ref()
    {
        Some(weight_file) => match sceneworks_core::base_weights::detect_import_source_weights(
            &staging_dir,
            weight_file,
        ) {
            Ok(detection) => {
                if let Err(reason) = import_detection_supported(&detection) {
                    return fail_job(
                        api,
                        &job.id,
                        "Model import is not supported for this file.",
                        Some(reason),
                    )
                    .await;
                }
                if matches!(
                    &detection,
                    BaseWeightDetection::Recognized(verdict)
                        if verdict.family.as_deref() == Some("mage-flow")
                ) && !sceneworks_core::base_weights::is_mage_flow_transformer_dir(&staging_dir)
                {
                    return fail_job(
                        api,
                        &job.id,
                        "Model import is not supported for this file.",
                        Some(
                            "Model import for the 'mage-flow' family requires a complete \
                             transformer directory containing config.json and \
                             diffusion_pytorch_model.safetensors; a bare weights file is refused \
                             because the loader cannot derive its architecture."
                                .to_owned(),
                        ),
                    )
                    .await;
                }
                // H3 source normalization owns native single files and complete indexed shards.
                if matches!(
                    &detection,
                    BaseWeightDetection::Recognized(verdict)
                        if verdict.family.as_deref() == Some(MINIMAX_H3_IMPORT_FAMILY)
                ) && !(sceneworks_core::base_weights::is_minimax_h3_import_source(&staging_dir)
                    || sceneworks_core::base_weights::is_minimax_h3_single_file(weight_file))
                {
                    return fail_job(
                        api,
                        &job.id,
                        "Model import is not supported for this file.",
                        Some(
                            "MiniMax-H3 import requires a dense native/ComfyUI single-file or a complete \
                             dense/sharded MiniMaxH3Transformer3DModel component directory. Unknown \
                             quantized dialects and unplanned weight files are refused."
                                .to_owned(),
                        ),
                    )
                    .await;
                }
                // sc-14108: a single-file base checkpoint is a bare DiT — neither a diffusers
                // directory nor a LoRA — so the LoRA-oriented `detect_model_family` below returns
                // None for it. Reuse the family the gate's base-weight verdict already resolved so
                // the import STAMPS it onto the manifest entry. Without a family the entry persists
                // "unassociated": the catalog shows "Needs Family" and, because Mac routing keys off
                // the family (sc-14019 `MLX_ROUTED_FAMILIES`), "Not On Mac" — so the model can't be
                // selected in the Image Studio.
                match detection {
                    BaseWeightDetection::Recognized(verdict) => {
                        let source = match verdict.component {
                            ComponentRole::Transformer
                                if verdict.family.as_deref() == Some("mage-flow")
                                    && sceneworks_core::base_weights::is_mage_flow_transformer_dir(
                                        &staging_dir,
                                    ) =>
                            {
                                "transformer_directory"
                            }
                            ComponentRole::Transformer
                                if verdict.family.as_deref() == Some(MINIMAX_H3_IMPORT_FAMILY) =>
                            {
                                // Proven a complete H3 component directory by the gate above.
                                MINIMAX_H3_IMPORT_SOURCE_SHAPE
                            }
                            ComponentRole::Transformer => "transformer_file",
                            ComponentRole::Checkpoint => "fused_checkpoint",
                            ComponentRole::TextEncoder | ComponentRole::Vae => {
                                return fail_job(
                                    api,
                                    &job.id,
                                    "Model import is not supported for this file.",
                                    Some(format!(
                                        "A {:?} component cannot be registered as a complete imported model.",
                                        verdict.component
                                    )),
                                )
                                .await;
                            }
                        };
                        (verdict.family, Some(source), Some(verdict.quant.as_str()))
                    }
                    BaseWeightDetection::Unrecognized { .. } => (None, None, None),
                }
            }
            Err(error) => {
                return fail_job(
                    api,
                    &job.id,
                    "Model import failed.",
                    Some(model_family_detection_error(error)),
                )
                .await;
            }
        },
        None => {
            return fail_job(
                api,
                &job.id,
                "Model import is not supported for this file.",
                Some(
                    "No safetensors base-weight file was found in the imported model; single-file \
                     base-checkpoint import expects a .safetensors transformer."
                        .to_owned(),
                ),
            )
            .await;
        }
    };

    let detected_family = match detect_model_family(&staging_dir) {
        // A diffusers dir / LoRA is classified here; a bare base DiT is not, so fall back to the
        // base-weight verdict's family (sc-14108).
        Ok(detected) => detected.or(base_weight_family),
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(model_family_detection_error(error)),
            )
            .await;
        }
    };
    if let Some(reason) = minimax_h3_source_family_error(&job.payload, detected_family.as_deref()) {
        return fail_job(api, &job.id, "Model import failed.", Some(reason)).await;
    }
    let supplied_family = optional_payload_string(&job.payload, "family").map(str::to_owned);
    let resolved_family = match reconcile_detected_family(supplied_family, detected_family) {
        Ok(family) => family,
        Err(mismatch) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(format!(
                    "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
                    mismatch.detected, mismatch.supplied, mismatch.detected
                )),
            )
            .await;
        }
    };

    // Attribution is content-derived and retained with the import so generation does not reread a
    // multi-gigabyte checkpoint on every image job. When integrity verification already streamed
    // the file, reuse that exact digest; otherwise hash it once now.
    if model_file_sha256.is_none() {
        if let Some(file) = imported_model_file.as_ref() {
            let (hash, identity) = stable_model_file_sha256(api, settings, &job.id, file).await?;
            model_file_sha256 = Some(hash);
            verified_model_file_identity = Some(identity);
        }
    }
    // Written into STAGING, so the tree that commits is already complete. The marker is a hidden
    // dot-file, which the inspector skips, so it is not part of the compiled plan.
    write_model_install_marker(
        &staging_dir,
        &job.payload,
        repo.unwrap_or(""),
        &job.id,
        verified_model_file_identity.as_ref(),
        model_file_sha256.as_deref(),
    )
    .await?;

    // ---- commit ---------------------------------------------------------------------------
    // Everything above ran against staged bytes. This is the point of no return: one rename, then
    // the plan/record/bindings publication. A refusal here still leaves nothing at `target_dir`.
    let primary_relative = match imported_model_file.as_ref() {
        Some(file) => managed_staged_relative_path(&staging_dir, file)?,
        None => {
            return fail_job(
                api,
                &job.id,
                "Model import is not supported for this file.",
                Some("No safetensors base-weight file was found in the imported model.".to_owned()),
            )
            .await
        }
    };
    // The DECLARED digest, not the one just computed from these bytes. Handing `finalize` the
    // file's own hash would make its integrity check compare the bytes to themselves and pass
    // unconditionally — a disabled guard that still looks like a guard. The block above verifies
    // and fails first with a more actionable message; this is the transactional line of defence
    // behind it, so a caller that reaches `finalize` by any other route is still checked.
    let declared_sha256 =
        optional_payload_string(&job.payload, "expectedSha256").and_then(normalize_sha256);
    let context = match import_plan_context(
        api,
        settings,
        job,
        imported_model_file.as_deref().unwrap_or(&staging_dir),
    )
    .await
    {
        Ok(context) => context,
        Err(error @ WorkerError::Canceled(_)) => return Err(error),
        Err(error) => {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(error.to_string()),
            )
            .await;
        }
    };
    // Finalize compiles the plan and streams every staged byte through SHA-256. Run inline it held
    // a runtime thread for minutes with no heartbeat (a ~30 GB MiniMax-H3 checkpoint), so the
    // stale-worker sweep marked the healthy import interrupted; any spawned heartbeat task queued on
    // that thread was stranded with it. Off the async runtime, under the job's heartbeat.
    let finalize = crate::load_progress::spawn_blocking(move || {
        Ok::<_, WorkerError>(ingest.finalize_with_context(
            &primary_relative,
            declared_sha256.as_deref(),
            context,
        ))
    });
    let install =
        match heartbeat_while_blocking(api, settings, &job.id, "managed import finalize", finalize)
            .await?
        {
            Ok(install) => install,
            Err(error) => {
                return fail_job(
                    api,
                    &job.id,
                    "Model import failed.",
                    Some(error.to_string()),
                )
                .await
            }
        };
    // The commit landed where the job was told to install. `managed_install_id_for_target` already
    // proved the two agree; re-checked here against the normalized form because everything below
    // records `target_dir` as the entry's installed path.
    debug_assert_eq!(
        crate::paths::normalize_existing_or_absolute(&install.install_path).ok(),
        Some(target_dir.clone())
    );
    if let Some(reason) =
        minimax_h3_source_family_error(&job.payload, Some(&install.compiled.plan.family))
    {
        plan_store
            .remove_managed(&install.install_id)
            .map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "{reason}; could not roll back refused managed import: {error}"
                ))
            })?;
        return fail_job(api, &job.id, "Model import failed.", Some(reason)).await;
    }
    // A CURATED managed NVFP4 variant confirms against its registration (sc-11045, epic 11037).
    //
    // `ManagedCheckpointVariantV1::confirm_installed` compares the committed install id, checkpoint
    // id, path, digest and — the reason it exists — the registered `size_bytes` against the file on
    // disk. It had no production caller at all, which made the registered size the decorative
    // number its own doc comment says it must not be: it is what a client renders as "how much will
    // this download", and nothing compared it to anything.
    //
    // Post-commit rather than at finalize: this is the one verb that checks a COMMITTED install
    // against the registration as a whole, and the digest verification inside `finalize` has
    // already refused substituted bytes by the time it runs. A failure here is a registration
    // defect (a stale or invented size), so it fails the job with the diagnostic rather than
    // recording a curated identity the registry does not actually describe. Non-variant imports —
    // every ordinary user import — have no registration and skip it entirely.
    if let Some(variant) =
        sceneworks_core::managed_checkpoint_variants::managed_nvfp4_variant(&install.install_id)
    {
        if let Err(error) = variant.confirm_installed(&install) {
            return fail_job(
                api,
                &job.id,
                "Model import failed.",
                Some(format!(
                    "The installed bytes do not match the managed variant's registration: {error}"
                )),
            )
            .await;
        }
    }
    if !install.duplicate_checkpoint_ids().is_empty() {
        // Reported, never acted on: a user may legitimately keep both a linked library copy and a
        // managed one, so neither is deleted and the import still succeeds (AC2).
        tracing::info!(
            checkpoint_id = %install.checkpoint_id,
            duplicates = %install.duplicate_checkpoint_ids().join(", "),
            "imported checkpoint duplicates checkpoints already known to this install"
        );
    }

    if let Some(manifest_entry) = job
        .payload
        .get("manifestEntry")
        .and_then(Value::as_object)
        .cloned()
    {
        let mut manifest_entry = manifest_entry;
        if let Some(family) = resolved_family.clone() {
            manifest_entry
                .entry("family")
                .or_insert(Value::String(family));
        }
        if let Some(source) = import_source_shape {
            manifest_entry.insert(
                "importSourceShape".to_owned(),
                Value::String(source.to_owned()),
            );
            // The plan binding (epic 20398): a managed install is selectable through the
            // plan-driven route by the checkpoint identity its published plan carries. Stamped
            // beside `importSourceShape` because the scheduler's imported claim requires BOTH — a
            // source shape and a source hint — and only an entry the base-weight gate classified
            // has a shape. `paths.model` is written too: the plan-driven route serves plain
            // text-to-image only, and the family's bespoke lane keeps every shape it does not
            // serve, so importing through the plan never removes a capability the entry had.
            manifest_entry.insert(
                "importPlan".to_owned(),
                json!({ "checkpointId": install.checkpoint_id }),
            );
        }
        if let Some(quant) = import_quant_format {
            manifest_entry.insert(
                "importQuantFormat".to_owned(),
                Value::String(quant.to_owned()),
            );
        }
        // sc-26680: a MiniMax-H3 DiT is a component of an ASSEMBLY, not a whole model. Bind it —
        // validated against the plan the commit just published — before the entry can exist, so
        // an incomplete or ambiguous assembly never becomes a selectable catalog row.
        if install.compiled.plan.family == MINIMAX_H3_IMPORT_FAMILY {
            match MiniMaxH3ImportAssembly::from_declared_plan(
                &install.compiled.plan,
                &manifest_entry,
            )
            .and_then(|assembly| {
                validate_resident_h3_interfaces(settings, &manifest_entry, &install.compiled.plan)?;
                Ok(assembly)
            }) {
                Ok(assembly) => {
                    manifest_entry.insert(
                        "importPlan".to_owned(),
                        json!({
                            "checkpointId": install.checkpoint_id,
                            "planId": install.compiled.plan.plan_id,
                            "semanticDigest": install.compiled.record.plan.semantic_digest,
                        }),
                    );
                    stamp_imported_minimax_h3_entry(&mut manifest_entry, &assembly);
                }
                Err(reason) => {
                    // Roll the commit back: the bytes are not a model SceneWorks can assemble, so
                    // they must not stay installed and addressable either.
                    if let Err(error) = plan_store.remove_managed(&install.install_id) {
                        tracing::warn!(
                            checkpoint_id = %install.checkpoint_id,
                            %error,
                            "could not roll back an unassemblable MiniMax-H3 import"
                        );
                    }
                    return fail_job(
                        api,
                        &job.id,
                        "Model import is not supported for this file.",
                        Some(format!(
                            "[checkpoint-plan:ambiguous-component] the imported MiniMax-H3 \
                             checkpoint is not a complete DiT assembly: {reason}"
                        )),
                    )
                    .await;
                }
            }
        }
        let model_type = manifest_entry
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("image")
            .to_owned();
        let family = manifest_entry
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_owned);
        apply_model_manifest_defaults(&mut manifest_entry, &model_type, family.as_deref());
        // `paths.model` names the directory this job actually INSTALLED into, so every entry that
        // reaches here gets one — including a MANAGED plan-backed entry, which carries both the
        // install path and `importPlan.checkpointId` and whose bespoke lane still serves the
        // request shapes the plan route does not (sc-20636).
        //
        // A LINKED import never reaches this code at all: it is dispatched to
        // `run_linked_checkpoint_import_job` at the top of this function and installs nothing, and
        // `linked_checkpoint_manifest_entry` strips any inherited `paths.model` itself (sc-20635).
        if let Some(paths) = manifest_entry
            .entry("paths")
            .or_insert_with(|| json!({}))
            .as_object_mut()
        {
            paths.insert(
                "model".to_owned(),
                Value::String(target_dir.display().to_string()),
            );
        }
        if let Err(error) =
            check_cancel(api, &job.id, "Model import canceled before registration.").await
        {
            plan_store
                .remove_managed(&install.install_id)
                .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
            return Err(error);
        }
        let manifest_path = model_manifest_target(settings, &job.payload)?;
        if let Err(error) = publish_claimed_model_import_manifest(
            api,
            job,
            &manifest_path,
            &settings.data_dir,
            manifest_entry,
            job.payload.get("repairExpectedBinding"),
        )
        .await
        {
            plan_store
                .remove_managed(&install.install_id)
                .map_err(|rollback| {
                    WorkerError::InvalidPayload(format!(
                        "{error}; import rollback failed: {rollback}"
                    ))
                })?;
            return Err(error);
        }
    }

    let mut result = JsonObject::new();
    result.insert(
        "modelId".to_owned(),
        job.payload.get("modelId").cloned().unwrap_or(Value::Null),
    );
    result.insert(
        "repo".to_owned(),
        repo.map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "sourceUrl".to_owned(),
        source_url
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "path".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    result.insert(
        "family".to_owned(),
        resolved_family.map(Value::String).unwrap_or(Value::Null),
    );
    result.insert(
        "importQuantFormat".to_owned(),
        import_quant_format
            .map(|quant| Value::String(quant.to_owned()))
            .unwrap_or(Value::Null),
    );
    // sc-20636: the duplicates the compile found, on the JOB RESULT and not only in a log line.
    // "This is the same checkpoint you already have as X" is the user's decision to make — nothing
    // deletes either copy — and a `tracing::info!` never reaches them. Always present (an empty
    // array when there are none) so a client can render it without probing for the key.
    result.insert(
        "duplicateCheckpointIds".to_owned(),
        json!(install.duplicate_checkpoint_ids()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            "Model import completed.",
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    let _ = cleanup_uploaded_import_source(settings, &job.payload).await;
    Ok(())
}
