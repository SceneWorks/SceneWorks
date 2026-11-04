/// Post the terminal `Completed` update for a Hugging Face cache download, building the shared
/// `{<id_key>, repo, path, storage:"huggingface_cache", completedAt}` result object (F-116). Both
/// model download paths (`modelId`) and the LoRA path (`loraId`) funnel through here so the result
/// shape and completion wording stay in one place.
async fn complete_hf_cache_download(
    api: &ApiClient,
    job: &JobSnapshot,
    id_key: &str,
    repo: &str,
    cache_path: &Path,
    message: &str,
) -> WorkerResult<()> {
    let mut result = JsonObject::new();
    result.insert(
        id_key.to_owned(),
        job.payload.get(id_key).cloned().unwrap_or(Value::Null),
    );
    result.insert("repo".to_owned(), Value::String(repo.to_owned()));
    result.insert(
        "path".to_owned(),
        Value::String(cache_path.display().to_string()),
    );
    result.insert(
        "storage".to_owned(),
        Value::String("huggingface_cache".to_owned()),
    );
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            message,
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_model_download_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = match required_payload_string(&job.payload, "repo") {
        Ok(repo) => repo,
        Err(error) => {
            fail_job(
                api,
                &job.id,
                "Model download is missing a repository.",
                Some(error.to_string()),
            )
            .await?;
            return Ok(());
        }
    };
    let managed_identity = sceneworks_core::managed_file::authoritative_job_identity(&job.payload)
        .map_err(WorkerError::InvalidPayload)?;
    let files = payload_string_array(&job.payload, "files");
    let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
    let fresh_download = optional_payload_string(&job.payload, "downloadAction") == Some("fresh");
    // The jobs API is the worker's trust boundary (unauthenticated for local use, LAN-exposed via
    // epic 4484), so reject any repo / revision / include-pattern that could escape the intended
    // Hugging Face path before it reaches `HuggingFaceSnapshot::resolve`'s URL builder or the
    // `download_snapshot_into_cache` refs/snapshot joins below — the on-demand tier fetches already
    // gate this in `ensure_hf_files_cached`, but this user-facing entry point did not (sc-13583).
    let repo = validate_hf_repo_id(repo)?;
    validate_hf_download_inputs(repo, revision, &files)?;
    // The worker is the trust boundary (jobs API is unauthenticated for local use), so a
    // client-supplied targetDir must be constrained to app-managed data/models the same way
    // import jobs are, not used verbatim.
    let target_dir = resolve_model_import_target(
        settings,
        &job.payload,
        settings
            .data_dir
            .join("models")
            .join(safe_download_dir(repo)),
    )?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.1,
            &format!("Downloading {repo}: estimating size."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "Model download canceled before transfer started.",
    )
    .await?;

    // The model-import download uses the native Rust downloader below. The Python
    // `hf`/`huggingface-cli` subprocess it used to prefer reported progress exactly once (0.12) and
    // never updated it, and had no forward-progress watchdog, so on a host with `hf` on PATH a model
    // download froze the bar at 12% and a hung HF CDN edge never recovered (sc-12227 / sc-12232,
    // issue #1554). The native path streams with real byte-level progress AND the stall watchdog
    // (sc-11939), resumes via HTTP Range, and keeps a Python dependency out of this flow (epic 3482).
    // Mirrors `run_lora_download_job`; the on-demand tier fetches share `ensure_hf_files_cached`.

    // Download into the standard Hugging Face hub cache (models--<org>--<name>),
    // not the private app store, so HF-sourced weights dedupe with other tools and
    // the Python loader instead of being duplicated under data/models (sc-1904).
    let repo_dir = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    let mut snapshot =
        HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
    if let Some(identity) = managed_identity {
        snapshot.enforce_managed_identity(identity)?;
    }
    // A download that resolves to ZERO files is not a success: the repo/revision has nothing under
    // the requested filter — e.g. a quant tier whose weights aren't uploaded yet (sc-8513 rollout).
    // Failing here, instead of downloading nothing and still writing a completion marker, surfaces a
    // clear error rather than a silent no-op that later reads as an empty/phantom install (sc-9909).
    if snapshot.files.is_empty() {
        let scope = if files.is_empty() {
            String::new()
        } else {
            format!(" matching {}", files.join(", "))
        };
        fail_job(
            api,
            &job.id,
            &format!("No files to download for {repo}{scope}. This tier may not be published yet."),
            Some(
                "The Hugging Face repository/revision matched zero files for the requested filter."
                    .to_owned(),
            ),
        )
        .await?;
        return Ok(());
    }
    // sc-12283: the same must hold PER PATTERN, not just in aggregate. `allow_pattern_matches` ORs
    // across the filter, so a tier whose `q8/transformer/*` matched but whose `q8/tokenizer/*`
    // matched NOTHING passed the check above, downloaded, completed, and wrote an install marker —
    // leaving a tier that is "installed" by every marker we keep and unloadable in practice. (The
    // load-side half of that failure is sc-12279, which now falls back to a complete sibling tier;
    // this stops the torn tier being created in the first place.)
    //
    // Safe to hard-fail: swept all 217 `downloads[].files` patterns across the 53 HF repos that
    // declare one (builtin.models.jsonc, comment-stripped through this workspace's own
    // `strip_jsonc_comments`) — every pattern matches at least one file today, so no download that
    // works now begins to fail. A future entry that trips this is a real publishing gap, and failing
    // loudly at download beats a phantom install the user only discovers at generation time.
    let unmatched = unmatched_patterns(&files, &snapshot);
    if !unmatched.is_empty() {
        fail_job(
            api,
            &job.id,
            &format!(
                "Incomplete download for {repo}: nothing matches {}. Installing it would leave a \
                 partial model that fails to load.",
                unmatched.join(", ")
            ),
            Some(format!(
                "{} of the {} declared file patterns matched zero files in the Hugging Face \
                 repository/revision.",
                unmatched.len(),
                files.len()
            )),
        )
        .await?;
        return Ok(());
    }
    if let Some(total_bytes) = snapshot.total_bytes() {
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Downloading,
                ProgressStage::Downloading,
                0.1,
                &format!("Downloading {repo}: 0 B of {}.", format_bytes(total_bytes)),
                None,
                None,
                None,
            ),
        )
        .await?;
    }

    let mut progress = DownloadProgress::new(
        repo,
        0,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    let snapshot_revision = download_snapshot_into_cache(
        &DownloadContext {
            api,
            client: http_client,
            settings,
            job_id: &job.id,
            cancel_message: "Model download canceled by user.",
            fresh_download,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let cache_path = huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(repo_dir.clone());
    // Some upstreams (Kolors sc-4764, Qwen-Image sc-6570) ship no fast `tokenizer.json`; overlay the
    // derived one so the in-process Rust generator/trainer can construct. No-op for every other repo.
    overlay_derived_tokenizer(api, settings, http_client, &job.id, repo, &cache_path).await?;
    // A lightweight install marker stays in the app store (parity with the CLI
    // path's marker_dir) so the catalog's data/models pointer and bookkeeping
    // remain intact; the weights themselves live only in the shared HF cache.
    let resolved_files = snapshot
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    let calibration_opt_in = job
        .payload
        .get("memoryCalibrationProvenanceRequired")
        .and_then(Value::as_bool)
        == Some(true);
    let artifact_tree_stamp = if calibration_opt_in {
        Some(resolved_files_tree_stamp(
            &repo_dir.join("snapshots").join(&snapshot_revision),
            &resolved_files,
        )?)
    } else {
        None
    };
    let resolved_tier = download_payload_resolved_tier(&job.payload);
    write_model_download_receipt(
        &target_dir,
        &job.payload,
        repo,
        &job.id,
        &resolved_files,
        Some(&snapshot_revision),
        crate::imports::DownloadArtifactReceipt {
            resolved_tier: resolved_tier.as_deref(),
            tree_stamp: artifact_tree_stamp.as_deref(),
            ..Default::default()
        },
    )
    .await?;

    if !reconcile_downloaded_model_family(api, job, &cache_path).await? {
        return Ok(());
    }

    // A locally derived tier (sc-22999, YuE2 q8 / q4): the original just fetched (or re-verified)
    // is converted on this machine by the audio lane's preparer and verified against the pins.
    if job
        .payload
        .contains_key(crate::local_derivation::PAYLOAD_KEY)
    {
        let original = repo_dir.join("snapshots").join(&snapshot_revision);
        // sc-26241: a FLUX 3 conversion runs the image provider's own converter; every other
        // derivation is the audio lane's preparer.
        let flux3 = job
            .payload
            .get(crate::local_derivation::PAYLOAD_KEY)
            .and_then(|block| block.get("conversion"))
            .and_then(Value::as_str)
            .and_then(crate::local_derivation::Flux3Conversion::from_id);
        match flux3 {
            Some(conversion) => {
                crate::local_derivation::derive_local_tier(
                    api,
                    settings,
                    job,
                    &original,
                    crate::local_derivation::flux3_preparer(
                        conversion,
                        repo.to_owned(),
                        snapshot_revision.clone(),
                    ),
                )
                .await?;
            }
            None => {
                crate::local_derivation::derive_local_tier(
                    api,
                    settings,
                    job,
                    &original,
                    crate::inference_runtime::prepare_audio_snapshot,
                )
                .await?;
            }
        }
    }

    complete_hf_cache_download(
        api,
        job,
        "modelId",
        repo,
        &cache_path,
        "Model download completed in the Hugging Face cache.",
    )
    .await
}

/// The declared `files` patterns that matched ZERO files in `snapshot` (sc-12283).
///
/// Testing against the RESOLVED files is sound: `snapshot.files` is every remote file matching ANY
/// pattern (the filter ORs — see `allow_pattern_matches`), so a pattern with no match among them
/// matched none in the repo. That OR is exactly why this check is needed: without it a filter is
/// satisfied as a whole while an individual component of it silently matched nothing.
///
/// Empty `files` (a deliberate whole-repo fetch) declares no per-pattern claim, so nothing to check —
/// the caller's aggregate `snapshot.files.is_empty()` test covers that case.
fn unmatched_patterns(files: &[String], snapshot: &HuggingFaceSnapshot) -> Vec<String> {
    files
        .iter()
        .filter(|pattern| {
            !snapshot
                .files
                .iter()
                .any(|file| pattern_matches(pattern, &file.path))
        })
        .cloned()
        .collect()
}

/// Download a built-in catalog LoRA's Hugging Face repo/file into the shared HF cache
/// (sc-5944). Mirrors the `run_model_download_job` cache path but skips the model-only
/// steps (family reconciliation, tokenizer overlay, install marker): a LoRA is a single
/// adapter file. Landing it in the HF cache is exactly what the API's install-state probe
/// (`lora_huggingface_cached_file`) and the engine's on-demand generation-time fetch read,
/// so the catalog entry flips to "installed" on the next `/api/v1/loras` refresh.
pub(crate) async fn run_lora_download_job(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let repo = match required_payload_string(&job.payload, "repo") {
        Ok(repo) => repo,
        Err(error) => {
            fail_job(
                api,
                &job.id,
                "LoRA download is missing a repository.",
                Some(error.to_string()),
            )
            .await?;
            return Ok(());
        }
    };
    let files = payload_string_array(&job.payload, "files");
    let revision = optional_payload_string(&job.payload, "revision").unwrap_or("main");
    // Trust-boundary validation for the LAN-exposed jobs API (epic 4484), mirroring
    // `run_model_download_job`: reject a path-escaping repo/revision/pattern before the URL builder
    // and the `download_snapshot_into_cache` refs/snapshot joins downstream (sc-13583).
    let repo = validate_hf_repo_id(repo)?;
    validate_hf_download_inputs(repo, revision, &files)?;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.1,
            &format!("Downloading LoRA {repo}: estimating size."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        "LoRA download canceled before transfer started.",
    )
    .await?;

    let repo_dir = huggingface_repo_cache_path(&settings.data_dir, repo).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "Unable to resolve Hugging Face cache path for {repo}."
        ))
    })?;
    let snapshot =
        HuggingFaceSnapshot::resolve(http_client, settings, repo, revision, &files).await?;
    // sc-12288: a LoRA download that fetched NOTHING is not a success. This path had no zero-file
    // check at all (the sibling `run_model_download_job` got one in sc-9909; this never did), so a
    // LoRA whose declared `source.file` is absent — renamed upstream, typo'd entry, unpublished —
    // downloaded nothing and reported the job COMPLETED. The API's install probe
    // (`lora_huggingface_cached_file`) then finds an empty cache and the catalog still reads "not
    // installed", so the user got a job that claimed success and a LoRA that never appeared, with no
    // error anywhere.
    //
    // An empty `files` list is NOT an omission here — it deliberately means "fetch the (small) whole
    // repo" (apps/rust-api `create_lora_download_job`) — but that only widens the filter, so resolving
    // zero files still means there was nothing to fetch.
    //
    // Safe to hard-fail: swept every entry in builtin.loras.jsonc against its repo — 8 entries, all
    // declaring an explicit `source.file`, all 8 present, no whole-repo entries. Nothing shipping
    // relies on this silently passing. (The 8th, `krea2_turbo_accel` sc-13882, pins a SHA revision;
    // its file was confirmed present at that snapshot.)
    if snapshot.files.is_empty() {
        let scope = if files.is_empty() {
            String::new()
        } else {
            format!(" matching {}", files.join(", "))
        };
        fail_job(
            api,
            &job.id,
            &format!(
                "No files to download for LoRA {repo}{scope}. The adapter may have been renamed or \
                 removed upstream."
            ),
            Some(
                "The Hugging Face repository/revision matched zero files for the requested filter."
                    .to_owned(),
            ),
        )
        .await?;
        return Ok(());
    }
    // The same per-pattern gap as sc-12283: `create_lora_download_job` accepts an explicit `files`
    // LIST, and the filter ORs across it, so a multi-file adapter that landed one half and not the
    // other would pass the aggregate check above and complete as a torn install.
    let unmatched = unmatched_patterns(&files, &snapshot);
    if !unmatched.is_empty() {
        fail_job(
            api,
            &job.id,
            &format!(
                "Incomplete download for LoRA {repo}: nothing matches {}. Installing it would leave \
                 a partial adapter.",
                unmatched.join(", ")
            ),
            Some(format!(
                "{} of the {} declared file patterns matched zero files in the Hugging Face \
                 repository/revision.",
                unmatched.len(),
                files.len()
            )),
        )
        .await?;
        return Ok(());
    }
    if let Some(total_bytes) = snapshot.total_bytes() {
        update_job(
            api,
            &job.id,
            progress_payload(
                JobStatus::Downloading,
                ProgressStage::Downloading,
                0.1,
                &format!(
                    "Downloading LoRA {repo}: 0 B of {}.",
                    format_bytes(total_bytes)
                ),
                None,
                None,
                None,
            ),
        )
        .await?;
    }

    let mut progress = DownloadProgress::new(
        repo,
        0,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    let resolved_revision = download_snapshot_into_cache(
        &DownloadContext {
            api,
            client: http_client,
            settings,
            job_id: &job.id,
            cancel_message: "LoRA download canceled by user.",
            fresh_download: false,
        },
        &repo_dir,
        revision,
        &snapshot,
        &mut progress,
    )
    .await?;
    let resolved_files = snapshot
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    // Persist the exact adapter digest at install time, not only after SceneWorks first generates
    // with it. That lets an imported image resolve to an already-installed HF LoRA immediately.
    // The API's receipt resolver selects the first resolved safetensors file in this same order.
    let selected_lora = snapshot.files.iter().find(|file| {
        Path::new(&file.path)
            .extension()
            .and_then(|ext| ext.to_str())
            == Some("safetensors")
    });
    let (lora_file_identity, lora_file_sha256) = if let Some(selected) = selected_lora {
        let selected_path = safe_join(
            &repo_dir.join("snapshots").join(&resolved_revision),
            &selected.path,
        )?;
        if let Some(hash) = selected.sha256.as_deref().and_then(normalize_sha256) {
            let identity = model_file_identity(&selected_path).ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "Downloaded LoRA file is missing: {}",
                    selected_path.display()
                ))
            })?;
            (Some(identity), Some(hash))
        } else {
            let (hash, identity) =
                stable_model_file_sha256(api, settings, &job.id, &selected_path).await?;
            (Some(identity), Some(hash))
        }
    } else {
        (None, None)
    };
    let receipt_dir = settings.data_dir.join("loras").join(safe_download_dir(
        optional_payload_string(&job.payload, "loraId").unwrap_or(repo),
    ));
    write_model_download_receipt(
        &receipt_dir,
        &job.payload,
        repo,
        &job.id,
        &resolved_files,
        Some(&resolved_revision),
        crate::imports::DownloadArtifactReceipt {
            lora_file_identity: lora_file_identity.as_ref(),
            lora_file_sha256: lora_file_sha256.as_deref(),
            ..Default::default()
        },
    )
    .await?;
    let cache_path = huggingface_snapshot_dir(&settings.data_dir, repo).unwrap_or(repo_dir);

    complete_hf_cache_download(
        api,
        job,
        "loraId",
        repo,
        &cache_path,
        "LoRA download completed in the Hugging Face cache.",
    )
    .await
}
