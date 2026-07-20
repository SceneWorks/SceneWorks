use super::*;

use sceneworks_core::credentials::normalize_host;
use sceneworks_core::lora_family::is_hidden_file;

const ALLOWED_MODEL_TYPES: &[&str] = &["image", "video", "audio", "utility"];
const MODEL_SIZE_CACHE_LIMIT: usize = 64;
// Failed estimates (offline, rate-limited, or size-less repo metadata) are
// negative-cached so a huggingface.co outage costs one 8s timeout per repo per
// TTL window instead of one per catalog load (sc-4169).
const MODEL_SIZE_NEGATIVE_TTL: Duration = Duration::from_secs(300);

fn validate_huggingface_repo(repo: &str) -> Result<(), ApiError> {
    let parts: Vec<_> = repo.trim().split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with('.')
                || part.ends_with('.')
                || !part.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
                })
        })
    {
        return Err(ApiError::bad_request(
            "Hugging Face repo must be in owner/name form",
        ));
    }
    Ok(())
}

#[derive(Debug, Default)]
pub(crate) struct ModelSizeCache {
    entries: HashMap<ModelSizeCacheKey, CachedSizeEstimate>,
    order: VecDeque<ModelSizeCacheKey>,
}

type ModelSizeCacheKey = (String, Vec<String>);

#[derive(Debug, Clone, Copy)]
struct CachedSizeEstimate {
    size_bytes: Option<u64>,
    expires_at: Option<std::time::Instant>,
}

impl ModelSizeCache {
    /// `Some(Some(bytes))` = cached estimate, `Some(None)` = cached failure
    /// (skip the network until the TTL lapses), `None` = cache miss.
    pub(crate) fn get(&mut self, key: &ModelSizeCacheKey) -> Option<Option<u64>> {
        if let Some(entry) = self.entries.get(key).copied() {
            if entry
                .expires_at
                .is_some_and(|expires_at| std::time::Instant::now() >= expires_at)
            {
                self.entries.remove(key);
                self.order.retain(|existing| existing != key);
                return None;
            }
            self.touch(key);
            return Some(entry.size_bytes);
        }
        None
    }

    pub(crate) fn insert(&mut self, key: ModelSizeCacheKey, value: u64) {
        self.insert_entry(
            key,
            CachedSizeEstimate {
                size_bytes: Some(value),
                expires_at: None,
            },
        );
    }

    pub(crate) fn insert_failure(&mut self, key: ModelSizeCacheKey) {
        self.insert_failure_expiring_at(key, std::time::Instant::now() + MODEL_SIZE_NEGATIVE_TTL);
    }

    pub(crate) fn insert_failure_expiring_at(
        &mut self,
        key: ModelSizeCacheKey,
        expires_at: std::time::Instant,
    ) {
        self.insert_entry(
            key,
            CachedSizeEstimate {
                size_bytes: None,
                expires_at: Some(expires_at),
            },
        );
    }

    fn insert_entry(&mut self, key: ModelSizeCacheKey, entry: CachedSizeEstimate) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, entry);
        while self.order.len() > MODEL_SIZE_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &ModelSizeCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DownloadContext {
    repo: String,
    files: Vec<String>,
    fallback_size_bytes: Option<u64>,
}

pub(crate) async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(model_catalog_sized(&state).await?))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuggingFaceCacheHealth {
    pub(crate) installed: bool,
    pub(crate) incomplete: bool,
    pub(crate) missing_files: Vec<String>,
}

impl HuggingFaceCacheHealth {
    fn missing(missing_files: Vec<String>) -> Self {
        Self {
            installed: false,
            incomplete: true,
            missing_files,
        }
    }

    fn installed() -> Self {
        Self {
            installed: true,
            incomplete: false,
            missing_files: Vec::new(),
        }
    }
}

pub(crate) async fn create_model_download_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    // Tier selection (sc-8508): an explicit `variant` installs that quant tier's download entry; an
    // absent variant installs the default tier (back-compat). A variant the model doesn't advertise
    // is a 400 rather than a silent wrong-tier install.
    let download = match payload
        .variant
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        Some(variant) => model_download_for_variant(&model, variant).ok_or_else(|| {
            ApiError::bad_request(format!(
                "Model does not define a '{variant}' download variant"
            ))
        })?,
        None => model_download(&model).ok_or_else(|| {
            ApiError::bad_request("Model does not define a Hugging Face download")
        })?,
    };
    // The selected `download` is always the primary/tier entry — `model_download` and
    // `model_download_for_variant` skip co-requisites (sc-9696), so a co-requisite can never be
    // installed as if it were the model itself.
    let requested_variant = payload
        .variant
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let job_payload = build_model_download_job_payload(
        &model,
        &model_id,
        &download,
        requested_variant,
        true,
        &state.settings.data_dir,
    )?;

    // Co-requisites (sc-9696): dependencies that must install ALONGSIDE the primary — e.g. the PiD
    // decoder's shared gemma-2-2b-it caption encoder, or 10Eros's cond_safe distill LoRA. Without
    // them the feature silently no-ops (for PiD, `resolve_pid_weights` falls back to the native VAE
    // with no error). The catalog already filtered `downloads` to this OS, so every co-requisite
    // here applies. Each is queued as its own ModelDownload job (the worker is one-repo-per-job);
    // the catalog reports the entry installed only once all of them are cached
    // (`install_state_for`). `include_family: false` because a co-requisite (e.g. a text encoder)
    // is a different artifact than the model's primary checkpoint and must not be reconciled
    // against the model's family.
    let requested_gpu = requested_gpu_or_auto(payload.requested_gpu);
    for co_requisite in model_co_requisite_downloads(&model) {
        let co_payload = build_model_download_job_payload(
            &model,
            &model_id,
            &co_requisite,
            None,
            false,
            &state.settings.data_dir,
        )?;
        create_generation_job(
            state.clone(),
            JobType::ModelDownload,
            None,
            None,
            co_payload,
            requested_gpu.clone(),
        )
        .await?;
    }

    // The primary job is the one returned to the caller (its id is what the download UI tracks).
    let job = create_generation_job(
        state,
        JobType::ModelDownload,
        None,
        None,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

/// Build the worker `ModelDownload` job payload for one `download` entry of `model`. Factored out
/// (sc-9696) so the primary download and each co-requisite share identical payload shaping.
/// `explicit_variant` records a request-selected quant tier (falling back to the entry's own
/// `variant`); `include_family` forwards the model's declared family for the worker's post-download
/// family reconcile (sc-1663) — pass `false` for co-requisites, whose weights are a different artifact
/// than the model's primary checkpoint.
fn build_model_download_job_payload(
    model: &Value,
    model_id: &str,
    download: &Value,
    explicit_variant: Option<&str>,
    include_family: bool,
    data_dir: &FsPath,
) -> Result<JsonObject, ApiError> {
    let repo = required_string_field(download, "repo")?.to_owned();
    let files = download
        .get("files")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.to_owned()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id)
                .to_owned(),
        ),
    );
    job_payload.insert(
        "provider".to_owned(),
        Value::String(required_string_field(download, "provider")?.to_owned()),
    );
    job_payload.insert("repo".to_owned(), Value::String(repo.clone()));
    job_payload.insert("files".to_owned(), json!(files));
    // Record which quant tier this job installs (sc-8508) so the download record + per-variant
    // install tracking agree on the tier. Falls back to the selected entry's own `variant` when the
    // request omitted one (the default tier may still be a labeled variant).
    if let Some(variant) = explicit_variant
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            download
                .get("variant")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
    {
        job_payload.insert("variant".to_owned(), Value::String(variant));
    }
    // Forward the catalog-declared family so the worker can re-verify the downloaded
    // weights match it (parity with model import). The catalog is project-curated, but
    // a mis-declared family would otherwise silently mismatch downstream adapter
    // selection; the worker reconciles and fails on a confident conflict (sc-1663).
    if include_family {
        if let Some(family) = model.get("family").and_then(Value::as_str) {
            if !family.trim().is_empty() {
                job_payload.insert("family".to_owned(), Value::String(family.to_owned()));
            }
        }
    }
    job_payload.insert(
        "targetDir".to_owned(),
        Value::String(
            data_dir
                .join("models")
                .join(safe_download_dir(&repo))
                .display()
                .to_string(),
        ),
    );
    Ok(job_payload)
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
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
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
    let output_dir = state
        .settings
        .data_dir
        .join("models")
        .join("mlx")
        .join(&model_id);
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.clone()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&model_id)
                .to_owned(),
        ),
    );
    job_payload.insert("sourceRepo".to_owned(), Value::String(source_repo));
    job_payload.insert(
        "outputDir".to_owned(),
        Value::String(output_dir.display().to_string()),
    );
    job_payload.insert("dtype".to_owned(), Value::String("bfloat16".to_owned()));
    // Optional converter discriminator + inputs (sc-2235). Default (absent) is the
    // mlx-video Wan converter. A FLUX.2-klein community fine-tune declares
    // `mlx.converter` + the single-file source + the base repo whose
    // VAE/text-encoder/tokenizer are borrowed during assembly.
    if let Some(converter) = mlx
        .get("converter")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert("converter".to_owned(), Value::String(converter.to_owned()));
    }
    if let Some(source_file) = mlx
        .get("convertSourceFile")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert(
            "sourceFile".to_owned(),
            Value::String(source_file.to_owned()),
        );
    }
    if let Some(base_repo) = mlx
        .get("convertBaseRepo")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert("baseRepo".to_owned(), Value::String(base_repo.to_owned()));
    }
    if quantize_only {
        job_payload.insert("quantizeOnly".to_owned(), Value::Bool(true));
    }
    if let Some(bits) = payload.quantize_bits {
        job_payload.insert("quantizeBits".to_owned(), Value::from(bits));
    }
    if let Some(group_size) = payload.quantize_group_size {
        job_payload.insert("quantizeGroupSize".to_owned(), Value::from(group_size));
    }

    let job = create_generation_job(
        state,
        JobType::ModelConvert,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Query(query): Query<CatalogDeleteQuery>,
) -> Result<Json<Value>, ApiError> {
    let permanent = query.permanent.unwrap_or(false);
    let catalog = model_catalog(&state).await?;
    let model = catalog
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    // Peek (not remove) the manifest entry so that if moving the files to the OS
    // trash fails we can leave the catalog untouched and prompt for confirmation.
    let manifest_entry = load_manifest_entries(&state, &manifest_path, "models")
        .await?
        .into_iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id.as_str()));
    let cleanup_source = manifest_entry.as_ref().unwrap_or(&model);
    let allowed_roots = vec![
        state.settings.data_dir.join("models"),
        huggingface_hub_cache_dir(&state.settings.data_dir),
    ];
    let removal = remove_owned_artifacts(
        model_artifact_paths(cleanup_source, &state.settings.data_dir),
        &allowed_roots,
        permanent,
    )
    .await?;
    // Some owned files could not reach the OS trash and nothing was permanently
    // deleted. Leave the registry entry in place and ask the client to confirm.
    if !permanent && !removal.trash_failed_paths.is_empty() {
        return Ok(Json(json!({
            "id": model_id,
            "kind": "model",
            "trashUnavailable": true,
            "trashFailedPaths": removal.trash_failed_paths,
            "removedManifestEntry": false,
            "removedLocalArtifacts": !removal.removed_paths.is_empty(),
            "removedPaths": removal.removed_paths,
            "retainedPaths": removal.retained_paths,
        })));
    }
    let removed_entry =
        remove_catalog_manifest_entry(&state, &manifest_path, "models", &model_id).await?;
    if removed_entry.is_none() && removal.removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in model catalog entries are read-only unless local files are installed",
        ));
    }
    let warnings = catalog_delete_warnings(&state, "model", &model_id, None).await?;
    let policy = if removed_entry.is_some() {
        "Removed the model registry entry and SceneWorks-owned local model files."
    } else {
        "Built-in model catalog entries are retained; SceneWorks-owned local model files were removed."
    };
    Ok(Json(json!({
        "id": model_id,
        "kind": "model",
        "trashed": !permanent,
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

/// Delete ONE installed quant tier of a model and reclaim its disk, leaving the other tiers
/// (and the catalog entry) intact (sc-12024, epic 8506). The counterpart to per-tier download
/// (sc-8509): a user who fetched q8 to A/B against q4 can drop the unused tier without nuking
/// the whole model. Unlike `delete_model` — which removes the whole repo dir AND the registry
/// entry — this is scoped to the tier's `files` and never touches the manifest, so the model
/// stays catalogued and re-downloadable; deleting the last remaining tier simply flips it back
/// to not-installed. Unlike `delete_model` this deletes PERMANENTLY (never the OS trash): a tier is
/// many loose HF-cache blobs + snapshot symlinks, and trashing them one-by-one drove a macOS
/// per-file permission-prompt loop ("you don't have permission to access some of the items") — and a
/// tier isn't restorable from loose trashed blobs anyway (sc-12088). Same ownership guard as
/// `delete_model` (`<data>/models` + the HF hub cache).
pub(crate) async fn delete_model_variant(
    State(state): State<AppState>,
    Path((model_id, variant)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let variant = variant.trim().to_ascii_lowercase();
    let catalog = model_catalog(&state).await?;
    let model = catalog
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
        })?;
    let data_dir = &state.settings.data_dir;
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    // A tier lives in one of two storage shapes. Download-matrix models (`hasVariantMatrix`) keep the
    // tier as a `files`-filtered slice of a shared HF cache repo (sc-12024); convert-at-install
    // models (Anima) keep it as a real `<converted>/<tier>/` dir emitted by one convert job
    // (sc-12025). Resolve whichever this model uses; a variant that is neither has nothing to delete.
    let removal = if let Some(download) = model_download_for_variant(&model, &variant) {
        let repo = download
            .get("repo")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let files = string_array_field(&download, "files");
        // A tier with no `files` scope is the whole repo (a single-variant "default"), not a
        // deletable slice of a shared cache — refuse rather than risk wiping every tier. The UI
        // only offers this on real quant tiers (bf16/q8/q4), which always carry a `files` glob.
        if files.is_empty() {
            return Err(ApiError::bad_request(format!(
                "Tier '{variant}' has no file scope; delete the whole model instead"
            )));
        }
        let repo_cache = huggingface_repo_cache_path(data_dir, &repo);
        let managed_dir = Some(data_dir.join("models").join(safe_download_dir(&repo)));
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_tier_artifacts(repo_cache, managed_dir, &files, &allowed_roots, true).await?
    } else if model_has_convert_tier(&model, &variant) {
        // Convert-at-install: the tier is a real dir under the converted MLX tree. Prefer the
        // catalog's resolved `mlxConvertedPath`; fall back to the canonical convert output dir.
        let converted = model
            .get("mlxConvertedPath")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("models").join("mlx").join(&model_id));
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_converted_tier(converted.join(&variant), &allowed_roots, true).await?
    } else {
        return Err(ApiError::bad_request(format!(
            "Model does not advertise a '{variant}' quant tier"
        )));
    };
    if removal.removed_paths.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Tier '{variant}' is not installed"
        )));
    }
    Ok(Json(json!({
        "id": model_id,
        "variant": variant,
        "kind": "model-variant",
        // Permanent delete: no OS trash, no undo (sc-12088).
        "trashed": false,
        // A tier delete NEVER removes the registry entry: the model stays in the catalog so the
        // tier can be re-downloaded. Emitted false so the web keeps the model card in place.
        "removedManifestEntry": false,
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "reclaimedBytes": removal.reclaimed_bytes,
        "reclaimedLabel": format_bytes(removal.reclaimed_bytes),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
    })))
}

/// Result of removing a single quant tier's on-disk artifacts (sc-12024).
#[derive(Default)]
pub(crate) struct TierRemoval {
    /// Paths (tier symlinks/files + their exclusive blobs) moved to the OS trash or unlinked.
    pub(crate) removed_paths: Vec<String>,
    /// Paths left in place because they are not inside a SceneWorks-owned root.
    pub(crate) retained_paths: Vec<String>,
    /// Owned paths that could NOT be moved to the OS trash (recycle bin disabled, unsupported
    /// volume, …). Nothing was deleted for these; the caller prompts before a permanent delete.
    pub(crate) trash_failed_paths: Vec<String>,
    /// Bytes actually reclaimed — the summed size of the data-bearing files/blobs removed.
    pub(crate) reclaimed_bytes: u64,
}

/// Remove ONE quant tier's artifacts from a download-matrix model's storage, reclaiming disk.
///
/// A download-matrix model keeps every tier (bf16/q8/q4) in ONE shared Hugging Face hub-cache
/// repo: the real bytes live in `blobs/<etag>` and each tier's files are relative SYMLINKS into
/// `blobs/` (`download_snapshot_into_cache`, crates/sceneworks-worker/src/downloads.rs). Deleting
/// the tier's snapshot symlinks alone frees nothing — the blobs behind them must go too. This
/// walks every snapshot revision under the repo cache (and the app-managed mirror dir, where a
/// turnkey install lands real files), selects the files matching the tier's `files` globs, and
/// removes those directory entries PLUS the blobs they resolve to — while PROTECTING any blob
/// still referenced by a retained tier (a shared etag). Emptied tier/snapshot dirs, and the whole
/// repo cache dir once no tier's payload remains, are pruned (best-effort; only ever unlinks EMPTY
/// dirs, so a surviving tier is never touched). `reclaimed_bytes` reflects only what actually left
/// disk. An empty `tier_files` is a no-op — the caller must never scope a delete to "everything".
pub(crate) async fn remove_tier_artifacts(
    repo_cache: Option<PathBuf>,
    managed_dir: Option<PathBuf>,
    tier_files: &[String],
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<TierRemoval, ApiError> {
    if tier_files.is_empty() {
        return Ok(TierRemoval::default());
    }
    // The directories to scan: every snapshot revision under the HF repo cache, plus the managed
    // mirror dir. Each is scanned independently and files are matched RELATIVE to their scan dir.
    let mut scan_dirs: Vec<PathBuf> = Vec::new();
    if let Some(repo_cache) = repo_cache.as_ref() {
        if huggingface_repo_cache_exists(repo_cache) {
            scan_dirs.extend(huggingface_snapshot_dirs(repo_cache));
        }
    }
    if let Some(managed_dir) = managed_dir.as_ref() {
        if managed_dir.is_dir() {
            scan_dirs.push(managed_dir.clone());
        }
    }

    // Split every file under the scan dirs into this tier's entries vs the retained rest. A
    // retained file's real data (blob) must survive even if a tier symlink shares its etag.
    let mut tier_entries: Vec<PathBuf> = Vec::new();
    let mut retained_reals: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for dir in &scan_dirs {
        for rel in snapshot_files(dir) {
            let abs = dir.join(&rel);
            if tier_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel))
            {
                tier_entries.push(abs);
            } else if let Ok(real) = tokio::fs::canonicalize(&abs).await {
                retained_reals.insert(real);
            }
        }
    }

    // Build the ordered removal plan: unlink the tier's directory ENTRIES first (so a symlink
    // still resolves to its blob for the ownership check), THEN the blobs those symlinks resolve
    // to (skipping any shared with a retained tier). `data_sizes` records the byte size of every
    // data-bearing path so reclaimed bytes reflect exactly what leaves disk.
    let mut ordered: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut data_sizes: std::collections::HashMap<PathBuf, u64> = std::collections::HashMap::new();
    for entry in &tier_entries {
        // A real file (managed mirror) IS its own data holder; a symlink's bytes live in its blob.
        if let Ok(link_meta) = tokio::fs::symlink_metadata(entry).await {
            if !link_meta.file_type().is_symlink() && link_meta.is_file() {
                data_sizes.insert(entry.clone(), link_meta.len());
            }
        }
        if seen.insert(entry.clone()) {
            ordered.push(entry.clone());
        }
    }
    for entry in &tier_entries {
        // Resolve the snapshot entry to the blob it references and remove that blob unless a retained
        // tier shares it. On macOS/Linux the entry is a SYMLINK so `canonicalize` yields the blob. On
        // Windows the HF cache uses HARDLINKS, which `canonicalize` does NOT resolve to a different
        // path (`real == entry`), so the blob's second name under blobs/ is not reclaimed here — the
        // Windows hardlink reverse-map is tracked in sc-12038. macOS/Linux (primary targets) reclaim
        // fully; the unix `variant_delete_tests` cover it.
        if let Ok(real) = tokio::fs::canonicalize(entry).await {
            if &real != entry && !retained_reals.contains(&real) {
                if let Ok(meta) = tokio::fs::metadata(&real).await {
                    data_sizes.entry(real.clone()).or_insert(meta.len());
                }
                if seen.insert(real.clone()) {
                    ordered.push(real);
                }
            }
        }
    }

    let removal = remove_owned_artifacts(ordered, allowed_roots, permanent).await?;
    let reclaimed_bytes = removal
        .removed_paths
        .iter()
        .filter_map(|path| data_sizes.get(FsPath::new(path)))
        .sum();
    let tier_removal = TierRemoval {
        removed_paths: removal.removed_paths,
        retained_paths: removal.retained_paths,
        trash_failed_paths: removal.trash_failed_paths,
        reclaimed_bytes,
    };

    // Best-effort tidy once the removal itself succeeded: drop now-empty tier/snapshot dirs, and
    // the whole repo cache dir once no tier's payload remains (otherwise only the tiny refs/
    // skeleton would linger). Only ever removes EMPTY dirs.
    if tier_removal.trash_failed_paths.is_empty() {
        if let Some(repo_cache) = repo_cache.as_ref() {
            prune_empty_repo_cache(repo_cache).await;
        }
        if let Some(managed_dir) = managed_dir.as_ref() {
            remove_empty_dirs(managed_dir).await;
        }
    }

    Ok(tier_removal)
}

/// Recursively remove empty subdirectories under `dir`, then `dir` itself if it ends up empty.
/// Best-effort (ignores errors) and only ever unlinks EMPTY directories, so a sibling tier's
/// surviving files can never be removed by it.
async fn remove_empty_dirs(dir: &FsPath) {
    let mut children: Vec<PathBuf> = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            children.push(entry.path());
        }
    } else {
        return;
    }
    for child in children {
        if child.is_dir() {
            Box::pin(remove_empty_dirs(&child)).await;
        }
    }
    let _ = tokio::fs::remove_dir(dir).await;
}

/// Prune a download-matrix repo cache dir after a tier delete: remove emptied snapshot/blob
/// subtrees, and — when no payload remains (no blobs, no snapshot files) — the whole repo cache
/// dir, so a fully-drained repo doesn't linger as a bare `refs/` skeleton. Best-effort.
async fn prune_empty_repo_cache(repo_cache: &FsPath) {
    remove_empty_dirs(&repo_cache.join("snapshots")).await;
    remove_empty_dirs(&repo_cache.join("blobs")).await;
    let has_blobs = std::fs::read_dir(repo_cache.join("blobs"))
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);
    let has_snapshot_files = huggingface_snapshot_dirs(repo_cache)
        .iter()
        .any(|snapshot| !snapshot_files(snapshot).is_empty());
    if !has_blobs && !has_snapshot_files {
        let _ = tokio::fs::remove_dir_all(repo_cache).await;
    }
}

/// Whether `model` advertises `variant` as a convert-at-install tier — i.e. it appears in the
/// catalog's `mlxTiers` (the on-disk convert-output tiers of a converted MLX model, sc-10730).
fn model_has_convert_tier(model: &Value, variant: &str) -> bool {
    model
        .get("mlxTiers")
        .and_then(Value::as_array)
        .is_some_and(|tiers| {
            tiers
                .iter()
                .filter_map(Value::as_str)
                .any(|tier| tier.eq_ignore_ascii_case(variant))
        })
}

/// Remove ONE convert-at-install tier dir (`<converted>/<tier>/`) and reclaim its disk (sc-12025).
/// Convert-at-install models (Anima) emit every tier from one convert job as a real per-tier dir
/// holding a packed DiT plus SYMLINKS to the shared dense TE/VAE (whose targets live OUTSIDE the tier
/// dir). Removing the tier dir frees only the packed DiT + the symlink entries — never the shared
/// source, which the other tiers still reference — so `reclaimed_bytes` counts only the real
/// (non-symlink) files under the tier. When this was the LAST tier with weights, the whole converted
/// dir is dropped so the model cleanly reverts to "needs conversion" rather than lingering as a bare
/// `model_index.json` marker.
async fn remove_converted_tier(
    tier_dir: PathBuf,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<TierRemoval, ApiError> {
    if !tier_dir.is_dir() {
        return Ok(TierRemoval::default());
    }
    let reclaimable = converted_tier_real_bytes(&tier_dir);
    let removal = remove_owned_artifacts(vec![tier_dir.clone()], allowed_roots, permanent).await?;
    let removed = removal
        .removed_paths
        .iter()
        .any(|path| FsPath::new(path) == tier_dir);
    // If no sibling tier retains weights, drop the whole converted dir (marker included) so the model
    // reverts to a clean not-converted state instead of a bare marker. Best-effort.
    if removed && removal.trash_failed_paths.is_empty() {
        if let Some(converted) = tier_dir.parent() {
            let any_tier_left = ["bf16", "q8", "q4"]
                .iter()
                .any(|tier| tier_subdir_has_weights(&converted.join(tier)));
            if !any_tier_left {
                let _ = tokio::fs::remove_dir_all(converted).await;
            }
        }
    }
    Ok(TierRemoval {
        removed_paths: removal.removed_paths,
        retained_paths: removal.retained_paths,
        trash_failed_paths: removal.trash_failed_paths,
        reclaimed_bytes: if removed { reclaimable } else { 0 },
    })
}

/// Sum the bytes of the REAL (non-symlink) files under a converted tier dir — the packed DiT — so a
/// tier delete reports only what it actually frees. The shared TE/VAE are symlinks to a source
/// outside the tier dir; following them would over-count disk that a tier delete does not reclaim.
fn converted_tier_real_bytes(tier_dir: &FsPath) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![tier_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.file_type().is_symlink() {
                continue; // shared TE/VAE — its target lives outside the tier dir; not reclaimed
            }
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

/// Kill-switch for the model upload/import endpoint (sc-7081, epic 7080). Disabled on
/// every platform until a real compatibility-check + conversion pipeline exists behind it:
/// today an imported checkpoint never reaches a runnable engine. macOS dropped the torch
/// worker (MLX-only, sc-3492) and resolves engines from a compile-time table a novel
/// imported id is never in; the off-Mac diffusers path only loads full repos via
/// `from_pretrained`, not the single files this endpoint accepts. Flip to `true` once the
/// pipeline gates imports on an architecture-compatibility verdict (kept as a fn, not a
/// `const`, so the guarded handler body stays reachable — no `unreachable_code`).
fn model_import_enabled() -> bool {
    false
}

const MODEL_IMPORT_DISABLED_DETAIL: &str = "Model import is temporarily disabled while native \
     model support and conversion are being built. (LoRA import is unaffected.)";

pub(crate) async fn create_model_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    // sc-7081 (epic 7080): refuse before staging/queueing, covering both the JSON and
    // multipart entrypoints. The route stays mounted so a direct API client gets an
    // actionable 403 rather than a 404. See `model_import_enabled` for the rationale.
    if !model_import_enabled() {
        return Err(ApiError::forbidden(MODEL_IMPORT_DISABLED_DETAIL).into_response());
    }

    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if is_multipart {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()).into_response())?;
        let (payload, staged_path) = model_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_model_import_job(state, payload).await;
        if result.is_err() {
            cleanup_staged_model_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<ModelImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_model_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

pub(crate) async fn queue_model_import_job(
    state: AppState,
    mut payload: ModelImportRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, or source path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    if let Some(repo) = payload.repo.as_deref() {
        validate_huggingface_repo(repo)?;
    }
    let model_type = match payload.model_type.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => {
            let normalized = value.to_ascii_lowercase();
            if !ALLOWED_MODEL_TYPES.contains(&normalized.as_str()) {
                return Err(ApiError::bad_request(format!(
                    "Model type must be one of {}",
                    ALLOWED_MODEL_TYPES.join(", ")
                )));
            }
            normalized
        }
        _ => "image".to_owned(),
    };
    payload.model_type = Some(model_type.clone());
    if let Some(family) = payload.family.take() {
        let models = model_catalog(&state).await?;
        payload.family = Some(validate_lora_family(&models, &family)?);
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            payload
                .source_url
                .as_deref()
                .and_then(|value| lora_source_url_file_stem(value).ok())
        })
        .or_else(|| {
            payload.source_path.as_deref().and_then(|path| {
                FsPath::new(path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "Imported Model".to_owned());
    let model_id = payload
        .model_id
        .clone()
        .unwrap_or_else(|| slugify_lora_id(&name));
    let existing_ids = model_catalog(&state)
        .await?
        .into_iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    if existing_ids.contains(&model_id) {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' already exists. Pick a different id or delete the existing model first."
        )));
    }
    let target_name = safe_download_dir(&model_id);
    let target_dir = state
        .settings
        .data_dir
        .join("models")
        .join("imports")
        .join(&target_name);
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    let source_path_rel = format!("models/imports/{target_name}");
    let allowed_source_roots = vec![state.settings.data_dir.join("models")];
    if let Some(source_path) = payload.source_path.as_deref() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("model-uploads")]
        } else {
            allowed_source_roots
        };
        validate_lora_import_source_path(source_path, &allowed_source_roots)?;
        let detected =
            detect_model_family(FsPath::new(source_path)).map_err(model_family_inspection_error)?;
        payload.family = reconcile_model_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": model_id,
        "name": name,
        "type": model_type,
        "source": {
            "provider": model_import_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path_rel,
        },
        "files": payload.files.clone(),
        "paths": {
            "model": target_dir.display().to_string(),
        },
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
    if let Some(source_url) = payload.source_url.clone() {
        if let Some(source) = manifest_entry
            .get_mut("source")
            .and_then(Value::as_object_mut)
        {
            source.insert("url".to_owned(), Value::String(source_url));
        }
    }
    if let Some(family) = payload.family.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("family".to_owned(), Value::String(family));
        }
    }
    if let Some(object) = manifest_entry.as_object_mut() {
        apply_model_manifest_defaults(object, &model_type, payload.family.as_deref());
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("modelId".to_owned(), manifest_entry["id"].clone());
    payload.insert("modelName".to_owned(), manifest_entry["name"].clone());
    payload.insert(
        "targetDir".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    let job = create_generation_job(
        state,
        JobType::ModelImport,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn model_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(ModelImportRequest, PathBuf), ApiError> {
    let mut payload = ModelImportRequest {
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        family: None,
        expected_sha256: None,
        uploaded_source_path: false,
    };
    let mut staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one model file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("model.safetensors"));
                let path =
                    write_model_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }

            let value = field
                .text()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match field_name.as_str() {
                "modelId" => payload.model_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "type" => payload.model_type = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "repo" => payload.repo = Some(value.to_owned()),
                "sourceUrl" => payload.source_url = Some(value.to_owned()),
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = parse_result {
        if let Some(path) = staged_path.as_deref() {
            cleanup_staged_model_upload(path).await;
        }
        return Err(error);
    }

    let Some(staged_path) = staged_path else {
        return Err(ApiError::bad_request("Upload file field is required"));
    };
    Ok((payload, staged_path))
}

pub(crate) async fn write_model_upload_field_to_staged_file(
    state: &AppState,
    field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("model-uploads")
        .join(format!("upload-{}", Uuid::new_v4().simple()));
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(filename);
    // sc-8886 (F-084): shared streaming writer. Cleanup removes the staged file AND its
    // per-upload parent directory.
    stream_multipart_field_to_file(
        field,
        &temp_path,
        max_model_upload_bytes(),
        || {
            format!(
                "Uploaded model file exceeds the {} limit",
                format_bytes(max_model_upload_bytes() as u64)
            )
        },
        || cleanup_staged_model_upload(&temp_path),
    )
    .await?;
    Ok(temp_path)
}

pub(crate) async fn cleanup_staged_model_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

pub(crate) fn model_import_source_provider(payload: &ModelImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

pub(crate) fn model_family_inspection_error(error: SafetensorsHeaderError) -> ApiError {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect model file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("Model file has an invalid safetensors header".to_owned())
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => {
            ApiError::bad_request(format!(
            "Model file is incomplete or corrupt ({actual} bytes on disk, but its header declares \
             at least {declared} bytes of tensor data); the file was likely truncated during \
             download. Re-download the complete file."
        ))
        }
    }
}

/// Applies the import-time policy for base models: confident detection rejects
/// a mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection accepts the supplied family
/// unchanged (and leaves things unset if none was supplied).
pub(crate) fn reconcile_model_family(
    supplied: Option<String>,
    detected: Option<String>,
    _context: &str,
) -> Result<Option<String>, ApiError> {
    reconcile_detected_family(supplied, detected).map_err(|mismatch| {
        ApiError::bad_request(format!(
            "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
            mismatch.detected, mismatch.supplied, mismatch.detected
        ))
    })
}

pub(crate) fn max_model_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_MODEL_UPLOAD_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        if limit > 0 {
            return limit;
        }
    }
    MAX_MODEL_UPLOAD_BYTES
}

/// Catalog without live Hugging Face size estimation: download sizes fall back to
/// manifest metadata only. This is the right call for job validation, LoRA/preset
/// CRUD, download/convert job creation, and delete — none of which read the
/// byte-accurate download size — so an unreachable huggingface.co can't stall
/// those paths (sc-4169).
pub(crate) async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    model_catalog_inner(state, false).await
}

/// Catalog with live Hugging Face download-size estimates (negative-cached on
/// failure). Reserved for `GET /models`, the one surface that displays
/// download sizes.
pub(crate) async fn model_catalog_sized(state: &AppState) -> Result<Vec<Value>, ApiError> {
    model_catalog_inner(state, true).await
}

// sc-4205 (F-API-12): the per-model install/cache state, formerly threaded through a
// 5-tuple that was easy to mis-order. Named fields make the catalog loop legible.
struct ModelCatalogEntryState {
    downloadable: bool,
    installed_path: Option<String>,
    installed: bool,
    cache_incomplete: bool,
    missing_required_files: Vec<String>,
    update_available: bool,
}

#[derive(Debug, PartialEq)]
struct ReceiptFileSet {
    files: Vec<String>,
    revision: Option<String>,
}

fn receipt_file_sets(managed_path: &FsPath, repo: &str) -> Vec<ReceiptFileSet> {
    let Ok(bytes) = std::fs::read(managed_path.join(".sceneworks-download-complete.json")) else {
        return Vec::new();
    };
    let Ok(receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return Vec::new();
    };
    let entries = receipt
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![receipt]);
    entries
        .into_iter()
        .filter_map(|entry| {
            if entry.get("repo").and_then(Value::as_str) != Some(repo) {
                return None;
            }
            let files = entry
                .get("resolvedFiles")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let revision = entry
                .get("snapshotRevision")
                .and_then(Value::as_str)
                .map(str::to_owned);
            (!files.is_empty()).then_some(ReceiptFileSet { files, revision })
        })
        .collect()
}

/// Whether a snapshot the receipt's files resolve into is an actually LOADABLE install (not merely a
/// set of files that exist). A backfill (sc-13076) records whatever was on disk, so an interrupted
/// download left a torn tier — its `model_index.json` + a stray config present, but the
/// transformer/vae weights missing — whose recorded files all exist yet cannot load. When the
/// receipt/tier files form a diffusers tier subdir (`["<tier>/*"]`, or a receipt whose files share one
/// leading dir), require that subdir to pass the same per-component weight check the cache-health path
/// uses. A non-diffusers set (no `<tier>/model_index.json`, or a flat single-variant filter) keeps the
/// prior file-existence contract.
fn snapshot_tier_is_loadable(snapshot: &FsPath, files: &[String]) -> bool {
    match tier_subdir_name(files) {
        Some(tier) => {
            let tier_dir = snapshot.join(&tier);
            !path_is_readable_file(&tier_dir.join("model_index.json"))
                || diffusers_snapshot_health(&tier_dir).installed
        }
        None => true,
    }
}

fn receipt_files_present(data_dir: &FsPath, repo: &str, receipt: &ReceiptFileSet) -> bool {
    !receipt.files.is_empty()
        && huggingface_repo_cache_path(data_dir, repo)
            .map(|root| {
                let matches = crate::huggingface_snapshot_dirs(&root)
                    .into_iter()
                    .filter(|snapshot| {
                        receipt
                            .files
                            .iter()
                            .all(|file| snapshot.join(file).is_file())
                    })
                    // A torn tier's recorded files all exist but the install can't load — it must not
                    // count as a "usable stale" install that keeps the model falsely installed.
                    .filter(|snapshot| snapshot_tier_is_loadable(snapshot, &receipt.files))
                    .collect::<Vec<_>>();
                receipt
                    .revision
                    .as_deref()
                    .map_or(matches.len() == 1, |revision| {
                        matches.iter().any(|snapshot| {
                            snapshot.file_name().and_then(|v| v.to_str()) == Some(revision)
                        })
                    })
            })
            .unwrap_or(false)
}

fn backfill_current_receipt(
    managed_path: &FsPath,
    model: &Value,
    context: &DownloadContext,
    data_dir: &FsPath,
) {
    if !receipt_file_sets(managed_path, &context.repo).is_empty() {
        return;
    }
    let receipts = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .filter_map(|entry| {
            let repo = entry.get("repo")?.as_str()?;
            let files = string_array_field(entry, "files");
            let root = huggingface_repo_cache_path(data_dir, repo)?;
            let snapshot = crate::huggingface_snapshot_dirs(&root).into_iter().find(|snapshot| {
                files.iter().all(|pattern| snapshot_contains_pattern(snapshot, pattern))
            })?;
            // Never manufacture a receipt for a torn tier: a `<tier>/*` glob matches as soon as one
            // metadata file exists, so backfilling it would record a "complete" install that cannot
            // load. Require the tier to actually hold its weights before preserving it (sc-13076).
            if !snapshot_tier_is_loadable(&snapshot, &files) {
                return None;
            }
            let resolved = snapshot_files(&snapshot).into_iter()
                .filter(|file| allow_pattern_matches(file, &files)).collect::<Vec<_>>();
            (!resolved.is_empty()).then(|| json!({
                "schemaVersion": 2, "repo": repo,
                "modelId": model.get("id").cloned().unwrap_or(Value::Null),
                "variant": entry.get("variant").cloned().unwrap_or_else(|| Value::String("default".to_owned())),
                "manifestFiles": files, "resolvedFiles": resolved, "backfilled": true,
            }))
        }).collect::<Vec<_>>();
    if receipts.is_empty() {
        return;
    }
    let mut receipt = receipts[0].clone();
    receipt
        .as_object_mut()
        .unwrap()
        .insert("receipts".to_owned(), Value::Array(receipts));
    let _ = std::fs::create_dir_all(managed_path);
    let _ = serde_json::to_vec_pretty(&receipt).ok().and_then(|bytes| {
        std::fs::write(
            managed_path.join(".sceneworks-download-complete.json"),
            bytes,
        )
        .ok()
    });
}

#[cfg(test)]
mod download_receipt_tests {
    use super::*;

    #[test]
    fn multi_repo_marker_filters_nested_receipts_by_requested_repo() {
        let temp = tempfile::tempdir().unwrap();
        let managed = temp.path().join("models/owner--primary");
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(managed.join(".sceneworks-download-complete.json"), serde_json::to_vec(&json!({
            "repo": "owner/corequisite",
            "receipts": [
                {"repo":"owner/primary", "resolvedFiles":["model.safetensors"], "snapshotRevision":"primary-rev"},
                {"repo":"owner/corequisite", "resolvedFiles":["encoder.safetensors"], "snapshotRevision":"dependency-rev"}
            ]
        })).unwrap()).unwrap();

        let primary = receipt_file_sets(&managed, "owner/primary");
        assert_eq!(
            primary,
            vec![ReceiptFileSet {
                files: vec!["model.safetensors".to_owned()],
                revision: Some("primary-rev".to_owned())
            }]
        );
        let dependency = receipt_file_sets(&managed, "owner/corequisite");
        assert_eq!(
            dependency,
            vec![ReceiptFileSet {
                files: vec!["encoder.safetensors".to_owned()],
                revision: Some("dependency-rev".to_owned())
            }]
        );
    }

    #[test]
    fn complete_pre_receipt_install_is_backfilled_and_protected_after_rename() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/backfill";
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .unwrap()
            .join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
        let original = json!({"id":"backfill-model", "downloads":[{"provider":"huggingface", "repo":repo, "files":["old.safetensors"]}]});

        let initial = install_state_for(
            model_download_context(&original).unwrap(),
            &original,
            data_dir,
        );
        assert!(initial.installed);
        let marker = data_dir
            .join("models")
            .join(safe_download_dir(repo))
            .join(".sceneworks-download-complete.json");
        let receipt: Value = serde_json::from_slice(&std::fs::read(marker).unwrap()).unwrap();
        assert_eq!(receipt["resolvedFiles"], json!(["old.safetensors"]));
        assert_eq!(receipt["backfilled"], true);

        let renamed = json!({"id":"backfill-model", "downloads":[{"provider":"huggingface", "repo":repo, "files":["new.safetensors"]}]});
        let protected = install_state_for(
            model_download_context(&renamed).unwrap(),
            &renamed,
            data_dir,
        );
        assert!(
            protected.installed,
            "backfilled exact old set remains usable"
        );
        assert!(protected.update_available, "rename is offered as an update");
    }

    #[test]
    fn receipt_remains_usable_when_current_manifest_file_changes() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/model";
        let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
        let snapshot = cache.join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
        let managed = data_dir.join("models/owner--model");
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "schemaVersion": 2, "repo": repo,
                "resolvedFiles": ["old.safetensors"]
            }))
            .unwrap(),
        )
        .unwrap();

        let files = receipt_file_sets(&managed, repo);
        assert_eq!(files[0].files, vec!["old.safetensors".to_owned()]);
        assert!(receipt_files_present(data_dir, repo, &files[0]));
        assert!(!huggingface_cache_health(&cache, &["new.safetensors".to_owned()]).installed);

        let ambiguous = cache.join("snapshots/rev-b");
        std::fs::create_dir_all(&ambiguous).unwrap();
        std::fs::write(ambiguous.join("old.safetensors"), b"other weights").unwrap();
        assert!(
            !receipt_files_present(data_dir, repo, &files[0]),
            "legacy receipt must identify one snapshot"
        );

        std::fs::remove_file(snapshot.join("old.safetensors")).unwrap();
        std::fs::remove_file(ambiguous.join("old.safetensors")).unwrap();
        assert!(
            !receipt_files_present(data_dir, repo, &files[0]),
            "torn stale install is missing"
        );
    }

    #[test]
    fn catalog_distinguishes_usable_stale_from_torn_and_points_at_cache() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/model";
        let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
        let snapshot = cache.join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "schemaVersion": 2, "repo": repo, "resolvedFiles": ["old.safetensors"]
            }))
            .unwrap(),
        )
        .unwrap();
        let model = json!({"id":"model", "downloads":[{
            "provider":"huggingface", "repo":repo, "files":["new.safetensors"]
        }]});
        let context = model_download_context(&model).unwrap().unwrap();
        let stale = install_state_for(Some(context), &model, data_dir);
        assert!(stale.installed);
        assert!(stale.update_available);
        assert_eq!(
            stale.installed_path.as_deref(),
            Some(cache.to_string_lossy().as_ref())
        );

        std::fs::remove_file(snapshot.join("old.safetensors")).unwrap();
        let torn = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(!torn.installed);
        assert!(!torn.update_available);
    }

    #[test]
    fn breaking_and_corequisite_softness_matrix() {
        for breaking in [false, true] {
            for soft in [false, true] {
                let temp = tempfile::tempdir().unwrap();
                let data_dir = temp.path();
                let repo = "owner/model";
                let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
                let snapshot = cache.join("snapshots/rev-a");
                std::fs::create_dir_all(&snapshot).unwrap();
                std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
                let managed = data_dir.join("models").join(safe_download_dir(repo));
                std::fs::create_dir_all(&managed).unwrap();
                std::fs::write(
                    managed.join(".sceneworks-download-complete.json"),
                    serde_json::to_vec(&json!({
                        "schemaVersion": 2, "repo": repo,
                        "resolvedFiles": ["old.safetensors"]
                    }))
                    .unwrap(),
                )
                .unwrap();
                let model = json!({
                    "id": "model",
                    "downloads": [
                        {"provider":"huggingface", "repo":repo,
                         "files":["new.safetensors"], "breaking":breaking},
                        {"provider":"huggingface", "repo":"owner/dependency",
                         "coRequisite":true, "required": if soft { "soft" } else { "hard" }}
                    ]
                });
                let state =
                    install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
                assert_eq!(
                    state.installed,
                    !breaking && soft,
                    "breaking={breaking}, soft={soft}"
                );
                assert!(
                    state.update_available,
                    "every stale/soft combination offers an update"
                );
                if !soft {
                    assert!(state
                        .missing_required_files
                        .iter()
                        .any(|file| file.contains("owner/dependency")));
                }
                if !breaking && !soft {
                    let mut omitted = model.clone();
                    omitted.as_object_mut().unwrap()["downloads"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("breaking");
                    omitted.as_object_mut().unwrap()["downloads"][1]
                        .as_object_mut()
                        .unwrap()
                        .remove("required");
                    let defaulted = install_state_for(
                        model_download_context(&omitted).unwrap(),
                        &omitted,
                        data_dir,
                    );
                    assert!(!defaulted.installed, "omitted required defaults to hard");
                    assert!(
                        defaulted.update_available,
                        "omitted breaking defaults to false"
                    );
                }
            }
        }
    }
}

// Resolve a model's install/cache state from its (optional) download source. A
// downloadable model checks the HF cache + the SceneWorks-managed dir; a non-download
// model (a local manifest entry) checks its declared installed path; otherwise it's
// simply absent.
fn install_state_for(
    download_context: Option<DownloadContext>,
    model: &Value,
    data_dir: &FsPath,
) -> ModelCatalogEntryState {
    if let Some(download_context) = download_context {
        let managed_path = data_dir
            .join("models")
            .join(safe_download_dir(&download_context.repo));
        let cache_path = huggingface_repo_cache_path(data_dir, &download_context.repo);
        // Quant-matrix models (sc-8506/8508): the top-level install state aggregates across ALL
        // selectable tiers, not just the default one. A model that offers bf16/q8/q4 counts as
        // installed when ANY tier is fully present, and is only "incomplete" when a tier is genuinely
        // torn (partially downloaded) AND no complete tier exists. Installing a single valid tier —
        // even a non-default one — must never surface as an incomplete/repairable cache (sc-9907),
        // because that rendered a false "Cached files are incomplete" warning + Fix button on a
        // perfectly good install. Single-variant models keep the default-tier contract below.
        let (cache_installed, cache_incomplete, mut missing_required_files) =
            if model_has_variant_matrix(model) {
                let variants = model_variant_states(model, data_dir);
                let any_installed = variants.iter().any(|variant| variant.installed);
                // Only a torn tier (some-but-not-all files present) with no complete sibling is a real
                // repair candidate; a validly absent tier is "missing", not "incomplete".
                let torn = variants
                    .iter()
                    .find(|variant| variant.cache_incomplete && !variant.installed);
                let incomplete = !any_installed && torn.is_some();
                let missing = if any_installed {
                    Vec::new()
                } else {
                    torn.map(|variant| variant.missing_required_files.clone())
                        .unwrap_or_default()
                };
                (any_installed, incomplete, missing)
            } else {
                let cache_health = cache_path
                    .as_ref()
                    .map(|path| huggingface_cache_health(path, &download_context.files));
                let installed = cache_health.as_ref().is_some_and(|health| health.installed);
                let incomplete = cache_health
                    .as_ref()
                    .is_some_and(|health| health.incomplete);
                let missing = cache_health
                    .as_ref()
                    .map(|health| health.missing_files.clone())
                    .unwrap_or_default();
                (installed, incomplete, missing)
            };
        // A quant-matrix model's top-level state is the aggregate of its tier-aware variant states
        // computed above (cache_installed = "any tier installed"). The repo-level managed marker must
        // NOT independently mark it installed (sc-9909): a stale .sceneworks-download-complete.json
        // left by an empty download would otherwise read the whole model as installed while every tier
        // reads missing. Single-variant models keep the repo-level managed contract.
        let receipt_file_sets = receipt_file_sets(&managed_path, &download_context.repo);
        let managed_installed = !model_has_variant_matrix(model)
            && receipt_file_sets.is_empty()
            && model_is_installed(&managed_path);
        if cache_installed {
            backfill_current_receipt(&managed_path, model, &download_context, data_dir);
        }
        let stale_files_present = !cache_installed
            && receipt_file_sets
                .iter()
                .any(|receipt| receipt_files_present(data_dir, &download_context.repo, receipt));
        let breaking_update = model
            .get("breaking")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || model
                .get("downloads")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|entry| {
                    !is_co_requisite_download(entry)
                        && entry.get("repo").and_then(Value::as_str)
                            == Some(download_context.repo.as_str())
                        && string_array_field(entry, "files") == download_context.files
                        && entry
                            .get("breaking")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                });
        let usable_stale = stale_files_present && !breaking_update;
        let primary_installed = managed_installed || cache_installed || usable_stale;
        let installed_path = if cache_installed || cache_incomplete || usable_stale {
            cache_path.clone()
        } else {
            Some(managed_path)
        };
        // Co-requisites (sc-9696): the entry counts as installed only when the primary AND every
        // co-requisite dependency (e.g. the PiD decoder's shared gemma-2-2b-it caption encoder) are
        // cached. Gating on this keeps a feature that silently no-ops without its dependency (PiD →
        // native VAE) from advertising as ready, and a present primary with a missing/partial
        // co-requisite surfaces as a repairable partial install (cache_incomplete → repairAvailable),
        // whose repair re-runs the download job that now fetches the co-requisite too.
        let mut hard_co_requisites_installed = true;
        let mut soft_co_requisite_update = false;
        let mut co_requisite_incomplete = false;
        for co_requisite in model_co_requisite_downloads(model) {
            let Some(repo) = co_requisite.get("repo").and_then(Value::as_str) else {
                continue;
            };
            let files = string_array_field(&co_requisite, "files");
            let health = huggingface_repo_cache_path(data_dir, repo)
                .map(|path| huggingface_cache_health(&path, &files));
            if health.as_ref().is_some_and(|health| health.installed) {
                continue;
            }
            let soft = co_requisite.get("required").and_then(Value::as_str) == Some("soft");
            if soft {
                soft_co_requisite_update = true;
            } else {
                hard_co_requisites_installed = false;
                co_requisite_incomplete |= health.as_ref().is_some_and(|health| health.incomplete);
            }
            match health
                .as_ref()
                .map(|health| health.missing_files.as_slice())
            {
                Some(missing) if !missing.is_empty() && !soft => missing_required_files
                    .extend(missing.iter().map(|file| format!("{repo}/{file}"))),
                _ if !soft => missing_required_files.push(repo.to_owned()),
                _ => {}
            }
        }
        ModelCatalogEntryState {
            downloadable: true,
            installed_path: installed_path.map(|path| path.display().to_string()),
            installed: primary_installed && hard_co_requisites_installed,
            cache_incomplete: cache_incomplete
                || (primary_installed && !hard_co_requisites_installed)
                || co_requisite_incomplete,
            missing_required_files,
            update_available: stale_files_present || soft_co_requisite_update,
        }
    } else if let Some(installed_path) = model_manifest_installed_path(model, data_dir) {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: Some(installed_path.display().to_string()),
            installed: model_is_installed(&installed_path),
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
        }
    } else {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: None,
            installed: false,
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
        }
    }
}

// Per-variant install state (sc-8508, epic 8506): a single downloadable tier of a quant-matrix
// model. `install_state_for` reports the DEFAULT variant's install state (back-compat single-variant
// contract); `model_variants` reports one of these per declared download entry so the catalog knows
// WHICH tiers are on disk, not just whether *a* variant is.
struct ModelVariantState {
    /// The tier key: an explicit `downloads[].variant` (bf16/q8/q4), else "default" for a
    /// single-variant model (which has exactly one entry).
    variant: String,
    /// Whether this specific tier's files are present in the HF cache.
    installed: bool,
    /// Resolved install path for this tier (the shared repo cache root; tiers live as `files`-
    /// filtered subdirs within it). `None` when the repo has never been fetched.
    installed_path: Option<String>,
    /// This tier's incomplete-cache signal (some but not all `files` present).
    cache_incomplete: bool,
    /// Files this tier is missing from the cache (empty when complete or absent).
    missing_required_files: Vec<String>,
    /// This tier's estimated download size (from `downloads[].estimatedSizeBytes` /
    /// `footprint.diskSizeBytes`).
    download_size_bytes: Option<u64>,
    /// The raw `downloads[].footprint` object (disk size + optional measured memory), passed
    /// through verbatim for the RAM-suggestion surfaces (sc-8509/8516). `Null` when absent.
    footprint: Value,
}

// Whether `model`'s `downloads` array is a quant-matrix — i.e. at least one supported entry carries
// an explicit non-empty `variant` key (q4/q8/bf16). Entry COUNT is deliberately NOT a discriminator:
// the manifest uses multiple download entries for non-tier reasons too (alternate sources, native
// fallbacks, co-requisite TE repos — e.g. PiD backbone + gemma-2-2b-it, boogu mlx+native, krea, wan,
// ltx). Those are not quant matrices, so only variant-presence flags a model as one (sc-8508).
fn model_has_variant_matrix(model: &Value) -> bool {
    let Some(downloads) = model.get("downloads").and_then(Value::as_array) else {
        return false;
    };
    downloads
        .iter()
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .any(|entry| {
            entry
                .get("variant")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
        })
}

// Build the per-variant install state for every supported download entry. A single-variant model
// yields one entry keyed "default"; a quant-matrix model yields one per tier. Each entry's install
// state is probed independently against the HF cache using that tier's own `files` filter, so the
// catalog can advertise (e.g.) bf16 installed while q4 is missing.
fn model_variant_states(model: &Value, data_dir: &FsPath) -> Vec<ModelVariantState> {
    let Some(downloads) = model.get("downloads").and_then(Value::as_array) else {
        return Vec::new();
    };
    // Emitted variant keys must be unique: the single-variant case is exactly one "default", and a
    // quant matrix has one entry per distinct q4/q8/bf16 tier. Guard against a manifest that maps two
    // supported entries to the same key (unlabeled alternate sources both collapsing to "default", or
    // a duplicated `variant`) — keep the first, drop the rest, so downstream per-variant tracking
    // never emits two same-keyed states (sc-8508).
    let mut seen_variants = std::collections::HashSet::new();
    downloads
        .iter()
        // Co-requisites (sc-9696) are dependencies, not selectable tiers — never a variant state.
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .filter(|entry| {
            let key = entry
                .get("variant")
                .and_then(Value::as_str)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "default".to_owned());
            seen_variants.insert(key)
        })
        .map(|entry| {
            let repo = entry
                .get("repo")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let files = string_array_field(entry, "files");
            let cache_path = huggingface_repo_cache_path(data_dir, &repo);
            let cache_health = cache_path
                .as_ref()
                .map(|path| huggingface_cache_health(path, &files));
            let installed = cache_health.as_ref().is_some_and(|health| health.installed);
            let cache_incomplete = cache_health
                .as_ref()
                .is_some_and(|health| health.incomplete);
            let missing_required_files = cache_health
                .as_ref()
                .map(|health| health.missing_files.clone())
                .unwrap_or_default();
            // The managed dir mirrors the default-download install path; a variant present there (a
            // directly-downloaded turnkey) counts as installed too — but the check must be TIER-aware.
            // A quant-matrix repo writes ONE repo-level completion marker no matter which tier was
            // fetched, so keying a per-tier "installed" on the bare marker made EVERY tier report
            // installed after any single tier's download (sc-9909). Require the tier's own files to
            // actually exist under the managed dir, not just the marker.
            let managed_path = data_dir.join("models").join(safe_download_dir(&repo));
            let managed_installed = managed_tier_installed(&managed_path, &files);
            let installed_path = if installed || cache_incomplete {
                cache_path
            } else if managed_installed {
                Some(managed_path)
            } else {
                None
            };
            ModelVariantState {
                variant: entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "default".to_owned()),
                installed: installed || managed_installed,
                installed_path: installed_path.map(|path| path.display().to_string()),
                cache_incomplete,
                missing_required_files,
                download_size_bytes: manifest_download_size_bytes(model, entry)
                    .or_else(|| variant_footprint_disk_bytes(entry)),
                footprint: entry.get("footprint").cloned().unwrap_or(Value::Null),
            }
        })
        .collect()
}

// Whether a tier's OWN artifacts live in the app-managed turnkey dir (data/models/<repo>), as opposed
// to the shared HF cache. The repo-level completion marker (.sceneworks-download-complete.json) alone
// does NOT certify a tier: a quant-matrix repo writes exactly one marker regardless of which tier was
// downloaded, so a bare-marker check reported every tier of a repo installed after any single tier's
// fetch (sc-9909). Require BOTH the marker AND — for a tier that declares a `files` filter — that the
// tier's files actually exist under the managed dir. A single-variant turnkey (empty `files`) is
// certified by the marker alone, preserving the pre-matrix contract.
fn managed_tier_installed(managed_path: &FsPath, files: &[String]) -> bool {
    if !model_is_installed(managed_path) {
        return false;
    }
    files.is_empty()
        || files
            .iter()
            .all(|pattern| snapshot_contains_pattern(managed_path, pattern))
}

// The on-disk size a `downloads[].footprint.diskSizeBytes` declares, if any — the tier-scoped
// footprint signal (sc-8508) used as a fallback size when `estimatedSizeBytes` is absent.
fn variant_footprint_disk_bytes(download: &Value) -> Option<u64> {
    download
        .get("footprint")
        .and_then(|footprint| footprint.get("diskSizeBytes"))
        .and_then(json_size_to_u64)
}

// Gated-model signal (sc-1898): a machine-readable `gated` flag plus the credential
// host the download requires, so the Models screen can route the user to the
// credential screen before a download will succeed. The host honors an explicit
// manifest `credentialHost` and otherwise derives from the download provider/source
// URL; `licenseUrl` passes through untouched.
fn apply_gating_fields(object: &mut JsonObject) {
    let gated = object
        .get("gated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    object.insert("gated".to_owned(), Value::Bool(gated));
    if gated {
        let credential_host = object
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
            .or_else(|| derive_credential_host(object));
        object.insert(
            "credentialHost".to_owned(),
            credential_host.map(Value::String).unwrap_or(Value::Null),
        );
    }
}

// Mac UI gating (sc-3486): per-model Rust/MLX support so the web client can hide/
// disable a torch-only model in the pickers, plus (macOS only) the MLX availability +
// conversion status for models that declare an `mlx` variant. Additive fields the
// web/Docker build ignores; the client only acts on macSupport when the capabilities
// endpoint reports `macGatingActive`, so non-Mac pickers are untouched.
// Per-variant catalog fields (sc-8508): emit a `variants` array — one object per declared quant
// tier — plus a `hasVariantMatrix` boolean the web uses to decide whether to render a tier picker.
// A single-variant model gets a one-element array keyed "default" that mirrors the top-level
// install state, so the existing single-variant contract is unchanged.
fn apply_variant_fields(object: &mut JsonObject, data_dir: &FsPath) {
    let model = Value::Object(object.clone());
    let has_matrix = model_has_variant_matrix(&model);
    let variants = model_variant_states(&model, data_dir)
        .into_iter()
        .map(|variant| {
            json!({
                "variant": variant.variant,
                "installed": variant.installed,
                "installState": if variant.installed { "installed" } else { "missing" },
                "cacheState": if variant.cache_incomplete {
                    "incomplete"
                } else if variant.installed {
                    "complete"
                } else {
                    "missing"
                },
                "installedPath": variant
                    .installed_path
                    .map(Value::String)
                    .unwrap_or(Value::Null),
                "missingRequiredFiles": variant.missing_required_files,
                "downloadSizeBytes": variant
                    .download_size_bytes
                    .map(|value| json!(value))
                    .unwrap_or(Value::Null),
                "footprint": variant.footprint,
            })
        })
        .collect::<Vec<_>>();
    object.insert("hasVariantMatrix".to_owned(), Value::Bool(has_matrix));
    object.insert("variants".to_owned(), Value::Array(variants));
}

/// The convert-output quant tiers present under a converted MLX dir (sc-10730), highest-fidelity first.
/// Convert-at-install models (e.g. Anima) write `<converted>/<tier>/<backbone>/…` for each of
/// bf16/q8/q4 in ONE convert job — the tiers are convert OUTPUTS, not per-tier downloads. This lets the
/// Studio offer a generation-time tier picker via the decoupled `mlxTiers` catalog field WITHOUT the
/// download variant-matrix (`hasVariantMatrix`), whose `QuantDownloadPanel` would render bogus per-tier
/// download buttons for a model that has no per-tier download. Empty for a flat converted dir (no tier
/// subdirs) → the web renders no picker. Mirrors the worker tier resolvers' "tier present" probe so the
/// catalog and `anima_tier_subdir` agree on which tiers are loadable.
fn mlx_convert_output_tiers(converted_dir: &FsPath) -> Vec<&'static str> {
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| tier_subdir_has_weights(&converted_dir.join(tier)))
        .collect()
}

/// Whether a converted tier subdir holds loadable weights: a non-hidden `.safetensors` / `.index.json`
/// under a known backbone dir (`diffusion_models/` for Anima's Cosmos DiT, `transformer/` for other
/// DiTs, `unet/` for SDXL) or flat in the tier dir. A hidden `._*` AppleDouble sidecar is not a weight
/// (SceneWorks#1333), mirroring the worker resolvers.
fn tier_subdir_has_weights(tier_dir: &FsPath) -> bool {
    if !tier_dir.is_dir() {
        return false;
    }
    let dir_has_weight = |dir: &FsPath| -> bool {
        std::fs::read_dir(dir).is_ok_and(|entries| {
            entries.flatten().any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                !name.starts_with("._")
                    && (name.ends_with(".safetensors") || name.ends_with(".index.json"))
            })
        })
    };
    // A known backbone subdir (`diffusion_models/` Anima, `transformer/` DiTs, `unet/` SDXL), or flat
    // in the tier dir itself.
    dir_has_weight(tier_dir)
        || ["diffusion_models", "transformer", "unet"]
            .into_iter()
            .any(|sub| dir_has_weight(&tier_dir.join(sub)))
}

fn apply_mac_and_mlx_fields(object: &mut JsonObject, data_dir: &FsPath) {
    // Per-model quality FLOOR (sc-10731, epic 10721): surface the manifest `mlx.minQualityTier` as a
    // top-level `minQualityTier` so the web `defaultTierSelection` can clamp the DEFAULT generation tier
    // UP to it (a floored model — Anima base/aesthetic = q8 — never lets a low global "default quality"
    // setting land the default on the washed q4). An EXPLICIT picker pick below the floor is still
    // honored, with a non-blocking advisory. Decoupled top-level field, mirroring `mlxTiers` — the web
    // reads one stable key rather than reaching into the passed-through `mlx` sub-object. Emitted on every
    // platform where the manifest declares it (the picker only renders where >1 tier installs, but the
    // field is cheap and lets any surface read the floor). Only bf16/q8/q4 are valid; others are dropped.
    if let Some(floor) = object
        .get("mlx")
        .and_then(|mlx| mlx.get("minQualityTier"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|tier| matches!(*tier, "bf16" | "q8" | "q4"))
        .map(str::to_owned)
    {
        object.insert("minQualityTier".to_owned(), Value::String(floor));
    }
    let mac_support = {
        let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
        let model_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        model_mac_support(id, model_type)
    };
    if let Ok(mac_support) = serde_json::to_value(mac_support) {
        object.insert("macSupport".to_owned(), mac_support);
    }
    let mlx_status = if cfg!(target_os = "macos") {
        mlx_catalog_status(object, data_dir)
    } else {
        None
    };
    if let Some(status) = mlx_status {
        // Generation-time tier picker for convert-at-install models (sc-10730): surface the on-disk
        // convert-output tiers as `mlxTiers`, DECOUPLED from `hasVariantMatrix` so the Models download
        // panel is untouched. Only when the model is actually converted (its tier subdirs exist).
        if let Some(converted) = status.converted_path.as_deref() {
            let tiers = mlx_convert_output_tiers(converted);
            if !tiers.is_empty() {
                object.insert("mlxTiers".to_owned(), json!(tiers));
            }
        }
        object.insert(
            "mlxInstallState".to_owned(),
            Value::String(status.install_state.to_owned()),
        );
        object.insert(
            "mlxConversionState".to_owned(),
            Value::String(status.conversion_state.to_owned()),
        );
        object.insert(
            "mlxConvertedPath".to_owned(),
            status
                .converted_path
                .map(|path| Value::String(path.display().to_string()))
                .unwrap_or(Value::Null),
        );
        object.insert(
            "updateAvailable".to_owned(),
            Value::Bool(
                status.update_available
                    || object
                        .get("updateAvailable")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
            ),
        );
    }
}

async fn model_catalog_inner(
    state: &AppState,
    estimate_sizes: bool,
) -> Result<Vec<Value>, ApiError> {
    // sc-8819 (F-017): observe full-catalog assembly (the per-model FS install-state probe
    // sweep) so a test can assert it runs once per job-create.
    #[cfg(test)]
    crate::test_note_model_catalog_build();
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    let mut models = merge_entries_by_id(builtin, user);
    // Resolve per-platform download sources before computing install state/size: some video models
    // carry both a native MLX-convert checkpoint (macOS) and a diffusers/torch checkpoint
    // (Windows/Linux). Keep only the entries applicable to this OS so the download job, status,
    // size, and the frontend all agree on the right repo (sc-3240).
    for model in &mut models {
        retain_downloads_for_os(model, std::env::consts::OS);
    }
    let download_contexts = models
        .iter()
        .map(model_download_context)
        .collect::<Result<Vec<_>, _>>()?;
    let download_size_bytes = join_all(download_contexts.iter().map(|context| async move {
        match context {
            Some(context) if estimate_sizes => {
                estimate_huggingface_download_size(state, &context.repo, &context.files).await
            }
            _ => None,
        }
    }))
    .await;

    let data_dir = state.settings.data_dir.clone();
    // sc-10667: surface assembled external ComfyUI base models alongside the manifest
    // catalog. Cloned before the blocking closure, mirroring the external-LoRA merge in
    // `lora_catalog`; the scan is filesystem-bound and runs on the blocking pool below.
    let external_roots = state.settings.external_model_roots.clone();
    let external_base_cache = state.external_base_model_cache.clone();
    // sc-4202 (F-API-3): the per-model install-state probes below hit the filesystem
    // (huggingface_cache_health snapshot walks, model_is_installed, mlx_catalog_status)
    // for every model. Assemble the catalog on the blocking pool so these synchronous
    // walks don't stall a tokio worker thread under load or on a slow/network volume.
    let models = tokio::task::spawn_blocking(move || -> Result<Vec<Value>, ApiError> {
        for (model, (download_context, download_size_bytes)) in models
            .iter_mut()
            .zip(download_contexts.into_iter().zip(download_size_bytes))
        {
            let fallback_size_bytes = download_context
                .as_ref()
                .and_then(|context| context.fallback_size_bytes);
            let primary_size_bytes = download_size_bytes.or(fallback_size_bytes);
            let download_size_estimated =
                download_size_bytes.is_none() && fallback_size_bytes.is_some();
            // Co-requisites (sc-9696) install alongside the primary, so the displayed footprint must
            // include them (e.g. PiD's ~2.7 GB checkpoint + ~5.2 GB gemma-2-2b-it). Their sizes come
            // from the manifest (the live HF estimate above only sizes the primary repo).
            let co_requisite_size_bytes: u64 = model_co_requisite_downloads(model)
                .iter()
                .filter_map(|download| manifest_download_size_bytes(model, download))
                .sum();
            let effective_download_size_bytes = match primary_size_bytes {
                Some(primary) => Some(primary + co_requisite_size_bytes),
                None if co_requisite_size_bytes > 0 => Some(co_requisite_size_bytes),
                None => None,
            };
            let state = install_state_for(download_context, model, &data_dir);
            let object = model
                .as_object_mut()
                .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
            let model_id = object.get("id").and_then(Value::as_str).unwrap_or_default();
            let user_managed = user_model_ids.contains(model_id);
            object.insert(
                "catalogScope".to_owned(),
                Value::String(if user_managed { "user" } else { "builtin" }.to_owned()),
            );
            object.insert("downloadable".to_owned(), Value::Bool(state.downloadable));
            object.insert(
                "downloadSizeBytes".to_owned(),
                effective_download_size_bytes
                    .map(|value| json!(value))
                    .unwrap_or(Value::Null),
            );
            object.insert(
                "downloadSizeLabel".to_owned(),
                effective_download_size_bytes
                    .map(format_bytes)
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );
            object.insert(
                "downloadSizeEstimated".to_owned(),
                Value::Bool(download_size_estimated),
            );
            object.insert(
                "installState".to_owned(),
                Value::String(
                    if state.installed {
                        "installed"
                    } else {
                        "missing"
                    }
                    .to_owned(),
                ),
            );
            object.insert(
                "cacheState".to_owned(),
                Value::String(
                    if state.cache_incomplete {
                        "incomplete"
                    } else if state.installed {
                        "complete"
                    } else {
                        "missing"
                    }
                    .to_owned(),
                ),
            );
            object.insert(
                "missingRequiredFiles".to_owned(),
                Value::Array(
                    state
                        .missing_required_files
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                ),
            );
            object.insert(
                "repairAvailable".to_owned(),
                Value::Bool(state.downloadable && state.cache_incomplete),
            );
            object.insert(
                "updateAvailable".to_owned(),
                Value::Bool(state.update_available),
            );
            object.insert(
                "installedPath".to_owned(),
                state
                    .installed_path
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            );
            object.insert(
                "removable".to_owned(),
                Value::Bool(user_managed || state.installed),
            );
            // Per-variant install tracking (sc-8508, epic 8506): one entry per declared quant tier,
            // each with its own installed flag + path + size + footprint. A single-variant model
            // still emits exactly one "default" variant, so the array is a superset of the
            // (retained) top-level installState/installedPath fields — nothing existing regresses.
            apply_variant_fields(object, &data_dir);
            apply_gating_fields(object);
            apply_mac_and_mlx_fields(object, &data_dir);
        }
        // Append assembled external base models (empty unless external roots are
        // configured; always empty on macOS). They carry their own catalogScope /
        // installState / usable fields and deliberately skip the manifest
        // install-state sweep above, exactly as external LoRAs skip
        // `normalize_lora_entry` in `lora_catalog`.
        let external = {
            let mut cache = external_base_cache.lock();
            crate::external_base_models::scan_external_base_models(&external_roots, &mut cache)
        };
        models.extend(external);
        models.sort_by(|left, right| {
            let left_key = (
                left.get("type").and_then(Value::as_str).unwrap_or_default(),
                left.get("name").and_then(Value::as_str).unwrap_or_default(),
            );
            let right_key = (
                right
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                right
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            left_key.cmp(&right_key)
        });
        Ok(models)
    })
    .await
    .map_err(|err| ApiError::internal(format!("model catalog assembly task failed: {err}")))??;
    Ok(models)
}

/// Resolve the merged model manifest entry for `model_id` so the GPU worker no
/// longer re-parses `builtin.models.jsonc`/`user.models.jsonc` itself — Rust is
/// the single owner of manifest parsing/merging (story 1653). The merged entry
/// is injected into video job payloads as `modelManifestEntry`. Returns `{}`
/// when the model is absent from both manifests, which the worker treats the
/// same as before (fall back to the model's default repo).
pub(crate) async fn resolve_model_manifest_entry(
    state: &AppState,
    model_id: &str,
) -> Result<Value, ApiError> {
    // External ComfyUI base models (epic 10451 Phase 2, sc-10667/10668) are synthesized in the
    // catalog, not declared in a jsonc manifest, so the jsonc lookup below would return `{}` and
    // the worker would never receive their `components[]` (the DiT/TE/VAE paths). Forward the
    // assembled row for an `external_base_*` id instead, so the worker can load them in place.
    // Blocking FS scan → run on the blocking pool, mirroring `model_catalog`.
    if model_id.starts_with(crate::external_base_models::EXTERNAL_ID_PREFIX) {
        let roots = state.settings.external_model_roots.clone();
        let cache = state.external_base_model_cache.clone();
        let id = model_id.to_owned();
        let row = tokio::task::spawn_blocking(move || {
            let mut cache = cache.lock();
            crate::external_base_models::scan_external_base_models(&roots, &mut cache)
                .into_iter()
                .find(|row| row.get("id").and_then(Value::as_str) == Some(id.as_str()))
        })
        .await
        .map_err(|err| ApiError::internal(format!("external base scan task failed: {err}")))?;
        // Absent (root unconfigured, file vanished) → `{}`, the same fall-back the worker already
        // handles for an unknown model id.
        return Ok(row.unwrap_or_else(|| json!({})));
    }
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let find = |entries: &[Value]| -> Option<Value> {
        entries
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
    };
    let mut entry = merge_model_manifest_entry(find(&builtin), find(&user));
    inject_converted_model_path(&mut entry, &state.settings.data_dir);
    Ok(entry)
}

/// Populate the `modelPath` seam for convert-at-install MLX models. The worker's
/// `resolve_weights_dir` loads such a model from the locally-assembled converted
/// dir via `modelManifestEntry.modelPath`, but nothing else writes that key — the
/// raw source repo is a single safetensors file with no diffusers layout, so
/// without this the worker falls back to it and fails with "No such file or
/// directory" (e.g. flux2_klein_9b_true_v2). `mlx_catalog_status` is the single
/// source of truth for whether the conversion has produced a usable local dir.
/// No-op when the model needs no conversion, is not yet converted, or the manifest
/// already pins an explicit `modelPath`.
pub(crate) fn inject_converted_model_path(entry: &mut Value, data_dir: &FsPath) {
    let Some(object) = entry.as_object_mut() else {
        return;
    };
    let already_set = object
        .get("modelPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    if already_set {
        return;
    }
    if let Some(converted) =
        mlx_catalog_status(object, data_dir).and_then(|status| status.converted_path)
    {
        object.insert(
            "modelPath".to_owned(),
            Value::String(converted.display().to_string()),
        );
    }
}

/// One-level-deep merge of the builtin and user manifest entries for a single
/// model id. Mirrors the worker's former `ltx_model_manifest_entry` exactly so
/// this migration is behavior-preserving: user top-level keys override builtin
/// (shallow), and the nested config blocks the adapters read are merged
/// key-by-key rather than replaced wholesale. (This is intentionally deeper than
/// `merge_entries_by_id`, which the model catalog uses for display.)
pub(crate) fn merge_model_manifest_entry(builtin: Option<Value>, user: Option<Value>) -> Value {
    const NESTED_KEYS: [&str; 6] = [
        "paths",
        "resources",
        "defaults",
        "limits",
        "loraCompatibility",
        "ui",
    ];
    match (builtin, user) {
        (builtin, None) => builtin.unwrap_or_else(|| Value::Object(JsonObject::new())),
        (None, Some(user)) => user,
        (Some(builtin), Some(user)) => {
            let mut merged = builtin.clone();
            merge_object(&mut merged, user.clone());
            for key in NESTED_KEYS {
                let builtin_nested = builtin.get(key).and_then(Value::as_object);
                let user_nested = user.get(key).and_then(Value::as_object);
                if builtin_nested.is_none() && user_nested.is_none() {
                    continue;
                }
                let mut nested = builtin_nested.cloned().unwrap_or_default();
                if let Some(user_nested) = user_nested {
                    for (nested_key, value) in user_nested {
                        nested.insert(nested_key.clone(), value.clone());
                    }
                }
                if let Some(object) = merged.as_object_mut() {
                    object.insert(key.to_owned(), Value::Object(nested));
                }
            }
            merged
        }
    }
}

/// Restrict a model's `downloads` to the entries applicable to `os` (`std::env::consts::OS`).
/// A download entry with a `platforms` array applies only to the listed OSes; an entry without one
/// is platform-agnostic and always kept. Some video models ship two source repos for the same model
/// — the native MLX-convert checkpoint on macOS vs the diffusers/torch checkpoint on Windows/Linux
/// (sc-3240, Wan2.2) — so filtering here makes the download job, install status, size, and the
/// frontend's `downloads[0]` all resolve to the right per-platform repo from one seam. No-op unless
/// at least one entry is platform-tagged, so single-repo models are untouched.
pub(crate) fn retain_downloads_for_os(model: &mut Value, os: &str) {
    let Some(downloads) = model.get_mut("downloads").and_then(Value::as_array_mut) else {
        return;
    };
    if !downloads
        .iter()
        .any(|entry| entry.get("platforms").is_some())
    {
        return;
    }
    downloads.retain(
        |entry| match entry.get("platforms").and_then(Value::as_array) {
            Some(platforms) => platforms.iter().any(|p| p.as_str() == Some(os)),
            None => true,
        },
    );
}

pub(crate) fn model_download(model: &Value) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let mut fallback = None;
    for download in downloads {
        // Co-requisites (sc-9696) install alongside the primary, never AS it — skip them when
        // choosing the canonical entry for size/install-path/download.
        if !is_supported_model_download(download) || is_co_requisite_download(download) {
            continue;
        }
        fallback.get_or_insert(download);
        if download.get("default").and_then(Value::as_bool) == Some(true) {
            return Some(download.clone());
        }
    }
    fallback.cloned()
}

/// True when a download entry is a co-requisite dependency (sc-9696): fetched ALONGSIDE the primary
/// download rather than as a pick-one alternate, and gating the entry's install state. See the
/// manifest schema `downloads[].coRequisite`.
pub(crate) fn is_co_requisite_download(download: &Value) -> bool {
    download.get("coRequisite").and_then(Value::as_bool) == Some(true)
}

/// The co-requisite download entries for `model` (sc-9696) — the dependencies that must install
/// alongside the primary (e.g. the PiD decoder's shared gemma-2-2b-it caption encoder). The catalog
/// has already restricted `downloads` to the current OS (`retain_downloads_for_os`), so every entry
/// returned applies to this platform. Only provider-supported entries are returned.
pub(crate) fn model_co_requisite_downloads(model: &Value) -> Vec<Value> {
    model
        .get("downloads")
        .and_then(Value::as_array)
        .map(|downloads| {
            downloads
                .iter()
                .filter(|download| {
                    is_co_requisite_download(download) && is_supported_model_download(download)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Select a specific quant tier's download entry for a quant-matrix model (sc-8508). Returns the
/// supported `downloads` entry whose `variant` matches `variant` (case-insensitive). `None` when the
/// model declares no such tier — the caller surfaces a 400 rather than silently installing the wrong
/// tier. A `None` `variant` argument means "the default tier" and is handled by [`model_download`].
pub(crate) fn model_download_for_variant(model: &Value, variant: &str) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let wanted = variant.trim().to_ascii_lowercase();
    downloads
        .iter()
        .find(|download| {
            is_supported_model_download(download)
                && !is_co_requisite_download(download)
                && download
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_ascii_lowercase())
                    .as_deref()
                    == Some(wanted.as_str())
        })
        .cloned()
}

/// Best-effort credential host for a gated model when the manifest entry doesn't
/// set `credentialHost` explicitly: an explicit per-download `credentialHost`,
/// else the well-known host for the provider (`huggingface` ⇒ `huggingface.co`),
/// else the host of a `sourceUrl`. Normalized (scheme/path stripped, lower-cased)
/// to match how credentials are keyed in the store.
fn derive_credential_host(model: &serde_json::Map<String, Value>) -> Option<String> {
    let downloads = model.get("downloads")?.as_array()?;
    for download in downloads {
        if let Some(host) = download
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
        if download.get("provider").and_then(Value::as_str) == Some("huggingface") {
            return Some("huggingface.co".to_owned());
        }
        if let Some(host) = download
            .get("sourceUrl")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
    }
    None
}

pub(crate) fn is_supported_model_download(download: &Value) -> bool {
    download.get("provider").and_then(Value::as_str) == Some("huggingface")
        && download
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| !repo.is_empty())
}

pub(crate) fn model_download_context(model: &Value) -> Result<Option<DownloadContext>, ApiError> {
    let Some(download) = model_download(model) else {
        return Ok(None);
    };
    Ok(Some(DownloadContext {
        repo: required_string_field(&download, "repo")?.to_owned(),
        files: string_array_field(&download, "files"),
        fallback_size_bytes: manifest_download_size_bytes(model, &download),
    }))
}

pub(crate) fn huggingface_cache_health(
    repo_root: &FsPath,
    files: &[String],
) -> HuggingFaceCacheHealth {
    if !huggingface_repo_cache_exists(repo_root) {
        return HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: Vec::new(),
        };
    }
    let snapshots = huggingface_snapshot_dirs(repo_root);
    if snapshots.is_empty() {
        // The repo cache dir exists but holds no snapshot revision at all — an empty skeleton
        // (bare refs/blobs, e.g. a download that resolved zero files against an unpublished tier, or
        // a cache whose weights were pruned). Nothing is partially there, so this is MISSING, not a
        // repairable "incomplete": reporting incomplete surfaced a confusing "Cached files are
        // incomplete: snapshots/<revision>" banner for a tier that simply was never downloaded
        // (sc-9909). incomplete:false keeps it a clean not-installed state.
        return HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: vec!["snapshots/<revision>".to_owned()],
        };
    }
    if !files.is_empty() {
        return huggingface_filtered_cache_health(&snapshots, files);
    }

    let mut best_missing = Vec::new();
    for snapshot in snapshots {
        if path_is_readable_file(&snapshot.join("model_index.json")) {
            let health = diffusers_snapshot_health(&snapshot);
            if health.installed {
                return health;
            }
            if best_missing.is_empty() || health.missing_files.len() < best_missing.len() {
                best_missing = health.missing_files;
            }
            continue;
        }
        if path_is_readable_file(&snapshot.join("config.json"))
            || snapshot_has_payload_file(&snapshot)
        {
            return HuggingFaceCacheHealth::installed();
        }
        if best_missing.is_empty() {
            best_missing.push("model_index.json".to_owned());
        }
    }
    HuggingFaceCacheHealth::missing(best_missing)
}

/// If every pattern in a tier's `files` filter is confined to ONE leading directory — the standard
/// quant-tier layout `["q8/*"]` → `q8` (also `["bf16/*"]`, `["q4/*"]`) — return that directory.
/// `None` for a flat single-variant filter (`["*.safetensors"]`, whose leading component is itself a
/// glob) or patterns that span multiple top-level dirs; those are not a tier subdir and keep the
/// coarse glob check.
fn tier_subdir_name(files: &[String]) -> Option<String> {
    let mut tier: Option<&str> = None;
    for pattern in files {
        let (head, rest) = pattern.split_once('/')?;
        if head.is_empty() || rest.is_empty() || pattern_contains_glob(head) {
            return None;
        }
        match tier {
            None => tier = Some(head),
            Some(existing) if existing == head => {}
            Some(_) => return None,
        }
    }
    tier.map(str::to_owned)
}

/// The tier a `<dir>/*` whole-subdir glob names (`"q8/*"` → `"q8"`). `None` for any other pattern —
/// a specific file (`"q8/turbo_lora.safetensors"`) or a non-tier glob — so the coarse presence check
/// stays authoritative for explicit files.
fn whole_subdir_glob_tier(pattern: &str) -> Option<&str> {
    let (head, rest) = pattern.split_once('/')?;
    (rest == "*" && !head.is_empty() && !pattern_contains_glob(head)).then_some(head)
}

fn huggingface_filtered_cache_health(
    snapshots: &[PathBuf],
    files: &[String],
) -> HuggingFaceCacheHealth {
    let mut missing = files
        .iter()
        .filter(|pattern| {
            !snapshots
                .iter()
                .any(|snapshot| snapshot_contains_pattern(snapshot, pattern))
        })
        .cloned()
        .collect::<Vec<_>>();
    // Whether the COARSE check found none of the filter's patterns present — the "cleanly absent
    // tier" signal, captured before the tier-completeness augmentation below can add entries.
    let coarse_all_absent = missing.len() == files.len();

    // A `<tier>/*` whole-subdir glob is satisfied as soon as a SINGLE file under `<tier>/` exists, so
    // the coarse check never notices missing weights INSIDE the tier: a torn download (its
    // `model_index.json` + a config or two present, but the transformer/vae weights gone) reported a
    // green "Installed" badge, then failed to load at generation (`No such file or directory`). When a
    // whole-subdir tier is a diffusers pipeline (has `<tier>/model_index.json`), fold its missing
    // weight-bearing components — the SAME per-component check the non-tiered path uses — into
    // `missing`, scoped under the tier. Additional explicit patterns (e.g. a `<tier>/lora.safetensors`
    // co-requisite) are left to the coarse check above, so this never masks an explicitly-listed file.
    // A cleanly-absent tier (no `<tier>/model_index.json` present) adds nothing and stays MISSING, not
    // a repairable "incomplete" — so a valid single-quant install raises no spurious repair prompt
    // (sc-9907/sc-9909).
    for pattern in files {
        let Some(tier) = whole_subdir_glob_tier(pattern) else {
            continue;
        };
        let Some(tier_dir) = snapshots
            .iter()
            .map(|snapshot| snapshot.join(tier))
            .find(|dir| path_is_readable_file(&dir.join("model_index.json")))
        else {
            continue;
        };
        for component in diffusers_snapshot_health(&tier_dir).missing_files {
            let scoped = format!("{tier}/{component}");
            if !missing.contains(&scoped) {
                missing.push(scoped);
            }
        }
    }

    if missing.is_empty() {
        // Every expected file/pattern is present AND (for a diffusers tier) its weights are on disk.
        HuggingFaceCacheHealth::installed()
    } else if coarse_all_absent {
        // NONE of this filter's expected patterns are present: the tier is cleanly absent, not torn.
        // A quant-matrix model keeps every tier in ONE shared repo cache (bf16/, q8/, q4/ subdirs),
        // so downloading one tier populates the repo snapshot the OTHER tiers' filters also probe.
        // Reporting a not-downloaded tier as `incomplete` is what surfaced a false "Cached files are
        // incomplete" warning + Fix button for a perfectly valid single-quant install (sc-9907).
        // "You didn't download this tier" (missing) must stay distinct from "this tier is
        // half-downloaded" (incomplete) so nothing upstream raises a spurious repair prompt.
        HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: missing,
        }
    } else {
        // Some expected files present but not all — an explicit file is absent, or a diffusers tier is
        // torn (weights missing). A re-fetch repairs it.
        HuggingFaceCacheHealth::missing(missing)
    }
}

fn snapshot_contains_pattern(snapshot: &FsPath, pattern: &str) -> bool {
    if pattern_contains_glob(pattern) {
        return snapshot_files(snapshot)
            .into_iter()
            .any(|path| pattern_matches(pattern, &path));
    }
    path_is_readable_file(&snapshot.join(pattern))
}

fn pattern_contains_glob(pattern: &str) -> bool {
    pattern
        .chars()
        .any(|character| matches!(character, '*' | '?' | '[' | ']'))
}

fn diffusers_snapshot_health(snapshot: &FsPath) -> HuggingFaceCacheHealth {
    let model_index_path = snapshot.join("model_index.json");
    let Ok(contents) = std::fs::read_to_string(&model_index_path) else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };
    let Ok(index) = serde_json::from_str::<Value>(&contents) else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };
    let Some(index) = index.as_object() else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };

    let mut missing = Vec::new();
    for (component, spec) in index {
        if component.starts_with('_') || spec.is_null() {
            continue;
        }
        let class_name = spec
            .as_array()
            .and_then(|items| items.get(1))
            .and_then(Value::as_str)
            .unwrap_or_default();
        // diffusers records optional components that the pipeline doesn't use
        // as `[null, null]` (e.g. ChromaPipeline's `feature_extractor` and
        // `image_encoder`). These have no directory or files on disk by design,
        // so an empty class name means "absent" — skip it rather than reporting
        // its config/weights as missing and marking the whole model incomplete.
        if class_name.is_empty() {
            continue;
        }
        if diffusers_component_requires_weights(component, class_name) {
            // Weight-bearing components (unet, transformer, vae, text_encoder,
            // controlnet, …) reliably ship a `config.json` alongside their
            // weight files, so require both.
            if !path_is_readable_file(&snapshot.join(format!("{component}/config.json"))) {
                missing.push(format!("{component}/config.json"));
            }
            if !diffusers_component_has_weight_file(snapshot, component) {
                missing.push(format!("{component}/<weights>"));
            }
        } else if !diffusers_component_dir_nonempty(snapshot, component) {
            // Weightless auxiliary components (scheduler, tokenizer, feature
            // extractors, and image/video/composite processors) ship config
            // files whose names vary by class — scheduler_config.json,
            // tokenizer_config.json, preprocessor_config.json, and more. Hard
            // coding each variant is what produced repeated false "incomplete"
            // reports (Chroma's null optionals, Qwen2VLProcessor), so only
            // require the component directory to exist and hold at least one
            // file. A genuinely missing/partial component still trips this.
            missing.push(format!("{component}/<config>"));
        }
    }
    if missing.is_empty() {
        HuggingFaceCacheHealth::installed()
    } else {
        missing.sort();
        missing.dedup();
        HuggingFaceCacheHealth::missing(missing)
    }
}

/// Classifies a diffusers `model_index.json` component as weight-bearing.
/// Schedulers, tokenizers, feature extractors, and composite `*Processor`
/// wrappers (e.g. Qwen2VLProcessor) carry no model weights — `contains("processor")`
/// subsumes `imageprocessor` and the composite processors.
fn diffusers_component_requires_weights(component: &str, class_name: &str) -> bool {
    let class = class_name.to_ascii_lowercase();
    !(component.contains("scheduler")
        || class.contains("scheduler")
        || class.contains("tokenizer")
        || class.contains("featureextractor")
        || class.contains("processor"))
}

/// Whether a component directory exists and holds at least one file. Used as the
/// completeness signal for weightless auxiliary components, whose config file
/// names vary too much by class to enumerate reliably.
fn diffusers_component_dir_nonempty(snapshot: &FsPath, component: &str) -> bool {
    std::fs::read_dir(snapshot.join(component))
        .map(|entries| {
            entries.flatten().any(|entry| {
                let path = entry.path();
                !is_hidden_file(&path) && path_is_readable_file(&path)
            })
        })
        .unwrap_or(false)
}

fn diffusers_component_has_weight_file(snapshot: &FsPath, component: &str) -> bool {
    let component_dir = snapshot.join(component);
    let Ok(entries) = std::fs::read_dir(component_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        !is_hidden_file(&path)
            && path_is_readable_file(&path)
            && (name.ends_with(".safetensors")
                || name.ends_with(".bin")
                || name.ends_with(".msgpack")
                || name.ends_with(".gguf"))
    })
}

fn snapshot_has_payload_file(snapshot: &FsPath) -> bool {
    snapshot_files(snapshot).into_iter().any(|path| {
        let lower = path.to_ascii_lowercase();
        !lower.ends_with(".md")
            && !lower.ends_with(".png")
            && !lower.ends_with(".jpg")
            && !lower.ends_with(".jpeg")
            && !lower.ends_with(".gitattributes")
    })
}

/// Every readable file under `snapshot`, snapshot-relative, `/`-separated.
///
/// Hidden entries are excluded. They are not payload, and — because this list backs
/// [`snapshot_contains_pattern`]'s glob branch — a `._model.safetensors` sidecar would
/// otherwise satisfy a required `*.safetensors` pattern, reporting a model installed
/// while its real weights file is absent (SceneWorks#1333).
fn snapshot_files(snapshot: &FsPath) -> Vec<String> {
    let mut output = Vec::new();
    let mut stack = vec![snapshot.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_hidden_file(&path) {
                continue;
            } else if path_is_readable_file(&path) {
                if let Ok(relative) = path.strip_prefix(snapshot) {
                    output.push(relative.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    output
}

fn path_is_readable_file(path: &FsPath) -> bool {
    if std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file()) {
        return true;
    }
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.file_type().is_symlink() {
        return false;
    }
    std::fs::File::open(path).is_ok()
}

pub(crate) fn manifest_download_size_bytes(model: &Value, download: &Value) -> Option<u64> {
    // Prefer the selected download entry, then fall back to legacy model-level metadata.
    ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
        .iter()
        .find_map(|field| download.get(*field).and_then(json_size_to_u64))
        .or_else(|| {
            ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
                .iter()
                .find_map(|field| model.get(*field).and_then(json_size_to_u64))
        })
}

pub(crate) async fn estimate_huggingface_download_size(
    state: &AppState,
    repo: &str,
    files: &[String],
) -> Option<u64> {
    if matches!(
        std::env::var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    ) {
        return None;
    }
    let cache_key = (repo.to_owned(), files.to_vec());
    if let Some(cached) = state.model_size_cache.lock().get(&cache_key) {
        return cached;
    }
    let url = format!(
        "https://huggingface.co/api/models/{}?blobs=true",
        quote_huggingface_repo(repo)
    );
    let estimate =
        estimate_huggingface_download_size_uncached(&state.http_client, &url, files).await;
    match estimate {
        Some(estimate) => state.model_size_cache.lock().insert(cache_key, estimate),
        None => state.model_size_cache.lock().insert_failure(cache_key),
    }
    estimate
}

pub(crate) async fn estimate_huggingface_download_size_uncached(
    client: &reqwest::Client,
    url: &str,
    files: &[String],
) -> Option<u64> {
    let payload = tokio::time::timeout(Duration::from_secs(8), async {
        client
            .get(url.to_owned())
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json::<Value>()
            .await
            .ok()
    })
    .await
    .ok()??;
    let siblings = payload.get("siblings")?.as_array()?;
    download_size_from_siblings(siblings, files)
}

pub(crate) fn download_size_from_siblings(siblings: &[Value], files: &[String]) -> Option<u64> {
    let mut total = 0_u64;
    let mut found_size = false;
    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !allow_pattern_matches(filename, files) {
            continue;
        }
        let Some(size) = sibling.get("size").and_then(json_size_to_u64) else {
            continue;
        };
        found_size = true;
        total = total.saturating_add(size);
    }
    found_size.then_some(total)
}

pub(crate) fn model_is_installed(path: &FsPath) -> bool {
    path.is_dir() && path.join(".sceneworks-download-complete.json").is_file()
}

pub(crate) struct MlxCatalogStatus {
    pub(crate) install_state: &'static str,
    pub(crate) conversion_state: &'static str,
    pub(crate) converted_path: Option<PathBuf>,
    /// A newer source checkpoint is available than the one this install was converted from.
    /// True only for a converted `requiresConversion` model whose manifest `convertSourceFile`
    /// is NOT present in the `convertSourceRepo` cache (the installed converted dir carries no
    /// version stamp, so the source cache is the proxy — see `convert_source_file_cached`).
    pub(crate) update_available: bool,
}

/// Whether the named source `file` is present in any cached snapshot of `repo` — the proxy for
/// "the current manifest source has been downloaded." Keys off the manifest fields alone, so it
/// works for every convert-at-install model with no per-model logic.
fn convert_source_file_cached(data_dir: &FsPath, repo: &str, file: &str) -> bool {
    huggingface_repo_cache_path(data_dir, repo)
        .map(|root| crate::huggingface_snapshot_dirs(&root))
        .unwrap_or_default()
        .iter()
        .any(|snapshot| snapshot.join(file).is_file())
}

/// macOS Model Manager status for a model's `mlx` variant. Returns `None` when the
/// model declares no `mlx` block.
///
/// `conversion_state`:
/// - `ready`            turnkey MLX repo (no conversion needed)
/// - `converted`        requiresConversion and the local MLX dir exists
/// - `needs_conversion` source checkpoint present, MLX dir absent
/// - `needs_source`     source checkpoint not downloaded yet
///
/// `install_state` is `installed` when the usable MLX artifact exists.
pub(crate) fn mlx_catalog_status(
    model: &serde_json::Map<String, Value>,
    data_dir: &FsPath,
) -> Option<MlxCatalogStatus> {
    let mlx = model.get("mlx").and_then(Value::as_object)?;
    let repo_cached = |repo: &str| {
        huggingface_repo_cache_path(data_dir, repo)
            .as_deref()
            .is_some_and(huggingface_repo_cache_exists)
    };
    if mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let converted_dir = data_dir.join("models").join("mlx").join(model_id);
        // mlx-video converters write a top-level config.json; the FLUX.2-klein
        // diffusers converter (sc-2235) writes a diffusers model_index.json. Either
        // marks a finished local MLX artifact.
        if converted_dir.join("config.json").is_file()
            || converted_dir.join("model_index.json").is_file()
        {
            // The converted artifact records no source version, so use the source cache as the
            // proxy: if the manifest's current `convertSourceFile` is NOT cached, this install was
            // built from an older source → an update is available. A dir-based converter with no
            // `convertSourceFile` simply never reports an update (no false positives).
            let update_available = match (
                mlx.get("convertSourceRepo").and_then(Value::as_str),
                mlx.get("convertSourceFile").and_then(Value::as_str),
            ) {
                (Some(repo), Some(file)) if !file.trim().is_empty() => {
                    !convert_source_file_cached(data_dir, repo, file)
                }
                _ => false,
            };
            return Some(MlxCatalogStatus {
                install_state: "installed",
                conversion_state: "converted",
                converted_path: Some(converted_dir),
                update_available,
            });
        }
        let source_present = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: "missing",
            conversion_state: if source_present {
                "needs_conversion"
            } else {
                "needs_source"
            },
            converted_path: None,
            update_available: false,
        })
    } else {
        let repo_installed = mlx
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        // A turnkey model may still be served by a pre-existing local conversion at
        // <data>/models/mlx/<id> — the worker's resolve_*_model_dir prefers a local dir over
        // the turnkey download. Count that as installed too, so a model flipped from
        // requiresConversion → turnkey (sc-5599) doesn't read as "missing" for users who had
        // already converted it locally.
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let local_dir = data_dir.join("models").join("mlx").join(model_id);
        let local_installed = local_dir.join("config.json").is_file();
        Some(MlxCatalogStatus {
            install_state: if repo_installed || local_installed {
                "installed"
            } else {
                "missing"
            },
            conversion_state: "ready",
            converted_path: local_installed.then_some(local_dir),
            // Turnkey models have no local conversion to go stale (they track their repo directly).
            update_available: false,
        })
    }
}

pub(crate) fn model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = model_manifest_installed_path(model, data_dir) {
        paths.push(path);
    }
    if let Some(repo) = model_download(model).and_then(|download| {
        download
            .get("repo")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }) {
        paths.push(data_dir.join("models").join(safe_download_dir(&repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, &repo) {
            paths.push(cache_path);
        }
    }
    if let Some(source_path) = model
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        paths.push(if path.is_absolute() {
            path
        } else {
            data_dir.join(path)
        });
    }
    unique_paths(paths)
}

pub(crate) fn model_manifest_installed_path(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let raw_path = model
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if raw_path.contains("${") {
        return None;
    }
    let path = PathBuf::from(raw_path);
    Some(if path.is_absolute() {
        path
    } else {
        data_dir.join(path)
    })
}

#[cfg(test)]
mod gated_credential_tests {
    use super::*;
    use serde_json::json;

    fn map(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn derives_huggingface_host_from_provider() {
        let model = map(json!({
            "downloads": [{ "provider": "huggingface", "repo": "black-forest-labs/FLUX.1-dev", "files": [] }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("huggingface.co")
        );
    }

    #[test]
    fn prefers_explicit_download_credential_host() {
        let model = map(json!({
            "downloads": [{ "provider": "civitai", "credentialHost": "https://Civitai.com/", "sourceUrl": "https://civitai.com/api/x" }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("civitai.com")
        );
    }

    #[test]
    fn falls_back_to_source_url_host() {
        let model = map(json!({
            "downloads": [{ "provider": "url", "sourceUrl": "https://models.example.com/path/file.safetensors" }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("models.example.com")
        );
    }

    #[test]
    fn no_downloads_yields_none() {
        assert_eq!(derive_credential_host(&map(json!({}))), None);
    }

    // sc-7872: the SD3.5 gated entries download direct from the gated stabilityai/*
    // repos (no re-host), so the credential host derives to huggingface.co exactly
    // like FLUX.2-dev — driving the same stored-HF-token download path.
    #[test]
    fn derives_huggingface_host_for_stabilityai_sd3_5_repos() {
        for repo in [
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "stabilityai/stable-diffusion-3.5-medium",
        ] {
            let model = map(json!({
                "downloads": [{ "provider": "huggingface", "repo": repo, "files": ["transformer/*"] }]
            }));
            assert_eq!(
                derive_credential_host(&model).as_deref(),
                Some("huggingface.co"),
                "repo {repo} should derive huggingface.co",
            );
        }
    }

    // sc-7872: a gated SD3.5 entry round-trips through apply_gating_fields with its
    // explicit huggingface.co credential host preserved (the field the web client
    // reads to gate the download + surface the credential prompt). licenseUrl is
    // untouched, so the model card links the stabilityai license page.
    #[test]
    fn sd3_5_gated_entry_preserves_credential_host_and_license() {
        let mut model = map(json!({
            "id": "sd3_5_large",
            "gated": true,
            "credentialHost": "huggingface.co",
            "licenseUrl": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
            "downloads": [{ "provider": "huggingface", "repo": "stabilityai/stable-diffusion-3.5-large", "files": ["transformer/*"] }]
        }));
        apply_gating_fields(&mut model);
        assert_eq!(model.get("gated").and_then(Value::as_bool), Some(true));
        assert_eq!(
            model.get("credentialHost").and_then(Value::as_str),
            Some("huggingface.co"),
        );
        assert_eq!(
            model.get("licenseUrl").and_then(Value::as_str),
            Some("https://huggingface.co/stabilityai/stable-diffusion-3.5-large"),
        );
    }
}

#[cfg(test)]
mod variant_install_tests {
    use super::*;
    use serde_json::json;

    /// Seed a HuggingFace repo cache snapshot under `data_dir` containing `files` (repo-relative
    /// paths). Mirrors the on-disk layout `model_variant_states` probes.
    fn seed_cache(data_dir: &FsPath, repo: &str, files: &[&str]) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let snapshot = cache.join("snapshots").join("abc123");
        for file in files {
            let path = snapshot.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
    }

    fn quant_matrix_model(repo: &str) -> Value {
        json!({
            "id": "matrix_model",
            "downloads": [
                {
                    "provider": "huggingface",
                    "repo": repo,
                    "variant": "q4",
                    "default": true,
                    "files": ["q4/*"],
                    "footprint": { "diskSizeBytes": 5_000_000_000_u64 }
                },
                {
                    "provider": "huggingface",
                    "repo": repo,
                    "variant": "q8",
                    "files": ["q8/*"],
                    "footprint": { "diskSizeBytes": 10_000_000_000_u64, "peakMemoryBytes": null }
                },
                {
                    "provider": "huggingface",
                    "repo": repo,
                    "variant": "bf16",
                    "files": ["bf16/*"],
                    "estimatedSizeBytes": 20_000_000_000_u64
                }
            ]
        })
    }

    #[test]
    fn detects_matrix_and_single_variant_shapes() {
        // A variant-keyed multi-entry model → a matrix.
        assert!(model_has_variant_matrix(&quant_matrix_model(
            "SceneWorks/matrix"
        )));
        // A single entry with an explicit variant → still a matrix (tier-tracked).
        assert!(model_has_variant_matrix(&json!({
            "downloads": [{ "provider": "huggingface", "repo": "o/m", "variant": "q4" }]
        })));
        // A single unlabeled entry → NOT a matrix (back-compat single-variant).
        assert!(!model_has_variant_matrix(&json!({
            "downloads": [{ "provider": "huggingface", "repo": "o/m" }]
        })));
        // MULTIPLE unlabeled entries (alternate sources / co-requisite TE repos / native fallback)
        // → NOT a matrix. Entry count is not the discriminator; only an explicit `variant` is
        // (sc-8508). Guards against the old `supported.len() > 1` heuristic that falsely flagged
        // ~30 multi-repo models.
        assert!(!model_has_variant_matrix(&json!({
            "downloads": [
                { "provider": "huggingface", "repo": "org/backbone" },
                { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it" }
            ]
        })));
        // An empty-string variant is not a real tier label → not a matrix.
        assert!(!model_has_variant_matrix(&json!({
            "downloads": [
                { "provider": "huggingface", "repo": "org/a", "variant": "" },
                { "provider": "huggingface", "repo": "org/b" }
            ]
        })));
        // No downloads → not a matrix.
        assert!(!model_has_variant_matrix(&json!({ "id": "x" })));
    }

    #[test]
    fn alternate_source_multi_entry_yields_one_default_variant() {
        // Two unlabeled download entries (alternate sources / co-requisite TE repo) must NOT be
        // treated as a quant matrix: both would otherwise collapse to a duplicate "default" key.
        // The dedup guard emits exactly one "default" variant, matching the single-variant contract.
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let model = json!({
            "id": "alt_source",
            "downloads": [
                { "provider": "huggingface", "repo": "org/backbone" },
                { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it" }
            ]
        });
        assert!(!model_has_variant_matrix(&model));
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 1, "alternate-source model emits one variant");
        assert_eq!(states[0].variant, "default");
    }

    #[test]
    fn variant_keys_are_unique_across_emitted_states() {
        // Every emitted variant key must be unique. A manifest that duplicates a variant (or maps
        // two entries to the same key) keeps only the first; downstream tracking never emits two
        // same-keyed states (sc-8508).
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();

        // Genuine matrix → three distinct keys.
        let matrix = quant_matrix_model("SceneWorks/matrix");
        let states = model_variant_states(&matrix, data_dir);
        let mut keys: Vec<_> = states.iter().map(|s| s.variant.clone()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys, vec!["bf16", "q4", "q8"]);

        // Two entries sharing a variant key → collapsed to one (first wins).
        let dup = json!({
            "id": "dup",
            "downloads": [
                { "provider": "huggingface", "repo": "org/a", "variant": "q4", "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "org/b", "variant": "q4", "files": ["q4-alt/*"] }
            ]
        });
        let dup_states = model_variant_states(&dup, data_dir);
        assert_eq!(dup_states.len(), 1);
        assert_eq!(dup_states[0].variant, "q4");
    }

    #[test]
    fn single_variant_model_yields_one_default_variant() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let model = json!({
            "id": "single",
            "downloads": [{ "provider": "huggingface", "repo": "owner/single" }]
        });
        // Not installed yet.
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].variant, "default");
        assert!(!states[0].installed);

        // Seed the cache with a payload file → installed.
        seed_cache(data_dir, "owner/single", &["model.safetensors"]);
        let states = model_variant_states(&model, data_dir);
        assert!(states[0].installed);
        assert!(states[0].installed_path.is_some());
    }

    /// Mark a convert-at-install model "converted" by writing its local MLX `config.json`.
    fn seed_converted(data_dir: &FsPath, model_id: &str) {
        let dir = data_dir.join("models").join("mlx").join(model_id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
    }

    /// A converted convert-at-install model reports `updateAvailable` iff the manifest's current
    /// `convertSourceFile` is NOT in the source cache (the converted dir carries no version stamp,
    /// so the cache is the proxy). Generic: keys only off the manifest fields.
    #[test]
    fn mlx_update_available_tracks_source_file_in_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "TenStrip/LTX2.3-10Eros";
        let model = json!({
            "id": "ltx_2_3_eros",
            "mlx": {
                "requiresConversion": true,
                "converter": "ltx_video",
                "convertSourceRepo": repo,
                "convertSourceFile": "10Eros_v1.3_bf16.safetensors"
            }
        })
        .as_object()
        .unwrap()
        .clone();

        // Not converted + nothing cached → not installed, no update signal.
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert_eq!(status.install_state, "missing");
        assert!(!status.update_available);

        // Converted, but only the OLDER source is cached (manifest now points at v1.3) → stale.
        seed_converted(data_dir, "ltx_2_3_eros");
        seed_cache(data_dir, repo, &["10Eros_v1_bf16.safetensors"]);
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert_eq!(status.install_state, "installed");
        assert_eq!(status.conversion_state, "converted");
        assert!(
            status.update_available,
            "current source not cached → update available"
        );

        // The manifest's current source file is now cached → up to date.
        seed_cache(data_dir, repo, &["10Eros_v1.3_bf16.safetensors"]);
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert!(
            !status.update_available,
            "current source cached → no update"
        );
    }

    /// A dir-based converter (no `convertSourceFile`) never reports an update — the mechanism
    /// degrades to a no-op rather than misfiring, so it's safe to leave enabled for all models.
    #[test]
    fn mlx_update_unavailable_without_convert_source_file() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let model = json!({
            "id": "flux2_dev",
            "mlx": {
                "requiresConversion": true,
                "converter": "flux2_dev_quant",
                "convertSourceRepo": "black-forest-labs/FLUX.2-dev"
            }
        })
        .as_object()
        .unwrap()
        .clone();
        seed_converted(data_dir, "flux2_dev");
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert_eq!(status.conversion_state, "converted");
        assert!(
            !status.update_available,
            "no convertSourceFile → never reports an update"
        );
    }

    #[test]
    fn per_variant_tracking_reports_which_tiers_are_on_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // Only bf16 is downloaded (its `files` filter matches only the bf16/ tree).
        seed_cache(data_dir, repo, &["bf16/model.safetensors"]);

        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 3);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

        // bf16 present; q4 + q8 absent — the whole point of per-variant tracking.
        assert!(by_variant("bf16").installed, "bf16 should read installed");
        assert!(!by_variant("q4").installed, "q4 should read missing");
        assert!(!by_variant("q8").installed, "q8 should read missing");

        // Footprint + size flow through: q4 uses footprint.diskSizeBytes, bf16 uses estimatedSizeBytes.
        assert_eq!(by_variant("q4").download_size_bytes, Some(5_000_000_000));
        assert_eq!(by_variant("bf16").download_size_bytes, Some(20_000_000_000));
        assert_eq!(
            by_variant("q8").footprint.get("peakMemoryBytes"),
            Some(&Value::Null)
        );
    }

    /// Seed one quant tier as a diffusers pipeline snapshot: always a `model_index.json` +
    /// weightless scheduler/tokenizer configs; the transformer/vae/text_encoder weights only when
    /// `complete`. A `complete: false` tier mirrors a torn download (interrupted, or weights pruned)
    /// — its files satisfy the coarse `<tier>/*` glob but it cannot load.
    fn seed_diffusers_tier(data_dir: &FsPath, repo: &str, tier: &str, complete: bool) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let root = cache.join("snapshots").join("abc123").join(tier);
        let write = |rel: &str, body: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        };
        write(
            "model_index.json",
            r#"{
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"]
            }"#,
        );
        // Weightless components ship only config, present in a torn tier too (this is what makes the
        // coarse glob match and wrongly report "installed").
        write("scheduler/scheduler_config.json", "{}");
        write("tokenizer/tokenizer_config.json", "{}");
        write("text_encoder/config.json", "{}");
        if complete {
            write("transformer/config.json", "{}");
            write("transformer/model.safetensors", "weights");
            write("vae/config.json", "{}");
            write("vae/model.safetensors", "weights");
            write("text_encoder/model.safetensors", "weights");
        }
    }

    /// The regression this fix closes: a torn diffusers tier (its `model_index.json` + a config or two
    /// present, but the transformer/vae weights missing) satisfied the coarse `<tier>/*` glob and so
    /// reported a green "Installed" badge — then failed to load at generation with `No such file or
    /// directory`. A tier must read installed only when its weight-bearing components actually hold
    /// weights; an absent tier stays a clean "missing", not a repairable "incomplete".
    #[test]
    fn torn_diffusers_tier_reads_incomplete_not_installed() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // q4 complete (loads), q8 torn (metadata only — the transformer weights never arrived),
        // bf16 never fetched.
        seed_diffusers_tier(data_dir, repo, "q4", true);
        seed_diffusers_tier(data_dir, repo, "q8", false);

        let states = model_variant_states(&model, data_dir);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

        assert!(
            by_variant("q4").installed,
            "complete q4 tier must read installed"
        );
        assert!(
            !by_variant("q8").installed,
            "torn q8 tier must NOT read installed just because its metadata files match the glob"
        );
        assert!(
            by_variant("q8").cache_incomplete,
            "a torn (half-present) tier is a repairable incomplete, not a clean missing"
        );
        assert!(
            !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
            "a never-fetched tier stays a clean missing (no spurious repair prompt — sc-9907)"
        );

        // Model-level state aggregates: q4 is complete, so the model is installed overall — the torn
        // q8 must not drag the whole model to incomplete (sc-9907), but it also must not itself count.
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(state.installed, "model installed via the complete q4 tier");
    }

    /// A model whose ONLY on-disk tier is torn must read NOT installed at the model level — including
    /// via the "usable stale" receipt path (sc-13076 backfilled a receipt for whatever was on disk,
    /// so a metadata-only tier produced a receipt whose files all exist yet cannot load).
    #[test]
    fn model_with_only_a_torn_tier_is_not_installed() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // Only q8 present, and torn. Plus a backfilled receipt recording its (weightless) files as if
        // complete — exactly the shape found on the reporter's disk.
        seed_diffusers_tier(data_dir, repo, "q8", false);
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "repo": repo,
                "receipts": [{
                    "repo": repo, "modelId": "matrix_model", "variant": "q8",
                    "manifestFiles": ["q8/*"],
                    "resolvedFiles": [
                        "q8/model_index.json", "q8/scheduler/scheduler_config.json",
                        "q8/tokenizer/tokenizer_config.json", "q8/text_encoder/config.json"
                    ],
                    "backfilled": true
                }]
            }))
            .unwrap(),
        )
        .unwrap();

        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            !state.installed,
            "a torn-only install must not read installed via the usable-stale receipt path"
        );
    }

    #[test]
    fn variant_footprint_disk_bytes_reads_required_field() {
        let entry = json!({ "footprint": { "diskSizeBytes": 42 } });
        assert_eq!(variant_footprint_disk_bytes(&entry), Some(42));
        assert_eq!(variant_footprint_disk_bytes(&json!({})), None);
    }

    #[test]
    fn variant_download_selector_picks_the_right_tier() {
        let model = quant_matrix_model("SceneWorks/matrix");
        // Case-insensitive match on the declared variant.
        assert_eq!(
            model_download_for_variant(&model, "Q8")
                .and_then(|d| d.get("files").cloned())
                .and_then(|f| f.as_array().and_then(|a| a.first().cloned())),
            Some(Value::String("q8/*".to_owned()))
        );
        // Unknown tier → None (the handler turns this into a 400).
        assert!(model_download_for_variant(&model, "int8").is_none());
        // The default selector still picks the `default: true` (q4) entry — back-compat.
        assert_eq!(
            model_download(&model)
                .and_then(|d| d.get("variant").and_then(Value::as_str).map(str::to_owned)),
            Some("q4".to_owned())
        );
    }
}

#[cfg(test)]
mod mlx_tier_probe_tests {
    use super::*;

    fn write_weight(dir: &std::path::Path, backbone: &str, file: &str) {
        let d = dir.join(backbone);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(file), b"x").unwrap();
    }

    #[test]
    fn convert_output_tiers_probes_diffusion_models_highest_first_ignoring_appledouble() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Anima layout: <tier>/diffusion_models/<dit>.safetensors present for bf16 + q8; q4 has only a
        // hidden `._` AppleDouble sidecar, which must NOT count as a loadable tier (SceneWorks#1333).
        write_weight(
            &root.join("bf16"),
            "diffusion_models",
            "anima-base-v1.0.safetensors",
        );
        write_weight(
            &root.join("q8"),
            "diffusion_models",
            "anima-base-v1.0.safetensors",
        );
        write_weight(
            &root.join("q4"),
            "diffusion_models",
            "._anima-base-v1.0.safetensors",
        );
        // Highest-fidelity first, q4 excluded.
        assert_eq!(mlx_convert_output_tiers(root), vec!["bf16", "q8"]);
    }

    #[test]
    fn convert_output_tiers_handles_transformer_flat_and_empty_layouts() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_weight(&root.join("q8"), "transformer", "model.safetensors");
        // Flat layout: a sharded index sits directly in the tier dir (no backbone subdir).
        std::fs::create_dir_all(root.join("bf16")).unwrap();
        std::fs::write(
            root.join("bf16")
                .join("diffusion_pytorch_model.safetensors.index.json"),
            b"x",
        )
        .unwrap();
        assert_eq!(mlx_convert_output_tiers(root), vec!["bf16", "q8"]);
        // A flat converted dir (no tier subdirs) yields no tiers → the web renders no picker.
        let flat = tempfile::tempdir().unwrap();
        std::fs::write(flat.path().join("model_index.json"), b"{}").unwrap();
        assert!(mlx_convert_output_tiers(flat.path()).is_empty());
    }

    // Full catalog path: a converted convert-at-install model (Anima) emits `mlxTiers` from
    // `apply_mac_and_mlx_fields`, so /models carries the Studio picker data. macOS-only (the mlx status
    // probe is `cfg!(target_os = "macos")`).
    #[test]
    #[cfg(target_os = "macos")]
    fn catalog_emits_mlxtiers_for_converted_anima() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let converted = data_dir.join("models").join("mlx").join("anima_base");
        std::fs::create_dir_all(&converted).unwrap();
        std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
        for tier in ["bf16", "q8", "q4"] {
            let dm = converted.join(tier).join("diffusion_models");
            std::fs::create_dir_all(&dm).unwrap();
            std::fs::write(dm.join("anima-base-v1.0.safetensors"), b"x").unwrap();
        }
        let mut object = json!({
            "id": "anima_base",
            "type": "image",
            "mlx": { "requiresConversion": true }
        })
        .as_object()
        .unwrap()
        .clone();
        apply_mac_and_mlx_fields(&mut object, data_dir);
        assert_eq!(
            object.get("mlxConversionState").and_then(Value::as_str),
            Some("converted")
        );
        let tiers: Vec<&str> = object
            .get("mlxTiers")
            .and_then(Value::as_array)
            .expect("mlxTiers emitted")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(tiers, vec!["bf16", "q8", "q4"]);
        // Decoupled from the download matrix — the picker must NOT flip `hasVariantMatrix`.
        assert!(object.get("hasVariantMatrix").is_none());
    }

    // Per-model quality floor (sc-10731): `apply_mac_and_mlx_fields` surfaces the manifest
    // `mlx.minQualityTier` as a top-level `minQualityTier` so the web can clamp the DEFAULT tier up to
    // it. Platform-independent (not gated on the macOS mlx-status probe), and only a valid bf16/q8/q4
    // value is emitted — a bogus floor is dropped, an absent floor emits nothing.
    #[test]
    fn catalog_emits_min_quality_floor_from_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let apply = |mlx: Value| {
            let mut object = json!({ "id": "anima_base", "type": "image", "mlx": mlx })
                .as_object()
                .unwrap()
                .clone();
            apply_mac_and_mlx_fields(&mut object, data_dir);
            object
                .get("minQualityTier")
                .and_then(Value::as_str)
                .map(str::to_owned)
        };
        // A declared q8 floor is surfaced verbatim as a top-level field.
        assert_eq!(
            apply(json!({ "minQualityTier": "q8" })),
            Some("q8".to_owned())
        );
        // A model with no floor emits nothing (default absent = q4-tolerant, no clamp).
        assert_eq!(apply(json!({ "requiresConversion": true })), None);
        // An invalid floor value is dropped rather than surfaced.
        assert_eq!(apply(json!({ "minQualityTier": "q2" })), None);
    }
}

// Per-tier delete (sc-12024, epic 8506). Exercises the blob-aware reclamation on a realistic HF
// hub-cache layout — real `blobs/<etag>` files with snapshot SYMLINKS into them — which is the whole
// reason a tier delete is non-trivial: unlinking the tier's snapshot symlinks alone frees nothing.
// unix-gated because the fixtures use symlinks (the production cache layout on macOS/Linux).
#[cfg(all(test, unix))]
mod variant_delete_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    /// Write (once) a `blobs/<etag>` file of `bytes.len()` bytes and return its path.
    fn blob(repo: &FsPath, etag: &str, bytes: &[u8]) -> PathBuf {
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let path = blobs.join(etag);
        if !path.exists() {
            std::fs::write(&path, bytes).unwrap();
        }
        path
    }

    /// Materialize `snapshots/rev/<rel>` as a symlink to `blob_path` (the production cache links
    /// relatively; an absolute link resolves identically under `canonicalize`).
    fn link(repo: &FsPath, rel: &str, blob_path: &FsPath) {
        let link_path = repo.join("snapshots").join("rev").join(rel);
        std::fs::create_dir_all(link_path.parent().unwrap()).unwrap();
        symlink(blob_path, &link_path).unwrap();
    }

    /// Seed one snapshot file backed by its own fresh blob of `size` bytes.
    fn seed(repo: &FsPath, rel: &str, etag: &str, size: usize) {
        let blob_path = blob(repo, etag, &vec![0u8; size]);
        link(repo, rel, &blob_path);
    }

    #[tokio::test]
    async fn deletes_only_the_target_tiers_blobs_and_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        seed(&repo, "q4/model.safetensors", "q4a", 100);
        seed(&repo, "q4/config.json", "q4b", 200);
        seed(&repo, "q8/model.safetensors", "q8a", 500);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/*".to_owned()],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // q4's blobs AND snapshot symlinks are gone; the emptied q4 dir is pruned.
        assert!(!repo.join("blobs/q4a").exists());
        assert!(!repo.join("blobs/q4b").exists());
        assert!(!repo.join("snapshots/rev/q4").exists());
        // q8 is fully intact.
        assert!(repo.join("blobs/q8a").exists());
        assert!(repo.join("snapshots/rev/q8/model.safetensors").exists());
        // Reclaimed bytes = q4's blob sizes only (100 + 200), never the shared skeleton.
        assert_eq!(removal.reclaimed_bytes, 300);
        assert!(removal.trash_failed_paths.is_empty());
        assert!(!removal.removed_paths.is_empty());
    }

    #[tokio::test]
    async fn retains_a_blob_shared_with_a_surviving_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        // A single blob referenced by BOTH tiers (identical etag/content), plus a q4-only blob.
        let shared = blob(&repo, "shared", &vec![0u8; 400]);
        link(&repo, "q4/shared.safetensors", &shared);
        link(&repo, "q8/shared.safetensors", &shared);
        seed(&repo, "q4/only.safetensors", "q4only", 100);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/*".to_owned()],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // The shared blob survives — q8 still references it — and q8's link still resolves.
        assert!(repo.join("blobs/shared").exists());
        assert!(repo.join("snapshots/rev/q8/shared.safetensors").exists());
        // q4's exclusive blob and all q4 links are removed.
        assert!(!repo.join("blobs/q4only").exists());
        assert!(!repo.join("snapshots/rev/q4").exists());
        // Only the exclusive blob's bytes count as reclaimed; the shared blob does not.
        assert_eq!(removal.reclaimed_bytes, 100);
    }

    #[tokio::test]
    async fn draining_the_last_tier_removes_the_repo_cache_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        seed(&repo, "q4/model.safetensors", "q4a", 100);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/*".to_owned()],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // No tier remains → the whole models--repo dir is pruned (no bare refs/ skeleton left behind).
        assert!(!repo.exists());
        assert_eq!(removal.reclaimed_bytes, 100);
    }

    #[tokio::test]
    async fn removes_real_tier_files_from_the_managed_mirror() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let managed = models.join("Org__repo");
        let write = |rel: &str, size: usize| {
            let path = managed.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, vec![0u8; size]).unwrap();
        };
        // A turnkey install writes REAL files (not blob symlinks) under the managed dir.
        write("q4/model.safetensors", 300);
        write("q8/model.safetensors", 500);

        let removal = remove_tier_artifacts(
            None,
            Some(managed.clone()),
            &["q4/*".to_owned()],
            std::slice::from_ref(&models),
            true,
        )
        .await
        .unwrap();

        assert!(!managed.join("q4").exists());
        assert!(managed.join("q8/model.safetensors").exists());
        assert_eq!(removal.reclaimed_bytes, 300);
    }

    #[tokio::test]
    async fn empty_tier_files_is_a_no_op() {
        // The "never scope a delete to everything" guard: an empty file filter removes nothing.
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        seed(&repo, "q4/model.safetensors", "q4a", 100);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &[],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        assert!(repo.join("blobs/q4a").exists());
        assert!(removal.removed_paths.is_empty());
        assert_eq!(removal.reclaimed_bytes, 0);
    }

    // Convert-at-install (Anima) tiers are real `<converted>/<tier>/` dirs with a packed DiT plus
    // SYMLINKS to a shared TE/VAE source that lives outside the tier dirs (sc-12025).
    fn seed_convert_tier(converted: &FsPath, tier: &str, dit_bytes: usize, shared_te: &FsPath) {
        let tier_dir = converted.join(tier);
        let dm = tier_dir.join("diffusion_models");
        std::fs::create_dir_all(&dm).unwrap();
        std::fs::write(dm.join("dit.safetensors"), vec![0u8; dit_bytes]).unwrap();
        let te = tier_dir.join("text_encoders");
        std::fs::create_dir_all(&te).unwrap();
        symlink(shared_te, te.join("te.safetensors")).unwrap();
    }

    #[tokio::test]
    async fn removes_a_convert_tier_counting_only_real_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let converted = models.join("mlx").join("anima_base");
        std::fs::create_dir_all(&converted).unwrap();
        std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
        // The shared TE source lives OUTSIDE the tier dirs — deleting a tier must never free it.
        let shared = models.join("source").join("te.safetensors");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::write(&shared, vec![0u8; 999]).unwrap();
        seed_convert_tier(&converted, "q4", 300, &shared);
        seed_convert_tier(&converted, "q8", 500, &shared);

        let removal =
            remove_converted_tier(converted.join("q4"), std::slice::from_ref(&models), true)
                .await
                .unwrap();

        assert!(!converted.join("q4").exists());
        assert!(converted
            .join("q8/diffusion_models/dit.safetensors")
            .exists());
        // The shared TE source and the converted marker both survive (q8 still installed).
        assert!(shared.exists());
        assert!(converted.join("model_index.json").exists());
        // Only q4's real DiT bytes count — the symlinked shared TE is not reclaimed.
        assert_eq!(removal.reclaimed_bytes, 300);
    }

    #[tokio::test]
    async fn draining_the_last_convert_tier_removes_the_converted_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let converted = models.join("mlx").join("anima_base");
        std::fs::create_dir_all(&converted).unwrap();
        std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
        let shared = models.join("source").join("te.safetensors");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::write(&shared, vec![0u8; 999]).unwrap();
        seed_convert_tier(&converted, "q4", 300, &shared);

        let removal =
            remove_converted_tier(converted.join("q4"), std::slice::from_ref(&models), true)
                .await
                .unwrap();

        // No tier remains → the whole converted dir (marker included) is gone; the shared source is
        // NOT (it belongs to the download, not the convert output).
        assert!(!converted.exists());
        assert!(shared.exists());
        assert_eq!(removal.reclaimed_bytes, 300);
    }
}
