use super::*;

pub(super) fn validate_huggingface_repo(repo: &str) -> Result<&str, ApiError> {
    sceneworks_core::hf_repo_id::namespaced(repo).map_err(ApiError::bad_request)
}

/// Import-compatibility gate for a base checkpoint (sc-14019, epic 14015): resolve the primary
/// weight file under `source`, classify it with the base-weight detector, and refuse the import
/// unless the verdict is one a real loader accepts today
/// ([`sceneworks_core::base_weights::import_supported`]). The refusal reason is surfaced to the API
/// client as a `400` so the upload flow gets a synchronous, actionable message rather than a queued
/// job that fails later. This gate is **additive** to path confinement — `source` has already been
/// confined by `validate_import_source_path` (LAN-exposed jobs API, epic 4484); this never
/// widens or bypasses that. The worker re-runs the same predicate over the downloaded bytes so repo/
/// URL imports (whose file is not on disk at queue time) are covered there.
pub(super) fn import_source_supported(source: &FsPath) -> Result<Option<String>, ApiError> {
    let weight_file = imported_model_primary_weight_file(source)
        .or_else(|| sceneworks_core::base_weights::minimax_h3_import_primary(source));
    let Some(weight_file) = weight_file else {
        return Err(ApiError::bad_request(
            "No safetensors base-weight file was found to import; single-file base-checkpoint import expects a .safetensors transformer.",
        ));
    };
    let detection =
        sceneworks_core::base_weights::detect_import_source_weights(source, &weight_file)
            .map_err(model_family_inspection_error)?;
    import_detection_supported(&detection).map_err(ApiError::bad_request)?;
    if matches!(
        &detection,
        BaseWeightDetection::Recognized(verdict)
            if verdict.family.as_deref() == Some("mage-flow")
    ) && (!source.is_dir()
        || !sceneworks_core::base_weights::is_mage_flow_transformer_dir(source))
    {
        return Err(ApiError::bad_request(
            "Model import for the 'mage-flow' family requires a complete transformer directory \
             containing config.json and diffusion_pytorch_model.safetensors; a bare weights file \
             is refused because the loader cannot derive its architecture.",
        ));
    }
    // H3 source shapes are codec-owned: dense/sharded diffusers or native single-file. The worker
    // canonicalizes native bytes through the shared derivative cache before the provider opens it.
    if matches!(
        &detection,
        BaseWeightDetection::Recognized(verdict)
            if verdict.family.as_deref() == Some("minimax-h3")
    ) && !sceneworks_core::base_weights::is_minimax_h3_import_source(source)
    {
        return Err(ApiError::bad_request(
            "MiniMax-H3 import requires a dense native/ComfyUI single-file or a complete dense/sharded \
             MiniMaxH3Transformer3DModel transformer directory; unknown quantized dialects and \
             directories with unplanned weights are refused.",
        ));
    }
    Ok(match detection {
        BaseWeightDetection::Recognized(verdict) => verdict.family,
        BaseWeightDetection::Unrecognized { .. } => None,
    })
}

#[allow(clippy::result_large_err)]
pub(crate) async fn create_model_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    model_import_request(state, request, None).await
}

#[allow(clippy::result_large_err)]
pub(crate) async fn repair_model_import_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    model_import_request(state, request, Some(model_id)).await
}

#[allow(clippy::result_large_err)]
pub(super) async fn model_import_request(
    state: AppState,
    request: AxumRequest,
    repair_model_id: Option<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
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
        let result = queue_model_import_job(state, payload, repair_model_id).await;
        if result.is_err() {
            cleanup_staged_model_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<ModelImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_model_import_job(state, payload, repair_model_id)
        .await
        .map_err(IntoResponse::into_response)
}

/// The `(rootId, relativePath)` pair a linked checkpoint import names, if this request is one.
///
/// Both or neither: a request carrying only one half names no checkpoint, and silently treating it
/// as a fetch import would queue a job with no source at all.
pub(super) fn linked_checkpoint_selection(
    payload: &ModelImportRequest,
) -> Result<Option<(String, String)>, ApiError> {
    let root_id = payload
        .linked_root_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let relative_path = payload
        .linked_relative_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (root_id, relative_path) {
        (Some(root_id), Some(relative_path)) => {
            Ok(Some((root_id.to_owned(), relative_path.to_owned())))
        }
        (None, None) => Ok(None),
        _ => Err(ApiError::bad_request(
            "A linked checkpoint import requires both linkedRootId and linkedRelativePath",
        )),
    }
}

/// What a managed install records about where its bytes came from (sc-20636). Serialized onto the
/// queued job as `importProvenance` and handed to the ingest, which puts it in the persisted plan.
///
/// It carries no secret: `credential_host` names the host whose stored credential the download will
/// use, never the credential, and `url` is already stripped of userinfo and secret-bearing query
/// parameters by `checkpoint_ingest::sanitize_provenance_url`.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelImportProvenance {
    pub(crate) source: &'static str,
    pub(crate) reference: Option<String>,
    pub(crate) url: Option<String>,
    pub(crate) version_id: Option<String>,
    pub(crate) file_id: Option<String>,
    pub(crate) credential_host: Option<String>,
}

impl ModelImportProvenance {
    pub(super) fn to_json(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("source".to_owned(), Value::String(self.source.to_owned()));
        for (key, value) in [
            ("reference", &self.reference),
            ("url", &self.url),
            ("versionId", &self.version_id),
            ("fileId", &self.file_id),
            ("credentialHost", &self.credential_host),
        ] {
            if let Some(value) = value {
                object.insert(key.to_owned(), Value::String(value.clone()));
            }
        }
        Value::Object(object)
    }

    /// The `source.provider` value the model manifest entry records for this import.
    ///
    /// Deliberately COARSER than [`Self::source`]: `upload` and `local-copy` both record the
    /// historical `"local"`. That is the value every model entry written before epic 20398 carries
    /// and every reader of this field was written against, so splitting it in two would be a silent
    /// manifest-vocabulary change nothing asked for. The finer distinction is not lost — it is
    /// recorded in `importProvenance.source`, a new field with no legacy readers. A REMOTE import
    /// does widen the provider (`civitai` rather than the generic `url` its transfer shares),
    /// because that names a genuinely new provider rather than re-spelling an existing one.
    pub(super) fn manifest_provider(&self) -> &'static str {
        match self.source {
            "upload" | "local-copy" => "local",
            remote => remote,
        }
    }
}

/// Reconcile a discriminated source's `expectedSha256` with the flat `payload.expectedSha256`.
///
/// Both set and naming the SAME digest is honoured — a sha256 is hex, so case and surrounding
/// whitespace are not a difference. Both set and naming DIFFERENT digests is refused, because
/// resolving that by precedence would verify the transferred bytes against a digest the caller did
/// not choose and report success. A blank flat field is not a value, so it is not a conflict.
pub(super) fn reconcile_expected_sha256(
    from_source: Option<String>,
    from_flat: Option<String>,
) -> Result<Option<String>, ApiError> {
    let blank_is_absent = |value: Option<String>| value.filter(|digest| !digest.trim().is_empty());
    match (blank_is_absent(from_source), blank_is_absent(from_flat)) {
        (Some(source_digest), Some(flat_digest)) => {
            let normalize = |digest: &str| digest.trim().to_ascii_lowercase();
            if normalize(&source_digest) != normalize(&flat_digest) {
                return Err(ApiError::bad_request(format!(
                    "Provide either the discriminated `source` or the legacy `expectedSha256` field, not both: they name different digests ({} and {})",
                    source_digest.trim(),
                    flat_digest.trim()
                )));
            }
            Ok(Some(source_digest))
        }
        (Some(digest), None) | (None, Some(digest)) => Ok(Some(digest)),
        (None, None) => Ok(None),
    }
}

pub(super) fn normalize_import_repo_ids(payload: &mut ModelImportRequest) -> Result<(), ApiError> {
    if let Some(repo) = payload.repo.as_mut() {
        *repo = validate_huggingface_repo(repo)?.to_owned();
    }
    if let Some(ModelImportSourceV1::HuggingFace { repo, .. }) = payload.source.as_mut() {
        *repo = validate_huggingface_repo(repo)?.to_owned();
    }
    Ok(())
}

/// Collapse the discriminated `source` (when present) onto the flat `repo`/`sourceUrl`/`sourcePath`
/// fields the rest of the import path already speaks, and describe what was chosen as provenance.
///
/// Both spellings converge HERE, before any validation, so there is one import pipeline rather than
/// a new one beside the old: a legacy `{"repo": "..."}` body and a
/// `{"source": {"kind": "huggingFace", "repo": "..."}}` body produce the identical job payload apart
/// from the provenance detail the discriminated form additionally carries.
///
/// A request that sets both a discriminated source and a conflicting flat field is refused rather
/// than resolved by precedence — silently honouring one and dropping the other is how an import
/// fetches something the caller did not ask for.
pub(crate) fn normalize_model_import_source(
    payload: &mut ModelImportRequest,
) -> Result<(ModelImportProvenance, Option<String>), ApiError> {
    normalize_import_repo_ids(payload)?;
    let Some(source) = payload.source.take() else {
        // Legacy flat request. `uploadedSourcePath` is set only by the multipart handler, so it is
        // what distinguishes an upload from a copy of a file the user already had.
        let provenance = if let Some(repo) = payload.repo.as_deref() {
            ModelImportProvenance {
                source: "huggingface",
                reference: Some(repo.to_owned()),
                ..Default::default()
            }
        } else if let Some(url) = payload.source_url.as_deref() {
            ModelImportProvenance {
                source: "url",
                url: sceneworks_core::checkpoint_ingest::sanitize_provenance_url(url),
                credential_host: sceneworks_core::checkpoint_ingest::provenance_credential_host(
                    url,
                ),
                ..Default::default()
            }
        } else if payload.uploaded_source_path {
            ModelImportProvenance {
                source: "upload",
                reference: payload.files.first().cloned(),
                ..Default::default()
            }
        } else {
            ModelImportProvenance {
                source: "local-copy",
                reference: payload
                    .source_path
                    .as_deref()
                    .and_then(|path| FsPath::new(path).file_name()?.to_str())
                    .map(str::to_owned),
                ..Default::default()
            }
        };
        return Ok((provenance, None));
    };

    let conflict = |field: &str| {
        Err(ApiError::bad_request(format!(
            "Provide either the discriminated `source` or the legacy `{field}` field, not both"
        )))
    };
    if !option_str_is_empty(payload.repo.as_deref()) {
        return conflict("repo");
    }
    if !option_str_is_empty(payload.source_url.as_deref()) {
        return conflict("sourceUrl");
    }
    // An upload fills `sourcePath` server-side from the multipart `file` part, which is not a
    // caller-supplied conflict.
    if !option_str_is_empty(payload.source_path.as_deref()) && !payload.uploaded_source_path {
        return conflict("sourcePath");
    }

    let mut revision = None;
    let provenance = match source {
        ModelImportSourceV1::Upload { staged_path } => {
            let staged_path = staged_path
                .or_else(|| payload.source_path.clone())
                .ok_or_else(|| {
                    ApiError::bad_request(
                        "An upload source requires the file to be sent in the same multipart request",
                    )
                })?;
            payload.source_path = Some(staged_path);
            payload.uploaded_source_path = true;
            ModelImportProvenance {
                source: "upload",
                reference: payload.files.first().cloned(),
                ..Default::default()
            }
        }
        ModelImportSourceV1::LocalPath { path } => {
            let reference = FsPath::new(&path)
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned);
            payload.source_path = Some(path);
            payload.uploaded_source_path = false;
            ModelImportProvenance {
                source: "local-copy",
                reference,
                ..Default::default()
            }
        }
        ModelImportSourceV1::Url {
            url,
            expected_sha256,
        } => {
            let provenance = ModelImportProvenance {
                source: "url",
                url: sceneworks_core::checkpoint_ingest::sanitize_provenance_url(&url),
                credential_host: sceneworks_core::checkpoint_ingest::provenance_credential_host(
                    &url,
                ),
                ..Default::default()
            };
            payload.source_url = Some(url);
            payload.expected_sha256 =
                reconcile_expected_sha256(expected_sha256, payload.expected_sha256.take())?;
            provenance
        }
        ModelImportSourceV1::HuggingFace {
            repo,
            revision: requested_revision,
            files,
        } => {
            if !files.is_empty() {
                payload.files = files;
            }
            revision = requested_revision.clone();
            let reference = match requested_revision.as_deref() {
                Some(revision) => format!("{repo}@{revision}"),
                None => repo.clone(),
            };
            payload.repo = Some(repo);
            ModelImportProvenance {
                source: "huggingface",
                reference: Some(reference),
                version_id: requested_revision,
                file_id: payload.files.first().cloned(),
                ..Default::default()
            }
        }
        ModelImportSourceV1::Civitai {
            url,
            model_version_id,
            file_id,
            expected_sha256,
        } => {
            let provenance = ModelImportProvenance {
                source: "civitai",
                reference: model_version_id
                    .as_deref()
                    .map(|version| format!("modelVersion/{version}")),
                url: sceneworks_core::checkpoint_ingest::sanitize_provenance_url(&url),
                version_id: model_version_id,
                file_id,
                credential_host: sceneworks_core::checkpoint_ingest::provenance_credential_host(
                    &url,
                ),
            };
            // Civitai downloads through the ordinary source-URL path: same transfer, same
            // host-keyed credential attachment, same size limits. Only the recorded identity
            // differs.
            payload.source_url = Some(url);
            payload.expected_sha256 =
                reconcile_expected_sha256(expected_sha256, payload.expected_sha256.take())?;
            provenance
        }
    };
    Ok((provenance, revision))
}

/// Re-resolve a managed NVFP4 variant's pin SERVER-SIDE (sc-11043, epic 11037).
///
/// `GET /api/v1/models/managed-variants` advertises, for each registered variant, the exact import
/// body that installs it. That body is a convenience for the client, and a convenience is not a
/// guarantee: without this, a client could `POST` `modelId: "nvfp4-krea-2-turbo"` with any repo,
/// revision or file it liked — or omit `expectedSha256` entirely, which the worker's
/// `if let Some(..)` check treats as "nothing to verify" — and SceneWorks would record unverified
/// bytes under the curated variant's identity, its checkpoint id, and its provenance. AC3's
/// "verified file checksum" has to be enforced by the server or it is only client-cooperative.
///
/// So the registration wins. The registry is keyed on the very id the request claims, so when the
/// id names a variant, the repo, revision, file list and `expectedSha256` are taken from the
/// registration rather than from the body. A request that explicitly names a DIFFERENT pin is
/// refused rather than silently rewritten — that is a client bug or an attempt, and both deserve a
/// diagnostic. There is still exactly ONE import pipeline: this only fills in the fetch, and
/// everything after it (normalisation, licence gate, install-id validation, the ingest) is the
/// generic path every other import walks.
///
/// A managed variant is a pinned FETCH of specific upstream bytes, so the ownership modes that do
/// not fetch — a linked library reference, a local copy, an upload, a bare URL — cannot claim one.
pub(super) fn resolve_managed_variant_pin(
    payload: &mut ModelImportRequest,
) -> Result<(), ApiError> {
    use sceneworks_core::managed_checkpoint_variants::{
        match_managed_nvfp4_variant_id, ManagedVariantIdMatch,
    };
    // Case-INSENSITIVELY, because a managed variant's id is its install-directory name and its
    // checkpoint identity, and `safe_download_dir` preserves whatever case the request supplied
    // while NTFS and a default APFS volume do not. An exact-match lookup therefore missed
    // `"NVFP4-Krea-2-Turbo"` — no registered repo, revision, file or checksum was applied — while
    // the distinct-id collision check saw a new id and the filesystem saw the curated variant's own
    // directory. Arbitrary bytes then landed under the curated identity with nothing verified.
    //
    // A near miss is REFUSED, never normalized to the canonical id. Rewriting it would install
    // something the request did not name, which is the same class of silent substitution this
    // function exists to refuse everywhere else; the diagnostic names the canonical id so a client
    // bug is one edit away from correct.
    let variant = match match_managed_nvfp4_variant_id(payload.model_id.as_deref().unwrap_or("")) {
        ManagedVariantIdMatch::Unregistered => return Ok(()),
        ManagedVariantIdMatch::Exact(variant) => variant,
        ManagedVariantIdMatch::NearMiss(variant) => {
            return Err(ApiError::bad_request(format!(
                "Model id '{}' differs only in case from the managed NVFP4 variant '{}', which \
                 shares its install directory and checkpoint identity on this platform. Import it \
                 under the exact id '{}', or choose a model id that is not a case variant of a \
                 managed variant.",
                payload.model_id.as_deref().unwrap_or("").trim(),
                variant.variant_id,
                variant.variant_id
            )));
        }
    };
    let claim = |detail: &str| {
        Err(ApiError::bad_request(format!(
            "Model id '{}' is a managed NVFP4 variant pinned to {}@{} {}; {detail}",
            variant.variant_id, variant.repo, variant.revision, variant.repo_file
        )))
    };

    if payload.linked_root_id.is_some() || payload.linked_relative_path.is_some() {
        return claim(
            "it is installed by fetching those pinned bytes, so it cannot be claimed by a linked \
             library reference. Import your own copy under a different model id.",
        );
    }
    if !option_str_is_empty(payload.source_url.as_deref())
        || !option_str_is_empty(payload.source_path.as_deref())
    {
        return claim(
            "it cannot be installed from a source URL or a local path. Import your own copy under \
             a different model id.",
        );
    }
    match payload.source.as_ref() {
        None | Some(ModelImportSourceV1::HuggingFace { .. }) => {}
        Some(_) => {
            return claim(
                "it is a Hugging Face repo import. Import your own copy under a different model \
                 id.",
            )
        }
    }
    if let Some(repo) = payload.repo.as_deref().map(str::trim) {
        if !repo.is_empty() && repo != variant.repo {
            return claim(&format!("this request names repo '{repo}'"));
        }
    }
    if let Some(ModelImportSourceV1::HuggingFace {
        repo,
        revision,
        files,
    }) = payload.source.as_ref()
    {
        if repo.trim() != variant.repo {
            return claim(&format!("this request names repo '{}'", repo.trim()));
        }
        if let Some(revision) = revision.as_deref().map(str::trim) {
            if !revision.is_empty() && revision != variant.revision {
                return claim(&format!("this request names revision '{revision}'"));
            }
        }
        if !files.is_empty() && files.iter().any(|file| file.trim() != variant.repo_file) {
            return claim("this request names a different file in the repo");
        }
    }
    if !payload.files.is_empty()
        && payload
            .files
            .iter()
            .any(|file| file.trim() != variant.repo_file)
    {
        return claim("this request names a different file in the repo");
    }
    if let Some(expected) = payload.expected_sha256.as_deref().map(str::trim) {
        if !expected.is_empty() && !expected.eq_ignore_ascii_case(&variant.sha256) {
            return claim(&format!(
                "this request asserts checksum '{expected}', not the pinned '{}'",
                variant.sha256
            ));
        }
    }
    if let Some(family) = payload.family.as_deref().map(str::trim) {
        if !family.is_empty()
            && sceneworks_core::lora_family::canonical_lora_family(family) != variant.family
        {
            return claim(&format!("this request names family '{family}'"));
        }
    }

    // The registration is now the source of truth for every fetch fact, including the ones the
    // request left out. `expectedSha256` in particular is filled in rather than defaulted to
    // "unverified", which is the whole point.
    payload.repo = None;
    payload.source_url = None;
    payload.source_path = None;
    payload.files = Vec::new();
    payload.source = Some(ModelImportSourceV1::HuggingFace {
        repo: variant.repo.clone(),
        revision: Some(variant.revision.clone()),
        files: vec![variant.repo_file.clone()],
    });
    payload.expected_sha256 = Some(variant.sha256.clone());
    payload.family = Some(variant.family.clone());
    if payload
        .name
        .as_deref()
        .map(str::trim)
        .unwrap_or("")
        .is_empty()
    {
        payload.name = Some(variant.display_name.clone());
    }
    Ok(())
}

pub(crate) async fn queue_model_import_job(
    state: AppState,
    mut payload: ModelImportRequest,
    repair_model_id: Option<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    normalize_import_repo_ids(&mut payload)?;
    let repair = if let Some(model_id) = repair_model_id {
        if payload.model_id.as_deref().is_some_and(|id| id != model_id) {
            return Err(ApiError::bad_request(
                "Repair modelId must match the selected model",
            ));
        }
        if linked_checkpoint_selection(&payload)?.is_some() {
            return Err(ApiError::bad_request(
                "Linked originals use relink/rescan, not managed repair",
            ));
        }
        let model = find_model_by_id(&state, &model_id)
            .await?
            .ok_or_else(|| model_not_found("The imported model to repair no longer exists"))?;
        let binding = sceneworks_core::model_artifacts::provenance::managed_repair_binding(
            &state.settings.data_dir,
            model
                .as_object()
                .ok_or_else(|| ApiError::bad_request("Invalid imported model"))?,
        )
        .map_err(ApiError::bad_request)?;
        payload.model_id = Some(model_id);
        Some(binding)
    } else {
        None
    };
    // A curated managed NVFP4 variant is identified by its `modelId`, and its pin is re-resolved
    // from the registry before anything else reads the body — so no client-supplied repo,
    // revision, file or (absent) checksum can record unverified bytes under that identity.
    resolve_managed_variant_pin(&mut payload)?;
    // Linked (in-place) ownership: an approved library root plus a confined relative path, and
    // never an absolute path (epic 20398, sc-20635). It is an ALTERNATIVE ownership mode rather
    // than another way to FETCH, so it is resolved before source normalisation and excludes it
    // entirely — including sc-20636's discriminated `source`, which only ever names a transfer.
    let linked = linked_checkpoint_selection(&payload)?;
    if let Some((root_id, relative_path)) = linked.clone() {
        if !option_str_is_empty(payload.repo.as_deref())
            || !option_str_is_empty(payload.source_url.as_deref())
            || !option_str_is_empty(payload.source_path.as_deref())
            || payload.source.is_some()
        {
            return Err(ApiError::bad_request(
                "A linked checkpoint import references a library in place; do not also provide a \
                 repo, source URL, source path, or a discriminated source",
            ));
        }
        // Fail fast on an unapproved or unavailable root, and on a path shape the plan store would
        // refuse anyway, so a job is never queued for a source that cannot compile. The same
        // validator `compile_linked` uses, not a second copy of its rules.
        sceneworks_core::checkpoint_plan_store::validate_linked_relative_path(&relative_path)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let data_dir = state.settings.data_dir.clone();
        let probed_root_id = root_id.clone();
        tokio::task::spawn_blocking(move || {
            sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data_dir)
                .resolve_root(&probed_root_id)
        })
        .await
        .map_err(|error| ApiError::internal(format!("Approved root lookup failed: {error}")))?
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
        // The VALIDATED values are what the job carries. Without this the route validated the
        // trimmed pair and then serialised the raw one, so `" root-abc "` passed the API and
        // failed in the worker — and the queued `manifestEntry` (built from the trimmed pair
        // below) and the job payload disagreed about which checkpoint was being imported.
        payload.linked_root_id = Some(root_id);
        payload.linked_relative_path = Some(relative_path);
    }
    // Source normalisation collapses sc-20636's discriminated `source` onto the flat
    // repo/sourceUrl/sourcePath fields. A linked import has no transfer to normalise — it was
    // refused above if it named one — so it names its own provenance and skips the fetch checks.
    let (provenance, revision) = match &linked {
        Some((root_id, relative_path)) => (
            ModelImportProvenance {
                source: "linked-library",
                reference: Some(format!("{root_id}/{relative_path}")),
                ..Default::default()
            },
            None,
        ),
        None => normalize_model_import_source(&mut payload)?,
    };
    if linked.is_none()
        && option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, source path, or an approved library root id \
             with a relative path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    if let Some(repo) = payload.repo.as_deref() {
        payload.repo = Some(validate_huggingface_repo(repo)?.to_owned());
    }
    // Licence acknowledgment, keyed on the repo the import will FETCH (sc-17227). This route had no
    // licence logic at all and did not read
    // the catalog for the source — so `{"repo": "MiniMaxAI/MiniMax-H3"}` pulled the restricted
    // weights from upstream while `POST /api/v1/models/:id/download` was answering 403 for the same
    // bytes. The same predicate the raw jobs route uses, so there is one mechanism, not two.
    ensure_license_acknowledged_for_source(
        &state,
        &[payload.repo.as_deref()],
        payload.source_url.as_deref(),
        payload.license_acknowledged,
    )
    .await?;
    let mut model_type = match payload.model_type.as_deref().map(str::trim) {
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
        let models = model_catalog_snapshot(&state).await?;
        let normalized = normalize_lora_family(&family);
        payload.family = Some(
            if models.iter().any(|model| {
                model
                    .get("family")
                    .and_then(Value::as_str)
                    .is_some_and(|own_family| normalize_lora_family(own_family) == normalized)
            }) {
                normalized
            } else {
                validate_lora_family(&models, &family)?
            },
        );
    }
    // A task/base declaration constrains architecture even if the caller omitted `family`.
    // Local bytes reconcile below; deferred remote/linked inspection must carry this same
    // constraint into the worker instead of treating it as descriptive catalog metadata.
    validate_minimax_h3_import_declaration(&payload)?;
    let declaration = json!({"family":payload.family,"baseModel":payload.base_model,"minimaxH3Partition":payload.minimax_h3_partition});
    if sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(
        declaration.as_object().expect("declaration object"),
    ) {
        payload.family = Some("minimax-h3".to_owned());
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            linked.as_ref().and_then(|(_, relative_path)| {
                FsPath::new(relative_path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
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
    let existing_ids = model_catalog_snapshot(&state)
        .await?
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    if existing_ids.contains(&model_id) && repair.is_none() {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' already exists. Use its explicit repair operation to replace the selection."
        )));
    }
    // A repair never writes over the original installation. Publication changes one catalog
    // selection atomically; old jobs/leases still name their exact original plan and bytes.
    let target_name = if repair.is_some() {
        format!(
            "{}-repair-{}",
            safe_download_dir(&model_id),
            Uuid::new_v4().simple()
        )
    } else {
        safe_download_dir(&model_id)
    };
    // The install id the WORKER will use is the SANITIZED `target_name`, not the raw model id, and
    // it has to be a usable directory name before a job exists: the ingest resolves it through
    // `CheckpointPlanStore::install_dir`, which validates. Without this a caller-supplied `modelId`
    // that sanitizes to (say) a dot-leading name is answered 201 and then dies in the worker with
    // an `InvalidInstallId` the caller never sees. Validated at the resolved value, because that is
    // the string the worker will actually address a path with.
    if let Err(error) = sceneworks_core::checkpoint_plan_store::validate_install_id(&target_name) {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' cannot be installed: {error}"
        )));
    }
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
    if let Some(source_path) = payload.source_path.clone() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("model-uploads")]
        } else {
            allowed_source_roots
        };
        let (source_path, detected) = tokio::task::spawn_blocking(move || {
            validate_import_source_path(&source_path, &allowed_source_roots)?;
            // Compatibility gate (sc-14019): refuse, with the detector's typed
            // reason, any local checkpoint whose shape has no loader today.
            let base_family = import_source_supported(FsPath::new(&source_path))?;
            let detected = detect_model_family(FsPath::new(&source_path))
                .map_err(model_family_inspection_error)?
                .or(base_family);
            sceneworks_core::minimax_h3_import::validate_declared_source_family(
                declaration.as_object().expect("declaration object"),
                detected.as_deref(),
            )
            .map_err(ApiError::bad_request)?;
            Ok::<_, ApiError>((source_path, detected))
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("Model import inspection task failed: {error}"))
        })??;
        payload.family = reconcile_model_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    validate_minimax_h3_import_declaration(&payload)?;
    if payload.family.as_deref() == Some("minimax-h3") || payload.minimax_h3_partition.is_some() {
        model_type = "video".to_owned();
        payload.model_type = Some(model_type.clone());
    }
    // Reuse the selected base's existing co-requisite/download/license workflow. The imported
    // DiT is never a catalog download: only the shared closure and declared load-only companion
    // are provisioned, identically for managed transfers and approved linked sources.
    let import_components = if payload.family.as_deref() == Some("minimax-h3") {
        let base_id = payload.base_model.as_deref().expect("validated H3 base");
        let base = find_model_by_id(&state, base_id)
            .await?
            .ok_or_else(|| ApiError::bad_request("The selected compatible base is unavailable"))?;
        let default = sceneworks_core::model_artifacts::artifact_selection::model_download(&base)
            .ok_or_else(|| {
            ApiError::bad_request("The compatible base has no installable components")
        })?;
        let declaration =
            json!({"baseModel":base_id,"minimaxH3Partition":payload.minimax_h3_partition});
        let default_tier = default.get("variant").and_then(Value::as_str);
        let mut selected_tier = default_tier.unwrap_or("bf16").to_owned();
        let mut rows = sceneworks_core::minimax_h3_import::import_component_downloads(
            declaration.as_object().expect("declaration"),
            &base,
            default_tier,
            std::env::consts::OS,
        )
        .map_err(ApiError::bad_request)?;
        // Reuse a complete resident closure even when it is not the base's default tier.
        for tier in ["q4", "q8", "bf16"] {
            if let Ok(candidate) = sceneworks_core::minimax_h3_import::import_component_downloads(
                declaration.as_object().expect("declaration"),
                &base,
                Some(tier),
                std::env::consts::OS,
            ) {
                if candidate.iter().all(|row| imported_component_presence(&state.settings.data_dir, row)
                    == sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed) {
                    rows = candidate;
                    selected_tier = tier.to_owned();
                    break;
                }
            }
        }
        rows.push(
            sceneworks_core::minimax_h3_import::import_config_download(
                &base,
                &selected_tier,
                std::env::consts::OS,
            )
            .map_err(ApiError::bad_request)?,
        );
        let repos: Vec<_> = rows
            .iter()
            .map(|row| row.get("repo").and_then(Value::as_str))
            .collect();
        ensure_license_acknowledged_for_source(&state, &repos, None, payload.license_acknowledged)
            .await?;
        Some((base, rows, selected_tier))
    } else {
        None
    };
    let timestamp = now_rfc3339();
    // A linked entry names its library by root id and relative path and carries NO install path:
    // nothing is copied, and the plan-driven route reads the plan's own layers. The worker stamps
    // `importPlan.checkpointId` onto this entry once the full-content compile succeeds — a
    // candidate that never compiles never becomes selectable (E7).
    let mut manifest_entry = match &linked {
        Some((root_id, relative_path)) => json!({
            "id": model_id,
            "name": name,
            "type": model_type,
            "source": {
                "provider": provenance.manifest_provider(),
                "rootId": root_id,
                "relativePath": relative_path,
            },
            "files": payload.files.clone(),
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }),
        None => json!({
            "id": model_id,
            "name": name,
            "type": model_type,
            "source": {
                // The provider the SOURCE normalisation resolved, so a Civitai import is recorded
                // as "civitai" rather than as the generic "url" its transfer happens to share.
                // Coarser than provenance for the two non-remote shapes — see `manifest_provider`.
                "provider": provenance.manifest_provider(),
                "repo": payload.repo.clone(),
                "path": source_path_rel,
            },
            "files": payload.files.clone(),
            "paths": {
                "model": target_dir.display().to_string(),
            },
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }),
    };
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
        if let Some(base_model) = &payload.base_model {
            object.insert("baseModel".to_owned(), Value::String(base_model.clone()));
        }
        if let Some(partition) = &payload.minimax_h3_partition {
            object.insert(
                "minimaxH3Partition".to_owned(),
                Value::String(partition.clone()),
            );
        }
        if let Some((_, rows, tier)) = &import_components {
            object.insert("downloads".to_owned(), json!(rows));
            object.insert("importComponentTier".to_owned(), json!(tier));
        }
        apply_model_manifest_defaults(object, &model_type, payload.family.as_deref());
    }
    if let Some(entry) = manifest_entry.as_object() {
        ensure_inherited_component_licenses_acknowledged(
            &state,
            entry,
            payload.license_acknowledged,
        )
        .await?;
    }
    let ownership_mode = payload.ownership_mode;
    let mut payload = to_json_object(&payload)?;
    // The ingest contract the worker executes: managed ownership, plus the provenance the install's
    // persisted plan will carry. Written unconditionally so a legacy flat request is a managed
    // import with recorded provenance too — not a second, unprovenanced code path (AC3).
    payload.insert(
        "ownershipMode".to_owned(),
        serde_json::to_value(ownership_mode)
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    payload.insert("importProvenance".to_owned(), provenance.to_json());
    if let Some(revision) = revision {
        payload.insert("revision".to_owned(), Value::String(revision));
    }
    payload.insert("modelId".to_owned(), manifest_entry["id"].clone());
    payload.insert("modelName".to_owned(), manifest_entry["name"].clone());
    // A linked import installs nothing, so it gets no install directory: handing one to the worker
    // would name a path that must never be created for this ownership mode.
    if linked.is_none() {
        payload.insert(
            "targetDir".to_owned(),
            Value::String(target_dir.display().to_string()),
        );
    }
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    if let Some(binding) = repair {
        payload.insert("repairExpectedBinding".into(), binding);
    }
    if let Some((base, rows, _)) = import_components {
        // sc-26687 / sc-26690: the SAME inherited-component gate at the DOWNLOAD boundary. Closing
        // only the generation door would still let an unacknowledged component's bytes land, so a
        // resident component is fetched only once its own terms are acknowledged. Idempotent with
        // the generation-admission gate in `create_generation_job_with_status`.
        if let Some(entry) = payload.get("manifestEntry").and_then(Value::as_object) {
            ensure_inherited_component_licenses_acknowledged(
                &state,
                entry,
                payload
                    .get(LICENSE_ACKNOWLEDGED_PAYLOAD_KEY)
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            )
            .await?;
        }
        for row in rows {
            // The import worker fetches this literal config through the shared installer only if
            // inspection finds no source-owned claim. It writes no primary download receipt.
            if row.get("componentId").and_then(Value::as_str) == Some("import_transformer_config") {
                continue;
            }
            if imported_component_presence(&state.settings.data_dir, &row)
                == sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed {
                continue;
            }
            let component = build_model_download_job_payload(
                &base,
                &model_id,
                &row,
                None,
                false,
                payload
                    .get("licenseAcknowledged")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                &state.settings.data_dir,
            )?;
            create_generation_job(
                state.clone(),
                JobType::ModelDownload,
                None,
                None,
                component,
                "auto".to_owned(),
            )
            .await?;
        }
    }
    let job = create_generation_job(
        state,
        JobType::ModelImport,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

// The detector establishes architecture, never whether a fine-tune was trained for first/last
// frames, references, or both. A caller must declare that compatibility before an H3 job is queued;
// remote bytes are checked again by the shared worker importer after detection.
pub(super) fn validate_minimax_h3_import_declaration(
    payload: &ModelImportRequest,
) -> Result<(), ApiError> {
    let family = payload.family.as_deref();
    let base = payload.base_model.as_deref();
    let partition = payload.minimax_h3_partition.as_deref();
    let declares_h3 = partition.is_some() || matches!(base, Some("minimax_h3" | "minimax_h3_ref"));
    if family != Some("minimax-h3") && !declares_h3 {
        return Ok(());
    }
    if family.is_some_and(|family| family != "minimax-h3") {
        return Err(ApiError::bad_request(
            "baseModel and minimaxH3Partition declare MiniMax-H3 compatibility, but family is not minimax-h3",
        ));
    }
    let declaration = json!({"baseModel":base,"minimaxH3Partition":partition});
    sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(
        declaration.as_object().expect("declaration object"),
    )
    .map(|_| ())
    .map_err(ApiError::bad_request)
}

pub(crate) async fn model_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(ModelImportRequest, PathBuf), ApiError> {
    let mut payload = ModelImportRequest {
        ownership_mode: OwnershipModeV1::default(),
        source: None,
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        linked_root_id: None,
        linked_relative_path: None,
        files: Vec::new(),
        family: None,
        base_model: None,
        minimax_h3_partition: None,
        expected_sha256: None,
        license_acknowledged: false,
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
                "baseModel" => payload.base_model = Some(value.to_owned()),
                "minimaxH3Partition" => payload.minimax_h3_partition = Some(value.to_owned()),
                "repo" => payload.repo = Some(value.to_owned()),
                "sourceUrl" => payload.source_url = Some(value.to_owned()),
                // The multipart form accepts `repo`/`sourceUrl` too, so it can reach a
                // licence-restricted repo exactly like the JSON body and needs the same way to
                // assert the acknowledgment (sc-17227). Anything other than "true" leaves it false
                // — the assertion is affirmative or it is not made.
                "licenseAcknowledged" => {
                    payload.license_acknowledged = value.eq_ignore_ascii_case("true")
                }
                // The multipart form takes the same two epic-20398 fields as the JSON body, decoded
                // by the same serde impls, so an unknown ownership mode or malformed discriminated
                // source is refused here exactly as it is there rather than being ignored (E7).
                "ownershipMode" => {
                    payload.ownership_mode = serde_json::from_value(Value::String(value.to_owned()))
                        .map_err(|error| {
                            ApiError::bad_request(format!("Invalid ownershipMode: {error}"))
                        })?
                }
                "source" => {
                    payload.source = Some(serde_json::from_str(value).map_err(|error| {
                        ApiError::bad_request(format!("Invalid import source: {error}"))
                    })?)
                }
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
    crate::uploads::stage_upload(
        state,
        field,
        "model-uploads",
        Some(filename),
        max_model_upload_bytes(),
        || {
            format!(
                "Uploaded model file exceeds the {} limit",
                format_bytes(max_model_upload_bytes() as u64)
            )
        },
    )
    .await
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
