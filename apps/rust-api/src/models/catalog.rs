use super::*;

/// Import request types are an API policy, not the builtin catalog's model-type vocabulary.
/// A builtin vector model does not provide a generic vector checkpoint import path.
pub(super) const ALLOWED_MODEL_TYPES: &[&str] = &["image", "video", "audio", "utility"];

#[cfg(test)]
#[test]
fn import_type_vocabulary_remains_separate_from_builtin_types() {
    assert_eq!(ALLOWED_MODEL_TYPES, ["image", "video", "audio", "utility"]);
    assert!(!ALLOWED_MODEL_TYPES.contains(&"vector"));
}
pub(crate) const MODEL_CATALOG_PROBE_CONCURRENCY: usize = 16;
pub(super) static MODEL_CATALOG_PROBE_PERMITS: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> =
    std::sync::OnceLock::new();

/// Generation-keyed, process-local snapshot of the model catalog after its
/// filesystem install-state sweep.
///
/// A separate async serializer makes concurrent cold `/models`, global preset,
/// project preset, and job-validation requests join one sweep without holding
/// the short synchronous state lock across filesystem work. `invalidate`
/// advances the generation and clears the snapshot in that same critical
/// section. Publication validates its generation under the same lock, so a
/// writer racing a cold build forces a rebuild instead of publishing or
/// returning stale state. Errors are never cached.
#[derive(Default)]
pub(crate) struct ModelCatalogCache {
    pub(super) state: Mutex<ModelCatalogCacheState>,
    pub(super) build_serializer: tokio::sync::Mutex<()>,
}

#[derive(Default)]
pub(super) struct ModelCatalogCacheState {
    pub(super) generation: u64,
    pub(super) snapshot: Option<(u64, &'static str, Arc<Vec<Value>>)>,
    /// Last observed resolved-cache publish key. `None` until the first observation, which is why
    /// the first reading never counts as a change — an API that has not looked yet has nothing to
    /// have gone stale against.
    pub(super) local_tier_key: Option<u64>,
}

impl ModelCatalogCache {
    pub(crate) fn invalidate(&self) {
        let mut state = self.state.lock();
        state.generation = state.generation.wrapping_add(1);
        state.snapshot = None;
    }

    /// Invalidates the snapshot when the resolved cache's published set has moved since the last
    /// catalog read (sc-19712 F-4).
    ///
    /// Every other invalidation site is a mutation this process performed, but promotions are
    /// published by the **worker**, in another process, so no in-process signal exists. Without
    /// this the catalog kept serving a model's pre-promotion availability indefinitely — and
    /// `modelLibraryUnavailable()` keys the Model Manager's blocked state on exactly that field,
    /// so a model that had just completed a job with the drive unplugged was still presented as
    /// needing the drive reconnected.
    ///
    /// A key check, not a timer: a TTL would leave the same wrong answer up for its whole window
    /// and rebuild pointlessly for the rest of time. The key only moves when a bundle is published
    /// or withdrawn, so a rebuild happens once per real change and never on a load.
    pub(crate) fn note_local_tier_key(&self, key: u64) {
        let mut state = self.state.lock();
        if state.local_tier_key == Some(key) {
            return;
        }
        let first_observation = state.local_tier_key.is_none();
        state.local_tier_key = Some(key);
        if first_observation {
            return;
        }
        state.generation = state.generation.wrapping_add(1);
        state.snapshot = None;
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.state.lock().generation
    }
}

pub(crate) async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let mut models = model_catalog_sized(&state).await?;
    // Presentation-only hide for converted builtins. Keep the internal snapshot intact so
    // explicit-id admission and unfiltered license/source checks still see the model.
    models.retain(|model| {
        let Some(id) = model.get("id").and_then(Value::as_str) else {
            return true;
        };
        sceneworks_core::registry::builtin_model_spec(id).is_none()
            || model.get("hidden").and_then(Value::as_bool) != Some(true)
    });
    Ok(Json(models))
}

/// Select from the current shared catalog before cloning. Linked-source availability is probed
/// for the selected row on every call, just as it is for the full catalog response.
pub(crate) async fn find_model_catalog_entry(
    state: &AppState,
    predicate: impl Fn(&Value) -> bool,
) -> Result<Option<Value>, ApiError> {
    let snapshot = model_catalog_snapshot(state).await?;
    let Some(model) = snapshot.iter().find(|entry| predicate(entry)) else {
        return Ok(None);
    };
    let mut selected = vec![model.clone()];
    crate::model_sources::refresh_live_external_availability(
        &state.settings.data_dir,
        &mut selected,
    )
    .await?;
    Ok(selected.pop())
}

pub(crate) async fn find_model_by_id(
    state: &AppState,
    id: &str,
) -> Result<Option<Value>, ApiError> {
    find_model_catalog_entry(state, |entry| {
        entry.get("id").and_then(Value::as_str) == Some(id)
    })
    .await
}

pub(crate) fn model_not_found(detail: impl Into<String>) -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        detail: detail.into(),
        context: None,
        code: None,
    }
}

pub(crate) async fn require_model_by_id(state: &AppState, id: &str) -> Result<Value, ApiError> {
    find_model_by_id(state, id)
        .await?
        .ok_or_else(|| model_not_found("Model not found"))
}

/// Catalog without live Hugging Face size estimation: download sizes fall back to
/// manifest metadata only. This is the right call for job validation, LoRA/preset
/// CRUD, download/convert job creation, and delete — none of which read the
/// byte-accurate download size — so an unreachable huggingface.co can't stall
/// those paths (sc-4169).
pub(crate) async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let mut models = model_catalog_snapshot(state).await?.as_ref().clone();
    crate::model_sources::refresh_live_external_availability(&state.settings.data_dir, &mut models)
        .await?;
    Ok(models)
}

/// Catalog with live Hugging Face download-size estimates (negative-cached on
/// failure). Reserved for `GET /models`, the one surface that displays
/// download sizes.
pub(crate) async fn model_catalog_sized(state: &AppState) -> Result<Vec<Value>, ApiError> {
    // Preserve SC-14800's cold-path overlap: Hugging Face metadata estimation
    // begins from the cheap manifest inputs while the shared install-state
    // snapshot is being built (or joined) independently.
    let size_estimates = estimate_current_model_catalog_sizes(state);
    let snapshot = model_catalog_snapshot(state);
    let (size_estimates, snapshot) = tokio::join!(size_estimates, snapshot);
    let size_estimates = size_estimates?;
    let mut models = snapshot?.as_ref().clone();
    crate::model_sources::refresh_live_external_availability(&state.settings.data_dir, &mut models)
        .await?;
    let selection_catalog = models.clone();
    for model in &mut models {
        let context = model_download_context(model)?;
        let live_estimate = context.as_ref().and_then(|context| {
            size_estimates
                .get(&(context.repo.clone(), context.files.clone()))
                .copied()
                .flatten()
        });
        apply_model_catalog_size_fields(model, context.as_ref(), live_estimate)?;
        apply_runtime_text_encoder_options(
            model,
            &selection_catalog,
            &state.settings.data_dir,
            &state.settings.external_model_roots,
        )?;
        annotate_commercial_use_alternatives(model, &selection_catalog);
    }
    Ok(models)
}

/// Resolve a declared `commercialUse` pointer against this catalog (sc-22998): a model whose
/// weights are not commercially eligible gets `commercialUse.alternatives`, the ids of its
/// `alternativeFamily` that are themselves eligible here — YuE2 lists the YuE1 entries. The declared
/// block is otherwise passed through untouched, and nothing here reroutes anything.
///
/// This EXPOSES the verdict; it enforces nothing. The routes that enforce it (sc-22999) are the
/// YuE2 job route and its execution-time eligibility check (`yue2_jobs`) and timeline exports,
/// which also persist the licence acknowledgment and noncommercial policy into provenance.
pub(super) fn annotate_commercial_use_alternatives(model: &mut Value, catalog: &[Value]) {
    if model.get("commercialUse").is_none() {
        return;
    }
    let Some(id) = model.get("id").and_then(Value::as_str).map(str::to_owned) else {
        return;
    };
    if let Ok(sceneworks_core::model_usage_policy::CommercialUseVerdict::Refused {
        alternatives,
        ..
    }) = sceneworks_core::model_usage_policy::commercial_use_verdict(catalog, &id)
    {
        if let Some(block) = model
            .get_mut("commercialUse")
            .and_then(Value::as_object_mut)
        {
            block.insert("alternatives".to_owned(), json!(alternatives));
        }
    }
}

/// Add runtime-only text-encoder choices to the public model catalog. The worker owns enumeration so
/// the API and generation resolver share one completeness predicate. This runs on the response clone,
/// outside the cached catalog snapshot: an operator who stages a complete alternate and refreshes
/// Models sees it without a server restart, while no repo/revision/download metadata is persisted.
pub(super) fn apply_runtime_text_encoder_options(
    model: &mut Value,
    catalog: &[Value],
    data_dir: &FsPath,
    external_roots: &[PathBuf],
) -> Result<(), ApiError> {
    let mut options =
        sceneworks_worker::image_text_encoder_options(catalog, model, data_dir, external_roots)
            .into_iter()
            .map(|option| serde_json::to_value(option).expect("text encoder option serializes"))
            .collect::<Vec<_>>();
    if options.is_empty() {
        let adapter = model
            .get("adapter")
            .and_then(Value::as_str)
            .unwrap_or_default();
        options.extend(
            sceneworks_worker::text_encoder_options_for_adapter(adapter, data_dir)
                .into_iter()
                .map(|option| {
                    serde_json::to_value(option).expect("video text encoder option serializes")
                }),
        );
    }
    set_runtime_text_encoder_options(model, options)
}

/// Resolve an authored Image Studio encoder selection through the worker-owned descriptor contract.
/// Only the opaque id comes from the client. The path-bearing resolution is rebuilt from a fresh
/// server catalog at every create/retry/duplicate boundary and is never trusted as request input.
pub(crate) async fn resolve_selected_image_text_encoder(
    state: &AppState,
    job_payload: &JsonObject,
    model_id: &str,
    manifest_entry: &mut Value,
) -> Result<(), ApiError> {
    let selected = match job_payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("textEncoderModel"))
    {
        None => {
            if let Some(object) = manifest_entry.as_object_mut() {
                object.remove("resolvedTextEncoder");
            }
            return Ok(());
        }
        Some(Value::String(id)) if id == "default" => {
            if let Some(object) = manifest_entry.as_object_mut() {
                object.remove("resolvedTextEncoder");
            }
            return Ok(());
        }
        Some(Value::String(id)) => id.as_str(),
        Some(_) => return Err(ApiError::bad_request(
            "advanced.textEncoderModel must be a string option id returned by GET /api/v1/models",
        )),
    };

    let catalog = model_catalog_snapshot(state).await?;
    let catalog_model = catalog
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .unwrap_or(manifest_entry);
    let resolution = sceneworks_worker::resolve_image_text_encoder_selection(
        catalog.as_ref(),
        catalog_model,
        selected,
        &state.settings.data_dir,
        &state.settings.external_model_roots,
    )
    .map_err(ApiError::bad_request)?
    .ok_or_else(|| ApiError::internal("non-default text encoder resolved as default"))?;
    let object = manifest_entry
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    object.insert("resolvedTextEncoder".to_owned(), resolution);
    Ok(())
}

pub(super) fn set_runtime_text_encoder_options(
    model: &mut Value,
    options: Vec<Value>,
) -> Result<(), ApiError> {
    let object = model
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    if options.is_empty() {
        object.remove("textEncoderOptions");
    } else {
        object.insert("textEncoderOptions".to_owned(), Value::Array(options));
    }
    Ok(())
}

pub(crate) async fn model_catalog_snapshot(state: &AppState) -> Result<Arc<Vec<Value>>, ApiError> {
    let backend = crate::generation::enqueue_backend(state);
    // Before trusting the cached snapshot, check whether the resolved cache's published set moved
    // under us — the worker publishes promotions in another process, so this is the only signal
    // the API gets (sc-19712 F-4). It is a handful of stats, cheap enough to pay per read.
    crate::model_sources::note_local_tier_freshness(state);
    {
        let cache_state = state.model_catalog_cache.state.lock();
        if let Some((snapshot_generation, snapshot_backend, snapshot)) =
            cache_state.snapshot.as_ref()
        {
            if *snapshot_generation == cache_state.generation && *snapshot_backend == backend {
                return Ok(snapshot.clone());
            }
        }
    }

    // sc-28037 (GitHub #3235): the build runs as its own task, not inside this
    // request's future. Its filesystem sweep is `spawn_blocking` work that a
    // dropped future cannot stop, so a client that disconnected mid-build used
    // to release the serializer while that sweep kept running, and the next
    // caller started a second sweep on top of it. Detached, an abandoned build
    // still finishes and publishes, and every later caller joins its result.
    let builder_state = state.clone();
    tokio::spawn(
        async move { build_and_publish_model_catalog_snapshot(&builder_state, backend).await },
    )
    .await
    .map_err(|err| ApiError::internal(format!("model catalog build task failed: {err}")))?
}

pub(super) async fn build_and_publish_model_catalog_snapshot(
    state: &AppState,
    backend: &'static str,
) -> Result<Arc<Vec<Value>>, ApiError> {
    // Serialize only builders. The synchronous generation/snapshot state lock
    // is never held across the expensive filesystem sweep.
    let _build_guard = state.model_catalog_cache.build_serializer.lock().await;
    loop {
        let generation = {
            // Another builder may have populated the cache while this caller
            // waited for the async serializer.
            let cache_state = state.model_catalog_cache.state.lock();
            if let Some((snapshot_generation, snapshot_backend, snapshot)) =
                cache_state.snapshot.as_ref()
            {
                if *snapshot_generation == cache_state.generation && *snapshot_backend == backend {
                    return Ok(snapshot.clone());
                }
            }
            cache_state.generation
        };

        let mut models = build_model_catalog_snapshot(state).await?;
        for model in &mut models {
            project_moss_edit_surface(model, backend);
        }
        let snapshot = Arc::new(models);

        #[cfg(test)]
        state.test_hooks.pause_model_catalog_before_publish().await;

        let mut cache_state = state.model_catalog_cache.state.lock();
        if cache_state.generation == generation {
            cache_state.snapshot = Some((generation, backend, snapshot.clone()));
            return Ok(snapshot);
        }
        drop(cache_state);
        // A model writer completed while the filesystem sweep was running.
        // Keep the async serializer and rebuild at the new generation. The
        // stale result was never published or returned.
    }
}

pub(super) fn project_moss_edit_surface(model: &mut Value, backend: &str) {
    let Some(id) = model.get("id").and_then(Value::as_str) else {
        return;
    };
    if !sceneworks_core::moss_image::is_moss(id) {
        return;
    }
    let available = sceneworks_core::moss_image::edit_backend_eligible(id, backend);
    let Some(object) = model.as_object_mut() else {
        return;
    };
    let capabilities = object.entry("capabilities").or_insert_with(|| json!([]));
    let Some(capabilities) = capabilities.as_array_mut() else {
        return;
    };
    capabilities.retain(|mode| mode != "edit_image" && mode != "image_edit");
    if available {
        capabilities.push(json!("edit_image"));
    }
}

/// Withdraw a synthesized LoRA advertisement that no lane on THIS deployment can honour (the
/// sc-15328 class).
///
/// An imported / fine-tuned image model has no manifest row: `apply_model_manifest_defaults`
/// synthesizes `loraCompatibility.families = [family]` from the family token alone. For some
/// families that is a promise nothing keeps — currently a Mage-Flow fine-tune on either native
/// backend. Imported Krea 2 used to be another case, but both native single-file entrypoints now
/// accept adapters (sc-18480), so its synthesized advertisement remains intact.
/// Left standing, the picker offers adapters, `validate_lora_specs_for_model` passes, the job is
/// created, and NO worker claims it: it sits on "Waiting for an available GPU worker" forever, with
/// no error and no terminal state. That is strictly worse than a rejection.
///
/// Applied HERE — on the catalog projection every read goes through — rather than baked into the
/// stored manifest at import time, because the verdict is a property of the DEPLOYMENT, not of the
/// checkpoint. Keeping this a deployment projection also makes future backend-specific capability
/// changes safe when a data dir moves between platforms.
///
/// 🔴 The withdrawal is an EXPLICIT EMPTY `families` array, never `remove("loraCompatibility")`.
/// Removing the key is a no-op: `families_from_value_chain` (lib.rs) falls back to the top-level
/// `family` field, which an imported entry must carry for routing — so the strip sc-15328 shipped
/// changed nothing and its lane still hangs. An empty array is non-null, so it short-circuits that
/// fallback and `validate_lora_specs_for_model` refuses the submission LOUDLY and terminally with
/// "has no declared LoRA families". `supported: false` is the web's signal to fail CLOSED, because
/// `loraMatchesModel` treats an empty family set as "cannot gate" and would otherwise stay
/// permissive and keep offering every LoRA.
/// Binds [`apply_imported_lora_advertisement_for_lanes`] to the lanes THIS build can run: macOS
/// ships the in-process MLX worker and no candle engine; Windows/Linux/Docker ship candle and no
/// MLX. The verdict is derived from the same per-lane request gates the scheduler uses, so the
/// advertisement and claim behavior stay aligned.
///
/// The lane split is a ONE-LINE binding here and a parameter below precisely so the behaviour is
/// testable on both lanes from either platform.
pub(super) fn apply_imported_lora_advertisement(object: &mut JsonObject) {
    // Deliberately the BUILD's linked engines, not `generation::enqueue_backend` — see that
    // function's "Scope" note. The pair is consulted asymmetrically below (a withdrawal crosses
    // lanes, a positive advertisement never does), which a one-hot routing answer cannot express,
    // and a macOS build links no candle engine to derive a positive candle verdict from.
    let mlx_lane = cfg!(target_os = "macos");
    // Resolved as a VERDICT rather than by checking whether the first pass wrote anything:
    // `apply_model_manifest_defaults` can already have synthesized a `loraCompatibility` object, so
    // "the key exists" says nothing about whether this projection decided anything.
    let verdict = imported_lora_advertisement_verdict(object, mlx_lane, !mlx_lane).or_else(|| {
        // This build's lane had NO OPINION, which is not the same as "adapters are fine". Leaving
        // the entry untouched restores the permissive `family` fallback — the precise silent
        // re-advertisement sc-15328 fixed, where the API accepted a LoRA submission no worker could
        // claim and the job queued forever with no error. A generated Mage full fine-tune on any
        // non-macOS deployment is exactly that shape: candle registers no `mage-flow` imported
        // provider, so only MLX can say "serves the family, refuses adapters".
        //
        // So a WITHDRAWAL crosses lanes; a positive advertisement never does. Offering adapters
        // this build cannot claim is the same hang pointed the other way, which is what the filter
        // to the refusing verdict prevents.
        imported_lora_advertisement_verdict(object, !mlx_lane, mlx_lane).filter(|serves| !serves)
    });
    if let Some(serves_loras) = verdict {
        write_imported_lora_advertisement(object, serves_loras);
    }
}

/// Test-facing lane-scoped projection: resolve ONE lane pair and write its verdict verbatim, with no
/// cross-lane fallback. Production goes through [`apply_imported_lora_advertisement`], which adds
/// the withdrawal fallback; keeping this seam separate is what lets a test state each lane's own
/// truth from either platform without the fallback blurring the two.
#[cfg(test)]
pub(super) fn apply_imported_lora_advertisement_for_lanes(
    object: &mut JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) {
    let Some(serves_loras) =
        imported_lora_advertisement_verdict(object, mlx_lane_available, candle_lane_available)
    else {
        return;
    };
    write_imported_lora_advertisement(object, serves_loras);
}

/// The advertisement verdict for one lane pair, with no mutation: `Some(true)` serves adapters,
/// `Some(false)` serves the family and refuses them, `None` is "not on this seam at all" (a builtin,
/// a non-image row, or a family no supplied lane registers).
pub(super) fn imported_lora_advertisement_verdict(
    object: &JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) -> Option<bool> {
    // An imported plan-backed H3 video entry serves the family's adapter surface when ANY valid
    // supported shape is claimable with a compatible adapter: every UI mode is probed with its
    // real conditioning through the same predicates the support projection uses, so a
    // reference-partition (Ref2VA) import is advertised through its reference shape rather than
    // withdrawn by a probe no shape of it can answer.
    if object.get("type").and_then(Value::as_str) == Some("video")
        && sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(object)
    {
        return Some(
            sceneworks_core::jobs_store::imported_video_adapter_surface_claimable(
                object,
                mlx_lane_available,
                candle_lane_available,
            ),
        );
    }
    if object.get("type").and_then(Value::as_str) != Some("image") {
        return None;
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|id| !id.is_empty())?;
    let family = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)?;
    let source = object
        .get("importSourceShape")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|source| {
            matches!(
                *source,
                "transformer_file" | "fused_checkpoint" | "transformer_directory" | "comfy_ui_tree"
            )
        })?;
    sceneworks_core::jobs_store::imported_image_model_lora_advertisement(
        &id,
        &family,
        source,
        mlx_lane_available,
        candle_lane_available,
    )
}

/// Write the verdict onto the entry. Split from the resolution above so the production binding can
/// resolve twice — its own lane, then a cross-lane withdrawal — without projecting twice.
pub(super) fn write_imported_lora_advertisement(object: &mut JsonObject, serves_loras: bool) {
    let Some(family) = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)
    else {
        return;
    };
    let compatibility = object
        .entry("loraCompatibility".to_owned())
        .or_insert_with(|| json!({}));
    if !compatibility.is_object() {
        *compatibility = json!({});
    }
    let Some(compatibility) = compatibility.as_object_mut() else {
        return;
    };
    if serves_loras {
        let families = compatibility
            .entry("families".to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        if !families.is_array() {
            *families = Value::Array(Vec::new());
        }
        if let Some(families) = families.as_array_mut() {
            if !families
                .iter()
                .any(|value| value.as_str() == Some(family.as_str()))
            {
                families.push(Value::String(family));
            }
        }
        compatibility.insert("supported".to_owned(), Value::Bool(true));
    } else {
        // Explicit empty wins over permissive family fallback in the validator.
        compatibility.insert("families".to_owned(), Value::Array(Vec::new()));
        compatibility.insert("supported".to_owned(), Value::Bool(false));
    }
}

/// A manifest that DECLARES `loraCompatibility.families: []` (as `qwen_image_2_1` did until
/// sc-24156 gave it the `qwen-image-2-1` family) is refused by `validate_lora_specs_for_model`
/// with "has no declared LoRA families" — but the web's `loraMatchesModel` reads an empty family set as "cannot gate" and stays
/// permissive, so it offered every LoRA and auto-applied the Krea `image_edit` LoRA in edit mode.
/// Stamp the same `supported: false` the imported withdrawal writes, so the web fails closed on
/// exactly the models the API refuses. An explicit `supported` already on the entry is kept.
pub(super) fn mark_empty_lora_advertisement_unsupported(object: &mut JsonObject) {
    let Some(compatibility) = object
        .get_mut("loraCompatibility")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    if compatibility
        .get("families")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        compatibility
            .entry("supported".to_owned())
            .or_insert(Value::Bool(false));
    }
}

/// Project the exact active-provider surface for a stamped imported source shape. Family siblings
/// are never unioned: a Krea transformer file, SDXL fused checkpoint, Mage directory, and external
/// ComfyUI tree each see only registrations for their structural loader.
pub(super) fn apply_imported_provider_surface(object: &mut JsonObject) {
    // The BUILD's linked engines, like its `apply_imported_lora_advertisement` sibling — not
    // `generation::enqueue_backend`. See that function's "Scope" note.
    let mlx_lane = cfg!(target_os = "macos");
    apply_imported_provider_surface_for_lanes(object, mlx_lane, !mlx_lane);
}

/// The Mac and off-Mac support blocks for an imported, plan-backed video entry (sc-26680).
///
/// The id-keyed pass in `apply_mac_and_mlx_fields` asks the claim predicates about a probe carrying
/// no manifest entry, so for an imported id every plan-backed predicate answers `false` and the
/// Video Studio would hide a model both native lanes serve. This replaces both blocks with the
/// entry-keyed verdicts, derived from the SAME predicates the scheduler and the no-lane gate ask
/// about a submitted job. Emitted on every platform, like the id-keyed blocks it replaces.
pub(super) fn apply_imported_video_support_surface(object: &mut JsonObject) {
    // The existing picker must not synthesize a q4/q8 shared precision for an import whose
    // admitted closure is bf16 (or vice versa). Imported DiT precision remains source-owned.
    let tiers = sceneworks_core::minimax_h3_import::selected_import_component_tier(object)
        .map(|tier| vec![tier])
        .unwrap_or_default();
    object.insert("runtimeQuantTiers".to_owned(), json!(tiers));
    if let Ok(mac_support) =
        serde_json::to_value(sceneworks_core::jobs_store::imported_video_model_mac_support(object))
    {
        object.insert("macSupport".to_owned(), mac_support);
    }
    if let Ok(candle_support) = serde_json::to_value(
        sceneworks_core::jobs_store::imported_video_model_candle_support(object),
    ) {
        object.insert("candleSupport".to_owned(), candle_support);
    }
}

/// The product surface of an imported, plan-backed checkpoint (sc-26687): what was imported
/// (source variant), what it can do (supported operations), where it can run (supported backends),
/// whether its components are assembled (component readiness), and what to do about the ones that
/// are not (actionable repair steps).
///
/// ONE derivation behind Model Manager, Video Studio, the API and MCP, so the repair a card offers
/// is the same repair the API describes. Derived server-side from the entry and its bound plan; the
/// client never supplies any of it. Written on the response clone beside the neighbouring
/// `modelAvailability` / `modelResolution` / `cacheEligibility` projections — these describe this
/// deployment and this install state, so they are runtime projections rather than persisted
/// manifest keys.
pub(super) fn apply_imported_checkpoint_product_surface(
    object: &mut JsonObject,
    data_dir: &FsPath,
    source_present: Option<bool>,
) {
    if object.get("catalogScope").and_then(Value::as_str) == Some("builtin") {
        return;
    }
    if sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(object).is_none() {
        return;
    }
    object.insert(
        "importedSource".to_owned(),
        imported_source_description(object),
    );
    object.insert(
        "supportedOperations".to_owned(),
        Value::Array(imported_supported_operations(object)),
    );
    // Omitted rather than emptied when the catalog carries NO per-lane verdict at all. An empty
    // array is a real "neither backend here", and saying that about an entry nothing has judged
    // would read as a refusal the scheduler never made.
    if let Some(backends) = imported_supported_backends(object) {
        object.insert("supportedBackends".to_owned(), Value::Array(backends));
    }
    let (readiness, repairs) = imported_component_readiness(object, data_dir, source_present);
    object.insert("componentReadiness".to_owned(), readiness);
    object.insert("repairActions".to_owned(), Value::Array(repairs));
}

/// Which source variant was imported and how it is held.
///
/// `sourceVariant` is the axis that actually distinguishes one imported checkpoint from its
/// siblings. For MiniMax-H3 that is the task partition the checkpoint DECLARED (fl2va / ref2va /
/// hybrid) — filenames and directory names prove nothing about task compatibility, which is
/// precisely why the declaration is the variant. An import carrying no such declaration has only
/// the structural source shape to distinguish it by, so that is what it reports.
pub(super) fn imported_source_description(object: &JsonObject) -> Value {
    let checkpoint_id =
        sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(object).unwrap_or_default();
    // Ownership rides the checkpoint id's own prefix — the convention every reader of
    // `importPlan.checkpointId` already speaks (`apps/web/src/checkpointLibrary.js`
    // `modelOwnership`), so this and the web's linked/managed split cannot drift apart.
    let ownership_mode = if checkpoint_id.starts_with("linked/") {
        "linked"
    } else {
        "managed"
    };
    let declared_partition =
        sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(object)
            .ok()
            .map(|partition| partition.as_str());
    let source_shape = object
        .get("importSourceShape")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    // The assembly names the imported DiT under `transformer`, except on a Ref2VA import where the
    // DiT is the reference partition and lives under `transformer_ref`. Find whichever binding the
    // plan actually owns rather than hard-coding one key.
    let plan_binding = sceneworks_core::minimax_h3_import::entry_assembly(object)
        .and_then(|assembly| assembly.get("components"))
        .and_then(Value::as_object)
        .and_then(|components| {
            components
                .iter()
                .find(|(_, binding)| binding.get("source").and_then(Value::as_str) == Some("plan"))
        });
    let transformer_directory = plan_binding
        .and_then(|(_, binding)| binding.get("directory"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let sharded = plan_binding
        .and_then(|(_, binding)| binding.get("layerIds"))
        .and_then(Value::as_array)
        .is_some_and(|layers| layers.len() > 1);
    json!({
        "family": object.get("family").cloned().unwrap_or(Value::Null),
        "sourceShape": source_shape,
        "ownershipMode": ownership_mode,
        "sourceVariant": declared_partition.or(source_shape),
        "declaredPartition": declared_partition,
        "baseModel": object.get("baseModel").cloned().unwrap_or(Value::Null),
        "componentTier": object.get("importComponentTier").cloned().unwrap_or(Value::Null),
        "transformerDirectory": transformer_directory,
        "sharded": sharded,
    })
}

/// The operations this import serves, from its EXPLICIT task contract.
///
/// An H3 import answers from the partition it declared, because identical shapes do not prove
/// partition compatibility. Any other import's operations are whatever its registered route
/// projected onto `capabilities`, which is already the operation contract for it — and the field
/// the recipe gate (`recipe_presets::validate_recipe_preset_model_workflow`) reads, so the two can
/// never disagree about what a saved recipe may ask for.
pub(super) fn imported_supported_operations(object: &JsonObject) -> Vec<Value> {
    if let Ok(partition) =
        sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(object)
    {
        let mut operations = Vec::new();
        if partition.supports_base() {
            operations.extend(["text_to_video", "image_to_video", "first_last_frame"]);
        }
        if partition.supports_references() {
            operations.push("reference_to_video");
        }
        return operations
            .into_iter()
            .map(|operation| Value::String(operation.to_owned()))
            .collect();
    }
    object
        .get("capabilities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The native backends this deployment can actually claim the import on, read off the same
/// per-lane verdicts the scheduler's claim predicates and the Video Studio gate consult.
///
/// `None` when the entry carries neither verdict — "nobody has judged this", which is a different
/// answer from `Some(vec![])` ("both lanes looked and neither will claim it").
pub(super) fn imported_supported_backends(object: &JsonObject) -> Option<Vec<Value>> {
    let verdict = |key: &str| {
        object
            .get(key)
            .and_then(|block| block.get("supported"))
            .and_then(Value::as_bool)
    };
    let (mac, candle) = (verdict("macSupport"), verdict("candleSupport"));
    if mac.is_none() && candle.is_none() {
        return None;
    }
    let mut backends = Vec::new();
    if mac == Some(true) {
        backends.push(Value::String("mlx".to_owned()));
    }
    if candle == Some(true) {
        backends.push(Value::String("candle-cuda".to_owned()));
    }
    Some(backends)
}

/// Whether this import's assembly is COMPLETE here, component by component, plus the repair that
/// would make each missing component complete.
///
/// The imported DiT is the entry's own plan/source closure, independent of aggregate cache state. The
/// rest are the resident components the assembly reuses from the declared compatible base, each
/// resolved through the shared component-presence probe — never through a filename or a repository
/// allowlist.
pub(super) fn imported_component_readiness(
    object: &JsonObject,
    data_dir: &FsPath,
    source_present: Option<bool>,
) -> (Value, Vec<Value>) {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;

    let mut components = Vec::new();
    let mut repairs = Vec::new();
    let state_name = |presence: CoRequisitePresence| match presence {
        CoRequisitePresence::Installed => "installed",
        CoRequisitePresence::Incomplete => "incomplete",
        CoRequisitePresence::Absent => "missing",
    };

    // A declaration the contract refuses (an absent or incompatible `baseModel` / task partition)
    // cannot be repaired by downloading anything: it needs the user to declare the compatible base.
    let declaration_ok = sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(object)
        .is_ok()
        || !sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(object);
    if !declaration_ok {
        repairs.push(json!({
            "id": "declare-compatible-base",
            "kind": "select-base",
            "label": "Declare the compatible base model and task partition (fl2va, ref2va, or hybrid)",
        }));
    }

    let dit_present = source_present.unwrap_or_else(|| imported_source_present(object, data_dir));
    components.push(json!({
        "componentId": "transformer",
        "role": "imported_dit",
        "source": "plan",
        "model": Value::Null,
        "present": dit_present,
        "state": if dit_present { "installed" } else { "missing" },
    }));
    if !dit_present {
        // The damaged SOURCE is never re-fetched by the stock model downloader: a managed import
        // must go through the checkpoint import panel again (the inspector re-validates and
        // re-ingests the bytes under managed ownership), and a linked checkpoint keeps the Model
        // Manager's linked-repair affordance, which is the seam that decides relink versus rescan
        // from the library root index — deliberately not re-derived from a path here.
        let linked = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(object)
            .is_some_and(|id| id.starts_with("linked/"));
        repairs.push(if linked {
            json!({
                "id": "repair-linked-checkpoint",
                "kind": "rescan",
                "label": "Relink or rescan the linked checkpoint in Model Manager",
            })
        } else {
            json!({
                "id": "repair-imported-checkpoint",
                "kind": "reimport",
                "label": "Re-import the damaged checkpoint in the checkpoint import panel",
            })
        });
    }

    for (index, row) in object
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let presence = imported_component_presence(data_dir, row);
        let component_id = imported_component_id(row, index);
        let source = row.get("repo").and_then(Value::as_str).unwrap_or_default();
        let present = presence == CoRequisitePresence::Installed;
        components.push(json!({
            "componentId": component_id,
            "role": "resident",
            "source": "resident",
            "model": source,
            "present": present,
            "state": state_name(presence),
        }));
        if !present {
            repairs.push(json!({
                "id": format!("install-component:{component_id}"),
                "kind": "download",
                "componentId": component_id,
                "variant": object.get("importComponentTier").cloned().unwrap_or(Value::Null),
                "label": format!("Download the compatible {component_id} from {source}"),
            }));
        }
    }

    // An incomplete assembly is never READY: `components` says which member is at fault and
    // `repairActions` says what to do, but nothing here may let a half-provisioned import look
    // selectable-and-runnable.
    let ready = declaration_ok
        && components
            .iter()
            .all(|component| component.get("present").and_then(Value::as_bool) == Some(true));
    (json!({ "ready": ready, "components": components }), repairs)
}

pub(super) fn apply_imported_provider_surface_for_lanes(
    object: &mut JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) {
    if object.get("catalogScope").and_then(Value::as_str) != Some("builtin")
        && object.get("type").and_then(Value::as_str) == Some("video")
        && sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(object)
    {
        apply_imported_video_support_surface(object);
        return;
    }
    if object.get("type").and_then(Value::as_str) != Some("image")
        || object.get("catalogScope").and_then(Value::as_str) == Some("builtin")
    {
        return;
    }
    let Some(family) = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)
    else {
        return;
    };
    let Some(source) = object
        .get("importSourceShape")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|source| {
            matches!(
                *source,
                "transformer_file" | "fused_checkpoint" | "transformer_directory" | "comfy_ui_tree"
            )
        })
        .map(str::to_owned)
    else {
        return;
    };

    let native_nvfp4 = sceneworks_core::jobs_store::imported_entry_source_codec(object)
        == Some(sceneworks_core::checkpoint_weight_facts::NVFP4_CODEC_ID);
    let mlx_routes = sceneworks_core::jobs_store::imported_provider_routes("mlx", &family)
        .filter(|route| !native_nvfp4 && route.source == source)
        .collect::<Vec<_>>();
    let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    let mac_support = if mlx_routes.is_empty() {
        serde_json::to_value(sceneworks_core::jobs_store::model_mac_support(
            id, "image", None,
        ))
    } else {
        Ok(json!({
            "supported": true,
            "features": {
                "pose": mlx_routes.iter().any(|route| {
                    route.operation == "pose"
                        && route.conditioning.iter().any(|kind| kind == "control")
                }),
                "reference": mlx_routes.iter().any(|route| {
                    route.operation == "generate"
                        && route
                            .conditioning
                            .iter()
                            .any(|kind| matches!(kind.as_str(), "reference" | "multi_reference"))
                }),
                "edit": mlx_routes.iter().any(|route| route.operation == "edit"),
                "lycoris": mlx_routes
                    .iter()
                    .any(|route| route.supports_lora || route.supports_lokr),
            }
        }))
    };
    if let Ok(mac_support) = mac_support {
        object.insert("macSupport".to_owned(), mac_support);
    }

    let routes = [
        ("mlx", mlx_lane_available),
        ("candle", candle_lane_available),
    ]
    .into_iter()
    .filter(|(backend, available)| *available && (!native_nvfp4 || *backend == "candle"))
    .flat_map(|(backend, _)| {
        sceneworks_core::jobs_store::imported_provider_routes(backend, &family)
    })
    .filter(|route| route.source == source)
    .collect::<Vec<_>>();
    for (backend, available) in [
        ("mlx", mlx_lane_available),
        ("candle", candle_lane_available),
    ] {
        if available && (!native_nvfp4 || backend == "candle") {
            let backend_routes =
                sceneworks_core::jobs_store::imported_provider_routes(backend, &family)
                    .filter(|route| route.source == source)
                    .collect::<Vec<_>>();
            apply_imported_sampling_surface(object, backend, &backend_routes);
        }
    }
    project_imported_operation_surface(object, &routes);
}

/// Imported checkpoints inherit the sampling surface of their exact generation provider, so a new
/// checkpoint id does not erase the sampler/scheduler menus. Preserve explicit authored overrides.
fn apply_imported_sampling_surface(
    object: &mut JsonObject,
    backend: &str,
    routes: &[&sceneworks_core::jobs_store::ImportedProviderSurface],
) {
    let Some(route) = routes.iter().find(|route| route.operation == "generate") else {
        return;
    };
    let Some(template) =
        sceneworks_core::builtin_manifests::builtin_model_entry(&route.provider_id)
    else {
        return;
    };
    let authored_limits = object.get("limits").cloned();
    let limits = template
        .get(backend)
        .and_then(|lane| lane.get("limits"))
        .or_else(|| template.get("limits"));
    for target in ["limits", "defaults", "image"] {
        let source = if target == "limits" {
            limits
        } else {
            template.get(target)
        };
        let Some(source) = source.and_then(Value::as_object) else {
            continue;
        };
        let keys: &[&str] = match target {
            "limits" => &["samplers", "schedulers", "guidanceMethods"],
            "defaults" => &[
                "sampler",
                "scheduler",
                "schedulerShift",
                "guidanceScale",
                "steps",
                "guidanceMethod",
            ],
            _ => &[
                "supportsGuidance",
                "supportsNegativePrompt",
                "negativePromptRequiresGuidance",
                "guidanceOffValue",
                "supportsSchedulerShift",
            ],
        };
        let target = object.entry(target.to_owned()).or_insert_with(|| json!({}));
        if let Some(target) = target.as_object_mut() {
            for &key in keys {
                if let Some(value) = source.get(key) {
                    target
                        .entry(key.to_owned())
                        .or_insert_with(|| value.clone());
                }
            }
        }
    }
    if let Some(source) = limits.and_then(Value::as_object) {
        let lane = object
            .entry(backend.to_owned())
            .or_insert_with(|| json!({}));
        if let Some(lane) = lane.as_object_mut() {
            let limits = lane.entry("limits".to_owned()).or_insert_with(|| json!({}));
            if let Some(limits) = limits.as_object_mut() {
                for key in ["samplers", "schedulers", "guidanceMethods"] {
                    if let Some(value) = authored_limits
                        .as_ref()
                        .and_then(|limits| limits.get(key))
                        .or_else(|| source.get(key))
                    {
                        limits
                            .entry(key.to_owned())
                            .or_insert_with(|| value.clone());
                    }
                }
            }
        }
    }
}

/// Replace family-wide defaults with the operations exposed by this exact source/backend route set.
/// Imported rows are initially enriched by `apply_model_manifest_defaults`, so merely adding provider
/// capabilities leaves unsupported sibling operations visible. This projection is intentionally
/// subtractive for every route-owned field while preserving unrelated author metadata.
pub(super) fn project_imported_operation_surface(
    object: &mut JsonObject,
    routes: &[&sceneworks_core::jobs_store::ImportedProviderSurface],
) {
    let generates_reference = routes.iter().any(|route| {
        route.operation == "generate"
            && route
                .conditioning
                .iter()
                .any(|kind| matches!(kind.as_str(), "reference" | "multi_reference"))
    });
    let pose = routes.iter().any(|route| {
        route.operation == "pose" && route.conditioning.iter().any(|kind| kind == "control")
    });
    let control_base = pose
        .then(|| {
            routes
                .iter()
                .find(|route| route.operation == "generate")
                .and_then(|route| {
                    sceneworks_core::builtin_manifests::builtin_model_entry(&route.provider_id)
                })
        })
        .flatten();
    let edit = routes.iter().any(|route| route.operation == "edit");
    let edit_multi_reference = routes.iter().any(|route| {
        route.operation == "edit"
            && route
                .conditioning
                .iter()
                .any(|kind| kind == "multi_reference")
    });
    let multi_phase = routes.iter().any(|route| route.operation == "multi_phase");
    let generates = routes.iter().any(|route| {
        matches!(
            route.operation.as_str(),
            "generate" | "pose" | "multi_phase"
        )
    });

    let capabilities = object
        .entry("capabilities".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !capabilities.is_array() {
        *capabilities = Value::Array(Vec::new());
    }
    if let Some(capabilities) = capabilities.as_array_mut() {
        capabilities.retain(|value| {
            !matches!(
                value.as_str(),
                Some("text_to_image" | "edit_image" | "image_to_image" | "character_image")
            )
        });
        if generates {
            capabilities.push(Value::String("text_to_image".to_owned()));
        }
        if edit {
            capabilities.push(Value::String("edit_image".to_owned()));
        }
    }

    let ui = object.entry("ui".to_owned()).or_insert_with(|| json!({}));
    if !ui.is_object() {
        *ui = json!({});
    }
    if let Some(ui) = ui.as_object_mut() {
        for key in ["controlScale", "controlOverlay", "controlOverlayBaseModel"] {
            ui.remove(key);
        }
        if generates_reference {
            ui.insert("img2img".to_owned(), Value::Bool(true));
        } else {
            ui.remove("img2img");
            ui.remove("img2imgStrength");
        }
        if pose {
            ui.insert("poseLibrary".to_owned(), Value::Bool(true));
            ui.insert("poseControlScale".to_owned(), Value::Bool(true));
            ui.insert("controlModes".to_owned(), json!(["pose"]));
            if let Some(base) = control_base {
                if let Some(scale) = base.get("ui").and_then(|ui| ui.get("controlScale")) {
                    ui.insert("controlScale".to_owned(), scale.clone());
                }
                if base
                    .get("ui")
                    .and_then(|ui| ui.get("controlOverlay"))
                    .and_then(Value::as_bool)
                    == Some(true)
                {
                    ui.insert("controlOverlay".to_owned(), Value::Bool(true));
                    ui.insert("controlOverlayBaseModel".to_owned(), base["id"].clone());
                }
            }
        } else {
            ui.remove("poseLibrary");
            ui.remove("poseControlScale");
            ui.remove("controlModes");
        }
        if !edit_multi_reference {
            ui.remove("editReferences");
        }
    }

    // An empty route set means this backend has no opinion about adapter compatibility. Do not
    // manufacture an empty object: catalog consumers distinguish abstention from an explicit
    // provider-owned compatibility surface. Once at least one exact route exists, keep the existing
    // object-shaped projection so supported routes retain a stable schema even when they expose no
    // acceleration adapter.
    if routes.is_empty() {
        object.remove("loraCompatibility");
    } else {
        let compatibility = object
            .entry("loraCompatibility".to_owned())
            .or_insert_with(|| json!({}));
        if let Some(compatibility) = compatibility.as_object_mut() {
            if let Some(types) = compatibility.get_mut("types").and_then(Value::as_array_mut) {
                types.retain(|value| value.as_str() != Some("acceleration"));
            }
            if multi_phase {
                let types = compatibility
                    .entry("types".to_owned())
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Some(types) = types.as_array_mut() {
                    types.push(Value::String("acceleration".to_owned()));
                }
            }
        }
    }

    object.remove("runtimeQuantTiers");

    // A checkpoint STORED as NVFP4 has exactly one runtime tier: its own (sc-11043, epic 11037).
    // The q4/q8/bf16 sweep below describes what a route can quantize a dense import TO, which is a
    // question that does not arise here — these bytes are already packed E2M1 and nothing
    // re-quantizes them. Offering the sweep for such an entry advertised `q4` for an NVFP4
    // checkpoint, which is precisely the alias E2 forbids, and the admission gate then refused
    // every job the offer produced: the picker showed tiers the router would not accept.
    //
    // Keyed on the SOURCE codec — what the weights are stored in — never on what a load would
    // materialize them as, which is a per-run receipt this catalog projection cannot know.
    if sceneworks_core::jobs_store::imported_entry_source_codec(object)
        == Some(sceneworks_core::checkpoint_weight_facts::NVFP4_CODEC_ID)
    {
        object.insert(
            "runtimeQuantTiers".to_owned(),
            Value::Array(vec![Value::String("nvfp4".to_owned())]),
        );
        return;
    }

    let mut runtime_quant_tiers = Vec::new();
    for tier in ["q4", "q8"] {
        if routes.iter().any(|route| {
            route
                .supported_quants
                .iter()
                .any(|candidate| candidate == tier)
        }) {
            runtime_quant_tiers.push(Value::String(tier.to_owned()));
        }
    }
    if routes.iter().any(|route| route.source != "comfy_ui_tree") {
        runtime_quant_tiers.push(Value::String("bf16".to_owned()));
    }
    if !runtime_quant_tiers.is_empty() {
        object.insert(
            "runtimeQuantTiers".to_owned(),
            Value::Array(runtime_quant_tiers),
        );
    }
}

/// True when this entry is a convert-at-install model that loads from a **converted artifact on the
/// platform this build targets** — so its conversion state belongs in the catalog here (sc-20529).
///
/// The converter half is [`sceneworks_core::jobs_store::convert_artifact_required_here`], the SAME
/// predicate the worker's unconverted-model preflight gates on. Re-implementing the
/// converter-membership test here (the original shape of this function) put two copies of one
/// platform rule on either side of the API/worker seam, where they can drift into an API that
/// offers a convert affordance the worker refuses to honour, or the reverse.
///
/// Only the `requiresConversion` read stays local: it is a property of THIS catalog entry, not of
/// the converter registry, and it is load-bearing — without it a turnkey model that merely names a
/// converter would start reporting MLX status off-Mac.
///
/// Platform-aware, like the core helper: off macOS only `CANDLE_NATIVE_CONVERTERS` qualify, on macOS
/// every native converter does (where the call site's `cfg!(target_os = "macos")` arm has already
/// won anyway — macOS surfaces MLX status for every entry, convert-at-install or not).
pub(super) fn entry_requires_converted_artifact_here(object: &JsonObject) -> bool {
    let Some(mlx) = object.get("mlx").and_then(Value::as_object) else {
        return false;
    };
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    mlx.get("converter")
        .and_then(Value::as_str)
        .map(str::trim)
        .is_some_and(sceneworks_core::jobs_store::convert_artifact_required_here)
}

pub(super) fn apply_mac_and_mlx_fields(object: &mut JsonObject, data_dir: &FsPath) {
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
        // Preserve the catalog-declared family in the shared probe call, but do not authorize an
        // imported loader from family identity alone. This pass supplies builtin id-keyed defaults;
        // `apply_imported_provider_surface` runs later with the full manifest entry and replaces an
        // imported row's provisional verdict from its exact family + `importSourceShape` routes.
        let family = object
            .get("family")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        // A trained checkpoint of a builtin generation engine (the Iris full-model export,
        // sc-25685) renders through that engine's id-keyed lane, so it carries the engine's verdict.
        let id: &str = match crate::iris_training::trained_task_checkpoint(object) {
            Some(objective) => objective.base_model(),
            None => id,
        };
        model_mac_support(id, model_type, family)
    };
    if let Ok(mac_support) = serde_json::to_value(mac_support) {
        object.insert("macSupport".to_owned(), mac_support);
    }
    // The off-Mac twin (sc-19570). Emitted on EVERY platform, exactly like `macSupport`: the client
    // decides whether to act on it from `candleGatingActive`, and a block that appeared only on the
    // platform it gates could never be asserted from a Mac test run — which is precisely how the
    // off-Mac half of this defect stayed invisible for as long as it did. No `family` argument: the
    // block carries the per-video-mode verdict, and video routing is id-keyed (route-by-family is
    // an image-lane mechanism).
    let candle_support = {
        let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
        let model_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id: &str = match crate::iris_training::trained_task_checkpoint(object) {
            Some(objective) => objective.base_model(),
            None => id,
        };
        model_candle_support(id, model_type)
    };
    if let Ok(candle_support) = serde_json::to_value(candle_support) {
        object.insert("candleSupport".to_owned(), candle_support);
    }
    // A LOCAL DISK probe for MLX convert-output tier dirs, so it is a platform fact and must NOT go
    // through `generation::enqueue_backend`: keying it off `candle_required` would hide `mlxTiers`
    // and its per-tier state from the Studio on macOS for directories that are genuinely present.
    //
    // Off macOS the probe is NARROWED, not skipped (sc-20529). A convert-at-install model whose
    // converter has a real off-Mac implementation — `CANDLE_NATIVE_CONVERTERS`, today
    // `flux2_klein_diffusers` / `flux2_klein_9b_true_v2` and the backend-neutral `supra2_img`
    // (sc-24214) — needs the same convert affordance and the
    // same converted/needs_conversion/needs_source states on Windows and Linux that macOS gets: the
    // candle converter (sc-7459) is real, `POST /models/{id}/convert` is not platform-gated, and
    // `inject_converted_model_path` already reads the converted dir on every platform. Without these
    // fields the Studio had no way to show that the conversion was still outstanding, so the model
    // read as installed off its raw single-file download and failed at load.
    //
    // It stays NARROW deliberately. Probing every `requiresConversion` model off-Mac would flip
    // Anima's three variants to `missing` there: `anima_quant` is macOS-only, and off-Mac Anima
    // loads the raw `circlestone-labs/Anima` `split_files/` tree with no converted dir at all. The
    // const is the contract that keeps those two cases apart.
    let mlx_status = if cfg!(target_os = "macos") || entry_requires_converted_artifact_here(object)
    {
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
                // Owned before the mutable inserts below so the `family` read doesn't overlap the borrow;
                // it selects the per-tier completeness predicate for the states (sc-13513).
                let family = object
                    .get("family")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let model_id = object
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                object.insert("mlxTiers".to_owned(), json!(tiers));
                // The FULL possible tier set with per-tier state, so the Studio shows un-converted tiers
                // disabled rather than omitting them (the web reads `mlxTierStates` in preference to the
                // installed-only `mlxTiers`). Emitted only when the model is actually converted (>=1 tier
                // present), matching `mlxTiers` — an unconverted model still renders no picker.
                object.insert(
                    "mlxTierStates".to_owned(),
                    Value::Array(mlx_convert_output_tier_states(
                        converted, &family, &model_id,
                    )),
                );
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
        // The engine's reason a marked conversion would not load (sc-24216); absent otherwise so
        // every other model's catalog shape is unchanged.
        match status.conversion_problem.as_deref() {
            Some(problem) => {
                object.insert(
                    "mlxConversionProblem".to_owned(),
                    Value::String(problem.to_owned()),
                );
            }
            None => {
                object.remove("mlxConversionProblem");
            }
        }
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

pub(super) async fn run_blocking_catalog_sweep<I, O, F>(
    items: Vec<I>,
    operation: F,
) -> Result<Vec<O>, ApiError>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I) -> Result<O, ApiError> + Send + Sync + 'static,
{
    let operation = Arc::new(operation);
    let mut items = items.into_iter();
    let mut output = Vec::new();
    loop {
        let batch = items
            .by_ref()
            .take(MODEL_CATALOG_PROBE_CONCURRENCY)
            .collect::<Vec<_>>();
        if batch.is_empty() {
            break;
        }
        let permits = MODEL_CATALOG_PROBE_PERMITS
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MODEL_CATALOG_PROBE_CONCURRENCY)))
            .clone();
        let mut tasks = Vec::with_capacity(batch.len());
        for item in batch {
            let permit = permits
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| ApiError::internal("model catalog probe limiter closed"))?;
            let operation = operation.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                let _permit = permit;
                operation(item)
            }));
        }
        let results = join_all(tasks).await;
        for result in results {
            output.push(result.map_err(|err| {
                ApiError::internal(format!("model catalog probe task failed: {err}"))
            })??);
        }
    }
    Ok(output)
}

pub(super) fn apply_model_catalog_size_fields(
    model: &mut Value,
    download_context: Option<&DownloadContext>,
    download_size_bytes: Option<u64>,
) -> Result<(), ApiError> {
    let fallback_size_bytes = download_context.and_then(|context| context.fallback_size_bytes);
    let primary_size_bytes = download_size_bytes.or(fallback_size_bytes);
    let download_size_estimated = download_size_bytes.is_none() && fallback_size_bytes.is_some();
    // Co-requisites (sc-9696) install alongside the primary, so the displayed footprint must
    // include them (e.g. PiD's ~2.7 GB checkpoint + ~5.2 GB gemma-2-2b-it). Their sizes come
    // from the manifest (the live HF estimate above only sizes the primary repo).
    // Only the co-requisites the SELECTED tier actually pulls (sc-14980) — otherwise a q4 Mage-Flow
    // install would advertise the sum of all three shared text-encoder tiers.
    // The displayed footprint is for the DEFAULT download, so size only the co-requisites that
    // download actually pulls (sc-14980) — otherwise a q4 Mage-Flow install would advertise the sum
    // of all three shared text-encoder tiers (16.1 GB instead of 2.51 GB).
    let default_variant = model_download(model)
        .as_ref()
        .and_then(|download| download.get("variant").and_then(Value::as_str))
        .map(str::to_owned);
    let co_requisite_size_bytes: u64 =
        model_co_requisite_downloads_for_variant(model, default_variant.as_deref())
            .iter()
            .filter_map(|download| manifest_download_size_bytes(model, download))
            .sum();
    let effective_download_size_bytes = match primary_size_bytes {
        Some(primary) => Some(primary + co_requisite_size_bytes),
        None if co_requisite_size_bytes > 0 => Some(co_requisite_size_bytes),
        None => None,
    };
    {
        let object = model
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
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
    }
    refresh_variant_download_sizes(model)
}

pub(super) fn apply_model_catalog_entry(
    mut model: Value,
    download_context: Option<DownloadContext>,
    data_dir: &FsPath,
    user_model_ids: &std::collections::HashSet<String>,
    builtin_model_ids: &std::collections::HashSet<String>,
    local_artifacts: &[sceneworks_core::model_artifacts::ResolvedModelArtifact],
) -> Result<Value, ApiError> {
    #[cfg(test)]
    test_delay_catalog_probe(&model);
    let state = install_state_for(download_context, &model, data_dir);
    // sc-19708: the typed availability judgement for this entry's default selected closure on
    // this host, through the ONE shared resolver. Catalog nuances layered on top of it:
    // an install living in an app-owned path is local-ready regardless of the library, and an
    // installed-but-incomplete cache stays `incomplete` rather than `missing`.
    let availability_resolution =
        crate::model_sources::availability_for_entry(data_dir, &model, None, local_artifacts);
    // sc-19712 F-5: what a local copy of this entry could EVER cover, which is a different question
    // from where it resolves from today. A model with soft co-requisites or a revision-less
    // requirement showed the identical "local copy" affordance as a fully cacheable one, so the
    // badge promised protection from a disconnect that the cache cannot actually deliver.
    let cache_eligibility =
        crate::model_sources::cache_eligibility_for_entry(data_dir, &model, None);
    use sceneworks_core::model_artifacts::external_library::ModelAvailability;
    let managed_models = data_dir.join("models");
    let installed_in_app_owned_path = state.installed
        && state
            .installed_path
            .as_deref()
            .map(FsPath::new)
            .is_some_and(|path| path.starts_with(&managed_models));
    let (availability, availability_resolution) = if state.cache_incomplete
        && model.get("family").and_then(Value::as_str) == Some("minimax-h3")
        && model
            .as_object()
            .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
            .is_some()
    {
        (ModelAvailability::Incomplete, Some(availability_resolution))
    } else if installed_in_app_owned_path
        && !matches!(
            availability_resolution.availability,
            ModelAvailability::LocalReady
        )
    {
        (ModelAvailability::LocalReady, None)
    } else if availability_resolution.availability == ModelAvailability::Missing
        && state.cache_incomplete
    {
        (ModelAvailability::Incomplete, Some(availability_resolution))
    } else {
        (
            availability_resolution.availability.clone(),
            Some(availability_resolution),
        )
    };
    let platform_cleanup_only =
        model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true);
    let cleanup_residue_only = platform_cleanup_only
        && !state.installed
        && cleanup_with_model_artifact_paths(&model, data_dir)
            .iter()
            .any(|path| std::fs::symlink_metadata(path).is_ok());
    // sc-22988: which options of each co-requisite choice group are installed here (YuE2's
    // `decoder`: `standard` / `legacy`), so a surface that offers the choice can disable an option
    // whose component is not installed. Present only on a model that declares a choice group.
    let installed_choices = installed_co_requisite_choices(&model, data_dir);
    let object = model
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    if let Some(choices) = installed_choices {
        object.insert("installedChoices".to_owned(), choices);
    }
    object.insert(
        "modelAvailability".to_owned(),
        serde_json::to_value(&availability).map_err(|error| {
            ApiError::internal(format!("serialize model availability: {error}"))
        })?,
    );
    object.insert(
        "modelResolution".to_owned(),
        availability_resolution
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| ApiError::internal(format!("serialize model resolution: {error}")))?
            .unwrap_or(Value::Null),
    );
    object.insert(
        "cacheEligibility".to_owned(),
        cache_eligibility
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| ApiError::internal(format!("serialize cache eligibility: {error}")))?
            .unwrap_or(Value::Null),
    );
    let model_id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    // Ownership and routing authority are deliberately separate. A user overlay on a builtin is
    // removable (deleting it restores builtin defaults), but it cannot reclassify that protected id
    // as an imported checkpoint and bypass/trigger source-shaped provider validation.
    let user_managed = user_model_ids.contains(model_id);
    let builtin_routing_authority = builtin_model_ids.contains(model_id);
    object.insert(
        "catalogScope".to_owned(),
        Value::String(
            if builtin_routing_authority {
                "builtin"
            } else {
                "user"
            }
            .to_owned(),
        ),
    );
    stamp_legacy_import_source_shape(object, data_dir, !builtin_routing_authority);
    object.insert(
        "downloadable".to_owned(),
        Value::Bool(state.downloadable && !platform_cleanup_only),
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
            if state.cache_incomplete || cleanup_residue_only {
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
        Value::Bool(state.downloadable && state.cache_incomplete && !platform_cleanup_only),
    );
    object.insert(
        "updateAvailable".to_owned(),
        Value::Bool(state.update_available && !platform_cleanup_only),
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
        Value::Bool(user_managed || state.installed || platform_cleanup_only),
    );
    if platform_cleanup_only {
        object.insert("usable".to_owned(), Value::Bool(false));
        object.insert("recommended".to_owned(), Value::Bool(false));
    }
    // Per-variant install tracking (sc-8508, epic 8506): one entry per declared quant tier,
    // each with its own installed flag + path + size + footprint. A single-variant model
    // still emits exactly one "default" variant, so the array is a superset of the
    // (retained) top-level installState/installedPath fields — nothing existing regresses.
    apply_variant_fields(object, data_dir);
    apply_gating_fields(object);
    apply_mac_and_mlx_fields(object, data_dir);
    apply_imported_provider_surface(object);
    apply_imported_lora_advertisement(object);
    mark_empty_lora_advertisement_unsupported(object);
    // Live denoise preview support (sc-16965, epic 16948): `preview.byBackend`, read from the
    // generated `config/manifests/builtin.preview-support.jsonc` rather than from a registry, because
    // THIS process may link no engines at all (docker/rust.Dockerfile builds the API without
    // `backend-candle`, so `Registry::new()` here is empty and a serve-time derivation would report
    // "nothing supports preview" on every server). Engine-KEYED on purpose — the flag genuinely
    // diverges by backend and never fully collapses. Additive: a model the generated table does not
    // know gets no `preview` key, which the UI reads as "unknown" and renders exactly as before.
    apply_last_only_video_capability(object);
    sceneworks_core::preview_support::apply_to_model_entry(object);
    sceneworks_core::decoder_support::apply_to_model_entry(object);
    apply_decoder_availability(object, data_dir);
    apply_conditional_component_state(object, data_dir);
    apply_imported_checkpoint_product_surface(object, data_dir, state.imported_source_present);
    if platform_cleanup_only {
        // The tombstone exists only to expose whole-model Delete. Strip every tier/conversion
        // action projection even though the preserved manifest metadata is still needed by the
        // deletion resolver to locate old files.
        object.insert("hasVariantMatrix".to_owned(), Value::Bool(false));
        object.remove("variants");
        object.remove("mlxConversionState");
        object.remove("mlxInstallState");
        object.remove("mlxTiers");
        object.remove("mlxTierStates");
    }
    Ok(model)
}

// Use exactly the entry validation uses, including resolved imported partitions.
pub(super) fn apply_last_only_video_capability(object: &mut JsonObject) {
    let imported_partition = object.get("id").and_then(Value::as_str).and_then(|_| {
        use sceneworks_core::minimax_h3_import::{
            entry_is_imported_minimax_h3, MiniMaxH3Partition,
        };
        entry_is_imported_minimax_h3(object)
            .then(|| MiniMaxH3Partition::from_entry(object).ok())
            .flatten()
    });
    let accepts_last_only =
        sceneworks_core::h3_keyframes::accepts_last_only_resolved_image_to_video(
            object.get("id").and_then(Value::as_str).unwrap_or_default(),
            "image_to_video",
            true,
            imported_partition.map(|partition| partition.as_str()),
        );
    object.insert(
        "supportsLastOnlyImageToVideo".to_owned(),
        Value::Bool(accepts_last_only),
    );
}

/// Give a provider-aliased model row the canonical provider's exact optional decoder donors.
///
/// Imported Krea entries intentionally carry `downloads: []`: their primary DiT is an in-place user
/// file, not a downloadable model. Decoder eligibility is nevertheless provider-derived, and the
/// selected decoder must resolve the same pinned standalone VAE as the builtin Turbo provider. Clone
/// only soft co-requisites whose component ids occur in the provider's generated decoder facts;
/// primary downloads and unrelated components (including text encoders) are never inherited.
pub(super) fn inherit_provider_decoder_downloads(model: &mut Value, builtin_models: &[Value]) {
    let Some(object) = model.as_object() else {
        return;
    };
    let Some(model_id) = object.get("id").and_then(Value::as_str) else {
        return;
    };
    let Some(provider_id) =
        sceneworks_core::decoder_support::provider_id_for_backend(object, "mlx")
    else {
        return;
    };
    if provider_id == model_id {
        return;
    }
    let component_ids = sceneworks_core::decoder_support::component_ids_for_backend(object, "mlx");
    if component_ids.is_empty() {
        return;
    }
    let Some(provider) = builtin_models.iter().find(|candidate| {
        candidate.get("id").and_then(Value::as_str) == Some(provider_id.as_str())
    }) else {
        return;
    };
    let inherited = provider
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| {
            download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                && download.get("required").and_then(Value::as_str) == Some("soft")
                && download
                    .get("componentId")
                    .and_then(Value::as_str)
                    .is_some_and(|component_id| {
                        component_ids
                            .iter()
                            .any(|expected| expected == component_id)
                    })
                && download
                    .get("revision")
                    .and_then(Value::as_str)
                    .is_some_and(|revision| {
                        revision.len() == 40
                            && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
                && download
                    .get("repo")
                    .and_then(Value::as_str)
                    .is_some_and(|repo| !repo.trim().is_empty())
                && download
                    .get("files")
                    .and_then(Value::as_array)
                    .is_some_and(|files| !files.is_empty())
        })
        .cloned()
        .collect::<Vec<_>>();
    if inherited.is_empty() {
        return;
    }

    let object = model.as_object_mut().expect("checked object above");
    let existing = object
        .remove("downloads")
        .and_then(|downloads| downloads.as_array().cloned())
        .unwrap_or_default();
    // Canonical rows lead so both the API availability probe and the worker's exact component
    // resolver bind the generated decoder id to the provider's source even if a user manifest
    // happens to contain another soft component with the generic `vae` id.
    let mut downloads = inherited.clone();
    downloads.extend(
        existing
            .into_iter()
            .filter(|download| !inherited.iter().any(|canonical| canonical == download)),
    );
    object.insert("downloads".to_owned(), Value::Array(downloads));
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct OptionalComponentAvailability {
    pub(super) installed: bool,
    pub(super) estimated_size_bytes: Option<u64>,
}

/// Enrich descriptor-derived decoder options with the local state of their pinned, soft
/// co-requisite. Eligibility comes exclusively from engine facts; this function answers only whether
/// the declared standalone component is present. A full Wan model install is neither consulted nor
/// implied.
pub(super) fn apply_decoder_availability(object: &mut JsonObject, data_dir: &FsPath) {
    let mut components: std::collections::BTreeMap<String, OptionalComponentAvailability> =
        std::collections::BTreeMap::new();
    for download in object
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if download.get("coRequisite").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let Some(component_id) = download.get("componentId").and_then(Value::as_str) else {
            continue;
        };
        let installed = pinned_component_download_installed(download, data_dir);
        let size = download
            .get("estimatedSizeBytes")
            .and_then(Value::as_u64)
            .or_else(|| {
                download
                    .get("footprint")
                    .and_then(|footprint| footprint.get("diskSizeBytes"))
                    .and_then(Value::as_u64)
            });
        // The provider-inheritance seam prepends its canonical exact row. First declaration wins so
        // a later user-authored soft `vae` row cannot make the Wan option appear installed or divert
        // the worker to a different file.
        components
            .entry(component_id.to_owned())
            .or_insert(OptionalComponentAvailability {
                installed,
                estimated_size_bytes: size,
            });
    }

    let Some(by_backend) = object
        .get_mut(sceneworks_core::decoder_support::DECODERS_FIELD)
        .and_then(Value::as_object_mut)
        .and_then(|decoders| {
            decoders
                .get_mut(sceneworks_core::decoder_support::BY_BACKEND_FIELD)
                .and_then(Value::as_object_mut)
        })
    else {
        return;
    };
    for options in by_backend.values_mut().filter_map(Value::as_array_mut) {
        for option in options.iter_mut().filter_map(Value::as_object_mut) {
            let state = option
                .get("componentId")
                .and_then(Value::as_str)
                .and_then(|id| components.get(id))
                .copied()
                .unwrap_or_default();
            option.insert("available".to_owned(), Value::Bool(state.installed));
            if let Some(bytes) = state.estimated_size_bytes {
                option.insert("estimatedSizeBytes".to_owned(), Value::from(bytes));
            }
        }
    }
}

pub(super) fn pinned_component_download_installed(download: &Value, data_dir: &FsPath) -> bool {
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return false;
    };
    let Some(revision) = download.get("revision").and_then(Value::as_str) else {
        return false;
    };
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return false;
    }
    let files = string_array_field(download, "files");
    if files.is_empty() {
        return false;
    }
    // Through the typed source-library resolver, not a hand-built `snapshots/<rev>` root — the
    // model-path inventory audit (sc-19703) forbids direct root construction on this file.
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision)) else {
        return false;
    };
    files
        .iter()
        .all(|pattern| snapshot_contains_pattern(&snapshot, pattern))
}

/// Backfill the explicit source-shape discriminator for user entries created before SC-18312.
/// This is structural inspection under the app-managed model root, never a family guess: a Mage
/// transformer directory must carry both of its required files; a single-file import is classified
/// by the same header-only detector the importer uses. The worker repeats confinement and shape
/// validation before opening weights.
pub(super) fn stamp_legacy_import_source_shape(
    object: &mut JsonObject,
    data_dir: &FsPath,
    user_managed: bool,
) {
    if !user_managed || object.contains_key("importSourceShape") {
        return;
    }
    let Some(raw) = object
        .get("modelPath")
        .and_then(Value::as_str)
        .or_else(|| {
            object
                .get("paths")
                .and_then(Value::as_object)
                .and_then(|paths| paths.get("model"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|path| !path.is_empty())
    else {
        return;
    };
    let Ok(path) = std::fs::canonicalize(raw) else {
        return;
    };
    let Ok(model_root) = std::fs::canonicalize(data_dir.join("models")) else {
        return;
    };
    if !path.starts_with(model_root) {
        return;
    }
    if object.get("family").and_then(Value::as_str) == Some("mage-flow")
        && path.is_dir()
        && sceneworks_core::base_weights::is_mage_flow_transformer_dir(&path)
    {
        object.insert(
            "importSourceShape".to_owned(),
            Value::String("transformer_directory".to_owned()),
        );
        return;
    }
    let Some(weight_file) = imported_model_primary_weight_file(&path) else {
        return;
    };
    let Ok(BaseWeightDetection::Recognized(verdict)) = detect_base_weight_file(&weight_file) else {
        return;
    };
    let family = object.get("family").and_then(Value::as_str);
    let source = match (family, verdict.family.as_deref(), verdict.component) {
        (Some("krea_2"), Some("krea_2"), ComponentRole::Transformer) => "transformer_file",
        (Some("sdxl"), Some("sdxl"), ComponentRole::Checkpoint) => "fused_checkpoint",
        _ => return,
    };
    object.insert(
        "importSourceShape".to_owned(),
        Value::String(source.to_owned()),
    );
}

pub(super) type ModelCatalogWorkItem = (Value, Option<DownloadContext>);

/// Registered builtins use one field-authority rule for catalog, license, and job reads.
/// Other rows keep the existing manifest merge until they enter the registry.
pub(crate) fn merge_model_entries_with_registry(
    builtin: Vec<Value>,
    user: Vec<Value>,
) -> Vec<Value> {
    let builtin_ids: std::collections::HashSet<String> = builtin
        .iter()
        .filter_map(|entry| entry.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut merged = merge_entries_by_id(builtin, user.clone());
    for entry in &mut merged {
        let Some(id) = entry.get("id").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        if let Some(spec) = sceneworks_core::registry::builtin_model_spec(&id) {
            let user_entry = user
                .iter()
                .find(|candidate| candidate.get("id").and_then(Value::as_str) == Some(id.as_str()));
            *entry = spec.project(user_entry);
        }
        if builtin_ids.contains(&id) {
            sceneworks_core::registry::project_builtin_memory_evidence(entry)
                .expect("validated builtin memory evidence projection");
        }
    }
    merged
}

pub(super) fn group_model_catalog_work_items(
    work_items: Vec<ModelCatalogWorkItem>,
) -> Vec<Vec<ModelCatalogWorkItem>> {
    let mut groups = Vec::<Vec<ModelCatalogWorkItem>>::new();
    let mut repo_groups = HashMap::<String, usize>::new();
    for item in work_items {
        let (_, download_context) = &item;
        let repo = download_context
            .as_ref()
            .map(|context| context.repo.clone());
        if let Some(repo) = repo {
            if let Some(index) = repo_groups.get(&repo).copied() {
                groups[index].push(item);
            } else {
                repo_groups.insert(repo, groups.len());
                groups.push(vec![item]);
            }
        } else {
            groups.push(vec![item]);
        }
    }
    groups
}

/// Built-in + user model manifest entries merged by id, with NO platform filtering — the raw
/// authored catalog. [`load_model_catalog_inputs`] narrows `downloads` to the running OS on top of
/// this; [`license_acknowledgment_repo_index`] deliberately reads it unfiltered, because a licence
/// requirement must not depend on which OS is asking. Both manifest reads are mtime/size-cached
/// (`load_manifest_entries`), so the second consumer costs a stat and a clone.
pub(crate) async fn merged_model_manifest_entries(
    state: &AppState,
) -> Result<(Vec<Value>, std::collections::HashSet<String>), ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    Ok((
        merge_model_entries_with_registry(builtin, user),
        user_model_ids,
    ))
}

pub(super) async fn load_model_catalog_inputs(
    state: &AppState,
) -> Result<
    (
        Vec<Value>,
        Vec<Option<DownloadContext>>,
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
    ),
    ApiError,
> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    let decoder_providers = merge_model_entries_with_registry(builtin.clone(), Vec::new());
    let mut builtin_model_ids = builtin
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    let mut models = merge_model_entries_with_registry(builtin, user);
    for model in &models {
        if let Some(id) = model.get("id").and_then(Value::as_str) {
            if sceneworks_core::registry::builtin_model_spec(id).is_some() {
                builtin_model_ids.insert(id.to_owned());
            }
        }
    }
    // A trained checkpoint (Iris full export) takes its generation surface from its engine's entry
    // on every read, never from a registration-time snapshot.
    for model in &mut models {
        crate::iris_training::overlay_trained_checkpoint_surface(model, &decoder_providers)?;
    }
    retain_models_for_os(&mut models, std::env::consts::OS, &state.settings.data_dir)?;
    // Resolve per-platform download sources before computing install state/size: some video models
    // carry both a native MLX-convert checkpoint (macOS) and a diffusers/torch checkpoint
    // (Windows/Linux). Keep only the entries applicable to this OS so the download job, status,
    // size, and the frontend all agree on the right repo (sc-3240).
    for model in &mut models {
        inherit_provider_decoder_downloads(model, &decoder_providers);
        if model.get("platformCleanupOnly").and_then(Value::as_bool) != Some(true) {
            retain_downloads_for_os(model, std::env::consts::OS);
        }
    }
    let download_contexts = models
        .iter()
        .map(model_download_context)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((models, download_contexts, user_model_ids, builtin_model_ids))
}

pub(super) async fn build_model_catalog_snapshot(state: &AppState) -> Result<Vec<Value>, ApiError> {
    // sc-8819 (F-017): observe full-catalog assembly (the per-model FS install-state probe
    // sweep) so tests can assert request-scoped and process-shared reuse.
    #[cfg(test)]
    crate::test_note_model_catalog_build();
    let (models, download_contexts, user_model_ids, builtin_model_ids) =
        load_model_catalog_inputs(state).await?;
    let builtin_model_ids = Arc::new(builtin_model_ids);

    let data_dir = Arc::new(state.settings.data_dir.clone());
    let user_model_ids = Arc::new(user_model_ids);
    // App-owned resolved-local artifacts, enumerated ONCE per snapshot build (read-only: no
    // session, no usage refresh) so every entry's availability can detect local-ready coverage.
    let local_artifacts = Arc::new(crate::model_sources::local_resolved_artifacts(state));
    let work_items = models
        .into_iter()
        .zip(download_contexts.clone())
        .collect::<Vec<_>>();
    let work_groups = group_model_catalog_work_items(work_items);
    let data_dir_for_sweep = data_dir.clone();
    let user_model_ids_for_sweep = user_model_ids.clone();
    let builtin_model_ids_for_sweep = builtin_model_ids.clone();
    let local_artifacts_for_sweep = local_artifacts.clone();

    // sc-14530: each model's install-state resolution is independent but may spend seconds
    // waiting on network-volume metadata. Dispatch bounded blocking tasks per primary-repo group
    // so that latency overlaps instead of accumulating serially across the full catalog, while
    // shared-repo receipt handling stays deterministic.
    let catalog_sweep = run_blocking_catalog_sweep(work_groups, move |work_group| {
        work_group
            .into_iter()
            .map(|(model, download_context)| {
                apply_model_catalog_entry(
                    model,
                    download_context,
                    &data_dir_for_sweep,
                    &user_model_ids_for_sweep,
                    &builtin_model_ids_for_sweep,
                    &local_artifacts_for_sweep,
                )
            })
            .collect::<Result<Vec<_>, _>>()
    });

    // External ComfyUI discovery is independent from manifest install-state resolution; overlap
    // that filesystem scan with the per-entry sweep too.
    let external_roots = state.settings.external_model_roots.clone();
    let external_base_cache = state.external_base_model_cache.clone();
    let external_scan = tokio::task::spawn_blocking(move || {
        let mut cache = external_base_cache.lock();
        crate::external_base_models::scan_external_base_models(&external_roots, &mut cache)
    });
    let (models, external) = tokio::join!(catalog_sweep, external_scan);
    let mut models = models?.into_iter().flatten().collect::<Vec<_>>();
    for model in &mut models {
        let context = model_download_context(model)?;
        // The shared snapshot carries deterministic manifest fallbacks only.
        // `/models` overlays live estimates on its response clone; presets and
        // job validation never wait on unrelated Hugging Face network calls.
        apply_model_catalog_size_fields(model, context.as_ref(), None)?;
    }
    let mut external = external.map_err(|err| {
        ApiError::internal(format!("external model catalog scan task failed: {err}"))
    })?;
    for model in &mut external {
        if let Some(object) = model.as_object_mut() {
            apply_imported_provider_surface(object);
            apply_imported_lora_advertisement(object);
        }
    }
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
    let builtin_entry = find(&builtin);
    let user_entry = find(&user);
    let registered = sceneworks_core::registry::builtin_model_spec(model_id);
    let mut catalog_scope =
        if builtin_entry.is_some() || (registered.is_some() && user_entry.is_some()) {
            Some("builtin")
        } else if user_entry.is_some() {
            Some("user")
        } else {
            None
        };
    let mut entry = if let Some(spec) =
        registered.filter(|_| builtin_entry.is_some() || user_entry.is_some())
    {
        spec.project(user_entry.as_ref())
    } else {
        merge_model_manifest_entry(builtin_entry, user_entry)
    };
    if catalog_scope == Some("user") {
        crate::iris_training::overlay_trained_checkpoint_surface(&mut entry, &builtin)?;
    }
    if entry.as_object().is_none_or(JsonObject::is_empty) {
        // The model-source seam (sc-19708) resolves fixed utility models (upscalers, detectors,
        // the face stack) for job admission even when the configured manifest directory is
        // missing or trimmed — the compiled-in builtin manifest is the complete fallback so a
        // stripped config dir cannot silently strip typed model-source coverage.
        if let Some(embedded) = embedded_builtin_catalog_entry(|candidate| {
            candidate.get("id").and_then(Value::as_str) == Some(model_id)
        })? {
            entry = embedded;
            catalog_scope = Some("builtin");
        }
    }
    inherit_provider_decoder_downloads(&mut entry, &builtin);
    if let (Some(scope), Some(object)) = (catalog_scope, entry.as_object_mut()) {
        // The merged worker-facing entry must carry the same routing authority as `/models`.
        // A user manifest may override builtin presentation/defaults without turning the protected
        // builtin identity into an imported checkpoint. Genuine user-only ids remain user-scoped.
        object.insert("catalogScope".to_owned(), Value::String(scope.to_owned()));
        stamp_legacy_import_source_shape(object, &state.settings.data_dir, scope == "user");
    }
    inject_converted_model_path(&mut entry, &state.settings.data_dir);
    if let Some(object) = entry.as_object_mut() {
        // Job admission and overlay authorization consume this server-owned entry too.
        // Reapply the same provider surface the catalog exposes instead of trusting client UI data.
        apply_imported_provider_surface(object);
        // The worker receives the same descriptor-derived selection table and local pinned-file
        // availability as the catalog UI. This keeps enqueue validation and execution on one fact set.
        sceneworks_core::decoder_support::apply_to_model_entry(object);
        apply_decoder_availability(object, &state.settings.data_dir);
    }
    Ok(entry)
}

/// The compiled-in builtin model manifest, used as the last-resort catalog authority when the
/// on-disk manifest directory does not carry an entry. Kept complete and cross-platform: the
/// caller derives per-platform closures from it, so filtering here would let a macOS API omit
/// Candle-only co-requisites from a Windows/Linux worker carrier.
pub(crate) fn embedded_builtin_catalog_entry(
    predicate: impl Fn(&Value) -> bool,
) -> Result<Option<Value>, ApiError> {
    let manifest = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .map_err(|error| ApiError::internal(format!("embedded model manifest invalid: {error}")))?;
    let entry = manifest
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|entry| predicate(entry))
        .cloned();
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

/// Remove whole video products that have no route on `os`. This is intentionally narrower than the
/// legacy image-model `macOnly` label: only video entries use it as a catalog-withdrawal contract.
pub(super) fn retain_models_for_os(
    models: &mut Vec<Value>,
    os: &str,
    data_dir: &FsPath,
) -> Result<(), ApiError> {
    if os == "macos" {
        return Ok(());
    }
    let mut retained = Vec::with_capacity(models.len());
    for mut model in std::mem::take(models) {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let withdrawn_video = video_model_withdrawn_on_platform(model_id, &model, os);
        if !withdrawn_video {
            retained.push(model);
            continue;
        }

        // Previously shipped off-Mac routes may already occupy substantial disk. Keep a catalog
        // tombstone only when the old artifact is installed or partial so Model Manager and DELETE
        // can reclaim it; a fresh machine sees no row at all. Preserve its original downloads on
        // the tombstone solely to resolve install state and owned cleanup paths.
        let context = model_download_context(&model)?;
        let state = install_state_for(context, &model, data_dir);
        let cleanup_residue = cleanup_with_model_artifact_paths(&model, data_dir)
            .iter()
            .any(|path| std::fs::symlink_metadata(path).is_ok());
        if state.installed || state.cache_incomplete || cleanup_residue {
            let object = model
                .as_object_mut()
                .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
            object.insert("platformCleanupOnly".to_owned(), Value::Bool(true));
            retained.push(model);
        }
    }
    *models = retained;
    Ok(())
}

/// Product withdrawals are authoritative by stable model id, not by a user-overridable manifest
/// presentation flag. `macOnly` remains the generic platform annotation, while this policy keeps a
/// same-id user manifest from restoring a product whose off-Mac runtime was explicitly removed.
pub(crate) fn video_model_withdrawn_on_platform(
    model_id: &str,
    model_manifest_entry: &Value,
    platform: &str,
) -> bool {
    platform != "macos"
        && (matches!(model_id, "ltx_2_3_eros")
            || (model_manifest_entry.get("type").and_then(Value::as_str) == Some("video")
                && model_manifest_entry.get("macOnly").and_then(Value::as_bool) == Some(true)))
}
