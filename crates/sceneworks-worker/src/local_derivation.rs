//! Locally derived model tiers (sc-22999): the deriver the `localDerivation` catalog rows name.
//!
//! A derived tier (YuE2's `q8` / `q4`) is never downloaded: its weights are CC BY-NC and have no
//! recorded redistribution basis, so the engine derives them on this machine from the verified
//! original. The API queues an ordinary `ModelDownload` of the ORIGINAL carrying a
//! `localDerivation` block; after the fetch, [`derive_local_tier`] runs the audio lane's snapshot
//! preparer (`PrepareSpec { quantize }` → `candle_audio_yue2::tier::convert`, which verifies its
//! output against the engine's `TIER_PINS`) into
//! `artifact_selection::local_derivation_snapshot_dir`, then re-verifies the result against the
//! catalog's own pinned bytes and digest. Only then does the catalog report the tier installed.
//!
//! Nothing here touches the original: a derivation that fails or does not verify removes only the
//! derived snapshot it wrote.
//!
//! FLUX 3 Image Klein's native `q4` / `q8` packages (sc-26241) are the second user: the same
//! fetch → derive → verify flow, with [`flux3_preparer`] running the inference FLUX 3 converter
//! of the backend that loads the package (Candle GGML off-Mac, MLX affine on macOS) instead of
//! the audio preparer.

use super::*;

use gen_core::core_llm::{PrepareReport, PrepareSpec, Quantize};
use sceneworks_core::model_artifacts::artifact_selection::{
    derived_snapshot_state, local_derivation, local_derivation_snapshot_dir, DerivedSnapshotState,
};

/// The payload key the API stamps on a derived-tier download.
pub(crate) const PAYLOAD_KEY: &str = "localDerivation";

fn quantize_of(variant: &str) -> WorkerResult<Quantize> {
    match variant {
        "q8" => Ok(Quantize::Q8),
        "q4" => Ok(Quantize::Q4),
        other => Err(WorkerError::InvalidPayload(format!(
            "localDerivation: '{other}' is not a derivable tier (q8 or q4)"
        ))),
    }
}

fn working_dir(dir: &Path) -> PathBuf {
    let mut name = dir
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".partial");
    dir.with_file_name(name)
}

/// Derive the tier a download job's payload names from the original at `source` (see the
/// [module docs](self)). Returns the verified derived snapshot directory.
pub(crate) async fn derive_local_tier(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    source: &Path,
    prepare: impl FnOnce(&PrepareSpec) -> Result<PrepareReport, String> + Send + 'static,
) -> WorkerResult<PathBuf> {
    let block = job
        .payload
        .get(PAYLOAD_KEY)
        .cloned()
        .ok_or_else(|| WorkerError::InvalidPayload("no localDerivation block".into()))?;
    let variant = block
        .get("variant")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let derivation = local_derivation(&json!({ PAYLOAD_KEY: block })).ok_or_else(|| {
        WorkerError::InvalidPayload("localDerivation: the block is incomplete".into())
    })?;
    let model_id = job
        .payload
        .get("modelId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let quantize = quantize_of(&variant)?;
    let out_dir =
        local_derivation_snapshot_dir(&settings.data_dir, model_id, &variant, &derivation)
            .ok_or_else(|| {
                WorkerError::InvalidPayload(
                    "localDerivation: the model id, variant or conversion is not a plain path \
                     segment"
                        .into(),
                )
            })?;
    match derived_snapshot_state(&out_dir, &derivation) {
        DerivedSnapshotState::Verified => {
            if !derivation.conversion.starts_with("flux3-")
                || crate::flux3_converted_bundle_check(source, &out_dir).is_ok()
            {
                return Ok(out_dir);
            }
            // A pinned weights file alone does not certify the config, licenses, receipt or
            // external components. Repair only this owned output, never the original.
            std::fs::remove_dir_all(&out_dir)?;
        }
        // Not this derivation's bytes: it is our own derived artifact, never the original.
        DerivedSnapshotState::Invalid(_) => std::fs::remove_dir_all(&out_dir)?,
        DerivedSnapshotState::Absent => {}
    }
    // An interrupted derivation's working directory: the converter refuses to start beside it.
    let work = working_dir(&out_dir);
    if work.exists() {
        std::fs::remove_dir_all(&work)?;
    }
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Downloading,
            ProgressStage::Downloading,
            0.95,
            &format!("Deriving the {variant} tier locally from the original."),
            None,
            None,
            None,
        ),
    )
    .await?;
    let spec = PrepareSpec {
        source: source.to_path_buf(),
        out_dir: out_dir.clone(),
        quantize: Some(quantize),
    };
    // One bounded, CPU-bound conversion with no step loop to interrupt: keep the heartbeat alive
    // while it runs (a multi-GB conversion outlasts the stale-worker timeout).
    let task = crate::load_progress::spawn_blocking(move || {
        prepare(&spec).map_err(|error| WorkerError::Engine(format!("tier derivation: {error}")))
    });
    let derived = heartbeat_while_blocking(api, settings, &job.id, "tier derivation", task).await;
    if let Err(error) = derived {
        if out_dir.exists() {
            let _ = std::fs::remove_dir_all(&out_dir);
        }
        return Err(error);
    }
    match derived_snapshot_state(&out_dir, &derivation) {
        DerivedSnapshotState::Verified => {
            if derivation.conversion.starts_with("flux3-") {
                crate::flux3_converted_bundle_check(source, &out_dir)
                    .map_err(WorkerError::InvalidPayload)?;
            }
            Ok(out_dir)
        }
        other => {
            let _ = std::fs::remove_dir_all(&out_dir);
            Err(WorkerError::Engine(format!(
                "the derived {variant} tier does not match the catalog's pinned weights \
                 ({other:?}); it was removed"
            )))
        }
    }
}

/// One FLUX 3 Image Klein native conversion (sc-26241), named by its catalog
/// `localDerivation.conversion` id. Each converts the ORIGINAL dense package with the inference
/// converter of the backend that loads it — never another backend's package, never a publisher
/// FP8R/NVFP4 artifact — into a NEW owned directory published atomically with a full-checksum
/// receipt. The text encoder and autoencoder stay EXTERNAL (the pinned snapshot's own files).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flux3Conversion {
    /// Candle GGML Q4_1 (`candle_gen_flux3::convert::convert_ggml`).
    CandleGgmlQ4_1,
    /// Candle GGML Q8_0.
    CandleGgmlQ8_0,
    /// MLX affine W`bits`A16 at `group` (`gen_core_flux3::native_convert::convert_affine`, the
    /// backend-neutral pack MLX's `quantized_matmul` reads; CPU-only and deterministic).
    MlxAffine { bits: u32, group: u32 },
}

impl Flux3Conversion {
    /// Every conversion the catalog may name, by id.
    pub(crate) const ALL: [(&'static str, Self); 4] = [
        ("flux3-candle-ggml-q4_1-v1", Self::CandleGgmlQ4_1),
        ("flux3-candle-ggml-q8_0-v1", Self::CandleGgmlQ8_0),
        (
            "flux3-mlx-affine-q4-g64-v1",
            Self::MlxAffine { bits: 4, group: 64 },
        ),
        (
            "flux3-mlx-affine-q8-g64-v1",
            Self::MlxAffine { bits: 8, group: 64 },
        ),
    ];

    pub(crate) fn from_id(id: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find_map(|(name, conversion)| (name == id).then_some(conversion))
    }

    /// The tier this conversion produces (`localDerivation` rows are keyed by it).
    pub(crate) const fn quantize(self) -> Quantize {
        match self {
            Self::CandleGgmlQ4_1 | Self::MlxAffine { bits: 4, .. } => Quantize::Q4,
            Self::CandleGgmlQ8_0 | Self::MlxAffine { .. } => Quantize::Q8,
        }
    }
}

/// The `model` both published FLUX 3 checkpoints declare in `config.json`. The base checkpoint does
/// NOT name itself `-base`: its released config is the distilled one without `dit.flow_map`, and
/// its native header has no `ref_time_in`.
const FLUX3_RELEASED_CONFIG_MODEL: &str = "flux-3-image-klein";

/// Bind the source's actual checkpoint to its frozen repository identity before converting. Merely
/// pointing the base environment variable at distilled weights is not base proof. Identity is the
/// frozen repository + revision, then inference's header-authoritative checkpoint resolution (the
/// flow map comes from the native `model.safetensors` header, an optional config flag must agree)
/// and its shared route check for the matching catalog model, plus dense storage and the released
/// model name. SceneWorks never re-derives the flow map from `config.json`.
fn validate_flux3_dense_identity(
    root: &Path,
    repository: &str,
    revision: &str,
) -> Result<(), String> {
    use gen_core_flux3::{
        BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION, DISTILLED_WEIGHTS_REPO,
        DISTILLED_WEIGHTS_REVISION,
    };
    let (expected_revision, model_id) = match repository {
        DISTILLED_WEIGHTS_REPO => (DISTILLED_WEIGHTS_REVISION, "flux3_klein"),
        BASE_WEIGHTS_REPO => (BASE_WEIGHTS_REVISION, "flux3_klein_base"),
        _ => return Err("FLUX3 conversion source is not a frozen checkpoint repository".into()),
    };
    if revision != expected_revision {
        return Err(format!(
            "FLUX3 {repository} requires revision {expected_revision}"
        ));
    }
    let resolved = gen_core_flux3::resolve_checkpoint(root).map_err(|e| e.to_string())?;
    resolved
        .validate_route(model_id)
        .map_err(|e| e.to_string())?;
    if resolved.model != FLUX3_RELEASED_CONFIG_MODEL || resolved.quantization.is_some() {
        return Err(format!(
            "FLUX3 {repository}@{revision} requires its own dense released configuration; another checkpoint or packed source is not a substitute"
        ));
    }
    Ok(())
}

/// The preparer [`derive_local_tier`] runs for a FLUX 3 conversion: the inference converter over
/// the fetched original at `spec.source`, writing `spec.out_dir`. `repository`/`revision` are the
/// original's Hub identity, which the converter validates against the frozen FLUX 3 checkpoints
/// and records in the receipt.
pub(crate) fn flux3_preparer(
    conversion: Flux3Conversion,
    repository: String,
    revision: String,
) -> impl FnOnce(&PrepareSpec) -> Result<PrepareReport, String> + Send + 'static {
    move |spec: &PrepareSpec| {
        if spec.quantize != Some(conversion.quantize()) {
            return Err(format!(
                "{conversion:?} produces {:?}, but the derivation asked for {:?}",
                conversion.quantize(),
                spec.quantize
            ));
        }
        validate_flux3_dense_identity(&spec.source, &repository, &revision)?;
        let source = gen_core_flux3::native_convert::ConversionSource {
            root: spec.source.clone(),
            repository,
            revision,
        };
        let cancel = gen_core::CancelFlag::new();
        let receipt = match conversion {
            Flux3Conversion::MlxAffine { bits, group } => {
                gen_core_flux3::native_convert::convert_affine(
                    &source,
                    bits,
                    group,
                    &spec.out_dir,
                    &cancel,
                )
                .map_err(|error| error.to_string())?
            }
            Flux3Conversion::CandleGgmlQ4_1 | Flux3Conversion::CandleGgmlQ8_0 => {
                convert_flux3_ggml(conversion, &source, &spec.out_dir, &cancel)?
            }
        };
        let num_tensors = receipt
            .components
            .iter()
            .map(|component| component.quantized_targets.len() + component.dense_targets.len())
            .sum();
        Ok(PrepareReport {
            input_format: gen_core::core_llm::ModelFormat::Safetensors,
            quantized: spec.quantize,
            out_dir: spec.out_dir.clone(),
            num_tensors,
            passthrough: false,
        })
    }
}

/// The Candle GGML pack is Candle's own quantizer, linked only by the Candle runtime bundle.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn convert_flux3_ggml(
    conversion: Flux3Conversion,
    source: &gen_core_flux3::native_convert::ConversionSource,
    out_dir: &Path,
    cancel: &gen_core::CancelFlag,
) -> Result<gen_core_flux3::conversion::ConversionReceipt, String> {
    use gen_core_flux3::artifacts::NativeDitStorage;
    let storage = match conversion {
        Flux3Conversion::CandleGgmlQ4_1 => NativeDitStorage::GgmlQ4_1,
        Flux3Conversion::CandleGgmlQ8_0 => NativeDitStorage::GgmlQ8_0,
        Flux3Conversion::MlxAffine { .. } => {
            return Err("an MLX affine conversion is not a GGML pack".to_owned())
        }
    };
    runtime_cuda::providers::flux3::convert::convert_ggml(source, storage, out_dir, cancel)
        .map_err(|error| error.to_string())
}

/// A build without the Candle bundle cannot produce Candle's GGML blocks; it refuses rather than
/// writing anything (the job stays a typed failure, never a substitute format).
#[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
fn convert_flux3_ggml(
    conversion: Flux3Conversion,
    _source: &gen_core_flux3::native_convert::ConversionSource,
    _out_dir: &Path,
    _cancel: &gen_core::CancelFlag,
) -> Result<gen_core_flux3::conversion::ConversionReceipt, String> {
    Err(format!(
        "{conversion:?} is Candle's GGML pack; this worker does not link the Candle runtime"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use axum::extract::Path as AxumPath;
    use axum::routing::post;
    use axum::{Json, Router};

    const WEIGHTS: &[u8] = b"derived tier weights";

    fn pins(weights: &[u8]) -> Value {
        json!({
            "variant": "q8",
            "fromVariant": "bf16",
            "conversion": "yue2-ggml-tier-v1",
            "weightsFile": "model.safetensors",
            "weightsBytes": weights.len(),
            "weightsSha256": format!("{:x}", Sha256::digest(weights)),
        })
    }

    async fn stub_api() -> String {
        async fn progress(AxumPath(id): AxumPath<String>) -> Json<Value> {
            Json(job_json(&id))
        }
        async fn heartbeat() -> Json<Value> {
            Json(json!({}))
        }
        let app = Router::new()
            .route("/api/v1/jobs/:job_id/progress", post(progress))
            .route("/api/v1/workers/:worker_id/heartbeat", post(heartbeat));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        crate::load_progress::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    fn job_json(id: &str) -> Value {
        let mut value = serde_json::to_value(job(json!({}))).unwrap();
        value["id"] = json!(id);
        value
    }

    fn job(payload: Value) -> JobSnapshot {
        serde_json::from_value(json!({
            "id": "job-derive", "type": "model_download", "status": "running",
            "projectId": null, "projectName": null, "payload": payload, "result": {},
            "requestedGpu": "auto", "assignedGpu": null, "workerId": "w", "progress": 0,
            "stage": "running", "message": "", "error": null, "etaSeconds": null,
            "elapsedSeconds": null, "attempts": 1, "sourceJobId": null, "duplicateOfJobId": null,
            "cancelRequested": false, "createdAt": "2026-09-26T00:00:00Z",
            "updatedAt": "2026-09-26T00:00:00Z", "startedAt": null, "completedAt": null,
            "canceledAt": null, "lastHeartbeatAt": null
        }))
        .expect("job snapshot parses")
    }

    fn settings(api_url: String, data_dir: &Path) -> Settings {
        let mut settings = Settings::from_env();
        settings.api_url = api_url;
        settings.worker_id = "w".into();
        settings.data_dir = data_dir.to_path_buf();
        settings
    }

    /// A preparer standing in for the engine's: writes `bytes` as the tier weights.
    fn writer(
        bytes: &'static [u8],
        calls: Arc<AtomicUsize>,
    ) -> impl FnOnce(&PrepareSpec) -> Result<PrepareReport, String> + Send + 'static {
        move |spec: &PrepareSpec| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(spec.quantize, Some(Quantize::Q8));
            std::fs::create_dir_all(&spec.out_dir).unwrap();
            std::fs::write(spec.out_dir.join("model.safetensors"), bytes).unwrap();
            Ok(PrepareReport {
                input_format: gen_core::core_llm::ModelFormat::Safetensors,
                quantized: spec.quantize,
                out_dir: spec.out_dir.clone(),
                num_tensors: 1,
                passthrough: false,
            })
        }
    }

    #[tokio::test]
    async fn a_derived_tier_is_written_where_the_catalog_reads_it_and_verified() {
        let data = tempfile::tempdir().unwrap();
        let source = data.path().join("original");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("model.safetensors"), b"the bf16 original").unwrap();
        let settings = settings(stub_api().await, data.path());
        let api = ApiClient::new(&settings);
        let job = job(json!({ "modelId": "yue2", "localDerivation": pins(WEIGHTS) }));
        let calls = Arc::new(AtomicUsize::new(0));

        let dir = derive_local_tier(
            &api,
            &settings,
            &job,
            &source,
            writer(WEIGHTS, calls.clone()),
        )
        .await
        .expect("derivation succeeds");
        assert_eq!(
            dir,
            data.path().join("models/derived/yue2/q8/yue2-ggml-tier-v1")
        );
        assert_eq!(
            derived_snapshot_state(
                &dir,
                &local_derivation(&json!({ PAYLOAD_KEY: pins(WEIGHTS) })).unwrap()
            ),
            DerivedSnapshotState::Verified
        );
        // Mutation that reds this: dropping the Verified early return re-runs the conversion on
        // every install of an already derived tier.
        derive_local_tier(
            &api,
            &settings,
            &job,
            &source,
            writer(WEIGHTS, calls.clone()),
        )
        .await
        .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a verified tier is not re-derived"
        );
        assert_eq!(
            std::fs::read(source.join("model.safetensors")).unwrap(),
            b"the bf16 original",
            "the original is never touched"
        );
    }

    #[tokio::test]
    async fn a_derivation_that_does_not_match_its_pins_is_removed_and_fails() {
        let data = tempfile::tempdir().unwrap();
        let source = data.path().join("original");
        std::fs::create_dir_all(&source).unwrap();
        let settings = settings(stub_api().await, data.path());
        let api = ApiClient::new(&settings);
        let job = job(json!({ "modelId": "yue2", "localDerivation": pins(WEIGHTS) }));
        // Mutation that reds this: skipping the post-derivation verification reports a tier whose
        // bytes differ from the pins as installed.
        let error = derive_local_tier(
            &api,
            &settings,
            &job,
            &source,
            writer(b"other bytes", Arc::new(AtomicUsize::new(0))),
        )
        .await
        .expect_err("unpinned bytes are refused");
        assert!(error.to_string().contains("pinned weights"), "{error}");
        assert!(!data
            .path()
            .join("models/derived/yue2/q8/yue2-ggml-tier-v1")
            .exists());
        assert!(source.is_dir(), "the original is never removed");
    }

    /// Every FLUX 3 conversion id the catalog names is one this worker knows, produces the tier
    /// its row is keyed by, and derives from the dense `bf16` original on its own platform (Candle
    /// GGML off-Mac, MLX affine on macOS) — never from another derived or publisher package.
    #[test]
    fn flux3_catalog_derivations_name_known_conversions_for_their_platform() {
        let entry = crate::tests::builtin_model_entry("flux3_klein");
        let rows: Vec<&Value> = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row.get("localDerivation").is_some())
            .collect();
        assert_eq!(rows.len(), 4, "q4 + q8 on each of the two native platforms");
        for row in rows {
            let block = &row["localDerivation"];
            let id = block["conversion"].as_str().unwrap();
            let conversion = Flux3Conversion::from_id(id)
                .unwrap_or_else(|| panic!("{id}: no FLUX 3 converter of that name"));
            let variant = row["variant"].as_str().unwrap();
            assert_eq!(quantize_of(variant).unwrap(), conversion.quantize(), "{id}");
            assert_eq!(block["fromVariant"], "bf16", "{id}");
            let platforms: Vec<&str> = row["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .collect();
            match conversion {
                Flux3Conversion::MlxAffine { .. } => assert_eq!(platforms, ["macos"], "{id}"),
                _ => assert_eq!(platforms, ["windows", "linux"], "{id}"),
            }
            assert!(local_derivation(row).is_some(), "{id}: incomplete pins");
        }
    }

    /// A derivation asked for the wrong tier refuses before writing anything.
    #[test]
    fn a_flux3_preparer_refuses_a_tier_it_does_not_produce() {
        let dir = tempfile::tempdir().unwrap();
        let spec = PrepareSpec {
            source: dir.path().join("source"),
            out_dir: dir.path().join("out"),
            quantize: Some(Quantize::Q4),
        };
        let error = flux3_preparer(
            Flux3Conversion::MlxAffine { bits: 8, group: 64 },
            "black-forest-labs/flux-3-image-klein".into(),
            "8454017f9ebaada30c998fccd9f67a80c7580f68".into(),
        )(&spec)
        .expect_err("q8 conversion asked for q4");
        assert!(error.contains("Q8"), "{error}");
        assert!(!spec.out_dir.exists());
    }

    /// Observe actual conversion bytes without promoting a self-reported receipt to trusted pins.
    /// This seam is also used with tiny local fixture files; those observations are NOT model proof.
    fn observe_flux3_conversion(
        model_id: &str,
        repository: &str,
        revision: &str,
        conversion_id: &str,
        source: &Path,
        output: &Path,
    ) -> Result<Value, String> {
        use gen_core_flux3::conversion::{hash_source_file, open_conversion};
        let conversion = Flux3Conversion::from_id(conversion_id)
            .ok_or_else(|| format!("unknown FLUX3 conversion {conversion_id}"))?;
        let expected_model = match repository {
            gen_core_flux3::DISTILLED_WEIGHTS_REPO => "flux3_klein",
            gen_core_flux3::BASE_WEIGHTS_REPO => "flux3_klein_base",
            _ => return Err("unknown FLUX3 checkpoint repository".into()),
        };
        if model_id != expected_model {
            return Err("FLUX3 capture model/repository mismatch".into());
        }
        let receipt = open_conversion(output).map_err(|e| e.to_string())?;
        if receipt.source_repository != repository || receipt.source_revision != revision {
            return Err("FLUX3 capture source identity differs from the conversion receipt".into());
        }
        let (algorithm, bits, group, backend, format) = match conversion {
            Flux3Conversion::CandleGgmlQ4_1 => ("ggml-q4_1", 4, 32, "candle", "ggml-q4_1"),
            Flux3Conversion::CandleGgmlQ8_0 => ("ggml-q8_0", 8, 32, "candle", "ggml-q8_0"),
            Flux3Conversion::MlxAffine { bits, group } => {
                ("mlx-affine", bits, group, "mlx", "mlx-affine")
            }
        };
        if receipt.algorithm != algorithm
            || receipt.bits != Some(bits)
            || receipt.group_size != Some(group)
        {
            return Err("FLUX3 capture conversion recipe differs from the actual package".into());
        }
        receipt.verify_external(source).map_err(|e| e.to_string())?;
        let mut source_inventory = Vec::new();
        for (name, expected) in &receipt.source_artifacts {
            let actual = hash_source_file(&source.join(name)).map_err(|e| e.to_string())?;
            if &actual.sha256 != expected {
                return Err(format!("FLUX3 source checksum changed: {name}"));
            }
            source_inventory.push(json!({"path":name,"bytes":actual.bytes,"sha256":actual.sha256}));
        }
        let weights =
            hash_source_file(&output.join("model.safetensors")).map_err(|e| e.to_string())?;
        let tier = if bits == 4 { "q4" } else { "q8" };
        Ok(json!({
            "schemaVersion":1, "evidenceKind":"conversion_observation_not_trusted_pins",
            "artifactKey":format!("{model_id}:{backend}:{tier}"),
            "modelId":model_id,"backend":backend,"tier":tier,
            "repository":repository,"resolvedRevision":revision,
            "conversionId":conversion_id,"backendFormat":format,
            "weightsFile":"model.safetensors","weightsBytes":weights.bytes,
            "weightsSha256":weights.sha256,"sourceInventory":source_inventory,
            "conversionReceipt":receipt.to_json().map_err(|e| e.to_string())?,
        }))
    }

    #[test]
    fn flux3_capture_refuses_distilled_config_presented_as_base_or_wrong_revision() {
        use gen_core_flux3::{
            BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION, DISTILLED_WEIGHTS_REPO,
            DISTILLED_WEIGHTS_REVISION,
        };
        let dir = tempfile::tempdir().unwrap();
        // REAL released distilled key structure (8454017f): `dit.flow_map: true`, 16 steps, no
        // guidance. Tiny dimensions; identity never depends on geometry.
        let distilled_config = json!({
            "format_version":1,"model":"flux-3-image-klein","quantization":null,"torch_dtype":"bfloat16",
            "dit":{"hidden":64,"heads":2,"depth":1,"depth_single":1,"mlp_ratio":3.0,
                "axes_dim":[8,8,8,8],"theta":10000,"in_channels":{"image":16,"image_cond":16,"text":256},
                "out_modes":["image"],"vec_in_dim":null,"image_rope_scale":0.5,"flow_map":true},
            "text_encoder":{"path":"text_encoder","tokenizer":"text_encoder","layers":[1],"pad_multiple":80,"max_tokens":8192,"dtype":"bfloat16"},
            "vae":{"path":"ae.safetensors","dtype":"float32"},
            "sampler":{"steps":16,"guidance":null,"alpha":6.93,"size":[1024,1024]}
        });
        // REAL released base key structure (1469d27c): the SAME `model`, NO `dit.flow_map` key,
        // CFG sampler defaults.
        let mut base_config = distilled_config.clone();
        base_config["dit"]
            .as_object_mut()
            .unwrap()
            .remove("flow_map");
        base_config["sampler"]["steps"] = json!(50);
        base_config["sampler"]["guidance"] = json!(3.5);
        let write = |config: &Value, distilled_header: Option<bool>| {
            std::fs::write(
                dir.path().join("config.json"),
                serde_json::to_vec(config).unwrap(),
            )
            .unwrap();
            let weights = dir.path().join("model.safetensors");
            match distilled_header {
                // Real released tensor names/dtypes, tiny shapes, actual small zero payload.
                Some(distilled) => {
                    crate::tests::write_flux3_dit_stub(dir.path(), distilled);
                }
                None => {
                    let _ = std::fs::remove_file(&weights);
                }
            }
        };
        let base =
            || validate_flux3_dense_identity(dir.path(), BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION);
        let distilled = || {
            validate_flux3_dense_identity(
                dir.path(),
                DISTILLED_WEIGHTS_REPO,
                DISTILLED_WEIGHTS_REVISION,
            )
        };

        write(&distilled_config, Some(true));
        assert_eq!(distilled(), Ok(()));
        assert!(base().is_err(), "a flow-map checkpoint is never the base");
        assert!(validate_flux3_dense_identity(
            dir.path(),
            DISTILLED_WEIGHTS_REPO,
            BASE_WEIGHTS_REVISION
        )
        .is_err());

        write(&base_config, Some(false));
        assert_eq!(base(), Ok(()));
        assert!(
            distilled().is_err(),
            "a non-flow-map checkpoint is never the distilled"
        );

        // The native HEADER is authoritative: a base-shaped config cannot relabel distilled weights.
        write(&base_config, Some(true));
        assert!(
            base().is_err(),
            "ref_time_in in the header resolves the flow map, not the config"
        );
        // An explicit config flag must agree with the header; an explicit `false` is the base.
        let mut explicit = base_config.clone();
        explicit["dit"]["flow_map"] = json!(true);
        write(&explicit, Some(false));
        assert!(
            base().is_err(),
            "config/header flow-map disagreement is refused"
        );
        explicit["dit"]["flow_map"] = json!(false);
        write(&explicit, Some(false));
        assert_eq!(base(), Ok(()));
        // No native weights header: a typed refusal, never a config-only identity.
        write(&base_config, None);
        assert!(base().is_err(), "missing model.safetensors is refused");

        // The base identity refuses a renamed/foreign model label or packed storage.
        for (key, value) in [
            ("model", json!("flux-3-image-klein-base")),
            ("model", json!("flux-2-klein")),
            ("quantization", json!("fp8r")),
        ] {
            let mut changed = base_config.clone();
            changed[key] = value;
            write(&changed, Some(false));
            assert!(base().is_err(), "{key} mutation must not pass as base");
        }
    }

    #[test]
    fn flux3_artifact_observations_bind_checkpoint_recipe_full_inventory_and_actual_bytes() {
        use gen_core_flux3::conversion::{
            hash_source_file, ComponentInventory, ComponentPlacement, ComponentRole,
            ConversionReceipt,
        };
        use std::collections::BTreeMap;
        // Deliberately NOT tensors or real-weight evidence: these tiny files exercise the capture
        // seam's actual disk hashing and receipt validation, not the native quantizer's numerics.
        for (model, repository, revision) in [
            (
                "flux3_klein",
                gen_core_flux3::DISTILLED_WEIGHTS_REPO,
                gen_core_flux3::DISTILLED_WEIGHTS_REVISION,
            ),
            (
                "flux3_klein_base",
                gen_core_flux3::BASE_WEIGHTS_REPO,
                gen_core_flux3::BASE_WEIGHTS_REVISION,
            ),
        ] {
            for (id, conversion) in Flux3Conversion::ALL {
                let dir = tempfile::tempdir().unwrap();
                let source = dir.path().join("source");
                let output = dir.path().join("output");
                std::fs::create_dir_all(source.join("text_encoder")).unwrap();
                std::fs::create_dir(&output).unwrap();
                std::fs::write(
                    source.join("model.safetensors"),
                    format!("fixture original {model}"),
                )
                .unwrap();
                let mut components = Vec::new();
                let mut source_artifacts = BTreeMap::new();
                let source_weight = hash_source_file(&source.join("model.safetensors")).unwrap();
                source_artifacts.insert("model.safetensors".to_owned(), source_weight.sha256);
                for (role, names) in [
                    (
                        ComponentRole::TextEncoder,
                        vec![
                            "text_encoder/config.json",
                            "text_encoder/tokenizer.json",
                            "text_encoder/tokenizer_config.json",
                            "text_encoder/model.safetensors",
                        ],
                    ),
                    (ComponentRole::Autoencoder, vec!["ae.safetensors"]),
                ] {
                    let mut files: BTreeMap<String, _> = BTreeMap::new();
                    for name in names {
                        std::fs::write(source.join(name), format!("fixture {model} {name}"))
                            .unwrap();
                        let file = hash_source_file(&source.join(name)).unwrap();
                        source_artifacts.insert(name.into(), file.sha256.clone());
                        files.insert(name.into(), file);
                    }
                    components.push(ComponentInventory {
                        role,
                        placement: ComponentPlacement::External,
                        source_repository: repository.into(),
                        source_revision: revision.into(),
                        source_artifacts: files
                            .iter()
                            .map(|(p, f)| (p.clone(), f.sha256.clone()))
                            .collect(),
                        representation: "fixture-dense-not-real-weights".into(),
                        files,
                        quantized_targets: vec![],
                        dense_targets: vec![],
                    });
                }
                // The converted package is a native checkpoint boundary: real released tensor
                // NAMES/dtypes (flow-map `ref_time_in` only for the distilled source), tiny
                // shapes, actual small zero payload. Not packed numerics or real weights.
                crate::tests::write_flux3_dit_stub(&output, model == "flux3_klein");
                let weight = hash_source_file(&output.join("model.safetensors")).unwrap();
                components.push(ComponentInventory {
                    role: ComponentRole::Dit,
                    placement: ComponentPlacement::Owned,
                    source_repository: repository.into(),
                    source_revision: revision.into(),
                    source_artifacts: BTreeMap::from([(
                        "model.safetensors".into(),
                        source_artifacts["model.safetensors"].clone(),
                    )]),
                    representation: "fixture-packed-not-real-weights".into(),
                    files: BTreeMap::from([("model.safetensors".into(), weight.clone())]),
                    quantized_targets: vec!["fixture.weight".into()],
                    dense_targets: vec![],
                });
                let (algorithm, bits, group_size, backend) = match conversion {
                    Flux3Conversion::CandleGgmlQ4_1 => ("ggml-q4_1", 4, 32, "candle"),
                    Flux3Conversion::CandleGgmlQ8_0 => ("ggml-q8_0", 8, 32, "candle"),
                    Flux3Conversion::MlxAffine { bits, group } => {
                        ("mlx-affine", bits, group, "mlx")
                    }
                };
                // Real released config key structure: distilled declares `flow_map: true`, the
                // base omits it; the converter replaces `quantization` with its storage marker.
                let mut package_config = json!({
                    "format_version":1,"model":"flux-3-image-klein","torch_dtype":"bfloat16",
                    "quantization":{"format":algorithm,"bits":bits,"group_size":group_size,"algorithm_version":1},
                    "dit":{"hidden":64,"heads":2,"depth":1,"depth_single":1,"mlp_ratio":3.0,
                        "axes_dim":[8,8,8,8],"theta":10000,"in_channels":{"image":16,"image_cond":16,"text":256},
                        "out_modes":["image"],"vec_in_dim":null,"image_rope_scale":0.5,"flow_map":true},
                    "text_encoder":{"path":"text_encoder","tokenizer":"text_encoder","layers":[1],"pad_multiple":80,"max_tokens":8192,"dtype":"bfloat16"},
                    "vae":{"path":"ae.safetensors","dtype":"float32"},
                    "sampler":{"steps":16,"guidance":null,"alpha":6.93,"size":[1024,1024]}
                });
                if model == "flux3_klein_base" {
                    package_config["dit"]
                        .as_object_mut()
                        .unwrap()
                        .remove("flow_map");
                    package_config["sampler"]["steps"] = json!(50);
                    package_config["sampler"]["guidance"] = json!(3.5);
                }
                std::fs::write(
                    output.join("config.json"),
                    serde_json::to_vec(&package_config).unwrap(),
                )
                .unwrap();
                let mut package_files = BTreeMap::new();
                for name in ["config.json", "LICENSE.md", "NOTICE.md"] {
                    if name != "config.json" {
                        std::fs::write(output.join(name), b"fixture").unwrap();
                    }
                    package_files
                        .insert(name.into(), hash_source_file(&output.join(name)).unwrap());
                }
                let receipt = ConversionReceipt {
                    source_repository: repository.into(),
                    source_revision: revision.into(),
                    source_artifacts,
                    algorithm: algorithm.into(),
                    algorithm_version: 1,
                    bits: Some(bits),
                    group_size: Some(group_size),
                    components,
                    package_files,
                };
                std::fs::write(
                    output.join("flux3-conversion-receipt.json"),
                    serde_json::to_vec(&receipt.to_json().unwrap()).unwrap(),
                )
                .unwrap();
                let observe =
                    || observe_flux3_conversion(model, repository, revision, id, &source, &output);
                let one = observe().unwrap();
                assert_eq!(
                    one,
                    observe().unwrap(),
                    "observations are reproducible, with no timestamp/path entropy"
                );
                assert_eq!(one["artifactKey"], format!("{model}:{backend}:q{bits}"));
                assert_eq!(one["weightsBytes"], weight.bytes);
                assert_eq!(one["weightsSha256"], weight.sha256);
                assert_eq!(one["sourceInventory"].as_array().unwrap().len(), 6);
                assert_eq!(
                    one["conversionReceipt"]["components"]
                        .as_array()
                        .unwrap()
                        .len(),
                    3
                );
                let wrong = if model == "flux3_klein" {
                    "flux3_klein_base"
                } else {
                    "flux3_klein"
                };
                assert!(observe_flux3_conversion(
                    wrong, repository, revision, id, &source, &output
                )
                .is_err());
                std::fs::write(source.join("ae.safetensors"), b"corrupted component").unwrap();
                assert!(
                    observe().is_err(),
                    "external corruption must not be captured"
                );
                std::fs::write(
                    source.join("ae.safetensors"),
                    format!("fixture {model} ae.safetensors"),
                )
                .unwrap();
                std::fs::write(output.join("model.safetensors"), b"corrupted packed bytes")
                    .unwrap();
                assert!(observe().is_err(), "owned corruption must not be captured");
            }
        }
    }

    /// Explicit local-only artifact preparation/capture for E5. Unlike the legacy pin verifier,
    /// this does not require existing q4/q8 catalog rows (base rows MUST wait for genuine pins).
    /// Every requested recipe must be linked; none is silently skipped. Existing outputs are
    /// verified, never deleted/replaced. The report contains observations, not trusted expectations.
    #[test]
    #[ignore = "operator-only conversion; requires explicit model, local source/output and conversion IDs"]
    fn flux3_capture_conversion_artifacts() {
        let model = std::env::var("FLUX3_CAPTURE_MODEL_ID")
            .expect("FLUX3_CAPTURE_MODEL_ID: flux3_klein or flux3_klein_base");
        assert!(matches!(model.as_str(), "flux3_klein" | "flux3_klein_base"));
        let source = PathBuf::from(
            std::env::var("FLUX3_CAPTURE_SOURCE_DIR")
                .expect("FLUX3_CAPTURE_SOURCE_DIR: caller-provisioned ORIGINAL dense path only"),
        );
        let out = PathBuf::from(
            std::env::var("FLUX3_CAPTURE_OUT_DIR")
                .expect("FLUX3_CAPTURE_OUT_DIR: caller-owned output directory"),
        );
        let recipes = std::env::var("FLUX3_CAPTURE_CONVERSIONS")
            .expect("FLUX3_CAPTURE_CONVERSIONS: comma-separated exact conversion IDs");
        let entry = crate::tests::builtin_model_entry(&model);
        let original = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["variant"] == "bf16")
            .expect("dense original");
        let repository = original["repo"].as_str().unwrap();
        let revision = original["revision"].as_str().unwrap();
        validate_flux3_dense_identity(&source, repository, revision).unwrap();
        let mut reports = std::collections::BTreeMap::new();
        for id in recipes.split(',') {
            let conversion = Flux3Conversion::from_id(id).expect("exact native conversion ID");
            let tier = if conversion.quantize() == Quantize::Q4 {
                "q4"
            } else {
                "q8"
            };
            let output = out.join(&model).join(tier).join(id);
            assert!(!reports.contains_key(id), "duplicate recipe");
            if !output.exists() {
                flux3_preparer(conversion, repository.into(), revision.into())(&PrepareSpec {
                    source: source.clone(),
                    out_dir: output.clone(),
                    quantize: Some(conversion.quantize()),
                })
                .unwrap_or_else(|error| panic!("{model}:{id}: {error}"));
            }
            crate::flux3_converted_bundle_check(&source, &output).unwrap();
            let report =
                observe_flux3_conversion(&model, repository, revision, id, &source, &output)
                    .unwrap();
            reports.insert(id.to_owned(), report);
        }
        println!("{}", serde_json::to_string_pretty(&reports).unwrap());
    }

    /// Reproducibility receipt for the catalog pins (sc-26241). Converts the real pinned dense
    /// snapshot (`FLUX3_KLEIN_SNAPSHOT_DIR`, a caller-provisioned path) with each conversion this
    /// build links into `FLUX3_DERIVE_OUT_DIR`, then checks the output against the catalog's
    /// `weightsBytes` / `weightsSha256` — the pins are these outputs, not estimates. The Candle
    /// GGML pair needs a `backend-candle` build; the MLX affine pair is backend-neutral and runs on
    /// any host. CPU only; tens of GB of output.
    #[test]
    #[ignore = "converts the real 43 GB FLUX 3 snapshot; set FLUX3_KLEIN_SNAPSHOT_DIR and FLUX3_DERIVE_OUT_DIR"]
    fn flux3_conversions_reproduce_the_catalog_pins() {
        let source = PathBuf::from(
            std::env::var("FLUX3_KLEIN_SNAPSHOT_DIR").expect("FLUX3_KLEIN_SNAPSHOT_DIR"),
        );
        let out =
            PathBuf::from(std::env::var("FLUX3_DERIVE_OUT_DIR").expect("FLUX3_DERIVE_OUT_DIR"));
        let entry = crate::tests::builtin_model_entry("flux3_klein");
        let original = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["variant"] == "bf16")
            .expect("the dense original row")
            .clone();
        let mut mismatches = Vec::new();
        for row in entry["downloads"].as_array().unwrap() {
            let Some(derivation) = local_derivation(row) else {
                continue;
            };
            let conversion = Flux3Conversion::from_id(&derivation.conversion).unwrap();
            let linked = cfg!(all(not(target_os = "macos"), feature = "backend-candle"))
                || matches!(conversion, Flux3Conversion::MlxAffine { .. });
            if !linked {
                eprintln!(
                    "{}: not linked by this build; skipped",
                    derivation.conversion
                );
                continue;
            }
            let dir = out.join(&derivation.conversion);
            if derived_snapshot_state(&dir, &derivation) != DerivedSnapshotState::Verified {
                if dir.exists() {
                    std::fs::remove_dir_all(&dir).unwrap();
                }
                flux3_preparer(
                    conversion,
                    original["repo"].as_str().unwrap().to_owned(),
                    original["revision"].as_str().unwrap().to_owned(),
                )(&PrepareSpec {
                    source: source.clone(),
                    out_dir: dir.clone(),
                    quantize: Some(conversion.quantize()),
                })
                .unwrap_or_else(|error| panic!("{}: {error}", derivation.conversion));
            }
            crate::flux3_converted_bundle_check(&source, &dir)
                .unwrap_or_else(|error| panic!("{} inventory: {error}", derivation.conversion));
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            if matches!(
                conversion,
                Flux3Conversion::CandleGgmlQ4_1 | Flux3Conversion::CandleGgmlQ8_0
            ) {
                use gen_core_flux3::artifacts::NativeDitStorage;
                use runtime_cuda::providers::flux3::memory_strategy::{
                    dit_bytes, native_dit_bytes,
                };
                let config = gen_core_flux3::resolve_checkpoint(&dir).unwrap();
                let storage = NativeDitStorage::from_package(&dir).unwrap();
                let dense = dit_bytes(
                    "flux3_klein",
                    &source.join("model.safetensors"),
                    gen_core::Precision::Bf16,
                )
                .unwrap();
                let packed = native_dit_bytes(
                    "flux3_klein",
                    &dir.join("model.safetensors"),
                    gen_core::Precision::Bf16,
                    storage,
                    &config.dit,
                )
                .unwrap();
                let floor =
                    (32.0 - (dense as f64 - packed as f64) / 1024_f64.powi(3)).ceil() as u64;
                let tier = if conversion.quantize() == Quantize::Q4 {
                    "q4"
                } else {
                    "q8"
                };
                assert_eq!(entry["candle"]["minMemoryGbByTier"][tier], floor,
                    "the derived floor replaces the dense DiT term with inference's exact packed pricing");
            }
            let weights = dir.join(&derivation.weights_file);
            let bytes = std::fs::metadata(&weights).unwrap().len();
            let mut hasher = Sha256::new();
            std::io::copy(&mut std::fs::File::open(&weights).unwrap(), &mut hasher).unwrap();
            let sha = format!("{:x}", hasher.finalize());
            eprintln!(
                "{}: weightsBytes {bytes} weightsSha256 {sha}",
                derivation.conversion
            );
            if bytes != derivation.weights_bytes || sha != derivation.weights_sha256 {
                mismatches.push(derivation.conversion.clone());
            }
        }
        assert!(
            mismatches.is_empty(),
            "outputs differ from the catalog pins: {mismatches:?}"
        );
    }
}
