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
    let task = tokio::task::spawn_blocking(move || {
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
    /// MLX affine W`bits`A16 at `group` (`gen_core::flux3::native_convert::convert_affine`, the
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
        let source = gen_core::flux3::native_convert::ConversionSource {
            root: spec.source.clone(),
            repository,
            revision,
        };
        let cancel = gen_core::CancelFlag::new();
        let receipt = match conversion {
            Flux3Conversion::MlxAffine { bits, group } => {
                gen_core::flux3::native_convert::convert_affine(
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
    source: &gen_core::flux3::native_convert::ConversionSource,
    out_dir: &Path,
    cancel: &gen_core::CancelFlag,
) -> Result<gen_core::flux3::conversion::ConversionReceipt, String> {
    use gen_core::flux3::artifacts::NativeDitStorage;
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
    _source: &gen_core::flux3::native_convert::ConversionSource,
    _out_dir: &Path,
    _cancel: &gen_core::CancelFlag,
) -> Result<gen_core::flux3::conversion::ConversionReceipt, String> {
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
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
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
