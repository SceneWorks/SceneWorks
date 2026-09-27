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
        DerivedSnapshotState::Verified => return Ok(out_dir),
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
        DerivedSnapshotState::Verified => Ok(out_dir),
        other => {
            let _ = std::fs::remove_dir_all(&out_dir);
            Err(WorkerError::Engine(format!(
                "the derived {variant} tier does not match the catalog's pinned weights \
                 ({other:?}); it was removed"
            )))
        }
    }
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
}
