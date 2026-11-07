//! Hosted-only CPU integration: mounted retry/repair API plus the actual utility worker.
//! Synthetic headers/components only; no native model execution or external download endpoint.
use super::support::*;
use sceneworks_core::jobs_store::{CreateJob, JobsStore};
use sceneworks_worker::minimax_h3_import_fixture as fixture;

fn manifests(settings: &crate::Settings, models: Value) {
    let dir = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&dir).unwrap();
    for (file, key) in [
        ("builtin.models.jsonc", "models"),
        ("user.models.jsonc", "models"),
        ("builtin.loras.jsonc", "loras"),
        ("user.loras.jsonc", "loras"),
        ("builtin.recipe-presets.jsonc", "presets"),
        ("user.recipe-presets.jsonc", "presets"),
    ] {
        let rows = if file == "user.models.jsonc" {
            models.clone()
        } else {
            json!([])
        };
        std::fs::write(
            dir.join(file),
            json!({"schemaVersion":1,key:rows}).to_string(),
        )
        .unwrap();
    }
}

struct Runtime {
    server: tokio::task::JoinHandle<()>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Drop for Runtime {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
        self.server.abort();
    }
}
async fn start_worker(app: axum::Router, settings: &crate::Settings) -> Runtime {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap();
    });
    let worker_settings = sceneworks_worker::Settings {
        api_url: format!("http://{address}"),
        access_token: None,
        data_dir: settings.data_dir.clone(),
        config_dir: settings.config_dir.clone(),
        resolved_cache: Default::default(),
        worker_id: "import-recovery-worker".into(),
        gpu_id: "cpu".into(),
        is_child_worker: true,
        poll_seconds: 1,
        heartbeat_seconds: 1,
        shutdown_timeout_seconds: 1,
        huggingface_base_url: "http://127.0.0.1:9".into(),
        huggingface_token: None,
        credentials: vec![],
        max_lora_url_bytes: 1 << 20,
        max_model_url_bytes: 1 << 20,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: false,
        backend_candle_enabled: false,
        external_model_roots: vec![],
        gpu_memory_limit_bytes: 0,
    };
    let (stop, stop_rx) = tokio::sync::oneshot::channel();
    let worker = std::thread::Builder::new().name("import-recovery-test-worker".into())
        .stack_size(sceneworks_worker::WORKER_ENTRY_STACK_BYTES).spawn(move || {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async move {
                tokio::select! {
                    result = sceneworks_worker::run_worker_loop(worker_settings) => { result.unwrap(); }
                    _ = stop_rx => {}
                }
            });
        }).unwrap();
    Runtime {
        server,
        stop: Some(stop),
        worker: Some(worker),
    }
}
async fn terminal(app: &axum::Router, id: &str) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            let (status, job) = request(
                app.clone(),
                "GET",
                &format!("/api/v1/jobs/{id}"),
                Value::Null,
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{job}");
            if matches!(
                job["status"].as_str(),
                Some("completed" | "failed" | "canceled" | "interrupted")
            ) {
                return job;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("synthetic import must reach a terminal state within 30 seconds")
}
fn seed_import(
    settings: &crate::Settings,
    model: &fixture::ManagedMiniMaxH3Import,
) -> sceneworks_core::contracts::JobSnapshot {
    let store = JobsStore::new(&settings.jobs_db_path);
    store.initialize().unwrap();
    let mut entry = model.entry.clone();
    entry.remove("importPlan");
    let payload = json!({"modelId":entry["id"],"modelName":entry["name"],"sourcePath":settings.data_dir.join("models/incoming").join(entry["id"].as_str().unwrap()),
        "targetDir":model.install_dir,"manifestPath":settings.config_dir.join("manifests/user.models.jsonc"),
        "ownershipMode":"managed","manifestEntry":entry,"importProvenance":{"source":"local"}});
    let job = store
        .create_job(CreateJob {
            job_type: crate::JobType::ModelImport,
            project_id: None,
            project_name: None,
            payload: payload.as_object().unwrap().clone(),
            requested_gpu: "auto".into(),
            source_job_id: None,
            duplicate_of_job_id: None,
            attempts: 1,
            initial_status: None,
        })
        .unwrap();
    // A stopped original cannot be claimed by the live worker. The durable installation/marker
    // below emulates a process dying specifically after commit and before manifest publication.
    store.cancel_job(&job.id).unwrap();
    job
}

#[tokio::test]
async fn actual_api_retry_chain_recovers_committed_h3_and_refuses_unrelated_or_changed_imports() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let model = fixture::stage_managed_minimax_h3_import(&settings.data_dir, "recover_original");
    manifests(&settings, json!([]));
    let original = seed_import(&settings, &model);
    sceneworks_worker::seal_import_recovery_fixture(
        &settings.data_dir,
        &model.install_dir,
        &original.payload,
        &original.id,
    )
    .await
    .unwrap();
    let source = settings.data_dir.join("models/incoming/recover_original");
    std::fs::remove_dir_all(&source).unwrap();
    let primary = model
        .install_dir
        .join("diffusion_pytorch_model.safetensors");
    let original_bytes = std::fs::read(&primary).unwrap();
    let app = create_app(settings.clone()).unwrap();
    let (status, unrelated) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/duplicate", original.id),
        json!({"payloadChanges": {}}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{unrelated}");
    assert!(
        unrelated["sourceJobId"].is_null(),
        "a duplicate is not authenticated retry ancestry"
    );
    let (status, changed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/retry", original.id),
        json!({"payloadChanges":{"expectedSha256":"f".repeat(64)}}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{changed}");
    let (status, first_retry) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/retry", original.id),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{first_retry}");
    request(
        app.clone(),
        "POST",
        &format!(
            "/api/v1/jobs/{}/cancel",
            first_retry["id"].as_str().unwrap()
        ),
        json!({}),
    )
    .await;
    let (status, retry) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/retry", first_retry["id"].as_str().unwrap()),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{retry}");
    assert_ne!(retry["id"], original.id);
    assert_eq!(retry["sourceJobId"], first_retry["id"]);
    assert_eq!(retry["attempts"], 3);
    assert_eq!(
        retry["payload"],
        serde_json::to_value(&original.payload).unwrap()
    );
    let _runtime = start_worker(app.clone(), &settings).await;
    let unrelated = terminal(&app, unrelated["id"].as_str().unwrap()).await;
    assert_eq!(
        unrelated["status"], "failed",
        "unrelated install adoption must fail: {unrelated}"
    );
    let changed = terminal(&app, changed["id"].as_str().unwrap()).await;
    assert_eq!(
        changed["status"], "failed",
        "changed request cannot adopt old bytes: {changed}"
    );
    assert!(
        changed["error"]
            .as_str()
            .unwrap_or_default()
            .contains("Committed import request changed"),
        "the changed request must fail at the commitment boundary: {changed}"
    );
    let completed = terminal(&app, retry["id"].as_str().unwrap()).await;
    assert_eq!(
        completed["status"], "completed",
        "actual new-ID retry chain must recover: {completed}"
    );
    assert_eq!(completed["result"]["modelId"], "recover_original");
    assert_eq!(
        std::fs::read(primary).unwrap(),
        original_bytes,
        "no retry may replace committed bytes"
    );
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(settings.config_dir.join("manifests/user.models.jsonc")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["models"].as_array().unwrap().len(), 1);
    assert_eq!(manifest["models"][0]["id"], "recover_original");
    assert_eq!(
        manifest["models"][0]["importPlan"]["semanticDigest"],
        model.semantic_digest
    );
    let (status, again) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{}/retry", completed["id"].as_str().unwrap()),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{again}");
    assert_ne!(again["id"], completed["id"]);
    let repeated = terminal(&app, again["id"].as_str().unwrap()).await;
    assert_eq!(repeated["status"], "completed", "{repeated}");
    let unchanged: Value = serde_json::from_slice(
        &std::fs::read(settings.config_dir.join("manifests/user.models.jsonc")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        unchanged, manifest,
        "actual API retry after recovered publication is idempotent"
    );
}

#[tokio::test]
async fn managed_repair_api_keeps_original_selection_and_commits_separate_installation() {
    check_managed_repair(false, None).await;
}

#[tokio::test]
async fn missing_managed_original_is_repaired_through_actual_submission_not_a_second_selection() {
    check_managed_repair(true, None).await;
}

#[tokio::test]
async fn managed_repair_publication_cas_protects_changed_and_deleted_selections() {
    for change in ["changed", "deleted"] {
        check_managed_repair(false, Some(change)).await;
    }
}

async fn check_managed_repair(missing_original: bool, publication_change: Option<&str>) {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let model = fixture::stage_managed_minimax_h3_import(&settings.data_dir, "repair_selection");
    let original = Value::Object(model.entry.clone());
    manifests(&settings, json!([original]));
    let original_bytes = std::fs::read(
        model
            .install_dir
            .join("diffusion_pytorch_model.safetensors"),
    )
    .unwrap();
    let mut downloads = fixture::component_downloads("q4", "fl2va");
    downloads.extend([
        json!({"provider":"huggingface","repo":"fixture/base-config","revision":"3".repeat(40),"variant":"q4","default":true,"files":["transformer/config.json"]}),
        json!({"provider":"huggingface","repo":"fixture/base-config","revision":"3".repeat(40),"variant":"bf16","files":["transformer/config.json"]}),
    ]);
    let base = json!({"id":"minimax_h3","family":"minimax-h3","type":"video","name":"Synthetic compatibility base","downloads":downloads});
    std::fs::write(
        settings.config_dir.join("manifests/builtin.models.jsonc"),
        json!({"schemaVersion":1,"models":[base]}).to_string(),
    )
    .unwrap();
    let protected_original = if missing_original {
        let protected = settings.data_dir.join("models/protected-original");
        std::fs::rename(&model.install_dir, &protected).unwrap();
        protected
    } else {
        model.install_dir.clone()
    };
    let app = create_app(settings.clone()).unwrap();
    let body = json!({"modelId":"repair_selection","name":"New display name","family":"minimax-h3","type":"video",
        "baseModel":"minimax_h3","minimaxH3Partition":"fl2va", "licenseAcknowledged":true, "source":{"kind":"localPath","path":settings.data_dir.join("models/incoming/repair_selection")}});
    let (status, duplicate) =
        request(app.clone(), "POST", "/api/v1/models/import", body.clone()).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "ordinary create must still reject existing IDs: {duplicate}"
    );
    // Explicit acknowledgment models only this isolated synthetic API request. It is not user
    // consent or authority for real source acquisition; every fixture closure is already local,
    // the base config is source-owned, and the worker network endpoint is unroutable.
    let (status, queued) = request(
        app.clone(),
        "POST",
        "/api/v1/models/repair_selection/repair",
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{queued}");
    assert_eq!(queued["payload"]["modelId"], "repair_selection");
    assert_eq!(
        queued["payload"]["manifestEntry"]["name"],
        "New display name"
    );
    assert_ne!(
        queued["payload"]["targetDir"],
        model.install_dir.to_string_lossy().as_ref()
    );
    assert_eq!(
        queued["payload"]["repairExpectedBinding"]["importPlan"],
        model.entry["importPlan"]
    );
    if let Some(change) = publication_change {
        let replacement = if change == "changed" {
            let other =
                fixture::stage_managed_minimax_h3_import(&settings.data_dir, "new_selection_owner");
            let mut entry = other.entry.clone();
            entry.insert("id".into(), json!("repair_selection"));
            manifests(&settings, json!([entry]));
            Some(other)
        } else {
            manifests(&settings, json!([]));
            None
        };
        let manifest_path = settings.config_dir.join("manifests/user.models.jsonc");
        let expected = std::fs::read(&manifest_path).unwrap();
        let _runtime = start_worker(app.clone(), &settings).await;
        let failed = terminal(&app, queued["id"].as_str().unwrap()).await;
        assert_eq!(failed["status"], "failed", "{change}: {failed}");
        assert!(
            !std::path::Path::new(queued["payload"]["targetDir"].as_str().unwrap()).exists(),
            "CAS refusal rolls back the replacement, not the original"
        );
        assert_eq!(
            std::fs::read(manifest_path).unwrap(),
            expected,
            "{change}: no stale publication/resurrection"
        );
        assert_eq!(
            std::fs::read(protected_original.join("diffusion_pytorch_model.safetensors")).unwrap(),
            original_bytes
        );
        if let Some(other) = replacement {
            assert!(
                other.install_dir.exists(),
                "new selection owner is protected"
            );
        }
        return;
    }
    let source_file = settings
        .data_dir
        .join("models/incoming/repair_selection/diffusion_pytorch_model.safetensors");
    std::fs::write(&source_file, b"invalid source after enqueue").unwrap();
    let _runtime = start_worker(app.clone(), &settings).await;
    let failed = terminal(&app, queued["id"].as_str().unwrap()).await;
    assert_eq!(failed["status"], "failed", "{failed}");
    assert!(
        !std::path::Path::new(queued["payload"]["targetDir"].as_str().unwrap()).exists(),
        "failed repair must roll back only its new installation"
    );
    let manifest_path = settings.config_dir.join("manifests/user.models.jsonc");
    let unchanged: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(
        unchanged["models"][0]["importPlan"],
        model.entry["importPlan"]
    );
    assert_eq!(
        std::fs::read(protected_original.join("diffusion_pytorch_model.safetensors")).unwrap(),
        original_bytes
    );
    let mut replacement = original_bytes.clone();
    *replacement.last_mut().unwrap() ^= 1;
    std::fs::write(&source_file, &replacement).unwrap();
    let (status, repaired) = request(
        app.clone(),
        "POST",
        "/api/v1/models/repair_selection/repair",
        body,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{repaired}");
    let completed = terminal(&app, repaired["id"].as_str().unwrap()).await;
    assert_eq!(completed["status"], "completed", "{completed}");
    let installed: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(installed["models"].as_array().unwrap().len(), 1);
    let selection = &installed["models"][0];
    assert_eq!(selection["id"], "repair_selection");
    assert_eq!(selection["name"], "New display name");
    assert_ne!(
        selection["importPlan"]["semanticDigest"],
        model.semantic_digest
    );
    assert_eq!(
        std::fs::read(
            std::path::Path::new(selection["paths"]["model"].as_str().unwrap())
                .join("diffusion_pytorch_model.safetensors")
        )
        .unwrap(),
        replacement
    );
    assert_eq!(
        std::fs::read(protected_original.join("diffusion_pytorch_model.safetensors")).unwrap(),
        original_bytes,
        "successful replacement also preserves the previous source/leases"
    );
}

#[tokio::test]
async fn actual_api_retry_refuses_marker_tampering_legacy_authority_and_replaced_selection() {
    for mode in [
        "marker-job",
        "marker-commitment",
        "legacy-unbound",
        "selection-replaced",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let settings = test_settings(&temp);
        let model =
            fixture::stage_managed_minimax_h3_import(&settings.data_dir, "guarded_original");
        manifests(&settings, json!([]));
        let original = seed_import(&settings, &model);
        sceneworks_worker::seal_import_recovery_fixture(
            &settings.data_dir,
            &model.install_dir,
            &original.payload,
            &original.id,
        )
        .await
        .unwrap();
        if mode == "legacy-unbound" {
            let store = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(
                &settings.data_dir,
            );
            let resolved = store.resolve(&model.checkpoint_id).unwrap();
            store
                .compile_managed_with_context(
                    "guarded_original",
                    "diffusion_pytorch_model.safetensors",
                    sceneworks_core::checkpoint_import::ManagedProvenanceV1::of_source(
                        "legacy-import",
                    ),
                    resolved.plan.context,
                )
                .unwrap();
        }
        let replacement = if mode == "selection-replaced" {
            let other =
                fixture::stage_managed_minimax_h3_import(&settings.data_dir, "different_owner");
            let mut entry = other.entry.clone();
            entry.insert("id".into(), json!("guarded_original"));
            manifests(&settings, json!([entry]));
            Some(other)
        } else {
            None
        };
        let before_manifest =
            std::fs::read(settings.config_dir.join("manifests/user.models.jsonc")).unwrap();
        let before_bytes = std::fs::read(
            model
                .install_dir
                .join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        let app = create_app(settings.clone()).unwrap();
        let (status, retry) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/jobs/{}/retry", original.id),
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{mode}: {retry}");
        let marker_path = model.install_dir.join(".sceneworks-download-complete.json");
        let mut marker: Value =
            serde_json::from_slice(&std::fs::read(&marker_path).unwrap()).unwrap();
        if mode == "marker-job" {
            marker["jobId"] = retry["id"].clone();
        }
        if mode == "marker-commitment" {
            marker["importCommitment"] = json!(format!("sha256:{}", "f".repeat(64)));
        }
        std::fs::write(marker_path, marker.to_string()).unwrap();
        let _runtime = start_worker(app.clone(), &settings).await;
        let failed = terminal(&app, retry["id"].as_str().unwrap()).await;
        assert_eq!(failed["status"], "failed", "{mode}: {failed}");
        let diagnostic = failed["error"].as_str().unwrap_or_default();
        let expected_reason = match mode {
            "marker-commitment" => "Committed import request changed",
            "marker-job" | "legacy-unbound" => "immutable installation ownership",
            "selection-replaced" => "selection changed before repair publication",
            _ => unreachable!(),
        };
        assert!(
            diagnostic.contains(expected_reason),
            "{mode}: actionable refusal must name {expected_reason}: {failed}"
        );
        assert_eq!(
            std::fs::read(settings.config_dir.join("manifests/user.models.jsonc")).unwrap(),
            before_manifest,
            "{mode}: selection must stay untouched"
        );
        assert_eq!(
            std::fs::read(
                model
                    .install_dir
                    .join("diffusion_pytorch_model.safetensors")
            )
            .unwrap(),
            before_bytes,
            "{mode}: no unrelated/legacy bytes may be removed"
        );
        if let Some(other) = replacement {
            assert!(other.install_dir.is_dir(), "replacement owner is protected");
        }
    }
}

#[tokio::test]
async fn managed_repair_api_refuses_linked_builtin_mismatched_and_missing_selections() {
    let temp = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp);
    let model = fixture::stage_managed_minimax_h3_import(&settings.data_dir, "managed_owner");
    let mut builtin = model.entry.clone();
    builtin.insert("id".into(), json!("minimax_h3"));
    let store =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    let source = settings.data_dir.join("models/incoming/managed_owner");
    let root = store.approve_root(source.parent().unwrap()).unwrap();
    let managed = store.resolve(&model.checkpoint_id).unwrap();
    let linked = store
        .compile_linked_with_context(&root.root_id, "managed_owner", managed.plan.context)
        .unwrap();
    let mut linked_entry = model.entry.clone();
    linked_entry.insert("id".into(), json!("linked_owner"));
    linked_entry.remove("paths");
    linked_entry.insert(
        "source".into(),
        json!({"provider":"linked-library","rootId":root.root_id,"relativePath":"managed_owner"}),
    );
    linked_entry.insert("importPlan".into(),json!({"checkpointId":linked.checkpoint_id,"planId":linked.plan.plan_id,"semanticDigest":linked.record.plan.semantic_digest}));
    let assembly = sceneworks_core::minimax_h3_import::MiniMaxH3ImportAssembly::from_declared_plan(
        &linked.plan,
        &linked_entry,
    )
    .unwrap();
    sceneworks_core::minimax_h3_import::stamp_imported_minimax_h3_entry(
        &mut linked_entry,
        &assembly,
    );
    manifests(&settings, json!([model.entry, builtin, linked_entry]));
    let app = create_app(settings.clone()).unwrap();
    let base = json!({"source":{"kind":"url","url":"https://example.test/synthetic.safetensors"}});
    for id in ["minimax_h3", "linked_owner", "managed_owner"] {
        let mut body = base.clone();
        if id == "managed_owner" {
            body["modelId"] = json!("another_model");
        }
        let (status, error) = request(
            app.clone(),
            "POST",
            &format!("/api/v1/models/{id}/repair"),
            body,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{id}: {error}");
    }
    let (status, error) = request(app, "POST", "/api/v1/models/missing/repair", base).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{error}");
    assert!(source.is_dir(), "linked originals remain untouched");
    assert!(
        model.install_dir.is_dir(),
        "unrelated managed install remains untouched"
    );
}
