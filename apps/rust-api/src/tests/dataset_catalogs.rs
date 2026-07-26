//! Dataset Catalog lifecycle, query-contract, and path-confinement route tests.

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use parquet::data_type::{ByteArray, ByteArrayType, Int32Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use sceneworks_core::catalog_store::{
    catalog_timestamp_now, CatalogProcessingLease, CatalogProcessingProgress,
    CatalogProcessingState, CatalogRegistry, CatalogSourceConfig, NewCatalogRecord,
};
use sceneworks_worker::catalog_parquet_scanner::MAX_PARQUET_SHARDS;

use super::support::*;

fn catalog_record(id: &str, medium: &str, person_count: u64) -> NewCatalogRecord {
    NewCatalogRecord {
        id: id.to_owned(),
        image_path: format!("images/{id}.jpg"),
        thumbnail_path: Some(format!("thumbnails/{id}.jpg")),
        embedding_path: None,
        artifact_path: None,
        metadata: json!({
            "medium": medium,
            "personCount": person_count,
            "analysis": { "fullBody": person_count == 1 }
        }),
    }
}

fn write_catalog_parquet(path: &std::path::Path, row_count: usize) {
    let schema = Arc::new(
        parse_message_type(
            "message catalog {
                REQUIRED BINARY URL (UTF8);
                REQUIRED BINARY TEXT (UTF8);
                REQUIRED INT32 WIDTH;
                REQUIRED INT32 HEIGHT;
            }",
        )
        .expect("test Parquet schema parses"),
    );
    let mut writer = SerializedFileWriter::new(
        File::create(path).expect("test Parquet file creates"),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .expect("test Parquet writer creates");
    let mut row_group = writer.next_row_group().expect("row group creates");
    let urls = (0..row_count)
        .map(|index| format!("https://example.test/{index}.jpg"))
        .collect::<Vec<_>>();
    let captions = (0..row_count)
        .map(|index| format!("catalog row {index}"))
        .collect::<Vec<_>>();
    let mut column_index = 0;
    while let Some(mut column) = row_group.next_column().expect("column advances") {
        match column_index {
            0 | 1 => {
                let values = if column_index == 0 {
                    urls.iter()
                } else {
                    captions.iter()
                }
                .map(|value| ByteArray::from(value.as_str()))
                .collect::<Vec<_>>();
                column
                    .typed::<ByteArrayType>()
                    .write_batch(&values, None, None)
                    .expect("string column writes");
            }
            2 | 3 => {
                column
                    .typed::<Int32Type>()
                    .write_batch(&vec![512; row_count], None, None)
                    .expect("dimension column writes");
            }
            _ => unreachable!("fixture schema has four columns"),
        }
        column.close().expect("column closes");
        column_index += 1;
    }
    row_group.close().expect("row group closes");
    writer.close().expect("Parquet writer closes");
}

#[tokio::test]
async fn catalog_routes_persist_status_and_return_bounded_filtered_pages_and_facets() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let config_dir = settings.config_dir.clone();
    let catalog_root = temporary.path().join("external-catalog");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 0);
    let app = create_app(settings).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Large catalog",
            "path": catalog_root.clone(),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source.clone()],
                "options": { "captionColumn": "TEXT" }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["contractVersion"], 1);
    assert_eq!(created["availability"], "available");
    assert_eq!(created["sourceConfig"]["kind"], "parquet");
    assert_eq!(created["counts"]["recordCount"], 0);
    assert!(created["storage"]["totalBytes"].as_u64().unwrap() > 0);
    let catalog_id = created["id"].as_str().unwrap().to_owned();
    let mut initial_scan_completed = false;
    for _ in 0..100 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{catalog_id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "completed" {
            initial_scan_completed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(initial_scan_completed, "empty fixture scan completes");

    let registry = CatalogRegistry::new(&config_dir);
    let mut catalog = registry.open_attached(&catalog_id).unwrap();
    catalog
        .append_records(&[
            catalog_record("one", "photo", 1),
            catalog_record("two", "photo", 1),
            catalog_record("three", "photo", 2),
            catalog_record("four", "illustration", 1),
        ])
        .unwrap();
    let mut contract = catalog.contract_state().unwrap();
    contract.analyzer_versions =
        BTreeMap::from([("person_detector".to_owned(), "model@sha256:abc".to_owned())]);
    contract.checkpoints = BTreeMap::from([("scan".to_owned(), json!({"shard": 4, "row": 80}))]);
    contract.processing = CatalogProcessingProgress {
        state: CatalogProcessingState::Paused,
        candidate_count: 12,
        processed_count: 4,
        accepted_count: 3,
        rejected_count: 1,
        error_count: 1,
        message: Some("paused by user".to_owned()),
        updated_at: "2026-07-26T00:00:00Z".to_owned(),
    };
    catalog.set_contract_state(&contract).unwrap();
    catalog.close();

    let (status, detail) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{catalog_id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail["schemaVersion"], 1);
    assert_eq!(
        detail["analyzerVersions"]["person_detector"],
        "model@sha256:abc"
    );
    assert_eq!(detail["checkpoints"]["scan"]["row"], 80);
    assert_eq!(detail["counts"]["recordCount"], 4);
    assert_eq!(detail["counts"]["rejectedCount"], 1);
    assert_eq!(detail["counts"]["errorCount"], 1);
    assert_eq!(detail["processing"]["state"], "paused");

    let query_uri = format!("/api/v1/catalogs/{catalog_id}/query");
    let (status, first_page) = request(
        app.clone(),
        "POST",
        &query_uri,
        json!({
            "limit": 2,
            "filters": [{ "field": "medium", "values": ["photo"] }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first_page}");
    assert_eq!(first_page["items"].as_array().unwrap().len(), 2);
    let cursor = first_page["nextCursor"]
        .as_str()
        .expect("bounded next cursor");
    let (status, second_page) = request(
        app.clone(),
        "POST",
        &query_uri,
        json!({
            "cursor": cursor,
            "limit": 2,
            "filters": [{ "field": "medium", "values": ["photo"] }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(second_page["items"].as_array().unwrap().len(), 1);
    assert!(second_page["nextCursor"].is_null());

    let (status, facets) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/facets"),
        json!({ "fields": ["medium", "analysis.fullBody"], "limitPerFacet": 1 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{facets}");
    assert_eq!(facets["facets"].as_array().unwrap().len(), 2);
    assert_eq!(
        facets["facets"][0]["values"].as_array().unwrap().len(),
        1,
        "facet response is bounded"
    );
    assert_eq!(facets["facets"][0]["values"][0]["value"], "photo");
    assert_eq!(facets["facets"][0]["values"][0]["count"], 3);

    for invalid_body in [
        json!({"limit": 0}),
        json!({"limit": 201}),
        json!({"cursor": "not-a-cursor"}),
        json!({"limit": 1, "path": catalog_root}),
        json!({"filters": [{"field": "medium); drop table catalog_records; --", "values": ["photo"]}]}),
    ] {
        let (status, _) = request(app.clone(), "POST", &query_uri, invalid_body).await;
        assert!(
            matches!(
                status,
                StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
            ),
            "invalid query contract must be rejected, got {status}"
        );
    }
}

#[tokio::test]
async fn catalog_id_routes_are_registry_scoped_and_detach_differs_from_delete_on_disk() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let catalog_root = temporary.path().join("selected-catalog");
    let app = create_app(settings).expect("app creates");

    let (status, relative_error) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({"name": "Relative", "path": "relative/catalog"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(!relative_error
        .to_string()
        .contains(temporary.path().to_string_lossy().as_ref()));

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({"name": "Lifecycle", "path": catalog_root.clone()}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let catalog_id = created["id"].as_str().unwrap().to_owned();

    let (status, missing) = request(
        app.clone(),
        "GET",
        "/api/v1/catalogs/00000000000000000000000000000000",
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing["code"], "catalog_not_found");
    assert!(
        !missing
            .to_string()
            .contains(temporary.path().to_string_lossy().as_ref()),
        "safe errors do not disclose filesystem roots"
    );

    let (status, detached) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/catalogs/{catalog_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detached["detached"], true);
    assert_eq!(detached["deletedOnDisk"], false);
    assert!(catalog_root.is_dir(), "detach preserves catalog files");

    let (status, attached) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs/attach",
        json!({"path": catalog_root.clone()}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{attached}");
    assert_eq!(attached["id"], catalog_id);

    let (status, deleted) = request(
        app,
        "DELETE",
        &format!("/api/v1/catalogs/{catalog_id}/on-disk"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted["deletedOnDisk"], true);
    assert!(
        !catalog_root.exists(),
        "only explicit on-disk delete removes files"
    );
}

#[tokio::test]
async fn corrupt_attached_catalog_errors_are_typed_and_do_not_leak_paths() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let catalog_root = temporary.path().join("private").join("catalog");
    let app = create_app(settings).expect("app creates");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({"name": "Corrupt", "path": catalog_root.clone()}),
    )
    .await;
    let catalog_id = created["id"].as_str().unwrap();
    std::fs::write(catalog_root.join("catalog.json"), b"not json").unwrap();

    let (status, error) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{catalog_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error["code"], "catalog_corrupt");
    assert!(
        !error
            .to_string()
            .contains(temporary.path().to_string_lossy().as_ref()),
        "client error must not expose the attached filesystem path"
    );
}

#[tokio::test]
async fn catalog_pause_and_resume_persist_desired_processing_state() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let config_dir = settings.config_dir.clone();
    let app = create_app(settings).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Controllable",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let id = created["id"].as_str().unwrap();

    let registry = CatalogRegistry::new(config_dir);
    let catalog = registry.open_attached(id).unwrap();
    catalog
        .set_processing_progress(&CatalogProcessingProgress {
            state: CatalogProcessingState::Running,
            updated_at: catalog_timestamp_now(),
            ..CatalogProcessingProgress::default()
        })
        .unwrap();
    let lease = CatalogProcessingLease::try_acquire(&catalog).unwrap();
    let (status, paused) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/pause"),
        json!({ "expectedRevision": 0 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{paused}");
    assert_eq!(paused["processing"]["state"], "running");
    assert_eq!(paused["processingControl"]["desiredState"], "paused");
    assert_eq!(paused["processingControl"]["revision"], 1);
    drop(lease);
    let mut actual = catalog.contract_state().unwrap().processing;
    actual.state = CatalogProcessingState::Paused;
    actual.updated_at = catalog_timestamp_now();
    catalog.set_processing_progress(&actual).unwrap();

    let (status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": 1 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    assert_eq!(resumed["processing"]["state"], "paused");
    assert_eq!(resumed["processingControl"]["desiredState"], "running");
    assert_eq!(resumed["processingControl"]["revision"], 2);

    let (invalid_status, invalid) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": 2 }),
    )
    .await;
    assert_eq!(invalid_status, StatusCode::CONFLICT, "{invalid}");
    assert_eq!(invalid["code"], "catalog_processing_conflict");

    let (_, persisted) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(persisted["processing"]["state"], "paused");
    assert_eq!(persisted["processingControl"]["desiredState"], "running");
}

#[tokio::test]
async fn stale_running_status_is_reconciled_and_lifecycle_recovers() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let config_dir = settings.config_dir.clone();
    let app = create_app(settings).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Interrupted",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let id = created["id"].as_str().unwrap();

    let catalog = CatalogRegistry::new(config_dir).open_attached(id).unwrap();
    catalog
        .set_processing_progress(&CatalogProcessingProgress {
            state: CatalogProcessingState::Running,
            processed_count: 17,
            updated_at: catalog_timestamp_now(),
            ..CatalogProcessingProgress::default()
        })
        .unwrap();
    catalog.close();

    let (status, reconciled) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reconciled["processing"]["state"], "failed");
    assert_eq!(reconciled["processing"]["processedCount"], 17);
    assert!(reconciled["processing"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("interrupted")));

    let (status, detached) = request(
        app,
        "DELETE",
        &format!("/api/v1/catalogs/{id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{detached}");
    assert_eq!(detached["detached"], true);
}

#[tokio::test]
async fn parquet_create_pause_resume_and_completion_run_through_public_api_scheduler() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 50_000);
    let app = create_app(settings).expect("app creates");

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Scheduled",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {
                    "batchSize": 25
                }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(
        created["processing"]["state"], "idle",
        "create response must not claim the background task acquired its lease"
    );
    let id = created["id"].as_str().unwrap();

    let mut paused_response = None;
    for _ in 0..500 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "running" {
            let revision = status_body["processingControl"]["revision"]
                .as_u64()
                .unwrap();
            let (pause_status, pause_body) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/catalogs/{id}/pause"),
                json!({ "expectedRevision": revision }),
            )
            .await;
            if pause_status == StatusCode::OK {
                paused_response = Some(pause_body);
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let pause_requested = paused_response.expect("active scan accepts pause through public API");
    assert_eq!(
        pause_requested["processingControl"]["desiredState"],
        "paused"
    );

    let mut paused = None;
    for _ in 0..500 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "paused" {
            paused = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let paused = paused.expect("scanner cooperatively pauses at a bounded batch boundary");
    assert!(paused["processing"]["processedCount"].as_u64().unwrap() > 0);

    let revision = paused["processingControl"]["revision"].as_u64().unwrap();
    let (resume_status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": revision }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    assert_eq!(resumed["processing"]["state"], "paused");
    assert_eq!(resumed["processingControl"]["desiredState"], "running");

    let mut completed = None;
    for _ in 0..2_000 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "completed" {
            completed = Some(status_body);
            break;
        }
        assert_ne!(
            status_body["processing"]["state"], "failed",
            "scheduler failure: {status_body}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let completed = completed.expect("resumed public API scan reaches completion");
    assert_eq!(completed["counts"]["processedCount"], 50_000);
    assert_eq!(completed["counts"]["recordCount"], 50_000);
}

#[tokio::test]
async fn resume_during_paused_task_teardown_is_handed_to_a_successor() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100_000);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Paused teardown handoff",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": { "batchSize": 25 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id");

    let mut pause_response = None;
    for _ in 0..1_000 {
        let (_, current) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "running" {
            state
                .catalog_scan_before_terminal_exit_once
                .store(true, Ordering::SeqCst);
            let terminal = state.catalog_scan_terminal_exit_reached.notified();
            let (pause_status, paused) = request(
                app.clone(),
                "POST",
                &format!("/api/v1/catalogs/{id}/pause"),
                json!({
                    "expectedRevision": current["processingControl"]["revision"]
                }),
            )
            .await;
            if pause_status == StatusCode::OK {
                tokio::time::timeout(Duration::from_secs(5), terminal)
                    .await
                    .expect("scanner publishes paused and releases its lease");
                pause_response = Some(paused);
                break;
            }
            state
                .catalog_scan_before_terminal_exit_once
                .store(false, Ordering::SeqCst);
            state.catalog_scan_terminal_exit_release.notify_waiters();
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let paused = pause_response.expect("pause wins before completion");
    let (resume_status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({
            "expectedRevision": paused["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    assert_eq!(resumed["processingControl"]["desiredState"], "running");
    assert_eq!(
        state.catalog_scan_supervisor.active_count().await,
        1,
        "the terminal generation remains registered while restart is queued"
    );
    state.catalog_scan_terminal_exit_release.notify_waiters();

    let mut completed = None;
    for _ in 0..2_000 {
        let (_, current) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("queued successor completes")["counts"]["recordCount"],
        100_000
    );
}

#[tokio::test]
async fn successor_queued_after_closure_is_recovered_by_new_appstate() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100_000);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Shutdown queued successor",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": { "batchSize": 25 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id").to_owned();

    let mut running = None;
    for _ in 0..1_000 {
        let (_, current) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "running" {
            running = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let running = running.expect("scan starts before completion");
    let (cleanup_reached, cleanup_release) = state
        .catalog_scan_supervisor
        .install_wrapper_cleanup_barriers();
    let (pause_status, paused) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/pause"),
        json!({
            "expectedRevision": running["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(pause_status, StatusCode::OK, "{paused}");
    tokio::time::timeout(Duration::from_secs(5), cleanup_reached.wait())
        .await
        .expect("scan closure returns before wrapper cleanup");
    let (resume_status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({
            "expectedRevision": paused["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    assert_eq!(
        state.catalog_scan_supervisor.active_count().await,
        1,
        "successor is queued behind the terminal generation"
    );

    let (shutdown, _) = tokio::join!(
        state.catalog_scan_supervisor.shutdown(),
        cleanup_release.wait()
    );
    assert_eq!(shutdown.requested, 1);
    assert_eq!(shutdown.joined, 1);
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let stranded = registry.open_attached(&id).expect("catalog opens");
    assert_eq!(
        stranded
            .contract_state()
            .expect("contract reads")
            .processing
            .state,
        CatalogProcessingState::Paused
    );
    assert_eq!(
        stranded
            .processing_control()
            .expect("control reads")
            .desired_state,
        sceneworks_core::catalog_store::CatalogProcessingDesiredState::Running
    );
    stranded.close();

    let (restarted_app, restarted_state) =
        create_app_with_state(settings).expect("restarted app creates");
    let (_, recovering) = request(
        restarted_app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_ne!(recovering["processing"]["state"], "failed");
    assert_eq!(
        restarted_state.catalog_scan_supervisor.active_count().await,
        1,
        "new AppState status discovery schedules the persisted successor"
    );
    let mut completed = None;
    for _ in 0..2_000 {
        let (_, current) = request(
            restarted_app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("public restart completes")["counts"]["recordCount"],
        100_000
    );
}

#[tokio::test]
async fn interrupted_bounded_driver_is_automatically_recovered_from_its_checkpoint() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 30_000);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    state
        .catalog_scan_stop_after_pass_once
        .store(true, Ordering::SeqCst);

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Interrupted bounded scan",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": { "batchSize": 100 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().unwrap();

    let mut saw_durable_checkpoint = false;
    let mut completed = None;
    for _ in 0..2_000 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        saw_durable_checkpoint |= status_body["processing"]["processedCount"] == 25_000;
        if status_body["processing"]["state"] == "completed" {
            completed = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let completed = completed.expect("status recovery resumes the durable checkpoint");
    assert!(
        saw_durable_checkpoint,
        "the bounded pass checkpoint must be visible before completion"
    );
    assert_eq!(completed["counts"]["recordCount"], 30_000);
}

#[tokio::test]
async fn status_retries_after_a_recovered_generation_exits_incomplete() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 80_000);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    state
        .catalog_scan_stop_after_pass_once
        .store(true, Ordering::SeqCst);

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Repeated durable recovery",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": { "batchSize": 100 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id").to_owned();

    for _ in 0..1_000 {
        if state.catalog_scan_supervisor.tracked_count().await == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(state.catalog_scan_supervisor.tracked_count().await, 0);

    let before_recovered_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(Arc::clone(&before_recovered_start));
    let (_, first_recovery) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(first_recovery["processing"]["processedCount"], 25_000);
    assert_eq!(state.catalog_scan_supervisor.active_count().await, 1);
    state
        .catalog_scan_stop_after_pass_once
        .store(true, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(5), before_recovered_start.wait())
        .await
        .expect("recovered generation reaches its prestart barrier");

    for _ in 0..1_000 {
        if state.catalog_scan_supervisor.tracked_count().await == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(state.catalog_scan_supervisor.tracked_count().await, 0);
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let twice_interrupted = registry.open_attached(&id).expect("catalog opens");
    assert_eq!(
        twice_interrupted
            .contract_state()
            .expect("contract reads")
            .processing
            .processed_count,
        50_000
    );
    twice_interrupted.close();

    let (_, second_recovery) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(second_recovery["processing"]["processedCount"], 50_000);
    let mut completed = None;
    for _ in 0..2_000 {
        let (_, current) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(
        completed.expect("second recovery completes")["counts"]["recordCount"],
        80_000
    );
}

#[tokio::test]
async fn transient_prestart_read_lease_cannot_strand_scheduled_processing() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 1_000);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    let before_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(Arc::clone(&before_start));

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Lease race",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().unwrap();
    let catalog = CatalogRegistry::new(temporary.path().join("config"))
        .open_attached(id)
        .unwrap();
    let transient_read_lease = CatalogProcessingLease::try_acquire(&catalog).unwrap();
    before_start.wait().await;

    let (status, during_race) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(during_race["processing"]["state"], "idle");
    tokio::time::sleep(Duration::from_millis(30)).await;
    drop(transient_read_lease);
    catalog.close();

    let mut completed = None;
    for _ in 0..1_000 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "completed" {
            completed = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert_eq!(
        completed.expect("scheduler retries after transient read lease")["counts"]["recordCount"],
        1_000
    );
}

#[tokio::test]
async fn unsupported_automatic_catalog_source_is_actionable() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let config_dir = settings.config_dir.clone();
    let source = temporary.path().join("images");
    std::fs::create_dir(&source).unwrap();
    let app = create_app(settings).expect("app creates");

    let (status, error) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Unsupported",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "filesystem",
                "paths": [source],
                "options": {}
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("supports Parquet")));

    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Attached unsupported",
            "path": temporary.path().join("attached-catalog")
        }),
    )
    .await;
    let id = created["id"].as_str().unwrap();
    let catalog = CatalogRegistry::new(config_dir).open_attached(id).unwrap();
    let mut contract = catalog.contract_state().unwrap();
    contract.source_config = Some(CatalogSourceConfig {
        kind: "filesystem".to_owned(),
        paths: vec![source],
        options: json!({}),
    });
    contract.processing.state = CatalogProcessingState::Failed;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog.set_contract_state(&contract).unwrap();
    catalog.close();

    let (status, error) = request(
        app,
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": 0 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("supports Parquet")));
    let control = CatalogRegistry::new(temporary.path().join("config"))
        .open_attached(id)
        .unwrap()
        .processing_control()
        .unwrap();
    assert_eq!(
        control.revision, 0,
        "invalid resume does not advance intent"
    );
}

#[tokio::test]
async fn invalid_parquet_options_are_rejected_before_create_or_control_cas() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let config_dir = settings.config_dir.clone();
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 10);
    let app = create_app(settings).expect("app creates");

    for (index, options) in [
        json!({"batchSize": 0}),
        json!({"minWidth": 100, "maxWidth": 99}),
        json!({"urlColumn": "missing_column"}),
        json!({"captionIncludes": [""]}),
        json!({"unknownOption": true}),
    ]
    .into_iter()
    .enumerate()
    {
        let catalog_root = temporary.path().join(format!("invalid-{index}"));
        let (status, error) = request(
            app.clone(),
            "POST",
            "/api/v1/catalogs",
            json!({
                "name": format!("Invalid {index}"),
                "path": catalog_root,
                "sourceConfig": {
                    "kind": "parquet",
                    "paths": [source.clone()],
                    "options": options
                }
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        assert!(
            !catalog_root.exists(),
            "invalid plan must not persist catalog {index}"
        );
    }

    let catalog_root = temporary.path().join("zero-budget");
    let (status, error) = tokio::time::timeout(
        Duration::from_secs(2),
        request(
            app.clone(),
            "POST",
            "/api/v1/catalogs",
            json!({
                "name": "Zero budget",
                "path": catalog_root,
                "sourceConfig": {
                    "kind": "parquet",
                    "paths": [source.clone()],
                    "options": { "maxRows": 0 }
                }
            }),
        ),
    )
    .await
    .expect("zero maxRows is rejected synchronously rather than scheduling a loop");
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("scheduler-managed")));
    assert!(!catalog_root.exists());

    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Invalid resume",
            "path": temporary.path().join("resume-catalog")
        }),
    )
    .await;
    let id = created["id"].as_str().unwrap();
    let catalog = CatalogRegistry::new(config_dir).open_attached(id).unwrap();
    let mut contract = catalog.contract_state().unwrap();
    contract.source_config = Some(CatalogSourceConfig {
        kind: "parquet".to_owned(),
        paths: vec![source],
        options: json!({"batchSize": 0}),
    });
    contract.processing.state = CatalogProcessingState::Failed;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog.set_contract_state(&contract).unwrap();
    catalog.close();

    let (status, error) = request(
        app,
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": 0 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    let catalog = CatalogRegistry::new(temporary.path().join("config"))
        .open_attached(id)
        .unwrap();
    assert_eq!(catalog.processing_control().unwrap().revision, 0);
    assert_eq!(
        catalog.contract_state().unwrap().processing.state,
        CatalogProcessingState::Failed
    );
}

#[tokio::test(flavor = "current_thread")]
async fn slow_parquet_preflight_keeps_health_responsive() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 0);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    state
        .catalog_scan_preflight_delay_ms_once
        .store(250, Ordering::SeqCst);
    let preflight_started = state.catalog_scan_preflight_started.notified();
    let create = tokio::spawn(request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Slow preflight",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    ));
    tokio::time::timeout(Duration::from_secs(2), preflight_started)
        .await
        .expect("blocking-pool preflight starts");

    let health = tokio::time::timeout(
        Duration::from_millis(75),
        request(app, "GET", "/api/v1/health", Value::Null),
    )
    .await
    .expect("health remains responsive while Parquet metadata preflight is blocked");
    assert_eq!(health.0, StatusCode::OK);
    let created = create.await.expect("create task joins");
    assert_eq!(created.0, StatusCode::CREATED, "{}", created.1);
    state.catalog_scan_supervisor.shutdown().await;
}

#[tokio::test]
async fn excessive_parquet_shards_are_rejected_before_catalog_persistence() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("shards");
    std::fs::create_dir(&source).expect("shard directory creates");
    for index in 0..=MAX_PARQUET_SHARDS {
        File::create(source.join(format!("{index:05}.parquet"))).expect("empty shard creates");
    }
    let catalog_root = temporary.path().join("catalog");
    let app = create_app(test_settings(&temporary)).expect("app creates");

    let (status, error) = request(
        app,
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Too many shards",
            "path": catalog_root,
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains(&MAX_PARQUET_SHARDS.to_string())));
    assert!(
        !catalog_root.exists(),
        "the shard limit must be checked before catalog persistence"
    );
}

#[tokio::test]
async fn graceful_catalog_shutdown_drains_and_public_restart_resumes_checkpoint() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 50_000);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Shutdown restart",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": { "batchSize": 25 }
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id").to_owned();

    let mut running = None;
    for _ in 0..500 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "running"
            && status_body["processing"]["processedCount"]
                .as_u64()
                .unwrap_or(0)
                > 0
        {
            running = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    running.expect("scan reaches a durable running checkpoint");

    let shutdown = state.catalog_scan_supervisor.shutdown().await;
    assert_eq!(shutdown.requested, 1);
    assert_eq!(shutdown.joined, 1, "{shutdown:?}");
    assert_eq!(state.catalog_scan_supervisor.active_count().await, 0);

    let (_, interrupted) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(interrupted["processing"]["state"], "failed");
    assert!(interrupted["processing"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("shutdown")));
    let checkpoint_count = interrupted["processing"]["processedCount"]
        .as_u64()
        .expect("processed count");
    assert!(checkpoint_count > 0 && checkpoint_count < 50_000);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (_, stable) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(
        stable["processing"]["processedCount"].as_u64(),
        Some(checkpoint_count),
        "no catalog task may continue after supervisor shutdown returns"
    );

    let restarted_app = create_app(settings).expect("restarted app creates");
    let revision = stable["processingControl"]["revision"]
        .as_u64()
        .expect("revision");
    let (status, resumed) = request(
        restarted_app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": revision }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    let mut completed = None;
    for _ in 0..2_000 {
        let (_, status_body) = request(
            restarted_app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "completed" {
            completed = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("restart completes from checkpoint")["counts"]["recordCount"],
        50_000
    );
}

#[tokio::test]
async fn resume_waits_through_long_prestart_lease_contention_and_completes() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 1_000);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Long resume contention",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    let id = created["id"].as_str().expect("catalog id").to_owned();
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let catalog = registry.open_attached(&id).expect("catalog opens");
    let mut contract = catalog.contract_state().expect("contract reads");
    contract.source_config = Some(CatalogSourceConfig {
        kind: "parquet".to_owned(),
        paths: vec![source],
        options: json!({}),
    });
    contract.processing.state = CatalogProcessingState::Running;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog
        .set_contract_state(&contract)
        .expect("contract updates");
    let paused_control = catalog
        .request_processing_control(
            0,
            sceneworks_core::catalog_store::CatalogProcessingDesiredState::Paused,
        )
        .expect("pause intent persists");
    contract.processing.state = CatalogProcessingState::Paused;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog
        .set_contract_state(&contract)
        .expect("paused progress persists");
    catalog.close();

    let before_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(before_start.clone());
    let (status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": paused_control.revision }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resumed}");

    let contended_catalog = registry.open_attached(&id).expect("catalog reopens");
    let held_lease =
        CatalogProcessingLease::try_acquire(&contended_catalog).expect("test lease acquires");
    before_start.wait().await;
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert_eq!(
        state.catalog_scan_supervisor.active_count().await,
        1,
        "desired-running work must remain supervised beyond the old 510ms retry window"
    );
    drop(held_lease);
    contended_catalog.close();

    let mut completed = None;
    for _ in 0..1_000 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "completed" {
            completed = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("scan starts after long lease contention")["counts"]["recordCount"],
        1_000
    );
}

#[tokio::test]
async fn resume_after_busy_retry_terminal_decision_is_handed_to_a_successor() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Busy retry handoff",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    let id = created["id"].as_str().expect("catalog id").to_owned();
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let catalog = registry.open_attached(&id).expect("catalog opens");
    let mut contract = catalog.contract_state().expect("contract reads");
    contract.source_config = Some(CatalogSourceConfig {
        kind: "parquet".to_owned(),
        paths: vec![source],
        options: json!({}),
    });
    contract.processing.state = CatalogProcessingState::Running;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog
        .set_contract_state(&contract)
        .expect("contract updates");
    let paused = catalog
        .request_processing_control(
            0,
            sceneworks_core::catalog_store::CatalogProcessingDesiredState::Paused,
        )
        .expect("initial pause intent");
    contract.processing.state = CatalogProcessingState::Paused;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog
        .set_contract_state(&contract)
        .expect("paused progress persists");
    catalog.close();

    let before_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(before_start.clone());
    let (status, running) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": paused.revision }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{running}");
    let contended = registry.open_attached(&id).expect("catalog reopens");
    let held_lease = CatalogProcessingLease::try_acquire(&contended).expect("test lease");
    contended
        .set_processing_progress(&CatalogProcessingProgress {
            state: CatalogProcessingState::Running,
            message: Some("contended processor is stopping".to_owned()),
            updated_at: catalog_timestamp_now(),
            ..CatalogProcessingProgress::default()
        })
        .expect("active progress publishes");
    state
        .catalog_scan_before_terminal_exit_once
        .store(true, Ordering::SeqCst);
    let terminal = state.catalog_scan_terminal_exit_reached.notified();
    before_start.wait().await;

    let (pause_status, paused_again) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/pause"),
        json!({
            "expectedRevision": running["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(pause_status, StatusCode::OK, "{paused_again}");
    tokio::time::timeout(Duration::from_secs(2), terminal)
        .await
        .expect("Busy retry observes terminal pause decision");
    drop(held_lease);
    contended.close();

    let (resume_status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({
            "expectedRevision": paused_again["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    state.catalog_scan_terminal_exit_release.notify_waiters();

    let mut completed = None;
    for _ in 0..1_000 {
        let (_, current) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("queued Busy successor completes")["counts"]["recordCount"],
        100
    );
}

#[tokio::test]
async fn new_appstate_recovers_after_shutdown_lease_contention_clears() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
    let (_, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Canceled resume contention",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    let id = created["id"].as_str().expect("catalog id").to_owned();
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let catalog = registry.open_attached(&id).expect("catalog opens");
    let mut contract = catalog.contract_state().expect("contract reads");
    contract.source_config = Some(CatalogSourceConfig {
        kind: "parquet".to_owned(),
        paths: vec![source],
        options: json!({}),
    });
    contract.processing.state = CatalogProcessingState::Running;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog
        .set_contract_state(&contract)
        .expect("contract updates");
    let paused_control = catalog
        .request_processing_control(
            0,
            sceneworks_core::catalog_store::CatalogProcessingDesiredState::Paused,
        )
        .expect("pause intent persists");
    contract.processing.state = CatalogProcessingState::Paused;
    contract.processing.updated_at = catalog_timestamp_now();
    catalog
        .set_contract_state(&contract)
        .expect("paused progress persists");
    catalog.close();

    let before_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(before_start.clone());
    let (status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": paused_control.revision }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    let contended_catalog = registry.open_attached(&id).expect("catalog reopens");
    let held_lease =
        CatalogProcessingLease::try_acquire(&contended_catalog).expect("test lease acquires");
    before_start.wait().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let shutdown = tokio::time::timeout(
        Duration::from_secs(1),
        state.catalog_scan_supervisor.shutdown(),
    )
    .await
    .expect("cancellation interrupts the retry backoff");
    assert_eq!(shutdown.requested, 1);
    assert_eq!(shutdown.joined, 1);
    let (restarted_app, restarted_state) =
        create_app_with_state(settings).expect("restarted app creates");
    let (_, externally_owned) = request(
        restarted_app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(externally_owned["counts"]["recordCount"], 0);
    assert_eq!(externally_owned["processing"]["state"], "paused");
    assert_eq!(
        restarted_state.catalog_scan_supervisor.active_count().await,
        0,
        "a live external lease must not start a duplicate processor"
    );

    drop(held_lease);
    contended_catalog.close();
    let (_, recovery_started) = request(
        restarted_app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_ne!(recovery_started["processing"]["state"], "failed");
    let mut completed = None;
    for _ in 0..1_000 {
        let (_, current) = request(
            restarted_app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("status recovery completes")["counts"]["recordCount"],
        100
    );
}

#[tokio::test]
async fn invalid_or_missing_persisted_recovery_plan_fails_once_without_launch_loop() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let invalid_source = temporary.path().join("source.csv");
    std::fs::File::create(&invalid_source).expect("invalid-kind source fixture");

    for (name, source_config) in [
        (
            "Invalid recovery plan",
            Some(CatalogSourceConfig {
                kind: "csv".to_owned(),
                paths: vec![invalid_source],
                options: json!({}),
            }),
        ),
        ("Missing recovery plan", None),
    ] {
        let (_, created) = request(
            app.clone(),
            "POST",
            "/api/v1/catalogs",
            json!({
                "name": name,
                "path": temporary.path().join(name.replace(' ', "-"))
            }),
        )
        .await;
        let id = created["id"].as_str().expect("catalog id").to_owned();
        let catalog = registry.open_attached(&id).expect("catalog opens");
        let mut contract = catalog.contract_state().expect("contract reads");
        contract.source_config = source_config;
        if contract.source_config.is_none() {
            contract.checkpoints.insert(
                "scanner.parquet.checkpoint.v1".to_owned(),
                json!({ "fixture": "interrupted scan with missing plan" }),
            );
        }
        contract.processing.state = CatalogProcessingState::Paused;
        contract.processing.updated_at = catalog_timestamp_now();
        catalog
            .set_contract_state(&contract)
            .expect("recovery fixture persists");
        catalog.close();

        let (_, failed) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        assert_eq!(failed["processing"]["state"], "failed", "{failed}");
        let message = failed["processing"]["message"]
            .as_str()
            .expect("actionable failure")
            .to_owned();
        assert!(message.contains("repair") && message.contains("resume"));
        let (_, repeated) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        assert_eq!(repeated["processing"]["message"], message);
        assert_eq!(
            state.catalog_scan_supervisor.active_count().await,
            0,
            "invalid persisted plan must not launch repeatedly"
        );
    }
}

#[tokio::test]
async fn cancellation_while_waiting_for_busy_lease_is_restartable() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
    let before_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(before_start.clone());
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Busy cancellation",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id").to_owned();
    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let contended = registry.open_attached(&id).expect("catalog opens");
    let held_lease = CatalogProcessingLease::try_acquire(&contended).expect("test lease");
    before_start.wait().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let (shutdown, ()) = tokio::join!(state.catalog_scan_supervisor.shutdown(), async move {
        drop(held_lease);
        contended.close();
    });
    assert_eq!(shutdown.requested, 1);
    assert_eq!(shutdown.joined, 1);
    let (_, interrupted) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(interrupted["processing"]["state"], "failed");
    assert!(interrupted["processing"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("restart")));

    let restarted_app = create_app(settings).expect("restarted app creates");
    let (resume_status, resumed) = request(
        restarted_app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({
            "expectedRevision": interrupted["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    let mut completed = None;
    for _ in 0..1_000 {
        let (_, current) = request(
            restarted_app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("public restart completes")["counts"]["recordCount"],
        100
    );
}

#[tokio::test]
async fn cancellation_while_waiting_for_global_scan_permit_is_restartable() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100);
    let (app, state) = create_app_with_state(settings.clone()).expect("app and state create");
    let before_start = Arc::new(tokio::sync::Barrier::new(2));
    *state.catalog_scan_before_driver_start_once.lock() = Some(before_start.clone());
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Permit cancellation",
            "path": temporary.path().join("catalog"),
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id").to_owned();
    let held_permits = state
        .catalog_scan_work_slots
        .clone()
        .acquire_many_owned(2)
        .await
        .expect("global permits acquire");
    before_start.wait().await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let shutdown = state.catalog_scan_supervisor.shutdown().await;
    assert_eq!(shutdown.requested, 1);
    assert_eq!(shutdown.joined, 1);
    drop(held_permits);
    let (_, interrupted) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(interrupted["processing"]["state"], "failed");
    assert!(interrupted["processing"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("restart")));

    let restarted_app = create_app(settings).expect("restarted app creates");
    let (resume_status, resumed) = request(
        restarted_app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({
            "expectedRevision": interrupted["processingControl"]["revision"]
        }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    let mut completed = None;
    for _ in 0..1_000 {
        let (_, current) = request(
            restarted_app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if current["processing"]["state"] == "completed" {
            completed = Some(current);
            break;
        }
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    assert_eq!(
        completed.expect("public restart completes")["counts"]["recordCount"],
        100
    );
}

#[tokio::test(flavor = "current_thread")]
async fn saturated_preflight_admission_rejects_extra_work_without_blocking_catalog_status() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 0);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    let (_, existing) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Existing",
            "path": temporary.path().join("existing")
        }),
    )
    .await;
    let existing_id = existing["id"].as_str().expect("existing id");
    let first_permit = state
        .catalog_scan_preflight_slots
        .clone()
        .acquire_owned()
        .await
        .expect("first slot");
    let second_permit = state
        .catalog_scan_preflight_slots
        .clone()
        .acquire_owned()
        .await
        .expect("second slot");
    state
        .catalog_scan_preflight_admission_timeout_ms
        .store(50, Ordering::SeqCst);
    let rejected_root = temporary.path().join("rejected");
    let extra = tokio::spawn(request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Excess preflight",
            "path": rejected_root,
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    ));

    let status = tokio::time::timeout(
        Duration::from_secs(2),
        request(
            app,
            "GET",
            &format!("/api/v1/catalogs/{existing_id}/status"),
            Value::Null,
        ),
    )
    .await
    .expect("catalog DB/status remains responsive while admission is saturated");
    assert_eq!(status.0, StatusCode::OK);
    let rejected = extra.await.expect("extra request joins");
    assert_eq!(
        rejected.0,
        StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        rejected.1
    );
    assert!(rejected.1["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("busy")));
    assert!(!rejected_root.exists());
    drop(first_permit);
    drop(second_permit);
}

#[tokio::test(flavor = "current_thread")]
async fn timed_out_preflight_cancels_remaining_validation_work() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 0);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    state
        .catalog_scan_preflight_delay_ms_once
        .store(500, Ordering::SeqCst);
    state
        .catalog_scan_preflight_execution_timeout_ms
        .store(50, Ordering::SeqCst);
    let started = state.catalog_scan_preflight_started.notified();
    let catalog_root = temporary.path().join("timed-out");
    let request_task = tokio::spawn(request(
        app,
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Timed out preflight",
            "path": catalog_root,
            "sourceConfig": {
                "kind": "parquet",
                "paths": [source],
                "options": {}
            }
        }),
    ));
    tokio::time::timeout(Duration::from_secs(1), started)
        .await
        .expect("preflight starts");
    let response = request_task.await.expect("request joins");
    assert_eq!(
        response.0,
        StatusCode::SERVICE_UNAVAILABLE,
        "{}",
        response.1
    );
    assert!(response.1["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("timed out")));
    assert!(!catalog_root.exists());

    tokio::time::sleep(Duration::from_millis(30)).await;
    let stopped_ticks = state
        .catalog_scan_preflight_test_ticks
        .load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(
        state
            .catalog_scan_preflight_test_ticks
            .load(Ordering::SeqCst),
        stopped_ticks,
        "timeout cancellation must stop further validation work"
    );
    assert_eq!(state.catalog_scan_preflight_slots.available_permits(), 2);
}
