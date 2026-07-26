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
async fn interrupted_bounded_driver_is_reconciled_and_public_restart_completes() {
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

    let mut failed = None;
    for _ in 0..2_000 {
        let (_, status_body) = request(
            app.clone(),
            "GET",
            &format!("/api/v1/catalogs/{id}/status"),
            Value::Null,
        )
        .await;
        if status_body["processing"]["state"] == "failed" {
            failed = Some(status_body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let failed = failed.expect("driver interruption between passes reconciles to failed");
    assert_eq!(failed["processing"]["processedCount"], 25_000);
    assert!(failed["processing"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("interrupted")));

    let revision = failed["processingControl"]["revision"].as_u64().unwrap();
    let (status, restarted) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": revision }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{restarted}");
    assert_eq!(restarted["processing"]["state"], "failed");
    assert_eq!(restarted["processingControl"]["desiredState"], "running");

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
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let completed = completed.expect("public restart resumes the durable checkpoint");
    assert_eq!(completed["counts"]["recordCount"], 30_000);
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
    state
        .catalog_scan_supervisor
        .shutdown(Duration::from_secs(2))
        .await;
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

    let shutdown = state
        .catalog_scan_supervisor
        .shutdown(Duration::from_secs(2))
        .await;
    assert_eq!(shutdown.requested, 1);
    assert!(!shutdown.timed_out, "{shutdown:?}");
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
