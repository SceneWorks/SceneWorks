//! Dataset Catalog lifecycle, query-contract, and path-confinement route tests.

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};

use parquet::data_type::{ByteArray, ByteArrayType, Int32Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use sceneworks_core::catalog_store::{
    catalog_timestamp_now, CatalogProcessingLease, CatalogProcessingProgress,
    CatalogProcessingState, CatalogRecordFilter, CatalogRegistry, CatalogSearchRequest,
    CatalogSourceConfig, NewCatalogRecord,
};
use sceneworks_worker::catalog_parquet_scanner::MAX_PARQUET_SHARDS;
use tower::ServiceExt as _;

use super::support::*;

fn catalog_scan_hook_test_lock() -> &'static TestSerializationLock {
    static LOCK: OnceLock<TestSerializationLock> = OnceLock::new();
    LOCK.get_or_init(|| {
        // The default 300 s hold bound was sized against the old flat 60 s
        // scan-idle cap. This lock's holders wait on work-derived budgets
        // (`catalog_scan_idle_budget`), so its hold bound must be derived from
        // the same work or the wedge detector fires on a legitimately slow
        // scan the budget still allows. Worst legitimate holder is
        // `status_retries_after_a_recovered_generation_exits_incomplete`:
        // three 80_000-row idle waits plus two 60 s bounded meeting points.
        let worst_holder = 3 * catalog_scan_idle_budget(80_000) + Duration::from_secs(120);
        TestSerializationLock::with_max_hold("catalog_scan_hook_test_lock", worst_holder)
    })
}

async fn lock_catalog_scan_hook() -> TestSerializationGuard<'static> {
    catalog_scan_hook_test_lock().acquire().await
}

/// Fixed cost of getting a scan scheduled, started and reaped, independent of
/// how many rows it has to get through.
const SCAN_IDLE_BASE: Duration = Duration::from_secs(30);

/// Throughput floor a catalog scan must beat to count as making progress.
///
/// Worst throughput measured while calibrating this (see
/// [`catalog_scan_idle_budget`]) was ~894 rows/s. 300 leaves a ~3x margin under
/// that, so the budget tracks the work rather than the host.
const SCAN_IDLE_ROWS_PER_SEC: u64 = 300;

/// Wall-clock budget for one catalog scan to reach idle, derived from the row
/// count that scan actually has to get through.
///
/// A single flat cap for every scan in this module was the defect: the same
/// 60 s covered a 10-row scan and a 100_000-row one. It is not remotely enough
/// for the big ones on a contended host. Measured on an 18-core Mac with this
/// whole module running in parallel under 48 external CPU hogs (1-minute load
/// average 72 rising to 139):
///
/// | rows      | worst observed wait | budget |
/// |-----------|---------------------|--------|
/// | 30_000    | 4.5 s               | 130 s  |
/// | 80_000    | 20.9 s              | 296 s  |
/// | 100_000   | 111.8 s             | 363 s  |
///
/// A lighter run of the same module (load average 73-106) put the worst
/// 100_000-row wait at 69.4 s — already past the flat cap, which is why those
/// two tests failed on a loaded dev host while CI, uncontended at ~19 s for that
/// same scan, stayed green. sc-17723 hit this class before and answered it by
/// raising the magic numbers; deriving from the work is the successor, because
/// it does not need re-tuning per host.
///
/// Note this is TIGHTER than the old flat cap for every scan below 9_000 rows,
/// which is most of the module — a wedged small scan now fails in 30 s rather
/// than 60 s. Only the genuinely heavy scans get a long budget, and even the
/// largest (100_000 rows, 363 s) reports well inside `parity-rust`'s 30-minute
/// cap on a job that runs in ~9 minutes.
fn catalog_scan_idle_budget(rows: u64) -> Duration {
    SCAN_IDLE_BASE + Duration::from_secs(rows / SCAN_IDLE_ROWS_PER_SEC)
}

async fn wait_for_catalog_scan_idle(
    supervisor: &crate::catalog_scan_supervisor::CatalogScanSupervisor,
    rows: u64,
) {
    let budget = catalog_scan_idle_budget(rows);
    let started = std::time::Instant::now();
    if tokio::time::timeout(budget, supervisor.wait_for_idle())
        .await
        .is_err()
    {
        panic!(
            "catalog scan supervisor never reached idle for a {rows}-row scan: waited {:?} \
             against a {budget:?} budget. That budget already allows {SCAN_IDLE_ROWS_PER_SEC} \
             rows/s, ~3x slower than the worst measured host, so this scan is wedged rather \
             than merely slow",
            started.elapsed()
        );
    }
}

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
            "analysis": {
                "fullBody": person_count == 1,
                "qualifiedSingleFullBody": person_count == 1
            }
        }),
    }
}

fn cached_materialization_record(
    root: &std::path::Path,
    id: &str,
    rgb: [u8; 3],
) -> NewCatalogRecord {
    let relative = format!("images/cache/{id}.png");
    let path = root.join(&relative);
    std::fs::create_dir_all(path.parent().expect("cache parent")).expect("cache dir creates");
    image::RgbImage::from_pixel(32, 32, image::Rgb(rgb))
        .save(&path)
        .expect("cache image writes");
    let content_hash =
        sceneworks_core::media_convert::file_content_hash(&path).expect("cache hashes");
    let source_caption = format!(
        "source caption {id} {} CAPTION_TAIL_SENTINEL",
        "🦀".repeat(2_000)
    );
    let generated_caption = format!(
        "generated caption {id} {} GENERATED_TAIL_SENTINEL",
        "界".repeat(3_000)
    );
    let mut generated_tags = vec![json!({"value": "outdoor"}), json!({"value": "full_body"})];
    generated_tags.extend((0..40).map(|index| {
        json!({
            "value": format!("tag-{index}-{}-TAG_TAIL_SENTINEL", "界".repeat(100)),
            "confidence": 0.75
        })
    }));
    NewCatalogRecord {
        id: id.to_owned(),
        image_path: format!("images/{id}.png"),
        thumbnail_path: None,
        embedding_path: None,
        artifact_path: None,
        metadata: json!({
            "url": format!("https://example.test/{id}.png"),
            "caption": source_caption,
            "medium": "photograph",
            "personCount": 1,
            "analysis": {
                "medium": "photograph",
                "personCount": 1,
                "qualifiedSingleFullBody": true,
                "unrelatedPayload": "not-provenance".repeat(32_000),
                "structured": {
                    "schemaVersion": 1,
                    "inputFingerprint": "f".repeat(1_024),
                    "executionOrder": ["person", "face", "pose", "derived"],
                    "person": {
                        "analyzer": {
                            "analyzerVersion": "person-v1",
                            "modelId": "person-model",
                            "modelVersion": "sha256:person",
                            "status": "succeeded",
                            "thresholds": {"unbounded": "x".repeat(32_000)}
                        }
                    }
                },
                "visionLanguage": {
                    "schemaVersion": 1,
                    "caption": generated_caption,
                    "tags": generated_tags,
                    "analyzerVersion": "vlm-v1",
                    "modelId": "caption-model",
                    "modelRevision": "sha256:caption",
                    "runtimeRevision": "runtime-v1",
                    "backend": "mlx",
                    "provider": "local",
                    "taxonomyVersion": "taxonomy-v1",
                    "status": "succeeded"
                },
                "semanticEmbedding": {
                    "schemaVersion": 1,
                    "analyzerVersion": "embedding-v1",
                    "modelId": "clip-model",
                    "modelRevision": "sha256:clip",
                    "runtimeRevision": "runtime-v1",
                    "backend": "candle",
                    "provider": "local",
                    "inputFingerprint": "sha256:input",
                    "status": "succeeded"
                }
            },
            "acquisition": {
                "status": "available",
                "availability": "available",
                "cachePath": relative,
                "contentHash": content_hash,
                "width": 32,
                "height": 32
            }
        }),
    }
}

#[tokio::test]
async fn materialization_replaces_invalid_cache_deterministically_and_survives_catalog_deletion() {
    let _hook_guard = lock_catalog_scan_hook().await;
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let catalog_root = temporary.path().join("materialize-catalog");
    let registry = CatalogRegistry::new(&settings.config_dir);
    let mut catalog = registry
        .create_catalog(&catalog_root, "Materialize")
        .expect("catalog creates");
    let records = [
        cached_materialization_record(&catalog_root, "one", [255, 0, 0]),
        cached_materialization_record(&catalog_root, "two", [0, 255, 0]),
        cached_materialization_record(&catalog_root, "three", [0, 0, 255]),
        cached_materialization_record(&catalog_root, "four", [255, 255, 0]),
    ];
    catalog.append_records(&records).expect("records append");
    let mut contract = catalog.contract_state().expect("contract reads");
    contract.analyzer_versions =
        BTreeMap::from([("vision_tagger".to_owned(), "model@sha256:abc".to_owned())]);
    catalog
        .set_contract_state(&contract)
        .expect("analyzer version persists");
    let order = catalog
        .search_records(&CatalogSearchRequest {
            limit: 200,
            filters: vec![
                CatalogRecordFilter {
                    field: "personCount".to_owned(),
                    values: vec!["1".to_owned()],
                },
                CatalogRecordFilter {
                    field: "medium".to_owned(),
                    values: vec!["photograph".to_owned()],
                },
            ],
            sample_seed: Some(77),
            ..CatalogSearchRequest::default()
        })
        .expect("sample query")
        .records;
    let unavailable_id = order[0].id.clone();
    let mut unavailable = records
        .iter()
        .find(|record| record.id == unavailable_id)
        .expect("ordered record exists")
        .clone();
    let invalid_cache = catalog_root.join("images/cache/invalid.png");
    std::fs::write(&invalid_cache, b"not an image").expect("invalid cache writes");
    unavailable.metadata["url"] = json!("http://127.0.0.1/private.png");
    unavailable.metadata["acquisition"]["cachePath"] = json!("images/cache/invalid.png");
    unavailable.metadata["acquisition"]["contentHash"] = json!("00");
    catalog
        .append_records(&[unavailable])
        .expect("unavailable update persists");
    let catalog_id = catalog.descriptor().id.clone();
    drop(catalog);

    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Catalog materialization"}),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let body = json!({
        "projectId": project_id,
        "name": "Seeded catalog dataset",
        "requestedCount": 2,
        "seed": 77,
        "deduplicationPolicy": "content_and_perceptual",
        "query": {
            "limit": 48,
            "filters": [
                {"field": "personCount", "values": ["1", "1"]},
                {"field": "medium", "values": ["photograph"]}
            ],
            "text": "",
            "deduplicate": false
        },
        "includeGeneratedCaption": true,
        "includeGeneratedTags": true,
        "idempotencyKey": "materialization-retry-1"
    });
    let (status, created) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/materialize"),
        body.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["requestedCount"], 2);
    assert_eq!(created["materializedCount"], 2);
    assert_eq!(created["skipped"][0]["recordId"], unavailable_id);
    assert_eq!(created["skipped"][0]["reason"], "unavailable");
    assert_eq!(
        created["dataset"]["catalogMaterialization"]["selectedRecordIds"],
        created["selectedRecordIds"]
    );
    assert_eq!(
        created["dataset"]["catalogMaterialization"]["query"]["filters"][0]["field"], "medium",
        "canonical query sorts filters"
    );
    assert_eq!(
        created["dataset"]["catalogMaterialization"]["query"]["filters"][1]["values"],
        json!(["1"]),
        "canonical query deduplicates values"
    );
    assert_eq!(
        created["dataset"]["catalogMaterialization"]["analyzerVersions"]["vision_tagger"],
        "model@sha256:abc"
    );
    assert!(created["dataset"]["items"][0]["caption"]["text"]
        .as_str()
        .expect("caption")
        .contains("generated caption"));
    assert!(created["dataset"]["items"][0]["caption"]["text"]
        .as_str()
        .expect("caption")
        .contains("full_body"));
    assert!(
        created["dataset"]["items"][0]["catalogSource"]["sourceCaption"]
            .as_str()
            .expect("source caption snapshot")
            .starts_with("source caption")
    );
    assert_eq!(
        created["dataset"]["catalogMaterialization"]["selectedRecordVersions"]
            .as_array()
            .expect("record versions")
            .len(),
        2
    );
    assert_eq!(
        created["dataset"]["catalogMaterialization"]["catalog"]["generation"], 2,
        "record append plus unavailable-record update advance catalog generation"
    );
    assert!(
        created["dataset"]["catalogMaterialization"]["materializationFingerprint"]
            .as_str()
            .is_some_and(|value| value.len() == 64)
    );
    assert!(
        created["dataset"]["catalogMaterialization"]["requestFingerprint"]
            .as_str()
            .is_some_and(|value| value.len() == 64)
    );
    assert!(
        created["dataset"]["catalogMaterialization"]["selectedRecordVersions"][0]
            ["analyzerProvenance"]
            .is_object()
    );
    let analyzer_provenance = &created["dataset"]["catalogMaterialization"]
        ["selectedRecordVersions"][0]["analyzerProvenance"];
    assert_eq!(
        analyzer_provenance["structured"]["inputFingerprint"]
            .as_str()
            .expect("bounded input fingerprint")
            .len(),
        512
    );
    assert_eq!(
        analyzer_provenance["structured"]["analyzers"]["person"]["modelVersion"],
        "sha256:person"
    );
    assert_eq!(
        analyzer_provenance["visionLanguage"]["runtimeRevision"],
        "runtime-v1"
    );
    assert_eq!(analyzer_provenance["visionLanguage"]["schemaVersion"], 1);
    assert_eq!(analyzer_provenance["visionLanguage"]["backend"], "mlx");
    assert_eq!(
        analyzer_provenance["semanticEmbedding"]["modelRevision"],
        "sha256:clip"
    );
    assert_eq!(
        analyzer_provenance["semanticEmbedding"]["provider"],
        "local"
    );
    assert!(
        !created.to_string().contains("not-provenance"),
        "unrelated and oversized analysis metadata is never copied into the project manifest"
    );
    assert!(
        analyzer_provenance["structured"]["analyzers"]["person"]
            .get("thresholds")
            .is_none(),
        "potentially unbounded analyzer thresholds are represented by the bounded global analyzer snapshot"
    );
    let first_item = &created["dataset"]["items"][0];
    assert!(
        first_item["caption"]["text"]
            .as_str()
            .expect("bounded caption")
            .len()
            <= 8_192
    );
    let catalog_source = &first_item["catalogSource"];
    assert!(
        catalog_source["sourceCaption"]
            .as_str()
            .expect("bounded source caption")
            .len()
            <= 2_048
    );
    assert!(
        catalog_source["generatedCaption"]
            .as_str()
            .expect("bounded generated caption")
            .len()
            <= 2_048
    );
    assert!(catalog_source["generatedTags"]
        .as_array()
        .is_some_and(|tags| tags.len() == 32));
    assert_eq!(
        catalog_source["truncation"]["sourceCaption"]["truncated"],
        true
    );
    assert!(catalog_source["truncation"]["sourceCaption"]["sha256"]
        .as_str()
        .is_some_and(|value| value.len() == 64));
    assert_eq!(
        catalog_source["truncation"]["generatedTags"]["originalCount"],
        42
    );
    for sentinel in [
        "CAPTION_TAIL_SENTINEL",
        "GENERATED_TAIL_SENTINEL",
        "TAG_TAIL_SENTINEL",
    ] {
        assert!(
            !created.to_string().contains(sentinel),
            "oversized caption/tag tails must not be persisted"
        );
    }
    let dataset_id = created["dataset"]["id"].as_str().expect("dataset id");

    let mut conflicting_retry = body.clone();
    conflicting_retry["requestedCount"] = json!(1);
    let (conflict_status, conflict) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/materialize"),
        conflicting_retry,
    )
    .await;
    assert_eq!(conflict_status, StatusCode::CONFLICT, "{conflict}");
    assert!(conflict["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("idempotencyKey")));

    let (delete_status, _) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/catalogs/{catalog_id}/on-disk"),
        Value::Null,
    )
    .await;
    assert_eq!(delete_status, StatusCode::OK);
    let (get_status, detached_dataset) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(get_status, StatusCode::OK);
    assert_eq!(detached_dataset["items"].as_array().unwrap().len(), 2);
    let (readiness_status, readiness) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/readiness"),
        Value::Null,
    )
    .await;
    assert_eq!(readiness_status, StatusCode::OK, "{readiness}");

    let (retry_status, retried) = request(
        app,
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/materialize"),
        body,
    )
    .await;
    assert_eq!(retry_status, StatusCode::OK, "{retried}");
    assert_eq!(retried["reusedExisting"], true);
    assert_eq!(retried["dataset"]["id"], dataset_id);
    assert_eq!(retried["selectedRecordIds"], created["selectedRecordIds"]);
    assert_eq!(retried["skipped"], created["skipped"]);
}

#[tokio::test]
async fn materialization_none_policy_allows_identical_verified_content() {
    let _hook_guard = lock_catalog_scan_hook().await;
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let catalog_root = temporary.path().join("duplicate-catalog");
    let registry = CatalogRegistry::new(&settings.config_dir);
    let mut catalog = registry
        .create_catalog(&catalog_root, "Duplicates")
        .expect("catalog creates");
    let first = cached_materialization_record(&catalog_root, "duplicate-a", [12, 34, 56]);
    let mut second = first.clone();
    second.id = "duplicate-b".to_owned();
    second.image_path = "images/duplicate-b.png".to_owned();
    second.metadata["caption"] = json!("second identical image");
    catalog
        .append_records(&[first, second])
        .expect("duplicate records append");
    let catalog_id = catalog.descriptor().id.clone();
    drop(catalog);
    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Duplicate policy"}),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id");
    let (invalid_status, invalid) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/materialize"),
        json!({
            "projectId": project_id,
            "name": "Invalid count",
            "requestedCount": 0,
            "seed": 5,
            "deduplicationPolicy": "none",
            "query": {"limit": 48},
            "idempotencyKey": "invalid-zero"
        }),
    )
    .await;
    assert_eq!(invalid_status, StatusCode::BAD_REQUEST);
    assert!(invalid["detail"]
        .as_str()
        .expect("invalid detail")
        .contains("requestedCount"));
    let (seed_status, seed_error) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/materialize"),
        json!({
            "projectId": project_id,
            "name": "Unsafe seed",
            "requestedCount": 1,
            "seed": 9_007_199_254_740_992_u64,
            "deduplicationPolicy": "none",
            "query": {"limit": 48},
            "idempotencyKey": "invalid-seed"
        }),
    )
    .await;
    assert_eq!(seed_status, StatusCode::BAD_REQUEST, "{seed_error}");
    assert!(seed_error["detail"]
        .as_str()
        .is_some_and(|detail| detail.contains("seed")));
    let (status, created) = request(
        app,
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/materialize"),
        json!({
            "projectId": project_id,
            "name": "Keep duplicates",
            "requestedCount": 2,
            "seed": 5,
            "deduplicationPolicy": "none",
            "query": {"limit": 48},
            "idempotencyKey": "duplicates-none"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["materializedCount"], 2);
    assert_eq!(created["dataset"]["items"].as_array().unwrap().len(), 2);
    assert_eq!(created["skipped"], json!([]));
}

#[tokio::test]
async fn timed_out_cache_verifier_cannot_overwrite_its_deterministic_replacement() {
    let _hook_guard = lock_catalog_scan_hook().await;
    timed_out_cache_verifier_case(Duration::ZERO).await;
}

/// sc-19764. Same case, but the staging verifier is stalled past the hooked
/// candidate's budget so the budget expires BEFORE the `before_stage` meeting
/// point — the ordering a loaded CI runner produces on its own.
///
/// The timeout signal used to be a two-party barrier wait taken on the
/// materialization future's own thread, i.e. on the test's current-thread
/// runtime. In this ordering the control side has not parked yet and can only
/// get there by being polled by that same, now-blocked runtime: the test wedged
/// permanently, still holding `catalog_scan_hook_test_lock`, and took every
/// later user of that lock down with it. On PR #2346 that surfaced as 594 of
/// 597 tests printing and then 21 silent minutes into the 30-minute
/// `parity-rust` cap.
#[tokio::test]
async fn timed_out_cache_verifier_survives_a_budget_that_expires_before_the_stage_rendezvous() {
    let _hook_guard = lock_catalog_scan_hook().await;
    timed_out_cache_verifier_case(Duration::from_millis(400)).await;
}

async fn timed_out_cache_verifier_case(verifier_delay: Duration) {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let catalog_root = temporary.path().join("late-cache-catalog");
    let registry = CatalogRegistry::new(&settings.config_dir);
    let mut catalog = registry
        .create_catalog(&catalog_root, "Late cache verifier")
        .expect("catalog creates");
    let records = [
        cached_materialization_record(&catalog_root, "late-a", [255, 0, 0]),
        cached_materialization_record(&catalog_root, "replacement-b", [0, 255, 0]),
    ];
    catalog.append_records(&records).expect("records append");
    let mut ordered = Vec::new();
    let mut cursor = None;
    loop {
        let page = catalog
            .search_records(&CatalogSearchRequest {
                cursor,
                limit: 200,
                sample_seed: Some(44),
                deduplicate: false,
                ..CatalogSearchRequest::default()
            })
            .expect("ordered records");
        ordered.extend(page.records);
        let Some(next) = page.next_cursor else {
            break;
        };
        cursor = Some(next);
    }
    let timed_out_id = ordered[0].id.clone();
    let expected_replacement_id = ordered[1].id.clone();
    let expected_replacement_hash = ordered[1].metadata["acquisition"]["contentHash"]
        .as_str()
        .expect("replacement hash")
        .to_owned();
    let catalog_id = catalog.descriptor().id.clone();
    drop(catalog);

    let app = create_app(settings).expect("app creates");
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({"name": "Late cache replacement"}),
    )
    .await;
    let project_id = project["id"].as_str().expect("project id").to_owned();
    let control = crate::catalogs::install_catalog_cache_stage_test_hook(verifier_delay);
    let materialization = tokio::spawn(async move {
        request(
            app,
            "POST",
            &format!("/api/v1/catalogs/{catalog_id}/materialize"),
            json!({
                "projectId": project_id,
                "name": "Late cache replacement",
                "requestedCount": 1,
                "seed": 44,
                "deduplicationPolicy": "none",
                "query": {"limit": 48},
                "idempotencyKey": "late-cache-replacement"
            }),
        )
        .await
    });

    control.wait_before_stage().await;
    control.wait_for_timeout().await;
    control.wait_before_project_copy().await;
    control.release_stage().await;
    control.wait_for_stage_completion().await;
    control.release_project_copy().await;
    let (status, created) = materialization.await.expect("request task joins");
    assert_eq!(status, StatusCode::CREATED, "{created}");
    assert_eq!(created["skipped"][0]["recordId"], timed_out_id);
    assert_eq!(created["selectedRecordIds"][0], expected_replacement_id);
    assert_eq!(
        created["dataset"]["items"][0]["contentHash"],
        expected_replacement_hash
    );
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
    let (app, state) = create_app_with_state(settings).expect("app and state create");

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
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 0).await;
    let (_, initial_status) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{catalog_id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(
        initial_status["processing"]["state"], "completed",
        "empty fixture scan completes"
    );
    assert!(
        initial_status["storage"].is_null(),
        "hot status responses must not recursively account artifact files"
    );

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
    std::fs::write(catalog_root.join("thumbnails/one.jpg"), b"thumbnail-bytes").unwrap();
    let mut contract = catalog.contract_state().unwrap();
    contract.analyzer_versions =
        BTreeMap::from([("person_detector".to_owned(), "model@sha256:abc".to_owned())]);
    contract.checkpoints = BTreeMap::from([("scan".to_owned(), json!({"shard": 4, "row": 80}))]);
    contract.processing = CatalogProcessingProgress {
        state: CatalogProcessingState::Running,
        candidate_count: 12,
        processed_count: 4,
        accepted_count: 3,
        rejected_count: 1,
        error_count: 1,
        message: Some("paused by user".to_owned()),
        updated_at: "2026-07-26T00:00:00Z".to_owned(),
    };
    catalog.set_contract_state(&contract).unwrap();
    catalog
        .request_processing_control(
            0,
            sceneworks_core::catalog_store::CatalogProcessingDesiredState::Paused,
        )
        .expect("fixture pause intent persists");
    contract.processing.state = CatalogProcessingState::Paused;
    catalog
        .set_processing_progress(&contract.processing)
        .expect("fixture paused progress persists");
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

    let curation_uri = format!("/api/v1/catalogs/{catalog_id}/curation/query");
    let primary_query = json!({
        "limit": 2,
        "filters": [
            { "field": "medium", "values": ["photo"] },
            { "field": "personCount", "values": ["1"] },
            { "field": "qualifiedSingleFullBody", "values": ["true"] }
        ],
        "sampleSeed": 14959,
        "deduplicate": true,
        "includeTotal": true
    });
    let (status, curated) =
        request(app.clone(), "POST", &curation_uri, primary_query.clone()).await;
    assert_eq!(status, StatusCode::OK, "{curated}");
    assert!(
        !curated["items"].as_array().unwrap().is_empty(),
        "the seeded keyset may wrap onto a second page"
    );
    assert_eq!(curated["totalCount"], 2);
    assert!(curated["nextCursor"].is_string());
    let (status, replay) = request(app.clone(), "POST", &curation_uri, primary_query.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(replay["items"], curated["items"]);
    let thumbnail_response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .uri(format!(
                    "/api/v1/catalogs/{catalog_id}/records/one/thumbnail"
                ))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(thumbnail_response.status(), StatusCode::OK);
    assert_eq!(thumbnail_response.headers()["content-type"], "image/jpeg");
    assert_eq!(
        axum::body::to_bytes(thumbnail_response.into_body(), 1024)
            .await
            .unwrap()
            .as_ref(),
        b"thumbnail-bytes"
    );
    let (status, _) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{catalog_id}/records/not-in-catalog/thumbnail"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, indexed_facets) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{catalog_id}/curation/facets"),
        json!({
            "query": primary_query,
            "fields": ["medium", "personCount", "qualifiedSingleFullBody"],
            "limitPerFacet": 10
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{indexed_facets}");
    assert_eq!(indexed_facets["facets"][0]["values"][0]["value"], "photo");

    let review_uri = format!("/api/v1/catalogs/{catalog_id}/records/one/review");
    let (status, missing_reason) = request(
        app.clone(),
        "PUT",
        &review_uri,
        json!({ "decision": "exclude" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{missing_reason}");
    let (status, review) = request(
        app.clone(),
        "PUT",
        &review_uri,
        json!({ "decision": "exclude", "rejectionReason": "cropped feet" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{review}");
    assert_eq!(review["rejectionReason"], "cropped feet");
    let (status, included) = request(
        app.clone(),
        "PUT",
        &review_uri,
        json!({ "decision": "include", "rejectionReason": "must be cleared" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(included["rejectionReason"].is_null());

    let views_uri = format!("/api/v1/catalogs/{catalog_id}/saved-views");
    let (status, saved) = request(
        app.clone(),
        "POST",
        &views_uri,
        json!({ "name": "Full body photos", "query": {
            "limit": 48,
            "filters": [
                { "field": "medium", "values": ["photo"] },
                { "field": "personCount", "values": ["1"] },
                { "field": "qualifiedSingleFullBody", "values": ["true"] }
            ],
            "sampleSeed": 14959
        }}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{saved}");
    assert_eq!(saved["revision"], 0);
    let (status, duplicate) = request(
        app.clone(),
        "POST",
        &views_uri,
        json!({
            "name": "full BODY photos",
            "query": saved["query"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{duplicate}");
    let saved_id = saved["id"].as_str().unwrap();
    let saved_item_uri = format!("{views_uri}/{saved_id}");
    let (status, updated) = request(
        app.clone(),
        "PUT",
        &saved_item_uri,
        json!({
            "name": "Reviewed photos",
            "expectedRevision": 0,
            "query": saved["query"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{updated}");
    assert_eq!(updated["revision"], 1);
    let (status, stale) = request(
        app.clone(),
        "PUT",
        &saved_item_uri,
        json!({
            "name": "Stale edit",
            "expectedRevision": 0,
            "query": saved["query"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{stale}");

    for invalid_body in [
        json!({"limit": 10, "text": "!!!"}),
        json!({"limit": 10, "filters": [{
            "field": "analysis.notIndexed", "values": ["anything"]
        }]}),
    ] {
        let (status, _) = request(app.clone(), "POST", &curation_uri, invalid_body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

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
async fn catalog_status_omits_unbounded_storage_walk_while_detail_remains_exact() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let config_dir = settings.config_dir.clone();
    let catalog_root = temporary.path().join("storage-scale-catalog");
    let (app, _) = create_app_with_state(settings).expect("app and state create");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Storage scale catalog",
            "path": catalog_root.clone()
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let catalog_id = created["id"].as_str().unwrap().to_owned();

    let registry = CatalogRegistry::new(config_dir);
    let mut catalog = registry.open_attached(&catalog_id).unwrap();
    catalog
        .append_records(&[
            catalog_record("scale-one", "photo", 1),
            catalog_record("scale-two", "photo", 1),
            catalog_record("scale-three", "illustration", 0),
        ])
        .unwrap();
    catalog.close();

    const DIRECTORY_COUNT: usize = 16;
    const FILES_PER_DIRECTORY: usize = 32;
    const ARTIFACT: &[u8] = b"artifact-bytes";
    for directory_index in 0..DIRECTORY_COUNT {
        let directory = catalog_root
            .join("artifacts")
            .join(format!("scale-{directory_index:02}"));
        std::fs::create_dir_all(&directory).unwrap();
        for file_index in 0..FILES_PER_DIRECTORY {
            std::fs::write(
                directory.join(format!("artifact-{file_index:02}.bin")),
                ARTIFACT,
            )
            .unwrap();
        }
    }

    let (status, hot_status) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{catalog_id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{hot_status}");
    assert_eq!(hot_status["counts"]["recordCount"], 3);
    assert!(
        hot_status["storage"].is_null(),
        "the polling contract structurally excludes the recursive storage walk"
    );

    let (status, detail) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{catalog_id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{detail}");
    assert_eq!(
        detail["storage"]["artifactBytes"],
        (DIRECTORY_COUNT * FILES_PER_DIRECTORY * ARTIFACT.len()) as u64
    );
    assert_eq!(detail["counts"]["recordCount"], 3);
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
async fn analyzer_configuration_is_typed_validated_revisioned_and_persistent() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Analyzer settings",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id");
    assert_eq!(created["analyzerConfig"]["revision"], 0);
    assert_eq!(
        created["analyzerConfig"]["settings"]["structuredAnalysisEnabled"],
        true
    );

    let settings = json!({
        "structuredAnalysisEnabled": true,
        "visionAnalysisEnabled": true,
        "semanticEmbeddingsEnabled": true,
        "thresholds": {
            "personMinConfidence": 0.4,
            "faceMinConfidence": 0.7,
            "poseMinKeypointConfidence": 0.35,
            "prominentFrameFraction": 0.25,
            "frameEdgeMargin": 0.02,
            "minPoseCoverage": 0.8
        }
    });
    let (status, updated) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/catalogs/{id}/analyzer-config"),
        json!({
            "expectedRevision": 0,
            "settings": settings
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{updated}");
    assert_eq!(updated["analyzerConfig"]["revision"], 1);
    assert_eq!(updated["analyzerConfig"]["settings"], settings);

    let (status, conflict) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/catalogs/{id}/analyzer-config"),
        json!({
            "expectedRevision": 0,
            "settings": settings
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{conflict}");
    assert_eq!(conflict["code"], "catalog_analyzer_config_conflict");

    let mut invalid = settings.clone();
    invalid["thresholds"]["frameEdgeMargin"] = json!(0.5);
    let (status, invalid_response) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/catalogs/{id}/analyzer-config"),
        json!({
            "expectedRevision": 1,
            "settings": invalid
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{invalid_response}");
    assert_eq!(invalid_response["code"], "catalog_invalid");

    let mut missing_structured_gate = settings.clone();
    missing_structured_gate["structuredAnalysisEnabled"] = json!(false);
    let (status, invalid_response) = request(
        app.clone(),
        "PUT",
        &format!("/api/v1/catalogs/{id}/analyzer-config"),
        json!({
            "expectedRevision": 1,
            "settings": missing_structured_gate
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{invalid_response}");
    assert_eq!(invalid_response["code"], "catalog_invalid");

    let (_, persisted) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(persisted["analyzerConfig"]["revision"], 1);
    assert_eq!(persisted["analyzerConfig"]["settings"], settings);
}

#[tokio::test]
async fn catalog_analysis_uses_typed_revisioned_route_and_never_accepts_catalog_paths() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Semantic catalog",
            "path": temporary.path().join("catalog")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id");

    let (status, job) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/analyze"),
        json!({
            "expectedAnalyzerConfigRevision": 0,
            "requestedGpu": "gpu-1",
            "batchSize": 8
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["type"], "catalog_analysis");
    assert_eq!(job["requestedGpu"], "gpu-1");
    assert_eq!(job["payload"]["catalogId"], id);
    assert_eq!(job["payload"]["analyzerConfigRevision"], 0);
    assert_eq!(job["payload"]["batchSize"], 8);
    assert!(
        job["payload"].get("catalogRoot").is_none() && job["payload"].get("path").is_none(),
        "the worker must resolve catalog identity only through the attached registry"
    );

    let (status, stale) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/analyze"),
        json!({
            "expectedAnalyzerConfigRevision": 99,
            "requestedGpu": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{stale}");

    let (status, raw) = request(
        app,
        "POST",
        "/api/v1/jobs",
        json!({
            "type": "catalog_analysis",
            "payload": {
                "catalogId": id,
                "catalogRoot": temporary.path().join("attacker-controlled")
            },
            "requestedGpu": "gpu-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{raw}");
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
async fn unavailable_catalog_can_be_detached_without_touching_its_relocated_files() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let catalog_path = temporary.path().join("catalog");
    let relocated_path = temporary.path().join("relocated-catalog");
    let app = create_app(test_settings(&temporary)).expect("app creates");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Relocated",
            "path": catalog_path
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("catalog id");
    std::fs::rename(&catalog_path, &relocated_path).expect("catalog relocates outside registry");

    let (status, listed) = request(app.clone(), "GET", "/api/v1/catalogs", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(listed[0]["id"], id);
    assert_eq!(listed[0]["availability"], "unavailable");

    let (status, detached) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/catalogs/{id}"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{detached}");
    assert_eq!(detached["detached"], true);
    assert_eq!(detached["deletedOnDisk"], false);
    assert!(
        relocated_path.exists(),
        "detach never deletes relocated data"
    );

    let (_, listed) = request(app, "GET", "/api/v1/catalogs", Value::Null).await;
    assert_eq!(listed, json!([]));
}

#[tokio::test]
async fn parquet_create_pause_resume_and_completion_run_through_public_api_scheduler() {
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 50_000);
    let (app, state) = create_app_with_state(settings).expect("app and state create");

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
    // Deadline-based waits throughout this test, not fixed iteration counts: the old
    // budgets were calibrated on the fast self-hosted M5 and flaked once the suite moved
    // to the slower hosted macos-26 runners (sc-17723). These are completion waits, not
    // boundary assertions, so generous ceilings delete no coverage.
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while std::time::Instant::now() < deadline && paused_response.is_none() {
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
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
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

    // The driver publishes its terminal paused status before returning and
    // releasing the catalog processing lease. Status is therefore not a
    // quiescence signal: wait on the supervisor's event-driven teardown seam
    // before issuing a new lifecycle mutation that must acquire that lease.
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 50_000).await;

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
    // Resuming the 50k-record scan is the long pole of this test on a loaded runner (the
    // old 2000 x 5ms budget was the M5-calibrated one, and 120s after that was still a
    // hand-picked number). Same completion wait as `wait_for_catalog_scan_idle`, so it
    // takes the same work-derived budget.
    let deadline = std::time::Instant::now() + catalog_scan_idle_budget(50_000);
    while std::time::Instant::now() < deadline {
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
    let _hook_guard = lock_catalog_scan_hook().await;
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
                tokio::time::timeout(Duration::from_secs(60), terminal)
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

    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 100_000).await;
    let (_, completed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(completed["processing"]["state"], "completed", "{completed}");
    assert_eq!(completed["counts"]["recordCount"], 100_000);
}

#[tokio::test]
async fn successor_queued_after_closure_is_recovered_by_new_appstate() {
    let _hook_guard = lock_catalog_scan_hook().await;
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
    tokio::time::timeout(Duration::from_secs(60), cleanup_reached.wait())
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
    let stranded_contract = stranded.contract_state().expect("contract reads");
    assert_eq!(
        stranded_contract.processing.state,
        CatalogProcessingState::Paused,
        "{:?}",
        stranded_contract.processing
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
    wait_for_catalog_scan_idle(&restarted_state.catalog_scan_supervisor, 100_000).await;
    let (_, completed) = request(
        restarted_app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(completed["processing"]["state"], "completed", "{completed}");
    assert_eq!(completed["counts"]["recordCount"], 100_000);
}

#[tokio::test]
async fn transient_sqlite_contention_retries_and_completes_exactly() {
    let _hook_guard = lock_catalog_scan_hook().await;
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 1_000);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    state
        .catalog_scan_injected_sqlite_busy_failures
        .store(2, std::sync::atomic::Ordering::SeqCst);

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Contention retry",
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
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 1_000).await;
    let (_, completed) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(completed["processing"]["state"], "completed", "{completed}");
    assert_eq!(completed["counts"]["recordCount"], 1_000);
    assert_eq!(
        state
            .catalog_scan_injected_sqlite_busy_failures
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

#[tokio::test]
async fn exhausted_sqlite_contention_is_visible_and_actionable() {
    let _hook_guard = lock_catalog_scan_hook().await;
    let temporary = tempfile::tempdir().expect("temp directory");
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 10);
    let (app, state) =
        create_app_with_state(test_settings(&temporary)).expect("app and state create");
    state.catalog_scan_injected_sqlite_busy_failures.store(
        u64::from(crate::catalogs::CATALOG_SCAN_MAX_CONTENTION_RETRIES) + 1,
        std::sync::atomic::Ordering::SeqCst,
    );

    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Contention exhaustion",
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
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 10).await;
    let (_, failed) = request(
        app,
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(failed["processing"]["state"], "failed", "{failed}");
    assert!(
        failed["processing"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("database contention")),
        "{failed}"
    );
}

#[tokio::test]
async fn shutdown_during_contention_backoff_preserves_paused_resume_boundary() {
    let _hook_guard = lock_catalog_scan_hook().await;
    let temporary = tempfile::tempdir().expect("temp directory");
    let settings = test_settings(&temporary);
    let source = temporary.path().join("source.parquet");
    write_catalog_parquet(&source, 100_000);
    let (app, state) = create_app_with_state(settings).expect("app and state create");
    let (status, created) = request(
        app.clone(),
        "POST",
        "/api/v1/catalogs",
        json!({
            "name": "Backoff shutdown",
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
    let running = running.expect("scan starts");
    let (pause_status, paused) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/pause"),
        json!({"expectedRevision": running["processingControl"]["revision"]}),
    )
    .await;
    assert_eq!(pause_status, StatusCode::OK, "{paused}");
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 100_000).await;
    let (_, durable_pause) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(durable_pause["processing"]["state"], "paused");

    state
        .catalog_scan_injected_sqlite_busy_failures
        .store(1, std::sync::atomic::Ordering::SeqCst);
    let backoff_started = state.catalog_scan_contention_backoff_started.notified();
    tokio::pin!(backoff_started);
    backoff_started.as_mut().enable();
    let (resume_status, resumed) = request(
        app,
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({"expectedRevision": durable_pause["processingControl"]["revision"]}),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");
    tokio::time::timeout(Duration::from_secs(60), backoff_started)
        .await
        .expect("injected contention enters backoff");
    let shutdown = state.catalog_scan_supervisor.shutdown().await;
    assert_eq!(shutdown.requested, 1);
    assert_eq!(shutdown.joined, 1);

    let registry = CatalogRegistry::new(temporary.path().join("config"));
    let catalog = registry.open_attached(&id).expect("catalog opens");
    let contract = catalog.contract_state().expect("contract reads");
    assert_eq!(contract.processing.state, CatalogProcessingState::Paused);
    assert_eq!(
        catalog.processing_control().unwrap().desired_state,
        sceneworks_core::catalog_store::CatalogProcessingDesiredState::Running
    );
}

#[tokio::test]
async fn interrupted_bounded_driver_is_automatically_recovered_from_its_checkpoint() {
    let _hook_guard = lock_catalog_scan_hook().await;
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

    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 30_000).await;
    let (_, recovery_started) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(recovery_started["processing"]["processedCount"], 25_000);
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 30_000).await;
    let (_, completed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(completed["processing"]["state"], "completed", "{completed}");
    assert_eq!(completed["counts"]["recordCount"], 30_000);
}

#[tokio::test]
async fn status_retries_after_a_recovered_generation_exits_incomplete() {
    let _hook_guard = lock_catalog_scan_hook().await;
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

    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 80_000).await;

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
    tokio::time::timeout(Duration::from_secs(60), before_recovered_start.wait())
        .await
        .expect("recovered generation reaches its prestart barrier");

    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 80_000).await;
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
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 80_000).await;
    let (_, completed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(completed["processing"]["state"], "completed", "{completed}");
    assert_eq!(completed["counts"]["recordCount"], 80_000);
}

#[tokio::test]
async fn runtime_scan_failure_stays_terminal_until_explicit_resume() {
    let _hook_guard = lock_catalog_scan_hook().await;
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
            "name": "Terminal runtime failure",
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

    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 30_000).await;

    let original = temporary.path().join("source-original.parquet");
    std::fs::rename(&source, &original).expect("original source moves aside");
    write_catalog_parquet(&source, 30_000);
    let (_, recovery_started) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(recovery_started["processing"]["processedCount"], 25_000);
    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 30_000).await;

    let (_, failed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(failed["processing"]["state"], "failed", "{failed}");
    assert!(failed["processing"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("inspect the worker logs")));
    assert_eq!(
        state.catalog_scan_supervisor.active_count().await,
        0,
        "status must not automatically retry a terminal runtime failure"
    );
    let (_, repeated) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(repeated["processing"], failed["processing"]);
    assert_eq!(state.catalog_scan_supervisor.active_count().await, 0);

    std::fs::remove_file(&source).expect("replacement source removes");
    std::fs::rename(&original, &source).expect("original source restores");
    let revision = failed["processingControl"]["revision"]
        .as_u64()
        .expect("processing revision");
    let (resume_status, resumed) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/catalogs/{id}/resume"),
        json!({ "expectedRevision": revision }),
    )
    .await;
    assert_eq!(resume_status, StatusCode::OK, "{resumed}");

    wait_for_catalog_scan_idle(&state.catalog_scan_supervisor, 30_000).await;
    let (_, completed) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/catalogs/{id}/status"),
        Value::Null,
    )
    .await;
    assert_eq!(completed["processing"]["state"], "completed", "{completed}");
    assert_eq!(completed["counts"]["recordCount"], 30_000);
}

#[tokio::test]
async fn transient_prestart_read_lease_cannot_strand_scheduled_processing() {
    let _hook_guard = lock_catalog_scan_hook().await;
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
    // Deadline-based, not a fixed iteration count — M5-calibrated budgets flake on the
    // slower hosted macos-26 runners (sc-17723).
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
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
    // Deadline-based, not a fixed iteration count: 2000 x 3ms (~6s) was calibrated on
    // the fast self-hosted M5 and flaked once the suite moved to the slower hosted
    // macos-26 runners, where resuming a 50k-record scan can outlast that budget under
    // load (sc-17723). This is a completion wait, not a boundary assertion, so a
    // generous ceiling deletes no coverage — the assert below still demands the full
    // record count. Work-derived rather than hand-picked, same as
    // `wait_for_catalog_scan_idle`.
    let resume_deadline = std::time::Instant::now() + catalog_scan_idle_budget(50_000);
    while std::time::Instant::now() < resume_deadline {
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
    let _hook_guard = lock_catalog_scan_hook().await;
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
    let _hook_guard = lock_catalog_scan_hook().await;
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
    let _hook_guard = lock_catalog_scan_hook().await;
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
    let _hook_guard = lock_catalog_scan_hook().await;
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
    let _hook_guard = lock_catalog_scan_hook().await;
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
