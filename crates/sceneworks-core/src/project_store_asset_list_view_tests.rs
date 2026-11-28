//! Summary listings and the asset change journal (incremental catalog refresh).

use super::{
    connect_project_db, AssetListFilter, AssetListView, AssetScope, AssetStatusPatch, ProjectStore,
    ProjectStoreError, ASSET_INDEX_VERSION, ASSET_INDEX_VERSION_KEY, PROJECT_SCHEMA_VERSION,
};
use rusqlite::params;
use serde_json::{json, Value};

const ALL_INCLUDING_HIDDEN: AssetListFilter = AssetListFilter {
    include_rejected: true,
    include_trashed: true,
    scope: AssetScope::All,
};

const VISIBLE_ONLY: AssetListFilter = AssetListFilter {
    include_rejected: false,
    include_trashed: false,
    scope: AssetScope::All,
};

struct Fixture {
    _temp_dir: tempfile::TempDir,
    store: ProjectStore,
    project_id: String,
    project_path: std::path::PathBuf,
}

fn fixture() -> Fixture {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Catalog").expect("project creates");
    let project_path = store.find_project_path(&project.id).unwrap();
    store
        .write_generation_set(
            &project.id,
            "job-1",
            &json!({
                "id": "set",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": "a shared set prompt",
                "negativePrompt": "a shared negative",
                "createdAt": "2026-05-25T00:00:00Z",
                "recipe": {
                    "prompt": "a shared set prompt",
                    "seed": 7,
                    "loras": [{ "id": "lora-a", "weight": 0.8 }],
                    "rawAdapterSettings": { "promptBatch": ["x".repeat(4096)] },
                },
            }),
            None,
        )
        .expect("generation set writes");
    Fixture {
        project_id: project.id,
        project_path,
        _temp_dir: temp_dir,
        store,
    }
}

fn persist(fixture: &Fixture, asset_id: &str, created_at: &str) {
    let media = format!("assets/images/set/{asset_id}.png");
    std::fs::create_dir_all(fixture.project_path.join("assets/images/set")).unwrap();
    std::fs::write(fixture.project_path.join(&media), b"png").unwrap();
    fixture
        .store
        .persist_generated_asset(
            &fixture.project_id,
            "job-1",
            "set",
            &json!({
                "assetId": asset_id,
                "mediaPath": media,
                "mimeType": "image/png",
                "displayName": asset_id,
                "createdAt": created_at,
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": format!("prompt for {asset_id}"),
                "seed": 42,
                "loras": [{ "id": "lora-a", "weight": 0.8, "path": "x".repeat(2048) }],
                "rawAdapterSettings": {
                    "faceLikeness": { "score": 0.91, "detected": true },
                    "vedaExecution": { "applied": true },
                    "promptBatch": ["y".repeat(4096)],
                    "mossImageEditExecutionReport": { "samples": ["z".repeat(4096)] },
                },
            }),
        )
        .expect("asset persists");
}

fn ids(assets: &[Value]) -> Vec<&str> {
    assets
        .iter()
        .map(|asset| asset["id"].as_str().unwrap())
        .collect()
}

#[test]
fn summary_view_drops_replay_bulk_but_keeps_catalog_fields_and_detail_stays_full() {
    let fixture = fixture();
    persist(&fixture, "asset_a", "2026-05-25T00:00:01Z");

    let (full, _) = fixture
        .store
        .list_assets_view_with_filesystem_operations(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Full,
        )
        .unwrap();
    let (summary, _) = fixture
        .store
        .list_assets_view_with_filesystem_operations(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
        )
        .unwrap();
    let (full, summary) = (&full[0], &summary[0]);
    // Precondition: the full envelope really carries the bulk the summary drops.
    assert!(full.pointer("/generationSet/recipe").is_some());
    assert!(full.pointer("/recipe/loras/0").is_some());
    assert!(full
        .pointer("/recipe/rawAdapterSettings/promptBatch")
        .is_some());
    assert!(full.get("listView").is_none(), "full items are not marked");

    assert_eq!(summary["listView"], json!("summary"));
    for pointer in [
        "/modelProvenance",
        "/generationSet/recipe",
        "/generationSet/prompt",
        "/generationSet/negativePrompt",
        "/recipe/loras",
        "/recipe/modelProvenance",
        "/recipe/rawAdapterSettings/promptBatch",
        "/recipe/rawAdapterSettings/mossImageEditExecutionReport",
    ] {
        assert!(
            summary.pointer(pointer).is_none(),
            "summary must drop {pointer}"
        );
    }
    // Everything a catalog surface reads is unchanged.
    for pointer in [
        "/id",
        "/type",
        "/displayName",
        "/createdAt",
        "/file",
        "/url",
        "/status",
        "/lineage",
        "/metadata",
        "/generationSetId",
        "/generationSet/id",
        "/generationSet/jobId",
        "/recipe/prompt",
        "/recipe/mode",
        "/recipe/model",
        "/recipe/seed",
        "/recipe/normalizedSettings",
        "/recipe/rawAdapterSettings/faceLikeness",
        "/recipe/rawAdapterSettings/vedaExecution",
    ] {
        assert_eq!(
            summary.pointer(pointer),
            full.pointer(pointer),
            "summary must keep {pointer}"
        );
    }
    assert!(
        serde_json::to_vec(summary).unwrap().len() * 4 < serde_json::to_vec(full).unwrap().len(),
        "the summary is a fraction of the full envelope"
    );

    // The detail route is unchanged: it still returns the full envelope.
    let detail = fixture
        .store
        .get_asset(&fixture.project_id, "asset_a")
        .unwrap();
    assert!(detail.pointer("/recipe/loras/0").is_some());
    assert!(detail
        .pointer("/recipe/rawAdapterSettings/promptBatch")
        .is_some());
    assert!(detail.get("listView").is_none());
}

#[test]
fn change_cursor_returns_only_new_changed_and_removed_assets() {
    let fixture = fixture();
    let store = &fixture.store;
    let project_id = fixture.project_id.as_str();
    persist(&fixture, "asset_a", "2026-05-25T00:00:01Z");
    persist(&fixture, "asset_b", "2026-05-25T00:00:02Z");

    // No cursor: the whole listing, flagged as a reset.
    let (first, _) = store
        .list_asset_changes(project_id, VISIBLE_ONLY, AssetListView::Summary, None)
        .unwrap();
    assert!(first.reset);
    assert_eq!(ids(&first.assets), ["asset_b", "asset_a"]);
    assert!(first.removed_ids.is_empty());
    assert_eq!(first.assets[0]["listView"], json!("summary"));

    // Nothing changed: an empty delta, and the cursor does not move.
    let (unchanged, _) = store
        .list_asset_changes(
            project_id,
            VISIBLE_ONLY,
            AssetListView::Summary,
            Some(&first.cursor),
        )
        .unwrap();
    assert!(!unchanged.reset);
    assert!(unchanged.assets.is_empty() && unchanged.removed_ids.is_empty());
    assert_eq!(unchanged.cursor, first.cursor);

    // A new asset and a changed one come back; untouched assets do not.
    persist(&fixture, "asset_c", "2026-05-25T00:00:03Z");
    store
        .update_asset_status(
            project_id,
            "asset_a",
            AssetStatusPatch {
                favorite: Some(true),
                ..AssetStatusPatch::default()
            },
        )
        .unwrap();
    let (delta, _) = store
        .list_asset_changes(
            project_id,
            VISIBLE_ONLY,
            AssetListView::Summary,
            Some(&first.cursor),
        )
        .unwrap();
    assert!(!delta.reset);
    assert_eq!(ids(&delta.assets), ["asset_c", "asset_a"]);
    assert_eq!(delta.assets[1]["status"]["favorite"], json!(true));
    assert!(delta.removed_ids.is_empty());
    assert_ne!(delta.cursor, first.cursor);

    // Trashing out of the filter and purging both surface as removals; a client that includes
    // hidden assets sees the trashed one as a change instead.
    store.delete_asset(project_id, "asset_c").unwrap();
    store.purge_asset(project_id, "asset_b", true).unwrap();
    let (removed, _) = store
        .list_asset_changes(
            project_id,
            VISIBLE_ONLY,
            AssetListView::Summary,
            Some(&delta.cursor),
        )
        .unwrap();
    assert!(removed.assets.is_empty());
    let mut removed_ids = removed.removed_ids.clone();
    removed_ids.sort();
    assert_eq!(removed_ids, ["asset_b", "asset_c"]);
    let (hidden_included, _) = store
        .list_asset_changes(
            project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
            Some(&delta.cursor),
        )
        .unwrap();
    assert_eq!(ids(&hidden_included.assets), ["asset_c"]);
    assert_eq!(hidden_included.assets[0]["status"]["trashed"], json!(true));
    assert_eq!(hidden_included.removed_ids, ["asset_b"]);

    // Replaying the deltas onto the first listing reproduces a fresh full listing.
    let (fresh, _) = store
        .list_asset_changes(
            project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
            None,
        )
        .unwrap();
    assert_eq!(ids(&fresh.assets), ["asset_c", "asset_a"]);
    assert_eq!(fresh.cursor, hidden_included.cursor);
}

#[test]
fn change_cursor_from_another_database_resets_and_malformed_cursor_is_rejected() {
    let fixture = fixture();
    persist(&fixture, "asset_a", "2026-05-25T00:00:01Z");
    let (first, _) = fixture
        .store
        .list_asset_changes(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Full,
            None,
        )
        .unwrap();
    let (epoch, seq) = first.cursor.rsplit_once('.').unwrap();
    assert!(!epoch.is_empty() && seq.parse::<i64>().unwrap() > 0);

    // A cursor minted by a different project.db (another epoch), or one AHEAD of this
    // database's journal, cannot be merged against: the answer is a full reset.
    for stale in [format!("someotherepoch.{seq}"), format!("{epoch}.999999")] {
        let (changes, _) = fixture
            .store
            .list_asset_changes(
                &fixture.project_id,
                ALL_INCLUDING_HIDDEN,
                AssetListView::Full,
                Some(&stale),
            )
            .unwrap();
        assert!(changes.reset, "{stale} must reset");
        assert_eq!(ids(&changes.assets), ["asset_a"]);
        assert_eq!(changes.cursor, first.cursor);
    }
    for malformed in ["nodot", "epoch.-1", "epoch.x", ".5", "bad epoch.5"] {
        let error = fixture
            .store
            .list_asset_changes(
                &fixture.project_id,
                ALL_INCLUDING_HIDDEN,
                AssetListView::Full,
                Some(malformed),
            )
            .unwrap_err();
        assert!(
            matches!(error, ProjectStoreError::BadRequest(_)),
            "{malformed}: {error:?}"
        );
    }
}

#[test]
fn full_reindex_restamps_every_asset_so_incremental_clients_resync() {
    let fixture = fixture();
    persist(&fixture, "asset_a", "2026-05-25T00:00:01Z");
    persist(&fixture, "asset_b", "2026-05-25T00:00:02Z");
    let (first, _) = fixture
        .store
        .list_asset_changes(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
            None,
        )
        .unwrap();
    fixture.store.reindex_project(&fixture.project_id).unwrap();
    let (after, _) = fixture
        .store
        .list_asset_changes(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
            Some(&first.cursor),
        )
        .unwrap();
    assert!(!after.reset);
    assert_eq!(ids(&after.assets), ["asset_b", "asset_a"]);
    assert!(after.removed_ids.is_empty());
}

/// A v8 database (indexed, no journal) upgrades in place: the migration backfills the journal
/// from the rows already indexed and does NOT rebuild the index from sidecars.
#[test]
fn v8_database_backfills_the_journal_without_a_sidecar_rebuild() {
    let fixture = fixture();
    persist(&fixture, "asset_a", "2026-05-25T00:00:01Z");
    {
        let connection = connect_project_db(&fixture.project_path).unwrap();
        connection
            .execute_batch(
                "
                drop trigger asset_changes_after_insert;
                drop trigger asset_changes_after_update;
                drop trigger asset_changes_after_delete;
                drop table asset_changes;
                delete from project_metadata where key = 'assetChangeEpoch';
                pragma user_version = 8;
                ",
            )
            .unwrap();
        let index_version: String = connection
            .query_row(
                "select value from project_metadata where key = ?1",
                params![ASSET_INDEX_VERSION_KEY],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(index_version, ASSET_INDEX_VERSION.to_string());
    }
    let (changes, operations) = fixture
        .store
        .list_asset_changes(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
            None,
        )
        .unwrap();
    assert_eq!(ids(&changes.assets), ["asset_a"]);
    assert_eq!(
        operations.sidecar_reads, 0,
        "the schema bump must not re-read sidecars"
    );
    let connection = connect_project_db(&fixture.project_path).unwrap();
    let version: i64 = connection
        .query_row("pragma user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, PROJECT_SCHEMA_VERSION);
    drop(connection);

    // The backfilled journal drives incremental reads from here on.
    persist(&fixture, "asset_b", "2026-05-25T00:00:02Z");
    let (delta, _) = fixture
        .store
        .list_asset_changes(
            &fixture.project_id,
            ALL_INCLUDING_HIDDEN,
            AssetListView::Summary,
            Some(&changes.cursor),
        )
        .unwrap();
    assert!(!delta.reset);
    assert_eq!(ids(&delta.assets), ["asset_b"]);
}

#[test]
fn listing_and_change_scan_use_indexes() {
    let fixture = fixture();
    let connection = connect_project_db(&fixture.project_path).unwrap();
    let plan = |sql: &str| -> String {
        let mut statement = connection
            .prepare(&format!("explain query plan {sql}"))
            .unwrap();
        statement
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join("\n")
    };
    let listing = plan("select id from assets order by created_at desc");
    assert!(
        listing.contains("idx_assets_listing") && !listing.contains("TEMP B-TREE"),
        "the listing order must come from the index, not a sort: {listing}"
    );
    // The scoped Library read filters on columns the listing index carries, so
    // hidden (e.g. Character Studio) rows are rejected from the index entry.
    let scoped = plan(
        "select id, asset_json from assets \
         where (0 or rejected = 0) and (0 or trashed = 0) and (null is null or type = null) \
           and (origin is null or origin not in ('character_studio', 'vector_workflow_intermediate')) \
         order by created_at desc",
    );
    assert!(
        scoped.contains("idx_assets_listing") && !scoped.contains("TEMP B-TREE"),
        "the scoped listing must scan the listing index in order: {scoped}"
    );
    let changes = plan("select asset_id from asset_changes where seq > 10");
    assert!(
        changes.contains("idx_asset_changes_seq"),
        "the change scan must be an index range: {changes}"
    );
    let head = plan("select max(seq) from asset_changes");
    assert!(
        head.contains("idx_asset_changes_seq") || head.contains("COVERING INDEX"),
        "the journal head must not scan the table: {head}"
    );
}
