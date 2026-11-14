use super::{
    apply_project_migrations, backfill_upscale_variant_lineage, build_generated_asset_sidecar,
    build_image_sidecar_parts, build_video_sidecar_parts, connect_project_db,
    ensure_project_db_ready, fail_next_asset_index_db_mutation, fail_next_asset_poster_db_read,
    fail_next_asset_sidecar_remove, fail_next_upscale_lineage_db_read, find_timeline_file,
    guess_mime_from_filename, index_timeline, install_ensure_ready_before_lock_hook,
    is_safe_relative_path, is_safe_upload_extension, normalize_asset_tags, normalize_image_upload,
    read_json, read_registry_payload, upload_extension, upscale_lineage_group, write_json,
    AssetIndexMutation, AssetScope, AssetStatusPatch, CharacterCreateInput, CharacterLookInput,
    CharacterReferenceInput, FilmReferenceInput, ProjectStore, ProjectStoreError, UploadAsset,
    WorkflowScan, ASSET_INDEX_DIRTY_MARKER, ASSET_INDEX_VERSION_KEY, GLOBAL_KEYPOINTS_PROJECT_ID,
    GLOBAL_POSES_PROJECT_ID, IMPORTED_WORKFLOW_KEY, MAX_REGISTRY_BYTES, ORPHANED_SIDECAR_DIR,
    PROJECT_FOLDERS, PROJECT_SCHEMA_VERSION, REFERENCE_PACK_SCHEMA_VERSION, SAFE_UPLOAD_EXTENSIONS,
    UPSCALE_LINEAGE_QUERY,
};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use std::{fs, path::Path};

fn jpeg_fixture(rgb: [u8; 3]) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90)
        .encode(&rgb, 1, 1, image::ExtendedColorType::Rgb8)
        .expect("test JPEG encodes");
    bytes
}

#[cfg(unix)]
fn create_file_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, destination)
}

#[cfg(windows)]
fn create_file_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(source, destination)
}

#[cfg(unix)]
fn create_dir_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, destination)
}

#[cfg(windows)]
fn create_dir_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(source, destination)
}

#[test]
fn recent_project_registry_cache_reuses_reads_and_invalidates_on_external_change() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let writer = ProjectStore::new(&data_dir, "test-version");
    let created = writer.create_project("Cached").expect("project creates");

    let reader = ProjectStore::new(&data_dir, "test-version");
    assert_eq!(
        reader.list_projects().expect("first list"),
        vec![created.clone()]
    );
    assert_eq!(reader.lock.lock().disk_reads, 1);
    assert_eq!(
        reader.list_projects().expect("cached list"),
        vec![created.clone()]
    );
    assert_eq!(
        reader.lock.lock().disk_reads,
        1,
        "an unchanged registry should not be reread"
    );

    let registry_path = reader.registry_path();
    let mut registry = std::fs::read(&registry_path).expect("registry reads");
    registry.push(b'\n');
    std::fs::write(&registry_path, registry).expect("external registry change writes");

    assert_eq!(
        reader.list_projects().expect("invalidated list"),
        vec![created]
    );
    assert_eq!(
        reader.lock.lock().disk_reads,
        2,
        "a changed registry fingerprint must invalidate the cache"
    );
}

#[test]
fn oversized_recent_project_registry_fails_before_fingerprinting_or_parsing() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_dir).expect("data dir creates");
    let registry_path = data_dir.join("recent-projects.json");
    std::fs::File::create(&registry_path)
        .expect("registry creates")
        .set_len(MAX_REGISTRY_BYTES + 1)
        .expect("oversized sparse registry creates");

    let assert_size_error = |error: ProjectStoreError| match error {
        ProjectStoreError::Io(error) => {
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert!(
                error.to_string().contains("maximum supported size"),
                "size failure should be actionable: {error}"
            );
        }
        other => panic!("expected deterministic registry size error, got {other}"),
    };

    assert_size_error(
        read_registry_payload(&registry_path)
            .expect_err("bounded registry reader must reject oversized input"),
    );
    assert_size_error(
        ProjectStore::new(&data_dir, "test-version")
            .list_projects()
            .expect_err("cache fingerprint must reject oversized input before parsing"),
    );
}

#[cfg(unix)]
#[test]
fn recent_project_registry_cache_detects_same_size_rewrite_with_restored_mtime() {
    use std::fs::{File, FileTimes, OpenOptions};
    use std::io::Write;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let writer = ProjectStore::new(&data_dir, "test-version");
    writer.create_project("Cached").expect("project creates");

    let reader = ProjectStore::new(&data_dir, "test-version");
    reader.list_projects().expect("first list");
    assert_eq!(reader.lock.lock().disk_reads, 1);

    let registry_path = reader.registry_path();
    let metadata = std::fs::metadata(&registry_path).expect("registry metadata");
    let original = std::fs::read_to_string(&registry_path).expect("registry reads");
    let rewritten = original.replacen("\"Cached\"", "\"Mutate\"", 1);
    assert_ne!(rewritten, original, "test must rewrite registry content");
    assert_eq!(
        rewritten.len(),
        original.len(),
        "test rewrite must preserve file length"
    );
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&registry_path)
        .expect("registry opens for rewrite");
    file.write_all(rewritten.as_bytes())
        .expect("same-size registry rewrite");
    file.sync_all().expect("registry rewrite syncs");
    File::open(&registry_path)
        .expect("registry reopens")
        .set_times(
            FileTimes::new()
                .set_accessed(metadata.accessed().expect("accessed time"))
                .set_modified(metadata.modified().expect("modified time")),
        )
        .expect("mtime restores");

    reader.list_projects().expect("invalidated list");
    let cache = reader.lock.lock();
    assert_eq!(
        cache.disk_reads, 2,
        "filesystem change revision must invalidate same-size same-mtime rewrites"
    );
    assert_eq!(
        cache.items[0].name.as_deref(),
        Some("Mutate"),
        "cache must contain the externally rewritten registry"
    );
}

#[cfg(windows)]
#[test]
fn windows_registry_cache_detects_between_window_rewrite_with_restored_mtime() {
    use std::fs::{FileTimes, OpenOptions};

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let writer = ProjectStore::new(&data_dir, "test-version");
    let created = writer.create_project("Cached").expect("project creates");
    let registry_path = writer.registry_path();

    // Trailing JSON whitespace makes the registry large enough for a
    // localized interior rewrite without changing its parsed contents.
    let mut original = std::fs::read(&registry_path).expect("registry reads");
    let json_len = original.len();
    original.resize(16 * 1024, b' ');
    std::fs::write(&registry_path, &original).expect("padded registry writes");

    let reader = ProjectStore::new(&data_dir, "test-version");
    assert_eq!(
        reader.list_projects().expect("first list"),
        vec![created.clone()]
    );
    assert_eq!(reader.lock.lock().disk_reads, 1);
    let metadata = std::fs::metadata(&registry_path).expect("registry metadata");

    let mutation_index = json_len + (original.len() - json_len) / 2;
    let mut rewritten = original.clone();
    assert_eq!(rewritten[mutation_index], b' ');
    rewritten[mutation_index] = b'\n';
    assert_eq!(rewritten.len(), original.len());

    std::fs::write(&registry_path, rewritten).expect("same-size registry rewrite");
    OpenOptions::new()
        .write(true)
        .open(&registry_path)
        .expect("registry opens to restore timestamps")
        .set_times(
            FileTimes::new()
                .set_accessed(metadata.accessed().expect("accessed time"))
                .set_modified(metadata.modified().expect("modified time")),
        )
        .expect("mtime restores");
    assert_eq!(
        std::fs::metadata(&registry_path)
            .expect("restored metadata")
            .modified()
            .expect("restored mtime"),
        metadata.modified().expect("original mtime"),
        "fixture must restore last-write time exactly"
    );

    assert_eq!(
        reader.list_projects().expect("invalidated list"),
        vec![created]
    );
    assert_eq!(
        reader.lock.lock().disk_reads,
        2,
        "the exact Windows content fingerprint must invalidate a same-size \
             restored-mtime edit at any byte"
    );
}

#[test]
fn recent_project_registry_cache_serializes_concurrent_initial_reads() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let writer = ProjectStore::new(&data_dir, "test-version");
    let created = writer
        .create_project("Concurrent")
        .expect("project creates");
    let reader = Arc::new(ProjectStore::new(&data_dir, "test-version"));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let reader = Arc::clone(&reader);
            std::thread::spawn(move || reader.list_projects().expect("project list"))
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().expect("thread joins"), vec![created.clone()]);
    }
    assert_eq!(
        reader.lock.lock().disk_reads,
        1,
        "the synchronized cache should perform one disk read"
    );
}

#[test]
fn stage_draft_file_preserves_media_rules_and_rejects_escapes() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source");
    let destination = temp.path().join("destination");
    fs::create_dir_all(source.join("nested")).unwrap();
    let source = fs::canonicalize(source).unwrap();
    fs::create_dir_all(&destination).unwrap();
    fs::write(source.join("nested/image.png"), b"reference fixture").unwrap();
    fs::write(source.join("clip.wav"), b"audio fixture").unwrap();
    super::stage_draft_file(
        &source,
        &destination,
        "nested/image.png",
        "hero",
        "reference",
        "image",
        false,
    )
    .unwrap();
    super::stage_draft_file(
        &source,
        &destination,
        "clip.wav",
        "music",
        "sound",
        "audio",
        false,
    )
    .unwrap();
    assert_eq!(
        fs::read(destination.join("nested/image.png")).unwrap(),
        b"reference fixture"
    );
    assert_eq!(
        fs::read(destination.join("clip.wav")).unwrap(),
        b"audio fixture"
    );
    super::stage_draft_file(
        &source,
        &destination,
        "dialogue.wav",
        "speech",
        "sound",
        "audio",
        true,
    )
    .unwrap();
    assert!(!destination.join("dialogue.wav").exists());
    let error = super::stage_draft_file(
        &source,
        &destination,
        "missing.wav",
        "music",
        "sound",
        "audio",
        false,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("missing its staged audio missing.wav"));
    for (kind, media) in [("reference", "image"), ("sound", "audio")] {
        let error = super::stage_draft_file(
            &source,
            &destination,
            "../escape",
            "role",
            kind,
            media,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsafe file path"));
        #[cfg(unix)]
        {
            let outside = temp.path().join(format!("outside-{kind}"));
            fs::write(&outside, b"outside").unwrap();
            let link = format!("link-{kind}");
            std::os::unix::fs::symlink(&outside, source.join(&link)).unwrap();
            let error =
                super::stage_draft_file(&source, &destination, &link, "role", kind, media, false)
                    .unwrap_err();
            assert!(error
                .to_string()
                .contains("outside its immutable draft input"));
        }
    }
}

// Real processes bypass the in-process project mutex. These fixtures perform
// only SQLite/control work; the voice vector is opaque stored fixture data.
fn spawn_project_db_child(project: &Path, mode: &str, token: &str) -> std::process::Child {
    std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "project_store::tests::project_db_cross_process_child",
            "--nocapture",
        ])
        .env("SW_PROJECT_DB_CHILD", mode)
        .env("SW_PROJECT_DB_PATH", project)
        .env("SW_PROJECT_DB_TOKEN", token)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap()
}

fn wait_for_db_child_marker(path: &Path) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "child marker missing: {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn assert_db_child_succeeded(child: std::process::Child) {
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "child failed: {} {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn project_db_cross_process_child() {
    let Ok(mode) = std::env::var("SW_PROJECT_DB_CHILD") else {
        return;
    };
    let project = std::path::PathBuf::from(std::env::var_os("SW_PROJECT_DB_PATH").unwrap());
    let token = std::env::var("SW_PROJECT_DB_TOKEN").unwrap();
    if mode == "hold" {
        let connection = connect_project_db(&project).unwrap();
        let transaction = rusqlite::Transaction::new_unchecked(
            &connection,
            rusqlite::TransactionBehavior::Immediate,
        )
        .unwrap();
        fs::write(project.join(format!("{token}.ready")), b"ready").unwrap();
        wait_for_db_child_marker(&project.join(format!("{token}.release")));
        transaction.commit().unwrap();
    } else {
        let connection = connect_project_db(&project).unwrap();
        if mode == "migrate" {
            assert_eq!(super::project_schema_version(&connection).unwrap(), 2);
        }
        fs::write(project.join(format!("{token}.ready")), b"ready").unwrap();
        if mode == "migrate" {
            apply_project_migrations(&connection).unwrap();
            // A repeat must be a no-op, including when another process won.
            apply_project_migrations(&connection).unwrap();
        } else if mode == "deferred" {
            // Reproduce the original lock upgrade with a real competing process.
            let transaction = connection.unchecked_transaction().unwrap();
            transaction
                .query_row("select count(*) from project_metadata", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap();
            let error = transaction
                .execute(
                    "insert into project_metadata (key, value) values ('deferred-fixture', '1')",
                    [],
                )
                .unwrap_err();
            assert!(
                matches!(error, rusqlite::Error::SqliteFailure(ref sqlite, _) if sqlite.code == rusqlite::ErrorCode::DatabaseBusy)
            );
        } else {
            let store = crate::voice_store::SavedVoiceStore::new(&project);
            store
                .create_saved_voice(
                    "project_fixture",
                    crate::voice_store::SavedVoiceCreateInput {
                        name: "Fixture voice".to_owned(),
                        reference_audio_asset_id: "asset_fixture".to_owned(),
                        embedding: vec![0.0; crate::voice_store::VOICE_EMBEDDING_DIM],
                    },
                    0.92,
                )
                .unwrap();
        }
    }
    fs::write(project.join(format!("{token}.done")), b"done").unwrap();
}

#[test]
fn project_writer_waits_for_other_process_before_read_then_write() {
    let temp = tempfile::tempdir().unwrap();
    let mut connection = connect_project_db(temp.path()).unwrap();
    apply_project_migrations(&connection).unwrap();
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    assert_db_child_succeeded(spawn_project_db_child(temp.path(), "deferred", "upgrade"));
    let mut child = spawn_project_db_child(temp.path(), "voice", "writer");
    wait_for_db_child_marker(&temp.path().join("writer.ready"));
    std::thread::sleep(Duration::from_millis(150));
    assert!(
        child.try_wait().unwrap().is_none(),
        "read-before-write mutator must wait for the competing writer"
    );
    assert!(!temp.path().join("writer.done").exists());
    transaction.commit().unwrap();
    assert_db_child_succeeded(child);
    assert_eq!(
        connection
            .query_row("select count(*) from saved_voices", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn project_migrations_are_idempotent_across_waiting_processes() {
    let temp = tempfile::tempdir().unwrap();
    let mut connection = connect_project_db(temp.path()).unwrap();
    connection
        .execute_batch(
            "create table training_datasets (
            id text primary key, project_id text not null, name text not null,
            modality text not null, status text not null, version integer not null,
            file_path text not null, created_at text not null, updated_at text not null);
            pragma user_version = 2;",
        )
        .unwrap();
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    let mut children = [
        spawn_project_db_child(temp.path(), "migrate", "first"),
        spawn_project_db_child(temp.path(), "migrate", "second"),
    ];
    for token in ["first", "second"] {
        wait_for_db_child_marker(&temp.path().join(format!("{token}.ready")));
    }
    std::thread::sleep(Duration::from_millis(150));
    for child in &mut children {
        assert!(
            child.try_wait().unwrap().is_none(),
            "migration must wait for the cross-process lock"
        );
    }
    transaction.commit().unwrap();
    for child in children {
        assert_db_child_succeeded(child);
    }
    assert_eq!(
        super::project_schema_version(&connection).unwrap(),
        PROJECT_SCHEMA_VERSION
    );
    connection
        .prepare("select character_id, item_count from training_datasets")
        .unwrap();
    assert!(
        connection.is_autocommit(),
        "migration leaves no transaction open"
    );
    // A normal mutator after migration must never nest a migration transaction.
    index_timeline(
        temp.path(),
        &json!({"id":"timeline_fixture"}),
        "timelines/fixture.json",
    )
    .unwrap();
}

#[test]
fn project_migration_timeout_preserves_typed_sqlite_busy_error() {
    let temp = tempfile::tempdir().unwrap();
    let connection = connect_project_db(temp.path()).unwrap();
    let child = spawn_project_db_child(temp.path(), "hold", "holder");
    wait_for_db_child_marker(&temp.path().join("holder.ready"));
    connection.busy_timeout(Duration::from_millis(100)).unwrap();
    let error = apply_project_migrations(&connection).unwrap_err();
    fs::write(temp.path().join("holder.release"), b"release").unwrap();
    assert_db_child_succeeded(child);
    assert!(
        matches!(error, ProjectStoreError::Sqlite(rusqlite::Error::SqliteFailure(ref sqlite, _)) if sqlite.code == rusqlite::ErrorCode::DatabaseBusy)
    );
    assert_eq!(
        super::project_schema_version(&connection).unwrap(),
        0,
        "failed migration must not partially stamp the schema"
    );
    apply_project_migrations(&connection).unwrap();
}

/// sc-2022 added `training_datasets.character_id` to the migration without
/// bumping `PROJECT_SCHEMA_VERSION`, so DBs already stamped at the prior
/// version (2) hit the early-return gate and never got the column — the
/// dataset list query failed with "no such column: character_id". This
/// pins that an old-version DB lacking the column receives it on the next
/// `apply_project_migrations` and can be queried.
#[test]
fn migrations_backfill_training_dataset_character_id_on_old_db() {
    let connection = Connection::open_in_memory().expect("in-memory db");
    // Reproduce the pre-character_id table shape and stamp the DB at the
    // version it would have reached before sc-2022 landed.
    connection
        .execute_batch(
            "
                create table training_datasets (
                  id text primary key,
                  project_id text not null,
                  name text not null,
                  modality text not null,
                  status text not null,
                  version integer not null,
                  item_count integer not null default 0,
                  file_path text not null,
                  created_at text not null,
                  updated_at text not null
                );
                pragma user_version = 2;
                ",
        )
        .expect("seed old schema");

    let has_character_id = |conn: &Connection| {
        let mut statement = conn
            .prepare("pragma table_info(training_datasets)")
            .expect("table_info");
        let columns: Vec<String> = statement
            .query_map([], |row| row.get::<_, String>("name"))
            .expect("columns")
            .filter_map(Result::ok)
            .collect();
        columns.iter().any(|name| name == "character_id")
    };
    assert!(
        !has_character_id(&connection),
        "precondition: old DB lacks character_id"
    );

    apply_project_migrations(&connection).expect("migration runs");

    assert!(
        has_character_id(&connection),
        "migration must add character_id to a DB stamped at the prior version"
    );
    let stamped: i64 = connection
        .query_row("pragma user_version", [], |row| row.get(0))
        .expect("user_version");
    assert_eq!(stamped, PROJECT_SCHEMA_VERSION);
    // The exact list query that surfaced the bug now succeeds.
    connection
        .prepare(
            "select id, project_id, name, modality, status, version, item_count, \
                 created_at, updated_at, file_path, character_id \
                 from training_datasets where project_id = ?1 \
                 order by updated_at desc, name asc",
        )
        .expect("dataset list query prepares against migrated schema");
}

/// Deterministic fingerprint of the full project.db schema a fresh
/// `apply_project_migrations` produces: the stamped `user_version` plus every
/// user table's columns (name/type/notnull/default/pk in declaration order)
/// and every user-defined index. Used by the schema-drift guard below.
fn project_db_schema_fingerprint(connection: &Connection) -> String {
    let mut lines = Vec::new();
    let version: i64 = connection
        .query_row("pragma user_version", [], |row| row.get(0))
        .expect("user_version");
    lines.push(format!("user_version={version}"));

    let mut table_names = connection
        .prepare(
            "select name from sqlite_master where type = 'table' \
                 and name not like 'sqlite_%' order by name",
        )
        .expect("table query");
    let tables: Vec<String> = table_names
        .query_map([], |row| row.get::<_, String>(0))
        .expect("tables")
        .filter_map(Result::ok)
        .collect();
    for table in &tables {
        let mut columns = connection
            .prepare(&format!("pragma table_info({table})"))
            .expect("table_info");
        let column_specs: Vec<String> = columns
            .query_map([], |row| {
                let name: String = row.get("name")?;
                let ctype: String = row.get("type")?;
                let notnull: i64 = row.get("notnull")?;
                let default: Option<String> = row.get("dflt_value")?;
                let pk: i64 = row.get("pk")?;
                Ok(format!(
                    "{name} {ctype} notnull={notnull} default={} pk={pk}",
                    default.as_deref().unwrap_or("NULL")
                ))
            })
            .expect("columns")
            .filter_map(Result::ok)
            .collect();
        lines.push(format!("table {table}: {}", column_specs.join(", ")));
    }

    let mut index_query = connection
        .prepare(
            "select name, tbl_name from sqlite_master where type = 'index' \
                 and sql is not null and name not like 'sqlite_%' order by name",
        )
        .expect("index query");
    let indexes: Vec<String> = index_query
        .query_map([], |row| {
            Ok(format!(
                "index {} on {}",
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?
            ))
        })
        .expect("indexes")
        .filter_map(Result::ok)
        .collect();
    lines.extend(indexes);

    lines.join("\n")
}

/// The expected project.db schema, version included. Regenerate ONLY
/// alongside a deliberate schema change — and when you do, you MUST bump
/// `PROJECT_SCHEMA_VERSION` so the `user_version=` line below changes too.
const EXPECTED_PROJECT_DB_SCHEMA: &str = concat!(
        "user_version=8\n",
        "table assets: id TEXT notnull=0 default=NULL pk=1, type TEXT notnull=1 default=NULL pk=0, display_name TEXT notnull=1 default=NULL pk=0, file_path TEXT notnull=1 default=NULL pk=0, generation_set_id TEXT notnull=0 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, favorite INTEGER notnull=1 default=0 pk=0, rating INTEGER notnull=1 default=0 pk=0, rejected INTEGER notnull=1 default=0 pk=0, trashed INTEGER notnull=1 default=0 pk=0, sidecar_path TEXT notnull=0 default=NULL pk=0, origin TEXT notnull=0 default=NULL pk=0, asset_json TEXT notnull=0 default=NULL pk=0, source_asset_id TEXT notnull=0 default=NULL pk=0, upscaled_from_asset_id TEXT notnull=0 default=NULL pk=0, sidecar_size INTEGER notnull=0 default=NULL pk=0, sidecar_modified_ns TEXT notnull=0 default=NULL pk=0, sidecar_sha256 TEXT notnull=0 default=NULL pk=0, poster_bytes BLOB notnull=0 default=NULL pk=0, poster_sha256 TEXT notnull=0 default=NULL pk=0\n",
        "table character_looks: id TEXT notnull=0 default=NULL pk=1, character_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, description TEXT notnull=1 default='' pk=0, approved_reference_ids TEXT notnull=1 default='[]' pk=0, recipe_settings TEXT notnull=1 default='{}' pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "table character_loras: id TEXT notnull=0 default=NULL pk=1, character_id TEXT notnull=1 default=NULL pk=0, lora_id TEXT notnull=0 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, source_path TEXT notnull=0 default=NULL pk=0, project_path TEXT notnull=0 default=NULL pk=0, copied_into_project INTEGER notnull=1 default=0 pk=0, category TEXT notnull=1 default='character' pk=0, scope TEXT notnull=1 default='project' pk=0, trigger_words TEXT notnull=1 default='[]' pk=0, default_weight REAL notnull=1 default=1.0 pk=0, compatibility TEXT notnull=1 default='{}' pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "table character_references: character_id TEXT notnull=1 default=NULL pk=1, asset_id TEXT notnull=1 default=NULL pk=2, approved INTEGER notnull=1 default=0 pk=0, role TEXT notnull=1 default='reference' pk=0, notes TEXT notnull=1 default='' pk=0, added_at TEXT notnull=1 default=NULL pk=0, approved_at TEXT notnull=0 default=NULL pk=0\n",
        "table characters: id TEXT notnull=0 default=NULL pk=1, project_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, type TEXT notnull=1 default=NULL pk=0, description TEXT notnull=1 default='' pk=0, sidecar_path TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0, archived INTEGER notnull=1 default=0 pk=0\n",
        "table generation_sets: id TEXT notnull=0 default=NULL pk=1, mode TEXT notnull=1 default=NULL pk=0, model TEXT notnull=1 default=NULL pk=0, prompt TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, job_id TEXT notnull=0 default=NULL pk=0\n",
        "table project_metadata: key TEXT notnull=0 default=NULL pk=1, value TEXT notnull=1 default=NULL pk=0\n",
        "table saved_voices: id TEXT notnull=0 default=NULL pk=1, project_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, reference_audio_asset_id TEXT notnull=1 default=NULL pk=0, embedding TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0\n",
        "table timelines: id TEXT notnull=0 default=NULL pk=1, name TEXT notnull=1 default=NULL pk=0, file_path TEXT notnull=1 default=NULL pk=0, aspect_ratio TEXT notnull=1 default=NULL pk=0, width INTEGER notnull=1 default=NULL pk=0, height INTEGER notnull=1 default=NULL pk=0, fps INTEGER notnull=1 default=NULL pk=0, duration REAL notnull=1 default=0 pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "table training_datasets: id TEXT notnull=0 default=NULL pk=1, project_id TEXT notnull=1 default=NULL pk=0, name TEXT notnull=1 default=NULL pk=0, modality TEXT notnull=1 default=NULL pk=0, status TEXT notnull=1 default=NULL pk=0, version INTEGER notnull=1 default=NULL pk=0, item_count INTEGER notnull=1 default=0 pk=0, character_id TEXT notnull=0 default=NULL pk=0, file_path TEXT notnull=1 default=NULL pk=0, created_at TEXT notnull=1 default=NULL pk=0, updated_at TEXT notnull=1 default=NULL pk=0\n",
        "index idx_assets_source_asset on assets\n",
        "index idx_assets_upscaled_from on assets\n",
        "index idx_training_datasets_project_updated on training_datasets",
    );

/// Guards the failure mode behind sc-2537: adding a column/table/index to the
/// migration without bumping `PROJECT_SCHEMA_VERSION` leaves existing DBs
/// (stamped at the old version) stuck behind the early-return gate, so they
/// never receive the change and crash on the next query. This pins the full
/// schema — including the stamped `user_version` — so any schema change makes
/// the test fail loudly and demand a matching version bump before it can pass.
#[test]
fn project_db_schema_matches_snapshot_and_forces_version_bump() {
    let connection = Connection::open_in_memory().expect("in-memory db");
    apply_project_migrations(&connection).expect("migration runs");
    let actual = project_db_schema_fingerprint(&connection);
    assert_eq!(
        actual, EXPECTED_PROJECT_DB_SCHEMA,
        "\n\nproject.db schema drift detected.\n\
             If this change is intentional you MUST:\n  \
             1. bump PROJECT_SCHEMA_VERSION (so existing DBs re-run the migration), and\n  \
             2. replace EXPECTED_PROJECT_DB_SCHEMA with the actual schema below.\n\
             Forgetting step 1 is exactly the sc-2537 bug.\n\n\
             ----- actual schema -----\n{actual}\n-------------------------\n"
    );
}

#[test]
fn completed_asset_index_backfill_retries_after_failure() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Retry").expect("project creates");
    let project_path = store.find_project_path(&project.id).expect("project path");
    let connection = connect_project_db(&project_path).expect("db opens");
    connection
        .execute(
            "delete from project_metadata where key = ?1",
            params![ASSET_INDEX_VERSION_KEY],
        )
        .expect("remove completion marker");
    let stamped: i64 = connection
        .query_row("pragma user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(stamped, PROJECT_SCHEMA_VERSION);
    drop(connection);

    let generation_sets = project_path.join("generation-sets");
    std::fs::remove_dir(&generation_sets).expect("empty generation set directory removes");
    std::fs::write(&generation_sets, b"not a directory").expect("failure fixture");
    assert!(
        ensure_project_db_ready(&project_path).is_err(),
        "reindex must fail while a required index source is unreadable"
    );
    let connection = connect_project_db(&project_path).unwrap();
    let marker: Option<String> = connection
        .query_row(
            "select value from project_metadata where key = ?1",
            params![ASSET_INDEX_VERSION_KEY],
            |row| row.get(0),
        )
        .optional()
        .unwrap();
    assert!(
        marker.is_none(),
        "failed rebuild must not be marked complete"
    );
    drop(connection);

    std::fs::remove_file(&generation_sets).unwrap();
    std::fs::create_dir(&generation_sets).unwrap();
    ensure_project_db_ready(&project_path).expect("next open retries and completes");
    let connection = connect_project_db(&project_path).unwrap();
    let marker: String = connection
        .query_row(
            "select value from project_metadata where key = ?1",
            params![ASSET_INDEX_VERSION_KEY],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(marker, PROJECT_SCHEMA_VERSION.to_string());
}

#[test]
fn upscale_lineage_query_uses_indexes_and_ignores_unrelated_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Lineage").expect("project creates");
    let project_path = store.find_project_path(&project.id).unwrap();
    let connection = connect_project_db(&project_path).unwrap();
    connection
        .execute_batch(
            "
                with recursive seq(n) as (
                  values(1) union all select n + 1 from seq where n < 5000
                )
                insert into assets (
                  id,type,display_name,file_path,created_at
                )
                select printf('unrelated-%05d', n), 'image', 'unrelated',
                       printf('assets/images/unrelated-%05d.png', n), ''
                  from seq;
                insert into assets (id,type,display_name,file_path,created_at)
                  values ('base','image','base','base.png','');
                insert into assets (
                  id,type,display_name,file_path,created_at,upscaled_from_asset_id
                ) values
                  ('child','image','child','child.png','','base'),
                  ('grandchild','image','grandchild','grandchild.png','','child');
                ",
        )
        .unwrap();
    let plan_sql = format!("explain query plan {UPSCALE_LINEAGE_QUERY}");
    let details: Vec<String> = connection
        .prepare(&plan_sql)
        .unwrap()
        .query_map(params!["grandchild"], |row| row.get::<_, String>(3))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let plan = details.join("\n");
    assert!(
        plan.contains("idx_assets_upscaled_from"),
        "child traversal must use the lineage index:\n{plan}"
    );
    assert!(
        plan.contains("sqlite_autoindex_assets_1"),
        "parent traversal must use the assets primary key:\n{plan}"
    );
    drop(connection);

    let group = upscale_lineage_group(&project_path, "grandchild").expect("lineage query");
    assert_eq!(group.first().map(String::as_str), Some("grandchild"));
    assert_eq!(group.len(), 3);
    assert!(group.contains(&"base".to_owned()));
    assert!(group.contains(&"child".to_owned()));
}

#[test]
fn normalize_asset_tags_trims_lowercases_and_deduplicates() {
    let tags = normalize_asset_tags(&[
        " Portrait ".to_owned(),
        "portrait".to_owned(),
        "Reference".to_owned(),
        "".to_owned(),
    ])
    .expect("normalized tags");

    assert_eq!(tags, vec!["portrait", "reference"]);
}

#[test]
fn build_generated_asset_sidecar_assembles_the_contract_schema() {
    // sc-1656: Rust owns the generated-asset sidecar schema now. Pin the
    // envelope it assembles from worker facts to the shared contract
    // (resource_sidecars.json imageAsset + recipe required keys) and the
    // load-bearing field mappings.
    let fact = json!({
        "assetId": "asset_abc",
        "mediaPath": "assets/images/genset_x/2026-05-25_z_image_turbo_city_0001.png",
        "mimeType": "image/png",
        "width": 1280,
        "height": 768,
        "normalizedWidth": 1024,
        "normalizedHeight": 768,
        "count": 1,
        "family": "z-image",
        "seed": 42,
        "index": 0,
        "displayName": "city #1",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "city",
        "negativePrompt": "",
        "loras": [],
        "stylePreset": "none",
        "characterId": "character-1",
        "characterLookId": "look-1",
        "sourceAssetId": Value::Null,
        "rawAdapterSettings": {"steps": 8},
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_x", &fact);

    for key in [
        "schemaVersion",
        "id",
        "projectId",
        "generationSetId",
        "type",
        "displayName",
        "createdAt",
        "file",
        "status",
        "recipe",
        "lineage",
    ] {
        assert!(asset.get(key).is_some(), "missing top-level key {key}");
    }
    for key in [
        "mode",
        "model",
        "adapter",
        "prompt",
        "negativePrompt",
        "seed",
        "loras",
        "normalizedSettings",
        "rawAdapterSettings",
    ] {
        assert!(
            asset["recipe"].get(key).is_some(),
            "missing recipe key {key}"
        );
    }
    assert_eq!(asset["type"], json!("image"));
    assert_eq!(asset["id"], json!("asset_abc"));
    assert_eq!(asset["projectId"], json!("project-1"));
    assert_eq!(asset["generationSetId"], json!("genset_x"));
    assert_eq!(asset["file"]["path"], fact["mediaPath"]);
    assert_eq!(asset["file"]["width"], json!(1280));
    assert_eq!(asset["recipe"]["adapter"], json!("z_image_diffusers"));
    assert_eq!(asset["recipe"]["seed"], json!(42));
    assert_eq!(asset["recipe"]["normalizedSettings"]["width"], json!(1024));
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["family"],
        json!("z-image")
    );
    // Character metadata propagates into normalizedSettings but conditioning
    // stays inactive (formerly pinned Python-side by
    // test_character_image_recipe_marks_conditioning_inactive).
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["characterId"],
        json!("character-1")
    );
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["characterLookId"],
        json!("look-1")
    );
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["characterConditioningActive"],
        json!(false)
    );
    assert_eq!(asset["lineage"]["jobId"], json!("job-1"));
}

#[test]
fn generated_workflow_intermediate_is_hidden_on_its_first_sidecar_write() {
    let mut fact = json!({
        "assetId": "asset_intermediate",
        "mediaPath": "assets/images/genset_workflow/intermediate.png",
        "mimeType": "image/png",
        "width": 1024,
        "height": 1024,
        "displayName": "workflow raster",
        "createdAt": "2026-08-29T00:00:00Z",
        "mode": "text_to_image",
        "model": "raster_model",
        "prompt": "mark",
        "vectorWorkflowOwnership": {
            "role": "retained_intermediate",
            "publication": "unpublished",
            "workflowId": "vwf_workflow1",
            "parentJobId": "job_parent1",
            "childJobId": "job_child1",
            "hidden": true
        }
    });
    let asset = build_generated_asset_sidecar("project-1", "job_child1", "genset_workflow", &fact);
    assert_eq!(asset["origin"], json!("vector_workflow_intermediate"));
    assert_eq!(
        asset["extra"]["vectorWorkflow"],
        fact["vectorWorkflowOwnership"]
    );

    fact["vectorWorkflowOwnership"]
        .as_object_mut()
        .expect("ownership object")
        .remove("publication");
    let untrusted =
        build_generated_asset_sidecar("project-1", "job_child1", "genset_workflow", &fact);
    assert_ne!(untrusted["origin"], json!("vector_workflow_intermediate"));
    assert!(untrusted.get("extra").is_none());
}

#[test]
fn build_generated_vector_sidecar_keeps_svg_source_and_png_preview_distinct() {
    let fact = json!({
        "assetId": "asset_vector",
        "type": "vector",
        "mediaPath": "assets/images/genset_v/asset_vector/vector.svg",
        "mimeType": "image/svg+xml",
        "width": 12,
        "height": 8,
        "createdAt": "2026-08-29T00:00:00Z",
        "mode": "image_to_svg",
        "model": "starvector_1b",
        "adapter": "starvector",
        "prompt": "preserve the bold silhouette",
        "sourceAssetId": "asset_source",
        "sampling": {
            "temperature": 0.2,
            "topP": 0.9,
            "topK": 0,
            "repetitionPenalty": 1.0,
            "repetitionContext": 0,
            "seed": 17,
        },
        "detailBudget": {
            "maxNewTokens": 4096,
            "maxSvgBytes": 262144,
            "maxWallTimeMs": 120000,
        },
        "sanitizerVersion": "sceneworks-inert-svg-v1",
        "rendererVersion": "resvg-0.45",
        "workflow": {
            "kind": "create_from_prompt",
            "id": "vwf_1",
            "parentJobId": "job-1",
            "childJobId": "job-raster",
            "intermediateAssetId": "asset_source",
            "rasterStage": {
                "model": "flux_schnell",
                "revision": "1111111111111111111111111111111111111111"
            },
            "vectorStage": {
                "model": "starvector_1b",
                "revision": "2222222222222222222222222222222222222222"
            }
        },
        "count": 1,
        "normalizedWidth": 12,
        "normalizedHeight": 8,
        "preview": {
            "path": "assets/images/genset_v/asset_vector/preview.png",
            "mimeType": "image/png",
            "width": 12,
            "height": 8,
        },
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_v", &fact);
    assert_eq!(asset["type"], "vector");
    assert_eq!(asset["file"]["path"], fact["mediaPath"]);
    assert_eq!(asset["file"]["mimeType"], "image/svg+xml");
    assert_eq!(asset["preview"]["mimeType"], "image/png");
    assert_ne!(asset["file"]["path"], asset["preview"]["path"]);
    assert_eq!(asset["recipe"]["mode"], "image_to_svg");
    assert_eq!(asset["recipe"]["model"], "starvector_1b");
    assert_eq!(asset["recipe"]["sampling"]["seed"], 17);
    assert_eq!(asset["recipe"]["detailBudget"]["maxNewTokens"], 4096);
    assert_eq!(
        asset["recipe"]["sanitizerVersion"],
        "sceneworks-inert-svg-v1"
    );
    assert_eq!(asset["recipe"]["rendererVersion"], "resvg-0.45");
    assert_eq!(
        asset["recipe"]["workflow"]["rasterStage"]["revision"],
        "1111111111111111111111111111111111111111"
    );
    assert_eq!(
        asset["recipe"]["workflow"]["vectorStage"]["revision"],
        "2222222222222222222222222222222222222222"
    );
    assert_eq!(asset["lineage"]["sourceAssetId"], "asset_source");
    assert_eq!(asset["lineage"]["parents"], json!(["asset_source"]));
    assert_eq!(asset["lineage"]["workflowId"], "vwf_1");
    assert_eq!(asset["lineage"]["intermediateAssetId"], "asset_source");

    let encoded = serde_json::to_vec(&asset).expect("vector sidecar serializes");
    let replayed: Value = serde_json::from_slice(&encoded).expect("vector sidecar parses");
    assert_eq!(replayed["recipe"], asset["recipe"]);
    assert_eq!(replayed["lineage"], asset["lineage"]);
}

/// sc-4408: a scored generation's `rawAdapterSettings.faceLikeness` block (attached worker-side by
/// `attach_face_likeness`) survives verbatim through `build_image_sidecar_parts` into
/// `recipe.rawAdapterSettings` — the persistence pathway the scoring surfaces (4409/4410/4411) rely
/// on. The omit-when-absent contract is covered too: no `faceLikeness` key in ⇒ none out.
#[test]
fn build_image_sidecar_carries_face_likeness_when_present_and_omits_when_absent() {
    let base_fact = |raw: Value| {
        json!({
            "assetId": "asset_fl",
            "mediaPath": "assets/images/genset_x/img_0001.png",
            "mimeType": "image/png",
            "width": 1024,
            "height": 1024,
            "mode": "character_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "portrait",
            "loras": [],
            "rawAdapterSettings": raw,
        })
    };

    // Present: the detected:true block (and a detected:false N/A block) passes through verbatim.
    let scored = json!({
        "faceLikeness": {
            "score": 0.873,
            "detected": true,
            "method": "arcface_antelopev2",
            "sourceAssetId": "asset_source_fixture",
        }
    });
    let asset =
        build_generated_asset_sidecar("project-1", "job-1", "genset_x", &base_fact(scored.clone()));
    assert_eq!(asset["recipe"]["rawAdapterSettings"], scored);
    assert_eq!(
        asset["recipe"]["rawAdapterSettings"]["faceLikeness"]["detected"],
        json!(true)
    );

    let na = json!({
        "faceLikeness": {
            "score": Value::Null,
            "detected": false,
            "method": "arcface_antelopev2",
            "sourceAssetId": "asset_source_fixture",
            "reason": "no_face",
        }
    });
    let asset_na =
        build_generated_asset_sidecar("project-1", "job-1", "genset_x", &base_fact(na.clone()));
    assert_eq!(asset_na["recipe"]["rawAdapterSettings"], na);

    // Absent: no faceLikeness key in rawAdapterSettings ⇒ none in the sidecar (clean, no crash).
    let asset_absent =
        build_generated_asset_sidecar("project-1", "job-1", "genset_x", &base_fact(json!({})));
    assert_eq!(asset_absent["recipe"]["rawAdapterSettings"], json!({}));
    assert!(
        asset_absent["recipe"]["rawAdapterSettings"]
            .get("faceLikeness")
            .is_none(),
        "absent score ⇒ faceLikeness omitted from sidecar"
    );
}

/// sc-5721 (CORE-002→F): `persist_generated_asset` joins the worker-supplied
/// `mediaPath` into the project tree and `create_dir_all`/`write`s a sidecar there,
/// so a traversal or absolute path must be rejected before any filesystem write —
/// the worker→API boundary must not write outside the project root. A normal
/// relative path is still accepted.
#[test]
fn persist_generated_asset_rejects_unsafe_media_path() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Boundary").expect("project creates");

    let fact_with = |media_path: &str| {
        json!({
            "assetId": "asset_safe1",
            "mediaPath": media_path,
            "mimeType": "image/png",
            "width": 64,
            "height": 64,
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "x",
            "loras": [],
        })
    };

    for unsafe_path in [
        "../../../../tmp/escape.png",
        "/etc/passwd",
        "assets/../../escape.png",
    ] {
        let result = store.persist_generated_asset(
            &project.id,
            "job-1",
            "genset_x",
            &fact_with(unsafe_path),
        );
        assert!(
            matches!(result, Err(ProjectStoreError::BadRequest(_))),
            "expected {unsafe_path:?} to be rejected, got {result:?}"
        );
    }

    // A normal project-relative path is still accepted and written under the project.
    let safe_fact = json!({
        "assetId": "asset_safe1",
        "mediaPath": "assets/images/genset_x/asset_safe1.png",
        "mimeType": "image/png",
        "width": 64,
        "height": 64,
        "normalizedWidth": 64,
        "normalizedHeight": 64,
        "count": 1,
        "family": "z-image",
        "seed": 1,
        "index": 0,
        "displayName": "safe #1",
        "createdAt": "2026-06-15T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "x",
        "negativePrompt": "",
        "loras": [],
        "stylePreset": "none",
        "rawAdapterSettings": {"steps": 8},
    });
    let safe = store
        .persist_generated_asset(&project.id, "job-1", "genset_x", &safe_fact)
        .expect("safe media path persists");
    assert_eq!(safe["id"], json!("asset_safe1"));
}

/// sc-25682: a depth asset persists its raw `.npy` as the `file`, keeps the derived preview
/// separate, carries the reload contract (`depth`) and reopenable settings, and is removed
/// whole — preview included — by purge.
#[test]
fn a_depth_asset_keeps_raw_npy_and_preview_apart_and_purges_both() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Depth").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let npy_rel = "assets/depth/genset_d/2026-10-11_depth_abcd.npy";
    let preview_rel = "assets/depth/genset_d/2026-10-11_depth_abcd.preview.png";
    std::fs::create_dir_all(project_path.join("assets/depth/genset_d")).unwrap();
    std::fs::write(project_path.join(npy_rel), b"\x93NUMPY").unwrap();
    std::fs::write(project_path.join(preview_rel), b"png").unwrap();
    let depth_block = json!({
        "valueConvention": "relative_log_depth",
        "dtype": "float32",
        "byteOrder": "little",
        "shape": [3, 4],
        "modelId": "iris_3b_depth",
        "configSha256": "c".repeat(64),
    });
    let fact = |preview_path: &str| {
        json!({
            "assetId": "asset_depth1",
            "type": "depth",
            "mediaPath": npy_rel,
            "mimeType": "application/x-npy",
            "width": 4,
            "height": 3,
            "displayName": "Photo (depth)",
            "createdAt": "2026-10-11T00:00:00Z",
            "mode": "image_depth",
            "model": "iris_3b_depth",
            "adapter": "mlx",
            "sourceAssetId": "asset_source",
            "parents": ["asset_source"],
            "rawAdapterSettings": { "depth": { "resolution": { "mode": "capped", "maxSide": 1024 } } },
            "preview": { "path": preview_path, "mimeType": "image/png", "width": 4, "height": 3 },
            "depth": depth_block,
        })
    };
    for unsafe_preview in ["../../escape.png", "/etc/passwd"] {
        assert!(
            matches!(
                store.persist_generated_asset(
                    &project.id,
                    "job-d",
                    "genset_d",
                    &fact(unsafe_preview)
                ),
                Err(ProjectStoreError::BadRequest(_))
            ),
            "{unsafe_preview} must be refused"
        );
    }
    store
        .persist_generated_asset(&project.id, "job-d", "genset_d", &fact(preview_rel))
        .expect("depth asset persists");
    let asset = store
        .get_asset(&project.id, "asset_depth1")
        .expect("reloads");
    assert_eq!(asset["type"], "depth");
    assert_eq!(asset["file"]["path"], npy_rel);
    assert_eq!(asset["file"]["mimeType"], "application/x-npy");
    assert_eq!(asset["preview"]["path"], preview_rel);
    assert_eq!(asset["depth"], depth_block);
    assert_eq!(asset["lineage"]["sourceAssetId"], "asset_source");
    assert_eq!(asset["lineage"]["parents"], json!(["asset_source"]));
    assert_eq!(asset["lineage"]["jobId"], "job-d");
    assert_eq!(
        asset["recipe"]["rawAdapterSettings"]["depth"]["resolution"]["maxSide"],
        1024
    );

    // An index rebuild re-reads sidecars from disk: the depth asset must survive it whole.
    store.reindex_project(&project.id).expect("reindexes");
    let reindexed = store
        .get_asset(&project.id, "asset_depth1")
        .expect("depth asset survives a reindex");
    assert_eq!(reindexed["type"], "depth");
    assert_eq!(reindexed["file"]["path"], npy_rel);
    assert_eq!(reindexed["preview"]["path"], preview_rel);
    assert_eq!(reindexed["depth"], depth_block);

    store
        .purge_asset(&project.id, "asset_depth1", true)
        .expect("purges");
    assert!(!project_path.join(npy_rel).exists(), "raw map removed");
    assert!(
        !project_path.join(preview_rel).exists(),
        "derived preview removed, not orphaned"
    );
}

/// sc-8871 (F-069): `save_timeline` derives the write path from the
/// client-supplied `id` (last 8 chars slugged into the filename) and
/// `relative_string`'s lexical strip lets an unnormalized `..`-bearing id
/// through, so — matching the asset/character/track/dataset guards — the id
/// must be charset-checked before any path is derived. A traversal or
/// separator-bearing id is rejected with `BadRequest` and writes nothing;
/// a normal id still saves under the project's `timelines/` dir.
#[test]
fn save_timeline_rejects_unsafe_id_before_deriving_path() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Boundary").expect("project creates");
    let timelines_dir = std::path::Path::new(&project.path).join("timelines");

    let timeline_with = |id: &str| {
        json!({
            "id": id,
            "projectId": project.id,
            "name": "My Timeline",
        })
    };

    for unsafe_id in [
        "../../../../tmp/escape",
        "timeline/../../escape",
        "a/b/c",
        "id.with.dots",
        "  ",
        "",
    ] {
        let result = store.save_timeline(&project.id, timeline_with(unsafe_id));
        assert!(
            matches!(result, Err(ProjectStoreError::BadRequest(_))),
            "expected id {unsafe_id:?} to be rejected, got {result:?}"
        );
    }

    // Nothing should have been written for any rejected id — no file can land
    // inside (or escape) the timelines dir.
    let timeline_files: Vec<_> = std::fs::read_dir(&timelines_dir)
        .map(|entries| entries.filter_map(Result::ok).collect())
        .unwrap_or_default();
    assert!(
        timeline_files.is_empty(),
        "rejected ids must not write any timeline file, found {timeline_files:?}"
    );
    // And the traversal target above the project root must not exist either.
    assert!(
        !temp_dir
            .path()
            .join("escape.sceneworks.timeline.json")
            .exists(),
        "traversal id must not write outside the project dir"
    );

    // A normal safe id still saves correctly under timelines/.
    let saved = store
        .save_timeline(&project.id, timeline_with("timeline_abc123"))
        .expect("safe timeline id persists");
    assert_eq!(saved["id"], json!("timeline_abc123"));
    let written: Vec<_> = std::fs::read_dir(&timelines_dir)
        .expect("timelines dir exists after save")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        })
        .collect();
    assert_eq!(
        written.len(),
        1,
        "exactly one timeline file should be written for the safe id, found {written:?}"
    );
}

#[test]
fn write_generation_set_embeds_replayable_recipe_from_first_asset_fact() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Fixture").expect("project creates");
    let generation_set = json!({
        "id": "genset_recipe",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "prompt": "city",
        "negativePrompt": "",
        "count": 1,
        "createdAt": "2026-05-25T00:00:00Z",
    });
    let fact = json!({
        "assetId": "asset_abc",
        "mediaPath": "assets/images/genset_recipe/city.png",
        "mimeType": "image/png",
        "width": 1280,
        "height": 768,
        "normalizedWidth": 1024,
        "normalizedHeight": 768,
        "count": 1,
        "family": "z-image",
        "seed": 42,
        "displayName": "city #1",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "city",
        "negativePrompt": "",
        "loras": [{"id": "style_lora", "weight": 0.8}],
        "stylePreset": "none",
        "rawAdapterSettings": {"steps": 8},
    });

    store
        .write_generation_set(&project.id, "job-1", &generation_set, Some(&fact))
        .expect("generation set writes");

    let written: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(&project.path).join("generation-sets/genset_recipe.json"),
        )
        .expect("generation set file exists"),
    )
    .expect("generation set json parses");
    assert_eq!(written["recipe"]["adapter"], json!("z_image_diffusers"));
    assert_eq!(written["recipe"]["seed"], json!(42));
    assert_eq!(written["recipe"]["loras"][0]["id"], json!("style_lora"));
    assert_eq!(
        written["recipe"]["normalizedSettings"]["width"],
        json!(1024)
    );
    assert_eq!(written["recipe"]["rawAdapterSettings"]["steps"], json!(8));
}

/// sc-13582 / F-001: the generation-set id is worker-reported and joined
/// into the generation-sets write path; a traversal id must be rejected
/// before any directory/file is created, mirroring the
/// persist_generated_asset guards (sc-5721/sc-4211).
#[test]
fn write_generation_set_rejects_traversal_id() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Fixture").expect("project creates");
    let generation_set = json!({
        "id": "../../escaped",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "prompt": "city",
        "count": 1,
        "createdAt": "2026-05-25T00:00:00Z",
    });

    let error = store
        .write_generation_set(&project.id, "job-1", &generation_set, None)
        .expect_err("traversal id is rejected");
    assert!(
        matches!(&error, ProjectStoreError::BadRequest(message) if message.contains("generation set id")),
        "expected invalid-id BadRequest, got {error:?}"
    );
    // The atomic-rename writer must never have produced the escaped file
    // (generation-sets/../../escaped.json resolves above the project root).
    let project_path = std::path::Path::new(&project.path);
    assert!(!project_path
        .join("generation-sets/../../escaped.json")
        .exists());
}

/// sc-13582 / F-001 (read half): a sidecar-supplied `generationSetId` is
/// joined into the generation-sets read path at list time; a tampered
/// traversal id must be skipped, not resolved to a file outside the
/// project and embedded into the API response.
#[test]
fn list_assets_skips_traversal_generation_set_id() {
    use crate::store_util::{read_json, write_json};
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Gallery").expect("project creates");
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "genset_ok",
            &json!({
                "assetId": "asset_one",
                "mediaPath": "assets/images/genset_ok/asset_one.png",
                "mimeType": "image/png",
                "displayName": "city #1",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "city",
            }),
        )
        .expect("asset persists");

    // Tamper the on-disk sidecar the way a compromised worker/file could,
    // and plant the traversal target outside the project root.
    let project_path = std::path::Path::new(&project.path);
    let sidecar_path = project_path.join("assets/images/genset_ok/asset_one.sceneworks.json");
    let mut sidecar = read_json(&sidecar_path).expect("sidecar reads");
    sidecar["generationSetId"] = json!("../../leak");
    write_json(&sidecar_path, &sidecar).expect("tampered sidecar writes");
    let leak_path = project_path.join("generation-sets/../../leak.json");
    std::fs::create_dir_all(project_path.join("generation-sets")).expect("sets dir");
    write_json(&leak_path, &json!({"secret": "outside-project"})).expect("leak plant writes");

    let assets = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("list");
    assert_eq!(assets.len(), 1);
    assert!(
        assets[0].get("generationSet").is_none(),
        "traversal generationSetId must not resolve/embed a file from outside the project: {:?}",
        assets[0].get("generationSet")
    );
}

/// sc-4270 / F-CORE-10: list_assets resolves the embedded `generationSet`
/// through a per-call cache (read once per set, not once per asset). Two
/// assets sharing a set must each still carry the full, correct set — proving
/// the cache doesn't drop or corrupt the shared value, and that dedup keeps
/// both distinct assets.
#[test]
fn list_assets_embeds_shared_generation_set_for_each_asset() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Gallery").expect("project creates");

    let generation_set = json!({
        "id": "genset_shared",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "prompt": "city",
        "count": 2,
        "createdAt": "2026-05-25T00:00:00Z",
    });
    let make_fact = |asset_id: &str, name: &str| {
        json!({
            "assetId": asset_id,
            "mediaPath": format!("assets/images/genset_shared/{asset_id}.png"),
            "mimeType": "image/png",
            "displayName": name,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "city",
        })
    };

    store
        .write_generation_set(
            &project.id,
            "job-1",
            &generation_set,
            Some(&make_fact("asset_one", "city #1")),
        )
        .expect("generation set writes");
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "genset_shared",
            &make_fact("asset_one", "city #1"),
        )
        .expect("asset one persists");
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "genset_shared",
            &make_fact("asset_two", "city #2"),
        )
        .expect("asset two persists");

    let assets = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("list");
    assert_eq!(assets.len(), 2, "both distinct assets are listed (deduped)");
    for asset in &assets {
        assert_eq!(
            asset["generationSet"]["id"],
            json!("genset_shared"),
            "each asset embeds the shared generation set"
        );
    }
}

#[test]
fn move_asset_to_library_promotes_origin_and_detaches_character() {
    use crate::store_util::{read_json, write_json};
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Gallery").expect("project creates");

    // A Character Studio output: persisted, then marked up to look like a
    // character-scoped asset (origin character_studio + recipe characterId +
    // characterReferences) — the state the Library excludes and the Character
    // Assets grid keys on.
    let fact = json!({
        "assetId": "char_shot",
        "mediaPath": "assets/images/genset_char/char_shot.png",
        "mimeType": "image/png",
        "displayName": "Mira test #1",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "character_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "portrait",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "genset_char", &fact)
        .expect("asset persists");

    let project_path = store.find_project_path(&project.id).expect("project path");
    let sidecar = project_path.join("assets/images/genset_char/char_shot.sceneworks.json");
    {
        let mut asset = read_json(&sidecar).expect("read sidecar");
        let object = asset.as_object_mut().expect("asset object");
        object.insert("origin".to_owned(), json!("character_studio"));
        let recipe = object
            .entry("recipe")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("recipe object");
        recipe
            .entry("normalizedSettings")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("normalizedSettings object")
            .insert("characterId".to_owned(), json!("char-1"));
        object
            .entry("metadata")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("metadata object")
            .insert(
                "characterReferences".to_owned(),
                json!([{ "characterId": "char-1" }]),
            );
        write_json(&sidecar, &asset).expect("write sidecar");
    }
    store
        .index_asset_sidecar(&project.id, &sidecar)
        .expect("reindex");

    // Precondition: excluded from the Library by origin.
    let before = store
        .get_asset(&project.id, "char_shot")
        .expect("get before");
    assert_eq!(before["origin"], json!("character_studio"));

    // sc-10205: the endpoint returns the moved group (requested asset first).
    let moved_group = store
        .move_asset_to_library(&project.id, "char_shot")
        .expect("move to library");
    let moved = &moved_group[0];

    // Origin promoted (image media -> image_studio), so the Library no longer excludes it.
    assert_eq!(moved["origin"], json!("image_studio"));
    // Character association dropped on both vectors the grid filters on.
    assert!(
        moved["recipe"]["normalizedSettings"]
            .get("characterId")
            .is_none(),
        "recipe characterId is cleared"
    );
    assert!(
        moved
            .get("metadata")
            .and_then(|metadata| metadata.get("characterReferences"))
            .map(Value::is_null)
            .unwrap_or(true),
        "characterReferences is cleared"
    );

    // Persisted: a fresh read reflects the move.
    let after = store
        .get_asset(&project.id, "char_shot")
        .expect("get after");
    assert_eq!(after["origin"], json!("image_studio"));
}

/// sc-10200: the true-move twin of `move_asset_to_library`. A Library asset
/// moved into a character must leave the Library (origin flip), anchor on the
/// target via `metadata.characterReferences` ONLY (no curated `references[]`
/// entry — the "Approved set" is hand-curated), detach from every other
/// character, and survive a curated reference add/remove cycle on the target.
#[test]
fn move_asset_to_character_flips_origin_and_anchors_without_curated_reference() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Gallery").expect("project creates");

    // A plain Image Studio output: origin image_studio, library-visible.
    let fact = json!({
        "assetId": "lib_shot",
        "mediaPath": "assets/images/genset_lib/lib_shot.png",
        "mimeType": "image/png",
        "displayName": "Plate #1",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "plate",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "genset_lib", &fact)
        .expect("asset persists");

    let mira = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Mira".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("Mira creates");
    let dax = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Dax".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("Dax creates");
    let mira_id = mira["id"].as_str().expect("Mira id").to_owned();
    let dax_id = dax["id"].as_str().expect("Dax id").to_owned();

    // Pre-existing curated link on Mira (the legacy "Move" behavior): the move
    // must clean this up rather than leave a dual membership behind.
    store
        .add_character_reference(
            &project.id,
            &mira_id,
            CharacterReferenceInput {
                asset_id: "lib_shot".to_owned(),
                approved: false,
                role: "asset".to_owned(),
                notes: String::new(),
            },
        )
        .expect("Mira link");

    // An unknown target must reject before mutating anything.
    assert!(store
        .move_asset_to_character(&project.id, "lib_shot", "char_missing")
        .is_err());

    // sc-10205: the endpoint returns the moved group (requested asset first).
    let moved_group = store
        .move_asset_to_character(&project.id, "lib_shot", &dax_id)
        .expect("move to Dax");
    let moved = &moved_group[0];

    // Origin flipped, so the Library allow-list now excludes the asset.
    assert_eq!(moved["origin"], json!("character_studio"));
    // Anchored on Dax via metadata only, tagged as a move (not a sidecar mirror).
    let links = moved["metadata"]["characterReferences"]
        .as_array()
        .expect("anchor links");
    assert_eq!(links.len(), 1);
    assert_eq!(links[0]["characterId"], json!(dax_id));
    assert_eq!(links[0]["source"], json!("library-move"));
    // Neither character carries a curated references[] entry: Mira's stale link
    // is gone and Dax's Approved set stays hand-curated.
    let mira_after = store
        .get_character(&project.id, &mira_id)
        .expect("Mira after");
    assert_eq!(mira_after["references"].as_array().map(Vec::len), Some(0));
    let dax_after = store
        .get_character(&project.id, &dax_id)
        .expect("Dax after");
    assert_eq!(dax_after["references"].as_array().map(Vec::len), Some(0));

    // A curated reference add/remove cycle on the target must not orphan the
    // asset: the sidecar mirror rebuild only manages its own entries.
    store
        .add_character_reference(
            &project.id,
            &dax_id,
            CharacterReferenceInput {
                asset_id: "lib_shot".to_owned(),
                approved: true,
                role: "reference".to_owned(),
                notes: String::new(),
            },
        )
        .expect("Dax curated link");
    store
        .remove_character_reference(&project.id, &dax_id, "lib_shot")
        .expect("Dax curated unlink");
    let cycled = store
        .get_asset(&project.id, "lib_shot")
        .expect("get cycled");
    let cycled_links = cycled["metadata"]["characterReferences"]
        .as_array()
        .expect("anchor survives");
    assert_eq!(cycled_links.len(), 1);
    assert_eq!(cycled_links[0]["source"], json!("library-move"));

    // Reversible: move-to-library restores a library origin and drops the anchor.
    let back_group = store
        .move_asset_to_library(&project.id, "lib_shot")
        .expect("move back");
    let back = &back_group[0];
    assert_eq!(back["origin"], json!("image_studio"));
    assert!(back["metadata"]["characterReferences"].is_null());
}

/// sc-10205: a linked original/upscaled pair renders as ONE folded Library
/// tile (the upscaled child is the visible asset), so moving the visible
/// asset must carry the whole fold group — in both directions. Splitting the
/// pair strands the hidden mate in the source collection.
#[test]
fn move_asset_carries_the_upscale_fold_group_both_directions() {
    use crate::store_util::{read_json, write_json};
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Gallery").expect("project creates");

    for (asset_id, path) in [
        ("base_shot", "assets/images/genset_pair/base_shot.png"),
        ("up_shot", "assets/images/genset_pair/base_shot_up2x.png"),
    ] {
        let fact = json!({
            "assetId": asset_id,
            "mediaPath": path,
            "mimeType": "image/png",
            "displayName": asset_id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "plate",
        });
        store
            .persist_generated_asset(&project.id, "job-1", "genset_pair", &fact)
            .expect("asset persists");
    }

    // Stamp the fold lineage the Library reads (sc-10117 keys) on the child.
    let project_path = store.find_project_path(&project.id).expect("project path");
    let child_sidecar =
        project_path.join("assets/images/genset_pair/base_shot_up2x.sceneworks.json");
    {
        let mut asset = read_json(&child_sidecar).expect("read child");
        super::apply_upscale_variant_link(&mut asset, "base_shot", 2, "seedvr2");
        write_json(&child_sidecar, &asset).expect("write child");
    }
    store
        .index_asset_sidecar(&project.id, &child_sidecar)
        .expect("reindex child");

    let character = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Mira".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("character creates");
    let character_id = character["id"].as_str().expect("character id").to_owned();

    // Move the VISIBLE folded asset (the upscaled child): both must move.
    let moved = store
        .move_asset_to_character(&project.id, "up_shot", &character_id)
        .expect("move pair to character");
    let moved = moved.as_array().expect("moved group");
    assert_eq!(moved.len(), 2, "the whole fold group moves");
    assert_eq!(moved[0]["id"], json!("up_shot"), "requested asset first");
    for asset in moved {
        assert_eq!(asset["origin"], json!("character_studio"));
        assert_eq!(
            asset["metadata"]["characterReferences"][0]["characterId"],
            json!(character_id)
        );
    }
    let base_after = store
        .get_asset(&project.id, "base_shot")
        .expect("base after");
    assert_eq!(
        base_after["origin"],
        json!("character_studio"),
        "the hidden fold-mate left the Library too"
    );

    // And back: moving the child to the Library brings the original along.
    let back = store
        .move_asset_to_library(&project.id, "up_shot")
        .expect("move pair back");
    let back = back.as_array().expect("back group");
    assert_eq!(back.len(), 2);
    for asset in back {
        assert_eq!(asset["origin"], json!("image_studio"));
    }
}

#[test]
fn injected_sqlite_reads_abort_or_record_asset_moves() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store
        .create_project("Move DB failure")
        .expect("project creates");
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "genset_move",
            &json!({
                "assetId": "move_me",
                "mediaPath": "assets/images/genset_move/move_me.png",
                "mimeType": "image/png",
                "displayName": "Move me",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "character_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "safe",
            }),
        )
        .expect("asset persists");
    let project_path = store.find_project_path(&project.id).expect("project path");
    let sidecar = project_path.join("assets/images/genset_move/move_me.sceneworks.json");
    let media = project_path.join("assets/images/genset_move/move_me.png");
    std::fs::write(&media, b"image").expect("media writes");

    fail_next_upscale_lineage_db_read();
    assert!(matches!(
        store.move_asset_to_library(&project.id, "move_me"),
        Err(ProjectStoreError::Sqlite(_))
    ));
    assert!(project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
    assert_eq!(
        read_json(&sidecar).expect("unmoved sidecar")["origin"],
        json!("character_studio"),
        "the group query failed before changing a partial move"
    );
    store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("dirty marker repairs after failed group query");

    fail_next_asset_poster_db_read();
    let error = store
        .move_asset_to_library(&project.id, "move_me")
        .expect_err("poster database failure propagates");
    assert!(matches!(error, ProjectStoreError::Sqlite(_)), "{error:?}");
    assert!(project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
    let repaired = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("dirty marker records and repairs the post-index failure");
    assert_eq!(repaired.len(), 1);
    assert_eq!(repaired[0]["origin"], json!("image_studio"));
    assert!(!project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
}

/// V-4: a pre-migration project surfaces an EMPTY `assets` table even though
/// its assets are still on disk as `.sceneworks.json` sidecars (these DBs
/// predate the asset index / `sidecar_path` column and were never reindexed).
/// `list_assets` must auto-reindex from the on-disk sidecars on first open so
/// the library is not silently empty. This reproduces that state by wiping the
/// index rows (leaving the sidecar on disk) and asserts the asset comes back.
#[test]
fn list_assets_auto_reindexes_pre_migration_project_with_sidecars_on_disk() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store
        .create_project("Pre-migration")
        .expect("project creates");

    let fact = json!({
        "assetId": "asset_legacy",
        "mediaPath": "assets/images/genset_legacy/asset_legacy.png",
        "mimeType": "image/png",
        "displayName": "legacy #1",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "old",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "genset_legacy", &fact)
        .expect("asset persists (media + sidecar + index row)");

    // Simulate the pre-migration state: the sidecar is on disk but the index
    // table is empty (as it would be for a DB created before the asset index
    // existed and never reindexed).
    let project_path = store.find_project_path(&project.id).expect("project path");
    {
        let connection = connect_project_db(&project_path).expect("open db");
        connection
            .execute("delete from assets", [])
            .expect("clear index");
        let remaining: i64 = connection
            .query_row("select count(*) from assets", [], |row| row.get(0))
            .expect("count");
        assert_eq!(remaining, 0, "precondition: index table is empty");
    }
    assert!(
        project_path
            .join("assets/images/genset_legacy/asset_legacy.sceneworks.json")
            .exists(),
        "precondition: the asset sidecar is still on disk"
    );

    // First open: list_assets must rebuild the index from the sidecar.
    let assets = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("list auto-reindexes instead of returning empty");
    assert_eq!(
        assets.len(),
        1,
        "the on-disk asset is recovered via auto-reindex, not silently dropped"
    );
    assert_eq!(assets[0]["id"], json!("asset_legacy"));

    // Idempotent: the index is now populated, so a second open does not depend
    // on (or re-run) the reindex and still returns the asset.
    let again = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("second list");
    assert_eq!(again.len(), 1, "result is stable on subsequent opens");
}

#[test]
fn unsupported_recipe_stays_visible_through_lookup_and_reindex_without_rewrite() {
    let temp = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp.path().join("data"), "test-version");
    let project = store.create_project("Future recipe history").unwrap();
    let fact = json!({
        "assetId":"historic", "mediaPath":"assets/images/history/shot_0_up2x.png",
        "mimeType":"image/png", "displayName":"Historic", "createdAt":"2026-05-25T00:00:00Z",
        "mode":"text_to_image", "model":"z_image_turbo", "prompt":"keep me"
    });
    store
        .persist_generated_asset(&project.id, "old-job", "history", &fact)
        .unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    // A real matching parent makes lineage backfill eligible; the unknown child must still
    // remain byte-identical rather than having its refusal annotation written into the sidecar.
    let mut base = fact.clone();
    base["assetId"] = json!("historic-base");
    base["mediaPath"] = json!("assets/images/history/shot_0.png");
    store
        .persist_generated_asset(&project.id, "old-job", "history", &base)
        .unwrap();
    fs::write(
        project_path.join("assets/images/history/shot_0.png"),
        b"historical base",
    )
    .unwrap();
    let media = project_path.join("assets/images/history/shot_0_up2x.png");
    fs::write(&media, b"historical media").unwrap();
    let sidecar = media.with_extension("sceneworks.json");
    let mut asset: Value = serde_json::from_slice(&fs::read(&sidecar).unwrap()).unwrap();
    asset["recipe"] = json!({"optionsSchemaVersion":9,"prompt":"keep me",
        "rawAdapterSettings":{"upscale":{"scale":2}}});
    asset["provenance"] = json!({"checkpoint":"legacy"});
    let original = serde_json::to_vec_pretty(&asset).unwrap();
    fs::write(&sidecar, &original).unwrap();
    let connection = connect_project_db(&project_path).unwrap();
    connection.execute("delete from assets", []).unwrap();
    // Exercise the real unindexed sidecar fallback, not direct hydration of a fabricated row.
    let found = store.get_asset(&project.id, "historic").unwrap();
    assert_eq!(
        found["optionsUpgradeError"]["code"],
        "UnsupportedSchemaVersion"
    );
    assert_eq!(found["recipe"], asset["recipe"]);
    // Library's empty-index recovery and explicit reindex must both retain the same asset.
    for explicit in [false, true] {
        if explicit {
            store.reindex_project(&project.id).unwrap();
        }
        let listed = store
            .list_assets(&project.id, false, false, AssetScope::All)
            .unwrap();
        assert_eq!(listed.len(), 2);
        let visible = listed
            .iter()
            .find(|asset| asset["id"] == "historic")
            .unwrap();
        assert_eq!(visible["id"], "historic");
        assert_eq!(visible["recipe"], asset["recipe"]);
        assert_eq!(visible["provenance"], asset["provenance"]);
        assert_eq!(
            visible["optionsUpgradeError"]["code"],
            "UnsupportedSchemaVersion"
        );
        assert_eq!(fs::read(&sidecar).unwrap(), original);
    }
    // Read-modify-write callers still refuse this carrier.
    assert!(matches!(
        store.update_asset_status(
            &project.id,
            "historic",
            AssetStatusPatch {
                favorite: Some(true),
                ..Default::default()
            }
        ),
        Err(ProjectStoreError::FieldInvalid {
            code: "UnsupportedSchemaVersion",
            ..
        })
    ));
    assert_eq!(fs::read(&sidecar).unwrap(), original);
}

#[test]
fn list_assets_rebuild_skips_a_corrupt_sidecar_and_stays_recovered() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store
        .create_project("Corrupt rebuild")
        .expect("project creates");
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "genset_ok",
            &json!({
                "assetId": "healthy",
                "mediaPath": "assets/images/genset_ok/healthy.png",
                "mimeType": "image/png",
                "displayName": "Healthy",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "safe",
            }),
        )
        .expect("healthy asset persists");
    let project_path = store.find_project_path(&project.id).expect("project path");
    std::fs::write(
        project_path.join("assets/images/genset_ok/corrupt.sceneworks.json"),
        b"{not valid json",
    )
    .expect("corrupt sidecar writes");
    connect_project_db(&project_path)
        .expect("open db")
        .execute("delete from assets", [])
        .expect("clear index to force automatic rebuild");

    let listed = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("corrupt sidecar does not disable listing");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["id"], json!("healthy"));
    let repeated = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("successful rebuild is recorded despite corrupt sidecar");
    assert_eq!(repeated.len(), 1);
    assert_eq!(repeated[0]["id"], json!("healthy"));
}

#[test]
fn list_assets_external_same_size_edit_requires_explicit_reindex() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Indexed").expect("project creates");
    let fact = json!({
        "assetId": "asset_cached",
        "mediaPath": "assets/images/set/asset_cached.png",
        "mimeType": "image/png",
        "displayName": "Before",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "cache",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "set", &fact)
        .expect("asset persists");
    let project_path = store.find_project_path(&project.id).unwrap();
    let sidecar = project_path.join("assets/images/set/asset_cached.sceneworks.json");
    std::fs::write(
        project_path.join("assets/images/set/asset_cached.png"),
        b"png",
    )
    .unwrap();
    let mut asset = read_json(&sidecar).unwrap();
    asset["displayName"] = json!("After!");
    let old_metadata = std::fs::metadata(&sidecar).unwrap();
    write_json(&sidecar, &asset).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&sidecar)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new()
                .set_accessed(old_metadata.accessed().unwrap())
                .set_modified(old_metadata.modified().unwrap()),
        )
        .unwrap();
    assert_eq!(
        std::fs::metadata(&sidecar).unwrap().len(),
        old_metadata.len(),
        "fixture is a same-size rewrite"
    );
    assert_eq!(
        std::fs::metadata(&sidecar).unwrap().modified().unwrap(),
        old_metadata.modified().unwrap(),
        "fixture restores the original modification timestamp"
    );

    let (listed, operations) = store
        .list_assets_with_filesystem_operations(&project.id, false, false, AssetScope::All)
        .expect("clean indexed list");
    assert_eq!(
        listed[0]["displayName"],
        json!("Before"),
        "out-of-band edits do not bypass the explicit reindex contract"
    );
    assert_eq!(operations.sidecar_reads, 0);

    store
        .reindex_project(&project.id)
        .expect("explicit reindex");
    let listed = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("list after reindex");
    assert_eq!(
        listed[0]["displayName"],
        json!("After!"),
        "reindex reads content, so timestamps are irrelevant to reconciliation"
    );
}

#[test]
fn reindex_uses_authoritative_project_id_for_asset_urls() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Imported sidecar").unwrap();
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "set",
            &json!({
                "assetId": "asset_imported",
                "mediaPath": "assets/images/set/asset_imported.png",
                "mimeType": "image/png",
                "displayName": "Imported",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "benchmark",
                "adapter": "benchmark",
                "prompt": "imported",
            }),
        )
        .unwrap();

    let project_path = store.find_project_path(&project.id).unwrap();
    std::fs::write(
        project_path.join("assets/images/set/asset_imported.png"),
        b"png",
    )
    .unwrap();
    let sidecar = project_path.join("assets/images/set/asset_imported.sceneworks.json");
    let mut asset = read_json(&sidecar).unwrap();
    asset["projectId"] = json!("project_from_another_machine");
    write_json(&sidecar, &asset).unwrap();

    store.reindex_project(&project.id).unwrap();
    let listed = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(
        listed[0]["url"],
        json!(format!(
            "/api/v1/projects/{}/files/assets/images/set/asset_imported.png",
            project.id
        )),
        "mutable imported sidecar metadata must not select the URL project"
    );
    assert_eq!(
        listed[0]["projectId"],
        json!(project.id),
        "the indexed envelope also reports its authoritative current project"
    );
}

#[test]
fn list_assets_external_generation_set_edit_requires_explicit_reindex() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Indexed generation set").unwrap();
    store
        .write_generation_set(
            &project.id,
            "job-1",
            &json!({
                "id": "set",
                "mode": "text_to_image",
                "model": "benchmark",
                "prompt": "Before",
                "createdAt": "2026-05-25T00:00:00Z",
            }),
            None,
        )
        .unwrap();
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "set",
            &json!({
                "assetId": "asset_cached",
                "mediaPath": "assets/images/set/asset_cached.png",
                "mimeType": "image/png",
                "displayName": "Cached",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "benchmark",
                "adapter": "benchmark",
                "prompt": "Before",
            }),
        )
        .unwrap();

    let project_path = store.find_project_path(&project.id).unwrap();
    std::fs::write(
        project_path.join("assets/images/set/asset_cached.png"),
        b"png",
    )
    .unwrap();
    let generation_set_path = project_path.join("generation-sets/set.json");
    let mut generation_set = read_json(&generation_set_path).unwrap();
    generation_set["prompt"] = json!("After!");
    write_json(&generation_set_path, &generation_set).unwrap();

    let (listed, operations) = store
        .list_assets_with_filesystem_operations(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(listed[0]["generationSet"]["prompt"], json!("Before"));
    assert_eq!(
        operations.generation_set_reads, 0,
        "clean listing serves the indexed generation set"
    );

    store.reindex_project(&project.id).unwrap();
    let listed = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(
        listed[0]["generationSet"]["prompt"],
        json!("After!"),
        "explicit reindex refreshes externally edited generation-set JSON"
    );
}

#[test]
fn write_generation_set_refreshes_indexed_assets_before_returning() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Generation set mutation").unwrap();
    let generation_set = |prompt: &str| {
        json!({
            "id": "set",
            "mode": "text_to_image",
            "model": "benchmark",
            "prompt": prompt,
            "createdAt": "2026-05-25T00:00:00Z",
        })
    };
    store
        .write_generation_set(&project.id, "job-1", &generation_set("Before"), None)
        .unwrap();
    store
        .persist_generated_asset(
            &project.id,
            "job-1",
            "set",
            &json!({
                "assetId": "asset_cached",
                "mediaPath": "assets/images/set/asset_cached.png",
                "mimeType": "image/png",
                "displayName": "Cached",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "text_to_image",
                "model": "benchmark",
                "adapter": "benchmark",
                "prompt": "Before",
            }),
        )
        .unwrap();

    store
        .write_generation_set(&project.id, "job-1", &generation_set("After!"), None)
        .unwrap();
    let (listed, operations) = store
        .list_assets_with_filesystem_operations(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(listed[0]["generationSet"]["prompt"], json!("After!"));
    assert_eq!(operations.generation_set_reads, 0);
}

#[test]
fn list_assets_many_clean_rows_has_constant_filesystem_operations() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Many indexed assets").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let mut one_row_operations = None;
    for index in 0..128 {
        let asset_id = format!("asset_{index:03}");
        let generation_set_id = format!("set_{index:03}");
        store
            .write_generation_set(
                &project.id,
                "job-1",
                &json!({
                    "id": generation_set_id,
                    "mode": "image_to_video",
                    "model": "benchmark",
                    "prompt": "indexed",
                    "createdAt": "2026-05-25T00:00:00Z",
                }),
                None,
            )
            .unwrap();
        let media_rel = format!("assets/videos/{generation_set_id}/{asset_id}.mp4");
        let poster_path = project_path.join(&media_rel).with_extension("poster.jpg");
        std::fs::create_dir_all(poster_path.parent().unwrap()).unwrap();
        std::fs::write(
            &poster_path,
            jpeg_fixture([index as u8, (index * 2) as u8, (index * 3) as u8]),
        )
        .unwrap();
        let fact = json!({
            "assetId": asset_id,
            "mediaPath": media_rel,
            "mimeType": "video/mp4",
            "type": "video",
            "displayName": format!("Asset {index:03}"),
            "createdAt": format!("2026-05-25T00:{:02}:{:02}Z", index / 60, index % 60),
            "mode": "image_to_video",
            "model": "benchmark",
            "adapter": "benchmark",
            "prompt": "indexed",
        });
        store
            .persist_generated_asset(&project.id, "job-1", &generation_set_id, &fact)
            .unwrap();
        if index == 0 {
            one_row_operations = Some(
                store
                    .list_assets_with_filesystem_operations(
                        &project.id,
                        false,
                        false,
                        AssetScope::All,
                    )
                    .unwrap()
                    .1,
            );
        }
    }

    let one_row_operations = one_row_operations.unwrap();
    let (assets, operations) = store
        .list_assets_with_filesystem_operations(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(assets.len(), 128);
    assert_eq!(
        operations.sidecar_reads, 0,
        "clean indexed listing must not read or hash one sidecar per row"
    );
    assert_eq!(operations.generation_set_reads, 0);
    assert_eq!(operations.poster_stats, 0);
    assert_eq!(operations.poster_reads, 0);
    assert_eq!(operations.db_opens, 1);
    assert_eq!(operations.registry_opens, 1);
    assert_eq!(operations.registry_metadata_reads, 1);
    assert!(operations.path_stats >= 1);
    assert_eq!(operations.directory_scans, 0);
    assert_eq!(operations.index_marker_reads, 1);
    assert_eq!(operations.index_marker_writes, 0);
    assert_eq!(operations.index_marker_removes, 0);
    assert_eq!(
        operations.total(),
        one_row_operations.total(),
        "clean-list filesystem work must be independent of returned row count"
    );
    let list_source = include_str!("project_store/asset_queries.rs")
        .split_once("fn list_assets_filtered(")
        .unwrap()
        .1
        .split_once("pub fn get_asset(")
        .unwrap()
        .0;
    assert!(
        !list_source.contains("read_dir") && !list_source.contains("file_type"),
        "the clean-list implementation must not hide poster enumeration behind one counter"
    );
}

#[test]
fn list_assets_skips_corrupt_indexed_envelope_without_hiding_healthy_rows() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Corrupt envelope").unwrap();
    for (index, asset_id) in [
        "healthy",
        "invalid_json",
        "impersonator",
        "unsafe_path",
        "missing_field",
    ]
    .into_iter()
    .enumerate()
    {
        let fact = json!({
            "assetId": asset_id,
            "mediaPath": format!("assets/images/set/{asset_id}.png"),
            "mimeType": "image/png",
            "displayName": asset_id,
            "createdAt": format!("2026-05-25T00:00:0{index}Z"),
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "cache",
        });
        store
            .persist_generated_asset(&project.id, "job-1", "set", &fact)
            .unwrap();
    }
    let project_path = store.find_project_path(&project.id).unwrap();
    let connection = connect_project_db(&project_path).unwrap();
    let healthy_json: String = connection
        .query_row(
            "select asset_json from assets where id = 'healthy'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    connection
        .execute(
            "update assets set asset_json = '{not-json' where id = 'invalid_json'",
            [],
        )
        .unwrap();
    connection
        .execute(
            "update assets set asset_json = ?1 where id = 'impersonator'",
            params![healthy_json],
        )
        .unwrap();
    let unsafe_path = json!({
        "id": "unsafe_path",
        "type": "image",
        "displayName": "Unsafe",
        "createdAt": "2026-05-25T00:00:03Z",
        "file": {"path": "../outside.png"}
    });
    connection
        .execute(
            "update assets set asset_json = ?1 where id = 'unsafe_path'",
            params![serde_json::to_string(&unsafe_path).unwrap()],
        )
        .unwrap();
    let missing_field = json!({
        "id": "missing_field",
        "type": "image",
        "createdAt": "2026-05-25T00:00:04Z",
        "file": {"path": "assets/images/set/missing_field.png"}
    });
    connection
        .execute(
            "update assets set asset_json = ?1 where id = 'missing_field'",
            params![serde_json::to_string(&missing_field).unwrap()],
        )
        .unwrap();
    drop(connection);

    let (assets, operations) = store
        .list_assets_with_filesystem_operations(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(assets.len(), 1);
    assert_eq!(assets[0]["id"], json!("healthy"));
    assert_eq!(
        operations.sidecar_reads, 0,
        "invalid JSON and parseable structural corruption fail closed without sidecar IO"
    );
}

#[test]
fn store_mutation_updates_indexed_envelope_before_returning() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Immediate mutation").unwrap();
    let fact = json!({
        "assetId": "asset_mutated",
        "mediaPath": "assets/images/set/asset_mutated.png",
        "mimeType": "image/png",
        "displayName": "Mutated",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "cache",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "set", &fact)
        .unwrap();
    store
        .update_asset_status(
            &project.id,
            "asset_mutated",
            AssetStatusPatch {
                favorite: Some(true),
                rating: Some(4),
                rejected: Some(true),
                trashed: None,
            },
        )
        .unwrap();

    let (assets, operations) = store
        .list_assets_with_filesystem_operations(&project.id, true, false, AssetScope::All)
        .unwrap();
    assert_eq!(assets[0]["status"]["favorite"], json!(true));
    assert_eq!(assets[0]["status"]["rating"], json!(4));
    assert_eq!(assets[0]["status"]["rejected"], json!(true));
    assert_eq!(operations.sidecar_reads, 0);
}

#[test]
fn dirty_marker_repairs_create_update_trash_and_purge_after_db_failure() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let fact = |asset_id: &str| {
        json!({
            "assetId": asset_id,
            "mediaPath": format!("assets/images/set/{asset_id}.png"),
            "mimeType": "image/png",
            "displayName": asset_id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "repair",
        })
    };
    let marker_exists = |project_path: &Path| project_path.join(ASSET_INDEX_DIRTY_MARKER).exists();

    // Create: the sidecar commits, the injected DB failure leaves the marker,
    // and the next list rebuilds the previously absent row.
    let created = store.create_project("Create repair").unwrap();
    let created_path = store.find_project_path(&created.id).unwrap();
    fail_next_asset_index_db_mutation();
    assert!(store
        .persist_generated_asset(&created.id, "job-1", "set", &fact("created"))
        .is_err());
    std::fs::write(created_path.join("assets/images/set/created.png"), b"png").unwrap();
    assert!(marker_exists(&created_path));
    let assets = store
        .list_assets(&created.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(assets[0]["id"], json!("created"));
    assert!(!marker_exists(&created_path));

    // Update: the sidecar carries the new status even though its first DB
    // upsert failed; repair makes it immediately authoritative.
    let updated = store.create_project("Update repair").unwrap();
    let updated_path = store.find_project_path(&updated.id).unwrap();
    store
        .persist_generated_asset(&updated.id, "job-1", "set", &fact("updated"))
        .unwrap();
    std::fs::write(updated_path.join("assets/images/set/updated.png"), b"png").unwrap();
    fail_next_asset_index_db_mutation();
    assert!(store
        .update_asset_status(
            &updated.id,
            "updated",
            AssetStatusPatch {
                favorite: Some(true),
                ..AssetStatusPatch::default()
            },
        )
        .is_err());
    assert!(marker_exists(&updated_path));
    let assets = store
        .list_assets(&updated.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(assets[0]["status"]["favorite"], json!(true));
    assert!(!marker_exists(&updated_path));

    // Trash: media and sidecar have moved before the injected DB failure.
    // Reindex follows the moved sidecar and replaces the old live row.
    let trashed = store.create_project("Trash repair").unwrap();
    let trashed_path = store.find_project_path(&trashed.id).unwrap();
    store
        .persist_generated_asset(&trashed.id, "job-1", "set", &fact("trashed"))
        .unwrap();
    std::fs::write(trashed_path.join("assets/images/set/trashed.png"), b"png").unwrap();
    fail_next_asset_index_db_mutation();
    assert!(store.delete_asset(&trashed.id, "trashed").is_err());
    assert!(marker_exists(&trashed_path));
    let assets = store
        .list_assets(&trashed.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(assets[0]["status"]["trashed"], json!(true));
    assert_eq!(
        assets[0]["file"]["path"],
        json!("trash/trashed/trashed.png")
    );
    assert!(!marker_exists(&trashed_path));

    // Purge: files are gone before the injected delete failure. Repair
    // rebuilds the table from the now-empty sidecar set, removing the ghost.
    let purged = store.create_project("Purge repair").unwrap();
    let purged_path = store.find_project_path(&purged.id).unwrap();
    store
        .persist_generated_asset(&purged.id, "job-1", "set", &fact("purged"))
        .unwrap();
    std::fs::write(purged_path.join("assets/images/set/purged.png"), b"png").unwrap();
    fail_next_asset_index_db_mutation();
    assert!(store.purge_asset(&purged.id, "purged", true).is_err());
    assert!(marker_exists(&purged_path));
    assert!(store
        .list_assets(&purged.id, true, true, AssetScope::All)
        .unwrap()
        .is_empty());
    assert!(!marker_exists(&purged_path));
}

#[test]
fn concurrent_list_waits_for_mutation_owner_before_repairing_marker() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    let project = store.create_project("Serialized repair").unwrap();
    let fact = json!({
        "assetId": "serialized",
        "mediaPath": "assets/images/set/serialized.png",
        "mimeType": "image/png",
        "displayName": "Before",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "repair",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "set", &fact)
        .unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    std::fs::write(
        project_path.join("assets/images/set/serialized.png"),
        b"png",
    )
    .unwrap();
    let sidecar_path = project_path.join("assets/images/set/serialized.sceneworks.json");

    // Simulate a process interruption after the sidecar write: the guard
    // owns the marker while a concurrent list reaches the repair boundary.
    let mutation = AssetIndexMutation::begin(&project_path).unwrap();
    let mut asset = read_json(&sidecar_path).unwrap();
    asset["displayName"] = json!("After");
    write_json(&sidecar_path, &asset).unwrap();

    let barrier = Arc::new(std::sync::Barrier::new(2));
    install_ensure_ready_before_lock_hook(project_path.clone(), Arc::clone(&barrier));
    let (sender, receiver) = std::sync::mpsc::channel();
    let project_id = project.id.clone();
    let list_thread = std::thread::spawn(move || {
        let reader = ProjectStore::new(data_dir, "test-version");
        sender
            .send(reader.list_assets(&project_id, true, true, AssetScope::All))
            .unwrap();
    });
    barrier.wait();
    assert!(
        project_path.join(ASSET_INDEX_DIRTY_MARKER).exists(),
        "the waiting list cannot clear the active owner's marker"
    );
    assert!(matches!(
        receiver.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));

    drop(mutation);
    let listed = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("list resumes after mutation guard release")
        .expect("repair succeeds");
    list_thread.join().unwrap();
    assert_eq!(listed[0]["displayName"], json!("After"));
    assert!(!project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
}

#[test]
fn native_asset_persistence_repairs_after_index_failure() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Native persistence repair").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let fact = json!({
        "assetId": "native_asset",
        "mediaPath": "assets/frames/native_asset.png",
        "mimeType": "image/png",
        "displayName": "Native asset",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "frame_extract",
        "model": "ffmpeg",
        "adapter": "ffmpeg",
        "prompt": "native",
    });
    let asset = build_generated_asset_sidecar(&project.id, "job-1", "native-set", &fact);
    let media_path = project_path.join("assets/frames/native_asset.png");
    std::fs::create_dir_all(media_path.parent().unwrap()).unwrap();
    std::fs::write(&media_path, b"png").unwrap();
    let sidecar_path = media_path.with_extension("sceneworks.json");

    fail_next_asset_index_db_mutation();
    assert!(store
        .persist_native_asset_sidecar(&project.id, &sidecar_path, asset)
        .is_err());
    assert!(sidecar_path.exists());
    assert!(project_path
        .join("recipes/native_asset.recipe.json")
        .exists());
    assert!(project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());

    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(listed[0]["id"], json!("native_asset"));
    assert!(!project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
}

#[test]
fn purge_sidecar_remove_failure_is_pruned_before_marker_clear() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Partial purge repair").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let fact = json!({
        "assetId": "partial_purge",
        "mediaPath": "assets/images/set/partial_purge.png",
        "mimeType": "image/png",
        "displayName": "Partial purge",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "repair",
    });
    store
        .persist_generated_asset(&project.id, "job-1", "set", &fact)
        .unwrap();
    let media_path = project_path.join("assets/images/set/partial_purge.png");
    std::fs::write(&media_path, b"png").unwrap();
    let sidecar_path = media_path.with_extension("sceneworks.json");

    fail_next_asset_sidecar_remove();
    assert!(store
        .purge_asset(&project.id, "partial_purge", true)
        .is_err());
    assert!(!media_path.exists());
    assert!(sidecar_path.exists());
    assert!(project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());

    assert!(store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap()
        .is_empty());
    assert!(!sidecar_path.exists());
    assert!(project_path
        .join(ORPHANED_SIDECAR_DIR)
        .join("partial_purge.sceneworks.json")
        .exists());
    assert!(!project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
}

#[test]
fn indexed_listing_preserves_order_filters_and_library_scope() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Selection").unwrap();
    for (asset_id, created_at, mode) in [
        ("older", "2026-05-25T00:00:01Z", "text_to_image"),
        ("rejected", "2026-05-25T00:00:02Z", "text_to_image"),
        ("character", "2026-05-25T00:00:03Z", "character_image"),
    ] {
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "set",
                &json!({
                    "assetId": asset_id,
                    "mediaPath": format!("assets/images/set/{asset_id}.png"),
                    "mimeType": "image/png",
                    "displayName": asset_id,
                    "createdAt": created_at,
                    "mode": mode,
                    "model": "z_image_turbo",
                    "adapter": "z_image_diffusers",
                    "prompt": "selection",
                }),
            )
            .unwrap();
    }
    store
        .update_asset_status(
            &project.id,
            "rejected",
            AssetStatusPatch {
                rejected: Some(true),
                ..AssetStatusPatch::default()
            },
        )
        .unwrap();

    let visible = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .unwrap();
    assert_eq!(
        visible
            .iter()
            .map(|asset| asset["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["character", "older"],
        "created-at descending order and rejected filtering are DB-backed"
    );
    let including_rejected = store
        .list_assets(&project.id, true, false, AssetScope::All)
        .unwrap();
    assert_eq!(
        including_rejected
            .iter()
            .map(|asset| asset["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["character", "rejected", "older"]
    );
    let library = store
        .list_assets(&project.id, true, false, AssetScope::Library)
        .unwrap();
    assert_eq!(
        library
            .iter()
            .map(|asset| asset["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["rejected", "older"],
        "Library scope excludes character-studio outputs"
    );
}

/// V-4 guard: a genuinely empty project (no sidecars on disk) must NOT trigger
/// a reindex and must return an empty list cleanly rather than erroring.
#[test]
fn list_assets_returns_empty_for_truly_empty_project_without_reindex_loop() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Empty").expect("project creates");

    let assets = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("empty project lists cleanly");
    assert!(
        assets.is_empty(),
        "no sidecars on disk => empty library, no spurious assets"
    );
}

/// Fact for a generated image asset in `genset`. `persist_generated_asset`
/// writes the sidecar + index row but NOT the media file (the worker writes
/// media), so a test controls "media present" by writing the `.png` itself.
fn orphan_test_fact(asset_id: &str) -> Value {
    json!({
        "assetId": asset_id,
        "mediaPath": format!("assets/images/genset/{asset_id}.png"),
        "mimeType": "image/png",
        "displayName": asset_id,
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "adapter": "z_image_diffusers",
        "prompt": "x",
    })
}

#[test]
fn prune_orphaned_assets_removes_only_rows_whose_media_is_missing() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Prune").expect("project creates");
    let project_path = store.find_project_path(&project.id).expect("project path");

    store
        .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("keep"))
        .expect("keep persists");
    store
        .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("gone"))
        .expect("gone persists");
    // `keep` has its media on disk; `gone` does not (its media was purged).
    std::fs::write(
        project_path.join("assets/images/genset/keep.png"),
        b"png-bytes",
    )
    .expect("write keep media");

    // Precondition: both rows are indexed and listed (list_assets checks the
    // sidecar, not the media, so it still serves the media-less `gone`).
    let before = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("list before prune");
    assert_eq!(before.len(), 2, "both assets listed before prune");

    let pruned = store
        .prune_orphaned_assets(&project.id)
        .expect("prune succeeds");
    assert_eq!(pruned, 1, "only the media-less row is pruned");

    let after = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("list after prune");
    assert_eq!(after.len(), 1, "the purged asset is no longer served");
    assert_eq!(after[0]["id"], json!("keep"));

    // Idempotent: a second pass finds nothing to prune.
    assert_eq!(
        store
            .prune_orphaned_assets(&project.id)
            .expect("second prune"),
        0
    );

    // The orphaned sidecar is moved OUT of the scanned asset tree into the
    // `.orphaned-sidecars/` quarantine (declutter + no resurrection), but retained
    // there (moved, not deleted) so the recipe metadata stays recoverable.
    assert!(
        !project_path
            .join("assets/images/genset/gone.sceneworks.json")
            .exists(),
        "orphaned sidecar is removed from the scanned asset folder"
    );
    assert!(
        project_path
            .join(".orphaned-sidecars/gone.sceneworks.json")
            .exists(),
        "orphaned sidecar is retained in the quarantine"
    );
}

#[test]
fn prune_all_orphaned_assets_sweeps_every_registered_project() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    // Two projects, each with one asset whose media was never written (purged).
    for name in ["Alpha", "Beta"] {
        let project = store.create_project(name).expect("project creates");
        store
            .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("gone"))
            .expect("gone persists");
    }

    let total = store
        .prune_all_orphaned_assets()
        .expect("sweep succeeds across projects");
    assert_eq!(total, 2, "one orphan pruned from each project");

    // A second sweep is a no-op.
    assert_eq!(store.prune_all_orphaned_assets().expect("second sweep"), 0);
}

#[test]
fn prune_quarantines_sidecar_so_empty_table_reindex_does_not_resurrect() {
    // A project whose ONLY asset is orphaned: after pruning, the assets table is
    // empty, so `list_assets`' auto-reindex (fires only when the table is empty)
    // would rebuild the row from the sidecar — UNLESS the sidecar has been moved
    // out of the scanned tree. This is the resurrection edge sc-10467 closes.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("AllOrphans").expect("project creates");
    store
        .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("gone"))
        .expect("gone persists");

    assert_eq!(
        store.prune_orphaned_assets(&project.id).expect("prune"),
        1,
        "the sole (media-less) asset is pruned, emptying the table"
    );

    // The empty table would trigger auto-reindex here; the quarantined sidecar is
    // no longer scannable, so nothing is rebuilt.
    let assets = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("list after prune");
    assert!(
        assets.is_empty(),
        "orphan is not resurrected via the empty-table auto-reindex"
    );
}

#[test]
fn reindex_project_does_not_resurrect_orphaned_assets() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Reindex").expect("project creates");
    let project_path = store.find_project_path(&project.id).expect("project path");

    store
        .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("present"))
        .expect("present persists");
    store
        .persist_generated_asset(&project.id, "job-1", "genset", &orphan_test_fact("orphan"))
        .expect("orphan persists");
    // Only `present` has media on disk; both sidecars exist (so a rebuild would
    // re-add both rows without the post-reindex prune).
    std::fs::write(
        project_path.join("assets/images/genset/present.png"),
        b"png-bytes",
    )
    .expect("write present media");

    let counts = store.reindex_project(&project.id).expect("reindex works");
    assert_eq!(
        counts.assets, 1,
        "reindex re-adds both sidecars then prunes the media-less one"
    );

    let assets = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("list after reindex");
    assert_eq!(assets.len(), 1, "the orphan is not resurrected by reindex");
    assert_eq!(assets[0]["id"], json!("present"));
}

#[test]
fn list_assets_advertises_poster_url_only_when_the_poster_exists() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Videos").expect("project creates");
    let project_path = store.find_project_path(&project.id).expect("project path");

    let video_fact = |id: &str| {
        json!({
            "assetId": id,
            "mediaPath": format!("assets/videos/genset/{id}.mp4"),
            "mimeType": "video/mp4",
            "type": "video",
            "displayName": id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "image_to_video",
            "model": "wan",
            "adapter": "wan",
            "prompt": "x",
        })
    };
    // The worker writes the media and best-effort poster before reporting the
    // completed asset fact. Only `withposter` has that sibling when indexed.
    let poster_path = project_path.join("assets/videos/genset/withposter.poster.jpg");
    let poster_bytes = jpeg_fixture([12, 34, 56]);
    std::fs::create_dir_all(poster_path.parent().unwrap()).expect("video dir");
    std::fs::write(&poster_path, &poster_bytes).expect("write poster");
    std::fs::write(
        project_path.join("assets/videos/genset/withposter.mp4"),
        b"video",
    )
    .expect("write video");
    std::fs::write(
        project_path.join("assets/videos/genset/noposter.mp4"),
        b"video",
    )
    .expect("write video");
    store
        .persist_generated_asset(&project.id, "job-1", "genset", &video_fact("withposter"))
        .expect("withposter persists");
    store
        .persist_generated_asset(&project.id, "job-1", "genset", &video_fact("noposter"))
        .expect("noposter persists");

    let (assets, operations) = store
        .list_assets_with_filesystem_operations(&project.id, true, true, AssetScope::All)
        .expect("list");
    assert_eq!(
        operations.poster_stats, 0,
        "poster presence is served from the indexed envelope"
    );
    let by_id = |id: &str| {
        assets
            .iter()
            .find(|asset| asset["id"] == id)
            .unwrap_or_else(|| panic!("asset {id} present"))
    };
    let poster_url = by_id("withposter")["posterUrl"]
        .as_str()
        .expect("poster URL");
    let poster_sha256 = poster_url.rsplit('/').next().expect("poster digest");
    assert_eq!(poster_sha256.len(), 64);
    assert_eq!(
        store
            .get_asset_poster(&project.id, "withposter", poster_sha256)
            .expect("advertised poster resolves")
            .bytes,
        poster_bytes
    );
    assert!(
        by_id("noposter").get("posterUrl").is_none(),
        "no posterUrl is advertised when the poster file is absent"
    );

    std::fs::remove_file(&poster_path).expect("external source poster removal");
    let (assets, operations) = store
        .list_assets_with_filesystem_operations(&project.id, true, true, AssetScope::All)
        .expect("ordinary list after removal");
    assert_eq!(operations.poster_stats, 0);
    assert_eq!(operations.poster_reads, 0);
    assert_eq!(operations.directory_scans, 0);
    assert_eq!(
        assets
            .iter()
            .find(|asset| asset["id"] == "withposter")
            .and_then(|asset| asset.get("posterUrl"))
            .and_then(Value::as_str),
        Some(poster_url),
        "source deletion cannot invalidate the DB-backed advertised URL"
    );
    assert_eq!(
        store
            .get_asset_poster(&project.id, "withposter", poster_sha256)
            .expect("DB-backed poster survives source deletion")
            .bytes,
        poster_bytes
    );

    store
        .reindex_project(&project.id)
        .expect("explicit reindex");
    let assets = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("list after reindex");
    assert!(
        assets
            .iter()
            .find(|asset| asset["id"] == "withposter")
            .and_then(|asset| asset.get("posterUrl"))
            .is_none(),
        "reindex atomically clears the URL and blob when its source is gone"
    );
    assert!(matches!(
        store.get_asset_poster(&project.id, "withposter", poster_sha256),
        Err(ProjectStoreError::NotFound(_))
    ));
}

#[test]
fn indexed_poster_rejects_malformed_and_oversized_jpeg_payloads() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Poster validation").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let fact = |id: &str| {
        json!({
            "assetId": id,
            "mediaPath": format!("assets/videos/set/{id}.mp4"),
            "mimeType": "video/mp4",
            "type": "video",
            "displayName": id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "image_to_video",
            "model": "wan",
            "adapter": "wan",
            "prompt": "x",
        })
    };

    for id in ["malformed", "oversized"] {
        let media = project_path.join(format!("assets/videos/set/{id}.mp4"));
        std::fs::create_dir_all(media.parent().unwrap()).unwrap();
        std::fs::write(&media, b"video").unwrap();
        let payload = if id == "malformed" {
            vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, b'J', b'F', b'I', b'F']
        } else {
            vec![0u8; 16 * 1024 * 1024 + 1]
        };
        std::fs::write(media.with_extension("poster.jpg"), payload).unwrap();
        store
            .persist_generated_asset(&project.id, "job", "set", &fact(id))
            .unwrap();
    }

    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(listed.len(), 2);
    assert!(
        listed.iter().all(|asset| asset.get("posterUrl").is_none()),
        "neither JPEG-looking malformed bytes nor an oversized file may be advertised"
    );
}

#[test]
fn get_asset_poster_fails_closed_when_indexed_blob_or_digest_is_tampered() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Poster integrity").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let media = project_path.join("assets/videos/set/integrity.mp4");
    let original = jpeg_fixture([1, 2, 3]);
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, b"video").unwrap();
    std::fs::write(media.with_extension("poster.jpg"), &original).unwrap();
    store
        .persist_generated_asset(
            &project.id,
            "job",
            "set",
            &json!({
                "assetId": "integrity",
                "mediaPath": "assets/videos/set/integrity.mp4",
                "mimeType": "video/mp4",
                "type": "video",
                "displayName": "Integrity",
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "image_to_video",
                "model": "wan",
                "adapter": "wan",
                "prompt": "x",
            }),
        )
        .unwrap();
    let advertised_sha = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap()[0]["posterUrl"]
        .as_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap()
        .to_owned();
    let connection = Connection::open(project_path.join("project.db")).unwrap();

    connection
        .execute(
            "update assets set poster_bytes = ?1 where id = 'integrity'",
            params![jpeg_fixture([9, 8, 7])],
        )
        .unwrap();
    assert!(matches!(
        store.get_asset_poster(&project.id, "integrity", &advertised_sha),
        Err(ProjectStoreError::NotFound(_))
    ));

    connection
        .execute(
            "update assets set poster_bytes = ?1, poster_sha256 = ?2 where id = 'integrity'",
            params![original, "0".repeat(64)],
        )
        .unwrap();
    assert!(matches!(
        store.get_asset_poster(&project.id, "integrity", &advertised_sha),
        Err(ProjectStoreError::NotFound(_))
    ));
}

#[test]
fn indexed_poster_rejects_symlink_sources_and_never_writes_a_poster_destination() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Poster symlinks").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let video_dir = project_path.join("assets/videos/set");
    std::fs::create_dir_all(&video_dir).unwrap();
    std::fs::write(video_dir.join("linked.mp4"), b"video").unwrap();
    let outside_poster = temp_dir.path().join("outside-secret.jpg");
    let outside_bytes = jpeg_fixture([11, 22, 33]);
    std::fs::write(&outside_poster, &outside_bytes).unwrap();
    let linked_poster = video_dir.join("linked.poster.jpg");
    if let Err(error) = create_file_symlink(&outside_poster, &linked_poster) {
        eprintln!("symlink privilege unavailable; skipping symlink fixture: {error}");
        return;
    }
    let fact = |id: &str| {
        json!({
            "assetId": id,
            "mediaPath": format!("assets/videos/set/{id}.mp4"),
            "mimeType": "video/mp4",
            "type": "video",
            "displayName": id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "image_to_video",
            "model": "wan",
            "adapter": "wan",
            "prompt": "x",
        })
    };
    store
        .persist_generated_asset(&project.id, "job", "set", &fact("linked"))
        .unwrap();
    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    assert!(listed[0].get("posterUrl").is_none());
    assert_eq!(std::fs::read(&outside_poster).unwrap(), outside_bytes);

    let outside_destination = temp_dir.path().join("outside-destination");
    std::fs::create_dir_all(&outside_destination).unwrap();
    let destination_link = project_path.join("assets/posters");
    if let Err(error) = create_dir_symlink(&outside_destination, &destination_link) {
        eprintln!("directory symlink privilege unavailable; skipping destination fixture: {error}");
        return;
    }
    std::fs::write(video_dir.join("contained.mp4"), b"video").unwrap();
    let contained_bytes = jpeg_fixture([44, 55, 66]);
    std::fs::write(video_dir.join("contained.poster.jpg"), &contained_bytes).unwrap();
    store
        .persist_generated_asset(&project.id, "job", "set", &fact("contained"))
        .unwrap();
    assert_eq!(
        std::fs::read_dir(&outside_destination).unwrap().count(),
        0,
        "DB-backed indexing performs no poster destination filesystem writes"
    );
    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    let contained = listed
        .iter()
        .find(|asset| asset["id"] == "contained")
        .unwrap();
    let sha = contained["posterUrl"]
        .as_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap();
    assert_eq!(
        store
            .get_asset_poster(&project.id, "contained", sha)
            .unwrap()
            .bytes,
        contained_bytes
    );
}

#[test]
fn same_id_overwrite_atomically_clears_and_replaces_poster_blob() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Poster overwrite").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let media = project_path.join("assets/videos/set/reused.mp4");
    let poster = media.with_extension("poster.jpg");
    let poster_a = jpeg_fixture([10, 20, 30]);
    let poster_b = jpeg_fixture([40, 50, 60]);
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, b"video-a").unwrap();
    std::fs::write(&poster, &poster_a).unwrap();
    let fact = json!({
        "assetId": "reused",
        "mediaPath": "assets/videos/set/reused.mp4",
        "mimeType": "video/mp4",
        "type": "video",
        "displayName": "Reused",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "image_to_video",
        "model": "wan",
        "adapter": "wan",
        "prompt": "x",
    });
    store
        .persist_generated_asset(&project.id, "job", "set", &fact)
        .unwrap();
    let old_url = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap()[0]["posterUrl"]
        .as_str()
        .unwrap()
        .to_owned();
    let old_sha = old_url.rsplit('/').next().unwrap().to_owned();

    std::fs::remove_file(&poster).unwrap();
    std::fs::write(&media, b"video-b").unwrap();
    store
        .persist_generated_asset(&project.id, "job", "set", &fact)
        .unwrap();
    assert!(store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap()[0]
        .get("posterUrl")
        .is_none());
    assert!(matches!(
        store.get_asset_poster(&project.id, "reused", &old_sha),
        Err(ProjectStoreError::NotFound(_))
    ));

    std::fs::write(&poster, &poster_b).unwrap();
    store
        .persist_generated_asset(&project.id, "job", "set", &fact)
        .unwrap();
    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    let new_sha = listed[0]["posterUrl"]
        .as_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap();
    assert_ne!(new_sha, old_sha);
    assert_eq!(
        store
            .get_asset_poster(&project.id, "reused", new_sha)
            .unwrap()
            .bytes,
        poster_b
    );
}

#[test]
fn purge_and_orphan_cleanup_prevent_poster_inheritance_on_id_reuse() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let fact = |id: &str| {
        json!({
            "assetId": id,
            "mediaPath": format!("assets/videos/set/{id}.mp4"),
            "mimeType": "video/mp4",
            "type": "video",
            "displayName": id,
            "createdAt": "2026-05-25T00:00:00Z",
            "mode": "image_to_video",
            "model": "wan",
            "adapter": "wan",
            "prompt": "x",
        })
    };

    for lifecycle in ["purge", "orphan"] {
        let project = store.create_project(lifecycle).unwrap();
        let project_path = store.find_project_path(&project.id).unwrap();
        let media = project_path.join(format!("assets/videos/set/{lifecycle}.mp4"));
        let poster = media.with_extension("poster.jpg");
        std::fs::create_dir_all(media.parent().unwrap()).unwrap();
        std::fs::write(&media, b"old-video").unwrap();
        std::fs::write(&poster, jpeg_fixture([70, 80, 90])).unwrap();
        store
            .persist_generated_asset(&project.id, "job", "set", &fact(lifecycle))
            .unwrap();
        let old_sha = store
            .list_assets(&project.id, true, true, AssetScope::All)
            .unwrap()[0]["posterUrl"]
            .as_str()
            .unwrap()
            .rsplit('/')
            .next()
            .unwrap()
            .to_owned();

        if lifecycle == "purge" {
            store.purge_asset(&project.id, lifecycle, true).unwrap();
        } else {
            std::fs::remove_file(&media).unwrap();
            assert_eq!(store.prune_orphaned_assets(&project.id).unwrap(), 1);
        }
        assert!(
            !poster.exists(),
            "{lifecycle} removes the old source poster"
        );
        assert!(matches!(
            store.get_asset_poster(&project.id, lifecycle, &old_sha),
            Err(ProjectStoreError::NotFound(_))
        ));

        std::fs::create_dir_all(media.parent().unwrap()).unwrap();
        std::fs::write(&media, b"new-video-without-poster").unwrap();
        store
            .persist_generated_asset(&project.id, "job", "set", &fact(lifecycle))
            .unwrap();
        assert!(
            store
                .list_assets(&project.id, true, true, AssetScope::All)
                .unwrap()[0]
                .get("posterUrl")
                .is_none(),
            "{lifecycle} id reuse cannot inherit the predecessor poster"
        );
    }
}

#[test]
fn duplicate_sidecar_ids_keep_envelope_and_poster_blob_from_one_winner() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Duplicate ids").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    for (folder, file, poster_rgb) in [
        ("a", "first", [101, 102, 103]),
        ("b", "second", [201, 202, 203]),
    ] {
        let media_rel = format!("assets/videos/{folder}/{file}.mp4");
        let media = project_path.join(&media_rel);
        std::fs::create_dir_all(media.parent().unwrap()).unwrap();
        std::fs::write(&media, b"video").unwrap();
        std::fs::write(media.with_extension("poster.jpg"), jpeg_fixture(poster_rgb)).unwrap();
        let asset = build_generated_asset_sidecar(
            &project.id,
            "job",
            "set",
            &json!({
                "assetId": "duplicate",
                "mediaPath": media_rel,
                "mimeType": "video/mp4",
                "type": "video",
                "displayName": file,
                "createdAt": "2026-05-25T00:00:00Z",
                "mode": "image_to_video",
                "model": "wan",
                "adapter": "wan",
                "prompt": "x",
            }),
        );
        write_json(&media.with_extension("sceneworks.json"), &asset).unwrap();
    }

    store.reindex_project(&project.id).unwrap();
    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(listed.len(), 1);
    let media_path = listed[0]["file"]["path"].as_str().unwrap();
    let expected = if media_path.contains("/a/") {
        jpeg_fixture([101, 102, 103])
    } else {
        jpeg_fixture([201, 202, 203])
    };
    let sha = listed[0]["posterUrl"]
        .as_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap();
    assert_eq!(
        store
            .get_asset_poster(&project.id, "duplicate", sha)
            .unwrap()
            .bytes,
        expected,
        "the URL, envelope file path, and blob are replaced in one row transaction"
    );
}

#[test]
fn dirty_marker_repair_reconciles_poster_blob_after_db_failure() {
    let temp_dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Poster repair").unwrap();
    let project_path = store.find_project_path(&project.id).unwrap();
    let media = project_path.join("assets/videos/set/repaired.mp4");
    let poster = media.with_extension("poster.jpg");
    let before = jpeg_fixture([3, 4, 5]);
    let after = jpeg_fixture([6, 7, 8]);
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, b"video").unwrap();
    std::fs::write(&poster, &before).unwrap();
    let fact = json!({
        "assetId": "repaired",
        "mediaPath": "assets/videos/set/repaired.mp4",
        "mimeType": "video/mp4",
        "type": "video",
        "displayName": "Repaired",
        "createdAt": "2026-05-25T00:00:00Z",
        "mode": "image_to_video",
        "model": "wan",
        "adapter": "wan",
        "prompt": "x",
    });
    store
        .persist_generated_asset(&project.id, "job", "set", &fact)
        .unwrap();
    let old_sha = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap()[0]["posterUrl"]
        .as_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap()
        .to_owned();

    std::fs::write(&poster, &after).unwrap();
    fail_next_asset_index_db_mutation();
    assert!(store
        .update_asset_status(
            &project.id,
            "repaired",
            AssetStatusPatch {
                favorite: Some(true),
                ..AssetStatusPatch::default()
            },
        )
        .is_err());
    assert!(project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .unwrap();
    assert_eq!(listed[0]["status"]["favorite"], true);
    let new_sha = listed[0]["posterUrl"]
        .as_str()
        .unwrap()
        .rsplit('/')
        .next()
        .unwrap();
    assert_ne!(new_sha, old_sha);
    assert_eq!(
        store
            .get_asset_poster(&project.id, "repaired", new_sha)
            .unwrap()
            .bytes,
        after
    );
    assert!(matches!(
        store.get_asset_poster(&project.id, "repaired", &old_sha),
        Err(ProjectStoreError::NotFound(_))
    ));
    assert!(!project_path.join(ASSET_INDEX_DIRTY_MARKER).exists());
}

#[test]
fn build_generated_asset_sidecar_derives_video_type_from_mime() {
    let fact = json!({
        "assetId": "asset_v",
        "mediaPath": "assets/videos/genset_y/clip.mp4",
        "mimeType": "video/mp4",
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_y", &fact);
    assert_eq!(asset["type"], json!("video"));
    assert_eq!(asset["file"]["mimeType"], json!("video/mp4"));
}

/// sc-13404: an audio fact assembles a `type: "audio"` asset whose `file` block carries the
/// measured duration/sampleRate/channels (no width/height/fps) and whose recipe round-trips the
/// requested voice/language. Mirrors the video-type derivation test.
#[test]
fn build_generated_asset_sidecar_assembles_audio() {
    let fact = json!({
        "type": "audio",
        "assetId": "asset_a",
        "mediaPath": "assets/audios/genset_z/hello.wav",
        "mimeType": "audio/wav",
        "duration": 3.25,
        "sampleRate": 24_000,
        "channels": 1,
        "family": "kokoro",
        "displayName": "Hello from SceneWorks audio.",
        "model": "kokoro_82m",
        "adapter": "kokoro",
        "prompt": "Hello from SceneWorks audio.",
        "voice": "af_heart",
        "language": "en-US",
        "targetDurationSecs": 3.0,
        "seed": 42,
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_z", &fact);
    assert_eq!(asset["type"], json!("audio"));
    // Origin is derived to the Audio Studio, not the image_studio catch-all.
    assert_eq!(asset["origin"], json!("audio_studio"));
    // File block: measured audio facts, no width/height/fps.
    assert_eq!(asset["file"]["mimeType"], json!("audio/wav"));
    assert_eq!(asset["file"]["duration"], json!(3.25));
    assert_eq!(asset["file"]["sampleRate"], json!(24_000));
    assert_eq!(asset["file"]["channels"], json!(1));
    assert_eq!(asset["file"]["width"], Value::Null);
    assert_eq!(asset["file"]["fps"], Value::Null);
    // Recipe replays the requested knobs.
    assert_eq!(asset["recipe"]["model"], json!("kokoro_82m"));
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["voice"],
        json!("af_heart")
    );
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["language"],
        json!("en-US")
    );
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["targetDurationSecs"],
        json!(3.0)
    );
    assert_eq!(asset["lineage"]["jobId"], json!("job-1"));
}

#[test]
fn build_generated_asset_sidecar_preserves_variant_lineage_and_extra() {
    let fact = json!({
        "assetId": "asset_upscaled",
        "mediaPath": "assets/images/genset_x/city_0001_upscaled_x2.png",
        "mimeType": "image/png",
        "sourceAssetId": "asset_original",
        "parents": ["asset_original"],
        "extra": {
            "isUpscaled": true,
            "upscaledFromAssetId": "asset_original",
            "factor": 2,
            "engine": "real-esrgan",
        },
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset_x", &fact);

    assert_eq!(asset["lineage"]["sourceAssetId"], json!("asset_original"));
    assert_eq!(asset["lineage"]["parents"], json!(["asset_original"]));
    assert_eq!(asset["extra"]["isUpscaled"], json!(true));
    assert_eq!(
        asset["extra"]["upscaledFromAssetId"],
        json!("asset_original")
    );
    assert_eq!(asset["extra"]["factor"], json!(2));
    assert_eq!(asset["extra"]["engine"], json!("real-esrgan"));
}

/// sc-10117: the inline Image Studio upscale post-pass used to link its `(Nx upscaled)` variant
/// to the base only via a dead `upscaledFrom` field, so existing upscaled assets have no fold
/// lineage. The backfill reconstructs it from the `_up{N}x` filename within the generation set so
/// the pair collapses in the Library. Additive + idempotent.
#[test]
fn backfill_reconstructs_inline_upscale_variant_lineage() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Upscales").expect("project creates");

    // A base image and its inline-upscaled variant in one generation set. The upscaled fact
    // mirrors the OLD write_upscaled_asset output: an `upscale` recipe record but NO
    // sourceAssetId/extra link (the dead `upscaledFrom` field never reached the sidecar).
    let base = json!({
        "assetId": "base_1",
        "mediaPath": "assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001.png",
        "mimeType": "image/png",
        "displayName": "Scene #1",
        "createdAt": "2026-07-01T00:00:00Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
    });
    let upscaled = json!({
        "assetId": "up_1",
        "mediaPath": "assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001_up2x.png",
        "mimeType": "image/png",
        "displayName": "Scene #1 (2x upscaled)",
        "createdAt": "2026-07-01T00:00:01Z",
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "rawAdapterSettings": { "upscale": { "enabled": true, "engine": "seedvr2", "factor": 2 } },
    });
    store
        .persist_generated_asset(&project.id, "job-1", "genset_up", &base)
        .expect("base persists");
    store
        .persist_generated_asset(&project.id, "job-1", "genset_up", &upscaled)
        .expect("upscaled persists");

    let project_path = store.find_project_path(&project.id).expect("project path");
    let up_sidecar = project_path
        .join("assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001_up2x.sceneworks.json");

    // Precondition: the upscaled sidecar has no fold link yet.
    let before = read_json(&up_sidecar).expect("read upscaled sidecar");
    assert!(
        before.pointer("/extra/upscaledFromAssetId").is_none(),
        "precondition: the buggy inline upscale left no fold link"
    );
    assert!(before
        .pointer("/lineage/sourceAssetId")
        .and_then(Value::as_str)
        .is_none());

    let healed = backfill_upscale_variant_lineage(&project_path).expect("backfill runs");
    assert_eq!(healed, 1, "exactly the one upscaled variant is healed");

    let after = read_json(&up_sidecar).expect("re-read healed sidecar");
    assert_eq!(after["extra"]["upscaledFromAssetId"], json!("base_1"));
    assert_eq!(after["extra"]["isUpscaled"], json!(true));
    assert_eq!(after["extra"]["factor"], json!(2));
    assert_eq!(after["extra"]["engine"], json!("seedvr2"));
    assert_eq!(after["lineage"]["sourceAssetId"], json!("base_1"));
    assert_eq!(after["lineage"]["parents"], json!(["base_1"]));

    // The base sidecar is untouched — it must NOT be marked as an upscale.
    let base_sidecar = project_path
        .join("assets/images/genset_up/2026-07-01_z_image_turbo_scene_0001.sceneworks.json");
    let base_after = read_json(&base_sidecar).expect("read base sidecar");
    assert!(
        base_after.pointer("/extra/isUpscaled").is_none(),
        "the base image must not be marked upscaled"
    );

    // Idempotent: a second pass heals nothing (the link is now present).
    let again = backfill_upscale_variant_lineage(&project_path).expect("second backfill");
    assert_eq!(again, 0, "already-linked variants are skipped");
}

#[test]
fn build_generated_asset_sidecar_assembles_video_replace_person() {
    // sc-1656 slice 3: the video branch carries duration/fps, has no
    // stylePreset, builds video-shaped normalizedSettings + lineage, and fills
    // the honest replace_person defaults when no replacementStatus is reported.
    let fact = json!({
        "type": "video",
        "assetId": "asset-output",
        "mediaPath": "assets/videos/replacement.mp4",
        "mimeType": "video/mp4",
        "width": 1280, "height": 720, "duration": 6.0, "fps": 25,
        "quality": "balanced", "family": "ltx-video",
        "seed": 44, "displayName": "Replace the hero",
        "createdAt": "2026-05-17T00:00:00Z",
        "mode": "replace_person", "model": "wan_2_2", "adapter": "wan_video",
        "prompt": "Replace the hero", "negativePrompt": "", "loras": [],
        "rawAdapterSettings": {},
        "sourceClipAssetId": "asset-video",
        "personTrackId": "track-1",
        "replacementMode": "full_person_keep_outfit",
        "characterId": "character-1", "characterLookId": "look-1",
        "timelineContext": {},
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
    assert_eq!(asset["type"], json!("video"));
    assert_eq!(asset["file"]["duration"], json!(6.0));
    assert_eq!(asset["file"]["fps"], json!(25));
    assert!(asset["recipe"].get("stylePreset").is_none());
    let normalized = &asset["recipe"]["normalizedSettings"];
    assert_eq!(normalized["personTrackId"], json!("track-1"));
    assert_eq!(
        normalized["replacementMode"],
        json!("full_person_keep_outfit")
    );
    assert_eq!(normalized["timelineContextRef"], json!("lineage.timeline"));
    assert_eq!(normalized["personDetectionActive"], json!(false));
    assert_eq!(normalized["replacementActive"], json!(false));
    assert_eq!(asset["lineage"]["sourceClipAssetId"], json!("asset-video"));
    assert_eq!(asset["lineage"]["characterId"], json!("character-1"));
    assert_eq!(asset["lineage"]["parents"], json!(["asset-video"]));
}

#[test]
fn video_sidecar_persists_observed_veda_separately_from_replay_controls() {
    let controls = json!({"enabled":true,"generated":"90%","reference":"12",
            "fullAttentionLayers":[0],"fullAttentionSteps":[1]});
    for outcome in [
        json!({"effective":"sparse","kernel":"actual-native-kernel","plan":"trained",
                "planMatch":"exact","totalAttentionFraction":0.2}),
        json!({"effective":"dense","totalAttentionFraction":1.0,
                "fallbackReason":"actual-device self-test failed"}),
    ] {
        let fact = json!({"type":"video","assetId":"asset-1", "mediaPath":"assets/videos/h3.mp4",
                "mimeType":"video/mp4","width":1280,"height":720,"fps":25,"duration":6.0,
                "model":"minimax_h3", "rawAdapterSettings":{"veda":controls,"vedaExecution":outcome}});
        let sidecar = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
        let persisted: Value =
            serde_json::from_str(&serde_json::to_string(&sidecar).unwrap()).unwrap();
        assert_eq!(persisted["recipe"]["rawAdapterSettings"]["veda"], controls);
        assert_eq!(
            persisted["recipe"]["rawAdapterSettings"]["vedaExecution"],
            outcome
        );
        assert!(persisted["recipe"]["normalizedSettings"]
            .get("vedaExecution")
            .is_none());
    }
    let old = build_generated_asset_sidecar(
        "project-1",
        "job-2",
        "genset-2",
        &json!({
            "type":"video","assetId":"asset-2","mimeType":"video/mp4","rawAdapterSettings":{}
        }),
    );
    assert!(old["recipe"]["rawAdapterSettings"]
        .get("vedaExecution")
        .is_none());
}

/// sc-12371: `asset.file` MEASURES the clip; `recipe.normalizedSettings` REPLAYS the ask. The
/// same fact feeds both and they must not be given the same number.
///
/// The trap this pins is a real one that nearly shipped: making `file.duration` honest by
/// changing the fact's `duration` key ALSO rewrites `normalizedSettings.duration`, which
/// sc-12324's "re-run this generation" rebuilds the payload from. A 6 s ask would replay as
/// 5.96 s — off the model's `limits.durations` menu, which sc-12347 now enforces server-side,
/// so the re-run could be refused outright.
///
/// DISCRIMINATING: the measured pair (149 frames / 5.96 s) differs from the requested pair
/// (6.0 s), so collapsing the two in EITHER direction is RED here.
#[test]
fn video_sidecar_file_measures_the_clip_while_normalized_settings_replay_the_ask() {
    let fact = json!({
        "type": "video",
        "assetId": "asset-1",
        "mediaPath": "assets/videos/hero.mp4",
        "mimeType": "video/mp4",
        "width": 1280, "height": 720,
        // What the user PICKED off the model's menus.
        "duration": 6.0, "fps": 25,
        // What a Wan engine actually rendered for that ask: the 4k+1 stride's 149 frames.
        "encodedFrameCount": 149, "encodedDuration": 5.96, "encodedFps": 25,
        "quality": "balanced", "family": "wan-video",
        "seed": 7, "displayName": "Hero", "createdAt": "2026-07-16T00:00:00Z",
        "mode": "text_to_video", "model": "bernini", "adapter": "mlx_bernini",
        "prompt": "hero", "negativePrompt": "", "loras": [],
        "rawAdapterSettings": { "frameCount": 149 },
        "timelineContext": {},
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);

    // The file block describes the mp4 that exists.
    assert_eq!(asset["file"]["duration"], json!(5.96));
    assert_eq!(asset["file"]["frameCount"], json!(149));
    assert_eq!(asset["file"]["fps"], json!(25));
    assert_ne!(
        asset["file"]["duration"],
        json!(6.0),
        "file.duration must not echo the request — that is the sc-12371 lie"
    );

    // The recipe replays what was asked for, so the studio form's duration select still has an
    // option to land on.
    let normalized = &asset["recipe"]["normalizedSettings"];
    assert_eq!(normalized["duration"], json!(6.0));
    assert_eq!(normalized["fps"], json!(25));
    assert_ne!(
        normalized["duration"],
        json!(5.96),
        "normalizedSettings must round-trip the user's ask — a measured duration is off the \
             model's limits.durations menu and sc-12347 enforces that menu"
    );
}

/// sc-12371: a fact written before the worker measured its clip (every asset already on disk)
/// still gets a `file.duration` — the requested value it has always reported. A missing
/// duration would be worse than a slightly optimistic one.
#[test]
fn video_sidecar_file_falls_back_to_the_requested_duration_for_older_facts() {
    let fact = json!({
        "type": "video",
        "assetId": "asset-old",
        "mediaPath": "assets/videos/old.mp4",
        "mimeType": "video/mp4",
        "width": 1280, "height": 720, "duration": 6.0, "fps": 25,
        "quality": "balanced", "family": "wan-video",
        "seed": 7, "displayName": "Old", "createdAt": "2026-05-17T00:00:00Z",
        "mode": "text_to_video", "model": "wan_2_2", "adapter": "mlx_wan",
        "prompt": "old", "negativePrompt": "", "loras": [],
        "rawAdapterSettings": {},
        "timelineContext": {},
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
    assert_eq!(asset["file"]["duration"], json!(6.0));
    assert_eq!(asset["file"]["fps"], json!(25));
    // No measurement was recorded, so the file block reports no frame count rather than
    // inventing one.
    assert_eq!(asset["file"]["frameCount"], Value::Null);
}

/// sc-19577: `file.hasAudio` is THREE-STATE, and each state is asserted against a fact that
/// genuinely has it.
///
/// This is the acute case for the sc-19488 "a test asserting a default passes whether or not the
/// code ran" trap: `true` alone would pass against a hard-coded `true`, and `false` alone against
/// an omission. So all three facts are built and all three outcomes pinned:
///
/// * a render that produced audio → `true`;
/// * a render that produced none → `false` (NOT absent — "measured, and silent" is a fact);
/// * a fact written before sc-19577 → ABSENT (NOT `false` — claiming an LTX clip from last month
///   is silent is a fresh lie about a file on disk).
#[test]
fn video_sidecar_records_the_muxed_soundtrack_as_three_distinct_states() {
    let fact = |audio: Option<bool>| {
        let mut fact = json!({
            "type": "video",
            "assetId": "asset-av",
            "mediaPath": "assets/videos/av.mp4",
            "mimeType": "video/mp4",
            "width": 1344, "height": 768, "duration": 5.1667, "fps": 24,
            "encodedFrameCount": 124, "encodedDuration": 5.1667, "encodedFps": 24,
            "quality": "balanced", "family": "minimax-h3",
            "seed": 7, "displayName": "AV", "createdAt": "2026-08-15T00:00:00Z",
            "mode": "text_to_video", "model": "minimax_h3", "adapter": "mlx_minimax_h3",
            "prompt": "a lighthouse keeper hums", "negativePrompt": "", "loras": [],
            "rawAdapterSettings": {},
            "timelineContext": {},
        });
        if let (Some(audio), Some(object)) = (audio, fact.as_object_mut()) {
            object.insert("hasAudio".to_owned(), json!(audio));
        }
        build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact)
    };

    assert_eq!(fact(Some(true))["file"]["hasAudio"], json!(true));
    assert_eq!(
        fact(Some(false))["file"]["hasAudio"],
        json!(false),
        "a render that produced no audio must record that it did not — the SAME model id \
             produces both, so the record is the only place the difference lives"
    );
    assert!(
        fact(None)["file"].get("hasAudio").is_none(),
        "a pre-sc-19577 fact must leave the key ABSENT; defaulting it to false would claim \
             every existing LTX render is silent"
    );

    // A non-boolean `hasAudio` is not a measurement and must not be carried through as one.
    let mut junk = json!({
        "type": "video", "assetId": "a", "mediaPath": "assets/videos/j.mp4",
        "mimeType": "video/mp4", "width": 16, "height": 16, "duration": 1.0, "fps": 24,
        "quality": "balanced", "family": "ltx", "seed": 1, "displayName": "J",
        "createdAt": "2026-08-15T00:00:00Z", "mode": "text_to_video", "model": "ltx_2_3",
        "adapter": "mlx_ltx", "prompt": "j", "negativePrompt": "", "loras": [],
        "rawAdapterSettings": {}, "timelineContext": {},
    });
    junk.as_object_mut()
        .expect("object")
        .insert("hasAudio".to_owned(), json!("yes"));
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &junk);
    assert!(asset["file"].get("hasAudio").is_none());
}

/// sc-12345: the fit and the list-valued source ids survive onto `recipe.normalizedSettings`
/// so the sc-12324 replay can rebuild the form, and every source asset lands in
/// `lineage.parents` — not just the singular ones. ads2v is the densest mode.
#[test]
fn build_generated_asset_sidecar_carries_video_fit_and_multi_source_ids() {
    let fact = json!({
        "type": "video",
        "assetId": "asset-output",
        "mediaPath": "assets/videos/ad.mp4",
        "mimeType": "video/mp4",
        "width": 1280, "height": 720, "duration": 6.0, "fps": 25,
        "quality": "balanced", "family": "bernini",
        "seed": 44, "displayName": "The hero drives past",
        "createdAt": "2026-07-16T00:00:00Z",
        "mode": "ads2v", "model": "bernini_2", "adapter": "mlx_bernini",
        "prompt": "the hero drives past", "negativePrompt": "", "loras": [],
        "rawAdapterSettings": {},
        "fitMode": "pad",
        "sourceClipAssetId": "clip-main",
        "referenceClipAssetId": "clip-ref",
        "referenceAssetIds": ["ref-1", "ref-2"],
        "sourceClipAssetIds": [],
        "timelineContext": {},
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
    let normalized = &asset["recipe"]["normalizedSettings"];
    assert_eq!(normalized["fitMode"], json!("pad"));
    assert_eq!(normalized["referenceClipAssetId"], json!("clip-ref"));
    assert_eq!(normalized["referenceAssetIds"], json!(["ref-1", "ref-2"]));
    // Every source the clip was derived from — the reference video and the subject
    // references are parents just as much as the main clip.
    assert_eq!(
        asset["lineage"]["parents"],
        json!(["clip-main", "clip-ref", "ref-1", "ref-2"])
    );

    // mv2v's clip array is a parent set too.
    let mv2v = json!({
        "type": "video",
        "assetId": "asset-mv2v",
        "mediaPath": "assets/videos/stitch.mp4",
        "mimeType": "video/mp4",
        "mode": "multi_video_to_video", "model": "bernini_2",
        "sourceClipAssetIds": ["clip-a", "clip-b"],
    });
    let stitched = build_generated_asset_sidecar("project-1", "job-2", "genset-1", &mv2v);
    assert_eq!(
        stitched["recipe"]["normalizedSettings"]["sourceClipAssetIds"],
        json!(["clip-a", "clip-b"])
    );
    assert_eq!(stitched["lineage"]["parents"], json!(["clip-a", "clip-b"]));

    // A fact written before sc-12345 has no such keys: the lists stay array-shaped so the
    // replay reader never distinguishes "absent" from "empty", and the fit reads null.
    let legacy = json!({
        "type": "video",
        "assetId": "asset-old",
        "mediaPath": "assets/videos/old.mp4",
        "mimeType": "video/mp4",
        "mode": "text_to_video", "model": "ltx_2_3",
    });
    let old = build_generated_asset_sidecar("project-1", "job-3", "genset-1", &legacy);
    let old_normalized = &old["recipe"]["normalizedSettings"];
    assert_eq!(old_normalized["referenceAssetIds"], json!([]));
    assert_eq!(old_normalized["sourceClipAssetIds"], json!([]));
    assert_eq!(old_normalized["referenceAudioAssetIds"], json!([]));
    assert_eq!(old_normalized["fitMode"], Value::Null);
    assert_eq!(old["lineage"]["parents"], json!([]));

    // sc-17160: the audio references replay and count as parents for the same reason the
    // image and clip lists do — a re-run that omits them renders the same prompt with its
    // audio conditioning silently gone, which is invisible in the output.
    let ref2va = json!({
        "type": "video",
        "assetId": "asset-ref2va",
        "mediaPath": "assets/videos/ref2va.mp4",
        "mimeType": "video/mp4",
        "mode": "reference_to_video", "model": "minimax_h3",
        "referenceAssetIds": ["ref-1"],
        "referenceAudioAssetIds": ["voice-1", "music-1"],
    });
    let multimodal = build_generated_asset_sidecar("project-1", "job-4", "genset-1", &ref2va);
    assert_eq!(
        multimodal["recipe"]["normalizedSettings"]["referenceAudioAssetIds"],
        json!(["voice-1", "music-1"])
    );
    assert_eq!(
        multimodal["lineage"]["parents"],
        json!(["ref-1", "voice-1", "music-1"])
    );
}

#[test]
fn build_generated_asset_sidecar_merges_replacement_status() {
    let fact = json!({
        "type": "video",
        "assetId": "a",
        "mediaPath": "assets/videos/x.mp4",
        "mimeType": "video/mp4",
        "mode": "replace_person",
        "replacementStatus": {"replacementActive": true, "maskMode": "segmentation"},
    });
    let asset = build_generated_asset_sidecar("p", "j", "g", &fact);
    let normalized = &asset["recipe"]["normalizedSettings"];
    assert_eq!(normalized["replacementActive"], json!(true));
    assert_eq!(normalized["maskMode"], json!("segmentation"));
    assert_eq!(normalized["personDetectionActive"], json!(false));
}

#[test]
fn build_generated_asset_sidecar_assembles_document() {
    // sc-1656 slice 4: interleaved document asset. The worker writes the body
    // JSON; Rust builds this sidecar — type document, mimeType
    // application/json, recipe.mode interleave, lineage.parents = image ids.
    let fact = json!({
        "type": "document",
        "assetId": "asset-doc",
        "mediaPath": "assets/documents/doc_1.json",
        "mimeType": "application/json",
        "displayName": "An illustrated guide",
        "createdAt": "2026-05-17T00:00:00Z",
        "mode": "interleave",
        "model": "sensenova_u1_8b",
        "adapter": "sensenova_u1",
        "prompt": "An illustrated guide",
        "negativePrompt": "",
        "seed": 7,
        "loras": [],
        "rawAdapterSettings": {"maxImages": 6},
        "maxImages": 6,
        "resolution": "2048x1152",
        "imageCount": 2,
        "parents": ["asset-img-1", "asset-img-2"],
    });
    let asset = build_generated_asset_sidecar("project-1", "job-1", "genset-1", &fact);
    assert_eq!(asset["type"], json!("document"));
    assert_eq!(asset["file"]["mimeType"], json!("application/json"));
    assert_eq!(asset["file"]["path"], json!("assets/documents/doc_1.json"));
    assert_eq!(asset["recipe"]["mode"], json!("interleave"));
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["imageCount"],
        json!(2)
    );
    assert_eq!(
        asset["recipe"]["normalizedSettings"]["resolution"],
        json!("2048x1152")
    );
    assert_eq!(
        asset["lineage"]["parents"],
        json!(["asset-img-1", "asset-img-2"])
    );
    assert_eq!(asset["generationSetId"], json!("genset-1"));
}

#[test]
fn project_create_writes_python_compatible_files_and_registry() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

    let project = store.create_project("My Project").expect("project creates");

    assert!(project.id.starts_with("project_"));
    assert!(project.path.ends_with("my-project.sceneworks"));
    for folder in PROJECT_FOLDERS {
        assert!(std::path::Path::new(&project.path).join(folder).exists());
    }
    let registry = std::fs::read_to_string(temp_dir.path().join("data/recent-projects.json"))
        .expect("registry reads");
    assert!(registry.contains(&project.id));
    assert!(std::path::Path::new(&project.path)
        .join("project.db")
        .exists());
}

#[test]
fn ensure_global_poses_project_is_idempotent_and_hidden() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

    store
        .create_project("Visible Project")
        .expect("project creates");

    let poses = store
        .ensure_global_poses_project()
        .expect("poses project ensures");
    assert_eq!(poses.id, GLOBAL_POSES_PROJECT_ID);
    assert!(std::path::Path::new(&poses.path)
        .join("assets/poses")
        .exists());

    // Idempotent: a second ensure returns the same project path, no duplicate.
    let again = store
        .ensure_global_poses_project()
        .expect("ensure idempotent");
    assert_eq!(again.path, poses.path);

    // Hidden from the project switcher, but addressable directly by id.
    let listed = store.list_projects().expect("list");
    assert!(listed.iter().all(|p| p.id != GLOBAL_POSES_PROJECT_ID));
    assert_eq!(
        store
            .get_project(GLOBAL_POSES_PROJECT_ID)
            .expect("get reserved")
            .id,
        GLOBAL_POSES_PROJECT_ID
    );
}

#[test]
fn list_projects_skips_a_corrupt_entry_and_returns_the_healthy_ones() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

    let good = store.create_project("Healthy One").expect("first project");
    let broken = store.create_project("Corrupt One").expect("second project");
    let also_good = store.create_project("Healthy Two").expect("third project");

    // Corrupt one project's `project.json` after the fact: truncate it to an
    // object missing every required field (id/name/createdAt) so
    // `read_project_summary` fails for that entry only.
    let broken_file = std::path::Path::new(&broken.path).join("project.json");
    std::fs::write(&broken_file, b"{}").expect("clobber project.json");

    let listed = store.list_projects().expect("list survives a bad entry");

    let ids: Vec<&str> = listed.iter().map(|p| p.id.as_str()).collect();
    assert!(
        ids.contains(&good.id.as_str()),
        "healthy project before the bad entry is still listed"
    );
    assert!(
        ids.contains(&also_good.id.as_str()),
        "healthy project after the bad entry is still listed"
    );
    assert!(
        !ids.contains(&broken.id.as_str()),
        "the corrupt entry is skipped, not surfaced"
    );
    assert_eq!(listed.len(), 2, "exactly the two healthy projects remain");
}

#[test]
fn create_pose_asset_copies_skeleton_and_writes_pose_sidecar() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");

    // Simulate the worker's pose_detect output in the shared cache.
    let job_id = "job_pose_1";
    let cache_dir = data_dir.join("cache").join("pose_detect").join(job_id);
    std::fs::create_dir_all(&cache_dir).expect("cache dir");
    let skeleton = cache_dir.join("photo_p0_skel.png");
    std::fs::write(&skeleton, b"\x89PNG fake skeleton bytes").expect("skeleton writes");

    let spec = json!({
        "jobId": job_id,
        "skeletonFile": "photo_p0_skel.png",
        "displayName": "Arm Raised",
        "category": "Dance",
        "tags": ["Dynamic", "dynamic", "  hero  "],
        "width": 768,
        "height": 1280,
        "pose": {
            "personIndex": 0,
            "facing": "front",
            "bbox": [0.1, 0.1, 0.9, 0.9],
            "keypoints": [[0.5, 0.1, 0.9]],
            "hands": [[], []],
            "face": [],
            "sourceAspect": 0.6,
            "sourceAssetId": "asset_src_1"
        }
    });

    let asset = store.create_pose_asset(&spec).expect("pose asset creates");

    assert!(asset["id"].as_str().unwrap().starts_with("asset_"));
    assert_eq!(asset["type"], json!("pose"));
    assert_eq!(asset["projectId"], json!(GLOBAL_POSES_PROJECT_ID));
    assert_eq!(asset["displayName"], json!("Arm Raised"));
    // Category folded into pose metadata for grouping; tags normalized + deduped.
    assert_eq!(asset["pose"]["category"], json!("Dance"));
    assert_eq!(asset["pose"]["facing"], json!("front"));
    assert_eq!(asset["tags"], json!(["dynamic", "hero"]));
    assert_eq!(asset["lineage"]["jobId"], json!(job_id));
    assert_eq!(asset["lineage"]["sourceAssetId"], json!("asset_src_1"));
    assert_eq!(asset["lineage"]["parents"], json!(["asset_src_1"]));
    assert_eq!(asset["file"]["mimeType"], json!("image/png"));
    assert_eq!(asset["file"]["width"], json!(768));

    // The PNG was copied into the reserved project's assets/poses folder.
    let project_path =
        std::path::PathBuf::from(store.get_project(GLOBAL_POSES_PROJECT_ID).unwrap().path);
    let rel = asset["file"]["path"].as_str().unwrap();
    assert!(rel.starts_with("assets/poses/"));
    assert!(project_path.join(rel).exists());

    // Indexed: it comes back from list_assets on the reserved project, pose intact.
    let listed = store
        .list_assets(GLOBAL_POSES_PROJECT_ID, true, true, AssetScope::All)
        .expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["id"], asset["id"]);
    assert_eq!(listed[0]["pose"]["category"], json!("Dance"));
}

#[test]
fn import_asset_records_source_lineage_and_provenance() {
    // Image Editor Save (sc-2434): a rasterized edit is uploaded with the id
    // of the asset it was opened from + an edit-chain provenance blob. The new
    // asset must link back via lineage and carry the provenance in `extra`,
    // while the original source asset is left untouched.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Edits").expect("project creates");

    // The original source asset (a plain import, no lineage).
    let src_file = temp_dir.path().join("source.png");
    std::fs::write(&src_file, b"\x89PNG source bytes").expect("source writes");
    let source = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: src_file,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("source imports");
    let source_id = source["id"].as_str().expect("source id").to_owned();
    assert_eq!(source["lineage"]["sourceAssetId"], Value::Null);

    // The edited bitmap saved from the editor, derived from the source.
    let edited_file = temp_dir.path().join("source-edited.png");
    std::fs::write(&edited_file, b"\x89PNG edited bytes").expect("edited writes");
    let provenance = json!({
        "editor": "image_editor",
        "edits": [{ "op": "crop" }, { "op": "upscale", "engine": "real-esrgan", "factor": 4 }],
    });
    let edited = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "source-edited.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: edited_file,
                source_asset_id: Some(source_id.clone()),
                provenance: Some(provenance.clone()),
            },
        )
        .expect("edited imports");

    // New, distinct asset that links back to the source and is Library-visible.
    assert_ne!(edited["id"], source["id"]);
    assert_eq!(edited["origin"], json!("upload"));
    assert_eq!(edited["lineage"]["sourceAssetId"], json!(source_id));
    assert_eq!(edited["lineage"]["parents"], json!([source_id]));
    // Provenance survives normalization in the top-level `extra` slot.
    assert_eq!(edited["extra"], provenance);

    // The source asset itself is untouched: still no lineage.
    let source_after = store
        .get_asset(&project.id, &source_id)
        .expect("source still present");
    assert_eq!(source_after["lineage"]["sourceAssetId"], Value::Null);
    assert!(source_after.get("extra").is_none());
}

// -----------------------------------------------------------------------------------------
// sc-15949: the embedded workflow on asset import
// -----------------------------------------------------------------------------------------
//
// The reader and its adversarial corpus are sc-15947's and live in `workflow_png`; nothing
// below re-tests them. What is under test here is the IMPORT boundary: that a found envelope
// lands in `extra.importedWorkflow` without the record pretending the image was generated
// here, that an absent chunk changes nothing, that a hostile one costs the user only that
// field, and that the scan walks chunks rather than pixels.

/// A small deterministic image. Nine by seven so the row stride is not a power of two.
fn workflow_rgb_fixture() -> image::RgbImage {
    image::RgbImage::from_fn(9, 7, |x, y| {
        image::Rgb([
            u8::try_from(x * 20).expect("fits"),
            u8::try_from(y * 30).expect("fits"),
            u8::try_from((x + y) * 10).expect("fits"),
        ])
    })
}

/// A sanitized envelope, built through the real parser so the fixture is exactly what the
/// contract admits — not a hand-rolled struct that could carry something `parse` would refuse.
fn workflow_fixture(prompt: &str) -> crate::workflow_share::WorkflowShare {
    crate::workflow_share::parse_workflow_share_json(
        &json!({
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "producer": {
                "name": "SceneWorks",
                "url": "https://github.com/SceneWorks/SceneWorks",
                "version": "0.8.1",
            },
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": prompt,
            "negativePrompt": "text, watermark",
            "seed": 880_412,
            "width": 9,
            "height": 7,
            "advanced": { "steps": 8, "sampler": "euler" },
        })
        .to_string(),
    )
    .expect("the fixture envelope parses")
}

/// A PNG as SceneWorks writes one, through the real sc-15947 writer: `Some` embeds the chunk,
/// `None` is the byte-identical opt-out.
fn sceneworks_png(share: Option<&crate::workflow_share::WorkflowShare>) -> Vec<u8> {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("generated.png");
    crate::workflow_png::write_workflow_chunk(&workflow_rgb_fixture(), &path, share)
        .expect("the fixture PNG writes");
    std::fs::read(&path).expect("the fixture PNG reads back")
}

/// Byte offset one past IHDR: signature + (length + type + 13 bytes + CRC). Where our writer
/// puts the chunk, and where the hand-built fixtures below splice theirs.
const AFTER_IHDR: usize = 8 + 4 + 4 + 13 + 4;

/// Bitwise CRC-32, so a hand-framed chunk (or a patched IHDR/IDAT) is well formed rather than
/// "passing" because the decoder threw it out on a checksum.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = if crc & 1 == 1 { 0xEDB8_8320 } else { 0 };
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}

/// A framed PNG chunk. `declared_len` overrides the length field so a chunk can lie about how
/// much data follows it.
fn framed_chunk(kind: &[u8; 4], data: &[u8], declared_len: Option<u32>) -> Vec<u8> {
    let length = declared_len.unwrap_or_else(|| u32::try_from(data.len()).expect("fits u32"));
    let mut checked = kind.to_vec();
    checked.extend_from_slice(data);
    let mut out = length.to_be_bytes().to_vec();
    out.extend_from_slice(&checked);
    out.extend_from_slice(&crc32(&checked).to_be_bytes());
    out
}

/// The `iTXt` data layout: keyword NUL flag method language NUL translated NUL text.
fn itxt_data(keyword: &str, compressed: bool, text: &[u8]) -> Vec<u8> {
    let mut data = keyword.as_bytes().to_vec();
    data.push(0);
    data.push(u8::from(compressed));
    data.push(0);
    data.push(0);
    data.push(0);
    data.extend_from_slice(text);
    data
}

/// One `iTXt` chunk framed by `png` itself, under any keyword, compressed or not.
fn framed_itxt(keyword: &str, text: &str, compressed: bool) -> Vec<u8> {
    use png::text_metadata::{EncodableTextChunk, ITXtChunk};

    let mut chunk = ITXtChunk::new(keyword, text);
    chunk.compressed = compressed;
    let mut out = Vec::new();
    chunk.encode(&mut out).expect("the chunk encodes");
    out
}

fn splice_after_ihdr(png: &[u8], chunk: &[u8]) -> Vec<u8> {
    let mut out = png[..AFTER_IHDR].to_vec();
    out.extend_from_slice(chunk);
    out.extend_from_slice(&png[AFTER_IHDR..]);
    out
}

/// Offset of a chunk header of type `kind`, found by walking the framing.
fn chunk_offset(png: &[u8], kind: &[u8; 4]) -> usize {
    let mut cursor = 8;
    loop {
        assert!(
            cursor + 8 <= png.len(),
            "walked off the end looking for kind"
        );
        if &png[cursor + 4..cursor + 8] == kind.as_slice() {
            return cursor;
        }
        let length = usize::try_from(u32::from_be_bytes(
            png[cursor..cursor + 4].try_into().expect("a length field"),
        ))
        .expect("a chunk length fits usize");
        cursor += 12 + length;
    }
}

/// Replace the first IDAT's payload with bytes that are not a zlib stream, re-framing the CRC
/// so the file stays structurally perfect and ONLY the compressed pixel data is unreadable.
///
/// The whole point of the no-decode assertion: a file no decoder can get pixels out of, which
/// the import must nonetheless read a recipe from and store verbatim.
fn break_the_pixel_stream(png: &[u8]) -> Vec<u8> {
    let idat = chunk_offset(png, b"IDAT");
    let length = usize::try_from(u32::from_be_bytes(
        png[idat..idat + 4].try_into().expect("a length field"),
    ))
    .expect("fits usize");
    // 0x00 0x00 fails zlib's header check (CMF/FLG must be divisible by 31), so inflate
    // refuses immediately rather than allocating anything.
    let garbage = vec![0_u8; length];
    let mut out = png[..idat].to_vec();
    out.extend_from_slice(&framed_chunk(b"IDAT", &garbage, None));
    out.extend_from_slice(&png[idat + 12 + length..]);
    out
}

/// Import `bytes` as `name` into `project_id`, staging them the way the API's multipart handler
/// does (a temp file the store consumes).
fn import_bytes(
    store: &ProjectStore,
    project_id: &str,
    staging: &Path,
    name: &str,
    bytes: &[u8],
) -> Result<Value, ProjectStoreError> {
    let source = staging.join(format!(
        "{}-{name}",
        crate::store_util::random_hex(6).expect("hex")
    ));
    std::fs::write(&source, bytes).expect("staged upload writes");
    store.import_asset(
        project_id,
        UploadAsset {
            filename: name.to_owned(),
            content_type: Some("image/png".to_owned()),
            source_path: source,
            source_asset_id: None,
            provenance: None,
        },
    )
}

/// The stub recipe `import_asset` synthesizes for an upload, as it must stay. sc-15949 records
/// a found envelope BESIDE this, never inside it.
fn assert_upload_stub_recipe(asset: &Value, filename: &str) {
    assert_eq!(
        asset["recipe"]["mode"],
        json!("upload"),
        "an uploaded image was uploaded, not generated here"
    );
    assert_eq!(asset["recipe"]["model"], json!("manual-import"));
    assert_eq!(asset["recipe"]["adapter"], json!("api-upload"));
    assert_eq!(asset["recipe"]["prompt"], json!(filename));
    assert_eq!(asset["recipe"]["negativePrompt"], json!(""));
    assert_eq!(asset["recipe"]["seed"], json!(0));
    assert_eq!(asset["recipe"]["loras"], json!([]));
    assert_eq!(asset["recipe"]["stylePreset"], json!("none"));
    assert_eq!(asset["recipe"]["normalizedSettings"], json!({}));
    assert_eq!(asset["origin"], json!("upload"));
    // No generation set and no job: nothing here ran.
    assert_eq!(asset["generationSetId"], Value::Null);
    assert_eq!(asset["lineage"]["jobId"], Value::Null);
}

#[test]
fn import_asset_records_an_embedded_workflow_without_faking_a_generation() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Shared").expect("project creates");
    let staging = temp_dir.path().join("staging");
    std::fs::create_dir_all(&staging).expect("staging dir");

    let share = workflow_fixture("a lighthouse in heavy fog");
    let bytes = sceneworks_png(Some(&share));
    let asset = import_bytes(&store, &project.id, &staging, "shared.png", &bytes)
        .expect("a SceneWorks PNG imports");

    // The envelope is recorded, whole, exactly as the reader sanitized it.
    assert_eq!(
        asset["extra"][IMPORTED_WORKFLOW_KEY],
        serde_json::to_value(&share).expect("the envelope serializes"),
        "the parsed envelope must land in extra.importedWorkflow"
    );
    // …and the record stays honest. The envelope says `text_to_image`; the ASSET does not.
    assert_eq!(
        asset["extra"][IMPORTED_WORKFLOW_KEY]["mode"],
        json!("text_to_image")
    );
    assert_upload_stub_recipe(&asset, "shared.png");
    // Nothing was hoisted out of the envelope into the asset's own record: not the seed we
    // never ran, not the model, not the prompt.
    assert_ne!(asset["recipe"]["seed"], json!(880_412));
    assert_ne!(asset["recipe"]["model"], json!("z_image_turbo"));
    assert_ne!(
        asset["recipe"]["prompt"],
        json!("a lighthouse in heavy fog")
    );
    // The stored bytes are the user's file, untouched (sc-6143's passthrough).
    let project_path = store.find_project_path(&project.id).expect("project path");
    let stored = project_path.join(asset["file"]["path"].as_str().expect("path"));
    assert_eq!(std::fs::read(&stored).expect("read stored"), bytes);

    // `assets.asset_json` is `serde_json::to_string(asset)` over the whole sidecar, so the
    // field rides the existing envelope blob with no schema migration. Proven through the
    // index rather than asserted from the source: read the column back.
    let asset_id = asset["id"].as_str().expect("id").to_owned();
    let indexed: String = Connection::open(project_path.join("project.db"))
        .expect("opens project.db")
        .query_row(
            "select asset_json from assets where id = ?1",
            params![asset_id],
            |row| row.get(0),
        )
        .expect("the row is indexed");
    let indexed: Value = serde_json::from_str(&indexed).expect("asset_json is JSON");
    assert_eq!(
        indexed["extra"][IMPORTED_WORKFLOW_KEY], asset["extra"][IMPORTED_WORKFLOW_KEY],
        "asset_json must round-trip the new field"
    );

    // And it survives a full index rebuild from disk — the sidecar is the source of truth, so
    // hydration must not be the thing that was carrying it.
    store.reindex_project(&project.id).expect("reindex runs");
    let after = store
        .get_asset(&project.id, &asset_id)
        .expect("the asset survives reindex");
    assert_eq!(
        after["extra"][IMPORTED_WORKFLOW_KEY], asset["extra"][IMPORTED_WORKFLOW_KEY],
        "the imported workflow must survive an index rebuild"
    );
    assert_eq!(after["recipe"]["mode"], json!("upload"));
}

#[test]
fn import_asset_without_a_workflow_chunk_behaves_exactly_as_before() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Plain").expect("project creates");
    let staging = temp_dir.path().join("staging");
    std::fs::create_dir_all(&staging).expect("staging dir");

    // The opt-out write path: a real PNG with no chunk at all.
    let plain = sceneworks_png(None);
    let asset = import_bytes(&store, &project.id, &staging, "plain.png", &plain)
        .expect("a plain PNG imports");
    assert!(
        asset.get("extra").is_none(),
        "no chunk must mean no new fields at all, got {:?}",
        asset.get("extra")
    );
    assert_upload_stub_recipe(&asset, "plain.png");

    // And with provenance, `extra` is still the provenance blob and nothing else.
    let provenance = json!({ "editor": "image_editor", "edits": [{ "op": "crop" }] });
    let source = staging.join("edited.png");
    std::fs::write(&source, &plain).expect("staged upload writes");
    let edited = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "edited.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: Some(provenance.clone()),
            },
        )
        .expect("imports");
    assert_eq!(edited["extra"], provenance);
    assert!(edited["extra"].get(IMPORTED_WORKFLOW_KEY).is_none());
}

#[test]
fn import_asset_merges_a_workflow_beside_the_editor_provenance() {
    // The two writers of `extra` have to coexist: an Image Editor save of a shared image
    // carries an edit chain AND a chunk. Neither may evict the other.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Both").expect("project creates");

    let share = workflow_fixture("a lighthouse");
    let source = temp_dir.path().join("both.png");
    std::fs::write(&source, sceneworks_png(Some(&share))).expect("staged upload writes");
    let provenance = json!({
        "editor": "image_editor",
        "edits": [{ "op": "crop" }],
        // A caller-supplied value under OUR key must not win: the field is defined as what the
        // import parsed and sanitized, so a client cannot inject a recipe through provenance.
        IMPORTED_WORKFLOW_KEY: { "sceneworksWorkflow": "image", "prompt": "injected" },
    });
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "both.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: Some(provenance),
            },
        )
        .expect("imports");

    assert_eq!(asset["extra"]["editor"], json!("image_editor"));
    assert_eq!(asset["extra"]["edits"], json!([{ "op": "crop" }]));
    assert_eq!(
        asset["extra"][IMPORTED_WORKFLOW_KEY],
        serde_json::to_value(&share).expect("serializes"),
        "the import's own sanitized envelope must win over one supplied in provenance"
    );

    // The case that made "inserted last" a false claim: the SAME injected provenance, on a file
    // with NO chunk. There is nothing to insert, so ordering protects nothing — the key has to
    // be REMOVED, or a caller's free JSON lands under it untouched. Everything below is what
    // was reaching the sidecar before the removal: a traversal `loras[].repo` in the one field
    // sc-15950/sc-15952 turn into a cache path, an unsanitized absolute `model`, and an
    // `advanced` map the allow-list would have emptied.
    let plain = temp_dir.path().join("plain.png");
    std::fs::write(&plain, sceneworks_png(None)).expect("staged upload writes");
    let injected = json!({
        "editor": "image_editor",
        IMPORTED_WORKFLOW_KEY: {
            "sceneworksWorkflow": "image",
            "schemaVersion": 1,
            "model": "C:\\Users\\Victim\\models\\evil",
            "prompt": "injected",
            "loras": [{ "repo": "../../../../etc/passwd" }],
            "advanced": { "anythingAtAll": { "nested": "free JSON" } },
        },
    });
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "plain.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: plain,
                source_asset_id: None,
                provenance: Some(injected),
            },
        )
        .expect("imports");
    assert_eq!(
        asset["extra"],
        json!({ "editor": "image_editor" }),
        "with no chunk to record, `{IMPORTED_WORKFLOW_KEY}` must be ABSENT — the key is either \
             the envelope this import parsed or nothing, never a caller's free JSON"
    );
    let encoded = serde_json::to_string(&asset["extra"]).expect("serializes");
    for leak in ["etc/passwd", "Victim", "anythingAtAll", "injected"] {
        assert!(
            !encoded.contains(leak),
            "{leak} was injected through provenance and reached the sidecar: {encoded}"
        );
    }
}

#[test]
fn import_asset_refuses_a_provenance_that_is_not_an_object() {
    // `extra` is an object in the sidecar contract, and a scalar `provenance` used to become the
    // whole slot — which, once sc-15949 gave that slot a second writer, silently discarded a
    // workflow the reader had already found in the file. `Some(Value::Null)` is reachable from
    // the API today (any non-blank JSON text was accepted), so this is not hypothetical.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Provenance").expect("project creates");

    let share = workflow_fixture("a lighthouse");
    let embedded = sceneworks_png(Some(&share));
    for provenance in [json!(42), json!("editor"), json!([1, 2]), Value::Null] {
        let source = temp_dir.path().join(format!(
            "scalar-{}.png",
            crate::store_util::random_hex(6).expect("hex")
        ));
        std::fs::write(&source, &embedded).expect("staged upload writes");
        let error = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "scalar.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: source,
                    source_asset_id: None,
                    provenance: Some(provenance.clone()),
                },
            )
            .expect_err("a non-object provenance must be refused, not silently preferred");
        assert!(
            matches!(&error, ProjectStoreError::BadRequest(message)
                    if message.contains("provenance must be a JSON object")),
            "{provenance} produced the wrong error: {error}"
        );
    }

    // An empty object is not a non-object: it keeps its slot, exactly as it did before this
    // story, and still makes room for the envelope beside it.
    let source = temp_dir.path().join("empty-provenance.png");
    std::fs::write(&source, sceneworks_png(None)).expect("staged upload writes");
    let plain = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "empty.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: Some(json!({})),
            },
        )
        .expect("imports");
    assert_eq!(
        plain["extra"],
        json!({}),
        "`provenance: {{}}` wrote `extra: {{}}` before sc-15949 and must still"
    );

    let source = temp_dir.path().join("empty-provenance-with-chunk.png");
    std::fs::write(&source, &embedded).expect("staged upload writes");
    let shared = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "shared.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: Some(json!({})),
            },
        )
        .expect("imports");
    assert_eq!(
        shared["extra"][IMPORTED_WORKFLOW_KEY],
        serde_json::to_value(&share).expect("serializes")
    );
}

#[test]
fn a_hostile_or_malformed_chunk_still_imports_as_a_plain_asset() {
    // The corpus is sc-15947's, reframed at this boundary: every one of these is a file a
    // stranger could hand us, and every one must produce a successful plain import. A bad chunk
    // costs the user the recipe field and NOTHING else — not the upload, not the bytes.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Hostile").expect("project creates");
    let staging = temp_dir.path().join("staging");
    std::fs::create_dir_all(&staging).expect("staging dir");

    let plain = sceneworks_png(None);
    let good = sceneworks_png(Some(&workflow_fixture("a lighthouse")));
    let envelope = json!({
        "sceneworksWorkflow": "image",
        "schemaVersion": 1,
        "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "prompt": "a lighthouse",
    })
    .to_string();

    let mut corpus: Vec<(String, Vec<u8>)> = vec![
        // Two chunks under our keyword: which recipe made the image is ambiguous, so neither.
        (
            "duplicate chunks".to_owned(),
            splice_after_ihdr(
                &good,
                &framed_itxt(crate::workflow_png::WORKFLOW_CHUNK_KEYWORD, &envelope, true),
            ),
        ),
        // A zip bomb: 8 MiB of NUL in a chunk of a few hundred bytes.
        (
            "zip bomb".to_owned(),
            splice_after_ihdr(
                &plain,
                &framed_itxt(
                    crate::workflow_png::WORKFLOW_CHUNK_KEYWORD,
                    &"\0".repeat(8 * 1024 * 1024),
                    true,
                ),
            ),
        ),
        // Uncompressed text past the 1 MiB cap, so it is the text cap that has to catch it.
        (
            "text over the cap".to_owned(),
            splice_after_ihdr(
                &plain,
                &framed_itxt(
                    crate::workflow_png::WORKFLOW_CHUNK_KEYWORD,
                    &"A".repeat(crate::workflow_png::MAX_WORKFLOW_TEXT_BYTES + 1),
                    false,
                ),
            ),
        ),
        // A chunk header claiming the largest length a PNG may declare, in a small file.
        (
            "lying chunk length".to_owned(),
            splice_after_ihdr(
                &plain,
                &framed_chunk(
                    b"iTXt",
                    &itxt_data(crate::workflow_png::WORKFLOW_CHUNK_KEYWORD, false, b"{}"),
                    Some(0x7FFF_FFFF),
                ),
            ),
        ),
        // Text that decodes fine and is not an envelope.
        (
            "not JSON".to_owned(),
            splice_after_ihdr(
                &plain,
                &framed_itxt(
                    crate::workflow_png::WORKFLOW_CHUNK_KEYWORD,
                    "steps: 28, sampler: euler",
                    true,
                ),
            ),
        ),
        (
            "JSON of the wrong shape".to_owned(),
            splice_after_ihdr(
                &plain,
                &framed_itxt(
                    crate::workflow_png::WORKFLOW_CHUNK_KEYWORD,
                    "[1, 2, 3]",
                    true,
                ),
            ),
        ),
        (
            "a schema version from the future".to_owned(),
            splice_after_ihdr(
                &plain,
                &framed_itxt(
                    crate::workflow_png::WORKFLOW_CHUNK_KEYWORD,
                    &envelope.replace("\"schemaVersion\":1", "\"schemaVersion\":9999"),
                    true,
                ),
            ),
        ),
        // A ComfyUI / Automatic1111 export: text chunks, none of them ours.
        (
            "a foreign writer's chunks".to_owned(),
            splice_after_ihdr(&plain, &framed_itxt("parameters", "steps: 28", true)),
        ),
    ];
    // A file whose pixels are gone but whose chunk is intact, and the same with no chunk.
    corpus.push((
        "broken pixels, good chunk".to_owned(),
        break_the_pixel_stream(&good),
    ));
    corpus.push((
        "broken pixels, no chunk".to_owned(),
        break_the_pixel_stream(&plain),
    ));
    // Truncations of a file that DOES carry a workflow. A partial chunk must never yield a
    // partial envelope; a cut past the chunk legitimately still reads it.
    for cut in (AFTER_IHDR..good.len()).step_by(11) {
        corpus.push((format!("truncated at {cut}"), good[..cut].to_vec()));
    }
    // And a bit flipped in each region of the same file.
    for offset in (8..good.len()).step_by(23) {
        let mut corrupt = good.clone();
        corrupt[offset] ^= 0xFF;
        corpus.push((format!("bit flipped at {offset}"), corrupt));
    }
    // The corpus size, derived and printed rather than described in prose — a claim about it was
    // wrong by nearly 2x ("~110 imports") because nobody ever multiplied it out.
    //
    // NOT asserted as a literal, and that is deliberate: the count is a function of the fixture
    // PNG's byte length, which is the length of a DEFLATE stream produced by whatever compressor
    // the `png` crate links. It is 492 bytes here and has been observed at 417, and neither
    // number is a property of this test. So the derivation is asserted and the count is printed;
    // a run's own output is the only honest source for it.
    let truncations = (AFTER_IHDR..good.len()).step_by(11).count();
    let flips = (8..good.len()).step_by(23).count();
    assert_eq!(
        corpus.len(),
        8 + 2 + truncations + flips,
        "8 named fixtures + 2 broken-pixel files + one truncation every 11 bytes + one bit flip \
             every 23 bytes"
    );
    println!(
        "sc-15949: {} imports over the corpus ({} byte fixture: {truncations} truncations, \
             {flips} bit flips)",
        corpus.len(),
        good.len()
    );

    let intact = serde_json::to_value(workflow_fixture("a lighthouse")).expect("serializes");
    for (label, bytes) in corpus {
        let asset = import_bytes(&store, &project.id, &staging, "hostile.png", &bytes)
            .unwrap_or_else(|error| panic!("{label}: the upload must still succeed ({error})"));
        // The user got their image, byte for byte.
        let project_path = store.find_project_path(&project.id).expect("project path");
        let stored = project_path.join(asset["file"]["path"].as_str().expect("path"));
        assert_eq!(
            std::fs::read(&stored).expect("read stored"),
            bytes,
            "{label}: the stored bytes must be the file the user handed us"
        );
        assert_upload_stub_recipe(&asset, "hostile.png");
        // And either no workflow at all, or the intact one — never a partial or malformed
        // envelope pieced out of a damaged file.
        match asset.get("extra") {
            None => {}
            Some(extra) => {
                let recorded = &extra[IMPORTED_WORKFLOW_KEY];
                assert_eq!(
                    recorded, &intact,
                    "{label}: a damaged file produced a workflow that is not the intact one"
                );
            }
        }
    }
}

#[test]
fn path_shaped_values_in_a_hostile_chunk_do_not_survive_the_import() {
    // The trust boundary, at this boundary. Every classified string in the envelope below is a
    // location: absolute POSIX and Windows paths, a UNC share, `file://`, `~/`, and a traversal
    // dressed up as a Hugging Face repo id. None of them may reach `extra.importedWorkflow` —
    // sc-15952 turns `loras[].repo` into a cache path and renders the rest.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Paths").expect("project creates");

    let hostile = json!({
        "sceneworksWorkflow": "image",
        "schemaVersion": 1,
        "producer": {
            "name": "/opt/sceneworks/bin",
            "url": "file:///D:/Users/Michael/secret.html",
            "version": "0.8.1-dirty-Michael",
        },
        "mode": "/etc/passwd",
        "model": "C:\\Users\\Michael\\models\\z_image",
        "prompt": "a lighthouse",
        "styleId": "file%3A%2F%2FD%3A%2Fsecret",
        "stylePreset": "\\\\fileserver\\share\\styles\\noir",
        "fitMode": "~/models/fit",
        "upscale": { "enabled": true, "engine": "/usr/local/bin/realesrgan" },
        "loras": [
            { "name": "D:\\loras\\private.safetensors", "repo": "../../../../etc/passwd" },
            { "name": "Ghibli watercolor", "repo": "acme/ghibli", "weight": 0.8 },
        ],
        "inputs": [
            { "kind": "source/../../etc", "count": 1 },
            { "kind": "source", "count": 1, "controlMode": "/dev/null" },
        ],
        "advanced": {
            "sampler": "C:\\Windows\\System32\\euler",
            "steps": 8,
        },
    })
    .to_string();
    let bytes = splice_after_ihdr(
        &sceneworks_png(None),
        &framed_itxt(crate::workflow_png::WORKFLOW_CHUNK_KEYWORD, &hostile, true),
    );
    let source = temp_dir.path().join("hostile.png");
    std::fs::write(&source, &bytes).expect("staged upload writes");
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "hostile.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("a hostile chunk must not fail the upload");
    let recorded = &asset["extra"][IMPORTED_WORKFLOW_KEY];

    // Reduced to empty rather than echoed: these are the fields sc-15952 shows as provenance.
    assert_eq!(recorded["producer"]["name"], json!(""));
    assert_eq!(recorded["producer"]["url"], json!(""));
    assert_eq!(recorded["producer"]["version"], json!(""));
    assert_eq!(recorded["mode"], json!(""));
    assert_eq!(recorded["model"], json!(""));
    // Optional labels are dropped outright, key and all.
    for dropped in ["styleId", "stylePreset", "fitMode"] {
        assert!(
            recorded.get(dropped).is_none(),
            "{dropped} was path-shaped and must not have survived: {:?}",
            recorded.get(dropped)
        );
    }
    assert_eq!(recorded["upscale"]["enabled"], json!(true));
    assert!(recorded["upscale"].get("engine").is_none());
    // The traversal LoRA is dropped entirely (name and repo both refused, nothing left to
    // say); the legitimate one beside it survives, so this is not passing by dropping all.
    assert_eq!(
        recorded["loras"],
        json!([{ "name": "Ghibli watercolor", "repo": "acme/ghibli", "weight": 0.8 }])
    );
    // A kind outside the closed vocabulary is dropped; a path in `controlMode` is stripped.
    assert_eq!(
        recorded["inputs"],
        json!([{ "kind": "source", "count": 1 }])
    );
    // The `advanced` allow-list is shape-checked as well as key-checked.
    assert_eq!(recorded["advanced"], json!({ "steps": 8 }));

    // Nothing path-shaped anywhere in the recorded envelope — asserted over the whole tree
    // rather than field by field, so a field added later cannot quietly open a hole. The
    // authored PROSE fields are exempt by the sc-15946 contract (a prompt that names a
    // directory is the user's own text and mangling it would be the worse failure), so the
    // sweep skips them by KEY, exactly as the sanitizer does.
    fn assert_no_locations(value: &Value, path: &str) {
        match value {
            Value::String(text) => assert!(
                !crate::workflow_share::is_path_shaped(text),
                "{path} survived as a location: {text:?}"
            ),
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    assert_no_locations(item, &format!("{path}[{index}]"));
                }
            }
            Value::Object(map) => {
                for (key, item) in map {
                    if matches!(
                        key.as_str(),
                        "prompt"
                            | "negativePrompt"
                            | "stylePrompt"
                            | "intent"
                            | "runtimePrompt"
                            | "systemMessage"
                    ) {
                        continue;
                    }
                    assert_no_locations(item, &format!("{path}.{key}"));
                }
            }
            _ => {}
        }
    }
    assert_no_locations(recorded, "importedWorkflow");
    // The record itself is still an upload.
    assert_upload_stub_recipe(&asset, "hostile.png");
}

/// Every string in these envelopes is legitimate; only the volume is hostile.
///
/// sc-15947's 1 MiB cap is on the DECOMPRESSED chunk text, and a repetitive envelope compresses
/// ~100x — so what an attacker pays is the compressed chunk and what we pay is the decompressed
/// envelope, persisted into the sidecar, the `assets.asset_json` row and every `list_assets`
/// response for that asset (sc-14797 / sc-14798) forever.
///
/// The history is why this is a table of cases rather than one fixture. The first fix was
/// per-collection caps, measured on a fixture whose own size fell with the fix — so the reported
/// "111x -> 8x" was an artifact of a shrinking denominator, and the real worst case at that point
/// was WORSE than before. Each row below is a vector found after the caps were in place, and each
/// is measured in ABSOLUTE bytes at both ends:
///
/// | case | PNG | recorded before | recorded after |
/// |------|-----|-----------------|----------------|
/// | six prose slots of U+1F600 x the cap | 3,633 | 480,321 | 98,654 |
/// | 112,128 `null` coordinates over 64 poses | 5,630 | 681,995 | 98,671 |
/// | six ASCII prose slots, nothing over any cap | 1,668 | 120,321 | 98,654 |
/// | 200,000 empty arrays under one `keypoints` | 4,589 | 720,347 | 98,667 |
/// | every bound at once, poses at the slot budget | 3,045 | 221,407 | none recorded |
///
/// Measured by running THIS test with the guards reverted one set at a time, not from a separate
/// harness, so the "before" column is the same fixture as the "after" one. The `list_assets`
/// payload for one asset tracks the recorded envelope within a kilobyte in every row.
///
/// The ratio is deliberately not the assertion. The PNG carries the COMPRESSED envelope, so a
/// repetitive payload will always have a large ratio; what is bounded is the absolute size of
/// what we persist, by `workflow_share::WORKFLOW_SHARE_MAX_BYTES`, and that is what is asserted.
#[test]
fn a_chunk_cannot_amplify_itself_into_the_asset_index() {
    /// The recorded envelope must stay under this many bytes, in every case below.
    ///
    /// `WORKFLOW_SHARE_MAX_BYTES` plus the sidecar's own fields, asserted against the shipped
    /// constant rather than chosen — a ceiling this test picked for itself would drift away from
    /// the one the reader enforces. The old form of this assertion was 200 KiB and passed for the
    /// wrong reason: the fixture used three ASCII prose slots, so the six-slot multi-byte case it
    /// was meant to cover was 480 kB and never ran.
    const CEILING: usize = crate::workflow_share::WORKFLOW_SHARE_MAX_BYTES + 4 * 1024;

    let temp_dir = tempfile::tempdir().expect("temp dir");
    // 4-byte scalars, so the six prose slots are as wide in BYTES as the bound allows. This is
    // the case the char-counted bound missed entirely.
    let emoji = "\u{1F600}".repeat(20_000);
    let ascii = "z".repeat(20_000);
    // 1,752 nulls x 64 poses = 112,128 coordinate slots, none of which cost anything against a
    // budget that counted only `Value::Number`.
    let nulls: Vec<Value> = (0..1_752).map(|_| Value::Null).collect();
    let null_poses: Vec<Value> = (0..64)
        .map(|_| json!({ "keypoints": nulls.clone() }))
        .collect();
    // `[]` is vacuously "all coordinates", so it passed the shape check AND cost nothing.
    let empty_arrays: Vec<Value> = (0..200_000).map(|_| json!([])).collect();
    // Every bound at once: six prose slots, every allow-listed scalar label at its own cap, a
    // pose set at the slot budget, a full phase schedule. Each field is individually legal, and
    // together they are what per-field bounds cannot see.
    let label = "L".repeat(200);
    let point = json!([0.123_456_789_012_345_67, 0.987_654_321_098_765_4, 0.55]);
    let skeleton: Vec<Value> = (0..128).map(|_| point.clone()).collect();
    let mut maxed = crate::contracts::JsonObject::new();
    for key in [
        "resolution",
        "sampler",
        "scheduler",
        "schedulerShift",
        "steps",
        "guidanceScale",
        "guidanceMethod",
        "enhancePrompt",
        "usePid",
        "pidTarget",
        "ipAdapterScale",
        "controlnetConditioningScale",
        "trueCfgScale",
        "strength",
        "textStyleGain",
        "viewAngle",
        "faceRestore",
        "controlMode",
        "controlScale",
        "styleId",
        "cnScale",
        "angleSet",
        "imageGuidanceScale",
    ] {
        maxed.insert(key.to_owned(), json!(label));
    }
    maxed.insert(
        "poses".to_owned(),
        json!((0..16)
            .map(|_| json!({ "keypoints": skeleton.clone() }))
            .collect::<Vec<Value>>()),
    );
    maxed.insert(
        "phases".to_owned(),
        json!((0..8)
            .map(|_| json!({
                "steps": 4,
                "guidance": 3.5,
                "loras": (0..5)
                    .map(|index| json!({ "index": index, "weight": 0.123_456_789 }))
                    .collect::<Vec<Value>>()
            }))
            .collect::<Vec<Value>>()),
    );

    let cases: Vec<(&str, &String, crate::contracts::JsonObject)> = vec![
        (
            "six 4-byte prose slots",
            &emoji,
            crate::contracts::JsonObject::new(),
        ),
        (
            "112,128 null coordinates",
            &ascii,
            json!({ "poses": null_poses })
                .as_object()
                .cloned()
                .expect("object"),
        ),
        (
            "six ASCII prose slots",
            &ascii,
            crate::contracts::JsonObject::new(),
        ),
        (
            "200,000 empty arrays",
            &ascii,
            json!({ "poses": [{ "keypoints": empty_arrays }] })
                .as_object()
                .cloned()
                .expect("object"),
        ),
        ("every bound at once", &ascii, maxed),
    ];

    for (label, prose, extra) in cases {
        // A store per case, so the `list_assets` figure is what ONE asset costs a listing rather
        // than the running total of every case before it.
        let store = ProjectStore::new(
            temp_dir
                .path()
                .join(crate::store_util::random_hex(6).expect("hex")),
            "test-version",
        );
        let project = store.create_project("Amplify").expect("project creates");
        let mut advanced = json!({
            "steps": 8,
            "stylePrompt": prose,
            "systemMessage": prose,
            "structuredPrompt": { "intent": prose, "runtimePrompt": prose },
        })
        .as_object()
        .cloned()
        .expect("object");
        advanced.extend(extra);
        let envelope = json!({
                "sceneworksWorkflow": "image",
                "schemaVersion": 1,
                "producer": { "name": "SceneWorks", "url": "https://example.invalid", "version": "0.8.1" },
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": prose,
                "negativePrompt": prose,
                // Identical entries, because that is the shape that maximizes the compression ratio
                // and so the leverage.
                // 500 rather than 8,000 only so the widest prose case still fits under sc-15947's
                // 1 MiB text cap; 500 is 100x `MAX_SHARE_LORAS` and 125x `MAX_SHARE_INPUTS`, and the
                // unit tests in `workflow_share` run the 8,000 case.
                "loras": (0..500)
                    .map(|_| json!({ "name": "Ghibli watercolor", "repo": "acme/ghibli", "weight": 0.8 }))
                    .collect::<Vec<Value>>(),
                "inputs": (0..500)
                    .map(|_| json!({ "kind": "reference", "count": 1 }))
                    .collect::<Vec<Value>>(),
                "advanced": advanced,
            })
            .to_string();
        assert!(
            envelope.len() < crate::workflow_png::MAX_WORKFLOW_TEXT_BYTES,
            "{label}: the fixture must be UNDER the text cap ({} bytes) or the reader refuses it \
                 and these bounds are never reached",
            envelope.len()
        );
        let bytes = splice_after_ihdr(
            &sceneworks_png(None),
            &framed_itxt(crate::workflow_png::WORKFLOW_CHUNK_KEYWORD, &envelope, true),
        );
        let source = temp_dir.path().join(format!(
            "{}.png",
            crate::store_util::random_hex(6).expect("hex")
        ));
        std::fs::write(&source, &bytes).expect("staged upload writes");
        let asset = store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: "amplify.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: source,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("a bounded envelope still imports");

        let recorded = &asset["extra"][IMPORTED_WORKFLOW_KEY];
        // Measured and PRINTED before anything is asserted, so reverting a bound to check this
        // test still bites also reports what the amplification was.
        let recorded_bytes = serde_json::to_string(recorded).expect("serializes").len();
        let payload_bytes = serde_json::to_string(
            &store
                .list_assets(&project.id, false, false, AssetScope::All)
                .expect("lists"),
        )
        .expect("serializes")
        .len();
        let asset_json_bytes: String = Connection::open(
            store
                .find_project_path(&project.id)
                .expect("project path")
                .join("project.db"),
        )
        .expect("opens project.db")
        .query_row(
            "select asset_json from assets where id = ?1",
            params![asset["id"].as_str().expect("id")],
            |row| row.get(0),
        )
        .expect("the row is indexed");
        println!(
            "sc-15949 [{label}]: {} byte PNG -> {recorded_bytes} byte workflow, {} byte \
                 asset_json, {payload_bytes} byte list_assets payload",
            bytes.len(),
            asset_json_bytes.len()
        );

        for (what, measured) in [
            ("the recorded envelope", recorded_bytes),
            ("asset_json", asset_json_bytes.len()),
            ("the list_assets payload for one asset", payload_bytes),
        ] {
            assert!(
                measured < CEILING,
                "{label}: {what} is {measured} bytes, over the {CEILING} byte ceiling"
            );
        }

        if recorded.is_null() {
            // "Every bound at once" is over `WORKFLOW_SHARE_MAX_BYTES` and degrades to NO
            // workflow rather than to a partial record, which is the whole point of a ceiling on
            // the envelope instead of another cap on a field.
            assert!(
                asset["extra"].get(IMPORTED_WORKFLOW_KEY).is_none(),
                "an over-ceiling envelope must leave the key absent, not record a fragment"
            );
            continue;
        }
        // Every unbounded collection is gone; the recipe around them still travels, so this is
        // not passing by refusing the envelope wholesale.
        assert!(
            recorded.get("loras").is_none(),
            "{label}: {:?}",
            recorded.get("loras")
        );
        assert!(recorded.get("inputs").is_none());
        // `advanced.poses` is bounded rather than absent, because a pose entry whose coordinates
        // were refused still occupies its slot as `{}` — the entry count is load-bearing (poses
        // replace `count` variations). What must not survive is the VOLUME.
        // `advanced.poses` is bounded rather than absent, because a pose entry whose coordinates
        // were refused still occupies its slot as `{}` — the entry count is load-bearing (poses
        // replace `count` variations). What must not survive is the VOLUME.
        if let Some(poses) = recorded["advanced"].get("poses") {
            let poses_bytes = serde_json::to_string(poses).expect("serializes").len();
            assert!(
                poses_bytes < 4 * 1024,
                "{label}: advanced.poses is {poses_bytes} bytes"
            );
        }
        assert_eq!(recorded["advanced"]["steps"], json!(8));
        assert_eq!(recorded["model"], json!("z_image_turbo"));
        // And the drops are LEGIBLE rather than silent: an absent `loras` and a genuinely
        // LoRA-free recipe serialize identically, so the envelope says which one this is.
        let omitted: Vec<&str> = recorded["omitted"]
            .as_array()
            .expect("the omission marker travels")
            .iter()
            .map(|name| name.as_str().expect("a name"))
            .collect();
        assert!(
            omitted.contains(&"loras") && omitted.contains(&"inputs"),
            "{label}: {omitted:?}"
        );
    }
}

/// The measurement behind the table in [`the_workflow_scan_reads_chunks_and_never_the_pixels`].
///
/// Ignored because it writes hundreds of megabytes and asserts a shape rather than a threshold —
/// but it is IN the tree rather than done out of band, because that table has been wrong once
/// already. Refresh it with:
///
/// ```text
/// cargo test -p sceneworks-core --release --lib the_cost_of_the_scan_against_file_size \
///     -- --ignored --nocapture
/// ```
///
/// Two files per size: one carrying the chunk (the SceneWorks case, answered in pass one) and one
/// carrying none (every foreign PNG, which walks the whole file). Incompressible pixels, so the
/// bytes on disk are the bytes the scan has to move past — a smooth render compresses away and
/// makes the miss look far cheaper than it is on a real photograph.
#[test]
#[ignore = "measurement: writes >200 MB of PNG; run explicitly to refresh the table"]
fn the_cost_of_the_scan_against_file_size() {
    /// A cheap LCG, so "incompressible" needs no dependency.
    fn noise(side: u32) -> image::RgbImage {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        image::RgbImage::from_fn(side, side, |_, _| {
            let mut channel = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                u8::try_from((state >> 33) & 0xFF).expect("masked to a byte")
            };
            image::Rgb([channel(), channel(), channel()])
        })
    }
    fn elapsed_micros(mut work: impl FnMut()) -> u128 {
        // Three passes, best of; the first pays for the page cache.
        (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                work();
                start.elapsed().as_micros()
            })
            .min()
            .expect("three passes")
    }

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let share = workflow_fixture("a lighthouse");
    println!("| side | bytes | scan, chunk found | scan, no chunk | plain read | FULL DECODE |");
    for side in [1024_u32, 2048, 4096, 6144] {
        let pixels = noise(side);
        for (label, embedded) in [("hit", true), ("miss", false)] {
            let path = temp_dir.path().join(format!("{side}-{label}.png"));
            crate::workflow_png::write_workflow_chunk(&pixels, &path, embedded.then_some(&share))
                .expect("writes");
            let bytes = std::fs::metadata(&path).expect("metadata").len();
            let scan = elapsed_micros(|| {
                let _ = crate::workflow_png::read_workflow_chunk_file(&path);
            });
            let read = elapsed_micros(|| {
                let _ = std::fs::read(&path).expect("reads");
            });
            let decode = elapsed_micros(|| {
                let _ = image::open(&path).expect("decodes");
            });
            println!(
                "| {side}² {label} | {bytes} | {scan} µs | plain read {read} µs | decode \
                     {decode} µs | scan/read {:.2}x | scan/decode {:.0}% |",
                scan as f64 / read as f64,
                100.0 * scan as f64 / decode as f64
            );
        }
    }
}

#[test]
fn the_workflow_scan_reads_chunks_and_never_the_pixels() {
    // The regression this guards is a quiet one: making the scan work by decoding the image.
    // `normalize_image_upload` exists to store a natively-supported upload WITHOUT a decode,
    // and a chunk walk that inflated the pixel stream would turn every upload into one.
    //
    // Proven by contradiction on a file whose pixels CANNOT be decoded: framing and CRCs are
    // perfect, the workflow chunk is intact, and the IDAT payload is not a zlib stream at all.
    // Any decode on this path would fail. The import succeeds and reads the recipe, so nothing
    // decoded.
    //
    // Measured alongside it by `the_cost_of_the_scan_against_file_size` below (release,
    // incompressible pixels so the bytes on disk are bytes the scan must move past — a smooth
    // render compresses away and flatters the miss):
    //
    // |         |    on disk | scan, chunk found | scan, no chunk | plain read | FULL DECODE |
    // |---------|------------|-------------------|----------------|------------|-------------|
    // | 1024²   |    3.0 MiB |             44 µs |         870 µs |     681 µs |     1.39 ms |
    // | 2048²   |   12.0 MiB |             89 µs |        3.99 ms |    2.64 ms |     5.29 ms |
    // | 4096²   |   48.0 MiB |             96 µs |       17.08 ms |   13.45 ms |    24.77 ms |
    // | 6144²   |  108.0 MiB |            101 µs |       41.04 ms |   30.53 ms |    57.03 ms |
    //
    // Two different curves, and only one of them is flat.
    //
    // The HIT — a SceneWorks image — answers in pass one and stops at the first IDAT: 44 µs to
    // 101 µs across a 36x range of file size. That is the structural property this test defends,
    // and it holds: no pixel inflate, and nothing allocated in proportion to the pixels.
    //
    // The MISS — every foreign PNG — is O(file), because sc-15947's reader makes a second pass
    // over the framing past IDAT to be sure there is no late chunk and no duplicate. It costs
    // ~1.3x a plain read of the same file and 63-76% of a full decode; at 48 MiB it is 70%.
    // Roughly 0.36 s per GB imported, and NET-NEW: `move_or_copy_file` renames the upload rather
    // than reading it, and nothing else on the import path touches the image media, so before
    // sc-15949 a large foreign PNG was never read end to end at all.
    //
    // So the honest claim is narrow: the scan is not a decode and does not scale with pixels on
    // the SceneWorks path, and on the foreign path it is one sequential pass — cheaper than
    // decoding, but the same order, not the "2-3%" an earlier version of this comment claimed
    // from small compressible renders.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("NoDecode").expect("project creates");
    let staging = temp_dir.path().join("staging");
    std::fs::create_dir_all(&staging).expect("staging dir");

    let share = workflow_fixture("a lighthouse");
    let undecodable = break_the_pixel_stream(&sceneworks_png(Some(&share)));

    // First: the file really is undecodable, so the assertion below has teeth.
    let probe = staging.join("probe.png");
    std::fs::write(&probe, &undecodable).expect("writes");
    assert!(
        image::open(&probe).is_err(),
        "the fixture must be undecodable or it proves nothing about decoding"
    );

    // The scan alone finds the workflow in it.
    assert_eq!(
        crate::workflow_png::read_workflow_chunk_file(&probe),
        Ok(Some(share.clone())),
        "the chunk walk must not need the pixels"
    );

    // And so does the import, which stores the bytes verbatim.
    let asset = import_bytes(
        &store,
        &project.id,
        &staging,
        "undecodable.png",
        &undecodable,
    )
    .expect("an undecodable-but-well-framed PNG still imports");
    assert_eq!(
        asset["extra"][IMPORTED_WORKFLOW_KEY],
        serde_json::to_value(&share).expect("serializes")
    );
    let project_path = store.find_project_path(&project.id).expect("project path");
    let stored = project_path.join(asset["file"]["path"].as_str().expect("path"));
    assert_eq!(std::fs::read(&stored).expect("read stored"), undecodable);
    // Dimensions stay unread, as they are today: nothing opened the image to fill them in.
    assert_eq!(asset["file"]["width"], Value::Null);
    assert_eq!(asset["file"]["height"], Value::Null);
}

#[test]
fn the_scan_precedes_both_the_passthrough_and_the_transcode() {
    // The ordering AC, pinned where it is visible. A transcode re-encodes from pixels and drops
    // every ancillary chunk (the test below proves that on real bytes), so a scan placed after
    // it could only ever report an absence — and a scan placed after the natively-supported
    // `return` would never run for a PNG at all. Both are one moved line away, and neither is
    // observable end to end today: the sc-15947 reader is PNG-only, so no format that TAKES the
    // transcode branch can carry our chunk in the first place. Source structure is where the
    // guarantee lives, so that is what this asserts — the same technique
    // `the_reader_has_exactly_one_parse_path` uses in `tests/workflow_png.rs`.
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("project_store")
            .join("uploads.rs"),
    )
    .expect("reads this module");
    let body = source
        .split("fn normalize_image_upload(")
        .nth(1)
        .expect("the function is here")
        .split("\nfn ")
        .next()
        .expect("the function ends");
    let code: String = body
        .lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<&str>>()
        .join("\n");

    let scan = code
        .find("read_upload_workflow(source_path)")
        .expect("normalize_image_upload no longer scans the ORIGINAL upload for a workflow");
    let passthrough = code
        .find("kind.is_natively_supported()")
        .expect("the natively-supported branch moved");
    let transcode = code
        .find("transcode_to_png(")
        .expect("the transcode call moved");
    assert!(
        scan < passthrough,
        "the chunk scan must run BEFORE the natively-supported passthrough returns, or a PNG \
             — the only format that carries our chunk — is never looked at"
    );
    assert!(
        scan < transcode,
        "the chunk scan must run BEFORE the transcode, which re-encodes from pixels and drops \
             every ancillary chunk"
    );
}

#[test]
fn a_transcode_destroys_the_chunk_so_only_a_pre_transcode_scan_can_find_it() {
    // The premise the ordering rests on, measured on real bytes with the real transcoder rather
    // than asserted from the AVIF/HEIC docs. Skipped where no transcoder is installed (CI
    // Linux/Windows images do not all carry ffmpeg); macOS always has `sips`.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let share = workflow_fixture("a lighthouse");
    let embedded = temp_dir.path().join("embedded.png");
    std::fs::write(&embedded, sceneworks_png(Some(&share))).expect("writes");
    assert_eq!(
        crate::workflow_png::read_workflow_chunk_file(&embedded),
        Ok(Some(share)),
        "the pre-transcode file carries the workflow"
    );

    let transcoded = temp_dir.path().join("transcoded.png");
    if let Err(error) = crate::media_convert::transcode_to_png(&embedded, &transcoded) {
        println!(
            "sc-15949: no image transcoder reachable on this host ({error}); skipping the \
                 metadata-loss assertion"
        );
        return;
    }
    println!("sc-15949: the real transcoder ran; asserting the chunk did not survive it");
    assert_eq!(
        crate::workflow_png::read_workflow_chunk_file(&transcoded),
        Ok(None),
        "a transcode re-encodes from pixels, so the chunk cannot survive it — which is why the \
             scan has to happen first"
    );
}

#[test]
fn the_training_dataset_lane_opts_out_of_the_scan() {
    // A dataset item has no `extra` and no reader for a recipe, so it must not pay for a
    // metadata walk per image on a bulk import. Asserted on `normalize_image_upload` directly,
    // since the opt-out is invisible in the dataset record by definition.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let source = temp_dir.path().join("embedded.png");
    std::fs::write(
        &source,
        sceneworks_png(Some(&workflow_fixture("a lighthouse"))),
    )
    .expect("writes");

    let read = normalize_image_upload(
        &source,
        "image/png",
        "embedded.png",
        temp_dir.path(),
        WorkflowScan::Read,
    )
    .expect("normalizes");
    assert!(read.workflow.is_some(), "the reading lane finds the chunk");

    let skipped = normalize_image_upload(
        &source,
        "image/png",
        "embedded.png",
        temp_dir.path(),
        WorkflowScan::Skip,
    )
    .expect("normalizes");
    assert!(
        skipped.workflow.is_none(),
        "Skip must not scan, even when the file does carry a workflow"
    );
    // Both lanes agree on everything else — the scan is the only difference.
    assert_eq!(read.content_type, skipped.content_type);
    assert_eq!(read.extension, skipped.extension);
    assert_eq!(read.source_path, skipped.source_path);
}

/// sc-13517: the saved-voice id→clip hop end-to-end, cheaply. A saved voice's stored
/// `referenceAudioAssetId` resolves back to the exact library clip on disk through
/// `resolve_asset_media_path` — the link the register + generate flow relies on, proven directly
/// rather than only by composition in the release DoD render. `resolve_asset_media_path` is
/// media-type agnostic (it resolves the sidecar's `file.path`), so an imported image asset
/// exercises the identical id→path resolution the audio reference clip uses.
#[test]
fn saved_voice_reference_id_resolves_to_the_clip_path() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Voices").expect("project creates");

    let src = temp_dir.path().join("clip.png");
    std::fs::write(&src, b"\x89PNG reference clip bytes").expect("source writes");
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "clip.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: src,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("asset imports");
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let (voice, _dup) = store
        .create_saved_voice(
            &project.id,
            crate::voice_store::SavedVoiceCreateInput {
                name: "Narrator".to_owned(),
                reference_audio_asset_id: asset_id.clone(),
                embedding: vec![0.1; crate::voice_store::VOICE_EMBEDDING_DIM],
            },
            crate::voice_store::DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("register saved voice");
    // The store persisted the reference pointer verbatim.
    let stored_ref = voice["referenceAudioAssetId"].as_str().expect("ref id");
    assert_eq!(stored_ref, asset_id);

    // The stored referenceAudioAssetId resolves to the real clip file on disk.
    let resolved = store
        .resolve_asset_media_path(&project.id, stored_ref)
        .expect("resolve clip");
    assert!(
        resolved.is_file(),
        "resolved clip path must exist: {}",
        resolved.display()
    );
    let expected_rel = store
        .get_asset(&project.id, &asset_id)
        .expect("asset")
        .pointer("/file/path")
        .and_then(Value::as_str)
        .expect("file path")
        .to_owned();
    assert!(
        resolved.ends_with(&expected_rel),
        "resolved {} must end with the asset's relative path {expected_rel}",
        resolved.display()
    );
}

/// sc-13517: `resolve_asset_media_path` resolves a real asset AND rejects a path-traversal id
/// (the user-controlled `../` escape vector) before joining it into any filesystem path — the
/// direct unit test the review flagged as missing.
#[test]
fn resolve_asset_media_path_resolves_and_rejects_traversal() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Guard").expect("project creates");

    let src = temp_dir.path().join("clip.png");
    std::fs::write(&src, b"\x89PNG bytes").expect("source writes");
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "clip.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: src,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("asset imports");
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    // Safe id → a real file under the project directory.
    let resolved = store
        .resolve_asset_media_path(&project.id, &asset_id)
        .expect("resolve safe id");
    assert!(resolved.is_file());
    assert!(resolved.starts_with(temp_dir.path().canonicalize().unwrap()));

    // A `../` escape id is rejected outright (never joined into a path).
    for escape in ["../../etc/passwd", "..", "a/../../b"] {
        let result = store.resolve_asset_media_path(&project.id, escape);
        assert!(
            matches!(result, Err(ProjectStoreError::BadRequest(_))),
            "traversal id {escape:?} must be rejected, got {result:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn resolve_asset_media_path_rejects_replaced_media_symlink() {
    let temporary = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temporary.path().join("data"), "test");
    let project = store.create_project("Symlink boundary").unwrap();
    let source = temporary.path().join("source.png");
    fs::write(&source, b"fixture").unwrap();
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .unwrap();
    let id = asset["id"].as_str().unwrap();
    let media = store.resolve_asset_media_path(&project.id, id).unwrap();
    let outside = temporary.path().join("outside.png");
    fs::write(&outside, b"outside").unwrap();
    fs::remove_file(&media).unwrap();
    std::os::unix::fs::symlink(&outside, &media).unwrap();
    assert!(matches!(
        store.resolve_asset_media_path(&project.id, id),
        Err(ProjectStoreError::BadRequest(_))
    ));
    assert!(store
        .project_file(
            &project.id,
            asset.pointer("/file/path").and_then(Value::as_str).unwrap()
        )
        .is_err());
}

/// sc-13517: `saved_voices` rows are DB-authoritative and MUST survive a reindex. The reindex
/// rebuilds the sidecar-backed tables (assets / generation_sets / timelines / characters) but must
/// leave `saved_voices` untouched — this guards that claim against a future schema bump that runs
/// the reindex on every existing project.
#[test]
fn saved_voices_survive_reindex() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Reindex").expect("project creates");

    let (voice, _dup) = store
        .create_saved_voice(
            &project.id,
            crate::voice_store::SavedVoiceCreateInput {
                name: "Narrator".to_owned(),
                reference_audio_asset_id: "asset_ref".to_owned(),
                embedding: vec![0.5; crate::voice_store::VOICE_EMBEDDING_DIM],
            },
            crate::voice_store::DEFAULT_VOICE_DEDUP_THRESHOLD,
        )
        .expect("register saved voice");
    let voice_id = voice["id"].as_str().expect("voice id").to_owned();
    assert_eq!(store.list_saved_voices(&project.id).expect("list").len(), 1);

    // An explicit reindex rebuilds the sidecar tables from disk.
    store.reindex_project(&project.id).expect("reindex runs");

    let after = store
        .list_saved_voices(&project.id)
        .expect("list after reindex");
    assert_eq!(after.len(), 1, "the saved voice must survive reindex");
    assert_eq!(after[0]["id"].as_str(), Some(voice_id.as_str()));
    assert_eq!(after[0]["name"].as_str(), Some("Narrator"));
}

/// sc-6143: a natively-supported image is stored byte-for-byte (no transcode, no re-encode),
/// even when its declared content type is wrong — classification is by content.
#[test]
fn import_asset_leaves_supported_format_unchanged() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Keep").expect("project creates");

    // Valid PNG signature + arbitrary tail; declared (wrongly) as AVIF.
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    bytes.extend_from_slice(b"the rest of a png");
    let src = temp_dir.path().join("misnamed.avif");
    std::fs::write(&src, &bytes).expect("source writes");

    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "misnamed.avif".to_owned(),
                content_type: Some("image/avif".to_owned()),
                source_path: src,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("imports");

    // Recorded as PNG (its real format), and the stored bytes are untouched.
    assert_eq!(asset["file"]["mimeType"], json!("image/png"));
    let project_path = store.find_project_path(&project.id).expect("project path");
    let stored = project_path.join(asset["file"]["path"].as_str().expect("path"));
    assert_eq!(std::fs::read(&stored).expect("read stored"), bytes);
}

/// sc-8872 (F-070): the stored extension is derived from the declared/sniffed mime, never
/// trusted from the client filename. A `video/mp4` upload named `evil.html` must NOT be stored
/// `.html` (which `project_file` would then serve as `text/html`, enabling stored XSS on the
/// API origin). `.svg`/`.js`/`.html` neutralize; a genuine mime always wins.
#[test]
fn upload_extension_derives_from_mime_not_client_filename() {
    // Declared video mime wins over a dangerous client extension.
    assert_eq!(upload_extension("evil.html", "video/mp4"), ".mp4");
    assert_eq!(upload_extension("evil.svg", "video/mp4"), ".mp4");
    assert_eq!(upload_extension("evil.js", "video/webm"), ".webm");
    // A legit media extension on a matching mime is preserved.
    assert_eq!(upload_extension("clip.mp4", "video/mp4"), ".mp4");
    assert_eq!(upload_extension("clip.webm", "video/webm"), ".webm");
    // Unknown mime: a safe client media extension is honored, a dangerous one neutralizes.
    assert_eq!(
        upload_extension("clip.mp4", "application/octet-stream"),
        ".mp4"
    );
    assert_eq!(
        upload_extension("evil.html", "application/octet-stream"),
        ".bin"
    );
    // SVG is script-capable and deliberately excluded, so it never survives as `.svg`.
    assert_eq!(upload_extension("logo.svg", "image/svg+xml"), ".bin");
    // Every allow-listed extension re-derives to an inert image/video/audio serve mime — never
    // `text/html` or `image/svg+xml` — which is the property `project_file` relies on.
    for extension in SAFE_UPLOAD_EXTENSIONS {
        let mime = guess_mime_from_filename(&format!("stored.{extension}"))
            .unwrap_or_else(|| format!("no mime for .{extension}"));
        assert!(
            mime.starts_with("image/") || mime.starts_with("video/") || mime.starts_with("audio/"),
            ".{extension} serves inert media mime, got {mime}"
        );
        assert_ne!(mime, "image/svg+xml", ".{extension} must not serve svg");
    }
    assert!(is_safe_upload_extension("mp4"));
    assert!(!is_safe_upload_extension("svg"));
    assert!(!is_safe_upload_extension("html"));
}

#[test]
fn import_asset_refuses_svg_even_when_the_client_claims_png() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("No raw SVG").expect("project creates");
    let source = temp_dir.path().join("untrusted.svg");
    std::fs::write(
        &source,
        b"<svg xmlns=\"http://www.w3.org/2000/svg\"><script/></svg>",
    )
    .expect("fixture writes");

    let error = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "untrusted.svg".to_owned(),
                // The filename check must still win when the client lies about the MIME type.
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect_err("generic SVG import is forbidden");
    assert!(matches!(error, ProjectStoreError::BadRequest(_)));
    assert!(store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("list")
        .is_empty());
    assert!(
        !Path::new(&project.path)
            .join("assets/uploads")
            .read_dir()
            .expect("upload dir")
            .any(|entry| entry.is_ok()),
        "rejected SVG leaves neither asset nor sidecar"
    );
}

/// sc-8872 (F-070): end-to-end — a `video/mp4` upload named `evil.html` is stored with a `.mp4`
/// extension and `project_file` serves it as `video/mp4`, never `text/html`. A legitimate
/// `.webm` upload still round-trips. This pins the content-type-confusion fix through the exact
/// serve path the file endpoint uses.
#[test]
fn import_asset_video_stores_and_serves_safe_mime_regardless_of_client_filename() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Serve").expect("project creates");

    // A `video/mp4` upload whose client filename lies about its type.
    let evil_src = temp_dir.path().join("upload-evil");
    std::fs::write(&evil_src, b"<script>alert(1)</script> not really html").expect("source writes");
    let evil = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "evil.html".to_owned(),
                content_type: Some("video/mp4".to_owned()),
                source_path: evil_src,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("imports");

    let evil_rel = evil["file"]["path"].as_str().expect("path");
    assert!(
        evil_rel.ends_with(".mp4"),
        "video stored as .mp4, not the client .html, got {evil_rel}"
    );
    assert_eq!(evil["type"], json!("video"));
    // The user's original filename is still shown, only the stored name is neutralized.
    assert_eq!(evil["displayName"], json!("evil.html"));

    // The file-serving endpoint derives its Content-Type from the stored name — it must now be
    // an inert video type, never `text/html`.
    let served = store.project_file(&project.id, evil_rel).expect("serves");
    assert_eq!(served.content_type, "video/mp4");
    assert_ne!(served.content_type, "text/html");

    // A legitimate webm upload still stores and serves correctly.
    let webm_src = temp_dir.path().join("upload-webm");
    std::fs::write(&webm_src, b"fake but non-empty webm bytes").expect("source writes");
    let webm = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "clip.webm".to_owned(),
                content_type: Some("video/webm".to_owned()),
                source_path: webm_src,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("imports");
    let webm_rel = webm["file"]["path"].as_str().expect("path");
    assert!(webm_rel.ends_with(".webm"), "webm kept, got {webm_rel}");
    let webm_served = store.project_file(&project.id, webm_rel).expect("serves");
    assert_eq!(webm_served.content_type, "video/webm");
}

/// A minimal RIFF/WAVE file carrying `samples` frames of 32-bit FLOAT PCM (`audioFormat = 3`).
///
/// Float on purpose: it is a perfectly ordinary `.wav` that `read_wav_pcm16` REFUSES ("must be
/// PCM 16-bit"), so a store that admitted audio without converting it would produce an asset
/// the product cannot read. It is the cheapest fixture that proves the conversion RAN rather
/// than that the bytes were copied.
#[cfg(test)]
fn float_wav(sample_rate: u32, channels: u16, samples: usize) -> Vec<u8> {
    let bits = 32u16;
    let block_align = channels * bits / 8;
    let data_len = samples * usize::from(block_align);
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for index in 0..samples {
        let value = ((index % 64) as f32 / 64.0) - 0.5;
        wav.extend_from_slice(&value.to_le_bytes());
    }
    wav
}

/// sc-18650: an AUDIO upload is admitted, typed `audio`, normalized to the PCM-16 RIFF/WAVE the
/// product's only audio decoder reads, and carries the clip facts an Audio Studio render does.
///
/// This is the whole reason the reference-audio picker's dropzone existed with a zero success
/// rate: `import_asset` refused everything but `image/`+`video/`, so a user handed
/// "Could not import the selected audio file." for every file they chose. Each assertion below
/// is one link in the chain that had to hold for that dropzone to mean anything.
///
/// Soft-skips where no ffmpeg is reachable, the same posture as the sibling transcode test —
/// there is no `sips` equivalent for audio, so ffmpeg is the only converter on every platform.
#[test]
fn import_asset_admits_audio_and_normalizes_it_to_pcm16_wav() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Audio").expect("project creates");

    let source = temp_dir.path().join("upload-take");
    std::fs::write(&source, float_wav(24_000, 1, 6_000)).expect("source writes");
    // `content_type: None` drives the `guess_mime_from_filename` path — the shape a browser
    // produces when it reports `application/octet-stream` for a media file it does not know.
    let imported = match store.import_asset(
        &project.id,
        UploadAsset {
            filename: "take.wav".to_owned(),
            content_type: None,
            source_path: source,
            source_asset_id: None,
            provenance: None,
        },
    ) {
        Ok(asset) => asset,
        Err(error) => {
            let message = error.to_string();
            if message.contains("ffmpeg") {
                println!("sc-18650: no ffmpeg reachable on this host ({message}); skipping");
                return;
            }
            panic!("an audio upload must import: {message}");
        }
    };

    // 1. It is an AUDIO asset, the same kind an Audio Studio render produces — so it appears in
    //    `VideoStudio`'s `assets.filter((asset) => asset.type === "audio")` and satisfies the
    //    picker's `assetCanRenderAsAudio`.
    assert_eq!(imported["type"], json!("audio"));
    assert_eq!(imported["origin"], json!("upload"));
    assert_eq!(imported["file"]["mimeType"], json!("audio/wav"));

    // 2. It is stored under an allow-listed extension and SERVES an inert audio mime, the
    //    sc-8872 property every upload has to keep.
    let rel = imported["file"]["path"].as_str().expect("path");
    assert!(rel.ends_with(".wav"), "audio stores as .wav, got {rel}");
    let served = store.project_file(&project.id, rel).expect("serves");
    assert!(
        served.content_type.starts_with("audio/"),
        "an audio asset serves an audio mime, got {}",
        served.content_type
    );

    // 3. The stored bytes are PCM 16-bit — the source was 32-bit float, which `read_wav_pcm16`
    //    refuses. A store that copied the upload through would fail here, and only here.
    let stored = std::fs::read(Path::new(&project.path).join(rel)).expect("stored reads");
    assert_eq!(&stored[0..4], b"RIFF");
    assert_eq!(&stored[8..12], b"WAVE");
    assert_eq!(
        u16::from_le_bytes([stored[20], stored[21]]),
        1,
        "the stored WAV declares PCM (audioFormat 1), not the source's IEEE float"
    );
    assert_eq!(
        u16::from_le_bytes([stored[34], stored[35]]),
        16,
        "the stored WAV is 16-bit"
    );

    // 4. The sidecar carries the clip facts the audio tile, transport and waveform read — and
    //    they are the SOURCE's rate and layout, because the conversion normalizes the ENCODING
    //    and nothing else.
    assert_eq!(imported["file"]["sampleRate"], json!(24_000));
    assert_eq!(imported["file"]["channels"], json!(1));
    let duration = imported["file"]["duration"].as_f64().expect("duration");
    assert!(
        (duration - 0.25).abs() < 0.01,
        "6000 frames @ 24 kHz is 0.25 s, got {duration}"
    );

    // 5. Everything that is not media is still refused — widening the gate must not have
    //    turned it into "anything goes".
    let script = temp_dir.path().join("upload-script");
    std::fs::write(&script, b"#!/bin/sh\nrm -rf /\n").expect("source writes");
    let refused = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "payload.sh".to_owned(),
                content_type: Some("text/x-shellscript".to_owned()),
                source_path: script,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect_err("a non-media upload is still refused");
    assert!(
        refused.to_string().contains("image, video and audio"),
        "the refusal names what IS supported: {refused}"
    );
}

/// sc-6143: a valid-but-unsupported image (BMP here) is transcoded to lossless PNG at import,
/// the original upload temp is cleaned up, and the user-facing display name is retained.
/// macOS-only (real `sips`); the ffmpeg path is covered by the worker tests.
#[cfg(target_os = "macos")]
#[test]
fn import_asset_transcodes_unsupported_bmp_to_png() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Convert").expect("project creates");

    // A valid 1×1 24-bit BMP. The transcode decision here comes from
    // `ImageKind::is_natively_supported` (a fixed png/jpeg/webp list), NOT from what the
    // compiled `image` build can decode — so this fixture is unaffected by the workspace
    // feature union, which does include `bmp` (sc-15052).
    let mut bmp = Vec::new();
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&58u32.to_le_bytes());
    bmp.extend_from_slice(&0u16.to_le_bytes());
    bmp.extend_from_slice(&0u16.to_le_bytes());
    bmp.extend_from_slice(&54u32.to_le_bytes());
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&1i32.to_le_bytes());
    bmp.extend_from_slice(&1i32.to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&2835i32.to_le_bytes());
    bmp.extend_from_slice(&2835i32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&[0x20, 0x40, 0x80, 0x00]);

    let src = temp_dir.path().join("photo.bmp");
    std::fs::write(&src, &bmp).expect("source writes");

    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "photo.bmp".to_owned(),
                content_type: Some("image/bmp".to_owned()),
                source_path: src.clone(),
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("imports");

    // Stored as PNG: mime, extension, and actual on-disk magic bytes.
    assert_eq!(asset["file"]["mimeType"], json!("image/png"));
    let rel = asset["file"]["path"].as_str().expect("path");
    assert!(rel.ends_with(".png"), "stored as png, got {rel}");
    assert_eq!(asset["type"], json!("image"));
    // Display name keeps the user's original filename.
    assert_eq!(asset["displayName"], json!("photo.bmp"));

    let project_path = store.find_project_path(&project.id).expect("project path");
    let stored = std::fs::read(project_path.join(rel)).expect("read stored");
    assert_eq!(
        crate::media_convert::sniff_image_kind(&stored),
        Some(crate::media_convert::ImageKind::Png)
    );
    // The original upload temp was consumed (transcode moved the PNG, not the BMP).
    assert!(!src.exists(), "original upload temp removed");
}

#[test]
fn create_pose_asset_rejects_path_traversal() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");

    // A real file outside the pose cache that traversal would try to reach.
    std::fs::create_dir_all(&data_dir).expect("data dir");
    std::fs::write(data_dir.join("secret.png"), b"top secret").expect("secret writes");
    let job_id = "job_pose_evil";
    std::fs::create_dir_all(data_dir.join("cache").join("pose_detect").join(job_id))
        .expect("cache dir");

    let spec = json!({
        "jobId": job_id,
        "skeletonFile": "../../secret.png",
        "category": "x"
    });
    let err = store
        .create_pose_asset(&spec)
        .expect_err("traversal rejected");
    assert!(matches!(err, ProjectStoreError::BadRequest(_)));
}

/// sc-4211 / F-CORE-7: asset ids are joined into `trash/<id>` and recipe
/// paths, so the asset-sidecar chokepoint must reject a traversal id before
/// any path use. Without it, `delete_asset`/`purge_asset` could create or
/// `remove_dir_all` directories outside the project's trash folder.
#[test]
fn asset_methods_reject_path_traversal_ids() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Assets").expect("project creates");

    // A real sidecar outside the project that traversal would try to reach.
    let outside = temp_dir.path().join("outside");
    std::fs::create_dir_all(&outside).expect("outside dir");
    std::fs::write(outside.join("victim.txt"), b"do not touch").expect("victim writes");

    let evil = "../../../outside/victim";
    assert!(matches!(
        store.get_asset(&project.id, evil),
        Err(ProjectStoreError::BadRequest(_))
    ));
    assert!(matches!(
        store.delete_asset(&project.id, evil),
        Err(ProjectStoreError::BadRequest(_))
    ));
    assert!(matches!(
        store.purge_asset(&project.id, evil, true),
        Err(ProjectStoreError::BadRequest(_))
    ));
    // The traversal target is untouched.
    assert!(outside.join("victim.txt").exists());
}

#[test]
fn delete_asset_rejects_unsafe_sidecar_media_path() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Assets").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let image_dir = project_path.join("assets/images");
    let outside = temp_dir.path().join("outside.png");
    std::fs::write(&outside, b"outside").expect("outside media writes");
    std::fs::write(
        image_dir.join("unsafe.sceneworks.json"),
        serde_json::to_string_pretty(&json!({
            "id": "asset-unsafe",
            "type": "image",
            "displayName": "Unsafe",
            "createdAt": "2026-06-15T00:00:00Z",
            "file": {"path": outside.to_string_lossy()},
            "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
        }))
        .expect("json"),
    )
    .expect("sidecar writes");

    let error = store
        .delete_asset(&project.id, "asset-unsafe")
        .expect_err("unsafe media path rejected");
    assert!(matches!(error, ProjectStoreError::BadRequest(_)));
    assert!(outside.exists(), "delete must not move outside media");
}

#[test]
fn purge_asset_rejects_unsafe_sidecar_media_paths_before_deletion() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Assets").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let image_dir = project_path.join("assets/images");

    let absolute_target = temp_dir.path().join("absolute-victim.png");
    let traversal_target = project_path.join("../../traversal-victim.png");
    std::fs::write(&absolute_target, b"absolute").expect("absolute victim writes");
    std::fs::write(&traversal_target, b"traversal").expect("traversal victim writes");

    for (asset_id, media_path, target) in [
        (
            "purge-absolute",
            absolute_target.to_string_lossy().into_owned(),
            &absolute_target,
        ),
        (
            "purge-traversal",
            "../../traversal-victim.png".to_owned(),
            &traversal_target,
        ),
    ] {
        let sidecar_path = image_dir.join(format!("{asset_id}.sceneworks.json"));
        std::fs::write(
            &sidecar_path,
            serde_json::to_string_pretty(&json!({
                "id": asset_id,
                "type": "image",
                "displayName": asset_id,
                "createdAt": "2026-06-15T00:00:00Z",
                "file": {"path": media_path},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");

        for permanent in [false, true] {
            let error = store
                .purge_asset(&project.id, asset_id, permanent)
                .expect_err("unsafe media path rejected");
            assert!(matches!(error, ProjectStoreError::BadRequest(_)));
            assert!(target.exists(), "purge must not delete outside media");
            assert!(sidecar_path.exists(), "purge must not delete the sidecar");
        }
    }
}

#[test]
fn purge_asset_rejects_unsafe_indexed_sidecar_paths_before_deletion() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

    for (asset_id, indexed_sidecar_path) in [
        (
            "indexed-absolute",
            temp_dir.path().join("outside.sceneworks.json"),
        ),
        (
            "indexed-traversal",
            temp_dir.path().join("data/traversal.sceneworks.json"),
        ),
    ] {
        let project = store.create_project(asset_id).expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let media_rel = format!("assets/images/{asset_id}.png");
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "set",
                &json!({
                    "assetId": asset_id,
                    "mediaPath": media_rel,
                    "mimeType": "image/png",
                    "displayName": asset_id,
                    "createdAt": "2026-06-15T00:00:00Z",
                    "mode": "text_to_image",
                    "model": "test",
                    "adapter": "test",
                    "prompt": "test"
                }),
            )
            .expect("asset persists");
        let media_path = project_path.join(&media_rel);
        std::fs::write(&media_path, b"media").expect("media writes");
        let safe_sidecar_path = media_path.with_extension("sceneworks.json");
        std::fs::write(&indexed_sidecar_path, b"outside sidecar").expect("outside sidecar writes");

        let indexed_value = if asset_id == "indexed-traversal" {
            "../../traversal.sceneworks.json".to_owned()
        } else {
            indexed_sidecar_path.to_string_lossy().into_owned()
        };
        connect_project_db(&project_path)
            .expect("project db")
            .execute(
                "update assets set sidecar_path = ?1 where id = ?2",
                params![indexed_value, asset_id],
            )
            .expect("indexed sidecar path updates");

        for permanent in [false, true] {
            let error = store
                .purge_asset(&project.id, asset_id, permanent)
                .expect_err("unsafe indexed sidecar path rejected");
            assert!(matches!(error, ProjectStoreError::BadRequest(_)));
            assert!(
                indexed_sidecar_path.exists(),
                "outside sidecar is untouched"
            );
            assert!(safe_sidecar_path.exists(), "safe sidecar is untouched");
            assert!(media_path.exists(), "media is untouched");
        }
    }
}

#[test]
fn purge_asset_rejects_linked_media_and_sidecar_parents_before_deletion() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

    for permanent in [false, true] {
        let mode = if permanent { "permanent" } else { "trash" };
        let project = store
            .create_project(&format!("Linked media {mode}"))
            .expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let asset_id = format!("linked-media-{mode}");
        let safe_media_rel = format!("assets/images/{asset_id}.png");
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "set",
                &json!({
                    "assetId": asset_id,
                    "mediaPath": safe_media_rel,
                    "mimeType": "image/png",
                    "displayName": asset_id,
                    "createdAt": "2026-08-28T00:00:00Z",
                    "mode": "text_to_image",
                    "model": "test",
                    "adapter": "test",
                    "prompt": "test"
                }),
            )
            .expect("safe asset persists");
        let safe_sidecar = project_path
            .join(&safe_media_rel)
            .with_extension("sceneworks.json");

        let external_dir = temp_dir.path().join(format!("external-media-{mode}"));
        std::fs::create_dir_all(&external_dir).expect("external dir creates");
        let linked_parent = project_path.join("assets").join(format!("linked-{mode}"));
        if let Err(error) = create_dir_symlink(&external_dir, &linked_parent) {
            eprintln!("symlink privilege unavailable; skipping purge fixture: {error}");
            return;
        }
        let external_media = external_dir.join(format!("{asset_id}.png"));
        std::fs::write(&external_media, b"external media").expect("external media writes");

        let mut sidecar: Value =
            serde_json::from_slice(&std::fs::read(&safe_sidecar).expect("safe sidecar reads"))
                .expect("sidecar decodes");
        sidecar["file"]["path"] = Value::String(format!("assets/linked-{mode}/{asset_id}.png"));
        write_json(&safe_sidecar, &sidecar).expect("sidecar media path updates");
        let sidecar_before = std::fs::read(&safe_sidecar).expect("sidecar snapshots");

        let error = store
            .purge_asset(&project.id, &asset_id, permanent)
            .expect_err("linked media parent is refused");
        assert!(matches!(
            error,
            ProjectStoreError::BadRequest(ref detail)
                if detail == "Asset media path contains a symlink or reparse point"
        ));
        assert_eq!(
            std::fs::read(&external_media).expect("external media remains"),
            b"external media"
        );
        assert_eq!(
            std::fs::read(&safe_sidecar).expect("sidecar remains"),
            sidecar_before
        );
    }

    for permanent in [false, true] {
        let mode = if permanent { "permanent" } else { "trash" };
        let project = store
            .create_project(&format!("Linked sidecar {mode}"))
            .expect("project creates");
        let project_path = std::path::PathBuf::from(&project.path);
        let external_dir = temp_dir.path().join(format!("external-sidecar-{mode}"));
        std::fs::create_dir_all(&external_dir).expect("external dir creates");
        let linked_parent = project_path
            .join("assets")
            .join(format!("linked-sidecar-{mode}"));
        if let Err(error) = create_dir_symlink(&external_dir, &linked_parent) {
            eprintln!("symlink privilege unavailable; skipping purge fixture: {error}");
            return;
        }

        let asset_id = format!("linked-sidecar-{mode}");
        let media_rel = format!("assets/linked-sidecar-{mode}/{asset_id}.png");
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "set",
                &json!({
                    "assetId": asset_id,
                    "mediaPath": media_rel,
                    "mimeType": "image/png",
                    "displayName": asset_id,
                    "createdAt": "2026-08-28T00:00:00Z",
                    "mode": "text_to_image",
                    "model": "test",
                    "adapter": "test",
                    "prompt": "test"
                }),
            )
            .expect("linked fixture persists");
        let external_media = external_dir.join(format!("{asset_id}.png"));
        let external_sidecar = external_media.with_extension("sceneworks.json");
        std::fs::write(&external_media, b"external media").expect("external media writes");
        let sidecar_before = std::fs::read(&external_sidecar).expect("sidecar snapshots");

        let error = store
            .purge_asset(&project.id, &asset_id, permanent)
            .expect_err("linked sidecar parent is refused");
        assert!(matches!(
            error,
            ProjectStoreError::BadRequest(ref detail)
                if detail == "Asset sidecar path contains a symlink or reparse point"
        ));
        assert_eq!(
            std::fs::read(&external_media).expect("external media remains"),
            b"external media"
        );
        assert_eq!(
            std::fs::read(&external_sidecar).expect("external sidecar remains"),
            sidecar_before
        );
    }
}

#[test]
fn purge_asset_removes_safe_media_and_sidecar_in_trash_and_permanent_modes() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");

    for permanent in [false, true] {
        let project = store
            .create_project(if permanent {
                "Permanent purge"
            } else {
                "Trash purge"
            })
            .expect("project creates");
        let asset_id = if permanent {
            "permanent-purge"
        } else {
            "trash-purge"
        };
        let media_rel = format!("assets/images/{asset_id}.png");
        store
            .persist_generated_asset(
                &project.id,
                "job-1",
                "set",
                &json!({
                    "assetId": asset_id,
                    "mediaPath": media_rel,
                    "mimeType": "image/png",
                    "displayName": asset_id,
                    "createdAt": "2026-06-15T00:00:00Z",
                    "mode": "text_to_image",
                    "model": "test",
                    "adapter": "test",
                    "prompt": "test"
                }),
            )
            .expect("asset persists");
        let project_path = std::path::PathBuf::from(&project.path);
        let media_path = project_path.join(&media_rel);
        std::fs::write(&media_path, b"media").expect("media writes");
        let sidecar_path = media_path.with_extension("sceneworks.json");

        let result = store
            .purge_asset(&project.id, asset_id, permanent)
            .expect("safe asset purges");
        assert_eq!(result.status, "purged");
        assert!(!media_path.exists(), "media is removed");
        assert!(!sidecar_path.exists(), "sidecar is removed");
        assert!(matches!(
            store.get_asset(&project.id, asset_id),
            Err(ProjectStoreError::NotFound(_))
        ));
    }
}

#[test]
fn find_timeline_file_ignores_unsafe_indexed_path() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Timeline").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let timeline = json!({
        "id": "timeline-1",
        "name": "Main",
        "aspectRatio": "16:9",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "duration": 0.0,
        "createdAt": "2026-06-15T00:00:00Z",
        "updatedAt": "2026-06-15T00:00:00Z"
    });
    let safe_path = project_path
        .join("timelines")
        .join("main.sceneworks.timeline.json");
    std::fs::write(
        &safe_path,
        serde_json::to_string_pretty(&timeline).expect("json"),
    )
    .expect("timeline writes");
    index_timeline(&project_path, &timeline, "../outside.timeline.json").expect("index writes");

    let found = find_timeline_file(&project_path, "timeline-1").expect("timeline found");
    assert_eq!(
        found.path.canonicalize().expect("canonical found path"),
        safe_path.canonicalize().expect("canonical safe path")
    );
    assert_eq!(
        found.relative_path,
        "timelines/main.sceneworks.timeline.json"
    );
}

/// A timeline saved before sc-22712 existed still loads, still round-trips, and still sounds
/// the same.
///
/// The new fields are `gain` and `role` on a track and `fadeInSeconds` / `fadeOutSeconds` /
/// `generatedAudio` on an item. None of them can be REQUIRED, because every timeline already on
/// disk lacks all five — a validator that rejected them would make the editor unable to open
/// the user's own projects, and one that defaulted `gain` to anything but unity would change
/// what an untouched project sounds like. So the check here is not "the keys appear": it is
/// that the values they appear with are the ones that mean "nothing has changed".
#[test]
fn a_timeline_saved_before_gain_and_fades_existed_still_round_trips_at_unity() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Legacy").expect("project creates");
    let created = store
        .create_timeline(&project.id, "Old cut", "16:9", 24)
        .expect("timeline creates");
    let timeline_id = created["id"].as_str().expect("timeline id").to_owned();

    // Exactly the shape the store wrote before this story: three tracks with `locked`/`muted`
    // and no `role`/`gain`, and an item with no fades and no generated-audio policy.
    let legacy = json!({
        "id": timeline_id,
        "projectId": project.id,
        "name": "Old cut",
        "aspectRatio": "16:9",
        "width": 1280,
        "height": 720,
        "fps": 24,
        "duration": 0.0,
        "tracks": [
            {"id": "track_main", "name": "Main", "kind": "video", "locked": false, "muted": false, "items": [{
                "id": "item_abcdef0123456789abcdef0123456789",
                "trackId": "track_main",
                "assetId": "asset_legacy",
                "displayName": "Legacy clip",
                "type": "video",
                "sourceIn": 0.0,
                "sourceOut": 4.0,
                "timelineStart": 0.0,
                "timelineEnd": 4.0,
                "speed": 1.0,
                "fit": "fit",
                "volume": 1.0
            }]},
            {"id": "track_overlay", "name": "Overlay", "kind": "overlay", "locked": false, "muted": false, "items": []},
            {"id": "track_audio", "name": "Audio", "kind": "audio", "locked": true, "muted": true, "items": []}
        ],
        "transitions": []
    });
    // Model an actual legacy file (revision 0), rather than a stale revisionless
    // client overwriting the freshly created revision 1 document.
    let legacy_file = store.timeline_file(&project.id, &timeline_id).unwrap();
    super::write_json(&legacy_file.path, &legacy).unwrap();
    let saved = store
        .save_timeline(&project.id, legacy)
        .expect("a pre-sc-22712 timeline still saves");

    let tracks = saved["tracks"].as_array().expect("tracks");
    assert_eq!(tracks.len(), 3);
    // Role is inferred from what the document already said, and gain defaults to unity — so a
    // timeline nobody has touched sounds exactly as it did.
    assert_eq!(tracks[0]["role"], json!("picture"));
    assert_eq!(tracks[1]["role"], json!("overlay"));
    assert_eq!(tracks[2]["role"], json!("sound"));
    for track in tracks {
        assert_eq!(
            track["gain"],
            json!(1.0),
            "an untouched track must default to unity gain: {track}"
        );
    }
    // The flags the document DID carry are untouched.
    assert_eq!(tracks[2]["locked"], json!(true));
    assert_eq!(tracks[2]["muted"], json!(true));

    let item = &tracks[0]["items"][0];
    assert_eq!(item["fadeInSeconds"], json!(0.0));
    assert_eq!(item["fadeOutSeconds"], json!(0.0));
    assert_eq!(
        item["generatedAudio"],
        json!("mute"),
        "the default cannot be `include`: a legacy picture item whose clip happens to carry \
             audio would start being mixed into exports that never had it"
    );
    // Nothing else about the item moved.
    assert_eq!(item["assetId"], json!("asset_legacy"));
    assert_eq!(item["sourceOut"], json!(4.0));
    assert_eq!(item["timelineEnd"], json!(4.0));
    assert_eq!(saved["duration"], json!(4.0));

    // And it round-trips: saving what was read back changes nothing.
    let again = store
        .save_timeline(&project.id, saved.clone())
        .expect("the defaulted document saves again");
    for key in ["tracks", "duration", "fps", "aspectRatio"] {
        assert_eq!(again[key], saved[key], "{key} is not stable across a save");
    }
}

/// The new fields are validated, not merely defaulted.
#[test]
fn track_gain_and_item_generated_audio_are_range_checked() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Ranges").expect("project creates");
    let created = store
        .create_timeline(&project.id, "Cut", "16:9", 24)
        .expect("timeline creates");

    let with = |mutate: &dyn Fn(&mut Value)| {
        let mut timeline = created.clone();
        mutate(&mut timeline);
        store.save_timeline(&project.id, timeline)
    };

    assert!(
        with(&|timeline| timeline["tracks"][2]["gain"] = json!(9.0)).is_err(),
        "a gain of 9 is 19 dB of boost and is refused"
    );
    assert!(
        with(&|timeline| timeline["tracks"][2]["gain"] = json!(-1.0)).is_err(),
        "a negative gain is refused"
    );
    assert!(
        with(&|timeline| timeline["tracks"][2]["role"] = json!("foley")).is_err(),
        "an unknown track role is refused rather than silently kept"
    );
    assert!(
        with(&|timeline| {
            timeline["tracks"][0]["items"] = json!([{
                "id": "item_abcdef0123456789abcdef0123456789",
                "trackId": "track_main",
                "assetId": "asset_x",
                "displayName": "Clip",
                "generatedAudio": "maybe"
            }]);
        })
        .is_err(),
        "`generatedAudio` takes include|mute and nothing else — a third spelling would be a \
             policy nobody implemented"
    );
    assert!(
        with(&|timeline| timeline["tracks"][2]["gain"] = json!(2.5)).is_ok(),
        "a legitimate boost inside the range is accepted"
    );
}

#[test]
fn renaming_a_timeline_removes_the_stale_file_before_reindex() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store
        .create_project("Timeline rename")
        .expect("project creates");
    let original = store
        .create_timeline(&project.id, "Old cut", "16:9", 24)
        .expect("timeline creates");
    let timeline_id = original["id"].as_str().expect("timeline id").to_owned();
    let old_file = store
        .timeline_file(&project.id, &timeline_id)
        .expect("old timeline file")
        .path;
    let mut renamed = original;
    renamed["name"] = json!("Final cut");
    renamed["tracks"][0]["items"] = json!([]);
    store
        .save_existing_timeline(&project.id, &timeline_id, renamed)
        .expect("rename saves");

    let project_path = std::path::PathBuf::from(&project.path);
    let files: Vec<_> = std::fs::read_dir(project_path.join("timelines"))
        .expect("timeline directory")
        .map(|entry| entry.expect("timeline entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".sceneworks.timeline.json"))
        })
        .collect();
    assert_eq!(files.len(), 1, "the old timeline file is removed");
    assert!(
        !old_file.exists(),
        "the pre-rename file cannot be reindexed"
    );
    assert!(files[0]
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("final-cut-")));

    let counts = store
        .reindex_project(&project.id)
        .expect("reindex succeeds");
    assert_eq!(counts.timelines, 1);
    let recovered = store
        .get_timeline(&project.id, &timeline_id)
        .expect("renamed timeline remains authoritative");
    assert_eq!(recovered["name"], json!("Final cut"));
}

// ---- Key Point Library (sc-4434) ---------------------------------------------------

fn stage_kps_upload(data_dir: &std::path::Path, name: &str) -> String {
    let dir = data_dir.join("cache").join("keypoint-uploads");
    std::fs::create_dir_all(&dir).expect("uploads dir");
    let path = dir.join(name);
    std::fs::write(&path, b"\x89PNG\r\n\x1a\n fake png bytes").expect("staged image");
    path.to_string_lossy().into_owned()
}

fn front_kps() -> serde_json::Value {
    json!([
        [0.40, 0.34],
        [0.59, 0.34],
        [0.50, 0.43],
        [0.43, 0.53],
        [0.58, 0.53]
    ])
}

#[test]
fn keypoint_reads_keep_the_asset_type_filter_at_the_sql_seam() {
    let source = include_str!("project_store/asset_queries.rs");
    let list = source
        .split_once("fn list_assets_filtered(")
        .expect("filtered asset query")
        .1
        .split_once("pub fn get_asset(")
        .expect("end of asset listing")
        .0;
    assert!(
        list.contains("and (?3 is null or type = ?3)"),
        "the shared asset reader must push its optional type filter into SQLite"
    );
    let source = include_str!("project_store/keypoints.rs");
    let keypoints = source
        .split_once("fn list_user_keypoint_presets(")
        .expect("keypoint reader")
        .1
        .split_once("fn known_preset_ids(")
        .expect("end of keypoint reader")
        .0;
    assert!(keypoints.contains("Some(\"keypoint\")"));
    assert!(
        !keypoints.contains(".filter("),
        "keypoint reads must not fetch every asset and filter in Rust"
    );
}

#[test]
fn create_keypoint_asset_persists_with_source_image() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    let upload = stage_kps_upload(&data_dir, "upload-abc.png");

    let asset = store
        .create_keypoint_asset(&json!({
            "name": "My Front",
            "kps": front_kps(),
            "sourceUploadPath": upload,
        }))
        .expect("preset persists");
    assert_eq!(asset["type"], "keypoint");
    assert_eq!(asset["displayName"], "My Front");
    assert_eq!(asset["keypoint"]["builtin"], false);
    assert_eq!(asset["keypoint"]["kps"].as_array().unwrap().len(), 5);
    // The source image was copied into the library.
    let media_rel = asset["file"]["path"].as_str().expect("media path");
    let project_path = store
        .find_project_path(GLOBAL_KEYPOINTS_PROJECT_ID)
        .expect("project");
    assert!(
        project_path.join(media_rel).exists(),
        "source image retained"
    );

    // list_keypoint_presets = the 11 built-ins + this user preset.
    let presets = store.list_keypoint_presets().expect("list");
    assert_eq!(presets.len(), 12);
    assert_eq!(presets[0]["id"], "builtin_front");
    assert!(presets[0]["builtin"].as_bool().unwrap());
    let user = presets.last().unwrap();
    assert_eq!(user["name"], "My Front");
    assert_eq!(user["builtin"], false);
    assert!(user["sourceImageRef"].as_str().is_some());
}

#[test]
fn keypoint_presets_survive_schema_version_and_dirty_reindex() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    let keypoints_project = store
        .ensure_global_keypoints_project()
        .expect("keypoint project provisions");
    let project_path = std::path::PathBuf::from(&keypoints_project.path);
    assert!(
        project_path.join("assets/keypoints").is_dir(),
        "provisioning creates the authoritative keypoint sidecar folder"
    );

    let upload = stage_kps_upload(&data_dir, "upload-reindex.png");
    let preset = store
        .create_keypoint_asset(&json!({
            "name": "Reindex Angle",
            "kps": front_kps(),
            "sourceUploadPath": upload,
        }))
        .expect("preset persists");
    let preset_id = preset["id"].as_str().expect("preset id").to_owned();
    let collection = store
        .upsert_keypoint_collection(&json!({
            "name": "Reindex collection",
            "orderedPresetIds": [preset_id],
            "isDefault": true,
        }))
        .expect("collection persists");
    let collection_id = collection["id"].as_str().expect("collection id").to_owned();

    let assert_preset_lists_and_resolves = || {
        let presets = store.list_keypoint_presets().expect("preset listing");
        assert!(
            presets
                .iter()
                .any(|preset| preset["id"] == json!(preset_id)),
            "the recovered sidecar is visible through the Key Point Library"
        );
        let (resolved_collection_id, resolved) = store
            .resolve_angle_collection(Some(&collection_id))
            .expect("collection resolution");
        assert_eq!(resolved_collection_id, collection_id);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].preset_id, preset_id);
        assert_eq!(resolved[0].name, "Reindex Angle");
    };

    // Simulate a schema-version rebuild: the persisted index was cleared and its version
    // stamp is stale, while the user-owned keypoint sidecar remains authoritative on disk.
    {
        let connection = connect_project_db(&project_path).expect("open index");
        connection
            .execute("delete from assets", [])
            .expect("clear index");
        connection
            .execute(
                "update project_metadata set value = ?1 where key = ?2",
                params![
                    (PROJECT_SCHEMA_VERSION - 1).to_string(),
                    ASSET_INDEX_VERSION_KEY
                ],
            )
            .expect("stale index version");
        connection
            .execute_batch(&format!(
                "pragma user_version = {}",
                PROJECT_SCHEMA_VERSION - 1
            ))
            .expect("stale schema version");
    }
    assert!(
        project_path
            .join(format!("assets/keypoints/{preset_id}.sceneworks.json"))
            .exists(),
        "precondition: keypoint sidecar survives the cleared index"
    );
    assert_preset_lists_and_resolves();

    // Simulate a crashed index mutation: dirty-marker repair clears the rows again and must
    // discover the same sidecar before it removes the marker.
    {
        let connection = connect_project_db(&project_path).expect("open index");
        connection
            .execute("delete from assets", [])
            .expect("clear index for dirty repair");
    }
    std::fs::write(project_path.join(ASSET_INDEX_DIRTY_MARKER), b"repair")
        .expect("write dirty marker");
    assert_preset_lists_and_resolves();
    assert!(
        !project_path.join(ASSET_INDEX_DIRTY_MARKER).exists(),
        "successful keypoint reindex clears the dirty marker"
    );
}

#[test]
fn create_keypoint_asset_labels_extensionless_upload_by_content() {
    // Regression: uploads stage as `upload-<uuid>.tmp` (no real extension). The saved
    // library asset must take its extension + mime from the file CONTENT, not the `.tmp`
    // name, or a JPEG capture gets mislabeled image/png.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    let dir = data_dir.join("cache").join("keypoint-uploads");
    std::fs::create_dir_all(&dir).expect("uploads dir");
    let upload = dir.join("upload-deadbeef.tmp");
    // Minimal JPEG SOI + APP0 marker bytes — enough for the magic-byte sniff.
    std::fs::write(&upload, [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]).expect("staged jpeg");

    let asset = store
        .create_keypoint_asset(&json!({
            "name": "From Photo",
            "kps": front_kps(),
            "sourceUploadPath": upload.to_string_lossy(),
        }))
        .expect("preset persists");
    let media_rel = asset["file"]["path"].as_str().expect("media path");
    assert!(
        media_rel.ends_with(".jpg"),
        "expected .jpg, got {media_rel}"
    );
    assert_eq!(asset["file"]["mimeType"], "image/jpeg");
}

/// sc-6143: a keypoint capture in a valid-but-unsupported format (here BMP — the same decode gap
/// AVIF hits) is transcoded to a real PNG as it lands in the library, never copied verbatim under
/// a mislabeled `.png` name. macOS-only (relies on `sips`); the ffmpeg path off macOS is identical.
#[cfg(target_os = "macos")]
#[test]
fn create_keypoint_asset_transcodes_an_unsupported_capture_to_png() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    let dir = data_dir.join("cache").join("keypoint-uploads");
    std::fs::create_dir_all(&dir).expect("uploads dir");
    // Staged with no real extension (`.tmp`), and the bytes are BMP — unsupported by the worker's
    // image build, the exact case that previously slipped through mislabeled as PNG.
    let upload = dir.join("upload-cafef00d.tmp");
    std::fs::write(&upload, sniff_bmp_bytes()).expect("staged bmp");

    let asset = store
        .create_keypoint_asset(&json!({
            "name": "From BMP",
            "kps": front_kps(),
            "sourceUploadPath": upload.to_string_lossy(),
        }))
        .expect("preset persists");

    let media_rel = asset["file"]["path"].as_str().expect("media path");
    assert!(
        media_rel.ends_with(".png"),
        "expected .png, got {media_rel}"
    );
    assert_eq!(asset["file"]["mimeType"], "image/png");
    // The stored bytes are genuinely PNG (sniffed by content), not the original BMP renamed.
    let project_path = store
        .find_project_path(GLOBAL_KEYPOINTS_PROJECT_ID)
        .expect("project");
    let stored = std::fs::read(project_path.join(media_rel)).expect("read stored");
    assert_eq!(
        crate::media_convert::sniff_image_kind(&stored),
        Some(crate::media_convert::ImageKind::Png)
    );
}

/// A valid 1×1 24-bit BMP (no Rust image dep needed to build one).
#[cfg(target_os = "macos")]
fn sniff_bmp_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"BM");
    bytes.extend_from_slice(&58u32.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&54u32.to_le_bytes());
    bytes.extend_from_slice(&40u32.to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&1i32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&24u16.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&2835i32.to_le_bytes());
    bytes.extend_from_slice(&2835i32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&[0x20, 0x40, 0x80, 0x00]);
    bytes
}

#[test]
fn ingest_image_names_extensionless_magic_bytes() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    for (name, bytes, expected) in [
        (
            "png",
            b"\x89PNG\r\n\x1a\n....".as_slice(),
            ("png", "image/png"),
        ),
        (
            "jpg",
            [0xff, 0xd8, 0xff, 0xe0].as_slice(),
            ("jpg", "image/jpeg"),
        ),
        (
            "webp",
            b"RIFF\0\0\0\0WEBPVP8 ".as_slice(),
            ("webp", "image/webp"),
        ),
        ("other", b"not an image".as_slice(), ("png", "image/png")),
    ] {
        let source = temp_dir.path().join(format!("{name}.bin"));
        std::fs::write(&source, bytes).unwrap();
        let result = crate::media_convert::ingest_image_with_naming(
            &source,
            &temp_dir.path().join(format!("stored-{name}")),
            crate::media_convert::ImageCopyNaming::Canonical,
        )
        .unwrap();
        assert_eq!((result.extension.as_str(), result.mime), expected);
        assert_eq!(std::fs::read(result.path).unwrap(), bytes);
    }
}

#[test]
fn create_keypoint_asset_rejects_bad_inputs() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    let upload = stage_kps_upload(&data_dir, "upload-xyz.png");

    // Wrong kps arity.
    assert!(matches!(
        store.create_keypoint_asset(&json!({
            "name": "Bad", "kps": [[0.1, 0.1]], "sourceUploadPath": &upload
        })),
        Err(ProjectStoreError::BadRequest(_))
    ));
    // Out-of-range kps.
    assert!(matches!(
        store.create_keypoint_asset(&json!({
            "name": "Bad",
            "kps": [[1.4, 0.1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            "sourceUploadPath": &upload
        })),
        Err(ProjectStoreError::BadRequest(_))
    ));
    // Path traversal outside the uploads cache.
    std::fs::write(data_dir.join("secret.png"), b"secret").expect("secret");
    assert!(store
            .create_keypoint_asset(&json!({
                "name": "Evil", "kps": front_kps(),
                "sourceUploadPath": data_dir.join("cache/keypoint-uploads/../../secret.png").to_string_lossy()
            }))
            .is_err());
}

#[test]
fn keypoint_collections_default_to_builtin_then_user() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");

    // Fresh: only the built-in default collection, and it is the default.
    let collections = store.list_keypoint_collections().expect("list");
    assert_eq!(collections.len(), 1);
    assert_eq!(
        collections[0]["id"],
        crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID
    );
    assert_eq!(collections[0]["isDefault"], true);
    assert_eq!(
        collections[0]["orderedPresetIds"].as_array().unwrap().len(),
        11
    );

    // Create a user collection from a subset of built-ins, mark it default.
    let created = store
        .upsert_keypoint_collection(&json!({
            "name": "Just profiles",
            "orderedPresetIds": ["builtin_left_profile", "builtin_right_profile"],
            "isDefault": true,
        }))
        .expect("upsert");
    let collection_id = created["id"].as_str().unwrap().to_owned();

    let collections = store.list_keypoint_collections().expect("list");
    assert_eq!(collections.len(), 2);
    // Built-in default now yields to the user's default.
    assert_eq!(collections[0]["isDefault"], false);
    assert_eq!(collections[1]["isDefault"], true);
    assert_eq!(collections[1]["name"], "Just profiles");

    // Referencing an unknown preset is rejected.
    assert!(matches!(
        store.upsert_keypoint_collection(&json!({
            "name": "Bad", "orderedPresetIds": ["builtin_nope"]
        })),
        Err(ProjectStoreError::BadRequest(_))
    ));

    // Reset default back to the built-in.
    let collections = store
        .set_default_keypoint_collection(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID)
        .expect("set default");
    assert_eq!(collections[0]["isDefault"], true);
    assert_eq!(collections[1]["isDefault"], false);

    // Delete the user collection; the built-in cannot be deleted.
    assert!(store
        .delete_keypoint_collection(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID)
        .is_err());
    store
        .delete_keypoint_collection(&collection_id)
        .expect("delete user collection");
    assert_eq!(store.list_keypoint_collections().expect("list").len(), 1);
}

#[test]
fn resolve_angle_collection_default_override_and_custom_preset() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");

    // No user collection → the built-in 11 in order.
    let (id, presets) = store.resolve_angle_collection(None).expect("resolve");
    assert_eq!(id, crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID);
    assert_eq!(presets.len(), 11);
    assert_eq!(presets[0].preset_id, "builtin_front");
    assert_eq!(presets[0].angle.as_deref(), Some("front"));

    // A user preset (real source image) + a collection mixing it with a built-in.
    let upload = stage_kps_upload(&data_dir, "upload-mix.png");
    let preset = store
        .create_keypoint_asset(&json!({
            "name": "My Angle", "kps": front_kps(), "sourceUploadPath": upload
        }))
        .expect("preset");
    let preset_id = preset["id"].as_str().unwrap().to_owned();
    let collection = store
        .upsert_keypoint_collection(&json!({
            "name": "Mix",
            "orderedPresetIds": [preset_id, "builtin_left_profile"],
            "isDefault": true,
        }))
        .expect("collection");
    let collection_id = collection["id"].as_str().unwrap().to_owned();

    // Default now resolves to the user's collection (2 presets, in order).
    let (id, presets) = store.resolve_angle_collection(None).expect("resolve");
    assert_eq!(id, collection_id);
    assert_eq!(presets.len(), 2);
    assert_eq!(presets[0].name, "My Angle");
    assert!(
        presets[0].angle.is_none(),
        "custom preset has no canonical angle"
    );
    assert_eq!(presets[1].preset_id, "builtin_left_profile");
    assert_eq!(presets[1].angle.as_deref(), Some("left_profile"));

    // Explicit override to the built-in default beats the user default.
    let (id, presets) = store
        .resolve_angle_collection(Some(crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID))
        .expect("resolve override");
    assert_eq!(id, crate::angle_kps::BUILTIN_DEFAULT_COLLECTION_ID);
    assert_eq!(presets.len(), 11);
}

#[test]
fn keypoint_library_hidden_from_project_switcher() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_dir = temp_dir.path().join("data");
    let store = ProjectStore::new(&data_dir, "test-version");
    store.ensure_global_keypoints_project().expect("ensure");
    let visible = store.list_projects().expect("list projects");
    assert!(visible
        .iter()
        .all(|project| project.id != GLOBAL_KEYPOINTS_PROJECT_ID));
}

#[test]
fn reindex_rebuilds_python_project_tables() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Fixture").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let image_dir = project_path.join("assets/images");
    std::fs::write(image_dir.join("image.png"), b"image").expect("image writes");
    std::fs::write(
        image_dir.join("image.sceneworks.json"),
        serde_json::to_string_pretty(&json!({
            "id": "asset-1",
            "type": "image",
            "displayName": "Image",
            "createdAt": "2026-05-17T00:00:00Z",
            "generationSetId": "genset-1",
            "file": {"path": "assets/images/image.png"},
            "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
        }))
        .expect("json"),
    )
    .expect("sidecar writes");
    std::fs::write(
        project_path.join("generation-sets/genset-1.json"),
        serde_json::to_string_pretty(&json!({
            "id": "genset-1",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "test",
            "createdAt": "2026-05-17T00:00:00Z",
            "jobId": "job-1"
        }))
        .expect("json"),
    )
    .expect("genset writes");
    std::fs::write(
        project_path.join("timelines/main.sceneworks.timeline.json"),
        serde_json::to_string_pretty(&json!({
            "id": "timeline-1",
            "name": "Main",
            "aspectRatio": "16:9",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "duration": 3.5,
            "createdAt": "2026-05-17T00:00:00Z",
            "updatedAt": "2026-05-17T00:00:00Z"
        }))
        .expect("json"),
    )
    .expect("timeline writes");

    let counts = store.reindex_project(&project.id).expect("reindex works");

    assert_eq!(counts.assets, 1);
    assert_eq!(counts.generation_sets, 1);
    assert_eq!(counts.timelines, 1);
}

#[test]
fn document_assets_are_indexed_and_listed() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Docs").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);

    // assets/documents is created at project init (PROJECT_FOLDERS).
    assert!(project_path.join("assets/documents").exists());

    let document_dir = project_path.join("assets/documents");
    std::fs::write(
        document_dir.join("doc_1.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "doc_1",
            "projectId": project.id,
            "jobId": "job-1",
            "model": "sensenova_u1_8b",
            "prompt": "illustrated guide",
            "createdAt": "2026-05-23T00:00:00Z",
            "segments": [
                {"type": "text", "text": "Step one."},
                {"type": "image", "assetId": "asset-img-1", "path": "assets/images/a.png"}
            ]
        }))
        .expect("json"),
    )
    .expect("document writes");
    std::fs::write(
        document_dir.join("doc_1.sceneworks.json"),
        serde_json::to_string_pretty(&json!({
            "id": "doc_1",
            "type": "document",
            "displayName": "Illustrated guide",
            "createdAt": "2026-05-23T00:00:00Z",
            "file": {"path": "assets/documents/doc_1.json"},
            "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false}
        }))
        .expect("json"),
    )
    .expect("sidecar writes");

    let counts = store.reindex_project(&project.id).expect("reindex works");
    assert_eq!(counts.assets, 1);

    let assets = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("assets list");
    assert_eq!(assets.len(), 1);
    assert_eq!(assets[0]["id"], "doc_1");
    assert_eq!(assets[0]["type"], "document");
    // The document asset carries a derived origin (sc-2024).
    assert_eq!(assets[0]["origin"], "document_studio");
}

#[test]
fn asset_reads_include_generation_set_recipe_when_available() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Recipes").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let image_dir = project_path.join("assets/images");
    std::fs::write(image_dir.join("image.png"), b"image").expect("image writes");
    std::fs::write(
            image_dir.join("image.sceneworks.json"),
            serde_json::to_string_pretty(&json!({
                "id": "asset-1",
                "type": "image",
                "displayName": "Image",
                "createdAt": "2026-06-07T00:00:00Z",
                "generationSetId": "genset_recipe",
                "file": {"path": "assets/images/image.png"},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false},
                "recipe": {
                    "mode": "text_to_image",
                    "model": "z_image_turbo",
                    "adapter": "z_image_diffusers",
                    "prompt": "single asset",
                    "negativePrompt": "",
                    "seed": 7,
                    "loras": [],
                    "normalizedSettings": {},
                    "rawAdapterSettings": {}
                },
                "lineage": {"parents": [], "sourceAssetId": Value::Null, "sourceTimestamp": Value::Null, "jobId": "job-1"}
            }))
            .expect("json"),
        )
        .expect("sidecar writes");
    std::fs::write(
        project_path.join("generation-sets/genset_recipe.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "genset_recipe",
            "projectId": project.id,
            "jobId": "job-1",
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "prompt": "batch prompt",
            "negativePrompt": "",
            "count": 4,
            "createdAt": "2026-06-07T00:00:00Z",
            "recipe": {
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "adapter": "z_image_diffusers",
                "prompt": "batch prompt",
                "negativePrompt": "",
                "seed": 42,
                "loras": [{"id": "style_lora", "weight": 0.75}],
                "normalizedSettings": {"count": 4, "width": 1024, "height": 1024},
                "rawAdapterSettings": {"steps": 8}
            }
        }))
        .expect("json"),
    )
    .expect("generation set writes");

    store.reindex_project(&project.id).expect("reindex works");

    let assets = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("assets list");
    assert_eq!(
        assets[0]["generationSet"]["recipe"]["prompt"],
        "batch prompt"
    );
    assert_eq!(assets[0]["generationSet"]["recipe"]["seed"], json!(42));
    assert_eq!(
        assets[0]["generationSet"]["recipe"]["normalizedSettings"]["count"],
        json!(4)
    );

    let detail = store
        .get_asset(&project.id, "asset-1")
        .expect("asset detail reads");
    assert_eq!(
        detail["generationSet"]["recipe"]["loras"][0]["id"],
        "style_lora"
    );
}

#[test]
fn library_scope_excludes_character_studio_outputs() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Scoped").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    let image_dir = project_path.join("assets/images");

    // Two image asset sidecars written WITHOUT an explicit `origin` field, to
    // exercise the derive-on-reindex backfill: a normal Image Studio output
    // and a Character Studio test output (recipe.mode == "character_image").
    let write_sidecar = |id: &str, mode: &str| {
        std::fs::write(
            image_dir.join(format!("{id}.png")),
            b"not-a-real-png".as_slice(),
        )
        .expect("media writes");
        std::fs::write(
            image_dir.join(format!("{id}.sceneworks.json")),
            serde_json::to_string_pretty(&json!({
                "id": id,
                "type": "image",
                "displayName": id,
                "createdAt": "2026-05-23T00:00:00Z",
                "file": {"path": format!("assets/images/{id}.png")},
                "status": {"favorite": false, "rating": 0, "rejected": false, "trashed": false},
                "recipe": {"mode": mode},
            }))
            .expect("json"),
        )
        .expect("sidecar writes");
    };
    write_sidecar("img_studio_1", "text_to_image");
    write_sidecar("char_test_1", "character_image");

    store.reindex_project(&project.id).expect("reindex works");

    // All scope returns both, each with a derived origin.
    let all = store
        .list_assets(&project.id, false, false, AssetScope::All)
        .expect("all list");
    assert_eq!(all.len(), 2);
    let origin_of = |id: &str| {
        all.iter()
            .find(|asset| asset["id"] == id)
            .map(|asset| asset["origin"].as_str().unwrap_or_default().to_owned())
            .unwrap_or_default()
    };
    assert_eq!(origin_of("img_studio_1"), "image_studio");
    assert_eq!(origin_of("char_test_1"), "character_studio");

    // Library scope drops the Character Studio output, keeps the studio image.
    let library = store
        .list_assets(&project.id, false, false, AssetScope::Library)
        .expect("library list");
    assert_eq!(library.len(), 1);
    assert_eq!(library[0]["id"], "img_studio_1");
}

#[test]
fn person_tracks_list_and_detail_match_python_sidecar_shape() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Fixture").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    std::fs::write(
        project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": project.id,
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [],
            "status": {}
        }))
        .expect("json"),
    )
    .expect("track sidecar writes");

    let tracks = store.list_person_tracks(&project.id).expect("tracks list");
    assert_eq!(tracks[0]["id"], "track_1");
    assert_eq!(
        tracks[0]["path"],
        "person-tracks/track_1.sceneworks.person-track.json"
    );

    let track = store
        .get_person_track(&project.id, "track_1")
        .expect("track detail");
    assert_eq!(track["name"], "Hero");
    assert!(store.get_person_track(&project.id, "../track_1").is_err());
    assert!(store.get_person_track(&project.id, "track~1").is_err());
}

#[test]
fn person_track_corrections_persist_validate_and_clear() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp_dir.path().join("data"), "test-version");
    let project = store.create_project("Fixture").expect("project creates");
    let project_path = std::path::PathBuf::from(&project.path);
    std::fs::write(
        project_path.join("person-tracks/track_1.sceneworks.person-track.json"),
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "id": "track_1",
            "projectId": project.id,
            "name": "Hero",
            "createdAt": "2026-05-17T00:00:00Z",
            "sourceAssetId": "asset-video",
            "representativeFrameAssetId": "asset-frame",
            "frames": [
                {"timestamp": 0.0, "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.5}},
                {"timestamp": 0.5, "box": {"x": 0.3, "y": 0.1, "width": 0.2, "height": 0.5}}
            ],
            "corrections": [],
            "status": {"correctionState": "ready_for_box_corrections"}
        }))
        .expect("json"),
    )
    .expect("track sidecar writes");

    // A box adjustment and a rejection persist; the server stamps metadata and
    // drops the trailing no-op entry (no box, not rejected) for frame 0.
    let updated = store
            .set_person_track_corrections(
                &project.id,
                "track_1",
                vec![
                    json!({"frameIndex": 0, "box": {"x": 0.5, "y": 0.2, "width": 0.25, "height": 0.4}, "author": "editor"}),
                    json!({"frameIndex": 1, "rejected": true}),
                    json!({"frameIndex": 0, "rejected": false}),
                ],
            )
            .expect("corrections persist");
    let corrections = updated["corrections"]
        .as_array()
        .expect("corrections array");
    assert_eq!(corrections.len(), 2);
    assert_eq!(corrections[0]["frameIndex"], 0);
    assert_eq!(corrections[0]["box"]["x"], 0.5);
    assert_eq!(corrections[0]["author"], "editor");
    assert_eq!(corrections[0]["source"], "manual");
    assert!(corrections[0]["createdAt"].is_string());
    assert_eq!(corrections[1]["frameIndex"], 1);
    assert_eq!(corrections[1]["rejected"], true);
    assert_eq!(
        updated["status"]["correctionState"],
        "box_corrections_applied"
    );

    // Out-of-range frame indices and out-of-bounds boxes are rejected.
    assert!(store
        .set_person_track_corrections(
            &project.id,
            "track_1",
            vec![json!({"frameIndex": 9, "rejected": true})]
        )
        .is_err());
    assert!(store
        .set_person_track_corrections(
            &project.id,
            "track_1",
            vec![
                json!({"frameIndex": 0, "box": {"x": 1.5, "y": 0.0, "width": 0.2, "height": 0.2}})
            ]
        )
        .is_err());

    // Clearing corrections returns the track to the ready state.
    let cleared = store
        .set_person_track_corrections(&project.id, "track_1", vec![])
        .expect("corrections clear");
    assert!(cleared["corrections"].as_array().expect("array").is_empty());
    assert_eq!(
        cleared["status"]["correctionState"],
        "ready_for_box_corrections"
    );
}

#[test]
fn project_file_paths_reject_traversal_and_backslashes() {
    assert!(is_safe_relative_path("assets/images/image.png"));
    assert!(!is_safe_relative_path("../outside.txt"));
    assert!(!is_safe_relative_path("assets\\..\\outside.txt"));
    assert!(!is_safe_relative_path("/absolute/path.png"));
}

#[test]
fn public_project_files_allow_product_media_and_refuse_internal_state() {
    let temp = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store.create_project("Public files").unwrap();
    let root = store.find_project_path(&project.id).unwrap();
    for path in [
        "assets/images/picture.png",
        "assets/videos/movie.mp4",
        "assets/tensors/latents.safetensors",
        "assets/tensors/genset/0000.sceneworks-workflow.json",
        "assets/tensors/genset/10000.sceneworks-workflow.json",
        "assets/uploads/measurement.npy",
        "assets/depth/set/map.npy",
        "training/uploads/legacy.bin",
        "training/datasets/ds/prepared/item.safetensors",
        "training/datasets/ds/sidecars/item.depth.npy",
        "training/datasets/ds/sidecars/item.mask.npy",
        "exports/latents.npz",
        "assets/uploads/source.bin",
        "assets/uploads/legacy.ogv",
        "assets/documents/script.json",
        "assets/uploads/source.svg",
        "training/datasets/ds/masks/mask.png",
        "training/samples/lora/step-000250/sample.png",
        "loras/lora/samples/step-000250/sample.png",
        "training/loras/legacy/samples/step-000250/sample.png",
        "films/draft-assets/film/references/ref.webp",
        "films/drafts/film.json",
        "films/runs/run/authoring.json",
        "films/runs/run/plan.json",
        "timeline-exports/cut.json",
        "exports/cut.zip",
    ] {
        let target = root.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"public fixture").unwrap();
        assert!(store.project_file(&project.id, path).is_ok(), "{path}");
    }
    for path in [
        "project.db",
        "project.json",
        ".marker",
        "assets/.secret.png",
        "assets/images/.hidden/secret.png",
        "assets/images/project.db",
        "assets/images/state.sqlite3",
        "assets/images/image.sceneworks.json",
        "assets/images/partial.png.tmp",
        "cache/preview.png",
        "recipes/recipe.json",
        "films/runs/run/locator.json",
        "films/planning/operations/request.json",
        "training/datasets/ds/dataset.json",
        "loras/lora/manifest.jsonc",
        "loras/lora/optimizer.safetensors",
        "loras/lora/samples/optimizer.safetensors",
        "training/datasets/ds/latents/cache.npz",
        "training/datasets/ds/optimizer.safetensors",
        "training/datasets/ds/state.bin",
        "training/datasets/ds/sidecars/state.npz",
        "training/datasets/ds/prepared/optimizer.bin",
        "assets/images/state.safetensors",
        "assets/images/index.json",
        "assets/tensors/latent.sceneworks-workflow.json",
        "assets/tensors/genset/state.json",
        "assets/tensors/genset/0000.sceneworks.json",
        "assets/tensors/genset/000.sceneworks-workflow.json",
        "assets/tensors/genset/latent.sceneworks-workflow.json",
        "assets/tensors/genset/nested/0000.sceneworks-workflow.json",
        "generation-sets/run.json",
        "characters/mira.json",
        "person-tracks/track.sceneworks.person-track.json",
    ] {
        let target = root.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        // Keep the real database/registry metadata intact so the later
        // symlink checks still exercise canonical confinement on a valid project.
        if !matches!(path, "project.db" | "project.json") {
            fs::write(&target, b"private fixture").unwrap();
        }
        assert!(
            matches!(
                store.project_file(&project.id, path),
                Err(super::ProjectStoreError::NotFound(_))
            ),
            "{path}"
        );
    }
    assert!(store.project_file(&project.id, "../escape.png").is_err());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(
            root.join("project.db"),
            root.join("assets/tensors/genset/0001.sceneworks-workflow.json"),
        )
        .unwrap();
        assert!(store
            .project_file(
                &project.id,
                "assets/tensors/genset/0001.sceneworks-workflow.json"
            )
            .is_err());
        std::os::unix::fs::symlink(
            root.join("project.db"),
            root.join("assets/images/alias.png"),
        )
        .unwrap();
        assert!(store
            .project_file(&project.id, "assets/images/alias.png")
            .is_err());
        let outside = temp.path().join("outside.png");
        fs::write(&outside, b"outside").unwrap();
        std::os::unix::fs::symlink(outside, root.join("assets/images/escape.png")).unwrap();
        assert!(store
            .project_file(&project.id, "assets/images/escape.png")
            .is_err());
    }
}

#[test]
fn mime_guess_covers_modern_image_uploads() {
    assert_eq!(
        guess_mime_from_filename("reference.heic").as_deref(),
        Some("image/heic")
    );
    assert_eq!(
        guess_mime_from_filename("reference.avif").as_deref(),
        Some("image/avif")
    );
}

#[test]
fn film_run_revision_barrier_refuses_concurrent_save_without_creating_run() {
    let temp = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store.create_project("Revision race").unwrap();
    let draft = store
        .create_film_draft(&project.id, "film_race", "Race")
        .unwrap();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let validated_revision = store
                .get_film_draft(&project.id, &draft.id)
                .unwrap()
                .revision;
            barrier.wait();
            barrier.wait();
            store.create_film_run_at_revision(
                &project.id,
                "run_stale",
                &draft.id,
                validated_revision,
                vec![],
            )
        });
        barrier.wait();
        let mut newer = draft.clone();
        newer.production_plan.shots[0].prompt = "Changed by the other session".to_owned();
        let newer = store
            .save_film_draft(&project.id, &draft.id, newer)
            .unwrap();
        barrier.wait();
        let conflict = handle.join().unwrap().unwrap_err();
        assert!(matches!(&conflict, ProjectStoreError::Conflict(_)));
        assert!(conflict.to_string().contains("revision conflict"));
        assert!(store.list_film_runs(&project.id).unwrap().is_empty());
        assert!(!std::path::Path::new(&project.path)
            .join("films/runs/run_stale")
            .exists());
        let locator = store
            .create_film_run_at_revision(
                &project.id,
                "run_current",
                &draft.id,
                newer.revision,
                vec![],
            )
            .unwrap();
        assert_eq!(locator.draft_revision, newer.revision);
        assert_eq!(locator.selected_shot_ids, vec!["SH010"]);
        let files = store.film_run_files(&project.id, &locator.id).unwrap();
        assert_eq!(
            read_json(&files.plan).unwrap(),
            serde_json::to_value(&newer.production_plan).unwrap()
        );
        assert_eq!(read_json(&files.review_plan).unwrap(), newer.review_plan);
    });
}

/// sc-24026. A draft written before plan schema version 3 has `shots[].sound` and no
/// `shots[].audio`. `Shot` is `deny_unknown_fields` and `audio` has no default, so BOTH faults
/// are hard serde errors: without the read carry-forward every one of these calls is a 500, and
/// because `list_film_drafts` propagates, one stale draft takes the whole Films list with it —
/// the user cannot even reach the draft to repair it.
#[test]
fn a_draft_stored_before_the_audio_sentence_opens_lists_and_saves_with_its_sound_carried_over() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store
        .create_project("Stale draft")
        .expect("project creates");
    let created = store
        .create_film_draft(&project.id, "film_stale", "Stale")
        .expect("draft creates");
    let draft_path = store
        .find_project_path(&project.id)
        .expect("project path")
        .join("films/drafts")
        .join("film_stale.json");

    // Rewrite the stored file as the previous build wrote it: plan schema 2, the shot's
    // sentence under `sound`, no `audio` key at all. Shot 0 said something about sound; a
    // second shot never did, which is the other half of the migration.
    let mut stored = read_json(&draft_path).expect("stored draft reads");
    {
        let plan = stored
            .get_mut("productionPlan")
            .and_then(Value::as_object_mut)
            .expect("plan object");
        plan.insert("schemaVersion".to_owned(), Value::from(2));
        let shots = plan
            .get_mut("shots")
            .and_then(Value::as_array_mut)
            .expect("shots array");
        let mut silent = shots[0].clone();
        let first = shots[0].as_object_mut().expect("shot object");
        first.remove("audio");
        first.insert(
            "sound".to_owned(),
            Value::from("Room tone and a door latch. No music."),
        );
        let second = silent.as_object_mut().expect("shot object");
        second.insert("id".to_owned(), Value::from("SH020"));
        second.remove("audio");
        shots.push(silent);
    }
    write_json(&draft_path, &stored).expect("stale draft writes");

    for opened in [
        store
            .get_film_draft(&project.id, "film_stale")
            .expect("a stale draft still opens"),
        store
            .list_film_drafts(&project.id)
            .expect("a stale draft does not take the project's film list down")
            .into_iter()
            .find(|draft| draft.id == "film_stale")
            .expect("the stale draft is listed"),
    ] {
        assert_eq!(
            opened.production_plan.schema_version,
            crate::film_plan::PLAN_SCHEMA_VERSION
        );
        assert_eq!(
            opened.production_plan.shots[0].audio, "Room tone and a door latch. No music.",
            "the version 2 sentence is carried into the field that dispatches it"
        );
        // The shot that never said anything comes back blank rather than inventing prose...
        assert_eq!(opened.production_plan.shots[1].audio, "");
    }

    // ...and blank is not silently accepted: it surfaces as the shot-named finding the
    // workspace's Audio field is there to clear.
    let opened = store
        .get_film_draft(&project.id, "film_stale")
        .expect("a stale draft still opens");
    let findings = crate::film_plan::validate_plan_structure(&opened.production_plan);
    assert!(
        findings
            .iter()
            .any(|finding| finding.shot_id.as_deref() == Some("SH020") && finding.field == "audio"),
        "{findings:?}"
    );
    assert!(
        !findings
            .iter()
            .any(|finding| finding.field == "schemaVersion"),
        "the carried-forward plan is not also refused by version: {findings:?}"
    );

    // The PUT the user makes after fixing it reads the stored document back to check the
    // revision, so it goes through the same carry-forward.
    let mut edited = opened;
    edited.production_plan.shots[1].audio = "Silence. No audio.".to_owned();
    let saved = store
        .save_film_draft(&project.id, "film_stale", edited)
        .expect("the repaired draft saves");
    assert_eq!(saved.revision, created.revision + 1);
    assert_eq!(saved.production_plan.shots[1].audio, "Silence. No audio.");
    assert!(
        !read_json(&draft_path)
            .expect("saved draft reads")
            .to_string()
            .contains("\"sound\":\"Room tone"),
        "the version 2 key is gone from the persisted document, not merely from the decode"
    );
}

#[test]
fn film_runs_retain_exact_authoring_revision_across_later_draft_saves() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store
        .create_project("Pinned authoring")
        .expect("project creates");
    let mut first = store
        .create_film_draft(&project.id, "film_authoring", "Authoring")
        .expect("draft creates");
    first.original_script = "INT. WORKSHOP - NIGHT\nMIRA enters with a red parcel.".to_owned();
    first.brief = "Keep the parcel visible throughout the handoff.".to_owned();
    first.structured_brief.synopsis = "A courier completes a late delivery.".to_owned();
    first.structured_brief.style_notes = "Warm practical light; restrained camera.".to_owned();
    first.structured_brief.target_total_seconds = 12.5;
    first.structured_brief.beats = vec![crate::film_workspace::FilmBeat {
        id: "beat_delivery".to_owned(),
        summary: "Mira hands over the parcel.".to_owned(),
    }];
    first.structured_brief.dialogue = vec![crate::film_workspace::FilmDialogueLine {
        id: "line_delivery".to_owned(),
        beat_id: "beat_delivery".to_owned(),
        speaker: "MIRA".to_owned(),
        text: "Delivery. It's on the bench.".to_owned(),
    }];
    first.planning.provider = "external".to_owned();
    first.planning.connection_id = Some("connection_first".to_owned());
    first.planning.thinking_mode = "enabled".to_owned();
    first.planning.refine_prompts = true;
    first.planning.send_reference_pixels = true;
    first.production_plan.shots[0].prompt = "Mira enters with a red parcel.".to_owned();
    let first = store
        .save_film_draft(&project.id, "film_authoring", first)
        .expect("first revision saves");
    let first_locator = store
        .create_film_run_at_revision(
            &project.id,
            "run_first_authoring",
            &first.id,
            first.revision,
            vec![],
        )
        .expect("first run pins");
    let first_files = store
        .film_run_files(&project.id, &first_locator.id)
        .expect("first run files");
    let first_render_documents = [
        fs::read(&first_files.plan).expect("plan reads"),
        fs::read(&first_files.reference_pack).expect("references read"),
        fs::read(&first_files.review_plan).expect("review reads"),
    ];
    assert_eq!(
        store
            .get_film_run_authoring_snapshot(&project.id, &first_locator.id)
            .expect("snapshot reads"),
        Some(first.clone())
    );
    let snapshot_path = first_locator
        .authoring_snapshot_path
        .as_deref()
        .expect("new locator names its authoring snapshot");
    assert_eq!(
        store
            .project_file(&project.id, snapshot_path)
            .expect("snapshot is readable through the contained project-file contract")
            .path,
        first_files.authoring_snapshot.canonicalize().unwrap()
    );

    let mut second = first.clone();
    second.original_script = "EXT. STATION - DAWN\nMIRA boards the first train.".to_owned();
    second.brief = "Hold on the empty platform after departure.".to_owned();
    second.structured_brief.synopsis = "A courier leaves town after the delivery.".to_owned();
    second.structured_brief.style_notes = "Cold dawn light; locked camera.".to_owned();
    second.structured_brief.target_total_seconds = 18.0;
    second.structured_brief.beats[0].summary = "Mira boards the train.".to_owned();
    second.structured_brief.dialogue[0].text = "I made the delivery.".to_owned();
    second.planning.connection_id = Some("connection_second".to_owned());
    second.planning.thinking_mode = "disabled".to_owned();
    second.planning.refine_prompts = false;
    second.planning.send_reference_pixels = false;
    second.production_plan.shots[0].prompt = "Mira boards a train at dawn.".to_owned();
    let second = store
        .save_film_draft(&project.id, "film_authoring", second)
        .expect("second revision saves");
    assert_eq!(second.revision, first.revision + 1);
    assert_eq!(
        store
            .get_film_run_authoring_snapshot(&project.id, &first_locator.id)
            .expect("first snapshot still reads"),
        Some(first)
    );
    assert_eq!(
        first_render_documents,
        [
            fs::read(&first_files.plan).expect("pinned plan still reads"),
            fs::read(&first_files.reference_pack).expect("pinned references still read"),
            fs::read(&first_files.review_plan).expect("pinned review still reads"),
        ],
        "adding an authoring snapshot must not change existing pinned render documents"
    );

    let second_locator = store
        .create_film_run_at_revision(
            &project.id,
            "run_second_authoring",
            &second.id,
            second.revision,
            vec![],
        )
        .expect("second run pins");
    assert_eq!(second_locator.draft_revision, second.revision);
    assert_eq!(
        store
            .get_film_run_authoring_snapshot(&project.id, &second_locator.id)
            .expect("second snapshot reads"),
        Some(second)
    );
}

/// A draft persisted before the reference pack moved to version 2 keeps working (sc-24024).
///
/// The pack version is a DOCUMENT contract; a draft is project-store state with no author and
/// no edit surface for a `schemaVersion` key, so every read carries it forward. Without that,
/// the first reference or sound added to any pre-existing draft — and every pack PUT from the
/// React editor — would fail with `BadRequest` naming a version the user cannot change.
#[test]
fn a_draft_persisted_at_reference_pack_version_1_still_accepts_a_reference() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store
        .create_project("Legacy pack")
        .expect("project creates");
    let draft = store
        .create_film_draft(&project.id, "film_legacy_pack", "Legacy pack")
        .expect("draft creates");

    // Exactly what is on disk for every draft written before this commit.
    let draft_path = store
        .project_file(&project.id, "films/drafts/film_legacy_pack.json")
        .expect("the draft document is readable")
        .path;
    let mut persisted = read_json(&draft_path).expect("draft reads");
    persisted["referencePack"]["schemaVersion"] = json!(1);
    write_json(&draft_path, &persisted).expect("version 1 draft writes");
    assert_eq!(
        read_json(&draft_path).unwrap()["referencePack"]["schemaVersion"],
        json!(1),
        "the fixture has to be a version 1 document on disk"
    );

    // The read path carries it forward rather than handing the validator a stale number.
    assert_eq!(
        store
            .get_film_draft(&project.id, "film_legacy_pack")
            .expect("legacy draft reads")
            .reference_pack
            .schema_version,
        REFERENCE_PACK_SCHEMA_VERSION
    );

    let source = temp.path().join("plate.png");
    fs::write(&source, b"\x89PNG bytes").expect("source writes");
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "plate.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("asset imports");
    let updated = store
        .add_film_reference(
            &project.id,
            "film_legacy_pack",
            FilmReferenceInput {
                draft_revision: draft.revision,
                asset_id: Some(asset["id"].as_str().expect("asset id").to_owned()),
                role: "workshop_plate".to_owned(),
                kind: "plate".to_owned(),
                description: "Wide plate.".to_owned(),
                locator: None,
                approved: true,
            },
        )
        .expect("a version 1 draft still accepts a reference");
    assert_eq!(
        updated.reference_pack.schema_version,
        REFERENCE_PACK_SCHEMA_VERSION
    );
    assert_eq!(updated.reference_pack.references.len(), 1);

    // The entry this route authors names BOTH the library asset it came from and the file it
    // was stored as. `validate_reference_pack` refuses a `sourceAssetId` with no `file`
    // (sc-24025), so a route that wrote only the asset id would author a draft the workspace
    // then rejects — and the pack is validated inside `add_film_reference`, so this test is
    // the reason that refusal can never reach the UI's own add path.
    let added = &updated.reference_pack.references[0];
    assert_eq!(
        added.source_asset_id.as_deref(),
        Some(asset["id"].as_str().expect("asset id"))
    );
    assert_eq!(
        added.file(),
        Some(format!("references/{}.png", asset["id"].as_str().expect("asset id")).as_str()),
        "an image-backed role must name the file it is stored as"
    );
    assert!(crate::film_plan::validate_reference_pack(&updated.reference_pack).is_empty());
}

/// sc-24028. The Film workspace authors a DESCRIBED-ONLY role (sc-24025) through this same
/// route with no `assetId`: the panel has no second path and no second validator, so the
/// entry it stores has to be fileless, survive a re-read, and pass the core's own pack check.
#[test]
fn add_film_reference_without_an_asset_stores_a_described_only_role() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store
        .create_project("Described roles")
        .expect("project creates");
    let draft = store
        .create_film_draft(&project.id, "film_described", "Described")
        .expect("draft creates");

    let updated = store
        .add_film_reference(
            &project.id,
            "film_described",
            FilmReferenceInput {
                draft_revision: draft.revision,
                asset_id: None,
                role: "courier".to_owned(),
                kind: "character".to_owned(),
                description: "The courier: blue jacket, carries the parcel.".to_owned(),
                locator: None,
                approved: true,
            },
        )
        .expect("a role with no image is authored from its description alone");

    let added = &updated.reference_pack.references[0];
    assert!(
        added.is_described_only(),
        "a role added with no assetId must have no file: {added:?}"
    );
    assert!(
        added.source_asset_id.is_none(),
        "a fileless entry may not name an image asset either: {added:?}"
    );
    assert_eq!(
        added.description,
        "The courier: blue jacket, carries the parcel."
    );
    assert!(
        crate::film_plan::validate_reference_pack(&updated.reference_pack).is_empty(),
        "the stored pack must pass the core validator unchanged"
    );

    // It survives the round trip the workspace reloads through.
    let reloaded = store
        .get_film_draft(&project.id, "film_described")
        .expect("draft reads back");
    assert_eq!(
        reloaded.reference_pack.references,
        updated.reference_pack.references
    );
    assert!(reloaded.reference_pack.references[0].is_described_only());

    // And a described-only role with nothing said about it is refused in the CORE's words,
    // not in a sentence this route invents.
    let refused = store
        .add_film_reference(
            &project.id,
            "film_described",
            FilmReferenceInput {
                draft_revision: reloaded.revision,
                asset_id: None,
                role: "silent".to_owned(),
                kind: "character".to_owned(),
                description: "   ".to_owned(),
                locator: None,
                approved: false,
            },
        )
        .expect_err("a role with neither an image nor words says nothing");
    let message = refused.to_string();
    assert!(
        message.contains("no `file` and no `description`"),
        "expected the core's described-only refusal, got {message:?}"
    );
    assert_eq!(
        store
            .get_film_draft(&project.id, "film_described")
            .expect("draft reads")
            .reference_pack
            .references
            .len(),
        1,
        "a refused add must not leave a partial entry behind"
    );
}

/// sc-24028. Two roles on ONE library image (sc-24024): the pack ends up with one `file`
/// under two roles, and the core requires a distinct locator on each — which is the refusal
/// the reference panel surfaces under its locator field.
#[test]
fn add_film_reference_twice_on_one_asset_shares_a_file_and_requires_locators() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store
        .create_project("Shared image")
        .expect("project creates");
    let draft = store
        .create_film_draft(&project.id, "film_shared", "Shared")
        .expect("draft creates");
    let source = temp.path().join("pair.png");
    fs::write(&source, b"\x89PNG bytes").expect("source writes");
    let asset = store
        .import_asset(
            &project.id,
            UploadAsset {
                filename: "pair.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("asset imports");
    let asset_id = asset["id"].as_str().expect("asset id").to_owned();

    let first = store
        .add_film_reference(
            &project.id,
            "film_shared",
            FilmReferenceInput {
                draft_revision: draft.revision,
                asset_id: Some(asset_id.clone()),
                role: "courier".to_owned(),
                kind: "character".to_owned(),
                description: "The courier: blue jacket.".to_owned(),
                locator: Some("the woman on the left".to_owned()),
                approved: true,
            },
        )
        .expect("the first role on an image needs no locator, but may carry one");

    // A second role on the same asset with NO locator is refused, naming both roles.
    let refused = store
        .add_film_reference(
            &project.id,
            "film_shared",
            FilmReferenceInput {
                draft_revision: first.revision,
                asset_id: Some(asset_id.clone()),
                role: "guard".to_owned(),
                kind: "character".to_owned(),
                description: "The guard: grey coat.".to_owned(),
                locator: None,
                approved: true,
            },
        )
        .expect_err("two roles on one image cannot both be unlocated");
    let message = refused.to_string();
    assert!(
        message.contains("share the file") && message.contains("locator"),
        "expected the core's shared-file locator refusal, got {message:?}"
    );

    let second = store
        .add_film_reference(
            &project.id,
            "film_shared",
            FilmReferenceInput {
                draft_revision: first.revision,
                asset_id: Some(asset_id.clone()),
                role: "guard".to_owned(),
                kind: "character".to_owned(),
                description: "The guard: grey coat.".to_owned(),
                locator: Some("the man on the right".to_owned()),
                approved: true,
            },
        )
        .expect("a distinct locator on each role admits both");

    let shared_file = format!("references/{asset_id}.png");
    let files: Vec<Option<&str>> = second
        .reference_pack
        .references
        .iter()
        .map(crate::film_plan::ReferenceEntry::file)
        .collect();
    assert_eq!(
        files,
        vec![Some(shared_file.as_str()), Some(shared_file.as_str())],
        "adding one asset twice must produce ONE file under two roles"
    );
    assert!(crate::film_plan::validate_reference_pack(&second.reference_pack).is_empty());
    assert_eq!(
        store
            .get_film_draft(&project.id, "film_shared")
            .expect("draft reads")
            .reference_pack
            .references,
        second.reference_pack.references,
        "both roles survive the reload the workspace does"
    );
}

#[test]
fn legacy_film_run_reports_missing_authoring_snapshot_without_fabrication() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store.create_project("Legacy run").expect("project creates");
    let draft = store
        .create_film_draft(&project.id, "film_legacy", "Legacy")
        .expect("draft creates");
    let locator = store
        .create_film_run(&project.id, "run_legacy", &draft.id, vec![], None)
        .expect("run creates");
    let files = store
        .film_run_files(&project.id, &locator.id)
        .expect("run files");
    let locator_path = files.directory.join("locator.json");
    let mut legacy = read_json(&locator_path).expect("locator reads");
    legacy["schemaVersion"] = json!(1);
    legacy
        .as_object_mut()
        .expect("locator object")
        .remove("authoringSnapshotPath");
    write_json(&locator_path, &legacy).expect("legacy locator writes");
    fs::remove_file(&files.authoring_snapshot).expect("new-only snapshot removed");

    let legacy_locator = store
        .get_film_run(&project.id, &locator.id)
        .expect("legacy locator remains readable");
    assert_eq!(legacy_locator.schema_version, 1);
    assert_eq!(legacy_locator.authoring_snapshot_path, None);
    assert_eq!(
        store
            .get_film_run_authoring_snapshot(&project.id, &locator.id)
            .expect("legacy absence is readable"),
        None
    );
    assert!(files.plan.is_file());
    assert!(files.reference_pack.is_file());
    assert!(files.review_plan.is_file());
}

#[test]
fn film_save_and_pin_add_missing_review_questions_without_replacing_custom_questions() {
    let temp = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store.create_project("Questions").unwrap();
    let mut draft = store
        .create_film_draft(&project.id, "film_questions", "Questions")
        .unwrap();
    draft.review_plan["shots"]["SH010"]["questions"][0]["ask"] =
        json!("Is the blue parcel visible?");
    let authored = draft.review_plan["shots"]["SH010"].clone();
    let mut second = draft.production_plan.shots[0].clone();
    second.id = "SH020".to_owned();
    second.end_state = "Parcel handed over".to_owned();
    draft.production_plan.shots.push(second);
    let saved = store
        .save_film_draft(&project.id, &draft.id.clone(), draft)
        .unwrap();
    assert_eq!(saved.review_plan["shots"]["SH010"], authored);
    assert_eq!(
        saved.review_plan["shots"]["SH020"]["questions"][0]["intended"],
        "Parcel handed over"
    );
    store
        .create_film_run_at_revision(
            &project.id,
            "run_questions",
            &saved.id,
            saved.revision,
            vec![],
        )
        .unwrap();
    let files = store.film_run_files(&project.id, "run_questions").unwrap();
    assert_eq!(read_json(&files.review_plan).unwrap(), saved.review_plan);
}

#[test]
fn film_generated_topology_keeps_detached_custom_authoring_but_pins_only_current_shots() {
    let mut draft =
        crate::film_workspace::FilmDraft::manual_one_shot("project", "film", "Topology");
    draft.review_plan["shots"]["SH010"]["questions"][0]["ask"] =
        json!("Custom question retained for the old shot");
    let old = draft.review_plan["shots"]["SH010"].clone();
    draft.production_plan.shots[0].id = "SH020".to_owned();
    draft.reconcile_review_plan();
    assert_eq!(draft.review_plan["shots"]["SH010"], old);
    let pinned = draft.review_plan_for_run();
    assert!(pinned["shots"].get("SH010").is_none());
    assert_eq!(
        pinned["shots"]["SH020"],
        draft.review_plan["shots"]["SH020"]
    );
}

#[test]
fn film_run_pins_review_plan_independently_of_later_draft_edits() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test");
    let project = store
        .create_project("Pinned review")
        .expect("project creates");
    let mut draft = store
        .create_film_draft(&project.id, "film_review", "Review")
        .expect("draft creates");
    draft.review_plan["description"] = json!("Pinned first version");
    let draft = store
        .save_film_draft(&project.id, "film_review", draft)
        .expect("draft saves");
    store
        .create_film_run(
            &project.id,
            "run_review",
            "film_review",
            vec!["SH010".to_owned()],
            None,
        )
        .expect("run creates");
    let files = store
        .film_run_files(&project.id, "run_review")
        .expect("run files");
    let pinned = read_json(&files.review_plan).expect("pinned review reads");
    assert_eq!(pinned["description"], "Pinned first version");

    let mut edited = draft;
    edited.review_plan["description"] = json!("Future version");
    let edited = store
        .save_film_draft(&project.id, "film_review", edited)
        .expect("later draft saves");
    assert_eq!(
        read_json(&files.review_plan).expect("pinned review still reads"),
        pinned
    );

    let mut legacy = edited;
    legacy.review_plan = json!({"schemaVersion": 1, "questions": []});
    store
        .save_film_draft(&project.id, "film_review", legacy)
        .expect("legacy draft saves");
    store
        .create_film_run(
            &project.id,
            "run_legacy_review",
            "film_review",
            vec!["SH010".to_owned()],
            None,
        )
        .expect("legacy run creates");
    let legacy_files = store
        .film_run_files(&project.id, "run_legacy_review")
        .expect("legacy files");
    let legacy_pin = read_json(&legacy_files.review_plan).expect("legacy pin reads");
    assert_eq!(
        legacy_pin["shots"]["SH010"]["questions"][0]["frames"],
        "last"
    );
}

#[test]
fn concurrent_look_adds_do_not_lose_updates() {
    // sc-1633: create_character_look does a read-modify-write of the character
    // sidecar (read looks -> prepend -> write). The per-project file lock makes
    // overlapping calls serialize, so concurrent threads can't clobber each
    // other's appended look. Without the lock this race drops updates and the
    // final count comes up short. Asserts the lock is wired into the mutation.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let store = Arc::new(ProjectStore::new(
        temp_dir.path().join("data"),
        "test-version",
    ));
    let project = store.create_project("Race").expect("project creates");
    let character = store
        .create_character(
            &project.id,
            CharacterCreateInput {
                name: "Hero".to_owned(),
                character_type: "person".to_owned(),
                description: String::new(),
            },
        )
        .expect("character creates");
    let character_id = character
        .get("id")
        .and_then(Value::as_str)
        .expect("character id")
        .to_owned();

    let threads = 4;
    let per_thread = 12;
    std::thread::scope(|scope| {
        for worker in 0..threads {
            let store = Arc::clone(&store);
            let project_id = project.id.clone();
            let character_id = character_id.clone();
            scope.spawn(move || {
                for index in 0..per_thread {
                    store
                        .create_character_look(
                            &project_id,
                            &character_id,
                            CharacterLookInput {
                                name: format!("look-{worker}-{index}"),
                                description: String::new(),
                                approved_reference_ids: Vec::new(),
                                recipe_settings: serde_json::Map::new(),
                            },
                        )
                        .expect("look adds");
                }
            });
        }
    });

    let character = store
        .get_character(&project.id, &character_id)
        .expect("character reads");
    let looks = character
        .get("looks")
        .and_then(Value::as_array)
        .expect("looks array");
    assert_eq!(looks.len(), threads * per_thread);
}

#[test]
fn vector_workflow_intermediate_is_hidden_and_cleanup_requires_exact_ownership() {
    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test-app");
    let project = store.create_project("Vector workflow").expect("project");
    let import = |name: &str| {
        let source = temp.path().join(name);
        std::fs::write(&source, b"\x89PNG\r\n\x1a\n fixture").expect("source writes");
        store
            .import_asset(
                &project.id,
                UploadAsset {
                    filename: name.to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: source,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("asset imports")
    };
    let intermediate = import("intermediate.png");
    let user_asset = import("user.png");
    let intermediate_id = intermediate["id"].as_str().expect("id").to_owned();
    let user_asset_id = user_asset["id"].as_str().expect("id").to_owned();

    let marked = store
        .mark_vector_workflow_intermediate(
            &project.id,
            &intermediate_id,
            "vwf_owned",
            "job_parent",
            "job_child",
        )
        .expect("intermediate marked");
    assert_eq!(marked["origin"], json!("vector_workflow_intermediate"));
    assert_eq!(
        marked["extra"]["vectorWorkflow"],
        json!({
            "role": "retained_intermediate",
            "publication": "unpublished",
            "workflowId": "vwf_owned",
            "parentJobId": "job_parent",
            "childJobId": "job_child",
            "hidden": true
        })
    );
    let listed = store
        .list_assets(&project.id, true, true, AssetScope::All)
        .expect("assets list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["id"], json!(user_asset_id));
    assert!(store.get_asset(&project.id, &intermediate_id).is_ok());

    assert!(!store
        .purge_vector_workflow_intermediate(
            &project.id,
            &intermediate_id,
            "vwf_other",
            "job_parent",
        )
        .expect("foreign workflow refuses"));
    assert!(!store
        .purge_vector_workflow_intermediate(&project.id, &user_asset_id, "vwf_owned", "job_parent",)
        .expect("user asset refuses"));
    assert!(store.get_asset(&project.id, &intermediate_id).is_ok());
    assert!(store.get_asset(&project.id, &user_asset_id).is_ok());

    assert!(store
        .purge_vector_workflow_intermediate(
            &project.id,
            &intermediate_id,
            "vwf_owned",
            "job_parent",
        )
        .expect("owned cleanup"));
    assert!(matches!(
        store.get_asset(&project.id, &intermediate_id),
        Err(ProjectStoreError::NotFound(_))
    ));
    assert!(store.get_asset(&project.id, &user_asset_id).is_ok());
    assert!(!store
        .purge_vector_workflow_intermediate(
            &project.id,
            &intermediate_id,
            "vwf_owned",
            "job_parent",
        )
        .expect("cleanup is idempotent"));
}

/// issue #1435 / sc-11855: a workspace folder that rejects writes must fail
/// with the actionable `StorageNotWritable` error (naming the folder), not a
/// raw `SQLITE_READONLY`/`unable to open database file` that the UI shows
/// verbatim. A read-only `projects` dir is the portable stand-in for the
/// user's granular write denial.
#[cfg(unix)]
#[test]
fn create_project_on_unwritable_projects_dir_reports_storage_not_writable() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().expect("temp dir");
    let store = ProjectStore::new(temp.path().join("data"), "test-app");
    let projects = store.projects_dir();
    std::fs::create_dir_all(&projects).expect("seed projects dir");
    std::fs::set_permissions(&projects, std::fs::Permissions::from_mode(0o555))
        .expect("lock projects dir read-only");

    let result = store.create_project("Test");

    // Restore perms so the tempdir teardown can remove the tree.
    let _ = std::fs::set_permissions(&projects, std::fs::Permissions::from_mode(0o755));

    match result {
        Err(ProjectStoreError::StorageNotWritable(detail)) => {
            assert!(
                detail.contains("workspace folder") && detail.contains("Settings"),
                "message should be actionable and name the workspace folder: {detail}"
            );
        }
        other => panic!("expected StorageNotWritable, got {other:?}"),
    }
}
// sc-26687: a recorded recipe is the ONLY trace of what a user asked for, so the ordered
// conditioning and the imported identity have to survive it exactly. These assert the VALUES a
// replay reads back — list ORDER included, because the engine labels the supplied media
// positionally — rather than that some field was passed through.
fn imported_video_fact() -> Value {
    json!({
        "mediaPath": "assets/videos/genset_h3/clip.mp4",
        "mimeType": "video/mp4",
        "type": "video",
        "mode": "reference_to_video",
        "model": "my_singularity_ref2va",
        "family": "minimax-h3",
        "prompt": "walk",
        "negativePrompt": "",
        "seed": 7,
        "loras": [],
        "duration": 6,
        "fps": 25,
        "width": 768,
        "height": 512,
        "sourceClipAssetIds": ["clip_c", "clip_a", "clip_b"],
        "referenceAssetIds": ["img_3", "img_1", "img_2"],
        "referenceAudioAssetIds": ["aud_2", "aud_1"],
        "rawAdapterSettings": {
            "importPlan": {
                "checkpointId": "managed/install-42",
                "planId": "checkpoint-plan-7",
                "semanticDigest": "abc",
                "assembly": { "schemaVersion": 1, "partition": "ref2va" }
            }
        }
    })
}

#[test]
fn video_recipe_round_trips_ordered_conditioning_and_the_imported_identity() {
    let (_, recipe, _) = build_video_sidecar_parts("job-1", &imported_video_fact());
    let settings = &recipe["normalizedSettings"];
    // Deliberately NOT sorted: the recorded order is the order the user picked, and a replay
    // that re-orders the references conditions the render on different media positions.
    assert_eq!(
        settings["sourceClipAssetIds"],
        json!(["clip_c", "clip_a", "clip_b"])
    );
    assert_eq!(
        settings["referenceAssetIds"],
        json!(["img_3", "img_1", "img_2"])
    );
    assert_eq!(
        settings["referenceAudioAssetIds"],
        json!(["aud_2", "aud_1"])
    );
    assert_eq!(recipe["model"], json!("my_singularity_ref2va"));
    assert_eq!(
        recipe["importedModelRef"]["modelId"],
        json!("my_singularity_ref2va")
    );
    assert_eq!(
        recipe["importedModelRef"]["checkpointId"],
        json!("managed/install-42")
    );
    assert_eq!(
        recipe["importedModelRef"]["planId"],
        json!("checkpoint-plan-7")
    );
    assert_eq!(recipe["importedModelRef"]["semanticDigest"], json!("abc"));
    assert_eq!(
        recipe["importedModelRef"]["declaredPartition"],
        json!("ref2va")
    );
}

#[test]
fn image_recipe_round_trips_the_ordered_reference_list_and_the_edit_source() {
    let fact = json!({
        "mediaPath": "assets/images/genset_edit/out.png",
        "mimeType": "image/png",
        "type": "image",
        "mode": "edit_image",
        "model": "krea_transformer",
        "family": "krea",
        "prompt": "make it dusk",
        "negativePrompt": "",
        "seed": 3,
        "loras": [],
        "normalizedWidth": 1024,
        "normalizedHeight": 1024,
        "count": 1,
        "sourceAssetId": "src_1",
        "referenceAssetIds": ["ref_b", "ref_a", "ref_c"],
        "rawAdapterSettings": {}
    });
    let (_, recipe, _) = build_image_sidecar_parts("job-2", &fact);
    let settings = &recipe["normalizedSettings"];
    assert_eq!(settings["sourceAssetId"], json!("src_1"));
    assert_eq!(
        settings["referenceAssetIds"],
        json!(["ref_b", "ref_a", "ref_c"])
    );
}

#[test]
fn a_builtin_generation_records_no_imported_identity() {
    let mut fact = imported_video_fact();
    fact["rawAdapterSettings"] = json!({});
    let (_, recipe, _) = build_video_sidecar_parts("job-3", &fact);
    assert!(
        recipe.get("importedModelRef").is_none(),
        "a generation with no bound plan must not claim an import"
    );
    // The ordered conditioning is a property of the REQUEST, not of the import, so it is
    // recorded either way — the two must not be lost together.
    assert_eq!(
        recipe["normalizedSettings"]["referenceAssetIds"],
        json!(["img_3", "img_1", "img_2"])
    );
}
