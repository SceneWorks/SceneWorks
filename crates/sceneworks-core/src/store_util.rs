use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::OnceLock;

use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use rusqlite::Connection;
use serde::Serialize;
use serde_json::Value;

use crate::contracts::StringEnum;

use crate::project_store::{ProjectStoreError, ProjectStoreResult};

/// All stores wait up to five seconds for a competing process. Writers must also
/// use BEGIN IMMEDIATE before reading, since SQLite cannot wait on lock upgrades.
pub(crate) const SQLITE_BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Preserve each store's established durability and concurrency policy.
/// Project (including character, voice and training indexes) keeps SQLite's
/// rollback journal and default synchronization. Jobs attempts WAL but warns and
/// continues if unavailable; Catalog requires WAL and uses NORMAL synchronization,
/// disk-backed temporary sorts and a bounded page cache/checkpoint interval.
pub(crate) enum Profile<'a> {
    Project,
    Jobs { database_path: &'a Path },
    Catalog { wal_autocheckpoint_pages: u32 },
}

pub(crate) fn configure_connection(
    connection: &Connection,
    profile: Profile<'_>,
) -> rusqlite::Result<()> {
    connection.busy_timeout(SQLITE_BUSY_TIMEOUT)?;
    match profile {
        Profile::Project => {}
        Profile::Jobs { database_path } => {
            if let Err(error) = connection.pragma_update(None, "journal_mode", "wal") {
                // Never remove live WAL/SHM sidecars or silently force DELETE mode.
                tracing::warn!(
                    event = "sqlite_wal_enable_failed",
                    dbPath = %database_path.display(),
                    error = %error,
                    "could not enable SQLite WAL mode; continuing in the default rollback-journal \
                     mode — cross-process write concurrency will be more serialized than usual"
                );
            }
            connection.pragma_update(None, "foreign_keys", "on")?;
        }
        Profile::Catalog {
            wal_autocheckpoint_pages,
        } => {
            let journal_mode: String =
                connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
            if !journal_mode.eq_ignore_ascii_case("wal") {
                connection.pragma_update(None, "journal_mode", "wal")?;
            }
            connection.pragma_update(None, "synchronous", "normal")?;
            connection.pragma_update(None, "foreign_keys", true)?;
            connection.pragma_update(None, "temp_store", "file")?;
            connection.pragma_update(None, "cache_size", -65_536)?;
            connection.pragma_update(None, "wal_autocheckpoint", wal_autocheckpoint_pages)?;
        }
    }
    Ok(())
}

pub(crate) fn read_json(path: &Path) -> ProjectStoreResult<Value> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

/// Store-error adapter for the shared neutral filesystem writer.
pub(crate) fn atomic_write(path: &Path, contents: &[u8]) -> ProjectStoreResult<()> {
    Ok(crate::fsutil::write_atomic(path, contents)?)
}

pub(crate) fn write_json<T: Serialize>(path: &Path, payload: &T) -> ProjectStoreResult<()> {
    let mut output = serde_json::to_string_pretty(payload)?;
    output.push('\n');
    atomic_write(path, output.as_bytes())
}

const PROJECT_LOCK_STRIPES: usize = 256;

/// Process-wide striped locks that serialize file read-modify-write within a
/// single project. A store resolves the project path, then holds this guard
/// across its read -> mutate -> write so concurrent API/worker mutations to the
/// same project can't lose each other's sidecar/JSON updates (sc-1633).
///
/// Striping (rather than a per-path map) keeps this allocation-free with no
/// unbounded growth; distinct projects only contend on the rare hash collision.
/// The lock is *reentrant* so a mutating method that calls another mutating
/// method on the same project (e.g. `create_timeline` -> `save_timeline`) does
/// not self-deadlock on the single blocking thread that runs the call chain.
///
/// Scope: this guards mutations that flow through `ProjectStore` within one
/// process. It does not coordinate separate OS processes writing the same
/// project directory; that is out of scope for the single-API desktop model.
fn project_locks() -> &'static [ReentrantMutex<()>] {
    static LOCKS: OnceLock<Vec<ReentrantMutex<()>>> = OnceLock::new();
    LOCKS.get_or_init(|| {
        (0..PROJECT_LOCK_STRIPES)
            .map(|_| ReentrantMutex::new(()))
            .collect()
    })
}

pub(crate) fn lock_project_files(project_path: &Path) -> ReentrantMutexGuard<'static, ()> {
    lock_store_path(project_path)
}

/// Same striped process-wide lock, keyed on an arbitrary store file/dir path
/// rather than a project dir — for stores that live outside a project (e.g. the
/// config-dir credential file, sc-8813). Hold the guard across the whole
/// read -> mutate -> write. Same scope caveat as [`lock_project_files`]: it
/// serializes threads within one process, not separate OS processes.
pub(crate) fn lock_store_path(path: &Path) -> ReentrantMutexGuard<'static, ()> {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    let index = (hasher.finish() as usize) % PROJECT_LOCK_STRIPES;
    project_locks()[index].lock()
}

pub(crate) fn relative_string(root: &Path, path: &Path) -> ProjectStoreResult<String> {
    Ok(path
        .strip_prefix(root)
        .map_err(|_| ProjectStoreError::BadRequest("Path is outside project".to_owned()))?
        .to_string_lossy()
        .replace('\\', "/"))
}

pub fn is_safe_relative_path(relative_path: &str) -> bool {
    crate::safe_path::relative_normal_components(relative_path).is_ok()
}

pub use crate::ids::{is_safe_id, is_safe_id_with_max};

/// Required string field; whitespace and empty strings are validated by each store's domain rules.
pub(crate) fn required_str<'a>(value: &'a Value, key: &str) -> ProjectStoreResult<&'a str> {
    required_str_with(value, key, || format!("Missing required field: {key}"))
}

pub(crate) fn required_str_with<'a>(
    value: &'a Value,
    key: &str,
    message: impl FnOnce() -> String,
) -> ProjectStoreResult<&'a str> {
    optional_str(value, key).ok_or_else(|| ProjectStoreError::BadRequest(message()))
}

pub(crate) fn optional_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

pub(crate) fn optional_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

pub(crate) fn optional_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

pub(crate) fn optional_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

/// Generate a random lowercase-hex id of `bytes` random bytes (so `2 * bytes`
/// hex chars). Previously this ran `hex(randomblob(n))` through a fresh
/// in-memory SQLite connection per call, so saving a timeline with N new items
/// opened N connections and turned SQLite-init failures into id-generation
/// failures (sc-4209 / F-CORE-5). Now it pulls from the OS CSPRNG via
/// `getrandom` and hex-encodes in Rust — no connection, no SQLite failure
/// surface.
pub(crate) fn random_hex(bytes: usize) -> ProjectStoreResult<String> {
    crate::hashing::random_hex(bytes)
        .map_err(|error| ProjectStoreError::Io(std::io::Error::other(error.to_string())))
}

/// Deserialize a single string into one of the crate's [`StringEnum`] types.
///
/// The [`StringEnum`] bound (not a bare `DeserializeOwned`) is what makes the
/// `expect` sound: every `string_enum!`-generated type maps an unrecognized
/// string to its `Unknown(String)` variant, so deserializing a `String` value
/// can't fail. The bound stops this helper from being reused with a stricter
/// enum whose `Deserialize` could error and panic the whole store call on one
/// bad DB row (sc-4269 / F-CORE-9).
pub(crate) fn parse_string_enum<T>(value: &str) -> T
where
    T: StringEnum,
{
    serde_json::from_value(Value::String(value.to_owned()))
        .expect("string enum deserialization is infallible")
}

/// Idempotent additive migration: add `column` (`column definition`) to `table`
/// if `pragma table_info` doesn't already list it.
///
/// Hoisted from three near-identical per-store copies that had drifted on how
/// they read the PRAGMA column name — one used the positional index `1`, the
/// others the `"name"` accessor. This standardizes on the by-name accessor
/// (sc-4271 / F-CORE-11). Returns the raw `rusqlite::Error` so each store's `?`
/// maps it into its own error type.
pub(crate) fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> rusqlite::Result<()> {
    let mut statement = connection.prepare(&format!("pragma table_info({table})"))?;
    let exists = statement
        .query_map([], |row| row.get::<_, String>("name"))?
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|existing| existing == column);
    if !exists {
        connection.execute(
            &format!("alter table {table} add column {column} {definition}"),
            [],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{atomic_write, ensure_column, lock_project_files, parse_string_enum, random_hex};
    use crate::contracts::JobStatus;
    use rusqlite::Connection;
    use std::collections::HashSet;

    #[test]
    fn connection_profiles_preserve_store_pragma_contracts() {
        let temp = tempfile::tempdir().unwrap();
        for (name, profile, journal, foreign_keys, synchronous) in [
            ("project", super::Profile::Project, None, None, None),
            (
                "jobs",
                super::Profile::Jobs {
                    database_path: temp.path(),
                },
                Some("wal"),
                Some(1),
                None,
            ),
            (
                "catalog",
                super::Profile::Catalog {
                    wal_autocheckpoint_pages: 20_000,
                },
                Some("wal"),
                Some(1),
                Some(1),
            ),
        ] {
            let connection = Connection::open(temp.path().join(format!("{name}.db"))).unwrap();
            // Preserve the actual bundled SQLite defaults rather than assuming
            // foreign_keys is disabled (bundled builds may enable it at compile time).
            let default_foreign_keys: i64 = connection
                .pragma_query_value(None, "foreign_keys", |row| row.get(0))
                .unwrap();
            let default_synchronous: i64 = connection
                .pragma_query_value(None, "synchronous", |row| row.get(0))
                .unwrap();
            let default_journal: String = connection
                .pragma_query_value(None, "journal_mode", |row| row.get(0))
                .unwrap();
            super::configure_connection(&connection, profile).unwrap();
            let actual_journal: String = connection
                .pragma_query_value(None, "journal_mode", |row| row.get(0))
                .unwrap();
            assert_eq!(actual_journal, journal.unwrap_or(&default_journal));
            for (pragma, expected) in [
                (
                    "busy_timeout",
                    super::SQLITE_BUSY_TIMEOUT.as_millis() as i64,
                ),
                ("foreign_keys", foreign_keys.unwrap_or(default_foreign_keys)),
                ("synchronous", synchronous.unwrap_or(default_synchronous)),
            ] {
                let actual: i64 = connection
                    .pragma_query_value(None, pragma, |row| row.get(0))
                    .unwrap();
                assert_eq!(actual, expected, "{name} {pragma}");
            }
            if name == "catalog" {
                for (pragma, expected) in [
                    ("temp_store", 1),
                    ("cache_size", -65_536),
                    ("wal_autocheckpoint", 20_000),
                ] {
                    let actual: i64 = connection
                        .pragma_query_value(None, pragma, |row| row.get(0))
                        .unwrap();
                    assert_eq!(actual, expected, "catalog {pragma}");
                }
            }
        }
    }

    #[test]
    fn required_strings_preserve_domain_text_and_diagnostics() {
        let value = serde_json::json!({"raw":"  text  ","empty":"","wrong":7});
        assert_eq!(super::required_str(&value, "raw").unwrap(), "  text  ");
        assert_eq!(super::required_str(&value, "empty").unwrap(), "");
        for key in ["wrong", "missing"] {
            assert_eq!(
                super::required_str(&value, key).unwrap_err().to_string(),
                format!("Missing required field: {key}")
            );
            assert_eq!(
                super::required_str_with(&value, key, || "Display Name is required".into())
                    .unwrap_err()
                    .to_string(),
                "Display Name is required"
            );
        }
    }

    /// sc-4271 / F-CORE-11: the hoisted `ensure_column` adds a missing column and
    /// is a no-op when it already exists (the three per-store copies it replaced
    /// disagreed on the PRAGMA accessor; this is the single by-name version).
    #[test]
    fn ensure_column_adds_missing_column_and_is_idempotent() {
        let connection = Connection::open_in_memory().expect("in-memory db");
        connection
            .execute("create table widgets (id text primary key)", [])
            .expect("seed table");

        let has_color = |conn: &Connection| {
            let mut statement = conn.prepare("pragma table_info(widgets)").expect("pragma");
            let columns: Vec<String> = statement
                .query_map([], |row| row.get::<_, String>("name"))
                .expect("columns")
                .filter_map(Result::ok)
                .collect();
            columns.iter().any(|name| name == "color")
        };
        assert!(!has_color(&connection), "precondition: no color column");

        ensure_column(&connection, "widgets", "color", "text").expect("adds column");
        assert!(has_color(&connection), "column added");

        // Second call is a no-op (would error if it re-ran the ALTER).
        ensure_column(&connection, "widgets", "color", "text").expect("idempotent");
        assert!(has_color(&connection));
    }

    /// sc-4269 / F-CORE-9: the `StringEnum` bound makes `parse_string_enum`
    /// infallible — a value never written by the app (or a hand-edited DB row)
    /// maps to the enum's `Unknown` variant instead of panicking, and known
    /// values still map to their variant.
    #[test]
    fn parse_string_enum_maps_unknown_without_panicking() {
        assert_eq!(
            parse_string_enum::<JobStatus>("totally-bogus-status"),
            JobStatus::Unknown("totally-bogus-status".to_owned())
        );
        assert_eq!(
            parse_string_enum::<JobStatus>("running"),
            JobStatus::Running
        );
    }

    /// sc-4209 / F-CORE-5: `random_hex(n)` returns `2n` lowercase-hex chars,
    /// generated without a SQLite connection. Covers the length/charset contract
    /// and that successive ids differ (i.e. it is actually random, not constant).
    #[test]
    fn random_hex_produces_lowercase_hex_of_expected_length() {
        for bytes in [1usize, 4, 16] {
            let id = random_hex(bytes).expect("id generates");
            assert_eq!(
                id.len(),
                bytes * 2,
                "{bytes} bytes -> {} hex chars",
                bytes * 2
            );
            assert!(
                id.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "id {id:?} is lowercase hex"
            );
        }

        // Distinct across many calls — a constant/low-entropy generator would collide.
        let ids = (0..1000)
            .map(|_| random_hex(16).expect("id generates"))
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), 1000, "16-byte ids are unique across 1000 draws");
    }

    /// sc-8949 / F-147: `atomic_write` fsyncs the temp before renaming, so the
    /// committed file carries the full contents and no `.tmp` staging file is left
    /// behind. (A power-loss torn write can't be reproduced in a unit test, but this
    /// pins the write-then-sync-then-rename result and guards against a regression
    /// that drops the sync and re-introduces the deterministic temp name.)
    #[test]
    fn atomic_write_commits_full_contents_and_leaves_no_temp() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nested").join("doc.json");
        let payload = br#"{"k":"v"}"#;
        atomic_write(&path, payload).expect("atomic_write succeeds");

        assert_eq!(std::fs::read(&path).expect("file reads"), payload);

        // No `*.tmp` staging file survived the commit.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .expect("dir reads")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "unexpected temp files: {leftovers:?}");
    }

    #[test]
    fn atomic_write_tolerates_concurrent_writers_to_same_path() {
        // sc-1633: the temp file is uniquely named, so overlapping writers to the
        // same target never collide. With the old deterministic `*.tmp` name they
        // interleaved bytes and the second `rename` failed (the temp was already
        // renamed away) — which would surface here as an Err from `atomic_write`.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("doc.json");
        std::thread::scope(|scope| {
            for writer in 0..8 {
                let path = path.clone();
                scope.spawn(move || {
                    for _ in 0..50 {
                        atomic_write(&path, format!("{{\"writer\":{writer}}}").as_bytes())
                            .expect("atomic_write never collides on its temp file");
                    }
                });
            }
        });
        // The final file is exactly one writer's complete value, never torn.
        let contents = std::fs::read_to_string(&path).expect("file reads");
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).expect("final contents are valid json");
        assert!(parsed
            .get("writer")
            .and_then(|value| value.as_i64())
            .is_some());
    }

    #[test]
    fn lock_project_files_serializes_read_modify_write() {
        // sc-1633: holding the per-project lock across read -> mutate -> write makes
        // overlapping increments serialize. Without it the read/write window races
        // and the total comes up short of the expected count.
        let dir = tempfile::tempdir().expect("temp dir");
        let project = dir.path().join("project");
        std::fs::create_dir_all(&project).expect("project dir");
        let counter = project.join("counter.json");
        std::fs::write(&counter, "0").expect("counter seed");

        let threads = 8;
        let per_thread = 50;
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let project = project.clone();
                let counter = counter.clone();
                scope.spawn(move || {
                    for _ in 0..per_thread {
                        let _guard = lock_project_files(&project);
                        let current: i64 = std::fs::read_to_string(&counter)
                            .expect("counter reads")
                            .trim()
                            .parse()
                            .expect("counter parses");
                        atomic_write(&counter, (current + 1).to_string().as_bytes())
                            .expect("counter writes");
                    }
                });
            }
        });

        let total: i64 = std::fs::read_to_string(&counter)
            .expect("counter reads")
            .trim()
            .parse()
            .expect("counter parses");
        assert_eq!(total, (threads * per_thread) as i64);
    }
}
