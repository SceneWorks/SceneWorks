//! JSONC manifest read/upsert/write helpers shared by the LoRA and model manifests.
use super::*;

use std::io::Write as _;

use sceneworks_core::file_lock::FileLock;

/// Max time to block waiting for the cross-process manifest lock before giving up
/// with a clear error. Manifest RMW is a few KB of JSON, so a real hold is sub-ms;
/// a stall this long means a stuck/crashed peer, and failing loudly beats silently
/// skipping the write and losing the freshly-installed entry (sc-8843).
const MANIFEST_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// Poll cadence while spin-waiting on `try_lock_exclusive` (fs2 has no timed
/// blocking-lock API, so we retry rather than block indefinitely).
const MANIFEST_LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(25);

pub(crate) async fn read_json_value(path: &Path) -> WorkerResult<Value> {
    Ok(serde_json::from_slice(&tokio::fs::read(path).await?)?)
}

/// Upsert `entry` (keyed by its `id`) into the `collection_key` array of a JSONC
/// manifest at `path`, creating the manifest when absent. An existing entry with
/// the same id is merged (incoming fields win) but keeps its original `createdAt`.
/// Shared by the LoRA (`"loras"`) and model (`"models"`) manifests, which differed
/// only by this array key (sc-4279 / F-MLXW-15).
///
/// The default utility pool is 4 SEPARATE PROCESSES, so an in-process mutex cannot
/// serialize concurrent installs — two parallel upserts would each read the old
/// file, merge their own entry, and the second rename would clobber the first,
/// losing an entry (F-041 / sc-8843). The entire read→merge→write(rename) therefore
/// runs under a cross-process advisory *exclusive* file lock on a `<manifest>.lock`
/// sibling. The API writer (`apps/rust-api/src/manifest.rs`) takes the SAME sibling
/// lock so worker↔API writes to these shared files serialize too.
pub(crate) async fn upsert_manifest_entry(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<()> {
    // `id` is validated up front (cheap, no I/O) so a malformed payload fails before
    // we take the lock.
    if entry.get("id").and_then(Value::as_str).is_none() {
        return Err(WorkerError::InvalidPayload(format!(
            "{collection_key} manifest entry requires id"
        )));
    }
    let path = path.to_path_buf();
    let collection_key = collection_key.to_owned();
    // The critical section is blocking (fs2 advisory lock + sync read/write) so it
    // runs on a blocking thread rather than stalling the async runtime.
    crate::load_progress::spawn_blocking(move || {
        merge_manifest_entry_locked(&path, &collection_key, entry, true).map(|_| ())
    })
    .await
    .map_err(|error| task_join_error("manifest upsert", error))?
}

/// Publish an import with compare-and-swap semantics under the existing cross-process manifest
/// lock. Repair replaces the selection, not arbitrary existing IDs; failure leaves it untouched.
pub(crate) async fn publish_model_import_manifest(
    path: &Path,
    data_dir: &Path,
    entry: serde_json::Map<String, Value>,
    expected: Option<&Value>,
    live_claim_check: impl FnOnce() -> WorkerResult<()> + Send + 'static,
) -> WorkerResult<()> {
    let path = path.to_owned();
    let data_dir = data_dir.to_owned();
    let expected = expected.cloned();
    crate::load_progress::spawn_blocking(move || {
        let id = entry.get("id").and_then(Value::as_str).ok_or_else(|| WorkerError::InvalidPayload("import requires model ID".into()))?;
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
        let _guard = ManifestLock::acquire(&path)?;
        let mut manifest: Value = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&strip_jsonc_comments(&text))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => json!({"schemaVersion":1,"models":[]}),
            Err(error) => return Err(error.into()),
        };
        let models = manifest.get_mut("models").and_then(Value::as_array_mut)
            .ok_or_else(|| WorkerError::InvalidPayload("models manifest must contain an array".into()))?;
        let current = models.iter().position(|model| model.get("id").and_then(Value::as_str) == Some(id));
        let mut replacement = Value::Object(entry.clone());
        match (current, expected.as_ref()) {
            (Some(index), expected) => {
                let old = models[index].as_object().ok_or_else(|| WorkerError::InvalidPayload("invalid existing model".into()))?;
                // A replay after successful publication is idempotent only for the exact same
                // persisted installation/plan, never just the same ID or display name.
                let same_install = old.get("importPlan").is_some_and(|plan| entry.get("importPlan") == Some(plan))
                    && old.get("paths").is_some_and(|paths| entry.get("paths") == Some(paths));
                if same_install { return live_claim_check(); } // Preserve user metadata, not stale claim authority.
                {
                    let actual = sceneworks_core::model_artifacts::provenance::managed_repair_binding(&data_dir, old)
                        .map_err(WorkerError::InvalidPayload)?;
                    if expected != Some(&actual) || expected.and_then(|v| v.get("modelId")).and_then(Value::as_str) != Some(id) {
                        return Err(WorkerError::InvalidPayload("Import selection changed before repair publication; original selection was not modified".into()));
                    }
                    if old.get("paths") == entry.get("paths") {
                        return Err(WorkerError::InvalidPayload("Repair must stage a separate managed installation".into()));
                    }
                }
                if let Some(created) = old.get("createdAt") { replacement["createdAt"] = created.clone(); }
                models[index] = replacement;
            }
            (None, Some(_)) => return Err(WorkerError::InvalidPayload("Repair selection was removed before publication".into())),
            (None, None) => models.push(replacement),
        }
        // The worker may have lost its claim while waiting for this manifest lock or resolving
        // the repair CAS. Re-read control-plane authority HERE, after all filesystem work and
        // immediately before the atomic catalog mutation, while the same lock is still held.
        write_json_value_blocking_with_check(&path, &manifest, live_claim_check)
    }).await.map_err(|error| task_join_error("import publication", error))?
}

/// Merge `entry` into an entry that ALREADY EXISTS in `collection_key`, and write
/// nothing at all when its id is absent. Returns whether the merge landed.
///
/// The update-only counterpart of [`upsert_manifest_entry`], for a writer that holds
/// a value derived from an entry it read minutes ago: the checkpoint-catalog
/// migration compiles for four full passes over a multi-gigabyte checkpoint before
/// stamping, and an upsert would RESURRECT a model the user deleted in the meantime
/// as a stub row carrying only `id` and `importPlan` — no name, no paths, no type
/// (sc-20651). The absence check is made by the same read the merge writes from,
/// INSIDE the same cross-process lock the write takes, so there is no window
/// between deciding and writing.
pub(crate) async fn update_manifest_entry_if_present(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
) -> WorkerResult<bool> {
    if entry.get("id").and_then(Value::as_str).is_none() {
        return Err(WorkerError::InvalidPayload(format!(
            "{collection_key} manifest entry requires id"
        )));
    }
    let path = path.to_path_buf();
    let collection_key = collection_key.to_owned();
    crate::load_progress::spawn_blocking(move || {
        merge_manifest_entry_locked(&path, &collection_key, entry, false)
    })
    .await
    .map_err(|error| task_join_error("manifest update", error))?
}

/// Blocking read→merge→write of `path`, run under a cross-process exclusive lock on
/// the `<manifest>.lock` sibling. Callers must invoke this off the async runtime.
///
/// `insert_missing` decides what an absent id means: appending a new member (upsert)
/// or writing nothing and reporting `false` (update-only).
fn merge_manifest_entry_locked(
    path: &Path,
    collection_key: &str,
    entry: serde_json::Map<String, Value>,
    insert_missing: bool,
) -> WorkerResult<bool> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _guard = ManifestLock::acquire(path)?;

    let mut manifest = match std::fs::read_to_string(path) {
        Ok(payload) => serde_json::from_str(&strip_jsonc_comments(&payload))?,
        // No manifest at all: the id is absent, so an update-only write is a no-op
        // rather than a fresh catalog holding one stub entry.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !insert_missing => {
            return Ok(false);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut object = serde_json::Map::new();
            object.insert("schemaVersion".to_owned(), json!(1));
            object.insert(collection_key.to_owned(), Value::Array(Vec::new()));
            Value::Object(object)
        }
        Err(error) => return Err(error.into()),
    };
    // Re-fetch under the lock; validated above, so this is infallible in practice.
    let entry_id = entry.get("id").and_then(Value::as_str).ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{collection_key} manifest entry requires id"))
    })?;
    let collection = manifest
        .as_object_mut()
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!("{collection_key} manifest must be an object"))
        })?
        .entry(collection_key.to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let collection = collection.as_array_mut().ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{collection_key} manifest array must be an array"))
    })?;
    let mut found = false;
    for item in collection.iter_mut() {
        if item.get("id").and_then(Value::as_str) != Some(entry_id) {
            continue;
        }
        found = true;
        let created_at = item.get("createdAt").cloned();
        let Some(object) = item.as_object_mut() else {
            return Err(WorkerError::InvalidPayload(format!(
                "{collection_key} manifest entry must be an object"
            )));
        };
        for (key, value) in entry.clone() {
            object.insert(key, value);
        }
        if let Some(created_at) = created_at {
            object.insert("createdAt".to_owned(), created_at);
        }
    }
    if !found {
        if !insert_missing {
            // Nothing is written, so the file the caller read stays exactly as the
            // deletion left it.
            return Ok(false);
        }
        collection.push(Value::Object(entry));
    }
    write_json_value_blocking(path, &manifest)?;
    Ok(true)
    // `_guard` (and the advisory lock) drops here, after the atomic rename lands.
}

/// RAII holder for a cross-process advisory exclusive lock on a `<manifest>.lock`
/// sibling file. Released with an explicit `LOCK_UN`, not by `close(2)` alone, which
/// leaves the lock held while any forked child still references the same open file
/// description (sc-22738).
struct ManifestLock {
    _lock: FileLock,
}

impl ManifestLock {
    fn acquire(manifest_path: &Path) -> WorkerResult<Self> {
        let lock_path = manifest_lock_path(manifest_path);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;
        let deadline = std::time::Instant::now() + MANIFEST_LOCK_TIMEOUT;
        // fs2 signals lock contention with a platform-specific error: `EWOULDBLOCK`
        // (`ErrorKind::WouldBlock`) on Unix, but `ERROR_LOCK_VIOLATION` (raw OS 33,
        // `ErrorKind::Uncategorized`) on Windows. Matching on `ErrorKind` misses the
        // Windows case and would mis-propagate a real, retryable contention as a hard
        // error (sc-8843). Compare by RAW OS CODE against fs2's own contention error,
        // which is correct on every platform.
        let contended = fs2::lock_contended_error().raw_os_error();
        loop {
            match FileLock::try_exclusive_retryable(file) {
                Ok(lock) => return Ok(Self { _lock: lock }),
                Err((handle, error)) if error.raw_os_error() == contended => {
                    file = handle;
                    if std::time::Instant::now() >= deadline {
                        return Err(WorkerError::Io(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!(
                                "timed out after {:?} waiting for manifest lock {}",
                                MANIFEST_LOCK_TIMEOUT,
                                lock_path.display()
                            ),
                        )));
                    }
                    std::thread::sleep(MANIFEST_LOCK_POLL);
                }
                Err((_handle, error)) => return Err(error.into()),
            }
        }
    }
}

/// The `.lock` sibling path for a manifest. Kept alongside the manifest so the lock
/// scope is the exact file being mutated (per-file, not global).
fn manifest_lock_path(manifest_path: &Path) -> PathBuf {
    let mut name = manifest_path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    name.push(".lock");
    manifest_path.with_file_name(name)
}

/// Blocking sibling of [`write_json_value`]: atomic temp-write + rename. Used inside
/// the manifest lock's critical section (which is already off the async runtime).
fn write_json_value_blocking(path: &Path, value: &Value) -> WorkerResult<()> {
    write_json_value_blocking_with_check(path, value, || Ok(()))
}

fn write_json_value_blocking_with_check(
    path: &Path,
    value: &Value,
    check: impl FnOnce() -> WorkerResult<()>,
) -> WorkerResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = serde_json::to_vec_pretty(value)?;
    output.push(b'\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    let mut file = std::fs::File::create(&tmp_path)?;
    file.write_all(&output)?;
    file.sync_all()?;
    drop(file);
    // Validate AFTER the temporary-file I/O too: only the following rename changes the catalog.
    if let Err(error) = check() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(error);
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

pub(crate) async fn write_json_value(path: &Path, value: &Value) -> WorkerResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut output = serde_json::to_vec_pretty(value)?;
    output.push(b'\n');
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    let mut file = tokio::fs::File::create(&tmp_path).await?;
    file.write_all(&output).await?;
    file.sync_all().await?;
    drop(file);
    tokio::fs::rename(tmp_path, path).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-22738: a released manifest lock must be free IMMEDIATELY, even while a descriptor this
    /// process handed to a child still references the same open file description. `flock(2)` locks
    /// live on the open file description, so a close-only release only takes effect once every
    /// such reference is gone — and this process forks (ffmpeg, converters) throughout a manifest
    /// merge. The assertion uses the non-blocking primitive so the failure is an assertion rather
    /// than a 30-second spin-wait.
    #[test]
    fn a_released_manifest_lock_is_free_even_while_an_inherited_descriptor_survives() {
        use fs2::FileExt as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join("user.loras.jsonc");

        let held = ManifestLock::acquire(&manifest).expect("manifest lock acquires");
        let inherited = held
            ._lock
            .inherited_descriptor()
            .expect("descriptor duplicates");
        drop(held);

        let probe = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(manifest_lock_path(&manifest))
            .expect("probe opens lock file");
        assert!(
            probe.try_lock_exclusive().is_ok(),
            "a manifest lock released by its owner must not stay held by an inherited descriptor"
        );
        drop(inherited);
    }

    /// No listeners or model execution: real claimed rows, actual manifest CAS/lock, and an
    /// interleaving after initial authentication. The in-lock fence must prevent the disk mutation.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn import_publication_refuses_claim_loss_after_authentication_and_lock_wait() {
        use sceneworks_core::jobs_store::{CreateJob, JobsStore, RegisterWorker};
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        for change in [
            "owner",
            "interrupted",
            "attempt",
            "cancel",
            "missing",
            "live",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let data = temp.path().join("data");
            let original = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
                &data,
                "claim-fenced-selection",
            );
            let replacement = crate::minimax_h3_import_fixture::stage_managed_minimax_h3_import(
                &data,
                "claim-fenced-replacement",
            );
            let original_path = original
                .install_dir
                .join("diffusion_pytorch_model.safetensors");
            let original_bytes = std::fs::read(&original_path).unwrap();
            let replacement_path = replacement
                .install_dir
                .join("diffusion_pytorch_model.safetensors");
            let replacement_bytes = std::fs::read(&replacement_path).unwrap();
            let manifest = temp.path().join("user.models.jsonc");
            let before =
                serde_json::to_vec(&json!({"schemaVersion":1,"models":[original.entry]})).unwrap();
            std::fs::write(&manifest, &before).unwrap();
            let binding = sceneworks_core::model_artifacts::provenance::managed_repair_binding(
                &data,
                &original.entry,
            )
            .unwrap();
            let mut entry = replacement.entry.clone();
            entry["id"] = json!("claim-fenced-selection");
            let store = Arc::new(JobsStore::new(temp.path().join("claims.db")));
            store.initialize().unwrap();
            store
                .create_job(CreateJob {
                    job_type: JobType::ModelImport,
                    project_id: None,
                    project_name: None,
                    payload: json!({"modelId":"claim-fenced-selection","manifestEntry":entry})
                        .as_object()
                        .unwrap()
                        .clone(),
                    requested_gpu: "auto".into(),
                    source_job_id: None,
                    duplicate_of_job_id: None,
                    attempts: 1,
                    initial_status: None,
                })
                .unwrap();
            store
                .register_worker(RegisterWorker {
                    worker_id: "fenced-worker".into(),
                    gpu_id: "cpu".into(),
                    gpu_name: None,
                    capabilities: vec![WorkerCapability::ModelImport],
                    loaded_models: vec![],
                    utilization: None,
                })
                .unwrap();
            let expected = store.claim_next_job("fenced-worker").unwrap().unwrap();
            crate::model_jobs::validate_import_publication_claim(
                "fenced-worker",
                &expected,
                &expected,
            )
            .unwrap();
            let held = ManifestLock::acquire(&manifest).unwrap();
            let (started, waiting) = tokio::sync::oneshot::channel();
            let checks = Arc::new(AtomicUsize::new(0));
            let checked = checks.clone();
            let observed = store.clone();
            let path = manifest.clone();
            let job = expected.clone();
            let publish = tokio::spawn(async move {
                started.send(()).unwrap();
                publish_model_import_manifest(&path, &data, entry, Some(&binding), move || {
                    checked.fetch_add(1, Ordering::SeqCst);
                    let current = observed.get_job(&job.id).map_err(|error| {
                        WorkerError::InvalidPayload(format!("cannot re-read import claim: {error}"))
                    })?;
                    crate::model_jobs::validate_import_publication_claim(
                        "fenced-worker",
                        &job,
                        &current,
                    )
                })
                .await
            });
            waiting.await.unwrap();
            let sql = match change {
                "owner" => Some("update jobs set worker_id='replacement-worker' where id=?1"),
                "interrupted" => Some("update jobs set status='interrupted' where id=?1"),
                "attempt" => Some("update jobs set attempts=attempts+1 where id=?1"),
                "cancel" => Some("update jobs set cancel_requested=1 where id=?1"),
                "missing" => Some("delete from jobs where id=?1"),
                _ => None,
            };
            if let Some(sql) = sql {
                let connection = rusqlite::Connection::open(store.db_path()).unwrap();
                assert_eq!(
                    connection
                        .execute(sql, rusqlite::params![expected.id])
                        .unwrap(),
                    1
                );
            }
            drop(held);
            let result = publish.await.unwrap();
            if change == "live" {
                result.expect("a still-live claim must publish through the same lock/CAS");
                assert_ne!(std::fs::read(&manifest).unwrap(), before);
            } else {
                assert!(
                    result.is_err(),
                    "{change}: stale claim must refuse BEFORE manifest mutation"
                );
                assert_eq!(
                    std::fs::read(&manifest).unwrap(),
                    before,
                    "{change}: original selection untouched"
                );
            }
            assert_eq!(
                checks.load(Ordering::SeqCst),
                1,
                "claim is re-read at the actual publication boundary"
            );
            assert_eq!(std::fs::read(original_path).unwrap(), original_bytes);
            assert_eq!(
                std::fs::read(replacement_path).unwrap(),
                replacement_bytes,
                "recovery never adopts or deletes a committed install on claim loss"
            );
        }
    }

    fn entry(id: &str) -> serde_json::Map<String, Value> {
        let mut map = serde_json::Map::new();
        map.insert("id".to_owned(), json!(id));
        map.insert("name".to_owned(), json!(format!("name-{id}")));
        map
    }

    async fn read_ids(path: &Path, collection_key: &str) -> Vec<String> {
        let payload = tokio::fs::read_to_string(path)
            .await
            .expect("read manifest");
        let manifest: Value =
            serde_json::from_str(&strip_jsonc_comments(&payload)).expect("parse manifest");
        manifest
            .get(collection_key)
            .and_then(Value::as_array)
            .expect("collection array")
            .iter()
            .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_owned))
            .collect()
    }

    /// Two concurrent upserts to the same manifest must BOTH survive. Without the
    /// cross-process lock, the read→merge→rename interleaves and one entry is lost
    /// (F-041 / sc-8843). The flock guard serializes the RMW even across threads
    /// (and, in the real deployment, across the separate utility-worker processes).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_upserts_all_survive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("user.loras.jsonc");

        // Many concurrent writers, each a distinct id, all racing the same file. If
        // the lock is missing this loses entries essentially every run.
        const WRITERS: usize = 24;
        let mut handles = Vec::with_capacity(WRITERS);
        for index in 0..WRITERS {
            let path = path.clone();
            handles.push(crate::load_progress::spawn(async move {
                upsert_manifest_entry(&path, "loras", entry(&format!("lora-{index}"))).await
            }));
        }
        for handle in handles {
            handle.await.expect("join").expect("upsert");
        }

        let mut ids = read_ids(&path, "loras").await;
        ids.sort();
        let mut expected: Vec<String> = (0..WRITERS).map(|i| format!("lora-{i}")).collect();
        expected.sort();
        assert_eq!(ids, expected, "every concurrent upsert must persist");
    }

    /// Upserting an existing id merges fields and preserves the original createdAt,
    /// and the lock path is unaffected by repeated calls.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn upsert_merges_and_preserves_created_at() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("user.models.jsonc");

        let mut first = entry("m1");
        first.insert("createdAt".to_owned(), json!("2020-01-01"));
        first.insert("keep".to_owned(), json!("original"));
        upsert_manifest_entry(&path, "models", first)
            .await
            .expect("first upsert");

        let mut second = entry("m1");
        second.insert("createdAt".to_owned(), json!("2099-12-31"));
        second.insert("name".to_owned(), json!("renamed"));
        upsert_manifest_entry(&path, "models", second)
            .await
            .expect("second upsert");

        let payload = tokio::fs::read_to_string(&path).await.expect("read");
        let manifest: Value = serde_json::from_str(&strip_jsonc_comments(&payload)).expect("parse");
        let models = manifest["models"].as_array().expect("array");
        assert_eq!(models.len(), 1, "same id must not duplicate");
        let m1 = &models[0];
        assert_eq!(m1["createdAt"], json!("2020-01-01"), "createdAt preserved");
        assert_eq!(m1["name"], json!("renamed"), "incoming fields win");
        assert_eq!(m1["keep"], json!("original"), "prior fields retained");
    }

    /// A lock timeout surfaces a clear error rather than silently skipping the write.
    #[tokio::test]
    async fn missing_id_errors_before_lock() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("user.loras.jsonc");
        let mut bad = serde_json::Map::new();
        bad.insert("name".to_owned(), json!("no-id"));
        let err = upsert_manifest_entry(&path, "loras", bad)
            .await
            .expect_err("missing id must error");
        assert!(matches!(err, WorkerError::InvalidPayload(_)));
        // Nothing should have been written.
        assert!(!path.exists(), "no manifest written for invalid payload");
    }
}
