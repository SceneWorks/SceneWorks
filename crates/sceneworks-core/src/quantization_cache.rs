//! Saved weight quantizations use the existing resolved-artifact store and its leases,
//! atomic publication, source lifecycle and retention. Completed weights load without revalidation,
//! and publication records each packed file by size only: it is never read back or hashed.
use crate::model_artifacts::resolved_cache::{
    DerivedArtifactPlan, ReservationOutcome, ResolvedCacheLease, ResolvedCacheStore,
};
use crate::model_artifacts::{
    ArtifactFile, ArtifactIdentity, ArtifactMemberRole, ArtifactSourceLibrary,
    ModelArtifactResolver, ResolvedBundleMember,
};
pub use gen_core::{QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The same provenance must be used when saving and deleting an import's derivatives.
pub fn source_identity(entry: &crate::contracts::JsonObject) -> Option<&str> {
    crate::jobs_store::checkpoint_plan_checkpoint_id(entry)
        .or_else(|| entry.get("id").and_then(serde_json::Value::as_str))
}

pub struct SavedQuantizations {
    data_dir: PathBuf,
    store: std::sync::Mutex<Option<ResolvedCacheStore>>,
    source: PathBuf,
    owner: String,
    checkpoint: String,
    hits: AtomicU64,
    writes: AtomicU64,
}
struct SavedWeight {
    path: PathBuf,
    _lease: ResolvedCacheLease,
}
impl QuantizationArtifact for SavedWeight {
    fn path(&self) -> &Path {
        &self.path
    }
}
fn error(error: impl std::fmt::Display) -> gen_core::Error {
    gen_core::Error::Msg(format!("Saved quantization: {error}"))
}
fn check_deadline(deadline: Instant) -> std::io::Result<()> {
    if Instant::now() >= deadline {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "saved quantization acquisition deadline exceeded; retry the generation",
        ));
    }
    Ok(())
}
impl SavedQuantizations {
    pub fn new(data_dir: &Path, source: PathBuf, owner: String, checkpoint: String) -> Self {
        Self {
            data_dir: data_dir.to_owned(),
            store: std::sync::Mutex::new(None),
            source,
            owner,
            checkpoint,
            hits: AtomicU64::new(0),
            writes: AtomicU64::new(0),
        }
    }
    pub fn counts(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.writes.load(Ordering::Relaxed),
        )
    }
}
fn artifact_plan(key: &QuantizationArtifactKey) -> gen_core::Result<(DerivedArtifactPlan, String)> {
    let extension = key.extension();
    if !matches!(extension, "ggml" | "safetensors") {
        return Err(error("unsupported packed artifact extension"));
    }
    let filename = format!("projection.{extension}");
    // The store uses 40-hex revisions; retain the rest in the variant so all 256 key bits
    // still participate in the immutable artifact identity.
    let variant = format!("{extension}-{}", &key.digest()[40..]);
    let identity = ArtifactIdentity::pinned(
        "sceneworks-weight-quantization/projection",
        &key.digest()[..40],
        variant,
    )
    .map_err(error)?;
    let member = ResolvedBundleMember {
        role: ArtifactMemberRole::Primary,
        component_id: Some("saved-quantization".to_owned()),
        source: identity.clone(),
        tier: None,
        source_subpath: PathBuf::new(),
        destination: PathBuf::new(),
        files: vec![ArtifactFile::new(&filename).map_err(error)?],
    };
    let plan = DerivedArtifactPlan::new(identity, vec![member]).map_err(error)?;
    Ok((plan, filename))
}

impl SavedQuantizations {
    fn materialize_with_timeout(
        &self,
        key: &QuantizationArtifactKey,
        produce: &mut dyn FnMut(&Path) -> gen_core::Result<()>,
        wait_timeout: Duration,
    ) -> gen_core::Result<Box<dyn QuantizationArtifact>> {
        let store = {
            let mut slot = self
                .store
                .lock()
                .map_err(|_| error("artifact store lock poisoned"))?;
            if slot.is_none() {
                *slot = Some(ResolvedCacheStore::open(&self.data_dir).map_err(error)?);
            }
            slot.as_ref().unwrap().clone()
        };
        let source_root = if self.source.is_file() {
            self.source
                .parent()
                .ok_or_else(|| error("checkpoint has no source directory"))?
        } else {
            &self.source
        };
        let resolver =
            ModelArtifactResolver::new(ArtifactSourceLibrary::new(source_root).map_err(error)?);
        let (plan, filename) = artifact_plan(key)?;
        let cache_key = plan.cache_key().map_err(error)?;
        let deadline = Instant::now() + wait_timeout;
        loop {
            check_deadline(deadline).map_err(error)?;
            if let Some(lease) = store
                .acquire_complete_with_check(&cache_key, &resolver, &self.owner, &|| {
                    check_deadline(deadline)
                })
                .map_err(error)?
            {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Box::new(SavedWeight {
                    path: store
                        .bundle_path(&cache_key)
                        .map_err(error)?
                        .join(&filename),
                    _lease: lease,
                }));
            }
            match store
                .reserve_derived(&plan, source_root, &self.checkpoint, &self.owner)
                .map_err(error)?
            {
                ReservationOutcome::AlreadyComplete(_) => std::thread::yield_now(),
                ReservationOutcome::Contended => {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                ReservationOutcome::Acquired(reservation) => {
                    let reservation = *reservation;
                    reservation.prepare_for_materialization().map_err(error)?;
                    let path = reservation.staging_path().join(&filename);
                    if let Err(failure) = produce(&path) {
                        reservation.discard().map_err(error)?;
                        return Err(failure);
                    }
                    let file = std::fs::OpenOptions::new().write(true).open(&path)?;
                    crate::fsutil::sync_file(&file)?;
                    drop(file);
                    // Packed weights are a local, in-process function of an already-trusted
                    // checkpoint: publish by size, never re-read and hash them.
                    reservation
                        .publish_produced_size_only(&plan)
                        .map_err(error)?;
                    self.writes.fetch_add(1, Ordering::Relaxed);
                    // Packing itself is not an acquisition wait; give the new lease its own bound.
                    let deadline = Instant::now() + wait_timeout;
                    let lease = store
                        .acquire_complete_with_check(&cache_key, &resolver, &self.owner, &|| {
                            check_deadline(deadline)
                        })
                        .map_err(error)?
                        .ok_or_else(|| error("saved weights disappeared before loading"))?;
                    return Ok(Box::new(SavedWeight {
                        path: store.bundle_path(&cache_key).map_err(error)?.join(filename),
                        _lease: lease,
                    }));
                }
            }
        }
    }
}

impl QuantizationArtifactCache for SavedQuantizations {
    fn materialize(
        &self,
        key: &QuantizationArtifactKey,
        produce: &mut dyn FnMut(&Path) -> gen_core::Result<()>,
    ) -> gen_core::Result<Box<dyn QuantizationArtifact>> {
        self.materialize_with_timeout(key, produce, Duration::from_secs(60))
    }

    fn invalidate(&self, key: &QuantizationArtifactKey) -> gen_core::Result<bool> {
        let store = self
            .store
            .lock()
            .map_err(|_| error("artifact store lock poisoned"))?
            .clone();
        let Some(store) = store else {
            return Ok(false);
        };
        let (plan, _) = artifact_plan(key)?;
        let cache_key = plan.cache_key().map_err(error)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(error)?
            .as_secs();
        store.remove_entry(&cache_key, now).map_err(error)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_active_producer_cannot_hold_a_cache_reader_past_its_deadline() {
        let data = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let cache = || {
            SavedQuantizations::new(
                data.path(),
                source.path().to_owned(),
                "test".into(),
                "test/checkpoint".into(),
            )
        };
        let writer = cache();
        let reader = cache();
        *reader.store.lock().unwrap() = Some(ResolvedCacheStore::open(data.path()).unwrap());
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let key = || QuantizationArtifactKey::new("q4-deadline-v1", &[4], b"source", "ggml");
        let writer_key = key();
        let producer = std::thread::spawn(move || {
            let artifact = writer
                .materialize(&writer_key, &mut |path| {
                    ready_tx.send(()).unwrap();
                    let _ = release_rx.recv_timeout(Duration::from_secs(3));
                    std::fs::write(path, b"packed")?;
                    Ok(())
                })
                .unwrap();
            assert_eq!(std::fs::read(artifact.path()).unwrap(), b"packed");
        });
        ready_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let mut repacked = false;
        let started = Instant::now();
        let result = reader.materialize_with_timeout(
            &key(),
            &mut |path| {
                repacked = true;
                std::fs::write(path, b"unexpected")?;
                Ok(())
            },
            Duration::from_millis(100),
        );
        let elapsed = started.elapsed();
        let _ = release_tx.send(());
        producer.join().unwrap();
        assert!(
            elapsed < Duration::from_secs(1),
            "reader waited {elapsed:?}"
        );
        assert!(!repacked);
        assert!(result
            .err()
            .expect("reader must time out")
            .to_string()
            .contains("deadline"));
        reader
            .materialize(&key(), &mut |_| {
                panic!("retry must reuse the producer's completed output")
            })
            .unwrap();
    }
    #[test]
    fn saved_weights_survive_a_fresh_store_and_changed_inputs_get_new_artifacts() {
        let data = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let source_path = source.path().join("checkpoint.safetensors");
        std::fs::write(&source_path, b"source weights").unwrap();
        let store = || {
            SavedQuantizations::new(
                data.path(),
                source_path.clone(),
                "community".into(),
                "managed/community".into(),
            )
        };
        let key = QuantizationArtifactKey::new("candle-q8-test-v1", &[2, 2], b"original", "ggml");
        let first = store();
        let artifact = first
            .materialize(&key, &mut |path| {
                std::fs::write(path, b"packed")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(std::fs::read(artifact.path()).unwrap(), b"packed");
        drop(artifact);
        drop(first);
        let reopened = store();
        let (artifact, hashes) = crate::model_artifacts::observe_content_hashes(|| {
            reopened.materialize(&key, &mut |_| {
                panic!("a saved quantization must never run its quantizer again")
            })
        });
        let artifact = artifact.unwrap();
        assert_eq!(
            hashes, 0,
            "fresh-process cache hits must not hash the saved weights"
        );
        assert_eq!(std::fs::read(artifact.path()).unwrap(), b"packed");
        assert_eq!(reopened.counts(), (1, 0));
        let changed =
            QuantizationArtifactKey::new("candle-q8-test-v1", &[2, 2], b"changed", "ggml");
        let changed = reopened
            .materialize(&changed, &mut |path| {
                std::fs::write(path, b"new-packed")?;
                Ok(())
            })
            .unwrap();
        assert_ne!(changed.path(), artifact.path());
    }
    #[test]
    fn packing_a_new_weight_never_reads_back_or_hashes_what_it_wrote() {
        let data = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let cache = SavedQuantizations::new(
            data.path(),
            source.path().to_owned(),
            "model".into(),
            "model/checkpoint".into(),
        );
        let key = QuantizationArtifactKey::new("size-only-v1", &[2, 2], b"weights", "ggml");
        let (artifact, hashes) = crate::model_artifacts::observe_content_hashes(|| {
            cache.materialize(&key, &mut |path| {
                std::fs::write(path, b"packed")?;
                Ok(())
            })
        });
        let artifact = artifact.unwrap();
        assert_eq!(
            hashes, 0,
            "a cache miss publishes packed weights without hashing them"
        );
        assert_eq!(std::fs::read(artifact.path()).unwrap(), b"packed");
        assert_eq!(cache.counts(), (0, 1));

        let (plan, filename) = artifact_plan(&key).unwrap();
        let metadata = ResolvedCacheStore::open(data.path())
            .unwrap()
            .lookup_complete(&plan.cache_key().unwrap())
            .unwrap()
            .expect("the packed weight is published complete");
        let files: Vec<_> = metadata
            .artifact
            .closure
            .members
            .iter()
            .flat_map(|member| member.files.iter())
            .collect();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path, PathBuf::from(&filename));
        assert_eq!(files[0].size_bytes, Some(6), "size still feeds retention");
        assert_eq!(files[0].sha256, None, "no digest is computed or recorded");
    }
    #[test]
    fn unreadable_runtime_weights_can_be_invalidated_after_the_loader_releases_its_lease() {
        let data = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let cache = SavedQuantizations::new(
            data.path(),
            source.path().to_owned(),
            "model".into(),
            "model/checkpoint".into(),
        );
        let key = QuantizationArtifactKey::new("runtime-repair-v1", &[4], b"source", "ggml");
        let artifact = cache
            .materialize(&key, &mut |path| {
                std::fs::write(path, b"packed")?;
                Ok(())
            })
            .unwrap();
        assert!(
            cache.invalidate(&key).is_err(),
            "active runtime leases must remain protected"
        );
        let path = artifact.path().to_owned();
        drop(artifact);
        std::fs::remove_file(&path).unwrap();
        let (trusted, hashes) = crate::model_artifacts::observe_content_hashes(|| {
            cache.materialize(&key, &mut |_| {
                panic!("ready cache must reach the runtime reader directly")
            })
        });
        let trusted = trusted.unwrap();
        assert_eq!(hashes, 0);
        assert_eq!(trusted.path(), path);
        drop(trusted);
        assert!(cache.invalidate(&key).unwrap());
        let repaired = cache
            .materialize(&key, &mut |path| {
                std::fs::write(path, b"repaired")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(std::fs::read(repaired.path()).unwrap(), b"repaired");
        assert_eq!(cache.counts(), (1, 2));
    }

    #[test]
    fn failed_quantization_never_publishes_a_partial_copy() {
        let data = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let store = SavedQuantizations::new(
            data.path(),
            source.path().to_owned(),
            "community".into(),
            "managed/community".into(),
        );
        let key = QuantizationArtifactKey::new("q4-test-v1", &[4], b"source", "ggml");
        assert!(store
            .materialize(&key, &mut |path| {
                std::fs::write(path, b"partial")?;
                Err(gen_core::Error::Canceled)
            })
            .is_err());
        let artifact = store
            .materialize(&key, &mut |path| {
                std::fs::write(path, b"complete")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(std::fs::read(artifact.path()).unwrap(), b"complete");
    }
}
