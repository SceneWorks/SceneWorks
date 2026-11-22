//! Saved weight quantizations use the existing resolved-artifact store and its leases,
//! atomic publication, content checks, source lifecycle and retention.
use crate::model_artifacts::resolved_cache::{
    DerivedArtifactPlan, ReservationOutcome, ResolvedCacheLease, ResolvedCacheStore,
};
use crate::model_artifacts::{
    ArtifactFile, ArtifactIdentity, ArtifactMemberRole, ArtifactSourceLibrary,
    ModelArtifactResolver, ResolvedBundleMember,
};
use gen_core::{QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
impl QuantizationArtifactCache for SavedQuantizations {
    fn materialize(
        &self,
        key: &QuantizationArtifactKey,
        produce: &mut dyn FnMut(&Path) -> gen_core::Result<()>,
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
        let cache_key = plan.cache_key().map_err(error)?;
        let started = std::time::Instant::now();
        loop {
            if started.elapsed() > std::time::Duration::from_secs(60) {
                return Err(error(
                    "saved weights could not be acquired; retry the generation",
                ));
            }
            if let Some(lease) = store
                .acquire_complete(&cache_key, &resolver, &self.owner)
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
                    if started.elapsed() > std::time::Duration::from_secs(60) {
                        return Err(error(
                            "another worker is still saving these weights; retry the generation",
                        ));
                    }
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
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&path)?
                        .sync_all()?;
                    reservation.publish_produced(&plan).map_err(error)?;
                    self.writes.fetch_add(1, Ordering::Relaxed);
                    let lease = store
                        .acquire_complete(&cache_key, &resolver, &self.owner)
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

#[cfg(test)]
mod tests {
    use super::*;
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
        let artifact = reopened
            .materialize(&key, &mut |_| {
                panic!("a saved quantization must never run its quantizer again")
            })
            .unwrap();
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
