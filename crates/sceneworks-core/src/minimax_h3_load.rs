//! Imported H3's prepared load closure. Reuses gen-core seals and the worker's existing warm key;
//! it does not attest tensor use or measure allocations.

use gen_core::{LoadSpec, PinnedWeightsFile, WeightsSource};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Marker carried across the stringly core boundary when a native checked pin observes the
/// cooperative cancellation. The worker maps it back to its typed Canceled error, so a cancel that
/// trips inside the native hash loop surfaces as the job's own cancellation, never as drift.
pub const H3_PREPARATION_CANCELED_MARKER: &str =
    "[h3-preparation-canceled] checkpoint preparation canceled";

/// Forward the native artifact-seal observer's typed work record into the ACTIVE job span
/// (`h3_import_preparation` / `h3_import_component_acquisition`). Every value is the native
/// layer's own observation — acquisition kind, state, real hashed bytes and declared size — never
/// re-derived or estimated; memo acquisitions report zero hashed bytes by contract.
pub fn observe_artifact_seal_progress(progress: &gen_core::runtime::ArtifactSealProgress) {
    tracing::info!(
        event = "artifact_seal_hash_progress",
        path = %progress.loader_path.display(),
        acquisition = ?progress.acquisition,
        state = ?progress.state,
        hashed_bytes = progress.hashed_bytes,
        source_bytes = progress.source_bytes,
        "Actual native artifact-seal acquisition work"
    );
}

/// Authority retained from plan resolution, including exact normalized derivative hashes when
/// applicable. Reacquiring a seal later must never authorize replacement of this checkpoint.
#[derive(Clone, Debug)]
pub struct H3SourceAuthority {
    seals: Vec<PinnedWeightsFile>,
    directory: PathBuf,
    planned: BTreeSet<String>,
    component_staging: std::sync::Arc<
        parking_lot::Mutex<Vec<crate::model_artifacts::provenance::H3ComponentStagingReceipt>>,
    >,
    _leases: Option<std::sync::Arc<H3SourceLeases>>,
}

#[derive(Debug)]
struct H3SourceLeases {
    _source: crate::file_lock::FileLock,
    _derivative: Option<crate::model_artifacts::resolved_cache::ResolvedCacheLease>,
}

impl H3SourceAuthority {
    pub fn new(
        seals: Vec<PinnedWeightsFile>,
        directory: PathBuf,
        planned: BTreeSet<String>,
    ) -> Result<Self, String> {
        let authority = Self {
            seals,
            directory,
            planned,
            component_staging: Default::default(),
            _leases: None,
        };
        authority.validate()?;
        Ok(authority)
    }

    /// Preparation can outlive a canceled async owner (`spawn_blocking` is not abortable once
    /// running). Every authority clone therefore retains both leases until the last reader exits.
    pub fn with_leases(
        mut self,
        source: crate::file_lock::FileLock,
        derivative: Option<crate::model_artifacts::resolved_cache::ResolvedCacheLease>,
    ) -> Self {
        self._leases = Some(std::sync::Arc::new(H3SourceLeases {
            _source: source,
            _derivative: derivative,
        }));
        self
    }

    /// Receipt identity from the derivative lease already retained by this authority. Returning
    /// an owned key does not move or release either lease; blocking preparation/native loading
    /// keeps both alive through the existing Arc clones, including after async cancellation.
    pub fn derivative_cache_key(&self) -> Option<String> {
        self._leases
            .as_ref()?
            ._derivative
            .as_ref()
            .map(|derivative| {
                derivative
                    .artifact()
                    .cache_key()
                    .expect("leased artifact identity is validated")
            })
    }

    /// The existing staging operation and its provenance are inseparable. Clones retain these
    /// private receipts alongside source/derivative leases through preparation, cache hits and cancel.
    pub fn stage_component(
        &self,
        data_dir: &Path,
        assembly: &Value,
        link: &Path,
        target: &Path,
    ) -> Result<(), String> {
        let receipts = crate::model_artifacts::provenance::stage_h3_component_with_receipts(
            data_dir, assembly, link, target,
        )?;
        self.component_staging.lock().extend(receipts);
        Ok(())
    }

    pub fn component_staging(
        &self,
    ) -> Vec<crate::model_artifacts::provenance::H3ComponentStagingReceipt> {
        self.component_staging.lock().clone()
    }

    pub fn validate(&self) -> Result<(), String> {
        for receipt in self.component_staging.lock().iter() {
            receipt.validate()?;
        }
        for seal in &self.seals {
            seal.verify_unchanged().map_err(|e| e.to_string())?;
        }
        for path in members(&self.directory)? {
            if path
                .extension()
                .is_some_and(|e| e == "safetensors" || e == "gguf")
                && !self
                    .planned
                    .contains(&path.file_name().unwrap().to_string_lossy().into_owned())
            {
                return Err(format!("unplanned H3 backbone file {}", path.display()));
            }
        }
        Ok(())
    }
}

/// Metadata-only validation after acquisition: no per-step content hashing. Native H3 discovers
/// the reference partition as a sibling, not as an accepted named component. Its seals remain in
/// this receipt and the worker's supplemental warm identity, never in an unsupported provider slot.
#[derive(Clone, Debug)]
pub struct PreparedH3Load {
    directories: BTreeMap<PathBuf, Option<Vec<PathBuf>>>,
    files: Vec<PinnedWeightsFile>,
    spec_files: Vec<PinnedWeightsFile>,
    authority: Option<H3SourceAuthority>,
}

impl PreparedH3Load {
    pub fn prepare(spec: &mut LoadSpec) -> Result<Self, String> {
        Self::prepare_authorized(spec, None)
    }

    pub fn prepare_authorized(
        spec: &mut LoadSpec,
        authority: Option<H3SourceAuthority>,
    ) -> Result<Self, String> {
        Self::prepare_authorized_with_check(spec, authority, None, &|| Ok(()))
    }

    /// `companion` names a frozen component directory the import committed but the direct DiT
    /// path does not expose as its sibling (the Candle FL2VA direct-directory route). Its files
    /// are pinned into the prepared receipt exactly like a staged sibling's, so the frozen
    /// component closure verifies — no staging, no substitution, and no cross-volume links.
    pub fn prepare_authorized_with_check(
        spec: &mut LoadSpec,
        authority: Option<H3SourceAuthority>,
        companion: Option<PathBuf>,
        check: &dyn Fn() -> std::io::Result<()>,
    ) -> Result<Self, String> {
        check().map_err(|error| error.to_string())?;
        if let Some(authority) = &authority {
            authority.validate()?;
        }
        let root = directory(&spec.weights)?;
        let dit = spec
            .components
            .get("transformer")
            .map(directory)
            .transpose()?
            .unwrap_or_else(|| root.join("transformer"));
        let encoder = spec
            .components
            .get("text_encoder")
            .map(directory)
            .transpose()?
            .unwrap_or_else(|| root.join("text_encoder"));
        let mut dirs = vec![
            dit.with_file_name("transformer_ref"),
            dit,
            encoder,
            root.join("vae"),
            root.join("audio_vae"),
            root.join("tokenizer"),
            root.join("FL2VA/audio_vae"),
        ];
        if let Some(companion) = companion {
            dirs.push(companion);
        }
        // Include additional explicitly configured native components (for example Veda), but do
        // not walk the whole shared snapshot or any unrelated cache/model tree.
        dirs.extend(spec.components.values().filter_map(|source| match source {
            WeightsSource::Dir(path) => Some(path.clone()),
            _ => None,
        }));
        let mut directories = BTreeMap::new();
        for dir in dirs {
            check().map_err(|error| error.to_string())?;
            let dir = std::path::absolute(dir).map_err(|e| e.to_string())?;
            let snapshot = if dir.is_dir() {
                Some(members(&dir)?)
            } else {
                None
            };
            directories.insert(dir, snapshot);
        }
        let mut paths: BTreeSet<_> = directories.values().flatten().flatten().cloned().collect();
        for path in spec.file_source_paths() {
            paths.insert(std::path::absolute(path).map_err(|e| e.to_string())?);
        }
        let files = paths
            .into_iter()
            .map(|path| {
                // Native-main late consumer delta: the typed checked pin carries the SW-side
                // cooperative cancellation and reports the native layer's OWN actual acquisition
                // work (acquisition kind, state, real hashed/source bytes) into the active job
                // span. An Interrupted check is the typed native cancellation; other io errors
                // keep their typed cause.
                let pin = PinnedWeightsFile::pin_with_progress(
                    path,
                    move || match check() {
                        Ok(()) => Ok(()),
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                            Err(gen_core::Error::Canceled)
                        }
                        Err(error) => Err(gen_core::Error::Io(error)),
                    },
                    |progress| observe_artifact_seal_progress(progress),
                )
                .map_err(|error| match error {
                    gen_core::Error::Canceled => H3_PREPARATION_CANCELED_MARKER.to_owned(),
                    other => other.to_string(),
                })?;
                Ok::<_, String>(pin)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let configured_dirs = spec
            .directory_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        let configured_files = spec
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<BTreeSet<_>>>()
            .map_err(|e| e.to_string())?;
        let spec_files: Vec<_> = files
            .iter()
            .filter(|pin| {
                configured_files.contains(pin.loader_path())
                    || configured_dirs
                        .iter()
                        .any(|dir| pin.loader_path().starts_with(dir))
            })
            .cloned()
            .collect();
        let receipt = Self {
            directories,
            files,
            spec_files,
            authority,
        };
        receipt.validate_sources()?;
        spec.prepare_with_file_pins(receipt.spec_files.iter().cloned())
            .map_err(|e| e.to_string())?;
        receipt.validate(spec)?;
        check().map_err(|error| error.to_string())?;
        Ok(receipt)
    }

    pub fn validate(&self, spec: &LoadSpec) -> Result<(), String> {
        self.validate_sources()?;
        spec.validate_prepared_file_pins()
            .map_err(|e| e.to_string())?;
        if spec
            .prepared_file_pins()
            .iter()
            .map(|(_, pin)| pin)
            .ne(self.spec_files.iter())
        {
            return Err("H3 prepared load no longer names the acquired file closure".into());
        }
        Ok(())
    }

    fn validate_sources(&self) -> Result<(), String> {
        if let Some(authority) = &self.authority {
            authority.validate()?;
        }
        for (dir, expected) in &self.directories {
            let current = if dir.is_dir() {
                Some(members(dir)?)
            } else {
                None
            };
            if current != *expected {
                return Err(format!(
                    "H3 component membership changed before execution: {}",
                    dir.display()
                ));
            }
        }
        for pin in &self.files {
            pin.verify_unchanged().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn files(&self) -> &[PinnedWeightsFile] {
        &self.files
    }

    pub fn component_staging(
        &self,
    ) -> Vec<crate::model_artifacts::provenance::H3ComponentStagingReceipt> {
        self.authority
            .as_ref()
            .map(H3SourceAuthority::component_staging)
            .unwrap_or_default()
    }

    /// Prepared file closure, not `physicalReadHashes`: content identity is not proof of tensor use.
    pub fn prepared_sources_json(&self) -> Value {
        Value::Array(self.files.iter().map(|pin| json!({
            "path": pin.loader_path(),
            "sha256": pin.content_sha256().iter().map(|b| format!("{b:02x}")).collect::<String>(),
        })).collect())
    }
}

fn directory(source: &WeightsSource) -> Result<PathBuf, String> {
    match source {
        WeightsSource::Dir(path) => Ok(path.clone()),
        WeightsSource::File(_) => Err("H3 component must be a resolved directory".into()),
    }
}

fn members(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext,
                    "safetensors" | "gguf" | "json" | "yaml" | "yml" | "model" | "txt" | "tiktoken"
                )
            })
        {
            if !path.is_file() {
                return Err(format!(
                    "H3 component member is not a file: {}",
                    path.display()
                ));
            }
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Terminal review Issue 2: a direct-path import whose frozen companion is NOT the DiT
    /// directory's sibling must still ACQUIRE that companion into the prepared receipt — the
    /// frozen component closure verifies against `receipt.files()`, and a direct-directory load
    /// has no sibling rule to rely on. The companion is pinned at its actually resolved location;
    /// no staging, no copies, and nothing is installed on the provider spec that the engine did
    /// not ask for.
    #[test]
    fn an_explicit_companion_is_acquired_into_the_prepared_receipt_for_direct_paths() {
        let temp = tempfile::tempdir().unwrap();
        let put = |dir: &Path, name: &str| {
            std::fs::create_dir_all(dir).unwrap();
            let path = dir.join(name);
            std::fs::write(&path, [name.as_bytes(), b"-fixture-bytes"].concat()).unwrap();
            path
        };
        let root = temp.path().join("root");
        let dit = put(&root.join("transformer"), "dit.safetensors");
        put(&root.join("text_encoder"), "te.safetensors");
        // The companion lives OUTSIDE the weights root — exactly the resolved library snapshot a
        // Candle FL2VA import commits, which nothing about the direct DiT path exposes.
        let companion_dir = temp.path().join("library/q4/transformer_ref");
        let companion_file = put(&companion_dir, "companion.safetensors");

        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.components.insert(
            "transformer".to_owned(),
            WeightsSource::Dir(dit.parent().unwrap().to_path_buf()),
        );

        // Without the explicit companion, the sibling rule finds nothing outside the root: the
        // receipt must not invent the frozen component.
        let mut plain = spec.clone();
        let without =
            PreparedH3Load::prepare_authorized_with_check(&mut plain, None, None, &|| Ok(()))
                .unwrap();
        assert!(
            without
                .files()
                .iter()
                .all(|pin| pin.loader_path() != &companion_file),
            "no sibling heuristic may invent a companion the caller did not acquire"
        );

        // With it, the companion's real files are pinned into the SAME receipt and the provider
        // spec stays untouched (acquisition-only, exactly as the frozen closure expects).
        let mut direct = spec.clone();
        let with = PreparedH3Load::prepare_authorized_with_check(
            &mut direct,
            None,
            Some(companion_dir.clone()),
            &|| Ok(()),
        )
        .unwrap();
        assert!(
            with.files()
                .iter()
                .any(|pin| pin.loader_path() == &companion_file),
            "the explicit companion is acquired into the prepared receipt"
        );
        assert!(
            direct
                .prepared_file_pins()
                .iter()
                .all(|(_, pin)| pin.loader_path() != &companion_file),
            "companion acquisition installs nothing on the provider spec"
        );
        // The check above would pass vacuously against an empty pin set, so the spec's OWN
        // configured files must still be pinned: preparation installs the spec closure and
        // nothing beyond it.
        assert!(
            !direct.prepared_file_pins().is_empty(),
            "the spec's own configured files are still pinned"
        );
        with.validate(&direct).unwrap();

        // And the companion seal is content-bound: mutating the byte under the same path yields
        // a DIFFERENT acquired digest on re-preparation, so a frozen closure recorded against the
        // original content can never verify against the mutated file.
        let acquired_digest = with
            .files()
            .iter()
            .find(|pin| pin.loader_path() == &companion_file)
            .expect("the companion was acquired")
            .content_sha256()
            .to_vec();
        std::fs::write(&companion_file, b"mutated-companion-bytes").unwrap();
        let mut mutated = spec;
        let reprepared = PreparedH3Load::prepare_authorized_with_check(
            &mut mutated,
            None,
            Some(companion_dir),
            &|| Ok(()),
        )
        .unwrap();
        let mutated_digest = reprepared
            .files()
            .iter()
            .find(|pin| pin.loader_path() == &companion_file)
            .expect("the companion is re-acquired")
            .content_sha256()
            .to_vec();
        assert_ne!(
            acquired_digest, mutated_digest,
            "the acquired companion seal follows the live file content"
        );
    }
}
