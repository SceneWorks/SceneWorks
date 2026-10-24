//! Imported H3's prepared load closure. Reuses gen-core seals and the worker's existing warm key;
//! it does not attest tensor use or measure allocations.

use gen_core::{LoadSpec, PinnedWeightsFile, WeightsSource};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Authority retained from plan resolution, including exact normalized derivative hashes when
/// applicable. Reacquiring a seal later must never authorize replacement of this checkpoint.
#[derive(Clone, Debug)]
pub struct H3SourceAuthority {
    seals: Vec<PinnedWeightsFile>,
    directory: PathBuf,
    planned: BTreeSet<String>,
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

    /// Receipt identity comes only from the derivative lease retained by this authority. Clones
    /// keep the same lease alive; source-only loads have no derivative receipt to claim.
    pub fn derivative_cache_key(&self) -> Option<String> {
        let derivative = self._leases.as_ref()?._derivative.as_ref()?;
        Some(
            derivative
                .artifact()
                .cache_key()
                .expect("leased artifact identity is validated"),
        )
    }

    pub fn validate(&self) -> Result<(), String> {
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
        // Include additional explicitly configured native components (for example Veda), but do
        // not walk the whole shared snapshot or any unrelated cache/model tree.
        dirs.extend(spec.components.values().filter_map(|source| match source {
            WeightsSource::Dir(path) => Some(path.clone()),
            _ => None,
        }));
        let mut directories = BTreeMap::new();
        for dir in dirs {
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
            .map(|path| PinnedWeightsFile::pin(path).map_err(|e| e.to_string()))
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
