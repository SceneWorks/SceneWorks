//! Private advertisements for selected worker-owned auxiliary model sources.
//! Physical paths stay on the worker; the API receives only logical IDs and file hashes.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sceneworks_core::contracts::JsonObject;
use sceneworks_core::model_artifacts::portable_binding::{
    PortableArtifactFile, PortableSourceClaim, PortableSourceProvenance,
};
use sha2::{Digest, Sha256};

use crate::{Settings, WorkerError, WorkerResult};

const STATIC_SOURCE_IDS: &[&str] = &[
    "ltx_amoral_gemma_3_12b",
    "flux_ip_adapter",
    "flux_ip_encoder",
    "sdxl_ip_adapter",
    "sdxl_ip_encoder",
    "depth_anything_v2_small",
];

pub(crate) fn static_source_claims(settings: &Settings) -> Vec<PortableSourceClaim> {
    let empty = JsonObject::new();
    STATIC_SOURCE_IDS
        .iter()
        .filter_map(|source_id| {
            match capture_selected_source(source_id, settings, &empty) {
                Ok(Some((claim, _))) => Some(claim),
                Ok(None) => None,
                Err(error) => {
                    tracing::warn!(source_id, %error, "private auxiliary source could not be advertised");
                    None
                }
            }
        })
        .collect()
}

/// Capture the exact effective source that the loader resolver would select for this payload.
/// The companion path is worker-private and is never serialized into the claim.
pub(crate) fn capture_selected_source(
    source_id: &str,
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Option<(PortableSourceClaim, PathBuf)>> {
    let Some((selected, operator_selected)) =
        crate::portable_aux_sources::resolve_selected_aux_source(
            source_id,
            &settings.data_dir,
            payload,
        )
    else {
        return Ok(None);
    };
    let provenance = if !operator_selected {
        sceneworks_core::control_weights::shipped_auxiliary_source(source_id)
            .and_then(|source| {
                sceneworks_core::hf_home::model_source_library(&settings.data_dir)
                    .discover_snapshot(source.repo, Some(source.revision))
                    .ok()
                    .filter(|(_, root)| selected.starts_with(root))
                    .map(|_| PortableSourceProvenance::PinnedRepo {
                        repository: source.repo.to_owned(),
                        revision: source.revision.to_owned(),
                    })
            })
            .unwrap_or(PortableSourceProvenance::OperatorOverride)
    } else {
        PortableSourceProvenance::OperatorOverride
    };
    let files = capture_files(&selected)?;
    let claim = PortableSourceClaim::new(source_id, provenance, files)
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
    Ok(Some((claim, selected)))
}

fn capture_files(selected: &Path) -> WorkerResult<Vec<PortableArtifactFile>> {
    let mut found = Vec::new();
    if selected.is_file() {
        let name = selected.file_name().ok_or_else(|| {
            WorkerError::InvalidPayload("auxiliary source has no file name".to_owned())
        })?;
        capture_one(selected, PathBuf::from(name), &mut found)?;
    } else if selected.is_dir() {
        capture_tree(selected, selected, &mut found)?;
    } else {
        return Err(WorkerError::InvalidPayload(
            "auxiliary source is unavailable".to_owned(),
        ));
    }
    if found.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "auxiliary source has no regular files".to_owned(),
        ));
    }
    found.sort();
    Ok(found)
}

fn capture_tree(
    root: &Path,
    current: &Path,
    found: &mut Vec<PortableArtifactFile>,
) -> WorkerResult<()> {
    for entry in std::fs::read_dir(current).map_err(WorkerError::Io)? {
        let path = entry.map_err(WorkerError::Io)?.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(WorkerError::Io)?;
        if metadata.is_dir() {
            capture_tree(root, &path, found)?;
        } else if metadata.is_file() || metadata.file_type().is_symlink() && path.is_file() {
            let relative = path
                .strip_prefix(root)
                .expect("traversed path is inside root")
                .to_path_buf();
            capture_one(&path, relative, found)?;
        } else {
            return Err(WorkerError::InvalidPayload(
                "auxiliary source contains a non-file entry".to_owned(),
            ));
        }
    }
    Ok(())
}

fn capture_one(
    path: &Path,
    relative_path: PathBuf,
    found: &mut Vec<PortableArtifactFile>,
) -> WorkerResult<()> {
    let mut input = File::open(path).map_err(WorkerError::Io)?;
    let metadata = input.metadata().map_err(WorkerError::Io)?;
    if !metadata.is_file() {
        return Err(WorkerError::InvalidPayload(
            "auxiliary source is not a regular file".to_owned(),
        ));
    }
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer).map_err(WorkerError::Io)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| WorkerError::Io(io::Error::other("auxiliary source is too large")))?;
    }
    if bytes != metadata.len() {
        return Err(WorkerError::InvalidPayload(
            "auxiliary source changed while hashing".to_owned(),
        ));
    }
    found.push(PortableArtifactFile {
        relative_path,
        size_bytes: bytes,
        sha256: format!("sha256:{:x}", digest.finalize()),
    });
    Ok(())
}
