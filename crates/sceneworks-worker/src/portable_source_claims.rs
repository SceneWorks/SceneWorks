//! Private advertisements for selected worker-owned auxiliary model sources.
//! Physical paths stay on the worker; the API receives only logical IDs and file hashes.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};

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
    "instantid_mlx",
    "instantid_controlnet",
    "instantid_openpose",
    "pulid_adapter",
    "pulid_eva",
    "pulid_bisenet",
    "pulid_scrfd",
    "pulid_arcface",
    "iris_depth_export",
    "iris_generation_text_encoder",
];

static INVENTORY_STARTED: AtomicBool = AtomicBool::new(false);
static PENDING_INVENTORY: OnceLock<Mutex<Option<Vec<PortableSourceClaim>>>> = OnceLock::new();

fn pending_inventory() -> &'static Mutex<Option<Vec<PortableSourceClaim>>> {
    PENDING_INVENTORY.get_or_init(|| Mutex::new(None))
}

/// Registration stays fast even when installed source closure spans terabytes. Exactly one
/// blocking inventory hashes it in the background; the next ordinary heartbeat publishes the
/// complete, fresh private claim vector. Until then the API cannot route bound remote jobs here.
pub(crate) fn start_background_inventory(settings: Settings) {
    if INVENTORY_STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    tokio::spawn(async move {
        let result =
            crate::load_progress::spawn_blocking(move || static_source_claims(&settings)).await;
        match result {
            Ok(claims) => {
                *pending_inventory()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(claims);
            }
            Err(error) => {
                tracing::warn!(%error, "private source inventory failed");
                INVENTORY_STARTED.store(false, Ordering::Release);
            }
        }
    });
}

pub(crate) fn pending_inventory_claims() -> Option<Vec<PortableSourceClaim>> {
    pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

pub(crate) fn mark_inventory_published() {
    *pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

pub(crate) fn static_source_claims(settings: &Settings) -> Vec<PortableSourceClaim> {
    let empty = JsonObject::new();
    let mut claims: Vec<_> = STATIC_SOURCE_IDS
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
        .collect();
    claims.extend(flux2_conversion_source_claims(settings));
    claims.extend(registered_digest_source_claims(settings));
    claims.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    let mut unique = Vec::new();
    let mut index = 0;
    while index < claims.len() {
        let end = index
            + claims[index..]
                .iter()
                .take_while(|claim| claim.source_id == claims[index].source_id)
                .count();
        // The ID includes the exact file ledger. Repeated registrations with identical bytes
        // advertise one availability; registrations with different bytes have different IDs.
        unique.push(claims[index].clone());
        index = end;
    }
    unique
}

fn registered_digest_source_claims(settings: &Settings) -> Vec<PortableSourceClaim> {
    if settings.config_dir.as_os_str().is_empty() {
        return Vec::new();
    }
    let mut claims = Vec::new();
    let manifests = settings.config_dir.join("manifests");
    for model in catalog_records(&manifests.join("user.models.jsonc"), "models") {
        let Some(id) = model.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let primary = model
            .get("modelPath")
            .or_else(|| model.get("paths").and_then(|paths| paths.get("model")))
            .or_else(|| model.get("installedPath"))
            .and_then(serde_json::Value::as_str)
            .map(PathBuf::from);
        if let Some(path) = primary {
            if let Ok(path) = crate::paths::normalize_app_managed_model_entry_path(
                settings,
                &path,
                "registered model source",
            ) {
                if let Ok(files) = capture_registered_files(&path) {
                    append_registered_claim_files(&mut claims, "model", id, None, files.clone());
                    append_registered_claim_files(&mut claims, "control", id, None, files);
                }
            }
        }
        for component in model
            .get("components")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(component_id) = component.get("id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let path = component
                .get("path")
                .or_else(|| model.get("paths").and_then(|paths| paths.get(component_id)))
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from);
            if let Some(path) = path {
                if let Ok(path) = crate::paths::normalize_app_managed_model_entry_path(
                    settings,
                    &path,
                    "registered component source",
                ) {
                    append_registered_claim(
                        &mut claims,
                        "component",
                        id,
                        Some(component_id),
                        &path,
                    );
                }
            }
        }
    }
    let mut lora_manifests = vec![manifests.join("user.loras.jsonc")];
    let projects =
        sceneworks_core::project_store::ProjectStore::new(settings.data_dir.clone(), "worker");
    if let Ok(projects) = projects.list_projects() {
        lora_manifests.extend(
            projects
                .into_iter()
                .map(|project| PathBuf::from(project.path).join("loras/manifest.jsonc")),
        );
    }
    for manifest in lora_manifests {
        for lora in catalog_records(&manifest, "loras") {
            let Some(id) = lora.get("id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(path) = sceneworks_core::lora_family::adapter_record_path(&lora) else {
                continue;
            };
            if let Ok(path) = crate::paths::normalize_app_managed_lora_path(settings, &path) {
                append_registered_claim(&mut claims, "lora", id, None, &path);
            }
        }
    }
    claims
}

fn catalog_records(path: &Path, collection: &str) -> Vec<serde_json::Value> {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(
        &sceneworks_core::jsonc::strip_jsonc_comments(&raw),
    ) else {
        return Vec::new();
    };
    value
        .get(collection)
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn append_registered_claim(
    claims: &mut Vec<PortableSourceClaim>,
    kind: &str,
    model_id: &str,
    component_id: Option<&str>,
    selected: &Path,
) {
    let Ok(files) = capture_registered_files(selected) else {
        return;
    };
    append_registered_claim_files(claims, kind, model_id, component_id, files);
}

pub(crate) fn capture_registered_files(selected: &Path) -> WorkerResult<Vec<PortableArtifactFile>> {
    let mut files = capture_files(selected)?;
    if selected.is_dir() {
        files.retain(|file| {
            !matches!(
                file.relative_path.to_str(),
                Some(
                    ".sceneworks-artifact-provenance.json"
                        | ".sceneworks-artifact-provenance.lock"
                        | ".sceneworks-produced-artifact.json"
                        | ".sceneworks-produced-artifact.lock"
                )
            )
        });
    }
    if files.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "registered source has no loadable files".to_owned(),
        ));
    }
    Ok(files)
}

fn append_registered_claim_files(
    claims: &mut Vec<PortableSourceClaim>,
    kind: &str,
    model_id: &str,
    component_id: Option<&str>,
    files: Vec<PortableArtifactFile>,
) {
    let Ok(ledger) =
        sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(&files)
    else {
        return;
    };
    let Ok(source_id) = sceneworks_core::portable_sources::registered_digest_claim_id(
        kind,
        model_id,
        component_id,
        &ledger,
    ) else {
        return;
    };
    if let Ok(claim) =
        PortableSourceClaim::new(source_id, PortableSourceProvenance::OperatorOverride, files)
    {
        claims.push(claim);
    }
}

fn flux2_conversion_source_claims(settings: &Settings) -> Vec<PortableSourceClaim> {
    let mut claims = Vec::new();
    let hub = sceneworks_core::hf_home::huggingface_hub_cache_dir(&settings.data_dir);
    let Ok(repos) = std::fs::read_dir(hub) else {
        return claims;
    };
    for repo in repos.flatten() {
        let name = repo.file_name();
        let Some(slug) = name.to_str().and_then(|name| name.strip_prefix("models--")) else {
            continue;
        };
        let Some((owner, name)) = slug.split_once("--") else {
            continue;
        };
        let repository = format!("{owner}/{name}");
        let Ok(revisions) = std::fs::read_dir(repo.path().join("snapshots")) else {
            continue;
        };
        for revision in revisions.flatten() {
            let Some(revision_name) = revision.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if revision_name.len() != 40
                || !revision_name.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                continue;
            }
            let mut tiers = vec!["root".to_owned()];
            if let Ok(entries) = std::fs::read_dir(revision.path()) {
                tiers.extend(entries.flatten().filter_map(|entry| {
                    entry.file_type().ok().filter(|kind| kind.is_dir())?;
                    entry.file_name().to_str().map(str::to_owned)
                }));
            }
            for tier in tiers {
                let Ok(source_id) = sceneworks_core::portable_sources::flux2_convert_base_claim_id(
                    &repository,
                    &revision_name,
                    &tier,
                ) else {
                    continue;
                };
                match capture_selected_source(&source_id, settings, &JsonObject::new()) {
                    Ok(Some((claim, _))) => claims.push(claim),
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(source_id, %error, "FLUX.2 conversion source was not advertised")
                    }
                }
            }
        }
    }
    claims
}

/// Capture the exact effective source that the loader resolver would select for this payload.
/// The companion path is worker-private and is never serialized into the claim.
pub(crate) fn capture_selected_source(
    source_id: &str,
    settings: &Settings,
    payload: &JsonObject,
) -> WorkerResult<Option<(PortableSourceClaim, PathBuf)>> {
    if let Some((repository, revision, tier)) =
        sceneworks_core::portable_sources::parse_flux2_convert_base_claim_id(source_id)
    {
        let hub = sceneworks_core::hf_home::huggingface_hub_cache_dir(&settings.data_dir);
        let library =
            sceneworks_core::model_artifacts::ArtifactSourceLibrary::new(hub).map_err(|_| {
                WorkerError::InvalidPayload("conversion source root is invalid".to_owned())
            })?;
        let Ok((_, snapshot)) = library.discover_snapshot(repository, Some(revision)) else {
            return Ok(None);
        };
        let selected = if tier == "root" {
            snapshot
        } else {
            snapshot.join(tier)
        };
        let Some(files) = capture_flux2_borrowed_files(&selected)? else {
            return Ok(None);
        };
        let claim = PortableSourceClaim::new(
            source_id,
            PortableSourceProvenance::PinnedRepo {
                repository: repository.to_owned(),
                revision: revision.to_owned(),
            },
            files,
        )
        .map_err(|_| {
            WorkerError::InvalidPayload("conversion source claim is invalid".to_owned())
        })?;
        return Ok(Some((claim, selected)));
    }
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

fn capture_flux2_borrowed_files(
    selected: &Path,
) -> WorkerResult<Option<Vec<PortableArtifactFile>>> {
    if !["vae", "text_encoder", "tokenizer"]
        .iter()
        .all(|component| selected.join(component).is_dir())
    {
        return Ok(None);
    }
    let mut files = Vec::new();
    for component in ["vae", "text_encoder", "tokenizer"] {
        capture_tree(selected, &selected.join(component), &mut files)?;
    }
    files.sort();
    if files.is_empty() {
        return Ok(None);
    }
    Ok(Some(files))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn registered_digest_claim_uses_logical_id_and_excludes_receipt_markers() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/imports/demo");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("model.safetensors"), b"weights").unwrap();
        std::fs::write(model.join(".sceneworks-produced-artifact.json"), b"receipt").unwrap();
        let mut settings = Settings::for_test(data);
        settings.config_dir = temp.path().join("config");
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[{
                "id":"imported-demo","paths":{"model":model}
            }]}))
            .unwrap(),
        )
        .unwrap();
        let files = capture_registered_files(&model).unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(&files)
                .unwrap();
        let id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "model",
            "imported-demo",
            None,
            &ledger,
        )
        .unwrap();
        let hub = temp.path().join("empty-hub");
        std::fs::create_dir_all(&hub).unwrap();
        let claims =
            crate::test_env::temp_env_vars(&[("HF_HUB_CACHE", hub.to_str().unwrap())], || {
                static_source_claims(&settings)
            });
        let claim = claims.iter().find(|claim| claim.source_id == id).unwrap();
        assert_eq!(claim.files.len(), 1);
        assert_eq!(claim.files[0].relative_path, Path::new("model.safetensors"));
        assert_eq!(claim.provenance, PortableSourceProvenance::OperatorOverride);
    }

    #[test]
    fn registered_digest_duplicate_id_dedupes_equal_bytes_and_retains_distinct_bytes() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let one = data.join("models/one");
        let equal = data.join("models/equal");
        let other = data.join("models/other");
        for directory in [&one, &equal, &other] {
            std::fs::create_dir_all(directory).unwrap();
        }
        std::fs::write(one.join("model.safetensors"), b"same").unwrap();
        std::fs::write(equal.join("model.safetensors"), b"same").unwrap();
        std::fs::write(other.join("model.safetensors"), b"different").unwrap();
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[
                {"id":"shared","modelPath":one},
                {"id":"shared","modelPath":equal},
                {"id":"shared","modelPath":other}
            ]}))
            .unwrap(),
        )
        .unwrap();
        let hub = temp.path().join("empty-hub");
        std::fs::create_dir_all(&hub).unwrap();
        let claims =
            crate::test_env::temp_env_vars(&[("HF_HUB_CACHE", hub.to_str().unwrap())], || {
                static_source_claims(&settings)
            });
        let model_claims: Vec<_> = claims
            .iter()
            .filter(|claim| claim.source_id.starts_with("registered_digest:model:"))
            .collect();
        assert_eq!(model_claims.len(), 2);
        assert_ne!(model_claims[0].source_id, model_claims[1].source_id);
    }

    #[test]
    fn dynamic_flux2_claim_freezes_each_installed_revision_and_tier() {
        let temp = tempfile::TempDir::new().unwrap();
        let hub = temp.path().join("hub");
        let revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let tier = hub
            .join("models--guardfx--borrow/snapshots")
            .join(revision)
            .join("custom-q5");
        for component in ["vae", "text_encoder", "tokenizer"] {
            let file = tier.join(component).join("config.json");
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, component.as_bytes()).unwrap();
        }
        let settings = Settings::for_test(temp.path().join("data"));
        crate::test_env::temp_env_vars(&[("HF_HUB_CACHE", hub.to_str().unwrap())], || {
            let id = sceneworks_core::portable_sources::flux2_convert_base_claim_id(
                "guardfx/borrow",
                revision,
                "custom-q5",
            )
            .unwrap();
            let (claim, selected) = capture_selected_source(&id, &settings, &JsonObject::new())
                .unwrap()
                .unwrap();
            assert_eq!(selected, tier);
            assert_eq!(claim.files.len(), 3);
            assert!(claim
                .files
                .iter()
                .any(|file| file.relative_path == Path::new("vae/config.json")));
            assert!(static_source_claims(&settings)
                .iter()
                .any(|source| source.source_id == id));
        });
    }
}
