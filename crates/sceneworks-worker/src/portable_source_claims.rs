//! Private advertisements for selected worker-owned auxiliary model sources.
//! Physical paths stay on the worker; the API receives only logical IDs and file hashes.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};
use std::time::{Duration, Instant};

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
    "sdxl_detail_tile_control",
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
static PENDING_INVENTORY: OnceLock<Mutex<InventoryProgress>> = OnceLock::new();
static INVENTORY_SCHEDULE: OnceLock<Mutex<InventorySchedule>> = OnceLock::new();
// Catalog writes are detected on the next ordinary heartbeat; this slower sweep covers a file
// appearing at an already-registered path or an operator-installed source without a catalog edit.
const INVENTORY_RESCAN_INTERVAL: Duration = Duration::from_secs(30 * 60);

#[derive(Default)]
struct InventorySchedule {
    next_due: Option<Instant>,
    catalog_hash: Option<[u8; 32]>,
}

#[derive(Default)]
struct InventoryProgress {
    claims: Vec<PortableSourceClaim>,
    generation: u64,
    acknowledged: u64,
}

impl InventoryProgress {
    fn publish(&mut self, claim: PortableSourceClaim) {
        if let Some(existing) = self
            .claims
            .iter_mut()
            .find(|existing| existing.source_id == claim.source_id)
        {
            if *existing != claim {
                *existing = claim;
                self.generation += 1;
            }
        } else {
            self.claims.push(claim);
            self.claims
                .sort_by(|left, right| left.source_id.cmp(&right.source_id));
            self.generation += 1;
        }
    }

    fn replace_scanned(&mut self, claims: Vec<PortableSourceClaim>) {
        if self.claims != claims {
            self.claims = claims;
            self.generation += 1;
        }
    }

    fn pending(&self) -> Option<(u64, Vec<PortableSourceClaim>)> {
        (self.generation > self.acknowledged).then(|| (self.generation, self.claims.clone()))
    }

    fn acknowledge(&mut self, generation: u64) {
        self.acknowledged = self.acknowledged.max(generation.min(self.generation));
    }
}

fn pending_inventory() -> &'static Mutex<InventoryProgress> {
    PENDING_INVENTORY.get_or_init(|| Mutex::new(InventoryProgress::default()))
}

fn inventory_schedule() -> &'static Mutex<InventorySchedule> {
    INVENTORY_SCHEDULE.get_or_init(|| Mutex::new(InventorySchedule::default()))
}

fn scan_due(schedule: &InventorySchedule, now: Instant, catalog_hash: [u8; 32]) -> bool {
    schedule.next_due.is_none_or(|next| now >= next) || schedule.catalog_hash != Some(catalog_hash)
}

fn catalog_hash(settings: &Settings) -> [u8; 32] {
    let manifests = settings.config_dir.join("manifests");
    let mut paths = vec![
        manifests.join("user.models.jsonc"),
        manifests.join("user.loras.jsonc"),
        manifests.join("user.control_overlays.jsonc"),
    ];
    paths.extend(converted_builtin_model_ids().into_iter().map(|id| {
        settings
            .data_dir
            .join("models/mlx")
            .join(id)
            .join(".sceneworks-produced-artifact.json")
    }));
    let projects =
        sceneworks_core::project_store::ProjectStore::new(settings.data_dir.clone(), "worker");
    if let Ok(projects) = projects.list_projects() {
        paths.extend(
            projects
                .iter()
                .map(|project| PathBuf::from(&project.path).join("loras/manifest.jsonc")),
        );
        paths.extend(
            projects
                .into_iter()
                .map(|project| PathBuf::from(project.path).join("control-overlays/manifest.jsonc")),
        );
    }
    paths.sort();
    let mut digest = Sha256::new();
    for path in paths {
        digest.update(path.to_string_lossy().as_bytes());
        match std::fs::read(&path) {
            Ok(bytes) => {
                digest.update([1]);
                digest.update((bytes.len() as u64).to_le_bytes());
                digest.update(bytes);
            }
            Err(_) => digest.update([0]),
        }
    }
    digest.finalize().into()
}

/// Heartbeats schedule a bounded background rescan so imports, conversions, and trained LoRAs
/// appear without a worker restart. Hashing never runs on the heartbeat path.
pub(crate) fn refresh_background_inventory(settings: Settings) {
    let now = Instant::now();
    let hash = catalog_hash(&settings);
    let schedule = inventory_schedule()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !scan_due(&schedule, now, hash) || INVENTORY_STARTED.load(Ordering::Acquire) {
        return;
    }
    drop(schedule);
    start_background_inventory(settings);
}

/// Registration stays fast even when installed source closure spans terabytes. Exactly one
/// blocking inventory hashes it in the background. Each completed source is published on the
/// next ordinary heartbeat; unverified sources remain ineligible until their own hash completes.
pub(crate) fn start_background_inventory(settings: Settings) {
    if INVENTORY_STARTED.swap(true, Ordering::AcqRel) {
        // Registration replaces the server's claim vector. If a scan is already running,
        // republish its completed claims and let it continue producing fresh ones.
        pending_inventory()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .acknowledged = 0;
        return;
    }
    // Registration replaces the server vector even when the preceding scan finished. Republish
    // the last verified snapshot while its replacement is being built.
    pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .acknowledged = 0;
    *inventory_schedule()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = InventorySchedule {
        next_due: Some(Instant::now() + INVENTORY_RESCAN_INTERVAL),
        catalog_hash: Some(catalog_hash(&settings)),
    };
    tokio::spawn(async move {
        let result = crate::load_progress::spawn_blocking(move || {
            let mut scanned = Vec::new();
            collect_source_claims(&settings, |claim| {
                scanned.push(claim.clone());
                pending_inventory()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .publish(claim);
            });
            scanned.sort_by(|left, right| left.source_id.cmp(&right.source_id));
            scanned.dedup_by(|left, right| left.source_id == right.source_id);
            pending_inventory()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .replace_scanned(scanned);
        })
        .await;
        match result {
            Ok(()) => {
                INVENTORY_STARTED.store(false, Ordering::Release);
            }
            Err(error) => {
                tracing::warn!(%error, "private source inventory failed");
                inventory_schedule()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .next_due = None;
                INVENTORY_STARTED.store(false, Ordering::Release);
            }
        }
    });
}

pub(crate) fn pending_inventory_claims() -> Option<(u64, Vec<PortableSourceClaim>)> {
    pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending()
}

pub(crate) fn mark_inventory_published(generation: u64) {
    pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .acknowledge(generation);
}

#[cfg(test)]
fn static_source_claims(settings: &Settings) -> Vec<PortableSourceClaim> {
    let mut claims = Vec::new();
    collect_source_claims(settings, |claim| claims.push(claim));
    claims.sort_by(|left, right| left.source_id.cmp(&right.source_id));
    claims.dedup_by(|left, right| left.source_id == right.source_id);
    claims
}

fn collect_source_claims(settings: &Settings, mut publish: impl FnMut(PortableSourceClaim)) {
    let empty = JsonObject::new();
    for source_id in STATIC_SOURCE_IDS {
        match capture_selected_source(source_id, settings, &empty) {
            Ok(Some((claim, _))) => publish(claim),
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(source_id, %error, "private auxiliary source could not be advertised")
            }
        }
    }
    flux2_conversion_source_claims(settings, &mut publish);
    registered_digest_source_claims(settings, publish);
}

fn registered_digest_source_claims(
    settings: &Settings,
    mut publish: impl FnMut(PortableSourceClaim),
) {
    if settings.config_dir.as_os_str().is_empty() {
        return;
    }
    let mut claims = Vec::new();
    let manifests = settings.config_dir.join("manifests");
    let checkpoint_plans =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
    for model in catalog_records(&manifests.join("user.models.jsonc"), "models") {
        let Some(id) = model.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let mut captured = std::collections::BTreeMap::<PathBuf, Vec<PortableArtifactFile>>::new();
        for (key, path) in registered_model_locators(&model) {
            if let Ok(path) = crate::paths::normalize_app_managed_model_entry_path(
                settings,
                &path,
                "registered model source",
            ) {
                let files = if let Some(files) = captured.get(&path) {
                    Some(files.clone())
                } else {
                    capture_registered_files(&path).ok().inspect(|files| {
                        captured.insert(path.clone(), files.clone());
                    })
                };
                if let Some(files) = files {
                    append_registered_claim_files(&mut claims, "model", id, None, files.clone());
                    append_registered_claim_files(&mut claims, "control", id, None, files.clone());
                    append_registered_claim_files(
                        &mut claims,
                        "component",
                        id,
                        Some(&key),
                        files.clone(),
                    );
                    append_registered_model_bytes(&mut claims, id, &files);
                }
            }
            for claim in claims.drain(..) {
                publish(claim);
            }
        }
        if let Some(checkpoint_id) = model
            .pointer("/importPlan/checkpointId")
            .and_then(serde_json::Value::as_str)
        {
            // Only the trusted plan store may expose a linked library layer. It verifies the
            // approved root, compiled plan, and source bytes before availability is advertised.
            if let Ok(resolved) = checkpoint_plans.resolve(checkpoint_id) {
                for layer in resolved.layers {
                    if let Ok(files) = capture_registered_files(&layer.path) {
                        // The same bytes may have another linked root ID or filename on a
                        // different host. The private plan binds each logical layer; availability
                        // advertises canonical file content for this registered model ID.
                        append_registered_model_bytes(&mut claims, id, &files);
                    }
                }
                for claim in claims.drain(..) {
                    publish(claim);
                }
            }
        }
    }
    for id in converted_builtin_model_ids() {
        let Some(root) = registered_converted_builtin_model_root(settings, &id) else {
            continue;
        };
        if let Some(files) = capture_converted_builtin_files(settings, &root) {
            append_registered_claim_files(&mut claims, "model", &id, None, files);
            for claim in claims.drain(..) {
                publish(claim);
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
            if let Ok(path) = normalize_catalog_lora_path(settings, &manifest, &path) {
                append_registered_claim(&mut claims, "lora", id, None, &path);
            }
            for claim in claims.drain(..) {
                publish(claim);
            }
        }
    }
    for manifest in control_overlay_manifests(settings) {
        for overlay in catalog_records(&manifest, "controlOverlays") {
            let Some(id) = overlay.get("id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(path) = control_overlay_entry_file(settings, &manifest, &overlay) else {
                continue;
            };
            append_registered_claim(&mut claims, "control", id, None, &path);
            for claim in claims.drain(..) {
                publish(claim);
            }
        }
    }
}

fn converted_builtin_model_ids() -> Vec<String> {
    let Ok(manifest) = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
    else {
        return Vec::new();
    };
    manifest
        .get("models")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            entry
                .pointer("/mlx/requiresConversion")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
        })
        .filter_map(|entry| entry.get("id").and_then(serde_json::Value::as_str))
        .map(str::to_owned)
        .collect()
}

pub(crate) fn is_converted_builtin_model_id(id: &str) -> bool {
    converted_builtin_model_ids()
        .iter()
        .any(|builtin| builtin == id)
}

/// A builtin conversion has no user-model record. Its receipt and fixed app-owned output root
/// supply the worker-side registration; API admission separately verifies the completed job.
fn registered_converted_builtin_model_root(settings: &Settings, id: &str) -> Option<PathBuf> {
    if !is_converted_builtin_model_id(id) {
        return None;
    }
    let root = settings.data_dir.join("models/mlx").join(id);
    let root = crate::paths::normalize_app_managed_model_entry_path(
        settings,
        &root,
        "converted builtin model",
    )
    .ok()?;
    if !root.is_dir() {
        return None;
    }
    let marker = root.join(".sceneworks-produced-artifact.json");
    let receipt: crate::ProducedArtifactReceipt =
        serde_json::from_slice(&std::fs::read(marker).ok()?).ok()?;
    crate::verify_produced_artifact_receipt(
        &root,
        &receipt.producer_job_id,
        &receipt.source_artifact_id,
        id,
    )
    .ok()?;
    Some(root)
}

fn capture_converted_builtin_files(
    settings: &Settings,
    root: &Path,
) -> Option<Vec<PortableArtifactFile>> {
    let files = capture_registered_files(root).ok()?;
    for file in &files {
        crate::paths::normalize_app_managed_model_entry_path(
            settings,
            &root.join(&file.relative_path),
            "converted builtin member",
        )
        .ok()?;
    }
    Some(files)
}

pub(crate) fn registered_converted_builtin_model_path(
    settings: &Settings,
    id: &str,
    frozen_files: &[PortableArtifactFile],
) -> Option<PathBuf> {
    let root = registered_converted_builtin_model_root(settings, id)?;
    capture_converted_builtin_files(settings, &root)
        .filter(|files| files == frozen_files)
        .map(|_| root)
}

fn control_overlay_manifests(settings: &Settings) -> Vec<PathBuf> {
    let mut manifests = vec![settings
        .config_dir
        .join("manifests/user.control_overlays.jsonc")];
    let projects =
        sceneworks_core::project_store::ProjectStore::new(settings.data_dir.clone(), "worker");
    if let Ok(projects) = projects.list_projects() {
        manifests.extend(
            projects
                .into_iter()
                .map(|project| PathBuf::from(project.path).join("control-overlays/manifest.jsonc")),
        );
    }
    manifests
}

/// Resolve the same local file the API's control-overlay catalog selects. Hosted rows use a
/// pinned Revision binding and do not advertise an operator Digest source.
fn control_overlay_entry_file(
    settings: &Settings,
    manifest: &Path,
    overlay: &serde_json::Value,
) -> Option<PathBuf> {
    let source = overlay.get("source");
    if source
        .and_then(|source| source.get("provider"))
        .or_else(|| overlay.get("provider"))
        .and_then(serde_json::Value::as_str)
        == Some("huggingface")
    {
        return None;
    }
    let raw = source
        .and_then(|source| source.get("path"))
        .or_else(|| overlay.get("path"))
        .and_then(serde_json::Value::as_str)?;
    let raw = Path::new(raw);
    let scoped = if raw.is_absolute() {
        raw.to_path_buf()
    } else if manifest
        == settings
            .config_dir
            .join("manifests/user.control_overlays.jsonc")
    {
        settings.data_dir.join(raw)
    } else if manifest
        .file_name()
        .is_some_and(|name| name == "manifest.jsonc")
        && manifest.parent().is_some_and(|parent| {
            parent
                .file_name()
                .is_some_and(|name| name == "control-overlays")
        })
    {
        manifest.parent()?.parent()?.join(raw)
    } else {
        return None;
    };
    let path = if scoped.is_file() {
        scoped
    } else {
        let file = overlay
            .get("files")
            .and_then(serde_json::Value::as_array)?
            .first()?
            .as_str()?;
        scoped.join(file)
    };
    let path = crate::paths::normalize_app_managed_model_entry_path(
        settings,
        &path,
        "registered control overlay",
    )
    .ok()?;
    path.is_file().then_some(path)
}

pub(crate) fn registered_control_overlay_path(
    settings: &Settings,
    payload: &JsonObject,
    overlay_id: &str,
    frozen_files: &[PortableArtifactFile],
) -> WorkerResult<Option<PathBuf>> {
    let Some(weights) = payload
        .get("advanced")
        .and_then(|advanced| advanced.get("controlWeights"))
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(None);
    };
    if weights.get("overlayId").and_then(serde_json::Value::as_str) != Some(overlay_id)
        || weights
            .get("_catalogAuthorized")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        return Ok(None);
    }
    let mut manifests = Vec::new();
    if let Some(project) = crate::paths::project_path_for_payload(settings, payload)? {
        manifests.push(project.join("control-overlays/manifest.jsonc"));
    }
    manifests.push(
        settings
            .config_dir
            .join("manifests/user.control_overlays.jsonc"),
    );
    for manifest in manifests {
        for overlay in catalog_records(&manifest, "controlOverlays") {
            if overlay.get("id").and_then(serde_json::Value::as_str) != Some(overlay_id) {
                continue;
            }
            let Some(path) = control_overlay_entry_file(settings, &manifest, &overlay) else {
                continue;
            };
            if capture_registered_files(&path).is_ok_and(|files| files == frozen_files) {
                return Ok(Some(path));
            }
        }
    }
    Ok(None)
}

/// Catalog source paths are relative to their declared scope, not the worker's process cwd.
/// The global manifest uses the data directory; a project manifest lives under
/// `<project>/loras/manifest.jsonc`. The final normalization still confines the resolved path.
pub(crate) fn normalize_catalog_lora_path(
    settings: &Settings,
    manifest: &Path,
    source: &Path,
) -> WorkerResult<PathBuf> {
    let scoped = if source.is_absolute() {
        source.to_path_buf()
    } else if manifest == settings.config_dir.join("manifests/user.loras.jsonc") {
        settings.data_dir.join(source)
    } else if manifest
        .file_name()
        .is_some_and(|name| name == "manifest.jsonc")
        && manifest
            .parent()
            .is_some_and(|parent| parent.file_name().is_some_and(|name| name == "loras"))
    {
        manifest
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| {
                WorkerError::InvalidPayload("project LoRA scope is unavailable".to_owned())
            })?
            .join(source)
    } else {
        return Err(WorkerError::InvalidPayload(
            "LoRA catalog scope is unavailable".to_owned(),
        ));
    };
    crate::paths::normalize_app_managed_lora_path(settings, &scoped)
}

/// The local catalog's loader locators, named with the same relative keys as API admission.
/// The API prefixes secondary IDs with the model slot; the private worker availability claim
/// uses this slot-independent key so one registration serves any carrier position.
pub(crate) fn registered_model_locators(entry: &serde_json::Value) -> Vec<(String, PathBuf)> {
    let mut locators = Vec::new();
    for key in ["modelPath", "installedPath"] {
        if let Some(path) = entry.get(key).and_then(serde_json::Value::as_str) {
            locators.push((key.to_owned(), PathBuf::from(path)));
        }
    }
    if let Some(paths) = entry.get("paths").and_then(serde_json::Value::as_object) {
        for (key, path) in paths {
            if let Some(path) = path.as_str() {
                locators.push((format!("paths/{key}"), PathBuf::from(path)));
            }
        }
    }
    if let Some(components) = entry
        .get("components")
        .and_then(serde_json::Value::as_array)
    {
        for (index, component) in components.iter().enumerate() {
            if let Some(path) = component.get("path").and_then(serde_json::Value::as_str) {
                locators.push((format!("components/{index}"), PathBuf::from(path)));
            }
        }
    }
    locators.sort_by_key(|(key, _)| match key.as_str() {
        "modelPath" => 0,
        "installedPath" => 1,
        "paths/model" => 2,
        _ => 3,
    });
    locators
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

fn append_registered_model_bytes(
    claims: &mut Vec<PortableSourceClaim>,
    model_id: &str,
    files: &[PortableArtifactFile],
) {
    for file in files {
        append_registered_claim_files(
            claims,
            "model-bytes",
            model_id,
            None,
            vec![PortableArtifactFile {
                relative_path: PathBuf::from("artifact"),
                size_bytes: file.size_bytes,
                sha256: file.sha256.clone(),
            }],
        );
    }
}

/// Resolve one frozen plan layer by exact bytes from this worker's registered model sources.
/// The API host's root ID, plan ID and filename never participate; all candidates came from the
/// worker catalog's authorized locators or its own verified checkpoint plan.
pub(crate) fn registered_model_content_path(
    settings: &Settings,
    model_id: &str,
    expected: &PortableArtifactFile,
) -> WorkerResult<Option<PathBuf>> {
    if expected.relative_path != Path::new("artifact") {
        return Ok(None);
    }
    let records = catalog_records(
        &settings.config_dir.join("manifests/user.models.jsonc"),
        "models",
    );
    let matching = records
        .iter()
        .filter(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(model_id))
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Ok(None);
    }
    let model = matching[0];
    let mut candidates = Vec::new();
    for (_, raw) in registered_model_locators(model) {
        let Ok(path) = crate::paths::normalize_app_managed_model_entry_path(
            settings,
            &raw,
            "registered model source",
        ) else {
            continue;
        };
        candidates.push(path);
    }
    if let Some(checkpoint_id) = model
        .pointer("/importPlan/checkpointId")
        .and_then(serde_json::Value::as_str)
    {
        let store =
            sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&settings.data_dir);
        if let Ok(resolved) = store.resolve(checkpoint_id) {
            candidates.extend(resolved.layers.into_iter().map(|layer| layer.path));
        }
    }
    candidates.sort();
    candidates.dedup();
    for selected in candidates {
        let Ok(files) = capture_registered_files(&selected) else {
            continue;
        };
        for file in files {
            if file.size_bytes == expected.size_bytes && file.sha256 == expected.sha256 {
                let path = if selected.is_file() {
                    selected.clone()
                } else {
                    selected.join(&file.relative_path)
                };
                return Ok(Some(path));
            }
        }
    }
    Ok(None)
}

fn flux2_conversion_source_claims(
    settings: &Settings,
    mut publish: impl FnMut(PortableSourceClaim),
) {
    let hub = sceneworks_core::hf_home::huggingface_hub_cache_dir(&settings.data_dir);
    let Ok(repos) = std::fs::read_dir(hub) else {
        return;
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
                    Ok(Some((claim, _))) => publish(claim),
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(source_id, %error, "FLUX.2 conversion source was not advertised")
                    }
                }
            }
        }
    }
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
    fn default_detail_tile_source_is_advertised_for_exact_worker_claim() {
        use sceneworks_core::model_artifacts::portable_binding::{
            PortableArtifactMember, PortableSourceIdentity,
        };
        use sceneworks_core::model_artifacts::ArtifactMemberRole;

        let temp = tempfile::TempDir::new().unwrap();
        let hub = temp.path().join("hub");
        let settings = Settings::for_test(temp.path().join("data"));
        let source =
            sceneworks_core::control_weights::shipped_auxiliary_source("sdxl_detail_tile_control")
                .unwrap();
        let file = hub
            .join("models--xinsir--controlnet-tile-sdxl-1.0/snapshots")
            .join(source.revision)
            .join("diffusion_pytorch_model.safetensors");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"tile control weights").unwrap();

        crate::test_env::temp_env_vars(&[("HF_HUB_CACHE", hub.to_str().unwrap())], || {
            let (selected_claim, selected_path) =
                capture_selected_source("sdxl_detail_tile_control", &settings, &JsonObject::new())
                    .unwrap()
                    .unwrap();
            assert_eq!(selected_path, file);
            assert_eq!(
                selected_claim.provenance,
                PortableSourceProvenance::PinnedRepo {
                    repository: source.repo.to_owned(),
                    revision: source.revision.to_owned(),
                }
            );
            let advertised = static_source_claims(&settings)
                .into_iter()
                .find(|claim| claim.source_id == "sdxl_detail_tile_control")
                .expect("default Detail tile source must be in worker inventory");
            assert_eq!(advertised, selected_claim);
            let member = PortableArtifactMember {
                role: ArtifactMemberRole::Control,
                component_id: Some("sdxl_detail_tile_control".to_owned()),
                source: PortableSourceIdentity::WorkerClaim {
                    source_id: advertised.source_id.clone(),
                    provenance: advertised.provenance.clone(),
                    closure_sha256: advertised.closure_sha256.clone(),
                },
                tier: None,
                files: advertised.files.clone(),
            };
            assert!(advertised.matches_worker_claim_member(&member).is_ok());
            std::fs::write(&file, b"changed tile weights").unwrap();
            let changed = static_source_claims(&settings)
                .into_iter()
                .find(|claim| claim.source_id == "sdxl_detail_tile_control")
                .unwrap();
            assert!(changed.matches_worker_claim_member(&member).is_err());
        });
    }

    #[test]
    fn receipt_backed_builtin_conversion_advertises_and_rebinds_exact_model_bytes() {
        let temp = tempfile::TempDir::new().unwrap();
        let mut settings = Settings::for_test(temp.path().join("data"));
        settings.config_dir = temp.path().join("config");
        let id = "flux2_klein_9b_true_v2";
        let root = settings.data_dir.join("models/mlx").join(id);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model_index.json"), b"{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"converted weights").unwrap();
        let before = catalog_hash(&settings);
        let mut claims = Vec::new();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        assert!(claims.is_empty(), "unreceipted bytes cannot be advertised");
        assert!(registered_converted_builtin_model_path(&settings, id, &[]).is_none());

        crate::write_produced_artifact_receipt(&root, "convert-job", "source/repo", id).unwrap();
        assert_ne!(
            catalog_hash(&settings),
            before,
            "new receipt schedules a rescan"
        );
        let files = capture_registered_files(&root).unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(&files)
                .unwrap();
        let expected = sceneworks_core::portable_sources::registered_digest_claim_id(
            "model", id, None, &ledger,
        )
        .unwrap();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        assert!(claims
            .iter()
            .any(|claim| claim.source_id == expected && claim.files == files));
        assert_eq!(
            registered_converted_builtin_model_path(&settings, id, &files).as_deref(),
            Some(std::fs::canonicalize(&root).unwrap().as_path())
        );

        std::fs::write(root.join("model.safetensors"), b"substituted weights").unwrap();
        assert!(registered_converted_builtin_model_path(&settings, id, &files).is_none());
        let mut rescanned = Vec::new();
        registered_digest_source_claims(&settings, |claim| rescanned.push(claim));
        assert!(!rescanned.iter().any(|claim| claim.source_id == expected));
        #[cfg(unix)]
        {
            std::fs::write(root.join("model.safetensors"), b"converted weights").unwrap();
            let outside = temp.path().join("outside.safetensors");
            std::fs::write(&outside, b"outside").unwrap();
            std::os::unix::fs::symlink(&outside, root.join("escaped.safetensors")).unwrap();
            crate::write_produced_artifact_receipt(&root, "convert-job", "source/repo", id)
                .unwrap();
            assert!(registered_converted_builtin_model_path(
                &settings,
                id,
                &capture_registered_files(&root).unwrap()
            )
            .is_none());
        }
    }

    #[test]
    fn registered_control_overlay_claims_rebind_only_matching_confined_bytes() {
        let temp = tempfile::TempDir::new().unwrap();
        let mut settings = Settings::for_test(temp.path().join("data"));
        settings.config_dir = temp.path().join("config");
        let global_file = settings.data_dir.join("overlays/global.safetensors");
        std::fs::create_dir_all(global_file.parent().unwrap()).unwrap();
        std::fs::write(&global_file, b"global overlay").unwrap();
        let global_manifest = settings
            .config_dir
            .join("manifests/user.control_overlays.jsonc");
        std::fs::create_dir_all(global_manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &global_manifest,
            serde_json::to_vec(&json!({"controlOverlays":[
                {"id":"global-pose","source":{"path":"overlays/global.safetensors"}}
            ]}))
            .unwrap(),
        )
        .unwrap();
        let store =
            sceneworks_core::project_store::ProjectStore::new(settings.data_dir.clone(), "worker");
        let project = store.create_project("Pose").unwrap();
        let project_path = PathBuf::from(&project.path);
        let project_file = project_path.join("control-overlays/pose.safetensors");
        std::fs::create_dir_all(project_file.parent().unwrap()).unwrap();
        std::fs::write(&project_file, b"project overlay").unwrap();
        std::fs::write(
            project_file.parent().unwrap().join("manifest.jsonc"),
            serde_json::to_vec(&json!({"controlOverlays":[
                {"id":"project-pose","source":{"path":"control-overlays/pose.safetensors"}}
            ]}))
            .unwrap(),
        )
        .unwrap();
        let project_files = capture_registered_files(&project_file).unwrap();
        let mut claims = Vec::new();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        let claim_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "control",
            "project-pose",
            None,
            &sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &project_files,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(claims
            .iter()
            .any(|claim| claim.source_id == claim_id && claim.files == project_files));
        let payload = json!({"projectId":project.id,"advanced":{"controlWeights":{
            "overlayId":"project-pose","_catalogAuthorized":true
        }}})
        .as_object()
        .unwrap()
        .clone();
        assert_eq!(
            registered_control_overlay_path(&settings, &payload, "project-pose", &project_files)
                .unwrap()
                .as_deref(),
            Some(std::fs::canonicalize(&project_file).unwrap().as_path())
        );
        std::fs::write(&project_file, b"changed overlay").unwrap();
        assert!(registered_control_overlay_path(
            &settings,
            &payload,
            "project-pose",
            &project_files
        )
        .unwrap()
        .is_none());
        let global_files = capture_registered_files(&global_file).unwrap();
        let global_claim_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "control",
            "global-pose",
            None,
            &sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &global_files,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(claims
            .iter()
            .any(|claim| claim.source_id == global_claim_id && claim.files == global_files));
        #[cfg(unix)]
        {
            let escaped = temp.path().join("escaped.safetensors");
            std::fs::write(&escaped, b"outside").unwrap();
            let link = settings.data_dir.join("overlays/escape.safetensors");
            std::os::unix::fs::symlink(&escaped, &link).unwrap();
            assert!(control_overlay_entry_file(
                &settings,
                &global_manifest,
                &json!({"source":{"path":"overlays/escape.safetensors"}})
            )
            .is_none());
        }
    }

    #[test]
    fn registered_model_bytes_rebind_across_filename_and_refuse_substitution() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("worker");
        let worker_file = data.join("models/renamed-worker.safetensors");
        std::fs::create_dir_all(worker_file.parent().unwrap()).unwrap();
        std::fs::write(&worker_file, b"same plan layer bytes").unwrap();
        let mut settings = Settings::for_test(data);
        settings.config_dir = temp.path().join("config");
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[{"id":"linked-model", "modelPath":worker_file}]}))
                .unwrap(),
        )
        .unwrap();
        let sha256 = format!("sha256:{:x}", Sha256::digest(b"same plan layer bytes"));
        let expected = PortableArtifactFile {
            relative_path: PathBuf::from("artifact"),
            size_bytes: b"same plan layer bytes".len() as u64,
            sha256,
        };
        assert_eq!(
            registered_model_content_path(&settings, "linked-model", &expected)
                .unwrap()
                .as_deref(),
            Some(std::fs::canonicalize(&worker_file).unwrap().as_path())
        );
        let claim_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "model-bytes",
            "linked-model",
            None,
            &sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                std::slice::from_ref(&expected),
            )
            .unwrap(),
        )
        .unwrap();
        let mut claims = Vec::new();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        assert!(claims
            .iter()
            .any(|claim| claim.source_id == claim_id && claim.files == vec![expected.clone()]));
        std::fs::write(&worker_file, b"substituted plan layer").unwrap();
        assert!(
            registered_model_content_path(&settings, "linked-model", &expected)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn incremental_claim_publication_preserves_progress_across_heartbeat_ack() {
        let make = |source_id| {
            PortableSourceClaim::new(
                source_id,
                PortableSourceProvenance::OperatorOverride,
                vec![PortableArtifactFile {
                    relative_path: PathBuf::from("model.safetensors"),
                    size_bytes: 1,
                    sha256: format!("sha256:{}", "a".repeat(64)),
                }],
            )
            .unwrap()
        };
        let mut progress = InventoryProgress::default();
        progress.publish(make("first"));
        let (first_generation, first_claims) = progress.pending().unwrap();
        assert_eq!(first_claims.len(), 1);
        progress.publish(make("second"));
        progress.acknowledge(first_generation);
        let (second_generation, second_claims) = progress.pending().unwrap();
        assert_eq!(second_claims.len(), 2);
        progress.acknowledge(second_generation);
        assert!(progress.pending().is_none());
        progress.publish(make("second"));
        assert!(progress.pending().is_none());
    }

    #[test]
    fn rescans_publish_new_sources_and_withdraw_removed_or_changed_claims() {
        let make = |source_id: &str, byte: char| {
            PortableSourceClaim::new(
                source_id,
                PortableSourceProvenance::OperatorOverride,
                vec![PortableArtifactFile {
                    relative_path: PathBuf::from("model.safetensors"),
                    size_bytes: 1,
                    sha256: format!("sha256:{}", byte.to_string().repeat(64)),
                }],
            )
            .unwrap()
        };
        let mut progress = InventoryProgress::default();
        progress.publish(make("removed", 'a'));
        progress.acknowledge(progress.pending().unwrap().0);
        progress.publish(make("new", 'b'));
        assert_eq!(progress.pending().unwrap().1.len(), 2);
        progress.publish(make("removed", 'c'));
        assert_eq!(
            progress.pending().unwrap().1[1].files[0].sha256,
            format!("sha256:{}", "c".repeat(64))
        );
        progress.replace_scanned(vec![make("new", 'b')]);
        let (generation, claims) = progress.pending().unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].source_id, "new");
        progress.acknowledge(generation);
        assert!(progress.pending().is_none());

        let now = Instant::now();
        let hash = [1; 32];
        assert!(scan_due(&InventorySchedule::default(), now, hash));
        let schedule = InventorySchedule {
            next_due: Some(now + INVENTORY_RESCAN_INTERVAL),
            catalog_hash: Some(hash),
        };
        assert!(!scan_due(&schedule, now, hash));
        assert!(scan_due(&schedule, now, [2; 32]));
        assert!(scan_due(&schedule, now + INVENTORY_RESCAN_INTERVAL, hash));
    }

    #[test]
    fn changed_catalog_is_claimable_after_rescan_without_worker_restart() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let manifest = settings.config_dir.join("manifests/user.loras.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, br#"{"loras":[]}"#).unwrap();
        let mut progress = InventoryProgress::default();
        let original_hash = catalog_hash(&settings);
        let adapter = data.join("loras/trained/adapter.safetensors");
        std::fs::create_dir_all(adapter.parent().unwrap()).unwrap();
        std::fs::write(&adapter, b"trained bytes").unwrap();
        std::fs::write(
            &manifest,
            br#"{"loras":[{"id":"trained","source":{"path":"loras/trained"}}]}"#,
        )
        .unwrap();
        let updated_hash = catalog_hash(&settings);
        assert_ne!(original_hash, updated_hash);
        let schedule = InventorySchedule {
            next_due: Some(Instant::now() + INVENTORY_RESCAN_INTERVAL),
            catalog_hash: Some(original_hash),
        };
        assert!(scan_due(&schedule, Instant::now(), updated_hash));
        let mut scanned = Vec::new();
        registered_digest_source_claims(&settings, |claim| scanned.push(claim));
        progress.replace_scanned(scanned);
        let (generation, claims) = progress
            .pending()
            .expect("new trained source is advertised");
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0].files,
            capture_registered_files(adapter.parent().unwrap()).unwrap()
        );
        progress.acknowledge(generation);

        std::fs::write(&manifest, br#"{"loras":[]}"#).unwrap();
        let mut rescanned = Vec::new();
        registered_digest_source_claims(&settings, |claim| rescanned.push(claim));
        progress.replace_scanned(rescanned);
        assert!(
            progress.pending().unwrap().1.is_empty(),
            "removed source is withdrawn"
        );
    }

    #[test]
    fn catalog_lora_relative_paths_use_global_or_project_scope() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let global = data.join("loras/global/model.safetensors");
        let project = data.join("projects/demo/loras/project/model.safetensors");
        for file in [&global, &project] {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, b"adapter").unwrap();
        }
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let global_manifest = settings.config_dir.join("manifests/user.loras.jsonc");
        let project_manifest = data.join("projects/demo/loras/manifest.jsonc");
        assert_eq!(
            normalize_catalog_lora_path(
                &settings,
                &global_manifest,
                Path::new("loras/global/model.safetensors")
            )
            .unwrap(),
            global.canonicalize().unwrap()
        );
        assert_eq!(
            normalize_catalog_lora_path(
                &settings,
                &project_manifest,
                Path::new("loras/project/model.safetensors")
            )
            .unwrap(),
            project.canonicalize().unwrap()
        );
        assert!(
            normalize_catalog_lora_path(&settings, &global_manifest, Path::new("../escape"))
                .is_err()
        );

        std::fs::create_dir_all(global_manifest.parent().unwrap()).unwrap();
        std::fs::write(&global_manifest, serde_json::to_vec(&json!({"loras":[{"id":"global","source":{"path":"loras/global/model.safetensors"}}]})).unwrap()).unwrap();
        let mut claims = Vec::new();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        let files = capture_registered_files(&global).unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(&files)
                .unwrap();
        let expected = sceneworks_core::portable_sources::registered_digest_claim_id(
            "lora", "global", None, &ledger,
        )
        .unwrap();
        assert!(claims
            .iter()
            .any(|claim| claim.source_id == expected && claim.files == files));
    }

    #[test]
    fn failed_registered_source_does_not_block_later_verified_claim() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let valid = data.join("models/valid/model.safetensors");
        std::fs::create_dir_all(valid.parent().unwrap()).unwrap();
        std::fs::write(&valid, b"valid").unwrap();
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[
                {"id":"missing","modelPath":data.join("models/missing")},
                {"id":"valid","modelPath":valid}
            ]}))
            .unwrap(),
        )
        .unwrap();
        let mut published = Vec::new();
        registered_digest_source_claims(&settings, |claim| published.push(claim));
        assert_eq!(published.len(), 4); // model, control, locator alias, generic content
        assert!(published
            .iter()
            .all(|claim| claim.source_id.starts_with("registered_digest:")));
    }

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
    fn registered_multifile_claims_use_slot_independent_locator_keys() {
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let model = data.join("models/multi");
        std::fs::create_dir_all(&model).unwrap();
        for name in ["model", "vae", "encoder"] {
            std::fs::write(model.join(format!("{name}.safetensors")), name.as_bytes()).unwrap();
        }
        let mut settings = Settings::for_test(data);
        settings.config_dir = temp.path().join("config");
        let manifest = settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &manifest,
            serde_json::to_vec(&json!({"models":[{
                "id":"multi", "modelPath":model.join("model.safetensors"),
                "paths":{"vae":model.join("vae.safetensors")},
                "components":[{"id":"semantic-encoder", "path":model.join("encoder.safetensors")}]
            }]}))
            .unwrap(),
        )
        .unwrap();
        let mut claims = Vec::new();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        for (key, name) in [("paths/vae", "vae"), ("components/0", "encoder")] {
            let files =
                capture_registered_files(&model.join(format!("{name}.safetensors"))).unwrap();
            let ledger =
                sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                    &files,
                )
                .unwrap();
            let id = sceneworks_core::portable_sources::registered_digest_claim_id(
                "component",
                "multi",
                Some(key),
                &ledger,
            )
            .unwrap();
            assert!(claims
                .iter()
                .any(|claim| claim.source_id == id && claim.files == files));
        }
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
