//! Private advertisements for selected worker-owned auxiliary model sources.
//! Physical paths stay on the worker; the API receives only logical IDs and file hashes.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock,
};
use std::time::{Duration, Instant};

use sceneworks_core::contracts::JsonObject;
use sceneworks_core::model_artifacts::portable_binding::{
    PortableArtifactFile, PortableSourceClaim, PortableSourceProvenance,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{Settings, WorkerError, WorkerResult};

const STATIC_SOURCE_IDS: &[&str] = &[
    "ltx_amoral_gemma_3_12b",
    "ltx_training_gemma_override",
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
static CHALLENGE_ATTEMPTS: OnceLock<Mutex<BTreeMap<String, Instant>>> = OnceLock::new();
static CHALLENGE_SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
// Catalog writes are detected on the next ordinary heartbeat; this slower sweep covers a file
// appearing at an already-registered path or an operator-installed source without a catalog edit.
const INVENTORY_RESCAN_INTERVAL: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PortableSourceChallenge {
    pub(crate) id: String,
    source_id: String,
    files: Vec<PortableArtifactFile>,
}

pub(crate) fn start_app_managed_challenges(
    api: crate::api_client::ApiClient,
    settings: Settings,
    challenges: Vec<PortableSourceChallenge>,
    capability: String,
) {
    let attempts = CHALLENGE_ATTEMPTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let slots = CHALLENGE_SLOTS.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)));
    let mut grouped = BTreeMap::<String, Vec<PortableSourceChallenge>>::new();
    for challenge in challenges {
        grouped
            .entry(challenge.source_id.clone())
            .or_default()
            .push(challenge);
    }
    for (source_id, mut group) in grouped {
        let mut guard = attempts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        guard.retain(|_, until| *until > now);
        if guard.contains_key(&source_id) {
            continue;
        }
        let Ok(permit) = slots.clone().try_acquire_owned() else {
            continue;
        };
        guard.insert(source_id.clone(), now + Duration::from_secs(2 * 60 + 10));
        drop(guard);
        let challenge = group.remove(0);
        let mut ids = group
            .into_iter()
            .map(|entry| entry.id)
            .collect::<std::collections::BTreeSet<_>>();
        ids.insert(challenge.id.clone());
        let api = api.clone();
        let settings = settings.clone();
        let capability = capability.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let ledger =
                sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                    &challenge.files,
                );
            let local = settings.clone();
            let proved = tokio::task::spawn_blocking(move || {
                prove_app_managed_challenges(&local, vec![challenge])
            })
            .await
            .is_ok_and(|result| result.unwrap_or(false));
            if proved {
                if let Ok(ledger_sha256) = ledger {
                    let path = format!(
                        "/api/v1/workers/{}/portable-source-challenges",
                        settings.worker_id
                    );
                    let fresh: WorkerResult<Vec<PortableSourceChallenge>> = api
                        .post_json(&path, &serde_json::json!({"capability":capability}))
                        .await;
                    if let Ok(fresh) = fresh {
                        ids.extend(
                            fresh
                                .into_iter()
                                .filter(|entry| entry.source_id == source_id)
                                .map(|entry| entry.id),
                        );
                    }
                    for id in ids {
                        let _: WorkerResult<serde_json::Value> = api.post_json(
                            &format!("/api/v1/workers/{}/portable-source-proof", settings.worker_id),
                            &serde_json::json!({"id":id,"capability":capability,"ledgerSha256":ledger_sha256}),
                        ).await;
                    }
                }
            }
            if let Some(attempts) = CHALLENGE_ATTEMPTS.get() {
                attempts
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(source_id, Instant::now() + Duration::from_secs(10));
            }
        });
    }
}

/// Resolve a path-free, API-authored probe against this worker's declared model roots. A claim
/// is published only after the complete local file set, sizes and hashes match the challenge.
pub(crate) fn prove_app_managed_challenges(
    settings: &Settings,
    challenges: Vec<PortableSourceChallenge>,
) -> WorkerResult<bool> {
    let mut published = false;
    let mut seen = std::collections::BTreeSet::new();
    for challenge in challenges {
        if !seen.insert(challenge.source_id.clone()) {
            continue;
        }
        let _challenge_id = challenge.id;
        let source_id = challenge.source_id;
        let Some(_source) = find_app_managed_exact_source(settings, &challenge.files)? else {
            continue;
        };
        let claim = PortableSourceClaim::new(
            source_id,
            PortableSourceProvenance::OperatorOverride,
            challenge.files,
        )
        .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        pending_inventory()
            .lock()
            .expect("inventory lock poisoned")
            .publish_probe(claim);
        published = true;
    }
    Ok(published)
}

/// Also used at the pre-loader guard, so a stale claim cannot authorize changed bytes.
pub(crate) fn find_app_managed_exact_source(
    settings: &Settings,
    expected: &[PortableArtifactFile],
) -> WorkerResult<Option<PathBuf>> {
    let Some(first) = expected.first() else {
        return Ok(None);
    };
    let roots = crate::paths::allowed_model_roots(settings)?;
    let mut visited = std::collections::BTreeSet::new();
    let mut stack = roots.clone();
    let deadline = Instant::now() + Duration::from_secs(2 * 60);
    while let Some(path) = stack.pop() {
        if Instant::now() > deadline || visited.len() > 1_000_000 {
            return Err(WorkerError::InvalidPayload(
                "app-managed source probe exceeded its bounded scan".to_owned(),
            ));
        }
        let Ok(canonical) = std::fs::canonicalize(&path) else {
            continue;
        };
        if !visited.insert(canonical.clone()) {
            continue;
        }
        if !roots.iter().any(|root| canonical.starts_with(root)) {
            continue;
        }
        if path
            .join(&first.relative_path)
            .metadata()
            .is_ok_and(|meta| meta.is_file() && meta.len() == first.size_bytes)
            && capture_registered_files(&path).is_ok_and(|files| files == expected)
        {
            return Ok(Some(path));
        }
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                stack.push(entry.path());
            }
        }
    }
    Ok(None)
}

pub(crate) fn selected_app_managed_exact_source(
    settings: &Settings,
    selection: &str,
    expected: &[PortableArtifactFile],
) -> WorkerResult<Option<PathBuf>> {
    let selected = crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, selection)
        .unwrap_or_else(|| {
            let path = PathBuf::from(selection);
            if path.is_absolute() {
                path
            } else {
                settings.data_dir.join(path)
            }
        });
    if selected.is_dir() {
        if capture_registered_files(&selected).is_ok_and(|files| files == expected) {
            return Ok(Some(selected));
        }
        return Err(WorkerError::InvalidPayload(
            "selected utility model differs from admitted bytes".to_owned(),
        ));
    }
    find_app_managed_exact_source(settings, expected)
}

#[derive(Default)]
struct InventorySchedule {
    next_due: Option<Instant>,
    catalog_hash: Option<[u8; 32]>,
}

#[derive(Default)]
struct InventoryProgress {
    claims: Vec<PortableSourceClaim>,
    scanned: Vec<PortableSourceClaim>,
    // A proved source may back a queued job for arbitrarily long while this worker is busy.
    // Keep its last proved claim until replacement or worker restart; the pre-loader guard
    // rechecks the exact bytes before opening the selected source.
    probed: BTreeMap<String, (PortableSourceClaim, Instant)>,
    generation: u64,
    acknowledged_by_worker: BTreeMap<String, u64>,
}

impl InventoryProgress {
    fn publish(&mut self, claim: PortableSourceClaim) {
        if let Some(existing) = self
            .scanned
            .iter_mut()
            .find(|existing| existing.source_id == claim.source_id)
        {
            *existing = claim.clone();
        } else {
            self.scanned.push(claim.clone());
        }
        self.publish_claim(claim);
    }

    fn publish_probe(&mut self, claim: PortableSourceClaim) {
        self.probed
            .insert(claim.source_id.clone(), (claim.clone(), Instant::now()));
        self.publish_claim(claim);
    }

    fn publish_claim(&mut self, claim: PortableSourceClaim) {
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
        self.scanned = claims;
        self.rebuild_claims();
    }

    fn rebuild_claims(&mut self) {
        let mut claims = self.scanned.clone();
        for (claim, _) in self.probed.values() {
            if !claims
                .iter()
                .any(|existing| existing.source_id == claim.source_id)
            {
                claims.push(claim.clone());
            }
        }
        claims.sort_by(|left, right| left.source_id.cmp(&right.source_id));
        if self.claims != claims {
            self.claims = claims;
            self.generation += 1;
        }
    }

    fn pending(&self, worker_id: &str) -> Option<(u64, Vec<PortableSourceClaim>)> {
        (self.generation
            > self
                .acknowledged_by_worker
                .get(worker_id)
                .copied()
                .unwrap_or(0))
        .then(|| (self.generation, self.claims.clone()))
    }

    fn acknowledge(&mut self, worker_id: &str, generation: u64) {
        let acknowledged = self
            .acknowledged_by_worker
            .entry(worker_id.to_owned())
            .or_default();
        *acknowledged = (*acknowledged).max(generation.min(self.generation));
    }

    fn reset_worker(&mut self, worker_id: &str) {
        self.acknowledged_by_worker.remove(worker_id);
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

fn registered_projects(settings: &Settings) -> Vec<sceneworks_core::project_store::ProjectSummary> {
    let store =
        sceneworks_core::project_store::ProjectStore::new(settings.data_dir.clone(), "worker");
    store.list_projects_read_only().unwrap_or_default()
}

fn catalog_hash(settings: &Settings) -> [u8; 32] {
    let manifests = settings.config_dir.join("manifests");
    let mut paths = vec![
        manifests.join("user.models.jsonc"),
        manifests.join("builtin.loras.jsonc"),
        manifests.join("user.loras.jsonc"),
        manifests.join("user.control_overlays.jsonc"),
    ];
    paths.extend(
        catalog_records(&manifests.join("builtin.loras.jsonc"), "loras")
            .into_iter()
            .filter_map(|entry| {
                entry
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .map(|id| {
                settings
                    .data_dir
                    .join("loras")
                    .join(crate::paths::safe_download_dir(&id))
                    .join(".sceneworks-download-complete.json")
            }),
    );
    // Builtin model download receipts (SW-7030). `run_model_download_job` installs each builtin
    // catalog `downloads[].repo` into `data/models/<safe_download_dir(repo)>` (the API enqueues
    // exactly that targetDir) and writes its completion receipt there. The initial inventory can
    // finish before the user installs a builtin auxiliary such as `controlnet_tile_sdxl`, and the
    // builtin manifest itself never changes, so without this marker a completed install leaves the
    // scan stale until the 30-minute rescan. The receipt is tiny JSON metadata — the watched file
    // never hashes the multi-GiB weights it stands in for.
    paths.extend(builtin_model_download_receipt_paths(settings));
    paths.extend(converted_builtin_model_ids().into_iter().map(|id| {
        settings
            .data_dir
            .join("models/mlx")
            .join(id)
            .join(".sceneworks-produced-artifact.json")
    }));
    let projects = registered_projects(settings);
    for project in &projects {
        paths.extend(character_sidecar_paths(Path::new(&project.path)));
    }
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
    // Registration replaces only this worker's server-side claim vector. Another worker in the
    // same process must retain its independent publication progress across this registration.
    pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .reset_worker(&settings.worker_id);
    if INVENTORY_STARTED.swap(true, Ordering::AcqRel) {
        // Registration replaces the server's claim vector. If a scan is already running,
        // republish its completed claims and let it continue producing fresh ones.
        return;
    }
    // Republish this worker's last verified snapshot while its replacement is being built.
    *inventory_schedule()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = InventorySchedule {
        next_due: Some(Instant::now() + INVENTORY_RESCAN_INTERVAL),
        catalog_hash: Some(catalog_hash(&settings)),
    };
    #[cfg(test)]
    let fixture_cache_root =
        sceneworks_core::hf_home::test_support::CacheRootGuard::retain(&settings.data_dir);
    tokio::spawn(async move {
        let result = crate::load_progress::spawn_blocking(move || {
            #[cfg(test)]
            let _fixture_cache_root = fixture_cache_root;
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

pub(crate) fn pending_inventory_claims(worker_id: &str) -> Option<(u64, Vec<PortableSourceClaim>)> {
    let mut inventory = pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    inventory.rebuild_claims();
    inventory.pending(worker_id)
}

/// The API refused a heartbeat with `PORTABLE_SOURCE_CLAIMS_REQUIRED_CODE` (it restarted and holds
/// no claims proven under this worker's key). Forget this worker's acknowledgment and return the
/// whole current claim vector — empty included — for an explicit republication.
pub(crate) fn inventory_claims_for_republication(
    worker_id: &str,
) -> (u64, Vec<PortableSourceClaim>) {
    let mut inventory = pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    inventory.rebuild_claims();
    inventory.reset_worker(worker_id);
    (inventory.generation, inventory.claims.clone())
}

pub(crate) fn mark_inventory_published(worker_id: &str, generation: u64) {
    pending_inventory()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .acknowledge(worker_id, generation);
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
    for (model_id, path, is_file) in
        crate::text_encoder_selection::portable_registered_source_paths(settings)
    {
        if let Ok((_, files)) = selected_text_encoder_root_and_files(&path, is_file) {
            let mut claims = Vec::new();
            append_registered_claim_files(
                &mut claims,
                "text-encoder",
                &format!("{model_id}:{}", if is_file { "file" } else { "directory" }),
                None,
                files,
            );
            for claim in claims {
                publish(claim);
            }
        }
    }
    external_lora_source_claims(settings, &mut publish);
    external_base_source_claims(settings, &mut publish);
    registered_digest_source_claims(settings, publish);
}

/// Advertise the operator's externally scanned ComfyUI base models (SW-7030,
/// review 4533688093). Runnable `external_base_…` rows are forwarded to the
/// worker as local Digest entries (primary anchor plus `components/{index}`
/// members), so the worker must advertise registered digest claims under
/// exactly the claim ids admission computes: kind `model` for the primary and
/// kind `component` with component id `components/{index}` for each declared
/// component. The assembly scan is shared with the API through
/// `sceneworks_core::external_base_models`, so both sides assign the same id
/// to the same tree.
fn external_base_source_claims(settings: &Settings, publish: &mut impl FnMut(PortableSourceClaim)) {
    if settings.external_model_roots.is_empty() {
        return;
    }
    let mut claims = Vec::new();
    for row in external_base_rows(settings) {
        let Some(id) = row.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if let Some(anchor) = row
            .get("installedPath")
            .and_then(serde_json::Value::as_str)
            .and_then(|path| confine_managed_path(settings, Path::new(path)))
        {
            append_registered_claim(settings, &mut claims, "model", id, None, &anchor);
        }
        if let Some(components) = row.get("components").and_then(serde_json::Value::as_array) {
            for (index, component) in components.iter().enumerate() {
                let Some(confined) = component
                    .get("path")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|path| confine_managed_path(settings, Path::new(path)))
                else {
                    continue;
                };
                append_registered_claim(
                    settings,
                    &mut claims,
                    "component",
                    id,
                    Some(&format!("components/{index}")),
                    &confined,
                );
            }
        }
    }
    for claim in claims {
        publish(claim);
    }
}

/// The assembled external base-model rows the shared scan produces here.
fn external_base_rows(settings: &Settings) -> Vec<serde_json::Value> {
    let mut cache = sceneworks_core::external_base_models::ExternalBaseModelCache::default();
    sceneworks_core::external_base_models::scan_external_base_models(
        &settings.external_model_roots,
        &mut cache,
    )
}

/// Confine a scan-reported path to this worker's managed roots, returning the
/// canonical target admission capture would hash.
fn confine_managed_path(settings: &Settings, path: &Path) -> Option<PathBuf> {
    let roots = crate::paths::allowed_model_roots(settings).ok()?;
    sceneworks_core::model_artifacts::confine_artifact_path(path, &roots).ok()
}

/// Resolve an `external_base_…` model (or one of its `components/{index}`
/// members) to this worker's confined copy of the exact frozen bytes, or
/// `None` when the id is not external, is not present under the configured
/// roots, or its bytes differ from the frozen ledger.
pub(crate) fn registered_external_base_path(
    settings: &Settings,
    id: &str,
    component_id: Option<&str>,
    frozen_files: &[PortableArtifactFile],
) -> WorkerResult<Option<PathBuf>> {
    if !id.starts_with(sceneworks_core::external_base_models::EXTERNAL_ID_PREFIX) {
        return Ok(None);
    }
    for row in external_base_rows(settings) {
        if row.get("id").and_then(serde_json::Value::as_str) != Some(id) {
            continue;
        }
        let candidate = match component_id {
            None => row
                .get("installedPath")
                .and_then(serde_json::Value::as_str)
                .map(PathBuf::from),
            Some(component) => {
                let Some(index) = component
                    .strip_prefix("components/")
                    .and_then(|index| index.parse::<usize>().ok())
                else {
                    return Ok(None);
                };
                row.get("components")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|components| components.get(index))
                    .and_then(|component| component.get("path"))
                    .and_then(serde_json::Value::as_str)
                    .map(PathBuf::from)
            }
        };
        let Some(confined) = candidate.and_then(|path| confine_managed_path(settings, &path))
        else {
            continue;
        };
        let roots = crate::paths::allowed_model_roots(settings).unwrap_or_default();
        if capture_registered_files_with_roots(&confined, &roots)
            .is_ok_and(|files| files == frozen_files)
        {
            return Ok(Some(confined));
        }
    }
    Ok(None)
}

/// Advertise the operator's externally scanned ComfyUI LoRAs (SW-7030, review
/// 4531591689). The API's `lora_catalog` publishes these synthetic `external_…`
/// rows as installed, so portable admission requires a registered digest claim
/// for exactly that id; the scan is shared with the API through
/// `sceneworks_core::external_loras`, so both sides assign the same id to the
/// same file. Each candidate is confined through
/// `normalize_app_managed_lora_path` before its bytes are hashed, exactly like
/// manifest-backed adapters. This runs regardless of `config_dir`: external
/// roots come from the process environment, never from a manifest.
fn external_lora_source_claims(settings: &Settings, publish: &mut impl FnMut(PortableSourceClaim)) {
    if settings.external_model_roots.is_empty() {
        return;
    }
    let mut claims = Vec::new();
    for (id, path) in external_lora_entries(settings) {
        let Ok(confined) = crate::paths::normalize_app_managed_lora_path(settings, &path) else {
            continue;
        };
        append_registered_claim(settings, &mut claims, "lora", &id, None, &confined);
    }
    for claim in claims {
        publish(claim);
    }
}

/// The `(id, path)` pairs the shared external-LoRA scan assigns on this worker.
fn external_lora_entries(settings: &Settings) -> Vec<(String, PathBuf)> {
    let mut cache = sceneworks_core::external_loras::ExternalLoraCache::default();
    sceneworks_core::external_loras::scan_external_loras(&settings.external_model_roots, &mut cache)
        .into_iter()
        .filter_map(|row| {
            let id = row.get("id")?.as_str()?.to_owned();
            let path = row.pointer("/source/path")?.as_str()?.to_owned();
            Some((id, PathBuf::from(path)))
        })
        .collect()
}

/// Resolve an `external_…` LoRA id to this worker's confined copy of the exact
/// frozen bytes, or `None` when the id is not external, is not present under
/// the configured roots, or its bytes differ from the frozen ledger.
pub(crate) fn registered_external_lora_path(
    settings: &Settings,
    id: &str,
    frozen_files: &[PortableArtifactFile],
) -> WorkerResult<Option<PathBuf>> {
    if !id.starts_with(sceneworks_core::external_loras::EXTERNAL_ID_PREFIX) {
        return Ok(None);
    }
    for (entry_id, path) in external_lora_entries(settings) {
        if entry_id != id {
            continue;
        }
        let Ok(confined) = crate::paths::normalize_app_managed_lora_path(settings, &path) else {
            continue;
        };
        let roots = crate::paths::allowed_model_roots(settings).unwrap_or_default();
        if capture_registered_files_with_roots(&confined, &roots)
            .is_ok_and(|files| files == frozen_files)
        {
            return Ok(Some(confined));
        }
    }
    Ok(None)
}

fn selected_text_encoder_root_and_files(
    selected: &Path,
    is_file: bool,
) -> WorkerResult<(PathBuf, Vec<PortableArtifactFile>)> {
    if !is_file {
        if !selected.is_dir() {
            return Err(WorkerError::InvalidPayload(
                "selected text encoder is not a directory".to_owned(),
            ));
        }
        return capture_registered_files(selected).map(|files| (selected.to_path_buf(), files));
    }
    if !selected.is_file() {
        return Err(WorkerError::InvalidPayload(
            "selected text encoder is not a file".to_owned(),
        ));
    }
    let parent = selected.parent().ok_or_else(|| {
        WorkerError::InvalidPayload("selected text encoder has no parent".to_owned())
    })?;
    let mut files = capture_files(selected)?;
    files.extend(capture_files(&parent.join("config.json"))?);
    files.sort();
    files.dedup();
    Ok((parent.to_path_buf(), files))
}

/// Rebind an API-selected encoder by its frozen file ledger, never by the API host's path hash.
pub(crate) fn registered_text_encoder_path(
    settings: &Settings,
    model_id: &str,
    is_file: bool,
    expected: &[PortableArtifactFile],
) -> WorkerResult<Option<(PathBuf, PathBuf)>> {
    for (candidate_model, path, candidate_file) in
        crate::text_encoder_selection::portable_registered_source_paths(settings)
    {
        if candidate_model != model_id || candidate_file != is_file {
            continue;
        }
        let Ok((root, files)) = selected_text_encoder_root_and_files(&path, is_file) else {
            continue;
        };
        if files == expected {
            return Ok(Some((root, path)));
        }
    }
    Ok(None)
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
    let roots = crate::paths::allowed_model_roots(settings).unwrap_or_default();
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
                    capture_registered_files_with_roots(&path, &roots)
                        .ok()
                        .inspect(|files| {
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
    for lora in catalog_records(&manifests.join("builtin.loras.jsonc"), "loras") {
        let Some(id) = lora.get("id").and_then(serde_json::Value::as_str) else {
            continue;
        };
        for path in builtin_lora_source_paths(settings, &lora) {
            append_registered_claim(settings, &mut claims, "lora", id, None, &path);
            for claim in claims.drain(..) {
                publish(claim);
            }
        }
    }
    let mut lora_manifests = vec![manifests.join("user.loras.jsonc")];
    lora_manifests.extend(
        registered_projects(settings)
            .into_iter()
            .map(|project| PathBuf::from(project.path).join("loras/manifest.jsonc")),
    );
    for manifest in lora_manifests {
        for lora in catalog_records(&manifest, "loras") {
            let Some(id) = lora.get("id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(path) = sceneworks_core::lora_family::adapter_record_path(&lora) else {
                continue;
            };
            if let Ok(path) = normalize_catalog_lora_path(settings, &manifest, &path) {
                append_registered_claim(settings, &mut claims, "lora", id, None, &path);
            }
            for claim in claims.drain(..) {
                publish(claim);
            }
        }
    }
    for project in registered_projects(settings) {
        let project_path = Path::new(&project.path);
        let characters =
            sceneworks_core::character_store::CharacterStore::new(&settings.data_dir, project_path);
        let Ok(characters) = characters.list_characters(&project.id, true) else {
            continue;
        };
        for character in characters {
            for link in character
                .get("loras")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
            {
                let Some(id) = link.get("id").and_then(serde_json::Value::as_str) else {
                    continue;
                };
                if let Some(path) = character_lora_copy_path(settings, project_path, link) {
                    append_registered_claim(settings, &mut claims, "lora", id, None, &path);
                }
                for claim in claims.drain(..) {
                    publish(claim);
                }
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
            append_registered_claim(settings, &mut claims, "control", id, None, &path);
            for claim in claims.drain(..) {
                publish(claim);
            }
        }
    }
}

fn character_sidecar_paths(project_path: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(project_path.join("characters")) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.ends_with(sceneworks_core::character_store::CHARACTER_SIDECAR_PATTERN)
                })
        })
        .collect()
}

fn character_lora_copy_path(
    settings: &Settings,
    project_path: &Path,
    link: &serde_json::Value,
) -> Option<PathBuf> {
    let id = link.get("id")?.as_str()?;
    if !id.starts_with("character_lora_") || link.get("category")?.as_str()? != "character" {
        return None;
    }
    let relative = link.get("projectPath")?.as_str()?;
    let within_loras = relative.strip_prefix("loras/")?;
    let selected = sceneworks_core::safe_path::confine_under(
        &project_path.join("loras"),
        within_loras,
        sceneworks_core::safe_path::Confinement::SymlinkSafe,
    )
    .ok()?;
    crate::paths::normalize_app_managed_lora_path(settings, &selected).ok()
}

pub(crate) fn registered_character_lora_path(
    settings: &Settings,
    project_path: &Path,
    project_id: &str,
    character_id: &str,
    link_id: &str,
    frozen_files: &[PortableArtifactFile],
) -> WorkerResult<Option<PathBuf>> {
    if !link_id.starts_with("character_lora_") {
        return Ok(None);
    }
    let store =
        sceneworks_core::character_store::CharacterStore::new(&settings.data_dir, project_path);
    let Ok(character) = store.get_character(project_id, character_id) else {
        return Ok(None);
    };
    for link in character
        .get("loras")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|link| link.get("id").and_then(serde_json::Value::as_str) == Some(link_id))
    {
        let Some(path) = character_lora_copy_path(settings, project_path, link) else {
            continue;
        };
        if capture_registered_files(&path).is_ok_and(|files| files == frozen_files) {
            return Ok(Some(path));
        }
    }
    Ok(None)
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

/// The builtin catalog's model download receipt paths (SW-7030): one
/// `data/models/<safe_download_dir(repo)>/.sceneworks-download-complete.json` per
/// `downloads[].repo` the embedded builtin manifest declares. Model Manager enqueues exactly that
/// targetDir for each builtin `downloads[]` entry, and `run_model_download_job` writes its
/// completion receipt there, so watching these markers detects a builtin auxiliary
/// (`controlnet_tile_sdxl`, …) installed after the initial inventory scan. Pure metadata-only
/// path construction — the multi-GiB weights these receipts stand in for are never hashed.
fn builtin_model_download_receipt_paths(settings: &Settings) -> Vec<PathBuf> {
    let Ok(manifest) = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
    else {
        return Vec::new();
    };
    manifest
        .get("models")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("downloads").and_then(serde_json::Value::as_array))
        .flatten()
        .filter_map(|download| download.get("repo").and_then(serde_json::Value::as_str))
        .map(|repo| {
            settings
                .data_dir
                .join("models")
                .join(crate::paths::safe_download_dir(repo))
                .join(".sceneworks-download-complete.json")
        })
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
    manifests.extend(
        registered_projects(settings)
            .into_iter()
            .map(|project| PathBuf::from(project.path).join("control-overlays/manifest.jsonc")),
    );
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
    // A model manifest goes through the shared version normalizer: an unknown future version is
    // refused (nothing advertised from it) instead of being read as the current format.
    let parsed = if collection == "models" {
        sceneworks_core::model_manifest_version::parse_user_model_manifest(&raw).ok()
    } else {
        serde_json::from_str::<serde_json::Value>(&sceneworks_core::jsonc::strip_jsonc_comments(
            &raw,
        ))
        .ok()
    };
    let Some(value) = parsed else {
        return Vec::new();
    };
    value
        .get(collection)
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The built-in LoRA catalog names pinned Hugging Face files rather than local paths. Resolve
/// only its pinned snapshot and file; never use a job's API-host path.
fn builtin_lora_source_paths(settings: &Settings, entry: &serde_json::Value) -> Vec<PathBuf> {
    let Some(id) = entry.get("id").and_then(serde_json::Value::as_str) else {
        return Vec::new();
    };
    let Some(source) = entry.get("source") else {
        return Vec::new();
    };
    if source.get("provider").and_then(serde_json::Value::as_str) != Some("huggingface") {
        return Vec::new();
    }
    let Some(repo) = source.get("repo").and_then(serde_json::Value::as_str) else {
        return Vec::new();
    };
    let Some(revision) = source.get("revision").and_then(serde_json::Value::as_str) else {
        return Vec::new();
    };
    let Some(file) = source.get("file").and_then(serde_json::Value::as_str) else {
        return Vec::new();
    };
    if revision.len() != 40
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        || sceneworks_core::model_artifacts::ArtifactFile::new(file).is_err()
    {
        return Vec::new();
    }
    let library = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(library.clone());
    let repository_root = library.repository_root(repo).ok();
    let mut paths = Vec::new();
    let mut push = |snapshot: &Path, file: &str| {
        if sceneworks_core::model_artifacts::ArtifactFile::new(file).is_err() {
            return;
        }
        let candidate = snapshot.join(file);
        if candidate.extension().and_then(|value| value.to_str()) != Some("safetensors") {
            return;
        }
        let Ok(resolved) = candidate.canonicalize() else {
            return;
        };
        // Standard HF snapshots symlink into their repository's `blobs/` sibling. A leased
        // local-tier snapshot may live elsewhere; in that case confine to that exact root.
        let allowed = repository_root
            .as_deref()
            .filter(|root| snapshot.starts_with(root))
            .unwrap_or(snapshot);
        let Ok(allowed) = allowed.canonicalize() else {
            return;
        };
        if resolved.starts_with(&allowed) && resolved.is_file() && !paths.contains(&candidate) {
            paths.push(candidate);
        }
    };
    if let Ok((_, snapshot)) = resolver.discover_source_reference(repo, revision) {
        push(&snapshot, file);
    }
    let marker = settings
        .data_dir
        .join("loras")
        .join(crate::paths::safe_download_dir(id))
        .join(".sceneworks-download-complete.json");
    if let Ok(bytes) = std::fs::read(marker) {
        if let Ok(receipt) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            let rows = receipt
                .get("receipts")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_else(|| vec![receipt]);
            for row in rows.iter().rev() {
                if row.get("repo").and_then(serde_json::Value::as_str) != Some(repo)
                    || row
                        .get("snapshotRevision")
                        .and_then(serde_json::Value::as_str)
                        != Some(revision)
                {
                    continue;
                }
                let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision))
                else {
                    continue;
                };
                if let Some(files) = row
                    .get("resolvedFiles")
                    .and_then(serde_json::Value::as_array)
                {
                    if files
                        .iter()
                        .any(|candidate| candidate.as_str() == Some(file))
                    {
                        push(&snapshot, file);
                    }
                }
            }
        }
    }
    paths
}

/// Rebind a selected built-in LoRA under this worker's own catalog and exact local byte ledger.
pub(crate) fn registered_builtin_lora_path(
    settings: &Settings,
    id: &str,
    expected: &[PortableArtifactFile],
) -> Option<PathBuf> {
    let manifest = settings.config_dir.join("manifests/builtin.loras.jsonc");
    let matches = catalog_records(&manifest, "loras")
        .into_iter()
        .filter(|entry| entry.get("id").and_then(serde_json::Value::as_str) == Some(id))
        .collect::<Vec<_>>();
    let [entry] = matches.as_slice() else {
        return None;
    };
    builtin_lora_source_paths(settings, entry)
        .into_iter()
        .find(|path| capture_registered_files(path).is_ok_and(|files| files == expected))
}

fn append_registered_claim(
    settings: &Settings,
    claims: &mut Vec<PortableSourceClaim>,
    kind: &str,
    model_id: &str,
    component_id: Option<&str>,
    selected: &Path,
) {
    let roots = crate::paths::allowed_model_roots(settings).unwrap_or_default();
    let Ok(files) = capture_registered_files_with_roots(selected, &roots) else {
        return;
    };
    append_registered_claim_files(claims, kind, model_id, component_id, files);
}

pub(crate) fn capture_registered_files(selected: &Path) -> WorkerResult<Vec<PortableArtifactFile>> {
    capture_registered_files_with_roots(selected, &[])
}

pub(crate) fn capture_registered_files_with_roots(
    selected: &Path,
    allowed_roots: &[PathBuf],
) -> WorkerResult<Vec<PortableArtifactFile>> {
    let mut files = capture_files_with_roots(selected, allowed_roots)?;
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
    let library = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
    let Ok(repos) = std::fs::read_dir(library.root()) else {
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
        let Ok(snapshots) = library.snapshots_root_from_repository_root(&repo.path()) else {
            continue;
        };
        let Ok(revisions) = std::fs::read_dir(snapshots) else {
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
        let source_library = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
        let library = sceneworks_core::model_artifacts::ArtifactSourceLibrary::new(
            source_library.root().to_path_buf(),
        )
        .map_err(|_| WorkerError::InvalidPayload("conversion source root is invalid".to_owned()))?;
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
    let roots = crate::paths::allowed_model_roots(settings).unwrap_or_default();
    let files = capture_files_with_roots(&selected, &roots)?;
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
    let canonical_root = std::fs::canonicalize(selected).map_err(WorkerError::Io)?;
    for component in ["vae", "text_encoder", "tokenizer"] {
        let allowed = vec![canonical_root.clone()];
        let mut visited = std::collections::HashSet::from([canonical_root.clone()]);
        capture_tree(
            selected,
            &selected.join(component),
            &allowed,
            &mut visited,
            &mut files,
            0,
        )?;
    }
    files.sort();
    if files.is_empty() {
        return Ok(None);
    }
    Ok(Some(files))
}

fn capture_files(selected: &Path) -> WorkerResult<Vec<PortableArtifactFile>> {
    capture_files_with_roots(selected, &[])
}

/// Capture a registered source's file ledger, following directory links the
/// way API admission capture does (SW-7030, review 4533400138): a linked
/// directory (e.g. `text_encoder/` pointing at shared components elsewhere
/// under the data dir) keeps its logical relative path, while confinement and
/// hashing resolve through the canonical target. A link whose target escapes
/// every allowed root — the canonical tree root plus any caller-supplied
/// managed roots — is refused, and canonical directories are visited once so
/// link cycles cannot recurse forever.
fn capture_files_with_roots(
    selected: &Path,
    allowed_roots: &[PathBuf],
) -> WorkerResult<Vec<PortableArtifactFile>> {
    let mut found = Vec::new();
    if selected.is_file() {
        let name = selected.file_name().ok_or_else(|| {
            WorkerError::InvalidPayload("auxiliary source has no file name".to_owned())
        })?;
        capture_one(selected, PathBuf::from(name), &mut found)?;
    } else if selected.is_dir() {
        let canonical_root = std::fs::canonicalize(selected).map_err(WorkerError::Io)?;
        let mut allowed: Vec<PathBuf> = allowed_roots
            .iter()
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .collect();
        allowed.push(canonical_root.clone());
        let mut visited = std::collections::HashSet::from([canonical_root]);
        capture_tree(selected, selected, &allowed, &mut visited, &mut found, 0)?;
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

const MAX_CAPTURE_DEPTH: usize = 16;

fn capture_tree(
    root: &Path,
    current: &Path,
    allowed: &[PathBuf],
    visited: &mut std::collections::HashSet<PathBuf>,
    found: &mut Vec<PortableArtifactFile>,
    depth: usize,
) -> WorkerResult<()> {
    if depth > MAX_CAPTURE_DEPTH {
        return Err(WorkerError::InvalidPayload(
            "auxiliary source tree is too deep".to_owned(),
        ));
    }
    let canonical_current = std::fs::canonicalize(current).map_err(WorkerError::Io)?;
    for entry in std::fs::read_dir(&canonical_current).map_err(WorkerError::Io)? {
        let entry = entry.map_err(WorkerError::Io)?;
        let path = current.join(entry.file_name());
        let canonical = std::fs::canonicalize(&path).map_err(WorkerError::Io)?;
        if canonical.is_dir() {
            if !allowed.iter().any(|root| canonical.starts_with(root)) {
                return Err(WorkerError::InvalidPayload(
                    "auxiliary source link escapes its managed roots".to_owned(),
                ));
            }
            if visited.insert(canonical) {
                capture_tree(root, &path, allowed, visited, found, depth + 1)?;
            }
        } else if canonical.is_file() {
            let relative = path
                .strip_prefix(root)
                .expect("traversed path is inside root")
                .to_path_buf();
            capture_one(&canonical, relative, found)?;
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

    /// Review finding 7: a read-only worker consumer of `user.models.jsonc` goes through the shared
    /// version normalizer, so a future-version file advertises nothing instead of being read as
    /// the current format, while an old v1 (or header-less) file still loads.
    #[test]
    fn user_model_manifest_readers_refuse_future_versions() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("user.models.jsonc");
        std::fs::write(
            &path,
            json!({"schemaVersion":3,"models":[{"id":"later"}]}).to_string(),
        )
        .unwrap();
        assert!(catalog_records(&path, "models").is_empty());
        std::fs::write(&path, json!({"models":[{"id":"older"}]}).to_string()).unwrap();
        assert_eq!(catalog_records(&path, "models").len(), 1);
        // Non-model collections keep their own format.
        std::fs::write(
            &path,
            json!({"schemaVersion":3,"loras":[{"id":"style"}]}).to_string(),
        )
        .unwrap();
        assert_eq!(catalog_records(&path, "loras").len(), 1);
    }

    #[test]
    fn catalog_hash_does_not_initialize_training_directories_without_projects() {
        let temp = tempfile::tempdir().unwrap();
        let settings = Settings::for_test(temp.path().join("worker-data"));
        assert!(!settings.data_dir.join("loras").exists());
        let _ = catalog_hash(&settings);
        assert!(!settings.data_dir.join("loras").exists());
    }

    #[test]
    fn catalog_hash_preserves_registered_projects_without_initializing_training_directories() {
        let temp = tempfile::tempdir().unwrap();
        let settings = Settings::for_test(temp.path().join("worker-data"));
        let store = sceneworks_core::project_store::ProjectStore::new(&settings.data_dir, "worker");
        let created = store.create_project("Existing").unwrap();
        std::fs::remove_dir(settings.data_dir.join("loras")).unwrap();
        assert_eq!(registered_projects(&settings), vec![created]);
        let _ = catalog_hash(&settings);
        assert!(!settings.data_dir.join("loras").exists());
    }

    #[test]
    fn app_managed_probe_finds_cross_root_bytes_and_refuses_substitution() {
        let temp = tempfile::tempdir().unwrap();
        let settings = Settings::for_test(temp.path().join("worker-data"));
        let selected = settings.data_dir.join("different/tree/clip");
        std::fs::create_dir_all(&selected).unwrap();
        std::fs::write(selected.join("model.safetensors"), b"frozen clip bytes").unwrap();
        let expected = capture_registered_files(&selected).unwrap();
        let api_path = "/api-host/another/root/clip";
        assert_eq!(
            selected_app_managed_exact_source(&settings, api_path, &expected).unwrap(),
            Some(selected.canonicalize().unwrap())
        );
        let challenge = PortableSourceChallenge {
            id: "test-challenge".to_owned(),
            source_id: "test_app_managed_clip".to_owned(),
            files: expected.clone(),
        };
        prove_app_managed_challenges(&settings, vec![challenge]).unwrap();
        assert!(pending_inventory_claims(&settings.worker_id)
            .unwrap()
            .1
            .iter()
            .any(|claim| claim.source_id == "test_app_managed_clip" && claim.files == expected));

        let substituted = settings.data_dir.join("different/tree/substituted");
        std::fs::create_dir_all(&substituted).unwrap();
        std::fs::write(substituted.join("model.safetensors"), b"other clip bytes").unwrap();
        assert!(selected_app_managed_exact_source(
            &settings,
            substituted.to_str().unwrap(),
            &expected,
        )
        .is_err());
        std::fs::remove_dir_all(&selected).unwrap();
        assert!(find_app_managed_exact_source(&settings, &expected)
            .unwrap()
            .is_none());
        #[cfg(unix)]
        {
            let escaped = settings.data_dir.join("escaped");
            std::os::unix::fs::symlink(temp.path().join("outside"), &escaped).unwrap();
            std::fs::create_dir_all(temp.path().join("outside")).unwrap();
            std::fs::write(
                temp.path().join("outside/model.safetensors"),
                b"frozen clip bytes",
            )
            .unwrap();
            assert!(find_app_managed_exact_source(&settings, &expected)
                .unwrap()
                .is_none());
        }
    }

    #[test]
    fn proved_utility_source_survives_busy_queue_and_rechecks_bytes_before_load() {
        let temp = tempfile::tempdir().unwrap();
        let settings = Settings::for_test(temp.path().join("worker-data"));
        let selected = settings.data_dir.join("models/utility");
        std::fs::create_dir_all(&selected).unwrap();
        let weights = selected.join("model.safetensors");
        std::fs::write(&weights, b"source").unwrap();
        let files = capture_registered_files(&selected).unwrap();
        let claim = PortableSourceClaim::new(
            "test_queued_utility".to_owned(),
            PortableSourceProvenance::OperatorOverride,
            files.clone(),
        )
        .unwrap();
        let mut inventory = InventoryProgress::default();
        inventory.publish_probe(claim.clone());
        let generation = inventory.generation;
        inventory.acknowledge(&settings.worker_id, generation);
        // The worker remains busy beyond the old 60-second probe lifetime, and an ordinary
        // inventory sweep does not discover this uncataloged app-managed source.
        inventory.probed.get_mut(&claim.source_id).unwrap().1 =
            Instant::now() - Duration::from_secs(61);
        inventory.replace_scanned(Vec::new());
        assert!(inventory.claims.contains(&claim));
        inventory.reset_worker(&settings.worker_id);
        assert!(inventory
            .pending(&settings.worker_id)
            .unwrap()
            .1
            .contains(&claim));

        // A retained scheduling claim does not authorize a changed checkpoint at the loader.
        std::fs::write(&weights, b"mutate").unwrap();
        assert!(
            selected_app_managed_exact_source(&settings, selected.to_str().unwrap(), &files,)
                .is_err()
        );
    }

    #[test]
    fn builtin_hf_lora_is_advertised_and_rebound_by_exact_worker_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let mut settings = Settings::for_test(temp.path().join("worker-data"));
        settings.config_dir = temp.path().join("worker-config");
        let hub = temp.path().join("worker-hub");
        let _hub = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            settings.data_dir.clone(),
            hub.clone(),
        );
        let manifest = settings.config_dir.join("manifests/builtin.loras.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        let repo = "example/pinned-lora";
        let revision = "1111111111111111111111111111111111111111";
        let filename = "adapter.safetensors";
        std::fs::write(&manifest, json!({"loras":[{"id":"builtin_adapter","source":{"provider":"huggingface","repo":repo,"revision":revision,"file":filename}}]}).to_string()).unwrap();
        let worker_file = hub
            .join("models--example--pinned-lora/snapshots")
            .join(revision)
            .join(filename);
        std::fs::create_dir_all(worker_file.parent().unwrap()).unwrap();
        std::fs::write(&worker_file, b"exact adapter bytes").unwrap();
        let api_file = temp.path().join("api-root/adapter.safetensors");
        std::fs::create_dir_all(api_file.parent().unwrap()).unwrap();
        std::fs::write(&api_file, b"exact adapter bytes").unwrap();
        let expected = capture_registered_files(&api_file).unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &expected,
            )
            .unwrap();
        let id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "lora",
            "builtin_adapter",
            None,
            &ledger,
        )
        .unwrap();
        assert!(static_source_claims(&settings)
            .iter()
            .any(|claim| claim.source_id == id && claim.files == expected));
        assert_eq!(
            registered_builtin_lora_path(&settings, "builtin_adapter", &expected),
            Some(worker_file.clone())
        );
        assert!(registered_builtin_lora_path(&settings, "other_adapter", &expected).is_none());
        std::fs::write(&worker_file, b"replaced adapter bytes").unwrap();
        assert!(registered_builtin_lora_path(&settings, "builtin_adapter", &expected).is_none());
        #[cfg(unix)]
        {
            std::fs::remove_file(&worker_file).unwrap();
            let outside = temp.path().join("outside.safetensors");
            std::fs::write(&outside, b"exact adapter bytes").unwrap();
            std::os::unix::fs::symlink(&outside, &worker_file).unwrap();
            assert!(
                registered_builtin_lora_path(&settings, "builtin_adapter", &expected).is_none()
            );
            assert!(!static_source_claims(&settings)
                .iter()
                .any(|claim| claim.source_id == id));
        }
    }

    #[test]
    fn builtin_hf_lora_receipt_cannot_select_another_revision_or_file() {
        let temp = tempfile::tempdir().unwrap();
        let mut settings = Settings::for_test(temp.path().join("worker-data"));
        settings.config_dir = temp.path().join("worker-config");
        let hub = temp.path().join("worker-hub");
        let _hub = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            settings.data_dir.clone(),
            hub.clone(),
        );
        let repo = "example/pinned-lora";
        let revision = "1111111111111111111111111111111111111111";
        let stale_revision = "2222222222222222222222222222222222222222";
        let manifest = settings.config_dir.join("manifests/builtin.loras.jsonc");
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, json!({"loras":[{"id":"pinned","source":{
            "provider":"huggingface","repo":repo,"revision":revision,"file":"selected.safetensors"
        }}]}).to_string()).unwrap();
        let snapshot_root = hub.join("models--example--pinned-lora/snapshots");
        let selected = snapshot_root.join(revision).join("selected.safetensors");
        let alternate = snapshot_root.join(revision).join("alternate.safetensors");
        let stale = snapshot_root
            .join(stale_revision)
            .join("selected.safetensors");
        for path in [&selected, &alternate, &stale] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"same bytes").unwrap();
        }
        let receipt = settings
            .data_dir
            .join("loras/pinned/.sceneworks-download-complete.json");
        std::fs::create_dir_all(receipt.parent().unwrap()).unwrap();
        let entry = catalog_records(&manifest, "loras").remove(0);
        let write_receipt = |snapshot_revision: &str, file: &str| {
            std::fs::write(
                &receipt,
                json!({"repo":repo,
                    "snapshotRevision":snapshot_revision,"resolvedFiles":[file]
                })
                .to_string(),
            )
            .unwrap();
        };
        // Remove the direct pinned file so the receipt selector is the only candidate.
        std::fs::remove_file(&selected).unwrap();
        write_receipt(stale_revision, "selected.safetensors");
        assert!(builtin_lora_source_paths(&settings, &entry).is_empty());
        write_receipt(revision, "alternate.safetensors");
        assert!(builtin_lora_source_paths(&settings, &entry).is_empty());
        std::fs::write(&selected, b"same bytes").unwrap();
        write_receipt(revision, "selected.safetensors");
        assert_eq!(builtin_lora_source_paths(&settings, &entry), vec![selected]);
    }

    #[test]
    fn file_selected_encoder_ledger_includes_its_behavior_config() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("encoder");
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("model.safetensors");
        std::fs::write(&file, b"weights").unwrap();
        std::fs::write(root.join("config.json"), b"config").unwrap();
        let (selected_root, before) = selected_text_encoder_root_and_files(&file, true).unwrap();
        assert_eq!(selected_root, root);
        assert_eq!(
            before
                .iter()
                .map(|file| file.relative_path.as_path())
                .collect::<Vec<_>>(),
            vec![Path::new("config.json"), Path::new("model.safetensors")]
        );
        std::fs::write(root.join("config.json"), b"changed").unwrap();
        assert_ne!(
            selected_text_encoder_root_and_files(&file, true).unwrap().1,
            before
        );
    }

    #[cfg(native_lane)]
    #[test]
    fn selected_encoder_claim_rebinds_equal_bytes_across_roots_and_refuses_tamper() {
        let temp = tempfile::tempdir().unwrap();
        let api_source = temp.path().join("api/encoder");
        let worker_source = temp.path().join("worker/encoder");
        let production = crate::inference_runtime::media_encoder_contract("qwen_image").unwrap();
        let bounded = gen_core::EncoderContract {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            output_width: 64,
            loaded_hidden_layers: 1,
            max_position_embeddings: 512,
            mrope_section: &[4, 6, 6],
            selected_hidden_layers: &[1],
            ..production
        };
        let _contract =
            crate::inference_runtime::scoped_test_media_encoder_contract(&["qwen_image"], bounded);
        gen_core_testkit::write_encoder_contract_fixture(&api_source, bounded).unwrap();
        gen_core_testkit::write_encoder_contract_fixture(&worker_source, bounded).unwrap();
        let api_option = crate::text_encoder_selection::opaque_id(&gen_core::WeightsSource::Dir(
            api_source.clone(),
        ))
        .unwrap();
        let worker_option = crate::text_encoder_selection::opaque_id(
            &gen_core::WeightsSource::Dir(worker_source.clone()),
        )
        .unwrap();
        assert_ne!(
            api_option, worker_option,
            "host paths must not enter the content claim"
        );
        let mut settings = Settings::for_test(temp.path().join("worker-data"));
        let _cache_root = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            settings.data_dir.clone(),
            settings.data_dir.join("cache/huggingface/hub"),
        );
        settings.config_dir = temp.path().join("worker-config");
        settings.external_model_roots = vec![temp.path().join("worker")];
        std::fs::create_dir_all(settings.config_dir.join("manifests")).unwrap();
        std::fs::write(
            settings.config_dir.join("manifests/user.models.jsonc"),
            json!({"models":[{"id":"qwen_image","installedPath":worker_source}]}).to_string(),
        )
        .unwrap();
        let (_, frozen) = selected_text_encoder_root_and_files(&api_source, false).unwrap();
        let (selected_root, selected_path) =
            registered_text_encoder_path(&settings, "qwen_image", false, &frozen)
                .unwrap()
                .expect("same ledger under worker root is available");
        assert_eq!(
            selected_root,
            std::fs::canonicalize(&worker_source).unwrap()
        );
        assert_eq!(
            selected_path,
            std::fs::canonicalize(&worker_source).unwrap()
        );
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &frozen,
            )
            .unwrap();
        let claim_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "text-encoder",
            "qwen_image:directory",
            None,
            &ledger,
        )
        .unwrap();
        assert!(static_source_claims(&settings)
            .iter()
            .any(|claim| claim.source_id == claim_id && claim.files == frozen));
        std::fs::write(worker_source.join("config.json"), b"changed").unwrap();
        assert!(
            registered_text_encoder_path(&settings, "qwen_image", false, &frozen)
                .unwrap()
                .is_none()
        );
    }

    #[cfg(all(native_lane, target_os = "windows"))]
    #[test]
    fn external_qwen_encoder_claim_rebinds_across_roots_only_when_registered() {
        let temp = tempfile::tempdir().unwrap();
        let api_source = temp.path().join("api/text_encoders/encoder");
        let worker_root = temp.path().join("worker");
        let worker_source = worker_root.join("text_encoders/encoder");
        let production = crate::inference_runtime::media_encoder_contract("qwen_image").unwrap();
        let bounded = gen_core::EncoderContract {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            output_width: 64,
            loaded_hidden_layers: 1,
            max_position_embeddings: 512,
            mrope_section: &[4, 6, 6],
            selected_hidden_layers: &[1],
            ..production
        };
        let _contract =
            crate::inference_runtime::scoped_test_media_encoder_contract(&["qwen_image"], bounded);
        gen_core_testkit::write_encoder_contract_fixture(&api_source, bounded).unwrap();
        gen_core_testkit::write_encoder_contract_fixture(&worker_source, bounded).unwrap();
        let api_id = crate::text_encoder_selection::opaque_id(&gen_core::WeightsSource::Dir(
            api_source.clone(),
        ))
        .unwrap();
        let worker_id = crate::text_encoder_selection::opaque_id(&gen_core::WeightsSource::Dir(
            worker_source.clone(),
        ))
        .unwrap();
        assert_ne!(api_id, worker_id);
        let mut settings = Settings::for_test(temp.path().join("worker-data"));
        let _cache_root = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            settings.data_dir.clone(),
            settings.data_dir.join("cache/huggingface/hub"),
        );
        settings.external_model_roots = vec![worker_root];
        let (_, frozen) = selected_text_encoder_root_and_files(&api_source, false).unwrap();
        let model_key = "external-comfyui:qwen-image";
        assert!(
            registered_text_encoder_path(&settings, model_key, false, &frozen)
                .unwrap()
                .is_some()
        );
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &frozen,
            )
            .unwrap();
        let claim_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "text-encoder",
            &format!("{model_key}:directory"),
            None,
            &ledger,
        )
        .unwrap();
        assert!(static_source_claims(&settings)
            .iter()
            .any(|claim| claim.source_id == claim_id && claim.files == frozen));
        settings.external_model_roots.clear();
        assert!(
            registered_text_encoder_path(&settings, model_key, false, &frozen)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn character_sidecar_lora_is_advertised_and_rebound_to_its_project_copy() {
        use sceneworks_core::character_store::{
            CharacterCreateInput, CharacterLoraInput, CharacterStore,
        };
        use sceneworks_core::project_store::ProjectStore;

        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let projects = ProjectStore::new(data.clone(), "worker");
        let project = projects.create_project("Character claim").unwrap();
        let project_path = PathBuf::from(&project.path);
        let characters = CharacterStore::new(&data, &project_path);
        let character = characters
            .create_character(
                &project.id,
                CharacterCreateInput {
                    name: "Mira".to_owned(),
                    character_type: "person".to_owned(),
                    description: String::new(),
                },
            )
            .unwrap();
        let character_id = character["id"].as_str().unwrap();
        let before = catalog_hash(&settings);
        let source = data.join("loras/mira.safetensors");
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::write(&source, b"original weights").unwrap();
        let attached = characters
            .attach_lora(
                &project.id,
                character_id,
                CharacterLoraInput {
                    lora_id: None,
                    name: "Mira LoRA".to_owned(),
                    source_path: Some(source.to_string_lossy().into_owned()),
                    trigger_words: Vec::new(),
                    default_weight: 1.0,
                    compatibility: serde_json::Map::new(),
                    scope: "project".to_owned(),
                },
            )
            .unwrap();
        assert_ne!(
            catalog_hash(&settings),
            before,
            "sidecar edit schedules a rescan"
        );
        let link = &attached["loras"][0];
        let link_id = link["id"].as_str().unwrap();
        let copy = project_path.join(link["projectPath"].as_str().unwrap());
        let canonical_copy = std::fs::canonicalize(&copy).unwrap();
        let files = capture_registered_files(&copy).unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(&files)
                .unwrap();
        let expected_id = sceneworks_core::portable_sources::registered_digest_claim_id(
            "lora", link_id, None, &ledger,
        )
        .unwrap();
        let mut claims = Vec::new();
        registered_digest_source_claims(&settings, |claim| claims.push(claim));
        assert!(claims
            .iter()
            .any(|claim| claim.source_id == expected_id && claim.files == files));
        assert_eq!(
            registered_character_lora_path(
                &settings,
                &project_path,
                &project.id,
                character_id,
                link_id,
                &files,
            )
            .unwrap(),
            Some(canonical_copy.clone())
        );
        std::fs::write(&source, b"changed original").unwrap();
        assert_eq!(
            registered_character_lora_path(
                &settings,
                &project_path,
                &project.id,
                character_id,
                link_id,
                &files,
            )
            .unwrap(),
            Some(canonical_copy),
            "the detached original cannot replace the project copy"
        );
        std::fs::write(&copy, b"changed project copy").unwrap();
        assert!(registered_character_lora_path(
            &settings,
            &project_path,
            &project.id,
            character_id,
            link_id,
            &files,
        )
        .unwrap()
        .is_none());
    }

    #[cfg(native_lane)]
    #[test]
    fn complete_ltx_training_gemma_override_is_advertised() {
        let temp = tempfile::tempdir().unwrap();
        let gemma = temp.path().join("gemma");
        std::fs::create_dir(&gemma).unwrap();
        std::fs::write(
            gemma.join("config.json"),
            br#"{"model_type":"gemma3_text"}"#,
        )
        .unwrap();
        std::fs::write(gemma.join("tokenizer.json"), br#"{"model":{"type":"BPE"}}"#).unwrap();
        let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut weights = (header.len() as u64).to_le_bytes().to_vec();
        weights.extend_from_slice(header);
        weights.push(7);
        std::fs::write(gemma.join("model.safetensors"), weights).unwrap();
        let settings = Settings::for_test(temp.path().join("data"));

        let hub = temp.path().join("hub");
        std::fs::create_dir(&hub).unwrap();
        crate::test_env::temp_env_vars(
            &[
                ("LTX_GEMMA_DIR", gemma.to_str().unwrap()),
                ("HF_HUB_CACHE", hub.to_str().unwrap()),
            ],
            || {
                let advertised = static_source_claims(&settings)
                    .into_iter()
                    .find(|claim| claim.source_id == "ltx_training_gemma_override")
                    .expect("complete operator Gemma must be advertised for portable training");
                assert_eq!(
                    advertised.provenance,
                    PortableSourceProvenance::OperatorOverride
                );
                assert_eq!(advertised.files.len(), 3);
            },
        );
    }

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
        let (first_generation, first_claims) = progress.pending("worker-a").unwrap();
        assert_eq!(first_claims.len(), 1);
        progress.publish(make("second"));
        progress.acknowledge("worker-a", first_generation);
        let (second_generation, second_claims) = progress.pending("worker-a").unwrap();
        assert_eq!(second_claims.len(), 2);
        progress.acknowledge("worker-a", second_generation);
        assert!(progress.pending("worker-a").is_none());
        progress.publish(make("second"));
        assert!(progress.pending("worker-a").is_none());
    }

    #[test]
    fn sibling_worker_must_publish_the_same_inventory_generation_independently() {
        let claim = PortableSourceClaim::new(
            "shared-source",
            PortableSourceProvenance::OperatorOverride,
            vec![PortableArtifactFile {
                relative_path: PathBuf::from("model.safetensors"),
                size_bytes: 1,
                sha256: format!("sha256:{}", "a".repeat(64)),
            }],
        )
        .unwrap();
        let mut progress = InventoryProgress::default();
        progress.publish(claim.clone());
        let (generation, first_claims) = progress.pending("worker-a").unwrap();
        assert_eq!(first_claims, vec![claim.clone()]);
        progress.acknowledge("worker-a", generation);
        assert!(progress.pending("worker-a").is_none());
        assert_eq!(
            progress.pending("worker-b"),
            Some((generation, vec![claim.clone()])),
            "worker A's heartbeat must not consume worker B's advertisement"
        );
        progress.acknowledge("worker-b", generation);
        assert!(progress.pending("worker-b").is_none());

        // A re-registration clears only A's server vector; B remains caught up. The next
        // completed rescan, including a withdrawal, must then reach both workers.
        progress.reset_worker("worker-a");
        assert_eq!(
            progress.pending("worker-a"),
            Some((generation, vec![claim]))
        );
        assert!(progress.pending("worker-b").is_none());
        progress.replace_scanned(Vec::new());
        let (withdrawal, claims) = progress.pending("worker-b").unwrap();
        assert!(claims.is_empty());
        progress.acknowledge("worker-b", withdrawal);
        assert_eq!(progress.pending("worker-a"), Some((withdrawal, Vec::new())));
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
        progress.acknowledge("worker-a", progress.pending("worker-a").unwrap().0);
        progress.publish(make("new", 'b'));
        assert_eq!(progress.pending("worker-a").unwrap().1.len(), 2);
        progress.publish(make("removed", 'c'));
        assert_eq!(
            progress.pending("worker-a").unwrap().1[1].files[0].sha256,
            format!("sha256:{}", "c".repeat(64))
        );
        progress.replace_scanned(vec![make("new", 'b')]);
        let (generation, claims) = progress.pending("worker-a").unwrap();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].source_id, "new");
        progress.acknowledge("worker-a", generation);
        assert!(progress.pending("worker-a").is_none());

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
            .pending("worker-a")
            .expect("new trained source is advertised");
        assert_eq!(claims.len(), 1);
        assert_eq!(
            claims[0].files,
            capture_registered_files(adapter.parent().unwrap()).unwrap()
        );
        progress.acknowledge("worker-a", generation);

        std::fs::write(&manifest, br#"{"loras":[]}"#).unwrap();
        let mut rescanned = Vec::new();
        registered_digest_source_claims(&settings, |claim| rescanned.push(claim));
        progress.replace_scanned(rescanned);
        assert!(
            progress.pending("worker-a").unwrap().1.is_empty(),
            "removed source is withdrawn"
        );
    }

    #[test]
    fn builtin_model_download_receipt_schedules_inventory_rescan() {
        // SW-7030: the initial inventory can finish before the user installs a builtin
        // auxiliary (`controlnet_tile_sdxl`). `run_model_download_job` then writes its
        // completion receipt under data/models/<safe_download_dir(repo)>/.sceneworks-download-complete.json
        // — the targetDir Model Manager enqueues for the catalog `downloads[]` entry. That
        // receipt must change the watched catalog so the next heartbeat rescans and
        // advertises `sdxl_detail_tile_control` instead of waiting out the 30-minute interval.
        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let mut settings = Settings::for_test(data.clone());
        settings.config_dir = temp.path().join("config");
        let source =
            sceneworks_core::control_weights::shipped_auxiliary_source("sdxl_detail_tile_control")
                .unwrap();
        // The worker's initial inventory ran to completion before anything was installed.
        let before = catalog_hash(&settings);
        let schedule = InventorySchedule {
            next_due: Some(Instant::now() + INVENTORY_RESCAN_INTERVAL),
            catalog_hash: Some(before),
        };
        // The user installs the builtin auxiliary: `run_model_download_job` lands the weights
        // in the HF cache the Detail enhancer resolves...
        let hub = temp.path().join("hub");
        let file = hub
            .join("models--xinsir--controlnet-tile-sdxl-1.0/snapshots")
            .join(source.revision)
            .join("diffusion_pytorch_model.safetensors");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"tile control weights").unwrap();
        // ...and writes its completion receipt into the app store target dir.
        let target = data
            .join("models")
            .join(crate::paths::safe_download_dir(source.repo));
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(
            target.join(".sceneworks-download-complete.json"),
            br#"{"schemaVersion":2,"repo":"xinsir/controlnet-tile-sdxl-1.0"}"#,
        )
        .unwrap();
        // The receipt changes the watched catalog, so the heartbeat rescan fires early...
        let after = catalog_hash(&settings);
        assert_ne!(
            before, after,
            "a completed builtin model download must schedule an inventory rescan"
        );
        assert!(scan_due(&schedule, Instant::now(), after));
        // ...and the rescan advertises the Detail tile source from the installed snapshot.
        crate::test_env::temp_env_vars(&[("HF_HUB_CACHE", hub.to_str().unwrap())], || {
            let mut scanned = Vec::new();
            collect_source_claims(&settings, |claim| scanned.push(claim));
            assert!(
                scanned
                    .iter()
                    .any(|claim| claim.source_id == "sdxl_detail_tile_control"),
                "the installed builtin auxiliary is advertised without the 30-minute rescan"
            );
        });
    }

    #[test]
    fn external_lora_is_advertised_and_rebound_by_exact_worker_bytes() {
        // SW-7030 (review 4531591689): an externally scanned ComfyUI LoRA gets a
        // synthetic `external_…` catalog row from the API, so the worker must
        // advertise the matching registered digest claim and rebind it by exact
        // bytes under its own configured root.
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path().join("ComfyUI/models");
        let adapter = root.join("loras/Wan/detailz-wan.safetensors");
        std::fs::create_dir_all(adapter.parent().unwrap()).unwrap();
        let header = serde_json::json!({
            "diffusion_model.blocks.0.cross_attn.k.lora_down.weight":
                {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "diffusion_model.blocks.0.cross_attn.k.lora_up.weight":
                {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&[0_u8; 8]);
        std::fs::write(&adapter, &bytes).unwrap();

        let mut settings = Settings::for_test(temp.path().join("worker-data"));
        settings.external_model_roots = vec![root];
        let entries = external_lora_entries(&settings);
        assert_eq!(entries.len(), 1, "the shared scan finds the adapter");
        let (id, _) = entries.into_iter().next().unwrap();
        assert!(id.starts_with("external_"), "synthetic id: {id}");

        let frozen = capture_registered_files(&adapter).unwrap();
        // Scope to the external-LoRA pass: the full inventory also hashes any
        // real weights installed on the host for unrelated static sources.
        let mut claims = Vec::new();
        external_lora_source_claims(&settings, &mut |claim| claims.push(claim));
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &frozen,
            )
            .unwrap();
        let expected_claim = sceneworks_core::portable_sources::registered_digest_claim_id(
            "lora", &id, None, &ledger,
        )
        .unwrap();
        assert!(
            claims
                .iter()
                .any(|claim| claim.source_id == expected_claim && claim.files == frozen),
            "the external LoRA is advertised with its exact byte ledger"
        );
        let resolved = registered_external_lora_path(&settings, &id, &frozen)
            .unwrap()
            .expect("matching bytes resolve");
        assert!(resolved.ends_with("detailz-wan.safetensors"));

        // Altered bytes no longer match the frozen ledger.
        let mut altered = bytes.clone();
        altered.extend_from_slice(b"tampered");
        std::fs::write(&adapter, altered).unwrap();
        assert!(
            registered_external_lora_path(&settings, &id, &frozen)
                .unwrap()
                .is_none(),
            "substituted bytes must not rebind"
        );
    }

    #[test]
    fn external_base_model_is_advertised_and_rebound_by_exact_worker_bytes() {
        // SW-7030 (review 4533688093): a runnable external ComfyUI base model is
        // forwarded as a local Digest entry, so the worker must advertise the
        // matching `model` digest claim and rebind the anchor by exact bytes.
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path().join("ComfyUI/models");
        let anchor = root.join("diffusion_models/anchor.safetensors");
        std::fs::create_dir_all(anchor.parent().unwrap()).unwrap();
        let header = serde_json::json!({
            "model.diffusion_model.input_blocks.0.0.weight":
                {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(&[0_u8; 4]);
        std::fs::write(&anchor, &bytes).unwrap();

        let mut settings = Settings::for_test(temp.path().join("worker-data"));
        settings.external_model_roots = vec![root];
        let rows = external_base_rows(&settings);
        assert_eq!(rows.len(), 1, "the shared scan assembles the anchor");
        let id = rows[0]["id"].as_str().unwrap().to_owned();
        assert!(id.starts_with("external_base_"), "synthetic id: {id}");

        let frozen = capture_registered_files(&anchor).unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(
                &frozen,
            )
            .unwrap();
        let expected_claim = sceneworks_core::portable_sources::registered_digest_claim_id(
            "model", &id, None, &ledger,
        )
        .unwrap();
        let mut claims = Vec::new();
        external_base_source_claims(&settings, &mut |claim| claims.push(claim));
        assert!(
            claims
                .iter()
                .any(|claim| claim.source_id == expected_claim && claim.files == frozen),
            "the external base model is advertised with its exact byte ledger"
        );
        assert!(
            registered_external_base_path(&settings, &id, None, &frozen)
                .unwrap()
                .is_some(),
            "matching bytes resolve"
        );

        let mut altered = bytes.clone();
        altered.extend_from_slice(b"tampered");
        std::fs::write(&anchor, altered).unwrap();
        assert!(
            registered_external_base_path(&settings, &id, None, &frozen)
                .unwrap()
                .is_none(),
            "substituted bytes must not rebind"
        );
    }

    #[cfg(unix)]
    #[test]
    fn registered_directory_model_traverses_linked_directories_within_managed_roots() {
        // SW-7030 (review 4533400138): a registered local directory model whose
        // `text_encoder/` is a symlink to shared components elsewhere under the
        // data dir must still produce its exact file ledger, with logical
        // relative paths, while a link escaping every managed root is refused.
        use std::os::unix::fs::symlink;

        let temp = tempfile::TempDir::new().unwrap();
        let data = temp.path().join("data");
        let shared = data.join("models/shared/text_encoder");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(shared.join("model.safetensors"), b"encoder bytes").unwrap();
        let model = data.join("models/local/demo");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("dit.safetensors"), b"dit bytes").unwrap();
        symlink(&shared, model.join("text_encoder")).unwrap();

        let settings = Settings::for_test(data.clone());
        let roots = crate::paths::allowed_model_roots(&settings).unwrap();
        let files = capture_registered_files_with_roots(&model, &roots).unwrap();
        let relatives: Vec<PathBuf> = files
            .iter()
            .map(|file| file.relative_path.clone())
            .collect();
        assert!(relatives.contains(&PathBuf::from("dit.safetensors")));
        assert!(relatives.contains(&PathBuf::from("text_encoder/model.safetensors")));

        // Review 4533885128: the `user.models.jsonc` inventory loop must also
        // pass managed roots so this model's claim is actually published.
        let mut inv_settings = Settings::for_test(data.clone());
        inv_settings.config_dir = temp.path().join("config");
        let user_manifest = inv_settings.config_dir.join("manifests/user.models.jsonc");
        std::fs::create_dir_all(user_manifest.parent().unwrap()).unwrap();
        std::fs::write(
            &user_manifest,
            serde_json::json!({
                "models": [{"id": "linked_demo", "installedPath": model.display().to_string()}]
            })
            .to_string(),
        )
        .unwrap();
        let ledger =
            sceneworks_core::model_artifacts::portable_binding::portable_file_ledger_sha256(&files)
                .unwrap();
        let expected_claim = sceneworks_core::portable_sources::registered_digest_claim_id(
            "model",
            "linked_demo",
            None,
            &ledger,
        )
        .unwrap();
        let mut published = Vec::new();
        registered_digest_source_claims(&inv_settings, |claim| published.push(claim));
        assert!(
            published
                .iter()
                .any(|claim| claim.source_id == expected_claim),
            "inventory must advertise the linked directory model"
        );

        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.safetensors"), b"secret").unwrap();
        symlink(&outside, model.join("escaped")).unwrap();
        assert!(capture_registered_files_with_roots(&model, &roots).is_err());
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
