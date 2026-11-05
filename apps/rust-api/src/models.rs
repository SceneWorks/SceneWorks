use super::*;

use sceneworks_core::base_weights::{
    detect_base_weight_file, import_detection_supported, imported_model_primary_weight_file,
    BaseWeightDetection, ComponentRole,
};
use sceneworks_core::credentials::normalize_host;
use sceneworks_core::lora_family::is_hidden_file;

const ALLOWED_MODEL_TYPES: &[&str] = &["image", "video", "audio", "utility"];
// The shipped catalog currently has more than 64 distinct (repo, files) contexts. Keep ample
// headroom for catalog growth so an in-order scan cannot evict the entries the next scan is about
// to read and degrade into a zero-hit cycle.
const MODEL_SIZE_CACHE_LIMIT: usize = 256;
pub(super) const MODEL_CATALOG_PROBE_CONCURRENCY: usize = 16;
static MODEL_CATALOG_PROBE_PERMITS: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> =
    std::sync::OnceLock::new();
// Failed estimates (offline, rate-limited, or size-less repo metadata) are
// negative-cached so a huggingface.co outage costs one 8s timeout per repo per
// TTL window instead of one per catalog load (sc-4169).
const MODEL_SIZE_NEGATIVE_TTL: Duration = Duration::from_secs(300);
// Successful metadata can change when a repository is updated in place. Refresh it periodically
// without making ordinary warm catalog loads depend on Hugging Face.
const MODEL_SIZE_POSITIVE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Generation-keyed, process-local snapshot of the model catalog after its
/// filesystem install-state sweep.
///
/// A separate async serializer makes concurrent cold `/models`, global preset,
/// project preset, and job-validation requests join one sweep without holding
/// the short synchronous state lock across filesystem work. `invalidate`
/// advances the generation and clears the snapshot in that same critical
/// section. Publication validates its generation under the same lock, so a
/// writer racing a cold build forces a rebuild instead of publishing or
/// returning stale state. Errors are never cached.
#[derive(Default)]
pub(crate) struct ModelCatalogCache {
    state: Mutex<ModelCatalogCacheState>,
    build_serializer: tokio::sync::Mutex<()>,
}

#[derive(Default)]
struct ModelCatalogCacheState {
    generation: u64,
    snapshot: Option<(u64, &'static str, Arc<Vec<Value>>)>,
    /// Last observed resolved-cache publish key. `None` until the first observation, which is why
    /// the first reading never counts as a change — an API that has not looked yet has nothing to
    /// have gone stale against.
    local_tier_key: Option<u64>,
}

impl ModelCatalogCache {
    pub(crate) fn invalidate(&self) {
        let mut state = self.state.lock();
        state.generation = state.generation.wrapping_add(1);
        state.snapshot = None;
    }

    /// Invalidates the snapshot when the resolved cache's published set has moved since the last
    /// catalog read (sc-19712 F-4).
    ///
    /// Every other invalidation site is a mutation this process performed, but promotions are
    /// published by the **worker**, in another process, so no in-process signal exists. Without
    /// this the catalog kept serving a model's pre-promotion availability indefinitely — and
    /// `modelLibraryUnavailable()` keys the Model Manager's blocked state on exactly that field,
    /// so a model that had just completed a job with the drive unplugged was still presented as
    /// needing the drive reconnected.
    ///
    /// A key check, not a timer: a TTL would leave the same wrong answer up for its whole window
    /// and rebuild pointlessly for the rest of time. The key only moves when a bundle is published
    /// or withdrawn, so a rebuild happens once per real change and never on a load.
    pub(crate) fn note_local_tier_key(&self, key: u64) {
        let mut state = self.state.lock();
        if state.local_tier_key == Some(key) {
            return;
        }
        let first_observation = state.local_tier_key.is_none();
        state.local_tier_key = Some(key);
        if first_observation {
            return;
        }
        state.generation = state.generation.wrapping_add(1);
        state.snapshot = None;
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.state.lock().generation
    }
}

fn validate_huggingface_repo(repo: &str) -> Result<&str, ApiError> {
    sceneworks_core::hf_repo_id::namespaced(repo).map_err(ApiError::bad_request)
}

#[derive(Debug, Default)]
pub(crate) struct ModelSizeCache {
    entries: HashMap<ModelSizeCacheKey, CachedSizeEstimate>,
    order: VecDeque<ModelSizeCacheKey>,
    in_flight: HashMap<ModelSizeCacheKey, Arc<ModelSizeInFlight>>,
}

type ModelSizeCacheKey = (String, Vec<String>);

#[derive(Debug, Clone, Copy)]
enum ModelSizeFlightStatus {
    Pending,
    Complete(Option<u64>),
    Aborted,
}

#[derive(Debug)]
struct ModelSizeInFlight {
    status: Mutex<ModelSizeFlightStatus>,
    changed: tokio::sync::Notify,
}

impl ModelSizeInFlight {
    fn pending() -> Self {
        Self {
            status: Mutex::new(ModelSizeFlightStatus::Pending),
            changed: tokio::sync::Notify::new(),
        }
    }

    async fn wait(&self) -> ModelSizeFlightStatus {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let status = *self.status.lock();
            if !matches!(status, ModelSizeFlightStatus::Pending) {
                return status;
            }
            changed.await;
        }
    }

    fn complete(&self, estimate: Option<u64>) {
        *self.status.lock() = ModelSizeFlightStatus::Complete(estimate);
        self.changed.notify_waiters();
    }

    fn abort(&self) {
        *self.status.lock() = ModelSizeFlightStatus::Aborted;
        self.changed.notify_waiters();
    }
}

enum ModelSizeLookup {
    Cached(Option<u64>),
    Lead(Arc<ModelSizeInFlight>),
    Follow(Arc<ModelSizeInFlight>),
    Unshared,
}

#[derive(Debug, Clone, Copy)]
struct CachedSizeEstimate {
    size_bytes: Option<u64>,
    expires_at: Option<std::time::Instant>,
}

impl ModelSizeCache {
    /// `Some(Some(bytes))` = cached estimate, `Some(None)` = cached failure
    /// (skip the network until the TTL lapses), `None` = cache miss.
    pub(crate) fn get(&mut self, key: &ModelSizeCacheKey) -> Option<Option<u64>> {
        if let Some(entry) = self.entries.get(key).copied() {
            if entry
                .expires_at
                .is_some_and(|expires_at| std::time::Instant::now() >= expires_at)
            {
                self.entries.remove(key);
                self.order.retain(|existing| existing != key);
                return None;
            }
            self.touch(key);
            return Some(entry.size_bytes);
        }
        None
    }

    fn lookup_or_start(&mut self, key: &ModelSizeCacheKey) -> ModelSizeLookup {
        if let Some(cached) = self.get(key) {
            return ModelSizeLookup::Cached(cached);
        }
        if let Some(in_flight) = self.in_flight.get(key) {
            return ModelSizeLookup::Follow(in_flight.clone());
        }
        if self.in_flight.len() >= MODEL_SIZE_CACHE_LIMIT {
            return ModelSizeLookup::Unshared;
        }
        let in_flight = Arc::new(ModelSizeInFlight::pending());
        self.in_flight.insert(key.clone(), in_flight.clone());
        ModelSizeLookup::Lead(in_flight)
    }

    fn finish(
        &mut self,
        key: &ModelSizeCacheKey,
        in_flight: &Arc<ModelSizeInFlight>,
        estimate: Option<u64>,
    ) {
        if self
            .in_flight
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, in_flight))
        {
            self.in_flight.remove(key);
            match estimate {
                Some(estimate) => self.insert(key.clone(), estimate),
                None => self.insert_failure(key.clone()),
            }
        }
        in_flight.complete(estimate);
    }

    fn abort(&mut self, key: &ModelSizeCacheKey, in_flight: &Arc<ModelSizeInFlight>) {
        if self
            .in_flight
            .get(key)
            .is_some_and(|current| Arc::ptr_eq(current, in_flight))
        {
            self.in_flight.remove(key);
        }
        in_flight.abort();
    }

    pub(crate) fn insert(&mut self, key: ModelSizeCacheKey, value: u64) {
        self.insert_entry(
            key,
            CachedSizeEstimate {
                size_bytes: Some(value),
                expires_at: Some(std::time::Instant::now() + MODEL_SIZE_POSITIVE_TTL),
            },
        );
    }

    pub(crate) fn insert_failure(&mut self, key: ModelSizeCacheKey) {
        self.insert_failure_expiring_at(key, std::time::Instant::now() + MODEL_SIZE_NEGATIVE_TTL);
    }

    pub(crate) fn insert_failure_expiring_at(
        &mut self,
        key: ModelSizeCacheKey,
        expires_at: std::time::Instant,
    ) {
        self.insert_entry(
            key,
            CachedSizeEstimate {
                size_bytes: None,
                expires_at: Some(expires_at),
            },
        );
    }

    fn insert_entry(&mut self, key: ModelSizeCacheKey, entry: CachedSizeEstimate) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, entry);
        while self.order.len() > MODEL_SIZE_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    fn touch(&mut self, key: &ModelSizeCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

struct ModelSizeFlightLeader {
    cache: Arc<Mutex<ModelSizeCache>>,
    key: ModelSizeCacheKey,
    in_flight: Arc<ModelSizeInFlight>,
    finished: bool,
}

impl ModelSizeFlightLeader {
    fn new(
        cache: Arc<Mutex<ModelSizeCache>>,
        key: ModelSizeCacheKey,
        in_flight: Arc<ModelSizeInFlight>,
    ) -> Self {
        Self {
            cache,
            key,
            in_flight,
            finished: false,
        }
    }

    fn finish(mut self, estimate: Option<u64>) {
        self.cache
            .lock()
            .finish(&self.key, &self.in_flight, estimate);
        self.finished = true;
    }
}

impl Drop for ModelSizeFlightLeader {
    fn drop(&mut self) {
        if !self.finished {
            self.cache.lock().abort(&self.key, &self.in_flight);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DownloadContext {
    repo: String,
    files: Vec<String>,
    fallback_size_bytes: Option<u64>,
}

pub(crate) async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(model_catalog_sized(&state).await?))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuggingFaceCacheHealth {
    pub(crate) installed: bool,
    pub(crate) incomplete: bool,
    pub(crate) missing_files: Vec<String>,
}

impl HuggingFaceCacheHealth {
    fn missing(missing_files: Vec<String>) -> Self {
        Self {
            installed: false,
            incomplete: true,
            missing_files,
        }
    }

    fn installed() -> Self {
        Self {
            installed: true,
            incomplete: false,
            missing_files: Vec::new(),
        }
    }
}

/// Machine-readable code on the license-acknowledgment rejection (sc-17227). Mirrored in the web
/// client as `LICENSE_ACK_ERROR_CODE` (`apps/web/src/licenseAcknowledgment.js`) so both halves of
/// the gate name the same refusal rather than matching on prose.
pub(crate) const LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE: &str = "license_acknowledgment_required";

/// True when the catalog entry declares that the user must accept its license before the weights
/// may be downloaded (`requiresLicenseAcknowledgment`, sc-17227). Deliberately does NOT include
/// `gated`: see the call site for why the two are enforced differently.
pub(crate) fn model_requires_license_acknowledgment(model: &Value) -> bool {
    model
        .get("requiresLicenseAcknowledgment")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The payload key every job-creation door uses to carry the caller's acknowledgment through to
/// the queue (sc-17227). Stamped by `create_model_download_job` once its own gate has passed, so a
/// legacy explicit assertions survive retry; server-record authorization is rechecked
/// against current terms and withdrawal on each request.
pub(crate) const LICENSE_ACKNOWLEDGED_PAYLOAD_KEY: &str = "licenseAcknowledged";

/// Error code for a fetch of a repo a catalog entry declares as a BLOCKED conditional component
/// (sc-22998: YuE2's SheetSage2 / MERT-v2-FullSong cover closure). Distinct from
/// [`LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE`] because no acknowledgment clears it: the block is an
/// owner decision the user cannot make from a checkbox.
pub(crate) const COMPONENT_BLOCKED_CODE: &str = "component_blocked";

/// The derivation a `ModelDownload` job runs after fetching the original (sc-22999): the worker
/// derives this tier locally through the audio-lane preparer into
/// `local_derivation_snapshot_dir` and verifies it against the pinned weights.
pub(crate) const LOCAL_DERIVATION_PAYLOAD_KEY: &str = "localDerivation";

/// Canonical comparison key for a Hugging Face `owner/name`. Lowercased so a case-variant repo
/// string cannot walk past a gate keyed on the catalog's spelling — the hub resolves `owner/Name`
/// and `owner/name` to the same repository, so treating them as different would be a bypass.
///
/// A trailing `.git` is stripped for the same reason: `MiniMaxAI/MiniMax-H3.git` is the git-remote
/// spelling of the same repository, it passes the worker's `validate_hf_repo_id`, and it was the
/// one spelling that missed this index — leaving Hugging Face's own 401 as the only thing between
/// the request and the weights. Stripped AFTER the trailing-slash trim (and re-trimmed) so
/// `…/MiniMax-H3.git/` and `…/MiniMax-H3/.git` both normalize too.
fn huggingface_repo_key(repo: &str) -> Option<String> {
    let repo = repo.trim().trim_end_matches('/').trim();
    let repo = match repo.rfind('.') {
        // `rfind` yields a char boundary, so the slice is safe; compared case-insensitively
        // because the lowercasing below happens only after this strip.
        Some(dot) if repo[dot..].eq_ignore_ascii_case(".git") => repo[..dot].trim_end_matches('/'),
        _ => repo,
    };
    let repo = repo.trim();
    if repo.is_empty() {
        return None;
    }
    Some(repo.to_ascii_lowercase())
}

/// The `owner/name` a huggingface.co URL addresses, or `None` for any other host. `/models/import`
/// accepts a `sourceUrl` as an alternative to `repo`, and
/// `https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/…` fetches exactly the same bytes as
/// `repo: "MiniMaxAI/MiniMax-H3"`, so a repo-keyed gate that read only `repo` would leave the
/// equivalent request open.
fn huggingface_repo_from_url(url: &str) -> Option<String> {
    let url = url.trim();
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let (host, path) = rest.split_once('/')?;
    let host = host.split('@').next_back().unwrap_or(host);
    let host = host.split(':').next().unwrap_or(host).to_ascii_lowercase();
    if host != "huggingface.co" && host != "www.huggingface.co" && host != "hf.co" {
        return None;
    }
    // `/models/<owner>/<name>` and `/<owner>/<name>` both address a model repo; `datasets/…` and
    // `spaces/…` are different namespaces and are left alone.
    let path = path.strip_prefix("models/").unwrap_or(path);
    let mut segments = path.split('/').filter(|segment| !segment.is_empty());
    let owner = segments.next()?;
    if matches!(owner, "datasets" | "spaces" | "api") {
        return None;
    }
    let name = segments.next()?;
    Some(format!("{owner}/{name}"))
}

/// Every Hugging Face repo declared by a catalog entry that requires a license acknowledgment,
/// mapped to the entry that declares it (sc-17227). Includes co-requisite rows: MiniMax-H3's text
/// encoder and both VAEs come straight from `MiniMaxAI/MiniMax-H3`, which is the repo the review's
/// bypass named, and a primary-only index would have missed it.
///
/// Also every `conditionalComponents[].repo` (sc-22998). Those repos are not `downloads[]` rows, so
/// no install fetches them, but every repo-keyed door (`POST /api/v1/jobs`, `/models/import`,
/// `/loras/import`, `/loras/:id/download`) takes a caller-supplied repo verbatim, so the gate must
/// know them. A component carrying `blocked` is indexed for EVERY entry, acknowledgment or not, and
/// is refused outright by [`ensure_license_acknowledged_for_source`]; a blocked entry takes
/// precedence over an acknowledgment entry for the same repo.
///
/// Read from the UNFILTERED manifest entries on purpose. The catalog snapshot narrows `downloads`
/// to the running OS (`retain_downloads_for_os`), and every MiniMax-H3 row is platform-scoped: the
/// MLX tiers and their co-requisites are `platforms: ["macos"]` and sc-19558's raw-snapshot set is
/// `platforms: ["windows", "linux"]`. An index built from the snapshot would therefore see only the
/// subset that survived the filter on the running host, and the gate would be keyed on a partial
/// view of the repos an entry can actually fetch — on exactly the hosts where the LAN-exposed jobs
/// API (epic 4484) is most likely to be reachable. A licence requirement is not a platform
/// capability.
async fn license_acknowledgment_repo_index(
    state: &AppState,
) -> Result<std::collections::BTreeMap<String, LicenseAcknowledgmentSource>, ApiError> {
    let (models, _) = merged_model_manifest_entries(state).await?;
    Ok(license_acknowledgment_index_for(&models))
}

/// [`license_acknowledgment_repo_index`] over already-loaded manifest entries.
fn license_acknowledgment_index_for(
    models: &[Value],
) -> std::collections::BTreeMap<String, LicenseAcknowledgmentSource> {
    license_acknowledgment_index_for_selected(models, None)
}

fn license_acknowledgment_index_for_selected(
    models: &[Value],
    selected_model_id: Option<&str>,
) -> std::collections::BTreeMap<String, LicenseAcknowledgmentSource> {
    let mut index: std::collections::BTreeMap<String, LicenseAcknowledgmentSource> =
        std::collections::BTreeMap::new();
    // A catalog-validated selection owns its declared shared repositories. Raw
    // repository requests retain canonical manifest order; an arbitrary accepted
    // model must never grant access. Blocked components from ANY entry still win.
    let selected = |model: &&Value| {
        selected_model_id.is_some_and(|id| model.get("id").and_then(Value::as_str) == Some(id))
    };
    for model in models
        .iter()
        .filter(selected)
        .chain(models.iter().filter(|model| !selected(model)))
    {
        let Some(model_id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        let model_name = model
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(model_id)
            .to_owned();
        let source = |blocked: Option<BlockedComponent>| LicenseAcknowledgmentSource {
            model_id: model_id.to_owned(),
            model_name: model_name.clone(),
            terms: model.clone(),
            blocked,
        };
        let conditional = model
            .get("conditionalComponents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        // Blocked conditional components first, for EVERY entry: no acknowledgment unblocks them.
        for component in conditional.clone() {
            let Some(block) = component.get("blocked") else {
                continue;
            };
            let Some(key) = component
                .get("repo")
                .and_then(Value::as_str)
                .and_then(huggingface_repo_key)
            else {
                continue;
            };
            let text = |field: &str| {
                block
                    .get(field)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            index.insert(
                key,
                source(Some(BlockedComponent {
                    component_id: component
                        .get("componentId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    reason: text("reason"),
                    unblock: text("unblock"),
                })),
            );
        }
        if !model_requires_license_acknowledgment(model) {
            continue;
        }
        let downloads = model
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        for row in downloads.chain(conditional) {
            let Some(key) = row
                .get("repo")
                .and_then(Value::as_str)
                .and_then(huggingface_repo_key)
            else {
                continue;
            };
            index.entry(key).or_insert_with(|| source(None));
        }
    }
    index
}

/// A conditional component a catalog entry declares as `blocked` (sc-22998).
#[derive(Clone)]
pub(crate) struct BlockedComponent {
    pub(crate) component_id: String,
    pub(crate) reason: String,
    pub(crate) unblock: String,
}

/// The catalog entry whose licence acknowledgment covers a fetch of some repo. The NAME travels
/// with the id because the surfaces that have to explain the requirement — the LoRA rows on the
/// Models screen — have no licence copy of their own and must point the user at the model card
/// that does.
#[derive(Clone)]
pub(crate) struct LicenseAcknowledgmentSource {
    pub(crate) model_id: String,
    pub(crate) model_name: String,
    terms: Value,
    /// Set when the repo is a BLOCKED conditional component (sc-22998): refused whatever the
    /// caller asserts.
    pub(crate) blocked: Option<BlockedComponent>,
}

/// Client-visible keys naming the model whose licence acknowledgment covers a catalog row that is
/// not itself a model (sc-17227). Written onto LoRA catalog rows by `list_loras`.
pub(crate) const LICENSE_ACKNOWLEDGMENT_MODEL_ID_KEY: &str = "licenseAcknowledgmentModelId";
pub(crate) const LICENSE_ACKNOWLEDGMENT_MODEL_NAME_KEY: &str = "licenseAcknowledgmentModelName";

/// Stamp each catalog row whose Hugging Face source repo is licence-gated with the model that
/// gates it (sc-17227), so a client can raise the SAME acknowledgment gate it raises on a model
/// and send the assertion the server now requires.
///
/// Without this, `create_lora_download_job`'s repo-keyed gate is unsatisfiable from the shipped UI:
/// the row carries nothing that says an acknowledgment is needed, `createLoraDownloadJob` sends no
/// `licenseAcknowledged`, and the click yields a bare 403 with no checkbox anywhere to clear it.
///
/// Derived here rather than authored in `builtin.loras.jsonc` on purpose. A manifest flag is a
/// second copy of a fact the model manifest already states, and the two drift; this reads the one
/// source. It is also the only form that is correct on every host — the index is built from the
/// UNFILTERED model manifest, so it does not evaporate on a platform where the gating model's
/// download rows are filtered out, which is exactly where a client-side re-derivation would fail.
///
/// Applied at the CATALOG-READ door (`list_loras`) and not inside `lora_catalog`, which the
/// per-job-create validation sweep also calls (sc-8819): the annotation is for rendering, and the
/// enforcement path resolves the repo itself.
pub(crate) async fn annotate_license_acknowledgment_sources(
    state: &AppState,
    rows: &mut [Value],
    repo_of: impl Fn(&Value) -> Option<String>,
) -> Result<(), ApiError> {
    let keyed: Vec<(usize, String)> = rows
        .iter()
        .enumerate()
        .filter_map(|(position, row)| {
            repo_of(row)
                .as_deref()
                .and_then(huggingface_repo_key)
                .map(|key| (position, key))
        })
        .collect();
    if keyed.is_empty() {
        return Ok(());
    }
    let index = license_acknowledgment_repo_index(state).await?;
    if index.is_empty() {
        return Ok(());
    }
    for (position, key) in keyed {
        let Some(source) = index
            .get(key.as_str())
            .filter(|source| source.blocked.is_none())
        else {
            continue;
        };
        let Some(object) = rows[position].as_object_mut() else {
            continue;
        };
        object.insert(
            LICENSE_ACKNOWLEDGMENT_MODEL_ID_KEY.to_owned(),
            Value::String(source.model_id.clone()),
        );
        object.insert(
            LICENSE_ACKNOWLEDGMENT_MODEL_NAME_KEY.to_owned(),
            Value::String(source.model_name.clone()),
        );
    }
    Ok(())
}

/// The license-acknowledgment refusal for a request that named its weights by REPO (sc-17227).
///
/// [`create_model_download_job`] gates the typed `POST /api/v1/models/:id/download` by catalog id.
/// That is not the only door: `POST /api/v1/jobs` enqueues a `model_download` payload VERBATIM
/// (`repo` + `files` + `revision`, no catalog lookup anywhere between the request and
/// `run_model_download_job`), and `POST /api/v1/models/import` fetches a caller-supplied repo or
/// URL with no licence logic of its own. Both reached `MiniMaxAI/MiniMax-H3` — a PUBLIC repo, so
/// nothing upstream refuses them — while the typed route answered 403. Keying on the repo rather
/// than on the model id is what lets ONE mechanism close both: the payloads have no `modelId` to
/// look up, but they must name the repo or they cannot fetch anything.
///
/// `repos` is a LIST because "the repo this request will fetch" is not always spelled `repo`: a
/// `model_convert` payload names its download target in `baseRepo` (the LTX converter's
/// `ensure_ltx_upscaler_cached` fetches it, and `upscalerFile` is a glob, so `"**"` pulls the whole
/// repo). Checking every repo-bearing key of a payload is what keeps this ONE predicate rather than
/// one per job type — a new key is an addition to the list, not a second gate.
///
/// `acknowledged` is a legacy explicit acceptance assertion. Without it, the existing
/// terms-bound server acknowledgment is required (see
/// `docs/minimax-h3-use-restriction-safeguards.md`).
pub(crate) async fn ensure_license_acknowledged_for_source(
    state: &AppState,
    repos: &[Option<&str>],
    source_url: Option<&str>,
    acknowledged: bool,
) -> Result<(), ApiError> {
    ensure_license_acknowledged_for_source_inner(state, repos, source_url, acknowledged, None).await
}

// Only catalog-resolved model/component download routes supply a selected ID.
// Other repository-bearing doors deliberately use the public canonical wrapper.
async fn ensure_license_acknowledged_for_source_inner(
    state: &AppState,
    repos: &[Option<&str>],
    source_url: Option<&str>,
    acknowledged: bool,
    selected_model_id: Option<&str>,
) -> Result<(), ApiError> {
    // Each candidate keeps the caller's own spelling next to the lookup key, so the refusal echoes
    // what was actually requested rather than the lowercased index key.
    let candidates: Vec<(String, String)> = repos
        .iter()
        .copied()
        .flatten()
        .map(str::to_owned)
        .chain(source_url.and_then(huggingface_repo_from_url))
        .filter_map(|named| huggingface_repo_key(&named).map(|key| (named, key)))
        .collect();
    if candidates.is_empty() {
        return Ok(());
    }
    let index = if selected_model_id.is_some() {
        let (models, _) = merged_model_manifest_entries(state).await?;
        license_acknowledgment_index_for_selected(&models, selected_model_id)
    } else {
        license_acknowledgment_repo_index(state).await?
    };
    // A blocked conditional component is refused before anything else, and an acknowledgment does
    // not clear it (sc-22998).
    if let Some((requested, source, blocked)) = candidates.iter().find_map(|(named, key)| {
        index.get(key.as_str()).and_then(|source| {
            source
                .blocked
                .as_ref()
                .map(|blocked| (named, source, blocked))
        })
    }) {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            detail: format!(
                "'{requested}' is the '{component}' component of '{model_id}', which is blocked: \
                 {reason} Unblock condition: {unblock}",
                component = blocked.component_id,
                model_id = source.model_id,
                reason = blocked.reason,
                unblock = blocked.unblock,
            ),
            code: Some(COMPONENT_BLOCKED_CODE),
            context: None,
        });
    }
    // Every restricted source must have a current acknowledgment. Checking only the
    // first repo could authorize a second component governed by different terms.
    let mut missing = None;
    for (requested, key) in &candidates {
        let Some(source) = index.get(key.as_str()) else {
            continue;
        };
        if !acknowledged
            && crate::yue2_jobs::current_license_acknowledgment(state, &source.terms)
                .await?
                .is_none()
        {
            missing = Some((requested, source));
            break;
        }
    }
    let Some((requested, source)) = missing else {
        return Ok(());
    };
    let model_id = &source.model_id;
    Err(ApiError {
        status: StatusCode::FORBIDDEN,
        detail: format!(
            "'{requested}' supplies '{model_id}', which requires accepting its license before \
             download. Accept the license on the Models screen, or send \
             `licenseAcknowledged: true` to assert that the user has accepted it."
        ),
        code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
        context: None,
    })
}

/// Fine-tune license text never replaces obligations of the components it actually reuses.
/// The caller supplies a server-resolved entry; new imports use their immutable declaration and
/// older H3 entries derive obligations from the trusted builtin component catalog.
pub(crate) async fn ensure_inherited_component_licenses_acknowledged(
    state: &AppState,
    entry: &JsonObject,
    license_acknowledged: bool,
) -> Result<(), ApiError> {
    use sceneworks_core::license_acknowledgments as acknowledgments;
    // Architecture is not an import claim. Ordinary stock/custom HF or local catalog rows keep
    // their own license guards; only positive plan, declaration, or import-origin evidence owes
    // inherited-component authority. Never let a false author license/stock flag erase evidence.
    let import_path = |path: &str| {
        path.starts_with("models/imports/")
            || path.contains("/models/imports/")
            || path.starts_with("linked/")
    };
    let source = entry.get("source").and_then(Value::as_object);
    let import_origin = sceneworks_core::jobs_store::imported_entry_installed_path(entry)
        .is_some_and(import_path)
        || source.is_some_and(|source| {
            source.contains_key("rootId")
                || source
                    .get("path")
                    .and_then(Value::as_str)
                    .is_some_and(import_path)
        });
    let has_import_evidence = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
        .is_some()
        || [
            "importPlan",
            "baseModel",
            "minimaxH3Partition",
            "importSourceShape",
            "importComponentTier",
            "importQuantFormat",
            "modelDeclaration",
            "modelProvenance",
        ]
        .iter()
        .any(|key| entry.contains_key(*key))
        || entry
            .get("inheritedLicenseAcknowledgments")
            .is_some_and(|value| value.as_array().is_none_or(|rows| !rows.is_empty()))
        || import_origin;
    if !has_import_evidence {
        return Ok(());
    }
    let declaration =
        sceneworks_core::model_artifacts::provenance::import_declaration_context(entry)
            .map_err(ApiError::bad_request)?;
    let bound = sceneworks_core::model_artifacts::provenance::resolved_model_provenance(
        &state.settings.data_dir,
        entry,
    )
    .map_err(ApiError::bad_request)?;
    let inherited = bound
        .as_ref()
        .and_then(|value| value.get("baseLineage"))
        .and_then(|_| {
            bound
                .as_ref()
                .and_then(|value| value.get("inheritedLicenseAcknowledgments"))
        })
        .or_else(|| {
            declaration
                .get("modelDeclaration")
                .and_then(|value| value.get("inheritedLicenseAcknowledgments"))
        });
    let path = acknowledgments::acknowledgments_file(&state.settings.config_dir);
    for component in inherited.and_then(Value::as_array).into_iter().flatten() {
        if component
            .get("requiresLicenseAcknowledgment")
            .and_then(Value::as_bool)
            != Some(true)
        {
            continue;
        }
        let id = component
            .get("modelId")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                ApiError::bad_request("Inherited component license has no model identity")
            })?;
        let terms = json!({"id":id,"license":component.get("license"),
            "licenseUrl":component.get("licenseUrl"),"licenseNotice":component.get("licenseNotice")});
        if component.get("termsSha256").and_then(Value::as_str)
            != Some(acknowledgments::terms_sha256(&terms).as_str())
        {
            return Err(ApiError::bad_request(
                "Inherited component license commitment is invalid",
            ));
        }
        if license_acknowledged {
            acknowledgments::record_acknowledgment(
                &path,
                &terms,
                "api-import-components",
                &sceneworks_core::time::utc_now(),
            )
            .map_err(|error| ApiError::internal(error.to_string()))?;
        } else if acknowledgments::current_acknowledgment(&path, &terms)
            .map_err(|error| ApiError::internal(error.to_string()))?
            .is_none()
        {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                detail: format!("Component '{id}' requires accepting its inherited license. Accept the component terms on the Models screen or send licenseAcknowledged: true."),
                code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE), context: None,
            });
        }
    }
    Ok(())
}

/// Every payload key that can name a Hugging Face repo the WORKER will fetch (sc-17227). Keep this
/// aligned with the worker's own readers: `run_model_download_job` / `run_model_import_job` /
/// `run_lora_*_job` take `repo` (`crates/sceneworks-worker/src/model_jobs.rs`), and
/// `resolve_convert_plan` takes `baseRepo` — which the LTX arm hands to `ensure_ltx_upscaler_cached`
/// → `ensure_hf_files_cached`, a real download. `sourceRepo` is listed because it is the other repo
/// a convert payload names; it resolves against the local cache today (`huggingface_snapshot_dir`),
/// so gating it costs nothing and removes the question of which of the two a future arm fetches.
const LICENSE_GATED_REPO_PAYLOAD_KEYS: &[&str] = &["repo", "baseRepo", "sourceRepo"];

/// [`ensure_license_acknowledged_for_source`] over a raw job payload — the shape
/// `POST /api/v1/jobs` (and the retry/duplicate re-validation) hands to the worker verbatim.
pub(crate) async fn ensure_job_payload_license_acknowledged(
    state: &AppState,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    let repos: Vec<Option<&str>> = LICENSE_GATED_REPO_PAYLOAD_KEYS
        .iter()
        .map(|key| payload.get(*key).and_then(Value::as_str))
        .collect();
    ensure_license_acknowledged_for_source(
        state,
        &repos,
        payload.get("sourceUrl").and_then(Value::as_str),
        payload
            .get(LICENSE_ACKNOWLEDGED_PAYLOAD_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false),
    )
    .await
}

async fn queue_catalog_download(
    state: AppState,
    payload: JsonObject,
    requested_gpu: String,
    binding: Value,
) -> Result<JobSnapshot, ApiError> {
    create_generation_job_with_binding(
        state,
        JobType::ModelDownload,
        None,
        None,
        payload,
        requested_gpu,
        (None, Some(binding)),
    )
    .await
}

fn catalog_download_binding(
    model_id: &str,
    owner_id: &str,
    kind: &str,
    row: &Value,
    payload: &JsonObject,
    purpose: Option<&str>,
    component: Option<&str>,
) -> Value {
    json!({"modelId": model_id, "ownerId": owner_id, "kind": kind, "row": row,
        "payload": payload, "purpose": purpose, "component": component})
}

pub(crate) async fn validate_catalog_download_replay(
    state: &AppState,
    binding: &Value,
    payload: &JsonObject,
) -> Result<(), ApiError> {
    let invalid = || {
        ApiError::bad_request(
            "The saved catalog download no longer matches its authorized selection",
        )
    };
    // A patch cannot turn an accepted selection into a different worker operation.
    if !sceneworks_core::jobs_store::download_binding_matches_payload(binding, payload) {
        return Err(invalid());
    }
    let model_id = binding
        .get("modelId")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let owner_id = binding
        .get("ownerId")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let row = binding.get("row").ok_or_else(invalid)?;
    let catalog = model_catalog(state).await?;
    let model = catalog
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(model_id))
        .ok_or_else(invalid)?;
    let owner = catalog
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(owner_id))
        .ok_or_else(invalid)?;
    let valid = match binding.get("kind").and_then(Value::as_str) {
        Some("download") if model_id == owner_id => model
            .get("downloads")
            .and_then(Value::as_array)
            .is_some_and(|rows| {
                rows.contains(row)
                    && binding
                        .get("selectedDownload")
                        .is_none_or(|selected| rows.contains(selected))
            }),
        Some("conditional") if model_id == owner_id => {
            let purpose = binding
                .get("purpose")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            sceneworks_core::model_usage_policy::conditional_component_downloads(model, purpose)
                .map_err(|_| invalid())?
                .contains(row)
        }
        Some("importComponent") => {
            if model.get("baseModel").and_then(Value::as_str) != Some(owner_id) {
                return Err(invalid());
            }
            let component = binding
                .get("component")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            let current = bound_import_component_download(
                model,
                owner,
                component,
                None,
                &state.settings.data_dir,
                true,
            )
            .map_err(|_| invalid())?;
            ensure_inherited_component_licenses_acknowledged(
                state,
                model.as_object().ok_or_else(invalid)?,
                false,
            )
            .await?;
            &current == row
        }
        _ => false,
    };
    if !valid {
        return Err(invalid());
    }
    for entry in [model, owner] {
        if model_requires_license_acknowledgment(entry)
            && crate::yue2_jobs::current_license_acknowledgment(state, entry)
                .await?
                .is_none()
        {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                detail: "Current model license acknowledgment is required".to_owned(),
                code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
                context: None,
            });
        }
    }
    let repos = LICENSE_GATED_REPO_PAYLOAD_KEYS
        .iter()
        .map(|key| payload.get(*key).and_then(Value::as_str))
        .collect::<Vec<_>>();
    ensure_license_acknowledged_for_source_inner(
        state,
        &repos,
        payload.get("sourceUrl").and_then(Value::as_str),
        false,
        Some(owner_id),
    )
    .await
}

pub(crate) async fn create_model_download_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = require_model_by_id(&state, &model_id).await?;
    // License-acknowledgment gate (sc-17227), enforced HERE and not only in the web client.
    //
    // The web client refuses an unacknowledged download at `createModelDownloadJob`, but that code
    // only runs in the client. This endpoint is reachable from a browser on another machine (the
    // remote-access lane, epic 4484), from a workflow envelope's suggested action, and from curl,
    // and no client-side check binds any of those.
    //
    // Scoped to `requiresLicenseAcknowledgment` — NOT to `gated`. A gated model's download fails at
    // Hugging Face with a 401 without a saved credential, so an unacknowledged fetch never lands its
    // weights; adding the requirement there would 4xx every existing gated download whose client
    // predates the field, for no gain. A `requiresLicenseAcknowledgment` model's repo is PUBLIC —
    // nothing upstream refuses it — so this rejection is the only thing between the request and the
    // weights, which is why the flag defaults to `false` and must be asserted, not assumed.
    if model_requires_license_acknowledgment(&model)
        && !payload.license_acknowledged
        && crate::yue2_jobs::current_license_acknowledgment(&state, &model)
            .await?
            .is_none()
    {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            detail: format!(
                "Model '{model_id}' requires accepting its license before download. Accept the \
                 license on the Models screen, or send `licenseAcknowledged: true` to assert that \
                 the user has accepted it."
            ),
            code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
            context: None,
        });
    }
    if let Some(component_id) = payload.component_id.as_deref() {
        ensure_model_not_cleanup_only(&model)?;
        if !payload.choices.is_empty() {
            return Err(ApiError::bad_request(
                "Imported component repair takes no co-requisite choices",
            ));
        }
        let base_id = model
            .get("baseModel")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let base = find_model_by_id(&state, base_id).await?.ok_or_else(|| {
            ApiError::bad_request("Imported component repair requires its compatible base")
        })?;
        let download = bound_import_component_download(
            &model,
            &base,
            component_id,
            payload.variant.as_deref(),
            &state.settings.data_dir,
            false,
        )
        .map_err(ApiError::bad_request)?;
        if payload.license_acknowledged {
            for entry in [&model, &base] {
                if model_requires_license_acknowledgment(entry) {
                    crate::yue2_jobs::record_license_acknowledgment(&state, entry, "download")
                        .await?;
                }
            }
        }
        ensure_inherited_component_licenses_acknowledged(
            &state,
            model.as_object().expect("validated import"),
            payload.license_acknowledged,
        )
        .await?;
        ensure_license_acknowledged_for_source_inner(
            &state,
            &[download.get("repo").and_then(Value::as_str)],
            None,
            payload.license_acknowledged,
            base.get("id").and_then(Value::as_str),
        )
        .await?;
        let mut job_payload = build_model_download_job_payload(
            &model,
            &model_id,
            &download,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        job_payload.insert("componentId".to_owned(), json!(component_id));
        job_payload.insert(
            "importComponentTier".to_owned(),
            model["importComponentTier"].clone(),
        );
        let binding = catalog_download_binding(
            &model_id,
            base["id"].as_str().expect("validated base"),
            "importComponent",
            &download,
            &job_payload,
            None,
            Some(component_id),
        );
        let job = queue_catalog_download(
            state,
            job_payload,
            requested_gpu_or_auto(payload.requested_gpu),
            binding,
        )
        .await?;
        return Ok((StatusCode::CREATED, Json(public_job_snapshot(job))));
    }
    ensure_model_downloadable(&model)?;
    // sc-22999: the acceptance is recorded server-side for the entry's current terms, so a queued
    // YuE2 job can re-check it at execution and a withdrawn acknowledgment refuses the job.
    if model_requires_license_acknowledgment(&model) && payload.license_acknowledged {
        crate::yue2_jobs::record_license_acknowledgment(&state, &model, "download").await?;
    }
    // Tier selection (sc-8508): an explicit `variant` installs that quant tier's download entry; an
    // absent variant installs the default tier (back-compat). A variant the model doesn't advertise
    // is a 400 rather than a silent wrong-tier install.
    let download = match payload
        .variant
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        Some(variant) => model_download_for_variant(&model, variant).ok_or_else(|| {
            ApiError::bad_request(format!(
                "Model does not define a '{variant}' download variant"
            ))
        })?,
        None => model_download(&model).ok_or_else(|| {
            ApiError::bad_request("Model does not define a Hugging Face download")
        })?,
    };
    let selected_download = download.clone();
    // sc-24112: a DECLARED-but-unpublished tier is refused HERE, at the one place a download job is
    // created, rather than by hiding the row — the catalog must keep enumerating it so the tier
    // axis, the memory ladder and the picker are all real before the bytes exist. Its revision is
    // the null SHA, so without this the job would queue a fetch that cannot resolve and the user
    // would see an opaque download failure instead of the reason.
    if is_pending_artifact_download(&download) {
        return Err(ApiError::bad_request(format!(
            "Model '{model_id}' declares the '{}' tier but its artifact is not published yet; \
             install a different tier.",
            download
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("selected")
        )));
    }
    // sc-22998 / sc-22999: a locally derived tier (YuE2 q8 / q4) is not an artifact anyone can
    // download — it is produced on this machine from the `fromVariant` original. Its row's `files`
    // name the ORIGINAL, so the job fetches (or re-verifies) that original under the ORIGINAL's
    // variant (its receipt is the original's, never the tier's) and carries the derivation for the
    // worker's audio-lane preparer to run afterwards. The tier reads installed only once its
    // snapshot verifies against the pinned weights (`derived_snapshot_state`).
    let derivation =
        if sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
            &download,
        ) {
            let variant = download
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("selected")
                .to_owned();
            let derivation =
                sceneworks_core::model_artifacts::artifact_selection::local_derivation(&download)
                    .ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "Model '{model_id}': the '{variant}' tier declares a malformed \
                         localDerivation block."
                    ))
                })?;
            Some((variant, derivation))
        } else {
            None
        };
    let download = match &derivation {
        Some((variant, derivation)) => model_download_for_variant(&model, &derivation.from_variant)
            .filter(|original| {
                !sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
                    original,
                )
            })
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "Model '{model_id}': the '{variant}' tier derives from '{}', which this \
                     catalog does not define as a download.",
                    derivation.from_variant
                ))
            })?,
        None => download,
    };
    // The selected `download` is always the primary/tier entry — `model_download` and
    // `model_download_for_variant` skip co-requisites (sc-9696), so a co-requisite can never be
    // installed as if it were the model itself.
    let requested_variant = match &derivation {
        Some((_, derivation)) => Some(derivation.from_variant.as_str()),
        None => payload
            .variant
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    };

    // Only the SELECTED tier's co-requisites (sc-14980). Mage-Flow's shared text encoder exists as
    // three per-tier subtrees; fetching all of them would pull 16.1 GB of text encoder for a q4
    // install that needs 2.51 GB. Tier-agnostic co-requisites (every other family) are unaffected —
    // they carry no `variant` and always apply. Read the tier off the resolved `download` rather than
    // the request so the default-tier install (no explicit `variant`) picks up its tier too.
    let selected_variant = download
        .get("variant")
        .and_then(Value::as_str)
        .map(str::to_owned);
    // Choice groups (sc-22998): of a group's options — YuE2's standard / legacy decoder — only the
    // one the request names (else the manifest default) is queued; the others stay add-ons. An
    // unknown group or option is refused rather than answered with a different option.
    //
    // The request's choices are validated first (an unknown group or option is refused), then any
    // group the request leaves unset takes the option already partly on disk before the default, so
    // a REPAIR of a half-downloaded legacy decoder completes it instead of fetching the standard one.
    sceneworks_core::model_artifacts::artifact_selection::resolve_co_requisite_choices(
        &model,
        selected_variant.as_deref(),
        &payload.choices,
    )
    .map_err(|error| ApiError::bad_request(format!("Model '{model_id}': {error}")))?;
    let repair_choices =
        sceneworks_core::model_artifacts::artifact_selection::co_requisite_choices_for_repair(
            &model,
            selected_variant.as_deref(),
            &payload.choices,
            |download| co_requisite_presence(&state.settings.data_dir, download),
        );
    let choices =
        sceneworks_core::model_artifacts::artifact_selection::resolve_co_requisite_choices(
            &model,
            selected_variant.as_deref(),
            &repair_choices,
        )
        .map_err(|error| ApiError::bad_request(format!("Model '{model_id}': {error}")))?;
    let co_requisites =
        sceneworks_core::model_artifacts::artifact_selection::model_co_requisite_downloads_for_selection(
            &model,
            selected_variant.as_deref(),
            &choices,
        );

    // The REPO-keyed half of the same gate (sc-17227). The check above is keyed on the catalog id
    // in the PATH, so it fires only when the entry that id names declares
    // `requiresLicenseAcknowledgment` itself. Every other door — `POST /api/v1/jobs`,
    // `/models/import`, `/loras/import` — is keyed on the repo the request will FETCH, resolved
    // against `license_acknowledgment_repo_index`. That asymmetry left two doors onto one set of
    // weights: an entry that does not carry the flag but whose `downloads` name a repo a flagged
    // entry declares would have been fetched here while the generic queue answered 403 for the same
    // repo. Shared co-requisite rows are exactly that shape, and the manifest already uses it.
    //
    // Unreachable in the shipped catalog today — every restricted repo reference lives inside an
    // entry that carries the flag, and the manifest audit's
    // `test_every_entry_naming_a_license_gated_repo_carries_the_flag_itself` keeps it that way —
    // but that is a property of the current manifest, not of this route. An
    // ADDITION, not a replacement: the id check above keeps its own refusal text, which names the
    // model the caller asked for rather than the repo that supplies it.
    let queued_repos: Vec<String> = std::iter::once(&download)
        .chain(co_requisites.iter())
        .filter_map(|entry| entry.get("repo").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let queued_repos: Vec<Option<&str>> = queued_repos
        .iter()
        .map(|repo| Some(repo.as_str()))
        .collect();
    ensure_license_acknowledged_for_source_inner(
        &state,
        &queued_repos,
        None,
        payload.license_acknowledged,
        Some(&model_id),
    )
    .await?;

    let mut job_payload = build_model_download_job_payload(
        &model,
        &model_id,
        &download,
        requested_variant,
        true,
        payload.license_acknowledged,
        &state.settings.data_dir,
    )?;
    if let Some((variant, derivation)) = &derivation {
        job_payload.insert(
            LOCAL_DERIVATION_PAYLOAD_KEY.to_owned(),
            json!({
                "variant": variant,
                "fromVariant": derivation.from_variant,
                "conversion": derivation.conversion,
                "weightsFile": derivation.weights_file,
                "weightsBytes": derivation.weights_bytes,
                "weightsSha256": derivation.weights_sha256,
            }),
        );
    }

    // Co-requisites (sc-9696): dependencies that must install ALONGSIDE the primary — e.g. the PiD
    // decoder's shared gemma-2-2b-it caption encoder, or 10Eros's cond_safe distill LoRA. Without
    // them the feature silently no-ops (for PiD, `resolve_pid_weights` falls back to the native VAE
    // with no error). The catalog already filtered `downloads` to this OS, so every co-requisite
    // here applies. Each is queued as its own ModelDownload job (the worker is one-repo-per-job);
    // the catalog reports the entry installed only once all of them are cached
    // (`install_state_for`). `include_family: false` because a co-requisite (e.g. a text encoder)
    // is a different artifact than the model's primary checkpoint and must not be reconciled
    // against the model's family.
    let requested_gpu = requested_gpu_or_auto(payload.requested_gpu);
    for co_requisite in &co_requisites {
        if co_requisite_satisfied_by_exact_legacy_rename(&state.settings.data_dir, co_requisite) {
            continue;
        }
        let co_payload = build_model_download_job_payload(
            &model,
            &model_id,
            co_requisite,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        let binding = catalog_download_binding(
            &model_id,
            &model_id,
            "download",
            co_requisite,
            &co_payload,
            None,
            None,
        );
        queue_catalog_download(state.clone(), co_payload, requested_gpu.clone(), binding).await?;
    }

    // The primary job is the one returned to the caller (its id is what the download UI tracks).
    let mut binding = catalog_download_binding(
        &model_id,
        &model_id,
        "download",
        &download,
        &job_payload,
        None,
        None,
    );
    binding["selectedDownload"] = selected_download;
    let job = queue_catalog_download(state, job_payload, requested_gpu, binding).await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// A purpose the entry declares no conditional components for (sc-23002).
pub(crate) const CONDITIONAL_COMPONENTS_NOT_DECLARED_CODE: &str =
    "conditional_components_not_declared";

/// `POST /api/v1/models/:model_id/conditional-components/:purpose/download` (sc-23002) — acquire the
/// pinned components a model needs only for `purpose` (YuE2's SheetSage2 + MERT-v2-FullSong cover
/// closure). They are never `downloads[]` rows, so no install reaches them; this is the one door, and
/// it applies the entry's gates exactly as the model install does:
///
/// * a `blocked` component refuses the whole purpose (403 `component_blocked`, reason + unblock);
/// * an entry that requires a licence acknowledgment refuses without one (403
///   `license_acknowledgment_required`) and records it server-side when asserted, for the entry's
///   CURRENT terms — the same acknowledgment YuE2 jobs re-check at execution;
/// * the repo-keyed half of the gate runs over the repos it will queue, as on every other door.
///
/// Each component not already installed is queued as its own `model_download` job (the worker is
/// one repo per job) at its pinned revision and file list, carrying the acknowledgment so a retry
/// re-validates. Components already on disk are reported, not re-fetched.
pub(crate) async fn create_conditional_components_download_jobs(
    State(state): State<AppState>,
    Path((model_id, purpose)): Path<(String, String)>,
    ApiJson(payload): ApiJson<ConditionalComponentsDownloadRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    use sceneworks_core::model_usage_policy::{
        conditional_component_downloads, ConditionalComponentsError,
    };
    let model = require_model_by_id(&state, &model_id).await?;
    let rows = conditional_component_downloads(&model, &purpose).map_err(|error| match error {
        ConditionalComponentsError::NotDeclared { .. } => ApiError::typed(
            StatusCode::NOT_FOUND,
            format!("Model '{model_id}': {error}."),
            CONDITIONAL_COMPONENTS_NOT_DECLARED_CODE,
            json!({ "purpose": purpose }),
        ),
        ConditionalComponentsError::Blocked { ref blocked, .. } => ApiError::typed(
            StatusCode::FORBIDDEN,
            format!("Model '{model_id}': {error}"),
            COMPONENT_BLOCKED_CODE,
            json!({
                "purpose": purpose,
                "blocked": blocked.iter().map(|(component_id, reason, unblock)| json!({
                    "componentId": component_id, "reason": reason, "unblock": unblock,
                })).collect::<Vec<_>>(),
            }),
        ),
    })?;
    if model_requires_license_acknowledgment(&model) {
        if !payload.license_acknowledged
            && crate::yue2_jobs::current_license_acknowledgment(&state, &model)
                .await?
                .is_none()
        {
            return Err(ApiError {
                status: StatusCode::FORBIDDEN,
                detail: format!(
                    "Model '{model_id}' requires accepting its license before its '{purpose}' \
                     components download. Accept the license on the Models screen, or send \
                     `licenseAcknowledged: true` to assert that the user has accepted it."
                ),
                code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
                context: None,
            });
        }
        if payload.license_acknowledged {
            crate::yue2_jobs::record_license_acknowledgment(&state, &model, "download").await?;
        }
    }
    let repos: Vec<String> = rows
        .iter()
        .filter_map(|row| row.get("repo").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let repo_refs: Vec<Option<&str>> = repos.iter().map(|repo| Some(repo.as_str())).collect();
    ensure_license_acknowledged_for_source_inner(
        &state,
        &repo_refs,
        None,
        payload.license_acknowledged,
        Some(&model_id),
    )
    .await?;
    let requested_gpu = requested_gpu_or_auto(payload.requested_gpu);
    let mut jobs = Vec::new();
    let mut components = Vec::new();
    for row in &rows {
        let component_id = row.get("componentId").cloned().unwrap_or(Value::Null);
        let presence = if model_id.starts_with("moss_image_") {
            pinned_component_presence(&state.settings.data_dir, row)
        } else {
            co_requisite_presence(&state.settings.data_dir, row)
        };
        if presence == CoRequisitePresence::Installed {
            components.push(json!({
                "componentId": component_id,
                "repo": row.get("repo"),
                "revision": row.get("revision"),
                "status": "installed",
            }));
            continue;
        }
        let job_payload = build_model_download_job_payload(
            &model,
            &model_id,
            row,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        let binding = catalog_download_binding(
            &model_id,
            &model_id,
            "conditional",
            row,
            &job_payload,
            Some(&purpose),
            None,
        );
        let job =
            queue_catalog_download(state.clone(), job_payload, requested_gpu.clone(), binding)
                .await?;
        components.push(json!({
            "componentId": component_id,
            "repo": row.get("repo"),
            "revision": row.get("revision"),
            "status": "queued",
            "jobId": job.id,
        }));
        jobs.push(public_job_snapshot(job));
    }
    Ok((
        StatusCode::CREATED,
        Json(json!({ "purpose": purpose, "jobs": jobs, "components": components })),
    ))
}

/// Stamp each conditional component row of a catalog entry with its install state (sc-23002) and
/// summarize each purpose — `conditionalPurposes.<purpose>.installState` is `installed` only when
/// every component the purpose needs is, `incomplete` when any is partly on disk, else `missing`,
/// and `blocked` when any carries a block — so a client can tell whether a cover from a recording
/// can run without guessing from file names.
fn apply_conditional_component_state(object: &mut JsonObject, data_dir: &FsPath) {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let Some(rows) = object
        .get_mut("conditionalComponents")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let mut purposes: std::collections::BTreeMap<String, (bool, bool, bool)> =
        std::collections::BTreeMap::new();
    for row in rows.iter_mut() {
        let presence = if row
            .get("componentId")
            .and_then(Value::as_str)
            .is_some_and(|id| id.starts_with("moss-dmd-") || id == "moss-image-edit")
        {
            pinned_component_presence(data_dir, row)
        } else {
            co_requisite_presence(data_dir, row)
        };
        let blocked = row.get("blocked").is_some();
        let state = match presence {
            CoRequisitePresence::Installed => "installed",
            CoRequisitePresence::Incomplete => "incomplete",
            CoRequisitePresence::Absent => "missing",
        };
        for purpose in row
            .get("requiredFor")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let entry = purposes
                .entry(purpose.to_owned())
                .or_insert((true, false, false));
            entry.0 &= presence == CoRequisitePresence::Installed;
            entry.1 |= presence == CoRequisitePresence::Incomplete;
            entry.2 |= blocked;
        }
        if let Some(row) = row.as_object_mut() {
            row.insert("installState".to_owned(), json!(state));
        }
    }
    let summary: JsonObject = purposes
        .into_iter()
        .map(|(purpose, (installed, incomplete, blocked))| {
            let state = if installed {
                "installed"
            } else if incomplete {
                "incomplete"
            } else {
                "missing"
            };
            (
                purpose,
                json!({ "installState": state, "blocked": blocked }),
            )
        })
        .collect();
    object.insert("conditionalPurposes".to_owned(), Value::Object(summary));
}

/// The install state of `model`'s `purpose` components right now (sc-23002): `Ok(())` when every
/// one is installed; otherwise the component ids that are not.
pub(crate) fn conditional_components_missing(
    data_dir: &FsPath,
    model: &Value,
    purpose: &str,
) -> Result<Vec<Value>, sceneworks_core::model_usage_policy::ConditionalComponentsError> {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let rows =
        sceneworks_core::model_usage_policy::conditional_component_downloads(model, purpose)?;
    Ok(rows
        .into_iter()
        .filter(|row| co_requisite_presence(data_dir, row) != CoRequisitePresence::Installed)
        .map(|row| {
            json!({
                "componentId": row.get("componentId"),
                "repo": row.get("repo"),
                "revision": row.get("revision"),
            })
        })
        .collect())
}

/// Source validity is independent of aggregate resident-component readiness. The store reuses its
/// sealed source stamps and rehashes only changed/racy entries, as at the existing install probe.
fn imported_source_present(entry: &JsonObject, data_dir: &FsPath) -> bool {
    let Some(checkpoint_id) = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
    else {
        return false;
    };
    let Ok(resolved) = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir)
        .resolve(checkpoint_id)
    else {
        return false;
    };
    if sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(entry) {
        sceneworks_core::minimax_h3_import::verify_entry_assembly(entry, &resolved.plan).is_ok()
    } else {
        entry
            .get("importPlan")
            .and_then(|binding| binding.get("planId"))
            .and_then(Value::as_str)
            .is_none_or(|plan_id| resolved.plan.plan_id == plan_id)
    }
}

/// Stable row identity, including legacy resident companions without a component name.
fn imported_component_id(row: &Value, index: usize) -> String {
    row.get("componentId")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("resident-{index}"))
}

/// Unknown identifiers and alternate tiers never fall back to the primary DiT download. Resolve
/// against BOTH the persisted closure and the compatible base's authorized resident rows.
fn bound_import_component_download(
    model: &Value,
    base: &Value,
    component_id: &str,
    requested_variant: Option<&str>,
    data_dir: &FsPath,
    allow_installed: bool,
) -> Result<Value, String> {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let entry = model
        .as_object()
        .ok_or("Imported component repair requires an import")?;
    if !sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(entry)
        || !imported_source_present(entry, data_dir)
    {
        return Err(
            "Repair the imported source by relinking, rescanning or re-importing it first".into(),
        );
    }
    let tier = sceneworks_core::minimax_h3_import::selected_import_component_tier(entry)?;
    if requested_variant.is_some_and(|requested| requested != tier) {
        return Err("Component repair must retain the imported assembly's selected tier".into());
    }
    let rows = model
        .get("downloads")
        .and_then(Value::as_array)
        .ok_or("The import has no bound resident components")?;
    let candidates: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(index, row)| imported_component_id(row, *index) == component_id)
        .collect();
    if candidates.len() != 1 {
        return Err("Unknown or ambiguous imported resident component".into());
    }
    let (_, row) = candidates[0];
    let expected = sceneworks_core::minimax_h3_import::import_component_downloads(
        entry,
        base,
        Some(tier),
        std::env::consts::OS,
    )?;
    if !is_co_requisite_download(row) || !expected.contains(row) {
        return Err(
            "The bound component no longer matches the compatible base; re-import the checkpoint"
                .into(),
        );
    }
    if !allow_installed
        && imported_component_presence(data_dir, row) == CoRequisitePresence::Installed
    {
        return Err("This imported resident component is already installed".into());
    }
    Ok(row.clone())
}

fn ensure_model_downloadable(model: &Value) -> Result<(), ApiError> {
    ensure_model_not_cleanup_only(model)?;
    if model.get("downloadable").and_then(Value::as_bool) == Some(false) {
        return Err(ApiError::bad_request(
            "This model is not available for download.",
        ));
    }
    Ok(())
}

fn ensure_model_not_cleanup_only(model: &Value) -> Result<(), ApiError> {
    if model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true) {
        return Err(ApiError::bad_request(
            "This model is retained only so its existing local files can be removed. No download, conversion, or generation route is available on this platform.",
        ));
    }
    Ok(())
}

/// Build the worker `ModelDownload` job payload for one `download` entry of `model`. Factored out
/// (sc-9696) so the primary download and each co-requisite share identical payload shaping.
/// `explicit_variant` records a request-selected quant tier (falling back to the entry's own
/// `variant`); `include_family` forwards the model's declared family for the worker's post-download
/// family reconcile (sc-1663) — pass `false` for co-requisites, whose weights are a different artifact
/// than the model's primary checkpoint.
const MEMORY_CALIBRATION_PROVENANCE_REQUIRED: &str = "memoryCalibrationProvenanceRequired";

fn catalog_requires_memory_calibration_provenance(model: &Value) -> bool {
    model
        .get("mlx")
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(Value::as_array)
        .is_some_and(|calibrations| !calibrations.is_empty())
}

fn insert_memory_calibration_provenance_requirement(
    job_payload: &mut JsonObject,
    model: &Value,
    primary_artifact: bool,
) {
    job_payload.insert(
        MEMORY_CALIBRATION_PROVENANCE_REQUIRED.to_owned(),
        Value::Bool(primary_artifact && catalog_requires_memory_calibration_provenance(model)),
    );
}

fn build_model_download_job_payload(
    model: &Value,
    model_id: &str,
    download: &Value,
    explicit_variant: Option<&str>,
    include_family: bool,
    license_acknowledged: bool,
    data_dir: &FsPath,
) -> Result<JsonObject, ApiError> {
    let repo = required_string_field(download, "repo")?.to_owned();
    let files = download
        .get("files")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut job_payload = JsonObject::new();
    job_payload.insert("modelId".to_owned(), Value::String(model_id.to_owned()));
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id)
                .to_owned(),
        ),
    );
    // The cold install-time digest is authorized only by the server's catalog entry. The typed
    // client request cannot supply or override this flag, and co-requisites stay inert because the
    // calibration artifact identity is the primary checkpoint.
    insert_memory_calibration_provenance_requirement(&mut job_payload, model, include_family);
    // Preserve legacy explicit assertions verbatim. A download authorized by the
    // server store must recheck that store on retry/duplicate, rather than minting
    // a permanent assertion that could survive withdrawal or changed terms.
    if license_acknowledged {
        job_payload.insert(
            LICENSE_ACKNOWLEDGED_PAYLOAD_KEY.to_owned(),
            Value::Bool(true),
        );
    }
    job_payload.insert(
        "provider".to_owned(),
        Value::String(required_string_field(download, "provider")?.to_owned()),
    );
    job_payload.insert("repo".to_owned(), Value::String(repo.clone()));
    job_payload.insert("files".to_owned(), json!(files));
    if let Some(identity) =
        sceneworks_core::managed_file::ManagedFileIdentity::from_download(download)
            .map_err(ApiError::bad_request)?
    {
        identity.stamp(&mut job_payload);
    }

    // Forward an explicit pinned `revision` (sc-13541) so the worker fetches the exact commit SHA the
    // runtime resolver reads, not `main`. The worker defaults to `main` when this is absent, so
    // omitting `revision` on an entry keeps its behavior unchanged. Required for companion weights a
    // provider resolves via a pinned-SHA `hf_get_pinned` (chatterbox_tts's ve/perth co-requisites): the
    // resolver refuses a non-SHA revision and reads `snapshots/<sha>/`, so a main-branch predownload
    // would land in the wrong snapshot and offline generation would fail.
    if let Some(revision) = download
        .get("revision")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        job_payload.insert("revision".to_owned(), Value::String(revision.to_owned()));
    }
    // Record which quant tier this job installs (sc-8508) so the download record + per-variant
    // install tracking agree on the tier. Falls back to the selected entry's own `variant` when the
    // request omitted one (the default tier may still be a labeled variant).
    if let Some(variant) = explicit_variant
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            download
                .get("variant")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
    {
        job_payload.insert("variant".to_owned(), Value::String(variant));
    }
    // Forward the catalog-declared family so the worker can re-verify the downloaded
    // weights match it (parity with model import). The catalog is project-curated, but
    // a mis-declared family would otherwise silently mismatch downstream adapter
    // selection; the worker reconciles and fails on a confident conflict (sc-1663).
    if include_family {
        if let Some(family) = model.get("family").and_then(Value::as_str) {
            if !family.trim().is_empty() {
                job_payload.insert("family".to_owned(), Value::String(family.to_owned()));
            }
        }
    }
    job_payload.insert(
        "targetDir".to_owned(),
        Value::String(
            data_dir
                .join("models")
                .join(safe_download_dir(&repo))
                .display()
                .to_string(),
        ),
    );
    if let Some(identity) = sceneworks_core::managed_file::authoritative_job_identity(&job_payload)
        .map_err(ApiError::bad_request)?
    {
        identity.stamp(&mut job_payload);
    }
    Ok(job_payload)
}

struct ModelConvertJobPayload<'a> {
    model: &'a Value,
    model_id: &'a str,
    mlx: &'a JsonObject,
    source_repo: &'a str,
    output_dir: &'a FsPath,
    quantize_only: bool,
    request: &'a ModelConvertRequest,
}

fn build_model_convert_job_payload(input: ModelConvertJobPayload<'_>) -> JsonObject {
    let mut job_payload = JsonObject::new();
    job_payload.insert(
        "modelId".to_owned(),
        Value::String(input.model_id.to_owned()),
    );
    job_payload.insert(
        "modelName".to_owned(),
        Value::String(
            input
                .model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(input.model_id)
                .to_owned(),
        ),
    );
    // This boolean is catalog-derived. Repository, source revision, output variant, and fixed tier
    // remain resolver/job facts and are never accepted from the client.
    insert_memory_calibration_provenance_requirement(&mut job_payload, input.model, true);
    job_payload.insert(
        "sourceRepo".to_owned(),
        Value::String(input.source_repo.to_owned()),
    );
    job_payload.insert(
        "outputDir".to_owned(),
        Value::String(input.output_dir.display().to_string()),
    );
    job_payload.insert("dtype".to_owned(), Value::String("bfloat16".to_owned()));
    // Optional converter discriminator + inputs (sc-2235). Default (absent) is the
    // mlx-video Wan converter. A FLUX.2-klein community fine-tune declares
    // `mlx.converter` + the single-file source + the base repo whose
    // VAE/text-encoder/tokenizer are borrowed during assembly.
    if let Some(converter) = input
        .mlx
        .get("converter")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert("converter".to_owned(), Value::String(converter.to_owned()));
    }
    if let Some(source_file) = input
        .mlx
        .get("convertSourceFile")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert(
            "sourceFile".to_owned(),
            Value::String(source_file.to_owned()),
        );
    }
    if let Some(base_repo) = input
        .mlx
        .get("convertBaseRepo")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert("baseRepo".to_owned(), Value::String(base_repo.to_owned()));
    }
    // The quant-tier subdir under `convertBaseRepo` to borrow the base components from (sc-14978):
    // the FLUX.2-klein re-host keeps each tier in its own subdir, so the borrowed
    // VAE/text-encoder/tokenizer live under `<tier>/`, not the snapshot root. Absent for root-layout
    // diffusers bases.
    if let Some(base_subdir) = input
        .mlx
        .get("convertBaseSubdir")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        job_payload.insert(
            "baseSubdir".to_owned(),
            Value::String(base_subdir.to_owned()),
        );
    }
    if input.quantize_only {
        job_payload.insert("quantizeOnly".to_owned(), Value::Bool(true));
    }
    if let Some(bits) = input.request.quantize_bits {
        job_payload.insert("quantizeBits".to_owned(), Value::from(bits));
    }
    if let Some(group_size) = input.request.quantize_group_size {
        job_payload.insert("quantizeGroupSize".to_owned(), Value::from(group_size));
    }
    job_payload
}

#[cfg(test)]
#[path = "models/tests/memory_calibration_job_payload_tests.rs"]
mod memory_calibration_job_payload_tests;

/// Convert a model's native checkpoint into the local MLX format (macOS/Apple
/// Silicon). Only valid for models whose manifest declares `mlx.requiresConversion`
/// (Wan TI2V-5B/I2V-A14B, LTX-2.3 eros, FLUX.2-klein); turnkey MLX models need no conversion. The
/// native source checkpoint must already be downloaded; the worker converts it in-process via the
/// linked `mlx-gen-*` converters, selected by the `mlx.converter` discriminator (sc-3240).
pub(crate) async fn create_model_convert_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelConvertRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    // Derive and confine the route-selected destination before catalog assembly performs any
    // filesystem/cache inspection. Path-shaped IDs must fail at the request boundary rather than
    // falling through to a misleading catalog 404 after that work.
    let output_dir = state
        .settings
        .data_dir
        .join("models")
        .join("mlx")
        .join(&model_id);
    sceneworks_worker::resolve_model_convert_output(
        &state.settings.data_dir,
        &output_dir.display().to_string(),
    )
    .map_err(|error| ApiError::bad_request(error.to_string()))?;

    let model = require_model_by_id(&state, &model_id).await?;
    ensure_model_not_cleanup_only(&model)?;
    let mlx = model
        .get("mlx")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("Model has no MLX variant to convert"))?;
    let requires_conversion = mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let quantize = payload.quantize_bits.is_some();
    // Two sources: models that require conversion read the native checkpoint (convertSourceRepo);
    // turnkey MLX models (a pre-converted bf16 `repo`) carried a legacy in-place quantize path. The
    // native Rust converters don't re-quantize an already-converted dir, so the worker now rejects
    // `quantize_only` with a clear message (sc-3240) — quantize during native conversion instead.
    let (source_repo, quantize_only) = if requires_conversion {
        let repo = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("MLX conversion source repo is not configured"))?;
        (repo.to_owned(), false)
    } else if quantize {
        let repo = mlx
            .get("repo")
            .and_then(Value::as_str)
            .filter(|repo| !repo.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("Model has no MLX repo to quantize"))?;
        (repo.to_owned(), true)
    } else {
        return Err(ApiError::bad_request(
            "Model does not require MLX conversion",
        ));
    };
    // Pre-flight the source weights (sc-14708 follow-up). The worker resolves the same HF snapshot and
    // fails the job when the checkpoint is absent, which reads as a defect to the user: the three Anima
    // variants share `circlestone-labs/Anima`, their per-variant downloads are serialized, and the
    // sibling cards flip to *downloaded* the moment THEIR files land — so converting the variant whose
    // 4 GB DiT is still streaming produced a bare "Anima source DiT is missing." Refuse it here, while
    // the request can still carry an explanation. Only `requiresConversion` reads a native checkpoint;
    // the legacy quantize-only path has its own worker-side rejection.
    if requires_conversion {
        if let Some(download) = store_call(state.clone(), {
            let model_id = model_id.clone();
            move |store, _timeout| store.find_active_model_download_job(&model_id)
        })
        .await?
        {
            let model_name = model
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id.as_str());
            return Err(ApiError::conflict(format!(
                "{model_name} is still downloading ({percent}%). Wait for the download to finish \
                 before converting it.",
                percent = (download.progress.as_f64().unwrap_or(0.0) * 100.0).round() as i64,
            )));
        }
        if let Some(source_file) = mlx
            .get("convertSourceFile")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            match sceneworks_worker::convert_source_state(
                &state.settings.data_dir,
                &source_repo,
                source_file,
            ) {
                sceneworks_worker::ConvertSourceState::Ready => {}
                sceneworks_worker::ConvertSourceState::RepoNotCached => {
                    return Err(ApiError::conflict(format!(
                        "{source_repo} is not downloaded yet. Download it before converting."
                    )));
                }
                sceneworks_worker::ConvertSourceState::FileMissing => {
                    return Err(ApiError::conflict(format!(
                        "{source_file} has not finished downloading from {source_repo}. Wait for \
                         the download to complete, then convert."
                    )));
                }
            }
        }
    }

    let job_payload = build_model_convert_job_payload(ModelConvertJobPayload {
        model: &model,
        model_id: &model_id,
        mlx,
        source_repo: &source_repo,
        output_dir: &output_dir,
        quantize_only,
        request: &payload,
    });

    let job = create_generation_job(
        state,
        JobType::ModelConvert,
        None,
        None,
        job_payload,
        requested_gpu_or_auto(payload.requested_gpu),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

/// Distinct source repositories a model owns. Co-requisite downloads are excluded: they are
/// shared dependencies rather than this model's own source, and resolved-cache entries are
/// selected by *primary* provenance.
pub(crate) fn owned_source_repositories(model: &Value) -> Vec<String> {
    let mut repositories = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| !is_co_requisite_download(entry))
        .filter_map(|entry| entry.get("repo").and_then(Value::as_str))
        .filter(|repo| !repo.trim().is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    repositories.sort();
    repositories.dedup();
    repositories
}

/// Reconciles the resolved-model hot cache with a source that has just been deleted (sc-19710).
///
/// Without this, entries whose authoritative source is gone can never be reclaimed: automatic
/// retention refuses to evict an artifact it cannot re-verify as a second copy, so an orphaned
/// entry would be held `SourceUnverified` forever — exactly the stranded state the epic forbids.
///
/// Reconciliation failures never fail the delete, whose files are already removed; they are logged
/// and returned as warnings so the orphan is surfaced rather than silently stranded.
async fn reconcile_deleted_source(
    state: &AppState,
    selectors: Vec<sceneworks_core::model_artifacts::resolved_cache::SourceLifecycleSelector>,
) -> Vec<String> {
    use sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheStore;

    if selectors.is_empty() {
        return Vec::new();
    }
    let data_dir = state.settings.data_dir.clone();
    // Reconcile whenever a managed store exists, even if the policy was since disabled: entries
    // materialized while it was enabled must still be reconciled rather than orphaned. Never
    // create the store as a side effect of a delete.
    if !data_dir.join("models").join("resolved").exists() {
        return Vec::new();
    }
    let now = sceneworks_core::time::now_unix_seconds().max(0) as u64;
    let outcome = tokio::task::spawn_blocking(move || {
        let store = ResolvedCacheStore::open(&data_dir)?;
        let mut warnings = Vec::new();
        for selector in selectors {
            let report = store.reconcile_removed_source(&selector, now)?;
            for deferred in report.deferred {
                warnings.push(format!(
                    "A cached copy of {} is still in use and will be reclaimed after it is released.",
                    selector.repository
                ));
                tracing::info!(
                    repository = %selector.repository,
                    cache_key = %deferred.cache_key,
                    hold = %deferred.hold,
                    "resolved-cache reconciliation deferred an in-use entry"
                );
            }
            for stranded in report.unmatched_unreadable {
                warnings.push(format!(
                    "A damaged resolved-cache entry ({stranded}) could not be matched to {} and was left in place.",
                    selector.repository
                ));
            }
        }
        Ok::<_, sceneworks_core::model_artifacts::resolved_cache::ResolvedCacheError>(warnings)
    })
    .await;
    match outcome {
        Ok(Ok(warnings)) => warnings,
        Ok(Err(error)) => {
            tracing::warn!(error = %error, "resolved-cache reconciliation failed after a source delete");
            vec![format!(
                "Cached copies of this model could not be reconciled and may still occupy disk: {error}"
            )]
        }
        Err(error) => {
            tracing::warn!(error = %error, "resolved-cache reconciliation task failed");
            vec![
                "Cached copies of this model could not be reconciled and may still occupy disk."
                    .to_owned(),
            ]
        }
    }
}

/// Drop the plan-store documents and cached derivatives a deleted LINKED model owned.
///
/// Keyed on `importPlan.checkpointId` — the same discriminator the scheduler and the worker's
/// route selection use — so a model that is not plan-backed does nothing here. Never touches the
/// linked library and never removes the approved root: only this checkpoint's record, plan,
/// bindings and derivatives.
///
/// Failures become warnings rather than failing the delete: the manifest entry is already gone by
/// this point, so refusing here would leave the catalog and the plan store disagreeing with no way
/// to retry.
async fn forget_linked_checkpoint(state: &AppState, entry: &Value) -> Vec<String> {
    let Some(checkpoint_id) = entry
        .as_object()
        .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
        .map(str::to_owned)
    else {
        return Vec::new();
    };
    let manifest = state
        .settings
        .config_dir
        .join("manifests/user.models.jsonc");
    match load_manifest_entries(state, &manifest, "models").await {
        Ok(entries)
            if entries.iter().any(|entry| {
                entry
                    .as_object()
                    .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
                    == Some(checkpoint_id.as_str())
            }) =>
        {
            return Vec::new()
        }
        Ok(_) => {}
        Err(error) => {
            return vec![format!(
            "Retained the checkpoint plan because remaining model ownership could not be read: {}",
            error.detail
        )]
        }
    }
    let data_dir = state.settings.data_dir.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore::open(&data_dir)?
            .invalidate_checkpoint(&checkpoint_id)
    })
    .await;
    match outcome {
        Ok(Ok(_)) => Vec::new(),
        Ok(Err(error)) => {
            tracing::warn!(error = %error, "forgetting a linked checkpoint failed after a model delete");
            vec![format!(
                "This model's import plan and cached derivatives could not be removed: {error}"
            )]
        }
        Err(error) => {
            tracing::warn!(error = %error, "the linked-checkpoint cleanup task failed");
            vec!["This model's import plan and cached derivatives could not be removed.".to_owned()]
        }
    }
}

pub(crate) async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Query(query): Query<CatalogDeleteQuery>,
) -> Result<Json<Value>, ApiError> {
    let permanent = query.permanent.unwrap_or(false);
    let catalogs = JobCatalogSnapshot::default();
    let model = catalogs
        .models(&state)
        .await?
        .iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .cloned()
        .ok_or_else(|| model_not_found("Model not found"))?;
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    // Peek (not remove) the manifest entry so that if moving the files to the OS
    // trash fails we can leave the catalog untouched and prompt for confirmation.
    let manifest_entry = load_manifest_entries(&state, &manifest_path, "models")
        .await?
        .into_iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id.as_str()));
    let cleanup_source = manifest_entry.as_ref().unwrap_or(&model);
    // Keep a cross-process source guard through the actual trash/delete operation, not merely a
    // preflight. Dense imports have no derivative lease, but are just as in-use as normalized ones.
    let source_guard = cleanup_source
        .as_object()
        .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
        .map(|checkpoint_id| {
            sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(
                &state.settings.data_dir,
            )
            .lock_source_for_removal(checkpoint_id)
        })
        .transpose()
        .map_err(|error| ApiError {
            status: StatusCode::CONFLICT,
            detail: error.to_string(),
            context: None,
            code: Some("checkpoint_source_in_use"),
        })?;
    let allowed_roots = vec![
        state.settings.data_dir.join("models"),
        sceneworks_core::hf_home::model_source_library(&state.settings.data_dir)
            .root()
            .to_owned(),
    ];
    let removal = match remove_whole_model_artifacts(
        catalogs.models(&state).await?,
        &model_id,
        cleanup_source,
        &state.settings.data_dir,
        &allowed_roots,
        permanent,
    )
    .await
    {
        Ok(removal) => {
            if !removal.removed_paths.is_empty() {
                // Invalidate immediately after the filesystem mutation. Later
                // trash-failure and manifest/error returns must not leave the
                // warm install-state snapshot describing paths already moved.
                state.model_catalog_cache.invalidate();
            }
            removal
        }
        Err(error) => {
            // Artifact removal is intentionally incremental. An inspection or
            // unlink error may follow successful earlier removals whose result
            // cannot be recovered from the error, so conservatively refresh.
            state.model_catalog_cache.invalidate();
            return Err(error);
        }
    };
    drop(source_guard);
    // Some owned files could not reach the OS trash and nothing was permanently
    // deleted. Leave the registry entry in place and ask the client to confirm.
    if !permanent && !removal.trash_failed_paths.is_empty() {
        return Ok(Json(json!({
            "id": model_id,
            "kind": "model",
            "trashUnavailable": true,
            "trashFailedPaths": removal.trash_failed_paths,
            "removedManifestEntry": false,
            "removedLocalArtifacts": !removal.removed_paths.is_empty(),
            "removedPaths": removal.removed_paths,
            "retainedPaths": removal.retained_paths,
        })));
    }
    let removed_entry =
        remove_catalog_manifest_entry(&state, &manifest_path, "models", &model_id).await?;
    if removed_entry.is_none() && removal.removed_paths.is_empty() {
        return Err(ApiError::bad_request(
            "Built-in model catalog entries are read-only unless local files are installed",
        ));
    }
    let mut warnings =
        catalog_delete_warnings(&state, "model", &model_id, None, Some(&catalogs)).await?;
    // A plan-backed (linked) model's bytes are not ours, but its plan documents and derivatives
    // are. Without this the manifest entry went away and the record, plan, bindings and every
    // cached derivative stayed behind forever — reachable by nothing, reclaimable by nothing.
    // E6 still holds exactly: this drops SceneWorks-owned documents, never a library file, and
    // never the approved root itself (other checkpoints under it stay valid).
    warnings.extend(forget_linked_checkpoint(&state, cleanup_source).await);
    // The source is gone; reclaim (or surface) every resolved-cache entry that depended on it.
    warnings.extend(
        reconcile_deleted_source(
            &state,
            owned_source_repositories(cleanup_source)
                .into_iter()
                .map(|repository| {
                    sceneworks_core::model_artifacts::resolved_cache::SourceLifecycleSelector {
                        repository,
                        revision: None,
                        tier: None,
                    }
                })
                .collect(),
        )
        .await,
    );
    // ---- checkpoint teardown (epic 20398) ---------------------------------------------------
    // `remove_whole_model_artifacts` removed the BYTES. A plan-backed entry also owns documents
    // under `<data>/checkpoints/` — catalog record, plan, bindings — which that sweep never sees.
    // Left behind for a MANAGED install they make the id permanently un-reimportable: the manifest
    // entry is gone, so this route's `existing_ids` check passes on the next import, and the
    // worker's `ManagedIngest::begin` then refuses forever with `InstallIdTaken` naming a path that
    // no longer exists.
    //
    // Deliberately one block that discriminates on ownership, because the two ownerships need
    // opposite treatment and only ONE of them is this story's.
    let plan_checkpoint_id = cleanup_source
        .as_object()
        .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
        .map(str::to_owned);
    match plan_checkpoint_id.as_deref().map(|checkpoint_id| {
        (
            checkpoint_id,
            sceneworks_core::checkpoint_plan_store::managed_install_id(checkpoint_id),
        )
    }) {
        // MANAGED: SceneWorks owns the install tree, so the record, plan and bindings go with it.
        Some((checkpoint_id, Some(install_id))) => {
            let data_dir = state.settings.data_dir.clone();
            let owned_install_id = install_id.to_owned();
            let owned_checkpoint_id = checkpoint_id.to_owned();
            // Through the DERIVATIVE store, not `CheckpointPlanStore::remove_managed` directly:
            // the plan store's own removal drops the record and `remove_dir_all`s the install tree
            // with nothing asking about derivatives first, so a leased or pinned derivative was
            // orphaned — its source bytes and plan gone, and its `derivedFrom` checkpoint id (the
            // only handle any later invalidation has) no longer resolving to anything. The
            // derivative store removes derivatives FIRST and refuses the whole teardown if one is
            // still in use, which is the same ordering the linked path above already had
            // (sc-20651 feature-end review).
            //
            // But a REFUSAL cannot be honoured HERE, and only here: by this point
            // `remove_whole_model_artifacts` has already deleted the install bytes and the manifest
            // entry is already gone. Holding the plan record back for the derivative's sake would
            // therefore strand the record with no bytes under it, and `ManagedIngest::begin` would
            // refuse this install id FOREVER with `InstallIdTaken` naming a path that no longer
            // exists — the model becomes permanently un-reimportable. So the record is cleared
            // regardless, and the derivative that could not be reclaimed is NAMED in a warning:
            // it is orphaned either way once the bytes are gone, and a named orphan is one the user
            // can find and evict, while a silent one is not. Every other caller of
            // `remove_managed_install` still gets the refuse-first behaviour, which is correct for
            // them: their bytes are still there to hold.
            let outcome = tokio::task::spawn_blocking(move || {
                managed_checkpoint_teardown(&data_dir, &owned_install_id, &owned_checkpoint_id)
            })
            .await;
            let report = match outcome {
                Ok(report) => report,
                Err(error) => ManagedTeardownReport {
                    failure: Some(error.to_string()),
                    orphaned_derivatives: Vec::new(),
                    fallback_failure: Some(error.to_string()),
                },
            };
            // The files are already gone; a failure here cannot un-delete the model, so it is
            // surfaced rather than raised. The warning names the checkpoint because that id is the
            // only handle on the stranded record.
            if let Some(failure) = report.failure {
                tracing::warn!(
                    checkpoint_id = %checkpoint_id,
                    error = %failure,
                    orphaned_derivatives = report.orphaned_derivatives.len(),
                    cleared = report.fallback_failure.is_none(),
                    "managed checkpoint teardown failed after a model delete"
                );
                if let Some(fallback_failure) = report.fallback_failure {
                    warnings.push(format!(
                        "The managed checkpoint {checkpoint_id} could not be removed ({fallback_failure}); re-importing this model id may be refused until it is cleared."
                    ));
                } else {
                    let orphans = if report.orphaned_derivatives.is_empty() {
                        "none".to_owned()
                    } else {
                        report.orphaned_derivatives.join(", ")
                    };
                    warnings.push(format!(
                        "The managed checkpoint {checkpoint_id} was removed, but its cached derivatives could not be reclaimed ({failure}); orphaned resolved-cache entries: {orphans}. Unpin or release them to free that space."
                    ));
                }
            }
        }
        // LINKED (or any non-managed checkpoint id): the bytes are the user's own library copy,
        // which this route never owned, so `remove_managed` must not be reached — it would be the
        // wrong operation on the wrong tree. Invalidating a linked entry's plan and the derivatives
        // compiled from it is sc-20635's work; this arm is where it lands.
        Some((_, None)) => {}
        None => {}
    }
    let policy = if removed_entry.is_some() {
        "Removed the model registry entry and SceneWorks-owned local model files."
    } else {
        "Built-in model catalog entries are retained; SceneWorks-owned local model files were removed."
    };
    Ok(Json(json!({
        "id": model_id,
        "kind": "model",
        "trashed": !permanent,
        "removedManifestEntry": removed_entry.is_some(),
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
        "warnings": warnings,
        "policy": policy,
    })))
}

/// What the managed checkpoint teardown did after a model delete, when it did not simply succeed.
struct ManagedTeardownReport {
    /// The refusal that stopped the derivatives-first teardown. `None` when it succeeded outright,
    /// which is the only case that produces no warning.
    failure: Option<String>,
    /// The derivative cache keys still present after that refusal — the entries the delete could
    /// not reclaim, named so the user can find them.
    orphaned_derivatives: Vec<String>,
    /// The record-clearing fallback's OWN refusal. `Some` means the plan record really is stranded
    /// and the install id may stay un-reimportable, which is a materially different warning.
    fallback_failure: Option<String>,
}

/// Tear down a managed install after its bytes and its manifest entry are already gone.
///
/// Derivatives first (so a derivative in use is reclaimed cleanly when it can be), then — if that
/// refuses — the plan record UNCONDITIONALLY, because leaving it behind is what poisons the install
/// id forever. Blocking throughout; callers must run it off the async runtime.
fn managed_checkpoint_teardown(
    data_dir: &std::path::Path,
    install_id: &str,
    checkpoint_id: &str,
) -> ManagedTeardownReport {
    use sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore;

    let store = CheckpointDerivativeStore::open(data_dir);
    let failure = match &store {
        Ok(store) => match store.remove_managed_install(install_id) {
            Ok(_) => {
                return ManagedTeardownReport {
                    failure: None,
                    orphaned_derivatives: Vec::new(),
                    fallback_failure: None,
                }
            }
            Err(error) => error.to_string(),
        },
        Err(error) => error.to_string(),
    };
    let orphaned_derivatives = store
        .as_ref()
        .map(|store| surviving_derivatives(store, checkpoint_id))
        .unwrap_or_default();
    let fallback_failure =
        sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir)
            .remove_managed(install_id)
            .err()
            .map(|error| error.to_string());
    ManagedTeardownReport {
        failure: Some(failure),
        orphaned_derivatives,
        fallback_failure,
    }
}

/// The derivative cache keys still recorded as produced from `checkpoint_id`.
///
/// Read AFTER a refused teardown, so it names exactly the entries that were not reclaimed. An
/// enumeration failure yields an empty list rather than an error: this only enriches a warning that
/// is already being raised, and losing the warning would be worse than losing the names.
fn surviving_derivatives(
    store: &sceneworks_core::checkpoint_derivative::CheckpointDerivativeStore,
    checkpoint_id: &str,
) -> Vec<String> {
    store
        .cache()
        .enumerate()
        .map(|entries| {
            entries
                .into_iter()
                .filter_map(|summary| {
                    let metadata = summary.metadata?;
                    (metadata.production.is_derived()
                        && metadata.derived_from.as_deref() == Some(checkpoint_id))
                    .then_some(metadata.cache_key)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The file scopes `model` OWNS inside `repo` — the union of its PRIMARY (non-co-requisite) download
/// entries that point at `repo`. Co-requisite rows are deliberately excluded; see the comment on the
/// filter below for why that is what makes this "owned" rather than merely "referenced".
///
/// `None` when ANY of those primaries declares no `files` filter: that is a claim on the WHOLE repo,
/// which cannot be expressed as a scoped removal, so the caller keeps the blanket path removal rather
/// than silently reclaiming less than the user asked for.
fn model_repo_file_scopes(model: &Value, repo: &str) -> Option<Vec<String>> {
    let mut scopes = Vec::new();
    for download in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        // Co-requisites are EXCLUDED, and that is what makes this "owned" rather than "referenced".
        // A co-requisite may be shared by several models and is never removed by a model delete
        // (the `coRequisite` schema note says so outright). sc-19573 made this load-bearing: both
        // MiniMax-H3 entries now co-require the OTHER's DiT partition from the shared repo, so
        // counting co-requisite rows here would make an entry claim its sibling's subtree as its own
        // — and, symmetrically in `other_entries_repo_file_scopes`, retain its own. The two sets
        // would cover everything, `selected && !retained` would never hold, and a whole-model delete
        // would silently reclaim nothing. `delete_model_variant`'s `retained_files` already applies
        // exactly this filter for exactly this reason.
        .filter(|entry| !is_co_requisite_download(entry))
        .filter(|entry| entry.get("repo").and_then(Value::as_str) == Some(repo))
    {
        let files = string_array_field(download, "files");
        if files.is_empty() {
            return None;
        }
        for file in files {
            if !scopes.contains(&file) {
                scopes.push(file);
            }
        }
    }
    (!scopes.is_empty()).then_some(scopes)
}

/// The file scopes every catalog entry OTHER than `model_id` claims inside `repo` (sc-19078).
///
/// Thirteen catalog groups put two or more entries in ONE Hugging Face repo, and a whole-model delete
/// resolves that repo's cache path ([`model_artifact_paths`]) — so removing one entry took the SIBLING
/// entry's bytes with it. For most of those groups the two entries name the same `files`, so the
/// removal at least matched what both wanted. MiniMax-H3 is the group where it becomes destructive:
/// `minimax_h3` owns `{tier}/transformer` and `minimax_h3_ref` owns `{tier}/transformer_ref` inside
/// `SceneWorks/minimax-h3-mlx` — DIFFERENT weights, up to 66.3 GB per tier — so deleting the
/// text-to-video entry wiped an installed reference model the user never asked to remove.
///
/// Co-requisite rows are included: a sibling's shared component living in the same repo is still bytes
/// that sibling needs. Nothing here is conditioned on the sibling being INSTALLED — an installed
/// sibling is exactly the case that matters, and for a sibling that is absent every one of these
/// patterns matches no file on disk, so retaining them costs the delete nothing.
///
/// The two kinds are returned SEPARATELY, because the caller may only ever subtract from one of them.
/// See [`SiblingRepoScopes`].
#[derive(Debug, Default, PartialEq, Eq)]
struct SiblingRepoScopes {
    /// Scopes a sibling entry claims as its OWN primary weights. These are the sc-19078 subject and
    /// are NEVER subtracted from: the sibling is a separate installed model, and unlinking bytes it
    /// names as its own primary download is precisely the data loss this function exists to prevent.
    primaries: Vec<String>,
    /// Scopes a sibling entry claims only as a CO-REQUISITE — a shared component, or (MiniMax-H3)
    /// the deleted entry's own partition that the sibling's engine also opens. Overlap with the
    /// deleted entry's own primaries is subtracted from THIS half only, so that "delete this model"
    /// still frees the model's own weights instead of no-opping.
    co_requisites: Vec<String>,
}

impl SiblingRepoScopes {
    fn is_empty(&self) -> bool {
        self.primaries.is_empty() && self.co_requisites.is_empty()
    }

    /// The retained set for [`remove_tier_artifacts`]: every sibling primary, untouched, plus the
    /// co-requisite scopes that do NOT overlap `own_files`.
    ///
    /// Subtracting the overlap from the co-requisite half ONLY is the whole point of the split
    /// (sc-19573 review). Subtracting it from the union instead collapses the retained set to `[]`
    /// for every group whose sibling names the SAME primary `files` — `flux_dev`/`pulid_flux_dev`,
    /// `z_image_turbo`/`z_image_edit`, `bernini`/`bernini_image`, `realvisxl`/`instantid_realvisxl`,
    /// `qwen_image_edit_2511`/`_lightning`, `ideogram_4`/`ideogram_4_turbo` — and strips the shared
    /// text-encoder/VAE out of the `anima_*` trio. `remove_tier_artifacts`'s `selected && !retained`
    /// would then unlink the sibling's blobs with `permanent=true`: tens of GB, unrecoverable without
    /// re-download, i.e. exactly the sc-19078 defect re-introduced.
    fn retained_files(&self, own_files: &[String]) -> Vec<String> {
        let mut retained = self.primaries.clone();
        for file in &self.co_requisites {
            if !own_files.contains(file) && !retained.contains(file) {
                retained.push(file.clone());
            }
        }
        retained
    }
}

fn other_entries_repo_file_scopes(
    catalog: &[Value],
    model_id: &str,
    repo: &str,
) -> SiblingRepoScopes {
    let mut scopes = SiblingRepoScopes::default();
    for entry in catalog
        .iter()
        .filter(|entry| entry.get("id").and_then(Value::as_str) != Some(model_id))
    {
        for download in entry
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|download| download.get("repo").and_then(Value::as_str) == Some(repo))
        {
            let bucket = if is_co_requisite_download(download) {
                &mut scopes.co_requisites
            } else {
                &mut scopes.primaries
            };
            for file in string_array_field(download, "files") {
                if !bucket.contains(&file) {
                    bucket.push(file);
                }
            }
        }
    }
    scopes
}

/// Remove a whole model's owned artifacts for [`delete_model`], keeping a shared-repo sibling's bytes.
///
/// The default path is unchanged: every path in [`model_artifact_paths`] is removed wholesale, which
/// is right for the ~80 catalog entries that own their download repo outright. When the primary repo is
/// ALSO claimed by another catalog entry, the repo's two storage locations (the app-managed mirror dir
/// and the Hugging Face hub cache) are removed SCOPED instead — via the same
/// [`remove_tier_artifacts`] machinery the per-tier delete uses, with the sibling's declared files as
/// the retained set — so this entry's own subtrees and their exclusive blobs go and the sibling's stay.
/// Every other artifact path (a manifest `paths.model`, an imported `source.path`) is entry-exclusive
/// and still removed wholesale.
///
/// An entry that declares NO file scope inside a shared repo ([`model_repo_file_scopes`] → `None`)
/// keeps the blanket removal: it claims the whole repo, so there is no honest narrower scope, and
/// today's behavior is preserved rather than quietly reclaiming nothing.
async fn remove_whole_model_artifacts(
    catalog: &[Value],
    model_id: &str,
    cleanup_source: &Value,
    data_dir: &FsPath,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<ArtifactRemoval, ApiError> {
    let all_paths = model_artifact_paths(cleanup_source, data_dir);
    let shared = model_download(cleanup_source)
        .and_then(|download| {
            download
                .get("repo")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .and_then(|repo| {
            let siblings = other_entries_repo_file_scopes(catalog, model_id, &repo);
            let own = model_repo_file_scopes(cleanup_source, &repo)?;
            (!siblings.is_empty()).then_some((repo, own, siblings))
        });
    let Some((repo, own_files, sibling_scopes)) = shared else {
        return remove_owned_artifacts(all_paths, allowed_roots, permanent).await;
    };
    // sc-19573 — a sibling may co-require THIS entry's own subtree, and then the retained set would
    // cover everything the delete selected: `selected && !retained` would never hold and the user's
    // explicit "delete this model" would silently reclaim zero bytes.
    //
    // Both MiniMax-H3 entries are in that shape now — each co-requires the other's DiT partition,
    // because the engine opens both on every load. Removing the overlap resolves it in the honest
    // direction: the delete does what it says, and the sibling entry's install state drops to
    // incomplete + `repairAvailable`, which is the truth (it can no longer load) and is re-fetchable
    // in one click. Retaining instead would leave a user who asked to free 56 GB with a no-op and no
    // explanation.
    //
    // The subtraction applies to the CO-REQUISITE half ONLY ([`SiblingRepoScopes::retained_files`]).
    // A sibling's PRIMARY scopes are never subtracted — six catalog groups pair two entries that name
    // the IDENTICAL primary `files` in one repo, and subtracting there would empty the retained set
    // and let this delete unlink the sibling's own weights.
    //
    // Computed AFTER the `is_empty` gate above, so an entry whose only sibling claim is an overlap
    // still takes the SCOPED path rather than falling back to the blanket whole-repo removal that
    // sc-19078 exists to prevent.
    let sibling_files = sibling_scopes.retained_files(&own_files);

    let managed_dir = data_dir.join("models").join(safe_download_dir(&repo));
    let repo_cache = huggingface_repo_cache_path(data_dir, &repo);
    // Everything that is NOT the shared repo's storage: still this entry's alone, still removed whole.
    let exclusive = all_paths
        .into_iter()
        .filter(|path| {
            path != &managed_dir && repo_cache.as_ref().is_none_or(|cache| path != cache)
        })
        .collect::<Vec<_>>();
    let mut removal = remove_owned_artifacts(exclusive, allowed_roots, permanent).await?;
    let scoped = remove_tier_artifacts(
        repo_cache,
        Some(managed_dir),
        &own_files,
        &sibling_files,
        allowed_roots,
        permanent,
    )
    .await?;
    removal.removed_paths.extend(scoped.removed_paths);
    removal.retained_paths.extend(scoped.retained_paths);
    removal.trash_failed_paths.extend(scoped.trash_failed_paths);
    Ok(removal)
}

/// Delete ONE installed quant tier of a model and reclaim its disk, leaving the other tiers
/// (and the catalog entry) intact (sc-12024, epic 8506). The counterpart to per-tier download
/// (sc-8509): a user who fetched q8 to A/B against q4 can drop the unused tier without nuking
/// the whole model. Unlike `delete_model` — which removes the whole repo dir AND the registry
/// entry — this is scoped to the tier's `files` and never touches the manifest, so the model
/// stays catalogued and re-downloadable; deleting the last remaining tier simply flips it back
/// to not-installed. Unlike `delete_model` this deletes PERMANENTLY (never the OS trash): a tier is
/// many loose HF-cache blobs + snapshot symlinks, and trashing them one-by-one drove a macOS
/// per-file permission-prompt loop ("you don't have permission to access some of the items") — and a
/// tier isn't restorable from loose trashed blobs anyway (sc-12088). Same ownership guard as
/// `delete_model` (`<data>/models` + the HF hub cache).
pub(crate) async fn delete_model_variant(
    State(state): State<AppState>,
    Path((model_id, variant)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let variant = variant.trim().to_ascii_lowercase();
    let model = require_model_by_id(&state, &model_id).await?;
    let data_dir = &state.settings.data_dir;
    let allowed_roots = vec![
        data_dir.join("models"),
        sceneworks_core::hf_home::model_source_library(data_dir)
            .root()
            .to_owned(),
    ];
    // A tier lives in one of two storage shapes. Download-matrix models (`hasVariantMatrix`) keep the
    // tier as a `files`-filtered slice of a shared HF cache repo (sc-12024); convert-at-install
    // models (Anima) keep it as a real `<converted>/<tier>/` dir emitted by one convert job
    // (sc-12025). Resolve whichever this model uses; a variant that is neither has nothing to delete.
    let derived = model_download_for_variant(&model, &variant).filter(|download| {
        sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(download)
    });
    let removal_result = if let Some(download) = derived {
        // sc-22998: a locally derived tier owns ONLY its derived snapshot. Its row's `files` are the
        // original's, which belong to the `fromVariant` tier and are never touched here.
        let snapshot = sceneworks_core::model_artifacts::artifact_selection::local_derivation(
            &download,
        )
        .and_then(|derivation| {
            sceneworks_core::model_artifacts::artifact_selection::local_derivation_snapshot_dir(
                data_dir,
                &model_id,
                &variant,
                &derivation,
            )
        })
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "Tier '{variant}' declares a malformed local derivation"
            ))
        })?;
        if !snapshot.is_dir() {
            return Err(ApiError::bad_request(format!(
                "Tier '{variant}' is not installed"
            )));
        }
        let reclaimable = converted_tier_real_bytes(&snapshot);
        remove_owned_artifacts(vec![snapshot], &allowed_roots, true)
            .await
            .map(|removal| TierRemoval {
                reclaimed_bytes: if removal.removed_paths.is_empty() {
                    0
                } else {
                    reclaimable
                },
                removed_paths: removal.removed_paths,
                retained_paths: removal.retained_paths,
                trash_failed_paths: removal.trash_failed_paths,
            })
    } else if let Some(download) = model_download_for_variant(&model, &variant) {
        let scope = tier_delete_scope(&model, &download, &variant)?;
        let repo_cache = huggingface_repo_cache_path(data_dir, &scope.repo);
        let managed_dir = Some(data_dir.join("models").join(safe_download_dir(&scope.repo)));
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_tier_artifacts(
            repo_cache,
            managed_dir,
            &scope.files,
            &scope.retained_files,
            &allowed_roots,
            true,
        )
        .await
    } else if model_has_convert_tier(&model, &variant) {
        // Convert-at-install: the tier is a real dir under the converted MLX tree. Prefer the
        // catalog's resolved `mlxConvertedPath`; fall back to the canonical convert output dir.
        let converted = model
            .get("mlxConvertedPath")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("models").join("mlx").join(&model_id));
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_converted_tier(converted.join(&variant), &allowed_roots, true).await
    } else {
        return Err(ApiError::bad_request(format!(
            "Model does not advertise a '{variant}' quant tier"
        )));
    };
    let removal = match removal_result {
        Ok(removal) => removal,
        Err(error) => {
            // Tier removal is also incremental (snapshot links, blobs, then
            // managed paths). Refresh conservatively if any later step errors.
            state.model_catalog_cache.invalidate();
            return Err(error);
        }
    };
    // Invalidate before every post-removal early return. A successful removal
    // may have changed installState/variantStates even when the route later
    // decides there was no independently reclaimable logical tier.
    state.model_catalog_cache.invalidate();
    let shared_logical_tier = removal.removed_paths.is_empty()
        && !removal.retained_paths.is_empty()
        && model_download_for_variant(&model, &variant).is_some();
    if removal.removed_paths.is_empty() && !shared_logical_tier {
        return Err(ApiError::bad_request(format!(
            "Tier '{variant}' is not installed"
        )));
    }
    // A single-tier delete only orphans resolved-cache entries built from that tier; entries for
    // the model's other tiers stay valid, so the selector is scoped by tier.
    let cache_warnings = reconcile_deleted_source(
        &state,
        owned_source_repositories(&model)
            .into_iter()
            .map(|repository| {
                sceneworks_core::model_artifacts::resolved_cache::SourceLifecycleSelector {
                    repository,
                    revision: None,
                    tier: Some(variant.clone()),
                }
            })
            .collect(),
    )
    .await;
    Ok(Json(json!({
        "id": model_id,
        "variant": variant,
        "kind": "model-variant",
        "warnings": cache_warnings,
        // Permanent delete: no OS trash, no undo (sc-12088).
        "trashed": false,
        // A tier delete NEVER removes the registry entry: the model stays in the catalog so the
        // tier can be re-downloaded. Emitted false so the web keeps the model card in place.
        "removedManifestEntry": false,
        "removedLocalArtifacts": !removal.removed_paths.is_empty(),
        "reclaimedBytes": removal.reclaimed_bytes,
        "reclaimedLabel": format_bytes(removal.reclaimed_bytes),
        "removedPaths": removal.removed_paths,
        "retainedPaths": removal.retained_paths,
    })))
}

/// What one per-tier delete removes: the repo it lives in, the globs that select its files, and
/// the globs a sibling tier in the SAME repo still owns.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TierDeleteScope {
    pub(crate) repo: String,
    pub(crate) files: Vec<String>,
    pub(crate) retained_files: Vec<String>,
}

/// Whether a tier row with NO `files` scope is still reclaimable on its own (sc-24112): no other
/// download row of the same model — tier, alternate or co-requisite — names its `repo`, so that
/// repo's whole snapshot IS this tier. `qwen_image_2_1`'s bf16 tier is the released upstream
/// snapshot, the only row on `Qwen/Qwen-Image-2.1`, while its q8/q4 live in the SceneWorks re-host;
/// deleting it can strand nothing. A scope-less row on a repo a sibling shares is still the model,
/// not a slice of it, and stays refused.
pub(crate) fn is_sole_repo_tier(model: &Value, download: &Value) -> bool {
    // Only a DECLARED tier: an untagged single-download row is the "default" pseudo-variant, which
    // the per-tier route cannot address at all.
    let tagged = download
        .get("variant")
        .and_then(Value::as_str)
        .is_some_and(|variant| !variant.trim().is_empty());
    let Some(repo) = download
        .get("repo")
        .and_then(Value::as_str)
        .filter(|_| tagged)
    else {
        return false;
    };
    let sharers = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry.get("repo").and_then(Value::as_str) == Some(repo))
        .count();
    sharers == 1
}

/// Scope a per-tier delete of `variant` (its catalog row is `download`).
///
/// A row with a `files` scope deletes that slice and protects every path a sibling logical tier in
/// the same repo still references (Mage-Flow's overlapping load-time tiers reclaim zero bytes
/// rather than corrupting the snapshot the remaining choices use). A row with NO scope is refused
/// ("delete the whole model instead") unless it is the sole row on its repo
/// ([`is_sole_repo_tier`]), in which case the whole repo snapshot is the tier and every file in it
/// goes.
pub(crate) fn tier_delete_scope(
    model: &Value,
    download: &Value,
    variant: &str,
) -> Result<TierDeleteScope, ApiError> {
    let repo = download
        .get("repo")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let files = string_array_field(download, "files");
    if files.is_empty() {
        if is_sole_repo_tier(model, download) {
            return Ok(TierDeleteScope {
                repo,
                files: vec!["*".to_owned()],
                retained_files: Vec::new(),
            });
        }
        return Err(ApiError::bad_request(format!(
            "Tier '{variant}' has no file scope; delete the whole model instead"
        )));
    }
    let retained_files = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| {
            !is_co_requisite_download(entry)
                // sc-22998: a locally derived tier's `files` name the ORIGINAL it is derived from;
                // it does not own them, so deleting the original must not "retain" them for it.
                && !sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
                    entry,
                )
                && entry.get("repo").and_then(Value::as_str) == Some(repo.as_str())
                && entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.eq_ignore_ascii_case(variant))
        })
        .flat_map(|entry| string_array_field(entry, "files"))
        .collect::<Vec<_>>();
    Ok(TierDeleteScope {
        repo,
        files,
        retained_files,
    })
}
/// Result of removing a single quant tier's on-disk artifacts (sc-12024).
#[derive(Default)]
pub(crate) struct TierRemoval {
    /// Paths (tier symlinks/files + their exclusive blobs) moved to the OS trash or unlinked.
    pub(crate) removed_paths: Vec<String>,
    /// Paths left in place because they are not inside a SceneWorks-owned root.
    pub(crate) retained_paths: Vec<String>,
    /// Owned paths that could NOT be moved to the OS trash (recycle bin disabled, unsupported
    /// volume, …). Nothing was deleted for these; the caller prompts before a permanent delete.
    pub(crate) trash_failed_paths: Vec<String>,
    /// Bytes actually reclaimed — the summed size of the data-bearing files/blobs removed.
    pub(crate) reclaimed_bytes: u64,
}

/// Remove ONE quant tier's artifacts from a download-matrix model's storage, reclaiming disk.
///
/// A download-matrix model keeps every tier (bf16/q8/q4) in ONE shared Hugging Face hub-cache
/// repo: the real bytes live in `blobs/<etag>` and each tier's files are relative SYMLINKS into
/// `blobs/` (`download_snapshot_into_cache`, crates/sceneworks-worker/src/downloads.rs). Deleting
/// the tier's snapshot symlinks alone frees nothing — the blobs behind them must go too. This
/// walks every snapshot revision under the repo cache (and the app-managed mirror dir, where a
/// turnkey install lands real files), selects the files matching the tier's `files` globs, and
/// removes those directory entries PLUS the blobs they resolve to — while PROTECTING any blob
/// still referenced by a retained tier (a shared etag). Emptied tier/snapshot dirs, and the whole
/// repo cache dir once no tier's payload remains, are pruned (best-effort; only ever unlinks EMPTY
/// dirs, so a surviving tier is never touched). `reclaimed_bytes` reflects only what actually left
/// disk. An empty `tier_files` is a no-op — the caller must never scope a delete to "everything".
pub(crate) async fn remove_tier_artifacts(
    repo_cache: Option<PathBuf>,
    managed_dir: Option<PathBuf>,
    tier_files: &[String],
    retained_files: &[String],
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<TierRemoval, ApiError> {
    if tier_files.is_empty() {
        return Ok(TierRemoval::default());
    }
    // The directories to scan: every snapshot revision under the HF repo cache, plus the managed
    // mirror dir. Each is scanned independently and files are matched RELATIVE to their scan dir.
    let mut scan_dirs: Vec<PathBuf> = Vec::new();
    if let Some(repo_cache) = repo_cache.as_ref() {
        if huggingface_repo_cache_exists(repo_cache) {
            scan_dirs.extend(huggingface_snapshot_dirs(repo_cache));
        }
    }
    if let Some(managed_dir) = managed_dir.as_ref() {
        if managed_dir.is_dir() {
            scan_dirs.push(managed_dir.clone());
        }
    }

    // Split every file under the scan dirs into this tier's entries vs the retained rest. A
    // retained file's real data (blob) must survive even if a tier symlink shares its etag.
    let mut tier_entries: Vec<PathBuf> = Vec::new();
    let mut retained_reals: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for dir in &scan_dirs {
        for rel in snapshot_files(dir) {
            let abs = dir.join(&rel);
            let selected = tier_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel));
            let retained = retained_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel));
            if selected && !retained {
                tier_entries.push(abs);
            } else {
                if let Ok(real) = tokio::fs::canonicalize(&abs).await {
                    retained_reals.insert(real);
                }
            }
        }
    }

    let mut manifest_retained = Vec::new();
    for dir in &scan_dirs {
        for rel in snapshot_files(dir) {
            if tier_files
                .iter()
                .any(|pattern| pattern_matches(pattern, &rel))
                && retained_files
                    .iter()
                    .any(|pattern| pattern_matches(pattern, &rel))
            {
                manifest_retained.push(dir.join(rel).display().to_string());
            }
        }
    }

    // Build the ordered removal plan: unlink the tier's directory ENTRIES first (so a symlink
    // still resolves to its blob for the ownership check), THEN the blobs those symlinks resolve
    // to (skipping any shared with a retained tier). `data_sizes` records the byte size of every
    // data-bearing path so reclaimed bytes reflect exactly what leaves disk.
    let mut ordered: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut data_sizes: std::collections::HashMap<PathBuf, u64> = std::collections::HashMap::new();
    for entry in &tier_entries {
        // A real file (managed mirror) IS its own data holder; a symlink's bytes live in its blob.
        if let Ok(link_meta) = tokio::fs::symlink_metadata(entry).await {
            if !link_meta.file_type().is_symlink() && link_meta.is_file() {
                data_sizes.insert(entry.clone(), link_meta.len());
            }
        }
        if seen.insert(entry.clone()) {
            ordered.push(entry.clone());
        }
    }
    for entry in &tier_entries {
        // Resolve the snapshot entry to the blob it references and remove that blob unless a retained
        // tier shares it. On macOS/Linux the entry is a SYMLINK so `canonicalize` yields the blob. On
        // Windows the HF cache uses HARDLINKS, which `canonicalize` does NOT resolve to a different
        // path (`real == entry`), so the blob's second name under blobs/ is not reclaimed here — the
        // Windows hardlink reverse-map is tracked in sc-12038. macOS/Linux (primary targets) reclaim
        // fully; the unix `variant_delete_tests` cover it.
        if let Ok(real) = tokio::fs::canonicalize(entry).await {
            if &real != entry && !retained_reals.contains(&real) {
                if let Ok(meta) = tokio::fs::metadata(&real).await {
                    data_sizes.entry(real.clone()).or_insert(meta.len());
                }
                if seen.insert(real.clone()) {
                    ordered.push(real);
                }
            }
        }
    }

    let mut removal = remove_owned_artifacts(ordered, allowed_roots, permanent).await?;
    removal.retained_paths.extend(manifest_retained);
    let reclaimed_bytes = removal
        .removed_paths
        .iter()
        .filter_map(|path| data_sizes.get(FsPath::new(path)))
        .sum();
    let tier_removal = TierRemoval {
        removed_paths: removal.removed_paths,
        retained_paths: removal.retained_paths,
        trash_failed_paths: removal.trash_failed_paths,
        reclaimed_bytes,
    };

    // Best-effort tidy once the removal itself succeeded: drop now-empty tier/snapshot dirs, and
    // the whole repo cache dir once no tier's payload remains (otherwise only the tiny refs/
    // skeleton would linger). Only ever removes EMPTY dirs.
    if tier_removal.trash_failed_paths.is_empty() {
        if let Some(repo_cache) = repo_cache.as_ref() {
            prune_empty_repo_cache(repo_cache).await;
        }
        if let Some(managed_dir) = managed_dir.as_ref() {
            remove_empty_dirs(managed_dir).await;
        }
    }

    Ok(tier_removal)
}

/// Recursively remove empty subdirectories under `dir`, then `dir` itself if it ends up empty.
/// Best-effort (ignores errors) and only ever unlinks EMPTY directories, so a sibling tier's
/// surviving files can never be removed by it.
async fn remove_empty_dirs(dir: &FsPath) {
    let dir = dir.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || remove_empty_dirs_on_disk(&dir)).await;
}

fn remove_empty_dirs_on_disk(dir: &FsPath) {
    let mut pending = vec![(dir.to_path_buf(), false)];
    while let Some((path, visited)) = pending.pop() {
        if visited {
            let _ = std::fs::remove_dir(&path);
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        let children = entries
            .map_while(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        pending.push((path, true));
        for child in children.into_iter().rev() {
            if child.is_dir() {
                pending.push((child, false));
            }
        }
    }
}

/// Prune a download-matrix repo cache after tier removal. The whole walk is
/// blocking and best effort; no filesystem work runs on the async worker.
async fn prune_empty_repo_cache(repo_cache: &FsPath) {
    let repo_cache = repo_cache.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || prune_empty_repo_cache_on_disk(&repo_cache)).await;
}

fn prune_empty_repo_cache_on_disk(repo_cache: &FsPath) {
    let Some(source_root) = repo_cache.parent() else {
        return;
    };
    let Ok(source_library) =
        sceneworks_core::model_artifacts::ArtifactSourceLibrary::new(source_root)
    else {
        return;
    };
    let Ok(snapshots_root) = source_library.snapshots_root_from_repository_root(repo_cache) else {
        return;
    };
    remove_empty_dirs_on_disk(&snapshots_root);
    remove_empty_dirs_on_disk(&repo_cache.join("blobs"));
    let has_blobs = std::fs::read_dir(repo_cache.join("blobs"))
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);
    let has_snapshot_files = huggingface_snapshot_dirs(repo_cache)
        .iter()
        .any(|snapshot| !snapshot_files(snapshot).is_empty());
    if !has_blobs && !has_snapshot_files {
        let _ = std::fs::remove_dir_all(repo_cache);
    }
}

/// Whether `model` advertises `variant` as a convert-at-install tier — i.e. it appears in the
/// catalog's `mlxTiers` (the on-disk convert-output tiers of a converted MLX model, sc-10730).
fn model_has_convert_tier(model: &Value, variant: &str) -> bool {
    model
        .get("mlxTiers")
        .and_then(Value::as_array)
        .is_some_and(|tiers| {
            tiers
                .iter()
                .filter_map(Value::as_str)
                .any(|tier| tier.eq_ignore_ascii_case(variant))
        })
}

/// Remove ONE convert-at-install tier dir (`<converted>/<tier>/`) and reclaim its disk (sc-12025).
/// Convert-at-install models (Anima) emit every tier from one convert job as a real per-tier dir
/// holding a packed DiT plus SYMLINKS to the shared dense TE/VAE (whose targets live OUTSIDE the tier
/// dir). Removing the tier dir frees only the packed DiT + the symlink entries — never the shared
/// source, which the other tiers still reference — so `reclaimed_bytes` counts only the real
/// (non-symlink) files under the tier. When this was the LAST tier with weights, the whole converted
/// dir is dropped so the model cleanly reverts to "needs conversion" rather than lingering as a bare
/// `model_index.json` marker.
async fn remove_converted_tier(
    tier_dir: PathBuf,
    allowed_roots: &[PathBuf],
    permanent: bool,
) -> Result<TierRemoval, ApiError> {
    if !tier_dir.is_dir() {
        return Ok(TierRemoval::default());
    }
    let reclaimable = converted_tier_real_bytes(&tier_dir);
    let removal = remove_owned_artifacts(vec![tier_dir.clone()], allowed_roots, permanent).await?;
    let removed = removal
        .removed_paths
        .iter()
        .any(|path| FsPath::new(path) == tier_dir);
    // If no sibling tier retains weights, drop the whole converted dir (marker included) so the model
    // reverts to a clean not-converted state instead of a bare marker. Best-effort.
    if removed && removal.trash_failed_paths.is_empty() {
        if let Some(converted) = tier_dir.parent() {
            let any_tier_left = ["bf16", "q8", "q4"]
                .iter()
                .any(|tier| tier_subdir_has_weights(&converted.join(tier)));
            if !any_tier_left {
                let _ = tokio::fs::remove_dir_all(converted).await;
            }
        }
    }
    Ok(TierRemoval {
        removed_paths: removal.removed_paths,
        retained_paths: removal.retained_paths,
        trash_failed_paths: removal.trash_failed_paths,
        reclaimed_bytes: if removed { reclaimable } else { 0 },
    })
}

/// Sum the bytes of the REAL (non-symlink) files under a converted tier dir — the packed DiT — so a
/// tier delete reports only what it actually frees. The shared TE/VAE are symlinks to a source
/// outside the tier dir; following them would over-count disk that a tier delete does not reclaim.
fn converted_tier_real_bytes(tier_dir: &FsPath) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![tier_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.file_type().is_symlink() {
                continue; // shared TE/VAE — its target lives outside the tier dir; not reclaimed
            }
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

/// Import-compatibility gate for a base checkpoint (sc-14019, epic 14015): resolve the primary
/// weight file under `source`, classify it with the base-weight detector, and refuse the import
/// unless the verdict is one a real loader accepts today
/// ([`sceneworks_core::base_weights::import_supported`]). The refusal reason is surfaced to the API
/// client as a `400` so the upload flow gets a synchronous, actionable message rather than a queued
/// job that fails later. This gate is **additive** to path confinement — `source` has already been
/// confined by `validate_import_source_path` (LAN-exposed jobs API, epic 4484); this never
/// widens or bypasses that. The worker re-runs the same predicate over the downloaded bytes so repo/
/// URL imports (whose file is not on disk at queue time) are covered there.
fn import_source_supported(source: &FsPath) -> Result<Option<String>, ApiError> {
    let weight_file = imported_model_primary_weight_file(source)
        .or_else(|| sceneworks_core::base_weights::minimax_h3_import_primary(source));
    let Some(weight_file) = weight_file else {
        return Err(ApiError::bad_request(
            "No safetensors base-weight file was found to import; single-file base-checkpoint import expects a .safetensors transformer.",
        ));
    };
    let detection =
        sceneworks_core::base_weights::detect_import_source_weights(source, &weight_file)
            .map_err(model_family_inspection_error)?;
    import_detection_supported(&detection).map_err(ApiError::bad_request)?;
    if matches!(
        &detection,
        BaseWeightDetection::Recognized(verdict)
            if verdict.family.as_deref() == Some("mage-flow")
    ) && (!source.is_dir()
        || !sceneworks_core::base_weights::is_mage_flow_transformer_dir(source))
    {
        return Err(ApiError::bad_request(
            "Model import for the 'mage-flow' family requires a complete transformer directory \
             containing config.json and diffusion_pytorch_model.safetensors; a bare weights file \
             is refused because the loader cannot derive its architecture.",
        ));
    }
    // H3 source shapes are codec-owned: dense/sharded diffusers or native single-file. The worker
    // canonicalizes native bytes through the shared derivative cache before the provider opens it.
    if matches!(
        &detection,
        BaseWeightDetection::Recognized(verdict)
            if verdict.family.as_deref() == Some("minimax-h3")
    ) && !sceneworks_core::base_weights::is_minimax_h3_import_source(source)
    {
        return Err(ApiError::bad_request(
            "MiniMax-H3 import requires a dense native/ComfyUI single-file or a complete dense/sharded \
             MiniMaxH3Transformer3DModel transformer directory; unknown quantized dialects and \
             directories with unplanned weights are refused.",
        ));
    }
    Ok(match detection {
        BaseWeightDetection::Recognized(verdict) => verdict.family,
        BaseWeightDetection::Unrecognized { .. } => None,
    })
}

#[allow(clippy::result_large_err)]
pub(crate) async fn create_model_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    model_import_request(state, request, None).await
}

#[allow(clippy::result_large_err)]
pub(crate) async fn repair_model_import_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    model_import_request(state, request, Some(model_id)).await
}

#[allow(clippy::result_large_err)]
async fn model_import_request(
    state: AppState,
    request: AxumRequest,
    repair_model_id: Option<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    let is_multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data"));
    if is_multipart {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()).into_response())?;
        let (payload, staged_path) = model_import_request_from_multipart(&state, multipart)
            .await
            .map_err(IntoResponse::into_response)?;
        let result = queue_model_import_job(state, payload, repair_model_id).await;
        if result.is_err() {
            cleanup_staged_model_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<ModelImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_model_import_job(state, payload, repair_model_id)
        .await
        .map_err(IntoResponse::into_response)
}

/// The `(rootId, relativePath)` pair a linked checkpoint import names, if this request is one.
///
/// Both or neither: a request carrying only one half names no checkpoint, and silently treating it
/// as a fetch import would queue a job with no source at all.
fn linked_checkpoint_selection(
    payload: &ModelImportRequest,
) -> Result<Option<(String, String)>, ApiError> {
    let root_id = payload
        .linked_root_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let relative_path = payload
        .linked_relative_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (root_id, relative_path) {
        (Some(root_id), Some(relative_path)) => {
            Ok(Some((root_id.to_owned(), relative_path.to_owned())))
        }
        (None, None) => Ok(None),
        _ => Err(ApiError::bad_request(
            "A linked checkpoint import requires both linkedRootId and linkedRelativePath",
        )),
    }
}

/// What a managed install records about where its bytes came from (sc-20636). Serialized onto the
/// queued job as `importProvenance` and handed to the ingest, which puts it in the persisted plan.
///
/// It carries no secret: `credential_host` names the host whose stored credential the download will
/// use, never the credential, and `url` is already stripped of userinfo and secret-bearing query
/// parameters by `checkpoint_ingest::sanitize_provenance_url`.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelImportProvenance {
    pub(crate) source: &'static str,
    pub(crate) reference: Option<String>,
    pub(crate) url: Option<String>,
    pub(crate) version_id: Option<String>,
    pub(crate) file_id: Option<String>,
    pub(crate) credential_host: Option<String>,
}

impl ModelImportProvenance {
    fn to_json(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("source".to_owned(), Value::String(self.source.to_owned()));
        for (key, value) in [
            ("reference", &self.reference),
            ("url", &self.url),
            ("versionId", &self.version_id),
            ("fileId", &self.file_id),
            ("credentialHost", &self.credential_host),
        ] {
            if let Some(value) = value {
                object.insert(key.to_owned(), Value::String(value.clone()));
            }
        }
        Value::Object(object)
    }

    /// The `source.provider` value the model manifest entry records for this import.
    ///
    /// Deliberately COARSER than [`Self::source`]: `upload` and `local-copy` both record the
    /// historical `"local"`. That is the value every model entry written before epic 20398 carries
    /// and every reader of this field was written against, so splitting it in two would be a silent
    /// manifest-vocabulary change nothing asked for. The finer distinction is not lost — it is
    /// recorded in `importProvenance.source`, a new field with no legacy readers. A REMOTE import
    /// does widen the provider (`civitai` rather than the generic `url` its transfer shares),
    /// because that names a genuinely new provider rather than re-spelling an existing one.
    fn manifest_provider(&self) -> &'static str {
        match self.source {
            "upload" | "local-copy" => "local",
            remote => remote,
        }
    }
}

/// Reconcile a discriminated source's `expectedSha256` with the flat `payload.expectedSha256`.
///
/// Both set and naming the SAME digest is honoured — a sha256 is hex, so case and surrounding
/// whitespace are not a difference. Both set and naming DIFFERENT digests is refused, because
/// resolving that by precedence would verify the transferred bytes against a digest the caller did
/// not choose and report success. A blank flat field is not a value, so it is not a conflict.
fn reconcile_expected_sha256(
    from_source: Option<String>,
    from_flat: Option<String>,
) -> Result<Option<String>, ApiError> {
    let blank_is_absent = |value: Option<String>| value.filter(|digest| !digest.trim().is_empty());
    match (blank_is_absent(from_source), blank_is_absent(from_flat)) {
        (Some(source_digest), Some(flat_digest)) => {
            let normalize = |digest: &str| digest.trim().to_ascii_lowercase();
            if normalize(&source_digest) != normalize(&flat_digest) {
                return Err(ApiError::bad_request(format!(
                    "Provide either the discriminated `source` or the legacy `expectedSha256` field, not both: they name different digests ({} and {})",
                    source_digest.trim(),
                    flat_digest.trim()
                )));
            }
            Ok(Some(source_digest))
        }
        (Some(digest), None) | (None, Some(digest)) => Ok(Some(digest)),
        (None, None) => Ok(None),
    }
}

fn normalize_import_repo_ids(payload: &mut ModelImportRequest) -> Result<(), ApiError> {
    if let Some(repo) = payload.repo.as_mut() {
        *repo = validate_huggingface_repo(repo)?.to_owned();
    }
    if let Some(ModelImportSourceV1::HuggingFace { repo, .. }) = payload.source.as_mut() {
        *repo = validate_huggingface_repo(repo)?.to_owned();
    }
    Ok(())
}

/// Collapse the discriminated `source` (when present) onto the flat `repo`/`sourceUrl`/`sourcePath`
/// fields the rest of the import path already speaks, and describe what was chosen as provenance.
///
/// Both spellings converge HERE, before any validation, so there is one import pipeline rather than
/// a new one beside the old: a legacy `{"repo": "..."}` body and a
/// `{"source": {"kind": "huggingFace", "repo": "..."}}` body produce the identical job payload apart
/// from the provenance detail the discriminated form additionally carries.
///
/// A request that sets both a discriminated source and a conflicting flat field is refused rather
/// than resolved by precedence — silently honouring one and dropping the other is how an import
/// fetches something the caller did not ask for.
pub(crate) fn normalize_model_import_source(
    payload: &mut ModelImportRequest,
) -> Result<(ModelImportProvenance, Option<String>), ApiError> {
    normalize_import_repo_ids(payload)?;
    let Some(source) = payload.source.take() else {
        // Legacy flat request. `uploadedSourcePath` is set only by the multipart handler, so it is
        // what distinguishes an upload from a copy of a file the user already had.
        let provenance = if let Some(repo) = payload.repo.as_deref() {
            ModelImportProvenance {
                source: "huggingface",
                reference: Some(repo.to_owned()),
                ..Default::default()
            }
        } else if let Some(url) = payload.source_url.as_deref() {
            ModelImportProvenance {
                source: "url",
                url: sceneworks_core::checkpoint_ingest::sanitize_provenance_url(url),
                credential_host: sceneworks_core::checkpoint_ingest::provenance_credential_host(
                    url,
                ),
                ..Default::default()
            }
        } else if payload.uploaded_source_path {
            ModelImportProvenance {
                source: "upload",
                reference: payload.files.first().cloned(),
                ..Default::default()
            }
        } else {
            ModelImportProvenance {
                source: "local-copy",
                reference: payload
                    .source_path
                    .as_deref()
                    .and_then(|path| FsPath::new(path).file_name()?.to_str())
                    .map(str::to_owned),
                ..Default::default()
            }
        };
        return Ok((provenance, None));
    };

    let conflict = |field: &str| {
        Err(ApiError::bad_request(format!(
            "Provide either the discriminated `source` or the legacy `{field}` field, not both"
        )))
    };
    if !option_str_is_empty(payload.repo.as_deref()) {
        return conflict("repo");
    }
    if !option_str_is_empty(payload.source_url.as_deref()) {
        return conflict("sourceUrl");
    }
    // An upload fills `sourcePath` server-side from the multipart `file` part, which is not a
    // caller-supplied conflict.
    if !option_str_is_empty(payload.source_path.as_deref()) && !payload.uploaded_source_path {
        return conflict("sourcePath");
    }

    let mut revision = None;
    let provenance = match source {
        ModelImportSourceV1::Upload { staged_path } => {
            let staged_path = staged_path
                .or_else(|| payload.source_path.clone())
                .ok_or_else(|| {
                    ApiError::bad_request(
                        "An upload source requires the file to be sent in the same multipart request",
                    )
                })?;
            payload.source_path = Some(staged_path);
            payload.uploaded_source_path = true;
            ModelImportProvenance {
                source: "upload",
                reference: payload.files.first().cloned(),
                ..Default::default()
            }
        }
        ModelImportSourceV1::LocalPath { path } => {
            let reference = FsPath::new(&path)
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_owned);
            payload.source_path = Some(path);
            payload.uploaded_source_path = false;
            ModelImportProvenance {
                source: "local-copy",
                reference,
                ..Default::default()
            }
        }
        ModelImportSourceV1::Url {
            url,
            expected_sha256,
        } => {
            let provenance = ModelImportProvenance {
                source: "url",
                url: sceneworks_core::checkpoint_ingest::sanitize_provenance_url(&url),
                credential_host: sceneworks_core::checkpoint_ingest::provenance_credential_host(
                    &url,
                ),
                ..Default::default()
            };
            payload.source_url = Some(url);
            payload.expected_sha256 =
                reconcile_expected_sha256(expected_sha256, payload.expected_sha256.take())?;
            provenance
        }
        ModelImportSourceV1::HuggingFace {
            repo,
            revision: requested_revision,
            files,
        } => {
            if !files.is_empty() {
                payload.files = files;
            }
            revision = requested_revision.clone();
            let reference = match requested_revision.as_deref() {
                Some(revision) => format!("{repo}@{revision}"),
                None => repo.clone(),
            };
            payload.repo = Some(repo);
            ModelImportProvenance {
                source: "huggingface",
                reference: Some(reference),
                version_id: requested_revision,
                file_id: payload.files.first().cloned(),
                ..Default::default()
            }
        }
        ModelImportSourceV1::Civitai {
            url,
            model_version_id,
            file_id,
            expected_sha256,
        } => {
            let provenance = ModelImportProvenance {
                source: "civitai",
                reference: model_version_id
                    .as_deref()
                    .map(|version| format!("modelVersion/{version}")),
                url: sceneworks_core::checkpoint_ingest::sanitize_provenance_url(&url),
                version_id: model_version_id,
                file_id,
                credential_host: sceneworks_core::checkpoint_ingest::provenance_credential_host(
                    &url,
                ),
            };
            // Civitai downloads through the ordinary source-URL path: same transfer, same
            // host-keyed credential attachment, same size limits. Only the recorded identity
            // differs.
            payload.source_url = Some(url);
            payload.expected_sha256 =
                reconcile_expected_sha256(expected_sha256, payload.expected_sha256.take())?;
            provenance
        }
    };
    Ok((provenance, revision))
}

/// Re-resolve a managed NVFP4 variant's pin SERVER-SIDE (sc-11043, epic 11037).
///
/// `GET /api/v1/models/managed-variants` advertises, for each registered variant, the exact import
/// body that installs it. That body is a convenience for the client, and a convenience is not a
/// guarantee: without this, a client could `POST` `modelId: "nvfp4-krea-2-turbo"` with any repo,
/// revision or file it liked — or omit `expectedSha256` entirely, which the worker's
/// `if let Some(..)` check treats as "nothing to verify" — and SceneWorks would record unverified
/// bytes under the curated variant's identity, its checkpoint id, and its provenance. AC3's
/// "verified file checksum" has to be enforced by the server or it is only client-cooperative.
///
/// So the registration wins. The registry is keyed on the very id the request claims, so when the
/// id names a variant, the repo, revision, file list and `expectedSha256` are taken from the
/// registration rather than from the body. A request that explicitly names a DIFFERENT pin is
/// refused rather than silently rewritten — that is a client bug or an attempt, and both deserve a
/// diagnostic. There is still exactly ONE import pipeline: this only fills in the fetch, and
/// everything after it (normalisation, licence gate, install-id validation, the ingest) is the
/// generic path every other import walks.
///
/// A managed variant is a pinned FETCH of specific upstream bytes, so the ownership modes that do
/// not fetch — a linked library reference, a local copy, an upload, a bare URL — cannot claim one.
fn resolve_managed_variant_pin(payload: &mut ModelImportRequest) -> Result<(), ApiError> {
    use sceneworks_core::managed_checkpoint_variants::{
        match_managed_nvfp4_variant_id, ManagedVariantIdMatch,
    };
    // Case-INSENSITIVELY, because a managed variant's id is its install-directory name and its
    // checkpoint identity, and `safe_download_dir` preserves whatever case the request supplied
    // while NTFS and a default APFS volume do not. An exact-match lookup therefore missed
    // `"NVFP4-Krea-2-Turbo"` — no registered repo, revision, file or checksum was applied — while
    // the distinct-id collision check saw a new id and the filesystem saw the curated variant's own
    // directory. Arbitrary bytes then landed under the curated identity with nothing verified.
    //
    // A near miss is REFUSED, never normalized to the canonical id. Rewriting it would install
    // something the request did not name, which is the same class of silent substitution this
    // function exists to refuse everywhere else; the diagnostic names the canonical id so a client
    // bug is one edit away from correct.
    let variant = match match_managed_nvfp4_variant_id(payload.model_id.as_deref().unwrap_or("")) {
        ManagedVariantIdMatch::Unregistered => return Ok(()),
        ManagedVariantIdMatch::Exact(variant) => variant,
        ManagedVariantIdMatch::NearMiss(variant) => {
            return Err(ApiError::bad_request(format!(
                "Model id '{}' differs only in case from the managed NVFP4 variant '{}', which \
                 shares its install directory and checkpoint identity on this platform. Import it \
                 under the exact id '{}', or choose a model id that is not a case variant of a \
                 managed variant.",
                payload.model_id.as_deref().unwrap_or("").trim(),
                variant.variant_id,
                variant.variant_id
            )));
        }
    };
    let claim = |detail: &str| {
        Err(ApiError::bad_request(format!(
            "Model id '{}' is a managed NVFP4 variant pinned to {}@{} {}; {detail}",
            variant.variant_id, variant.repo, variant.revision, variant.repo_file
        )))
    };

    if payload.linked_root_id.is_some() || payload.linked_relative_path.is_some() {
        return claim(
            "it is installed by fetching those pinned bytes, so it cannot be claimed by a linked \
             library reference. Import your own copy under a different model id.",
        );
    }
    if !option_str_is_empty(payload.source_url.as_deref())
        || !option_str_is_empty(payload.source_path.as_deref())
    {
        return claim(
            "it cannot be installed from a source URL or a local path. Import your own copy under \
             a different model id.",
        );
    }
    match payload.source.as_ref() {
        None | Some(ModelImportSourceV1::HuggingFace { .. }) => {}
        Some(_) => {
            return claim(
                "it is a Hugging Face repo import. Import your own copy under a different model \
                 id.",
            )
        }
    }
    if let Some(repo) = payload.repo.as_deref().map(str::trim) {
        if !repo.is_empty() && repo != variant.repo {
            return claim(&format!("this request names repo '{repo}'"));
        }
    }
    if let Some(ModelImportSourceV1::HuggingFace {
        repo,
        revision,
        files,
    }) = payload.source.as_ref()
    {
        if repo.trim() != variant.repo {
            return claim(&format!("this request names repo '{}'", repo.trim()));
        }
        if let Some(revision) = revision.as_deref().map(str::trim) {
            if !revision.is_empty() && revision != variant.revision {
                return claim(&format!("this request names revision '{revision}'"));
            }
        }
        if !files.is_empty() && files.iter().any(|file| file.trim() != variant.repo_file) {
            return claim("this request names a different file in the repo");
        }
    }
    if !payload.files.is_empty()
        && payload
            .files
            .iter()
            .any(|file| file.trim() != variant.repo_file)
    {
        return claim("this request names a different file in the repo");
    }
    if let Some(expected) = payload.expected_sha256.as_deref().map(str::trim) {
        if !expected.is_empty() && !expected.eq_ignore_ascii_case(&variant.sha256) {
            return claim(&format!(
                "this request asserts checksum '{expected}', not the pinned '{}'",
                variant.sha256
            ));
        }
    }
    if let Some(family) = payload.family.as_deref().map(str::trim) {
        if !family.is_empty()
            && sceneworks_core::lora_family::canonical_lora_family(family) != variant.family
        {
            return claim(&format!("this request names family '{family}'"));
        }
    }

    // The registration is now the source of truth for every fetch fact, including the ones the
    // request left out. `expectedSha256` in particular is filled in rather than defaulted to
    // "unverified", which is the whole point.
    payload.repo = None;
    payload.source_url = None;
    payload.source_path = None;
    payload.files = Vec::new();
    payload.source = Some(ModelImportSourceV1::HuggingFace {
        repo: variant.repo.clone(),
        revision: Some(variant.revision.clone()),
        files: vec![variant.repo_file.clone()],
    });
    payload.expected_sha256 = Some(variant.sha256.clone());
    payload.family = Some(variant.family.clone());
    if payload
        .name
        .as_deref()
        .map(str::trim)
        .unwrap_or("")
        .is_empty()
    {
        payload.name = Some(variant.display_name.clone());
    }
    Ok(())
}

pub(crate) async fn queue_model_import_job(
    state: AppState,
    mut payload: ModelImportRequest,
    repair_model_id: Option<String>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    normalize_import_repo_ids(&mut payload)?;
    let repair = if let Some(model_id) = repair_model_id {
        if payload.model_id.as_deref().is_some_and(|id| id != model_id) {
            return Err(ApiError::bad_request(
                "Repair modelId must match the selected model",
            ));
        }
        if linked_checkpoint_selection(&payload)?.is_some() {
            return Err(ApiError::bad_request(
                "Linked originals use relink/rescan, not managed repair",
            ));
        }
        let model = find_model_by_id(&state, &model_id)
            .await?
            .ok_or_else(|| model_not_found("The imported model to repair no longer exists"))?;
        let binding = sceneworks_core::model_artifacts::provenance::managed_repair_binding(
            &state.settings.data_dir,
            model
                .as_object()
                .ok_or_else(|| ApiError::bad_request("Invalid imported model"))?,
        )
        .map_err(ApiError::bad_request)?;
        payload.model_id = Some(model_id);
        Some(binding)
    } else {
        None
    };
    // A curated managed NVFP4 variant is identified by its `modelId`, and its pin is re-resolved
    // from the registry before anything else reads the body — so no client-supplied repo,
    // revision, file or (absent) checksum can record unverified bytes under that identity.
    resolve_managed_variant_pin(&mut payload)?;
    // Linked (in-place) ownership: an approved library root plus a confined relative path, and
    // never an absolute path (epic 20398, sc-20635). It is an ALTERNATIVE ownership mode rather
    // than another way to FETCH, so it is resolved before source normalisation and excludes it
    // entirely — including sc-20636's discriminated `source`, which only ever names a transfer.
    let linked = linked_checkpoint_selection(&payload)?;
    if let Some((root_id, relative_path)) = linked.clone() {
        if !option_str_is_empty(payload.repo.as_deref())
            || !option_str_is_empty(payload.source_url.as_deref())
            || !option_str_is_empty(payload.source_path.as_deref())
            || payload.source.is_some()
        {
            return Err(ApiError::bad_request(
                "A linked checkpoint import references a library in place; do not also provide a \
                 repo, source URL, source path, or a discriminated source",
            ));
        }
        // Fail fast on an unapproved or unavailable root, and on a path shape the plan store would
        // refuse anyway, so a job is never queued for a source that cannot compile. The same
        // validator `compile_linked` uses, not a second copy of its rules.
        sceneworks_core::checkpoint_plan_store::validate_linked_relative_path(&relative_path)
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
        let data_dir = state.settings.data_dir.clone();
        let probed_root_id = root_id.clone();
        tokio::task::spawn_blocking(move || {
            sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(&data_dir)
                .resolve_root(&probed_root_id)
        })
        .await
        .map_err(|error| ApiError::internal(format!("Approved root lookup failed: {error}")))?
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
        // The VALIDATED values are what the job carries. Without this the route validated the
        // trimmed pair and then serialised the raw one, so `" root-abc "` passed the API and
        // failed in the worker — and the queued `manifestEntry` (built from the trimmed pair
        // below) and the job payload disagreed about which checkpoint was being imported.
        payload.linked_root_id = Some(root_id);
        payload.linked_relative_path = Some(relative_path);
    }
    // Source normalisation collapses sc-20636's discriminated `source` onto the flat
    // repo/sourceUrl/sourcePath fields. A linked import has no transfer to normalise — it was
    // refused above if it named one — so it names its own provenance and skips the fetch checks.
    let (provenance, revision) = match &linked {
        Some((root_id, relative_path)) => (
            ModelImportProvenance {
                source: "linked-library",
                reference: Some(format!("{root_id}/{relative_path}")),
                ..Default::default()
            },
            None,
        ),
        None => normalize_model_import_source(&mut payload)?,
    };
    if linked.is_none()
        && option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, source path, or an approved library root id \
             with a relative path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    if let Some(repo) = payload.repo.as_deref() {
        payload.repo = Some(validate_huggingface_repo(repo)?.to_owned());
    }
    // Licence acknowledgment, keyed on the repo the import will FETCH (sc-17227). This route had no
    // licence logic at all and did not read
    // the catalog for the source — so `{"repo": "MiniMaxAI/MiniMax-H3"}` pulled the restricted
    // weights from upstream while `POST /api/v1/models/:id/download` was answering 403 for the same
    // bytes. The same predicate the raw jobs route uses, so there is one mechanism, not two.
    ensure_license_acknowledged_for_source(
        &state,
        &[payload.repo.as_deref()],
        payload.source_url.as_deref(),
        payload.license_acknowledged,
    )
    .await?;
    let mut model_type = match payload.model_type.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => {
            let normalized = value.to_ascii_lowercase();
            if !ALLOWED_MODEL_TYPES.contains(&normalized.as_str()) {
                return Err(ApiError::bad_request(format!(
                    "Model type must be one of {}",
                    ALLOWED_MODEL_TYPES.join(", ")
                )));
            }
            normalized
        }
        _ => "image".to_owned(),
    };
    payload.model_type = Some(model_type.clone());
    if let Some(family) = payload.family.take() {
        let models = model_catalog_snapshot(&state).await?;
        let normalized = normalize_lora_family(&family);
        payload.family = Some(
            if models.iter().any(|model| {
                model
                    .get("family")
                    .and_then(Value::as_str)
                    .is_some_and(|own_family| normalize_lora_family(own_family) == normalized)
            }) {
                normalized
            } else {
                validate_lora_family(&models, &family)?
            },
        );
    }
    // A task/base declaration constrains architecture even if the caller omitted `family`.
    // Local bytes reconcile below; deferred remote/linked inspection must carry this same
    // constraint into the worker instead of treating it as descriptive catalog metadata.
    validate_minimax_h3_import_declaration(&payload)?;
    let declaration = json!({"family":payload.family,"baseModel":payload.base_model,"minimaxH3Partition":payload.minimax_h3_partition});
    if sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(
        declaration.as_object().expect("declaration object"),
    ) {
        payload.family = Some("minimax-h3".to_owned());
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
        .or_else(|| {
            linked.as_ref().and_then(|(_, relative_path)| {
                FsPath::new(relative_path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .or_else(|| {
            payload
                .source_url
                .as_deref()
                .and_then(|value| lora_source_url_file_stem(value).ok())
        })
        .or_else(|| {
            payload.source_path.as_deref().and_then(|path| {
                FsPath::new(path)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::to_owned)
            })
        })
        .unwrap_or_else(|| "Imported Model".to_owned());
    let model_id = payload
        .model_id
        .clone()
        .unwrap_or_else(|| slugify_lora_id(&name));
    let existing_ids = model_catalog_snapshot(&state)
        .await?
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    if existing_ids.contains(&model_id) && repair.is_none() {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' already exists. Use its explicit repair operation to replace the selection."
        )));
    }
    // A repair never writes over the original installation. Publication changes one catalog
    // selection atomically; old jobs/leases still name their exact original plan and bytes.
    let target_name = if repair.is_some() {
        format!(
            "{}-repair-{}",
            safe_download_dir(&model_id),
            Uuid::new_v4().simple()
        )
    } else {
        safe_download_dir(&model_id)
    };
    // The install id the WORKER will use is the SANITIZED `target_name`, not the raw model id, and
    // it has to be a usable directory name before a job exists: the ingest resolves it through
    // `CheckpointPlanStore::install_dir`, which validates. Without this a caller-supplied `modelId`
    // that sanitizes to (say) a dot-leading name is answered 201 and then dies in the worker with
    // an `InvalidInstallId` the caller never sees. Validated at the resolved value, because that is
    // the string the worker will actually address a path with.
    if let Err(error) = sceneworks_core::checkpoint_plan_store::validate_install_id(&target_name) {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' cannot be installed: {error}"
        )));
    }
    let target_dir = state
        .settings
        .data_dir
        .join("models")
        .join("imports")
        .join(&target_name);
    let manifest_path = state
        .settings
        .config_dir
        .join("manifests")
        .join("user.models.jsonc");
    let source_path_rel = format!("models/imports/{target_name}");
    let allowed_source_roots = vec![state.settings.data_dir.join("models")];
    if let Some(source_path) = payload.source_path.clone() {
        let allowed_source_roots = if payload.uploaded_source_path {
            vec![state.settings.data_dir.join("cache").join("model-uploads")]
        } else {
            allowed_source_roots
        };
        let (source_path, detected) = tokio::task::spawn_blocking(move || {
            validate_import_source_path(&source_path, &allowed_source_roots)?;
            // Compatibility gate (sc-14019): refuse, with the detector's typed
            // reason, any local checkpoint whose shape has no loader today.
            let base_family = import_source_supported(FsPath::new(&source_path))?;
            let detected = detect_model_family(FsPath::new(&source_path))
                .map_err(model_family_inspection_error)?
                .or(base_family);
            sceneworks_core::minimax_h3_import::validate_declared_source_family(
                declaration.as_object().expect("declaration object"),
                detected.as_deref(),
            )
            .map_err(ApiError::bad_request)?;
            Ok::<_, ApiError>((source_path, detected))
        })
        .await
        .map_err(|error| {
            ApiError::internal(format!("Model import inspection task failed: {error}"))
        })??;
        payload.family = reconcile_model_family(
            payload.family.take(),
            detected,
            &format!("source_path={source_path}"),
        )?;
    }
    validate_minimax_h3_import_declaration(&payload)?;
    if payload.family.as_deref() == Some("minimax-h3") || payload.minimax_h3_partition.is_some() {
        model_type = "video".to_owned();
        payload.model_type = Some(model_type.clone());
    }
    // Reuse the selected base's existing co-requisite/download/license workflow. The imported
    // DiT is never a catalog download: only the shared closure and declared load-only companion
    // are provisioned, identically for managed transfers and approved linked sources.
    let import_components = if payload.family.as_deref() == Some("minimax-h3") {
        let base_id = payload.base_model.as_deref().expect("validated H3 base");
        let base = find_model_by_id(&state, base_id)
            .await?
            .ok_or_else(|| ApiError::bad_request("The selected compatible base is unavailable"))?;
        let default = sceneworks_core::model_artifacts::artifact_selection::model_download(&base)
            .ok_or_else(|| {
            ApiError::bad_request("The compatible base has no installable components")
        })?;
        let declaration =
            json!({"baseModel":base_id,"minimaxH3Partition":payload.minimax_h3_partition});
        let default_tier = default.get("variant").and_then(Value::as_str);
        let mut selected_tier = default_tier.unwrap_or("bf16").to_owned();
        let mut rows = sceneworks_core::minimax_h3_import::import_component_downloads(
            declaration.as_object().expect("declaration"),
            &base,
            default_tier,
            std::env::consts::OS,
        )
        .map_err(ApiError::bad_request)?;
        // Reuse a complete resident closure even when it is not the base's default tier.
        for tier in ["q4", "q8", "bf16"] {
            if let Ok(candidate) = sceneworks_core::minimax_h3_import::import_component_downloads(
                declaration.as_object().expect("declaration"),
                &base,
                Some(tier),
                std::env::consts::OS,
            ) {
                if candidate.iter().all(|row| imported_component_presence(&state.settings.data_dir, row)
                    == sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed) {
                    rows = candidate;
                    selected_tier = tier.to_owned();
                    break;
                }
            }
        }
        rows.push(
            sceneworks_core::minimax_h3_import::import_config_download(
                &base,
                &selected_tier,
                std::env::consts::OS,
            )
            .map_err(ApiError::bad_request)?,
        );
        let repos: Vec<_> = rows
            .iter()
            .map(|row| row.get("repo").and_then(Value::as_str))
            .collect();
        ensure_license_acknowledged_for_source(&state, &repos, None, payload.license_acknowledged)
            .await?;
        Some((base, rows, selected_tier))
    } else {
        None
    };
    let timestamp = now_rfc3339();
    // A linked entry names its library by root id and relative path and carries NO install path:
    // nothing is copied, and the plan-driven route reads the plan's own layers. The worker stamps
    // `importPlan.checkpointId` onto this entry once the full-content compile succeeds — a
    // candidate that never compiles never becomes selectable (E7).
    let mut manifest_entry = match &linked {
        Some((root_id, relative_path)) => json!({
            "id": model_id,
            "name": name,
            "type": model_type,
            "source": {
                "provider": provenance.manifest_provider(),
                "rootId": root_id,
                "relativePath": relative_path,
            },
            "files": payload.files.clone(),
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }),
        None => json!({
            "id": model_id,
            "name": name,
            "type": model_type,
            "source": {
                // The provider the SOURCE normalisation resolved, so a Civitai import is recorded
                // as "civitai" rather than as the generic "url" its transfer happens to share.
                // Coarser than provenance for the two non-remote shapes — see `manifest_provider`.
                "provider": provenance.manifest_provider(),
                "repo": payload.repo.clone(),
                "path": source_path_rel,
            },
            "files": payload.files.clone(),
            "paths": {
                "model": target_dir.display().to_string(),
            },
            "createdAt": timestamp,
            "updatedAt": timestamp,
        }),
    };
    if let Some(source_url) = payload.source_url.clone() {
        if let Some(source) = manifest_entry
            .get_mut("source")
            .and_then(Value::as_object_mut)
        {
            source.insert("url".to_owned(), Value::String(source_url));
        }
    }
    if let Some(family) = payload.family.clone() {
        if let Some(object) = manifest_entry.as_object_mut() {
            object.insert("family".to_owned(), Value::String(family));
        }
    }
    if let Some(object) = manifest_entry.as_object_mut() {
        if let Some(base_model) = &payload.base_model {
            object.insert("baseModel".to_owned(), Value::String(base_model.clone()));
        }
        if let Some(partition) = &payload.minimax_h3_partition {
            object.insert(
                "minimaxH3Partition".to_owned(),
                Value::String(partition.clone()),
            );
        }
        if let Some((_, rows, tier)) = &import_components {
            object.insert("downloads".to_owned(), json!(rows));
            object.insert("importComponentTier".to_owned(), json!(tier));
        }
        apply_model_manifest_defaults(object, &model_type, payload.family.as_deref());
    }
    if let Some(entry) = manifest_entry.as_object() {
        ensure_inherited_component_licenses_acknowledged(
            &state,
            entry,
            payload.license_acknowledged,
        )
        .await?;
    }
    let ownership_mode = payload.ownership_mode;
    let mut payload = to_json_object(&payload)?;
    // The ingest contract the worker executes: managed ownership, plus the provenance the install's
    // persisted plan will carry. Written unconditionally so a legacy flat request is a managed
    // import with recorded provenance too — not a second, unprovenanced code path (AC3).
    payload.insert(
        "ownershipMode".to_owned(),
        serde_json::to_value(ownership_mode)
            .map_err(|error| ApiError::internal(error.to_string()))?,
    );
    payload.insert("importProvenance".to_owned(), provenance.to_json());
    if let Some(revision) = revision {
        payload.insert("revision".to_owned(), Value::String(revision));
    }
    payload.insert("modelId".to_owned(), manifest_entry["id"].clone());
    payload.insert("modelName".to_owned(), manifest_entry["name"].clone());
    // A linked import installs nothing, so it gets no install directory: handing one to the worker
    // would name a path that must never be created for this ownership mode.
    if linked.is_none() {
        payload.insert(
            "targetDir".to_owned(),
            Value::String(target_dir.display().to_string()),
        );
    }
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    if let Some(binding) = repair {
        payload.insert("repairExpectedBinding".into(), binding);
    }
    if let Some((base, rows, _)) = import_components {
        // sc-26687 / sc-26690: the SAME inherited-component gate at the DOWNLOAD boundary. Closing
        // only the generation door would still let an unacknowledged component's bytes land, so a
        // resident component is fetched only once its own terms are acknowledged. Idempotent with
        // the generation-admission gate in `create_generation_job_with_status`.
        if let Some(entry) = payload.get("manifestEntry").and_then(Value::as_object) {
            ensure_inherited_component_licenses_acknowledged(
                &state,
                entry,
                payload
                    .get(LICENSE_ACKNOWLEDGED_PAYLOAD_KEY)
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            )
            .await?;
        }
        for row in rows {
            // The import worker fetches this literal config through the shared installer only if
            // inspection finds no source-owned claim. It writes no primary download receipt.
            if row.get("componentId").and_then(Value::as_str) == Some("import_transformer_config") {
                continue;
            }
            if imported_component_presence(&state.settings.data_dir, &row)
                == sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed {
                continue;
            }
            let component = build_model_download_job_payload(
                &base,
                &model_id,
                &row,
                None,
                false,
                payload
                    .get("licenseAcknowledged")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                &state.settings.data_dir,
            )?;
            create_generation_job(
                state.clone(),
                JobType::ModelDownload,
                None,
                None,
                component,
                "auto".to_owned(),
            )
            .await?;
        }
    }
    let job = create_generation_job(
        state,
        JobType::ModelImport,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(public_job_snapshot(job))))
}

// The detector establishes architecture, never whether a fine-tune was trained for first/last
// frames, references, or both. A caller must declare that compatibility before an H3 job is queued;
// remote bytes are checked again by the shared worker importer after detection.
fn validate_minimax_h3_import_declaration(payload: &ModelImportRequest) -> Result<(), ApiError> {
    let family = payload.family.as_deref();
    let base = payload.base_model.as_deref();
    let partition = payload.minimax_h3_partition.as_deref();
    let declares_h3 = partition.is_some() || matches!(base, Some("minimax_h3" | "minimax_h3_ref"));
    if family != Some("minimax-h3") && !declares_h3 {
        return Ok(());
    }
    if family.is_some_and(|family| family != "minimax-h3") {
        return Err(ApiError::bad_request(
            "baseModel and minimaxH3Partition declare MiniMax-H3 compatibility, but family is not minimax-h3",
        ));
    }
    let declaration = json!({"baseModel":base,"minimaxH3Partition":partition});
    sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(
        declaration.as_object().expect("declaration object"),
    )
    .map(|_| ())
    .map_err(ApiError::bad_request)
}

pub(crate) async fn model_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(ModelImportRequest, PathBuf), ApiError> {
    let mut payload = ModelImportRequest {
        ownership_mode: OwnershipModeV1::default(),
        source: None,
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        linked_root_id: None,
        linked_relative_path: None,
        files: Vec::new(),
        family: None,
        base_model: None,
        minimax_h3_partition: None,
        expected_sha256: None,
        license_acknowledged: false,
        uploaded_source_path: false,
    };
    let mut staged_path = None;

    let parse_result = async {
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| ApiError::bad_request(error.to_string()))?
        {
            let field_name = field.name().unwrap_or("").to_owned();
            if field_name == "file" {
                if staged_path.is_some() {
                    return Err(ApiError::bad_request("Only one model file can be uploaded"));
                }
                let upload_name =
                    sanitized_upload_filename(field.file_name().unwrap_or("model.safetensors"));
                let path =
                    write_model_upload_field_to_staged_file(state, field, &upload_name).await?;
                payload.source_path = Some(path.display().to_string());
                payload.files = vec![upload_name];
                payload.uploaded_source_path = true;
                staged_path = Some(path);
                continue;
            }

            let value = field
                .text()
                .await
                .map_err(|error| ApiError::bad_request(error.to_string()))?;
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match field_name.as_str() {
                "modelId" => payload.model_id = Some(value.to_owned()),
                "name" => payload.name = Some(value.to_owned()),
                "type" => payload.model_type = Some(value.to_owned()),
                "family" => payload.family = Some(value.to_owned()),
                "baseModel" => payload.base_model = Some(value.to_owned()),
                "minimaxH3Partition" => payload.minimax_h3_partition = Some(value.to_owned()),
                "repo" => payload.repo = Some(value.to_owned()),
                "sourceUrl" => payload.source_url = Some(value.to_owned()),
                // The multipart form accepts `repo`/`sourceUrl` too, so it can reach a
                // licence-restricted repo exactly like the JSON body and needs the same way to
                // assert the acknowledgment (sc-17227). Anything other than "true" leaves it false
                // — the assertion is affirmative or it is not made.
                "licenseAcknowledged" => {
                    payload.license_acknowledged = value.eq_ignore_ascii_case("true")
                }
                // The multipart form takes the same two epic-20398 fields as the JSON body, decoded
                // by the same serde impls, so an unknown ownership mode or malformed discriminated
                // source is refused here exactly as it is there rather than being ignored (E7).
                "ownershipMode" => {
                    payload.ownership_mode = serde_json::from_value(Value::String(value.to_owned()))
                        .map_err(|error| {
                            ApiError::bad_request(format!("Invalid ownershipMode: {error}"))
                        })?
                }
                "source" => {
                    payload.source = Some(serde_json::from_str(value).map_err(|error| {
                        ApiError::bad_request(format!("Invalid import source: {error}"))
                    })?)
                }
                _ => {}
            }
        }
        Ok(())
    }
    .await;
    if let Err(error) = parse_result {
        if let Some(path) = staged_path.as_deref() {
            cleanup_staged_model_upload(path).await;
        }
        return Err(error);
    }

    let Some(staged_path) = staged_path else {
        return Err(ApiError::bad_request("Upload file field is required"));
    };
    Ok((payload, staged_path))
}

pub(crate) async fn write_model_upload_field_to_staged_file(
    state: &AppState,
    field: axum::extract::multipart::Field<'_>,
    filename: &str,
) -> Result<PathBuf, ApiError> {
    crate::uploads::stage_upload(
        state,
        field,
        "model-uploads",
        Some(filename),
        max_model_upload_bytes(),
        || {
            format!(
                "Uploaded model file exceeds the {} limit",
                format_bytes(max_model_upload_bytes() as u64)
            )
        },
    )
    .await
}

pub(crate) use crate::uploads::cleanup_staged_upload as cleanup_staged_model_upload;

pub(crate) fn model_family_inspection_error(error: SafetensorsHeaderError) -> ApiError {
    match error {
        SafetensorsHeaderError::Io(io_error) => {
            ApiError::bad_request(format!("Unable to inspect model file: {io_error}"))
        }
        SafetensorsHeaderError::InvalidHeader => {
            ApiError::bad_request("Model file has an invalid safetensors header".to_owned())
        }
        SafetensorsHeaderError::IncompleteData { declared, actual } => {
            ApiError::bad_request(format!(
            "Model file is incomplete or corrupt ({actual} bytes on disk, but its header declares \
             at least {declared} bytes of tensor data); the file was likely truncated during \
             download. Re-download the complete file."
        ))
        }
    }
}

/// Applies the import-time policy for base models: confident detection rejects
/// a mismatched user-supplied family; an unsupplied family is filled in from
/// the detection; an inconclusive detection accepts the supplied family
/// unchanged (and leaves things unset if none was supplied).
pub(crate) fn reconcile_model_family(
    supplied: Option<String>,
    detected: Option<String>,
    _context: &str,
) -> Result<Option<String>, ApiError> {
    reconcile_detected_family(supplied, detected).map_err(|mismatch| {
        ApiError::bad_request(format!(
            "Model files appear to be {}, but family was declared as {}. Re-import with family {} or pick different files.",
            mismatch.detected, mismatch.supplied, mismatch.detected
        ))
    })
}

pub(crate) fn max_model_upload_bytes() -> usize {
    #[cfg(test)]
    {
        let limit = TEST_MAX_MODEL_UPLOAD_BYTES.load(std::sync::atomic::Ordering::SeqCst);
        if limit > 0 {
            return limit;
        }
    }
    MAX_MODEL_UPLOAD_BYTES
}

/// Select from the current shared catalog before cloning. Linked-source availability is probed
/// for the selected row on every call, just as it is for the full catalog response.
pub(crate) async fn find_model_catalog_entry(
    state: &AppState,
    predicate: impl Fn(&Value) -> bool,
) -> Result<Option<Value>, ApiError> {
    let snapshot = model_catalog_snapshot(state).await?;
    let Some(model) = snapshot.iter().find(|entry| predicate(entry)) else {
        return Ok(None);
    };
    let mut selected = vec![model.clone()];
    crate::model_sources::refresh_live_external_availability(
        &state.settings.data_dir,
        &mut selected,
    )
    .await?;
    Ok(selected.pop())
}

pub(crate) async fn find_model_by_id(
    state: &AppState,
    id: &str,
) -> Result<Option<Value>, ApiError> {
    find_model_catalog_entry(state, |entry| {
        entry.get("id").and_then(Value::as_str) == Some(id)
    })
    .await
}

pub(crate) fn model_not_found(detail: impl Into<String>) -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        detail: detail.into(),
        context: None,
        code: None,
    }
}

pub(crate) async fn require_model_by_id(state: &AppState, id: &str) -> Result<Value, ApiError> {
    find_model_by_id(state, id)
        .await?
        .ok_or_else(|| model_not_found("Model not found"))
}

/// Catalog without live Hugging Face size estimation: download sizes fall back to
/// manifest metadata only. This is the right call for job validation, LoRA/preset
/// CRUD, download/convert job creation, and delete — none of which read the
/// byte-accurate download size — so an unreachable huggingface.co can't stall
/// those paths (sc-4169).
pub(crate) async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    let mut models = model_catalog_snapshot(state).await?.as_ref().clone();
    crate::model_sources::refresh_live_external_availability(&state.settings.data_dir, &mut models)
        .await?;
    Ok(models)
}

/// Catalog with live Hugging Face download-size estimates (negative-cached on
/// failure). Reserved for `GET /models`, the one surface that displays
/// download sizes.
pub(crate) async fn model_catalog_sized(state: &AppState) -> Result<Vec<Value>, ApiError> {
    // Preserve SC-14800's cold-path overlap: Hugging Face metadata estimation
    // begins from the cheap manifest inputs while the shared install-state
    // snapshot is being built (or joined) independently.
    let size_estimates = estimate_current_model_catalog_sizes(state);
    let snapshot = model_catalog_snapshot(state);
    let (size_estimates, snapshot) = tokio::join!(size_estimates, snapshot);
    let size_estimates = size_estimates?;
    let mut models = snapshot?.as_ref().clone();
    crate::model_sources::refresh_live_external_availability(&state.settings.data_dir, &mut models)
        .await?;
    let selection_catalog = models.clone();
    for model in &mut models {
        let context = model_download_context(model)?;
        let live_estimate = context.as_ref().and_then(|context| {
            size_estimates
                .get(&(context.repo.clone(), context.files.clone()))
                .copied()
                .flatten()
        });
        apply_model_catalog_size_fields(model, context.as_ref(), live_estimate)?;
        apply_runtime_text_encoder_options(
            model,
            &selection_catalog,
            &state.settings.data_dir,
            &state.settings.external_model_roots,
        )?;
        annotate_commercial_use_alternatives(model, &selection_catalog);
    }
    Ok(models)
}

/// Resolve a declared `commercialUse` pointer against this catalog (sc-22998): a model whose
/// weights are not commercially eligible gets `commercialUse.alternatives`, the ids of its
/// `alternativeFamily` that are themselves eligible here — YuE2 lists the YuE1 entries. The declared
/// block is otherwise passed through untouched, and nothing here reroutes anything.
///
/// This EXPOSES the verdict; it enforces nothing. The routes that enforce it (sc-22999) are the
/// YuE2 job route and its execution-time eligibility check (`yue2_jobs`) and timeline exports,
/// which also persist the licence acknowledgment and noncommercial policy into provenance.
fn annotate_commercial_use_alternatives(model: &mut Value, catalog: &[Value]) {
    if model.get("commercialUse").is_none() {
        return;
    }
    let Some(id) = model.get("id").and_then(Value::as_str).map(str::to_owned) else {
        return;
    };
    if let Ok(sceneworks_core::model_usage_policy::CommercialUseVerdict::Refused {
        alternatives,
        ..
    }) = sceneworks_core::model_usage_policy::commercial_use_verdict(catalog, &id)
    {
        if let Some(block) = model
            .get_mut("commercialUse")
            .and_then(Value::as_object_mut)
        {
            block.insert("alternatives".to_owned(), json!(alternatives));
        }
    }
}

/// Add runtime-only text-encoder choices to the public model catalog. The worker owns enumeration so
/// the API and generation resolver share one completeness predicate. This runs on the response clone,
/// outside the cached catalog snapshot: an operator who stages a complete alternate and refreshes
/// Models sees it without a server restart, while no repo/revision/download metadata is persisted.
fn apply_runtime_text_encoder_options(
    model: &mut Value,
    catalog: &[Value],
    data_dir: &FsPath,
    external_roots: &[PathBuf],
) -> Result<(), ApiError> {
    let mut options =
        sceneworks_worker::image_text_encoder_options(catalog, model, data_dir, external_roots)
            .into_iter()
            .map(|option| serde_json::to_value(option).expect("text encoder option serializes"))
            .collect::<Vec<_>>();
    if options.is_empty() {
        let adapter = model
            .get("adapter")
            .and_then(Value::as_str)
            .unwrap_or_default();
        options.extend(
            sceneworks_worker::text_encoder_options_for_adapter(adapter, data_dir)
                .into_iter()
                .map(|option| {
                    serde_json::to_value(option).expect("video text encoder option serializes")
                }),
        );
    }
    set_runtime_text_encoder_options(model, options)
}

/// Resolve an authored Image Studio encoder selection through the worker-owned descriptor contract.
/// Only the opaque id comes from the client. The path-bearing resolution is rebuilt from a fresh
/// server catalog at every create/retry/duplicate boundary and is never trusted as request input.
pub(crate) async fn resolve_selected_image_text_encoder(
    state: &AppState,
    job_payload: &JsonObject,
    model_id: &str,
    manifest_entry: &mut Value,
) -> Result<(), ApiError> {
    let selected = match job_payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get("textEncoderModel"))
    {
        None => {
            if let Some(object) = manifest_entry.as_object_mut() {
                object.remove("resolvedTextEncoder");
            }
            return Ok(());
        }
        Some(Value::String(id)) if id == "default" => {
            if let Some(object) = manifest_entry.as_object_mut() {
                object.remove("resolvedTextEncoder");
            }
            return Ok(());
        }
        Some(Value::String(id)) => id.as_str(),
        Some(_) => return Err(ApiError::bad_request(
            "advanced.textEncoderModel must be a string option id returned by GET /api/v1/models",
        )),
    };

    let catalog = model_catalog_snapshot(state).await?;
    let catalog_model = catalog
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .unwrap_or(manifest_entry);
    let resolution = sceneworks_worker::resolve_image_text_encoder_selection(
        catalog.as_ref(),
        catalog_model,
        selected,
        &state.settings.data_dir,
        &state.settings.external_model_roots,
    )
    .map_err(ApiError::bad_request)?
    .ok_or_else(|| ApiError::internal("non-default text encoder resolved as default"))?;
    let object = manifest_entry
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    object.insert("resolvedTextEncoder".to_owned(), resolution);
    Ok(())
}

fn set_runtime_text_encoder_options(
    model: &mut Value,
    options: Vec<Value>,
) -> Result<(), ApiError> {
    let object = model
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    if options.is_empty() {
        object.remove("textEncoderOptions");
    } else {
        object.insert("textEncoderOptions".to_owned(), Value::Array(options));
    }
    Ok(())
}

#[cfg(test)]
#[path = "models/tests/runtime_text_encoder_option_tests.rs"]
mod runtime_text_encoder_option_tests;

pub(crate) async fn model_catalog_snapshot(state: &AppState) -> Result<Arc<Vec<Value>>, ApiError> {
    let backend = crate::generation::enqueue_backend(state);
    // Before trusting the cached snapshot, check whether the resolved cache's published set moved
    // under us — the worker publishes promotions in another process, so this is the only signal
    // the API gets (sc-19712 F-4). It is a handful of stats, cheap enough to pay per read.
    crate::model_sources::note_local_tier_freshness(state);
    {
        let cache_state = state.model_catalog_cache.state.lock();
        if let Some((snapshot_generation, snapshot_backend, snapshot)) =
            cache_state.snapshot.as_ref()
        {
            if *snapshot_generation == cache_state.generation && *snapshot_backend == backend {
                return Ok(snapshot.clone());
            }
        }
    }

    // sc-28037 (GitHub #3235): the build runs as its own task, not inside this
    // request's future. Its filesystem sweep is `spawn_blocking` work that a
    // dropped future cannot stop, so a client that disconnected mid-build used
    // to release the serializer while that sweep kept running, and the next
    // caller started a second sweep on top of it. Detached, an abandoned build
    // still finishes and publishes, and every later caller joins its result.
    let builder_state = state.clone();
    tokio::spawn(
        async move { build_and_publish_model_catalog_snapshot(&builder_state, backend).await },
    )
    .await
    .map_err(|err| ApiError::internal(format!("model catalog build task failed: {err}")))?
}

async fn build_and_publish_model_catalog_snapshot(
    state: &AppState,
    backend: &'static str,
) -> Result<Arc<Vec<Value>>, ApiError> {
    // Serialize only builders. The synchronous generation/snapshot state lock
    // is never held across the expensive filesystem sweep.
    let _build_guard = state.model_catalog_cache.build_serializer.lock().await;
    loop {
        let generation = {
            // Another builder may have populated the cache while this caller
            // waited for the async serializer.
            let cache_state = state.model_catalog_cache.state.lock();
            if let Some((snapshot_generation, snapshot_backend, snapshot)) =
                cache_state.snapshot.as_ref()
            {
                if *snapshot_generation == cache_state.generation && *snapshot_backend == backend {
                    return Ok(snapshot.clone());
                }
            }
            cache_state.generation
        };

        let mut models = build_model_catalog_snapshot(state).await?;
        for model in &mut models {
            project_moss_edit_surface(model, backend);
        }
        let snapshot = Arc::new(models);

        #[cfg(test)]
        state.test_hooks.pause_model_catalog_before_publish().await;

        let mut cache_state = state.model_catalog_cache.state.lock();
        if cache_state.generation == generation {
            cache_state.snapshot = Some((generation, backend, snapshot.clone()));
            return Ok(snapshot);
        }
        drop(cache_state);
        // A model writer completed while the filesystem sweep was running.
        // Keep the async serializer and rebuild at the new generation. The
        // stale result was never published or returned.
    }
}

fn project_moss_edit_surface(model: &mut Value, backend: &str) {
    let Some(id) = model.get("id").and_then(Value::as_str) else {
        return;
    };
    if !sceneworks_core::moss_image::is_moss(id) {
        return;
    }
    let available = sceneworks_core::moss_image::edit_backend_eligible(id, backend);
    let Some(object) = model.as_object_mut() else {
        return;
    };
    let capabilities = object.entry("capabilities").or_insert_with(|| json!([]));
    let Some(capabilities) = capabilities.as_array_mut() else {
        return;
    };
    capabilities.retain(|mode| mode != "edit_image" && mode != "image_edit");
    if available {
        capabilities.push(json!("edit_image"));
    }
}

// sc-4205 (F-API-12): the per-model install/cache state, formerly threaded through a
// 5-tuple that was easy to mis-order. Named fields make the catalog loop legible.
struct ModelCatalogEntryState {
    downloadable: bool,
    installed_path: Option<String>,
    installed: bool,
    cache_incomplete: bool,
    missing_required_files: Vec<String>,
    update_available: bool,
    // Reuse the existing plan/source probe for the product projection, never resolve it twice.
    imported_source_present: Option<bool>,
}

#[derive(Debug, PartialEq)]
struct ReceiptFileSet {
    files: Vec<String>,
    revision: Option<String>,
}

fn receipt_entries(managed_path: &FsPath) -> Vec<Value> {
    let Ok(bytes) = std::fs::read(managed_path.join(".sceneworks-download-complete.json")) else {
        return Vec::new();
    };
    let Ok(receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return Vec::new();
    };
    receipt
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![receipt])
}

fn receipt_file_sets(
    managed_path: &FsPath,
    repo: &str,
    model_id: Option<&str>,
) -> Vec<ReceiptFileSet> {
    receipt_entries(managed_path)
        .into_iter()
        .filter_map(|entry| {
            if entry.get("repo").and_then(Value::as_str) != Some(repo) {
                return None;
            }
            // Shared repos back multiple catalog cards. A model-specific receipt must protect only
            // the card that produced it; receipts predating modelId remain generic for compatibility.
            if let (Some(expected), Some(actual)) =
                (model_id, entry.get("modelId").and_then(Value::as_str))
            {
                if actual != expected {
                    return None;
                }
            }
            let files = entry
                .get("resolvedFiles")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let revision = entry
                .get("snapshotRevision")
                .and_then(Value::as_str)
                .map(str::to_owned);
            (!files.is_empty()).then_some(ReceiptFileSet { files, revision })
        })
        .collect()
}

/// Whether a snapshot the receipt's files resolve into is an actually LOADABLE install (not merely a
/// set of files that exist). A backfill (sc-13076) records whatever was on disk, so an interrupted
/// download left a torn tier — its `model_index.json` + a stray config present, but the
/// transformer/vae weights missing — whose recorded files all exist yet cannot load. When the
/// receipt/tier files form a diffusers tier subdir (`["<tier>/*"]`, or a receipt whose files share one
/// leading dir), require that subdir to pass the same per-component weight check the cache-health path
/// uses. A non-diffusers set (no `<tier>/model_index.json`, or a flat single-variant filter) keeps the
/// prior file-existence contract.
///
/// `family_complete` is the model's shared per-tier predicate ([`no_model_index_family_predicate`]) for
/// a no-`model_index` MLX turnkey, or `None`. Without it this check was diffusers-only, so a FLAT tier
/// (SenseNova/SANA/Boogu/Anima) passed unconditionally: a backfill could mint a "complete" receipt for a
/// tier whose tokenizer never landed, and that receipt then kept the model reading installed through the
/// usable-stale path even once the cache-health lane had correctly demoted it (sc-14432).
fn snapshot_tier_is_loadable(
    snapshot: &FsPath,
    files: &[String],
    family_complete: Option<fn(&FsPath) -> bool>,
) -> bool {
    match tier_subdir_name(files) {
        Some(tier) => {
            let tier_dir = snapshot.join(&tier);
            let diffusers_ok = !path_is_readable_file(&tier_dir.join("model_index.json"))
                || diffusers_snapshot_health(&tier_dir).installed;
            match family_complete {
                Some(complete) => diffusers_ok && complete(&tier_dir),
                None => diffusers_ok,
            }
        }
        None => true,
    }
}

fn receipt_files_present(
    data_dir: &FsPath,
    repo: &str,
    receipt: &ReceiptFileSet,
    family_complete: Option<fn(&FsPath) -> bool>,
) -> bool {
    !receipt.files.is_empty()
        && huggingface_repo_cache_path(data_dir, repo)
            .map(|root| {
                let matches = crate::huggingface_snapshot_dirs(&root)
                    .into_iter()
                    .filter(|snapshot| {
                        receipt
                            .files
                            .iter()
                            .all(|file| snapshot.join(file).is_file())
                    })
                    // A torn tier's recorded files all exist but the install can't load — it must not
                    // count as a "usable stale" install that keeps the model falsely installed.
                    .filter(|snapshot| {
                        snapshot_tier_is_loadable(snapshot, &receipt.files, family_complete)
                    })
                    // A backfill records whatever it FOUND, so a receipt can name a shard index plus
                    // only the shards that happened to land. Every recorded index must still resolve
                    // to a complete shard set on disk, or this receipt is laundering an interrupted
                    // download into "installed" (sc-20526). Stat-only: the index paths are already in
                    // the receipt, so this costs no directory walk.
                    .filter(|snapshot| listed_shard_indexes_are_complete(snapshot, &receipt.files))
                    .collect::<Vec<_>>();
                receipt
                    .revision
                    .as_deref()
                    .map_or(matches.len() == 1, |revision| {
                        matches.iter().any(|snapshot| {
                            snapshot.file_name().and_then(|v| v.to_str()) == Some(revision)
                        })
                    })
            })
            .unwrap_or(false)
}

// Catalog entries can share a Hugging Face repo. The catalog sweep probes entries concurrently,
// so serialize only the final receipt recheck/write (not the expensive snapshot walk) to prevent
// two first-seen entries from truncating the same backfill file at once.
static RECEIPT_BACKFILL_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// [`no_model_index_family_predicate`] for a whole manifest `model` entry — the receipt lanes carry the
/// model, not a pre-split family/id pair.
fn model_family_tier_predicate(model: &Value) -> Option<fn(&FsPath) -> bool> {
    no_model_index_family_predicate(
        model
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        model.get("id").and_then(Value::as_str).unwrap_or_default(),
    )
}

/// Whether a receipt entry is PROVABLY torn: every cached snapshot that holds one of the shard
/// indexes the entry recorded is missing at least one shard that index names, and no snapshot holds
/// a complete set (sc-20526).
///
/// Deliberately positive-evidence only. A receipt whose repo cache is gone, whose files simply are
/// not where this box expects them, or that records no shard index at all is left ALONE — read-side
/// validation already refuses to count such a receipt as installed, and silently deleting a receipt
/// on ambiguous evidence would destroy the relocation proof an external-library install depends on.
fn receipt_entry_is_provably_torn(entry: &Value, data_dir: &FsPath) -> bool {
    if entry.get("backfilled").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    let Some(repo) = entry.get("repo").and_then(Value::as_str) else {
        return false;
    };
    let indexes = entry
        .get("resolvedFiles")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|file| file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX))
        .collect::<Vec<_>>();
    if indexes.is_empty() {
        return false;
    }
    let Some(root) = huggingface_repo_cache_path(data_dir, repo) else {
        return false;
    };
    let mut saw_torn = false;
    for snapshot in crate::huggingface_snapshot_dirs(&root) {
        let present = indexes
            .iter()
            .filter(|file| path_is_readable_file(&snapshot.join(file)))
            .collect::<Vec<_>>();
        if present.is_empty() {
            continue;
        }
        // One complete snapshot is enough: cached revisions are alternatives, and a re-download
        // lands beside the torn revision it replaces rather than deleting it.
        if present.len() == indexes.len()
            && listed_shard_indexes_are_complete(
                &snapshot,
                &indexes
                    .iter()
                    .map(|file| (*file).to_owned())
                    .collect::<Vec<_>>(),
            )
        {
            return false;
        }
        saw_torn = true;
    }
    saw_torn
}

/// Repair receipts that a pre-sc-20526 backfill minted over a partially downloaded tier.
///
/// The offending shape is `backfilled: true` with a shard index whose `weight_map` names files that
/// were never downloaded — the receipt recorded the partial on-disk set as if it were complete, and
/// nothing invalidated it, so affected users needed a manual delete. Dropping the entry here lets the
/// next scan re-backfill honestly (the backfill writer early-outs while ANY receipt for the model
/// survives) and heals every other receipt consumer, since they all read this one file.
///
/// Cost: one small JSON read per managed model directory plus a stat per recorded index and shard.
/// It runs on the catalog sweep, never on the per-job hot path.
fn repair_torn_backfilled_receipts(managed_path: &FsPath, data_dir: &FsPath) {
    let entries = receipt_entries(managed_path);
    if entries.is_empty()
        || !entries
            .iter()
            .any(|entry| receipt_entry_is_provably_torn(entry, data_dir))
    {
        return;
    }
    let _write_guard = RECEIPT_BACKFILL_WRITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Ok(_receipt_lock) = sceneworks_core::download_receipt::lock(managed_path) else {
        return;
    };
    // Re-read under the lock: a concurrent backfill may have rewritten the file.
    let kept = receipt_entries(managed_path)
        .into_iter()
        .filter(|entry| !receipt_entry_is_provably_torn(entry, data_dir))
        .collect::<Vec<_>>();
    let receipt_path = managed_path.join(".sceneworks-download-complete.json");
    let Some(first) = kept.first().cloned() else {
        // Every entry was a false-complete: the model is not installed, so the receipt must go
        // rather than linger as a bare marker `model_is_installed` would still honour.
        let _ = std::fs::remove_file(&receipt_path);
        return;
    };
    let mut receipt = first;
    if let Some(object) = receipt.as_object_mut() {
        object.insert("receipts".to_owned(), Value::Array(kept));
    }
    let _ = sceneworks_core::download_receipt::write(&receipt_path, &receipt);
}

/// Heal legacy install identity before the catalog and local-cache eligibility read it.
fn repair_missing_snapshot_revisions(managed: &FsPath, model: &Value, data_dir: &FsPath) {
    use sceneworks_core::download_receipt;
    let marker = managed.join(".sceneworks-download-complete.json");
    if !marker.is_file() {
        return;
    }
    let Ok(_lock) = download_receipt::lock(managed) else {
        return;
    };
    let Ok(bytes) = std::fs::read(&marker) else {
        return;
    };
    let Ok(mut receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let family_complete = model_family_tier_predicate(model);
    let repair = |entry: &mut Value| {
        if entry
            .get("modelId")
            .and_then(Value::as_str)
            .is_some_and(|id| Some(id) != model.get("id").and_then(Value::as_str))
        {
            return false;
        }
        let Some(repo) = entry.get("repo").and_then(Value::as_str) else {
            return false;
        };
        let Some(root) = huggingface_repo_cache_path(data_dir, repo) else {
            return false;
        };
        let Some(revision) = download_receipt::recover_revision(entry, &root) else {
            return false;
        };
        let Ok((_, snapshot)) = sceneworks_core::hf_home::model_source_library(data_dir)
            .discover_snapshot(repo, Some(&revision))
        else {
            return false;
        };
        let files = string_array_field(entry, "resolvedFiles");
        if !snapshot_tier_is_loadable(&snapshot, &files, family_complete)
            || !listed_shard_indexes_are_complete(&snapshot, &files)
        {
            return false;
        }
        entry["snapshotRevision"] = Value::String(revision);
        true
    };
    // The newest receipt is mirrored at the top level. Repair both representations without
    // rebuilding the envelope or dropping unrelated variants and fields.
    let mut changed = repair(&mut receipt);
    if let Some(entries) = receipt.get_mut("receipts").and_then(Value::as_array_mut) {
        for entry in entries {
            changed |= repair(entry);
        }
    }
    if changed {
        if let Err(error) = download_receipt::write(&marker, &receipt) {
            tracing::warn!(%error, path = %marker.display(), "could not repair missing snapshot revisions");
        }
    }
}

fn backfill_current_receipt(
    managed_path: &FsPath,
    model: &Value,
    context: &DownloadContext,
    data_dir: &FsPath,
) {
    let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
    if !receipt_file_sets(managed_path, &context.repo, Some(model_id)).is_empty() {
        return;
    }
    let family_complete = model_family_tier_predicate(model);
    let receipts = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .filter_map(|entry| {
            let repo = entry.get("repo")?.as_str()?;
            let files = string_array_field(entry, "files");
            let root = huggingface_repo_cache_path(data_dir, repo)?;
            let snapshot = crate::huggingface_snapshot_dirs(&root).into_iter().find(|snapshot| {
                files.iter().all(|pattern| snapshot_contains_pattern(snapshot, pattern))
            })?;
            // Never manufacture a receipt for a torn tier: a `<tier>/*` glob matches as soon as one
            // metadata file exists, so backfilling it would record a "complete" install that cannot
            // load. Require the tier to actually hold its weights before preserving it (sc-13076), and
            // — for a no-`model_index` turnkey, where "holds its weights" is family-specific — to pass
            // the shared per-tier predicate too (sc-14432).
            if !snapshot_tier_is_loadable(&snapshot, &files, family_complete) {
                return None;
            }
            let resolved = snapshot_files(&snapshot).into_iter()
                .filter(|file| allow_pattern_matches(file, &files)).collect::<Vec<_>>();
            // Record only a file set that SATISFIES what the repo declares, never merely what was
            // found (sc-20526). A `<tier>/*` glob and a whole-repo fetch declare no per-file claim,
            // so the sharded-safetensors indexes IN the resolved set are the workable ground truth:
            // every shard their `weight_map` names must be on disk. Otherwise this would mint a
            // "complete" receipt for a tier missing ~4 GB of its text encoder (lens_turbo bf16) and
            // that receipt would then keep the model reading installed through the usable-stale path.
            if !listed_shard_indexes_are_complete(&snapshot, &resolved) {
                return None;
            }
            // Record WHICH snapshot these files were read from (sc-19712 F-5). A revision-less
            // receipt makes its whole repository unserveable from the resolved cache — there is no
            // pair to compare coverage against — so a backfilled install silently dropped out of
            // the local tier while still being promotable, occupying cache bytes it could never be
            // served from. The revision is not unknowable here: the snapshot directory selected
            // just above IS `.../snapshots/<revision>`, and `resolvedFiles` were read from it. Only
            // a genuinely immutable 40-hex name is recorded; anything else is omitted rather than
            // written as a revision the source tier could not re-resolve. The worker's
            // `establish_receipt_tree_stamp` already patches receipts this same way.
            let snapshot_revision = snapshot
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| {
                    sceneworks_core::model_artifacts::validate_immutable_revision(name).is_ok()
                })
                .map(|name| Value::String(name.to_owned()))
                .unwrap_or(Value::Null);
            (!resolved.is_empty()).then(|| json!({
                "schemaVersion": 2, "repo": repo,
                "modelId": model.get("id").cloned().unwrap_or(Value::Null),
                "variant": entry.get("variant").cloned().unwrap_or_else(|| Value::String("default".to_owned())),
                "manifestFiles": files, "resolvedFiles": resolved, "backfilled": true,
                "snapshotRevision": snapshot_revision,
            }))
        }).collect::<Vec<_>>();
    if receipts.is_empty() {
        return;
    }
    if std::fs::create_dir_all(managed_path).is_err() {
        return;
    }
    let _write_guard = RECEIPT_BACKFILL_WRITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Ok(_receipt_lock) = sceneworks_core::download_receipt::lock(managed_path) else {
        return;
    };
    if !receipt_file_sets(managed_path, &context.repo, Some(model_id)).is_empty() {
        return;
    }
    let mut merged = receipt_entries(managed_path);
    for new_entry in receipts {
        let identity = (
            new_entry.get("repo").and_then(Value::as_str),
            new_entry.get("modelId").and_then(Value::as_str),
            new_entry.get("variant").and_then(Value::as_str),
        );
        if let Some(existing) = merged.iter_mut().find(|existing| {
            (
                existing.get("repo").and_then(Value::as_str),
                existing.get("modelId").and_then(Value::as_str),
                existing.get("variant").and_then(Value::as_str),
            ) == identity
        }) {
            *existing = new_entry;
        } else {
            merged.push(new_entry);
        }
    }
    let mut receipt = merged[0].clone();
    receipt
        .as_object_mut()
        .unwrap()
        .insert("receipts".to_owned(), Value::Array(merged));
    let _ = sceneworks_core::download_receipt::write(
        &managed_path.join(".sceneworks-download-complete.json"),
        &receipt,
    );
}

#[cfg(test)]
#[path = "models/tests/import_gate_tests.rs"]
mod import_gate_tests;

#[cfg(test)]
#[path = "models/tests/download_receipt_tests.rs"]
mod download_receipt_tests;

// Resolve a model's install/cache state from its (optional) download source. A
// downloadable model checks the HF cache + the SceneWorks-managed dir; a non-download
// model (a local manifest entry) checks its declared installed path; otherwise it's
// simply absent.
fn is_flux3_runtime_model(model_id: &str) -> bool {
    matches!(model_id, "flux3_klein" | "flux3_klein_base")
}

fn flux3_selected_revision<'a>(download: Option<&'a Value>, repo: &str) -> Option<&'a str> {
    let download = download?;
    if download.get("repo").and_then(Value::as_str) != Some(repo) {
        return None;
    }
    let revision = download.get("revision")?.as_str()?;
    sceneworks_core::model_artifacts::validate_immutable_revision(revision)
        .is_ok()
        .then_some(revision)
}

fn flux3_selected_snapshot(
    data_dir: &FsPath,
    repo: &str,
    download: Option<&Value>,
    model_id: &str,
) -> Option<PathBuf> {
    let revision = flux3_selected_revision(download, repo)?;
    sceneworks_worker::flux3_receipted_snapshot(data_dir, repo, revision, model_id)
}

// Runtime readiness is separate from whole-repository download completion. A readable DiT is
// not a usable FLUX 3 bundle when its bundled encoder or an indexed shard is absent. Other
// families retain their existing cache policy; this card selects only its exact repo/revision.
fn model_cache_health_for_download(
    model_id: &str,
    repo: &str,
    download: Option<&Value>,
    cache_path: Option<&FsPath>,
    files: &[String],
    data_dir: &FsPath,
) -> Option<HuggingFaceCacheHealth> {
    if !is_flux3_runtime_model(model_id) {
        return cache_path.map(|root| huggingface_cache_health(root, files));
    }
    let Some(revision) = flux3_selected_revision(download, repo) else {
        return Some(HuggingFaceCacheHealth::missing(vec![
            "FLUX 3 requires its selected immutable repository revision".to_owned(),
        ]));
    };
    let Some(snapshot) = flux3_selected_snapshot(data_dir, repo, download, model_id) else {
        return Some(HuggingFaceCacheHealth::missing(vec![
            "FLUX 3 install verification required: no trusted exact-revision install receipt. Install or verify this pinned package through the downloader; complete cache files alone are not selectable.".to_owned(),
        ]));
    };
    if !snapshot.is_dir() {
        return Some(HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: vec![format!("snapshots/{revision}")],
        });
    }
    if let Some(tier @ ("fp8r" | "nvfp4")) = download
        .and_then(|row| row.get("variant"))
        .and_then(Value::as_str)
    {
        if !snapshot.join("variants").join(tier).exists() {
            return Some(HuggingFaceCacheHealth {
                installed: false,
                incomplete: false,
                missing_files: vec![format!("variants/{tier}")],
            });
        }
    }
    Some(
        match sceneworks_worker::flux3_runtime_tier_bundle_check(
            &snapshot,
            download
                .and_then(|d| d.get("variant"))
                .and_then(Value::as_str)
                .unwrap_or("bf16"),
            model_id,
        ) {
            Ok(()) => HuggingFaceCacheHealth::installed(),
            Err(error) => HuggingFaceCacheHealth::missing(vec![error]),
        },
    )
}

fn install_state_for(
    download_context: Option<DownloadContext>,
    model: &Value,
    data_dir: &FsPath,
) -> ModelCatalogEntryState {
    if model.get("family").and_then(Value::as_str) == Some("minimax-h3")
        && model
            .as_object()
            .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
            .is_some()
    {
        let entry = model.as_object().expect("imported entry");
        let checkpoint_id = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(entry)
            .expect("imported plan");
        let resolved = sceneworks_core::checkpoint_plan_store::CheckpointPlanStore::open(data_dir)
            .resolve(checkpoint_id);
        let mut missing = Vec::new();
        let mut bound_config = None;
        match resolved {
            Ok(resolved) => {
                if let Err(reason) =
                    sceneworks_core::minimax_h3_import::verify_entry_assembly(entry, &resolved.plan)
                {
                    missing.push(reason);
                }
                bound_config =
                    sceneworks_core::minimax_h3_import::bound_config(&resolved.plan.context)
                        .ok()
                        .flatten()
                        .map(|config| config.0);
            }
            Err(error) => missing.push(error.to_string()),
        }
        let source_present = missing.is_empty();
        let rows = model_co_requisite_downloads(model);
        if let Err(reason) =
            sceneworks_core::minimax_h3_import::selected_import_component_tier(entry)
        {
            missing.push(reason);
        }
        if rows.is_empty() {
            missing.push(
                "The import has no compatible component closure; re-import with the selected base"
                    .to_owned(),
            );
        }
        let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
            sceneworks_core::hf_home::model_source_library(data_dir),
        );
        for row in rows {
            if imported_component_presence(data_dir, &row)
                != sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence::Installed {
                missing.push(format!("{}: {}", row.get("repo").and_then(Value::as_str).unwrap_or("component"),
                    string_array_field(&row, "files").join(", ")));
            }
            if let Some(config) = &bound_config {
                if let (Some(repo), Some(revision)) = (
                    row.get("repo").and_then(Value::as_str),
                    row.get("revision").and_then(Value::as_str),
                ) {
                    if let Ok((_, snapshot)) =
                        resolver.discover_source_snapshot(repo, Some(revision))
                    {
                        if let Err(reason) = sceneworks_core::minimax_h3_import::import_component_configuration_compatible(&snapshot, &row, config) {
                            missing.push(reason);
                        }
                    }
                }
            }
        }
        return ModelCatalogEntryState {
            downloadable: false,
            installed_path: model_manifest_installed_path(model, data_dir)
                .map(|path| path.display().to_string()),
            installed: missing.is_empty(),
            cache_incomplete: !missing.is_empty(),
            missing_required_files: missing,
            update_available: false,
            imported_source_present: Some(source_present),
        };
    }
    if let Some(download_context) = download_context {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let selected_download = model_download(model);
        let managed_path = data_dir
            .join("models")
            .join(safe_download_dir(&download_context.repo));
        let cache_path = huggingface_repo_cache_path(data_dir, &download_context.repo);
        // Quant-matrix models (sc-8506/8508): the top-level install state aggregates across ALL
        // selectable tiers, not just the default one. A model that offers bf16/q8/q4 counts as
        // installed when ANY tier is fully present, and is only "incomplete" when a tier is genuinely
        // torn (partially downloaded) AND no complete tier exists. Installing a single valid tier —
        // even a non-default one — must never surface as an incomplete/repairable cache (sc-9907),
        // because that rendered a false "Cached files are incomplete" warning + Fix button on a
        // perfectly good install. Single-variant models keep the default-tier contract below.
        let mut tier_dependencies_missing = false;
        let (cache_installed, cache_incomplete, mut missing_required_files) =
            if model_has_variant_matrix(model) {
                let variants = model_variant_states(model, data_dir);
                tier_dependencies_missing =
                    variants.iter().any(|variant| variant.dependencies_missing);
                let any_installed = variants.iter().any(|variant| variant.installed);
                // A TORN tier — some of its files present but not all — is a genuine repair candidate:
                // it will fail to load, and re-downloading that tier fixes it. A never-fetched tier is
                // `missing` (`cache_incomplete == false` — nothing on disk to be torn), NOT torn, so it
                // never enters this find. That distinction is what makes it safe to surface repair even
                // when a complete sibling exists: sc-9907 (a clean single-tier install must not read
                // "incomplete + Fix" just because its OTHER tiers were never downloaded) is preserved by
                // `torn` excluding missing tiers, WITHOUT also suppressing a genuinely half-downloaded
                // sibling (sc-14431 — the old `!any_installed &&` guard hid a torn q8 behind a complete
                // q4, so chroma/sdxl/qwen/wan/lens all reported `repairAvailable: false` over a torn tier).
                let torn = variants
                    .iter()
                    .find(|variant| variant.cache_incomplete && !variant.installed);
                let incomplete = torn.is_some();
                // Report the torn tier's missing files so the model-level repair knows what to re-fetch,
                // even though a sibling tier is complete (`any_installed`).
                let missing = torn
                    .map(|variant| variant.missing_required_files.clone())
                    .unwrap_or_default();
                (any_installed, incomplete, missing)
            } else {
                let cache_health = model_cache_health_for_download(
                    model_id,
                    &download_context.repo,
                    selected_download.as_ref(),
                    cache_path.as_deref(),
                    &download_context.files,
                    data_dir,
                );
                let installed = cache_health.as_ref().is_some_and(|health| health.installed);
                let incomplete = cache_health
                    .as_ref()
                    .is_some_and(|health| health.incomplete);
                let missing = cache_health
                    .as_ref()
                    .map(|health| health.missing_files.clone())
                    .unwrap_or_default();
                // sc-13513: a single-variant no-`model_index` turnkey (Boogu's `base/` default download,
                // whose component globs match stray weights) reads coarse-installed while actually torn.
                // Downgrade so the TOP-LEVEL badge agrees with the per-variant state and the worker's
                // loader gate — otherwise `installState:"installed"`/`repairAvailable:false` renders a
                // false green over an install that then fails to load. SANA (variant-matrix) is already
                // covered by the aggregate branch above; Anima's exact-path filter catches torn coarsely.
                let family = model
                    .get("family")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if installed
                    && no_model_index_tier_is_torn(
                        family,
                        model.get("id").and_then(Value::as_str).unwrap_or_default(),
                        &download_context.files,
                        cache_path.as_deref(),
                        &managed_path,
                    )
                {
                    (false, true, vec![torn_tier_marker(&download_context.files)])
                } else {
                    (installed, incomplete, missing)
                }
            };
        // A quant-matrix model's top-level state is the aggregate of its tier-aware variant states
        // computed above (cache_installed = "any tier installed"). The repo-level managed marker must
        // NOT independently mark it installed (sc-9909): a stale .sceneworks-download-complete.json
        // left by an empty download would otherwise read the whole model as installed while every tier
        // reads missing. Single-variant models keep the repo-level managed contract.
        // sc-20526: drop any receipt a pre-fix backfill minted over a partially downloaded tier
        // BEFORE it is read, so an affected install self-heals on the next scan instead of needing a
        // manual delete — and so the backfill below can re-record the tier honestly.
        if !is_flux3_runtime_model(model_id) {
            repair_torn_backfilled_receipts(&managed_path, data_dir);
            repair_missing_snapshot_revisions(&managed_path, model, data_dir);
        }
        let receipt_file_sets = receipt_file_sets(
            &managed_path,
            &download_context.repo,
            model.get("id").and_then(Value::as_str),
        );
        let managed_installed = if is_flux3_runtime_model(model_id) {
            false // The shared receipt-resolved cache health already covers managed snapshots.
        } else {
            !model_has_variant_matrix(model)
                && receipt_file_sets.is_empty()
                && model_is_installed(&managed_path)
        };
        // Usable native FLUX 3 components do not prove every extra whole-repo file downloaded.
        // Only the actual downloader may publish that completion receipt for this source.
        if cache_installed && !is_flux3_runtime_model(model_id) {
            backfill_current_receipt(&managed_path, model, &download_context, data_dir);
        }
        let stale_files_present = !is_flux3_runtime_model(model_id)
            && !cache_installed
            && receipt_file_sets.iter().any(|receipt| {
                receipt_files_present(
                    data_dir,
                    &download_context.repo,
                    receipt,
                    model_family_tier_predicate(model),
                )
            });
        let breaking_update = model
            .get("breaking")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || model
                .get("downloads")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|entry| {
                    !is_co_requisite_download(entry)
                        && entry.get("repo").and_then(Value::as_str)
                            == Some(download_context.repo.as_str())
                        && string_array_field(entry, "files") == download_context.files
                        && entry
                            .get("breaking")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                });
        // A receipt proves the primary files exist, not that their dependencies still do.
        // Do not let it override missing dependencies. A partial nonbreaking primary update
        // may still use its complete older receipt.
        let usable_stale = stale_files_present && !breaking_update && !tier_dependencies_missing;
        let primary_installed = managed_installed || cache_installed || usable_stale;
        let installed_path = if is_flux3_runtime_model(model_id) {
            if cache_installed || (cache_incomplete && !managed_installed) {
                flux3_selected_snapshot(
                    data_dir,
                    &download_context.repo,
                    selected_download.as_ref(),
                    model_id,
                )
            } else if managed_installed {
                Some(managed_path.clone())
            } else {
                None
            }
        } else if cache_installed || cache_incomplete || usable_stale {
            cache_path.clone()
        } else {
            Some(managed_path)
        };
        // Co-requisites (sc-9696): the entry counts as installed only when the primary AND every
        // co-requisite dependency (e.g. the PiD decoder's shared gemma-2-2b-it caption encoder) are
        // cached. Gating on this keeps a feature that silently no-ops without its dependency (PiD →
        // native VAE) from advertising as ready, and a present primary with a missing/partial
        // co-requisite surfaces as a repairable partial install (cache_incomplete → repairAvailable),
        // whose repair re-runs the download job that now fetches the co-requisite too.
        let mut hard_co_requisites_installed = true;
        let mut soft_co_requisite_update = false;
        let mut co_requisite_incomplete = false;
        // Tier-scoped co-requisites (sc-14980) are gated like the tiers themselves: this state is a
        // model-level AGGREGATE that counts a quant-matrix model installed when ANY tier is fully
        // present, so requiring every tier's shared text encoder would report a complete, working q4
        // install as incomplete forever. Satisfy them when at least one tier's set is fully cached;
        // tier-agnostic co-requisites (every other family) keep the strict all-must-be-present rule.
        let tier_scoped: Vec<Value> = model_co_requisite_downloads(model)
            .into_iter()
            .filter(|download| co_requisite_variant(download).is_some())
            .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft"))
            .collect();
        // Current matrix variants already check their matching dependency sets with the primary.
        // Retain the dependency floor when a receipt is supplying an older primary instead.
        if (!model_has_variant_matrix(model) || usable_stale) && !tier_scoped.is_empty() {
            let tiers: std::collections::BTreeSet<String> = tier_scoped
                .iter()
                .filter_map(co_requisite_variant)
                .collect();
            let any_tier_complete = tiers.iter().any(|tier| {
                tier_scoped
                    .iter()
                    .filter(|download| co_requisite_variant(download).as_deref() == Some(tier))
                    .all(|download| {
                        co_requisite_cache_health(data_dir, download)
                            .is_some_and(|health| health.installed)
                    })
            });
            if !any_tier_complete {
                hard_co_requisites_installed = false;
                missing_required_files.extend(
                    tier_scoped
                        .iter()
                        .filter_map(|download| {
                            download
                                .get("repo")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .collect::<std::collections::BTreeSet<_>>(),
                );
            }
        }
        // A choice group (sc-22998, YuE2's decoder) is satisfied by ANY installed option; when none
        // is, a partially installed option (the one a repair completes), else the default, is what
        // the install reports missing.
        for co_requisite in co_requisite_rows_gating_install(
            model_co_requisite_downloads(model)
                .into_iter()
                .filter(|download| co_requisite_variant(download).is_none())
                .collect(),
            |download| co_requisite_presence(data_dir, download),
        ) {
            let Some(repo) = co_requisite.get("repo").and_then(Value::as_str) else {
                continue;
            };
            let health = co_requisite_cache_health(data_dir, &co_requisite);
            if health.as_ref().is_some_and(|health| health.installed) {
                continue;
            }
            let soft = co_requisite.get("required").and_then(Value::as_str) == Some("soft");
            if soft {
                soft_co_requisite_update = true;
            } else {
                hard_co_requisites_installed = false;
                co_requisite_incomplete |= health.as_ref().is_some_and(|health| health.incomplete);
            }
            match health
                .as_ref()
                .map(|health| health.missing_files.as_slice())
            {
                Some(missing) if !missing.is_empty() && !soft => missing_required_files
                    .extend(missing.iter().map(|file| format!("{repo}/{file}"))),
                _ if !soft => missing_required_files.push(repo.to_owned()),
                _ => {}
            }
        }
        ModelCatalogEntryState {
            downloadable: true,
            installed_path: installed_path.map(|path| path.display().to_string()),
            installed: primary_installed && hard_co_requisites_installed,
            cache_incomplete: cache_incomplete
                || (primary_installed && !hard_co_requisites_installed)
                || co_requisite_incomplete,
            missing_required_files,
            update_available: stale_files_present || soft_co_requisite_update,
            imported_source_present: None,
        }
    } else if let Some(installed_path) = model_manifest_installed_path(model, data_dir) {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: Some(installed_path.display().to_string()),
            installed: model_is_installed(&installed_path),
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
            imported_source_present: None,
        }
    } else {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: None,
            installed: false,
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
            imported_source_present: None,
        }
    }
}

/// Per co-requisite choice group, the options whose component is installed on this host
/// (`{"decoder": ["standard"]}`); `None` for a model with no choice group.
fn installed_co_requisite_choices(model: &Value, data_dir: &FsPath) -> Option<Value> {
    use sceneworks_core::model_artifacts::artifact_selection::{
        co_requisite_choice, model_co_requisite_downloads,
    };
    let mut groups: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for download in model_co_requisite_downloads(model) {
        let Some(choice) = co_requisite_choice(&download) else {
            continue;
        };
        let options = groups.entry(choice.group).or_default();
        let installed =
            co_requisite_cache_health(data_dir, &download).is_some_and(|health| health.installed);
        if installed && !options.contains(&choice.option) {
            options.push(choice.option);
        }
    }
    (!groups.is_empty()).then(|| json!(groups))
}

/// [`co_requisite_cache_health`] as the three-way presence choice-group gating reads (sc-22998).
/// Exact-revision availability for MOSS request-selected components. A sibling cached snapshot
/// cannot make the selected package installed. Hash authentication remains the native load gate.
pub(crate) fn pinned_component_presence(
    data_dir: &FsPath,
    download: &Value,
) -> sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return CoRequisitePresence::Absent;
    };
    let Some(revision) = download.get("revision").and_then(Value::as_str) else {
        return CoRequisitePresence::Absent;
    };
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision)) else {
        return CoRequisitePresence::Absent;
    };
    let files = string_array_field(download, "files");
    let health = huggingface_filtered_cache_health(&[snapshot], &files);
    if health.installed {
        CoRequisitePresence::Installed
    } else if health.incomplete {
        CoRequisitePresence::Incomplete
    } else {
        CoRequisitePresence::Absent
    }
}

fn imported_component_presence(
    data_dir: &FsPath,
    download: &Value,
) -> sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    let presence = if download.get("revision").and_then(Value::as_str).is_some() {
        pinned_component_presence(data_dir, download)
    } else {
        co_requisite_presence(data_dir, download)
    };
    if presence != CoRequisitePresence::Installed {
        return presence;
    }
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return CoRequisitePresence::Absent;
    };
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let complete = resolver
        .discover_source_snapshot(repo, download.get("revision").and_then(Value::as_str))
        .is_ok_and(|(_, snapshot)| {
            sceneworks_core::minimax_h3_import::import_component_files_complete(&snapshot, download)
        });
    if complete {
        CoRequisitePresence::Installed
    } else {
        CoRequisitePresence::Incomplete
    }
}

fn co_requisite_presence(
    data_dir: &FsPath,
    download: &Value,
) -> sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;
    match co_requisite_cache_health(data_dir, download) {
        Some(health) if health.installed => CoRequisitePresence::Installed,
        Some(health) if health.incomplete => CoRequisitePresence::Incomplete,
        _ => CoRequisitePresence::Absent,
    }
}

/// Resolve install-state health for a hard/soft co-requisite, including exact cache-only aliases
/// for upstream repository renames. This deliberately does not mutate the cache while serving the
/// catalog. It lets an existing immutable legacy snapshot remain usable/installed; the worker's
/// load-time resolver then consumes the same validated plain source and hard-links the canonical
/// cache namespace from those local bytes when possible.
fn co_requisite_cache_health(
    data_dir: &FsPath,
    download: &Value,
) -> Option<HuggingFaceCacheHealth> {
    match sceneworks_core::managed_file::ManagedFileIdentity::from_download(download) {
        Ok(Some(identity)) => {
            let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
                sceneworks_core::hf_home::model_source_library(data_dir),
            );
            let valid = resolver
                .discover_source_snapshot(&identity.repo, Some(&identity.revision))
                .map(|(_, snapshot)| snapshot.join(&identity.file))
                .is_ok_and(|path| identity.verify(&path).is_ok());
            return Some(if valid {
                HuggingFaceCacheHealth::installed()
            } else {
                HuggingFaceCacheHealth::missing(vec![identity.file])
            });
        }
        Err(_) => {
            return Some(HuggingFaceCacheHealth::missing(string_array_field(
                download, "files",
            )))
        }
        Ok(None) => {}
    }
    let repo = download.get("repo").and_then(Value::as_str)?;
    let files = string_array_field(download, "files");
    let exact_rename = download
        .get("revision")
        .and_then(Value::as_str)
        .and_then(|revision| {
            sceneworks_core::hf_repo_renames::exact_huggingface_repo_rename(repo, revision, &files)
        });
    let current = match exact_rename {
        Some(rename) => {
            exact_rename_side_cache_health(data_dir, rename.current_repo, rename, &files)
        }
        None => huggingface_repo_cache_path(data_dir, repo)
            .map(|path| huggingface_cache_health(&path, &files)),
    };
    if current.as_ref().is_some_and(|health| health.installed) {
        return current;
    }

    let exact_legacy = exact_rename.and_then(|rename| {
        exact_rename_side_cache_health(data_dir, rename.legacy_repo, rename, &files)
    });

    match (current, exact_legacy) {
        (_, Some(legacy)) if legacy.installed => Some(legacy),
        (Some(current), _) if current.incomplete => Some(current),
        (_, Some(legacy)) if legacy.incomplete => Some(legacy),
        (current, legacy) => current.or(legacy),
    }
}

fn co_requisite_satisfied_by_exact_legacy_rename(data_dir: &FsPath, download: &Value) -> bool {
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return false;
    };
    let Some(revision) = download.get("revision").and_then(Value::as_str) else {
        return false;
    };
    let files = string_array_field(download, "files");
    let Some(rename) =
        sceneworks_core::hf_repo_renames::exact_huggingface_repo_rename(repo, revision, &files)
    else {
        return false;
    };
    exact_rename_side_cache_health(data_dir, rename.legacy_repo, rename, &files)
        .is_some_and(|health| health.installed)
}

/// Evaluate an exact legacy alias with the same canonical-source confinement used by the worker's
/// load-time relink. A readable symlink is not sufficient: if it escapes the legacy repository
/// cache, treating it as installed would suppress the canonical repair download even though the
/// worker must reject it.
fn exact_rename_side_cache_health(
    data_dir: &FsPath,
    repo: &str,
    rename: &sceneworks_core::hf_repo_renames::HuggingFaceRepoRename,
    files: &[String],
) -> Option<HuggingFaceCacheHealth> {
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    if !huggingface_repo_cache_exists(&repo_root) {
        return Some(HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: Vec::new(),
        });
    }
    // Through the typed source-library seam rather than raw joins (sc-19711's artifact inventory
    // forbids the API constructing a model root directly; `crates/sceneworks-worker/src/model_jobs.rs`
    // is the one classified infrastructure surface allowed to). `repo_root` came from
    // `huggingface_repo_cache_path`, i.e. this same library, so the seam accepts it; `.root()` is the
    // very `huggingface_hub_cache_dir(data_dir)` this used to pass, unchanged.
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    let snapshot = library
        .snapshots_root_from_repository_root(&repo_root)
        .ok()?
        .join(rename.revision);
    if sceneworks_core::hf_repo_renames::validate_exact_huggingface_repo_rename_snapshot(
        library.root(),
        &repo_root,
        &snapshot,
        rename,
    )
    .is_some()
    {
        Some(HuggingFaceCacheHealth::installed())
    } else {
        Some(HuggingFaceCacheHealth::missing(files.to_vec()))
    }
}

/// Install-state health for a tier whose pinned revision SUPERSEDES an exact older one
/// ([`sceneworks_core::hf_revision_supersessions`]) while the tier exists ONLY at that older
/// revision — the v0.9.0 install of the Qwen-Image 2.1 q8/q4 re-host after sc-25909 republished it
/// to add each tier's NOTICE. The worker carries such a tier to the current pin cache-only at load
/// time, but only when the legacy snapshot holds every carried file at its exact size; the coarse
/// revision-agnostic probe would otherwise call a partial legacy tier installed and the render would
/// fail "not installed". Like the rename seam, serving the catalog never mutates the cache.
///
/// `None` (the ordinary probe decides) for every row without a declaration, and whenever the tier
/// is already present at the current pin.
fn superseded_revision_tier_cache_health(
    data_dir: &FsPath,
    download: &Value,
    files: &[String],
) -> Option<HuggingFaceCacheHealth> {
    use sceneworks_core::hf_revision_supersessions::{
        exact_huggingface_revision_supersession, supersession_tier,
        validate_superseded_legacy_snapshot,
    };
    let repo = download.get("repo").and_then(Value::as_str)?;
    let revision = download.get("revision").and_then(Value::as_str)?;
    let supersession = exact_huggingface_revision_supersession(repo, revision, files)?;
    let tier = supersession_tier(supersession)?;
    let repo_root = huggingface_repo_cache_path(data_dir, repo)?;
    if !huggingface_repo_cache_exists(&repo_root) {
        return None;
    }
    let library = sceneworks_core::hf_home::model_source_library(data_dir);
    let snapshots = library
        .snapshots_root_from_repository_root(&repo_root)
        .ok()?;
    let legacy_snapshot = snapshots.join(supersession.legacy_revision);
    if snapshots
        .join(supersession.current_revision)
        .join(tier)
        .exists()
        || !legacy_snapshot.join(tier).exists()
    {
        return None;
    }
    if validate_superseded_legacy_snapshot(
        library.root(),
        &repo_root,
        &legacy_snapshot,
        supersession,
    )
    .is_some()
    {
        return Some(HuggingFaceCacheHealth::installed());
    }
    let missing = supersession
        .carried
        .iter()
        .filter(|file| {
            std::fs::metadata(legacy_snapshot.join(file.path)).map_or(true, |metadata| {
                !metadata.is_file() || metadata.len() != file.size
            })
        })
        .map(|file| file.path.to_owned())
        .collect::<Vec<_>>();
    Some(HuggingFaceCacheHealth::missing(if missing.is_empty() {
        // Every entry reads at its size yet confinement failed (an escaping link): name the tier.
        vec![format!("{tier}/*")]
    } else {
        missing
    }))
}

// Per-variant install state (sc-8508, epic 8506): a single downloadable tier of a quant-matrix
// model. `install_state_for` reports the DEFAULT variant's install state (back-compat single-variant
// contract); `model_variants` reports one of these per declared download entry so the catalog knows
// WHICH tiers are on disk, not just whether *a* variant is.
struct ModelVariantState {
    /// The tier key: an explicit `downloads[].variant` (bf16/q8/q4), else "default" for a
    /// single-variant model (which has exactly one entry).
    variant: String,
    /// Whether this specific tier's files are present in the HF cache.
    installed: bool,
    /// Resolved install path for this tier (the shared repo cache root; tiers live as `files`-
    /// filtered subdirs within it). `None` when the repo has never been fetched.
    installed_path: Option<String>,
    /// This tier's incomplete-cache signal (some but not all `files` present).
    cache_incomplete: bool,
    /// A present primary is missing required companions; primary-only receipts cannot repair this.
    dependencies_missing: bool,
    /// Files this tier is missing from the cache (empty when complete or absent).
    missing_required_files: Vec<String>,
    /// This tier's estimated download size (from `downloads[].estimatedSizeBytes` /
    /// `footprint.diskSizeBytes`).
    download_size_bytes: Option<u64>,
    /// The raw `downloads[].footprint` object (disk size + optional measured memory), passed
    /// through verbatim for the RAM-suggestion surfaces (sc-8509/8516). `Null` when absent.
    footprint: Value,
    /// This tier is DECLARED but its artifact is not published yet (`downloads[].pendingArtifact`,
    /// sc-24112). The picker still lists it — the tier axis is real — but it can never be queued,
    /// never reads `installed`, and is rendered unavailable rather than as a download button.
    pending_artifact: bool,
    /// Whether the PER-TIER delete can reclaim this tier on its own (never for a pending row):
    /// true when the row carries a non-empty `files` scope, OR when it is a scope-less tier that is
    /// the sole row on its repo ([`is_sole_repo_tier`]) — `qwen_image_2_1`'s bf16 tier, which IS
    /// the whole upstream snapshot and shares that repo with nothing. A scope-less row on a repo a
    /// sibling shares is the model rather than a slice of it, and `DELETE
    /// /models/:id/variants/:variant` refuses it with "delete the whole model instead", so the UI
    /// must not offer a per-tier delete for it. Both answers come from [`tier_delete_scope`]'s own
    /// rule, so the button and the route cannot disagree.
    tier_deletable: bool,
    /// sc-22998: `Some` only for a locally derived tier (`downloads[].localDerivation`). `true` while
    /// no snapshot verified against the pinned derived weights exists at its location — the tier
    /// is then not installed; `POST /models/:id/download {variant}` fetches the original and the
    /// worker derives the tier locally (sc-22999).
    derivation_pending: Option<bool>,
}

// Whether `model`'s `downloads` array is a quant-matrix — i.e. at least one supported entry carries
// an explicit non-empty `variant` key (q4/q8/bf16). Entry COUNT is deliberately NOT a discriminator:
// the manifest uses multiple download entries for non-tier reasons too (alternate sources, native
// fallbacks, co-requisite TE repos — e.g. PiD backbone + gemma-2-2b-it, boogu mlx+native, krea, wan,
// ltx). Those are not quant matrices, so only variant-presence flags a model as one (sc-8508).
fn model_has_variant_matrix(model: &Value) -> bool {
    let Some(downloads) = model.get("downloads").and_then(Value::as_array) else {
        return false;
    };
    downloads
        .iter()
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .any(|entry| {
            entry
                .get("variant")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty())
        })
}

// The shared per-tier completeness predicate for a no-`model_index` MLX turnkey
// (`sceneworks_core::mlx_tier_completeness`), keyed on the manifest `family`, or `None` for a family
// whose per-tier completeness is already accurate — diffusers turnkeys (flux/qwen/…) via the
// `model_index.json` augmentation in `huggingface_filtered_cache_health`, or a family with no bespoke
// per-tier layout. The catalog uses this to downgrade a tier that clears the coarse `<tier>/*` glob but
// is actually torn (backbone present, text-encoder/VAE/tokenizer missing) to `incomplete`, so the
// /models report agrees with the worker's tier resolvers, which gate on these SAME predicates
// (sc-13513). Anima is included so its convert-output states (`mlx_convert_output_tier_states`) tighten;
// on its variants[] source download the exact-path `files` filter already reports torn accurately, so
// this is a no-op there.
//
// `model_id` refines the choice where one family spans two on-disk contracts: the `sensenova-u1`
// family covers both the base ids and the distilled `_fast` twins, and only the latter need the
// 8-step distill accounted for (sc-14432).
fn no_model_index_family_predicate(family: &str, model_id: &str) -> Option<fn(&FsPath) -> bool> {
    use sceneworks_core::mlx_tier_completeness as tc;
    match family {
        "anima" => Some(tc::anima_tier_complete),
        "boogu" => Some(tc::boogu_tier_complete),
        "sana" => Some(tc::sana_tier_complete),
        "scail2" => Some(tc::scail2_tier_complete),
        // sc-14432: the SenseNova-U1 turnkeys ship a FLAT unified tier (backbone + config at the tier
        // root, no `model_index.json`), so without a predicate a torn tier read `installed` and then
        // failed at load — "complete but unloadable", with re-downloading the only (useless) repair.
        // The `_fast` twins additionally need the pre-merged distill marker to be loadable, so the id —
        // not the family — picks the predicate. Dispatched through the SHARED id list the worker's tier
        // resolver uses, so an id the worker would not tighten is not tightened here either.
        "sensenova-u1" => tc::sensenova_tier_predicate(model_id),
        // Both MiniMax-H3 entries load both DiT partitions. Each partition's glob can match
        // an interrupted download, so require both indexed shard sets before advertising a tier.
        "minimax-h3" if tc::minimax_h3_tier_predicate(model_id).is_some() => {
            Some(|dir| tc::minimax_h3_tier_complete(dir) && tc::minimax_h3_ref_tier_complete(dir))
        }
        _ => None,
    }
}

// Whether a no-`model_index` MLX turnkey tier that CLEARED the coarse presence check is actually TORN —
// backbone present, but a text-encoder / VAE / tokenizer component missing per the shared family
// predicate. `false` when the family has no bespoke predicate, the `files` filter isn't a single tier
// subdir (e.g. the candle whole-repo SANA entry, `files: []`), or SOME candidate location holds a
// complete tier. Drives the coarse `installed` → `incomplete` downgrade in BOTH the per-variant states
// (`model_variant_states`) and the top-level aggregate (`install_state_for`), so the /models badge, the
// variant state, and the worker's loader gate all agree (sc-13513). Every cache snapshot AND the managed
// dir is checked with `any` — a SANA repo keeps two snapshots (tiered + flat), and a complete tier in one
// location must not be dragged down by a sibling that lacks it.
fn no_model_index_tier_is_torn(
    family: &str,
    model_id: &str,
    files: &[String],
    cache_path: Option<&FsPath>,
    managed_path: &FsPath,
) -> bool {
    let Some(complete) = no_model_index_family_predicate(family, model_id) else {
        return false;
    };
    let Some(tier) = tier_subdir_name(files) else {
        return false;
    };
    let mut tier_bases = cache_path
        .map(huggingface_snapshot_dirs)
        .unwrap_or_default();
    tier_bases.push(managed_path.to_path_buf());
    !tier_bases.iter().any(|base| complete(&base.join(&tier)))
}

// The `missingRequiredFiles` marker for a torn no-`model_index` tier (`no_model_index_tier_is_torn`),
// scoped to the tier subdir when the filter names one.
fn torn_tier_marker(files: &[String]) -> String {
    tier_subdir_name(files)
        .map(|tier| format!("{tier}/ (incomplete: missing model components)"))
        .unwrap_or_else(|| "incomplete: missing model components".to_owned())
}

// Select the canonical source entry for each emitted variant. A single-variant model yields one
// "default" entry; a quant matrix yields one per tier. Install probing and the later live-size
// refresh share this selector so their response-array positions cannot drift.
fn model_variant_downloads(model: &Value) -> Vec<&Value> {
    let Some(downloads) = model.get("downloads").and_then(Value::as_array) else {
        return Vec::new();
    };
    // Emitted variant keys must be unique: the single-variant case is exactly one "default", and a
    // quant matrix has one entry per distinct q4/q8/bf16 tier. Guard against a manifest that maps two
    // supported entries to the same key (unlabeled alternate sources both collapsing to "default", or
    // a duplicated `variant`) — keep the first, drop the rest, so downstream per-variant tracking
    // never emits two same-keyed states (sc-8508).
    let mut seen_variants = std::collections::HashSet::new();
    downloads
        .iter()
        // Co-requisites (sc-9696) are dependencies, not selectable tiers — never a variant state.
        .filter(|entry| is_supported_model_download(entry) && !is_co_requisite_download(entry))
        .filter(|entry| {
            let key = entry
                .get("variant")
                .and_then(Value::as_str)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "default".to_owned());
            seen_variants.insert(key)
        })
        .collect()
}

// Build the per-variant install state for every selected download entry. Each entry is probed
// independently against the HF cache using that tier's own `files` filter, so the catalog can
// advertise (e.g.) bf16 installed while q4 is missing.
fn model_variant_states(model: &Value, data_dir: &FsPath) -> Vec<ModelVariantState> {
    // sc-13513: the manifest `family` selects the shared per-tier completeness predicate for the
    // no-`model_index` MLX turnkeys (SANA/Boogu) so a torn tier reads `incomplete`, not `installed`.
    let family = model
        .get("family")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
    model_variant_downloads(model)
        .into_iter()
        .map(|entry| {
            let repo = entry
                .get("repo")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let files = string_array_field(entry, "files");
            let cache_path = huggingface_repo_cache_path(data_dir, &repo);
            let cache_health = superseded_revision_tier_cache_health(data_dir, entry, &files)
                .or_else(|| {
                    model_cache_health_for_download(
                        model_id,
                        &repo,
                        Some(entry),
                        cache_path.as_deref(),
                        &files,
                        data_dir,
                    )
                });
            let cache_installed = cache_health.as_ref().is_some_and(|health| health.installed);
            let mut cache_incomplete = cache_health
                .as_ref()
                .is_some_and(|health| health.incomplete);
            let mut missing_required_files = cache_health
                .as_ref()
                .map(|health| health.missing_files.clone())
                .unwrap_or_default();
            // The managed dir mirrors the default-download install path; a variant present there (a
            // directly-downloaded turnkey) counts as installed too — but the check must be TIER-aware.
            // A quant-matrix repo writes ONE repo-level completion marker no matter which tier was
            // fetched, so keying a per-tier "installed" on the bare marker made EVERY tier report
            // installed after any single tier's download (sc-9909). Require the tier's own files to
            // actually exist under the managed dir, not just the marker.
            let managed_path = data_dir.join("models").join(safe_download_dir(&repo));
            let managed_installed = if is_flux3_runtime_model(model_id) {
                false // The shared receipt resolver already judged this root.
            } else {
                managed_tier_installed(&managed_path, &files)
            };
            let mut installed = cache_installed || managed_installed;

            // sc-13513: the coarse `<tier>/*` glob (and the managed presence check) is satisfied by ANY
            // single nested file, so a TORN tier of a no-`model_index` MLX turnkey (SANA/Boogu — backbone
            // present, text-encoder/VAE/tokenizer missing) read `installed`. The shared family predicate
            // downgrades it to `incomplete`. Only ever downgrades a coarse `installed`; a clean-missing
            // tier is never promoted, and a family without a bespoke predicate (diffusers turnkeys,
            // accurate via the `model_index.json` augmentation) is untouched.
            if installed
                && no_model_index_tier_is_torn(
                    family,
                    model_id,
                    &files,
                    cache_path.as_deref(),
                    &managed_path,
                )
            {
                installed = false;
                cache_incomplete = true;
                let marker = torn_tier_marker(&files);
                if !missing_required_files.contains(&marker) {
                    missing_required_files.push(marker);
                }
            }

            // sc-22998: a locally derived tier's `files` are its ORIGINAL's, so the cache probe
            // above describes the original, never the tier. It is installed only when a derived
            // snapshot at its declared location verifies against the pinned derived weights (and
            // its companions, checked below, are present); otherwise it is `derivationPending`.
            let mut derivation_pending = None;
            let mut derived_path = None;
            let mut derived_exists = false;
            if sceneworks_core::model_artifacts::artifact_selection::declares_local_derivation(
                entry,
            ) {
                use sceneworks_core::model_artifacts::artifact_selection::{
                    derived_snapshot_state, local_derivation, local_derivation_snapshot_dir,
                    DerivedSnapshotState,
                };
                let variant = entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let located = local_derivation(entry).and_then(|derivation| {
                    local_derivation_snapshot_dir(data_dir, model_id, variant, &derivation)
                        .map(|dir| (dir, derivation))
                });
                let state = match &located {
                    Some((dir, derivation)) => derived_snapshot_state(dir, derivation),
                    None => DerivedSnapshotState::Invalid(
                        "the tier declares a malformed local derivation".to_owned(),
                    ),
                };
                missing_required_files.clear();
                cache_incomplete = false;
                let verified = state == DerivedSnapshotState::Verified;
                derived_exists = !matches!(state, DerivedSnapshotState::Absent);
                if let DerivedSnapshotState::Invalid(why) = &state {
                    missing_required_files.push(format!("derived {variant} snapshot: {why}"));
                }
                // The companion check below still applies to a verified derived tier.
                installed = verified && (!is_flux3_runtime_model(model_id) || cache_installed || managed_installed);
                if verified && !installed {
                    missing_required_files.push("FLUX 3 derived tier requires its complete pinned original text encoder and autoencoder".to_owned());
                }
                derivation_pending = Some(!installed);
                derived_path = located
                    .filter(|_| derived_exists)
                    .map(|(dir, _)| dir.display().to_string());
            }

            // A tier is usable only with its own required companions. Checking these only at
            // model level let a complete q4 encoder certify bf16 and hid its repair action.
            // An absent primary remains absent even if shared files from another tier exist.
            let mut dependencies_missing = false;
            if installed || cache_incomplete {
                // Any installed option of a choice group satisfies it (sc-22998).
                for download in co_requisite_rows_gating_install(
                    model_co_requisite_downloads_for_variant_all_options(
                        model,
                        entry.get("variant").and_then(Value::as_str),
                    ),
                    |download| co_requisite_presence(data_dir, download),
                )
                .into_iter()
                .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft"))
                {
                    let health = co_requisite_cache_health(data_dir, &download);
                    if health.as_ref().is_some_and(|health| health.installed) {
                        continue;
                    }
                    installed = false;
                    cache_incomplete = true;
                    dependencies_missing = true;
                    let repo = download
                        .get("repo")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let missing = health
                        .map(|health| health.missing_files)
                        .filter(|files| !files.is_empty())
                        .unwrap_or_else(|| string_array_field(&download, "files"));
                    if missing.is_empty() {
                        missing_required_files.push(repo.to_owned());
                    } else {
                        missing_required_files
                            .extend(missing.iter().map(|file| format!("{repo}/{file}")));
                    }
                }
            }

            let installed_path = if is_flux3_runtime_model(model_id) {
                if cache_installed || (cache_incomplete && !managed_installed) {
                    flux3_selected_snapshot(data_dir, &repo, Some(entry), model_id)
                } else if managed_installed {
                    Some(managed_path.clone())
                } else {
                    None
                }
            } else if cache_installed || cache_incomplete {
                cache_path
            } else if managed_installed {
                Some(managed_path)
            } else {
                None
            };
            // sc-24112: a DECLARED-but-unpublished tier can never be installed, whatever the cache
            // probe above happened to find. Forcing it here rather than filtering the row out keeps
            // the tier in the picker (the axis is real) while making every downstream
            // "installed?"/"queue it?" answer false through the ONE field they all read.
            let pending_artifact = is_pending_artifact_download(entry);
            if pending_artifact {
                installed = false;
                cache_incomplete = false;
            }
            ModelVariantState {
                variant: entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "default".to_owned()),
                installed,
                installed_path: if derivation_pending.is_some() {
                    derived_path
                } else {
                    installed_path.map(|path| path.display().to_string())
                },
                cache_incomplete,
                dependencies_missing,
                missing_required_files,
                download_size_bytes: manifest_download_size_bytes(model, entry)
                    .or_else(|| variant_footprint_disk_bytes(entry)),
                footprint: entry.get("footprint").cloned().unwrap_or(Value::Null),
                pending_artifact,
                tier_deletable: !pending_artifact
                    && if derivation_pending.is_some() {
                        // Only the derived snapshot is ever deleted, so only one that exists.
                        derived_exists
                    } else {
                        !files.is_empty() || is_sole_repo_tier(model, entry)
                    },
                derivation_pending,
            }
        })
        .collect()
}

// Whether a tier's OWN artifacts live in the app-managed turnkey dir (data/models/<repo>), as opposed
// to the shared HF cache. The repo-level completion marker (.sceneworks-download-complete.json) alone
// does NOT certify a tier: a quant-matrix repo writes exactly one marker regardless of which tier was
// downloaded, so a bare-marker check reported every tier of a repo installed after any single tier's
// fetch (sc-9909). Require BOTH the marker AND — for a tier that declares a `files` filter — that the
// tier's files actually exist under the managed dir. A single-variant turnkey (empty `files`) is
// certified by the marker alone, preserving the pre-matrix contract.
fn managed_tier_installed(managed_path: &FsPath, files: &[String]) -> bool {
    if !model_is_installed(managed_path) {
        return false;
    }
    files.is_empty()
        || files
            .iter()
            .all(|pattern| snapshot_contains_pattern(managed_path, pattern))
}

// The on-disk size a `downloads[].footprint.diskSizeBytes` declares, if any — the tier-scoped
// footprint signal (sc-8508) used as a fallback size when `estimatedSizeBytes` is absent.
fn variant_footprint_disk_bytes(download: &Value) -> Option<u64> {
    download
        .get("footprint")
        .and_then(|footprint| footprint.get("diskSizeBytes"))
        .and_then(json_size_to_u64)
}

// Gated-model signal (sc-1898): a machine-readable `gated` flag plus the credential
// host the download requires, so the Models screen can route the user to the
// credential screen before a download will succeed. The host honors an explicit
// manifest `credentialHost` and otherwise derives from the download provider/source
// URL; `licenseUrl` passes through untouched.
fn apply_gating_fields(object: &mut JsonObject) {
    let gated = object
        .get("gated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    object.insert("gated".to_owned(), Value::Bool(gated));
    if gated {
        let credential_host = object
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
            .or_else(|| derive_credential_host(object));
        object.insert(
            "credentialHost".to_owned(),
            credential_host.map(Value::String).unwrap_or(Value::Null),
        );
    }
}

// Mac UI gating (sc-3486): per-model Rust/MLX support so the web client can hide/
// disable a model with no native Mac lane in the pickers, plus (macOS only) the MLX availability +
// conversion status for models that declare an `mlx` variant. Additive fields the
// web/Docker build ignores; the client only acts on macSupport when the capabilities
// endpoint reports `macGatingActive`, so non-Mac pickers are untouched.
// Per-variant catalog fields (sc-8508): emit a `variants` array — one object per declared quant
// tier — plus a `hasVariantMatrix` boolean the web uses to decide whether to render a tier picker.
// A single-variant model gets a one-element array keyed "default" that mirrors the top-level
// install state, so the existing single-variant contract is unchanged.
fn apply_variant_fields(object: &mut JsonObject, data_dir: &FsPath) {
    let model = Value::Object(object.clone());
    let has_matrix = model_has_variant_matrix(&model);
    let variants = model_variant_states(&model, data_dir)
        .into_iter()
        .map(|variant| {
            let mut row = json!({
                "variant": variant.variant,
                "installed": variant.installed,
                // sc-24112: a pending tier is neither installed nor installable. `"pending"` is a
                // THIRD install state rather than `"missing"` so the web can say "not published
                // yet" instead of offering a download button that would queue a fetch of a
                // revision that does not resolve.
                "installState": if variant.pending_artifact {
                    "pending"
                } else if variant.installed {
                    "installed"
                } else if variant.derivation_pending == Some(true) {
                    // sc-22998: a locally derived tier with no verified derived snapshot. A distinct
                    // state so the web offers neither a download (there is nothing to fetch) nor
                    // "missing" (which it would read as "download me").
                    "derivationPending"
                } else {
                    "missing"
                },
                "pendingArtifact": variant.pending_artifact,
                "tierDeletable": variant.tier_deletable,
                "cacheState": if variant.cache_incomplete {
                    "incomplete"
                } else if variant.installed {
                    "complete"
                } else {
                    "missing"
                },
                "installedPath": variant
                    .installed_path
                    .map(Value::String)
                    .unwrap_or(Value::Null),
                "missingRequiredFiles": variant.missing_required_files,
                "downloadSizeBytes": variant
                    .download_size_bytes
                    .map(|value| json!(value))
                    .unwrap_or(Value::Null),
                "footprint": variant.footprint,
            });
            // Emitted only on a locally derived tier, so every other model's variant rows keep
            // their existing shape.
            if let (Some(pending), Some(row)) = (variant.derivation_pending, row.as_object_mut()) {
                row.insert("derivationPending".to_owned(), Value::Bool(pending));
            }
            row
        })
        .collect::<Vec<_>>();
    object.insert("hasVariantMatrix".to_owned(), Value::Bool(has_matrix));
    object.insert("variants".to_owned(), Value::Array(variants));
}

fn refresh_variant_download_sizes(model: &mut Value) -> Result<(), ApiError> {
    // A platformCleanupOnly tombstone strips the whole variants projection in
    // `apply_model_catalog_entry` — the missing array is its contract, not corruption, and there
    // is nothing to refresh. Keep the strict array guard below for every real catalog entry.
    if model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true) {
        return Ok(());
    }
    // `apply_variant_fields` runs in the blocking install-state sweep while live HF estimation runs
    // concurrently. Refresh only the already-built response fields after both complete: rebuilding
    // variants here would repeat filesystem probes serially and undo the intended overlap.
    let sizes = model_variant_downloads(model)
        .into_iter()
        .map(|download| {
            manifest_download_size_bytes(model, download)
                .or_else(|| variant_footprint_disk_bytes(download))
        })
        .collect::<Vec<_>>();
    let variants = model
        .get_mut("variants")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ApiError::internal("Model catalog variants must be an array"))?;
    if variants.len() != sizes.len() {
        return Err(ApiError::internal(
            "Model catalog variant count changed during size enrichment",
        ));
    }
    for (variant, size) in variants.iter_mut().zip(sizes) {
        let object = variant
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model catalog variant must be an object"))?;
        object.insert(
            "downloadSizeBytes".to_owned(),
            size.map(|value| json!(value)).unwrap_or(Value::Null),
        );
    }
    Ok(())
}

/// The convert-output quant tiers present under a converted MLX dir (sc-10730), highest-fidelity first.
/// Convert-at-install models (e.g. Anima) write `<converted>/<tier>/<backbone>/…` for each of
/// bf16/q8/q4 in ONE convert job — the tiers are convert OUTPUTS, not per-tier downloads. This lets the
/// Studio offer a generation-time tier picker via the decoupled `mlxTiers` catalog field WITHOUT the
/// download variant-matrix (`hasVariantMatrix`), whose `QuantDownloadPanel` would render bogus per-tier
/// download buttons for a model that has no per-tier download. Empty for a flat converted dir (no tier
/// subdirs) → the web renders no picker. Mirrors the worker tier resolvers' "tier present" probe so the
/// catalog and `anima_tier_subdir` agree on which tiers are loadable.
fn mlx_convert_output_tiers(converted_dir: &FsPath) -> Vec<&'static str> {
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| tier_subdir_has_weights(&converted_dir.join(tier)))
        .collect()
}

/// Per-tier install state for a convert-at-install model's FULL possible tier set (q4/q8/bf16), so the
/// Studio picker can show EVERY tier with the un-converted ones DISABLED — the same "show all, disable
/// unavailable" rule the download-matrix `variants[]` array gives. Unlike [`mlx_convert_output_tiers`]
/// (installed tiers only, which can only ever GROW the picker), this lists all three whether or not they
/// are on disk. Three states per tier: `"installed"`/`"complete"` (loadable), `"missing"`/`"incomplete"`
/// (a TORN tier — backbone present, a component gone; repairable by re-convert), and
/// `"missing"`/`"missing"` (never converted for this tier).
///
/// Completeness is family-specific (sc-13513): a convert-at-install family with a bespoke no-`model_index`
/// layout (Anima's `diffusion_models/ + text_encoders/ + vae/`) is gated on the SAME shared per-tier
/// predicate the worker's `anima_tier_subdir` resolver uses ([`no_model_index_family_predicate`]), so a
/// torn convert output reads `incomplete` rather than `installed`. Any other convert family keeps the
/// coarse [`tier_subdir_has_weights`] backbone probe.
fn mlx_convert_output_tier_states(
    converted_dir: &FsPath,
    family: &str,
    model_id: &str,
) -> Vec<Value> {
    let predicate = no_model_index_family_predicate(family, model_id);
    ["q4", "q8", "bf16"]
        .into_iter()
        .map(|tier| {
            let dir = converted_dir.join(tier);
            let has_backbone = tier_subdir_has_weights(&dir);
            let complete = match predicate {
                Some(complete) => complete(&dir),
                None => has_backbone,
            };
            let (install_state, cache_state) = if complete {
                ("installed", "complete")
            } else if has_backbone {
                ("missing", "incomplete")
            } else {
                ("missing", "missing")
            };
            // Convert-at-install tiers have no downloads[] row of their own, so this runtime catalog
            // state is the only truthful place to surface their per-tier size. Count the files the
            // tier can load, including symlinked shared weights: unlike converted_tier_real_bytes
            // (deletion accounting), this is a residency estimate, not reclaimable disk accounting.
            let disk_size_bytes = converted_tier_loaded_bytes(&dir);
            json!({
                "tier": tier,
                "installState": install_state,
                "cacheState": cache_state,
                "diskSizeBytes": disk_size_bytes,
            })
        })
        .collect()
}

/// Sum the bytes visible to a converted tier load, including symlinked weight files.
///
/// This intentionally differs from [`converted_tier_real_bytes`], which excludes symlinks because it
/// answers "what would deleting this tier reclaim?". The catalog memory floor instead needs "what
/// bytes can this tier hold resident?", so a shared text encoder or VAE linked into the tier must count.
/// Directory symlinks are followed because a converter may link a shared component directory rather
/// than its individual files. Canonical-directory de-duplication prevents cycles and counts a shared
/// component once even when more than one in-tier path aliases it.
fn converted_tier_loaded_bytes(tier_dir: &FsPath) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![tier_dir.to_path_buf()];
    let mut visited_dirs = std::collections::HashSet::new();
    let mut visited_files = std::collections::HashSet::new();
    while let Some(dir) = stack.pop() {
        let canonical = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
        if !visited_dirs.insert(canonical) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if is_hidden_file(&path) {
                continue;
            }
            let Ok(link_meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if link_meta.file_type().is_symlink() {
                if let Ok(target_meta) = std::fs::metadata(&path) {
                    if target_meta.is_file() {
                        let canonical =
                            std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                        if visited_files.insert(canonical) {
                            total = total.saturating_add(target_meta.len());
                        }
                    } else if target_meta.is_dir() {
                        stack.push(path);
                    }
                }
            } else if link_meta.is_dir() {
                stack.push(path);
            } else if link_meta.is_file() {
                let canonical = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if visited_files.insert(canonical) {
                    total = total.saturating_add(link_meta.len());
                }
            }
        }
    }
    (total > 0).then_some(total)
}

/// Whether a converted tier subdir holds loadable weights: a non-hidden `.safetensors` / `.index.json`
/// under a known backbone dir (`diffusion_models/` for Anima's Cosmos DiT, `transformer/` for other
/// DiTs, `unet/` for SDXL) or flat in the tier dir. A hidden `._*` AppleDouble sidecar is not a weight
/// (SceneWorks#1333), mirroring the worker resolvers.
fn tier_subdir_has_weights(tier_dir: &FsPath) -> bool {
    if !tier_dir.is_dir() {
        return false;
    }
    let dir_has_weight = |dir: &FsPath| -> bool {
        std::fs::read_dir(dir).is_ok_and(|entries| {
            entries.flatten().any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                !name.starts_with("._")
                    && (name.ends_with(".safetensors") || name.ends_with(".index.json"))
            })
        })
    };
    // A known backbone subdir (`diffusion_models/` Anima, `transformer/` DiTs, `unet/` SDXL), or flat
    // in the tier dir itself.
    dir_has_weight(tier_dir)
        || ["diffusion_models", "transformer", "unet"]
            .into_iter()
            .any(|sub| dir_has_weight(&tier_dir.join(sub)))
}

/// Withdraw a synthesized LoRA advertisement that no lane on THIS deployment can honour (the
/// sc-15328 class).
///
/// An imported / fine-tuned image model has no manifest row: `apply_model_manifest_defaults`
/// synthesizes `loraCompatibility.families = [family]` from the family token alone. For some
/// families that is a promise nothing keeps — currently a Mage-Flow fine-tune on either native
/// backend. Imported Krea 2 used to be another case, but both native single-file entrypoints now
/// accept adapters (sc-18480), so its synthesized advertisement remains intact.
/// Left standing, the picker offers adapters, `validate_lora_specs_for_model` passes, the job is
/// created, and NO worker claims it: it sits on "Waiting for an available GPU worker" forever, with
/// no error and no terminal state. That is strictly worse than a rejection.
///
/// Applied HERE — on the catalog projection every read goes through — rather than baked into the
/// stored manifest at import time, because the verdict is a property of the DEPLOYMENT, not of the
/// checkpoint. Keeping this a deployment projection also makes future backend-specific capability
/// changes safe when a data dir moves between platforms.
///
/// 🔴 The withdrawal is an EXPLICIT EMPTY `families` array, never `remove("loraCompatibility")`.
/// Removing the key is a no-op: `families_from_value_chain` (lib.rs) falls back to the top-level
/// `family` field, which an imported entry must carry for routing — so the strip sc-15328 shipped
/// changed nothing and its lane still hangs. An empty array is non-null, so it short-circuits that
/// fallback and `validate_lora_specs_for_model` refuses the submission LOUDLY and terminally with
/// "has no declared LoRA families". `supported: false` is the web's signal to fail CLOSED, because
/// `loraMatchesModel` treats an empty family set as "cannot gate" and would otherwise stay
/// permissive and keep offering every LoRA.
/// Binds [`apply_imported_lora_advertisement_for_lanes`] to the lanes THIS build can run: macOS
/// ships the in-process MLX worker and no candle engine; Windows/Linux/Docker ship candle and no
/// MLX. The verdict is derived from the same per-lane request gates the scheduler uses, so the
/// advertisement and claim behavior stay aligned.
///
/// The lane split is a ONE-LINE binding here and a parameter below precisely so the behaviour is
/// testable on both lanes from either platform.
fn apply_imported_lora_advertisement(object: &mut JsonObject) {
    // Deliberately the BUILD's linked engines, not `generation::enqueue_backend` — see that
    // function's "Scope" note. The pair is consulted asymmetrically below (a withdrawal crosses
    // lanes, a positive advertisement never does), which a one-hot routing answer cannot express,
    // and a macOS build links no candle engine to derive a positive candle verdict from.
    let mlx_lane = cfg!(target_os = "macos");
    // Resolved as a VERDICT rather than by checking whether the first pass wrote anything:
    // `apply_model_manifest_defaults` can already have synthesized a `loraCompatibility` object, so
    // "the key exists" says nothing about whether this projection decided anything.
    let verdict = imported_lora_advertisement_verdict(object, mlx_lane, !mlx_lane).or_else(|| {
        // This build's lane had NO OPINION, which is not the same as "adapters are fine". Leaving
        // the entry untouched restores the permissive `family` fallback — the precise silent
        // re-advertisement sc-15328 fixed, where the API accepted a LoRA submission no worker could
        // claim and the job queued forever with no error. A generated Mage full fine-tune on any
        // non-macOS deployment is exactly that shape: candle registers no `mage-flow` imported
        // provider, so only MLX can say "serves the family, refuses adapters".
        //
        // So a WITHDRAWAL crosses lanes; a positive advertisement never does. Offering adapters
        // this build cannot claim is the same hang pointed the other way, which is what the filter
        // to the refusing verdict prevents.
        imported_lora_advertisement_verdict(object, !mlx_lane, mlx_lane).filter(|serves| !serves)
    });
    if let Some(serves_loras) = verdict {
        write_imported_lora_advertisement(object, serves_loras);
    }
}

/// Test-facing lane-scoped projection: resolve ONE lane pair and write its verdict verbatim, with no
/// cross-lane fallback. Production goes through [`apply_imported_lora_advertisement`], which adds
/// the withdrawal fallback; keeping this seam separate is what lets a test state each lane's own
/// truth from either platform without the fallback blurring the two.
#[cfg(test)]
fn apply_imported_lora_advertisement_for_lanes(
    object: &mut JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) {
    let Some(serves_loras) =
        imported_lora_advertisement_verdict(object, mlx_lane_available, candle_lane_available)
    else {
        return;
    };
    write_imported_lora_advertisement(object, serves_loras);
}

/// The advertisement verdict for one lane pair, with no mutation: `Some(true)` serves adapters,
/// `Some(false)` serves the family and refuses them, `None` is "not on this seam at all" (a builtin,
/// a non-image row, or a family no supplied lane registers).
fn imported_lora_advertisement_verdict(
    object: &JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) -> Option<bool> {
    // An imported plan-backed H3 video entry serves the family's adapter surface when ANY valid
    // supported shape is claimable with a compatible adapter: every UI mode is probed with its
    // real conditioning through the same predicates the support projection uses, so a
    // reference-partition (Ref2VA) import is advertised through its reference shape rather than
    // withdrawn by a probe no shape of it can answer.
    if object.get("type").and_then(Value::as_str) == Some("video")
        && sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(object)
    {
        return Some(
            sceneworks_core::jobs_store::imported_video_adapter_surface_claimable(
                object,
                mlx_lane_available,
                candle_lane_available,
            ),
        );
    }
    if object.get("type").and_then(Value::as_str) != Some("image") {
        return None;
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|id| !id.is_empty())?;
    let family = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)?;
    let source = object
        .get("importSourceShape")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|source| {
            matches!(
                *source,
                "transformer_file" | "fused_checkpoint" | "transformer_directory" | "comfy_ui_tree"
            )
        })?;
    sceneworks_core::jobs_store::imported_image_model_lora_advertisement(
        &id,
        &family,
        source,
        mlx_lane_available,
        candle_lane_available,
    )
}

/// Write the verdict onto the entry. Split from the resolution above so the production binding can
/// resolve twice — its own lane, then a cross-lane withdrawal — without projecting twice.
fn write_imported_lora_advertisement(object: &mut JsonObject, serves_loras: bool) {
    let Some(family) = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)
    else {
        return;
    };
    let compatibility = object
        .entry("loraCompatibility".to_owned())
        .or_insert_with(|| json!({}));
    if !compatibility.is_object() {
        *compatibility = json!({});
    }
    let Some(compatibility) = compatibility.as_object_mut() else {
        return;
    };
    if serves_loras {
        let families = compatibility
            .entry("families".to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        if !families.is_array() {
            *families = Value::Array(Vec::new());
        }
        if let Some(families) = families.as_array_mut() {
            if !families
                .iter()
                .any(|value| value.as_str() == Some(family.as_str()))
            {
                families.push(Value::String(family));
            }
        }
        compatibility.insert("supported".to_owned(), Value::Bool(true));
    } else {
        // Explicit empty wins over permissive family fallback in the validator.
        compatibility.insert("families".to_owned(), Value::Array(Vec::new()));
        compatibility.insert("supported".to_owned(), Value::Bool(false));
    }
}

/// A manifest that DECLARES `loraCompatibility.families: []` (as `qwen_image_2_1` did until
/// sc-24156 gave it the `qwen-image-2-1` family) is refused by `validate_lora_specs_for_model`
/// with "has no declared LoRA families" — but the web's `loraMatchesModel` reads an empty family set as "cannot gate" and stays
/// permissive, so it offered every LoRA and auto-applied the Krea `image_edit` LoRA in edit mode.
/// Stamp the same `supported: false` the imported withdrawal writes, so the web fails closed on
/// exactly the models the API refuses. An explicit `supported` already on the entry is kept.
fn mark_empty_lora_advertisement_unsupported(object: &mut JsonObject) {
    let Some(compatibility) = object
        .get_mut("loraCompatibility")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    if compatibility
        .get("families")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        compatibility
            .entry("supported".to_owned())
            .or_insert(Value::Bool(false));
    }
}

/// Project the exact active-provider surface for a stamped imported source shape. Family siblings
/// are never unioned: a Krea transformer file, SDXL fused checkpoint, Mage directory, and external
/// ComfyUI tree each see only registrations for their structural loader.
fn apply_imported_provider_surface(object: &mut JsonObject) {
    // The BUILD's linked engines, like its `apply_imported_lora_advertisement` sibling — not
    // `generation::enqueue_backend`. See that function's "Scope" note.
    let mlx_lane = cfg!(target_os = "macos");
    apply_imported_provider_surface_for_lanes(object, mlx_lane, !mlx_lane);
}

/// The Mac and off-Mac support blocks for an imported, plan-backed video entry (sc-26680).
///
/// The id-keyed pass in `apply_mac_and_mlx_fields` asks the claim predicates about a probe carrying
/// no manifest entry, so for an imported id every plan-backed predicate answers `false` and the
/// Video Studio would hide a model both native lanes serve. This replaces both blocks with the
/// entry-keyed verdicts, derived from the SAME predicates the scheduler and the no-lane gate ask
/// about a submitted job. Emitted on every platform, like the id-keyed blocks it replaces.
fn apply_imported_video_support_surface(object: &mut JsonObject) {
    // The existing picker must not synthesize a q4/q8 shared precision for an import whose
    // admitted closure is bf16 (or vice versa). Imported DiT precision remains source-owned.
    let tiers = sceneworks_core::minimax_h3_import::selected_import_component_tier(object)
        .map(|tier| vec![tier])
        .unwrap_or_default();
    object.insert("runtimeQuantTiers".to_owned(), json!(tiers));
    if let Ok(mac_support) =
        serde_json::to_value(sceneworks_core::jobs_store::imported_video_model_mac_support(object))
    {
        object.insert("macSupport".to_owned(), mac_support);
    }
    if let Ok(candle_support) = serde_json::to_value(
        sceneworks_core::jobs_store::imported_video_model_candle_support(object),
    ) {
        object.insert("candleSupport".to_owned(), candle_support);
    }
}

/// The product surface of an imported, plan-backed checkpoint (sc-26687): what was imported
/// (source variant), what it can do (supported operations), where it can run (supported backends),
/// whether its components are assembled (component readiness), and what to do about the ones that
/// are not (actionable repair steps).
///
/// ONE derivation behind Model Manager, Video Studio, the API and MCP, so the repair a card offers
/// is the same repair the API describes. Derived server-side from the entry and its bound plan; the
/// client never supplies any of it. Written on the response clone beside the neighbouring
/// `modelAvailability` / `modelResolution` / `cacheEligibility` projections — these describe this
/// deployment and this install state, so they are runtime projections rather than persisted
/// manifest keys.
fn apply_imported_checkpoint_product_surface(
    object: &mut JsonObject,
    data_dir: &FsPath,
    source_present: Option<bool>,
) {
    if object.get("catalogScope").and_then(Value::as_str) == Some("builtin") {
        return;
    }
    if sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(object).is_none() {
        return;
    }
    object.insert(
        "importedSource".to_owned(),
        imported_source_description(object),
    );
    object.insert(
        "supportedOperations".to_owned(),
        Value::Array(imported_supported_operations(object)),
    );
    // Omitted rather than emptied when the catalog carries NO per-lane verdict at all. An empty
    // array is a real "neither backend here", and saying that about an entry nothing has judged
    // would read as a refusal the scheduler never made.
    if let Some(backends) = imported_supported_backends(object) {
        object.insert("supportedBackends".to_owned(), Value::Array(backends));
    }
    let (readiness, repairs) = imported_component_readiness(object, data_dir, source_present);
    object.insert("componentReadiness".to_owned(), readiness);
    object.insert("repairActions".to_owned(), Value::Array(repairs));
}

/// Which source variant was imported and how it is held.
///
/// `sourceVariant` is the axis that actually distinguishes one imported checkpoint from its
/// siblings. For MiniMax-H3 that is the task partition the checkpoint DECLARED (fl2va / ref2va /
/// hybrid) — filenames and directory names prove nothing about task compatibility, which is
/// precisely why the declaration is the variant. An import carrying no such declaration has only
/// the structural source shape to distinguish it by, so that is what it reports.
fn imported_source_description(object: &JsonObject) -> Value {
    let checkpoint_id =
        sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(object).unwrap_or_default();
    // Ownership rides the checkpoint id's own prefix — the convention every reader of
    // `importPlan.checkpointId` already speaks (`apps/web/src/checkpointLibrary.js`
    // `modelOwnership`), so this and the web's linked/managed split cannot drift apart.
    let ownership_mode = if checkpoint_id.starts_with("linked/") {
        "linked"
    } else {
        "managed"
    };
    let declared_partition =
        sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(object)
            .ok()
            .map(|partition| partition.as_str());
    let source_shape = object
        .get("importSourceShape")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    // The assembly names the imported DiT under `transformer`, except on a Ref2VA import where the
    // DiT is the reference partition and lives under `transformer_ref`. Find whichever binding the
    // plan actually owns rather than hard-coding one key.
    let plan_binding = sceneworks_core::minimax_h3_import::entry_assembly(object)
        .and_then(|assembly| assembly.get("components"))
        .and_then(Value::as_object)
        .and_then(|components| {
            components
                .iter()
                .find(|(_, binding)| binding.get("source").and_then(Value::as_str) == Some("plan"))
        });
    let transformer_directory = plan_binding
        .and_then(|(_, binding)| binding.get("directory"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let sharded = plan_binding
        .and_then(|(_, binding)| binding.get("layerIds"))
        .and_then(Value::as_array)
        .is_some_and(|layers| layers.len() > 1);
    json!({
        "family": object.get("family").cloned().unwrap_or(Value::Null),
        "sourceShape": source_shape,
        "ownershipMode": ownership_mode,
        "sourceVariant": declared_partition.or(source_shape),
        "declaredPartition": declared_partition,
        "baseModel": object.get("baseModel").cloned().unwrap_or(Value::Null),
        "componentTier": object.get("importComponentTier").cloned().unwrap_or(Value::Null),
        "transformerDirectory": transformer_directory,
        "sharded": sharded,
    })
}

/// The operations this import serves, from its EXPLICIT task contract.
///
/// An H3 import answers from the partition it declared, because identical shapes do not prove
/// partition compatibility. Any other import's operations are whatever its registered route
/// projected onto `capabilities`, which is already the operation contract for it — and the field
/// the recipe gate (`recipe_presets::validate_recipe_preset_model_workflow`) reads, so the two can
/// never disagree about what a saved recipe may ask for.
fn imported_supported_operations(object: &JsonObject) -> Vec<Value> {
    if let Ok(partition) =
        sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(object)
    {
        let mut operations = Vec::new();
        if partition.supports_base() {
            operations.extend(["text_to_video", "image_to_video", "first_last_frame"]);
        }
        if partition.supports_references() {
            operations.push("reference_to_video");
        }
        return operations
            .into_iter()
            .map(|operation| Value::String(operation.to_owned()))
            .collect();
    }
    object
        .get("capabilities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The native backends this deployment can actually claim the import on, read off the same
/// per-lane verdicts the scheduler's claim predicates and the Video Studio gate consult.
///
/// `None` when the entry carries neither verdict — "nobody has judged this", which is a different
/// answer from `Some(vec![])` ("both lanes looked and neither will claim it").
fn imported_supported_backends(object: &JsonObject) -> Option<Vec<Value>> {
    let verdict = |key: &str| {
        object
            .get(key)
            .and_then(|block| block.get("supported"))
            .and_then(Value::as_bool)
    };
    let (mac, candle) = (verdict("macSupport"), verdict("candleSupport"));
    if mac.is_none() && candle.is_none() {
        return None;
    }
    let mut backends = Vec::new();
    if mac == Some(true) {
        backends.push(Value::String("mlx".to_owned()));
    }
    if candle == Some(true) {
        backends.push(Value::String("candle-cuda".to_owned()));
    }
    Some(backends)
}

/// Whether this import's assembly is COMPLETE here, component by component, plus the repair that
/// would make each missing component complete.
///
/// The imported DiT is the entry's own plan/source closure, independent of aggregate cache state. The
/// rest are the resident components the assembly reuses from the declared compatible base, each
/// resolved through the shared component-presence probe — never through a filename or a repository
/// allowlist.
fn imported_component_readiness(
    object: &JsonObject,
    data_dir: &FsPath,
    source_present: Option<bool>,
) -> (Value, Vec<Value>) {
    use sceneworks_core::model_artifacts::artifact_selection::CoRequisitePresence;

    let mut components = Vec::new();
    let mut repairs = Vec::new();
    let state_name = |presence: CoRequisitePresence| match presence {
        CoRequisitePresence::Installed => "installed",
        CoRequisitePresence::Incomplete => "incomplete",
        CoRequisitePresence::Absent => "missing",
    };

    // A declaration the contract refuses (an absent or incompatible `baseModel` / task partition)
    // cannot be repaired by downloading anything: it needs the user to declare the compatible base.
    let declaration_ok = sceneworks_core::minimax_h3_import::MiniMaxH3Partition::from_entry(object)
        .is_ok()
        || !sceneworks_core::minimax_h3_import::entry_requires_minimax_h3_family(object);
    if !declaration_ok {
        repairs.push(json!({
            "id": "declare-compatible-base",
            "kind": "select-base",
            "label": "Declare the compatible base model and task partition (fl2va, ref2va, or hybrid)",
        }));
    }

    let dit_present = source_present.unwrap_or_else(|| imported_source_present(object, data_dir));
    components.push(json!({
        "componentId": "transformer",
        "role": "imported_dit",
        "source": "plan",
        "model": Value::Null,
        "present": dit_present,
        "state": if dit_present { "installed" } else { "missing" },
    }));
    if !dit_present {
        // The damaged SOURCE is never re-fetched by the stock model downloader: a managed import
        // must go through the checkpoint import panel again (the inspector re-validates and
        // re-ingests the bytes under managed ownership), and a linked checkpoint keeps the Model
        // Manager's linked-repair affordance, which is the seam that decides relink versus rescan
        // from the library root index — deliberately not re-derived from a path here.
        let linked = sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id(object)
            .is_some_and(|id| id.starts_with("linked/"));
        repairs.push(if linked {
            json!({
                "id": "repair-linked-checkpoint",
                "kind": "rescan",
                "label": "Relink or rescan the linked checkpoint in Model Manager",
            })
        } else {
            json!({
                "id": "repair-imported-checkpoint",
                "kind": "reimport",
                "label": "Re-import the damaged checkpoint in the checkpoint import panel",
            })
        });
    }

    for (index, row) in object
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let presence = imported_component_presence(data_dir, row);
        let component_id = imported_component_id(row, index);
        let source = row.get("repo").and_then(Value::as_str).unwrap_or_default();
        let present = presence == CoRequisitePresence::Installed;
        components.push(json!({
            "componentId": component_id,
            "role": "resident",
            "source": "resident",
            "model": source,
            "present": present,
            "state": state_name(presence),
        }));
        if !present {
            repairs.push(json!({
                "id": format!("install-component:{component_id}"),
                "kind": "download",
                "componentId": component_id,
                "variant": object.get("importComponentTier").cloned().unwrap_or(Value::Null),
                "label": format!("Download the compatible {component_id} from {source}"),
            }));
        }
    }

    // An incomplete assembly is never READY: `components` says which member is at fault and
    // `repairActions` says what to do, but nothing here may let a half-provisioned import look
    // selectable-and-runnable.
    let ready = declaration_ok
        && components
            .iter()
            .all(|component| component.get("present").and_then(Value::as_bool) == Some(true));
    (json!({ "ready": ready, "components": components }), repairs)
}

fn apply_imported_provider_surface_for_lanes(
    object: &mut JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) {
    if object.get("catalogScope").and_then(Value::as_str) != Some("builtin")
        && object.get("type").and_then(Value::as_str) == Some("video")
        && sceneworks_core::minimax_h3_import::entry_is_imported_minimax_h3(object)
    {
        apply_imported_video_support_surface(object);
        return;
    }
    if object.get("type").and_then(Value::as_str) != Some("image")
        || object.get("catalogScope").and_then(Value::as_str) == Some("builtin")
    {
        return;
    }
    let Some(family) = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)
    else {
        return;
    };
    let Some(source) = object
        .get("importSourceShape")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|source| {
            matches!(
                *source,
                "transformer_file" | "fused_checkpoint" | "transformer_directory" | "comfy_ui_tree"
            )
        })
        .map(str::to_owned)
    else {
        return;
    };

    let mlx_routes = sceneworks_core::jobs_store::imported_provider_routes("mlx", &family)
        .filter(|route| route.source == source)
        .collect::<Vec<_>>();
    let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    let mac_support = if mlx_routes.is_empty() {
        serde_json::to_value(sceneworks_core::jobs_store::model_mac_support(
            id, "image", None,
        ))
    } else {
        Ok(json!({
            "supported": true,
            "features": {
                "pose": mlx_routes.iter().any(|route| {
                    route.operation == "pose"
                        && route.conditioning.iter().any(|kind| kind == "control")
                }),
                "reference": mlx_routes.iter().any(|route| {
                    route.operation == "generate"
                        && route
                            .conditioning
                            .iter()
                            .any(|kind| matches!(kind.as_str(), "reference" | "multi_reference"))
                }),
                "edit": mlx_routes.iter().any(|route| route.operation == "edit"),
                "lycoris": mlx_routes
                    .iter()
                    .any(|route| route.supports_lora || route.supports_lokr),
            }
        }))
    };
    if let Ok(mac_support) = mac_support {
        object.insert("macSupport".to_owned(), mac_support);
    }

    let routes = [
        ("mlx", mlx_lane_available),
        ("candle", candle_lane_available),
    ]
    .into_iter()
    .filter(|(_, available)| *available)
    .flat_map(|(backend, _)| {
        sceneworks_core::jobs_store::imported_provider_routes(backend, &family)
    })
    .filter(|route| route.source == source)
    .collect::<Vec<_>>();
    project_imported_operation_surface(object, &routes);
}

/// Replace family-wide defaults with the operations exposed by this exact source/backend route set.
/// Imported rows are initially enriched by `apply_model_manifest_defaults`, so merely adding provider
/// capabilities leaves unsupported sibling operations visible. This projection is intentionally
/// subtractive for every route-owned field while preserving unrelated author metadata.
fn project_imported_operation_surface(
    object: &mut JsonObject,
    routes: &[&sceneworks_core::jobs_store::ImportedProviderSurface],
) {
    let generates_reference = routes.iter().any(|route| {
        route.operation == "generate"
            && route
                .conditioning
                .iter()
                .any(|kind| matches!(kind.as_str(), "reference" | "multi_reference"))
    });
    let pose = routes.iter().any(|route| {
        route.operation == "pose" && route.conditioning.iter().any(|kind| kind == "control")
    });
    let edit = routes.iter().any(|route| route.operation == "edit");
    let edit_multi_reference = routes.iter().any(|route| {
        route.operation == "edit"
            && route
                .conditioning
                .iter()
                .any(|kind| kind == "multi_reference")
    });
    let multi_phase = routes.iter().any(|route| route.operation == "multi_phase");
    let generates = routes.iter().any(|route| {
        matches!(
            route.operation.as_str(),
            "generate" | "pose" | "multi_phase"
        )
    });

    let capabilities = object
        .entry("capabilities".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !capabilities.is_array() {
        *capabilities = Value::Array(Vec::new());
    }
    if let Some(capabilities) = capabilities.as_array_mut() {
        capabilities.retain(|value| {
            !matches!(
                value.as_str(),
                Some("text_to_image" | "edit_image" | "image_to_image" | "character_image")
            )
        });
        if generates {
            capabilities.push(Value::String("text_to_image".to_owned()));
        }
        if edit {
            capabilities.push(Value::String("edit_image".to_owned()));
        }
    }

    let ui = object.entry("ui".to_owned()).or_insert_with(|| json!({}));
    if !ui.is_object() {
        *ui = json!({});
    }
    if let Some(ui) = ui.as_object_mut() {
        if generates_reference {
            ui.insert("img2img".to_owned(), Value::Bool(true));
        } else {
            ui.remove("img2img");
            ui.remove("img2imgStrength");
        }
        if pose {
            ui.insert("poseLibrary".to_owned(), Value::Bool(true));
            ui.insert("poseControlScale".to_owned(), Value::Bool(true));
            ui.insert("controlModes".to_owned(), json!(["pose"]));
        } else {
            ui.remove("poseLibrary");
            ui.remove("poseControlScale");
            ui.remove("controlModes");
        }
        if !edit_multi_reference {
            ui.remove("editReferences");
        }
    }

    // An empty route set means this backend has no opinion about adapter compatibility. Do not
    // manufacture an empty object: catalog consumers distinguish abstention from an explicit
    // provider-owned compatibility surface. Once at least one exact route exists, keep the existing
    // object-shaped projection so supported routes retain a stable schema even when they expose no
    // acceleration adapter.
    if routes.is_empty() {
        object.remove("loraCompatibility");
    } else {
        let compatibility = object
            .entry("loraCompatibility".to_owned())
            .or_insert_with(|| json!({}));
        if let Some(compatibility) = compatibility.as_object_mut() {
            if let Some(types) = compatibility.get_mut("types").and_then(Value::as_array_mut) {
                types.retain(|value| value.as_str() != Some("acceleration"));
            }
            if multi_phase {
                let types = compatibility
                    .entry("types".to_owned())
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Some(types) = types.as_array_mut() {
                    types.push(Value::String("acceleration".to_owned()));
                }
            }
        }
    }

    object.remove("runtimeQuantTiers");

    // A checkpoint STORED as NVFP4 has exactly one runtime tier: its own (sc-11043, epic 11037).
    // The q4/q8/bf16 sweep below describes what a route can quantize a dense import TO, which is a
    // question that does not arise here — these bytes are already packed E2M1 and nothing
    // re-quantizes them. Offering the sweep for such an entry advertised `q4` for an NVFP4
    // checkpoint, which is precisely the alias E2 forbids, and the admission gate then refused
    // every job the offer produced: the picker showed tiers the router would not accept.
    //
    // Keyed on the SOURCE codec — what the weights are stored in — never on what a load would
    // materialize them as, which is a per-run receipt this catalog projection cannot know.
    if sceneworks_core::jobs_store::imported_entry_source_codec(object)
        == Some(sceneworks_core::checkpoint_weight_facts::NVFP4_CODEC_ID)
    {
        object.insert(
            "runtimeQuantTiers".to_owned(),
            Value::Array(vec![Value::String("nvfp4".to_owned())]),
        );
        return;
    }

    let mut runtime_quant_tiers = Vec::new();
    for tier in ["q4", "q8"] {
        if routes.iter().any(|route| {
            route
                .supported_quants
                .iter()
                .any(|candidate| candidate == tier)
        }) {
            runtime_quant_tiers.push(Value::String(tier.to_owned()));
        }
    }
    if routes.iter().any(|route| route.source != "comfy_ui_tree") {
        runtime_quant_tiers.push(Value::String("bf16".to_owned()));
    }
    if !runtime_quant_tiers.is_empty() {
        object.insert(
            "runtimeQuantTiers".to_owned(),
            Value::Array(runtime_quant_tiers),
        );
    }
}

/// True when this entry is a convert-at-install model that loads from a **converted artifact on the
/// platform this build targets** — so its conversion state belongs in the catalog here (sc-20529).
///
/// The converter half is [`sceneworks_core::jobs_store::convert_artifact_required_here`], the SAME
/// predicate the worker's unconverted-model preflight gates on. Re-implementing the
/// converter-membership test here (the original shape of this function) put two copies of one
/// platform rule on either side of the API/worker seam, where they can drift into an API that
/// offers a convert affordance the worker refuses to honour, or the reverse.
///
/// Only the `requiresConversion` read stays local: it is a property of THIS catalog entry, not of
/// the converter registry, and it is load-bearing — without it a turnkey model that merely names a
/// converter would start reporting MLX status off-Mac.
///
/// Platform-aware, like the core helper: off macOS only `CANDLE_NATIVE_CONVERTERS` qualify, on macOS
/// every native converter does (where the call site's `cfg!(target_os = "macos")` arm has already
/// won anyway — macOS surfaces MLX status for every entry, convert-at-install or not).
fn entry_requires_converted_artifact_here(object: &JsonObject) -> bool {
    let Some(mlx) = object.get("mlx").and_then(Value::as_object) else {
        return false;
    };
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    mlx.get("converter")
        .and_then(Value::as_str)
        .map(str::trim)
        .is_some_and(sceneworks_core::jobs_store::convert_artifact_required_here)
}

fn apply_mac_and_mlx_fields(object: &mut JsonObject, data_dir: &FsPath) {
    // Per-model quality FLOOR (sc-10731, epic 10721): surface the manifest `mlx.minQualityTier` as a
    // top-level `minQualityTier` so the web `defaultTierSelection` can clamp the DEFAULT generation tier
    // UP to it (a floored model — Anima base/aesthetic = q8 — never lets a low global "default quality"
    // setting land the default on the washed q4). An EXPLICIT picker pick below the floor is still
    // honored, with a non-blocking advisory. Decoupled top-level field, mirroring `mlxTiers` — the web
    // reads one stable key rather than reaching into the passed-through `mlx` sub-object. Emitted on every
    // platform where the manifest declares it (the picker only renders where >1 tier installs, but the
    // field is cheap and lets any surface read the floor). Only bf16/q8/q4 are valid; others are dropped.
    if let Some(floor) = object
        .get("mlx")
        .and_then(|mlx| mlx.get("minQualityTier"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|tier| matches!(*tier, "bf16" | "q8" | "q4"))
        .map(str::to_owned)
    {
        object.insert("minQualityTier".to_owned(), Value::String(floor));
    }
    let mac_support = {
        let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
        let model_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // Preserve the catalog-declared family in the shared probe call, but do not authorize an
        // imported loader from family identity alone. This pass supplies builtin id-keyed defaults;
        // `apply_imported_provider_surface` runs later with the full manifest entry and replaces an
        // imported row's provisional verdict from its exact family + `importSourceShape` routes.
        let family = object
            .get("family")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        // A trained checkpoint of a builtin generation engine (the Iris full-model export,
        // sc-25685) renders through that engine's id-keyed lane, so it carries the engine's verdict.
        let id: &str = match crate::iris_training::trained_task_checkpoint(object) {
            Some(objective) => objective.base_model(),
            None => id,
        };
        model_mac_support(id, model_type, family)
    };
    if let Ok(mac_support) = serde_json::to_value(mac_support) {
        object.insert("macSupport".to_owned(), mac_support);
    }
    // The off-Mac twin (sc-19570). Emitted on EVERY platform, exactly like `macSupport`: the client
    // decides whether to act on it from `candleGatingActive`, and a block that appeared only on the
    // platform it gates could never be asserted from a Mac test run — which is precisely how the
    // off-Mac half of this defect stayed invisible for as long as it did. No `family` argument: the
    // block carries the per-video-mode verdict, and video routing is id-keyed (route-by-family is
    // an image-lane mechanism).
    let candle_support = {
        let id = object.get("id").and_then(Value::as_str).unwrap_or_default();
        let model_type = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id: &str = match crate::iris_training::trained_task_checkpoint(object) {
            Some(objective) => objective.base_model(),
            None => id,
        };
        model_candle_support(id, model_type)
    };
    if let Ok(candle_support) = serde_json::to_value(candle_support) {
        object.insert("candleSupport".to_owned(), candle_support);
    }
    // A LOCAL DISK probe for MLX convert-output tier dirs, so it is a platform fact and must NOT go
    // through `generation::enqueue_backend`: keying it off `candle_required` would hide `mlxTiers`
    // and its per-tier state from the Studio on macOS for directories that are genuinely present.
    //
    // Off macOS the probe is NARROWED, not skipped (sc-20529). A convert-at-install model whose
    // converter has a real off-Mac implementation — `CANDLE_NATIVE_CONVERTERS`, today
    // `flux2_klein_diffusers` / `flux2_klein_9b_true_v2` and the backend-neutral `supra2_img`
    // (sc-24214) — needs the same convert affordance and the
    // same converted/needs_conversion/needs_source states on Windows and Linux that macOS gets: the
    // candle converter (sc-7459) is real, `POST /models/{id}/convert` is not platform-gated, and
    // `inject_converted_model_path` already reads the converted dir on every platform. Without these
    // fields the Studio had no way to show that the conversion was still outstanding, so the model
    // read as installed off its raw single-file download and failed at load.
    //
    // It stays NARROW deliberately. Probing every `requiresConversion` model off-Mac would flip
    // Anima's three variants to `missing` there: `anima_quant` is macOS-only, and off-Mac Anima
    // loads the raw `circlestone-labs/Anima` `split_files/` tree with no converted dir at all. The
    // const is the contract that keeps those two cases apart.
    let mlx_status = if cfg!(target_os = "macos") || entry_requires_converted_artifact_here(object)
    {
        mlx_catalog_status(object, data_dir)
    } else {
        None
    };
    if let Some(status) = mlx_status {
        // Generation-time tier picker for convert-at-install models (sc-10730): surface the on-disk
        // convert-output tiers as `mlxTiers`, DECOUPLED from `hasVariantMatrix` so the Models download
        // panel is untouched. Only when the model is actually converted (its tier subdirs exist).
        if let Some(converted) = status.converted_path.as_deref() {
            let tiers = mlx_convert_output_tiers(converted);
            if !tiers.is_empty() {
                // Owned before the mutable inserts below so the `family` read doesn't overlap the borrow;
                // it selects the per-tier completeness predicate for the states (sc-13513).
                let family = object
                    .get("family")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let model_id = object
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                object.insert("mlxTiers".to_owned(), json!(tiers));
                // The FULL possible tier set with per-tier state, so the Studio shows un-converted tiers
                // disabled rather than omitting them (the web reads `mlxTierStates` in preference to the
                // installed-only `mlxTiers`). Emitted only when the model is actually converted (>=1 tier
                // present), matching `mlxTiers` — an unconverted model still renders no picker.
                object.insert(
                    "mlxTierStates".to_owned(),
                    Value::Array(mlx_convert_output_tier_states(
                        converted, &family, &model_id,
                    )),
                );
            }
        }
        object.insert(
            "mlxInstallState".to_owned(),
            Value::String(status.install_state.to_owned()),
        );
        object.insert(
            "mlxConversionState".to_owned(),
            Value::String(status.conversion_state.to_owned()),
        );
        // The engine's reason a marked conversion would not load (sc-24216); absent otherwise so
        // every other model's catalog shape is unchanged.
        match status.conversion_problem.as_deref() {
            Some(problem) => {
                object.insert(
                    "mlxConversionProblem".to_owned(),
                    Value::String(problem.to_owned()),
                );
            }
            None => {
                object.remove("mlxConversionProblem");
            }
        }
        object.insert(
            "mlxConvertedPath".to_owned(),
            status
                .converted_path
                .map(|path| Value::String(path.display().to_string()))
                .unwrap_or(Value::Null),
        );
        object.insert(
            "updateAvailable".to_owned(),
            Value::Bool(
                status.update_available
                    || object
                        .get("updateAvailable")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
            ),
        );
    }
}

async fn run_blocking_catalog_sweep<I, O, F>(
    items: Vec<I>,
    operation: F,
) -> Result<Vec<O>, ApiError>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I) -> Result<O, ApiError> + Send + Sync + 'static,
{
    let operation = Arc::new(operation);
    let mut items = items.into_iter();
    let mut output = Vec::new();
    loop {
        let batch = items
            .by_ref()
            .take(MODEL_CATALOG_PROBE_CONCURRENCY)
            .collect::<Vec<_>>();
        if batch.is_empty() {
            break;
        }
        let permits = MODEL_CATALOG_PROBE_PERMITS
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MODEL_CATALOG_PROBE_CONCURRENCY)))
            .clone();
        let mut tasks = Vec::with_capacity(batch.len());
        for item in batch {
            let permit = permits
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| ApiError::internal("model catalog probe limiter closed"))?;
            let operation = operation.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                let _permit = permit;
                operation(item)
            }));
        }
        let results = join_all(tasks).await;
        for result in results {
            output.push(result.map_err(|err| {
                ApiError::internal(format!("model catalog probe task failed: {err}"))
            })??);
        }
    }
    Ok(output)
}

fn apply_model_catalog_size_fields(
    model: &mut Value,
    download_context: Option<&DownloadContext>,
    download_size_bytes: Option<u64>,
) -> Result<(), ApiError> {
    let fallback_size_bytes = download_context.and_then(|context| context.fallback_size_bytes);
    let primary_size_bytes = download_size_bytes.or(fallback_size_bytes);
    let download_size_estimated = download_size_bytes.is_none() && fallback_size_bytes.is_some();
    // Co-requisites (sc-9696) install alongside the primary, so the displayed footprint must
    // include them (e.g. PiD's ~2.7 GB checkpoint + ~5.2 GB gemma-2-2b-it). Their sizes come
    // from the manifest (the live HF estimate above only sizes the primary repo).
    // Only the co-requisites the SELECTED tier actually pulls (sc-14980) — otherwise a q4 Mage-Flow
    // install would advertise the sum of all three shared text-encoder tiers.
    // The displayed footprint is for the DEFAULT download, so size only the co-requisites that
    // download actually pulls (sc-14980) — otherwise a q4 Mage-Flow install would advertise the sum
    // of all three shared text-encoder tiers (16.1 GB instead of 2.51 GB).
    let default_variant = model_download(model)
        .as_ref()
        .and_then(|download| download.get("variant").and_then(Value::as_str))
        .map(str::to_owned);
    let co_requisite_size_bytes: u64 =
        model_co_requisite_downloads_for_variant(model, default_variant.as_deref())
            .iter()
            .filter_map(|download| manifest_download_size_bytes(model, download))
            .sum();
    let effective_download_size_bytes = match primary_size_bytes {
        Some(primary) => Some(primary + co_requisite_size_bytes),
        None if co_requisite_size_bytes > 0 => Some(co_requisite_size_bytes),
        None => None,
    };
    {
        let object = model
            .as_object_mut()
            .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
        object.insert(
            "downloadSizeBytes".to_owned(),
            effective_download_size_bytes
                .map(|value| json!(value))
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeLabel".to_owned(),
            effective_download_size_bytes
                .map(format_bytes)
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        object.insert(
            "downloadSizeEstimated".to_owned(),
            Value::Bool(download_size_estimated),
        );
    }
    refresh_variant_download_sizes(model)
}

fn apply_model_catalog_entry(
    mut model: Value,
    download_context: Option<DownloadContext>,
    data_dir: &FsPath,
    user_model_ids: &std::collections::HashSet<String>,
    builtin_model_ids: &std::collections::HashSet<String>,
    local_artifacts: &[sceneworks_core::model_artifacts::ResolvedModelArtifact],
) -> Result<Value, ApiError> {
    #[cfg(test)]
    test_delay_catalog_probe(&model);
    let state = install_state_for(download_context, &model, data_dir);
    // sc-19708: the typed availability judgement for this entry's default selected closure on
    // this host, through the ONE shared resolver. Catalog nuances layered on top of it:
    // an install living in an app-owned path is local-ready regardless of the library, and an
    // installed-but-incomplete cache stays `incomplete` rather than `missing`.
    let availability_resolution =
        crate::model_sources::availability_for_entry(data_dir, &model, None, local_artifacts);
    // sc-19712 F-5: what a local copy of this entry could EVER cover, which is a different question
    // from where it resolves from today. A model with soft co-requisites or a revision-less
    // requirement showed the identical "local copy" affordance as a fully cacheable one, so the
    // badge promised protection from a disconnect that the cache cannot actually deliver.
    let cache_eligibility =
        crate::model_sources::cache_eligibility_for_entry(data_dir, &model, None);
    use sceneworks_core::model_artifacts::external_library::ModelAvailability;
    let managed_models = data_dir.join("models");
    let installed_in_app_owned_path = state.installed
        && state
            .installed_path
            .as_deref()
            .map(FsPath::new)
            .is_some_and(|path| path.starts_with(&managed_models));
    let (availability, availability_resolution) = if state.cache_incomplete
        && model.get("family").and_then(Value::as_str) == Some("minimax-h3")
        && model
            .as_object()
            .and_then(sceneworks_core::jobs_store::checkpoint_plan_checkpoint_id)
            .is_some()
    {
        (ModelAvailability::Incomplete, Some(availability_resolution))
    } else if installed_in_app_owned_path
        && !matches!(
            availability_resolution.availability,
            ModelAvailability::LocalReady
        )
    {
        (ModelAvailability::LocalReady, None)
    } else if availability_resolution.availability == ModelAvailability::Missing
        && state.cache_incomplete
    {
        (ModelAvailability::Incomplete, Some(availability_resolution))
    } else {
        (
            availability_resolution.availability.clone(),
            Some(availability_resolution),
        )
    };
    let platform_cleanup_only =
        model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true);
    let cleanup_residue_only = platform_cleanup_only
        && !state.installed
        && cleanup_with_model_artifact_paths(&model, data_dir)
            .iter()
            .any(|path| std::fs::symlink_metadata(path).is_ok());
    // sc-22988: which options of each co-requisite choice group are installed here (YuE2's
    // `decoder`: `standard` / `legacy`), so a surface that offers the choice can disable an option
    // whose component is not installed. Present only on a model that declares a choice group.
    let installed_choices = installed_co_requisite_choices(&model, data_dir);
    let object = model
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    if let Some(choices) = installed_choices {
        object.insert("installedChoices".to_owned(), choices);
    }
    object.insert(
        "modelAvailability".to_owned(),
        serde_json::to_value(&availability).map_err(|error| {
            ApiError::internal(format!("serialize model availability: {error}"))
        })?,
    );
    object.insert(
        "modelResolution".to_owned(),
        availability_resolution
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| ApiError::internal(format!("serialize model resolution: {error}")))?
            .unwrap_or(Value::Null),
    );
    object.insert(
        "cacheEligibility".to_owned(),
        cache_eligibility
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| ApiError::internal(format!("serialize cache eligibility: {error}")))?
            .unwrap_or(Value::Null),
    );
    let model_id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    // Ownership and routing authority are deliberately separate. A user overlay on a builtin is
    // removable (deleting it restores builtin defaults), but it cannot reclassify that protected id
    // as an imported checkpoint and bypass/trigger source-shaped provider validation.
    let user_managed = user_model_ids.contains(model_id);
    let builtin_routing_authority = builtin_model_ids.contains(model_id);
    object.insert(
        "catalogScope".to_owned(),
        Value::String(
            if builtin_routing_authority {
                "builtin"
            } else {
                "user"
            }
            .to_owned(),
        ),
    );
    stamp_legacy_import_source_shape(object, data_dir, !builtin_routing_authority);
    object.insert(
        "downloadable".to_owned(),
        Value::Bool(state.downloadable && !platform_cleanup_only),
    );
    object.insert(
        "installState".to_owned(),
        Value::String(
            if state.installed {
                "installed"
            } else {
                "missing"
            }
            .to_owned(),
        ),
    );
    object.insert(
        "cacheState".to_owned(),
        Value::String(
            if state.cache_incomplete || cleanup_residue_only {
                "incomplete"
            } else if state.installed {
                "complete"
            } else {
                "missing"
            }
            .to_owned(),
        ),
    );
    object.insert(
        "missingRequiredFiles".to_owned(),
        Value::Array(
            state
                .missing_required_files
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    object.insert(
        "repairAvailable".to_owned(),
        Value::Bool(state.downloadable && state.cache_incomplete && !platform_cleanup_only),
    );
    object.insert(
        "updateAvailable".to_owned(),
        Value::Bool(state.update_available && !platform_cleanup_only),
    );
    object.insert(
        "installedPath".to_owned(),
        state
            .installed_path
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    object.insert(
        "removable".to_owned(),
        Value::Bool(user_managed || state.installed || platform_cleanup_only),
    );
    if platform_cleanup_only {
        object.insert("usable".to_owned(), Value::Bool(false));
        object.insert("recommended".to_owned(), Value::Bool(false));
    }
    // Per-variant install tracking (sc-8508, epic 8506): one entry per declared quant tier,
    // each with its own installed flag + path + size + footprint. A single-variant model
    // still emits exactly one "default" variant, so the array is a superset of the
    // (retained) top-level installState/installedPath fields — nothing existing regresses.
    apply_variant_fields(object, data_dir);
    apply_gating_fields(object);
    apply_mac_and_mlx_fields(object, data_dir);
    apply_imported_provider_surface(object);
    apply_imported_lora_advertisement(object);
    mark_empty_lora_advertisement_unsupported(object);
    // Live denoise preview support (sc-16965, epic 16948): `preview.byBackend`, read from the
    // generated `config/manifests/builtin.preview-support.jsonc` rather than from a registry, because
    // THIS process may link no engines at all (docker/rust.Dockerfile builds the API without
    // `backend-candle`, so `Registry::new()` here is empty and a serve-time derivation would report
    // "nothing supports preview" on every server). Engine-KEYED on purpose — the flag genuinely
    // diverges by backend and never fully collapses. Additive: a model the generated table does not
    // know gets no `preview` key, which the UI reads as "unknown" and renders exactly as before.
    apply_last_only_video_capability(object);
    sceneworks_core::preview_support::apply_to_model_entry(object);
    sceneworks_core::decoder_support::apply_to_model_entry(object);
    apply_decoder_availability(object, data_dir);
    apply_conditional_component_state(object, data_dir);
    apply_imported_checkpoint_product_surface(object, data_dir, state.imported_source_present);
    if platform_cleanup_only {
        // The tombstone exists only to expose whole-model Delete. Strip every tier/conversion
        // action projection even though the preserved manifest metadata is still needed by the
        // deletion resolver to locate old files.
        object.insert("hasVariantMatrix".to_owned(), Value::Bool(false));
        object.remove("variants");
        object.remove("mlxConversionState");
        object.remove("mlxInstallState");
        object.remove("mlxTiers");
        object.remove("mlxTierStates");
    }
    Ok(model)
}

// Use exactly the entry validation uses, including resolved imported partitions.
fn apply_last_only_video_capability(object: &mut JsonObject) {
    let imported_partition = object.get("id").and_then(Value::as_str).and_then(|_| {
        use sceneworks_core::minimax_h3_import::{
            entry_is_imported_minimax_h3, MiniMaxH3Partition,
        };
        entry_is_imported_minimax_h3(object)
            .then(|| MiniMaxH3Partition::from_entry(object).ok())
            .flatten()
    });
    let accepts_last_only =
        sceneworks_core::h3_keyframes::accepts_last_only_resolved_image_to_video(
            object.get("id").and_then(Value::as_str).unwrap_or_default(),
            "image_to_video",
            true,
            imported_partition.map(|partition| partition.as_str()),
        );
    object.insert(
        "supportsLastOnlyImageToVideo".to_owned(),
        Value::Bool(accepts_last_only),
    );
}

#[cfg(test)]
#[path = "models/tests/last_only_projection_tests.rs"]
mod last_only_projection_tests;

/// Give a provider-aliased model row the canonical provider's exact optional decoder donors.
///
/// Imported Krea entries intentionally carry `downloads: []`: their primary DiT is an in-place user
/// file, not a downloadable model. Decoder eligibility is nevertheless provider-derived, and the
/// selected decoder must resolve the same pinned standalone VAE as the builtin Turbo provider. Clone
/// only soft co-requisites whose component ids occur in the provider's generated decoder facts;
/// primary downloads and unrelated components (including text encoders) are never inherited.
fn inherit_provider_decoder_downloads(model: &mut Value, builtin_models: &[Value]) {
    let Some(object) = model.as_object() else {
        return;
    };
    let Some(model_id) = object.get("id").and_then(Value::as_str) else {
        return;
    };
    let Some(provider_id) =
        sceneworks_core::decoder_support::provider_id_for_backend(object, "mlx")
    else {
        return;
    };
    if provider_id == model_id {
        return;
    }
    let component_ids = sceneworks_core::decoder_support::component_ids_for_backend(object, "mlx");
    if component_ids.is_empty() {
        return;
    }
    let Some(provider) = builtin_models.iter().find(|candidate| {
        candidate.get("id").and_then(Value::as_str) == Some(provider_id.as_str())
    }) else {
        return;
    };
    let inherited = provider
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| {
            download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                && download.get("required").and_then(Value::as_str) == Some("soft")
                && download
                    .get("componentId")
                    .and_then(Value::as_str)
                    .is_some_and(|component_id| {
                        component_ids
                            .iter()
                            .any(|expected| expected == component_id)
                    })
                && download
                    .get("revision")
                    .and_then(Value::as_str)
                    .is_some_and(|revision| {
                        revision.len() == 40
                            && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
                && download
                    .get("repo")
                    .and_then(Value::as_str)
                    .is_some_and(|repo| !repo.trim().is_empty())
                && download
                    .get("files")
                    .and_then(Value::as_array)
                    .is_some_and(|files| !files.is_empty())
        })
        .cloned()
        .collect::<Vec<_>>();
    if inherited.is_empty() {
        return;
    }

    let object = model.as_object_mut().expect("checked object above");
    let existing = object
        .remove("downloads")
        .and_then(|downloads| downloads.as_array().cloned())
        .unwrap_or_default();
    // Canonical rows lead so both the API availability probe and the worker's exact component
    // resolver bind the generated decoder id to the provider's source even if a user manifest
    // happens to contain another soft component with the generic `vae` id.
    let mut downloads = inherited.clone();
    downloads.extend(
        existing
            .into_iter()
            .filter(|download| !inherited.iter().any(|canonical| canonical == download)),
    );
    object.insert("downloads".to_owned(), Value::Array(downloads));
}

#[derive(Clone, Copy, Debug, Default)]
struct OptionalComponentAvailability {
    installed: bool,
    estimated_size_bytes: Option<u64>,
}

/// Enrich descriptor-derived decoder options with the local state of their pinned, soft
/// co-requisite. Eligibility comes exclusively from engine facts; this function answers only whether
/// the declared standalone component is present. A full Wan model install is neither consulted nor
/// implied.
fn apply_decoder_availability(object: &mut JsonObject, data_dir: &FsPath) {
    let mut components: std::collections::BTreeMap<String, OptionalComponentAvailability> =
        std::collections::BTreeMap::new();
    for download in object
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if download.get("coRequisite").and_then(Value::as_bool) != Some(true) {
            continue;
        }
        let Some(component_id) = download.get("componentId").and_then(Value::as_str) else {
            continue;
        };
        let installed = pinned_component_download_installed(download, data_dir);
        let size = download
            .get("estimatedSizeBytes")
            .and_then(Value::as_u64)
            .or_else(|| {
                download
                    .get("footprint")
                    .and_then(|footprint| footprint.get("diskSizeBytes"))
                    .and_then(Value::as_u64)
            });
        // The provider-inheritance seam prepends its canonical exact row. First declaration wins so
        // a later user-authored soft `vae` row cannot make the Wan option appear installed or divert
        // the worker to a different file.
        components
            .entry(component_id.to_owned())
            .or_insert(OptionalComponentAvailability {
                installed,
                estimated_size_bytes: size,
            });
    }

    let Some(by_backend) = object
        .get_mut(sceneworks_core::decoder_support::DECODERS_FIELD)
        .and_then(Value::as_object_mut)
        .and_then(|decoders| {
            decoders
                .get_mut(sceneworks_core::decoder_support::BY_BACKEND_FIELD)
                .and_then(Value::as_object_mut)
        })
    else {
        return;
    };
    for options in by_backend.values_mut().filter_map(Value::as_array_mut) {
        for option in options.iter_mut().filter_map(Value::as_object_mut) {
            let state = option
                .get("componentId")
                .and_then(Value::as_str)
                .and_then(|id| components.get(id))
                .copied()
                .unwrap_or_default();
            option.insert("available".to_owned(), Value::Bool(state.installed));
            if let Some(bytes) = state.estimated_size_bytes {
                option.insert("estimatedSizeBytes".to_owned(), Value::from(bytes));
            }
        }
    }
}

fn pinned_component_download_installed(download: &Value, data_dir: &FsPath) -> bool {
    let Some(repo) = download.get("repo").and_then(Value::as_str) else {
        return false;
    };
    let Some(revision) = download.get("revision").and_then(Value::as_str) else {
        return false;
    };
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return false;
    }
    let files = string_array_field(download, "files");
    if files.is_empty() {
        return false;
    }
    // Through the typed source-library resolver, not a hand-built `snapshots/<rev>` root — the
    // model-path inventory audit (sc-19703) forbids direct root construction on this file.
    let resolver = sceneworks_core::model_artifacts::ModelArtifactResolver::new(
        sceneworks_core::hf_home::model_source_library(data_dir),
    );
    let Ok((_, snapshot)) = resolver.discover_source_snapshot(repo, Some(revision)) else {
        return false;
    };
    files
        .iter()
        .all(|pattern| snapshot_contains_pattern(&snapshot, pattern))
}

#[cfg(test)]
#[path = "models/tests/decoder_availability_tests.rs"]
mod decoder_availability_tests;

/// Backfill the explicit source-shape discriminator for user entries created before SC-18312.
/// This is structural inspection under the app-managed model root, never a family guess: a Mage
/// transformer directory must carry both of its required files; a single-file import is classified
/// by the same header-only detector the importer uses. The worker repeats confinement and shape
/// validation before opening weights.
fn stamp_legacy_import_source_shape(
    object: &mut JsonObject,
    data_dir: &FsPath,
    user_managed: bool,
) {
    if !user_managed || object.contains_key("importSourceShape") {
        return;
    }
    let Some(raw) = object
        .get("modelPath")
        .and_then(Value::as_str)
        .or_else(|| {
            object
                .get("paths")
                .and_then(Value::as_object)
                .and_then(|paths| paths.get("model"))
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|path| !path.is_empty())
    else {
        return;
    };
    let Ok(path) = std::fs::canonicalize(raw) else {
        return;
    };
    let Ok(model_root) = std::fs::canonicalize(data_dir.join("models")) else {
        return;
    };
    if !path.starts_with(model_root) {
        return;
    }
    if object.get("family").and_then(Value::as_str) == Some("mage-flow")
        && path.is_dir()
        && sceneworks_core::base_weights::is_mage_flow_transformer_dir(&path)
    {
        object.insert(
            "importSourceShape".to_owned(),
            Value::String("transformer_directory".to_owned()),
        );
        return;
    }
    let Some(weight_file) = imported_model_primary_weight_file(&path) else {
        return;
    };
    let Ok(BaseWeightDetection::Recognized(verdict)) = detect_base_weight_file(&weight_file) else {
        return;
    };
    let family = object.get("family").and_then(Value::as_str);
    let source = match (family, verdict.family.as_deref(), verdict.component) {
        (Some("krea_2"), Some("krea_2"), ComponentRole::Transformer) => "transformer_file",
        (Some("sdxl"), Some("sdxl"), ComponentRole::Checkpoint) => "fused_checkpoint",
        _ => return,
    };
    object.insert(
        "importSourceShape".to_owned(),
        Value::String(source.to_owned()),
    );
}

type ModelCatalogWorkItem = (Value, Option<DownloadContext>);
const MODEL_SIZE_ESTIMATE_CONCURRENCY: usize = 8;

async fn collect_bounded_ordered<I, T, F, Fut, R>(
    items: I,
    concurrency: usize,
    operation: F,
) -> Vec<R>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T) -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let pending = futures_util::StreamExt::map(stream::iter(items), operation);
    let bounded = futures_util::StreamExt::buffered(pending, concurrency.max(1));
    futures_util::StreamExt::collect(bounded).await
}

#[cfg(test)]
#[path = "models/tests/model_size_concurrency_tests.rs"]
mod model_size_concurrency_tests;

fn group_model_catalog_work_items(
    work_items: Vec<ModelCatalogWorkItem>,
) -> Vec<Vec<ModelCatalogWorkItem>> {
    let mut groups = Vec::<Vec<ModelCatalogWorkItem>>::new();
    let mut repo_groups = HashMap::<String, usize>::new();
    for item in work_items {
        let (_, download_context) = &item;
        let repo = download_context
            .as_ref()
            .map(|context| context.repo.clone());
        if let Some(repo) = repo {
            if let Some(index) = repo_groups.get(&repo).copied() {
                groups[index].push(item);
            } else {
                repo_groups.insert(repo, groups.len());
                groups.push(vec![item]);
            }
        } else {
            groups.push(vec![item]);
        }
    }
    groups
}

async fn estimate_model_catalog_sizes(
    state: &AppState,
    download_contexts: &[Option<DownloadContext>],
    enabled: bool,
) -> HashMap<ModelSizeCacheKey, Option<u64>> {
    if !enabled {
        return HashMap::new();
    }
    let mut seen = std::collections::HashSet::new();
    let unique = download_contexts
        .iter()
        .flatten()
        .filter_map(|context| {
            let key = (context.repo.clone(), context.files.clone());
            seen.insert(key.clone()).then_some((key, context.clone()))
        })
        .collect::<Vec<_>>();
    collect_bounded_ordered(
        unique,
        MODEL_SIZE_ESTIMATE_CONCURRENCY,
        |(key, context)| async move {
            let estimate =
                estimate_huggingface_download_size(state, &context.repo, &context.files).await;
            (key, estimate)
        },
    )
    .await
    .into_iter()
    .collect()
}

/// Built-in + user model manifest entries merged by id, with NO platform filtering — the raw
/// authored catalog. [`load_model_catalog_inputs`] narrows `downloads` to the running OS on top of
/// this; [`license_acknowledgment_repo_index`] deliberately reads it unfiltered, because a licence
/// requirement must not depend on which OS is asking. Both manifest reads are mtime/size-cached
/// (`load_manifest_entries`), so the second consumer costs a stat and a clone.
async fn merged_model_manifest_entries(
    state: &AppState,
) -> Result<(Vec<Value>, std::collections::HashSet<String>), ApiError> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    Ok((merge_entries_by_id(builtin, user), user_model_ids))
}

async fn load_model_catalog_inputs(
    state: &AppState,
) -> Result<
    (
        Vec<Value>,
        Vec<Option<DownloadContext>>,
        std::collections::HashSet<String>,
        std::collections::HashSet<String>,
    ),
    ApiError,
> {
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let user_model_ids = user
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    let decoder_providers = builtin.clone();
    let builtin_model_ids = builtin
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    let mut models = merge_entries_by_id(builtin, user);
    // A trained checkpoint (Iris full export) takes its generation surface from its engine's entry
    // on every read, never from a registration-time snapshot.
    for model in &mut models {
        crate::iris_training::overlay_trained_checkpoint_surface(model, &decoder_providers)?;
    }
    retain_models_for_os(&mut models, std::env::consts::OS, &state.settings.data_dir)?;
    // Resolve per-platform download sources before computing install state/size: some video models
    // carry both a native MLX-convert checkpoint (macOS) and a diffusers/torch checkpoint
    // (Windows/Linux). Keep only the entries applicable to this OS so the download job, status,
    // size, and the frontend all agree on the right repo (sc-3240).
    for model in &mut models {
        inherit_provider_decoder_downloads(model, &decoder_providers);
        if model.get("platformCleanupOnly").and_then(Value::as_bool) != Some(true) {
            retain_downloads_for_os(model, std::env::consts::OS);
        }
    }
    let download_contexts = models
        .iter()
        .map(model_download_context)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((models, download_contexts, user_model_ids, builtin_model_ids))
}

async fn estimate_current_model_catalog_sizes(
    state: &AppState,
) -> Result<HashMap<ModelSizeCacheKey, Option<u64>>, ApiError> {
    let (_, download_contexts, _, _) = load_model_catalog_inputs(state).await?;
    Ok(estimate_model_catalog_sizes(state, &download_contexts, true).await)
}

async fn build_model_catalog_snapshot(state: &AppState) -> Result<Vec<Value>, ApiError> {
    // sc-8819 (F-017): observe full-catalog assembly (the per-model FS install-state probe
    // sweep) so tests can assert request-scoped and process-shared reuse.
    #[cfg(test)]
    crate::test_note_model_catalog_build();
    let (models, download_contexts, user_model_ids, builtin_model_ids) =
        load_model_catalog_inputs(state).await?;
    let builtin_model_ids = Arc::new(builtin_model_ids);

    let data_dir = Arc::new(state.settings.data_dir.clone());
    let user_model_ids = Arc::new(user_model_ids);
    // App-owned resolved-local artifacts, enumerated ONCE per snapshot build (read-only: no
    // session, no usage refresh) so every entry's availability can detect local-ready coverage.
    let local_artifacts = Arc::new(crate::model_sources::local_resolved_artifacts(state));
    let work_items = models
        .into_iter()
        .zip(download_contexts.clone())
        .collect::<Vec<_>>();
    let work_groups = group_model_catalog_work_items(work_items);
    let data_dir_for_sweep = data_dir.clone();
    let user_model_ids_for_sweep = user_model_ids.clone();
    let builtin_model_ids_for_sweep = builtin_model_ids.clone();
    let local_artifacts_for_sweep = local_artifacts.clone();

    // sc-14530: each model's install-state resolution is independent but may spend seconds
    // waiting on network-volume metadata. Dispatch bounded blocking tasks per primary-repo group
    // so that latency overlaps instead of accumulating serially across the full catalog, while
    // shared-repo receipt handling stays deterministic.
    let catalog_sweep = run_blocking_catalog_sweep(work_groups, move |work_group| {
        work_group
            .into_iter()
            .map(|(model, download_context)| {
                apply_model_catalog_entry(
                    model,
                    download_context,
                    &data_dir_for_sweep,
                    &user_model_ids_for_sweep,
                    &builtin_model_ids_for_sweep,
                    &local_artifacts_for_sweep,
                )
            })
            .collect::<Result<Vec<_>, _>>()
    });

    // External ComfyUI discovery is independent from manifest install-state resolution; overlap
    // that filesystem scan with the per-entry sweep too.
    let external_roots = state.settings.external_model_roots.clone();
    let external_base_cache = state.external_base_model_cache.clone();
    let external_scan = tokio::task::spawn_blocking(move || {
        let mut cache = external_base_cache.lock();
        crate::external_base_models::scan_external_base_models(&external_roots, &mut cache)
    });
    let (models, external) = tokio::join!(catalog_sweep, external_scan);
    let mut models = models?.into_iter().flatten().collect::<Vec<_>>();
    for model in &mut models {
        let context = model_download_context(model)?;
        // The shared snapshot carries deterministic manifest fallbacks only.
        // `/models` overlays live estimates on its response clone; presets and
        // job validation never wait on unrelated Hugging Face network calls.
        apply_model_catalog_size_fields(model, context.as_ref(), None)?;
    }
    let mut external = external.map_err(|err| {
        ApiError::internal(format!("external model catalog scan task failed: {err}"))
    })?;
    for model in &mut external {
        if let Some(object) = model.as_object_mut() {
            apply_imported_provider_surface(object);
            apply_imported_lora_advertisement(object);
        }
    }
    models.extend(external);
    models.sort_by(|left, right| {
        let left_key = (
            left.get("type").and_then(Value::as_str).unwrap_or_default(),
            left.get("name").and_then(Value::as_str).unwrap_or_default(),
        );
        let right_key = (
            right
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            right
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        left_key.cmp(&right_key)
    });

    Ok(models)
}

/// Resolve the merged model manifest entry for `model_id` so the GPU worker no
/// longer re-parses `builtin.models.jsonc`/`user.models.jsonc` itself — Rust is
/// the single owner of manifest parsing/merging (story 1653). The merged entry
/// is injected into video job payloads as `modelManifestEntry`. Returns `{}`
/// when the model is absent from both manifests, which the worker treats the
/// same as before (fall back to the model's default repo).
pub(crate) async fn resolve_model_manifest_entry(
    state: &AppState,
    model_id: &str,
) -> Result<Value, ApiError> {
    // External ComfyUI base models (epic 10451 Phase 2, sc-10667/10668) are synthesized in the
    // catalog, not declared in a jsonc manifest, so the jsonc lookup below would return `{}` and
    // the worker would never receive their `components[]` (the DiT/TE/VAE paths). Forward the
    // assembled row for an `external_base_*` id instead, so the worker can load them in place.
    // Blocking FS scan → run on the blocking pool, mirroring `model_catalog`.
    if model_id.starts_with(crate::external_base_models::EXTERNAL_ID_PREFIX) {
        let roots = state.settings.external_model_roots.clone();
        let cache = state.external_base_model_cache.clone();
        let id = model_id.to_owned();
        let row = tokio::task::spawn_blocking(move || {
            let mut cache = cache.lock();
            crate::external_base_models::scan_external_base_models(&roots, &mut cache)
                .into_iter()
                .find(|row| row.get("id").and_then(Value::as_str) == Some(id.as_str()))
        })
        .await
        .map_err(|err| ApiError::internal(format!("external base scan task failed: {err}")))?;
        // Absent (root unconfigured, file vanished) → `{}`, the same fall-back the worker already
        // handles for an unknown model id.
        return Ok(row.unwrap_or_else(|| json!({})));
    }
    let manifest_dir = state.settings.config_dir.join("manifests");
    let builtin =
        load_manifest_entries(state, &manifest_dir.join("builtin.models.jsonc"), "models").await?;
    let user =
        load_manifest_entries(state, &manifest_dir.join("user.models.jsonc"), "models").await?;
    let find = |entries: &[Value]| -> Option<Value> {
        entries
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
    };
    let builtin_entry = find(&builtin);
    let user_entry = find(&user);
    let mut catalog_scope = if builtin_entry.is_some() {
        Some("builtin")
    } else if user_entry.is_some() {
        Some("user")
    } else {
        None
    };
    let mut entry = merge_model_manifest_entry(builtin_entry, user_entry);
    if catalog_scope == Some("user") {
        crate::iris_training::overlay_trained_checkpoint_surface(&mut entry, &builtin)?;
    }
    if entry.as_object().is_none_or(JsonObject::is_empty) {
        // The model-source seam (sc-19708) resolves fixed utility models (upscalers, detectors,
        // the face stack) for job admission even when the configured manifest directory is
        // missing or trimmed — the compiled-in builtin manifest is the complete fallback so a
        // stripped config dir cannot silently strip typed model-source coverage.
        if let Some(embedded) = embedded_builtin_catalog_entry(|candidate| {
            candidate.get("id").and_then(Value::as_str) == Some(model_id)
        })? {
            entry = embedded;
            catalog_scope = Some("builtin");
        }
    }
    inherit_provider_decoder_downloads(&mut entry, &builtin);
    if let (Some(scope), Some(object)) = (catalog_scope, entry.as_object_mut()) {
        // The merged worker-facing entry must carry the same routing authority as `/models`.
        // A user manifest may override builtin presentation/defaults without turning the protected
        // builtin identity into an imported checkpoint. Genuine user-only ids remain user-scoped.
        object.insert("catalogScope".to_owned(), Value::String(scope.to_owned()));
        stamp_legacy_import_source_shape(object, &state.settings.data_dir, scope == "user");
    }
    inject_converted_model_path(&mut entry, &state.settings.data_dir);
    if let Some(object) = entry.as_object_mut() {
        // The worker receives the same descriptor-derived selection table and local pinned-file
        // availability as the catalog UI. This keeps enqueue validation and execution on one fact set.
        sceneworks_core::decoder_support::apply_to_model_entry(object);
        apply_decoder_availability(object, &state.settings.data_dir);
    }
    Ok(entry)
}

/// The compiled-in builtin model manifest, used as the last-resort catalog authority when the
/// on-disk manifest directory does not carry an entry. Kept complete and cross-platform: the
/// caller derives per-platform closures from it, so filtering here would let a macOS API omit
/// Candle-only co-requisites from a Windows/Linux worker carrier.
pub(crate) fn embedded_builtin_catalog_entry(
    predicate: impl Fn(&Value) -> bool,
) -> Result<Option<Value>, ApiError> {
    let manifest = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .map_err(|error| ApiError::internal(format!("embedded model manifest invalid: {error}")))?;
    let entry = manifest
        .get("models")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|entry| predicate(entry))
        .cloned();
    Ok(entry)
}

/// Populate the `modelPath` seam for convert-at-install MLX models. The worker's
/// `resolve_weights_dir` loads such a model from the locally-assembled converted
/// dir via `modelManifestEntry.modelPath`, but nothing else writes that key — the
/// raw source repo is a single safetensors file with no diffusers layout, so
/// without this the worker falls back to it and fails with "No such file or
/// directory" (e.g. flux2_klein_9b_true_v2). `mlx_catalog_status` is the single
/// source of truth for whether the conversion has produced a usable local dir.
/// No-op when the model needs no conversion, is not yet converted, or the manifest
/// already pins an explicit `modelPath`.
pub(crate) fn inject_converted_model_path(entry: &mut Value, data_dir: &FsPath) {
    let Some(object) = entry.as_object_mut() else {
        return;
    };
    let already_set = object
        .get("modelPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    if already_set {
        return;
    }
    if let Some(converted) =
        mlx_catalog_status(object, data_dir).and_then(|status| status.converted_path)
    {
        object.insert(
            "modelPath".to_owned(),
            Value::String(converted.display().to_string()),
        );
    }
}

/// One-level-deep merge of the builtin and user manifest entries for a single
/// model id. Mirrors the worker's former `ltx_model_manifest_entry` exactly so
/// this migration is behavior-preserving: user top-level keys override builtin
/// (shallow), and the nested config blocks the adapters read are merged
/// key-by-key rather than replaced wholesale. (This is intentionally deeper than
/// `merge_entries_by_id`, which the model catalog uses for display.)
pub(crate) fn merge_model_manifest_entry(builtin: Option<Value>, user: Option<Value>) -> Value {
    const NESTED_KEYS: [&str; 6] = [
        "paths",
        "resources",
        "defaults",
        "limits",
        "loraCompatibility",
        "ui",
    ];
    match (builtin, user) {
        (builtin, None) => builtin.unwrap_or_else(|| Value::Object(JsonObject::new())),
        (None, Some(user)) => user,
        (Some(builtin), Some(user)) => {
            let mut merged = builtin.clone();
            merge_object(&mut merged, user.clone());
            for key in NESTED_KEYS {
                let builtin_nested = builtin.get(key).and_then(Value::as_object);
                let user_nested = user.get(key).and_then(Value::as_object);
                if builtin_nested.is_none() && user_nested.is_none() {
                    continue;
                }
                let mut nested = builtin_nested.cloned().unwrap_or_default();
                if let Some(user_nested) = user_nested {
                    for (nested_key, value) in user_nested {
                        nested.insert(nested_key.clone(), value.clone());
                    }
                }
                if let Some(object) = merged.as_object_mut() {
                    object.insert(key.to_owned(), Value::Object(nested));
                }
            }
            merged
        }
    }
}

/// Restrict a model's `downloads` to the entries applicable to `os` (`std::env::consts::OS`).
/// A download entry with a `platforms` array applies only to the listed OSes; an entry without one
/// is platform-agnostic and always kept. Some video models ship two source repos for the same model
/// — the native MLX-convert checkpoint on macOS vs the diffusers/torch checkpoint on Windows/Linux
/// (sc-3240, Wan2.2) — so filtering here makes the download job, install status, size, and the
/// frontend's `downloads[0]` all resolve to the right per-platform repo from one seam. No-op unless
/// at least one entry is platform-tagged, so single-repo models are untouched.
///
/// sc-19708: the manifest-shape predicates below now live in
/// `sceneworks_core::model_artifacts::artifact_selection` because the availability seam (API
/// preflight AND the worker's pre-loader guard) selects the same closures. These re-exports keep
/// the API's install/display call sites on the identical definitions so they cannot drift.
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os;

pub(crate) use sceneworks_core::model_artifacts::artifact_selection::model_download;

/// Remove whole video products that have no route on `os`. This is intentionally narrower than the
/// legacy image-model `macOnly` label: only video entries use it as a catalog-withdrawal contract.
fn retain_models_for_os(
    models: &mut Vec<Value>,
    os: &str,
    data_dir: &FsPath,
) -> Result<(), ApiError> {
    if os == "macos" {
        return Ok(());
    }
    let mut retained = Vec::with_capacity(models.len());
    for mut model in std::mem::take(models) {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let withdrawn_video = video_model_withdrawn_on_platform(model_id, &model, os);
        if !withdrawn_video {
            retained.push(model);
            continue;
        }

        // Previously shipped off-Mac routes may already occupy substantial disk. Keep a catalog
        // tombstone only when the old artifact is installed or partial so Model Manager and DELETE
        // can reclaim it; a fresh machine sees no row at all. Preserve its original downloads on
        // the tombstone solely to resolve install state and owned cleanup paths.
        let context = model_download_context(&model)?;
        let state = install_state_for(context, &model, data_dir);
        let cleanup_residue = cleanup_with_model_artifact_paths(&model, data_dir)
            .iter()
            .any(|path| std::fs::symlink_metadata(path).is_ok());
        if state.installed || state.cache_incomplete || cleanup_residue {
            let object = model
                .as_object_mut()
                .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
            object.insert("platformCleanupOnly".to_owned(), Value::Bool(true));
            retained.push(model);
        }
    }
    *models = retained;
    Ok(())
}

/// Product withdrawals are authoritative by stable model id, not by a user-overridable manifest
/// presentation flag. `macOnly` remains the generic platform annotation, while this policy keeps a
/// same-id user manifest from restoring a product whose off-Mac runtime was explicitly removed.
pub(crate) fn video_model_withdrawn_on_platform(
    model_id: &str,
    model_manifest_entry: &Value,
    platform: &str,
) -> bool {
    platform != "macos"
        && (matches!(model_id, "ltx_2_3_eros")
            || (model_manifest_entry.get("type").and_then(Value::as_str) == Some("video")
                && model_manifest_entry.get("macOnly").and_then(Value::as_bool) == Some(true)))
}

/// True when a download entry is a co-requisite dependency (sc-9696): fetched ALONGSIDE the primary
/// download rather than as a pick-one alternate, and gating the entry's install state. See the
/// manifest schema `downloads[].coRequisite`.
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::is_co_requisite_download;

/// The co-requisite download entries for `model` (sc-9696) — the dependencies that must install
/// alongside the primary (e.g. the PiD decoder's shared gemma-2-2b-it caption encoder). The catalog
/// has already restricted `downloads` to the current OS (`retain_downloads_for_os`), so every entry
/// returned applies to this platform. Only provider-supported entries are returned.
/// Whether a co-requisite row is scoped to ONE quant tier (sc-14980).
///
/// Every co-requisite before sc-14980 is tier-agnostic — the PiD decoder's gemma caption encoder,
/// chatterbox's `ve`/`perth`, MMAudio's five components — and carries no `variant`, so it applies to
/// whatever tier is selected. Mage-Flow's shared Qwen3-VL text encoder and Mage-VAE are the first
/// that are themselves per-tier: they exist as `q4`/`q8`/`bf16` subtrees of one components mirror,
/// and only the one matching the selected tier should be fetched, sized, or gated on. Keying that on
/// the presence of `variant` keeps every existing co-requisite on exactly its current path.
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::{
    co_requisite_rows_gating_install, co_requisite_variant, is_pending_artifact_download,
    model_co_requisite_downloads, model_co_requisite_downloads_for_variant,
    model_co_requisite_downloads_for_variant_all_options, model_download_for_variant,
};

/// Best-effort credential host for a gated model when the manifest entry doesn't
/// set `credentialHost` explicitly: an explicit per-download `credentialHost`,
/// else the well-known host for the provider (`huggingface` ⇒ `huggingface.co`),
/// else the host of a `sourceUrl`. Normalized (scheme/path stripped, lower-cased)
/// to match how credentials are keyed in the store.
fn derive_credential_host(model: &serde_json::Map<String, Value>) -> Option<String> {
    let downloads = model.get("downloads")?.as_array()?;
    for download in downloads {
        if let Some(host) = download
            .get("credentialHost")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
        if download.get("provider").and_then(Value::as_str) == Some("huggingface") {
            return Some("huggingface.co".to_owned());
        }
        if let Some(host) = download
            .get("sourceUrl")
            .and_then(Value::as_str)
            .map(normalize_host)
            .filter(|host| !host.is_empty())
        {
            return Some(host);
        }
    }
    None
}

pub(crate) use sceneworks_core::model_artifacts::artifact_selection::is_supported_model_download;

pub(crate) fn model_download_context(model: &Value) -> Result<Option<DownloadContext>, ApiError> {
    let Some(download) = model_download(model) else {
        return Ok(None);
    };
    Ok(Some(DownloadContext {
        repo: required_string_field(&download, "repo")?.to_owned(),
        files: string_array_field(&download, "files"),
        fallback_size_bytes: manifest_download_size_bytes(model, &download),
    }))
}

pub(crate) fn huggingface_cache_health(
    repo_root: &FsPath,
    files: &[String],
) -> HuggingFaceCacheHealth {
    if !huggingface_repo_cache_exists(repo_root) {
        return HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: Vec::new(),
        };
    }
    let snapshots = huggingface_snapshot_dirs(repo_root);
    if snapshots.is_empty() {
        // The repo cache dir exists but holds no snapshot revision at all — an empty skeleton
        // (bare refs/blobs, e.g. a download that resolved zero files against an unpublished tier, or
        // a cache whose weights were pruned). Nothing is partially there, so this is MISSING, not a
        // repairable "incomplete": reporting incomplete surfaced a confusing "Cached files are
        // incomplete: snapshots/<revision>" banner for a tier that simply was never downloaded
        // (sc-9909). incomplete:false keeps it a clean not-installed state.
        return HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: vec!["snapshots/<revision>".to_owned()],
        };
    }
    if !files.is_empty() {
        return huggingface_filtered_cache_health(&snapshots, files);
    }

    let mut best_missing = Vec::new();
    for snapshot in snapshots {
        if path_is_readable_file(&snapshot.join("model_index.json")) {
            let health = diffusers_snapshot_health(&snapshot);
            if health.installed {
                return health;
            }
            if best_missing.is_empty() || health.missing_files.len() < best_missing.len() {
                best_missing = health.missing_files;
            }
            continue;
        }
        // A whole-repo (empty `files`) fetch of a non-diffusers repo declares no per-file claim, so
        // "some payload landed" was the only signal. That accepts an interrupted sharded download as
        // installed. The repo's own `*.safetensors.index.json` files are the declared set for their
        // shards, so validate them here (sc-20526).
        //
        // Cost, stated honestly: ONE snapshot walk per candidate snapshot, whose result answers both
        // the payload question and the index hunt. This lane previously short-circuited on a
        // `config.json` stat and walked nothing, so a `config.json`-bearing snapshot now pays a walk
        // it did not pay before. That walk is not removable by making it lazy: FINDING the indexes is
        // the check, and the check must run before this returns `installed`. What each index then
        // costs is one small JSON read plus a stat per distinct shard — no hashing, no header parse,
        // no tensor read. The whole lane runs on the install/catalog sweep, never per job.
        let files_on_disk = snapshot_files(&snapshot);
        if path_is_readable_file(&snapshot.join("config.json"))
            || snapshot_has_payload_file_in(&files_on_disk)
        {
            let torn = torn_shard_indexes_in(&snapshot, &files_on_disk);
            if torn.is_empty() {
                return HuggingFaceCacheHealth::installed();
            }
            if best_missing.is_empty() || torn.len() < best_missing.len() {
                best_missing = torn;
            }
            continue;
        }
        if best_missing.is_empty() {
            best_missing.push("model_index.json".to_owned());
        }
    }
    HuggingFaceCacheHealth::missing(best_missing)
}

/// If every pattern in a tier's `files` filter is confined to ONE leading directory — the standard
/// quant-tier layout `["q8/*"]` → `q8` (also `["bf16/*"]`, `["q4/*"]`) — return that directory.
/// `None` for a flat single-variant filter (`["*.safetensors"]`, whose leading component is itself a
/// glob) or patterns that span multiple top-level dirs; those are not a tier subdir and keep the
/// coarse glob check.
fn tier_subdir_name(files: &[String]) -> Option<String> {
    let mut tier: Option<&str> = None;
    for pattern in files {
        let (head, rest) = pattern.split_once('/')?;
        if head.is_empty() || rest.is_empty() || pattern_contains_glob(head) {
            return None;
        }
        match tier {
            None => tier = Some(head),
            Some(existing) if existing == head => {}
            Some(_) => return None,
        }
    }
    tier.map(str::to_owned)
}

/// The tier a `<dir>/*` whole-subdir glob names (`"q8/*"` → `"q8"`). `None` for any other pattern —
/// a specific file (`"q8/turbo_lora.safetensors"`) or a non-tier glob — so the coarse presence check
/// stays authoritative for explicit files.
fn whole_subdir_glob_tier(pattern: &str) -> Option<&str> {
    let (head, rest) = pattern.split_once('/')?;
    (rest == "*" && !head.is_empty() && !pattern_contains_glob(head)).then_some(head)
}

fn huggingface_filtered_cache_health(
    snapshots: &[PathBuf],
    files: &[String],
) -> HuggingFaceCacheHealth {
    let mut missing = files
        .iter()
        .filter(|pattern| {
            !snapshots
                .iter()
                .any(|snapshot| snapshot_contains_pattern(snapshot, pattern))
        })
        .cloned()
        .collect::<Vec<_>>();
    // Whether the COARSE check found none of the filter's patterns present — the "cleanly absent
    // tier" signal, captured before the tier-completeness augmentation below can add entries.
    let coarse_all_absent = missing.len() == files.len();

    // A flat sharded Transformers checkpoint commonly declares an index plus `model-*.safetensors`.
    // The glob is satisfied by the FIRST shard, so the coarse pattern check alone can call a torn
    // 15-shard model installed. Evaluate every candidate revision as one coherent snapshot and fold
    // in the shard set declared by its selected index. This reuses the whole-repo/receipt validator;
    // no tensor bytes or headers are read, only the small index and one stat per named shard.
    let has_flat_shard_glob = files.iter().any(|pattern| {
        !pattern.contains('/')
            && pattern_contains_glob(pattern)
            && pattern.to_ascii_lowercase().ends_with(".safetensors")
    });
    let indexed_health = has_flat_shard_glob.then(|| {
        snapshots
            .iter()
            .filter_map(|snapshot| {
                let files_on_disk = snapshot_files(snapshot);
                let selected_indexes = files_on_disk
                    .iter()
                    .filter(|file| {
                        file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX)
                            && files.iter().any(|pattern| pattern_matches(pattern, file))
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                if selected_indexes.is_empty() {
                    return None;
                }
                let mut candidate_missing = files
                    .iter()
                    .filter(|pattern| !snapshot_contains_pattern(snapshot, pattern))
                    .cloned()
                    .collect::<Vec<_>>();
                for shard in torn_shard_indexes_in(snapshot, &selected_indexes) {
                    if !candidate_missing.contains(&shard) {
                        candidate_missing.push(shard);
                    }
                }
                Some(candidate_missing)
            })
            .min_by_key(Vec::len)
    });
    if let Some(Some(indexed_missing)) = indexed_health {
        // Required metadata, index and shards must coexist in ONE revision. Replacing the coarse
        // cross-snapshot result also prevents different revisions from satisfying different files.
        missing = indexed_missing;
    }

    // Flat diffusers snapshots (Mage-Flow's logical q4/q8/bf16 load-time choices) list the
    // root `model_index.json` plus component globs rather than one `<tier>/*` subdir. The coarse
    // glob check only proves that each directory contains *something*; it does not prove the
    // model_index is valid or that every declared component has valid config + weights. Apply the
    // same full snapshot health used by the unfiltered path whenever this filter selects the root
    // model index. Multiple cached revisions are alternatives: one complete revision is enough.
    let flat_diffusers = snapshots
        .iter()
        .filter(|snapshot| {
            path_is_readable_file(&snapshot.join("model_index.json"))
                && files
                    .iter()
                    .any(|pattern| pattern_matches(pattern, "model_index.json"))
        })
        .collect::<Vec<_>>();
    if !flat_diffusers.is_empty() {
        let health = flat_diffusers
            .iter()
            .map(|snapshot| diffusers_snapshot_health(snapshot))
            .min_by_key(|health| health.missing_files.len())
            .expect("non-empty flat diffusers candidates");
        if !health.installed {
            for component in health.missing_files {
                if !missing.contains(&component) {
                    missing.push(component);
                }
            }
        }
    }

    // A `<tier>/*` whole-subdir glob is satisfied as soon as a SINGLE file under `<tier>/` exists, so
    // the coarse check never notices missing weights INSIDE the tier: a torn download (its
    // `model_index.json` + a config or two present, but the transformer/vae weights gone) reported a
    // green "Installed" badge, then failed to load at generation (`No such file or directory`). When a
    // whole-subdir tier is a diffusers pipeline (has `<tier>/model_index.json`), fold its missing
    // weight-bearing components — the SAME per-component check the non-tiered path uses — into
    // `missing`, scoped under the tier. Additional explicit patterns (e.g. a `<tier>/lora.safetensors`
    // co-requisite) are left to the coarse check above, so this never masks an explicitly-listed file.
    // A cleanly-absent tier (no `<tier>/model_index.json` present) adds nothing and stays MISSING, not
    // a repairable "incomplete" — so a valid single-quant install raises no spurious repair prompt
    // (sc-9907/sc-9909).
    for pattern in files {
        let Some(tier) = whole_subdir_glob_tier(pattern) else {
            continue;
        };
        // Evaluate the MOST-COMPLETE cached revision for this tier, not merely the one
        // `huggingface_snapshot_dirs` happens to rank first. A quant-matrix repo keeps every
        // revision's tiers in ONE shared cache, so after a manifest revision bump a freshly
        // re-downloaded complete tier coexists with the older torn snapshot it replaced. Picking
        // the first snapshot (which `refs/main` or a higher file count can front) let that stale
        // torn revision mask the complete one and report a permanent false "incomplete" — e.g. the
        // FLUX.2 MLX re-hosts whose old snapshots predated the `scheduler/` config (GH #1858): even
        // after re-download the card stayed incomplete because the old snapshot was fronted. Take the
        // fewest-missing tier across cached revisions — the SAME `min_by_key` the flat-diffusers and
        // unfiltered paths already use — so "is this tier complete in ANY cached revision?" is the
        // question, matching that a single valid install makes the model usable.
        let Some(health) = snapshots
            .iter()
            .map(|snapshot| snapshot.join(tier))
            .filter(|dir| path_is_readable_file(&dir.join("model_index.json")))
            .map(|dir| diffusers_snapshot_health(&dir))
            .min_by_key(|health| health.missing_files.len())
        else {
            continue;
        };
        for component in health.missing_files {
            let scoped = format!("{tier}/{component}");
            if !missing.contains(&scoped) {
                missing.push(scoped);
            }
        }
    }

    if missing.is_empty() {
        // Every expected file/pattern is present AND (for a diffusers tier) its weights are on disk.
        HuggingFaceCacheHealth::installed()
    } else if coarse_all_absent {
        // NONE of this filter's expected patterns are present: the tier is cleanly absent, not torn.
        // A quant-matrix model keeps every tier in ONE shared repo cache (bf16/, q8/, q4/ subdirs),
        // so downloading one tier populates the repo snapshot the OTHER tiers' filters also probe.
        // Reporting a not-downloaded tier as `incomplete` is what surfaced a false "Cached files are
        // incomplete" warning + Fix button for a perfectly valid single-quant install (sc-9907).
        // "You didn't download this tier" (missing) must stay distinct from "this tier is
        // half-downloaded" (incomplete) so nothing upstream raises a spurious repair prompt.
        HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: missing,
        }
    } else {
        // Some expected files present but not all — an explicit file is absent, or a diffusers tier is
        // torn (weights missing). A re-fetch repairs it.
        HuggingFaceCacheHealth::missing(missing)
    }
}

fn snapshot_contains_pattern(snapshot: &FsPath, pattern: &str) -> bool {
    if pattern_contains_glob(pattern) {
        return snapshot_files(snapshot)
            .into_iter()
            .any(|path| pattern_matches(pattern, &path));
    }
    path_is_readable_file(&snapshot.join(pattern))
}

fn pattern_contains_glob(pattern: &str) -> bool {
    pattern
        .chars()
        .any(|character| matches!(character, '*' | '?' | '[' | ']'))
}

fn diffusers_snapshot_health(snapshot: &FsPath) -> HuggingFaceCacheHealth {
    let model_index_path = snapshot.join("model_index.json");
    let Ok(contents) = std::fs::read_to_string(&model_index_path) else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };
    let Ok(index) = serde_json::from_str::<Value>(&contents) else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };
    let Some(index) = index.as_object() else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };

    let mut missing = Vec::new();
    let is_mage = index
        .get("_class_name")
        .and_then(Value::as_str)
        .is_some_and(|value| value == "MageFlowPipeline");
    if is_mage
        && sceneworks_core::lora_family::detect_model_family(snapshot)
            .ok()
            .flatten()
            .as_deref()
            != Some("mage-flow")
    {
        missing.push("transformer/config.json (invalid Mage config)".to_owned());
    }
    for (component, spec) in index {
        if component.starts_with('_') || spec.is_null() {
            continue;
        }
        let class_name = spec
            .as_array()
            .and_then(|items| items.get(1))
            .and_then(Value::as_str)
            .unwrap_or_default();
        // diffusers records optional components that the pipeline doesn't use
        // as `[null, null]` (e.g. ChromaPipeline's `feature_extractor` and
        // `image_encoder`). These have no directory or files on disk by design,
        // so an empty class name means "absent" — skip it rather than reporting
        // its config/weights as missing and marking the whole model incomplete.
        if class_name.is_empty() {
            continue;
        }
        if diffusers_component_requires_weights(component, class_name) {
            // Weight-bearing components (unet, transformer, vae, text_encoder,
            // controlnet, …) reliably ship a `config.json` alongside their
            // weight files, so require both.
            if !path_is_valid_json_object(&snapshot.join(format!("{component}/config.json"))) {
                missing.push(format!("{component}/config.json"));
            }
            // ONE listing of the component directory, asked two questions below (sc-20526).
            let component_entries = diffusers_component_entries(snapshot, component);
            if !diffusers_component_has_weight_file_in(&component_entries) {
                missing.push(format!("{component}/<weights>"));
            } else if is_mage && !diffusers_component_safetensors_are_valid(snapshot, component) {
                missing.push(format!("{component}/<weights> (malformed safetensors)"));
            } else {
                // A sharded component holds "a weight file" as soon as ONE shard landed. The
                // component's own index declares the rest, so an interrupted download is only
                // visible by comparing `weight_map` against disk (sc-20526).
                for shard in
                    component_missing_shards_in(&snapshot.join(component), &component_entries)
                {
                    missing.push(format!("{component}/{shard}"));
                }
            }
        } else if is_mage && component == "tokenizer" {
            // Mage's Qwen3-VL AutoProcessor is a logical `tokenizer` component in model_index.json,
            // but the published diffusers snapshots colocate its tokenizer + vision processor
            // configs under `text_encoder/` beside the Qwen3-VL weights.
            for config in ["tokenizer_config.json", "preprocessor_config.json"] {
                if !path_is_valid_json_object(&snapshot.join("text_encoder").join(config)) {
                    missing.push(format!("text_encoder/{config}"));
                }
            }
        } else if !diffusers_component_has_valid_config_file(snapshot, component) {
            // Weightless auxiliary components (scheduler, tokenizer, feature
            // extractors, and image/video/composite processors) ship config
            // files whose names vary by class — scheduler_config.json,
            // tokenizer_config.json, preprocessor_config.json, and more. Hard
            // coding each variant is what produced repeated false "incomplete"
            // reports (Chroma's null optionals, Qwen2VLProcessor), so only
            // require the component directory to exist and hold at least one
            // file. A genuinely missing/partial component still trips this.
            missing.push(format!("{component}/<config>"));
        }
    }
    if missing.is_empty() {
        HuggingFaceCacheHealth::installed()
    } else {
        missing.sort();
        missing.dedup();
        HuggingFaceCacheHealth::missing(missing)
    }
}

/// Classifies a diffusers `model_index.json` component as weight-bearing.
/// Schedulers, tokenizers, feature extractors, and composite `*Processor`
/// wrappers (e.g. Qwen2VLProcessor) carry no model weights — `contains("processor")`
/// subsumes `imageprocessor` and the composite processors.
fn diffusers_component_requires_weights(component: &str, class_name: &str) -> bool {
    let class = class_name.to_ascii_lowercase();
    !(component.contains("scheduler")
        || class.contains("scheduler")
        || class.contains("tokenizer")
        || class.contains("featureextractor")
        || class.contains("processor"))
}

fn path_is_valid_json_object(path: &FsPath) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|contents| serde_json::from_slice::<Value>(&contents).ok())
        .is_some_and(|value| value.is_object())
}

/// Weightless diffusers components use different JSON config filenames, but every shipped
/// scheduler/tokenizer/processor has at least one JSON object. Requiring a parseable object rejects
/// an empty or corrupted config without hard-coding class-specific filenames.
fn diffusers_component_has_valid_config_file(snapshot: &FsPath, component: &str) -> bool {
    std::fs::read_dir(snapshot.join(component))
        .map(|entries| {
            entries.flatten().any(|entry| {
                let path = entry.path();
                !is_hidden_file(&path)
                    && path.extension().and_then(|value| value.to_str()) == Some("json")
                    && path_is_valid_json_object(&path)
            })
        })
        .unwrap_or(false)
}

/// Every non-hidden readable entry directly inside a diffusers component directory.
///
/// The weight-file probe and the shard-index probe ask different questions of the SAME listing, so
/// the health loop reads the directory once and hands the result to both (sc-20526). Threading the
/// list is what makes the "one `read_dir` per component" cost claim true rather than aspirational.
fn diffusers_component_entries(snapshot: &FsPath, component: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(snapshot.join(component)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| !is_hidden_file(path) && path_is_readable_file(path))
        .collect()
}

fn diffusers_component_has_weight_file_in(entries: &[PathBuf]) -> bool {
    entries.iter().any(|path| {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        name.ends_with(".safetensors")
            || name.ends_with(".bin")
            || name.ends_with(".msgpack")
            || name.ends_with(".gguf")
    })
}

/// Shards that a component directory's `*.safetensors.index.json` files name but that are not on
/// disk, reported relative to the component (sc-20526).
///
/// A sharded component satisfies [`diffusers_component_has_weight_file_in`] as soon as ONE shard
/// landed, so an interrupted download left `model.safetensors.index.json` + the last shard and read
/// "installed" — then died at the first forward pass with `cannot find tensor
/// model.embed_tokens.weight` (lens_turbo bf16). The index IS the declared file set for a sharded
/// component, so comparing it against disk is the workable ground truth even when the manifest
/// declares only a `<tier>/*` glob or a whole-repo fetch.
///
/// Cost: no directory read at all — `entries` is the listing
/// [`diffusers_component_entries`] already spent for the weight-file probe — plus one small JSON
/// read per index and a stat per distinct shard. No hashing, no header parse, no tensor read.
fn component_missing_shards_in(dir: &FsPath, entries: &[PathBuf]) -> Vec<String> {
    let mut missing = Vec::new();
    for path in entries
        .iter()
        .filter(|path| sceneworks_core::safetensors::is_safetensors_index_path(path))
    {
        missing.extend(sceneworks_core::safetensors::missing_indexed_shards(
            dir, path,
        ));
    }
    missing.sort();
    missing.dedup();
    missing
}

/// Whether every `*.safetensors.index.json` NAMED IN `files` (snapshot-relative paths, as a receipt's
/// `resolvedFiles` or a backfill's resolved set records them) has all of its shards on disk.
///
/// This is the receipt-lane half of the shard check and costs nothing beyond a stat per shard: the
/// file list is already in hand, so no directory walk is needed to find the indexes (sc-20526).
fn listed_shard_indexes_are_complete(snapshot: &FsPath, files: &[String]) -> bool {
    files
        .iter()
        .filter(|file| file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX))
        .all(|file| {
            let path = snapshot.join(file);
            path.parent().is_some_and(|dir| {
                sceneworks_core::safetensors::indexed_shards_are_present(dir, &path)
            })
        })
}

fn diffusers_component_safetensors_are_valid(snapshot: &FsPath, component: &str) -> bool {
    let files = snapshot_files(&snapshot.join(component))
        .into_iter()
        .filter(|path| path.to_ascii_lowercase().ends_with(".safetensors"))
        .collect::<Vec<_>>();
    !files.is_empty()
        && files.iter().all(|path| {
            sceneworks_core::safetensors::file_is_structurally_valid(
                &snapshot.join(component).join(path),
            )
        })
}

/// Whether an already-walked snapshot file list holds anything that is not documentation/artwork.
///
/// Takes the list rather than the directory so the whole-repo health probe walks once and reuses the
/// result for both the payload question and the shard-index validation (sc-20526).
fn snapshot_has_payload_file_in(files: &[String]) -> bool {
    files.iter().any(|path| {
        let lower = path.to_ascii_lowercase();
        !lower.ends_with(".md")
            && !lower.ends_with(".png")
            && !lower.ends_with(".jpg")
            && !lower.ends_with(".jpeg")
            && !lower.ends_with(".gitattributes")
    })
}

/// Snapshot-relative shard paths that a `*.safetensors.index.json` in `files` names but that are not
/// on disk. Takes the already-walked file list so no extra directory traversal is spent (sc-20526).
fn torn_shard_indexes_in(snapshot: &FsPath, files: &[String]) -> Vec<String> {
    let mut missing = files
        .iter()
        .filter(|file| file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX))
        .flat_map(|file| {
            let path = snapshot.join(file);
            let dir = path.parent().map(FsPath::to_path_buf).unwrap_or_default();
            let prefix = match file.rsplit_once('/') {
                Some((parent, _)) => format!("{parent}/"),
                None => String::new(),
            };
            sceneworks_core::safetensors::missing_indexed_shards(&dir, &path)
                .into_iter()
                .map(move |shard| format!("{prefix}{shard}"))
        })
        .collect::<Vec<_>>();
    missing.sort();
    missing.dedup();
    missing
}

/// Every readable file under `snapshot`, snapshot-relative, `/`-separated.
///
/// Hidden entries are excluded. They are not payload, and — because this list backs
/// [`snapshot_contains_pattern`]'s glob branch — a `._model.safetensors` sidecar would
/// otherwise satisfy a required `*.safetensors` pattern, reporting a model installed
/// while its real weights file is absent (SceneWorks#1333).
fn snapshot_files(snapshot: &FsPath) -> Vec<String> {
    let mut output = Vec::new();
    let mut stack = vec![snapshot.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_hidden_file(&path) {
                continue;
            } else if path_is_readable_file(&path) {
                if let Ok(relative) = path.strip_prefix(snapshot) {
                    output.push(relative.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    output
}

fn path_is_readable_file(path: &FsPath) -> bool {
    if std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file()) {
        return true;
    }
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.file_type().is_symlink() {
        return false;
    }
    std::fs::File::open(path).is_ok()
}

pub(crate) fn manifest_download_size_bytes(model: &Value, download: &Value) -> Option<u64> {
    // Prefer the selected download entry, then fall back to legacy model-level metadata.
    ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
        .iter()
        .find_map(|field| download.get(*field).and_then(json_size_to_u64))
        .or_else(|| {
            ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"]
                .iter()
                .find_map(|field| model.get(*field).and_then(json_size_to_u64))
        })
}

fn model_size_estimation_disabled(_state: &AppState) -> bool {
    #[cfg(test)]
    if let Some(disabled) = *_state
        .test_hooks
        .model_size_estimate_disabled_override
        .lock()
    {
        return disabled;
    }
    sceneworks_core::env_util::env_flag("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE", false)
}

pub(crate) async fn estimate_huggingface_download_size(
    state: &AppState,
    repo: &str,
    files: &[String],
) -> Option<u64> {
    if model_size_estimation_disabled(state) {
        return None;
    }
    let cache_key = (repo.to_owned(), files.to_vec());
    loop {
        let lookup = state.model_size_cache.lock().lookup_or_start(&cache_key);
        match lookup {
            ModelSizeLookup::Cached(cached) => return cached,
            ModelSizeLookup::Follow(in_flight) => {
                #[cfg(test)]
                if let Some(hook) = state
                    .test_hooks
                    .model_size_estimate_test_hook
                    .lock()
                    .clone()
                {
                    hook.note_follower();
                }
                match in_flight.wait().await {
                    ModelSizeFlightStatus::Complete(estimate) => return estimate,
                    ModelSizeFlightStatus::Aborted => continue,
                    ModelSizeFlightStatus::Pending => unreachable!("wait returns a terminal state"),
                }
            }
            ModelSizeLookup::Lead(in_flight) => {
                let leader = ModelSizeFlightLeader::new(
                    state.model_size_cache.clone(),
                    cache_key.clone(),
                    in_flight,
                );
                let estimate = estimate_huggingface_download_size_request(state, repo, files).await;
                leader.finish(estimate);
                return estimate;
            }
            ModelSizeLookup::Unshared => {
                let estimate = estimate_huggingface_download_size_request(state, repo, files).await;
                match estimate {
                    Some(estimate) => state
                        .model_size_cache
                        .lock()
                        .insert(cache_key.clone(), estimate),
                    None => state
                        .model_size_cache
                        .lock()
                        .insert_failure(cache_key.clone()),
                }
                return estimate;
            }
        }
    }
}

async fn estimate_huggingface_download_size_request(
    state: &AppState,
    repo: &str,
    files: &[String],
) -> Option<u64> {
    #[cfg(test)]
    let test_hook = {
        state
            .test_hooks
            .model_size_estimate_test_hook
            .lock()
            .clone()
    };
    #[cfg(test)]
    if let Some(hook) = test_hook {
        return hook.request().await;
    }
    let url = format!(
        "https://huggingface.co/api/models/{}?blobs=true",
        quote_huggingface_repo(repo)
    );
    estimate_huggingface_download_size_uncached(&state.http_client, &url, files).await
}

pub(crate) async fn estimate_huggingface_download_size_uncached(
    client: &reqwest::Client,
    url: &str,
    files: &[String],
) -> Option<u64> {
    let payload = tokio::time::timeout(Duration::from_secs(8), async {
        client
            .get(url.to_owned())
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json::<Value>()
            .await
            .ok()
    })
    .await
    .ok()??;
    let siblings = payload.get("siblings")?.as_array()?;
    download_size_from_siblings(siblings, files)
}

pub(crate) fn download_size_from_siblings(siblings: &[Value], files: &[String]) -> Option<u64> {
    let mut total = 0_u64;
    let mut found_size = false;
    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !allow_pattern_matches(filename, files) {
            continue;
        }
        let Some(size) = sibling.get("size").and_then(json_size_to_u64) else {
            continue;
        };
        found_size = true;
        total = total.saturating_add(size);
    }
    found_size.then_some(total)
}

pub(crate) fn model_is_installed(path: &FsPath) -> bool {
    path.is_dir() && path.join(".sceneworks-download-complete.json").is_file()
}

pub(crate) struct MlxCatalogStatus {
    pub(crate) install_state: &'static str,
    pub(crate) conversion_state: &'static str,
    pub(crate) converted_path: Option<PathBuf>,
    /// A newer source checkpoint is available than the one this install was converted from.
    /// True only for a converted `requiresConversion` model whose manifest `convertSourceFile`
    /// is NOT present in the `convertSourceRepo` cache (the installed converted dir carries no
    /// version stamp, so the source cache is the proxy — see `convert_source_file_cached`).
    pub(crate) update_available: bool,
    /// Why a converted artifact that carries the install marker would still be refused at load
    /// (sc-24216): the engine's own pin-contract reason, from
    /// [`sceneworks_core::converted_artifacts`]. `Some` only for a converter with a declared
    /// contract whose marked output fails it — the status then reads as needing conversion, so the
    /// Model Manager offers Convert again instead of a model that cannot load.
    pub(crate) conversion_problem: Option<String>,
}

/// Whether the named source `file` is present in any cached snapshot of `repo` — the proxy for
/// "the current manifest source has been downloaded." Keys off the manifest fields alone, so it
/// works for every convert-at-install model with no per-model logic.
fn convert_source_file_cached(data_dir: &FsPath, repo: &str, file: &str) -> bool {
    huggingface_repo_cache_path(data_dir, repo)
        .map(|root| crate::huggingface_snapshot_dirs(&root))
        .unwrap_or_default()
        .iter()
        .any(|snapshot| snapshot.join(file).is_file())
}

/// macOS Model Manager status for a model's `mlx` variant. Returns `None` when the
/// model declares no `mlx` block.
///
/// `conversion_state`:
/// - `ready`            turnkey MLX repo (no conversion needed)
/// - `converted`        requiresConversion and the local MLX dir exists
/// - `needs_conversion` source checkpoint present, MLX dir absent
/// - `needs_source`     source checkpoint not downloaded yet
///
/// `install_state` is `installed` when the usable MLX artifact exists.
pub(crate) fn mlx_catalog_status(
    model: &serde_json::Map<String, Value>,
    data_dir: &FsPath,
) -> Option<MlxCatalogStatus> {
    let mlx = model.get("mlx").and_then(Value::as_object)?;
    let repo_cached = |repo: &str| {
        huggingface_repo_cache_path(data_dir, repo)
            .as_deref()
            .is_some_and(huggingface_repo_cache_exists)
    };
    if mlx
        .get("requiresConversion")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let converted_dir = data_dir.join("models").join("mlx").join(model_id);
        // mlx-video converters write a top-level config.json; the FLUX.2-klein
        // diffusers converter (sc-2235) writes a diffusers model_index.json. Either
        // marks a finished local MLX artifact.
        let marked = converted_dir.join("config.json").is_file()
            || converted_dir.join("model_index.json").is_file();
        // A marked artifact from a converter with an engine pin contract must also satisfy it
        // (manifest, pins, file sizes — metadata reads only); a truncated, edited or half-written
        // bundle is reported with the engine's reason rather than as installed (sc-24216).
        let conversion_problem = if marked {
            sceneworks_core::converted_artifacts::converted_artifact_problem(
                mlx.get("converter")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                &converted_dir,
                sceneworks_core::converted_artifacts::ArtifactCheck::Sizes,
            )
        } else {
            None
        };
        if marked && conversion_problem.is_none() {
            // The converted artifact records no source version, so use the source cache as the
            // proxy: if the manifest's current `convertSourceFile` is NOT cached, this install was
            // built from an older source → an update is available. A dir-based converter with no
            // `convertSourceFile` simply never reports an update (no false positives).
            let update_available = match (
                mlx.get("convertSourceRepo").and_then(Value::as_str),
                mlx.get("convertSourceFile").and_then(Value::as_str),
            ) {
                (Some(repo), Some(file)) if !file.trim().is_empty() => {
                    !convert_source_file_cached(data_dir, repo, file)
                }
                _ => false,
            };
            return Some(MlxCatalogStatus {
                install_state: "installed",
                conversion_state: "converted",
                converted_path: Some(converted_dir),
                update_available,
                conversion_problem: None,
            });
        }
        let source_present = mlx
            .get("convertSourceRepo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        Some(MlxCatalogStatus {
            install_state: "missing",
            conversion_state: if source_present {
                "needs_conversion"
            } else {
                "needs_source"
            },
            converted_path: None,
            update_available: false,
            conversion_problem,
        })
    } else {
        let repo_installed = mlx
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(repo_cached);
        // A turnkey model may still be served by a pre-existing local conversion at
        // <data>/models/mlx/<id> — the worker's resolve_*_model_dir prefers a local dir over
        // the turnkey download. Count that as installed too, so a model flipped from
        // requiresConversion → turnkey (sc-5599) doesn't read as "missing" for users who had
        // already converted it locally.
        let model_id = model.get("id").and_then(Value::as_str).unwrap_or_default();
        let local_dir = data_dir.join("models").join("mlx").join(model_id);
        let local_installed = local_dir.join("config.json").is_file();
        Some(MlxCatalogStatus {
            install_state: if repo_installed || local_installed {
                "installed"
            } else {
                "missing"
            },
            conversion_state: "ready",
            converted_path: local_installed.then_some(local_dir),
            // Turnkey models have no local conversion to go stale (they track their repo directly).
            update_available: false,
            conversion_problem: None,
        })
    }
}

pub(crate) fn model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = model_manifest_installed_path(model, data_dir) {
        paths.push(path);
    }
    // A convert-at-install model's converted output (`<data>/models/mlx/<id>`, the `outputDir` every
    // convert job is given) is owned by this entry alone, so a whole-model delete reclaims it with
    // the source checkpoint (sc-24216). Before this the delete removed the source only and the
    // marker left behind kept the model reading as installed. Co-requisites (shared components such
    // as a text encoder or VAE) are never listed here — see `is_co_requisite_download`.
    if let Some(converted) = converted_artifact_dir(model, data_dir) {
        paths.push(converted);
    }
    let cleanup_downloads = model_download(model).into_iter().collect::<Vec<_>>();
    if model.get("platformCleanupOnly").and_then(Value::as_bool) == Some(true) {
        paths.extend(cleanup_with_model_artifact_paths(model, data_dir));
    }
    for repo in cleanup_downloads.into_iter().filter_map(|download| {
        download
            .get("repo")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }) {
        paths.push(data_dir.join("models").join(safe_download_dir(&repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, &repo) {
            paths.push(cache_path);
        }
    }
    if let Some(source_path) = model
        .get("source")
        .and_then(Value::as_object)
        .and_then(|source| source.get("path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains("${"))
    {
        let path = PathBuf::from(source_path);
        let run_dir = if path.is_absolute() {
            path
        } else {
            data_dir.join(path)
        };
        // An Iris training run's checkpoint root and its `.prior` retirement (sc-25690), named by
        // the engine contract. They sit inside the run directory, which goes first so a trash
        // delete moves the run as one item; the named paths then catch any the directory missed.
        let checkpoint_dirs = crate::iris_training::iris_run_checkpoint_paths(model, &run_dir);
        paths.push(run_dir);
        paths.extend(checkpoint_dirs);
    }
    unique_paths(paths)
}

/// The converted-artifact dir a `mlx.requiresConversion` entry owns: `<data>/models/mlx/<id>`, the
/// convert job's `outputDir`. `None` for every other entry, and for an id that is not one plain path
/// component (never joined outside the conversion root).
fn converted_artifact_dir(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let requires_conversion = model
        .get("mlx")
        .and_then(|mlx| mlx.get("requiresConversion"))
        .and_then(Value::as_bool)
        == Some(true);
    if !requires_conversion {
        return None;
    }
    let id = model.get("id").and_then(Value::as_str)?;
    let mut components = FsPath::new(id).components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(_)), None) => {
            Some(data_dir.join("models").join("mlx").join(id))
        }
        _ => None,
    }
}

fn cleanup_with_model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for repo in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| download.get("cleanupWithModel").and_then(Value::as_bool) == Some(true))
        .filter_map(|download| download.get("repo").and_then(Value::as_str))
    {
        paths.push(data_dir.join("models").join(safe_download_dir(repo)));
        if let Some(cache_path) = huggingface_repo_cache_path(data_dir, repo) {
            paths.push(cache_path);
        }
    }
    unique_paths(paths)
}

pub(crate) fn model_manifest_installed_path(model: &Value, data_dir: &FsPath) -> Option<PathBuf> {
    let raw_path = model
        .get("paths")
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if raw_path.contains("${") {
        return None;
    }
    let path = PathBuf::from(raw_path);
    Some(if path.is_absolute() {
        path
    } else {
        data_dir.join(path)
    })
}

#[cfg(test)]
#[path = "models/tests/gated_credential_tests.rs"]
mod gated_credential_tests;

#[cfg(test)]
#[path = "models/tests/variant_install_tests.rs"]
mod variant_install_tests;

#[cfg(test)]
#[path = "models/tests/mlx_tier_probe_tests.rs"]
mod mlx_tier_probe_tests;

// Per-tier delete (sc-12024, epic 8506). Exercises the blob-aware reclamation on a realistic HF
// hub-cache layout — real `blobs/<etag>` files with snapshot SYMLINKS into them — which is the whole
// reason a tier delete is non-trivial: unlinking the tier's snapshot symlinks alone frees nothing.
// unix-gated because the fixtures use symlinks (the production cache layout on macOS/Linux).
#[cfg(all(test, unix))]
#[path = "models/tests/variant_delete_tests.rs"]
mod variant_delete_tests;

/// Lane-parameterized coverage for the imported-model LoRA advertisement withdrawal.
///
/// These drive `apply_imported_lora_advertisement_for_lanes` directly rather than through the
/// platform-bound wrapper, so BOTH deployment topologies are exercised on every CI platform.
#[cfg(test)]
#[path = "models/tests/imported_lora_advertisement_tests.rs"]
mod imported_lora_advertisement_tests;

/// sc-20636 (epic 20398): the import API's ownership mode + discriminated source, and the
/// normalisation that keeps every legacy flat request working and managed.
#[cfg(test)]
#[path = "models/tests/model_import_source_tests.rs"]
mod model_import_source_tests;

#[cfg(test)]
#[path = "models/tests/managed_predictor_queue_tests.rs"]
mod managed_predictor_queue_tests;

#[cfg(test)]
#[path = "models/tests/minimax_h3_import_declaration_tests.rs"]
mod minimax_h3_import_declaration_tests;

/// The imported-checkpoint product surface (sc-26687) — what Model Manager, Video Studio, the API
/// and MCP all read. These assert the VALUES a user acts on: the declared task contract's
/// operations, the backends that will actually claim the import, and the repair a missing
/// component calls for. Not that a field was written.
#[cfg(test)]
#[path = "models/tests/imported_checkpoint_product_surface_tests.rs"]
mod imported_checkpoint_product_surface_tests;

#[cfg(test)]
pub(crate) use crate::test_hooks::{
    test_catalog_probe_limit, test_catalog_probe_peak, test_delay_catalog_probe,
    test_reset_catalog_probe_concurrency, ModelCatalogBeforePublishTestHook,
    ModelSizeEstimateTestHook,
};

#[cfg(test)]
#[path = "models/tests/io_offload_tests.rs"]
mod io_offload_tests;

#[cfg(test)]
#[path = "models/tests/model_contract_metadata_tests.rs"]
mod model_contract_metadata_tests;
