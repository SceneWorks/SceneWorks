use super::*;

use sceneworks_core::base_weights::{detect_base_weight_file, import_detection_supported};
use sceneworks_core::credentials::normalize_host;
use sceneworks_core::lora_family::is_hidden_file;

const ALLOWED_MODEL_TYPES: &[&str] = &["image", "video", "audio", "utility"];
// The shipped catalog currently has more than 64 distinct (repo, files) contexts. Keep ample
// headroom for catalog growth so an in-order scan cannot evict the entries the next scan is about
// to read and degrade into a zero-hit cycle.
const MODEL_SIZE_CACHE_LIMIT: usize = 256;
const MODEL_CATALOG_PROBE_CONCURRENCY: usize = 16;
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
    #[cfg(test)]
    before_publish_test_hook: Mutex<Option<ModelCatalogBeforePublishTestHook>>,
}

#[derive(Default)]
struct ModelCatalogCacheState {
    generation: u64,
    snapshot: Option<(u64, Arc<Vec<Value>>)>,
}

impl ModelCatalogCache {
    pub(crate) fn invalidate(&self) {
        let mut state = self.state.lock();
        state.generation = state.generation.wrapping_add(1);
        state.snapshot = None;
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.state.lock().generation
    }

    #[cfg(test)]
    pub(crate) fn set_before_publish_test_hook(&self, hook: ModelCatalogBeforePublishTestHook) {
        *self.before_publish_test_hook.lock() = Some(hook);
    }

    #[cfg(test)]
    async fn pause_before_publish_for_test(&self) {
        let hook = self.before_publish_test_hook.lock().take();
        if let Some(hook) = hook {
            hook.pause().await;
        }
    }
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ModelCatalogBeforePublishTestHook {
    reached: Arc<tokio::sync::Semaphore>,
    release: Arc<tokio::sync::Semaphore>,
}

#[cfg(test)]
impl ModelCatalogBeforePublishTestHook {
    pub(crate) fn blocked() -> Self {
        Self {
            reached: Arc::new(tokio::sync::Semaphore::new(0)),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
        }
    }

    pub(crate) async fn wait_until_reached(&self) {
        self.reached
            .acquire()
            .await
            .expect("model-catalog publish hook remains open")
            .forget();
    }

    pub(crate) fn release(&self) {
        self.release.add_permits(1);
    }

    async fn pause(&self) {
        self.reached.add_permits(1);
        self.release
            .acquire()
            .await
            .expect("model-catalog publish hook remains open")
            .forget();
    }
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ModelSizeEstimateTestHook {
    calls: Arc<std::sync::atomic::AtomicUsize>,
    followers: Arc<std::sync::atomic::AtomicUsize>,
    started: Arc<tokio::sync::Semaphore>,
    joined: Arc<tokio::sync::Semaphore>,
    release: Option<Arc<tokio::sync::Semaphore>>,
    response: Arc<Mutex<Option<u64>>>,
}

#[cfg(test)]
impl ModelSizeEstimateTestHook {
    pub(crate) fn immediate(response: Option<u64>) -> Self {
        Self {
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            followers: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            started: Arc::new(tokio::sync::Semaphore::new(0)),
            joined: Arc::new(tokio::sync::Semaphore::new(0)),
            release: None,
            response: Arc::new(Mutex::new(response)),
        }
    }

    pub(crate) fn blocked(response: Option<u64>) -> Self {
        Self {
            release: Some(Arc::new(tokio::sync::Semaphore::new(0))),
            ..Self::immediate(response)
        }
    }

    pub(crate) fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub(crate) fn follower_count(&self) -> usize {
        self.followers.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub(crate) async fn wait_for_call(&self) {
        self.started
            .acquire()
            .await
            .expect("model-size test hook remains open")
            .forget();
    }

    pub(crate) async fn wait_for_follower(&self) {
        self.joined
            .acquire()
            .await
            .expect("model-size test hook remains open")
            .forget();
    }

    pub(crate) fn set_response(&self, response: Option<u64>) {
        *self.response.lock() = response;
    }

    pub(crate) fn release_one(&self) {
        self.release
            .as_ref()
            .expect("blocked model-size test hook")
            .add_permits(1);
    }

    fn note_follower(&self) {
        self.followers
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.joined.add_permits(1);
    }

    async fn request(&self) -> Option<u64> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.started.add_permits(1);
        if let Some(release) = &self.release {
            release
                .acquire()
                .await
                .expect("model-size test hook remains open")
                .forget();
        }
        *self.response.lock()
    }
}

fn validate_huggingface_repo(repo: &str) -> Result<(), ApiError> {
    let parts: Vec<_> = repo.trim().split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with('.')
                || part.ends_with('.')
                || !part.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
                })
        })
    {
        return Err(ApiError::bad_request(
            "Hugging Face repo must be in owner/name form",
        ));
    }
    Ok(())
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
fn model_requires_license_acknowledgment(model: &Value) -> bool {
    model
        .get("requiresLicenseAcknowledgment")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// The payload key every job-creation door uses to carry the caller's acknowledgment through to
/// the queue (sc-17227). Stamped by `create_model_download_job` once its own gate has passed, so a
/// RETRY of a legitimately-authorized download re-validates against the same assertion rather than
/// being refused for a field the typed route never wrote.
pub(crate) const LICENSE_ACKNOWLEDGED_PAYLOAD_KEY: &str = "licenseAcknowledged";

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
    let mut index = std::collections::BTreeMap::new();
    for model in models {
        if !model_requires_license_acknowledgment(&model) {
            continue;
        }
        let Some(model_id) = model.get("id").and_then(Value::as_str) else {
            continue;
        };
        let model_name = model
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(model_id)
            .to_owned();
        for download in model
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(key) = download
                .get("repo")
                .and_then(Value::as_str)
                .and_then(huggingface_repo_key)
            else {
                continue;
            };
            index
                .entry(key)
                .or_insert_with(|| LicenseAcknowledgmentSource {
                    model_id: model_id.to_owned(),
                    model_name: model_name.clone(),
                });
        }
    }
    Ok(index)
}

/// The catalog entry whose licence acknowledgment covers a fetch of some repo. The NAME travels
/// with the id because the surfaces that have to explain the requirement — the LoRA rows on the
/// Models screen — have no licence copy of their own and must point the user at the model card
/// that does.
#[derive(Clone)]
pub(crate) struct LicenseAcknowledgmentSource {
    pub(crate) model_id: String,
    pub(crate) model_name: String,
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
        let Some(source) = index.get(key.as_str()) else {
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
/// `acknowledged` is the caller's own assertion, exactly as on the typed route: the gate obtains
/// an affirmative acknowledgment, it is not an authorization check (see
/// `docs/minimax-h3-use-restriction-safeguards.md`).
pub(crate) async fn ensure_license_acknowledged_for_source(
    state: &AppState,
    repos: &[Option<&str>],
    source_url: Option<&str>,
    acknowledged: bool,
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
    let index = license_acknowledgment_repo_index(state).await?;
    let Some((requested, source)) = candidates
        .iter()
        .find_map(|(named, key)| index.get(key.as_str()).map(|source| (named, source)))
    else {
        return Ok(());
    };
    if acknowledged {
        return Ok(());
    }
    let model_id = &source.model_id;
    Err(ApiError {
        status: StatusCode::FORBIDDEN,
        detail: format!(
            "'{requested}' supplies '{model_id}', which requires accepting its license before \
             download. Accept the license on the Models screen, or send \
             `licenseAcknowledged: true` to assert that the user has accepted it."
        ),
        code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
    })
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

pub(crate) async fn create_model_download_job(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    ApiJson(payload): ApiJson<ModelDownloadRequest>,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
            code: None,
        })?;
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
    if model_requires_license_acknowledgment(&model) && !payload.license_acknowledged {
        return Err(ApiError {
            status: StatusCode::FORBIDDEN,
            detail: format!(
                "Model '{model_id}' requires accepting its license before download. Accept the \
                 license on the Models screen, or send `licenseAcknowledged: true` to assert that \
                 the user has accepted it."
            ),
            code: Some(LICENSE_ACKNOWLEDGMENT_REQUIRED_CODE),
        });
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
    // The selected `download` is always the primary/tier entry — `model_download` and
    // `model_download_for_variant` skip co-requisites (sc-9696), so a co-requisite can never be
    // installed as if it were the model itself.
    let requested_variant = payload
        .variant
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    // Only the SELECTED tier's co-requisites (sc-14980). Mage-Flow's shared text encoder exists as
    // three per-tier subtrees; fetching all of them would pull 16.1 GB of text encoder for a q4
    // install that needs 2.51 GB. Tier-agnostic co-requisites (every other family) are unaffected —
    // they carry no `variant` and always apply. Read the tier off the resolved `download` rather than
    // the request so the default-tier install (no explicit `variant`) picks up its tier too.
    let selected_variant = download
        .get("variant")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let co_requisites =
        model_co_requisite_downloads_for_variant(&model, selected_variant.as_deref());

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
    ensure_license_acknowledged_for_source(
        &state,
        &queued_repos,
        None,
        payload.license_acknowledged,
    )
    .await?;

    let job_payload = build_model_download_job_payload(
        &model,
        &model_id,
        &download,
        requested_variant,
        true,
        payload.license_acknowledged,
        &state.settings.data_dir,
    )?;

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
        let co_payload = build_model_download_job_payload(
            &model,
            &model_id,
            co_requisite,
            None,
            false,
            payload.license_acknowledged,
            &state.settings.data_dir,
        )?;
        create_generation_job(
            state.clone(),
            JobType::ModelDownload,
            None,
            None,
            co_payload,
            requested_gpu.clone(),
        )
        .await?;
    }

    // The primary job is the one returned to the caller (its id is what the download UI tracks).
    let job = create_generation_job(
        state,
        JobType::ModelDownload,
        None,
        None,
        job_payload,
        requested_gpu,
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
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
    // Record the acknowledgment ON the job (sc-17227). `create_model_download_job` — the only
    // non-test caller — has already refused the request unless the caller asserted it, so reaching
    // here for a gated fetch means the assertion was made. Writing it into the payload is what
    // keeps RETRY and DUPLICATE working: those re-run `validate_raw_job_payload` over the stored
    // payload, and the repo-keyed gate there would otherwise refuse a download the typed route had
    // already authorized. Co-requisites carry it too — `MiniMaxAI/MiniMax-H3` is itself a
    // co-requisite repo, and it is the one the review's bypass named.
    //
    // Keyed on the FLAG OR the caller's own assertion, not on the flag alone. The two gates in
    // `create_model_download_job` do not fire on the same predicate: the id gate reads the entry's
    // flag, while the repo gate reads the repos the job will queue. For the shape the repo gate
    // exists to catch — an entry with NO flag whose download names a repo a flagged entry declares
    // — a flag-keyed stamp writes nothing, and the RETRY of that authorized download is then
    // refused by the repo gate over its own stored `repo`. `license_acknowledged` is the caller's
    // assertion carried verbatim, so the stamp records exactly what was asserted rather than
    // re-deriving it from a predicate that already disagreed once.
    if model_requires_license_acknowledgment(model) || license_acknowledged {
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
mod memory_calibration_job_payload_tests {
    use super::*;

    fn model(calibrated: bool) -> Value {
        let calibrations = if calibrated {
            json!([{ "binding": "fixture" }])
        } else {
            json!([])
        };
        json!({
            "id": "fixture-model",
            "name": "Fixture Model",
            "family": "fixture",
            "mlx": {
                "requiresConversion": true,
                "convertSourceRepo": "owner/source",
                "calibrations": calibrations
            }
        })
    }

    fn download() -> Value {
        json!({
            "provider": "huggingface",
            "repo": "owner/artifact",
            "files": ["weights.safetensors"]
        })
    }

    #[test]
    fn download_payload_derives_provenance_cost_only_from_the_catalog_primary_artifact() {
        let data = tempfile::tempdir().expect("data dir");
        let calibrated = model(true);
        let primary = build_model_download_job_payload(
            &calibrated,
            "fixture-model",
            &download(),
            None,
            true,
            false,
            data.path(),
        )
        .expect("primary payload");
        assert_eq!(
            primary.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
            Some(&Value::Bool(true))
        );

        let co_requisite = build_model_download_job_payload(
            &calibrated,
            "fixture-model",
            &download(),
            None,
            false,
            false,
            data.path(),
        )
        .expect("co-requisite payload");
        assert_eq!(
            co_requisite.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
            Some(&Value::Bool(false)),
            "co-requisites are not the artifact named by the calibration binding"
        );

        let client: ModelDownloadRequest = serde_json::from_value(json!({
            "memoryCalibrationProvenanceRequired": true
        }))
        .expect("typed request ignores unknown spoofed fields");
        let uncalibrated = build_model_download_job_payload(
            &model(false),
            "fixture-model",
            &download(),
            client.variant.as_deref(),
            true,
            false,
            data.path(),
        )
        .expect("uncalibrated payload");
        assert_eq!(
            uncalibrated.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
            Some(&Value::Bool(false)),
            "a client-supplied lookalike cannot enable the cold provenance cost"
        );
    }

    #[test]
    fn convert_payload_derives_provenance_cost_only_from_the_catalog() {
        let data = tempfile::tempdir().expect("data dir");
        let request: ModelConvertRequest = serde_json::from_value(json!({
            "quantizeBits": 4,
            "memoryCalibrationProvenanceRequired": false
        }))
        .expect("typed request ignores unknown spoofed fields");
        let calibrated = model(true);
        let calibrated_payload = build_model_convert_job_payload(ModelConvertJobPayload {
            model: &calibrated,
            model_id: "fixture-model",
            mlx: calibrated["mlx"].as_object().expect("mlx"),
            source_repo: "owner/source",
            output_dir: data.path(),
            quantize_only: false,
            request: &request,
        });
        assert_eq!(
            calibrated_payload.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
            Some(&Value::Bool(true)),
            "a client-supplied lookalike cannot disable catalog-authorized provenance"
        );

        let spoofed: ModelConvertRequest = serde_json::from_value(json!({
            "memoryCalibrationProvenanceRequired": true
        }))
        .expect("typed request ignores unknown spoofed fields");
        let uncalibrated = model(false);
        let uncalibrated_payload = build_model_convert_job_payload(ModelConvertJobPayload {
            model: &uncalibrated,
            model_id: "fixture-model",
            mlx: uncalibrated["mlx"].as_object().expect("mlx"),
            source_repo: "owner/source",
            output_dir: data.path(),
            quantize_only: false,
            request: &spoofed,
        });
        assert_eq!(
            uncalibrated_payload.get(MEMORY_CALIBRATION_PROVENANCE_REQUIRED),
            Some(&Value::Bool(false)),
            "a client-supplied lookalike cannot enable the cold provenance cost"
        );
    }
}

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

    let model = model_catalog(&state)
        .await?
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
            code: None,
        })?;
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
    Ok((StatusCode::CREATED, Json(job)))
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
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
            code: None,
        })?;
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
    let allowed_roots = vec![
        state.settings.data_dir.join("models"),
        huggingface_hub_cache_dir(&state.settings.data_dir),
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
    let warnings =
        catalog_delete_warnings(&state, "model", &model_id, None, Some(&catalogs)).await?;
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
            // `is_some_and` rather than `is_none_or`: the latter is stable only since 1.82 and the
            // workspace MSRV is 1.80 (`clippy::incompatible_msrv` is denied).
            path != &managed_dir && !repo_cache.as_ref().is_some_and(|cache| path == cache)
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
    let catalog = model_catalog(&state).await?;
    let model = catalog
        .into_iter()
        .find(|item| item.get("id").and_then(Value::as_str) == Some(model_id.as_str()))
        .ok_or_else(|| ApiError {
            status: StatusCode::NOT_FOUND,
            detail: "Model not found".to_owned(),
            code: None,
        })?;
    let data_dir = &state.settings.data_dir;
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
    // A tier lives in one of two storage shapes. Download-matrix models (`hasVariantMatrix`) keep the
    // tier as a `files`-filtered slice of a shared HF cache repo (sc-12024); convert-at-install
    // models (Anima) keep it as a real `<converted>/<tier>/` dir emitted by one convert job
    // (sc-12025). Resolve whichever this model uses; a variant that is neither has nothing to delete.
    let removal_result = if let Some(download) = model_download_for_variant(&model, &variant) {
        let repo = download
            .get("repo")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let files = string_array_field(&download, "files");
        // A tier with no `files` scope is the whole repo (a single-variant "default"), not a
        // deletable slice of a shared cache — refuse rather than risk wiping every tier. The UI
        // only offers this on real quant tiers (bf16/q8/q4), which always carry a `files` glob.
        if files.is_empty() {
            return Err(ApiError::bad_request(format!(
                "Tier '{variant}' has no file scope; delete the whole model instead"
            )));
        }
        let repo_cache = huggingface_repo_cache_path(data_dir, &repo);
        let managed_dir = Some(data_dir.join("models").join(safe_download_dir(&repo)));
        // Some families expose load-time quant choices over one dense snapshot (Mage-Flow):
        // their q4/q8/bf16 entries intentionally overlap. Protect every path still referenced
        // by a sibling logical tier; a delete then truthfully reclaims zero bytes rather than
        // corrupting the snapshot used by the remaining choices.
        let retained_files = model
            .get("downloads")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|entry| {
                !is_co_requisite_download(entry)
                    && entry.get("repo").and_then(Value::as_str) == Some(repo.as_str())
                    && entry
                        .get("variant")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.eq_ignore_ascii_case(&variant))
            })
            .flat_map(|entry| string_array_field(entry, "files"))
            .collect::<Vec<_>>();
        // Always permanent (skip the OS trash) — see the fn doc (sc-12088).
        remove_tier_artifacts(
            repo_cache,
            managed_dir,
            &files,
            &retained_files,
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
    Ok(Json(json!({
        "id": model_id,
        "variant": variant,
        "kind": "model-variant",
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
    let mut children: Vec<PathBuf> = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            children.push(entry.path());
        }
    } else {
        return;
    }
    for child in children {
        if child.is_dir() {
            Box::pin(remove_empty_dirs(&child)).await;
        }
    }
    let _ = tokio::fs::remove_dir(dir).await;
}

/// Prune a download-matrix repo cache dir after a tier delete: remove emptied snapshot/blob
/// subtrees, and — when no payload remains (no blobs, no snapshot files) — the whole repo cache
/// dir, so a fully-drained repo doesn't linger as a bare `refs/` skeleton. Best-effort.
async fn prune_empty_repo_cache(repo_cache: &FsPath) {
    remove_empty_dirs(&repo_cache.join("snapshots")).await;
    remove_empty_dirs(&repo_cache.join("blobs")).await;
    let has_blobs = std::fs::read_dir(repo_cache.join("blobs"))
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);
    let has_snapshot_files = huggingface_snapshot_dirs(repo_cache)
        .iter()
        .any(|snapshot| !snapshot_files(snapshot).is_empty());
    if !has_blobs && !has_snapshot_files {
        let _ = tokio::fs::remove_dir_all(repo_cache).await;
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

/// Kill-switch for the model upload/import endpoint (sc-7081, epic 7080). Enabled as of sc-14019
/// (epic 14015): a base-checkpoint import is now gated on the architecture-compatibility verdict of
/// the base-weight detector ([`sceneworks_core::base_weights`]) via [`import_source_supported`] at
/// queue time and the worker's `run_model_import_job` over the downloaded bytes. The gate accepts
/// ONLY `(family, component, quant)` triples with a real loader today (`import_supported` — currently
/// a dense-bf16 or descriptor-gated int8-per-row Krea 2 DiT, routed to the Krea engine via the S0d
/// family path); every other file
/// is refused with a typed reason (NEVER a silent fallback). Kept as a fn, not a `const`, so the
/// guarded handler body stays reachable (no `unreachable_code`) and the switch is trivially
/// re-flippable if a regression surfaces.
fn model_import_enabled() -> bool {
    true
}

const MODEL_IMPORT_DISABLED_DETAIL: &str = "Model import is temporarily disabled while native \
     model support and conversion are being built. (LoRA import is unaffected.)";

/// Import-compatibility gate for a base checkpoint (sc-14019, epic 14015): resolve the primary
/// weight file under `source`, classify it with the base-weight detector, and refuse the import
/// unless the verdict is one a real loader accepts today
/// ([`sceneworks_core::base_weights::import_supported`]). The refusal reason is surfaced to the API
/// client as a `400` so the upload flow gets a synchronous, actionable message rather than a queued
/// job that fails later. This gate is **additive** to path confinement — `source` has already been
/// confined by `validate_lora_import_source_path` (LAN-exposed jobs API, epic 4484); this never
/// widens or bypasses that. The worker re-runs the same predicate over the downloaded bytes so repo/
/// URL imports (whose file is not on disk at queue time) are covered there.
fn import_source_supported(source: &FsPath) -> Result<(), ApiError> {
    let weight_file = if source.is_dir() {
        first_safetensors_path(source)
    } else {
        Some(source.to_path_buf())
    };
    let Some(weight_file) = weight_file else {
        return Err(ApiError::bad_request(
            "No safetensors base-weight file was found to import; single-file base-checkpoint import expects a .safetensors transformer.",
        ));
    };
    let detection = detect_base_weight_file(&weight_file).map_err(model_family_inspection_error)?;
    import_detection_supported(&detection).map_err(ApiError::bad_request)
}

pub(crate) async fn create_model_import_job(
    State(state): State<AppState>,
    request: AxumRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), Response> {
    // sc-7081 (epic 7080): refuse before staging/queueing, covering both the JSON and
    // multipart entrypoints. The route stays mounted so a direct API client gets an
    // actionable 403 rather than a 404. See `model_import_enabled` for the rationale.
    if !model_import_enabled() {
        return Err(ApiError::forbidden(MODEL_IMPORT_DISABLED_DETAIL).into_response());
    }

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
        let result = queue_model_import_job(state, payload).await;
        if result.is_err() {
            cleanup_staged_model_upload(&staged_path).await;
        }
        return result.map_err(IntoResponse::into_response);
    }

    let payload = Json::<ModelImportRequest>::from_request(request, &state)
        .await
        .map(|Json(payload)| payload)
        .map_err(json_rejection_response)?;
    queue_model_import_job(state, payload)
        .await
        .map_err(IntoResponse::into_response)
}

pub(crate) async fn queue_model_import_job(
    state: AppState,
    mut payload: ModelImportRequest,
) -> Result<(StatusCode, Json<JobSnapshot>), ApiError> {
    if option_str_is_empty(payload.repo.as_deref())
        && option_str_is_empty(payload.source_url.as_deref())
        && option_str_is_empty(payload.source_path.as_deref())
    {
        return Err(ApiError::bad_request(
            "Provide a Hugging Face repo, source URL, or source path",
        ));
    }
    if let Some(source_url) = payload.source_url.as_deref() {
        validate_source_url(source_url)?;
    }
    if let Some(repo) = payload.repo.as_deref() {
        validate_huggingface_repo(repo)?;
    }
    // Licence acknowledgment, keyed on the repo the import will FETCH (sc-17227). This route had no
    // licence logic at all — `model_import_enabled()` hard-returns `true` and nothing below reads
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
    let model_type = match payload.model_type.as_deref().map(str::trim) {
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
        let models = model_catalog(&state).await?;
        payload.family = Some(validate_lora_family(&models, &family)?);
    }
    let name = payload
        .name
        .clone()
        .or_else(|| payload.repo.clone())
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
    let existing_ids = model_catalog(&state)
        .await?
        .into_iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect::<std::collections::HashSet<_>>();
    if existing_ids.contains(&model_id) {
        return Err(ApiError::bad_request(format!(
            "Model id '{model_id}' already exists. Pick a different id or delete the existing model first."
        )));
    }
    let target_name = safe_download_dir(&model_id);
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
            validate_lora_import_source_path(&source_path, &allowed_source_roots)?;
            // Compatibility gate (sc-14019): refuse, with the detector's typed
            // reason, any local checkpoint whose shape has no loader today.
            import_source_supported(FsPath::new(&source_path))?;
            let detected = detect_model_family(FsPath::new(&source_path))
                .map_err(model_family_inspection_error)?;
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
    let timestamp = now_rfc3339();
    let mut manifest_entry = json!({
        "id": model_id,
        "name": name,
        "type": model_type,
        "source": {
            "provider": model_import_source_provider(&payload),
            "repo": payload.repo.clone(),
            "path": source_path_rel,
        },
        "files": payload.files.clone(),
        "paths": {
            "model": target_dir.display().to_string(),
        },
        "createdAt": timestamp,
        "updatedAt": timestamp,
    });
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
        apply_model_manifest_defaults(object, &model_type, payload.family.as_deref());
    }
    let mut payload = to_json_object(&payload)?;
    payload.insert("modelId".to_owned(), manifest_entry["id"].clone());
    payload.insert("modelName".to_owned(), manifest_entry["name"].clone());
    payload.insert(
        "targetDir".to_owned(),
        Value::String(target_dir.display().to_string()),
    );
    payload.insert(
        "manifestPath".to_owned(),
        Value::String(manifest_path.display().to_string()),
    );
    payload.insert("manifestEntry".to_owned(), manifest_entry);
    let job = create_generation_job(
        state,
        JobType::ModelImport,
        None,
        None,
        payload,
        "auto".to_owned(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(job)))
}

pub(crate) async fn model_import_request_from_multipart(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<(ModelImportRequest, PathBuf), ApiError> {
    let mut payload = ModelImportRequest {
        model_id: None,
        name: None,
        model_type: None,
        repo: None,
        source_url: None,
        source_path: None,
        files: Vec::new(),
        family: None,
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
                "repo" => payload.repo = Some(value.to_owned()),
                "sourceUrl" => payload.source_url = Some(value.to_owned()),
                // The multipart form accepts `repo`/`sourceUrl` too, so it can reach a
                // licence-restricted repo exactly like the JSON body and needs the same way to
                // assert the acknowledgment (sc-17227). Anything other than "true" leaves it false
                // — the assertion is affirmative or it is not made.
                "licenseAcknowledged" => {
                    payload.license_acknowledged = value.eq_ignore_ascii_case("true")
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
    let upload_dir = state
        .settings
        .data_dir
        .join("cache")
        .join("model-uploads")
        .join(format!("upload-{}", Uuid::new_v4().simple()));
    tokio::fs::create_dir_all(&upload_dir)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let temp_path = upload_dir.join(filename);
    // sc-8886 (F-084): shared streaming writer. Cleanup removes the staged file AND its
    // per-upload parent directory.
    stream_multipart_field_to_file(
        field,
        &temp_path,
        max_model_upload_bytes(),
        || {
            format!(
                "Uploaded model file exceeds the {} limit",
                format_bytes(max_model_upload_bytes() as u64)
            )
        },
        || cleanup_staged_model_upload(&temp_path),
    )
    .await?;
    Ok(temp_path)
}

pub(crate) async fn cleanup_staged_model_upload(path: &FsPath) {
    let _ = tokio::fs::remove_file(path).await;
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::remove_dir(parent).await;
    }
}

pub(crate) fn model_import_source_provider(payload: &ModelImportRequest) -> &'static str {
    if payload.repo.is_some() {
        "huggingface"
    } else if payload.source_url.is_some() {
        "url"
    } else {
        "local"
    }
}

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

/// Catalog without live Hugging Face size estimation: download sizes fall back to
/// manifest metadata only. This is the right call for job validation, LoRA/preset
/// CRUD, download/convert job creation, and delete — none of which read the
/// byte-accurate download size — so an unreachable huggingface.co can't stall
/// those paths (sc-4169).
pub(crate) async fn model_catalog(state: &AppState) -> Result<Vec<Value>, ApiError> {
    Ok(model_catalog_snapshot(state).await?.as_ref().clone())
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
    for model in &mut models {
        let context = model_download_context(model)?;
        let live_estimate = context.as_ref().and_then(|context| {
            size_estimates
                .get(&(context.repo.clone(), context.files.clone()))
                .copied()
                .flatten()
        });
        apply_model_catalog_size_fields(model, context.as_ref(), live_estimate)?;
        apply_runtime_text_encoder_options(model, &state.settings.data_dir)?;
    }
    Ok(models)
}

/// Add runtime-only text-encoder choices to the public model catalog. The worker owns enumeration so
/// the API and generation resolver share one completeness predicate. This runs on the response clone,
/// outside the cached catalog snapshot: an operator who stages a complete alternate and refreshes
/// Models sees it without a server restart, while no repo/revision/download metadata is persisted.
fn apply_runtime_text_encoder_options(
    model: &mut Value,
    data_dir: &FsPath,
) -> Result<(), ApiError> {
    let adapter = model
        .get("adapter")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let options = sceneworks_worker::text_encoder_options_for_adapter(adapter, data_dir);
    set_runtime_text_encoder_options(model, options)
}

fn set_runtime_text_encoder_options(
    model: &mut Value,
    options: Vec<sceneworks_worker::TextEncoderOption>,
) -> Result<(), ApiError> {
    let object = model
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    if options.is_empty() {
        object.remove("textEncoderOptions");
    } else {
        object.insert("textEncoderOptions".to_owned(), json!(options));
    }
    Ok(())
}

#[cfg(test)]
mod runtime_text_encoder_option_tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use crate::tests::support::isolate_hf_cache;

    #[test]
    fn catalog_field_is_generic_and_contains_no_distribution_metadata() {
        let mut model = json!({ "id": "future_video", "adapter": "future_adapter" });
        set_runtime_text_encoder_options(
            &mut model,
            vec![
                sceneworks_worker::TextEncoderOption {
                    id: "default",
                    label: "Bundled encoder (default)",
                    description: "Uses the installed encoder.",
                    is_default: true,
                },
                sceneworks_worker::TextEncoderOption {
                    id: "future_staged_encoder",
                    label: "Staged alternate",
                    description: "Uses a complete operator-staged alternate.",
                    is_default: false,
                },
            ],
        )
        .unwrap();

        let options = model["textEncoderOptions"].as_array().unwrap();
        assert_eq!(options.len(), 2);
        assert_eq!(options[1]["id"], "future_staged_encoder");
        assert_eq!(options[1]["isDefault"], false);
        for forbidden in ["repo", "revision", "files", "download"] {
            assert!(
                options.iter().all(|option| option.get(forbidden).is_none()),
                "runtime options must not become distribution metadata: {forbidden}"
            );
        }
    }

    #[test]
    fn models_without_a_runtime_surface_emit_no_selector_field() {
        let mut model = json!({
            "id": "fixed_encoder_model",
            "textEncoderOptions": [{ "id": "stale" }]
        });
        set_runtime_text_encoder_options(&mut model, Vec::new()).unwrap();
        assert!(model.get("textEncoderOptions").is_none());
    }

    #[cfg(target_os = "macos")]
    fn write_tiny_safetensors(path: &FsPath) {
        let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.push(0);
        std::fs::write(path, bytes).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn live_options_refresh_when_alternate_becomes_valid_or_corrupt() {
        let _env = isolate_hf_cache();
        let data_dir = tempfile::tempdir().unwrap();
        let repo = sceneworks_core::hf_home::huggingface_repo_cache_path(
            data_dir.path(),
            "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit",
        )
        .unwrap();
        let snapshot = repo.join("snapshots").join("test-revision");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(
            snapshot.join("config.json"),
            br#"{"model_type":"gemma3_text"}"#,
        )
        .unwrap();
        std::fs::write(
            snapshot.join("tokenizer.json"),
            br#"{"model":{"type":"BPE"}}"#,
        )
        .unwrap();
        write_tiny_safetensors(&snapshot.join("model.safetensors"));

        let mut model = json!({ "id": "ltx_2_3", "adapter": "ltx_video" });
        apply_runtime_text_encoder_options(&mut model, data_dir.path()).unwrap();
        assert_eq!(model["textEncoderOptions"].as_array().unwrap().len(), 2);

        std::fs::write(snapshot.join("model.safetensors"), b"truncated").unwrap();
        apply_runtime_text_encoder_options(&mut model, data_dir.path()).unwrap();
        let options = model["textEncoderOptions"].as_array().unwrap();
        assert_eq!(options.len(), 1);
        assert_eq!(options[0]["id"], "default");

        std::fs::remove_dir_all(&repo).unwrap();
        apply_runtime_text_encoder_options(&mut model, data_dir.path()).unwrap();
        assert_eq!(model["textEncoderOptions"].as_array().unwrap().len(), 1);
    }
}

async fn model_catalog_snapshot(state: &AppState) -> Result<Arc<Vec<Value>>, ApiError> {
    {
        let cache_state = state.model_catalog_cache.state.lock();
        if let Some((snapshot_generation, snapshot)) = cache_state.snapshot.as_ref() {
            if *snapshot_generation == cache_state.generation {
                return Ok(snapshot.clone());
            }
        }
    }

    // Serialize only builders. The synchronous generation/snapshot state lock
    // is never held across the expensive filesystem sweep.
    let _build_guard = state.model_catalog_cache.build_serializer.lock().await;
    loop {
        let generation = {
            // Another builder may have populated the cache while this caller
            // waited for the async serializer.
            let cache_state = state.model_catalog_cache.state.lock();
            if let Some((snapshot_generation, snapshot)) = cache_state.snapshot.as_ref() {
                if *snapshot_generation == cache_state.generation {
                    return Ok(snapshot.clone());
                }
            }
            cache_state.generation
        };

        let snapshot = Arc::new(build_model_catalog_snapshot(state).await?);

        #[cfg(test)]
        state
            .model_catalog_cache
            .pause_before_publish_for_test()
            .await;

        let mut cache_state = state.model_catalog_cache.state.lock();
        if cache_state.generation == generation {
            cache_state.snapshot = Some((generation, snapshot.clone()));
            return Ok(snapshot);
        }
        drop(cache_state);
        // A model writer completed while the filesystem sweep was running.
        // Keep the async serializer and rebuild at the new generation. The
        // stale result was never published or returned.
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
            (!resolved.is_empty()).then(|| json!({
                "schemaVersion": 2, "repo": repo,
                "modelId": model.get("id").cloned().unwrap_or(Value::Null),
                "variant": entry.get("variant").cloned().unwrap_or_else(|| Value::String("default".to_owned())),
                "manifestFiles": files, "resolvedFiles": resolved, "backfilled": true,
            }))
        }).collect::<Vec<_>>();
    if receipts.is_empty() {
        return;
    }
    let _write_guard = RECEIPT_BACKFILL_WRITE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let _ = std::fs::create_dir_all(managed_path);
    let _ = serde_json::to_vec_pretty(&receipt).ok().and_then(|bytes| {
        std::fs::write(
            managed_path.join(".sceneworks-download-complete.json"),
            bytes,
        )
        .ok()
    });
}

#[cfg(test)]
mod import_gate_tests {
    //! The queue-time base-checkpoint compatibility gate (sc-14019, epic 14015): `import_source_supported`
    //! must accept a dense-bf16 or int8-per-row Krea 2 DiT and refuse unsupported triples with an
    //! actionable reason, so the LAN-exposed import endpoint can never queue an un-runnable file.
    use super::*;

    /// Write a safetensors file whose header declares `(name, dtype)` tensors. The declared tensor
    /// data must be present or the header is rejected as truncated (sc-6072), so the payload is padded
    /// to the declared offsets — mirrors `external_base_models::tests::write_safetensors`.
    fn write_safetensors(path: &FsPath, entries: &[(&str, &str)]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("parent dir");
        }
        let mut header = serde_json::Map::new();
        for (index, (key, dtype)) in entries.iter().enumerate() {
            let start = index * 4;
            header.insert(
                (*key).to_owned(),
                json!({ "dtype": dtype, "shape": [1], "data_offsets": [start, start + 4] }),
            );
        }
        let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header json");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend(std::iter::repeat(0_u8).take(entries.len() * 4));
        std::fs::write(path, bytes).expect("write safetensors");
    }

    /// A ComfyUI-native dense-bf16 Krea 2 DiT: the unique `txtfusion.` tower + BFL-style `blocks.*`
    /// keys, all BF16 → detector `(krea_2, Transformer, Bf16)` — one supported import triple.
    fn krea2_bf16_dit_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("model.diffusion_model.blocks.0.attn.wq.weight", "BF16"),
            ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
            (
                "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight",
                "BF16",
            ),
            ("model.diffusion_model.txtfusion.projector.weight", "BF16"),
        ]
    }

    #[test]
    fn krea2_bf16_upload_is_accepted() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("kreamania_variant5.safetensors");
        write_safetensors(&file, &krea2_bf16_dit_keys());
        assert!(
            import_source_supported(&file).is_ok(),
            "a dense-bf16 Krea 2 DiT upload must pass the import gate"
        );
    }

    #[test]
    fn krea2_int8_per_row_upload_is_accepted() {
        // Header detection deliberately identifies the convention from bulk I8 weights plus
        // `.comfy_quant`; the inference loader validates the actual descriptor JSON and scale shapes
        // before dequantization. This reduced fixture pins the queue-time classification/gate seam.
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("kreamania_variant4.safetensors");
        let mut names = Vec::new();
        for index in 0..6 {
            names.push((
                format!("model.diffusion_model.blocks.{index}.attn.wq.weight"),
                "I8",
            ));
            names.push((
                format!("model.diffusion_model.blocks.{index}.attn.wq.weight_scale"),
                "F32",
            ));
            names.push((
                format!("model.diffusion_model.blocks.{index}.attn.wq.comfy_quant"),
                "U8",
            ));
        }
        names.push(("model.diffusion_model.blocks.0.mod.lin".to_owned(), "BF16"));
        names.push((
            "model.diffusion_model.txtfusion.projector.weight".to_owned(),
            "F32",
        ));
        let entries: Vec<(&str, &str)> = names
            .iter()
            .map(|(name, dtype)| (name.as_str(), *dtype))
            .collect();
        write_safetensors(&file, &entries);

        assert!(
            import_source_supported(&file).is_ok(),
            "a Krea 2 int8-per-row DiT upload must pass the import gate"
        );
    }

    #[test]
    fn krea2_packed_quant_upload_is_refused_with_reason() {
        // Same family/component but `.comfy_quant` packed int8 → no loader yet → 400 with a reason.
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("kreamania_variant4.safetensors");
        write_safetensors(
            &file,
            &[
                ("model.diffusion_model.blocks.0.attn.wq.weight", "I8"),
                ("model.diffusion_model.blocks.0.attn.wq.comfy_quant", "U8"),
                ("model.diffusion_model.blocks.0.mod.lin", "BF16"),
                (
                    "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wq.comfy_quant",
                    "U8",
                ),
            ],
        );
        let error = import_source_supported(&file).expect_err("packed quant must be refused");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.detail.contains("bf16"), "detail: {}", error.detail);
    }

    #[test]
    fn unsupported_family_upload_is_refused() {
        // A qwen-image plain-fp8 DiT is recognized but has no import loader → refused.
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("qwen_image.safetensors");
        write_safetensors(
            &file,
            &[
                ("model.diffusion_model.img_in.weight", "F8_E4M3"),
                (
                    "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
                    "F8_E4M3",
                ),
                (
                    "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
                    "F8_E4M3",
                ),
            ],
        );
        let error = import_source_supported(&file).expect_err("qwen import must be refused");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            error.detail.contains("qwen-image"),
            "detail should name the unsupported family: {}",
            error.detail
        );
    }

    #[test]
    fn unrecognized_file_upload_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("mystery.safetensors");
        write_safetensors(
            &file,
            &[("some.mystery.tensor", "BF16"), ("another.mystery", "BF16")],
        );
        let error = import_source_supported(&file).expect_err("unrecognized file must be refused");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }
}

#[cfg(test)]
mod download_receipt_tests {
    use super::*;
    // Reuse the ONE crate-wide HF-cache guard (sc-13834): tests here seed/resolve the cache via the
    // env-first `huggingface_repo_cache_path`, so without this they would resolve into a developer's
    // real HF cache when HF_HOME is set. Serialize on the same `HF_ENV_LOCK`; never add a second lock.
    use crate::tests::support::isolate_hf_cache;

    fn builtin_models_entry(model_id: &str) -> Value {
        let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .map(|(_, contents)| *contents)
            .expect("builtin.models.jsonc present");
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
                .expect("builtin.models.jsonc parses");
        manifest["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
            .cloned()
            .unwrap_or_else(|| panic!("builtin entry {model_id} present"))
    }

    #[test]
    fn multi_repo_marker_filters_nested_receipts_by_requested_repo() {
        let temp = tempfile::tempdir().unwrap();
        let managed = temp.path().join("models/owner--primary");
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(managed.join(".sceneworks-download-complete.json"), serde_json::to_vec(&json!({
            "repo": "owner/corequisite",
            "receipts": [
                {"repo":"owner/primary", "resolvedFiles":["model.safetensors"], "snapshotRevision":"primary-rev"},
                {"repo":"owner/corequisite", "resolvedFiles":["encoder.safetensors"], "snapshotRevision":"dependency-rev"}
            ]
        })).unwrap()).unwrap();

        let primary = receipt_file_sets(&managed, "owner/primary", None);
        assert_eq!(
            primary,
            vec![ReceiptFileSet {
                files: vec!["model.safetensors".to_owned()],
                revision: Some("primary-rev".to_owned())
            }]
        );
        let dependency = receipt_file_sets(&managed, "owner/corequisite", None);
        assert_eq!(
            dependency,
            vec![ReceiptFileSet {
                files: vec!["encoder.safetensors".to_owned()],
                revision: Some("dependency-rev".to_owned())
            }]
        );
    }

    /// sc-14432: the RECEIPT lane was diffusers-only, so a flat no-`model_index` tier passed
    /// `snapshot_tier_is_loadable` unconditionally. A receipt recording exactly what the `<tier>/*` glob
    /// matched (the very shape the story cited as evidence of a bad re-host) therefore kept the model
    /// reading installed through the usable-stale path even once the cache-health lane had correctly
    /// demoted the tier. Both receipt halves — this one and `backfill_current_receipt` — now consult the
    /// shared family predicate.
    #[test]
    fn a_torn_flat_tier_is_not_counted_as_a_usable_stale_install() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "SceneWorks/sensenova-u1-8b-mlx";
        let model = json!({
            "id": "sensenova_u1_8b",
            "family": "sensenova-u1",
            "downloads": [{ "provider": "huggingface", "repo": repo, "variant": "q4", "files": ["q4/*"] }]
        });
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .unwrap()
            .join("snapshots/rev-a");
        let tier = snapshot.join("q4");
        std::fs::create_dir_all(&tier).unwrap();
        // The backbone landed; the tokenizer never did, and there is no sibling tier to borrow one from.
        std::fs::write(tier.join("model.safetensors"), b"weights").unwrap();
        std::fs::write(tier.join("config.json"), b"{}").unwrap();

        let marker_dir = data_dir.join("models").join(safe_download_dir(repo));
        let marker = marker_dir.join(".sceneworks-download-complete.json");

        // A pre-existing receipt (written before this tightening, or by a download that recorded exactly
        // what the glob matched) whose every recorded file DOES exist — the usable-stale route. The tier
        // still cannot load, so it must not resurrect the install.
        std::fs::create_dir_all(&marker_dir).unwrap();
        std::fs::write(
            &marker,
            serde_json::to_vec(&json!({
                "schemaVersion": 2, "repo": repo, "modelId": "sensenova_u1_8b", "variant": "q4",
                "manifestFiles": ["q4/*"],
                "resolvedFiles": ["q4/model.safetensors", "q4/config.json"],
            }))
            .unwrap(),
        )
        .unwrap();

        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            !state.installed,
            "a receipt whose files all exist must not keep a torn flat tier installed"
        );

        // Mutation check: the tokenizer arriving makes the tier genuinely loadable, and the SAME receipt
        // now legitimately counts — proving the assertion above discriminates on loadability, not on the
        // receipt merely being present.
        std::fs::write(tier.join("tokenizer.json"), b"{}").unwrap();
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(state.installed);
    }

    #[test]
    fn complete_pre_receipt_install_is_backfilled_and_protected_after_rename() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/backfill";
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .unwrap()
            .join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
        let original = json!({"id":"backfill-model", "downloads":[{"provider":"huggingface", "repo":repo, "files":["old.safetensors"]}]});

        let initial = install_state_for(
            model_download_context(&original).unwrap(),
            &original,
            data_dir,
        );
        assert!(initial.installed);
        let marker = data_dir
            .join("models")
            .join(safe_download_dir(repo))
            .join(".sceneworks-download-complete.json");
        let receipt: Value = serde_json::from_slice(&std::fs::read(marker).unwrap()).unwrap();
        assert_eq!(receipt["resolvedFiles"], json!(["old.safetensors"]));
        assert_eq!(receipt["backfilled"], true);

        let renamed = json!({"id":"backfill-model", "downloads":[{"provider":"huggingface", "repo":repo, "files":["new.safetensors"]}]});
        let protected = install_state_for(
            model_download_context(&renamed).unwrap(),
            &renamed,
            data_dir,
        );
        assert!(
            protected.installed,
            "backfilled exact old set remains usable"
        );
        assert!(protected.update_available, "rename is offered as an update");
    }

    #[test]
    fn shared_repo_backfill_preserves_each_models_receipt() {
        let _env = isolate_hf_cache();
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/shared";
        let snapshot = huggingface_repo_cache_path(data_dir, repo)
            .unwrap()
            .join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("base.safetensors"), b"base weights").unwrap();
        std::fs::write(snapshot.join("turbo.safetensors"), b"turbo weights").unwrap();
        let base = json!({
            "id": "shared-base",
            "downloads": [{
                "provider": "huggingface",
                "repo": repo,
                "files": ["base.safetensors"]
            }]
        });
        let turbo = json!({
            "id": "shared-turbo",
            "downloads": [{
                "provider": "huggingface",
                "repo": repo,
                "files": ["turbo.safetensors"]
            }]
        });

        assert!(
            install_state_for(model_download_context(&base).unwrap(), &base, data_dir).installed
        );
        assert!(
            install_state_for(model_download_context(&turbo).unwrap(), &turbo, data_dir).installed
        );

        let managed = data_dir.join("models").join(safe_download_dir(repo));
        let base_receipts = receipt_file_sets(&managed, repo, Some("shared-base"));
        let turbo_receipts = receipt_file_sets(&managed, repo, Some("shared-turbo"));
        assert_eq!(base_receipts[0].files, vec!["base.safetensors".to_owned()]);
        assert_eq!(
            turbo_receipts[0].files,
            vec!["turbo.safetensors".to_owned()]
        );
        assert_eq!(
            receipt_entries(&managed).len(),
            2,
            "backfill must merge shared-repo model receipts instead of making the first writer win"
        );
    }

    #[test]
    fn receipt_remains_usable_when_current_manifest_file_changes() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/model";
        let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
        let snapshot = cache.join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
        let managed = data_dir.join("models/owner--model");
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "schemaVersion": 2, "repo": repo,
                "resolvedFiles": ["old.safetensors"]
            }))
            .unwrap(),
        )
        .unwrap();

        let files = receipt_file_sets(&managed, repo, None);
        assert_eq!(files[0].files, vec!["old.safetensors".to_owned()]);
        assert!(receipt_files_present(data_dir, repo, &files[0], None));
        assert!(!huggingface_cache_health(&cache, &["new.safetensors".to_owned()]).installed);

        let ambiguous = cache.join("snapshots/rev-b");
        std::fs::create_dir_all(&ambiguous).unwrap();
        std::fs::write(ambiguous.join("old.safetensors"), b"other weights").unwrap();
        assert!(
            !receipt_files_present(data_dir, repo, &files[0], None),
            "legacy receipt must identify one snapshot"
        );

        std::fs::remove_file(snapshot.join("old.safetensors")).unwrap();
        std::fs::remove_file(ambiguous.join("old.safetensors")).unwrap();
        assert!(
            !receipt_files_present(data_dir, repo, &files[0], None),
            "torn stale install is missing"
        );
    }

    #[test]
    fn catalog_distinguishes_usable_stale_from_torn_and_points_at_cache() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        let repo = "owner/model";
        let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
        let snapshot = cache.join("snapshots/rev-a");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "schemaVersion": 2, "repo": repo, "resolvedFiles": ["old.safetensors"]
            }))
            .unwrap(),
        )
        .unwrap();
        let model = json!({"id":"model", "downloads":[{
            "provider":"huggingface", "repo":repo, "files":["new.safetensors"]
        }]});
        let context = model_download_context(&model).unwrap().unwrap();
        let stale = install_state_for(Some(context), &model, data_dir);
        assert!(stale.installed);
        assert!(stale.update_available);
        assert_eq!(
            stale.installed_path.as_deref(),
            Some(cache.to_string_lossy().as_ref())
        );

        std::fs::remove_file(snapshot.join("old.safetensors")).unwrap();
        let torn = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(!torn.installed);
        assert!(!torn.update_available);
    }

    #[test]
    fn breaking_and_corequisite_softness_matrix() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        for breaking in [false, true] {
            for soft in [false, true] {
                let temp = tempfile::tempdir().unwrap();
                let data_dir = temp.path();
                let repo = "owner/model";
                let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
                let snapshot = cache.join("snapshots/rev-a");
                std::fs::create_dir_all(&snapshot).unwrap();
                std::fs::write(snapshot.join("old.safetensors"), b"weights").unwrap();
                let managed = data_dir.join("models").join(safe_download_dir(repo));
                std::fs::create_dir_all(&managed).unwrap();
                std::fs::write(
                    managed.join(".sceneworks-download-complete.json"),
                    serde_json::to_vec(&json!({
                        "schemaVersion": 2, "repo": repo,
                        "resolvedFiles": ["old.safetensors"]
                    }))
                    .unwrap(),
                )
                .unwrap();
                let model = json!({
                    "id": "model",
                    "downloads": [
                        {"provider":"huggingface", "repo":repo,
                         "files":["new.safetensors"], "breaking":breaking},
                        {"provider":"huggingface", "repo":"owner/dependency",
                         "coRequisite":true, "required": if soft { "soft" } else { "hard" }}
                    ]
                });
                let state =
                    install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
                assert_eq!(
                    state.installed,
                    !breaking && soft,
                    "breaking={breaking}, soft={soft}"
                );
                assert!(
                    state.update_available,
                    "every stale/soft combination offers an update"
                );
                if !soft {
                    assert!(state
                        .missing_required_files
                        .iter()
                        .any(|file| file.contains("owner/dependency")));
                }
                if !breaking && !soft {
                    let mut omitted = model.clone();
                    omitted.as_object_mut().unwrap()["downloads"][0]
                        .as_object_mut()
                        .unwrap()
                        .remove("breaking");
                    omitted.as_object_mut().unwrap()["downloads"][1]
                        .as_object_mut()
                        .unwrap()
                        .remove("required");
                    let defaulted = install_state_for(
                        model_download_context(&omitted).unwrap(),
                        &omitted,
                        data_dir,
                    );
                    assert!(!defaulted.installed, "omitted required defaults to hard");
                    assert!(
                        defaulted.update_available,
                        "omitted breaking defaults to false"
                    );
                }
            }
        }
    }

    /// sc-13680: the AUDIO lane's install-state now accounts for chatterbox_tts's two HARD
    /// co-requisites (the `perth` + `voice_embedding` companion weights). Previously no audio model
    /// used a co-requisite, so this generic gate (shared with PiD-gemma / Wan-Lightning) was untested
    /// on the audio lane. A present primary snapshot with the perth/ve co-requisites ABSENT must NOT
    /// report installed (it must surface as a repairable partial), and staging both must flip it to
    /// installed — proving the perth+VE rehoming is enforced end to end.
    #[test]
    fn chatterbox_tts_install_state_gates_on_the_perth_and_ve_co_requisites() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path();
        // The primary chatterbox snapshot (T3 + S3Gen + tokenizer) is present…
        let primary_repo = "ResembleAI/chatterbox";
        let primary_snapshot = huggingface_repo_cache_path(data_dir, primary_repo)
            .unwrap()
            .join("snapshots/main");
        std::fs::create_dir_all(&primary_snapshot).unwrap();
        for file in ["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"] {
            std::fs::write(primary_snapshot.join(file), b"weights").unwrap();
        }

        // …the chatterbox_tts catalog shape: primary + the two hard co-requisites (ve + perth),
        // mirroring builtin.models.jsonc (componentId is inert to install-state, which gates on
        // repo/files presence).
        let model = json!({
            "id": "chatterbox_tts",
            "downloads": [
                { "provider": "huggingface", "repo": primary_repo,
                  "files": ["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"] },
                { "provider": "huggingface", "repo": "ResembleAI/chatterbox",
                  "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", "coRequisite": true,
                  "componentId": "voice_embedding", "files": ["ve.safetensors"] },
                { "provider": "huggingface", "repo": "SceneWorks/perth-implicit",
                  "revision": "80b60f9caead09b8d3b512bda0b24038f28c08ec", "coRequisite": true,
                  "componentId": "perth", "files": ["perth_implicit.safetensors"] }
            ]
        });

        // Co-requisites absent → NOT installed; both missing repos surface for repair.
        let missing = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            !missing.installed,
            "a present primary with the perth+ve co-requisites absent must not report installed"
        );
        assert!(
            missing.cache_incomplete,
            "the missing hard co-requisites make this a repairable partial install"
        );
        assert!(
            missing
                .missing_required_files
                .iter()
                .any(|entry| entry.contains("SceneWorks/perth-implicit"))
                && missing
                    .missing_required_files
                    .iter()
                    .any(|entry| entry.contains("ResembleAI/chatterbox")),
            "both missing co-requisite repos must be reported, got {:?}",
            missing.missing_required_files
        );

        // Stage BOTH co-requisites at their pinned snapshots → install-state flips to installed.
        let ve = huggingface_repo_cache_path(data_dir, "ResembleAI/chatterbox")
            .unwrap()
            .join("snapshots/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18");
        std::fs::create_dir_all(&ve).unwrap();
        std::fs::write(ve.join("ve.safetensors"), b"weights").unwrap();
        let perth = huggingface_repo_cache_path(data_dir, "SceneWorks/perth-implicit")
            .unwrap()
            .join("snapshots/80b60f9caead09b8d3b512bda0b24038f28c08ec");
        std::fs::create_dir_all(&perth).unwrap();
        std::fs::write(perth.join("perth_implicit.safetensors"), b"weights").unwrap();

        let installed =
            install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            installed.installed,
            "primary + both hard co-requisites present must report installed"
        );
        assert!(
            installed.missing_required_files.is_empty(),
            "nothing is missing once the perth+ve co-requisites are staged, got {:?}",
            installed.missing_required_files
        );
    }

    /// sc-13684: MMAudio is a PURE 5-component assembly, so each tier catalogs a non-coRequisite `dit`
    /// primary anchor plus FIVE hard component coRequisites (clip/synchformer/dit/vae/vocoder). Because
    /// install-state gates on the hard coRequisites, an mmaudio tier stays not-installed until ALL five
    /// component snapshots are present — even with the primary anchor on disk. Binds to the LIVE
    /// builtin manifest so a dropped/renamed component coRequisite (or a lost pinned revision) fails here.
    #[test]
    fn mmaudio_install_state_gates_on_all_five_component_co_requisites() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        for model_id in ["mmaudio_small_16k", "mmaudio_large_44k"] {
            let temp = tempfile::tempdir().unwrap();
            let data_dir = temp.path();
            let model = builtin_models_entry(model_id);

            // Every component is a hard coRequisite, and mmaudio catalogs FIVE of them.
            let co_requisites = model_co_requisite_downloads(&model);
            assert_eq!(
                co_requisites.len(),
                5,
                "{model_id}: MMAudio must catalog all five component coRequisites"
            );

            // Stage ONLY the primary anchor (the `dit` file) — no component coRequisite yet.
            let context = model_download_context(&model).unwrap().unwrap();
            let anchor_snapshot = huggingface_repo_cache_path(data_dir, &context.repo)
                .unwrap()
                .join("snapshots/eb13a1a98fdbec91753775c57b074ccdfc60587c");
            std::fs::create_dir_all(&anchor_snapshot).unwrap();
            for file in &context.files {
                let path = anchor_snapshot.join(file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"weights").unwrap();
            }

            let missing = install_state_for(Some(context.clone()), &model, data_dir);
            assert!(
                !missing.installed,
                "{model_id}: a present anchor with the five components absent must not report installed"
            );
            assert!(
                missing.cache_incomplete,
                "{model_id}: missing hard component coRequisites make this a repairable partial install"
            );

            // Stage every component coRequisite at its pinned snapshot → install-state flips to installed.
            for co_requisite in &co_requisites {
                let repo = co_requisite.get("repo").and_then(Value::as_str).unwrap();
                let revision = co_requisite
                    .get("revision")
                    .and_then(Value::as_str)
                    .unwrap();
                let snapshot = huggingface_repo_cache_path(data_dir, repo)
                    .unwrap()
                    .join("snapshots")
                    .join(revision);
                std::fs::create_dir_all(&snapshot).unwrap();
                for file in string_array_field(co_requisite, "files") {
                    let path = snapshot.join(&file);
                    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                    std::fs::write(path, b"weights").unwrap();
                }
            }

            let installed = install_state_for(Some(context), &model, data_dir);
            assert!(
                installed.installed,
                "{model_id}: anchor + all five component coRequisites present must report installed"
            );
            assert!(
                installed.missing_required_files.is_empty(),
                "{model_id}: nothing is missing once every component is staged, got {:?}",
                installed.missing_required_files
            );
        }
    }

    /// sc-13681 / sc-13686: each MOSS TTS entry advertises its RVQ codec as a HARD component
    /// coRequisite (`moss_ttsd_v05` → XY_Tokenizer, `moss_tts_realtime` → MOSS-Audio-Tokenizer). Because
    /// install-state gates on hard coRequisites, a MOSS entry stays not-installed until the codec
    /// snapshot is present — even with its primary AR snapshot on disk — so the Model Manager reports a
    /// repairable partial install and names the missing codec repo, then flips to installed once the
    /// codec is staged. Binds to the LIVE builtin manifest so a dropped/renamed codec coRequisite or a
    /// lost pinned revision fails HERE, not silently at synth time. The negative path's twin at the
    /// worker's `resolve_co_requisites` seam (model_jobs::…a_missing_moss_codec_fails…) proves the JOB
    /// then fails at load with the actionable error; this proves the catalog never advertises it ready.
    #[test]
    fn moss_tts_install_state_gates_on_the_codec_co_requisite() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        for model_id in ["moss_ttsd_v05", "moss_tts_realtime"] {
            let temp = tempfile::tempdir().unwrap();
            let data_dir = temp.path();
            let model = builtin_models_entry(model_id);

            // Exactly one hard codec coRequisite, keyed on the `codec` component id.
            let co_requisites = model_co_requisite_downloads(&model);
            assert_eq!(
                co_requisites.len(),
                1,
                "{model_id}: MOSS advertises exactly the codec coRequisite"
            );
            let codec = &co_requisites[0];
            assert_eq!(
                codec.get("componentId").and_then(Value::as_str),
                Some("codec"),
                "{model_id}: the coRequisite is the `codec` component"
            );
            let codec_repo = codec.get("repo").and_then(Value::as_str).unwrap();

            // Stage the primary AR snapshot (the manifest's non-coRequisite download) — codec absent.
            let context = model_download_context(&model).unwrap().unwrap();
            let primary = model
                .get("downloads")
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .find(|entry| {
                    is_supported_model_download(entry) && !is_co_requisite_download(entry)
                })
                .expect("MOSS entry has a primary download");
            let primary_rev = primary.get("revision").and_then(Value::as_str).unwrap();
            let primary_snapshot = huggingface_repo_cache_path(data_dir, &context.repo)
                .unwrap()
                .join("snapshots")
                .join(primary_rev);
            std::fs::create_dir_all(&primary_snapshot).unwrap();
            for file in &context.files {
                let path = primary_snapshot.join(file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"weights").unwrap();
            }

            let missing = install_state_for(Some(context.clone()), &model, data_dir);
            assert!(
                !missing.installed,
                "{model_id}: primary present but codec absent must not report installed"
            );
            assert!(
                missing.cache_incomplete,
                "{model_id}: the missing hard codec makes this a repairable partial install"
            );
            assert!(
                missing
                    .missing_required_files
                    .iter()
                    .any(|entry| entry.contains(codec_repo)),
                "{model_id}: the missing codec repo must be reported, got {:?}",
                missing.missing_required_files
            );

            // Stage the codec snapshot at its pinned revision → install-state flips to installed.
            let codec_rev = codec.get("revision").and_then(Value::as_str).unwrap();
            let codec_snapshot = huggingface_repo_cache_path(data_dir, codec_repo)
                .unwrap()
                .join("snapshots")
                .join(codec_rev);
            std::fs::create_dir_all(&codec_snapshot).unwrap();
            for file in string_array_field(codec, "files") {
                let path = codec_snapshot.join(&file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"weights").unwrap();
            }

            let installed = install_state_for(Some(context), &model, data_dir);
            assert!(
                installed.installed,
                "{model_id}: primary + codec present must report installed"
            );
            assert!(
                installed.missing_required_files.is_empty(),
                "{model_id}: nothing missing once the codec is staged, got {:?}",
                installed.missing_required_files
            );
        }
    }

    /// sc-13682: the three SDXL shared components (CLIP-L/bigG tokenizers + fp16-fix VAE) are declared as
    /// candle-only (`platforms: [windows, linux]`) hard co-requisites on every candle-SDXL base +
    /// InstantID. On macOS `retain_downloads_for_os` strips them, so the self-contained MLX turnkey's
    /// install state does NOT gate on them; on the candle OSes all three are retained and gate the entry.
    /// Binds to the LIVE builtin manifest so a lost platform tag or a dropped component fails here.
    #[test]
    fn sdxl_shared_components_are_candle_only_and_gate_install_state_off_macos() {
        use std::collections::BTreeSet;

        let want: BTreeSet<String> = ["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"]
            .iter()
            .map(|id| (*id).to_owned())
            .collect();

        for model_id in [
            "sdxl",
            "realvisxl",
            "realvisxl_lightning",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
            "instantid_realvisxl",
        ] {
            let entry = builtin_models_entry(model_id);

            // macOS: the candle-only components are filtered out, so the MLX turnkey gates on NO coReq.
            let mut macos = entry.clone();
            retain_downloads_for_os(&mut macos, "macos");
            assert!(
                model_co_requisite_downloads(&macos).is_empty(),
                "{model_id}: the SDXL CLIP/VAE components must not gate the macOS MLX turnkey install state",
            );

            // Windows + Linux: all three components are retained and gate the candle entry.
            for os in ["windows", "linux"] {
                let mut candle = entry.clone();
                retain_downloads_for_os(&mut candle, os);
                let ids: BTreeSet<String> = model_co_requisite_downloads(&candle)
                    .iter()
                    .filter_map(|download| {
                        download
                            .get("componentId")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .collect();
                assert_eq!(
                    ids, want,
                    "{model_id} on {os}: the candle SDXL entry must gate on all three shared components",
                );
            }
        }
    }

    /// The live SCAIL-2 manifest must give off-Mac users the dense shared package and the catalog's
    /// install badge must enforce the exact provider-required six-file layout. This binds product
    /// advertisement, Model Manager filtering, and worker loadability without duplicating weights.
    #[test]
    fn scail2_shared_bf16_package_is_installable_off_macos_and_fails_closed() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let original = builtin_models_entry("scail2_14b");

        for os in ["windows", "linux"] {
            let mut model = original.clone();
            retain_downloads_for_os(&mut model, os);
            let downloads = model["downloads"].as_array().unwrap();
            assert_eq!(downloads.len(), 1, "{os} must expose one installable tier");
            assert_eq!(downloads[0]["variant"], "bf16");
            assert_eq!(downloads[0]["files"], json!(["bf16/*"]));
            assert_eq!(model_download(&model).unwrap()["variant"], "bf16");
        }

        let mut model = original;
        retain_downloads_for_os(&mut model, "windows");
        let download = model_download(&model).unwrap();
        let repo = download["repo"].as_str().unwrap();
        let revision = download["revision"].as_str().unwrap();
        let tier = huggingface_repo_cache_path(data.path(), repo)
            .unwrap()
            .join("snapshots")
            .join(revision)
            .join("bf16");
        std::fs::create_dir_all(&tier).unwrap();

        std::fs::write(tier.join("dit.safetensors"), b"").unwrap();
        let partial =
            install_state_for(model_download_context(&model).unwrap(), &model, data.path());
        assert!(!partial.installed);
        assert!(partial.cache_incomplete);
        assert!(
            partial
                .missing_required_files
                .iter()
                .any(|file| file.contains("bf16") && file.contains("incomplete")),
            "got {:?}",
            partial.missing_required_files
        );

        for file in sceneworks_core::mlx_tier_completeness::SCAIL2_TIER_FILES {
            std::fs::write(tier.join(file), b"").unwrap();
        }
        let installed =
            install_state_for(model_download_context(&model).unwrap(), &model, data.path());
        assert!(installed.installed);
        assert!(!installed.cache_incomplete);

        std::fs::remove_file(tier.join("t5_encoder.safetensors")).unwrap();
        let torn = install_state_for(model_download_context(&model).unwrap(), &model, data.path());
        assert!(!torn.installed, "a provider-required file was removed");
        assert!(torn.cache_incomplete);

        let description = model["ui"]["description"].as_str().unwrap();
        assert!(description.contains("Candle on NVIDIA Windows/Linux"));
        assert!(!description.contains("macOS native MLX only"));
    }
}

// Resolve a model's install/cache state from its (optional) download source. A
// downloadable model checks the HF cache + the SceneWorks-managed dir; a non-download
// model (a local manifest entry) checks its declared installed path; otherwise it's
// simply absent.
fn install_state_for(
    download_context: Option<DownloadContext>,
    model: &Value,
    data_dir: &FsPath,
) -> ModelCatalogEntryState {
    if let Some(download_context) = download_context {
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
        let (cache_installed, cache_incomplete, mut missing_required_files) =
            if model_has_variant_matrix(model) {
                let variants = model_variant_states(model, data_dir);
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
                let cache_health = cache_path
                    .as_ref()
                    .map(|path| huggingface_cache_health(path, &download_context.files));
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
        let receipt_file_sets = receipt_file_sets(
            &managed_path,
            &download_context.repo,
            model.get("id").and_then(Value::as_str),
        );
        let managed_installed = !model_has_variant_matrix(model)
            && receipt_file_sets.is_empty()
            && model_is_installed(&managed_path);
        if cache_installed {
            backfill_current_receipt(&managed_path, model, &download_context, data_dir);
        }
        let stale_files_present = !cache_installed
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
        let usable_stale = stale_files_present && !breaking_update;
        let primary_installed = managed_installed || cache_installed || usable_stale;
        let installed_path = if cache_installed || cache_incomplete || usable_stale {
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
            .collect();
        if !tier_scoped.is_empty() {
            let tiers: std::collections::BTreeSet<String> = tier_scoped
                .iter()
                .filter_map(co_requisite_variant)
                .collect();
            let any_tier_complete = tiers.iter().any(|tier| {
                tier_scoped
                    .iter()
                    .filter(|download| co_requisite_variant(download).as_deref() == Some(tier))
                    .all(|download| {
                        download
                            .get("repo")
                            .and_then(Value::as_str)
                            .and_then(|repo| huggingface_repo_cache_path(data_dir, repo))
                            .map(|path| {
                                huggingface_cache_health(
                                    &path,
                                    &string_array_field(download, "files"),
                                )
                            })
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
        for co_requisite in model_co_requisite_downloads(model)
            .into_iter()
            .filter(|download| co_requisite_variant(download).is_none())
        {
            let Some(repo) = co_requisite.get("repo").and_then(Value::as_str) else {
                continue;
            };
            let files = string_array_field(&co_requisite, "files");
            let health = huggingface_repo_cache_path(data_dir, repo)
                .map(|path| huggingface_cache_health(&path, &files));
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
        }
    } else if let Some(installed_path) = model_manifest_installed_path(model, data_dir) {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: Some(installed_path.display().to_string()),
            installed: model_is_installed(&installed_path),
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
        }
    } else {
        ModelCatalogEntryState {
            downloadable: false,
            installed_path: None,
            installed: false,
            cache_incomplete: false,
            missing_required_files: Vec::new(),
            update_available: false,
        }
    }
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
    /// Files this tier is missing from the cache (empty when complete or absent).
    missing_required_files: Vec<String>,
    /// This tier's estimated download size (from `downloads[].estimatedSizeBytes` /
    /// `footprint.diskSizeBytes`).
    download_size_bytes: Option<u64>,
    /// The raw `downloads[].footprint` object (disk size + optional measured memory), passed
    /// through verbatim for the RAM-suggestion surfaces (sc-8509/8516). `Null` when absent.
    footprint: Value,
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
        // sc-19078: the MiniMax-H3 tiers ship two DiT partition dirs (`{tier}/transformer` and
        // `{tier}/transformer_ref`) and NO `model_index.json` at either level, so the coarse
        // `q4/transformer/*` glob is satisfied by a single landed file out of fourteen shards. Like
        // SenseNova the id — not the family — picks the predicate: the two catalog entries share the
        // `minimax-h3` family but own DIFFERENT partitions of one repo, so a family-only predicate
        // would have to demand both and report a reference-only install as torn forever.
        "minimax-h3" => tc::minimax_h3_tier_predicate(model_id),
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
            let cache_health = cache_path
                .as_ref()
                .map(|path| huggingface_cache_health(path, &files));
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
            let managed_installed = managed_tier_installed(&managed_path, &files);
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

            let installed_path = if cache_installed || cache_incomplete {
                cache_path
            } else if managed_installed {
                Some(managed_path)
            } else {
                None
            };
            ModelVariantState {
                variant: entry
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_owned())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_else(|| "default".to_owned()),
                installed,
                installed_path: installed_path.map(|path| path.display().to_string()),
                cache_incomplete,
                missing_required_files,
                download_size_bytes: manifest_download_size_bytes(model, entry)
                    .or_else(|| variant_footprint_disk_bytes(entry)),
                footprint: entry.get("footprint").cloned().unwrap_or(Value::Null),
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
            json!({
                "variant": variant.variant,
                "installed": variant.installed,
                "installState": if variant.installed { "installed" } else { "missing" },
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
            })
        })
        .collect::<Vec<_>>();
    object.insert("hasVariantMatrix".to_owned(), Value::Bool(has_matrix));
    object.insert("variants".to_owned(), Value::Array(variants));
}

fn refresh_variant_download_sizes(model: &mut Value) -> Result<(), ApiError> {
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
    let mlx_lane = cfg!(target_os = "macos");
    apply_imported_lora_advertisement_for_lanes(object, mlx_lane, !mlx_lane);
}

fn apply_imported_lora_advertisement_for_lanes(
    object: &mut JsonObject,
    mlx_lane_available: bool,
    candle_lane_available: bool,
) {
    if object.get("type").and_then(Value::as_str) != Some("image") {
        return;
    }
    let Some(id) = object
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|id| !id.is_empty())
    else {
        return;
    };
    let Some(family) = object
        .get("family")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|family| !family.is_empty())
        .map(str::to_owned)
    else {
        return;
    };
    let serves_loras = sceneworks_core::jobs_store::imported_image_model_lora_advertisement(
        &id,
        &family,
        mlx_lane_available,
        candle_lane_available,
    );
    if serves_loras != Some(false) {
        return;
    }
    let compatibility = object
        .entry("loraCompatibility".to_owned())
        .or_insert_with(|| json!({}));
    let Some(compatibility) = compatibility.as_object_mut() else {
        return;
    };
    // Preserve every other key (`types` drives the multi-phase surface); only the families
    // promise is withdrawn.
    compatibility.insert("families".to_owned(), Value::Array(Vec::new()));
    compatibility.insert("supported".to_owned(), Value::Bool(false));
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
        // Forward the catalog-declared family so an imported/user model whose id is in no routing
        // table still routes to its family's MLX engine (route-by-family, sc-14019) instead of
        // reporting "not available on Mac". Builtin ids are unaffected (they route by id).
        let family = object
            .get("family")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
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
        model_candle_support(id, model_type)
    };
    if let Ok(candle_support) = serde_json::to_value(candle_support) {
        object.insert("candleSupport".to_owned(), candle_support);
    }
    let mlx_status = if cfg!(target_os = "macos") {
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

#[cfg(test)]
static TEST_CATALOG_PROBES_ACTIVE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_CATALOG_PROBES_PEAK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn test_reset_catalog_probe_concurrency() {
    use std::sync::atomic::Ordering;
    // DRAIN, do not clobber. Delayed probes run via spawn_blocking on pool OS threads
    // that cannot be cancelled mid-sleep, and the probe tests run concurrently in one
    // binary against these process-global atomics (their sleeps are bounded: <=250ms).
    // Storing 0 into ACTIVE while a concurrent test's probe was mid-sleep made that
    // probe's later decrement wrap to usize::MAX, and the next probe's `+ 1` panicked
    // with "attempt to add with overflow" — a cross-test 500 that only surfaced on the
    // slower hosted macos-26 runners once the workspace suite moved there (sc-17723).
    // Waiting live probes out keeps ACTIVE's +1/-1 pairing intact, so it can never go
    // below zero; only PEAK is forced. The probe tests additionally serialize behind
    // catalog_probe_test_lock() in tests/catalog.rs, which also keeps a neighbor's
    // probes from clobbering a PEAK measurement — this drain is the belt-and-braces
    // for any future probe user outside that lock.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while TEST_CATALOG_PROBES_ACTIVE.load(Ordering::SeqCst) != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "catalog probe stragglers did not drain within 10s — a probe thread is leaking"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    TEST_CATALOG_PROBES_PEAK.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn test_catalog_probe_peak() -> usize {
    TEST_CATALOG_PROBES_PEAK.load(std::sync::atomic::Ordering::SeqCst)
}

#[cfg(test)]
pub(crate) fn test_catalog_probe_limit() -> usize {
    MODEL_CATALOG_PROBE_CONCURRENCY
}

#[cfg(test)]
fn test_delay_catalog_probe(model: &Value) {
    use std::sync::atomic::Ordering;

    let Some(delay_ms) = model
        .get("__testCatalogProbeDelayMs")
        .and_then(Value::as_u64)
    else {
        return;
    };
    // saturating_add / checked_sub: with the draining reset above, ACTIVE can no longer
    // underflow — but these keep any future reset-style edit from reintroducing the
    // wrap-then-panic class (a straggler now parks at 0 instead of poisoning every later
    // sample with usize::MAX).
    let active = TEST_CATALOG_PROBES_ACTIVE
        .fetch_add(1, Ordering::SeqCst)
        .saturating_add(1);
    TEST_CATALOG_PROBES_PEAK.fetch_max(active, Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(delay_ms));
    let _ = TEST_CATALOG_PROBES_ACTIVE.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
        count.checked_sub(1)
    });
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
) -> Result<Value, ApiError> {
    #[cfg(test)]
    test_delay_catalog_probe(&model);
    let state = install_state_for(download_context, &model, data_dir);
    let object = model
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("Model manifest entry must be an object"))?;
    let model_id = object.get("id").and_then(Value::as_str).unwrap_or_default();
    let user_managed = user_model_ids.contains(model_id);
    object.insert(
        "catalogScope".to_owned(),
        Value::String(if user_managed { "user" } else { "builtin" }.to_owned()),
    );
    object.insert("downloadable".to_owned(), Value::Bool(state.downloadable));
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
            if state.cache_incomplete {
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
        Value::Bool(state.downloadable && state.cache_incomplete),
    );
    object.insert(
        "updateAvailable".to_owned(),
        Value::Bool(state.update_available),
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
        Value::Bool(user_managed || state.installed),
    );
    // Per-variant install tracking (sc-8508, epic 8506): one entry per declared quant tier,
    // each with its own installed flag + path + size + footprint. A single-variant model
    // still emits exactly one "default" variant, so the array is a superset of the
    // (retained) top-level installState/installedPath fields — nothing existing regresses.
    apply_variant_fields(object, data_dir);
    apply_gating_fields(object);
    apply_mac_and_mlx_fields(object, data_dir);
    apply_imported_lora_advertisement(object);
    // Live denoise preview support (sc-16965, epic 16948): `preview.byBackend`, read from the
    // generated `config/manifests/builtin.preview-support.jsonc` rather than from a registry, because
    // THIS process may link no engines at all (docker/rust.Dockerfile builds the API without
    // `backend-candle`, so `Registry::new()` here is empty and a serve-time derivation would report
    // "nothing supports preview" on every server). Engine-KEYED on purpose — the flag genuinely
    // diverges by backend and never fully collapses. Additive: a model the generated table does not
    // know gets no `preview` key, which the UI reads as "unknown" and renders exactly as before.
    sceneworks_core::preview_support::apply_to_model_entry(object);
    Ok(model)
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
mod model_size_concurrency_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn bounded_collection_caps_concurrency_and_preserves_order() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let output = collect_bounded_ordered(0..12, 3, |value| {
            let active = active.clone();
            let peak = peak.clone();
            async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                value
            }
        })
        .await;

        assert_eq!(output, (0..12).collect::<Vec<_>>());
        assert_eq!(peak.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn builtin_download_contexts_fit_in_size_cache() {
        let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
        let manifest: Value = serde_json::from_str(&crate::strip_jsonc_comments(raw))
            .expect("builtin manifest parses");
        // macOS gains one over windows/linux for each mac-only entry; sc-8444 added
        // `krea_realtime_14b` (macOS-only — there is no candle Krea Realtime engine), taking it
        // from 81 to 82.
        //
        // sc-17627 then declared the three person-vision utilities that were previously job-time
        // auto-downloads with no catalog entry at all: `sam3_person_segment` (both platforms),
        // `sam2_person_segment` (macOS-only — `mod person_segment` is `#[cfg(target_os = "macos")]`)
        // and `person_detector` (one platform-scoped row each, so exactly one survives
        // `retain_downloads_for_os` per OS). macOS +3 → 85, windows/linux +2 → 82. `real_esrgan`
        // swapped its repo rather than adding one, so it does not move these counts.
        //
        // sc-17632 then declared `seedvr2_upscaler`, the last of that same class: a job-time
        // auto-download with no catalog entry, fetched TWICE into two `<data_dir>/cache` subtrees.
        // One download row, no `platforms` scoping (both the image and video SeedVR2 lanes run on
        // macOS and on the off-Mac candle lane), so every OS gains exactly one: macOS 85 → 86,
        // windows/linux 82 → 83.
        //
        // sc-17634 declared `dwpose_pose_detector`, the LAST of that class and the only one that
        // was not a Hugging Face download at all (two openmmlab `.zip` bundles, re-hosted at
        // `SceneWorks/dwpose-onnx` so it can be installed like everything else). One download row
        // carrying both ONNX graphs, no `platforms` scoping — the pose lane runs on macOS and on
        // the off-Mac candle lane — so every OS gains exactly one: macOS 86 → 87, windows/linux
        // 83 → 84.
        //
        // SCAIL-2 bf16 is now the shared cross-backend package, so Windows and Linux
        // each gain its exact pinned download context while macOS keeps the same one.
        //
        // sc-18481 retired AuraSR from the installable catalog because every production backend
        // rejects its dead `engine:aura-sr` route. Its unscoped download row had contributed one
        // context on every OS, so removing it reduces macOS 87 → 86 and windows/linux 85 → 84.
        //
        // sc-17158 declared the MiniMax-H3 pair. Both entries share ONE repo
        // (`SceneWorks/minimax-h3-mlx`) and are distinguished only by their default tier's `files`
        // predicate — `q4/transformer/*` versus `q4/transformer_ref/*` — so the context key
        // `(repo, files)` still separates them and macOS gains exactly two. Windows/Linux gained
        // NOTHING at the time: every MiniMax-H3 download row was `platforms: ["macos"]`, so
        // `retain_downloads_for_os` emptied both entries there and `model_download_context` yielded
        // `None`. That asymmetry is the point of running this loop per OS.
        //
        // sc-19558 then gave `minimax_h3` — and ONLY `minimax_h3` — an off-Mac artifact: a
        // `platforms: ["windows", "linux"]` set reading the raw upstream `MiniMaxAI/MiniMax-H3`
        // snapshot, which is the layout `candle-gen-minimax-h3::REQUIRED_COMPONENT_DIRS` loads. Its
        // ONE primary row (`transformer/*`) is a new `(repo, files)` context off-Mac, so
        // windows/linux gain exactly one. `minimax_h3_ref` deliberately gained no off-Mac row, which
        // is why that is +1 and not +2.
        //
        // sc-20267 then widened `minimax_h3`'s q4/q8 tier rows to `["macos","windows","linux"]`. That
        // SWAPS which key that +1 is off-Mac without changing the count: `model_download` prefers the
        // `default: true` row, so the off-Mac context is now
        // `(SceneWorks/minimax-h3-mlx, ["q4/transformer/*"])` rather than
        // `(MiniMaxAI/MiniMax-H3, ["transformer/*"])`, and no other off-Mac entry contributes either
        // key. Recorded because the arithmetic below is unchanged while the reason for one of its terms
        // is not — a reader auditing this count off-Mac will find a repo the sc-19558 note says those
        // platforms never fetch.
        //
        // (The reason `minimax_h3_ref` has no off-Mac row is NOT that candle "default-denies ref2va" —
        // that premise was falsified by sc-17157, which is an ancestor of the pinned inference
        // revision. See the trailing note in that entry's `downloads` for the current reason.)
        //
        // THE NUMBERS BELOW ARE THE SYNC MERGE'S, not any single side's. Starting from the shared
        // 87 / 84 / 84, four independent deltas all apply:
        //   main  SCAIL-2 shared bf16 package      +0 / +1 / +1
        //   main  sc-18481 AuraSR retirement       −1 / −1 / −1   (its row was unscoped)
        //   epic  sc-17158 MiniMax-H3 pair         +2 / +0 / +0   (both rows macOS-only)
        //   epic  sc-19558 H3 off-Mac artifact     +0 / +1 / +1
        // giving 88 / 85 / 85. Each side read only its own pair and so read 86/84/84 (main) or
        // 89/85/85 (epic); neither was right once both landed.
        // Still far below `MODEL_SIZE_CACHE_LIMIT` (256), which is what this guard protects.
        for (os, expected_distinct_contexts) in
            [("macos", 88_usize), ("windows", 85), ("linux", 85)]
        {
            let mut keys = std::collections::HashSet::new();
            for mut model in manifest["models"]
                .as_array()
                .expect("models array")
                .iter()
                .cloned()
            {
                retain_downloads_for_os(&mut model, os);
                if let Some(context) = model_download_context(&model).expect("download context") {
                    keys.insert((context.repo, context.files));
                }
            }
            assert_eq!(
                keys.len(),
                expected_distinct_contexts,
                "{os} builtin download-context count changed; reconsider cache capacity"
            );
            assert!(
                keys.len() <= MODEL_SIZE_CACHE_LIMIT,
                "{os} builtin catalog has {} distinct download contexts but cache holds only {}",
                keys.len(),
                MODEL_SIZE_CACHE_LIMIT
            );
        }
    }

    #[test]
    fn successful_size_estimates_have_a_bounded_positive_ttl() {
        let mut cache = ModelSizeCache::default();
        let key = ("owner/model".to_owned(), vec!["*.safetensors".to_owned()]);
        let before = std::time::Instant::now();
        cache.insert(key.clone(), 1234);
        let expires_at = cache
            .entries
            .get(&key)
            .and_then(|entry| entry.expires_at)
            .expect("successful estimate expires");
        assert!(expires_at > before);
        assert!(expires_at <= before + MODEL_SIZE_POSITIVE_TTL + Duration::from_secs(1));
    }

    #[test]
    fn in_flight_key_registry_is_bounded() {
        let mut cache = ModelSizeCache::default();
        for index in 0..MODEL_SIZE_CACHE_LIMIT {
            let key = (format!("owner/model-{index}"), Vec::new());
            assert!(matches!(
                cache.lookup_or_start(&key),
                ModelSizeLookup::Lead(_)
            ));
        }
        let overflow = ("owner/overflow".to_owned(), Vec::new());
        assert!(matches!(
            cache.lookup_or_start(&overflow),
            ModelSizeLookup::Unshared
        ));
        assert_eq!(cache.in_flight.len(), MODEL_SIZE_CACHE_LIMIT);
    }
}

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
    ),
    ApiError,
> {
    let (mut models, user_model_ids) = merged_model_manifest_entries(state).await?;
    // Resolve per-platform download sources before computing install state/size: some video models
    // carry both a native MLX-convert checkpoint (macOS) and a diffusers/torch checkpoint
    // (Windows/Linux). Keep only the entries applicable to this OS so the download job, status,
    // size, and the frontend all agree on the right repo (sc-3240).
    for model in &mut models {
        retain_downloads_for_os(model, std::env::consts::OS);
    }
    let download_contexts = models
        .iter()
        .map(model_download_context)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((models, download_contexts, user_model_ids))
}

async fn estimate_current_model_catalog_sizes(
    state: &AppState,
) -> Result<HashMap<ModelSizeCacheKey, Option<u64>>, ApiError> {
    let (_, download_contexts, _) = load_model_catalog_inputs(state).await?;
    Ok(estimate_model_catalog_sizes(state, &download_contexts, true).await)
}

async fn build_model_catalog_snapshot(state: &AppState) -> Result<Vec<Value>, ApiError> {
    // sc-8819 (F-017): observe full-catalog assembly (the per-model FS install-state probe
    // sweep) so tests can assert request-scoped and process-shared reuse.
    #[cfg(test)]
    crate::test_note_model_catalog_build();
    let (models, download_contexts, user_model_ids) = load_model_catalog_inputs(state).await?;

    let data_dir = Arc::new(state.settings.data_dir.clone());
    let user_model_ids = Arc::new(user_model_ids);
    let work_items = models
        .into_iter()
        .zip(download_contexts.clone())
        .collect::<Vec<_>>();
    let work_groups = group_model_catalog_work_items(work_items);
    let data_dir_for_sweep = data_dir.clone();
    let user_model_ids_for_sweep = user_model_ids.clone();

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
    let external = external.map_err(|err| {
        ApiError::internal(format!("external model catalog scan task failed: {err}"))
    })?;
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
    let mut entry = merge_model_manifest_entry(find(&builtin), find(&user));
    inject_converted_model_path(&mut entry, &state.settings.data_dir);
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
pub(crate) fn retain_downloads_for_os(model: &mut Value, os: &str) {
    let Some(downloads) = model.get_mut("downloads").and_then(Value::as_array_mut) else {
        return;
    };
    if !downloads
        .iter()
        .any(|entry| entry.get("platforms").is_some())
    {
        return;
    }
    downloads.retain(
        |entry| match entry.get("platforms").and_then(Value::as_array) {
            Some(platforms) => platforms.iter().any(|p| p.as_str() == Some(os)),
            None => true,
        },
    );
}

pub(crate) fn model_download(model: &Value) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let mut fallback = None;
    for download in downloads {
        // Co-requisites (sc-9696) install alongside the primary, never AS it — skip them when
        // choosing the canonical entry for size/install-path/download.
        if !is_supported_model_download(download) || is_co_requisite_download(download) {
            continue;
        }
        fallback.get_or_insert(download);
        if download.get("default").and_then(Value::as_bool) == Some(true) {
            return Some(download.clone());
        }
    }
    fallback.cloned()
}

/// True when a download entry is a co-requisite dependency (sc-9696): fetched ALONGSIDE the primary
/// download rather than as a pick-one alternate, and gating the entry's install state. See the
/// manifest schema `downloads[].coRequisite`.
pub(crate) fn is_co_requisite_download(download: &Value) -> bool {
    download.get("coRequisite").and_then(Value::as_bool) == Some(true)
}

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
pub(crate) fn co_requisite_variant(download: &Value) -> Option<String> {
    download
        .get("variant")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}

/// The co-requisite downloads that apply to `variant` (sc-14980).
///
/// Tier-agnostic rows (no `variant`) always apply. A tier-scoped row applies only to its own tier;
/// when no tier is in scope, every tier-scoped row is returned so callers that genuinely aggregate
/// across tiers still see them all.
pub(crate) fn model_co_requisite_downloads_for_variant(
    model: &Value,
    variant: Option<&str>,
) -> Vec<Value> {
    let wanted = variant.map(|value| value.trim().to_ascii_lowercase());
    model_co_requisite_downloads(model)
        .into_iter()
        .filter(
            |download| match (co_requisite_variant(download), wanted.as_deref()) {
                (None, _) => true,
                (Some(_), None) => true,
                (Some(row), Some(wanted)) => row == wanted,
            },
        )
        .collect()
}

pub(crate) fn model_co_requisite_downloads(model: &Value) -> Vec<Value> {
    model
        .get("downloads")
        .and_then(Value::as_array)
        .map(|downloads| {
            downloads
                .iter()
                .filter(|download| {
                    is_co_requisite_download(download) && is_supported_model_download(download)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Select a specific quant tier's download entry for a quant-matrix model (sc-8508). Returns the
/// supported `downloads` entry whose `variant` matches `variant` (case-insensitive). `None` when the
/// model declares no such tier — the caller surfaces a 400 rather than silently installing the wrong
/// tier. A `None` `variant` argument means "the default tier" and is handled by [`model_download`].
pub(crate) fn model_download_for_variant(model: &Value, variant: &str) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let wanted = variant.trim().to_ascii_lowercase();
    downloads
        .iter()
        .find(|download| {
            is_supported_model_download(download)
                && !is_co_requisite_download(download)
                && download
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_ascii_lowercase())
                    .as_deref()
                    == Some(wanted.as_str())
        })
        .cloned()
}

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

pub(crate) fn is_supported_model_download(download: &Value) -> bool {
    download.get("provider").and_then(Value::as_str) == Some("huggingface")
        && download
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| !repo.is_empty())
}

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
        if path_is_readable_file(&snapshot.join("config.json"))
            || snapshot_has_payload_file(&snapshot)
        {
            return HuggingFaceCacheHealth::installed();
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
            if !diffusers_component_has_weight_file(snapshot, component) {
                missing.push(format!("{component}/<weights>"));
            } else if is_mage && !diffusers_component_safetensors_are_valid(snapshot, component) {
                missing.push(format!("{component}/<weights> (malformed safetensors)"));
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

fn diffusers_component_has_weight_file(snapshot: &FsPath, component: &str) -> bool {
    let component_dir = snapshot.join(component);
    let Ok(entries) = std::fs::read_dir(component_dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        !is_hidden_file(&path)
            && path_is_readable_file(&path)
            && (name.ends_with(".safetensors")
                || name.ends_with(".bin")
                || name.ends_with(".msgpack")
                || name.ends_with(".gguf"))
    })
}

fn diffusers_component_safetensors_are_valid(snapshot: &FsPath, component: &str) -> bool {
    let files = snapshot_files(&snapshot.join(component))
        .into_iter()
        .filter(|path| path.to_ascii_lowercase().ends_with(".safetensors"))
        .collect::<Vec<_>>();
    !files.is_empty()
        && files
            .iter()
            .all(|path| safetensors_header_is_valid(&snapshot.join(component).join(path)))
}

fn safetensors_header_is_valid(path: &FsPath) -> bool {
    use std::io::Read as _;

    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut length = [0_u8; 8];
    if file.read_exact(&mut length).is_err() {
        return false;
    }
    let header_len = u64::from_le_bytes(length);
    let Ok(file_len) = file.metadata().map(|metadata| metadata.len()) else {
        return false;
    };
    if header_len == 0 || header_len > file_len.saturating_sub(8) || header_len > 64 * 1024 * 1024 {
        return false;
    }
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header).is_ok()
        && serde_json::from_slice::<Value>(&header).is_ok_and(|value| value.is_object())
}

fn snapshot_has_payload_file(snapshot: &FsPath) -> bool {
    snapshot_files(snapshot).into_iter().any(|path| {
        let lower = path.to_ascii_lowercase();
        !lower.ends_with(".md")
            && !lower.ends_with(".png")
            && !lower.ends_with(".jpg")
            && !lower.ends_with(".jpeg")
            && !lower.ends_with(".gitattributes")
    })
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
    if let Some(disabled) = *_state.model_size_estimate_disabled_override.lock() {
        return disabled;
    }
    matches!(
        std::env::var("SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
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
                if let Some(hook) = state.model_size_estimate_test_hook.lock().clone() {
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
    let test_hook = { state.model_size_estimate_test_hook.lock().clone() };
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
        if converted_dir.join("config.json").is_file()
            || converted_dir.join("model_index.json").is_file()
        {
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
        })
    }
}

pub(crate) fn model_artifact_paths(model: &Value, data_dir: &FsPath) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = model_manifest_installed_path(model, data_dir) {
        paths.push(path);
    }
    if let Some(repo) = model_download(model).and_then(|download| {
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
        paths.push(if path.is_absolute() {
            path
        } else {
            data_dir.join(path)
        });
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
mod gated_credential_tests {
    use super::*;
    use serde_json::json;

    fn map(value: Value) -> serde_json::Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn derives_huggingface_host_from_provider() {
        let model = map(json!({
            "downloads": [{ "provider": "huggingface", "repo": "black-forest-labs/FLUX.1-dev", "files": [] }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("huggingface.co")
        );
    }

    #[test]
    fn prefers_explicit_download_credential_host() {
        let model = map(json!({
            "downloads": [{ "provider": "civitai", "credentialHost": "https://Civitai.com/", "sourceUrl": "https://civitai.com/api/x" }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("civitai.com")
        );
    }

    #[test]
    fn falls_back_to_source_url_host() {
        let model = map(json!({
            "downloads": [{ "provider": "url", "sourceUrl": "https://models.example.com/path/file.safetensors" }]
        }));
        assert_eq!(
            derive_credential_host(&model).as_deref(),
            Some("models.example.com")
        );
    }

    #[test]
    fn no_downloads_yields_none() {
        assert_eq!(derive_credential_host(&map(json!({}))), None);
    }

    // sc-7872: the SD3.5 gated entries download direct from the gated stabilityai/*
    // repos (no re-host), so the credential host derives to huggingface.co exactly
    // like FLUX.2-dev — driving the same stored-HF-token download path.
    #[test]
    fn derives_huggingface_host_for_stabilityai_sd3_5_repos() {
        for repo in [
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "stabilityai/stable-diffusion-3.5-medium",
        ] {
            let model = map(json!({
                "downloads": [{ "provider": "huggingface", "repo": repo, "files": ["transformer/*"] }]
            }));
            assert_eq!(
                derive_credential_host(&model).as_deref(),
                Some("huggingface.co"),
                "repo {repo} should derive huggingface.co",
            );
        }
    }

    // sc-7872: a gated SD3.5 entry round-trips through apply_gating_fields with its
    // explicit huggingface.co credential host preserved (the field the web client
    // reads to gate the download + surface the credential prompt). licenseUrl is
    // untouched, so the model card links the stabilityai license page.
    #[test]
    fn sd3_5_gated_entry_preserves_credential_host_and_license() {
        let mut model = map(json!({
            "id": "sd3_5_large",
            "gated": true,
            "credentialHost": "huggingface.co",
            "licenseUrl": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
            "downloads": [{ "provider": "huggingface", "repo": "stabilityai/stable-diffusion-3.5-large", "files": ["transformer/*"] }]
        }));
        apply_gating_fields(&mut model);
        assert_eq!(model.get("gated").and_then(Value::as_bool), Some(true));
        assert_eq!(
            model.get("credentialHost").and_then(Value::as_str),
            Some("huggingface.co"),
        );
        assert_eq!(
            model.get("licenseUrl").and_then(Value::as_str),
            Some("https://huggingface.co/stabilityai/stable-diffusion-3.5-large"),
        );
    }

    // sc-17227: an acknowledgment-only entry — MiniMax-H3, whose HF repo is PUBLIC. The catalog
    // must carry `requiresLicenseAcknowledgment` + `licenseNotice` through to the web client
    // untouched, and must NOT manufacture a `credentialHost` for it: the Models screen keys the
    // "Add token in Settings" / "Request access on Hugging Face" affordances off that host, and
    // there is no token to add and no access to request. Note the asymmetry with the gated case
    // above — `gated` is normalized to an explicit `false`, but the host is left absent, which is
    // exactly what `derive_credential_host` would have supplied had the two been coupled.
    #[test]
    fn license_acknowledgment_entry_keeps_its_fields_and_gains_no_credential_host() {
        let mut model = map(json!({
            "id": "minimax_h3",
            "requiresLicenseAcknowledgment": true,
            "licenseUrl": "https://huggingface.co/MiniMaxAI/MiniMax-H3",
            "licenseNotice": "Applicable Territory excludes the United States of America.",
            "downloads": [{ "provider": "huggingface", "repo": "SceneWorks/minimax-h3-mlx", "files": ["q4/transformer/*"] }]
        }));
        apply_gating_fields(&mut model);
        assert_eq!(
            model
                .get("requiresLicenseAcknowledgment")
                .and_then(Value::as_bool),
            Some(true),
            "the acknowledgment flag must survive to the web client",
        );
        assert_eq!(
            model.get("licenseNotice").and_then(Value::as_str),
            Some("Applicable Territory excludes the United States of America."),
        );
        assert_eq!(
            model.get("licenseUrl").and_then(Value::as_str),
            Some("https://huggingface.co/MiniMaxAI/MiniMax-H3"),
        );
        assert_eq!(model.get("gated").and_then(Value::as_bool), Some(false));
        assert!(
            !model.contains_key("credentialHost"),
            "a public-repo acknowledgment model must not be given a credential host: {model:?}",
        );
    }
}

#[cfg(test)]
mod variant_install_tests {
    use super::*;
    use serde_json::json;
    // Reuse the ONE crate-wide HF-cache guard (sc-13834) so these tests resolve the cache under
    // their tempdir `data_dir`, never a developer's real HF cache — and serialize on the same
    // `HF_ENV_LOCK` as the `crate::tests` suite (a second lock would defeat the mutual exclusion).
    use crate::tests::support::isolate_hf_cache;

    /// Seed a HuggingFace repo cache snapshot under `data_dir` containing `files` (repo-relative
    /// paths). Mirrors the on-disk layout `model_variant_states` probes.
    fn seed_cache(data_dir: &FsPath, repo: &str, files: &[&str]) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let snapshot = cache.join("snapshots").join("abc123");
        for file in files {
            let path = snapshot.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
    }

    fn quant_matrix_model(repo: &str) -> Value {
        json!({
            "id": "matrix_model",
            "downloads": [
                {
                    "provider": "huggingface",
                    "repo": repo,
                    "variant": "q4",
                    "default": true,
                    "files": ["q4/*"],
                    "footprint": { "diskSizeBytes": 5_000_000_000_u64 }
                },
                {
                    "provider": "huggingface",
                    "repo": repo,
                    "variant": "q8",
                    "files": ["q8/*"],
                    "footprint": { "diskSizeBytes": 10_000_000_000_u64, "peakMemoryBytes": null }
                },
                {
                    "provider": "huggingface",
                    "repo": repo,
                    "variant": "bf16",
                    "files": ["bf16/*"],
                    "estimatedSizeBytes": 20_000_000_000_u64
                }
            ]
        })
    }

    #[test]
    fn detects_matrix_and_single_variant_shapes() {
        // A variant-keyed multi-entry model → a matrix.
        assert!(model_has_variant_matrix(&quant_matrix_model(
            "SceneWorks/matrix"
        )));
        // A single entry with an explicit variant → still a matrix (tier-tracked).
        assert!(model_has_variant_matrix(&json!({
            "downloads": [{ "provider": "huggingface", "repo": "o/m", "variant": "q4" }]
        })));
        // A single unlabeled entry → NOT a matrix (back-compat single-variant).
        assert!(!model_has_variant_matrix(&json!({
            "downloads": [{ "provider": "huggingface", "repo": "o/m" }]
        })));
        // MULTIPLE unlabeled entries (alternate sources / co-requisite TE repos / native fallback)
        // → NOT a matrix. Entry count is not the discriminator; only an explicit `variant` is
        // (sc-8508). Guards against the old `supported.len() > 1` heuristic that falsely flagged
        // ~30 multi-repo models.
        assert!(!model_has_variant_matrix(&json!({
            "downloads": [
                { "provider": "huggingface", "repo": "org/backbone" },
                { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it" }
            ]
        })));
        // An empty-string variant is not a real tier label → not a matrix.
        assert!(!model_has_variant_matrix(&json!({
            "downloads": [
                { "provider": "huggingface", "repo": "org/a", "variant": "" },
                { "provider": "huggingface", "repo": "org/b" }
            ]
        })));
        // No downloads → not a matrix.
        assert!(!model_has_variant_matrix(&json!({ "id": "x" })));
    }

    #[test]
    fn alternate_source_multi_entry_yields_one_default_variant() {
        // Two unlabeled download entries (alternate sources / co-requisite TE repo) must NOT be
        // treated as a quant matrix: both would otherwise collapse to a duplicate "default" key.
        // The dedup guard emits exactly one "default" variant, matching the single-variant contract.
        let _env = isolate_hf_cache(); // resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let model = json!({
            "id": "alt_source",
            "downloads": [
                { "provider": "huggingface", "repo": "org/backbone" },
                { "provider": "huggingface", "repo": "SceneWorks/gemma-2-2b-it" }
            ]
        });
        assert!(!model_has_variant_matrix(&model));
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 1, "alternate-source model emits one variant");
        assert_eq!(states[0].variant, "default");
    }

    #[test]
    fn variant_keys_are_unique_across_emitted_states() {
        // Every emitted variant key must be unique. A manifest that duplicates a variant (or maps
        // two entries to the same key) keeps only the first; downstream tracking never emits two
        // same-keyed states (sc-8508).
        let _env = isolate_hf_cache(); // resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();

        // Genuine matrix → three distinct keys.
        let matrix = quant_matrix_model("SceneWorks/matrix");
        let states = model_variant_states(&matrix, data_dir);
        let mut keys: Vec<_> = states.iter().map(|s| s.variant.clone()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys, vec!["bf16", "q4", "q8"]);

        // Two entries sharing a variant key → collapsed to one (first wins).
        let dup = json!({
            "id": "dup",
            "downloads": [
                { "provider": "huggingface", "repo": "org/a", "variant": "q4", "files": ["q4/*"] },
                { "provider": "huggingface", "repo": "org/b", "variant": "q4", "files": ["q4-alt/*"] }
            ]
        });
        let dup_states = model_variant_states(&dup, data_dir);
        assert_eq!(dup_states.len(), 1);
        assert_eq!(dup_states[0].variant, "q4");
    }

    #[test]
    fn single_variant_model_yields_one_default_variant() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let model = json!({
            "id": "single",
            "downloads": [{ "provider": "huggingface", "repo": "owner/single" }]
        });
        // Not installed yet.
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].variant, "default");
        assert!(!states[0].installed);

        // Seed the cache with a payload file → installed.
        seed_cache(data_dir, "owner/single", &["model.safetensors"]);
        let states = model_variant_states(&model, data_dir);
        assert!(states[0].installed);
        assert!(states[0].installed_path.is_some());
    }

    /// Mark a convert-at-install model "converted" by writing its local MLX `config.json`.
    fn seed_converted(data_dir: &FsPath, model_id: &str) {
        let dir = data_dir.join("models").join("mlx").join(model_id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
    }

    /// A converted convert-at-install model reports `updateAvailable` iff the manifest's current
    /// `convertSourceFile` is NOT in the source cache (the converted dir carries no version stamp,
    /// so the cache is the proxy). Generic: keys only off the manifest fields.
    #[test]
    fn mlx_update_available_tracks_source_file_in_cache() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "TenStrip/LTX2.3-10Eros";
        let model = json!({
            "id": "ltx_2_3_eros",
            "mlx": {
                "requiresConversion": true,
                "converter": "ltx_video",
                "convertSourceRepo": repo,
                "convertSourceFile": "10Eros_v1.3_bf16.safetensors"
            }
        })
        .as_object()
        .unwrap()
        .clone();

        // Not converted + nothing cached → not installed, no update signal.
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert_eq!(status.install_state, "missing");
        assert!(!status.update_available);

        // Converted, but only the OLDER source is cached (manifest now points at v1.3) → stale.
        seed_converted(data_dir, "ltx_2_3_eros");
        seed_cache(data_dir, repo, &["10Eros_v1_bf16.safetensors"]);
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert_eq!(status.install_state, "installed");
        assert_eq!(status.conversion_state, "converted");
        assert!(
            status.update_available,
            "current source not cached → update available"
        );

        // The manifest's current source file is now cached → up to date.
        seed_cache(data_dir, repo, &["10Eros_v1.3_bf16.safetensors"]);
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert!(
            !status.update_available,
            "current source cached → no update"
        );
    }

    /// A dir-based converter (no `convertSourceFile`) never reports an update — the mechanism
    /// degrades to a no-op rather than misfiring, so it's safe to leave enabled for all models.
    #[test]
    fn mlx_update_unavailable_without_convert_source_file() {
        let _env = isolate_hf_cache(); // resolve the convert-source cache under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let model = json!({
            "id": "flux2_dev",
            "mlx": {
                "requiresConversion": true,
                "converter": "flux2_dev_quant",
                "convertSourceRepo": "black-forest-labs/FLUX.2-dev"
            }
        })
        .as_object()
        .unwrap()
        .clone();
        seed_converted(data_dir, "flux2_dev");
        let status = mlx_catalog_status(&model, data_dir).expect("status");
        assert_eq!(status.conversion_state, "converted");
        assert!(
            !status.update_available,
            "no convertSourceFile → never reports an update"
        );
    }

    #[test]
    fn per_variant_tracking_reports_which_tiers_are_on_disk() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // Only bf16 is downloaded (its `files` filter matches only the bf16/ tree).
        seed_cache(data_dir, repo, &["bf16/model.safetensors"]);

        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 3);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

        // bf16 present; q4 + q8 absent — the whole point of per-variant tracking.
        assert!(by_variant("bf16").installed, "bf16 should read installed");
        assert!(!by_variant("q4").installed, "q4 should read missing");
        assert!(!by_variant("q8").installed, "q8 should read missing");

        // Footprint + size flow through: q4 uses footprint.diskSizeBytes, bf16 uses estimatedSizeBytes.
        assert_eq!(by_variant("q4").download_size_bytes, Some(5_000_000_000));
        assert_eq!(by_variant("bf16").download_size_bytes, Some(20_000_000_000));
        assert_eq!(
            by_variant("q8").footprint.get("peakMemoryBytes"),
            Some(&Value::Null)
        );
    }

    /// Seed one quant tier as a diffusers pipeline snapshot: always a `model_index.json` +
    /// weightless scheduler/tokenizer configs; the transformer/vae/text_encoder weights only when
    /// `complete`. A `complete: false` tier mirrors a torn download (interrupted, or weights pruned)
    /// — its files satisfy the coarse `<tier>/*` glob but it cannot load.
    fn seed_diffusers_tier(data_dir: &FsPath, repo: &str, tier: &str, complete: bool) {
        seed_diffusers_tier_rev(data_dir, repo, "abc123", tier, complete);
    }

    fn seed_diffusers_tier_rev(
        data_dir: &FsPath,
        repo: &str,
        rev: &str,
        tier: &str,
        complete: bool,
    ) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let root = cache.join("snapshots").join(rev).join(tier);
        let write = |rel: &str, body: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        };
        write(
            "model_index.json",
            r#"{
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"]
            }"#,
        );
        // Weightless components ship only config, present in a torn tier too (this is what makes the
        // coarse glob match and wrongly report "installed").
        write("scheduler/scheduler_config.json", "{}");
        write("tokenizer/tokenizer_config.json", "{}");
        write("text_encoder/config.json", "{}");
        if complete {
            write("transformer/config.json", "{}");
            write("transformer/model.safetensors", "weights");
            write("vae/config.json", "{}");
            write("vae/model.safetensors", "weights");
            write("text_encoder/model.safetensors", "weights");
        }
    }

    /// The regression this fix closes: a torn diffusers tier (its `model_index.json` + a config or two
    /// present, but the transformer/vae weights missing) satisfied the coarse `<tier>/*` glob and so
    /// reported a green "Installed" badge — then failed to load at generation with `No such file or
    /// directory`. A tier must read installed only when its weight-bearing components actually hold
    /// weights; an absent tier stays a clean "missing", not a repairable "incomplete".
    #[test]
    fn torn_diffusers_tier_reads_incomplete_not_installed() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // q4 complete (loads), q8 torn (metadata only — the transformer weights never arrived),
        // bf16 never fetched.
        seed_diffusers_tier(data_dir, repo, "q4", true);
        seed_diffusers_tier(data_dir, repo, "q8", false);

        let states = model_variant_states(&model, data_dir);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

        assert!(
            by_variant("q4").installed,
            "complete q4 tier must read installed"
        );
        assert!(
            !by_variant("q8").installed,
            "torn q8 tier must NOT read installed just because its metadata files match the glob"
        );
        assert!(
            by_variant("q8").cache_incomplete,
            "a torn (half-present) tier is a repairable incomplete, not a clean missing"
        );
        assert!(
            !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
            "a never-fetched tier stays a clean missing (no spurious repair prompt — sc-9907)"
        );

        // Model-level state aggregates: q4 is complete, so the model is INSTALLED (usable) overall —
        // and because q8 is genuinely TORN, it is also repairable (sc-14431). The complete q4 keeps
        // `installed` true (the model still works via q4); the torn q8 raises `cache_incomplete` so the
        // model-level Fix button appears and knows what to re-fetch. This is the fix to the old
        // suppression that hid a torn sibling behind a complete one — distinct from sc-9907, where the
        // OTHER tiers were merely never downloaded (missing), not torn.
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(state.installed, "model installed via the complete q4 tier");
        assert!(
            state.cache_incomplete,
            "a genuinely torn q8 makes the model repairable even though q4 is complete (sc-14431)"
        );
        assert!(
            state
                .missing_required_files
                .iter()
                .any(|file| file.starts_with("q8/")),
            "the model-level repair must name the torn q8 tier's missing files, got {:?}",
            state.missing_required_files
        );
    }

    /// sc-14431: the model-level repair signal keys on a genuinely TORN tier alone, not on
    /// `!any_installed`. A complete sibling must NOT suppress repair for a half-downloaded tier — but a
    /// never-fetched (missing) sibling must still NOT raise a spurious repair (sc-9907 preserved).
    #[test]
    fn torn_tier_stays_repairable_behind_a_complete_sibling_but_missing_does_not() {
        let _env = isolate_hf_cache();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // Case A — complete q4 + TORN q8 (metadata only) + missing bf16 → repairable, naming q8.
        {
            let tmp = tempfile::tempdir().unwrap();
            let data_dir = tmp.path();
            seed_diffusers_tier(data_dir, repo, "q4", true);
            seed_diffusers_tier(data_dir, repo, "q8", false);
            let state =
                install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
            assert!(state.installed, "usable via the complete q4 tier");
            assert!(
                state.cache_incomplete,
                "a torn q8 behind a complete q4 must be repairable (sc-14431)"
            );
            assert!(state
                .missing_required_files
                .iter()
                .any(|file| file.starts_with("q8/")));
        }

        // Case B — complete q4 + q8/bf16 NEVER fetched (missing, not torn) → installed, NOT repairable.
        // This is the sc-9907 scenario the fix must not regress.
        {
            let tmp = tempfile::tempdir().unwrap();
            let data_dir = tmp.path();
            seed_diffusers_tier(data_dir, repo, "q4", true);
            let state =
                install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
            assert!(state.installed, "usable via the complete q4 tier");
            assert!(
                !state.cache_incomplete,
                "a clean single-tier install with the others merely un-downloaded must NOT read \
                 repairable (sc-9907)"
            );
            assert!(state.missing_required_files.is_empty());
        }
    }

    /// GH #1858: a manifest revision bump re-hosts a tier to add a previously-absent component (the
    /// FLUX.2 MLX re-hosts' `scheduler/`, whose `model_index.json` had always declared it). The freshly
    /// re-downloaded COMPLETE snapshot then coexists with the older torn snapshot it replaced — a
    /// quant-matrix repo keeps every revision in ONE shared cache, and repair just re-runs the download
    /// (nothing evicts the old snapshot). `huggingface_snapshot_dirs` can rank the OLD snapshot first
    /// (fronted by `refs/main`, or by a higher file count), so a per-tier check that reads only the
    /// first-ranked snapshot kept reporting the tier incomplete even after re-downloading. The tier
    /// completeness must instead ask "is this tier complete in ANY cached revision?".
    #[test]
    fn a_complete_revision_rescues_a_torn_older_snapshot_of_the_same_tier() {
        let _env = isolate_hf_cache();
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // Old revision: every weight present but NO `scheduler/` folder — exactly the pre-#1858 FLUX.2
        // re-host (its `model_index.json` declares `scheduler`, so it reads torn/incomplete).
        seed_diffusers_tier_rev(data_dir, repo, "oldrev", "bf16", true);
        let cache = huggingface_repo_cache_path(data_dir, repo).unwrap();
        std::fs::remove_dir_all(
            cache
                .join("snapshots")
                .join("oldrev")
                .join("bf16")
                .join("scheduler"),
        )
        .unwrap();
        // New revision: the re-host with the scheduler config added — a complete tier.
        seed_diffusers_tier_rev(data_dir, repo, "newrev", "bf16", true);
        // Reproduce the worst-case ranking: front the OLD (torn) snapshot via refs/main.
        std::fs::create_dir_all(cache.join("refs")).unwrap();
        std::fs::write(cache.join("refs").join("main"), "oldrev").unwrap();

        let states = model_variant_states(&model, data_dir);
        let bf16 = states.iter().find(|s| s.variant == "bf16").unwrap();
        assert!(
            bf16.installed && !bf16.cache_incomplete,
            "a complete cached revision must make the tier read installed even when a torn older \
             snapshot is fronted by refs/main — got installed={} incomplete={} missing={:?}",
            bf16.installed,
            bf16.cache_incomplete,
            bf16.missing_required_files
        );
    }

    /// A SANA quant-matrix turnkey (family `sana`). Ships NO `model_index.json`, so the diffusers
    /// completeness augmentation is a no-op — the tightening comes from the shared `sana_tier_complete`.
    fn sana_matrix_model(repo: &str) -> Value {
        json!({
            "id": "sana_sprint_1600m",
            "family": "sana",
            "downloads": [
                { "provider": "huggingface", "repo": repo, "variant": "q4", "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": repo, "variant": "q8", "files": ["q8/*"] },
                { "provider": "huggingface", "repo": repo, "variant": "bf16", "files": ["bf16/*"] }
            ]
        })
    }

    /// Seed a SANA quant tier: always the transformer + VAE weights, plus the Gemma text encoder + its
    /// tokenizer only when `complete`. A `complete: false` tier is TORN — its `<tier>/*` glob matches
    /// (transformer present) but the loader dies on the absent text encoder / tokenizer.
    fn seed_sana_tier(data_dir: &FsPath, repo: &str, tier: &str, complete: bool) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let root = cache.join("snapshots").join("abc123").join(tier);
        let write = |rel: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        };
        write("transformer/diffusion_pytorch_model.safetensors");
        write("vae/diffusion_pytorch_model.safetensors");
        if complete {
            write("text_encoder/gemma-2-2b-it.safetensors");
            write("text_encoder/tokenizer.json");
        }
    }

    /// The sc-13513 fix for a no-`model_index` quant-matrix turnkey: a torn SANA tier (its `<tier>/*`
    /// glob matches the transformer, but the Gemma text encoder + tokenizer never landed) reported a
    /// green "Installed" badge, then failed to load. It must read `incomplete`; a never-fetched tier
    /// stays a clean `missing`.
    #[test]
    fn torn_sana_tier_reads_incomplete_not_installed() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/Sana_Sprint_1.6B_1024px_mlx";
        let model = sana_matrix_model(repo);

        // q4 complete (loads), q8 torn (text encoder + tokenizer missing), bf16 never fetched.
        seed_sana_tier(data_dir, repo, "q4", true);
        seed_sana_tier(data_dir, repo, "q8", false);

        let states = model_variant_states(&model, data_dir);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();

        assert!(
            by_variant("q4").installed,
            "complete q4 tier must read installed"
        );
        assert!(
            !by_variant("q8").installed,
            "torn q8 tier must NOT read installed just because its transformer matched the glob"
        );
        assert!(
            by_variant("q8").cache_incomplete,
            "a torn (half-present) tier is a repairable incomplete, not a clean missing"
        );
        assert!(
            !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
            "a never-fetched tier stays a clean missing (no spurious repair prompt)"
        );

        // Mutation check: completing q8 (add the Gemma text encoder + tokenizer) flips it to installed,
        // proving the predicate discriminates on the text encoder, not merely the transformer glob.
        seed_sana_tier(data_dir, repo, "q8", true);
        let states = model_variant_states(&model, data_dir);
        assert!(states.iter().find(|s| s.variant == "q8").unwrap().installed);
    }

    /// A SenseNova-U1 quant-matrix turnkey (family `sensenova-u1`). Ships a FLAT unified tier — packed
    /// backbone + `config.json` + tokenizer files at the tier root, NO `model_index.json` and no
    /// component subdirs — so the diffusers augmentation is a no-op and the tightening comes from the
    /// shared `sensenova_tier_complete` (sc-14432).
    fn sensenova_matrix_model(repo: &str) -> Value {
        json!({
            "id": "sensenova_u1_8b",
            "family": "sensenova-u1",
            "downloads": [
                { "provider": "huggingface", "repo": repo, "variant": "q4", "default": true, "files": ["q4/*"] },
                { "provider": "huggingface", "repo": repo, "variant": "q8", "files": ["q8/*"] },
                { "provider": "huggingface", "repo": repo, "variant": "bf16", "files": ["bf16/*"] }
            ]
        })
    }

    /// Seed a SenseNova quant tier: always the packed backbone + `config.json`, plus its own
    /// `tokenizer.json` only when `with_tokenizer`. A tokenizer-less tier still matches its `<tier>/*`
    /// glob on the backbone, and loads ONLY if a sibling tier carries a tokenizer to borrow.
    fn seed_sensenova_tier(data_dir: &FsPath, repo: &str, tier: &str, with_tokenizer: bool) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let root = cache.join("snapshots").join("abc123").join(tier);
        let write = |rel: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        };
        write("model.safetensors");
        write("config.json");
        if with_tokenizer {
            write("tokenizer.json");
        }
    }

    /// sc-14432: a SenseNova tier whose `tokenizer.json` never landed cleared the coarse `<tier>/*` glob
    /// on its backbone and reported a green "Installed" badge, then died at load on the absent
    /// tokenizer — "complete but unloadable", with no repair offered. It must read `incomplete`, UNLESS a
    /// sibling tier carries a tokenizer the engine can borrow (`resolve_tokenizer_path`), in which case
    /// the tier genuinely loads and must keep reading installed.
    #[test]
    fn tokenizerless_sensenova_tier_reads_incomplete_unless_a_sibling_can_lend_one() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/sensenova-u1-8b-mlx";
        let model = sensenova_matrix_model(repo);

        // q4 alone, and it ships no tokenizer: nothing to borrow, so it cannot load.
        seed_sensenova_tier(data_dir, repo, "q4", false);
        let states = model_variant_states(&model, data_dir);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();
        assert!(
            !by_variant("q4").installed,
            "a tokenizer-less q4 with no sibling to borrow from must NOT read installed"
        );
        assert!(
            by_variant("q4").cache_incomplete,
            "it is a repairable incomplete, not a clean missing"
        );
        assert!(
            !by_variant("bf16").installed && !by_variant("bf16").cache_incomplete,
            "a never-fetched tier stays a clean missing (no spurious repair prompt)"
        );

        // Mutation check: installing q8 (which DOES ship the tokenizer) makes q4 loadable by borrowing
        // it — proving the predicate tracks the engine's sibling resolution, not mere file presence.
        seed_sensenova_tier(data_dir, repo, "q8", true);
        let states = model_variant_states(&model, data_dir);
        let by_variant = |name: &str| states.iter().find(|s| s.variant == name).unwrap();
        assert!(
            by_variant("q4").installed,
            "q4 borrows q8's tokenizer, so it loads and must read installed"
        );
        assert!(by_variant("q8").installed);
    }

    /// A Boogu turnkey (family `boogu`): a single unlabeled "default" download whose `files` filter
    /// enumerates the three component subdirs of the shipped `base/` Q8 subfolder.
    fn boogu_model(repo: &str) -> Value {
        json!({
            "id": "boogu_image",
            "family": "boogu",
            "downloads": [
                { "provider": "huggingface", "repo": repo, "files": ["base/transformer/*", "base/mllm/*", "base/vae/*"] }
            ]
        })
    }

    /// Seed a Boogu `base/` subfolder: transformer (weights + config), mllm weights, and VAE, plus the
    /// `mllm/tokenizer.json` only when `complete`. A `complete: false` tree is TORN — the loader crashes
    /// first on the missing tokenizer, but each `base/<dir>/*` glob matches the stray weights.
    fn seed_boogu_default(data_dir: &FsPath, repo: &str, complete: bool) {
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let base = cache.join("snapshots").join("abc123").join("base");
        let write = |rel: &str| {
            let path = base.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        };
        write("transformer/diffusion_pytorch_model.safetensors");
        write("transformer/config.json");
        write("mllm/model.safetensors");
        write("vae/diffusion_pytorch_model.safetensors");
        if complete {
            write("mllm/tokenizer.json");
        }
    }

    #[test]
    fn torn_boogu_default_reads_incomplete_not_installed() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/boogu-image-mlx";
        let model = boogu_model(repo);

        // Torn: transformer + mllm weights + vae present, but `mllm/tokenizer.json` never landed. Each
        // `base/<dir>/*` glob matches the stray weights, so the coarse check reads installed.
        seed_boogu_default(data_dir, repo, false);
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 1);
        assert!(
            !states[0].installed,
            "torn boogu default must NOT read installed just because the component globs matched"
        );
        assert!(
            states[0].cache_incomplete,
            "a torn boogu default is a repairable incomplete"
        );

        // Mutation check: adding the tokenizer completes it.
        seed_boogu_default(data_dir, repo, true);
        let states = model_variant_states(&model, data_dir);
        assert!(
            states[0].installed,
            "complete boogu default reads installed"
        );
    }

    /// The TOP-LEVEL badge must agree with the per-variant state. Boogu is a single-variant download, so
    /// `install_state_for` takes the non-matrix else branch — which, before sc-13513, used the coarse
    /// cache health and rendered a false-green `installState:"installed"`/`repairAvailable:false` over a
    /// torn install. The shared predicate must downgrade it there too.
    #[test]
    fn torn_boogu_top_level_badge_reads_incomplete_not_installed() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/boogu-image-mlx";
        let model = boogu_model(repo);

        seed_boogu_default(data_dir, repo, false);
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            !state.installed,
            "a torn boogu install must NOT read installed at the top level"
        );
        assert!(
            state.cache_incomplete,
            "a torn boogu install is a repairable incomplete at the top level"
        );

        // Completing it flips the top-level badge to installed.
        seed_boogu_default(data_dir, repo, true);
        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            state.installed,
            "a complete boogu install reads installed at the top level"
        );
        assert!(!state.cache_incomplete);
    }

    /// SANA repos keep TWO cached snapshots (a tiered one + an older flat one). A tier that is complete
    /// in the tiered snapshot must read installed even though the flat snapshot has no tier subdir — the
    /// check folds across snapshots with `any`, not `all` (guards the epic-13075 multi-snapshot trap).
    #[test]
    fn sana_tier_complete_in_one_of_two_snapshots_reads_installed() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/Sana_1600M_1024px_mlx";
        let model = sana_matrix_model(repo);

        // Tiered snapshot: a complete q4.
        seed_sana_tier(data_dir, repo, "q4", true);
        // A second, FLAT snapshot with weights at the root (no q4/ subdir) — must not drag q4 down.
        let cache = huggingface_repo_cache_path(data_dir, repo).expect("cache path");
        let flat = cache.join("snapshots").join("flat9999");
        for rel in [
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            let path = flat.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }

        let states = model_variant_states(&model, data_dir);
        let q4 = states.iter().find(|s| s.variant == "q4").unwrap();
        assert!(
            q4.installed,
            "q4 complete in the tiered snapshot reads installed despite a flat sibling snapshot"
        );
        assert!(!q4.cache_incomplete);
    }

    #[test]
    fn load_time_quant_variants_share_one_complete_snapshot_predicate() {
        let _env = isolate_hf_cache();
        let tmp = tempfile::tempdir().unwrap();
        let repo = "SceneWorks/Mage-Flow";
        let downloads = ["q4", "q8", "bf16"].map(|variant| {
            json!({
                "provider": "huggingface",
                "repo": repo,
                "variant": variant,
                "files": [
                    "model_index.json", "scheduler/*", "text_encoder/*",
                    "transformer/*", "vae/*"
                ]
            })
        });
        let model = json!({
            "id": "mage_flow",
            "family": "mage-flow",
            "downloads": downloads
        });
        let single = json!({
            "id": "mage_flow_single",
            "family": "mage-flow",
            "downloads": [{
                "provider": "huggingface",
                "repo": repo,
                "files": [
                    "model_index.json", "scheduler/*", "text_encoder/*",
                    "transformer/*", "vae/*"
                ]
            }]
        });

        fn write_json(path: &FsPath, value: Value) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
        }
        fn write_safetensors(path: &FsPath) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let header = br#"{"tensor":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.extend_from_slice(&[0, 0]);
            std::fs::write(path, bytes).unwrap();
        }
        fn seed_mage(data_dir: &FsPath, repo: &str) -> PathBuf {
            let snapshot = huggingface_repo_cache_path(data_dir, repo)
                .unwrap()
                .join("snapshots/abc123");
            write_json(
                &snapshot.join("model_index.json"),
                json!({
                    "_class_name": "MageFlowPipeline",
                    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                    "tokenizer": ["transformers", "AutoProcessor"],
                    "transformer": ["mage_flow", "MageFlow"],
                    "vae": ["mage_flow", "MageVAE"]
                }),
            );
            for (component, class) in [
                ("text_encoder", "Qwen3VLForConditionalGeneration"),
                ("vae", "MageVAE"),
            ] {
                write_json(
                    &snapshot.join(component).join("config.json"),
                    json!({"_class_name": class}),
                );
                write_safetensors(
                    &snapshot
                        .join(component)
                        .join("diffusion_pytorch_model.safetensors"),
                );
            }
            write_json(
                &snapshot.join("transformer/config.json"),
                json!({
                    "_class_name": "MageFlow",
                    "in_channels": 128,
                    "hidden_size": 3072,
                    "depth": 12
                }),
            );
            write_safetensors(&snapshot.join("transformer/diffusion_pytorch_model.safetensors"));
            write_json(
                &snapshot.join("scheduler/scheduler_config.json"),
                json!({"_class_name": "FlowMatchEulerDiscreteScheduler"}),
            );
            write_json(
                &snapshot.join("text_encoder/tokenizer_config.json"),
                json!({"version": "1.0"}),
            );
            write_json(
                &snapshot.join("text_encoder/preprocessor_config.json"),
                json!({"size": 384}),
            );
            snapshot
        }
        let assert_health = |data_dir: &FsPath, installed: bool, label: &str| {
            let states = model_variant_states(&model, data_dir);
            assert_eq!(states.len(), 3);
            assert!(
                states.iter().all(|state| {
                    state.installed == installed && state.cache_incomplete == !installed
                }),
                "{label}: every logical variant must share one health result"
            );
            let top = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
            assert_eq!(top.installed, installed, "{label}: matrix top-level");
            let single_top =
                install_state_for(model_download_context(&single).unwrap(), &single, data_dir);
            assert_eq!(single_top.installed, installed, "{label}: single-variant");
        };

        let valid_dir = tmp.path().join("valid");
        seed_mage(&valid_dir, repo);
        assert_health(&valid_dir, true, "complete Mage snapshot");

        let mutations = [
            ("model_index absent", "model_index.json", false),
            ("model_index malformed", "model_index.json", true),
            (
                "scheduler config absent",
                "scheduler/scheduler_config.json",
                false,
            ),
            (
                "scheduler config malformed",
                "scheduler/scheduler_config.json",
                true,
            ),
            (
                "tokenizer config absent",
                "text_encoder/tokenizer_config.json",
                false,
            ),
            (
                "tokenizer config malformed",
                "text_encoder/tokenizer_config.json",
                true,
            ),
            (
                "vision processor config absent",
                "text_encoder/preprocessor_config.json",
                false,
            ),
            (
                "vision processor config malformed",
                "text_encoder/preprocessor_config.json",
                true,
            ),
            (
                "text encoder config absent",
                "text_encoder/config.json",
                false,
            ),
            (
                "text encoder config malformed",
                "text_encoder/config.json",
                true,
            ),
            (
                "text encoder weights absent",
                "text_encoder/diffusion_pytorch_model.safetensors",
                false,
            ),
            (
                "text encoder weights malformed",
                "text_encoder/diffusion_pytorch_model.safetensors",
                true,
            ),
            (
                "transformer config absent",
                "transformer/config.json",
                false,
            ),
            (
                "transformer config malformed",
                "transformer/config.json",
                true,
            ),
            (
                "transformer weights absent",
                "transformer/diffusion_pytorch_model.safetensors",
                false,
            ),
            (
                "transformer weights malformed",
                "transformer/diffusion_pytorch_model.safetensors",
                true,
            ),
            ("VAE config absent", "vae/config.json", false),
            ("VAE config malformed", "vae/config.json", true),
            (
                "VAE weights absent",
                "vae/diffusion_pytorch_model.safetensors",
                false,
            ),
            (
                "VAE weights malformed",
                "vae/diffusion_pytorch_model.safetensors",
                true,
            ),
        ];
        for (index, (label, relative, malformed)) in mutations.into_iter().enumerate() {
            let data_dir = tmp.path().join(format!("mutation-{index}"));
            let snapshot = seed_mage(&data_dir, repo);
            let path = snapshot.join(relative);
            if malformed {
                std::fs::write(path, b"not valid").unwrap();
            } else {
                std::fs::remove_file(path).unwrap();
            }
            assert_health(&data_dir, false, label);
        }
    }

    /// Anima is convert-at-install; its variants[] "default" tracks the three exact `split_files/` SOURCE
    /// files (not a tier subdir). `tier_subdir_name` resolves those to `split_files`, so the downgrade
    /// RUNS — but must be a NO-OP: a complete source passes `anima_tier_complete` (the layout is
    /// compatible), and a torn source is already coarse-missing (exact-path filter), so `if installed` is
    /// false. This pins that no-op so a future divergence between the download filter and the predicate's
    /// pinned filenames surfaces as a RED test, not a silent false-incomplete on the Anima source badge.
    #[test]
    fn anima_source_download_variant_stays_installed_predicate_is_noop() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "circlestone-labs/Anima";
        let model = json!({
            "id": "anima_base",
            "family": "anima",
            "downloads": [
                { "provider": "huggingface", "repo": repo, "files": [
                    "split_files/diffusion_models/anima-base-v1.0.safetensors",
                    "split_files/text_encoders/qwen_3_06b_base.safetensors",
                    "split_files/vae/qwen_image_vae.safetensors"
                ]}
            ]
        });

        seed_cache(
            data_dir,
            repo,
            &[
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "split_files/vae/qwen_image_vae.safetensors",
            ],
        );
        let states = model_variant_states(&model, data_dir);
        assert_eq!(states.len(), 1);
        assert!(
            states[0].installed,
            "a complete anima source reads installed — the predicate must not false-incomplete it"
        );
        assert!(!states[0].cache_incomplete);
    }

    /// A model whose ONLY on-disk tier is torn must read NOT installed at the model level — including
    /// via the "usable stale" receipt path (sc-13076 backfilled a receipt for whatever was on disk,
    /// so a metadata-only tier produced a receipt whose files all exist yet cannot load).
    #[test]
    fn model_with_only_a_torn_tier_is_not_installed() {
        let _env = isolate_hf_cache(); // seed/resolve under the tempdir, never a dev's real HF cache (sc-13835)
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo = "SceneWorks/matrix";
        let model = quant_matrix_model(repo);

        // Only q8 present, and torn. Plus a backfilled receipt recording its (weightless) files as if
        // complete — exactly the shape found on the reporter's disk.
        seed_diffusers_tier(data_dir, repo, "q8", false);
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({
                "repo": repo,
                "receipts": [{
                    "repo": repo, "modelId": "matrix_model", "variant": "q8",
                    "manifestFiles": ["q8/*"],
                    "resolvedFiles": [
                        "q8/model_index.json", "q8/scheduler/scheduler_config.json",
                        "q8/tokenizer/tokenizer_config.json", "q8/text_encoder/config.json"
                    ],
                    "backfilled": true
                }]
            }))
            .unwrap(),
        )
        .unwrap();

        let state = install_state_for(model_download_context(&model).unwrap(), &model, data_dir);
        assert!(
            !state.installed,
            "a torn-only install must not read installed via the usable-stale receipt path"
        );
    }

    #[test]
    fn variant_footprint_disk_bytes_reads_required_field() {
        let entry = json!({ "footprint": { "diskSizeBytes": 42 } });
        assert_eq!(variant_footprint_disk_bytes(&entry), Some(42));
        assert_eq!(variant_footprint_disk_bytes(&json!({})), None);
    }

    #[test]
    fn variant_download_selector_picks_the_right_tier() {
        let model = quant_matrix_model("SceneWorks/matrix");
        // Case-insensitive match on the declared variant.
        assert_eq!(
            model_download_for_variant(&model, "Q8")
                .and_then(|d| d.get("files").cloned())
                .and_then(|f| f.as_array().and_then(|a| a.first().cloned())),
            Some(Value::String("q8/*".to_owned()))
        );
        // Unknown tier → None (the handler turns this into a 400).
        assert!(model_download_for_variant(&model, "int8").is_none());
        // The default selector still picks the `default: true` (q4) entry — back-compat.
        assert_eq!(
            model_download(&model)
                .and_then(|d| d.get("variant").and_then(Value::as_str).map(str::to_owned)),
            Some("q4".to_owned())
        );
    }
}

#[cfg(test)]
mod mlx_tier_probe_tests {
    use super::*;

    fn write_weight(dir: &std::path::Path, backbone: &str, file: &str) {
        let d = dir.join(backbone);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(file), b"x").unwrap();
    }

    #[test]
    fn convert_output_tiers_probes_diffusion_models_highest_first_ignoring_appledouble() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Anima layout: <tier>/diffusion_models/<dit>.safetensors present for bf16 + q8; q4 has only a
        // hidden `._` AppleDouble sidecar, which must NOT count as a loadable tier (SceneWorks#1333).
        write_weight(
            &root.join("bf16"),
            "diffusion_models",
            "anima-base-v1.0.safetensors",
        );
        write_weight(
            &root.join("q8"),
            "diffusion_models",
            "anima-base-v1.0.safetensors",
        );
        write_weight(
            &root.join("q4"),
            "diffusion_models",
            "._anima-base-v1.0.safetensors",
        );
        // Highest-fidelity first, q4 excluded.
        assert_eq!(mlx_convert_output_tiers(root), vec!["bf16", "q8"]);
    }

    #[test]
    fn convert_output_tiers_handles_transformer_flat_and_empty_layouts() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_weight(&root.join("q8"), "transformer", "model.safetensors");
        // Flat layout: a sharded index sits directly in the tier dir (no backbone subdir).
        std::fs::create_dir_all(root.join("bf16")).unwrap();
        std::fs::write(
            root.join("bf16")
                .join("diffusion_pytorch_model.safetensors.index.json"),
            b"x",
        )
        .unwrap();
        assert_eq!(mlx_convert_output_tiers(root), vec!["bf16", "q8"]);
        // A flat converted dir (no tier subdirs) yields no tiers → the web renders no picker.
        let flat = tempfile::tempdir().unwrap();
        std::fs::write(flat.path().join("model_index.json"), b"{}").unwrap();
        assert!(mlx_convert_output_tiers(flat.path()).is_empty());
    }

    /// Seed a converted Anima tier: always the DiT (backbone), plus the dense text encoder + VAE only
    /// when `complete`. A `complete: false` tier is TORN — its DiT satisfies the coarse
    /// `tier_subdir_has_weights` probe, but the loader would die on the missing text-encoder/VAE.
    fn seed_anima_tier(root: &std::path::Path, tier: &str, complete: bool) {
        let dir = root.join(tier);
        write_weight(&dir, "diffusion_models", "anima-base-v1.0.safetensors");
        if complete {
            write_weight(&dir, "text_encoders", "qwen_3_06b_base.safetensors");
            write_weight(&dir, "vae", "qwen_image_vae.safetensors");
        }
    }

    fn tier_state_map(states: &[Value]) -> std::collections::BTreeMap<String, (String, String)> {
        states
            .iter()
            .map(|state| {
                (
                    state["tier"].as_str().unwrap().to_owned(),
                    (
                        state["installState"].as_str().unwrap().to_owned(),
                        state["cacheState"].as_str().unwrap().to_owned(),
                    ),
                )
            })
            .collect()
    }

    #[test]
    fn convert_output_tier_states_mark_complete_torn_and_absent_tiers() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // q4 complete (loads), q8 TORN (DiT present but text-encoder/VAE never landed), bf16 absent — all
        // THREE must appear so the picker can show each state (the "show all, disable unavailable" rule).
        seed_anima_tier(root, "q4", true);
        seed_anima_tier(root, "q8", false);

        // family "anima" gates on the shared per-tier predicate: the torn q8 reads incomplete, NOT
        // installed (sc-13513).
        let anima = mlx_convert_output_tier_states(root, "anima", "anima_base");
        assert_eq!(anima.len(), 3);
        let a = tier_state_map(&anima);
        assert_eq!(a["q4"], ("installed".into(), "complete".into()));
        assert_eq!(
            a["q8"],
            ("missing".into(), "incomplete".into()),
            "a torn DiT-only tier must read incomplete, not installed"
        );
        assert_eq!(a["bf16"], ("missing".into(), "missing".into()));
        let disk_bytes = |states: &[Value], tier: &str| {
            states
                .iter()
                .find(|state| state["tier"] == tier)
                .and_then(|state| state["diskSizeBytes"].as_u64())
        };
        assert_eq!(
            disk_bytes(&anima, "q4"),
            Some(3),
            "the complete tier reports its DiT, text encoder, and VAE bytes"
        );
        assert_eq!(
            disk_bytes(&anima, "q8"),
            Some(1),
            "a torn tier reports the bytes that are actually present"
        );
        assert_eq!(disk_bytes(&anima, "bf16"), None);

        // A convert family with no bespoke predicate keeps the coarse backbone probe — a DiT-only tier
        // reads installed (byte-identical to the pre-sc-13513 behavior for any non-anima convert family).
        let coarse = tier_state_map(&mlx_convert_output_tier_states(root, "flux2", "flux2_dev"));
        assert_eq!(coarse["q8"], ("installed".into(), "complete".into()));

        // Mutation check: completing q8 (add the text encoder + VAE) flips it to installed — proving the
        // predicate discriminates on the components, not merely the backbone.
        seed_anima_tier(root, "q8", true);
        let a2 = tier_state_map(&mlx_convert_output_tier_states(root, "anima", "anima_base"));
        assert_eq!(a2["q8"], ("installed".into(), "complete".into()));
    }

    #[test]
    #[cfg(unix)]
    fn convert_output_size_counts_shared_directory_symlinks_once() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let tier = tmp.path().join("converted/q4");
        write_weight(&tier, "diffusion_models", "dit.safetensors");
        let shared = tmp.path().join("source/text_encoders");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(shared.join("encoder.safetensors"), b"shared").unwrap();
        symlink(&shared, tier.join("text_encoders")).unwrap();
        // A second alias to the same directory must not double-count the same shared component.
        symlink(&shared, tier.join("text_encoder_alias")).unwrap();

        assert_eq!(
            converted_tier_loaded_bytes(&tier),
            Some(1 + b"shared".len() as u64)
        );
    }

    // Full catalog path: a converted convert-at-install model (Anima) emits `mlxTiers` from
    // `apply_mac_and_mlx_fields`, so /models carries the Studio picker data. macOS-only (the mlx status
    // probe is `cfg!(target_os = "macos")`).
    #[test]
    #[cfg(target_os = "macos")]
    fn catalog_emits_mlxtiers_and_states_for_converted_anima() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let converted = data_dir.join("models").join("mlx").join("anima_base");
        std::fs::create_dir_all(&converted).unwrap();
        std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
        // bf16 + q4 fully converted (DiT + text encoder + VAE); q8 is TORN (DiT only) — a realistic
        // interrupted convert. The full `apply_mac_and_mlx_fields` path must carry the family predicate.
        let seed_tier = |tier: &str, complete: bool| {
            let base = converted.join(tier);
            let write = |sub: &str, file: &str| {
                let dir = base.join(sub);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join(file), b"x").unwrap();
            };
            write("diffusion_models", "anima-base-v1.0.safetensors");
            if complete {
                write("text_encoders", "qwen_3_06b_base.safetensors");
                write("vae", "qwen_image_vae.safetensors");
            }
        };
        seed_tier("bf16", true);
        seed_tier("q4", true);
        seed_tier("q8", false);

        let mut object = json!({
            "id": "anima_base",
            "type": "image",
            "family": "anima",
            "mlx": { "requiresConversion": true }
        })
        .as_object()
        .unwrap()
        .clone();
        apply_mac_and_mlx_fields(&mut object, data_dir);
        assert_eq!(
            object.get("mlxConversionState").and_then(Value::as_str),
            Some("converted")
        );
        // `mlxTiers` is the coarse installed-tier list — it drives whether the picker renders; the torn
        // q8 still appears (backbone present) so the user sees it, disabled.
        let tiers: Vec<&str> = object
            .get("mlxTiers")
            .and_then(Value::as_array)
            .expect("mlxTiers emitted")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(tiers, vec!["bf16", "q8", "q4"]);
        // `mlxTierStates` is authoritative: through the full catalog path the family predicate marks the
        // torn q8 incomplete and the complete tiers installed (sc-13513).
        let states = tier_state_map(
            object
                .get("mlxTierStates")
                .and_then(Value::as_array)
                .expect("mlxTierStates emitted"),
        );
        assert_eq!(states["bf16"], ("installed".into(), "complete".into()));
        assert_eq!(states["q4"], ("installed".into(), "complete".into()));
        assert_eq!(states["q8"], ("missing".into(), "incomplete".into()));
        let state_rows = object["mlxTierStates"].as_array().unwrap();
        let disk_bytes = |tier: &str| {
            state_rows
                .iter()
                .find(|state| state["tier"] == tier)
                .and_then(|state| state["diskSizeBytes"].as_u64())
        };
        assert_eq!(disk_bytes("bf16"), Some(3));
        assert_eq!(disk_bytes("q4"), Some(3));
        assert_eq!(disk_bytes("q8"), Some(1));
        // Decoupled from the download matrix — the picker must NOT flip `hasVariantMatrix`.
        assert!(object.get("hasVariantMatrix").is_none());
    }

    // Per-model quality floor (sc-10731): `apply_mac_and_mlx_fields` surfaces the manifest
    // `mlx.minQualityTier` as a top-level `minQualityTier` so the web can clamp the DEFAULT tier up to
    // it. Platform-independent (not gated on the macOS mlx-status probe), and only a valid bf16/q8/q4
    // value is emitted — a bogus floor is dropped, an absent floor emits nothing.
    #[test]
    fn catalog_emits_min_quality_floor_from_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let apply = |mlx: Value| {
            let mut object = json!({ "id": "anima_base", "type": "image", "mlx": mlx })
                .as_object()
                .unwrap()
                .clone();
            apply_mac_and_mlx_fields(&mut object, data_dir);
            object
                .get("minQualityTier")
                .and_then(Value::as_str)
                .map(str::to_owned)
        };
        // A declared q8 floor is surfaced verbatim as a top-level field.
        assert_eq!(
            apply(json!({ "minQualityTier": "q8" })),
            Some("q8".to_owned())
        );
        // A model with no floor emits nothing (default absent = q4-tolerant, no clamp).
        assert_eq!(apply(json!({ "requiresConversion": true })), None);
        // An invalid floor value is dropped rather than surfaced.
        assert_eq!(apply(json!({ "minQualityTier": "q2" })), None);
    }
}

// Per-tier delete (sc-12024, epic 8506). Exercises the blob-aware reclamation on a realistic HF
// hub-cache layout — real `blobs/<etag>` files with snapshot SYMLINKS into them — which is the whole
// reason a tier delete is non-trivial: unlinking the tier's snapshot symlinks alone frees nothing.
// unix-gated because the fixtures use symlinks (the production cache layout on macOS/Linux).
#[cfg(all(test, unix))]
mod variant_delete_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    /// Write (once) a `blobs/<etag>` file of `bytes.len()` bytes and return its path.
    fn blob(repo: &FsPath, etag: &str, bytes: &[u8]) -> PathBuf {
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let path = blobs.join(etag);
        if !path.exists() {
            std::fs::write(&path, bytes).unwrap();
        }
        path
    }

    /// Materialize `snapshots/rev/<rel>` as a symlink to `blob_path` (the production cache links
    /// relatively; an absolute link resolves identically under `canonicalize`).
    fn link(repo: &FsPath, rel: &str, blob_path: &FsPath) {
        let link_path = repo.join("snapshots").join("rev").join(rel);
        std::fs::create_dir_all(link_path.parent().unwrap()).unwrap();
        symlink(blob_path, &link_path).unwrap();
    }

    /// Seed one snapshot file backed by its own fresh blob of `size` bytes.
    fn seed(repo: &FsPath, rel: &str, etag: &str, size: usize) {
        let blob_path = blob(repo, etag, &vec![0u8; size]);
        link(repo, rel, &blob_path);
    }

    #[tokio::test]
    async fn deletes_only_the_target_tiers_blobs_and_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        seed(&repo, "q4/model.safetensors", "q4a", 100);
        seed(&repo, "q4/config.json", "q4b", 200);
        seed(&repo, "q8/model.safetensors", "q8a", 500);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/*".to_owned()],
            &[],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // q4's blobs AND snapshot symlinks are gone; the emptied q4 dir is pruned.
        assert!(!repo.join("blobs/q4a").exists());
        assert!(!repo.join("blobs/q4b").exists());
        assert!(!repo.join("snapshots/rev/q4").exists());
        // q8 is fully intact.
        assert!(repo.join("blobs/q8a").exists());
        assert!(repo.join("snapshots/rev/q8/model.safetensors").exists());
        // Reclaimed bytes = q4's blob sizes only (100 + 200), never the shared skeleton.
        assert_eq!(removal.reclaimed_bytes, 300);
        assert!(removal.trash_failed_paths.is_empty());
        assert!(!removal.removed_paths.is_empty());
    }

    #[tokio::test]
    async fn retains_a_blob_shared_with_a_surviving_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        // A single blob referenced by BOTH tiers (identical etag/content), plus a q4-only blob.
        let shared = blob(&repo, "shared", &vec![0u8; 400]);
        link(&repo, "q4/shared.safetensors", &shared);
        link(&repo, "q8/shared.safetensors", &shared);
        seed(&repo, "q4/only.safetensors", "q4only", 100);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/*".to_owned()],
            &[],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // The shared blob survives — q8 still references it — and q8's link still resolves.
        assert!(repo.join("blobs/shared").exists());
        assert!(repo.join("snapshots/rev/q8/shared.safetensors").exists());
        // q4's exclusive blob and all q4 links are removed.
        assert!(!repo.join("blobs/q4only").exists());
        assert!(!repo.join("snapshots/rev/q4").exists());
        // Only the exclusive blob's bytes count as reclaimed; the shared blob does not.
        assert_eq!(removal.reclaimed_bytes, 100);
    }

    #[tokio::test]
    async fn draining_the_last_tier_removes_the_repo_cache_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        seed(&repo, "q4/model.safetensors", "q4a", 100);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/*".to_owned()],
            &[],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // No tier remains → the whole models--repo dir is pruned (no bare refs/ skeleton left behind).
        assert!(!repo.exists());
        assert_eq!(removal.reclaimed_bytes, 100);
    }

    #[tokio::test]
    async fn removes_real_tier_files_from_the_managed_mirror() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let managed = models.join("Org__repo");
        let write = |rel: &str, size: usize| {
            let path = managed.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, vec![0u8; size]).unwrap();
        };
        // A turnkey install writes REAL files (not blob symlinks) under the managed dir.
        write("q4/model.safetensors", 300);
        write("q8/model.safetensors", 500);

        let removal = remove_tier_artifacts(
            None,
            Some(managed.clone()),
            &["q4/*".to_owned()],
            &[],
            std::slice::from_ref(&models),
            true,
        )
        .await
        .unwrap();

        assert!(!managed.join("q4").exists());
        assert!(managed.join("q8/model.safetensors").exists());
        assert_eq!(removal.reclaimed_bytes, 300);
    }

    #[tokio::test]
    async fn empty_tier_files_is_a_no_op() {
        // The "never scope a delete to everything" guard: an empty file filter removes nothing.
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--repo");
        seed(&repo, "q4/model.safetensors", "q4a", 100);

        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &[],
            &[],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        assert!(repo.join("blobs/q4a").exists());
        assert!(removal.removed_paths.is_empty());
        assert_eq!(removal.reclaimed_bytes, 0);
    }

    /// A model whose logical tiers SHARE one file predicate reclaims nothing and keeps the snapshot.
    ///
    /// This is the load-time-quant contract, and it is still the right behavior for any family that
    /// ships one dense snapshot per variant. Mage-Flow used to be the shipping example; since
    /// sc-14980 its tiers are physically distinct, so the fixture below is a synthetic overlapping
    /// model rather than a Mage row — see
    /// `mage_flow_per_tier_delete_reclaims_only_that_tiers_dit` for Mage's actual shape.
    #[tokio::test]
    async fn overlapping_logical_tiers_retain_the_shared_snapshot_and_reclaim_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--Org--overlapping-tiers");
        seed(
            &repo,
            "transformer/diffusion_pytorch_model.safetensors",
            "dit",
            100,
        );
        seed(&repo, "text_encoder/model.safetensors", "te", 200);
        seed(&repo, "vae/diffusion_pytorch_model.safetensors", "vae", 50);

        let complete = vec![
            "transformer/*".to_owned(),
            "text_encoder/*".to_owned(),
            "vae/*".to_owned(),
        ];
        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &complete,
            &complete,
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        assert_eq!(removal.reclaimed_bytes, 0);
        assert!(removal.removed_paths.is_empty());
        assert_eq!(removal.retained_paths.len(), 3);
        assert!(repo
            .join("snapshots/rev/transformer/diffusion_pytorch_model.safetensors")
            .exists());
        assert!(repo.join("blobs/te").exists());
        assert!(repo.join("blobs/vae").exists());
    }

    /// sc-14980 / sc-14979 — Mage-Flow's per-tier delete reclaims REAL bytes, and reclaims only its
    /// own tier's DiT.
    ///
    /// This is the physical-reclaim acceptance sc-14046 delegated. It replaces the honest-but-empty
    /// zero-byte outcome the load-time-quant layout could offer, and it is the reason the tiers had
    /// to become physically distinct artifacts. The byte counts are the REAL sizes measured on the
    /// uploaded mirrors, so a re-host that silently changed the layout would move them.
    ///
    /// Three things must hold simultaneously, and each is a distinct way the design could fail:
    ///   - the deleted tier's DiT bytes are actually reclaimed (not 0);
    ///   - the OTHER installed tier's DiT survives intact (predicates are disjoint);
    ///   - the SHARED text encoder + VAE survive — they live in a different repo entirely and are
    ///     co-requisites, so a variant/tier delete can never strand another installed variant.
    #[tokio::test]
    async fn mage_flow_per_tier_delete_reclaims_only_that_tiers_dit() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        // The variant mirror: q4 and q8 DiT tiers installed side by side.
        let variant = hub.join("models--SceneWorks--Mage-Flow-Base");
        // Real uploaded sizes, scaled down by 1e6 so the fixture stays a unit test; the RATIO and
        // the disjointness are what the assertions are about.
        const Q4_DIT: usize = 2316; // 2.317 GB
        const Q8_DIT: usize = 4374; // 4.374 GB
        seed(
            &variant,
            "q4/transformer/diffusion_pytorch_model.safetensors",
            "q4dit",
            Q4_DIT,
        );
        seed(&variant, "q4/model_index.json", "q4idx", 1);
        seed(
            &variant,
            "q8/transformer/diffusion_pytorch_model.safetensors",
            "q8dit",
            Q8_DIT,
        );
        seed(&variant, "q8/model_index.json", "q8idx", 1);
        // The SHARED components mirror — a different repo, reached only through co-requisite rows.
        let components = hub.join("models--SceneWorks--Mage-Flow-Components-mlx");
        seed(
            &components,
            "q4/text_encoder/model.safetensors",
            "q4te",
            2514,
        );
        seed(
            &components,
            "q4/vae/diffusion_pytorch_model.safetensors",
            "q4vae",
            345,
        );

        // Delete the q4 tier. `retained` carries the SURVIVING tiers' predicates, exactly as
        // `delete_model_variant` builds it (co-requisites are excluded there by construction, which
        // is why no components predicate appears here).
        let removal = remove_tier_artifacts(
            Some(variant.clone()),
            None,
            &["q4/*".to_owned()],
            &["q8/*".to_owned()],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // 1. Real bytes, not zero: exactly the q4 DiT + its tier index.
        assert_eq!(
            removal.reclaimed_bytes,
            (Q4_DIT + 1) as u64,
            "a Mage tier delete must reclaim that tier's own bytes — a 0 here means the tiers went \
             back to sharing one file predicate"
        );
        assert!(!removal.removed_paths.is_empty());
        assert!(!variant.join("blobs/q4dit").exists());
        assert!(!variant.join("snapshots/rev/q4").exists());

        // 2. The sibling tier is untouched.
        assert!(variant.join("blobs/q8dit").exists());
        assert!(variant
            .join("snapshots/rev/q8/transformer/diffusion_pytorch_model.safetensors")
            .exists());

        // 3. The shared components are untouched — different repo, never in the delete's scope.
        assert!(components.join("blobs/q4te").exists());
        assert!(components.join("blobs/q4vae").exists());
        assert!(components
            .join("snapshots/rev/q4/text_encoder/model.safetensors")
            .exists());
    }

    /// The `SceneWorks/minimax-h3-mlx` shape (sc-17150 / sc-17158): ONE repo holding two DiT
    /// partitions per tier, each owned by a DIFFERENT catalog entry. Seeds `tier`'s `transformer/`
    /// (owned by `minimax_h3`) and `transformer_ref/` (owned by `minimax_h3_ref`).
    ///
    /// The two partitions ship a BYTE-IDENTICAL `config.json` (they carry the same architecture and
    /// the same 638 tensor names; only the weights differ), so the hub cache stores it as ONE blob
    /// that both snapshot entries symlink to. That shared blob is the trap: unlinking it with the base
    /// partition would leave the reference partition's `config.json` dangling.
    fn seed_minimax_tier(repo: &FsPath, tier: &str, base_etag: &str, ref_etag: &str, size: usize) {
        let shared_config = blob(repo, &format!("{tier}-config"), b"{}");
        link(
            repo,
            &format!("{tier}/transformer/config.json"),
            &shared_config,
        );
        link(
            repo,
            &format!("{tier}/transformer_ref/config.json"),
            &shared_config,
        );
        for partition in ["transformer", "transformer_ref"] {
            let etag = if partition == "transformer" {
                base_etag
            } else {
                ref_etag
            };
            seed(
                repo,
                &format!("{tier}/{partition}/diffusion_pytorch_model.safetensors.index.json"),
                &format!("{etag}-index"),
                1,
            );
            seed(
                repo,
                &format!("{tier}/{partition}/diffusion_pytorch_model-00001-of-00001.safetensors"),
                etag,
                size,
            );
        }
    }

    /// sc-19078 — a MiniMax-H3 per-tier delete reclaims that tier's own partition and NOTHING else.
    ///
    /// Mirrors `mage_flow_per_tier_delete_reclaims_only_that_tiers_dit`, which is the shipping
    /// physical-per-tier precedent. H3 adds a dimension Mage does not have: the sibling that must
    /// survive is not only another TIER of the same entry but another CATALOG ENTRY's partition inside
    /// the same tier of the same repo — and the two partitions share a blob.
    ///
    /// Four things must hold at once, each a distinct way this could fail:
    ///   - the deleted tier's own partition bytes are actually reclaimed (not 0);
    ///   - the OTHER tier of the same entry survives (the tier predicates are disjoint);
    ///   - the SIBLING ENTRY's partition in the SAME tier survives (the partition predicates are
    ///     disjoint) — the case sc-17139's follow-ups flagged as reachable here for the first time;
    ///   - the blob the two partitions SHARE survives, so the sibling's `config.json` still resolves.
    #[tokio::test]
    async fn minimax_h3_per_tier_delete_reclaims_only_that_entrys_partition() {
        let tmp = tempfile::tempdir().unwrap();
        let hub = tmp.path().join("hub");
        let repo = hub.join("models--SceneWorks--minimax-h3-mlx");
        // Real hosted per-partition sizes scaled down by 1e6 (18,780,109,783 B q4 / 35,302,064,357 B
        // q8). The RATIO and the disjointness are what the assertions are about.
        const Q4_DIT: usize = 18780;
        const Q8_DIT: usize = 35302;
        seed_minimax_tier(&repo, "q4", "q4dit", "q4refdit", Q4_DIT);
        seed_minimax_tier(&repo, "q8", "q8dit", "q8refdit", Q8_DIT);

        // Delete `minimax_h3`'s q4 tier. `retained` carries the surviving tiers of THAT entry, exactly
        // as `delete_model_variant` builds it from the entry's own `downloads`.
        let removal = remove_tier_artifacts(
            Some(repo.clone()),
            None,
            &["q4/transformer/*".to_owned()],
            &[
                "q8/transformer/*".to_owned(),
                "bf16/transformer/*".to_owned(),
            ],
            std::slice::from_ref(&hub),
            true,
        )
        .await
        .unwrap();

        // 1. Real bytes: the q4 base partition's shard + its index, and nothing else. The shared
        //    `config.json` blob is NOT counted — it never left disk.
        assert_eq!(
            removal.reclaimed_bytes,
            (Q4_DIT + 1) as u64,
            "a MiniMax-H3 tier delete must reclaim that entry's own partition bytes"
        );
        assert!(!repo.join("blobs/q4dit").exists());
        assert!(!repo
            .join("snapshots/rev/q4/transformer/diffusion_pytorch_model-00001-of-00001.safetensors")
            .exists());

        // 2. The same entry's OTHER tier is untouched.
        assert!(repo.join("blobs/q8dit").exists());
        assert!(repo
            .join("snapshots/rev/q8/transformer/diffusion_pytorch_model-00001-of-00001.safetensors")
            .exists());

        // 3. The SIBLING ENTRY's partition inside the deleted tier is untouched — `minimax_h3_ref`
        //    stays installed at q4 even though its bytes live in the tier just deleted.
        assert!(repo.join("blobs/q4refdit").exists());
        assert!(repo
            .join(
                "snapshots/rev/q4/transformer_ref/diffusion_pytorch_model-00001-of-00001.safetensors"
            )
            .exists());

        // 4. The blob the two partitions SHARE survives and the sibling's link still resolves through
        //    it — a dangling `config.json` would make the reference entry unloadable while still
        //    reading installed.
        assert!(repo.join("blobs/q4-config").exists());
        let sibling_config = repo.join("snapshots/rev/q4/transformer_ref/config.json");
        assert!(
            std::fs::read(&sibling_config).is_ok(),
            "sibling config resolves"
        );
    }

    /// sc-19078 — the WHOLE-model delete is scoped when the download repo is shared.
    ///
    /// This is the destructive half. `model_artifact_paths` resolves the repo's cache dir, so before
    /// this the blanket `remove_dir_all` on `models--SceneWorks--minimax-h3-mlx` took every
    /// `transformer_ref/` tier with it — up to 132.6 GB of an installed model the user never asked to
    /// delete. `remove_whole_model_artifacts` removes the entry's own `files` scopes instead, with the
    /// sibling entry's scopes retained.
    #[tokio::test]
    async fn whole_model_delete_on_a_shared_repo_keeps_the_sibling_entrys_partitions() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo_name = "SceneWorks/minimax-h3-mlx";
        let repo = huggingface_repo_cache_path(data_dir, repo_name).unwrap();
        seed_minimax_tier(&repo, "q4", "q4dit", "q4refdit", 18780);
        seed_minimax_tier(&repo, "bf16", "bf16dit", "bf16refdit", 66280);

        let downloads = |partition: &str| {
            json!(["q4", "q8", "bf16"]
                .iter()
                .map(|tier| json!({
                    "provider": "huggingface",
                    "repo": repo_name,
                    "variant": tier,
                    "files": [format!("{tier}/{partition}/*")],
                }))
                .collect::<Vec<_>>())
        };
        let base = json!({ "id": "minimax_h3", "downloads": downloads("transformer") });
        let reference =
            json!({ "id": "minimax_h3_ref", "downloads": downloads("transformer_ref") });
        let catalog = vec![base.clone(), reference];
        let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

        let removal = remove_whole_model_artifacts(
            &catalog,
            "minimax_h3",
            &base,
            data_dir,
            &allowed_roots,
            true,
        )
        .await
        .unwrap();

        // Every tier of the deleted entry's own partition is gone…
        assert!(!removal.removed_paths.is_empty());
        for etag in ["q4dit", "bf16dit"] {
            assert!(!repo.join("blobs").join(etag).exists(), "{etag} removed");
        }
        assert!(!repo.join("snapshots/rev/q4/transformer").exists());
        assert!(!repo.join("snapshots/rev/bf16/transformer").exists());

        // …and every tier of the SIBLING entry's partition survives, blobs and links alike.
        for etag in ["q4refdit", "bf16refdit", "q4-config", "bf16-config"] {
            assert!(repo.join("blobs").join(etag).exists(), "{etag} retained");
        }
        for tier in ["q4", "bf16"] {
            let sibling = repo.join(format!("snapshots/rev/{tier}/transformer_ref"));
            assert!(sibling.join("config.json").exists(), "{tier} ref config");
            assert!(std::fs::read(sibling.join("config.json")).is_ok());
            assert!(sibling
                .join("diffusion_pytorch_model-00001-of-00001.safetensors")
                .exists());
        }
        // The repo cache dir itself must NOT be pruned — the sibling still lives in it.
        assert!(repo.is_dir(), "shared repo cache survives a scoped delete");
    }

    /// The exclusive-repo case is UNCHANGED: with no sibling claiming the repo, a whole-model delete
    /// still removes the repo cache wholesale, including files no `files` scope names.
    ///
    /// This is the non-vacuity partner of the test above — without it, scoping could silently become
    /// the universal path and quietly stop reclaiming the ~80 entries that own their repo outright.
    #[tokio::test]
    async fn whole_model_delete_on_an_exclusive_repo_still_removes_the_repo_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo_name = "Org/solo-model";
        let repo = huggingface_repo_cache_path(data_dir, repo_name).unwrap();
        seed(&repo, "q4/transformer/model.safetensors", "q4dit", 100);
        // A file NO declared scope names — only a blanket removal reaches it.
        seed(&repo, "README.md", "readme", 10);

        let model = json!({
            "id": "solo_model",
            "downloads": [{
                "provider": "huggingface",
                "repo": repo_name,
                "variant": "q4",
                "files": ["q4/transformer/*"],
            }],
        });
        let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

        remove_whole_model_artifacts(
            std::slice::from_ref(&model),
            "solo_model",
            &model,
            data_dir,
            &allowed_roots,
            true,
        )
        .await
        .unwrap();

        assert!(
            !repo.exists(),
            "an exclusively-owned repo cache is removed whole"
        );
    }

    /// A shared-repo entry that declares NO `files` scope keeps the blanket removal (`SceneWorks/bernini`
    /// is the shipping example — both entries claim the whole repo with `files: []`). There is no
    /// honest narrower scope for a whole-repo claim, so the documented behavior is preserved rather
    /// than quietly reclaiming nothing.
    #[tokio::test]
    async fn whole_model_delete_keeps_the_blanket_path_for_an_unscoped_shared_claim() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let repo_name = "SceneWorks/whole-repo-pair";
        let repo = huggingface_repo_cache_path(data_dir, repo_name).unwrap();
        seed(&repo, "model.safetensors", "dit", 100);

        let entry = |id: &str| {
            json!({
                "id": id,
                "downloads": [{ "provider": "huggingface", "repo": repo_name, "files": [] }],
            })
        };
        let first = entry("pair_a");
        let catalog = vec![first.clone(), entry("pair_b")];
        let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

        remove_whole_model_artifacts(&catalog, "pair_a", &first, data_dir, &allowed_roots, true)
            .await
            .unwrap();

        assert!(!repo.exists());
    }

    #[test]
    fn repo_file_scopes_union_tiers_and_reject_a_whole_repo_claim() {
        let model = json!({
            "id": "minimax_h3",
            "downloads": [
                { "repo": "SceneWorks/minimax-h3-mlx", "variant": "q4", "files": ["q4/transformer/*"] },
                { "repo": "SceneWorks/minimax-h3-mlx", "variant": "q8", "files": ["q8/transformer/*"] },
                { "repo": "MiniMaxAI/MiniMax-H3", "coRequisite": true, "files": ["vae/*"] },
            ],
        });
        // Only the named repo's rows, unioned across tiers — the co-requisite repo is a different repo
        // and contributes nothing to this repo's scope.
        assert_eq!(
            model_repo_file_scopes(&model, "SceneWorks/minimax-h3-mlx"),
            Some(vec![
                "q4/transformer/*".to_owned(),
                "q8/transformer/*".to_owned()
            ])
        );
        // A repo this entry does not claim at all has no scope.
        assert_eq!(model_repo_file_scopes(&model, "Org/unrelated"), None);
        // One unscoped row poisons the whole repo's scope: it is a claim on everything.
        let unscoped = json!({
            "id": "bernini",
            "downloads": [{ "repo": "SceneWorks/bernini", "files": [] }],
        });
        assert_eq!(
            model_repo_file_scopes(&unscoped, "SceneWorks/bernini"),
            None
        );
        // …and it poisons it even when a SCOPED sibling row is present in the same repo. This is the
        // case the empty-scopes fallback alone cannot express: without the early return the entry
        // would scope its delete to `q4/*` and strand everything else the unscoped row claims.
        let mixed = json!({
            "id": "mixed_claim",
            "downloads": [
                { "repo": "Org/mixed", "variant": "q4", "files": ["q4/*"] },
                { "repo": "Org/mixed", "files": [] },
            ],
        });
        assert_eq!(model_repo_file_scopes(&mixed, "Org/mixed"), None);

        // Sibling scopes exclude the entry itself and INCLUDE a sibling's co-requisite in that repo.
        let sibling = json!({
            "id": "minimax_h3_ref",
            "downloads": [
                { "repo": "SceneWorks/minimax-h3-mlx", "variant": "q4", "files": ["q4/transformer_ref/*"] },
                { "repo": "SceneWorks/minimax-h3-mlx", "coRequisite": true, "files": ["shared/*"] },
            ],
        });
        let catalog = vec![model.clone(), sibling];
        assert_eq!(
            other_entries_repo_file_scopes(&catalog, "minimax_h3", "SceneWorks/minimax-h3-mlx"),
            SiblingRepoScopes {
                primaries: vec!["q4/transformer_ref/*".to_owned()],
                co_requisites: vec!["shared/*".to_owned()],
            },
            "the two kinds must stay SEPARATE — only the co-requisite half may be subtracted from"
        );
        // Viewed from the sibling, the base entry's scopes are the ones retained.
        assert_eq!(
            other_entries_repo_file_scopes(&catalog, "minimax_h3_ref", "SceneWorks/minimax-h3-mlx"),
            SiblingRepoScopes {
                primaries: vec!["q4/transformer/*".to_owned(), "q8/transformer/*".to_owned()],
                co_requisites: Vec::new(),
            }
        );
        // A repo only this entry claims has no sibling scopes at all — the discriminator that keeps
        // the blanket path in force for the entries that own their repo outright.
        assert!(
            other_entries_repo_file_scopes(&catalog, "minimax_h3", "MiniMaxAI/MiniMax-H3")
                .is_empty()
        );
    }

    /// The retained set may subtract the deleted entry's own scopes from the sibling's CO-REQUISITE
    /// half only (sc-19573 review). Subtracting from the union destroys the sibling's weights.
    #[test]
    fn retained_files_never_subtracts_a_siblings_primary_scopes() {
        // The flux_dev ↔ pulid_flux_dev shape, and five more groups like it: the sibling names the
        // IDENTICAL primary `files`, because both entries really do load the same checkpoint. Every
        // one of those scopes must survive. Subtracting from the union yields `[]`, and then
        // `remove_tier_artifacts`'s `selected && !retained` unlinks the sibling's blobs with
        // `permanent=true` — tens of GB, unrecoverable without re-download.
        let own = vec!["q4/*".to_owned(), "q8/*".to_owned(), "bf16/*".to_owned()];
        let shared = SiblingRepoScopes {
            primaries: own.clone(),
            co_requisites: Vec::new(),
        };
        assert_eq!(
            shared.retained_files(&own),
            own,
            "an identically-scoped sibling primary must be retained IN FULL, not emptied"
        );

        // The anima trio: the sibling's primary is its own DiT, and the TE/VAE it shares with the
        // deleted entry ride the deleted entry's own primary rows too. Both must be retained.
        let anima_own = vec![
            "split_files/diffusion_models/anima-base-v1.0.safetensors".to_owned(),
            "split_files/text_encoders/qwen_3_06b_base.safetensors".to_owned(),
            "split_files/vae/qwen_image_vae.safetensors".to_owned(),
        ];
        let anima_sibling = SiblingRepoScopes {
            primaries: vec![
                "split_files/diffusion_models/anima-aesthetic-v1.0.safetensors".to_owned(),
                "split_files/text_encoders/qwen_3_06b_base.safetensors".to_owned(),
                "split_files/vae/qwen_image_vae.safetensors".to_owned(),
            ],
            co_requisites: Vec::new(),
        };
        assert!(
            anima_sibling
                .retained_files(&anima_own)
                .contains(&"split_files/text_encoders/qwen_3_06b_base.safetensors".to_owned()),
            "the shared text encoder anima_aesthetic/anima_turbo still need must be retained"
        );

        // MiniMax-H3, unchanged by the split: the sibling's PRIMARY is `transformer_ref`, and its
        // co-requisite claim on the deleted entry's own `transformer` is what gets subtracted, so
        // "delete minimax_h3" still frees the DiT the user asked to free.
        let mm_own = vec!["q4/transformer/*".to_owned()];
        let mm_sibling = SiblingRepoScopes {
            primaries: vec!["q4/transformer_ref/*".to_owned()],
            co_requisites: vec![
                "q4/transformer/*".to_owned(),
                "q4/text_encoder/*".to_owned(),
            ],
        };
        assert_eq!(
            mm_sibling.retained_files(&mm_own),
            vec![
                "q4/transformer_ref/*".to_owned(),
                "q4/text_encoder/*".to_owned()
            ],
            "the overlapping co-requisite is dropped; the sibling's primary and the shared TE stay"
        );
    }

    // Convert-at-install (Anima) tiers are real `<converted>/<tier>/` dirs with a packed DiT plus
    // SYMLINKS to a shared TE/VAE source that lives outside the tier dirs (sc-12025).
    fn seed_convert_tier(converted: &FsPath, tier: &str, dit_bytes: usize, shared_te: &FsPath) {
        let tier_dir = converted.join(tier);
        let dm = tier_dir.join("diffusion_models");
        std::fs::create_dir_all(&dm).unwrap();
        std::fs::write(dm.join("dit.safetensors"), vec![0u8; dit_bytes]).unwrap();
        let te = tier_dir.join("text_encoders");
        std::fs::create_dir_all(&te).unwrap();
        symlink(shared_te, te.join("te.safetensors")).unwrap();
    }

    #[tokio::test]
    async fn removes_a_convert_tier_counting_only_real_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let converted = models.join("mlx").join("anima_base");
        std::fs::create_dir_all(&converted).unwrap();
        std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
        // The shared TE source lives OUTSIDE the tier dirs — deleting a tier must never free it.
        let shared = models.join("source").join("te.safetensors");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::write(&shared, vec![0u8; 999]).unwrap();
        seed_convert_tier(&converted, "q4", 300, &shared);
        seed_convert_tier(&converted, "q8", 500, &shared);

        let removal =
            remove_converted_tier(converted.join("q4"), std::slice::from_ref(&models), true)
                .await
                .unwrap();

        assert!(!converted.join("q4").exists());
        assert!(converted
            .join("q8/diffusion_models/dit.safetensors")
            .exists());
        // The shared TE source and the converted marker both survive (q8 still installed).
        assert!(shared.exists());
        assert!(converted.join("model_index.json").exists());
        // Only q4's real DiT bytes count — the symlinked shared TE is not reclaimed.
        assert_eq!(removal.reclaimed_bytes, 300);
    }

    #[tokio::test]
    async fn draining_the_last_convert_tier_removes_the_converted_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        let converted = models.join("mlx").join("anima_base");
        std::fs::create_dir_all(&converted).unwrap();
        std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
        let shared = models.join("source").join("te.safetensors");
        std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
        std::fs::write(&shared, vec![0u8; 999]).unwrap();
        seed_convert_tier(&converted, "q4", 300, &shared);

        let removal =
            remove_converted_tier(converted.join("q4"), std::slice::from_ref(&models), true)
                .await
                .unwrap();

        // No tier remains → the whole converted dir (marker included) is gone; the shared source is
        // NOT (it belongs to the download, not the convert output).
        assert!(!converted.exists());
        assert!(shared.exists());
        assert_eq!(removal.reclaimed_bytes, 300);
    }
}

/// Lane-parameterized coverage for the imported-model LoRA advertisement withdrawal.
///
/// These drive `apply_imported_lora_advertisement_for_lanes` directly rather than through the
/// platform-bound wrapper, so BOTH deployment topologies are exercised on every CI platform.
#[cfg(test)]
mod imported_lora_advertisement_tests {
    use super::*;

    fn entry(id: &str, family: &str) -> JsonObject {
        json!({ "id": id, "type": "image", "family": family })
            .as_object()
            .expect("entry object")
            .clone()
    }

    fn withdrawn(object: &JsonObject) -> bool {
        object
            .get("loraCompatibility")
            .and_then(|value| value.get("families"))
            .and_then(Value::as_array)
            .is_some_and(|families| families.is_empty())
            && object["loraCompatibility"]["supported"] == json!(false)
    }

    /// Both native imported Krea 2 single-file loaders take adapters. Neither platform projection
    /// may withdraw the synthesized family promise now that each scheduler gate can claim it.
    #[test]
    fn imported_krea_2_advertisement_survives_on_both_native_lanes() {
        for (lane, mlx, candle) in [("MLX", true, false), ("Candle", false, true)] {
            let mut object = entry("user_kreamania_variant5", "krea_2");
            apply_imported_lora_advertisement_for_lanes(&mut object, mlx, candle);
            assert!(
                object.get("loraCompatibility").is_none(),
                "{lane} serves imported Krea 2 LoRAs; withdrawing would hide a working surface: \
                 {object:?}"
            );
        }
    }

    /// Generated Mage-Flow full fine-tunes render plain text-to-image on both native backends, but
    /// both provider seams reject adapters. Each projection must therefore withdraw only the LoRA
    /// promise while preserving the now-routable model itself.
    #[test]
    fn mage_flow_withdraws_adapters_on_both_native_lanes() {
        for (lane, mlx, candle) in [("MLX", true, false), ("Candle", false, true)] {
            let mut object = entry("finetune_9f3c", "mage-flow");
            apply_imported_lora_advertisement_for_lanes(&mut object, mlx, candle);
            assert!(
                withdrawn(&object),
                "{lane} renders generated Mage t2i but cannot load adapters: {object:?}"
            );
        }
    }

    /// SDXL genuinely serves adapters on both native loaders, and a builtin routes by id rather
    /// than family. Neither may be touched — this projection only ever removes a promise nothing
    /// can keep.
    #[test]
    fn honest_advertisements_and_builtins_are_never_rewritten() {
        for (mlx, candle) in [(true, false), (false, true)] {
            let mut sdxl = entry("community_xl", "sdxl");
            apply_imported_lora_advertisement_for_lanes(&mut sdxl, mlx, candle);
            assert!(
                sdxl.get("loraCompatibility").is_none(),
                "both fused SDXL loaders accept UNet adapters"
            );

            let mut builtin = entry("krea_2_turbo", "krea_2");
            apply_imported_lora_advertisement_for_lanes(&mut builtin, mlx, candle);
            assert!(
                builtin.get("loraCompatibility").is_none(),
                "a builtin routes by id; its shipped manifest row must not be rewritten"
            );
        }
    }

    /// A supported advertisement is byte-preserved, while a real withdrawal changes only the
    /// adapter verdict and keeps sibling compatibility metadata intact.
    #[test]
    fn support_and_withdrawal_preserve_sibling_compatibility_keys() {
        let compatibility =
            json!({ "families": ["krea-2"], "supported": true, "types": ["character", "style"] });
        let mut supported = entry("user_kreamania_variant5", "krea_2");
        supported.insert("loraCompatibility".to_owned(), compatibility.clone());
        apply_imported_lora_advertisement_for_lanes(&mut supported, false, true);
        assert_eq!(
            supported["loraCompatibility"], compatibility,
            "Candle's supported Krea adapter family and sibling keys must remain untouched"
        );

        let mut withdrawn_entry = entry("finetune_9f3c", "mage-flow");
        withdrawn_entry.insert(
            "loraCompatibility".to_owned(),
            json!({ "families": ["mage-flow"], "supported": true, "types": ["character", "style"] }),
        );
        apply_imported_lora_advertisement_for_lanes(&mut withdrawn_entry, false, true);
        assert!(withdrawn(&withdrawn_entry));
        assert_eq!(
            withdrawn_entry["loraCompatibility"]["types"],
            json!(["character", "style"])
        );
    }
}
