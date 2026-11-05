use super::*;

// The shipped catalog currently has more than 64 distinct (repo, files) contexts. Keep ample
// headroom for catalog growth so an in-order scan cannot evict the entries the next scan is about
// to read and degrade into a zero-hit cycle.
pub(super) const MODEL_SIZE_CACHE_LIMIT: usize = 256;
// Failed estimates (offline, rate-limited, or size-less repo metadata) are
// negative-cached so a huggingface.co outage costs one 8s timeout per repo per
// TTL window instead of one per catalog load (sc-4169).
pub(super) const MODEL_SIZE_NEGATIVE_TTL: Duration = Duration::from_secs(300);
// Successful metadata can change when a repository is updated in place. Refresh it periodically
// without making ordinary warm catalog loads depend on Hugging Face.
pub(super) const MODEL_SIZE_POSITIVE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Debug, Default)]
pub(crate) struct ModelSizeCache {
    pub(super) entries: HashMap<ModelSizeCacheKey, CachedSizeEstimate>,
    pub(super) order: VecDeque<ModelSizeCacheKey>,
    pub(super) in_flight: HashMap<ModelSizeCacheKey, Arc<ModelSizeInFlight>>,
}

pub(super) type ModelSizeCacheKey = (String, Vec<String>);

#[derive(Debug, Clone, Copy)]
pub(super) enum ModelSizeFlightStatus {
    Pending,
    Complete(Option<u64>),
    Aborted,
}

#[derive(Debug)]
pub(super) struct ModelSizeInFlight {
    pub(super) status: Mutex<ModelSizeFlightStatus>,
    pub(super) changed: tokio::sync::Notify,
}

impl ModelSizeInFlight {
    pub(super) fn pending() -> Self {
        Self {
            status: Mutex::new(ModelSizeFlightStatus::Pending),
            changed: tokio::sync::Notify::new(),
        }
    }

    pub(super) async fn wait(&self) -> ModelSizeFlightStatus {
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

    pub(super) fn complete(&self, estimate: Option<u64>) {
        *self.status.lock() = ModelSizeFlightStatus::Complete(estimate);
        self.changed.notify_waiters();
    }

    pub(super) fn abort(&self) {
        *self.status.lock() = ModelSizeFlightStatus::Aborted;
        self.changed.notify_waiters();
    }
}

pub(super) enum ModelSizeLookup {
    Cached(Option<u64>),
    Lead(Arc<ModelSizeInFlight>),
    Follow(Arc<ModelSizeInFlight>),
    Unshared,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CachedSizeEstimate {
    pub(super) size_bytes: Option<u64>,
    pub(super) expires_at: Option<std::time::Instant>,
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

    pub(super) fn lookup_or_start(&mut self, key: &ModelSizeCacheKey) -> ModelSizeLookup {
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

    pub(super) fn finish(
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

    pub(super) fn abort(&mut self, key: &ModelSizeCacheKey, in_flight: &Arc<ModelSizeInFlight>) {
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

    pub(super) fn insert_entry(&mut self, key: ModelSizeCacheKey, entry: CachedSizeEstimate) {
        self.order.retain(|existing| existing != &key);
        self.order.push_back(key.clone());
        self.entries.insert(key, entry);
        while self.order.len() > MODEL_SIZE_CACHE_LIMIT {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }

    pub(super) fn touch(&mut self, key: &ModelSizeCacheKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }
}

pub(super) struct ModelSizeFlightLeader {
    pub(super) cache: Arc<Mutex<ModelSizeCache>>,
    pub(super) key: ModelSizeCacheKey,
    pub(super) in_flight: Arc<ModelSizeInFlight>,
    pub(super) finished: bool,
}

impl ModelSizeFlightLeader {
    pub(super) fn new(
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

    pub(super) fn finish(mut self, estimate: Option<u64>) {
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
pub(super) const MODEL_SIZE_ESTIMATE_CONCURRENCY: usize = 8;

pub(super) async fn collect_bounded_ordered<I, T, F, Fut, R>(
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

pub(super) async fn estimate_model_catalog_sizes(
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

pub(super) async fn estimate_current_model_catalog_sizes(
    state: &AppState,
) -> Result<HashMap<ModelSizeCacheKey, Option<u64>>, ApiError> {
    let (_, download_contexts, _, _) = load_model_catalog_inputs(state).await?;
    Ok(estimate_model_catalog_sizes(state, &download_contexts, true).await)
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

pub(super) fn model_size_estimation_disabled(_state: &AppState) -> bool {
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

pub(super) async fn estimate_huggingface_download_size_request(
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
