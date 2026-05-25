use super::*;

#[derive(Debug, Clone)]
pub(crate) struct SnapshotFile {
    pub(crate) path: String,
    pub(crate) size: Option<u64>,
    pub(crate) download_url: String,
}

#[derive(Debug, Clone)]
pub(crate) struct HuggingFaceSnapshot {
    pub(crate) files: Vec<SnapshotFile>,
}

impl HuggingFaceSnapshot {
    pub(crate) async fn resolve(
        client: &reqwest::Client,
        settings: &Settings,
        repo: &str,
        revision: &str,
        files: &[String],
    ) -> WorkerResult<Self> {
        let base_url = settings.huggingface_base_url.trim_end_matches('/');
        let tree_url = format!(
            "{base_url}/api/models/{}/tree/{}?recursive=1&expand=1",
            quote_path(repo),
            quote_path(revision)
        );
        let payload = with_hf_auth(settings, client.get(tree_url))
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        let entries = if let Some(entries) = payload.as_array() {
            entries.clone()
        } else {
            payload
                .get("siblings")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let snapshot_files = entries
            .iter()
            .filter_map(|entry| snapshot_file_from_entry(base_url, repo, revision, entry))
            .filter(|file| allow_pattern_matches(&file.path, files))
            .collect();
        Ok(Self {
            files: snapshot_files,
        })
    }

    pub(crate) fn total_bytes(&self) -> Option<u64> {
        self.files
            .iter()
            .try_fold(0_u64, |total, file| Some(total.saturating_add(file.size?)))
    }
}

pub(crate) fn snapshot_file_from_entry(
    base_url: &str,
    repo: &str,
    revision: &str,
    entry: &Value,
) -> Option<SnapshotFile> {
    let kind = entry.get("type").and_then(Value::as_str);
    if kind.is_some_and(|kind| kind != "file") {
        return None;
    }
    let path = entry
        .get("path")
        .or_else(|| entry.get("rfilename"))
        .and_then(Value::as_str)?;
    Some(SnapshotFile {
        path: path.to_owned(),
        size: entry.get("size").and_then(json_size_to_u64),
        download_url: format!(
            "{base_url}/{}/resolve/{}/{}",
            quote_path(repo),
            quote_path(revision),
            quote_path(path)
        ),
    })
}

pub(crate) struct DownloadContext<'a> {
    pub(crate) api: &'a ApiClient,
    pub(crate) client: &'a reqwest::Client,
    pub(crate) settings: &'a Settings,
    pub(crate) job_id: &'a str,
    pub(crate) cancel_message: &'a str,
}

pub(crate) async fn download_snapshot(
    context: &DownloadContext<'_>,
    target_dir: &Path,
    snapshot: &HuggingFaceSnapshot,
    progress: &mut DownloadProgress<'_>,
) -> WorkerResult<()> {
    tokio::fs::create_dir_all(target_dir).await?;
    for file in &snapshot.files {
        check_cancel(context.api, context.job_id, context.cancel_message).await?;
        let target_path = safe_join(target_dir, &file.path)?;
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let existing_bytes = existing_download_bytes(&target_path, file.size).await?;
        if file.size.is_some_and(|size| existing_bytes == size) {
            continue;
        }
        let mut request = context.client.get(&file.download_url);
        if existing_bytes > 0 {
            request = request.header(header::RANGE, format!("bytes={existing_bytes}-"));
        }
        let response = with_hf_auth(context.settings, request).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(WorkerError::Http(response.error_for_status().unwrap_err()));
        }
        let appending = existing_bytes > 0 && status == StatusCode::PARTIAL_CONTENT;
        if existing_bytes > 0 && !appending {
            progress.discard_started_bytes(existing_bytes);
        }
        let mut response = response;
        let mut output = if appending {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&target_path)
                .await?
        } else {
            tokio::fs::File::create(&target_path).await?
        };
        let mut interval = tokio::time::interval(progress.report_interval());
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                chunk = response.chunk() => {
                    let Some(chunk) = chunk? else {
                        break;
                    };
                    output.write_all(&chunk).await?;
                    progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                }
                _ = interval.tick() => {
                    report_download_progress(context, progress).await?;
                }
            }
        }
        output.flush().await?;
        // A truncated transfer (e.g. the server closes the stream at what looks
        // like a clean EOF) would otherwise be treated as success: the install
        // marker gets written over a corrupt dir and the bad shard only surfaces
        // as an opaque load failure later. When the expected size is known,
        // verify it and remove the partial so the next attempt re-downloads.
        if let Some(expected) = file.size {
            let written = tokio::fs::metadata(&target_path).await?.len();
            if written != expected {
                let _ = tokio::fs::remove_file(&target_path).await;
                return Err(WorkerError::InvalidPayload(format!(
                    "{} download ended at {} but expected {}",
                    file.path,
                    format_bytes(written),
                    format_bytes(expected)
                )));
            }
        }
    }
    Ok(())
}

pub(crate) async fn download_lora_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "LoRA",
        context.settings.max_lora_url_bytes,
    )
    .await
}

pub(crate) async fn download_model_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
) -> WorkerResult<()> {
    download_source_url(
        context,
        source_url,
        target_dir,
        "Model",
        context.settings.max_model_url_bytes,
    )
    .await
}

pub(crate) async fn download_source_url(
    context: &DownloadContext<'_>,
    source_url: &str,
    target_dir: &Path,
    source_label: &str,
    max_bytes: u64,
) -> WorkerResult<()> {
    let url =
        parse_lora_source_url_with_private(source_url, context.settings.allow_private_lora_urls)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    validate_lora_url_dns(context.settings, &url).await?;
    let file_name = lora_source_url_file_name(source_url)
        .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    tokio::fs::create_dir_all(target_dir).await?;
    let target_path = target_dir.join(file_name);
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let total_bytes = lora_source_content_length(&client, source_url).await?;
    if total_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let existing_bytes = existing_download_bytes(&target_path, total_bytes).await?;
    if total_bytes.is_some_and(|total| total > 0 && existing_bytes == total) {
        return Ok(());
    }
    let mut request = client.get(source_url);
    if existing_bytes > 0 {
        request = request.header(header::RANGE, format!("bytes={existing_bytes}-"));
    }
    let mut response = request.send().await?;
    if response.status().is_redirection() {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl redirects are not allowed".to_owned(),
        ));
    }
    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        let range_total = response
            .headers()
            .get(header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(content_range_total);
        if total_bytes
            .or(range_total)
            .is_some_and(|total| total > 0 && existing_bytes == total)
        {
            return Ok(());
        }
    }
    response = response.error_for_status()?;
    let appending = existing_bytes > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    let expected_bytes = total_bytes.or_else(|| {
        response.content_length().map(|remaining| {
            if appending {
                existing_bytes + remaining
            } else {
                remaining
            }
        })
    });
    if expected_bytes.is_some_and(|total| total > max_bytes) {
        return Err(WorkerError::InvalidPayload(format!(
            "{source_label} sourceUrl exceeds the {} limit",
            format_bytes(max_bytes)
        )));
    }
    let mut progress = DownloadProgress::new(
        source_url,
        if appending { existing_bytes } else { 0 },
        expected_bytes,
        progress_report_interval(context.settings),
    );
    let mut output = if appending {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&target_path)
            .await?
    } else {
        tokio::fs::File::create(&target_path).await?
    };
    let mut interval = tokio::time::interval(progress.report_interval());
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    interval.tick().await;
    loop {
        tokio::select! {
            chunk = response.chunk() => {
                let Some(chunk) = chunk? else {
                    break;
                };
                check_cancel(context.api, context.job_id, context.cancel_message).await?;
                output.write_all(&chunk).await?;
                progress.record_transferred(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
                if progress.downloaded_bytes() > max_bytes {
                    return Err(WorkerError::InvalidPayload(format!(
                        "{source_label} sourceUrl exceeds the {} limit",
                        format_bytes(max_bytes)
                    )));
                }
            }
            _ = interval.tick() => {
                report_download_progress(context, &progress).await?;
            }
        }
    }
    output.flush().await?;
    if expected_bytes.is_some_and(|expected| progress.downloaded_bytes() != expected) {
        return Err(WorkerError::InvalidPayload(format!(
            "LoRA sourceUrl download ended at {} but expected {}",
            format_bytes(progress.downloaded_bytes()),
            format_bytes(expected_bytes.unwrap_or_default())
        )));
    }
    Ok(())
}

pub(crate) async fn lora_source_content_length(
    client: &reqwest::Client,
    source_url: &str,
) -> WorkerResult<Option<u64>> {
    let response = client.head(source_url).send().await?;
    if response.status().is_success() {
        return Ok(response.content_length().filter(|value| *value > 0));
    }
    if response.status().is_redirection() {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl redirects are not allowed".to_owned(),
        ));
    }
    if matches!(
        response.status(),
        StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED | StatusCode::FORBIDDEN
    ) {
        return Ok(None);
    }
    response.error_for_status()?;
    Ok(None)
}

pub(crate) fn content_range_total(value: &str) -> Option<u64> {
    value
        .rsplit_once('/')
        .and_then(|(_, total)| total.trim().parse::<u64>().ok())
}

pub(crate) async fn validate_lora_url_dns(
    settings: &Settings,
    url: &reqwest::Url,
) -> WorkerResult<()> {
    if settings.allow_private_lora_urls {
        return Ok(());
    }
    let Some(host) = url.host_str() else {
        return Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host is not allowed".to_owned(),
        ));
    };
    if let Ok(address) = host.parse::<IpAddr>() {
        validate_public_ip(address)
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
        return Ok(());
    }
    let port = url.port_or_known_default().unwrap_or(443);
    let mut resolved_any = false;
    for address in tokio::net::lookup_host((host, port)).await? {
        resolved_any = true;
        validate_public_ip(address.ip())
            .map_err(|error| WorkerError::InvalidPayload(error.message().to_owned()))?;
    }
    if resolved_any {
        Ok(())
    } else {
        Err(WorkerError::InvalidPayload(
            "LoRA sourceUrl host did not resolve".to_owned(),
        ))
    }
}

pub(crate) async fn report_download_progress(
    context: &DownloadContext<'_>,
    progress: &DownloadProgress<'_>,
) -> WorkerResult<()> {
    heartbeat(
        context.api,
        context.settings,
        WorkerStatus::Busy,
        Some(context.job_id),
    )
    .await?;
    update_job(context.api, context.job_id, progress.payload()).await?;
    check_cancel(context.api, context.job_id, context.cancel_message).await
}

pub(crate) struct DownloadProgress<'a> {
    repo: &'a str,
    started_bytes: u64,
    transferred_bytes: u64,
    total_bytes: Option<u64>,
    started_at: Instant,
    report_interval: Duration,
}

impl<'a> DownloadProgress<'a> {
    pub(crate) fn new(
        repo: &'a str,
        started_bytes: u64,
        total_bytes: Option<u64>,
        report_interval: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            repo,
            started_bytes,
            transferred_bytes: 0,
            total_bytes,
            started_at: now,
            report_interval,
        }
    }

    fn downloaded_bytes(&self) -> u64 {
        self.started_bytes.saturating_add(self.transferred_bytes)
    }

    fn record_transferred(&mut self, bytes: u64) {
        self.transferred_bytes = self.transferred_bytes.saturating_add(bytes);
    }

    fn discard_started_bytes(&mut self, bytes: u64) {
        self.started_bytes = self.started_bytes.saturating_sub(bytes);
    }

    fn report_interval(&self) -> Duration {
        self.report_interval
    }

    fn payload(&self) -> ProgressRequest {
        download_progress_payload(
            self.repo,
            self.downloaded_bytes(),
            self.total_bytes,
            self.started_bytes,
            self.started_at.elapsed(),
        )
    }
}

pub fn download_progress_payload(
    repo: &str,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    started_bytes: u64,
    elapsed: Duration,
) -> ProgressRequest {
    let transferred_bytes = downloaded_bytes.saturating_sub(started_bytes);
    let elapsed_seconds = elapsed.as_secs_f64().max(0.001);
    let rate = transferred_bytes as f64 / elapsed_seconds;
    let eta_seconds = total_bytes.and_then(|total| {
        if rate > 0.0 {
            let remaining = total.saturating_sub(downloaded_bytes) as f64;
            Some(number_from_f64((remaining / rate).max(0.0)))
        } else {
            None
        }
    });

    let (progress, message) = if let Some(total) = total_bytes {
        let ratio = if total == 0 {
            1.0
        } else {
            (downloaded_bytes as f64 / total as f64).clamp(0.0, 1.0)
        };
        let remaining = total.saturating_sub(downloaded_bytes);
        (
            0.1 + ratio * 0.85,
            format!(
                "Downloading {repo}: {} of {} ({} left).",
                format_bytes(downloaded_bytes),
                format_bytes(total),
                format_bytes(remaining)
            ),
        )
    } else {
        (
            0.1,
            format!(
                "Downloading {repo}: {} written.",
                format_bytes(downloaded_bytes)
            ),
        )
    };

    progress_payload(
        JobStatus::Downloading,
        ProgressStage::Downloading,
        progress,
        &message,
        None,
        None,
        eta_seconds,
    )
}
