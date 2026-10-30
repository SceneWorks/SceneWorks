//! Manual, published-weight GPU acceptance. No fixture model, synthetic worker,
//! downloaded weight, alternate trainer, or fabricated native receipt is used.
//! Every entry point is ignored; an absent explicit admission is an ERROR.
//! Raw native root RSS observations are retained before validation. The parent
//! authenticates this exact launched process's exit/stdio, never a descendant census.

use super::support::*;
use futures_util::FutureExt;
use sceneworks_worker::test_support::{
    self as native, acceptance_api_cancel_controller, acceptance_progress_snapshot, gen_core,
    observe_acceptance_trainer, AcceptanceProgressLedger, EvaluationManifest, TrainingProbe,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Instant, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, String>;
const SEAL: &str = "c02b05e3c40e6c9666554580ff0636d5853691ffe5f5db5d586d12468c82a0fd";
const PROMPT: &str = "sksmoss red circle centered on a pale background";
const WALL: u64 = 3600;

pub(super) fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
pub(super) fn text<'a>(v: &'a Value, field: &str) -> Result<&'a str> {
    v.get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing nonempty {field}: {v}"))
}
pub(super) fn positive(v: &Value, field: &str) -> Result<u64> {
    v.get(field)
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
        .ok_or_else(|| format!("missing positive native {field}: {v}"))
}
pub(super) fn env(key: &str) -> Result<String> {
    std::env::var(key)
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("manual admission requires {key}"))
}
pub(super) fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(super) fn file_sha(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
// The original cell clock and supervisor-owned control remain authoritative during
// cold authentication. Reuse the engine seal/memo; never trust a cache name/ETag.
pub(super) fn check_preflight_control(
    deadline: Instant,
    cancel_file: Option<&Path>,
) -> gen_core::Result<()> {
    if Instant::now() >= deadline {
        return Err(gen_core::Error::Msg("frozen case deadline exceeded".into()));
    }
    if let Some(path) = cancel_file {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            gen_core::Error::Msg(format!("invalid owned cancellation control: {error}"))
        })?;
        match value.get("cancel").and_then(Value::as_bool) {
            Some(true) => return Err(gen_core::Error::Canceled),
            Some(false) => {}
            None => {
                return Err(gen_core::Error::Msg(
                    "invalid owned cancellation boolean".into(),
                ))
            }
        }
    }
    Ok(())
}
pub(super) fn pin_preflight_weight(
    path: &Path,
    deadline: Instant,
    cancel_file: Option<&Path>,
    observe: impl FnMut(&gen_core::runtime::ArtifactSealProgress),
) -> gen_core::Result<gen_core::PinnedWeightsFile> {
    gen_core::PinnedWeightsFile::pin_with_progress(
        path,
        || check_preflight_control(deadline, cancel_file),
        observe,
    )
}
pub(super) fn json_file(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}
pub(super) fn owned_relative(path: &str) -> Result<&Path> {
    let path = Path::new(path);
    require(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "artifact path must be owned and relative",
    )?;
    Ok(path)
}
pub(super) fn unix_ms() -> Result<u64> {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis(),
    )
    .map_err(|e| e.to_string())
}
fn observed_iso_time() -> Result<String> {
    let millis = unix_ms()?;
    let seconds = i64::try_from(millis / 1000).map_err(|e| e.to_string())?;
    let base = sceneworks_core::time::format_unix_seconds(seconds);
    Ok(format!(
        "{}.{:03}Z",
        base.trim_end_matches('Z'),
        millis % 1000
    ))
}
pub(super) fn source_pin() -> Result<String> {
    let source: Value = serde_json::from_str(include_str!(
        "../../../../config/inference-third-party-source.json"
    ))
    .map_err(|e| e.to_string())?;
    Ok(text(&source, "inferenceRevision")?.to_owned())
}

// Samples the process containing the API, worker and all learned native work.
// This is a sampled maximum, not a kernel lifetime peak or a child-process census.
#[derive(Default)]
pub(super) struct RootProcessResources {
    samples: Vec<Value>,
    peak: u64,
    missing: u64,
    first_failure: Option<String>,
    last_sample: Option<Instant>,
    ceiling: Option<u64>,
}
impl RootProcessResources {
    pub(super) fn with_ceiling(ceiling: u64) -> Self {
        Self {
            ceiling: Some(ceiling),
            ..Default::default()
        }
    }
    pub(super) fn observe(&mut self, bytes: Option<u64>, at: String) -> Result<()> {
        if let Some(bytes) = bytes.filter(|bytes| *bytes > 0) {
            self.peak = self.peak.max(bytes);
            self.samples.push(json!({"at":at,"bytes":bytes}));
            if let Some(ceiling) = self.ceiling.filter(|ceiling| self.peak > *ceiling) {
                let peak = self.peak;
                self.first_failure.get_or_insert_with(|| {
                    format!("native root RSS exceeded frozen cap ({peak} > {ceiling})")
                });
            }
        } else {
            self.missing += 1;
            self.first_failure
                .get_or_insert_with(|| "native root RSS unavailable or zero".into());
        }
        self.result()
    }
    fn result(&self) -> Result<()> {
        self.first_failure
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }
    fn sample(&mut self, force: bool) -> Result<()> {
        if !force
            && self
                .last_sample
                .is_some_and(|at| at.elapsed() < Duration::from_millis(100))
        {
            return self.result();
        }
        self.last_sample = Some(Instant::now());
        self.observe(native::host_process_rss_bytes(), observed_iso_time()?)
    }
    fn receipt(&self) -> Value {
        json!({"scope":"native-api-worker-root", "pid":std::process::id(),
            "measurement":"native-current-process-rss-sampled-maximum", "sampleIntervalMs":100,
            "samples":self.samples,"sampledPeakBytes":self.peak,"missingMeasurementCount":self.missing,
            "resolvedCeilingBytes":self.ceiling,"firstFailure":self.first_failure,
            "metadataHelperAbsenceClaim":false})
    }
}

pub(super) struct Case {
    pub(super) source_backend: String,
    pub(super) backend: &'static str,
    pub(super) parent: String,
    pub(super) method: String,
    pub(super) id: String,
    pub(super) root: PathBuf,
    pub(super) dataset: PathBuf,
    pub(super) manifest: Value,
    pub(super) deadline: Instant,
    pub(super) started_at: String,
    pub(super) preflight_only: bool,
    pub(super) cross: bool,
    pub(super) record: Value,
    pub(super) resources: Arc<Mutex<RootProcessResources>>,
}
impl Case {
    fn admitted(
        parent: &str,
        method: &str,
        source_backend: &str,
        cross: bool,
        preflight: bool,
    ) -> Result<Self> {
        // All admission checks precede app creation, device probing and model construction.
        require(
            env("MOSS_REAL_WEIGHTS_ACCEPTANCE")? == "1",
            "manual GPU loan/admission missing",
        )?;
        require(
            env("GITHUB_EVENT_NAME")? == "workflow_dispatch",
            "real model execution is forbidden outside workflow_dispatch",
        )?;
        require(
            std::env::var_os("MOSS_CPU_RECEIPT_PATH").is_none(),
            "legacy MOSS_CPU_RECEIPT_PATH may not override owned probe output",
        )?;
        require(
            std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1"),
            "HF_HUB_OFFLINE=1 required; no weight downloads",
        )?;
        require(
            PathBuf::from(env("MOSS_ACCEPTANCE_CANCEL_FILE")?).is_absolute(),
            "owned cooperative cancellation file is required before model construction",
        )?;
        let backend = if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "cuda"
        };
        require(
            matches!(parent, "base" | "sft") && matches!(method, "lora" | "lokr"),
            "invalid cell",
        )?;
        require(
            matches!(source_backend, "mlx" | "cuda"),
            "invalid source backend",
        )?;
        require(
            (source_backend != backend) == cross,
            "cross-load source must be opposite the actual execution backend",
        )?;
        require(
            env("MOSS_ACCEPTANCE_PHASE")? == if cross { "cross-load" } else { "capture" },
            "explicit phase does not match selected source/backend",
        )?;
        let id = format!("{source_backend}/{parent}/{method}");
        require(
            env("MOSS_ACCEPTANCE_CASE_ID")? == id && env("MOSS_ACCEPTANCE_SOURCE_CASE_ID")? == id,
            "case/source ID does not match the exact selected test",
        )?;
        require(
            env("MOSS_ACCEPTANCE_EXECUTION_BACKEND")? == backend,
            "selected binary is not the actual execution backend",
        )?;
        require(
            (std::env::var("MOSS_ACCEPTANCE_PREFLIGHT_ONLY").as_deref() == Ok("1")) == preflight,
            "PREFLIGHT_ONLY is reserved for the exact device preflight test",
        )?;
        let manifest_path = PathBuf::from(env("MOSS_ACCEPTANCE_MANIFEST")?);
        require(
            file_sha(&manifest_path)? == SEAL,
            "campaign manifest bytes differ from the frozen seal",
        )?;
        let manifest = json_file(&manifest_path)?;
        let root = PathBuf::from(env("MOSS_ACCEPTANCE_OUTPUT")?);
        require(
            root.is_absolute(),
            "output must be an absolute owned artifact destination",
        )?;
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        require(
            !std::fs::symlink_metadata(&root)
                .map_err(|e| e.to_string())?
                .file_type()
                .is_symlink(),
            "artifact destination may not be a symlink",
        )?;
        let dataset = PathBuf::from(env("MOSS_ACCEPTANCE_DATASET_ROOT")?);
        require(
            dataset.is_absolute() && dataset.is_dir(),
            "sealed shared dataset artifact is missing",
        )?;
        let started_at = env(if cross {
            "MOSS_ACCEPTANCE_CROSS_LOAD_STARTED_AT"
        } else {
            "MOSS_ACCEPTANCE_CASE_STARTED_AT"
        })?;
        // The shared parser truncates fractional seconds DOWN. That can shorten,
        // never extend, the frozen interval supplied by the controller's real clock.
        let started = sceneworks_core::time::parse_utc_seconds(&started_at)
            .filter(|n| *n > 0)
            .ok_or("invalid actual job-start timestamp")?;
        let start_ms = u64::try_from(started)
            .map_err(|e| e.to_string())?
            .checked_mul(1000)
            .ok_or("clock overflow")?;
        let now = unix_ms()?;
        require(start_ms <= now, "case start is in the future")?;
        let elapsed = now - start_ms;
        require(
            elapsed < WALL * 1000,
            "frozen case deadline already expired",
        )?;
        let deadline = Instant::now() + Duration::from_millis(WALL * 1000 - elapsed);
        let record = json!({
            "schemaVersion":1,"recordKind":"campaign-acceptance-evidence", "fixtureLabel":null,
            "isCampaignEvidence":true,"acceptanceDisposition":"missing",
            "rawFormat":"moss-native-api-acceptance-candidate/v1",
            "manifestSha256":SEAL,"caseId":id,"sourceCaseId":id,"executionBackend":backend,
            "phase":if cross {"cross-load"} else {"capture"},
            "missing":["cleanup","crossBackendLoad","inspections"],"unproven":[],"failureReason":null,
            "raw":{"subruns":{},"api":[],"images":{},"startedAt":started_at,"wallLimitSeconds":WALL}
        });
        Ok(Self {
            source_backend: source_backend.into(),
            backend,
            parent: parent.into(),
            method: method.into(),
            id,
            root,
            dataset,
            manifest,
            deadline,
            started_at,
            preflight_only: preflight,
            cross,
            record,
            resources: Arc::new(Mutex::new(RootProcessResources {
                ceiling: std::env::var("MOSS_ACCEPTANCE_RESOLVED_CAP_BYTES")
                    .ok()
                    .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
                    .transpose()?
                    .filter(|value| *value > 0),
                ..Default::default()
            })),
        })
    }
    fn base(&self) -> String {
        format!("moss_image_{}", self.parent)
    }
    fn fraction(&self) -> f64 {
        if self.backend == "mlx" {
            0.60
        } else {
            0.85
        }
    }
    pub(super) fn check_resources(&self, force: bool) -> Result<()> {
        self.resources
            .lock()
            .map_err(|error| error.to_string())?
            .sample(force)
    }
    fn check_control(&self, cancel_file: Option<&Path>) -> gen_core::Result<()> {
        self.check_resources(false).map_err(gen_core::Error::Msg)?;
        check_preflight_control(self.deadline, cancel_file)
    }
    fn check_clock(&self) -> Result<()> {
        self.check_resources(false)?;
        require(
            Instant::now() < self.deadline,
            "frozen case deadline exceeded",
        )
    }
    pub(super) fn cancel_file(&self) -> Result<Option<PathBuf>> {
        let file = std::env::var_os("MOSS_ACCEPTANCE_CANCEL_FILE").map(PathBuf::from);
        if let Some(path) = &file {
            // The supervisor owns either a per-case control file or the EXACT
            // command-log sibling under this backend's owned output directory.
            // Never accept an arbitrary external cancellation input.
            let backend_root = self
                .root
                .parent()
                .and_then(Path::parent)
                .ok_or("owned backend output root missing")?;
            let fixed_layout = self
                .root
                .file_name()
                .is_some_and(|v| v == self.method.as_str())
                && self
                    .root
                    .parent()
                    .and_then(Path::file_name)
                    .is_some_and(|v| v == self.parent.as_str());
            let driver_owned = fixed_layout
                && ["case", "preflight"].iter().any(|phase| {
                    path == &backend_root.join(format!(
                        "{phase}-{}-{}.log.cancel",
                        self.parent, self.method
                    ))
                });
            require(
                path.is_absolute()
                    && !path
                        .components()
                        .any(|part| matches!(part, Component::ParentDir))
                    && (path.starts_with(&self.root)
                        || driver_owned
                        || (self.record["recordKind"] == "bounded-product-acceptance-evidence"
                            && path == &backend_root.join("bounded-product.log.cancel"))),
                "cancellation input is outside this case's owned controller scope",
            )?;
        }
        Ok(file)
    }
    pub(super) fn persist(&self) -> Result<()> {
        let mut record = self.record.clone();
        record["raw"]["rootProcess"] = self
            .resources
            .lock()
            .map_err(|error| error.to_string())?
            .receipt();
        let bytes = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
        let directory = if self.preflight_only {
            self.root.join("preflight")
        } else {
            self.root.clone()
        };
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let path = directory.join("evidence.json");
        require(
            !std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()),
            "evidence destination is a symlink",
        )?;
        std::fs::write(path, bytes).map_err(|e| e.to_string())
    }
    pub(super) fn retain(&self, relative: &str, bytes: &[u8]) -> Result<Value> {
        let relative = if self.preflight_only {
            format!("preflight/{relative}")
        } else {
            relative.to_owned()
        };
        let relative_path = owned_relative(&relative)?;
        let path = self.root.join(relative_path);
        std::fs::create_dir_all(path.parent().ok_or("artifact has no parent")?)
            .map_err(|e| e.to_string())?;
        // create_new prevents accidentally replacing another subrun's witness.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(json!({"path":relative,"sha256":sha(bytes)}))
    }
    pub(super) fn retain_json(&self, relative: &str, value: &Value) -> Result<Value> {
        self.retain(
            relative,
            &serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
        )
    }
    fn retain_file(&self, relative: &str, source: &Path) -> Result<Value> {
        let relative = if self.preflight_only {
            format!("preflight/{relative}")
        } else {
            relative.into()
        };
        let path = self.root.join(owned_relative(&relative)?);
        std::fs::create_dir_all(path.parent().ok_or("artifact parent missing")?)
            .map_err(|e| e.to_string())?;
        let before = file_sha(source)?;
        let mut input = std::fs::File::open(source).map_err(|e| e.to_string())?;
        let mut output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        std::io::copy(&mut input, &mut output).map_err(|e| e.to_string())?;
        output.sync_all().map_err(|e| e.to_string())?;
        require(
            file_sha(&path)? == before && file_sha(source)? == before,
            "native export changed during streamed evidence capture",
        )?;
        Ok(json!({"path":relative,"sha256":before}))
    }
    pub(super) async fn api(
        &mut self,
        app: &axum::Router,
        method: &str,
        route: &str,
        body: Value,
    ) -> Result<(StatusCode, Value)> {
        // Observation/teardown reads remain available after a deadline. Model
        // starts (submit/claim/import/generate) check the clock separately.
        let (status, response) = request(app.clone(), method, route, body.clone()).await;
        let index = self.record["raw"]["api"]
            .as_array()
            .ok_or("API ledger absent")?
            .len();
        let event = json!({"method":method,"route":route,"httpStatus":status.as_u16(),"request":body,"response":response,"atUnixMs":unix_ms()?});
        let artifact = self.retain_json(&format!("api/{index:04}.json"), &event)?;
        self.record["raw"]["api"]
            .as_array_mut()
            .ok_or("API ledger absent")?
            .push(artifact);
        self.persist()?;
        Ok((status, response))
    }
    pub(super) async fn get_job(&mut self, app: &axum::Router, id: &str) -> Result<Value> {
        let (status, job) = self
            .api(app, "GET", &format!("/api/v1/jobs/{id}"), Value::Null)
            .await?;
        require(
            status == StatusCode::OK,
            format!("job read: {status} {job}"),
        )?;
        Ok(job)
    }
    pub(super) async fn claim(
        &mut self,
        app: &axum::Router,
        worker: &sceneworks_worker::Settings,
        expected: &str,
    ) -> Result<sceneworks_core::contracts::JobSnapshot> {
        self.check_clock()?;
        let (status, value) = self
            .api(
                app,
                "POST",
                "/api/v1/jobs/claim",
                json!({"workerId":worker.worker_id}),
            )
            .await?;
        require(
            status == StatusCode::OK && value["job"]["id"] == expected,
            format!("claim did not select exact owned job {expected}: {value}"),
        )?;
        serde_json::from_value(value["job"].clone()).map_err(|e| e.to_string())
    }
}

pub(super) fn worker_settings(
    url: String,
    settings: &crate::Settings,
    gpu_id: String,
) -> sceneworks_worker::Settings {
    sceneworks_worker::Settings {
        api_url: url,
        access_token: None,
        data_dir: settings.data_dir.clone(),
        resolved_cache: settings.resolved_cache.clone(),
        config_dir: settings.config_dir.clone(),
        worker_id: format!("moss-acceptance-{}", uuid::Uuid::new_v4()),
        gpu_id,
        is_child_worker: false,
        poll_seconds: 1,
        heartbeat_seconds: 5,
        shutdown_timeout_seconds: 30,
        huggingface_base_url: "http://127.0.0.1:9".into(),
        huggingface_token: None,
        credentials: Vec::new(),
        max_lora_url_bytes: 1,
        max_model_url_bytes: 1,
        allow_private_lora_urls: false,
        utility_workers: 1,
        backend_mlx_enabled: cfg!(target_os = "macos"),
        backend_candle_enabled: cfg!(all(not(target_os = "macos"), feature = "backend-candle")),
        external_model_roots: Vec::new(),
        gpu_memory_limit_bytes: 0,
    }
}
pub(super) fn set_cap(settings: &sceneworks_worker::Settings, value: Value) -> Result<()> {
    std::fs::create_dir_all(&settings.config_dir).map_err(|e| e.to_string())?;
    std::fs::write(
        sceneworks_core::app_paths::gpu_memory_limit_file(&settings.config_dir),
        serde_json::to_vec(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

// Ordinary previews retain the sealed guidance 1. DMD regressions are a separate
// product mode whose native protocol forbids CFG: omit its guidance override.
pub(super) fn generation_advanced(mode: &str) -> Result<Value> {
    match mode {
        "off" => Ok(json!({"steps":8,"guidanceScale":1.0,"mossImage":{"acceleration":mode}})),
        "unmerged_bf16" | "full_fp32" => Ok(json!({"steps":8,"mossImage":{"acceleration":mode}})),
        _ => Err("unknown MOSS generation acceptance mode".into()),
    }
}

// Metadata-only union of the exact production selections exercised by one cell.
// Parent repository names alone are NOT a component map: Base also selects
// shared SFT files, and SFT's DMD modes select additional native components.
pub(super) fn selected_weight_files(
    parent: &str,
    cross: bool,
) -> Result<BTreeSet<(String, String, String)>> {
    use gen_core::moss_image::{MossDmdPrecision, MossImageAcceleration, MossImageOptions};
    let provider = match parent {
        "base" => "moss_image_base",
        "sft" => "moss_image_sft",
        _ => return Err("unknown MOSS acceptance parent".into()),
    };
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .ok_or("builtin model manifest absent")?
        .1;
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .map_err(|e| e.to_string())?;
    let entry = manifest["models"]
        .as_array()
        .ok_or("builtin models absent")?
        .iter()
        .find(|entry| entry["id"] == provider)
        .and_then(Value::as_object)
        .ok_or("MOSS production model entry absent")?;
    let modes = if parent == "base" || cross {
        vec![MossImageAcceleration::Off]
    } else {
        vec![
            MossImageAcceleration::Off,
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        ]
    };
    let mut files = BTreeSet::new();
    for mode in modes {
        let selected = sceneworks_core::moss_image::selected_components(
            entry,
            &MossImageOptions {
                acceleration: mode,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        for download in selected["downloads"]
            .as_array()
            .ok_or("production MOSS downloads absent")?
        {
            let repo = text(download, "repo")?;
            let revision = text(download, "revision")?;
            for file in download["files"]
                .as_array()
                .ok_or("production component files absent")?
            {
                let file = file
                    .as_str()
                    .ok_or("production component path is not text")?;
                require(
                    gen_core::moss_image::moss_image_asset(repo, revision, file).is_some(),
                    "production selected file has no frozen native byte identity",
                )?;
                files.insert((repo.to_owned(), revision.to_owned(), file.to_owned()));
            }
        }
    }
    require(!files.is_empty(), "production MOSS component map is empty")?;
    Ok(files)
}

/// Authentication uses the linked engine's frozen acquisition manifest, NOT a
/// directory name, cached ETag alone, fixture namespace or caller-supplied hash.
pub(super) fn weight_inventory(settings: &crate::Settings, case: &Case) -> Result<(Value, Value)> {
    let mut weights = Vec::new();
    let mut revisions = BTreeMap::new();
    let selected = selected_weight_files(&case.parent, case.cross)?;
    let cancel_file = case.cancel_file()?;
    for identity in gen_core::moss_image::MOSS_IMAGE_ASSETS {
        if !selected.contains(&(
            identity.repository.to_owned(),
            identity.revision.to_owned(),
            identity.relative_path.to_owned(),
        )) {
            continue;
        }
        let repository = sceneworks_core::hf_home::huggingface_repo_cache_path(
            &settings.data_dir,
            identity.repository,
        )
        .ok_or("official cached repository does not resolve")?;
        let path = repository
            .join("snapshots")
            .join(identity.revision)
            .join(identity.relative_path);
        case.check_control(cancel_file.as_deref())
            .map_err(|e| e.to_string())?;
        let present = path.is_file();
        let pin = if present {
            Some(
                gen_core::PinnedWeightsFile::pin_with_progress(
                    &path,
                    || case.check_control(cancel_file.as_deref()),
                    |_| {},
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let digest = pin.as_ref().map(|pin| {
            pin.content_sha256()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        });
        let size = pin.as_ref().map(|pin| pin.target_fingerprint().size);
        let trusted = digest.as_deref() == Some(identity.sha256) && size == Some(identity.bytes);
        weights.push(json!({"path":path,"repository":identity.repository,"revision":identity.revision,"relativePath":identity.relative_path,"present":present,"sha256":digest,"bytes":size,"expectedSha256":identity.sha256,"expectedBytes":identity.bytes,"trustedHashMatches":trusted}));
        revisions.insert(identity.repository, identity.revision);
    }
    require(
        !weights.is_empty() && weights.len() == selected.len(),
        "native file-identity inventory does not cover the complete production component map",
    )?;
    Ok((json!(weights), json!(revisions)))
}

async fn dataset(
    case: &mut Case,
    app: &axum::Router,
    project: &str,
) -> Result<(Value, EvaluationManifest)> {
    let seal = json_file(&case.dataset.join("dataset-manifest.json"))?;
    let items = case.manifest["dataset"]["items"]
        .as_array()
        .ok_or("campaign dataset absent")?
        .clone();
    require(
        items.len() == 3,
        "sealed dataset must have all three training items",
    )?;
    let sealed_items = seal["items"]
        .as_array()
        .ok_or("shared dataset manifest items missing")?;
    require(
        sealed_items.len() == items.len(),
        "shared dataset artifact has unexpected population",
    )?;
    let mut uploaded = Vec::new();
    let mut identities = Vec::new();
    let mut evaluation = None;
    for (index, item) in items.iter().enumerate() {
        let image_name = text(item, "image")?;
        let caption = text(item, "caption")?;
        require(item["mask"].is_null(), "campaign forbids masks")?;
        let bytes = std::fs::read(case.dataset.join(owned_relative(image_name)?))
            .map_err(|e| e.to_string())?;
        let hash = sha(&bytes);
        let declared = sealed_items
            .iter()
            .find(|v| v["image"] == image_name)
            .ok_or("shared dataset image absent")?;
        require(
            declared["imageSha256"] == hash
                && declared["caption"] == caption
                && declared["captionSha256"] == sha(caption.as_bytes()),
            "sealed source image/caption bytes changed",
        )?;
        let caption_bytes = std::fs::read(
            case.dataset
                .join(owned_relative(text(declared, "captionFile")?)?),
        )
        .map_err(|e| e.to_string())?;
        require(
            caption_bytes == caption.as_bytes() && declared["captionSha256"] == sha(&caption_bytes),
            "sealed actual caption file bytes changed",
        )?;
        let source_image = case.retain(&format!("dataset/{image_name}"), &bytes)?;
        let source_caption = case.retain(&format!("dataset/shape-{index}.txt"), &caption_bytes)?;
        let decoded = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
        require(
            decoded.width() == 256 && decoded.height() == 256,
            "shared dataset dimensions changed",
        )?;
        let (status, upload) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project}/training/uploads"),
            image_name,
            "image/png",
            &bytes,
        )
        .await;
        let upload_receipt = case.retain_json(
            &format!("dataset/upload-{index}.json"),
            &json!({"method":"POST","route":format!("/api/v1/projects/{project}/training/uploads"),"httpStatus":status.as_u16(),"request":{"fileName":image_name,"mimeType":"image/png","sourceImage":source_image,"sourceCaption":source_caption,"caption":caption},"response":upload}),
        )?;
        identities.push(json!({"image":image_name,"imageSha256":hash,"caption":caption,"captionSha256":sha(caption.as_bytes()),"uploadId":upload["id"],"uploadReceipt":upload_receipt}));
        case.record["raw"]["datasetUploads"] = json!(identities);
        case.persist()?;
        require(
            status == StatusCode::CREATED && text(&upload, "id")?.starts_with("dataset_upload_"),
            format!("real dataset upload refused: {upload}"),
        )?;
        require(
            upload["file"]["contentHash"] == hash
                && upload["file"]["width"] == 256
                && upload["file"]["height"] == 256,
            "uploaded bytes differ from shared dataset",
        )?;
        uploaded.push(json!({"path":upload["file"]["path"],"displayName":image_name,"caption":{"text":caption,"triggerWords":["sksmoss"]},"width":256,"height":256}));
        if index == 0 {
            evaluation = Some(
                EvaluationManifest::new(
                    gen_core::TrainingItem::captioned(
                        case.dataset.join(image_name),
                        caption.into(),
                    ),
                    256,
                    256,
                    0x4845_4c44_4f55_5401,
                    0.375,
                )
                .map_err(|e| e.to_string())?,
            );
        }
    }
    let (status, dataset) = case
        .api(
            app,
            "POST",
            &format!("/api/v1/projects/{project}/training/datasets"),
            json!({"name":"MOSS shared sealed dataset","modality":"image","items":uploaded}),
        )
        .await?;
    require(
        status == StatusCode::CREATED,
        format!("dataset creation: {dataset}"),
    )?;
    let stored = dataset["items"].as_array().ok_or("dataset has no items")?;
    require(stored.len() == 3, "stored dataset population changed")?;
    for (source, stored) in identities.iter().zip(stored) {
        require(
            stored["contentHash"] == source["imageSha256"]
                && stored["caption"]["text"] == source["caption"]
                && stored["caption"]["triggerWords"] == json!(["sksmoss"]),
            "stored dataset identity/caption mismatch",
        )?;
        text(stored, "id")?;
        text(stored, "path")?;
    }
    let evaluation = evaluation.ok_or("evaluation source missing")?;
    case.record["raw"]["datasetManifest"] = case.retain_json("dataset/manifest.json", &seal)?;
    case.record["evaluationIdentity"] = json!({"manifestHash":evaluation.identity,"computedAtRunTimeBeforeWeightsLoad":true,"imageSha256":identities[0]["imageSha256"],"caption":PROMPT,"maskSha256":null,"width":256,"height":256,"noiseSeedHex":"0x48454c444f555401","sigma":0.375,"crossBackendTensorValuesMayDiffer":true});
    case.persist()?;
    Ok((dataset, evaluation))
}

async fn submit(
    case: &mut Case,
    app: &axum::Router,
    project: &str,
    dataset: &Value,
    label: &str,
) -> Result<Value> {
    case.check_clock()?;
    let mut config = case.manifest["training"]["config"].clone();
    config["advanced"]["networkType"] = json!(case.method);
    let (status,job) = case.api(app,"POST",&format!("/api/v1/projects/{project}/training/jobs"),json!({"targetId":format!("{}_lora",case.base()),"datasetId":dataset["id"],"config":config,"outputName":format!("{}-{label}",case.id.replace('/',"-")),"dryRun":false})).await?;
    require(
        status == StatusCode::CREATED,
        format!("training submission failed: {job}"),
    )?;
    Ok(job)
}

struct Run {
    job: Value,
    receipt: Value,
    ledger: AcceptanceProgressLedger,
    request: gen_core::TrainingRequest,
}

fn validate_prepared_request(
    request: &gen_core::TrainingRequest,
    method: gen_core::NetworkType,
    fraction: f32,
    low_cap: bool,
) -> gen_core::Result<()> {
    let c = &request.config;
    let controls = c.steps == 60
        && c.seed == 1789
        && c.rank == 4
        && c.alpha.to_bits() == 6.0_f32.to_bits()
        && c.learning_rate.to_bits() == 0.0001_f32.to_bits()
        && c.batch_size == 1
        && c.gradient_accumulation == 1
        && c.save_every == 20
        && c.resolution == 256
        && c.train_dtype == "bf16"
        && !c.gradient_checkpointing
        && c.cache_latents
        && c.cache_text_embeddings
        && c.optimizer == "adamw"
        && c.weight_decay == 0.0
        && c.lr_scheduler == gen_core::train::LrSchedule::Constant
        && c.lr_warmup_steps == 0
        && c.network_type == method
        && c.decompose_factor == -1
        && c.lora_target_modules.is_empty()
        && !c.full_finetune
        && c.control_type.is_none()
        && c.subject_mask_loss.is_none()
        && c.timestep_type == "moss_stage3_v1"
        && c.timestep_bias == "balanced"
        && c.loss_type == "mse"
        && c.trigger_word.as_deref() == Some("sksmoss")
        && c.sample_every == 0
        && c.model_options.get("vae_posterior") == Some(&json!("posterior_mean_v1"))
        && request.items.len() == 3
        && request.items.iter().all(|item| {
            item.subject_mask_path.is_none()
                && item.control_image_path.is_none()
                && item.reference_image_paths.is_empty()
        });
    if !controls {
        return Err(gen_core::Error::Unsupported("actual prepared native request differs from the sealed MOSS training controls; no tuning/fallback is permitted".into()));
    }
    let cap = if low_cap {
        c.memory_limit_bytes.is_some_and(|v| v > 0) && c.memory_limit_fraction.is_none()
    } else {
        c.memory_limit_bytes.is_none()
            && c.memory_limit_fraction.map(f32::to_bits) == Some(fraction.to_bits())
    };
    if !cap {
        return Err(gen_core::Error::Unsupported("actual prepared request did not consume its owned global frozen/control memory-cap handoff".into()));
    }
    Ok(())
}

// Submission writes this informational plan snapshot before any worker is
// claimed. It is not a learned artifact; every other entry remains forbidden.
pub(super) fn validate_unpublished_training_outputs(
    outputs: &[PathBuf],
    submitted_plan: &Value,
) -> Result<()> {
    for output in outputs {
        require(
            output.file_name() == Some(std::ffi::OsStr::new("training-config.json")),
            "header-only/low-cap operation published output files",
        )?;
        let metadata = std::fs::symlink_metadata(output).map_err(|e| e.to_string())?;
        require(
            metadata.file_type().is_file(),
            "submission config snapshot must be a regular file, never a directory or symlink",
        )?;
        require(
            json_file(output)? == *submitted_plan,
            "submission config snapshot differs from the actual submitted plan",
        )?;
    }
    require(outputs.len() <= 1, "duplicate submission config snapshots")
}

enum Execution {
    Preflight,
    Train { cancel_at20: bool },
    LowCap,
}

pub(super) async fn cancel_controller(
    worker: &sceneworks_worker::Settings,
    job: &str,
    ledger: AcceptanceProgressLedger,
    stop: Arc<AtomicBool>,
    cancel_file: Option<PathBuf>,
    deadline: Instant,
    resources: Arc<Mutex<RootProcessResources>>,
) -> gen_core::Result<()> {
    let result =
        acceptance_api_cancel_controller(worker, job, ledger, stop, cancel_file, deadline, || {
            resources
                .lock()
                .map_err(|error| gen_core::Error::Msg(error.to_string()))?
                .sample(false)
                .map_err(gen_core::Error::Msg)
        })
        .await;
    if result.is_err() {
        // A controller failure/deadline is never acceptance success. Still send
        // a real API cancellation before waiting for the owned execution to join.
        // This is the existing bounded ten-second error-only teardown, not an
        // extension of the accepted case deadline or checkpoint-20 stimulus.
        // The shared API transport also rejects unsuccessful HTTP status codes.
        let response = native::request_acceptance_api_cancel(
            worker,
            job,
            Instant::now() + Duration::from_secs(10),
        )
        .await;
        if let Err(error) = response {
            return Err(gen_core::Error::Unsupported(format!("acceptance controller failed and API teardown cancellation could not be delivered: {error}; original={result:?}")));
        }
    }
    result
}

async fn utility(
    case: &Case,
    worker: &sceneworks_worker::Settings,
    job: sceneworks_core::contracts::JobSnapshot,
) -> Result<()> {
    case.check_clock()?;
    let id = job.id.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    let cancel_file = case.cancel_file()?;
    let execute = async {
        native::run_acceptance_utility_job(worker, job).await;
        stopped.store(true, Ordering::Release);
    };
    let (_, result) = tokio::join!(
        execute,
        cancel_controller(
            worker,
            &id,
            Default::default(),
            stop,
            cancel_file,
            case.deadline,
            Arc::clone(&case.resources),
        )
    );
    result.map_err(|e| e.to_string())?;
    case.check_clock()
}

/// A single execution drives the SAME production request-preparation/loader
/// seam as the worker. Neither loader nor observer changes the prepared request.
async fn execute(
    case: &mut Case,
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    submitted: &Value,
    probe: &TrainingProbe,
    label: &str,
    mode: Execution,
) -> Result<(Option<Run>, Value)> {
    let job_id = text(submitted, "id")?.to_owned();
    let snapshot = case.claim(app, worker, &job_id).await?;
    let plan: sceneworks_core::training::TrainingPlan =
        serde_json::from_value(submitted["payload"]["plan"].clone()).map_err(|e| e.to_string())?;
    let request_slot = Arc::new(Mutex::new(None::<gen_core::TrainingRequest>));
    let observer = Arc::clone(&request_slot);
    let loader_request = Arc::clone(&request_slot);
    let metadata = Arc::new(Mutex::new(Value::Null));
    let loader_metadata = Arc::clone(&metadata);
    let ledger: AcceptanceProgressLedger = Default::default();
    let loader_ledger = Arc::clone(&ledger);
    let stop = Arc::new(AtomicBool::new(false));
    let driver_stop = Arc::clone(&stop);
    let loader_probe = probe.clone();
    let expected_engine = case.base();
    let expected_method = gen_core::NetworkType::parse(&case.method);
    let frozen_fraction = case.fraction() as f32;
    let train = matches!(mode, Execution::Train { .. });
    let stall = matches!(mode, Execution::Train { cancel_at20: true });
    let low = matches!(mode, Execution::LowCap);
    let deadline = case.deadline;
    let token_id = format!("{}:{label}:{job_id}", case.id);
    let receipts_before = probe.receipts().map_err(|e| e.to_string())?.len();
    let cancel_file = case.cancel_file()?;
    let execution = native::run_training_execution_observing_request(
        worker,
        &snapshot,
        &plan,
        move |engine, spec| {
            if engine != expected_engine {
                return Err(gen_core::Error::Unsupported(
                    "provider changed from the exact parent".into(),
                ));
            }
            let request = loader_request
                .lock()
                .map_err(|e| gen_core::Error::Unsupported(e.to_string()))?
                .clone()
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(
                        "actual prepared request was not observed before loader".into(),
                    )
                })?;
            validate_prepared_request(&request, expected_method, frozen_fraction, low)?;
            let (trainer, observed) = match native::real_moss_trainer_with_probe(
                engine,
                spec,
                &request,
                loader_probe,
            ) {
                Ok(value) => value,
                Err(error) => {
                    *loader_metadata
                        .lock()
                        .map_err(|e| gen_core::Error::Unsupported(e.to_string()))? = json!({"refusal":error.to_string(),"weightsLoaded":false,"observationSchema":native::moss_acceptance_observation_schema()});
                    return Err(error);
                }
            };
            *loader_metadata
                .lock()
                .map_err(|e| gen_core::Error::Unsupported(e.to_string()))? = observed;
            if !train || low {
                return Err(gen_core::Error::Unsupported(
                    "manual header-only preflight stops before train/model tensor loading".into(),
                ));
            }
            observe_acceptance_trainer(trainer, loader_ledger, token_id, stall, deadline)
        },
        move |request| {
            if let Ok(mut slot) = observer.lock() {
                *slot = Some(request.clone());
            }
        },
    );
    let execution = async {
        let result = execution.await;
        driver_stop.store(true, Ordering::Release);
        result
    };
    // No timeout drops a native future. The support observer bounds its wait;
    // the owned controller sends cancellation through the API, and this join
    // continues driving the consumer until native tasks have really returned.
    let (result, control) = if train {
        tokio::join!(
            execution,
            cancel_controller(
                worker,
                &job_id,
                Arc::clone(&ledger),
                stop,
                cancel_file,
                deadline,
                Arc::clone(&case.resources),
            )
        )
    } else {
        (execution.await, Ok(()))
    };
    let observed = metadata.lock().map_err(|e| e.to_string())?.clone();
    let receipts = probe.receipts().map_err(|e| e.to_string())?;
    let raw_receipts = serde_json::to_value(&receipts).map_err(|e| e.to_string())?;
    let progress = acceptance_progress_snapshot(&ledger).map_err(|e| e.to_string())?;
    // Retain even on errors, BEFORE status/learning/cancellation assertions.
    let raw = json!({"result":result.as_ref().err().map(ToString::to_string),"controllerError":control.as_ref().err().map(ToString::to_string),"nativeMetadata":observed,"nativeReceipts":raw_receipts,"progress":progress,"atUnixMs":unix_ms()?});
    let diagnostic = case.retain_json(&format!("runs/{label}/raw.json"), &raw)?;
    case.record["raw"]["subruns"][label] = json!({"jobId":job_id,"diagnostic":diagnostic});
    case.persist()?;
    let request = request_slot
        .lock()
        .map_err(|e| e.to_string())?
        .clone()
        .ok_or("production seam did not expose the actual prepared request")?;
    case.record["raw"]["subruns"][label]["preparedNativeConfig"] = json!({"steps":request.config.steps,"seed":request.config.seed,"rank":request.config.rank,"alpha":request.config.alpha,"batchSize":request.config.batch_size,"gradientAccumulation":request.config.gradient_accumulation,"gradientCheckpointing":request.config.gradient_checkpointing,"cacheLatents":request.config.cache_latents,"cacheTextEmbeddings":request.config.cache_text_embeddings,"trainDtype":request.config.train_dtype,"learningRate":request.config.learning_rate,"optimizer":request.config.optimizer,"weightDecay":request.config.weight_decay,"lrScheduler":format!("{:?}",request.config.lr_scheduler),"lrWarmupSteps":request.config.lr_warmup_steps,"saveEvery":request.config.save_every,"resolution":request.config.resolution,"networkType":format!("{:?}",request.config.network_type),"decomposeFactor":request.config.decompose_factor,"loraTargetModules":request.config.lora_target_modules,"timestepType":request.config.timestep_type,"timestepBias":request.config.timestep_bias,"lossType":request.config.loss_type,"modelOptions":request.config.model_options,"memoryLimitBytes":request.config.memory_limit_bytes,"memoryLimitFraction":request.config.memory_limit_fraction,"resume":request.config.resume});
    case.persist()?;
    if let Err(error) = &result {
        if !matches!(error, sceneworks_worker::WorkerError::Canceled(_)) {
            // This test owns trainer construction, so it performs the same real
            // terminal progress POST the normal outer utility dispatcher owns.
            // Retain the original loader error verbatim; do not turn it into a
            // fabricated native run receipt or a completed model job.
            let (status,terminal)=case.api(app,"POST",&format!("/api/v1/jobs/{job_id}/progress"),json!({"workerId":worker.worker_id,"status":"failed","stage":"failed","progress":1.0,"message":"Owned acceptance execution returned","error":error.to_string()})).await?;
            require(
                status == StatusCode::OK,
                format!("real terminal failure reconciliation failed: {terminal}"),
            )?;
        }
    }
    if !train {
        let terminal = case.get_job(app, &job_id).await?;
        case.record["raw"]["subruns"][label]["job"] = terminal.clone();
        case.persist()?;
        require(
            receipts.len() == receipts_before,
            "header-only operation unexpectedly started native training",
        )?;
        require(
            terminal["status"] == "failed" && terminal["result"]["trainedArtifactExported"] != true,
            "header-only/low-cap job did not terminate without publication",
        )?;
        let outputs = match std::fs::read_dir(&request.output_dir) {
            Ok(entries) => entries
                .map(|e| e.map(|v| v.path()))
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(|e| e.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error.to_string()),
        };
        case.record["raw"]["subruns"][label]["publishedOutputInventory"] = json!(outputs);
        case.persist()?;
        validate_unpublished_training_outputs(&outputs, &submitted["payload"]["plan"])?;
        if let Some(snapshot) = outputs.first() {
            case.record["raw"]["subruns"][label]["submissionConfigSnapshot"] =
                case.retain_file(&format!("runs/{label}/training-config.json"), snapshot)?;
            case.persist()?;
        }
        return Ok((None, observed));
    }
    let job = case.get_job(app, &job_id).await?;
    let receipt = raw_receipts
        .as_array()
        .and_then(|v| v.get(receipts_before))
        .cloned()
        .ok_or("native run receipt missing")?;
    require(
        receipts.len() == receipts_before + 1,
        "one execution must produce exactly one actual native receipt",
    )?;
    let receipt_artifact =
        case.retain_json(&format!("runs/{label}/native-receipt.json"), &receipt)?;
    case.record["subruns"][label] = json!({"jobId":job_id,"nativeReceipt":receipt_artifact});
    case.record["raw"]["subruns"][label]["nativeReceipt"] = receipt_artifact;
    case.record["raw"]["subruns"][label]["job"] = job.clone();
    case.record["raw"]["subruns"][label]["progress"] = progress;
    case.persist()?;
    capture_native_bundle(case, &request, &receipt, label)?;
    control.map_err(|e| e.to_string())?;
    if stall {
        require(
            matches!(result, Err(sceneworks_worker::WorkerError::Canceled(_))),
            "durable-checkpoint API cancellation did not cancel the real training execution",
        )?;
        require(
            job["status"] == "canceled"
                && job["result"]["resumeCheckpointStep"] == 20
                && job["result"]["trainedArtifactExported"] != true,
            "canceled API result published a final or lost checkpoint20",
        )?;
    } else {
        result.map_err(|e| e.to_string())?;
        require(
            job["status"] == "completed",
            format!("training did not complete: {job}"),
        )?;
    }
    validate_receipt(&receipt, probe, stall)?;
    let steps: Vec<u32> = ledger.lock().map_err(|e| e.to_string())?.steps.clone();
    let start = if request.config.resume { 21 } else { 1 };
    let end = if stall { 20 } else { 60 };
    require(
        steps == (start..=end).collect::<Vec<_>>(),
        "native optimizer updates were skipped, replayed, or changed horizon",
    )?;
    Ok((
        Some(Run {
            job,
            receipt,
            ledger,
            request,
        }),
        observed,
    ))
}

fn capture_native_bundle(
    case: &mut Case,
    request: &gen_core::TrainingRequest,
    receipt: &Value,
    label: &str,
) -> Result<()> {
    let Some(checkpoint) = receipt["checkpoints"].as_array().and_then(|v| v.last()) else {
        return Ok(());
    };
    let step = checkpoint["state"]["step"]
        .as_u64()
        .ok_or("native checkpoint step missing")?;
    let step = u32::try_from(step).map_err(|e| e.to_string())?;
    let bundle = gen_core::moss_image_training::recovery::resume_bundle_path(
        &request.output_dir,
        &request.file_name,
        step,
    )
    .map_err(|e| e.to_string())?;
    let state = gen_core::moss_image_training::recovery::bundle::read(&bundle)
        .map_err(|e| e.to_string())?;
    let mut members = BTreeMap::new();
    for entry in std::fs::read_dir(&bundle).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        require(
            entry.file_type().map_err(|e| e.to_string())?.is_file(),
            "durable native bundle contains a nonregular member",
        )?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or("native bundle filename not UTF8")?
            .to_owned();
        members.insert(
            name.clone(),
            case.retain_file(
                &format!("recovery/{label}/step-{step}/{name}"),
                &entry.path(),
            )?,
        );
    }
    let expected = text(&checkpoint["factor_delta"], "final_sha256")?;
    let mut adapter = None;
    for entry in std::fs::read_dir(&request.output_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_file()
            && entry.path().extension().is_some_and(|v| v == "safetensors")
            && file_sha(&entry.path())? == expected
        {
            adapter = Some(case.retain_file(
                &format!("exports/{label}-checkpoint-{step}.safetensors"),
                &entry.path(),
            )?);
            break;
        }
    }
    let start = PathBuf::from(text(&receipt["run_start_adapter"], "path")?);
    let start_artifact = case.retain_file(
        &format!("exports/{label}-observed-start.safetensors"),
        &start,
    )?;
    case.record["raw"]["subruns"][label]["durableCheckpoint"] =
        json!({"state":state,"members":members,"adapter":adapter,"startExport":start_artifact});
    case.persist()?;
    require(
        adapter.is_some(),
        "actual durable checkpoint adapter bytes are missing",
    )
}

fn validate_receipt(receipt: &Value, probe: &TrainingProbe, canceled: bool) -> Result<()> {
    require(
        receipt["cpu_reference"] == false && receipt["run_scope_returned"] == true,
        "real GPU native scope did not return",
    )?;
    require(
        receipt["evaluation_manifest"] == probe.evaluation().identity,
        "native evaluation source manifest changed",
    )?;
    require(
        receipt["gradient_tensors"].as_u64().is_some_and(|n| n > 0)
            && receipt["gradients_finite"] == true
            && receipt["nonzero_gradient"] == true,
        "finite nonzero measured native gradients missing",
    )?;
    let frozen = receipt["frozen"]
        .as_object()
        .ok_or("native frozen receipts absent")?;
    require(
        !frozen.is_empty(),
        "native frozen component observations empty",
    )?;
    if !canceled {
        for component in ["backbone", "text_encoder", "vae"] {
            require(
                frozen.contains_key(component),
                format!("required measured frozen component absent: {component}"),
            )?;
        }
    }
    for (name, value) in frozen {
        require(
            value["before"].as_str().is_some_and(|s| !s.is_empty())
                && value["before"] == value["after"],
            format!("frozen component changed: {name}"),
        )?;
    }
    let delta = if canceled {
        &receipt["checkpoints"]
            .as_array()
            .and_then(|v| v.last())
            .ok_or("canceled native checkpoint receipt missing")?["factor_delta"]
    } else {
        &receipt["final_parameter_delta"]
    };
    for field in ["tensors", "elements", "changed_elements"] {
        positive(delta, field)?;
    }
    for field in ["max_abs_delta", "squared_l2_delta"] {
        require(
            delta[field]
                .as_f64()
                .is_some_and(|v| v.is_finite() && v > 0.0),
            format!("nonpositive numeric native factor delta {field}"),
        )?;
    }
    // Dynamic parsing deliberately compiles against the old pin. Missing optional
    // diagnostics fail preflight, never produce a fake measurement or a skip.
    let pair = receipt
        .get("prepared_evaluation")
        .filter(|v| v.is_object())
        .ok_or("linked engine did not populate actual prepared_evaluation")?;
    let before = &pair["before"];
    require(before["sigma"] == 0.375, "actual evaluation sigma changed")?;
    let before_loss = before["loss"]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or("actual fixed before loss missing/nonfinite")?;
    require(
        receipt["evaluation_before"].as_f64() == Some(before_loss),
        "scalar loss detached from actual prepared measurement",
    )?;
    let tensors = before["tensors"]
        .as_object()
        .ok_or("actual prepared tensors absent")?;
    for name in ["latent", "context", "noise"] {
        require(
            tensors.contains_key(name),
            format!("actual evaluation tensor {name} absent"),
        )?;
    }
    for (name, tensor) in tensors {
        require(
            tensor["content_encoding"] == "logical-row-major-f32-le-v1",
            format!("unsupported actual content encoding for {name}"),
        )?;
        let shape = tensor["shape"]
            .as_array()
            .ok_or("prepared tensor shape missing")?;
        let strides = tensor["strides"]
            .as_array()
            .ok_or("prepared tensor strides missing")?;
        text(tensor, "dtype")?;
        require(
            !shape.is_empty()
                && shape.iter().all(|v| v.as_u64().is_some_and(|n| n > 0))
                && strides.len() == shape.len()
                && strides.iter().all(|v| v.as_u64().is_some())
                && tensor["content_sha256"].as_str().is_some_and(|v| {
                    v.len() == 64
                        && v.bytes()
                            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                }),
            "incomplete measured prepared tensor descriptor",
        )?;
    }
    if canceled {
        require(
            pair["after"].is_null() && receipt["complete"] == false,
            "canceled run must not claim completed learning",
        )?;
    } else {
        let after = &pair["after"];
        let after_loss = after["loss"]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("actual fixed after loss missing/nonfinite")?;
        require(
            after["tensors"] == before["tensors"] && after["sigma"] == before["sigma"],
            "before/after did not reuse the same actual prepared tensor package",
        )?;
        require(
            receipt["evaluation_after"].as_f64() == Some(after_loss) && after_loss < before_loss,
            "strict actual heldout learning criterion after < before failed",
        )?;
        require(receipt["complete"] == true, "native receipt incomplete")?;
    }
    Ok(())
}

async fn import_adapter(
    case: &mut Case,
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    project: &str,
    source: &Path,
    label: &str,
) -> Result<Value> {
    case.check_clock()?;
    let hash = file_sha(source)?;
    let incoming = worker.data_dir.join("loras/incoming");
    std::fs::create_dir_all(&incoming).map_err(|e| e.to_string())?;
    let destination = incoming.join(format!("{label}-{}.safetensors", uuid::Uuid::new_v4()));
    std::fs::copy(source, &destination).map_err(|e| e.to_string())?;
    require(
        file_sha(&destination)? == hash,
        "transfer changed adapter bytes",
    )?;
    let (status,job)=case.api(app,"POST","/api/v1/loras/import",json!({"sourcePath":destination,"expectedSha256":hash,"baseModel":case.base(),"scope":"project","projectId":project,"name":label})).await?;
    require(
        status == StatusCode::CREATED,
        format!("adapter import submission failed: {job}"),
    )?;
    let id = text(&job, "id")?.to_owned();
    let (import_worker, registration) = native::register_acceptance_cpu_utility_worker(worker)
        .await
        .map_err(|e| e.to_string())?;
    case.record["raw"]["utilityWorkerRegistrations"][label] =
        serde_json::to_value(registration).map_err(|e| e.to_string())?;
    case.persist()?;
    let claimed = case.claim(app, &import_worker, &id).await?;
    let execution = utility(case, &import_worker, claimed).await;
    let completed = case.get_job(app, &id).await?;
    case.record["raw"]["utilityTerminal"][&id] = completed.clone();
    case.persist()?;
    execution?;
    require(
        completed["status"] == "completed"
            && completed["workerId"] == import_worker.worker_id
            && completed["assignedGpu"] == "cpu",
        format!("real CPU import utility job failed: {completed}"),
    )?;
    let (status, library) = case
        .api(
            app,
            "GET",
            &format!("/api/v1/loras?projectId={project}"),
            Value::Null,
        )
        .await?;
    require(
        status == StatusCode::OK,
        format!("library read failed: {library}"),
    )?;
    let items = library
        .as_array()
        .or_else(|| library["loras"].as_array())
        .ok_or("library response has no entries")?;
    let entry = items
        .iter()
        .find(|v| v["name"] == label)
        .cloned()
        .ok_or("completed import did not register actual library entry")?;
    let installed = PathBuf::from(text(&entry, "installedPath")?);
    require(
        installed.is_absolute(),
        "library installed path is not absolute",
    )?;
    let resolved = if installed.is_dir() {
        installed.join(
            destination
                .file_name()
                .ok_or("incoming adapter basename missing")?,
        )
    } else {
        installed
    };
    require(
        file_sha(&resolved)? == hash,
        "library resolution selected different adapter bytes",
    )?;
    case.record["raw"]["imports"][label] =
        json!({"job":completed,"library":entry,"sourceSha256":hash,"resolvedPath":resolved});
    case.persist()?;
    Ok(entry)
}

pub(super) fn resolve_training_library_adapter(entry: &Value, output: &Path) -> Result<PathBuf> {
    let installed = PathBuf::from(text(entry, "installedPath")?);
    require(
        installed.is_absolute(),
        "training library installed path is not absolute",
    )?;
    let declared = entry["files"]
        .as_array()
        .and_then(|files| files.first())
        .and_then(Value::as_str);
    let selected = if installed.is_dir() {
        sceneworks_core::lora_family::resolve_adapter_in_dir(&installed, declared)
            .ok_or("training library directory does not resolve an adapter")?
    } else {
        installed
    };
    let resolved = std::fs::canonicalize(&selected).map_err(|e| e.to_string())?;
    let expected = std::fs::canonicalize(output).map_err(|e| e.to_string())?;
    require(
        resolved.is_file() && resolved == expected,
        "library installed path does not resolve final adapter",
    )?;
    Ok(resolved)
}

async fn training_library_adapter(
    case: &mut Case,
    app: &axum::Router,
    project: &str,
    run: &Run,
) -> Result<Value> {
    let (status, library) = case
        .api(
            app,
            "GET",
            &format!("/api/v1/loras?projectId={project}"),
            Value::Null,
        )
        .await?;
    require(
        status == StatusCode::OK,
        "actual training library read failed",
    )?;
    let id = text(&run.job["result"], "loraId")?;
    let entry = library
        .as_array()
        .ok_or("library is not an array")?
        .iter()
        .find(|v| v["id"] == id)
        .cloned()
        .ok_or("real completed trainer did not register its library entry")?;
    let output = PathBuf::from(text(&run.job["result"], "outputPath")?);
    require(
        entry["baseModel"] == case.base()
            && entry["networkType"] == case.method
            && entry["source"]["provider"] == "training",
        "training library parent/method/source mismatch",
    )?;
    require(
        std::fs::canonicalize(&output).map_err(|e| e.to_string())?
            == std::fs::canonicalize(text(&run.receipt, "final_native_path")?)
                .map_err(|e| e.to_string())?
            && file_sha(&output)?
                == file_sha(Path::new(text(&run.receipt, "final_adapter_path")?))?,
        "actual library completion is not the final native-produced adapter",
    )?;
    resolve_training_library_adapter(&entry, &output)?;
    require(
        entry["provenance"]["configSnapshot"]["alpha"] == 6.0
            && entry["provenance"]["configSnapshot"]["rank"] == 4
            && entry["provenance"]["configSnapshot"]["seed"] == 1789,
        "library provenance changed sealed training controls",
    )?;
    case.record["raw"]["trainingLibraryRegistration"] =
        json!({"entry":entry,"outputPath":output,"sha256":file_sha(&output)?});
    case.persist()?;
    Ok(entry)
}

/// The sealed MOSS execution report attests the DMD stack, not user adapters.
/// Require the actual provider apply report joined to this exact selected custom file.
pub(super) fn require_custom_adapter_application(
    raw: &Value,
    id: &str,
    spec: &gen_core::AdapterSpec,
    sha256: &str,
) -> Result<()> {
    let rows = raw["adapterApply"]
        .as_array()
        .ok_or("custom adapterApply evidence missing")?;
    let selected: Vec<_> = rows.iter().filter(|row| row["id"] == id).collect();
    require(
        selected.len() == 1,
        "custom adapterApply identity missing or duplicated",
    )?;
    let row = selected[0];
    let kind = match spec.kind {
        gen_core::AdapterKind::Lora => "lora",
        gen_core::AdapterKind::Lokr => "lokr",
    };
    require(row["path"].as_str().is_some_and(|path| Path::new(path) == spec.path)
        && row["sha256"] == sha256 && row["scale"].as_f64() == Some(f64::from(spec.scale))
        && row["kind"] == kind && row["applied"].as_u64().is_some_and(|n| n > 0)
        && row["skipped"].as_array().is_some_and(|skipped| skipped.is_empty()),
        format!("selected CUSTOM library adapter {id} lacks exact successful provider application evidence"))
}

// The validated receipt already requires the exact mode-specific asset inventory.
// Full FP32 embeds time/LoRA in the merged transformer; only unmerged uses separate files.
pub(super) fn require_dmd_component_roles(
    mode: &str,
    roles: impl IntoIterator<Item = gen_core::moss_image::MossImageAssetRole>,
) -> Result<()> {
    use gen_core::moss_image::MossImageAssetRole;
    let roles: BTreeSet<_> = roles.into_iter().collect();
    let required: &[MossImageAssetRole] = match mode {
        "off" => &[],
        "full_fp32" => &[MossImageAssetRole::DmdDataspace],
        "unmerged_bf16" => &[
            MossImageAssetRole::DmdDataspace,
            MossImageAssetRole::DmdTime,
            MossImageAssetRole::DmdLora,
        ],
        _ => return Err("unknown DMD execution mode".into()),
    };
    require(
        required.iter().all(|role| roles.contains(role)),
        "actual DMD execution omitted accelerator component weight receipts",
    )
}

async fn generate(
    case: &mut Case,
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    project: &str,
    label: &str,
    adapter: Option<(&Value, f64)>,
    mode: &str,
) -> Result<Value> {
    generate_with_controls(
        case,
        app,
        worker,
        project,
        label,
        adapter,
        GenerationControls {
            mode,
            seed: 1789,
            advanced: generation_advanced(mode)?,
        },
    )
    .await
}

pub(super) struct GenerationControls<'a> {
    pub(super) mode: &'a str,
    pub(super) seed: u64,
    pub(super) advanced: Value,
}

pub(super) async fn generate_with_controls(
    case: &mut Case,
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    project: &str,
    label: &str,
    adapter: Option<(&Value, f64)>,
    controls: GenerationControls<'_>,
) -> Result<Value> {
    let GenerationControls {
        mode,
        seed,
        advanced,
    } = controls;
    case.check_clock()?;
    let loras = adapter
        .map(|(entry, weight)| json!([{"id":entry["id"],"weight":weight}]))
        .unwrap_or_else(|| json!([]));
    let payload = json!({"projectId":project,"model":case.base(),"prompt":PROMPT,"width":256,"height":256,"count":1,"seed":seed,"loras":loras,"advanced":advanced});
    let (status, job) = case.api(app, "POST", "/api/v1/image/jobs", payload).await?;
    if case.parent == "base" && mode != "off" {
        case.record["raw"]["images"][label] = json!({"httpStatus":status.as_u16(),"refusal":job});
        case.persist()?;
        require(
            status == StatusCode::BAD_REQUEST && job.to_string().to_lowercase().contains("sft"),
            format!("Base+DMD must refuse its exact parent, not missing weights: {job}"),
        )?;
        return Ok(job);
    }
    require(
        status == StatusCode::CREATED,
        format!("image submission: {job}"),
    )?;
    let id = text(&job, "id")?.to_owned();
    let snapshot: sceneworks_core::contracts::JobSnapshot =
        serde_json::from_value(job.clone()).map_err(|e| e.to_string())?;
    let cancel_file = case.cancel_file()?;
    let preload = (|| -> Result<Value> {
        let value = native::moss_generation_preload_with_check(worker, &snapshot, &mut || {
            case.check_control(cancel_file.as_deref())
        })
        .map_err(|e| e.to_string())?;
        require(
            value["selectedMode"] == mode && value["parent"] == case.parent,
            "actual generation preload changed selected mode/parent",
        )?;
        require(positive(&value,"requiredBytes")?<=positive(&case.record["raw"]["devicePreflight"],"resolvedCapBytes")?,"actual exported-adapter generation header demand exceeds unchanged frozen cap; refusing before model loading")?;
        Ok(value)
    })();
    let preload_artifact=case.retain_json(&format!("preload/{label}-actual-adapters.json"),&json!({"jobId":id,"actualSelectedHeaderDemand":preload.as_ref().ok(),"error":preload.as_ref().err(),"resolvedCapBytes":case.record["raw"]["devicePreflight"]["resolvedCapBytes"]}))?;
    case.record["raw"]["imagePreloads"][label] = preload_artifact;
    case.persist()?;
    if preload.is_err() {
        case.api(
            app,
            "POST",
            &format!("/api/v1/jobs/{id}/cancel"),
            Value::Null,
        )
        .await?;
    }
    preload?;
    let claimed = case.claim(app, worker, &id).await?;
    let execution = utility(case, worker, claimed).await;
    let completed = case.get_job(app, &id).await?;
    case.record["raw"]["utilityTerminal"][&id] = completed.clone();
    case.persist()?;
    execution?;
    let receipt = case.retain_json(&format!("images/{label}-job.json"), &completed)?;
    case.record["raw"]["images"][label] = json!({"jobReceipt":receipt});
    case.persist()?;
    require(
        completed["status"] == "completed",
        format!("real image worker did not complete: {completed}"),
    )?;
    let assets = completed["result"]["assets"]
        .as_array()
        .ok_or("completed image job has no actual assets")?;
    require(
        assets.len() == 1
            && completed["result"]["assetIds"]
                .as_array()
                .is_some_and(|v| v.len() == 1),
        "image job returned unexpected asset count",
    )?;
    let asset = &assets[0];
    let report = asset["recipe"]["rawAdapterSettings"]["mossImageExecutionReport"].clone();
    let typed: gen_core::moss_image::MossImageExecutionReport =
        serde_json::from_value(report.clone())
            .map_err(|e| format!("actual completed MOSS execution receipt invalid: {e}"))?;
    let data = typed.data();
    if case.record["recordKind"] == "bounded-product-acceptance-evidence" {
        require(
            data.resolved.variant == gen_core::moss_image::MossImageVariant::Base
                && data.resolved.steps == 2
                && data.resolved.guidance == 1.0
                && data.samples.len() == 1
                && data.samples[0].seed == Some(7)
                && data.images_per_prompt == 1
                && data.conditioning.len() == 1
                && data.conditioning[0].positive.sha256 == sha(PROMPT.as_bytes()),
            "bounded native sample changed Base, steps, guidance, seed, count or caption",
        )?;
    }
    let execution = serde_json::to_value(data.execution_device).map_err(|e| e.to_string())?;
    require(
        execution == case.backend,
        "native generation did not execute on the selected real GPU backend",
    )?;
    let acceleration =
        serde_json::to_value(data.resolved.acceleration).map_err(|e| e.to_string())?;
    require(
        acceleration == mode,
        "selected DMD/ordinary mode was not executed",
    )?;
    if let Some((entry, strength)) = adapter {
        let selected = native::prepare_selected_adapters(
            worker,
            completed["payload"]
                .as_object()
                .ok_or("actual executed image payload missing")?,
        )
        .map_err(|e| e.to_string())?;
        selected.ensure_unchanged().map_err(|e| e.to_string())?;
        if strength != 0.0 {
            let spec = selected
                .specs()
                .iter()
                .find(|s| f64::from(s.scale) == strength)
                .ok_or("actual production selection lost requested adapter strength")?;
            require_custom_adapter_application(
                &asset["recipe"]["rawAdapterSettings"],
                text(entry, "id")?,
                spec,
                &file_sha(&spec.path)?,
            )?;
        } else {
            require(
                selected.specs().iter().all(|s| s.scale == 0.0),
                "disable control retained a nonzero selected custom adapter",
            )?;
        }
    }
    require_dmd_component_roles(mode, data.assets.iter().map(|asset| asset.role))?;
    for asset in &data.assets {
        require(
            gen_core::moss_image::MOSS_IMAGE_ASSETS
                .iter()
                .any(|identity| {
                    identity.repository == asset.repository
                        && identity.revision == asset.revision
                        && identity.relative_path == asset.relative_path
                        && identity.sha256 == asset.sha256
                }),
            "actual consumed generation asset differs from frozen published source identity",
        )?;
    }
    let path = text(&asset["file"], "path")?;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/projects/{project}/files/{path}"))
                .body(Body::empty())
                .map_err(|e| e.to_string())?,
        )
        .await
        .map_err(|e| e.to_string())?;
    require(response.status() == StatusCode::OK, "actual PNG GET failed")?;
    let bytes = to_bytes(response.into_body(), 32 * 1024 * 1024)
        .await
        .map_err(|e| e.to_string())?;
    require(
        bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "worker output is not an actual PNG",
    )?;
    let decoded = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
    require(
        decoded.width() == 256 && decoded.height() == 256,
        "actual generated PNG dimensions changed",
    )?;
    let artifact = case.retain(&format!("images/{label}.png"), &bytes)?;
    let execution_receipt = case.retain_json(&format!("images/{label}-execution.json"), &report)?;
    let result = json!({"artifact":artifact,"pixelSha256":sha(decoded.to_rgb8().as_raw()),"executionReceipt":execution_receipt,"jobReceipt":receipt,"adapterStrength":adapter.map(|(_,v)|v),"prompt":PROMPT,"seed":seed,"width":256,"height":256});
    case.record["raw"]["images"][label] = result.clone();
    case.persist()?;
    case.check_clock()?;
    Ok(result)
}

fn retain_run_exports(case: &mut Case, run: &Run, label: &str) -> Result<(PathBuf, PathBuf)> {
    let start = PathBuf::from(text(&run.receipt["run_start_adapter"], "path")?);
    let final_path = PathBuf::from(text(&run.receipt, "final_adapter_path")?);
    let start_artifact = case.retain_file(&format!("exports/{label}-start.safetensors"), &start)?;
    let final_artifact =
        case.retain_file(&format!("exports/{label}-final.safetensors"), &final_path)?;
    require(
        start_artifact["sha256"] == run.receipt["run_start_adapter"]["sha256"],
        "run-start export hash differs from actual native receipt",
    )?;
    require(
        start_artifact["sha256"] == run.receipt["final_parameter_delta"]["initial_sha256"]
            && final_artifact["sha256"] == run.receipt["final_parameter_delta"]["final_sha256"],
        "positive numeric factor delta is not bound to retained native start/final exports",
    )?;
    case.record["raw"]["subruns"][label]["startExport"] = start_artifact;
    case.record["raw"]["subruns"][label]["finalExport"] = final_artifact;
    case.persist()?;
    Ok((start, final_path))
}

/// Exact logical equality, including ALL metadata. JSON key order and physical
/// tensor placement are representation details; dtype/shape and every payload
/// bit remain exact. Original artifacts and raw hashes are never rewritten.
pub(super) fn exact_safetensors_equal(left: &Path, right: &Path) -> Result<()> {
    fn header(path: &Path) -> Result<(Value, u64)> {
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
        let size = u64::from_le_bytes(prefix);
        require(
            size > 0 && size <= 16 * 1024 * 1024,
            "unbounded safetensors header",
        )?;
        require(
            sceneworks_core::safetensors::file_is_structurally_valid(path),
            "invalid recovery safetensors structure",
        )?;
        let mut encoded = vec![0u8; size as usize];
        file.read_exact(&mut encoded).map_err(|e| e.to_string())?;
        let header: Value = serde_json::from_slice(&encoded).map_err(|e| e.to_string())?;
        if let Some(metadata) = header.get("__metadata__") {
            require(
                metadata
                    .as_object()
                    .is_some_and(|values| values.values().all(Value::is_string)),
                "invalid safetensors metadata",
            )?;
        }
        Ok((header, 8 + size))
    }
    let (left_header, left_start) = header(left)?;
    let (right_header, right_start) = header(right)?;
    let left_entries = left_header
        .as_object()
        .ok_or("safetensors header is not an object")?;
    let right_entries = right_header
        .as_object()
        .ok_or("safetensors header is not an object")?;
    require(
        left_entries.keys().collect::<BTreeSet<_>>()
            == right_entries.keys().collect::<BTreeSet<_>>(),
        "recovery tensor/metadata keys differ",
    )?;
    require(
        left_header.get("__metadata__") == right_header.get("__metadata__"),
        "recovery safetensors metadata differs",
    )?;
    let mut a = std::fs::File::open(left).map_err(|e| e.to_string())?;
    let mut b = std::fs::File::open(right).map_err(|e| e.to_string())?;
    let mut a_bytes = [0u8; 65536];
    let mut b_bytes = [0u8; 65536];
    for (name, descriptor) in left_entries {
        if name == "__metadata__" {
            continue;
        }
        let other = &right_entries[name];
        let mut x = descriptor.clone();
        let mut y = other.clone();
        x.as_object_mut()
            .ok_or("invalid tensor descriptor")?
            .remove("data_offsets");
        y.as_object_mut()
            .ok_or("invalid tensor descriptor")?
            .remove("data_offsets");
        require(
            x == y,
            format!("recovery tensor descriptor differs: {name}"),
        )?;
        // Structural validation above authenticates each contiguous range.
        let offsets = |v: &Value| -> Result<(u64, u64)> {
            Ok((
                v["data_offsets"][0]
                    .as_u64()
                    .ok_or("missing tensor start")?,
                v["data_offsets"][1].as_u64().ok_or("missing tensor end")?,
            ))
        };
        let (start, end) = offsets(descriptor)?;
        let (other_start, other_end) = offsets(other)?;
        require(
            end - start == other_end - other_start,
            "recovery tensor byte length differs",
        )?;
        a.seek(SeekFrom::Start(left_start + start))
            .map_err(|e| e.to_string())?;
        b.seek(SeekFrom::Start(right_start + other_start))
            .map_err(|e| e.to_string())?;
        let mut remaining = end - start;
        while remaining > 0 {
            let count = remaining.min(a_bytes.len() as u64) as usize;
            a.read_exact(&mut a_bytes[..count])
                .map_err(|e| e.to_string())?;
            b.read_exact(&mut b_bytes[..count])
                .map_err(|e| e.to_string())?;
            require(
                a_bytes[..count] == b_bytes[..count],
                format!("recovery tensor payload differs: {name}"),
            )?;
            remaining -= count as u64;
        }
    }
    Ok(())
}

fn compare_recovery(case: &mut Case, reference: &Run, resumed: &Run, canceled: &Run) -> Result<()> {
    let reference_bundle = gen_core::moss_image_training::recovery::resume_bundle_path(
        &reference.request.output_dir,
        &reference.request.file_name,
        60,
    )
    .map_err(|e| e.to_string())?;
    let resumed_bundle = gen_core::moss_image_training::recovery::resume_bundle_path(
        &resumed.request.output_dir,
        &resumed.request.file_name,
        60,
    )
    .map_err(|e| e.to_string())?;
    let reference_state = json_file(&reference_bundle.join("run.json"))?;
    let resumed_state = json_file(&resumed_bundle.join("run.json"))?;
    // Native bundle state contains the entire scheduler/RNG/cursor/accumulation
    // state. Compare stored values, not a handpicked summary or optimizer step.
    let state_artifact = case.retain_json(
        "recovery/final-states.json",
        &json!({"reference":reference_state,"resumed":resumed_state}),
    )?;
    case.record["raw"]["recovery"] = json!({"finalStates":state_artifact,"canceledProgress":acceptance_progress_snapshot(&canceled.ledger).map_err(|e|e.to_string())?,"resumedProgress":acceptance_progress_snapshot(&resumed.ledger).map_err(|e|e.to_string())?});
    case.persist()?;
    require(reference_state==resumed_state,"reference/resumed complete native optimizer scheduler RNG cursor accumulation state differs")?;
    // Reauthenticate EACH native manifest against its original raw bytes; the
    // manifest hashes remain raw identities, not cross-run state comparisons.
    for bundle in [&reference_bundle, &resumed_bundle] {
        gen_core::moss_image_training::recovery::bundle::read(bundle).map_err(|e| e.to_string())?;
    }
    // Retain every original raw hash, including the two distinct manifests.
    let inventory = |root: &Path| -> Result<BTreeMap<String, String>> {
        fn walk(root: &Path, path: &Path, result: &mut BTreeMap<String, String>) -> Result<()> {
            for entry in std::fs::read_dir(path).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let ty = entry.file_type().map_err(|e| e.to_string())?;
                require(
                    !ty.is_symlink(),
                    "native resume bundle may not contain symlinks",
                )?;
                if ty.is_dir() {
                    walk(root, &entry.path(), result)?;
                } else {
                    result.insert(
                        entry
                            .path()
                            .strip_prefix(root)
                            .map_err(|e| e.to_string())?
                            .to_string_lossy()
                            .replace('\\', "/"),
                        file_sha(&entry.path())?,
                    );
                }
            }
            Ok(())
        }
        let mut values = BTreeMap::new();
        walk(root, root, &mut values)?;
        Ok(values)
    };
    let a = inventory(&reference_bundle)?;
    let b = inventory(&resumed_bundle)?;
    let artifact = case.retain_json(
        "recovery/bundle-inventory.json",
        &json!({"reference":a,"resumed":b}),
    )?;
    case.record["raw"]["recovery"]["bundleInventory"] = artifact;
    case.persist()?;
    let expected: BTreeSet<_> = [
        "complete.json",
        "run.json",
        "optimizer.safetensors",
        "tensors.safetensors",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    require(
        a.keys().cloned().collect::<BTreeSet<_>>() == expected
            && b.keys().cloned().collect::<BTreeSet<_>>() == expected,
        "native bundle population differs",
    )?;
    for member in ["optimizer.safetensors", "tensors.safetensors"] {
        exact_safetensors_equal(&reference_bundle.join(member), &resumed_bundle.join(member))?;
    }
    exact_safetensors_equal(
        Path::new(text(&reference.receipt, "final_adapter_path")?),
        Path::new(text(&resumed.receipt, "final_adapter_path")?),
    )?;
    require(
        reference.receipt["evaluation_after"] == resumed.receipt["evaluation_after"]
            && reference.job["result"]["finalLoss"] == resumed.job["result"]["finalLoss"],
        "actual reference/resumed numerical results differ",
    )?;
    let old = canceled.ledger.lock().map_err(|e| e.to_string())?;
    let new = resumed.ledger.lock().map_err(|e| e.to_string())?;
    require(
        old.observed_cancel
            .as_ref()
            .is_some_and(|f| f.is_cancelled())
            && new
                .observed_cancel
                .as_ref()
                .is_some_and(|f| !f.is_cancelled())
            && old.token_id != new.token_id,
        "resume did not retain canceled original and independently uncanceled request token",
    )?;
    Ok(())
}

async fn forecast_generation(
    case: &mut Case,
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    project: &str,
) -> Result<Value> {
    let mut forecasts = serde_json::Map::new();
    for mode in if case.parent == "sft" && !case.cross {
        vec!["off", "unmerged_bf16", "full_fp32"]
    } else {
        vec!["off"]
    } {
        case.check_clock()?;
        let cancel_file = case.cancel_file()?;
        let payload = json!({"projectId":project,"model":case.base(),"prompt":PROMPT,"width":256,"height":256,"count":1,"seed":1789,"loras":[],"advanced":generation_advanced(mode)?});
        let (status, job) = case.api(app, "POST", "/api/v1/image/jobs", payload).await?;
        require(
            status == StatusCode::CREATED,
            format!("native generation forecast submission failed: {job}"),
        )?;
        let id = text(&job, "id")?.to_owned();
        let snapshot = serde_json::from_value(job.clone()).map_err(|e| e.to_string());
        let forecast = snapshot.and_then(|job| {
            native::moss_generation_preload_with_check(worker, &job, &mut || {
                case.check_control(cancel_file.as_deref())
            })
            .map_err(|e| e.to_string())
        });
        // No generator is constructed and this queued forecast is never claimed.
        // The real cancellation route removes it from the owned queue even when
        // a header/source validation fails.
        let (cancel_status, canceled) = case
            .api(
                app,
                "POST",
                &format!("/api/v1/jobs/{id}/cancel"),
                Value::Null,
            )
            .await?;
        let raw = json!({"job":job,"headerForecast":forecast.as_ref().ok(),"error":forecast.as_ref().err(),"cancellation":canceled,"futureAdapterIncluded":false,"adapterSpecificCheckRequiredBeforeEveryActualLoad":true});
        let artifact = case.retain_json(&format!("preload/base-{mode}.json"), &raw)?;
        case.record["raw"]["generationForecasts"][mode] =
            json!({"receipt":artifact,"observation":raw});
        case.persist()?;
        require(
            cancel_status == StatusCode::OK,
            format!("unclaimed forecast job cancellation failed: {canceled}"),
        )?;
        forecasts.insert(mode.into(), forecast?);
    }
    Ok(Value::Object(forecasts))
}

/// Training plus each known selected base/DMD generation phase uses its actual
/// production header routine. Future adapters are NOT synthesized or estimated
/// here: every real generation repeats this same pre-load check against its
/// actual exported adapter before dispatch, under the unchanged frozen ceiling.
pub(super) fn preflight_verdict(
    metadata: &Value,
    weights: &Value,
    backend: &str,
    parent: &str,
    generation_forecasts: &Value,
    cross: bool,
) -> Result<u64> {
    require(weights.as_array().is_some_and(|v|!v.is_empty() && v.iter().all(|w|w["present"]==true && w["trustedHashMatches"]==true)),"official cached MOSS files missing or content hash differs from native frozen acquisition identities; install/repair the exact approved cache offline before admission")?;
    let mut demand = if cross {
        0
    } else {
        for field in [
            "gradientObservations",
            "fixedEvaluationPair",
            "numericFactorDelta",
            "preparedTensorSnapshots",
        ] {
            require(metadata["observationSchema"][field]==true,format!("linked native source lacks required diagnostic {field}; merge and pin the native hook before this campaign"))?;
        }
        let plan = &metadata["nativePlan"];
        let demand = positive(plan, "tensor_peak_bytes")?
            .checked_add(
                plan["host_staging_peak_bytes"]
                    .as_u64()
                    .ok_or("native staging demand missing")?,
            )
            .ok_or("native demand overflow")?
            .max(positive(plan, "encoder_only_peak_bytes")?);
        for phase in [
            "load_bytes",
            "cast_bytes",
            "cache_bytes",
            "backward_bytes",
            "preview_bytes",
            "checkpoint_bytes",
        ] {
            require(
                plan.get(phase).and_then(Value::as_u64).is_some(),
                format!("native training phase demand missing: {phase}"),
            )?;
        }
        demand
    };
    let phases = generation_forecasts
        .as_object()
        .ok_or("actual selected generation header forecasts are missing")?;
    for phase in if parent == "sft" && !cross {
        vec!["off", "unmerged_bf16", "full_fp32"]
    } else {
        vec!["off"]
    } {
        let actual = phases
            .get(phase)
            .ok_or_else(|| format!("native generation phase header demand missing: {phase}"))?;
        demand = demand.max(positive(actual, "requiredBytes")?);
        text(actual, "provenance")?;
    }
    require(
        metadata["backend"] == backend,
        "native selected device backend mismatch",
    )?;
    require(
        demand <= positive(metadata, "resolvedCapBytes")?,
        "native selected-phase demand exceeds the resolved product ceiling",
    )?;
    Ok(demand)
}

pub(super) fn claimed_device_identity(worker: &sceneworks_worker::Settings) -> Result<String> {
    if cfg!(target_os = "macos") {
        require(
            worker.gpu_id == "mlx",
            "production registration did not claim MLX",
        )?;
        return Ok("mlx".into());
    }
    // Production NVIDIA discovery registers the numeric nvidia-smi index. Bind
    // that exact claimed index to a measured UUID, then compare the UUID against
    // the allocator-selected CUDA context reported by the native header probe.
    let mut child = std::process::Command::new("nvidia-smi")
        .args([
            "--id",
            &worker.gpu_id,
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot authenticate claimed CUDA device UUID: {e}"))?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(_) => break,
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            None => {
                // Only this owned metadata helper is terminated; never a model,
                // GPU native task, worker process, or unrelated process tree.
                let _ = child.kill();
                let _ = child.wait();
                return Err("owned NVIDIA UUID metadata query exceeded five seconds".into());
            }
        }
    }
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    require(
        output.status.success(),
        format!(
            "claimed-device UUID metadata query refused: {}",
            String::from_utf8_lossy(&output.stderr)
        ),
    )?;
    let uuid = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let uuid = uuid.trim();
    require(
        uuid.starts_with("GPU-") && !uuid.contains(char::is_whitespace),
        "claimed device did not resolve exactly one real CUDA UUID",
    )?;
    Ok(uuid.into())
}

async fn body(
    case: &mut Case,
    app: &axum::Router,
    settings: &crate::Settings,
    worker: &sceneworks_worker::Settings,
) -> Result<()> {
    let (weights, revisions) = weight_inventory(settings, case)?;
    let (status, project) = case
        .api(
            app,
            "POST",
            "/api/v1/projects",
            json!({"name":format!("MOSS manual {}",case.id)}),
        )
        .await?;
    require(
        status == StatusCode::CREATED,
        format!("project creation failed: {project}"),
    )?;
    let project = text(&project, "id")?.to_owned();
    // Registration uses the real health/discovery/capability chain, never POSTs
    // a handwritten gpu/lora_train_execute capability fixture.
    let registered = native::register_real_acceptance_worker(worker)
        .await
        .map_err(|e| e.to_string())?;
    let mut selected_worker = worker.clone();
    selected_worker.gpu_id = registered.gpu_id.clone();
    let worker = &selected_worker;
    case.record["raw"]["workerRegistration"] =
        serde_json::to_value(registered).map_err(|e| e.to_string())?;
    case.persist()?;
    let (metadata, training) = if case.cross {
        (
            native::moss_generation_acceptance_device(worker).map_err(|e| e.to_string())?,
            None,
        )
    } else {
        let (dataset, evaluation) = dataset(case, app, &project).await?;
        let probe_path = case.root.join(if case.preflight_only {
            "preflight/native-probe.json"
        } else {
            "native-probe.json"
        });
        let probe = TrainingProbe::new(evaluation).with_artifact_fallback(probe_path);
        let submitted = submit(case, app, &project, &dataset, "header-preflight").await?;
        let (_, metadata) = execute(
            case,
            app,
            worker,
            &submitted,
            &probe,
            "header-preflight",
            Execution::Preflight,
        )
        .await?;
        (metadata, Some((dataset, probe)))
    };
    let claimed = claimed_device_identity(worker);
    let generation_forecasts = forecast_generation(case, app, worker, &project).await;
    let verdict=claimed.as_ref().map_err(Clone::clone).and_then(|claimed| {
        let actual=text(&metadata,"deviceIdentity")?;
        require(if case.backend=="mlx" {claimed=="mlx" && actual.starts_with("mlx-default-gpu/stream-")}else{actual.eq_ignore_ascii_case(claimed)},"native allocator-selected CUDA context/MLX device differs from actual production worker claim")?;
        preflight_verdict(&metadata, &weights, case.backend, &case.parent,generation_forecasts.as_ref().map_err(Clone::clone)?,case.cross)
    });
    if let Some(ceiling) = metadata["resolvedCapBytes"]
        .as_u64()
        .filter(|value| *value > 0)
    {
        let mut resources = case.resources.lock().map_err(|error| error.to_string())?;
        require(
            resources.ceiling.is_none_or(|previous| previous == ceiling),
            "native root ceiling differs from device preflight",
        )?;
        resources.ceiling = Some(ceiling);
        resources.sample(true)?;
    }
    let preflight = json!({"caseId":case.id,"sourceCaseId":case.id,"backend":case.backend,"executionBackend":case.backend,"phase":if case.cross {"cross-load"} else {"capture"},"workflowSource":env("MOSS_ACCEPTANCE_WORKFLOW_SOURCE")?,"inferencePin":source_pin()?,"sourcePin":source_pin()?,"manifestSha256":SEAL,"offline":true,"weightsLoaded":false,"modelRevisions":revisions,"weights":weights,"claimedDeviceIdentity":claimed.as_ref().ok(),"deviceIdentity":metadata["deviceIdentity"],"totalCapacityBytes":metadata["totalCapacityBytes"],"resolvedCapBytes":metadata["resolvedCapBytes"],"gpuMemoryLimitFraction":case.fraction(),"resolvedCapProvenance":metadata["resolvedCapProvenance"],"headerDemandProvenance":if case.cross {"production-selected ordinary prepare_moss_preload only; actual transferred adapter rechecked before generation under this same frozen cap"} else {"native training test_memory_plan plus production-selected ordinary/DMD prepare_moss_preload; future actual adapters separately rechecked before every real generation under this same frozen cap"},"generationForecasts":generation_forecasts.as_ref().ok(),"generationForecastError":generation_forecasts.as_ref().err(),"futureAdapterIncluded":false,"adapterSpecificCheckRequiredBeforeEveryActualLoad":true,"headerDerivedDemandBytes":verdict.as_ref().ok(),"observationSchema":metadata["observationSchema"],"nativeMetadata":metadata,"verdict":if verdict.is_ok(){"admitted"}else{"refused"},"refusalReason":verdict.as_ref().err()});
    case.record["raw"]["devicePreflight"] = preflight.clone();
    case.persist()?;
    let preflight_path = PathBuf::from(env("MOSS_ACCEPTANCE_DEVICE_PREFLIGHT")?);
    require(
        preflight_path.is_absolute() && preflight_path.starts_with(&case.root),
        "preflight receipt destination is not owned",
    )?;
    if case.preflight_only {
        std::fs::write(
            &preflight_path,
            serde_json::to_vec_pretty(&preflight).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    let demand = verdict?;
    if case.preflight_only {
        return Ok(());
    }
    let approved = json_file(&preflight_path)?;
    require(
        approved["verdict"] == "admitted"
            && approved["phase"] == preflight["phase"]
            && approved["caseId"] == case.id
            && approved["deviceIdentity"] == preflight["deviceIdentity"]
            && approved["resolvedCapBytes"] == preflight["resolvedCapBytes"]
            && approved["inferencePin"] == preflight["inferencePin"],
        "actual device/source preflight changed before model loading",
    )?;
    if case.cross {
        return cross_load(case, app, worker, &project).await;
    }
    let (dataset, probe) = training.ok_or("capture training context missing")?;
    // The deliberately infeasible control uses the production global-cap file,
    // never a trainer override. Always restore the frozen cap before assertions.
    let control_cap = positive(&metadata["nativePlan"], "tensor_peak_bytes")?
        .checked_sub(1)
        .filter(|v| *v > 0)
        .ok_or("invalid native training demand for low-cap control")?;
    set_cap(worker, json!(control_cap))?;
    let low_job = submit(case, app, &project, &dataset, "low-cap").await;
    let low = match low_job {
        Ok(job) => {
            execute(
                case,
                app,
                worker,
                &job,
                &probe,
                "low-cap",
                Execution::LowCap,
            )
            .await
        }
        Err(error) => Err(error),
    };
    set_cap(worker, json!({"fraction":case.fraction()}))?;
    let (_, low_metadata) = low?;
    case.record["raw"]["memoryRefusal"] = json!({"requestedControlCapBytes":control_cap,"nativeResponse":low_metadata,"projectedHeaderDemandBytes":demand,"campaignCeilingRaised":false});
    case.persist()?;
    require(
        low_metadata["refusal"]
            .as_str()
            .is_some_and(|s| s.contains("budget") || s.contains("memory")),
        "strictly infeasible control did not refuse inside real header admission",
    )?;
    let reference_job = submit(case, app, &project, &dataset, "uninterrupted").await?;
    let (reference, _) = execute(
        case,
        app,
        worker,
        &reference_job,
        &probe,
        "uninterrupted",
        Execution::Train { cancel_at20: false },
    )
    .await?;
    let reference = reference.ok_or("reference run absent")?;
    let (untrained, trained) = retain_run_exports(case, &reference, "uninterrupted")?;
    let cancel_job = submit(case, app, &project, &dataset, "canceled").await?;
    let (canceled, _) = execute(
        case,
        app,
        worker,
        &cancel_job,
        &probe,
        "canceled",
        Execution::Train { cancel_at20: true },
    )
    .await?;
    let canceled = canceled.ok_or("canceled run absent")?;
    let canceled_id = text(&cancel_job, "id")?;
    let (status, resumed_job) = case
        .api(
            app,
            "POST",
            &format!("/api/v1/jobs/{canceled_id}/retry"),
            json!({"resume":true}),
        )
        .await?;
    require(
        status == StatusCode::CREATED,
        format!("actual API resume failed: {resumed_job}"),
    )?;
    let mut expected = cancel_job["payload"]["plan"].clone();
    expected["config"]["advanced"]["resume"] = json!(true);
    require(
        resumed_job["payload"]["plan"] == expected,
        "API resume changed horizon/config/dataset identity",
    )?;
    let (resumed, _) = execute(
        case,
        app,
        worker,
        &resumed_job,
        &probe,
        "resumed",
        Execution::Train { cancel_at20: false },
    )
    .await?;
    let resumed = resumed.ok_or("resumed run absent")?;
    retain_run_exports(case, &resumed, "resumed")?;
    compare_recovery(case, &reference, &resumed, &canceled)?;
    if case.parent == "base" && case.method == "lora" {
        let fresh_cancel_job =
            submit(case, app, &project, &dataset, "fresh-reentry-canceled").await?;
        let (old, _) = execute(
            case,
            app,
            worker,
            &fresh_cancel_job,
            &probe,
            "fresh-reentry-canceled",
            Execution::Train { cancel_at20: true },
        )
        .await?;
        let old = old.ok_or("fresh-reentry canceled run absent")?;
        let fresh_job = submit(case, app, &project, &dataset, "fresh-reentry-new-run").await?;
        let (fresh, _) = execute(
            case,
            app,
            worker,
            &fresh_job,
            &probe,
            "fresh-reentry-new-run",
            Execution::Train { cancel_at20: false },
        )
        .await?;
        let fresh = fresh.ok_or("fresh independent run absent")?;
        let old_snapshot = acceptance_progress_snapshot(&old.ledger).map_err(|e| e.to_string())?;
        let new_snapshot =
            acceptance_progress_snapshot(&fresh.ledger).map_err(|e| e.to_string())?;
        let witness = json!({"anchorCaseId":case.id,"original":old_snapshot,"fresh":new_snapshot,"freshJob":fresh.job});
        case.record["raw"]["backendReentryWitness"] =
            case.retain_json("recovery/backend-reentry.json", &witness)?;
        case.persist()?;
        require(
            old_snapshot["tokenObservedCanceled"] == true
                && new_snapshot["tokenObservedCanceled"] == false
                && old_snapshot["tokenId"] != new_snapshot["tokenId"]
                && !fresh.request.config.resume,
            "fresh run reused/cleared the original cancellation token or resumed old state",
        )?;
        exact_safetensors_equal(
            Path::new(text(&fresh.receipt, "final_adapter_path")?),
            &trained,
        )?;
    }
    let initial = probe
        .initial_untrained_adapter()
        .map_err(|e| e.to_string())?
        .ok_or("native initial untrained adapter absent")?;
    require(
        initial.path == untrained,
        "before preview is not the original run-start untrained adapter",
    )?;
    let before_entry =
        import_adapter(case, app, worker, &project, &untrained, "before-untrained").await?;
    let after_entry = training_library_adapter(case, app, &project, &reference).await?;
    // Also round-trip the final file through the real import worker; preview
    // selection above still uses the original production training registration.
    import_adapter(
        case,
        app,
        worker,
        &project,
        &trained,
        "after-trained-import-roundtrip",
    )
    .await?;
    let baseline = generate(
        case,
        app,
        worker,
        &project,
        "no-adapter-baseline",
        None,
        "off",
    )
    .await?;
    generate(
        case,
        app,
        worker,
        &project,
        "before-untrained",
        Some((&before_entry, 1.0)),
        "off",
    )
    .await?;
    generate(
        case,
        app,
        worker,
        &project,
        "after-trained",
        Some((&after_entry, 1.0)),
        "off",
    )
    .await?;
    let disable = generate(
        case,
        app,
        worker,
        &project,
        "disable-control",
        Some((&after_entry, 0.0)),
        "off",
    )
    .await?;
    let replay = generate(
        case,
        app,
        worker,
        &project,
        "post-unload-replay",
        None,
        "off",
    )
    .await?;
    require(
        baseline["pixelSha256"] == disable["pixelSha256"]
            && baseline["pixelSha256"] == replay["pixelSha256"],
        "disable/unload controls do not match real ordinary baseline pixels",
    )?;
    for mode in ["unmerged_bf16", "full_fp32"] {
        generate(
            case,
            app,
            worker,
            &project,
            &format!("dmd-{mode}"),
            Some((&after_entry, 1.0)),
            mode,
        )
        .await?;
    }
    case.record["raw"]["transferAdapter"] =
        case.retain_file("transfer/final.safetensors", &trained)?;
    case.check_clock()?;
    case.persist()
}

async fn cross_load(
    case: &mut Case,
    app: &axum::Router,
    worker: &sceneworks_worker::Settings,
    project: &str,
) -> Result<()> {
    let peers = PathBuf::from(env("MOSS_ACCEPTANCE_PEER_ARTIFACTS")?);
    require(peers.is_absolute(), "peer artifact root must be absolute")?;
    let source_root = peers
        .join(format!("capture-{}", case.source_backend))
        .join(&case.parent)
        .join(&case.method);
    let evidence = json_file(&source_root.join("evidence.json"))?;
    require(
        evidence["caseId"] == case.id
            && evidence["executionBackend"] == case.source_backend
            && evidence["manifestSha256"] == SEAL,
        "cross-load peer source identity mismatch",
    )?;
    let transfer = &evidence["raw"]["transferAdapter"];
    let source = source_root.join(owned_relative(text(transfer, "path")?)?);
    require(
        file_sha(&source)? == text(transfer, "sha256")?,
        "transferred adapter differs from original source case export",
    )?;
    let received_adapter = case.retain_file("transfer/received.safetensors", &source)?;
    let entry = import_adapter(
        case,
        app,
        worker,
        project,
        &source,
        "cross-backend-transferred",
    )
    .await?;
    let image = generate(
        case,
        app,
        worker,
        project,
        "cross-backend-load",
        Some((&entry, 1.0)),
        "off",
    )
    .await?;
    case.record["raw"]["crossBackendLoad"] = json!({"sourceCaseId":case.id,"sourceBackend":case.source_backend,"importedBackend":case.backend,"parent":case.parent,"method":case.method,"receivedSha256":file_sha(&source)?,"transferredAdapter":received_adapter,"sourceAdapter":transfer,"generation":image,"phaseStartedAt":case.started_at,"completedAt":observed_iso_time()?,"wallLimitSeconds":WALL});
    case.record["missing"] = json!(["cleanup", "inspections"]);
    case.check_clock()?;
    case.persist()
}

pub(super) async fn observe_worker_http(
    directory: Arc<PathBuf>,
    sequence: Arc<AtomicU64>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let observation=async {
        let (parts,body)=request.into_parts();
        let method=parts.method.to_string();let route=parts.uri.path().to_owned();
        let request_bytes=to_bytes(body,16*1024*1024).await.map_err(|e|e.to_string())?;
        let request_value:Value=if request_bytes.is_empty(){Value::Null}else{serde_json::from_slice(&request_bytes).map_err(|e|e.to_string())?};
        let response=next.run(axum::extract::Request::from_parts(parts,Body::from(request_bytes.clone()))).await;
        let (parts,body)=response.into_parts();
        let response_bytes=to_bytes(body,32*1024*1024).await.map_err(|e|e.to_string())?;
        let response_value:Value=if response_bytes.is_empty(){Value::Null}else{serde_json::from_slice(&response_bytes).map_err(|e|e.to_string())?};
        let event=json!({"method":method,"route":route,"httpStatus":parts.status.as_u16(),"request":request_value,"response":response_value,"requestBodySha256":sha(&request_bytes),"responseBodySha256":sha(&response_bytes),"atUnixMs":unix_ms()?});
        let path=directory.join(format!("{:06}.json",sequence.fetch_add(1,Ordering::Relaxed)));
        let mut output=tokio::fs::OpenOptions::new().create_new(true).write(true).open(path).await.map_err(|e|e.to_string())?;
        tokio::io::AsyncWriteExt::write_all(&mut output,&serde_json::to_vec_pretty(&event).map_err(|e|e.to_string())?).await.map_err(|e|e.to_string())?;
        output.sync_all().await.map_err(|e|e.to_string())?;
        Ok::<_,String>(axum::response::Response::from_parts(parts,Body::from(response_bytes)))
    }.await;
    match observation {
        Ok(response) => response,
        Err(error) => {
            eprintln!("owned acceptance HTTP evidence capture failed: {error}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "owned acceptance HTTP evidence capture failed",
            )
                .into_response()
        }
    }
}

/// Match production startup in the owned temporary config before building the API.
pub(super) fn seed_acceptance_manifests(settings: &crate::Settings) -> Result<()> {
    sceneworks_core::builtin_manifests::seed_builtin_manifests(
        &settings.config_dir,
        sceneworks_core::builtin_manifests::SeedMode::Overwrite,
    )
    .map_err(|error| error.to_string())
}

async fn run(parent: &str, method: &str, source_backend: &str, cross: bool, preflight: bool) {
    let mut case = Case::admitted(parent, method, source_backend, cross, preflight)
        .unwrap_or_else(|e| panic!("MOSS manual admission refused before model construction: {e}"));
    case.persist().expect("initial owned failure record");
    // Match production startup before constructing isolated settings. Explicit
    // HF cache overrides remain authoritative; the env-less case uses the OS
    // shared cache rather than the temporary application's empty private cache.
    let _ = sceneworks_core::hf_home::ensure_default_huggingface_home();
    // App state is outside the uploadable evidence tree. Shared real HF snapshots
    // are read in place by normal source resolution; no model is copied/staged.
    let temp = tempfile::tempdir().expect("owned isolated app state");
    let mut settings = test_settings(&temp);
    settings.host_os = std::env::consts::OS.into();
    seed_acceptance_manifests(&settings).expect("owned shipped manifests");
    let app = create_app(settings.clone()).expect("isolated API");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("owned loopback API");
    let url = format!("http://{}", listener.local_addr().expect("local address"));
    // Production discovery resolves auto to its real registered index. The
    // measured index->UUID binding must match the allocator-selected context.
    let gpu_id = if cfg!(target_os = "macos") {
        "mlx".into()
    } else {
        "auto".into()
    };
    let worker = worker_settings(url, &settings, gpu_id);
    set_cap(&worker, json!({"fraction":case.fraction()})).expect("owned global GPU fraction");
    let http_directory = Arc::new(case.root.join(if case.preflight_only {
        "preflight/worker-http"
    } else {
        "worker-http"
    }));
    std::fs::create_dir_all(http_directory.as_path()).expect("owned HTTP observation directory");
    let http_sequence = Arc::new(AtomicU64::new(0));
    let served_app = app
        .clone()
        .layer(axum::middleware::from_fn(move |request, next| {
            observe_worker_http(
                Arc::clone(&http_directory),
                Arc::clone(&http_sequence),
                request,
                next,
            )
        }));
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn({
        let app = served_app;
        async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        }
    });
    let outcome = std::panic::AssertUnwindSafe(body(&mut case, &app, &settings, &worker))
        .catch_unwind()
        .await;
    // body never drops live native work on an error; every execute/utility call
    // has returned before this owned server is stopped. Never claim process exit
    // here: only the external supervisor can authenticate the final release.
    let _ = stop.send(());
    let server_result = server.await;
    let mut failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error),
        Err(panic) => Some(if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = panic.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "acceptance assertion panicked".into()
        }),
    };
    if let Err(first) = case.check_resources(true) {
        failure = Some(match failure {
            Some(later) if !later.contains(&first) => {
                format!("{first}; native completion: {later}")
            }
            Some(later) => later,
            None => first,
        });
    }
    if let Some(error) = failure.as_ref() {
        case.record["acceptanceDisposition"] = json!("failed");
        case.record["failureReason"] = json!(error);
    }
    case.record["raw"]["serverJoined"] = json!(matches!(server_result, Ok(Ok(()))));
    case.record["raw"]["nativeBodyReturnedAtUnixMs"] = json!(unix_ms().ok());
    case.persist()
        .expect("retain final observations before assertions");
    assert!(
        matches!(server_result, Ok(Ok(()))),
        "owned API did not shut down cleanly"
    );
    if let Some(error) = failure {
        panic!("MOSS {} failed: {error}", case.id);
    }
}

#[tokio::test]
#[ignore = "manual workflow_dispatch GPU admission required; header-only, no model tensor load"]
async fn moss_real_weights_device_preflight() {
    let id = env("MOSS_ACCEPTANCE_SOURCE_CASE_ID").expect("exact source case selector");
    let parts: Vec<_> = id.split('/').collect();
    assert_eq!(parts.len(), 3, "exact backend/parent/method selector");
    let actual = if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "cuda"
    };
    run(parts[1], parts[2], parts[0], parts[0] != actual, true).await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_mlx_base_lora_round_trip() {
    run("base", "lora", "mlx", false, false).await;
}
#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_mlx_base_lokr_round_trip() {
    run("base", "lokr", "mlx", false, false).await;
}
#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_mlx_sft_lora_round_trip() {
    run("sft", "lora", "mlx", false, false).await;
}
#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_mlx_sft_lokr_round_trip() {
    run("sft", "lokr", "mlx", false, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_cuda_base_lora_round_trip() {
    run("base", "lora", "cuda", false, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_cuda_base_lokr_round_trip() {
    run("base", "lokr", "cuda", false, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_cuda_sft_lora_round_trip() {
    run("sft", "lora", "cuda", false, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch real-weight GPU admission required"]
async fn moss_real_weights_cuda_sft_lokr_round_trip() {
    run("sft", "lokr", "cuda", false, false).await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_cuda_base_lora_cross_backend() {
    run("base", "lora", "cuda", true, false).await;
}
#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_cuda_base_lokr_cross_backend() {
    run("base", "lokr", "cuda", true, false).await;
}
#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_cuda_sft_lora_cross_backend() {
    run("sft", "lora", "cuda", true, false).await;
}
#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_cuda_sft_lokr_cross_backend() {
    run("sft", "lokr", "cuda", true, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_mlx_base_lora_cross_backend() {
    run("base", "lora", "mlx", true, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_mlx_base_lokr_cross_backend() {
    run("base", "lokr", "mlx", true, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_mlx_sft_lora_cross_backend() {
    run("sft", "lora", "mlx", true, false).await;
}
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[tokio::test]
#[ignore = "manual workflow_dispatch opposite-backend real-weight GPU admission required"]
async fn moss_real_weights_mlx_sft_lokr_cross_backend() {
    run("sft", "lokr", "mlx", true, false).await;
}
