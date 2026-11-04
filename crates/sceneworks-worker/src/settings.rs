//! Worker [`Settings`] and the environment-variable accessors that populate it.
use super::*;

#[derive(Clone)]
pub struct Settings {
    pub api_url: String,
    pub access_token: Option<String>,
    pub data_dir: PathBuf,
    /// App-owned resolved-model cache policy. Invalid environment is fail-closed to the finite,
    /// disabled shared default rather than enabling or unbounding the cache.
    pub resolved_cache: sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy,
    pub config_dir: PathBuf,
    pub worker_id: String,
    pub gpu_id: String,
    pub is_child_worker: bool,
    pub poll_seconds: u64,
    pub heartbeat_seconds: u64,
    pub shutdown_timeout_seconds: u64,
    pub huggingface_base_url: String,
    pub huggingface_token: Option<String>,
    /// Per-host download credentials from `SCENEWORKS_CREDENTIALS`, matched against
    /// LoRA/model `sourceUrl` hosts. HF auth still flows through `huggingface_token`.
    pub credentials: Vec<WorkerCredential>,
    pub max_lora_url_bytes: u64,
    pub max_model_url_bytes: u64,
    pub allow_private_lora_urls: bool,
    /// Number of CPU/utility worker processes to run when this worker is in
    /// dedicated `cpu` mode. Utility jobs (downloads, imports, frame extraction,
    /// timeline export, person detect/track) are I/O-bound and serialize per
    /// worker, so a small pool lets e.g. a quick upload run alongside a long
    /// download instead of queueing behind it.
    pub utility_workers: usize,
    /// Whether the MLX (Apple Silicon) tensor backend is enabled when deriving the worker's
    /// advertised capabilities from the linked engine registry (sc-3723). Default `true`.
    pub backend_mlx_enabled: bool,
    /// Whether the candle (Windows/CUDA + Linux/NVIDIA) tensor backend is enabled for capability
    /// derivation (sc-3723). **Defaults to on whenever the worker is built `--features
    /// backend-candle`** (sc-5502, epic 5483 Phase-7 cutover): the candle providers are at parity
    /// off-Mac, so a candle build advertises them by default — no `SCENEWORKS_BACKEND_CANDLE_ENABLED`
    /// needed. A non-candle build (Mac mlx, the desktop installer with no CUDA, any CPU/Linux build
    /// without the feature) defaults `false` and links no candle crate, so this is inert there. The
    /// env var still overrides either way (set `0` to disable native candle capabilities during a
    /// staged rollout; unsupported jobs then remain queued).
    pub backend_candle_enabled: bool,
    /// Operator-configured directories outside the app data dir and the Hugging Face cache
    /// that may be read for model weights — typically an existing ComfyUI `models/` tree
    /// (epic 10451, sc-10452). Widens the confinement allow-list in
    /// [`crate::paths::normalize_app_managed_lora_path`] **only**; it is never derived from a
    /// job payload, so the LAN-exposed jobs API (epic 4484) cannot introduce a root. Empty
    /// (feature off) unless `SCENEWORKS_EXTERNAL_MODEL_ROOTS` is set, and always empty on
    /// macOS — see `sceneworks_core::external_roots`.
    pub external_model_roots: Vec<PathBuf>,
    /// Configured GPU-memory ceiling in bytes, from `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES`.
    /// MOSS training transports a nonzero value through the native per-job config on both
    /// backends, intersecting the device budget rather than substituting a host-RAM budget.
    /// `0` means no additional per-job ceiling. Other job families retain their existing
    /// backend budget policy; the shared setting alone is not proof of native admission.
    pub gpu_memory_limit_bytes: u64,
}

impl fmt::Debug for Settings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("Settings");
        debug
            .field("api_url", &self.api_url)
            .field("access_token", &self.access_token.as_ref().map(|_| "***"))
            .field("data_dir", &self.data_dir);
        debug.field("resolved_cache", &self.resolved_cache);
        debug
            .field("config_dir", &self.config_dir)
            .field("worker_id", &self.worker_id)
            .field("gpu_id", &self.gpu_id)
            .field("is_child_worker", &self.is_child_worker)
            .field("poll_seconds", &self.poll_seconds)
            .field("heartbeat_seconds", &self.heartbeat_seconds)
            .field("shutdown_timeout_seconds", &self.shutdown_timeout_seconds)
            .field("huggingface_base_url", &self.huggingface_base_url)
            .field(
                "huggingface_token",
                &self.huggingface_token.as_ref().map(|_| "***"),
            )
            .field("credentials", &self.credentials)
            .field("max_lora_url_bytes", &self.max_lora_url_bytes)
            .field("max_model_url_bytes", &self.max_model_url_bytes)
            .field("allow_private_lora_urls", &self.allow_private_lora_urls)
            .field("utility_workers", &self.utility_workers)
            .field("backend_mlx_enabled", &self.backend_mlx_enabled)
            .field("backend_candle_enabled", &self.backend_candle_enabled)
            .field("external_model_roots", &self.external_model_roots)
            .field("gpu_memory_limit_bytes", &self.gpu_memory_limit_bytes)
            .finish()
    }
}

impl Settings {
    /// Latest explicit user cap, not MLX's derived default. A written zero clears
    /// a startup cap; an absent handoff preserves the launch-time environment.
    #[cfg(any(test, native_lane))]
    pub(crate) fn requested_gpu_memory_limit(
        &self,
    ) -> WorkerResult<sceneworks_core::app_paths::GpuMemoryLimitRequest> {
        use sceneworks_core::app_paths::{parse_gpu_memory_limit_request, GpuMemoryLimitRequest};
        if self.config_dir.as_os_str().is_empty() {
            return Ok(GpuMemoryLimitRequest::Bytes(self.gpu_memory_limit_bytes));
        }
        let path = sceneworks_core::app_paths::gpu_memory_limit_file(&self.config_dir);
        match std::fs::read_to_string(&path) {
            Ok(raw) => parse_gpu_memory_limit_request(&raw).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "GPU memory limit handoff {} is invalid: {error}",
                    path.display()
                ))
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(GpuMemoryLimitRequest::Bytes(self.gpu_memory_limit_bytes))
            }
            Err(error) => Err(error.into()),
        }
    }

    /// The launch-time policy is identical in production and tests.
    pub(crate) fn resolved_cache_policy(
        &self,
    ) -> sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy {
        self.resolved_cache.clone()
    }

    /// A minimal offline `Settings` for tests that only care about `data_dir`. Every field that
    /// could reach the network is pinned to the offline stub URL, so a test using this cannot
    /// accidentally acquire a live dependency.
    #[cfg(test)]
    pub(crate) fn for_test(data_dir: PathBuf) -> Self {
        Self {
            api_url: crate::test_env::OFFLINE_URL.to_owned(),
            access_token: None,
            data_dir,
            resolved_cache: Default::default(),
            config_dir: PathBuf::new(),
            worker_id: "worker".to_owned(),
            gpu_id: "cpu".to_owned(),
            is_child_worker: false,
            poll_seconds: 1,
            heartbeat_seconds: 1,
            shutdown_timeout_seconds: 1,
            huggingface_base_url: crate::test_env::OFFLINE_URL.to_owned(),
            huggingface_token: None,
            credentials: Vec::new(),
            max_lora_url_bytes: 1,
            max_model_url_bytes: 1,
            allow_private_lora_urls: false,
            utility_workers: 1,
            backend_mlx_enabled: false,
            backend_candle_enabled: false,
            external_model_roots: Vec::new(),
            gpu_memory_limit_bytes: 0,
        }
    }

    /// Compatibility accessor for existing fixture callers; invalid configuration never defaults.
    pub fn from_env() -> Self {
        Self::try_from_env().expect("invalid worker environment configuration")
    }

    /// Read worker configuration, refusing an invalid numeric override at startup.
    pub fn try_from_env() -> WorkerResult<Self> {
        // The download watchdog reads its value when creating a transfer; validate it
        // here as well so an invalid operator setting fails before work begins.
        env_u64_any(&["SCENEWORKS_DOWNLOAD_STALL_TIMEOUT_SECONDS"], 0)?;
        sceneworks_core::env_util::env_u64_any(&["SCENEWORKS_VIDEO_STALL_SECS"], 0, true)
            .map_err(WorkerError::InvalidPayload)?;
        let defaults = sceneworks_core::app_paths::AppPaths::platform_default();
        let config_dir = env_path_or("SCENEWORKS_CONFIG_DIR", &defaults.config_dir);
        let credentials_dir = sceneworks_core::credentials::credentials_dir(&config_dir);
        Ok(Self {
            api_url: env_string("SCENEWORKS_API_URL", DEFAULT_API_URL),
            access_token: std::env::var("SCENEWORKS_ACCESS_TOKEN")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            data_dir: env_path_or("SCENEWORKS_DATA_DIR", &defaults.data_dir),
            resolved_cache:
                sceneworks_core::model_artifacts::resolved_cache::ResolvedCachePolicy::from_env_or_safe_default(),
            config_dir: config_dir.clone(),
            worker_id: env_string("SCENEWORKS_WORKER_ID", "rust-utility-worker"),
            gpu_id: env_string("SCENEWORKS_GPU_ID", "cpu"),
            is_child_worker: sceneworks_core::env_util::env_flag("SCENEWORKS_WORKER_CHILD", false),
            poll_seconds: env_u64_any(
                &["SCENEWORKS_POLL_SECONDS", "SCENEWORKS_WORKER_POLL_SECONDS"],
                2,
            )?,
            heartbeat_seconds: env_u64_any(
                &[
                    "SCENEWORKS_HEARTBEAT_SECONDS",
                    "SCENEWORKS_WORKER_HEARTBEAT_SECONDS",
                ],
                10,
            )?,
            shutdown_timeout_seconds: env_u64_any(
                &["SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS"],
                10,
            )?,
            huggingface_base_url: env_string(
                "SCENEWORKS_HUGGINGFACE_BASE_URL",
                DEFAULT_HUGGINGFACE_BASE_URL,
            ),
            huggingface_token: std::env::var("HF_TOKEN")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            credentials: load_worker_credentials(&credentials_dir, &config_dir),
            max_lora_url_bytes: env_u64_any(
                &["SCENEWORKS_MAX_LORA_URL_BYTES"],
                DEFAULT_MAX_LORA_URL_BYTES,
            )?,
            max_model_url_bytes: env_u64_any(
                &["SCENEWORKS_MAX_MODEL_URL_BYTES"],
                DEFAULT_MAX_MODEL_URL_BYTES,
            )?,
            allow_private_lora_urls: sceneworks_core::env_util::env_flag("SCENEWORKS_ALLOW_PRIVATE_LORA_URLS", false),
            utility_workers: usize::try_from(env_u64_any(&["SCENEWORKS_UTILITY_WORKERS"], 4)?.max(1)).map_err(|_| WorkerError::InvalidPayload("SCENEWORKS_UTILITY_WORKERS exceeds this platform’s supported range".to_owned()))?,
            backend_mlx_enabled: env_bool("SCENEWORKS_BACKEND_MLX_ENABLED", true),
            // Phase-7 cutover (sc-5502): a candle build defaults the backend ON (the providers are at
            // off-Mac parity); a non-candle build defaults it off. The env var overrides either way.
            backend_candle_enabled: env_bool(
                "SCENEWORKS_BACKEND_CANDLE_ENABLED",
                cfg!(feature = "backend-candle"),
            ),
            gpu_memory_limit_bytes: env_u64_any(&["SCENEWORKS_GPU_MEMORY_LIMIT_BYTES"], 0)?,
            external_model_roots: sceneworks_core::external_roots::external_model_roots_from_env(),
        })
    }
}

pub(crate) use sceneworks_core::env_util::env_trimmed_string as env_string;

pub(crate) use sceneworks_core::env_util::env_path_or;

pub(crate) fn env_u64_any(keys: &[&str], default: u64) -> WorkerResult<u64> {
    sceneworks_core::env_util::env_u64_any(keys, default, false)
        .map_err(WorkerError::InvalidPayload)
}

pub(crate) use sceneworks_core::env_util::env_flag as env_bool;

#[cfg(test)]
mod numeric_env_tests {
    use super::*;

    #[test]
    fn numeric_env_rejects_invalid_primary_without_falling_through_to_alias() {
        let parse = |primary: Option<&str>, alias: Option<&str>| {
            sceneworks_core::env_util::env_u64_any_with(&["PRIMARY", "ALIAS"], 42, false, |key| {
                match if key == "PRIMARY" { primary } else { alias } {
                    Some(value) => Ok(value.to_owned()),
                    None => Err(std::env::VarError::NotPresent),
                }
            })
        };
        assert_eq!(parse(None, None).unwrap(), 42);
        assert_eq!(parse(None, Some("12")).unwrap(), 12);
        assert_eq!(parse(Some("0"), Some("12")).unwrap(), 0);
        for bad in ["abc", "256G", "", "-1", "18446744073709551616"] {
            let error = parse(Some(bad), Some("12")).unwrap_err().to_string();
            assert!(error.contains("PRIMARY"), "{bad:?}: {error}");
        }
    }

    #[test]
    fn settings_reject_invalid_limits_instead_of_using_defaults() {
        use sceneworks_core::test_env::EnvVars;
        let keys = [
            "SCENEWORKS_POLL_SECONDS",
            "SCENEWORKS_WORKER_POLL_SECONDS",
            "SCENEWORKS_HEARTBEAT_SECONDS",
            "SCENEWORKS_WORKER_HEARTBEAT_SECONDS",
            "SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
            "SCENEWORKS_MAX_LORA_URL_BYTES",
            "SCENEWORKS_MAX_MODEL_URL_BYTES",
            "SCENEWORKS_UTILITY_WORKERS",
            "SCENEWORKS_GPU_MEMORY_LIMIT_BYTES",
            "SCENEWORKS_DOWNLOAD_STALL_TIMEOUT_SECONDS",
            "SCENEWORKS_VIDEO_STALL_SECS",
        ];
        // `set` removes empty values; use one lock for the complete environment fixture.
        let scratch = tempfile::tempdir().unwrap();
        let config = scratch.path().to_str().unwrap();
        for (setting, value, refuses) in keys.into_iter().map(|key| (key, "256G", true)).chain([
            ("SCENEWORKS_VIDEO_STALL_SECS", "0", false),
            ("SCENEWORKS_VIDEO_STALL_SECS", " 12 ", false),
        ]) {
            let mut entries: Vec<_> = keys
                .iter()
                .map(|key| (*key, if *key == setting { value } else { "" }))
                .collect();
            entries.push(("SCENEWORKS_CONFIG_DIR", config));
            entries.push(("SCENEWORKS_CREDENTIALS", ""));
            let _env = EnvVars::set(&entries);
            match (refuses, Settings::try_from_env()) {
                (true, Err(error)) => assert!(error.to_string().contains(setting), "{error}"),
                (false, Ok(_)) => {}
                (true, Ok(_)) => panic!("invalid setting {setting} accepted"),
                (false, Err(error)) => panic!("valid setting {setting} refused: {error}"),
            }
        }
    }
}

#[cfg(test)]
mod memory_limit_tests {
    use super::*;
    use sceneworks_core::app_paths::GpuMemoryLimitRequest::{Bytes, Fraction};

    #[test]
    fn one_immutable_settings_value_observes_user_on_change_off_and_corruption() {
        let root = tempfile::tempdir().unwrap();
        let mut settings = Settings::for_test(root.path().join("data"));
        settings.config_dir = root.path().join("config");
        settings.gpu_memory_limit_bytes = 999;
        std::fs::create_dir_all(&settings.config_dir).unwrap();
        let settings = settings;
        let path = sceneworks_core::app_paths::gpu_memory_limit_file(&settings.config_dir);
        assert_eq!(settings.requested_gpu_memory_limit().unwrap(), Bytes(999));
        for (raw, expected) in [
            ("100", Bytes(100)),
            ("200", Bytes(200)),
            ("0", Bytes(0)),
            (r#"{"fraction":0.5}"#, Fraction(0.5)),
            (r#"{"fraction":0.75}"#, Fraction(0.75)),
            ("0", Bytes(0)),
        ] {
            std::fs::write(&path, raw).unwrap();
            assert_eq!(settings.requested_gpu_memory_limit().unwrap(), expected);
        }
        for invalid in ["", "-1", "1.5", "18446744073709551616", "unknown"] {
            std::fs::write(&path, invalid).unwrap();
            assert!(settings.requested_gpu_memory_limit().is_err(), "{invalid}");
        }
        std::fs::remove_file(&path).unwrap();
        assert_eq!(settings.requested_gpu_memory_limit().unwrap(), Bytes(999));
    }
}

#[cfg(test)]
mod test_builder_contracts {
    use super::*;
    #[test]
    fn fixture_settings_ignore_ambient_secrets_network_and_cache_policy() {
        let _env = crate::test_env::EnvVars::set(&[
            ("SCENEWORKS_API_URL", "https://ambient.invalid"),
            ("SCENEWORKS_ACCESS_TOKEN", "ambient-token"),
            ("HF_TOKEN", "ambient-hf-token"),
            ("SCENEWORKS_RESOLVED_CACHE_ENABLED", "1"),
        ]);
        let mut settings = Settings::for_test(PathBuf::from("fixture"));
        assert_eq!(settings.data_dir, PathBuf::from("fixture"));
        assert_eq!(settings.api_url, crate::test_env::OFFLINE_URL);
        assert_eq!(settings.huggingface_base_url, crate::test_env::OFFLINE_URL);
        assert!(settings.access_token.is_none() && settings.huggingface_token.is_none());
        assert!(settings.credentials.is_empty());
        assert_eq!(settings.resolved_cache_policy(), Default::default());
        settings.resolved_cache.enabled = true;
        assert!(
            settings.resolved_cache_policy().enabled,
            "the same field backs both build modes"
        );
        settings.resolved_cache.enabled = false;
        assert!(
            !settings.resolved_cache_policy().enabled,
            "ambient cache enablement cannot override the field"
        );
    }
}
