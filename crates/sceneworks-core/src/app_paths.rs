use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// OS-appropriate default locations for SceneWorks user data, configuration, and
/// cache. Environment overrides (`SCENEWORKS_DATA_DIR`, `SCENEWORKS_CONFIG_DIR`,
/// etc.) take precedence — this resolver only supplies defaults when those are
/// unset, so the Docker deployment and `SCENEWORKS_DATA_DIR=./data` dev workflow
/// are unaffected.
///
/// Layout per platform:
/// - **Windows:** `%APPDATA%\SceneWorks\data`, `%APPDATA%\SceneWorks\config`,
///   `%LOCALAPPDATA%\SceneWorks\cache`
/// - **macOS:** `~/Library/Application Support/SceneWorks/data` and `/config`,
///   `~/Library/Caches/SceneWorks`
/// - **Linux:** `$XDG_DATA_HOME/sceneworks`, `$XDG_CONFIG_HOME/sceneworks`,
///   `$XDG_CACHE_HOME/sceneworks`
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    /// Resolve platform default paths, falling back to repo-relative
    /// `data`/`config`/`data/cache` when no home directory is available (e.g.
    /// minimal container environments) so behavior degrades to the historical
    /// default rather than panicking.
    pub fn platform_default() -> Self {
        platform_default_paths().unwrap_or_else(repo_relative_paths)
    }

    /// Platform defaults **only** when the OS actually gave us a home directory —
    /// i.e. without the repo-relative fallback [`AppPaths::platform_default`] applies.
    ///
    /// That fallback is right for the data/config roots (it reproduces the historical
    /// default rather than panicking), but it resolves to a *relative* `config`, which
    /// means "whatever directory this process was started in". For anything that must
    /// never land in the current working directory — the credential store, which would
    /// otherwise put a plaintext token inside a git checkout — a relative answer is
    /// worse than no answer. Callers that care take the `None` and decide themselves.
    pub fn platform_default_checked() -> Option<Self> {
        platform_default_paths()
    }

    /// Create the data, config, and cache directories if they do not yet exist.
    pub fn ensure_exists(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(&self.cache_dir)?;
        Ok(())
    }
}

/// Filename (under [`AppPaths::config_dir`]) of the live GPU-memory-limit handoff the desktop
/// shell writes whenever the Settings slider changes, and the running MLX worker re-reads between
/// jobs (epic 7819, sc-7824). A bare decimal byte count; `0` means "no limit". Keeping it in the
/// shared config dir — which the desktop injects into the worker as `SCENEWORKS_CONFIG_DIR` — lets
/// the cap change live without a worker restart. An absent file leaves the worker on whatever it
/// applied at spawn from `SCENEWORKS_GPU_MEMORY_LIMIT_BYTES`.
pub fn gpu_memory_limit_file(config_dir: &Path) -> PathBuf {
    config_dir.join("gpu_memory_limit")
}

/// Filename (under [`AppPaths::config_dir`]) where the MLX worker publishes live GPU-memory
/// telemetry for the Settings readout (epic 7819, sc-7825). JSON-encoded [`GpuMemoryTelemetry`],
/// rewritten on a short interval. macOS/MLX only — candle/CPU workers never write it, so the
/// desktop telemetry command returns `None` there.
pub fn gpu_telemetry_file(config_dir: &Path) -> PathBuf {
    config_dir.join("gpu_telemetry.json")
}

/// Filename (under [`AppPaths::config_dir`]) of the durable UI preference store. Written by the
/// API's `PUT /api/v1/ui-preferences` (`apps/rust-api/src/preferences.rs`), which owns its shape;
/// this only names the file so a *reader* outside that module can find it.
///
/// Named here for the same reason [`gpu_memory_limit_file`] is: the desktop injects one
/// `SCENEWORKS_CONFIG_DIR` into every process it spawns, so a preference the API persists is
/// already sitting in a directory the worker holds. See [`embed_workflow_in_images`].
pub fn ui_preferences_file(config_dir: &Path) -> PathBuf {
    config_dir.join("ui-preferences.json")
}

/// The slice of [`ui_preferences_file`] the WORKER reads.
///
/// Deliberately NOT the API's whole `UiPreferences` type: that struct is the route's own contract
/// and grows with the UI, and a worker that deserialized all of it would fail to read a preference
/// file written by a newer build. Every field is an `Option` and unknown keys are ignored, so this
/// stays readable across versions in both directions.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkerVisiblePreferences {
    embed_workflow_in_images: Option<bool>,
}

/// Whether a generated PNG should carry the sanitized workflow envelope (epic 15945, sc-15948).
///
/// **Defaults to `true` when the preference is ABSENT** — no config dir, no file, valid JSON with
/// the key missing, or a body that is not JSON at all. A feature whose whole point is that a shared
/// image reloads its recipe is worthless off by default, and a user who does not want it should have
/// to find the switch once rather than opt in forever (sc-15953 owns that switch and the first-run
/// disclosure).
///
/// **Fails CLOSED on any other I/O error.** "Absent" and "unreadable" are not the same state, and
/// collapsing them is how an explicit opt-out silently inverts itself: a user who deliberately
/// turned embedding off would start embedding again the first time an antivirus scanner held the
/// file open or an ACL glitched. The envelope is sanitized, but it still carries their prompt, their
/// model and their LoRA repos — exactly what someone who turned it off was protecting. A transient
/// read failure costs one image with no chunk; the other direction costs a disclosure they refused.
///
/// Read at the WRITE SEAM rather than cached at worker startup, so flipping the toggle takes effect
/// on the next image instead of the next launch — the same live-handoff shape as
/// [`gpu_memory_limit_file`]. The cost is one small file read per JOB: the worker resolves this once
/// and the answer rides on the image plan (`ImagePlan::workflow_source`), so a 12-image batch reads
/// it once, against tens of milliseconds of PNG encode per image.
#[must_use]
pub fn embed_workflow_in_images(config_dir: &Path) -> bool {
    match std::fs::read_to_string(ui_preferences_file(config_dir)) {
        Ok(body) => embed_workflow_in_images_from_json(&body),
        // No preference file (and no config dir — Windows maps a missing parent to `NotFound`
        // too) is the fresh-install shape: default ON.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(_) => false,
    }
}

/// [`embed_workflow_in_images`] against an already-read preference file body.
///
/// Public so the WRITER of that file can assert against the reader directly — the API owns the
/// route and the field name, this crate owns the parse, and the two are only correct together
/// (`embed_workflow_in_images_round_trips_under_the_name_the_worker_reads` in
/// `apps/rust-api/src/preferences.rs`).
#[must_use]
pub fn embed_workflow_in_images_from_json(body: &str) -> bool {
    serde_json::from_str::<WorkerVisiblePreferences>(body)
        .ok()
        .and_then(|preferences| preferences.embed_workflow_in_images)
        .unwrap_or(true)
}

/// A snapshot of the MLX runtime's process-global memory counters (epic 7819, sc-7825), written by
/// the worker to [`gpu_telemetry_file`] and read back by the desktop shell for the Settings
/// display. All values are bytes. `limit_bytes` is the currently-applied soft ceiling (`0` = no
/// limit), tracked from what the worker actually applied rather than MLX's internal default budget.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuMemoryTelemetry {
    pub active_bytes: u64,
    pub peak_bytes: u64,
    pub cache_bytes: u64,
    pub limit_bytes: u64,
}

fn repo_relative_paths() -> AppPaths {
    AppPaths {
        data_dir: PathBuf::from("data"),
        config_dir: PathBuf::from("config"),
        cache_dir: PathBuf::from("data").join("cache"),
    }
}

#[cfg(target_os = "windows")]
fn platform_default_paths() -> Option<AppPaths> {
    let base = directories::BaseDirs::new()?;
    let roaming = base.data_dir(); // %APPDATA%
    let local = base.cache_dir(); // %LOCALAPPDATA%
    Some(AppPaths {
        data_dir: roaming.join("SceneWorks").join("data"),
        config_dir: roaming.join("SceneWorks").join("config"),
        cache_dir: local.join("SceneWorks").join("cache"),
    })
}

#[cfg(target_os = "macos")]
fn platform_default_paths() -> Option<AppPaths> {
    let base = directories::BaseDirs::new()?;
    let support = base.data_dir(); // ~/Library/Application Support
    Some(AppPaths {
        data_dir: support.join("SceneWorks").join("data"),
        config_dir: support.join("SceneWorks").join("config"),
        cache_dir: base.cache_dir().join("SceneWorks"), // ~/Library/Caches/SceneWorks
    })
}

#[cfg(all(unix, not(target_os = "macos")))]
fn platform_default_paths() -> Option<AppPaths> {
    let base = directories::BaseDirs::new()?;
    Some(AppPaths {
        data_dir: base.data_dir().join("sceneworks"), // $XDG_DATA_HOME/sceneworks
        config_dir: base.config_dir().join("sceneworks"), // $XDG_CONFIG_HOME/sceneworks
        cache_dir: base.cache_dir().join("sceneworks"), // $XDG_CACHE_HOME/sceneworks
    })
}

#[cfg(not(any(unix, target_os = "windows")))]
fn platform_default_paths() -> Option<AppPaths> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_and_config_dirs_are_distinct() {
        let paths = AppPaths::platform_default();
        assert_ne!(
            paths.data_dir, paths.config_dir,
            "data and config must resolve to separate directories"
        );
    }

    #[test]
    fn ensure_exists_creates_all_directories() {
        let temp = tempfile::tempdir().expect("temp dir");
        let paths = AppPaths {
            data_dir: temp.path().join("data"),
            config_dir: temp.path().join("config"),
            cache_dir: temp.path().join("cache"),
        };
        paths.ensure_exists().expect("directories are created");
        assert!(paths.data_dir.is_dir());
        assert!(paths.config_dir.is_dir());
        assert!(paths.cache_dir.is_dir());
    }

    /// An ABSENT preference defaults ON (sc-15948): a user who has never opened Settings and an
    /// install whose config dir does not exist yet both have to embed, or the feature silently
    /// does nothing on a fresh install.
    ///
    /// Deliberately separate from [`embedding_fails_closed_when_the_file_cannot_be_read`]. These
    /// were one test bundling "missing" with "unreadable", and that bundling is precisely what hid
    /// the distinction that matters: absence is a default, an I/O error is a failure, and they must
    /// not resolve the same way.
    #[test]
    fn embedding_defaults_on_when_the_preference_is_absent() {
        let temp = tempfile::tempdir().expect("temp dir");
        let config = temp.path();

        assert!(
            embed_workflow_in_images(&config.join("does-not-exist")),
            "a missing config dir must default ON"
        );
        assert!(
            embed_workflow_in_images(config),
            "a missing preference file must default ON"
        );

        // A file that answers nothing — unparseable, or parseable with the key absent or the wrong
        // type — is also an absent PREFERENCE, not a failure to read. It defaults ON.
        for body in [
            "",
            "{",
            "not json at all",
            "[]",
            "{}",
            r#"{"theme":"dark"}"#,
            r#"{"embedWorkflowInImages":null}"#,
            r#"{"embedWorkflowInImages":"yes"}"#,
        ] {
            std::fs::write(ui_preferences_file(config), body).expect("writes");
            assert!(embed_workflow_in_images(config), "{body:?} must default ON");
        }
    }

    /// An I/O error that is NOT "absent" fails CLOSED (sc-15948).
    ///
    /// The real-world causes are an antivirus scanner holding the file open, an ACL change, or a
    /// half-migrated profile. Whatever the cause, the last thing the user SAID may have been "do not
    /// embed", and a transient read error must not overturn it — the envelope carries their prompt,
    /// model and LoRA repos.
    ///
    /// A directory where the file belongs is the portable way to force a non-`NotFound` error:
    /// Windows refuses to open a directory for reading (`PermissionDenied`), and Unix opens it and
    /// then fails the read (`EISDIR`). Either way it is an `Err` that is not `NotFound`.
    #[test]
    fn embedding_fails_closed_when_the_file_cannot_be_read() {
        let temp = tempfile::tempdir().expect("temp dir");
        let config = temp.path();
        let preference_path = ui_preferences_file(config);
        std::fs::create_dir_all(&preference_path).expect("creates a directory in the file's place");

        let error = std::fs::read_to_string(&preference_path)
            .expect_err("reading a directory as a file must fail");
        assert_ne!(
            error.kind(),
            std::io::ErrorKind::NotFound,
            "this test only means something if the forced error is NOT the absent case: {error}"
        );
        assert!(
            !embed_workflow_in_images(config),
            "an unreadable preference file must fail CLOSED — a user who turned embedding off must \
             not start embedding again because of a transient read error"
        );
    }

    #[test]
    fn embedding_is_off_only_when_the_preference_says_so() {
        let temp = tempfile::tempdir().expect("temp dir");
        let config = temp.path();

        std::fs::write(
            ui_preferences_file(config),
            r#"{"theme":"dark","embedWorkflowInImages":false}"#,
        )
        .expect("writes");
        assert!(!embed_workflow_in_images(config));

        std::fs::write(
            ui_preferences_file(config),
            r#"{"embedWorkflowInImages":true}"#,
        )
        .expect("writes");
        assert!(embed_workflow_in_images(config));
    }
}
