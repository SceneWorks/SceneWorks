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
}
