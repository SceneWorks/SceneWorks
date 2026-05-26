//! Default Hugging Face cache home for the server/host-mode binaries (sc-1904
//! follow-up).
//!
//! Hugging Face tooling caches under `~/.cache/huggingface` on every platform
//! (huggingface_hub's `HF_HOME` default; mirrored by the Python worker's
//! `hf_cache.huggingface_cache_root` and the desktop's `shared_huggingface_home`).
//! When the rust-api / rust-worker binaries are started with none of the HF cache
//! env vars set (host mode), they otherwise fall back to `<data_dir>/cache/...`,
//! dumping downloads into the app's private data folder. Defaulting `HF_HOME` to
//! the OS Hugging Face home at startup keeps host-mode downloads in the shared
//! per-user cache, deduplicated with every other HF tool. The desktop and Docker
//! Compose already inject `HF_HOME`, so this only changes the env-less case.

use std::path::PathBuf;

/// The OS Hugging Face home, `~/.cache/huggingface` (the literal `~/.cache`, not
/// the platform cache dir — matching huggingface_hub and `Path.home()/.cache/
/// huggingface` in the Python worker). `None` when no home dir can be resolved.
pub fn os_huggingface_home() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|base| base.home_dir().join(".cache").join("huggingface"))
}

/// Pure decision: the value to default `HF_HOME` to when the process set none of
/// the HF cache env vars. Returns `None` (leave the environment untouched) when
/// any HF cache var is already set, or when no home dir is available. Taking the
/// env values + home as arguments keeps it deterministically testable.
pub fn default_huggingface_home(
    hf_hub_cache: Option<&str>,
    huggingface_hub_cache: Option<&str>,
    hf_home: Option<&str>,
    os_home: Option<PathBuf>,
) -> Option<PathBuf> {
    let is_set = |value: Option<&str>| value.map(str::trim).is_some_and(|value| !value.is_empty());
    if is_set(hf_hub_cache) || is_set(huggingface_hub_cache) || is_set(hf_home) {
        return None;
    }
    os_home
}

/// If no HF cache env var (`HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME`)
/// is set, point `HF_HOME` at the OS Hugging Face home so downloads land in the
/// shared `~/.cache/huggingface` cache rather than the app's data dir. Returns the
/// path it set, or `None` when it left the environment unchanged. Call once at
/// binary startup, before any cache resolution or worker spawn.
pub fn ensure_default_huggingface_home() -> Option<PathBuf> {
    let read = |key: &str| std::env::var(key).ok();
    let chosen = default_huggingface_home(
        read("HF_HUB_CACHE").as_deref(),
        read("HUGGINGFACE_HUB_CACHE").as_deref(),
        read("HF_HOME").as_deref(),
        os_huggingface_home(),
    )?;
    std::env::set_var("HF_HOME", &chosen);
    Some(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn defaults_to_os_home_when_no_hf_env_is_set() {
        let home = Some(PathBuf::from("/home/alice/.cache/huggingface"));
        assert_eq!(
            default_huggingface_home(None, None, None, home.clone()),
            home
        );
        // Blank/whitespace env values count as unset.
        assert_eq!(
            default_huggingface_home(Some(""), Some("   "), None, home.clone()),
            home
        );
    }

    #[test]
    fn leaves_env_untouched_when_any_hf_var_is_set() {
        let home = Some(PathBuf::from("/home/alice/.cache/huggingface"));
        assert_eq!(
            default_huggingface_home(Some("/mnt/hub"), None, None, home.clone()),
            None
        );
        assert_eq!(
            default_huggingface_home(None, Some("/mnt/hub"), None, home.clone()),
            None
        );
        assert_eq!(
            default_huggingface_home(None, None, Some("/srv/hf"), home),
            None
        );
    }

    #[test]
    fn yields_none_without_a_home_dir() {
        assert_eq!(default_huggingface_home(None, None, None, None), None);
    }

    #[test]
    fn os_home_ends_in_cache_huggingface() {
        // Environment-dependent, but every CI/dev runner has a home dir.
        if let Some(home) = os_huggingface_home() {
            assert!(home.ends_with(Path::new(".cache").join("huggingface")));
        }
    }
}
