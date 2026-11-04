//! Process-wide environment isolation for opt-in test support.
use std::sync::{Mutex, MutexGuard};

/// The single lock every env-touching test in this crate serializes on. Crate-level ON PURPOSE —
/// see the module docs. Do not add a second one.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Take the shared env lock, holding it until the returned guard drops.
///
/// Use this directly when a test needs to *read* an env var (or call code that does) and must not
/// have it changed underneath — the environment is process-global, so only holding the lock across
/// the whole read-then-use makes that pair atomic. When the test also sets a var, prefer
/// [`EnvVars::set`], which takes this lock for you.
///
/// Poisoning is recovered from: a panic in another env test must not cascade into a spurious
/// failure here, and there is no guarded data to be left inconsistent — we want the exclusion only.
///
/// NOT reentrant. A test holding this must not also call [`temp_env_var`] / [`temp_env_vars`] /
/// [`EnvVars::set`], which would self-deadlock.
pub fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

/// Env vars pinned for as long as this guard lives, restored on drop. Holds [`env_lock`], so no
/// other env test can observe or clobber the pinned values.
///
/// Restoring on `Drop` (rather than around a closure) means a panicking assertion still puts the
/// environment back: a leaked `HF_HUB_CACHE` would otherwise silently re-point every later test in
/// the process at the wrong cache.
#[must_use = "the vars are restored when this guard drops; `let _ = EnvVars::set(..)` drops it \
              immediately and pins nothing — bind it to a named `_env`-style local"]
pub struct EnvVars {
    restore: Vec<(String, Option<std::ffi::OsString>)>,
    // Dropped after the `Drop` body below has restored the values, so the vars are never visible to
    // the next lock holder in their pinned state.
    _guard: MutexGuard<'static, ()>,
}

impl EnvVars {
    /// Pin `vars` (an empty value ⇒ the var is REMOVED for the duration) until the guard drops.
    ///
    /// NOT reentrant — see [`env_lock`].
    pub fn set(vars: &[(&str, &str)]) -> Self {
        let vars: Vec<_> = vars
            .iter()
            .map(|(key, value)| {
                (
                    *key,
                    (!value.is_empty()).then(|| std::ffi::OsStr::new(value)),
                )
            })
            .collect();
        Self::set_os(&vars)
    }

    /// Pin native paths without a lossy UTF-8 conversion; `None` removes the variable.
    pub fn set_os(vars: &[(&str, Option<&std::ffi::OsStr>)]) -> Self {
        let guard = env_lock();
        let restore = vars
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var_os(key);
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
                ((*key).to_owned(), previous)
            })
            .collect();
        Self {
            restore,
            _guard: guard,
        }
    }
}

impl Drop for EnvVars {
    fn drop(&mut self) {
        for (key, previous) in &self.restore {
            match previous {
                Some(prior) => std::env::set_var(key, prior),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Set `key` to `value` (empty ⇒ removed) for the duration of `body`, then restore.
///
/// NOT reentrant — nesting these self-deadlocks. A test needing two vars must use
/// [`temp_env_vars`], which sets them under ONE acquisition.
pub fn temp_env_var<T>(key: &str, value: &str, body: impl FnOnce() -> T) -> T {
    temp_env_vars(&[(key, value)], body)
}

/// [`temp_env_var`] for several vars set together, under ONE acquisition of the shared lock.
pub fn temp_env_vars<T>(vars: &[(&str, &str)], body: impl FnOnce() -> T) -> T {
    let _vars = EnvVars::set(vars);
    body()
}

/// Clear all HF-cache overrides for the scope under the same process-global lock.
pub fn isolate_hf_cache() -> EnvVars {
    EnvVars::set(&[
        ("HF_HUB_CACHE", ""),
        ("HUGGINGFACE_HUB_CACHE", ""),
        ("HF_HOME", ""),
    ])
}
