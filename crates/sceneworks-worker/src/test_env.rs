//! The one process-global environment seam for this crate's tests (sc-12380).
//!
//! `cargo test -p sceneworks-worker --lib` builds ONE binary and runs every module's tests as
//! threads in ONE process, so `std::env::set_var` in any test is visible to all the others. Mutual
//! exclusion therefore has to be crate-wide: a per-module `static Mutex` is a DIFFERENT lock and
//! serializes nothing against the other modules. That is not hypothetical — `video_jobs` and
//! `training_jobs` each had their own `ENV_LOCK` while `image_jobs` took none, and all three wrote
//! `HF_HUB_CACHE`, so `ltx_eros_auto_injects_distill_lora_per_pass` still lost its cache dir to an
//! `image_jobs` writer (sc-12380 reproduced 6/6 on main AFTER the per-module lock was added).
//!
//! Hence: every test that reads OR writes an env var another test may write goes through here.
//!
//! Prefer pinning the value you need over branching on what the environment happens to hold. A
//! `if var_os(..).is_some() { return; }` guard is a silent PASS that asserts nothing on a box where
//! the var is set — it protects the suite from the environment by not testing at all.

// Not every target/feature combination exercises every helper here.
#![allow(dead_code)]

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
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
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
pub(crate) struct EnvVars {
    restore: Vec<(String, Option<String>)>,
    // Dropped after the `Drop` body below has restored the values, so the vars are never visible to
    // the next lock holder in their pinned state.
    _guard: MutexGuard<'static, ()>,
}

impl EnvVars {
    /// Pin `vars` (an empty value ⇒ the var is REMOVED for the duration) until the guard drops.
    ///
    /// NOT reentrant — see [`env_lock`].
    pub(crate) fn set(vars: &[(&str, &str)]) -> Self {
        let guard = env_lock();
        let restore = vars
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var(key).ok();
                if value.is_empty() {
                    std::env::remove_var(key);
                } else {
                    std::env::set_var(key, value);
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
pub(crate) fn temp_env_var<T>(key: &str, value: &str, body: impl FnOnce() -> T) -> T {
    temp_env_vars(&[(key, value)], body)
}

/// [`temp_env_var`] for several vars set together, under ONE acquisition of the shared lock.
pub(crate) fn temp_env_vars<T>(vars: &[(&str, &str)], body: impl FnOnce() -> T) -> T {
    let _vars = EnvVars::set(vars);
    body()
}
