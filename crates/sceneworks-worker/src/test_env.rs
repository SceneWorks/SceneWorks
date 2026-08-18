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

/// An unroutable URL. Port 0 is never listened on, so a request fails immediately and locally
/// instead of depending on a stub's fidelity — a test that reaches the network fails loudly.
pub(crate) const OFFLINE_URL: &str = "http://127.0.0.1:0";

/// [`Settings`] with EVERY network field pinned unroutable. **Use this instead of
/// `Settings::from_env()` in any test whose path could reach a download**, keeping the
/// `..offline_settings()` spread last so the fields you do pin still win:
///
/// ```ignore
/// let settings = Settings { data_dir: fixture, ..offline_settings() };
/// ```
///
/// `Settings::from_env()` defaults `api_url` to the real local API and `huggingface_base_url` to the
/// real `https://huggingface.co`, and a `..Settings::from_env()` spread silently inherits BOTH for
/// every field the test did not think to name. The two are NOT interchangeable, which is the trap:
/// a tier fetch dials `huggingface_base_url` (`HuggingFaceSnapshot::resolve` →
/// `{base_url}/api/models/…/tree/…`) while `api_url` only carries progress/heartbeat. Pin only
/// `api_url` and the failure path still goes red — but only after really resolving the tree against
/// the live hub, and for a public repo that resolve SUCCEEDS, so bytes can start landing first.
///
/// That exact bug shipped twice (#1577, then sc-12380 one field over), and a third site was found
/// mid-fix pinning `api_url` alone and saved only by an early-out upstream. Three of four call sites
/// remembered both; one did not. Hence a helper that cannot forget, rather than four comments asking
/// people to remember.
///
/// This pins struct fields only. The HF **cache dir** is a separate axis that a pinned `data_dir`
/// does NOT make hermetic — `huggingface_hub_cache_dir` reads `HF_HUB_CACHE` /
/// `HUGGINGFACE_HUB_CACHE` / `HF_HOME` BEFORE `data_dir` — so a test that resolves a cache must also
/// pin those via [`EnvVars::set`].
pub(crate) fn offline_settings() -> crate::Settings {
    crate::Settings {
        api_url: OFFLINE_URL.to_owned(),
        huggingface_base_url: OFFLINE_URL.to_owned(),
        ..crate::Settings::from_env()
    }
}

/// Install a process-global `tracing` subscriber whose only job is to keep callsite `Interest`
/// honest while a test captures events through a scoped (`with_default`) subscriber. Call it at the
/// top of every test that captures `tracing` output with `tracing::subscriber::with_default`.
///
/// Why (2026-08-17, `tracing_records_*` full-suite flake): `tracing-core` (0.1.36) caches
/// per-callsite `Interest` on the FIRST hit of each callsite anywhere in the process. While the
/// dispatcher registry has only ever held one dispatcher, that first-hit interest is computed from
/// the *registering thread's* current default dispatcher — not from the registry
/// (`Rebuilder::JustOne` in `tracing-core`'s `callsite.rs`). This test binary runs its tests on
/// many threads, almost all subscriber-less, so while one test holds the process' only scoped
/// subscriber, a concurrent test that first-hits a shared callsite (e.g. `select_strategy`'s
/// admission events, which dozens of subscriber-less selector tests also drive) caches
/// `Interest::never` from its NONE dispatcher — and every later hit of that callsite, including
/// the capturing test's own, is dropped before any subscriber is consulted. Observed as a capture
/// test failing under full-suite load with an INFO event missing while a later WARN event from the
/// same body was captured; which capture test fails moves run to run with the first-hit race.
///
/// A registered global default closes both holes at once: `dispatcher::get_default` on a
/// subscriber-less thread now resolves to this floor — whose `register_callsite` answers
/// `Interest::sometimes()`, deferring every event to its thread's own dispatcher — and the floor's
/// presence in the registry means the single-dispatcher fast path no longer applies while one
/// scoped capture is live. `enabled()` is `false`, so the floor itself never records anything and
/// subscriber-less threads stay silent exactly as before.
///
/// Idempotent. If a real global default won the slot first (`fmt().try_init()` in the ignored
/// smokes), the install fails harmlessly — any registered global whose interest is not `never`
/// for the captured level provides the same guarantee.
pub(crate) fn install_tracing_interest_floor() {
    struct InterestFloor;
    impl tracing::Subscriber for InterestFloor {
        fn register_callsite(
            &self,
            _: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::sometimes()
        }
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            false
        }
        // Unreachable while `enabled` is `false`; a well-formed Id is still required.
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, _: &tracing::Event<'_>) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }
    // `set_global_default` registers a dispatcher (and rebuilds the interest cache) even when it
    // loses the install race, so gate on `Once` rather than calling it unconditionally.
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let _ = tracing::subscriber::set_global_default(InterestFloor);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The floor's load-bearing property, end to end: a subscriber-less thread that first-hits a
    /// callsite while a scoped capture subscriber is live must not cache `Interest::never` for it.
    /// Without the floor the loss is deterministic here — the bare thread's registration evaluates
    /// its NONE dispatcher via the single-dispatcher fast path, the shared callsite caches
    /// `never`, and the capture below misses its own event (the exact shape of the 2026-08-17
    /// `tracing_records_*` full-suite failures). Gut [`install_tracing_interest_floor`]'s
    /// `register_callsite` — or drop the install — and this goes red.
    #[test]
    fn a_scoped_capture_survives_a_subscriberless_first_hit_of_its_callsite() {
        // ONE function so both threads hit the SAME macro callsite — two `info!` literals would
        // be two independent callsites and prove nothing.
        fn sentinel() {
            tracing::info!("interest-floor sentinel");
        }

        install_tracing_interest_floor();

        #[derive(Clone, Default)]
        struct Capture(std::sync::Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .without_time()
            .finish();
        // The scoped subscriber must be live ACROSS the bare-thread hit, matching the victim
        // shape: its registration is also what raises the process max level so the bare thread's
        // event reaches callsite registration at all.
        tracing::subscriber::with_default(subscriber, || {
            std::thread::spawn(sentinel)
                .join()
                .expect("bare emitter thread");
            sentinel();
        });
        let output = String::from_utf8(
            capture
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        )
        .expect("fmt output is utf-8");
        assert!(
            output.contains("interest-floor sentinel"),
            "a subscriber-less thread's first hit poisoned the callsite: {output:?}"
        );
    }

    /// [`offline_settings`] must move every network field OFF its real default. The `assert_ne`s are
    /// the load-bearing half: asserting only `== OFFLINE_URL` would still pass if someone
    /// "simplified" the helper to `Settings::from_env()` on a box where the env happens to hold that
    /// value, and it is dropping a pin — not the constant — that this exists to catch.
    #[test]
    fn offline_settings_pins_every_network_field_off_its_real_default() {
        let offline = offline_settings();

        assert_eq!(offline.api_url, OFFLINE_URL, "api_url must be unroutable");
        assert_eq!(
            offline.huggingface_base_url, OFFLINE_URL,
            "huggingface_base_url is what a tier fetch dials — it must be unroutable"
        );
        assert_ne!(
            offline.huggingface_base_url,
            crate::DEFAULT_HUGGINGFACE_BASE_URL,
            "the real hub must never survive into a test's Settings"
        );
        assert_ne!(
            offline.api_url,
            crate::DEFAULT_API_URL,
            "the real API must never survive into a test's Settings"
        );
    }
}
