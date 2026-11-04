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

pub(crate) use sceneworks_core::test_env::{
    env_lock, isolate_hf_cache, temp_env_var, temp_env_vars, EnvVars,
};

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
        max_lora_url_bytes: crate::DEFAULT_MAX_LORA_URL_BYTES,
        max_model_url_bytes: crate::DEFAULT_MAX_MODEL_URL_BYTES,
        backend_mlx_enabled: true,
        backend_candle_enabled: cfg!(feature = "backend-candle"),
        utility_workers: 4,
        poll_seconds: 2,
        heartbeat_seconds: 10,
        shutdown_timeout_seconds: 10,
        ..crate::Settings::for_test(std::path::PathBuf::from("data"))
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

/// The embedded shipped catalog; no filesystem, network or model execution.
pub(crate) fn shipped_catalog_entry(model_id: &str) -> sceneworks_core::contracts::JsonObject {
    sceneworks_core::builtin_manifests::builtin_model_entry(model_id)
        .unwrap_or_else(|| panic!("{model_id} is in the shipped catalog"))
        .clone()
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
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
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
