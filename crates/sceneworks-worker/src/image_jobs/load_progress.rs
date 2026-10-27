//! Coalesce provider activity without queuing obsolete verification messages behind slow POSTs.

use gen_core::{LoadActivity, LoadActivityObserver};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(super) struct PendingLoadActivity {
    activity: Option<LoadActivity>,
    queued: bool,
}

pub(super) fn observe(
    wake: impl Fn(Arc<Mutex<PendingLoadActivity>>) + Send + Sync + 'static,
) -> LoadActivityObserver {
    let pending = Arc::new(Mutex::new(PendingLoadActivity::default()));
    LoadActivityObserver::new(move |activity| {
        // A cold cache may hash thousands of tensor payloads. Retain just the latest state
        // behind one ordered wakeup, including the transition OUT of verification.
        let send = {
            let mut state = pending.lock().unwrap_or_else(|error| error.into_inner());
            state.activity = activity;
            !std::mem::replace(&mut state.queued, true)
        };
        if send {
            wake(pending.clone());
        }
    })
}

pub(super) fn take(pending: &Mutex<PendingLoadActivity>) -> Option<LoadActivity> {
    let mut state = pending.lock().unwrap_or_else(|error| error.into_inner());
    state.queued = false;
    state.activity
}

pub(super) fn message(activity: Option<LoadActivity>) -> &'static str {
    match activity {
        Some(LoadActivity::VerifyingFiles) => "verifying model files",
        Some(LoadActivity::TextEncoder) => "loading text encoder",
        Some(LoadActivity::Renderer) => "loading render components",
        Some(LoadActivity::EncodingPrompt) => "encoding prompt",
        None => "preparing model",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::LoadActivityScope;

    #[test]
    fn load_activity_coalesces_hashes_and_restores_the_actual_component() {
        let (tx, rx) = std::sync::mpsc::channel();
        let _observer = observe(move |pending| {
            let _ = tx.send(pending);
        });
        let text = LoadActivityScope::enter(LoadActivity::TextEncoder);
        for _ in 0..100 {
            let _hash = LoadActivityScope::enter(LoadActivity::VerifyingFiles);
        }
        let pending = rx.try_recv().unwrap();
        assert_eq!(take(&pending), Some(LoadActivity::TextEncoder));
        assert!(
            rx.try_recv().is_err(),
            "hashes must not fill a stale event backlog"
        );

        let hash = LoadActivityScope::enter(LoadActivity::VerifyingFiles);
        let pending = rx.try_recv().unwrap();
        assert_eq!(message(take(&pending)), "verifying model files");
        drop(hash);
        let pending = rx.try_recv().unwrap();
        assert_eq!(message(take(&pending)), "loading text encoder");
        drop(text);
        let pending = rx.try_recv().unwrap();
        assert_eq!(take(&pending), None);
    }

    #[test]
    fn a_warm_load_does_not_invent_a_verification_phase() {
        let (tx, rx) = std::sync::mpsc::channel();
        let _observer = observe(move |pending| {
            let _ = tx.send(pending);
        });
        assert!(
            rx.try_recv().is_err(),
            "installing an observer does no verification work"
        );
        let renderer = LoadActivityScope::enter(LoadActivity::Renderer);
        let pending = rx.try_recv().unwrap();
        assert_eq!(message(take(&pending)), "loading render components");
        drop(renderer);
        let pending = rx.try_recv().unwrap();
        assert_eq!(take(&pending), None);
        assert!(rx.try_recv().is_err());
    }
}
