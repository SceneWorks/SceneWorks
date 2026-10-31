//! Coalesce provider activity without queuing obsolete verification messages behind slow POSTs.

use gen_core::{LoadActivity, LoadActivityObserver, LoadActivityPhase};
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
    state.activity.take()
}

pub(super) fn message(activity: Option<LoadActivity>) -> String {
    let Some(activity) = activity else {
        return "preparing model".to_owned();
    };
    let verb = match activity.phase {
        LoadActivityPhase::Checking => "checking",
        LoadActivityPhase::Verifying => "verifying",
        LoadActivityPhase::Loading => "loading",
    };
    let file = activity
        .file
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let mut message = format!("{verb} {} — {file}", activity.component);
    // A selected component inventory is authoritative when present. Never manufacture a
    // percentage or a completed read from a mapped file, an unknown total, or a scope ending.
    let bytes = activity
        .component_total_bytes
        .map(|total| (activity.component_bytes_read, Some(total)))
        .unwrap_or((activity.bytes_read, activity.total_bytes));
    if let Some(total) = bytes.1 {
        message.push_str(&format!(" ({}/{total} bytes)", bytes.0));
    } else if bytes.0 > 0 {
        message.push_str(&format!(" ({} bytes read)", bytes.0));
    }
    if let Some(total) = activity.total_files {
        message.push_str(&format!(
            " ({}/{total} files checked)",
            activity.files_checked
        ));
    }
    message
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
        let text = LoadActivityScope::file(
            LoadActivityPhase::Loading,
            "text_encoder/model.safetensors",
            Some(100),
        );
        text.update(20);
        for _ in 0..100 {
            let _hash = LoadActivityScope::file(
                LoadActivityPhase::Verifying,
                "text_encoder/model.safetensors",
                None,
            );
        }
        let pending = rx.try_recv().unwrap();
        let activity = take(&pending).unwrap();
        assert_eq!(activity.phase, LoadActivityPhase::Loading);
        assert_eq!(activity.bytes_read, 20);
        assert_eq!(activity.component, "text encoder");
        assert!(
            rx.try_recv().is_err(),
            "hashes must not fill a stale event backlog"
        );

        let hash = LoadActivityScope::file(
            LoadActivityPhase::Verifying,
            "text_encoder/model.safetensors",
            None,
        );
        let pending = rx.try_recv().unwrap();
        assert_eq!(
            message(take(&pending)),
            "verifying text encoder — model.safetensors"
        );
        drop(hash);
        let pending = rx.try_recv().unwrap();
        assert_eq!(
            message(take(&pending)),
            "loading text encoder — model.safetensors (20/100 bytes)"
        );
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
        let renderer = LoadActivityScope::file(
            LoadActivityPhase::Loading,
            "transformer/model.safetensors",
            None,
        );
        let pending = rx.try_recv().unwrap();
        assert_eq!(
            message(take(&pending)),
            "loading transformer — model.safetensors"
        );
        drop(renderer);
        let pending = rx.try_recv().unwrap();
        assert_eq!(take(&pending), None);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn component_totals_and_unknown_reads_do_not_invent_completion() {
        let mut activity = LoadActivity {
            phase: LoadActivityPhase::Checking,
            component: "novel tower".into(),
            file: "private/root/shard.safetensors".into(),
            bytes_read: 7,
            total_bytes: None,
            component_bytes_read: 21,
            component_total_bytes: Some(200),
            files_checked: 1,
            total_files: Some(3),
        };
        assert_eq!(
            message(Some(activity.clone())),
            "checking novel tower — shard.safetensors (21/200 bytes) (1/3 files checked)"
        );
        activity.component_total_bytes = None;
        activity.total_files = None;
        assert_eq!(
            message(Some(activity.clone())),
            "checking novel tower — shard.safetensors (7 bytes read)"
        );
        activity.bytes_read = 0;
        assert_eq!(
            message(Some(activity)),
            "checking novel tower — shard.safetensors"
        );
        assert_eq!(message(None), "preparing model");
    }
}
