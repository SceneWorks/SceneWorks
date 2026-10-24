//! WebView2 renderer-death recovery (Windows).
//!
//! Motivating incident (2026-10-22): mid-batch, Chromium terminated the webview's
//! render process (software-raised exception 0xE0000008 out of KERNELBASE, i.e. a
//! managed kill — OOM under system commit pressure while WDDM was spilling ~76 GB
//! of GPU memory, or the hang watchdog; Crashpad dump in the EBWebView profile).
//! The browser process stayed alive but the app had no `ProcessFailed` handler, so
//! the window painted white for the rest of the session while the backend kept
//! rendering jobs the user could no longer watch.
//!
//! WebView2 reports child-process death through `ICoreWebView2.ProcessFailed`.
//! For the renderer kinds the documented recovery is `ICoreWebView2::Reload`,
//! which spawns a fresh renderer and reloads the page; the frontend then
//! re-mounts and re-syncs from the API sidecar, which owns all job state, so no
//! work is lost. This module installs that handler and rate-limits the reload so
//! a persistently-dying renderer cannot loop forever.
//!
//! Structure mirrors `cuda_provision_check.rs`: the decision logic is pure and
//! NOT cfg-gated (unit-tested on every host — no CI lane cargo-tests Windows
//! crates); only the COM wiring is `#[cfg(windows)]`.

#[cfg(any(windows, test))]
use std::collections::VecDeque;
#[cfg(any(windows, test))]
use std::time::{Duration, Instant};

/// Mirrors `COREWEBVIEW2_PROCESS_FAILED_KIND` from the WebView2 SDK
/// (WebView2.idl). Duplicated as plain `i32`s so `recovery_action` can be
/// exercised in tests on any platform without the Windows-only bindings.
/// Listed in full for readability; the match only names the renderer kinds and
/// the tests are the only consumers of the rest.
#[cfg(any(windows, test))]
#[allow(dead_code)]
mod kinds {
    pub const BROWSER_PROCESS_EXITED: i32 = 0;
    pub const RENDER_PROCESS_EXITED: i32 = 1;
    pub const RENDER_PROCESS_UNRESPONSIVE: i32 = 2;
    pub const FRAME_RENDER_PROCESS_EXITED: i32 = 3;
    pub const UTILITY_PROCESS_EXITED: i32 = 4;
    pub const SANDBOX_HELPER_PROCESS_EXITED: i32 = 5;
    pub const GPU_PROCESS_EXITED: i32 = 6;
    pub const PPAPI_PLUGIN_PROCESS_EXITED: i32 = 7;
    pub const PPAPI_BROKER_PROCESS_EXITED: i32 = 8;
    pub const UNKNOWN_PROCESS_EXITED: i32 = 9;
}

/// What to do when WebView2 reports a child process failure.
#[cfg(any(windows, test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryAction {
    /// `ICoreWebView2::Reload` — spawns a new renderer and reloads the page.
    Reload,
    /// Recovery is WebView2's job (GPU/utility processes are auto-relaunched)
    /// or impossible in place (browser process gone — the webview itself must
    /// be recreated, which wry cannot do from inside the event handler).
    LogOnly,
}

/// Only renderer failures are recoverable by reload: the page's own renderer
/// died (or hung), and reloading the page is the documented fix. A frame
/// renderer dying leaves the page half-functional, so reload it too. Every
/// other kind either heals itself or is beyond an in-place reload.
#[cfg(any(windows, test))]
pub(crate) fn recovery_action(kind: i32) -> RecoveryAction {
    match kind {
        kinds::RENDER_PROCESS_EXITED
        | kinds::RENDER_PROCESS_UNRESPONSIVE
        | kinds::FRAME_RENDER_PROCESS_EXITED => RecoveryAction::Reload,
        _ => RecoveryAction::LogOnly,
    }
}

/// Caps reloads so a renderer that dies on every launch can't send the window
/// into a reload loop (each reload re-runs the frontend boot, which is itself
/// expensive). Once `max` reloads happen inside `window`, further failures are
/// logged but left white — at that point something is systematically wrong and
/// churning is worse than a dead-but-stable window.
#[cfg(any(windows, test))]
pub(crate) struct ReloadGuard {
    max: usize,
    window: Duration,
    stamps: VecDeque<Instant>,
}

#[cfg(any(windows, test))]
impl ReloadGuard {
    pub(crate) fn new(max: usize, window: Duration) -> Self {
        Self {
            max,
            window,
            stamps: VecDeque::new(),
        }
    }

    /// Returns true and records a reload if fewer than `max` reloads occurred
    /// in the trailing `window`; false (no record) once the cap is hit.
    pub(crate) fn try_acquire(&mut self, now: Instant) -> bool {
        while let Some(&oldest) = self.stamps.front() {
            if now.duration_since(oldest) >= self.window {
                self.stamps.pop_front();
            } else {
                break;
            }
        }
        if self.stamps.len() >= self.max {
            return false;
        }
        self.stamps.push_back(now);
        true
    }
}

/// At most 3 auto-reloads within 5 minutes. A single renderer death during a
/// heavy batch is the expected case; three in five minutes means the page
/// itself is toxic and a human should look at the logs instead of a blinking
/// window.
#[cfg(windows)]
const MAX_RELOADS: usize = 3;
#[cfg(windows)]
const RELOAD_WINDOW: Duration = Duration::from_secs(300);

/// Installs the `ProcessFailed` handler on the main window's WebView2 control.
///
/// `with_webview` posts the closure onto the event loop, so the COM calls run
/// on the UI thread that owns the webview (WebView2 is apartment-bound); the
/// event then fires on that same thread, making a synchronous `Reload` inside
/// the handler safe. Failures here are logged but never fatal — a missing
/// handler only means we fall back to today's behavior (white window).
#[cfg(windows)]
pub(crate) fn install_process_failed_recovery(window: &tauri::WebviewWindow) {
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2ProcessFailedEventArgs2, COREWEBVIEW2_PROCESS_FAILED_KIND,
        COREWEBVIEW2_PROCESS_FAILED_REASON,
    };
    use webview2_com::ProcessFailedEventHandler;
    use windows_core::Interface;

    let window = window.clone();
    if let Err(e) = window.with_webview(move |platform| {
        let controller = platform.controller();
        let core = match unsafe { controller.CoreWebView2() } {
            Ok(core) => core,
            Err(e) => {
                tracing::error!("webview resilience: CoreWebView2 unavailable, ProcessFailed recovery not installed: {e}");
                return;
            }
        };

        let mut guard = ReloadGuard::new(MAX_RELOADS, RELOAD_WINDOW);
        let handler = ProcessFailedEventHandler::create(Box::new(move |sender, args| {
            let Some(args) = args else {
                return Ok(());
            };
            let mut kind = COREWEBVIEW2_PROCESS_FAILED_KIND(0);
            unsafe {
                let _ = args.ProcessFailedKind(&mut kind);
            }
            // Exit code, description, and Reason (OOM vs crash vs terminated, …)
            // all live on the args2 interface; absent on very old runtimes,
            // hence the fallible cast.
            let (exit_code, description, reason) =
                match args.cast::<ICoreWebView2ProcessFailedEventArgs2>() {
                Ok(args2) => {
                    let mut exit_code = 0;
                    let mut description = windows_core::PWSTR::null();
                    let mut reason = COREWEBVIEW2_PROCESS_FAILED_REASON(0);
                    unsafe {
                        let _ = args2.ExitCode(&mut exit_code);
                        let _ = args2.ProcessDescription(&mut description);
                        let _ = args2.Reason(&mut reason);
                    }
                    let description = unsafe { description.to_string() }.unwrap_or_default();
                    (exit_code, description, Some(reason.0))
                }
                Err(_) => (0, String::new(), None),
            };

            match recovery_action(kind.0) {
                RecoveryAction::Reload => {
                    if guard.try_acquire(Instant::now()) {
                        tracing::error!(
                            "webview child process failed (kind={} reason={reason:?} exit_code={exit_code} description={description:?}); \
                             reloading the webview to recover the UI (backend jobs are unaffected)",
                            kind.0
                        );
                        if let Some(webview) = sender {
                            unsafe { webview.Reload() }?;
                        }
                    } else {
                        tracing::error!(
                            "webview child process failed (kind={} reason={reason:?} exit_code={exit_code} description={description:?}) \
                             but the reload cap ({MAX_RELOADS} per {}s) is exhausted; leaving the window as-is",
                            kind.0,
                            RELOAD_WINDOW.as_secs()
                        );
                    }
                }
                RecoveryAction::LogOnly => {
                    tracing::error!(
                        "webview child process failed (kind={} reason={reason:?} exit_code={exit_code} description={description:?}); \
                         no in-place recovery for this process kind",
                        kind.0
                    );
                }
            }
            Ok(())
        }));

        let mut token = 0i64;
        if let Err(e) = unsafe { core.add_ProcessFailed(&handler, &mut token) } {
            tracing::error!("webview resilience: add_ProcessFailed failed: {e}");
        } else {
            tracing::info!("webview resilience: ProcessFailed recovery installed");
        }
    }) {
        tracing::error!("webview resilience: with_webview dispatch failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_kinds_reload() {
        assert_eq!(
            recovery_action(kinds::RENDER_PROCESS_EXITED),
            RecoveryAction::Reload
        );
        assert_eq!(
            recovery_action(kinds::RENDER_PROCESS_UNRESPONSIVE),
            RecoveryAction::Reload
        );
        assert_eq!(
            recovery_action(kinds::FRAME_RENDER_PROCESS_EXITED),
            RecoveryAction::Reload
        );
    }

    #[test]
    fn non_renderer_kinds_only_log() {
        // The browser process taking the whole control down can't be reloaded
        // away; GPU/utility/plugin/helper deaths are WebView2's to relaunch.
        for kind in [
            kinds::BROWSER_PROCESS_EXITED,
            kinds::UTILITY_PROCESS_EXITED,
            kinds::SANDBOX_HELPER_PROCESS_EXITED,
            kinds::GPU_PROCESS_EXITED,
            kinds::PPAPI_PLUGIN_PROCESS_EXITED,
            kinds::PPAPI_BROKER_PROCESS_EXITED,
            kinds::UNKNOWN_PROCESS_EXITED,
            -1,
            i32::MAX,
        ] {
            assert_eq!(
                recovery_action(kind),
                RecoveryAction::LogOnly,
                "kind {kind}"
            );
        }
    }

    #[test]
    fn guard_allows_up_to_max_then_blocks() {
        let t0 = Instant::now();
        let mut guard = ReloadGuard::new(3, Duration::from_secs(300));
        assert!(guard.try_acquire(t0));
        assert!(guard.try_acquire(t0 + Duration::from_secs(10)));
        assert!(guard.try_acquire(t0 + Duration::from_secs(20)));
        assert!(!guard.try_acquire(t0 + Duration::from_secs(30)));
    }

    #[test]
    fn guard_recovers_after_the_window_passes() {
        let t0 = Instant::now();
        let mut guard = ReloadGuard::new(2, Duration::from_secs(300));
        assert!(guard.try_acquire(t0));
        assert!(guard.try_acquire(t0 + Duration::from_secs(10)));
        assert!(!guard.try_acquire(t0 + Duration::from_secs(20)));
        // Both stamps age out at t0+300, so the next failure reloads again.
        assert!(guard.try_acquire(t0 + Duration::from_secs(301)));
    }

    #[test]
    fn blocked_attempts_do_not_consume_budget() {
        let t0 = Instant::now();
        let mut guard = ReloadGuard::new(1, Duration::from_secs(300));
        assert!(guard.try_acquire(t0));
        // Repeated blocked attempts must not extend the lockout.
        for i in 1..10 {
            assert!(!guard.try_acquire(t0 + Duration::from_secs(i)));
        }
        assert!(guard.try_acquire(t0 + Duration::from_secs(300)));
    }
}
