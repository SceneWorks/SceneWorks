//! In-app cross-platform auto-updater (sc-1355).
//!
//! On launch the release build asks the GitHub "latest release" pointer
//! (`plugins.updater.endpoints` in `tauri.conf.json`) whether a newer build
//! exists for this platform. The endpoint serves a `latest.json` manifest with a
//! per-target `platforms` map; `tauri-plugin-updater` picks `darwin-aarch64`,
//! `windows-x86_64`, or `linux-x86_64` for the running build, verifies the minisign
//! signature against `plugins.updater.pubkey`, and — on user accept — downloads,
//! installs, and restarts into the new version. Linux self-update is supported by
//! the AppImage release; `.deb` users upgrade through their package installer.
//!
//! The shell checks at startup and hourly. The sidebar reads the cached offer;
//! downloads and installation stay native so remote browsers cannot self-update.

use std::{
    ffi::{OsStr, OsString},
    path::Path,
    sync::Mutex,
    time::Duration,
};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};
use tauri_plugin_updater::{Update, UpdaterExt};

#[derive(Default)]
pub struct UpdateState(Mutex<UpdateSession>);

#[derive(Default)]
struct UpdateSession {
    available: Option<Update>,
    downloaded: Option<(Update, Vec<u8>)>,
    downloading: bool,
    startup_requested: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateStatus {
    version: Option<String>,
    downloading: bool,
    startup_requested: bool,
}

#[tauri::command]
pub fn get_update_status(state: State<'_, UpdateState>, ready: Option<bool>) -> UpdateStatus {
    let mut session = state.0.lock().expect("update lock");
    UpdateStatus {
        version: session
            .available
            .as_ref()
            .map(|update| update.version.clone()),
        downloading: session.downloading,
        startup_requested: ready.unwrap_or(false) && std::mem::take(&mut session.startup_requested),
    }
}

/// Release bundles only; Linux package-manager installations cannot self-update.
pub fn spawn_startup_check(app: &AppHandle) {
    if cfg!(debug_assertions) {
        return;
    }
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_none() {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut startup = true;
        let mut checks = tokio::time::interval(Duration::from_secs(60 * 60));
        checks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            checks.tick().await;
            match tokio::time::timeout(Duration::from_secs(30), check_for_update(&app, startup))
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(err)) => tracing::warn!(error = %err, "auto-update: check failed"),
                Err(err) => tracing::warn!(error = %err, "auto-update: check timed out"),
            }
            startup = false;
        }
    });
}

async fn check_for_update(app: &AppHandle, startup: bool) -> tauri_plugin_updater::Result<()> {
    let updater = app.updater_builder();
    #[cfg(target_os = "windows")]
    let updater = match std::env::current_exe()
        .ok()
        .and_then(|current_exe| nsis_install_dir_arg(&current_exe))
    {
        Some(install_dir_arg) => updater.installer_arg(install_dir_arg),
        None => updater,
    };
    let update = updater.build()?.check().await?;
    let prompt = if startup {
        update
            .as_ref()
            .map(|update| (update.current_version.clone(), update.version.clone()))
    } else {
        None
    };
    {
        let state = app.state::<UpdateState>();
        let mut session = state.0.lock().expect("update lock");
        if !session.downloading && session.downloaded.is_none() {
            session.available = update;
        }
    }
    let _ = app.emit("app-update-changed", ());
    if let Some((current, latest)) = prompt {
        let app_for_request = app.clone();
        app.dialog()
            .message(format!(
                "SceneWorks {latest} is available (you have {current}).\n\n\
                 Download and install it now? SceneWorks will restart to finish."
            ))
            .title("Update available")
            .kind(MessageDialogKind::Info)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Update now".into(),
                "Later".into(),
            ))
            .show(move |accepted| {
                if accepted {
                    app_for_request
                        .state::<UpdateState>()
                        .0
                        .lock()
                        .expect("update lock")
                        .startup_requested = true;
                    // The request is retained until the API-backed UI is ready, even if
                    // the dialog was accepted before that webview finished loading.
                    let _ = app_for_request.emit("app-update-changed", ());
                }
            });
    }
    Ok(())
}

/// Download and verify while sidecars remain available for the final work check.
#[tauri::command]
pub async fn download_app_update(app: AppHandle) -> Result<(), String> {
    let state = app.state::<UpdateState>();
    let update = {
        let mut session = state.0.lock().expect("update lock");
        if session.downloading {
            return Err("An update is already downloading.".into());
        }
        if session.downloaded.is_some() {
            return Ok(());
        }
        let update = session.available.clone().ok_or("No update is available.")?;
        session.downloading = true;
        update
    };
    let result = update.download(|_, _| {}, || {}).await;
    let mut session = state.0.lock().expect("update lock");
    session.downloading = false;
    match result {
        Ok(bytes) => {
            session.downloaded = Some((update, bytes));
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}

/// Release downloaded bytes when the user defers or the final work check fails.
#[tauri::command]
pub fn discard_app_update(state: State<'_, UpdateState>) {
    state.0.lock().expect("update lock").downloaded = None;
}

/// The UI checks/cancels live work again after the download before invoking this.
#[tauri::command]
pub async fn install_app_update(app: AppHandle) -> Result<(), String> {
    let prepared = app
        .state::<UpdateState>()
        .0
        .lock()
        .expect("update lock")
        .downloaded
        .take()
        .ok_or("The update has not finished downloading.")?;
    install_update(&app, prepared.0, prepared.1).await;
    Ok(())
}

/// Build NSIS's destination-directory argument from the running executable.
///
/// Construct this as an [`OsString`] rather than formatting a display path so
/// valid non-UTF-8 Windows paths are not changed. NSIS requires `/D=` to be the
/// final argument; the updater builder preserves insertion order and appends
/// installer arguments last.
///
/// Its only non-test caller is `#[cfg(target_os = "windows")]` — NSIS is the Windows
/// installer — but the tests below are unconditional, because the path logic they cover is
/// platform-neutral. That combination makes this dead code in a non-Windows *bin* build
/// while it stays live in the test target, so the carve-out is platform-scoped rather than
/// blanket (sc-16269).
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn nsis_install_dir_arg(current_exe: &Path) -> Option<OsString> {
    let install_dir = current_exe.parent()?;
    if install_dir.as_os_str().is_empty() {
        return None;
    }

    let mut arg = OsString::from(OsStr::new("/D="));
    arg.push(install_dir.as_os_str());
    Some(arg)
}

/// Install the downloaded, verified update, then restart into it. `restart()`
/// diverges (`-> !`), so it is the function's tail and nothing runs after it.
///
/// Download and install are split deliberately (sc-11015): the signed bundle is
/// fetched + verified FIRST, while the app and its sidecars keep running, so a
/// download/verify failure leaves the session fully intact (the caller shows a
/// recoverable error and the app stays usable). Only with the bytes in hand do we stop
/// the `sceneworks-api` sidecars — the Windows NSIS installer overwrites
/// `sceneworks-api.exe` in place, and a still-live sidecar holds a lock on it, which is
/// what produced the "Error opening file for writing: …\sceneworks-api.exe" abort.
/// `tauri-plugin-updater`'s NSIS `/UPDATE` only kills the main app binary
/// (`SceneWorks.exe`), never the sidecars this shell spawned.
async fn install_update(app: &AppHandle, update: Update, bytes: Vec<u8>) {
    // Stop the API + GPU-worker sidecars and BLOCK until `sceneworks-api.exe` is no
    // longer running, so the installer can overwrite it (no-op wait off Windows).
    tracing::info!("auto-update: stopping sidecars before install");
    crate::setup::stop_sidecars_for_update(app);

    // Hand off to the installer. On Windows `install` launches the NSIS updater and
    // terminates this process (the installer relaunches into the new build), so the
    // `restart()` tail below is reached only on macOS/Linux, where `install` returns.
    tracing::info!("auto-update: installing");
    if let Err(err) = update.install(bytes) {
        // The sidecars are already down, so the running session can't recover in place.
        // Tell the user, then relaunch to come back cleanly on the current (un-updated)
        // build rather than sit with a dead API window; they can retry the update.
        tracing::error!(error = %err, "auto-update: install failed after sidecar teardown; restarting to recover");
        app.dialog()
            .message(
                "The update could not be installed. SceneWorks will restart on the \
                 current version — you can try updating again from the releases page.",
            )
            .title("Update failed")
            .kind(MessageDialogKind::Error)
            .blocking_show();
        app.restart();
    }
    tracing::info!("auto-update: installed, restarting");
    app.restart()
}

#[cfg(test)]
mod tests {
    use super::nsis_install_dir_arg;
    use std::{ffi::OsString, path::Path};

    #[test]
    fn nsis_install_dir_uses_current_executable_parent() {
        assert_eq!(
            nsis_install_dir_arg(Path::new("/opt/SceneWorks/SceneWorks")),
            Some(OsString::from("/D=/opt/SceneWorks"))
        );
    }

    #[test]
    fn nsis_install_dir_preserves_spaces() {
        assert_eq!(
            nsis_install_dir_arg(Path::new("/opt/Program Files/SceneWorks/SceneWorks")),
            Some(OsString::from("/D=/opt/Program Files/SceneWorks"))
        );
    }

    #[test]
    fn nsis_install_dir_rejects_parentless_path() {
        assert_eq!(nsis_install_dir_arg(Path::new("")), None);
    }
}
