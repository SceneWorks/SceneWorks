// Hide the extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// How long to wait for the bundled API to answer /health before showing an
/// error in the window.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(30);

/// Holds the managed API sidecar child so it can be killed on app exit.
struct ApiSidecar(Mutex<Option<CommandChild>>);

/// Reserve an ephemeral localhost port by binding to :0 and releasing it. The
/// sidecar then binds the same port; the small race window is acceptable for a
/// single-user desktop app.
fn reserve_free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Platform-appropriate logs directory:
/// - macOS: `~/Library/Logs/SceneWorks`
/// - Windows: `%LOCALAPPDATA%\SceneWorks\logs`
/// - Linux: `$XDG_STATE_HOME/sceneworks/logs` or `~/.local/state/sceneworks/logs`
fn logs_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Logs")
            .join("SceneWorks");
    }
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("SceneWorks").join("logs");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Ok(state) = std::env::var("XDG_STATE_HOME") {
            return PathBuf::from(state).join("sceneworks").join("logs");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("state")
                .join("sceneworks")
                .join("logs");
        }
    }
    std::env::temp_dir().join("SceneWorks").join("logs")
}

/// Minimal blocking HTTP/1.0 GET of /api/v1/health; true on a 200 status line.
fn health_ok(port: u16) -> bool {
    use std::io::Read;
    use std::net::TcpStream;
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /api/v1/health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response
        .lines()
        .next()
        .is_some_and(|status_line| status_line.contains(" 200"))
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(ApiSidecar(Mutex::new(None)))
        .setup(|app| {
            let port = reserve_free_port()?;
            let base_url = format!("http://127.0.0.1:{port}");

            let dir = logs_dir();
            let _ = std::fs::create_dir_all(&dir);
            let log_path = dir.join("api.log");

            // Spawn the bundled API binary as a managed sidecar. Desktop runs the
            // single-process model: the API also runs the utility worker in-process.
            let (mut events, child) = app
                .shell()
                .sidecar("sceneworks-api")?
                .env("SCENEWORKS_API_HOST", "127.0.0.1")
                .env("SCENEWORKS_API_PORT", port.to_string())
                .env("SCENEWORKS_RUN_UTILITY_INPROCESS", "true")
                .spawn()?;
            app.state::<ApiSidecar>()
                .0
                .lock()
                .expect("sidecar lock")
                .replace(child);

            // Forward sidecar stdout/stderr to the log file.
            tauri::async_runtime::spawn(async move {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .ok();
                while let Some(event) = events.recv().await {
                    let entry = match event {
                        CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                            String::from_utf8_lossy(&bytes).into_owned()
                        }
                        CommandEvent::Terminated(payload) => format!(
                            "[desktop] api sidecar terminated: code={:?} signal={:?}\n",
                            payload.code, payload.signal
                        ),
                        CommandEvent::Error(error) => {
                            format!("[desktop] api sidecar error: {error}\n")
                        }
                        _ => continue,
                    };
                    if let Some(file) = file.as_mut() {
                        let _ = file.write_all(entry.as_bytes());
                        let _ = file.flush();
                    }
                }
            });

            // Health-gate the window: navigate to the local API once it answers,
            // or show a failure message after the timeout.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let deadline = Instant::now() + HEALTH_TIMEOUT;
                loop {
                    if health_ok(port) {
                        if let (Some(window), Ok(url)) =
                            (handle.get_webview_window("main"), base_url.parse())
                        {
                            let _ = window.navigate(url);
                        }
                        return;
                    }
                    if Instant::now() >= deadline {
                        if let Some(window) = handle.get_webview_window("main") {
                            let _ = window.eval(
                                "document.body.innerHTML = '<main><h1>SceneWorks</h1>\
                                 <p>The local API did not start in time. Check api.log.</p></main>';",
                            );
                        }
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(300));
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the SceneWorks desktop shell");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            if let Some(child) = app_handle
                .state::<ApiSidecar>()
                .0
                .lock()
                .expect("sidecar lock")
                .take()
            {
                let _ = child.kill();
            }
        }
    });
}
