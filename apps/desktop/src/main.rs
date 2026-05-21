// Hide the extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod setup;

use tauri::{Manager, RunEvent};

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(setup::Managed::default())
        .invoke_handler(tauri::generate_handler![setup::start_setup])
        .build(tauri::generate_context!())
        .expect("error while building the SceneWorks desktop shell");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            if let Some(child) = app_handle
                .state::<setup::Managed>()
                .api
                .lock()
                .expect("api lock")
                .take()
            {
                let _ = child.kill();
            }
        }
    });
}
