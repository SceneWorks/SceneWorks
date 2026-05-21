//! Desktop settings surface (sc-1350): data directory, Hugging Face token (OS
//! keychain), detected GPU info, and a worker restart. Commands are invoked from
//! the React settings screen when running inside the Tauri shell.

use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::DialogExt;

use crate::setup::{app_support_dir, Managed};

const KEYRING_SERVICE: &str = "SceneWorks";
const HF_TOKEN_ACCOUNT: &str = "huggingface_token";

#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    /// Override for the data directory; `None` uses the platform default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
}

fn settings_path() -> PathBuf {
    app_support_dir().join("settings.json")
}

pub fn load_settings() -> AppSettings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|body| serde_json::from_str(&body).ok())
        .unwrap_or_default()
}

fn save_settings(settings: &AppSettings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let body = serde_json::to_string_pretty(settings).map_err(|error| error.to_string())?;
    std::fs::write(&path, body).map_err(|error| error.to_string())
}

/// Hugging Face token from the OS keychain, used to inject `HF_TOKEN` into the
/// worker. Returns `None` when unset or unreadable.
pub fn read_hf_token() -> Option<String> {
    keyring::Entry::new(KEYRING_SERVICE, HF_TOKEN_ACCOUNT)
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    platform: String,
    devices: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unified_memory_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wired_limit_mb: Option<u64>,
}

fn run_capture(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[tauri::command]
pub fn get_app_settings() -> AppSettings {
    load_settings()
}

#[tauri::command]
pub fn set_data_dir(path: String) -> Result<AppSettings, String> {
    let mut settings = load_settings();
    let trimmed = path.trim();
    settings.data_dir = if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    };
    save_settings(&settings)?;
    Ok(settings)
}

#[tauri::command]
pub async fn choose_data_dir(app: AppHandle) -> Option<String> {
    app.dialog()
        .file()
        .blocking_pick_folder()
        .and_then(|file| file.into_path().ok())
        .map(|path| path.to_string_lossy().into_owned())
}

#[tauri::command]
pub fn reveal_in_os(path: String) -> Result<(), String> {
    let target = PathBuf::from(&path);
    let result = if cfg!(target_os = "macos") {
        Command::new("open").arg("-R").arg(&target).status()
    } else if cfg!(target_os = "windows") {
        Command::new("explorer")
            .arg(format!("/select,{}", target.display()))
            .status()
    } else {
        let dir = target.parent().unwrap_or(&target);
        Command::new("xdg-open").arg(dir).status()
    };
    result.map(|_| ()).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn hf_token_present() -> bool {
    read_hf_token().is_some()
}

#[tauri::command]
pub fn set_hf_token(token: String) -> Result<(), String> {
    let entry = keyring::Entry::new(KEYRING_SERVICE, HF_TOKEN_ACCOUNT)
        .map_err(|error| error.to_string())?;
    let trimmed = token.trim();
    if trimmed.is_empty() {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    } else {
        entry
            .set_password(trimmed)
            .map_err(|error| error.to_string())
    }
}

#[tauri::command]
pub fn restart_worker(app: AppHandle) {
    // Kill the current worker child; the supervisor restarts it.
    if let Some(child) = app
        .state::<Managed>()
        .worker
        .lock()
        .expect("worker lock")
        .take()
    {
        let _ = child.kill();
    }
}

#[tauri::command]
pub fn get_gpu_info() -> GpuInfo {
    #[cfg(target_os = "macos")]
    {
        let mut devices = Vec::new();
        if let Some(profile) = run_capture("system_profiler", &["SPDisplaysDataType"]) {
            for line in profile.lines() {
                if let Some((_, model)) = line.trim().split_once("Chipset Model:") {
                    devices.push(model.trim().to_owned());
                }
            }
        }
        let unified_memory_mb = run_capture("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok())
            .map(|bytes| bytes / (1024 * 1024));
        let wired_limit_mb = run_capture("sysctl", &["-n", "iogpu.wired_limit_mb"])
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0);
        GpuInfo {
            platform: "macos".to_owned(),
            devices,
            unified_memory_mb,
            wired_limit_mb,
        }
    }
    #[cfg(target_os = "windows")]
    {
        let mut devices = Vec::new();
        if let Some(output) = run_capture(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
        ) {
            for line in output.lines() {
                let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                match parts.as_slice() {
                    [name, memory, ..] => devices.push(format!("{name} ({memory} MB)")),
                    [name] => devices.push((*name).to_owned()),
                    _ => {}
                }
            }
        }
        GpuInfo {
            platform: "windows".to_owned(),
            devices,
            unified_memory_mb: None,
            wired_limit_mb: None,
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let devices = run_capture("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"])
            .map(|output| output.lines().map(str::to_owned).collect())
            .unwrap_or_default();
        GpuInfo {
            platform: "linux".to_owned(),
            devices,
            unified_memory_mb: None,
            wired_limit_mb: None,
        }
    }
}
