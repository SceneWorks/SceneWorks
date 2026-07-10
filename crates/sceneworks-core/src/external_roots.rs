//! Operator-configured external model roots (epic 10451 / sc-10452).
//!
//! Users arrive with gigabytes of weights already on disk — most often a ComfyUI
//! `models/` tree — and re-downloading them into the Hugging Face cache is the
//! single loudest complaint about install. An external root lets the app *read*
//! those files in place, with no copy and no re-download.
//!
//! Two properties make this safe enough to ship:
//!
//! 1. **Operator-set, never payload-derived.** The roots come from the process
//!    environment (`SCENEWORKS_EXTERNAL_MODEL_ROOTS`), so a job payload can never
//!    introduce one. The jobs API is LAN-exposed (epic 4484), and an unconfined
//!    weight path is an arbitrary-file-read primitive; widening the allow-list is
//!    a deployment decision, not a request-time one. Payload paths are still
//!    confined — they must resolve *under* an allowed root
//!    (`sceneworks-worker`'s `normalize_app_managed_lora_path`).
//! 2. **Off by default, and off on macOS.** Unset means the empty list, i.e.
//!    byte-identical prior behaviour. The ComfyUI single-file ecosystem is
//!    overwhelmingly CUDA, so the feature is gated to Windows/Linux rather than
//!    widening the Mac/MLX default for no user benefit.

use std::path::PathBuf;

/// The environment variable holding the operator's external model roots, as an
/// OS-native path list (`;`-separated on Windows, `:`-separated elsewhere —
/// parsed by [`std::env::split_paths`], so a Windows `C:\…` drive letter is not
/// mistaken for a separator).
pub const EXTERNAL_MODEL_ROOTS_ENV: &str = "SCENEWORKS_EXTERNAL_MODEL_ROOTS";

/// The ComfyUI subdirectory holding single-file LoRA adapters, relative to a
/// configured root. A root is expected to be a ComfyUI-style `models/` directory.
pub const COMFYUI_LORA_SUBDIR: &str = "loras";

/// Parse an OS path-list into external model roots.
///
/// Entries must be **absolute**: a root is a deployment-level declaration, and a
/// relative path would silently resolve against whatever working directory the
/// binary happened to start in. Blank entries are dropped and duplicates are
/// removed, preserving the operator's ordering.
///
/// Existence is deliberately *not* required — a root may live on a drive that is
/// not mounted yet — so callers must tolerate a missing directory. Nor is the
/// path canonicalized here: that needs the path to exist, and the confinement
/// checks canonicalize at use time.
pub fn parse_external_model_roots(raw: Option<&str>) -> Vec<PathBuf> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Vec::new();
    };
    let mut roots = Vec::new();
    for path in std::env::split_paths(raw) {
        if path.as_os_str().is_empty() || !path.is_absolute() {
            continue;
        }
        if !roots.contains(&path) {
            roots.push(path);
        }
    }
    roots
}

/// The external model roots for this process: [`parse_external_model_roots`] over
/// [`EXTERNAL_MODEL_ROOTS_ENV`], **always empty on macOS** (see the module docs —
/// the feature is Windows/Linux-gated).
///
/// Read once at binary startup and stored on each `Settings`, so the API and the
/// worker cannot disagree about the allow-list.
pub fn external_model_roots_from_env() -> Vec<PathBuf> {
    if cfg!(target_os = "macos") {
        return Vec::new();
    }
    parse_external_model_roots(std::env::var(EXTERNAL_MODEL_ROOTS_ENV).ok().as_deref())
}

/// The `loras` directory under each configured root that actually exists.
/// Non-existent roots (unmounted drive, typo) contribute nothing rather than
/// erroring — a misconfigured root must not take the catalog down.
pub fn comfyui_lora_dirs(roots: &[PathBuf]) -> Vec<PathBuf> {
    roots
        .iter()
        .map(|root| root.join(COMFYUI_LORA_SUBDIR))
        .filter(|dir| dir.is_dir())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Build an OS-native path list the same way an operator's shell would.
    fn joined(paths: &[&str]) -> String {
        std::env::join_paths(paths.iter().map(Path::new))
            .expect("join_paths")
            .into_string()
            .expect("utf-8")
    }

    #[cfg(windows)]
    const ABS_A: &str = r"C:\models\a";
    #[cfg(windows)]
    const ABS_B: &str = r"D:\weights\b";
    #[cfg(not(windows))]
    const ABS_A: &str = "/models/a";
    #[cfg(not(windows))]
    const ABS_B: &str = "/weights/b";

    #[test]
    fn unset_or_blank_yields_no_roots() {
        assert!(parse_external_model_roots(None).is_empty());
        assert!(parse_external_model_roots(Some("")).is_empty());
        assert!(parse_external_model_roots(Some("   ")).is_empty());
    }

    #[test]
    fn parses_an_os_path_list_of_absolute_roots() {
        let raw = joined(&[ABS_A, ABS_B]);
        assert_eq!(
            parse_external_model_roots(Some(&raw)),
            vec![PathBuf::from(ABS_A), PathBuf::from(ABS_B)]
        );
    }

    /// A Windows root carries a `C:` drive letter; naive `split(':')` would shear
    /// it in half. `split_paths` is OS-aware, so the drive letter survives.
    #[cfg(windows)]
    #[test]
    fn windows_drive_letters_survive_splitting() {
        let roots = parse_external_model_roots(Some(r"C:\models\a;D:\weights\b"));
        assert_eq!(
            roots,
            vec![
                PathBuf::from(r"C:\models\a"),
                PathBuf::from(r"D:\weights\b")
            ]
        );
    }

    /// A relative root would resolve against the process working directory —
    /// non-deterministic across the API and the worker, which start differently.
    #[test]
    fn relative_entries_are_rejected() {
        assert!(parse_external_model_roots(Some("models/loras")).is_empty());
        assert!(parse_external_model_roots(Some("../escape")).is_empty());
    }

    #[test]
    fn duplicates_collapse_and_order_is_preserved() {
        let raw = joined(&[ABS_B, ABS_A, ABS_B]);
        assert_eq!(
            parse_external_model_roots(Some(&raw)),
            vec![PathBuf::from(ABS_B), PathBuf::from(ABS_A)]
        );
    }

    #[test]
    fn comfyui_lora_dirs_skips_missing_roots() {
        let temp = tempfile::tempdir().expect("tempdir");
        let present = temp.path().join("present");
        std::fs::create_dir_all(present.join(COMFYUI_LORA_SUBDIR)).expect("mkdir");
        let absent = temp.path().join("absent");

        let dirs = comfyui_lora_dirs(&[present.clone(), absent]);
        assert_eq!(dirs, vec![present.join(COMFYUI_LORA_SUBDIR)]);
    }

    /// macOS is gated off entirely, so the env var must not introduce roots there.
    /// Off-Mac the var is honored. Asserted against the same process env.
    #[test]
    fn env_reader_is_macos_gated() {
        // Safety: single-threaded test, restored immediately.
        std::env::set_var(EXTERNAL_MODEL_ROOTS_ENV, ABS_A);
        let roots = external_model_roots_from_env();
        std::env::remove_var(EXTERNAL_MODEL_ROOTS_ENV);

        if cfg!(target_os = "macos") {
            assert!(roots.is_empty(), "macOS must never expose external roots");
        } else {
            assert_eq!(roots, vec![PathBuf::from(ABS_A)]);
        }
    }
}
