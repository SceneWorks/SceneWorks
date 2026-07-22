//! Pure, cross-platform filesystem predicates for the first-run GPU-runtime
//! provisioner (`cuda_provision`).
//!
//! `cuda_provision` is `#[cfg(target_os = "windows")]`, so any test living inside it
//! only ever runs on the Windows CI lane. These predicates — the "is this component
//! already extracted?" checks that gate the retry-skip guard (sc-13614) and the
//! pre-staged-redist completeness check — are factored out here, behind NO `cfg`, so
//! they compile and unit-test on any host (a plain `cargo test -p sceneworks-desktop`
//! on macOS/Linux exercises them). Off Windows the only non-test caller
//! (`cuda_provision`) is compiled out, so the `dead_code` lint is relaxed there; on
//! Windows every item has a real caller.
#![cfg_attr(not(target_os = "windows"), allow(dead_code))]

use std::fs;
use std::path::{Path, PathBuf};

/// True when `dir` holds a `*.dll` matching `needle`: an exact filename check when
/// `needle` ends in `.dll`, else a case-insensitive prefix match. The prefix form is
/// version-agnostic, so a CUDA point-release (`cudart64_12` → `cudart64_13`) still
/// resolves — the version-agnostic match the sc-5560 bundling resolver used.
pub(crate) fn dir_has_dll(dir: &Path, needle: &str) -> bool {
    if needle.ends_with(".dll") {
        return dir.join(needle).is_file();
    }
    let needle = needle.to_ascii_lowercase();
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .file_name()
            .to_str()
            .map(|name| {
                let name = name.to_ascii_lowercase();
                name.starts_with(&needle) && name.ends_with(".dll")
            })
            .unwrap_or(false)
    })
}

/// Path of a component's per-run completion marker under the provisioned root
/// (`<root>\.component-<slug>.ok`). Distinct from the top-level `.redist-marker` (which
/// records that the *whole* set succeeded): one of these is written per component, so a
/// run that fails partway still records which components already landed.
fn component_marker(root: &Path, slug: &str) -> PathBuf {
    root.join(format!(".component-{slug}.ok"))
}

/// Record that a component fully provisioned into place for `version`. The caller writes
/// this ONLY after the wheel downloaded, sha256-verified, and extracted with its full
/// expected DLL count — so the marker is a trustworthy completion witness that a
/// mid-unzip crash (a partial or truncated DLL set) cannot forge.
pub(crate) fn write_component_marker(root: &Path, slug: &str, version: &str) -> Result<(), String> {
    fs::write(component_marker(root, slug), version)
        .map_err(|error| format!("write component marker {slug}: {error}"))
}

/// True when component `slug` is already fully provisioned into `dest` for `version` and
/// so can be skipped on a retry (sc-13614) instead of re-downloading it.
///
/// Requires BOTH:
///   1. a completion marker for THIS `version` (written last, only after a verified full
///      extraction) — a partial or truncated extract never leaves one, so a corrupt
///      component is re-fetched rather than mistaken for complete; and
///   2. the component's sentinel DLL(s) still present on disk — so a component whose
///      files were deleted after the marker was written self-heals by re-downloading.
///
/// The marker is the primary guard against a partial/corrupt extract; the on-disk DLL
/// check guards against later deletion. Both must hold for a skip.
pub(crate) fn component_provisioned(
    root: &Path,
    dest: &Path,
    slug: &str,
    version: &str,
    sentinels: &[&str],
) -> bool {
    let marker_current = fs::read_to_string(component_marker(root, slug))
        .map(|marker| marker.trim() == version)
        .unwrap_or(false);
    marker_current && sentinels.iter().all(|dll| dir_has_dll(dest, dll))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, unique temp dir for a test (offline; cleaned by the caller). Rolled by
    /// hand (as the sibling `cuda_provision` tests do) so no `tempfile` dev-dep is added.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sw-provcheck-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch");
        dir
    }

    fn touch(dir: &Path, names: &[&str]) {
        fs::create_dir_all(dir).expect("create dir");
        for name in names {
            fs::write(dir.join(name), b"").expect("touch dll");
        }
    }

    /// `dir_has_dll` matches exact `.dll` names directly and everything else by
    /// case-insensitive prefix (so a CUDA point-release still resolves).
    #[test]
    fn dir_has_dll_exact_and_prefix() {
        let dir = scratch("has-dll");
        touch(&dir, &["cudart64_13.dll", "onnxruntime.dll", "notes.txt"]);

        // Exact name (ends in .dll).
        assert!(dir_has_dll(&dir, "onnxruntime.dll"));
        assert!(!dir_has_dll(&dir, "onnxruntime_providers_cuda.dll"));
        // Version-agnostic prefix: a 13.x cudart still satisfies the `cudart64_` sentinel.
        assert!(dir_has_dll(&dir, "cudart64_"));
        // Case-insensitive.
        assert!(dir_has_dll(&dir, "CUDART64_"));
        // Absent component.
        assert!(!dir_has_dll(&dir, "cudnn64_"));
        // A non-DLL file with a matching prefix does NOT count.
        assert!(!dir_has_dll(&dir, "notes"));

        let _ = fs::remove_dir_all(&dir);
    }

    /// `component_provisioned` discriminates a genuinely-complete component from every
    /// partial state: it is true ONLY when the current-version marker AND the sentinel
    /// DLL(s) are both present. This is the retry-skip safety net (sc-13614) — a partial
    /// extract (DLLs but no marker) or a stale/missing marker must NOT skip.
    #[test]
    fn component_provisioned_requires_marker_and_dlls() {
        const VERSION: &str = "cuda12.9-test-1";
        const SENTINELS: &[&str] = &["cublas64_", "cublasLt64_"];

        let root = scratch("provisioned");
        let dest = root.join("cuda");
        fs::create_dir_all(&dest).expect("create dest");

        // Nothing yet: no marker, no DLLs.
        assert!(!component_provisioned(
            &root, &dest, "cublas", VERSION, SENTINELS
        ));

        // DLLs present but no marker — the exact "partial/interrupted extract" shape a
        // mid-unzip crash leaves (files on disk, completion never recorded). Must NOT skip.
        touch(&dest, &["cublas64_12.dll", "cublasLt64_12.dll"]);
        assert!(!component_provisioned(
            &root, &dest, "cublas", VERSION, SENTINELS
        ));

        // Marker written last, after a verified full extraction → now complete → skip.
        write_component_marker(&root, "cublas", VERSION).expect("write marker");
        assert!(component_provisioned(
            &root, &dest, "cublas", VERSION, SENTINELS
        ));

        // A sentinel DLL deleted after the marker was written → no longer complete
        // (self-heals by re-downloading rather than trusting a stale marker).
        fs::remove_file(dest.join("cublasLt64_12.dll")).expect("rm dll");
        assert!(!component_provisioned(
            &root, &dest, "cublas", VERSION, SENTINELS
        ));

        // Restore the DLL but bump the version → the old-version marker no longer counts
        // (a REDIST_VERSION bump must re-provision every component).
        touch(&dest, &["cublasLt64_12.dll"]);
        assert!(component_provisioned(
            &root, &dest, "cublas", VERSION, SENTINELS
        ));
        assert!(!component_provisioned(
            &root,
            &dest,
            "cublas",
            "cuda12.9-test-2",
            SENTINELS
        ));

        let _ = fs::remove_dir_all(&root);
    }
}
