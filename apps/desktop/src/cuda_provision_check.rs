//! Shared download, verification, marker and wheel-extraction machinery for the
//! Windows/Linux GPU-runtime provisioners. Promotion and platform layouts stay in
//! their callers. Filesystem predicates and offline fixtures run on every host.
//!
// Production Mac startup does not provision a CUDA runtime. Shared network/archive
// code is also enabled in tests so both selection policies have native offline coverage.
#![cfg_attr(not(any(windows, target_os = "linux")), allow(dead_code))]

use std::fs;
use std::path::{Path, PathBuf};
#[cfg(any(windows, target_os = "linux", test))]
use {
    futures_util::StreamExt,
    sha2::{Digest, Sha256},
    std::io::Write,
};

/// True when `dir` holds a `*.dll` matching `needle`: an exact filename check when
/// `needle` ends in `.dll`, else a case-insensitive prefix match. The prefix form is
/// version-agnostic, so a CUDA point-release (`cudart64_12` → `cudart64_13`) still
/// resolves — the version-agnostic match the sc-5560 bundling resolver used.
#[cfg(any(windows, test))]
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
/// this ONLY after the wheel downloaded, sha256-verified, and the platform verified
/// its full expected library set — so the marker is a completion witness that a
/// partial or truncated extraction cannot forge.
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
#[cfg(any(windows, test))]
pub(crate) fn component_provisioned(
    root: &Path,
    dest: &Path,
    slug: &str,
    version: &str,
    sentinels: &[&str],
) -> bool {
    component_provisioned_with(root, dest, slug, version, sentinels, dir_has_dll)
}

/// Shared-object sentinels accept the exact SONAME or a dot-versioned suffix.
/// Unlike DLL prefixes, these comparisons are case-sensitive and require a file.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn dir_has_shared_object(dir: &Path, basename: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return false;
        };
        entry.path().is_file() && (name == basename || name.starts_with(&format!("{basename}.")))
    })
}

pub(crate) fn component_marker_current(root: &Path, slug: &str, version: &str) -> bool {
    fs::read_to_string(component_marker(root, slug))
        .map(|marker| marker.trim() == version)
        .unwrap_or(false)
}

pub(crate) fn component_provisioned_with(
    root: &Path,
    dest: &Path,
    slug: &str,
    version: &str,
    sentinels: &[&str],
    present: fn(&Path, &str) -> bool,
) -> bool {
    component_marker_current(root, slug, version)
        && sentinels.iter().all(|name| present(dest, name))
}

/// Callers additionally check their platform's complete sentinel layout.
pub(crate) fn all_markers_current<'a>(
    root: &Path,
    version: &str,
    slugs: impl IntoIterator<Item = &'a str>,
) -> bool {
    slugs
        .into_iter()
        .all(|slug| component_marker_current(root, slug, version))
}

pub(crate) fn top_marker_current(root: &Path, version: &str) -> bool {
    fs::read_to_string(root.join(".redist-marker"))
        .map(|marker| marker.trim() == version)
        .unwrap_or(false)
}

pub(crate) fn write_marker(root: &Path, version: &str) -> Result<(), String> {
    fs::write(root.join(".redist-marker"), version)
        .map_err(|error| format!("write marker: {error}"))
}

/// Invalidate witnesses before the platform starts destructive replacement.
pub(crate) fn invalidate_markers<'a>(root: &Path, slugs: impl IntoIterator<Item = &'a str>) {
    let _ = fs::remove_file(root.join(".redist-marker"));
    for slug in slugs {
        let _ = fs::remove_file(component_marker(root, slug));
    }
}

/// Connect and per-read deadlines, with no total deadline on multi-GB transfers.
#[cfg(any(windows, target_os = "linux", test))]
pub(crate) fn build_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(sceneworks_core::network_policy::HTTP_CONNECT_TIMEOUT)
        .read_timeout(sceneworks_core::network_policy::DOWNLOAD_READ_TIMEOUT)
        .build()
        .map_err(|error| format!("http client: {error}"))
}

/// Stream to disk and hash each chunk before callers can extract or mark completion.
/// Failed/partial downloads remain untrusted temporary files, as on both platforms.
#[cfg(any(windows, target_os = "linux", test))]
pub(crate) async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    label: &str,
    expected: &str,
    archive: &Path,
) -> Result<(), String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("download {label}: {error}"))?
        .error_for_status()
        .map_err(|error| format!("download {label}: {error}"))?;
    let mut file = fs::File::create(archive).map_err(|error| format!("create wheel: {error}"))?;
    let mut hasher = Sha256::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("download {label}: {error}"))?;
        hasher.update(&chunk);
        file.write_all(&chunk)
            .map_err(|error| format!("write {label}: {error}"))?;
    }
    file.flush()
        .map_err(|error| format!("write {label}: {error}"))?;
    drop(file);
    let actual = sceneworks_core::hashing::hex(&hasher.finalize());
    if actual != expected {
        return Err(format!(
            "{label}: sha256 mismatch (expected {expected}, got {actual})"
        ));
    }
    Ok(())
}

/// Preserve each platform's filename semantics while sharing archive confinement,
/// basename flattening and bounded entry-to-file copying.
#[cfg(any(windows, target_os = "linux", test))]
#[derive(Clone, Copy)]
pub(crate) enum LibraryFilter<'a> {
    #[cfg(any(windows, test))]
    Dlls(Option<&'a [&'a str]>),
    #[cfg(any(target_os = "linux", test))]
    SharedObjects(Option<&'a [&'a str]>),
}

#[cfg(any(windows, target_os = "linux", test))]
impl LibraryFilter<'_> {
    fn accepts(self, base: &str) -> bool {
        match self {
            #[cfg(any(windows, test))]
            Self::Dlls(names) => {
                base.to_ascii_lowercase().ends_with(".dll")
                    && names.is_none_or(|names| names.contains(&base))
            }
            #[cfg(any(target_os = "linux", test))]
            Self::SharedObjects(prefixes) => {
                base.contains(".so")
                    && prefixes.is_none_or(|prefixes| {
                        prefixes.iter().any(|prefix| {
                            base == *prefix || base.starts_with(&format!("{prefix}."))
                        })
                    })
            }
        }
    }
}

#[cfg(any(windows, target_os = "linux", test))]
pub(crate) fn extract_libraries(
    wheel: &Path,
    dest: &Path,
    filter: LibraryFilter<'_>,
) -> Result<usize, String> {
    fs::create_dir_all(dest).map_err(|error| format!("create {}: {error}", dest.display()))?;
    let file = fs::File::open(wheel).map_err(|error| format!("open wheel: {error}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| format!("open zip: {error}"))?;
    let mut written = 0;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("read zip entry: {error}"))?;
        let Some(path) = entry.enclosed_name() else {
            continue;
        };
        let Some(base) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !filter.accepts(base) {
            continue;
        }
        let mut output =
            fs::File::create(dest.join(base)).map_err(|error| format!("write {base}: {error}"))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|error| format!("extract {base}: {error}"))?;
        written += 1;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, unique temp dir for a test (offline), removed when the guard drops.
    ///
    /// This used to be rolled by hand off `temp_dir()` to avoid a `tempfile` dev-dep, and
    /// paid for it twice: the cleanup was a trailing line each test skipped on panic, and
    /// the pid+thread-id key repeated across runs, so a recycled PID inherited an earlier
    /// run's leftovers (sc-17707). Callers hold the guard and read `.path()`.
    fn scratch(tag: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("sw-provcheck-{tag}-"))
            .tempdir()
            .expect("create scratch")
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
        let scratch_guard = scratch("has-dll");
        let dir = scratch_guard.path();
        touch(dir, &["cudart64_13.dll", "onnxruntime.dll", "notes.txt"]);

        // Exact name (ends in .dll).
        assert!(dir_has_dll(dir, "onnxruntime.dll"));
        assert!(!dir_has_dll(dir, "onnxruntime_providers_cuda.dll"));
        // Version-agnostic prefix: a 13.x cudart still satisfies the `cudart64_` sentinel.
        assert!(dir_has_dll(dir, "cudart64_"));
        // Case-insensitive.
        assert!(dir_has_dll(dir, "CUDART64_"));
        // Absent component.
        assert!(!dir_has_dll(dir, "cudnn64_"));
        // A non-DLL file with a matching prefix does NOT count.
        assert!(!dir_has_dll(dir, "notes"));
    }

    /// `component_provisioned` discriminates a genuinely-complete component from every
    /// partial state: it is true ONLY when the current-version marker AND the sentinel
    /// DLL(s) are both present. This is the retry-skip safety net (sc-13614) — a partial
    /// extract (DLLs but no marker) or a stale/missing marker must NOT skip.
    #[test]
    fn component_provisioned_requires_marker_and_dlls() {
        const VERSION: &str = "cuda12.9-test-1";
        const SENTINELS: &[&str] = &["cublas64_", "cublasLt64_"];

        let root_guard = scratch("provisioned");
        let root = root_guard.path();
        let dest = root.join("cuda");
        fs::create_dir_all(&dest).expect("create dest");

        // Nothing yet: no marker, no DLLs.
        assert!(!component_provisioned(
            root, &dest, "cublas", VERSION, SENTINELS
        ));

        // DLLs present but no marker — the exact "partial/interrupted extract" shape a
        // mid-unzip crash leaves (files on disk, completion never recorded). Must NOT skip.
        touch(&dest, &["cublas64_12.dll", "cublasLt64_12.dll"]);
        assert!(!component_provisioned(
            root, &dest, "cublas", VERSION, SENTINELS
        ));

        // Marker written last, after a verified full extraction → now complete → skip.
        write_component_marker(root, "cublas", VERSION).expect("write marker");
        assert!(component_provisioned(
            root, &dest, "cublas", VERSION, SENTINELS
        ));

        // A sentinel DLL deleted after the marker was written → no longer complete
        // (self-heals by re-downloading rather than trusting a stale marker).
        fs::remove_file(dest.join("cublasLt64_12.dll")).expect("rm dll");
        assert!(!component_provisioned(
            root, &dest, "cublas", VERSION, SENTINELS
        ));

        // Restore the DLL but bump the version → the old-version marker no longer counts
        // (a REDIST_VERSION bump must re-provision every component).
        touch(&dest, &["cublasLt64_12.dll"]);
        assert!(component_provisioned(
            root, &dest, "cublas", VERSION, SENTINELS
        ));
        assert!(!component_provisioned(
            root,
            &dest,
            "cublas",
            "cuda12.9-test-2",
            SENTINELS
        ));
    }

    #[test]
    fn markers_pin_the_version_and_invalidate_only_the_selected_components() {
        let temp = scratch("markers");
        let root = temp.path();
        assert!(!top_marker_current(root, "v1"));
        write_marker(root, "v1\n").unwrap();
        write_component_marker(root, "cuda", "v1\n").unwrap();
        write_component_marker(root, "ort", "v1").unwrap();
        write_component_marker(root, "other", "v1").unwrap();
        assert!(top_marker_current(root, "v1"));
        assert!(!top_marker_current(root, "v2"));
        assert!(all_markers_current(root, "v1", ["cuda", "ort"]));
        assert!(!all_markers_current(root, "v2", ["cuda", "ort"]));
        invalidate_markers(root, ["cuda", "ort"]);
        assert!(!top_marker_current(root, "v1"));
        assert!(!all_markers_current(root, "v1", ["cuda", "ort"]));
        assert!(component_marker_current(root, "other", "v1"));
    }

    #[test]
    fn shared_objects_require_case_sensitive_files_and_current_component_evidence() {
        let temp = scratch("sonames");
        let root = temp.path();
        let dest = root.join("cuda/lib64");
        touch(&dest, &["libcudart.so.12.9", "libcudart.software"]);
        fs::create_dir(dest.join("libcublas.so.12")).unwrap();
        assert!(dir_has_shared_object(&dest, "libcudart.so"));
        assert!(!dir_has_shared_object(&dest, "LIBCUDART.so"));
        assert!(!dir_has_shared_object(&dest, "libcublas.so"));
        assert!(!component_provisioned_with(
            root,
            &dest,
            "runtime",
            "v1",
            &["libcudart.so"],
            dir_has_shared_object
        ));
        write_component_marker(root, "runtime", "v1").unwrap();
        assert!(component_provisioned_with(
            root,
            &dest,
            "runtime",
            "v1",
            &["libcudart.so"],
            dir_has_shared_object
        ));
        fs::remove_file(dest.join("libcudart.so.12.9")).unwrap();
        assert!(!component_provisioned_with(
            root,
            &dest,
            "runtime",
            "v1",
            &["libcudart.so"],
            dir_has_shared_object
        ));
    }

    #[test]
    fn both_wheel_policies_flatten_confined_entries_and_preserve_allowlist_rules() {
        use std::io::Write;
        let temp = scratch("archive");
        let wheel = temp.path().join("fixture.whl");
        let mut zip = zip::ZipWriter::new(fs::File::create(&wheel).unwrap());
        for name in [
            "pkg/bin/Runtime.DLL",
            "pkg/bin/other.dll",
            "pkg/lib/libort.so",
            "pkg/lib/libort.so.1",
            "pkg/lib/libother.so.2",
            "pkg/lib/libort.software",
            "../outside.dll",
            "/absolute.so",
            "pkg/notes.txt",
        ] {
            zip.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(name.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        let dlls = temp.path().join("dlls");
        assert_eq!(
            extract_libraries(&wheel, &dlls, LibraryFilter::Dlls(None)).unwrap(),
            2
        );
        assert_eq!(
            fs::read(dlls.join("Runtime.DLL")).unwrap(),
            b"pkg/bin/Runtime.DLL"
        );
        let exact = temp.path().join("exact");
        assert_eq!(
            extract_libraries(&wheel, &exact, LibraryFilter::Dlls(Some(&["runtime.dll"]))).unwrap(),
            0
        );
        assert_eq!(
            extract_libraries(&wheel, &exact, LibraryFilter::Dlls(Some(&["Runtime.DLL"]))).unwrap(),
            1
        );
        let all_so = temp.path().join("all-so");
        assert_eq!(
            extract_libraries(&wheel, &all_so, LibraryFilter::SharedObjects(None)).unwrap(),
            4
        );
        let so = temp.path().join("sonames");
        assert_eq!(
            extract_libraries(
                &wheel,
                &so,
                LibraryFilter::SharedObjects(Some(&["libort.so"]))
            )
            .unwrap(),
            2
        );
        assert_eq!(
            fs::read(so.join("libort.so.1")).unwrap(),
            b"pkg/lib/libort.so.1"
        );
        assert!(!temp.path().join("outside.dll").exists());
        assert!(!so.join("pkg").exists());
        assert!(!so.join("libort.software").exists());
        let corrupt = temp.path().join("corrupt.whl");
        fs::write(&corrupt, b"not a zip").unwrap();
        assert!(extract_libraries(&corrupt, &so, LibraryFilter::Dlls(None))
            .unwrap_err()
            .starts_with("open zip:"));
    }

    #[test]
    fn streamed_download_verifies_bytes_and_refuses_mismatch_status_and_truncation() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::{Duration, Instant};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            for response in [
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n",
                "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef",
                "HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                "HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\npartial",
            ] {
                let mut socket = loop {
                    match listener.accept() {
                        Ok((socket, _)) => break socket,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "fixture request did not arrive");
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("accept fixture request: {error}"),
                    }
                };
                socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut request = [0; 4096];
                assert!(socket.read(&mut request).unwrap() > 0);
                socket.write_all(response.as_bytes()).unwrap();
            }
        });
        let temp = scratch("download");
        let archive = temp.path().join("component.whl");
        let expected = sceneworks_core::hashing::sha256_bytes(b"abcdef");
        let results = tauri::async_runtime::block_on(async {
            let client = build_client().unwrap();
            let url = format!("http://{address}/fixture.whl");
            let mut results = Vec::new();
            for digest in [&expected[..], "wrong", &expected[..], &expected[..]] {
                results.push(download_verified(&client, &url, "fixture", digest, &archive).await);
            }
            results
        });
        server.join().unwrap();
        assert!(results[0].is_ok());
        assert_eq!(
            results[1].as_ref().unwrap_err(),
            &format!("fixture: sha256 mismatch (expected wrong, got {expected})")
        );
        assert!(results[2]
            .as_ref()
            .unwrap_err()
            .starts_with("download fixture:"));
        assert!(results[3]
            .as_ref()
            .unwrap_err()
            .starts_with("download fixture:"));
        assert!(!top_marker_current(temp.path(), "v1"));
    }
}
