// NTFS sparse-fixture discipline (sc-19053 / sc-20606, StorageFull incident 2026-08-19).
//
// Multi-gigabyte test fixtures across this crate fake model weights by `set_len`-extending a file
// to its real hosted size: the gates under test read `metadata().len()`, so the LOGICAL size is
// load-bearing and must stay byte-identical to the dense artifact. On APFS/ext4 that extension is
// free — the tail is a hole. On NTFS a `set_len` tail is FULLY ALLOCATED unless the file carries
// the sparse attribute (NTFS compression was measured NOT to help; see the sc-19053 note on
// `vram_gate::krea_test_artifact_root`), so the same fixtures cost their full logical size in real
// disk on the Windows self-hosted CUDA runners — tens of gigabytes per test process once the Wan /
// Mochi / ComfyUI fixtures stack up in parallel tests.
//
// Two shapes of remedy live here, shared by every multi-GB fixture in the crate:
//
//   * `set_sparse_len` / `set_sparse_valid_safetensor` — for files THIS crate creates: the sparse
//     attribute is set BEFORE the extension, so the tail is born a hole and there is no transient
//     full allocation at all. (`fsutil sparse setflag` is the no-`unsafe` route to
//     `FSCTL_SET_SPARSE`; the workspace forbids `unsafe`.)
//   * `sparsify_fixture_weights_tail` — the post-pass for files a foreign writer creates
//     (`gen_core_testkit`'s `File::create`/`CREATE_ALWAYS` resets attributes, so pre-flagging
//     cannot survive it): flag the finished file sparse, then explicitly deallocate the
//     never-written range past its safetensors header.
//
// Every routine is best-effort on the sparse conversion itself: on a non-NTFS temp volume or a
// missing `fsutil` the fixture still works, it just costs its full logical size on disk — say so
// on stderr rather than failing the test process.

/// Mark `path` sparse so subsequent `set_len` extension allocates nothing (Windows/NTFS).
/// Best-effort: on failure the file's later `set_len` tail fully allocates, and we say so.
#[cfg(windows)]
fn mark_fixture_sparse(path: &std::path::Path) {
    let flagged = std::process::Command::new("fsutil")
        .args(["sparse", "setflag"])
        .arg(path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !flagged {
        eprintln!(
            "sparse_fixture: could not mark {} sparse; its set_len tail will fully allocate on \
             this filesystem",
            path.display()
        );
    }
}

/// Non-Windows: `set_len` past EOF is already sparse on APFS/ext4 — nothing to do.
#[cfg(not(windows))]
fn mark_fixture_sparse(_path: &std::path::Path) {}

/// Create `path` (and its parents) as a SPARSE file of `bytes` logical length with no written
/// content. The logical size is what the fixtures' gates read; no real disk is allocated for the
/// tail on APFS/ext4 (holes by default) or NTFS (flagged sparse before the extension).
pub(crate) fn set_sparse_len(path: &std::path::Path, bytes: u64) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("sparse fixture parent");
    }
    std::fs::File::create(path).expect("sparse fixture create");
    mark_fixture_sparse(path);
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.set_len(bytes))
        .expect("sparse fixture size");
}

/// Like [`set_sparse_len`], but the file is a structurally VALID safetensors of exactly `bytes`
/// logical length: an 8-byte little-endian header length, a real single-tensor JSON header, and a
/// sparse zero tail standing in for the tensor data. For fixtures whose reader parses the header
/// rather than just summing `metadata().len()`.
pub(crate) fn set_sparse_valid_safetensor(
    path: &std::path::Path,
    bytes: u64,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut data_bytes = bytes
        .checked_sub(128)
        .ok_or_else(|| format!("{bytes} bytes is too small for a safetensors fixture"))?;
    let header = loop {
        let mut header = format!(
            r#"{{"weight":{{"dtype":"U8","shape":[{data_bytes}],"data_offsets":[0,{data_bytes}]}}}}"#
        );
        while (8 + header.len()) % 8 != 0 {
            header.push(' ');
        }
        let next_data_bytes = bytes
            .checked_sub(8 + header.len() as u64)
            .ok_or_else(|| format!("{bytes} bytes is too small for its safetensors header"))?;
        if next_data_bytes == data_bytes {
            break header;
        }
        data_bytes = next_data_bytes;
    };
    {
        use std::io::Write;
        let mut file = std::fs::File::create(path).map_err(|error| error.to_string())?;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| file.write_all(header.as_bytes()))
            .map_err(|error| error.to_string())?;
    }
    mark_fixture_sparse(path);
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.set_len(bytes))
        .map_err(|error| error.to_string())
}

/// Deallocate the never-written tail of every sizeable `.safetensors` fixture under `dir` while
/// preserving its logical length and its written safetensors header. This is the POST-pass for
/// files written by `gen_core_testkit` (whose `File::create`/`CREATE_ALWAYS` resets attributes, so
/// the flag-before-extend route above cannot be used); each file is briefly fully allocated
/// between the testkit's `set_len` and this pass. Best-effort, like the rest of this module.
#[cfg(windows)]
pub(crate) fn sparsify_fixture_weights_tail(dir: &std::path::Path) {
    use std::io::Read as _;

    // The NTFS sparse deallocation granularity; keep the header's final partial unit allocated.
    const SPARSE_UNIT: u64 = 64 * 1024;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("safetensors") {
            continue;
        }
        let Ok(mut file) = std::fs::File::open(&path) else {
            continue;
        };
        let mut header_len = [0_u8; 8];
        if file.read_exact(&mut header_len).is_err() {
            continue;
        }
        let Ok(metadata) = file.metadata() else {
            continue;
        };
        drop(file);
        // Everything past `8 + header_json_len` is the testkit's `set_len` tail: logically zeros,
        // never written, and safe to deallocate.
        let written_end = 8_u64.saturating_add(u64::from_le_bytes(header_len));
        let keep = written_end.div_ceil(SPARSE_UNIT) * SPARSE_UNIT;
        let total = metadata.len();
        if total <= keep {
            continue;
        }
        let fsutil = |args: &[&str]| {
            std::process::Command::new("fsutil")
                .args(args)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        };
        let path_text = path.to_string_lossy().into_owned();
        if !(fsutil(&["sparse", "setflag", &path_text])
            && fsutil(&[
                "sparse",
                "setrange",
                &path_text,
                &keep.to_string(),
                &(total - keep).to_string(),
            ]))
        {
            eprintln!(
                "sparsify_fixture_weights_tail: could not sparsify {}; the fixture keeps its \
                 full {total}-byte allocation on this filesystem",
                path.display()
            );
        }
    }
}

/// Non-Windows: `set_len` past EOF is already sparse on APFS/ext4 — nothing to do.
#[cfg(not(windows))]
pub(crate) fn sparsify_fixture_weights_tail(_dir: &std::path::Path) {}

/// Sum the allocated bytes `fsutil file queryallocranges` reports for `path`, or `None` when the
/// environment cannot answer (non-NTFS temp volume, missing `fsutil`, file not flagged sparse) —
/// the helpers are best-effort there by design, so the witness must not fail the test for it.
#[cfg(windows)]
fn windows_allocated_bytes(path: &std::path::Path, logical: u64) -> Option<u64> {
    let sparse_flagged = std::process::Command::new("fsutil")
        .args(["sparse", "queryflag"])
        .arg(path)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| String::from_utf8_lossy(&output.stdout).contains("is set as sparse"));
    if !sparse_flagged {
        return None;
    }
    let ranges = std::process::Command::new("fsutil")
        .args([
            "file",
            "queryallocranges",
            "offset=0",
            &format!("length={logical}"),
        ])
        .arg(path)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let mut allocated = 0_u64;
    for line in String::from_utf8_lossy(&ranges.stdout).lines() {
        let Some(length_text) = line.split("Length:").nth(1) else {
            continue;
        };
        let length_text = length_text.trim().trim_start_matches("0x");
        allocated += u64::from_str_radix(length_text, 16).ok()?;
    }
    Some(allocated)
}

/// The sc-20606 witness: both flag-before-extend helpers keep the LOGICAL size byte-exact (the
/// weight gates read it) while the multi-GB tail stays unallocated — verified against the real
/// filesystem on NTFS, where a bare `set_len` tail would have allocated in full (sc-19053).
#[test]
fn sparse_fixture_helpers_preserve_logical_size_without_allocating_the_tail() {
    use std::io::Read as _;

    // Big enough that a full allocation is unambiguous next to the 64 KiB sparse unit, small
    // enough to be harmless where sparse conversion is unavailable and the file goes dense.
    const LOGICAL: u64 = 256 * 1024 * 1024;

    let dir = tempfile::tempdir().expect("sparse fixture test dir");

    let raw = dir.path().join("nested").join("raw.safetensors");
    set_sparse_len(&raw, LOGICAL);
    assert_eq!(std::fs::metadata(&raw).expect("raw metadata").len(), LOGICAL);

    let valid = dir.path().join("valid.safetensors");
    set_sparse_valid_safetensor(&valid, LOGICAL).expect("valid safetensors fixture");
    assert_eq!(
        std::fs::metadata(&valid).expect("valid metadata").len(),
        LOGICAL
    );
    // The written prefix must be a parseable safetensors header spanning the whole file.
    let mut file = std::fs::File::open(&valid).expect("open valid fixture");
    let mut header_len = [0_u8; 8];
    file.read_exact(&mut header_len).expect("header length");
    let header_len = u64::from_le_bytes(header_len);
    let mut header = vec![0_u8; usize::try_from(header_len).expect("header fits")];
    file.read_exact(&mut header).expect("header bytes");
    let header: serde_json::Value =
        serde_json::from_slice(&header).expect("header is valid JSON");
    assert_eq!(
        header["weight"]["data_offsets"][1].as_u64(),
        Some(LOGICAL - 8 - header_len),
        "the declared tensor must span exactly the sparse tail"
    );

    // The post-pass, on a file written the way `gen_core_testkit` writes them: dense create,
    // header prefix, then a bare `set_len` (fully allocated on NTFS until the pass runs).
    let foreign_dir = dir.path().join("foreign");
    std::fs::create_dir_all(&foreign_dir).expect("foreign fixture dir");
    let foreign = foreign_dir.join("testkit.safetensors");
    {
        use std::io::Write as _;
        let header = br#"{"weight":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#;
        let mut file = std::fs::File::create(&foreign).expect("foreign fixture");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| file.write_all(header))
            .and_then(|()| file.set_len(LOGICAL))
            .expect("foreign fixture bytes");
    }
    sparsify_fixture_weights_tail(&foreign_dir);
    assert_eq!(
        std::fs::metadata(&foreign).expect("foreign metadata").len(),
        LOGICAL,
        "the post-pass must never change the logical size the weight gates read"
    );

    #[cfg(windows)]
    for path in [&raw, &valid, &foreign] {
        match windows_allocated_bytes(path, LOGICAL) {
            Some(allocated) => assert!(
                allocated <= 1024 * 1024,
                "{} allocates {allocated} bytes of its {LOGICAL}-byte logical size; the set_len \
                 tail is supposed to be a sparse hole",
                path.display()
            ),
            None => eprintln!(
                "sparse_fixture test: {} could not be flagged sparse here (non-NTFS temp volume \
                 or no fsutil); skipping the allocation witness",
                path.display()
            ),
        }
    }
}
