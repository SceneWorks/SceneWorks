//! Windows disk discipline for this crate's multi-gigabyte weight fixtures.
//!
//! ## Why this module exists (StorageFull incident, 2026-08-20)
//!
//! The memory gates under test read `metadata.len()` off real files, so their fixtures have to
//! carry REAL multi-gigabyte lengths — a Mochi preflight root is ~20 GB, the Wan A14B gate fixture
//! is 40 GiB, and the Krea encoder tier set is ~22 GB. Every one of them is created by extending an
//! empty file with `set_len` and never writing a byte past the header, which on APFS and ext4 costs
//! nothing: the tail is a hole.
//!
//! **On NTFS it is not.** A `set_len`-extended file is FULLY ALLOCATED unless the file carries
//! `FILE_ATTRIBUTE_SPARSE_FILE`, so the same fixtures that are free on the Macs cost their full
//! logical size on the Windows box — hundreds of gigabytes across a parallel `cargo test` run.
//! That filled `C:` twice and broke a candle-worker CI run with `StorageFull (os error 112)`.
//! NTFS compression was measured NOT to help: a compressed file's `set_len` tail still allocates in
//! full.
//!
//! Two entry points carry that discipline, matching the two ways a fixture gets its length:
//!
//! * [`create_sparse_weights`] — this crate creates the file itself. The sparse flag is set on the
//!   still-empty file BEFORE the `set_len`, so the tail is never allocated even transiently.
//! * [`sparsify_written_safetensors`] — `gen_core_testkit` creates the file (writing a real
//!   safetensors header, then a multi-gigabyte `set_len`). We cannot pre-flag a file someone else
//!   is about to `File::create`, because `CREATE_ALWAYS` resets attributes, so this is a POST-pass
//!   that flags the finished file and deallocates the never-written tail past its header.
//!
//! Both are best-effort by design: `fsutil` is the no-`unsafe` route to `FSCTL_SET_SPARSE` (the
//! workspace forbids `unsafe`), and if it is missing or the temp volume is not NTFS the fixture is
//! still correct — it just costs its full size. A test must never fail over disk bookkeeping, so a
//! failure prints and continues.
//!
//! Neither call changes what a test observes: the logical length `metadata.len()` reports is
//! exactly the length that was requested, every byte actually written is preserved, and the tail
//! reads back as the zeros it already was.
//!
//! [`set_sparse_valid_safetensor`] is a third fixture builder that lives here because it shares the
//! `set_len`-tail shape, but it does NOT carry the NTFS discipline: it writes a real safetensors
//! header between the `File::create` and the `set_len`, so it can neither pre-flag the way
//! [`create_sparse_weights`] does nor be reached by the post-pass above. Its callers today are all
//! macOS-only (`mlx_fit_gate`'s source-bound audit), where a `set_len` tail is a hole for free. A
//! Windows caller would pay the fixture's full logical size, so give it the pre-flag treatment
//! before adding one.

// The fixture sites that call these live behind `backend-candle` / `target_os` gates, so which
// helper is live depends on the target and feature set — same reason as `test_env`.
#![allow(dead_code)]

use std::path::Path;
use std::time::{Duration, SystemTime};

/// Fixtures below this size are left alone: a `fsutil` round trip costs more than the few
/// kilobytes it would save, and most fixtures in this crate are deliberately tiny.
#[cfg(windows)]
const SPARSE_THRESHOLD_BYTES: u64 = 64 * 1024 * 1024;

/// NTFS deallocates in 64 KiB units. The post-pass keeps the header's final partial unit.
#[cfg(windows)]
const SPARSE_UNIT_BYTES: u64 = 64 * 1024;

/// A leftover fixture directory is only swept once it is old enough that no test process still
/// running could own it. A `cargo test` run of this crate is minutes; a day is three orders of
/// magnitude of slack.
const STALE_FIXTURE_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Create `path` as a `bytes`-long, all-zero weights fixture without paying `bytes` of disk.
///
/// Replaces the bare `File::create(path).set_len(bytes)` idiom at every fixture site whose length
/// is measured in gigabytes. Parent directories are created. The resulting file has exactly
/// `bytes` logical length on every platform.
pub(crate) fn create_sparse_weights(path: &Path, bytes: u64) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("weights fixture parent directory");
    }
    let file = std::fs::File::create(path).expect("weights fixture file");
    #[cfg(windows)]
    {
        // Flag the file while it is still empty: a sparse file's `set_len` extends the valid data
        // length without allocating clusters, so unlike the post-pass below this never has a
        // fully-allocated moment at all.
        drop(file);
        if bytes >= SPARSE_THRESHOLD_BYTES {
            set_sparse_flag(path, bytes);
        }
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("weights fixture reopen")
            .set_len(bytes)
            .expect("weights fixture size");
    }
    #[cfg(not(windows))]
    {
        // APFS/ext4: extending past EOF is already a hole.
        file.set_len(bytes).expect("weights fixture size");
    }
}

/// Deallocate the never-written tail of every sizeable `.safetensors` file directly under `dir`.
///
/// For fixtures written by `gen_core_testkit`, whose final `set_len` is the encoder's dense logical
/// size — which must stay, because the weight-bytes gates read it. The written prefix (the 8-byte
/// header length plus the header JSON, rounded up to the NTFS deallocation unit) is preserved
/// byte for byte; only the zero tail past it is released.
///
/// No-op on non-Windows, where that tail is already a hole.
#[cfg(windows)]
pub(crate) fn sparsify_written_safetensors(dir: &Path) {
    use std::io::Read as _;

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
        let total = metadata.len();
        // Everything past `8 + header_json_len` is the testkit's `set_len` tail: logically zeros,
        // never written, and safe to release.
        let written_end = 8_u64.saturating_add(u64::from_le_bytes(header_len));
        let keep = written_end
            .div_ceil(SPARSE_UNIT_BYTES)
            .saturating_mul(SPARSE_UNIT_BYTES);
        if total < SPARSE_THRESHOLD_BYTES || total <= keep {
            continue;
        }
        if set_sparse_flag(&path, total) {
            deallocate_range(&path, keep, total - keep);
        }
    }
}

/// Non-Windows: the testkit's `set_len` tail is already a hole.
#[cfg(not(windows))]
pub(crate) fn sparsify_written_safetensors(_dir: &Path) {}

/// Create `path` as a structurally valid, `bytes`-long safetensors file: one `U8` tensor whose
/// declared data covers every byte past the header.
///
/// For a fixture a gate PARSES rather than merely stats. The header is space-padded to the
/// safetensors 8-byte alignment and then re-fitted until `8 + header.len() + data_offsets[1]` is
/// exactly `bytes`, so the declared tensor and the file's logical length agree. Only the header is
/// written; the tensor data is the `set_len` tail.
///
/// Unlike [`create_sparse_weights`], this does NOT mark the file sparse — see the module docs for
/// why, and read them before calling this from a lane that runs on NTFS.
pub(crate) fn set_sparse_valid_safetensor(path: &Path, bytes: u64) -> Result<(), String> {
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
    use std::io::Write;
    let mut file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .and_then(|()| file.write_all(header.as_bytes()))
        .and_then(|()| file.set_len(bytes))
        .map_err(|error| error.to_string())
}

/// `fsutil sparse setflag`, reporting rather than failing. Returns whether the flag is now set.
#[cfg(windows)]
fn set_sparse_flag(path: &Path, bytes: u64) -> bool {
    if fsutil(&["sparse", "setflag", &path.to_string_lossy()]) {
        return true;
    }
    eprintln!(
        "test_fixture_disk: could not mark {} sparse; the fixture keeps its full {bytes}-byte \
         allocation on this filesystem",
        path.display()
    );
    false
}

/// `fsutil sparse setrange`, reporting rather than failing.
#[cfg(windows)]
fn deallocate_range(path: &Path, offset: u64, length: u64) {
    if !fsutil(&[
        "sparse",
        "setrange",
        &path.to_string_lossy(),
        &offset.to_string(),
        &length.to_string(),
    ]) {
        eprintln!(
            "test_fixture_disk: could not release {length} bytes at {offset} of {}",
            path.display()
        );
    }
}

#[cfg(windows)]
fn fsutil(args: &[&str]) -> bool {
    std::process::Command::new("fsutil")
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// The `tempfile` prefix a fixture family must use so its leftovers are recognizable AND
/// attributable: `<family><pid>-`, e.g. `sceneworks-krea-memfix-31672-`.
///
/// The PID is not what makes the name unique — `tempfile` already does that per call — it is what
/// lets [`sweep_stale_fixtures`] tell a directory this process owns from one it does not.
pub(crate) fn process_keyed_prefix(family_prefix: &str) -> String {
    format!("{family_prefix}{}-", std::process::id())
}

/// Remove prior runs' leftovers of one fixture family from the shared temp root.
///
/// Only for a fixture whose guard genuinely cannot run: a `TempDir` parked in a `static` is never
/// dropped, so that fixture leaks one directory per test process BY CONSTRUCTION and something has
/// to collect them. A fixture with a live guard must not call this — it does not need it.
///
/// ## Why this cannot delete a directory another test process is using
///
/// `%TEMP%` on the build box is shared by many concurrent `cargo test` processes across several
/// worktrees, so a sweep that guessed wrong would delete a running test's fixture out from under
/// it. Three conditions must ALL hold before anything is removed, but they are NOT
/// interchangeable: each excludes a different population, and only ONE of them stands between this
/// sweep and a concurrent run of this same fixture family.
///
/// * The **age cutoff** is that one. A concurrent `cargo test` of this crate builds its fixture
///   under the SAME family prefix and a DIFFERENT pid, so it satisfies both other conditions —
///   being minutes old is the only thing that keeps it. This was observed live:
///   `sceneworks-krea-memfix-foMx94` from another agent's run appeared in `%TEMP%` while this
///   change was under review.
/// * **Prefix scoping** excludes unrelated tools' temp state (another crate's fixtures,
///   `tempfile`'s default `.tmp*` names, an editor's scratch dir). It does NOT exclude concurrent
///   runs of this family, which share the prefix by construction.
/// * The **PID tag** excludes this process's own directory, and nothing else. A concurrent run
///   carries a different pid and is not excluded by it.
///
/// So [`STALE_FIXTURE_AGE`] carries the concurrency guarantee alone. Shortening it does not leave
/// two other guards in place — it removes the only protection a concurrent run has. See
/// [`is_stale_fixture`].
pub(crate) fn sweep_stale_fixtures(family_prefix: &str) {
    // The one deliberate `temp_dir()` call in this crate's fixtures (allow-listed in
    // `tests/temp_fixture_guard.rs`): the directories being collected are, by definition, ones no
    // guard owns any more, so there is no `TempDir` to ask where they are.
    let root = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    let now = SystemTime::now();
    let own_pid = std::process::id();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let is_dir = entry.file_type().is_ok_and(|kind| kind.is_dir());
        let modified = entry.metadata().and_then(|data| data.modified()).ok();
        if is_stale_fixture(name, family_prefix, is_dir, modified, now, own_pid) {
            // A live handle or a racing sweep makes this fail; that is someone else's business.
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// The whole sweep decision, as a pure function so it can be asserted directly.
///
/// Each condition excludes a DIFFERENT population — they do not back each other up; see the
/// decomposition on [`sweep_stale_fixtures`] before changing any of them:
///
/// * **prefix scoping** — only directories this fixture family created are candidates. Unrelated
///   temp state (another crate's fixtures, `tempfile`'s default `.tmp*` names, an editor's scratch
///   dir) never matches, so a mis-typed family can only make the sweep do nothing. Concurrent runs
///   of THIS family share the prefix and are not excluded here.
/// * **age cutoff** — a candidate must be at least [`STALE_FIXTURE_AGE`] old. A directory a
///   concurrent process created is minutes old at most, so it can never qualify. This is the only
///   condition that protects a concurrent run of this family, and it holds even for a process
///   whose PID this one cannot see.
/// * **PID attribution** — a directory tagged with our own PID is ours and is excluded outright,
///   so re-entering initialization can never delete the fixture it is building. That is its whole
///   job: a concurrent run has a different PID and is left to the age cutoff. A directory with no
///   PID tag, or an unparseable one, is treated as another process's and likewise stays subject to
///   the age cutoff.
///
/// An unreadable or future-dated mtime fails CLOSED (keep the directory): the cost of keeping a
/// dead fixture one more day is disk, the cost of deleting a live one is a broken test run.
fn is_stale_fixture(
    name: &str,
    family_prefix: &str,
    is_dir: bool,
    modified: Option<SystemTime>,
    now: SystemTime,
    own_pid: u32,
) -> bool {
    if !is_dir {
        return false;
    }
    let Some(suffix) = name.strip_prefix(family_prefix) else {
        return false;
    };
    let owner = suffix
        .split('-')
        .next()
        .and_then(|segment| segment.parse::<u32>().ok());
    if owner == Some(own_pid) {
        return false;
    }
    let Some(modified) = modified else {
        return false;
    };
    now.duration_since(modified)
        .is_ok_and(|age| age >= STALE_FIXTURE_AGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAMILY: &str = "sceneworks-krea-memfix-";

    fn ago(seconds: u64) -> (Option<SystemTime>, SystemTime) {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(2_000_000_000);
        (Some(now - Duration::from_secs(seconds)), now)
    }

    /// The only shape that is swept: our family, a directory, another process, old enough.
    #[test]
    fn a_foreign_day_old_fixture_directory_is_swept() {
        let (modified, now) = ago(25 * 60 * 60);
        assert!(is_stale_fixture(
            &format!("{FAMILY}4321-Ab12Cd"),
            FAMILY,
            true,
            modified,
            now,
            1234
        ));
    }

    /// The age cutoff is what protects the ~50 test processes that share this box's `%TEMP%`: a
    /// fixture a concurrent run created moments ago must survive. Shortening or inverting
    /// [`STALE_FIXTURE_AGE`] turns this red.
    #[test]
    fn a_fresh_fixture_from_a_concurrent_process_is_never_swept() {
        for age_seconds in [0, 60, 60 * 60, 23 * 60 * 60, 24 * 60 * 60 - 1] {
            let (modified, now) = ago(age_seconds);
            assert!(
                !is_stale_fixture(
                    &format!("{FAMILY}4321-Ab12Cd"),
                    FAMILY,
                    true,
                    modified,
                    now,
                    1234
                ),
                "a {age_seconds}s-old fixture belongs to a run that may still be using it"
            );
        }
    }

    /// Prefix scoping. Dropping it — sweeping every old temp directory, or matching on a looser
    /// substring — would collect other crates' and other tools' state. Includes `tempfile`'s
    /// default `.tmp*` shape, which is exactly what an un-prefixed fixture would leave behind.
    #[test]
    fn only_this_fixture_familys_directories_are_candidates() {
        let (modified, now) = ago(400 * 60 * 60);
        for name in [
            ".tmp4WMd6o",
            "sceneworks-krea-memfi-4321-Ab12Cd",
            "krea-memfix-4321-Ab12Cd",
            "some-other-crate-4321",
            "cargo-installXYZ",
            "",
        ] {
            assert!(
                !is_stale_fixture(name, FAMILY, true, modified, now, 1234),
                "{name} is not this fixture family's directory"
            );
        }
    }

    /// PID attribution: our own directory is excluded whatever its age, so initialization can
    /// never delete the fixture it is in the middle of building.
    #[test]
    fn this_processs_own_fixture_is_excluded_at_any_age() {
        let (modified, now) = ago(400 * 60 * 60);
        assert!(!is_stale_fixture(
            &format!("{FAMILY}1234-Ab12Cd"),
            FAMILY,
            true,
            modified,
            now,
            1234
        ));
        assert!(process_keyed_prefix(FAMILY).starts_with(FAMILY));
        assert!(process_keyed_prefix(FAMILY).ends_with(&format!("{}-", std::process::id())));
    }

    /// Fail closed. A directory whose mtime cannot be read, or is dated in the future by a skewed
    /// clock, is kept — and a plain FILE is never removed by a directory sweep.
    #[test]
    fn unreadable_future_dated_and_non_directory_entries_are_kept() {
        let (modified, now) = ago(400 * 60 * 60);
        let name = format!("{FAMILY}4321-Ab12Cd");
        assert!(!is_stale_fixture(&name, FAMILY, false, modified, now, 1234));
        assert!(!is_stale_fixture(&name, FAMILY, true, None, now, 1234));
        let future = Some(now + Duration::from_secs(60));
        assert!(!is_stale_fixture(&name, FAMILY, true, future, now, 1234));
    }

    /// A legacy directory from before the PID tag existed has no parseable owner, so it is another
    /// process's as far as this sweep is concerned and the age cutoff decides.
    #[test]
    fn an_untagged_legacy_directory_is_still_swept_when_old() {
        let (modified, now) = ago(400 * 60 * 60);
        assert!(is_stale_fixture(
            &format!("{FAMILY}LUiX3O"),
            FAMILY,
            true,
            modified,
            now,
            1234
        ));
    }

    /// The disk claim itself, on the filesystem that has the problem: a gigabyte-scale fixture
    /// reports its full logical length and is a real sparse file. Without the `fsutil` pass this
    /// file costs 8 GiB of `C:`.
    #[cfg(windows)]
    #[test]
    fn a_multi_gigabyte_fixture_is_sparse_and_keeps_its_logical_length() {
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_SPARSE_FILE: u32 = 0x0000_0200;
        const BYTES: u64 = 8 * 1024 * 1024 * 1024;

        let guard = tempfile::Builder::new()
            .prefix("sceneworks-sparse-fixture-selfcheck-")
            .tempdir()
            .expect("temp dir");
        let path = guard.path().join("tier").join("model.safetensors");
        create_sparse_weights(&path, BYTES);

        let metadata = std::fs::metadata(&path).expect("fixture metadata");
        assert_eq!(
            metadata.len(),
            BYTES,
            "the gates read this length; sparseness must not change it"
        );
        assert_ne!(
            metadata.file_attributes() & FILE_ATTRIBUTE_SPARSE_FILE,
            0,
            "a multi-gigabyte fixture must be sparse on NTFS or it allocates in full"
        );
    }

    /// The post-pass over a file this crate did not create: the written safetensors header
    /// survives verbatim, the logical length survives, and the tail is released.
    #[cfg(windows)]
    #[test]
    fn the_post_pass_releases_a_written_fixtures_tail_without_touching_its_header() {
        use std::io::{Read as _, Write as _};
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_SPARSE_FILE: u32 = 0x0000_0200;
        const BYTES: u64 = 256 * 1024 * 1024;

        let guard = tempfile::Builder::new()
            .prefix("sceneworks-sparse-fixture-selfcheck-")
            .tempdir()
            .expect("temp dir");
        let dir = guard.path().join("text_encoder");
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let path = dir.join("model.safetensors");

        // The exact shape `gen_core_testkit` leaves behind: header length, header JSON, then a
        // `set_len` to the dense logical size with nothing written past the header.
        let header = br#"{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut file = std::fs::File::create(&path).expect("fixture file");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("header length");
        file.write_all(header).expect("header");
        file.set_len(BYTES).expect("dense logical size");
        drop(file);

        sparsify_written_safetensors(&dir);

        let metadata = std::fs::metadata(&path).expect("fixture metadata");
        assert_eq!(metadata.len(), BYTES, "logical length must be untouched");
        assert_ne!(
            metadata.file_attributes() & FILE_ATTRIBUTE_SPARSE_FILE,
            0,
            "the testkit's tail must be released or it allocates in full"
        );
        let mut written = vec![0_u8; 8 + header.len()];
        std::fs::File::open(&path)
            .expect("fixture reopen")
            .read_exact(&mut written)
            .expect("fixture header readback");
        let mut expected = (header.len() as u64).to_le_bytes().to_vec();
        expected.extend_from_slice(header);
        assert_eq!(
            written, expected,
            "every byte the testkit actually wrote must survive the sparse pass"
        );
    }
}
