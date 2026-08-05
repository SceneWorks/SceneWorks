// Source-level guard that keeps test fixtures off the shared temp root (sc-17641 / sc-17707).
//
// Every fixture in this crate now hangs off a `tempfile` guard that removes its directory on
// `Drop`. A guard cannot be expressed as a runtime assertion — nothing observes a future author
// typing `std::env::temp_dir()` back into a new test — so it is asserted against the source.

/// The `std::env::temp_dir()` call sites that are allowed to remain, and why.
///
/// Two kinds live here and NOTHING else:
///   * **production** work dirs — the worker really does stage frames and masks under the OS temp
///     root at runtime; those are not fixtures and are removed by the job that made them;
///   * **deliberate test exceptions** — a temp path that is the subject of the assertion rather
///     than scratch space for it, or an artifact a human is meant to open afterwards.
///
/// A count, not just a filename, so ADDING a call to an already-listed file fails too.
const ALLOWED: &[(&str, usize, &str)] = &[
    (
        "media_jobs.rs",
        1,
        "production: the person-track job's own frame work dir, removed by the job",
    ),
    (
        "video_jobs/seedvr2.rs",
        2,
        "production: SeedVR2 src/out frame staging for a real upscale run",
    ),
    (
        "video_jobs/vace.rs",
        2,
        "production: VACE replace-frames + work dir for a real render",
    ),
    (
        "snapshot_install.rs",
        1,
        "deliberate: `whisper_smoke_hub_cache`'s default root is a revision-keyed cache that is \
         SUPPOSED to outlive the run — a warm rerun re-verifies the ~279 MB pin instead of \
         refetching it, and a guard would force a cold download every time. Overridable via \
         SCENEWORKS_WHISPER_SMOKE_CACHE; a pin bump changes the directory.",
    ),
    (
        "audio_jobs.rs",
        2,
        "deliberate: SCENEWORKS_DOD_OUT fallback for a DoD .wav a human listens to afterwards — \
         a guard would delete it before it could be played",
    ),
    (
        "catalog_image_fetch.rs",
        1,
        "deliberate: the temp root is the SYMLINK TARGET in a path-escape test, never written to",
    ),
    (
        "tests/hf_and_family.rs",
        1,
        "deliberate: the assertion is about the temp ROOT itself — that a crafted download dir \
         cannot traverse above it. Nothing is created.",
    ),
    (
        "person_replace.rs",
        1,
        "deliberate: a data-dir argument for the no-frames error path, which returns before \
         touching the filesystem",
    ),
    (
        "video_jobs/tests.rs",
        1,
        "deliberate (sc-17707, named in the story): sc6139_i2v_pad_frame0.png is a debug artifact \
         a human opens after a real-weight probe run; a guard would delete it first",
    ),
];

/// No test fixture may build a path under the shared temp root by hand.
///
/// The shape this forbids was wrong twice over (sc-17641): the cleanup was a trailing line, which
/// a panicking test — exactly the one whose leftovers matter — skips, and uniqueness came from the
/// PROCESS, so a run landing on a recycled PID inherited an unrelated earlier run's leftovers.
/// `tempfile` fixes both halves: cleanup rides on `Drop`, and the name is unique per CALL.
///
/// Mutation-checked: re-introducing a single `std::env::temp_dir()` fixture anywhere under `src/`
/// turns this RED, and so does adding one to a file already on the allow-list.
#[test]
fn no_test_fixture_builds_a_path_under_the_shared_temp_root() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found: Vec<(String, usize)> = Vec::new();
    collect_temp_dir_calls(&src, &src, &mut found);
    found.sort();

    let allowed: std::collections::HashMap<&str, (usize, &str)> = ALLOWED
        .iter()
        .map(|(file, count, why)| (*file, (*count, *why)))
        .collect();

    let mut problems = Vec::new();
    for (file, count) in &found {
        match allowed.get(file.as_str()) {
            None => problems.push(format!(
                "{file}: {count} `std::env::temp_dir()` call(s) with no entry in ALLOWED. A test \
                 fixture must use `tempfile::tempdir()` (or `Builder::new().prefix(..)`) and hold \
                 the guard, so cleanup survives a panic; production work dirs go in ALLOWED with a \
                 reason."
            )),
            Some((expected, why)) if expected != count => problems.push(format!(
                "{file}: {count} `std::env::temp_dir()` call(s), ALLOWED says {expected} ({why})"
            )),
            Some(_) => {}
        }
    }
    for (file, (expected, why)) in &allowed {
        if !found.iter().any(|(f, _)| f == file) {
            problems.push(format!(
                "{file}: ALLOWED expects {expected} `std::env::temp_dir()` call(s) ({why}) but the \
                 file has none — drop the stale entry"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "temp-fixture guard (sc-17707):\n  {}",
        problems.join("\n  ")
    );
}

/// This file, relative to the crate's `src/`. Skipped by the scan below: it names the forbidden
/// call in its own prose and matcher, so counting itself would make the guard permanently red.
const SELF: &str = "tests/temp_fixture_guard.rs";

/// Count `std::env::temp_dir()` call sites per `.rs` file under `dir`, keyed by the path relative
/// to `root`. Matches the fully-qualified call only, so the many doc comments that *mention*
/// `temp_dir()` in prose do not register.
fn collect_temp_dir_calls(
    dir: &std::path::Path,
    root: &std::path::Path,
    out: &mut Vec<(String, usize)>,
) {
    let entries = std::fs::read_dir(dir).expect("worker src tree is readable");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_temp_dir_calls(&path, root, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            if path.strip_prefix(root).is_ok_and(|p| p == std::path::Path::new(SELF)) {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source file is utf-8");
            let count = text.matches("std::env::temp_dir()").count();
            if count > 0 {
                let relative = path
                    .strip_prefix(root)
                    .expect("walked path is under the src root")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((relative, count));
            }
        }
    }
}
