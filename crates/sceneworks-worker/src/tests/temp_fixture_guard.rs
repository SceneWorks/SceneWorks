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
        "src/media_jobs.rs",
        1,
        "production: the person-track job's own frame work dir, removed by the job",
    ),
    (
        "src/video_jobs/seedvr2.rs",
        2,
        "production: SeedVR2 src/out frame staging for a real upscale run",
    ),
    (
        "src/video_jobs/vace.rs",
        2,
        "production: VACE replace-frames + work dir for a real render",
    ),
    (
        "src/video_jobs/minimax_h3.rs",
        1,
        "production: the Ref2VA reference-clip work dir (extracted frames + the clip's own \
         soundtrack) for a real render, removed by the arm on EVERY exit including the refusals",
    ),
    (
        "src/video_jobs/reference_audio.rs",
        1,
        "production: the per-reference work dir the standalone audio references are normalized \
         into (sc-18650), removed by the resolver on EVERY exit including the refusals — the same \
         shape as the reference-clip work dir above",
    ),
    (
        "src/snapshot_install.rs",
        1,
        "deliberate: `whisper_smoke_hub_cache`'s default root is a revision-keyed cache that is \
         SUPPOSED to outlive the run — a warm rerun re-verifies the ~279 MB pin instead of \
         refetching it, and a guard would force a cold download every time. Overridable via \
         SCENEWORKS_WHISPER_SMOKE_CACHE; a pin bump changes the directory.",
    ),
    (
        "src/audio_jobs.rs",
        2,
        "deliberate: SCENEWORKS_DOD_OUT fallback for a DoD .wav a human listens to afterwards — \
         a guard would delete it before it could be played",
    ),
    (
        "src/catalog_image_fetch.rs",
        1,
        "deliberate: the temp root is the SYMLINK TARGET in a path-escape test, never written to",
    ),
    (
        "src/tests/hf_and_family.rs",
        1,
        "deliberate: the assertion is about the temp ROOT itself — that a crafted download dir \
         cannot traverse above it. Nothing is created.",
    ),
    (
        "src/video_jobs/tests.rs",
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
/// Mutation-checked: re-introducing a single `std::env::temp_dir()` fixture anywhere under the crate
/// turns this RED, and so does adding one to a file already on the allow-list.
#[test]
fn no_test_fixture_builds_a_path_under_the_shared_temp_root() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found: Vec<(String, usize)> = Vec::new();
    for sub in ROOTS {
        let dir = crate_root.join(sub);
        if dir.is_dir() {
            collect_temp_dir_calls(&dir, crate_root, &mut found);
        }
    }
    found.sort();

    let allowed: std::collections::HashMap<&str, (usize, &str)> = ALLOWED
        .iter()
        .map(|(file, count, why)| (*file, (*count, *why)))
        .collect();

    let mut problems = Vec::new();
    for (file, count) in &found {
        match allowed.get(file.as_str()) {
            None => problems.push(format!(
                "{file}: {count} `temp_dir()` call(s) with no entry in ALLOWED. A test \
                 fixture must use `tempfile::tempdir()` (or `Builder::new().prefix(..)`) and hold \
                 the guard, so cleanup survives a panic; production work dirs go in ALLOWED with a \
                 reason."
            )),
            Some((expected, why)) if expected != count => problems.push(format!(
                "{file}: {count} `temp_dir()` call(s), ALLOWED says {expected} ({why})"
            )),
            Some(_) => {}
        }
    }
    for (file, (expected, why)) in &allowed {
        if !found.iter().any(|(f, _)| f == file) {
            problems.push(format!(
                "{file}: ALLOWED expects {expected} `temp_dir()` call(s) ({why}) but the \
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

/// This file, relative to the crate root. Skipped by the scan below: it names the forbidden call
/// in its own prose and matcher, so counting itself would make the guard permanently red.
const SELF: &str = "src/tests/temp_fixture_guard.rs";

/// Directories under the crate root worth scanning: everything that can hold a `#[test]`.
/// `target/` is excluded because it holds generated code, not sources.
const ROOTS: &[&str] = &["src", "tests", "benches", "examples"];

/// Count temp-root call sites per `.rs` file under `dir`, keyed by the path relative to `root`.
///
/// Deliberately matches the bare `temp_dir(` token rather than the fully-qualified
/// `std::env::temp_dir()`: `use std::env; env::temp_dir()` and `use std::env::temp_dir;
/// temp_dir()` are the same defect, and pinning one spelling would let the next author defeat this
/// guard by writing the idiomatic one. Line comments are stripped first, so the many doc comments
/// that *mention* `temp_dir()` in prose do not register — which is also why the allow-list counts
/// below are call sites, not raw text hits.
fn collect_temp_dir_calls(
    dir: &std::path::Path,
    root: &std::path::Path,
    out: &mut Vec<(String, usize)>,
) {
    let entries = std::fs::read_dir(dir).expect("crate tree is readable");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_temp_dir_calls(&path, root, out);
            continue;
        }
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .expect("walked path is under the crate root")
            .to_string_lossy()
            .replace('\\', "/");
        if relative == SELF {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("source file is utf-8");
        let count = text
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .map(|code| code.matches("temp_dir(").count())
            .sum();
        if count > 0 {
            out.push((relative, count));
        }
    }
}
