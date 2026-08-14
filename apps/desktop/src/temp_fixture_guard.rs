//! Source-level guard keeping this crate's test fixtures off the shared temp root (sc-17707).
//!
//! The shell's `scratch(..)` / `scratch_dir(..)` / `runtime_root(..)` helpers used to build
//! `temp_dir()/<tag>-{pid}` by hand — explicitly to avoid a `tempfile` dev-dependency — and paid
//! for it twice: cleanup was a trailing line in each test, which a panicking test skips, and the
//! pid key repeated across runs, so a recycled PID started from an unrelated run's leftovers.
//! They now return `tempfile::TempDir` guards. Nothing observes a future author typing
//! `std::env::temp_dir()` back into a new test, so the guard is asserted against the source.
//!
//! The twin in `crates/sceneworks-worker/src/tests/temp_fixture_guard.rs` covers the worker.
//! This crate is deliberately not merged into that one: two of the four provisioner modules here
//! compile only on Windows/Linux, so this file's scan is the only check that sees them from any
//! host — reading the source needs no `cfg` to be satisfied.

/// The `std::env::temp_dir()` call sites that are allowed to remain, and why. A count, not just a
/// filename, so ADDING a call to an already-listed file fails too.
const ALLOWED: &[(&str, usize, &str)] = &[(
    "src/setup.rs",
    2,
    "production: the last-resort app-data / log dirs when no platform data dir resolves — real \
     runtime state the shell must keep, not fixtures",
)];

/// No test fixture may build a path under the shared temp root by hand.
///
/// Mutation-checked: re-introducing one `std::env::temp_dir()` fixture anywhere under the crate turns
/// this RED, and so does adding one to a file already on the allow-list.
#[test]
fn no_test_fixture_builds_a_path_under_the_shared_temp_root() {
    use std::collections::HashMap;

    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found: Vec<(String, usize)> = Vec::new();
    for sub in ROOTS {
        let dir = crate_root.join(sub);
        if dir.is_dir() {
            collect(&dir, crate_root, &mut found);
        }
    }
    found.sort();

    let allowed: HashMap<&str, (usize, &str)> = ALLOWED
        .iter()
        .map(|(file, count, why)| (*file, (*count, *why)))
        .collect();

    let mut problems = Vec::new();
    for (file, count) in &found {
        match allowed.get(file.as_str()) {
            None => problems.push(format!(
                "{file}: {count} `temp_dir()` call(s) with no entry in ALLOWED. A test \
                 fixture must use `tempfile::tempdir()` (or `Builder::new().prefix(..)`) and hold \
                 the guard, so cleanup survives a panic; real runtime paths go in ALLOWED with a \
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
const SELF: &str = "src/temp_fixture_guard.rs";

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
fn collect(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, usize)>) {
    let entries = std::fs::read_dir(dir).expect("crate tree is readable");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, root, out);
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
