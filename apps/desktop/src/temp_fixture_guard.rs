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
    "setup.rs",
    2,
    "production: the last-resort app-data / log dirs when no platform data dir resolves — real \
     runtime state the shell must keep, not fixtures",
)];

/// This file, skipped by the scan: it names the forbidden call in its own prose and matcher.
const SELF: &str = "temp_fixture_guard.rs";

/// No test fixture may build a path under the shared temp root by hand.
///
/// Mutation-checked: re-introducing one `std::env::temp_dir()` fixture anywhere under `src/` turns
/// this RED, and so does adding one to a file already on the allow-list.
#[test]
fn no_test_fixture_builds_a_path_under_the_shared_temp_root() {
    use std::collections::HashMap;

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found: Vec<(String, usize)> = Vec::new();
    collect(&src, &src, &mut found);
    found.sort();

    let allowed: HashMap<&str, (usize, &str)> = ALLOWED
        .iter()
        .map(|(file, count, why)| (*file, (*count, *why)))
        .collect();

    let mut problems = Vec::new();
    for (file, count) in &found {
        match allowed.get(file.as_str()) {
            None => problems.push(format!(
                "{file}: {count} `std::env::temp_dir()` call(s) with no entry in ALLOWED. A test \
                 fixture must use `tempfile::tempdir()` (or `Builder::new().prefix(..)`) and hold \
                 the guard, so cleanup survives a panic; real runtime paths go in ALLOWED with a \
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

/// Count `std::env::temp_dir()` call sites per `.rs` file under `dir`, keyed by the path relative
/// to `root`. Matches the fully-qualified call only, so doc comments that *mention* `temp_dir()`
/// in prose do not register.
fn collect(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, usize)>) {
    for entry in std::fs::read_dir(dir)
        .expect("desktop src tree is readable")
        .flatten()
    {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, root, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            if path
                .strip_prefix(root)
                .is_ok_and(|p| p == std::path::Path::new(SELF))
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source file is utf-8");
            let count = text.matches("std::env::temp_dir()").count();
            if count > 0 {
                out.push((
                    path.strip_prefix(root)
                        .expect("walked path is under the src root")
                        .to_string_lossy()
                        .replace('\\', "/"),
                    count,
                ));
            }
        }
    }
}
