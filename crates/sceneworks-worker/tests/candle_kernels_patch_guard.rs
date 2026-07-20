//! Structural guard for the vendored multi-arch candle-kernels `[patch]` (sc-7544 / sc-13510).
//!
//! candle-kernels compiles the GGUF quant/moe kernels into a static `libmoe.a` of SASS with a
//! single `-gencode` derived from `CUDA_COMPUTE_CAP`. At the cap=80 packaging baseline that is an
//! Ampere-only cubin with no PTX, so on Blackwell (sm_120) every quantized matmul silently
//! returns zeros — quantized models render solid black. The fix is the inference repo's vendored
//! candle-kernels (multi-arch fatbin build.rs), applied here through the root `Cargo.toml`
//! `[patch."https://github.com/huggingface/candle"]`.
//!
//! That patch only takes effect in the top-level workspace, and it already silently dropped out
//! once (sc-13510: the candle-gen -> inference cutover rebuilt the workspace without it, and
//! nothing failed). This test makes the loss loud: it asserts, from the committed `Cargo.lock`,
//! that candle-kernels resolves through the patch AND at the same inference revision as the
//! worker's direct pins. It parses files only — no CUDA, no GPU — so every CI lane runs it.
//!
//! If it fails after a pin bump: re-run `node scripts/bump-inference.mjs` (it rewrites the root
//! [patch] rev in lockstep with crates/sceneworks-worker/Cargo.toml and refreshes the lock).

use std::path::Path;

const UPSTREAM_CANDLE: &str = "github.com/huggingface/candle";
const INFERENCE_REPO: &str = "git+https://github.com/SceneWorks/inference";

/// `(name, source)` for every `[[package]]` block in a Cargo.lock. Path dependencies have no
/// `source` line and yield `None`. `[[patch.unused]]` blocks are ignored — an unused patch is
/// exactly the failure this guard exists to catch, so it must not satisfy the lookup.
fn parse_lock_packages(lock: &str) -> Vec<(String, Option<String>)> {
    let mut packages = Vec::new();
    let mut in_package = false;
    let mut name: Option<String> = None;
    let mut source: Option<String> = None;
    for line in lock.lines() {
        let line = line.trim();
        if line.starts_with("[[") {
            if in_package {
                if let Some(n) = name.take() {
                    packages.push((n, source.take()));
                }
            }
            in_package = line == "[[package]]";
            name = None;
            source = None;
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(v) = line.strip_prefix("name = ") {
            name = Some(v.trim_matches('"').to_string());
        } else if let Some(v) = line.strip_prefix("source = ") {
            source = Some(v.trim_matches('"').to_string());
        }
    }
    if in_package {
        if let Some(n) = name {
            packages.push((n, source));
        }
    }
    packages
}

/// The resolved commit a git source locked to (the `#<sha>` fragment).
fn resolved_commit(source: &str) -> Option<&str> {
    source.split('#').nth(1)
}

/// Core check, separated from file I/O so the red paths below are testable: candle-kernels must
/// resolve from the inference repo (the patch is live) at the same resolved commit as
/// sceneworks-gen-core (the patch rev is in lockstep with the worker's inference pins).
fn check_lock(lock: &str) -> Result<(), String> {
    let packages = parse_lock_packages(lock);
    let kernels: Vec<&Option<String>> = packages
        .iter()
        .filter(|(n, _)| n == "candle-kernels")
        .map(|(_, s)| s)
        .collect();
    let [kernels_source] = kernels.as_slice() else {
        return Err(format!(
            "expected exactly one candle-kernels package in Cargo.lock, found {}",
            kernels.len()
        ));
    };
    let Some(kernels_source) = kernels_source else {
        // A path source would mean a SceneWorks-local vendor copy this guard doesn't know about.
        return Err("candle-kernels has no source (unexpected path dependency)".to_string());
    };
    if kernels_source.contains(UPSTREAM_CANDLE) {
        return Err(format!(
            "candle-kernels resolves from upstream candle ({kernels_source}): the root Cargo.toml \
             [patch] to the inference repo's vendored multi-arch copy is not in effect, so \
             packaged quantized models silently break on Blackwell (sc-7544 / sc-13510)"
        ));
    }
    // The repo path must END at the repo name (`?rev=` query or fragment), so a lookalike
    // repo (e.g. .../inference-archive) cannot satisfy the check.
    let after_repo = kernels_source.strip_prefix(INFERENCE_REPO);
    if !matches!(after_repo, Some(rest) if rest.is_empty() || rest.starts_with('?') || rest.starts_with('#'))
    {
        return Err(format!(
            "candle-kernels resolves from an unexpected source: {kernels_source}"
        ));
    }
    let gen_core = packages
        .iter()
        .find(|(n, _)| n == "sceneworks-gen-core")
        .and_then(|(_, s)| s.as_deref())
        .ok_or("sceneworks-gen-core not found in Cargo.lock")?;
    match (resolved_commit(kernels_source), resolved_commit(gen_core)) {
        (Some(k), Some(g)) if k == g => Ok(()),
        (k, g) => Err(format!(
            "candle-kernels [patch] rev skews from the worker's inference pin \
             (candle-kernels {k:?} vs sceneworks-gen-core {g:?}): the vendored kernels no longer \
             match the pinned candle-core. Re-run `node scripts/bump-inference.mjs`."
        )),
    }
}

/// The committed workspace lockfile passes the guard.
#[test]
fn candle_kernels_resolves_through_the_inference_patch() {
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    let lock = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", lock_path.display()));
    if let Err(msg) = check_lock(&lock) {
        panic!("{msg}");
    }
}

// Red-path coverage: each canned lockfile is the guard's target failure, so a regression in the
// parser or the check itself cannot silently turn the real test above into a false green.

const GOOD_LOCK: &str = r#"
[[package]]
name = "candle-kernels"
version = "0.10.2"
source = "git+https://github.com/SceneWorks/inference?rev=d68b8b45#d68b8b457d76e0472393f0d7bfe0e79ae68278dd"

[[package]]
name = "sceneworks-gen-core"
version = "0.1.0"
source = "git+https://github.com/SceneWorks/inference?rev=d68b8b45#d68b8b457d76e0472393f0d7bfe0e79ae68278dd"
"#;

#[test]
fn guard_accepts_a_patched_lockstep_lock() {
    assert_eq!(check_lock(GOOD_LOCK), Ok(()));
}

#[test]
fn guard_rejects_upstream_candle_kernels() {
    // The sc-13510 regression verbatim: the patch dropped, candle-kernels back on upstream.
    let lock = GOOD_LOCK.replace(
        "git+https://github.com/SceneWorks/inference?rev=d68b8b45#d68b8b457d76e0472393f0d7bfe0e79ae68278dd\"\n\n[[package]]",
        "git+https://github.com/huggingface/candle?rev=1e6aa85e#1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d\"\n\n[[package]]",
    );
    let err = check_lock(&lock).unwrap_err();
    assert!(err.contains("upstream candle"), "{err}");
}

#[test]
fn guard_rejects_a_lookalike_repo() {
    // The repo check must not accept a source whose path merely starts with the inference repo's.
    let lock = GOOD_LOCK.replacen(
        "github.com/SceneWorks/inference?rev=d68b8b45",
        "github.com/SceneWorks/inference-archive?rev=d68b8b45",
        1,
    );
    let err = check_lock(&lock).unwrap_err();
    assert!(err.contains("unexpected source"), "{err}");
}

#[test]
fn guard_rejects_a_rev_skew() {
    // Worker pins bumped, root [patch] left behind: same repo, different resolved commit.
    let lock = GOOD_LOCK.replacen(
        "#d68b8b457d76e0472393f0d7bfe0e79ae68278dd",
        "#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        1,
    );
    let err = check_lock(&lock).unwrap_err();
    assert!(err.contains("skews"), "{err}");
}

#[test]
fn guard_rejects_a_missing_candle_kernels_package() {
    let lock = GOOD_LOCK.replacen("name = \"candle-kernels\"", "name = \"renamed\"", 1);
    let err = check_lock(&lock).unwrap_err();
    assert!(err.contains("exactly one candle-kernels"), "{err}");
}

#[test]
fn guard_ignores_patch_unused_blocks() {
    // An unused patch records the package under [[patch.unused]] — that must NOT count as a
    // resolution, it is precisely the broken state.
    let lock = GOOD_LOCK.replacen("[[package]]", "[[patch.unused]]", 1);
    let err = check_lock(&lock).unwrap_err();
    assert!(err.contains("exactly one candle-kernels"), "{err}");
}
