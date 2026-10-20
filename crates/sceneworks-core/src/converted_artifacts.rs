//! Integrity of a convert-at-install model's converted artifact (sc-24216, epic 24212).
//!
//! The catalog decides "converted" from a marker file alone (`mlx_catalog_status` looks for a
//! top-level `config.json` / `model_index.json`), and the worker trusts the injected `modelPath`.
//! For a converter whose output carries an engine-owned pin contract, that is not enough: a bundle
//! that was truncated, edited, or left half-written by an older build still reads as installed,
//! and the only symptom is the engine's raw load refusal naming internal paths.
//!
//! This module asks the ENGINE's own contract — never a SceneWorks re-implementation — whether the
//! artifact in `dir` would load, so the catalog can report it as needing conversion (with the
//! reason) and the worker can refuse with an actionable message before any load. Converters with no
//! declared contract report no problem: their behaviour is unchanged.
//!
//! - `supra2_img` → [`gen_core::supra2::read_pinned_bundle`]: schema, provider id, published
//!   geometry, pinned source identity, pinned component revisions, stored conditioning, and the
//!   size (always) and SHA-256 ([`ArtifactCheck::Digests`]) of every converted file.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// How much of the converted artifact is re-read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactCheck {
    /// Manifest + pin contract + file sizes: metadata reads only, cheap enough for every catalog
    /// scan.
    Sizes,
    /// [`ArtifactCheck::Sizes`] plus the SHA-256 of every converted file — catches same-size
    /// corruption. Use [`verified_converted_artifact_problem`] to pay it once per artifact state.
    Digests,
}

/// The converters whose output this module can check. A converter outside this list always
/// reports no problem.
pub const CONTRACT_CONVERTERS: &[&str] = &["supra2_img"];

/// The engine's reason the converted artifact in `dir` (written by `converter`) would be refused at
/// load, or `None` when it would load (or the converter declares no contract).
pub fn converted_artifact_problem(
    converter: &str,
    dir: &Path,
    check: ArtifactCheck,
) -> Option<String> {
    match converter {
        "supra2_img" => {
            let check = match check {
                ArtifactCheck::Sizes => gen_core::supra2::BundleCheck::SizesOnly,
                ArtifactCheck::Digests => gen_core::supra2::BundleCheck::Digests,
            };
            gen_core::supra2::read_pinned_bundle(dir, check)
                .err()
                .map(|error| error.to_string())
        }
        _ => None,
    }
}

/// [`converted_artifact_problem`] at [`ArtifactCheck::Digests`], hashed at most once per process
/// for an unchanged artifact. The memo key is the directory plus the (name, length, mtime) of every
/// top-level file, so a re-convert, a truncation or an in-place edit re-verifies; only a PASS is
/// remembered, so a broken artifact is reported every time it is asked about.
pub fn verified_converted_artifact_problem(converter: &str, dir: &Path) -> Option<String> {
    if !CONTRACT_CONVERTERS.contains(&converter) {
        return None;
    }
    static VERIFIED: OnceLock<Mutex<HashSet<(PathBuf, Vec<FileStamp>)>>> = OnceLock::new();
    let key = (dir.to_path_buf(), file_stamps(dir));
    let verified = VERIFIED.get_or_init(Mutex::default);
    if verified
        .lock()
        .map(|set| set.contains(&key))
        .unwrap_or(false)
    {
        return None;
    }
    let problem = converted_artifact_problem(converter, dir, ArtifactCheck::Digests);
    if problem.is_none() {
        if let Ok(mut set) = verified.lock() {
            set.insert(key);
        }
    }
    problem
}

type FileStamp = (String, u64, Option<SystemTime>);

fn file_stamps(dir: &Path) -> Vec<FileStamp> {
    let mut stamps: Vec<FileStamp> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            metadata.is_file().then(|| {
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    metadata.len(),
                    metadata.modified().ok(),
                )
            })
        })
        .collect();
    stamps.sort();
    stamps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_converter_without_a_contract_reports_no_problem() {
        let tmp = tempfile::tempdir().unwrap();
        for check in [ArtifactCheck::Sizes, ArtifactCheck::Digests] {
            assert_eq!(
                converted_artifact_problem("anima_quant", tmp.path(), check),
                None
            );
        }
        assert_eq!(
            verified_converted_artifact_problem("flux2_klein_diffusers", tmp.path()),
            None
        );
    }

    /// A marker-only directory (what an interrupted or pre-contract convert leaves) is refused with
    /// the engine's own reason, at both depths, and is never memoized as verified.
    #[test]
    fn a_supra2_dir_without_its_bundle_is_refused_with_the_engine_reason() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model_index.json"), b"{}").unwrap();
        for check in [ArtifactCheck::Sizes, ArtifactCheck::Digests] {
            let problem = converted_artifact_problem("supra2_img", tmp.path(), check)
                .expect("no bundle manifest");
            assert!(
                problem.contains(gen_core::supra2::BUNDLE_MANIFEST),
                "{problem}"
            );
        }
        assert!(verified_converted_artifact_problem("supra2_img", tmp.path()).is_some());
        assert!(verified_converted_artifact_problem("supra2_img", tmp.path()).is_some());
    }

    /// A bundle manifest that parses but does not hold the pinned conversion is refused.
    #[test]
    fn a_supra2_bundle_with_a_garbage_manifest_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join(gen_core::supra2::BUNDLE_MANIFEST),
            b"{\"schema\":\"nope\"}",
        )
        .unwrap();
        let problem =
            converted_artifact_problem("supra2_img", tmp.path(), ArtifactCheck::Sizes).unwrap();
        assert!(problem.contains("supra2"), "{problem}");
    }
}
