//! Integrity of a convert-at-install model's converted artifact (sc-24216, epic 24212).
//!
//! The catalog decides "converted" from a marker file alone (`mlx_catalog_status` looks for a
//! top-level `config.json` / `model_index.json`), and the worker trusts the injected `modelPath`.
//! For a converter whose output carries an engine-owned pin contract, that is not enough: a bundle
//! that was truncated, edited, or left half-written by an older build still reads as installed,
//! and the only symptom is the engine's raw load refusal naming internal paths.
//!
//! This module asks the ENGINE's own contract — never a SceneWorks re-implementation — whether the
//! artifact in `dir` would load. Converters with no declared contract report no problem: their
//! behaviour is unchanged.
//!
//! Two depths, split by cost:
//!
//! - [`converted_artifact_problem`] is the CHEAP check (manifest, pin contract, file sizes, plus a
//!   recorded verification failure — see below). The catalog runs it on every scan and the worker
//!   runs it before every load.
//! - [`diagnose_install`] is the FULL check (SHA-256 of every converted file and of every pinned
//!   co-requisite file). It runs only after a load has already FAILED, to turn the engine's raw
//!   error into an actionable one. The engine's own load re-hashes the bundle, so a healthy install
//!   is hashed exactly once per load and never by SceneWorks.
//!
//! A same-size corruption passes the cheap check and is only caught by the engine's load-time
//! digests. When [`diagnose_install`] confirms it, [`record_verification_failure_if_unchanged`]
//! writes [`VERIFICATION_FAILED_MARKER`] into the bundle directory (only if that directory is still
//! the one verified); the cheap check reports that recorded reason from then on, so the catalog
//! flips the model to `needs_conversion` and the Model Manager offers "Convert again". A re-convert replaces the whole directory, which clears the marker.
//!
//! - `supra2_img` → [`gen_core_checkpoint::supra2::read_pinned_bundle`] for the bundle, and the pinned
//!   [`gen_core_checkpoint::supra2::FLAN_T5_BASE`] / [`gen_core_checkpoint::supra2::SD_VAE`] snapshots (exact files,
//!   sizes, SHA-256, and their `config.json` geometry) for the co-requisites.

use std::path::{Path, PathBuf};

/// How much of the converted artifact is re-read by [`converted_artifact_problem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactCheck {
    /// Manifest + pin contract + file sizes (+ a recorded verification failure): metadata reads
    /// only, cheap enough for every catalog scan and every load.
    Sizes,
    /// [`ArtifactCheck::Sizes`] plus the SHA-256 of every converted file — catches same-size
    /// corruption. Hashes ~400 MB for Supra2; never run it on a hot path.
    Digests,
}

/// The converters whose output this module can check. A converter outside this list always
/// reports no problem.
pub const CONTRACT_CONVERTERS: &[&str] = &["supra2_img"];

/// Written into a converted directory when a full verification after a failed load found the
/// converted files themselves damaged. Its content is the engine's reason. Re-converting replaces
/// the directory and so removes it.
pub const VERIFICATION_FAILED_MARKER: &str = ".sceneworks-verification-failed";

/// The engine's reason the converted artifact in `dir` (written by `converter`) would be refused at
/// load, or `None` when it would load (or the converter declares no contract). A recorded
/// verification failure ([`VERIFICATION_FAILED_MARKER`]) is reported at either depth.
pub fn converted_artifact_problem(
    converter: &str,
    dir: &Path,
    check: ArtifactCheck,
) -> Option<String> {
    if !CONTRACT_CONVERTERS.contains(&converter) {
        return None;
    }
    if let Some(recorded) = recorded_verification_failure(dir) {
        return Some(recorded);
    }
    match converter {
        "supra2_img" => {
            let check = match check {
                ArtifactCheck::Sizes => gen_core_checkpoint::supra2::BundleCheck::SizesOnly,
                ArtifactCheck::Digests => gen_core_checkpoint::supra2::BundleCheck::Digests,
            };
            gen_core_checkpoint::supra2::read_pinned_bundle(dir, check)
                .err()
                .map(|error| error.to_string())
        }
        _ => None,
    }
}

/// The reason recorded by [`record_verification_failure`], if any.
pub fn recorded_verification_failure(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(VERIFICATION_FAILED_MARKER)).ok()?;
    let text = text.trim();
    Some(if text.is_empty() {
        "the converted files failed verification".to_owned()
    } else {
        text.to_owned()
    })
}

/// Record that the converted files in `dir` failed full verification, so the cheap check (and
/// with it the catalog) reports it until the model is converted again.
pub fn record_verification_failure(dir: &Path, problem: &str) -> std::io::Result<()> {
    std::fs::write(dir.join(VERIFICATION_FAILED_MARKER), problem)
}

/// Which directory currently sits at a converted-bundle path. A re-convert never edits a bundle in
/// place: it builds a fresh directory and renames it over the old one, so the directory's own
/// identity (device + inode on Unix, creation time everywhere) changes exactly when the bundle is
/// replaced. Writing the verification marker does not change it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleIdentity {
    #[cfg(unix)]
    dev_ino: (u64, u64),
    created: Option<std::time::SystemTime>,
}

/// The [`BundleIdentity`] of the directory at `dir`, or `None` when there is none.
pub fn bundle_identity(dir: &Path) -> Option<BundleIdentity> {
    let metadata = std::fs::metadata(dir).ok().filter(|meta| meta.is_dir())?;
    Some(BundleIdentity {
        #[cfg(unix)]
        dev_ino: {
            use std::os::unix::fs::MetadataExt;
            (metadata.dev(), metadata.ino())
        },
        created: metadata.created().ok(),
    })
}

/// [`record_verification_failure`] for the bundle that was VERIFIED: the marker lands only if the
/// directory at `dir` is still `verified` (see [`BundleIdentity`]). A full verification hashes for
/// seconds; a "Convert again" promoted meanwhile must not inherit the old bundle's verdict and read
/// as damaged. The reason is written to a temporary file in the directory, the identity is checked
/// again, and the temporary file is renamed onto the marker — so the marker is never partial, and a
/// swap at any point leaves the fresh bundle unmarked (a temporary file written into it is removed;
/// a rename after the swap finds no temporary file there). `Ok(false)` when the bundle changed.
pub fn record_verification_failure_if_unchanged(
    dir: &Path,
    verified: &BundleIdentity,
    problem: &str,
) -> std::io::Result<bool> {
    record_if_unchanged_with(dir, verified, problem, || {})
}

/// [`record_verification_failure_if_unchanged`] with a hook between the temporary write and the
/// second identity check — the seam the swap test drives.
fn record_if_unchanged_with(
    dir: &Path,
    verified: &BundleIdentity,
    problem: &str,
    after_temp_write: impl FnOnce(),
) -> std::io::Result<bool> {
    if bundle_identity(dir).as_ref() != Some(verified) {
        return Ok(false);
    }
    let temp = dir.join(format!(
        "{VERIFICATION_FAILED_MARKER}.{}.tmp",
        std::process::id()
    ));
    std::fs::write(&temp, problem)?;
    after_temp_write();
    if bundle_identity(dir).as_ref() != Some(verified) {
        let _ = std::fs::remove_file(&temp);
        return Ok(false);
    }
    std::fs::rename(&temp, dir.join(VERIFICATION_FAILED_MARKER))?;
    Ok(true)
}

/// What a full verification found wrong with an installed convert-at-install model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallFault {
    /// The converted files themselves (the bundle). Remedy: convert again.
    Converted { problem: String },
    /// A pinned co-requisite snapshot. Remedy: re-download that component.
    Component {
        /// The engine's component key (`text_encoder`, `vae`).
        component: &'static str,
        /// Human label ("Flan-T5-Base text encoder").
        label: &'static str,
        repo: &'static str,
        revision: &'static str,
        /// The resolved snapshot directory, when one is installed.
        snapshot: Option<PathBuf>,
        problem: String,
    },
}

/// Full verification of an installed model after a FAILED load (see the module docs): every pinned
/// co-requisite (resolved through `snapshot(repo, revision)`) and then the converted bundle, each
/// against its exact files, sizes and SHA-256. The first fault found is returned; `None` means the
/// installed files are intact and the failure has another cause. Blocking and expensive (hashes
/// every file) — call it off the async runtime, and only on a failure path.
pub fn diagnose_install(
    converter: &str,
    dir: &Path,
    snapshot: impl Fn(&str, &str) -> Option<PathBuf>,
) -> Option<InstallFault> {
    match converter {
        "supra2_img" => {
            use gen_core_checkpoint::supra2::{FLAN_T5_BASE, SD_VAE};
            let components = [
                (
                    gen_core_checkpoint::supra2::TEXT_ENCODER_COMPONENT,
                    "Flan-T5-Base text encoder",
                    FLAN_T5_BASE,
                ),
                (
                    gen_core_checkpoint::supra2::VAE_COMPONENT,
                    "SD VAE ft-MSE decoder",
                    SD_VAE,
                ),
            ];
            for (component, label, pin) in components {
                let fault = |snapshot: Option<PathBuf>, problem: String| InstallFault::Component {
                    component,
                    label,
                    repo: pin.repo,
                    revision: pin.revision,
                    snapshot,
                    problem,
                };
                let Some(snapshot_dir) = snapshot(pin.repo, pin.revision) else {
                    return Some(fault(None, "not installed".to_owned()));
                };
                let checked = pin.verify(&snapshot_dir).and_then(|()| {
                    if component == gen_core_checkpoint::supra2::VAE_COMPONENT {
                        gen_core_checkpoint::supra2::check_vae_config(&snapshot_dir)
                    } else {
                        gen_core_checkpoint::supra2::check_t5_config(
                            &snapshot_dir,
                            &gen_core_checkpoint::supra2::Supra2Config::published(),
                        )
                    }
                });
                if let Err(error) = checked {
                    return Some(fault(Some(snapshot_dir), error.to_string()));
                }
            }
            converted_artifact_problem(converter, dir, ArtifactCheck::Digests)
                .map(|problem| InstallFault::Converted { problem })
        }
        _ => None,
    }
}

/// Test support (used by the API and worker tests too): write a Supra2 bundle that satisfies the
/// pin contract at [`ArtifactCheck::Sizes`] — the pinned manifest and both converted files at
/// their exact pinned sizes (sparse, all zeros) — but whose CONTENT is wrong, i.e. a same-size
/// corruption that only a digest check catches. Also writes the catalog's install marker.
#[doc(hidden)]
pub fn write_same_size_corrupt_supra2_bundle(dir: &Path) -> std::io::Result<()> {
    use gen_core_checkpoint::supra2::*;
    std::fs::create_dir_all(dir)?;
    let file = |pin: PinnedFile| BundleFile {
        file: pin.path.to_owned(),
        bytes: pin.bytes,
        sha256: pin.sha256.to_owned(),
    };
    let source = SUPRA2_SOURCE.files[0];
    let config = Supra2Config::published();
    let manifest = BundleManifest {
        schema: BUNDLE_SCHEMA.to_owned(),
        provider_id: PROVIDER_ID.to_owned(),
        source: BundleSource {
            repo: Some(SUPRA2_SOURCE.repo.to_owned()),
            revision: Some(SUPRA2_SOURCE.revision.to_owned()),
            file: source.path.to_owned(),
            bytes: source.bytes,
            sha256: source.sha256.to_owned(),
            weights_key: "ema".to_owned(),
        },
        config,
        vae_scale: VAE_SCALE,
        denoiser: file(PINNED_DENOISER),
        denoiser_tensors: config.expected_tensors().len(),
        conditioning: Some(file(PINNED_CONDITIONING)),
        components: [
            (TEXT_ENCODER_COMPONENT, FLAN_T5_BASE),
            (VAE_COMPONENT, SD_VAE),
        ]
        .into_iter()
        .map(|(component, pin)| BundleComponent {
            component: component.to_owned(),
            repo: pin.repo.to_owned(),
            revision: pin.revision.to_owned(),
            license: pin.license.to_owned(),
            download_bytes: pin.download_bytes(),
        })
        .collect(),
    };
    for pin in [PINNED_DENOISER, PINNED_CONDITIONING] {
        std::fs::File::create(dir.join(pin.path))?.set_len(pin.bytes)?;
    }
    std::fs::write(
        dir.join(BUNDLE_MANIFEST),
        serde_json::to_vec_pretty(&manifest).map_err(std::io::Error::other)?,
    )?;
    std::fs::write(dir.join("model_index.json"), b"{}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_converter_without_a_contract_reports_no_problem() {
        let tmp = tempfile::tempdir().unwrap();
        // Even a stray marker is ignored for a converter with no contract.
        record_verification_failure(tmp.path(), "stray").unwrap();
        for check in [ArtifactCheck::Sizes, ArtifactCheck::Digests] {
            assert_eq!(
                converted_artifact_problem("anima_quant", tmp.path(), check),
                None
            );
        }
        assert_eq!(
            diagnose_install("flux2_klein_diffusers", tmp.path(), |_, _| None),
            None
        );
    }

    /// A marker-only directory (what an interrupted or pre-contract convert leaves) is refused with
    /// the engine's own reason, at both depths.
    #[test]
    fn a_supra2_dir_without_its_bundle_is_refused_with_the_engine_reason() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model_index.json"), b"{}").unwrap();
        for check in [ArtifactCheck::Sizes, ArtifactCheck::Digests] {
            let problem = converted_artifact_problem("supra2_img", tmp.path(), check)
                .expect("no bundle manifest");
            assert!(
                problem.contains("is not a converted Supra2 bundle"),
                "{problem}"
            );
        }
    }

    /// A bundle manifest that parses but does not hold the pinned conversion is refused.
    #[test]
    fn a_supra2_bundle_with_a_garbage_manifest_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path()
                .join(gen_core_checkpoint::supra2::BUNDLE_MANIFEST),
            b"{\"schema\":\"nope\"}",
        )
        .unwrap();
        let problem =
            converted_artifact_problem("supra2_img", tmp.path(), ArtifactCheck::Sizes).unwrap();
        assert!(problem.contains("supra2"), "{problem}");
    }

    /// Same-size corruption passes the cheap check, is caught by the digests, and once recorded
    /// the cheap check reports it too.
    #[test]
    fn same_size_corruption_is_caught_by_digests_and_then_by_the_recorded_marker() {
        let tmp = tempfile::tempdir().unwrap();
        write_same_size_corrupt_supra2_bundle(tmp.path()).unwrap();
        assert_eq!(
            converted_artifact_problem("supra2_img", tmp.path(), ArtifactCheck::Sizes),
            None,
            "sizes alone cannot see a same-size corruption"
        );
        let problem = converted_artifact_problem("supra2_img", tmp.path(), ArtifactCheck::Digests)
            .expect("the digests catch it");
        assert!(problem.contains("does not match its manifest"), "{problem}");
        record_verification_failure(tmp.path(), &problem).unwrap();
        assert_eq!(
            converted_artifact_problem("supra2_img", tmp.path(), ArtifactCheck::Sizes),
            Some(problem)
        );
    }

    /// The full diagnosis names a missing, then a damaged, co-requisite before the bundle.
    #[test]
    fn diagnosis_names_the_faulty_component_before_the_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("bundle");
        write_same_size_corrupt_supra2_bundle(&bundle).unwrap();
        let t5 = tmp.path().join("t5");
        let vae = tmp.path().join("vae");
        let resolve = |t5_present: bool, vae_present: bool| {
            let (t5, vae) = (t5.clone(), vae.clone());
            move |repo: &str, _: &str| match repo {
                "google/flan-t5-base" if t5_present => Some(t5.clone()),
                "stabilityai/sd-vae-ft-mse" if vae_present => Some(vae.clone()),
                _ => None,
            }
        };
        match diagnose_install("supra2_img", &bundle, resolve(false, false)) {
            Some(InstallFault::Component {
                component,
                snapshot: None,
                problem,
                ..
            }) => {
                assert_eq!(component, "text_encoder");
                assert_eq!(problem, "not installed");
            }
            other => panic!("{other:?}"),
        }
        // A truncated (wrong-size) text encoder weights file.
        std::fs::create_dir_all(&t5).unwrap();
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::write(t5.join(name), b"x").unwrap();
        }
        match diagnose_install("supra2_img", &bundle, resolve(true, true)) {
            Some(InstallFault::Component {
                component,
                label,
                snapshot: Some(dir),
                problem,
                ..
            }) => {
                assert_eq!(
                    (component, label),
                    ("text_encoder", "Flan-T5-Base text encoder")
                );
                assert_eq!(dir, t5);
                assert!(problem.contains("bytes"), "{problem}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// sc-24217 pre-merge review: the marker lands in the bundle that was verified, never in a
    /// fresh one a "Convert again" renamed over the same path meanwhile — and a write into the
    /// unchanged bundle is whole (no temporary file left behind).
    ///
    /// *Mutations that red this:* drop the first identity check (the marker lands in the fresh
    /// bundle); drop the second (the swap mid-write marks the fresh bundle).
    #[test]
    fn the_marker_only_lands_in_the_bundle_that_was_verified() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("supra2_img");
        std::fs::create_dir_all(&bundle).unwrap();
        let verified = bundle_identity(&bundle).expect("a directory");
        assert!(record_verification_failure_if_unchanged(&bundle, &verified, "bad").unwrap());
        assert_eq!(
            recorded_verification_failure(&bundle).as_deref(),
            Some("bad")
        );
        let leftovers: Vec<_> = std::fs::read_dir(&bundle)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            leftovers,
            [VERIFICATION_FAILED_MARKER],
            "no temporary file left"
        );

        // A re-convert: a fresh directory promoted over the verified one (the old one is kept
        // aside here only so its inode cannot be reused).
        let fresh = tmp.path().join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        std::fs::rename(&bundle, tmp.path().join("old")).unwrap();
        std::fs::rename(&fresh, &bundle).unwrap();
        assert!(!record_verification_failure_if_unchanged(&bundle, &verified, "bad").unwrap());
        assert_eq!(recorded_verification_failure(&bundle), None);
        assert_eq!(std::fs::read_dir(&bundle).unwrap().count(), 0);

        // A swap AFTER the first check, between the temporary write and the rename: the temporary
        // file went into the verified (now displaced) bundle, the fresh one stays unmarked.
        let verified = bundle_identity(&bundle).unwrap();
        let swapped = record_if_unchanged_with(&bundle, &verified, "bad", || {
            let fresh = tmp.path().join("fresh2");
            std::fs::create_dir_all(&fresh).unwrap();
            std::fs::rename(&bundle, tmp.path().join("old2")).unwrap();
            std::fs::rename(&fresh, &bundle).unwrap();
        })
        .unwrap();
        assert!(!swapped);
        assert_eq!(std::fs::read_dir(&bundle).unwrap().count(), 0);
        assert!(!record_verification_failure_if_unchanged(
            &tmp.path().join("gone"),
            &verified,
            "bad"
        )
        .unwrap());
    }
}
