//! Actionable errors for a convert-at-install model whose installed files are damaged (sc-24216).
//!
//! The image lane refuses a converted bundle that fails the engine's CHEAP check before any load
//! (`converted_artifact_preflight` in `image_jobs/base.rs`). Corruption that keeps every file's
//! size — a flipped byte in the converted denoiser, or in a shared co-requisite such as the
//! Flan-T5-Base encoder — only surfaces as the engine's own load-time error (a digest mismatch, a
//! safetensors parse failure, …), which names internal paths and no remedy.
//!
//! [`explain_failed_load`] runs after such a job has FAILED: for a model whose converter has an
//! engine pin contract it re-verifies every installed file in full
//! ([`sceneworks_core::converted_artifacts::diagnose_install`], off the async runtime) and, when it
//! finds the fault, replaces the raw error with one that names the damaged part and the remedy. A
//! damaged converted bundle is also RECORDED next to the bundle, so the catalog reports the model
//! as needing conversion and the Model Manager enables "Convert again". A healthy install is never
//! hashed by SceneWorks: only a job whose ENGINE LOAD failed pays for the diagnosis (a validation,
//! admission, out-of-memory, generation-time or cancel failure never does).

use crate::settings::Settings;
use crate::{WorkerError, WorkerResult};
use sceneworks_core::contracts::JsonObject;
use sceneworks_core::converted_artifacts::{self, InstallFault};
use serde_json::Value;
use std::path::PathBuf;

/// The display name a message should use: the catalog `name`, else the model id.
pub(crate) fn model_display_name<'a>(entry: &'a JsonObject, model_id: &'a str) -> &'a str {
    entry
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(model_id)
}

/// The refusal for damaged converted files: what is wrong, and the two remedies in order.
pub(crate) fn converted_files_message(name: &str, problem: &str) -> String {
    format!(
        "{name}'s converted model files are damaged or incomplete, so it cannot load. Open the \
         Model Manager and run Convert again to rebuild them from the downloaded checkpoint; if \
         the conversion then fails, delete {name} and download it again. Detail: {problem}"
    )
}

/// The refusal for a damaged or missing co-requisite component.
pub(crate) fn component_message(name: &str, fault: &InstallFault) -> Option<String> {
    let InstallFault::Component {
        label,
        repo,
        revision,
        snapshot,
        problem,
        ..
    } = fault
    else {
        return None;
    };
    Some(match snapshot {
        Some(dir) => format!(
            "{name}'s {label} ({repo}@{revision}) is damaged, so {name} cannot load. Delete the \
             folder {} and then use Fix (or Download) on {name} in the Model Manager to download \
             it again. This component may be shared with other models; they will use the fresh \
             copy too. Detail: {problem}",
            dir.display()
        ),
        None => format!(
            "{name}'s {label} ({repo}@{revision}) is not installed, so {name} cannot load. Use Fix \
             (or Download) on {name} in the Model Manager to download it."
        ),
    })
}

/// After an image job whose ENGINE LOAD failed ([`is_engine_load_failure`]): when the model is convert-at-install with an engine pin contract,
/// re-verify its installed files in full and, if one is at fault, return an actionable error in
/// place of `error` (recording a damaged converted bundle for the catalog). Every other failure —
/// and every other model — is returned unchanged.
pub(crate) async fn explain_failed_load(
    settings: &Settings,
    payload: &JsonObject,
    error: WorkerError,
) -> WorkerError {
    let data_dir = settings.data_dir.clone();
    explain_failed_load_with(settings, payload, error, move |converter, bundle| {
        converted_artifacts::diagnose_install(converter, bundle, |repo, revision| {
            crate::model_jobs::huggingface_pinned_snapshot_dir(&data_dir, repo, revision)
        })
    })
    .await
}

/// [`explain_failed_load`] with the full diagnosis injected (`diagnose(converter, bundle)`, run
/// off the async runtime) — the seam the tests drive, since a passing co-requisite needs the real
/// pinned bytes.
async fn explain_failed_load_with(
    settings: &Settings,
    payload: &JsonObject,
    error: WorkerError,
    diagnose: impl FnOnce(&str, &std::path::Path) -> Option<InstallFault> + Send + 'static,
) -> WorkerError {
    if !is_engine_load_failure(&error) {
        return error;
    }
    let Some((name, converter)) = contract_install(payload) else {
        return error;
    };
    let Some(bundle) = loaded_bundle_dir(settings, payload) else {
        return error;
    };
    let diagnosis = crate::load_progress::spawn_blocking(move || {
        // What sits at the bundle path NOW: the marker below lands only if a "Convert again"
        // has not promoted a fresh bundle over this one while it was being hashed.
        let verified = converted_artifacts::bundle_identity(&bundle);
        let fault = diagnose(&converter, &bundle);
        if let (Some(InstallFault::Converted { problem }), Some(verified)) = (&fault, &verified) {
            // Best effort: the actionable message below stands even if the record cannot be
            // written (a read-only bundle dir, or a bundle replaced meanwhile); the next failed
            // load diagnoses again.
            let _ = converted_artifacts::record_verification_failure_if_unchanged(
                &bundle, verified, problem,
            );
        }
        fault
    })
    .await;
    match diagnosis {
        Ok(Some(InstallFault::Converted { problem })) => {
            WorkerError::InvalidPayload(converted_files_message(&name, &problem))
        }
        Ok(Some(fault)) => match component_message(&name, &fault) {
            Some(message) => WorkerError::InvalidPayload(message),
            None => error,
        },
        // Intact install (the failure has another cause), or the diagnosis task itself failed:
        // the original error is the honest answer.
        Ok(None) | Err(_) => error,
    }
}

/// The context every image arm's generator load wraps an engine load error in
/// (`"{engine} load failed"` on MLX, `"candle {engine} load failed"` on Candle — through
/// `generator_cache`'s `classify_engine_error(load_error_context, …)`).
const ENGINE_LOAD_FAILURE: &str = " load failed: ";

/// Substrings of an out-of-memory load failure: the files are not the cause, so it is not diagnosed.
const OUT_OF_MEMORY_MARKERS: &[&str] = &[
    "out of memory",
    "OutOfMemory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "insufficient memory",
    "Attempting to allocate",
];

/// Is `error` the engine's own load-time failure — the only place a size-preserving corruption of
/// the converted bundle or a co-requisite surfaces (digest mismatch, safetensors parse, missing
/// tensor)? Everything else is never diagnosed, so a failed job does not re-hash ~1.7 GB for a cause
/// the install cannot have: a pre-load validation/routing refusal (`validate_request`, the step
/// ceiling, the geometry envelope), an admission or out-of-memory refusal, a generation-time
/// failure, or a cancel. A cheap pre-load integrity refusal (`converted_artifact_preflight`,
/// weight resolution, component staging) already carries its own actionable message.
fn is_engine_load_failure(error: &WorkerError) -> bool {
    match error {
        WorkerError::InvalidPayload(message) | WorkerError::Engine(message) => {
            message.contains(ENGINE_LOAD_FAILURE)
                && !OUT_OF_MEMORY_MARKERS
                    .iter()
                    .any(|marker| message.contains(marker))
        }
        _ => false,
    }
}

/// The bundle the job's load read: its `modelPath` through the SAME confined resolution the image
/// lane loads from ([`crate::image_jobs::resolve_model_path_dir`] — `advanced.modelPath` first,
/// then the injected entry path, refused outside the data dir). `None` when the job names no
/// `modelPath` or it does not resolve (the load refused it before any engine load), so nothing
/// outside the data dir is ever hashed or marked.
#[cfg(any(target_os = "macos", feature = "backend-candle"))]
fn loaded_bundle_dir(settings: &Settings, payload: &JsonObject) -> Option<PathBuf> {
    let request = sceneworks_core::image_request::ImageRequest::from_payload(payload);
    crate::image_jobs::resolve_model_path_dir(&request, settings)
        .ok()
        .flatten()
}

/// A build with no image engine linked loads no bundle (its image jobs stop at the stub gap, never
/// at an engine load), so there is nothing to diagnose.
#[cfg(not(any(target_os = "macos", feature = "backend-candle")))]
fn loaded_bundle_dir(_settings: &Settings, _payload: &JsonObject) -> Option<PathBuf> {
    None
}

/// `(display name, converter)` for a job whose model is convert-at-install with an engine pin
/// contract.
fn contract_install(payload: &JsonObject) -> Option<(String, String)> {
    let entry = payload
        .get("modelManifestEntry")
        .and_then(Value::as_object)?;
    let mlx = entry.get("mlx")?;
    if mlx.get("requiresConversion").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let converter = mlx.get("converter").and_then(Value::as_str)?.trim();
    if !converted_artifacts::CONTRACT_CONVERTERS.contains(&converter) {
        return None;
    }
    let model_id = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Some((
        model_display_name(entry, model_id).to_owned(),
        converter.to_owned(),
    ))
}

/// `explain_failed_load` over a job's result — the one call site's shape.
pub(crate) async fn explain_job_result(
    settings: &Settings,
    payload: &JsonObject,
    result: WorkerResult<()>,
) -> WorkerResult<()> {
    match result {
        Err(error) => Err(explain_failed_load(settings, payload, error).await),
        ok => ok,
    }
}

#[cfg(test)]
mod entry_point_tests {
    /// Tripwire: the production image-job entry point (whose dispatch needs a live API and GPU)
    /// routes every result through the explanation seam the `tests` module below exercises.
    #[test]
    fn the_image_job_entry_point_explains_its_failures() {
        let source = include_str!("image_jobs.rs");
        let body = source
            .split_once("pub(crate) async fn run_image_generate_job(")
            .expect("entry point present")
            .1
            .split_once("\nasync fn run_image_generate_job_unexplained(")
            .expect("the unexplained body follows")
            .0;
        assert!(
            body.contains("run_image_generate_job_unexplained(api, settings, job).await")
                && body.contains(
                    "crate::converted_install::explain_job_result(settings, &job.payload, result)"
                ),
            "{body}"
        );
    }
}

// The diagnosis resolves the bundle through the image lane's loader (`resolve_model_path_dir`),
// which only a build with an image engine links; a build without one never reaches an engine load.
#[cfg(all(test, any(target_os = "macos", feature = "backend-candle")))]
mod tests {
    use super::*;
    use serde_json::json;

    fn settings_at(data_dir: &std::path::Path) -> Settings {
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data_dir.to_path_buf();
        settings
    }

    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    fn payload(bundle: &std::path::Path) -> JsonObject {
        json!({
            "model": "supra2_img",
            "modelManifestEntry": {
                "id": "supra2_img",
                "name": "Supra2-IMG",
                "modelPath": bundle,
                "mlx": { "requiresConversion": true, "converter": "supra2_img" }
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    /// Stage `files` at their exact pinned sizes (zeros) in `pin`'s snapshot: present and
    /// size-correct, wrong content — a same-size corruption only the digests see.
    fn stage_same_size_component(
        data_dir: &std::path::Path,
        pin: gen_core_checkpoint::supra2::ComponentPin,
    ) {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(data_dir, pin.repo)
            .unwrap()
            .join("snapshots")
            .join(pin.revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        for file in pin.files {
            std::fs::File::create(snapshot.join(file.path))
                .unwrap()
                .set_len(file.bytes)
                .unwrap();
        }
    }

    /// An engine-level failure (here a raw load error) on a same-size-corrupt CONVERTED bundle —
    /// components verified (the injected diagnosis stands in for the real pinned component
    /// bytes) — becomes the actionable "Convert again" message, and the fault is RECORDED, so the
    /// cheap check the catalog and the next preflight run now reports it.
    #[tokio::test]
    async fn a_digest_failure_in_the_converted_bundle_is_explained_and_recorded() {
        let data = tempfile::tempdir().unwrap();
        let bundle = data.path().join("models/mlx/supra2_img");
        converted_artifacts::write_same_size_corrupt_supra2_bundle(&bundle).unwrap();
        let cheap = || {
            converted_artifacts::converted_artifact_problem(
                "supra2_img",
                &bundle,
                converted_artifacts::ArtifactCheck::Sizes,
            )
        };
        assert_eq!(
            cheap(),
            None,
            "a same-size corruption passes the cheap check"
        );
        let error = explain_failed_load_with(
            &settings_at(data.path()),
            &payload(&bundle),
            WorkerError::InvalidPayload(
                "mlx supra2_img load failed: unsupported: supra2: bundle file ...".to_owned(),
            ),
            |converter, bundle| {
                converted_artifacts::converted_artifact_problem(
                    converter,
                    bundle,
                    converted_artifacts::ArtifactCheck::Digests,
                )
                .map(|problem| InstallFault::Converted { problem })
            },
        )
        .await;
        let message = error.to_string();
        assert!(
            message.starts_with("Supra2-IMG's converted model files are damaged or incomplete")
                && message.contains("run Convert again")
                && message.contains("does not match its manifest"),
            "{message}"
        );
        let recorded = cheap().expect("the fault is recorded for the catalog");
        assert!(
            recorded.contains("does not match its manifest"),
            "{recorded}"
        );
    }

    /// A raw engine error on a model whose co-requisite is same-size corrupt is replaced by a
    /// message naming the component, its folder and the remedy; the converted bundle is NOT
    /// marked (it is not the fault).
    #[tokio::test]
    async fn a_corrupt_component_is_named_with_its_remedy() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let bundle = data.path().join("models/mlx/supra2_img");
        converted_artifacts::write_same_size_corrupt_supra2_bundle(&bundle).unwrap();
        stage_same_size_component(data.path(), gen_core_checkpoint::supra2::FLAN_T5_BASE);
        stage_same_size_component(data.path(), gen_core_checkpoint::supra2::SD_VAE);
        let error = explain_failed_load(
            &settings_at(data.path()),
            &payload(&bundle),
            WorkerError::Engine("mlx supra2_img load failed: safetensors header".to_owned()),
        )
        .await;
        let message = error.to_string();
        assert!(
            message.contains("Supra2-IMG's Flan-T5-Base text encoder (google/flan-t5-base@")
                && message.contains("Delete the folder")
                && message.contains("use Fix (or Download) on Supra2-IMG"),
            "{message}"
        );
        assert!(converted_artifacts::recorded_verification_failure(&bundle).is_none());
    }

    /// A missing component is named too; a cancel, an unrelated model and a model without a
    /// `modelPath` keep their original error untouched.
    #[tokio::test]
    async fn only_contract_installs_are_explained() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let bundle = data.path().join("models/mlx/supra2_img");
        converted_artifacts::write_same_size_corrupt_supra2_bundle(&bundle).unwrap();
        let settings = settings_at(data.path());
        // Through the job-result seam `run_image_generate_job` uses: a failure is explained, a
        // success passes through.
        let missing = explain_job_result(
            &settings,
            &payload(&bundle),
            Err(WorkerError::Engine(LOAD_FAILURE.to_owned())),
        )
        .await
        .expect_err("still a failure")
        .to_string();
        assert!(missing.contains("is not installed"), "{missing}");
        assert!(explain_job_result(&settings, &payload(&bundle), Ok(()))
            .await
            .is_ok());

        let canceled = explain_failed_load(
            &settings,
            &payload(&bundle),
            WorkerError::Canceled("user".to_owned()),
        )
        .await;
        assert!(matches!(canceled, WorkerError::Canceled(_)));

        let mut other = payload(&bundle);
        other["modelManifestEntry"]["mlx"]["converter"] = json!("anima_quant");
        let untouched = explain_failed_load(
            &settings,
            &other,
            WorkerError::Engine(LOAD_FAILURE.to_owned()),
        )
        .await;
        assert_eq!(untouched.to_string(), LOAD_FAILURE);

        let mut no_path = payload(&bundle);
        no_path["modelManifestEntry"]
            .as_object_mut()
            .unwrap()
            .remove("modelPath");
        let untouched = explain_failed_load(
            &settings,
            &no_path,
            WorkerError::Engine(LOAD_FAILURE.to_owned()),
        )
        .await;
        assert_eq!(untouched.to_string(), LOAD_FAILURE);
    }

    /// A diagnosis that reports the converted bundle damaged and records which dir it judged.
    fn judged_damaged(
        judged: std::sync::Arc<std::sync::Mutex<Vec<PathBuf>>>,
    ) -> impl FnOnce(&str, &std::path::Path) -> Option<InstallFault> + Send + 'static {
        move |_, bundle| {
            judged.lock().unwrap().push(bundle.to_path_buf());
            Some(InstallFault::Converted {
                problem: "denoiser.safetensors does not match its manifest".to_owned(),
            })
        }
    }

    /// sc-24217 pre-merge review: the diagnosis judges — and marks — the bundle the LOAD read,
    /// through the load's own confined resolution: `advanced.modelPath` wins over the injected
    /// entry path exactly as `resolve_weights_dir` does, and a path outside the data dir is never
    /// hashed or marked.
    ///
    /// *Mutations that red this:* read the bundle off `modelManifestEntry.modelPath` again (the
    /// installed bundle is judged and marked instead); drop the confinement (the outside dir is
    /// judged and marked).
    #[tokio::test]
    async fn the_diagnosis_judges_the_bundle_the_load_resolved() {
        let data = tempfile::tempdir().unwrap();
        let settings = settings_at(data.path());
        let installed = data.path().join("models/mlx/supra2_img");
        let other = data.path().join("models/mlx/supra2_other");
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let mut aimed = payload(&installed);
        aimed.insert("advanced".to_owned(), json!({ "modelPath": other }));
        let judged = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let error = explain_failed_load_with(
            &settings,
            &aimed,
            WorkerError::Engine(LOAD_FAILURE.to_owned()),
            judged_damaged(judged.clone()),
        )
        .await;
        assert!(error.to_string().contains("run Convert again"), "{error}");
        // The confined resolver canonicalizes (macOS temp dirs live under `/private`).
        assert_eq!(*judged.lock().unwrap(), vec![other.canonicalize().unwrap()]);
        assert!(converted_artifacts::recorded_verification_failure(&other).is_some());
        assert!(converted_artifacts::recorded_verification_failure(&installed).is_none());

        let outside = tempfile::tempdir().unwrap();
        let judged = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let error = explain_failed_load_with(
            &settings,
            &payload(outside.path()),
            WorkerError::Engine(LOAD_FAILURE.to_owned()),
            judged_damaged(judged.clone()),
        )
        .await;
        assert_eq!(error.to_string(), LOAD_FAILURE);
        assert!(judged.lock().unwrap().is_empty(), "never hashed");
        assert!(converted_artifacts::recorded_verification_failure(outside.path()).is_none());
    }

    /// sc-24217 pre-merge review: a "Convert again" that promotes a fresh bundle while a failed
    /// job's diagnosis is hashing the old one leaves the fresh bundle unmarked (the catalog keeps
    /// reading it as converted); the failed job still gets the actionable message.
    ///
    /// *Mutation that reds this:* record through the unguarded `record_verification_failure`.
    #[tokio::test]
    async fn a_bundle_replaced_during_the_diagnosis_is_not_marked() {
        let data = tempfile::tempdir().unwrap();
        let settings = settings_at(data.path());
        let bundle = data.path().join("models/mlx/supra2_img");
        std::fs::create_dir_all(&bundle).unwrap();
        let root = data.path().to_path_buf();
        let promoted = bundle.clone();
        let error = explain_failed_load_with(
            &settings,
            &payload(&bundle),
            WorkerError::Engine(LOAD_FAILURE.to_owned()),
            move |_, _| {
                // The convert job's promotion, mid-hash: a fresh dir renamed over the old one.
                let fresh = root.join("models/mlx/.supra2_img.converting");
                std::fs::create_dir_all(&fresh).unwrap();
                std::fs::rename(&promoted, root.join("models/mlx/.supra2_img.old")).unwrap();
                std::fs::rename(&fresh, &promoted).unwrap();
                Some(InstallFault::Converted {
                    problem: "denoiser.safetensors does not match its manifest".to_owned(),
                })
            },
        )
        .await;
        assert!(error.to_string().contains("run Convert again"), "{error}");
        assert!(converted_artifacts::recorded_verification_failure(&bundle).is_none());
        assert_eq!(std::fs::read_dir(&bundle).unwrap().count(), 0);
    }

    /// A raw engine load error, as `generator_cache` wraps it for the MLX arm.
    const LOAD_FAILURE: &str = "supra2_img load failed: backend op failed: safetensors header";

    /// Epic-review E3: only an ENGINE LOAD failure pays for the full-install diagnosis. A pre-load
    /// validation refusal, an admission refusal, a cheap pre-load integrity refusal (already
    /// actionable), an out-of-memory load, a generation-time failure and a cancel are returned
    /// untouched WITHOUT diagnosing; a load failure on either backend (the typed `Unsupported`
    /// digest refusal arrives as `InvalidPayload`, everything else as `Engine`) is diagnosed.
    ///
    /// *Mutations that red this:* diagnose every non-cancel error (the pre-fix behaviour); drop the
    /// out-of-memory exclusion; match only `Engine` (misses the typed digest refusal).
    #[tokio::test]
    async fn only_an_engine_load_failure_is_diagnosed() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let data = tempfile::tempdir().unwrap();
        let bundle = data.path().join("models/mlx/supra2_img");
        std::fs::create_dir_all(&bundle).unwrap();
        let settings = settings_at(data.path());
        let diagnose = |calls: std::sync::Arc<AtomicUsize>| {
            move |_: &str, _: &std::path::Path| {
                calls.fetch_add(1, Ordering::SeqCst);
                Some(InstallFault::Converted {
                    problem: "denoiser.safetensors does not match its manifest".to_owned(),
                })
            }
        };
        for error in [
            WorkerError::InvalidPayload(
                "supra2_img: advanced.faceRestore is not supported (Supra2-IMG takes prompt, \
                 seed, guidance, steps and image count only)"
                    .to_owned(),
            ),
            WorkerError::InvalidPayload(
                "supra2_img needs 3.1 GB but only 1.2 GB of unified memory is available".to_owned(),
            ),
            WorkerError::InvalidPayload(converted_files_message(
                "Supra2-IMG",
                "denoiser.safetensors is 12 bytes, expected 419430400",
            )),
            WorkerError::Engine(
                "candle supra2_img load failed: backend op failed: \
                 DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
                    .to_owned(),
            ),
            WorkerError::Engine(
                "supra2_img load failed: [metal::malloc] Attempting to allocate 9 GB".to_owned(),
            ),
            WorkerError::Engine("supra2 generation failed: backend op failed: nan".to_owned()),
            WorkerError::Canceled("user".to_owned()),
        ] {
            let calls = std::sync::Arc::new(AtomicUsize::new(0));
            let before = error.to_string();
            let after = explain_failed_load_with(
                &settings,
                &payload(&bundle),
                error,
                diagnose(calls.clone()),
            )
            .await;
            assert_eq!(after.to_string(), before, "returned untouched");
            assert_eq!(calls.load(Ordering::SeqCst), 0, "not diagnosed: {before}");
        }
        for error in [
            WorkerError::InvalidPayload(
                "supra2_img load failed: unsupported: supra2: bundle file denoiser.safetensors \
                 does not match its manifest"
                    .to_owned(),
            ),
            WorkerError::Engine(LOAD_FAILURE.to_owned()),
            WorkerError::Engine(
                "candle supra2_img load failed: backend op failed: safetensors header".to_owned(),
            ),
        ] {
            let calls = std::sync::Arc::new(AtomicUsize::new(0));
            let before = error.to_string();
            let after = explain_failed_load_with(
                &settings,
                &payload(&bundle),
                error,
                diagnose(calls.clone()),
            )
            .await;
            assert_eq!(calls.load(Ordering::SeqCst), 1, "diagnosed: {before}");
            assert!(
                after.to_string().contains("run Convert again"),
                "{before} -> {after}"
            );
        }
    }
}
