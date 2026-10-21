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
//! hashed by SceneWorks: only a failed job pays for the diagnosis.

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

/// After a FAILED image job: when the model is convert-at-install with an engine pin contract,
/// re-verify its installed files in full and, if one is at fault, return an actionable error in
/// place of `error` (recording a damaged converted bundle for the catalog). Every other failure —
/// and every other model — is returned unchanged.
pub(crate) async fn explain_failed_load(
    settings: &Settings,
    payload: &JsonObject,
    error: WorkerError,
) -> WorkerError {
    let data_dir = settings.data_dir.clone();
    explain_failed_load_with(payload, error, move |converter, bundle| {
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
    payload: &JsonObject,
    error: WorkerError,
    diagnose: impl FnOnce(&str, &std::path::Path) -> Option<InstallFault> + Send + 'static,
) -> WorkerError {
    if matches!(error, WorkerError::Canceled(_)) {
        return error;
    }
    let Some((name, converter, bundle)) = contract_install(payload) else {
        return error;
    };
    let diagnosis = tokio::task::spawn_blocking(move || {
        let fault = diagnose(&converter, &bundle);
        if let Some(InstallFault::Converted { problem }) = &fault {
            // Best effort: the actionable message below stands even if the record cannot be
            // written (a read-only bundle dir); the next failed load diagnoses again.
            let _ = converted_artifacts::record_verification_failure(&bundle, problem);
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

/// `(display name, converter, converted dir)` for a job whose model is convert-at-install with an
/// engine pin contract and an injected `modelPath`.
fn contract_install(payload: &JsonObject) -> Option<(String, String, PathBuf)> {
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
    let bundle = entry
        .get("modelPath")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())?;
    let model_id = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Some((
        model_display_name(entry, model_id).to_owned(),
        converter.to_owned(),
        PathBuf::from(bundle),
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
    fn stage_same_size_component(data_dir: &std::path::Path, pin: gen_core::supra2::ComponentPin) {
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
        stage_same_size_component(data.path(), gen_core::supra2::FLAN_T5_BASE);
        stage_same_size_component(data.path(), gen_core::supra2::SD_VAE);
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

    /// Tripwire: the production image-job entry point (whose dispatch needs a live API and GPU)
    /// routes every result through the explanation seam the tests above exercise.
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
            Err(WorkerError::Engine("raw".to_owned())),
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
        let untouched =
            explain_failed_load(&settings, &other, WorkerError::Engine("raw".to_owned())).await;
        assert_eq!(untouched.to_string(), "raw");

        let mut no_path = payload(&bundle);
        no_path["modelManifestEntry"]
            .as_object_mut()
            .unwrap()
            .remove("modelPath");
        let untouched =
            explain_failed_load(&settings, &no_path, WorkerError::Engine("raw".to_owned())).await;
        assert_eq!(untouched.to_string(), "raw");
    }
}
