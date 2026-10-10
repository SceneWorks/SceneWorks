//! Iris-3B generation-task wiring (sc-25679, epic 25678).
//!
//! The `iris_3b` engine (`mlx-gen-iris`) loads the GENERATION task from two resources:
//! `LoadSpec::weights` = the `speridlabs/iris-3b` backbone root (`config.yaml` + `model.safetensors`)
//! and `LoadSpec::components["text_encoder"]` = the pinned `Qwen/Qwen3-VL-4B-Instruct` snapshot. The
//! engine's descriptor advertises no `required_components` (the generic
//! `attach_required_components` seam therefore stages nothing for it), so the generation task's
//! resource set is read from the backend-neutral contract instead:
//! [`gen_core::iris::IrisTask::Generation`] names the text encoder as a resource and
//! [`gen_core::iris::TEXT_ENCODER_COMPONENT`] its component key. The component is resolved
//! cache-only through the same all-or-nothing `coRequisite` seam every other split model uses, so a
//! job never downloads at generation time and a missing encoder is a typed failure before load.
//!
//! The depth and restoration tasks (later stories) use [`IrisTask::uses_text_encoder`] = false and
//! never reach this staging.

use gen_core::iris::{IrisTask, FAMILY, GENERATION_MODEL_ID, TEXT_ENCODER_COMPONENT};
use gen_core::{LoadSpec, WeightsSource};
use sceneworks_core::contracts::JsonObject;
use serde_json::Value;

use crate::error::WorkerResult;
use crate::settings::Settings;

/// Is this registered engine id the Iris-3B generation route?
pub(crate) fn is_iris_generation_engine(engine_id: &str) -> bool {
    engine_id == GENERATION_MODEL_ID
}

/// The component keys the Iris generation task stages besides its backbone, from the shared
/// task contract (E4: only generation conditions on text).
fn generation_components() -> &'static [&'static str] {
    if IrisTask::Generation.uses_text_encoder() {
        &[TEXT_ENCODER_COMPONENT]
    } else {
        &[]
    }
}

/// Resolve the generation task's installed components (cache-only) for `model_id`'s catalog entry.
/// All-or-nothing: a missing or torn co-requisite is a typed `InvalidPayload` naming the component
/// and the Model Manager repair path, raised before any weights load.
pub(crate) fn resolve_generation_components(
    model_id: &str,
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<Vec<(String, WeightsSource)>> {
    let descriptor = gen_core::ModelDescriptor {
        id: GENERATION_MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: gen_core::Modality::Image,
        capabilities: gen_core::Capabilities::default(),
        encoder_contract: None,
        denoiser_output_latent_space: None,
        required_components: generation_components(),
        control_kinds: None,
    };
    let manifest = Value::Object(manifest_entry.clone());
    let components = crate::model_jobs::resolve_co_requisites(&descriptor, &manifest, settings)
        .map_err(|error| match error {
            crate::error::WorkerError::InvalidPayload(message) => {
                crate::error::WorkerError::InvalidPayload(message.replacen(
                    GENERATION_MODEL_ID,
                    model_id,
                    1,
                ))
            }
            other => other,
        })?;
    Ok(components.into_iter().collect())
}

/// Stage the Iris generation task's components onto its `LoadSpec`. A no-op for every other
/// engine, and for a component the generic descriptor seam already staged.
pub(crate) fn attach_iris_generation_components(
    spec: LoadSpec,
    engine_id: &str,
    model_id: &str,
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    if !is_iris_generation_engine(engine_id) {
        return Ok(spec);
    }
    let components = resolve_generation_components(model_id, manifest_entry, settings)?;
    Ok(components.into_iter().fold(spec, |spec, (id, source)| {
        if spec.components.contains_key(&id) {
            spec
        } else {
            spec.with_component(id, source)
        }
    }))
}

/// The manifest's `image.negativePromptRequiresGuidance` declaration: the negative prompt is the CFG
/// unconditional, so it only exists while guidance is on. ABSENT MEANS FALSE.
pub(crate) fn negative_prompt_requires_guidance(manifest_entry: &JsonObject) -> bool {
    manifest_entry
        .get("image")
        .and_then(|image| image.get("negativePromptRequiresGuidance"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Drop the negative prompt when the model declares it CFG-bound and the resolved guidance turns CFG
/// off (exactly 1.0, the engine's `uses_cfg` test). The engine refuses a non-empty negative prompt
/// there rather than ignoring it, and with CFG off it has no effect on the image either way.
pub(crate) fn gate_negative_prompt_on_guidance(
    manifest_entry: &JsonObject,
    guidance: Option<f32>,
    negative_prompt: Option<String>,
) -> Option<String> {
    if negative_prompt_requires_guidance(manifest_entry) && guidance == Some(1.0) {
        None
    } else {
        negative_prompt
    }
}

/// Iris has no scheduler axis: the engine refuses `scheduler_shift` by name. A stale or
/// API-supplied `schedulerShift` therefore never reaches it (the Image Studio hides the scheduler
/// controls for Iris, so a normal submit never carries one).
pub(crate) fn honored_scheduler_shift(
    engine_id: &str,
    scheduler_shift: Option<f32>,
) -> Option<f32> {
    if is_iris_generation_engine(engine_id) {
        None
    } else {
        scheduler_shift
    }
}

/// The Iris text-encoder directory a resolved spec carries, for diagnostics/tests.
#[cfg(test)]
pub(crate) fn staged_text_encoder_dir(spec: &LoadSpec) -> Option<&std::path::Path> {
    match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(dir)) => Some(dir.as_path()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    const TE_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
    const TE_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

    fn settings_at(data_dir: &Path) -> Settings {
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data_dir.to_path_buf();
        settings
    }

    /// Neutralize the ambient HF cache env so the cache resolves under the temp data dir.
    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    /// The shipped catalog entry, so the test exercises the real `coRequisite` declaration.
    fn shipped_iris_entry() -> JsonObject {
        let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text))
                .expect("parse builtin.models.jsonc");
        manifest
            .get("models")
            .and_then(Value::as_array)
            .and_then(|models| {
                models
                    .iter()
                    .find(|model| model.get("id").and_then(Value::as_str) == Some("iris_3b"))
            })
            .and_then(Value::as_object)
            .cloned()
            .expect("iris_3b is in the builtin catalog")
    }

    fn stage(root: &Path, repo: &str, revision: &str, file: &str) {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    fn stage_text_encoder(root: &Path) {
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ] {
            stage(root, TE_REPO, TE_REVISION, file);
        }
    }

    #[test]
    fn the_contract_names_only_the_text_encoder_for_generation() {
        assert_eq!(generation_components(), &["text_encoder"]);
        assert!(!IrisTask::Depth.uses_text_encoder());
        assert!(!IrisTask::Restoration.uses_text_encoder());
    }

    #[test]
    fn generation_stages_the_installed_text_encoder_snapshot() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        stage_text_encoder(data.path());
        let settings = settings_at(data.path());
        let backbone = data.path().join("backbone");
        let spec = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(backbone.clone())),
            "iris_3b",
            "iris_3b",
            &shipped_iris_entry(),
            &settings,
        )
        .expect("an installed encoder stages");
        let staged = staged_text_encoder_dir(&spec).expect("text_encoder staged as a Dir");
        assert!(
            staged.ends_with(format!("snapshots/{TE_REVISION}")),
            "{staged:?}"
        );
        assert_eq!(
            spec.components.len(),
            1,
            "only the generation task's encoder is staged"
        );
        assert_eq!(spec.weights, WeightsSource::Dir(backbone));
    }

    #[test]
    fn a_missing_text_encoder_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let settings = settings_at(data.path());
        let error = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            "iris_3b",
            &shipped_iris_entry(),
            &settings,
        )
        .expect_err("no encoder installed");
        let message = error.to_string();
        assert!(
            message.contains("text_encoder") && message.contains("not installed"),
            "{message}"
        );
    }

    #[test]
    fn a_torn_text_encoder_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        // Everything but the shards.
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            stage(data.path(), TE_REPO, TE_REVISION, file);
        }
        let error = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            "iris_3b",
            &shipped_iris_entry(),
            &settings_at(data.path()),
        )
        .expect_err("torn encoder");
        assert!(error.to_string().contains("missing"), "{error}");
    }

    #[test]
    fn other_engines_are_untouched() {
        let data = tempfile::tempdir().unwrap();
        let spec = attach_iris_generation_components(
            LoadSpec::new(WeightsSource::Dir(data.path().to_path_buf())),
            "mage_flow",
            "mage_flow",
            &JsonObject::new(),
            &settings_at(data.path()),
        )
        .expect("no-op");
        assert!(spec.components.is_empty());
    }

    #[test]
    fn negative_prompt_is_dropped_only_when_cfg_is_off_for_a_declaring_model() {
        let iris = shipped_iris_entry();
        assert!(negative_prompt_requires_guidance(&iris));
        let negative = || Some("blurry".to_owned());
        assert_eq!(
            gate_negative_prompt_on_guidance(&iris, Some(1.0), negative()),
            None
        );
        assert_eq!(
            gate_negative_prompt_on_guidance(&iris, Some(3.0), negative()),
            negative()
        );
        let undeclared = json!({ "id": "other" }).as_object().unwrap().clone();
        assert_eq!(
            gate_negative_prompt_on_guidance(&undeclared, Some(1.0), negative()),
            negative()
        );
    }

    #[test]
    fn schedule_shift_never_reaches_iris() {
        assert_eq!(honored_scheduler_shift("iris_3b", Some(3.0)), None);
        assert_eq!(
            honored_scheduler_shift("sensenova_u1_8b", Some(3.0)),
            Some(3.0)
        );
    }

    /// The generic lane derives what it forwards from the linked descriptor: Iris advertises no
    /// quant tier (so `mlx_candidate_quant` never sets `LoadSpec::quantize`), no sampler/scheduler
    /// names (so `normalize_sampling_knob` drops any requested one), and the catalog row offers none
    /// of those controls, so the request carries exactly the engine-honored surface.
    #[test]
    fn the_linked_descriptor_and_catalog_expose_only_honored_controls() {
        let model = crate::engines::mlx_model("iris_3b").expect("iris_3b is linked on macOS");
        assert_eq!(model.engine_id(), "iris_3b");
        assert!(!model.supports_quant());
        assert!(model.supports_guidance() && model.supports_negative_prompt());
        let caps = &model.descriptor.capabilities;
        assert!(caps.samplers.is_empty() && caps.schedulers.is_empty());
        assert!(!caps.supports_lora && !caps.supports_lokr && !caps.supports_true_cfg);
        assert_eq!(model.default_steps(), gen_core::iris::DEFAULT_STEPS);
        assert_eq!(model.default_guidance(), gen_core::iris::DEFAULT_CFG_SCALE);

        let entry = shipped_iris_entry();
        let limits = entry.get("limits").and_then(Value::as_object).unwrap();
        for absent in ["samplers", "schedulers", "guidanceMethods"] {
            assert!(!limits.contains_key(absent), "iris must not offer {absent}");
        }
        assert!(entry
            .get("mlx")
            .and_then(|mlx| mlx.get("quantize"))
            .is_none());
        assert_eq!(
            entry["loraCompatibility"]["families"]
                .as_array()
                .map(Vec::len),
            Some(0)
        );
        assert_eq!(
            limits["requiresDimensionsMultipleOf"].as_u64(),
            Some(u64::from(gen_core::iris::SIZE_MULTIPLE))
        );
        assert_eq!(
            limits["maxDimension"].as_u64(),
            Some(u64::from(caps.max_size))
        );
        assert_eq!(
            limits["minDimension"].as_u64(),
            Some(u64::from(caps.min_size))
        );
        assert_eq!(
            entry["defaults"]["steps"].as_u64(),
            Some(u64::from(gen_core::iris::DEFAULT_STEPS))
        );
    }

    /// Iris's release default (100 steps) is above the generic lane's historical 80-step ceiling;
    /// the ceiling rises to the model's own default so it is never clamped away.
    #[test]
    fn the_steps_ceiling_admits_the_models_own_default() {
        assert_eq!(
            super::super::max_requested_steps(gen_core::iris::DEFAULT_STEPS),
            gen_core::iris::DEFAULT_STEPS
        );
        assert_eq!(super::super::max_requested_steps(40), 80);
    }
}
