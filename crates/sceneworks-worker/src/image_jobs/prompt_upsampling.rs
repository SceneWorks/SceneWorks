//! Tensor-neutral FLUX 3 prompt-upsampling composition and provenance. The same functions run
//! in both native production lanes; CPU tests exercise them without linking a CUDA/MLX bundle.

use super::{parse_prompt_enhancement_fields, ImageRequest, Settings, WorkerError, WorkerResult};
use gen_core::flux3::pup::{
    UpsampleMode, MAX_EFFECTIVE_CAPTION_CHARS as PROMPT_ENHANCE_MAX_EFFECTIVE_CHARS,
};
use gen_core::{LoadSpec, WeightsSource};
use serde_json::{json, Value};
const PROMPT_ENHANCE_MAX_REASON_CHARS: usize = 512;

/// Stage FLUX 3's separately installed optional VLM only for an enabled request. This must run
/// before memory admission and cache identity on both native backends: a plain render never
/// requires or pays for it, and an enabled request never silently falls back to a missing model.
pub(super) fn attach_prompt_upsampler(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    if !matches!(engine_id, "flux3_klein" | "flux3_klein_base")
        || !parse_prompt_enhancement_fields(&request.advanced)?.0
    {
        return Ok(spec);
    }
    use gen_core::flux3::pup::{
        UpsamplePackage, UPSAMPLER_COMPONENT, UPSAMPLER_REPO, UPSAMPLER_REVISION,
    };
    let manifest = Value::Object(request.model_manifest_entry.clone());
    let declared = manifest
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|downloads| {
            downloads.iter().find(|download| {
                download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                    && download.get("componentId").and_then(Value::as_str)
                        == Some(UPSAMPLER_COMPONENT)
            })
        });
    if !declared.is_some_and(|download| {
        download.get("repo").and_then(Value::as_str) == Some(UPSAMPLER_REPO)
            && download.get("revision").and_then(Value::as_str) == Some(UPSAMPLER_REVISION)
    }) {
        return Err(WorkerError::InvalidPayload(format!(
            "FLUX 3 enhancement requires the pinned optional {UPSAMPLER_COMPONENT} component: {UPSAMPLER_REPO} @ {UPSAMPLER_REVISION}; refresh the model catalog"
        )));
    }
    let source = crate::model_jobs::resolve_optional_component(&manifest, UPSAMPLER_COMPONENT, settings)
        .ok_or_else(|| WorkerError::InvalidPayload(format!(
            "Install FLUX 3 Prompt Upsampler ({UPSAMPLER_REPO} @ {UPSAMPLER_REVISION}) in Models before enabling Enhance prompt, or turn enhancement off"
        )))?;
    let WeightsSource::Dir(dir) = &source else {
        return Err(WorkerError::InvalidPayload(
            "FLUX 3 prompt_upsampler must be a complete package directory, not a single file"
                .to_owned(),
        ));
    };
    UpsamplePackage::check(dir).map_err(|error| WorkerError::InvalidPayload(format!(
        "FLUX 3 prompt_upsampler is incomplete or incompatible: {error}; reinstall FLUX 3 Prompt Upsampler or turn enhancement off"
    )))?;
    Ok(spec.with_component(UPSAMPLER_COMPONENT, source))
}

pub(super) fn prompt_enhancement_fact(
    report: gen_core::PromptEnhancementReport,
    requested_prompt: &str,
) -> WorkerResult<Value> {
    if report.original_prompt != requested_prompt {
        return Err(WorkerError::Engine(
            "prompt-enhancement report original_prompt did not match the per-image request"
                .to_owned(),
        ));
    }
    if report.effective_prompt.trim().is_empty()
        || report.effective_prompt.chars().count() > PROMPT_ENHANCE_MAX_EFFECTIVE_CHARS
    {
        return Err(WorkerError::Engine(format!(
            "prompt-enhancement report effective_prompt must contain 1..={PROMPT_ENHANCE_MAX_EFFECTIVE_CHARS} characters"
        )));
    }
    let (outcome, fallback_reason) = match report.outcome {
        gen_core::PromptEnhancementOutcome::Enhanced => {
            if report.fallback_reason.is_some() {
                return Err(WorkerError::Engine(
                    "enhanced prompt report unexpectedly carried a fallback reason".to_owned(),
                ));
            }
            // A known PUP mode records that the VLM ran and its answer was accepted; an
            // already-rich prompt can validly be returned unchanged. Keep the legacy single-mode
            // FLUX.2 guard, and never trust arbitrary user-provided mode/report metadata.
            if report.effective_prompt == report.original_prompt
                && report
                    .mode
                    .as_deref()
                    .and_then(UpsampleMode::from_system_key)
                    .is_none()
            {
                return Err(WorkerError::Engine(
                    "enhanced prompt report did not rewrite the original prompt".to_owned(),
                ));
            }
            ("enhanced", None)
        }
        gen_core::PromptEnhancementOutcome::Fallback => {
            if report.effective_prompt != report.original_prompt {
                return Err(WorkerError::Engine(
                    "fallback prompt report did not preserve the original prompt".to_owned(),
                ));
            }
            let reason = report
                .fallback_reason
                .as_deref()
                .map(str::trim)
                .filter(|reason| {
                    !reason.is_empty()
                        && reason.chars().count() <= PROMPT_ENHANCE_MAX_REASON_CHARS
                        && !reason.chars().any(char::is_control)
                })
                .ok_or_else(|| {
                    WorkerError::Engine(
                        "fallback prompt report did not carry a safe bounded reason".to_owned(),
                    )
                })?;
            ("fallback", Some(reason.to_owned()))
        }
        gen_core::PromptEnhancementOutcome::Absent => {
            return Err(WorkerError::Engine(
                "enabled prompt enhancement reported an absent outcome".to_owned(),
            ));
        }
    };
    let mut fact = json!({
        "outcome": outcome,
        "originalPrompt": report.original_prompt,
        "effectivePrompt": report.effective_prompt,
        "fallbackReason": fallback_reason,
    });
    if let Some(mode) = report.mode {
        if !matches!(mode.as_str(), "t2i" | "i2i" | "t2i_bbox" | "i2i_bbox") {
            return Err(WorkerError::Engine(
                "prompt-enhancement report carried an unknown mode".to_owned(),
            ));
        }
        fact["mode"] = Value::String(mode);
    }
    if let Some(rows) = report.locked_annotations {
        if rows.chars().count() > PROMPT_ENHANCE_MAX_EFFECTIVE_CHARS
            || !gen_core::flux3::pup::loads(&rows).is_ok_and(|value| value.as_array().is_some())
        {
            return Err(WorkerError::Engine(
                "prompt-enhancement locked annotations must be a bounded JSON array".to_owned(),
            ));
        }
        // Keep the exact canvas-row text; parsing/reformatting would discard reproducibility bytes.
        fact["lockedAnnotations"] = Value::String(rows);
    }
    Ok(fact)
}

/// Coalesce per-token VLM callbacks before posting progress. The independent heartbeat/cancel
/// interval remains active even between these updates; no diffusion step is synthesized.
pub(super) fn publish_upsampling_progress(tokens: u32, max_tokens: u32) -> bool {
    tokens == 0 || tokens.is_multiple_of(32) || (max_tokens > 0 && tokens >= max_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::flux3::pup::{
        UPSAMPLER_COMPONENT, UPSAMPLER_REPO, UPSAMPLER_REQUIRED_FILES, UPSAMPLER_REVISION,
    };
    use std::path::{Path, PathBuf};

    fn setup() -> (tempfile::TempDir, Settings, crate::test_env::EnvVars) {
        let env = crate::test_env::EnvVars::set(&[
            ("HF_HOME", ""),
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
        ]);
        let root = tempfile::Builder::new()
            .prefix(&format!("sc26238-{}-", std::process::id()))
            .tempdir()
            .unwrap();
        let settings = Settings {
            data_dir: root.path().to_path_buf(),
            ..crate::test_env::offline_settings()
        };
        (root, settings, env)
    }

    fn request(enabled: bool) -> ImageRequest {
        ImageRequest::from_payload(json!({
            "model": "flux3_klein", "mode": "text_to_image", "prompt": "a fox",
            "advanced": { "enhancePrompt": enabled },
            "modelManifestEntry": { "downloads": [{
                "provider": "huggingface", "repo": UPSAMPLER_REPO, "revision": UPSAMPLER_REVISION,
                "coRequisite": true, "componentId": UPSAMPLER_COMPONENT, "required": "soft", "files": []
            }] }
        }).as_object().unwrap())
    }

    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(PathBuf::from("/unused-image-backbone")))
    }

    fn package_shape(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        // File-presence fixture only: never load these marker weights/tokenizer. The inference
        // native miniature tests independently prove that the real package loaders execute.
        for name in UPSAMPLER_REQUIRED_FILES {
            std::fs::write(dir.join(name), b"fixture").unwrap();
        }
        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type":"qwen3_5","vision_config":{}}"#,
        )
        .unwrap();
    }

    #[test]
    fn bypass_and_other_engines_never_require_or_attach_the_optional_model() {
        let (_root, settings, _env) = setup();
        let mut bypass = request(false);
        bypass.model_manifest_entry.clear();
        assert!(
            attach_prompt_upsampler(spec(), "flux3_klein", &bypass, &settings)
                .unwrap()
                .components
                .is_empty()
        );
        assert!(
            attach_prompt_upsampler(spec(), "flux2_dev", &request(true), &settings)
                .unwrap()
                .components
                .is_empty()
        );
    }

    #[test]
    fn enabled_requests_resolve_only_the_frozen_complete_package_shape() {
        let (root, settings, _env) = setup();
        let repo = root
            .path()
            .join("cache/huggingface/hub/models--black-forest-labs--flux-3-image-klein-pup");
        let stale = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        package_shape(&repo.join("snapshots").join(stale));
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), stale).unwrap();
        let req = request(true);
        let missing = attach_prompt_upsampler(spec(), "flux3_klein", &req, &settings)
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains("Install FLUX 3 Prompt Upsampler"),
            "{missing}"
        );
        let pinned = repo.join("snapshots").join(UPSAMPLER_REVISION);
        package_shape(&pinned);
        for engine_id in ["flux3_klein", "flux3_klein_base"] {
            let mut model_request = req.clone();
            model_request.model = engine_id.to_owned();
            let prepared =
                attach_prompt_upsampler(spec(), engine_id, &model_request, &settings).unwrap();
            assert_eq!(prepared.components.len(), 1);
            assert_eq!(prepared.weights, spec().weights);
            let Some(WeightsSource::Dir(attached)) = prepared.components.get(UPSAMPLER_COMPONENT)
            else {
                panic!("component directory missing")
            };
            assert_eq!(
                attached.canonicalize().unwrap(),
                pinned.canonicalize().unwrap()
            );
        }
        std::fs::remove_file(pinned.join("chat_template.jinja")).unwrap();
        let incomplete = attach_prompt_upsampler(spec(), "flux3_klein", &req, &settings)
            .unwrap_err()
            .to_string();
        assert!(
            incomplete.contains("chat_template.jinja") && incomplete.contains("reinstall"),
            "{incomplete}"
        );
        package_shape(&pinned);
        std::fs::write(
            pinned.join("config.json"),
            br#"{"model_type":"llama","vision_config":{}}"#,
        )
        .unwrap();
        assert!(
            attach_prompt_upsampler(spec(), "flux3_klein", &req, &settings)
                .unwrap_err()
                .to_string()
                .contains("incompatible")
        );
        let mut unpinned = request(true);
        unpinned.model_manifest_entry.get_mut("downloads").unwrap()[0]["revision"] = json!("main");
        assert!(
            attach_prompt_upsampler(spec(), "flux3_klein", &unpinned, &settings)
                .unwrap_err()
                .to_string()
                .contains("pinned optional")
        );
    }

    #[test]
    fn caption_metadata_preserves_mode_and_exact_locked_json_without_changing_legacy_facts() {
        let rows = "[ {\"id\":\"obj_1\", \"bbox\":[1,2,3,4], \"desc\":\"locked\"} ]";
        for mode in ["t2i", "i2i", "t2i_bbox", "i2i_bbox"] {
            let report = gen_core::PromptEnhancementReport::enhanced("a fox", "a red fox <obj_1>")
                .with_mode(mode)
                .with_locked_annotations(rows);
            let fact = prompt_enhancement_fact(report, "a fox").unwrap();
            assert_eq!(fact["mode"], mode);
            assert_eq!(fact["lockedAnnotations"], rows);
            assert_eq!(fact["originalPrompt"], "a fox");
            assert_eq!(fact["effectivePrompt"], "a red fox <obj_1>");
        }
        let legacy = prompt_enhancement_fact(
            gen_core::PromptEnhancementReport::enhanced("a fox", "a red fox"),
            "a fox",
        )
        .unwrap();
        assert!(legacy.get("mode").is_none() && legacy.get("lockedAnnotations").is_none());
        assert_eq!(legacy["outcome"], "enhanced");
        for report in [
            gen_core::PromptEnhancementReport::enhanced("another prompt", "a red fox"),
            gen_core::PromptEnhancementReport::enhanced("a fox", "a fox"),
            gen_core::PromptEnhancementReport::absent("a fox"),
            gen_core::PromptEnhancementReport::enhanced("a fox", "a red fox").with_mode("unknown"),
            gen_core::PromptEnhancementReport::enhanced("a fox", "a red fox")
                .with_locked_annotations("{}"),
        ] {
            assert!(prompt_enhancement_fact(report, "a fox").is_err());
        }
        let fallback = prompt_enhancement_fact(
            gen_core::PromptEnhancementReport::fallback("a fox", "reference_aspect_ratio"),
            "a fox",
        )
        .unwrap();
        assert_eq!(fallback["effectivePrompt"], "a fox");
        assert_eq!(fallback["fallbackReason"], "reference_aspect_ratio");
    }

    fn producer_plan(
        locked: Option<Vec<gen_core::flux3::pup::JsonValue>>,
    ) -> gen_core::flux3::pup::UpsamplePlan {
        gen_core::flux3::pup::UpsamplePlan {
            mode: if locked.is_some() {
                UpsampleMode::TextWithBoxes
            } else {
                UpsampleMode::Text
            },
            original_prompt: "already rich".to_owned(),
            system_prompt: String::new(),
            user_text: String::new(),
            locked,
            references: vec![],
            max_new_tokens: gen_core::flux3::pup::MAX_TOKENS,
        }
    }

    fn produce_then_persist(
        plan: &gen_core::flux3::pup::UpsamplePlan,
        raw: &str,
    ) -> WorkerResult<Value> {
        let outcome = plan
            .finish(raw)
            .map_err(|error| WorkerError::InvalidPayload(error.to_string()))?;
        prompt_enhancement_fact(outcome.report, &plan.original_prompt)
    }

    #[test]
    fn actual_pup_producer_unchanged_success_is_accepted_by_production_persistence() {
        let plan = producer_plan(None);
        let outcome = plan.finish("already rich").unwrap();
        assert_eq!(
            outcome.report.outcome,
            gen_core::PromptEnhancementOutcome::Enhanced
        );
        assert_eq!(outcome.report.fallback_reason, None);
        let fact = prompt_enhancement_fact(outcome.report, &plan.original_prompt).unwrap();
        assert_eq!(fact["originalPrompt"], "already rich");
        assert_eq!(fact["effectivePrompt"], "already rich");
        assert_eq!(fact["outcome"], "enhanced");
        assert!(fact["fallbackReason"].is_null());
    }

    #[test]
    fn actual_pup_producer_and_persistence_share_unicode_and_complete_locked_caption_bound() {
        use gen_core::flux3::pup::{self, JsonValue, MAX_EFFECTIVE_CAPTION_CHARS};
        let plan = producer_plan(None);
        let edge = "界".repeat(MAX_EFFECTIVE_CAPTION_CHARS);
        assert_eq!(edge.len(), MAX_EFFECTIVE_CAPTION_CHARS * 3);
        assert_eq!(
            produce_then_persist(&plan, &edge).unwrap()["effectivePrompt"],
            edge
        );
        assert!(produce_then_persist(&plan, &format!("界{edge}"))
            .unwrap_err()
            .to_string()
            .contains("effective caption"));
        let locked = pup::loads(r#"[{"id":"obj_1","bbox":[10,20,30,40],"desc":"locked"}]"#)
            .unwrap()
            .as_array()
            .unwrap()
            .to_vec();
        let baseline = pup::reconcile("scene <obj_1>", &locked).unwrap();
        let padding = MAX_EFFECTIVE_CAPTION_CHARS - baseline.chars().count();
        let raw_edge = format!("{}scene <obj_1>", "界".repeat(padding));
        let plan = producer_plan(Some(locked.clone()));
        let outcome = plan.finish(&raw_edge).unwrap();
        assert_eq!(outcome.caption.chars().count(), MAX_EFFECTIVE_CAPTION_CHARS);
        let rows = pup::dumps(&JsonValue::Array(locked.clone()));
        assert!(outcome.caption.ends_with(&rows));
        let fact = prompt_enhancement_fact(outcome.report, &plan.original_prompt).unwrap();
        assert_eq!(fact["lockedAnnotations"], rows);
        assert!(produce_then_persist(&plan, &format!("界{raw_edge}"))
            .unwrap_err()
            .to_string()
            .contains("effective caption"));
        assert_eq!(
            plan.locked.as_ref().unwrap(),
            &locked,
            "no locked rows were truncated or rewritten"
        );
    }

    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[test]
    fn native_progress_bridge_keeps_upsampling_separate_from_denoise_steps() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        super::super::send_gen_progress(
            &tx,
            3,
            gen_core::Progress::Upsampling {
                tokens: 17,
                max_tokens: 73,
            },
        );
        match rx.try_recv().unwrap() {
            super::super::GenEvent::Upsampling {
                index: 3,
                tokens: 17,
                max_tokens: 73,
            } => {}
            _ => panic!("upsampling must not become a denoise step"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn progress_is_coalesced_without_fabricating_denoise_steps() {
        assert!(publish_upsampling_progress(0, 64));
        assert!(!publish_upsampling_progress(1, 64));
        assert!(!publish_upsampling_progress(31, 64));
        assert!(publish_upsampling_progress(32, 64));
        assert!(!publish_upsampling_progress(33, 64));
        assert!(publish_upsampling_progress(63, 63));
        assert!(publish_upsampling_progress(64, 64));
    }
}
