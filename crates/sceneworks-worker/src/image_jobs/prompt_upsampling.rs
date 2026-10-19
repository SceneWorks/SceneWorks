//! Tensor-neutral FLUX 3 prompt-upsampling composition and provenance. The same functions run
//! in both native production lanes; CPU tests exercise them without linking a CUDA/MLX bundle.

use super::{parse_prompt_enhancement_fields, ImageRequest, Settings, WorkerError, WorkerResult};
use gen_core::flux3::pup::{
    UpsampleMode, MAX_EFFECTIVE_CAPTION_CHARS as PROMPT_ENHANCE_MAX_EFFECTIVE_CHARS,
};
use gen_core::{LoadSpec, WeightsSource};
use serde_json::{json, Value};
const PROMPT_ENHANCE_MAX_REASON_CHARS: usize = 512;

/// Optional prompt-enhancement settings resolved from a job request's `advanced` block and threaded
/// into a [`gen_core::GenerationRequest`] (sc-6135). Mirrors the LTX-2.3 video path (`advanced.enhancePrompt` /
/// `enhanceTemperature` / `enhanceMaxTokens`). FLUX.2-dev owns its integrated Mistral3 enhancer;
/// FLUX 3 Image Klein owns a separately provisioned optional Qwen3.5 VLM (sc-26238). API/worker
/// route checks fail closed for other engines; manifest `ui.promptEnhance` controls the existing
/// Image Studio toggle. Disabled requests do not attach the optional FLUX 3 component.

#[derive(Clone, Default)]
pub(crate) struct PromptEnhance {
    pub(super) enabled: bool,
    pub(super) temperature: Option<f32>,
    pub(super) max_tokens: Option<u32>,
}

impl PromptEnhance {
    /// Resolve from a job request's `advanced` settings without truthy/string coercions. These
    /// values drive a native LLM sampler and therefore have one typed, bounded contract shared with
    /// the API and inference providers.
    pub(crate) fn from_advanced(
        model_id: &str,
        advanced: &serde_json::Map<String, Value>,
    ) -> WorkerResult<Self> {
        let (enabled, temperature, max_tokens) =
            parse_prompt_enhancement_fields(model_id, advanced)?;
        Ok(Self {
            enabled,
            temperature,
            max_tokens,
        })
    }

    /// Write the resolved enhancement settings onto a `GenerationRequest`.
    pub(super) fn apply(
        &self,
        request: &mut gen_core::GenerationRequest,
        prompt_enhancement: gen_core::PromptEnhancementSink,
    ) {
        request.enhance_prompt = self.enabled;
        request.enhance_temperature = self.temperature;
        request.enhance_max_tokens = self.max_tokens;
        request.prompt_enhancement = if self.enabled {
            prompt_enhancement
        } else {
            gen_core::PromptEnhancementSink::default()
        };
    }
}

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
        || !parse_prompt_enhancement_fields(engine_id, &request.advanced)?.0
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
            model_request
                .advanced
                .insert("enhanceMaxTokens".to_owned(), json!(3_584));
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

    #[test]
    fn request_token_limits_share_sdk_model_policy_without_clamping() {
        for (model, limit) in [
            ("flux2_dev", 2_048),
            ("ltx_2_3", 2_048),
            ("flux3_klein_alias", 2_048),
            ("flux3_klein", 3_584),
            ("flux3_klein_base", 3_584),
        ] {
            assert_eq!(crate::prompt_enhancement_max_tokens(model), limit);
            assert_eq!(
                u64::from(gen_core::generator::max_enhance_tokens_for_model(model)),
                limit
            );
            for tokens in [0, 2_048, 2_049, 3_584, 3_585] {
                let advanced = json!({"enhancePrompt": true, "enhanceMaxTokens": tokens});
                let parsed = parse_prompt_enhancement_fields(model, advanced.as_object().unwrap());
                assert_eq!(
                    parsed.is_ok(),
                    tokens > 0 && tokens <= limit,
                    "{model}: {tokens}"
                );
                if let Ok((enabled, _, max_tokens)) = parsed {
                    assert!(enabled);
                    assert_eq!(
                        max_tokens,
                        Some(tokens as u32),
                        "never silently clamp a request"
                    );
                }
            }
            for enabled in [false, true] {
                let advanced = json!({"enhancePrompt": enabled});
                let parsed =
                    parse_prompt_enhancement_fields(model, advanced.as_object().unwrap()).unwrap();
                assert_eq!(parsed, (enabled, None, None));
            }
            let disabled = json!({"enhancePrompt": false, "enhanceMaxTokens": limit});
            assert!(parse_prompt_enhancement_fields(model, disabled.as_object().unwrap()).is_err());
        }
    }

    #[test]
    fn raw_absent_tuning_reaches_native_options_and_frozen_pup_budget() {
        use gen_core::flux3::pup::{self, UpsampleDecision, UpsampleOptions, UpsamplerInput};
        let root = tempfile::Builder::new()
            .prefix(&format!("sc26238-budget-{}-", std::process::id()))
            .tempdir()
            .unwrap();
        package_shape(root.path());
        // A two-token CPU-only tokenizer/template fixture exercises real package budgeting,
        // not model execution. The frozen contract suite separately tests full chat templates.
        std::fs::write(root.path().join("tokenizer.json"), serde_json::to_vec(&json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "budget": 1}, "unk_token": "[UNK]"}
        })).unwrap()).unwrap();
        std::fs::write(root.path().join("chat_template.jinja"), "budget").unwrap();
        let prompts = pup::SystemPrompts::builtin();
        std::fs::write(
            root.path().join("system_prompts.json"),
            serde_json::to_vec(&json!({
                "t2i": prompts.t2i, "i2i": prompts.i2i,
                "t2i_bbox": prompts.t2i_bbox, "i2i_bbox": prompts.i2i_bbox
            }))
            .unwrap(),
        )
        .unwrap();
        let package = pup::UpsamplePackage::open(root.path()).unwrap();
        for model in ["flux3_klein", "flux3_klein_base", "flux2_dev"] {
            for enabled in [false, true] {
                let advanced = json!({"enhancePrompt": enabled});
                let parsed =
                    PromptEnhance::from_advanced(model, advanced.as_object().unwrap()).unwrap();
                let mut native = gen_core::GenerationRequest {
                    prompt: "as typed".to_owned(),
                    ..Default::default()
                };
                parsed.apply(&mut native, gen_core::PromptEnhancementSink::default());
                let options = UpsampleOptions::from_generation_request(&native);
                assert_eq!(
                    options.max_tokens, None,
                    "{model}: preserve the model's absent default"
                );
                assert_eq!(options.temperature, None);
                assert_eq!(native.enhance_prompt, enabled);
                if model == "flux2_dev" {
                    continue;
                } // Its untouched native default is 512, not PUP.
                let input = UpsamplerInput {
                    prompt: native.prompt.clone(),
                    references: vec![],
                    boxes: None,
                    mode: UpsampleMode::Text,
                    bypass: !native.enhance_prompt,
                };
                match pup::decide(&input, &package, &options).unwrap() {
                    UpsampleDecision::Run(plan) => assert_eq!(plan.max_new_tokens, 3_584),
                    UpsampleDecision::Bypass(outcome) => assert_eq!(outcome.caption, "as typed"),
                }
                if enabled {
                    for max_tokens in [2, 3_584] {
                        let advanced =
                            json!({"enhancePrompt": true, "enhanceMaxTokens": max_tokens});
                        PromptEnhance::from_advanced(model, advanced.as_object().unwrap())
                            .unwrap()
                            .apply(&mut native, gen_core::PromptEnhancementSink::default());
                        let options = UpsampleOptions::from_generation_request(&native);
                        assert_eq!(options.max_tokens, Some(max_tokens));
                        let UpsampleDecision::Run(plan) =
                            pup::decide(&input, &package, &options).unwrap()
                        else {
                            panic!("enabled explicit request must run");
                        };
                        assert_eq!(plan.max_new_tokens, max_tokens);
                    }
                }
            }
        }
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

    // Production image consumer + HTTP API seam, with tensor-free provider events.
    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flux3_saved_provenance_and_progress_use_only_the_per_image_provider_report() {
        use crate::image_jobs::*;
        for (model, backend) in [("flux3_klein", "candle"), ("flux3_klein_base", "mlx")] {
            for mode in ["t2i", "i2i", "t2i_bbox", "i2i_bbox"] {
                let dir = tempfile::tempdir().unwrap();
                let project = dir.path().join("project");
                std::fs::create_dir_all(project.join("assets/images")).unwrap();
                let payload = json!({"projectId":"p", "model":model, "mode":"text_to_image",
                    "prompt":"a fox", "width":64, "height":64, "seed":17,
                    "advanced":{"enhancePrompt":true, "promptEnhancement":{"effectivePrompt":"forged"}}});
                let request = ImageRequest::from_payload(payload.as_object().unwrap());
                let plan = ImagePlan::with_count(&request, 1, None);
                let (url, posts) = crate::tests::spawn_no_user_cancel_capture_stub().await;
                let settings = crate::tests::test_settings(url, None);
                let api = ApiClient::new(&settings);
                let job: JobSnapshot =
                    serde_json::from_value(crate::tests::job_snapshot_json("job-1", false))
                        .unwrap();
                let rows = "[ {\"id\":\"obj_1\", \"bbox\":[1,2,3,4], \"desc\":\"locked\"} ]";
                let report =
                    gen_core::PromptEnhancementReport::enhanced("a fox", "a fox in snow <obj_1>")
                        .with_mode(mode)
                        .with_locked_annotations(rows);
                let (tx, rx) = tokio::sync::mpsc::channel(8);
                let blocking = tokio::task::spawn_blocking(move || {
                    for event in [
                        GenEvent::Upsampling {
                            index: 0,
                            tokens: 32,
                            max_tokens: 73,
                        },
                        GenEvent::PromptEnhancement {
                            index: 0,
                            expected_prompt: "a fox".to_owned(),
                            report,
                        },
                        GenEvent::Step {
                            index: 0,
                            current: 1,
                            total: 16,
                        },
                        GenEvent::Decoding { index: 0 },
                        GenEvent::Image {
                            index: 0,
                            seed: 17,
                            width: 64,
                            height: 64,
                            pixels: vec![128; 64 * 64 * 3],
                            face_likeness: None,
                        },
                    ] {
                        tx.blocking_send(event).unwrap();
                    }
                    Ok(())
                });
                let mut assets = Vec::new();
                consume_gen_events(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    &project,
                    backend,
                    "flux3",
                    &request.advanced,
                    1,
                    rx,
                    CancelFlag::new(),
                    blocking,
                    &mut assets,
                )
                .await
                .unwrap();
                assert_eq!(assets.len(), 1);
                let fact = &assets[0]["rawAdapterSettings"]["promptEnhancement"];
                assert_eq!(fact["originalPrompt"], "a fox");
                assert_eq!(fact["effectivePrompt"], "a fox in snow <obj_1>");
                assert_eq!(fact["mode"], mode);
                assert_eq!(fact["lockedAnnotations"], rows);
                let posts = posts.lock().unwrap();
                for detail in ["enhancing prompt (32/73 tokens)", "step 1/16", "decoding"] {
                    assert!(
                        posts.iter().any(|post| post["message"]
                            .as_str()
                            .is_some_and(|message| message.contains(detail))),
                        "missing {detail}: {posts:?}"
                    );
                }
            }
        }
    }

    #[cfg(any(target_os = "macos", feature = "backend-candle"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flux3_prefill_cancellation_stops_before_denoising_or_asset_writes() {
        use crate::image_jobs::*;
        let dir = tempfile::tempdir().unwrap();
        let payload = json!({"projectId":"p", "model":"flux3_klein", "mode":"text_to_image",
            "prompt":"a fox", "advanced":{"enhancePrompt":true}});
        let request = ImageRequest::from_payload(payload.as_object().unwrap());
        let plan = ImagePlan::with_count(&request, 1, None);
        let (url, posts) = crate::tests::spawn_analysis_cancel_stub().await;
        let mut settings = crate::tests::test_settings(url, None);
        settings.heartbeat_seconds = 5;
        let api = ApiClient::new(&settings);
        let job: JobSnapshot =
            serde_json::from_value(crate::tests::job_snapshot_json("job-1", false)).unwrap();
        let cancel = CancelFlag::new();
        let producer_cancel = cancel.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let blocking = tokio::spawn(async move {
            tx.send(GenEvent::Upsampling {
                index: 0,
                tokens: 0,
                max_tokens: 73,
            })
            .await
            .unwrap();
            // Quiet prefill: no denoise step is necessary to honor the API cancel.
            while !producer_cancel.is_cancelled() {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            drop(tx);
            Ok(())
        });
        let mut assets = Vec::new();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(8),
            consume_gen_events(
                &api,
                &settings,
                &job,
                &plan,
                dir.path(),
                "candle",
                "flux3",
                &request.advanced,
                1,
                rx,
                cancel.clone(),
                blocking,
                &mut assets,
            ),
        )
        .await
        .unwrap();
        assert!(matches!(result, Err(WorkerError::Canceled(_))));
        assert!(cancel.is_cancelled());
        assert!(assets.is_empty());
        let posts = posts.lock().unwrap();
        assert!(posts.iter().any(|post| post["status"] == "canceled"));
        assert!(!posts.iter().any(|post| post["message"]
            .as_str()
            .is_some_and(|message| message.contains("step 1/"))));
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
