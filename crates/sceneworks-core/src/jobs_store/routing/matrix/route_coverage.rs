//! PIPE-18 reconciliation, not a capability authority or an execution/fit claim.
//!
//! The old side is the independently published product matrix. The candidate side is evaluated
//! through the production predicates, then joined to the platform snapshots and bespoke selectors.
//! Keep the old side fixed during a mutation: deriving both sides from the candidate erases losses.
use super::*;

const LEGACY_MODEL_DIGEST: &str =
    "105abeb6caea45068044f0d4b6e6fcf8a6bcf58f003e81de7c016514fa454c82";
const DISPATCH: &str = include_str!("../../../../../sceneworks-worker/src/utility_dispatch.rs");
const LEGACY: &str = include_str!("../../../../../../config/backend-capabilities/matrix.json");

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteCoverage {
    pub schema_version: u32,
    pub evidence_kind: &'static str,
    pub sources: BTreeMap<String, String>,
    pub jobs: Vec<JobLifecycle>,
    pub cells: Vec<RouteCell>,
    pub registries: Vec<RegistrySurface>,
    pub profile_source_receipt: Value,
    pub model_aliases: Vec<Value>,
    pub direct_route_witnesses: Vec<Value>,
    pub imported_families: Vec<ImportedFamilyCapabilityRow>,
    pub imported_bindings: Vec<Value>,
    pub trainer_mappings: Vec<Value>,
    pub training_kernels: Vec<TrainingCapabilityRow>,
    pub missing: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteCell {
    pub platform: String,
    pub backend: String,
    pub model: String,
    pub origin: &'static str,
    pub canonical_request: Option<Value>,
    pub axis: &'static str,
    pub mode: String,
    pub conditioning: String,
    pub output: String,
    pub output_variants: Vec<String>,
    pub providers: Vec<String>,
    pub expected_provider_ids: Vec<String>,
    pub owner: &'static str,
    pub legacy_supported: Option<bool>,
    pub candidate_supported: Option<bool>,
    pub entry_point: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegistrySurface {
    pub platform: String,
    pub backend: String,
    pub surface: String,
    pub provider: String,
    pub descriptor: Option<Value>,
    pub owner: &'static str,
    pub public_cells: Vec<String>,
    pub disposition: &'static str,
    pub entry_point: String,
    pub callable_entry: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobLifecycle {
    pub job_type: String,
    pub lifecycle: &'static str,
    pub output: &'static str,
    pub owner: &'static str,
    pub entry_point: &'static str,
}

fn registry_owner(
    surface: &str,
    provider: &str,
    snapshot: &serde_json::Map<String, Value>,
) -> Result<&'static str, String> {
    if surface.starts_with("audio_") {
        return Ok("ENG-14");
    }
    Ok(match surface {
        "text_llm_ids" if provider.contains("starvector") => "ENG-7/APP-5",
        "text_llm_ids" => "ENG-14",
        "trainer_ids" => "ENG-8/ENG-9",
        "captioner_ids" | "image_embedder_ids" | "text_embedder_ids" => "ENG-11",
        "transform_ids" => "ENG-6/ENG-10",
        "snapshot_preparer_backends" => "ENG-9",
        "generator_ids" => {
            let family = snapshot["generator_capabilities"]
                .as_array()
                .and_then(|ds| ds.iter().find(|d| d["id"] == provider))
                .and_then(|d| d["family"].as_str())
                .ok_or_else(|| format!("generator {provider} has no descriptor family"))?;
            match family {
                "flux" | "pulid" => "ENG-1a",
                "chroma" | "flux2" | "lens" | "anima" | "sd3" | "stable-diffusion-3" | "sana"
                | "boogu" | "qwen-image" | "qwen-image-2-1" | "z-image" | "kolors" | "sdxl" => {
                    "ENG-2/ENG-5"
                }
                "wan" | "scail2" | "svd" | "mochi" => "ENG-3/ENG-5",
                "krea_2" => "ENG-4b",
                "flux3" | "moss_image" | "iris" | "ideogram" | "supra2" => "ENG-5",
                "ltx" | "minimax_h3" | "mage_flow" | "sensenova-u1" | "bernini"
                | "krea_realtime" | "seedvr2" => "ENG-6",
                other => return Err(format!("unowned native generator {provider}/{other}")),
            }
        }
        _ => return Err(format!("unowned registry surface {surface}")),
    })
}

fn registry_entry(platform: &str, surface: &str, provider: &str) -> String {
    let (lane, load, run) = match surface {
        "generator_ids" => ("media()", "load", "generate(GenerationRequest)"),
        "transform_ids" => ("media()", "load_transform", "transform(TransformRequest)"),
        "trainer_ids" => ("media()", "load_trainer", "train(TrainingRequest)"),
        "captioner_ids" => ("media()", "load_captioner", "caption(CaptionRequest)"),
        "image_embedder_ids" => ("media()", "load_image_embedder", "embed_image"),
        "text_embedder_ids" => ("media()", "load_text_embedder", "embed_text"),
        "text_llm_ids" => ("text()", "load_textllm", "generate/stream"),
        "audio_generator_ids" => ("audio()", "load", "generate(GenerationRequest)"),
        "audio_transform_ids" => ("audio()", "load_audio_transform", "transform"),
        "audio_voice_embedder_ids" => ("audio()", "load_voice_embedder", "embed"),
        "audio_transcriber_ids" => ("audio()", "load_transcriber", "transcribe"),
        "audio_embedder_ids" => ("audio()", "load_audio_embedder", "embed"),
        "snapshot_preparer_backends" => ("preparers()", "prepare_snapshot", "snapshot preparation"),
        "audio_snapshot_preparer_backends" => (
            "audio_preparers()",
            "prepare_snapshot",
            "snapshot preparation",
        ),
        _ => return format!("unclassified registry surface: {surface}"),
    };
    format!("runtime-{platform}::catalog().{lane}.{load}({provider}) -> {run}")
}

fn lifecycle(job: JobType) -> Result<JobLifecycle, String> {
    use JobType::*;
    let (kind, output, owner) = match &job {
        ImageGenerate | ImageEdit | ImageDetail | ImageUpscale => ("image", "image", "APP-5"),
        VectorGenerate => (
            "vector-token",
            "canonical-svg+derived-preview+terminal-metadata",
            "APP-5",
        ),
        ImageVqa => ("understanding", "text", "APP-6"),
        ImageInterleave => (
            "conditional-generation",
            "ordered-text/image-document",
            "APP-6",
        ),
        VideoGenerate | VideoExtend | VideoBridge | PersonReplace | VideoUpscale => {
            ("video", "video+optional-audio", "APP-6")
        }
        AudioGenerate => ("audio", "audio", "ENG-14/APP-6"),
        LoraTrain | ControlTraining => (
            "training",
            "adapter/checkpoint+training-metadata",
            "ENG-8/APP-6",
        ),
        ModelDownload | LoraDownload => ("installer", "installed-model-metadata", "APP-6"),
        ModelImport | LoraImport | ModelConvert => {
            ("import/conversion", "model/adapter-metadata", "APP-6")
        }
        DatasetParquetImport => ("io", "dataset-items", "APP-6"),
        FrameExtract => ("media-utility", "image-assets", "APP-6"),
        TimelineExport => ("media-utility", "video", "APP-6"),
        DatasetUpscale => ("image-utility", "dataset-images", "APP-6"),
        DatasetSubjectMask | ImageSegment => ("vision-utility", "mask", "APP-6"),
        ImageDepth => ("vision-utility", "float32-npy-depth+preview", "APP-6"),
        PersonDetect | PersonTrack | PoseDetect | KpsExtract | DatasetAnalysis
        | CatalogAnalysis | DatasetFaceAnalysis | FaceLikenessCompare => {
            ("vision-utility", "structured-metadata", "APP-6")
        }
        TrainingCaption | PromptRefine => ("language-utility", "text+metadata", "APP-6"),
        Placeholder => ("utility", "placeholder-result", "APP-6"),
        Unknown(name) => return Err(format!("unclassified job type {name}")),
    };
    Ok(JobLifecycle {
        job_type: job.as_str().to_owned(),
        lifecycle: kind,
        output,
        owner,
        entry_point: "crates/sceneworks-worker/src/utility_dispatch.rs::dispatch_utility_job",
    })
}

fn owner(model: &ModelCapabilityRow) -> Result<&'static str, String> {
    Ok(match model.model_type.as_str() {
        "vector" => "ENG-7/APP-5",
        "audio" => "ENG-14/APP-6",
        "utility" => "ENG-7/APP-6",
        "image" | "video" => match model.family.as_str() {
            "flux" => "ENG-1a/APP-5",
            "chroma" | "flux2-dev" | "flux2-klein" | "lens" | "anima" | "sd3" | "sana"
            | "boogu" | "qwen-image" | "qwen-image-2-1" | "z-image" | "kolors" | "sdxl" => {
                "ENG-2/ENG-5/APP-5"
            }
            "wan-video" | "scail2" | "svd" | "mochi" => "ENG-3/ENG-5/APP-6",
            "krea_2" => "ENG-4b/APP-5",
            "flux3" | "moss_image" | "iris" | "ideogram" | "supra2" => "ENG-5/APP-5",
            "ltx-video" | "minimax-h3" | "mage-flow" | "sensenova-u1" | "bernini"
            | "krea-realtime" => "ENG-6/APP-5/APP-6",
            other => return Err(format!("no migration owner for {}/{other}", model.id)),
        },
        other => return Err(format!("unclassified model type {other} for {}", model.id)),
    })
}

fn output(model: &ModelCapabilityRow, mode: &str) -> String {
    match mode {
        "vqa" | "training_caption" | "image_caption" | "prompt_refine" => "text".into(),
        "interleave" => "ordered-text/image-document".into(),
        "image_depth" => "float32-npy-depth+derived-preview".into(),
        "smart_segment" | "person_segment" => "mask".into(),
        _ if mode.starts_with("prompt_refine:") => "text".into(),
        _ if mode.starts_with("provider:audio_") => mode.trim_start_matches("provider:").into(),
        _ => match model.model_type.as_str() {
            "vector" => "canonical-svg+derived-preview+terminal-metadata",
            "image" => "image (provider alpha/count contract retained)",
            "video" => "video (provider audio/count contract retained)",
            "audio" => "audio",
            _ if mode.starts_with("engine:") => "image",
            _ => "structured-metadata",
        }
        .into(),
    }
}

fn axes(model: &ModelCapabilityRow) -> [(&'static str, &[CapabilityCell]); 6] {
    [
        ("operation", &model.operation_and_mode),
        ("conditioning", &model.conditioning_shape),
        ("adapter", &model.user_adapters),
        ("precision", &model.precision_tier),
        ("guidance", &model.guidance_method),
        ("preview", std::slice::from_ref(&model.preview)),
    ]
}

fn supported(cell: &CapabilityCell, backend: &str) -> Option<bool> {
    if backend == "mlx" {
        cell.mlx
    } else {
        cell.candle
    }
}

fn key(backend: &str, model: &str, axis: &str, capability: &str) -> String {
    format!("{backend}/{model}/{axis}/{capability}")
}

fn conditioning_meaning(model: &ModelCapabilityRow, axis: &str, mode: &str) -> &'static str {
    match mode {
        "control" => "structural ControlNet (pose/control image), not VAE initialization",
        "referenceRgba" => "ordered reference with native alpha, not flattened RGB",
        "mask" | "image_inpaint" => "source image plus inpainting mask",
        "image_to_image" => "source-image VAE initialization and strength blend",
        "reference" if model.family == "flux" && !model.id.starts_with("pulid") => {
            "XLabs IP-adapter image embedding, not VAE initialization"
        }
        "reference" if model.id.starts_with("pulid") || model.id.starts_with("instantid") => {
            "identity reference (PuLID/InstantID), not VAE initialization"
        }
        "reference"
            if model.model_type == "image"
                && (model.family.starts_with("flux2")
                    || model.family.starts_with("qwen")
                    || model.family == "mage-flow"
                    || model.family == "sensenova-u1") =>
        {
            "edit/reference conditioning; see canonical request and selected lane"
        }
        "reference" => "provider reference conditioning; see canonical request and selected lane",
        "multiReference" | "reduxRefs" => {
            "ordered multiple references; source retained for source-with-reference edit"
        }
        "image_to_svg" => "image prefix to native token driver",
        "character_image" => {
            "reference-conditioned image; provider selector defines identity versus edit"
        }
        "edit_image" => {
            "source-image edit; provider selector defines VAE init versus reference encoding"
        }
        "keyframe" | "first_last_frame" => "first/last endpoint keyframes",
        "referenceAudio" | "audioEdit" | "audioEditRegions" | "voiceEmbedding" => {
            "audio conditioning (distinct reference/edit/voice carriers)"
        }
        "videoClip" | "controlClip" | "videoSync" | "referenceVideo" => {
            "video conditioning (distinct clip/control/sync/reference carriers)"
        }
        "conversationHistory" => "ordered conversational feedback",
        _ if axis == "conditioning" => {
            "provider-specific typed conditioning (canonical request retained)"
        }
        _ => "see canonical request; feature axes do not add conditioning",
    }
}

fn output_variants(model: &ModelCapabilityRow, mode: &str) -> Vec<String> {
    if mode == "vqa"
        || mode == "interleave"
        || model.model_type != "image" && model.model_type != "video"
    {
        return vec![output(model, mode)];
    }
    match model.family.as_str() {
        "qwen-image-2-1" => vec![
            "RGB PNG; count".into(),
            "native straight-alpha RGBA PNG; transparentBackground; count".into(),
        ],
        "moss_image" => vec![
            "RGB PNG; one prompt × count".into(),
            "RGB PNGs; prompts × count with per-output seeds".into(),
        ],
        "minimax-h3" => vec!["joint video+synchronized stereo audio".into()],
        "ltx-video" => vec!["video".into(), "joint video+audio".into()],
        _ if model.model_type == "video" => {
            vec!["video; source audio passthrough where supported".into()]
        }
        _ => vec!["RGB image(s); count; editor/detail retains source alpha where supported".into()],
    }
}

fn canonical_request(
    model: &ManifestModel,
    axis: &str,
    mode: &str,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<Option<Value>, String> {
    let request = if axis == "conditioning" {
        conditioning_payload(model, mode, mlx, candle).ok()
    } else if axis == "operation" && model.model_type == "image" {
        operation_request(model, mode)
            .ok()
            .map(|(kind, payload, _)| (kind, payload))
    } else if axis == "operation" && model.model_type == "video" {
        let job = super::super::canonical_video_route_probe(&model.id, mode)?;
        return Ok(Some(
            json!({ "jobType": job.job_type, "payload": job.payload }),
        ));
    } else if axis == "operation" && model.model_type == "vector" {
        Some((
            JobType::VectorGenerate,
            json!({"mode": mode, "sourceAssetId": "probe"}),
        ))
    } else {
        None
    };
    request
        .map(|(kind, payload)| {
            probe_job(kind, &model.id, payload)
                .map(|job| json!({ "jobType": job.job_type, "payload": job.payload }))
        })
        .transpose()
}

fn binding(
    model: &ManifestModel,
    axis: &str,
    mode: &str,
    facts: &RuntimeDescriptorFacts,
    mlx: &RuntimeDescriptorFacts,
    candle: &RuntimeDescriptorFacts,
) -> Result<(Vec<String>, String), String> {
    let backend = &facts.snapshot.backend;
    if matches!(mode, "vqa" | "interleave") {
        return Ok((
            vec![format!("{}::native-{mode}", model.id)],
            format!(
                "crates/sceneworks-worker/src/sensenova_jobs.rs::run_{}_job",
                if mode == "vqa" { "vqa" } else { "interleave" }
            ),
        ));
    }
    if model.model_type == "vector" {
        let provider = model
            .vector
            .providers
            .get(backend)
            .ok_or("missing vector provider")?;
        return Ok((
            vec![provider.id.clone()],
            "crates/sceneworks-worker/src/vector_jobs.rs::run_vector_job".into(),
        ));
    }
    let probe = if axis == "conditioning" {
        conditioning_payload(model, mode, mlx, candle).ok()
    } else if model.model_type == "image" {
        operation_request(model, mode)
            .ok()
            .map(|(job, payload, _)| (job, payload))
    } else {
        None
    }
    .or_else(|| {
        (axis != "operation" && axis != "conditioning")
            .then(|| canonical_model_request(model, None, mlx, candle).ok())
            .flatten()
    });
    if let Some((job_type, payload)) = probe {
        let job = probe_job(job_type, &model.id, payload)?;
        if backend == "candle" {
            if let Some(lane) = super::super::candle::image_job_candle_lane(&job) {
                // Preserve the selected lane, including IP-adapter vs init and ControlNet vs base.
                let mut providers: Vec<String> = generator_descriptors(facts, &model.id)
                    .iter()
                    .map(|d| d.id.clone())
                    .collect();
                providers.push(format!("{}::{lane:?}", model.id));
                return Ok((providers,
                    "crates/sceneworks-core/src/jobs_store/routing/candle.rs::image_job_candle_lane -> crates/sceneworks-worker/src/image_jobs/candle.rs".into()));
            }
        }
        if bespoke_image_lane_support(backend, &model.id, axis, mode, &job)
            || (axis != "operation"
                && axis != "conditioning"
                && bespoke_image_lane_support(
                    backend,
                    &model.id,
                    "operation",
                    job.payload
                        .get("mode")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    &job,
                ))
        {
            return Ok((vec![format!("{}::{axis}/{mode}", model.id)],
                "crates/sceneworks-core/src/jobs_store/routing/mlx.rs::image_job_is_mlx_eligible -> crates/sceneworks-worker/src/image_jobs/mlx.rs".into()));
        }
    }
    let descriptors = if model.model_type == "video" && axis == "operation" {
        native_video_route_descriptors(facts, &model.id, mode)
    } else {
        generator_descriptors(facts, &model.id)
    };
    if !descriptors.is_empty() {
        return Ok((
            descriptors.iter().map(|d| d.id.clone()).collect(),
            if model.model_type == "video" {
                "crates/sceneworks-worker/src/video_jobs/mod.rs"
            } else if model.model_type == "audio" || mode.starts_with("provider:audio_") {
                "crates/sceneworks-worker/src/audio_jobs.rs"
            } else {
                "crates/sceneworks-worker/src/image_jobs.rs"
            }
            .into(),
        ));
    }
    // Utility and non-generator paths have their own lifecycle; an empty media lookup is legal.
    // This closed list is checked against the entire published product surface below.
    let entry = match mode {
        "training_caption" | "image_caption" => "caption_jobs.rs",
        "prompt_refine" | "prompt_refine:film_plan" | "prompt_refine:qwen_image_rewrite" => {
            "prompt_refine_jobs.rs"
        }
        "dataset_analysis" => "dataset_analysis_jobs.rs",
        "pose_detect" => "pose_jobs.rs",
        "person_detect" => "person_jobs.rs",
        "person_segment" | "smart_segment" => "segment_jobs.rs",
        "image_depth" => "depth_jobs.rs",
        "image_detail" => "image_jobs/detail.rs",
        "engine:iris" | "engine:real-esrgan" | "engine:seedvr2" => "upscale_jobs.rs",
        mode if mode.starts_with("provider:audio_") => "audio_jobs.rs",
        _ if axis != "operation" && model.model_type == "utility" => "utility_dispatch.rs",
        _ => {
            return Err(format!(
                "unresolved binding {backend}/{}/{axis}/{mode}",
                model.id
            ))
        }
    };
    Ok((
        vec![provider_alias(&model.id).into()],
        format!("crates/sceneworks-worker/src/{entry}"),
    ))
}

/// Reconcile the immutable shipped product input against current production routing and registries.
/// The returned report is suitable for a temporary CI artifact; do not check in another matrix.
/// No allocation or forward is performed by any model/backend.
pub fn generation_route_coverage() -> Result<RouteCoverage, String> {
    let report = reconcile(LEGACY, MLX_RUNTIME_FACTS, CANDLE_RUNTIME_FACTS)?;
    if report.sources["legacyProductModels"] != LEGACY_MODEL_DIGEST {
        return Err("the old product inventory changed; review baseline disposition independently of the candidate export".into());
    }
    Ok(report)
}

fn reconcile(legacy: &str, mlx_source: &str, candle_source: &str) -> Result<RouteCoverage, String> {
    let old: BackendCapabilityMatrix = serde_json::from_str(legacy).map_err(|e| e.to_string())?;
    let candidate = backend_capability_matrix_from_runtime_sources(mlx_source, candle_source)?;
    let mlx = runtime_facts(mlx_source, "mlx")?;
    let candle = runtime_facts(candle_source, "candle")?;
    let manifest: ManifestRoot =
        serde_json::from_str(&strip_jsonc_comments(MANIFEST)).map_err(|e| e.to_string())?;
    let mut sources = candidate.sources.clone();
    sources.insert(
        "jobDispatcher".into(),
        format!("{:x}", Sha256::digest(DISPATCH)),
    );
    sources.insert(
        "legacyProductModels".into(),
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&old.models).map_err(|e| e.to_string())?)
        ),
    );
    let mut cells = Vec::new();
    let mut missing = Vec::new();
    for model in &old.models {
        let current = candidate.models.iter().find(|m| m.id == model.id);
        let definition = manifest
            .models
            .iter()
            .find(|m| m.id == model.id)
            .ok_or_else(|| format!("legacy public model {} disappeared from manifest", model.id))?;
        for (axis, values) in axes(model) {
            for value in values {
                for facts in [&mlx, &candle] {
                    let backend = &facts.snapshot.backend;
                    let legacy_supported = supported(value, backend);
                    let candidate_supported = current
                        .and_then(|m| {
                            axes(m)
                                .into_iter()
                                .find(|(a, _)| *a == axis)
                                .and_then(|(_, values)| {
                                    values.iter().find(|c| c.capability == value.capability)
                                })
                        })
                        .and_then(|c| supported(c, backend));
                    if legacy_supported == Some(true) && candidate_supported != Some(true) {
                        missing.push(key(backend, &model.id, axis, &value.capability));
                    }
                    let (providers, entry_point) = if candidate_supported == Some(true) {
                        binding(definition, axis, &value.capability, facts, &mlx, &candle)?
                    } else {
                        (
                            Vec::new(),
                            "typed unsupported/not-applicable witness".into(),
                        )
                    };
                    cells.push(RouteCell {
                        platform: if backend == "mlx" {
                            "macos"
                        } else {
                            "windows/linux-cuda"
                        }
                        .into(),
                        backend: backend.clone(),
                        model: model.id.clone(),
                        origin: if value.capability.starts_with("provider:") {
                            "direct-engine"
                        } else {
                            "public-product"
                        },
                        canonical_request: canonical_request(
                            definition,
                            axis,
                            &value.capability,
                            &mlx,
                            &candle,
                        )?,
                        axis,
                        mode: value.capability.clone(),
                        conditioning: conditioning_meaning(model, axis, &value.capability).into(),
                        output: output(model, &value.capability),
                        output_variants: output_variants(model, &value.capability),
                        expected_provider_ids: if definition.model_type == "vector" {
                            definition
                                .vector
                                .providers
                                .get(backend)
                                .map(|p| vec![p.id.clone()])
                                .unwrap_or_default()
                        } else if let Some(provider) = facts.model_mappings.get(&model.id) {
                            vec![provider.clone()]
                        } else {
                            providers.clone()
                        },
                        providers,
                        owner: owner(model)?,
                        legacy_supported,
                        candidate_supported,
                        entry_point,
                    });
                }
            }
        }
    }
    for model in &candidate.models {
        if let Some(baseline) = old.models.iter().find(|m| m.id == model.id) {
            for (axis, values) in axes(model) {
                let prior = axes(baseline)
                    .into_iter()
                    .find(|(a, _)| *a == axis)
                    .unwrap()
                    .1;
                for value in values {
                    if !prior.iter().any(|v| v.capability == value.capability) {
                        return Err(format!(
                            "new public cell {}/{axis}/{} needs a reviewed baseline disposition",
                            model.id, value.capability
                        ));
                    }
                }
            }
        }
    }
    // A new public model must be dispositioned too; it cannot hide outside the old-side loop.
    for model in &candidate.models {
        if !old.models.iter().any(|old| old.id == model.id) {
            return Err(format!(
                "new public model {} needs a reviewed baseline disposition",
                model.id
            ));
        }
    }
    let mut imported_bindings = Vec::new();
    for (backend, source) in [
        ("mlx", MLX_DESCRIPTOR_FACTS),
        ("candle", CANDLE_DESCRIPTOR_FACTS),
    ] {
        let facts: Value = serde_json::from_str(source).map_err(|e| e.to_string())?;
        for imported in facts["imports"]
            .as_array()
            .ok_or("missing imported provider declarations")?
        {
            let family = imported["family"]
                .as_str()
                .ok_or("imported route has no family")?;
            let owner = match family {
                "krea_2" => "ENG-4b",
                "flux2" | "qwen-image" | "z-image" => "ENG-2",
                "wan-video" => "ENG-3/ENG-5",
                "sdxl" => "ENG-5",
                "mage-flow" => "ENG-6",
                other => return Err(format!("unowned imported route family {other}")),
            };
            let public_family = candidate
                .imported_families
                .iter()
                .any(|row| row.family == family);
            imported_bindings.push(json!({ "backend": backend, "origin": "imported-checkpoint", "declaration": imported,
                "owner": owner, "output": "provider image/video output", "publicFamilyEvaluated": public_family,
                "entryPoint": "runtime bundle catalog().media().imported_model_descriptor(family, source, operation) -> resolved provider load; product checkpoint_plan.rs when publicly eligible",
                "routingWitness": if public_family { "imported_families: production imported_family_rows checks family/source/operation against backend_supports" }
                    else { "direct engine imported binding; no SceneWorks import eligibility is inferred" } }));
        }
    }
    let mut trainer_mappings = Vec::new();
    let mut registries = Vec::new();
    let mut model_aliases = Vec::new();
    for source in [mlx_source, candle_source] {
        let runtime: Value = serde_json::from_str(source).map_err(|e| e.to_string())?;
        let snapshot = runtime["snapshot"].as_object().ok_or("missing snapshot")?;
        let platform = snapshot["platform"].as_str().ok_or("missing platform")?;
        let media_backend = snapshot["backend"].as_str().ok_or("missing backend")?;
        trainer_mappings.push(json!({ "platform": platform, "backend": media_backend, "mappings": runtime["trainerMappings"], "routingWitness": "training_kernels" }));
        for (model, provider) in runtime["modelMappings"]
            .as_object()
            .ok_or("missing model mappings")?
        {
            let registered = snapshot["generator_ids"]
                .as_array()
                .ok_or("missing generator IDs")?
                .contains(provider);
            model_aliases.push(json!({ "platform": platform, "backend": media_backend, "model": model,
                "provider": provider, "registeredMediaGenerator": registered,
                "publicModel": candidate.models.iter().any(|m| m.id == *model),
                "entryPoint": if registered { "runtime bundle media registry (public eligibility checked separately)" }
                    else { "production bespoke selector; alias alone does not establish eligibility" },
                "owner": "ENG-7" }));
        }
        for (surface, ids) in snapshot
            .iter()
            .filter(|(k, _)| k.ends_with("_ids") || k.ends_with("_backends"))
        {
            let backend = if surface.starts_with("audio_") {
                snapshot["audio_backend"]
                    .as_str()
                    .ok_or("audio surface without backend")?
            } else {
                media_backend
            };
            for id in ids.as_array().ok_or("registry IDs must be arrays")? {
                let provider = id.as_str().ok_or("registry ID must be a string")?;
                let public_cells: Vec<_> = cells
                    .iter()
                    .filter(|c| c.backend == backend && c.providers.iter().any(|p| p == provider))
                    .map(|c| key(backend, &c.model, c.axis, &c.mode))
                    .collect();
                registries.push(RegistrySurface {
                    platform: platform.into(),
                    backend: backend.into(),
                    surface: surface.clone(),
                    provider: provider.into(),
                    descriptor: snapshot
                        .get(if surface == "audio_generator_ids" {
                            "audio_generator_capabilities"
                        } else if surface == "trainer_ids" {
                            "trainer_capabilities"
                        } else {
                            "generator_capabilities"
                        })
                        .and_then(Value::as_array)
                        .and_then(|ds| ds.iter().find(|d| d["id"] == provider))
                        .cloned(),
                    owner: registry_owner(surface, provider, snapshot)?,
                    disposition: if public_cells.is_empty() {
                        "direct-engine/bundle-consumer"
                    } else {
                        "public-and-direct-engine"
                    },
                    public_cells,
                    entry_point: format!("runtime-{platform}::catalog().snapshot().{surface}"),
                    callable_entry: registry_entry(platform, surface, provider),
                });
            }
        }
    }
    // The mobile and CPU consumers do not submit SceneWorks jobs. Keep their independently
    // source-identified profile contracts rather than projecting desktop product eligibility.
    let profile_source_receipt: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/generation-route-profiles.json"
    ))
    .map_err(|e| e.to_string())?;
    for (surface, ids) in profile_source_receipt["ios"]
        .as_object()
        .ok_or("missing iOS profile")?
    {
        for id in ids.as_array().ok_or("profile IDs must be arrays")? {
            registries.push(RegistrySurface {
                platform: "ios".into(),
                backend: "mlx".into(),
                surface: surface.clone(),
                provider: id.as_str().ok_or("invalid profile ID")?.into(),
                descriptor: None,
                owner: "ENG-13",
                public_cells: Vec::new(),
                disposition: "direct-engine/bundle-consumer; source-identified profile",
                entry_point:
                    "runtime-ios::catalog() (media-enabled; LLM-only omits generators/transforms)"
                        .into(),
                callable_entry: registry_entry("ios", surface, id.as_str().unwrap()),
            });
        }
    }
    let cpu_llms = profile_source_receipt["cpu"]["text_llm_ids"]
        .as_array()
        .ok_or("missing CPU LLM profile")?;
    let candle_registries: Vec<_> = registries
        .iter()
        .filter(|r| r.platform == "cuda" && r.surface != "text_llm_ids")
        .map(|r| RegistrySurface {
            platform: "cpu".into(),
            backend: r.backend.clone(),
            surface: r.surface.clone(),
            provider: r.provider.clone(),
            descriptor: r.descriptor.clone(),
            owner: r.owner,
            public_cells: Vec::new(),
            disposition:
                "direct-engine/bundle-consumer; shared Candle catalog, separate device admission",
            entry_point:
                "runtime-cpu::catalog() (default media+audio; profiles may disable either)".into(),
            callable_entry: registry_entry("cpu", &r.surface, &r.provider),
        })
        .collect();
    registries.extend(candle_registries);
    for id in cpu_llms {
        registries.push(RegistrySurface {
            platform: "cpu".into(),
            backend: "candle".into(),
            surface: "text_llm_ids".into(),
            provider: id.as_str().ok_or("invalid CPU LLM ID")?.into(),
            descriptor: None,
            owner: if id.as_str().unwrap().contains("starvector") {
                "ENG-7/APP-5"
            } else {
                "ENG-14"
            },
            public_cells: Vec::new(),
            disposition: "direct-engine/bundle-consumer; source-identified profile",
            entry_point: "runtime-cpu::catalog().text()".into(),
            callable_entry: registry_entry("cpu", "text_llm_ids", id.as_str().unwrap()),
        });
    }
    let mut direct_route_witnesses = Vec::new();
    for model in manifest.models.iter().filter(|m| m.model_type == "vector") {
        let job = probe_job(
            JobType::VectorGenerate,
            &model.id,
            json!({ "mode": "text_to_svg", "prompt": "probe" }),
        )?;
        for facts in [&mlx, &candle] {
            direct_route_witnesses.push(json!({ "origin": "public-operation-negative-witness", "backend": facts.snapshot.backend,
                "provider": model.vector.providers.get(&facts.snapshot.backend).map(|p| &p.id), "model": model.id,
                "mode": "text_to_svg", "conditioning": "text-only prompt", "output": "SVG if eligible; no artifacts on refusal",
                "supported": backend_supports(&job, facts)?, "owner": "ENG-7/APP-5",
                "entryPoint": "VectorGenerate -> production vector eligibility -> vector_jobs.rs", "request": job.payload }));
        }
    }
    for provider in ["flux1_dev", "flux1_schnell"] {
        for (platform, backend) in [("macos", "mlx"), ("cuda", "candle"), ("cpu", "candle")] {
            direct_route_witnesses.push(json!({ "platform": platform, "backend": backend,
                "provider": provider, "mode": "image_to_image", "conditioning": "VAE init + strength blend",
                "output": "RGB image", "supported": false, "owner": "ENG-1a",
                "source": "source receipt: mlx-gen-flux/src/config.rs + model.rs; candle-gen-flux/src/lib.rs",
                "reason": "base FLUX.1 has no img2img init path; MLX Reference is XLabs IP and requires LoadSpec::with_ip_adapter; Candle reference/IP uses the separate bespoke worker lane" }));
        }
        direct_route_witnesses.push(json!({ "platform": "macos", "backend": "mlx", "provider": provider,
            "mode": "ip_adapter", "conditioning": "one Reference image with IP adapter scale",
            "output": "RGB image", "supported": true, "owner": "ENG-1a",
            "source": "source receipt: mlx-gen-flux/src/model.rs::extract_reference",
            "entryPoint": format!("runtime-macos::catalog().media().load({provider}, LoadSpec::with_ip_adapter) -> Generator::generate") }));
    }
    let flux2_init = mlx
        .snapshot
        .generator_capabilities
        .iter()
        .any(|d| d.id == "flux2_dev" && d.conditioning.iter().any(|c| c == "reference"));
    direct_route_witnesses.push(json!({ "platform": "macos", "backend": "mlx", "provider": "flux2_dev",
        "mode": "image_to_image", "conditioning": "one Reference image as VAE initialization with strength blend",
        "output": "RGB image", "supported": flux2_init, "owner": "ENG-2",
        "source": "source receipt: mlx-gen-flux2/src/config.rs::Flux2Variant::descriptor",
        "entryPoint": "runtime-macos::catalog().media().load(flux2_dev, LoadSpec) -> Generator::generate(GenerationRequest)",
        "disposition": "direct-engine route; product reference/edit eligibility remains independently selected" }));
    if !flux2_init {
        missing.push("macos/mlx/flux2_dev/direct-engine/image_to_image".into());
    }
    if let Some(surface) = registries.iter().find(|r| {
        r.callable_entry
            .starts_with("unclassified registry surface:")
    }) {
        return Err(surface.callable_entry.clone());
    }
    let jobs = JobType::known_values()
        .iter()
        .map(|wire| {
            let job: JobType = serde_json::from_value(json!(wire)).map_err(|e| e.to_string())?;
            lifecycle(job)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RouteCoverage {
        schema_version: 1,
        evidence_kind: "metadata-only; registration is not execution or device-fit evidence",
        sources,
        jobs,
        cells,
        registries,
        profile_source_receipt,
        model_aliases,
        direct_route_witnesses,
        imported_families: candidate.imported_families,
        imported_bindings,
        trainer_mappings,
        training_kernels: candidate.training_kernels,
        missing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_public_route_surface_has_bindings_or_negative_witnesses() {
        let report = generation_route_coverage().unwrap();
        assert!(report.missing.is_empty(), "{:?}", report.missing);
        assert_eq!(report.jobs.len(), JobType::known_values().len());
        assert_eq!(
            report.imported_families.len(),
            crate::base_weights::IMPORT_SUPPORTED_FAMILIES.len()
        );
        assert!(report.cells.iter().any(|c| c.model == "flux_dev"
            && c.axis == "conditioning"
            && c.mode == "control"
            && c.candidate_supported == Some(true)));
        assert!(report
            .cells
            .iter()
            .any(|c| c.model == "pulid_flux_dev" && c.candidate_supported == Some(true)));
    }

    #[test]
    fn removing_starvector_names_exactly_four_backend_provider_operation_cells() {
        fn without_starvector(source: &str) -> String {
            let mut value: Value = serde_json::from_str(source).unwrap();
            value["snapshot"]["text_llm_ids"]
                .as_array_mut()
                .unwrap()
                .retain(|id| !id.as_str().unwrap().contains("starvector"));
            serde_json::to_string(&value).unwrap()
        }
        let report = reconcile(
            LEGACY,
            &without_starvector(MLX_RUNTIME_FACTS),
            &without_starvector(CANDLE_RUNTIME_FACTS),
        )
        .unwrap();
        let expected_providers: BTreeSet<_> = report
            .cells
            .iter()
            .filter(|r| {
                r.model.starts_with("starvector_")
                    && r.axis == "operation"
                    && r.legacy_supported == Some(true)
                    && r.candidate_supported == Some(false)
            })
            .flat_map(|r| r.expected_provider_ids.iter().map(String::as_str))
            .collect();
        assert_eq!(
            expected_providers,
            [
                "mlx-starvector-1b",
                "mlx-starvector-8b",
                "candle-starvector-1b",
                "candle-starvector-8b"
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(
            report.missing,
            [
                "mlx/starvector_1b/operation/image_to_svg",
                "candle/starvector_1b/operation/image_to_svg",
                "mlx/starvector_8b/operation/image_to_svg",
                "candle/starvector_8b/operation/image_to_svg",
            ]
        );
    }

    #[test]
    fn published_product_input_cannot_silently_lose_a_baseline_cell() {
        let report = generation_route_coverage().unwrap();
        assert_eq!(report.sources["legacyProductModels"], "105abeb6caea45068044f0d4b6e6fcf8a6bcf58f003e81de7c016514fa454c82",
            "the old product inventory changed: review its disposition independently of the candidate export");
    }

    #[test]
    fn every_native_dispatch_arm_is_joined_to_a_lifecycle() {
        use syn::visit::Visit;
        struct Dispatch {
            variants: BTreeSet<String>,
        }
        fn variants(pattern: &syn::Pat, set: &mut BTreeSet<String>) {
            match pattern {
                syn::Pat::Path(path) => {
                    if path
                        .path
                        .segments
                        .iter()
                        .rev()
                        .nth(1)
                        .is_some_and(|part| part.ident == "JobType")
                    {
                        set.insert(path.path.segments.last().unwrap().ident.to_string());
                    }
                }
                syn::Pat::Or(or) => {
                    for case in &or.cases {
                        variants(case, set);
                    }
                }
                _ => (),
            }
        }
        impl<'ast> Visit<'ast> for Dispatch {
            fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
                if matches!(node.expr.as_ref(), syn::Expr::Field(field) if matches!(&field.member, syn::Member::Named(name) if name == "job_type"))
                {
                    for arm in &node.arms {
                        if arm.guard.is_none() {
                            variants(&arm.pat, &mut self.variants);
                        }
                    }
                    return;
                }
                syn::visit::visit_expr_match(self, node);
            }
        }
        let syntax = syn::parse_file(DISPATCH).unwrap();
        let function = syntax
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Fn(f) if f.sig.ident == "dispatch_utility_job" => Some(f),
                _ => None,
            })
            .expect("the public worker dispatch owner must exist");
        let mut dispatch = Dispatch {
            variants: BTreeSet::new(),
        };
        dispatch.visit_block(&function.block);
        let expected: BTreeSet<_> = JobType::known_values()
            .iter()
            .map(|wire| {
                format!(
                    "{:?}",
                    serde_json::from_value::<JobType>(json!(wire)).unwrap()
                )
            })
            .collect();
        assert_eq!(
            dispatch.variants, expected,
            "comments and nested handler text cannot replace a native job arm"
        );
    }

    #[test]
    fn semantic_routes_keep_bespoke_conditioning_and_output_contracts() {
        let report = generation_route_coverage().unwrap();
        let flux_ip = report
            .cells
            .iter()
            .find(|r| {
                r.backend == "candle"
                    && r.model == "flux_dev"
                    && r.axis == "conditioning"
                    && r.mode == "reference"
            })
            .unwrap();
        assert!(flux_ip
            .providers
            .iter()
            .any(|p| p.ends_with("::FluxIpAdapter")));
        assert!(flux_ip.conditioning.contains("IP-adapter"));
        for model in ["qwen_image_edit_2511", "qwen_image_edit_2511_lightning"] {
            let edit = report
                .cells
                .iter()
                .find(|r| {
                    r.backend == "candle"
                        && r.model == model
                        && r.axis == "operation"
                        && r.mode == "edit_image"
                })
                .unwrap();
            assert!(edit.providers.iter().any(|p| p.ends_with("::QwenEdit")));
        }
        assert_eq!(
            report
                .model_aliases
                .iter()
                .filter(|a| a["backend"] == "candle"
                    && a["provider"] == "qwen_image_edit"
                    && a["registeredMediaGenerator"] == false)
                .count(),
            4
        );
        let vqa = report
            .cells
            .iter()
            .find(|r| r.model == "sensenova_u1_8b" && r.mode == "vqa")
            .unwrap();
        assert!(vqa.entry_point.ends_with("sensenova_jobs.rs::run_vqa_job"));
        assert_eq!(vqa.output, "text");
        assert!(report
            .direct_route_witnesses
            .iter()
            .any(|r| r["provider"] == "flux2_dev"
                && r["mode"] == "image_to_image"
                && r["supported"] == true));
        assert!(!report.direct_route_witnesses.iter().any(|r| r["provider"]
            .as_str()
            .unwrap()
            .starts_with("flux1_")
            && r["mode"] == "image_to_image"
            && r["supported"] == true));
        let ios: Vec<_> = report
            .registries
            .iter()
            .filter(|r| r.platform == "ios" && r.surface == "generator_ids")
            .map(|r| r.provider.as_str())
            .collect();
        assert_eq!(ios.len(), 7);
        assert!(!ios.iter().any(|id| id.starts_with("flux1")));
        assert!(!report
            .registries
            .iter()
            .any(|r| r.platform == "cpu" && r.provider == "candle-starvector-8b"));
    }

    #[test]
    fn unowned_new_family_and_unknown_job_fail_closed() {
        let mut report: BackendCapabilityMatrix = serde_json::from_str(LEGACY).unwrap();
        report
            .models
            .iter_mut()
            .find(|m| m.model_type == "image")
            .unwrap()
            .family = "unassigned-family".into();
        assert!(reconcile(
            &serde_json::to_string(&report).unwrap(),
            MLX_RUNTIME_FACTS,
            CANDLE_RUNTIME_FACTS
        )
        .unwrap_err()
        .contains("no migration owner"));
        assert!(lifecycle(JobType::Unknown("new-job".into())).is_err());
    }
}
