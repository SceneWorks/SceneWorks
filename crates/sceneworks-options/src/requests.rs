//! Public non-training request options and their source-owned nested contracts.
//! Locations are exact wire paths. Metadata/transport fields retain their distinct intent.
//! Limits cite current owners; numeric clamps are explicitly different from refusal bounds.

use crate::*;

const HIRES_FIX: Predicate = Predicate::All(&[
    Predicate::Modality("image"),
    Predicate::ModelNotFalse("image.supportsHiresFix"),
]);

pub(crate) const SPECS: &[OptionSpec] = &[

    // image
    // apps/rust-api/src/dto.rs:824
    OptionSpec::new(
        "image.projectId",
        "image",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:826
    OptionSpec::new(
        "image.appearance",
        "image",
        "appearance",
        Shape::Object("AppearanceSelection"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.appearance",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: resolve_appearance_for_family validates catalog selections",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "selection supplied", action: "compose family photography directives and negative text", target: "image.prompt" },
    ])
    .ui("Appearance", "object"),
    // apps/rust-api/src/dto.rs:828
    OptionSpec::new(
        "image.mossImageAdvanced",
        "image",
        "mossImageAdvanced",
        Shape::Object("EditAdvanced"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "advanced.mossImageAdvanced -> MossImageAdvancedInputs",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: explicit MOSS edit_image only",
    ])
    .applies(Predicate::All(&[Predicate::Family("moss_image"), Predicate::Capability("edit_image")]))
    .compat("Top-level public carrier is normalized into advanced.mossImageAdvanced")
    .ui("Moss image advanced", "object"),
    // apps/rust-api/src/dto.rs:830
    OptionSpec::new(
        "image.projectName",
        "image",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Modality("image"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:832
    OptionSpec::new(
        "image.mode",
        "image",
        "mode",
        Shape::OneOf("text_to_image | edit_image | character_image | style_variations"),
        DefaultRule::Literal("\"text_to_image\""),
        IntentClass::WhatToMake,
        "ImageRequest.mode",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_image_job mode allow-list; selected model capability",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Mode", "select"),
    // apps/rust-api/src/dto.rs:834
    OptionSpec::new(
        "image.prompt",
        "image",
        "prompt",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "GenerationRequest.prompt",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: 1..4000 characters; may be empty with explicit MOSS edit positive embeddings",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Prompt", "text"),
    // apps/rust-api/src/dto.rs:839
    OptionSpec::new(
        "image.prompts",
        "image",
        "prompts",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "GenerationRequest.prompts",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: each prompt 1..4000 characters",
        "apps/rust-api/src/generation.rs: model image.maxPromptBatchImages; prompt must equal prompts[0]; batch/count product bound",
    ])
    .applies(Predicate::ModelField("image.maxPromptBatchImages"))
    .ui("Prompts", "list"),
    // apps/rust-api/src/dto.rs:841
    OptionSpec::new(
        "image.negativePrompt",
        "image",
        "negativePrompt",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "GenerationRequest.negative_prompt",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_image_prompt_extras text length and family contract",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Negative prompt", "text"),
    // apps/rust-api/src/dto.rs:843
    OptionSpec::new(
        "image.model",
        "image",
        "model",
        Shape::Text,
        DefaultRule::Literal("\"z_image_turbo\""),
        IntentClass::WhatToMake,
        "ImageRequest.model",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_model_id: non-empty plain path component; resolved catalog/provider must support the request",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Model", "text"),
    // apps/rust-api/src/dto.rs:850
    OptionSpec::new(
        "image.count",
        "image",
        "count",
        Shape::Integer,
        DefaultRule::Model("defaults.count"),
        IntentClass::WhatToMake,
        "ImageRequest.count",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: image_count_error: 1..8; selected model output count limits",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Count", "number"),
    // apps/rust-api/src/dto.rs:852
    OptionSpec::new(
        "image.seed",
        "image",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.seed",
    )
    .applies(Predicate::Modality("image"))
    .ui("Seed", "number"),
    // apps/rust-api/src/dto.rs:861
    OptionSpec::new(
        "image.width",
        "image",
        "width",
        Shape::Integer,
        DefaultRule::Model("defaults.resolution.width"),
        IntentClass::WhatToMake,
        "ImageRequest.width",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: declared geometry limits and required dimension multiple",
        "apps/rust-api/src/validators.rs: outer named dimension 16..4096; selected model floor remains binding",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Width", "number"),
    // apps/rust-api/src/dto.rs:863
    OptionSpec::new(
        "image.height",
        "image",
        "height",
        Shape::Integer,
        DefaultRule::Model("defaults.resolution.height"),
        IntentClass::WhatToMake,
        "ImageRequest.height",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: declared geometry limits and required dimension multiple",
        "apps/rust-api/src/validators.rs: outer named dimension 16..4096; selected model floor remains binding",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Height", "number"),
    // apps/rust-api/src/dto.rs:865
    OptionSpec::new(
        "image.stylePreset",
        "image",
        "stylePreset",
        Shape::Text,
        DefaultRule::Literal("\"cinematic\""),
        IntentClass::WhatToMake,
        "ImageRequest.style_preset",
    )
    .applies(Predicate::Modality("image"))
    .compat("Legacy DTO field; styleId is the catalog selector; source documents stylePreset as unused")
    .ui("Style preset", "text"),
    // apps/rust-api/src/dto.rs:873
    OptionSpec::new(
        "image.styleId",
        "image",
        "styleId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.style_id",
    )
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "headless request without client-resolved prompt", action: "compose catalog style directives into prompt", target: "GenerationRequest.prompt" },
    ])
    .ui("Style id", "text"),
    // apps/rust-api/src/dto.rs:875
    OptionSpec::new(
        "image.recipePresetId",
        "image",
        "recipePresetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "ImageRequest.recipe_preset_id",
    )
    .applies(Predicate::Modality("image"))
    .ui("Recipe preset id", "text"),
    // apps/rust-api/src/dto.rs:881
    OptionSpec::new(
        "image.presetLorasResolvedClientSide",
        "image",
        "presetLorasResolvedClientSide",
        Shape::Bool,
        DefaultRule::Absent,
        IntentClass::Context,
        "ImageRequest.preset_loras_resolved_client_side",
    )
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "true", action: "skip server preset LoRA merge", target: "image.recipePresetId" },
    ])
    .compat("Public transport coordination flag; preserve client-resolved intent")
    .ui("Preset loras resolved client side", "toggle"),
    // apps/rust-api/src/dto.rs:888
    OptionSpec::new(
        "image.presetPromptResolvedClientSide",
        "image",
        "presetPromptResolvedClientSide",
        Shape::Bool,
        DefaultRule::Absent,
        IntentClass::Context,
        "ImageRequest.preset_prompt_resolved_client_side",
    )
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "true", action: "skip server preset prompt composition", target: "image.recipePresetId" },
    ])
    .compat("Public transport coordination flag; preserve client-resolved intent")
    .ui("Preset prompt resolved client side", "toggle"),
    // apps/rust-api/src/dto.rs:890
    OptionSpec::new(
        "image.loras",
        "image",
        "loras",
        Shape::List("LoRA selection { id, weight }"),
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "GenerationRequest.loras",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 10 entries; crates/sceneworks-core/src/lora_family.rs: selected model/adapter family compatibility",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Loras", "list"),
    // apps/rust-api/src/dto.rs:892
    OptionSpec::new(
        "image.characterId",
        "image",
        "characterId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.character_id",
    )
    .applies(Predicate::Modality("image"))
    .ui("Character id", "text"),
    // apps/rust-api/src/dto.rs:894
    OptionSpec::new(
        "image.characterLookId",
        "image",
        "characterLookId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.character_look_id",
    )
    .applies(Predicate::Modality("image"))
    .ui("Character look id", "text"),
    // apps/rust-api/src/dto.rs:896
    OptionSpec::new(
        "image.sourceAssetId",
        "image",
        "sourceAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::Image edit initialization",
    )
    .applies(Predicate::Modality("image"))
    .ui("Source asset id", "text"),
    // apps/rust-api/src/dto.rs:901
    OptionSpec::new(
        "image.referenceAssetId",
        "image",
        "referenceAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::Reference / IP-Adapter reference",
    )
    .applies(Predicate::Modality("image"))
    .ui("Reference asset id", "text"),
    // apps/rust-api/src/dto.rs:910
    OptionSpec::new(
        "image.referenceAssetIds",
        "image",
        "referenceAssetIds",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "Conditioning::MultiReference",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: flux2 edit reference cap and family-specific conditioning limits",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "non-empty FLUX.2 reference list", action: "prefer multi-reference list over sourceAssetId", target: "Conditioning::MultiReference" },
    ])
    .ui("Reference asset ids", "list"),
    // apps/rust-api/src/dto.rs:914
    OptionSpec::new(
        "image.strength",
        "image",
        "strength",
        Shape::OneOf("JSON value retained for family validation; valid numeric strength is model-specific"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.strength",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: family validation retains exact JSON value; FLUX.3 refuses unsupported strength controls",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Preserve explicit carrier for family validation; do not silently drop or coerce unsupported input")
    .ui("Strength", "select"),
    // apps/rust-api/src/dto.rs:916
    OptionSpec::new(
        "image.referenceStrength",
        "image",
        "referenceStrength",
        Shape::OneOf("JSON value retained for family validation; valid numeric strength is model-specific"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.reference_strength",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: family validation retains exact JSON value; FLUX.3 refuses unsupported strength controls",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Preserve explicit carrier for family validation; do not silently drop or coerce unsupported input")
    .ui("Reference strength", "select"),
    // apps/rust-api/src/dto.rs:918
    OptionSpec::new(
        "image.denoisingStrength",
        "image",
        "denoisingStrength",
        Shape::OneOf("JSON value retained for family validation; valid numeric strength is model-specific"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.denoising_strength",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: family validation retains exact JSON value; FLUX.3 refuses unsupported strength controls",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Preserve explicit carrier for family validation; do not silently drop or coerce unsupported input")
    .ui("Denoising strength", "select"),
    // apps/rust-api/src/dto.rs:920
    OptionSpec::new(
        "image.imageStrength",
        "image",
        "imageStrength",
        Shape::OneOf("JSON value retained for family validation; valid numeric strength is model-specific"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.image_strength",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: family validation retains exact JSON value; FLUX.3 refuses unsupported strength controls",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Preserve explicit carrier for family validation; do not silently drop or coerce unsupported input")
    .ui("Image strength", "select"),
    // apps/rust-api/src/dto.rs:922
    OptionSpec::new(
        "image.ipAdapterScale",
        "image",
        "ipAdapterScale",
        Shape::OneOf("JSON value retained for family validation; valid numeric strength is model-specific"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "ImageRequest.ip_adapter_scale",
    )
    .aliases(&["ip_adapter_scale"])
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: family validation retains exact JSON value; FLUX.3 refuses unsupported strength controls",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Preserve explicit carrier for family validation; do not silently drop or coerce unsupported input")
    .ui("Ip adapter scale", "select"),
    // apps/rust-api/src/dto.rs:933
    OptionSpec::new(
        "image.maskAssetId",
        "image",
        "maskAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::Mask",
    )
    .applies(Predicate::Capability("image_inpaint"))
    .compat("Non-inpaint-capable models ignore the mask and perform whole-image editing")
    .ui("Mask asset id", "text"),
    // apps/rust-api/src/dto.rs:939
    OptionSpec::new(
        "image.fitMode",
        "image",
        "fitMode",
        Shape::OneOf("crop | pad | outpaint | stretch"),
        DefaultRule::Literal("\"crop\""),
        IntentClass::WhatToMake,
        "ImageRequest.fit_mode",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: normalize_fit_mode; unsupported/unknown values fall back to crop; outpaint requires inpainting capability",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "unknown token", action: "normalize to crop", target: "image.fitMode" },
    ])
    .ui("Fit mode", "select"),
    // apps/rust-api/src/dto.rs:941
    OptionSpec::new(
        "image.requestedGpu",
        "image",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),
    // apps/rust-api/src/dto.rs:943
    OptionSpec::new(
        "image.upscale",
        "image",
        "upscale",
        Shape::Object("ImageUpscaleRequest"),
        DefaultRule::Literal("{\"enabled\":false,\"factor\":2,\"engine\":\"real-esrgan\"}"),
        IntentClass::WhatToMake,
        "ImageRequest.upscale",
    )
    .applies(Predicate::Modality("image"))
    .ui("Upscale", "object"),
    // apps/rust-api/src/dto.rs:950
    OptionSpec::new(
        "image.hiresFix",
        "image",
        "hiresFix",
        Shape::Object("HiresFixRequest"),
        DefaultRule::Literal("{\"enabled\":false,\"steps\":0,\"denoisingStrength\":0.7,\"upscaleBy\":2.0}"),
        IntentClass::WhatToMake,
        "ImageRequest.hires_fix",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: hires_fix_request_error: text_to_image, img2img-capable model; excludes inline upscale, edit/reference/character/mask/control input, usePid, phases and lightning/lcm/hyper acceleration",
    ])
    .applies(HIRES_FIX)
    .ui("Hires fix", "object"),
    // apps/rust-api/src/dto.rs:952
    OptionSpec::new(
        "image.advanced",
        "image",
        "advanced",
        Shape::Object("extension container; leaf options are declared separately"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "ImageRequest.advanced",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_prompt_extras serialized advanced size guard",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Carrier only; do not present the object as a scalar option")
    .ui("Advanced", "object"),
    // apps/rust-api/src/dto.rs:956
    OptionSpec::new(
        "image.workflowParentId",
        "image",
        "workflowParentId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "ImageRequest.workflow_parent_id",
    )
    .applies(Predicate::Modality("image"))
    .internal(),
    // apps/rust-api/src/dto.rs:958
    OptionSpec::new(
        "image.workflowId",
        "image",
        "workflowId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "ImageRequest.workflow_id",
    )
    .applies(Predicate::Modality("image"))
    .internal(),

    // vector
    // apps/rust-api/src/dto.rs:1028
    OptionSpec::new(
        "vector.projectId",
        "vector",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:1030
    OptionSpec::new(
        "vector.projectName",
        "vector",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Modality("vector"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:1031
    OptionSpec::new(
        "vector.mode",
        "vector",
        "mode",
        Shape::OneOf("image_to_svg | text_to_svg"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "VectorRequest.mode",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: selected vector model modality capability; image_to_svg needs source asset, text_to_svg forbids source asset",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Mode", "select"),
    // apps/rust-api/src/dto.rs:1032
    OptionSpec::new(
        "vector.model",
        "vector",
        "model",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "VectorRequest.model",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_model_id: non-empty plain path component; resolved catalog/provider must support the request",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Model", "text"),
    // apps/rust-api/src/dto.rs:1034
    OptionSpec::new(
        "vector.sourceAssetId",
        "vector",
        "sourceAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VectorizationRequest.source",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: required only for image_to_svg; project-owned raster asset",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Source asset id", "text"),
    // apps/rust-api/src/dto.rs:1036
    OptionSpec::new(
        "vector.prompt",
        "vector",
        "prompt",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "VectorRequest.prompt",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: trimmed prompt <=4000 characters; required for text_to_svg; image_to_svg text guidance must be supported by model.vector.acceptsTextGuidance",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Prompt", "text"),
    // apps/rust-api/src/dto.rs:1038
    OptionSpec::new(
        "vector.sampling",
        "vector",
        "sampling",
        Shape::Object("VectorSampling"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "VectorRequest.sampling",
    )
    .applies(Predicate::Modality("vector"))
    .compat("Structured container; absent object and absent members use the DTO Default implementation")
    .ui("Sampling", "object"),
    // apps/rust-api/src/dto.rs:1040
    OptionSpec::new(
        "vector.detailBudget",
        "vector",
        "detailBudget",
        Shape::Object("VectorDetailBudget"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "VectorRequest.detail_budget",
    )
    .applies(Predicate::Modality("vector"))
    .compat("Structured container; absent object and absent members use the DTO Default implementation")
    .ui("Detail budget", "object"),
    // apps/rust-api/src/dto.rs:1042
    OptionSpec::new(
        "vector.requestedGpu",
        "vector",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Modality("vector"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),

    // vectorPrompt
    // apps/rust-api/src/dto.rs:1051
    OptionSpec::new(
        "vectorPrompt.projectId",
        "vectorPrompt",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:1053
    OptionSpec::new(
        "vectorPrompt.projectName",
        "vectorPrompt",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Modality("vector"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:1054
    OptionSpec::new(
        "vectorPrompt.prompt",
        "vectorPrompt",
        "prompt",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "ImageJobRequest.prompt -> VectorRequest source raster",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: composition validates an ordinary raster image request; 1..4000 characters",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Prompt", "text"),
    // apps/rust-api/src/dto.rs:1056
    OptionSpec::new(
        "vectorPrompt.negativePrompt",
        "vectorPrompt",
        "negativePrompt",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "ImageJobRequest.negative_prompt",
    )
    .applies(Predicate::Modality("vector"))
    .ui("Negative prompt", "text"),
    // apps/rust-api/src/dto.rs:1057
    OptionSpec::new(
        "vectorPrompt.rasterModel",
        "vectorPrompt",
        "rasterModel",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.raster_model",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: distinct resolved catalog model; raster must support text-to-image and vector must support image_to_svg",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Raster model", "text"),
    // apps/rust-api/src/dto.rs:1058
    OptionSpec::new(
        "vectorPrompt.vectorModel",
        "vectorPrompt",
        "vectorModel",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.vector_model",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: distinct resolved catalog model; raster must support text-to-image and vector must support image_to_svg",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Vector model", "text"),
    // apps/rust-api/src/dto.rs:1060
    OptionSpec::new(
        "vectorPrompt.seed",
        "vectorPrompt",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.seed",
    )
    .applies(Predicate::Modality("vector"))
    .ui("Seed", "number"),
    // apps/rust-api/src/dto.rs:1062
    OptionSpec::new(
        "vectorPrompt.width",
        "vectorPrompt",
        "width",
        Shape::Integer,
        DefaultRule::Model("raster model defaults.resolution.width"),
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.width",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: declared geometry limits and required dimension multiple",
        "apps/rust-api/src/validators.rs: outer named dimension 16..4096; selected model floor remains binding",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Width", "number"),
    // apps/rust-api/src/dto.rs:1064
    OptionSpec::new(
        "vectorPrompt.height",
        "vectorPrompt",
        "height",
        Shape::Integer,
        DefaultRule::Model("raster model defaults.resolution.height"),
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.height",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: declared geometry limits and required dimension multiple",
        "apps/rust-api/src/validators.rs: outer named dimension 16..4096; selected model floor remains binding",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Height", "number"),
    // apps/rust-api/src/dto.rs:1066
    OptionSpec::new(
        "vectorPrompt.sampling",
        "vectorPrompt",
        "sampling",
        Shape::Object("VectorSampling"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.sampling",
    )
    .applies(Predicate::Modality("vector"))
    .compat("Structured container; absent object and absent members use the DTO Default implementation")
    .ui("Sampling", "object"),
    // apps/rust-api/src/dto.rs:1068
    OptionSpec::new(
        "vectorPrompt.detailBudget",
        "vectorPrompt",
        "detailBudget",
        Shape::Object("VectorDetailBudget"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "VectorPromptWorkflowRequest.detail_budget",
    )
    .applies(Predicate::Modality("vector"))
    .compat("Structured container; absent object and absent members use the DTO Default implementation")
    .ui("Detail budget", "object"),
    // apps/rust-api/src/dto.rs:1070
    OptionSpec::new(
        "vectorPrompt.requestedGpu",
        "vectorPrompt",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Modality("vector"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),
    // apps/rust-api/src/dto.rs:1074
    OptionSpec::new(
        "vectorPrompt.expectedRasterRevision",
        "vectorPrompt",
        "expectedRasterRevision",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "VectorPromptWorkflowRequest.expected_raster_revision",
    )
    .applies(Predicate::Modality("vector"))
    .effects(&[
        Effect { when: "revision differs from resolved catalog", action: "refuse stale replay", target: "vectorPrompt.rasterModel" },
    ])
    .ui("Expected raster revision", "text"),
    // apps/rust-api/src/dto.rs:1076
    OptionSpec::new(
        "vectorPrompt.expectedVectorRevision",
        "vectorPrompt",
        "expectedVectorRevision",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "VectorPromptWorkflowRequest.expected_vector_revision",
    )
    .applies(Predicate::Modality("vector"))
    .effects(&[
        Effect { when: "revision differs from resolved catalog", action: "refuse stale replay", target: "vectorPrompt.vectorModel" },
    ])
    .ui("Expected vector revision", "text"),

    // vqa
    // apps/rust-api/src/dto.rs:1082
    OptionSpec::new(
        "vqa.projectId",
        "vqa",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Capability("vqa"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:1084
    OptionSpec::new(
        "vqa.projectName",
        "vqa",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Capability("vqa"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:1085
    OptionSpec::new(
        "vqa.sourceAssetId",
        "vqa",
        "sourceAssetId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "Conditioning::Image",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: nonblank project image asset id",
    ])
    .applies(Predicate::Capability("vqa"))
    .ui("Source asset id", "text"),
    // apps/rust-api/src/dto.rs:1086
    OptionSpec::new(
        "vqa.question",
        "vqa",
        "question",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "GenerationRequest.prompt",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: trimmed 1..4000 characters",
    ])
    .applies(Predicate::Capability("vqa"))
    .ui("Question", "text"),
    // apps/rust-api/src/dto.rs:1088
    OptionSpec::new(
        "vqa.model",
        "vqa",
        "model",
        Shape::Text,
        DefaultRule::Literal("\"sensenova_u1_8b\""),
        IntentClass::WhatToMake,
        "VqaRequest.model",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_model_id: non-empty plain path component; resolved catalog/provider must support the request",
    ])
    .applies(Predicate::Capability("vqa"))
    .ui("Model", "text"),
    // apps/rust-api/src/dto.rs:1090
    OptionSpec::new(
        "vqa.maxNewTokens",
        "vqa",
        "maxNewTokens",
        Shape::Integer,
        DefaultRule::Literal("256"),
        IntentClass::WhatToMake,
        "VqaRequest.max_new_tokens",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: integer 16..2048",
    ])
    .applies(Predicate::Capability("vqa"))
    .ui("Max new tokens", "number"),
    // apps/rust-api/src/dto.rs:1092
    OptionSpec::new(
        "vqa.requestedGpu",
        "vqa",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Capability("vqa"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),
    // apps/rust-api/src/dto.rs:1094
    OptionSpec::new(
        "vqa.advanced",
        "vqa",
        "advanced",
        Shape::Object("extension container; leaf options are declared separately"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "VqaRequest.advanced",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_prompt_extras serialized advanced size guard",
    ])
    .applies(Predicate::Capability("vqa"))
    .compat("Carrier only; do not present the object as a scalar option")
    .ui("Advanced", "object"),

    // interleave
    // apps/rust-api/src/dto.rs:1108
    OptionSpec::new(
        "interleave.projectId",
        "interleave",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:1110
    OptionSpec::new(
        "interleave.projectName",
        "interleave",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Capability("interleave"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:1111
    OptionSpec::new(
        "interleave.prompt",
        "interleave",
        "prompt",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "InterleaveRequest.prompt",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: trimmed nonblank; <=4000 characters",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Prompt", "text"),
    // apps/rust-api/src/dto.rs:1114
    OptionSpec::new(
        "interleave.sourceAssetIds",
        "interleave",
        "sourceAssetIds",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "InterleaveRequest.source_asset_ids",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: no blank ids; project-owned image conditioning",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Source asset ids", "list"),
    // apps/rust-api/src/dto.rs:1116
    OptionSpec::new(
        "interleave.model",
        "interleave",
        "model",
        Shape::Text,
        DefaultRule::Literal("\"sensenova_u1_8b\""),
        IntentClass::WhatToMake,
        "InterleaveRequest.model",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_model_id: non-empty plain path component; resolved catalog/provider must support the request",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Model", "text"),
    // apps/rust-api/src/dto.rs:1118
    OptionSpec::new(
        "interleave.maxImages",
        "interleave",
        "maxImages",
        Shape::Integer,
        DefaultRule::Literal("6"),
        IntentClass::WhatToMake,
        "InterleaveRequest.max_images",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: integer 1..10",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Max images", "number"),
    // apps/rust-api/src/dto.rs:1120
    OptionSpec::new(
        "interleave.width",
        "interleave",
        "width",
        Shape::Integer,
        DefaultRule::Literal("1024"),
        IntentClass::WhatToMake,
        "InterleaveRequest.width",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: dimension 256..4096",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Width", "number"),
    // apps/rust-api/src/dto.rs:1122
    OptionSpec::new(
        "interleave.height",
        "interleave",
        "height",
        Shape::Integer,
        DefaultRule::Literal("1024"),
        IntentClass::WhatToMake,
        "InterleaveRequest.height",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: dimension 256..4096",
    ])
    .applies(Predicate::Capability("interleave"))
    .ui("Height", "number"),
    // apps/rust-api/src/dto.rs:1124
    OptionSpec::new(
        "interleave.seed",
        "interleave",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "InterleaveRequest.seed",
    )
    .applies(Predicate::Capability("interleave"))
    .ui("Seed", "number"),
    // apps/rust-api/src/dto.rs:1126
    OptionSpec::new(
        "interleave.requestedGpu",
        "interleave",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Capability("interleave"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),
    // apps/rust-api/src/dto.rs:1128
    OptionSpec::new(
        "interleave.advanced",
        "interleave",
        "advanced",
        Shape::Object("extension container; leaf options are declared separately"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "InterleaveRequest.advanced",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_prompt_extras serialized advanced size guard",
    ])
    .applies(Predicate::Capability("interleave"))
    .compat("Carrier only; do not present the object as a scalar option")
    .ui("Advanced", "object"),

    // video
    // apps/rust-api/src/dto.rs:1142
    OptionSpec::new(
        "video.projectId",
        "video",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:1144
    OptionSpec::new(
        "video.appearance",
        "video",
        "appearance",
        Shape::Object("AppearanceSelection"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VideoRequest.appearance",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: resolve_appearance_for_family validates catalog selections",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "selection supplied", action: "compose family photography directives and negative text", target: "video.prompt" },
    ])
    .ui("Appearance", "object"),
    // apps/rust-api/src/dto.rs:1146
    OptionSpec::new(
        "video.projectName",
        "video",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Modality("video"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:1148
    OptionSpec::new(
        "video.mode",
        "video",
        "mode",
        Shape::OneOf("crates/sceneworks-core/src/video_request.rs::VIDEO_JOB_MODES"),
        DefaultRule::Literal("\"image_to_video\""),
        IntentClass::WhatToMake,
        "VideoRequest.mode",
    )
    .limits(&[
        "crates/sceneworks-core/src/video_request.rs: VIDEO_JOB_MODES",
        "apps/rust-api/src/validators.rs: each mode requires its distinct source/reference media",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Mode", "select"),
    // apps/rust-api/src/dto.rs:1149
    OptionSpec::new(
        "video.prompt",
        "video",
        "prompt",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "GenerationRequest.prompt",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: 1..4000 characters",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Prompt", "text"),
    // apps/rust-api/src/dto.rs:1151
    OptionSpec::new(
        "video.negativePrompt",
        "video",
        "negativePrompt",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "GenerationRequest.negative_prompt",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_prompt_extras length guard",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Negative prompt", "text"),
    // apps/rust-api/src/dto.rs:1153
    OptionSpec::new(
        "video.model",
        "video",
        "model",
        Shape::Text,
        DefaultRule::Literal("\"ltx_2_3\""),
        IntentClass::WhatToMake,
        "VideoRequest.model",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_model_id: non-empty plain path component; resolved catalog/provider must support the request",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Model", "text"),
    // apps/rust-api/src/dto.rs:1157
    OptionSpec::new(
        "video.expectedImportedModelRef",
        "video",
        "expectedImportedModelRef",
        Shape::Object("ImportedModelRef"),
        DefaultRule::Absent,
        IntentClass::Context,
        "imported model replay identity check",
    )
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "replay identity differs from resolved catalog", action: "refuse stale imported model identity", target: "video.model" },
    ])
    .ui("Expected imported model ref", "object"),
    // apps/rust-api/src/dto.rs:1167
    OptionSpec::new(
        "video.duration",
        "video",
        "duration",
        Shape::Number,
        DefaultRule::Model("defaults.duration"),
        IntentClass::WhatToMake,
        "GenerationRequest.video.duration",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 1..30 seconds outer bound",
        "crates/sceneworks-core/src/video_request.rs: limits.hardMaxDuration and model duration contract",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Duration", "number"),
    // apps/rust-api/src/dto.rs:1179
    OptionSpec::new(
        "video.fps",
        "video",
        "fps",
        Shape::Integer,
        DefaultRule::Model("defaults.fps"),
        IntentClass::WhatToMake,
        "GenerationRequest.video.fps",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: 1..60 outer bound",
        "crates/sceneworks-core/src/video_request.rs: limits.fps declared choices",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Fps", "number"),
    // apps/rust-api/src/dto.rs:1185
    OptionSpec::new(
        "video.width",
        "video",
        "width",
        Shape::Integer,
        DefaultRule::Model("defaults.resolution.width"),
        IntentClass::WhatToMake,
        "VideoRequest.width",
    )
    .limits(&[
        "crates/sceneworks-core/src/video_request.rs: declared geometry limits and required dimension multiple",
        "apps/rust-api/src/validators.rs: outer named dimension 256..1920",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Width", "number"),
    // apps/rust-api/src/dto.rs:1187
    OptionSpec::new(
        "video.height",
        "video",
        "height",
        Shape::Integer,
        DefaultRule::Model("defaults.resolution.height"),
        IntentClass::WhatToMake,
        "VideoRequest.height",
    )
    .limits(&[
        "crates/sceneworks-core/src/video_request.rs: declared geometry limits and required dimension multiple",
        "apps/rust-api/src/validators.rs: outer named dimension 256..1920",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Height", "number"),
    // apps/rust-api/src/dto.rs:1189
    OptionSpec::new(
        "video.quality",
        "video",
        "quality",
        Shape::Text,
        DefaultRule::Literal("\"balanced\""),
        IntentClass::WhatToMake,
        "VideoRequest.quality",
    )
    .applies(Predicate::Modality("video"))
    .ui("Quality", "text"),
    // apps/rust-api/src/dto.rs:1191
    OptionSpec::new(
        "video.seed",
        "video",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VideoRequest.seed",
    )
    .applies(Predicate::Modality("video"))
    .ui("Seed", "number"),
    // apps/rust-api/src/dto.rs:1193
    OptionSpec::new(
        "video.recipePresetId",
        "video",
        "recipePresetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "VideoRequest.recipe_preset_id",
    )
    .applies(Predicate::Modality("video"))
    .ui("Recipe preset id", "text"),
    // apps/rust-api/src/dto.rs:1197
    OptionSpec::new(
        "video.presetLorasResolvedClientSide",
        "video",
        "presetLorasResolvedClientSide",
        Shape::Bool,
        DefaultRule::Absent,
        IntentClass::Context,
        "VideoRequest.preset_loras_resolved_client_side",
    )
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "true", action: "skip server preset LoRA merge", target: "video.recipePresetId" },
    ])
    .compat("Public transport coordination flag; preserve client-resolved intent")
    .ui("Preset loras resolved client side", "toggle"),
    // apps/rust-api/src/dto.rs:1201
    OptionSpec::new(
        "video.presetPromptResolvedClientSide",
        "video",
        "presetPromptResolvedClientSide",
        Shape::Bool,
        DefaultRule::Absent,
        IntentClass::Context,
        "VideoRequest.preset_prompt_resolved_client_side",
    )
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "true", action: "skip server preset prompt composition", target: "video.recipePresetId" },
    ])
    .compat("Public transport coordination flag; preserve client-resolved intent")
    .ui("Preset prompt resolved client side", "toggle"),
    // apps/rust-api/src/dto.rs:1203
    OptionSpec::new(
        "video.loras",
        "video",
        "loras",
        Shape::List("LoRA selection { id, weight }"),
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "GenerationRequest.loras",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 10 entries; crates/sceneworks-core/src/lora_family.rs: selected model/adapter family compatibility",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Loras", "list"),
    // apps/rust-api/src/dto.rs:1205
    OptionSpec::new(
        "video.characterId",
        "video",
        "characterId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VideoRequest.character_id",
    )
    .applies(Predicate::Modality("video"))
    .ui("Character id", "text"),
    // apps/rust-api/src/dto.rs:1207
    OptionSpec::new(
        "video.characterLookId",
        "video",
        "characterLookId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VideoRequest.character_look_id",
    )
    .applies(Predicate::Modality("video"))
    .ui("Character look id", "text"),
    // apps/rust-api/src/dto.rs:1209
    OptionSpec::new(
        "video.personTrackId",
        "video",
        "personTrackId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VideoRequest.person_track_id",
    )
    .applies(Predicate::Modality("video"))
    .ui("Person track id", "text"),
    // apps/rust-api/src/dto.rs:1211
    OptionSpec::new(
        "video.replacementMode",
        "video",
        "replacementMode",
        Shape::Text,
        DefaultRule::Literal("\"face_only\""),
        IntentClass::WhatToMake,
        "VideoRequest.replacement_mode",
    )
    .applies(Predicate::Modality("video"))
    .ui("Replacement mode", "text"),
    // apps/rust-api/src/dto.rs:1213
    OptionSpec::new(
        "video.sourceAssetId",
        "video",
        "sourceAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "first-frame Conditioning::Image",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: per-mode required assets; crates/sceneworks-core/src/video_request.rs: model conditioning capabilities",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Source asset id", "text"),
    // apps/rust-api/src/dto.rs:1220
    OptionSpec::new(
        "video.fitMode",
        "video",
        "fitMode",
        Shape::OneOf("crop | pad | outpaint | stretch"),
        DefaultRule::Literal("\"crop\""),
        IntentClass::WhatToMake,
        "VideoRequest.fit_mode",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: normalize_fit_mode; unsupported/unknown values fall back to crop; outpaint requires inpainting capability",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "unknown token", action: "normalize to crop", target: "video.fitMode" },
    ])
    .ui("Fit mode", "select"),
    // apps/rust-api/src/dto.rs:1222
    OptionSpec::new(
        "video.lastFrameAssetId",
        "video",
        "lastFrameAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "last-frame conditioning",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: per-mode required assets; crates/sceneworks-core/src/video_request.rs: model conditioning capabilities",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Last frame asset id", "text"),
    // apps/rust-api/src/dto.rs:1224
    OptionSpec::new(
        "video.sourceClipAssetId",
        "video",
        "sourceClipAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "source clip conditioning",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: per-mode required assets; crates/sceneworks-core/src/video_request.rs: model conditioning capabilities",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Source clip asset id", "text"),
    // apps/rust-api/src/dto.rs:1230
    OptionSpec::new(
        "video.sourceClipAssetIds",
        "video",
        "sourceClipAssetIds",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "Conditioning::SourceClip",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 8 nonblank ids",
        "crates/sceneworks-core/src/video_request.rs: limits.maxSourceClipAssets; limits.maxCombinedReferenceAssets only when declared",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Source clip asset ids", "list"),
    // apps/rust-api/src/dto.rs:1232
    OptionSpec::new(
        "video.bridgeRightClipAssetId",
        "video",
        "bridgeRightClipAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "right bridge clip conditioning",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: per-mode required assets; crates/sceneworks-core/src/video_request.rs: model conditioning capabilities",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Bridge right clip asset id", "text"),
    // apps/rust-api/src/dto.rs:1238
    OptionSpec::new(
        "video.referenceAssetIds",
        "video",
        "referenceAssetIds",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "Conditioning::Reference",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 9 nonblank ids",
        "crates/sceneworks-core/src/video_request.rs: limits.maxReferenceAssets; limits.maxCombinedReferenceAssets only when declared",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Reference asset ids", "list"),
    // apps/rust-api/src/dto.rs:1243
    OptionSpec::new(
        "video.referenceClipAssetId",
        "video",
        "referenceClipAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "reference video conditioning",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: per-mode required assets; crates/sceneworks-core/src/video_request.rs: model conditioning capabilities",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Reference clip asset id", "text"),
    // apps/rust-api/src/dto.rs:1258
    OptionSpec::new(
        "video.referenceAudioAssetIds",
        "video",
        "referenceAudioAssetIds",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "Conditioning::ReferenceAudio",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 3 nonblank ids",
        "crates/sceneworks-core/src/video_request.rs: limits.maxReferenceAudioAssets; limits.maxCombinedReferenceAssets only when declared",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Reference audio asset ids", "list"),
    // apps/rust-api/src/dto.rs:1260
    OptionSpec::new(
        "video.requestedGpu",
        "video",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),
    // apps/rust-api/src/dto.rs:1262
    OptionSpec::new(
        "video.advanced",
        "video",
        "advanced",
        Shape::Object("extension container; leaf options are declared separately"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "VideoRequest.advanced",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_prompt_extras serialized advanced size guard",
    ])
    .applies(Predicate::Modality("video"))
    .compat("Carrier only; do not present the object as a scalar option")
    .ui("Advanced", "object"),

    // audio.script[]
    // apps/rust-api/src/dto.rs:1273
    OptionSpec::new(
        "audio.script[].text",
        "audio.script[]",
        "text",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "AudioParams.script[].text",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: nonblank text; sum across segments <=4000 characters",
    ])
    .applies(Predicate::ModelField("audio.supportsMultiSpeaker"))
    .ui("Text", "text"),
    // apps/rust-api/src/dto.rs:1275
    OptionSpec::new(
        "audio.script[].speaker",
        "audio.script[]",
        "speaker",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "AudioParams.script[].speaker",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 64 characters",
    ])
    .applies(Predicate::ModelField("audio.supportsMultiSpeaker"))
    .ui("Speaker", "text"),
    // apps/rust-api/src/dto.rs:1277
    OptionSpec::new(
        "audio.script[].style",
        "audio.script[]",
        "style",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "AudioParams.script[].style",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 64 characters",
    ])
    .applies(Predicate::ModelField("audio.supportsMultiSpeaker"))
    .ui("Style", "text"),

    // audio
    // apps/rust-api/src/dto.rs:1295
    OptionSpec::new(
        "audio.projectId",
        "audio",
        "projectId",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "Job.project_id",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty project id; project-scoped asset access",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Project id", "text"),
    // apps/rust-api/src/dto.rs:1297
    OptionSpec::new(
        "audio.projectName",
        "audio",
        "projectName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "Job.project_name",
    )
    .applies(Predicate::Modality("audio"))
    .ui("Project name", "text"),
    // apps/rust-api/src/dto.rs:1298
    OptionSpec::new(
        "audio.prompt",
        "audio",
        "prompt",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "GenerationRequest.prompt",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 4000 characters; nonblank unless non-empty script supplied",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Prompt", "text"),
    // apps/rust-api/src/dto.rs:1300
    OptionSpec::new(
        "audio.model",
        "audio",
        "model",
        Shape::Text,
        DefaultRule::Literal("\"kokoro_82m\""),
        IntentClass::WhatToMake,
        "GenerationRequest.model",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_model_id: non-empty plain path component; resolved catalog/provider must support the request",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Model", "text"),
    // apps/rust-api/src/dto.rs:1304
    OptionSpec::new(
        "audio.voice",
        "audio",
        "voice",
        Shape::Text,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.voice",
    )
    .limits(&[
        "crates/sceneworks-worker/src/audio_jobs.rs: advertised audio.voices; omitted uses model default voice",
    ])
    .applies(Predicate::ModelField("audio.voices"))
    .ui("Voice", "text"),
    // apps/rust-api/src/dto.rs:1309
    OptionSpec::new(
        "audio.language",
        "audio",
        "language",
        Shape::Text,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.language",
    )
    .limits(&[
        "crates/sceneworks-worker/src/audio_jobs.rs: advertised audio.languages",
    ])
    .applies(Predicate::ModelField("audio.languages"))
    .effects(&[
        Effect { when: "language supplied", action: "lowercase for generator", target: "AudioParams.language" },
    ])
    .ui("Language", "text"),
    // apps/rust-api/src/dto.rs:1313
    OptionSpec::new(
        "audio.targetDurationSecs",
        "audio",
        "targetDurationSecs",
        Shape::Number,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.target_duration",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..300 seconds outer bound",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected model audio.maxDurationSecs; absent uses natural length",
    ])
    .applies(Predicate::ModelField("audio.maxDurationSecs"))
    .ui("Target duration secs", "number"),
    // apps/rust-api/src/dto.rs:1323
    OptionSpec::new(
        "audio.script",
        "audio",
        "script",
        Shape::List("SpeechSegmentDto"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "AudioParams.script",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: non-empty list; sum of segment text <=4000 characters; model maxSpeakers",
    ])
    .applies(Predicate::ModelField("audio.supportsMultiSpeaker"))
    .ui("Script", "list"),
    // apps/rust-api/src/dto.rs:1329
    OptionSpec::new(
        "audio.guidance",
        "audio",
        "guidance",
        Shape::Number,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "GenerationRequest.guidance",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite; no blanket numeric range; segmented lyrics additionally requires nonnegative guidance",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsGuidance"))
    .ui("Guidance", "number"),
    // apps/rust-api/src/dto.rs:1332
    OptionSpec::new(
        "audio.schedulerShift",
        "audio",
        "schedulerShift",
        Shape::Number,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "GenerationRequest.scheduler_shift",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: positive finite",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsSchedulerShift"))
    .ui("Scheduler shift", "number"),
    // apps/rust-api/src/dto.rs:1338
    OptionSpec::new(
        "audio.steps",
        "audio",
        "steps",
        Shape::Integer,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "GenerationRequest.steps",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: integer 1..10000",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Steps", "number"),
    // apps/rust-api/src/dto.rs:1345
    OptionSpec::new(
        "audio.negativePrompt",
        "audio",
        "negativePrompt",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "GenerationRequest.negative_prompt",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 4000 characters; selected model must support negative prompt",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsNegativePrompt"))
    .ui("Negative prompt", "text"),
    // apps/rust-api/src/dto.rs:1349
    OptionSpec::new(
        "audio.bpm",
        "audio",
        "bpm",
        Shape::Number,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.bpm",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0 < bpm <=1000",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Bpm", "number"),
    // apps/rust-api/src/dto.rs:1353
    OptionSpec::new(
        "audio.musicalKey",
        "audio",
        "musicalKey",
        Shape::Text,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.musical_key",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 64 characters",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Musical key", "text"),
    // apps/rust-api/src/dto.rs:1357
    OptionSpec::new(
        "audio.lyrics",
        "audio",
        "lyrics",
        Shape::Text,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.lyrics",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: at most 4000 characters",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Lyrics", "text"),
    // apps/rust-api/src/dto.rs:1362
    OptionSpec::new(
        "audio.sourceAudioAssetId",
        "audio",
        "sourceAudioAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::AudioEdit.source",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: requires editMode; cannot combine with ICL reference",
    ])
    .applies(Predicate::ModelField("audio.editModes"))
    .ui("Source audio asset id", "text"),
    // apps/rust-api/src/dto.rs:1367
    OptionSpec::new(
        "audio.editMode",
        "audio",
        "editMode",
        Shape::OneOf("inpaint | repaint | extend | cover"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::AudioEdit.mode",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: case-insensitive token; sourceAudioAssetId and editMode required together; model audio.editModes",
    ])
    .applies(Predicate::ModelField("audio.editModes"))
    .ui("Edit mode", "select"),
    // apps/rust-api/src/dto.rs:1371
    OptionSpec::new(
        "audio.editRegionStartSecs",
        "audio",
        "editRegionStartSecs",
        Shape::Number,
        DefaultRule::Provider("source clip / edit mode default"),
        IntentClass::WhatToMake,
        "Conditioning::AudioEdit.region",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..3600 seconds; end > start when both supplied; worker validates source clip range",
    ])
    .applies(Predicate::ModelField("audio.editModes"))
    .ui("Edit region start secs", "number"),
    // apps/rust-api/src/dto.rs:1375
    OptionSpec::new(
        "audio.editRegionEndSecs",
        "audio",
        "editRegionEndSecs",
        Shape::Number,
        DefaultRule::Provider("source clip / edit mode default"),
        IntentClass::WhatToMake,
        "Conditioning::AudioEdit.region",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..3600 seconds; end > start when both supplied; worker validates source clip range",
    ])
    .applies(Predicate::ModelField("audio.editModes"))
    .ui("Edit region end secs", "number"),
    // apps/rust-api/src/dto.rs:1378
    OptionSpec::new(
        "audio.editStrength",
        "audio",
        "editStrength",
        Shape::Number,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.edit_strength",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..1",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.editModes"))
    .ui("Edit strength", "number"),
    // apps/rust-api/src/dto.rs:1385
    OptionSpec::new(
        "audio.referenceAudioAssetId",
        "audio",
        "referenceAudioAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "run_voice_clone_synthesis reference audio",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: cannot combine voice clone reference with iclMode",
    ])
    .applies(Predicate::Modality("audio"))
    .effects(&[
        Effect { when: "non-empty reference asset", action: "route through base TTS then voice conversion", target: "audio.baseModel" },
    ])
    .ui("Reference audio asset id", "text"),
    // apps/rust-api/src/dto.rs:1390
    OptionSpec::new(
        "audio.baseModel",
        "audio",
        "baseModel",
        Shape::Text,
        DefaultRule::Literal("\"kokoro_82m\""),
        IntentClass::WhatToMake,
        "voice clone base TTS generator",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: plain model id; selected catalog model must be audio",
    ])
    .applies(Predicate::All(&[
        Predicate::Any(&[Predicate::ModelContains("audio.conditioning", "ReferenceAudio"), Predicate::ModelContains("audio.conditioning", "VoiceEmbedding")]),
        Predicate::Not(&Predicate::ModelField("audio.supportsSegmentedLyrics")),
        Predicate::Not(&Predicate::ModelField("audio.supportsSymbolicSong")),
    ]))
    .ui("Base model", "text"),
    // apps/rust-api/src/dto.rs:1395
    OptionSpec::new(
        "audio.matchStrength",
        "audio",
        "matchStrength",
        Shape::Number,
        DefaultRule::Provider("voice converter posterior temperature; OpenVoice V2 default 0.3"),
        IntentClass::WhatToMake,
        "AudioTransformRequest.strength",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..1",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::All(&[
        Predicate::Any(&[Predicate::ModelContains("audio.conditioning", "ReferenceAudio"), Predicate::ModelContains("audio.conditioning", "VoiceEmbedding")]),
        Predicate::Not(&Predicate::ModelField("audio.supportsSegmentedLyrics")),
        Predicate::Not(&Predicate::ModelField("audio.supportsSymbolicSong")),
    ]))
    .ui("Match strength", "number"),
    // apps/rust-api/src/dto.rs:1402
    OptionSpec::new(
        "audio.segments",
        "audio",
        "segments",
        Shape::Integer,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.segments",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: integer 1..100",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsSegmentedLyrics"))
    .ui("Segments", "number"),
    // apps/rust-api/src/dto.rs:1404
    OptionSpec::new(
        "audio.maxNewTokensPerSegment",
        "audio",
        "maxNewTokensPerSegment",
        Shape::Integer,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.max_new_tokens_per_segment",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: integer 1..16382",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsSegmentedLyrics"))
    .ui("Max new tokens per segment", "number"),
    // apps/rust-api/src/dto.rs:1406
    OptionSpec::new(
        "audio.repetitionPenalty",
        "audio",
        "repetitionPenalty",
        Shape::Number,
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.repetition_penalty",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0 < value <=10",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsRepetitionPenalty"))
    .ui("Repetition penalty", "number"),
    // apps/rust-api/src/dto.rs:1411
    OptionSpec::new(
        "audio.guidanceEnabled",
        "audio",
        "guidanceEnabled",
        Shape::Bool,
        DefaultRule::Inherit("audio.guidance / provider schedule"),
        IntentClass::WhatToMake,
        "AudioParams.guidance_enabled",
    )
    .applies(Predicate::ModelField("audio.supportsSegmentedLyrics"))
    .effects(&[
        Effect { when: "false", action: "set guidance to 0.0; refuse simultaneous explicit guidance", target: "GenerationRequest.guidance" },
    ])
    .ui("Guidance enabled", "toggle"),
    // apps/rust-api/src/dto.rs:1417
    OptionSpec::new(
        "audio.iclMode",
        "audio",
        "iclMode",
        Shape::OneOf("single | dual"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::ReferenceAudio reference mode",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: mode and reference slots are exclusive; no voice clone or source edit reference",
    ])
    .applies(Predicate::All(&[Predicate::ModelField("audio.supportsSegmentedLyrics"), Predicate::ModelContains("audio.conditioning", "ReferenceAudio")]))
    .effects(&[
        Effect { when: "single", action: "require only iclReferenceAssetId", target: "audio.iclReferenceAssetId" },
        Effect { when: "dual", action: "require vocal and instrumental ids, exclude mixed reference", target: "audio.iclVocalAssetId + audio.iclInstrumentalAssetId" },
    ])
    .ui("Icl mode", "select"),
    // apps/rust-api/src/dto.rs:1419
    OptionSpec::new(
        "audio.iclReferenceAssetId",
        "audio",
        "iclReferenceAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::ReferenceAudio.iclReferenceAssetId",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: single mode uses mixed id; dual uses both vocal/instrumental ids; all ids project-scoped",
    ])
    .applies(Predicate::ModelContains("audio.conditioning", "ReferenceAudio"))
    .ui("Icl reference asset id", "text"),
    // apps/rust-api/src/dto.rs:1421
    OptionSpec::new(
        "audio.iclVocalAssetId",
        "audio",
        "iclVocalAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::ReferenceAudio.iclVocalAssetId",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: single mode uses mixed id; dual uses both vocal/instrumental ids; all ids project-scoped",
    ])
    .applies(Predicate::ModelContains("audio.conditioning", "ReferenceAudio"))
    .ui("Icl vocal asset id", "text"),
    // apps/rust-api/src/dto.rs:1423
    OptionSpec::new(
        "audio.iclInstrumentalAssetId",
        "audio",
        "iclInstrumentalAssetId",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "Conditioning::ReferenceAudio.iclInstrumentalAssetId",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: single mode uses mixed id; dual uses both vocal/instrumental ids; all ids project-scoped",
    ])
    .applies(Predicate::ModelContains("audio.conditioning", "ReferenceAudio"))
    .ui("Icl instrumental asset id", "text"),
    // apps/rust-api/src/dto.rs:1427
    OptionSpec::new(
        "audio.iclStartSecs",
        "audio",
        "iclStartSecs",
        Shape::Number,
        DefaultRule::Provider("reference window; absent start 0, absent end uses provider window/clip end"),
        IntentClass::WhatToMake,
        "AudioParams.reference_region.iclStartSecs",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..3600; must be less than effective end (30 seconds when omitted)",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsReferenceRegion"))
    .ui("Icl start secs", "number"),
    // apps/rust-api/src/dto.rs:1429
    OptionSpec::new(
        "audio.iclEndSecs",
        "audio",
        "iclEndSecs",
        Shape::Number,
        DefaultRule::Provider("reference window; absent start 0, absent end uses provider window/clip end"),
        IntentClass::WhatToMake,
        "AudioParams.reference_region.iclEndSecs",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: finite 0..3600; must be greater than supplied start",
        "crates/sceneworks-worker/src/audio_jobs.rs: selected provider capability/validation remains binding",
    ])
    .applies(Predicate::ModelField("audio.supportsReferenceRegion"))
    .ui("Icl end secs", "number"),
    // apps/rust-api/src/dto.rs:1434
    OptionSpec::new(
        "audio.quantTier",
        "audio",
        "quantTier",
        Shape::OneOf("bf16 | q8 | q4"),
        DefaultRule::Model("default installed tier, otherwise first installed tier"),
        IntentClass::MachineBudget,
        "LoadSpec quantization / model download variant",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: tier must appear in selected model downloads[].variant excluding coRequisite entries",
    ])
    .applies(Predicate::Modality("audio"))
    .ui("Quant tier", "select"),
    // apps/rust-api/src/dto.rs:1439
    OptionSpec::new(
        "audio.outputLimiter",
        "audio",
        "outputLimiter",
        Shape::OneOf("clamp | rescale"),
        DefaultRule::Provider("selected audio provider default; omission remains absent"),
        IntentClass::WhatToMake,
        "AudioParams.output_limiter",
    )
    .applies(Predicate::ModelField("audio.supportsOutputLimiter"))
    .ui("Output limiter", "select"),
    // apps/rust-api/src/dto.rs:1441
    OptionSpec::new(
        "audio.seed",
        "audio",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "GenerationRequest.seed",
    )
    .applies(Predicate::Modality("audio"))
    .ui("Seed", "number"),
    // apps/rust-api/src/dto.rs:1443
    OptionSpec::new(
        "audio.requestedGpu",
        "audio",
        "requestedGpu",
        Shape::Text,
        DefaultRule::Literal("\"auto\""),
        IntentClass::MachineBudget,
        "Job.requested_gpu -> worker routing",
    )
    .applies(Predicate::Modality("audio"))
    .effects(&[
        Effect { when: "explicit device", action: "route to requested worker/device", target: "Job.requested_gpu" },
    ])
    .ui("Requested gpu", "text"),
    // apps/rust-api/src/dto.rs:1445
    OptionSpec::new(
        "audio.advanced",
        "audio",
        "advanced",
        Shape::Object("extension container; leaf options are declared separately"),
        DefaultRule::Literal("{}"),
        IntentClass::WhatToMake,
        "GenerationRequest.advanced",
    )
    .limits(&[
        "apps/rust-api/src/validators.rs: validate_prompt_extras serialized advanced size guard",
    ])
    .applies(Predicate::Modality("audio"))
    .compat("Carrier only; do not present the object as a scalar option")
    .ui("Advanced", "object"),

    // image.upscale
    // crates/sceneworks-core/src/contracts.rs:815
    OptionSpec::new(
        "image.upscale.enabled",
        "image.upscale",
        "enabled",
        Shape::Bool,
        DefaultRule::Literal("false"),
        IntentClass::WhatToMake,
        "ImageUpscaleRequest.enabled",
    )
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "enabled", action: "clear", target: "advanced.usePid" },
        Effect { when: "enabled", action: "clear", target: "image.hiresFix.enabled" },
    ])
    .ui("Enabled", "toggle"),
    // crates/sceneworks-core/src/contracts.rs:817
    OptionSpec::new(
        "image.upscale.factor",
        "image.upscale",
        "factor",
        Shape::OneOf("2 | 4"),
        DefaultRule::Literal("2"),
        IntentClass::WhatToMake,
        "ImageUpscaleRequest.factor",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: enabled non-Iris engines require factor 2 or 4; Iris refuses non-default factor (use scale)",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Factor", "select"),
    // crates/sceneworks-core/src/contracts.rs:819
    OptionSpec::new(
        "image.upscale.engine",
        "image.upscale",
        "engine",
        Shape::Text,
        DefaultRule::Literal("\"real-esrgan\""),
        IntentClass::WhatToMake,
        "ImageUpscaleRequest.engine",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: enabled engine must not be empty; registered image transformer must support request",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Engine", "text"),

    // image.hiresFix
    // crates/sceneworks-core/src/contracts.rs:928
    OptionSpec::new(
        "image.hiresFix.enabled",
        "image.hiresFix",
        "enabled",
        Shape::Bool,
        DefaultRule::Literal("false"),
        IntentClass::WhatToMake,
        "HiresFixRequest.enabled",
    )
    .limits(&[
        "crates/sceneworks-core/src/image_request.rs: hires_fix_request_error owns model/mode and incompatible-control refusals",
    ])
    .applies(HIRES_FIX)
    .effects(&[
        Effect { when: "true", action: "run second img2img pass after generation", target: "HiresFixRequest" },
        Effect { when: "enabled", action: "clear", target: "advanced.usePid" },
        Effect { when: "enabled", action: "clear", target: "image.upscale.enabled" },
    ])
    .ui("Enabled", "toggle"),
    // crates/sceneworks-core/src/contracts.rs:930
    OptionSpec::new(
        "image.hiresFix.steps",
        "image.hiresFix",
        "steps",
        Shape::Integer,
        DefaultRule::Literal("0"),
        IntentClass::WhatToMake,
        "HiresFixRequest.steps",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: effective_steps: nonzero clamps to 1..150; 0 inherits first-pass steps",
    ])
    .applies(HIRES_FIX)
    .effects(&[
        Effect { when: "0", action: "inherit first-pass steps", target: "GenerationRequest.steps" },
    ])
    .ui("Steps", "number"),
    // crates/sceneworks-core/src/contracts.rs:932
    OptionSpec::new(
        "image.hiresFix.denoisingStrength",
        "image.hiresFix",
        "denoisingStrength",
        Shape::Number,
        DefaultRule::Literal("0.7"),
        IntentClass::WhatToMake,
        "HiresFixRequest.denoising_strength",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: effective_denoising_strength: clamp 0..1",
    ])
    .applies(HIRES_FIX)
    .ui("Denoising strength", "number"),
    // crates/sceneworks-core/src/contracts.rs:934
    OptionSpec::new(
        "image.hiresFix.upscaleBy",
        "image.hiresFix",
        "upscaleBy",
        Shape::Number,
        DefaultRule::Literal("2.0"),
        IntentClass::WhatToMake,
        "HiresFixRequest.upscale_by",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: effective_upscale_by: clamp 1..4",
    ])
    .applies(HIRES_FIX)
    .ui("Upscale by", "number"),
    // crates/sceneworks-core/src/contracts.rs:936
    OptionSpec::new(
        "image.hiresFix.cfgScale",
        "image.hiresFix",
        "cfgScale",
        Shape::Number,
        DefaultRule::Inherit("first-pass guidance"),
        IntentClass::WhatToMake,
        "HiresFixRequest.cfg_scale",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: effective_cfg_scale: finite values clamp 0..60; absent/non-finite inherits first-pass guidance",
    ])
    .applies(HIRES_FIX)
    .ui("Cfg scale", "number"),

    // videoUpscale
    // crates/sceneworks-core/src/contracts.rs:1000
    OptionSpec::new(
        "videoUpscale.sourceAssetId",
        "videoUpscale",
        "sourceAssetId",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.source_asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: non-empty source video required; worker project-scoped asset validation",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Source asset id", "text"),
    // crates/sceneworks-core/src/contracts.rs:1004
    OptionSpec::new(
        "videoUpscale.factor",
        "videoUpscale",
        "factor",
        Shape::OneOf("2 | 4"),
        DefaultRule::Literal("2"),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.factor",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: VideoUpscaleRequest; crates/sceneworks-worker/src/video_jobs/seedvr2.rs: native upscale validation",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "target dimensions provided", action: "explicit target geometry overrides factor", target: "videoUpscale.targetWidth / targetHeight" },
    ])
    .ui("Factor", "select"),
    // crates/sceneworks-core/src/contracts.rs:1007
    OptionSpec::new(
        "videoUpscale.targetWidth",
        "videoUpscale",
        "targetWidth",
        Shape::Integer,
        DefaultRule::Inherit("factor * source width"),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.target_width",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: worker snaps to multiple of 16",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "explicit value", action: "override factor-scaled source dimension and snap to multiple of 16", target: "output width" },
    ])
    .ui("Target width", "number"),
    // crates/sceneworks-core/src/contracts.rs:1010
    OptionSpec::new(
        "videoUpscale.targetHeight",
        "videoUpscale",
        "targetHeight",
        Shape::Integer,
        DefaultRule::Inherit("factor * source height"),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.target_height",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: worker snaps to multiple of 16",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "explicit value", action: "override factor-scaled source dimension and snap to multiple of 16", target: "output height" },
    ])
    .ui("Target height", "number"),
    // crates/sceneworks-core/src/contracts.rs:1013
    OptionSpec::new(
        "videoUpscale.engine",
        "videoUpscale",
        "engine",
        Shape::OneOf("seedvr2 | seedvr2_3b | empty legacy alias"),
        DefaultRule::Literal("\"seedvr2\""),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.engine",
    )
    .limits(&[
        "crates/sceneworks-worker/src/video_jobs/seedvr2.rs: trim and lowercase; other engines refused",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Engine", "select"),
    // crates/sceneworks-core/src/contracts.rs:1016
    OptionSpec::new(
        "videoUpscale.model",
        "videoUpscale",
        "model",
        Shape::Text,
        DefaultRule::Literal("\"seedvr2_3b\""),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.model -> output asset provenance",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: VideoUpscaleRequest; crates/sceneworks-worker/src/video_jobs/seedvr2.rs: native upscale validation",
    ])
    .applies(Predicate::Modality("video"))
    .compat("Current worker resolves fixed SeedVR2 checkpoint; this carrier labels recorded model provenance")
    .ui("Model", "text"),
    // crates/sceneworks-core/src/contracts.rs:1019
    OptionSpec::new(
        "videoUpscale.softness",
        "videoUpscale",
        "softness",
        Shape::Number,
        DefaultRule::Literal("0"),
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.softness",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: VideoUpscaleRequest; crates/sceneworks-worker/src/video_jobs/seedvr2.rs: native upscale validation",
    ])
    .applies(Predicate::Modality("video"))
    .ui("Softness", "number"),
    // crates/sceneworks-core/src/contracts.rs:1022
    OptionSpec::new(
        "videoUpscale.seed",
        "videoUpscale",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VideoUpscaleRequest.seed",
    )
    .applies(Predicate::Modality("video"))
    .ui("Seed", "number"),
    // crates/sceneworks-core/src/contracts.rs:1025
    OptionSpec::new(
        "videoUpscale.projectId",
        "videoUpscale",
        "projectId",
        Shape::Text,
        DefaultRule::Inherit("job.projectId"),
        IntentClass::Context,
        "Job.project_id",
    )
    .applies(Predicate::Modality("video"))
    .ui("Project id", "text"),
    // crates/sceneworks-core/src/contracts.rs:1028
    OptionSpec::new(
        "videoUpscale.displayName",
        "videoUpscale",
        "displayName",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::Context,
        "VideoUpscaleRequest.display_name",
    )
    .applies(Predicate::Modality("video"))
    .ui("Display name", "text"),

    // vector.sampling
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vector.sampling.temperature",
        "vector.sampling",
        "temperature",
        Shape::Number,
        DefaultRule::Literal("0.2"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.temperature -> TextLlmRequest.sampling.temperature",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: finite 0..2",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Temperature", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vector.sampling.topP",
        "vector.sampling",
        "topP",
        Shape::Number,
        DefaultRule::Literal("0.9"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.top_p -> TextLlmRequest.sampling.top_p",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: finite 0 < value <=1",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Top p", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vector.sampling.topK",
        "vector.sampling",
        "topK",
        Shape::Integer,
        DefaultRule::Literal("0"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.top_k -> TextLlmRequest.sampling.top_k",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 0..1000; 0 disables top-k",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Top k", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vector.sampling.repetitionPenalty",
        "vector.sampling",
        "repetitionPenalty",
        Shape::Number,
        DefaultRule::Literal("1.0"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.repetition_penalty -> TextLlmRequest.sampling.repetition_penalty",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: finite 0.1..2",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Repetition penalty", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vector.sampling.repetitionContext",
        "vector.sampling",
        "repetitionContext",
        Shape::Integer,
        DefaultRule::Literal("0"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.repetition_context -> TextLlmRequest.sampling.repetition_context",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 0..32768; 0 uses full generated context",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Repetition context", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vector.sampling.seed",
        "vector.sampling",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.seed -> TextLlmRequest.seed",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: optional u64 seed",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Seed", "number"),

    // vector.detailBudget
    // apps/rust-api/src/dto.rs:1003
    OptionSpec::new(
        "vector.detailBudget.maxNewTokens",
        "vector.detailBudget",
        "maxNewTokens",
        Shape::Integer,
        DefaultRule::Literal("4000"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.detail_budget.max_new_tokens",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 16..32768 outer bound",
        "apps/rust-api/src/generation.rs: must not exceed selected model.vector.maxNewTokens",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Max new tokens", "number"),
    // apps/rust-api/src/dto.rs:1003
    OptionSpec::new(
        "vector.detailBudget.maxSvgBytes",
        "vector.detailBudget",
        "maxSvgBytes",
        Shape::Integer,
        DefaultRule::Literal("262144"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.detail_budget.max_svg_bytes",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 1024..262144 outer bound",
        "apps/rust-api/src/generation.rs: must not exceed selected model.vector.maxSvgBytes",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Max svg bytes", "number"),
    // apps/rust-api/src/dto.rs:1003
    OptionSpec::new(
        "vector.detailBudget.maxWallTimeMs",
        "vector.detailBudget",
        "maxWallTimeMs",
        Shape::Integer,
        DefaultRule::Literal("300000"),
        IntentClass::MachineBudget,
        "VectorProviderRequest.detail_budget.max_wall_time_ms",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 1000..300000 outer bound",
        "apps/rust-api/src/generation.rs: must not exceed selected model.vector.maxWallTimeMs",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Max wall time ms", "number"),

    // vectorPrompt.sampling
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vectorPrompt.sampling.temperature",
        "vectorPrompt.sampling",
        "temperature",
        Shape::Number,
        DefaultRule::Literal("0.2"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.temperature -> TextLlmRequest.sampling.temperature",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: finite 0..2",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Temperature", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vectorPrompt.sampling.topP",
        "vectorPrompt.sampling",
        "topP",
        Shape::Number,
        DefaultRule::Literal("0.9"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.top_p -> TextLlmRequest.sampling.top_p",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: finite 0 < value <=1",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Top p", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vectorPrompt.sampling.topK",
        "vectorPrompt.sampling",
        "topK",
        Shape::Integer,
        DefaultRule::Literal("0"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.top_k -> TextLlmRequest.sampling.top_k",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 0..1000; 0 disables top-k",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Top k", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vectorPrompt.sampling.repetitionPenalty",
        "vectorPrompt.sampling",
        "repetitionPenalty",
        Shape::Number,
        DefaultRule::Literal("1.0"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.repetition_penalty -> TextLlmRequest.sampling.repetition_penalty",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: finite 0.1..2",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Repetition penalty", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vectorPrompt.sampling.repetitionContext",
        "vectorPrompt.sampling",
        "repetitionContext",
        Shape::Integer,
        DefaultRule::Literal("0"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.repetition_context -> TextLlmRequest.sampling.repetition_context",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 0..32768; 0 uses full generated context",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Repetition context", "number"),
    // apps/rust-api/src/dto.rs:979
    OptionSpec::new(
        "vectorPrompt.sampling.seed",
        "vectorPrompt.sampling",
        "seed",
        Shape::Integer,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "VectorProviderRequest.sampling.seed -> TextLlmRequest.seed",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: optional u64 seed",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Seed", "number"),

    // vectorPrompt.detailBudget
    // apps/rust-api/src/dto.rs:1003
    OptionSpec::new(
        "vectorPrompt.detailBudget.maxNewTokens",
        "vectorPrompt.detailBudget",
        "maxNewTokens",
        Shape::Integer,
        DefaultRule::Literal("4000"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.detail_budget.max_new_tokens",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 16..32768 outer bound",
        "apps/rust-api/src/generation.rs: must not exceed selected model.vector.maxNewTokens",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Max new tokens", "number"),
    // apps/rust-api/src/dto.rs:1003
    OptionSpec::new(
        "vectorPrompt.detailBudget.maxSvgBytes",
        "vectorPrompt.detailBudget",
        "maxSvgBytes",
        Shape::Integer,
        DefaultRule::Literal("262144"),
        IntentClass::WhatToMake,
        "VectorProviderRequest.detail_budget.max_svg_bytes",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 1024..262144 outer bound",
        "apps/rust-api/src/generation.rs: must not exceed selected model.vector.maxSvgBytes",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Max svg bytes", "number"),
    // apps/rust-api/src/dto.rs:1003
    OptionSpec::new(
        "vectorPrompt.detailBudget.maxWallTimeMs",
        "vectorPrompt.detailBudget",
        "maxWallTimeMs",
        Shape::Integer,
        DefaultRule::Literal("300000"),
        IntentClass::MachineBudget,
        "VectorProviderRequest.detail_budget.max_wall_time_ms",
    )
    .limits(&[
        "apps/rust-api/src/generation.rs: 1000..300000 outer bound",
        "apps/rust-api/src/generation.rs: must not exceed selected model.vector.maxWallTimeMs",
    ])
    .applies(Predicate::Modality("vector"))
    .ui("Max wall time ms", "number"),

    // image.upscale
    // crates/sceneworks-core/src/contracts.rs:833
    OptionSpec::new(
        "image.upscale.softness",
        "image.upscale",
        "softness",
        Shape::Number,
        DefaultRule::Literal("0.0"),
        IntentClass::WhatToMake,
        "ImageUpscaleRequest.softness()",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: ImageUpscaleRequest.validate",
        "crates/sceneworks-core/src/image_restoration.rs: clamp 0..1; Iris refuses this control",
    ])
    .applies(Predicate::Family("seedvr2"))
    .ui("Softness", "number"),
    // crates/sceneworks-core/src/contracts.rs:833
    OptionSpec::new(
        "image.upscale.scale",
        "image.upscale",
        "scale",
        Shape::Number,
        DefaultRule::Model("Iris restoration declared scale default"),
        IntentClass::WhatToMake,
        "IrisRestoreParams.scale",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: ImageUpscaleRequest.validate",
        "crates/sceneworks-core/src/image_restoration.rs: positive finite; Iris only",
    ])
    .applies(Predicate::Family("iris"))
    .ui("Scale", "number"),
    // crates/sceneworks-core/src/contracts.rs:833
    OptionSpec::new(
        "image.upscale.inputSizing",
        "image.upscale",
        "inputSizing",
        Shape::OneOf("budgeted | original"),
        DefaultRule::Literal("\"budgeted\""),
        IntentClass::WhatToMake,
        "IrisRestoreParams.input_sizing",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: ImageUpscaleRequest.validate",
        "crates/sceneworks-core/src/image_restoration.rs: Iris only; trim and lowercase",
    ])
    .applies(Predicate::Family("iris"))
    .ui("Input sizing", "select"),
    // crates/sceneworks-core/src/contracts.rs:833
    OptionSpec::new(
        "image.upscale.colorFix",
        "image.upscale",
        "colorFix",
        Shape::Bool,
        DefaultRule::Model("Iris restoration declared color correction default"),
        IntentClass::WhatToMake,
        "IrisRestoreParams.color_fix",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: ImageUpscaleRequest.validate",
        "crates/sceneworks-core/src/image_restoration.rs: Iris only",
    ])
    .applies(Predicate::Family("iris"))
    .ui("Color fix", "toggle"),
    // crates/sceneworks-core/src/contracts.rs:833
    OptionSpec::new(
        "image.upscale.irisModel",
        "image.upscale",
        "irisModel",
        Shape::Text,
        DefaultRule::Literal("\"iris_3b_restore\""),
        IntentClass::WhatToMake,
        "trained Iris restoration model export",
    )
    .limits(&[
        "crates/sceneworks-core/src/contracts.rs: ImageUpscaleRequest.validate",
        "crates/sceneworks-core/src/image_restoration.rs: Iris only; safe model id <=128 characters",
    ])
    .applies(Predicate::Family("iris"))
    .ui("Iris model", "text"),

    // image.appearance
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "image.appearance.camera",
        "image.appearance",
        "camera",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "image.prompt" },
    ])
    .ui("Camera", "text"),
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "image.appearance.film",
        "image.appearance",
        "film",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "image.prompt" },
    ])
    .ui("Film", "text"),
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "image.appearance.looks",
        "image.appearance",
        "looks",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "image.prompt" },
    ])
    .ui("Looks", "list"),
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "image.appearance.lighting",
        "image.appearance",
        "lighting",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("image"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "image.prompt" },
    ])
    .ui("Lighting", "text"),

    // video.appearance
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "video.appearance.camera",
        "video.appearance",
        "camera",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "video.prompt" },
    ])
    .ui("Camera", "text"),
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "video.appearance.film",
        "video.appearance",
        "film",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "video.prompt" },
    ])
    .ui("Film", "text"),
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "video.appearance.looks",
        "video.appearance",
        "looks",
        Shape::Strings,
        DefaultRule::Literal("[]"),
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "video.prompt" },
    ])
    .ui("Looks", "list"),
    // crates/sceneworks-core/src/appearance.rs:13
    OptionSpec::new(
        "video.appearance.lighting",
        "video.appearance",
        "lighting",
        Shape::Text,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "resolve_appearance_for_family -> prompt directives/negative prompt",
    )
    .limits(&[
        "crates/sceneworks-core/src/appearance.rs: selection must exist in its catalog category with a variant for the selected model family; duplicate selections are ignored",
    ])
    .applies(Predicate::Modality("video"))
    .effects(&[
        Effect { when: "selected", action: "compose model-family appearance text", target: "video.prompt" },
    ])
    .ui("Lighting", "text"),

    // image.mossImageAdvanced
    // crates/sceneworks-core/src/moss_image_advanced.rs:38
    OptionSpec::new(
        "image.mossImageAdvanced.positive",
        "image.mossImageAdvanced",
        "positive",
        Shape::Object("EmbeddingsRef"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "MossImageAdvancedInputs.positive",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: replaces prompt and prompts; both text carriers must be empty",
    ])
    .applies(Predicate::All(&[Predicate::Family("moss_image"), Predicate::Capability("edit_image")]))
    .ui("Positive", "object"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:38
    OptionSpec::new(
        "image.mossImageAdvanced.negative",
        "image.mossImageAdvanced",
        "negative",
        Shape::Object("EmbeddingsRef"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "MossImageAdvancedInputs.negative",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: mutually exclusive with negativePrompts and non-empty negativePrompt",
    ])
    .applies(Predicate::All(&[Predicate::Family("moss_image"), Predicate::Capability("edit_image")]))
    .ui("Negative", "object"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:38
    OptionSpec::new(
        "image.mossImageAdvanced.negativePrompts",
        "image.mossImageAdvanced",
        "negativePrompts",
        Shape::Strings,
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "MossImageAdvancedInputs.negative_prompts",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: 1..8 strings; each <=4000 characters; excludes negative embeddings and non-empty negativePrompt",
    ])
    .applies(Predicate::All(&[Predicate::Family("moss_image"), Predicate::Capability("edit_image")]))
    .ui("Negative prompts", "list"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:38
    OptionSpec::new(
        "image.mossImageAdvanced.noise",
        "image.mossImageAdvanced",
        "noise",
        Shape::Object("TensorRef"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "MossImageAdvancedInputs.noise",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: exact owned inert NPY noise artifact; source validates shape",
    ])
    .applies(Predicate::All(&[Predicate::Family("moss_image"), Predicate::Capability("edit_image")]))
    .ui("Noise", "object"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:38
    OptionSpec::new(
        "image.mossImageAdvanced.output",
        "image.mossImageAdvanced",
        "output",
        Shape::OneOf("images | latents"),
        DefaultRule::Literal("\"images\""),
        IntentClass::WhatToMake,
        "MossImageAdvancedInputs.output",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: latents refuses inline upscale",
    ])
    .applies(Predicate::All(&[Predicate::Family("moss_image"), Predicate::Capability("edit_image")]))
    .ui("Output", "select"),

    // image.mossImageAdvanced.positive
    // crates/sceneworks-core/src/moss_image_advanced.rs:31
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values",
        "image.mossImageAdvanced.positive",
        "values",
        Shape::Object("TensorRef"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "MossImageEmbeddings.values",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: project-owned artifact identity and exact embeddings shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Values", "object"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:31
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask",
        "image.mossImageAdvanced.positive",
        "mask",
        Shape::Object("TensorRef"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "MossImageEmbeddings.mask",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: project-owned uint8 artifact; shape matches embeddings batch/sequence",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Mask", "object"),

    // image.mossImageAdvanced.negative
    // crates/sceneworks-core/src/moss_image_advanced.rs:31
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values",
        "image.mossImageAdvanced.negative",
        "values",
        Shape::Object("TensorRef"),
        DefaultRule::Required,
        IntentClass::WhatToMake,
        "MossImageEmbeddings.values",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: project-owned artifact identity and exact embeddings shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Values", "object"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:31
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask",
        "image.mossImageAdvanced.negative",
        "mask",
        Shape::Object("TensorRef"),
        DefaultRule::Absent,
        IntentClass::WhatToMake,
        "MossImageEmbeddings.mask",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: project-owned uint8 artifact; shape matches embeddings batch/sequence",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Mask", "object"),

    // image.mossImageAdvanced.positive.values
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.assetId",
        "image.mossImageAdvanced.positive.values",
        "assetId",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "TensorRef.asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: non-empty project-owned tensor asset id; never a caller path",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Asset id", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.sha256",
        "image.mossImageAdvanced.positive.values",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TensorRef.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: exact 64 lowercase hexadecimal SHA256; reverify artifact bytes",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Sha256", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.metadata",
        "image.mossImageAdvanced.positive.values",
        "metadata",
        Shape::Object("TensorMetadata"),
        DefaultRule::Absent,
        IntentClass::Context,
        "TensorRef.metadata",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: optional measured metadata is re-read and compared to artifact; never trusted to authorize identity or shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Metadata", "object"),

    // image.mossImageAdvanced.positive.values.metadata
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.metadata.dtype",
        "image.mossImageAdvanced.positive.values.metadata",
        "dtype",
        Shape::OneOf("float32 | uint8"),
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.dtype",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; actual owned tensor dtype",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Dtype", "select"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.metadata.shape",
        "image.mossImageAdvanced.positive.values.metadata",
        "shape",
        Shape::Numbers,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.shape",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; 2..4 positive integer dimensions; exact model/role tensor shape",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Shape", "list"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.metadata.byteLength",
        "image.mossImageAdvanced.positive.values.metadata",
        "byteLength",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.byte_length",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; positive; at most 536870912 data bytes",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Byte length", "number"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.values.metadata.sha256",
        "image.mossImageAdvanced.positive.values.metadata",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; exact 64 lowercase hexadecimal SHA256",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Sha256", "text"),

    // image.mossImageAdvanced.positive.mask
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.assetId",
        "image.mossImageAdvanced.positive.mask",
        "assetId",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "TensorRef.asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: non-empty project-owned tensor asset id; never a caller path",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Asset id", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.sha256",
        "image.mossImageAdvanced.positive.mask",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TensorRef.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: exact 64 lowercase hexadecimal SHA256; reverify artifact bytes",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Sha256", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.metadata",
        "image.mossImageAdvanced.positive.mask",
        "metadata",
        Shape::Object("TensorMetadata"),
        DefaultRule::Absent,
        IntentClass::Context,
        "TensorRef.metadata",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: optional measured metadata is re-read and compared to artifact; never trusted to authorize identity or shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Metadata", "object"),

    // image.mossImageAdvanced.positive.mask.metadata
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.metadata.dtype",
        "image.mossImageAdvanced.positive.mask.metadata",
        "dtype",
        Shape::OneOf("float32 | uint8"),
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.dtype",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; actual owned tensor dtype",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Dtype", "select"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.metadata.shape",
        "image.mossImageAdvanced.positive.mask.metadata",
        "shape",
        Shape::Numbers,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.shape",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; 2..4 positive integer dimensions; exact model/role tensor shape",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Shape", "list"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.metadata.byteLength",
        "image.mossImageAdvanced.positive.mask.metadata",
        "byteLength",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.byte_length",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; positive; at most 536870912 data bytes",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Byte length", "number"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.positive.mask.metadata.sha256",
        "image.mossImageAdvanced.positive.mask.metadata",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; exact 64 lowercase hexadecimal SHA256",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Sha256", "text"),

    // image.mossImageAdvanced.negative.values
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.assetId",
        "image.mossImageAdvanced.negative.values",
        "assetId",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "TensorRef.asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: non-empty project-owned tensor asset id; never a caller path",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Asset id", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.sha256",
        "image.mossImageAdvanced.negative.values",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TensorRef.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: exact 64 lowercase hexadecimal SHA256; reverify artifact bytes",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Sha256", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.metadata",
        "image.mossImageAdvanced.negative.values",
        "metadata",
        Shape::Object("TensorMetadata"),
        DefaultRule::Absent,
        IntentClass::Context,
        "TensorRef.metadata",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: optional measured metadata is re-read and compared to artifact; never trusted to authorize identity or shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Metadata", "object"),

    // image.mossImageAdvanced.negative.values.metadata
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.metadata.dtype",
        "image.mossImageAdvanced.negative.values.metadata",
        "dtype",
        Shape::OneOf("float32 | uint8"),
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.dtype",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; actual owned tensor dtype",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Dtype", "select"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.metadata.shape",
        "image.mossImageAdvanced.negative.values.metadata",
        "shape",
        Shape::Numbers,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.shape",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; 2..4 positive integer dimensions; exact model/role tensor shape",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Shape", "list"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.metadata.byteLength",
        "image.mossImageAdvanced.negative.values.metadata",
        "byteLength",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.byte_length",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; positive; at most 536870912 data bytes",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Byte length", "number"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.values.metadata.sha256",
        "image.mossImageAdvanced.negative.values.metadata",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; exact 64 lowercase hexadecimal SHA256",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Sha256", "text"),

    // image.mossImageAdvanced.negative.mask
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.assetId",
        "image.mossImageAdvanced.negative.mask",
        "assetId",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "TensorRef.asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: non-empty project-owned tensor asset id; never a caller path",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Asset id", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.sha256",
        "image.mossImageAdvanced.negative.mask",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TensorRef.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: exact 64 lowercase hexadecimal SHA256; reverify artifact bytes",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Sha256", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.metadata",
        "image.mossImageAdvanced.negative.mask",
        "metadata",
        Shape::Object("TensorMetadata"),
        DefaultRule::Absent,
        IntentClass::Context,
        "TensorRef.metadata",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: optional measured metadata is re-read and compared to artifact; never trusted to authorize identity or shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Metadata", "object"),

    // image.mossImageAdvanced.negative.mask.metadata
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.metadata.dtype",
        "image.mossImageAdvanced.negative.mask.metadata",
        "dtype",
        Shape::OneOf("float32 | uint8"),
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.dtype",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; actual owned tensor dtype",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Dtype", "select"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.metadata.shape",
        "image.mossImageAdvanced.negative.mask.metadata",
        "shape",
        Shape::Numbers,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.shape",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; 2..4 positive integer dimensions; exact model/role tensor shape",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Shape", "list"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.metadata.byteLength",
        "image.mossImageAdvanced.negative.mask.metadata",
        "byteLength",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.byte_length",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; positive; at most 536870912 data bytes",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Byte length", "number"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.negative.mask.metadata.sha256",
        "image.mossImageAdvanced.negative.mask.metadata",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; exact 64 lowercase hexadecimal SHA256",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Sha256", "text"),

    // image.mossImageAdvanced.noise
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.noise.assetId",
        "image.mossImageAdvanced.noise",
        "assetId",
        Shape::Text,
        DefaultRule::Literal("\"\""),
        IntentClass::WhatToMake,
        "TensorRef.asset_id",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: non-empty project-owned tensor asset id; never a caller path",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Asset id", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.noise.sha256",
        "image.mossImageAdvanced.noise",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "TensorRef.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: exact 64 lowercase hexadecimal SHA256; reverify artifact bytes",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Sha256", "text"),
    // crates/sceneworks-core/src/moss_image_advanced.rs:21
    OptionSpec::new(
        "image.mossImageAdvanced.noise.metadata",
        "image.mossImageAdvanced.noise",
        "metadata",
        Shape::Object("TensorMetadata"),
        DefaultRule::Absent,
        IntentClass::Context,
        "TensorRef.metadata",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: optional measured metadata is re-read and compared to artifact; never trusted to authorize identity or shape",
    ])
    .applies(Predicate::Modality("image"))
    .ui("Metadata", "object"),

    // image.mossImageAdvanced.noise.metadata
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.noise.metadata.dtype",
        "image.mossImageAdvanced.noise.metadata",
        "dtype",
        Shape::OneOf("float32 | uint8"),
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.dtype",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; actual owned tensor dtype",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Dtype", "select"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.noise.metadata.shape",
        "image.mossImageAdvanced.noise.metadata",
        "shape",
        Shape::Numbers,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.shape",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; 2..4 positive integer dimensions; exact model/role tensor shape",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Shape", "list"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.noise.metadata.byteLength",
        "image.mossImageAdvanced.noise.metadata",
        "byteLength",
        Shape::Integer,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.byte_length",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; positive; at most 536870912 data bytes",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Byte length", "number"),
    // crates/sceneworks-core/src/tensor_artifact.rs:14
    OptionSpec::new(
        "image.mossImageAdvanced.noise.metadata.sha256",
        "image.mossImageAdvanced.noise.metadata",
        "sha256",
        Shape::Text,
        DefaultRule::Required,
        IntentClass::Context,
        "verified TensorMetadata.sha256",
    )
    .limits(&[
        "crates/sceneworks-core/src/moss_image_advanced.rs: schema; exact 64 lowercase hexadecimal SHA256",
    ])
    .applies(Predicate::Modality("image"))
    .compat("Client metadata is advisory; server remeasures and refuses mismatch")
    .ui("Sha256", "text"),
];
