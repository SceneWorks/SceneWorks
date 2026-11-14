//! Historical `advanced` map, including reader-only wire fields and host-owned receipts.
//! Sharing decisions remain with workflow_share: a Context option may be portable or private.
use crate::{
    DefaultRule::*,
    Effect,
    IntentClass::*,
    OptionSpec,
    Predicate::{self, *},
    Shape::*,
};

const IMAGE: Predicate = Modality("image");
const VIDEO: Predicate = Modality("video");
const VISUAL: Predicate = Any(&[IMAGE, VIDEO]);
const PHASES: Predicate = ModelContains("loraCompatibility.types", "acceleration");
const MOSS: Predicate = Family("moss_image");
const LTX: Predicate = Family("ltx-video");
const LTX25: Predicate = Any(&[Generator("ltx_2_5"), Generator("ltx_2_5_distilled")]);
const H3: Predicate = All(&[VIDEO, Family("minimax-h3")]);
const PID: Predicate = ModelField("ui.pid");
const POSE: Predicate = ModelField("ui.poseLibrary");
const CONTROL: Predicate = ModelField("ui.controlModes");
const ENHANCER: Predicate = Any(&[
    ModelField("ui.promptEnhance"),
    Feature("supports_prompt_enhancement"),
]);
const INTERLEAVE: Predicate = Capability("interleave");
const DETAIL: Predicate = Capability("image_detail");
const PULID: Predicate = Adapter("pulid_flux");
const CLIP: Predicate = Any(&[
    Capability("extend_clip"),
    Capability("video_bridge"),
    Capability("video_to_video"),
]);
const NUMBER_INPUT: &str = "Native numeric readers accept a JSON number or a trimmed numeric string; malformed values use the route fallback. Strict enqueue checks apply first.";
const LEGACY_FLAG: &str = "Legacy image flag accepts true, nonzero numbers and nonempty strings/arrays; strict route checks apply first.";

const fn effect(when: &'static str, action: &'static str, target: &'static str) -> Effect {
    Effect {
        when,
        action,
        target,
    }
}

macro_rules! opt {
    ($key:literal, $shape:expr, $default:expr, $predicate:expr, $intent:expr, $binding:literal) => {
        OptionSpec::new(
            concat!("advanced.", $key),
            "advanced",
            $key,
            $shape,
            $default,
            $intent,
            $binding,
        )
        .applies($predicate)
    };
}
macro_rules! context {
    ($key:literal, $shape:expr, $predicate:expr, $binding:literal) => {
        opt!($key, $shape, Absent, $predicate, Context, $binding).internal()
    };
}

macro_rules! nested {
    ($parent:literal, $key:literal, $shape:expr, $default:expr, $predicate:expr, $intent:expr, $binding:literal) => {
        OptionSpec::new(
            concat!("advanced.", $parent, ".", $key),
            concat!("advanced.", $parent),
            $key,
            $shape,
            $default,
            $intent,
            $binding,
        )
        .applies($predicate)
    };
}

pub(crate) const SPECS: &[OptionSpec] = &[

    nested!("mossImage", "acceleration", OneOf("off | unmerged_bf16 | full_fp32"), Literal("\"off\""), MOSS, WhatToMake, "MossImageOptions.acceleration")
        .effects(&[effect("DMD", "exclude", "advanced.sampler"),effect("DMD", "exclude", "advanced.scheduler"),effect("DMD", "exclude", "advanced.schedulerShift"),effect("DMD", "exclude", "advanced.mossImage.training_shift"),effect("DMD", "exclude", "image.negativePrompt"),effect("DMD", "override", "advanced.steps"),effect("DMD", "override", "advanced.guidanceScale")])
        .limits(&["MossImageResolved::resolve"]),
    nested!("mossImage", "max_sequence_length", Integer, Literal("4096"), MOSS, WhatToMake, "MossImageOptions.max_sequence_length").limits(&["1..=4096"]),
    nested!("mossImage", "uncond_skip_layers", Numbers, Provider("ordinary [22], DMD no skipping"), MOSS, WhatToMake, "MossImageOptions.uncond_skip_layers").limits(&["integer layers 0..=41"]).compat("Null uses default; [] disables skipping."),
    nested!("mossImage", "training_shift", Number, Literal("1"), MOSS, WhatToMake, "MossImageOptions.training_shift").limits(&["positive finite"]).compat("Omission and null use frozen grid shift 1."),
    nested!("mossImage", "slg", Object("layers,scale,start,stop"), Absent, MOSS, WhatToMake, "MossSkipLayerGuidance").limits(&["MossImageResolved::resolve"]),
    nested!("mossImage.slg", "layers", Numbers, Required, MOSS, WhatToMake, "MossSkipLayerGuidance.layers").limits(&["integer layers 0..=41"]),
    nested!("mossImage.slg", "scale", Number, Required, MOSS, WhatToMake, "MossSkipLayerGuidance.scale").limits(&["finite >=0"]),
    nested!("mossImage.slg", "start", Number, Required, MOSS, WhatToMake, "MossSkipLayerGuidance.start").limits(&["0 <= start <= stop <= 1"]),
    nested!("mossImage.slg", "stop", Number, Required, MOSS, WhatToMake, "MossSkipLayerGuidance.stop").limits(&["0 <= start <= stop <= 1"]),
    nested!("mossImageEdit", "reference_max_edge", Integer, Literal("512"), MOSS, WhatToMake, "MossImageEditInputs.reference_max_edge").limits(&["16..=4096"]),
    nested!("mossImageEdit", "use_kv_cache", Bool, Literal("true"), MOSS, MachineBudget, "MossImageEditInputs.use_kv_cache"),
    nested!("veda", "enabled", Bool, Literal("false"), H3, WhatToMake, "H3VedaSettings.enabled"),
    nested!("veda", "generated", Text, Literal("\"90%\""), H3, WhatToMake, "H3VedaSettings.generated").limits(&["percentage [0,100) or positive u32 tiles"]),
    nested!("veda", "reference", Text, Literal("\"90%\""), H3, WhatToMake, "H3VedaSettings.reference").limits(&["percentage [0,100) or positive u32 tiles"]),
    nested!("veda", "fullAttentionLayers", Text, Literal("\"\""), H3, WhatToMake, "H3VedaSettings.full_attention_layers").limits(&["h3_veda::indices native layer domain"]),
    nested!("veda", "fullAttentionSteps", Text, Literal("\"\""), H3, WhatToMake, "H3VedaSettings.full_attention_steps").limits(&["h3_veda::indices effective step domain"]),
    nested!("phases[]", "steps", Integer, Required, PHASES, WhatToMake, "MultiPhaseSpec.steps").limits(&["positive; phase sum within request limit"]),
    nested!("phases[]", "guidance", Number, Inherit("request guidance"), PHASES, WhatToMake, "MultiPhaseSpec.guidance").limits(&["finite; >= 0"]),
    nested!("phases[]", "loras", List("index,weight"), Literal("[]"), PHASES, WhatToMake, "MultiPhaseSpec.loras"),
    nested!("phases[].loras[]", "index", Integer, Required, PHASES, WhatToMake, "PhaseLoraRef.index").limits(&["index into request loras"]),
    nested!("phases[].loras[]", "weight", Number, Inherit("adapter load-time scale"), PHASES, WhatToMake, "PhaseLoraRef.weight").limits(&["finite"]),
    nested!("poses[]", "keypoints", List("x,y,confidence coordinates"), Absent, POSE, WhatToMake, "pose body skeleton"),
    nested!("poses[]", "hands", List("hand coordinate arrays"), Absent, POSE, WhatToMake, "pose hand skeleton"),
    nested!("poses[]", "face", List("face coordinate arrays"), Absent, POSE, WhatToMake, "pose face skeleton"),
    nested!("structuredPrompt", "version", Integer, Literal("1"), Family("ideogram"), Context, "structured prompt version"),
    nested!("structuredPrompt", "intent", Text, Literal("\"\""), Family("ideogram"), WhatToMake, "authored high-level description"),
    nested!("structuredPrompt", "caption", Object("Ideogram caption document"), Absent, Family("ideogram"), WhatToMake, "serialized Ideogram structured caption").ui("Structured caption", "ideogramCaption"),

    // Field names and constraints mirror ideogramCaption.js; rendering remains source-owned.
    nested!("structuredPrompt.caption", "high_level_description", Text, Absent, Family("ideogram"), WhatToMake, "Ideogram caption high-level description"),
    nested!("structuredPrompt.caption", "style_description", Object("aesthetics,lighting,photo or art_style,medium,color_palette"), Absent, Family("ideogram"), WhatToMake, "Ideogram caption style").limits(&["ideogramCaption::STYLE_KNOWN_KEYS; exactly one of photo/art_style; canonical key order depends on discriminator"]),
    nested!("structuredPrompt.caption", "compositional_deconstruction", Object("background,elements"), Required, Family("ideogram"), WhatToMake, "Ideogram caption composition").limits(&["ideogramCaption::COMPOSITION_KEY_ORDER; unknown keys rejected"]),
    nested!("structuredPrompt.caption.style_description", "aesthetics", Text, Absent, Family("ideogram"), WhatToMake, "Ideogram caption aesthetics").compat("Missing recommended style keys produce advisory key-order warnings."),
    nested!("structuredPrompt.caption.style_description", "lighting", Text, Absent, Family("ideogram"), WhatToMake, "Ideogram caption lighting"),
    nested!("structuredPrompt.caption.style_description", "medium", Text, Absent, Family("ideogram"), WhatToMake, "Ideogram caption medium"),
    nested!("structuredPrompt.caption.style_description", "photo", Text, Absent, Family("ideogram"), WhatToMake, "Ideogram photographic style discriminator")
        .effects(&[effect("present", "exclude", "advanced.structuredPrompt.caption.style_description.art_style")]),
    nested!("structuredPrompt.caption.style_description", "art_style", Text, Absent, Family("ideogram"), WhatToMake, "Ideogram art style discriminator")
        .effects(&[effect("present", "exclude", "advanced.structuredPrompt.caption.style_description.photo")]),
    nested!("structuredPrompt.caption.style_description", "color_palette", Strings, Absent, Family("ideogram"), WhatToMake, "Ideogram global color palette").limits(&["ideogramCaption::STYLE_PALETTE_MAX = 16; each matches ^#[0-9A-F]{6}$"]),
    nested!("structuredPrompt.caption.compositional_deconstruction", "background", Text, Literal("\"\""), Family("ideogram"), WhatToMake, "Ideogram composition background").compat("Required in a submitted composition; emptyCaption creates an empty string."),
    nested!("structuredPrompt.caption.compositional_deconstruction", "elements", List("type,bbox,text,desc,color_palette"), Literal("[]"), Family("ideogram"), WhatToMake, "Ideogram ordered layout elements").limits(&["ideogramCaption::ELEMENT_KNOWN_KEYS; text field only on text elements"]).compat("Required in a submitted composition; empty list produces advisory warning."),
    nested!("structuredPrompt.caption.compositional_deconstruction.elements[]", "type", OneOf("obj | text"), Required, Family("ideogram"), WhatToMake, "Ideogram element discriminator").limits(&["ideogramCaption::ELEMENT_TYPES"]),
    nested!("structuredPrompt.caption.compositional_deconstruction.elements[]", "bbox", Numbers, Absent, Family("ideogram"), WhatToMake, "Ideogram element bounding box").limits(&["ideogramCaption::BBOX_MIN = 0; BBOX_MAX = 1000; exactly four integers [ymin,xmin,ymax,xmax]; ymin <= ymax and xmin <= xmax"]).compat("Editor rounds/clamps coordinates; submitted caption validator rejects invalid boxes."),
    nested!("structuredPrompt.caption.compositional_deconstruction.elements[]", "text", Text, Literal("\"\""), Family("ideogram"), WhatToMake, "Ideogram rendered text").limits(&["type must be text; absent for obj"]),
    nested!("structuredPrompt.caption.compositional_deconstruction.elements[]", "desc", Text, Literal("\"\""), Family("ideogram"), WhatToMake, "Ideogram element description"),
    nested!("structuredPrompt.caption.compositional_deconstruction.elements[]", "color_palette", Strings, Absent, Family("ideogram"), WhatToMake, "Ideogram element color palette").limits(&["ideogramCaption::ELEMENT_PALETTE_MAX = 5; each matches ^#[0-9A-F]{6}$"]),
    nested!("structuredPrompt", "magicPromptBackend", Text, Absent, Family("ideogram"), Context, "caption author provenance"),
    nested!("structuredPrompt", "edited", Bool, Literal("false"), Family("ideogram"), Context, "caption edit provenance"),
    nested!("structuredPrompt", "runtimePrompt", Text, Literal("\"\""), Family("ideogram"), WhatToMake, "serialized structured prompt"),
    nested!("flux3_layout", "version", Integer, Literal("1"), Family("flux3"), Context, "Layout.version").limits(&["must equal 1"]),
    nested!("flux3_layout", "boxes", List("FLUX.3 placement/edit box"), Required, Family("flux3"), WhatToMake, "Layout.boxes").ui("Layout boxes", "layoutEditor"),
    nested!("flux3_layout.boxes[]", "id", Text, Required, Family("flux3"), Context, "stable box identity").limits(&["distinct nonempty text"]),
    nested!("flux3_layout.boxes[]", "desc", Text, Absent, Family("flux3"), WhatToMake, "box caption"),
    nested!("flux3_layout.boxes[]", "bbox", Numbers, Absent, Family("flux3"), WhatToMake, "placement coordinates").limits(&["flux3::Layout::from_json"]),
    nested!("flux3_layout.boxes[]", "src_bbox", Numbers, Absent, Family("flux3"), WhatToMake, "edit source coordinates").limits(&["flux3::Layout::from_json"]),
    nested!("flux3_layout.boxes[]", "tgt_bbox", Numbers, Absent, Family("flux3"), WhatToMake, "edit target coordinates").limits(&["flux3::Layout::from_json"]),
    nested!("flux3_layout.boxes[]", "bbox_space", Text, Literal("\"normalized\""), Family("flux3"), WhatToMake, "placement coordinate space").limits(&["flux3::layouts::Space"]),
    nested!("flux3_layout.boxes[]", "src_space", Text, Literal("\"normalized\""), Family("flux3"), WhatToMake, "source coordinate space").limits(&["flux3::layouts::Space"]),
    nested!("flux3_layout.boxes[]", "tgt_space", Text, Literal("\"normalized\""), Family("flux3"), WhatToMake, "target coordinate space").limits(&["flux3::layouts::Space"]),
    nested!("flux3_layout.boxes[]", "from", Text, Absent, Family("flux3"), WhatToMake, "layout edit origin").limits(&["flux3::pup::is_row_origin"]),
    nested!("flux3_layout.boxes[]", "reference_id", Text, Absent, Family("flux3"), WhatToMake, "reference attachment identity"),
    nested!("controlWeights", "overlayId", Text, Absent, CONTROL, WhatToMake, "trained overlay registry identity"),
    nested!("controlWeights", "path", Text, Absent, CONTROL, Context, "resolved local control weights").internal(),
    nested!("pidCheckpoint", "repo", Text, Provider("PID student repository"), PID, WhatToMake, "PiD student repository"),
    nested!("pidCheckpoint", "filename", Text, Provider("PID student filename"), PID, WhatToMake, "PiD student filename").limits(&["pid::safe_weight_filename"]),

    // Sampling: the model's menu and engine fallback remain independent of an explicit override.
    opt!("resolution", Text, Model("defaults.resolution"), VISUAL, WhatToMake, "request width/height and recipe resolution")
        .limits(&["limits.resolutions", "limits.minDimension", "limits.maxDimension", "limits.requiresDimensionsMultipleOf"]),
    opt!("sampler", Text, Model("defaults.sampler"), Any(&[ModelField("limits.samplers"), Feature("samplers")]), WhatToMake, "GenerationRequest.sampler")
        .limits(&["limits.samplers", "generator.samplers"]).compat("Blank and default mean absent; native route normalizes unsupported historical overrides."),
    opt!("scheduler", Text, Model("defaults.scheduler"), Any(&[ModelField("limits.schedulers"), Feature("schedulers")]), WhatToMake, "GenerationRequest.scheduler")
        .limits(&["limits.schedulers", "generator.schedulers"]).compat("Blank and default mean absent; native route normalizes unsupported historical overrides."),
    opt!("schedulerShift", Number, Model("defaults.schedulerShift"), Any(&[ModelField("image.supportsSchedulerShift"), ModelField("video.supportsSchedulerShift"), ModelField("limits.schedulers")]), WhatToMake, "GenerationRequest.scheduler_shift")
        .aliases(&["timestepShift"]).limits(&["image_sampling_controls::requested_scheduler_shift"]).compat("Image reads canonical first, then timestepShift (number or numeric string). SenseNova interleave uses timestepShift directly; nonpositive legacy SenseNova edit falls back to 3."),
    opt!("cfgInterval", Numbers, Provider("checkpoint CFG interval"), ModelField("image.supportsCfgInterval"), WhatToMake, "GenerationRequest.cfg_interval")
        .limits(&["0 <= start < end <= 1", "image_sampling_controls::requested_cfg_interval"]).effects(&[effect("CFG off", "disable", "advanced.cfgInterval")]).ui("Guidance interval", "numberPair"),
    opt!("captionOverflow", OneOf("warn | error | silent"), Provider("checkpoint caption policy"), ModelField("image.captionOverflowPolicies"), WhatToMake, "GenerationRequest.caption_overflow")
        .limits(&["image.captionOverflowPolicies"]),
    opt!("steps", Integer, Model("defaults.steps"), VISUAL, WhatToMake, "GenerationRequest.steps")
        .limits(&["limits.hardMinSteps", "limits.hardMaxSteps", "limits.steps", "sampling::max_requested_steps"]).compat(NUMBER_INPUT),
    opt!("guidanceScale", Number, Model("defaults.guidanceScale"), Any(&[All(&[IMAGE, ModelNotFalse("image.supportsGuidance")]), All(&[VIDEO, ModelNotFalse("video.supportsGuidance")]), INTERLEAVE]), WhatToMake, "GenerationRequest.guidance / InterleaveParams.cfg_scale")
        .limits(&["sampling::resolve_guidance", "ltx25_generation_options"]).compat(NUMBER_INPUT),
    opt!("guidanceMethod", Text, Provider("generator default guidance method"), Feature("supported_guidance_methods"), WhatToMake, "GenerationRequest.guidance_method")
        .limits(&["generator.supported_guidance_methods"]).compat("Empty/default use native method; CFG is the studio default."),
    opt!("numInferenceSteps", Integer, Literal("50"), INTERLEAVE, WhatToMake, "InterleaveParams.steps")
        .limits(&["clamp 1..=100"]).compat("Interleave accepts numeric strings; image completion also records its measured step count under this name."),
    opt!("enhancePrompt", Bool, Literal("false"), ENHANCER, WhatToMake, "GenerationRequest.enhance_prompt"),
    opt!("enhanceMaxTokens", Integer, Provider("prompt enhancer token budget"), ENHANCER, WhatToMake, "prompt-enhancement sampler max_new_tokens")
        .limits(&["1..=prompt_enhancement_max_tokens(model)"]).effects(&[effect("present", "require", "advanced.enhancePrompt")]),
    opt!("enhanceTemperature", Number, Provider("prompt enhancer sampling temperature"), ENHANCER, WhatToMake, "prompt-enhancement sampler temperature")
        .limits(&["0..=2 finite"]).effects(&[effect("present", "require", "advanced.enhancePrompt")]),
    opt!("transparentBackground", Bool, Literal("false"), Any(&[ModelField("supportsAlphaOutput"), Feature("supports_alpha_output")]), WhatToMake, "GenerationRequest.output_channels (RGBA)"),
    opt!("decoder", Text, Literal("\"native\""), Feature("decoders"), WhatToMake, "GenerationRequest decoder / terminal VAE")
        .limits(&["installed decoder components"]).effects(&[effect("nonNative", "clear", "advanced.usePid")]),
    opt!("usePid", Bool, Literal("false"), PID, WhatToMake, "PiD terminal decode")
        .effects(&[effect("enabled", "selectNative", "advanced.decoder"),effect("enabled", "exclude", "image.hiresFix"),effect("enabled", "exclude", "image.upscale")]).compat(LEGACY_FLAG),
    opt!("pidTarget", OneOf("2k | 4k"), Literal("\"4k\""), PID, WhatToMake, "PiD base-size cap"),
    opt!("pidCheckpoint", Object("PiD checkpoint {repo,filename}"), Model("ui.pid.checkpointId"), PID, WhatToMake, "PiD student artifact")
        .limits(&["pid::safe_weight_filename"]),
    opt!("pidGemma", Text, Provider("PID_GEMMA_REPO"), PID, WhatToMake, "PiD Gemma text encoder"),
    opt!("ipAdapterScale", Number, Model("ui.referenceStrengthDefault"), IMAGE, WhatToMake, "IP-Adapter image conditioning scale")
        .limits(&["ui.variationStrength", "provider IP-Adapter clamp"]).compat(NUMBER_INPUT),
    opt!("controlnetConditioningScale", Number, Model("ui.identityStructure.default"), ModelField("ui.identityStructure"), WhatToMake, "InstantID ControlNet conditioning scale")
        .limits(&["ui.identityStructure.min", "ui.identityStructure.max"]).compat(NUMBER_INPUT),
    opt!("trueCfgScale", Number, Model("ui.variationStrength.default"), Any(&[ModelField("ui.variationStrength"), Family("sensenova-u1"), Family("qwen-image")]), WhatToMake, "GenerationRequest.true_cfg")
        .limits(&["ui.variationStrength.min", "ui.variationStrength.max"]).compat(NUMBER_INPUT),
    opt!("strength", Number, Model("ui.img2imgStrength.default"), Any(&[ModelField("ui.img2img"), Capability("edit_image"), DETAIL]), WhatToMake, "img2img / detail denoising strength")
        .limits(&["ui.img2imgStrength.min", "ui.img2imgStrength.max", "ordered_image_reference_strength_error"]).compat(NUMBER_INPUT),
    opt!("referenceStrength", Number, Model("ui.referenceStrengthDefault"), IMAGE, WhatToMake, "identity-init conditioning strength")
        .limits(&["ordered_image_reference_strength_error","flux3::validate_request"]).compat(NUMBER_INPUT),
    opt!("textStyleGain", Number, Literal("1"), ModelField("ui.textStyleGain"), WhatToMake, "GenerationRequest text style tap gain")
        .limits(&["clamp 0.25..=1.75"]).compat(NUMBER_INPUT),
    opt!("viewAngle", Text, Absent, ModelField("ui.viewAngles"), WhatToMake, "character view conditioning").limits(&["ui.viewAngles"]),
    opt!("viewAngles", Strings, Absent, ModelField("ui.viewAngles"), WhatToMake, "character view selection / Supra2 unsupported-carrier refusal").limits(&["ui.viewAngles"]),
    opt!("poses", List("pose skeleton: keypoints, hands, face"), Absent, POSE, WhatToMake, "pose ControlNet conditioning")
        .limits(&["image_request::MAX_JOB_POSES","validators::validate_prompt_extras"]).ui("Poses", "poseEditor"),
    opt!("faceRestore", Bool, Literal("false"), POSE, WhatToMake, "character face restoration pass").compat(LEGACY_FLAG),
    opt!("controlMode", Text, Absent, CONTROL, WhatToMake, "ControlNet conditioning mode").limits(&["ui.controlModes"]),
    opt!("controlImage", Text, Absent, CONTROL, WhatToMake, "control conditioning project asset"),
    opt!("controlWeights", Object("overlayId plus resolved artifact identity"), Absent, CONTROL, WhatToMake, "trained ControlNet overlay"),
    opt!("controlScale", Number, Model("ui.controlScale.default"), CONTROL, WhatToMake, "ControlNet conditioning scale")
        .limits(&["ui.controlScale.min","ui.controlScale.max"]).compat(NUMBER_INPUT),
    opt!("cnScale", Number, Literal("0.7"), DETAIL, WhatToMake, "DetailRequest.cn_scale").limits(&["clamp 0.1..=1.5"]).compat(NUMBER_INPUT),
    opt!("tile", Integer, Literal("1024"), DETAIL, MachineBudget, "DetailRequest.tile").limits(&["clamp 512..=1536"]).compat(NUMBER_INPUT),
    opt!("overlap", Integer, Literal("128"), DETAIL, WhatToMake, "DetailRequest.overlap").limits(&["clamp 0..=512"]).compat(NUMBER_INPUT),
    opt!("prompt", Text, Provider("DETAIL_DEFAULT_PROMPT"), DETAIL, WhatToMake, "DetailRequest.prompt"),
    opt!("negativePrompt", Text, Provider("DETAIL_DEFAULT_NEGATIVE"), DETAIL, WhatToMake, "DetailRequest.negative"),
    opt!("tileControlNetRepo", Text, Provider("detail tile ControlNet repository"), DETAIL, WhatToMake, "detail ControlNet artifact"),
    opt!("ipAdapterRepo", Text, Provider("SDXL IP-Adapter repository"), Family("sdxl"), WhatToMake, "SDXL IP-Adapter snapshot"),
    opt!("idWeight", Number, Literal("1"), PULID, WhatToMake, "PuLID reference id_weight").limits(&["clamp 0..=3"]).compat(NUMBER_INPUT),
    opt!("timestepToStartCfg", Integer, Literal("4"), PULID, WhatToMake, "PuLID timestep_to_start_cfg").limits(&["clamp 0..=20"]).compat(NUMBER_INPUT),
    opt!("openPoseScale", Number, Literal("0.7"), POSE, WhatToMake, "OpenPose conditioning strength").limits(&["clamp 0..=2"]).compat(NUMBER_INPUT),
    opt!("angleSet", Bool, Literal("false"), ModelField("ui.viewAngles"), WhatToMake, "character turnaround batch").compat(LEGACY_FLAG),
    opt!("keypointCollectionId", Text, Inherit("active angle collection"), ModelField("ui.viewAngles"), WhatToMake, "Key Point Library selection"),
    opt!("systemMessage", Text, Provider("INTERLEAVE_SYSTEM_MESSAGE"), INTERLEAVE, WhatToMake, "InterleaveParams.system_message"),
    opt!("imageGuidanceScale", Number, Provider("mode-specific image conditioning default"), Any(&[INTERLEAVE, Family("sensenova-u1"), Family("qwen-image")]), WhatToMake, "InterleaveParams.img_cfg_scale / reference conditioning image CFG").compat(NUMBER_INPUT),
    opt!("maxImages", Integer, Literal("6"), INTERLEAVE, WhatToMake, "InterleaveParams.max_images").limits(&["clamp 1..=10"]).compat(NUMBER_INPUT),
    opt!("maxNewTokens", Integer, Literal("2048"), INTERLEAVE, WhatToMake, "InterleaveParams.max_new_tokens").limits(&["clamp 64..=8192"]).compat(NUMBER_INPUT),
    opt!("thinkMode", Bool, Literal("false"), INTERLEAVE, WhatToMake, "InterleaveParams.think_mode"),
    opt!("phases", List("steps, guidance, ordered LoRA indices/weights"), Absent, PHASES, WhatToMake, "multi-phase denoise plan")
        .limits(&["image_request phase plan"])
        .effects(&[effect("enabled", "override with phase step sum", "advanced.steps"), effect("enabled", "override with per-phase guidance", "advanced.guidanceScale"), effect("enabled", "exclude", "advanced.usePid"), effect("enabled", "exclude", "image.hiresFix"), effect("enabled", "exclude", "image.sourceAssetId"), effect("enabled", "exclude", "image.referenceAssetIds"), effect("enabled", "exclude", "advanced.poses")])
        .ui("Multi-phase denoise", "phaseEditor"),
    opt!("flashAttn", Bool, Literal("true"), IMAGE, MachineBudget, "Candle SDXL flash attention kernel"),
    opt!("mlxQuantize", Integer, Model("mlx.quantize"), VISUAL, MachineBudget, "model load quantization bits").limits(&["download variants","declared memory routes"]),
    opt!("mlxQuantizeExplicit", Bool, Literal("false"), VISUAL, MachineBudget, "user tier selection provenance"),
    opt!("convRot", Bool, Literal("false"), IMAGE, MachineBudget, "ConvRot load recipe").limits(&["declared memory routes"]),
    opt!("quantTier", Text, Provider("available checkpoint tiers"), VISUAL, MachineBudget, "artifact tier identity").limits(&["download variants","declared memory routes"]),
    opt!("quant", Text, Absent, IMAGE, MachineBudget, "legacy Krea tier selector").compat("Used after quantTier on the imported Krea route."),
    opt!("textEncoderModel", Text, Provider("bundled text encoder"), VISUAL, WhatToMake, "text encoder component selection").limits(&["text_encoder_selection"]),
    opt!("modelPath", Text, Absent, VISUAL, Context, "confined local model bundle").limits(&["resolve_app_managed_model_dir"]).ui("Model bundle", "localArtifact"),
    // A shape with no engine input is still accounted for when a public boundary reads and
    // refuses it. Naming this explicitly prevents a future validator from silently accepting it.
    opt!("denoisingStrength", Number, Absent, Family("flux3"), WhatToMake, "flux3 reference-strength refusal").internal(),
    opt!("imageStrength", Number, Absent, Family("flux3"), WhatToMake, "flux3 reference-strength refusal").internal(),
    opt!("ip_adapter_scale", Number, Absent, Family("flux3"), WhatToMake, "flux3 reference-strength refusal").internal(),
    opt!("controlnetScale", Number, Absent, MOSS, WhatToMake, "MOSS unsupported control refusal").internal(),
    opt!("guidanceEta", Number, Absent, MOSS, WhatToMake, "MOSS unsupported guidance refusal").internal(),
    opt!("guidanceMomentum", Number, Absent, MOSS, WhatToMake, "MOSS unsupported guidance refusal").internal(),
    opt!("guidanceNormThreshold", Number, Absent, MOSS, WhatToMake, "MOSS unsupported guidance refusal").internal(),
    opt!("appearance", Object("AppearanceSelection"), Absent, VISUAL, WhatToMake, "compose camera, film, looks and lighting").ui("Appearance", "appearanceMenu"),
    opt!("appearancePrompt", Text, Absent, VISUAL, WhatToMake, "authored text before appearance composition"),
    opt!("appearanceNegativePrompt", Text, Absent, VISUAL, WhatToMake, "authored negative before appearance composition"),
    opt!("appearancePromptBatch", Strings, Absent, IMAGE, WhatToMake, "authored prompts before batch composition").ui("Prompt batch", "promptBatch"),
    opt!("styleId", Text, Absent, VISUAL, WhatToMake, "style catalog selection"),
    opt!("stylePrompt", Text, Absent, VISUAL, WhatToMake, "authored text before style composition"),
    opt!("structuredPrompt", Object("Ideogram structured caption recipe"), Absent, Family("ideogram"), WhatToMake, "runtime structured caption").ui("Structured prompt", "ideogramCaption"),
    opt!("flux3_layout", Object("Flux3Layout"), Absent, Family("flux3"), WhatToMake, "FLUX.3 soft layout guidance").limits(&["flux3::validate_request"]).ui("Layout", "layoutEditor"),
    opt!("flux3_reference_ids", Strings, Absent, Family("flux3"), WhatToMake, "FLUX.3 reference attachment ordering").limits(&["unique; one per ordered reference, at most 10"]),
    opt!("mossImage", Object("MossImageOptions"), Provider("MossImageOptions::default"), MOSS, WhatToMake, "GenerationRequest.moss_image").limits(&["MossImageResolved::resolve"]).ui("MOSS generation", "mossImage"),
    opt!("mossImageEdit", Object("EditOptions"), Provider("EditOptions::default"), MOSS, WhatToMake, "MOSS edit input preparation").limits(&["moss_image::edit_options"]).ui("MOSS references", "mossImageEdit"),
    opt!("mossImageAdvanced", Object("EditAdvanced"), Absent, MOSS, WhatToMake, "MossImageAdvancedInputs").limits(&["moss_image_advanced::options"]).ui("MOSS edit tensors", "mossImageAdvanced"),
    opt!("promptBatch", Strings, Absent, IMAGE, WhatToMake, "ordered native prompt batch").limits(&["image.maxPromptBatchImages"]).ui("Prompt batch", "promptBatch"),
    // LTX-2.3's checkpoint table and LTX-2.5's backend-specific native routes are different declarations.
    opt!("motion", Text, Literal("\"slow push-in\""), VIDEO, WhatToMake, "motion preset in video recipe"),
    opt!("ltxPipeline", Text, Literal("\"auto\""), ModelField("resources.distilledCheckpoint"), WhatToMake, "legacy LTX pipeline selection"),
    opt!("distilledVariant", Text, Literal("\"1.1\""), ModelField("resources.distilledCheckpoint"), WhatToMake, "legacy LTX distilled checkpoint"),
    opt!("precision", Text, Literal("\"fp8\""), ModelField("resources.distilledCheckpoint"), MachineBudget, "legacy LTX weight precision"),
    opt!("quantization", Text, Literal("\"auto\""), VIDEO, MachineBudget, "video weight quantization"),
    opt!("transformerVariant", OneOf("distilled | dev"), Literal("\"distilled\""), LTX25, WhatToMake, "Ltx25TransformerVariant").effects(&[effect("dev", "require", "advanced.useDistillLora")]),
    opt!("vaeDecoder", OneOf("conv | diffusion"), Literal("\"conv\""), LTX25, WhatToMake, "Ltx25GenerationOptions.use_diffusion_decoder"),
    opt!("autoDuration", Bool, Literal("false"), LTX25, WhatToMake, "Ltx25GenerationOptions.auto_duration").effects(&[effect("enabled", "require", "advanced.autoDurationMinSeconds"),effect("enabled", "require", "advanced.autoDurationMaxSeconds")]),
    opt!("autoDurationMinSeconds", Number, Absent, LTX25, WhatToMake, "VideoAutoDurationRange.min_seconds").limits(&["positive finite, <= autoDurationMaxSeconds"]),
    opt!("autoDurationMaxSeconds", Number, Absent, LTX25, WhatToMake, "VideoAutoDurationRange.max_seconds").limits(&["positive finite, >= autoDurationMinSeconds"]),
    opt!("temporalUpsampleRounds", Integer, Literal("0"), LTX25, WhatToMake, "Ltx25GenerationOptions.temporal_upsample_rounds").limits(&["0..=LTX25_MAX_TEMPORAL_UPSAMPLE_ROUNDS"]),
    opt!("useDistillLora", Bool, Literal("true"), LTX, WhatToMake, "LTX distillation adapter"),
    opt!("distillStage1Strength", Number, Model("mlx.autoDistillLora.stage1Strength"), LTX, WhatToMake, "LTX distillation first-pass scale").limits(&["ltx25 dev fixes pass scales"]).compat(NUMBER_INPUT),
    opt!("distillStage2Strength", Number, Model("mlx.autoDistillLora.stage2Strength"), LTX, WhatToMake, "LTX distillation second-pass scale").limits(&["ltx25 dev fixes pass scales"]).compat(NUMBER_INPUT),
    opt!("useUncensoredEnhancer", Bool, Literal("false"), LTX, WhatToMake, "legacy LTX Gemma enhancer selection").effects(&[effect("true", "require", "advanced.enhancePrompt")]),
    opt!("noAudio", Bool, Literal("false"), LTX, WhatToMake, "VideoGenInput.video_mode=no_audio"),
    opt!("lightning", Bool, Model("video.supportsLightning"), ModelField("video.supportsLightning"), WhatToMake, "Wan Lightning 4-step recipe").effects(&[effect("enabled","override","advanced.steps"),effect("enabled","override","advanced.guidanceScale")]),
    opt!("videoCfgGuidanceScale", Number, Absent, LTX, WhatToMake, "legacy LTX adapter settings: video CFG"),
    opt!("videoStgGuidanceScale", Number, Absent, LTX, WhatToMake, "legacy LTX adapter settings: video STG"),
    opt!("videoRescaleScale", Number, Absent, LTX, WhatToMake, "legacy LTX adapter settings: video guidance rescale"),
    opt!("videoConditioningStrength", Number, Literal("1"), CLIP, WhatToMake, "source VideoClip conditioning strength").compat(NUMBER_INPUT),
    opt!("bridgeRightVideoConditioningStrength", Number, Literal("1"), Capability("video_bridge"), WhatToMake, "right VideoClip conditioning strength").compat(NUMBER_INPUT),
    opt!("imageConditioningStrength", Number, Literal("1"), LTX, WhatToMake, "first reference frame conditioning strength").compat(NUMBER_INPUT),
    opt!("lastFrameConditioningStrength", Number, Literal("1"), LTX, WhatToMake, "last reference frame conditioning strength").compat(NUMBER_INPUT),
    opt!("imageFrameIndex", Integer, Literal("0"), LTX, WhatToMake, "keyframe conditioning frame_idx").compat(NUMBER_INPUT),
    opt!("maskingStrength", Number, Literal("1"), Capability("replace_person"), WhatToMake, "mask conditioning strength").compat(NUMBER_INPUT),
    opt!("conditioningScale", Number, Literal("1"), VIDEO, WhatToMake, "video control scale").compat(NUMBER_INPUT),
    opt!("motionAnchorFrames", Integer, Provider("one third of output frames divided between boundaries"), CLIP, WhatToMake, "VACE retained boundary frames").limits(&["extend_anchor_frames: reserve five generated frames"]).compat(NUMBER_INPUT),
    opt!("veda", Object("H3VedaSettings"), Absent, H3, WhatToMake, "GenerationRequest.h3_veda").limits(&["h3_veda::requested_h3_veda"]).ui("Veda sparse attention", "h3Veda"),
    opt!("referenceImageShortEdge", Integer, Literal("2048"), H3, WhatToMake, "reference-image encoding short edge").limits(&["REFERENCE_IMAGE_SHORT_EDGE_MIN..=REFERENCE_IMAGE_SHORT_EDGE_MAX"]),
    opt!("referenceImageResizeMode", OneOf("match | max | diffusers"), Literal("\"match\""), H3, WhatToMake, "ReferenceImageResizeMode")
        .compat("New omission resolves to match; a historical attempt without a recorded mode replays as diffusers."),
    opt!("conditioningFps", Integer, Model("condFps"), Family("svd"), WhatToMake, "VideoGenInput.conditioning_fps").limits(&["clamp 1..=30"]).compat("Manifest condFps fallback, then 7; JSON integer or numeric string."),
    opt!("numFrames", Integer, Model("numFrames"), Family("svd"), WhatToMake, "VideoGenInput.frames").limits(&["clamp 1..=25"]).compat("Manifest fallback, then 25; JSON integer or numeric string."),
    opt!("motionBucketId", Integer, Model("motionBucketId"), Family("svd"), WhatToMake, "VideoGenInput.motion_bucket_id").limits(&["clamp 1..=255"]).compat("Manifest fallback, then 127; JSON integer or numeric string."),
    opt!("noiseAugStrength", Number, Model("noiseAugStrength"), Family("svd"), WhatToMake, "VideoGenInput.noise_aug_strength").compat("Manifest fallback, then 0.02; JSON number or numeric string, no clamp."),
    opt!("decodeChunkSize", Integer, Model("decodeChunkSize"), Family("svd"), MachineBudget, "VideoGenInput.decode_chunk_size").limits(&["clamp 1..=64"]).compat("Manifest fallback, then 8; JSON integer or numeric string."),
    opt!("segment", Bool, Literal("true"), Always, WhatToMake, "SAM2 person-track segmentation"),
    opt!("confidence", Number, Literal("0.25"), Always, WhatToMake, "person detection/tracking confidence threshold").limits(&["clamp 0.01..=1"]),
    opt!("timelineAction", OneOf("extend | replace | bridge"), Absent, VIDEO, Context, "timeline operation"),
    opt!("timelineContext", Object("timeline, item and track identities"), Absent, VIDEO, Context, "timeline result placement"),
    // Server/job context and resource bindings. These records remain discoverable to upgrades
    // and receipt inspection but a validator must never treat their presence as user authority.
    context!("durationHint", Text, VIDEO, "catalog duration hint echo"),
    context!("selectedPersonTrack", Object("PersonTrack"), VIDEO, "selection snapshot"),
    context!("replacementModeLabel", Text, VIDEO, "replacement mode display label"),
    context!("imageReference", Object("local image identity and media path"), IMAGE, "reference diagnostics"),
    context!("recipePresetId", Text, VISUAL, "resolved preset provenance"),
    context!("recipePresetName", Text, VISUAL, "resolved preset display name"),
    context!("recipePresetPrompt", Text, VISUAL, "resolved preset text"),
    context!("presetMissingLoras", Strings, VISUAL, "preset resolution diagnostics"),
    context!("mossOrdinarySettings", Object("ordinary studio settings"), MOSS, "restore inactive ordinary controls"),
    context!("mossImageExecutionReport", Object("MossImageExecutionReport"), MOSS, "MOSS provider receipt").aliases(&["moss_image_execution_report"]),
    context!("mossImageEditExecutionReport", Object("MossImageEditExecutionReport"), MOSS, "MOSS edit receipt").aliases(&["moss_image_edit_execution_report"]),
    context!("mossImageSampleIndex", Integer, MOSS, "native batch output ordinal"),
    context!("characterName", Text, IMAGE, "character job snapshot"),
    context!("characterType", Text, IMAGE, "character job snapshot"),
    context!("approvedReferenceIds", Strings, IMAGE, "character reference snapshot"),
    context!("look", Object("character look"), IMAGE, "character job snapshot"),
    context!("filmHarness", Object("film identity and shot attempt"), VISUAL, "film compilation provenance"),
    context!("modelRepo", Text, Always, "artifact source attribution").aliases(&["repo"]),
    context!("realModelInference", Bool, Always, "execution receipt"),
    context!("checkpointWeightFacts", Object("checkpoint weight identity"), Always, "weight provenance"),
    context!("memoryStrategy", Object("memory strategy"), Always, "memory execution receipt"),
    context!("memoryEvidenceRevision", Text, Always, "memory evidence provenance"),
    context!("promptEnhancement", Object("prompt enhancement receipt"), IMAGE, "worker-owned enhancement facts"),
    context!("upscale", Object("upscale receipt"), IMAGE, "regeneration strips completed upscale settings"),
    context!("hiresFix", Object("hires fix receipt"), IMAGE, "regeneration strips completed second pass"),
    context!("__portableAuxPaths", Object("worker-bound auxiliary paths"), Always, "portable auxiliary artifact leases"),
    context!("__portableBoundTier", Text, Always, "worker-bound tier identity"),
    context!("__portableFrozenSources", Bool, Always, "worker frozen-source guard"),
    context!("__portableGemmaFiles", List("PortableSourceFile"), Always, "worker-bound Gemma conversion inputs"),
];
