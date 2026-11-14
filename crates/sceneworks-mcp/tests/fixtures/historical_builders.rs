// Extracted verbatim production declarations/builders at d3f7b05bf22826e44fd43da19cff0985fb17d68b.
// Only the imports below are harness-owned. Do not replace this with current builder output.
use rmcp::schemars;
use sceneworks_core::video_request::{classify_reference_set, ReferenceSetVerdict};
use serde_json::{json, Map, Value};
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateImageArgs {
    #[schemars(
        schema_with = "appearance_schema",
        description = "Camera, film, multiple looks and lighting recipe IDs from list_appearance. The API resolves model-specific prompt directives and MOSS negatives; incompatible selections are refused."
    )]
    pub appearance: Option<Value>,

    #[schemars(description = "Project to generate into (from list_projects).")]
    pub project_id: String,
    #[schemars(
        description = "The image prompt (1-4000 characters); omit/empty only when edit positive embedding artifacts replace both prompt carriers."
    )]
    #[serde(default)]
    pub prompt: String,
    #[schemars(
        schema_with = "moss_image_advanced_schema",
        description = "MOSS edit-only native advanced inputs. Use exact project tensor assetId+sha256 from list_tensor_assets: FP32 positive/negative [B,S,4096], optional binary U8 masks[B,S], per-prompt negativePrompts, exact FP32 initial noise[B×count,128,H/16,W/16], output images or latents. These replace text inputs as documented; supplied noise yields seed:null. Latents return real downloadable FP32 NPY artifacts, never PNGs. No input-latents, strength, dtype or quantization option."
    )]
    pub moss_image_advanced: Option<Value>,
    #[schemars(
        description = "\"generate\" (default, text-to-image) or \"edit_image\" (image-to-image; needs sourceAssetId or referenceAssetIds)."
    )]
    pub mode: Option<String>,
    #[schemars(description = "Things to avoid in the image.")]
    pub negative_prompt: Option<String>,
    #[schemars(description = "Model id from list_models. Omit for the server default.")]
    pub model: Option<String>,
    #[schemars(description = "How many images to generate (1-8, default 1).")]
    pub count: Option<u32>,
    #[schemars(description = "Seed for reproducible output. Omit for random per-image seeds.")]
    pub seed: Option<i64>,
    #[schemars(description = "Output width in pixels (MOSS edit default 512; otherwise 1024).")]
    pub width: Option<u32>,
    #[schemars(description = "Output height in pixels (MOSS edit default 512; otherwise 1024).")]
    pub height: Option<u32>,
    #[schemars(description = "Style preset name (default \"cinematic\").")]
    pub style_preset: Option<String>,
    #[schemars(
        description = "LoRA adapters to apply: [{\"id\": <from list_loras>, \"weight\": 0.0-2.0}]."
    )]
    pub loras: Option<Vec<Value>>,
    #[schemars(description = "Character to condition on (character id).")]
    pub character_id: Option<String>,
    #[schemars(description = "Edit base image asset id (edit_image mode).")]
    pub source_asset_id: Option<String>,
    #[schemars(
        description = "Reference image asset id for IP-Adapter style/identity conditioning."
    )]
    pub reference_asset_id: Option<String>,
    #[schemars(
        description = "Multi-image reference set for a multi-reference edit (each id jointly conditions the edit)."
    )]
    pub reference_asset_ids: Option<Vec<String>>,
    #[schemars(
        description = "Inpaint mask asset id (white = edit region; inpaint-capable models only)."
    )]
    pub mask_asset_id: Option<String>,
    #[schemars(
        description = "Include SceneWorks workflow metadata in returned images. Default false: MCP outputs often travel to model-provider infrastructure, so workflow metadata is stripped unless explicitly requested."
    )]
    pub include_workflow: Option<bool>,
    // sc-25681: the per-model sampling controls, carried in `advanced` — the only keys the API
    // and worker read them from. Each is validated at enqueue against the model's catalog row (a
    // control the model does not declare is refused by name, never dropped), so the descriptions
    // say where to find what a model offers rather than restating a menu.
    #[schemars(description = "Denoise steps. Omit for the model's own default.")]
    pub steps: Option<u32>,
    #[schemars(
        description = "Classifier-free guidance scale. Omit for the model's default. 1.0 turns guidance off, so a model that binds its negative prompt / guidance interval to guidance refuses those at 1.0."
    )]
    pub guidance_scale: Option<f64>,
    #[schemars(
        description = "Sampler / solver name from the model's samplers menu (list_models). Iris-3B: \"dpmpp_2m\" (DPM-Solver++ order 2, default) or \"euler\" (order 1). Omit for the model default."
    )]
    pub sampler: Option<String>,
    #[schemars(
        description = "Schedule (timestep) shift, > 0, for a model with a shift axis (Iris-3B's default is its checkpoint's 4.0). Omit for the model default."
    )]
    pub scheduler_shift: Option<f64>,
    #[schemars(
        description = "Guidance interval [start, end] with 0 <= start < end <= 1 (model time: 1 = noise, 0 = clean): classifier-free guidance runs only inside it. Only for a model that offers one (Iris-3B); refused at guidance 1.0."
    )]
    pub cfg_interval: Option<[f64; 2]>,
    #[schemars(
        description = "What to do with a prompt longer than the text encoder's caption budget: \"warn\" (truncate and report a caption_truncated warning on the job and image), \"error\" (refuse the render) or \"silent\" (truncate quietly). Only for a model that offers it (Iris-3B); omit for the checkpoint's own policy."
    )]
    pub caption_overflow: Option<String>,
    #[schemars(
        description = "A native prompt batch: 2+ prompts rendered together in ONE pass, prompts × count <= the model's imageControls.maxPromptBatchImages (Iris-3B: 8). `prompt` must equal prompts[0]. Output order depends on the model: Iris-3B is round-robin, image k renders prompts[k % len]; MOSS-Image is prompt-major, image k renders prompts[k / count] using integer division. Both use seed + k. Only for a model declaring maxPromptBatchImages; omit otherwise."
    )]
    pub prompts: Option<Vec<String>>,
    #[schemars(
        description = "MOSS-Image controls, passed as advanced.mossImage. Canonical keys: acceleration (off default, unmerged_bf16, full_fp32; SFT only), max_sequence_length (1–4096), uncond_skip_layers (null author default [22], [] disables), training_shift, slg {layers,scale,start,stop}. DMD steps select prefix 1–8 (default 8); omit guidance/negative/sampler/scheduler/shift/SLG when accelerated. Ordinary/DMD controls are text-to-image only; edit_image also accepts max_sequence_length with acceleration off and no SLG/skip layers. Tag and JSON prompts are preserved."
    )]
    #[schemars(schema_with = "moss_image_options_schema")]
    pub moss_image: Option<Value>,
    #[schemars(
        description = "Native MOSS Base/SFT edit controls, mapped to advanced.mossImageEdit: reference_max_edge (16–4096, default 512, Lanczos resize without crop) and use_kv_cache (default true). Requires edit_image and ordered referenceAssetIds. Edit defaults: 512×512, 30 steps, CFG 4, schedulerShift 5. DMD, SLG, skip layers, masks and generic LoRAs are unsupported; max_sequence_length remains available through mossImage."
    )]
    #[schemars(schema_with = "moss_image_edit_options_schema")]
    pub moss_image_edit: Option<Value>,
}

fn moss_image_edit_options_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    sceneworks_core::moss_image::edit_options_schema()
        .try_into()
        .expect("MOSS edit options schema object")
}

fn appearance_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    sceneworks_core::appearance::selection_schema()
        .try_into()
        .expect("appearance schema")
}

fn moss_image_advanced_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    sceneworks_core::moss_image_advanced::schema()
        .try_into()
        .expect("MOSS advanced schema")
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SubmitVideoJobArgs {
    #[schemars(description = "Project to generate into (from list_projects).")]
    pub project_id: String,
    #[schemars(description = "The video prompt (1-4000 characters).")]
    pub prompt: String,
    #[schemars(
        description = "\"generate\" (default: text-to-video, or image-to-video when sourceAssetId is set, or first/last-frame when lastFrameAssetId is also set), \"reference\" (render from reference media; needs at least one referenceAssetIds or sourceClipAssetIds entry, with optional referenceAudioAssetIds), \"extend\" (continue a clip; needs sourceClipAssetId), \"bridge\" (fill between two clips; needs sourceClipAssetId + bridgeRightClipAssetId), \"person_replace\" (swap a tracked person; needs sourceClipAssetId + personTrackId + characterId), \"video_to_video\" (edit a clip; needs sourceClipAssetId), \"reference_video_to_video\" (edit a clip guided by reference images; needs sourceClipAssetId + referenceAssetIds), \"multi_video_to_video\" (blend several clips; needs 2+ sourceClipAssetIds), \"ads2v\" (edit a clip using a reference video + reference images; needs sourceClipAssetId + referenceClipAssetId + referenceAssetIds), or \"animate_character\" (animate a character with a driving video; needs sourceClipAssetId + referenceAssetIds or sourceAssetId)."
    )]
    pub mode: Option<String>,
    #[schemars(description = "Things to avoid in the video.")]
    pub negative_prompt: Option<String>,
    #[schemars(
        description = "Video model id from list_models (type \"video\"). Omit for the server default."
    )]
    pub model: Option<String>,
    #[schemars(description = "Clip length in seconds (1-30, default 6).")]
    pub duration: Option<f64>,
    #[schemars(description = "Frames per second (1-60, default 25).")]
    pub fps: Option<u32>,
    #[schemars(description = "Output width in pixels (256-1920, default 768).")]
    pub width: Option<u32>,
    #[schemars(description = "Output height in pixels (256-1920, default 512).")]
    pub height: Option<u32>,
    #[schemars(description = "Quality preset (\"draft\", \"balanced\" (default) or \"high\").")]
    pub quality: Option<String>,
    #[schemars(description = "Seed for reproducible output. Omit for a random seed.")]
    pub seed: Option<i64>,
    /// sc-17161. Denoise step count, carried in `advanced.steps` — the ONLY key the API reads it
    /// from (`sceneworks_core::video_request::requested_steps`). Without it this tool could only
    /// ever submit the model's `defaults.steps`, which for MiniMax-H3 is 50 and measured in HOURS
    /// at the default canvas; the cheap draft an agent would reach for was unreachable.
    /// Per-model floors are real (`limits.hardMinSteps`), so `list_models` reports them.
    #[schemars(
        description = "Denoise steps. Omit for the model's own default. Fewer steps render proportionally faster; a model may declare a minimum (minSteps from list_models) below which the request is refused rather than raised. If a step-distill adapter is attached (see loras), omitting this runs that adapter's own trained step count; setting it overrides that count."
    )]
    pub steps: Option<u32>,
    /// sc-18727. The turbo variant selector IS this field — a step-distill adapter is selected the
    /// same way any other LoRA is, which is what keeps the MCP surface, the Video Studio control and
    /// the generic picker on one payload key and one server-side resolver. The description says so
    /// explicitly because the effect is not guessable from an id: attaching
    /// `minimax_h3_turbo_4step_768p` changes the schedule, not just the weights.
    #[schemars(
        description = "LoRA adapters to apply: [{\"id\": <from list_loras>, \"weight\": 0.0-2.0}]. An adapter whose list_loras entry carries role=\"accelerator\" and a sampling block is a step-distill accelerator: attaching it also switches the render to that adapter's trained schedule (its sampling.steps and sampling.schedulerShift), which is how a MiniMax-H3 job renders in 4-8 steps instead of 50. Attach at most one accelerator per job — two asking for different schedules are refused."
    )]
    pub loras: Option<Vec<Value>>,
    #[schemars(
        description = "Character to condition on (character id; required for person_replace)."
    )]
    pub character_id: Option<String>,
    #[schemars(description = "Starting image asset id (generate mode: makes it image-to-video).")]
    pub source_asset_id: Option<String>,
    #[schemars(
        description = "Ending image asset id (generate mode: with sourceAssetId, makes it a first/last-frame generation)."
    )]
    pub last_frame_asset_id: Option<String>,
    #[schemars(
        description = "Source video clip asset id (extend: the clip to continue; bridge: the LEFT clip; person_replace: the clip to edit)."
    )]
    pub source_clip_asset_id: Option<String>,
    #[schemars(description = "RIGHT video clip asset id for bridge mode.")]
    pub bridge_right_clip_asset_id: Option<String>,
    /// sc-19576. The `ads2v` reference-VIDEO slot — a second source video, distinct from
    /// `sourceClipAssetId` (the clip being edited) and from `sourceClipAssetIds` (the reference
    /// clips of `reference` mode). Without this field `ads2v` was unreachable at the FIELD level,
    /// not merely the mode level: `validate_video_job` hard-requires `referenceClipAssetId` for it,
    /// so no combination of the existing arguments could have produced a valid payload.
    #[schemars(
        description = "Reference video clip asset id for \"ads2v\" mode — the video whose look/motion guides the edit, distinct from sourceClipAssetId (the clip being edited)."
    )]
    pub reference_clip_asset_id: Option<String>,
    /// sc-17161. `reference` mode's subject images. The API's `reference_to_video` arm takes them
    /// as a LIST (Bernini encodes each as a subject reference; MiniMax-H3 Ref2VA labels them
    /// `<Picture 1>`, `<Picture 2>` … IN THE ORDER GIVEN, so the order is part of the request).
    #[schemars(
        description = "Reference image asset ids for \"reference\" mode (subjects, characters, wardrobe, locations). Order is meaningful for models that label references in the prompt."
    )]
    pub reference_asset_ids: Option<Vec<String>>,
    /// sc-17161. Reference video clips for `reference` mode. Distinct from `sourceClipAssetId`,
    /// which is the single clip extend/bridge/person_replace edit.
    #[schemars(
        description = "Reference video clip asset ids for \"reference\" mode, on a model whose reference modes take motion clips as well as images. Models that do not take reference clips reject a non-empty list."
    )]
    pub source_clip_asset_ids: Option<Vec<String>>,
    /// sc-17160. Only models that DECLARE `limits.maxReferenceAudioAssets` accept these; every
    /// other video model refuses a non-empty list at enqueue, which is why the description says
    /// so rather than letting the caller discover it from a 400.
    #[schemars(
        description = "Audio clip asset ids to condition the render on (up to 3), for a model whose reference modes take audio as well as images. Every other video model rejects a non-empty list."
    )]
    pub reference_audio_asset_ids: Option<Vec<String>>,
    #[schemars(description = "Person track id to replace (person_replace mode).")]
    pub person_track_id: Option<String>,
    #[schemars(description = "person_replace scope: \"face_only\" (default) or \"full_body\".")]
    pub replacement_mode: Option<String>,
}

pub(crate) fn image_job_body(args: &GenerateImageArgs) -> Result<Value, String> {
    let mode = match args.mode.as_deref().map(str::trim).unwrap_or("generate") {
        "" | "generate" | "text_to_image" => "text_to_image",
        "edit_image" => "edit_image",
        other => {
            return Err(format!(
                "unsupported mode \"{other}\": use \"generate\" or \"edit_image\""
            ))
        }
    };
    if mode == "edit_image"
        && args.source_asset_id.is_none()
        && args
            .reference_asset_ids
            .as_deref()
            .is_none_or(|ids| ids.is_empty())
    {
        return Err(
            "edit_image mode requires a sourceAssetId (or referenceAssetIds for a \
             multi-reference edit)"
                .to_owned(),
        );
    }
    let mut body = Map::new();
    body.insert("projectId".to_owned(), json!(args.project_id));
    body.insert("mode".to_owned(), json!(mode));
    body.insert("prompt".to_owned(), json!(args.prompt));
    body.insert("count".to_owned(), json!(args.count.unwrap_or(1)));
    let optional = [
        ("appearance", args.appearance.clone()),
        (
            "negativePrompt",
            args.negative_prompt.as_ref().map(|v| json!(v)),
        ),
        ("model", args.model.as_ref().map(|v| json!(v))),
        ("seed", args.seed.map(|v| json!(v))),
        ("width", args.width.map(|v| json!(v))),
        ("height", args.height.map(|v| json!(v))),
        ("stylePreset", args.style_preset.as_ref().map(|v| json!(v))),
        ("loras", args.loras.as_ref().map(|v| json!(v))),
        ("characterId", args.character_id.as_ref().map(|v| json!(v))),
        (
            "sourceAssetId",
            args.source_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAssetId",
            args.reference_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAssetIds",
            args.reference_asset_ids.as_ref().map(|v| json!(v)),
        ),
        ("maskAssetId", args.mask_asset_id.as_ref().map(|v| json!(v))),
        (
            "prompts",
            args.prompts
                .as_ref()
                .filter(|prompts| !prompts.is_empty())
                .map(|v| json!(v)),
        ),
    ];
    for (key, value) in optional {
        if let Some(value) = value {
            body.insert(key.to_owned(), value);
        }
    }
    let advanced: Map<String, Value> = [
        ("mossImageAdvanced", args.moss_image_advanced.clone()),
        ("mossImage", args.moss_image.clone()),
        ("mossImageEdit", args.moss_image_edit.clone()),
        ("steps", args.steps.map(|v| json!(v))),
        ("guidanceScale", args.guidance_scale.map(|v| json!(v))),
        ("sampler", args.sampler.as_ref().map(|v| json!(v))),
        ("schedulerShift", args.scheduler_shift.map(|v| json!(v))),
        ("cfgInterval", args.cfg_interval.map(|v| json!(v))),
        (
            "captionOverflow",
            args.caption_overflow.as_ref().map(|v| json!(v)),
        ),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key.to_owned(), value)))
    .collect();
    if !advanced.is_empty() {
        body.insert("advanced".to_owned(), Value::Object(advanced));
    }
    // The same strict parser used by REST validates authored options and unsupported combinations.
    // It leaves omitted values for the API's model-aware defaults and never invents a receipt.
    if args.moss_image_edit.is_some() || args.moss_image_advanced.is_some() {
        sceneworks_core::moss_image::native_request(&body, &Default::default())?;
    }
    Ok(Value::Object(body))
}

/// Every `mode` string [`video_job_body`] accepts, in the order the menu presents them (sc-19576).
///
/// It exists so the unsupported-mode error can BUILD its menu instead of restating it: the
/// hand-typed menu was already one mode stale when this const was added, which is the same
/// "a message asserts what the code does not do" defect the five missing modes were.
///
/// It is NOT the source of truth for what is reachable — the `match` below is, and its arms are
/// what `video_job_body_reaches_every_mode_the_api_admits` drives to collect the emitted wire
/// modes. A name here with no arm would be an invisible lie, so that test calls the real mapper
/// for every entry rather than trusting this list.
pub(crate) const VIDEO_TOOL_MODES: &[&str] = &[
    "generate",
    "reference",
    "extend",
    "bridge",
    "person_replace",
    // sc-19576: the five wire modes `VIDEO_JOB_MODES` admitted and this tool could not reach. They
    // take their WIRE names rather than new task-level aliases — `generate` / `reference` /
    // `extend` / `bridge` / `person_replace` earned their aliases by being the shapes an agent asks
    // for in plain language, while these five have no shorter honest name than the mode itself, and
    // inventing one would put a second vocabulary between the agent and the `capabilities` array
    // `list_models` reports.
    "video_to_video",
    "reference_video_to_video",
    "multi_video_to_video",
    "ads2v",
    "animate_character",
];

/// Map `submit_video_job` args onto the `VideoJobRequest` wire shape (camelCase). The tool's
/// task-level modes map to the API's wire modes — `generate` picks `text_to_video` /
/// `image_to_video` / `first_last_frame` from the provided image assets — and the mode-specific
/// required assets are checked here so a bad call fails fast with a precise message instead of a
/// submitted-then-rejected job. Only provided optional fields are emitted so the API's serde
/// defaults stay authoritative.
///
/// Every required-asset check below mirrors the corresponding arm of `validate_video_job`
/// (`apps/rust-api/src/lib.rs`). Mirroring, not replacing: the API arm still runs, and the
/// per-model half (which media a given model actually takes, and how many) stays on the model's
/// declared caps at enqueue.
pub(crate) fn video_job_body(args: &SubmitVideoJobArgs) -> Result<Value, String> {
    let mode = match args.mode.as_deref().map(str::trim).unwrap_or("generate") {
        "" | "generate" => {
            if args.last_frame_asset_id.is_some() {
                if args.source_asset_id.is_none() {
                    return Err(
                        "generate mode with a lastFrameAssetId also needs a sourceAssetId \
                         (the first frame)"
                            .to_owned(),
                    );
                }
                "first_last_frame"
            } else if args.source_asset_id.is_some() {
                "image_to_video"
            } else {
                "text_to_video"
            }
        }
        // sc-17161 made `reference_to_video` reachable — it is Ref2VA's ONLY mode, and the mode
        // the `referenceAudioAssetIds` field sc-17160 added exists to feed, so the field was
        // reachable while the mode that consumes it was not.
        //
        // sc-19574 corrected WHICH sets are legal. sc-17161 mirrored `validate_video_job`'s
        // then-current "at least one reference of ANY kind", audio alone included. The reference
        // implementation says otherwise, in a docstring and in an assertion: diffusers `MiniMaxH3`
        // documents `MiniMaxH3AudioReference` as "never on its own … It never reaches the
        // conditioner", and `before_encoder.py` raises on `set(kinds) == {"audio"}`. An audio-only
        // set leaves the VISUAL stream unconditioned; the worker refuses it (sc-19508) and the API
        // now 400s it, so this refuses it first — which is the whole point of checking here.
        //
        // The rule is `sceneworks_core::video_request::classify_reference_set`, shared with the API
        // and the worker; only the wording is this tool's.
        "reference" => {
            let count = |ids: &Option<Vec<String>>| ids.as_deref().map_or(0, <[String]>::len);
            if classify_reference_set(
                count(&args.reference_asset_ids),
                count(&args.source_clip_asset_ids),
                count(&args.reference_audio_asset_ids),
            ) != ReferenceSetVerdict::Conditionable
            {
                return Err(
                    "reference mode requires at least one referenceAssetIds or sourceClipAssetIds \
                     entry — an audio reference conditions the soundtrack and cannot be the only \
                     reference"
                        .to_owned(),
                );
            }
            "reference_to_video"
        }
        "extend" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "extend mode requires a sourceClipAssetId (the clip to continue)".to_owned(),
                );
            }
            "extend_clip"
        }
        "bridge" => {
            if args.source_clip_asset_id.is_none() || args.bridge_right_clip_asset_id.is_none() {
                return Err(
                    "bridge mode requires a sourceClipAssetId (left clip) and a \
                     bridgeRightClipAssetId (right clip)"
                        .to_owned(),
                );
            }
            "video_bridge"
        }
        "person_replace" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "person_replace mode requires a sourceClipAssetId (the clip to edit)"
                        .to_owned(),
                );
            }
            if args.person_track_id.is_none() {
                return Err(
                    "person_replace mode requires a personTrackId (the person to replace)"
                        .to_owned(),
                );
            }
            if args.character_id.is_none() {
                return Err(
                    "person_replace mode requires a characterId (the replacement Character)"
                        .to_owned(),
                );
            }
            "replace_person"
        }
        // ---- sc-19576: the five modes `VIDEO_JOB_MODES` admitted and this tool could not reach.
        // Each carries the required-asset check from its own `validate_video_job` arm, so the tool
        // keeps its fail-fast contract instead of shipping a name whose every payload 400s — a
        // mode you can NAME but cannot submit a valid payload for is not reachable.
        "video_to_video" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "video_to_video mode requires a sourceClipAssetId (the clip to edit)"
                        .to_owned(),
                );
            }
            "video_to_video"
        }
        "reference_video_to_video" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "reference_video_to_video mode requires a sourceClipAssetId (the clip to edit)"
                        .to_owned(),
                );
            }
            if args
                .reference_asset_ids
                .as_deref()
                .is_none_or(<[String]>::is_empty)
            {
                return Err(
                    "reference_video_to_video mode requires at least one referenceAssetIds entry"
                        .to_owned(),
                );
            }
            "reference_video_to_video"
        }
        // mv2v blends the PLURAL `sourceClipAssetIds`, not the singular slot the edit modes use —
        // two fields with confusingly similar names, and the API counts the plural one.
        "multi_video_to_video" => {
            if args
                .source_clip_asset_ids
                .as_deref()
                .map_or(0, <[String]>::len)
                < 2
            {
                return Err(
                    "multi_video_to_video mode requires at least two sourceClipAssetIds entries \
                     (the clips to blend)"
                        .to_owned(),
                );
            }
            "multi_video_to_video"
        }
        "ads2v" => {
            if args.source_clip_asset_id.is_none() {
                return Err("ads2v mode requires a sourceClipAssetId (the clip to edit)".to_owned());
            }
            if args.reference_clip_asset_id.is_none() {
                return Err(
                    "ads2v mode requires a referenceClipAssetId (the reference video)".to_owned(),
                );
            }
            if args
                .reference_asset_ids
                .as_deref()
                .is_none_or(<[String]>::is_empty)
            {
                return Err("ads2v mode requires at least one referenceAssetIds entry".to_owned());
            }
            "ads2v"
        }
        // The character is `referenceAssetIds[0]` (preferred) or the i2v `sourceAssetId`; the
        // motion is `sourceClipAssetId`. Both halves are hard engine inputs.
        "animate_character" => {
            if args.source_clip_asset_id.is_none() {
                return Err(
                    "animate_character mode requires a sourceClipAssetId (the driving video)"
                        .to_owned(),
                );
            }
            if args
                .reference_asset_ids
                .as_deref()
                .is_none_or(<[String]>::is_empty)
                && args.source_asset_id.is_none()
            {
                return Err(
                    "animate_character mode requires a reference character image \
                     (referenceAssetIds or sourceAssetId)"
                        .to_owned(),
                );
            }
            "animate_character"
        }
        other => {
            // The menu is BUILT from `VIDEO_TOOL_MODES`, never retyped: the hand-written one was
            // already a mode stale, and a caller told to use a mode set that omits the one it needs
            // is worse off than one told nothing.
            let menu = VIDEO_TOOL_MODES
                .iter()
                .map(|mode| format!("\"{mode}\""))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("unsupported mode \"{other}\": use {menu}"));
        }
    };
    let mut body = Map::new();
    body.insert("projectId".to_owned(), json!(args.project_id));
    body.insert("mode".to_owned(), json!(mode));
    body.insert("prompt".to_owned(), json!(args.prompt));
    let optional = [
        (
            "negativePrompt",
            args.negative_prompt.as_ref().map(|v| json!(v)),
        ),
        ("model", args.model.as_ref().map(|v| json!(v))),
        ("duration", args.duration.map(|v| json!(v))),
        ("fps", args.fps.map(|v| json!(v))),
        ("width", args.width.map(|v| json!(v))),
        ("height", args.height.map(|v| json!(v))),
        ("quality", args.quality.as_ref().map(|v| json!(v))),
        ("seed", args.seed.map(|v| json!(v))),
        ("loras", args.loras.as_ref().map(|v| json!(v))),
        ("characterId", args.character_id.as_ref().map(|v| json!(v))),
        (
            "sourceAssetId",
            args.source_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "lastFrameAssetId",
            args.last_frame_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "sourceClipAssetId",
            args.source_clip_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "bridgeRightClipAssetId",
            args.bridge_right_clip_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAssetIds",
            args.reference_asset_ids.as_ref().map(|v| json!(v)),
        ),
        (
            "sourceClipAssetIds",
            args.source_clip_asset_ids.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceAudioAssetIds",
            args.reference_audio_asset_ids.as_ref().map(|v| json!(v)),
        ),
        (
            "referenceClipAssetId",
            args.reference_clip_asset_id.as_ref().map(|v| json!(v)),
        ),
        (
            "personTrackId",
            args.person_track_id.as_ref().map(|v| json!(v)),
        ),
        (
            "replacementMode",
            args.replacement_mode.as_ref().map(|v| json!(v)),
        ),
    ];
    for (key, value) in optional {
        if let Some(value) = value {
            body.insert(key.to_owned(), value);
        }
    }
    // `steps` rides in `advanced`, which is a verbatim passthrough map — `requested_steps` reads
    // `advanced.steps` and nothing else, so a top-level key would be silently dropped. Emitted
    // only when the caller named one, so an omitted step count still means "the model's own
    // default" rather than a blanket this tool invented (the same rule the optional keys follow).
    if let Some(steps) = args.steps {
        body.insert("advanced".to_owned(), json!({ "steps": steps }));
    }
    Ok(Value::Object(body))
}

fn moss_image_options_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    sceneworks_core::moss_image::options_schema()
        .try_into()
        .expect("MOSS options schema object")
}
