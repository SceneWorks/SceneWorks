//! Canonical generation wire contracts, shared by submission and raw worker validation.
use crate::contracts::{ContractNumber, HiresFixRequest, ImageUpscaleRequest, JsonObject};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageJobRequest<A = Value, M = Value> {
    pub project_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appearance: Option<A>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moss_image_advanced: Option<M>,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default = "default_image_mode")]
    pub mode: String,
    #[serde(default)]
    pub prompt: String,
    /// A NATIVE prompt batch (sc-25681): prompts the model renders together in one pass, only for a
    /// model declaring `image.maxPromptBatchImages` (Iris-3B). `prompt` must equal `prompts[0]`.
    /// Typed so serde forwards it; empty (and omitted from the job payload) for an ordinary request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<String>,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "default_image_model")]
    pub model: String,
    /// Batch size, or `None` when the caller named none — resolved from the model's declared
    /// `defaults.count` in `create_image_job`, not from a blanket 4 that **29 of the 45** image
    /// models contradict with `count: 1` (sc-12427 follow-up).
    ///
    /// `CharacterTestRequest` keeps the blanket: its route resolves no `modelManifestEntry`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    #[serde(default)]
    pub seed: Option<i64>,
    /// Output geometry, or `None` per side when the caller named none — resolved from the model's
    /// declared `defaults.resolution` in `create_image_job`, not from a blanket 1024 square that 5
    /// of the 45 image models do not declare (sc-12400). Mirrors `VideoJobRequest::width`.
    ///
    /// Only this DTO goes optional: `CharacterTestRequest` and `InterleaveJobRequest` keep the
    /// blanket because their routes resolve no `modelManifestEntry`, so there is no declared
    /// default to resolve from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default = "default_style_preset")]
    pub style_preset: String,
    // The Style-catalog id the caller picked (sc-13134): a group id or a sub-style id. A
    // headless/MCP client sends this + a raw prompt and the server folds the catalog style text
    // into `prompt` (the `Subject:`/`Style:` splice). The web app composes the prompt
    // client-side and sends `presetPromptResolvedClientSide` instead, so it omits this top-level
    // id (it records the picked id under `advanced.styleId` for replay) and is never re-folded.
    // Dedicated field over the dead `stylePreset`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_id: Option<String>,
    #[serde(default)]
    pub recipe_preset_id: Option<String>,
    // The web studio seeds a selected preset's LoRAs straight into `loras`, so it — not the
    // server — is authoritative for which preset LoRAs apply. When set, the preset-LoRA merge
    // is skipped so an edited or removed preset LoRA sticks. Headless/API clients omit it and
    // keep the server-side merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_loras_resolved_client_side: Option<bool>,
    // The web studio composes the prompt from the full preset stack (base model preset +
    // stacked general presets) client-side and sends it verbatim, so the server must not
    // re-fold a single preset's prefix/suffix. When set, `preset_prompt` is skipped and the
    // sent `prompt` is authoritative. Headless/API clients that send only recipePresetId
    // omit it and keep the server-side prefix/suffix fold. (epic 11949)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_prompt_resolved_client_side: Option<bool>,
    #[serde(default)]
    pub loras: Vec<Value>,
    #[serde(default)]
    pub character_id: Option<String>,
    #[serde(default)]
    pub character_look_id: Option<String>,
    #[serde(default)]
    pub source_asset_id: Option<String>,
    // Reference image for IP-Adapter (style/identity conditioning) — distinct from
    // source_asset_id (the img2img edit base). Drives the Character Studio
    // "many images from one reference" flow. ip_adapter_scale rides in `advanced`.
    #[serde(default)]
    pub reference_asset_id: Option<String>,
    // Multi-image reference set for a FLUX.2 multi-reference edit (sc-6211 / sc-6107): every id is a
    // reference image that jointly conditions the edit (DiT token concat → Conditioning::MultiReference),
    // distinct from the single `reference_asset_id`. Threaded to the worker payload as-is; the worker's
    // `flux2_edit_reference_ids` consumes a non-empty list (capped at MAX_EDIT_REFERENCES) and prefers it
    // over `source_asset_id`. Must be a typed field: without it serde drops the unknown top-level key on
    // deserialize and `to_json_object` never forwards it, so the references never reach the worker
    // (sc-6358). Mirrors VideoJobRequest.reference_asset_ids.
    #[serde(default)]
    pub reference_asset_ids: Vec<String>,
    // Keep these carriers visible to family validation. FLUX.3 must refuse unsupported strength,
    // not let serde erase it and enqueue a different request (including an explicit 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_strength: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denoising_strength: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_strength: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ip_adapter_scale: Option<Value>,
    #[serde(
        default,
        rename = "ip_adapter_scale",
        skip_serializing_if = "Option::is_none"
    )]
    pub ip_adapter_scale_snake: Option<Value>,
    // Optional inpaint mask asset (sc-2476): white = edit region. Honored only by
    // inpaint-capable models (the `image_inpaint` capability, SDXL family); others
    // ignore it and run the whole-image edit. Threaded to the worker payload as-is.
    #[serde(default)]
    pub mask_asset_id: Option<String>,
    // How the source is fitted to the output W×H on an edit (epic 2551): "crop"
    // (cover, default), "pad" (letterbox), "outpaint" (pad + generate the border —
    // inpaint-capable models only), "stretch" (legacy). Threaded to the worker as-is;
    // the worker normalizes unknown values back to crop.
    #[serde(default = "default_fit_mode")]
    pub fit_mode: String,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    #[serde(default, skip_serializing_if = "ImageUpscaleRequest::is_disabled")]
    pub upscale: ImageUpscaleRequest,
    /// The Image Studio "Hires.fix" toggle (epic 15983): an A1111-style second pass. The web sends
    /// it top-level (`imageJobRequest.js`), so — like `upscale` — it MUST be a typed field: without
    /// one, serde dropped the unknown key on deserialize and `to_json_object` never forwarded it,
    /// so the worker's `parse_hires_fix` never saw the block (the second pass silently never ran,
    /// and the worker's refusals never fired, sc-26644).
    #[serde(default, skip_serializing_if = "HiresFixRequest::is_disabled")]
    pub hires_fix: HiresFixRequest,
    #[serde(default)]
    pub advanced: JsonObject,
    /// Server-authored link for a disclosed composed workflow. Public callers cannot use this to
    /// influence routing; the workflow route creates it after selecting the parent id.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub workflow_parent_id: Option<String>,
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoJobRequest<A = Value> {
    pub project_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appearance: Option<A>,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default = "default_video_mode")]
    pub mode: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "default_video_model")]
    pub model: String,
    /// Original immutable import identity when replaying a recipe. Never a replacement catalog
    /// entry: the server resolves today's plan and refuses if it no longer matches this identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_imported_model_ref: Option<Value>,
    /// Clip length in seconds, or `None` when the caller named none — which the MCP server does
    /// whenever *its* caller does (`server.rs` omits the key rather than sending a default).
    ///
    /// Optional for the same reason as [`VideoJobRequest::fps`], and with a sharper edge: the
    /// blanket 6.0 that used to live here is past the `hardMaxDuration` of 7 of the 10 shipped video
    /// models, and sc-12297's cap reads whatever lands in the payload — so a request that named NO
    /// duration was rejected for "asking for 6s". `create_video_job` resolves `None` from the
    /// model's declared `defaults.duration` once the manifest entry is in hand (sc-12400).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<ContractNumber>,
    /// Frames per second, or `None` when the caller named none — which the MCP server does whenever
    /// *its* caller does (`server.rs` omits the key entirely rather than sending a default).
    ///
    /// Deliberately optional rather than `#[serde(default = …)]` = 25: that blanket is off-menu for
    /// 7 of the 10 shipped video models, so it silently rendered them at a rate they do not declare
    /// (mochi_1 at 25 instead of 30 — a 30 fps prior played 20% slow), and it is not a value any
    /// model chose. `create_video_job` resolves `None` from the model's declared `defaults.fps` once
    /// the manifest entry is in hand, then writes the resolved rate back into the payload. Keeping
    /// the blanket here would also make `limits.fps` unenforceable: the gate would reject a payload
    /// this route constructed itself (sc-12347).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Output geometry, or `None` per side when the caller named none — resolved from the model's
    /// declared `defaults.resolution` in `create_video_job`, not from a blanket 768×512 that 8 of
    /// the 10 shipped video models do not advertise anywhere (sc-12400). See
    /// [`VideoJobRequest::fps`] for the pattern.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default = "default_video_quality")]
    pub quality: String,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub recipe_preset_id: Option<String>,
    // See ImageJobRequest::preset_loras_resolved_client_side — the studio seeds preset LoRAs
    // into `loras`, so the server skips its merge when this is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_loras_resolved_client_side: Option<bool>,
    // See ImageJobRequest::preset_prompt_resolved_client_side — the studio composes the
    // preset-stack prompt client-side, so the server skips its prefix/suffix fold when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_prompt_resolved_client_side: Option<bool>,
    #[serde(default)]
    pub loras: Vec<Value>,
    #[serde(default)]
    pub character_id: Option<String>,
    #[serde(default)]
    pub character_look_id: Option<String>,
    #[serde(default)]
    pub person_track_id: Option<String>,
    #[serde(default = "default_replacement_mode")]
    pub replacement_mode: String,
    #[serde(default)]
    pub source_asset_id: Option<String>,
    // How the starting image is fitted to the output W×H for the image-conditioned
    // modes (image_to_video / first_last_frame), mirroring Image Studio Edit (sc-6139):
    // "crop" (cover, default) or "pad" (letterbox) — never distort. Threaded to the
    // worker as-is; the worker normalizes unknown values back to crop. Outpaint is
    // inpaint-only and not offered for video.
    #[serde(default = "default_fit_mode")]
    pub fit_mode: String,
    #[serde(default)]
    pub last_frame_asset_id: Option<String>,
    #[serde(default)]
    pub source_clip_asset_id: Option<String>,
    /// Multiple source clips for Bernini's multi-source-video edit mode
    /// (`multi_video_to_video` / mv2v, sc-5425). The worker pushes one
    /// `Conditioning::VideoClip` per clip; v2v/rv2v/ads2v use the single
    /// `source_clip_asset_id` instead.
    #[serde(default)]
    pub source_clip_asset_ids: Vec<String>,
    #[serde(default)]
    pub bridge_right_clip_asset_id: Option<String>,
    /// Subject reference images for Bernini's reference-driven video modes
    /// (`reference_to_video` / `reference_video_to_video` / `ads2v`, sc-4703 /
    /// sc-5425). Carried as a list so the mode can supply multiple references; the
    /// worker VAE/ViT-encodes each into the planner's `MultiReference` conditioning.
    #[serde(default)]
    pub reference_asset_ids: Vec<String>,
    /// The reference *video* slot for Bernini's `ads2v` mode (sc-5425): a second
    /// source video distinct from `source_clip_asset_id` (the clip being edited).
    /// The worker pushes it as a second `Conditioning::VideoClip`.
    #[serde(default)]
    pub reference_clip_asset_id: Option<String>,
    /// Reference **audio** clips for a multi-modal reference video mode (sc-17160,
    /// MiniMax-H3 Ref2VA). The audio sibling of [`Self::reference_asset_ids`]: the worker
    /// resolves each id through the project-scoped asset guard and pushes one
    /// `Conditioning::ReferenceAudio` per entry.
    ///
    /// Serialized unconditionally, exactly like the two id lists above, so the enqueued
    /// payload has one shape and a replay reader never has to distinguish "absent" from
    /// "empty" (the sc-12345 lesson).
    ///
    /// Bounded twice: by the payload-sanity blanket `MAX_VIDEO_REFERENCE_AUDIO_ASSET_IDS`
    /// in `validate_video_job`, and — the binding one — by the model's declared
    /// `limits.maxReferenceAudioAssets`, which defaults to 0 so no already-shipped video
    /// model accepts one.
    #[serde(default)]
    pub reference_audio_asset_ids: Vec<String>,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    #[serde(default)]
    pub advanced: JsonObject,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSegmentDto {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioJobRequest {
    pub project_id: String,
    #[serde(default)]
    pub project_name: Option<String>,
    pub prompt: String,
    #[serde(default = "default_audio_model")]
    pub model: String,
    /// Voice / speaker id (e.g. Kokoro's `af_heart`). `None` ⇒ the model's default voice. Gated
    /// against the model's advertised `audio.voices` by the worker's shared validation floor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Language code in the manifest's BCP-47 display casing (e.g. `"en-US"`). The worker lowercases
    /// it to what the Generator accepts (`en` / `en-us` / `en-gb`) — the language-casing seam
    /// (sc-13404). `None` ⇒ the model derives the accent from the voice id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Requested clip length in seconds. `None` ⇒ the model's natural length. Range-checked against
    /// the model's advertised `audio.maxDurationSecs` by the worker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_duration_secs: Option<f32>,
    /// A multi-speaker / long-form dialogue script (sc-13676) — an ordered list of spoken segments,
    /// each with its own text and an optional speaker label. Rides the worker-side
    /// [`gen_core::AudioParams::script`], which a model advertising
    /// [`gen_core::Capabilities::supports_multi_speaker`] (MOSS-TTSD) renders as one clip with each
    /// turn in its own voice. `None` (the default) is an ordinary single-voice request, byte-for-byte
    /// unaffected: a script sent to a model that does NOT advertise multi-speaker is a typed
    /// Unsupported at the gen-core floor. When a non-empty script is present the `prompt` may be empty
    /// (the script carries the text); a single-voice request still requires a prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<Vec<SpeechSegmentDto>>,
    /// CFG guidance scale for diffusion-audio models (Sound FX / MOSS-SoundEffect). Rides the
    /// top-level [`gen_core::GenerationRequest::guidance`] the worker builds. `None` ⇒ the model's
    /// default (MOSS: 4.0). Bounded to a blanket sane range here; the model's advertised guidance
    /// range is the real gate the generator's `validate` applies (sc-13409).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance: Option<f32>,
    /// Optional native flow schedule shift; omitted preserves the model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler_shift: Option<f32>,
    /// Solver step count for diffusion-audio models (Sound FX / MOSS-SoundEffect + Music / ACE-Step).
    /// Rides the top-level [`gen_core::GenerationRequest::steps`]. `None` ⇒ the model's default (MOSS:
    /// 100; ACE-Step turbo: 8). Bounded to a blanket sane ceiling here; the model's advertised step
    /// ceiling is the real gate the generator's `validate` applies (sc-13409 / sc-13410).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    /// Negative prompt — the traits to steer away from (music models that advertise it). Rides the
    /// top-level [`gen_core::GenerationRequest::negative_prompt`]. `None` ⇒ unconditional. The
    /// guidance-distilled ACE-Step turbo advertises no negative-prompt support, so the studio never
    /// sends one to it; a value posted to such a model is a typed Unsupported at the gen-core floor
    /// (the "honor advertised knobs, let the model gate" contract). sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Musical tempo in beats per minute (music models — ACE-Step). Rides the worker-side
    /// [`gen_core::AudioParams::bpm`]. `None` ⇒ the model derives its own tempo. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpm: Option<f32>,
    /// Musical key (e.g. `"C minor"`; music models). Free-form — rides
    /// [`gen_core::AudioParams::musical_key`]; each model documents what it accepts. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musical_key: Option<String>,
    /// Lyrics to sing / condition on (music models). Free-form, distinct from `prompt` — rides
    /// [`gen_core::AudioParams::lyrics`]; empty ⇒ instrumental. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lyrics: Option<String>,
    /// Extend/edit SOURCE track (a library `type: "audio"` asset id) for prompted source-audio editing
    /// (music models that advertise `audio.editModes`). The worker resolves the asset's WAV and builds a
    /// [`gen_core::Conditioning::AudioEdit`] from it. `None` ⇒ plain text-to-music. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_audio_asset_id: Option<String>,
    /// The edit operation for the source track — `"inpaint"` / `"repaint"` / `"extend"` / `"cover"`,
    /// gated against the model's advertised [`gen_core::Capabilities::audio_edit_modes`] by the
    /// generator's `validate`. Only consulted when `source_audio_asset_id` is set. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_mode: Option<String>,
    /// Edit-region start (seconds) for a bounded inpaint/repaint window. For `extend` the worker
    /// defaults it to the source clip's own length. `None` ⇒ the provider default. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_region_start_secs: Option<f32>,
    /// Edit-region end (seconds): the inpaint/repaint window end, OR — for `extend` — the appended
    /// clip's new TOTAL length. `None` ⇒ to the clip end (region modes). sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_region_end_secs: Option<f32>,
    /// Edit strength (0..=1) for the source-audio edit. `None` ⇒ the model default. sc-13410.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit_strength: Option<f32>,
    /// Voice Clone (sc-13411 C4): the reference-voice library `type: "audio"` asset id whose timbre is
    /// transferred onto the base TTS clip. Its presence routes the job onto the two-call
    /// Kokoro→OpenVoice chain (the worker's `run_voice_clone_synthesis`); `None` ⇒ an ordinary
    /// Speech/SFX/Music generation. The route additionally resolves + injects the base TTS model's
    /// manifest entry (`baseModelManifestEntry`) when this is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_audio_asset_id: Option<String>,
    /// Voice Clone base TTS model id — the "content" generator (Kokoro) whose speech OpenVoice re-timbres.
    /// Defaults to `kokoro_82m`. Only consulted on a voice-clone request; the route asserts it is a
    /// `type: "audio"` model and injects its manifest entry as `baseModelManifestEntry`.
    #[serde(default = "default_audio_base_model")]
    pub base_model: String,
    /// Voice Clone match strength — overrides OpenVoice V2's posterior-sampling temperature τ (rides the
    /// worker-side `AudioTransformRequest::strength`). `None` ⇒ the converter's own default (0.3).
    /// Bounded to a blanket sane 0..=1 range here; the converter re-checks it (finite, >= 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_strength: Option<f32>,
    /// Segmented-song controls (YuE lyrics2song, sc-19384). `segments` = how many lyric segments to
    /// render, `maxNewTokensPerSegment` = the stage-1 token budget per segment, `repetitionPenalty` =
    /// the stage-1 sampler penalty. Ride the worker-side `AudioParams` fields of the same names;
    /// accepted only for a model whose manifest advertises `audio.supportsSegmentedLyrics` /
    /// `audio.supportsRepetitionPenalty`. `None` ⇒ the model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_new_tokens_per_segment: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Guidance on/off (sc-19384). `false` switches classifier-free guidance off (the worker sends a
    /// `0.0` scale, which a segmented-song model reads as "no CFG") and may not be combined with a
    /// `guidance` scale; `true`/absent sends `guidance` as given (absent ⇒ the model's schedule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guidance_enabled: Option<bool>,
    /// In-context-learning reference mode for a YuE `_icl` checkpoint (sc-19384): `"single"` (one
    /// mixed clip, `iclReferenceAssetId`) or `"dual"` (`iclVocalAssetId` + `iclInstrumentalAssetId`).
    /// Library `type: "audio"` asset ids, resolved project-scoped by the worker. Distinct from
    /// `referenceAudioAssetId`, which routes a job onto the voice-clone chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icl_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icl_reference_asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icl_vocal_asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icl_instrumental_asset_id: Option<String>,
    /// The window of the ICL reference the prompt uses, in seconds (sc-19384). `None` ⇒ the model's
    /// default window (YuE: 0–30 s); an absent end ⇒ to the clip end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icl_start_secs: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icl_end_secs: Option<f32>,
    /// Weight tier (`bf16` / `q8` / `q4`) for a model that ships physical per-tier downloads (YuE,
    /// sc-19384). Must name one of the model's `downloads[].variant` tiers. `None` ⇒ the default
    /// tier when installed, else the first installed tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_tier: Option<String>,
    /// Output limiter for a segmented-song model (YuE, upstream `--rescale`, sc-19384): `"clamp"`
    /// (hard clip at ±0.99, the default) or `"rescale"` (scale the whole stem so its peak is 0.99).
    /// `None` ⇒ the model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_limiter: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    #[serde(default)]
    pub advanced: JsonObject,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VqaJobRequest {
    pub project_id: String,
    #[serde(default)]
    pub project_name: Option<String>,
    pub source_asset_id: String,
    pub question: String,
    #[serde(default = "default_vqa_model")]
    pub model: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_json_value"
    )]
    pub max_new_tokens: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_json_value"
    )]
    pub max_image_pixels: Option<Value>,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    #[serde(default)]
    pub advanced: JsonObject,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InterleaveJobRequest {
    pub project_id: String,
    #[serde(default)]
    pub project_name: Option<String>,
    pub prompt: String,
    // Optional input images for grounded (it2i) interleaved generation.
    #[serde(default)]
    pub source_asset_ids: Vec<String>,
    #[serde(default = "default_interleave_model")]
    pub model: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_json_value"
    )]
    pub max_images: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_json_value"
    )]
    pub width: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_json_value"
    )]
    pub height: Option<Value>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    #[serde(default)]
    pub advanced: JsonObject,
}

// A present null is authored data, distinct from an omitted carrier.
fn present_json_value<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

fn default_audio_base_model() -> String {
    "kokoro_82m".to_owned()
}

fn default_audio_model() -> String {
    "kokoro_82m".to_owned()
}

fn default_fit_mode() -> String {
    // epic 2551: never stretch by default. "crop" covers the frame undistorted; the
    // worker normalizes unknown values back to crop, so this is just the wire default.
    "crop".to_owned()
}

fn default_image_mode() -> String {
    "text_to_image".to_owned()
}

fn default_image_model() -> String {
    "z_image_turbo".to_owned()
}

fn default_image_size() -> u32 {
    1024
}

fn default_interleave_model() -> String {
    "sensenova_u1_8b".to_owned()
}

fn default_replacement_mode() -> String {
    "face_only".to_owned()
}

fn default_requested_gpu() -> String {
    "auto".to_owned()
}

fn default_style_preset() -> String {
    "cinematic".to_owned()
}

fn default_video_mode() -> String {
    "image_to_video".to_owned()
}

fn default_video_model() -> String {
    "ltx_2_3".to_owned()
}

fn default_video_quality() -> String {
    "balanced".to_owned()
}

fn default_vqa_model() -> String {
    "sensenova_u1_8b".to_owned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorMode {
    ImageToSvg,
    TextToSvg,
}

impl VectorMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ImageToSvg => "image_to_svg",
            Self::TextToSvg => "text_to_svg",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub repetition_context: u32,
    pub seed: Option<u64>,
}

impl Default for VectorSampling {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorDetailBudget {
    pub max_new_tokens: u32,
    pub max_svg_bytes: u32,
    pub max_wall_time_ms: u64,
}

pub const MAX_VECTOR_WALL_TIME_MS: u64 = 300_000;

impl Default for VectorDetailBudget {
    fn default() -> Self {
        Self {
            // Omitted requests retain the product's conservative cost default. Detailed callers
            // explicitly select the advertised model-specific context-derived capacity.
            max_new_tokens: 4_000,
            max_svg_bytes: 256 * 1_024,
            max_wall_time_ms: MAX_VECTOR_WALL_TIME_MS,
        }
    }
}

/// Typed Vector Studio request. SVG source is provider output only; callers can supply raster and
/// text conditioning, but never raw SVG for the worker to publish.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorRequest {
    pub project_id: String,
    #[serde(default)]
    pub project_name: Option<String>,
    pub mode: VectorMode,
    pub model: String,
    #[serde(default)]
    pub source_asset_id: Option<String>,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub sampling: VectorSampling,
    #[serde(default)]
    pub detail_budget: VectorDetailBudget,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptRefineRequest {
    // Optional so the `image_caption` task (epic 8102, sc-8108) — driven by a reference image, not
    // text — can omit it. The handler still requires a non-empty prompt for every other task.
    #[serde(default)]
    pub prompt: String,
    pub model_id: Option<String>,
    pub workflow: Option<String>,
    /// The selected model's Markdown prompt guide, sent by the client so the
    /// worker can build a guide-aware system prompt without filesystem access to
    /// the web assets.
    pub guide: Option<String>,
    /// `"refine"` (default), `"magic_prompt"` (expand a plain idea into a structured
    /// Ideogram 4 JSON caption, epic 4725/sc-5997), `"image_caption"` (turn a
    /// reference image into the same structured caption via the vision model, epic
    /// 8102/sc-8108), or `"image_describe"` (turn a reference image into a plain-text
    /// description for a non-structured t2i model, epic 8203/sc-8206). The two vision
    /// tasks carry a `source_asset_id` + `project_id` instead of a prompt; the handler
    /// resolves them to the worker's confined `imagePath`.
    pub task: Option<String>,
    /// Target image aspect ratio as `"W:H"` (magic-prompt only); drives bbox/layout
    /// decisions in the caption. Defaults to `"1:1"` worker-side when absent.
    pub aspect_ratio: Option<String>,
    /// Reference image for the vision tasks (`image_caption` / `image_describe`): a project
    /// asset id resolved to a confined on-disk path before the job is enqueued (epic 8102/8203).
    pub source_asset_id: Option<String>,
    /// Multiple reference images for a "mood board" synthesis (epic 8588, sc-8595): the vision task
    /// examines them together and emits ONE prompt/caption capturing the aesthetic they share. When
    /// non-empty this takes precedence over `source_asset_id`; each id is resolved to a confined path
    /// and forwarded as the worker's `imagePaths` array. Empty/absent → the single-image path above.
    #[serde(default)]
    pub source_asset_ids: Vec<String>,
    /// Project owning `source_asset_id` (required for the vision tasks so the
    /// asset's relative `file.path` can be resolved to an absolute path).
    pub project_id: Option<String>,
    /// HF repo string of the model the worker should load (the worker resolves by repo,
    /// not by catalog id). The web sends the vision model's repo for the vision tasks.
    pub model: Option<String>,
    /// Exact installed source model tier, when the utility offers multiple tiers.
    pub variant: Option<String>,
    /// Native planner reasoning control (`disabled`, `enabled`, or `auto`). Only the film-plan task
    /// forwards it; other prompt-refine products retain their established model defaults.
    pub thinking_mode: Option<String>,
    /// Plain-text describe style for `image_describe` (epic 8203, sc-8205): `"prose"`
    /// (default) or `"tags"` (booru/danbooru tags for anime SDXL checkpoints). Forwarded
    /// verbatim; the worker parses it (unknown/absent → prose).
    pub caption_style: Option<String>,
    /// Compressed KV cache opt-in for this generation (sc-20682): `"off"` or `"qualified"`.
    /// Absent leaves the worker's own default (`SCENEWORKS_LLM_KV_COMPRESSION`, off unless set).
    /// `"qualified"` never forces compression: the engine runs compressed only for a single
    /// sequence on a model matching one of its two measured architectures (Llama-3.2-3B-Instruct,
    /// dense Qwen3-1.7B) within that row's context range (`docs/kv-baseline-harness.md`); batched,
    /// short and other-model requests run dense with a reason, and the job result's
    /// `generation.kvCache` records what ran.
    pub kv_compression: Option<String>,
    /// MiniMax-H3-Image prompt expansion only (SW-7208): the model the expander runs,
    /// `minimax_h3_image_pe` (MiniMax-H3-Image-PE) or `vision_describe_qwen3_8_27b` (the
    /// abliterated Qwen3.8-27B the dataset describer uses). Absent takes the saved
    /// `h3ImageExpanderModel` UI preference, else MiniMax-H3-Image-PE.
    pub expander_model_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterTestRequest {
    pub prompt: String,
    #[serde(default = "default_image_model")]
    pub model: String,
    #[serde(default = "default_image_count")]
    pub count: u32,
    #[serde(default = "default_image_size")]
    pub width: u32,
    #[serde(default = "default_image_size")]
    pub height: u32,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    #[serde(default)]
    pub look_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorPromptWorkflowRequest {
    pub project_id: String,
    #[serde(default)]
    pub project_name: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub raster_model: String,
    pub vector_model: String,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default)]
    pub sampling: VectorSampling,
    #[serde(default)]
    pub detail_budget: VectorDetailBudget,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
    /// Optional immutable identities used only by replay. Fresh requests omit them; the API stamps
    /// authoritative revisions. Replay supplies them and fails typed if the catalog has moved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_raster_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_vector_revision: Option<String>,
}
fn default_image_count() -> u32 {
    4
}

/// Wire defaults used in refusal context before a malformed request can be deserialized.
pub(crate) fn default_model_for(kind: &str) -> Option<String> {
    match kind {
        "image" => Some(default_image_model()),
        "video" => Some(default_video_model()),
        "audio" => Some(default_audio_model()),
        "vqa" => Some(default_vqa_model()),
        "interleave" => Some(default_interleave_model()),
        _ => None,
    }
}
pub(crate) fn default_mode_for(kind: &str) -> Option<String> {
    match kind {
        "image" => Some(default_image_mode()),
        "video" => Some(default_video_mode()),
        _ => None,
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonDetectionJobRequest {
    #[serde(default)]
    pub advanced: JsonObject,
    #[serde(default)]
    pub confidence: Option<Value>,
    pub source_asset_id: String,
    #[serde(default)]
    pub source_timestamp: Option<f64>,
    /// Opt into the Rust utility worker's procedural preview instead of real,
    /// model-backed detection on the native MLX/candle worker. Defaults to real.
    #[serde(default)]
    pub preview: bool,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonTrackJobRequest {
    #[serde(default)]
    pub advanced: JsonObject,
    #[serde(default)]
    pub confidence: Option<Value>,
    pub source_asset_id: String,
    pub representative_frame_asset_id: String,
    pub detection: JsonObject,
    #[serde(default = "default_track_name")]
    pub track_name: String,
    /// Opt into the Rust utility worker's procedural preview instead of real,
    /// model-backed tracking on the native MLX/candle worker. Defaults to real.
    #[serde(default)]
    pub preview: bool,
    #[serde(default = "default_requested_gpu")]
    pub requested_gpu: String,
}

pub fn default_track_name() -> String {
    "Selected person".to_owned()
}
