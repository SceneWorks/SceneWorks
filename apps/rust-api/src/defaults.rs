//! Serde `#[serde(default = "...")]` value providers for the request DTOs.
//!
//! These are the wire-default constants for optional DTO fields, plus the
//! `bool_is_false` skip predicate. They were previously inlined in the crate
//! root grab-bag; collecting them here (sc-8890, F-088) keeps the ~40 tiny
//! providers out of `lib.rs` and next to nothing but each other, while the
//! `pub(crate) use defaults::*` re-export in `lib.rs` keeps every existing
//! `#[serde(default = "default_x")]` string path and sibling call site
//! resolving unchanged.

pub(crate) fn default_timeline_name() -> String {
    "Main timeline".to_owned()
}

pub(crate) fn default_aspect_ratio() -> String {
    "16:9".to_owned()
}

pub(crate) fn default_timeline_fps() -> u32 {
    30
}

pub(crate) fn default_export_resolution() -> u32 {
    720
}

pub(crate) fn default_frame_intended_use() -> String {
    "reuse".to_owned()
}

pub(crate) fn default_requested_gpu() -> String {
    "auto".to_owned()
}

pub(crate) fn default_training_captioner() -> String {
    "joy_caption".to_owned()
}

pub(crate) fn default_training_caption_model() -> String {
    "fancyfeast/llama-joycaption-beta-one-hf-llava".to_owned()
}

pub(crate) fn default_dataset_analysis_embedder() -> String {
    "clip_vit_l14".to_owned()
}

pub(crate) fn default_training_caption_type() -> String {
    "Descriptive".to_owned()
}

pub(crate) fn default_training_caption_length() -> String {
    "long".to_owned()
}

pub(crate) fn default_training_caption_temperature() -> f64 {
    0.6
}

pub(crate) fn default_training_caption_top_p() -> f64 {
    0.9
}

pub(crate) fn default_training_caption_max_new_tokens() -> u32 {
    256
}

pub(crate) fn default_lora_scope() -> String {
    "global".to_owned()
}

pub(crate) fn bool_is_false(value: &bool) -> bool {
    !*value
}

pub(crate) fn default_project_lora_scope() -> String {
    "project".to_owned()
}

pub(crate) fn default_character_type() -> String {
    "person".to_owned()
}

pub(crate) fn default_reference_role() -> String {
    "reference".to_owned()
}

pub(crate) fn default_character_lora_weight() -> f64 {
    0.8
}

pub(crate) fn default_track_name() -> String {
    "Selected person".to_owned()
}

pub(crate) fn default_image_mode() -> String {
    "text_to_image".to_owned()
}

pub(crate) fn default_image_model() -> String {
    "z_image_turbo".to_owned()
}

pub(crate) fn default_image_count() -> u32 {
    4
}

pub(crate) fn default_image_size() -> u32 {
    1024
}

pub(crate) fn default_style_preset() -> String {
    "cinematic".to_owned()
}

pub(crate) fn default_fit_mode() -> String {
    // epic 2551: never stretch by default. "crop" covers the frame undistorted; the
    // worker normalizes unknown values back to crop, so this is just the wire default.
    "crop".to_owned()
}

pub(crate) fn default_video_mode() -> String {
    "image_to_video".to_owned()
}

pub(crate) fn default_video_model() -> String {
    "ltx_2_3".to_owned()
}

/// The default audio model a `POST /api/v1/audio/jobs` resolves when the caller names none
/// (SceneWorks Audio Studio, epic 13400 / sc-13404). Kokoro-82M is the recommended Speech model
/// (`recommended: true` in the seeded manifest), 24 kHz mono TTS, Candle-native on every platform.
pub(crate) fn default_audio_model() -> String {
    "kokoro_82m".to_owned()
}

/// The default base TTS model for a Voice Clone job (sc-13411 C4): the "content" generator whose speech
/// OpenVoice V2 re-timbres toward the reference. Kokoro-82M is the recommended, always-installed Speech
/// model — the base clip's voice is itself a knob (`voice`), but the base generator is Kokoro.
pub(crate) fn default_audio_base_model() -> String {
    "kokoro_82m".to_owned()
}

// No `default_video_duration` either, and for a sharper reason than fps: the blanket 6.0 that used
// to live here is past the `hardMaxDuration` of 7 of the 10 shipped video models, so once sc-12297
// began enforcing that cap, this default made the API reject its own duration-less requests
// (sc-12400). An omitted duration resolves to the model's declared `defaults.duration` via core's
// `resolve_duration` — see `VideoJobRequest::duration`.

// No `default_video_fps`: an omitted fps resolves to the model's declared `defaults.fps` (core's
// `resolve_fps`, applied in `create_video_job` once the manifest entry is resolved), not to a
// blanket this layer invents. The blanket 25 that used to live here was off-menu for 7 of the 10
// shipped video models — see `VideoJobRequest::fps` (sc-12347).

// No `default_video_width` / `default_video_height` either: an omitted side resolves to the model's
// declared `defaults.resolution` (core's `default_resolution`). The blanket 768x512 that used to
// live here is not in `limits.resolutions` for 7 of the 9 shipped video models — e.g. `bernini`
// would render at 768x512 instead of its native, only-trained 848x480 (sc-12400). See `VideoJobRequest::width`.

pub(crate) fn default_video_quality() -> String {
    "balanced".to_owned()
}

pub(crate) fn default_replacement_mode() -> String {
    "face_only".to_owned()
}
