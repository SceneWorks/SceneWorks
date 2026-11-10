//! Native prompt refinement (epic 5095): candle on Windows/CUDA (sc-5525) + MLX on macOS (sc-5552).
//!
//! Routes the `prompt_refine` job to a native LLM provider through the unified LLM engine (epic 7153)
//! — a generic `core_llm::TextLlm` resolved model-first via `crate::inference_runtime::load_for_model`
//! (Anubis-Mini-8B, sc-6550), so the dispatch body is one backend-agnostic path: macOS (sc-7158) picks
//! mlx-llm's `mlx-llama`, the Windows/CUDA candle build (sc-7404) picks candle-llm's `candle-llama`.
//! Both retired their bespoke hand-rolled Llama decoders (`mlx-gen-prompt-refine` /
//! `candle-gen-prompt-refine`). Every shipping platform now has a native provider; a build with
//! neither (e.g. a candle-less Linux worker) simply never advertises the `prompt_refine`
//! capability, so the job is never dispatched to it.
//!
//! The `TextLlm` contract is generic (`system` + `prompt` + sampling → text), so the
//! prompt-refinement PRODUCT logic that lived in `prompt_refine.py` moves here caller-side: the
//! rewrite rules + image/video/audio medium switch + guide assembly (`build_refine_system_prompt`, into the
//! request `system`) and the reasoning-block / code-fence / surrounding-quote cleanup
//! (`clean_refine_output`, over the model reply). Sampling matches the Python path (temperature 0.7,
//! top_p 0.9), as does the empty-output → error behavior and the `{originalPrompt, refinedPrompt}`
//! result shape. The output budget no longer does: the Python path's 512 tokens could not cover the
//! `MAX_PROMPT_CHARS` contract once this task began carrying the per-shot film refine (sc-24029) —
//! see `DEFAULT_REFINE_MAX_NEW_TOKENS`.

use super::*;
#[cfg(native_lane)]
#[path = "prompt_refine_parse.rs"]
mod parse;
#[cfg(native_lane)]
use parse::{parse_refine_request, RefineRequest};
#[cfg(native_lane)]
#[path = "prompt_refine_finalize.rs"]
mod finalize;
#[cfg(any(test, native_lane))]
pub(crate) use crate::text_clean::clean_json_output;
#[cfg(any(test, native_lane))]
use crate::text_clean::{first_ci, strip_reasoning_and_fence};
#[cfg(native_lane)]
use finalize::{finalize_refine_job, RefineCompletion};

// The prompt-refinement / magic-prompt checkpoint — the coherent Anubis-8B (sc-6550 bake-off). It
// serves BOTH the free-text "Refine my prompt" rewrite AND the Ideogram magic-prompt JSON caption:
// the bake-off found the old 3B (and plain Llama-3.1-8B, stock + abliterated) stochastically emit
// SEMANTICALLY-DEGENERATE captions (subject as a `text` element, a reflexive transparent background)
// that placeholder Ideogram 4 at 1024²/48, while Anubis avoids them. Loads on this same config-driven
// Llama seam with no conversion (stock bf16, ~16GB). Overridable per-job via `payload.model`.
#[cfg(native_lane)]
const DEFAULT_REFINE_MODEL: &str = "TheDrummer/Anubis-Mini-8B-v1";
#[cfg(native_lane)]
const CANCEL_MESSAGE: &str = "Prompt refinement canceled by user.";
// Output-length cap. The two caption tasks (magic-prompt + image_caption) emit a full Ideogram JSON
// caption — multi-element, with bboxes, #RRGGBB palettes and (since sc-8199) optional per-element
// palettes — which truncates well past 2048 tokens on busy images (sc-8210: `EOF while parsing a
// list` mid-`elements`). 4096 ≈ ~11.6k chars of headroom; a well-formed caption emits EOS far below
// the cap, so a higher ceiling only rescues the truncating cases. Callers may still override via the
// `maxNewTokens` payload field.
#[cfg(any(test, native_lane))]
const DEFAULT_CAPTION_MAX_NEW_TOKENS: u32 = 4096;
// The free-text rewrite was budgeted as a one-liner (512 tokens) when its only caller refined a
// single sentence. It is now also the per-shot film refine (`film_planner.rs` sends no `task`, so a
// shot prompt classifies as `Rewrite`), whose output contract is `MAX_PROMPT_CHARS` = 4000 chars of
// multi-field MiniMax-H3 prose.
//
// Sized from MEASUREMENT, not from a nominal ratio: sc-24029's six shot rewrites at the old 512-token
// cap emitted 1584-2922 chars (`a5-image/smoke.log` in that story's evidence dir), i.e. this
// refiner's prose runs ~5.7 chars/token and 512 tokens tops out around 2900 chars — BELOW the
// 4000-char contract, so a rewrite with more to say stopped on `MaxTokens` mid-sentence rather than
// on EOS (a film shot shipped cut at 3026/4000 chars). At that measured ratio the contract needs only
// ~700 tokens; 1536 is a little over twice that, so the contract is reachable with real headroom for
// a denser rewrite. A rewrite that has said what it has to say still emits EOS far below the cap, so
// this only rescues the truncating cases.
//
// `the_rewrite_budget_covers_the_refined_prompt_char_contract` guards it at 2.83 chars/token, which
// is deliberately NOT the prose ratio above: it is the conservative floor the JSON-caption path
// implies (4096 ≈ ~11.6k chars), so the assertion holds even for the densest output this task can
// emit. 1536 clears that floor too (~4350 chars).
#[cfg(any(test, native_lane))]
const DEFAULT_REFINE_MAX_NEW_TOKENS: u32 = 1536;
// The prose/tags `image_describe` task (epic 8203) is shorter than a full structured JSON caption but
// longer than a one-line rewrite; give it a generous budget so a detailed paragraph is never truncated.
#[cfg(any(test, native_lane))]
const DEFAULT_DESCRIBE_MAX_NEW_TOKENS: u32 = 1024;
// A whole shot plan is the longest reply this job ever emits — one JSON object carrying every shot's
// prompt, states, framing and sound (sc-22713). Truncation there is not a degraded plan but an
// unparseable one, so it takes the same generous ceiling the captions take.
#[cfg(any(test, native_lane))]
const DEFAULT_FILM_PLAN_MAX_NEW_TOKENS: u32 = 4096;
// Resolve the output token budget: an explicit positive `maxNewTokens` override wins; otherwise the
// per-task default (the JSON tasks need the larger cap so the object closes; the prose-describe task
// sits in between). Keyed on the task value itself rather than on booleans re-derived per call site.
#[cfg(any(test, native_lane))]
fn resolve_max_new_tokens(
    payload: &serde_json::Map<String, Value>,
    task: RefineTask,
    // sc-24113: the Qwen rewrite's budget is PER REWRITER (the two cards publish different
    // ceilings), and which rewriter applies is a property of the REQUEST, not of the task. `None`
    // for every other task, and for a rewrite it falls back to the smaller of the two — never to
    // the worker's caption ceiling, which is far below either card.
    rewriter: Option<crate::qwen_prompt_rewrite::Rewriter>,
) -> u32 {
    payload
        .get("maxNewTokens")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(match task {
            RefineTask::MagicPrompt | RefineTask::ImageCaption => DEFAULT_CAPTION_MAX_NEW_TOKENS,
            RefineTask::FilmPlan => DEFAULT_FILM_PLAN_MAX_NEW_TOKENS,
            RefineTask::QwenImageRewrite => rewriter
                .unwrap_or(crate::qwen_prompt_rewrite::Rewriter::TextToImage)
                .max_new_tokens(),
            RefineTask::ImageDescribe => DEFAULT_DESCRIBE_MAX_NEW_TOKENS,
            RefineTask::Rewrite => DEFAULT_REFINE_MAX_NEW_TOKENS,
        })
}

/// Apply the JSON grammar only when it cannot capture a reasoning model inside its thinking
/// channel. Qwen3.6 opens `<think>` in the rendered generation prompt for both `Enabled` and
/// `Auto`; a JSON mask then rejects the `</think>` marker, so the complete JSON value lands in
/// `TextLlmOutput::thinking` and the final-answer text is empty. A no-think request renders a closed
/// reasoning block before generation and remains safe to constrain. Models without a thinking mode
/// (including the default Anubis refiner) also keep the existing JSON constraint in every mode.
#[cfg(any(test, native_lane))]
pub(crate) fn json_decode_constraint(
    emits_json: bool,
    supports_thinking: bool,
    thinking_mode: gen_core::core_llm::ThinkingMode,
) -> Option<gen_core::core_llm::Constraint> {
    use gen_core::core_llm::{Constraint, ThinkingMode};

    (emits_json
        && !(supports_thinking
            && matches!(thinking_mode, ThinkingMode::Enabled | ThinkingMode::Auto)))
    .then_some(Constraint::Json)
}

/// The `prompt_refine` job multiplexed FOUR tasks through five scattered booleans
/// (`is_magic`/`is_image_caption`/`is_image_describe`/`is_vision_task`/`is_caption_task`) re-derived in
/// six ladders — a shape where incoherent combinations were representable and which hid the F-003
/// cancel bug (sc-8921, F-119). This enum makes the task ONE value classified once from the payload's
/// `task` discriminator; every per-task property below is a method on it, so the dispatch body reads a
/// single `match` instead of re-testing booleans. The four variants are exactly the four real tasks:
/// - [`Rewrite`](RefineTask::Rewrite): the default free-text "refine my prompt" rewrite (warmer sampling).
/// - [`MagicPrompt`](RefineTask::MagicPrompt): a text idea → JSON Ideogram caption (sc-5997).
/// - [`ImageCaption`](RefineTask::ImageCaption): a reference image → JSON Ideogram caption (sc-8105).
/// - [`ImageDescribe`](RefineTask::ImageDescribe): a reference image → NL prose/tags description (sc-8204).
#[cfg(any(test, native_lane))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefineTask {
    Rewrite,
    MagicPrompt,
    ImageCaption,
    ImageDescribe,
    /// A brief -> ONE strict JSON shot plan (sc-22713, epic 22708). Text in, JSON out, no image.
    /// Selected by the `task` discriminator like the caption tasks, because a plan is a different
    /// JOB from a prompt rewrite rather than a different target model.
    FilmPlan,
    /// Qwen-Image 2.1's OFFICIAL prompt rewriting (sc-24113, epic 24107): the user's brief or edit
    /// instruction → one strict JSON object carrying a rewritten prompt AND an aspect-ratio
    /// suggestion, produced by whichever of the two published PE checkpoints the request selects.
    ///
    /// It is a task rather than a target-model branch of [`Rewrite`](RefineTask::Rewrite) for the
    /// same reason [`FilmPlan`](RefineTask::FilmPlan) is: it runs a DIFFERENT checkpoint with a
    /// DIFFERENT (vendor-frozen) system prompt and returns a structured document, none of which the
    /// generic medium-keyed rewrite rules describe. See [`crate::qwen_prompt_rewrite`].
    QwenImageRewrite,
}

#[cfg(any(test, native_lane))]
impl RefineTask {
    /// Classify the payload's `task` discriminator (case-insensitive). Any unknown/absent value is the
    /// default free-text [`Rewrite`](RefineTask::Rewrite) — the pre-refactor behavior.
    pub(crate) fn from_payload(task: Option<&str>) -> Self {
        match task.map(str::trim).unwrap_or_default() {
            t if t.eq_ignore_ascii_case("magic_prompt") => RefineTask::MagicPrompt,
            t if t.eq_ignore_ascii_case("image_caption") => RefineTask::ImageCaption,
            t if t.eq_ignore_ascii_case("image_describe") => RefineTask::ImageDescribe,
            t if t.eq_ignore_ascii_case("film_plan") => RefineTask::FilmPlan,
            t if t.eq_ignore_ascii_case(crate::qwen_prompt_rewrite::REWRITE_TASK) => {
                RefineTask::QwenImageRewrite
            }
            _ => RefineTask::Rewrite,
        }
    }

    /// A vision task is driven by a reference image (`Content::Image` user block), not a text prompt —
    /// so it requires an `imagePath`/`referencePath` and no `prompt`.
    pub(crate) fn is_vision(self) -> bool {
        matches!(self, RefineTask::ImageCaption | RefineTask::ImageDescribe)
    }

    /// Whether the task READS reference images when the payload carries them.
    ///
    /// A superset of [`is_vision`](RefineTask::is_vision), and the distinction is the point. A
    /// vision task is *driven* by its images: no prompt is required and at least one image must be
    /// present. The Qwen rewrite is driven by the user's TEXT and merely *accepts* images — the
    /// prompt is always required, and zero references is not an error but the signal that selects
    /// the text-to-image rewriter instead of the editing one.
    pub(crate) fn takes_reference_images(self) -> bool {
        self.is_vision() || matches!(self, RefineTask::QwenImageRewrite)
    }

    /// A caption task emits a JSON-constrained Ideogram caption (the decode is grammar-masked) and is
    /// validated as a schema-valid caption. `image_describe` is a vision task but NOT a caption task
    /// (it emits unconstrained prose/tags).
    pub(crate) fn is_caption(self) -> bool {
        matches!(self, RefineTask::MagicPrompt | RefineTask::ImageCaption)
    }

    /// Whether the reply is a JSON document rather than prose, and so is cleaned by isolating the
    /// outermost `{ … }` instead of unwrapping surrounding quotes. The caption tasks and the film
    /// plan are; the free-text rewrite and the prose describe are not.
    pub(crate) fn emits_json(self) -> bool {
        self.is_caption() || matches!(self, RefineTask::FilmPlan | RefineTask::QwenImageRewrite)
    }

    /// Sampling temperature: the caption tasks and the prose describe sample cool for a faithful,
    /// steady description; the free-text rewrite stays warmer for creative variation.
    pub(crate) fn temperature(self) -> f32 {
        match self {
            RefineTask::Rewrite => 0.7,
            RefineTask::MagicPrompt | RefineTask::ImageCaption | RefineTask::ImageDescribe => 0.4,
            // A plan is a structured document checked field by field, and a repair round costs a
            // whole decode: sample cooler still so field names and menu values come back exactly.
            RefineTask::FilmPlan => 0.3,
            // NOT a SceneWorks choice: 1.0 is the temperature both PE model cards publish, and the
            // rewriters were tuned at it. Frozen with `top_p`/`top_k` in `qwen_prompt_rewrite`.
            RefineTask::QwenImageRewrite => crate::qwen_prompt_rewrite::REWRITE_TEMPERATURE,
        }
    }

    /// The running-progress message shown while the model decodes.
    pub(crate) fn work_message(self) -> &'static str {
        match self {
            RefineTask::ImageCaption => "Captioning image…",
            RefineTask::ImageDescribe => "Describing image…",
            RefineTask::MagicPrompt => "Expanding to a caption…",
            RefineTask::FilmPlan => "Planning shots…",
            RefineTask::QwenImageRewrite => "Rewriting prompt…",
            RefineTask::Rewrite => "Refining prompt…",
        }
    }

    /// The terminal completion message.
    pub(crate) fn done_message(self) -> &'static str {
        match self {
            RefineTask::MagicPrompt | RefineTask::ImageCaption => "Caption ready.",
            RefineTask::ImageDescribe => "Description ready.",
            RefineTask::FilmPlan => "Shot plan ready.",
            // "Ready to review", not "applied": the rewrite is a suggestion the user edits and
            // accepts. The copy is part of the guarantee that nothing is replaced silently.
            RefineTask::QwenImageRewrite => "Rewritten prompt ready to review.",
            RefineTask::Rewrite => "Prompt refined.",
        }
    }
}
/// The image-carrier invariant the generate closure asserts, as a named predicate (sc-24113).
///
/// Extracted from the `debug_assert!` so it can be tested directly: the assert itself sits AFTER a
/// provider load, so no test in this lane can reach it, and the bug it hid was a wrong predicate
/// rather than a wrong call site.
///
/// **One-directional.** "Images are present ⇒ the task declares it takes them." The converse is
/// deliberately NOT asserted: a Qwen rewrite with zero references is the legitimate text-to-image
/// case, and it is what selects the T2I rewriter over the editing one.
///
/// This replaced `carries_image == is_vision_task`, which held only while the two VISION tasks were
/// the sole image carriers. The Qwen rewrite carries images with `is_vision()` FALSE — it is driven
/// by the user's text and merely accepts pictures — so the equality panicked every I2I rewrite in a
/// debug build.
#[cfg(any(test, native_lane))]
pub(crate) fn image_carrier_invariant_holds(
    carries_image: bool,
    takes_reference_images: bool,
) -> bool {
    !carries_image || takes_reference_images
}

// Architecture-pill label for the streamed progress (mirrors the candle image/video paths): the MLX
// twin on macOS, candle on the Windows candle build.
#[cfg(target_os = "macos")]
const REFINE_BACKEND: &str = "mlx";
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const REFINE_BACKEND: &str = "candle";

// Coalesced progress-post cadence (sc-8840, F-038). The token callback publishes every token into a
// latest-wins watch channel (never blocking generation); the job loop drains only the newest value on
// this tick, so a 4096-token caption emits at most a handful of `update_job` POSTs per second instead
// of thousands of sequential per-token POSTs. 250 ms keeps the progress bar visibly smooth while
// bounding API load and fully decoupling decode speed from API latency (worse over epic-4484 LAN).
#[cfg(native_lane)]
const PROGRESS_POST_INTERVAL: Duration = Duration::from_millis(250);

// ----------------------------------------------------------------------------------------------
// Product logic (pure, platform-independent) — ported from `prompt_refine.py` so the native worker
// (candle + MLX) owns the prompt assembly + reply cleanup the generic `TextLlm` contract does not.
// Compiled in the default `cargo test` gate (so the unit tests below run on every lane) and on the
// macOS + candle builds.
// ----------------------------------------------------------------------------------------------

/// The base rewrite rules with the `{medium}` placeholders filled (`image` / `video` / `audio`).
/// Image-only photographic guidance is attached only to the image medium; audio adds protection for
/// script/lyric text so a generic rewrite cannot silently change words intended to be spoken or sung.
#[cfg(any(test, native_lane))]
fn base_rules(medium: &str) -> String {
    let mut rules = vec![
        format!("You are a prompt rewriter for a generative {medium} model."),
        format!(
            "Rewrite the user's input into a single, precise {medium} prompt that follows the \
             model's prompt guide below."
        ),
        String::new(),
        "Rules:".to_owned(),
        "- Output exactly one rewritten prompt and nothing else — no explanations, reasoning, \
         commentary, options, or labels."
            .to_owned(),
        format!(
            "- Preserve the user's intent: do not change the subjects, attributes, actions, \
             relationships, or core setting they described. You may add concrete details only when \
             they make the {medium} more coherent and stay consistent with the user's meaning."
        ),
        "- If the user's prompt is already detailed and on-guide, make only minimal edits for \
         fluency."
            .to_owned(),
        "- Follow the guide's recommended structure, phrasing, and what-to-avoid guidance."
            .to_owned(),
        "- Match the user's language: if their prompt is not in English, respond in the same \
         language."
            .to_owned(),
        "- Do not wrap the output in quotes, markdown, JSON, or code fences unless those are part \
         of the described scene."
            .to_owned(),
    ];
    if medium.eq_ignore_ascii_case("image") {
        // sc-13881 (epic 13879): a refined PHOTOGRAPHIC prompt must keep concrete camera/lens/
        // film-stock specifications. Trim ONLY empty praise; preserve real visual specifications.
        rules.extend([
            "- Keep concrete photographic specifications the user wrote — camera body, lens and focal \
             length, film stock, and aperture (e.g. \"85mm f/1.4\", \"Kodak Portra 400\", \"shot on a \
             50mm lens\"). These carry real visual meaning; do not drop them when trimming."
                .to_owned(),
            "- When trimming for concision, remove ONLY empty praise that adds no visual information \
             (\"masterpiece\", \"stunning\", \"premium\", \"gorgeous\", \"award-winning\", \
             \"ultra-detailed\"), never the concrete camera, lens, film-stock, or lighting details."
                .to_owned(),
        ]);
    } else if medium.eq_ignore_ascii_case("audio") {
        rules.extend([
            "- Treat the selected model guide as authoritative about whether the input is an audio \
             description, a script, or lyrics."
                .to_owned(),
            "- If the input is a script or lyrics intended to be spoken or sung, preserve its words, \
             meaning, section labels, and speaker labels. Make only minimal punctuation or formatting \
             edits unless the guide explicitly asks for another transformation."
                .to_owned(),
            "- If the input describes music or sound, prefer concrete audible attributes such as \
             instrumentation or sound source, performance, rhythm or pacing, space, dynamics, and \
             production character. Do not add visual or camera terminology."
                .to_owned(),
        ]);
    }
    rules.join("\n")
}

/// Build the `system` message for the refiner: the rewrite rules (medium chosen from the workflow)
/// plus the model's prompt guide when one is supplied. Port of the Python `build_system_prompt`.
#[cfg(any(test, native_lane))]
fn build_refine_system_prompt(guide: Option<&str>, workflow: Option<&str>) -> String {
    let medium = match workflow.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("video") => "video",
        Some(value) if value.eq_ignore_ascii_case("audio") => "audio",
        _ => "image",
    };
    let rules = base_rules(medium);
    let guide = guide.unwrap_or("").trim();
    if guide.is_empty() {
        rules
    } else {
        format!("{rules}\n\n# Model prompt guide\n\n{guide}")
    }
}

// ----------------------------------------------------------------------------------------------
// MiniMax-H3 tailored rewrite (epic 17137, sc-17162) — the local stand-in for the WITHHELD hosted
// `H3-Context-IR` prompt-understanding component. Same shape as the Ideogram magic-prompt branch
// below: an embedded, model-keyed system prompt driving the SAME `prompt_refine` TextLlm, selected
// instead of the generic rewrite rules. It differs from magic-prompt in what selects it — Ideogram's
// asset is chosen by the payload's `task` discriminator (a caption is a different JOB), while this
// one is still the free-text Rewrite task and is chosen by the payload's `modelId` FAMILY, because
// the product difference is which model consumes the result. Opt-in end to end: nothing rewrites a
// prompt unless the user presses Refine and then Apply (`RefinePromptControl`).
// ----------------------------------------------------------------------------------------------

/// The tailored MiniMax-H3 rewrite system prompt (sc-17162), embedded verbatim and versioned like
/// the Ideogram assets, and parsed for its `[SYSTEM]` block by the same [`magic_section_from`]. Its
/// output shape is `H3-Context-IR`'s own published one — `integrated_multimodal_description:` /
/// `overall_soundscape:` / `non_diegetic_music:` plus `[Shot N] At MM:SS.mmm` timed-shot labels —
/// so the local refiner emits what the withheld hosted component would have. There is no `[USER]`
/// block: like the generic rewrite, the user turn is the user's own prompt.
#[cfg(any(test, native_lane))]
const MINIMAX_H3_REFINE_V1: &str = include_str!("minimax_h3_refine_v1.txt");

/// The seven markers MiniMax declares in H3's `tokenizer_config.json` `additional_special_tokens`
/// and NEVER TRAINED (sc-17143, story activity 18692): they appear in no vocabulary artifact, and
/// the embedding rows `transformers` assigns them are statistically indistinguishable from the
/// checkpoint's unused padding tail. The model card's examples use `<d>[English] …</d>` anyway, so a
/// refiner that has just read the prompt guide can easily reproduce them. Emitting one feeds an
/// untrained vector to the text encoder and spends prompt space for nothing, so the H3 rewrite is
/// both INSTRUCTED not to write them (the embedded asset) and post-filtered so it cannot
/// (`strip_untrained_markers`) — the instruction is advisory, the filter is the guarantee.
#[cfg(any(test, native_lane))]
pub(crate) const MINIMAX_H3_UNTRAINED_MARKERS: [&str; 7] = [
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
];

/// Whether this refine job's TARGET model is a MiniMax-H3 partition, and so takes the tailored
/// rewrite + the marker post-filter. Keyed through `sceneworks_core`'s own family predicate rather
/// than a second copy of the id list, so `minimax_h3` and `minimax_h3_ref` — and any later partition
/// on the same prefix — resolve identically here and in the video request path.
///
/// Deliberately keyed on the model ALONE, not on `(model, task)`: the ban is a property of what the
/// text encoder can represent, so a task added later cannot route around it. Every other model is
/// untouched.
#[cfg(any(test, native_lane))]
fn refines_for_minimax_h3(model_id: Option<&str>) -> bool {
    model_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some_and(sceneworks_core::video_request::is_minimax_h3_model)
}

/// Build the `system` message for a MiniMax-H3 rewrite: the embedded asset's `[SYSTEM]` block, plus
/// the model's own prompt guide when the caller supplied one (the web fetches
/// `/prompt-guides/minimax-h3.md` and forwards it, exactly as it does for the generic path). The
/// guide is appended under the SAME `# Model prompt guide` heading [`build_refine_system_prompt`]
/// uses, so the two paths present a guide identically.
#[cfg(any(test, native_lane))]
fn build_minimax_h3_refine_system_prompt(guide: Option<&str>) -> String {
    let rules = magic_section_from(MINIMAX_H3_REFINE_V1, "SYSTEM");
    let guide = guide.unwrap_or("").trim();
    if guide.is_empty() {
        rules
    } else {
        format!("{rules}\n\n# Model prompt guide\n\n{guide}")
    }
}

// ----------------------------------------------------------------------------------------------
// Film plan (epic 22708, sc-22713) — the local filmmaking harness's planner. Same shape as the
// Ideogram magic-prompt branch: an embedded, versioned system prompt driving the SAME `prompt_refine`
// TextLlm, selected by the payload's `task` discriminator because a shot plan is a different JOB from
// a prompt rewrite. Everything film-specific (the brief, its required beats, the approved reference
// roles and the installed model's declared menus) is composed caller-side and arrives as the user
// turn, so this asset carries only the contract every planning request shares.
// ----------------------------------------------------------------------------------------------

/// The film-plan system prompt, embedded verbatim and parsed for its `[SYSTEM]` block by the same
/// [`magic_section_from`] the other assets use. There is no `[USER]` block: the user turn is the
/// request the harness composed.
#[cfg(any(test, native_lane))]
const FILM_PLAN_V1: &str = include_str!("film_plan_v1.txt");

/// The `(system, user)` pair for a film-plan request: the embedded contract, plus the caller's
/// composed brief as the user turn.
#[cfg(any(test, native_lane))]
pub(crate) fn build_film_plan_messages(request: &str) -> (String, String) {
    (
        magic_section_from(FILM_PLAN_V1, "SYSTEM"),
        request.to_owned(),
    )
}

/// Select the rewrite `system` message: the tailored MiniMax-H3 prompt for an H3 target model, the
/// generic medium-keyed rewrite rules for everything else. The generic
/// [`build_refine_system_prompt`] is untouched, so no other model's refinement changes.
#[cfg(any(test, native_lane))]
fn build_rewrite_system_prompt(
    guide: Option<&str>,
    workflow: Option<&str>,
    model_id: Option<&str>,
) -> String {
    if refines_for_minimax_h3(model_id) {
        build_minimax_h3_refine_system_prompt(guide)
    } else {
        build_refine_system_prompt(guide, workflow)
    }
}

/// Remove every [`MINIMAX_H3_UNTRAINED_MARKERS`] occurrence from a refined prompt (case-insensitive
/// — the markers are ASCII, matched without lowercasing the surrounding Unicode text). STRIP rather
/// than reject, which is what this pipeline already does to invalid output: `clean_refine_output`
/// removes reasoning blocks, code fences and stray quotes rather than failing the job over them. A
/// reply that was ONLY markers cleans to empty and then hits the existing empty-output error, so the
/// reject path is still reachable where it should be.
///
/// Each marker becomes a SPACE, not nothing, so `word<d>word` cannot be glued into one token; the
/// runs of spaces that produces are collapsed per line. Text with no marker is returned byte for
/// byte — the collapse never runs on an untouched reply.
#[cfg(any(test, native_lane))]
fn strip_untrained_markers(text: &str) -> String {
    let mut stripped = text.to_owned();
    let mut found = false;
    for marker in MINIMAX_H3_UNTRAINED_MARKERS {
        while let Some(at) = first_ci(&stripped, marker) {
            stripped.replace_range(at..at + marker.len(), " ");
            found = true;
        }
    }
    if !found {
        return stripped;
    }
    stripped
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

/// The whole reply-side cleanup, in one place so it is drivable without an API or real weights: the
/// per-task clean (JSON isolation for a caption task, prose cleanup otherwise), then sc-17162's
/// MiniMax-H3 marker ban when the TARGET model is an H3 partition.
///
/// The ban runs after BOTH clean paths and is keyed on the model alone, so no task can route around
/// it, and it runs BEFORE the caller's empty-output check — a reply that was nothing but markers
/// fails as empty (the pipeline's existing rejection) rather than shipping whitespace. Enforcing it
/// here rather than only asking for it in the system prompt is the point: the refiner has just been
/// handed a prompt guide that quotes `<d>[English] …</d>` verbatim, so an instruction is a request
/// and this is the guarantee.
#[cfg(any(test, native_lane))]
fn finalize_refined_output(raw: &str, emits_json: bool, model_id: Option<&str>) -> String {
    let cleaned = if emits_json {
        clean_json_output(raw)
    } else {
        clean_refine_output(raw)
    };
    if refines_for_minimax_h3(model_id) {
        strip_untrained_markers(&cleaned)
    } else {
        cleaned
    }
}

/// Strip `<think>…</think>` reasoning blocks, a wrapping code fence, and matching surrounding quotes
/// from the model reply. Port of the Python `clean_output` (regex-free: the tags are ASCII, matched
/// case-insensitively without lowercasing the whole — Unicode-safe — string).
#[cfg(any(test, native_lane))]
fn clean_refine_output(text: &str) -> String {
    let mut text = strip_reasoning_and_fence(text);
    // Matching surrounding single/double quotes.
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 2 {
        let (first, last) = (chars[0], chars[chars.len() - 1]);
        if first == last && (first == '"' || first == '\'') {
            text = chars[1..chars.len() - 1]
                .iter()
                .collect::<String>()
                .trim()
                .to_owned();
        }
    }
    text
}

// ----------------------------------------------------------------------------------------------
// Magic-prompt expansion (epic 4725, sc-5997) — plain idea → structured JSON caption. Drives the
// SAME `prompt_refine` TextLlm (Anubis-Mini-8B) with Ideogram's open-source magic-prompt system prompt
// (`task: "magic_prompt"`) instead of the rewrite rules. The hosted Ideogram / OpenRouter Sonnet/Opus
// configs in the reference are replaced by the local model (native-first, offline). The caller (web)
// strips the non-schema `aspect_ratio` key + bboxes and validates the result against the sc-5993
// caption contract, so this side stays generic: build messages, run, extract the JSON object.
// ----------------------------------------------------------------------------------------------

/// Ideogram 4's magic-prompt system-prompt file, embedded verbatim. Source (Apache-2.0):
/// github.com/ideogram-oss/ideogram4 `src/ideogram4/magic_prompt_system_prompts/v1.txt`. Parsed for
/// its `[SYSTEM]` + `[USER]` sections at runtime (the `[META]` block is ignored).
#[cfg(any(test, native_lane))]
const MAGIC_PROMPT_V1: &str = include_str!("ideogram_magic_prompt_v1.txt");

/// Ideogram 4's image-grounded caption system prompt (epic 8102, sc-8105), embedded verbatim and
/// versioned like [`MAGIC_PROMPT_V1`]. Drives the SAME `prompt_refine` TextLlm seam — but through the
/// `core_llm` vision path: the model EXAMINES a reference image and emits a schema-valid Ideogram JSON
/// caption (style + grounded composition, bboxes kept) instead of expanding a text idea. Parsed for its
/// `[SYSTEM]` + `[USER]` sections at runtime by the same [`magic_section_from`] parser (the `[META]`
/// block is ignored).
#[cfg(any(test, native_lane))]
const IMAGE_CAPTION_V1: &str = include_str!("ideogram_image_caption_v1.txt");

/// Plain-text PROSE image-description system prompt (epic 8203, sc-8204), embedded verbatim and
/// versioned like [`IMAGE_CAPTION_V1`]. Drives the SAME `prompt_refine` `core_llm` vision path as
/// `image_caption`, but for NON-structured text-to-image models: the model EXAMINES a reference image
/// and emits a detailed natural-language description usable directly as a t2i prompt — NO JSON, NO
/// bboxes, NO output constraint. Parsed for its `[SYSTEM]` + `[USER]` sections by [`magic_section_from`].
#[cfg(any(test, native_lane))]
const IMAGE_DESCRIBE_V1: &str = include_str!("image_describe_v1.txt");

/// TAG-STYLE image-description system prompt (epic 8203, sc-8205), the booru/danbooru variant of
/// [`IMAGE_DESCRIBE_V1`]. Selected when the job payload carries `captionStyle:"tags"` — for anime/booru
/// SDXL checkpoints that consume comma-separated tags rather than prose. Same vision path, same prose
/// cleanup (a flat tag string, no JSON).
#[cfg(any(test, native_lane))]
const IMAGE_DESCRIBE_TAGS_V1: &str = include_str!("image_describe_tags_v1.txt");

/// Multi-image "mood board" PROSE synthesis system prompt (epic 8588, sc-8595) — the N>1 sibling of
/// [`IMAGE_DESCRIBE_V1`]. When a describe job carries MORE THAN ONE reference image, the model examines
/// them together and synthesizes ONE prose prompt capturing the mood/style/composition they SHARE
/// (rather than describing a single picture). Same `core_llm` vision path, same unconstrained-prose
/// cleanup; only the instruction differs. Selected by [`build_image_describe_messages`] when `multi`.
#[cfg(any(test, native_lane))]
const IMAGE_DESCRIBE_MOODBOARD_V1: &str = include_str!("image_describe_moodboard_v1.txt");

/// Multi-image "mood board" TAG-STYLE synthesis system prompt (epic 8588, sc-8595) — the N>1, booru
/// sibling of [`IMAGE_DESCRIBE_TAGS_V1`]. Synthesizes the SHARED style across several references into
/// one comma-separated tag list for an anime/booru SDXL checkpoint. Selected by
/// [`build_image_describe_messages`] when `multi` and the style is [`DescribeStyle::Tags`].
#[cfg(any(test, native_lane))]
const IMAGE_DESCRIBE_MOODBOARD_TAGS_V1: &str = include_str!("image_describe_moodboard_tags_v1.txt");

/// Multi-image "mood board" Ideogram JSON synthesis system prompt (epic 8588, sc-8595) — the N>1
/// sibling of [`IMAGE_CAPTION_V1`]. Grounds `style_description` in the look several references SHARE and
/// composes a coherent NEW scene in that style (the composition is synthesized, like `magic_prompt` from
/// a text idea — so the caption stays schema-valid without per-image grounded bboxes). Same JSON
/// constraint + `is_caption` validation. Selected by [`build_image_caption_messages`] when `multi`.
#[cfg(any(test, native_lane))]
const IMAGE_CAPTION_MOODBOARD_V1: &str = include_str!("ideogram_image_caption_moodboard_v1.txt");

/// Body of a `[NAME]` section in the magic-prompt file (port of the reference `_load_sections`):
/// section markers are a bracketed single word alone on a line. Returns the trimmed body, or empty.
#[cfg(any(test, native_lane))]
fn magic_section(name: &str) -> String {
    magic_section_from(MAGIC_PROMPT_V1, name)
}

/// Body of a `[NAME]` section in ANY bracketed-marker prompt file (the magic-prompt or the
/// image-caption asset). Section markers are a bracketed single word alone on a line. Returns the
/// trimmed body, or empty. Generalized from `magic_section` so the sc-8105 image-caption asset reuses
/// the same parser.
#[cfg(any(test, native_lane))]
pub(crate) fn magic_section_from(source: &str, name: &str) -> String {
    let mut capturing = false;
    let mut body: Vec<&str> = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let is_marker = trimmed.len() >= 2
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && !trimmed.contains(' ');
        if is_marker {
            if capturing {
                break;
            }
            capturing = trimmed[1..trimmed.len() - 1].eq_ignore_ascii_case(name);
            continue;
        }
        if capturing {
            body.push(line);
        }
    }
    body.join("\n").trim().to_owned()
}

/// The `(system, user)` chat messages for a magic-prompt expansion: the `[SYSTEM]` block, and the
/// `[USER]` template with `{{aspect_ratio}}` / `{{original_prompt}}` substituted (port of the
/// reference `build_messages`). `aspect_ratio` is `"W:H"`.
#[cfg(any(test, native_lane))]
pub(crate) fn build_magic_prompt_messages(prompt: &str, aspect_ratio: &str) -> (String, String) {
    let system = magic_section("SYSTEM");
    let mut user = magic_section("USER");
    if user.is_empty() {
        user = "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height).".to_owned();
    }
    user = user.replace("{{aspect_ratio}}", aspect_ratio);
    user = if user.contains("{{original_prompt}}") {
        user.replace("{{original_prompt}}", prompt)
    } else {
        format!("{user}\n\n{prompt}")
    };
    (system, user)
}

// ----------------------------------------------------------------------------------------------
// Image-grounded caption (epic 8102, sc-8105) — reference image → structured Ideogram JSON caption.
// Drives the SAME `prompt_refine` TextLlm seam (`task: "image_caption"`) through the `core_llm` VISION
// path: the user turn carries a `Content::Image(ImageRef)` alongside the `[USER]` instruction, and the
// `[SYSTEM]` block is the embedded `ideogram_image_caption_v1.txt` (observe-don't-populate, style +
// grounded composition, bboxes kept). The image drives the multimodal path at GENERATE time
// (`mlx-llama`'s loaded Qwen-VL vision tower reads the `Content::Image`). At RESOLUTION time the worker
// must NOT demand vision: core-llm's `select`/`meets` filters on each provider's STATIC descriptor, and
// no linked provider statically advertises BOTH vision and Json for a Qwen-VL snapshot (`mlx-llama` is
// text+Json until it loads; `mlx-joycaption` is vision-but-no-constraints and only serves LLaVA). So the
// worker resolves on the JSON constraint ALONE — that selects `mlx-llama`, which loads the Qwen-VL
// snapshot and flips to vision at load. The reply is cleaned with the SAME `clean_json_output` +
// validated with `sceneworks_core::ideogram_caption::is_caption`; malformed output is handled like
// magic-prompt (the empty/non-caption result errors caller-side).
// ----------------------------------------------------------------------------------------------

/// The `(system, user)` chat text for an image-caption run: the `[SYSTEM]` + `[USER]` blocks of the
/// embedded `ideogram_image_caption_v1.txt`. The reference image is attached to the user turn as a
/// separate `Content::Image` block by the caller (this only supplies the text).
#[cfg(any(test, native_lane))]
pub(crate) fn build_image_caption_messages(multi: bool) -> (String, String) {
    // With more than one reference image the model synthesizes ONE caption for a new image in the
    // aesthetic the board SHARES (mood board, sc-8595); the single-image asset is unchanged for N==1.
    let asset = if multi {
        IMAGE_CAPTION_MOODBOARD_V1
    } else {
        IMAGE_CAPTION_V1
    };
    let system = magic_section_from(asset, "SYSTEM");
    let mut user = magic_section_from(asset, "USER");
    if user.is_empty() {
        user = if multi {
            "Examine the reference images attached to this message as a single mood board and emit \
             the single JSON caption object as specified, grounding the style in the look they share \
             and composing a coherent new scene in that style."
        } else {
            "Examine the reference image attached to this message and emit the single JSON \
             caption object as specified. Describe only what is visible in the image."
        }
        .to_owned();
    }
    (system, user)
}

/// The plain-text describe style selected by the job payload's `captionStyle` field (epic 8203):
/// `Prose` (default — natural-language sentences, sc-8204) or `Tags` (booru/danbooru comma-tags for
/// anime/booru SDXL checkpoints, sc-8205). Any unknown/absent value falls back to `Prose` (defensive).
#[cfg(any(test, native_lane))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DescribeStyle {
    Prose,
    Tags,
}

#[cfg(any(test, native_lane))]
impl DescribeStyle {
    /// Parse the `captionStyle` payload value; only `"tags"` (case-insensitive) selects tags, everything
    /// else — including absent/empty/unknown — is prose.
    pub(crate) fn from_payload(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some(v) if v.eq_ignore_ascii_case("tags") => DescribeStyle::Tags,
            _ => DescribeStyle::Prose,
        }
    }
}

/// The `(system, user)` chat text for an image-DESCRIBE run (epic 8203): the `[SYSTEM]` + `[USER]`
/// blocks of the embedded describe asset for the chosen [`DescribeStyle`] — [`IMAGE_DESCRIBE_V1`]
/// (prose, sc-8204) or [`IMAGE_DESCRIBE_TAGS_V1`] (tags, sc-8205). Like [`build_image_caption_messages`]
/// the reference image is attached to the user turn as a separate `Content::Image` block by the caller;
/// this only supplies the text. Unlike the caption path the model emits plain text (no JSON constraint).
#[cfg(any(test, native_lane))]
pub(crate) fn build_image_describe_messages(style: DescribeStyle, multi: bool) -> (String, String) {
    // With more than one reference image the model synthesizes ONE prompt from the mood/style the board
    // SHARES (mood board, sc-8595) instead of describing a single picture; N==1 keeps the sc-8204/8205
    // single-image assets byte-for-byte.
    let asset = match (style, multi) {
        (DescribeStyle::Prose, false) => IMAGE_DESCRIBE_V1,
        (DescribeStyle::Prose, true) => IMAGE_DESCRIBE_MOODBOARD_V1,
        (DescribeStyle::Tags, false) => IMAGE_DESCRIBE_TAGS_V1,
        (DescribeStyle::Tags, true) => IMAGE_DESCRIBE_MOODBOARD_TAGS_V1,
    };
    let system = magic_section_from(asset, "SYSTEM");
    let mut user = magic_section_from(asset, "USER");
    if user.is_empty() {
        user = match (style, multi) {
            (DescribeStyle::Prose, false) => {
                "Examine the reference image attached to this message and write the single detailed \
                 plain-text description as specified. Describe only what is visible in the image."
            }
            (DescribeStyle::Prose, true) => {
                "Examine the reference images attached to this message as a single mood board and \
                 write ONE detailed plain-text prompt as specified, capturing the mood, style, \
                 palette, lighting, and composition they share."
            }
            (DescribeStyle::Tags, false) => {
                "Examine the reference image attached to this message and emit the single \
                 comma-separated booru-style tag list as specified. Tag only what is visible in the \
                 image."
            }
            (DescribeStyle::Tags, true) => {
                "Examine the reference images attached to this message as a single mood board and \
                 emit ONE comma-separated booru-style tag list as specified, capturing the style, \
                 mood, palette, and composition they share."
            }
        }
        .to_owned();
    }
    (system, user)
}

/// Pixel-count ceiling for a vision reference image before it reaches the Qwen-VL tower. The ViT runs
/// DENSE `n×n` self-attention over its `n = pixels / patch_size²` patches, so the attention-score
/// tensor grows as `num_heads · n²`: at the provider's smart-resize ceiling (a 16.7-MP / 4096² pixel
/// budget) a large reference image expands to ~65 536 patches and that single tensor is ~272 GB —
/// far past the Metal max-buffer limit, which is exactly the allocation that crashed `image_describe`.
/// 1 MP keeps `n ≈ 4096` (≈1 GB of attention scores) while staying well above the provider's 256²
/// `min_pixels` floor (so it is not upscaled back), and is ample resolution for a VLM to caption or
/// describe a scene (it matches the de-facto Qwen2-VL default budget). We shrink here, aspect-
/// preserving, rather than lean on the provider's ceiling because that ceiling is both too high for
/// dense attention AND hardcoded from `Qwen35ImageProcessor::default()` (provider.rs ignores the
/// model's `preprocessor_config.json`) — fixing that upstream is tracked separately.
const VISION_REFERENCE_MAX_PIXELS: u64 = 1_048_576; // 1024² — safe for the dense ViT attention

/// Shrink `image` (aspect-preserving) so its pixel count does not exceed `max_pixels`; an image already
/// within budget is returned untouched. The target box is the source dimensions scaled by
/// `sqrt(max_pixels / pixels)`, so the result is the largest same-ratio image inside the budget.
fn downscale_to_pixel_budget(image: image::DynamicImage, max_pixels: u64) -> image::DynamicImage {
    let (width, height) = (image.width(), image.height());
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= max_pixels || width == 0 || height == 0 {
        return image;
    }
    let scale = (max_pixels as f64 / pixels as f64).sqrt();
    let target_w = ((width as f64) * scale).floor().max(1.0) as u32;
    let target_h = ((height as f64) * scale).floor().max(1.0) as u32;
    image.resize(target_w, target_h, image::imageops::FilterType::Lanczos3)
}

/// Decode a reference image off disk into a `core_llm::ImageRef` (RGB8), mirroring the JoyCaption
/// `load_caption_image` pattern (`decode_image_any` → `to_rgb8`) but producing the vision contract's
/// tensor-free image type instead of the captioner's `gen_core::Image`. Used by the `image_caption`
/// and `image_describe` tasks to build the `Content::Image` block. The decoded image is capped to
/// [`VISION_REFERENCE_MAX_PIXELS`] BEFORE conversion so a large reference image cannot OOM the dense
/// ViT attention in the vision tower.
pub(crate) fn load_caption_image_ref(path: &Path) -> WorkerResult<gen_core::core_llm::ImageRef> {
    let decoded = crate::image_decode::decode_image_any(path).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "image-caption reference image {}: {error}",
            path.display()
        ))
    })?;
    vision_image_ref(decoded).map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "image-caption reference image {}: {error}",
            path.display()
        ))
    })
}

/// An already-decoded image as the vision contract's `ImageRef` (RGB8), capped to
/// [`VISION_REFERENCE_MAX_PIXELS`] first so it cannot OOM the dense ViT attention. Shared by
/// [`load_caption_image_ref`] and the dataset batch describer (SW-7082), which decodes the image
/// itself because it also reports the image's source pixel size.
pub(crate) fn vision_image_ref(
    decoded: image::DynamicImage,
) -> Result<gen_core::core_llm::ImageRef, String> {
    let decoded = downscale_to_pixel_budget(decoded, VISION_REFERENCE_MAX_PIXELS).to_rgb8();
    let (width, height) = (decoded.width(), decoded.height());
    gen_core::core_llm::ImageRef::new(width, height, decoded.into_raw())
        .map_err(|error| error.to_string())
}

// ----------------------------------------------------------------------------------------------
// Job handler — native MLX on macOS (sc-5552 / sc-7158) and candle on the Windows candle build
// (sc-5525 / sc-7404). The body is backend-agnostic: `core_llm::load_for_model` resolves whichever
// platform catalog supplies mlx-llama on macOS or candle-llama on the candle build model-first.
// A build with neither native provider never advertises the capability, so the job never arrives.
// ----------------------------------------------------------------------------------------------

#[cfg(native_lane)]
pub(crate) async fn run_prompt_refine_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    // The cooperative cancel handle is created here and threaded into the request built inside the
    // blocking task. Both lanes now run through the unified `core_llm` contract, so this is core-llm's
    // CancelFlag (via the gen_core re-export) on both — new()/clone()/cancel()/is_cancelled() — and the
    // dispatch body below is a single backend-agnostic path.
    use gen_core::core_llm::CancelFlag;

    let RefineRequest {
        task,
        original_prompt,
        image_refs,
        image_reference_count,
        qwen_rewriter,
        model,
        target_model_id,
        thinking_mode,
        thinking_mode_name,
        max_new_tokens,
        kv_compression,
        temperature,
        work_message,
        done_message,
        weights_dir,
        system,
        user_message,
    } = parse_refine_request(settings, job).await?;
    // Attribute the run to the active backend (MLX on macOS, candle off-Mac) on the streamed progress
    // + UI architecture pill (mirrors the image/video paths), not the gpu-id device label.
    let backend = REFINE_BACKEND;

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        refine_progress(
            JobStatus::LoadingModel,
            ProgressStage::LoadingModel,
            0.1,
            crate::load_progress::PREPARING_MODEL,
            None,
            backend,
        ),
    )
    .await?;
    check_cancel(api, &job.id, CANCEL_MESSAGE).await?;

    let cancel = CancelFlag::new();
    // Coalesced per-token progress (sc-8840, F-038). A native caption decodes up to 4096 tokens; the
    // old path sent EVERY token on a bounded(64) channel and fired a full `update_job` POST per token
    // — thousands of sequential POSTs, with `blocking_send` back-pressuring generation on API latency
    // (worse over epic-4484 LAN). Instead the token callback writes the latest `(current, total)` into
    // a **watch** channel (latest-wins, non-blocking — `send` never blocks the sender and never drops
    // the LATEST value), and the loop below posts only the newest snapshot on a fixed tick. Generation
    // is fully decoupled from API latency, intermediate ticks are coalesced away, and the FINAL token
    // count is always the watch's resident value so the terminal progress is never lost.
    let (progress_tx, progress_rx) = tokio::sync::watch::channel::<(u32, u32)>((0, max_new_tokens));
    let blocking_cancel = cancel.clone();
    let job_id = job.id.clone();
    let prompt = user_message;
    let engine_label = model.clone();
    // Run the load+generate on the shared refine-model cache thread (sc-8840, F-038): a resident model
    // keyed by weights dir is reused across interactive refine clicks instead of cold-loading the
    // ~16 GB snapshot every time, and is idle-evicted so memory stays bounded (mirrors the image/video
    // `generator_cache`). `with_cached_refiner` runs the closure ON that thread, so the `!Send`
    // provider never crosses a thread boundary; only the `String` result comes back. Wrapping it in a
    // The spawned task is joined by the shared stream supervisor, including on cancel/API failure.
    let refine_spec = gen_core::core_llm::LoadSpec {
        source: weights_dir.to_string_lossy().into_owned(),
        // See `catalog_semantic_jobs`: an explicit projector artifact is only for a separable
        // Prism GGUF load. The refiners (including the vision tasks) load snapshot directories
        // that carry their own projector, so the default `projector_source: None` preserves the
        // existing behaviour; every other load option keeps its default as well.
        ..Default::default()
    };
    // Whether the decode is constrained to valid JSON (the caption tasks and the film plan, sc-22713)
    // and whether a reference image rides the user turn (the vision tasks) — copied out of the `Copy`
    // `RefineTask` into plain bools so the blocking closure below names no enum, keeping its capture
    // set minimal.
    let emits_json = task.emits_json();
    // sc-24113: the blocking closure's image invariant is "this task DECLARES it takes reference
    // images", not "this task is a vision task" — the Qwen rewrite carries images while
    // `is_vision()` is false. Copied out as a bool for the same reason `emits_json` is: so the
    // closure's capture set stays minimal and names no enum.
    let takes_reference_images = task.takes_reference_images();
    // Resolution requirements (see the sc-8105 note below): JSON-emitting tasks still select a
    // provider that supports JSON constraints, preserving the existing provider/cache identity.
    // After load, [`json_decode_constraint`] decides whether to apply the mask to this decode: an
    // active thinking channel must first be allowed to close before the final JSON answer begins.
    let mut refine_reqs = gen_core::core_llm::ModelRequirements::default();
    if emits_json {
        refine_reqs = refine_reqs.with_constraint(gen_core::core_llm::Constraint::Json);
    }
    let blocking = crate::load_progress::spawn(crate::refine_model_cache::with_cached_refiner(
        refine_spec,
        refine_reqs,
        "prompt-refine load failed",
        move |refiner| -> WorkerResult<(gen_core::core_llm::TextLlmOutput, Value)> {
            emit_event(
                "prompt_refine_load_start",
                json!({ "jobId": job_id, "engine": engine_label }),
            );

            // Resolve the native provider model-first (no provider id) and stream through the
            // `core_llm::TextLlm` contract. One backend-agnostic path: the bundled provider
            // (mlx-llama on macOS, candle-llama on the Windows candle build) wins resolution. The provider
            // renders the model's own chat template, so the worker supplies only the system + user turns
            // (the product policy stays caller-side).
            //
            // sc-8105 resolution note: `core-llm`'s `select`/`meets` filters on each provider's STATIC
            // (weightless) descriptor BEFORE any `load` runs. `mlx-llama` statically advertises
            // `supports_vision:false` + `[Constraint::Json]` — it loads a Qwen-VL (`qwen3_5`) snapshot and
            // flips `supports_vision` on only at LOAD time (mlx-llm provider.rs:267). `mlx-joycaption`
            // statically advertises vision but NO constraints and only `can_load`s LLaVA (not Qwen-VL).
            // So NO provider statically satisfies BOTH vision AND Json for a Qwen-VL snapshot: demanding
            // `vision:true` at resolution (what `ModelRequirements::from_request` derives from the image
            // block) would make `select` return `Error::Unsupported` and never reach `load`. The image is
            // what drives the multimodal generate path at GENERATE time (LlamaProvider::generate gates on
            // its loaded `vision` tower), NOT what must be matched at resolution. So the image_caption path
            // resolves on the JSON constraint ALONE (no vision filter): that selects `mlx-llama`, which then
            // loads the Qwen-VL snapshot, flips to vision, and examines the `Content::Image`. (The other
            // tasks carry no image, so their `from_request` reqs never set the vision filter anyway.)
            let output = {
                use gen_core::core_llm::{Content, Message, Role, Sampling, StreamEvent};
                let mut messages = Vec::with_capacity(2);
                if !system.trim().is_empty() {
                    messages.push(Message::system(system));
                }
                // A vision task (image_caption / image_describe) user turn carries the reference image(s)
                // as `Content::Image` block(s) BEFORE the instruction text, so the loaded provider examines
                // the picture(s) at generate time. The provider collects every `Content::Image` in the turn
                // and expands per-image vision tokens (mlx-llama + candle-llama both loop over all images),
                // so a mood board of N references (sc-8595) rides one generate call. Every other task is a
                // plain text user turn.
                let carries_image = !image_refs.is_empty();
                if carries_image {
                    let mut content: Vec<Content> =
                        image_refs.into_iter().map(Content::Image).collect();
                    content.push(Content::text(prompt));
                    messages.push(Message {
                        role: Role::User,
                        content,
                        thinking: None,
                        tool_calls: Vec::new(),
                    });
                } else {
                    messages.push(Message::user(prompt));
                }
                // The bespoke prompt-refine samplers were plain temperature/top-p (no repetition
                // penalty / top-k); core-llm's defaults match (top_k 0, repetition_penalty 1.0).
                //
                // sc-24113: the Qwen rewriters are the one task that does NOT take the house
                // sampler. Their model cards publish the full recipe the checkpoints were tuned
                // at (temperature 1.0, top_p 0.95, top_k 20), and it is frozen verbatim in
                // `qwen_prompt_rewrite` beside the template it belongs to.
                let sampling = match qwen_rewriter {
                    Some(_) => Sampling {
                        temperature,
                        top_p: crate::qwen_prompt_rewrite::REWRITE_TOP_P,
                        top_k: crate::qwen_prompt_rewrite::REWRITE_TOP_K,
                        ..Sampling::default()
                    },
                    None => Sampling {
                        temperature,
                        top_p: 0.9,
                        ..Sampling::default()
                    },
                };
                // sc-6585 / sc-8105: a caption task (magic-prompt OR image-caption) must emit a
                // structurally-valid JSON caption, so constrain its decode to the JSON grammar; the
                // free-text rewrite is unconstrained. (On the candle lane this constraint actually
                // steers + masks the decode — the sc-7404 parity gain over `candle-gen-prompt-refine`.)
                // sc-22713 joins it: a shot plan is parsed field by field, so an unparseable reply
                // costs a whole repair round this prevents outright.
                //
                // `Constraint::Json` guarantees VALIDITY ONLY — that the emitted text parses as
                // JSON — and says nothing about the object's shape (core-llm has no schema or
                // grammar variant; `Json` is the only one). So a well-formed reply with a field
                // of the wrong type is exactly what this cannot prevent, and the sc-22713 smoke
                // duly produced one: `startState` as an object on every shot. The plan schema is
                // enforced after the decode by `film_planner::parse_planner_output`, which is the
                // only guarantee — if a schema-shaped constraint ever lands in core-llm, this is
                // where the FilmPlan task should take it.
                // A reasoning-capable model must be allowed to close its `<think>` block before
                // emitting the answer. Its final answer is still parsed and validated strictly,
                // and the planner's existing bounded repair loop handles malformed plans.
                let constraint = json_decode_constraint(
                    emits_json,
                    refiner.descriptor().capabilities.supports_thinking,
                    thinking_mode,
                );
                let request = refine_text_request(
                    messages,
                    sampling,
                    max_new_tokens,
                    constraint,
                    blocking_cancel.clone(),
                    thinking_mode,
                    kv_compression,
                );
                // The resolution requirements (WITHOUT the auto-vision `from_request` derives from an image
                // block) were built by the caller and folded into the cache key: only the request's output
                // constraint — the JSON grammar for a caption task; NONE for the prose `image_describe` task
                // (sc-8204), which resolves on architecture `can_load` ALONE. A Qwen-VL snapshot has no
                // statically vision+Json provider, so demanding vision here would fail `select` before
                // `load`; that still admits `mlx-llama` (the only provider that `can_load`s a Qwen-VL wrapper
                // — `mlx-joycaption` only loads LLaVA), which loads the snapshot and flips to vision at load.
                // (`carries_image` is asserted so the unused-binding lint stays satisfied and the intent —
                // "an image is present, yet we deliberately do NOT set the vision filter" — is explicit.)
                //
                // sc-24113 widened this from `carries_image == is_vision_task`. That equality was true
                // only while the VISION tasks were the sole image carriers; the Qwen rewrite carries
                // images with `is_vision()` FALSE (it is driven by the user's text and merely accepts
                // pictures), so the old form panicked every I2I rewrite in a debug build. The invariant
                // that actually holds — and the one worth asserting — is the one-directional implication:
                // if images are present, the task must be one that declares it takes them. The converse
                // is deliberately NOT asserted, because a rewrite with zero references is the legitimate
                // text-to-image case.
                debug_assert!(image_carrier_invariant_holds(
                    carries_image,
                    takes_reference_images
                ));
                emit_event(
                    "prompt_refine_load_complete",
                    json!({ "jobId": job_id, "engine": engine_label }),
                );
                if blocking_cancel.is_cancelled() {
                    return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
                }
                // Publish the latest `(current, total)` token count into the coalescing watch channel the
                // loop below reads. `send` is non-blocking and latest-wins — generation is NEVER
                // back-pressured by API latency (the F-038 fix), and the terminal count stays resident so
                // the final progress snapshot is never dropped. A send error means every receiver was
                // dropped (the consumer loop returned early on a POST failure / 409): trip the engine flag
                // so generation bails instead of running unheard (sc-8804, F-003 — the swallowed
                // closed-channel leak, preserved verbatim from the old bounded-channel behavior).
                //
                // The same per-token callback carries the sc-24029 MLX cache bound. MLX's freed-buffer
                // cache is PROCESS-GLOBAL, not per-thread; this callback is used because it is the
                // only hook interleaved with the decode itself — every other seam runs before
                // `generate` starts or after it returns, when the cache has already peaked. Side
                // effect of that global scope: up to once per 16 streamed token events (plus once at
                // decode end) this also discards buffers a CONCURRENT image render had cached, which
                // that render then re-allocates. Declared BEFORE `on_event` so it outlives the borrow
                // and its terminal clear fires after the generate below returns or errors.
                let mut cache_bound = crate::mlx_decode_cache::DecodeCacheBound::mlx();
                let mut on_event = |event: StreamEvent| {
                    if let StreamEvent::Token { index, .. } = event {
                        cache_bound.note_event();
                        if progress_tx
                            .send((index as u32 + 1, max_new_tokens))
                            .is_err()
                        {
                            blocking_cancel.cancel();
                        }
                    }
                };
                generate_refined(refiner, &request, &mut on_event, &job_id)?
            };

            Ok(output)
        },
    ));

    let mut supervisor = StreamSupervisor::new(
        cancel,
        blocking,
        progress_report_interval(settings),
        Some(PROGRESS_POST_INTERVAL),
    );
    let mut last_posted = None;
    let handling: WorkerResult<()> = async {
        loop {
            match supervisor.next::<()>(api, settings, &job.id, None).await? {
                StreamSignal::ProgressTick => {
                    let latest = *progress_rx.borrow();
                    if let Some((current, total)) = next_progress_post(latest, last_posted) {
                        post_refine_progress(api, &job.id, current, total, work_message, backend)
                            .await?;
                        last_posted = Some((current, total));
                    }
                }
                StreamSignal::Finished => break,
                StreamSignal::CancelRequested | StreamSignal::Event(()) => {}
            }
        }
        Ok(())
    }
    .await;
    let (canceled, result) = supervisor.finish(handling).await?;
    if canceled {
        mark_job_canceled(api, &job.id, CANCEL_MESSAGE).await?;
        return Err(WorkerError::Canceled(CANCEL_MESSAGE.to_owned()));
    }
    let (output, kv_cache) =
        result.map_err(|error| task_join_error("prompt refine task join", error))??;
    // Preserve the terminal coalesced snapshot even when generation finishes between ticks.
    let latest = *progress_rx.borrow();
    if let Some((current, total)) = next_progress_post(latest, last_posted) {
        post_refine_progress(api, &job.id, current, total, work_message, backend).await?;
    }
    finalize_refine_job(
        api,
        job,
        RefineCompletion {
            task,
            original_prompt,
            target_model_id,
            model,
            thinking_mode_name,
            max_new_tokens,
            done_message,
            qwen_rewriter,
            image_reference_count,
            emits_json,
            output,
            kv_cache,
        },
    )
    .await
}

/// On platforms with no native prompt-refine provider (neither the macOS MLX twin nor the Windows
/// candle build — e.g. a candle-less Linux worker), the `prompt_refine` capability is never
/// advertised, so this arm is unreachable in practice. Kept so the `run_utility_job` dispatch
/// compiles on all targets.
#[cfg(not(native_lane))]
pub(crate) async fn run_prompt_refine_job(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Err(WorkerError::InvalidPayload(
        "Native prompt refinement needs the macOS MLX worker or the Windows candle backend; it is \
         not supported on this platform."
            .to_owned(),
    ))
}

/// The coalescing decision (sc-8840, F-038): given the LATEST `(current, total)` token count from
/// the watch channel and the value already posted, return `Some(latest)` when it should be posted
/// (it moved) or `None` when it is a redundant repeat (a stalled decode holds the same value between
/// ticks, so we don't re-POST identical progress). Pure so the coalescing invariant — intermediate
/// repeats dropped, every distinct value (including the terminal one) posted exactly once — is unit
/// testable without an API or real weights.
#[cfg(any(test, native_lane))]
fn next_progress_post(latest: (u32, u32), last_posted: Option<(u32, u32)>) -> Option<(u32, u32)> {
    (last_posted != Some(latest)).then_some(latest)
}

/// Post one coalesced running-progress update (sc-8840, F-038). Maps a `(current, total)` token
/// count to the same `0.4 + 0.5 * within` fraction the old per-token loop used, so the only behavior
/// change is HOW OFTEN it fires (coalesced tick vs. per token), not the reported value.
#[cfg(native_lane)]
async fn post_refine_progress(
    api: &ApiClient,
    job_id: &str,
    current: u32,
    total: u32,
    work_message: &str,
    backend: &str,
) -> WorkerResult<()> {
    let within = if total > 0 {
        (f64::from(current) / f64::from(total)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    update_job(
        api,
        job_id,
        refine_progress(
            JobStatus::Running,
            ProgressStage::Running,
            0.4 + 0.5 * within,
            work_message,
            None,
            backend,
        ),
    )
    .await?;
    Ok(())
}

#[cfg(native_lane)]
fn refine_progress(
    status: JobStatus,
    stage: ProgressStage,
    progress: f64,
    message: &str,
    result: Option<JsonObject>,
    backend: &str,
) -> ProgressRequest {
    progress_payload_for_backend(status, stage, progress, message, result, backend)
}

/// The prompt-refine generation request: the task's turn, sampler, budget and decode constraint,
/// the job's cancel flag and reasoning mode, and its compressed-KV opt-in (sc-20682). Every other
/// field keeps the contract default (no seed, no MTP).
#[cfg(any(test, native_lane))]
fn refine_text_request(
    messages: Vec<gen_core::core_llm::Message>,
    sampling: gen_core::core_llm::Sampling,
    max_new_tokens: u32,
    constraint: Option<gen_core::core_llm::Constraint>,
    cancel: gen_core::core_llm::CancelFlag,
    thinking: gen_core::core_llm::ThinkingMode,
    kv_compression: gen_core::core_llm::KvCompressionPolicy,
) -> gen_core::core_llm::TextLlmRequest {
    gen_core::core_llm::TextLlmRequest {
        progress: Some(crate::load_progress::text_observer()),
        messages,
        sampling,
        max_new_tokens,
        seed: None,
        constraint,
        cancel,
        thinking,
        kv_compression,
        ..Default::default()
    }
}

/// Run the refine decode on `refiner` and record the generation's `llm_kv_cache` event whichever
/// way it ends — completed, or refused, canceled or failed by the engine (sc-20682, sc-20688). A
/// completed decode returns its output with the block its result carries: the event's own block.
#[cfg(native_lane)]
fn generate_refined(
    refiner: &dyn gen_core::core_llm::TextLlm,
    request: &gen_core::core_llm::TextLlmRequest,
    on_event: &mut dyn FnMut(gen_core::core_llm::StreamEvent),
    job_id: &str,
) -> WorkerResult<(gen_core::core_llm::TextLlmOutput, Value)> {
    let generation = crate::load_progress::generate_text(refiner, request, on_event);
    let record = crate::llm_kv_cache::KvCacheRecord::of(
        REFINE_BACKEND,
        request.kv_compression,
        &generation,
        |output| output.kv_cache.as_ref(),
    );
    record.emit(job_id);
    let output = generation.map_err(|error| {
        WorkerError::Engine(format!("prompt-refine generation failed: {error}"))
    })?;
    Ok((output, record.block().clone()))
}

/// How the decode ended, beside the budget it ended against (sc-24029).
///
/// Carried on BOTH the success and the failure result. A `Rewrite` that stops on
/// `FinishReason::Length` is non-empty and therefore a "success" as far as this job is concerned,
/// but its text is truncated mid-sentence — and the caller that cannot tolerate that (the film
/// planner's per-shot rewrite, whose output feeds a compiled request) has no other way to know.
/// Purely additive: no existing key changes shape, so every current reader is unaffected.
///
/// sc-20682 adds `kvCache`: the KV cache the generation asked for and ran on
/// ([`crate::llm_kv_cache::kv_cache_block`]) — whether it ran compressed and, if not, why.
#[cfg(any(test, native_lane))]
fn generation_block(
    finish_reason: Option<gen_core::core_llm::FinishReason>,
    usage: gen_core::core_llm::Usage,
    max_new_tokens: u32,
    kv_cache: Value,
) -> Value {
    json!({
        "finishReason": finish_reason_name(finish_reason),
        "usage": {
            "promptTokens": usage.prompt_tokens,
            "generatedTokens": usage.generated_tokens,
        },
        "maxNewTokens": max_new_tokens,
        "kvCache": kv_cache,
    })
}

/// The `prompt_refine` result payload, parity with the Python `run_prompt_refine_job`.
#[cfg(native_lane)]
#[allow(clippy::too_many_arguments)]
fn refine_result(
    original_prompt: &str,
    refined_prompt: &str,
    thinking: Option<&str>,
    model: &str,
    backend: &str,
    thinking_mode: &str,
    finish_reason: Option<gen_core::core_llm::FinishReason>,
    usage: gen_core::core_llm::Usage,
    max_new_tokens: u32,
    kv_cache: Value,
) -> JsonObject {
    let mut result = JsonObject::new();
    result.insert("originalPrompt".to_owned(), json!(original_prompt));
    result.insert("refinedPrompt".to_owned(), json!(refined_prompt));
    if let Some(thinking) = thinking.filter(|value| !value.trim().is_empty()) {
        result.insert("thinking".to_owned(), json!(thinking));
    }
    result.insert(
        "generation".to_owned(),
        generation_block(finish_reason, usage, max_new_tokens, kv_cache),
    );
    result.insert(
        "executionIdentity".to_owned(),
        json!({
            "provider": "native",
            "model": model,
            "backend": backend,
            "thinkingMode": thinking_mode,
        }),
    );
    result
}

#[cfg(any(test, native_lane))]
fn finish_reason_name(reason: Option<gen_core::core_llm::FinishReason>) -> &'static str {
    use gen_core::core_llm::FinishReason;

    match reason {
        Some(FinishReason::Stop) => "stop",
        Some(FinishReason::Length) => "length",
        Some(FinishReason::Cancelled) => "cancelled",
        Some(FinishReason::ContentFilter) => "content_filter",
        None => "unknown",
    }
}

#[cfg(any(test, native_lane))]
fn empty_refine_failure_detail(
    task: RefineTask,
    finish_reason: Option<gen_core::core_llm::FinishReason>,
    generated_tokens: u32,
    max_new_tokens: u32,
) -> String {
    if task == RefineTask::FilmPlan
        && finish_reason == Some(gen_core::core_llm::FinishReason::Length)
    {
        return format!(
            "The prompt-refinement model exhausted its {max_new_tokens}-token output budget before \
             producing a final answer (generated {generated_tokens} tokens). Select Thinking \
             disabled and try again, or continue with manual plan editing."
        );
    }
    "The prompt-refinement model returned an empty prompt.".to_owned()
}

#[cfg(any(test, native_lane))]
#[allow(clippy::too_many_arguments)]
fn refine_failure_result(
    thinking: Option<&str>,
    model: &str,
    backend: &str,
    thinking_mode: &str,
    finish_reason: Option<gen_core::core_llm::FinishReason>,
    usage: gen_core::core_llm::Usage,
    max_new_tokens: u32,
    kv_cache: Value,
) -> JsonObject {
    let mut result = JsonObject::new();
    if let Some(thinking) = thinking.filter(|value| !value.trim().is_empty()) {
        result.insert("thinking".to_owned(), json!(thinking));
    }
    result.insert(
        "generation".to_owned(),
        generation_block(finish_reason, usage, max_new_tokens, kv_cache),
    );
    result.insert(
        "executionIdentity".to_owned(),
        json!({
            "provider": "native",
            "model": model,
            "backend": backend,
            "thinkingMode": thinking_mode,
        }),
    );
    result
}

#[cfg(test)]
#[path = "prompt_refine_jobs_tests.rs"]
mod tests;
