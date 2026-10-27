// Job-status/job-type/quality enums live in jobTypes.js (single source of truth,
// sc-1657). Re-exported here so existing `from "./constants.js"` importers are
// unaffected.
export { terminalStatuses, actionStatuses } from "./jobTypes.js";

// How many jobs one queue listing carries. Mirrors `MAX_LISTED_JOBS` in
// crates/sceneworks-core/src/jobs_store.rs — `GET /api/v1/jobs` takes no offset, so the
// server's page depth and every client bound derived from it must agree or the UI
// silently shows less history than the server just sent.
export const MAX_LISTED_JOBS = 1000;

// The offline model-catalog mirror (`fallbackModels`) lives in modelCatalog.js (sc-24217) so it
// stays out of the initial bundle; import it from there, and only lazily in production code.

// Asset Library hygiene (sc-2024 / sc-8339): the Main Asset Library shows ONLY media
// produced by the Image / Video / Document studios plus manual uploads. This is a
// positive allow-list (not a single `!== "character_studio"` exclusion) so anything
// else — Character Studio outputs (`character_studio`), Pose/Key Point library assets
// (`pose_library` / `keypoint_library`), and any future origin — stays out by default.
// Origins are assigned by the backend (`crates/sceneworks-core/src/asset_index.rs`).
// `audio_studio` (epic 13400) was never added here when the Audio Studio landed, so every
// generated clip was filtered out of BOTH asset libraries — the Advanced Library screen and
// the Simple Assets grid, whose "Audio" filter pill could therefore never match anything.
// Audio is studio-generated media like the other three, so it belongs (epic 14361).
export const LIBRARY_ORIGINS = new Set([
  "image_studio",
  "video_studio",
  "audio_studio",
  "document_studio",
  "vector_studio",
  "upload",
]);

// True when an asset belongs in the Main Asset Library. A missing origin is treated as
// eligible: the normalized API always stamps one, so only legacy/non-normalized records
// lack it, and hiding them would be worse than showing them. A *recorded* origin must be
// in the allow-list.
export function isLibraryAsset(asset) {
  const origin = asset?.origin;
  return !origin || LIBRARY_ORIGINS.has(origin);
}

// SenseNova-U1 interleave resolution buckets (distinct from plain text-to-image).
// Mirrors the worker's interleave_resolution_for / upstream examples/interleave.
export const INTERLEAVE_RESOLUTION_OPTIONS = [
  "1536x1536",
  "2048x1152",
  "1152x2048",
  "1888x1248",
  "1248x1888",
  "1760x1312",
  "1312x1760",
];
export const DEFAULT_INTERLEAVE_RESOLUTION = "2048x1152";

// Reference strength for grounded (it2i) interleave — drives advanced.imageGuidanceScale,
// which the worker maps to the engine's img_cfg_scale (a CFG weight, not clamped by the
// backend). 1.0 is the upstream/edit baseline (references are full context, text CFG steers
// relative to them); higher holds the output tighter to the reference images, lower is more
// prompt-driven. The engine's own guidance is edit ≈ 1.0, character/identity ≈ 1.5. Range is
// deliberately tightened to the useful band (0.5–2.5) rather than Character Studio's wider
// 0.5–4.0 — past ~2.5 an image-CFG weight buys oversaturation, not adherence. Only meaningful
// when references are attached, so Document Studio only surfaces the slider (and sends the
// field) then.
export const DEFAULT_INTERLEAVE_IMAGE_GUIDANCE = 1.0;
export const INTERLEAVE_IMAGE_GUIDANCE_MIN = 0.5;
export const INTERLEAVE_IMAGE_GUIDANCE_MAX = 2.5;
export const INTERLEAVE_IMAGE_GUIDANCE_STEP = 0.1;

// Catalog id of the prompt-refinement / magic-prompt LLM (sc-5605 / sc-6550, builtin.models.jsonc):
// one coherent Anubis-8B serves BOTH the free-text "Refine my prompt" rewrite AND Ideogram 4's
// magic-prompt (plain idea -> structured JSON caption) — the sc-6550 bake-off found the old 3B / plain
// Llama-3.1-8B emit degenerate captions that placeholder. The native worker resolves the model by repo
// string, not this id; the web uses it to look up the catalog entry's install state and to enqueue its
// ModelDownload job when refine / magic-prompt needs it.
export const PROMPT_REFINE_MODEL_ID = "prompt_refine_anubis_8b";

// Catalog id of the dataset-captioning model (sc-5620, builtin.models.jsonc). Same pattern
// as PROMPT_REFINE_MODEL_ID — the native captioner resolves it by repo string; the web uses
// this id to look up install state and offer a download in the caption dialog when missing.
export const JOY_CAPTION_MODEL_ID = "joycaption_beta_one";

// Wan2.2 A14B MoE video engines (T2V + I2V). These two dual-expert models honor the
// Lightning fast-4-step toggle (sc-10047/sc-10048, epic 10043) — the worker derives the
// 4-step/CFG-off recipe from `advanced.lightning` for them and ignores it for the dense
// 5B (wan_2_2) and non-Wan engines, so the UI only surfaces the toggle for these ids.
export const WAN_A14B_LIGHTNING_MODEL_IDS = new Set(["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]);

// Wan A14B is a two-expert mixture, so its LoRAs come as a high/low-noise PAIR. These base
// models accept the optional low-noise expert upload in Model Manager (sc-1991) and are the
// ones the Simple Video Studio's LoRA hint applies to (epic 15404); the dense 5B (wan_2_2) is
// a single-file LoRA. Same ids as WAN_A14B_LIGHTNING_MODEL_IDS above but a DIFFERENT fact —
// that one is about the 4-step distilled sampling recipe — so they stay separate constants.
export const WAN_MOE_PAIRED_LORA_MODEL_IDS = new Set(["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]);

// Vision captioner (epic 8102, sc-8107/8108). Turns an uploaded reference image into a structured
// Ideogram-style JSON caption (style + grounded composition) for Ideogram 4 text-to-image variations.
// As with PROMPT_REFINE_MODEL_ID, the native worker resolves the model by its HF repo string (carried
// in the job payload's `model` field); the web uses the catalog id only to look up install state and
// offer a download (the eligibility gate is the sibling sc-8110). Native MLX and Candle workers
// both serve the production image-caption request.
export const VISION_CAPTION_MODEL_ID = "vision_caption_qwen3vl_8b";
export const VISION_CAPTION_MODEL_REPO = "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated";

// Qwen-Image 2.1 and its two OFFICIAL prompt rewriters (sc-24113, epic 24107).
//
// The rewriters are OPTIONAL, explicitly downloaded checkpoints — `autoDownload: false` in the
// manifest — and the image model works fully without either. Which of the two applies is decided by
// the REQUEST (references attached ⇒ the editing one) and never by a user-facing picker, so these
// ids are looked up by the caller, not offered as a choice. They mirror
// `crates/sceneworks-worker/src/qwen_prompt_rewrite.rs`'s `Rewriter::catalog_id`.
export const QWEN_IMAGE_2_1_MODEL_ID = "qwen_image_2_1";
export const QWEN_REWRITE_T2I_MODEL_ID = "qwen_image_2_1_pe_t2i";
export const QWEN_REWRITE_I2I_MODEL_ID = "qwen_image_2_1_pe_i2i";

// The rewrite TARGETS: base 2.1 and its distilled Turbo sibling (sc-25908, epic 25899) — one prompt
// format, one template. A mirror of `sceneworks_core::jobs_store::QWEN_IMAGE_2_1_MODEL_IDS`, the
// exact set the worker serves a rewrite for (`qwen_prompt_rewrite::is_rewrite_target_model`);
// `qwenImage21Turbo.test.jsx` reads the Rust list and holds this one to it. Keyed on the id, NOT
// the family: an imported custom checkpoint can carry family `qwen-image-2-1` and the worker
// refuses a rewrite targeting it.
export const QWEN_IMAGE_2_1_MODEL_IDS = Object.freeze([QWEN_IMAGE_2_1_MODEL_ID, "qwen_image_2_1_turbo"]);

// Whether the official Qwen rewriters apply to the selected generation model.
export function qwenRewriterAppliesTo(model) {
  return QWEN_IMAGE_2_1_MODEL_IDS.includes(model?.id);
}

// DWPose whole-body pose detector (sc-17634 re-host, `type: "utility"` in builtin.models.jsonc).
// One catalog entry carrying BOTH ONNX graphs the pose lane loads — YOLOX person boxes + the RTMW
// 133-point COCO-WholeBody model. Since sc-17634 the worker's resolver is CACHE-ONLY: it never
// downloads mid-job, so a missing install can only surface as a failed `pose_detect` job unless the
// UI checks first. As with PROMPT_REFINE_MODEL_ID / JOY_CAPTION_MODEL_ID the worker resolves the
// graphs by repo+revision, not by this id; the web uses the id purely to read install state and
// offer the download in place (the eligibility predicate is `poseDetectModelUsable`).
export const POSE_DETECT_MODEL_ID = "dwpose_pose_detector";

// YOLO11m person detector (sc-17629/17635, `type: "utility"`). The sibling of POSE_DETECT_MODEL_ID
// in every respect that matters here: epic 17625 made `person_jobs::require_detector_weights`
// cache-only too, so a missing install is an error at job time rather than an ~80 MB mid-job fetch.
// Catalog analysis needs BOTH this and DWPose before it can run structured analysis.
export const PERSON_DETECT_MODEL_ID = "person_detector";

// Default interleave system prompt (the think/no-think protocol). Prefilled in
// Document Studio; the worker falls back to this same text when the field is blank.
// Keep in sync with the native worker's interleave system-message default.
export const DEFAULT_INTERLEAVE_SYSTEM_MESSAGE = `You are a multimodal assistant capable of reasoning with both text and images. You support two modes:

Think Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.

Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.

After the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer.`;
