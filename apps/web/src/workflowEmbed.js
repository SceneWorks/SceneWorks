// The `embedWorkflowInImages` preference, its first-run disclosure, and the copy both surfaces
// show (sc-15953, epic 15945).
//
// The preference itself is already plumbed: `PUT /api/v1/ui-preferences` persists it into
// `ui-preferences.json`, and the WORKER reads it straight off that file at the PNG write seam
// through `sceneworks_core::app_paths::embed_workflow_in_images`. So flipping it here takes effect
// on the next job with no restart, and nothing in the web app has to tell the worker anything.
//
// Two durable flags, both server-side, both mirrored into localStorage only as an instant-paint
// cache. The durable copy is not an optimization — on the desktop shell the UI runs at the API's
// per-launch `http://127.0.0.1:<port>` origin, so origin-keyed localStorage does NOT survive a
// relaunch. A "we already told them" flag that lived only in localStorage would re-fire the
// disclosure on every launch, which is the failure mode the once-only requirement names.
//
// Absent means ON for the preference, and NOT SEEN for the notice. Both defaults match the Rust
// reader's: `embed_workflow_in_images_from_json` treats an absent key as `true`.

import { putUiPreferences } from "./uiPreferences.js";

const EMBED_STORAGE_KEY = "sceneworks-embed-workflow";
const NOTICE_STORAGE_KEY = "sceneworks-embed-workflow-notice-seen";

// The repository URL the envelope itself carries in `producer.url`, so a file that reaches a
// stranger is self-identifying. Pinned against the Rust `PRODUCER_URL` by
// `the_settings_copy_names_exactly_the_path_exempt_prose_fields`.
export const PRODUCER_URL = "https://github.com/SceneWorks/SceneWorks";

// The doc that is the contract for what travels, DERIVED from `PRODUCER_URL` rather than written
// out beside it — the link in the app and the link in a shared file are then the same URL by
// construction instead of by a comment saying they are.
export const WORKFLOW_SHARE_DOC_URL = `${PRODUCER_URL}/blob/main/docs/workflow-share-envelope.md`;

// The ONE name for the control that writes a copy with the recipe taken out. Three surfaces say it
// — the context menu, the button beside Save As, and the settings copy that tells you to use it —
// and before this they said three different things. `the_settings_copy_names_the_save_without_
// workflow_control_once` pins it against docs/workflow-share-envelope.md as well, so the document
// and the button agree too.
export const SAVE_WITHOUT_WORKFLOW_LABEL = "Save a copy without the workflow";

// The fields recorded exactly as the user authored them, and deliberately EXEMPT from the
// filesystem-path guard that drops every other field that looks like a location. This is the one
// claim in the settings copy that must not drift, because it is the claim a user acts on before
// sharing — so it is a declared list rather than a sentence, and
// `the_settings_copy_names_exactly_the_path_exempt_prose_fields` in
// `crates/sceneworks-core/tests/workflow_share_doc.rs` pins the left column against the
// `prose-fields` table of docs/workflow-share-envelope.md, in both directions. A seventh prose
// field added to the sanitizer fails this file, not just the doc.
export const EMBEDDED_PROSE_FIELDS = Object.freeze([
  ["prompt", "the prompt"],
  ["negativePrompt", "the negative prompt"],
  ["advanced.stylePrompt", "the style prompt"],
  ["advanced.systemMessage", "the system message"],
  ["advanced.structuredPrompt.intent", "a structured prompt's intent"],
  ["advanced.structuredPrompt.runtimePrompt", "the structured prompt the model received"],
]);

// "the prompt, the negative prompt, … and the structured prompt the model received".
export function proseFieldSentence() {
  return joinLabels(EMBEDDED_PROSE_FIELDS.map(([, label]) => label));
}

// "a, b, … and z", with repeats collapsed and declaration order preserved.
function joinLabels(labels) {
  const unique = [...new Set(labels)];
  if (unique.length < 2) {
    return unique[0] ?? "";
  }
  return `${unique.slice(0, -1).join(", ")} and ${unique[unique.length - 1]}`;
}

// ---------------------------------------------------------------------------
// The two lists a user acts on
// ---------------------------------------------------------------------------
//
// "Also in the file" and "Not in the file" are the paragraphs someone reads before deciding whether
// to send an image to a client, and until now they were prose pinned to nothing. That is not a
// theoretical gap: `advanced.poses` — the `keypoints` / `hands` / `face` coordinate arrays derived
// from a reference photo, facial landmarks included — travels, and the copy named none of it while
// the very next paragraph said the input images do not travel. A reader could reasonably conclude
// that nothing derived from a reference leaves, which is the forbidden direction: claiming more
// privacy than the allow-list delivers.
//
// So both lists are declared per KEY, exactly the way `EMBEDDED_PROSE_FIELDS` is, and the sentences
// are rendered from them rather than written out. `the_settings_copy_accounts_for_every_shared_and
// _withheld_field` in `crates/sceneworks-core/tests/workflow_share_doc.rs` pins the keys against the
// `envelope-fields`, `advanced-shared` and `advanced-withheld` tables of
// docs/workflow-share-envelope.md in both directions. A new allow-listed setting therefore cannot
// reach a shared image without someone deciding, in this file, what the copy says about it.
//
// Labels are shared constants rather than repeated string literals, because one phrase legitimately
// covers several keys ("its size" is `width` and `height`). Repeats collapse when the sentence is
// built, so the reader sees the phrase once and the pin still sees every key.

const L_MARKER = "a marker saying this is a SceneWorks recipe, and which version of the format it is";
const L_PRODUCER = "the SceneWorks name, project URL and released version of the build that wrote the file";
const L_MODE = "the generation mode";
const L_NAMES = "the model and style names";
const L_RESOURCE_HASHES =
  "the exact checkpoint and LoRA content hashes, when available, so galleries can attribute their versions and authors";
const L_PROSE = "the prompt fields named above, verbatim";
const L_SEED = "the seed for this image";
const L_SIZE = "the image size the run asked for";
const L_FIT = "how a source image was fitted";
const L_COUNT = "how many images the run asked for";
const L_UPSCALE = "any upscale settings";
const L_LORAS = "LoRA names, weights, Hugging Face repo ids and exact content hashes";
const L_SETTINGS =
  "the shared generation settings (sampler, scheduler, steps, guidance, strength and the other numeric knobs)";
const L_PASSES =
  "which optional passes ran — caption upsampling, the PiD or alternate decoder and PiD tier, face restoration — and the control type";
const L_ANGLE = "the head angle or turnaround the run asked for";
const L_POSES =
  "the pose, hand and face coordinates a pose selection produced — landmark arrays derived from a reference photo, facial landmarks included";
const L_PHASES = "the multi-phase denoise schedule, with each phase's steps, guidance and LoRA weights";
const L_INPUTS =
  "which kinds of input image or CLIP the recipe needs and how many of each, but never which ones";
// The video lane (sc-15956). Its knobs are the same CLASS as the image ones — what to make,
// never what this machine can afford — so several of them share the labels above.
const L_CLIP = "the clip length, frame rate and quality preset a video run asked for";
const L_MOTION = "the camera-motion preset, and which timeline action produced a clip";
const L_VIDEO_ENGINE =
  "which video pipeline, transformer, checkpoint and VAE decoder variants, and fast-step recipe a video run used";
const L_VIDEO_TIMING =
  "whether LTX predicted the clip duration, its requested bounds and the temporal-refinement count";
const L_TEXT_ENCODER = "which text encoder read the prompt";
const L_OMITTED = "a note naming any list that was too long to record";

// Every field the envelope can carry, and what the copy says about it. Keys prefixed `advanced.`
// are the allow-listed advanced settings; the rest are top-level envelope fields.
export const WORKFLOW_FIELDS_IN_FILE = Object.freeze([
  ["sceneworksWorkflow", L_MARKER],
  ["schemaVersion", L_MARKER],
  ["producer.name", L_PRODUCER],
  ["producer.url", L_PRODUCER],
  ["producer.version", L_PRODUCER],
  ["mode", L_MODE],
  ["model", L_NAMES],
  ["modelHash", L_RESOURCE_HASHES],
  ["stylePreset", L_NAMES],
  ["styleId", L_NAMES],
  ["advanced.styleId", L_NAMES],
  ["prompt", L_PROSE],
  ["negativePrompt", L_PROSE],
  ["advanced.stylePrompt", L_PROSE],
  ["advanced.systemMessage", L_PROSE],
  ["advanced.structuredPrompt", L_PROSE],
  ["seed", L_SEED],
  ["width", L_SIZE],
  ["height", L_SIZE],
  ["advanced.resolution", L_SIZE],
  ["fitMode", L_FIT],
  ["count", L_COUNT],
  ["upscale.enabled", L_UPSCALE],
  ["upscale.factor", L_UPSCALE],
  ["upscale.engine", L_UPSCALE],
  ["upscale.softness", L_UPSCALE],
  ["loras[].name", L_LORAS],
  ["loras[].weight", L_LORAS],
  ["loras[].repo", L_LORAS],
  ["loras[].hash", L_LORAS],
  ["advanced", L_SETTINGS],
  ["advanced.sampler", L_SETTINGS],
  ["advanced.scheduler", L_SETTINGS],
  ["advanced.schedulerShift", L_SETTINGS],
  ["advanced.steps", L_SETTINGS],
  ["advanced.guidanceScale", L_SETTINGS],
  ["advanced.guidanceMethod", L_SETTINGS],
  ["advanced.trueCfgScale", L_SETTINGS],
  ["advanced.strength", L_SETTINGS],
  ["advanced.cnScale", L_SETTINGS],
  ["advanced.controlScale", L_SETTINGS],
  ["advanced.ipAdapterScale", L_SETTINGS],
  ["advanced.controlnetConditioningScale", L_SETTINGS],
  ["advanced.textStyleGain", L_SETTINGS],
  ["advanced.imageGuidanceScale", L_SETTINGS],
  ["advanced.enhancePrompt", L_PASSES],
  ["advanced.usePid", L_PASSES],
  ["advanced.decoder", L_PASSES],
  ["advanced.pidTarget", L_PASSES],
  ["advanced.faceRestore", L_PASSES],
  ["advanced.controlMode", L_PASSES],
  ["advanced.viewAngle", L_ANGLE],
  ["advanced.angleSet", L_ANGLE],
  ["advanced.poses", L_POSES],
  ["advanced.phases", L_PHASES],
  ["inputs[].kind", L_INPUTS],
  ["inputs[].count", L_INPUTS],
  ["inputs[].controlMode", L_INPUTS],
  ["durationSeconds", L_CLIP],
  ["fps", L_CLIP],
  ["quality", L_CLIP],
  ["advanced.motion", L_MOTION],
  ["advanced.timelineAction", L_MOTION],
  ["advanced.ltxPipeline", L_VIDEO_ENGINE],
  ["advanced.distilledVariant", L_VIDEO_ENGINE],
  ["advanced.transformerVariant", L_VIDEO_ENGINE],
  ["advanced.vaeDecoder", L_VIDEO_ENGINE],
  ["advanced.autoDuration", L_VIDEO_TIMING],
  ["advanced.autoDurationMinSeconds", L_VIDEO_TIMING],
  ["advanced.autoDurationMaxSeconds", L_VIDEO_TIMING],
  ["advanced.temporalUpsampleRounds", L_VIDEO_TIMING],
  ["advanced.textEncoderModel", L_TEXT_ENCODER],
  ["advanced.lightning", L_VIDEO_ENGINE],
  ["advanced.videoCfgGuidanceScale", L_SETTINGS],
  ["advanced.videoStgGuidanceScale", L_SETTINGS],
  ["advanced.videoRescaleScale", L_SETTINGS],
  ["advanced.videoConditioningStrength", L_SETTINGS],
  ["advanced.bridgeRightVideoConditioningStrength", L_SETTINGS],
  ["omitted", L_OMITTED],
]);

const N_IDENTITY = "your user or machine name";
const N_PROJECT = "project, job and asset ids";
const N_TIME = "any timestamp";
const N_SEEDS = "the other seeds in a batch";
// Deliberately carries its own caveat. "The input images themselves" sitting next to a claim that
// the file is otherwise clean invites the reading that nothing derived from a reference travels —
// and `advanced.poses`, listed above, is exactly that.
const N_IMAGES = "the input images themselves, though what a pose selection traced from one does travel";
const N_BUDGET = "this machine's quality tier, attention kernel or GPU";
const N_LOCAL = "local library ids — a Key Point collection, a control image, a trained overlay, a recipe preset";
// The only claim in this list made for PRIVACY rather than for portability: a shared video must not
// disclose WHO was replaced, or which variant of a replacement ran (sc-15956).
//
// It carries its own caveat for the same reason `N_IMAGES` does, and the caveat is load-bearing.
// This bullet used to read "anything about a person replacement", and that was FALSE in the leading
// clause while true in every item it went on to enumerate: `mode` is in the "In the file" list
// above, and a person-replacement run writes it verbatim as `replace_person`. `model` says the same
// thing again — `wan_2_2_vace_fun_14b` is a replacement engine. So the TECHNIQUE travels, and a
// bullet claiming otherwise claimed more privacy than the allow-list delivers, which is the one
// direction this file's header forbids. What the allow-list actually withholds is the IDENTITY and
// the VARIANT — the track, the name, the source filename, the masks, and the Face Only / Full
// Person label — and that is what the sentence now says.
//
// The key-level pin in `workflow_share_doc.rs` cannot catch a copy error of this shape, because
// `mode` is not a key in either list; `the_settings_copy_does_not_overclaim_about_person_
// replacement` in that file asserts this text against a real `replace_person` envelope instead.
const N_PERSON =
  "who was replaced or which variant of a replacement ran — the selected track, the person's name, the original file name, the mask images, and the Face Only / Full Person label. The generation mode and the model do travel, so the file does say that a replacement is what ran";
const N_TIMELINE = "your timeline's name and the local ids of the clips on it";
const N_ADVICE = "the on-screen advice the studio shows about a model";

// What the copy claims is absent. Every key here must be absent from BOTH doc tables above, which
// is the direction that matters: a withheld key that quietly became shared would otherwise leave
// this paragraph claiming a privacy the file no longer has.
export const WORKFLOW_FIELDS_NOT_IN_FILE = Object.freeze([
  ["projectId", N_PROJECT],
  ["projectName", N_PROJECT],
  ["jobId", N_PROJECT],
  ["assetId", N_PROJECT],
  ["generationSetId", N_PROJECT],
  ["characterId", N_PROJECT],
  ["characterLookId", N_PROJECT],
  ["seeds", N_SEEDS],
  ["advanced.quantTier", N_BUDGET],
  ["advanced.mlxQuantize", N_BUDGET],
  ["advanced.mlxQuantizeExplicit", N_BUDGET],
  ["advanced.convRot", N_BUDGET],
  ["advanced.flashAttn", N_BUDGET],
  ["advanced.keypointCollectionId", N_LOCAL],
  ["advanced.controlImage", N_LOCAL],
  ["advanced.controlWeights", N_LOCAL],
  ["advanced.recipePresetId", N_LOCAL],
  ["advanced.presetMissingLoras", N_LOCAL],
  ["personTrackId", N_PERSON],
  ["replacementMode", N_PERSON],
  ["advanced.selectedPersonTrack", N_PERSON],
  ["advanced.replacementModeLabel", N_PERSON],
  ["advanced.timelineContext", N_TIMELINE],
  ["advanced.precision", N_BUDGET],
  ["advanced.quantization", N_BUDGET],
  ["advanced.durationHint", N_ADVICE],
]);

// The absences with no field name to pin, because the envelope has no slot for them at all — they
// are left behind by the field list being closed rather than by a rule written against a key. Kept
// beside the keyed list so the sentence reads as one thought.
const UNKEYED_ABSENCES = Object.freeze([N_IDENTITY, N_TIME, N_IMAGES]);

// The second block (sc-15957), and the reason it needs its own sentence rather than a row in the
// list above.
//
// Both lists are pinned per KEY against the two doc tables, and every key in this block is already
// in `WORKFLOW_FIELDS_IN_FILE` — it carries no field the envelope does not. So the key-level pin is
// silent about it, exactly the way it was silent about the person-replacement bullet, and for the
// same structural reason: the claim is about a FORMAT rather than about a field.
//
// And it is a claim a user acts on. Someone who reads "your recipe is in the file" pictures a block
// only SceneWorks understands; what is actually written is a second copy of the prompt, seed and
// settings in the format Civitai and most galleries read and DISPLAY. Those are different facts
// about what happens when the image is posted somewhere, and the narrower one would be claiming more
// privacy than the file delivers — the one direction this file's header forbids.
//
// Pinned as prose by `the_settings_copy_says_the_file_is_also_gallery_readable` in
// `crates/sceneworks-core/tests/workflow_share_doc.rs`, which renders a real envelope's trailer and
// checks this sentence against it.
export const GALLERY_READABLE_COPY =
  "A second copy of the prompt, negative prompt, seed, size, sampler, steps and guidance is written in the format image galleries read, so a site like Civitai displays them beside your image instead of showing it as plain pixels. It is the same information as the block above and nothing more; both are removed together.";

// The bullets of the "Also in the file" list: every declared label, in order, repeats collapsed.
//
// A LIST rather than one comma-joined sentence, and not for taste. Several of these phrases carry
// their own commas and dashes ("the pose, hand and face coordinates … facial landmarks included"),
// so joining fifteen of them produces a paragraph in which a reader cannot tell where one claim ends
// and the next begins — on the surface whose whole job is to be read before sharing a file.
export function inFileItems() {
  return [...new Set(WORKFLOW_FIELDS_IN_FILE.map(([, label]) => label))];
}

// The bullets of "Not in the file". The unkeyed absences lead, because they are what a user scans
// for first.
export function notInFileItems() {
  return [
    ...new Set([...UNKEYED_ABSENCES, ...WORKFLOW_FIELDS_NOT_IN_FILE.map(([, label]) => label)]),
  ];
}

function readFlag(key, fallback) {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const stored = window.localStorage.getItem(key);
    if (stored === "true") {
      return true;
    }
    if (stored === "false") {
      return false;
    }
    return fallback;
  } catch {
    // Private mode / blocked storage. The durable server copy still carries it.
    return fallback;
  }
}

function writeFlag(key, value) {
  if (typeof window === "undefined") {
    return Boolean(value);
  }
  try {
    window.localStorage.setItem(key, value ? "true" : "false");
  } catch {
    // Private mode / quota — the durable server copy still carries it.
  }
  return Boolean(value);
}

// Whether a generated file carries the recipe. Absent means ON, matching the worker's reader.
//
// ONE switch for images and videos, not two (sc-15956). It is a privacy control over what leaves
// in a file, and the reason someone turns it off does not stop applying at the container
// boundary. A separate video switch would have re-enabled embedding for everyone who had already
// opted out, the day the video lane shipped. The stored key keeps its `embedWorkflowInImages`
// name for back-compatibility — renaming it would reset every existing opt-out to the default —
// so it is the COPY that names both media, which is what the user actually reads.
export function readEmbedWorkflowInImages() {
  return readFlag(EMBED_STORAGE_KEY, true);
}

export function writeEmbedWorkflowInImages(value) {
  return writeFlag(EMBED_STORAGE_KEY, value);
}

// Durable one-time decision marker. The old `notice seen` storage/API name is retained so existing
// installs do not lose their completed decision during the modal redesign.
export function readWorkflowEmbedNoticeSeen() {
  return readFlag(NOTICE_STORAGE_KEY, false);
}

export function writeWorkflowEmbedNoticeSeen(value) {
  return writeFlag(NOTICE_STORAGE_KEY, value);
}

// Write one of these two flags to the DURABLE store, retrying before giving up.
//
// `.catch(() => {})` was not good enough here, and for a reason specific to this pair. The
// localStorage mirror is an instant-paint cache and nothing more — on the desktop shell the UI runs
// at the API's per-launch `http://127.0.0.1:<port>` origin, so that mirror is a fresh empty store
// every launch. A dropped PUT therefore does not degrade to "saved locally", it degrades to "not
// saved at all", and the two failures that follow are the ones a user notices: a deliberate privacy
// opt-out silently reverting to the ON default, and a one-time disclosure firing again as though it
// had never been dismissed.
//
// So: three attempts with a doubling backoff, and the rejection is propagated rather than eaten, so
// the caller can tell the user the setting did not stick. `sleep` is injectable because a test that
// proves the retry should not spend a second doing it.
export async function persistWorkflowEmbedPreference(
  patch,
  { attempts = 3, delayMs = 400, sleep } = {},
) {
  const wait = sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
  let lastError = new Error("The preference write was never attempted.");
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (attempt > 0) {
      await wait(delayMs * 2 ** (attempt - 1));
    }
    try {
      await putUiPreferences(patch);
      return true;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

// Watch the localStorage mirror for changes made by ANOTHER tab.
//
// Two tabs of the browser shell are two independent React trees. Before this, a second tab that was
// already mounted when the notice was dismissed went on holding `seen: false` and would show the
// disclosure again on its own next generation — the "once" in "shown once" being per tab rather
// than per user.
//
// `seen` moves in ONE direction only. It is a record that something was said, so another tab
// clearing its storage is not evidence the user was never told, and letting it move back would make
// a `localStorage.clear()` in one tab re-arm the disclosure in every other. The embed preference is
// mirrored both ways, because that one genuinely is a setting with two states.
export function subscribeWorkflowEmbedFlags(onChange) {
  if (typeof window === "undefined") {
    return () => {};
  }
  const handler = (event) => {
    // A null key is the whole store being cleared, which touches both of ours.
    if (event.key !== null && event.key !== EMBED_STORAGE_KEY && event.key !== NOTICE_STORAGE_KEY) {
      return;
    }
    onChange({
      embed: readEmbedWorkflowInImages(),
      seen: readWorkflowEmbedNoticeSeen() || undefined,
    });
  };
  window.addEventListener("storage", handler);
  return () => window.removeEventListener("storage", handler);
}
