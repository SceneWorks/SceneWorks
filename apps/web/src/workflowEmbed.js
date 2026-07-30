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

const L_MARKER = "a marker saying this is a SceneWorks recipe, and which version of the format";
const L_PRODUCER = "the SceneWorks name, project URL and released version of the build that wrote it";
const L_MODE = "the generation mode";
const L_NAMES = "the model and style names";
const L_PROSE = "the prompt fields named above";
const L_SEED = "the seed for this image";
const L_SIZE = "its size";
const L_FIT = "how it was fitted";
const L_COUNT = "how many images the run asked for";
const L_UPSCALE = "any upscale settings";
const L_LORAS = "LoRA names, weights and Hugging Face repo ids";
const L_SETTINGS =
  "the shared generation settings (sampler, scheduler, steps, guidance, strength and the other numeric knobs)";
const L_PASSES =
  "which optional passes ran — caption upsampling, the PiD decoder and its tier, face restoration — and the control type";
const L_ANGLE = "the head angle or turnaround the run asked for";
const L_POSES =
  "the pose, hand and face coordinates a pose selection produced — landmark arrays derived from a reference photo, facial landmarks included";
const L_PHASES = "the multi-phase denoise schedule, with each phase's steps, guidance and LoRA weights";
const L_INPUTS = "which kinds of input image the recipe needs and how many of each";
const L_OMITTED = "a note of any list that was too long to record";

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
]);

// The absences with no field name to pin, because the envelope has no slot for them at all — they
// are left behind by the field list being closed rather than by a rule written against a key. Kept
// beside the keyed list so the sentence reads as one thought.
const UNKEYED_ABSENCES = Object.freeze([N_IDENTITY, N_TIME, N_IMAGES]);

// "the generation mode, the model and style names, …" — the rendered "Also in the file" list.
export function inFileSentence() {
  return joinLabels(WORKFLOW_FIELDS_IN_FILE.map(([, label]) => label));
}

// The rendered "Not in the file" list. The unkeyed absences lead, because they are what a user
// scans for first.
export function notInFileSentence() {
  return joinLabels([
    ...UNKEYED_ABSENCES,
    ...WORKFLOW_FIELDS_NOT_IN_FILE.map(([, label]) => label),
  ]);
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

// Whether a generated PNG carries the recipe. Absent means ON, matching the worker's reader.
export function readEmbedWorkflowInImages() {
  return readFlag(EMBED_STORAGE_KEY, true);
}

export function writeEmbedWorkflowInImages(value) {
  return writeFlag(EMBED_STORAGE_KEY, value);
}

// Whether the first-run disclosure has already been shown and dismissed. Absent means NOT seen,
// which is what makes an existing install upgrading into this build get the notice once: the
// default is silently ON, and someone who has been generating for months has never been told.
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
