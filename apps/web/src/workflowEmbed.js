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

const EMBED_STORAGE_KEY = "sceneworks-embed-workflow";
const NOTICE_STORAGE_KEY = "sceneworks-embed-workflow-notice-seen";

// The doc that is the contract for what travels. Rooted at `PRODUCER_URL` — the same repository
// URL the envelope itself carries — so the link in the app and the link in a shared file agree.
export const WORKFLOW_SHARE_DOC_URL =
  "https://github.com/SceneWorks/SceneWorks/blob/main/docs/workflow-share-envelope.md";

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
  const names = EMBEDDED_PROSE_FIELDS.map(([, label]) => label);
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
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
