// Small initial-path preference seam. Sharing disclosure copy loads with its interaction.
import { putUiPreferences } from "./uiPreferences.js";
const EMBED_STORAGE_KEY = "sceneworks-embed-workflow";
const NOTICE_STORAGE_KEY = "sceneworks-embed-workflow-notice-seen";

// The ONE name for the control that writes a copy with the recipe taken out. Three surfaces say it
// — the context menu, the button beside Save As, and the settings copy that tells you to use it —
// and before this they said three different things. `the_settings_copy_names_the_save_without_
// workflow_control_once` pins it against docs/workflow-share-envelope.md as well, so the document
// and the button agree too.
export const SAVE_WITHOUT_WORKFLOW_LABEL = "Save a copy without the workflow";

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
