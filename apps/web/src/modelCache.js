// Resolved-model hot cache — client seam for the Settings storage card and the Model Manager's
// per-model local-copy controls (sc-19711).
//
// Two rules this module exists to enforce, and which every consumer inherits:
//
// 1. **Availability is read, never re-derived.** `modelAvailability` is the typed judgement the
//    ONE shared resolver produced (sc-19708). Nothing here inspects install paths, counts files,
//    or matches on error text to guess a state; an unrecognized value renders as "unknown" rather
//    than being coerced into a reassuring one.
// 2. **A removal is described from the backend's own preview.** The confirm copy is built from
//    `manual_removal_preview` — measured bytes, the typed pin answer, the source-availability
//    warning and the refusal reason — so the dialog cannot promise something the store would then
//    refuse. `pins.kind === "unknown"` is surfaced as "can't determine", never as "not pinned".
import { apiFetch } from "./api.js";
import { readAccessToken } from "./accessToken.js";
import { formatBytes } from "./formatting.js";

const SECONDS_PER_DAY = 24 * 60 * 60;
const BYTES_PER_GIB = 1024 * 1024 * 1024;

// The typed availability states from `ModelAvailability` (sc-19708), verbatim. The value is the
// wire form; nothing else in the UI may invent a sixth state.
export const MODEL_AVAILABILITY = {
  LOCAL_READY: "local_ready",
  EXTERNAL_READY: "external_ready",
  INSTALLED_EXTERNAL_UNAVAILABLE: "installed_external_unavailable",
  INCOMPLETE: "incomplete",
  MISSING: "missing",
};

// One badge per typed state. `tone` maps onto the screen's existing `.status-badge` modifiers, so
// this introduces no new visual idiom.
const AVAILABILITY_BADGES = {
  [MODEL_AVAILABILITY.LOCAL_READY]: {
    text: "local copy",
    tone: "installed",
    title: "A SceneWorks-owned copy is on this machine, so this model loads even if the external library is disconnected.",
  },
  [MODEL_AVAILABILITY.EXTERNAL_READY]: {
    text: "on external library",
    tone: "",
    title: "Installed on the configured external model library, which is currently connected.",
  },
  [MODEL_AVAILABILITY.INSTALLED_EXTERNAL_UNAVAILABLE]: {
    text: "library disconnected",
    tone: "warning",
    title: "This model is installed on an external library that isn't reachable right now. The installation is preserved; reconnect the library to use it.",
  },
  [MODEL_AVAILABILITY.INCOMPLETE]: {
    text: "incomplete",
    tone: "warning",
    title: "Some required files are missing from the installation.",
  },
  [MODEL_AVAILABILITY.MISSING]: {
    text: "not installed",
    tone: "",
    title: "No installation of this model was found.",
  },
};

// The badge for a catalog row's typed availability, or null when the row carries no judgement at
// all. An UNRECOGNIZED value is deliberately labelled rather than silently dropped or defaulted:
// a state this build doesn't know about must not read as "fine".
export function availabilityBadge(model) {
  const availability = model?.modelAvailability;
  if (typeof availability !== "string" || availability === "") {
    return null;
  }
  return (
    AVAILABILITY_BADGES[availability] ?? {
      text: "unknown state",
      tone: "warning",
      title: `This build doesn't recognize the availability state "${availability}".`,
    }
  );
}

// True only where a local copy can exist at all: the resolved cache holds copies of models whose
// authoritative source is an external library.
export function canHoldLocalCopy(model) {
  return (
    model?.modelAvailability === MODEL_AVAILABILITY.LOCAL_READY ||
    model?.modelAvailability === MODEL_AVAILABILITY.EXTERNAL_READY ||
    model?.modelAvailability === MODEL_AVAILABILITY.INSTALLED_EXTERNAL_UNAVAILABLE
  );
}

// ---- transport ------------------------------------------------------------------

export function fetchModelCache() {
  return apiFetch("/api/v1/model-cache", readAccessToken());
}

export function previewCacheRemoval(cacheKey) {
  return apiFetch("/api/v1/model-cache/removal-preview", readAccessToken(), {
    method: "POST",
    body: JSON.stringify({ cacheKey }),
  });
}

export function removeCacheEntry(cacheKey) {
  return apiFetch("/api/v1/model-cache/remove", readAccessToken(), {
    method: "POST",
    body: JSON.stringify({ cacheKey }),
  });
}

export function setCacheEntryPin(cacheKey, pinned) {
  return apiFetch("/api/v1/model-cache/pin", readAccessToken(), {
    method: "POST",
    body: JSON.stringify({ cacheKey, pinned }),
  });
}

// ---- projections ----------------------------------------------------------------

// The cache entries the backend joined to this model id. The join lives in the API (one identity
// mapping); the UI only indexes it.
export function entriesForModel(status, modelId) {
  if (!modelId || !Array.isArray(status?.entries)) {
    return [];
  }
  return status.entries.filter(
    (entry) => Array.isArray(entry.modelIds) && entry.modelIds.includes(modelId),
  );
}

export function gibToBytes(gib) {
  const value = Number(gib);
  return Number.isFinite(value) && value > 0 ? Math.round(value * BYTES_PER_GIB) : 0;
}

export function bytesToGib(bytes) {
  const value = Number(bytes);
  return Number.isFinite(value) && value > 0 ? value / BYTES_PER_GIB : 0;
}

export function daysToSeconds(days) {
  const value = Number(days);
  return Number.isFinite(value) && value > 0 ? Math.round(value * SECONDS_PER_DAY) : 0;
}

export function secondsToDays(seconds) {
  const value = Number(seconds);
  return Number.isFinite(value) && value > 0 ? value / SECONDS_PER_DAY : 0;
}

// True when the persisted policy the shell holds differs from the policy the running API captured
// at spawn — i.e. the change is saved but not yet in effect.
export function policyNeedsRestart(persisted, effective) {
  if (!persisted || !effective) {
    return false;
  }
  return (
    Boolean(persisted.enabled) !== Boolean(effective.enabled) ||
    Number(persisted.maxBytes) !== Number(effective.maxBytes) ||
    Number(persisted.inactivitySeconds) !== Number(effective.inactivitySeconds)
  );
}

// What turning the cache OFF would actually mean. Disabling stops new copies; it does not sweep,
// and saying otherwise would be a claim the backend never makes.
export function describeDisableConsequence(status) {
  const count = Number(status?.entryCount) || 0;
  if (count === 0) {
    return "No local copies exist yet, so turning this off changes nothing on disk.";
  }
  const used = formatBytes(status?.usedBytes ?? 0);
  const one = count === 1;
  return `SceneWorks stops making new local copies. The ${count} existing ${
    one ? "copy" : "copies"
  } (${used}) ${one ? "stays" : "stay"} on disk — remove ${
    one ? "it" : "them"
  } per model in Model Manager, or turn this back on and let automatic cleanup reclaim ${
    one ? "it" : "them"
  }.`;
}

// What LOWERING the size limit would actually mean. Nothing is swept at the moment of the change:
// automatic cleanup runs on the worker's own idle checkpoints, and pinned or in-use copies are
// never taken.
export function describeLimitConsequence(status, nextMaxBytes) {
  const used = Number(status?.usedBytes) || 0;
  const reclaimable = Number(status?.reclaimableBytes) || 0;
  const next = Number(nextMaxBytes) || 0;
  if (next <= 0 || used <= next) {
    return "";
  }
  const over = used - next;
  const eligible = Math.min(over, reclaimable);
  if (eligible <= 0) {
    return `Local copies already use ${formatBytes(used)}, above the new ${formatBytes(
      next,
    )} limit, but every copy is kept (pinned or in use), so nothing can be reclaimed automatically.`;
  }
  const shortfall = over - eligible;
  const tail =
    shortfall > 0
      ? ` The remaining ${formatBytes(shortfall)} is kept (pinned or in use) and will stay over the limit.`
      : "";
  return `Local copies use ${formatBytes(used)}. The next automatic cleanup will remove up to ${formatBytes(
    eligible,
  )} of unpinned, unused copies to get under ${formatBytes(next)}.${tail}`;
}

// ---- destructive-action copy ----------------------------------------------------

// Human sentences describing exactly what the backend's preview says, in the order a person needs
// them: what is reclaimed, what blocks it, what the pin state is, and what removal costs.
//
// `pins.kind === "unknown"` is its own sentence. It is NOT folded into "not pinned": the store
// could not read the entry's state, so the honest statement is that we can't tell.
export function describeRemovalPreview(preview) {
  if (!preview) {
    return [];
  }
  const lines = [];
  if (preview.blocked) {
    lines.push(`This copy can't be removed right now: ${preview.blocked}`);
  } else {
    lines.push(`Removing this local copy frees ${formatBytes(preview.reclaimableBytes)}.`);
  }
  if (preview.pins?.kind === "unknown") {
    lines.push(
      "SceneWorks couldn't determine whether this copy is being kept — its stored state can't be read.",
    );
  } else if (preview.pins?.kind === "known") {
    const owners = Array.isArray(preview.pins.owners) ? preview.pins.owners : [];
    if (preview.pins.artifactPinned) {
      lines.push("You marked this copy to keep locally. Allow automatic removal first.");
    }
    if (owners.length) {
      lines.push(
        `${owners.length} loaded ${owners.length === 1 ? "model is" : "models are"} still holding this copy.`,
      );
    }
  }
  if (preview.sourceUnavailableWarning) {
    lines.push(
      `The original files aren't reachable right now, so this model becomes unusable until its library is reconnected (${preview.sourceUnavailableWarning}).`,
    );
  } else {
    lines.push(
      "The original files in the model library are not touched — SceneWorks re-copies them when it next needs them.",
    );
  }
  return lines;
}

// True when the preview says the removal would go through. A blocked preview must never present a
// confirm button that the store will then refuse.
export function removalIsAllowed(preview) {
  return Boolean(preview) && !preview.blocked;
}
