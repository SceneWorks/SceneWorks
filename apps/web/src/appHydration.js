// Route-owned catalog hydration (sc-14783).
//
// Domains are deliberately explicit instead of inferred from component imports. That
// keeps startup/network behavior reviewable and gives tests a deterministic request
// contract. Global domains survive project switches; project domains are keyed by the
// active project and are reset/reloaded only when a consumer route needs them.
export const GLOBAL_HYDRATION_DOMAINS = Object.freeze(
  new Set(["models", "trainingTargets", "trainingPresets"]),
);

export const PROJECT_HYDRATION_DOMAINS = Object.freeze(
  new Set([
    "assets",
    "characters",
    "savedVoices",
    "loras",
    "presets",
    "promptBatches",
    "trainingDatasets",
    "personTracks",
    "timelines",
  ]),
);

const ADVANCED_ROUTE_DOMAINS = Object.freeze({
  Library: ["assets"],
  Queue: ["assets"],
  Models: ["models", "loras"],
  Settings: [],
  Stats: [],
  Logs: [],
  Licenses: [],
  Image: ["assets", "models", "loras", "presets", "promptBatches", "characters"],
  Video: ["assets", "models", "loras", "presets", "characters", "personTracks"],
  Audio: ["assets", "models", "savedVoices"],
  Characters: ["assets", "characters", "models", "loras", "presets", "trainingDatasets"],
  Document: ["assets", "models"],
  Train: [
    "assets",
    "characters",
    "models",
    "trainingTargets",
    "trainingPresets",
    "trainingDatasets",
  ],
  LibraryDataSets: [
    "assets",
    "characters",
    "models",
    "trainingTargets",
    "trainingPresets",
    "trainingDatasets",
  ],
  Poses: ["assets", "characters"],
  Keypoints: ["assets", "characters"],
  Presets: ["models", "loras", "presets"],
  Editor: ["assets", "characters", "models", "loras", "presets", "timelines"],
  ImageEditor: ["assets", "characters", "models", "loras"],
  Setup: ["models"],
});

// Default Library startup remains assets-first so its first useful paint is not
// blocked by secondary catalogs. Once assets settle, the selected detail panel
// needs these domains for character linking, image VQA, and audio model labels.
const DEFERRED_ROUTE_DOMAINS = Object.freeze({
  Library: ["characters", "models"],
});

const SIMPLE_ROUTE_DOMAINS = Object.freeze({
  image: ["assets", "models", "loras"],
  video: ["assets", "models"],
  audio: ["assets", "models"],
  assets: ["assets"],
  models: ["models", "loras"],
  queue: [],
  settings: [],
  licenses: [],
});

export function hydrationDomainsForRoute(route, { simple = false, deferred = false } = {}) {
  if (deferred) {
    return simple ? [] : DEFERRED_ROUTE_DOMAINS[route] ?? [];
  }
  const source = simple ? SIMPLE_ROUTE_DOMAINS : ADVANCED_ROUTE_DOMAINS;
  return source[route] ?? [];
}

export function hydrationLedgerKey(domain, projectId = null) {
  return PROJECT_HYDRATION_DOMAINS.has(domain)
    ? `${domain}:project:${projectId ?? "none"}`
    : `${domain}:global`;
}

export function isProjectHydrationDomain(domain) {
  return PROJECT_HYDRATION_DOMAINS.has(domain);
}
