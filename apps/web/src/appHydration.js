// Route-owned catalog hydration (sc-14783).
//
// Domains are deliberately explicit instead of inferred from component imports. That
// keeps startup/network behavior reviewable and gives tests a deterministic request
// contract. Global domains survive project switches; project domains are keyed by the
// active project and are reset/reloaded only when a consumer route needs them.
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
  VectorStudio: ["assets", "models"],
  Setup: ["models"],
});

// Default Library startup remains assets-first so its first useful paint is not
// blocked by secondary catalogs. Once assets settle, the selected detail panel
// needs these domains for character linking, image VQA, and audio model labels.
const DEFERRED_ROUTE_DOMAINS = Object.freeze({
  Library: ["characters", "models"],
});

const SIMPLE_ROUTE_DOMAINS = Object.freeze({
  image: ["assets", "models", "loras", "characters"],
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

// Which slice of the project's asset catalog a route needs. The Asset Library pages render only
// Library assets (`isLibraryAsset`), so they read the server's `scope=library` listing: in a
// project dominated by Character Studio output that is a fraction of the rows and bytes of the
// full catalog. Every other route keeps the full catalog, because its pickers, previews and
// reference checks can name a Character Studio image (an AssetPicker's Character tab, a restored
// Image Studio reference, a timeline item, a dataset's character images).
//
// The shell never narrows a catalog it already holds in full; see `useShellHydration`.
const LIBRARY_SCOPED_ASSET_ROUTES = Object.freeze({
  advanced: new Set(["Library"]),
  simple: new Set(["assets"]),
});

export function assetCatalogScopeForRoute(route, { simple = false } = {}) {
  const routes = simple ? LIBRARY_SCOPED_ASSET_ROUTES.simple : LIBRARY_SCOPED_ASSET_ROUTES.advanced;
  return routes.has(route) ? "library" : "all";
}

export function hydrationLedgerKey(domain, projectId = null) {
  return PROJECT_HYDRATION_DOMAINS.has(domain)
    ? `${domain}:project:${projectId ?? "none"}`
    : `${domain}:global`;
}

export function isProjectHydrationDomain(domain) {
  return PROJECT_HYDRATION_DOMAINS.has(domain);
}
