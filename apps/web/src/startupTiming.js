// Secret-safe, bounded startup timing. These names describe lifecycle phases
// only; no project, asset, URL, prompt, token, or path value is ever attached.
const PREFIX = "sceneworks.";

export const STARTUP_MARK_ORDER = Object.freeze([
  "bootstrap-start",
  "access-resolved",
  "media-authorization-settled",
  "projects-committed",
  "active-project-selected",
  "asset-request-start",
  "asset-request-settled",
  "assets-ready-render",
]);

const startupOnce = new Set();
let lastStartupIndex = -1;
if (globalThis.__SCENEWORKS_BOOTSTRAP_TIMING_STARTED__ === true) {
  startupOnce.add("bootstrap-start");
  lastStartupIndex = STARTUP_MARK_ORDER.indexOf("bootstrap-start");
}
const STARTUP_MEASURE_NAMES = [
  ...STARTUP_MARK_ORDER.slice(1).map(
    (name, index) => `${STARTUP_MARK_ORDER[index]}-to-${name}`,
  ),
  "bootstrap-to-assets-ready-render",
  "asset-request",
];

function timingApi() {
  const api = globalThis.performance;
  return api && typeof api.mark === "function" ? api : null;
}

function boundedMark(name) {
  const api = timingApi();
  if (!api) return false;
  const fullName = `${PREFIX}${name}`;
  api.clearMarks?.(fullName);
  api.mark(fullName);
  return true;
}

function boundedMeasure(name, start, end) {
  const api = timingApi();
  if (!api || typeof api.measure !== "function") return;
  const fullName = `${PREFIX}${name}`;
  api.clearMeasures?.(fullName);
  try {
    if (typeof start === "number" && typeof end === "number") {
      api.measure(fullName, { start, end });
    } else {
      api.measure(fullName, start, end);
    }
  } catch {
    // Timing must never affect startup. Older WebViews can expose `mark` while
    // lacking modern `measure` overloads.
  }
}

export function recordStartupMark(name) {
  if (!STARTUP_MARK_ORDER.includes(name) || startupOnce.has(name)) return false;
  const index = STARTUP_MARK_ORDER.indexOf(name);
  if (index < lastStartupIndex) return false;
  if (!boundedMark(name)) return false;
  startupOnce.add(name);
  lastStartupIndex = index;

  const previous = STARTUP_MARK_ORDER[index - 1];
  if (previous && startupOnce.has(previous)) {
    boundedMeasure(
      `${previous}-to-${name}`,
      `${PREFIX}${previous}`,
      `${PREFIX}${name}`,
    );
  }
  if (name === "assets-ready-render") {
    boundedMeasure(
      "asset-request-settled-to-assets-ready-render",
      `${PREFIX}asset-request-settled`,
      `${PREFIX}assets-ready-render`,
    );
    if (startupOnce.has("bootstrap-start")) {
      boundedMeasure(
        "bootstrap-to-assets-ready-render",
        `${PREFIX}bootstrap-start`,
        `${PREFIX}assets-ready-render`,
      );
    }
  }
  return true;
}

export function beginAssetRequest() {
  const api = timingApi();
  const startedAt = typeof api?.now === "function" ? api.now() : Date.now();
  boundedMark("asset-request-start");
  if (startupOnce.has("active-project-selected")) {
    boundedMeasure(
      "active-project-selected-to-asset-request-start",
      `${PREFIX}active-project-selected`,
      `${PREFIX}asset-request-start`,
    );
  }
  return startedAt;
}

export function settleAssetRequest(startedAt) {
  const api = timingApi();
  const settledAt = typeof api?.now === "function" ? api.now() : Date.now();
  boundedMark("asset-request-settled");
  boundedMeasure("asset-request", startedAt, settledAt);
}

// Test-only reset for this module's one-shot bookkeeping. It intentionally
// clears only SceneWorks-owned marks. Tests provide an isolated Performance API,
// so clearing measures cannot affect a real browser navigation.
export function resetStartupTimingForTests() {
  startupOnce.clear();
  lastStartupIndex = -1;
  const api = timingApi();
  if (!api) return;
  for (const name of STARTUP_MARK_ORDER) {
    api.clearMarks?.(`${PREFIX}${name}`);
  }
  for (const name of STARTUP_MEASURE_NAMES) {
    api.clearMeasures?.(`${PREFIX}${name}`);
  }
}
