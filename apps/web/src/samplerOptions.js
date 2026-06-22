// Shared labels + helpers for the configurable sampler/scheduler controls. The
// curated vocabulary (epic 7114) matches the native mlx-gen / candle-gen gen-core
// Solver / Scheduler registries exactly — the names sent in the job's `advanced`
// block ARE the engine's sampler/scheduler ids. The studios source their dropdown
// contents from each model's per-backend manifest `limits.samplers` /
// `limits.schedulers` arrays (base `limits` overridden by `mlx.limits` /
// `candle.limits` for the active backend) so invalid combos are never selectable.

export const SAMPLER_LABELS = {
  default: "Model default",
  euler: "Euler",
  euler_ancestral: "Euler ancestral",
  heun: "Heun (2nd-order)",
  dpmpp_2m: "DPM++ (2M)",
  dpmpp_sde: "DPM++ SDE",
  uni_pc: "UniPC",
  lcm: "LCM",
  ddim: "DDIM",
};

export const SCHEDULER_LABELS = {
  default: "Model default",
  normal: "Normal",
  simple: "Simple (uniform)",
  karras: "Karras",
  exponential: "Exponential",
  sgm_uniform: "SGM uniform",
  beta: "Beta",
  ddim_uniform: "DDIM uniform",
};

const SAMPLER_ORDER = [
  "default",
  "euler",
  "euler_ancestral",
  "heun",
  "dpmpp_2m",
  "dpmpp_sde",
  "uni_pc",
  "lcm",
  "ddim",
];
const SCHEDULER_ORDER = [
  "default",
  "normal",
  "simple",
  "karras",
  "exponential",
  "sgm_uniform",
  "beta",
  "ddim_uniform",
];

function uniqueOrdered(values, order) {
  const seen = new Set();
  const result = [];
  for (const key of order) {
    if (values.includes(key) && !seen.has(key)) {
      seen.add(key);
      result.push(key);
    }
  }
  // Append any keys not in our canonical ordering (forward-compat).
  for (const value of values) {
    if (typeof value === "string" && !seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  }
  return result;
}

// The effective `limits` for the active backend: the per-backend `mlx.limits` /
// `candle.limits` override when present, else the base `limits` (epic 7114 P5
// per-model-per-backend gating). `backend` is the active worker's backend
// ("mlx" | "candle"); a null/unknown backend falls back to the base menu.
function effectiveLimits(model, backend) {
  const override = backend ? model?.[backend]?.limits : null;
  return override ?? model?.limits;
}

// Pull the menu out of a model manifest entry, falling back to default-only.
// When the menu has fewer than 2 entries, the studio hides the dropdown — the
// caller can use `samplerMenu.length > 1` to gate rendering.
export function samplerOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.samplers) ? limits.samplers : ["default"];
  return uniqueOrdered(values, SAMPLER_ORDER);
}

export function schedulerOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.schedulers) ? limits.schedulers : ["default"];
  return uniqueOrdered(values, SCHEDULER_ORDER);
}

// Per-model defaults (UI initial values). The worker never reads these — they
// only set the studio form's initial sampler/scheduler choice. Falls back to
// "default" when the manifest doesn't pin one.
export function samplerDefaultFromModel(model) {
  const value = model?.defaults?.sampler;
  return typeof value === "string" && value.length ? value : "default";
}

export function schedulerDefaultFromModel(model) {
  const value = model?.defaults?.scheduler;
  return typeof value === "string" && value.length ? value : "default";
}

export function schedulerShiftDefaultFromModel(model) {
  const value = Number(model?.defaults?.schedulerShift);
  return Number.isFinite(value) && value > 0 ? value : 3.0;
}

export function stepsDefaultFromModel(model) {
  const value = Number(model?.defaults?.steps);
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function guidanceDefaultFromModel(model) {
  const value = Number(model?.defaults?.guidanceScale);
  return Number.isFinite(value) ? value : null;
}
