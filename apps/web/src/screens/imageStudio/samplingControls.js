// Image Studio-only sampling controls (sc-25681). Kept out of the shared `samplerOptions` chunk:
// only Image Studio renders them, so they ship inside its own chunk.
import { guidanceIsOff } from "../../samplerOptions.js";

// sc-25681: the caption-overflow menu labels (`image.captionOverflowPolicies`; "default" = the
// checkpoint's own policy).
export const CAPTION_OVERFLOW_LABELS = {
  default: "Default",
  warn: "Truncate + warn",
  error: "Refuse",
  silent: "Truncate",
};

// The CFG interval to send (sc-25681) from the "start, end" input ("" = 0, 1). `interval` is null
// when the model offers none (`image.supportsCfgInterval`), the full [0, 1] is selected, or CFG is
// off (the engine refuses an interval there); `error` when 0 ≤ start < end ≤ 1 fails.
export function resolveCfgInterval({ model, text, guidanceOverride }) {
  const [lo, hi] = (text || "0,1").split(",").map(Number);
  if (!model?.image?.supportsCfgInterval) return { interval: null, error: null };
  if (!(lo >= 0 && lo < hi && hi <= 1)) return { interval: null, error: "Need 0 ≤ start < end ≤ 1." };
  return { interval: guidanceIsOff(model, guidanceOverride) || hi - lo === 1 ? null : [lo, hi], error: null };
}

// Prompts per native prompt-batch job (sc-25681, `image.maxPromptBatchImages`): 1 = one job each.
export const promptBatchSize = (model, count) =>
  Math.max(1, Math.floor((model?.image?.maxPromptBatchImages || 1) / Math.max(1, count)));

// SW-7106: MiniMax-H3-Image's sampler (`image.supportsH3ImageSampler`). LCM is the default (decided
// 2026-11-12): ComfyUI's stochastic re-noise step, whose noise scale ramps from `sNoise` (step 0)
// to `sNoiseEnd` (last step), with optional noise clipping. Euler is the released deterministic
// step and stays selectable.
export const H3_IMAGE_SAMPLER_LABELS = {
  euler: "Euler (official)",
  lcm: "LCM (stochastic re-noise)",
};

export const H3_IMAGE_LCM_FIELDS = [
  { key: "sNoise", label: "Noise scale start", min: 0.01, max: 2, step: 0.01 },
  { key: "sNoiseEnd", label: "Noise scale end", min: 0, max: 2, step: 0.01 },
  { key: "noiseClipStd", label: "Noise clip (std, 0 = off)", min: 0, max: 4, step: 0.1 },
];

export const h3ImageSamplerDefaults = () => ({ sampler: "lcm", sNoise: 1.1, sNoiseEnd: 0.9, noiseClipStd: 0 });

export const supportsH3ImageSampler = (model) => Boolean(model?.image?.supportsH3ImageSampler);

// The `advanced.h3ImageSampler` block to send: null for a model without the control. Both samplers
// are sent explicitly (the job record shows what ran, and a selected recipe preset's server-side
// expansion can never replace the user's choice). Euler is `{sampler: "euler"}`. `error` names the first LCM field out of range.
// The bounds compare in f32: the worker records each scale as an f32, so a replayed 0.01 comes back
// as 0.009999999776482582 and must still sit inside [0.01, …] — while a truly out-of-range value,
// which differs at f32 precision, is still refused.
const withinF32Bounds = (value, min, max) =>
  Math.fround(value) >= Math.fround(min) && Math.fround(value) <= Math.fround(max);

export function resolveH3ImageSampler({ model, options }) {
  if (!supportsH3ImageSampler(model)) return { block: null, error: null };
  if (options?.sampler === "euler") return { block: { sampler: "euler" }, error: null };
  const block = { sampler: "lcm" };
  for (const { key, label, min, max } of H3_IMAGE_LCM_FIELDS) {
    const raw = options[key];
    const value = raw === "" || raw == null ? Number.NaN : Number(raw);
    if (!(Number.isFinite(value) && withinF32Bounds(value, min, max))) {
      return { block: null, error: `${label} must be between ${min} and ${max}.` };
    }
    block[key] = value;
  }
  return { block, error: null };
}

// A recorded `advanced.h3ImageSampler` (recipe replay, saved preset) as the control state.
// The worker records the EFFECTIVE sampler on every new asset, so a complete block restores as-is.
// A partial block (an older or hand-written recipe) fills each missing field exactly as the worker's
// `requested_h3_image_sampler` does, so replay runs what the original run did: sNoise 1.1; sNoiseEnd
// 0.9 when sNoise is also omitted, else equal to the given sNoise (ComfyUI's constant scale);
// noiseClipStd 0. Only a recipe with NO sampler block at all predates this feature recording it:
// it ran Euler, so it replays as Euler, and the LCM fields keep their defaults (1.1 / 0.9 / 0) for
// when the control is switched to LCM.
// Each restored number is shown as the shortest decimal with the same f32 value: the worker
// records the scales as f32, so a UI 0.01 comes back as 0.009999999776482582, which the number
// input's native `min="0.01"` would flag as out of range and the form would refuse to submit.
export function h3ImageSamplerFromRecipe(recorded) {
  const defaults = h3ImageSamplerDefaults();
  if (recorded?.sampler !== "lcm") return { ...defaults, sampler: "euler" };
  const sNoise = f32Decimal(recorded.sNoise ?? defaults.sNoise);
  return {
    sampler: "lcm",
    sNoise,
    sNoiseEnd: f32Decimal(recorded.sNoiseEnd ?? (recorded.sNoise == null ? defaults.sNoiseEnd : sNoise)),
    noiseClipStd: f32Decimal(recorded.noiseClipStd ?? defaults.noiseClipStd),
  };
}

// The shortest decimal whose f32 value equals `value`'s; anything that is not a finite number is
// returned unchanged for the resolver to refuse.
function f32Decimal(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return value;
  const target = Math.fround(value);
  for (let digits = 1; digits <= 17; digits += 1) {
    const candidate = Number(value.toPrecision(digits));
    if (Math.fround(candidate) === target) return candidate;
  }
  return value;
}

// SW-7181: Perturbed-Attention Guidance (PAG), capability-driven from `model.pag` projected by
// `ModelSpec::project()`. Scale 0 or blank omits `advanced.pag` (strict no-op).
export const supportsPag = (model) =>
  Boolean(model?.pag && typeof model.pag === "object" && Number.isInteger(model.pag.numBlocks) && model.pag.numBlocks > 0);

export const pagDefaults = () => ({ scale: "", layers: "", start: "", stop: "", adaptiveScale: "" });

const isBlankText = (value) =>
  value === "" || value == null || (typeof value === "string" && value.trim() === "");

function parsePagLayers(rawLayers, numBlocks) {
  if (isBlankText(rawLayers)) return { layers: null, error: null };
  let items;
  if (Array.isArray(rawLayers)) {
    if (rawLayers.length === 0) return { layers: null, error: null };
    items = rawLayers.map((item) => String(item));
  } else if (typeof rawLayers === "string") {
    items = rawLayers.trim().split(/[\s,]+/).filter(Boolean);
    if (items.length === 0) {
      return { layers: null, error: `PAG blocks must be integers between 0 and ${numBlocks - 1}.` };
    }
  } else {
    return { layers: null, error: `PAG blocks must be integers between 0 and ${numBlocks - 1}.` };
  }
  const seen = new Set();
  const parsed = [];
  for (const token of items) {
    if (!/^\d+$/.test(token)) {
      return { layers: null, error: `PAG blocks must be integers between 0 and ${numBlocks - 1}.` };
    }
    const index = Number(token);
    if (!Number.isSafeInteger(index) || index < 0 || index >= numBlocks) {
      return { layers: null, error: `PAG block index ${token} must be between 0 and ${numBlocks - 1}.` };
    }
    if (seen.has(index)) {
      return { layers: null, error: `PAG block index ${index} is duplicated.` };
    }
    seen.add(index);
    parsed.push(index);
  }
  return { layers: parsed, error: null };
}

function parsePagUnit(raw, label) {
  if (isBlankText(raw)) return { value: null, error: null };
  const value = Number(raw);
  if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value)) || !withinF32Bounds(value, 0, 1)) {
    return { value: null, error: `${label} must be between 0 and 1.` };
  }
  return { value, error: null };
}

export function resolvePag({ model, options }) {
  if (!supportsPag(model)) return { block: null, error: null };
  const rawScale = options?.scale;
  if (isBlankText(rawScale)) return { block: null, error: null };
  const scale = Number(rawScale);
  const maxScale = Number(model.pag.maxScale);
  if (!Number.isFinite(scale) || !Number.isFinite(Math.fround(scale)) || !withinF32Bounds(scale, 0, maxScale)) {
    return { block: null, error: `PAG scale must be between 0 and ${model.pag.maxScale}.` };
  }
  if (scale === 0) return { block: null, error: null };

  const { layers, error: layersError } = parsePagLayers(options?.layers, model.pag.numBlocks);
  if (layersError) return { block: null, error: layersError };

  const { value: start, error: startError } = parsePagUnit(options?.start, "PAG window start");
  if (startError) return { block: null, error: startError };

  const { value: stop, error: stopError } = parsePagUnit(options?.stop, "PAG window stop");
  if (stopError) return { block: null, error: stopError };

  const effStart = start ?? Number(model.pag.defaultStart);
  const effStop = stop ?? Number(model.pag.defaultStop);
  if (Math.fround(effStart) > Math.fround(effStop)) {
    return { block: null, error: `PAG window start (${effStart}) must be ≤ stop (${effStop}).` };
  }

  let adaptiveScale = null;
  const rawAdaptive = options?.adaptiveScale;
  if (!isBlankText(rawAdaptive)) {
    const value = Number(rawAdaptive);
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value)) || Math.fround(value) < 0) {
      return { block: null, error: "PAG adaptive scale must be 0 or greater." };
    }
    if (value > 0) adaptiveScale = value;
  }

  return {
    block: {
      scale,
      ...(layers ? { layers } : {}),
      ...(start != null ? { start } : {}),
      ...(stop != null ? { stop } : {}),
      ...(adaptiveScale != null ? { adaptiveScale } : {}),
    },
    error: null,
  };
}

// Rehydrate UI string state from a recorded `rawAdapterSettings.pag` or preset `defaults.pag`,
// ignoring any read-only execution receipt (`pagExecutionReport`).
export function pagFromRecipe(recorded) {
  const defaults = pagDefaults();
  if (!recorded || typeof recorded !== "object") return defaults;
  const rawScale = recorded.scale;
  if (isBlankText(rawScale)) return defaults;
  const numericScale = Number(rawScale);
  if (Number.isFinite(numericScale) && numericScale === 0) return defaults;

  const formatOptionalNumber = (value) => {
    if (isBlankText(value)) return "";
    if (typeof value === "string") return value;
    return Number.isFinite(value) ? String(f32Decimal(value)) : String(value);
  };

  const rawAdaptive = recorded.adaptiveScale ?? recorded.adaptive_scale;
  const numericAdaptive = Number(rawAdaptive);
  const adaptiveScale =
    isBlankText(rawAdaptive) || (Number.isFinite(numericAdaptive) && numericAdaptive === 0)
      ? ""
      : formatOptionalNumber(rawAdaptive);

  return {
    scale: formatOptionalNumber(rawScale),
    layers: Array.isArray(recorded.layers)
      ? recorded.layers.join(", ")
      : typeof recorded.layers === "string"
        ? recorded.layers
        : "",
    start: formatOptionalNumber(recorded.start),
    stop: formatOptionalNumber(recorded.stop),
    adaptiveScale,
  };
}
