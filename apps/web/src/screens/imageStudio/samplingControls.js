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
export function resolveH3ImageSampler({ model, options }) {
  if (!supportsH3ImageSampler(model)) return { block: null, error: null };
  if (options?.sampler === "euler") return { block: { sampler: "euler" }, error: null };
  const block = { sampler: "lcm" };
  for (const { key, label, min, max } of H3_IMAGE_LCM_FIELDS) {
    const raw = options[key];
    const value = raw === "" || raw == null ? Number.NaN : Number(raw);
    if (!(Number.isFinite(value) && value >= min && value <= max)) {
      return { block: null, error: `${label} must be between ${min} and ${max}.` };
    }
    block[key] = value;
  }
  return { block, error: null };
}

// A recorded `advanced.h3ImageSampler` (recipe replay, saved preset) as the control state. A job or
// preset recorded with no block ran before the LCM default existed, so it replays as Euler; the
// LCM fields keep their defaults so switching the control to LCM starts from 1.1 / 0.9 / 0.
export function h3ImageSamplerFromRecipe(recorded) {
  const defaults = h3ImageSamplerDefaults();
  if (recorded?.sampler !== "lcm") return { ...defaults, sampler: "euler" };
  return {
    ...defaults,
    sampler: "lcm",
    sNoise: recorded.sNoise ?? 1,
    sNoiseEnd: recorded.sNoiseEnd ?? recorded.sNoise ?? 1,
    noiseClipStd: recorded.noiseClipStd ?? 0,
  };
}
