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
export function h3ImageSamplerFromRecipe(recorded) {
  const defaults = h3ImageSamplerDefaults();
  if (recorded?.sampler !== "lcm") return { ...defaults, sampler: "euler" };
  const sNoise = recorded.sNoise ?? defaults.sNoise;
  return {
    sampler: "lcm",
    sNoise,
    sNoiseEnd: recorded.sNoiseEnd ?? (recorded.sNoise == null ? defaults.sNoiseEnd : sNoise),
    noiseClipStd: recorded.noiseClipStd ?? defaults.noiseClipStd,
  };
}
