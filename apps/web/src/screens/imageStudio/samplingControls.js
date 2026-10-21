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
