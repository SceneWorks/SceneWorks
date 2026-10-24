// Shared UI preflight; background replay loads it only for an actual MOSS request.
import { mossAdvancedWire, mossTensorReferences, mossTensorMatches } from "./mossImageAdvanced.js";

export function mossPositivePromptIssue(options, { prompt = "", prompts = [], effectivePrompt = prompt } = {}) {
  if (!options?.positive) return "";
  if (prompt !== "" || prompts.length) return "Supplied positive embeddings replace both prompt fields. Clear the prompt and prompt batch to use them.";
  if (effectivePrompt != null && effectivePrompt !== "") return "Supplied positive embeddings replace composed text. Clear the selected Style or prompt preset before generating.";
  return "";
}

export function mossAdvancedIssue(options, { assets, upscaleEnabled = false, prompt = "", prompts = [], effectivePrompt = prompt, negativePrompt = "", count = 1, width, height, context = 4096 } = {}) {
  try {
    const value = mossAdvancedWire(options);
    if (value.output === "latents" && upscaleEnabled) return "MOSS latent output cannot use inline upscale. Disable upscale or select images output.";
    const positiveIssue = mossPositivePromptIssue(value, { prompt, prompts, effectivePrompt });
    if (positiveIssue) return positiveIssue;
    if (prompts.length && (prompts.length > 8 || prompts.some((text) => typeof text !== "string" || !text || [...text].length > 4000) || prompt !== prompts[0])) return "Native prompt batches require 1–8 nonempty prompts; the main prompt must equal the first row.";
    if ((value.negative || value.negativePrompts) && negativePrompt) return "Clear the broadcast negative prompt before supplying negative tensors or per-prompt negatives.";
    const batch = value.positive?.values.metadata?.shape?.[0] ?? Math.max(prompts.length, 1);
    if (!Number.isInteger(batch * Number(count)) || batch * Number(count) < 1 || batch * Number(count) > 8) return "MOSS conditioning batch × count must be in 1–8.";
    if (value.negativePrompts && value.negativePrompts.length !== batch) return "Per-prompt negative strings must match the conditioning batch.";
    for (const [role, ref] of mossTensorReferences(value)) {
      if (assets && !assets.some((asset) => asset.id === ref.assetId && mossTensorMatches(asset, ref))) return `The exact ${role} tensor is missing or changed. Reimport its original NPY bytes before replay.`;
      const { dtype, shape = [], byteLength } = ref.metadata ?? {};
      if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength > 512 * 1024 * 1024) return `The ${role} tensor exceeds the artifact byte budget or lacks measured metadata.`;
      if (role.endsWith(".values") && (dtype !== "float32" || shape.length !== 3 || shape[0] !== batch || shape[1] < 1 || shape[1] > context || shape[2] !== 4096)) return `${role} must be FP32 [B,S,4096], with S within the context limit.`;
      if (role.endsWith(".mask")) {
        const expected = value[role.split(".")[0]].values.metadata?.shape?.slice(0, 2);
        if (dtype !== "uint8" || JSON.stringify(shape) !== JSON.stringify(expected)) return `${role} must be a binary U8 [B,S] mask matching its embeddings.`;
      }
      if (role === "noise" && (dtype !== "float32" || JSON.stringify(shape) !== JSON.stringify([batch * Number(count), 128, Number(height) / 16, Number(width) / 16]))) return "Initial noise must be exact FP32 [B×count,128,H/16,W/16] for the selected output dimensions.";
    }
    return "";
  } catch (error) { return error.message; }
}
