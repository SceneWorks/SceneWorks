// SW-6132 (feature SW-F287): the per-job opt-in for the two lossy CUDA attention kernels —
// INT8 QK attention (SageAttention-style) and Sol-Attn (sparse, block-routed). Neither is ever a
// default. The studio offers the control only on the candle (CUDA) backend and only for a model
// whose catalog entry advertises `lossyAttention` (derived server-side from inference's
// SHARED_ATTENTION_ROUTING); the API refuses anything else with a typed error.
//
// Wire (`advanced.lossyAttention`), mapping onto inference's `Int8AttentionConfig` / `SolConfig`:
//   { int8?: { pvPrecision?: "value_dtype" | "int8", kCentering?: "off" | "always" | "adaptive" },
//     sol?: { topkRatio | tau, sinkBlocks?, sinkQ?, tail?, queryChunkBlocks? } }
// The studio sends only what it exposes (kernel, INT8 P·V precision, Sol top-k ratio); the rest
// keeps inference's defaults.

export const LOSSY_KERNELS = ["int8", "sol"];

// INT8 is the recommended kernel: near-lossless on every measured model (Krea 2 Turbo 41–45 dB,
// SVD-XT 37 dB, SeedVR2 51 dB vs the exact path). Sol-Attn at 0.5 is a visibly different sample.
export const DEFAULT_SOL_TOPK_RATIO = 0.5;

export function lossyAttentionDefaults() {
  return { enabled: false, kernel: "int8", pvPrecision: "value_dtype", topkRatio: DEFAULT_SOL_TOPK_RATIO };
}

/** The model's `lossyAttention` capability (`{ int8: {served, reason?}, sol: {...} }`) or null. */
export function lossyAttentionCapability(model) {
  const capability = model?.lossyAttention;
  if (!capability || typeof capability !== "object") return null;
  return capability;
}

/** Kernels the model's attention can take, in recommendation order. */
export function servedLossyKernels(model) {
  const capability = lossyAttentionCapability(model);
  if (!capability) return [];
  return LOSSY_KERNELS.filter((kernel) => capability[kernel]?.served === true);
}

/** Whether the studio offers the control: the candle backend and at least one served kernel. */
export function lossyAttentionAvailable(backend, model) {
  return backend === "candle" && servedLossyKernels(model).length > 0;
}

/** A top-k ratio the API accepts — finite, in (0, 1] — else null. */
export function validSolTopkRatio(value) {
  const ratio = Number(value);
  return Number.isFinite(ratio) && ratio > 0 && ratio <= 1 ? ratio : null;
}

/**
 * Resolve the studio state to the outgoing `advanced.lossyAttention` wire. Returns
 * `{ block, error }`: `block` is null when the control is off, unavailable, or the chosen kernel
 * is not served by this model (so a sticky choice from another model never leaks onto one that
 * would refuse it); `error` names an out-of-range Sol top-k ratio.
 */
export function resolveLossyAttention({ backend, model, options }) {
  const state = { ...lossyAttentionDefaults(), ...(options ?? {}) };
  if (!state.enabled || !lossyAttentionAvailable(backend, model)) return { block: null, error: null };
  const served = servedLossyKernels(model);
  const kernel = served.includes(state.kernel) ? state.kernel : served[0];
  if (kernel === "int8") {
    const int8 = state.pvPrecision === "int8" ? { pvPrecision: "int8" } : {};
    return { block: { int8 }, error: null };
  }
  const topkRatio = validSolTopkRatio(state.topkRatio);
  if (topkRatio === null) {
    return { block: null, error: "Sol-Attn top-k ratio must be a number above 0 and at most 1." };
  }
  return { block: { sol: { topkRatio } }, error: null };
}

/** Studio state restored from a recipe's recorded `lossyAttention` (replay); defaults when absent. */
export function lossyAttentionFromRecipe(raw) {
  const defaults = lossyAttentionDefaults();
  if (!raw || typeof raw !== "object") return defaults;
  if (raw.int8 && typeof raw.int8 === "object") {
    return {
      ...defaults,
      enabled: true,
      kernel: "int8",
      pvPrecision: raw.int8.pvPrecision === "int8" ? "int8" : "value_dtype",
    };
  }
  if (raw.sol && typeof raw.sol === "object") {
    return {
      ...defaults,
      enabled: true,
      kernel: "sol",
      topkRatio: validSolTopkRatio(raw.sol.topkRatio) ?? DEFAULT_SOL_TOPK_RATIO,
    };
  }
  return defaults;
}

/** One-line summary of a recipe's `attentionExecution` receipt (did the kernel actually run). */
export function attentionExecutionSummary(receipt) {
  if (!receipt || typeof receipt !== "object") return null;
  const int8 = Number(receipt.int8) || 0;
  const sol = Number(receipt.sol) || 0;
  const total = Number(receipt.total) || 0;
  const parts = [];
  if (int8 > 0) parts.push(`INT8 ${int8}/${total} attention calls`);
  if (sol > 0) parts.push(`Sol-Attn ${sol}/${total} attention calls`);
  if (!parts.length) parts.push(`no lossy kernel ran (${total} attention calls)`);
  return parts.join(", ");
}
