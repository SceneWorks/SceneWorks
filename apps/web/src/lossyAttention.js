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
  // `recorded` is a replayed recipe's full `lossyAttention` object: every field the control does not
  // expose (Sol `tau`, sinks, tail, query chunking, INT8 K centering, a second kernel) rides through
  // it untouched, and only the fields the control shows are overlaid on it.
  return {
    enabled: false,
    kernel: "int8",
    pvPrecision: "value_dtype",
    topkRatio: DEFAULT_SOL_TOPK_RATIO,
    recorded: null,
  };
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

const SOL_RATIO_ERROR = "Sol-Attn top-k ratio must be a number above 0 and at most 1.";

function cloneKernelConfig(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? JSON.parse(JSON.stringify(value)) : null;
}

/** The kernels the studio state selects: one, or both when replaying a recipe that used both. */
export function selectedLossyKernels(state) {
  return state.kernel === "both" ? ["int8", "sol"] : [state.kernel];
}

/**
 * The ONE effective selection, shared by the control's display and the request resolver, so what is
 * shown is exactly what is sent (Codex P2 on #3538). A replayed recipe (`recorded`) keeps its own
 * kernels — both when it used both; a kernel this model cannot take is the resolver's error. A
 * preference (no `recorded`) is always a single kernel this model serves: its own when served,
 * else the first served (INT8, the recommended one); a stray `"both"` is never a preference.
 * Returns `{ kernels, kernel }`, `kernel` being the control's value (`"both"` for two).
 */
export function effectiveLossyKernels(state, served) {
  const recorded = state.recorded && typeof state.recorded === "object" ? state.recorded : null;
  if (recorded) {
    const kernels = selectedLossyKernels(state);
    return { kernels, kernel: kernels.length > 1 ? "both" : kernels[0] };
  }
  const kernel = LOSSY_KERNELS.includes(state.kernel) && served.includes(state.kernel) ? state.kernel : served[0];
  return { kernels: kernel ? [kernel] : [], kernel };
}

/**
 * The persisted studio setting as a PREFERENCE: no replayed `recorded` block, and a single kernel
 * (`"both"` exists only while a two-kernel recipe is being replayed). Applied on load, so an older
 * persisted value — `kernel: "both"`, a stale `recorded` — cannot act as a replay or send a kernel
 * the control does not show.
 */
export function lossyAttentionPreference(saved) {
  const defaults = lossyAttentionDefaults();
  const state = { ...defaults, ...(saved && typeof saved === "object" ? saved : {}) };
  return {
    enabled: Boolean(state.enabled),
    kernel: LOSSY_KERNELS.includes(state.kernel) ? state.kernel : defaults.kernel,
    pvPrecision: state.pvPrecision === "int8" ? "int8" : "value_dtype",
    topkRatio: validSolTopkRatio(state.topkRatio) ?? defaults.topkRatio,
    recorded: null,
  };
}

/**
 * Why a replayed recipe's lossy attention cannot run here: an MLX (non-CUDA) backend, or a model
 * (or no model) whose attention takes neither kernel. Turning the control off runs it exactly.
 */
export function lossyAttentionReplayBlockedError(backend) {
  return backend === "candle"
    ? "This recipe used lossy attention, which this model's attention cannot take; turn lossy attention off to run it on the exact path."
    : "This recipe used lossy attention, which runs only on CUDA (candle) workers; turn lossy attention off to run it on the exact path.";
}

/** Whether a replayed recipe's lossy attention is enabled but cannot be honored here. */
export function lossyAttentionReplayBlocked(backend, model, options) {
  const state = { ...lossyAttentionDefaults(), ...(options ?? {}) };
  return Boolean(state.enabled && state.recorded && !lossyAttentionAvailable(backend, model));
}

/**
 * Resolve the studio state to the outgoing `advanced.lossyAttention` wire. Returns
 * `{ block, error }`: `block` is null when the control is off or unavailable (an error when it is a
 * replayed recipe's configuration that cannot run here, silence for a sticky preference). Without a replayed
 * recipe, a kernel this model does not serve falls back to one it does (a sticky choice from another
 * model never leaks onto one that would refuse it). A replayed recipe's configuration is kept
 * whole — only the fields the control exposes are overlaid — and a recorded kernel this model
 * cannot take is an error rather than a silent substitution.
 */
export function resolveLossyAttention({ backend, model, options }) {
  const state = { ...lossyAttentionDefaults(), ...(options ?? {}) };
  if (!state.enabled) return { block: null, error: null };
  const recorded = state.recorded && typeof state.recorded === "object" ? state.recorded : null;
  if (!lossyAttentionAvailable(backend, model)) {
    // A sticky preference (no replayed recipe) is silently not offered here. A replayed recipe
    // that used lossy attention cannot be honored: block rather than run it on the exact path.
    return { block: null, error: recorded ? lossyAttentionReplayBlockedError(backend) : null };
  }
  const served = servedLossyKernels(model);
  const { kernels } = effectiveLossyKernels(state, served);
  const unserved = kernels.filter((kernel) => !served.includes(kernel));
  if (unserved.length) {
    return {
      block: null,
      error: `This model cannot take the recipe's ${unserved.map((kernel) => (kernel === "int8" ? "INT8" : "Sol-Attn")).join(" and ")} lossy attention; pick a kernel it serves.`,
    };
  }
  const block = {};
  for (const kernel of kernels) block[kernel] = cloneKernelConfig(recorded?.[kernel]) ?? {};
  if (block.int8 && (state.pvPrecision !== "value_dtype" || "pvPrecision" in block.int8)) {
    block.int8.pvPrecision = state.pvPrecision === "int8" ? "int8" : "value_dtype";
  }
  if (block.sol) {
    const edited = state.topkRatio !== null && state.topkRatio !== undefined && state.topkRatio !== "";
    if (edited) {
      const topkRatio = validSolTopkRatio(state.topkRatio);
      if (topkRatio === null) return { block: null, error: SOL_RATIO_ERROR };
      block.sol.topkRatio = topkRatio;
      delete block.sol.tau;
    } else if (!("tau" in block.sol) && !("topkRatio" in block.sol)) {
      return { block: null, error: SOL_RATIO_ERROR };
    }
  }
  return { block, error: null };
}

/**
 * Studio state restored from a recipe's recorded `lossyAttention` (replay); defaults when absent.
 * The whole recorded object is kept in `recorded`, so replaying with nothing changed re-sends it.
 */
export function lossyAttentionFromRecipe(raw) {
  const defaults = lossyAttentionDefaults();
  if (!raw || typeof raw !== "object") return defaults;
  const int8 = cloneKernelConfig(raw.int8);
  const sol = cloneKernelConfig(raw.sol);
  if (!int8 && !sol) return defaults;
  const recorded = { ...(int8 ? { int8 } : {}), ...(sol ? { sol } : {}) };
  return {
    ...defaults,
    enabled: true,
    kernel: int8 && sol ? "both" : int8 ? "int8" : "sol",
    pvPrecision: int8?.pvPrecision === "int8" ? "int8" : "value_dtype",
    // A tau-selected recipe has no ratio: `null` keeps its `tau` unless the user types one.
    topkRatio: sol ? (validSolTopkRatio(sol.topkRatio) ?? ("tau" in sol ? null : DEFAULT_SOL_TOPK_RATIO)) : DEFAULT_SOL_TOPK_RATIO,
    recorded,
  };
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
