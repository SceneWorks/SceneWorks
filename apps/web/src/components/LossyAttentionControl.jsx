import React from "react";
import {
  DEFAULT_SOL_TOPK_RATIO,
  lossyAttentionAvailable,
  lossyAttentionDefaults,
  effectiveLossyKernels,
  lossyAttentionReplayBlocked,
  servedLossyKernels,
} from "../lossyAttention.js";

// SW-6132 (feature SW-F287): the experimental, lossy attention kernels as a per-job opt-in.
// Collapsed by default and rendered only on the candle (CUDA) backend for a model whose attention
// can take at least one kernel; `backend` is the studio's `activeBackend` ("mlx" | "candle").
export const LOSSY_ATTENTION_TOOLTIP =
  "Experimental and lossy: replaces exact attention with an approximate CUDA kernel for this job. " +
  "INT8 QK attention is the recommended one (near-identical output); Sol-Attn sparse attention " +
  "skips blocks and changes the result. Calls a kernel can't take stay on the exact path; the " +
  "asset's recipe records how many calls each kernel actually ran.";

const KERNEL_LABELS = {
  int8: "INT8 QK attention (recommended)",
  sol: "Sol-Attn sparse attention",
  both: "INT8 + Sol-Attn (from the recipe)",
};

export function LossyAttentionControl({ backend, model, value, onChange, error }) {
  const state = { ...lossyAttentionDefaults(), ...(value ?? {}) };
  if (!lossyAttentionAvailable(backend, model)) {
    // A replayed recipe that used lossy attention stays visible here with its blocking error and
    // the switch to turn it off; anything else is not offered on this backend/model.
    if (!lossyAttentionReplayBlocked(backend, model, state)) return null;
    return (
      <details className="lossy-attention-control" open>
        <summary title={LOSSY_ATTENTION_TOOLTIP}>Experimental: lossy attention</summary>
        <label className="checkline" title={LOSSY_ATTENTION_TOOLTIP}>
          <input
            aria-label="Lossy attention (experimental)"
            checked
            onChange={(event) => onChange({ ...state, enabled: event.target.checked })}
            type="checkbox"
          />
          Lossy attention (experimental)
        </label>
        {error ? <span className="field-error">{error}</span> : null}
      </details>
    );
  }
  const served = servedLossyKernels(model);
  const recorded = state.recorded && typeof state.recorded === "object" ? state.recorded : null;
  // A replayed recipe keeps its own kernel choice (an unserved one surfaces as `error`); otherwise a
  // sticky choice from another model shows the first kernel this model serves.
  const replayedBoth = Boolean(recorded?.int8 && recorded?.sol);
  // The same effective selection the request resolver sends (`effectiveLossyKernels`).
  const { kernel, kernels } = effectiveLossyKernels(state, served);
  const update = (patch) => onChange({ ...state, kernel, ...patch });
  const options = [...served, ...(replayedBoth && served.length > 1 ? ["both"] : [])];
  const tau = recorded?.sol && "tau" in recorded.sol ? recorded.sol.tau : null;
  return (
    <details className="lossy-attention-control" open={state.enabled || undefined}>
      <summary title={LOSSY_ATTENTION_TOOLTIP}>Experimental: lossy attention</summary>
      <label className="checkline" title={LOSSY_ATTENTION_TOOLTIP}>
        <input
          aria-label="Lossy attention (experimental)"
          checked={state.enabled}
          onChange={(event) => update({ enabled: event.target.checked })}
          type="checkbox"
        />
        Lossy attention (experimental)
      </label>
      {state.enabled ? (
        <>
          <label>
            Kernel
            <select
              aria-label="Lossy attention kernel"
              onChange={(event) => update({ kernel: event.target.value })}
              value={kernel}
            >
              {(options.includes(kernel) ? options : [...options, kernel]).map((key) => (
                <option key={key} value={key}>
                  {KERNEL_LABELS[key] ?? key}
                </option>
              ))}
            </select>
          </label>
          {kernels.includes("int8") ? (
            <label>
              P·V precision
              <select
                aria-label="INT8 attention P·V precision"
                onChange={(event) => update({ pvPrecision: event.target.value })}
                value={state.pvPrecision}
              >
                <option value="value_dtype">16-bit P·V (default, more accurate)</option>
                <option value="int8">INT8 P·V (faster, more lossy)</option>
              </select>
            </label>
          ) : null}
          {kernels.includes("sol") ? (
            <label>
              Top-k ratio
              <input
                aria-label="Sol-Attn top-k ratio"
                max={1}
                min={0.05}
                onChange={(event) => update({ topkRatio: event.target.value })}
                placeholder={tau !== null ? `tau ${tau} (from the recipe)` : undefined}
                step={0.05}
                type="number"
                value={state.topkRatio ?? (tau !== null ? "" : DEFAULT_SOL_TOPK_RATIO)}
              />
            </label>
          ) : null}
          {recorded ? (
            <span className="field-hint">Recipe settings not shown here are kept as recorded.</span>
          ) : null}
          <span className="field-hint">
            Lossy: output differs from the exact path. CUDA only; not a default.
          </span>
          {error ? <span className="field-error">{error}</span> : null}
        </>
      ) : null}
    </details>
  );
}
