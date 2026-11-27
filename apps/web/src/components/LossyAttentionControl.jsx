import React from "react";
import {
  DEFAULT_SOL_TOPK_RATIO,
  attentionExecutionSummary,
  lossyAttentionAvailable,
  lossyAttentionDefaults,
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
};

export function LossyAttentionControl({ backend, model, value, onChange, error }) {
  if (!lossyAttentionAvailable(backend, model)) return null;
  const state = { ...lossyAttentionDefaults(), ...(value ?? {}) };
  const served = servedLossyKernels(model);
  const kernel = served.includes(state.kernel) ? state.kernel : served[0];
  const update = (patch) => onChange({ ...state, kernel, ...patch });
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
              {served.map((key) => (
                <option key={key} value={key}>
                  {KERNEL_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          {kernel === "int8" ? (
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
          ) : (
            <label>
              Top-k ratio
              <input
                aria-label="Sol-Attn top-k ratio"
                max={1}
                min={0.05}
                onChange={(event) => update({ topkRatio: event.target.value })}
                step={0.05}
                type="number"
                value={state.topkRatio ?? DEFAULT_SOL_TOPK_RATIO}
              />
            </label>
          )}
          <span className="field-hint">
            Lossy: output differs from the exact path. CUDA only; not a default.
          </span>
          {error ? <span className="field-error">{error}</span> : null}
        </>
      ) : null}
    </details>
  );
}

/** A lossy-attention asset's receipt: whether the requested kernel actually ran, from the
 *  worker's `rawAdapterSettings.attentionExecution` (recorded only for an opted-in job). */
export function AttentionExecutionNote({ asset, receipt = asset?.recipe?.rawAdapterSettings?.attentionExecution }) {
  const summary = attentionExecutionSummary(receipt);
  if (!summary) return null;
  return (
    <div className="attention-execution-note" role="note" aria-label="Lossy attention">
      Lossy attention: {summary}
    </div>
  );
}
