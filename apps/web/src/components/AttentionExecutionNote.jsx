import React from "react";
import { attentionExecutionSummary } from "../lossyAttention.js";

// SW-6132: a lossy-attention asset's receipt — whether the requested kernel actually ran, from the
// worker's `rawAdapterSettings.attentionExecution` (recorded only for an opted-in job). Loaded
// lazily by the asset panels (`LazyAttentionExecutionNote`), so it stays out of the initial bundle.
export default function AttentionExecutionNote({ asset, receipt = asset?.recipe?.rawAdapterSettings?.attentionExecution }) {
  const summary = attentionExecutionSummary(receipt);
  if (!summary) return null;
  return (
    <div className="attention-execution-note" role="note" aria-label="Lossy attention">
      Lossy attention: {summary}
    </div>
  );
}
