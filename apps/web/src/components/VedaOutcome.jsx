import React from "react";

// Provider receipts only: requested controls never imply that sparse execution occurred.
export function VedaOutcome({ asset, raw = asset?.recipe?.rawAdapterSettings, outcome = raw?.vedaExecution, requested = raw?.veda, label = "Observed attention" }) {
  const effective = { dense: "Dense", sparse: "Sparse", mixed: "Mixed sparse and dense", unobserved: "Unobserved" }[outcome?.effective];
  if (!effective) return null;
  const grid = (value) => Array.isArray(value) ? value.join(" × ") : null;
  const facts = [
    ["Kernel", outcome.kernel],
    ["Plan", outcome.plan && `${outcome.plan} (${outcome.planMatch}; trained ${grid(outcome.planGrid)}, executed ${grid(outcome.executionGrid)})`],
    ["Full attention computed", Number.isFinite(outcome.totalAttentionFraction) ? `${(100 * outcome.totalAttentionFraction).toFixed(2)}%` : null],
    ["Dense fallback", outcome.fallbackReason],
  ];
  return <div className="veda-outcome" role="group" aria-label={label}>
    <strong>{label}: {effective}</strong>
    {facts.map(([name, value]) => value && <div key={name}>{name}: {value}</div>)}
    {requested && <div>Requested controls: {JSON.stringify(requested)}</div>}
  </div>;
}

export const AssetVedaOutcome = VedaOutcome;
