import React from "react";

export function VedaControls({ value, onChange, steps }) {
  const update = ({ target }) => onChange({ ...value, [target.name]: target.type === "checkbox" ? target.checked : target.value });
  return <div className="lightning-toggle">
    <label className="checkline">
      <input name="enabled" type="checkbox" onChange={update} checked={value.enabled} />Veda sparse attention
    </label>
    {value.enabled ? <>
      {[["generated", "Generated sparsity"], ["reference", "Reference sparsity"], ["fullAttentionLayers", "Full-attention layers"], ["fullAttentionSteps", "Full-attention steps"]].map(([key, label]) => <label key={key}>
        {label}<input name={key} aria-label={label} onChange={update} value={value[key]} />
      </label>)}
      <p className="helper-copy">Budget: &lt;100% (90%) or positive key tiles (32); 0% = all keys. Layers 0–49; steps 0–{steps - 1}; zero-based lists/ranges (0, 2–4).</p>
    </> : null}
  </div>;
}
