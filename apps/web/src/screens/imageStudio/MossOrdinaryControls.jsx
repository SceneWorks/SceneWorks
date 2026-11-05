import React from "react";

export function MossOrdinaryControls({ options, disabled, onChange }) {
  return <>
    <label>Training shift<input disabled={disabled} type="number" min="0" step="any" value={options.training_shift ?? 1} onChange={(event) => onChange("training_shift", Number(event.target.value))} /></label>
    <label>Unconditional skip layers<MossLayerInput disabled={disabled} value={options.uncond_skip_layers} placeholder="[22] (author default); [] disables" onChange={(value) => onChange("uncond_skip_layers", value)} /></label>
    <label className="checkline"><input disabled={disabled} type="checkbox" checked={Boolean(options.slg)} onChange={(event) => onChange("slg", event.target.checked ? { layers: [22], scale: 1, start: 0, stop: 1 } : null)} />Skip-layer guidance</label>
    {options.slg ? <>
      <label>SLG layers<MossLayerInput disabled={disabled} value={options.slg.layers} emptyValue={[]} onChange={(value) => onChange("slg", { ...options.slg, layers: value })} /></label>
      {["scale", "start", "stop"].map((key) => <label key={key}>SLG {key}<input disabled={disabled} type="number" min="0" max={key === "scale" ? undefined : 1} step="any" value={options.slg[key]} onChange={(event) => onChange("slg", { ...options.slg, [key]: Number(event.target.value) })} /></label>)}
    </> : null}
  </>;
}

const mossLayerText = (value) => value == null ? "" : typeof value === "string" ? value : JSON.stringify(value);

function MossLayerInput({ value, onChange, emptyValue = null, ...props }) {
  return <input {...props} value={mossLayerText(value)} onChange={(event) => {
    const text = event.target.value;
    let next = text;
    try { next = text.trim() === "" ? emptyValue : JSON.parse(text) ?? emptyValue; }
    catch { /* Keep incomplete JSON editable until it becomes a valid layer list. */ }
    onChange(next);
  }} />;
}
