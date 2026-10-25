import React, { useState } from "react";
import { mossAdvancedDefaults, mossTensorRef } from "../mossImageAdvanced.js";

// Every shell uses the existing asset import action; no values are decoded in the browser.
export function MossTensorControls({ assets = [], value = mossAdvancedDefaults(), onChange, importAsset, promptBatch = [], onPromptBatchChange }) {
  const [status, setStatus] = useState("");
  const [imported, setImported] = useState([]);
  const tensors = [...assets, ...imported].filter((asset, index, all) => asset.type === "tensor" && all.findIndex((candidate) => candidate.id === asset.id) === index);
  function set(role, id) {
    const next = structuredClone(value);
    const [branch, field] = role.split(".");
    const ref = id ? mossTensorRef(tensors.find((asset) => asset.id === id)) : null;
    if (field) {
      if (ref) next[branch] = { ...next[branch], [field]: ref };
      else if (field === "values") delete next[branch];
      else if (next[branch]) delete next[branch][field];
    } else if (ref) next[branch] = ref;
    else delete next[branch];
    onChange(next);
  }
  function picker(role, label, dtype, rank, disabled = false) {
    const [branch, field] = role.split(".");
    const ref = field ? value[branch]?.[field] : value[branch];
    return <label key={role}>{label}<select disabled={disabled} value={ref?.assetId ?? ""} onChange={(event) => set(role, event.target.value)}>
      <option value="">None</option>
      {ref?.assetId && !tensors.some((asset) => asset.id === ref.assetId) ? <option value={ref.assetId}>Missing exact tensor: {ref.assetId}</option> : null}
      {tensors.filter((asset) => asset.tensor?.dtype === dtype && asset.tensor?.shape?.length === rank).map((asset) => <option key={asset.id} value={asset.id}>{asset.displayName ?? asset.id} — {asset.tensor.shape.join(" × ")}</option>)}
    </select>{ref ? <small>SHA256 {ref.sha256}</small> : null}</label>;
  }
  return <fieldset><legend>Native edit tensors</legend>
    <p>Import inert NPY files: little-endian FP32 embeddings [B,S,4096] and noise [B×count,128,H/16,W/16], or binary U8 masks [B,S]. Maximum 512 MiB per file. Exact bytes and identities are retained for replay.</p>
    {importAsset ? <label>Import tensor NPY<input type="file" accept=".npy,application/x-npy" onChange={async (event) => {
      const file = event.target.files?.[0]; event.target.value = "";
      if (!file) return;
      if (file.size > 512 * 1024 * 1024 + 4108) { setStatus("Tensor exceeds the 512 MiB artifact budget."); return; }
      setStatus("Importing tensor…");
      try { const asset = await importAsset(file, { throwOnError: true, select: false }); if (!asset || asset.type !== "tensor") throw new Error("Import did not return a measured tensor artifact."); setImported((current) => [...current, asset]); setStatus("Tensor imported. Choose its role below."); }
      catch (error) { setStatus(error.message); }
    }} /></label> : <p>Import tensor NPY files into this project in Library, then choose their roles here.</p>}
    {status ? <p role="status">{status}</p> : null}
    {picker("positive.values", "Positive embeddings", "float32", 3)}
    {picker("positive.mask", "Positive attention mask", "uint8", 2, !value.positive?.values)}
    {onPromptBatchChange ? <><label><input type="checkbox" checked={promptBatch.length > 0} disabled={Boolean(value.positive)} onChange={(event) => onPromptBatchChange(event.target.checked ? [""] : [])} />Native text prompt batch</label>
      {promptBatch.length ? <label>One positive prompt per conditioning row<textarea aria-label="One positive prompt per conditioning row" disabled={Boolean(value.positive)} value={promptBatch.join("\n")} onChange={(event) => onPromptBatchChange(event.target.value.split("\n"))} /></label> : null}</> : null}
    {picker("negative.values", "Negative embeddings", "float32", 3, Boolean(value.negativePrompts))}
    {picker("negative.mask", "Negative attention mask", "uint8", 2, !value.negative?.values)}
    <label><input type="checkbox" checked={Boolean(value.negativePrompts)} disabled={Boolean(value.negative)} onChange={(event) => { const next = { ...value }; if (event.target.checked) next.negativePrompts = [""]; else delete next.negativePrompts; onChange(next); }} />Per-prompt negative strings</label>
    {value.negativePrompts ? <label>One negative string per conditioning row<textarea aria-label="One negative string per conditioning row" value={value.negativePrompts.join("\n")} onChange={(event) => onChange({ ...value, negativePrompts: event.target.value.split("\n") })} /></label> : null}
    {picker("noise", "Explicit initial noise", "float32", 4)}
    {value.positive ? <p>Positive tensors replace both prompt fields. Clear text prompts before generating.</p> : null}
    {value.noise ? <p>Caller-supplied noise determines initialization. The execution seed is null; replay retains this exact noise artifact.</p> : null}
    <label>Native output<select value={value.output ?? "images"} onChange={(event) => onChange({ ...value, output: event.target.value })}><option value="images">Decoded images</option><option value="latents">FP32 sampled latents (NPY)</option></select></label>
    <p>Latents are downloadable tensors without an image preview. Reference images are still encoded. Adapter strength is fixed at 1; component dtype and quantization follow the pinned model policy.</p>
  </fieldset>;
}
