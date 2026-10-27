import React, { useState } from "react";
import { apiFetch } from "../api.js";
import { mossComponentState } from "../mossImage.js";
import { MossTensorControls } from "./MossTensorControls.jsx";

// The exact conditional edit package is managed independently of generic LoRAs and DMD.
export function MossImageEditControls({ model, token, jobs = [], options, onChange, generation, onGenerationChange, advanced, onAdvancedChange, assets, importAsset, promptBatch, onPromptBatchChange }) {
  const [download, setDownload] = useState({ busy: false, error: "", jobs: [] });
  const [acknowledged, setAcknowledged] = useState(false);
  const state = mossComponentState(model, null, "edit_image");
  const pending = download.jobs.map((queued) => jobs.find((job) => job.id === queued.id) ?? queued);
  const failed = pending.find((job) => job.status === "failed" || job.status === "canceled");
  async function install() {
    setDownload({ busy: true, error: "", jobs: [] });
    try {
      const result = await apiFetch(`/api/v1/models/${model.id}/conditional-components/moss_image_edit/download`, token, {
        method: "POST", body: JSON.stringify({ licenseAcknowledged: acknowledged }),
      });
      setDownload({ busy: false, error: "", jobs: result.jobs ?? [] });
    } catch (error) {
      setDownload({ busy: false, error: error.message, jobs: [] });
    }
  }
  return <fieldset className="moss-edit-controls">
    <legend>MOSS Image Edit</legend>
    <p>Exact adapter: chinoll/Moss-Image-Edit. Weight license: unspecified; base-model and code licenses do not establish adapter terms.</p>
    <p role="status">Edit adapter: {state.ready ? "installed" : download.busy ? "queueing" : state.state}.
      {pending.map((job) => ` ${job.status}${Number.isFinite(job.progress) ? ` ${Math.round(job.progress * 100)}%` : ""}`).join(";")}</p>
    {!state.ready ? <>
      <label><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} />I acknowledge the adapter source and unspecified weight license.</label>
      <button type="button" disabled={!acknowledged || download.busy || state.state === "blocked" || pending.some((job) => !["failed", "canceled", "completed"].includes(job.status))} onClick={install}>Install MOSS edit adapter</button>
    </> : null}
    {download.error || failed ? <p role="alert">{download.error || failed.error || "Edit adapter installation failed. Repair or retry the package before editing."}</p> : null}
    {state.state === "incomplete" || state.state === "blocked" ? <p role="alert">The pinned edit package is incomplete, corrupt, or blocked. Repair it in Models before editing.</p> : null}
    <label>Reference maximum edge<input type="number" min="16" max="4096" step="1" value={options.reference_max_edge} onChange={(event) => onChange({ ...options, reference_max_edge: Number(event.target.value) })} /></label>
    <label><input type="checkbox" checked={options.use_kv_cache} onChange={(event) => onChange({ ...options, use_kv_cache: event.target.checked })} />Cache reference K/V</label>
    {generation ? <>
      {[["width", "Output width", 16, 4096, 16], ["height", "Output height", 16, 4096, 16], ["steps", "Steps", 1, undefined, 1], ["guidanceScale", "Guidance", 0, undefined, "any"], ["schedulerShift", "Schedule shift", 0, undefined, "any"], ["training_shift", "Training shift", 0, undefined, "any"], ["max_sequence_length", "Context limit", 1, 4096, 1]].map(([key, label, min, max, step]) => <label key={key}>{label}<input type="number" value={generation[key] ?? (key === "training_shift" ? 1 : "")} min={min} max={max} step={step} onChange={(event) => onGenerationChange({ ...generation, [key]: Number(event.target.value) })} /></label>)}
      <label>Negative prompt<textarea aria-label="Negative prompt" value={generation.negativePrompt} onChange={(event) => onGenerationChange({ ...generation, negativePrompt: event.target.value })} /></label>
    </> : null}
    {onAdvancedChange ? <MossTensorControls value={advanced} onChange={onAdvancedChange} assets={assets} importAsset={importAsset} promptBatch={promptBatch} onPromptBatchChange={onPromptBatchChange} /> : null}
    <p>References retain their aspect ratios and are resized without crop, aligned to 16 pixels. Their order and repeated slots are preserved.</p>
  </fieldset>;
}
