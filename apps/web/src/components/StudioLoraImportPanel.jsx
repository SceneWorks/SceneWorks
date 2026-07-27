import React, { useMemo, useState } from "react";
import { KeywordTagEditor } from "./KeywordTagEditor.jsx";
import { modelLoraFamilies, normalizeLoraFamily } from "../presetUtils.js";

// In-studio LoRA import (studio-cleanup sc-15373). The Model Manager's Import LoRA form, lifted
// into the studio's Add-LoRA picker so a user who has no compatible LoRA — the exact moment the
// picker tells them so — can import one without leaving the screen. Same `.models-accent-band
// .models-import-panel` chrome, same `createLoraImportJob` seam, same trigger-keyword/notes
// metadata; the Wan A14B two-expert slot stays a Model Manager concern (it needs the base-model
// picker that goes with it), so this form covers the single-file / URL cases.
//
// Rendered as a <div role="group">, NOT a <form>: the studio screen is already one <form> and
// nested forms are invalid HTML, so the submit rides a click handler instead.
export function StudioLoraImportPanel({ models = [], activeProject, createLoraImportJob }) {
  const families = useMemo(
    () => Array.from(new Set(models.flatMap((model) => modelLoraFamilies(model)).filter(Boolean))).sort(),
    [models],
  );
  const [form, setForm] = useState({
    // Default to file upload, matching Model Manager: most LoRAs are a file the user already has.
    mode: "file",
    sourceUrl: "",
    file: null,
    name: "",
    scope: activeProject ? "project" : "global",
    family: "",
    triggerKeywords: [],
    notes: "",
  });
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState({ tone: "neutral", text: "" });
  // Re-mounts the file input so choosing the same file twice still emits a change event.
  const [fileInputKey, setFileInputKey] = useState(0);

  const isFileImport = form.mode === "file";
  const patch = (changes) => setForm((current) => ({ ...current, ...changes }));
  const importDisabled =
    importing ||
    !createLoraImportJob ||
    (form.scope === "project" && !activeProject) ||
    (isFileImport ? !form.file : !form.sourceUrl.trim());

  async function submit() {
    if (importDisabled) {
      return;
    }
    setImporting(true);
    setMessage({
      tone: "neutral",
      text: isFileImport ? "Uploading LoRA file before queueing import." : "",
    });
    try {
      const job = await createLoraImportJob({
        ...(isFileImport ? { file: form.file } : { sourceUrl: form.sourceUrl.trim() }),
        name: form.name.trim() || undefined,
        scope: form.scope,
        triggerWords: form.triggerKeywords,
        notes: form.notes.trim() || undefined,
        ...(form.family ? { family: form.family } : {}),
      });
      const loraId = job?.payload?.loraId;
      const resolvedFamily = job?.payload?.manifestEntry?.family;
      // Report the detected family in the same normalized vocabulary the dropdown uses.
      const detectionNote =
        !form.family && resolvedFamily ? ` Detected family: ${normalizeLoraFamily(resolvedFamily)}.` : "";
      patch({ sourceUrl: "", file: null, name: "", triggerKeywords: [], notes: "" });
      setFileInputKey((current) => current + 1);
      setMessage({
        tone: "success",
        text: `${loraId ? `LoRA import queued for ${loraId}.` : "LoRA import queued."}${detectionNote}`,
      });
    } catch (err) {
      setMessage({ tone: "error", text: err.message });
    } finally {
      setImporting(false);
    }
  }

  return (
    <div className="models-accent-band models-import-panel studio-lora-import" role="group" aria-label="Import LoRA">
      <div className="models-accent-band-head">
        <span className="models-accent-dot" aria-hidden="true" />
        <p className="eyebrow">Import LoRA</p>
        <span className="models-accent-band-caption">Add a LoRA from a URL or file — auto-detects family</span>
      </div>
      <div className="segmented-control compact-segment" role="group" aria-label="LoRA import source">
        <button
          className={form.mode === "url" ? "active" : ""}
          disabled={importing}
          onClick={() => patch({ mode: "url" })}
          type="button"
        >
          URL
        </button>
        <button
          className={form.mode === "file" ? "active" : ""}
          disabled={importing}
          onClick={() => patch({ mode: "file" })}
          type="button"
        >
          Upload
        </button>
      </div>
      <div className="models-import-grid">
        <label>
          Scope
          <select disabled={importing} onChange={(event) => patch({ scope: event.target.value })} value={form.scope}>
            <option value="global">Global</option>
            <option disabled={!activeProject} value="project">
              Project
            </option>
          </select>
        </label>
        <label>
          Family
          <select
            disabled={importing || !families.length}
            onChange={(event) => patch({ family: event.target.value })}
            value={form.family}
          >
            {families.length ? (
              <>
                <option value="">Auto-detect</option>
                {families.map((family) => (
                  <option key={family} value={family}>
                    {family}
                  </option>
                ))}
              </>
            ) : (
              <option value="">No model families</option>
            )}
          </select>
        </label>
        {isFileImport ? (
          <label>
            LoRA File
            <span className="file-picker-row">
              <span className="file-upload-button">
                Choose
                <input
                  accept=".safetensors,.ckpt,.pt,.bin"
                  disabled={importing}
                  key={fileInputKey}
                  onChange={(event) => patch({ file: event.target.files?.[0] ?? null })}
                  type="file"
                />
              </span>
              <span className="selected-file-name">{form.file?.name ?? "No file selected"}</span>
            </span>
          </label>
        ) : (
          <label>
            Source URL
            <input
              disabled={importing}
              onChange={(event) => patch({ sourceUrl: event.target.value })}
              placeholder="https://..."
              value={form.sourceUrl}
            />
          </label>
        )}
        <label>
          Name
          <input
            disabled={importing}
            onChange={(event) => patch({ name: event.target.value })}
            placeholder="Optional"
            value={form.name}
          />
        </label>
        <button disabled={importDisabled} onClick={submit} type="button">
          {importing ? (isFileImport ? "Uploading" : "Queueing...") : "Queue import"}
        </button>
      </div>
      <div className="lora-import-metadata">
        <label>
          Trigger keywords
          <KeywordTagEditor
            disabled={importing}
            onChange={(triggerKeywords) => patch({ triggerKeywords })}
            placeholder="e.g. sksStyle — press Enter or comma to add"
            value={form.triggerKeywords}
          />
        </label>
        <label>
          Notes
          <textarea
            disabled={importing}
            onChange={(event) => patch({ notes: event.target.value })}
            placeholder="How to use this LoRA — keyword combinations, recommended weight, and other tips."
            rows={2}
            value={form.notes}
          />
        </label>
      </div>
      {form.scope === "project" && !activeProject ? (
        <p className="helper-copy">Open a project before importing a project LoRA.</p>
      ) : null}
      {message.text ? (
        <p className={message.tone === "success" ? "inline-success" : "inline-warning"}>{message.text}</p>
      ) : null}
      <p className="helper-copy">
        Imported LoRAs land in your library and become selectable here once the import job completes — no trip to
        Model Manager.
      </p>
    </div>
  );
}
