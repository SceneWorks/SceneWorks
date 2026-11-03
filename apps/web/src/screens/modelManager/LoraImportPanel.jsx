import { equalItemProps } from "./memo.js";
import React from "react";

export const LoraImportPanel = React.memo(function LoraImportPanel({
  importLora,
  importForm,
  setImportForm,
  importingLora,
  activeProject,
  families,
  showBaseModelSelect,
  ltxBaseModelRequired,
  baseModelOptions,
  isFileImport,
  fileInputKey,
  showSecondaryFileSlot,
  importDisabled,
  moeMissingSecondary,
  importMessage,
  KeywordTagEditor,
 }) {
  return (
          <form className="models-accent-band models-import-panel" aria-label="Import LoRA" onSubmit={importLora}>
            <div className="models-accent-band-head">
              <span className="models-accent-dot" aria-hidden="true" />
              <p className="eyebrow">Import LoRA</p>
              <span className="models-accent-band-caption">Add a LoRA from a URL or file — auto-detects family</span>
            </div>
            <div className="segmented-control compact-segment" aria-label="LoRA import source">
              <button
                className={importForm.mode === "url" ? "active" : ""}
                disabled={importingLora}
                onClick={() => setImportForm((current) => ({ ...current, mode: "url" }))}
                type="button"
              >
                URL
              </button>
              <button
                className={importForm.mode === "file" ? "active" : ""}
                disabled={importingLora}
                onClick={() => setImportForm((current) => ({ ...current, mode: "file" }))}
                type="button"
              >
                Upload
              </button>
            </div>
            <div className="models-import-grid">
              <label>
                Scope
                <select
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, scope: event.target.value }))}
                  value={importForm.scope}
                >
                  <option value="global">Global</option>
                  <option disabled={!activeProject} value="project">
                    Project
                  </option>
                </select>
              </label>
              <label>
                Family
                <select
                  disabled={importingLora || !families.length}
                  onChange={(event) =>
                    setImportForm((current) => ({
                      ...current,
                      family: event.target.value,
                      baseModel: "",
                      secondaryFile: null,
                    }))
                  }
                  value={importForm.family}
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
              {showBaseModelSelect ? (
                <label>
                  Base model
                  <select
                    disabled={importingLora}
                    onChange={(event) => setImportForm((current) => ({ ...current, baseModel: event.target.value }))}
                    value={importForm.baseModel}
                  >
                    <option value="">{ltxBaseModelRequired ? "Select LTX base model" : "Auto / unspecified"}</option>
                    {baseModelOptions.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name ?? model.id}
                      </option>
                    ))}
                  </select>
                  {ltxBaseModelRequired ? (
                    <span className="field-hint">Required so LTX-2.3 and LTX-2.5 adapters stay on the correct base.</span>
                  ) : null}
                </label>
              ) : null}
              {isFileImport ? (
                <>
                  <label>
                    LoRA File
                    <span className="file-picker-row">
                      <span className="file-upload-button">
                        Choose
                        <input
                          accept=".safetensors,.ckpt,.pt,.bin"
                          disabled={importingLora}
                          key={fileInputKey}
                          onChange={(event) => setImportForm((current) => ({ ...current, file: event.target.files?.[0] ?? null }))}
                          type="file"
                        />
                      </span>
                      <span className="selected-file-name">{importForm.file?.name ?? "No file selected"}</span>
                    </span>
                  </label>
                  {showSecondaryFileSlot ? (
                    <label>
                      Low-noise expert (Wan A14B MoE)
                      <span className="file-picker-row">
                        <span className="file-upload-button">
                          Choose
                          <input
                            accept=".safetensors,.ckpt,.pt,.bin"
                            disabled={importingLora}
                            key={`secondary-${fileInputKey}`}
                            onChange={(event) => setImportForm((current) => ({ ...current, secondaryFile: event.target.files?.[0] ?? null }))}
                            type="file"
                          />
                        </span>
                        <span className="selected-file-name">{importForm.secondaryFile?.name ?? "No file selected"}</span>
                      </span>
                    </label>
                  ) : null}
                </>
              ) : (
                <label>
                  Source URL
                  <input
                    disabled={importingLora}
                    onChange={(event) => setImportForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                    placeholder="https://..."
                    value={importForm.sourceUrl}
                  />
                </label>
              )}
              <label>
                Name
                <input
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Optional"
                  value={importForm.name}
                />
              </label>
              <button disabled={importDisabled} type="submit">
                {importingLora ? (isFileImport ? "Uploading" : "Queueing...") : "Queue Import"}
              </button>
            </div>
            <div className="lora-import-metadata">
              <label>
                Trigger keywords
                <KeywordTagEditor
                  disabled={importingLora}
                  onChange={(triggerKeywords) =>
                    setImportForm((current) => ({ ...current, triggerKeywords }))
                  }
                  placeholder="e.g. sksStyle — press Enter or comma to add"
                  value={importForm.triggerKeywords}
                />
              </label>
              <label>
                Notes
                <textarea
                  disabled={importingLora}
                  onChange={(event) => setImportForm((current) => ({ ...current, notes: event.target.value }))}
                  placeholder="How to use this LoRA — keyword combinations, recommended weight, and other tips."
                  rows={2}
                  value={importForm.notes}
                />
              </label>
            </div>
            {showSecondaryFileSlot ? (
              <p className="helper-copy">
                Wan A14B is a two-expert model. Upload both the high-noise file and the low-noise expert so each expert
                gets its own weights.
              </p>
            ) : null}
            {moeMissingSecondary ? (
              <p className="inline-warning">
                No low-noise expert selected — this LoRA will load into the high-noise expert only, leaving the
                low-noise expert un-adapted.
              </p>
            ) : null}
            {importForm.scope === "project" && !activeProject ? <p className="helper-copy">Open a project before importing a project LoRA.</p> : null}
            {importMessage.text ? <p className={importMessage.tone === "success" ? "inline-success" : "inline-warning"}>{importMessage.text}</p> : null}
          </form>
  );
}, equalItemProps);
