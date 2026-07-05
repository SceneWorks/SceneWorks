import React, { useMemo, useRef, useState } from "react";

import {
  cardinality,
  expandBatch,
  extractKeys,
  missingKeys,
  splitPromptLines,
} from "../promptBatch.js";
import { fromPromptBatchImport, serializePromptBatchExport } from "../promptBatchIO.js";
import { Icon } from "./Icons.jsx";

// Batch authoring panel (sc-9955, epic 9952). Rendered in place of the single-prompt
// area when Image Studio's Batch mode is on. Owns the prompt-list textarea, the
// per-variable chip editors, the live preview + total, and the save/load/export/import
// controls. Pure UI over slice 1's engine (promptBatch.js) and slice 2's persistence
// (usePromptBatches via callbacks + promptBatchIO for the portable file). The actual
// fan-out on "Run batch" is wired by the parent (slice 4, sc-9956).

// One variable's value editor: an auto-expanding list of inputs. A typed value counts
// LIVE (no "press Enter to commit" step — that stranded a typed value as 0 values); a
// trailing empty box always sits at the end, so more values appear as you type. Each is
// its own input (not a comma split) so a value may contain commas ("red, wavy"). Empty
// values are ignored everywhere by the engine and stripped from the saved payload.
function VariableChips({ label, values, onChange }) {
  const list = Array.isArray(values) ? values : [];
  // Ensure exactly one trailing empty slot to type the next value into.
  const slots = list.length && list[list.length - 1].trim() === "" ? list : [...list, ""];
  const filled = list.filter((value) => value.trim() !== "").length;

  const setAt = (index, value) => {
    const next = [...slots];
    next[index] = value;
    onChange(next);
  };
  const removeAt = (index) => onChange(slots.filter((_, i) => i !== index));

  return (
    <div className="batch-var">
      <div className="batch-var-head">
        <code className="batch-var-key">{`{{${label}}}`}</code>
        <span className="batch-var-count">{filled === 1 ? "1 value" : `${filled} values`}</span>
      </div>
      <div className="batch-var-values">
        {slots.map((value, index) => (
          <div className="batch-value" key={index}>
            <input
              aria-label={`Value ${index + 1} for ${label}`}
              className="batch-var-input"
              onChange={(event) => setAt(index, event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                } else if (event.key === "Backspace" && value === "" && slots.length > 1) {
                  event.preventDefault();
                  removeAt(index);
                }
              }}
              placeholder={index === 0 ? "Type a value" : "Another value"}
              value={value}
            />
            {value.trim() !== "" ? (
              <button
                aria-label={`Remove ${value}`}
                className="batch-value-remove"
                onClick={() => removeAt(index)}
                type="button"
              >
                ×
              </button>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function BatchPromptPanel({
  promptsText,
  onPromptsTextChange,
  variableValues,
  onVariableValuesChange,
  count = 1,
  batches = [],
  projectId = null,
  name,
  onNameChange,
  scope,
  onScopeChange,
  loadedBatchId = null,
  onSave,
  onLoad,
  onDelete,
  onImport,
  busy = false,
  error = "",
}) {
  const fileInputRef = useRef(null);
  const [ioError, setIoError] = useState("");

  const prompts = useMemo(() => splitPromptLines(promptsText), [promptsText]);
  const keys = useMemo(() => extractKeys(prompts), [prompts]);
  const variables = useMemo(
    () => keys.map((key) => ({ key, values: variableValues[key] ?? [] })),
    [keys, variableValues],
  );
  const total = useMemo(() => cardinality(prompts, variables, count), [prompts, variables, count]);
  const previewPrompt = useMemo(() => expandBatch(prompts, variables)[0]?.prompt ?? "", [prompts, variables]);
  const missing = useMemo(() => missingKeys(prompts, variables), [prompts, variables]);

  const setKeyValues = (key, values) =>
    onVariableValuesChange({ ...variableValues, [key]: values });

  const currentExport = () => ({
    name,
    prompts,
    variables,
    lastValues: Object.fromEntries(variables.map((variable) => [variable.key, variable.values])),
  });

  const handleExport = () => {
    setIoError("");
    const blob = new Blob([serializePromptBatchExport(currentExport())], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    const slug = (name || "prompt-batch").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    anchor.download = `${slug || "prompt-batch"}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const handleImportFile = async (event) => {
    setIoError("");
    const file = event.target.files?.[0];
    event.target.value = ""; // allow re-importing the same file
    if (!file) return;
    try {
      const payload = fromPromptBatchImport(await file.text());
      onImport(payload);
    } catch (importError) {
      setIoError(importError.message);
    }
  };

  const promptCount = prompts.length;
  const saveDisabled = busy || !name.trim() || promptCount === 0;

  return (
    <div className="batch-panel">
      <div className="batch-panel-main">
        <label className="batch-field">
          <span className="batch-field-label">Prompts — one per line</span>
          <textarea
            aria-label="Batch prompts"
            className="batch-prompts"
            onChange={(event) => onPromptsTextChange(event.target.value)}
            placeholder={"{{name}} with {{hair}} hair, front view\n{{name}} profile, soft light\n\nUse --- on its own line for multi-line prompts"}
            value={promptsText}
          />
        </label>

        {keys.length > 0 ? (
          <div className="batch-vars">
            <span className="batch-field-label">Variables</span>
            {keys.map((key) => (
              <VariableChips
                key={key}
                label={key}
                values={variableValues[key] ?? []}
                onChange={(values) => setKeyValues(key, values)}
              />
            ))}
          </div>
        ) : (
          <p className="batch-hint">
            Add <code>{"{{placeholders}}"}</code> in your prompts (e.g. <code>{"{{name}}"}</code>) to get a value box per
            variable.
          </p>
        )}

        {previewPrompt ? (
          <div className="batch-preview">
            <span className="batch-field-label">First prompt preview</span>
            <p className="batch-preview-text">{previewPrompt}</p>
          </div>
        ) : null}

        <div className="batch-total" aria-live="polite">
          <strong>{total}</strong> {total === 1 ? "image" : "images"}
          <span className="batch-total-detail">
            {promptCount} {promptCount === 1 ? "prompt" : "prompts"} × {count} {count === 1 ? "variation" : "variations"}
            {variables.some((variable) => variable.values.length > 1) ? " × variable values" : ""}
          </span>
        </div>

        {missing.length > 0 ? (
          <p className="batch-warning" role="status">
            Add at least one value for: {missing.map((key) => `{{${key}}}`).join(", ")}
          </p>
        ) : null}
      </div>

      <div className="batch-panel-side">
        <div className="batch-save">
          <span className="batch-field-label">Save this batch</span>
          <input
            aria-label="Batch name"
            className="batch-name"
            onChange={(event) => onNameChange(event.target.value)}
            placeholder="Batch name"
            value={name}
          />
          <div className="batch-scope">
            <label>
              <input
                checked={scope === "global"}
                name="batch-scope"
                onChange={() => onScopeChange("global")}
                type="radio"
              />
              Global
            </label>
            <label className={projectId ? "" : "batch-scope-disabled"}>
              <input
                checked={scope === "project"}
                disabled={!projectId}
                name="batch-scope"
                onChange={() => onScopeChange("project")}
                type="radio"
              />
              This project
            </label>
          </div>
          <button className="batch-btn" disabled={saveDisabled} onClick={onSave} type="button">
            <Icon.Preset size={14} /> {loadedBatchId ? "Update" : "Save"}
          </button>
        </div>

        <div className="batch-load">
          <span className="batch-field-label">Saved batches</span>
          {batches.length ? (
            <ul className="batch-list">
              {batches.map((batch) => (
                <li className={batch.id === loadedBatchId ? "batch-list-item active" : "batch-list-item"} key={batch.id}>
                  <button className="batch-list-load" onClick={() => onLoad(batch)} type="button">
                    <Icon.Folder size={13} /> {batch.name}
                    <span className="batch-list-scope">{batch.scope}</span>
                  </button>
                  <button
                    aria-label={`Delete ${batch.name}`}
                    className="batch-list-delete"
                    onClick={() => onDelete(batch)}
                    type="button"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="batch-hint">No saved batches yet.</p>
          )}
        </div>

        <div className="batch-io">
          <button className="batch-btn ghost" disabled={promptCount === 0} onClick={handleExport} type="button">
            Export .json
          </button>
          <button className="batch-btn ghost" onClick={() => fileInputRef.current?.click()} type="button">
            Import .json
          </button>
          <input
            accept="application/json,.json"
            className="batch-file-input"
            onChange={handleImportFile}
            ref={fileInputRef}
            type="file"
          />
        </div>

        {error || ioError ? (
          <p className="batch-warning" role="alert">
            {error || ioError}
          </p>
        ) : null}
      </div>
    </div>
  );
}
