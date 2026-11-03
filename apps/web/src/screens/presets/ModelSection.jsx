import React from "react";

export function ModelSection({
  editable,
  form,
  imageModels,
  modelHelperLine,
  modelInstalled,
  models,
  renderSectionHead,
  selectedModel,
  updateField,
  videoModels,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead("Base model", "Which checkpoint this preset renders with.")}
        <div className="preset-model-grid">
          <label className="field">
            <span>Model</span>
            <select
              disabled={!editable}
              onChange={(event) => updateField("model", event.target.value)}
              value={form.model}
            >
              {form.model ? null : <option value="">{models.length ? "Select a model…" : "No models installed"}</option>}
              {/* A preset can be pinned to a model this install hasn't downloaded. Say so,
                  rather than rendering an empty select that silently rewrites it on save. */}
              {form.model && !modelInstalled ? (
                <option value={form.model}>{selectedModel?.name ?? form.model} — not installed</option>
              ) : null}
              <optgroup label="Image">
                {imageModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name ?? model.id}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Video">
                {videoModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name ?? model.id}
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <span className="preset-model-meta-line">{modelHelperLine(selectedModel)}</span>
        </div>
      </div>
    );

}
