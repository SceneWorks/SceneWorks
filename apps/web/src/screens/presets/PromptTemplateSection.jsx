import React from "react";

export function PromptTemplateSection({
  editable,
  form,
  isGeneral,
  renderSectionHead,
  updateField,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead(
          "Prompt template",
          isGeneral
            ? "Fragments added around your Studio prompt. Stack several general presets to combine them."
            : "Text wrapped around whatever you type in the Studio. The preview updates as you edit.",
        )}
        <div className="preset-prompt-grid">
          <label className="field">
            <span>{isGeneral ? "Prepend to prompt" : "Prefix"}</span>
            <textarea
              disabled={!editable}
              onChange={(event) => updateField("promptPrefix", event.target.value)}
              placeholder="e.g. Cinematic 35mm, warm tungsten,"
              rows={2}
              value={form.promptPrefix}
            />
          </label>
          <label className="field">
            <span>{isGeneral ? "Append to prompt" : "Suffix"}</span>
            <textarea
              disabled={!editable}
              onChange={(event) => updateField("promptSuffix", event.target.value)}
              placeholder="e.g. shallow depth of field, neutral grade"
              rows={2}
              value={form.promptSuffix}
            />
          </label>
        </div>
        <div className="preset-live-preview">
          <span className="eyebrow">Live preview</span>
          <p>
            {form.promptPrefix.trim() ? <span className="prefix">{form.promptPrefix.trim()} </span> : null}
            <span className="token">your prompt</span>
            {form.promptSuffix.trim() ? <span className="suffix"> {form.promptSuffix.trim()}</span> : null}
          </p>
        </div>
      </div>
    );

}
