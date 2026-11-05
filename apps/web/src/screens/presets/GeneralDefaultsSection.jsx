import React from "react";

export function GeneralDefaultsSection({
  GENERAL_ASPECT_RATIOS,
  editable,
  form,
  renderSectionHead,
  updateField,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead("Defaults", "Applied on top of whatever model and mode the Studio is on.")}
        <div className="preset-defaults-grid">
          <label className="field">
            <span>Aspect</span>
            <select disabled={!editable} onChange={(event) => updateField("aspect", event.target.value)} value={form.aspect}>
              <option value="">No default</option>
              {GENERAL_ASPECT_RATIOS.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Variations</span>
            <select disabled={!editable} onChange={(event) => updateField("count", event.target.value)} value={form.count}>
              <option value="">No default</option>
              {[1, 2, 3, 4, 6, 8].map((n) => (
                <option key={n} value={String(n)}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Sort order</span>
            <input
              disabled={!editable}
              onChange={(event) => updateField("order", event.target.value)}
              placeholder="0"
              type="number"
              value={form.order}
            />
          </label>
        </div>
        <label className="field prompt-field">
          <span>Negative prompt</span>
          <input
            disabled={!editable}
            onChange={(event) => updateField("negativePrompt", event.target.value)}
            placeholder="oversaturated, hands, text, watermark"
            value={form.negativePrompt}
          />
        </label>
      </div>
    );

}
