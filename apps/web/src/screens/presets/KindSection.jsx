import React from "react";

export function KindSection({
  creating,
  editable,
  form,
  renderSectionHead,
  updateField,
}) {

    // Kind is locked after create (like the id): switching an existing preset between kinds
    // would strand its model-specific fields or its stacked-fragment semantics. New presets
    // choose; existing presets show it read-only.
    const kinds = [
      ["model", "Model preset", "Pinned to one model + workflow. Full controls and LoRAs."],
      ["general", "General preset", "Model-agnostic. Prompt + aspect only; stacks onto any model."],
    ];
    return (
      <div className="preset-form-section">
        {renderSectionHead(
          "Type",
          "General presets layer prompt fragments onto any model in any workflow; model presets are pinned to one model.",
        )}
        <div aria-label="Preset type" className="segmented-control preset-kind" role="radiogroup">
          {kinds.map(([value, label, help]) => (
            <button
              aria-checked={form.kind === value}
              className={form.kind === value ? "active" : ""}
              disabled={!creating || !editable}
              key={value}
              onClick={() => updateField("kind", value)}
              role="radio"
              title={help}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    );

}
