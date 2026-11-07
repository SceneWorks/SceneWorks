import React from "react";

export function WorkflowSection({
  editable,
  form,
  renderSectionHead,
  segments,
  selectedModel,
  updateField,
}) {

    return (
      <div className="preset-form-section">
        {renderSectionHead("Workflow", "Generation modes the selected model supports.")}
        {segments.length ? (
          <div aria-label="Workflow" className="segmented-control preset-workflow" role="radiogroup">
            {segments.map((segment) => (
              <button
                aria-checked={form.segment === segment.key}
                className={form.segment === segment.key ? "active" : ""}
                disabled={!editable}
                key={segment.key}
                onClick={() => updateField("segment", segment.key)}
                role="radio"
                type="button"
              >
                {segment.label}
              </button>
            ))}
          </div>
        ) : (
          <p className="inline-warning">
            {selectedModel?.name ?? "This model"} declares no preset-capable workflow. Pick another model.
          </p>
        )}
      </div>
    );

}
