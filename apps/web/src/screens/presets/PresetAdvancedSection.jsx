import { SamplerSchedulerFields, SamplingStepsField } from "../../components/SamplerSchedulerFields.jsx";
import React from "react";

import { compactModeList } from "../../presetUtils.js";

export function PresetAdvancedSection({
  activeSegment,
  editable,
  form,
  samplerMenu,
  schedulerMenu,
  updateField,
}) {

    return (
      <div className="advanced-panel">
        <label className="field prompt-field">
          <span>Negative prompt</span>
          <input
            disabled={!editable}
            onChange={(event) => updateField("negativePrompt", event.target.value)}
            placeholder="oversaturated, hands, text, watermark"
            value={form.negativePrompt}
          />
        </label>
        <SamplingStepsField
          value={form.steps}
          onChange={(value) => updateField("steps", value)}
          inputProps={{
            disabled: !editable,
            max: "200",
            min: "1",
            placeholder: "model default",
            type: "number"
          }}
          variant="preset"
        />
        <label className="field">
          <span>Guidance</span>
          <input
            disabled={!editable}
            max="60"
            min="0"
            onChange={(event) => updateField("guidanceScale", event.target.value)}
            placeholder="model default"
            step="0.1"
            type="number"
            value={form.guidanceScale}
          />
        </label>
        <SamplerSchedulerFields
          sampler={form.sampler}
          scheduler={form.scheduler}
          samplerOptions={samplerMenu}
          schedulerOptions={schedulerMenu}
          onSamplerChange={(value) => updateField("sampler", value)}
          onSchedulerChange={(value) => updateField("scheduler", value)}
          variant="preset"
          disabled={!editable}
          emptyLabel="No default"
        />
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
        <label className="field">
          <span>Derived modes</span>
          <input disabled readOnly value={compactModeList(activeSegment.workflow)} />
        </label>
      </div>
    );

}
