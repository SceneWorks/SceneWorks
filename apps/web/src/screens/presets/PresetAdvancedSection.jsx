import React from "react";
import { SAMPLER_LABELS, SCHEDULER_LABELS } from "../../samplerOptions.js";
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
        <label className="field">
          <span>Steps</span>
          <input
            disabled={!editable}
            max="200"
            min="1"
            onChange={(event) => updateField("steps", event.target.value)}
            placeholder="model default"
            type="number"
            value={form.steps}
          />
        </label>
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
        {samplerMenu.length > 1 ? (
          <label className="field">
            <span>Sampler</span>
            <select disabled={!editable} onChange={(event) => updateField("sampler", event.target.value)} value={form.sampler}>
              <option value="">No default</option>
              {samplerMenu.map((value) => (
                <option key={value} value={value}>
                  {SAMPLER_LABELS[value] ?? value}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {schedulerMenu.length > 1 ? (
          <label className="field">
            <span>Scheduler</span>
            <select disabled={!editable} onChange={(event) => updateField("scheduler", event.target.value)} value={form.scheduler}>
              <option value="">No default</option>
              {schedulerMenu.map((value) => (
                <option key={value} value={value}>
                  {SCHEDULER_LABELS[value] ?? value}
                </option>
              ))}
            </select>
          </label>
        ) : null}
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
