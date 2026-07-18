import React, { useState } from "react";
import { Icon } from "../Icons.jsx";
import { LoraPickerSection } from "../../screens/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";
import { MOTIONS } from "./editorUtils.js";

// The right-hand generation-settings rail (design 2a, epic 12798). Presentational: it
// renders the controls owned by useEditorGeneration (`gen`) plus the contextual header
// + 2x2 action grid supplied by the screen. Mirrors the Video Studio control set so the
// same knobs that generate video elsewhere are available in the editor.
export function GenerationRail({ gen, header, contextActions = [], onGenerate, generateSummary, generateDisabled }) {
  const [showSave, setShowSave] = useState(false);
  const [presetName, setPresetName] = useState("");
  const [saveMsg, setSaveMsg] = useState("");
  const studio = gen.studio;

  async function handleSavePreset() {
    if (!presetName.trim()) {
      return;
    }
    const created = await gen.savePreset(presetName);
    if (created) {
      setSaveMsg(`Saved “${created.name ?? presetName}”`);
      setPresetName("");
      setShowSave(false);
    } else {
      setSaveMsg("Could not save preset.");
    }
  }

  return (
    <div className="ve-rail">
      <div className="ve-rail-head">
        <p className="ve-rail-eyebrow">{header?.eyebrow ?? "No selection"}</p>
        <h3 className="ve-rail-title">{header?.title ?? "Timeline"}</h3>
        <div className="ve-ctx-grid">
          {contextActions.map((action) => (
            <button
              className={`ve-ctx-btn${action.primary ? " primary" : ""}`}
              disabled={action.disabled}
              key={action.id}
              onClick={action.onClick}
              title={action.title ?? action.label}
              type="button"
            >
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="ve-rail-body">
        {/* Model + quality */}
        <div className="ve-section">
          <div className="ve-section-hd">
            <Icon.Model size={14} />
            Model &amp; quality
          </div>
          <label className="ve-field">
            <span className="ve-field-label">Video model</span>
            <select className="ve-select" onChange={(e) => gen.setModel(e.target.value)} value={gen.model}>
              {gen.videoModels.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name ?? item.id}
                </option>
              ))}
            </select>
          </label>
          <div className="ve-seg" role="radiogroup" aria-label="Quality">
            {gen.qualityChoices.map(([value, label]) => (
              <button
                aria-checked={gen.quality === value}
                className={`ve-seg-btn${gen.quality === value ? " active" : ""}`}
                key={value}
                onClick={() => gen.setQuality(value)}
                role="radio"
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Preset */}
        <div className="ve-section">
          <span className="ve-field-label">Preset</span>
          <div className="ve-preset-row">
            <select
              className="ve-select"
              onChange={(e) => studio.setSelectedPresetId(e.target.value)}
              value={studio.selectedPresetId ?? "__no_preset__"}
            >
              <option value="__no_preset__">— No preset —</option>
              {studio.availablePresets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.name ?? preset.id}
                </option>
              ))}
            </select>
            <button
              className="ve-icon-btn"
              onClick={() => setShowSave((open) => !open)}
              title="Save current settings as a preset"
              type="button"
            >
              <Icon.Save size={15} />
            </button>
          </div>
          {showSave ? (
            <div className="ve-save-form">
              <input
                className="ve-input"
                onChange={(e) => setPresetName(e.target.value)}
                placeholder="Preset name"
                value={presetName}
              />
              <button className="ve-mini-btn" disabled={!presetName.trim()} onClick={handleSavePreset} type="button">
                Save
              </button>
            </div>
          ) : null}
          {saveMsg ? <p className="ve-hint">{saveMsg}</p> : null}
        </div>

        {/* Output */}
        <div className="ve-section">
          <div className="ve-section-hd">Output</div>
          <div className="ve-output-grid">
            <label className="ve-field">
              <span className="ve-field-label">Resolution</span>
              <select className="ve-select" onChange={(e) => gen.setResolution(e.target.value)} value={gen.resolution}>
                {gen.resolutionOptions.map((value) => (
                  <option key={value} value={value}>
                    {String(value).replace("x", " × ")}
                  </option>
                ))}
              </select>
            </label>
            <label className="ve-field">
              <span className="ve-field-label">Frame rate</span>
              <select className="ve-select" onChange={(e) => gen.setFps(Number(e.target.value))} value={gen.fps}>
                {gen.fpsOptions.map((value) => (
                  <option key={value} value={value}>
                    {value} fps
                  </option>
                ))}
              </select>
            </label>
            <label className="ve-field">
              <span className="ve-field-label">Duration (s)</span>
              <select className="ve-select" onChange={(e) => gen.setDuration(Number(e.target.value))} value={gen.duration}>
                {gen.durationOptions.map((value) => (
                  <option key={value} value={value}>
                    {value}s
                  </option>
                ))}
              </select>
            </label>
            <label className="ve-field">
              <span className="ve-field-label">Motion · {gen.motionPct}%</span>
              <select className="ve-select" onChange={(e) => gen.setMotion(e.target.value)} value={gen.motion}>
                {MOTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {/* LoRAs (reused Video Studio picker) */}
        <div className="ve-section ve-lora-section">
          <LoraPickerSection
            compatibleLoras={studio.compatibleLoras}
            effectiveLoraWeight={studio.effectiveLoraWeight}
            loraEmptyMessage={studio.loraEmptyMessage}
            selectedLoraIds={studio.selectedLoraIds}
            selectedLoras={studio.selectedLoras}
            selectedModel={gen.selectedModel}
            setLoraWeight={studio.setLoraWeight}
            setShowIncompatibleLoras={studio.setShowIncompatibleLoras}
            showIncompatibleLoras={studio.showIncompatibleLoras}
            toggleLora={studio.toggleLora}
            userSelectedLoraCount={studio.userSelectedLoraCount}
          />
        </div>

        {/* Prompt + seed + negative */}
        <div className="ve-section">
          <label className="ve-field">
            <span className="ve-field-label">Prompt</span>
            <textarea
              className="ve-textarea"
              onChange={(e) => gen.setPrompt(e.target.value)}
              placeholder="Describe the motion to generate…"
              value={gen.prompt}
            />
          </label>
          <label className="ve-field">
            <span className="ve-field-label">Seed</span>
            <div className="ve-preset-row">
              <input
                className="ve-input"
                onChange={(e) => gen.setSeed(e.target.value)}
                placeholder="random"
                value={gen.seed}
              />
              <button className="ve-icon-btn" onClick={gen.randomizeSeed} title="Randomize seed" type="button">
                <Icon.Refresh size={15} />
              </button>
            </div>
          </label>
          <label className="ve-field">
            <span className="ve-field-label">Negative prompt</span>
            <textarea
              className="ve-textarea"
              onChange={(e) => gen.setNegativePrompt(e.target.value)}
              value={gen.negativePrompt}
            />
          </label>
        </div>

        {/* Advanced */}
        <div className="ve-adv">
          <button className="ve-adv-toggle" onClick={() => gen.setAdvancedOpen((v) => !v)} type="button">
            <span className="ve-section-hd">
              <Icon.Sliders size={14} />
              Advanced
            </span>
            <span className="ve-adv-chevron">{gen.advancedOpen ? "▾" : "▸"}</span>
          </button>
          {gen.advancedOpen ? (
            <div className="ve-adv-body">
              <div className="ve-output-grid">
                {gen.showSamplerPicker ? (
                  <label className="ve-field">
                    <span className="ve-field-label">Sampler</span>
                    <select className="ve-select" onChange={(e) => gen.setSampler(e.target.value)} value={gen.sampler}>
                      {gen.samplerOptions.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {gen.showSchedulerPicker ? (
                  <label className="ve-field">
                    <span className="ve-field-label">Scheduler</span>
                    <select className="ve-select" onChange={(e) => gen.setScheduler(e.target.value)} value={gen.scheduler}>
                      {gen.schedulerOptions.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                {gen.showSchedulerPicker && gen.scheduler !== "default" ? (
                  <label className="ve-field">
                    <span className="ve-field-label">Scheduler shift</span>
                    <input
                      className="ve-input"
                      max="10"
                      min="0.1"
                      onChange={(e) => gen.setSchedulerShift(Number(e.target.value))}
                      step="0.1"
                      type="number"
                      value={gen.schedulerShift}
                    />
                  </label>
                ) : null}
                <label className="ve-field">
                  <span className="ve-field-label">Steps</span>
                  <input
                    className="ve-input"
                    disabled={gen.lightningActive}
                    onChange={(e) => gen.setStepsOverride(e.target.value)}
                    placeholder={gen.lightningActive ? "4 (Lightning)" : "auto"}
                    value={gen.lightningActive ? "" : gen.stepsOverride}
                  />
                </label>
                <label className="ve-field">
                  <span className="ve-field-label">Guidance</span>
                  <input
                    className="ve-input"
                    disabled={gen.lightningActive}
                    onChange={(e) => gen.setGuidanceOverride(e.target.value)}
                    placeholder={gen.lightningActive ? "off (Lightning)" : "auto"}
                    value={gen.lightningActive ? "" : gen.guidanceOverride}
                  />
                </label>
              </div>

              {gen.showTierPicker ? (
                <div className="ve-field">
                  <span className="ve-field-label">Generation tier</span>
                  <div className="ve-chip-row">
                    {gen.availableTiers.map((tier) => (
                      <button
                        className={`ve-chip${gen.quantTier === tier ? " active" : ""}`}
                        key={tier}
                        onClick={() => gen.setQuantTier(tier)}
                        title={tierLabel(tier)}
                        type="button"
                      >
                        {tier.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}

              {gen.showLightning ? (
                <button
                  className="ve-toggle-row"
                  onClick={() => gen.setLightning((v) => !v)}
                  type="button"
                >
                  <span className="ve-toggle-text">
                    <strong>Lightning (4-step)</strong>
                    <small>Wan 2.2 A14B fast distilled</small>
                  </span>
                  <span className={`ve-switch${gen.lightning ? " on" : ""}`} aria-hidden="true">
                    <span className="ve-switch-knob" />
                  </span>
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      <div className="ve-rail-foot">
        <button className="ve-generate" disabled={generateDisabled} onClick={onGenerate} type="button">
          <Icon.Stars size={16} />
          Generate
        </button>
        <p className="ve-gen-summary">{generateSummary}</p>
      </div>
    </div>
  );
}
