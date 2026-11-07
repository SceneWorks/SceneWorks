import { ltx25VideoModelId, VIDEO_PRESET_MODES } from "./videoTierHelpers.js";
import React from "react";
import { StudioLoraImportPanel } from "../../components/StudioLoraImportPanel.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../../components/StudioUpdateNotice.jsx";
import { STYLE_GROUPS } from "../../data/styleCatalog.js";
import { noPresetId } from "../../presetUtils.js";
import { LoraPickerSection, SavePresetPanel, StyleAxisRow, TierPickerField } from "../../components/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";
import { formatDurationOption } from "../../videoModelLimits.js";
import { ModelAttribution } from "../../components/ModelAttribution.jsx";

export function VideoGenerationSettings({
  selectedModel,
  setUserModel,
  model,
  modelsForMode,
  mode,
  macVideoModels,
  createModelDownloadJob,
  recipeModelNotice,
  setResolution,
  resolution,
  resolutionOptions,
  setDuration,
  setAutoDuration,
  duration,
  durationOptions,
  fps,
  showTierPicker,
  quantTier,
  handleExecutionTierChange,
  tierPickerItems,
  tierSwitching,
  tierHasMemoryRisk,
  selectedLoras,
  selectedLoraIds,
  compatibleLoras,
  userSelectedLoraCount,
  toggleLora,
  effectiveLoraWeight,
  setLoraWeight,
  loraEmptyMessage,
  createLoraDownloadJob,
  activeProject,
  createLoraImportJob,
  models,
  appearance,
  styleAxisAvailable,
  styleId,
  setStyleId,
  selectedPreset,
  availablePresets,
  setSelectedPresetId,
  availableGeneralPresets,
  generalStackIds,
  toggleGeneralPreset,
  presetPromptParts,
  presetLoraDetails,
  presetName,
  setPresetName,
  savingPreset,
  presetSaveMessage,
  setPresetSaveMessage,
  handleSaveAsPreset,
  presetScope,
  setPresetScope
}) {
  return (
<div className="settings-bar">
            <div className="settings-bar-row">
              <label className="settings-field settings-field-model">
                Model
                <StudioUpdateBadge item={selectedModel} />
                <select
                  onChange={(event) => setUserModel(event.target.value)}
                  value={model}
                >
                  {/* Models gated on the selected tab (sc-5716): show only models that serve the
                      active mode, falling back to the full available list if none do (a reduced
                      catalog) so the picker is never empty. */}
                  {(modelsForMode(mode).length ? modelsForMode(mode) : macVideoModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {updateOptionLabel(item)}
                    </option>
                  ))}
                </select>
                <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
                {/* Licence-required attribution (sc-17227 §IV.2, landed on the generation surfaces
                    by sc-17161). The Models card alone is one screen a user may never revisit after
                    installing; this is where the model is used. Reads the manifest field — never a
                    second hard-coded copy of a string a licence specifies. */}
                <ModelAttribution model={selectedModel} className="studio-model-attribution" />
                {recipeModelNotice ? (
                  <span className="field-hint" role="status">
                    This clip was made with “{recipeModelNotice}”, which isn’t installed. Its
                    settings were restored — pick a model to re-run it.
                  </span>
                ) : null}
              </label>
              <label className="settings-field settings-field-aspect">
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " × ")}
                    </option>
                  ))}
                </select>
              </label>
              <label className="settings-field settings-field-count">
                Duration
                {/* A MENU, never a range: `limits.durations` is the model's exact renderable set,
                    and MiniMax-H3 is the case that makes the distinction visible — its fourteen
                    `17n + 5` lattice rungs are the ONLY lengths the checkpoint renders, so a
                    slider would emit durations the engine refuses (15.0s among them, which its own
                    docs advertise and which sits between the last rung and the next). The label
                    carries the frame count because that is what the lattice is counting. */}
                <select
                  onChange={(event) => {
                    setDuration(Number(event.target.value));
                    // A manual selection is an explicit duration request. Disable duration-head
                    // selection immediately so the visible value and the submitted payload agree.
                    if (model === ltx25VideoModelId) setAutoDuration(false);
                  }}
                  value={duration}
                >
                  {durationOptions.map((value) => (
                    <option key={value} value={value}>
                      {formatDurationOption(value, fps)}
                    </option>
                  ))}
                </select>
              </label>
              {/* Quant tier, not the abstract Fast/Balanced/Best segment (studio-cleanup
                  sc-15374): the settings bar names the concrete thing that will load. Quality
                  itself is still a live payload/preset field and moves to Advanced. */}
              {showTierPicker ? (
                <TierPickerField
                  className="settings-field settings-field-tier"
                  value={quantTier}
                  onChange={handleExecutionTierChange}
                  items={tierPickerItems}
                  tierSwitching={tierSwitching}
                  tierLabel={tierLabel}
                  title="Switch which installed MLX quant tier generates. Higher precision uses more memory; switching a heavy tier reloads it before the next generation."
                  warning={tierHasMemoryRisk ? (
                    <span className="field-hint quant-tier-memory-note">
                      Higher MLX video tiers may run out of memory on long or high-resolution clips.
                      Your pick is honored.
                    </span>
                  ) : null}
                />
              ) : null}
            </div>
            <LoraPickerSection
              selectedModel={selectedModel}
              selectedLoras={selectedLoras}
              selectedLoraIds={selectedLoraIds}
              compatibleLoras={compatibleLoras}
              userSelectedLoraCount={userSelectedLoraCount}
              toggleLora={toggleLora}
              effectiveLoraWeight={effectiveLoraWeight}
              setLoraWeight={setLoraWeight}
              loraEmptyMessage={loraEmptyMessage}
              onUpdateLora={createLoraDownloadJob}
              importPanel={(
                <StudioLoraImportPanel
                  activeProject={activeProject}
                  createLoraImportJob={createLoraImportJob}
                  models={models}
                />
              )}
            />
            {/* Style axis (sc-13135): the Style Catalog picker leads this row (hidden for promptless
                image-conditioned models and for booru-tag models), followed by the model's Style
                presets — mirrors the Image Studio. The catalog wraps the outgoing prompt
                (Subject:/Style:); "None" resets. */}
            <StyleAxisRow
              appearance={appearance}
              available={styleAxisAvailable}
              groups={STYLE_GROUPS}
              styleId={styleId}
              onStyleChange={setStyleId}
              selectedPreset={selectedPreset}
              presets={availablePresets}
              onPresetChange={setSelectedPresetId}
              generalPresets={availableGeneralPresets}
              generalStackIds={generalStackIds}
              onToggleGeneral={toggleGeneralPreset}
              noPresetValue={noPresetId}
              presetPromptParts={presetPromptParts}
              presetLoraDetails={presetLoraDetails}
              savePreset={(
                <SavePresetPanel
                  presetName={presetName}
                  setPresetName={setPresetName}
                  savingPreset={savingPreset}
                  presetSaveMessage={presetSaveMessage}
                  setPresetSaveMessage={setPresetSaveMessage}
                  onSave={handleSaveAsPreset}
                  presetScope={presetScope}
                  setPresetScope={setPresetScope}
                  activeProject={activeProject}
                  saveDisabled={!VIDEO_PRESET_MODES.includes(mode)}
                  saveTitle={VIDEO_PRESET_MODES.includes(mode) ? undefined : "Presets are available in Image→Video, Text→Video, or First/Last mode."}
                />
              )}
            />
          </div>
  );
}
