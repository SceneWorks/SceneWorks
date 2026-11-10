import React from "react";
import { StudioLoraImportPanel } from "../../components/StudioLoraImportPanel.jsx";
import { ModelAttribution } from "../../components/ModelAttribution.jsx";
import { StudioUpdateBadge, StudioUpdateNotice, updateOptionLabel } from "../../components/StudioUpdateNotice.jsx";
import { noPresetId } from "../../presetUtils.js";
import { LoraPickerSection, SavePresetPanel, StyleAxisRow, TierPickerField } from "../../components/generationStudio.jsx";
import { tierLabel } from "../../quantTier.js";
import { STYLE_GROUPS } from "../../data/styleCatalog.js";

export function ImageGenerationSettings({
  selectedModel,
  setModel,
  model,
  pickerModels,
  modeLabel,
  setActiveView,
  createModelDownloadJob,
  mossEdit,
  mossDmd,
  setMossField,
  mossComponents,
  mossDownload,
  mossDownloadProgress,
  downloadMossAcceleration,
  handleResolutionChange,
  resolution,
  resolutionOptions,
  formatResolutionLabel,
  patchGridNotice,
  nativeResolutionNotice,
  setCount,
  count,
  showTierPicker,
  quantTier,
  handleTierChange,
  tierPickerItems,
  tierSwitching,
  tierSuggested,
  confirmTier,
  tierBelowFloor,
  qualityFloor,
  tierExplicit,
  pickerSelectedLoras,
  pickerSelectedLoraIds,
  pickerCompatibleLoras,
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
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {pickerModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {updateOptionLabel(item)}
                    </option>
                  ))}
                </select>
                {!pickerModels.length ? (
                  <span className="field-hint" role="status">
                    No installed model supports {modeLabel} generation.{" "}
                    <button onClick={() => setActiveView("Models")} type="button">
                      Browse models
                    </button>
                  </span>
                ) : null}
                <StudioUpdateNotice item={selectedModel} onUpdate={createModelDownloadJob} />
                {/* Licence-required attribution (MiniMax-H3-Image §IV.2, SW-7100): the generation
                    surface, from the manifest field, as Video Studio does. */}
                <ModelAttribution model={selectedModel} className="studio-model-attribution" />
              </label>
                  {model === "moss_image_sft" && !mossEdit ? <label className="checkline">
                    <input type="checkbox" checked={Boolean(mossDmd)} onChange={(event) => setMossField("acceleration", event.target.checked ? "unmerged_bf16" : "off")} />
                    8-step acceleration
                  </label> : null}
                    {mossDmd && !mossComponents.ready ? <div role="status">
                      <p>8-step package: {mossDownload.busy ? "queueing" : mossComponents.state}. {mossDownloadProgress || "Install the selected acceleration components before generating."}</p>
                      <button type="button" disabled={mossDownload.busy} onClick={downloadMossAcceleration}>Download 8-step components</button>
                    </div> : null}
                    {mossDownload.error ? <p role="alert">{mossDownload.error}</p> : null}
              <label className="settings-field settings-field-aspect">
                Aspect
                <select onChange={(event) => handleResolutionChange(event.target.value)} value={resolution}>
                  {resolutionOptions.map((option) => (
                    <option key={option} value={option}>{formatResolutionLabel(option)}</option>
                  ))}
                </select>
                {/* sc-25681: informational only — the size is never restricted. */}
                {patchGridNotice ? (
                  <span className="field-hint" role="note">
                    {patchGridNotice}
                  </span>
                ) : null}
                {nativeResolutionNotice ? (
                  <span className="field-hint" role="note">
                    {nativeResolutionNotice}
                  </span>
                ) : null}
              </label>
              <label className="settings-field settings-field-count">
                Variations
                <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
              </label>
              {/* Quant tier belongs with the everyday model knobs, not buried in Advanced
                  (studio-cleanup sc-15374): it names the build that will actually load. */}
              {showTierPicker ? (
                <TierPickerField
                  className="settings-field settings-field-tier"
                  value={quantTier}
                  onChange={handleTierChange}
                  items={tierPickerItems}
                  tierSwitching={tierSwitching}
                  tierLabel={tierLabel}
                  suggested={tierSuggested}
                  onConfirm={confirmTier}
                  title="Switch which installed quant tier generates, for A/B comparison. Higher precision = larger memory footprint; switching a heavy tier reloads it before the next generation. Tiers you haven't downloaded are shown but disabled — install them from the Models page to enable."
                  warning={tierBelowFloor ? (
                    <span className="field-hint quant-tier-floor-note">
                      {tierLabel(quantTier)} is below the {tierLabel(qualityFloor)} recommended for{" "}
                      {selectedModel?.name ?? "this model"} — it can look washed or lose fine detail
                      here (quantization error is amplified under CFG).
                      {tierExplicit ? " Your pick is honored." : ""}
                    </span>
                  ) : null}
                />
              ) : null}
            </div>
            <LoraPickerSection
              selectedModel={selectedModel}
              selectedLoras={pickerSelectedLoras}
              selectedLoraIds={pickerSelectedLoraIds}
              compatibleLoras={pickerCompatibleLoras}
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
            {/* Style axis (sc-13135): the Style Catalog picker sits FIRST in this row, followed by the
                model's Style presets — both are style controls, so they share one row instead of the
                catalog picker floating in a standalone row under the composer. The Style Catalog
                composes the prompt (free-text) or merges into the caption (Ideogram, sc-13224); "None"
                resets to pass-through. Hidden for booru-tag models, whose convention the catalog's prose
                entries do not fit. NB: distinct from Krea's numeric "text style" (textStyleGain). */}
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
                />
              )}
            />
          </div>
  );
}
