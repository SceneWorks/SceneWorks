import React from "react";
import { StudioLoraImportPanel } from "../../components/StudioLoraImportPanel.jsx";
import { noPresetId } from "../../presetUtils.js";
import { LoraPickerSection, PresetStackPreview, SavePresetPanel, StyleAxisRow } from "../../components/generationStudio.jsx";
import { StyledPromptPreview } from "../../components/StyledPromptPreview.jsx";
import { STYLE_GROUPS } from "../../data/styleCatalog.js";

// Settings tab › LoRAs: the add-on-demand LoRA picker with the in-place Import LoRA form.
export function ImageLoraSettings({
  selectedModel,
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
}) {
  return (
    <div className="image-lora-settings">
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
    </div>
  );
}

// Settings tab › Style & look: the Style axis (catalog, looks, presets, general stack, save as
// preset), then the exact composed prompt the run will send and the general-preset stack preview.
export function ImageStyleSettings({
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
  setPresetScope,
  activeProject,
  stylePreviewActive,
  styledPreviewPrompt,
  promptCharLimit,
  appearanceNegativePreview,
  generalStack,
  composedStack,
  supportsNegativePrompt,
  stackAddsNegative,
  stackAddsCount,
}) {
  return (
    <div className="image-style-settings">
      {/* Style axis (sc-13135): the Style Catalog picker sits FIRST in this row, followed by the
          model's Style presets — both are style controls, so they share one row. The Style
          Catalog composes the prompt (free-text) or merges into the caption (Ideogram, sc-13224);
          "None" resets to pass-through. Hidden for booru-tag models, whose convention the
          catalog's prose entries do not fit. NB: distinct from Krea's numeric "text style"
          (textStyleGain), which lives under Sampling. */}
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
      {/* sc-13131: the EXACT composed prompt (Subject:/Style:, preserved sibling directives, and
          the own-`Style:` MERGE) the run will send once a style is active — reuses
          buildJobRequest so it can never drift from the payload. Hidden when no style applies. */}
      <StyledPromptPreview
        active={stylePreviewActive}
        composedPrompt={styledPreviewPrompt}
        maxChars={promptCharLimit}
        appearanceActive={appearance.active}
        negativePrompt={appearance.active ? appearanceNegativePreview : undefined}
      />
      {/* `stackAddsNegative` is ANDed with the engine's negative-prompt axis (sc-15299), the
          same way VideoStudio does it: the submit path blanks the composed negative for a
          CFG-free engine, so previewing "+ negative prompt" would advertise a contribution
          this job will not make. */}
      <PresetStackPreview
        generalStack={generalStack}
        composed={composedStack}
        stackAddsNegative={supportsNegativePrompt && stackAddsNegative}
        stackAddsCount={stackAddsCount}
      />
    </div>
  );
}
