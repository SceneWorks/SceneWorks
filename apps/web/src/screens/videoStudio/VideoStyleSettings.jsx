import { VIDEO_PRESET_MODES } from "./videoTierHelpers.js";
import React from "react";
import { STYLE_GROUPS } from "../../data/styleCatalog.js";
import { noPresetId } from "../../presetUtils.js";
import { PresetStackPreview, SavePresetPanel, StyleAxisRow } from "../../components/generationStudio.jsx";
import { StyledPromptPreview } from "../../components/StyledPromptPreview.jsx";

// Settings tab › Style & look: the Style Catalog row, the model's style presets, the general
// preset stack and Save as preset — then the two previews of what the run will actually send.
export function VideoStyleSettings({
  mode,
  activeProject,
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
  styleApplied,
  composedStylePrompt,
  styledNegativePrompt,
  generalStack,
  composedStack,
  stackAddsNegative,
  stackAddsCount,
}) {
  return (
    <div className="studio-grid video-settings-style">
      <div className="span-full video-settings-style-axis">
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

      {/* sc-13136: the EXACT composed prompt the run will send once a style is active — recomputed
          from the same base the submit uses so it can never drift. Hidden when no style applies. */}
      <div className="span-full video-settings-previews">
        <StyledPromptPreview active={styleApplied} composedPrompt={composedStylePrompt} appearanceActive={appearance.active} negativePrompt={styledNegativePrompt} />

        {/* `stackAddsNegative` is ANDed with the engine's negative-prompt axis (sc-8445) by the
            caller: this panel is captioned "Prompt sent", so a `Negative:` line for a CFG-free
            model would assert something that is NOT sent. */}
        <PresetStackPreview
          generalStack={generalStack}
          composed={composedStack}
          stackAddsNegative={stackAddsNegative}
          stackAddsCount={stackAddsCount}
        />
      </div>
    </div>
  );
}
