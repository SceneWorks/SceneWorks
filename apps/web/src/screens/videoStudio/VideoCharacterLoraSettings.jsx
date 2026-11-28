import React from "react";
import { CharacterLookFields } from "../../components/StudioFields.jsx";
import { StudioLoraImportPanel } from "../../components/StudioLoraImportPanel.jsx";
import { LoraPickerSection } from "../../components/generationStudio.jsx";

// Settings tab › Character & LoRAs: the character + look (formerly in Advanced) first, then the
// LoRA picker with its import panel (formerly in the settings bar).
export function VideoCharacterLoraSettings({
  setCharacterId,
  characterId,
  characters,
  setCharacterLookId,
  characterLookId,
  selectedModel,
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
}) {
  return (
    <div className="studio-grid video-settings-loras">
      <CharacterLookFields characters={characters} characterId={characterId} lookId={characterLookId}
        onCharacterChange={setCharacterId} onLookChange={setCharacterLookId} />
      {characterId ? (
        <div className="guidance-strip span-2">
          <strong>Character reference</strong>
          <span>
            Character and look are saved with the recipe; LTX image conditioning uses IC-LoRA when the selected preset includes one.
          </span>
        </div>
      ) : null}
      <div className="span-full video-settings-lora-picker">
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
      </div>
    </div>
  );
}
