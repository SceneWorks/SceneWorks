import React from "react";

import { LoraPickerSection } from "../generationStudio.jsx";

export const EditorLoraPanel = React.memo(function EditorLoraPanel({
  selectedModel,
  selectedLoras,
  selectedLoraIds,
  toggleLora,
  weightFor,
  setWeight,
  availableLoras,
  showIncompatible,
  setShowIncompatible,
  onUpdateLora,
}) {
  if (!availableLoras.length) return null;
  return (
    <div className="ie-section">
      <LoraPickerSection
        selectedModel={selectedModel}
        selectedLoras={selectedLoras}
        selectedLoraIds={selectedLoraIds}
        compatibleLoras={availableLoras}
        userSelectedLoraCount={selectedLoras.filter((lora) => lora.scope !== "builtin").length}
        showIncompatibleLoras={showIncompatible}
        setShowIncompatibleLoras={setShowIncompatible}
        toggleLora={toggleLora}
        effectiveLoraWeight={weightFor}
        setLoraWeight={setWeight}
        loraEmptyMessage="No LoRAs applied. Add a style or subject LoRA to steer the edit."
        onUpdateLora={onUpdateLora}
        showIncompatibleControl={false}
      />
    </div>
  );
});
