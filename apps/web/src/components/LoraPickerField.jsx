import React from "react";
import { LoraPickerSection } from "./LoraPickerSection.jsx";
import {
  MAX_JOB_LORAS_TOTAL,
  loraMatchesModel,
  loraWeight,
  serializeLora,
} from "../presetUtils.js";

// The per-job total cap (sc-8936): this Character Studio picker doesn't distinguish
// builtin vs user LoRAs, so it gates on the hard total the worker enforces.

// Family-filtered LoRA selection shared by the Character Studio Angle Set + Pose
// Library panels (sc-2223). Mirrors the Image/Video Studio behaviour without
// rebuilding the matcher: list only LoRAs whose family matches the active backbone
// (loraMatchesModel, sc-1927), expose a per-LoRA weight that defaults to the
// manifest weight, and emit the serialized `loras` array (serializeLora) the
// character_image payload carries top-level.
export function useLoraSelection(loras, model) {
  const compatibleLoras = React.useMemo(
    () => (Array.isArray(loras) ? loras.filter((lora) => !lora.presetManaged && lora.installState !== "missing" && loraMatchesModel(lora, model)) : []),
    [loras, model],
  );
  const compatibleKey = compatibleLoras.map((lora) => lora.id).join("|");
  const [selectedLoraIds, setSelectedLoraIds] = React.useState([]);
  const [weights, setWeights] = React.useState({});

  // Drop selections that no longer match when the backbone changes (a stale id from
  // another family must not ride along into the payload). Mirrors ImageStudio.
  React.useEffect(() => {
    setSelectedLoraIds((ids) => ids.filter((id) => compatibleLoras.some((lora) => lora.id === id)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compatibleKey]);

  const weightFor = React.useCallback(
    (lora) => {
      const override = Number(weights[lora.id]);
      return Number.isFinite(override) ? override : loraWeight(lora);
    },
    [weights],
  );

  const toggleLora = React.useCallback((lora) => {
    setSelectedLoraIds((ids) => {
      if (ids.includes(lora.id)) {
        return ids.filter((id) => id !== lora.id);
      }
      if (ids.length >= MAX_JOB_LORAS_TOTAL) {
        return ids; // the worker rejects more than MAX_JOB_LORAS_TOTAL per job
      }
      return [...ids, lora.id];
    });
  }, []);

  const setWeight = React.useCallback((id, value) => {
    setWeights((current) => ({ ...current, [id]: Number(value) }));
  }, []);

  const serializedLoras = React.useMemo(
    () =>
      selectedLoraIds
        .map((id) => compatibleLoras.find((lora) => lora.id === id))
        .filter(Boolean)
        .map((lora) => serializeLora(lora, { weight: weightFor(lora) })),
    [selectedLoraIds, compatibleLoras, weightFor],
  );

  return { model, compatibleLoras, selectedLoraIds, toggleLora, weightFor, setWeight, serializedLoras };
}

// Adapt Character/editor selection state to the same picker Image Studio uses.
export function LoraPickerField({ selection, label = "Style LoRAs (optional)", onUpdateLora }) {
  const { model, compatibleLoras, selectedLoraIds, toggleLora, weightFor, setWeight } = selection;
  const selectedLoras = selectedLoraIds.map((id) => compatibleLoras.find((lora) => lora.id === id)).filter(Boolean);
  return (
    <div className="character-lora-picker">
      <LoraPickerSection
        label={label}
        selectedModel={model}
        selectedLoras={selectedLoras}
        selectedLoraIds={selectedLoraIds}
        compatibleLoras={compatibleLoras}
        userSelectedLoraCount={selectedLoras.filter((lora) => lora.scope !== "builtin").length}
        isLoraDisabled={() => selectedLoraIds.length >= MAX_JOB_LORAS_TOTAL}
        toggleLora={toggleLora}
        effectiveLoraWeight={weightFor}
        setLoraWeight={setWeight}
        loraEmptyMessage={model ? `No installed LoRAs match ${model.name ?? model.id}.` : "No model selected"}
        onUpdateLora={onUpdateLora}
      />
    </div>
  );
}
