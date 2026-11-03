import { PRESET_LIMIT, PRESET_EXCLUDED } from "./labConstants.js";
import { useState } from "react";
import { restoreYue2Settings } from "../../yue2Lab.js";

// Extracted from Yue2SongLab along its existing concern boundary.
export function useYue2Presets({
  settings,
  identity,
  update,
  setSettings
}) {
  // Presets: named control snapshots that record the model, version and licence they were made
  // with, stored with the lab's settings (server-side).
  const [presetName, setPresetName] = useState("");

  function savePreset() {
    const name = presetName.trim();
    if (!name) return;
    const snapshot = Object.fromEntries(Object.entries(settings).filter(([key]) => !PRESET_EXCLUDED.has(key)));
    const preset = {
      id: `yue2preset_${Date.now().toString(36)}`,
      name,
      modelId: identity.id,
      version: identity.version,
      license: identity.license,
      experimental: identity.experimental,
      nonCommercial: identity.nonCommercial,
      settings: snapshot,
    };
    update({ presets: [preset, ...settings.presets.filter((item) => item.name !== name)].slice(0, PRESET_LIMIT) });
    setPresetName("");
  }

  function applyPreset(preset) {
    // A preset saved by an earlier build may still carry excluded keys; they never apply.
    const controls = Object.fromEntries(Object.entries(preset.settings).filter(([key]) => !PRESET_EXCLUDED.has(key)));
    // An old preset cannot silently inherit the current run's compute choice.
    if (!("computePolicy" in controls)) controls.computePolicy = controls.precision === "fp32" ? "fp32" : "";
    setSettings((current) => restoreYue2Settings({ ...current, ...controls }));
  }
  return { presetName, setPresetName, savePreset, applyPreset };
}
