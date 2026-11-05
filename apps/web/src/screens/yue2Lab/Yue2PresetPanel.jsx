import React from "react";
import { Icon } from "../../components/Icons.jsx";

export function Yue2PresetPanel({
  settings,
  applyPreset,
  update,
  setPresetName,
  presetName,
  savePreset
}) {
  return (
<div className="yue2-presets" data-testid="yue2-presets">
              <span className="eyebrow">Presets</span>
              {settings.presets.map((preset) => (
                <span className="preset-chip yue2-preset" key={preset.id}>
                  <button className="audio-link" onClick={() => applyPreset(preset)} type="button">
                    {preset.name}
                  </button>
                  <span className="yue2-muted">
                    {preset.version} · {preset.license || "licence restricted"}
                    {preset.nonCommercial ? " · Noncommercial" : ""}
                  </span>
                  <button
                    aria-label={`Delete preset ${preset.name}`}
                    className="audio-link"
                    onClick={() => update({ presets: settings.presets.filter((item) => item.id !== preset.id) })}
                    type="button"
                  >
                    <Icon.Close size={12} />
                  </button>
                </span>
              ))}
              <input
                aria-label="Preset name"
                onChange={(event) => setPresetName(event.target.value)}
                placeholder="Preset name"
                value={presetName}
              />
              <button className="secondary-action" disabled={!presetName.trim()} onClick={savePreset} type="button">
                <Icon.Save size={13} /> Save preset
              </button>
            </div>
  );
}
