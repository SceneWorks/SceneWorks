import React from "react";
import { fitModeOptions } from "../fitMode.js";
export { FIT_MODES, effectiveFitMode, fitModeOptions } from "../fitMode.js";

export function FitModeControl({ value, onChange, inpaintCapable = false, label = "Fit" }) {
  const options = fitModeOptions(inpaintCapable);
  return (
    <div className="fit-mode-field">
      <span className="asset-picker-label">{label}</span>
      <div className="segmented-control" role="group" aria-label="Fit mode">
        {options.map((mode) => (
          <button
            type="button"
            key={mode.id}
            className={value === mode.id ? "active" : ""}
            aria-pressed={value === mode.id}
            title={mode.hint}
            onClick={() => onChange(mode.id)}
          >
            {mode.label}
          </button>
        ))}
      </div>
    </div>
  );
}
