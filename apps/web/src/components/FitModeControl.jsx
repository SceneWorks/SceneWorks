import React from "react";

// Image-edit fit modes (epic 2551). The product rule is "never distort", so "stretch"
// is intentionally never offered here. "outpaint" fits the long edge then GENERATES the
// remaining border, so it only applies to inpaint-capable models (`image_inpaint`);
// other models fall back to pad, so it's hidden for them.
export const FIT_MODES = [
  { id: "crop", label: "Crop", hint: "Fill the frame, trim the edges — no distortion" },
  { id: "pad", label: "Pad", hint: "Fit the whole image, neutral bars — no distortion" },
  {
    id: "outpaint",
    label: "Outpaint",
    hint: "Fit the whole image, then generate the rest (inpaint-capable models)",
    inpaintOnly: true,
  },
];

// The options selectable for a model — outpaint only when it accepts an inpaint mask.
export function fitModeOptions(inpaintCapable) {
  return FIT_MODES.filter((mode) => !mode.inpaintOnly || inpaintCapable);
}

// Coerce a stored value to one currently selectable: a persisted "outpaint" must not
// stick when the active model can't inpaint (it would silently fall back to pad), and an
// unknown value resolves to the crop default. Pure — used for both display and payload.
export function effectiveFitMode(value, inpaintCapable) {
  return fitModeOptions(inpaintCapable).some((mode) => mode.id === value) ? value : "crop";
}

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
