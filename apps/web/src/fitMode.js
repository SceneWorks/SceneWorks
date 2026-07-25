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

export function fitModeOptions(inpaintCapable) {
  return FIT_MODES.filter((mode) => !mode.inpaintOnly || inpaintCapable);
}

export function effectiveFitMode(value, inpaintCapable) {
  return fitModeOptions(inpaintCapable).some((mode) => mode.id === value) ? value : "crop";
}
