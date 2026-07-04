// Pure color-grade math + Konva filter for the Image Editor's Color tool (sc-2439;
// curves + levels sc-6109). Extracted verbatim from ImageEditor.jsx (sc-9752, F-052
// follow-up) so the stateful `useColorGradeTool` hook can share it without importing
// back into ImageEditor.jsx (which would be a cycle). ImageEditor.jsx re-exports the
// tested pieces (`gradePixel`, `applyColorAdjustments`, `isIdentityAdjust`,
// `IDENTITY_COLOR_ADJUST`) to keep its public surface — and its test imports — unchanged.
// Behavior-preserving: the code is byte-identical to the pre-extraction definitions.
import { applyLevels, applyCurves } from "../../colorGrade.js";

// Color-grade controls (sc-2439). Each is a normalized −1..1 slider where 0 is the
// identity; `gradePixel` defines the math. Pure data so the panel + reset are trivial.
export const COLOR_ADJUSTMENTS = [
  { key: "brightness", label: "Brightness" },
  { key: "contrast", label: "Contrast" },
  { key: "saturation", label: "Saturation" },
  { key: "temperature", label: "Temperature" },
];

export const IDENTITY_COLOR_ADJUST = { brightness: 0, contrast: 0, saturation: 0, temperature: 0 };

const clamp8 = (value) => (value < 0 ? 0 : value > 255 ? 255 : Math.round(value));

// True when no grade is applied (all sliders at 0) — lets the preview/Apply skip work.
export function isIdentityAdjust(adjust) {
  return COLOR_ADJUSTMENTS.every(({ key }) => !(adjust?.[key]));
}

// Grade one RGB pixel by the −1..1 adjustments, in a fixed order: temperature
// (warm raises R / lowers B), brightness (additive), contrast (around mid-gray),
// then saturation (blend toward/away from luma). Pure + clamped for unit testing.
export function gradePixel([r, g, b], adjust) {
  const { brightness = 0, contrast = 0, saturation = 0, temperature = 0 } = adjust ?? {};
  r += temperature * 30;
  b -= temperature * 30;
  const add = brightness * 255;
  r += add;
  g += add;
  b += add;
  const cf = 1 + contrast;
  r = (r - 128) * cf + 128;
  g = (g - 128) * cf + 128;
  b = (b - 128) * cf + 128;
  const luma = 0.299 * r + 0.587 * g + 0.114 * b;
  const sf = 1 + saturation;
  r = luma + sf * (r - luma);
  g = luma + sf * (g - luma);
  b = luma + sf * (b - luma);
  return [clamp8(r), clamp8(g), clamp8(b)];
}

// Apply the grade to a flat RGBA buffer in place (alpha untouched). Shared by the
// Konva live-preview filter and the Apply bake, so preview === baked result.
export function applyColorAdjustments(data, adjust) {
  if (isIdentityAdjust(adjust)) return;
  for (let i = 0; i < data.length; i += 4) {
    const [r, g, b] = gradePixel([data[i], data[i + 1], data[i + 2]], adjust);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
}

// Konva custom filter for the live preview — reads the grade from the node's attrs
// (set declaratively by react-konva) and runs the shared math, so preview === bake.
// `gradeMode` selects which grade is previewed (sc-6109): the brightness/contrast
// "adjust", levels, or curves.
export function konvaColorFilter(imageData) {
  const mode = this.getAttr("gradeMode");
  if (mode === "levels") applyLevels(imageData.data, this.getAttr("gradeLevels"));
  else if (mode === "curves") applyCurves(imageData.data, this.getAttr("gradeCurves"));
  else applyColorAdjustments(imageData.data, this.getAttr("colorAdjust"));
}
