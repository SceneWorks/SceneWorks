

// Output aspect presets for the editor's canvas-extend / outpaint control (sc-2556).
// "match" keeps the working size, so the fit mode then has no border to act on.
export const EDIT_OUTPUT_ASPECTS = [
  { key: "match", label: "Match canvas", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
  { key: "9:16", label: "9:16", ratio: 9 / 16 },
  { key: "4:3", label: "4:3", ratio: 4 / 3 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "3:2", label: "3:2", ratio: 3 / 2 },
  { key: "2:3", label: "2:3", ratio: 2 / 3 },
];

export function editOutputAspectRatio(key) {
  return EDIT_OUTPUT_ASPECTS.find((aspect) => aspect.key === key)?.ratio ?? null;
}

// Snap an edit-output pixel dimension to a multiple of 16 (min 16). Every image engine
// requires dims divisible by 16 (VAE ×8 · patch ×2 — e.g. mlx-gen z-image's SIZE_MULTIPLE
// guard); an imported/cropped source at arbitrary dims (e.g. 832×1165) would otherwise be
// forwarded verbatim and hard-fail at generation. Mirrors snapCanvasDim's rounding, without
// its blank-canvas [256, 2048] clamp — an edit must not force-grow a small source to 256.
export function alignEditDim(px) {
  return Math.max(16, Math.round(px / 16) * 16);
}

// Output W×H for an editor edit given the target aspect + fit mode, keeping the working
// image at native scale (never upscales). "match"/unknown aspect → working size. crop =
// largest target-aspect rect INSIDE the image (trim the overflow); pad/outpaint =
// smallest target-aspect canvas CONTAINING the image (extend → border to fill). Result dims
// are always snapped to a multiple of 16 so the engine's size guard accepts them; the worker
// crop/pad-fits the source to these dims (never stretches). Pure.
export function editOutputDims(workingW, workingH, aspectKey, fitMode) {
  const ratio = editOutputAspectRatio(aspectKey);
  if (!ratio || !workingW || !workingH) {
    return { width: alignEditDim(workingW), height: alignEditDim(workingH) };
  }
  const imageRatio = workingW / workingH;
  let width;
  let height;
  if (fitMode === "crop") {
    // Cover: shrink to the target aspect within the image (trim).
    if (ratio >= imageRatio) {
      width = workingW;
      height = Math.round(workingW / ratio);
    } else {
      height = workingH;
      width = Math.round(workingH * ratio);
    }
  } else {
    // Pad / outpaint: extend to the target aspect around the image (add border).
    if (ratio >= imageRatio) {
      height = workingH;
      width = Math.round(workingH * ratio);
    } else {
      width = workingW;
      height = Math.round(workingW / ratio);
    }
  }
  return { width: alignEditDim(width), height: alignEditDim(height) };
}

// Predefined crop ratios (width / height). Rotate swaps to the transpose; 1:1 and
// Freeform are unaffected.
export const CROP_RATIOS = [
  { key: "free", label: "Freeform", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "3:4", label: "3:4", ratio: 3 / 4 },
  { key: "5:7", label: "5:7", ratio: 5 / 7 },
  { key: "8:10", label: "8:10", ratio: 8 / 10 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
];

export const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// Resolve a ratio key (+ rotate) to a concrete width/height ratio, or null for
// freeform. Rotating transposes non-square ratios (3:4 → 4:3); 1:1 is a no-op.
export function cropRatioForKey(key, rotated) {
  const found = CROP_RATIOS.find((entry) => entry.key === key);
  const base = found ? found.ratio : null;
  if (base == null || base === 1) return base;
  return rotated ? 1 / base : base;
}

// Largest rect of the given ratio that fits in the image, centered. Freeform
// (null ratio) defaults to a centered 80% box. Returns image-pixel coords.
export function centeredCropRect(imgW, imgH, ratio) {
  if (ratio == null) {
    const w = imgW * 0.8;
    const h = imgH * 0.8;
    return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
  }
  let w = imgW;
  let h = w / ratio;
  if (h > imgH) {
    h = imgH;
    w = h * ratio;
  }
  return { x: (imgW - w) / 2, y: (imgH - h) / 2, width: w, height: h };
}

// The four dim rectangles that mask everything outside the crop rect (image coords).
export function cropOverlayRects(imgW, imgH, rect) {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  return [
    { x: 0, y: 0, width: imgW, height: rect.y },
    { x: 0, y: bottom, width: imgW, height: imgH - bottom },
    { x: 0, y: rect.y, width: rect.x, height: rect.height },
    { x: right, y: rect.y, width: imgW - right, height: rect.height },
  ];
}

// ── Blank-canvas "New layout" (Workstream A, sc-6092) ────────────────────────
// A from-scratch substrate for layout-from-nothing (Ideogram text-to-image). The
// dimensions obey Ideogram's constraints: multiples of 16 within [256, 2048].
export const BLANK_CANVAS_MIN = 256;

export const BLANK_CANVAS_MAX = 2048;

export const BLANK_CANVAS_SIZES = [512, 768, 1024, 1536, 2048];

// Snap a pixel dimension to a multiple of 16 within [256, 2048] (Ideogram limits).
export function snapCanvasDim(px) {
  return clamp(Math.round(px / 16) * 16, BLANK_CANVAS_MIN, BLANK_CANVAS_MAX);
}

// Target W×H for a blank layout from an aspect preset + a long-side size. Both
// dims are multiples of 16 in [256, 2048]. "match"/unknown aspect → square. Pure.
export function blankCanvasDims(aspectKey, longSide) {
  const ratio = editOutputAspectRatio(aspectKey) ?? 1;
  let width;
  let height;
  if (ratio >= 1) {
    width = longSide;
    height = longSide / ratio;
  } else {
    height = longSide;
    width = longSide * ratio;
  }
  return { width: snapCanvasDim(width), height: snapCanvasDim(height) };
}
