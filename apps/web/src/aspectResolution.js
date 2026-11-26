// Ratio + megapixel resolution picker (Image Studio). The user picks a SHAPE (an aspect ratio) and a
// SIZE (megapixels); the pixel dimensions are derived from the two and the selected model's own
// envelope, instead of choosing from a fixed list of "WxH" buckets.
//
// The "WxH" resolution string stays the single source of truth for the studio — presets, recipes,
// the reference auto-snap, batch directives and the job payload all keep reading and writing it.
// This module is the pure translation layer between that string and the ratio/MP controls, so the
// picker is a VIEW over the resolution, never a second copy of it that could drift.
//
// The model's megapixel BAND is DECLARED, never guessed: it runs from the smallest to the largest
// preset area the model publishes in `limits.resolutions` (after the per-host memory gate has trimmed
// buckets this machine can't fit, sc-13959). The ceiling is the same number the API's admission
// envelope refuses above for a model that declares one (`admission_geometry.rs`, `max_preset_area`),
// so a derived size is never larger in area — or longer on any side — than a preset the model
// already offers. The floor keeps a model's own quality band: MiniMax-H3-Image's smallest preset is
// 2048² (it renders anything smaller at ~4 MP and resizes back), so a 1024² carried over from
// another model is lifted to 4 MP rather than silently wasting the render.

import { modelDimensionConstraints } from "./resolutionOverride.js";
import { parseResolution } from "./resolutionMatch.js";

// One megapixel, as image tools count it (1024² = 1.00 MP).
export const MEGAPIXEL = 1024 * 1024;

// Smallest size the picker offers to a model that declares no presets. Every side of every standard
// ratio stays above the 256 px API floor at this area (21:9 is the tightest: 768 × 336).
export const MIN_MEGAPIXELS = 0.25;

// Grid rounding can land a derived size a hair under the smallest preset's area (21:9 at the floor);
// such a size is still "at the floor", not below the model's band.
const FLOOR_TOLERANCE = 0.97;

// Default latent grid. Every shipped image family patchifies on a multiple of 16 (VAE ÷8 × patch 2),
// and every declared preset ladder already sits on it; a model that declares a coarser grid
// (`limits.requiresDimensionsMultipleOf`, e.g. Qwen-Image 2.1's 32) uses its own.
const DEFAULT_STRIDE = 16;

export const ASPECT_RATIOS = [
  { id: "1:1", width: 1, height: 1, label: "1:1 (Square)" },
  { id: "2:3", width: 2, height: 3, label: "2:3 (Portrait Photo)" },
  { id: "3:2", width: 3, height: 2, label: "3:2 (Photo)" },
  { id: "3:4", width: 3, height: 4, label: "3:4 (Portrait Standard)" },
  { id: "4:3", width: 4, height: 3, label: "4:3 (Standard)" },
  { id: "9:16", width: 9, height: 16, label: "9:16 (Portrait Widescreen)" },
  { id: "16:9", width: 16, height: 9, label: "16:9 (Widescreen)" },
  { id: "21:9", width: 21, height: 9, label: "21:9 (Ultrawide)" },
];

// How far (in log-aspect) a model preset may sit from a standard ratio and still count as that
// ratio. 8% covers the latent-grid-aligned ladders (1216x832 is "3:2", 1152x896 is "4:3") while a
// genuinely different shape — Mage-Flow's 4:1 panorama — stays its own entry.
const SAME_SHAPE_TOLERANCE = 0.08;

function gcd(a, b) {
  let x = Math.abs(Math.round(a));
  let y = Math.abs(Math.round(b));
  while (y) [x, y] = [y, x % y];
  return x || 1;
}

// The ratio menu for a model: the standard shapes, plus every shape from the model's OWN preset
// ladder that none of them approximates (Mage-Flow's 4:1 and 1:4), so moving from a bucket list to
// a ratio list never takes away a shape a model publishes. Extras are sorted in after the standard
// list, landscape-first by aspect.
export function aspectRatiosFor(options) {
  const extras = [];
  const sizes = (Array.isArray(options) ? options : []).map(parseResolution).filter(Boolean);
  for (const { width, height } of sizes) {
    const aspect = Math.log(width / height);
    const known = [...ASPECT_RATIOS, ...extras].some(
      (ratio) => Math.abs(Math.log(ratio.width / ratio.height) - aspect) < SAME_SHAPE_TOLERANCE,
    );
    if (known) continue;
    const divisor = gcd(width, height);
    const [w, h] = [width / divisor, height / divisor];
    extras.push({ id: `${w}:${h}`, width: w, height: h, label: `${w}:${h} (Model preset)` });
  }
  extras.sort((a, b) => b.width / b.height - a.width / a.height);
  return [...ASPECT_RATIOS, ...extras];
}

export function aspectRatioById(id, ratios = ASPECT_RATIOS) {
  return ratios.find((ratio) => ratio.id === id) ?? null;
}

// The ratio closest to a width/height, compared in log space so portrait and landscape distances
// are symmetric (2:3 is as far from 1:1 as 3:2 is).
export function nearestAspectRatio(width, height, ratios = ASPECT_RATIOS) {
  if (!(width > 0) || !(height > 0)) return ratios[0];
  const target = Math.log(width / height);
  let best = ratios[0];
  let bestDistance = Infinity;
  for (const ratio of ratios) {
    const distance = Math.abs(Math.log(ratio.width / ratio.height) - target);
    if (distance < bestDistance) {
      best = ratio;
      bestDistance = distance;
    }
  }
  return best;
}

export function megapixelsOf(width, height) {
  return (width * height) / MEGAPIXEL;
}

// The selected model's sizing envelope, from its declared geometry and its OFFERED (memory-gated)
// preset ladder `options`:
//   fixed    — the model offers exactly one size (Supra2's 256², SANA's 1024²); no ratio/MP choice.
//   stride   — the grid every derived side lands on.
//   minSide / maxSide — the per-side bounds (declared `minDimension`/`maxDimension`, else 256/4096).
//   maxArea  — the largest offered preset's area; `maxMegapixels` is the same in MP.
export function resolutionEnvelope(model, options) {
  const sizes = (Array.isArray(options) ? options : []).map(parseResolution).filter(Boolean);
  const constraints = modelDimensionConstraints(model);
  const stride = Math.max(DEFAULT_STRIDE, constraints.step > 1 ? constraints.step : 1);
  const largestPresetSide = sizes.length ? Math.max(...sizes.flatMap((size) => [size.width, size.height])) : 0;
  // A side may reach the declared/blanket maximum, but never below the longest preset side the
  // model already offers (a ladder is authoritative about its own members).
  const maxSide = Math.floor(Math.max(constraints.max, largestPresetSide) / stride) * stride;
  const minSide = Math.ceil(constraints.min / stride) * stride;
  const maxArea = sizes.length ? Math.max(...sizes.map((size) => size.width * size.height)) : MEGAPIXEL;
  const minArea = sizes.length ? Math.min(...sizes.map((size) => size.width * size.height)) : MIN_MEGAPIXELS * MEGAPIXEL;
  const fixed = sizes.length === 1;
  return {
    ratios: aspectRatiosFor(sizes.map((size) => `${size.width}x${size.height}`)),
    fixed,
    fixedResolution: fixed ? `${sizes[0].width}x${sizes[0].height}` : null,
    stride,
    declaredStride: constraints.step > 1 ? constraints.step : null,
    minSide,
    maxSide,
    minArea,
    maxArea,
    minMegapixels: minArea / MEGAPIXEL,
    maxMegapixels: maxArea / MEGAPIXEL,
  };
}

// The "WxH" for `ratio` at `megapixels` inside `envelope`. The requested area is clamped to the
// envelope, a side that would overrun `maxSide` shrinks the pair (keeping the ratio), and each side
// lands on the stride by whichever floor/ceil pairing best keeps the shape and size without
// exceeding the area ceiling — so the result never exceeds the largest preset the model offers.
export function resolutionFor(ratio, megapixels, envelope) {
  if (envelope.fixed) return envelope.fixedResolution;
  const shape = typeof ratio === "string" ? aspectRatioById(ratio, envelope.ratios) : ratio;
  if (!shape) return null;
  const { stride, minSide, maxSide, maxArea } = envelope;
  const requested = Number.isFinite(megapixels) ? megapixels : 1;
  const area = Math.min(Math.max(requested, envelope.minMegapixels), envelope.maxMegapixels) * MEGAPIXEL;
  const aspect = shape.width / shape.height;
  let width = Math.sqrt(area * aspect);
  let height = Math.sqrt(area / aspect);
  const overrun = Math.max(width / maxSide, height / maxSide, 1);
  width /= overrun;
  height /= overrun;
  // Every floor/ceil pairing on the grid, scored by how far it strays from the requested shape and
  // area; anything over the ceiling is ineligible. Floor/floor is always eligible (it is no larger
  // than the exact, already-clamped size), so a candidate always exists.
  const clampSide = (value) => Math.min(Math.max(value, minSide), maxSide);
  const sides = (value) => [Math.floor(value / stride) * stride, Math.ceil(value / stride) * stride].map(clampSide);
  const targetAspect = Math.log(aspect);
  let best = null;
  for (const w of sides(width)) {
    for (const h of sides(height)) {
      if (w * h > maxArea) continue;
      // Shape error weighs double: a square must stay square (2048x2048, not a 2048x2080 that is
      // closer in area), while a 1% area shortfall on a ceiling-capped size is invisible.
      const score = 2 * Math.abs(Math.log(w / h) - targetAspect) + Math.abs(Math.log((w * h) / area));
      if (!best || score < best.score) best = { w, h, score };
    }
  }
  const w = best ? best.w : clampSide(Math.floor(width / stride) * stride);
  const h = best ? best.h : clampSide(Math.floor(height / stride) * stride);
  return `${w}x${h}`;
}

// Whether a "WxH" can be kept as-is for this envelope: a fixed-size model accepts only its one size;
// otherwise both sides are inside the side bounds, the area is inside the model's preset band, and —
// for a model that DECLARES a grid — both sides are on it. An undeclared grid is not enforced on
// a kept value, so a restored ladder size from another model (1000x1000) is not rewritten needlessly.
export function resolutionFitsEnvelope(resolution, envelope) {
  if (envelope.fixed) return resolution === envelope.fixedResolution;
  const size = parseResolution(resolution);
  if (!size) return false;
  const { width, height } = size;
  const onGrid = !envelope.declaredStride ||
    (width % envelope.declaredStride === 0 && height % envelope.declaredStride === 0);
  return width >= envelope.minSide && height >= envelope.minSide &&
    width <= envelope.maxSide && height <= envelope.maxSide &&
    width * height <= envelope.maxArea && width * height >= envelope.minArea * FLOOR_TOLERANCE && onGrid;
}

// Carry a resolution into a new envelope: keep its shape (nearest named ratio) and its size, clamped
// to what the model can render. Used when the model changes and the current size no longer fits.
export function fitResolutionToEnvelope(resolution, envelope) {
  const size = parseResolution(resolution);
  if (!size) return null;
  return resolutionFor(
    nearestAspectRatio(size.width, size.height, envelope.ratios),
    megapixelsOf(size.width, size.height),
    envelope,
  );
}

// The megapixel stepper's increment, scaled to the model's range so a 1 MP model and a 36 MP one
// both take a sensible number of clicks end to end.
export function megapixelStep(maxMegapixels) {
  if (maxMegapixels <= 2) return 0.1;
  if (maxMegapixels <= 8) return 0.25;
  return 1;
}

// The next stepper value from `current` in `direction` (+1/-1), aligned to the step grid so a
// derived 0.96 MP steps to 1.00, not 1.06.
export function stepMegapixels(current, direction, envelope) {
  const step = megapixelStep(envelope.maxMegapixels);
  const epsilon = 1e-6;
  const next = direction > 0
    ? (Math.floor(current / step + epsilon) + 1) * step
    : (Math.ceil(current / step - epsilon) - 1) * step;
  return Math.min(Math.max(next, envelope.minMegapixels), envelope.maxMegapixels);
}

export function formatMegapixels(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}
