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

import { parseResolution } from "./resolutionMatch.js";

// One megapixel, as image tools count it (1024² = 1.00 MP).
export const MEGAPIXEL = 1024 * 1024;

// Smallest size the picker offers to a model that declares no presets. Every side of every standard
// ratio stays above the 256 px API floor at this area (21:9 is the tightest: 768 × 336).
export const MIN_MEGAPIXELS = 0.25;

// Grid rounding can land a derived size a hair under the smallest preset's area (21:9 at the floor);
// such a size is still "at the floor", not below the model's band.
const FLOOR_TOLERANCE = 0.97;

// Grid for a model with no presets to read one from.
const DEFAULT_STRIDE = 32;
// The coarsest grid read off a ladder. Every engine lattice is 8, 16 or 32 (the 256 Supra2 lattice
// belongs to a fixed-size model), so capping at 64 never drops below the engine's own grid.
const MAX_STRIDE = 64;

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

// A positive integer `limits[key]`, or null — the same typo tolerance as resolutionOverride.js.
function declaredLimit(model, key) {
  const value = Number(model?.limits?.[key]);
  return Number.isInteger(value) && value > 0 ? value : null;
}

// The selected model's sizing envelope, from its declared geometry and its OFFERED (memory-gated)
// preset ladder `options`. Everything here is read from facts the worker already proves for every
// shipped preset on every backend (`shipped_image_geometry_is_within_the_pinned_engine_envelope`):
// each preset sits inside the PINNED engine's [min_size, max_size] and on its request grid. The web
// cannot see the engine, so it derives only what those proofs make safe:
//   fixed    — the model offers exactly one size (Supra2's 256², SANA's 1024²); no ratio/MP choice.
//   stride   — the greatest common divisor of the preset sides (capped at 64), raised to a declared
//              `requiresDimensionsMultipleOf`. Every preset is on the engine grid, so their GCD is a
//              multiple of it on EVERY backend — SDXL/Kolors land on 64 (MLX enforces ÷32 there),
//              SenseNova on 32, Krea on 16 — with no per-backend table to drift.
//   minSide / maxSide — a declared `minDimension`/`maxDimension`, else the extent of the presets.
//              Engine side caps are not in the catalog and are often tighter than the 4096 API bound
//              (Krea 2, Z-Image and SenseNova engines all stop at 2048 a side; SenseNova clamps a
//              longer side and renders the wrong shape), so an undeclared model never goes past the
//              longest — or below the shortest — side it publishes.
//   minArea / maxArea — the smallest / largest preset area (the megapixel band).
export function resolutionEnvelope(model, options) {
  const sizes = (Array.isArray(options) ? options : []).map(parseResolution).filter(Boolean);
  const presetSides = sizes.flatMap((size) => [size.width, size.height]);
  const declaredStride = declaredLimit(model, "requiresDimensionsMultipleOf");
  const ladderGrid = presetSides.length
    ? gcd(presetSides.reduce((acc, side) => gcd(acc, side)), MAX_STRIDE)
    : DEFAULT_STRIDE;
  const stride = declaredStride && declaredStride > 1
    ? (ladderGrid * declaredStride) / gcd(ladderGrid, declaredStride)
    : ladderGrid;
  const declaredMin = declaredLimit(model, "minDimension");
  const declaredMax = declaredLimit(model, "maxDimension");
  const shortestPreset = presetSides.length ? Math.min(...presetSides) : 256;
  const longestPreset = presetSides.length ? Math.max(...presetSides) : 2048;
  const minSide = Math.ceil((declaredMin ?? shortestPreset) / stride) * stride;
  const maxSide = Math.floor((declaredMax ?? longestPreset) / stride) * stride;
  const maxArea = sizes.length ? Math.max(...sizes.map((size) => size.width * size.height)) : MEGAPIXEL;
  const minArea = sizes.length ? Math.min(...sizes.map((size) => size.width * size.height)) : MIN_MEGAPIXELS * MEGAPIXEL;
  const fixed = sizes.length === 1;
  const envelope = {
    ratios: aspectRatiosFor(sizes.map((size) => `${size.width}x${size.height}`)),
    fixed,
    fixedResolution: fixed ? `${sizes[0].width}x${sizes[0].height}` : null,
    stride,
    minSide,
    maxSide,
    minArea,
    maxArea,
    minMegapixels: minArea / MEGAPIXEL,
    maxMegapixels: maxArea / MEGAPIXEL,
  };
  // A ratio the side range cannot draw is disabled, never offered as a label over another shape:
  // Z-Image's presets span 720..1280 a side, so its "21:9" could only ever render as 1280x720 (16:9).
  envelope.ratios = envelope.ratios.map((ratio) => ({ ...ratio, available: fixed || ratioDrawable(ratio, envelope) }));
  envelope.availableRatios = envelope.ratios.filter((ratio) => ratio.available);
  return envelope;
}

// Whether `ratio` can be drawn as itself somewhere in the band (checked at the ceiling, where the
// side range is widest relative to the grid), within the same tolerance that names a preset by a
// ratio — so a published preset (SDXL's 896x1152 is its "3:4") never loses its shape.
function ratioDrawable(ratio, envelope) {
  const size = parseResolution(resolutionFor(ratio, envelope.maxMegapixels, envelope));
  return Boolean(size) &&
    Math.abs(Math.log(size.width / size.height) - Math.log(ratio.width / ratio.height)) < SAME_SHAPE_TOLERANCE;
}

// The "WxH" for `ratio` at `megapixels` inside `envelope`. The requested area is clamped to the band;
// a side under `minSide` grows the PAIR and a side over `maxSide` shrinks the pair, so the shape is
// kept rather than one side being clamped alone (Mage-Flow's 4:1 at its floor is 2048x512, not a
// 1024x512 that is really 2:1). Each side then lands on the stride by whichever floor/ceil pairing
// best keeps the shape and size without exceeding the area ceiling.
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
  const underrun = Math.max(minSide / width, minSide / height, 1);
  width *= underrun;
  height *= underrun;
  // The side ceiling wins over the floor when a ratio cannot satisfy both (a shape longer than
  // maxSide / minSide); the clamp below then keeps every side legal.
  const overrun = Math.max(width / maxSide, height / maxSide, 1);
  width /= overrun;
  height /= overrun;
  // Growing to the side floor can need more area than the band's ceiling at that shape; the
  // floor-satisfying size is then itself the ceiling.
  const ceiling = underrun > 1
    ? Math.max(maxArea, Math.ceil(width / stride) * stride * Math.ceil(height / stride) * stride)
    : maxArea;
  // Every floor/ceil pairing on the grid, scored by how far it strays from the requested shape and
  // area; anything over the ceiling is ineligible. Floor/floor is always eligible (it is no larger
  // than the exact, already-clamped size), so a candidate always exists.
  const clampSide = (value) => Math.min(Math.max(value, minSide), maxSide);
  const sides = (value) => [Math.floor(value / stride) * stride, Math.ceil(value / stride) * stride].map(clampSide);
  const targetAspect = Math.log(aspect);
  // Scored against the size AFTER the side floor/ceiling scaling, not the requested area.
  const targetArea = width * height;
  let best = null;
  for (const w of sides(width)) {
    for (const h of sides(height)) {
      if (w * h > ceiling) continue;
      // Shape error weighs double: a square must stay square (2048x2048, not a 2048x2080 that is
      // closer in area), while a 1% area shortfall on a ceiling-capped size is invisible.
      const score = 2 * Math.abs(Math.log(w / h) - targetAspect) + Math.abs(Math.log((w * h) / targetArea));
      if (!best || score < best.score) best = { w, h, score };
    }
  }
  const w = best ? best.w : clampSide(Math.floor(width / stride) * stride);
  const h = best ? best.h : clampSide(Math.floor(height / stride) * stride);
  return `${w}x${h}`;
}

// Whether a "WxH" can be kept as-is for this envelope: a fixed-size model accepts only its one size;
// otherwise both sides are inside the side bounds and on the model's grid, and the area is inside its
// preset band. The grid is enforced on a kept value too: a size carried over from another model
// (Krea's 880x1184) is off SDXL's MLX ÷32 lattice and must be refitted, not handed to the engine.
export function resolutionFitsEnvelope(resolution, envelope) {
  if (envelope.fixed) return resolution === envelope.fixedResolution;
  const size = parseResolution(resolution);
  if (!size) return false;
  const { width, height } = size;
  const onGrid = width % envelope.stride === 0 && height % envelope.stride === 0;
  const inside = width >= envelope.minSide && height >= envelope.minSide &&
    width <= envelope.maxSide && height <= envelope.maxSide &&
    width * height <= envelope.maxArea && onGrid;
  if (!inside) return false;
  if (width * height >= envelope.minArea * FLOOR_TOLERANCE) return true;
  // On a coarse grid the best size at the floor can sit a few percent under it (1216x1216 against a
  // 1.5 MP floor on a 64 px grid); the size the floor itself produces for this shape still fits.
  const shape = nearestAspectRatio(width, height, envelope.availableRatios ?? envelope.ratios);
  return resolutionFor(shape, envelope.minMegapixels, envelope) === resolution;
}

// Carry a resolution into a new envelope: keep its shape (nearest named ratio) and its size, clamped
// to what the model can render. Used when the model changes and the current size no longer fits.
export function fitResolutionToEnvelope(resolution, envelope) {
  const size = parseResolution(resolution);
  if (!size) return null;
  return resolutionFor(
    nearestAspectRatio(size.width, size.height, envelope.availableRatios ?? envelope.ratios),
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
