// Color-key selection is intentionally canvas-free: the editor uses the same
// selection both for the live preview and the one committed alpha write. Keeping
// it here makes the connected/global semantics testable without Konva or a DOM.

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function srgbToLinear(channel) {
  const value = channel / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

// Oklab is a compact perceptual space. Euclidean distances here track visible
// colour changes much more faithfully than raw RGB channel deltas.
export function rgbaToOklab(r, g, b) {
  const linearR = srgbToLinear(r);
  const linearG = srgbToLinear(g);
  const linearB = srgbToLinear(b);
  const l = Math.cbrt(0.4122214708 * linearR + 0.5363325363 * linearG + 0.0514459929 * linearB);
  const m = Math.cbrt(0.2119034982 * linearR + 0.6806995451 * linearG + 0.1073969566 * linearB);
  const s = Math.cbrt(0.0883024619 * linearR + 0.2817188376 * linearG + 0.6299787005 * linearB);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function oklabDistance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// UI values are 0..100. Oklab's useful RGB-gamut distance is roughly 0..0.5,
// so map that compactly while retaining room for a visibly soft edge.
export function colorKeyDistanceLimit(value) {
  return (clamp(Number(value) || 0, 0, 100) / 100) * 0.5;
}

function alphaMultiplier(distance, tolerance, softness) {
  if (distance <= tolerance) return 0;
  if (softness <= 0 || distance >= tolerance + softness) return 255;
  return Math.round(((distance - tolerance) / softness) * 255);
}

// Return an alpha multiplier for every source pixel. In connected mode the
// matching region is 4-neighbour connected to the eyedropped pixel; equal-colour
// islands on the subject remain untouched. Global mode deliberately visits every
// matching pixel instead.
export function colorKeyAlphaMultipliers(
  rgba,
  width,
  height,
  { x, y, tolerance = 12, softness = 8, global = false } = {},
) {
  const pixelCount = Math.max(0, width * height);
  const multipliers = new Uint8ClampedArray(pixelCount);
  multipliers.fill(255);
  if (!rgba || !pixelCount) return multipliers;

  const seedX = clamp(Math.floor(x), 0, width - 1);
  const seedY = clamp(Math.floor(y), 0, height - 1);
  const seedOffset = (seedY * width + seedX) * 4;
  const target = rgbaToOklab(rgba[seedOffset], rgba[seedOffset + 1], rgba[seedOffset + 2]);
  const threshold = colorKeyDistanceLimit(tolerance);
  const feather = colorKeyDistanceLimit(softness);
  const limit = threshold + feather;
  const distances = new Float32Array(pixelCount);
  const matches = new Uint8Array(pixelCount);

  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    const distance = oklabDistance(target, rgbaToOklab(rgba[offset], rgba[offset + 1], rgba[offset + 2]));
    distances[index] = distance;
    if (distance <= limit) matches[index] = 1;
  }

  const visit = (index) => {
    multipliers[index] = alphaMultiplier(distances[index], threshold, feather);
  };
  if (global) {
    for (let index = 0; index < pixelCount; index += 1) if (matches[index]) visit(index);
    return multipliers;
  }

  const start = seedY * width + seedX;
  if (!matches[start]) return multipliers;
  const seen = new Uint8Array(pixelCount);
  const queue = [start];
  seen[start] = 1;
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    visit(index);
    const px = index % width;
    const py = Math.floor(index / width);
    const neighbors = [index - 1, index + 1, index - width, index + width];
    for (const next of neighbors) {
      if (
        next < 0 ||
        next >= pixelCount ||
        (next === index - 1 && px === 0) ||
        (next === index + 1 && px === width - 1) ||
        (next === index - width && py === 0) ||
        (next === index + width && py === height - 1) ||
        seen[next] ||
        !matches[next]
      ) continue;
      seen[next] = 1;
      queue.push(next);
    }
  }
  return multipliers;
}

// Shared apply-as-alpha seam: callers supply a non-destructive alpha multiplier
// (255 = preserve, 0 = fully cut out). This composes with existing source alpha,
// so a later selection tool such as SAM3 cannot accidentally resurrect pixels.
export function applyAlphaMultipliersInPlace(rgba, multipliers) {
  const count = Math.min(Math.floor(rgba.length / 4), multipliers.length);
  for (let index = 0; index < count; index += 1) {
    const alphaOffset = index * 4 + 3;
    rgba[alphaOffset] = Math.round((rgba[alphaOffset] * multipliers[index]) / 255);
  }
  return rgba;
}

// Apply the editable SAM3 mask as alpha to a source bitmap. `keepSelected` uses
// the white selection directly; remove-selected reverses it. Both paths use the
// same alpha-multiplier seam as color key so existing transparency is preserved.
export function applyMaskSelectionToRgba(rgba, maskAlpha, keepSelected) {
  const multipliers = new Uint8ClampedArray(maskAlpha.length);
  for (let index = 0; index < maskAlpha.length; index += 1) {
    multipliers[index] = keepSelected ? maskAlpha[index] : 255 - maskAlpha[index];
  }
  return applyAlphaMultipliersInPlace(new Uint8ClampedArray(rgba), multipliers);
}

export function applyColorKeyToRgba(rgba, width, height, options) {
  const output = new Uint8ClampedArray(rgba);
  const multipliers = colorKeyAlphaMultipliers(output, width, height, options);
  return { rgba: applyAlphaMultipliersInPlace(output, multipliers), multipliers };
}

// Convert document coordinates into the active layer's source pixel space. This
// inverts the same translate/rotate/scale transform used by the canvas compositor.
export function documentPointToLayerPoint(point, transform = {}) {
  const x = point.x - (transform.x || 0);
  const y = point.y - (transform.y || 0);
  const angle = -((transform.rotation || 0) * Math.PI) / 180;
  const rotatedX = x * Math.cos(angle) - y * Math.sin(angle);
  const rotatedY = x * Math.sin(angle) + y * Math.cos(angle);
  const scaleX = transform.scaleX ?? 1;
  const scaleY = transform.scaleY ?? 1;
  if (!scaleX || !scaleY) return null;
  return { x: rotatedX / scaleX, y: rotatedY / scaleY };
}
