export function parseResolution(value) {
  if (typeof value !== "string") return null;
  const match = value.match(/^(\d+)x(\d+)$/);
  if (!match) return null;
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

// Pick the option whose aspect ratio is closest to the source. Aspect distance
// dominates; total-pixel distance breaks ties (a 1500×1500 source prefers
// 1024×1024 over 640×640).
export function pickClosestResolution(sourceWidth, sourceHeight, options) {
  if (!Array.isArray(options) || options.length === 0) return null;
  if (!Number.isFinite(sourceWidth) || !Number.isFinite(sourceHeight)) return null;
  if (sourceWidth <= 0 || sourceHeight <= 0) return null;
  const targetAspect = Math.log(sourceWidth / sourceHeight);
  const targetPixels = Math.log(sourceWidth * sourceHeight);
  let best = null;
  let bestScore = null;
  for (const option of options) {
    const dims = parseResolution(option);
    if (!dims) continue;
    const aspectDistance = Math.abs(Math.log(dims.width / dims.height) - targetAspect);
    const pixelDistance = Math.abs(Math.log(dims.width * dims.height) - targetPixels);
    const score = aspectDistance * 100 + pixelDistance;
    if (bestScore === null || score < bestScore) {
      best = option;
      bestScore = score;
    }
  }
  return best;
}
