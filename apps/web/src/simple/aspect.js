// Aspect labels for the Simple UI's "Size & aspect" picker (design handoff).
//
// The design's picker leads each tile with a shape name ("16:9") and keeps the exact
// pixels underneath ("1344×768"). The exact reduced ratio is NOT usable as that name:
// the generation buckets models declare are latent-grid-aligned, not clean ratios —
// 1344×768 reduces to 7:4, 1216×832 to 19:13, 896×1152 to 7:9. Those are correct and
// meaningless.
//
// So the label is the NEAREST name from the curated list below, chosen by relative
// difference, with the exact dimensions always shown beside it. That is what image tools
// do, and it makes the label a shape cue rather than a measurement — the measurement is
// the sub-label. It resolves 4 of the design's 6 sizes to exactly the names the mockup
// shows; the other two (1152×896, 896×1152) land on 5:4 / 4:5, which are strictly closer
// to those pixel counts than the mockup's 4:3 / 3:4.

const NAMED_ASPECTS = [
  [1, 1],
  [5, 4],
  [4, 5],
  [4, 3],
  [3, 4],
  [3, 2],
  [2, 3],
  [16, 10],
  [10, 16],
  [16, 9],
  [9, 16],
  [21, 9],
  [9, 21],
];

/**
 * Split a "WIDTHxHEIGHT" string into numbers, or null when it isn't one.
 * @param {string} resolution
 */
export function parseSize(resolution) {
  const [width, height] = String(resolution ?? "")
    .toLowerCase()
    .split("x")
    .map((part) => Number.parseInt(part, 10));
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }
  return { width, height };
}

/**
 * The nearest curated aspect name for a width/height, e.g. "16:9".
 * @param {number} width
 * @param {number} height
 */
export function nearestAspectName(width, height) {
  const ratio = width / height;
  let best = NAMED_ASPECTS[0];
  let bestDelta = Infinity;
  for (const [w, h] of NAMED_ASPECTS) {
    // Relative difference, so a 21:9 candidate isn't penalized for its larger magnitude.
    const delta = Math.abs(ratio - w / h) / (w / h);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = [w, h];
    }
  }
  return `${best[0]}:${best[1]}`;
}

/**
 * The picker tile's two lines for a resolution string: `{ label, sub }`. An unparseable
 * value degrades to the raw string with no sub-label rather than inventing a shape.
 * @param {string} resolution
 */
export function describeResolution(resolution) {
  const size = parseSize(resolution);
  if (!size) {
    return { label: String(resolution ?? ""), sub: "" };
  }
  return {
    label: nearestAspectName(size.width, size.height),
    sub: `${size.width}×${size.height}`,
  };
}

/** The one-line form the settings-bar trigger shows: "16:9 · 1344×768". */
export function resolutionSummary(resolution) {
  const { label, sub } = describeResolution(resolution);
  return sub ? `${label} · ${sub}` : label;
}
