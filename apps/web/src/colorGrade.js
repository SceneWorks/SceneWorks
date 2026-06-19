// Curves + levels color grading for the Image Editor (sc-6109, Workstream F of
// epic 6087). Pure, LUT-based per-channel tone math — no React/DOM beyond the flat
// RGBA buffer it mutates. Shared by the live Konva preview and the Apply bake, so
// the preview is exactly the baked result (same discipline as the brightness/
// contrast grade in ImageEditor.jsx).
//
// Channels: a grade carries a `master` map plus per-channel `r`/`g`/`b` maps.
// Composition is per-channel FIRST, then master on top: out = masterLut[channelLut[v]].

const CHANNELS = ["master", "r", "g", "b"];

function clampInt(v, lo, hi) {
  v = Math.round(v);
  return v < lo ? lo : v > hi ? hi : v;
}

// An identity 256→256 LUT (out === in).
function identityLut() {
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v += 1) lut[v] = v;
  return lut;
}

// Apply a {master,r,g,b} set of 256-entry LUTs to a flat RGBA buffer in place
// (alpha untouched): per-channel LUT first, then the master LUT on its output.
export function applyChannelLuts(data, { master, r, g, b }) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = master[r[data[i]]];
    data[i + 1] = master[g[data[i + 1]]];
    data[i + 2] = master[b[data[i + 2]]];
  }
}

// ── Levels ────────────────────────────────────────────────────────────────
// Per channel: input black point, white point, and gamma (midtone). The default
// is the identity (black 0, white 255, gamma 1).
const IDENTITY_LEVELS_CHANNEL = { black: 0, white: 255, gamma: 1 };

export const IDENTITY_LEVELS = {
  master: { ...IDENTITY_LEVELS_CHANNEL },
  r: { ...IDENTITY_LEVELS_CHANNEL },
  g: { ...IDENTITY_LEVELS_CHANNEL },
  b: { ...IDENTITY_LEVELS_CHANNEL },
};

export function isIdentityLevelsChannel(ch) {
  const { black = 0, white = 255, gamma = 1 } = ch ?? {};
  return black === 0 && white === 255 && gamma === 1;
}

export function isIdentityLevels(levels) {
  return CHANNELS.every((c) => isIdentityLevelsChannel(levels?.[c]));
}

// LUT[256] for one channel's input black/white/gamma. Values below black map to 0,
// above white to 255; the span is gamma-corrected (gamma>1 lifts midtones).
export function levelsChannelLut(channel) {
  const lut = new Uint8ClampedArray(256);
  const black = clampInt(channel?.black ?? 0, 0, 254);
  const white = clampInt(channel?.white ?? 255, black + 1, 255);
  const gamma = Math.min(9.99, Math.max(0.01, channel?.gamma ?? 1));
  const invGamma = 1 / gamma;
  const span = white - black;
  for (let v = 0; v < 256; v += 1) {
    let t = (v - black) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    lut[v] = Math.round(255 * Math.pow(t, invGamma));
  }
  return lut;
}

// Apply the full levels grade (master + per-channel) to an RGBA buffer in place.
export function applyLevels(data, levels) {
  if (isIdentityLevels(levels)) return;
  applyChannelLuts(data, {
    master: levelsChannelLut(levels.master),
    r: levelsChannelLut(levels.r),
    g: levelsChannelLut(levels.g),
    b: levelsChannelLut(levels.b),
  });
}

// ── Curves ──────────────────────────────────────────────────────────────────
// A tone curve is an ordered list of control points {x,y} in [0,255]; the default
// is the identity diagonal. Interpolated with monotone cubic (Fritsch–Carlson) so
// the curve is smooth AND never overshoots / reverses between points.
export const IDENTITY_CURVE = [
  { x: 0, y: 0 },
  { x: 255, y: 255 },
];

export const IDENTITY_CURVES = {
  master: IDENTITY_CURVE.map((p) => ({ ...p })),
  r: IDENTITY_CURVE.map((p) => ({ ...p })),
  g: IDENTITY_CURVE.map((p) => ({ ...p })),
  b: IDENTITY_CURVE.map((p) => ({ ...p })),
};

export function isIdentityCurve(points) {
  if (!Array.isArray(points) || points.length !== 2) return false;
  return points[0].x === 0 && points[0].y === 0 && points[1].x === 255 && points[1].y === 255;
}

export function isIdentityCurves(curves) {
  return CHANNELS.every((c) => isIdentityCurve(curves?.[c]));
}

// Sort + de-dupe control points by x (a later point at the same x wins), clamped
// to [0,255]. Returns a fresh array.
export function normalizeCurvePoints(points) {
  const byX = new Map();
  for (const p of points ?? []) {
    byX.set(clampInt(p.x, 0, 255), clampInt(p.y, 0, 255));
  }
  return [...byX.entries()].sort((a, b) => a[0] - b[0]).map(([x, y]) => ({ x, y }));
}

// Build a 256-entry LUT from control points via monotone cubic interpolation.
// Fewer than two distinct points → identity.
export function buildCurveLut(points) {
  const pts = normalizeCurvePoints(points);
  const lut = new Uint8ClampedArray(256);
  if (pts.length < 2) return identityLut();

  const n = pts.length;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  // Secant slopes + monotone tangents (Fritsch–Carlson).
  const delta = new Array(n - 1);
  for (let i = 0; i < n - 1; i += 1) delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
  const m = new Array(n);
  m[0] = delta[0];
  m[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i += 1) m[i] = (delta[i - 1] + delta[i]) / 2;
  for (let i = 0; i < n - 1; i += 1) {
    if (delta[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
    } else {
      const a = m[i] / delta[i];
      const b = m[i + 1] / delta[i];
      const s = a * a + b * b;
      if (s > 9) {
        const tau = 3 / Math.sqrt(s);
        m[i] = tau * a * delta[i];
        m[i + 1] = tau * b * delta[i];
      }
    }
  }

  let seg = 0;
  for (let v = 0; v < 256; v += 1) {
    if (v <= xs[0]) {
      lut[v] = Math.round(ys[0]);
      continue;
    }
    if (v >= xs[n - 1]) {
      lut[v] = Math.round(ys[n - 1]);
      continue;
    }
    while (seg < n - 1 && v > xs[seg + 1]) seg += 1;
    const h = xs[seg + 1] - xs[seg];
    const t = (v - xs[seg]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    lut[v] = Math.round(h00 * ys[seg] + h10 * h * m[seg] + h01 * ys[seg + 1] + h11 * h * m[seg + 1]);
  }
  return lut;
}

// Apply the full curves grade (master + per-channel) to an RGBA buffer in place.
export function applyCurves(data, curves) {
  if (isIdentityCurves(curves)) return;
  applyChannelLuts(data, {
    master: buildCurveLut(curves.master),
    r: buildCurveLut(curves.r),
    g: buildCurveLut(curves.g),
    b: buildCurveLut(curves.b),
  });
}

// ── Histogram ────────────────────────────────────────────────────────────────
// Per-channel + luma counts over an RGBA buffer, for the levels display.
export function computeHistogram(data) {
  const r = new Array(256).fill(0);
  const g = new Array(256).fill(0);
  const b = new Array(256).fill(0);
  const luma = new Array(256).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    r[data[i]] += 1;
    g[data[i + 1]] += 1;
    b[data[i + 2]] += 1;
    luma[Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2])] += 1;
  }
  return { r, g, b, luma };
}
