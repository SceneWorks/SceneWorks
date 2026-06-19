import { describe, expect, it } from "vitest";
import {
  IDENTITY_LEVELS,
  IDENTITY_CURVE,
  IDENTITY_CURVES,
  isIdentityLevels,
  isIdentityLevelsChannel,
  isIdentityCurve,
  isIdentityCurves,
  levelsChannelLut,
  applyLevels,
  normalizeCurvePoints,
  buildCurveLut,
  applyCurves,
  applyChannelLuts,
  computeHistogram,
} from "./colorGrade.js";

// A 1-pixel RGBA buffer for one (r,g,b); alpha 255.
const px = (r, g, b) => Uint8ClampedArray.from([r, g, b, 255]);

describe("levels (sc-6109)", () => {
  it("identity levels leave a pixel untouched", () => {
    expect(isIdentityLevels(IDENTITY_LEVELS)).toBe(true);
    expect(isIdentityLevelsChannel({ black: 0, white: 255, gamma: 1 })).toBe(true);
    const data = px(10, 128, 240);
    applyLevels(data, IDENTITY_LEVELS);
    expect([...data]).toEqual([10, 128, 240, 255]);
  });

  it("the identity LUT maps every value to itself", () => {
    const lut = levelsChannelLut({ black: 0, white: 255, gamma: 1 });
    expect(lut[0]).toBe(0);
    expect(lut[128]).toBe(128);
    expect(lut[255]).toBe(255);
  });

  it("black/white points clip and rescale the span", () => {
    const lut = levelsChannelLut({ black: 64, white: 192, gamma: 1 });
    expect(lut[64]).toBe(0); // at/below black → 0
    expect(lut[32]).toBe(0);
    expect(lut[192]).toBe(255); // at/above white → 255
    expect(lut[224]).toBe(255);
    expect(lut[128]).toBe(128); // midpoint of [64,192] → 128
  });

  it("gamma lifts (>1) or lowers (<1) the midtones, endpoints fixed", () => {
    const lift = levelsChannelLut({ black: 0, white: 255, gamma: 2 });
    const lower = levelsChannelLut({ black: 0, white: 255, gamma: 0.5 });
    expect(lift[0]).toBe(0);
    expect(lift[255]).toBe(255);
    expect(lift[128]).toBeGreaterThan(128); // gamma>1 brightens midtones
    expect(lower[128]).toBeLessThan(128); // gamma<1 darkens midtones
  });

  it("applies the master gamma uniformly across channels (midtones move, endpoints fixed)", () => {
    const data = px(128, 128, 128);
    applyLevels(data, { ...IDENTITY_LEVELS, master: { black: 0, white: 255, gamma: 0.5 } });
    expect(data[0]).toBeLessThan(128); // gamma<1 darkens the midtone
    expect(data[0]).toBe(data[1]); // applied uniformly per channel
    expect(data[1]).toBe(data[2]);
    // Endpoints are untouched by gamma.
    const white = px(255, 255, 255);
    applyLevels(white, { ...IDENTITY_LEVELS, master: { black: 0, white: 255, gamma: 0.5 } });
    expect([...white]).toEqual([255, 255, 255, 255]);
  });
});

describe("curves (sc-6109)", () => {
  it("identity curve maps every value to itself", () => {
    expect(isIdentityCurve(IDENTITY_CURVE)).toBe(true);
    expect(isIdentityCurves(IDENTITY_CURVES)).toBe(true);
    const lut = buildCurveLut(IDENTITY_CURVE);
    expect(lut[0]).toBe(0);
    expect(lut[128]).toBe(128);
    expect(lut[255]).toBe(255);
  });

  it("normalizeCurvePoints sorts, clamps, and de-dupes by x", () => {
    expect(normalizeCurvePoints([{ x: 255, y: 255 }, { x: 0, y: 0 }, { x: 300, y: -5 }])).toEqual([
      { x: 0, y: 0 },
      { x: 255, y: 0 }, // x clamped to 255, y clamped to 0; later point at x=255 wins over the first
    ]);
  });

  it("respects control points exactly and stays monotonic between them", () => {
    // Lift the midtone: (128 → 180).
    const lut = buildCurveLut([{ x: 0, y: 0 }, { x: 128, y: 180 }, { x: 255, y: 255 }]);
    expect(lut[0]).toBe(0);
    expect(lut[128]).toBe(180);
    expect(lut[255]).toBe(255);
    // Monotonic non-decreasing (Fritsch–Carlson never reverses).
    for (let v = 1; v < 256; v += 1) expect(lut[v]).toBeGreaterThanOrEqual(lut[v - 1]);
    // Midtone lift brightens the lower half.
    expect(lut[64]).toBeGreaterThan(64);
  });

  it("a flat segment never overshoots (monotone, no ringing)", () => {
    const lut = buildCurveLut([{ x: 0, y: 0 }, { x: 100, y: 200 }, { x: 150, y: 200 }, { x: 255, y: 255 }]);
    for (let v = 0; v < 256; v += 1) {
      expect(lut[v]).toBeGreaterThanOrEqual(0);
      expect(lut[v]).toBeLessThanOrEqual(255);
    }
    // Between the two y=200 points the curve holds ~200 without exceeding it much.
    expect(lut[125]).toBeGreaterThanOrEqual(195);
    expect(lut[125]).toBeLessThanOrEqual(205);
  });

  it("fewer than two distinct points falls back to identity", () => {
    const lut = buildCurveLut([{ x: 5, y: 250 }]);
    expect(lut[0]).toBe(0);
    expect(lut[255]).toBe(255);
  });

  it("applyCurves skips the identity and otherwise transforms pixels", () => {
    const same = px(10, 20, 30);
    applyCurves(same, IDENTITY_CURVES);
    expect([...same]).toEqual([10, 20, 30, 255]);

    // An inverting master curve (0→255, 255→0) negates every channel.
    const data = px(0, 128, 255);
    applyCurves(data, { ...IDENTITY_CURVES, master: [{ x: 0, y: 255 }, { x: 255, y: 0 }] });
    expect(data[0]).toBe(255);
    expect(data[2]).toBe(0);
  });
});

describe("LUT composition + histogram", () => {
  it("applyChannelLuts composes per-channel first, then master", () => {
    const id = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v += 1) id[v] = v;
    const plus10 = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v += 1) plus10[v] = Math.min(255, v + 10);
    const data = px(100, 100, 100);
    // master +10 over a red channel of +10 → red = 120, g/b = 110.
    applyChannelLuts(data, { master: plus10, r: plus10, g: id, b: id });
    expect(data[0]).toBe(120);
    expect(data[1]).toBe(110);
    expect(data[2]).toBe(110);
  });

  it("computeHistogram counts per channel + luma", () => {
    const data = Uint8ClampedArray.from([0, 0, 0, 255, 255, 255, 255, 255]); // black + white pixels
    const h = computeHistogram(data);
    expect(h.r[0]).toBe(1);
    expect(h.r[255]).toBe(1);
    expect(h.luma[0]).toBe(1);
    expect(h.luma[255]).toBe(1);
    expect(h.g.reduce((a, b) => a + b, 0)).toBe(2);
  });
});
