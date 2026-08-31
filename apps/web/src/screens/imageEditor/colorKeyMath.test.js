import { describe, expect, it } from "vitest";
import {
  applyAlphaMultipliersInPlace,
  applyColorKeyToRgba,
  colorKeyAlphaMultipliers,
  documentPointToLayerPoint,
} from "./colorKeyMath.js";

const pixels = (entries) => new Uint8ClampedArray(entries.flatMap(([r, g, b, a = 255]) => [r, g, b, a]));

describe("color-key alpha selection (sc-22462)", () => {
  it("removes only the 4-connected perceptual match by default", () => {
    // blue subject pixel at index 3 separates the identically green island at index 4.
    const source = pixels([
      [30, 180, 90], [31, 181, 91], [30, 180, 90], [30, 80, 220], [30, 180, 90],
    ]);
    const { rgba } = applyColorKeyToRgba(source, 5, 1, { x: 0, y: 0, tolerance: 6, softness: 0 });
    expect([...rgba.filter((_, index) => index % 4 === 3)]).toEqual([0, 0, 0, 255, 255]);
    // RGB remains intact; cutout is alpha-only.
    expect([...rgba.slice(0, 3)]).toEqual([30, 180, 90]);
  });

  it("global mode removes disconnected matches but leaves a distinguishable subject pixel", () => {
    const source = pixels([
      [30, 180, 90], [30, 80, 220], [30, 180, 90], [100, 130, 130],
    ]);
    const { rgba } = applyColorKeyToRgba(source, 4, 1, { x: 0, y: 0, tolerance: 6, softness: 0, global: true });
    expect([...rgba.filter((_, index) => index % 4 === 3)]).toEqual([0, 255, 0, 255]);
  });

  it("uses a soft perceptual edge and never resurrects existing transparency", () => {
    const source = pixels([[20, 170, 90, 80], [28, 176, 96, 200], [90, 80, 220, 255]]);
    const multipliers = colorKeyAlphaMultipliers(source, 3, 1, { x: 0, y: 0, tolerance: 2, softness: 20 });
    expect(multipliers[0]).toBe(0);
    expect(multipliers[1]).toBeGreaterThan(0);
    expect(multipliers[1]).toBeLessThan(255);
    const result = applyAlphaMultipliersInPlace(new Uint8ClampedArray(source), multipliers);
    expect(result[3]).toBe(0);
    expect(result[7]).toBeLessThan(200);
    expect(result[11]).toBe(255);
  });

  it("maps an eyedropper click through the active layer transform", () => {
    const point = documentPointToLayerPoint({ x: 15, y: 25 }, { x: 10, y: 20, scaleX: 2, scaleY: 1, rotation: 0 });
    expect(point).toEqual({ x: 2.5, y: 5 });
    const rotated = documentPointToLayerPoint({ x: 10, y: 13 }, { x: 10, y: 10, rotation: 90, scaleX: 1, scaleY: 1 });
    expect(rotated.x).toBeCloseTo(3);
    expect(rotated.y).toBeCloseTo(0);
  });
});
