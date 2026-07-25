import { describe, expect, it } from "vitest";
import {
  PEAK_RESOLUTION,
  idlePeaks,
  peaksFromSamples,
  resamplePeaks,
  waveformPath,
  waveformWidth,
} from "./audioWaveform.js";

// Waveform peaks seam (epic 14361 / sc-14362). The peaks are derived from the REAL clip, so
// the pure halves — downsampling, resampling and the path builder — are what these pin.

describe("peaksFromSamples", () => {
  it("downsamples to the requested bar count and normalizes the loudest bar to 1", () => {
    // Four windows, each with a different absolute peak; the max must scale to exactly 1.
    const samples = new Float32Array([0.1, 0.05, 0.4, 0.2, -0.8, 0.3, 0.2, 0.05]);
    const peaks = peaksFromSamples(samples, 4);
    expect(peaks).toHaveLength(4);
    expect(peaks.map((value) => Number(value.toFixed(3)))).toEqual([0.125, 0.5, 1, 0.25]);
  });

  it("uses the ABSOLUTE amplitude, so a negative-only window is not silent", () => {
    // Discriminating: a signed mean (or an unsigned max on raw values) would report ~0 here.
    const peaks = peaksFromSamples(new Float32Array([-0.5, -0.5, 0.25, 0.25]), 2);
    expect(peaks).toEqual([1, 0.5]);
  });

  it("returns zeros — never NaN — for silence or an empty buffer", () => {
    expect(peaksFromSamples(new Float32Array([0, 0, 0, 0]), 2)).toEqual([0, 0]);
    expect(peaksFromSamples(new Float32Array([]), 3)).toEqual([0, 0, 0]);
  });
});

describe("resamplePeaks", () => {
  it("takes the max within each window when narrowing", () => {
    expect(resamplePeaks([0.1, 0.9, 0.3, 0.2], 2)).toEqual([0.9, 0.3]);
  });

  it("passes an already-matching array through untouched", () => {
    const peaks = [0.2, 0.4];
    expect(resamplePeaks(peaks, 2)).toBe(peaks);
  });

  it("pads a missing decode to the requested length", () => {
    expect(resamplePeaks(null, 3)).toEqual([0, 0, 0]);
  });
});

describe("waveformPath", () => {
  it("emits ONE path of vertical strokes at the bar pitch, centred on the mid-line", () => {
    const path = waveformPath([1, 0.5], { bars: 2, height: 40, pitch: 5, inset: 3, minBar: 0 });
    // Room is (height/2 - inset) = 17: a full bar spans 3..37, a half bar 11.5..28.5.
    expect(path).toBe("M3 3.0V37.0M8 11.5V28.5");
  });

  it("floors silence at minBar so a quiet clip still reads as a waveform", () => {
    const path = waveformPath([0], { bars: 1, height: 40, inset: 3, minBar: 0.1 });
    expect(path).toBe("M3 18.3V21.7");
  });

  it("clamps an out-of-range peak rather than overflowing the box", () => {
    expect(waveformPath([9], { bars: 1, height: 40, inset: 3 })).toBe("M3 3.0V37.0");
  });

  it("widths the viewBox from the bar count and pitch", () => {
    expect(waveformWidth(4, 5, 3)).toBe(21);
    expect(waveformWidth(1, 5, 3)).toBe(6);
  });
});

describe("idlePeaks", () => {
  it("is a flat row at the surface's own resolution", () => {
    expect(idlePeaks(4)).toEqual([0, 0, 0, 0]);
    expect(idlePeaks()).toHaveLength(PEAK_RESOLUTION);
  });
});
