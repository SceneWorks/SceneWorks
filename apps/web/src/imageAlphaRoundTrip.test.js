// Alpha survives open → edit → save → export in the Image Editor (sc-24111).
//
// Everything on the web side was already alpha-clean when this was written: no `{ alpha: false }`
// context, no white or black pre-fill before `drawImage`, and `image/png` on every `toBlob`. What
// was missing was anything that would *notice* if that stopped being true. Every canvas fake in
// this repo hands back a zeroed `getImageData` buffer and an empty `Blob`, so a regression to an
// opaque context — the single likeliest way to lose the channel here — would leave the whole
// suite green.
//
// So these tests run the production compositing (`imageLayers.compositeLayersToCanvas`, the shared
// flatten behind Save, Download and the AI-op source) over a canvas fake that keeps real pixels,
// starting from the same committed RGBA fixture the Rust tests use and ending at real PNG bytes.
// See `testUtils/rgbaCanvas.js` for why that fake exists rather than node-canvas.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  addLayer,
  compositeLayersToCanvas,
  createLayer,
  singleLayerWorking,
} from "./imageLayers.js";
import {
  alphaHistogram,
  blobBytes,
  decodePng,
  fakeImage,
  fakeImageFromPng,
  makeCanvas,
  transparentPixelCount,
} from "./testUtils/rgbaCanvas.js";

const fixtureDir = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../tests/fixtures/alpha",
);

const rgbaFixture = () => readFileSync(path.join(fixtureDir, "alpha-64.png"));
const rgbFixture = () => readFileSync(path.join(fixtureDir, "opaque-rgb-64.png"));

/** The editor's flatten, end to end: layers in, PNG bytes out. */
async function flattenToPng(layers, width, height) {
  const canvas = makeCanvas();
  canvas.width = width;
  canvas.height = height;
  compositeLayersToCanvas(canvas.getContext("2d"), layers, { visibleOnly: true });
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  return { canvas, blob, bytes: await blobBytes(blob) };
}

const histogramObject = (data) => Object.fromEntries([...alphaHistogram(data)].sort((a, b) => a[0] - b[0]));

describe("the committed fixtures", () => {
  it("carries transparent, soft and opaque regions over non-black, non-white colour", () => {
    // Guards every other test here. A fixture regenerated flat would leave them all passing while
    // proving nothing, and one whose transparent pixels are black or white underneath would hide
    // exactly the flatten this story is about.
    const { width, height, data } = decodePng(rgbaFixture());
    expect([width, height]).toEqual([64, 64]);
    const histogram = alphaHistogram(data);
    expect(histogram.get(0)).toBeGreaterThan(0);
    expect(histogram.get(255)).toBeGreaterThan(0);
    expect(histogram.size).toBeGreaterThanOrEqual(8);
    for (let index = 0; index < data.length; index += 4) {
      if (data[index + 3] !== 0) continue;
      const rgb = [data[index], data[index + 1], data[index + 2]].join(",");
      expect(rgb).not.toBe("0,0,0");
      expect(rgb).not.toBe("255,255,255");
    }
  });

  it("has an RGB twin the codec reads back as fully opaque", () => {
    const { data } = decodePng(rgbFixture());
    expect(transparentPixelCount(data)).toBe(0);
    expect(alphaHistogram(data).get(255)).toBe(64 * 64);
  });
});

describe("open → save round trip", () => {
  it("keeps every alpha value when the opened image is flattened straight back out", async () => {
    const source = decodePng(rgbaFixture());
    const working = singleLayerWorking({
      id: "layer-1",
      image: fakeImageFromPng(rgbaFixture()),
      objectUrl: "blob:source",
      blob: new Blob([rgbaFixture()], { type: "image/png" }),
    });

    const { bytes } = await flattenToPng(working.layers, source.width, source.height);
    const exported = decodePng(bytes);

    expect(exported.width).toBe(source.width);
    expect(exported.height).toBe(source.height);
    expect(histogramObject(exported.data)).toEqual(histogramObject(source.data));
    expect(transparentPixelCount(exported.data)).toBe(transparentPixelCount(source.data));
  });

  it("never asks for an opaque context and never pre-fills the canvas", async () => {
    // The two ways this file could start lying: `getContext("2d", { alpha: false })` composites
    // every transparent pixel onto black, and a `fillRect` before the first `drawImage` does the
    // same with whatever `fillStyle` is set. Both are asserted against rather than assumed.
    const working = singleLayerWorking({
      id: "layer-1",
      image: fakeImageFromPng(rgbaFixture()),
      objectUrl: "blob:source",
      blob: new Blob([rgbaFixture()], { type: "image/png" }),
    });
    const { canvas } = await flattenToPng(working.layers, 64, 64);

    for (const options of canvas.contextOptions) {
      expect(options?.alpha).not.toBe(false);
    }
    expect(canvas.fillRects).toEqual([]);
    expect(canvas.toBlobMimes).toEqual(["image/png"]);
  });
});

describe("open → edit → save round trip", () => {
  it("keeps the source's transparency where an added layer does not cover it", async () => {
    const source = decodePng(rgbaFixture());
    // The edit: a second layer that is opaque over the left half (which is where the source's
    // fully transparent band lives) and fully transparent over the right.
    const overlay = new Uint8ClampedArray(64 * 64 * 4);
    for (let y = 0; y < 64; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const index = (y * 64 + x) * 4;
        overlay[index] = 10;
        overlay[index + 1] = 200;
        overlay[index + 2] = 90;
        overlay[index + 3] = 255;
      }
    }
    const working = addLayer(
      singleLayerWorking({
        id: "layer-1",
        image: fakeImageFromPng(rgbaFixture()),
        objectUrl: "blob:source",
        blob: new Blob([rgbaFixture()], { type: "image/png" }),
      }),
      createLayer({ id: "layer-2", name: "Cutout", image: fakeImage(64, 64, overlay) }),
    );

    const { bytes } = await flattenToPng(working.layers, 64, 64);
    const exported = decodePng(bytes);
    const alphaAt = (x, y) => exported.data[(y * 64 + x) * 4 + 3];

    // Painted over: opaque now.
    expect(alphaAt(2, 10)).toBe(255);
    // Still inside the source's transparent band, and the overlay does not reach it.
    expect(alphaAt(12, 10)).toBe(0);
    // The soft ramp is untouched: neither fully on nor fully off.
    expect(alphaAt(30, 10)).toBeGreaterThan(0);
    expect(alphaAt(30, 10)).toBeLessThan(255);
    // And the file as a whole still carries transparency — a flatten would report zero.
    expect(transparentPixelCount(exported.data)).toBeGreaterThan(0);
    expect(transparentPixelCount(exported.data)).toBeLessThan(
      transparentPixelCount(source.data),
    );
  });

  it("respects layer opacity without collapsing the channel", async () => {
    const working = singleLayerWorking({
      id: "layer-1",
      image: fakeImageFromPng(rgbaFixture()),
      objectUrl: "blob:source",
      blob: new Blob([rgbaFixture()], { type: "image/png" }),
    });
    working.layers[0].opacity = 0.5;

    const { bytes } = await flattenToPng(working.layers, 64, 64);
    const exported = decodePng(bytes);
    const histogram = alphaHistogram(exported.data);

    // The fully opaque band is now half-opaque, the fully transparent band is still gone, and the
    // ramp is still a ramp rather than two values.
    expect(histogram.get(0)).toBeGreaterThan(0);
    expect(histogram.get(255)).toBeUndefined();
    expect(histogram.size).toBeGreaterThanOrEqual(8);
  });
});

describe("an opaque image is unchanged", () => {
  it("exports RGB content with a fully opaque alpha channel and no transparency invented", async () => {
    const source = decodePng(rgbFixture());
    const working = singleLayerWorking({
      id: "layer-1",
      image: fakeImageFromPng(rgbFixture()),
      objectUrl: "blob:source",
      blob: new Blob([rgbFixture()], { type: "image/png" }),
    });

    const { bytes } = await flattenToPng(working.layers, source.width, source.height);
    const exported = decodePng(bytes);

    expect(transparentPixelCount(exported.data)).toBe(0);
    expect(alphaHistogram(exported.data).get(255)).toBe(64 * 64);
    // The colour is untouched too, so "alpha preserved" is not being bought with a colour shift.
    expect([...exported.data]).toEqual([...source.data]);
  });
});
