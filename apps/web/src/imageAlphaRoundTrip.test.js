// Alpha survives open → edit → save → export in the Image Editor (sc-24111).
//
// Everything on the web side was already alpha-clean when this was written: no `{ alpha: false }`
// context, no white or black pre-fill before `drawImage`, and `image/png` on every `toBlob`. What
// was missing was anything that would *notice* if that stopped being true. Every canvas fake in
// this repo hands back a zeroed `getImageData` buffer and an empty `Blob`, so a regression to an
// opaque context — the likeliest way to lose the channel here — would leave the whole suite green.
//
// These tests drive the PRODUCTION export: `ImageEditor.workingToPngFile`, which is what both Save
// (it imports the returned File as a Library asset) and Download (in its re-encoding mode) call,
// over `imageLayers.compositeLayersToCanvas`, which is the shared flatten underneath it. The first
// version of this file re-implemented that flatten in a local helper, which meant the guard
// assertions were about the helper: switching the real encode to `image/jpeg` left them green.
// They now red. See `testUtils/rgbaCanvas.js` for why the canvas fake holds real pixels rather
// than pulling in node-canvas.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it, vi } from "vitest";

// konva's node build pulls in the native `canvas` package; nothing here mounts a <Stage>, so keep
// it out of the import graph the same way `ImageEditor.test.jsx` does.
vi.mock("react-konva", async () => {
  const React = await import("react");
  const passthrough = (name) => ({ children }) => React.createElement("div", { "data-konva": name }, children);
  return { Stage: passthrough("stage"), Layer: passthrough("layer"), Image: () => null, Rect: () => null };
});

import { addLayer, createLayer, singleLayerWorking } from "./imageLayers.js";
import { compositeWorkingToCanvas, workingToPngFile } from "./screens/ImageEditor.jsx";
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

/**
 * A `document` whose `createElement("canvas")` hands back a canvas that actually holds pixels.
 *
 * This is the ONLY thing the tests substitute. Everything between it and the PNG bytes —
 * the flatten, the layer transforms, the blend modes, the encode and its mime — is production
 * code reached through `workingToPngFile`.
 */
function pixelDocument() {
  const made = [];
  const documentRef = {
    createElement(tag) {
      if (tag !== "canvas") throw new Error(`unexpected createElement(${tag})`);
      const canvas = makeCanvas();
      made.push(canvas);
      return canvas;
    },
  };
  return { documentRef, made };
}

/** Open a document from PNG bytes, the way the editor's `blobToImage` + `singleLayerWorking` do. */
function openDocument(bytes, { width = 64, height = 64 } = {}) {
  const working = singleLayerWorking({
    id: "layer-1",
    image: fakeImageFromPng(bytes),
    objectUrl: "blob:source",
    blob: new Blob([bytes], { type: "image/png" }),
    source: new File([bytes], "cutout.png", { type: "image/png" }),
  });
  working.width = width;
  working.height = height;
  return working;
}

/** Save/Download, end to end, through the real exporter. */
async function exportPng(working) {
  const { documentRef, made } = pixelDocument();
  const file = await workingToPngFile(working, undefined, { documentRef });
  return { file, canvases: made, image: decodePng(await blobBytes(file)) };
}

const histogramObject = (data) =>
  Object.fromEntries([...alphaHistogram(data)].sort((a, b) => a[0] - b[0]));

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the committed fixtures", () => {
  it("carry transparent, soft and opaque regions over non-black, non-white colour", () => {
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

  it("include an RGB twin the codec reads back as fully opaque", () => {
    const { data } = decodePng(rgbFixture());
    expect(transparentPixelCount(data)).toBe(0);
    expect(alphaHistogram(data).get(255)).toBe(64 * 64);
  });
});

describe("open → save round trip", () => {
  it("keeps every alpha value through the real exporter", async () => {
    const source = decodePng(rgbaFixture());
    const { image, file } = await exportPng(openDocument(rgbaFixture()));

    expect(file.type).toBe("image/png");
    expect(file.name).toBe("cutout.png");
    expect([image.width, image.height]).toEqual([source.width, source.height]);
    expect(histogramObject(image.data)).toEqual(histogramObject(source.data));
    expect(transparentPixelCount(image.data)).toBe(transparentPixelCount(source.data));
  });

  it("never asks for an opaque context and never pre-fills the canvas", async () => {
    // The two ways the real export could start lying: `getContext("2d", { alpha: false })`
    // composites every transparent pixel onto black, and a `fillRect` before the first
    // `drawImage` does the same with whatever `fillStyle` is set. Both are recorded by the fake
    // canvas and asserted here against the PRODUCTION flatten.
    const { canvases } = await exportPng(openDocument(rgbaFixture()));

    expect(canvases).toHaveLength(1);
    for (const canvas of canvases) {
      for (const options of canvas.contextOptions) {
        expect(options?.alpha).not.toBe(false);
      }
      expect(canvas.fillRects).toEqual([]);
      expect(canvas.toBlobMimes).toEqual(["image/png"]);
    }
  });

  it("composites at the document size the editor holds", async () => {
    const { canvases } = await exportPng(openDocument(rgbaFixture()));
    expect([canvases[0].width, canvases[0].height]).toEqual([64, 64]);
  });

  it("refuses to export with no working document", async () => {
    const { documentRef } = pixelDocument();
    await expect(workingToPngFile(null, undefined, { documentRef })).rejects.toThrow(
      "No working image.",
    );
  });
});

describe("open → edit → save round trip", () => {
  it("keeps the source's transparency where an added layer does not cover it", async () => {
    const source = decodePng(rgbaFixture());
    // The edit: a second layer, opaque over the left eighth — which is inside the source's fully
    // transparent band — and fully transparent everywhere else.
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
      openDocument(rgbaFixture()),
      createLayer({ id: "layer-2", name: "Cutout", image: fakeImage(64, 64, overlay) }),
    );

    const { image } = await exportPng(working);
    const alphaAt = (x, y) => image.data[(y * 64 + x) * 4 + 3];

    expect(alphaAt(2, 10)).toBe(255); // painted over
    expect(alphaAt(12, 10)).toBe(0); // still inside the transparent band
    expect(alphaAt(30, 10)).toBeGreaterThan(0); // the soft ramp is still soft
    expect(alphaAt(30, 10)).toBeLessThan(255);
    expect(transparentPixelCount(image.data)).toBeGreaterThan(0);
    expect(transparentPixelCount(image.data)).toBeLessThan(
      transparentPixelCount(source.data),
    );
  });

  it("respects layer opacity without collapsing the channel", async () => {
    const working = openDocument(rgbaFixture());
    working.layers[0].opacity = 0.5;

    const { image } = await exportPng(working);
    const histogram = alphaHistogram(image.data);

    // The fully opaque band is now half-opaque, the fully transparent band is still gone, and the
    // ramp is still a ramp rather than two values.
    expect(histogram.get(0)).toBeGreaterThan(0);
    expect(histogram.get(255)).toBeUndefined();
    expect(histogram.size).toBeGreaterThanOrEqual(8);
  });

  it("drops a hidden layer without dropping the channel", async () => {
    const overlay = new Uint8ClampedArray(64 * 64 * 4).fill(255);
    const working = addLayer(
      openDocument(rgbaFixture()),
      createLayer({
        id: "layer-2",
        name: "Hidden",
        image: fakeImage(64, 64, overlay),
        visible: false,
      }),
    );
    const { image } = await exportPng(working);
    expect(transparentPixelCount(image.data)).toBe(
      transparentPixelCount(decodePng(rgbaFixture()).data),
    );
  });
});

describe("an opaque image is unchanged", () => {
  it("exports RGB content with a fully opaque alpha channel and no transparency invented", async () => {
    const source = decodePng(rgbFixture());
    const { image } = await exportPng(openDocument(rgbFixture()));

    expect(transparentPixelCount(image.data)).toBe(0);
    expect(alphaHistogram(image.data).get(255)).toBe(64 * 64);
    // The colour is untouched too, so "alpha preserved" is not being bought with a colour shift.
    expect([...image.data]).toEqual([...source.data]);
  });
});

describe("the flatten itself", () => {
  it("is reachable on its own and sizes the canvas to the document", () => {
    // `compositeWorkingToCanvas` is what `bakeBoxesToFile` and the AI-op source path use directly,
    // so it carries the same guarantee without going through an encode.
    const { documentRef } = pixelDocument();
    const canvas = compositeWorkingToCanvas(openDocument(rgbaFixture()), { documentRef });
    expect([canvas.width, canvas.height]).toEqual([64, 64]);
    expect(canvas.contextOptions.every((options) => options?.alpha !== false)).toBe(true);
    expect(transparentPixelCount(canvas.data)).toBeGreaterThan(0);
  });
});
