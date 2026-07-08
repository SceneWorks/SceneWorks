import { describe, expect, it, vi } from "vitest";

// konva's node build pulls in the native `canvas` package (not installed / unusable in
// jsdom). Importing ImageEditor.jsx below drags react-konva into the graph, so stub it —
// mirroring ImageEditor.test.jsx. We only read module-scope re-exports, never mount.
vi.mock("react-konva", async () => {
  const React = await import("react");
  const passthrough = (name) => ({ children }) => React.createElement("div", { "data-konva": name }, children);
  return {
    Stage: passthrough("stage"),
    Layer: passthrough("layer"),
    Image: () => null,
    Line: () => null,
    Rect: () => null,
    Transformer: () => null,
  };
});

// Focused tests for the per-tool modules extracted from ImageEditor.jsx (sc-9752, F-052
// follow-up). The behavior of the pure helpers is exhaustively covered by
// ImageEditor.test.jsx via the re-exports; these tests pin the extraction contract:
//   1. every symbol ImageEditor re-exports is the SAME reference the module defines
//      (so the public surface is genuinely the extracted code, not a copy), and
//   2. the modules stand alone (import + run without pulling ImageEditor.jsx in), which
//      is the whole point of putting them in cycle-free helper modules.
import * as editor from "../ImageEditor.jsx";
import * as colorGradeMath from "./colorGradeMath.js";
import * as boxGeometry from "./boxGeometry.js";
import * as maskShared from "./maskShared.js";

describe("colorGradeMath module (sc-9752)", () => {
  it("is the source of the editor's re-exported color-grade helpers (same references)", () => {
    for (const name of [
      "COLOR_ADJUSTMENTS",
      "IDENTITY_COLOR_ADJUST",
      "isIdentityAdjust",
      "gradePixel",
      "applyColorAdjustments",
      "konvaColorFilter",
    ]) {
      expect(editor[name]).toBe(colorGradeMath[name]);
    }
  });

  it("stands alone: the pure grade math runs without ImageEditor.jsx", () => {
    expect(colorGradeMath.isIdentityAdjust(colorGradeMath.IDENTITY_COLOR_ADJUST)).toBe(true);
    // Brightness +1 saturates to white; identity leaves a pixel untouched.
    expect(colorGradeMath.gradePixel([100, 100, 100], { brightness: 1 })).toEqual([255, 255, 255]);
    expect(colorGradeMath.gradePixel([10, 20, 30], colorGradeMath.IDENTITY_COLOR_ADJUST)).toEqual([10, 20, 30]);
  });

  it("konvaColorFilter dispatches on the node's gradeMode attr (adjust default)", () => {
    // A minimal fake Konva node: getAttr returns the requested attr. The "adjust" branch
    // runs applyColorAdjustments on imageData.data in place.
    const attrs = { gradeMode: "adjust", colorAdjust: { brightness: 1 } };
    const node = { getAttr: (k) => attrs[k] };
    const imageData = { data: new Uint8ClampedArray([100, 100, 100, 255]) };
    colorGradeMath.konvaColorFilter.call(node, imageData);
    expect([imageData.data[0], imageData.data[1], imageData.data[2]]).toEqual([255, 255, 255]);
    expect(imageData.data[3]).toBe(255); // alpha untouched
  });
});

describe("boxGeometry module (sc-9752)", () => {
  it("is the source of the editor's re-exported box helpers (same references)", () => {
    for (const name of [
      "BOX_TYPES",
      "MAX_BOX_PALETTE",
      "MAX_DOCUMENT_PALETTE",
      "isValidHexColor",
      "rectToBbox",
      "bboxToRect",
      "boxPaletteIsValid",
      "documentPalette",
      "documentPaletteIsValid",
      "boxIsValid",
      "BOX_PALETTE",
      "MIN_BOX_PX",
      "rectFromPoints",
      "clampRectToCanvas",
      "makeBox",
      "boxFillStyle",
      "addPaletteColor",
      "removePaletteColor",
      "boxMetadataGaps",
      "paintBoxesOnContext",
      "colorName",
      "composeColorPrompt",
      "boxesToIdeogramElements",
    ]) {
      expect(editor[name]).toBe(boxGeometry[name]);
    }
  });

  it("stands alone: box geometry runs without ImageEditor.jsx", () => {
    expect(boxGeometry.rectFromPoints({ x: 40, y: 60 }, { x: 10, y: 20 })).toEqual({
      x: 10,
      y: 20,
      width: 30,
      height: 40,
    });
    // Sub-minimum drag is grown to MIN_BOX_PX by clampRectToCanvas.
    const tiny = boxGeometry.clampRectToCanvas({ x: 5, y: 5, width: 2, height: 1 }, 800, 600);
    expect(tiny.width).toBe(boxGeometry.MIN_BOX_PX);
    expect(tiny.height).toBe(boxGeometry.MIN_BOX_PX);
  });
});

describe("maskShared module (sc-9752)", () => {
  it("is the source of the editor's re-exported mask helpers (same references)", () => {
    for (const name of [
      "buildSegmentJobBody",
      "rectToSegmentBox",
      "tintMaskRgbaInPlace",
      "MASK_PREVIEW_RGBA",
      "maskHasContent",
    ]) {
      expect(editor[name]).toBe(maskShared[name]);
    }
  });

  it("stands alone: mask helpers run without ImageEditor.jsx", () => {
    // A non-erase stroke with a drawn segment is mask content; an erase-only stroke isn't.
    expect(maskShared.maskHasContent([{ points: [10, 10], size: 40, erase: false }])).toBe(true);
    expect(maskShared.maskHasContent([{ points: [0, 0, 5, 5], size: 40, erase: true }])).toBe(false);
    // A dragged-up-left rect is ordered to a positive [x1,y1,x2,y2] box.
    expect(maskShared.rectToSegmentBox({ x: 110, y: 220, width: -100, height: -200 })).toEqual([10, 20, 110, 220]);
  });
});
