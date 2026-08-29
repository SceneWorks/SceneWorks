import { describe, expect, it } from "vitest";
import { VECTOR_DETAIL_PRESETS, vectorSourceAssets } from "./VectorStudio.jsx";

describe("Vector Studio source boundary", () => {
  it("admits only active project raster assets and keeps SVG out of conditioning", () => {
    const assets = [
      { id: "png", projectId: "p1", type: "image", file: { mimeType: "image/png" }, status: {} },
      { id: "svg", projectId: "p1", type: "vector", file: { mimeType: "image/svg+xml" }, preview: { path: "preview.png" }, status: {} },
      { id: "other", projectId: "p2", type: "image", file: { mimeType: "image/png" }, status: {} },
      { id: "trashed", projectId: "p1", type: "image", file: { mimeType: "image/png" }, status: { trashed: true } },
    ];
    expect(vectorSourceAssets(assets, "p1").map((asset) => asset.id)).toEqual(["png"]);
  });

  it("keeps every selectable detail payload bounded", () => {
    for (const preset of Object.values(VECTOR_DETAIL_PRESETS)) {
      expect(preset.maxNewTokens).toBeGreaterThan(0);
      expect(preset.maxSvgBytes).toBeGreaterThan(0);
      expect(preset.maxWallTimeMs).toBeGreaterThan(0);
    }
  });
});
