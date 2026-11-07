import React, { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot, click } from "../testUtils/dom.js";
import { AssetGrid, FullscreenPreview } from "./assetPanels.jsx";
let root, container;
afterEach(async () => { if (root) await unmountRoot(root, container); });
describe("exact asset tile identity", () => {
  it("distinguishes image and latent outputs with duplicate display names in the rendered grid", async () => {
    ({ root, container } = mountRoot());
    const assets = ["base-output", "base-duplicate", "sft-output"].map((id, index) => ({
      id, type: index === 0 ? "image" : "tensor", displayName: "MOSS sampled latents 1", status: {},
      file: { path: `${id}.${index === 0 ? "png" : "npy"}`, mimeType: index === 0 ? "image/png" : "application/x-npy" },
    }));
    const onPreview = vi.fn();
    await act(async () => root.render(<AssetGrid assets={assets} onPreview={onPreview} setSelectedAssetId={vi.fn()} />));
    expect(container.querySelectorAll(".asset-tile strong")).toHaveLength(3);
    for (const asset of assets) {
      const tiles = container.querySelectorAll(`.asset-tile[data-asset-id="${asset.id}"]`);
      expect(tiles).toHaveLength(1);
      await act(async () => tiles[0].dispatchEvent(new MouseEvent("dblclick", { bubbles: true })));
      expect(onPreview).toHaveBeenLastCalledWith(asset);
    }
  });
});
describe("native tensor result viewer", () => {
  it("exposes exact tensor and companion downloads and recipe replay without fabricated image or canvas affordances", async () => {
    ({ root, container } = mountRoot());
    const onUseRecipe = vi.fn();
    const asset = { id: "latent-4", projectId: "p", type: "tensor", displayName: "Sampled latents 4", status: {}, file: { path: "assets/tensors/g/0003.npy", mimeType: "application/x-npy" }, tensor: { dtype: "float32", shape: [1, 128, 32, 32], byteLength: 524288, sha256: "a".repeat(64) }, recipe: { model: "moss_image_sft", mode: "edit_image", seed: null, rawAdapterSettings: { mossImageAdvanced: { output: "latents", noise: { assetId: "noise", sha256: "b".repeat(64) } } } }, extra: { workflowCompanion: { path: "assets/tensors/g/0003.sceneworks-workflow.json", sha256: "c".repeat(64) } } };
    await act(async () => root.render(<FullscreenPreview asset={asset} sourceAsset={{ id: "source", type: "image", file: { path: "source.png", mimeType: "image/png" }, projectId: "p" }} onClose={() => {}} onUseRecipe={onUseRecipe} onEditImage={vi.fn()} onEditInStudio={vi.fn()} />));
    expect(document.body.querySelector("img,canvas,video")).toBeNull();
    expect(document.body.querySelector(".preview-zoom-viewport")).toBeNull();
    expect(document.body.querySelector('[aria-label="Zoom controls"]')).toBeNull();
    expect(document.body.textContent).not.toContain("Compare original");
    expect(document.body.textContent).toContain("No image preview");
    expect(document.body.textContent).toContain("a".repeat(64));
    const companion = [...document.body.querySelectorAll("a")].find((link) => link.textContent.includes("portable workflow"));
    expect(companion.href).toContain("0003.sceneworks-workflow.json");
    const recipe = [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Use this recipe");
    await click(recipe); expect(onUseRecipe).toHaveBeenCalledWith(asset, { keepSeed: false });
    expect(document.body.textContent).not.toContain("Keep seed");
  });
});
