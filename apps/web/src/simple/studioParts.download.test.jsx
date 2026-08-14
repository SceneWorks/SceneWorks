import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

// sc-15953 — the Simple shell has a without-the-workflow download.
//
// The finding this file exists for: `WorkflowEmbedDetails`, which the Simple Settings screen
// renders, tells the user that to share an image already on disk without its recipe they should use
// "Save a copy without the workflow". The only place that control existed was the ADVANCED shell's
// right-click context menu. The Simple shell's only egress was `DownloadButton`, a bare
// `<a href={assetUrl(asset)} download>` with no strip param — and the same argument that puts the
// toggle in this shell ("it is the default on a phone") makes the missing remedy worse, because a
// phone has no right-click.
//
// `assetMedia.jsx` is mocked at the URL seam so the anchor's href is a fixed, inspectable string;
// everything else here is the real component.

vi.mock("../components/assetMedia.jsx", () => ({
  assetUrl: (asset) => (asset?.relativePath ? `/api/v1/projects/p1/files/${asset.relativePath}` : ""),
  AssetThumbnail: () => null,
}));

const { DownloadButton } = await import("./studioParts.jsx");

const SIMPLE_UI = { toast: vi.fn(), breakpoint: "desktop" };

function pngAsset(overrides = {}) {
  return {
    id: "asset_1",
    displayName: "shot.png",
    relativePath: "assets/images/shot.png",
    file: { path: "assets/images/shot.png", mimeType: "image/png" },
    ...overrides,
  };
}

function anchors(container) {
  return [...container.querySelectorAll("a")].map((a) => a.getAttribute("href"));
}

function buttons(container) {
  return [...container.querySelectorAll("button")];
}

describe("DownloadButton — the without-workflow copy (sc-15953)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    SIMPLE_UI.toast.mockClear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  async function render(asset, props = {}) {
    await act(async () => {
      root.render(
        <SimpleUiContext.Provider value={SIMPLE_UI}>
          <DownloadButton asset={asset} {...props} />
        </SimpleUiContext.Provider>,
      );
    });
  }

  it("offers a second anchor carrying the strip param for a PNG", async () => {
    await render(pngAsset());
    const hrefs = anchors(container);
    expect(hrefs).toContain("/api/v1/projects/p1/files/assets/images/shot.png");
    expect(hrefs).toContain("/api/v1/projects/p1/files/assets/images/shot.png?stripWorkflow=true");
    // A real anchor, not a fetch: the media URL already carries the auth ticket in remote mode
    // (sc-8810), and the strip is a query param on the path that ticket's allow-list matches on.
    expect(buttons(container)).toHaveLength(2);
  });

  it("names the control exactly as every other surface does", async () => {
    await render(pngAsset());
    const label = buttons(container)[1].getAttribute("aria-label");
    expect(label).toBe("Save a copy without the workflow");
    // Reachable by touch: it is a button in the flow, not a right-click target.
    expect(buttons(container)[1].getAttribute("title")).toBe(label);
  });

  it("clicks the stripped anchor, not the plain one", async () => {
    await render(pngAsset());
    const clicked = [];
    for (const anchor of container.querySelectorAll("a")) {
      anchor.click = () => clicked.push(anchor.getAttribute("href"));
    }
    await act(async () => {
      buttons(container)[1].click();
    });
    expect(clicked).toEqual([
      "/api/v1/projects/p1/files/assets/images/shot.png?stripWorkflow=true",
    ]);
    expect(SIMPLE_UI.toast).toHaveBeenCalledWith("Downloading without the recipe…");
  });

  it("leaves the plain download exactly what it was", async () => {
    await render(pngAsset());
    const clicked = [];
    for (const anchor of container.querySelectorAll("a")) {
      anchor.click = () => clicked.push(anchor.getAttribute("href"));
    }
    await act(async () => {
      buttons(container)[0].click();
    });
    expect(clicked).toEqual(["/api/v1/projects/p1/files/assets/images/shot.png"]);
    expect(SIMPLE_UI.toast).toHaveBeenCalledWith("Downloading…");
  });

  it("omits it for a format the envelope cannot ride in", async () => {
    // Offering a control that can do nothing would imply the file might be carrying a workflow.
    await render(
      pngAsset({
        displayName: "clip.mp4",
        relativePath: "assets/videos/clip.mp4",
        file: { path: "assets/videos/clip.mp4", mimeType: "video/mp4" },
      }),
    );
    expect(buttons(container)).toHaveLength(1);
    expect(anchors(container)).toEqual(["/api/v1/projects/p1/files/assets/videos/clip.mp4"]);
  });

  it("omits it when the asset has no URL at all", async () => {
    await render({ id: "asset_2", displayName: "pending.png" });
    expect(buttons(container)).toHaveLength(1);
  });

  it("labels itself in the preview variant, where there is room for words", async () => {
    await render(pngAsset(), { className: "", label: "Download" });
    expect(buttons(container).map((b) => b.textContent.trim())).toEqual([
      "Download",
      "Without recipe",
    ]);
  });
});
