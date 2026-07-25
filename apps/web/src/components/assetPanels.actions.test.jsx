import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssetCard, AssetDetail } from "./assetPanels.jsx";

function asset(id, status = {}) {
  return {
    id,
    type: "image",
    displayName: `${id}.png`,
    file: { mimeType: "image/png", path: `${id}.png` },
    status,
    tags: [],
  };
}

describe("asset panel shared controls", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("uses a distinct datalist id for concurrently mounted tag editors", () => {
    const props = {
      deleteAsset: vi.fn(),
      purgeAsset: vi.fn(),
      onPreview: vi.fn(),
      onSendImage: vi.fn(),
      onSendVideo: vi.fn(),
      onSendEditor: vi.fn(),
      updateAssetStatus: vi.fn(),
      updateAssetTags: vi.fn(),
    };
    act(() =>
      root.render(
        <>
          <AssetDetail {...props} asset={asset("one")} />
          <AssetDetail {...props} asset={asset("two")} />
        </>,
      ),
    );
    const inputs = [...container.querySelectorAll('input[aria-label="Add asset tag"]')];
    expect(inputs).toHaveLength(2);
    expect(inputs[0].getAttribute("list")).not.toBe(inputs[1].getAttribute("list"));
    for (const input of inputs) {
      expect(container.querySelector(`datalist[id="${input.getAttribute("list")}"]`)).not.toBeNull();
    }
  });

  it("keeps status labels and mutations identical in cards and details", () => {
    const favorite = asset("favorite", { favorite: true });
    const updateAssetStatus = vi.fn();
    const common = {
      deleteAsset: vi.fn(),
      purgeAsset: vi.fn(),
      onPreview: vi.fn(),
      updateAssetStatus,
    };
    act(() =>
      root.render(
        <>
          <AssetCard {...common} asset={favorite} />
          <AssetDetail
            {...common}
            asset={favorite}
            onSendImage={vi.fn()}
            onSendVideo={vi.fn()}
            onSendEditor={vi.fn()}
            updateAssetTags={vi.fn()}
          />
        </>,
      ),
    );
    const unfavorite = [...container.querySelectorAll("button")].filter(
      (button) => button.textContent === "Unfavorite",
    );
    expect(unfavorite).toHaveLength(2);
    act(() => unfavorite[0].click());
    act(() => unfavorite[1].click());
    expect(updateAssetStatus).toHaveBeenNthCalledWith(1, favorite, { favorite: false });
    expect(updateAssetStatus).toHaveBeenNthCalledWith(2, favorite, { favorite: false });
  });
});
