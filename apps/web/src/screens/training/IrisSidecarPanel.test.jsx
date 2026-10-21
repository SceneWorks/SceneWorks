import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { IrisSidecarPanel } from "./IrisSidecarPanel.jsx";

describe("IrisSidecarPanel (sc-25687 / sc-25688)", () => {
  let container;
  let root;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  const items = [
    { id: "i1", displayName: "one.png", irisDepth: { path: "s/one.npy" } },
    { id: "i2", displayName: "two.png" },
  ];

  async function open() {
    const details = container.querySelector("details");
    details.open = true;
    await act(async () => details.dispatchEvent(new Event("toggle")));
  }

  function pick(input, files) {
    Object.defineProperty(input, "files", { configurable: true, value: files });
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  it("bulk-attaches depth maps by filename with the depth value fields and reports unmatched files", async () => {
    const onSidecar = vi.fn().mockResolvedValue({});
    await act(async () => root.render(<IrisSidecarPanel items={items} onSidecar={onSidecar} />));
    await open();
    expect(container.textContent).toContain("Depth map ✓");

    const scale = [...container.querySelectorAll("label")].find((label) => label.textContent.startsWith("Depth scale"));
    await act(async () => {
      const input = scale.querySelector("input");
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(input, "0.001");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const bulk = container.querySelector('input[aria-label="Attach depth maps by filename"]');
    const files = [{ name: "two_depth.png" }, { name: "stray.npy" }];
    await act(async () => pick(bulk, files));

    expect(onSidecar).toHaveBeenCalledTimes(1);
    expect(onSidecar).toHaveBeenCalledWith(items[1], "depth", files[0], { scale: "0.001" });
    expect(container.textContent).toContain("1 sidecar change applied. No item matched: stray.npy.");
  });

  it("removes an attached sidecar", async () => {
    const onSidecar = vi.fn().mockResolvedValue({});
    await act(async () => root.render(<IrisSidecarPanel items={items} onSidecar={onSidecar} />));
    await open();
    await act(async () => container.querySelector('button[aria-label="Remove depth map of one.png"]').click());
    expect(onSidecar).toHaveBeenCalledWith(items[0], "depth", null, undefined);
  });
});
