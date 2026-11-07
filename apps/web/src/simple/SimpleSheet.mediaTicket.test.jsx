import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { setMediaTicket } from "../api.js";
import { SimpleSheetOptions } from "./SimpleSheet.jsx";

it("refreshes a cached reference sheet without fetching or altering the selected asset", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const fetchSpy = vi.spyOn(globalThis, "fetch");
  const asset = { id: "img", type: "image", projectId: "p", file: { path: "assets/a.png", mimeType: "image/png" } };
  const options = [{ value: asset.id, label: "Image", thumbnailAsset: asset }];
  const onSelect = vi.fn();
  try {
    setMediaTicket("old-ticket");
    await act(async () => root.render(<SimpleSheetOptions kind="list" options={options} onSelect={onSelect} />));
    expect(container.querySelector("img").src).toContain("ticket=old-ticket");
    await act(async () => setMediaTicket("new-ticket"));
    const image = container.querySelector("img");
    expect(image.src).toContain("ticket=new-ticket");
    expect(image.src).not.toContain("old-ticket");
    expect(image.getAttribute("referrerpolicy")).toBe("no-referrer");
    expect(fetchSpy).not.toHaveBeenCalled();
    await act(async () => container.querySelector("button").click());
    expect(onSelect).toHaveBeenCalledOnce();
    expect(onSelect).toHaveBeenCalledWith("img");
  } finally {
    await act(async () => root.unmount());
    container.remove();
    setMediaTicket("");
    fetchSpy.mockRestore();
  }
});
