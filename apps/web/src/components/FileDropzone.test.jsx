import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FileDropzone } from "./FileDropzone.jsx";

describe("FileDropzone", () => {
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

  it("preserves drag claiming and clears the file input after dispatch", () => {
    const onActiveChange = vi.fn();
    const onFiles = vi.fn();
    act(() =>
      root.render(
        <FileDropzone
          accept="image/*"
          active
          hint="Drop an image"
          onActiveChange={onActiveChange}
          onFiles={onFiles}
        />,
      ),
    );
    const dropzone = container.querySelector(".dataset-add-dropzone");
    const drag = new Event("dragover", { bubbles: true, cancelable: true });
    dropzone.dispatchEvent(drag);
    expect(drag.defaultPrevented).toBe(true);
    expect(onActiveChange).toHaveBeenCalledWith(true);
    const dropped = [new File(["image"], "drop.png", { type: "image/png" })];
    const drop = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(drop, "dataTransfer", { value: { files: dropped } });
    act(() => dropzone.dispatchEvent(drop));
    expect(drop.defaultPrevented).toBe(true);
    expect(onActiveChange).toHaveBeenCalledWith(false);
    expect(onFiles).toHaveBeenCalledWith(dropped);

    const input = container.querySelector('input[type="file"]');
    const files = { 0: new File(["image"], "one.png", { type: "image/png" }), length: 1 };
    Object.defineProperty(input, "files", { configurable: true, value: files });
    act(() => input.dispatchEvent(new Event("change", { bubbles: true })));
    expect(onFiles).toHaveBeenCalledWith(files);
    expect(input.value).toBe("");
  });
});
