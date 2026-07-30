// Issue #1308: a file dropped outside a real dropzone must not navigate the
// webview to the image (replacing the whole UI). The guard installs a
// window-level fallback that swallows any drag no in-app dropzone claimed.
//
// sc-15951 rides on the same "nothing else claimed it" branch: an unclaimed drop of one file is
// handed to a workflow inspector. The tests below pin BOTH halves of that — that the handler
// fires for a stray drop, and that it cannot fire for a drop a real dropzone handled.
import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useDropNavigationGuard } from "./useDropNavigationGuard.js";

function Harness({ onUnclaimedFileDrop }) {
  useDropNavigationGuard({ onUnclaimedFileDrop });
  return null;
}

// jsdom's Event has no dataTransfer; attach a minimal writable stand-in so we
// can observe the dropEffect the guard sets.
function dragEvent(type, { defaultPrevented = false, files = [] } = {}) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  event.dataTransfer = { dropEffect: "copy", files };
  if (defaultPrevented) {
    event.preventDefault();
  }
  return event;
}

function pngFile(name = "shared.png") {
  return new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], name, { type: "image/png" });
}

describe("useDropNavigationGuard", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    act(() => {
      root = createRoot(container);
      root.render(<Harness />);
    });
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("swallows an unclaimed drop so the browser never navigates to the file", () => {
    const event = dragEvent("drop");
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(event.dataTransfer.dropEffect).toBe("none");
  });

  it("marks an unclaimed dragover as not-allowed and prevents its default", () => {
    const event = dragEvent("dragover");
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(event.dataTransfer.dropEffect).toBe("none");
  });

  it("leaves a drag a real dropzone already claimed untouched", () => {
    // A dropzone calls preventDefault() as the event bubbles through it; by the
    // time it reaches window `defaultPrevented` is already true and the guard
    // must bow out rather than override the dropzone's copy affordance.
    const event = dragEvent("drop", { defaultPrevented: true });
    window.dispatchEvent(event);
    expect(event.dataTransfer.dropEffect).toBe("copy");
  });

  it("removes its window listeners on unmount", () => {
    act(() => {
      root.unmount();
    });
    root = null;
    const event = dragEvent("drop");
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(event.dataTransfer.dropEffect).toBe("copy");

    // Re-establish a root so the shared afterEach unmount stays valid.
    act(() => {
      root = createRoot(container);
      root.render(<Harness />);
    });
  });

  // ---- The workflow inspector hand-off (sc-15951) --------------------------------------

  it("hands an unclaimed single-file drop to the inspector, still swallowing it", () => {
    const onUnclaimedFileDrop = vi.fn();
    act(() => root.render(<Harness onUnclaimedFileDrop={onUnclaimedFileDrop} />));
    const file = pngFile();
    const event = dragEvent("drop", { files: [file] });
    window.dispatchEvent(event);
    expect(onUnclaimedFileDrop).toHaveBeenCalledWith(file);
    // The swallow is unconditional and already done by the time the handler runs, so a slow or
    // throwing inspector can never let the webview navigate.
    expect(event.defaultPrevented).toBe(true);
  });

  it("never offers a drop a real dropzone claimed", () => {
    // The regression that would silently steal AssetPicker's and the Image Editor's uploads:
    // the inspector must sit strictly inside the `!defaultPrevented` branch.
    const onUnclaimedFileDrop = vi.fn();
    act(() => root.render(<Harness onUnclaimedFileDrop={onUnclaimedFileDrop} />));
    window.dispatchEvent(dragEvent("drop", { defaultPrevented: true, files: [pngFile()] }));
    expect(onUnclaimedFileDrop).not.toHaveBeenCalled();
  });

  it("never offers a dragover, only a drop", () => {
    const onUnclaimedFileDrop = vi.fn();
    act(() => root.render(<Harness onUnclaimedFileDrop={onUnclaimedFileDrop} />));
    window.dispatchEvent(dragEvent("dragover", { files: [pngFile()] }));
    expect(onUnclaimedFileDrop).not.toHaveBeenCalled();
  });

  it("never offers a drag carrying no files, or more than one", () => {
    // A text or URL drag carries no files; a multi-file drag is a bulk gesture no single-recipe
    // offer could answer honestly. Both keep the pre-sc-15951 silent swallow.
    const onUnclaimedFileDrop = vi.fn();
    act(() => root.render(<Harness onUnclaimedFileDrop={onUnclaimedFileDrop} />));
    window.dispatchEvent(dragEvent("drop", { files: [] }));
    window.dispatchEvent(dragEvent("drop", { files: [pngFile("a.png"), pngFile("b.png")] }));
    expect(onUnclaimedFileDrop).not.toHaveBeenCalled();
  });

  it("keeps its listeners across a handler change rather than re-subscribing", () => {
    // The handler lives in a ref for exactly this: a drag in flight across a listener teardown
    // would land on nothing at all and navigate.
    const addSpy = vi.spyOn(window, "addEventListener");
    const first = vi.fn();
    const second = vi.fn();
    act(() => root.render(<Harness onUnclaimedFileDrop={first} />));
    const addsAfterFirst = addSpy.mock.calls.length;
    act(() => root.render(<Harness onUnclaimedFileDrop={second} />));
    expect(addSpy.mock.calls.length).toBe(addsAfterFirst);
    window.dispatchEvent(dragEvent("drop", { files: [pngFile()] }));
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
    addSpy.mockRestore();
  });

  it("swallows a stray drop unchanged when no inspector is wired at all", () => {
    act(() => root.render(<Harness />));
    const event = dragEvent("drop", { files: [pngFile()] });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(event.dataTransfer.dropEffect).toBe("none");
  });
});
