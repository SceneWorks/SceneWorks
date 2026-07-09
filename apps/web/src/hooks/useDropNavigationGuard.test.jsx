// Issue #1308: a file dropped outside a real dropzone must not navigate the
// webview to the image (replacing the whole UI). The guard installs a
// window-level fallback that swallows any drag no in-app dropzone claimed.
import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { useDropNavigationGuard } from "./useDropNavigationGuard.js";

function Harness() {
  useDropNavigationGuard();
  return null;
}

// jsdom's Event has no dataTransfer; attach a minimal writable stand-in so we
// can observe the dropEffect the guard sets.
function dragEvent(type, { defaultPrevented = false } = {}) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  event.dataTransfer = { dropEffect: "copy" };
  if (defaultPrevented) {
    event.preventDefault();
  }
  return event;
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
});
