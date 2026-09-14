import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Timeline } from "./Timeline.jsx";

let container;
let root;

function renderTimeline(onScrub = vi.fn(), zoom = 1) {
  if (!root) {
    root = createRoot(container);
  }
  act(() => {
    root.render(
      <Timeline
        assetsById={{}}
        duration={10}
        onAddAudioTrack={() => {}}
        onScrub={onScrub}
        onSelectGap={() => {}}
        onSelectItem={() => {}}
        onSelectKey={() => {}}
        onToggleMute={() => {}}
        onToggleSnap={() => {}}
        onToggleSolo={() => {}}
        onToggleVisible={() => {}}
        onZoomIn={() => {}}
        onZoomOut={() => {}}
        playheadSeconds={0}
        snap={false}
        timeline={{ tracks: [] }}
        zoom={zoom}
      />,
    );
  });
  return onScrub;
}

function ruler() {
  return container.querySelector(".ve-ruler");
}

function lane() {
  return container.querySelector(".ve-lanes");
}

async function mouseDown(node, clientX) {
  await act(async () => {
    node.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, clientX }));
  });
}

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  if (root) {
    act(() => root.unmount());
    root = null;
  }
  container.remove();
});

describe("Timeline ruler drag lifecycle (sc-21905)", () => {
  it("removes the exact window listeners when unmounted during a ruler drag", async () => {
    const addSpy = vi.spyOn(window, "addEventListener");
    const removeSpy = vi.spyOn(window, "removeEventListener");
    try {
      renderTimeline();
      vi.spyOn(lane(), "getBoundingClientRect").mockReturnValue({ left: 0, width: 100 });
      await mouseDown(ruler(), 50);

      const mousemoveAdd = addSpy.mock.calls.find(([type]) => type === "mousemove");
      const mouseupAdd = addSpy.mock.calls.find(([type]) => type === "mouseup");
      expect(mousemoveAdd).toBeDefined();
      expect(mouseupAdd).toBeDefined();

      act(() => root.unmount());
      root = null;
      expect(removeSpy).toHaveBeenCalledWith("mousemove", mousemoveAdd[1]);
      expect(removeSpy).toHaveBeenCalledWith("mouseup", mouseupAdd[1]);
    } finally {
      addSpy.mockRestore();
      removeSpy.mockRestore();
    }
  });

  it("reads the current lane geometry at the start of each new scrub", async () => {
    const onScrub = renderTimeline();
    let rect = { left: 0, width: 100 };
    vi.spyOn(lane(), "getBoundingClientRect").mockImplementation(() => rect);

    await mouseDown(ruler(), 60);
    await act(async () => window.dispatchEvent(new MouseEvent("mouseup")));
    rect = { left: 20, width: 200 };
    renderTimeline(onScrub, 2);
    await mouseDown(ruler(), 120);

    expect(onScrub).toHaveBeenNthCalledWith(1, 6);
    expect(onScrub).toHaveBeenNthCalledWith(2, 5);
  });
});
