import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import {
  ImageEditorToolPanel,
  setImageEditorToolPanelRenderObserverForTests,
  useStableImageEditorToolPanelScope,
} from "./ImageEditorToolPanel.jsx";

let container;
let root;
beforeEach(() => ({ container, root } = mountRoot()));
afterEach(async () => {
  setImageEditorToolPanelRenderObserverForTests(null);
  await unmountRoot(root, container);
});

describe("ImageEditorToolPanel memo boundary", () => {
  it("skips parent-only rerenders while a rendered handler sees its latest closure", async () => {
    const observer = vi.fn();
    const fitCalls = [];
    const working = { width: 640, height: 480 };
    setImageEditorToolPanelRenderObserverForTests(observer);
    function ImageEditorParentHarness() {
      const [stageWidth, setStageWidth] = React.useState(100);
      const scope = {
        working,
        layerCount: 1,
        fitToView: () => fitCalls.push(stageWidth),
        actualSize: () => {},
      };
      const stableScope = useStableImageEditorToolPanelScope(scope);
      return (
        <>
          <button data-action="parent-update" onClick={() => setStageWidth(200)} type="button">
            resize parent
          </button>
          <ImageEditorToolPanel panelKey="move" scope={stableScope} />
        </>
      );
    }

    await act(async () => root.render(<ImageEditorParentHarness />));
    await act(async () => {
      container.querySelector("[data-action='parent-update']")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(observer).toHaveBeenCalledTimes(1);
    await act(async () => {
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "Fit to view")
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(fitCalls).toEqual([200]);
  });
});
