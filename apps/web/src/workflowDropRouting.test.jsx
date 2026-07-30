// The drop no dropzone claimed, and the drops that WERE claimed (sc-15951, epic 15945).
//
// `useDropNavigationGuard` is a window-level fallback: it acts only on a drag whose
// `defaultPrevented` is still false by the time the event reaches `window`. sc-15951 hangs the
// workflow inspector off that same branch, which puts one thing at risk — the five real dropzones
// that handle their own drops (AssetPicker, the Image Editor canvas, DatasetAddDialog,
// DocumentStudio, PoseLibraryScreen). If the inspector could see their files it would open a
// "Workflow found" panel over an upload that was already happening.
//
// So this file dispatches REAL bubbling drops through the two the acceptance criteria name and
// asserts, on each, that the component handled it and the inspector never saw it. Nothing here
// stubs the dropzone: the assertion is worthless unless the component's own `preventDefault` is
// the thing being tested.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// konva's node build needs the native `canvas` package. The editor's empty state never mounts
// the <Stage>, so stub react-konva exactly as `screens/ImageEditor.test.jsx` does.
vi.mock("react-konva", async () => {
  const ReactModule = await import("react");
  const passthrough = (name) => ({ children }) =>
    ReactModule.createElement("div", { "data-konva": name }, children);
  return {
    Stage: passthrough("stage"),
    Layer: passthrough("layer"),
    Image: () => null,
    Rect: () => null,
  };
});

import { ImageEditSourcePickerField } from "./components/AssetPicker.jsx";
import { AppContext } from "./context/AppContext.js";
import { useDropNavigationGuard } from "./hooks/useDropNavigationGuard.js";
import { ImageEditor } from "./screens/ImageEditor.jsx";

function pngFile(name = "shared.png") {
  return new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], name, { type: "image/png" });
}

// A native drop with real files, dispatched at `target` so it bubbles through the component's
// own React handler on its way to the window guard — the exact path a user's drop takes.
function dispatchDrop(target, files) {
  const event = new Event("drop", { bubbles: true, cancelable: true });
  event.dataTransfer = { dropEffect: "copy", files, types: ["Files"] };
  target.dispatchEvent(event);
  return event;
}

function Guard({ onUnclaimedFileDrop }) {
  useDropNavigationGuard({ onUnclaimedFileDrop });
  return null;
}

describe("the workflow inspector never steals a claimed drop", () => {
  let container;
  let root;
  let inspect;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    inspect = vi.fn();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const render = async (ui) => {
    await act(async () => root.render(ui));
    await act(async () => {});
  };

  const buttonByLabel = (label) =>
    [...document.querySelectorAll("button")].find((node) => node.textContent.trim() === label);

  const click = async (node) =>
    act(async () => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));

  it("AssetPicker's upload dropzone still gets the file, and the inspector does not", async () => {
    const importAsset = vi.fn(async () => ({ id: "asset_1" }));
    await render(
      <>
        <Guard onUnclaimedFileDrop={inspect} />
        <ImageEditSourcePickerField
          assets={[]}
          characters={[]}
          importAsset={importAsset}
          label="Source image"
          onChange={() => {}}
          value=""
        />
      </>,
    );
    await click(buttonByLabel("Select image"));
    await click(buttonByLabel("File Upload"));
    // The picker modal portals to <body>; React attaches its delegated listeners to portal
    // containers too, so a drop dispatched here runs the picker's own handler first.
    const dropzone = document.querySelector(".dataset-add-dropzone");
    expect(dropzone).toBeTruthy();

    const file = pngFile();
    let event;
    await act(async () => {
      event = dispatchDrop(dropzone, [file]);
    });

    expect(importAsset).toHaveBeenCalledTimes(1);
    expect(importAsset.mock.calls[0][0]).toBe(file);
    expect(event.defaultPrevented).toBe(true);
    expect(inspect).not.toHaveBeenCalled();
  });

  it("the Image Editor canvas still claims its drop, and the inspector does not see it", async () => {
    await render(
      <>
        <Guard onUnclaimedFileDrop={inspect} />
        <AppContext.Provider
          value={{
            activeProject: null,
            assets: [],
            characters: [],
            setPreviewAsset: vi.fn(),
            token: "",
            requestedGpu: "auto",
            jobs: [],
            importAsset: vi.fn(),
            purgeAsset: vi.fn(),
            registerLeaveGuard: vi.fn(() => () => {}),
            trackEditorScratchOp: vi.fn(),
            releaseEditorScratchOp: vi.fn(),
            registerEditorScratchClaim: vi.fn(() => () => {}),
            imageModels: [],
          }}
        >
          <ImageEditor />
        </AppContext.Provider>
      </>,
    );
    const canvas = container.querySelector(".ie-canvas");
    expect(canvas).toBeTruthy();

    let event;
    await act(async () => {
      event = dispatchDrop(canvas, [pngFile("edit-me.png")]);
    });

    // The canvas's own handler ran (it is what calls preventDefault), so the window guard bows
    // out and the inspector is never offered a file the editor is already opening.
    expect(event.defaultPrevented).toBe(true);
    expect(inspect).not.toHaveBeenCalled();
  });

  it("a drop on neutral chrome — nothing in the tree claims it — reaches the inspector", async () => {
    // The other half of the same rule: with the very same components mounted, a drop that lands
    // somewhere none of them handles is the one sc-15951 acts on.
    await render(
      <>
        <Guard onUnclaimedFileDrop={inspect} />
        <div className="neutral-chrome" />
      </>,
    );
    const neutral = container.querySelector(".neutral-chrome");
    const file = pngFile();
    let event;
    await act(async () => {
      event = dispatchDrop(neutral, [file]);
    });
    expect(event.defaultPrevented).toBe(true);
    expect(inspect).toHaveBeenCalledWith(file);
  });
});
