import React, { act, useRef, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountRoot, unmountRoot } from "../../testUtils/dom.js";
import { useMaskTool } from "./useMaskTool.js";

function layer(id, width, height, blob = { id }) {
  return {
    id,
    blob,
    image: { naturalWidth: width, naturalHeight: height },
    transform: { x: 14, y: 9, scaleX: 1.5, scaleY: 0.75, rotation: 18 },
  };
}

function documentWith(activeLayerId = "a") {
  return {
    width: 100,
    height: 80,
    activeLayerId,
    source: { name: "layers.png" },
    layers: [layer("a", 20, 10), layer("b", 40, 30)],
  };
}

describe("useMaskTool coordinate and ownership contracts", () => {
  let container;
  let root;
  let latest;
  let capturedOp;
  let setHarnessTool;
  let originalCreateElement;

  beforeEach(() => {
    ({ container, root } = mountRoot());
    originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tagName, options) => {
      if (tagName !== "canvas") return originalCreateElement(tagName, options);
      const canvas = {
        width: 0,
        height: 0,
        getContext: () => ({
          arc: vi.fn(),
          beginPath: vi.fn(),
          drawImage: vi.fn(),
          fill: vi.fn(),
          fillRect: vi.fn(),
          getImageData: (_x, _y, width, height) => ({
            data: new Uint8ClampedArray(width * height * 4),
          }),
          lineTo: vi.fn(),
          moveTo: vi.fn(),
          putImageData: vi.fn(),
          stroke: vi.fn(),
        }),
        toBlob: (callback) => callback(new Blob(["mask"], { type: "image/png" })),
      };
      return canvas;
    });
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, blob: async () => new Blob(["mask"]) })));
    vi.stubGlobal("URL", { ...URL, revokeObjectURL: vi.fn() });
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function Harness({ canMask = false, initialTool = "cutout", working }) {
    const [tool, setTool] = useState(initialTool);
    setHarnessTool = setTool;
    const workingRef = useRef(working);
    workingRef.current = working;
    latest = useMaskTool({
      working,
      tool,
      canMask,
      smartSelectSupported: true,
      aiOp: null,
      activeProject: { id: "project", name: "Project" },
      requestedGpu: "auto",
      runAiOp: async (operation) => {
        capturedOp = operation;
        setTool("move");
        return true;
      },
      stagePointToImage: () => ({ x: 70, y: 60 }),
      stagePointToActiveLayerImage: () => ({ x: 3, y: 4 }),
      getWorking: () => workingRef.current,
      blobToImage: async () => ({
        image: { naturalWidth: working.width, naturalHeight: working.height },
        objectUrl: "blob:mask",
      }),
      setTool,
    });
    return <div data-tool={tool} />;
  }

  it("returns a Cutout-launched SAM3 result to Cutout with an editable mask when canMask is false", async () => {
    const working = documentWith("a");
    await act(async () => root.render(<Harness working={working} />));

    await act(async () => latest.runSmartSelect({ x: 1, y: 2, width: 12, height: 6 }));
    expect(capturedOp.layerSource).toBe("activeLayer");
    expect(container.firstChild.dataset.tool).toBe("move");

    await act(async () => capturedOp.onComplete({ id: "mask-result" }));
    expect(container.firstChild.dataset.tool).toBe("cutout");
    expect(latest.maskMode).toBe(true);
    expect(latest.maskSubTool).toBe("brush");
    expect(latest.maskBaseImage).toBeTruthy();
    expect(latest.maskTargetLayerId).toBe("a");
  });

  it("drops both installed and pending cutout selections when the active layer changes", async () => {
    const workingA = documentWith("a");
    await act(async () => root.render(<Harness working={workingA} />));
    await act(async () => {
      latest.installMaskBaseCanvas({ width: 20, height: 10 }, { source: "colorKey", targetLayerId: "a" });
      latest.setMaskMode(true);
    });
    expect(latest.maskTargetLayerId).toBe("a");

    await act(async () => root.render(<Harness working={documentWith("b")} />));
    expect(latest.maskBaseImage).toBeNull();
    expect(latest.maskTargetLayerId).toBeNull();

    await act(async () => root.render(<Harness working={workingA} />));
    await act(async () => latest.runSmartSelect({ x: 1, y: 2, width: 12, height: 6 }));
    const pendingCompletion = capturedOp.onComplete;
    await act(async () => root.render(<Harness working={documentWith("b")} />));
    await act(async () => pendingCompletion({ id: "late-mask-result" }));
    expect(container.firstChild.dataset.tool).toBe("move");
    expect(latest.maskBaseImage).toBeNull();
    expect(latest.maskTargetLayerId).toBeNull();
  });

  it("does not carry an active-layer cutout mask into AI Edit's document coordinate context", async () => {
    await act(async () => root.render(<Harness canMask working={documentWith("a")} />));
    await act(async () => {
      latest.installMaskBaseCanvas({ width: 20, height: 10 }, { source: "colorKey", targetLayerId: "a" });
      latest.setMaskMode(true);
    });
    expect(latest.maskBaseImage).toBeTruthy();

    await act(async () => setHarnessTool("edit"));
    expect(latest.maskBaseImage).toBeNull();
    expect(latest.maskTargetLayerId).toBeNull();
    expect(latest.maskMode).toBe(false);
    expect(latest.maskCoordinateSpace).toBe("document");
  });

  it("keeps transformed multilayer AI Edit masks in document coordinates and document dimensions", async () => {
    const stagePointToImage = vi.fn(() => ({ x: 70, y: 60 }));
    const stagePointToActiveLayerImage = vi.fn(() => ({ x: 3, y: 4 }));
    const working = documentWith("a");

    function EditHarness() {
      const [tool, setTool] = useState("edit");
      latest = useMaskTool({
        working,
        tool,
        canMask: true,
        smartSelectSupported: true,
        aiOp: null,
        activeProject: { id: "project", name: "Project" },
        requestedGpu: "auto",
        runAiOp: async (operation) => {
          capturedOp = operation;
          setTool("move");
          return true;
        },
        stagePointToImage,
        stagePointToActiveLayerImage,
        getWorking: () => working,
        blobToImage: async () => ({ image: {}, objectUrl: "blob:mask" }),
        setTool,
      });
      return null;
    }

    await act(async () => root.render(<EditHarness />));
    await act(async () => latest.setMaskMode(true));
    await act(async () => latest.maskPointerDown({}));
    expect(stagePointToImage).toHaveBeenCalledTimes(1);
    expect(stagePointToActiveLayerImage).not.toHaveBeenCalled();
    expect(latest.maskLines[0].points).toEqual([70, 60]);
    const raster = latest.rasterizeMaskToCanvas();
    expect([raster.width, raster.height]).toEqual([100, 80]);

    await act(async () => latest.runSmartSelect({ x: 10, y: 8, width: 20, height: 18 }));
    expect(capturedOp.layerSource).toBe("composite");
  });
});
