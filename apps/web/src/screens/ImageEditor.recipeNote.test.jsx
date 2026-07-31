import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// konva's node build pulls in the native `canvas` package (not installed, and not usable in
// jsdom). Nothing here mounts the <Stage> — the container has no size in jsdom, so the editor's
// own gate keeps it unmounted — so stub react-konva to keep konva out of the import graph, exactly
// as `ImageEditor.test.jsx` does.
vi.mock("react-konva", async () => {
  const React = await import("react");
  const passthrough = (name) => ({ children }) => React.createElement("div", { "data-konva": name }, children);
  return { Stage: passthrough("stage"), Layer: passthrough("layer"), Image: () => null, Rect: () => null };
});

import { ImageEditor } from "./ImageEditor.jsx";
import { AppContext } from "../context/AppContext.js";

const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4]);

const IMAGE_ASSET = {
  id: "asset-recipe",
  type: "image",
  displayName: "Generated plate",
  projectId: "project_1",
  file: { path: "assets/images/plate.png" },
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project_1", name: "Project" },
    assets: [IMAGE_ASSET],
    characters: [],
    setPreviewAsset: vi.fn(),
    token: "t",
    requestedGpu: "auto",
    jobs: [],
    importAsset: vi.fn(),
    purgeAsset: vi.fn(),
    registerLeaveGuard: vi.fn(() => () => {}),
    trackEditorScratchOp: vi.fn(),
    releaseEditorScratchOp: vi.fn(),
    registerEditorScratchClaim: vi.fn(() => () => {}),
    imageModels: [],
    editorLaunch: { id: "launch-1", assetId: IMAGE_ASSET.id },
    clearEditorLaunch: vi.fn(),
    ...overrides,
  };
}

// ── What the editor's Download pill says, end to end (sc-15954) ──────────────
//
// The unit tests beside this one pin `editorDownloadPlan` / `editorDownloadNote` as pure
// functions. This file pins the WIRING those functions sit behind, which is where the two
// AC-forbidden outcomes actually lived: the verdict is asynchronous now, and a window in which
// "not yet known" renders as "no recipe" would ship a recipe with no pill on an untouched export
// and drop one with no pill on an edited one.
describe("the Image Editor's recipe pill asks the one reader (sc-15954)", () => {
  let container;
  let root;
  let originalImage;
  // The transport is stubbed at `fetch` rather than at `inspectWorkflowFile`, so the multipart the
  // editor actually builds is exercised too — and so this file needs no module mock, which would
  // knock konva off the browser resolution the rest of these tests rely on. `inspect` records the
  // inspect calls; every other URL is the asset media fetch.
  const inspect = vi.fn();

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    // jsdom decodes nothing, so the editor's `blobToImage` never resolves without this. 1024² is
    // deliberately under EDITOR_CANVAS_MAX_SIDE: the downscale branch needs a real 2D context,
    // which jsdom has not got, and the geometry it produces is covered by the pure-function tests.
    originalImage = global.Image;
    global.Image = class {
      constructor() {
        this.naturalWidth = 1024;
        this.naturalHeight = 1024;
        this.onload = null;
        this.onerror = null;
      }
      set src(_value) {
        queueMicrotask(() => this.onload?.());
      }
    };
    global.URL.createObjectURL = vi.fn(() => "blob:editor");
    global.URL.revokeObjectURL = vi.fn();
    vi.spyOn(global, "fetch").mockImplementation(async (url, options) => {
      if (String(url).includes("/api/v1/workflows/inspect")) {
        return inspect(url, options);
      }
      return {
        ok: true,
        status: 200,
        blob: async () => new Blob([PNG_BYTES], { type: "image/png" }),
      };
    });
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    global.Image = originalImage;
    inspect.mockReset();
    vi.restoreAllMocks();
  });

  async function render(context = baseContext()) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageEditor />
        </AppContext.Provider>,
      );
    });
    // The open is a chain of awaits (fetch → blob → decode → describe → install) and the inspect
    // effect only fires off the document that chain installs, so the render is flushed until the
    // header appears rather than a fixed number of times.
    for (let tick = 0; tick < 40 && !container.querySelector(".ie-recipe-note"); tick += 1) {
      await act(async () => {});
    }
  }

  const pill = () => container.querySelector(".ie-recipe-note");
  const pillText = () => pill()?.textContent.trim() ?? null;
  const body = (payload, { ok = true, status = 200 } = {}) => ({
    ok,
    status,
    json: async () => payload,
  });

  it("renders 'Checking for a recipe…' while the reader has not answered", async () => {
    let settle;
    inspect.mockImplementation(
      () =>
        new Promise((resolve) => {
          settle = resolve;
        }),
    );
    await render();

    // The AC-critical assertion: the pre-answer state is VISIBLE and is not the empty
    // "this file has no recipe" render.
    expect(pillText()).toBe("Checking for a recipe…");
    expect(pill().getAttribute("data-tone")).toBe("pending");
    expect(pill().getAttribute("data-empty")).toBeNull();

    await act(async () => {
      settle(body({ status: "workflow", workflow: {}, resolution: {} }));
    });
    await act(async () => {});
    expect(pillText()).toBe("Recipe included");
    expect(pill().getAttribute("data-tone")).toBe("included");
    expect(pill().getAttribute("title")).toContain("Save a copy without the workflow");
  });

  it("sends the opened file to POST /api/v1/workflows/inspect, scoped to the project", async () => {
    inspect.mockResolvedValue(body({ status: "no_workflow", workflow: null, resolution: null }));
    await render();

    expect(inspect).toHaveBeenCalledTimes(1);
    const [, options] = inspect.mock.calls[0];
    expect(options.method).toBe("POST");
    expect(options.body).toBeInstanceOf(FormData);
    expect(options.body.get("file").name).toBe("Generated plate.png");
    expect(options.body.get("projectId")).toBe("project_1");
    expect(options.signal).toBeInstanceOf(AbortSignal);
  });

  // A verdict of "no recipe" is the only state with nothing to say. The live region still exists —
  // a region inserted into the DOM already populated is not reliably announced, so it is mounted
  // for the life of the document and emptied instead of unmounted.
  it("empties the pill for a confirmed absence but keeps the live region mounted", async () => {
    inspect.mockResolvedValue(body({ status: "no_workflow", workflow: null, resolution: null }));
    await render();

    expect(pillText()).toBe("");
    expect(pill().getAttribute("role")).toBe("status");
    expect(pill().getAttribute("data-empty")).toBe("true");
  });

  // The failure direction. Offline, a 413, a 500, a 422 for a file this build refuses to read —
  // none of those is "no recipe", and reporting one as one is the silent loss AC2 forbids.
  it.each([
    ["a transport failure", () => Promise.reject(new Error("network down"))],
    ["a 413 from the endpoint's own cap", () => Promise.resolve(body({ code: "workflow_inspect_too_large" }, { ok: false, status: 413 }))],
    ["a 422 for a file this build refuses to read", () => Promise.resolve(body({ code: "workflow_inspect_unreadable" }, { ok: false, status: 422 }))],
    ["a status from a newer server", () => Promise.resolve(body({ status: "something_new" }))],
  ])("renders 'Recipe unknown' for %s", async (_label, response) => {
    inspect.mockImplementation(response);
    await render();

    expect(pillText()).toBe("Recipe unknown");
    expect(pill().getAttribute("data-tone")).toBe("unknown");
    expect(pill().getAttribute("title")).toMatch(/not that it has none/);
  });
});
