import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// AC2 on the CATALOG CARD (epic 20398, sc-20650).
//
// Two claims, both of which the pre-epic screen got wrong by construction because it had no way to
// know a checkpoint's ownership:
//
//   1. A linked checkpoint whose library is detached, or whose file drifted, must render its
//      CORRECTIVE ACTION — "Needs relink" / "Needs rescan" with the button that clears it — not the
//      "missing" badge every uninstalled row gets. A user shown "missing" re-downloads bytes they
//      already own on a drive they only have to plug in.
//   2. Deleting a LINKED row and deleting a MANAGED row promise opposite things, so they must say
//      opposite things. Crossing them is the most damaging copy defect on this screen.
//
// The transport is mocked; the pure state→affordance mapping is the real module, so a change to
// what `linkedCorrection` decides is visible here rather than papered over by a hand-written stub.

const scanResult = vi.fn();
const rescan = vi.fn();
const update = vi.fn();

vi.mock("../checkpointLibrary.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    fetchLibraryRoots: vi.fn(async () => ({ roots: [{ rootId: "root-a", path: "/Volumes/Models", displayLabel: "Models" }] })),
    scanLibraryRoot: vi.fn(async () => scanResult()),
    rescanLibraryCheckpoint: vi.fn((...args) => rescan(...args)),
    updateLibraryRoot: vi.fn((...args) => update(...args)),
  };
});

const appConfirm = vi.fn(async () => true);
vi.mock("../appConfirm.jsx", () => ({ appConfirm: (...args) => appConfirm(...args) }));

const LINKED_MODEL = {
  id: "my_sdxl",
  name: "My SDXL",
  type: "image",
  family: "sdxl",
  catalogScope: "user",
  installState: "missing",
  capabilities: ["text_to_image"],
  importPlan: { checkpointId: "linked/root-a/sdxl.safetensors" },
  source: { provider: "linked-library", rootId: "root-a", relativePath: "sdxl.safetensors" },
  ui: { description: "A checkpoint in my own library." },
};

const MANAGED_MODEL = {
  id: "fetched",
  name: "Fetched Model",
  type: "image",
  family: "sdxl",
  catalogScope: "user",
  installState: "installed",
  importPlan: { checkpointId: "managed/install-1" },
  source: { provider: "civitai", url: "https://civitai.com/x" },
  ui: { description: "A checkpoint SceneWorks copied in." },
};

function statusFor(state, detail) {
  return { checkpointId: "linked/root-a/sdxl.safetensors", rootId: "root-a", relativePath: "sdxl.safetensors", state, detail };
}

describe("ModelManagerScreen linked state and ownership-aware removal (sc-20650)", () => {
  let container;
  let root;
  let ModelManagerScreen;
  let AppContext;
  let deleteModel;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    appConfirm.mockClear();
    appConfirm.mockResolvedValue(true);
    rescan.mockReset();
    rescan.mockResolvedValue(statusFor("ready", null));
    update.mockReset();
    update.mockResolvedValue({ rootId: "root-a" });
    scanResult.mockReset();
    scanResult.mockReturnValue({ root: { rootId: "root-a" }, available: true, candidates: [], unmatched: [], diagnostics: [] });
    deleteModel = vi.fn(async () => ({ removedManifestEntry: true }));
    window.__TAURI__ = { core: { invoke: vi.fn(async () => "/Volumes/Moved") } };
    vi.resetModules();
    ({ AppContext } = await import("../context/AppContext.js"));
    ({ ModelManagerScreen } = await import("./ModelManagerScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function render(models) {
    await act(async () => {
      root.render(
        <AppContext.Provider
          value={{
            activeProject: null,
            jobs: [],
            loras: [],
            models,
            presets: [],
            token: "tok",
            jobAction: () => {},
            setActiveView: () => {},
            deleteLora: () => {},
            deleteModel,
            createModelDownloadJob: () => {},
            createModelConvertJob: () => {},
            createLoraImportJob: () => {},
            createModelImportJob: () => {},
          }}
        >
          <ModelManagerScreen />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  function card(name) {
    return [...container.querySelectorAll(".model-card")].find((node) => node.textContent.includes(name));
  }

  function buttonIn(scope, name) {
    return [...scope.querySelectorAll("button")].find((node) => node.textContent.trim() === name);
  }

  async function click(node) {
    expect(node, "the control under test exists").toBeTruthy();
    await act(async () => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  }

  it("renders Needs relink with a relink button instead of the missing badge", async () => {
    scanResult.mockReturnValue({
      root: { rootId: "root-a" },
      available: false,
      candidates: [],
      unmatched: [statusFor("needs_relink", "[checkpoint-plan:root-unavailable] /Volumes/Models is not available")],
      diagnostics: [],
    });
    await render([LINKED_MODEL]);
    const node = card("My SDXL");
    const badge = node.querySelector(".model-card-status .status-badge");
    expect(badge.textContent).toBe("needs relink");
    expect(node.querySelector(".model-card-status").textContent).not.toContain("missing");
    const fix = node.querySelector('[role="group"][aria-label="My SDXL Needs relink"]');
    expect(fix).toBeTruthy();
    expect(fix.textContent).toContain("[checkpoint-plan:root-unavailable]");
    expect(buttonIn(fix, "Relink library")).toBeTruthy();
  });

  it("renders Needs rescan and clears it through the rescan route", async () => {
    scanResult.mockReturnValue({
      root: { rootId: "root-a" },
      available: true,
      candidates: [
        {
          checkpointId: "linked/root-a/sdxl.safetensors",
          candidate: { relativePath: "sdxl.safetensors" },
          status: statusFor("needs_rescan", "[checkpoint-plan:source-drifted] digests differ"),
          selectable: false,
        },
      ],
      unmatched: [],
      diagnostics: [],
    });
    await render([LINKED_MODEL]);
    const node = card("My SDXL");
    expect(node.querySelector(".model-card-status .status-badge").textContent).toBe("needs rescan");
    const fix = node.querySelector('[role="group"][aria-label="My SDXL Needs rescan"]');
    await click(buttonIn(fix, "Rescan checkpoint"));
    expect(rescan).toHaveBeenCalledWith("tok", "root-a", "sdxl.safetensors");
  });

  it("relinks through the library route with the folder the desktop bridge returned", async () => {
    scanResult.mockReturnValue({
      root: { rootId: "root-a" },
      available: false,
      candidates: [],
      unmatched: [statusFor("needs_relink", "")],
      diagnostics: [],
    });
    await render([LINKED_MODEL]);
    await click(buttonIn(card("My SDXL"), "Relink library"));
    expect(update).toHaveBeenCalledWith("tok", "root-a", { path: "/Volumes/Moved" });
  });

  it("names the source a plan-backed row was imported from", async () => {
    await render([LINKED_MODEL, MANAGED_MODEL]);
    expect(card("My SDXL").querySelector(".model-card-provenance").textContent).toContain("Linked library");
    expect(card("Fetched Model").querySelector(".model-card-provenance").textContent).toContain("Civitai");
  });

  it("promises a linked delete never touches the user's files", async () => {
    await render([LINKED_MODEL]);
    await click(buttonIn(card("My SDXL"), "Delete"));
    const copy = appConfirm.mock.calls.at(-1)[0];
    expect(copy.title).toBe("Remove from SceneWorks?");
    expect(copy.confirmLabel).toBe("Remove");
    expect(copy.message).toMatch(/never opened, moved or deleted/);
    expect(copy.message).not.toMatch(/removes those files from this machine/);
    expect(deleteModel).toHaveBeenCalled();
  });

  it("warns that a managed delete removes the files, and says so only there", async () => {
    await render([MANAGED_MODEL]);
    await click(buttonIn(card("Fetched Model"), "Delete"));
    const copy = appConfirm.mock.calls.at(-1)[0];
    expect(copy.title).toBe("Delete model?");
    expect(copy.confirmLabel).toBe("Delete");
    expect(copy.message).toMatch(/removes those files from this machine/);
    expect(copy.message).not.toMatch(/never opened, moved or deleted/);
  });

  it("leaves a row with no import plan on the pre-epic confirmation", async () => {
    // A built-in catalog entry has no ownership to discriminate on; guessing "managed" over it
    // would overstate what its delete actually does.
    await render([{ ...MANAGED_MODEL, id: "builtin", name: "Builtin", importPlan: undefined, source: undefined, catalogScope: "builtin" }]);
    await click(buttonIn(card("Builtin"), "Delete"));
    const copy = appConfirm.mock.calls.at(-1)[0];
    expect(copy.message).toContain("Built-in catalog identity stays protected");
  });
});
