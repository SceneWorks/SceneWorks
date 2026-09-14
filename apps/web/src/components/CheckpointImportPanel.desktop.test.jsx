import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The desktop-bridge half of AC1 (epic 20398, sc-20650).
//
// `runtime.js` derives `isDesktop` from `window.__TAURI__` AT MODULE LOAD, and the panel's default
// `pickFolder` is derived from it at module load too. So the bridge has to be installed BEFORE the
// module graph is imported — hence the per-test `vi.resetModules()` + dynamic import, the same
// pattern ModelManagerScreen.test.jsx uses for the keychain transport.
//
// Two claims, and the second is the one that would rot silently: the desktop build reaches the
// NATIVE folder chooser rather than asking the user to type an absolute path, and a remote browser
// on the same build reaches the typed field instead of a dead button that can never resolve.

const ROOT = { rootId: "root-a", path: "/Volumes/Models", label: "", displayLabel: "Models" };

const LIBRARY = () => ({
  fetchRoots: vi.fn(async () => ({ roots: [] })),
  approve: vi.fn(async () => ROOT),
  update: vi.fn(async () => ROOT),
  remove: vi.fn(async () => ({ root: ROOT, removedCheckpoints: [] })),
  scan: vi.fn(async () => ({ root: ROOT, available: true, candidates: [], unmatched: [], diagnostics: [] })),
  rescan: vi.fn(async () => ({ state: "ready" })),
});

describe("CheckpointImportPanel desktop bridge", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    vi.resetModules();
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function mount(props) {
    const { AppContext } = await import("../context/AppContext.js");
    const { CheckpointImportPanel } = await import("./CheckpointImportPanel.jsx");
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ jobs: [], models: [], workersById: {}, visibleWorkers: [] }}>
          <CheckpointImportPanel defaultOpen library={LIBRARY()} onImportModel={vi.fn()} token="tok" {...props} />
        </AppContext.Provider>,
      );
    });
  }

  function buttonNamed(name) {
    return [...container.querySelectorAll("button")].find((node) => node.textContent.trim() === name);
  }

  async function click(node) {
    expect(node, "the control under test exists").toBeTruthy();
    await act(async () => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  }

  it("picks the library folder through the desktop bridge's choose_folder command", async () => {
    const invoke = vi.fn(async (command) => (command === "choose_folder" ? "/Volumes/Picked" : null));
    window.__TAURI__ = { core: { invoke } };
    const library = LIBRARY();
    await mount({ library });

    await click(buttonNamed("Choose folder"));
    expect(invoke).toHaveBeenCalledWith("choose_folder", undefined);

    await click(buttonNamed("Add library"));
    expect(library.approve).toHaveBeenCalledWith("tok", { path: "/Volumes/Picked", label: undefined });
  });

  it("picks the managed local-copy source through the same bridge", async () => {
    const invoke = vi.fn(async () => "/Users/me/checkpoints");
    window.__TAURI__ = { core: { invoke } };
    const onImportModel = vi.fn(async () => ({ payload: { modelId: "m" } }));
    await mount({ onImportModel });

    await click([...container.querySelectorAll('[role="radio"]')].find((node) => node.textContent.startsWith("Add to SceneWorks")));
    await click(buttonNamed("Local copy"));
    await click(buttonNamed("Choose source folder"));
    expect(invoke).toHaveBeenCalledWith("choose_folder", undefined);

    await click(buttonNamed("Queue Import"));
    expect(onImportModel).toHaveBeenCalledWith({
      ownershipMode: "managed",
      type: "image",
      source: { kind: "localPath", path: "/Users/me/checkpoints" },
    });
  });

  it("survives a cancelled native picker without wedging the form", async () => {
    // The chooser resolves null when the user dismisses it, and rejects if the ACL grant is
    // missing. Neither may leave the panel stuck or clear a path the user already typed.
    const invoke = vi.fn(async () => {
      throw new Error("forbidden");
    });
    window.__TAURI__ = { core: { invoke } };
    const library = LIBRARY();
    await mount({ library });

    await click(buttonNamed("Choose folder"));
    await click(buttonNamed("Add library"));
    // No path was ever chosen, so the panel refuses rather than posting an empty one.
    expect(library.approve).not.toHaveBeenCalled();
    expect(container.querySelector('[role="status"]').textContent).toMatch(/folder your checkpoints live in/);
  });

  it("falls back to a typed path in a remote browser (no bridge)", async () => {
    const library = LIBRARY();
    await mount({ library });
    expect(buttonNamed("Choose folder")).toBeUndefined();
    const input = [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith("Library folder"))
      .querySelector("input");
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    await act(async () => {
      setter.call(input, "/srv/models");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await click(buttonNamed("Add library"));
    expect(library.approve).toHaveBeenCalledWith("tok", { path: "/srv/models", label: undefined });
  });
});
