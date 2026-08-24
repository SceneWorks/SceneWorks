import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { CheckpointImportPanel } from "./CheckpointImportPanel.jsx";

// Drives the WHOLE unified experience — both ownerships, the linked lifecycle, the five managed
// inputs, cancel, retry, typed-refusal surfacing, and the accessibility contract — against injected
// transports. Nothing here asserts on a snapshot: every accessibility claim queries a role or an
// accessible name, because a snapshot passes just as happily when the roles are gone.

vi.mock("../appConfirm.jsx", () => ({ appConfirm: vi.fn(async () => true) }));

const { appConfirm } = await import("../appConfirm.jsx");

const ROOT = { rootId: "root-a", path: "/Volumes/Models", label: "", displayLabel: "Models" };

function readyScan(overrides = {}) {
  return {
    root: ROOT,
    available: true,
    candidates: [
      {
        checkpointId: "linked/root-a/sdxl.safetensors",
        candidate: { relativePath: "sdxl.safetensors", container: "safetensors", sizeBytes: 6543210, headerFamily: "sdxl" },
        status: { checkpointId: "linked/root-a/sdxl.safetensors", rootId: "root-a", relativePath: "sdxl.safetensors", state: "ready", detail: null },
        selectable: true,
      },
    ],
    unmatched: [],
    diagnostics: [],
    ...overrides,
  };
}

function apiError(message, { code, reason, status = 409 } = {}) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  error.context = reason ? { reason } : null;
  return error;
}

function libraryStub(overrides = {}) {
  return {
    fetchRoots: vi.fn(async () => ({ roots: [ROOT] })),
    approve: vi.fn(async () => ROOT),
    update: vi.fn(async () => ROOT),
    remove: vi.fn(async () => ({ root: ROOT, removedCheckpoints: ["linked/root-a/sdxl.safetensors"] })),
    scan: vi.fn(async () => readyScan()),
    rescan: vi.fn(async () => ({ checkpointId: "linked/root-a/sdxl.safetensors", rootId: "root-a", relativePath: "sdxl.safetensors", state: "ready", detail: null })),
    ...overrides,
  };
}

describe("CheckpointImportPanel", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    appConfirm.mockClear();
    appConfirm.mockResolvedValue(true);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  async function render(props = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={{ jobs: [], models: [], workersById: {}, visibleWorkers: [] }}>
          <CheckpointImportPanel
            defaultOpen
            library={libraryStub()}
            onImportModel={async () => ({ payload: { modelId: "imported" } })}
            token="tok"
            {...props}
          />
        </AppContext.Provider>,
      );
    });
  }

  function byRole(role, name) {
    return [...container.querySelectorAll(`[role="${role}"], ${role === "button" ? "button" : role}`)].find((node) =>
      name ? accessibleName(node) === name || accessibleName(node).includes(name) : true,
    );
  }

  function accessibleName(node) {
    return (node.getAttribute("aria-label") ?? node.textContent ?? "").trim();
  }

  function buttonNamed(name) {
    return [...container.querySelectorAll("button")].find((node) => node.textContent.trim() === name);
  }

  async function click(node) {
    expect(node, "the control under test exists").toBeTruthy();
    await act(async () => {
      node.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  }

  async function type(input, value) {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    await act(async () => {
      setter.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  function labelled(text) {
    return [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith(text))
      ?.querySelector("input, select");
  }

  function status() {
    return container.querySelector('[role="status"]')?.textContent ?? "";
  }

  // ------------------------------------------------------------------- AC1: the two choices

  it("offers both ownership choices as one radio group, linked selected first", async () => {
    await render();
    const group = container.querySelector('[role="radiogroup"][aria-label="Model ownership"]');
    expect(group).toBeTruthy();
    const options = [...group.querySelectorAll('[role="radio"]')];
    expect(options.map((node) => node.querySelector(".checkpoint-ownership-label").textContent)).toEqual([
      "Use existing model library",
      "Add to SceneWorks",
    ]);
    expect(options[0].getAttribute("aria-checked")).toBe("true");
    expect(options[1].getAttribute("aria-checked")).toBe("false");
  });

  it("is collapsed until asked, and the toggle reports its own state", async () => {
    await render({ defaultOpen: false });
    const toggle = container.querySelector(".checkpoint-import-toggle");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(container.querySelector('[role="radiogroup"]')).toBeNull();
    await click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(container.querySelector('[role="radiogroup"][aria-label="Model ownership"]')).toBeTruthy();
  });

  it("reaches all five managed inputs from the Add to SceneWorks choice", async () => {
    await render();
    await click(byRole("radio", "Add to SceneWorks"));
    const sources = container.querySelector('[role="radiogroup"][aria-label="Where the checkpoint comes from"]');
    expect([...sources.querySelectorAll('[role="radio"]')].map((node) => node.textContent)).toEqual([
      "Upload",
      "Local copy",
      "URL",
      "Hugging Face",
      "Civitai",
    ]);
  });

  it("runs both ownerships through the same import submission", async () => {
    const onImportModel = vi.fn(async () => ({ payload: { modelId: "m" } }));
    await render({ onImportModel });
    await click(buttonNamed("Use this checkpoint"));
    expect(onImportModel).toHaveBeenCalledWith({
      linkedRootId: "root-a",
      linkedRelativePath: "sdxl.safetensors",
      type: "image",
      name: "sdxl.safetensors",
    });

    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Hugging Face"));
    await type(labelled("Hugging Face repo"), "org/model");
    await click(buttonNamed("Queue Import"));
    expect(onImportModel).toHaveBeenLastCalledWith({
      ownershipMode: "managed",
      type: "image",
      source: { kind: "huggingFace", repo: "org/model" },
    });
  });

  // ------------------------------------------------------- AC1: linked-library management

  it("adds, rescans, relinks, renames and removes a library root", async () => {
    const library = libraryStub();
    await render({ library });
    await type(labelled("Library folder"), "/Volumes/Models");
    await click(buttonNamed("Add library"));
    expect(library.approve).toHaveBeenCalledWith("tok", { path: "/Volumes/Models", label: undefined });

    await click(buttonNamed("Rescan library"));
    expect(library.scan).toHaveBeenCalledWith("tok", "root-a");

    await type(labelled("Library folder"), "/Volumes/Moved");
    await click(buttonNamed("Relink library"));
    expect(library.update).toHaveBeenCalledWith("tok", "root-a", { path: "/Volumes/Moved" });

    await click(buttonNamed("Rename"));
    await type(labelled("New name"), "Big drive");
    await click(buttonNamed("Save name"));
    expect(library.update).toHaveBeenLastCalledWith("tok", "root-a", { label: "Big drive" });

    await click(buttonNamed("Remove library"));
    expect(library.remove).toHaveBeenCalledWith("tok", "root-a");
  });

  it("uses the desktop bridge folder picker when one is available", async () => {
    const pickFolder = vi.fn(async () => "/Volumes/Picked");
    const library = libraryStub();
    await render({ library, pickFolder });
    await click(buttonNamed("Choose folder"));
    expect(pickFolder).toHaveBeenCalled();
    await click(buttonNamed("Add library"));
    expect(library.approve).toHaveBeenCalledWith("tok", { path: "/Volumes/Picked", label: undefined });
  });

  it("types the path instead when there is no bridge (remote browser)", async () => {
    const library = libraryStub();
    await render({ library, pickFolder: null });
    expect(buttonNamed("Choose folder")).toBeUndefined();
    await type(labelled("Library folder"), "/srv/models");
    await click(buttonNamed("Add library"));
    expect(library.approve).toHaveBeenCalledWith("tok", { path: "/srv/models", label: undefined });
  });

  // --------------------------------------------- AC2: corrective action, not "missing"

  it("shows Needs Relink with a relink button instead of calling the library missing", async () => {
    const library = libraryStub({ scan: vi.fn(async () => readyScan({ available: false, candidates: [] })) });
    await render({ library });
    const region = container.querySelector('[role="group"][aria-label="Library needs relink"]');
    expect(region).toBeTruthy();
    expect(region.textContent).toContain("Needs relink");
    expect(region.textContent).not.toMatch(/uninstalled|not installed|missing/i);
    expect([...region.querySelectorAll("button")].map((node) => node.textContent)).toContain("Relink library");
  });

  it("shows Needs Rescan on a drifted checkpoint with the store's own diagnostic", async () => {
    const drifted = readyScan();
    drifted.candidates[0].status = {
      ...drifted.candidates[0].status,
      state: "needs_rescan",
      detail: "[checkpoint-plan:source-drifted] recorded ab… now cd…",
    };
    drifted.candidates[0].selectable = false;
    const library = libraryStub({ scan: vi.fn(async () => drifted) });
    await render({ library });
    const region = container.querySelector('[role="group"][aria-label="sdxl.safetensors Needs rescan"]');
    expect(region).toBeTruthy();
    expect(region.textContent).toContain("[checkpoint-plan:source-drifted]");
    expect(region.textContent).not.toMatch(/uninstalled|not installed/i);
    await click([...region.querySelectorAll("button")].find((node) => node.textContent === "Rescan checkpoint"));
    expect(library.rescan).toHaveBeenCalledWith("tok", "root-a", "sdxl.safetensors");
  });

  it("keeps persisted checkpoints the scan no longer sees, with a rescan affordance", async () => {
    const library = libraryStub({
      scan: vi.fn(async () =>
        readyScan({
          candidates: [],
          unmatched: [
            { checkpointId: "linked/root-a/gone.safetensors", rootId: "root-a", relativePath: "gone.safetensors", state: "needs_rescan", detail: "[checkpoint-plan:source-missing] gone" },
          ],
        }),
      ),
    });
    await render({ library });
    const region = container.querySelector('[role="group"][aria-label="Checkpoints no longer found"]');
    expect(region.textContent).toContain("gone.safetensors");
    expect(region.textContent).toContain("[checkpoint-plan:source-missing]");
    expect(region.textContent).not.toMatch(/uninstalled/i);
  });

  it("shows capabilities and backend eligibility for a compiled checkpoint", async () => {
    await render({
      models: [
        {
          id: "my-sdxl",
          capabilities: ["text_to_image", "style_variations"],
          macSupport: { supported: false, reason: "torch_only" },
          importPlan: { checkpointId: "linked/root-a/sdxl.safetensors" },
        },
      ],
      macCapabilities: { macGatingActive: true },
    });
    const chips = [...container.querySelectorAll('[aria-label="sdxl.safetensors capabilities"] .chip')].map((node) => node.textContent);
    // The retired mode is dropped rather than advertised as something the studios can run.
    expect(chips).toEqual(["Text to Image"]);
    expect(container.querySelector(".checkpoint-eligibility").textContent).toBeTruthy();
  });

  it("warns about duplicate checkpoints without calling the import a failure", async () => {
    await render({
      completedJobs: [{ id: "j1", result: { duplicateCheckpointIds: ["managed/install-9"] } }],
    });
    const note = container.querySelector(".checkpoint-duplicate-warning");
    expect(note.textContent).toContain("managed/install-9");
    expect(note.textContent).not.toMatch(/failed|error/i);
  });

  it("names the library it is forgetting and promises the files are untouched", async () => {
    const library = libraryStub();
    await render({ library });
    await click(buttonNamed("Remove library"));
    expect(appConfirm).toHaveBeenCalled();
    const copy = appConfirm.mock.calls[0][0];
    expect(copy.confirmLabel).toBe("Forget library");
    expect(copy.message).toMatch(/left exactly as they are/);
    expect(status()).toMatch(/Your files were not touched/);
  });

  it("does not remove a library when the confirmation is declined", async () => {
    appConfirm.mockResolvedValue(false);
    const library = libraryStub();
    await render({ library });
    await click(buttonNamed("Remove library"));
    expect(library.remove).not.toHaveBeenCalled();
  });

  // ------------------------------------------------------- AC3: error, retry, cancel, a11y

  it("surfaces the typed refusal reason rather than a generic failure", async () => {
    const library = libraryStub({
      approve: vi.fn(async () => {
        throw apiError("[checkpoint-plan:root-not-approvable] /nope cannot be approved as a root: not a directory", {
          code: "checkpoint_library_rejected",
          reason: "root-not-approvable",
          status: 400,
        });
      }),
    });
    await render({ library, pickFolder: null });
    await type(labelled("Library folder"), "/nope");
    await click(buttonNamed("Add library"));
    expect(status()).toContain("[checkpoint-plan:root-not-approvable]");
    expect(status()).not.toMatch(/something went wrong/i);
  });

  it("explains the local-only refusal in its own words and keeps the server sentence", async () => {
    const library = libraryStub({
      approve: vi.fn(async () => {
        throw apiError("A model library can only be added or relinked from SceneWorks running on this machine.", {
          code: "checkpoint_library_not_permitted",
          reason: "not_a_local_client",
          status: 403,
        });
      }),
    });
    await render({ library, pickFolder: null });
    await type(labelled("Library folder"), "/srv/models");
    await click(buttonNamed("Add library"));
    expect(status()).toMatch(/running on this machine/);
  });

  it("retries the exact submission that failed, not whatever the form holds later", async () => {
    const onImportModel = vi
      .fn()
      .mockRejectedValueOnce(apiError("[checkpoint-plan:unrunnable-source] no loader", {
        code: "checkpoint_library_rejected",
        reason: "unrunnable-source",
      }))
      .mockResolvedValueOnce({ payload: { modelId: "m" } });
    await render({ onImportModel });
    await click(buttonNamed("Use this checkpoint"));
    expect(status()).toContain("[checkpoint-plan:unrunnable-source]");
    await click(buttonNamed("Try again"));
    expect(onImportModel).toHaveBeenCalledTimes(2);
    expect(onImportModel.mock.calls[1][0]).toEqual(onImportModel.mock.calls[0][0]);
    expect(status()).toContain("Import queued");
  });

  it("hands a queued import to the progress card so it can be cancelled", async () => {
    const onCancelJob = vi.fn();
    await render({
      pendingJobs: [{ id: "job-1", type: "model_import", status: "running", payload: { modelId: "m" } }],
      onCancelJob,
    });
    const cancel = [...container.querySelectorAll("button")].find((node) => /cancel/i.test(node.textContent));
    expect(cancel).toBeTruthy();
    await click(cancel);
    expect(onCancelJob).toHaveBeenCalled();
  });

  it("refuses an empty managed form by naming the missing field", async () => {
    const onImportModel = vi.fn();
    await render({ onImportModel });
    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Queue Import"));
    expect(onImportModel).not.toHaveBeenCalled();
    expect(status()).toMatch(/file/i);
  });

  it("warns when the selected source needs a credential that is not stored", async () => {
    await render({ credentials: [] });
    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Civitai"));
    expect(container.querySelector(".checkpoint-credential-notice").textContent).toContain("civitai.com");
    await click(buttonNamed("Hugging Face"));
    expect(container.querySelector(".checkpoint-credential-notice").textContent).toContain("huggingface.co");
  });

  it("drops the credential notice once the credential is stored", async () => {
    await render({ credentials: [{ host: "civitai.com", present: true }] });
    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Civitai"));
    expect(container.querySelector(".checkpoint-credential-notice")).toBeNull();
  });

  it("labels every region and announces outcomes in a live region", async () => {
    await render();
    expect(container.querySelector("section").getAttribute("aria-labelledby")).toBe("checkpoint-import-heading");
    expect(container.querySelector("#checkpoint-import-heading").textContent).toBe("Add a model");
    expect(container.querySelector('[aria-label="Linked libraries"]')).toBeTruthy();
    expect(container.querySelector('[aria-label="Checkpoints in this library"]')).toBeTruthy();
    const live = container.querySelector('[role="status"]');
    expect(live.getAttribute("aria-live")).toBe("polite");
  });
});
