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

  // The panel has TWO status regions — the library-lifecycle track and the import track — because
  // the two run concurrently and neither may overwrite the other. Assertions that only care that a
  // sentence reached the user read both.
  function status() {
    return [...container.querySelectorAll('[role="status"]')].map((node) => node.textContent).join(" ");
  }

  function importStatus() {
    return container.querySelector(".checkpoint-import-status")?.textContent ?? "";
  }

  function libraryStatus() {
    return container.querySelector(".checkpoint-library-status")?.textContent ?? "";
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

  it("reports the stored credential positively rather than dropping the notice", async () => {
    await render({ credentials: [{ host: "civitai.com", present: true }] });
    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Civitai"));
    const notice = container.querySelector(".checkpoint-credential-notice");
    expect(notice.textContent).toMatch(/A civitai\.com credential is stored/);
    expect(notice.textContent).not.toMatch(/No civitai\.com credential/);
  });

  // The notice must distinguish "looked and found none" from "never looked". A caller that does
  // not read the keychain gets silence — the old `credentials = []` default made every such screen
  // claim a credential was missing.
  it("says nothing about credentials when the caller never read them", async () => {
    await render({});
    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Civitai"));
    expect(container.querySelector(".checkpoint-credential-notice")).toBeNull();
  });

  it("offers a way to reach Settings from the missing-credential notice", async () => {
    const onOpenSettings = vi.fn();
    await render({ credentials: [], onOpenSettings });
    await click(byRole("radio", "Add to SceneWorks"));
    await click(buttonNamed("Civitai"));
    await click(buttonNamed("Add token in Settings"));
    expect(onOpenSettings).toHaveBeenCalled();
  });

  // --------------------------------------------- AC3 (cont.): the two tracks never speak for each other

  it("keeps the import guard closed when a library scan resolves mid-import", async () => {
    let releaseScan;
    let releaseImport;
    const library = libraryStub({
      scan: vi.fn(async () => {
        // First scan (the mount auto-scan) resolves immediately; the second is held open so it can
        // be made to resolve DURING the import.
        if (library.scan.mock.calls.length > 1) {
          await new Promise((resolve) => {
            releaseScan = resolve;
          });
        }
        return readyScan();
      }),
    });
    const onImportModel = vi.fn(
      () =>
        new Promise((resolve) => {
          releaseImport = () => resolve({ payload: { modelId: "m" } });
        }),
    );
    await render({ library, onImportModel });

    await click(buttonNamed("Use this checkpoint"));
    // A second root scan starts while the import is still in flight, then finishes first.
    await click(buttonNamed("Rescan library"));
    await act(async () => {
      releaseScan?.();
    });

    // The import is still running, so its button is still disabled: a scan's `finally` no longer
    // clears the import guard, which is what allowed a double submit.
    const reuse = buttonNamed("Use this checkpoint");
    expect(reuse.disabled).toBe(true);
    await click(reuse);
    expect(onImportModel).toHaveBeenCalledTimes(1);

    await act(async () => {
      releaseImport?.();
    });
    expect(importStatus()).toContain("Import queued");
  });

  it("keeps an import error and its Try again through a later successful scan", async () => {
    const library = libraryStub();
    const onImportModel = vi.fn(async () => {
      throw apiError("[checkpoint-plan:unrunnable-source] no loader", {
        code: "checkpoint_library_rejected",
        reason: "unrunnable-source",
      });
    });
    await render({ library, onImportModel });
    await click(buttonNamed("Use this checkpoint"));
    expect(importStatus()).toContain("[checkpoint-plan:unrunnable-source]");

    await click(buttonNamed("Rescan library"));
    // The scan succeeded and said so on its OWN track; the import's failure is still on screen and
    // still actionable.
    expect(importStatus()).toContain("[checkpoint-plan:unrunnable-source]");
    expect(buttonNamed("Try again")).toBeTruthy();
  });

  // "Try again" re-POSTs `lastAttempt`. Gating it on any error tone meant a LIBRARY failure that
  // happened after a perfectly successful import offered to re-send that import.
  it("does not offer Try again beside a library failure after a successful import", async () => {
    const library = libraryStub({
      update: vi.fn(async () => {
        throw apiError("[checkpoint-plan:root-unavailable] /gone is not there", {
          code: "checkpoint_library_rejected",
          reason: "root-unavailable",
        });
      }),
    });
    const onImportModel = vi.fn(async () => ({ payload: { modelId: "m" } }));
    await render({ library, onImportModel, pickFolder: null });

    await click(buttonNamed("Use this checkpoint"));
    expect(importStatus()).toContain("Import queued");
    expect(buttonNamed("Try again")).toBeFalsy();

    await type(labelled("Library folder"), "/gone");
    await click(buttonNamed("Relink library"));
    expect(libraryStatus()).toContain("[checkpoint-plan:root-unavailable]");
    // The import track still reads "queued", so there is nothing to retry.
    expect(importStatus()).toContain("Import queued");
    expect(buttonNamed("Try again")).toBeFalsy();
  });

  it("still offers Try again on the import's own error while a library message is showing", async () => {
    const onImportModel = vi.fn(async () => {
      throw apiError("[checkpoint-plan:unrunnable-source] no loader", {
        code: "checkpoint_library_rejected",
        reason: "unrunnable-source",
      });
    });
    await render({ onImportModel });
    await click(buttonNamed("Use this checkpoint"));
    await click(buttonNamed("Remove library"));
    expect(libraryStatus()).toMatch(/Your files were not touched/);
    expect(buttonNamed("Try again")).toBeTruthy();
  });

  it("imports the RENDERED root, not one whose scan is still in flight", async () => {
    const SECOND = { rootId: "root-b", path: "/Volumes/Other", label: "", displayLabel: "Other" };
    let releaseSecond;
    const library = libraryStub({
      fetchRoots: vi.fn(async () => ({ roots: [ROOT, SECOND] })),
      scan: vi.fn(async (_token, rootId) => {
        if (rootId === "root-b") {
          await new Promise((resolve) => {
            releaseSecond = resolve;
          });
          return readyScan({ root: SECOND });
        }
        return readyScan();
      }),
    });
    const onImportModel = vi.fn(async () => ({ payload: { modelId: "m" } }));
    await render({ library, onImportModel });

    // Start root-b's scan; `activeRootId` moves to root-b immediately while root-b's answer is
    // still pending.
    const rescanButtons = [...container.querySelectorAll("button")].filter((node) => node.textContent === "Rescan library");
    await click(rescanButtons[1]);
    // Nothing from the stale root is rendered any more, so there is no wrong-root candidate to
    // click at all.
    expect(buttonNamed("Use this checkpoint")).toBeFalsy();

    await act(async () => {
      releaseSecond?.();
    });
    await click(buttonNamed("Use this checkpoint"));
    expect(onImportModel.mock.calls[0][0].linkedRootId).toBe("root-b");
  });

  // The candidate actions act on the root the RENDERED scan names, not on whatever id the panel
  // last asked about. The scan is the authority on which root its candidates belong to.
  it("acts on the root the rendered scan names, not the id the panel last requested", async () => {
    const canonical = { rootId: "root-a-canonical", path: "/Volumes/Models", displayLabel: "Models" };
    const library = libraryStub({
      scan: vi.fn(async () =>
        readyScan({
          root: canonical,
          candidates: [
            {
              checkpointId: "linked/root-a/drifted.safetensors",
              candidate: { relativePath: "drifted.safetensors" },
              status: { checkpointId: "linked/root-a/drifted.safetensors", rootId: "root-a-canonical", relativePath: "drifted.safetensors", state: "needs_rescan", detail: "[checkpoint-plan:source-drifted] digests differ" },
              selectable: false,
            },
            {
              checkpointId: "linked/root-a/sdxl.safetensors",
              candidate: { relativePath: "sdxl.safetensors" },
              status: { checkpointId: "linked/root-a/sdxl.safetensors", rootId: "root-a-canonical", relativePath: "sdxl.safetensors", state: "ready", detail: null },
              selectable: true,
            },
          ],
        }),
      ),
    });
    const onImportModel = vi.fn(async () => ({ payload: { modelId: "m" } }));
    await render({ library, onImportModel });

    await click(buttonNamed("Rescan checkpoint"));
    expect(library.rescan).toHaveBeenCalledWith("tok", "root-a-canonical", "drifted.safetensors");

    await click(buttonNamed("Use this checkpoint"));
    expect(onImportModel.mock.calls[0][0].linkedRootId).toBe("root-a-canonical");
  });

  // The panel's lifecycle actions change the same records the host screen's catalog renders, so
  // each one tells the host to re-read them.
  it("asks the host to refresh its catalog after relink, forget and rescan", async () => {
    const onRefreshCatalog = vi.fn();
    const library = libraryStub({
      scan: vi.fn(async () => readyScan({ unmatched: [{ checkpointId: "c", rootId: "root-a", relativePath: "gone.safetensors", state: "needs_rescan", detail: null }] })),
    });
    await render({ library, onRefreshCatalog, pickFolder: null });
    await type(labelled("Library folder"), "/Volumes/Moved");
    await click(buttonNamed("Relink library"));
    expect(onRefreshCatalog).toHaveBeenCalledTimes(1);
    await click(buttonNamed("Rescan checkpoint"));
    expect(onRefreshCatalog).toHaveBeenCalledTimes(2);
    await click(buttonNamed("Remove library"));
    expect(onRefreshCatalog).toHaveBeenCalledTimes(3);
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
