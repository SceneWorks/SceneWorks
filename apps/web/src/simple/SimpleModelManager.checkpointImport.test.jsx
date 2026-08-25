import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleModelManager } from "./SimpleModelManager.jsx";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

// AC1 for the Simple shell (epic 20398, sc-20650): the two ownership choices, the linked-library
// lifecycle and the five managed inputs must be reachable from Simple as well — not just from the
// advanced Models screen. Simple hands off almost everything to the advanced shell, so this is the
// test that keeps the ONE exception honest.
//
// Driven through the real component and the real AppContext, with only the library transport and
// the desktop bridge injected — so what is asserted is what a Simple user can actually reach.

describe("SimpleModelManager checkpoint import (sc-20650)", () => {
  let container;
  let root;
  let createModelImportJob;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
    createModelImportJob = vi.fn(async () => ({ payload: { modelId: "imported" } }));
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.restoreAllMocks();
  });

  async function render(context = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider
          value={{
            models: [],
            loras: [],
            jobs: [],
            token: "tok",
            createModelDownloadJob: vi.fn(),
            createLoraDownloadJob: vi.fn(),
            createModelImportJob,
            jobAction: vi.fn(),
            macCapabilities: {},
            visibleWorkers: [],
            workersById: {},
            ...context,
          }}
        >
          <SimpleUiContext.Provider value={{ toast: vi.fn(), openInAdvanced: vi.fn() }}>
            <SimpleModelManager />
          </SimpleUiContext.Provider>
        </AppContext.Provider>,
      );
    });
  }

  function section() {
    return container.querySelector(".checkpoint-import");
  }

  async function click(node) {
    expect(node, "the control under test exists").toBeTruthy();
    await act(async () => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  }

  function buttonNamed(name) {
    return [...container.querySelectorAll("button")].find((node) => node.textContent.trim() === name);
  }

  it("offers the two ownership choices from Simple, collapsed until asked", async () => {
    await render();
    expect(section()).toBeTruthy();
    expect(section().classList.contains("compact")).toBe(true);
    const toggle = section().querySelector(".checkpoint-import-toggle");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    // Collapsed means the catalog is still the page's subject.
    expect(container.querySelector('[role="radiogroup"]')).toBeNull();

    await click(toggle);
    const choices = [...container.querySelectorAll('[aria-label="Model ownership"] [role="radio"]')];
    expect(choices.map((node) => node.querySelector(".checkpoint-ownership-label").textContent)).toEqual([
      "Use existing model library",
      "Add to SceneWorks",
    ]);
  });

  it("reaches all five managed inputs from Simple", async () => {
    await render();
    await click(section().querySelector(".checkpoint-import-toggle"));
    await click([...container.querySelectorAll('[role="radio"]')].find((node) => node.textContent.startsWith("Add to SceneWorks")));
    const sources = container.querySelector('[aria-label="Where the checkpoint comes from"]');
    expect([...sources.querySelectorAll('[role="radio"]')].map((node) => node.textContent)).toEqual([
      "Upload",
      "Local copy",
      "URL",
      "Hugging Face",
      "Civitai",
    ]);
  });

  it("keeps the catalog rows Simple already had", async () => {
    await render({ models: [{ id: "z", name: "Z-Image", type: "image", installState: "installed" }] });
    expect(container.textContent).toContain("Z-Image");
    expect(buttonNamed("Manage")).toBeTruthy();
  });

  it("shows a running import and its cancel control without opening the disclosure", async () => {
    const jobAction = vi.fn();
    await render({
      jobs: [{ id: "job-1", type: "model_import", status: "running", payload: { modelId: "m" } }],
      jobAction,
    });
    expect(section().querySelector(".checkpoint-import-toggle").getAttribute("aria-expanded")).toBe("false");
    expect(container.textContent).toContain("Imports in progress");
    const cancel = [...container.querySelectorAll("button")].find((node) => /cancel/i.test(node.textContent));
    await click(cancel);
    expect(jobAction).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1" }), "cancel");
  });

  // Simple mounts the SAME panel, so it has to feed it the same inputs. Without a `families` list
  // the Family select is permanently disabled and reads "No known families" — a reduced copy of the
  // panel, which is exactly what mounting the shared component is supposed to prevent.
  it("offers the real family list in the managed pane", async () => {
    await render({
      models: [
        { id: "z", name: "Z-Image", type: "image", installState: "installed", loraCompatibility: { families: ["sdxl"] } },
        { id: "f", name: "Flux", type: "image", installState: "installed", loraCompatibility: { families: ["flux2"] } },
      ],
    });
    await click(section().querySelector(".checkpoint-import-toggle"));
    await click([...container.querySelectorAll('[role="radio"]')].find((node) => node.textContent.startsWith("Add to SceneWorks")));
    const family = [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith("Family"))
      .querySelector("select");
    expect(family.disabled).toBe(false);
    expect([...family.options].map((option) => option.value)).toEqual(["", "flux2", "sdxl"]);
  });

  // Simple never reads the keychain, so the panel must not claim a credential is missing on its
  // behalf. (Simple renders no linked-status badge of its own; its catalog refresh is wired to the
  // context's `refreshData`, asserted through the panel's own onRefreshCatalog contract.)
  it("makes no credential claim it never checked", async () => {
    await render();
    await click(section().querySelector(".checkpoint-import-toggle"));
    await click([...container.querySelectorAll('[role="radio"]')].find((node) => node.textContent.startsWith("Add to SceneWorks")));
    await click(buttonNamed("Civitai"));
    expect(container.querySelector(".checkpoint-credential-notice")).toBeNull();
  });

  // The submit path. Every other test here proves the panel RENDERS from Simple; none of them
  // proved Simple actually hands it a way to enqueue, so dropping `onImportModel` from the mount
  // left the whole disclosure inert and no test noticed. Driving a real managed submission through
  // to the context's `createModelImportJob` is what makes that wire load-bearing.
  it("enqueues a managed import through the context's createModelImportJob", async () => {
    await render();
    await click(section().querySelector(".checkpoint-import-toggle"));
    await click(
      [...container.querySelectorAll('[role="radio"]')].find((node) =>
        node.textContent.startsWith("Add to SceneWorks"),
      ),
    );
    await click(buttonNamed("Hugging Face"));
    const repo = [...container.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith("Hugging Face repo"))
      .querySelector("input");
    const setValue = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    await act(async () => {
      setValue.call(repo, "org/model");
      repo.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(createModelImportJob).not.toHaveBeenCalled();

    await click(buttonNamed("Queue Import"));

    expect(createModelImportJob).toHaveBeenCalledTimes(1);
    expect(createModelImportJob).toHaveBeenCalledWith({
      ownershipMode: "managed",
      type: "image",
      source: { kind: "huggingFace", repo: "org/model" },
    });
  });

  it("surfaces a duplicate-checkpoint warning from a completed import", async () => {
    await render({
      jobs: [{ id: "job-2", type: "model_import", status: "completed", result: { duplicateCheckpointIds: ["managed/i-1"] } }],
    });
    expect(container.querySelector(".checkpoint-duplicate-warning").textContent).toContain("managed/i-1");
  });
});
