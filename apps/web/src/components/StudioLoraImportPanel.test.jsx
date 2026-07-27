import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";
import { StudioLoraImportPanel } from "./StudioLoraImportPanel.jsx";

// sc-15373 — the Model Manager import form, lifted into the studio's Add-LoRA picker. What must
// hold: it is NOT a <form> (the studio screen is already one, and nested forms are invalid), it
// submits through the same `createLoraImportJob` seam with the same payload shape, and it clears
// itself for a second import.
const models = [
  { id: "z_image_turbo", family: "z-image" },
  { id: "qwen_image", family: "qwen-image" },
];

const button = (container, text) =>
  [...container.querySelectorAll("button")].find((node) => node.textContent.trim() === text);
const labelled = (container, text) =>
  [...container.querySelectorAll("label")]
    .find((node) => node.textContent.trim().startsWith(text))
    ?.querySelector("input, select, textarea");

describe("StudioLoraImportPanel", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function render(props = {}) {
    await act(async () => {
      root.render(
        <StudioLoraImportPanel
          activeProject={{ id: "project-1", name: "Noir" }}
          createLoraImportJob={vi.fn()}
          models={models}
          {...props}
        />,
      );
    });
  }

  it("renders as a group, never a nested form", async () => {
    await render();
    expect(container.querySelector("form")).toBeNull();
    expect(container.querySelector('[role="group"][aria-label="Import LoRA"]')).toBeTruthy();
  });

  it("offers Auto-detect plus every model family", async () => {
    await render();
    expect([...labelled(container, "Family").options].map((option) => option.value)).toEqual([
      "",
      "qwen-image",
      "z-image",
    ]);
  });

  it("queues a URL import with the trigger keywords and clears for the next one", async () => {
    const createLoraImportJob = vi.fn(async () => ({
      payload: { loraId: "harbor_dusk", manifestEntry: { family: "z_image" } },
    }));
    await render({ createLoraImportJob });

    await click(button(container, "URL"));
    await act(async () => setInput(labelled(container, "Source URL"), "https://example.test/harbor.safetensors"));
    const keywords = container.querySelector(".kw-input");
    await act(async () => setInput(keywords, "sksStyle"));
    await act(async () => {
      keywords.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "Enter" }));
    });
    await click(button(container, "Queue import"));

    expect(createLoraImportJob).toHaveBeenCalledTimes(1);
    expect(createLoraImportJob.mock.calls[0][0]).toMatchObject({
      sourceUrl: "https://example.test/harbor.safetensors",
      scope: "project",
      triggerWords: ["sksStyle"],
    });
    // The detected family is reported in the dropdown's own normalized vocabulary.
    expect(container.textContent).toContain("LoRA import queued for harbor_dusk.");
    expect(container.textContent).toContain("Detected family: z-image.");
    expect(labelled(container, "Source URL").value).toBe("");
  });

  it("cannot submit an empty source, or a project import with no project open", async () => {
    const createLoraImportJob = vi.fn();
    await render({ createLoraImportJob, activeProject: null });

    expect(button(container, "Queue import").disabled).toBe(true);
    await click(button(container, "URL"));
    await act(async () => setInput(labelled(container, "Source URL"), "https://example.test/harbor.safetensors"));
    // With no project open the panel defaults to global scope, so the URL alone unblocks it.
    expect(labelled(container, "Scope").value).toBe("global");
    expect(button(container, "Queue import").disabled).toBe(false);

    await act(async () => setSelect(labelled(container, "Scope"), "project"));
    expect(button(container, "Queue import").disabled).toBe(true);
    expect(container.textContent).toContain("Open a project before importing a project LoRA.");
    expect(createLoraImportJob).not.toHaveBeenCalled();
  });

  it("surfaces the import error instead of clearing the form", async () => {
    const createLoraImportJob = vi.fn(async () => {
      throw new Error("Unsupported adapter format");
    });
    await render({ createLoraImportJob });

    await click(button(container, "URL"));
    await act(async () => setInput(labelled(container, "Source URL"), "https://example.test/bad.safetensors"));
    await click(button(container, "Queue import"));

    expect(container.querySelector(".inline-warning").textContent).toContain("Unsupported adapter format");
    expect(labelled(container, "Source URL").value).toBe("https://example.test/bad.safetensors");
  });
});
