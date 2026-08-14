import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { EDIT_PROMPT_TEMPLATES } from "../data/editPromptTemplates.js";

// The built-in edit instructions on Simple's Edit tab. Simple has no advanced Instruction
// box and no style axis for editing, so these pills are the only starting point it offers —
// and they must stay off the Text tab, where "deblur this image" is not a scene.

const IMAGE_MODEL = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image", "edit_image"],
  installState: "installed",
  limits: { resolutions: ["1024x1024"] },
};

function context() {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    jobs: [],
    imageModels: [IMAGE_MODEL],
    videoModels: [],
    audioModels: [],
    models: [IMAGE_MODEL],
    loras: [],
    imageLocalJobs: [],
    videoLocalJobs: [],
    audioLocalJobs: [],
    visibleWorkers: [],
    macCapabilities: null,
    theme: "light",
    changeTheme: () => {},
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createVideoJob: vi.fn(async () => null),
    createAudioJob: vi.fn(async () => null),
    createModelDownloadJob: vi.fn(async () => null),
    createLoraDownloadJob: vi.fn(async () => null),
    createLoraImportJob: vi.fn(async () => null),
    jobAction: vi.fn(async () => {}),
    rememberLocalGenerationJob: vi.fn(),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
  };
}

describe("Simple Image Studio — built-in edit instructions", () => {
  let container;
  let root;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
    await act(async () => {
      root.render(
        <AppContext.Provider value={context()}>
          <SimpleShell
            accent="teal"
            lockedToSimple={false}
            onAccentChange={() => {}}
            onModeChange={() => {}}
            onSimpleDefaultChange={() => {}}
            simpleDefault
          />
        </AppContext.Provider>,
      );
    });
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  it("shows the pills on Edit only, and fills the prompt with the full instruction", async () => {
    expect(container.querySelector(".su-edit-templates")).toBeNull();

    await click([...container.querySelectorAll(".su-tab")].find((tab) => tab.textContent === "Edit"));

    const pills = [...container.querySelectorAll(".su-edit-templates .su-pill-btn")];
    expect(pills.map((pill) => pill.textContent)).toEqual(EDIT_PROMPT_TEMPLATES.map((template) => template.label));

    await click(pills.find((pill) => pill.textContent === "Restore photo"));
    expect(container.querySelector("#su-image-prompt").value).toBe(
      "colorize and restore this photo, like it was taken in modern day",
    );

    // Back on Text the row is gone — a scene prompt is not an instruction.
    await click([...container.querySelectorAll(".su-tab")].find((tab) => tab.textContent === "Text"));
    expect(container.querySelector(".su-edit-templates")).toBeNull();
  });
});
