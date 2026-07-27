import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

// A studio UNMOUNTS when you navigate away (the shell renders exactly one screen), so
// without the shell-held store every knob would be re-seeded from the catalog on the way
// back — the model snapping to whichever entry happens to sort first. These tests drive the
// real shell through a real round trip and assert the studio comes back as it was left.

// Two models on purpose: with only one, "the selection was kept" and "the selection was
// re-seeded to models[0]" are the same assertion and the test would pass either way.
const Z_IMAGE = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  capabilities: ["text_to_image"],
  installState: "installed",
  limits: { resolutions: ["1024x1024", "1344x768"] },
};
const ANIMA = {
  id: "anima_2b",
  name: "Anima 2B",
  type: "image",
  capabilities: ["text_to_image"],
  installState: "installed",
  limits: { resolutions: ["1024x1024", "768x768"] },
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    jobs: [],
    // Anima first, exactly as the catalog orders it — so a reset is visible as a snap-back.
    imageModels: [ANIMA, Z_IMAGE],
    videoModels: [],
    audioModels: [],
    models: [ANIMA, Z_IMAGE],
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
    jobAction: vi.fn(async () => {}),
    rememberLocalGenerationJob: vi.fn(),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

function renderShell(root, context) {
  return act(async () => {
    root.render(
      <AppContext.Provider value={context}>
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
}

function navButton(container, label) {
  return [...container.querySelectorAll(".su-nav-item")].find((node) =>
    node.textContent.startsWith(label),
  );
}

function fieldValue(container, label) {
  const field = [...container.querySelectorAll(".su-field")].find(
    (node) => node.querySelector("label")?.textContent.trim() === label,
  );
  return field?.querySelector(".su-select span")?.textContent.trim() ?? null;
}

async function openField(container, label) {
  const field = [...container.querySelectorAll(".su-field")].find(
    (node) => node.querySelector("label")?.textContent.trim() === label,
  );
  await click(field.querySelector(".su-select"));
}

async function chooseOption(container, text) {
  const row = [...container.querySelectorAll(".su-option-row, .su-option-tile")].find((node) =>
    node.textContent.includes(text),
  );
  expect(row).toBeTruthy();
  await click(row);
}

async function typePrompt(container, value) {
  const textarea = container.querySelector("#su-image-prompt");
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    ).set;
    setter.call(textarea, value);
    textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
  });
}

describe("Simple studio state across navigation", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.restoreAllMocks();
  });

  it("keeps the picked model, prompt and resolution when you leave the studio and come back", async () => {
    await renderShell(root, baseContext());

    // Seeded from the catalog on a cold open — the first eligible model.
    expect(fieldValue(container, "Model")).toBe("Anima 2B");

    await typePrompt(container, "a lighthouse at dusk");
    await openField(container, "Model");
    await chooseOption(container, "Z-Image");
    expect(fieldValue(container, "Model")).toBe("Z-Image");

    await openField(container, "Resolution");
    await chooseOption(container, "1344");
    const size = fieldValue(container, "Resolution");

    await click(navButton(container, "Queue"));
    expect(container.querySelector("#su-image-prompt")).toBeNull();

    await click(navButton(container, "Image"));
    expect(fieldValue(container, "Model")).toBe("Z-Image");
    expect(container.querySelector("#su-image-prompt").value).toBe("a lighthouse at dusk");
    expect(fieldValue(container, "Resolution")).toBe(size);
  });

  it("keeps each studio's state separate", async () => {
    await renderShell(root, baseContext());
    await typePrompt(container, "image prompt");

    await click(navButton(container, "Audio"));
    const audioPrompt = container.querySelector("#su-audio-prompt");
    expect(audioPrompt.value).toBe("");

    await click(navButton(container, "Image"));
    expect(container.querySelector("#su-image-prompt").value).toBe("image prompt");
  });

  it("still re-seeds a studio the user never touched", async () => {
    await renderShell(root, baseContext());
    await click(navButton(container, "Queue"));
    await click(navButton(container, "Image"));
    // Nothing was chosen, so the catalog default still applies on the way back — the store
    // must not have persisted a half-seeded empty value.
    expect(fieldValue(container, "Model")).toBe("Anima 2B");
    expect(fieldValue(container, "Resolution")).toBeTruthy();
  });
});
