// sc-25679 — Iris-3B's CFG-bound negative prompt in the SIMPLE image studio.
//
// Iris's negative prompt IS the CFG unconditional (`image.negativePromptRequiresGuidance`): with
// guidance at 1.0 the engine refuses one. Simple disables the box and withholds the value there,
// and — like Image Studio — says WHY, so text the user already typed is never a silent drop.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, setInput, unmountRoot } from "../testUtils/dom.js";

// `setInput` drives the HTMLInputElement value setter, which jsdom refuses on a textarea.
// Same trick, the matching prototype.
function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    "value",
  ).set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async (path) =>
      path === "/api/v1/host-capabilities"
        ? { memoryGb: 128, memoryKind: "unified", platform: "macos" }
        : {},
    ),
  };
});

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = (() => {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
})();

// The SHIPPED entry, so the CFG-bound negative-prompt declaration is the real one.
const IRIS = {
  ...manifestModels.find((model) => model.id === "iris_3b"),
  installState: "installed",
  usable: true,
};

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Default" },
    assets: [],
    recentImageAssets: [],
    recentVideoAssets: [],
    jobs: [],
    imageModels: [IRIS],
    videoModels: [],
    audioModels: [],
    models: [IRIS],
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
    qwenRewritePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    updateAssetStatus: vi.fn(),
    setSelectedAssetId: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

describe("SimpleImageStudio with Iris 3B (sc-25679)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.clearAllMocks();
  });

  async function openImage(context = baseContext()) {
    await act(async () => {
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
    return context;
  }

  const fieldById = (id) => container.querySelector(`#${id}`);
  const negativeHint = () =>
    fieldById("su-image-negative")?.parentElement?.querySelector(".field-hint") ?? null;

  async function openAdvanced() {
    const fold = container.querySelector("details.su-advanced");
    expect(fold, "the advanced disclosure renders").toBeTruthy();
    await act(async () => {
      fold.open = true;
      fold.dispatchEvent(new Event("toggle", { bubbles: false }));
    });
  }

  it("explains, disables and withholds a typed negative prompt once guidance turns CFG off", async () => {
    const context = await openImage();
    await openAdvanced();

    await act(async () => setTextarea(container.querySelector("#su-image-prompt"), "a lighthouse"));
    await act(async () => setTextarea(fieldById("su-image-negative"), "blurry"));
    // At the model's default guidance (3.0) the box is live and carries no hint.
    expect(fieldById("su-image-negative").disabled).toBe(false);
    expect(negativeHint()).toBeNull();

    await act(async () => setInput(fieldById("su-image-guidance"), "1"));
    const box = fieldById("su-image-negative");
    expect(box.disabled).toBe(true);
    // The typed text stays (raise guidance and it applies again) ...
    expect(box.value).toBe("blurry");
    // ... and the reason is visible beside it.
    expect(negativeHint()?.textContent).toMatch(/only applies with guidance above 1\.0/);

    await click(container.querySelector(".su-generate"));
    expect(context.createImageJob).toHaveBeenCalled();
    const request = context.createImageJob.mock.calls[0][0];
    expect(request.negativePrompt).toBe("");
    expect(request.advanced.guidanceScale).toBe(1);
  });
});
