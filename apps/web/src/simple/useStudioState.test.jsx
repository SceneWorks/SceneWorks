import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";
import { readSimpleStudioSnapshot, seedSimpleStudiosFromServer } from "../simpleStudioStore.js";

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

function renderShell(root, context, props = {}) {
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
          {...props}
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
    // The Simple studios now persist their settings (localStorage cache + the durable
    // ui-preferences copy), so without this a snapshot flushed by one case restores into
    // the next and the catalog-default assertions read the previous test's picks.
    window.localStorage.clear();
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

  // The relaunch half. A desktop relaunch gives the app a NEW `127.0.0.1:<port>` origin and
  // therefore an EMPTY localStorage, so these simulate the real thing: unmount, clear the
  // cache, seed only from the durable server copy, mount again.
  it("restores the studio from the durable server copy after a relaunch", async () => {
    await renderShell(root, baseContext(), { preferencesHydrated: true });
    await typePrompt(container, "a lighthouse at dusk");
    await openField(container, "Model");
    await chooseOption(container, "Z-Image");

    // Unmounting flushes the pending debounce, which is what a switch to Advanced (or a quit)
    // does. Capture what would have reached the server.
    await unmountRoot(root, container);
    const persisted = JSON.parse(window.localStorage.getItem("sceneworks-simple-studio"));
    expect(persisted["project-1"].image.prompt).toBe("a lighthouse at dusk");
    expect(persisted["project-1"].image.model).toBe("z_image");

    // Relaunch: the cache is gone with the old origin; only the server copy survives.
    window.localStorage.clear();
    seedSimpleStudiosFromServer(persisted);

    ({ container, root } = mountRoot());
    await renderShell(root, baseContext(), { preferencesHydrated: true });
    expect(container.querySelector("#su-image-prompt").value).toBe("a lighthouse at dusk");
    expect(fieldValue(container, "Model")).toBe("Z-Image");
  });

  it("keeps each workspace's settings separate", async () => {
    const one = { id: "project-1", name: "One" };
    const two = { id: "project-2", name: "Two" };
    await renderShell(root, baseContext({ activeProject: one }), { preferencesHydrated: true });
    await typePrompt(container, "prompt for project one");
    await unmountRoot(root, container);

    // A different workspace starts clean rather than inheriting project-1's prompt.
    ({ container, root } = mountRoot());
    await renderShell(root, baseContext({ activeProject: two }), { preferencesHydrated: true });
    expect(container.querySelector("#su-image-prompt").value).toBe("");
    await typePrompt(container, "prompt for project two");
    await unmountRoot(root, container);

    // …and returning to project-1 brings ITS prompt back, not project-2's.
    ({ container, root } = mountRoot());
    await renderShell(root, baseContext({ activeProject: one }), { preferencesHydrated: true });
    expect(container.querySelector("#su-image-prompt").value).toBe("prompt for project one");
  });

  // sc-11962's failure mode, in the durable lane: on a relaunch the cache is empty and the
  // GET is still in flight, so a studio that seeded and persisted during that window would
  // overwrite the stored snapshot with catalog defaults before anyone had read it.
  it("does not write catalog defaults over the stored copy before preferences hydrate", async () => {
    await renderShell(root, baseContext(), { preferencesHydrated: false });
    // The studio is live and has seeded itself from the catalog...
    expect(fieldValue(container, "Model")).toBe("Anima 2B");
    await typePrompt(container, "typed before hydration");
    await unmountRoot(root, container);
    // ...but nothing reached the durable store, so the copy on disk is still the user's.
    expect(window.localStorage.getItem("sceneworks-simple-studio")).toBeNull();
    // Leave afterEach a live root to tear down.
    ({ container, root } = mountRoot());
  });

  it("restores once preferences hydrate, without the studio having to be revisited", async () => {
    // A snapshot already on disk, as if written in a previous session.
    seedSimpleStudiosFromServer({
      "project-1": { image: { model: "z_image", prompt: "from a previous session" } },
    });

    // Mount pre-hydration: the studio shows catalog defaults, because the cache must not be
    // trusted yet (on a real relaunch it would be empty at this point).
    await renderShell(root, baseContext(), { preferencesHydrated: false });
    expect(fieldValue(container, "Model")).toBe("Anima 2B");

    // The GET lands. The shell remounts the studio so the restored values are read by its
    // state initialisers — no revisit, no navigation required.
    await renderShell(root, baseContext(), { preferencesHydrated: true });
    expect(fieldValue(container, "Model")).toBe("Z-Image");
    expect(container.querySelector("#su-image-prompt").value).toBe("from a previous session");
  });

  // sc-15412's open question: the reference id now outlives the library, so a restored one can
  // name an asset the user deleted in between.
  it("drops a restored reference whose asset no longer exists, but only once the catalog loads", async () => {
    const REFERENCE = { id: "asset-9", type: "image", projectId: "project-1", url: "/m/a.png" };
    seedSimpleStudiosFromServer({
      "project-1": { image: { model: "z_image", referenceAssetId: "asset-9" } },
    });

    // Each phase reads the snapshot AFTER an unmount, because the write-through is debounced
    // and the unmount flush is what makes the stored value observable.
    const runWith = async (assets) => {
      ({ container, root } = mountRoot());
      await renderShell(root, baseContext({ assets }), { preferencesHydrated: true });
      await unmountRoot(root, container);
      return readSimpleStudioSnapshot("project-1").image.referenceAssetId;
    };

    // Catalog still loading (`assets` empty): the id must SURVIVE. Pruning here would strip a
    // perfectly good reference on every cold start — that is the sc-11962 trap.
    expect(await runWith([])).toBe("asset-9");

    // Catalog loaded and the asset is still there: kept.
    expect(await runWith([REFERENCE])).toBe("asset-9");

    // Catalog loaded and the asset is gone: dropped, so the Generate gate and the tile agree
    // instead of submitting a reference the job would fail on.
    const other = { id: "asset-1", type: "image", projectId: "project-1", url: "/m/b.png" };
    expect(await runWith([other])).toBeNull();

    // Leave afterEach a live root to tear down.
    ({ container, root } = mountRoot());
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
