import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleShell } from "./SimpleShell.jsx";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

// The Simple shell mounts inside App's providers and reads everything through them, so a
// single legacy <AppContext.Provider> is enough to drive it in isolation (see AppContext's
// fallback contract). These tests cover the shell's own contract: navigation, the picker
// sheet, the toast, and that a Generate goes through the real job-creation action with the
// real payload — not a mock.

const IMAGE_MODEL = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  capabilities: ["text_to_image", "edit_image"],
  installState: "installed",
  limits: { resolutions: ["1024x1024", "1344x768"] },
  // Z-Image declares reference-guided generation in the real catalog, with this exact
  // strength config — so the Text tab's Reference tile is live for it.
  ui: { img2img: true, img2imgStrength: { default: 0.5, min: 0, max: 1, step: 0.05 } },
};

const REFERENCE_ASSET = {
  id: "asset-9",
  type: "image",
  projectId: "project-1",
  displayName: "cliff_ref.png",
  url: "/media/cliff_ref.png",
};

function baseContext(overrides = {}) {
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

function buttonWithText(container, text) {
  return [...container.querySelectorAll("button")].find(
    (node) => node.textContent.trim() === text,
  );
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

describe("SimpleShell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // jsdom has no ResizeObserver; the shell's measurement hook degrades to the
    // unmeasured (desktop) band, which is what these tests exercise.
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    vi.restoreAllMocks();
  });

  it("opens on the Image Studio with the persistent sidebar", async () => {
    await renderShell(root, baseContext());
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Image Studio");
    expect(container.querySelector(".su-nav")).toBeTruthy();
    // Desktop band ⇒ no hamburger.
    expect(container.querySelector(".su-hamburger")).toBeNull();
    // The Simple/Advanced switch lives in the sidebar footer.
    expect(container.querySelector(".su-nav-footer .su-switch")).toBeTruthy();
  });

  it("navigates between screens from the sidebar", async () => {
    await renderShell(root, baseContext());
    await click(buttonWithText(container, "Queue"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Queue");
    await click(buttonWithText(container, "Licenses"));
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Licenses");
  });

  it("shows the live queue count as a nav badge and drops it when nothing is pending", async () => {
    await renderShell(
      root,
      baseContext({
        jobs: [
          { id: "a", type: "image_generate", status: "running" },
          { id: "b", type: "image_generate", status: "queued" },
          { id: "c", type: "image_generate", status: "completed" },
        ],
      }),
    );
    expect(container.querySelector(".su-nav-badge").textContent).toBe("2");

    await renderShell(root, baseContext({ jobs: [{ id: "c", type: "image_generate", status: "completed" }] }));
    expect(container.querySelector(".su-nav-badge")).toBeNull();
  });

  it("opens the resolution picker as a sheet and applies the pick", async () => {
    await renderShell(root, baseContext());
    const resolutionButton = [...container.querySelectorAll(".su-select")].find((node) =>
      node.textContent.includes("1:1"),
    );
    expect(resolutionButton).toBeTruthy();
    await click(resolutionButton);

    const sheet = container.querySelector(".su-sheet");
    expect(sheet).toBeTruthy();
    expect(sheet.querySelector(".su-sheet-title").textContent).toBe("Size & aspect");

    const wide = [...sheet.querySelectorAll(".su-option-tile")].find((node) =>
      node.textContent.includes("16:9"),
    );
    await click(wide);
    // Sheet closes and the trigger reflects the new value.
    expect(container.querySelector(".su-sheet")).toBeNull();
    expect(
      [...container.querySelectorAll(".su-select")].some((node) => node.textContent.includes("16:9")),
    ).toBe(true);
  });

  it("opens the prompt guide overlay and closes it again", async () => {
    await renderShell(root, baseContext());
    await click(buttonWithText(container, "Prompt guide"));
    expect(container.textContent).toContain("Lead with the subject");
    await click(container.querySelector(".su-sheet-head .su-icon-btn"));
    expect(container.textContent).not.toContain("Lead with the subject");
  });

  it("submits a real image job through createImageJob and remembers it locally", async () => {
    const context = baseContext();
    await renderShell(root, context);

    const textarea = container.querySelector("#su-image-prompt");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value",
      ).set;
      setter.call(textarea, "a lighthouse at golden hour");
      textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    await click(container.querySelector(".su-generate"));

    expect(context.createImageJob).toHaveBeenCalledTimes(1);
    const payload = context.createImageJob.mock.calls[0][0];
    expect(payload).toMatchObject({
      mode: "text_to_image",
      model: "z_image",
      prompt: "a lighthouse at golden hour",
      count: 1,
      width: 1024,
      height: 1024,
    });
    expect(context.rememberLocalGenerationJob).toHaveBeenCalledWith("image", { id: "job-1" });
  });

  // Regression for a shipped bug: the Reference tile armed, showed SET + a thumbnail and
  // toasted "Reference attached", but the id was routed to `sourceAssetId` — which
  // buildImageJobRequest discards outside edit mode — so the run silently ignored it. This
  // drives the REAL tile through the REAL picker sheet, which is what the payload-only
  // tests could not catch.
  it("attaches a Text-mode reference through the tile and sends it with a strength", async () => {
    const context = baseContext({ assets: [REFERENCE_ASSET], recentImageAssets: [REFERENCE_ASSET] });
    await renderShell(root, context);

    await typePrompt(container, "a lighthouse");
    await click(buttonWithText(container, "ReferenceTap to attach an image"));

    // The picker sheet lists the project's images; choose the one asset.
    const row = [...container.querySelectorAll(".su-option-row")].find((node) =>
      node.textContent.includes("cliff_ref.png"),
    );
    expect(row).toBeTruthy();
    await click(row);

    // Armed: the tile reports SET rather than the empty hint.
    expect(container.querySelector(".su-set-badge")).toBeTruthy();

    await click(container.querySelector(".su-generate"));

    const payload = context.createImageJob.mock.calls[0][0];
    expect(payload.referenceAssetId).toBe("asset-9");
    expect(payload.advanced.strength).toBe(0.5);
    expect(payload.sourceAssetId).toBeNull();
  });

  it("disables the Reference tile, with a reason, on a model that can't use one", async () => {
    const plainModel = { ...IMAGE_MODEL, id: "sdxl", name: "SDXL", ui: undefined };
    const context = baseContext({
      imageModels: [plainModel],
      models: [plainModel],
      assets: [REFERENCE_ASSET],
      recentImageAssets: [REFERENCE_ASSET],
    });
    await renderShell(root, context);

    const tile = [...container.querySelectorAll(".su-tile")].find((node) =>
      node.textContent.includes("Reference"),
    );
    expect(tile.disabled).toBe(true);
    expect(tile.textContent).toContain("SDXL can’t use a reference image");
  });

  it("refuses to submit with an empty prompt", async () => {
    const context = baseContext();
    await renderShell(root, context);
    expect(container.querySelector(".su-generate").disabled).toBe(true);
    await click(container.querySelector(".su-generate"));
    expect(context.createImageJob).not.toHaveBeenCalled();
  });

  it("locks the mode switch and explains why when the viewport forces Simple", async () => {
    await renderShell(root, baseContext(), { lockedToSimple: true });
    const [simple, advanced] = container.querySelectorAll(".su-nav-footer .su-switch button");
    expect(simple.disabled).toBe(true);
    expect(advanced.disabled).toBe(true);
    expect(container.textContent).toContain("Phones always use the Simple UI.");
  });

  // "Manage" points the WORKSPACE at its Models screen — which is invisible while the
  // Simple shell is rendering. So it has to flip the shell too, or the button is inert.
  it("hands 'Manage' off to the full Models screen AND switches shells", async () => {
    const context = baseContext();
    const onModeChange = vi.fn();
    await renderShell(root, context, { onModeChange });
    await click(buttonWithText(container, "Model Manager"));

    const manage = buttonWithText(container, "Manage");
    expect(manage).toBeTruthy();
    await click(manage);

    expect(context.setActiveView).toHaveBeenCalledWith("Models");
    expect(onModeChange).toHaveBeenCalledWith("advanced");
  });

  it("refuses the hand-off with a reason when the viewport locks Simple on", async () => {
    const context = baseContext();
    const onModeChange = vi.fn();
    await renderShell(root, context, { onModeChange, lockedToSimple: true });
    await click(buttonWithText(container, "Model Manager"));
    await click(buttonWithText(container, "Manage"));

    expect(context.setActiveView).not.toHaveBeenCalled();
    expect(onModeChange).not.toHaveBeenCalled();
    expect(container.querySelector(".su-toast").textContent).toContain("Advanced workspace");
  });

  it("enqueues a real download for a model that isn't installed", async () => {
    const context = baseContext({
      models: [IMAGE_MODEL, { ...IMAGE_MODEL, id: "flux_dev", name: "FLUX.1 [dev]", installState: "missing" }],
    });
    await renderShell(root, context);
    await click(buttonWithText(container, "Model Manager"));
    await click(buttonWithText(container, "Download"));

    expect(context.createModelDownloadJob).toHaveBeenCalledTimes(1);
    expect(context.createModelDownloadJob.mock.calls[0][0].id).toBe("flux_dev");
  });

  it("reports the mode switch to the host app", async () => {
    const onModeChange = vi.fn();
    await renderShell(root, baseContext(), { onModeChange });
    const advanced = [...container.querySelectorAll(".su-nav-footer .su-switch button")].find(
      (node) => node.textContent === "Advanced",
    );
    await click(advanced);
    expect(onModeChange).toHaveBeenCalledWith("advanced");
  });
});
