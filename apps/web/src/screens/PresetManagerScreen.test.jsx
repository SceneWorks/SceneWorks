import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresetManagerScreen } from "./PresetManagerScreen.jsx";
import { withAppContext, field, changeField } from "../main.testSupport.jsx";

const imageModel = {
  id: "z_image_turbo",
  name: "Z-Image",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image", "edit_image", "character_image"],
  downloadSizeLabel: "5.1 GB",
};

const videoModel = {
  id: "ltx_2_3",
  name: "LTX",
  type: "video",
  family: "ltx-video",
  capabilities: ["text_to_video", "image_to_video"],
};

const presets = [
  {
    id: "cinematic",
    name: "Cinematic Portrait",
    scope: "global",
    workflow: "text_to_image",
    model: "z_image_turbo",
    updatedAt: "2026-07-09T00:00:00Z",
    lastUsedAt: "2026-07-06T00:00:00Z",
    prompt: { prefix: "cinematic portrait of", suffix: ", 85mm" },
    defaults: { resolution: "1024x1024", count: 4, quality: "balanced" },
    loras: [{ id: "global_detail", weight: 0.7 }],
  },
  {
    id: "anime_key",
    name: "Anime Key Visual",
    scope: "project",
    workflow: "edit_image",
    model: "z_image_turbo",
    updatedAt: "2026-07-05T00:00:00Z",
    lastUsedAt: "2026-07-08T00:00:00Z",
    ui: { description: "cel shading" },
  },
  {
    id: "clip",
    name: "Bridge Clip",
    scope: "global",
    workflow: "text_to_video",
    model: "ltx_2_3",
    updatedAt: "2026-07-03T00:00:00Z",
    defaults: { duration: 6, fps: 24 },
  },
];

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-1", name: "Noir" },
    createPreset: vi.fn(async (payload) => payload),
    updatePreset: vi.fn(async (id, payload) => ({ ...payload, id })),
    duplicatePreset: vi.fn(async (id) => ({ id: `${id}_copy` })),
    deletePreset: vi.fn(async (id) => ({ id, archived: true })),
    imageModels: [imageModel],
    videoModels: [videoModel],
    loras: [],
    presets,
    sendPresetToStudio: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

describe("PresetManagerScreen", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
  });

  async function render(context = baseContext()) {
    root = createRoot(container);
    await act(async () => {
      root.render(withAppContext(context, <PresetManagerScreen />));
    });
    return context;
  }

  const cardNames = () => [...container.querySelectorAll(".preset-card strong")].map((node) => node.textContent);
  const clickButton = async (label) => {
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === label).click();
    });
  };

  it("filters the grid by search, scope, and type", async () => {
    await render();
    expect(cardNames()).toHaveLength(3);
    expect(container.textContent).toContain("3 presets · 2 global");

    await changeField(container.querySelector("input[type=search]"), "anime");
    expect(cardNames()).toEqual(["Anime Key Visual"]);

    await changeField(container.querySelector("input[type=search]"), "");
    await clickButton("Global");
    expect(cardNames()).toEqual(expect.arrayContaining(["Cinematic Portrait", "Bridge Clip"]));
    expect(cardNames()).not.toContain("Anime Key Visual");
    expect(container.textContent).toContain("2 presets · 2 global");

    await clickButton("All");
    await changeField(container.querySelector("select[aria-label='Type']"), "edit_image");
    expect(cardNames()).toEqual(["Anime Key Visual"]);
  });

  it("sorts by recently updated, then by name or scope on request", async () => {
    await render();
    const sortSelect = () => container.querySelector("select[aria-label='Sort presets']");

    // Default sort is `updatedAt` descending: Cinematic 07-09, Anime 07-05, Bridge 07-03.
    expect(cardNames()).toEqual(["Cinematic Portrait", "Anime Key Visual", "Bridge Clip"]);

    await changeField(sortSelect(), "name");
    expect(cardNames()).toEqual(["Anime Key Visual", "Bridge Clip", "Cinematic Portrait"]);

    // global (Bridge, Cinematic) before project (Anime), name-tiebroken within a scope.
    await changeField(sortSelect(), "scope");
    expect(cardNames()).toEqual(["Bridge Clip", "Cinematic Portrait", "Anime Key Visual"]);
  });

  it("sorts by recently used, sinking never-used presets to the bottom (sc-10520)", async () => {
    await render();
    const sortSelect = () => container.querySelector("select[aria-label='Sort presets']");

    // Anime used 07-08, Cinematic used 07-06, Bridge never used (no lastUsedAt) → last,
    // even though Bridge's ordering differs under `updated`/`name`/`scope`.
    await changeField(sortSelect(), "used");
    expect(cardNames()).toEqual(["Anime Key Visual", "Cinematic Portrait", "Bridge Clip"]);
  });

  it("hands a preset to the studio that can run it", async () => {
    const context = await render();
    const clipCard = [...container.querySelectorAll(".preset-card")].find((card) => card.textContent.includes("Bridge Clip"));
    await act(async () => {
      clipCard.querySelector(".preset-card-use").click();
    });
    expect(context.sendPresetToStudio).toHaveBeenCalledWith(expect.objectContaining({ id: "clip" }));
  });

  it("blocks Use in Studio for a preset pinned to an uninstalled model", async () => {
    // `imageModels` excludes installState: "missing", but the full catalog still names it.
    const uninstalled = { ...imageModel, id: "flux2_dev", name: "FLUX.2 [dev]", installState: "missing" };
    await render(
      baseContext({
        models: [imageModel, videoModel, uninstalled],
        presets: [{ ...presets[0], id: "pinned", name: "Pinned", model: "flux2_dev" }],
      }),
    );

    const card = container.querySelector(".preset-card");
    // The name resolves from the full catalog even though the model can't be selected.
    expect(card.textContent).toContain("FLUX.2 [dev]");
    const use = card.querySelector(".preset-card-use");
    expect(use.disabled).toBe(true);
    expect(use.getAttribute("title")).toBe("Install FLUX.2 [dev] to use this preset");
  });

  it("derives the Workflow options from the selected model", async () => {
    await render();
    await clickButton("New preset");

    // The image model advertises text_to_image + edit_image + character_image.
    const segmentLabels = () => [...container.querySelectorAll(".preset-workflow button")].map((b) => b.textContent);
    expect(segmentLabels()).toEqual(["Text", "Edit", "Character"]);
    expect(container.textContent).toContain("txt2img · img2img · character · 5.1 GB");

    // Switching to a video model re-derives the segment and drops the image modes.
    await changeField(field(container, "Model"), "ltx_2_3");
    expect(segmentLabels()).toEqual(["Text → Video", "Image → Video"]);
    // Defaults swap to the video knobs.
    expect(field(container, "Duration")).toBeDefined();
    expect(field(container, "Variations")).toBeUndefined();
  });

  it("persists Character as workflow text_to_image plus defaults.mode character_image", async () => {
    const context = await render();
    await clickButton("New preset");
    await changeField(field(container, "Name"), "Aurora");
    await clickButton("Character");
    await clickButton("Create preset");

    expect(context.createPreset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "aurora",
        // character_image is NOT a RecipePresetWorkflow — it rides as the sub-mode.
        workflow: "text_to_image",
        defaults: expect.objectContaining({ mode: "character_image" }),
      }),
    );
  });

  // sc-10548: PATCH replaces `defaults` wholesale, so anything the editor doesn't render
  // must be carried through explicitly or a rename destroys it.
  it("preserves defaults the editor doesn't render, and still clears ones it does", async () => {
    const studioAuthored = {
      ...presets[0],
      id: "studio_made",
      name: "Studio Made",
      loras: [],
      defaults: {
        mode: "text_to_image",
        resolution: "1024x1024",
        steps: 30,
        // None of these have a control in the Preset editor.
        upscaleFactor: 2,
        ipAdapterScale: 0.6,
        guidanceMethod: "cfg_pp",
        prompt: "a literal prompt",
      },
    };
    const context = await render(baseContext({ presets: [studioAuthored] }));

    await act(async () => {
      container.querySelector(".preset-card .secondary-action").click();
    });
    await changeField(field(container, "Name"), "Studio Made v2");
    // Clear a field the editor DOES own — that must actually remove the key.
    await changeField(field(container, "Aspect"), "");
    await clickButton("Save preset");

    const [, payload] = context.updatePreset.mock.calls[0];
    expect(payload.defaults).toEqual({
      mode: "text_to_image",
      steps: 30,
      upscaleFactor: 2,
      ipAdapterScale: 0.6,
      guidanceMethod: "cfg_pp",
      prompt: "a literal prompt",
    });
  });

  it("shows Draft for a new preset and Unsaved changes once an existing one is edited", async () => {
    await render();
    await clickButton("New preset");
    expect(container.querySelector(".preset-status-pill").textContent).toContain("Draft");

    await clickButton("All presets");
    await act(async () => {
      [...container.querySelectorAll(".preset-card")]
        .find((card) => card.textContent.includes("Cinematic Portrait"))
        .querySelector(".secondary-action")
        .click();
    });
    expect(container.querySelector(".preset-status-pill").textContent).toContain("Saved");

    await changeField(field(container, "Name"), "Cinematic Portrait v2");
    expect(container.querySelector(".preset-status-pill").textContent).toContain("Unsaved changes");
  });
});
