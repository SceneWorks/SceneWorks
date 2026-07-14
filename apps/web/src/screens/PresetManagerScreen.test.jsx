import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PresetManagerScreen } from "./PresetManagerScreen.jsx";
import { withAppContext, field, changeField } from "../main.testSupport.jsx";
import { appConfirm } from "../appConfirm.jsx";

// The unsaved-changes guard routes through the desktop-safe appConfirm (sc-11969), not
// window.confirm (which no-ops in the Tauri WebView). Mock it so a test controls the
// user's choice and can assert the guard fired without mounting a real <ConfirmHost/>.
vi.mock("../appConfirm.jsx", () => {
  const appConfirm = vi.fn(() => Promise.resolve(true));
  return { appConfirm, useConfirm: () => appConfirm, ConfirmHost: () => null };
});

const imageModel = {
  id: "z_image_turbo",
  name: "Z-Image",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image", "edit_image", "character_image"],
  downloadSizeLabel: "5.1 GB",
};

// A second image model whose manifest declares a narrow resolution menu — the source
// the editor must derive the Aspect list from (sc-10589).
const fluxModel = {
  id: "flux_dev",
  name: "FLUX [dev]",
  type: "image",
  family: "flux",
  capabilities: ["text_to_image"],
  limits: { resolutions: ["768x768", "1024x1024", "1280x720", "720x1280"] },
};

const videoModel = {
  id: "ltx_2_3",
  name: "LTX",
  type: "video",
  family: "ltx-video",
  capabilities: ["text_to_video", "image_to_video"],
  limits: { resolutions: ["768x512", "1280x720"], durations: [4, 6, 8], fps: [24, 25] },
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
    appConfirm.mockClear();
    appConfirm.mockResolvedValue(true);
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

  // sc-10589: the Aspect/Resolution/Duration/Frames menus come from the selected model's
  // effective limits, so the editor can't offer a default the studio would clamp away.
  it("derives the Aspect menu from the selected model's limits.resolutions", async () => {
    await render(baseContext({ imageModels: [imageModel, fluxModel] }));
    await clickButton("New preset");

    // z_image_turbo declares no limits → the static fallback list.
    const aspectValues = () => [...field(container, "Aspect").options].map((option) => option.value);
    expect(aspectValues()).toEqual(["", "1024x1024", "1536x1024", "1024x1536", "2048x1152"]);

    // flux_dev declares a narrower menu — the editor follows it.
    await changeField(field(container, "Model"), "flux_dev");
    expect(aspectValues()).toEqual(["", "768x768", "1024x1024", "1280x720", "720x1280"]);
  });

  it("derives video Resolution/Duration/Frames menus and clears a now-invalid value on switch", async () => {
    await render();
    await clickButton("New preset");
    await changeField(field(container, "Model"), "ltx_2_3");

    // The video model's manifest lists these exactly.
    expect([...field(container, "Resolution").options].map((o) => o.value)).toEqual(["", "768x512", "1280x720"]);
    expect([...field(container, "Duration").options].map((o) => o.value)).toEqual(["", "4", "6", "8"]);
    expect([...field(container, "Frames").options].map((o) => o.value)).toEqual(["", "24", "25"]);
  });

  it("clears a resolution the newly selected same-type model no longer lists", async () => {
    await render(baseContext({ imageModels: [imageModel, fluxModel] }));
    await clickButton("New preset");

    // 1536x1024 is in z_image_turbo's fallback menu but not in flux_dev's.
    await changeField(field(container, "Aspect"), "1536x1024");
    expect(field(container, "Aspect").value).toBe("1536x1024");

    await changeField(field(container, "Model"), "flux_dev");
    expect(field(container, "Aspect").value).toBe("");
  });

  it("flags an out-of-menu stored resolution and blocks the save until it's fixed", async () => {
    const stale = {
      id: "stale",
      name: "Stale",
      scope: "global",
      workflow: "text_to_image",
      model: "flux_dev",
      defaults: { resolution: "2048x1152" },
    };
    await render(baseContext({ imageModels: [imageModel, fluxModel], presets: [stale] }));
    await act(async () => {
      container.querySelector(".preset-card .secondary-action").click();
    });

    const aspect = field(container, "Aspect");
    // The stored value is shown and selected, flagged rather than blanked.
    expect(aspect.value).toBe("2048x1152");
    expect(aspect.textContent).toContain("not in this model");
    expect(container.textContent).toContain("isn't one this model supports");
    expect([...container.querySelectorAll("button[type='submit']")].every((b) => b.disabled)).toBe(true);

    // Picking a supported option unblocks the save.
    await changeField(aspect, "1024x1024");
    expect([...container.querySelectorAll("button[type='submit']")].some((b) => !b.disabled)).toBe(true);
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

  // epic 11949: a general (model-agnostic) preset carries only prompt fragments + aspect +
  // variations + negative, with no model/workflow/LoRA controls, and saves with kind:general.
  it("creates a general preset with prompt fragments and aspect, and no model", async () => {
    const context = await render();
    await clickButton("New preset");
    await clickButton("General preset");

    // The model-specific controls are gone.
    expect(field(container, "Model")).toBeUndefined();
    expect(container.querySelector(".preset-workflow")).toBeNull();
    // Prompt inputs are relabelled to signal the fragment/stacking model.
    expect(field(container, "Prepend to prompt")).toBeDefined();
    expect(field(container, "Append to prompt")).toBeDefined();
    // Aspect is a ratio menu, not a WxH resolution list.
    expect([...field(container, "Aspect").options].map((o) => o.value)).toEqual([
      "",
      "1:1",
      "3:2",
      "2:3",
      "16:9",
      "9:16",
      "4:3",
      "3:4",
    ]);

    await changeField(field(container, "Name"), "Kodak Portra");
    await changeField(field(container, "Append to prompt"), "Kodak Portra 400");
    await changeField(field(container, "Aspect"), "16:9");
    await changeField(field(container, "Variations"), "2");
    await changeField(field(container, "Negative prompt"), "blurry");
    await clickButton("Create preset");

    expect(context.createPreset).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "kodak_portra",
        kind: "general",
        prompt: { suffix: "Kodak Portra 400" },
        defaults: expect.objectContaining({ aspect: "16:9", count: 2, negativePrompt: "blurry" }),
      }),
    );
    const [payload] = context.createPreset.mock.calls[0];
    expect(payload.model).toBeUndefined();
    expect(payload.workflow).toBeUndefined();
    expect(payload.loras).toBeUndefined();
  });

  it("lists a general preset with a badge, always runnable, and filters by General type", async () => {
    const generalPreset = {
      id: "film_stock",
      name: "Film Stock",
      kind: "general",
      scope: "global",
      updatedAt: "2026-07-10T00:00:00Z",
      prompt: { suffix: "Kodak Portra 400" },
      defaults: { aspect: "3:2", count: 2 },
    };
    await render(baseContext({ presets: [...presets, generalPreset] }));

    const card = [...container.querySelectorAll(".preset-card")].find((c) => c.textContent.includes("Film Stock"));
    expect(card.querySelector(".preset-kind-chip")?.textContent).toBe("General");
    expect(card.textContent).toContain("General · any model");
    // Runnable even though it pins no model (no install gate).
    expect(card.querySelector(".preset-card-use").disabled).toBe(false);

    // The General type filter shows only general presets…
    await changeField(container.querySelector("select[aria-label='Type']"), "general");
    expect(cardNames()).toEqual(["Film Stock"]);
    // …and a workflow filter never surfaces a general preset.
    await changeField(container.querySelector("select[aria-label='Type']"), "text_to_image");
    expect(cardNames()).not.toContain("Film Stock");
  });

  // sc-11969 (epic 11958, S10): the editor's DESTRUCTIVE in-screen transitions guard an
  // in-progress edit behind the desktop-safe appConfirm; plain navigation (keep-alive)
  // never prompts, which is covered at the App level in App.presetManagerKeepAlive.test.jsx.
  describe("unsaved-changes guard on destructive transitions", () => {
    const pill = () => container.querySelector(".preset-status-pill")?.textContent ?? "";
    const editorForm = () => container.querySelector(".preset-editor-form");

    async function editCard(name) {
      await act(async () => {
        [...container.querySelectorAll(".preset-card")]
          .find((card) => card.textContent.includes(name))
          .querySelector(".secondary-action")
          .click();
      });
    }

    // Click a button by exact visible text, then drain appConfirm()'s promise and its
    // .then(proceed) plus the resulting re-render — all inside act — so the guarded
    // transition has fully settled before the assertions run.
    async function clickAndFlush(label) {
      await act(async () => {
        [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === label).click();
        for (let index = 0; index < 4; index += 1) {
          await Promise.resolve();
        }
      });
    }

    it("prompts via the desktop-safe appConfirm before leaving a dirty edit, and cancel keeps it", async () => {
      appConfirm.mockResolvedValue(false);
      await render();
      await editCard("Cinematic Portrait");
      await changeField(field(container, "Name"), "Cinematic Portrait v2");
      expect(pill()).toContain("Unsaved changes");

      await clickAndFlush("All presets");
      // A danger-toned confirm was shown (appConfirm, not window.confirm).
      expect(appConfirm).toHaveBeenCalledWith(expect.objectContaining({ tone: "danger" }));
      // Cancelled → still in the editor with the edit intact.
      expect(editorForm()).not.toBeNull();
      expect(field(container, "Name").value).toBe("Cinematic Portrait v2");
      expect(pill()).toContain("Unsaved changes");
    });

    it("discards the edit and returns to the list when the leave is confirmed", async () => {
      appConfirm.mockResolvedValue(true);
      await render();
      await editCard("Cinematic Portrait");
      await changeField(field(container, "Name"), "Cinematic Portrait v2");

      await clickAndFlush("All presets");
      expect(appConfirm).toHaveBeenCalledTimes(1);
      // Back on the list; the edit was dropped (never saved), so the card keeps its name.
      expect(editorForm()).toBeNull();
      expect(cardNames()).toContain("Cinematic Portrait");
    });

    it("leaves a clean edit immediately, without prompting", async () => {
      await render();
      await editCard("Cinematic Portrait");
      // No field change → the form is clean.
      await clickAndFlush("All presets");
      expect(appConfirm).not.toHaveBeenCalled();
      expect(editorForm()).toBeNull();
    });

    it("guards the Cancel button in the editor footer the same way", async () => {
      appConfirm.mockResolvedValue(false);
      await render();
      await editCard("Cinematic Portrait");
      await changeField(field(container, "Name"), "Cinematic Portrait v2");

      // Two Cancel buttons (head + footer) both route through the guarded backToList.
      const cancels = [...container.querySelectorAll("button")].filter((b) => b.textContent.trim() === "Cancel");
      expect(cancels.length).toBeGreaterThan(1);
      await act(async () => {
        cancels[cancels.length - 1].click();
        for (let index = 0; index < 4; index += 1) {
          await Promise.resolve();
        }
      });
      expect(appConfirm).toHaveBeenCalledWith(expect.objectContaining({ tone: "danger" }));
      expect(editorForm()).not.toBeNull();
    });

    it("reverts to the saved baseline via the explicit Discard action after confirming", async () => {
      appConfirm.mockResolvedValue(true);
      await render();
      await editCard("Cinematic Portrait");
      await changeField(field(container, "Name"), "Edited Name");
      expect(pill()).toContain("Unsaved changes");
      // The Discard affordance only appears while dirty.
      expect([...container.querySelectorAll("button")].some((b) => b.textContent.trim() === "Discard")).toBe(true);

      await clickAndFlush("Discard");
      expect(appConfirm).toHaveBeenCalled();
      // Reverted: name restored to baseline, pill back to Saved, still editing.
      expect(field(container, "Name").value).toBe("Cinematic Portrait");
      expect(pill()).toContain("Saved");
      expect(editorForm()).not.toBeNull();
    });

    it("keeps the edit when the Discard action is cancelled", async () => {
      appConfirm.mockResolvedValue(false);
      await render();
      await editCard("Cinematic Portrait");
      await changeField(field(container, "Name"), "Edited Name");

      await clickAndFlush("Discard");
      expect(appConfirm).toHaveBeenCalled();
      expect(field(container, "Name").value).toBe("Edited Name");
      expect(pill()).toContain("Unsaved changes");
    });
  });
});
