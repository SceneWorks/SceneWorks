import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setFileInput, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

// Pose loaders fetch best-effort on mount; stub the API so render never touches
// the network. The studio's own mutations go through context fns, not apiFetch.
vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    apiFetch: vi.fn(async () => ({})),
  };
});

import { AppContext } from "../context/AppContext.js";
import {
  buildStructuredPromptRecipe,
  parseMagicPromptCaption,
  serializeCaption,
} from "../ideogramCaption.js";
import { PROMPT_REFINE_MODEL_ID, VISION_CAPTION_MODEL_ID } from "../constants.js";
import { ImageStudio } from "./ImageStudio.jsx";

const Z_IMAGE = {
  id: "z_image_turbo",
  name: "Z Image Turbo",
  type: "image",
  family: "z-image",
  capabilities: ["text_to_image"],
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024", "1536x1024"] },
  loraCompatibility: {},
  ui: {},
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [Z_IMAGE],
    importAsset: vi.fn(),
    latestImageAssets: [],
    recentImageAssets: [],
    studioLaunch: null,
    imageLocalJobs: [],
    loras: [],
    jobAction: vi.fn(),
    rememberLocalGenerationJob: vi.fn(),
    setActiveView: vi.fn(),
    setPreviewAsset: vi.fn(),
    presets: [],
    requestedGpu: "",
    selectedAsset: null,
    setRequestedGpu: vi.fn(),
    updateAssetStatus: vi.fn(),
    ...overrides,
  };
}

const saveButton = (container) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent.includes("Save as Preset"));
const nameInput = (container) => container.querySelector('input[aria-label="Preset name"]');
const field = (container, labelText) => {
  const label = [...container.querySelectorAll("label")].find((node) =>
    node.textContent.trim().startsWith(labelText),
  );
  return label?.querySelector("input, select");
};

describe("ImageStudio Save as Preset", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {}); // flush mount effects (pose loaders, etc.)
  }

  // The Advanced disclosure belongs INSIDE the work-panel, not floating beneath it:
  // the Purpose zone is one elevated card. It shipped detached in sc-10476 and was
  // pulled back in by sc-10490 — assert the nesting so it cannot drift again.
  it("nests the Advanced disclosure inside the work-panel", async () => {
    await render(baseContext());

    expect(document.body.querySelector(".work-panel .advanced-section")).toBeTruthy();
    expect(document.body.querySelector(".studio-shell > .advanced-section")).toBeNull();
  });

  // `.advanced-panel` is a 3-column grid, so a bare hint span is a grid ITEM: it eats a
  // cell and shifts every control after it into the wrong column (sc-10493). Hints belong
  // inside the <label> they describe.
  it("puts no bare hint spans in the advanced grid", async () => {
    await render(baseContext());
    await click(document.body.querySelector(".advanced-section-toggle"));

    expect(document.body.querySelector(".advanced-panel > .field-hint")).toBeNull();
    expect(document.body.textContent).not.toContain("Custom size overrides the Aspect dropdown");
  });

  it("snapshots the current config into a preset payload without the seed", async () => {
    const context = baseContext();
    await render(context);

    // Save-as-preset now lives inside the Advanced panel (UI-refinement 2b).
    await click(document.body.querySelector(".advanced-section-toggle"));
    const input = nameInput(container);
    expect(input).toBeTruthy();
    await act(async () => setInput(input, "Atrium Look"));
    await click(saveButton(container));

    expect(context.createPreset).toHaveBeenCalledTimes(1);
    const payload = context.createPreset.mock.calls[0][0];
    expect(payload).toMatchObject({
      id: "atrium_look",
      name: "Atrium Look",
      scope: "project",
      workflow: "text_to_image",
      model: "z_image_turbo",
    });
    // The literal prompt rides in defaults; the seed never does.
    expect(payload.defaults.prompt).toBe("A cinematic frame of a neon street at midnight");
    expect(payload.defaults).not.toHaveProperty("seed");
    expect(container.textContent).toContain('Saved "Atrium Look" to this project.');
  });

  it("blocks a duplicate name client-side before calling the API", async () => {
    const context = baseContext({
      presets: [
        {
          id: "atrium_look",
          name: "Atrium Look",
          scope: "project",
          workflow: "text_to_image",
          model: "z_image_turbo",
          modes: ["text_to_image", "character_image", "style_variations"],
        },
      ],
    });
    await render(context);

    await click(document.body.querySelector(".advanced-section-toggle"));
    await act(async () => setInput(nameInput(container), "Atrium Look"));
    await click(saveButton(container));

    expect(context.createPreset).not.toHaveBeenCalled();
    expect(container.textContent).toContain("already exists");
  });
});

describe("ImageStudio advanced model defaults", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  it("resets advanced overrides to the newly selected model defaults", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [
          {
            ...Z_IMAGE,
            defaults: {
              resolution: "1024x1024",
              sampler: "euler",
              scheduler: "shift",
              schedulerShift: 1.5,
              steps: 12,
              guidanceScale: 2.5,
            },
            limits: {
              resolutions: ["1024x1024", "1536x1024"],
              samplers: ["default", "euler", "unipc"],
              schedulers: ["default", "shift", "karras"],
            },
          },
          {
            id: "qwen_image",
            name: "Qwen Image",
            type: "image",
            family: "qwen-image",
            capabilities: ["text_to_image"],
            defaults: {
              resolution: "1536x1024",
              sampler: "unipc",
              scheduler: "shift",
              schedulerShift: 4.2,
              steps: 28,
              guidanceScale: 6.5,
            },
            limits: {
              resolutions: ["1024x1024", "1536x1024"],
              samplers: ["default", "euler", "unipc"],
              schedulers: ["default", "shift", "karras"],
            },
            loraCompatibility: {},
            ui: {},
          },
        ],
      }),
    );

    await click(document.body.querySelector(".advanced-section-toggle"));
    await act(async () => setSelect(field(container, "Sampler"), "euler"));
    await act(async () => setSelect(field(container, "Scheduler"), "shift"));
    await act(async () => setInput(field(container, "Schedule shift"), "7.7"));
    await act(async () => setInput(field(container, "Steps"), "44"));
    await act(async () => setInput(field(container, "Guidance"), "11"));

    await act(async () => setSelect(field(container, "Model"), "qwen_image"));
    await act(async () => {});

    expect(field(container, "Sampler").value).toBe("unipc");
    expect(field(container, "Scheduler").value).toBe("shift");
    expect(field(container, "Schedule shift").value).toBe("4.2");
    expect(field(container, "Steps").value).toBe("");
    expect(field(container, "Steps").placeholder).toBe("28");
    expect(field(container, "Guidance").value).toBe("");
    expect(field(container, "Guidance").placeholder).toBe("6.5");

    await click([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.model).toBe("qwen_image");
    expect(payload.advanced).toMatchObject({
      resolution: "1536x1024",
      sampler: "unipc",
      scheduler: "shift",
      schedulerShift: 4.2,
    });
    expect(payload.advanced).not.toHaveProperty("steps");
    expect(payload.advanced).not.toHaveProperty("guidanceScale");
  });
});

describe("ImageStudio guidance method picker (epic 7434, sc-7449)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // SDXL-shaped fixture: advertises CFG++ on the base menu so the picker renders
  // regardless of the test backend (the per-backend resolution is unit-tested in
  // samplerOptions.test.js).
  const SDXL = {
    ...Z_IMAGE,
    id: "sdxl",
    name: "Stable Diffusion XL",
    family: "sdxl",
    defaults: { resolution: "1024x1024", guidanceScale: 7 },
    limits: { resolutions: ["1024x1024"], guidanceMethods: ["cfg", "cfg_pp"] },
  };

  const openAdvanced = async () =>
    click(document.body.querySelector(".advanced-section-toggle"));
  const generate = async () =>
    click([...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate"));

  it("hides the picker for a model that advertises no alternate methods", async () => {
    await render(baseContext({ imageModels: [Z_IMAGE] }));
    await openAdvanced();
    expect(field(container, "Guidance method")).toBeUndefined();
  });

  it("shows CFG++, defaults to the cfg no-op, and omits it from the payload", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(baseContext({ createImageJob, imageModels: [SDXL] }));
    await openAdvanced();
    const picker = field(container, "Guidance method");
    expect(picker).toBeTruthy();
    expect([...picker.options].map((o) => o.value)).toEqual(["cfg", "cfg_pp"]);
    expect(picker.value).toBe("cfg");
    // The CFG++ low-cfg hint only appears once cfg_pp is selected.
    expect(container.textContent).not.toContain("reparameterizes guidance");

    await generate();
    // cfg is the N1 no-op — never sent, so existing recipes stay byte-identical.
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("guidanceMethod");
  });

  it("emits the selected non-default method and surfaces the low-cfg hint", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(baseContext({ createImageJob, imageModels: [SDXL] }));
    await openAdvanced();
    await act(async () => setSelect(field(container, "Guidance method"), "cfg_pp"));
    expect(container.textContent).toContain("reparameterizes guidance");

    await generate();
    expect(createImageJob.mock.calls[0][0].advanced.guidanceMethod).toBe("cfg_pp");
  });

  it("round-trips the guidance method from a recipe (restore → re-emit)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [SDXL],
        studioLaunch: {
          id: "launch-1",
          view: "Image",
          assetId: "asset-1",
          recipe: {
            model: "sdxl",
            mode: "text_to_image",
            prompt: "a fox",
            rawAdapterSettings: { guidanceMethod: "cfg_pp" },
          },
        },
      }),
    );
    await openAdvanced();
    expect(field(container, "Guidance method").value).toBe("cfg_pp");

    await generate();
    expect(createImageJob.mock.calls[0][0].advanced.guidanceMethod).toBe("cfg_pp");
  });
});

describe("ImageStudio edit source picker", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  async function openEditSourcePicker(context) {
    await render(context);
    await click([...document.body.querySelectorAll(".mode-tabs button")].find((button) => button.textContent === "Edit"));
    await click([...document.body.querySelectorAll(".asset-picker-head button")].find((button) => button.textContent === "Select image"));
    return document.body.querySelector('[role="dialog"]');
  }

  it("limits Image Edit source selection to active project images and shows the requested source tabs", async () => {
    const active = { id: "asset-active", projectId: "project_1", type: "image", displayName: "Active Plate", status: { trashed: false } };
    const trashed = { id: "asset-trashed", projectId: "project_1", type: "image", displayName: "Discarded Plate", status: { trashed: true } };
    const rejected = { id: "asset-rejected", projectId: "project_1", type: "image", displayName: "Rejected Plate", status: { rejected: true } };
    const otherProject = { id: "asset-other", projectId: "project_2", type: "image", displayName: "Other Project Plate", status: {} };
    const video = { id: "asset-video", projectId: "project_1", type: "video", displayName: "Video Clip", status: {} };

    const dialog = await openEditSourcePicker(
      baseContext({
        assets: [active, trashed, rejected, otherProject, video],
        imageModels: [{ ...Z_IMAGE, capabilities: ["edit_image"] }],
        selectedAsset: null,
      }),
    );

    const sourceTabs = [...dialog.querySelectorAll('[role="tab"]')].map((button) => button.textContent.trim());
    expect(sourceTabs).toEqual(["Assets1", "File Upload", "Character0"]);
    expect(dialog.textContent).toContain("Active Plate");
    expect(dialog.textContent).not.toContain("Discarded Plate");
    expect(dialog.textContent).not.toContain("Rejected Plate");
    expect(dialog.textContent).not.toContain("Other Project Plate");
    expect(dialog.textContent).not.toContain("Video Clip");
    expect(dialog.textContent).not.toContain("Renders");
  });

  it("filters the Character source tab by project, character, and active status", async () => {
    const mira = { id: "char-1", name: "Mira", approvedReferences: [{ assetId: "ref-mira" }] };
    const echo = { id: "char-2", name: "Echo", approvedReferences: [] };
    const assets = [
      { id: "ref-mira", projectId: "project_1", type: "image", displayName: "Mira Reference", status: {} },
      {
        id: "mira-render",
        projectId: "project_1",
        type: "image",
        displayName: "Mira Render",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: {},
      },
      {
        id: "mira-trash",
        projectId: "project_1",
        type: "image",
        displayName: "Mira Discarded",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: { trashed: true },
      },
      {
        id: "echo-render",
        projectId: "project_1",
        type: "image",
        displayName: "Echo Render",
        recipe: { normalizedSettings: { characterId: "char-2" } },
        status: {},
      },
      {
        id: "mira-other-project",
        projectId: "project_2",
        type: "image",
        displayName: "Mira Elsewhere",
        recipe: { normalizedSettings: { characterId: "char-1" } },
        status: {},
      },
    ];

    const dialog = await openEditSourcePicker(
      baseContext({
        assets,
        characters: [mira, echo],
        imageModels: [{ ...Z_IMAGE, capabilities: ["edit_image"] }],
        selectedAsset: null,
      }),
    );

    await click([...dialog.querySelectorAll('[role="tab"]')].find((button) => button.textContent.includes("Character")));
    expect(dialog.textContent).toContain("Mira Reference");
    expect(dialog.textContent).toContain("Mira Render");
    expect(dialog.textContent).not.toContain("Mira Discarded");
    expect(dialog.textContent).not.toContain("Echo Render");
    expect(dialog.textContent).not.toContain("Mira Elsewhere");

    await act(async () => {
      dialog.querySelector(".asset-picker-card").click();
    });
    await click([...dialog.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));

    expect(container.textContent).toContain("Mira Reference");
  });

  it("imports a File Upload source and submits it as the edit source image", async () => {
    const imported = {
      id: "uploaded-source",
      projectId: "project_1",
      type: "image",
      displayName: "uploaded.png",
      status: {},
    };
    const importAsset = vi.fn(async () => imported);
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));

    const dialog = await openEditSourcePicker(
      baseContext({
        assets: [],
        createImageJob,
        imageModels: [{ ...Z_IMAGE, capabilities: ["edit_image"] }],
        importAsset,
        selectedAsset: null,
      }),
    );

    await click([...dialog.querySelectorAll('[role="tab"]')].find((button) => button.textContent === "File Upload"));
    const file = new File(["image"], "source.png", { type: "image/png" });
    await act(async () => setFileInput(dialog.querySelector('input[type="file"]'), [file]));
    await act(async () => {});

    expect(importAsset).toHaveBeenCalledWith(file, { throwOnError: true });
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();

    await click([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate"));
    expect(createImageJob).toHaveBeenCalledWith(expect.objectContaining({ mode: "edit_image", sourceAssetId: "uploaded-source" }));
  });

  it("uses the multi-image reference picker for a multiReference model and submits referenceAssetIds (sc-6211)", async () => {
    const refA = { id: "ref-a", projectId: "project_1", type: "image", displayName: "Ref A", status: {} };
    const refB = { id: "ref-b", projectId: "project_1", type: "image", displayName: "Ref B", status: {} };
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const FLUX2_DEV = {
      ...Z_IMAGE,
      id: "flux2_dev",
      name: "FLUX.2 dev",
      capabilities: ["text_to_image", "edit_image"],
      ui: { multiReference: true },
    };

    await render(
      baseContext({
        assets: [refA, refB],
        createImageJob,
        imageModels: [FLUX2_DEV],
        selectedAsset: null,
      }),
    );
    await click([...document.body.querySelectorAll(".mode-tabs button")].find((button) => button.textContent === "Edit"));

    // The multi-image picker ("Select images") replaces the single source picker ("Select image").
    const headButtons = () => [...document.body.querySelectorAll(".asset-picker-head button")];
    expect(headButtons().some((button) => button.textContent === "Select images")).toBe(true);
    expect(headButtons().some((button) => button.textContent === "Select image")).toBe(false);

    await click(headButtons().find((button) => button.textContent === "Select images"));
    const dialog = document.body.querySelector('[role="dialog"]');
    const cards = [...dialog.querySelectorAll(".asset-picker-card")];
    await click(cards[0]);
    await click(cards[1]);
    await click([...dialog.querySelectorAll("button")].find((button) => button.textContent === "Use Selection"));

    await click([...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.mode).toBe("edit_image");
    expect(payload.referenceAssetIds).toEqual(["ref-a", "ref-b"]);
    expect(payload.sourceAssetId).toBeNull();
  });
});

// The Krea 2 image-edit surface (epic 10871, P4.1): edit REQUIRES the `image_edit`-role LoRA (R5),
// which the studio MANAGES for the user — auto-applied when installed, a one-click download when
// not — rather than leaving it to be hand-picked.
describe("ImageStudio Krea image edit LoRA (epic 10871)", () => {
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

  const KREA_RAW = {
    ...Z_IMAGE,
    id: "krea_2_raw",
    name: "Krea 2 Raw",
    family: "krea_2",
    capabilities: ["text_to_image", "edit_image"],
  };
  const EDIT_LORA = {
    id: "krea2_identity_edit",
    name: "Krea 2 Identity Edit",
    family: "krea_2",
    conditioningRole: "image_edit",
    defaultWeight: 1,
    scope: "builtin",
    installedPath: "/loras/krea2_identity_edit",
    installState: "installed",
  };
  const SOURCE = { id: "src-plate", projectId: "project_1", type: "image", displayName: "Plate", status: {} };

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }
  const enterEdit = () =>
    click([...document.body.querySelectorAll(".mode-tabs button")].find((b) => b.textContent === "Edit"));

  it("auto-applies the installed edit LoRA to the payload with its conditioning role (R5)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [KREA_RAW], loras: [EDIT_LORA], assets: [SOURCE], selectedAsset: null }),
    );
    await enterEdit();
    // The managed note tells the user it's automatic — no manual picking.
    expect(container.textContent).toContain("applied automatically");

    // Pick the source image, then generate.
    await click([...document.body.querySelectorAll(".asset-picker-head button")].find((b) => b.textContent === "Select image"));
    const dialog = document.body.querySelector('[role="dialog"]');
    await act(async () => dialog.querySelector(".asset-picker-card").click());
    await click([...dialog.querySelectorAll("button")].find((b) => b.textContent === "Use Selection"));
    await click([...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate"));

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.mode).toBe("edit_image");
    expect(payload.sourceAssetId).toBe("src-plate");
    const editEntry = payload.loras.find((l) => l.id === "krea2_identity_edit");
    expect(editEntry).toBeTruthy();
    expect(editEntry.conditioningRole).toBe("image_edit");
    // Identity strength (sc-11798): with no slider interaction the payload carries the manifest
    // default weight, and the Identity strength control renders for the managed edit LoRA.
    expect(editEntry.weight).toBe(1);
    expect(container.textContent).toContain("Identity strength");
    // Deduped — auto-applied exactly once.
    expect(payload.loras.filter((l) => l.id === "krea2_identity_edit")).toHaveLength(1);
  });

  it("offers a one-click download and blocks Generate when the edit LoRA is not installed", async () => {
    const createLoraDownloadJob = vi.fn();
    await render(
      baseContext({
        imageModels: [KREA_RAW],
        loras: [{ ...EDIT_LORA, installState: "missing", installedPath: null }],
        assets: [SOURCE],
        createLoraDownloadJob,
      }),
    );
    await enterEdit();

    const download = [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Download");
    expect(download).toBeTruthy();
    const generate = [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate");
    expect(generate.disabled).toBe(true);

    await click(download);
    expect(createLoraDownloadJob).toHaveBeenCalledWith(expect.objectContaining({ id: "krea2_identity_edit" }));
  });

  it("keeps the managed edit LoRA out of the manual picker while still offering other LoRAs", async () => {
    const KREA_STYLE = { id: "krea_style", name: "Krea Style", family: "krea_2", scope: "global", installState: "installed", installedPath: "/loras/krea_style" };
    await render(baseContext({ imageModels: [KREA_RAW], loras: [EDIT_LORA, KREA_STYLE], assets: [SOURCE] }));
    await enterEdit();
    await click(document.body.querySelector(".advanced-section-toggle"));

    // The managed edit LoRA is filtered out — only the non-managed style LoRA is on offer.
    const addButton = document.body.querySelector(".lora-add");
    expect(addButton.getAttribute("data-count")).toBe("· 1 available");
    await click(addButton);
    const rows = [...document.body.querySelectorAll(".lora-pick-row")].map((node) => node.textContent);
    expect(rows.some((text) => text.includes("Krea Style"))).toBe(true);
    expect(rows.some((text) => text.includes("Krea 2 Identity Edit"))).toBe(false);
  });

  // Two-reference edit (epic 10871 P1.3): a model whose `ui.editReferences` adds an optional second
  // source — any two images, image 1 (required) + image 2 (optional), fixed order.
  const KREA_RAW_TWOREF = {
    ...KREA_RAW,
    ui: {
      editReferences: {
        secondaryLabel: "Image 2 (optional)",
        secondaryHint: "Optional — a second image to combine with Image 1.",
      },
    },
  };
  const SCENE = { id: "scene-plate", projectId: "project_1", type: "image", displayName: "Scene Plate", status: {} };
  const PERSON = { id: "person-plate", projectId: "project_1", type: "image", displayName: "Person Plate", status: {} };

  // Select an asset into the next empty source picker (the first picker's button flips to "Change"
  // once set, so the remaining "Select image" button is always the next empty slot).
  async function pickNextSource(assetName) {
    const btn = [...document.body.querySelectorAll(".asset-picker-head button")].find(
      (b) => b.textContent === "Select image",
    );
    await click(btn);
    const dialog = document.body.querySelector('[role="dialog"]');
    const card = [...dialog.querySelectorAll(".asset-picker-card")].find((c) => c.textContent.includes(assetName));
    await act(async () => card.click());
    await click([...dialog.querySelectorAll("button")].find((b) => b.textContent === "Use Selection"));
  }

  it("renders the optional second-image picker only when the model declares ui.editReferences", async () => {
    const { root: root2, container: c2 } = mountRoot();
    // Plain Krea (no editReferences) → no second-image slot.
    await act(async () => {
      root2.render(
        <AppContext.Provider value={baseContext({ imageModels: [KREA_RAW], loras: [EDIT_LORA], assets: [SCENE] })}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
    await click([...document.body.querySelectorAll(".mode-tabs button")].find((b) => b.textContent === "Edit"));
    expect(document.body.textContent).not.toContain("Image 2 (optional)");
    await unmountRoot(root2, c2);

    // Krea with editReferences → the labeled optional second-image slot appears.
    await render(baseContext({ imageModels: [KREA_RAW_TWOREF], loras: [EDIT_LORA], assets: [SCENE, PERSON] }));
    await enterEdit();
    expect(container.textContent).toContain("Image 2 (optional)");
    expect(container.textContent).toContain("No second image selected (optional)");
  });

  it("sends the ordered [image1, image2] pair as referenceAssetIds when a second image is chosen", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [KREA_RAW_TWOREF], loras: [EDIT_LORA], assets: [SCENE, PERSON], selectedAsset: null }),
    );
    await enterEdit();
    await pickNextSource("Scene Plate"); // → sourceAssetId (image 1)
    await pickNextSource("Person Plate"); // → editSecondAssetId (image 2)
    await click([...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate"));

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.mode).toBe("edit_image");
    // Fixed order preserved: image 1 first, image 2 second.
    expect(payload.referenceAssetIds).toEqual(["scene-plate", "person-plate"]);
    // The single sourceAssetId is dropped in favor of the ordered pair.
    expect(payload.sourceAssetId).toBeNull();
    // The edit LoRA is still auto-applied (R5).
    expect(payload.loras.some((l) => l.id === "krea2_identity_edit")).toBe(true);
  });

  it("falls back to the single sourceAssetId when no second image is chosen (image 2 is optional)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [KREA_RAW_TWOREF], loras: [EDIT_LORA], assets: [SCENE, PERSON], selectedAsset: null }),
    );
    await enterEdit();
    await pickNextSource("Scene Plate");
    await click([...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate"));

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.sourceAssetId).toBe("scene-plate");
    expect(payload.referenceAssetIds).toBeUndefined();
  });
});

describe("ImageStudio model picker capability gating", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // One model per capability class so each mode's picker can be checked in isolation.
  const T2I = { ...Z_IMAGE, id: "t2i_only", name: "T2I Only", capabilities: ["text_to_image"] };
  const VARIATIONS = {
    ...Z_IMAGE,
    id: "variations_model",
    name: "Variations Model",
    capabilities: ["text_to_image", "style_variations"],
  };
  const EDIT_ONLY = { ...Z_IMAGE, id: "edit_only", name: "Edit Only", capabilities: ["edit_image", "image_to_image"] };
  const CHARACTER_ONLY = { ...Z_IMAGE, id: "character_only", name: "Character Only", capabilities: ["character_image"] };
  const MAC_CAPS = {
    macGatingActive: true,
    platform: "darwin",
    notAvailableLabel: "Not available on Mac (MLX only)",
    features: {},
    training: { supportedKernels: [], lokrOnWanSupported: false },
  };
  const LENS_TURBO = {
    ...Z_IMAGE,
    id: "lens_turbo",
    name: "Lens-Turbo",
    capabilities: ["text_to_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };
  const QWEN_EDIT = {
    ...Z_IMAGE,
    id: "qwen_image_edit",
    name: "Qwen Image Edit",
    capabilities: ["edit_image"],
    macSupport: { supported: true, features: { edit: true, reference: false } },
  };
  const TORCH_ONLY_EDIT = {
    ...Z_IMAGE,
    id: "torch_only_edit",
    name: "Torch-only Edit",
    capabilities: ["edit_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };

  const modelOptionValues = () => [...field(container, "Model").options].map((option) => option.value);
  const modeButton = (label) =>
    [...document.body.querySelectorAll(".mode-tabs button")].find((button) => button.textContent === label);

  it("Text tab lists only text_to_image models, excluding edit-only and character-only (sc-5549)", async () => {
    await render(baseContext({ imageModels: [EDIT_ONLY, T2I, VARIATIONS, CHARACTER_ONLY] }));

    const options = modelOptionValues();
    expect(options).toContain("t2i_only");
    expect(options).toContain("variations_model"); // declares text_to_image
    expect(options).not.toContain("edit_only");
    expect(options).not.toContain("character_only");
  });

  it("enables the Mac Edit tab when any available model supports edit mode (sc-5589)", async () => {
    await render(
      baseContext({
        imageModels: [LENS_TURBO, TORCH_ONLY_EDIT, QWEN_EDIT],
        macCapabilities: MAC_CAPS,
      }),
    );

    expect(field(container, "Model").value).toBe("lens_turbo");
    expect(modeButton("Edit").disabled).toBe(false);

    await click(modeButton("Edit"));
    await act(async () => {});

    expect(modeButton("Edit").className).toContain("active");
    expect(field(container, "Model").value).toBe("qwen_image_edit");
    expect(modelOptionValues()).toEqual(["qwen_image_edit"]);
  });

  it("disables the Mac Edit tab when no available model supports edit mode", async () => {
    await render(
      baseContext({
        imageModels: [LENS_TURBO, TORCH_ONLY_EDIT],
        macCapabilities: MAC_CAPS,
      }),
    );

    expect(modeButton("Edit").disabled).toBe(true);
    expect(modeButton("Edit").title).toBe("No available Mac model supports this mode.");
  });

  // Boogu-Image-0.1 (epic 6387 / sc-6400) is backend-driven — no dedicated JSX. Base/Turbo are
  // text-to-image, Edit is the instruction-edit checkpoint, and (unlike Ideogram) Boogu is
  // natural-language, so it must render the plain prompt textarea, NOT the structured caption builder.
  const BOOGU_BASE = {
    ...Z_IMAGE,
    id: "boogu_image",
    name: "Boogu Image",
    family: "boogu",
    capabilities: ["text_to_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };
  const BOOGU_TURBO = {
    ...Z_IMAGE,
    id: "boogu_image_turbo",
    name: "Boogu Image Turbo",
    family: "boogu",
    capabilities: ["text_to_image"],
    macSupport: { supported: true, features: { edit: false, reference: false } },
  };
  const BOOGU_EDIT = {
    ...Z_IMAGE,
    id: "boogu_image_edit",
    name: "Boogu Image Edit",
    family: "boogu",
    capabilities: ["edit_image"],
    macSupport: { supported: true, features: { edit: true, reference: false } },
  };

  it("surfaces Boogu Base/Turbo in Text (plain prompt, not the structured builder) and Edit in the Edit tab (sc-6400)", async () => {
    await render(
      baseContext({
        imageModels: [BOOGU_BASE, BOOGU_TURBO, BOOGU_EDIT],
        macCapabilities: MAC_CAPS,
      }),
    );

    // Text tab: Base + Turbo (text_to_image); the Edit checkpoint is excluded.
    const textOptions = modelOptionValues();
    expect(textOptions).toContain("boogu_image");
    expect(textOptions).toContain("boogu_image_turbo");
    expect(textOptions).not.toContain("boogu_image_edit");

    // Boogu is natural-language (no `structuredPrompt`) → the plain prompt textarea, NOT the
    // Ideogram structured-caption builder.
    expect(document.body.querySelector('textarea[aria-label="Prompt"]')).toBeTruthy();

    // Edit tab enabled, and lists only the Edit checkpoint.
    expect(modeButton("Edit").disabled).toBe(false);
    await click(modeButton("Edit"));
    await act(async () => {});
    expect(field(container, "Model").value).toBe("boogu_image_edit");
    expect(modelOptionValues()).toEqual(["boogu_image_edit"]);
  });

  it("offers the Refine-my-prompt control for Boogu — prompt enhancement reuses prompt_refine (sc-6401)", async () => {
    await render(baseContext({ imageModels: [BOOGU_BASE, BOOGU_TURBO], macCapabilities: MAC_CAPS }));

    // Boogu is non-structured, so the plain-prompt path renders RefinePromptControl ("Refine my
    // prompt"). It drives the prompt_refine utility with Boogu's prompt guide as the rewriter context
    // (S4) — the optional, user-editable enhancement step; raw prompt remains the fallback.
    const refineButton = [...document.body.querySelectorAll("button")].find((b) =>
      b.textContent.includes("Refine my prompt"),
    );
    expect(refineButton).toBeTruthy();
  });

  // The Boogu precision checkbox specifically (class-scoped so it never collides with the tier
  // picker's "Full precision (bf16)" <option> text when both would otherwise be in the DOM).
  const precisionLabel = (root) =>
    root.querySelector("label.boogu-precision-toggle");
  const openAdvanced = async (root) =>
    click(root.querySelector(".advanced-section-toggle"));

  it("exposes the Full-precision (bf16) toggle for Boogu in Advanced when ui.precisionToggle is set (sc-6568)", async () => {
    await render(
      baseContext({
        imageModels: [{ ...BOOGU_BASE, ui: { precisionToggle: true } }],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    const toggle = precisionLabel(container);
    expect(toggle).toBeTruthy();
    // Default off → the packed Q8 build (no mlxQuantize emitted).
    expect(toggle.querySelector('input[type="checkbox"]').checked).toBe(false);
  });

  it("hides the precision toggle when the model omits ui.precisionToggle — catalog-gated, not family-hardcoded (sc-6568)", async () => {
    await render(baseContext({ imageModels: [BOOGU_BASE], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    expect(precisionLabel(container)).toBeFalsy();
  });

  // Generation-time quant-tier toggle (sc-8515). A quant-matrix model exposes per-tier install
  // state; the studio renders a picker only when >1 tier is installed, and routes the pick via
  // advanced.mlxQuantize (bf16→0, q8→8, q4→4). `installed` lists which tier keys are on disk.
  const matrixModel = (installed, defaultTier = "q4") => ({
    ...Z_IMAGE,
    id: "z_image_turbo",
    hasVariantMatrix: true,
    variants: ["q4", "q8", "bf16"].map((tier) => ({
      variant: tier,
      default: tier === defaultTier,
      installState: installed.includes(tier) ? "installed" : "missing",
    })),
  });
  const tierPicker = (root) =>
    [...root.querySelectorAll("label.quant-tier-picker select")][0] ?? null;
  const generateButton = () =>
    [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate");

  it("shows the quant-tier picker with only installed tiers when >1 is installed (sc-8515)", async () => {
    await render(baseContext({ imageModels: [matrixModel(["q4", "bf16"])], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    const picker = tierPicker(container);
    expect(picker).toBeTruthy();
    const options = [...picker.options].map((o) => o.value);
    // q8 is declared but not installed → excluded; smallest→largest order.
    expect(options).toEqual(["q4", "bf16"]);
  });

  it("hides the picker when only one tier is installed (sc-8515)", async () => {
    await render(baseContext({ imageModels: [matrixModel(["q4"])], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    expect(tierPicker(container)).toBeFalsy();
  });

  it("hides the picker for a model with no variant matrix (sc-8515)", async () => {
    await render(baseContext({ imageModels: [Z_IMAGE], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    expect(tierPicker(container)).toBeFalsy();
  });

  // PiD decode and Upscale both super-resolve, so they're mutually exclusive: enabling one
  // disables the other (and the upscale sub-controls).
  it("makes PiD and Upscale mutually exclusive in Advanced", async () => {
    const PID_MODEL = {
      ...Z_IMAGE,
      id: "qwen_image",
      name: "Qwen Image",
      ui: { pid: { checkpointId: "pid_qwenimage" } },
    };
    await render(
      baseContext({
        imageModels: [PID_MODEL],
        models: [{ id: "pid_qwenimage", installState: "installed" }],
      }),
    );
    await openAdvanced(container);
    await act(async () => {});

    const pid = () => container.querySelector('.pid-decoder-toggle input[type="checkbox"]');
    const upscale = () => container.querySelector('.upscale-toggle input[type="checkbox"]');
    expect(pid()).toBeTruthy();
    expect(upscale()).toBeTruthy();
    // Both start enabled.
    expect(pid().disabled).toBe(false);
    expect(upscale().disabled).toBe(false);

    // PiD on → Upscale + its Scale/Engine sub-controls disable.
    await act(async () => pid().click());
    expect(upscale().disabled).toBe(true);
    expect(field(container, "Scale").disabled).toBe(true);
    expect(field(container, "Engine").disabled).toBe(true);

    // PiD off, Upscale on → PiD disables.
    await act(async () => pid().click());
    await act(async () => upscale().click());
    expect(pid().disabled).toBe(true);
  });

  it("omits advanced.mlxQuantize on Generate when only one tier is installed (sc-8515)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [matrixModel(["q4"])],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    // Picker is hidden (single tier), so the pick must never leak into the payload.
    expect(tierPicker(container)).toBeFalsy();
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("mlxQuantize");
  });

  it("omits advanced.mlxQuantize on Generate for a model with no variant matrix (sc-8515)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [Z_IMAGE], macCapabilities: MAC_CAPS }),
    );
    await openAdvanced(container);
    await act(async () => {});
    expect(tierPicker(container)).toBeFalsy();
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("mlxQuantize");
  });

  it("defaults to the declared default tier and sends its mlxQuantize on Generate (sc-8515)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [matrixModel(["q4", "q8", "bf16"], "q4")],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    expect(tierPicker(container).value).toBe("q4");
    await click(generateButton());
    // Default q4 → mlxQuantize 4.
    expect(createImageJob.mock.calls[0][0].advanced.mlxQuantize).toBe(4);
  });

  it("routes the selected tier through advanced.mlxQuantize (bf16→0, sc-8515)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [matrixModel(["q4", "bf16"])],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    setSelect(tierPicker(container), "bf16");
    await act(async () => {});
    // Reload-always: switching surfaces a transient "loading" note.
    expect(container.querySelector("label.quant-tier-picker [role='status']")).toBeTruthy();
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.mlxQuantize).toBe(0);
  });

  it("routes q8 through advanced.mlxQuantize as 8 (sc-8515)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [matrixModel(["q8", "bf16"])],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    setSelect(tierPicker(container), "q8");
    await act(async () => {});
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.mlxQuantize).toBe(8);
  });

  // Per-model quality floor (sc-10731). A floored convert-at-install model (Anima base = q8 floor) is a
  // decoupled `mlxTiers` model: the DEFAULT clamps UP to the floor, and an EXPLICIT below-floor pick is
  // honored (never silently switched) but flagged with a non-blocking advisory.
  // `floor` is passed explicitly (no default) so `undefined` yields a genuinely non-floored model.
  const flooredModel = (mlxTiers, floor) => ({
    ...Z_IMAGE,
    id: "anima_base",
    name: "Anima 2B",
    hasVariantMatrix: false,
    variants: undefined,
    mlxTiers,
    minQualityTier: floor,
  });
  const floorNote = () => container.querySelector(".quant-tier-floor-note");

  it("floors the default tier to q8 and flags an explicit below-floor pick (sc-10731)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    // Even with the global default set to q4, the q8-floored model clamps the DEFAULT up to q8.
    window.localStorage.setItem("sceneworks-default-generation-quality", "q4");
    await render(
      baseContext({
        createImageJob,
        imageModels: [flooredModel(["q4", "q8"], "q8")],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    const picker = tierPicker(container);
    expect(picker).toBeTruthy();
    // Acceptance #1: global q4 + floored model → q8 (floored), and no advisory at the floor.
    expect(picker.value).toBe("q8");
    expect(floorNote()).toBeFalsy();
    // Acceptance #2: explicitly pick the below-floor q4 → honored (value stays q4) + advisory appears.
    setSelect(picker, "q4");
    await act(async () => {});
    expect(tierPicker(container).value).toBe("q4");
    expect(floorNote()).toBeTruthy();
    // The explicit q4 is honored on Generate (not switched back to the floor).
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.mlxQuantize).toBe(4);
  });

  it("does not flag or clamp a NON-floored model's q4 default (sc-10731, acceptance #3)", async () => {
    // Same convert-at-install shape but no floor → the global q4 default is honored, no advisory.
    window.localStorage.setItem("sceneworks-default-generation-quality", "q4");
    await render(
      baseContext({
        imageModels: [flooredModel(["q4", "q8"], undefined)],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    expect(tierPicker(container).value).toBe("q4");
    expect(floorNote()).toBeFalsy();
  });

  // Disjointness guard (sc-8515): the tier picker and Boogu's ui.precisionToggle both write
  // advanced.mlxQuantize and MUST never co-render/co-emit. In the catalog they are disjoint —
  // Boogu downloads via `base/`-style subfolder globs (no downloads[].variant keys), so it is
  // not a hasVariantMatrix model and showTierPicker is false for it. This constructs the
  // (catalog-impossible) both-set model to prove the `!showTierPicker` guard keeps them apart.
  it("suppresses the precision toggle and emits only the tier quant when a matrix model also sets precisionToggle (sc-8515)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({
        createImageJob,
        imageModels: [{ ...matrixModel(["q4", "bf16"], "q4"), ui: { precisionToggle: true } }],
        macCapabilities: MAC_CAPS,
      }),
    );
    await openAdvanced(container);
    await act(async () => {});
    // Tier picker wins; the Boogu precision checkbox is suppressed (guarded by !showTierPicker).
    expect(tierPicker(container)).toBeTruthy();
    expect(precisionLabel(container)).toBeFalsy();
    // Default tier q4 → mlxQuantize 4, NOT the precision-toggle's bf16 sentinel 0.
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.mlxQuantize).toBe(4);
  });

  // Per-(screen, model) sticky last-tier (sc-10727 / epic 10721). An explicit pick is persisted and
  // reused as the default on the next visit to that model on this screen, surviving app restarts.
  it("persists the explicit tier pick per (screen, model) across a remount, above the declared default (sc-10727)", async () => {
    // All three tiers installed; declared default is q4.
    const model = matrixModel(["q4", "q8", "bf16"], "q4");
    await render(baseContext({ imageModels: [model], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    // First visit: seeds on the declared default (no sticky stored yet).
    expect(tierPicker(container).value).toBe("q4");
    // User explicitly picks q8 → written to the persistent per-(screen, model) store.
    setSelect(tierPicker(container), "q8");
    await act(async () => {});
    expect(tierPicker(container).value).toBe("q8");

    // Simulate an app restart: tear the tree down and mount a FRESH one (React state is gone; only
    // the persisted sticky survives), then re-render the SAME model. (The Advanced panel's open
    // state persists in the studio-settings blob, so it may already be open — guard the toggle.)
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render(baseContext({ imageModels: [model], macCapabilities: MAC_CAPS }));
    if (!tierPicker(container)) {
      await openAdvanced(container);
      await act(async () => {});
    }
    // The sticky q8 now seeds the picker, winning over the declared default q4 — persistence +
    // precedence (sticky > declared/base default) proven end-to-end.
    expect(tierPicker(container).value).toBe("q8");
  });

  it("keeps the sticky independent per model — a pick on model X does not affect model Y (sc-10727)", async () => {
    const modelX = matrixModel(["q4", "q8", "bf16"], "q4");
    const modelY = { ...matrixModel(["q4", "q8", "bf16"], "q4"), id: "other_model", name: "Other Model" };

    await render(baseContext({ imageModels: [modelX], macCapabilities: MAC_CAPS }));
    await openAdvanced(container);
    await act(async () => {});
    setSelect(tierPicker(container), "q8");
    await act(async () => {});
    expect(tierPicker(container).value).toBe("q8");

    // Fresh mount, render a DIFFERENT model (Y). Its picker seeds on its OWN default (q4), untouched
    // by X's q8 sticky — a global (non-model-keyed) store would wrongly surface q8 here. (Advanced
    // may already be open from the persisted studio-settings blob — guard the toggle.)
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render(baseContext({ imageModels: [modelY], macCapabilities: MAC_CAPS }));
    if (!tierPicker(container)) {
      await openAdvanced(container);
      await act(async () => {});
    }
    expect(tierPicker(container).value).toBe("q4");
  });
});

describe("ImageStudio structured-prompt recipe round-trip (sc-6147)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const IDEOGRAM = {
    ...Z_IMAGE,
    id: "ideogram_4",
    name: "Ideogram 4",
    family: "ideogram",
    capabilities: ["text_to_image"],
    structuredPrompt: true,
  };

  const CAPTION = {
    high_level_description: "A red fox in the snow",
    compositional_deconstruction: {
      background: "A snowy pine forest",
      elements: [{ type: "obj", desc: "a red fox sitting upright" }],
    },
  };

  const generateButton = () =>
    [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Generate");

  it("restores the builder from a recipe, then re-emits the same caption + blob on Generate", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const structuredPrompt = buildStructuredPromptRecipe({
      intent: "a red fox in the snow",
      caption: CAPTION,
      magicPromptBackend: "prompt_refine",
    });
    await render(
      baseContext({
        createImageJob,
        imageModels: [IDEOGRAM],
        studioLaunch: {
          id: "launch-1",
          view: "Image",
          assetId: "asset-1",
          // Mirrors a stored Ideogram asset: prompt = serialized caption, with the
          // full structured-prompt blob under rawAdapterSettings.structuredPrompt.
          recipe: {
            model: "ideogram_4",
            mode: "text_to_image",
            prompt: serializeCaption(CAPTION),
            rawAdapterSettings: { structuredPrompt },
          },
        },
      }),
    );

    // Restore selected the structured model and rehydrated the builder (Generate is
    // enabled, which requires a valid, non-empty caption in the form — not plain text).
    expect(field(container, "Model").value).toBe("ideogram_4");
    expect(generateButton().disabled).toBe(false);

    await click(generateButton());

    const payload = createImageJob.mock.calls[0][0];
    // Top-level prompt is the canonical serialized caption — byte-identical to source.
    expect(payload.prompt).toBe(serializeCaption(CAPTION));
    // The full structured-prompt blob round-trips through advanced (→ rawAdapterSettings).
    expect(payload.advanced.structuredPrompt.caption).toEqual(CAPTION);
    expect(payload.advanced.structuredPrompt.intent).toBe("a red fox in the snow");
    expect(payload.advanced.structuredPrompt.magicPromptBackend).toBe("prompt_refine");
    expect(payload.advanced.structuredPrompt.runtimePrompt).toBe(serializeCaption(CAPTION));
  });

  it("does not attach a structured-prompt blob for non-structured models", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(baseContext({ createImageJob, imageModels: [Z_IMAGE] }));

    await click(generateButton());

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.advanced.structuredPrompt).toBeUndefined();
  });
});

describe("ImageStudio reference-image → JSON caption (epic 8102, sc-8108)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const IDEOGRAM = {
    ...Z_IMAGE,
    id: "ideogram_4",
    name: "Ideogram 4",
    family: "ideogram",
    capabilities: ["text_to_image"],
    structuredPrompt: true,
  };

  const REF_ASSET = {
    id: "ref-asset-1",
    type: "image",
    projectId: "project_1",
    file: { path: "uploads/ref.png", mimeType: "image/png" },
  };

  // sc-8110: the reference-image flow goes live only when the vision captioner is installed (the
  // ModelAvailabilityGate gates it). Supply the catalog entry so the live picker + button render.
  const VISION_MODEL_INSTALLED = {
    id: VISION_CAPTION_MODEL_ID,
    type: "utility",
    macOnly: false, // cross-platform as of sc-8116 (epic 8103) — matches the real catalog entry
    installState: "installed",
  };

  // The vision model reply carries a grounded bbox; parseVisionCaption KEEPS bboxes (strips only
  // aspect_ratio), so the injected caption must still carry the box.
  const VISION_REPLY = JSON.stringify({
    aspect_ratio: "1:1",
    high_level_description: "a red fox in the snow",
    compositional_deconstruction: {
      background: "a snowy forest",
      elements: [{ type: "obj", bbox: [100, 100, 600, 600], desc: "a red fox" }],
    },
  });

  const buttonByText = (text) =>
    [...document.body.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

  // Switch the structured builder to its Plain-text tab, where the reference-image flow lives.
  async function openPlainTab() {
    const plainTab = [...document.body.querySelectorAll(".structured-mode button")].find(
      (b) => b.textContent.trim() === "Plain text",
    );
    await click(plainTab);
  }

  it("shows the reference-image flow for Ideogram 4 in text-to-image mode", async () => {
    await render(
      baseContext({ imageModels: [IDEOGRAM], models: [VISION_MODEL_INSTALLED], imageCaption: vi.fn() }),
    );
    await openPlainTab();
    expect(buttonByText("✨ Generate JSON from image")).toBeTruthy();
    expect(document.body.querySelector(".structured-reference")).toBeTruthy();
  });

  it("gates the reference flow behind a download offer when the captioner is missing (sc-8110)", async () => {
    await render(
      baseContext({
        imageModels: [IDEOGRAM],
        models: [{ ...VISION_MODEL_INSTALLED, installState: "missing", recommended: true, name: "Vision Captioner" }],
        imageCaption: vi.fn(),
      }),
    );
    await openPlainTab();
    // The section is present (Ideogram 4 + t2i), but the live button is hidden behind the gate.
    expect(document.body.querySelector(".structured-reference")).toBeTruthy();
    expect(buttonByText("✨ Generate JSON from image")).toBeFalsy();
    expect(document.body.querySelector(".model-availability-gate")).toBeTruthy();
    expect(buttonByText("Download")).toBeTruthy();
  });

  it("captions a reference image into the builder, keeping bboxes (sc-8108)", async () => {
    const imageCaption = vi.fn(async () => VISION_REPLY);
    await render(
      baseContext({ imageModels: [IDEOGRAM], models: [VISION_MODEL_INSTALLED], assets: [REF_ASSET], imageCaption }),
    );
    await openPlainTab();

    // Pick the reference through the asset picker modal so the button enables.
    await click(buttonByText("Select reference image"));
    await click(document.body.querySelector(".asset-picker-card"));
    await click(buttonByText("Use Selection"));

    await click(buttonByText("✨ Generate JSON from image"));
    await act(async () => {});

    // Dispatched with the asset id, the project id, and the vision model's HF repo string.
    expect(imageCaption).toHaveBeenCalledTimes(1);
    const arg = imageCaption.mock.calls[0][0];
    expect(arg.sourceAssetId).toBe("ref-asset-1");
    expect(arg.projectId).toBe("project_1");
    expect(arg.model).toBe("huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated");

    // The builder is populated (it switched to the form) and the grounded bbox survived.
    const preview = document.body.querySelector('[aria-label="Caption preview"]');
    expect(preview).toBeTruthy();
    expect(preview.textContent).toContain("a red fox");
    expect(preview.textContent).toContain("100");
    // aspect_ratio is NOT a schema key and must be stripped.
    expect(preview.textContent).not.toContain("aspect_ratio");
  });
});

describe("ImageStudio Ideogram 4 auto-expand on plain-text Generate (sc-6501)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const IDEOGRAM = {
    ...Z_IMAGE,
    id: "ideogram_4",
    name: "Ideogram 4",
    family: "ideogram",
    capabilities: ["text_to_image"],
    structuredPrompt: true,
  };

  const REFINE_READY = { id: PROMPT_REFINE_MODEL_ID, name: "Prompt Refiner", installState: "ready" };

  // A raw magic-prompt model reply (JSON string), as the worker would return it. `onMagicExpand`
  // runs it through parseMagicPromptCaption, so the caption the studio sends is EXPANDED.
  const RAW_CAPTION = JSON.stringify({
    aspect_ratio: "1:1",
    high_level_description: "A red fox on a sunny beach",
    compositional_deconstruction: {
      background: "a sunlit sandy beach with gentle waves",
      elements: [{ type: "obj", desc: "a red fox sitting on the sand" }],
    },
  });
  const EXPANDED = parseMagicPromptCaption(RAW_CAPTION).caption;

  const buttonByText = (text) =>
    [...document.body.querySelectorAll("button")].find((b) => b.textContent.trim() === text);
  const generateButton = () => buttonByText("Generate");

  function setTextArea(element, value) {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    ).set;
    setter.call(element, value);
    element.dispatchEvent(new window.Event("input", { bubbles: true }));
  }

  async function enterPlainText(text) {
    // Switch the builder to its Plain text tab, then type the idea.
    await click(buttonByText("Plain text"));
    await act(async () => {
      setTextArea(document.body.querySelector('textarea[aria-label="Plain prompt"]'), text);
    });
  }

  it("auto-expands plain text to a JSON caption and never submits raw text", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const magicPrompt = vi.fn(async () => RAW_CAPTION);
    await render(
      baseContext({ createImageJob, magicPrompt, imageModels: [IDEOGRAM], models: [REFINE_READY] }),
    );

    await enterPlainText("a fox on a beach");
    await click(generateButton());
    await act(async () => {});

    expect(magicPrompt).toHaveBeenCalledTimes(1);
    const payload = createImageJob.mock.calls[0][0];
    // The engine receives the serialized JSON caption — NEVER the raw plain text.
    expect(payload.prompt).toBe(serializeCaption(EXPANDED));
    expect(payload.prompt).not.toBe("a fox on a beach");
    // Recipe records the expanded caption, the original idea, and the magic-prompt backend.
    expect(payload.advanced.structuredPrompt.caption).toEqual(EXPANDED);
    expect(payload.advanced.structuredPrompt.intent).toBe("a fox on a beach");
    expect(payload.advanced.structuredPrompt.magicPromptBackend).toBe(PROMPT_REFINE_MODEL_ID);
  });

  it("blocks generation (never raw text) when the prompt-refiner model is missing", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const magicPrompt = vi.fn(async () => RAW_CAPTION);
    await render(
      baseContext({
        createImageJob,
        magicPrompt,
        createModelDownloadJob: vi.fn(),
        imageModels: [IDEOGRAM],
        models: [{ id: PROMPT_REFINE_MODEL_ID, installState: "missing" }],
      }),
    );

    await enterPlainText("a fox on a beach");
    await click(generateButton());
    await act(async () => {});

    expect(magicPrompt).not.toHaveBeenCalled();
    expect(createImageJob).not.toHaveBeenCalled();
    // The block is surfaced (not silently dropped, never sent as raw text).
    const surfaced = [...document.body.querySelectorAll('[role="alert"]')].some((n) =>
      /download the prompt-refiner model/i.test(n.textContent),
    );
    expect(surfaced).toBe(true);
  });
});

describe("ImageStudio PiD decoder toggle (sc-7851)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // PiD-eligible image model: declares the qwenimage backbone via ui.pid (mirrors the
  // manifest + worker pid_backbone_for). The checkpoint rides the full catalog (`models`)
  // as its own installable entry (sc-7852), distinct from the image-model picker.
  const PID_QWEN = { ...Z_IMAGE, id: "qwen_image", name: "Qwen Image", ui: { pid: { checkpointId: "pid_qwenimage" } } };
  const PID_CKPT = (installState) => ({ id: "pid_qwenimage", type: "utility", installState });

  const openAdvanced = async () =>
    click(document.body.querySelector(".advanced-section-toggle"));
  const pidLabel = () =>
    [...document.body.querySelectorAll("label")].find((l) => l.textContent.includes("PiD decoder"));
  const generateButton = () =>
    [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate");

  it("shows the toggle (default off) when eligible AND the checkpoint is installed", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("installed")] }));
    await openAdvanced();
    await act(async () => {});
    const toggle = pidLabel();
    expect(toggle).toBeTruthy();
    // Non-commercial marker is surfaced on the toggle copy.
    expect(toggle.textContent).toContain("Non-Commercial");
    expect(toggle.querySelector('input[type="checkbox"]').checked).toBe(false);
  });

  it("hides the toggle when the checkpoint is present but not installed (fail-closed)", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("missing")] }));
    await openAdvanced();
    await act(async () => {});
    expect(pidLabel()).toBeFalsy();
  });

  it("hides the toggle when the checkpoint entry is absent from the catalog (today's pre-sc-7852 state)", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN] }));
    await openAdvanced();
    await act(async () => {});
    expect(pidLabel()).toBeFalsy();
  });

  it("hides the toggle for a non-eligible model even when a PiD checkpoint is installed", async () => {
    await render(baseContext({ imageModels: [Z_IMAGE], models: [Z_IMAGE, PID_CKPT("installed")] }));
    await openAdvanced();
    await act(async () => {});
    expect(pidLabel()).toBeFalsy();
  });

  it("emits advanced.usePid:true only when shown AND toggled on", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("installed")] }),
    );
    await openAdvanced();
    await act(async () => {});

    // Default off → no usePid in the payload.
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("usePid");

    // Toggle on → usePid:true rides advanced.
    await act(async () => pidLabel().querySelector('input[type="checkbox"]').click());
    await click(generateButton());
    expect(createImageJob.mock.calls[1][0].advanced.usePid).toBe(true);
  });

  const pidTargetSelect = () => document.body.querySelector(".pid-target-select select");

  it("reveals the 2K/4K output selector only when PiD is on (default 4K)", async () => {
    await render(baseContext({ imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("installed")] }));
    await openAdvanced();
    await act(async () => {});
    // Off → no selector.
    expect(pidTargetSelect()).toBeFalsy();
    // On → selector appears, defaulting to 4k.
    await act(async () => pidLabel().querySelector('input[type="checkbox"]').click());
    expect(pidTargetSelect()).toBeTruthy();
    expect(pidTargetSelect().value).toBe("4k");
  });

  it("emits advanced.pidTarget:'2k' only when 2K is picked (4K default omits it)", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await render(
      baseContext({ createImageJob, imageModels: [PID_QWEN], models: [PID_QWEN, PID_CKPT("installed")] }),
    );
    await openAdvanced();
    await act(async () => {});
    await act(async () => pidLabel().querySelector('input[type="checkbox"]').click());

    // Default 4K → usePid but no pidTarget.
    await click(generateButton());
    expect(createImageJob.mock.calls[0][0].advanced.usePid).toBe(true);
    expect(createImageJob.mock.calls[0][0].advanced).not.toHaveProperty("pidTarget");

    // Pick 2K → pidTarget:"2k" rides advanced.
    await act(async () => {
      const select = pidTargetSelect();
      select.value = "2k";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await click(generateButton());
    expect(createImageJob.mock.calls[1][0].advanced.pidTarget).toBe("2k");
  });
});

describe("ImageStudio Image-reference (img2img) tile (epic 8588, sc-8593/sc-10195)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // Krea-like img2img model: a `ui.img2img` toggle, plain text-to-image capability, no vision
  // captioner in context (baseContext leaves `imageDescribe` undefined → describe flow unavailable).
  const KREA_IMG2IMG = { ...Z_IMAGE, id: "krea_2_turbo", name: "Krea 2 Turbo", ui: { img2img: true } };
  const tileByText = (text) =>
    [...document.body.querySelectorAll("button")].find((b) => b.textContent.includes(text));

  it("hides the 'Image reference' tile for a plain t2i model without ui.img2img", async () => {
    // sc-10195: img2img is its own tile, gated purely on `ui.img2img`. A plain model with no captioner
    // shows neither the img2img nor the describe tile.
    await render(baseContext({ imageModels: [Z_IMAGE], models: [Z_IMAGE] }));
    expect(tileByText("Image reference")).toBeFalsy();
    expect(tileByText("Prompt from image")).toBeFalsy();
  });

  it("shows the 'Image reference' tile for a ui.img2img model even without the vision captioner", async () => {
    // img2img needs no captioner — the tile appears on the flag alone, decoupled from describe (sc-10195).
    await render(baseContext({ imageModels: [KREA_IMG2IMG], models: [KREA_IMG2IMG] }));
    expect(tileByText("Image reference")).toBeTruthy();
    // The describe tile stays hidden without a captioner, proving the two are independent now.
    expect(tileByText("Prompt from image")).toBeFalsy();
  });

  // Ideogram-like structured-prompt img2img model (epic 8588 A4.4, sc-10192): the free-text prompt-tools
  // strip is replaced by the JSON-caption builder, but the img2img "Image reference" tile must still
  // surface — reference-guided latent-init is orthogonal to how the prompt is authored.
  const IDEOGRAM_IMG2IMG = {
    ...Z_IMAGE,
    id: "ideogram_4",
    name: "Ideogram 4",
    family: "ideogram",
    structuredPrompt: true,
    ui: { img2img: true },
  };

  it("shows the 'Image reference' tile for a structured-prompt img2img model (Ideogram)", async () => {
    await render(baseContext({ imageModels: [IDEOGRAM_IMG2IMG], models: [IDEOGRAM_IMG2IMG] }));
    // The tile coexists with the structured caption builder…
    expect(tileByText("Image reference")).toBeTruthy();
    expect(document.body.querySelector(".structured-prompt-builder, .prompt-input-row.structured")).toBeTruthy();
    // …but the free-text-only tiles ("Prompt from image" / "Refine my prompt") stay out of the strip —
    // the caption builder owns image→caption + magic-expand, so this is the slim img2img-only strip.
    expect(tileByText("Refine my prompt")).toBeFalsy();
  });

  it("activating the structured img2img tile reveals the reference-guidance panel", async () => {
    await render(baseContext({ imageModels: [IDEOGRAM_IMG2IMG], models: [IDEOGRAM_IMG2IMG] }));
    await click(tileByText("Image reference"));
    // The img2img panel's hint distinguishes it from the caption builder's own reference→caption picker.
    expect(
      [...document.body.querySelectorAll(".prompt-tool-panel .structured-hint")].some((p) =>
        p.textContent.includes("guide the render (image-to-image)"),
      ),
    ).toBe(true);
  });
});

describe("ImageStudio strict-control panel (epic 8236, sc-8245)", () => {
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

  async function render(context) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <ImageStudio />
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  // A Fun-Union backbone advertising all three control modes (mirrors the manifest controlModes/
  // controlScale). The pose-only fixture omits canny/depth so the picker gates to a single tab.
  const FULL_CONTROL = {
    ...Z_IMAGE,
    id: "flux2_dev",
    name: "FLUX.2-dev",
    family: "flux2",
    ui: {
      poseLibrary: true,
      controlModes: ["pose", "canny", "depth"],
      controlScale: { label: "Control strength", default: 0.75, min: 0, max: 2, step: 0.05 },
    },
  };
  const POSE_ONLY = {
    ...Z_IMAGE,
    id: "pose_only",
    name: "Pose Only",
    family: "pose",
    ui: {
      poseLibrary: true,
      controlModes: ["pose"],
      controlScale: { label: "Control strength", default: 0.9, min: 0, max: 2, step: 0.05 },
    },
  };
  const NO_CONTROL = { ...Z_IMAGE, id: "plain", name: "Plain", family: "plain", ui: {} };

  const controlTabs = () =>
    [...document.body.querySelectorAll(".control-mode-tab")].map((b) => b.textContent.trim());
  const controlTabByLabel = (label) =>
    [...document.body.querySelectorAll(".control-mode-tab")].find((b) => b.textContent.trim() === label);
  // The structure-control panel is collapsed by default; expand it so the gated inner content
  // (mode tabs, control-image upload, slider) mounts before the assertions below.
  const expandControlPanel = async () => click(document.body.querySelector(".control-panel-toggle"));
  const generate = async () =>
    click([...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate"));

  it("gates the picker to the backbone's supported modes (all three)", async () => {
    await render(baseContext({ imageModels: [FULL_CONTROL] }));
    await expandControlPanel();
    expect(controlTabs()).toEqual(["Pose", "Canny", "Depth"]);
  });

  it("shows only the pose tab for a pose-only backbone", async () => {
    await render(baseContext({ imageModels: [POSE_ONLY] }));
    await expandControlPanel();
    expect(controlTabs()).toEqual(["Pose"]);
  });

  it("hides the panel entirely for a backbone with no control modes", async () => {
    await render(baseContext({ imageModels: [NO_CONTROL] }));
    expect(document.body.querySelector(".control-panel")).toBeNull();
  });

  it("re-gates and resets an unsupported mode when the backbone switches", async () => {
    await render(baseContext({ imageModels: [FULL_CONTROL, POSE_ONLY] }));
    await expandControlPanel();
    // Pick canny on the multi-mode backbone.
    await click(controlTabByLabel("Canny"));
    expect(controlTabByLabel("Canny").getAttribute("aria-pressed")).toBe("true");
    // Switch to the pose-only backbone → canny is gone, only pose remains, and pose is active.
    await act(async () => setSelect(field(container, "Model"), "pose_only"));
    await act(async () => {});
    expect(controlTabs()).toEqual(["Pose"]);
    expect(controlTabByLabel("Pose").getAttribute("aria-pressed")).toBe("true");
  });

  it("canny preprocess (derive) sends the control image as sourceAssetId + controlMode/controlScale", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const plate = { id: "ctrl-plate", projectId: "project_1", type: "image", displayName: "Plate", status: {} };
    await render(baseContext({ createImageJob, imageModels: [FULL_CONTROL], assets: [plate] }));
    await expandControlPanel();
    await click(controlTabByLabel("Canny"));
    // Open the control-image picker and double-click the asset to confirm it.
    await click(
      [...document.body.querySelectorAll(".asset-picker-head button")].find((b) => b.textContent === "Select image"),
    );
    const card = [...document.body.querySelectorAll(".asset-picker-card")].find((b) =>
      b.textContent.includes("Plate"),
    );
    await act(async () => card.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true })));

    await generate();
    const payload = createImageJob.mock.calls[0][0];
    // Derive mode → the asset rides as the source the worker auto-derives from; NOT advanced.controlImage.
    expect(payload.sourceAssetId).toBe("ctrl-plate");
    expect(payload.advanced.controlMode).toBe("canny");
    expect(payload.advanced).not.toHaveProperty("controlImage");
    expect(payload.advanced.controlScale).toBe(0.75);
  });

  it("use-as-is passthrough sends the control image as advanced.controlImage, not sourceAssetId", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    const plate = { id: "ctrl-plate", projectId: "project_1", type: "image", displayName: "Plate", status: {} };
    await render(baseContext({ createImageJob, imageModels: [FULL_CONTROL], assets: [plate] }));
    await expandControlPanel();
    await click(controlTabByLabel("Depth"));
    await click(
      [...document.body.querySelectorAll(".asset-picker-head button")].find((b) => b.textContent === "Select image"),
    );
    const card = [...document.body.querySelectorAll(".asset-picker-card")].find((b) =>
      b.textContent.includes("Plate"),
    );
    await act(async () => card.dispatchEvent(new window.MouseEvent("dblclick", { bubbles: true })));
    // Flip the preprocess toggle ON → use-as-is passthrough.
    const toggle = [...document.body.querySelectorAll(".control-image-section input[type='checkbox']")][0];
    await act(async () => toggle.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));

    await generate();
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.advanced.controlImage).toBe("ctrl-plate");
    expect(payload.sourceAssetId).toBeNull();
    expect(payload.advanced.controlMode).toBe("depth");
    expect(payload.advanced.controlScale).toBe(0.75);
  });
});
