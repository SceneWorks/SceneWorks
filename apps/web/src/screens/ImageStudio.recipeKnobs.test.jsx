// Every allow-listed knob a replayed recipe carries reaches a real Image Studio control
// (sc-15951, epic 15945).
//
// WHY THIS FILE EXISTS. The first cut of the dropped-workflow panel displayed ten knobs —
// `enhancePrompt`, `usePid`, `pidTarget`, `strength`, `textStyleGain`, `faceRestore`,
// `controlMode`, `controlScale`, `poses`, `phases` — as ordinary "Settings" rows beside
// "Steps — 28", while the injection effect named none of them. The user was told a recipe replayed
// faithfully when it had not, which is the exact failure epic 15945 exists to prevent.
//
// So every assertion here reads the RENDERED CONTROL, not the recipe and not the adapter. A test
// that asserted `rawAdapterSettings` would have passed against the broken build: the values were
// always in the recipe: nothing read them.
import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mountRoot, unmountRoot } from "../testUtils/dom.js";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: vi.fn(async () => ({})) };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";
import { recipeFromWorkflowShare } from "../workflowShare.js";

// One model that advertises every lane the knobs belong to, so a single mount can observe all of
// them: img2img, Krea's text-style gain, strict control (pose/canny/depth), FLUX-style prompt
// enhancement, a PiD backbone, and the acceleration LoRA compat that gates multi-phase denoise.
const EVERY_LANE = {
  id: "krea_2_raw",
  name: "Krea 2 Raw",
  type: "image",
  family: "krea2",
  capabilities: ["text_to_image"],
  installState: "installed",
  defaults: { resolution: "1024x1024" },
  limits: {
    resolutions: ["1024x1024", "1536x1024"],
    samplers: ["default", "euler"],
    schedulers: ["default", "shift"],
  },
  loraCompatibility: { types: ["acceleration"] },
  ui: {
    img2img: true,
    textStyleGain: { default: 1.0, min: 0.25, max: 1.75, step: 0.05, label: "Text style" },
    controlModes: ["pose", "canny", "depth"],
    controlScale: { default: 0.7, min: 0, max: 2, step: 0.05 },
    promptEnhance: true,
    pid: { checkpointId: "pid_qwenimage" },
  },
};

// A SECOND installed model, so a recipe can move the studio off whatever it mounted on — which is
// what arms the model-change reset the control knobs have to survive.
const OTHER_MODEL = {
  ...EVERY_LANE,
  id: "z_image_turbo",
  name: "Z Image Turbo",
  family: "z-image",
  loraCompatibility: {},
  ui: {
    ...EVERY_LANE.ui,
    controlScale: { default: 0.25, min: 0, max: 2, step: 0.05 },
    textStyleGain: { default: 1.0, min: 0.25, max: 1.75, step: 0.05, label: "Text style" },
  },
};

const FLUX2_DEV = {
  id: "flux2_dev",
  name: "FLUX.2 Dev",
  type: "image",
  family: "flux2-dev",
  capabilities: ["text_to_image", "edit_image", "character_image"],
  installState: "installed",
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024"] },
  ui: { promptEnhance: true },
};

const FLUX3 = {
  id: "flux3_klein", name: "FLUX 3 Klein", type: "image", family: "flux3",
  capabilities: ["text_to_image", "edit_image", "character_image"], installState: "installed",
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024", "1536x1024"], minDimension: 64,
    maxDimension: 4096, requiresDimensionsMultipleOf: 16 },
  ui: { multiReference: true, hideReferenceStrength: true },
};
const FLUX3_REFERENCE = {
  id: "flux3-reference", type: "image", projectId: "project_1", displayName: "83 by 67 reference",
  width: 83, height: 67, status: {}, file: { path: "assets/images/reference.png", mimeType: "image/png" },
};

function flux3Recipe(width, height) {
  return {
    mode: "edit_image", model: FLUX3.id, prompt: "a lighthouse in heavy fog", loras: [],
    normalizedSettings: { width: 80, height: 64, count: 1 },
    rawAdapterSettings: {
      referenceAssetIds: [FLUX3_REFERENCE.id],
      flux3ReferenceEditing: { output: { requestedWidth: width, requestedHeight: height,
        canvas: { width: 83, height: 67 }, actualSize: { width: 80, height: 64 } } },
    },
  };
}

const PID_CHECKPOINT = { id: "pid_qwenimage", name: "PiD decoder", installState: "installed" };

const ACCEL_LORA = {
  id: "krea2_turbo_accel",
  name: "Krea 2 Turbo accelerator",
  family: "krea2",
  scope: "global",
  installState: "installed",
  role: "accelerator",
  type: "acceleration",
};
const STYLE_LORA = {
  id: "lora_grain",
  name: "Film grain",
  family: "krea2",
  scope: "global",
  installState: "installed",
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(),
    createPreset: vi.fn(),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [EVERY_LANE, OTHER_MODEL, FLUX2_DEV, FLUX3],
    models: [EVERY_LANE, OTHER_MODEL, FLUX2_DEV, FLUX3, PID_CHECKPOINT],
    importAsset: vi.fn(),
    latestImageAssets: [],
    recentImageAssets: [],
    studioLaunch: null,
    imageLocalJobs: [],
    loras: [ACCEL_LORA, STYLE_LORA],
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

// The advanced disclosures render their children only when open, and their open state is part of
// the persisted studio snapshot — so the snapshot is how the test opens them, without simulating
// three clicks that are not what is under test.
function seedOpenDisclosures(extra = {}) {
  window.localStorage.setItem(
    "sceneworks-studio-image-project_1",
    JSON.stringify({ advancedOpen: true, multiPhaseOpen: true, ...extra }),
  );
}

function launch(recipe, overrides = {}) {
  return {
    id: `launch_${Math.random().toString(16).slice(2)}`,
    view: "Image",
    recipe,
    replaySeed: null,
    ...overrides,
  };
}

// The knobs under test, as one recipe. Values are deliberately non-default so a control still on
// its model default cannot pass by accident.
function everyKnobRecipe(overrides = {}) {
  return {
    mode: "text_to_image",
    model: EVERY_LANE.id,
    prompt: "a lighthouse in heavy fog",
    negativePrompt: "",
    loras: [],
    normalizedSettings: { width: 1024, height: 1024, count: 1 },
    rawAdapterSettings: {
      steps: 28,
      guidanceScale: 4.5,
      enhancePrompt: true,
      usePid: true,
      pidTarget: "2k",
      strength: 0.35,
      textStyleGain: 1.45,
      faceRestore: true,
      controlMode: "depth",
      controlScale: 0.85,
      ...overrides,
    },
  };
}

describe("a replayed recipe reaches every allow-listed control", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    seedOpenDisclosures();
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

  const checkbox = (labelText) =>
    [...document.body.querySelectorAll("label")]
      .find((node) => node.textContent.includes(labelText))
      ?.querySelector('input[type="checkbox"]');
  const control = (labelText, selector = "input, select") =>
    [...document.body.querySelectorAll("label")]
      .find((node) => node.textContent.trim().startsWith(labelText))
      ?.querySelector(selector);
  const buttonByText = (text) =>
    [...document.body.querySelectorAll("button")].find((node) =>
      node.textContent.includes(text),
    );
  const click = async (node) =>
    act(async () => node.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  // The strict-control disclosure keeps its own local open state (collapsed by default), so the
  // panel has to be expanded before its controls are in the DOM at all.
  const openStructureControl = () => click(buttonByText("Structure control"));
  const activeControlModeLabel = () =>
    document.body.querySelector(".control-mode-tab.active")?.textContent.trim();
  const modelSelect = () =>
    [...document.body.querySelectorAll("select")].find((node) =>
      [...node.options].some((option) => option.value === EVERY_LANE.id),
    );
  const phaseTitles = () =>
    [...document.body.querySelectorAll(".multiphase-phase-title")].map((node) =>
      node.textContent.trim(),
    );
  const multiPhaseToggle = () =>
    document.body.querySelector(".multiphase-section input[type='checkbox']") ??
    [...document.body.querySelectorAll("label")]
      .find((node) => node.textContent.includes("Multi-phase"))
      ?.querySelector('input[type="checkbox"]');
  // The studio writes its live state through to this snapshot, which is where the phase list can
  // be read in the editor's own id-keyed shape — the remapped structure, not a rendering of it.
  const persistedStudioSettings = () =>
    JSON.parse(window.localStorage.getItem("sceneworks-studio-image-project_1") ?? "{}");

  const editControl = async (node, value) => act(async () => {
    if (node.tagName === "INPUT") {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(node, value);
      node.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      node.value = value;
      node.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });

  it.each(["flux3ReferenceEditing", "flux3Layout"])("restores actual worker-shaped ordered references from %s without a raw referenceAssetIds field", async (factKey) => {
    const other = { ...FLUX3_REFERENCE, id: "other-reference", displayName: "Other reference" };
    const orderedAssets = [FLUX3_REFERENCE.id, other.id, FLUX3_REFERENCE.id];
    const instanceIds = ["first-instance", "middle-instance", "last-instance"];
    const layout = { version: 1, boxes: [{ id: "worker-row", desc: "remove second occurrence", from: "ref_image_2",
      reference_id: "last-instance", src_bbox: [0, -0.123456789012345, 1, 1], tgt_bbox: null }] };
    const recipe = flux3Recipe(null, null);
    delete recipe.rawAdapterSettings.referenceAssetIds;
    recipe.rawAdapterSettings.flux3_layout = layout;
    const references = orderedAssets.map((assetId, ordinal) => ({ ordinal, assetId, orientedSize: [83, 67], contentSha256: "a".repeat(64) }));
    if (factKey === "flux3ReferenceEditing") {
      recipe.rawAdapterSettings.flux3ReferenceEditing.references = references;
      recipe.rawAdapterSettings.flux3_reference_ids = instanceIds;
    } else {
      delete recipe.rawAdapterSettings.flux3ReferenceEditing;
      recipe.rawAdapterSettings.flux3Layout = { references, referenceInstanceIds: instanceIds, authoring: layout };
    }
    expect(recipe.rawAdapterSettings).not.toHaveProperty("referenceAssetIds");
    seedOpenDisclosures({ model: FLUX3.id, mode: "edit_image", referenceAssetIds: ["stale-picker-asset"],
      flux3ReferenceSlots: [{ assetId: "stale-picker-asset", id: "stale-instance" }] });
    const createImageJob = vi.fn(async () => ({ id: "worker-replay" }));
    await render(baseContext({ assets: [FLUX3_REFERENCE, other], createImageJob, studioLaunch: launch(recipe) }));
    await click(document.querySelector('[aria-label="Select worker-row"]'));
    expect(document.querySelector('[aria-label="Source reference"]').value).toBe("last-instance");
    expect(persistedStudioSettings().referenceAssetIds).toEqual(orderedAssets);
    expect(persistedStudioSettings().flux3Layout).toEqual(layout);
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls[0][0].referenceAssetIds).toEqual(orderedAssets);
    expect(createImageJob.mock.calls[0][0].advanced.flux3_reference_ids).toEqual(instanceIds);
    expect(createImageJob.mock.calls[0][0].advanced.flux3_layout).toEqual(layout);
    expect(recipe.rawAdapterSettings).not.toHaveProperty("referenceAssetIds");
  });

  it.each(["approved selection", "character switch", "approved fallback"])("binds an ordinal-only source to its OLD attachment before %s", async (change) => {
    const other = { ...FLUX3_REFERENCE, id: "replacement-reference", displayName: "Replacement reference" };
    const first = { id: "first-character", name: "First character", looks: [], approvedReferences: [
      { assetId: FLUX3_REFERENCE.id, asset: FLUX3_REFERENCE }, { assetId: other.id, asset: other },
    ] };
    const second = { id: "second-character", name: "Second character", looks: [], approvedReferences: [{ assetId: other.id, asset: other }] };
    const box = { id: "ordinal-source", desc: "remove vase", from: "ref_image_0", src_bbox: [0, 0.123456789012345, 1, 1], tgt_bbox: null };
    const recipe = flux3Recipe(null, null);
    recipe.mode = "character_image";
    recipe.normalizedSettings.characterId = first.id;
    delete recipe.rawAdapterSettings.referenceAssetIds;
    recipe.rawAdapterSettings.flux3_layout = { version: 1, boxes: [box] };
    recipe.rawAdapterSettings.flux3_reference_ids = ["original-instance"];
    recipe.rawAdapterSettings.flux3ReferenceEditing.references = [{ ordinal: 0, assetId: FLUX3_REFERENCE.id,
      orientedSize: [83, 67], contentSha256: "b".repeat(64) }];
    const createImageJob = vi.fn(async () => ({ id: "replacement-job" }));
    const context = baseContext({ assets: [FLUX3_REFERENCE, other], characters: [first, second], createImageJob, studioLaunch: launch(recipe) });
    await render(context);
    await click(document.querySelector('[aria-label="Select ordinal-source"]'));
    expect(persistedStudioSettings().flux3Layout.boxes[0]).toEqual(box);
    expect(document.querySelector('[aria-label="Source reference"]').value).toBe("original-instance");
    if (change === "approved selection") {
      await click(document.querySelector('[aria-label="Use Replacement reference as reference"]'));
    } else if (change === "character switch") {
      await editControl([...document.querySelectorAll("select")].find((node) =>
        [...node.options].some(({ value }) => value === second.id)), second.id);
    } else {
      await render({ ...context, characters: [{ ...first, approvedReferences: [{ assetId: other.id, asset: other }] }, second] });
    }
    expect(document.querySelector('[aria-label="Source reference"]').value).toBe("");
    expect(document.querySelector('[role="alert"]').textContent).toMatch(/source attachment is missing/);
    const expected = { ...box, reference_id: "original-instance" };
    expect(persistedStudioSettings().flux3Layout.boxes[0]).toEqual(expected);
    await click(buttonByText("Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.referenceAssetId).toBe(other.id);
    expect(payload.advanced.flux3_reference_ids).toHaveLength(1);
    expect(payload.advanced.flux3_reference_ids[0]).not.toBe("original-instance");
    expect(payload.advanced.flux3_layout.boxes[0]).toEqual(expected);
  });

  it("replays the original layout document and duplicate attachment identities through reorder and submit", async () => {
    const layout = { version: 1, boxes: [
      { id: "precise", desc: '猫 "vase"', bbox: [0, -0.123456789012345, 1, 0.987654321098765] },
      { id: "remove", desc: "remove second instance", from: "ref_image_1", reference_id: "slot-second", src_bbox: [0, 0.111111111111111, 1, 1], tgt_bbox: null },
    ] };
    const recipe = flux3Recipe(null, null);
    recipe.rawAdapterSettings.referenceAssetIds = [FLUX3_REFERENCE.id, FLUX3_REFERENCE.id];
    recipe.rawAdapterSettings.flux3_layout = layout;
    recipe.rawAdapterSettings.flux3_reference_ids = ["slot-first", "slot-second"];
    const createImageJob = vi.fn(async () => ({ id: "layout-job" }));
    await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob, studioLaunch: launch(recipe) }));
    await click(document.querySelector('[aria-label="Select remove"]'));
    expect(document.querySelector('[aria-label="Source reference"]').value).toBe("slot-second");
    expect(document.querySelector('[aria-label="Target top"]')).toBeNull();
    await click(document.querySelector('[aria-label="Move Image 2 earlier"]'));
    expect(document.querySelector('[aria-label="Source reference"]').value).toBe("slot-second");
    await click(buttonByText("Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.referenceAssetIds).toEqual([FLUX3_REFERENCE.id, FLUX3_REFERENCE.id]);
    expect(payload.advanced.flux3_reference_ids).toEqual(["slot-second", "slot-first"]);
    expect(payload.advanced.flux3_layout).toEqual(layout);
    expect(persistedStudioSettings().flux3Layout).toEqual(layout);
    expect(persistedStudioSettings().flux3ReferenceSlots.map(({ id }) => id)).toEqual(["slot-second", "slot-first"]);
    // Remove one occurrence, not every attachment with this asset id.
    await click(document.querySelector('[aria-label="Remove Image 2"]'));
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls.at(-1)[0].referenceAssetIds).toEqual([FLUX3_REFERENCE.id]);
    expect(createImageJob.mock.calls.at(-1)[0].advanced.flux3_reference_ids).toEqual(["slot-second"]);
    expect(createImageJob.mock.calls.at(-1)[0].advanced.flux3_layout).toEqual(layout);
  });

  it("binds an ordinal-only native source before explicit reorder/removal without retargeting or rounding", async () => {
    const box = { id: "native-row", desc: "vase", bbox: null, from: "ref_image_1", src_bbox: [0, 0.123456789012345, 1, 1], tgt_bbox: null };
    const recipe = flux3Recipe(null, null);
    recipe.rawAdapterSettings.referenceAssetIds = [FLUX3_REFERENCE.id, FLUX3_REFERENCE.id];
    recipe.rawAdapterSettings.flux3_layout = { version: 1, boxes: [box] };
    recipe.rawAdapterSettings.flux3_reference_ids = ["first", "second"];
    const createImageJob = vi.fn(async () => ({ id: "native-layout" }));
    await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob, studioLaunch: launch(recipe) }));
    await click(document.querySelector('[aria-label="Select native-row"]'));
    expect(persistedStudioSettings().flux3Layout.boxes[0]).toEqual(box);
    await click(document.querySelector('[aria-label="Move Image 2 earlier"]'));
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls[0][0].advanced.flux3_layout.boxes[0]).toEqual({ ...box, reference_id: "second" });
    await click(document.querySelector('[aria-label="Remove Image 1"]'));
    expect(document.querySelector('[aria-label="Source reference"]').value).toBe("");
    expect(document.querySelector('[role="alert"]').textContent).toMatch(/source attachment is missing/);
    expect(persistedStudioSettings().flux3Layout.boxes[0]).toEqual({ ...box, reference_id: "second" });
  });

  it("preserves an absent native removal target until explicitly moved, then edits target geometry", async () => {
    const box = { id: "absent-target", desc: "vase", from: "ref_image_0", reference_id: "native-slot", src_bbox: [0, 0, 1, 1] };
    const recipe = flux3Recipe(null, null);
    recipe.rawAdapterSettings.flux3_layout = { version: 1, boxes: [box] };
    recipe.rawAdapterSettings.flux3_reference_ids = ["native-slot"];
    const createImageJob = vi.fn(async () => ({ id: "moved-native-removal" }));
    await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob, studioLaunch: launch(recipe) }));
    await click(document.querySelector('[aria-label="Select absent-target"]'));
    expect(persistedStudioSettings().flux3Layout.boxes[0]).toEqual(box);
    expect(document.querySelector('[aria-label="Select absent-target"]').textContent).toContain("Removal");
    expect(document.querySelector('[aria-label="Target left"]')).toBeNull();
    expect(buttonByText("Move object to target").disabled).toBe(false);
    expect(buttonByText("Remove object from output").disabled).toBe(true);
    await click(buttonByText("Move object to target"));
    await editControl(document.querySelector('[aria-label="Target left"]'), "0.123456789012345");
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls[0][0].advanced.flux3_layout.boxes[0]).toEqual({ ...box, tgt_bbox: [0.25, 0.123456789012345, 0.75, 0.75] });
  });

  it("authors labels, placements, moves and null-target removals in the actual Studio", async () => {
    const createImageJob = vi.fn(async () => ({ id: "authored-layout" }));
    await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob,
      studioLaunch: launch(flux3Recipe(null, null)) }));
    await click(buttonByText("Add placement box"));
    await editControl(document.querySelector('[aria-label="Box label"]'), "a red vase");
    await editControl(document.querySelector('[aria-label="Target left"]'), "0.123456789012345");
    await click(buttonByText("Add source edit"));
    await editControl(document.querySelector('[aria-label="Box label"]'), "move blue vase");
    await editControl(document.querySelector('[aria-label="Source top"]'), "0");
    await editControl(document.querySelector('[aria-label="Target right"]'), "1");
    await click(buttonByText("Remove object from output"));
    expect(document.querySelector('[aria-label="Target right"]')).toBeNull();
    await click(buttonByText("Move object to target"));
    expect(document.querySelector('[aria-label="Target right"]')).not.toBeNull();
    await click(buttonByText("Remove object from output"));
    await click(buttonByText("Generate"));
    const { advanced } = createImageJob.mock.calls[0][0];
    expect(advanced.flux3_layout.boxes).toMatchObject([
      { desc: "a red vase", bbox: [0.25, 0.123456789012345, 0.75, 0.75] },
      { desc: "move blue vase", from: "ref_image_0", reference_id: advanced.flux3_reference_ids[0], src_bbox: [0, 0.25, 0.75, 0.75], tgt_bbox: null },
    ]);
    expect(advanced).not.toHaveProperty("strength");
    expect(advanced).not.toHaveProperty("controlScale");
  });

  it("restores saved layout and slot ids on mount but hides and omits them on other models", async () => {
    const layout = { version: 1, boxes: [{ id: "saved", desc: "saved label", bbox: [0, 0, 1, 1] }] };
    seedOpenDisclosures({ model: FLUX3.id, mode: "edit_image", prompt: "vase", referenceAssetIds: [FLUX3_REFERENCE.id],
      flux3Layout: layout, flux3ReferenceSlots: [{ assetId: FLUX3_REFERENCE.id, id: "saved-slot" }] });
    const createImageJob = vi.fn(async () => ({ id: "saved-layout" }));
    await render(baseContext({ assets: [FLUX3_REFERENCE], selectedAsset: FLUX3_REFERENCE, createImageJob }));
    await click(document.querySelector('[aria-label="Select saved"]'));
    expect(document.querySelector('[aria-label="Box label"]').value).toBe("saved label");
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls[0][0].advanced.flux3_reference_ids).toEqual(["saved-slot"]);
    await editControl([...document.querySelectorAll("select")].find((node) =>
      [...node.options].some(({ value }) => value === FLUX2_DEV.id)), FLUX2_DEV.id);
    expect(document.querySelector('[aria-label="FLUX 3 layout editor"]')).toBeNull();
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls.at(-1)[0].advanced).not.toHaveProperty("flux3_layout");
  });

  it.each([[null, null], [512, 768], [512, null], [null, 768], [16, 48]])(
    "restores FLUX3 requested axes %s / %s through mounted submission, not actual image geometry",
    async (width, height) => {
      seedOpenDisclosures({ widthOverride: "2048", heightOverride: "1536" });
      const createImageJob = vi.fn(async () => ({ id: "job-flux3" }));
      await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob,
        studioLaunch: launch(flux3Recipe(width, height)) }));
      expect(control("Width override").value).toBe(width == null ? "" : String(width));
      expect(control("Height override").value).toBe(height == null ? "" : String(height));
      expect(control("Width override").min).toBe("16");
      expect(buttonByText("Generate").disabled).toBe(false);
      await click(buttonByText("Generate"));
      const payload = createImageJob.mock.calls[0][0];
      expect(payload.referenceAssetIds).toEqual([FLUX3_REFERENCE.id]);
      if (width == null) expect(payload).not.toHaveProperty("width");
      else expect(payload.width).toBe(width);
      if (height == null) expect(payload).not.toHaveProperty("height");
      else expect(payload.height).toBe(height);
    },
  );

  it.each([[undefined, undefined], [512, 768], [512, undefined], [undefined, 768], [16, 48]])(
    "submits imported FLUX3 reference workflow axes %s / %s without forging worker facts",
    async (width, height) => {
      const recipe = recipeFromWorkflowShare({
        sceneworksWorkflow: "image", mode: "edit_image", model: FLUX3.id,
        prompt: "a lighthouse", width, height, inputs: [{ kind: "reference" }],
      }, { model: { state: "resolved", catalogId: FLUX3.id }, loras: [], styles: [] },
      { referenceAssetIds: [FLUX3_REFERENCE.id] });
      expect(recipe.rawAdapterSettings).not.toHaveProperty("flux3ReferenceEditing");
      const createImageJob = vi.fn(async () => ({ id: "job-shared-flux3" }));
      await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob, studioLaunch: launch(recipe) }));
      expect(buttonByText("Generate").disabled).toBe(false);
      await click(buttonByText("Generate"));
      const payload = createImageJob.mock.calls[0][0];
      expect(payload.advanced).not.toHaveProperty("flux3ReferenceEditing");
      if (width == null) expect(payload).not.toHaveProperty("width");
      else expect(payload.width).toBe(width);
      if (height == null) expect(payload).not.toHaveProperty("height");
      else expect(payload.height).toBe(height);
    },
  );

  it.each([[undefined, undefined], [512, 768], [512, undefined], [undefined, 48], [16, 48]])(
    "submits source-only imported FLUX3 workflow axes %s / %s through the existing launch binding",
    async (width, height) => {
      // The shipped FLUX3 UI uses the single-source edit picker, not multiReference.
      const model = { ...FLUX3, ui: { hideReferenceStrength: true } };
      const sourceAssetId = FLUX3_REFERENCE.id;
      const recipe = recipeFromWorkflowShare({
        sceneworksWorkflow: "image", mode: "edit_image", model: model.id,
        prompt: "a lighthouse", width, height, inputs: [{ kind: "source", count: 1 }],
      }, { model: { state: "resolved", catalogId: model.id }, loras: [], styles: [] },
      { sourceAssetId });
      expect(recipe.rawAdapterSettings).not.toHaveProperty("flux3ReferenceEditing");
      expect(recipe.rawAdapterSettings).not.toHaveProperty("referenceAssetIds");
      const createImageJob = vi.fn(async () => ({ id: "job-source-workflow" }));
      await render(baseContext({ imageModels: [model], models: [model], assets: [FLUX3_REFERENCE],
        createImageJob, studioLaunch: launch(recipe, { sourceAssetId }) }));
      expect(buttonByText("Generate").disabled).toBe(false);
      await click(buttonByText("Generate"));
      const payload = createImageJob.mock.calls[0][0];
      expect(payload.sourceAssetId).toBe(sourceAssetId);
      if (width == null) expect(payload).not.toHaveProperty("width");
      else expect(payload.width).toBe(width);
      if (height == null) expect(payload).not.toHaveProperty("height");
      else expect(payload.height).toBe(height);
      expect(payload.advanced).not.toHaveProperty("flux3ReferenceEditing");
    },
  );

  it.each(["Width override", "Height override", "Aspect"])(
    "disarms automatic reference sizing on a subsequent %s edit", async (field) => {
      const createImageJob = vi.fn(async () => ({ id: "job-edited-flux3" }));
      await render(baseContext({ assets: [FLUX3_REFERENCE], createImageJob,
        studioLaunch: launch(flux3Recipe(null, null)) }));
      await editControl(control(field), field === "Aspect" ? "1536x1024" : "512");
      await click(buttonByText("Generate"));
      expect(createImageJob.mock.calls[0][0]).toMatchObject({
        width: field === "Width override" ? 512 : field === "Aspect" ? 1536 : 1024,
        height: field === "Height override" ? 512 : 1024,
      });
      expect(control("Width override").placeholder).not.toContain("Auto");
      expect(control("Height override").placeholder).not.toContain("Auto");
    },
  );

  it.each(["edit_image", "character_image"])(
    "uses optional small reference geometry with the emitted %s single-reference source", async (mode) => {
      const model = { ...FLUX3, ui: { hideReferenceStrength: true } };
      const character = { id: "flux3-character", name: "Reference character", looks: [],
        approvedReferences: [{ assetId: FLUX3_REFERENCE.id, asset: FLUX3_REFERENCE }] };
      const recipe = flux3Recipe(null, 48);
      recipe.mode = mode;
      delete recipe.rawAdapterSettings.referenceAssetIds;
      recipe.rawAdapterSettings.referenceAssetId = FLUX3_REFERENCE.id;
      recipe.normalizedSettings.characterId = character.id;
      recipe.normalizedSettings.sourceAssetId = FLUX3_REFERENCE.id;
      const createImageJob = vi.fn(async () => ({ id: "job-source-flux3" }));
      await render(baseContext({ imageModels: [model], models: [model], assets: [FLUX3_REFERENCE],
        characters: [character], createImageJob, studioLaunch: launch(recipe) }));
      expect(buttonByText("Generate").disabled).toBe(false);
      expect(control("Width override").min).toBe("16");
      await click(buttonByText("Generate"));
      const payload = createImageJob.mock.calls[0][0];
      expect(payload).not.toHaveProperty("width");
      expect(payload.height).toBe(48);
      expect(payload[mode === "edit_image" ? "sourceAssetId" : "referenceAssetId"]).toBe(FLUX3_REFERENCE.id);
    },
  );

  it("keeps authored omitted axes while the model catalog is still loading", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-late-catalog" }));
    const context = baseContext({ imageModels: [], models: [], assets: [FLUX3_REFERENCE], createImageJob,
      studioLaunch: launch(flux3Recipe(null, 48)) });
    await render(context);
    await render({ ...context, imageModels: [FLUX3], models: [FLUX3] });
    expect(buttonByText("Generate").disabled).toBe(false);
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls[0][0]).not.toHaveProperty("width");
    expect(createImageJob.mock.calls[0][0].height).toBe(48);
  });

  it("drops automatic sizing on a same-model non-recipe launch", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-fresh-flux3" }));
    const context = baseContext({ assets: [FLUX3_REFERENCE], createImageJob,
      studioLaunch: launch(flux3Recipe(null, null)) });
    await render(context);
    await render({ ...context, studioLaunch: { id: "fresh-intent", view: "Image", mode: "edit_image" } });
    await click(buttonByText("Generate"));
    expect(createImageJob.mock.calls[0][0]).toMatchObject({ width: 1024, height: 1024 });
  });

  it("clears automatic sizing at a later ordinary recipe and retains the plain T2I floor", async () => {
    const context = baseContext({ assets: [FLUX3_REFERENCE],
      studioLaunch: launch(flux3Recipe(null, null)) });
    await render(context);
    const recipe = { ...flux3Recipe(16, 48), mode: "text_to_image" };
    await render({ ...context, studioLaunch: launch(recipe) });
    expect(control("Width override").min).toBe("64");
    expect(buttonByText("Generate").disabled).toBe(true);
    await render({ ...context, studioLaunch: launch(everyKnobRecipe()) });
    expect(control("Width override").value).toBe("");
    expect(control("Height override").value).toBe("");
    expect(control("Width override").min).toBe("256");
  });

  it("sets the PiD decoder and its output tier without exposing a forged Krea enhancement flag", async () => {
    await render(baseContext({ studioLaunch: launch(everyKnobRecipe()) }));

    expect(checkbox("Enhance prompt")).toBeUndefined();
    expect(checkbox("PiD decoder").checked).toBe(true);
    expect(document.body.querySelector(".pid-target-select select").value).toBe("2k");
  });

  it("restores prompt enhancement only onto a route that can execute it", async () => {
    const recipe = everyKnobRecipe();
    recipe.model = FLUX2_DEV.id;
    recipe.rawAdapterSettings = { enhancePrompt: true };
    await render(baseContext({ studioLaunch: launch(recipe) }));

    expect(checkbox("Enhance prompt").checked).toBe(true);
  });

  it("clears replayed enhancement on Candle character routes and cannot submit it with a reference", async () => {
    const reference = {
      id: "asset_flux_character",
      type: "image",
      projectId: "project_1",
      displayName: "Flux character",
      status: {},
      file: { path: "assets/images/flux-character.png", mimeType: "image/png" },
    };
    const character = {
      id: "character_flux",
      name: "Flux character",
      looks: [],
      approvedReferences: [{ assetId: reference.id, asset: reference }],
    };
    const recipe = everyKnobRecipe();
    recipe.mode = "character_image";
    recipe.model = FLUX2_DEV.id;
    recipe.normalizedSettings = {
      ...recipe.normalizedSettings,
      characterId: character.id,
    };
    recipe.rawAdapterSettings = {
      enhancePrompt: true,
      referenceAssetId: reference.id,
    };
    const createImageJob = vi.fn(async () => ({ id: "job-flux-character" }));
    await render(
      baseContext({
        assets: [reference],
        characters: [character],
        createImageJob,
        studioLaunch: launch(recipe),
      }),
    );

    expect(checkbox("Enhance prompt")).toBeUndefined();
    expect(persistedStudioSettings().enhancePrompt).toBe(false);
    await click(buttonByText("Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.mode).toBe("character_image");
    expect(payload.referenceAssetId).toBe(reference.id);
    expect(payload.advanced).not.toHaveProperty("enhancePrompt");
  });

  it("leaves the boolean decoders OFF when the recipe did not record them", async () => {
    // `buildImageJobAdvanced` emits these only when they are on, so an absent key means the run had
    // them off — inheriting the user's own sticky toggle would replay someone else's recipe with a
    // decoder they never used.
    seedOpenDisclosures({ enhancePrompt: true, usePid: true, pidTarget: "2k" });
    const recipe = everyKnobRecipe();
    delete recipe.rawAdapterSettings.enhancePrompt;
    delete recipe.rawAdapterSettings.usePid;
    delete recipe.rawAdapterSettings.pidTarget;
    await render(baseContext({ studioLaunch: launch(recipe) }));

    expect(checkbox("Enhance prompt")).toBeUndefined();
    expect(checkbox("PiD decoder").checked).toBe(false);
  });

  it("sets the Krea text-style gain", async () => {
    await render(baseContext({ studioLaunch: launch(everyKnobRecipe()) }));

    expect(control("Text style").value).toBe("1.45");
  });

  it("sets the control type and control strength", async () => {
    await render(baseContext({ studioLaunch: launch(everyKnobRecipe()) }));
    await openStructureControl();

    expect(activeControlModeLabel()).toBe("Depth");
    expect(document.body.querySelector(".control-scale input").value).toBe("0.85");
  });

  // THE ORDERING PROOF. `setModel` arms the model-change reset (`ImageStudio.jsx`), which writes
  // `setControlMode` / `setControlScale` / `setTextStyleGain` from the NEW model's declared
  // defaults — on the commit AFTER this effect ran. So a recipe that moves the studio to a
  // DIFFERENT model is the only shape that can catch that reset clobbering the replay, and it is
  // the shape "Use this workflow" takes almost every time.
  //
  // The two models declare different `controlScale` defaults (0.25 → 0.7), so a reset that won
  // would leave a value that is neither the recipe's nor a coincidence.
  it("keeps the recipe's control knobs when the launch also changes the model", async () => {
    seedOpenDisclosures({ model: OTHER_MODEL.id, controlMode: "pose", controlScale: 0.25 });
    const recipe = everyKnobRecipe();
    expect(recipe.model).toBe(EVERY_LANE.id);
    expect(EVERY_LANE.ui.controlScale.default).not.toBe(recipe.rawAdapterSettings.controlScale);

    await render(baseContext({ studioLaunch: launch(recipe) }));
    await openStructureControl();

    expect(modelSelect().value).toBe(EVERY_LANE.id);
    expect(activeControlModeLabel()).toBe("Depth");
    // 0.7 here would mean the model-change reset landed after the recipe and won.
    expect(document.body.querySelector(".control-scale input").value).toBe("0.85");
    expect(control("Text style").value).toBe("1.45");
  });

  it("sets img2img strength, which the reference slider shows once a reference is picked", async () => {
    const asset = {
      id: "asset_ref",
      type: "image",
      projectId: "project_1",
      displayName: "A lighthouse",
      status: {},
      file: { path: "assets/images/a.png", mimeType: "image/png" },
    };
    await render(baseContext({ assets: [asset], studioLaunch: launch(everyKnobRecipe()) }));

    // The slider renders only once there is a reference to be strong RELATIVE TO, so picking one
    // is the user step this replay cannot perform for them. The point of the assertion is that the
    // strength waiting behind it is the recipe's, not the 0.5 default.
    await click(buttonByText("Image reference"));
    await click(buttonByText("Select reference image"));
    await click(document.body.querySelector(".asset-picker-card"));
    await click(buttonByText("Use Selection"));

    expect(document.body.querySelector(".img2img-strength input").value).toBe("0.35");
  });

  it("sets face restoration on the pose panel", async () => {
    const reference = {
      id: "asset_face",
      type: "image",
      projectId: "project_1",
      displayName: "Mira",
      status: {},
      file: { path: "assets/images/mira.png", mimeType: "image/png" },
    };
    const character = {
      id: "char_1",
      name: "Mira",
      looks: [],
      approvedReferences: [{ assetId: reference.id, asset: reference }],
    };
    const poseModel = {
      ...EVERY_LANE,
      capabilities: ["text_to_image", "character_image"],
      ui: { ...EVERY_LANE.ui, poseLibrary: true },
    };
    const recipe = everyKnobRecipe();
    recipe.mode = "character_image";
    recipe.normalizedSettings = { ...recipe.normalizedSettings, characterId: character.id };
    recipe.rawAdapterSettings.referenceAssetId = reference.id;

    await render(
      baseContext({
        assets: [reference],
        characters: [character],
        imageModels: [poseModel, OTHER_MODEL],
        models: [poseModel, OTHER_MODEL, PID_CHECKPOINT],
        studioLaunch: launch(recipe),
      }),
    );

    expect(checkbox("Restore face").checked).toBe(true);
  });

  it("replays shared pose records through the visible strict-pose picker and exact submit path", async () => {
    seedOpenDisclosures({ model: OTHER_MODEL.id, controlMode: "depth", count: 7 });
    const createImageJob = vi.fn(async () => ({ id: "job-pose-replay" }));
    const recipe = everyKnobRecipe({
      poses: [
        {
          keypoints: [[0.1, 0.2, 0.9]],
          hands: [[[0.3, 0.4]]],
          face: [[0.5, 0.6]],
        },
        { keypoints: [[0.7, 0.8]] },
      ],
    });
    delete recipe.rawAdapterSettings.controlMode;
    const studioLaunch = launch(recipe, { id: "shared-launch" });

    await render(baseContext({ createImageJob, studioLaunch }));

    // The launch switches models, but the model-default reset must not erase its selection.
    expect(modelSelect().value).toBe(EVERY_LANE.id);
    expect(document.body.querySelector(".control-panel .advanced-section-toggle").getAttribute("aria-expanded")).toBe("true");
    expect(activeControlModeLabel()).toBe("Pose");
    expect(document.body.querySelector(".pose-category-name").textContent).toBe("Shared workflow");
    expect(document.body.querySelectorAll(".pose-thumb-placeholder")).toHaveLength(2);
    expect(document.body.textContent).toContain("2 poses selected");

    // Duplicates are independent records; removing the second leaves exactly the first payload.
    await click(document.body.querySelector('[aria-label="Deselect pose Shared workflow pose 2"]'));
    expect(document.body.textContent).toContain("1 pose selected");
    await click(buttonByText("Generate"));

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.count).toBe(1);
    expect(payload.advanced.faceRestore).toBe(true);
    expect(payload.advanced).not.toHaveProperty("controlMode");
    expect(payload.advanced.poses).toEqual([
      {
        id: "workflow:shared-launch:0",
        keypoints: [[0.1, 0.2, 0.9]],
        hands: [[[0.3, 0.4]]],
        face: [[0.5, 0.6]],
      },
    ]);

    // A subsequent user model change owns a safe reset; transient records do not leak globally.
    await act(async () => {
      modelSelect().value = OTHER_MODEL.id;
      modelSelect().dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => {});
    expect(document.body.querySelector(".pose-thumb-placeholder")).toBeNull();
  });

  it("clears a prior shared-pose launch when the next recipe omits advanced.poses", async () => {
    const first = everyKnobRecipe({ poses: [{ keypoints: [[1, 2]] }] });
    await render(baseContext({ studioLaunch: launch(first, { id: "with-poses" }) }));
    expect(document.body.querySelector(".pose-thumb-placeholder")).not.toBeNull();

    const second = everyKnobRecipe();
    const context = baseContext({ studioLaunch: launch(second, { id: "without-poses" }) });
    await render(context);
    expect(document.body.querySelector(".pose-thumb-placeholder")).toBeNull();
  });

  it("clears shared poses and face restoration at a same-model non-recipe launch boundary", async () => {
    const reference = {
      id: "asset_boundary_face",
      type: "image",
      projectId: "project_1",
      displayName: "Boundary face",
      status: {},
      file: { path: "assets/images/boundary.png", mimeType: "image/png" },
    };
    const character = {
      id: "char_boundary",
      name: "Boundary",
      looks: [],
      approvedReferences: [{ assetId: reference.id, asset: reference }],
    };
    const poseModel = {
      ...EVERY_LANE,
      capabilities: ["text_to_image", "character_image"],
      ui: { ...EVERY_LANE.ui, poseLibrary: true },
    };
    const recipe = everyKnobRecipe({
      poses: [{ keypoints: [[0.2, 0.8]] }],
      faceRestore: true,
    });
    recipe.mode = "character_image";
    recipe.normalizedSettings = {
      ...recipe.normalizedSettings,
      characterId: character.id,
    };
    recipe.rawAdapterSettings.referenceAssetId = reference.id;
    const createImageJob = vi.fn(async () => ({ id: "job-boundary" }));
    const common = {
      assets: [reference],
      characters: [character],
      createImageJob,
      imageModels: [poseModel],
      models: [poseModel, PID_CHECKPOINT],
    };

    await render(
      baseContext({
        ...common,
        studioLaunch: launch(recipe, { id: "recipe-boundary" }),
      }),
    );
    expect(document.body.querySelector(".pose-thumb-placeholder")).not.toBeNull();
    expect(checkbox("Restore face").checked).toBe(true);

    const characterLaunch = {
      id: "character-boundary",
      view: "Image",
      mode: "character_image",
      characterId: character.id,
      referenceAssetId: reference.id,
    };
    await render(baseContext({ ...common, studioLaunch: characterLaunch }));

    expect(document.body.querySelector(".pose-thumb-placeholder")).toBeNull();
    expect(document.body.textContent).not.toContain("pose selected");
    expect(checkbox("Restore face").checked).toBe(false);

    // Re-rendering the same launch with a new context object is not a new boundary. A choice the
    // user makes inside that launch must persist until a genuinely new launch id arrives.
    await click(checkbox("Restore face"));
    expect(checkbox("Restore face").checked).toBe(true);
    await render(baseContext({ ...common, studioLaunch: { ...characterLaunch } }));
    expect(checkbox("Restore face").checked).toBe(true);

    await click(buttonByText("Generate"));
    const payload = createImageJob.mock.calls[0][0];
    expect(payload.advanced).not.toHaveProperty("poses");
    expect(payload.advanced).not.toHaveProperty("faceRestore");
  });

  it("consumes no reset skip for a same-model recipe before a later manual model switch", async () => {
    seedOpenDisclosures({ model: EVERY_LANE.id, controlMode: "pose" });
    const recipe = everyKnobRecipe({ poses: [{ keypoints: [[0.4, 0.6]] }] });
    await render(baseContext({ studioLaunch: launch(recipe, { id: "same-model-poses" }) }));

    expect(document.body.querySelector(".pose-thumb-placeholder")).not.toBeNull();
    expect(document.body.querySelector(".control-panel .advanced-section-toggle").getAttribute("aria-expanded")).toBe("true");

    await act(async () => {
      modelSelect().value = OTHER_MODEL.id;
      modelSelect().dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => {});

    expect(document.body.querySelector(".pose-thumb-placeholder")).toBeNull();
    expect(document.body.textContent).not.toContain("pose selected");
    expect(document.body.querySelector(".control-panel .advanced-section-toggle").getAttribute("aria-expanded")).toBe("false");
  });

  // Multi-phase denoise. A 3-phase schedule replaying as a single pass is a DIFFERENT image, not a
  // degraded one, so the schedule has to survive — including the `loras[].index` -> resolved-id
  // remap, which is the only genuinely new code on this path.
  it("replays a multi-phase schedule, remapping each phase's LoRA index to its id", async () => {
    const recipe = everyKnobRecipe();
    recipe.loras = [{ id: STYLE_LORA.id, weight: 0.8 }, { id: ACCEL_LORA.id }];
    recipe.rawAdapterSettings.phases = [
      { steps: 4, guidance: 3.5, loras: [] },
      { steps: 6, guidance: 0, loras: [{ index: 1, weight: 0.9 }] },
      { steps: 2, guidance: 1.5, loras: [{ index: 0 }] },
    ];
    await render(baseContext({ studioLaunch: launch(recipe) }));

    // The editor is on screen with all three phases — a single-pass replay would show none.
    expect(phaseTitles()).toEqual(["Phase 1", "Phase 2", "Phase 3"]);
    expect(multiPhaseToggle().checked).toBe(true);

    // …and each phase's LoRA reference now names the resolved LoRA by its stable id, which is the
    // shape the editor mutates and `serializePhases` turns back into an index at submit.
    expect(persistedStudioSettings().multiPhasePhases).toEqual([
      { steps: 4, guidance: 3.5, loras: [] },
      { steps: 6, guidance: 0, loras: [{ id: ACCEL_LORA.id, weight: 0.9 }] },
      { steps: 2, guidance: 1.5, loras: [{ id: STYLE_LORA.id, weight: null }] },
    ]);
  });

  it("clears a stale schedule when the replayed recipe has no phases", async () => {
    seedOpenDisclosures({
      multiPhaseEnabled: true,
      multiPhasePhases: [{ steps: 8, guidance: 2, loras: [] }],
    });
    await render(baseContext({ studioLaunch: launch(everyKnobRecipe()) }));

    expect(phaseTitles()).toEqual([]);
    expect(multiPhaseToggle().checked).toBe(false);
  });
});
