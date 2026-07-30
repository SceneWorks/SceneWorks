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
    imageModels: [EVERY_LANE, OTHER_MODEL],
    models: [EVERY_LANE, OTHER_MODEL, PID_CHECKPOINT],
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

  it("sets prompt upsampling, the PiD decoder and its output tier", async () => {
    await render(baseContext({ studioLaunch: launch(everyKnobRecipe()) }));

    expect(checkbox("Enhance prompt").checked).toBe(true);
    expect(checkbox("PiD decoder").checked).toBe(true);
    expect(document.body.querySelector(".pid-target-select select").value).toBe("2k");
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

    expect(checkbox("Enhance prompt").checked).toBe(false);
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
