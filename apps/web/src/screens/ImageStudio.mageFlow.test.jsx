import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, setInput, setSelect, unmountRoot } from "../testUtils/dom.js";

// Studio UI coverage for the Mage-Flow native-resolution image family (sc-14058, epic 14034 P6).
// Mage rides the generic manifest/tier/catalog machinery, so these tests assert the family-specific
// behaviors the story owns: it appears in BOTH the generation and edit pickers, its 4:1 aspect presets
// and ÷16 native-resolution stride are enforced in the Studio (as a surfaced ERROR, not a silent
// requirement), its per-variant step/CFG defaults flow through, its edit mode exposes source +
// instruction + multi-reference, and its quant tier is never silently switched.

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, apiFetch: vi.fn(async () => ({})) };
});

import { AppContext } from "../context/AppContext.js";
import { ImageStudio } from "./ImageStudio.jsx";

// The full native-resolution aspect ladder the manifest advertises for Mage-Flow (every entry ÷16,
// 512–2048 per side, incl. the 4:1 preset 2048x512 and its 1:4 inverse 512x2048).
const MAGE_RESOLUTIONS = [
  "512x512", "768x768", "1024x1024", "1536x1536", "2048x2048",
  "1280x720", "1536x1024", "2048x1024", "2048x512",
  "720x1280", "1024x1536", "1024x2048", "512x2048",
];

const MAGE_GUIDE = { title: "Mage-Flow Prompt Guide", path: "/prompt-guides/mage-flow.md" };

// Mage-Flow RL (gen). Deliberately declares NO memory floor so the resolution fit-gate never trims a
// bucket in the jsdom (unknown-memory) harness — every advertised aspect is offered.
const MAGE_RL = {
  id: "mage_flow",
  name: "Mage-Flow RL",
  type: "image",
  family: "mage-flow",
  capabilities: ["text_to_image"],
  defaults: { resolution: "1024x1024", steps: 20, guidanceScale: 5 },
  limits: { resolutions: MAGE_RESOLUTIONS, requiresDimensionsMultipleOf: 16 },
  loraCompatibility: {},
  ui: { promptGuide: MAGE_GUIDE },
};

const MAGE_TURBO = {
  ...MAGE_RL,
  id: "mage_flow_turbo",
  name: "Mage-Flow Turbo",
  defaults: { resolution: "1024x1024", steps: 4, guidanceScale: 1 },
};

const MAGE_EDIT = {
  ...MAGE_RL,
  id: "mage_flow_edit",
  name: "Mage-Flow Edit",
  capabilities: ["edit_image"],
  defaults: { resolution: "1024x1024", steps: 30, guidanceScale: 5 },
  ui: { sourceWithMultiReference: true, promptHint: "Describe the edit.", promptGuide: MAGE_GUIDE },
};

// A plain (non-native-resolution) model: no stride declared, so its free override must stay on the
// global 256–4096 envelope with NO stride — the discriminating control case.
const PLAIN = {
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

const MAC_CAPS = {
  macGatingActive: true,
  platform: "darwin",
  notAvailableLabel: "Not available on Mac (MLX only)",
  features: {},
  training: { supportedKernels: [], lokrOnWanSupported: false },
};

function baseContext(overrides = {}) {
  return {
    token: "test-token",
    activeProject: { id: "project_1", name: "My Project" },
    assets: [],
    characters: [],
    createImageJob: vi.fn(async () => ({ id: "job-1" })),
    createPreset: vi.fn(async (payload) => ({ id: payload.id })),
    refinePrompt: vi.fn(),
    deleteAsset: vi.fn(),
    purgeAsset: vi.fn(),
    gpuOptions: [],
    imageModels: [MAGE_RL],
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

// Label-scoped input/select lookup (the label text prefix identifies the field).
const field = (container, labelText) => {
  const label = [...container.querySelectorAll("label")].find((node) =>
    node.textContent.trim().startsWith(labelText),
  );
  return label?.querySelector("input, select, textarea");
};
// Exact-label lookup (disambiguates "Guidance" from "Guidance method").
const exactField = (container, labelText) => {
  const label = [...container.querySelectorAll("label")].find(
    (node) => node.querySelector("input, select, textarea") && node.firstChild?.textContent?.trim() === labelText,
  );
  return label?.querySelector("input, select, textarea");
};
const optionValues = (select) => [...select.options].map((option) => option.value);
const optionLabels = (select) => [...select.options].map((option) => option.textContent);
const generateButton = () =>
  [...document.body.querySelectorAll("button")].find((b) => b.textContent === "Generate");
const modeTab = (label) =>
  [...document.body.querySelectorAll(".mode-tabs button")].find((b) => b.textContent === label);
const headButtons = () => [...document.body.querySelectorAll(".asset-picker-head button")];

// Set a controlled <textarea> value through the native setter (setInput only handles <input>).
function setTextarea(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

describe("ImageStudio — Mage-Flow family (sc-14058)", () => {
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
  const openAdvanced = () => click(document.body.querySelector(".advanced-section-toggle"));

  it("lists the Mage-Flow family in BOTH the generation and the edit model pickers", async () => {
    await render(baseContext({ imageModels: [MAGE_RL, MAGE_TURBO, MAGE_EDIT] }));
    // Generation (Text mode): the gen variants are selectable, the edit variant is NOT.
    let picker = field(container, "Model");
    expect(picker).toBeTruthy();
    expect(optionLabels(picker)).toEqual(expect.arrayContaining(["Mage-Flow RL", "Mage-Flow Turbo"]));
    expect(optionLabels(picker)).not.toContain("Mage-Flow Edit");

    // Edit mode: the edit variant is selectable, the gen variants are NOT.
    await click(modeTab("Edit"));
    await act(async () => {});
    picker = field(container, "Model");
    expect(optionLabels(picker)).toContain("Mage-Flow Edit");
    expect(optionLabels(picker)).not.toContain("Mage-Flow RL");
  });

  it("offers the 4:1 aspect preset and its inverse in the resolution menu", async () => {
    await render(baseContext({ imageModels: [MAGE_RL] }));
    const aspect = field(container, "Aspect");
    expect(aspect).toBeTruthy();
    expect(optionValues(aspect)).toEqual(expect.arrayContaining(["2048x512", "512x2048", "2048x2048"]));
    // Default seeds to the native square.
    expect(aspect.value).toBe("1024x1024");
  });

  it("stamps the ÷16 native-resolution stride and 512–2048 range on the free Width/Height overrides", async () => {
    await render(baseContext({ imageModels: [MAGE_RL] }));
    await openAdvanced();
    await act(async () => {});
    const widthInput = field(container, "Width override");
    const heightInput = field(container, "Height override");
    expect(widthInput.getAttribute("step")).toBe("16");
    expect(widthInput.getAttribute("min")).toBe("512");
    expect(widthInput.getAttribute("max")).toBe("2048");
    expect(heightInput.getAttribute("step")).toBe("16");
    expect(heightInput.getAttribute("min")).toBe("512");
    expect(heightInput.getAttribute("max")).toBe("2048");
  });

  it("surfaces an off-stride override as an ERROR that blocks Generate, and clears on a ÷16 value", async () => {
    await render(baseContext({ imageModels: [MAGE_RL] }));
    // A valid prompt so the dimension override is the ONLY thing that can block Generate.
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a red boat"));
    await openAdvanced();
    await act(async () => {});

    // 1000 is within 512–2048 but not a multiple of 16 → surfaced error + Generate disabled.
    await act(async () => setInput(field(container, "Width override"), "1000"));
    expect(container.textContent).toContain("must be a multiple of 16");
    expect(generateButton().disabled).toBe(true);

    // 1536 is ÷16 and in range → the error clears and Generate is enabled again.
    await act(async () => setInput(field(container, "Width override"), "1536"));
    expect(container.textContent).not.toContain("must be a multiple of 16");
    expect(generateButton().disabled).toBe(false);
  });

  it("rejects a below-native override with the model's actual range in the message (not 256–4096)", async () => {
    await render(baseContext({ imageModels: [MAGE_RL] }));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a red boat"));
    await openAdvanced();
    await act(async () => {});
    await act(async () => setInput(field(container, "Width override"), "256")); // ÷16 clean, below 512
    expect(container.textContent).toContain("must be between 512 and 2048");
    expect(container.textContent).not.toContain("between 256 and 4096");
    expect(generateButton().disabled).toBe(true);
  });

  it("does NOT impose a stride on a non-native model — its free override stays 256–4096 with no step", async () => {
    await render(baseContext({ imageModels: [PLAIN] }));
    await act(async () => setTextarea(container.querySelector('textarea[aria-label="Prompt"]'), "a red boat"));
    await openAdvanced();
    await act(async () => {});
    const widthInput = field(container, "Width override");
    expect(widthInput.getAttribute("step")).toBe("1");
    expect(widthInput.getAttribute("min")).toBe("256");
    expect(widthInput.getAttribute("max")).toBe("4096");
    // 1000 (not ÷16) is perfectly valid for a plain model — no stride error, Generate stays enabled.
    await act(async () => setInput(widthInput, "1000"));
    expect(container.textContent).not.toContain("must be a multiple of 16");
    expect(generateButton().disabled).toBe(false);
  });

  it("wires Turbo's 4-step / CFG-off defaults and seeds the native default resolution", async () => {
    await render(baseContext({ imageModels: [MAGE_TURBO] }));
    await openAdvanced();
    await act(async () => {});
    expect(field(container, "Aspect").value).toBe("1024x1024");
    expect(field(container, "Steps").getAttribute("placeholder")).toBe("4");
    expect(exactField(container, "Guidance").getAttribute("placeholder")).toBe("1");
  });

  it("wires RL's 20-step / CFG-5 defaults", async () => {
    await render(baseContext({ imageModels: [MAGE_RL] }));
    await openAdvanced();
    await act(async () => {});
    expect(field(container, "Steps").getAttribute("placeholder")).toBe("20");
    expect(exactField(container, "Guidance").getAttribute("placeholder")).toBe("5");
  });

  it("exposes edit-mode source + instruction + optional multi-reference for a Mage edit model", async () => {
    await render(baseContext({ imageModels: [MAGE_EDIT] }));
    await click(modeTab("Edit"));
    await act(async () => {});
    // The instruction field is the shared prompt textarea.
    expect(container.querySelector('textarea[aria-label="Prompt"]')).toBeTruthy();
    // A single required Source image picker...
    expect(headButtons().some((b) => b.textContent === "Select image")).toBe(true);
    // ...PLUS an optional ordered multi-reference set (ui.sourceWithMultiReference), distinct from the
    // multiReference models that REPLACE the single source.
    expect(container.textContent).toContain("Additional references");
  });

  it("shows every quant tier for a multi-tier Mage model and never silently switches the chosen tier", async () => {
    const mageMatrix = {
      ...MAGE_RL,
      hasVariantMatrix: true,
      variants: ["q4", "q8", "bf16"].map((tier) => ({
        variant: tier,
        default: tier === "q4",
        installState: "installed",
      })),
    };
    await render(baseContext({ imageModels: [mageMatrix], macCapabilities: MAC_CAPS }));
    await openAdvanced();
    await act(async () => {});
    const picker = [...container.querySelectorAll("label.quant-tier-picker select")][0];
    expect(picker).toBeTruthy();
    expect(optionValues(picker)).toEqual(["q4", "q8", "bf16"]);

    // The user picks bf16 — a creative choice that must STICK across re-render/effects (no auto-fallback).
    await act(async () => setSelect(picker, "bf16"));
    await act(async () => {});
    const after = [...container.querySelectorAll("label.quant-tier-picker select")][0];
    expect(after.value).toBe("bf16");
  });
});
