import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  loadStudioSettings: vi.fn(() => ({})),
  writeStudioSettings: vi.fn(),
  studio: { selectedPreset: null },
}));

vi.mock("../../hooks/useStudioSettings.js", () => ({
  loadStudioSettings: mocks.loadStudioSettings,
  useStudioSettingsWriter: (...args) => mocks.writeStudioSettings(...args),
}));

vi.mock("../generationStudio.jsx", () => ({
  useGenerationStudio: () => ({
    selectedPreset: mocks.studio.selectedPreset,
    selectedPresetId: mocks.studio.selectedPreset?.id ?? null,
    selectedLoras: [],
    selectedLoraIds: [],
    loraWeights: {},
    showIncompatibleLoras: false,
    generalStackIds: [],
    effectiveLoraWeight: () => 1,
  }),
}));

import { useEditorGeneration } from "./useEditorGeneration.js";

const GUIDED = { id: "guided-video" };
const CFG_FREE = {
  id: "cfg-free-video",
  video: { supportsGuidance: false, supportsNegativePrompt: false },
};

describe("useEditorGeneration catalog axes (sc-15441)", () => {
  let container;
  let root;
  let latest;
  let context;

  function Harness({ videoModels }) {
    latest = useEditorGeneration({ context: { ...context, videoModels } });
    return null;
  }

  function render(videoModels) {
    act(() => {
      root.render(<Harness videoModels={videoModels} />);
    });
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    mocks.loadStudioSettings.mockReturnValue({});
    mocks.studio.selectedPreset = null;
    context = {
      activeProject: { id: "project-1" },
      preferencesHydrated: true,
      createPreset: vi.fn(async (payload) => payload),
    };
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  it("keeps both axes for a video model that declares no video block", async () => {
    render([GUIDED]);
    act(() => {
      latest.setNegativePrompt("blurry");
      latest.setGuidanceOverride("6.5");
    });

    expect(latest.supportsGuidance).toBe(true);
    expect(latest.supportsNegativePrompt).toBe(true);
    expect(latest.buildBasePayload().negativePrompt).toBe("blurry");
    expect(latest.buildBasePayload().advanced.guidanceScale).toBe(6.5);

    await act(async () => latest.savePreset("Guided"));
    const defaults = context.createPreset.mock.calls[0][0].defaults;
    expect(defaults.negativePrompt).toBe("blurry");
    expect(defaults.guidanceScale).toBe(6.5);
  });

  it("suppresses retained values after switching to a CFG-free model in payload, settings, and presets", async () => {
    render([GUIDED, CFG_FREE]);
    act(() => {
      latest.setNegativePrompt("blurry, low quality");
      latest.setGuidanceOverride("7.5");
    });
    act(() => latest.setModel(CFG_FREE.id));

    expect(latest.supportsGuidance).toBe(false);
    expect(latest.supportsNegativePrompt).toBe(false);
    expect(latest.buildBasePayload().negativePrompt).toBe("");
    expect(latest.buildBasePayload().advanced).not.toHaveProperty("guidanceScale");

    const settings = mocks.writeStudioSettings.mock.calls.at(-1)[2];
    expect(settings.negativePrompt).toBe("");
    expect(settings.guidanceScale).toBe("");

    await act(async () => latest.savePreset("CFG free"));
    const defaults = context.createPreset.mock.calls[0][0].defaults;
    expect(defaults).not.toHaveProperty("negativePrompt");
    expect(defaults).not.toHaveProperty("guidanceScale");
  });

  it("does not apply preset defaults for hidden axes", () => {
    mocks.studio.selectedPreset = {
      id: "stale-preset",
      defaults: { negativePrompt: "ghost", guidanceScale: 9 },
    };
    render([CFG_FREE]);

    expect(latest.negativePrompt).toBe("");
    expect(latest.guidanceOverride).toBe("");
    expect(latest.buildBasePayload().negativePrompt).toBe("");
    expect(latest.buildBasePayload().advanced).not.toHaveProperty("guidanceScale");
  });

  it("filters persisted stale values restored directly onto a CFG-free model", () => {
    mocks.loadStudioSettings.mockReturnValue({
      model: CFG_FREE.id,
      negativePrompt: "restored ghost",
      guidanceScale: "8",
    });
    render([GUIDED, CFG_FREE]);

    expect(latest.model).toBe(CFG_FREE.id);
    expect(latest.buildBasePayload().negativePrompt).toBe("");
    expect(latest.buildBasePayload().advanced).not.toHaveProperty("guidanceScale");
  });
});
