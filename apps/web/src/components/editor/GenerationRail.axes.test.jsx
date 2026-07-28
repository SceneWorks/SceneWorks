import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../generationStudio.jsx", () => ({
  LoraPickerSection: () => null,
}));

import { GenerationRail } from "./GenerationRail.jsx";

function generationState(capabilities) {
  return {
    studio: {
      availablePresets: [],
      selectedPresetId: null,
      setSelectedPresetId: vi.fn(),
    },
    selectedModel: { id: "video-model" },
    videoModels: [{ id: "video-model" }],
    model: "video-model",
    setModel: vi.fn(),
    qualityChoices: [["balanced", "Balanced"]],
    quality: "balanced",
    setQuality: vi.fn(),
    resolution: "1280x720",
    resolutionOptions: ["1280x720"],
    setResolution: vi.fn(),
    fps: 24,
    fpsOptions: [24],
    setFps: vi.fn(),
    duration: 6,
    durationOptions: [6],
    setDuration: vi.fn(),
    motion: "Balanced",
    motionPct: 50,
    setMotion: vi.fn(),
    prompt: "A fox",
    setPrompt: vi.fn(),
    seed: "",
    setSeed: vi.fn(),
    randomizeSeed: vi.fn(),
    negativePrompt: "blurry",
    setNegativePrompt: vi.fn(),
    advancedOpen: true,
    setAdvancedOpen: vi.fn(),
    showSamplerPicker: false,
    showSchedulerPicker: false,
    stepsOverride: "",
    setStepsOverride: vi.fn(),
    guidanceOverride: "6.5",
    setGuidanceOverride: vi.fn(),
    lightningActive: false,
    showTierPicker: false,
    showLightning: false,
    savePreset: vi.fn(),
    ...capabilities,
  };
}

describe("GenerationRail catalog axes (sc-15441)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  function render(gen) {
    act(() => {
      root.render(<GenerationRail gen={gen} onGenerate={vi.fn()} />);
    });
  }

  it("shows Guidance and Negative prompt when the video axes are supported", () => {
    render(generationState({ supportsGuidance: true, supportsNegativePrompt: true }));
    expect(container.textContent).toContain("Guidance");
    expect(container.textContent).toContain("Negative prompt");
  });

  it("hides Guidance and Negative prompt when the video axes are absent", () => {
    render(generationState({ supportsGuidance: false, supportsNegativePrompt: false }));
    expect(container.textContent).not.toContain("Guidance");
    expect(container.textContent).not.toContain("Negative prompt");
  });
});
