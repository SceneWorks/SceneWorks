import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { loadStudioSettingsMock } = vi.hoisted(() => ({ loadStudioSettingsMock: vi.fn(() => ({})) }));
vi.mock("../../hooks/useStudioSettings.js", () => ({
  loadStudioSettings: loadStudioSettingsMock,
  useStudioSettingsWriter: () => {},
}));
// The preset/LoRA machinery is orthogonal to tier selection; stub it so the harness needs no catalog.
vi.mock("../generationStudio.jsx", () => ({
  useGenerationStudio: () => ({
    selectedPreset: null,
    selectedPresetId: null,
    selectedLoras: [],
    selectedLoraIds: [],
    loraWeights: {},
    showIncompatibleLoras: false,
    generalStackIds: [],
    effectiveLoraWeight: () => 1,
  }),
}));

import { useEditorGeneration } from "./useEditorGeneration.js";

// A download-matrix video model: `variants` carries every POSSIBLE tier, `installState` says which
// are actually on disk. `installed` lists the tiers to mark installed.
function matrixModel(possible, installed) {
  return {
    id: "vid_model",
    hasVariantMatrix: true,
    variants: possible.map((tier) => ({
      variant: tier,
      installState: installed.includes(tier) ? "installed" : "missing",
    })),
  };
}

// A single-tier model (no variant matrix): one possible tier, so the picker never renders.
function singleTierModel(tier) {
  return { id: "vid_model", mlxTiers: [tier] };
}

describe("useEditorGeneration — quant tier rides the payload (sc-12090, #1516)", () => {
  let container;
  let root;
  let latest;

  function Harness({ model }) {
    latest = useEditorGeneration({
      context: { activeProject: { id: "p1" }, videoModels: model ? [model] : [] },
    });
    return null;
  }

  function render(model) {
    act(() => {
      root.render(<Harness model={model} />);
    });
  }

  beforeEach(() => {
    loadStudioSettingsMock.mockReturnValue({});
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  it("sends the installed tier even though a single-tier model hides the picker", () => {
    render(singleTierModel("bf16"));

    // One possible tier ⇒ no picker. The old payload gated mlxQuantize on `showTierPicker`, so the
    // request carried no tier at all and the worker fell back to its own q8 default — against a tier
    // this user never downloaded. bf16 maps to 0, which must survive the falsy-value spread.
    expect(latest.showTierPicker).toBe(false);
    expect(latest.buildBasePayload().advanced.mlxQuantize).toBe(0);
  });

  it("never sends a tier that stopped being installed under the same model", () => {
    // Both tiers on disk; the user picks q8.
    render(matrixModel(["q4", "q8"], ["q4", "q8"]));
    act(() => latest.selectTier("q8"));
    expect(latest.buildBasePayload().advanced.mlxQuantize).toBe(8);

    // A catalog refresh now reports q8 gone (deleted to reclaim disk, or newly detected torn) with
    // the MODEL ID UNCHANGED — the transition the model-id-keyed clamp never observes. Continuing to
    // send 8 is exactly the #1516 symptom: the VRAM gate budgets a tier that is not on disk and
    // rejects the job. Both the payload and the picker's own state must follow the installed set.
    render(matrixModel(["q4", "q8"], ["q4"]));
    expect(latest.buildBasePayload().advanced.mlxQuantize).toBe(4);
    expect(latest.quantTier).toBe("q4");
  });

  it("honors an explicit pick of an installed tier", () => {
    render(matrixModel(["q4", "q8"], ["q4", "q8"]));

    act(() => latest.selectTier("q8"));
    expect(latest.buildBasePayload().advanced.mlxQuantize).toBe(8);
  });

  it("refuses to select an un-downloaded tier", () => {
    loadStudioSettingsMock.mockReturnValue({ quantTier: "q4" });
    render(matrixModel(["q4", "q8"], ["q4"]));

    act(() => latest.selectTier("q8"));
    expect(latest.quantTier).toBe("q4");
    expect(latest.buildBasePayload().advanced.mlxQuantize).toBe(4);
  });

  it("omits the tier entirely when nothing is installed", () => {
    render(matrixModel(["q4", "q8"], []));

    // No installed tier to name — the worker resolves one itself rather than being handed a
    // tier that is guaranteed to be absent.
    expect(latest.buildBasePayload().advanced.mlxQuantize).toBeUndefined();
  });
});
