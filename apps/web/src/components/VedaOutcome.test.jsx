import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react";
import { describe, expect, it } from "vitest";
import { AssetVedaOutcome, VedaOutcome } from "./VedaOutcome.jsx";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

describe("persisted observed Veda results", () => {
  it("shows actual sparse plan and fraction separately from replay controls, then clears on later jobs", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    const raw = { veda: { enabled: true, generated: "90%", reference: "12", fullAttentionLayers: [0], fullAttentionSteps: [1] },
      vedaExecution: { effective: "mixed", kernel: "native-kernel", plan: "trained_T", planMatch: "nearest", planGrid: [2,8,9], executionGrid: [2,7,9], totalAttentionFraction: 0.4375 } };
    act(() => root.render(<AssetVedaOutcome asset={{ recipe: { rawAdapterSettings: raw } }} />));
    expect(container.textContent).toContain("Mixed sparse and dense");
    expect(container.textContent).toContain("native-kernel");
    expect(container.textContent).toContain("nearest; trained 2 × 8 × 9, executed 2 × 7 × 9");
    expect(container.textContent).toContain("43.75%");
    expect(container.textContent).toContain('"reference":"12"');
    act(() => root.render(<AssetVedaOutcome asset={{ recipe: { rawAdapterSettings: { veda: { enabled: true } } } }} />));
    expect(container.textContent).toBe("");
    act(() => root.unmount());
  });
  it.each(["missing predictor", "actual-device self-test forced failure"])("shows effective dense for %s", (reason) => {
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() => root.render(<VedaOutcome outcome={{ effective: "dense", totalAttentionFraction: 1, fallbackReason: reason }} requested={{ enabled: true }} />));
    expect(container.textContent).toContain("Observed attention: Dense");
    expect(container.textContent).toContain("100.00%");
    expect(container.textContent).toContain(reason);
    expect(container.textContent).toContain('"enabled":true');
    act(() => root.unmount());
  });
});
