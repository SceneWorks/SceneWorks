import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { IrisCurriculaOptions, IrisRunMetrics } from "./IrisCurriculaOptions.jsx";
import { irisDraftFromDefaults } from "../../training/irisOptions.js";

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

const control = (label) =>
  [...container.querySelectorAll("label")]
    .find((node) => node.textContent.startsWith(label))
    ?.querySelector("select, input") ?? null;

describe("IrisCurriculaOptions (sc-25689)", () => {
  it("applies an official stage, renders only in-play controls and shows the plan check", async () => {
    const onApplyPreset = vi.fn();
    const updateConfigDraft = vi.fn();
    const presets = [
      { id: "stage1_256", parentStage: null, config: {} },
      { id: "stage2_512", parentStage: "stage1_256", config: {} },
    ];
    await act(async () =>
      root.render(
        <IrisCurriculaOptions
          configDraft={{ iris: { ...irisDraftFromDefaults({}), "repa.weight": "0.5" }, batchSize: "2", gradientAccumulation: "2" }}
          configValidity={{}}
          onApplyPreset={onApplyPreset}
          planCheck={{ teacher: { installed: false }, refusal: { kind: "invalid_mesh", message: "The mesh does not tile the world" } }}
          presets={presets}
          updateConfigDraft={updateConfigDraft}
        />,
      ),
    );
    const preset = control("Official stage preset");
    await act(async () => {
      preset.value = "stage2_512";
      preset.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onApplyPreset).toHaveBeenCalledWith(presets[1]);
    expect(control("REPA variant")).not.toBeNull();
    expect(control("Sharding")).toBeNull();
    expect(container.textContent).toContain("Iris REPA Teacher");
    expect(container.querySelector('[role="alert"]').textContent).toContain("does not tile");
    expect(container.querySelector('[role="status"]').textContent).toContain("Global batch 4");
    const ranks = control("Ranks");
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(ranks, "2");
      ranks.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(updateConfigDraft).toHaveBeenCalledWith("iris", expect.objectContaining({ "dist.world.world_size": "2" }));
  });

  it("tabulates a run's holdout validation", async () => {
    await act(async () => root.render(<IrisRunMetrics metrics={{ validation: [{ step: 50, "val/loss": 0.31, "val/loss_t0": 0.2 }] }} />));
    expect(container.querySelector("table").textContent).toContain("0.3100");
  });
});
