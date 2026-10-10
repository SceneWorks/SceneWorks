import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getOptionsMock, previewOptionsMock } = vi.hoisted(() => ({
  getOptionsMock: vi.fn(),
  previewOptionsMock: vi.fn(),
}));

vi.mock("../../api/films.js", () => ({
  getFilmRenderOptions: getOptionsMock,
  previewFilmRenderOptions: previewOptionsMock,
}));

import { FilmRenderOptions } from "./FilmRenderOptions.jsx";

const resolved = (overrides = {}) => ({
  selectedRegime: "recommended_turbo",
  recommendedTurbo: { available: true, adapterIds: ["minimax_h3_turbo_4step_v01"], effectiveSteps: 4 },
  quality: { adapterIds: [], effectiveSteps: 30 },
  effective: { adapterIds: ["minimax_h3_turbo_4step_v01"], effectiveSteps: 4 },
  ...overrides,
});

function makeDraft(overrides = {}) {
  return {
    id: "film_1",
    revision: 3,
    renderRegime: "recommended_turbo",
    productionPlan: {
      model: { id: "minimax_h3", tier: "q4", resolution: "576x320", loras: ["minimax_h3_turbo_4step_v01"], advanced: { steps: 4 } },
      shots: [{ id: "SH010", resolution: "576x320", conditioning: { mode: "text_to_video", referenceRoles: [] } }],
    },
    referencePack: { references: [] },
    ...overrides,
  };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  getOptionsMock.mockReset();
  previewOptionsMock.mockReset();
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
  vi.useRealTimers();
});

async function render(props = {}) {
  const defaults = {
    disabled: false,
    draft: makeDraft(),
    onChange: vi.fn(),
    projectId: "project_1",
    token: "token",
  };
  const current = { ...defaults, ...props };
  root = createRoot(container);
  await act(async () => { root.render(<FilmRenderOptions {...current} />); await Promise.resolve(); });
  return {
    props: current,
    rerender: async (next) => {
      Object.assign(current, next);
      await act(async () => { root.render(<FilmRenderOptions {...current} />); await Promise.resolve(); });
    },
  };
}

describe("FilmRenderOptions", () => {
  it("loads persisted options and applies an explicit quality choice", async () => {
    getOptionsMock.mockResolvedValue(resolved());
    const view = await render();

    expect(getOptionsMock).toHaveBeenCalledWith("project_1", "film_1", "token", expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(container.textContent).toContain("minimax_h3_turbo_4step_v01 · 4 steps");
    expect(container.textContent).toContain("base model · 30 steps");
    const quality = [...container.querySelectorAll('input[type="radio"]')][1];
    await act(async () => quality.click());
    const mutator = view.props.onChange.mock.calls[0][0];
    const next = makeDraft();
    mutator(next);
    expect(next.renderRegime).toBe("quality");
  });

  it("previews unsaved request changes and ignores an obsolete response", async () => {
    vi.useFakeTimers();
    getOptionsMock.mockResolvedValue(resolved());
    let resolveFirst;
    const first = new Promise((resolve) => { resolveFirst = resolve; });
    previewOptionsMock
      .mockReturnValueOnce(first)
      .mockResolvedValueOnce(resolved({ effective: { adapterIds: ["latest_adapter"], effectiveSteps: 6 } }));
    const view = await render();

    const firstDraft = makeDraft();
    firstDraft.productionPlan.model.resolution = "768x432";
    await view.rerender({ draft: firstDraft });
    await act(async () => { await vi.advanceTimersByTimeAsync(120); });
    expect(previewOptionsMock).toHaveBeenCalledTimes(1);

    const latestDraft = structuredClone(firstDraft);
    latestDraft.productionPlan.shots[0].conditioning.mode = "reference_to_video";
    latestDraft.referencePack.references = [{ role: "hero", kind: "image", approved: true, assetId: "asset_1" }];
    await view.rerender({ draft: latestDraft });
    await act(async () => { await vi.advanceTimersByTimeAsync(120); });
    expect(previewOptionsMock).toHaveBeenCalledTimes(2);
    expect(previewOptionsMock.mock.calls[1][2]).toBe(latestDraft);
    expect(container.textContent).toContain("latest_adapter · 6 steps");

    await act(async () => { resolveFirst(resolved({ effective: { adapterIds: ["obsolete_adapter"], effectiveSteps: 8 } })); await Promise.resolve(); });
    expect(container.textContent).not.toContain("obsolete_adapter");
  });

  it("explains and disables an unavailable recommended turbo choice", async () => {
    getOptionsMock.mockResolvedValue(resolved({
      selectedRegime: "quality",
      recommendedTurbo: { available: false, adapterIds: [], effectiveSteps: null, unavailableReason: "incompatible_resolution" },
      effective: { adapterIds: [], effectiveSteps: 30 },
    }));
    await render({ draft: makeDraft({ renderRegime: "quality" }) });

    const radios = [...container.querySelectorAll('input[type="radio"]')];
    expect(radios[0].disabled).toBe(true);
    expect(radios[1].checked).toBe(true);
    expect(container.textContent).toContain("selected resolution is incompatible");
  });

  it("shows the server's custom selection for a legacy draft and preserves authored controls in preview", async () => {
    vi.useFakeTimers();
    const legacy = makeDraft({ renderRegime: undefined });
    legacy.productionPlan.model.loras = ["authored_adapter"];
    legacy.productionPlan.model.advanced.steps = 9;
    getOptionsMock.mockResolvedValue(resolved({
      selectedRegime: "custom",
      effective: { adapterIds: ["authored_adapter"], effectiveSteps: 9 },
    }));
    previewOptionsMock.mockResolvedValue(resolved({
      selectedRegime: "custom",
      effective: { adapterIds: ["authored_adapter"], effectiveSteps: 9 },
    }));
    const view = await render({ draft: legacy });

    expect([...container.querySelectorAll('input[type="radio"]')][2].checked).toBe(true);
    const edited = structuredClone(legacy);
    edited.productionPlan.model.resolution = "768x432";
    await view.rerender({ draft: edited });
    await act(async () => { await vi.advanceTimersByTimeAsync(120); });

    const previewed = previewOptionsMock.mock.calls[0][2];
    expect(previewed.renderRegime).toBeUndefined();
    expect(previewed.productionPlan.model.loras).toEqual(["authored_adapter"]);
    expect(previewed.productionPlan.model.advanced.steps).toBe(9);
  });
});
