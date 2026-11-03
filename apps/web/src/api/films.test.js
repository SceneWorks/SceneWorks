import { beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({ apiFetch: apiFetchMock }));

import { filmRequest, getFilmRun, listFilmRuns, startFilmRun, getFilmRenderOptions, previewFilmRenderOptions, startFilmPlanning } from "./films.js";

describe("film render options API", () => {
  beforeEach(() => apiFetchMock.mockReset());

  it("encodes film and run identifiers and forwards cancellation", () => {
    const signal = new AbortController().signal;
    filmRequest("p/1", "f/1", "t", { signal }, "/references");
    listFilmRuns("p/1", "t", { signal });
    getFilmRun("p/1", "r/1", "t", { signal });
    startFilmRun("p/1", "r/1", "start", "t", { signal });
    expect(apiFetchMock.mock.calls).toEqual([
      ["/api/v1/projects/p%2F1/films/f%2F1/references", "t", { signal }],
      ["/api/v1/projects/p%2F1/film-runs", "t", { signal }],
      ["/api/v1/projects/p%2F1/film-runs/r%2F1", "t", { signal }],
      ["/api/v1/projects/p%2F1/film-runs/r%2F1/start", "t", { signal, method: "POST" }],
    ]);
  });

  it("starts planning with the author's repair-round limit", () => {
    startFilmPlanning("project_1", "film_1", 4, 75, "token");

    expect(apiFetchMock).toHaveBeenCalledWith("/api/v1/projects/project_1/films/film_1/planning", "token", {
      method: "POST",
      body: JSON.stringify({ maxRepairRounds: 4, llmTimeoutSeconds: 75 }),
    });
  });

  it("loads persisted options and previews the complete unsaved resolver input", () => {
    const signal = new AbortController().signal;
    const draft = {
      revision: 7,
      renderRegime: "quality",
      productionPlan: { model: { id: "minimax_h3" }, shots: [] },
      referencePack: { references: [] },
    };

    getFilmRenderOptions("project / one", "film / one", "token", { signal });
    previewFilmRenderOptions("project / one", "film / one", draft, "token", { signal });

    expect(apiFetchMock).toHaveBeenNthCalledWith(1, "/api/v1/projects/project%20%2F%20one/films/film%20%2F%20one/render-options", "token", { signal });
    expect(apiFetchMock).toHaveBeenNthCalledWith(2, "/api/v1/projects/project%20%2F%20one/films/film%20%2F%20one/render-options", "token", {
      signal,
      method: "POST",
      body: JSON.stringify({
        draftRevision: 7,
        productionPlan: draft.productionPlan,
        referencePack: draft.referencePack,
        renderRegime: "quality",
      }),
    });
  });

  it("previews a legacy draft as explicit custom without changing authored controls", () => {
    const draft = {
      revision: 4,
      productionPlan: {
        model: { id: "minimax_h3", loras: ["authored_adapter"], advanced: { steps: 9 } },
        shots: [],
      },
      referencePack: { references: [] },
    };

    previewFilmRenderOptions("project_1", "film_1", draft, "token");

    const body = JSON.parse(apiFetchMock.mock.calls[0][2].body);
    expect(body.renderRegime).toBe("custom");
    expect(body.productionPlan.model.loras).toEqual(["authored_adapter"]);
    expect(body.productionPlan.model.advanced.steps).toBe(9);
    expect(draft).not.toHaveProperty("renderRegime");
  });
});
