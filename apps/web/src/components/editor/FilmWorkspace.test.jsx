import React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppContext } from "../../context/AppContext.js";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../../api.js", () => ({ apiFetch: apiFetchMock }));

import { FilmWorkspace } from "./FilmWorkspace.jsx";

function draft(overrides = {}) {
  return {
    schemaVersion: 1,
    id: "film_1",
    projectId: "project_1",
    revision: 1,
    title: "First film",
    originalScript: "",
    brief: "",
    planning: { provider: "prompt_refiner" },
    productionPlan: {
      schemaVersion: 2,
      id: "film_1",
      version: 1,
      title: "First film",
      synopsis: "",
      model: { id: "minimax_h3", tier: "q4", fps: 24, resolution: "576x320" },
      limits: { maxRunSeconds: 3600, maxShotSeconds: 2700, maxAttemptsPerShot: 1, maxMemoryGb: 96 },
      sound: {},
      shots: [{
        id: "SH010", beat: "Opening shot", framing: "wide", prompt: "",
        targetDurationSeconds: 5.1667, startState: "Opening state", endState: "Closing state",
        conditioning: { mode: "text_to_video", referenceRoles: [] }, continuityRoles: [],
      }],
    },
    referencePack: { schemaVersion: 1, id: "film_1-references", version: 1, description: "", references: [], sound: [] },
    reviewPlan: { schemaVersion: 1, questions: [] },
    createdAt: "2026-09-16T00:00:00Z",
    updatedAt: "2026-09-16T00:00:00Z",
    ...overrides,
  };
}

let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  apiFetchMock.mockReset();
});

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
});

async function renderWorkspace() {
  root = createRoot(container);
  await act(async () => {
    root.render(
      <AppContext.Provider value={{
        activeProject: { id: "project_1", name: "Project" }, token: "",
        refreshTimelines: vi.fn(), setSelectedTimelineId: vi.fn(),
      }}>
        <FilmWorkspace />
      </AppContext.Provider>,
    );
    await Promise.resolve();
  });
}

function changeValue(element, value) {
  const prototype = element.tagName === "TEXTAREA"
    ? window.HTMLTextAreaElement.prototype
    : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value").set.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("FilmWorkspace", () => {
  it("creates, edits, saves, and reopens a project film draft without JSON authoring", async () => {
    const created = draft();
    const saved = draft({ revision: 2, title: "Workshop delivery" });
    apiFetchMock
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce(created)
      .mockResolvedValueOnce(saved)
      .mockResolvedValueOnce([saved]);

    await renderWorkspace();
    const newButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "New film draft");
    await act(async () => { newButton.click(); await Promise.resolve(); });

    const title = container.querySelector('input[value="First film"]');
    await act(async () => {
      changeValue(title, "Workshop delivery");
      const prompt = container.querySelector('textarea[aria-label="Shot prompt"]');
      changeValue(prompt, "A courier enters a workshop carrying a red parcel.");
    });
    const save = [...container.querySelectorAll("button")].find((button) => button.textContent === "Save draft");
    await act(async () => { save.click(); await Promise.resolve(); });
    expect(apiFetchMock).toHaveBeenCalledWith(
      "/api/v1/projects/project_1/films/film_1",
      "",
      expect.objectContaining({ method: "PUT" }),
    );
    const body = JSON.parse(apiFetchMock.mock.calls[2][2].body);
    expect(body.title).toBe("Workshop delivery");
    expect(body.productionPlan.shots[0].prompt).toContain("red parcel");

    act(() => root.unmount());
    root = null;
    await renderWorkspace();
    expect(container.querySelector('select[aria-label="Film draft"]').value).toBe("film_1");
    expect(container.querySelector('input[value="Workshop delivery"]')).not.toBeNull();
  });
});
